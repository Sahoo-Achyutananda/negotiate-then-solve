"""The model's lookups (agents.tool_agent), with a fake model: what the code makes of a choice, what it sends
back when the choice is wrong, and the limit on tries. No API calls."""

from datetime import date, datetime

import pytest

from agents.tool_agent import MAX_TRIES, Choice, GrammarChoice, ToolAgent
from language.local import grammar_schema
from web.app import World

WEEK = 13


class FakeModel:
    """Answers with the given choices in turn, and keeps the prompts it was sent."""

    def __init__(self, *choices: Choice) -> None:
        self.choices = list(choices)
        self.prompts: list[str] = []

    def generate(self, system: str, prompt: str, schema: type[Choice]) -> Choice:
        self.prompts.append(prompt)
        return self.choices.pop(0)


@pytest.fixture(scope="module")
def world():
    return World(parser_mode="offline", seed_history=False, deadline=60)


def person(world, name):
    return next(f.id for f in world.instance.faculty if f.name == name)


def agent(world, *choices, sender="Dr. Menon"):
    who = person(world, sender) if sender.startswith("Dr.") else sender
    return ToolAgent(world.instance, FakeModel(*choices), who, WEEK, today=date(2026, 10, 21))


def test_names_become_ids_and_me_is_the_sender(world):
    a = agent(world, Choice(tool="free_hours", teachers=["me", "Dr. Iyer"], days=["Fri"], week=WEEK))
    call = a.read("when are iyer and I both around on friday?")
    assert call.tool == "free_hours" and call.week == WEEK and call.confident
    assert call.args == {"teachers": [person(world, "Dr. Menon"), person(world, "Dr. Iyer")], "days": ["Fri"]}


def test_the_prompt_has_the_names_and_the_question_but_no_timetable(world):
    a = agent(world, Choice(tool="none", week=WEEK))
    a.read("when's Iyer around this week?")
    prompt = a.client.prompts[0]
    assert "Dr. Iyer" in prompt and "Lab 2 (lab, 60 seats, gpu, routers)" in prompt and "week 13" in prompt
    assert "<question>when's Iyer around this week?</question>" in prompt
    assert "C-0" not in prompt and "Room 101 (lecture" in prompt  # names only: no IDs, no classes


def test_none_is_not_a_timetable_question(world):
    assert agent(world, Choice(tool="none", week=WEEK)).read("can I have Friday off?") is None


def test_a_wrong_name_is_sent_back_and_corrected(world):
    a = agent(world, Choice(tool="classes_of", teachers=["Dr. Iyar"], week=WEEK),
              Choice(tool="classes_of", teachers=["Dr. Iyer"], week=WEEK))
    call = a.read("what does iyar teach")
    assert call.args == {"teacher": person(world, "Dr. Iyer")}
    assert 'no teacher is called "Dr. Iyar"' in a.client.prompts[1]
    assert [t["problem"] is None for t in a.trace] == [False, True]


def test_after_the_last_try_it_says_so_and_guesses_nothing(world):
    wrong = Choice(tool="free_hours", days=["Sat"], teachers=["me"], week=WEEK)
    a = agent(world, *[wrong] * MAX_TRIES)
    call = a.read("am I free on saturday")
    assert not call.confident and "couldn't work out" in call.note
    assert len(a.client.prompts) == MAX_TRIES and all(t["problem"] for t in a.trace)


@pytest.mark.parametrize("choice, problem", [
    (Choice(tool="free_rooms", kind="lab", equipment=["laser"], week=WEEK), "equipment ['laser']"),
    (Choice(tool="free_hours", teachers=["me"], hours=[4], week=WEEK), "not lunch"),
    (Choice(tool="classes_of", teachers=["Dr. Iyer", "Dr. Rao"], week=WEEK), "at most one teacher"),
])
def test_what_is_checked(world, choice, problem):
    a = agent(world, *[choice] * MAX_TRIES)
    a.read("question")
    assert problem in a.trace[0]["problem"]


def test_a_week_the_semester_does_not_have_is_said_not_retried(world):
    """ "Free slots in week 18?": the model read it right; sending it back ended in the regex answering this week."""
    a = agent(world, Choice(tool="free_hours", teachers=["me"], week=18))
    call = a.read("my free slots in week 18?")
    weeks = world.instance.calendar.weeks
    assert not call.confident and call.note == f"The semester has {weeks} weeks (1–{weeks}); week 18 isn't in it."
    assert len(a.client.prompts) == 1


def test_nobody_named_is_the_asker_and_a_room_is_a_lecture_room(world):
    """The checker's defaults (v2's misses T-020, T-137, T-042): the model leaving these out is not a mistake."""
    call = agent(world, Choice(tool="free_hours", days=["Wed"], part="morning", week=WEEK)).read("any free time on wednesday morning?")
    assert call.args["teachers"] == [person(world, "Dr. Menon")]
    call = agent(world, Choice(tool="free_rooms", days=["Fri"], hours=[7], week=WEEK)).read("which rooms are free on Fri at 4 pm")
    assert call.args["kind"] == "lecture"


def test_something_not_in_the_department_is_not_guessed(world):
    call = agent(world, Choice(tool="free_hours", unknown=["Dr. Smith"], week=WEEK)).read("is smith free?")
    assert not call.confident and "Dr. Smith" in call.note
    call = agent(world, Choice(tool="free_rooms", kind="lecture", unknown=["auditorium"], week=WEEK)).read("auditorium free?")
    assert not call.confident and "auditorium" in call.note


def test_a_class_representatives_me_is_their_section(world):
    call = agent(world, Choice(tool="free_hours", teachers=["me", "Dr. Iyer"], week=WEEK), sender="ST-G-04").read(
        "when are we and iyer both free")
    assert call.args == {"teachers": [person(world, "Dr. Iyer")], "sections": ["G-04"]}


@pytest.mark.parametrize("choice, tool, args", [
    (Choice(tool="who_is_free", of="teachers", days=["Mon"], hours=[1], week=WEEK),
     "who_is_free", {"days": ["Mon"], "slots": [1]}),
    (Choice(tool="who_is_free", of="sections", days=["Thu"], part="afternoon", week=WEEK),
     "who_is_free", {"days": ["Thu"], "part": "afternoon", "sections": True}),
    (Choice(tool="describe", what="rooms", kind="lab", week=WEEK), "describe", {"what": "rooms", "kind": "lab"}),
    (Choice(tool="describe", what="teachers", kind="lab", week=WEEK), "describe", {"what": "teachers"}),
    (Choice(tool="describe", sections=["Section 4"], week=WEEK), "describe", {"section": "G-04"}),
    (Choice(tool="changes", week=WEEK), "changes", {}),
    (Choice(tool="free_hours", teachers=["me"], days=["Wed"], part="afternoon", week=WEEK),
     "free_hours", {"teachers": ["F-102"], "days": ["Wed"], "part": "afternoon"}),
    (Choice(tool="free_hours", teachers=["me"], hours=[5], part="afternoon", week=WEEK),  # a time beats a part
     "free_hours", {"teachers": ["F-102"], "slots": [5]}),
    (Choice(tool="free_hours", teachers=["me"], sections=["Section 1"], kind="lab", length=2, week=WEEK),
     "free_hours", {"teachers": ["F-102"], "sections": ["G-01"], "length": 2, "kind": "lab"}),
    (Choice(tool="classes_of", teachers=["me"], count=True, week=WEEK), "classes_of", {"teacher": "F-102", "count": True}),
])
def test_the_newer_tools(world, choice, tool, args):
    call = agent(world, choice).read("question")
    assert call.tool == tool and call.args == args and call.week == WEEK


def test_next_class_counts_from_now(world):
    a = ToolAgent(world.instance, FakeModel(Choice(tool="classes_of", teachers=["me"], next=True, week=WEEK)),
                  person(world, "Dr. Menon"), WEEK, today=date(2026, 10, 23), now=datetime(2026, 10, 23, 18))
    call = a.read("when's my next class?")
    assert call.args == {"teacher": person(world, "Dr. Menon"), "after": ["Mon", 0]} and call.week == WEEK + 1


@pytest.mark.parametrize("choice, problem", [
    (Choice(tool="describe", week=WEEK), "describe needs what"),
    (Choice(tool="free_hours", kind="lab", rooms=["Lab 2"], week=WEEK), "use free_rooms"),
    (Choice(tool="free_hours", teachers=["me"], length=5, week=WEEK), "at most 3 hours"),
    (Choice(tool="classes_of", course="Machine Learning", next=True, week=WEEK), "next needs a teacher"),
    (Choice(tool="changes", teachers=["Dr. Iyer", "Dr. Rao"], week=WEEK), "at most one teacher"),
])
def test_what_is_checked_in_the_newer_tools(world, choice, problem):
    a = agent(world, *[choice] * MAX_TRIES)
    a.read("question")
    assert problem in a.trace[0]["problem"]


def test_a_date_is_placed_by_the_calendar_not_the_model(world):
    start = date(2026, 7, 27)
    wrong_week = Choice(tool="free_hours", rooms=["Lab 2"], dates=["2026-10-28"], days=["Fri"], week=3)
    a = ToolAgent(world.instance, FakeModel(wrong_week), "F-102", WEEK, today=date(2026, 10, 21), semester_start=start)
    call = a.read("is lab 2 free on 28 oct?")
    assert call.week == 14 and call.args["days"] == ["Wed"]  # the model's week and day are replaced
    saturday = Choice(tool="free_hours", rooms=["Lab 2"], dates=["2026-10-31"], week=WEEK)
    a = ToolAgent(world.instance, FakeModel(saturday), "F-102", WEEK, today=date(2026, 10, 21), semester_start=start)
    assert "no classes on Saturday" in a.read("is lab 2 free on 31 oct?").note
    bad = Choice(tool="free_hours", rooms=["Lab 2"], dates=["28/10"], week=WEEK)
    a = ToolAgent(world.instance, FakeModel(*[bad] * MAX_TRIES), "F-102", WEEK, semester_start=start)
    a.read("is lab 2 free on 28/10?")
    assert "YYYY-MM-DD" in a.trace[0]["problem"]


def test_a_local_model_is_forced_to_write_only_the_tool():
    # the server's grammar makes every field before the last required one required; only "tool" may be
    assert grammar_schema(GrammarChoice.model_json_schema())["required"] == ["tool"]
    assert len(grammar_schema(Choice.model_json_schema())["required"]) > 10  # why Choice itself won't do


def test_a_week_left_out_is_this_week(world):
    no_week = GrammarChoice(tool="free_hours", teachers=["me"])
    call = agent(world, no_week).read("am I free?")
    assert call.week == WEEK and call.args == {"teachers": [person(world, "Dr. Menon")]}


def test_free_rooms_and_classes_of_take_the_regex_shapes(world):
    call = agent(world, Choice(tool="free_rooms", kind="lab", equipment=["gpu"], days=["Tue"], hours=[6, 5],
                               week=WEEK + 1)).read("any gpu lab free next tuesday afternoon 2 or 3?")
    assert call.args == {"kind": "lab", "equipment": ["gpu"], "days": ["Tue"], "slots": [5, 6]} and call.week == WEEK + 1
    call = agent(world, Choice(tool="classes_of", course="ML", week=WEEK)).read("when's ML?")
    assert call.args == {"course": next(c for c, t in world.instance.course_titles.items() if t == "Machine Learning")}


# -- "all", "each", "every": the model writes ["all"], the code puts in every teacher, section or room ---------


def test_all_teachers_free_together_is_every_teacher(world):
    call = agent(world, Choice(tool="free_hours", teachers=["all"], week=WEEK)).read("when are all faculty members free together?")
    assert call.tool == "free_hours" and call.args["teachers"] == [f.id for f in world.instance.faculty]


def test_each_section_is_the_list_of_sections(world):
    call = agent(world, Choice(tool="describe", sections=["all"], week=WEEK)).read("what courses does each section take?")
    assert (call.tool, call.args) == ("describe", {"what": "sections"})
    call = agent(world, Choice(tool="describe", rooms=["All"], kind="lab", week=WEEK)).read("list every lab")
    assert call.args == {"what": "rooms", "kind": "lab"}


def test_every_teacher_is_no_filter_in_classes_and_changes(world):
    call = agent(world, Choice(tool="classes_of", teachers=["all"], days=["Fri"], week=WEEK)).read("which teachers teach on Friday?")
    assert (call.tool, call.args) == ("classes_of", {"days": ["Fri"]})
    call = agent(world, Choice(tool="changes", sections=["all"], week=WEEK)).read("what changed for every section?")
    assert (call.tool, call.args) == ("changes", {})


# -- roles: the model writes the role, the code finds who holds it now ------------------------------------------


def hod(world):
    return next(f for f in world.instance.faculty if f.role.value == "hod")


def test_the_hod_is_whoever_holds_the_role(world):
    call = agent(world, Choice(tool="free_hours", roles=["hod"], days=["Tue"], week=WEEK)).read("is the HOD free on tuesday?")
    assert call.args == {"teachers": [hod(world).id], "days": ["Tue"], "as_role": {hod(world).id: "hod"}}
    # the HOD changes: the same reading now means the new one, with no retraining
    new = next(f for f in world.instance.faculty if f.role.value != "hod")
    a = ToolAgent(world.instance, FakeModel(Choice(tool="free_hours", roles=["hod"], week=WEEK)), person(world, "Dr. Menon"),
                  WEEK, today=date(2026, 10, 21), roles_of={new.id: "hod"})
    assert a.read("when is the HOD free?").args["teachers"] == [new.id]


def test_a_role_with_no_timetable_or_nobody_is_said_so(world):
    office = agent(world, Choice(tool="free_hours", roles=["coordinator"], week=WEEK)).read("when is the timetable office free?")
    assert not office.confident and "no classes in the timetable" in office.note
    nobody = agent(world, Choice(tool="free_hours", roles=["dean"], week=WEEK)).read("when is the dean free?")
    assert not nobody.confident and nobody.note == "There is no dean in the department data."


def test_roles_mix_with_names_and_class_reps_are_their_sections(world):
    call = agent(world, Choice(tool="free_hours", teachers=["me"], roles=["hod", "student"], week=WEEK)).read(
        "when are the HOD, the CRs and I free?")
    assert call.args["teachers"] == [person(world, "Dr. Menon"), hod(world).id]
    assert call.args["sections"] == [g.id for g in world.instance.groups]


def test_one_holder_needed_where_the_tool_takes_one(world):
    call = agent(world, Choice(tool="classes_of", roles=["student"], week=WEEK)).read("what do the class reps have today?")
    assert not call.confident and "Which one do you mean?" in call.note


def test_the_prompt_lists_roles_but_not_who_holds_them(world):
    a = agent(world, Choice(tool="none", week=WEEK))
    a.read("when is the HOD free?")
    line = next(x for x in a.client.prompts[0].splitlines() if x.startswith("Roles:"))
    assert "hod (head of department" in line and "faculty (" not in line.replace("guest_faculty (", "")
    assert hod(world).name not in line


# -- a course's teachers, the earliest time, changes of one kind ----------------------------------------------


def test_a_course_means_its_teachers_in_free_hours(world):
    s = world.instance.sessions[0]
    title = world.instance.course_title(s.course)
    call = agent(world, Choice(tool="free_hours", course=title, week=WEEK)).read(f"when is the {title} teacher free?")
    assert call.args["teachers"] == [s.faculty] and call.args["via_course"] == s.course


def test_earliest_and_changes_of_one_kind_are_kept(world):
    call = agent(world, Choice(tool="free_hours", teachers=["me"], sections=["Section 1"], first=True, week=WEEK)).read(
        "the earliest Section 1 and I are both free")
    assert call.args["first"] is True
    call = agent(world, Choice(tool="changes", change="cancelled", week=WEEK + 1)).read("any cancelled classes next week?")
    assert (call.tool, call.args, call.week) == ("changes", {"change": "cancelled"}, WEEK + 1)


@pytest.fixture(scope="module")
def shared(world):
    """The department with a course as they often are and the generated one never is: two co-teachers (one of
    its classes taught by a second teacher), the same for every section that takes it."""
    data = world.instance.model_dump()
    first = data["sessions"][0]
    co = next(s for s in data["sessions"] if s["course"] == first["course"] and s["id"] != first["id"])
    co["faculty"] = next(f["id"] for f in data["faculty"] if f["id"] not in (first["faculty"], "F-102"))
    from core.instance import Instance

    return Instance.model_validate(data), first, co


def test_co_teachers_are_answered_each_on_their_own(shared):
    from agents.lookup import WeekTimetable
    from agents.questions import Answerer

    inst, first, co = shared
    title = inst.course_titles[first["course"]]
    asker = next(f.id for f in inst.faculty if f.id not in (first["faculty"], co["faculty"]))  # not one of them ("You")
    a = ToolAgent(inst, FakeModel(Choice(tool="free_hours", course=title, week=WEEK)), asker, WEEK, today=date(2026, 10, 21))
    call = a.read(f"when is the {title} teacher free?")
    assert call.confident and call.args["each"] is True
    assert set(call.args["teachers"]) == {first["faculty"], co["faculty"]}
    names = [inst.faculty_by_id[f].name for f in (first["faculty"], co["faculty"])]
    from web.app import World

    week = WeekTimetable(inst, World(parser_mode="offline", seed_history=False, deadline=60).store, WEEK)
    reply = Answerer(inst, asker).answer(call, week)
    assert reply.startswith(f"{title} is taught by {names[0]} and {names[1]}. ")
    assert all(f"{n} is free on" in reply or f"{n} has no free hours" in reply for n in names), reply  # one sentence each


def test_both_teachers_of_a_course_together(shared):
    inst, first, co = shared
    title = inst.course_titles[first["course"]]
    a = ToolAgent(inst, FakeModel(Choice(tool="free_hours", course=title, teachers=["all"], week=WEEK)), "F-102", WEEK,
                  today=date(2026, 10, 21))
    call = a.read(f"when are both {title} teachers free?")
    assert call.confident and set(call.args["teachers"]) >= {first["faculty"], co["faculty"]}
    assert len(call.args["teachers"]) < len(inst.faculty)  # the course's teachers, not every teacher
