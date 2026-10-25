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
    (Choice(tool="free_hours", week=WEEK), "needs at least one teacher"),
    (Choice(tool="free_rooms", week=WEEK), "needs kind"),
    (Choice(tool="free_rooms", kind="lab", equipment=["laser"], week=WEEK), "equipment ['laser']"),
    (Choice(tool="free_hours", teachers=["me"], hours=[4], week=WEEK), "not lunch"),
    (Choice(tool="classes_of", teachers=["Dr. Iyer", "Dr. Rao"], week=WEEK), "at most one teacher"),
    (Choice(tool="free_hours", teachers=["me"], week=40), "week 40 does not exist"),
])
def test_what_is_checked(world, choice, problem):
    a = agent(world, *[choice] * MAX_TRIES)
    a.read("question")
    assert problem in a.trace[0]["problem"]


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
