"""v3's lookups (agents.more_lookups) and its reading (ToolAgent(multi=True)): several lookups per question, the new
tools, room_for and several weeks. No API calls: a fake model gives the readings."""

from datetime import date

import pytest

from agents.lookup import WeekTimetable
from agents.more_lookups import Records, RequestRow
from agents.questions import Answerer, ToolCall
from agents.tool_agent import ChoiceV3, Lookups, ToolAgent
from web.app import World

WEEK = 13


class FakeModel:
    def __init__(self, *answers) -> None:
        self.answers = list(answers)

    def generate(self, system, prompt, schema):
        self.system = system
        return self.answers.pop(0)


@pytest.fixture(scope="module")
def world():
    return World(parser_mode="offline", seed_history=False, deadline=60)


@pytest.fixture(scope="module")
def week(world):
    return WeekTimetable(world.instance, world.store, WEEK)


def person(world, name):
    return next(f.id for f in world.instance.faculty if f.name == name)


def read(world, *choices, sender="Dr. Menon"):
    who = person(world, sender) if sender.startswith("Dr.") else sender
    model = FakeModel(Lookups(lookups=list(choices)))
    return ToolAgent(world.instance, model, who, WEEK, today=date(2026, 10, 21), multi=True).read("question"), model


def answer(world, call, week, sender="Dr. Menon", **records):
    who = person(world, sender) if sender.startswith("Dr.") else sender
    rec = Records(world.instance, world.store, **records)
    return Answerer(world.instance, who, rec).answer(call, week)


# -- reading: several lookups, the v3 prompt --------------------------------------------------------------


def test_several_lookups_become_one_multi_call_answered_in_order(world, week):
    call, model = read(world, ChoiceV3(tool="free_hours", sections=["Section 1"], week=WEEK),
                       ChoiceV3(tool="free_hours", rooms=["Room 104"], days=["Mon"], week=WEEK))
    assert "my_requests" in model.system and "lookups" in model.system  # the v3 prompt
    assert call.tool == "multi" and [p.args for p in call.args["parts"]] == [{"sections": ["G-01"]},
                                                                               {"rooms": ["R-104"], "days": ["Mon"]}]
    text = answer(world, call, week)
    assert text.index("Section 1") < text.index("Room 104")


def test_one_lookup_is_plain_and_none_is_not_a_timetable_question(world):
    call, _ = read(world, ChoiceV3(tool="calendar", week=WEEK))
    assert call.tool == "calendar"
    assert read(world, ChoiceV3(tool="none", week=WEEK))[0] is None


def test_a_wrong_part_is_sent_back_with_its_number(world):
    model = FakeModel(Lookups(lookups=[ChoiceV3(tool="calendar", week=WEEK), ChoiceV3(tool="move_options", week=WEEK)]),
                      Lookups(lookups=[ChoiceV3(tool="calendar", week=WEEK)]))
    a = ToolAgent(world.instance, model, person(world, "Dr. Menon"), WEEK, today=date(2026, 10, 21), multi=True)
    assert a.read("question").tool == "calendar"
    assert a.trace[0]["problem"].startswith("lookup 2: move_options needs the course")


def test_v2_keeps_its_own_prompt(world):
    from agents.tool_agent import SYSTEM, SYSTEM_V3, Choice

    model = FakeModel(Choice(tool="none", week=WEEK))
    ToolAgent(world.instance, model, person(world, "Dr. Menon"), WEEK).read("hi")
    assert model.system == SYSTEM and "my_requests" not in SYSTEM and "my_requests" in SYSTEM_V3


# -- the new tools -----------------------------------------------------------------------------------------


def test_calendar(world, week):
    text = answer(world, ToolCall("calendar", {"days": ["Mon"]}, 14), week, semester_start=date(2026, 7, 27),
                  current_week=13)
    assert text.startswith("This is week 13 of") and "Monday 26 October" in text


def test_my_requests(world, week):
    me = person(world, "Dr. Menon")
    rows = [RequestRow("R-00001", me, "Please move my DS lecture to Tuesday 9 am", "published", "Done.", "2026-10-20T10:00"),
            RequestRow("R-00002", me, "cancel my Friday class", "forwarded", None, "2026-10-21T10:00"),
            RequestRow("R-00003", "F-199", "someone else's", "published", None, "2026-10-21T11:00")]
    latest = answer(world, ToolCall("my_requests", {"which": "latest"}, WEEK), week, requests=rows)
    assert latest.startswith("R-00002") and "forwarded to the timetable office" in latest
    assert "still open" in answer(world, ToolCall("my_requests", {"which": "open"}, WEEK), week, requests=rows)
    assert answer(world, ToolCall("my_requests", {"id": "R-00003"}, WEEK), week, requests=rows) == \
        "R-00003 is not one of your requests."
    assert "R-00003" in answer(world, ToolCall("my_requests", {"which": "waiting"}, WEEK), week, requests=rows,
                               waiting_on={me: {"R-00003"}})


def test_request_numbers_are_read_in_any_form(world):
    call, _ = read(world, ChoiceV3(tool="my_requests", request_id="r18", week=WEEK))
    assert call.args == {"id": "R-00018"}


def test_rules_fairness_bookings_and_what_if_without_their_data(world, week):
    assert answer(world, ToolCall("my_rules", {}, WEEK), week).startswith(("You have not set", "Your "))
    assert answer(world, ToolCall("my_fairness", {}, WEEK), week).startswith("You have")
    assert answer(world, ToolCall("bookings", {}, WEEK), week) == "There are no room bookings."
    assert "can't try out changes" in answer(world, ToolCall("what_if", {"request": "move my DS lecture"}, WEEK), week)


def test_clash_check_counts_and_gaps(world, week):
    clash = answer(world, ToolCall("clash_check", {"sections": ["G-01", "G-02"]}, WEEK), week)
    assert "Section 1 and Section 2" in clash and "same time" in clash
    most = answer(world, ToolCall("counts", {"of": "teachers", "measure": "teaching_hours", "order": "most"}, WEEK), week)
    assert "the most hours of class" in most and most.count(";") >= 3
    gap = answer(world, ToolCall("gaps", {"teachers": [person(world, "Dr. Menon")], "measure": "free"}, WEEK), week)
    assert "longest free block" in gap


def test_move_options(world, week):
    me = person(world, "Dr. Menon")
    s = next(c.session for c in week.classes if c.session.faculty == me and c.session.duration == 1)
    text = answer(world, ToolCall("move_options", {"sessions": [s.id]}, WEEK), week)
    assert text.startswith("Your ") and ("could move to" in text or "no free time" in text)
    call, _ = read(world, ChoiceV3(tool="move_options", course=world.instance.course_title(s.course), week=WEEK))
    assert s.id in call.args["sessions"]


def test_room_for_and_several_weeks(world, week):
    lab = next(r for r in world.instance.rooms if r.type.value == "lab")
    call, _ = read(world, ChoiceV3(tool="free_hours", sections=["Section 1", "Section 2"], rooms=[lab.name],
                                   room_for=["Section 1", "Section 2"], week=WEEK))
    assert call.args["room_for"] == ["G-01", "G-02"]
    need = sum(world.instance.group_by_id[g].size for g in ("G-01", "G-02"))
    text = answer(world, call, week)
    assert (f"have {need} students together" in text) == (lab.capacity < need)
    call, _ = read(world, ChoiceV3(tool="free_hours", teachers=["me"], weeks=[13, 14], week=WEEK))
    assert call.args["weeks"] == [13, 14]
    assert answer(world, call, week).startswith(("Free in every one of weeks 13–14", "There is no hour free"))


def test_records_from_the_app(world):
    rec = Records.from_world(world)
    assert rec.instance is world.instance and isinstance(rec.requests, list)


def test_what_if_is_a_dry_run(world, week):
    me = person(world, "Dr. Menon")
    s = next(c for c in week.classes if c.session.faculty == me and c.session.duration == 1)
    title = world.instance.course_title(s.session.course)
    day = next(d for d in ("Mon", "Tue", "Wed", "Thu", "Fri") if d != s.placement.day)
    versions = len(world.store.versions())
    text = answer(world, ToolCall("what_if", {"request": f"move my {title} {s.session.kind.value} to {day} at 9 am in week {WEEK}, it has to be then"}, WEEK),
                  week, what_if=world.what_if)
    assert text.startswith(("If you sent it now", "Nothing would have to move", "That can't be fitted",
                            "To try that out", "That change would be refused", "I couldn't read"))
    assert len(world.store.versions()) == versions  # nothing stored


def test_a_club_is_named_with_or_without_club(world):
    call, _ = read(world, ChoiceV3(tool="bookings", club="the Music Club", week=WEEK))
    assert call.args == {"club": "Music"}


def test_v3_has_no_lists_of_names_and_says_what_it_cannot_find(world):
    from agents.tool_agent import SYSTEM_V3

    call, model = read(world, ChoiceV3(tool="free_hours", teachers=["dr bose"], week=WEEK))
    assert not call.confident and call.note == 'I don\'t know a teacher called "dr bose". Could you check the name?'
    assert "Teachers:" not in model.system and "Dr. Iyer" not in SYSTEM_V3
    call, _ = read(world, ChoiceV3(tool="free_hours", teachers=["iyer sir"], sections=["sec 2"], week=WEEK))
    assert call.args == {"teachers": [person(world, "Dr. Iyer")], "sections": ["G-02"]}


def test_short_forms_of_rooms_and_sections_are_found(world):
    from agents.lookup import Directory

    d = Directory(world.instance)
    assert [r.name for r in d.find("rm 104")] == ["Room 104"] and [r.name for r in d.find("classroom 103")] == ["Room 103"]
    assert [g.name for g in d.find("grp 4")] == ["Section 4"]
    assert not d.find("seminar hall")  # a place the timetable doesn't have stays unknown


# -- each one on its own, or together ----------------------------------------------------------------------


def test_separately_is_each_ones_own_free_hours_and_without_it_together(world, week):
    sections = ["Section 1", "sec 2", "section-3"]
    call, _ = read(world, ChoiceV3(tool="free_hours", sections=sections, days=["Mon"], separately=True, week=WEEK))
    assert call.tool == "multi" and [p.args for p in call.args["parts"]] == [
        {"sections": [g], "days": ["Mon"]} for g in ("G-01", "G-02", "G-03")]
    said = answer(world, call, week)
    assert said.startswith("Section 1 is free") and "Section 2 is free" in said and "Section 3" in said
    call, _ = read(world, ChoiceV3(tool="free_hours", sections=sections, days=["Mon"], week=WEEK))
    assert call.tool == "free_hours" and call.args["sections"] == ["G-01", "G-02", "G-03"]  # one time for all


def test_separately_splits_any_mix_and_every_one(world):
    me = person(world, "Dr. Menon")
    call, _ = read(world, ChoiceV3(tool="free_hours", teachers=["me"], rooms=["Lab 1"], hours=[6], days=["Tue"],
                                   separately=True, week=WEEK))
    assert [p.args for p in call.args["parts"]] == [{"teachers": [me], "slots": [6], "days": ["Tue"]},
                                                    {"rooms": ["L-1"], "slots": [6], "days": ["Tue"]}]
    call, _ = read(world, ChoiceV3(tool="gaps", sections=["all"], separately=True, week=WEEK))
    assert [p.args["sections"] for p in call.args["parts"]] == [[g.id] for g in world.instance.groups]
    call, _ = read(world, ChoiceV3(tool="free_hours", sections=["Section 1"], separately=True, week=WEEK))
    assert call.tool == "free_hours"  # one named: nothing to split


def test_a_tool_of_one_asked_about_several_answers_each(world, week):
    call, _ = read(world, ChoiceV3(tool="classes_of", sections=["Section 1", "Section 2"], days=["Mon"], week=WEEK),
                   ChoiceV3(tool="describe", rooms=["Lab 1", "Room 104"], week=WEEK))
    assert [(p.tool, p.args) for p in call.args["parts"]] == [
        ("classes_of", {"section": "G-01", "days": ["Mon"]}), ("classes_of", {"section": "G-02", "days": ["Mon"]}),
        ("describe", {"room": "L-1"}), ("describe", {"room": "R-104"})]
    said = answer(world, call, week)
    assert said.index("Section 1's classes") < said.index("Section 2's classes")


# -- the checker honours what the model read (fixes after testing v3) ----------------------------------------


def test_a_name_marked_unknown_that_is_on_the_staff_is_known(world):
    """v3 sees no list of names: a bare "Priya" may be a student to it; the department says she is Dr. Priya Menon."""
    from evaluation.lookup_set import with_first_names

    inst = with_first_names(world.instance)
    model = FakeModel(Lookups(lookups=[ChoiceV3(tool="classes_of", teachers=["Priya"], unknown=["Priya"], days=["Wed"],
                                                week=WEEK)]),
                      Lookups(lookups=[ChoiceV3(tool="free_hours", unknown=["Rohit Sharma"], week=WEEK)]),
                      Lookups(lookups=[ChoiceV3(tool="free_rooms", unknown=["exam hall"], week=WEEK)]))
    a = ToolAgent(inst, model, inst.faculty[0].id, WEEK, today=date(2026, 10, 21), multi=True)
    priya = next(f.id for f in inst.faculty if f.name == "Dr. Priya Menon")
    assert a.read("does Priya have anything on wednesday?").args == {"teacher": priya, "days": ["Wed"]}
    rohit = next(f.id for f in inst.faculty if f.name == "Dr. Rohit Sharma")
    assert a.read("Rohit Sharma free?").args["teachers"] == [rohit]  # only in unknown: put in its list
    call = a.read("is the exam hall free?")
    assert not call.confident  # not the department's: still unknown


def test_dates_in_two_weeks_are_one_lookup_a_week(world, week):
    """ "This Friday and next Monday": a lookup is about one week, so the dates of each week are their own."""
    from evaluation.lookup_set import SEMESTER_START

    model = FakeModel(Lookups(lookups=[ChoiceV3(tool="free_hours", sections=["Section 1", "Section 2"],
                                                dates=["2026-10-23", "2026-10-26"], week=WEEK)]))
    a = ToolAgent(world.instance, model, person(world, "Dr. Menon"), WEEK, today=date(2026, 10, 21),
                  semester_start=SEMESTER_START, multi=True)
    call = a.read("are sections 1 and 2 free this friday and next monday?")
    assert call.tool == "multi" and [(p.week, p.args["days"]) for p in call.args["parts"]] == [(13, ["Fri"]), (14, ["Mon"])]


def test_weeks_the_semester_does_not_have_are_said(world):
    call, _ = read(world, ChoiceV3(tool="free_hours", teachers=["me"], weeks=[15, 16, 17], week=WEEK))
    assert not call.confident and "week 17 isn't in it" in call.note
