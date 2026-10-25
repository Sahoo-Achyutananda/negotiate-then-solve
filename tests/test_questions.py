"""Questions about the timetable (agents.questions): what the regex reader makes of them, and the replies."""

import json
from datetime import date, datetime
from pathlib import Path

import pytest

from agents.lookup import Directory, WeekTimetable
from agents.questions import Answerer, QuestionReader, ToolCall, _spans
from language.corpus import FULL_DAY, hour
from pipeline.store import Store
from web.app import World

WEEK = 13
WEDNESDAY = date(2026, 10, 21)  # "today" for the tests: "tomorrow" is Thursday
QUESTIONS = Path(__file__).parents[1] / "data" / "lookup-questions.jsonl"


@pytest.fixture(scope="module")
def world():
    return World(parser_mode="offline", seed_history=False, deadline=60)


def person(world, name):
    return next(f.id for f in world.instance.faculty if f.name == name)


@pytest.fixture(scope="module")
def reader(world):
    return QuestionReader(world.instance, sender=person(world, "Dr. Menon"), current_week=WEEK, today=WEDNESDAY)


@pytest.mark.parametrize("text, tool, week", [
    ("What are my free slots this week?", "free_hours", WEEK),
    ("any free hours next week?", "free_hours", WEEK + 1),
    ("free slots in week 15 for section 4", "free_hours", 15),
    ("what do I teach tomorrow?", "classes_of", WEEK),
])
def test_reads_the_kind_of_question_and_the_week(reader, text, tool, week):
    call = reader.read(text)
    assert call is not None and call.confident and call.tool == tool and call.week == week


def test_both_free_names_me_and_the_colleague(world, reader):
    call = reader.read("When are Dr. Iyer and I both free on Friday?")
    assert call.args == {"teachers": [person(world, "Dr. Menon"), person(world, "Dr. Iyer")], "days": ["Fri"]}


def test_a_room_at_an_hour(world, reader):
    call = reader.read("is lab 2 free on thursday at 3 pm")
    lab2 = next(r.id for r in world.instance.rooms if r.name == "Lab 2")
    assert call.args == {"rooms": [lab2], "days": ["Thu"], "slots": [6]}  # 3 pm is slot 6; nobody else asked about


def test_tomorrow_is_the_next_teaching_day(world, reader):
    assert reader.read("what do I teach tomorrow?").args["days"] == ["Thu"]
    friday = QuestionReader(world.instance, person(world, "Dr. Menon"), WEEK, today=date(2026, 10, 23))
    call = friday.read("what do I teach tomorrow?")
    assert call.args["days"] == ["Mon"] and call.week == WEEK + 1  # the weekend: Monday of the next week


def test_a_course_and_nobody_named_means_everyones_classes(world, reader):
    call = reader.read("When is Machine Learning this week?")
    assert call.tool == "classes_of" and "teacher" not in call.args and call.args["course"]


@pytest.mark.parametrize("text", [
    "Can I move my Friday lecture to Monday?",
    "Please cancel my class tomorrow.",
    "Can I hold my class before lunch, from 12 to 1?",
    "I want Friday free",
    "What is the wifi password for the library?",
])
def test_requests_and_other_questions_are_not_read_as_timetable_questions(reader, text):
    assert reader.read(text) is None


@pytest.mark.parametrize("text, note", [
    ("Is Dr. Smith free on Monday?", "don't know a teacher called Smith"),
    ("is room 999 free?", "don't know room 999"),
])
def test_unknown_names_are_not_guessed(reader, text, note):
    call = reader.read(text)
    assert not call.confident and note in call.note


def test_spans_merge_hours_into_ranges():
    assert _spans([("Mon", 2), ("Mon", 3), ("Mon", 5), ("Tue", 0)]) == "Monday 11 am–1 pm and 2–3 pm; Tuesday 9–10 am"
    assert _spans([("Mon", 2), ("Mon", 3), ("Mon", 5), ("Mon", 7)]) == "Monday 11 am–1 pm, 2–3 pm and 4–5 pm"


# -- replies ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def week(world):
    return WeekTimetable(world.instance, world.store, WEEK)


def test_free_hours_reply_lists_exactly_the_free_hours(world, week):
    me = person(world, "Dr. Menon")
    reply = Answerer(world.instance, me).answer(ToolCall("free_hours", {"teachers": [me]}, WEEK), week)
    assert reply == f"You are free on {_spans(week.free_hours(teachers=[me]))} (week {WEEK})."


def test_room_at_an_hour_is_yes_or_no(world, week):
    used = week.classes[0]
    room, (day, slot) = used.placement.room, used.hours()[0]
    a = Answerer(world.instance, person(world, "Dr. Menon"))
    assert a.answer(ToolCall("free_hours", {"rooms": [room], "days": [day], "slots": [slot]}, WEEK), week).startswith("No,")
    free = week.free_hours(rooms=[room])[0]
    assert a.answer(ToolCall("free_hours", {"rooms": [room], "days": [free[0]], "slots": [free[1]]}, WEEK), week).startswith("Yes,")


def test_classes_reply_names_every_class(world, week):
    iyer = person(world, "Dr. Iyer")
    reply = Answerer(world.instance, person(world, "Dr. Menon")).answer(ToolCall("classes_of", {"teacher": iyer}, WEEK), week)
    assert reply.startswith("Dr. Iyer teaches in week 13: ") and reply.count(";") == len(week.classes_of(teacher=iyer)) - 1


def test_a_teacher_who_does_not_teach_the_course_is_told_who_does(world, week):
    c = week.classes[0]
    course, teacher = c.session.course, c.session.faculty
    other = next(f.id for f in world.instance.faculty if f.id != teacher
                 and not week.classes_of(teacher=f.id, course=course))
    reply = Answerer(world.instance, person(world, "Dr. Menon")).answer(
        ToolCall("classes_of", {"teacher": other, "course": course}, WEEK), week)
    taught_by = world.instance.faculty_by_id[teacher].name
    assert "doesn't teach" in reply and f"taught by {taught_by}" in reply, reply


def test_a_section_question_names_each_teacher(world, week):
    c = week.classes[0]
    section = c.session.groups[0]
    reply = Answerer(world.instance, person(world, "Dr. Menon")).answer(
        ToolCall("classes_of", {"section": section}, WEEK), week)
    assert reply.startswith(f"{world.instance.group_by_id[section].name}'s classes in week {WEEK}: ")
    assert all(world.instance.faculty_by_id[x.session.faculty].name in reply for x in week.classes_of(section=section))


# -- the question set: how many the regex reader gets right (the baseline for a model, ROADMAP.md step 5) --


def expected_call(world, me, row):
    """The question set names people and things; the reader returns IDs."""
    d = Directory(world.instance)

    def ids(names):
        return [me if n == "me" else (x.id if not isinstance(x := d.find(n)[0], str) else x) for n in names]

    out = {}
    for k, v in row.get("args", {}).items():
        out[k] = ids(v) if k in ("teachers", "sections", "rooms") else ids([v])[0] if k in ("teacher", "section", "room", "course") else v
    return out


def test_question_set_score(world):
    me = person(world, "Dr. Menon")
    rows = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    reader = QuestionReader(world.instance, sender=me, current_week=WEEK, today=WEDNESDAY)
    right, misses = 0, []
    for row in rows:
        call = reader.read(row["question"])
        if row["tool"] is None:
            ok = call is None
        elif row["tool"] == "unclear":
            ok = call is not None and not call.confident
        else:
            week = {"this": WEEK, "next": WEEK + 1}.get(row["week"], row["week"])
            ok = (call is not None and call.confident and call.tool == row["tool"] and call.week == week
                  and call.args == expected_call(world, me, row))
        right += ok
        if not ok:
            misses.append(f"{row['id']} {row['question']!r} -> {call}")
    print(f"\nregex reader: {right}/{len(rows)} right" + "".join(f"\n  miss {m}" for m in misses))
    assert right >= len(rows) - 2, misses  # the baseline; known misses: wordings the regex has no pattern for


def test_which_rooms_are_free(world, reader, week):
    call = reader.read("are there any labs free next week on monday ??")
    assert call.tool == "free_rooms" and call.args == {"kind": "lab", "days": ["Mon"]} and call.week == WEEK + 1
    reply = Answerer(world.instance, person(world, "Dr. Menon")).answer(ToolCall("free_rooms", {"kind": "lab", "days": ["Mon"]}, WEEK), week)
    labs = week.free_rooms(kind="lab", days=["Mon"])
    assert reply.startswith(f"Free labs on Monday (week {WEEK}): ") and all(r.name in reply for r, _ in labs)


def test_which_rooms_are_free_at_an_hour(world, week):
    a = Answerer(world.instance, person(world, "Dr. Menon"))
    reply = a.answer(ToolCall("free_rooms", {"kind": "lecture", "days": ["Fri"], "slots": [5]}, WEEK), week)
    free = [r.name for r, h in week.free_rooms(kind="lecture", days=["Fri"]) if ("Fri", 5) in h]
    assert reply == (f"Lecture rooms free on Friday at 2 pm (week {WEEK}): " + ", ".join(free[:-1]) + (" and " if len(free) > 1 else "") + free[-1] + "."
                     if free else f"No lecture rooms are free on Friday at 2 pm (week {WEEK}).")


def test_the_auditorium_is_not_in_the_timetable(reader):
    call = reader.read("is the auditorium free on friday?")
    assert not call.confident and "auditorium" in call.note


# -- the newer tools: describe, who_is_free, changes; next class, counts, parts of the day, a room too -----------


@pytest.mark.parametrize("text, tool, args", [
    ("can you enlist all rooms with their capacities?", "describe", {"what": "rooms"}),
    ("list all labs", "describe", {"what": "rooms", "kind": "lab"}),
    ("list all teachers", "describe", {"what": "teachers"}),
    ("who is free on monday at 10 am?", "who_is_free", {"days": ["Mon"], "slots": [1]}),
    ("anyone free thursday afternoon?", "who_is_free", {"days": ["Thu"], "part": "afternoon"}),
    ("which sections are free on friday at 9 am?", "who_is_free", {"days": ["Fri"], "slots": [0], "sections": True}),
    ("what changed this week?", "changes", {}),
    ("any labs free on monday morning?", "free_rooms", {"kind": "lab", "days": ["Mon"], "part": "morning"}),
])
def test_reads_the_newer_tools(reader, text, tool, args):
    call = reader.read(text)
    assert call.tool == tool and call.args == args and call.week == WEEK


def test_reads_me_into_the_newer_options(world, reader):
    me = person(world, "Dr. Menon")
    assert reader.read("how many hours do I teach this week?").args == {"teacher": me, "count": True}
    assert reader.read("was my class cancelled this week?").args == {"teacher": me}
    assert reader.read("am I free this afternoon?").args == {"teachers": [me], "days": ["Wed"], "part": "afternoon"}
    call = reader.read("when are section 1 and I free for 2 hours with a lab?")
    assert call.args == {"teachers": [me], "sections": ["G-01"], "length": 2, "kind": "lab"}


def test_next_class_counts_from_now(world):
    me = person(world, "Dr. Menon")
    at = QuestionReader(world.instance, me, WEEK, today=WEDNESDAY, now=datetime(2026, 10, 21, 10, 30))
    assert at.read("when is my next class?").args == {"teacher": me, "after": ["Wed", 2]}  # 10:30: from 11 am
    friday_night = QuestionReader(world.instance, me, WEEK, today=date(2026, 10, 23), now=datetime(2026, 10, 23, 18))
    call = friday_night.read("when is my next class?")
    assert call.args["after"] == ["Mon", 0] and call.week == WEEK + 1


def test_describe_replies(world, week):
    a = Answerer(world.instance, person(world, "Dr. Menon"))
    rooms = a.answer(ToolCall("describe", {"what": "rooms"}, WEEK), week)
    assert all(f"{r.name} ({r.type.value}, {r.capacity} seats" in rooms for r in world.instance.rooms)
    g = world.instance.groups[0]
    assert a.answer(ToolCall("describe", {"section": g.id}, WEEK), week).startswith(f"{g.name} has {g.size} students")
    assert "this is week 13" in a.answer(ToolCall("describe", {"what": "calendar"}, WEEK), week)


def test_who_is_free_leaves_out_the_busy_and_the_asker(world, week):
    c = week.classes[0]
    (day, slot), busy = c.hours()[0], c.session.faculty
    me = next(f.id for f in world.instance.faculty if f.id != busy)
    reply = Answerer(world.instance, me).answer(ToolCall("who_is_free", {"days": [day], "slots": [slot]}, WEEK), week)
    assert world.instance.faculty_by_id[busy].name not in reply and world.instance.faculty_by_id[me].name not in reply


def test_counts_and_next_class_replies(world, week):
    me = person(world, "Dr. Menon")
    a = Answerer(world.instance, me)
    mine = week.classes_of(teacher=me)
    hours = sum(c.session.duration for c in mine)
    assert a.answer(ToolCall("classes_of", {"teacher": me, "count": True}, WEEK), week) == \
        f"You have {len(mine)} classes ({hours} hours) in week {WEEK}."
    first = mine[0]
    reply = a.answer(ToolCall("classes_of", {"teacher": me, "after": [first.placement.day, first.placement.slot]}, WEEK), week)
    assert reply.startswith("Your next class: ") and world.instance.course_title(first.session.course) in reply
    last = mine[-1]
    assert "no more classes" in a.answer(ToolCall(
        "classes_of", {"teacher": me, "after": [last.placement.day, last.placement.slot + 1]}, WEEK), week)


def test_changes_reply(world):
    store = Store()  # a copy of the published timetable; one class not held in WEEK
    semester = world.store.current_version()
    store.publish_version(store.propose_version(dict(semester.assignment), "test").version, "test")
    gone = WeekTimetable(world.instance, store, WEEK).classes[0]
    v = store.propose_version(dict(semester.assignment), "test", week=WEEK, cancelled=[gone.session.id])
    store.publish_version(v.version, "test")
    a = Answerer(world.instance, person(world, "Dr. Menon"))
    p, what = gone.placement, f"{world.instance.course_title(gone.session.course)} {gone.session.kind.value}"
    assert a.answer(ToolCall("changes", {}, WEEK), WeekTimetable(world.instance, store, WEEK)) == \
        f"Changes in week {WEEK}: {what} on {FULL_DAY[p.day]} {hour(p.slot)} is not held."
    assert a.answer(ToolCall("changes", {}, WEEK + 1), WeekTimetable(world.instance, store, WEEK + 1)) == \
        f"No changes in week {WEEK + 1}: the timetable is as planned."


def test_all_of_them_is_said_as_all(world, week):
    a = Answerer(world.instance, person(world, "Dr. Menon"))
    everyone = [f.id for f in world.instance.faculty]
    reply = a.answer(ToolCall("free_hours", {"teachers": everyone}, WEEK), week)
    assert reply.startswith("All teachers are free on") or reply.startswith("All teachers have no free hours")
    assert "Dr. Iyer" not in reply
    with_one = a.answer(ToolCall("free_hours", {"teachers": everyone, "sections": [world.instance.groups[0].id]}, WEEK), week)
    assert with_one.startswith(f"All teachers and {world.instance.groups[0].name} ")


def test_the_sections_list_says_what_each_studies(world, week):
    reply = Answerer(world.instance, person(world, "Dr. Menon")).answer(ToolCall("describe", {"what": "sections"}, WEEK), week)
    for g in world.instance.groups:
        course = next(world.instance.course_title(s.course) for s in world.instance.sessions if g.id in s.groups)
        assert f"{g.name} ({g.size} students): " in reply and course in reply


def test_classes_with_nobody_named_are_everyones(world, week):
    reply = Answerer(world.instance, person(world, "Dr. Menon")).answer(ToolCall("classes_of", {"days": ["Fri"]}, WEEK), week)
    assert reply.startswith(f"Classes on Friday in week {WEEK}: ") and "Dr." in reply  # each with its teacher


def test_a_role_is_said_with_who_holds_it(world, week):
    a = Answerer(world.instance, person(world, "Dr. Menon"))
    head = next(f for f in world.instance.faculty if f.role.value == "hod")
    reply = a.answer(ToolCall("free_hours", {"teachers": [head.id], "as_role": {head.id: "hod"}}, WEEK), week)
    assert reply.startswith(f"The HOD ({head.name}) is free on") or reply.startswith(f"The HOD ({head.name}) has no")
    who = a.answer(ToolCall("describe", {"teacher": head.id, "as_role": {head.id: "hod"}}, WEEK), week)
    assert who.startswith(f"The HOD is {head.name}, who teaches")


def test_changes_of_one_kind_answer_yes_or_no(world):
    store = Store()  # as test_changes_reply: one class not held in WEEK
    semester = world.store.current_version()
    store.publish_version(store.propose_version(dict(semester.assignment), "test").version, "test")
    gone = WeekTimetable(world.instance, store, WEEK).classes[0]
    store.publish_version(store.propose_version(dict(semester.assignment), "test", week=WEEK,
                                                cancelled=[gone.session.id]).version, "test")
    week = WeekTimetable(world.instance, store, WEEK)
    a = Answerer(world.instance, person(world, "Dr. Menon"))
    assert a.answer(ToolCall("changes", {"change": "cancelled"}, WEEK), week).startswith(
        f"Yes, 1 class is not held in week {WEEK}: ")
    assert a.answer(ToolCall("changes", {"change": "moved"}, WEEK), week) == \
        f"No classes are moved in week {WEEK}. (1 other change that week: ask what changed to see it.)"
    assert a.answer(ToolCall("changes", {"change": "extra"}, WEEK), week).startswith("There are no extra classes")


def test_earliest_and_a_course_s_teachers(world, week):
    me = person(world, "Dr. Menon")
    a = Answerer(world.instance, me)
    free = week.free_hours([me])
    reply = a.answer(ToolCall("free_hours", {"teachers": [me], "first": True}, WEEK), week)
    assert reply == f"The earliest you are free is {_spans([free[0]])} (week {WEEK})."
    s = next(x for x in world.instance.sessions if x.faculty != me)  # someone else's course ("You" otherwise)
    teacher = world.instance.faculty_by_id[s.faculty]
    reply = a.answer(ToolCall("free_hours", {"teachers": [s.faculty], "via_course": s.course}, WEEK), week)
    assert reply.startswith(f"{world.instance.course_title(s.course)} is taught by {teacher.name}. {teacher.name} ")
