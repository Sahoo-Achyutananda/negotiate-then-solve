"""The test set for timetable questions (evaluation.lookup_set): templates filled from the department, names in
every form, and the scoring. No API calls."""

import json
import random
from datetime import date

import pytest

from agents.lookup import Directory
from agents.questions import QuestionReader, ToolCall
from agents.questions import dates_in, week_and_day
from evaluation.lookup_set import (NOW, PLACEHOLDER, SEMESTER_START, TEMPLATES, WEEK, Filled, TemplateFiller,
                                   fill_all, right, with_first_names)
from web.app import World


@pytest.fixture(scope="module")
def instance():
    return with_first_names(World(parser_mode="offline", seed_history=False, deadline=60).instance)


@pytest.fixture(scope="module")
def rows():
    return [json.loads(x) for x in TEMPLATES.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_first_names_are_added_and_one_is_shared(instance):
    names = [f.name for f in instance.faculty]
    assert "Dr. Imran Khan" in names and all(len(n.split()) == 3 for n in names)
    firsts = [n.split()[1] for n in names]
    assert any(firsts.count(x) > 1 for x in firsts)  # an ambiguous first name, on purpose


def test_every_template_fills_with_no_placeholder_left(instance, rows):
    questions = fill_all(instance, rows, fills=3)
    assert {f.id for f in questions} == {r["id"] for r in rows}  # each template gives at least one question
    assert not any(PLACEHOLDER.search(f.question) or PLACEHOLDER.search(json.dumps(f.args)) for f in questions)
    assert len({f.question for f in questions}) == len(questions)


def test_the_same_seed_asks_the_same_questions(instance, rows):
    assert [f.question for f in fill_all(instance, rows[:20])] == [f.question for f in fill_all(instance, rows[:20])]


def test_labels_name_what_the_question_names(instance):
    row = {"id": "X", "template": "is {teacher.any} free on {day.any} at {time}?", "tool": "free_hours",
           "args": {"teachers": ["{teacher}"], "days": ["{day}"], "slots": ["{time}"]}}
    for k in range(20):
        f = TemplateFiller(instance).fill(row, random.Random(k))
        teacher = instance.faculty_by_id[f.args["teachers"][0]]
        assert f.asker != teacher.id
        assert Directory(instance).find(f.question.split(" free on")[0].removeprefix("is ")) == [teacher]
        assert f.args["slots"][0] != instance.calendar.lunch_slot


@pytest.mark.parametrize("text", ["is Dr. Imran Khan free on monday?", "is Imran Khan free on monday?",
                                  "is imran free on monday?", "is khan free on monday?", "is Khan sir free on monday?"])
def test_the_regex_reads_a_teacher_in_any_form(instance, text):
    khan = next(f.id for f in instance.faculty if f.name.endswith("Khan"))
    call = QuestionReader(instance, "F-102", WEEK, today=date(2026, 10, 21)).read(text)
    assert call.args == {"teachers": [khan], "days": ["Mon"]}


def test_a_shared_first_name_is_asked_about_and_a_full_name_is_not(instance):
    reader = QuestionReader(instance, "F-102", WEEK, today=date(2026, 10, 21))
    shared = next(f.name.split()[1] for f in instance.faculty
                  if [g.name.split()[1] for g in instance.faculty].count(f.name.split()[1]) > 1)
    call = reader.read(f"is {shared} free on monday?")
    assert not call.confident and call.note.startswith("Which one do you mean")
    one = next(f for f in instance.faculty if f.name.split()[1] == shared)
    call = reader.read(f"is {' '.join(one.name.split()[1:])} free on monday?")  # "Anil Rao": one person
    assert call.confident and call.args["teachers"] == [one.id]


@pytest.mark.parametrize("text", ["why is Imran absent on monday?", "when are the holidays?", "when do mid-sems start?"])
def test_what_the_timetable_must_not_or_cannot_say(instance, text):
    call = QuestionReader(instance, "F-102", WEEK, today=date(2026, 10, 21)).read(text)
    assert call is not None and not call.confident


@pytest.mark.parametrize("text", ["28 October", "28th Oct", "Oct 28", "28/10", "28-10-2026", "Wednesday 28 October",
                                  "28th of oct"])
def test_dates_in_the_ways_people_write_them(text):
    assert dates_in(f"is lab 2 free on {text}?", date(2026, 10, 21)) == [date(2026, 10, 28)]


@pytest.mark.parametrize("text", ["free 2-4 pm on monday", "at 10.30 am", "a room for 30 students", "section-1"])
def test_times_and_numbers_are_not_dates(text):
    assert dates_in(text, date(2026, 10, 21)) == []


def test_a_date_is_a_week_and_a_day(instance):
    assert week_and_day(instance, SEMESTER_START, date(2026, 10, 28)) == (14, "Wed")
    assert week_and_day(instance, SEMESTER_START, date(2026, 7, 27)) == (1, "Mon")
    assert "no classes on Saturday" in week_and_day(instance, SEMESTER_START, date(2026, 10, 31))
    assert "outside this semester" in week_and_day(instance, SEMESTER_START, date(2026, 12, 25))


def test_the_regex_reads_a_date(instance):
    reader = QuestionReader(instance, "F-102", WEEK, today=date(2026, 10, 21), semester_start=SEMESTER_START)
    lab2 = next(r.id for r in instance.rooms if r.name == "Lab 2")
    call = reader.read("is Lab 2 free on 28/10 at 2 pm?")
    assert call.args == {"rooms": [lab2], "days": ["Wed"], "slots": [5]} and call.week == 14
    assert not reader.read("is Lab 2 free on 31 October?").confident  # a Saturday
    no_start = QuestionReader(instance, "F-102", WEEK, today=date(2026, 10, 21))
    assert not no_start.read("is Lab 2 free on 28 October?").confident  # no calendar: not guessed


def test_scoring():
    want = Filled("X", "g", "q", "F-102", "free_hours", {"teachers": ["F-102", "F-103"]}, WEEK,
                  also=[{"tool": "classes_of", "args": {"teacher": "F-103"}}])
    assert right(want, ToolCall("free_hours", {"teachers": ["F-103", "F-102"]}, WEEK))  # any order
    assert right(want, ToolCall("classes_of", {"teacher": "F-103"}, WEEK))  # also counts
    assert not right(want, ToolCall("free_hours", {"teachers": ["F-102"]}, WEEK))
    assert not right(want, ToolCall("free_hours", {"teachers": ["F-102", "F-103"]}, WEEK + 1))
    assert right(Filled("X", "g", "q", "F-102", None), None)
    assert not right(Filled("X", "g", "q", "F-102", None), ToolCall("free_hours", {}, WEEK))
    assert right(Filled("X", "g", "q", "F-102", "unclear"), ToolCall("unclear", week=WEEK, confident=False))


def test_regex_score_on_the_test_set(instance, rows):
    """The baseline Gemini must beat (printed with -s); requests read as lookups are listed, not hidden."""
    questions = fill_all(instance, rows)
    got = [(f, QuestionReader(instance, f.asker, WEEK, today=NOW.date(), now=NOW,
                              semester_start=SEMESTER_START).read(f.question)) for f in questions]
    ok = sum(right(f, c) for f, c in got)
    requests_as_lookups = [f.question for f, c in got if f.tool is None and c is not None]
    print(f"\nregex on the test set: {ok}/{len(questions)}; requests or rule questions read as lookups: "
          f"{len(requests_as_lookups)}")
    assert ok > len(questions) // 2


# -- other departments (ROADMAP step 6): does a reader read, or remember the department? ---------------------


def test_a_generated_department_has_the_same_size_but_other_pairs():
    from evaluation.lookup_set import department

    base = World(parser_mode="offline", seed_history=False, deadline=60).instance
    other = department(base, seed=7)
    assert (len(other.faculty), len(other.groups), len(other.rooms)) == (len(base.faculty), len(base.groups), len(base.rooms))
    pairs = lambda i: {(s.course, s.faculty) for s in i.sessions}  # noqa: E731
    assert pairs(other) != pairs(base)


def test_shuffling_moves_the_names_the_hod_and_the_titles():
    from evaluation.lookup_set import department

    base = World(parser_mode="offline", seed_history=False, deadline=60).instance
    shuffled = department(base, shuffle=True)
    hod = lambda i: next(f.id for f in i.faculty if f.role.value == "hod")  # noqa: E731
    assert sorted(f.name for f in shuffled.faculty) == sorted(f.name for f in base.faculty)
    assert [f.name for f in shuffled.faculty] != [f.name for f in base.faculty]
    assert hod(shuffled) != hod(base) or [f.name for f in shuffled.faculty][0] != base.faculty[0].name
    assert sorted(shuffled.course_titles.values()) == sorted(base.course_titles.values())
    assert shuffled.course_titles != base.course_titles


def test_the_questions_are_labelled_for_the_department_asked_about(rows):
    from evaluation.lookup_set import department

    base = World(parser_mode="offline", seed_history=False, deadline=60).instance
    other = with_first_names(department(base, seed=3, shuffle=True))
    questions = fill_all(other, rows, fills=1)
    ids = {f.id for f in other.faculty} | {g.id for g in other.groups} | {r.id for r in other.rooms} | set(other.course_titles)
    named = [v for f in questions for k, v in f.args.items() if k in ("teachers", "sections", "rooms") and isinstance(v, list)]
    assert named and all(x in ids for v in named for x in v)
    # the regex reads names from the department, so it scores about the same on a department it was never tuned on
    reader = lambda f: QuestionReader(other, f.asker, WEEK, today=NOW.date(), now=NOW, semester_start=SEMESTER_START)  # noqa: E731
    assert sum(right(f, reader(f).read(f.question)) for f in questions) > len(questions) // 2


def test_a_seed_that_makes_the_apps_own_department_is_refused():
    from evaluation.lookup_set import department

    base = World(parser_mode="offline", seed_history=False, deadline=60).instance  # the demo: seed 1 at this size
    with pytest.raises(ValueError, match="same department"):
        department(base, seed=1)
