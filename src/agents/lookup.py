"""Answers to questions about the timetable, computed from it (never from a model): which classes are held
in a week, when people, sections and rooms are free, and who or what a name in a question means.

    week = WeekTimetable(instance, store, week=13)
    week.classes_of(teacher="F-103", course="C-015")    # when does Dr. Iyer teach Machine Learning?
    week.free_hours(teachers=["F-102", "F-103"])         # when are Dr. Menon and Dr. Iyer both free?
    Directory(instance).find("lab 2")                    # [Room Lab 2]

A model may later decide which of these to call (ROADMAP.md, steps 2 and 4); what they return is the
timetable's answer.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from core.instance import Faculty, Group, Instance, Room, Session
from core.schemas import Placement
from pipeline.store import Store


class ScheduledClass:
    """One class held in one week: a Session (what it is) at a Placement (when and where)."""

    def __init__(self, session: Session, placement: Placement, week: int) -> None:
        self.session = session
        self.placement = placement
        self.week = week

    def hours(self) -> list[tuple[str, int]]:
        """The (day, slot) pairs it takes up: a 2-hour practical takes two."""
        p = self.placement
        return [(p.day, slot) for slot in range(p.slot, p.slot + self.session.duration)]

    def __repr__(self) -> str:
        return f"ScheduledClass({self.session.id} on {self.placement.day} slot {self.placement.slot}, week {self.week})"


class WeekTimetable:
    """The timetable as it stands in one week, as the Timetable page shows it: the week's own version if it
    has one (an absence, a closed room), else the semester's; classes not held that week left out; that
    week's extra classes added."""

    def __init__(self, instance: Instance, store: Store, week: int) -> None:
        self.instance = instance
        self.week = week
        semester = store.current_version()
        version = store.current_version(week) or semester
        placed = dict(version.assignment) if version else {}
        not_held = set(version.cancelled) if version else set()
        sessions = dict(instance.session_by_id)
        self.extras = store.extras(week)
        for extra in self.extras:
            sessions[extra.session.id] = extra.session
            placed.setdefault(extra.session.id, extra.placement)
        self.classes = [ScheduledClass(sessions[sid], p, week)
                        for sid, p in placed.items() if sid in sessions and sid not in not_held]
        # for changes(): where the semester timetable has each class, and what this week leaves out
        self.semester_placed = dict(semester.assignment) if semester else {}
        self.placed, self.not_held = placed, not_held

    def classes_of(self, teacher: str | None = None, section: str | None = None, room: str | None = None,
                   course: str | None = None, day: str | None = None) -> list[ScheduledClass]:
        """Tool ``classes_of``: the week's classes that match every filter given, in time order.

        Answers: "what do I teach on Wednesday?", "when does Dr. Iyer teach ML?", "who teaches Section 2 on
        Monday?", "what's in Lab 3 on Wednesday?", "how many hours do I teach?" (counted from the list).
        Args: teacher, section, room, course (IDs: "F-103", "G-02", "L-3", "C-021"); day ("Mon"). None: any.
        Returns: [ScheduledClass]."""
        found = [c for c in self.classes
                 if (teacher is None or c.session.faculty == teacher)
                 and (section is None or section in c.session.groups)
                 and (room is None or c.placement.room == room)
                 and (course is None or c.session.course == course)
                 and (day is None or c.placement.day == day)]
        days = self.instance.calendar.days
        return sorted(found, key=lambda c: (days.index(c.placement.day), c.placement.slot))

    def free_hours(self, teachers: Sequence[str] = (), sections: Sequence[str] = (), rooms: Sequence[str] = (),
                   days: Sequence[str] | None = None, length: int = 1) -> list[tuple[str, int]]:
        """Tool ``free_hours``: the hours when ALL the given teachers, sections and rooms are free; never lunch.

        Answers: "what are my free slots?", "when are Dr. Iyer and I both free on Friday?", "is Lab 2 free on
        Thursday at 3 pm?", "free slots for Section 4", "a 2-hour slot when I'm free" (``length=2``).
        Args: teachers, sections, rooms (lists of IDs); days (["Mon", ...]; None: the whole week);
        length (only hours in a run of at least that many free hours in a row; lunch breaks a run).
        Returns: [(day, slot)]."""
        busy = set()
        for c in self.classes:
            if (c.session.faculty in teachers or set(c.session.groups) & set(sections)
                    or c.placement.room in rooms):
                busy.update(c.hours())
        cal = self.instance.calendar
        free = [(day, slot) for day in cal.days if days is None or day in days
                for slot in range(cal.slots_per_day)
                if slot != cal.lunch_slot and (day, slot) not in busy]
        return in_runs(free, length)

    def free_with_room(self, teachers: Sequence[str] = (), sections: Sequence[str] = (), kind: str | None = None,
                       equipment: Sequence[str] = (), min_capacity: int | None = None,
                       days: Sequence[str] | None = None, length: int = 1) -> list[tuple[Room, list[tuple[str, int]]]]:
        """Tool ``free_hours`` with a room needed: for each room that fits, the hours when it AND all the people
        are free (planning an extra class or a meeting).

        Answers: "when can Section 1 and I have a 2-hour class in a lab?", "when are Sections 1 and 2 free, with
        a room for 110?".
        Args: teachers, sections (lists of IDs); kind ("lecture" | "lab"; None: any); equipment (["gpu"]);
        min_capacity (seats); days; length (as in ``free_hours``).
        Returns: [(Room, [(day, slot)])], rooms with no such hours left out."""
        people = set(self.free_hours(teachers, sections, days=days))
        out = []
        for room in self.instance.rooms:
            if self._fits(room, kind, equipment, min_capacity):
                both = [h for h in self.free_hours(rooms=[room.id], days=days) if h in people]
                if hours := in_runs(both, length):
                    out.append((room, hours))
        return out

    def who_is_free(self, days: Sequence[str] | None = None, slots: Sequence[int] | None = None,
                    course: str | None = None, sections: bool = False) -> list[tuple[Faculty | Group, list[tuple[str, int]]]]:
        """Tool ``who_is_free``: which teachers (or sections) are free at the hours asked.

        Answers: "who is free on Monday at 10 am?", "anyone free Thursday afternoon?", "who could cover my Data
        Structures class?" (``course``), "which sections are free on Friday at 9 am?" (``sections=True``).
        Args: days (None: the whole week); slots (None: every hour); course (ID: only teachers who teach it);
        sections (True: sections instead of teachers).
        Returns: [(Faculty | Group, [(day, slot)])]: each one with their free hours among those asked."""
        if sections:
            people = list(self.instance.groups)
        else:
            teach = {s.faculty for s in self.instance.sessions if course is None or s.course == course}
            people = [f for f in self.instance.faculty if f.id in teach]
        out = []
        for p in people:
            free = (self.free_hours(sections=[p.id], days=days) if sections
                    else self.free_hours(teachers=[p.id], days=days))
            if slots:
                free = [h for h in free if h[1] in slots]
            if free:
                out.append((p, free))
        return out

    def next_class(self, after: tuple[str, int], teacher: str | None = None,
                   section: str | None = None) -> ScheduledClass | None:
        """Tool ``classes_of`` with ``after``: the first class starting at or after a time, this week.

        Answers: "when is my next class?", "where should Section 3 be next?".
        Args: after ((day, slot): usually the time now); teacher or section (ID).
        Returns: ScheduledClass, or None when there are no more classes this week."""
        days = self.instance.calendar.days
        start = (days.index(after[0]), after[1])
        return next((c for c in self.classes_of(teacher=teacher, section=section)
                     if (days.index(c.placement.day), c.placement.slot) >= start), None)

    def changes(self, teacher: str | None = None, section: str | None = None) -> list[Change]:
        """Tool ``changes``: how this week differs from the semester timetable. It says THAT a class is not held,
        never why (reasons are private: P-PRIVACY).

        Answers: "what changed this week?", "was my class cancelled?", "any extra classes for Section 3?".
        Args: teacher or section (ID; None: everyone's changes).
        Returns: [Change]: kind "not_held", "moved" (before -> after) or "extra"."""
        out = []
        for sid, before in self.semester_placed.items():
            s = self.instance.session_by_id.get(sid)
            if s is None or not _about(s, teacher, section):
                continue
            if sid in self.not_held:
                out.append(Change("not_held", s, before, None))
            elif (after := self.placed.get(sid)) is not None and after != before:
                out.append(Change("moved", s, before, after))
        for x in self.extras:
            if _about(x.session, teacher, section):
                out.append(Change("extra", x.session, None, x.placement))
        return out

    def free_rooms(self, kind: str | None = None, equipment: Sequence[str] = (), min_capacity: int | None = None,
                   days: Sequence[str] | None = None) -> list[tuple[Room, list[tuple[str, int]]]]:
        """Tool ``free_rooms``: which rooms are free, and when.

        Answers: "any labs free on Monday?", "which rooms are free on Friday at 2 pm?", "a lab with GPUs free on
        Tuesday?", "is there a free classroom for 70 students?".
        Args: kind ("lecture" | "lab"; None: any); equipment (["gpu"]: the room has all of it); min_capacity
        (seats); days (None: the whole week).
        Returns: [(Room, [(day, slot)])]: each room with its free hours, rooms with none left out."""
        rooms = [r for r in self.instance.rooms if self._fits(r, kind, equipment, min_capacity)]
        out = [(r, self.free_hours(rooms=[r.id], days=days)) for r in rooms]
        return [(r, hours) for r, hours in out if hours]

    @staticmethod
    def _fits(room: Room, kind: str | None, equipment: Sequence[str], min_capacity: int | None) -> bool:
        return ((kind is None or room.type.value == kind) and set(equipment) <= set(room.equipment)
                and (min_capacity is None or room.capacity >= min_capacity))


class Change:
    """One way a week differs from the semester timetable: a class ``"not_held"``, ``"moved"`` (from ``before``
    to ``after``) or an ``"extra"`` class."""

    def __init__(self, kind: str, session: Session, before: Placement | None, after: Placement | None) -> None:
        self.kind = kind
        self.session = session
        self.before = before
        self.after = after

    def __repr__(self) -> str:
        return f"Change({self.kind} {self.session.id}: {self.before} -> {self.after})"


class Department:
    """Tool ``describe``: facts about the department that do not depend on the week, from the department file.

    Answers: "list all rooms with their capacities", "list the labs", "how big is Section 4?", "which courses
    does Dr. Iyer teach?", "who teaches TOC?", "list the teachers", "how many hours a week does Dr. Rao teach?".
    Args (per method): kind ("lecture" | "lab"), teacher or course (IDs).
    Returns: the department's own objects (Room, Group, Faculty, Session) or counts."""

    def __init__(self, instance: Instance) -> None:
        self.instance = instance

    def rooms(self, kind: str | None = None) -> list[Room]:
        """Rooms of a kind ("lab", "lecture"; None: all), as the department file lists them."""
        return [r for r in self.instance.rooms if kind is None or r.type.value == kind]

    def sections(self) -> list[Group]:
        """Every section with its size: "list the sections", "how many students are in each section?"."""
        return list(self.instance.groups)

    def teachers(self) -> list[Faculty]:
        """Every teacher: "list the teachers" (their courses come from ``courses_of``)."""
        return list(self.instance.faculty)

    def courses_of(self, teacher: str) -> list[str]:
        """The courses (IDs) a teacher teaches, in the department file's order."""
        return list(dict.fromkeys(s.course for s in self.instance.sessions if s.faculty == teacher))

    def teachers_of(self, course: str) -> list[Faculty]:
        ids = {s.faculty for s in self.instance.sessions if s.course == course}
        return [f for f in self.instance.faculty if f.id in ids]

    def sections_of(self, course: str) -> list[Group]:
        ids = {g for s in self.instance.sessions if s.course == course for g in s.groups}
        return [g for g in self.instance.groups if g.id in ids]

    def sessions_of(self, course: str) -> list[Session]:
        """A course's weekly classes (lectures, tutorials, practicals)."""
        return [s for s in self.instance.sessions if s.course == course]

    def weekly_hours(self, teacher: str) -> int:
        """Teaching hours a week in the semester timetable (P-LOAD's limit is about these)."""
        return sum(s.duration for s in self.instance.sessions if s.faculty == teacher)


def part_slots(instance: Instance, part: str) -> list[int]:
    """The slots of "morning" (before lunch) or "afternoon" (after it); never the lunch hour."""
    cal = instance.calendar
    lunch = cal.lunch_slot if cal.lunch_slot is not None else cal.slots_per_day // 2
    return list(range(0, lunch)) if part == "morning" else list(range(lunch + 1, cal.slots_per_day))


def in_runs(hours: list[tuple[str, int]], length: int) -> list[tuple[str, int]]:
    """Only the hours in a run of at least ``length`` hours in a row on one day (the hours given, in order)."""
    if length <= 1:
        return hours
    have, keep = set(hours), set()
    for day, slot in hours:
        run = [(day, s) for s in range(slot, slot + length)]
        if all(h in have for h in run):
            keep.update(run)
    return [h for h in hours if h in keep]


def _about(session: Session, teacher: str | None, section: str | None) -> bool:
    return (teacher is None or session.faculty == teacher) and (section is None or section in session.groups)


class Directory:
    """Finds the teachers, sections, rooms and courses a question names, the way people name them:
    "iyer", "Dr. Iyer", "section 4", "lab 2", "machine learning", "ML"."""

    def __init__(self, instance: Instance) -> None:
        self.instance = instance

    def find(self, text: str) -> list[Faculty | Group | Room | str]:
        """Everything ``text`` could mean: Faculty, Group and Room objects, and course IDs (e.g. "C-015")."""
        words = _words(text)
        if not words:
            return []
        found: list[Faculty | Group | Room | str] = []
        found += [f for f in self.instance.faculty if _matches(words, f.name)]
        found += [g for g in self.instance.groups if _matches(words, g.name)]
        found += [r for r in self.instance.rooms if _matches(words, r.name)]
        for course, title in self.instance.course_titles.items():
            if _matches(words, title) or _matches(words, course) or "".join(words) in _initials(title):
                found.append(course)
        return found


TITLES = {"dr", "prof", "professor", "mr", "ms", "mrs", "sir", "maam", "madam"}  # "Khan sir", "Iyer ma'am"
# short and other forms people write for the words in names: "rm 104", "classroom 101", "sec 2" (a prefix like
# "sec" already matches "section"; these are the ones that don't start the same)
ALIASES: dict[str, str] = {"rm": "room", "classroom": "room", "laboratory": "lab", "grp": "section", "group": "section",
                           "batch": "section"}  # not "lecture" ("DS lecture" is a course) nor "hall" (not a room)


def _words(text: str) -> list[str]:
    """Lower-case words without titles, short forms written out: "Dr. Iyer" -> ["iyer"], "Lab 2" -> ["lab", "2"],
    "rm 104" -> ["room", "104"]."""
    return [ALIASES.get(w, w) for w in re.findall(r"[a-z0-9]+", text.lower().replace("'", "")) if w not in TITLES]


def _matches(words: list[str], name: str) -> bool:
    """Every word asked for starts a word of the name: "iyer" ~ "Dr. Iyer", "machine" ~ "Machine Learning",
    "lab 2" ~ "Lab 2" (but not "Lab 20")."""
    names = _words(name)
    return all(any(n == w or (not w.isdigit() and n.startswith(w)) for n in names) for w in words)


def _initials(title: str) -> set[str]:
    """How a course is abbreviated: "Machine Learning" -> {"ml"}, "Theory of Computation" -> {"toc", "tc"}."""
    words = _words(title)
    return {"".join(w[0] for w in words), "".join(w[0] for w in words if w not in {"of", "and", "the", "in", "for"})}
