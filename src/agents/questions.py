"""Questions about the timetable, in English: what a question asks (``QuestionReader``, regex for now) and the
reply (``Answerer``), with the facts from ``agents.lookup``.

    reader = QuestionReader(instance, sender="F-102", current_week=13)
    call = reader.read("when are Dr. Iyer and I both free on Friday?")
    # ToolCall(tool="free_hours", args={"teachers": ["F-102", "F-103"], "days": ["Fri"]}, week=13)
    Answerer(instance, sender="F-102").answer(call, WeekTimetable(instance, store, call.week))
    # "You and Dr. Iyer are both free on Friday 10 am–12 pm and 3–4 pm (week 13)."

The reader only says WHAT to look up; ``lookup`` says what is true. A model can later replace the reader:
it produces the same ``ToolCall``, and nothing else changes (ROADMAP.md, step 4).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from core.instance import Faculty, Group, Instance, Room
from language.corpus import FULL_DAY, hour

from .lookup import TITLES, Department, Directory, ScheduledClass, WeekTimetable, part_slots

# what makes a message a timetable question, and what makes it a request instead
FREE = re.compile(r"\b(free|available|availability|empty|vacant|gaps?|unbooked)\b", re.IGNORECASE)
CLASSES = re.compile(r"\b(teach(es|ing)?|classes|class|lectures?|practicals?|tutorials?|timetable|schedule)\b|"
                     r"\bwhat do (i|we) have\b|\bwhen (is|are|does|do)\b", re.IGNORECASE)
REQUEST = re.compile(r"\b(cancel|move|reschedule|shift|swap|exchange|postpone|prepone|book|please|kindly)\b|"
                     r"\b(i|we) (want|need|would like|'d like)\b|"
                     r"\b(can|could|may) (i|we|you) (have|get|put|hold|teach|keep|take|make)\b", re.IGNORECASE)
# a question about what the rules allow, not about what the timetable has: for the rule search, not a lookup
RULE = re.compile(r"\b(allowed|permitted|rules?|polic(y|ies)|regulations?|approv\w*)\b|"
                  r"\bhow (many|much|far|long|often)\b.*\b(can|may|must|should)\b|"
                  r"\b(maximum|minimum|at most|at least)\b.*\b(can|may|must|allowed)\b", re.IGNORECASE)
ME = re.compile(r"\b(i|me|my|mine|myself|we|us|our)\b", re.IGNORECASE)
DAY = re.compile(r"\b(mon|tues?|wed(?:nes)?|thu(?:rs?)?|fri)(day)?s?\b", re.IGNORECASE)
WEEK = re.compile(r"\bweek\s*(\d{1,2})\b", re.IGNORECASE)
TIME = re.compile(r"\b(\d{1,2})(?::\d\d)?\s*(am|pm)\b", re.IGNORECASE)
TEACHER = re.compile(r"\b(?:dr|prof|professor)\.?\s+([a-z]+)", re.IGNORECASE)
SECTION = re.compile(r"\bsection\s*(\d+)\b", re.IGNORECASE)
ROOM = re.compile(r"\b(lab|room)\s*(\d+)\b", re.IGNORECASE)
INITIALS = re.compile(r"\b[A-Z]{2,4}\b")  # "ML", "TOC" as written (not lower case: "is", "do")
# a kind of room, not one room ("any labs", "a free classroom", "lecture halls"; "Lab 2" is ROOM)
LABS = re.compile(r"\b(labs?|laborator(y|ies))\b(?!\s*\d)", re.IGNORECASE)
ROOMS = re.compile(r"\b(rooms?|classrooms?|lecture (halls?|rooms?)|halls?)\b(?!\s*\d)", re.IGNORECASE)
NOT_IN_TIMETABLE = re.compile(r"\b(auditorium|seminar (hall|room)s?|conference (hall|room)s?)\b", re.IGNORECASE)
SEATS = re.compile(r"\bfor\s+(\d{2,3})\b|\b(\d{2,3})\s*(?:students|people|seats|persons)\b", re.IGNORECASE)
# the newer tools and options: only wordings that are common and unambiguous (a model reads the rest)
PART = re.compile(r"\b(morning|afternoon)s?\b", re.IGNORECASE)
LENGTH = re.compile(r"\b(2|two|3|three)[- ]hours?\b", re.IGNORECASE)
NEXT = re.compile(r"\bnext (class|lecture|lab|practical|tutorial)\b", re.IGNORECASE)
COUNT = re.compile(r"\bhow many (hours|classes|lectures)\b", re.IGNORECASE)
WHO_FREE = re.compile(r"\bwho(?:'s| is| are)? (free|available)\b|\b(any(one|body)|some(one|body)) (free|available)\b|"
                      r"\bwhich (teachers|faculty|sections) (are |is )?(free|available)\b", re.IGNORECASE)
DESCRIBE = re.compile(r"\b(list|all)\b.*\b(rooms|labs|classrooms|sections|teachers|faculty|courses)\b|"
                      r"\bcapacit(y|ies)\b|\bhow (big|large) is\b|\bhow many students\b", re.IGNORECASE)
CHANGES = re.compile(r"\bwhat(?:'s| has| have)? changed\b|\bchanges\b|\bcancell?ed\b|\bnot (being )?held\b",
                     re.IGNORECASE)
# questions the timetable must not, or cannot, answer: said so, never guessed
PRIVATE = re.compile(r"\bwhy\b.*\b(absent|away|on leave|off|not (coming|here|in)|cancell?ed|missing)\b", re.IGNORECASE)
NOT_IN_DATA = re.compile(r"\b(holidays?|vacations?|exams?|examinations?|mid-?sems?|end-?sems?)\b", re.IGNORECASE)
# Roles as answers say them ("The HOD (Dr. Rao) is free on ..."); a role not listed is said from its name
# ("guest_faculty" -> "the guest faculty"), so a role added to core.schemas.Role needs nothing here.
ROLE_NAMES = {"hod": "the HOD", "coordinator": "the timetable coordinator", "lab_incharge": "the lab in-charge",
              "student": "the class representatives"}


# "changes" of one kind, as the model names them -> as WeekTimetable.changes() gives them
CHANGE_KINDS = {"cancelled": "not_held", "moved": "moved", "extra": "extra"}


def role_name(role: str) -> str:
    return ROLE_NAMES.get(role, f"the {role.replace('_', ' ')}")


PRIVATE_NOTE = "I can't say why someone is away: the reasons are private. I can tell you which classes are not held."
NOT_IN_DATA_NOTE = "I don't know that: holidays and exam dates are not in the timetable."
# calendar dates: "28 October", "28th Oct", "Oct 28", "28/10", "28-10-2026" (day first, as written in India)
MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
_MON = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
DATE_DM = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{_MON}(?:,?\s+(\d{{4}}))?", re.IGNORECASE)
DATE_MD = re.compile(rf"\b{_MON}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(\d{{4}}))?", re.IGNORECASE)
DATE_NUM = re.compile(r"\b(\d{1,2})[/.-](\d{1,2})(?:[/.-](\d{2,4}))?\b(?!\s*(?:am|pm|hours?|hrs?)\b)", re.IGNORECASE)


@dataclass
class ToolCall:
    """What to look up: the same shape whether the regex or (later) a model decided it."""

    tool: str  # "free_hours" | "classes_of" | "free_rooms" | "unclear"
    args: dict = field(default_factory=dict)
    week: int = 1
    confident: bool = True
    note: str = ""  # when not confident: what to ask or say instead


class QuestionReader:
    """Reads a question's shape with regex, names with ``Directory``. Returns None for anything that is not a
    timetable question (a request for a change, a question about the rules), so another part handles it."""

    def __init__(self, instance: Instance, sender: str, current_week: int, today: date | None = None,
                 now: datetime | None = None, semester_start: date | None = None) -> None:
        self.instance = instance
        self.sender = sender
        self.current_week = current_week
        self.today = today or date.today()
        self.now = now  # for "my next class"; the clock when not given
        self.semester_start = semester_start  # for dates ("28 October" -> week 14, Wednesday); None: no dates
        self.directory = Directory(instance)

    def read(self, text: str) -> ToolCall | None:
        if REQUEST.search(text):
            return None  # a change is being asked for, not information
        if RULE.search(text):
            return None  # what the rules allow: the rule search answers it
        if PRIVATE.search(text):
            return ToolCall("unclear", week=self.current_week, confident=False, note=PRIVATE_NOTE)
        if NOT_IN_DATA.search(text):
            return ToolCall("unclear", week=self.current_week, confident=False, note=NOT_IN_DATA_NOTE)
        free, classes = FREE.search(text), CLASSES.search(text)
        newer = CHANGES.search(text) or WHO_FREE.search(text) or DESCRIBE.search(text) or COUNT.search(text)
        if not free and not classes and not newer:
            return None
        week, days = self._when(text)
        if dated := dates_in(text, self.today):  # a date is more exact than a day name or "this week"
            if self.semester_start is None:
                return ToolCall("unclear", week=week, confident=False, note="I can't tell which week that date is in.")
            found = [week_and_day(self.instance, self.semester_start, d) for d in dated]
            if bad := next((x for x in found if isinstance(x, str)), None):
                return ToolCall("unclear", week=week, confident=False, note=bad)
            week = found[0][0]
            days = list(dict.fromkeys(d for w, d in found if w == week))
        teachers, sections, rooms, courses, unclear = self._names(text)
        slots = sorted({_slot(int(h), ap) for h, ap in TIME.findall(text)} & set(range(self.instance.calendar.slots_per_day)))
        part = (m.group(1).lower() if (m := PART.search(text)) and not slots else None)
        if NOT_IN_TIMETABLE.search(text) and not rooms:
            return ToolCall("unclear", week=week, confident=False,
                            note="The timetable has lecture rooms and labs only; the auditorium and seminar halls "
                                 "are booked through the timetable office.")
        if unclear:
            return ToolCall("unclear", week=week, confident=False, note=unclear)
        kind = "lab" if LABS.search(text) else "lecture" if ROOMS.search(text) else None

        if CHANGES.search(text):  # "what changed this week?", "was my class cancelled?"
            if ME.search(text):
                self._add_me(teachers, sections)
            args = {"teacher": teachers[0] if teachers else None, "section": sections[0] if sections else None}
            return ToolCall("changes", {k: v for k, v in args.items() if v}, week)
        if WHO_FREE.search(text):  # "who is free on Monday at 10 am?"
            args = {"days": days, "slots": slots, "part": part, "course": courses[0] if courses else None,
                    "sections": bool(re.search(r"\bsections?\b", text, re.IGNORECASE))}
            return ToolCall("who_is_free", {k: v for k, v in args.items() if v}, week)
        if DESCRIBE.search(text) and not free:  # "list all rooms with their capacities", "how big is section 4"
            return ToolCall("describe", self._describe_args(text, teachers, sections, rooms, courses), week)

        if free and kind and not (teachers or sections or rooms):  # "any labs free on Monday?": which rooms
            args = {"kind": kind, "equipment": self._equipment(text), "min_capacity": self._seats(text),
                    "days": days, "slots": slots, "part": part}
            return ToolCall("free_rooms", {k: v for k, v in args.items() if v}, week)
        if ME.search(text) or not (teachers or sections or rooms or courses):  # "my free hours", or no one named
            self._add_me(teachers, sections)
        if free:
            length = {"2": 2, "two": 2, "3": 3, "three": 3}[m.group(1).lower()] if (m := LENGTH.search(text)) else None
            args = {"teachers": teachers, "sections": sections, "rooms": rooms, "days": days, "slots": slots,
                    "part": part, "length": length}
            if kind:  # "when can section 1 and I use a lab?": free, and a room of that kind free too
                args |= {"kind": kind, "equipment": self._equipment(text), "min_capacity": self._seats(text)}
            return ToolCall("free_hours", {k: v for k, v in args.items() if v}, week)
        args = {"teacher": teachers[0] if teachers else None, "section": sections[0] if sections else None,
                "room": rooms[0] if rooms else None, "course": courses[0] if courses else None, "days": days,
                "count": bool(COUNT.search(text))}
        if NEXT.search(text):  # "when is my next class?": the first one from now
            week, after = self._from_now()
            args["after"] = list(after)
        return ToolCall("classes_of", {k: v for k, v in args.items() if v}, week)

    def _describe_args(self, text: str, teachers: list[str], sections: list[str], rooms: list[str],
                       courses: list[str]) -> dict:
        """One thing named ("how big is section 4"), else the kind of thing listed ("list all labs")."""
        if sections:
            return {"section": sections[0]}
        if rooms:
            return {"room": rooms[0]}
        if courses:
            return {"course": courses[0]}
        if teachers:
            return {"teacher": teachers[0]}
        low = text.lower()
        what = next((w for w, words in (("sections", ("section",)), ("teachers", ("teacher", "faculty")),
                                        ("courses", ("course",)), ("rooms", ("room", "lab", "capacit")))
                     if any(x in low for x in words)), "rooms")
        # "all rooms" means every room; only "labs" or "classrooms"/"lecture rooms" narrow it
        kind = "lab" if LABS.search(text) else "lecture" if re.search(r"\b(classrooms?|lecture)\b", low) else None
        return {"what": what, "kind": kind} if what == "rooms" and kind else {"what": what}

    def _from_now(self) -> tuple[int, tuple[str, int]]:
        return from_now(self.instance, self.current_week, self.now or datetime.now())

    def _when(self, text: str) -> tuple[int, list[str]]:
        """The week asked about (this week unless it says otherwise) and the days, if any."""
        low, week, days = text.lower(), self.current_week, []
        if m := WEEK.search(text):
            week = int(m.group(1))
        elif "next week" in low:
            week += 1
        for word, shift in (("today|tonight|this (?:morning|afternoon)", 0), ("tomorrow", 1)):
            if re.search(rf"\b(?:{word})\b", low):
                d = self.today + timedelta(days=shift)
                if d.weekday() > 4:  # a weekend: the next Monday, in the next week
                    d += timedelta(days=7 - d.weekday())
                    week += 1
                days.append(self.instance.calendar.days[d.weekday()])
        for m in DAY.finditer(text):
            day = m.group(1)[:3].title()
            if day in self.instance.calendar.days and day not in days:
                days.append(day)
        weeks = self.instance.calendar.weeks
        return min(max(week, 1), weeks), days

    def _names(self, text: str) -> tuple[list[str], list[str], list[str], list[str], str]:
        """Teachers, sections, rooms and courses the question names (IDs), and what is unclear, if anything."""
        teachers, unclear = self._teachers(text)
        sections, rooms, courses = [], [], []
        for m in SECTION.finditer(text):
            found = [x for x in self.directory.find(m.group(0)) if isinstance(x, Group)]
            sections += [g.id for g in found[:1]]
            unclear = unclear or ("" if found else f"I don't know {m.group(0)}.")
        for m in ROOM.finditer(text):
            found = [x for x in self.directory.find(m.group(0)) if isinstance(x, Room)]
            rooms += [r.id for r in found[:1]]
            unclear = unclear or ("" if found else f"I don't know {m.group(0)}.")
        low = text.lower()
        for course, title in self.instance.course_titles.items():
            if title.lower() in low or course.lower() in low:
                courses.append(course)
        for abbr in INITIALS.findall(text):
            courses += [c for c in self.directory.find(abbr) if isinstance(c, str) and c not in courses]
        return teachers, sections, rooms, courses, unclear

    def _teachers(self, text: str) -> tuple[list[str], str]:
        """The teachers a question names, in any form the department's names allow: "Dr. Imran Khan", "Imran
        Khan", "Khan", "imran", "khan sir". A full name is matched before its parts, so "Imran Khan" is one
        person even when another teacher is also called Imran; a first name two teachers share is asked about."""
        low, found, taken, unclear = text.lower().replace("'", ""), [], [], ""
        parts = {f.id: [w for w in re.findall(r"[a-z]+", f.name.lower()) if w not in TITLES] for f in self.instance.faculty}

        def at(phrase: str) -> list[tuple[int, int]]:
            return [m.span() for m in re.finditer(rf"\b{re.escape(phrase)}\b", low)
                    if not any(a <= m.start() < b for a, b in taken)]

        for size in (3, 2):  # whole names first ("imran khan"), then single words
            for f in self.instance.faculty:
                if len(parts[f.id]) >= size and (spans := at(" ".join(parts[f.id][-size:]))):
                    found.append(f.id)
                    taken.extend(spans)
        surnames = {f.id: parts[f.id][-1] for f in self.instance.faculty if parts[f.id]}
        for fid, surname in surnames.items():
            if fid not in found and (spans := at(surname)):
                found.append(fid)
                taken.extend(spans)
        firsts: dict[str, list[Faculty]] = {}
        for f in self.instance.faculty:
            if len(parts[f.id]) > 1 and f.id not in found:
                firsts.setdefault(parts[f.id][0], []).append(f)
        for first, people in firsts.items():
            if at(first):
                if len(people) == 1:
                    found.append(people[0].id)
                else:
                    unclear = f"Which one do you mean: {_listed([p.name for p in people])}?"
        for m in TEACHER.finditer(text):  # "Dr. Smith": a title with a name the department does not have
            if not any(a <= m.start(1) < b for a, b in taken) and not self.directory.find(m.group(1)):
                unclear = unclear or f"I don't know a teacher called {m.group(1).title()}."
        return list(dict.fromkeys(found)), unclear

    def _equipment(self, text: str) -> list[str]:
        """The rooms' equipment the question names: "a lab with GPUs" -> ["gpu"]."""
        known = {e for r in self.instance.rooms for e in r.equipment}
        return sorted(e for e in known if re.search(rf"\b{re.escape(e)}s?\b", text, re.IGNORECASE))

    def _seats(self, text: str) -> int | None:
        """The seats asked for: "a room for 70 students" -> 70."""
        m = SEATS.search(text)
        return int(m.group(1) or m.group(2)) if m else None

    def _add_me(self, teachers: list[str], sections: list[str]) -> None:
        """The sender: a teacher by their ID, a class representative ("ST-G-04") by their section."""
        if self.sender.startswith("ST-"):
            if self.sender[3:] not in sections:
                sections.insert(0, self.sender[3:])
        elif self.sender not in teachers:
            teachers.insert(0, self.sender)


class Answerer:
    """Runs a ToolCall on the week's timetable and writes the reply from what it returns."""

    def __init__(self, instance: Instance, sender: str, records=None, sees_all: bool = False) -> None:
        """``records``: the app's data the timetable doesn't hold (``more_lookups.Records``: requests, rules,
        bookings, other weeks); the lookups that need it say so without it. ``sees_all``: the asker may see every
        request (the timetable office)."""
        self.instance = instance
        self.sender = sender
        self.records = records
        self.sees_all = sees_all

    def answer(self, call: ToolCall, week: WeekTimetable) -> str:
        if not call.confident:
            return call.note
        if call.tool == "multi":  # several lookups in one question (v3): each answered, in order
            return " ".join(self.answer(p, self._week_of(p, week)) for p in call.args["parts"])
        if len(call.args.get("weeks", [])) > 1:  # several weeks: each, and for free hours the hours free in all
            return self._weeks(call, week)
        from . import more_lookups

        if call.tool in more_lookups.MORE_TOOLS:
            return more_lookups.answer(self, call, week)
        if call.tool == "free_hours" and call.args.get("room_for"):
            return self._room_for(call, week)
        if call.tool == "free_hours":
            if call.args.get("kind") or call.args.get("min_capacity") or call.args.get("equipment"):
                return self._free_with_room(call, week)
            return self._free(call, week)
        if call.tool == "classes_of":
            if "after" in call.args:
                return self._next(call, week)
            return self._classes(call, week)
        if call.tool == "free_rooms":
            return self._free_rooms(call, week)
        if call.tool == "who_is_free":
            return self._who_is_free(call, week)
        if call.tool == "describe":
            return self._describe(call)
        if call.tool == "changes":
            return self._changes(call, week)
        return "I can answer questions like: my free hours this week, when Dr. Iyer teaches Machine Learning, " \
               "or whether Lab 2 is free on Friday."

    def _week_of(self, call: ToolCall, week: WeekTimetable) -> WeekTimetable:
        """The timetable of ``call``'s week: ``week`` when it is that one, else read from the records' store."""
        store = getattr(self.records, "store", None)
        if call.week == week.week or store is None:
            return week
        return WeekTimetable(self.instance, store, call.week)

    def _weeks(self, call: ToolCall, week: WeekTimetable) -> str:
        """ "Am I free every Tuesday this month?": the lookup in each week asked, and, for free hours, the hours free
        in every one of them."""
        weeks = sorted(call.args["weeks"])
        one = {k: v for k, v in call.args.items() if k != "weeks"}
        parts, common = [], None
        for w in weeks:
            c = ToolCall(call.tool, one, w)
            wk = self._week_of(c, week)
            parts.append(f"Week {w}: {self.answer(c, wk)}")
            if call.tool == "free_hours" and not one.get("slots"):
                hours = set(self._in_part(wk.free_hours(one.get("teachers", ()), one.get("sections", ()),
                                                        one.get("rooms", ()), one.get("days"),
                                                        length=one.get("length", 1)), one))
                common = hours if common is None else common & hours
        if common is not None:
            days = self.instance.calendar.days
            every = sorted(common, key=lambda h: (days.index(h[0]), h[1]))
            parts.insert(0, (f"Free in every one of weeks {weeks[0]}–{weeks[-1]}: {_spans(every)}." if every else
                             f"There is no hour free in every one of weeks {weeks[0]}–{weeks[-1]}."))
        return " ".join(parts)

    def _room_for(self, call: ToolCall, week: WeekTimetable) -> str:
        """ "Can Lab 3 hold Sections 1 and 2?": the free hours as asked, then whether each room named seats the
        sections together, and, when one doesn't, rooms that do and are free then."""
        a, i = call.args, self.instance
        need = sum(i.group_by_id[g].size for g in a["room_for"])
        plain = {k: v for k, v in a.items() if k != "room_for"}
        groups = _listed([i.group_by_id[g].name for g in a["room_for"]])
        if not a.get("rooms"):  # no room named: the hours when they're free with any room big enough
            plain["min_capacity"] = max(need, plain.get("min_capacity") or 0)
            return self._free_with_room(ToolCall("free_hours", plain, call.week), week)
        said = self._free(ToolCall("free_hours", plain, call.week), week)
        small = [r for r in (i.room_by_id[x] for x in a["rooms"]) if r.capacity < need]
        if not small:
            fits = _listed([f"{r.name} ({r.capacity} seats)" for r in (i.room_by_id[x] for x in a["rooms"])])
            return f"{said} {fits} can seat {groups} together ({need} students)."
        names = _listed([f"{r.name} has {r.capacity} seats" for r in small])
        free = set(week.free_hours(a.get("teachers", ()), a.get("sections", ()), (), a.get("days")))
        others = []
        for room in i.rooms:
            if room.capacity >= need and room.id not in a["rooms"]:
                hours = [h for h in week.free_hours(rooms=[room.id], days=a.get("days")) if h in free]
                if hours := self._in_part(hours, a):
                    others.append(f"{room.name} ({room.capacity} seats), free {_spans(hours[:6])}")
        alt = (f" Rooms that can: {'; '.join(others[:3])}." if others else
               f" No room seats {need} at those hours.")
        return f"{said} But {names}, and {groups} have {need} students together, so it can't hold them at once.{alt}"

    def _name(self, who: tuple[str, str]) -> str:
        """One teacher, section or room by name ("you" for the asker): ("teacher", "F-103") -> "Dr. Iyer"."""
        kind, i = who
        if kind == "teacher":
            return "you" if i == self.sender else self.instance.faculty_by_id[i].name
        if kind == "section":
            return "your section" if self.sender == f"ST-{i}" else self.instance.group_by_id[i].name
        return self.instance.room_by_id[i].name

    def _spans(self, hours: list[tuple[str, int]]) -> str:
        return _spans(hours)

    def _free(self, call: ToolCall, week: WeekTimetable) -> str:
        a = call.args
        free = week.free_hours(a.get("teachers", ()), a.get("sections", ()), a.get("rooms", ()), a.get("days"),
                               length=a.get("length", 1))
        free = self._in_part(free, a)
        who, plural = self._who(a)
        when = f"week {call.week}"
        taught = ""  # "when is the DS teacher free?": who that is, first
        if a.get("via_course"):
            course_teachers = {s.faculty for s in self.instance.sessions if s.course == a["via_course"]}
            by = [self.instance.faculty_by_id[f].name for f in a.get("teachers", []) if f in course_teachers]
            taught = f"{self.instance.course_title(a['via_course'])} is taught by {_listed(by)}. "
        if a.get("each"):  # co-teachers: each one's free time on its own, with whoever else was asked about
            co = [t for t in a.get("teachers", []) if t in course_teachers]
            rest = [t for t in a.get("teachers", []) if t not in co]
            parts = [self._free(ToolCall("free_hours", {**a, "teachers": rest + [t], "each": False, "via_course": None},
                                         call.week), week) for t in co]
            return taught + " ".join(parts)
        if a.get("first") and free:  # only the earliest: the first hour, or the first run of ``length`` hours
            day, slot = free[0]
            run = [(day, s) for s in range(slot, slot + a.get("length", 1)) if (day, s) in free]
            both = " both" if plural and len(self._people(a)) == 2 else ""
            return taught + _cap(f"the earliest {who} {'are' if plural else 'is'}{both} free is {_spans(run)} ({when}).")
        if a.get("slots"):  # "is Lab 2 free on Thursday at 3 pm?": yes or no for those hours
            days = a.get("days") or self.instance.calendar.days
            asked = [(d, s) for d in days for s in a["slots"]]
            taken = [h for h in asked if h not in free]
            at = ", ".join(f"{FULL_DAY[d]} {hour(s)}" for d, s in asked)
            if not taken:
                return taught + f"Yes, {who} {'are' if plural else 'is'} free on {at} ({when})."  # who mid-sentence
            return taught + f"No, {who} {'are' if plural else 'is'} not free on " \
                   f"{', '.join(f'{FULL_DAY[d]} {hour(s)}' for d, s in taken)} ({when})."
        run = f" ({a['length']} hours in a row)" if a.get("length", 1) > 1 else ""
        if not free:
            on = f" on {', '.join(FULL_DAY[d] for d in a['days'])}" if a.get("days") else ""
            part = f" in the {a['part']}" if a.get("part") else ""
            return taught + _cap(f"{who} {'have' if plural else 'has'} no free hours{run}{on}{part} in {when}.")
        both = " both" if plural and len(self._people(a)) == 2 else ""
        return taught + _cap(f"{who} {'are' if plural else 'is'}{both} free{run} on {_spans(free)} ({when}).")

    def _free_with_room(self, call: ToolCall, week: WeekTimetable) -> str:
        """ "When can Section 1 and I have a 2-hour class in a lab?": per room, the hours when it and they are free."""
        a = call.args
        found = week.free_with_room(a.get("teachers", ()), a.get("sections", ()), a.get("kind"),
                                    a.get("equipment", ()), a.get("min_capacity"), a.get("days"), a.get("length", 1))
        found = [(r, h) for r, h in ((r, self._in_part(h, a)) for r, h in found) if h]
        who, plural = self._who(a)
        room = self._room_words(a)
        run = f" for {a['length']} hours in a row" if a.get("length", 1) > 1 else ""
        if not found:
            return _cap(f"there is no time{run} when {who} {'are' if plural else 'is'} free and {room} is free "
                        f"in week {call.week}.")
        return _cap(f"{who} {'are' if plural else 'is'} free{run} with {room} in week {call.week}: ") + "; ".join(
            f"{r.name} ({_spans(h)})" for r, h in found) + "."

    def _next(self, call: ToolCall, week: WeekTimetable) -> str:
        """ "When is my next class?": the first one at or after the time asked (the time now, usually)."""
        a = call.args
        found = week.next_class(tuple(a["after"]), a.get("teacher"), a.get("section"))
        who, _ = self._who({"teachers": [a["teacher"]] if a.get("teacher") else [],
                            "sections": [a["section"]] if a.get("section") else []})
        if found is None:
            return _cap(f"{who} {'have' if who == 'you' else 'has'} no more classes in week {call.week}.")
        mine = "your" if who == "you" else f"{who}'s"
        return _cap(f"{mine} next class: {self._one(found)}, week {call.week}.")

    def _who_is_free(self, call: ToolCall, week: WeekTimetable) -> str:
        """ "Who is free on Monday at 10 am?": the teachers (or sections) free then, the asker left out."""
        a = call.args
        slots = a.get("slots") or (part_slots(self.instance, a["part"]) if a.get("part") else None)
        found = [(p, h) for p, h in week.who_is_free(a.get("days"), slots, a.get("course"), a.get("sections", False))
                 if p.id != self.sender]
        what = "sections" if a.get("sections") else "teachers"
        if a.get("course"):
            what += f" of {self.instance.course_title(a['course'])}"
        on = f" on {_listed([FULL_DAY[d] for d in a['days']])}" if a.get("days") else ""
        at = f" at {' and '.join(hour(s) for s in a['slots'])}" if a.get("slots") else \
            f" in the {a['part']}" if a.get("part") else ""
        if not found:
            return _cap(f"no {what} are free{on}{at} in week {call.week}.")
        if a.get("slots"):  # one hour (or a few): only those free at every one, by name
            every = [p.name for p, h in found if len(h) == len(a["slots"]) * len(a.get("days") or self.instance.calendar.days)]
            if every:
                return _cap(f"{what} free{on}{at} (week {call.week}): {_listed(every)}.")
        one_day = FULL_DAY[a["days"][0]] + " " if len(a.get("days", [])) == 1 else ""  # said once, not per person
        return _cap(f"{what} free{on}{at} (week {call.week}): ") + "; ".join(
            f"{p.name} {_spans(h).removeprefix(one_day)}" if one_day else f"{p.name} ({_spans(h)})"
            for p, h in found) + "."

    def _describe(self, call: ToolCall) -> str:
        """Department facts: rooms and their seats, sections and their sizes, who teaches what, the calendar."""
        a, d, i = call.args, Department(self.instance), self.instance
        if a.get("room"):
            r = i.room_by_id[a["room"]]
            return f"{r.name}: {self._room_facts(r)}."
        if a.get("section"):
            g = i.group_by_id[a["section"]]
            courses = list(dict.fromkeys(i.course_title(s.course) for s in i.sessions if g.id in s.groups))
            return f"{g.name} has {g.size} students; courses: {_listed(courses)}."
        if a.get("teacher"):
            f = i.faculty_by_id[a["teacher"]]
            role = a.get("as_role", {}).get(f.id)
            who = ("You teach" if f.id == self.sender else
                   _cap(f"{role_name(role)} is {f.name}, who teaches") if role else f"{f.name} teaches")
            return (f"{who} {_listed([i.course_title(c) for c in d.courses_of(f.id)])}: "
                    f"{d.weekly_hours(f.id)} hours a week.")
        if a.get("course"):
            c = a["course"]
            kinds = Counter(s.kind.value for s in d.sessions_of(c))
            classes = _listed([f"{n} {k}{'s' if n > 1 else ''}" for k, n in kinds.items()])
            return (f"{i.course_title(c)}: taught by {_listed([f.name for f in d.teachers_of(c)])} to "
                    f"{_listed([g.name for g in d.sections_of(c)])}; each week {classes}.")
        what = a.get("what")
        if what == "rooms":
            kind = a.get("kind")
            name = {"lab": "Labs", "lecture": "Lecture rooms"}.get(kind, "Rooms")
            return f"{name}: " + "; ".join(f"{r.name} ({self._room_facts(r)})" for r in d.rooms(kind)) + "."
        if what == "sections":  # each with what it studies, as the teachers' list gives what they teach
            return "Sections: " + "; ".join(
                f"{g.name} ({g.size} students): "
                + _listed(list(dict.fromkeys(i.course_title(s.course) for s in i.sessions if g.id in s.groups)))
                for g in d.sections()) + "."
        if what == "teachers":
            return "Teachers: " + "; ".join(
                f"{f.name} ({_listed([i.course_title(c) for c in d.courses_of(f.id)]) or 'no classes'})"
                for f in d.teachers()) + "."
        if what == "courses":
            return "Courses: " + "; ".join(
                f"{t} ({_listed([f.name for f in d.teachers_of(c)])})" for c, t in i.course_titles.items()
                if d.teachers_of(c)) + "."
        if what == "calendar":
            cal = i.calendar
            return (f"Classes run {FULL_DAY[cal.days[0]]} to {FULL_DAY[cal.days[-1]]}, "
                    f"{_range(0, cal.slots_per_day)}, with lunch {_range(cal.lunch_slot, cal.lunch_slot + 1)}. "
                    f"The semester has {cal.weeks} teaching weeks; this is week {call.week}.")
        return "I can list the rooms, sections, teachers or courses, or tell you about one of them."

    def _changes(self, call: ToolCall, week: WeekTimetable) -> str:
        """ "What changed this week?": classes not held, moved, and extra; never why (P-PRIVACY)."""
        a = call.args
        found = week.changes(a.get("teacher"), a.get("section"))
        whose = "" if not (a.get("teacher") or a.get("section")) else " for " + self._who(
            {"teachers": [a["teacher"]] if a.get("teacher") else [],
             "sections": [a["section"]] if a.get("section") else [], "as_role": a.get("as_role", {})})[0]
        if kind := a.get("change"):  # "any cancelled classes next week?": yes or no, about that kind only
            mine = [c for c in found if c.kind == CHANGE_KINDS[kind]]
            others, found = len(found) - len(mine), mine
            if not found:
                none = (f"There are no extra classes{whose} in week {call.week}." if kind == "extra" else
                        f"No classes{whose} are {'cancelled' if kind == 'cancelled' else 'moved'} in week {call.week}.")
                more = (f" ({others} other change{'s' if others != 1 else ''} that week: ask what changed to see "
                        f"{'them' if others != 1 else 'it'}.)") if others else ""
                return none + more
            n = len(found)
            done = {"cancelled": "not held", "moved": "moved", "extra": "added"}[kind]
            lead = f"Yes, {n} class{'es' if n != 1 else ''}{whose} {'is' if n == 1 else 'are'} {done} in week {call.week}: "
        else:
            lead = f"Changes{whose} in week {call.week}: "
        if not found:
            return f"No changes{whose} in week {call.week}: the timetable is as planned."
        said = []
        for c in found:
            what = f"{self.instance.course_title(c.session.course)} {c.session.kind.value}"
            if c.kind == "not_held":
                said.append(f"{what} on {FULL_DAY[c.before.day]} {hour(c.before.slot)} is not held")
            elif c.kind == "moved":
                said.append(f"{what} moved from {FULL_DAY[c.before.day]} {hour(c.before.slot)} to "
                            f"{FULL_DAY[c.after.day]} {hour(c.after.slot)}")
            else:
                said.append(f"extra {what} on {FULL_DAY[c.after.day]} {hour(c.after.slot)}")
        return lead + "; ".join(said) + "."

    def _in_part(self, hours: list[tuple[str, int]], a: dict) -> list[tuple[str, int]]:
        """Only the morning's or the afternoon's hours, when the question says which."""
        if not a.get("part"):
            return hours
        keep = set(part_slots(self.instance, a["part"]))
        return [h for h in hours if h[1] in keep]

    def _room_words(self, a: dict) -> str:
        """ "a lab with gpu for 70": the room a question needs, in words."""
        words = {"lab": "a lab", "lecture": "a lecture room"}.get(a.get("kind"), "a room")
        if a.get("equipment"):
            words += f" with {_listed(a['equipment'])}"
        if a.get("min_capacity"):
            words += f" for {a['min_capacity']}"
        return words

    @staticmethod
    def _room_facts(r: Room) -> str:
        return f"{r.type.value}, {r.capacity} seats" + (f"; {', '.join(r.equipment)}" if r.equipment else "")

    def _free_rooms(self, call: ToolCall, week: WeekTimetable) -> str:
        """ "Free labs on Monday (week 14): Lab 1 9 am–1 pm; Lab 3 2–5 pm." or, for one hour, which rooms."""
        a = call.args
        found = week.free_rooms(a.get("kind"), a.get("equipment", ()), a.get("min_capacity"), a.get("days"))
        found = [(r, h) for r, h in ((r, self._in_part(h, a)) for r, h in found) if h]
        what = {"lab": "labs", "lecture": "lecture rooms"}.get(a.get("kind"), "rooms")
        if a.get("equipment"):
            what += f" with {_listed(a['equipment'])}"
        if a.get("min_capacity"):
            what += f" for {a['min_capacity']}"
        on = f" on {_listed([FULL_DAY[d] for d in a['days']])}" if a.get("days") else ""
        if a.get("slots"):  # "which labs are free on Monday at 3 pm?": the rooms free at every hour asked
            at = " and ".join(hour(s) for s in a["slots"])
            days = a.get("days") or self.instance.calendar.days
            free = [r.name for r, hours in found if all((d, s) in hours for d in days for s in a["slots"])]
            if not free:
                return _cap(f"no {what} are free{on} at {at} (week {call.week}).")
            return _cap(f"{what} free{on} at {at} (week {call.week}): {_listed(free)}.")
        if not found:
            return _cap(f"no {what} are free{on} in week {call.week}.")
        one_day = FULL_DAY[a["days"][0]] + " " if len(a.get("days", [])) == 1 else ""  # said once, not per room
        return _cap(f"free {what}{on} (week {call.week}): ") + "; ".join(
            f"{r.name} {_spans(h).removeprefix(one_day)}" for r, h in found) + "."

    def _classes(self, call: ToolCall, week: WeekTimetable) -> str:
        a = call.args
        found = week.classes_of(a.get("teacher"), a.get("section"), a.get("room"), a.get("course"))
        if a.get("days"):
            found = [c for c in found if c.placement.day in a["days"]]
        who, _ = self._who({"teachers": [a["teacher"]] if a.get("teacher") else [],
                            "sections": [a["section"]] if a.get("section") else [],
                            "rooms": [a["room"]] if a.get("room") else [], "as_role": a.get("as_role", {})})
        course = self.instance.course_title(a["course"]) if a.get("course") else ""
        when = f"week {call.week}"
        if a.get("count"):  # "how many hours do I teach this week?"
            hours = sum(c.session.duration for c in found)
            on = f" on {_listed([FULL_DAY[d] for d in a['days']])}" if a.get("days") else ""
            verb = "have" if who == "you" else "has"
            return _cap(f"{who} {verb} {len(found)}{' ' + course if course else ''} "
                        f"class{'es' if len(found) != 1 else ''} ({hours} hour{'s' if hours != 1 else ''}){on} in {when}.")
        by_teacher = not a.get("teacher")  # name each class's teacher unless the question is about one teacher
        listed = "; ".join(self._one(c, teacher=by_teacher) for c in found)
        if not found and a.get("teacher") and a.get("course"):
            # "When does Dr. Iyer teach ML?" when someone else does: say who
            others = week.classes_of(course=a["course"])
            if others:
                teachers = _listed(list(dict.fromkeys(self.instance.faculty_by_id[c.session.faculty].name for c in others)))
                lead = f"{who} {"don't" if who == "you" else "doesn't"} teach {course} in {when}. It is taught by {teachers}: "
                return _cap(lead) + "; ".join(self._one(c) for c in others) + "."
        if not found:
            return f"No{' ' + course if course else ''} classes found for {who} in {when}."
        if a.get("teacher"):
            verb = "teach" if who == "you" else "teaches"
            return _cap(f"{who} {verb}{' ' + course if course else ''} in {when}: ") + listed + "."
        if a.get("section"):
            return _cap(f"{who}'s{' ' + course if course else ''} classes in {when}: ") + listed + "."
        if a.get("room"):
            return f"{course + ' c' if course else 'C'}lasses in {who} in {when}: " + listed + "."
        if course:
            return f"{course} in {when}: " + listed + "."  # a course asked about, nobody named: everyone's classes of it
        on = f" on {_listed([FULL_DAY[d] for d in a['days']])}" if a.get("days") else ""
        return f"Classes{on} in {when}: " + listed + "."  # every teacher ("which teachers teach on Friday?")

    def _one(self, c: ScheduledClass, teacher: bool = False) -> str:
        s, p = c.session, c.placement
        room = self.instance.room_by_id[p.room].name if p.room in self.instance.room_by_id else p.room
        by = f", {self.instance.faculty_by_id[s.faculty].name}" if teacher else ""
        when = f"{FULL_DAY[p.day]} {_range(p.slot, p.slot + s.duration)}"
        return f"{self.instance.course_title(s.course)} {s.kind.value}, {when} ({room}{by})"

    def _people(self, a: dict) -> list[str]:
        return [*a.get("teachers", []), *a.get("sections", []), *a.get("rooms", [])]

    def _who(self, a: dict) -> tuple[str, bool]:
        """ "you", "you and Dr. Iyer", "Section 4", "Lab 2", "all teachers": and whether it is plural ("are",
        "have"). Every teacher, section or room of the department is said as "all teachers" (not the list)."""
        i = self.instance
        every = [w for w, have, ids in (("teachers", a.get("teachers", []), [f.id for f in i.faculty]),
                                        ("sections", a.get("sections", []), [g.id for g in i.groups]),
                                        ("rooms", a.get("rooms", []), [r.id for r in i.rooms]))
                 if len(ids) > 1 and set(have) >= set(ids)]
        if every:
            rest = {k: [] if k in every else a.get(k, []) for k in ("teachers", "sections", "rooms")}
            others, _ = self._who(rest) if any(rest.values()) else ("", False)
            return _listed([f"all {w}" for w in every] + ([others] if others else [])), True
        names, as_role, by_role = [], a.get("as_role", {}), {}  # a role named: "the HOD (Dr. Rao)"
        for t in a.get("teachers", []):
            name = "you" if t == self.sender else self.instance.faculty_by_id[t].name
            (by_role.setdefault(as_role[t], []) if t in as_role else names).append(name)
        for g in a.get("sections", []):
            name = "your section" if self.sender == f"ST-{g}" else self.instance.group_by_id[g].name
            (by_role.setdefault(as_role[g], []) if g in as_role else names).append(name)
        names += [self.instance.room_by_id[r].name for r in a.get("rooms", [])]
        roles = [f"{role_name(r)} ({_listed(people)})" for r, people in by_role.items()]
        if not names and not roles:
            return "anyone", False
        count = len(names) + sum(len(p) for p in by_role.values())
        return _listed(roles + names), count > 1 or names == ["you"]


def dates_in(text: str, today: date) -> list[date]:
    """The calendar dates a question names, in order; the year is today's unless it says another."""
    found: list[tuple[int, date]] = []
    for pattern, order in ((DATE_DM, "dm"), (DATE_MD, "md"), (DATE_NUM, "num")):
        for m in pattern.finditer(text):
            if any(start <= m.start() < start + 12 for start, _ in found):  # already read by an earlier pattern
                continue
            if order == "dm":
                day, month, year = int(m.group(1)), MONTHS[m.group(2).lower()[:3]], m.group(3)
            elif order == "md":
                day, month, year = int(m.group(2)), MONTHS[m.group(1).lower()[:3]], m.group(3)
            else:
                day, month, year = int(m.group(1)), int(m.group(2)), m.group(3)
            y = int(year) + (2000 if year and len(year) == 2 else 0) if year else today.year
            try:
                found.append((m.start(), date(y, month, day)))
            except ValueError:  # "31/02", or "2-14" that was never a date
                continue
    return [d for _, d in sorted(found)]


def week_and_day(instance: Instance, semester_start: date, d: date) -> tuple[int, str] | str:
    """The teaching week and day of a date, or why it has none (a weekend, outside the semester)."""
    weeks = instance.calendar.weeks
    week = (d - semester_start).days // 7 + 1
    if not 1 <= week <= weeks:
        return f"{d.day} {d:%B} is outside this semester (weeks 1–{weeks})."
    if d.weekday() >= len(instance.calendar.days):
        return f"There are no classes on {d:%A} {d.day} {d:%B}."
    return week, instance.calendar.days[d.weekday()]


def from_now(instance: Instance, current_week: int, now: datetime) -> tuple[int, tuple[str, int]]:
    """The week and the (day, slot) from which "my next class" counts: now, or the next teaching hour."""
    cal = instance.calendar
    week, d = current_week, now.date()
    slot = now.hour - 9 + (1 if now.minute else 0)  # slot 0 is 9 am; a class already started is not "next"
    if slot >= cal.slots_per_day:  # after the day's last hour: the next day
        d, slot = d + timedelta(days=1), 0
    if d.weekday() > 4:  # a weekend: Monday of the next week
        d, slot = d + timedelta(days=7 - d.weekday()), 0
        week += 1
    return min(week, cal.weeks), (cal.days[d.weekday()], max(slot, 0))


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _listed(items: list[str]) -> str:
    """["a"] -> "a"; ["a", "b"] -> "a and b"; ["a", "b", "c"] -> "a, b and c"."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _slot(h: int, ampm: str) -> int:
    return h % 12 + (12 if ampm.lower() == "pm" else 0) - 9


def _range(start: int, end: int) -> str:
    """Slots 5 to 6 -> "2–3 pm"; 2 to 4 -> "11 am–1 pm" (am/pm written once when both ends share it)."""
    a, b = hour(start), hour(end)
    return f"{a.split()[0]}–{b}" if a.split()[1] == b.split()[1] else f"{a}–{b}"


def _spans(free: list[tuple[str, int]]) -> str:
    """[("Mon", 2), ("Mon", 3), ("Mon", 5), ("Mon", 7)] -> "Monday 11 am–1 pm, 2–3 pm and 4–5 pm"."""
    out, by_day = [], {}
    for d, s in free:
        by_day.setdefault(d, []).append(s)
    for d, slots in by_day.items():
        runs, start = [], slots[0]
        for prev, s in zip(slots, slots[1:] + [None], strict=False):
            if s != prev + 1:
                runs.append(_range(start, prev + 1))
                start = s
        out.append(f"{FULL_DAY[d]} {_listed(runs)}")
    return "; ".join(out)
