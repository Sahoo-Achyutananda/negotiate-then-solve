"""The test set for timetable questions: templates filled from the department, scored for the regex reader,
Gemini (``ToolAgent``) and both together (ROADMAP.md, step 4).

    uv run python -m evaluation.lookup_set                          # the regex only: no API calls
    uv run python -m evaluation.lookup_set --gemini gemini-3.8-flash # also Gemini (cached: reruns are free)
    uv run python -m evaluation.lookup_set --local http://10.1.25.191:8080/v1   # also the nts-lookup-v1 adapter
    uv run python -m evaluation.lookup_set --local --department 1 --shuffle      # a department it has not seen

``data/lookup-test-templates.jsonl`` has templates, not questions: *"is {teacher.any} free on {day.any} at
{time.any}?"* with the expected lookup in the same placeholders, ``{"teachers": ["{teacher}"], ...}``. The
filler picks the values from the department (with a seed, so every run asks the same questions) and fills
both, so a label is right whatever was picked, and the set works with any department file.

The questions are asked on a copy of the department whose teachers have first names (``with_first_names``),
two of them the same, so "imran", "Imran Khan" and an ambiguous "Anil" are tested; the team's data is not
changed. These templates are the TEST set: training templates (step 5) must be different ones.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from agents.lookup import Directory
from agents.questions import REQUEST, QuestionReader, ToolCall, from_now
from core.instance import Faculty, Group, Instance, Room
from language.corpus import FULL_DAY, hour

TEMPLATES = Path(__file__).parents[2] / "data" / "lookup-test-templates.jsonl"
TEMPLATES_V2 = Path(__file__).parents[2] / "data" / "lookup-test-templates-v2.jsonl"  # the kinds v2 adds
NOW = datetime(2026, 10, 21, 10, 30)  # a Wednesday morning in week 13: "today", "tomorrow", "next class"
WEEK = 13
SEMESTER_START = date(2026, 7, 27)  # Monday of week 1 (the semester planner's start), so NOW is in week 13
FIRST_NAMES = {"Rao": "Anil", "Menon": "Priya", "Iyer": "Lakshmi", "Khan": "Imran", "Das": "Anil",
               "Pillai": "Meera", "Sharma": "Rohit", "Gupta": "Neha"}  # "Anil": two teachers, on purpose
UNKNOWN_TEACHERS = ["Dr. Bose", "Prof. Fernandes", "Dr. Mehta"]
STUDENTS = ["Rahul", "Ananya", "Vikram"]
PLACEHOLDER = re.compile(r"\{([a-z_0-9]+)(?:\.([a-z_]+))?\}")


def with_first_names(instance: Instance) -> Instance:
    """A copy of the department whose teachers have first names: "Dr. Khan" -> "Dr. Imran Khan"."""
    data = instance.model_dump()
    for f in data["faculty"]:
        title, *rest = f["name"].split()
        if rest and rest[-1] in FIRST_NAMES:
            f["name"] = f"{title} {FIRST_NAMES[rest[-1]]} {rest[-1]}"
    return Instance.model_validate(data)


def department(base: Instance, seed: int | None = None, shuffle: bool = False, shared: bool = False) -> Instance:
    """The department to ask about (ROADMAP step 6: does the model read the question, or remember the department
    it was trained on?). ``seed``: one made by ``core.generator`` with the same numbers of teachers, sections,
    courses and rooms as ``base`` (other teacher-course pairs, section sizes, rooms with labs); None: ``base``.
    ``shuffle``: names, the HOD and course titles moved around (the generator always gives F-101 the name
    "Dr. Rao" and the HOD, and Section 1 the same courses), so a model that learned who is who is caught.
    The expected lookups are filled from whichever department this is, so they are right for it."""
    if seed is not None:
        from core.generator import generate_department

        courses = len(base.course_titles) // max(len(base.groups), 1)
        made = generate_department(  # ``shared``: courses as departments have them, some with two teachers
            seed, n_faculty=len(base.faculty), n_groups=len(base.groups),
            courses_per_group=max(len(base.course_titles) // 2, 1) if shared else max(courses, 1),
            n_lecture_rooms=sum(r.type.value == "lecture" for r in base.rooms),
            n_labs=sum(r.type.value == "lab" for r in base.rooms), calendar=base.calendar,
            shared_courses=shared, co_teaching=0.4 if shared else 0.0)
        pairs = lambda i: {(s.course, s.faculty) for s in i.sessions}  # noqa: E731
        if pairs(made) == pairs(base) and [g.size for g in made.groups] == [g.size for g in base.groups]:
            # the app's own department is a generated one (seed 1 at the demo's size): that seed asks it again
            raise ValueError(f"seed {seed} makes the same department as {base.name!r}; choose another seed")
        base = made.model_copy(update={"name": f"{made.name} (like {base.name})"})
    if not shuffle:
        return base
    rng = random.Random(f"shuffle-{seed}")
    data = base.model_dump()
    names = [f["name"] for f in data["faculty"]]
    rng.shuffle(names)
    roles = [f["role"] for f in data["faculty"]]
    rng.shuffle(roles)  # the HOD lands on someone else
    for f, name, role in zip(data["faculty"], names, roles, strict=True):
        f["name"], f["role"] = name, role
    titles = list(data["course_titles"].values())
    rng.shuffle(titles)
    data["course_titles"] = dict(zip(data["course_titles"], titles, strict=True))
    data["name"] = f"{base.name}, shuffled"
    return Instance.model_validate(data)


@dataclass
class Filled:
    """One question from a template, with the lookup it should give."""

    id: str
    group: str
    question: str
    asker: str
    tool: str | None  # a tool, "unclear" (must not be guessed) or None (not a timetable question)
    args: dict = field(default_factory=dict)
    week: int = WEEK
    also: list[dict] = field(default_factory=list)  # other answers that count as right (ambiguous wordings)
    split: list[dict] = field(default_factory=list)  # "each one on its own": the lookup for each, in any order


class TemplateFiller:
    """Fills a template's placeholders with the department's teachers, sections, rooms, courses, days and
    times, in the forms people write them ("Dr. Imran Khan", "khan", "imran"; "Tuesday", "tue"; "2 pm", "2ish"),
    and its expected lookup with the matching IDs."""

    def __init__(self, instance: Instance, current_week: int = WEEK, now: datetime = NOW,
                 semester_start: date = SEMESTER_START) -> None:
        self.instance = instance
        self.current_week = current_week
        self.now = now
        self.semester_start = semester_start
        self.directory = Directory(instance)

    def fill(self, row: dict, rng: random.Random, asker: str | None = None) -> Filled | None:
        """None when the department has nothing to fill a placeholder with (no shared first name, ...).
        ``asker``: who asks (a conversation's person, nts-memory's data); by default one picked for the row."""
        self.rng, self.picked, self.said = rng, {}, {}
        rep = row.get("asker") == "rep"
        teaching = sorted({s.faculty for s in self.instance.sessions})
        self.asker = asker or (f"ST-{rng.choice(self.instance.groups).id}" if rep else rng.choice(teaching))
        try:
            question = PLACEHOLDER.sub(lambda m: self._say(m.group(1), m.group(2)), row["template"])
            week = self._week(row)
            args = {k: self._label(k, v) for k, v in row.get("args", {}).items()}
        except LookupError:
            return None
        if args.pop("next", False):
            week, after = from_now(self.instance, self.current_week, self.now)
            args["after"] = list(after)
        self._resolve(row["tool"], args)
        also = [{"tool": a["tool"], "args": {k: self._label(k, v) for k, v in a.get("args", {}).items()}}
                for a in row.get("also", [])]
        args = {k: v for k, v in args.items() if v not in (None, [], False)}
        return Filled(row["id"], row.get("group", row["tool"] or "none"), question, self.asker, row["tool"],
                      args, week, also, apart(row["tool"], args, row["split"]) if row.get("split") else [])

    def _resolve(self, tool: str | None, args: dict) -> None:
        """The words the model writes for the code to resolve (v2), turned into the IDs the lookup ends with, as
        ``ToolAgent`` resolves them in this department: ``"all"`` (every teacher, section or room), ``roles``
        (whoever holds them here) and, in ``free_hours``, a ``course`` (its teachers; with ``"all"`` teachers,
        only the course's: "both DS teachers")."""
        i = self.instance
        course = args.pop("course", None) if tool == "free_hours" else None
        every = {"teachers": [f.id for f in i.faculty], "sections": [g.id for g in i.groups],
                 "rooms": [r.id for r in i.rooms]}
        for key, ids in every.items():
            if isinstance(args.get(key), list) and "all" in args[key]:  # who_is_free's "sections" is a flag
                rest = [x for x in args[key] if x != "all"]
                args[key] = rest if key == "teachers" and course else list(dict.fromkeys(rest + ids))
        if course:
            taught = [s.faculty for s in i.sessions if s.course == course]
            args["teachers"] = list(dict.fromkeys(args.get("teachers", []) + taught))
        roles = args.pop("roles", [])
        for role in roles:
            holders = [p for p, r in self.roles_of.items() if r == role]
            args.setdefault("teachers", []).extend(p for p in holders if p in i.faculty_by_id and p not in args["teachers"])
            reps = [p[3:] for p in holders if p.startswith("ST-") and p[3:] in i.group_by_id]
            args.setdefault("sections", []).extend(g for g in reps if g not in args["sections"])
        if roles and tool in ("describe", "classes_of", "changes"):  # tools of one person: "teacher", "section"
            for many, one in (("teachers", "teacher"), ("sections", "section")):
                if len(found := args.pop(many, [])) == 1:
                    args[one] = found[0]

    @property
    def roles_of(self) -> dict[str, str]:
        """Who holds which role in this department: the sign-in directory the app (and ToolAgent) builds."""
        if not hasattr(self, "_roles_of"):
            from pipeline.ingest import Directory as People

            self._roles_of = {x.person: x.role.value for x in People.from_instance(self.instance).by_email.values()}
        return self._roles_of

    # -- what a placeholder is: picked once per question, so "{teacher}" is the same teacher everywhere ----

    def _pick(self, name: str):
        if name in self.picked:
            return self.picked[name]
        i, rng = self.instance, self.rng
        teaching = [f for f in i.faculty if any(s.faculty == f.id for s in i.sessions)]
        others = [f for f in teaching if f.id != self.asker and f not in self.picked.values()]
        lecture = [r for r in i.rooms if r.type.value == "lecture"]
        labs = [r for r in i.rooms if r.type.value == "lab"]
        taught = list(dict.fromkeys(s.course for s in i.sessions))
        base = name.rstrip("0123456789")
        if base == "teacher":
            value = rng.choice(others)
        elif base == "section":
            value = rng.choice([g for g in i.groups if g not in self.picked.values()
                                and f"ST-{g.id}" != self.asker])
        elif base == "room":
            value = rng.choice([r for r in lecture if r not in self.picked.values()])
        elif base == "lab":
            value = rng.choice([r for r in labs if r not in self.picked.values()])
        elif base == "course":
            value = rng.choice(taught)
        elif name == "my_course":
            mine = [s.course for s in i.sessions
                    if s.faculty == self.asker or any(f"ST-{g}" == self.asker for g in s.groups)]
            value = rng.choice(list(dict.fromkeys(mine)))
        elif base == "day":
            value = rng.choice([d for d in i.calendar.days if d not in self.picked.values()])
        elif base == "time":
            value = rng.choice([s for s in range(i.calendar.slots_per_day) if s != i.calendar.lunch_slot])
        elif name == "seats":
            value = rng.choice(sorted({r.capacity for r in lecture}))  # a size some room fits
        elif name == "equipment":
            value = rng.choice(sorted({e for r in labs for e in r.equipment}))
        elif name == "part":
            value = rng.choice(["morning", "afternoon"])
        elif name == "date":  # a teaching day in the semester
            week = rng.randint(1, i.calendar.weeks)
            value = self.semester_start + timedelta(weeks=week - 1, days=rng.randrange(len(i.calendar.days)))
        elif name == "today":  # the day "today" means (a template on a weekend day has nothing to fill)
            if self.now.weekday() >= len(i.calendar.days):
                raise LookupError(name)
            value = self.now.date()
        elif name == "tomorrow":  # the next teaching day: Friday's tomorrow is Monday, in the next week
            value = self.now.date() + timedelta(days=1)
            while value.weekday() >= len(i.calendar.days):
                value += timedelta(days=1)
        elif name == "weekend_date":
            value = self.semester_start + timedelta(weeks=rng.randint(0, i.calendar.weeks - 1), days=rng.choice([5, 6]))
        elif name == "outside_date":
            value = self.semester_start + timedelta(weeks=i.calendar.weeks + rng.randint(2, 8), days=rng.randrange(5))
        elif name == "unknown_teacher":
            value = rng.choice(UNKNOWN_TEACHERS)
        elif name == "student":
            value = rng.choice(STUDENTS)
        elif name == "shared_first":
            firsts: dict[str, int] = {}
            for f in i.faculty:
                if len(words := f.name.split()) == 3:
                    firsts[words[1]] = firsts.get(words[1], 0) + 1
            shared = sorted(n for n, k in firsts.items() if k > 1)
            if not shared:
                raise LookupError(name)
            value = rng.choice(shared)
        else:
            raise KeyError(f"unknown placeholder {{{name}}}")
        self.picked[name] = value
        return value

    def _course_of(self, teacher: Faculty) -> str:
        key = f"course_of_{teacher.id}"
        if key not in self.picked:
            self.picked[key] = self.rng.choice(list(dict.fromkeys(s.course for s in self.instance.sessions
                                                                  if s.faculty == teacher.id)))
        return self.picked[key]

    # -- the question's words ---------------------------------------------------------------------------------

    def _say(self, name: str, form: str | None) -> str:
        """``_text``, remembering how a teacher, section, room or course was written ("khan sir", "sec 2", "ML"), for a
        reader that copies names as written (nts-lookup-v3)."""
        text = self._text(name, form)
        if form == "n":  # "Sections 1, 2 and 3": the model writes each in full ("Section 2"), not "2"
            self.said.setdefault(name, self._pick(name).name)
        elif form != "course":  # "{teacher.course}" writes the teacher's course, not the teacher
            self.said.setdefault(name, text)
        return text

    def _text(self, name: str, form: str | None) -> str:
        if name.startswith("teacher") and form == "course":
            return self._course_text(self._course_of(self._pick(name)), None)
        value = self._pick(name)
        if isinstance(value, Faculty):
            return self._teacher_text(value, form)
        if isinstance(value, Group):
            n = value.name.split()[-1]
            return {"n": n, "any": self.rng.choice([value.name, value.name.lower(), f"sec {n}", f"section-{n}"])
                    }.get(form, value.name)
        if isinstance(value, Room):
            n = value.name.split()[-1]
            forms = [value.name, value.name.lower(), f"{value.name.split()[0]}-{n}"]
            return {"n": n, "any": self.rng.choice(forms)}.get(form, value.name)
        if name.startswith("course") or name == "my_course":
            return self._course_text(value, form)
        if name in ("today", "tomorrow"):
            return name
        if isinstance(value, date):
            n = value.day
            th = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
            forms = [f"{n} {value:%B}", f"{n}{th} {value:%b}", f"{value:%b} {n}", f"{n}/{value.month}",
                     f"{n:02d}-{value.month:02d}-{value.year}", f"{value:%A} {n} {value:%B}"]
            return self.rng.choice(forms) if form == "any" else forms[0]
        if name.startswith("day"):
            full = FULL_DAY[value]
            return self.rng.choice([full, full.lower(), value.lower(), value]) if form == "any" else full
        if name.startswith("time"):
            h = int(hour(value).split()[0])
            ampm = hour(value).split()[1]
            forms = [hour(value), f"{h}{ampm}", f"{h}:00 {ampm}", f"{h + 12 if ampm == 'pm' and h != 12 else h}:00",
                     f"{h}ish"]
            return self.rng.choice(forms) if form == "any" else hour(value)
        return str(value)

    def _teacher_text(self, f: Faculty, form: str | None) -> str:
        title, *rest = f.name.split()
        surname, first = rest[-1], (rest[0] if len(rest) > 1 else None)
        if form == "surname":
            return surname
        if form == "first":
            return first or surname
        if form != "any":
            return f.name
        forms = [f.name, f"{title} {surname}", f"{title.rstrip('.')} {surname}", surname, surname.lower()]
        if first and len([x for x in self.instance.faculty if x.name.split()[1:2] == [first]]) == 1:
            forms += [f"{first} {surname}", first, first.lower()]  # a first name only when it is theirs alone
        return self.rng.choice(forms)

    def _course_text(self, course: str, form: str | None) -> str:
        title = self.instance.course_title(course)
        short = "".join(w[0] for w in title.split() if w.lower() not in {"of", "and", "the", "in", "for"}).upper()
        if form == "short":
            return short
        if form != "any":
            return title
        forms = [title, title.lower()]
        if [c for c in self.directory.find(short) if isinstance(c, str)] == [course]:  # "ML" means only this one
            forms.append(short)
        return self.rng.choice(forms)

    def _week(self, row: dict) -> int:
        week = row.get("week", "this")  # fill_all sets it from what it put for "{when}"
        if isinstance(week, str) and (m := PLACEHOLDER.fullmatch(week)):  # "{date}", "{tomorrow}": its week
            return (self._pick(m.group(1)) - self.semester_start).days // 7 + 1
        return {"this": self.current_week, "next": self.current_week + 1}.get(week, week)

    # -- the expected lookup ------------------------------------------------------------------------------

    def _label(self, key: str, value):
        if isinstance(value, list):
            return [self._label(key, v) for v in value]
        if value == "me":
            rep = self.asker.startswith("ST-")
            return self.asker[3:] if rep and key in ("sections", "section") else self.asker
        if not (isinstance(value, str) and (m := PLACEHOLDER.fullmatch(value))):
            return value
        name, form = m.group(1), m.group(2)
        if name.startswith("teacher") and form == "course":
            return self._course_of(self._pick(name))
        picked = self._pick(name)
        if isinstance(picked, Faculty | Group | Room):
            return picked.id
        if isinstance(picked, date):  # in "days": the date's day of the week
            return self.instance.calendar.days[picked.weekday()]
        if name.startswith("time") or name in ("seats", "length"):
            return int(picked)
        return picked  # a course ID, a day ("Tue"), a part, equipment


def fill_all(instance: Instance, rows: list[dict], fills: int = 3, seed: int = 0) -> list[Filled]:
    """Every template filled ``fills`` times (fewer when it has no placeholders), the same on every run."""
    filler, out = TemplateFiller(instance), []
    for row in rows:
        seen = set()
        for k in range(fills):
            rng = random.Random(f"{seed}-{row['id']}-{k}")
            when = rng.choice([("this week", WEEK), ("next week", WEEK + 1), ("", WEEK), ("in week {n}", None)])
            n = rng.choice([w for w in range(1, instance.calendar.weeks + 1) if w not in (WEEK, WEEK + 1)])
            text, week = when[0].replace("{n}", str(n)), when[1] or n
            filled = row["template"].replace("{when}", text).replace("  ", " ").replace(" ?", "?").strip()
            row_k = {**row, "template": filled}
            if "{when}" in row["template"]:
                row_k["week"] = week
            f = filler.fill(row_k, rng)
            if f is not None and f.question not in seen:
                seen.add(f.question)
                out.append(f)
    return out


ONE = {"teachers": "teacher", "sections": "section", "rooms": "room"}


def apart(tool: str | None, args: dict, keys: list[str]) -> list[dict]:
    """A label for "each one on its own" ("when is Section 1, Section 2 and Section 3 free?", "what do Sections 1
    and 2 have on Monday?"): the arguments of the lookup for each teacher, section or room in ``keys``, the rest as
    given. Tools of one teacher, section or room (classes_of, changes, describe) name it in the singular."""
    single = tool in ("classes_of", "changes", "describe")
    rest = {k: v for k, v in args.items() if k not in keys}
    return [rest | ({ONE[k]: x} if single else {k: [x]}) for k in keys for x in args.get(k, [])]


def right(f: Filled, call: ToolCall | None, partial: bool = False) -> bool:
    """Whether a reading is the expected one: the tool, the week and the arguments (lists in any order).
    ``partial``: the label gives only some arguments (v3's tool templates: "move_options" names the course, not the
    class's ID), and those must match. A label split into one lookup each (``Filled.split``): every one of them,
    in any order, and nothing else."""
    if f.split:
        got = call.args["parts"] if call is not None and call.tool == "multi" else [call]
        if any(g is None or not g.confident or g.tool != f.tool or g.week != f.week for g in got):
            return False
        return sorted(json.dumps(_norm(g.args), sort_keys=True) for g in got) == \
            sorted(json.dumps(_norm(x), sort_keys=True) for x in f.split)
    if partial and f.tool not in (None, "unclear"):
        if call is None or not call.confident or call.tool != f.tool or call.week != f.week:
            return False
        got = _norm(call.args)
        return all(got.get(k) == v for k, v in _norm(f.args).items())
    if f.tool is None:
        return call is None
    if call is None:
        return False
    if f.tool == "unclear" and not call.confident:
        return True
    if not call.confident or call.week != f.week:
        return False
    wants = ([] if f.tool == "unclear" else [{"tool": f.tool, "args": f.args}]) + f.also
    return any(call.tool == want["tool"] and _norm(call.args) == _norm(want["args"]) for want in wants)


SAID_ONLY = {"as_role", "via_course", "each"}  # how the answer is worded, worked out from the rest: not scored


def _norm(args: dict) -> dict:
    return {k: sorted(v) if isinstance(v, list) and k != "after" else v
            for k, v in args.items() if v not in (None, [], False) and k not in SAID_ONLY}


class Unreachable(Exception):
    """The model could not be reached for a while: the run stops, to be resumed (``--resume``), rather than
    count the questions it could not ask as wrong."""


def _with_retries(read, question: str, tries: int = 10, wait: float = 30.0):
    """A dropped connection (Gemini's or the local server's) is tried again after a pause; still down after
    ``tries``, the run stops (``Unreachable``)."""
    import time

    import httpx

    from language.llm import LLMError

    for k in range(tries):
        try:
            return read(question)
        except (httpx.TransportError, ConnectionError, LLMError) as e:
            if isinstance(e, LLMError) and "not reachable" not in str(e) and "timed out" not in str(e):
                raise  # a real error from the model, not the network
            print(f"  network error ({type(e).__name__}), trying again in {wait:.0f} s ({k + 1}/{tries})", flush=True)
            if k < tries - 1:
                time.sleep(wait)
    raise Unreachable(f"the model was unreachable {tries} times in a row")


def _show(call: ToolCall | None) -> str:
    if call is None:
        return "None"
    return f"{call.tool} {call.args} w{call.week}" if call.confident else f"unclear: {call.note}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gemini", metavar="MODEL", help="also ask Gemini (ToolAgent), e.g. gemini-3.8-flash")
    ap.add_argument("--local", metavar="URL", nargs="?", const="http://localhost:8080/v1", default=None,
                    help="also ask the local model on llama-server (ToolAgent), e.g. http://10.1.25.191:8080/v1")
    ap.add_argument("--adapter", default="nts-lookup-v1", help="with --local: the LoRA adapter (its GGUF file name)")
    ap.add_argument("--fills", type=int, default=3, help="questions per template")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rpm", type=float, default=None, help="Gemini calls a minute (a paid key: e.g. 60)")
    ap.add_argument("--rpd", type=int, default=None, help="Gemini calls a day (a paid key: e.g. 5000)")
    ap.add_argument("--out", type=Path, default=Path("runs/lookup-set.jsonl"), help="every question and reading")
    ap.add_argument("--misses", action="store_true", help="print every miss")
    ap.add_argument("--department", type=int, metavar="SEED", default=None,
                    help="ask about a generated department of the same size (not the app's; ROADMAP step 6)")
    ap.add_argument("--shuffle", action="store_true", help="move names, the HOD and course titles around")
    ap.add_argument("--shared", action="store_true",
                    help="with --department: courses shared by every section, some with two teachers (run D)")
    ap.add_argument("--templates", type=Path, default=TEMPLATES,
                    help=f"the test templates (default {TEMPLATES.name}; v2's new kinds: {TEMPLATES_V2.name})")
    ap.add_argument("--names-listed", action="store_true",
                    help="with --multi: the department's names in the prompt (v3 as first built), not names as written")
    ap.add_argument("--multi", action="store_true",
                    help="read with v3's prompt and answer (several lookups, more tools): for nts-lookup-v3")
    ap.add_argument("--resume", action="store_true",
                    help="keep the questions already in --out and ask only the rest (after a dropped connection)")
    a = ap.parse_args()

    from web.app import World

    w = World(parser_mode="offline", seed_history=False, deadline=60)
    if a.shared and a.department is None:
        raise SystemExit("--shared needs --department SEED")
    instance = with_first_names(department(w.instance, a.department, a.shuffle, a.shared))
    print("department:", instance.name)
    rows = [json.loads(x) for x in a.templates.read_text(encoding="utf-8").splitlines() if x.strip()]
    partial = {r["id"] for r in rows if r.get("choice") and r["tool"] not in (None, "unclear")}  # v3's tool templates
    questions = fill_all(instance, rows, a.fills, a.seed)
    client, model = None, None
    from agents.tool_agent import ToolAgent

    if a.gemini and a.local:
        raise SystemExit("one model per run: --gemini or --local")
    if a.gemini:
        from language.llm import GeminiClient

        client, model = GeminiClient(model=a.gemini, rpm=a.rpm, rpd=a.rpd), "gemini"
    elif a.local:
        from language.local import LocalClient

        client, model = LocalClient(base_url=a.local, adapter=a.adapter, timeout=300), "local"
        print("local model:", client.model)

    readers = ["regex"] + ([model, "both"] if client else [])
    score: dict[str, dict[str, list[int]]] = {r: {} for r in readers}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, str]] = set()
    if a.resume and a.out.exists():  # the questions already asked count as they were scored
        for line in a.out.read_text(encoding="utf-8").splitlines():
            if line.strip() and all(f"{r}_right" in (row := json.loads(line)) for r in readers):
                done.add((row["id"], row["question"]))
                for r in readers:
                    score[r].setdefault(row["group"], []).append(row[f"{r}_right"])
        print(f"resuming: {len(done)} questions already asked")
    with a.out.open("a" if a.resume else "w", encoding="utf-8") as out:
        for f in questions:
            if (f.id, f.question) in done:
                continue
            got = {"regex": QuestionReader(instance, f.asker, WEEK, today=NOW.date(), now=NOW,
                                            semester_start=SEMESTER_START).read(f.question)}
            trace = []
            if client:
                agent = ToolAgent(instance, client, f.asker, WEEK, today=NOW.date(), now=NOW, multi=a.multi,
                                  names_listed=a.names_listed if a.multi else None,
                                  semester_start=SEMESTER_START)
                try:
                    got[model] = _with_retries(agent.read, f.question)
                except Unreachable as e:
                    raise SystemExit(f"\nstopped: {e}. {sum(len(v) for v in score['regex'].values())} questions are in "
                                     f"{a.out}; run the same command with --resume to go on") from None
                trace = agent.trace
                got["both"] = got["regex"] if got["regex"] is not None or REQUEST.search(f.question) else got[model]
            for r in readers:
                ok = right(f, got[r], f.id in partial)
                score[r].setdefault(f.group, []).append(ok)
                if a.misses and not ok and r != "both":
                    print(f"miss {r:6} {f.id} {f.question!r}\n       want {f.tool} {f.args} w{f.week}"
                          f"\n       got  {_show(got[r])}")
            out.write(json.dumps({"id": f.id, "group": f.group, "question": f.question, "asker": f.asker,
                                  "want": {"tool": f.tool, "args": f.args, "week": f.week},
                                  **{r: _show(got[r]) for r in readers}, "tries": trace,
                                  **{f"{r}_right": right(f, got[r], f.id in partial) for r in readers}}) + "\n")
            out.flush()  # each question kept at once, so a run cut short can be resumed

    groups = sorted({g for s in score.values() for g in s})
    print(f"\n{len(questions)} questions from {len(rows)} templates\n")
    print(f"{'group':14}" + "".join(f"{r:>12}" for r in readers))
    for g in groups + ["ALL"]:
        cells = []
        for r in readers:
            oks = [x for v in score[r].values() for x in v] if g == "ALL" else score[r].get(g, [])
            cells.append(f"{sum(oks)}/{len(oks)}")
        print(f"{g:14}" + "".join(f"{c:>12}" for c in cells))
    print(f"\nevery question and reading: {a.out}")


if __name__ == "__main__":
    main()
