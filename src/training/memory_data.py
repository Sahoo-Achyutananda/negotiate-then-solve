"""Training and test data for nts-memory (agents.memory): conversations of the app's own kinds, each ending with a
message to make stand on its own, and the answer it must give.

    uv run python -m training.memory_data                 # writes data/memory-train/{train,val,test}.jsonl

Every conversation is built from the lookup templates and the app's own replies, so what the follow-up means is
known exactly: the earlier question is a filled template, the assistant's reply is the real answer (``Answerer`` on
the app's timetable), and the full form of the follow-up is the same template filled again with something changed,
or a question about something the reply gave. Kinds (``SHARES``):

- ``change``: one thing changed ("and Friday?", "what about Dr. Rao?", "next week?", "and in the afternoon?",
  "for 80?", "with a projector?") or corrected ("no, I meant Section 3"); some in Hinglish ("aur Friday ko?");
- ``two``: two things at once ("and Friday at 3?"); ``relative``: from the earlier time or day ("an hour later?",
  "the day after?"); ``chain``: a follow-up of a follow-up (the history has the first one as typed, or as read);
- ``refer``: someone or something the reply gave ("who teaches OS?" -> "is she free at 3?"; "which labs are free?"
  -> "is the second one free at 3 too?"; "am I free Thu at 2?" -> "move my DS lecture there"); ``list``: an item of
  another list ("who is free at 10?" -> "the first one"; "my classes on Monday?" -> "move the second one to Friday");
  ``two_back``: what it refers to is two exchanges back; ``both``: a course with two teachers ("is she free?":
  which one?; "are they both free?": both);
- ``reply``: the reply to a question the assistant asked ("Who do you mean?" -> "Dr. Iyer"; "Which one do you mean:
  Dr. Anil Rao or Dr. Anil Das?" -> "Das"): the earlier question with it filled in;
- ``request``: about a request just sent: its status, cancelling it, changing it ("make it Thursday instead");
  ``whatif``: after a what-if ("ok do it" -> the change as a request); ``yes``: "yes" typed to "send this as a
  request?" (the request itself);
- ``each``: "each on its own" or "together" switched in a follow-up ("and separately?");
- ``same``: a message that stands on its own, a third right after a question on the same subject: {"same": true};
- ``ask``: leaning on something the conversation doesn't have: asked back; ``mixed``: a follow-up and a question
  of its own in one message.

Askers: teachers, class representatives ("we", "our") and the timetable office. An unrelated exchange often comes
first. Departments: the team's and renamed copies (other names, the same timetable, so the replies are real), and,
for courses with two teachers, generated departments with shared courses. The test set: the test templates, the
last two wordings of every list (never trained on), other names.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from datetime import timedelta
from pathlib import Path

from agents.memory import SYSTEM, Memory, Rewrite, Turn, _short
from agents.questions import FULL_DAY, Answerer, ToolCall, _range
from agents.views import WeekTimetable
from core.instance import Instance
from evaluation.lookup_set import (PLACEHOLDER, SEMESTER_START, TEMPLATES as TEST_V1, TEMPLATES_V2 as TEST_V2,
                                   TemplateFiller, department, with_first_names)
from language.corpus import hour

from .lookup_data import ROOT, TEMPLATES as TRAIN_V2, a_moment, renamed
from .lookup_data_v3 import TEMPLATES as TRAIN_V3, TEST_TEMPLATES as TEST_V3

OUT = ROOT / "data" / "memory-train"
OFFICE = "C-TT"  # the timetable office
TURN1_TOOLS = ("free_hours", "classes_of", "free_rooms", "who_is_free", "describe", "changes")
SHARES = {"change": 22, "two": 3, "relative": 3, "chain": 4, "refer": 11, "list": 4, "two_back": 3, "both": 2,
          "reply": 5, "request": 6, "whatif": 3, "yes": 2, "each": 2, "same": 24, "ask": 4, "mixed": 4}
FIRST_PERSON = re.compile(r"\b(I|me|my|we|our|us)\b", re.IGNORECASE)

# Wordings. In every list the last two are the test set's only.
FORMS = {
    "day": ["and {v}?", "what about {v}?", "and on {v}?", "how about {v}?", "same for {v}", "{v}?", "and {v}",
            "what about on {v} then?", "and for {v}?", "ok and {v}?"],
    "time": ["and at {v}?", "what about {v}?", "how about {v}?", "{v} then?", "and {v}?", "what about at {v}?",
             "same at {v}?", "at {v} instead?"],
    "name": ["and {v}?", "what about {v}?", "same for {v}", "how about {v}?", "and for {v}?", "{v}?",
             "and {v} too", "now {v}", "what about for {v}?", "same question for {v}"],
    "week": ["and {v}?", "what about {v}?", "same {v}?", "{v}?", "how about {v}?", "and for {v}", "ok, {v}?",
             "what about {v} then?"],
    "part": ["and in the {v}?", "what about the {v}?", "{v} then?", "and the {v}?", "how about the {v}?",
             "same in the {v}?", "and {v}?", "what about in the {v}"],
    "seats": ["for {v}?", "and for {v}?", "what about {v} seats?", "and for {v} people?", "{v} seats then?",
              "how about for {v}?", "a room for {v} instead?", "and {v}?"],
    "equipment": ["and with {v}?", "what about {v}?", "one with {v}?", "and {v}?", "with {v} instead?",
                  "how about with {v}?", "same with {v}", "any with {v}?"],
    "correct": ["no, I meant {v}", "sorry, {v}", "I meant {v}, not {old}", "not {old}, {v}", "oops, {v} I mean",
                "no wait, {v}", "{v}, not {old}", "correction: {v}"],
}
HINGLISH = {  # some people mix in Hindi: "aur Friday ko?" = "and on Friday?"
    "day": ["aur {v} ko?", "{v} ko?", "{v} ka kya?", "aur {v} ko kya?", "{v} ko bhi?", "aur {v} ka?"],
    "name": ["aur {v}?", "{v} ka kya?", "{v} ke liye?", "aur {v} ke liye?", "{v} ka bhi?", "aur {v} ka?"],
    "correct": ["nahi, {v}", "mera matlab {v}", "nahi nahi, {v}", "galti se, {v} chahiye tha", "{v} bola tha",
                "sorry, {v} chahiye"],
}
TWO = ["and {a} {b}?", "what about {a}, {b}?", "{a} {b} then?", "and {a}, {b}?", "same for {a} {b}",
       "how about {a} {b}?", "ok and {a} {b}?", "and for {a} {b}?"]
RELATIVE = {
    ("time", 1): ["an hour later?", "and an hour later?", "what about an hour later?", "one hour after that?",
                  "and 1 hr later?", "an hour after?"],
    ("time", -1): ["an hour earlier?", "and an hour before?", "what about an hour earlier?", "one hour before that?",
                   "1 hr earlier?", "an hour before then?"],
    ("day", 1): ["the day after?", "and the next day?", "what about the day after?", "the following day?",
                 "and the day after that?", "next day?"],
    ("day", -1): ["the day before?", "and the day before?", "what about the previous day?", "a day earlier?",
                  "the day before that?", "previous day?"],
}
WHEN = ["this week", "next week", ""]
PRONOUNS = [("is she", "her"), ("is he", "him"), ("are they", "them")]  # one teacher in the conversation: any of
# them means that teacher (the department has no genders, and a name doesn't tell one)
ABOUT_TEACHER = [  # who teaches X -> about that teacher: {S} "is she", {O} "her", {T} the teacher
    ("{S} free on {day} at {time}?", "is {T} free on {day} at {time}?"),
    ("when {S} free {when}?", "when is {T} free {when}?"),
    ("what else does {O} teach?", "what does {T} teach?"),
    ("when can I meet {O} {when}?", "when are {T} and I both free {when}?"),
    ("{S} around on {day}?", "is {T} free on {day}?"),
    ("what classes does {O} have on {day}?", "what classes does {T} have on {day}?"),
    ("{S} free this afternoon?", "is {T} free this afternoon?"),
    ("kya woh {day} ko free hai?", "is {T} free on {day}?"),  # Hinglish: "woh" is he or she
    ("whats {O} timetable {when}", "what is {T}'s timetable {when}"),
    ("ok when {S} free on {day}?", "when is {T} free on {day}?"),
    ("{S} free {when} at all?", "is {T} free {when} at all?"),
]
BOTH = [  # a course with two (or more) teachers: "they" is all of them
    ("are they both free on {day}?", "are {T} both free on {day}?"),
    ("when are both of them free {when}?", "when are {T} both free {when}?"),
    ("are both free at {time} on {day}?", "are {T} both free on {day} at {time}?"),
    ("when are they all free {when}?", "when are {T} all free {when}?"),
    ("and when are they free together on {day}?", "when are {T} free together on {day}?"),
    ("both of them free {day} {time}?", "are {T} both free on {day} at {time}?"),
]
ABOUT_ROOM = [  # which rooms are free -> one of them: {R} the room, {nth} "first"
    ("is the {nth} one free at {time} too?", "is {R} free on {day} at {time}?"),
    ("what's in the {nth} one on {day2}?", "what's in {R} on {day2}?"),
    ("how many seats does the {nth} one have?", "how many seats does {R} have?"),
    ("when is the {nth} one free {when}?", "when is {R} free {when}?"),
    ("is the {nth} one free all afternoon?", "is {R} free on {day} afternoon?"),
    ("the {nth} one, is it free at {time} as well?", "is {R} free on {day} at {time}?"),
    ("tell me about the {nth} one", "tell me about {R}"),
    ("whats in the {nth} room {when}", "what's in {R} {when}"),
]
ABOUT_PERSON = [  # who is free -> one of them: {T}
    ("is the {nth} one free at {time} too?", "is {T} free on {day} at {time}?"),
    ("what does the {nth} one teach?", "what does {T} teach?"),
    ("when is the {nth} one free {when}?", "when is {T} free {when}?"),
    ("the {nth} one, free on {day2}?", "is {T} free on {day2}?"),
    ("can I meet the {nth} one {when}?", "when are {T} and I both free {when}?"),
    ("the {nth} one's timetable on {day2}?", "what is {T}'s timetable on {day2}?"),
    ("what classes does the {nth} one have {when}?", "what classes does {T} have {when}?"),
    ("is the {nth} person free after that?", "is {T} free on {day} at {after}?"),  # the next hour
]
ABOUT_CLASS = [  # my classes -> one of them: {C} course, {K} kind, {D} its day, {H} its time
    ("which room is the {nth} one in?", "which room is my {C} {K} on {D} {H} in?"),
    ("move the {nth} one to {day2}", "move my {C} {K} on {D} {H} to {day2}"),
    ("can I shift the {nth} one to {day2}?", "can I shift my {C} {K} on {D} {H} to {day2}?"),
    ("cancel the {nth} one", "cancel my {C} {K} on {D} {H}"),
    ("who teaches the {nth} one?", "who teaches {C}?"),
    ("please move the {nth} class to {day2}", "please move my {C} {K} on {D} {H} to {day2}"),
    ("swap the {nth} one with {day2}", "swap my {C} {K} on {D} {H} to {day2}"),
    ("drop the {nth} one this week", "cancel my {C} {K} on {D} {H} this week"),
]
ABOUT_TIME = [  # when am I (are we) free -> that time: {D} the day, {H} the one free span the reply gave ("2–4 pm")
    ("is {section} free then?", "is {section} free on {D} {H}?"),
    ("is {teacher} free then too?", "is {teacher} free on {D} {H}?"),
    ("any lab free then?", "is any lab free on {D} {H}?"),
    ("move my {course} lecture there", "move my {course} lecture to {D} {H}"),
    ("can I shift my {course} class to then?", "can I shift my {course} class to {D} {H}?"),
    ("any classroom free at that time?", "is any classroom free on {D} {H}?"),
    ("please move my {course} class to that slot", "please move my {course} class to {D} {H}"),
    ("is {section} free at that time?", "is {section} free on {D} {H}?"),
]
ABOUT_TIME_REP = [  # the same, asked by a class representative
    ("is {teacher} free then?", "is {teacher} free on {D} {H}?"),
    ("any lab free then?", "is any lab free on {D} {H}?"),
    ("can our {course} class move there?", "can our {course} class move to {D} {H}?"),
    ("is {teacher} also free at that time?", "is {teacher} free on {D} {H}?"),
    ("any classroom free then?", "is any classroom free on {D} {H}?"),
    ("can we have an extra {course} class then?", "can we have an extra {course} class on {D} {H}?"),
]
ABOUT_REQUEST = [  # a request just sent -> it, by its number {R}
    ("has it been approved?", "has {R} been approved?"),
    ("what's the status of that?", "what's the status of {R}?"),
    ("any update on it?", "any update on {R}?"),
    ("did that go through?", "did {R} go through?"),
    ("approve hua?", "has {R} been approved?"),
    ("is it done yet?", "is {R} done yet?"),
    ("what happened to that request?", "what happened to {R}?"),
    ("was it accepted?", "was {R} accepted?"),
    ("any news on my request?", "any news on {R}?"),
]
CANCEL_REQUEST = [
    ("cancel it", "cancel {R}"),
    ("withdraw that request", "withdraw {R}"),
    ("actually cancel that", "cancel {R}"),
    ("cancel kar do", "cancel {R}"),
    ("never mind, cancel it", "cancel {R}"),
    ("please drop that request", "drop {R}"),
    ("take it back", "withdraw {R}"),
]
CHANGE_REQUEST = [  # {day}: another day
    ("make it {day} instead", "change {R} to {day}"),
    ("actually make that {day}", "change {R} to {day}"),
    ("can you change it to {day}?", "can you change {R} to {day}?"),
    ("{day} instead please", "change {R} to {day} please"),
    ("change it to {day}", "change {R} to {day}"),
    ("I'd rather have {day} for that", "change {R} to {day}"),
]
DO_IT = ["ok do it", "send that", "go ahead, request it", "yes make that request", "ok send it as a request",
         "fine, go ahead", "do it then", "please request that"]
YES = ["yes", "yes please", "send it", "yes send it", "ok go ahead", "yep, send", "sure", "yes, request it"]
SEPARATELY = ["and separately?", "each on their own?", "what about each one separately?", "and individually?",
              "separately please", "each of them?"]
TOGETHER = ["and together?", "when are they all free together?", "a time they're all free?", "common free time?",
            "together then?", "and all of them at once?"]
VAGUE = {  # a question with a name left vague, and the question back
    "teacher": (["her", "him", "that teacher", "that teacher"], "Who do you mean?"),
    "section": (["that section"], "Which section do you mean?"),
    "room": (["that room"], "Which room do you mean?"),
    "lab": (["that lab"], "Which lab do you mean?"),
    "day": (["that day"], "Which day do you mean?"),
}
REPLY = ["{v}", "I meant {v}", "{v} sorry", "oh, {v}", "{v} please", "sorry, {v}", "it's {v}", "{v}!"]
ASK = [  # leaning on what the conversation lacks: (follow-up, the question back, what it lacks)
    ("is that room free at 3?", "Which room do you mean?", "room"),
    ("is the second one free then?", "Which one do you mean?", "list"),
    ("is she free on Friday?", "Who do you mean?", "teacher"),
    ("what about him?", "Who do you mean?", "teacher"),
    ("has it been approved?", "Which request do you mean?", "request"),
    ("move my class there", "Where do you mean: which day and time?", "time"),
    ("and that lab?", "Which lab do you mean?", "room"),
    ("is he free then?", "Who do you mean?", "teacher"),
]
JOIN = ["{a} also {b}", "{a} and {b}", "{a} {b}", "{a}; {b}", "{a} also, {b}"]
NTH = ["first", "second", "third"]


class Skip(Exception):
    """This conversation can't be made (nothing to fill, nothing listed): another is made instead."""


def _forms(forms: list, test: bool) -> list:
    return forms[-2:] if test else forms[:-2]


def _kind_of(name: str) -> str | None:
    """What a placeholder is, for a follow-up that changes it."""
    base = name.rstrip("0123456789")
    if base in ("teacher", "section", "room", "lab", "course"):
        return "name"
    return {"day": "day", "time": "time", "part": "part", "seats": "seats", "equipment": "equipment"}.get(base)


def _tidy(text: str) -> str:
    return re.sub(r"\s+", " ", text).replace(" ?", "?").strip()


def _short_title(title: str) -> str:
    return "".join(w[0] for w in title.split() if w.lower() not in {"of", "and", "the", "in", "for"}).upper()


def _listed(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


class Refiller(TemplateFiller):
    """A template filled again with the earlier question's picks and wordings, but the placeholders in ``change``
    given other values (``force``: these ones)."""

    def __init__(self, *a, picks: dict, said: dict, change=(), force: dict | None = None, **k) -> None:
        super().__init__(*a, **k)
        self.fixed, self.fixed_said, self.change, self.force = picks, said, set(change), force or {}

    def _pick(self, name: str):
        if name in self.picked:
            return self.picked[name]
        if name in self.force:
            self.picked[name] = self.force[name]
            return self.force[name]
        if name in self.fixed and name not in self.change:
            self.picked[name] = self.fixed[name]
            return self.fixed[name]
        for _ in range(12):  # another value than before, and not one the question already has
            value = super()._pick(name)
            if name not in self.change or value not in self.fixed.values():
                return value
            del self.picked[name]
        raise LookupError(name)

    def _say(self, name: str, form: str | None) -> str:
        if name in self.fixed_said and name not in self.change and name not in self.force and form == "any":
            self.said.setdefault(name, self.fixed_said[name])  # the words picked before, again
            return self.fixed_said[name]
        return super()._say(name, form)  # a fixed form ("{section.n}": "2") is the same words again anyway


class Builder:
    """Conversations in the given departments, from the given templates (``pool``: questions; ``requests``);
    ``shared``: departments with courses of two teachers (only their facts are used, not a timetable)."""

    def __init__(self, store, departments: list[tuple[str, Instance]], shared: list[tuple[str, Instance]],
                 pool: list[dict], requests: list[dict], test: bool, seed: int) -> None:
        self.store, self.departments, self.shared = store, departments, shared
        self.pool, self.requests, self.test, self.seed = pool, requests, test, seed
        self.skipped: Counter = Counter()

    def row(self, k: int, kind: str) -> dict | None:
        self.r = r = random.Random(f"memory-{self.seed}-{self.test}-{kind}-{k}")
        self.dept_name, self.dept = self.departments[k % len(self.departments)]
        now, self.week = a_moment(r, self.dept.calendar.weeks)
        self.now = now.replace(hour=r.randint(9, 17))
        try:
            asker, history, message, answer, distractor = getattr(self, f"_{kind}")()
            if distractor:
                history = self._exchange(asker) + history
        except (Skip, LookupError) as e:
            self.skipped[f"{kind}: {type(e).__name__}"] += 1
            return None
        at = self.now - timedelta(minutes=r.randint(1, 5) + 3 * len(history))
        for t in history:
            t.at, at = at, at + timedelta(minutes=r.randint(0, 2))
        prompt = Memory(self.dept, None, asker, self.week, now=self.now).prompt(message, history)
        return {"id": f"{kind}-{k}", "task": "memory", "kind": kind, "department": self.dept_name,
                "meta": {"asker": asker, "now": self.now.isoformat(), "week": self.week, "message": message,
                         "history": [[t.who, t.text, t.at.isoformat()] for t in history],
                         "answer": answer.model_dump(exclude_defaults=True)},
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt},
                             {"role": "assistant", "content": answer.model_dump_json(exclude_defaults=True)}]}

    # -- filling and answering ------------------------------------------------------------------------------

    def _filler(self, cls=TemplateFiller, **k) -> TemplateFiller:
        return cls(self.dept, current_week=self.week, now=self.now, semester_start=SEMESTER_START, **k)

    def _fill(self, row: dict, when: str, asker: str | None = None, filler: TemplateFiller | None = None):
        filler = filler or self._filler()
        text = _tidy(row["template"].replace("{when}", when))
        f = filler.fill({**row, "template": text}, self.r, asker=asker)
        if f is None:
            raise Skip(row["id"])
        return filler, f

    def _answer(self, f) -> str:
        """The reply the app gives to the filled question ``f``."""
        try:
            return Answerer(self.dept, f.asker).answer(ToolCall(f.tool, dict(f.args), f.week),
                                                       WeekTimetable(self.dept, self.store, f.week))
        except Exception as e:  # noqa: BLE001 - a template the answerer can't take without the app's records
            raise Skip(f"{f.id}: {type(e).__name__}") from e

    def _say_answer(self, asker: str, tool: str, args: dict) -> str:
        return Answerer(self.dept, asker).answer(ToolCall(tool, args, self.week), WeekTimetable(self.dept, self.store, self.week))

    def _rows_for(self, asker: str | None, rows: list[dict] | None = None) -> list[dict]:
        rows = rows or self.pool
        if asker == OFFICE:  # the timetable office asks about others: no "I", "my", "we"
            return [x for x in rows if x.get("asker") != "rep" and not FIRST_PERSON.search(x["template"])
                    and "me" not in json.dumps(x.get("args", {}))]
        return rows if asker is None else [x for x in rows if (x.get("asker") == "rep") == asker.startswith("ST-")]

    def _exchange(self, asker: str | None = None, rows: list[dict] | None = None) -> list[Turn]:
        """A question (by ``asker``, else by whoever the template's filling picks) and the app's reply."""
        _, f = self._fill(self.r.choice(self._rows_for(asker, rows)), self.r.choice(WHEN), asker)
        self.asker = f.asker
        return [Turn("you", f.question), Turn("assistant", self._answer(f))]

    def _teaching(self) -> str:
        return self.r.choice(sorted({s.faculty for s in self.dept.sessions}))

    def _rep(self) -> str:
        return f"ST-{self.r.choice(self.dept.groups).id}"

    def _day(self, day: str) -> str:
        full = FULL_DAY[day]
        return self.r.choice([full, full, full.lower(), day.lower()])

    def _slot(self, but: tuple = ()) -> int:
        cal = self.dept.calendar
        return self.r.choice([s for s in range(cal.slots_per_day) if s != cal.lunch_slot and s not in but])

    def _office_or(self, p: float = 0.12) -> str | None:
        """Sometimes the timetable office asks (None: whoever the template's filling picks)."""
        return OFFICE if self.r.random() < p else None

    # -- changes ----------------------------------------------------------------------------------------------

    def _changeable(self, row: dict) -> list[str]:
        """The placeholders a follow-up can change: alone of their kind ("what about Section 1?" after three
        sections doesn't say which one it replaces), and not a teacher said with their course ("Dr Gupta's DS
        class": another teacher, another course)."""
        names = list(dict.fromkeys(m.group(1) for m in PLACEHOLDER.finditer(row["template"]) if _kind_of(m.group(1))))
        bases = Counter(n.rstrip("0123456789") for n in names)
        return [n for n in names if bases[n.rstrip("0123456789")] == 1 and "{" + n + ".course}" not in row["template"]]

    def _first(self, asker: str | None = None, need: int = 1):
        """An earlier question that a follow-up can change (``need`` things at least, the week counted)."""
        r = self.r
        rows = [x for x in self._rows_for(asker) if len(self._changeable(x)) + ("{when}" in x["template"]) >= need]
        row = r.choice(rows)
        when = r.choice(WHEN)
        filler, f1 = self._fill(row, when, asker)
        return row, when, filler, f1

    def _refill(self, row, when, filler, f1, change: set, force: dict | None = None):
        return self._fill(row, when, f1.asker, self._filler(Refiller, picks=dict(filler.picked), said=dict(filler.said),
                                                           change=change, force=force))

    def _new_week(self, when: str) -> str:
        options = [w for w in ("this week", "next week", f"in week {self.week + 2}") if w != (when or "this week")]
        return self.r.choice([w for w in options if not w.startswith("in week") or self.week + 2 <= self.dept.calendar.weeks])

    def _one_change(self, row, when, filler, f1):
        """The question with one thing changed: (the full question, the new value as said, the old one, its kind)."""
        change = self.r.choice(self._changeable(row) + (["{when}"] if "{when}" in row["template"] else []))
        if change == "{when}":
            new = self._new_week(when)
            _, f2 = self._refill(row, new, filler, f1, set())
            return f2, new.removeprefix("in "), when or "this week", "week"
        refill, f2 = self._refill(row, when, filler, f1, {change})
        return f2, refill.said.get(change, ""), filler.said.get(change, ""), _kind_of(change)

    def _change(self, correct: bool | None = None):
        r = self.r
        row, when, filler, f1 = self._first(self._office_or())
        f2, v, old, kind = self._one_change(row, when, filler, f1)
        if not v or f2.question == f1.question:
            raise Skip("nothing changed")
        correct = (r.random() < 0.25) if correct is None else correct
        forms = FORMS["correct" if correct and old else kind]
        if r.random() < 0.08 and (key := "correct" if correct and old else kind) in HINGLISH:
            forms = HINGLISH[key]
        follow = r.choice(_forms(forms, self.test)).format(v=v, old=old)
        history = [Turn("you", f1.question), Turn("assistant", self._answer(f1))]
        self.full = f2.question  # the follow-up in full, for "same" and "mixed"
        return f1.asker, history, follow, Rewrite(message=f2.question), r.random() < 0.4

    def _two(self):
        r = self.r
        row, when, filler, f1 = self._first(need=2)
        options = self._changeable(row) + (["{when}"] if "{when}" in row["template"] else [])
        a, b = sorted(r.sample(options, 2), key=lambda n: row["template"].find(n if n == "{when}" else "{" + n))
        new = self._new_week(when) if "{when}" in (a, b) else when
        refill, f2 = self._refill(row, new, filler, f1, {a, b} - {"{when}"})
        said = {n: (new.removeprefix("in ") if n == "{when}" else refill.said.get(n, "")) for n in (a, b)}
        if not all(said.values()) or f2.question == f1.question:
            raise Skip("nothing changed")
        follow = r.choice(_forms(TWO, self.test)).format(a=said[a], b=said[b])
        return f1.asker, [Turn("you", f1.question), Turn("assistant", self._answer(f1))], follow, \
            Rewrite(message=f2.question), r.random() < 0.3

    def _relative(self):
        r = self.r
        cal = self.dept.calendar
        rows = [x for x in self.pool if {n.rstrip("0123456789") for n in self._changeable(x)} & {"time", "day"}]
        row, when = r.choice(rows), r.choice(WHEN)
        filler, f1 = self._fill(row, when)
        name = r.choice([n for n in self._changeable(row) if n.rstrip("0123456789") in ("time", "day")])
        kind, step = name.rstrip("0123456789"), r.choice([1, -1])
        old = filler.picked[name]
        if kind == "time":
            new = old + step
            if not 0 <= new < cal.slots_per_day or new == cal.lunch_slot:
                raise Skip("no such hour")
        else:
            at = cal.days.index(old) + step
            if not 0 <= at < len(cal.days):
                raise Skip("no such day")
            new = cal.days[at]
        _, f2 = self._refill(row, when, filler, f1, set(), force={name: new})
        follow = r.choice(_forms(RELATIVE[(kind, step)], self.test))
        return f1.asker, [Turn("you", f1.question), Turn("assistant", self._answer(f1))], follow, \
            Rewrite(message=f2.question), False

    def _chain(self):
        """A follow-up of a follow-up: the history has the first follow-up as typed (as an app without memory
        stores it) or as it was read (as this app stores it, ``read_as``)."""
        r = self.r
        row, when, filler, f1 = self._first(need=1)
        if not (names := self._changeable(row)):
            raise Skip("nothing to change twice")
        name = r.choice(names)
        kind = _kind_of(name)
        refill2, f2 = self._refill(row, when, filler, f1, {name})
        for _ in range(6):  # a third value: neither the first question's nor the first follow-up's
            refill3, f3 = self._refill(row, when, filler, f1, {name})
            if refill3.picked[name] != refill2.picked[name]:
                break
        else:
            raise Skip("no third value")
        old, v1, v2 = filler.said.get(name, ""), refill2.said.get(name, ""), refill3.said.get(name, "")
        if not (old and v1 and v2) or len({f1.question, f2.question, f3.question}) < 3:
            raise Skip("nothing changed")
        first = r.choice(_forms(FORMS[kind], self.test)).format(v=v1, old=old)
        second = r.choice(_forms(FORMS[kind], self.test)).format(v=v2, old=v1)
        typed = r.random() < 0.5
        history = [Turn("you", f1.question), Turn("assistant", self._answer(f1)),
                   Turn("you", first if typed else f2.question), Turn("assistant", self._answer(f2))]
        return f1.asker, history, second, Rewrite(message=f3.question), False

    # -- what the reply gave --------------------------------------------------------------------------------

    def _refer(self):
        return getattr(self, self.r.choice(["_teacher", "_teacher", "_room", "_time"]))()

    def _teacher(self, asker: str | None = None, pronouns: list = PRONOUNS):
        r, d = self.r, self.dept
        rep = asker.startswith("ST-") if asker else r.random() < 0.3
        asker = asker or (self._rep() if rep else self._teaching())
        alone = [c for c in dict.fromkeys(s.course for s in d.sessions)
                 if len({s.faculty for s in d.sessions if s.course == c}) == 1]
        course = r.choice(alone)
        teacher = d.faculty_by_id[next(s.faculty for s in d.sessions if s.course == course)]
        said = r.choice([d.course_title(course), d.course_title(course).lower(), _short_title(d.course_title(course))])
        q = r.choice([f"who teaches {said}?", f"who takes {said}?", f"which teacher has {said}?",
                      f"who is the {said} teacher?"] + ([f"who teaches our {said}?"] if rep else []))
        answer = Answerer(d, asker).answer(ToolCall("describe", {"course": course}, self.week), None)
        s, o = r.choice(pronouns)
        fill = {"S": s, "O": o, "T": teacher.name, "day": self._day(r.choice(d.calendar.days)),
                "time": hour(self._slot()), "when": r.choice(WHEN)}
        forms = [x for x in _forms(ABOUT_TEACHER, self.test) if not (rep and "I " in x[1])]
        follow, full = r.choice(forms)
        return asker, [Turn("you", q), Turn("assistant", answer)], _tidy(follow.format(**fill)), \
            Rewrite(message=_tidy(full.format(**fill))), False

    def _room(self, asker: str | None = None):
        r, d = self.r, self.dept
        asker = asker or self._teaching()
        kind, day, slot = r.choice(["lab", "lecture"]), r.choice(d.calendar.days), self._slot()
        what = ("labs", "lab") if kind == "lab" else ("classrooms", "classroom")
        q = r.choice([f"which {what[0]} are free on {self._day(day)} at {hour(slot)}?",
                      f"any {what[1]} free {self._day(day)} {hour(slot)}?"])
        answer = _short(self._say_answer(asker, "free_rooms", {"kind": kind, "days": [day], "slots": [slot]}))
        listed = [name for _, name in sorted((answer.find(x.name), x.name) for x in d.rooms if x.name in answer)]
        if len(listed) < 2:
            raise Skip("fewer than two rooms free")
        n = r.randrange(min(3, len(listed)))
        fill = {"nth": NTH[n], "R": listed[n], "time": hour(self._slot(but=(slot,))), "day": self._day(day),
                "when": r.choice(WHEN), "day2": self._day(r.choice(d.calendar.days))}
        follow, full = r.choice(_forms(ABOUT_ROOM, self.test))
        return asker, [Turn("you", q), Turn("assistant", answer)], _tidy(follow.format(**fill)), \
            Rewrite(message=_tidy(full.format(**fill))), False

    def _time(self):
        r, d = self.r, self.dept
        rep = r.random() < 0.3
        asker, day = (self._rep() if rep else self._teaching()), r.choice(d.calendar.days)
        who = {"sections": [asker[3:]]} if rep else {"teachers": [asker]}
        week = WeekTimetable(d, self.store, self.week)
        free = [s for _, s in week.free_hours(who.get("teachers", []), who.get("sections", []), [], [day])]
        if not free:
            raise Skip("not free that day")
        runs: list[list[int]] = []  # the free hours as spans of hours in a row, as the reply says them
        for s in sorted(free):
            if runs and s == runs[-1][-1] + 1:
                runs[-1].append(s)
            else:
                runs.append([s])
        spans = [_range(x[0], x[-1] + 1) for x in runs]
        said, i_am = self._day(day), ("are we", "we", "our") if rep else ("am I", "I'm", "my")
        args = {**who, "days": [day]}
        # free hours in a day are mostly several spans ("then": which one? asked back); a narrower question gives
        # one time, which "then" / "there" means: one hour asked about, or the earliest
        how = r.choice(["all", "hour", "hour", "first"])
        if how == "hour":
            slot = r.choice(free)
            q = r.choice([f"{i_am[0]} free on {said} at {hour(slot)}?", f"{i_am[0]} free {said} {hour(slot)}?"])
            args["slots"], spans = [slot], [hour(slot)]
        elif how == "first":
            q = r.choice([f"when's the earliest {i_am[1]} free on {said}?", f"{i_am[2]} first free hour on {said}?"])
            args["first"], spans = True, [_range(free[0], free[0] + 1)]
        else:
            q = r.choice([f"when {i_am[0]} free on {said}?", f"{i_am[2]} free slots on {said}?"])
        answer = self._say_answer(asker, "free_hours", args)
        mine = [s.course for s in d.sessions if (s.faculty == asker if not rep else asker[3:] in s.groups)]
        title = d.course_title(r.choice(list(dict.fromkeys(mine))))
        fill = {"section": r.choice(d.groups).name, "teacher": r.choice([f for f in d.faculty if f.id != asker]).name,
                "course": r.choice([title, _short_title(title)]), "D": said, "H": spans[0]}
        follow, full = r.choice(_forms(ABOUT_TIME_REP if rep else ABOUT_TIME, self.test))
        history = [Turn("you", q), Turn("assistant", answer)]
        if len(spans) > 1:  # "then" after several free spans: which one? asked back, not guessed
            return asker, history, follow.format(**fill), \
                Rewrite(ask=f"Which time do you mean: {', '.join(spans[:-1])} or {spans[-1]}?"), False
        return asker, history, follow.format(**fill), Rewrite(message=full.format(**fill)), False

    def _list(self):
        return self._people() if self.r.random() < 0.5 else self._classes()

    def _people(self):
        r, d = self.r, self.dept
        asker, day, slot = self._teaching(), r.choice(d.calendar.days), self._slot()
        q = r.choice([f"who is free on {self._day(day)} at {hour(slot)}?", f"which teachers are free {self._day(day)} {hour(slot)}?"])
        answer = _short(self._say_answer(asker, "who_is_free", {"days": [day], "slots": [slot]}))
        named = [f.name for f in d.faculty if re.search(rf"\b{re.escape(f.name)}\b", answer)]
        listed = sorted(named, key=answer.find)
        if len(listed) < 2:
            raise Skip("fewer than two free")
        n = r.randrange(min(3, len(listed)))
        after = slot + 1 + (slot + 1 == d.calendar.lunch_slot)  # "after that": the next teaching hour
        fill = {"nth": NTH[n], "T": listed[n], "time": hour(self._slot(but=(slot,))), "day": self._day(day),
                "when": r.choice(WHEN), "day2": self._day(r.choice(d.calendar.days)), "after": hour(after)}
        follow, full = r.choice(_forms([x for x in ABOUT_PERSON if after < d.calendar.slots_per_day
                                        or "{after}" not in x[1]], self.test))
        return asker, [Turn("you", q), Turn("assistant", answer)], _tidy(follow.format(**fill)), \
            Rewrite(message=_tidy(full.format(**fill))), False

    def _classes(self):
        r, d = self.r, self.dept
        asker = self._teaching()
        days = r.sample(d.calendar.days, r.choice([1, 2]))
        q = r.choice([f"what are my classes on {' and '.join(self._day(x) for x in days)}?",
                      f"what do I teach on {' and '.join(self._day(x) for x in days)}?"])
        answer = _short(self._say_answer(asker, "classes_of", {"teacher": asker, "days": days}))
        found = re.findall(r"([A-Z][\w&' ]+?) (lecture|tutorial|practical), (\w+day) ([\d:–\- ]+(?:am|pm))", answer)
        if len(found) < 2:
            raise Skip("fewer than two classes")
        n = r.randrange(min(3, len(found)))
        course, kind, day, when = found[n]
        other = self._day(r.choice([x for x in d.calendar.days if FULL_DAY[x] != day]))
        fill = {"nth": NTH[n], "C": r.choice([course, _short_title(course)]), "K": kind, "D": day, "H": when,
                "day2": other}
        follow, full = r.choice(_forms(ABOUT_CLASS, self.test))
        return asker, [Turn("you", q), Turn("assistant", answer)], _tidy(follow.format(**fill)), \
            Rewrite(message=_tidy(full.format(**fill))), False

    def _two_back(self):
        """What the follow-up means is two exchanges back: a teacher before a list of rooms, or rooms before a
        teacher (neither reply names the other kind)."""
        asker = self._teaching()
        # "they" with a list of rooms in the chat could be the rooms: here only "she" or "he" means the teacher
        teacher, rooms = self._teacher(asker, PRONOUNS[:2]), self._room(asker)
        first, last = (teacher, rooms) if self.r.random() < 0.5 else (rooms, teacher)
        return asker, first[1] + last[1], first[2], first[3], False

    def _both(self):
        """A course with two (or more) teachers: "she" can't be told (asked back); "they" is all of them."""
        r = self.r
        self.dept_name, self.dept = r.choice(self.shared)
        d = self.dept
        asker = self._teaching()
        shared = [(c, sorted({s.faculty for s in d.sessions if s.course == c}))
                  for c in dict.fromkeys(s.course for s in d.sessions)]
        shared = [(c, f) for c, f in shared if len(f) > 1]
        if not shared:
            raise Skip("no course with two teachers")
        course, ids = r.choice(shared)
        names = [d.faculty_by_id[x].name for x in ids]
        title = d.course_title(course)
        q = r.choice([f"who teaches {title}?", f"who takes {title.lower()}?", f"who are the {_short_title(title)} teachers?"])
        answer = Answerer(d, asker).answer(ToolCall("describe", {"course": course}, self.week), None)
        fill = {"T": _listed(names), "day": self._day(r.choice(d.calendar.days)), "time": hour(self._slot()),
                "when": r.choice(WHEN)}
        history = [Turn("you", q), Turn("assistant", answer)]
        if r.random() < 0.5:  # "she" when the course has two teachers: which one?
            s, o = r.choice(PRONOUNS[:2])
            follow, _ = r.choice(_forms([x for x in ABOUT_TEACHER if "{S}" in x[0] or "{O}" in x[0]], self.test))
            either = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]
            return asker, history, _tidy(follow.format(**{**fill, "S": s, "O": o})), \
                Rewrite(ask=f"Who do you mean: {either}?"), False
        follow, full = r.choice(_forms(BOTH, self.test))
        return asker, history, _tidy(follow.format(**fill)), Rewrite(message=_tidy(full.format(**fill))), False

    # -- questions back, requests ------------------------------------------------------------------------------

    def _reply(self):
        """The reply to a question the assistant asked: the earlier question with what it asked filled in."""
        r = self.r
        pairs = self._shared_firsts()
        if pairs and r.random() < 0.4:
            return self._which_one(pairs)
        rows = [x for x in self.pool if any(n.rstrip("0123456789") in VAGUE for n in self._changeable(x))]
        row = r.choice(rows)
        filler, f1 = self._fill(row, r.choice(WHEN))
        name = r.choice([n for n in self._changeable(row) if n.rstrip("0123456789") in VAGUE])
        said = filler.said.get(name, "")
        if not said or f1.question.count(said) != 1:
            raise Skip("the name can't be left vague")
        vague, back = VAGUE[name.rstrip("0123456789")]
        q0 = f1.question.replace(said, r.choice(vague))
        v = filler._text(name, "any") if r.random() < 0.5 else said
        follow = r.choice(_forms(REPLY, self.test)).format(v=v)
        return f1.asker, [Turn("you", q0), Turn("assistant", back)], follow, Rewrite(message=f1.question), False

    def _shared_firsts(self) -> list[list]:
        firsts: dict[str, list] = {}
        for f in self.dept.faculty:
            if len(p := f.name.split()) == 3 and any(s.faculty == f.id for s in self.dept.sessions):
                firsts.setdefault(p[1], []).append(f)
        return [v for v in firsts.values() if len(v) > 1]

    def _which_one(self, pairs):
        """ "Which one do you mean: Dr. Anil Rao or Dr. Anil Das?" (the lookup's own question back) -> "Das"."""
        r = self.r
        both = r.choice(pairs)
        rows = [x for x in self.pool if any(n.rstrip("0123456789") == "teacher" for n in self._changeable(x))]
        row = r.choice(rows)
        name = next(n for n in self._changeable(row) if n.rstrip("0123456789") == "teacher")
        chosen = r.choice(both)
        filler, f1 = self._fill(row, r.choice(WHEN), None, self._filler(Refiller, picks={}, said={}, force={name: chosen}))
        said = filler.said.get(name, "")
        if not said or f1.question.count(said) != 1 or chosen.id == f1.asker:
            raise Skip("can't be asked that way")
        first = chosen.name.split()[1]
        q0 = f1.question.replace(said, r.choice([first, first.lower(), f"{first} sir"]))
        back = f"Which one do you mean: {' or '.join(f.name for f in both)}?"
        title, _, surname = chosen.name.split()
        nth = NTH[both.index(chosen)] if both.index(chosen) < 3 else None
        v = r.choice([surname, surname.lower(), chosen.name, f"{title} {surname}"] + ([f"the {nth} one"] if nth else []))
        follow = r.choice(_forms(REPLY, self.test)).format(v=v)
        return f1.asker, [Turn("you", q0), Turn("assistant", back)], follow, \
            Rewrite(message=f1.question.replace(said, chosen.name)), False

    def _request(self):
        r, d = self.r, self.dept
        _, f = self._fill(r.choice(self.requests), r.choice(WHEN))
        rid = f"R-{r.randint(3, 120):05d}"
        history = [Turn("you", f.question), Turn("assistant", f"Request {rid} sent: “{f.question}”.")]
        what = r.choice(["status", "status", "cancel", "change"])
        if what == "change":
            day = self._day(r.choice([x for x in d.calendar.days if FULL_DAY[x].lower() not in f.question.lower()]))
            follow, full = r.choice(_forms(CHANGE_REQUEST, self.test))
            return f.asker, history, follow.format(day=day), Rewrite(message=full.format(R=rid, day=day)), False
        follow, full = r.choice(_forms(CANCEL_REQUEST if what == "cancel" else ABOUT_REQUEST, self.test))
        return f.asker, history, follow, Rewrite(message=full.format(R=rid)), r.random() < 0.3

    def _whatif(self):
        """After a what-if ("what moves if I shift my DS lecture to Thursday 10 am?"): "ok do it" is the change as a
        request; "and to Friday instead?" is the what-if again with that changed."""
        r, d = self.r, self.dept
        asker = self._teaching()
        title = d.course_title(r.choice(list(dict.fromkeys(s.course for s in d.sessions if s.faculty == asker))))
        course, day, slot = r.choice([title, _short_title(title)]), r.choice(d.calendar.days), self._slot()
        kind = r.choice(["lecture", "class"])
        said, at = self._day(day), hour(slot)
        q = r.choice([f"what happens if I move my {course} {kind} to {said} at {at}?",
                      f"would anything move if I shift my {course} {kind} to {said} {at}?",
                      f"what if my {course} {kind} goes to {said} at {at}?"])
        others = r.choice([0, 0, 1, 2])  # the solver's reply, as World.what_if words it
        moved = "No other class would move." if not others else \
            f"{others} other class{'es' if others > 1 else ''} would move: " + "; ".join(
                f"{d.course_title(s.course)} {s.kind.value} to {FULL_DAY[r.choice(d.calendar.days)]} {hour(self._slot())}"
                for s in r.sample(d.sessions, others)) + "."
        answer = (f"If you sent it now (week {self.week}): your {title} {kind} moves to {FULL_DAY[day]} {at}. {moved} "
                  "Nothing has been changed: send it as a request to make it happen.")
        history = [Turn("you", q), Turn("assistant", answer)]
        if r.random() < 0.75:
            follow = r.choice(_forms(DO_IT, self.test))
            return asker, history, follow, Rewrite(message=f"move my {course} {kind} to {said} at {at}"), False
        day2 = r.choice([x for x in d.calendar.days if x != day])
        follow = r.choice(_forms(FORMS["day"], self.test)).format(v=self._day(day2), old=said)
        return asker, history, follow, Rewrite(message=q.replace(said, FULL_DAY[day2], 1)), False

    def _yes(self):
        r = self.r
        _, f = self._fill(r.choice(self.requests), r.choice(WHEN))
        history = [Turn("you", f.question), Turn("assistant", f"Send this as a request? “{f.question}”")]
        return f.asker, history, r.choice(_forms(YES, self.test)), Rewrite(message=f.question), False

    def _each(self):
        """ "Each on its own" or "together", switched in a follow-up."""
        r, d = self.r, self.dept
        asker = self._teaching()
        if r.random() < 0.5:
            things = [g.name for g in r.sample(d.groups, r.choice([2, 3]))]
            ids = {"sections": [g.id for g in d.groups if g.name in things]}
        else:
            things = [f.name for f in r.sample([f for f in d.faculty if f.id != asker], r.choice([2, 3]))]
            ids = {"teachers": [f.id for f in d.faculty if f.name in things]}
        day = r.choice(d.calendar.days)
        on = r.choice(["", f" on {self._day(day)}"])
        days = {"days": [day]} if on else {}
        together, apart = f"when are {_listed(things)} all free together{on}?", f"when is each of {_listed(things)} free{on}?"
        if r.random() < 0.5:
            q, answer = f"when are {_listed(things)} free{on}?", self._say_answer(asker, "free_hours", {**ids, **days})
            follow, full = r.choice(_forms(SEPARATELY, self.test)), apart
        else:
            key = next(iter(ids))
            q = apart
            answer = " ".join(self._say_answer(asker, "free_hours", {key: [x], **days}) for x in ids[key])
            follow, full = r.choice(_forms(TOGETHER, self.test)), together
        return asker, [Turn("you", q), Turn("assistant", _short(answer))], follow, Rewrite(message=full), False

    # -- the rest ---------------------------------------------------------------------------------------------

    def _same(self):
        r = self.r
        if r.random() < 0.3:  # the same subject, asked in full: nothing to fill in
            asker, history, _, _, distractor = self._change(correct=False)
            return asker, history, self.full, Rewrite(same=True), distractor
        history = self._exchange(self._office_or())
        _, f = self._fill(r.choice(self._rows_for(self.asker)), r.choice(WHEN), self.asker)
        return self.asker, history, f.question, Rewrite(same=True), r.random() < 0.4

    def _ask(self):
        r = self.r
        follow, back, lacks = r.choice(_forms(ASK, self.test))
        rows = [x for x in self.pool if x["tool"] in ("describe", "changes", "classes_of", "who_is_free")
                and not any(m.group(1).rstrip("0123456789") in ("teacher", "room", "lab")
                            for m in PLACEHOLDER.finditer(x["template"]))]
        for _ in range(10):
            history = self._exchange(rows=rows)
            said = " ".join(t.text for t in history)
            has = {"room": any(x.name in said for x in self.dept.rooms) or re.search(r"\b(room|lab)", said, re.I),
                   "teacher": any(x.name in said for x in self.dept.faculty) or re.search(r"\b(Dr|Prof)\.", said),
                   "list": ", " in said or ";" in said, "request": "R-" in said, "time": re.search(r"\d (am|pm)", said)}
            if not has[lacks]:
                return self.asker, history, follow, Rewrite(ask=back), False
        raise Skip("every reply had it")

    def _mixed(self):
        asker, history, follow, rewrite, distractor = self._change(correct=False)
        _, f = self._fill(self.r.choice(self._rows_for(asker)), "", asker)
        join = self.r.choice(JOIN)
        return asker, history, join.format(a=follow, b=f.question), \
            Rewrite(message=join.format(a=rewrite.message, b=f.question)), distractor


def build_departments(instance: Instance, test: bool, seed: int = 0) -> list[tuple[str, Instance]]:
    """The departments the conversations happen in (the same every time for a seed): renamed copies of the team's
    (the test set's with other names than any training one's), then (``shared-…``) generated departments whose
    courses have two teachers, used only for their facts."""
    rng = random.Random(f"memory-departments-{seed}-{test}")
    if test:
        return [(f"test-renamed-{k}", renamed(instance, rng, first_names=k % 2 == 0, move_hod=True)) for k in range(3)] + \
               [("test-shared-8", renamed(department(instance, 8, shared=True), rng, first_names=True))]
    return [("original", instance), ("first-names", with_first_names(instance))] + \
           [(f"renamed-{k}", renamed(instance, rng, first_names=k % 2 == 0, move_hod=k >= 2)) for k in range(4)] + \
           [(f"shared-s{s}", renamed(department(instance, s, shared=True), rng, first_names=s % 2 == 0)) for s in (4, 5)]


def build(instance: Instance, store, n: int, test: bool, seed: int = 0) -> tuple[list[dict], Counter, Counter]:
    """``n`` conversations, kinds as ``SHARES``; the test set (``test``) from the test templates and wordings."""
    every = build_departments(instance, test, seed)
    departments = [x for x in every if "shared" not in x[0]]
    shared = [x for x in every if "shared" in x[0]]
    files = (TEST_V1, TEST_V2, TEST_V3) if test else (TRAIN_V2, TRAIN_V3)
    rows = [json.loads(x) for p in files for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]
    pool = [x for x in rows if x["tool"] in TURN1_TOOLS]
    requests = [x for x in rows if x["tool"] is None and x.get("group") == "request"]
    if not requests:
        requests = [x for x in (json.loads(y) for y in TRAIN_V2.read_text(encoding="utf-8").splitlines() if y.strip())
                    if x["tool"] is None and x.get("group") == "request"]
    b = Builder(store, departments, shared, pool, requests, test, seed)
    kinds = [k for k, share in SHARES.items() for _ in range(share)]
    out, cover, k = [], Counter(), 0
    while len(out) < n and k < n * 4:
        kind = kinds[k % len(kinds)] if test else random.Random(f"{seed}-{k}").choice(kinds)
        if (row := b.row(k, kind)) is not None:
            out.append(row)
            cover[kind] += 1
        k += 1
    return out, cover, b.skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, default=2800, help="conversations for training (a tenth kept for validation)")
    ap.add_argument("--test", type=int, default=400, help="conversations for the test set")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    from web.app import World

    w = World(parser_mode="offline", seed_history=False, deadline=60)
    train, cover, skipped = build(w.instance, w.store, a.train, test=False, seed=a.seed)
    test, test_cover, test_skipped = build(w.instance, w.store, a.test, test=True, seed=a.seed)
    random.Random(a.seed).shuffle(train)
    val, train = train[: len(train) // 10], train[len(train) // 10:]
    a.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val), ("test", test)):
        with (a.out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    manifest = {"task": "memory", "version": 1, "train": len(train), "val": len(val), "test": len(test),
                "seed": a.seed, "kinds": dict(cover), "test_kinds": dict(test_cover)}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{len(train)} train, {len(val)} validation, {len(test)} test conversations in {a.out}")
    print("kinds:", dict(cover), "| test:", dict(test_cover))
    print("skipped:", dict(skipped + test_skipped))


if __name__ == "__main__":
    main()
