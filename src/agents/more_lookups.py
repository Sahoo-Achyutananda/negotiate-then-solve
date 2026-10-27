"""The lookups added for nts-lookup-v3 (ROADMAP step 8): questions about the week beyond free hours and classes,
and about the asker's own requests, rules, fairness and bookings. Like every lookup, the answer is computed by
code from the data; a model only chooses which lookup.

    records = Records.from_world(world)                      # the app's data the timetable doesn't hold
    Answerer(instance, sender, records).answer(ToolCall("my_requests", {"which": "latest"}, 13), week)

Tools (``ToolCall.tool``) and their arguments (IDs, after ToolAgent's checks):

| tool          | asks                                                        | args                                         |
|---------------|-------------------------------------------------------------|----------------------------------------------|
| calendar      | which week is it, a week's dates, weeks left                 | (the week)                                   |
| my_requests   | status of my requests, what happened to R-00018              | which: latest/open/waiting/all; id           |
| my_rules      | the wishes and absences I have set                           | (none)                                       |
| clash_check   | do these people / sections / rooms have overlapping classes  | teachers, sections, rooms; days              |
| move_options  | where one class could go this week                           | session; days, part, slots                   |
| counts        | totals and rankings: who teaches most, my hours              | measure, of, order; teachers, sections; days |
| gaps          | longest free block, classes back to back                     | measure: free/busy; teachers, sections, rooms; days |
| my_fairness   | how often I have given way                                   | (none)                                       |
| bookings      | club room bookings and their status                          | club                                         |
| what_if       | what would move if a change were made (a dry run)            | request (the change in words)                |

``free_hours`` also takes ``room_for`` (a room must seat these sections together) and every week-based lookup
takes ``weeks`` (several weeks; see ``Answerer.answer``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING

from core.instance import Instance
from language.corpus import FULL_DAY, hour

from .lookup import WeekTimetable, part_slots

if TYPE_CHECKING:
    from .questions import Answerer, ToolCall

MORE_TOOLS = ("calendar", "my_requests", "my_rules", "clash_check", "move_options", "counts", "gaps",
              "my_fairness", "bookings", "what_if")
FINISHED = {"published", "refused", "denied", "answered", "forwarded", "withdrawn"}
STATUS_WORDS = {"received": "received", "classified": "being read", "policy_checked": "being checked",
                "compiled": "being solved", "solved": "being solved", "negotiating": "in negotiation",
                "fairness_audited": "being checked", "awaiting_approval": "waiting for the coordinator's approval",
                "published": "published", "refused": "refused", "denied": "denied", "escalated": "escalated for a decision",
                "clarification_requested": "waiting for your answer to a question", "answered": "answered",
                "forwarded": "forwarded to the timetable office", "withdrawn": "withdrawn"}


@dataclass
class RequestRow:
    id: str
    sender: str
    text: str
    status: str
    reply: str | None
    received: str


@dataclass
class Records:
    """What the new lookups read besides the timetable, as the app has it now. Every field is optional: a lookup
    whose data is missing says it cannot answer here."""

    instance: Instance
    store: object | None = None  # pipeline.store.Store: timetables of other weeks, constraints, the ledger
    requests: list[RequestRow] = field(default_factory=list)
    waiting_on: dict[str, set[str]] = field(default_factory=dict)  # person -> request IDs that wait on their answer
    bookings: list = field(default_factory=list)  # semester.clubs.ClubDecision
    semester: str = "current"
    semester_start: date | None = None
    current_week: int | None = None
    what_if: Callable[[str, str, int], str] | None = None  # (change in words, sender, week) -> the dry run's answer

    @classmethod
    def from_world(cls, w) -> Records:
        """The records of a running ``web.world.World``."""
        rows = []
        for r in w.store.requests():
            case = w.orch.cases.get(r.id)
            rows.append(RequestRow(r.id, r.sender_id, r.raw_text, (case.status if case else r.status).value,
                                   case.reply if case else None, r.received_at.isoformat()))
        waiting: dict[str, set[str]] = {}
        for item in w.inbox.items.values():
            if item.reply is None and item.case_id:
                waiting.setdefault(item.to, set()).add(item.case_id)
        for row in rows:
            if row.status == "clarification_requested":
                waiting.setdefault(row.sender, set()).add(row.id)
        state = getattr(getattr(w, "semester", None), "state", None)
        start = getattr(state, "semester_start", None)
        return cls(w.instance, w.store, rows, waiting, list(getattr(state, "bookings", []) or []),
                   semester=getattr(w.orch, "semester", "current"),
                   semester_start=date.fromisoformat(start) if isinstance(start, str) else start,
                   current_week=w.current_week() if hasattr(w, "current_week") else None,
                   what_if=getattr(w, "what_if", None))


def answer(a: Answerer, call: ToolCall, week: WeekTimetable) -> str:
    """The reply for one of ``MORE_TOOLS``."""
    return {"calendar": _calendar, "my_requests": _my_requests, "my_rules": _my_rules, "clash_check": _clash,
            "move_options": _move_options, "counts": _counts, "gaps": _gaps, "my_fairness": _fairness,
            "bookings": _bookings, "what_if": _what_if}[call.tool](a, call, week)


# -- the week ------------------------------------------------------------------------------------------------


def _calendar(a: Answerer, call: ToolCall, week: WeekTimetable) -> str:
    cal, rec = a.instance.calendar, a.records
    now = rec.current_week if rec and rec.current_week else None
    said = []
    if now:
        said.append(f"This is week {now} of {cal.weeks}; {max(cal.weeks - now, 0)} "
                    f"week{'s' if cal.weeks - now != 1 else ''} left after it.")
    if rec and rec.semester_start:
        monday = rec.semester_start + timedelta(weeks=call.week - 1)
        last = monday + timedelta(days=len(cal.days) - 1)
        said.append(f"Week {call.week} runs from {_date(monday)} to {_date(last)}.")
        for d in call.args.get("days", []):
            said.append(f"{FULL_DAY[d]} of week {call.week} is {_date(monday + timedelta(days=cal.days.index(d)))}.")
    elif not now:
        said.append(f"The semester has {cal.weeks} teaching weeks; the dates are not in the data here.")
    return " ".join(said)


def _date(d: date) -> str:
    return f"{d:%A} {d.day} {d:%B}"


def _clash(a: Answerer, call: ToolCall, week: WeekTimetable) -> str:
    """Overlaps among the people, sections and rooms named: the hours when more than one of them has a class, and
    any real clash (one of them in two classes at once, which a published timetable never has)."""
    x = call.args
    who = [("teacher", t) for t in x.get("teachers", [])] + [("section", g) for g in x.get("sections", [])] + \
          [("room", r) for r in x.get("rooms", [])]
    busy: dict[tuple[str, str], dict[tuple[str, int], list]] = {}
    for kind, i in who:
        hours: dict[tuple[str, int], list] = {}
        for c in week.classes_of(**{kind: i}):
            if x.get("days") and c.placement.day not in x["days"]:
                continue
            for h in c.hours():
                hours.setdefault(h, []).append(c)
        busy[(kind, i)] = hours
    clashes = [(k, h) for k, hours in busy.items() for h, cs in hours.items() if len(cs) > 1]
    names, _ = a._who({"teachers": x.get("teachers", []), "sections": x.get("sections", []), "rooms": x.get("rooms", [])})
    lead = (f"Real clashes (two classes at once): " + "; ".join(f"{a._name(k)} on {FULL_DAY[h[0]]} {hour(h[1])}"
                                                                  for k, h in clashes[:5]) + ". ") if clashes else ""
    if len(who) < 2:
        return lead or _cap(f"{names} {'have' if names == 'you' else 'has'} no clashes in week {call.week}.")
    common = sorted(set.intersection(*(set(h) for h in busy.values())), key=lambda h: (a.instance.calendar.days.index(h[0]), h[1]))
    if not common:
        return lead + _cap(f"{names} never have classes at the same time in week {call.week}.")
    return lead + _cap(f"{names} all have classes at the same time on {a._spans(common)} (week {call.week}).")


def _move_options(a: Answerer, call: ToolCall, week: WeekTimetable) -> str:
    """Where one class could go in the week: hours when its teacher and sections are free (for its whole length)
    and a room of its kind, with its equipment and enough seats, is free too; its own time left out."""
    x, inst = call.args, a.instance
    placed = {c.session.id: c for c in week.classes}
    ids = x.get("sessions") or [x["session"]]
    if x.get("class_day"):  # "my Friday DS lecture": the one on that day this week
        ids = [i for i in ids if i in placed and placed[i].placement.day == x["class_day"]] or []
        if not ids:
            return f"You have no such class on {FULL_DAY[x['class_day']]} in week {call.week}."
    if len(ids) > 1:
        which = "; ".join(f"{inst.course_title(inst.session_by_id[i].course)} {inst.session_by_id[i].kind.value} on "
                          f"{FULL_DAY[placed[i].placement.day]} {hour(placed[i].placement.slot)}" for i in ids if i in placed)
        return f"Which class do you mean: {which}?"
    s = inst.session_by_id[ids[0]]
    here = placed.get(s.id)
    seats = sum(inst.group_by_id[g].size for g in s.groups)
    kind = s.room_type.value if s.room_type else "lecture"
    days = x.get("days") or None
    found = week.free_with_room([s.faculty], list(s.groups), kind, list(s.equipment), seats, days, s.duration)
    keep = set(x["slots"]) if x.get("slots") else set(part_slots(inst, x["part"])) if x.get("part") else None
    starts: dict[tuple[str, int], list[str]] = {}
    for room, hours in found:
        have = set(hours)
        for d, q in hours:
            if keep is not None and q not in keep:
                continue
            if all((d, q + k) in have for k in range(s.duration)) and not (here and (d, q) == (here.placement.day, here.placement.slot)):
                starts.setdefault((d, q), []).append(room.name)
    what = f"{inst.course_title(s.course)} {s.kind.value}"
    if not starts:
        return f"There is no free time for your {what} in week {call.week}" + \
               (f" on {', '.join(FULL_DAY[d] for d in days)}" if days else "") + \
               " when its teacher, its sections and a suitable room are all free."
    order = sorted(starts, key=lambda h: (inst.calendar.days.index(h[0]), h[1]))
    now = f" (now {FULL_DAY[here.placement.day]} {hour(here.placement.slot)})" if here else ""
    listed = "; ".join(f"{FULL_DAY[d]} {_range(q, q + s.duration)} ({', '.join(starts[(d, q)][:2])})" for d, q in order[:8])
    more = f"; and {len(order) - 8} more" if len(order) > 8 else ""
    return f"Your {what}{now} could move to: {listed}{more} (week {call.week})."


def _counts(a: Answerer, call: ToolCall, week: WeekTimetable) -> str:
    """Totals per teacher, section or room, ranked: "who teaches the most?", "how many hours does each section
    have?", "which room is used least?"."""
    x, inst = call.args, a.instance
    of, measure, order = x.get("of", "teachers"), x.get("measure", "teaching_hours"), x.get("order")
    pool = {"teachers": [(f.id, f.name, "teacher") for f in inst.faculty if any(s.faculty == f.id for s in inst.sessions)],
            "sections": [(g.id, g.name, "section") for g in inst.groups],
            "rooms": [(r.id, r.name, "room") for r in inst.rooms]}[of]
    named = set(x.get("teachers", [])) | set(x.get("sections", [])) | set(x.get("rooms", []))
    if named:
        pool = [p for p in pool if p[0] in named]
    days = x.get("days")
    rows = []
    for i, name, kind in pool:
        classes = [c for c in week.classes_of(**{kind: i}) if not days or c.placement.day in days]
        if measure == "free_hours":
            n = len(week.free_hours(**{f"{kind}s" if kind != "room" else "rooms": [i]}, days=days))
        elif measure == "classes":
            n = len(classes)
        else:
            n = sum(c.session.duration for c in classes)
        rows.append((n, a._name((kind, i)), i))
    unit = {"teaching_hours": "hours of class", "classes": "classes", "free_hours": "free hours"}[measure]
    rows.sort(key=lambda r: (-r[0] if order != "least" else r[0], r[1]))
    on = f" on {', '.join(FULL_DAY[d] for d in days)}" if days else ""
    if not rows:
        return "There is nobody to count."
    if order in ("most", "least"):
        best = [r for r in rows if r[0] == rows[0][0]]
        who = " and ".join(r[1] for r in best)
        return _cap(f"{who} {'have' if len(best) > 1 or who == 'you' else 'has'} the {order} {unit}{on} in week "
                    f"{call.week}: {rows[0][0]}. ") + "All: " + "; ".join(f"{r[1]} {r[0]}" for r in rows) + "."
    if len(rows) == 1:
        return _cap(f"{rows[0][1]} {'have' if rows[0][1] == 'you' else 'has'} {rows[0][0]} {unit}{on} in week {call.week}.")
    return _cap(f"{unit}{on} in week {call.week}: ") + "; ".join(f"{r[1]} {r[0]}" for r in rows) + "."


def _gaps(a: Answerer, call: ToolCall, week: WeekTimetable) -> str:
    """The longest run of free hours ("my longest free block") or of class hours back to back ("do I have three
    classes in a row?"), per day, for the people, sections or rooms named together."""
    x, inst = call.args, a.instance
    cal = inst.calendar
    days = x.get("days") or list(cal.days)
    free = set(week.free_hours(x.get("teachers", ()), x.get("sections", ()), x.get("rooms", ()), days))
    want_free = x.get("measure", "free") == "free"
    best: list[tuple[int, str, int]] = []
    for d in days:
        run, start = 0, 0
        for q in range(cal.slots_per_day + 1):
            on = q < cal.slots_per_day and q != cal.lunch_slot and (((d, q) in free) == want_free)
            if on:
                start, run = (q, 1) if run == 0 else (start, run + 1)
            elif run:
                best.append((run, d, start))
                run = 0
    who, plural = a._who({k: x.get(k, []) for k in ("teachers", "sections", "rooms")})
    if not best:
        return _cap(f"{who} {'have' if plural else 'has'} no {'free hours' if want_free else 'classes'} on those days "
                    f"in week {call.week}.")
    best.sort(key=lambda r: (-r[0], cal.days.index(r[1]), r[2]))
    top = [r for r in best if r[0] == best[0][0]]
    when = "; ".join(f"{FULL_DAY[d]} {_range(q, q + n)}" for n, d, q in top)
    if want_free:
        return _cap(f"{who}'s longest free block{'s are' if len(top) > 1 else ' is'} {best[0][0]} hour"
                    f"{'s' if best[0][0] != 1 else ''}: {when} (week {call.week}).")
    return _cap(f"{who}'s longest run of classes back to back is {best[0][0]} hour{'s' if best[0][0] != 1 else ''}: "
                f"{when} (week {call.week}).")


# -- the asker's own records -------------------------------------------------------------------------------


def _my_requests(a: Answerer, call: ToolCall, week: WeekTimetable) -> str:
    rec, x, me = a.records, call.args, a.sender
    if rec is None:
        return "I can't see the requests from here."
    mine = [r for r in rec.requests if r.sender == me]
    waiting = rec.waiting_on.get(me, set())
    if x.get("id"):
        r = next((r for r in rec.requests if r.id == x["id"]), None)
        if r is None or (r.sender != me and r.id not in waiting and not a.sees_all):
            return f"{x['id']} is not one of your requests."
        return _request_line(r, mine=r.sender == me)
    which = x.get("which", "latest")
    if which == "waiting":
        rows = [r for r in rec.requests if r.id in waiting]
        if not rows:
            return "Nothing is waiting for your answer."
        return f"{len(rows)} request{'s wait' if len(rows) != 1 else ' waits'} for your answer: " + \
               " ".join(_request_line(r, mine=r.sender == me) for r in rows)
    if not mine:
        return "You have not sent any requests yet."
    mine.sort(key=lambda r: r.received)
    if which == "latest":
        return _request_line(mine[-1], mine=True)
    if which == "open":
        rows = [r for r in mine if r.status not in FINISHED]
        if not rows:
            return "None of your requests is still open."
        return f"{len(rows)} of your requests {'is' if len(rows) == 1 else 'are'} still open: " + \
               " ".join(_request_line(r, mine=True) for r in rows)
    return f"Your requests ({len(mine)}): " + " ".join(_request_line(r, mine=True) for r in mine[-6:])


def _request_line(r: RequestRow, mine: bool) -> str:
    text = " ".join(r.text.split())
    short = text if len(text) <= 70 else text[:69].rsplit(" ", 1)[0] + "..."
    state = STATUS_WORDS.get(r.status, r.status.replace("_", " "))
    reply = f" {r.reply}" if r.reply and r.status in FINISHED | {"clarification_requested", "awaiting_approval"} else ""
    whose = "" if mine else " (a colleague's request)"
    return f"{r.id}{whose} (\"{short}\") is {state}.{reply}"


def _my_rules(a: Answerer, call: ToolCall, week: WeekTimetable) -> str:
    rec = a.records
    if rec is None or rec.store is None:
        return "I can't see the rules from here."
    from pipeline.orchestrator import explain_constraint

    mine = [c for c in rec.store.constraints() if c.owner == a.sender]
    if not mine:
        return "You have not set any wishes or absences."
    lines = [f"{'firm' if c.hard else 'a wish'}: {explain_constraint(a.instance, c)}" for c in mine[:10]]
    more = f" (and {len(mine) - 10} more)" if len(mine) > 10 else ""
    return f"Your {len(mine)} standing rule{'s' if len(mine) != 1 else ''}: " + "; ".join(lines) + more + "."


def _fairness(a: Answerer, call: ToolCall, week: WeekTimetable) -> str:
    rec = a.records
    if rec is None or rec.store is None:
        return "I can't see the fairness record from here."
    from .ledger import ConcessionLedger

    ledger = ConcessionLedger(rec.store.ledger())
    times = ledger.times(rec.semester)
    mine = times.get(a.sender, 0)
    teaching = sorted({s.faculty for s in a.instance.sessions})
    rank = sorted((times.get(t, 0) for t in teaching), reverse=True)
    more = sum(1 for n in rank if n > mine)
    if mine == 0:
        return "You have not given way to anyone's request this semester."
    return (f"You have given way {mine} time{'s' if mine != 1 else ''} this semester; {more} of {len(teaching)} "
            f"teachers have given way more often. Whoever has given way less is asked first next time.")


def _bookings(a: Answerer, call: ToolCall, week: WeekTimetable) -> str:
    rec = a.records
    rows = list(rec.bookings) if rec else []
    club = (call.args.get("club") or "").lower()
    if club:
        rows = [b for b in rows if club in b.request.club.lower()]
    if not rows:
        return f"There are no room bookings{' for ' + call.args['club'] if club else ''}."
    words = {"booked": "booked", "needs_approval": "waiting for the timetable office", "rejected": "turned down"}
    return "; ".join(f"{b.request.club}, {b.request.activity} on {b.request.date} {b.request.start}–{b.request.end}: "
                     f"{words.get(b.status, b.status)}" + (f" in {b.room}" if b.room and b.status == "booked" else "")
                     for b in rows[-6:]) + "."


def _what_if(a: Answerer, call: ToolCall, week: WeekTimetable) -> str:
    rec = a.records
    if rec is None or rec.what_if is None:
        return "I can't try out changes from here; send it as a request to see what happens."
    return rec.what_if(call.args["request"], a.sender, call.week)


def _range(start: int, end: int) -> str:
    from .questions import _range as r

    return r(start, end)


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]
