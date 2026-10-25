"""The part of the timetable an answer is about, for the chat to draw beside it (CHAT-PLAN.md, step 2).

    view_of(instance, call, WeekTimetable(instance, store, call.week))
    # {"layout": "week", "week": 13, "days": ["Fri"], "slots": [0, 1, 2, 3], "lunch": 4,
    #  "cells": [{"day": "Fri", "slot": 0, "span": 1, "kind": "free"}, ...]}

Computed with the same lookups (agents.lookup) and the same filters (days, hours, morning/afternoon, hours in a row,
a room needed) as ``Answerer`` uses for the written answer, so the picture and the words agree. Two layouts:

``week``: one timetable, days across and hours down, as on the Timetable page. For free hours (one person, a
          section, several together, a room), classes, the next class, a week's changes.
``rows``: several timetables for one day, one row per room or person, hours across. For which rooms are free
          and who is free (only when the question is about one day: a week of rows does not fit in a chat).

Answers with nothing to draw (department facts, counts, "I don't know …") have no view (None).
"""

from __future__ import annotations

from core.instance import Instance

from .lookup import WeekTimetable, part_slots
from .questions import CHANGE_KINDS, ToolCall


def view_of(instance: Instance, call: ToolCall, week: WeekTimetable, sender: str | None = None) -> dict | None:
    """The slice for ``call``'s answer, or None when there is nothing to draw. ``sender``: left out of "who is
    free", as in the written answer."""
    if not call.confident:
        return None
    a = call.args
    if a.get("each"):  # a course's co-teachers, each on their own: one set of hours would mix them up
        return None
    if call.tool == "free_hours" and not (a.get("kind") or a.get("equipment") or a.get("min_capacity")):
        return _free_hours(instance, call, week)
    if call.tool == "free_hours":  # a room needed too: one row per room that fits
        found = week.free_with_room(a.get("teachers", ()), a.get("sections", ()), a.get("kind"), a.get("equipment", ()),
                                    a.get("min_capacity"), a.get("days"), a.get("length", 1))
        return _rows(instance, call, [(r.name, h) for r, h in found])
    if call.tool == "free_rooms":
        found = week.free_rooms(a.get("kind"), a.get("equipment", ()), a.get("min_capacity"), a.get("days"))
        return _rows(instance, call, [(r.name, h) for r, h in found])
    if call.tool == "who_is_free":
        slots = a.get("slots") or (part_slots(instance, a["part"]) if a.get("part") else None)
        found = week.who_is_free(a.get("days"), slots, a.get("course"), a.get("sections", False))
        return _rows(instance, call, [(p.name, h) for p, h in found if p.id != sender])
    if call.tool == "classes_of":
        return _classes(instance, call, week)
    if call.tool == "changes":
        return _changes(instance, call, week)
    return None  # describe: facts, not times


def free_hours_of(instance: Instance, call: ToolCall, week: WeekTimetable) -> list[tuple[str, int]]:
    """The free hours a ``free_hours`` answer is about, every filter applied: the hours the timetable shades
    (``/api/ask``'s ``free``). With a room needed too, the hours when the people and at least one such room are free.
    Nothing for co-teachers answered each on their own (``each``)."""
    a = call.args
    if a.get("each"):
        return []
    if a.get("kind") or a.get("equipment") or a.get("min_capacity"):
        found = week.free_with_room(a.get("teachers", ()), a.get("sections", ()), a.get("kind"), a.get("equipment", ()),
                                    a.get("min_capacity"), a.get("days"), a.get("length", 1))
        free = sorted({h for _, hours in found for h in hours},
                      key=lambda h: (instance.calendar.days.index(h[0]), h[1]))
    else:
        free = week.free_hours(a.get("teachers", ()), a.get("sections", ()), a.get("rooms", ()), a.get("days"),
                               length=a.get("length", 1))
    keep = set(a["slots"]) if a.get("slots") else set(part_slots(instance, a["part"])) if a.get("part") else None
    free = [h for h in free if keep is None or h[1] in keep]
    if a.get("first") and free:  # only the earliest: its first hour, or its first run of ``length`` hours
        day, slot = free[0]
        free = [(day, s) for s in range(slot, slot + a.get("length", 1)) if (day, s) in free]
    return free


# -- layout "week": one timetable -----------------------------------------------------------------------------


def _free_hours(instance: Instance, call: ToolCall, week: WeekTimetable) -> dict:
    a = call.args
    free = free_hours_of(instance, call, week)
    cells = [_cell(d, s, "free") for d, s in free]
    if a.get("slots"):  # "is Lab 2 free at 3 pm?": the hours asked, free or taken
        days = a.get("days") or instance.calendar.days
        cells = [_cell(d, s, "free" if (d, s) in free else "busy") for d in days for s in a["slots"]]
    return _week(instance, call, cells, days=a.get("days"))


def _classes(instance: Instance, call: ToolCall, week: WeekTimetable) -> dict | None:
    a = call.args
    if "after" in a:  # "my next class": that one class
        found = [c for c in [week.next_class(tuple(a["after"]), a.get("teacher"), a.get("section"))] if c]
    else:
        found = week.classes_of(a.get("teacher"), a.get("section"), a.get("room"), a.get("course"))
        found = [c for c in found if not a.get("days") or c.placement.day in a["days"]]
    if not found:
        return None
    cells = [_class_cell(instance, c.session, c.placement, "class", teacher=not a.get("teacher")) for c in found]
    return _week(instance, call, cells, days=a.get("days"))


def _changes(instance: Instance, call: ToolCall, week: WeekTimetable) -> dict | None:
    a = call.args
    cells = []
    for c in week.changes(a.get("teacher"), a.get("section")):
        if a.get("change") and c.kind != CHANGE_KINDS[a["change"]]:  # only the kind asked about
            continue
        if c.kind == "not_held":
            cells.append(_class_cell(instance, c.session, c.before, "cancelled"))
        elif c.kind == "moved":
            cells.append(_class_cell(instance, c.session, c.before, "moved_from"))
            cells.append(_class_cell(instance, c.session, c.after, "moved_to"))
        else:
            cells.append(_class_cell(instance, c.session, c.after, "extra"))
    return _week(instance, call, cells) if cells else None


def _week(instance: Instance, call: ToolCall, cells: list[dict], days: list[str] | None = None) -> dict:
    """Only the days asked (or, when none was, the days with something on them) and the hours from the first
    to the last cell, lunch included when it falls between."""
    cal = instance.calendar
    shown = days or [d for d in cal.days if any(c["day"] == d for c in cells)] or list(cal.days)
    hours = [s for c in cells for s in range(c["slot"], c["slot"] + c["span"])]
    slots = list(range(min(hours), max(hours) + 1)) if hours else list(range(cal.slots_per_day))
    return {"layout": "week", "week": call.week, "days": [d for d in cal.days if d in shown], "slots": slots,
            "lunch": cal.lunch_slot, "cells": cells}


# -- layout "rows": several timetables for one day -------------------------------------------------------------


def _rows(instance: Instance, call: ToolCall, found: list[tuple[str, list[tuple[str, int]]]]) -> dict | None:
    """One row per room or person, their free hours on the one day asked about; None for several days."""
    a = call.args
    days = a.get("days") or []
    if len(days) != 1 or not found:
        return None
    day, cal = days[0], instance.calendar
    keep = set(a["slots"]) if a.get("slots") else set(part_slots(instance, a["part"])) if a.get("part") else None
    rows = [{"name": name, "free": sorted(s for d, s in hours if d == day and (keep is None or s in keep))}
            for name, hours in found]
    rows = [r for r in rows if r["free"]]
    if not rows:
        return None
    hours = [s for r in rows for s in r["free"]]
    slots = sorted(keep) if keep else list(range(min(hours), max(hours) + 1))
    return {"layout": "rows", "week": call.week, "day": day, "slots": slots, "lunch": cal.lunch_slot, "rows": rows}


# -- cells ----------------------------------------------------------------------------------------------------


def _cell(day: str, slot: int, kind: str) -> dict:
    return {"day": day, "slot": slot, "span": 1, "kind": kind}


def _class_cell(instance: Instance, session, placement, kind: str, teacher: bool = True) -> dict:
    """A class in the slice: its title and kind, and under it the room and the section(s) (and the teacher)."""
    room = instance.room_by_id[placement.room].name if placement.room in instance.room_by_id else placement.room
    sections = ", ".join(instance.group_by_id[g].name for g in session.groups if g in instance.group_by_id)
    sub = " · ".join(x for x in (room, sections, instance.faculty_by_id[session.faculty].name if teacher else "") if x)
    return {"day": placement.day, "slot": placement.slot, "span": session.duration, "kind": kind,
            "label": f"{instance.course_title(session.course)} {session.kind.value}", "sub": sub,
            "session": session.id}
