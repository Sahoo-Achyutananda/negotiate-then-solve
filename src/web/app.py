"""JSON API behind the React front end (``frontend/``).

    uv run nts-web                        # API on http://127.0.0.1:8000 (serves frontend/dist if built)
    NTS_PARSER=gemini uv run nts-web      # System Two + policy agent on Gemini instead of the offline rules
    NTS_PARSER=local uv run nts-web       # fine-tuned model on llama-server (NTS_LOCAL_URL, default :8080)
    uv run nts-web --parser local --test-data 20   # benchmark department, 20 held-out test requests replayed

One ``World`` holds a demo department and the real pipeline: intake, System
One routing, the parser (offline rules, Gemini or the local compiler), the policy agent, the
negotiator with solver-verified options, the ledger, the store and the
approval gate. Requests run in background threads; the UI follows them
through the event log. When the negotiator needs someone's answer, the
message waits in that person's inbox until they reply (or a simulator
replies for them, on request or on autopilot).

Authentication is mocked: ``POST /api/login`` accepts any directory person
and the client sends ``X-User`` with each request. Replace with the college
SSO (OIDC) for real use; authorisation decisions are already server-side.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import threading
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agents.documents import SUFFIXES, ingest
from agents.explainer import TIER_NAME, describe, placement_text, session_name
from agents.ledger import ConcessionLedger
from agents.lookup import WeekTimetable
from agents.memory import turns_of
from agents.negotiation import Message, Reply
from agents.policy import RETRIEVERS, Rule, make_retriever, tokens
from agents.priority import JUSTIFICATION, ROLE_AUTHORITY, Weights
from agents.questions import Answerer
from agents.views import free_hours_of, view_of
from core.graph import add_conflict, build_graph
from core.instance import Session
from core.schemas import DAYS, Placement, Request, RequestStatus, Tier
from evaluation.stats import mean
from language.corpus import FULL_DAY, hour
from language.intents import IntentOutput, Unresolved
from language.intents import weeks as period_weeks
from language.llm import FREE_TIER, LLMError
from pipeline.ingest import UnknownSender
from pipeline.orchestrator import Case, NotAuthorised
from pipeline.store import EXTRA_PREFIX

from .wording import TIER_LABEL, plain
from .wording import TIER_WHO as TIER_WHO_PLAIN
from .world import (
    COORDINATOR,
    OVERSIGHT,
    ROOT,
    SEMESTER,
    VIEWS,
    InboxItem,
    World,
    views_for,
)

TIER_WHO = {0: "Nobody", 1: "Dean / academic council", 2: "Academic office", 3: "Owner, with an alternative",
            4: "Owner or HoD", 5: "Owner, freely"}
TIER_EXAMPLES = {0: "No double-booking; room capacity; lab equipment", 1: "Max 3 consecutive hours; lunch break",
                 2: "Exam blocks; guest faculty windows", 3: "Conference, medical leave, clinical duty",
                 4: "Lab needs GPUs or routers; room type", 5: "No classes before 10; fewer gaps"}


# ---------------------------------------------------------------------------
# JSON views
# ---------------------------------------------------------------------------


def _placement(w: World, p: Placement | None) -> dict | None:
    if p is None:
        return None
    return {"day": p.day, "slot": p.slot, "room": p.room, "text": placement_text(w.instance, p),
            "time": hour(p.slot)}


def _constraint(w: World, c) -> dict:
    return {"id": c.id, "type": c.type.value, "hard": c.hard, "tier": int(c.tier), "tier_name": TIER_NAME[c.tier],
            "tier_label": TIER_LABEL[int(c.tier)], "tier_who": TIER_WHO_PLAIN[int(c.tier)],
            "owner": c.owner, "owner_name": w.name_of(c.owner), "text": plain(describe(w.instance, c)),
            "when": c.when.model_dump(), "justification": c.justification.value,
            "source": c.source.model_dump(), "valid": c.valid.model_dump()}


def _case_weeks(case: Case | None) -> list[int] | None:
    """The teaching weeks a request is about; None when it holds every week (or nothing says)."""
    if case is None or not case.constraints or any(c.when.weeks is None for c in case.constraints):
        return None
    return sorted({n for c in case.constraints for n in c.when.weeks})


def _date_of(w: World, week: int, day: str) -> date:
    start = date.fromisoformat(w.semester.state.semester_start)
    return start - timedelta(days=start.weekday()) + timedelta(weeks=week - 1, days=DAYS.index(day))


def _dated(w: World, p: dict, weeks: list[int] | None) -> dict:
    """When a placement first happens: its date in the request's (first) week, and how many weeks it covers."""
    if not weeks or p["day"] not in DAYS:
        return p
    return p | {"date": _date_of(w, weeks[0], p["day"]).isoformat(), "weeks": len(weeks), "week": weeks[0]}


def _message(w: World, m: Message | dict, case: Case | None = None) -> dict:
    m = m if isinstance(m, dict) else m.model_dump()
    weeks = _case_weeks(case)
    return {**m, "to_name": w.name_of(m["to"]), "text": plain(m["text"]),
            "facts": [{**f, "text": plain(f["text"])} for f in m.get("facts", [])],
            "claims": [{**c, "text": plain(c["text"])} for c in m.get("claims", [])],
            "offers": [{**o, "placements": [{"session": s, "session_name": session_name(w.instance, s),
                                              **_dated(w, _placement(w, Placement(**p)), weeks)}
                                             for s, p in o["placements"].items()]}
                       for o in m["offers"]]}


def _summary(w: World, r: Request, case: Case | None) -> dict:
    status = case.status.value if case else r.status.value
    return {"id": r.id, "sender": r.sender_id, "sender_name": w.name_of(r.sender_id), "role": r.role.value,
            "channel": r.channel.value, "text": r.raw_text, "received_at": r.received_at.isoformat(),
            "makeup": r.makeup,
            "status": status, "route": case.route if case else "",
            "type": case.parse.output.request_type if case and case.parse and case.parse.output else None,
            "rounds": case.outcome.rounds if case and case.outcome else 0,
            "running": r.id in w.threads and w.threads[r.id].is_alive(),
            # who an escalation went to, so each person's home page lists only the decisions that are theirs
            "escalated_to": case.outcome.escalation.to if case and case.outcome and case.outcome.escalation else None,
            # who a forwarded message went to, and whether the coordinator has answered it yet
            "forwarded_to": case.forwarded_to if case else None,
            "handled": bool(case and case.handled)}


def _case_detail(w: World, case: Case) -> dict:
    inst = w.instance
    d = _summary(w, case.request, case)
    p = case.parse
    d["routing"] = case.routing
    d["timings"] = {k: round(v, 3) for k, v in case.timings.items()}
    d["parse"] = None if p is None else {
        "action": p.action.value, "refusal": p.refusal, "errors": p.errors,
        "output": p.output.model_dump() if p.output else None,
        # as recorded (a class named on the wrong day retargeted to the one on that day), else as read
        "constraints": [_constraint(w, c) for c in (case.constraints or p.constraints)]}
    pol = case.policy
    d["policy"] = None if pol is None else {
        "verdict": pol.verdict.value, "cited": [{"id": i, "cite": next(r.cite() for r in w.rules if r.id == i)}
                                                for i in pol.cited if any(r.id == i for r in w.rules)],
        "obligations": pol.obligations, "retrieved": pol.retrieved, "hallucinated": pol.hallucinated,
        "explanation": pol.explanation, "alternative": pol.alternative, "answer": pol.answer,
        "uncited_escalation": pol.uncited_escalation, "mode": getattr(w.policy, "mode", w.models["policy"])}
    o = case.outcome
    if o is None:
        d["outcome"] = None
    else:
        all_cons = {c.id: c for c in w.store.constraints(active_only=False)}
        d["outcome"] = {
            "status": o.status, "step": o.step, "rounds": o.rounds, "relaxed": o.relaxed,
            "mus_log": [[_constraint(w, all_cons[i]) if i in all_cons else {"id": i, "text": i} for i in mus]
                        for mus in o.mus_log],
            "messages": [_message(w, m, case) for m in o.messages],
            "replies": [r.model_dump(exclude_none=True) for r in o.replies],
            "concessions": [e.model_dump() | {"name": w.name_of(e.stakeholder)} for e in o.concessions],
            "notices": {k: [plain(t) for t in v] for k, v in o.notices.items()},
            "escalation": None if o.escalation is None else {"to": o.escalation.to, "to_name": w.name_of(
                o.escalation.to) if o.escalation.to in inst.faculty_by_id else o.escalation.to,
                "reason": plain(o.escalation.reason), "text": plain(o.escalation.text),
                # what granting would set aside, and why it cannot be granted (None: it can)
                "set_aside": [_constraint(w, c) for c in w.orch.overrides(case)],
                "cannot_grant": w.orch.cannot_grant(case)},
            "solver": None if o.result is None else {"status": o.result.status, "wall_time": o.result.wall_time,
                                                     "moved": o.result.moved,
                                                     "soft_violations": o.result.soft_violations}}
    if case.proposal:
        before = w.store.version(case.proposal.parent) if case.proposal.parent else None
        under = w.orch.builds_on(case)  # an earlier change still waiting: it is approved first
        d["proposal"] = {"version": case.proposal.version, "week": case.proposal.week,
                         "diff": _diff(w, before.assignment if before else {}, case.proposal.assignment),
                         "builds_on": under.id if under else None}
    else:
        d["proposal"] = None
    d["fairness"] = case.fairness
    d["decision"] = None if case.decision is None else {**case.decision, "by_name": w.name_of(case.decision["by"])}
    x = case.extra
    d["extra"] = None if x is None else {**x, "placement": _placement(w, Placement(**x["placement"]))
                                         if x.get("placement") else None}
    d["answer"] = None if case.handled is None else {**case.handled, "by_name": w.name_of(case.handled["by"])}
    d["reply"] = plain(case.reply)
    d["notices"] = {k: plain(v) for k, v in case.notices.items()}
    d["inbox"] = [_inbox_item(w, i) for i in w.inbox.items.values() if i.case_id == case.id]
    d["events"] = w.store.events(case.id)
    return d


def _sessions(w: World) -> dict[str, Session]:
    """The weekly sessions and every extra class (withdrawn ones too: old versions still name them)."""
    return {**w.instance.session_by_id, **{x.session.id: x.session for x in w.store.extras(active_only=False)}}


def _session_name(w: World, sid: str, sessions: dict[str, Session]) -> str:
    if not sid.startswith(EXTRA_PREFIX) or sid not in sessions:
        return session_name(w.instance, sid)
    s = sessions[sid]
    return f"the extra {w.instance.course_title(s.course)} {s.kind.value} ({sid})"


def _diff(w: World, before: dict[str, Placement], after: dict[str, Placement]) -> list[dict]:
    out = []
    sessions = _sessions(w)
    for sid in sorted(set(before) | set(after)):
        if before.get(sid) != after.get(sid):
            s = sessions.get(sid)
            out.append({"session": sid, "session_name": _session_name(w, sid, sessions), "extra": sid.startswith(EXTRA_PREFIX),
                        "faculty": s.faculty if s else None, "faculty_name": w.name_of(s.faculty) if s else None,
                        "groups": s.groups if s else [],
                        "before": _placement(w, before.get(sid)), "after": _placement(w, after.get(sid))})
    return out


def _inbox_item(w: World, i: InboxItem) -> dict:
    return {"id": i.id, "case": i.case_id, "to": i.to, "to_name": w.name_of(i.to),
            "created": datetime.fromtimestamp(i.created, UTC).isoformat(),
            "deadline": datetime.fromtimestamp(i.created + w.deadline, UTC).isoformat(),
            "message": _message(w, i.message, w.orch.cases.get(i.case_id) if i.case_id else None),
            "reply": i.reply.model_dump(exclude_none=True) if i.reply else None,
            "answered_by": i.answered_by}


# ---------------------------------------------------------------------------
# The inbox: everything addressed to one person, in one list
# ---------------------------------------------------------------------------

S = RequestStatus
_REQUEST_TITLE = {
    S.AWAITING_APPROVAL: "Your request is waiting for approval",
    S.CLARIFICATION: "A question about your request",
    S.DENIED: "Your request was declined",
    S.REFUSED: "Your request could not be accepted",
    S.ESCALATED: "Your request was sent up for a decision",
    S.ANSWERED: "An answer to your question",
    S.WITHDRAWN: "You withdrew your request",
}


def _event_time(at: str) -> str:
    """Events are logged in local time; inbox items are compared in UTC."""
    return datetime.fromisoformat(at).astimezone(UTC).isoformat()


def _feed(w: World, u: dict, decides, handles) -> list[dict]:
    """What ``u`` is told, newest first: negotiation messages, changes to their timetable, news of
    their own requests, and requests waiting on them. ``unread`` items have an event the person has not
    opened yet; ``needs_you`` ones wait for an answer or a decision."""
    me = u["id"]
    seen = w.store.seen(me)
    last: dict[str, dict] = {}
    settled: dict[str, dict] = {}  # case -> its last "published" or "withdrawn" event
    for e in w.store.events(limit=None):
        if e["case"]:
            last[e["case"]] = e
            if e["kind"] in (S.PUBLISHED.value, S.WITHDRAWN.value):
                settled[e["case"]] = e
    out: list[dict] = []

    def add(key: str, n: int, kind: str, case: Case | None, title: str, text: str, at: str, *,
            sender: str | None = None, needs_you: bool = False, **more) -> None:
        out.append({"id": key, "n": n, "kind": kind, "case": case.id if case else None, "title": title,
                    "text": plain(text), "at": at, "from": sender, "from_name": w.name_of(sender) if sender else None,
                    "status": case.status.value if case else None, "request": case.request.raw_text if case else None,
                    "needs_you": needs_you, "unread": needs_you or seen.get(key, 0) < n, **more})

    for i in w.inbox.items.values():
        if i.to != me and me != COORDINATOR:
            continue
        mine, n = i.to == me, 2 if i.reply else 1
        case = w.orch.cases.get(i.case_id) if i.case_id else None
        item = _inbox_item(w, i)
        choice = i.message.explanation_mode == "choice"
        add(f"msg:{i.id}", n, "negotiation", case,
            ("Choose a time for your request" if choice else "A clash with your timetable") if mine else
            (f"A time for {w.name_of(i.to)} to choose" if choice else f"Negotiation with {w.name_of(i.to)}"),
            i.message.text.split("\n")[0], item["created"], sender=COORDINATOR, needs_you=mine and i.reply is None,
            message=item)
        if not mine:
            out[-1]["unread"] = False  # the office follows these; they are not addressed to it

    for case in w.orch.cases.values():
        ev = last.get(case.id)
        if ev is None:
            continue
        at, n = _event_time(ev["at"]), ev["n"]
        own = case.request.sender_id == me
        done = settled.get(case.id)
        # someone else's request changed my timetable (or what I gave way for was withdrawn)
        if not own and me in case.notices and done is not None:
            withdrawn = done["kind"] == S.WITHDRAWN.value
            add(f"change:{case.id}", done["n"], "change", case,
                "A change you agreed to no longer applies" if withdrawn else "Your timetable has changed",
                case.notices[me], _event_time(done["at"]),
                sender=case.request.sender_id if withdrawn else done.get("approver") or COORDINATOR,
                version=done.get("version"))
        if own:
            status = case.status
            if status == S.PUBLISHED:
                version = done.get("version") if done else None
                if version is None:
                    title = "Nothing was changed" if done and done.get("unmet") else "Your request already fits the timetable"
                    text = case.reply
                else:
                    who = w.name_of(done.get("approver") or COORDINATOR)
                    title = "Your request was approved"
                    text = f"Approved by {who}; the timetable now includes it (version {version})."
                    if me in case.notices:
                        text += " " + case.notices[me]
                add(f"req:{case.id}", n, "request", case, title, text, at, sender=COORDINATOR, version=version)
            elif status == S.FORWARDED:
                answer = case.handled
                add(f"req:{case.id}", n, "request", case,
                    f"A reply from {w.name_of(answer['by'])}" if answer else "Your message was passed to the timetable office",
                    case.reply, at, sender=answer["by"] if answer else COORDINATOR)
            elif status in _REQUEST_TITLE and case.reply:
                add(f"req:{case.id}", n, "request", case, _REQUEST_TITLE[status], case.reply, at,
                    sender=(case.decision or {}).get("by") or COORDINATOR, needs_you=status == S.CLARIFICATION)
        # a refused request's reply says the coordinator has been informed: this is that notice
        if me == COORDINATOR and not own and case.status == S.REFUSED:
            add(f"refused:{case.id}", n, "alert", case, "A request was refused",
                f"Stopped before it reached the timetable: {case.parse.refusal}." if case.parse and case.parse.refusal
                else "Stopped before it reached the timetable.", at, sender=case.request.sender_id,
                **({"action": "reply"} if case.handled is None else {}))  # the office may still answer it
        # waiting on me
        if me == COORDINATOR and case.status == S.AWAITING_APPROVAL:
            add(f"approve:{case.id}", n, "action", case, "A change is waiting for your approval", case.request.raw_text,
                at, sender=case.request.sender_id, needs_you=True, action="approve")
        if case.status == S.ESCALATED and decides(u, case):
            add(f"decide:{case.id}", n, "action", case, "Escalated to you for a decision", case.request.raw_text, at,
                sender=case.request.sender_id, needs_you=True, action="decide")
        if handles(u, case):
            add(f"forward:{case.id}", n, "action", case, "A message passed to you for a reply", case.request.raw_text,
                at, sender=case.request.sender_id, needs_you=True, action="reply")
    return sorted(out, key=lambda x: x["at"], reverse=True)


def _reading(w: World, m: Message, r: Reply) -> str:
    """A typed reply in plain words, for the sender to confirm."""
    days = ", ".join(FULL_DAY.get(d, d) for d in r.counter_days or [])
    if r.decision == "accept":
        offer = next((o for o in m.offers if o.key == (r.choice or "").strip().upper()), None)
        where = "; ".join(f"{session_name(w.instance, s)} on {placement_text(w.instance, p)}"
                          for s, p in offer.placements.items()) if offer else ""
        return f"You accept option {r.choice}" + (f": {where}." if where else ".")
    if r.decision == "counter":
        slots = sorted(r.counter_slots or [])
        when = f" from {hour(slots[0])} to {hour(slots[-1] + 1)}" if slots else ""
        return f"None of the options works; you could do {days or 'other days'}{when} instead."
    if r.decision == "propose" and r.proposal:
        return f"You propose {placement_text(w.instance, r.proposal)} instead."
    if r.decision == "reject":
        return "You decline all the options" + (f" ({r.reason})." if r.reason else ".")
    if r.decision == "clarify":
        return f"You ask a question first: {r.question or r.text}"
    return "The system could not read this as an answer; it would go to the timetable coordinator" + (
        f" ({r.reason})." if r.reason else ".")


def _timetable(w: World, assignment: dict[str, Placement]) -> list[dict]:
    inst = w.instance
    rows = []
    extras = [x.session for x in w.store.extras(active_only=False) if x.session.id in assignment]
    for s in [*inst.sessions, *extras]:
        p = assignment.get(s.id)
        if p is None:
            continue
        rows.append({"session": s.id, "extra": s.id.startswith(EXTRA_PREFIX), "course": s.course, "title": inst.course_title(s.course), "kind": s.kind.value,
                     "faculty": s.faculty, "faculty_name": w.name_of(s.faculty), "groups": s.groups,
                     "group_names": [inst.group_by_id[g].name for g in s.groups], "duration": s.duration,
                     "day": p.day, "slot": p.slot, "room": p.room,
                     "room_name": inst.room_by_id[p.room].name if p.room in inst.room_by_id else p.room})
    return rows


def _pct(xs: list[float], q: float) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[min(len(xs) - 1, round(q / 100 * (len(xs) - 1)))]


_CACHE_MODEL: dict[str, str] = {}  # cache file -> model; cached answers never change, so each file is read once
_CACHE_LOCK = threading.Lock()


def _llm_usage() -> list[dict]:
    cache = ROOT / "runs" / "llm_cache"
    today = date.today().isoformat()
    out = []
    with _CACHE_LOCK:  # requests run in a thread pool
        if cache.exists():
            names = {f.name for f in cache.glob("*.json")}
            for name in names - _CACHE_MODEL.keys():
                try:
                    _CACHE_MODEL[name] = json.loads((cache / name).read_text(encoding="utf-8")).get("model", "?")
                except (OSError, ValueError):
                    continue
            for gone in _CACHE_MODEL.keys() - names:
                del _CACHE_MODEL[gone]
        counts = Counter(_CACHE_MODEL.values())
    models = set(counts) | {m for m in FREE_TIER if (cache / f"_quota-{m}-{today}.txt").exists()}
    for m in sorted(models):
        q = cache / f"_quota-{m}-{today}.txt"
        rpm, rpd = FREE_TIER.get(m, (None, None))
        out.append({"model": m, "used_today": int(q.read_text()) if q.exists() else 0, "rpd": rpd, "rpm": rpm,
                    "cached_responses": counts.get(m, 0)})
    return out


# The main negotiation experiment: three runs that together make the reduced plan (README, Interpretation layer)
MAIN_RUNS = ("negotiation-main-ours-4b.json", "negotiation-main-baselines-gemini.json", "negotiation-main-offline.json")
CONFIG_ORDER = ["ours-llm", "ours", "B4", "B3", "B2", "B1", "A1", "A2", "A3", "A1-offline", "oracle"]

# One row per model: (label, replies, parsing, policy, swaps) result files in runs/; None where not run
MODEL_ROWS = [
    ("Qwen3.5-4B fine-tuned · Q8_0", "replies-4b-nts-q8", "parsing-4b-nts-q8-paratest", "policy-4b-nts-q8-test", "swaps-4b-nts-q8"),
    ("Qwen3.5-4B fine-tuned · Q6_K", "replies-4b-nts-q6", "parsing-4b-nts-q6-paratest", "policy-4b-nts-q6-test", "swaps-4b-nts-q6"),
    ("Qwen3.5-2B fine-tuned v2 · Q8_0", "replies-2b-nts-q8", "parsing-2b-nts-q8-paratest", "policy-2b-nts-q8-test", "swaps-2b-nts-q8"),
    ("Qwen3.5-2B fine-tuned v1 · Q5_K_M", "replies-2b", "parsing-local-mt-paratest", "policy-local-mt-fix-test", None),
    ("Gemini 3.5 Flash-Lite (prompted)", "replies-gemini", "parsing-lite-paratest-full", "policy-test", None),
    ("Qwen3.5-4B base · Q8_0", "replies-4b-base", None, None, None),
    ("Qwen3.5-2B base · Q5_K_M", "replies-2b-base", None, None, None),
    ("Rules (keyword parser, rule swap reader)", "replies-rule", None, None, "swaps-rules-20260929"),
]


def _experiments() -> list[dict]:
    runs = ROOT / "runs"
    out = []

    def load(name: str):
        p = runs / name
        if not p.exists():
            return None, None
        return json.loads(p.read_text(encoding="utf-8")), datetime.fromtimestamp(p.stat().st_mtime, UTC).isoformat()

    parts = [(rep, at) for rep, at in map(load, MAIN_RUNS) if rep]
    if parts:
        configs: dict[str, dict] = {}
        paired: dict = {}
        models = []
        for rep, _ in parts:
            configs |= {k: v["summary"] for k, v in rep["configs"].items()}
            paired |= rep.get("paired") or {}
            if rep["models"].get("agent"):
                models.append({"configs": list(rep["configs"]), "model": rep["models"]["agent"]})
        extra, _ = load("negotiation-main-paired.json")
        paired |= extra or {}
        sim = next((rep["models"]["simulator"] for rep, _ in parts if rep["models"].get("simulator")), None)
        order = sorted(configs, key=lambda k: CONFIG_ORDER.index(k) if k in CONFIG_ORDER else len(CONFIG_ORDER))
        out.append({"id": "negotiation-main", "kind": "negotiation", "at": max(at for _, at in parts),
                    "title": "Main experiment: our system against the baselines on 60 scenarios",
                    "models": models, "simulator": sim,
                    "configs": {k: configs[k] for k in order}, "paired": paired})

    rows, latest = [], None
    for label, rep_f, parse_f, pol_f, swap_f in MODEL_ROWS:
        rep, a1 = load(f"{rep_f}.json") if rep_f else (None, None)
        par, a2 = load(f"{parse_f}.json") if parse_f else (None, None)
        pol, a3 = load(f"{pol_f}.json") if pol_f else (None, None)
        swp, a4 = load(f"{swap_f}.json") if swap_f else (None, None)
        if not (rep or par or pol or swp):
            continue
        latest = max(x for x in (latest, a1, a2, a3, a4) if x)
        r = (rep or {}).get("summary", rep or {})
        two = (par or {}).get("system_two", {})
        rows.append({
            "model": label,
            "reply_tool": r.get("tool_accuracy"), "reply_args": r.get("argument_accuracy"),
            "to_coordinator": r.get("to_coordinator"),
            "parse_action": two.get("action_accuracy"), "compile_exact": two.get("compile_exact_match"),
            "injections": f"{two['injections_handled']}/{two['injections_total']}" if two else None,
            "latency_p50": two.get("latency_s_p50"),
            "policy_allow_deny": (pol or {}).get("allow_deny_accuracy"), "deny_recall": (pol or {}).get("deny_recall"),
            "makeup_recall": (pol or {}).get("makeup_obligation_recall"),
            "swap_pairs": (swp or {}).get("pair_accuracy"), "swap_wrong": (swp or {}).get("wrong_commitments"),
        })
    if rows:
        out.append({"id": "models", "kind": "models", "at": latest, "rows": rows,
                    "title": "Models on the component benchmarks",
                    "subtitle": "Replies: 100 cases (Benchmark D). Parsing: 151 paraphrased test requests. "
                                "Policy: 109 test requests. Swaps: 120 requests (no model was trained on swaps)."})
    safety = sorted(runs.glob("safety-*.json"), key=lambda f: f.stat().st_mtime) if runs.exists() else []
    if safety:
        rep = json.loads(safety[-1].read_text(encoding="utf-8"))
        out.append({"id": "safety", "title": f"Safety set ({safety[-1].stem.split('-')[1]} parser)",
                    "at": datetime.fromtimestamp(safety[-1].stat().st_mtime, UTC).isoformat(), "kind": "safety",
                    "summary": {k: v for k, v in rep.items() if k != "rows"}})
    return out


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------


class WorldBody(BaseModel):
    name: str = ""
    preset: str | None = None  # "campus"
    filename: str | None = None
    data: str | None = None  # base64 of a course offering document


class LoginBody(BaseModel):
    person: str


class AskBody(BaseModel):
    text: str


class ConfirmBody(BaseModel):
    """The answer to the chat's "send this as a request?": ``{"n": 12, "as": "request" | "question" | "cancel"}``."""

    model_config = ConfigDict(populate_by_name=True)
    n: int
    as_: Literal["request", "question", "cancel"] = Field(alias="as")


# a request that has run its course: it can be taken off its sender's list ("Delete from list")
FINISHED = frozenset({RequestStatus.PUBLISHED, RequestStatus.REFUSED, RequestStatus.DENIED, RequestStatus.ANSWERED,
                      RequestStatus.FORWARDED, RequestStatus.WITHDRAWN})

# a message that is only a greeting or thanks: the chat says what it can do instead of filing it as a request
SMALL_TALK = re.compile(r"^\s*(hi+|hello+|hey+|hii+|good (morning|afternoon|evening)|thanks?( you)?|thank u|thx|"
                        r"ok(ay)?|cool|bye|great)\b[\s!.?,]*$", re.IGNORECASE)


class RequestBody(BaseModel):
    text: str
    makeup: bool = False  # "hold make-up classes": off, classes missed in an absence are cancelled


class FormRequestBody(BaseModel):
    """A request made with clicks: the request in words (what people read) and what the pipeline acts on:
    ``intent``, a cancellation, a move or a swap of one of the sender's classes; or ``intents``, several classes
    of a range of dates (the range builder): cancellations, or the sender away at their classes' times
    (``unavailable``, all in one week: one week is repaired per request) so the solver finds new times.
    ``makeup``: hold make-up classes (a range moved is always made up)."""

    text: str
    intent: dict | None = None
    intents: list[dict] | None = None
    makeup: bool = False


MAX_RANGE_WEEKS = 4  # the longest range of dates a request may cover, until the coordinator sets another


class SettingsBody(BaseModel):
    max_range_weeks: int = Field(ge=1, le=52)


class ReplyBody(BaseModel):
    decision: str
    choice: str | None = None
    counter_days: list[str] | None = None
    counter_slots: list[int] | None = None
    text: str = ""


class ReplyTextBody(BaseModel):
    text: str
    preview: bool = True


class RejectBody(BaseModel):
    reason: str = ""


class DecisionBody(BaseModel):
    grant: bool
    note: str = ""


class HandleBody(BaseModel):
    note: str


class ActBody(BaseModel):
    text: str = ""  # the request the office makes of it; the forwarded message itself when empty


class SeenBody(BaseModel):
    items: dict[str, int]  # inbox item -> the event of it now read


class AutopilotBody(BaseModel):
    person: str
    on: bool


class DocumentBody(BaseModel):
    name: str
    data: str  # base64 of the file


MAX_DOCUMENT_BYTES = 20 * 1024 * 1024


def _rule_view(r: Rule) -> dict:
    return {"id": r.id, "number": r.number, "title": r.title, "text": r.text, "source": r.source,
            "pages": list(r.pages), "method": r.method}


def create_web_app(world: World | None = None, *, worlds_dir: Path | None = None, restore: bool | None = None,
                   campus: bool = False) -> FastAPI:
    """``world``: the demo world (built when omitted). Worlds made from offering documents are kept
    in ``worlds_dir`` and restored at start (by default only when the demo world is built here, so tests
    that pass their own world start alone); ``campus`` also makes the demo campus if it is missing."""
    from .worlds import DEMO, Registry, WorldStarting, current_world

    registry = Registry(world or World(), worlds_dir, restore=world is None if restore is None else restore)
    if campus:
        registry.campus()
    app = FastAPI(title="Negotiate, Then Solve API", version="1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def which_world(request, call_next):
        """The world a request means: ``X-World`` (the demo when absent)."""
        token = current_world.set(request.headers.get("x-world") or "demo")
        try:
            return await call_next(request)
        finally:
            current_world.reset(token)

    def W() -> World:
        try:
            return registry.get()
        except WorldStarting as e:
            raise HTTPException(503, str(e)) from None

    def user(x_user: str = Header(...)) -> dict:
        w = W()
        ident = next((i for i in w.directory.by_email.values() if i.person == x_user), None)
        if ident is None:
            raise HTTPException(401, "unknown user")
        return {"id": x_user, "role": ident.role.value, "name": w.name_of(x_user)}

    def coordinator(u: dict = Depends(user)) -> dict:
        if u["id"] != COORDINATOR:
            raise HTTPException(403, "only the timetable coordinator may do this")
        return u

    def view(name: str):
        def check(u: dict = Depends(user)) -> dict:
            if u["role"] not in VIEWS[name]:
                raise HTTPException(403, "not available for your role")
            return u
        return check

    def visible(u: dict, case: Case | None, r: Request) -> bool:
        if u["id"] == COORDINATOR or u["role"] in ("hod", "dean"):
            return True
        if r.sender_id == u["id"]:
            return True
        return any(i.case_id == r.id and i.to == u["id"] for i in W().inbox.items.values())

    from .semester import register as register_semester

    def rebuild_from(doc, sizes) -> dict:
        """Give the current world the people, sections, rooms and weekly timetable of an offering document."""
        e = registry.rebuild(current_world.get(), doc, sizes)
        return {"world": e["id"], "status": e["status"]}

    register_semester(app, W, user, view, coordinator, rebuild_from)
    from .study import register as register_study

    def demo_world() -> World:
        try:
            return registry.get(DEMO)
        except WorldStarting as e:
            raise HTTPException(503, str(e)) from None

    register_study(app, demo_world, user, coordinator)

    # -- auth (mock) ------------------------------------------------------------------

    @app.get("/api/personas")
    def personas() -> list[dict]:
        w = W()
        return [{"id": i.person, "name": w.name_of(i.person), "role": i.role.value, "email": e}
                for e, i in sorted(w.directory.by_email.items(), key=lambda x: x[1].person)]

    @app.post("/api/login")
    def login(body: LoginBody) -> dict:
        u = user(body.person)
        return u | {"views": views_for(u["role"])}

    @app.get("/api/me")
    def me(u: dict = Depends(user)) -> dict:
        return u | {"views": views_for(u["role"])}

    # -- core ---------------------------------------------------------------------------

    @app.get("/api/overview")
    def overview(u: dict = Depends(user)) -> dict:
        w = W()
        reqs = w.store.requests()
        cases = w.orch.cases
        statuses = Counter((cases[r.id].status if r.id in cases else r.status).value for r in reqs)
        teaching = sorted({s.faculty for s in w.instance.sessions})
        ledger = ConcessionLedger(w.store.ledger())
        cur = w.store.current_version()
        outcomes = [c.outcome for c in cases.values() if c.outcome]
        return {
            "department": w.instance.name, "semester": SEMESTER, "seeding": w.seeding,
            "counts": dict(statuses), "total": len(reqs),
            "pending_approvals": len(w.orch.pending()),
            # escalations sent to this person and still open (the Approvals page lists them too)
            "my_decisions": sum(1 for c in cases.values() if c.status == RequestStatus.ESCALATED and decides(u, c)),
            # messages forwarded to the office and not answered yet (listed on the Approvals page)
            "my_forwarded": sum(1 for c in cases.values() if handles(u, c)),
            "my_inbox": sum(1 for i in _feed(w, u, decides, handles) if i["unread"]),
            # requests that wait on this person's answer (the Requests tab's count): their own asking a question
            # back, and others' sending them options
            "my_waiting": len({i.case_id for i in w.inbox.items.values() if i.to == u["id"] and i.reply is None
                               and i.case_id} | {r.id for r in reqs if r.sender_id == u["id"] and r.id in cases
                                                  and cases[r.id].status == RequestStatus.CLARIFICATION}),
            "published_version": cur.version if cur else None,
            "gini": ledger.gini(teaching, SEMESTER),
            "negotiations": {"agreed": sum(o.status == "agreed" for o in outcomes),
                             "escalated": sum(o.status == "escalated" for o in outcomes),
                             "feasible": sum(o.status == "feasible" for o in outcomes),
                             "mean_rounds": mean([o.rounds for o in outcomes if o.rounds])},
            "recent": w.store.events(limit=None)[-12:],
            "models": w.models, "sessions": len(w.instance.sessions), "faculty": len(w.instance.faculty),
            "model": None if w.model_link is None else {**w.model_link.status(), "mode": w.parser_mode},
            "rooms": len(w.instance.rooms), "groups": len(w.instance.groups),
            "test_run": w.test_run,
        }

    @app.get("/api/instance")
    def instance(u: dict = Depends(user)) -> dict:
        w = W()
        inst = w.instance
        return {"name": inst.name, "calendar": inst.calendar.model_dump(),
                "slots": [{"slot": i, "label": hour(i)} for i in range(inst.calendar.slots_per_day)],
                "day_names": {d: FULL_DAY[d] for d in inst.calendar.days},
                "faculty": [{"id": f.id, "name": f.name, "role": f.role.value, "reports_to": f.reports_to,
                             "email": w.directory.email_of(f.id)} for f in inst.faculty],
                "groups": [g.model_dump() for g in inst.groups],
                "rooms": [r.model_dump() for r in inst.rooms],
                "sessions": [{**s.model_dump(), "title": inst.course_title(s.course)} for s in inst.sessions],
                "policy": inst.policy.model_dump()}

    @app.post("/api/cases/{case_id}/clarify")
    def clarify(case_id: str, body: RequestBody, u: dict = Depends(user)) -> dict:
        """The sender's answer to the question their request was sent back with."""
        try:
            W().clarify(case_id, u["id"], body.text)
        except KeyError:
            raise HTTPException(404) from None
        except PermissionError as e:
            raise HTTPException(403, str(e)) from None
        except ValueError as e:
            raise HTTPException(409, str(e)) from None
        return {"id": case_id}

    @app.get("/api/weekly-changes")
    def weekly_changes(u: dict = Depends(view("history"))) -> list[dict]:
        """Changes published through requests in the weekly timetable, newest first: shown beside the
        semester plan, which they do not change (it has its own 85-minute grid)."""
        w = W()
        out = []
        for c in w.orch.cases.values():
            if c.status != RequestStatus.PUBLISHED:
                continue
            v = w.store.version(c.proposal.version) if c.proposal else None  # as published, with its approver
            parent = w.store.version(v.parent) if v and v.parent else None
            diff = _diff(w, parent.assignment, v.assignment) if v and parent else []
            out.append({"case": c.id, "sender_name": w.name_of(c.request.sender_id), "text": plain(c.request.raw_text),
                        "received_at": c.request.received_at.isoformat(), "version": v.version if v else None,
                        "week": v.week if v else (c.constraints[0].when.weeks[0] if c.constraints and c.constraints[0].when.weeks else None),
                        "moved": [d for d in diff if d["after"]], "cancelled": [d for d in diff if not d["after"]],
                        "decided_by": w.name_of(v.approved_by) if v and v.approved_by else None})
        out.sort(key=lambda x: (x["version"] or 0, x["received_at"]), reverse=True)
        return out

    @app.get("/api/calendar")
    def calendar(u: dict = Depends(user)) -> dict:
        """The semester's teaching weeks with their dates; for each, the days the signed-in person has
        classes (a teacher's own, a class rep's section's) in that week's timetable, the days a class of
        theirs is cancelled, and whether the week has changes of its own."""
        w = W()
        inst = w.instance
        start = date.fromisoformat(w.semester.state.semester_start)
        start -= timedelta(days=start.weekday())  # weeks run from Monday
        n_weeks = inst.calendar.weeks
        sessions = _sessions(w)  # extra classes are only in the week they were added to
        if u["id"].startswith("ST-"):
            mine = {s.id for s in sessions.values() if u["id"][3:] in s.groups}
        else:
            mine = {s.id for s in sessions.values() if s.faculty == u["id"]}
        semester = w.store.current_version()
        rows = []
        for n in range(1, n_weeks + 1):
            own = w.store.current_version(n)
            v = own or semester
            placed = v.assignment if v else {}
            cancelled = set(v.cancelled) & mine if v else set()
            classes = [{"day": p.day, "slot": p.slot, "time": hour(p.slot), "room": p.room, "session": sid,
                        "title": inst.course_title(sessions[sid].course), "extra": sid.startswith(EXTRA_PREFIX),
                        "kind": sessions[sid].kind.value, "cancelled": sid in cancelled}
                       for sid in sorted(mine)
                       if (p := placed.get(sid) or (semester.assignment.get(sid) if sid in cancelled and semester else None))]
            classes.sort(key=lambda c: (inst.calendar.days.index(c["day"]), c["slot"]))
            rows.append({"week": n, "monday": (start + timedelta(weeks=n - 1)).isoformat(), "classes": classes,
                         "own_changes": own is not None,
                         "class_days": sorted({placed[s].day for s in mine if s in placed},
                                              key=inst.calendar.days.index),
                         "cancelled_days": sorted({semester.assignment[s].day for s in cancelled
                                                   if semester and s in semester.assignment},
                                                  key=inst.calendar.days.index)})
        today = (date.today() - start).days // 7 + 1
        return {"days": inst.calendar.days, "day_names": {d: FULL_DAY[d] for d in inst.calendar.days},
                "current_week": min(max(today, 1), n_weeks), "weeks": rows}

    @app.post("/api/ask")
    def ask(body: AskBody, u: dict = Depends(user)) -> dict:
        """A question about the timetable ("my free slots this week", "when does Dr. Iyer teach ML?"), answered
        from the published timetable (agents.lookup, agents.questions) for anyone signed in. Nothing is stored:
        a question is not a request. Anything that is not a timetable question (a request, a question about the
        rules) comes back as ``not_a_question``, so the message can go the usual way. ``reader`` says who read the
        question: the model (the local ``lookup`` adapter), the regex, or the request words that ruled it out."""
        w = W()
        call, reader = w.read_question(body.text, u["id"])
        if call is None:
            return {"kind": "not_a_question", "reader": reader}
        return {"kind": "timetable", **_lookup_answer(w, call, u["id"], reader)}

    # -- the chat: one conversation per person, for questions and requests alike (CHAT-PLAN.md) --------------

    @app.get("/api/chat")
    def chat(after: int = 0, u: dict = Depends(user)) -> list[dict]:
        """The signed-in person's chat, oldest first (``after``: only messages after that number). A request shows
        only as a card with its number; what happens to it after (questions back, options, the outcome) is on
        its page in the Requests tab. Notice cards of earlier versions are left out."""
        return [m for m in W().store.chat(u["id"], after) if m["kind"] != "notice"]

    @app.post("/api/chat")
    def chat_send(body: RequestBody, u: dict = Depends(user)) -> list[dict]:
        """A message to the chat (CHAT-PLAN.md). In order:
        - a timetable question → answered (``kind: lookup``);
        - a question about the rules → answered from them, with every rule cited (``kind: rule``; no request);
        - a greeting or thanks → what the chat can do (``kind: other``);
        - anything else → "send this as a request?" (``kind: confirm``): nothing is sent until the person
          confirms (``POST /api/chat/confirm``).
        First, a follow-up ("and Friday?", "move my DS lecture there") is made to stand on its own from the last
        messages of today (agents.memory): what it was read as is kept with the person's message (``read_as``);
        when it leans on something the chat doesn't have, the reply asks what they mean.
        Returns the new messages: the person's, then the reply."""
        w = W()
        text, me = body.text.strip(), u["id"]
        if not text:
            raise HTTPException(422, "write a message first")
        chat = w.store.chat(me)
        follow = w.followup(text, me, turns_of(chat, today=date.today()))
        read = follow.text if follow and follow.changed else text
        mine = {"text": text} | ({"read_as": read} if read != text else {})
        last = next((m for m in reversed(chat) if m["role"] == "assistant"), None)
        if (follow and follow.changed and last and last["kind"] == "confirm" and not last.get("answered")
                and read.strip() == last["text"].strip()):  # "yes" to "send this as a request?": it is sent
            w.store.update_chat(me, last["n"], {"answered": "request"})
            return [w.store.add_chat(me, "you", "text", mine), _send_request(w, me, last)]
        if follow and follow.ask:
            reply = ("other", {"text": follow.ask})
        elif (q := w.read_question(read, me)) and q[0] is not None:
            reply = ("lookup", _lookup_answer(w, q[0], me, q[1]))
        elif (rule := w.answer_rule(read, me)) is not None:
            reply = ("rule", rule)
        elif SMALL_TALK.match(text):
            reply = ("other", {"text": "I can answer questions about the timetable (“am I free on Friday?”) and "
                                       "about the rules (“how many online classes are allowed?”), and make "
                                       "requests (“move my Monday lecture to Tuesday”)."})
        else:  # a request: sent as it was read ("move my DS lecture there" with "there" filled in)
            reply = ("confirm", {"text": read, "makeup": body.makeup, "reader": q[1]})
        return [w.store.add_chat(me, "you", "text", mine), w.store.add_chat(me, "assistant", reply[0], reply[1])]

    @app.post("/api/chat/confirm")
    def chat_confirm(body: ConfirmBody, u: dict = Depends(user)) -> list[dict]:
        """The answer to "send this as a request?" (message ``n``): ``request`` sends it (as ``POST /api/requests``)
        and the chat gets a request card; ``question`` reads it again as a timetable question, without the
        request-words check; ``cancel`` drops it. A confirm is answered once. Returns the new messages."""
        w = W()
        me = u["id"]
        m = w.store.chat_message(me, body.n)
        if m is None or m["kind"] != "confirm":
            raise HTTPException(404, "no such message to confirm")
        if m.get("answered"):
            raise HTTPException(409, f"already answered: {m['answered']}")
        w.store.update_chat(me, body.n, {"answered": body.as_})
        if body.as_ == "cancel":
            return []
        if body.as_ == "question":
            call, reader = w.read_question(m["text"], me, gate=False)
            if call is None:
                return [w.store.add_chat(me, "assistant", "other", {
                    "text": "I couldn't read that as a question about the timetable. Try asking it another way, "
                            "or send it as a request."})]
            return [w.store.add_chat(me, "assistant", "lookup", _lookup_answer(w, call, me, reader))]
        return [_send_request(w, me, m)]

    def _send_request(w: World, me: str, m: dict) -> dict:
        """The request a "send this as a request?" message (``m``) asked about, sent; its card."""
        try:
            r = w.intake.from_portal(me, m["text"], makeup=m.get("makeup"))
        except UnknownSender:
            raise HTTPException(403, "unknown user") from None
        if r is None:
            raise HTTPException(409, "duplicate request")
        # what was asked kept with the card: a follow-up ("has it been approved?") sees which request it was
        card = w.store.add_chat(me, "assistant", "request", {"case": r.id, "request_text": m["text"]})
        w.submit(r)
        return card

    @app.delete("/api/chat")
    def chat_clear(u: dict = Depends(user)) -> dict:
        """Empties the person's chat; their requests stay in the Requests list."""
        return {"cleared": W().store.clear_chat(u["id"])}

    def _lookup_answer(w: World, call, sender: str, reader: str) -> dict:
        """A timetable question's answer: the reply, the lookup, who read it, the part of the timetable to draw
        beside it (``view``, agents.views; None when there is nothing to draw), and for free hours the hours
        themselves (``free``, for the timetable to shade)."""
        from agents.more_lookups import Records

        week = WeekTimetable(w.instance, w.store, call.week)
        records = Records.from_world(w)  # requests, rules, bookings, other weeks: v3's lookups read them
        answerer = Answerer(w.instance, sender, records, sees_all=sender == COORDINATOR)

        def shown(c) -> dict:
            args = ({"parts": [shown(p) for p in c.args["parts"]]} if c.tool == "multi" else c.args)
            return {"tool": c.tool, "args": args, "week": c.week, "confident": c.confident}

        # several lookups (v3): the timetable beside the answer is the first part's
        first = call.args["parts"][0] if call.tool == "multi" else call
        first_week = week if first.week == call.week else WeekTimetable(w.instance, w.store, first.week)
        out = {"answer": answerer.answer(call, week), "reader": reader, "call": shown(call),
               "view": view_of(w.instance, first, first_week, sender)}
        if first.tool == "free_hours" and first.confident:
            out["free"] = [{"day": d, "slot": s} for d, s in free_hours_of(w.instance, first, first_week)]
        return out

    @app.post("/api/requests")
    def submit(body: RequestBody, u: dict = Depends(user)) -> dict:
        w = W()
        try:
            r = w.intake.from_portal(u["id"], body.text, makeup=body.makeup)
        except UnknownSender:
            raise HTTPException(403, "unknown user") from None
        if r is None:
            raise HTTPException(409, "duplicate request")
        # sent on purpose (the request builder): in the chat too, as what was asked and its request card
        w.store.add_chat(u["id"], "you", "text", {"text": body.text})
        w.store.add_chat(u["id"], "assistant", "request", {"case": r.id})
        w.submit(r)
        return {"id": r.id}

    @app.post("/api/requests/form")
    def submit_form(body: FormRequestBody, u: dict = Depends(user)) -> dict:
        """A request made with clicks. Its intent is taken as given, so nothing is left to a parser:
        it is checked here (one of the sender's own classes) and resolved like a parsed one."""
        w = W()
        given = body.intents if body.intents is not None else [body.intent] if body.intent is not None else []
        if not given:
            raise HTTPException(422, "nothing to request")
        try:
            out = IntentOutput(action="compile", intents=given)
        except ValidationError as e:
            raise HTTPException(422, f"not a request this form can make: {e.errors()[0]['msg']}") from None
        kinds = {i.kind for i in out.intents}
        if len(out.intents) == 1 and kinds <= {"cancel", "move", "swap"}:
            pass
        elif len(kinds) != 1 or not kinds <= {"cancel", "unavailable"}:
            raise HTTPException(422, "several classes are cancelled together, or moved together (unavailable)")
        now = w.orch.current_week() if w.orch.current_week else None
        weeks: set[int] = set()
        for intent in out.intents:
            try:
                asked = period_weeks(w.instance, intent.period, now) or []
            except Unresolved as e:
                raise HTTPException(422, e.question) from None
            if not asked and intent.kind == "unavailable":
                raise HTTPException(422, "say which week you are away")
            if now is not None and asked and asked[0] < now:
                raise HTTPException(422, f"Week {asked[0]} has passed; it is week {now} now.")
            weeks |= set(asked)
            if intent.kind == "unavailable":
                if intent.who is not None:
                    raise HTTPException(403, "only you can be away in your request")
                continue
            mine = intent.mine if intent.kind == "swap" else intent.which
            session = w.instance.session_by_id.get(mine.session or "")
            if session is None or session.faculty != u["id"]:
                raise HTTPException(403, "only your own classes can be cancelled, moved or swapped")
        if len(out.intents) > 1:
            longest = w.store.setting("max_range_weeks", MAX_RANGE_WEEKS)
            if weeks and max(weeks) - min(weeks) + 1 > longest:
                raise HTTPException(422, f"A request can cover at most {longest} week{'s' if longest != 1 else ''}.")
            if kinds == {"unavailable"} and len(weeks) > 1:
                raise HTTPException(422, "classes are moved one week per request")
        try:
            r = w.intake.from_portal(u["id"], body.text, makeup=body.makeup or kinds == {"unavailable"})
        except UnknownSender:
            raise HTTPException(403, "unknown user") from None
        if r is None:
            raise HTTPException(409, "duplicate request")
        # made from the timetable (the chat's popup): in the chat too, as what was asked and its request card
        w.store.add_chat(u["id"], "you", "text", {"text": body.text})
        w.store.add_chat(u["id"], "assistant", "request", {"case": r.id})
        w.submit(r, form=out)
        return {"id": r.id}

    @app.get("/api/cases")
    def cases(scope: str = "mine", u: dict = Depends(user)) -> list[dict]:
        """``mine``: the person's own requests, without those they took off their list, and others' requests that
        wait on their answer (options sent to them); each with whether they may withdraw it and whether it waits
        on them (``waits_on_you``: a question back, or options to choose from). ``all``: every request they may
        see."""
        w = W()
        out = []
        me = u["id"]
        hidden = w.store.hidden(me) if scope == "mine" else set()
        asked = {i.case_id for i in w.inbox.items.values() if i.to == me and i.reply is None and i.case_id}
        for r in reversed(w.store.requests()):
            case = w.orch.cases.get(r.id)
            if scope == "mine" and ((r.sender_id != me and r.id not in asked) or r.id in hidden):
                continue
            if visible(u, case, r) or r.id in asked:
                s = _summary(w, r, case)
                out.append(s | ({"can_withdraw": bool(case and withdraws(w, u, case)),
                                 "waits_on_you": r.id in asked or (r.sender_id == me
                                                                    and s["status"] == "clarification_requested")}
                                if scope == "mine" else {}))
        return out

    @app.post("/api/cases/{case_id}/hide")
    def hide_case(case_id: str, u: dict = Depends(user)) -> dict:
        """"Delete from list": takes one of the person's own finished requests off their list. It stays in the
        records (and on the oversight pages); only their list changes."""
        w = W()
        r = w.store.request(case_id)
        if r is None:
            raise HTTPException(404)
        if r.sender_id != u["id"]:
            raise HTTPException(403, "only your own requests can be taken off your list")
        case = w.orch.cases.get(case_id)
        status = case.status if case else r.status
        if status not in FINISHED:
            raise HTTPException(409, f"it is not finished yet ({status.value}): withdraw it first")
        w.store.hide(u["id"], case_id)
        return {"id": case_id, "hidden": True}

    @app.get("/api/cases/{case_id}")
    def case(case_id: str, u: dict = Depends(user)) -> dict:
        w = W()
        r = w.store.request(case_id)
        if r is None:
            raise HTTPException(404)
        c = w.orch.cases.get(case_id)
        if not visible(u, c, r):
            raise HTTPException(403)
        if c is None:
            return {**_summary(w, r, None), "events": w.store.events(case_id)}
        return _case_detail(w, c) | {"can_decide": c.status == RequestStatus.ESCALATED and decides(u, c),
                                     "can_handle": handles(u, c), "can_withdraw": withdraws(w, u, c)}

    def withdraws(w: World, u: dict, case: Case) -> bool:
        """The sender, while nothing the request asked for is in the published timetable."""
        running = case.id in w.threads and w.threads[case.id].is_alive()
        return u["id"] == case.request.sender_id and not running and w.orch.withdrawable(case) is None

    def handles(u: dict, case: Case) -> bool:
        """A message forwarded to the coordinator, or refused, still unanswered, and this is the coordinator."""
        return u["id"] == COORDINATOR and W().orch.office_answers(case) and case.handled is None

    def decides(u: dict, case: Case) -> bool:
        """The person an escalation was sent to: "coordinator", "dean", "hod", or the heads by id."""
        to = case.outcome.escalation.to if case.outcome and case.outcome.escalation else ""
        if to == "coordinator":
            return u["id"] == COORDINATOR
        if to in ("dean", "hod"):
            return u["role"] == to
        return u["id"] in to.split(", ")

    @app.post("/api/cases/{case_id}/decide")
    def decide(case_id: str, body: DecisionBody, u: dict = Depends(user)) -> dict:
        """Grant or decline an escalated request; granting re-solves it and then it waits for approval."""
        w = W()
        c = w.orch.cases.get(case_id)
        if c is None:
            raise HTTPException(404)
        if c.status != RequestStatus.ESCALATED:
            raise HTTPException(409, f"this request is not waiting for a decision ({c.status.value})")
        if not decides(u, c):
            raise HTTPException(403, "this decision was sent to someone else")
        if body.grant and (why := w.orch.cannot_grant(c)):
            raise HTTPException(409, why)
        w.decide(case_id, u["id"], body.grant, body.note.strip())
        return {"status": "deciding"}

    @app.post("/api/cases/{case_id}/handle")
    def handle(case_id: str, body: HandleBody, u: dict = Depends(user)) -> dict:
        """The coordinator answers a message forwarded to them; the answer becomes the sender's reply."""
        w = W()
        c = w.orch.cases.get(case_id)
        if c is None:
            raise HTTPException(404)
        if u["id"] != COORDINATOR:
            raise HTTPException(403, "only the timetable office answers forwarded messages")
        try:
            w.orch.handle(case_id, u["id"], body.note, name=w.name_of(u["id"]))
        except ValueError as e:
            raise HTTPException(409 if body.note.strip() else 422, str(e)) from None
        return _case_detail(w, c)

    @app.post("/api/cases/{case_id}/act")
    def act(case_id: str, body: ActBody, u: dict = Depends(user)) -> dict:
        """The coordinator acts on a forwarded message: ``text`` (by default the message itself) becomes a
        request of theirs, solved and negotiated like any other; the sender is told which request it is."""
        w = W()
        c = w.orch.cases.get(case_id)
        if c is None:
            raise HTTPException(404)
        if u["id"] != COORDINATOR:
            raise HTTPException(403, "only the timetable office acts on forwarded messages")
        if not w.orch.office_answers(c) or c.handled is not None:
            raise HTTPException(409, "this message is not waiting for the timetable office")
        r = w.intake.from_portal(u["id"], (body.text or c.request.raw_text).strip())
        if r is None:
            raise HTTPException(409, "duplicate request")
        w.orch.act_on(case_id, u["id"], r)
        w.submit(r)
        return {"id": r.id, "case": _case_detail(w, c)}

    @app.post("/api/cases/{case_id}/withdraw")
    def withdraw(case_id: str, u: dict = Depends(user)) -> dict:
        """The sender takes a request back; everything it set in motion is undone."""
        w = W()
        c = w.orch.cases.get(case_id)
        if c is None:
            raise HTTPException(404)
        if u["id"] != c.request.sender_id:
            raise HTTPException(403, "only the sender can withdraw a request")
        if c.id in w.threads and w.threads[c.id].is_alive():
            raise HTTPException(409, "it is still being worked on; try again when it stops")
        try:
            w.orch.withdraw(case_id, u["id"])
        except ValueError as e:
            raise HTTPException(409, str(e)) from None
        return _case_detail(w, c)

    @app.get("/api/events")
    def events(after: int = 0, case: str | None = None, limit: int = 200, u: dict = Depends(user)) -> list[dict]:
        w = W()
        rows = w.store.events(case, after=after, limit=limit)
        if u["role"] in OVERSIGHT:
            return rows
        mine = {r.id for r in w.store.requests() if r.sender_id == u["id"]}
        mine |= {i.case_id for i in w.inbox.items.values() if i.to == u["id"]}
        return [e for e in rows if e.get("case") in mine]

    # -- negotiation inbox ------------------------------------------------------------------------

    @app.get("/api/inbox")
    def inbox(u: dict = Depends(user)) -> list[dict]:
        w = W()
        items = [i for i in w.inbox.items.values() if i.to == u["id"] or u["id"] == COORDINATOR]
        return [_inbox_item(w, i) for i in sorted(items, key=lambda i: -i.created)]

    @app.get("/api/feed")
    def feed(u: dict = Depends(user)) -> list[dict]:
        """Everything addressed to this person: the one place they hear from the timetable office."""
        return _feed(W(), u, decides, handles)

    @app.post("/api/feed/seen")
    def feed_seen(body: SeenBody, u: dict = Depends(user)) -> dict:
        W().store.mark_seen(u["id"], body.items)
        return {"ok": True}

    @app.post("/api/inbox/{item_id}/reply")
    def reply(item_id: str, body: ReplyBody, u: dict = Depends(user)) -> dict:
        w = W()
        item = w.inbox.items.get(item_id)
        if item is None:
            raise HTTPException(404)
        if item.to != u["id"]:
            raise HTTPException(403, "this message is addressed to someone else")
        if body.decision not in ("accept", "reject", "counter"):
            raise HTTPException(422, "decision must be accept, reject or counter")
        try:
            w.inbox.answer(item_id, Reply(**body.model_dump()), u["id"])
        except ValueError as e:
            raise HTTPException(409, str(e)) from None
        return _inbox_item(w, item)

    @app.post("/api/inbox/{item_id}/reply-text")
    def reply_text(item_id: str, body: ReplyTextBody, u: dict = Depends(user)) -> dict:
        """A reply in the person's own words, read into one tool call by the
        reply parser. ``preview`` returns the reading without sending it."""
        w = W()
        item = w.inbox.items.get(item_id)
        if item is None:
            raise HTTPException(404)
        if item.to != u["id"]:
            raise HTTPException(403, "this message is addressed to someone else")
        text = body.text.strip()
        if not text:
            raise HTTPException(422, "write a reply first")
        try:
            got = w.reply_parser.parse(item.message, text)
        except LLMError as e:
            raise HTTPException(503, f"the reply model is unavailable: {e}") from None
        got = got.model_copy(update={"text": text})
        view = {"reply": got.model_dump(exclude_none=True), "reading": _reading(w, item.message, got),
                "parser": w.models.get("replies")}
        if body.preview:
            return view
        try:
            w.inbox.answer(item_id, got, u["id"])
        except ValueError as e:
            raise HTTPException(409, str(e)) from None
        return view | {"item": _inbox_item(w, item)}

    @app.post("/api/inbox/{item_id}/simulate")
    def simulate(item_id: str, u: dict = Depends(user)) -> dict:
        w = W()
        item = w.inbox.items.get(item_id)
        if item is None:
            raise HTTPException(404)
        if u["id"] not in (item.to, COORDINATOR):
            raise HTTPException(403)
        try:
            w.inbox.answer(item_id, w.simulators[item.to].respond(item.message), "simulator")
        except ValueError as e:
            raise HTTPException(409, str(e)) from None
        return _inbox_item(w, item)

    @app.get("/api/settings")
    def settings(u: dict = Depends(user)) -> dict:
        """What the coordinator set for everyone: the longest range of dates a request may cover."""
        return {"max_range_weeks": W().store.setting("max_range_weeks", MAX_RANGE_WEEKS)}

    @app.post("/api/settings")
    def set_settings(body: SettingsBody, u: dict = Depends(view("settings"))) -> dict:
        W().store.set_setting("max_range_weeks", body.max_range_weeks)
        return {"max_range_weeks": body.max_range_weeks}

    @app.get("/api/autopilot")
    def autopilot(u: dict = Depends(user)) -> dict:
        w = W()
        return {f.id: w.autopilot.get(f.id, False) for f in w.instance.faculty}

    @app.post("/api/autopilot")
    def set_autopilot(body: AutopilotBody, u: dict = Depends(user)) -> dict:
        w = W()
        if u["id"] not in (body.person, COORDINATOR):
            raise HTTPException(403)
        w.autopilot[body.person] = body.on
        return {body.person: body.on}

    # -- approvals and versions -------------------------------------------------------------------

    @app.get("/api/approvals")
    def approvals(u: dict = Depends(view("approvals"))) -> list[dict]:
        w = W()
        return [_case_detail(w, c) for c in w.orch.pending()]

    @app.post("/api/approvals/{case_id}/approve")
    def approve(case_id: str, u: dict = Depends(user)) -> dict:
        w = W()
        try:
            v = w.orch.approve(case_id, u["id"])
        except NotAuthorised as e:
            raise HTTPException(403, str(e)) from None
        except (KeyError, ValueError) as e:
            raise HTTPException(409, str(e)) from None
        return {"version": v.version, "notified": w.orch.cases[case_id].notices}

    @app.post("/api/approvals/{case_id}/reject")
    def reject(case_id: str, body: RejectBody, u: dict = Depends(user)) -> dict:
        w = W()
        try:
            w.orch.reject(case_id, u["id"], body.reason)
        except NotAuthorised as e:
            raise HTTPException(403, str(e)) from None
        return {"status": w.orch.cases[case_id].status.value}

    @app.get("/api/versions")
    def versions(u: dict = Depends(view("history"))) -> list[dict]:
        w = W()
        rows = w.store.versions()
        # a semester change carried into a week with its own repair ("carry-<version>") is part of that
        # change, not a change of its own: it is listed under the version it came from
        carried: dict[int, list[dict]] = {}
        for v in rows:
            if (v["case"] or "").startswith("carry-"):
                carried.setdefault(int(v["case"].removeprefix("carry-")), []).append(
                    {"version": v["version"], "week": v["week"]})
        return [{**v, "approved_by_name": w.name_of(v["approved_by"]) if v["approved_by"] else None,
                 "carried": carried.get(v["version"], [])}
                for v in reversed(rows) if not (v["case"] or "").startswith("carry-")]

    @app.get("/api/timetable")
    def timetable(version: int | None = None, week: int | None = None, u: dict = Depends(user)) -> dict:
        w = W()
        if version and u["role"] not in VIEWS["history"]:
            raise HTTPException(403, "only the timetable office sees earlier and proposed versions")
        v = w.store.version(version) if version else (w.store.current_version(week) or w.store.current_version())
        if v is None:
            raise HTTPException(404)
        parent = w.store.version(v.parent) if v.parent else None
        changed = {d["session"] for d in _diff(w, parent.assignment, v.assignment)} if parent else set()
        # weeks with their own repair (an absence, a room closed): everyone can look at them
        weeks = sorted({x["week"] for x in w.store.versions() if x["published"] and x["week"] is not None})
        cancelled = [{"session": sid, "session_name": session_name(w.instance, sid),
                      "faculty_name": w.name_of(w.instance.session_by_id[sid].faculty),
                      "groups": w.instance.session_by_id[sid].groups} for sid in v.cancelled]
        return {"version": v.version, "week": v.week, "approved_by": v.approved_by, "parent": v.parent,
                "entries": _timetable(w, v.assignment), "changed": sorted(changed), "weeks": weeks,
                "cancelled": cancelled}

    @app.get("/api/versions/diff")
    def diff(a: int, b: int, u: dict = Depends(view("history"))) -> list[dict]:
        w = W()
        va, vb = w.store.version(a), w.store.version(b)
        if va is None or vb is None:
            raise HTTPException(404)
        return _diff(w, va.assignment, vb.assignment)

    @app.post("/api/versions/{version}/rollback")
    def rollback(version: int, u: dict = Depends(coordinator)) -> dict:
        w = W()
        v = w.store.rollback(version, u["id"])
        w.store.log(None, "rollback", to=version, version=v.version, approver=u["id"])
        if v.week is None:
            w.orch.carry_into_weeks(v, u["id"])
        return {"version": v.version}

    # -- transparency -------------------------------------------------------------------------------

    @app.get("/api/ledger")
    def ledger(u: dict = Depends(view("fairness"))) -> dict:
        w = W()
        entries = w.store.ledger()
        led = ConcessionLedger(entries)
        teaching = sorted({s.faculty for s in w.instance.sessions})
        known = {c.id: c for c in w.store.constraints(active_only=False)}
        # the semester's Gini just after each entry, so the page can show how each concession moved it
        after = [ConcessionLedger(entries[:i + 1]).gini(teaching, e.semester) for i, e in enumerate(entries)]
        return {"semester": SEMESTER, "gini": led.gini(teaching, SEMESTER), "decay": led.decay,
                "entries": [e.model_dump() | {"name": w.name_of(e.stakeholder), "gini_after": g,
                                              "case": known[e.constraint_id].source.request
                                              if e.constraint_id in known else None}
                            for e, g in zip(entries, after, strict=True)],
                "stakeholders": [{"id": f, "name": w.name_of(f), "credit": led.credit(f, SEMESTER),
                                  "concessions": led.times().get(f, 0),
                                  "burden": led.counts().get(f, 0.0)} for f in teaching]}

    @app.get("/api/graph")
    def graph(u: dict = Depends(view("graph"))) -> dict:
        w = W()
        teaching = sorted({s.faculty for s in w.instance.sessions})
        led = ConcessionLedger(w.store.ledger())
        cons = [c for c in w.store.constraints() if c.tier >= Tier.COMMITMENT or c.owner]
        g = build_graph(w.instance, cons, {f: led.credit(f, SEMESTER) for f in teaching})
        for c in w.orch.cases.values():
            for mus in (c.outcome.mus_log if c.outcome else []):
                add_conflict(g, [i for i in mus if i in g])
        nodes = [{"id": n, **{k: v for k, v in d.items()}, "label": w.name_of(n) if d.get("kind") == "stakeholder"
                  else n} for n, d in g.nodes(data=True)]
        edges = [{"source": u_, "target": v, "rel": d.get("rel")} for u_, v, d in g.edges(data=True)]
        return {"nodes": nodes, "edges": edges}

    @app.get("/api/handbook")
    def handbook(u: dict = Depends(user)) -> list[dict]:
        return [_rule_view(r) for r in W().rules]

    @app.get("/api/handbook/documents")
    def documents(u: dict = Depends(view("documents"))) -> dict:
        w = W()
        return {"ocr": {"backend": w.ocr.name, "model": getattr(w.ocr.backend, "model", w.ocr.name)},
                "accepts": sorted(SUFFIXES), "documents": [d.summary() for d in w.documents]}

    @app.post("/api/handbook/documents")
    def upload_document(body: DocumentBody, u: dict = Depends(coordinator)) -> dict:
        import base64
        import binascii

        w = W()
        name = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(body.name).name).strip(" .")
        if not name or Path(name).suffix.lower() not in SUFFIXES or name == "handbook.md":
            raise HTTPException(422, f"unsupported file; accepted: {', '.join(sorted(SUFFIXES))}")
        try:
            data = base64.b64decode(body.data, validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(422, "data must be base64") from None
        if len(data) > MAX_DOCUMENT_BYTES:
            raise HTTPException(413, "file larger than 20 MB")
        w.policy_dir.mkdir(parents=True, exist_ok=True)
        path = w.policy_dir / name
        path.write_bytes(data)
        report = ingest(path, w.ocr)
        if report.error or not report.rules:
            path.unlink(missing_ok=True)
            raise HTTPException(422, report.error or "no text found in the document")
        w.reload_policies()
        w.store.log(None, "policy_document", name=name, rules=[r.id for r in report.rules], by=u["id"])
        return report.summary() | {"rules": [_rule_view(r) for r in w.rules if r.source == name]}

    @app.delete("/api/handbook/documents/{name}")
    def delete_document(name: str, u: dict = Depends(coordinator)) -> dict:
        w = W()
        path = w.policy_dir / Path(name).name
        if not path.is_file() or path.name == "handbook.md":
            raise HTTPException(404)
        path.unlink()
        w.reload_policies()
        return {"removed": name}

    @app.get("/api/handbook/search")
    def search(q: str, k: int = 5, mode: str | None = None, u: dict = Depends(user)) -> dict:
        """The policy agent's retriever by default; ``mode`` compares bm25, dense or hybrid."""
        w = W()
        kind = mode or w.retriever_kind
        if kind not in RETRIEVERS:
            raise HTTPException(422, f"mode must be one of {', '.join(RETRIEVERS)}")
        retriever = w.policy.retriever if kind == w.retriever_kind else make_retriever(w.rules, kind)
        scores = retriever.scores(q)
        ranked = sorted(zip(scores, w.rules), key=lambda x: -x[0])[:k]
        return {"query": q, "tokens": tokens(q), "mode": kind,
                "results": [_rule_view(r) | {"score": round(s, 4)} for s, r in ranked if s > 0]}

    @app.get("/api/transparency")
    def transparency(u: dict = Depends(user)) -> dict:
        w = W()
        return {
            "tiers": [{"tier": t.value, "name": TIER_NAME[t], "who": TIER_WHO[t.value],
                       "examples": TIER_EXAMPLES[t.value], "relaxable_in_negotiation": t.value in (3, 4)}
                      for t in Tier],
            "weights": Weights().__dict__, "role_authority": {r.value: v for r, v in ROLE_AUTHORITY.items()},
            "justification": {j.value: v for j, v in JUSTIFICATION.items()},
            "ladder": [
                {"step": 1, "name": "Auto-substitute", "text": "If the hard constraints fit, repair the timetable "
                 "moving as little as possible and notify those affected."},
                {"step": 2, "name": "Auto-relax", "text": "Preferences that cannot be kept are dropped by the "
                 "objective; their owners are told, and can send the request again as a requirement."},
                {"step": 3, "name": "Negotiate", "text": "Find the minimal conflict, the cheapest minimal corrections "
                 "(Tier 3-4 only), and ask the owner of the cheapest with 2-3 solver-verified alternatives."},
                {"step": 4, "name": "Escalate", "text": "On deadlock, no reply, or a Tier 0-2 conflict, brief the "
                 "person with authority along the reporting chain."}],
            "negotiation": {"max_rounds": 6, "options_per_message": 3, "asks_per_owner": 2,
                            "reply_deadline_s": w.deadline, "ledger_decay_per_semester": 0.5},
            "system_one": w.system_one_info, "models": w.models,
            "safety": ["Nothing is published without the coordinator's approval.",
                       "Authority is checked by code before any constraint is created; the model cannot override it.",
                       "Message text is data: models return structured data and code decides what happens.",
                       "Explanations name constraints, never private reasons; leaks are measured.",
                       "Tier 0-2 constraints are never relaxed without a named person with authority."],
        }

    @app.get("/api/observability")
    def observability(u: dict = Depends(view("health"))) -> dict:
        w = W()
        cases = list(w.orch.cases.values())
        timing = {k: [c.timings[k] for c in cases if k in c.timings] for k in ("parse", "policy", "solve_and_negotiate")}
        events = w.store.events()
        solver = [e.get("solver_seconds") for e in events if e["kind"] == "solved" and e.get("solver_seconds")]
        msgs = [m for c in cases if c.outcome for m in c.outcome.messages]
        routes = Counter(c.route for c in cases)
        s1 = [c.routing for c in cases if c.routing]
        return {
            "uptime_s": (datetime.now(UTC) - w.started).total_seconds(),
            "models": w.models, "llm": _llm_usage(),
            "stages": {k: {"n": len(v), "p50": _pct(v, 50), "p95": _pct(v, 95), "max": max(v) if v else None}
                       for k, v in timing.items()},
            "solver": {"n": len(solver), "p50": _pct(solver, 50), "p95": _pct(solver, 95)},
            "routing": {"counts": dict(routes), "tau": w.tau,
                        "system_one_latency_ms_p50": _pct([r["latency_ms"] for r in s1], 50),
                        "fast_path_share": mean([r["fast_path"] for r in s1])},
            "explanations": {"messages": len(msgs), "faithfulness": mean([m.faithfulness for m in msgs]),
                             "leaks": sum(len(m.leaks) for m in msgs),
                             "rejected_claims": sum(m.rejected_claims for m in msgs)},
            "events_by_kind": dict(Counter(e["kind"] for e in events)),
            "errors": [e for e in events if e["kind"] == "error"][-10:],
            "safety": {"refused": sum(c.status == RequestStatus.REFUSED for c in cases),
                       "denied": sum(c.status == RequestStatus.DENIED for c in cases),
                       "unapproved_publishes": 0 if all(e.get("approver") == COORDINATOR for e in events
                                                        if e["kind"] == "published") else 1,
                       "leaks": sum(len(m.leaks) for m in msgs)},
            "threads_running": sum(t.is_alive() for t in w.threads.values()),
        }

    @app.get("/api/experiments")
    def experiments(u: dict = Depends(view("experiments"))) -> list[dict]:
        return _experiments()

    @app.post("/api/demo/reset")
    def reset(u: dict = Depends(coordinator)) -> dict:
        """Put the current world back to its demo state, as it stood when the start-up history had
        finished replaying; nothing is replayed (study answers are kept)."""
        try:
            registry.restore(current_world.get())
        except ValueError as e:
            raise HTTPException(409, str(e)) from None
        return {"ok": True}

    # -- worlds ---------------------------------------------------------------------------------

    @app.get("/api/worlds")
    def worlds() -> list[dict]:
        """Every world, for the sign-in page (before anyone is signed in) and the office's switcher."""
        return registry.list()

    @app.post("/api/worlds")
    def new_world(body: WorldBody, u: dict = Depends(coordinator)) -> dict:
        from semester.offerings import parse_offerings

        if body.preset == "campus":
            e = registry.campus()
        else:
            if not body.data:
                raise HTTPException(422, "give a course offering document, or preset: campus")
            folder = registry.folder / "uploads"
            folder.mkdir(parents=True, exist_ok=True)
            name = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(body.filename or "offerings.pdf").name)
            path = folder / name
            try:
                path.write_bytes(base64.b64decode(body.data, validate=True))
            except (binascii.Error, ValueError):
                raise HTTPException(422, "data must be base64") from None
            doc = parse_offerings(path)
            if not doc.courses or not doc.cohorts:
                raise HTTPException(422, "no courses and programmes found in that document")
            e = registry.create(body.name or doc.source or name, doc, source=name)
        return {k: e[k] for k in ("id", "name", "status")}

    @app.delete("/api/worlds/{world_id}")
    def delete_world(world_id: str, u: dict = Depends(coordinator)) -> dict:
        try:
            registry.delete(world_id)
        except ValueError as e:
            raise HTTPException(409, str(e)) from None
        return {"deleted": world_id}

    dist = ROOT / "frontend" / "dist"
    if dist.exists():
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles

        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}")
        def spa(path: str) -> FileResponse:
            f = dist / path
            if path and f.is_file():
                return FileResponse(f)
            # the page names this build's hashed chunks, so it must never be served stale:
            # an old copy asks for chunks a rebuild has deleted (404s on lazy pages)
            return FileResponse(dist / "index.html", headers={"Cache-Control": "no-cache"})

    return app


def _newest(paths: list[Path]) -> float:
    """Latest modification time among the given files and everything under the given folders."""
    times = [0.0]
    for p in paths:
        if p.is_dir():
            times += [f.stat().st_mtime for f in p.rglob("*") if f.is_file()]
        elif p.exists():
            times.append(p.stat().st_mtime)
    return max(times)


def build_frontend(force: bool = False) -> None:
    """Build frontend/dist with npm when it is missing or older than the sources (npm install first if needed)."""
    import shutil
    import subprocess
    import sys

    fe = ROOT / "frontend"
    if not (fe / "package.json").exists():
        return
    built = fe / "dist" / "index.html"
    sources = [fe / "src", fe / "public", fe / "index.html", fe / "package.json", fe / "package-lock.json",
               fe / "vite.config.ts", fe / "tsconfig.json"]
    if not force and built.exists() and built.stat().st_mtime >= _newest(sources):
        return
    npm = shutil.which("npm")
    if npm is None:
        print("nts-web: npm not found, cannot build the front end (install Node.js, or pass --no-build)"
              + ("; serving the existing build" if built.exists() else "; only /api will be served"), file=sys.stderr)
        return
    # npm writes node_modules/.package-lock.json on every install: older than package-lock.json means stale packages
    installed = fe / "node_modules" / ".package-lock.json"
    lock = fe / "package-lock.json"
    steps = []
    if not installed.exists() or (lock.exists() and lock.stat().st_mtime > installed.stat().st_mtime):
        steps.append([npm, "install"])
    steps.append([npm, "run", "build"])
    for cmd in steps:
        print(f"nts-web: {' '.join(cmd[1:])} in frontend/ ...", flush=True)
        if subprocess.run(cmd, cwd=fe).returncode != 0:
            sys.exit(f"nts-web: 'npm {' '.join(cmd[1:])}' failed (fix the error above, or pass --no-build)")


def run() -> None:
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(prog="nts-web", description="API + built UI (flags override the NTS_* variables)")
    ap.add_argument("--parser", choices=["offline", "gemini", "local"], help="NTS_PARSER")
    ap.add_argument("--test-data", type=int, metavar="N", nargs="?", const=20,
                    help="benchmark department, replaying N held-out test requests (default 20) instead of the demo history")
    build = ap.add_mutually_exclusive_group()
    build.add_argument("--no-build", action="store_true",
                       help="do not build the front end at start-up (NTS_BUILD=0); serve frontend/dist as it is")
    build.add_argument("--rebuild", action="store_true", help="rebuild the front end even if it looks up to date")
    args = ap.parse_args()
    # the UI is built only when its sources changed since the last build (npm install too when packages changed)
    if not args.no_build and os.environ.get("NTS_BUILD") != "0":
        build_frontend(force=args.rebuild)
    if args.parser:
        os.environ["NTS_PARSER"] = args.parser
    if args.test_data is not None:
        os.environ["NTS_TEST_DATA"] = str(args.test_data)
    # repeated solves (the start-up history, a demo run again) replay from disk; NTS_SOLVE_CACHE=0 solves afresh
    from core.solver import use_solve_cache

    use_solve_cache(None if os.environ.get("NTS_SOLVE_CACHE") == "0" else ROOT / "runs" / "solve_cache")
    # each world's state survives a restart (runs/state/<world>/); NTS_PERSIST=0 replays the history every time
    from .world import use_state_dir

    use_state_dir(None if os.environ.get("NTS_PERSIST") == "0" else ROOT / "runs" / "state")
    # the demo campus is made once (then restored like any world); NTS_CAMPUS=0 skips it
    uvicorn.run(create_web_app(campus=os.environ.get("NTS_CAMPUS", "1") != "0"), host=os.environ.get("NTS_HOST", "127.0.0.1"),
                port=int(os.environ.get("NTS_PORT", "8000")))
