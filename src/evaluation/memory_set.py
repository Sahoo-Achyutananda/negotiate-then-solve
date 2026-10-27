"""nts-memory on its test set (training.memory_data): is each message made to stand on its own as it should be?

    uv run python -m evaluation.memory_set --gemini gemini-3.8-flash           # Gemini with the memory prompt
    uv run python -m evaluation.memory_set --local http://10.1.25.191:8080/v1  # the nts-memory adapter (``memory``)

Each conversation expects one of: ``same`` (the message stands on its own), ``ask`` (it leans on something the
conversation lacks), or a rewrite. A rewrite is right when it names what the expected one names: the same
teachers, sections, rooms and courses (in any form: "sec 2" is Section 2), days, times, week and request numbers,
and is a question or a request as the expected one is. Words may differ ("when is Dr. Rao free on Friday?" and
"is Dr. Rao free friday?" are the same lookup).
Scores per kind (change, refer, request, same, ask, mixed); every reading goes to ``--out``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from agents.memory import Memory, Turn
from agents.questions import QuestionReader
from core.instance import Instance
from evaluation.lookup_set import Unreachable, _with_retries

ROOT = Path(__file__).parents[2]
DAYS = {"mon": "Mon", "tue": "Tue", "wed": "Wed", "thu": "Thu", "fri": "Fri", "sat": "Sat", "sun": "Sun"}
DAY = re.compile(r"\b(mon|tue|wed|thu|fri|sat|sun)(day|s|sday|nesday|rs|rsday|urday)?\b|\b(today|tomorrow|tmrw)\b",
                 re.I)
TIME = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|ish)\b|\b(\d{1,2}):(\d{2})\b", re.I)
WEEK = re.compile(r"\b(?:(next|this|nxt) (?:week|wk)|(?:week|wk) ?(\d+))\b", re.I)
SHORT = re.compile(r"\b(sec(?:tion)?s?|grp|group|batch|rm|room|classroom|lab)[\s\-]*(\d+)\b", re.I)
REQUEST = re.compile(r"\b(mov(e|ing)|shift(ing)?|cancel(ling)?|swap(ping)?|exchange|book(ing)?|reschedul(e|ing)|"
                     r"postpon(e|ing)|request|withdraw|drop|can i have|can i get|i want|please)\b", re.I)


def features(text: str, instance: Instance, asker: str, week: int) -> dict:
    """What a message names, in a form two wordings of it share."""
    reader = QuestionReader(instance, asker, week)
    teachers, sections, rooms, courses, _ = reader._names(text)
    for m in SHORT.finditer(text):  # "sec 2", "grp 4", "rm 104", "lab-3": short forms the regex reader's misses
        kind = "section" if m.group(1).lower().startswith(("sec", "grp", "group", "batch")) else \
            "lab" if m.group(1).lower().startswith("lab") else "room"
        for x in reader.directory.find(f"{kind} {m.group(2)}"):
            (sections if kind == "section" else rooms).append(x.id)
    days = set()
    for m in DAY.finditer(text):
        days.add(m.group(3).lower().replace("tmrw", "tomorrow") if m.group(3) else DAYS[m.group(1).lower()])
    times = set()
    for m in TIME.finditer(text):
        h = int(m.group(1) or m.group(4))
        if (m.group(3) or "").lower() == "pm" and h != 12 or (m.group(3) or "").lower() == "ish" and h < 8:
            h += 12
        times.add(h)
    weeks = {f"week {n}" if n else f"{w.lower().replace('nxt', 'next')} week" for w, n in WEEK.findall(text)}
    weeks -= {"this week", f"week {week}"}  # the current week, by its number or not at all, is this week
    ids = {int(x) for x in re.findall(r"\bR-?0*(\d+)\b", text, re.I)}
    return {"teachers": set(teachers), "sections": set(sections), "rooms": set(rooms), "courses": set(courses),
            "days": days, "times": times, "weeks": weeks, "requests": ids, "request": bool(REQUEST.search(text))}


def right(row: dict, got, dept: Instance) -> bool:
    want, m = row["meta"]["answer"], row["meta"]
    if "same" in want:
        return got.ask is None and not got.changed
    if "ask" in want:
        return got.ask is not None
    if got.ask is not None or not got.changed:
        return False
    f = lambda t: features(t, dept, m["asker"], m["week"])  # noqa: E731
    got_f, want_f = f(got.text), f(want["message"])
    if want_f["requests"]:  # about a request: its number ("any news on R-00039?"; what it asked may be added)
        return got_f["requests"] == want_f["requests"]
    if not want_f["request"]:  # a request without request words ("two of our classes overlap"): any wording of it
        got_f["request"] = False
    return got_f == want_f


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", type=Path, default=ROOT / "data" / "memory-train" / "test.jsonl")
    ap.add_argument("--gemini", metavar="MODEL")
    ap.add_argument("--local", metavar="URL")
    ap.add_argument("--adapter", default=None, help="with --local: the adapter (default: adapters-lora.toml's memory)")
    ap.add_argument("--rpm", type=float, default=None)
    ap.add_argument("--rpd", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--misses", action="store_true", help="print every miss")
    a = ap.parse_args()

    from training.memory_data import build_departments
    from web.app import World

    if a.gemini:
        from language.llm import GeminiClient

        client, name = GeminiClient(model=a.gemini, rpm=a.rpm, rpd=a.rpd), a.gemini
    elif a.local:
        from language.adapters import local_client, registry

        client = local_client("memory", base_url=a.local, adapter=a.adapter, reg=registry(ROOT / "adapters-lora.toml"))
        name = a.adapter or "local"
    else:
        raise SystemExit("--gemini MODEL or --local URL")
    instance = World(parser_mode="offline", seed_history=False, deadline=60).instance
    departments = dict(build_departments(instance, test=True))
    rows = [json.loads(x) for x in a.test.read_text(encoding="utf-8").splitlines() if x.strip()][: a.limit]
    out = a.out or ROOT / "runs" / f"memory-{re.sub(r'[^a-z0-9.-]+', '-', name.lower())}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    score: dict[str, list[bool]] = defaultdict(list)
    with out.open("w", encoding="utf-8") as f:
        for n, row in enumerate(rows, 1):
            m, dept = row["meta"], departments[row["department"]]
            history = [Turn(who, text, datetime.fromisoformat(at)) for who, text, at in m["history"]]
            memory = Memory(dept, client, m["asker"], m["week"], now=datetime.fromisoformat(m["now"]))
            try:  # a dropped connection is tried again; the calls already made are cached, so a rerun is free
                got = _with_retries(lambda text: memory.rewrite(text, history), m["message"])
            except Unreachable as e:
                raise SystemExit(f"\nstopped at {n}/{len(rows)}: {e}; run the same command again (cached)") from None
            ok = right(row, got, dept)
            score[row["kind"]].append(ok)
            said = {"ask": got.ask} if got.ask else {"message": got.text} if got.changed else {"same": True}
            if a.misses and not ok:
                print(f"miss {row['kind']:8} {m['message']!r}\n      want {m['answer']}\n      got  {said}")
            f.write(json.dumps({"id": row["id"], "kind": row["kind"], "message": m["message"], "want": m["answer"],
                                "got": said, "right": ok, "tries": got.trace}, ensure_ascii=False) + "\n")
            if n % 50 == 0:
                print(f"{n}/{len(rows)}", flush=True)
    print(f"\n{name} on {len(rows)} conversations")
    for kind, oks in sorted(score.items()):
        print(f"  {kind:10} {sum(oks)}/{len(oks)}")
    every = [x for v in score.values() for x in v]
    print(f"  {'ALL':10} {sum(every)}/{len(every)}  ({100 * sum(every) / max(len(every), 1):.1f}%)")
    if hasattr(client, "usage"):
        print(client.usage)
    print(f"every reading: {out}")


if __name__ == "__main__":
    main()
