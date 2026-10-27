"""Gemini's part of nts-memory's data: other wordings of the follow-ups, each kept only when Gemini, reading it in
its conversation as the memory model would (``agents.memory``), makes it the same full message as the original.

    uv run python -m training.memory_distill --sample 300 --model gemini-3.8-flash --rpm 1000 --rpd 5000
    # reads data/memory-train/train.jsonl, writes distilled.jsonl and disagreements.jsonl in the same folder

The template follow-ups are a few fixed wordings ("and {day}?", "what about {name}?"); people write "fri?", "wat
abt rao", "same q for sec 3 pls". A rewording keeps leaning on the conversation (it must not spell out what it
refers to: that would be a message that stands on its own, answered "same"); the check makes sure it still means
the same, and it is trained with the original's (known) answer. Calls are cached: a rerun costs nothing.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from agents.memory import SYSTEM, Memory, Turn
from evaluation.memory_set import right

from .memory_data import OUT, build_departments

REWORD = """You rewrite the newest message of a chat with a university timetable assistant the way people really type
follow-ups: short, casual, abbreviations (sec, wk, tmrw, fri, DS), typos now and then, "sir"/"ma'am", or polite.
Keep its meaning exactly, and keep it leaning on the conversation just as it does: don't spell out what it refers
to ("and friday?" may become "fri?" or "what about friday then", never "when is Dr. Rao free on Friday?").
Keep every name, day, time, week and number it has. Write as the same person ("I", "my", "we" as they are)."""
KINDS = ("change", "two", "relative", "chain", "refer", "list", "two_back", "both", "reply", "request", "whatif",
         "yes", "each", "ask", "mixed")  # every kind but "same" (a message that needs nothing, reworded, still does)


class Rewordings(BaseModel):
    items: list[str] = Field(min_length=1, max_length=3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=OUT)
    ap.add_argument("--sample", type=int, default=300, help="follow-ups to reword")
    ap.add_argument("--per", type=int, default=2, help="rewordings per follow-up")
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--rpm", type=float, default=None)
    ap.add_argument("--rpd", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kinds", nargs="*", default=list(KINDS), help="reword only conversations of these kinds")
    ap.add_argument("--keep", action="store_true", help="keep the rewordings already in distilled.jsonl, add to them")
    a = ap.parse_args()

    from language.llm import GeminiClient
    from web.app import World

    instance = World(parser_mode="offline", seed_history=False, deadline=60).instance
    departments = dict(build_departments(instance, test=False))
    gemini = GeminiClient(model=a.model, rpm=a.rpm, rpd=a.rpd)
    rows = [json.loads(x) for x in (a.data / "train.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if r["kind"] in a.kinds]
    random.Random(a.seed).shuffle(rows)
    kept, off, failed = [], [], 0
    earlier = [json.loads(x) for x in (a.data / "distilled.jsonl").read_text(encoding="utf-8").splitlines()
               if x.strip()] if a.keep and (a.data / "distilled.jsonl").exists() else []
    have = {x["messages"][1]["content"] for x in earlier}
    for n, r in enumerate(rows[: a.sample], 1):
        m, dept = r["meta"], departments[r["department"]]
        history = [Turn(who, text, datetime.fromisoformat(at)) for who, text, at in m["history"]]
        memory = Memory(dept, gemini, m["asker"], m["week"], now=datetime.fromisoformat(m["now"]))
        shown = "\n".join(f"{'You' if t.who == 'you' else 'Assistant'}: {t.text}" for t in history)
        try:
            words = gemini.generate(REWORD, f"Conversation:\n{shown}\n\nNewest message: {m['message']}\n\n"
                                            f"Write {a.per} rewordings of the newest message.", Rewordings).items
        except Exception as e:  # noqa: BLE001 - one failed call is skipped, the run goes on
            failed += 1
            print(f"  {n}: reword failed ({type(e).__name__})", flush=True)
            continue
        for text in words[: a.per]:
            if text.strip().lower() == m["message"].strip().lower():
                continue
            try:
                got = memory.rewrite(text, history)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"  {n}: read failed ({type(e).__name__})", flush=True)
                continue
            if right(r, got, dept):
                prompt = memory.prompt(text, history)
                if prompt in have:  # kept by an earlier run (--keep): not twice
                    continue
                have.add(prompt)
                kept.append({**r, "id": f"{r['id']}-gemini-{len(earlier) + len(kept)}", "kind": f"{r['kind']}+reworded",
                             "meta": {**m, "message": text},
                             "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt},
                                          r["messages"][2]]})
            else:
                said = {"ask": got.ask} if got.ask else {"message": got.text} if got.changed else {"same": True}
                off.append({"id": r["id"], "message": m["message"], "reworded": text, "want": m["answer"],
                            "gemini": said})
        if n % 25 == 0:
            print(f"{n}/{min(a.sample, len(rows))}: kept {len(kept)}, disagreements {len(off)}, failed {failed}",
                  flush=True)
    with (a.data / "distilled.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in earlier + kept)
    with (a.data / "disagreements.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in off)
    print(f"kept {len(kept)} rewordings, {len(off)} disagreements, {failed} failed calls; Gemini usage: {gemini.usage}")


if __name__ == "__main__":
    main()
