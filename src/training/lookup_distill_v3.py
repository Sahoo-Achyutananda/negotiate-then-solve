"""Gemini's part of nts-lookup-v3's data (distillation): natural rewordings of the template questions, each kept
only when Gemini, reading it as the v3 model would (``ToolAgent(multi=True)``), gets the same lookups as the
question it rewords. The kept rows teach the original (checked) answer with the new wording.

    uv run python -m training.lookup_distill_v3 --sample 400 --model gemini-3.8-flash --rpm 60
    # reads data/lookup-train-v3/train.jsonl, writes data/lookup-train-v3/distilled.jsonl and disagreements.jsonl
    uv run python -m training.lookup_distill_v3 --focus V3-045 ... mixed 4 --previous distilled-old.jsonl
    # only rows of those templates, kinds, or with at least 4 lookups; earlier rewordings kept (prompt updated)

Why both steps: a template's wording is a little stiff ("which section has the most free time"); people write
"which section's got the most free periods??". Gemini writes those. A rewording can drift ("most free time" ->
"least busy teacher"): reading it again and comparing what it resolves to catches that, so the label stays right.
Disagreements are kept apart (a bad rewording, or a reading Gemini gets wrong: both worth a look, neither trained
on). Gemini's answers are cached (``language.llm``): a rerun costs nothing.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from agents.tool_agent import SYSTEM_V3, Lookups, ToolAgent
from evaluation.lookup_set import SEMESTER_START, _norm

from .lookup_data import departments
from .lookup_data_v3 import OUT, FixedLookups, _messages

REWORD = """You rewrite messages that people send to a university timetable assistant, the way they would really
type them in a chat: shorter or longer, casual, with typos now and then, abbreviations (sec, wk, tmrw, DS for Data
Structures), "sir"/"ma'am", no capital letters, or very polite.
Keep the meaning exactly: every name, section, room, course, day, date, time, week number and every request in the
message, nothing added and nothing left out. If the message asks two or three things, the rewording asks all of them.
Write each rewording as the same person (keep "I", "my", "we", "our" as they are).
Keep whether several people, sections or rooms are asked about each on their own ("when is A, B and C free",
"each", "separately") or together ("when are A, B and C free", "both", "together", "a common hour")."""
# words of "each one on its own": an earlier rewording (read before the prompt had separately=true) that has them
# may have been read as "together"; it is left out rather than trusted
SPLIT_WORDS = re.compile(r"\b(each|separately|separate|individually|respectively|one by one)\b", re.IGNORECASE)


class Rewordings(BaseModel):
    items: list[str] = Field(min_length=1, max_length=3)


def resolved(call) -> list[tuple]:
    """A reading as what it resolves to: each lookup's tool, week and arguments (lists in any order)."""
    if call is None:
        return []
    parts = call.args["parts"] if call.tool == "multi" else [call]
    return [(p.tool, p.week, json.dumps(_norm(p.args), sort_keys=True, default=str)) if p.confident else ("unclear",)
            for p in parts]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=OUT, help="the v3 data folder (training.lookup_data_v3)")
    ap.add_argument("--sample", type=int, default=400, help="questions to reword")
    ap.add_argument("--per", type=int, default=2, help="rewordings per question")
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--rpm", type=float, default=None)
    ap.add_argument("--rpd", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--focus", nargs="*", default=[],
                    help="reword only rows of these templates (V3-045) or kinds (mixed), or with at least N lookups (4)")
    ap.add_argument("--previous", type=Path, help="earlier rewordings to keep (a distilled.jsonl in --data): their "
                    "prompt is brought up to date; those with words of 'each one on its own' are left out")
    a = ap.parse_args()

    from language.llm import GeminiClient
    from web.app import World

    instance = World(parser_mode="offline", seed_history=False, deadline=60).instance
    gemini = GeminiClient(model=a.model, rpm=a.rpm, rpd=a.rpd)
    rows = [json.loads(x) for x in (a.data / "train.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if "meta" in r]
    if a.focus:
        least = min([int(x) for x in a.focus if x.isdigit()] or [99])
        rows = [r for r in rows if set(a.focus) & {*r["templates"], r["kind"]}
                or len(json.loads(r["messages"][2]["content"])["lookups"]) >= least]
    random.Random(a.seed).shuffle(rows)
    kept, off, failed = [], [], 0
    depts: dict[int, dict] = {}
    earlier = []
    if a.previous:
        for x in (a.data / a.previous).read_text(encoding="utf-8").splitlines():
            r = json.loads(x)
            text = r["messages"][1]["content"].split("<question>", 1)[1].rsplit("</question>", 1)[0]
            tools = {x["tool"] for x in json.loads(r["messages"][2]["content"])["lookups"]}
            if not (SPLIT_WORDS.search(text) and tools & {"free_hours", "gaps"}):  # where "each" changes the answer
                earlier.append({**r, "messages": [{"role": "system", "content": SYSTEM_V3}, *r["messages"][1:]]})
        print(f"{len(earlier)} earlier rewordings kept")
    for n, r in enumerate(rows[: a.sample], 1):
        meta = r["meta"]
        dept = dict(depts.setdefault(meta.get("seed", 0), dict(departments(instance, meta.get("seed", 0)))))[r["department"]]
        now = datetime.fromisoformat(meta["now"])
        question = r["messages"][1]["content"].split("<question>", 1)[1].rsplit("</question>", 1)[0]
        gold = Lookups.model_validate_json(r["messages"][2]["content"])

        def agent(client) -> ToolAgent:
            return ToolAgent(dept, client, meta["asker"], meta["week"], today=now.date(), now=now,
                             semester_start=SEMESTER_START, multi=True)

        want = resolved(agent(FixedLookups(gold)).read(question))
        try:
            words = gemini.generate(REWORD, f"Message:\n{question}\n\nWrite {a.per} rewordings.", Rewordings).items
        except Exception as e:  # noqa: BLE001 - one failed call is skipped, the run goes on
            failed += 1
            print(f"  {n}: reword failed ({type(e).__name__})", flush=True)
            continue
        for text in words[: a.per]:
            try:
                reader = agent(gemini)
                got = resolved(reader.read(text))
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"  {n}: read failed ({type(e).__name__})", flush=True)
                continue
            if got == want or sorted(got) == sorted(want):
                # the same lookups; in another order when the rewording asks in another order ("Lab 2 or Lab 3"):
                # then Gemini's own answer, in the rewording's order, is the label
                label = gold if got == want else Lookups.model_validate(reader.trace[-1]["choice"])
                kept.append({**r, "id": f"{r['id']}-gemini-{len(earlier) + len(kept)}", "kind": f"{r['kind']}+reworded",
                             "messages": _messages(agent(FixedLookups(label)), text, label)})
            else:
                off.append({"id": r["id"], "question": question, "reworded": text, "want": want, "gemini": got})
        if n % 25 == 0:
            print(f"{n}/{min(a.sample, len(rows))}: kept {len(kept)}, disagreements {len(off)}, failed {failed}", flush=True)
    with (a.data / "distilled.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in earlier + kept)
    with (a.data / "disagreements.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(x, ensure_ascii=False, default=str) + "\n" for x in off)
    print(f"kept {len(kept)} rewordings, {len(off)} disagreements, {failed} failed calls; Gemini usage: {gemini.usage}")


if __name__ == "__main__":
    main()
