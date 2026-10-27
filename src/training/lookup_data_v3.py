"""Training data for nts-lookup-v3 (ROADMAP step 8): the v3 prompt (``SYSTEM_V3``) and answer (``Lookups``: every
lookup a message asks for), built to continue training from nts-lookup-v2's adapter.

    uv run python -m training.lookup_data_v3                       # writes data/lookup-train-v3/{train,val}.jsonl
    uv run python -m training.lookup_data_v3 --replay 900 --multi 1200 --mixed 150 --fills 12

Three kinds of rows, every one checked by ToolAgent's own checks before it is written (as v1 and v2's):

1. **new tools** (``data/lookup-v3-train-templates.jsonl``): calendar, my_requests, my_rules, clash_check,
   move_options, counts, gaps, my_fairness, bookings, what_if, room_for, several weeks, and "each one on its own"
   against "together" ("when is Section 1, Section 2 and Section 3 free?" / "when are Sections 1, 2 and 3 free?");
2. **several lookups in one message**: 2 to 5 questions of any kind (v2's templates and the new ones) joined
   ("A and B", "A? Also, B", "A; B; also C", a numbered list), answered with the list of their lookups, in order;
   and **mixed**: one or two questions with a request ("... and move my Friday class to Monday"), the request
   answered "none" (the request reader's, not lookup's);
3. **replay**: a sample of v2's single-lookup rows written again in v3's prompt and answer (a list of one), so the
   continued training does not forget what v2 knows.

Optional 4th kind, from Gemini (``training.lookup_distill_v3``): natural rewordings of these questions, kept only
when Gemini, reading them as the v3 model would, gets the same lookups.

Departments, askers and "now" vary as in v2 (``training.lookup_data``). Templates that are test templates are
refused. About one template in ten is kept for validation only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from agents.questions import ToolCall
from agents.tool_agent import SYSTEM_V3, ChoiceV3, Lookups, ToolAgent
from core.instance import Instance
from evaluation.lookup_set import PLACEHOLDER, SEMESTER_START, TemplateFiller, _norm, apart

from .lookup_data import ROOT, TEMPLATES as V2_TEMPLATES, _named, _shape, a_moment, choice_for, departments

TEMPLATES = ROOT / "data" / "lookup-v3-train-templates.jsonl"
TEST_TEMPLATES = ROOT / "data" / "lookup-v3-test-templates.jsonl"
OUT = ROOT / "data" / "lookup-train-v3"
JOINERS = ["{a} and {b}", "{a}? and {b}?", "{a}. Also, {b}", "{a}; {b}", "{a}, and also {b}", "{a}? {b}?"]
# placeholders of the second to fifth part get other names, so "{teacher}" in part 2 can be another teacher:
# part i's "{teacher}", "{teacher2}", "{teacher3}" are "{teacher<i>1}", "{teacher<i>2}", "{teacher<i>3}"
RENAME = {i: {f"{base}{j}": f"{base}{i}{j or 1}" for base in ("teacher", "section", "room", "lab", "course", "day", "time")
              for j in ("", "2", "3")} for i in range(2, 6)}
PARTS = [2] * 50 + [3] * 25 + [4] * 15 + [5] * 10  # how many questions a message of several has, in percent
FIELDS_V3 = {"room_for", "weeks", "which", "request_id", "measure", "order", "class_day", "class_kind", "club",
             "request", "first", "change", "roles", "of", "what", "separately"}


class FixedLookups:
    """Answers with the lookups it is given: to run a row's answer through ToolAgent's own checks."""

    def __init__(self, lookups: Lookups) -> None:
        self.lookups = lookups

    def generate(self, system, prompt, schema):
        return self.lookups


def _rename(obj, names: dict[str, str]):
    """A template, choice or label with its placeholders renamed ("{teacher}" -> "{teacher7}")."""
    if isinstance(obj, str):
        return PLACEHOLDER.sub(lambda m: "{" + names.get(m.group(1), m.group(1)) + (f".{m.group(2)}" if m.group(2) else "") + "}", obj)
    if isinstance(obj, list):
        return [_rename(x, names) for x in obj]
    if isinstance(obj, dict):
        return {k: _rename(v, names) for k, v in obj.items()}
    return obj


def choice_v3(row: dict, filled, filler: TemplateFiller, week: int) -> ChoiceV3:
    """The lookup a part teaches, as a ChoiceV3: its explicit ``choice`` (v3's tools), else v2's (``choice_for``)."""
    if row.get("choice"):
        c = {}
        for k, v in row["choice"].items():
            if k == "request":  # the change in words: its placeholders filled in as text
                c[k] = PLACEHOLDER.sub(lambda m: filler._text(m.group(1), m.group(2)), v)
            else:
                c[k] = _named(k, v, filler)
        return ChoiceV3(**{"week": filled.week, **c})
    v2 = choice_for(row, filled, filler, week)
    return ChoiceV3(**v2.model_dump(exclude_defaults=True), separately=bool(row.get("separately")))


def as_written(choice: ChoiceV3, filler: TemplateFiller) -> ChoiceV3:
    """The lookup with teachers, sections, rooms and courses written as the question wrote them ("khan sir", "sec 2",
    "ML"), not as the department lists them: v3 has no lists in its prompt; the code finds who or what they are."""
    from core.instance import Faculty, Group, Room

    said: dict[str, str] = {}
    for name, obj in filler.picked.items():
        if name not in filler.said:
            continue
        if isinstance(obj, Faculty | Group | Room):
            said.setdefault(obj.name, filler.said[name])
        elif isinstance(obj, str) and (name.startswith("course") or name == "my_course"):
            said.setdefault(filler.instance.course_title(obj), filler.said[name])
    update = {k: [said.get(x, x) for x in getattr(choice, k)] for k in ("teachers", "sections", "rooms", "room_for")
              if getattr(choice, k)}
    if choice.course:
        update["course"] = said.get(choice.course, choice.course)
    return choice.model_copy(update=update)


def matches(want: dict, call: ToolCall | None) -> bool:
    """A part's label against what it resolves to: the tool, the week, and every argument the label gives (a label
    may give only some: "move_options" names the course, not the class's ID)."""
    if want["tool"] is None:
        return call is None
    if call is None:
        return False
    if want["tool"] == "unclear":
        return not call.confident
    if call.tool != want["tool"] or call.week != want["week"] or not call.confident:
        return False
    got = _norm(call.args)
    return all(got.get(k) == v for k, v in _norm(want["args"]).items())


def _messages(agent: ToolAgent, question: str, answer: Lookups) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_V3}, {"role": "user", "content": agent._prompt(question)},
            {"role": "assistant", "content": answer.model_dump_json(exclude_defaults=True)}]


def build(instance: Instance, fills: int, multi: int, replay: int, seed: int,
          mixed: int = 0) -> tuple[list[dict], list[dict], list[str], Counter]:
    v2 = [json.loads(x) for x in V2_TEMPLATES.read_text(encoding="utf-8").splitlines() if x.strip()]
    v3 = [json.loads(x) for x in TEMPLATES.read_text(encoding="utf-8").splitlines() if x.strip()]
    tests = set()
    for path in (TEST_TEMPLATES, ROOT / "data" / "lookup-test-templates.jsonl", ROOT / "data" / "lookup-test-templates-v2.jsonl"):
        if path.exists():
            tests |= {_shape(json.loads(x)["template"]) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()}
    if clash := [r["id"] for r in v3 if _shape(r["template"]) in tests]:
        raise SystemExit(f"training templates that are test templates (change them): {clash}")
    depts = departments(instance, seed)
    rng = random.Random(f"v3-{seed}")
    train, val, dropped, cover = [], [], [], Counter()

    def held_out(ids: list[str]) -> bool:
        return any(int(hashlib.sha1(i.encode()).hexdigest(), 16) % 10 == 0 for i in ids)

    def one(parts: list[dict], k: int, kind: str) -> None:
        """A message of one or more parts, filled in one department by one asker at one "now", checked, kept."""
        r = random.Random(f"{seed}-{kind}-{'+'.join(p['id'] for p in parts)}-{k}")
        dept_name, dept = depts[k % len(depts)]
        now, week = a_moment(r, dept.calendar.weeks)
        filler = TemplateFiller(dept, current_week=week, now=now, semester_start=SEMESTER_START)
        # "this week / next week" stays where each part says it (a reader ties it to the words around it); every part
        # that has one has the same, and a part without one is about this week
        when = r.choice([("this week", week), ("next week", week + 1), ("", week)])
        if when[1] > dept.calendar.weeks:
            when = ("", week)
        rows = []
        for i, p in enumerate(parts, 1):
            p = _rename(p, RENAME[i]) if i > 1 else p
            had_when = "{when}" in p["template"]
            p = {**p, "template": re.sub(r"\s+", " ", p["template"].replace("{when}", when[0])).replace(" ?", "?").strip()}
            if had_when:
                p["week"] = when[1]
            rows.append(p)
        if len({p.get("asker", "teacher") for p in rows}) > 1:
            return  # a class rep's "we" and a teacher's "my" don't go in one message
        if len(rows) == 1:
            template = rows[0]["template"]
        else:
            heads = [p["template"].rstrip(" ?.") for p in rows]
            if len(rows) == 2:
                template = r.choice(JOINERS).format(a=heads[0], b=heads[1])
            else:  # "A; B; also C", "A? B? C?", or a numbered list (four or five questions often are)
                template = r.choice(["; ".join(heads[:-1]) + r.choice([", and ", "; also "]) + heads[-1],
                                     "? ".join(heads) + "?",
                                     "\n".join(f"{n}. {h}?" for n, h in enumerate(heads, 1))])
        joined = {k2: v for k2, v in rows[0].items() if k2 not in ("choice", "week")}
        filled = filler.fill(joined | {"template": template, "args": {}, "tool": "free_hours"}, r)
        if filled is None:
            return
        from agents.questions import from_now

        wants, choices = [], []
        try:
            for p in rows:
                p_week = filler._week(p)
                f = type(filled)(p["id"], p.get("group", ""), filled.question, filled.asker, p["tool"],
                                 {k2: filler._label(k2, v) for k2, v in p.get("args", {}).items()}, p_week)
                if f.args.pop("next", False):  # "my next class": the first one from now
                    p_week, after = from_now(dept, week, now)
                    f.week, f.args["after"] = p_week, list(after)
                filler._resolve(p["tool"], f.args)
                args = {k2: v for k2, v in f.args.items() if v not in (None, [], False)}
                # "each one on its own": a lookup each (ToolAgent splits it, so it is checked split)
                wants += [{"tool": p["tool"], "args": x, "week": p_week}
                          for x in (apart(p["tool"], args, p["split"]) if p.get("split") else [args])]
                choices.append(as_written(choice_v3(p, f, filler, week), filler) if p["tool"] is not None
                               else ChoiceV3(tool="none", week=week))
        except (KeyError, ValueError, LookupError) as e:
            dropped.append(f"{kind} {[p['id'] for p in parts]} {filled.question!r}: {e}")
            return
        answer = Lookups(lookups=choices)
        agent = ToolAgent(dept, FixedLookups(answer), filled.asker, week, today=now.date(), now=now,
                          semester_start=SEMESTER_START, multi=True)
        call = agent.read(filled.question)
        got = call.args["parts"] if call is not None and call.tool == "multi" else ([call] if call else [])
        expect = [w for w in wants if w["tool"] is not None]
        ok = (not expect and call is None) or (len(got) == len(expect) and all(matches(w, g) for w, g in zip(expect, got)))
        if not ok or any(t["problem"] for t in agent.trace):
            dropped.append(f"{kind} {[p['id'] for p in parts]} {filled.question!r}: answer {answer.model_dump(exclude_defaults=True)} "
                           f"gives {call}; want {wants}")
            return
        row = {"id": f"{kind}-{'+'.join(p['id'] for p in parts)}-{k}", "task": "lookup", "kind": kind,
               "templates": [p["id"] for p in parts], "department": dept_name,
               # who asked, when: what lookup_distill_v3 needs to read a rewording the same way
               "meta": {"asker": filled.asker, "now": now.isoformat(), "week": week, "seed": seed},
               "messages": _messages(agent, filled.question, answer)}
        (val if held_out([p["id"] for p in parts]) else train).append(row)
        cover[f"kind {kind}"] += 1
        cover[f"parts {len(parts)}"] += 1
        for c in choices:
            cover[f"tool {c.tool}"] += 1
            for fld in FIELDS_V3 & set(c.model_dump(exclude_defaults=True)):
                cover[f"field {fld}"] += 1

    for row in v3:  # 1. the new tools
        for k in range(fills):
            one([row], k, "new")
    pool = [r for r in v2 + v3 if r["tool"] not in (None, "unclear")]  # questions only: parts of a message
    def same_asker(first: dict, n: int) -> list[dict]:  # a class rep's "we" and a teacher's "my" don't mix
        others = [r for r in pool if r is not first and r.get("asker", "teacher") == first.get("asker", "teacher")]
        return [first] + rng.sample(others, n - 1)

    for k in range(multi):  # 2. several lookups
        one(same_asker(rng.choice(pool), rng.choice(PARTS)), k, "multi")
    requests = [r for r in v2 if r["tool"] is None and r.get("group") == "request"]
    for k in range(mixed):  # 3. questions and a request in one message: the request is "none", left to its reader
        parts = same_asker(rng.choice([r for r in pool if r.get("asker") != "rep"]), rng.choice([1, 1, 2]))
        parts.insert(rng.randrange(len(parts) + 1), rng.choice(requests))
        one(parts, k, "mixed")
    for k in range(replay):  # 4. v2's kinds again, in v3's form
        one([rng.choice(v2)], k, "replay")
    return train, val, dropped, cover


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fills", type=int, default=12, help="questions per new-tool template")
    ap.add_argument("--multi", type=int, default=1200, help="messages with several lookups (2 to 5)")
    ap.add_argument("--mixed", type=int, default=150, help="messages with questions and a request")
    ap.add_argument("--replay", type=int, default=900, help="v2 rows again, in v3's form (against forgetting)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    from web.app import World

    instance = World(parser_mode="offline", seed_history=False, deadline=60).instance
    train, val, dropped, cover = build(instance, a.fills, a.multi, a.replay, a.seed, a.mixed)
    a.out.mkdir(parents=True, exist_ok=True)
    random.Random(a.seed).shuffle(train)
    for name, rows in (("train", train), ("val", val)):
        with (a.out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    manifest = {"task": "lookup", "version": 3, "train": len(train), "val": len(val), "fills": a.fills, "multi": a.multi,
                "mixed": a.mixed, "replay": a.replay, "seed": a.seed, "dropped": len(dropped), "cover": dict(sorted(cover.items()))}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{len(train)} train and {len(val)} validation rows in {a.out}; {len(dropped)} left out")
    for k, v in sorted(cover.items()):
        print(f"  {k:28} {v}")
    for d in dropped[:25]:
        print("LEFT OUT", d)


if __name__ == "__main__":
    main()
