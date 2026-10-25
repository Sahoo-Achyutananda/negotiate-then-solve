"""Training data for the lookup adapter (ROADMAP.md, step 5): questions about the timetable and the lookup each
should give, in exactly the format ``ToolAgent`` sends and expects, so a model trained on it drops in for Gemini.

    uv run python -m training.lookup_data                 # writes data/lookup-train/{train,val}.jsonl
    uv run python -m training.lookup_data --fills 24      # more questions per template

One row is one chat, as the local server is sent it (``language.local``):
  system:    agents.tool_agent.SYSTEM (the tools and rules)
  user:      ToolAgent._prompt(question): who asks, today and the time, the week, days, hours, names
  assistant: the lookup, as the Choice JSON (names, not IDs), e.g.
             {"tool":"free_hours","teachers":["Dr. Imran Khan"],"days":["Tue"],"hours":[5],"week":13}

Where the questions come from: ``data/lookup-train-templates.jsonl``, templates filled by the same filler as
the test set (``evaluation.lookup_set``), but DIFFERENT templates: a training template that is also a test
template is refused, so the test measures wordings the model has not seen. Variety in every row:
* the department: the team's file, the first-name copy, and copies with other teachers' names, so the model
  learns to copy names from the prompt instead of remembering them;
* who asks (any teacher, or a class representative), and "today" (any teaching day of weeks 2-15, any hour,
  sometimes a weekend), which changes what "tomorrow", "this week" and "next class" mean;
* every placeholder in the forms people write it ("imran", "Dr Khan", "sec 2", "tue", "2ish", "28/10").

Every row is checked before it is written: its answer goes through ToolAgent's own checks (names, days,
hours, the week, dates) and must give the lookup the template expects. A row that does not is a mistake in a
template, reported and left out.

Validation: about one template in ten is kept out of training altogether (``val.jsonl``), so the notebook's
check measures new wordings, not new fills of known ones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from datetime import date, datetime, time, timedelta
from pathlib import Path

from agents.questions import ToolCall
from agents.tool_agent import SYSTEM, Choice, ToolAgent
from core.instance import Faculty, Group, Instance, Room
from evaluation.lookup_set import (PLACEHOLDER, SEMESTER_START, TEMPLATES as TEST_TEMPLATES, TEMPLATES_V2 as TEST_TEMPLATES_V2,
                                   Filled, TemplateFiller, right, with_first_names)

ROOT = Path(__file__).parents[2]
TEMPLATES = ROOT / "data" / "lookup-train-templates.jsonl"
OUT = ROOT / "data" / "lookup-train"

# other names for the teachers, so the model copies names from the prompt instead of remembering them
FIRSTS = ["Arjun", "Kavya", "Sanjay", "Deepa", "Farhan", "Ritu", "Manoj", "Sneha", "Vivek", "Anjali", "Karthik",
          "Pooja", "Suresh", "Divya", "Naveen", "Shalini", "Harish", "Lata", "Rahim", "Geeta"]
SURNAMES = ["Reddy", "Nair", "Joshi", "Banerjee", "Kulkarni", "Hegde", "Saxena", "Chatterjee", "Bhat", "Mishra",
            "Qureshi", "Varma", "Shetty", "Agarwal", "Ghosh", "Patil", "Sinha", "Fernandes", "Rao", "Kapoor"]


def renamed(instance: Instance, rng: random.Random, first_names: bool, move_hod: bool = False) -> Instance:
    """A copy of the department whose teachers have other names (with or without first names; sometimes two
    with the same first name, so "ambiguous" has examples). ``move_hod``: another teacher is the HOD, so "the
    HOD" is learned as a role to look up, not as a name."""
    data = instance.model_dump()
    if move_hod:
        roles = [f["role"] for f in data["faculty"]]
        rng.shuffle(roles)
        for f, role in zip(data["faculty"], roles, strict=True):
            f["role"] = role
    surnames = rng.sample(SURNAMES, len(data["faculty"]))
    firsts = rng.sample(FIRSTS, len(data["faculty"]))
    if first_names and rng.random() < 0.5:
        firsts[1] = firsts[0]  # a shared first name
    for f, s, fn in zip(data["faculty"], surnames, firsts, strict=False):
        title = rng.choice(["Dr.", "Dr.", "Prof."])
        f["name"] = f"{title} {fn} {s}" if first_names else f"{title} {s}"
    return Instance.model_validate(data)


SHARED_SEEDS = (4, 5, 6, 7)  # v2: generated departments with shared courses (seeds 2, 3 and 8 are test runs B and D)


def departments(instance: Instance, seed: int) -> list[tuple[str, Instance]]:
    """The departments the questions are asked in: the team's, the first-name copy, four renamed ones (the HOD
    moved in two of them), and, since v2, departments with courses as they usually are (one course for every
    section, some with two teachers), renamed, so no department's facts can be learned."""
    from evaluation.lookup_set import department

    rng = random.Random(f"departments-{seed}")
    out = [("original", instance), ("first-names", with_first_names(instance))]
    out += [(f"renamed-{k}", renamed(instance, rng, first_names=k % 2 == 0, move_hod=k >= 2)) for k in range(4)]
    out += [(f"shared-s{s}", renamed(department(instance, s, shared=True), rng, first_names=s % 2 == 0, move_hod=True))
            for s in SHARED_SEEDS]
    return out


def a_moment(rng: random.Random, weeks: int) -> tuple[datetime, int]:
    """A "now" in the semester (mostly a teaching day, sometimes a weekend) and its week."""
    week = rng.randint(2, weeks - 1)
    day = rng.randrange(5) if rng.random() < 0.9 else rng.choice([5, 6])
    now = datetime.combine(SEMESTER_START + timedelta(weeks=week - 1, days=day),
                           time(rng.randint(8, 18), rng.choice([0, 15, 30, 45])))
    return now, week


class FixedModel:
    """Answers with the one choice it is given: to run a row's answer through ToolAgent's own checks."""

    def __init__(self, choice: Choice) -> None:
        self.choice = choice

    def generate(self, system: str, prompt: str, schema: type[Choice]) -> Choice:
        return self.choice


def choice_for(row: dict, filled: Filled, filler: TemplateFiller, current_week: int) -> Choice:
    """The answer a row teaches: the template's explicit ``choice``, ``none`` for requests and rule questions,
    or its expected lookup written the way the model answers (names, not IDs; "me" kept)."""
    if row.get("choice"):
        c = {k: _named(k, v, filler) for k, v in row["choice"].items()}
        return Choice(**{"week": filled.week, **c})  # the question's week ("next week"), not always this one
    if row["tool"] is None:
        return Choice(tool="none", week=current_week)
    a, c = row.get("args", {}), {"tool": row["tool"], "week": filled.week}
    for key, value in a.items():
        if key == "sections" and value is True:  # who_is_free over sections, not a list of sections
            c["of"] = "sections"
        elif key in ("teachers", "sections", "rooms"):
            c[key] = [_named(key, v, filler) for v in value]
        elif key in ("teacher", "section", "room"):
            c[key + "s"] = [_named(key, value, filler)]
        elif key == "days":  # a calendar date is copied as a date; the code works out its week and day
            for v in value:
                name = (PLACEHOLDER.fullmatch(v) or [None, None])[1] if isinstance(v, str) else None
                if name in ("date", "weekend_date", "outside_date"):
                    c.setdefault("dates", []).append(filler.picked[name].isoformat())
                else:
                    c.setdefault("days", []).append(filler._label("days", v))
        elif key == "slots":
            c["hours"] = [filler._label(key, v) for v in value]
        elif key == "course":
            c["course"] = _named(key, value, filler)
        elif key in ("part", "length", "kind", "min_capacity", "what", "next", "count", "first", "change"):
            c[key] = filler._label(key, value) if key not in ("next", "first") else True
        elif key == "roles":  # v2: the role, never who holds it
            c["roles"] = list(value)
        elif key == "equipment":
            c["equipment"] = [filler._label(key, v) for v in value]
        else:
            raise KeyError(f"{row['id']}: no Choice field for {key!r}")
    if row["tool"] == "who_is_free":
        c.setdefault("of", "teachers")
    return Choice(**c)


def _named(key: str, value, filler: TemplateFiller):
    """A placeholder as the model writes it: a teacher, section or room by its listed name, a course by its
    title, a date as YYYY-MM-DD; anything else (days, hours, "me") as the label has it."""
    if isinstance(value, list):
        return [_named(key, v, filler) for v in value]
    if value == "me" or not (isinstance(value, str) and (m := PLACEHOLDER.fullmatch(value))):
        return value
    name, form = m.group(1), m.group(2)
    if name.startswith("teacher") and form == "course":
        return filler.instance.course_title(filler._course_of(filler._pick(name)))
    picked = filler._pick(name)
    if isinstance(picked, Faculty | Group | Room):
        return picked.name
    if name.startswith("course") or name == "my_course":
        return filler.instance.course_title(picked)
    if name in ("unknown_teacher", "student", "shared_first"):
        return filler._text(name, None)
    if isinstance(picked, date):
        return picked.isoformat()
    return filler._label(key, value)


def _shape(template: str) -> str:
    """A template's wording without its placeholders' forms, to compare training and test templates."""
    return re.sub(r"\s+", " ", PLACEHOLDER.sub("{}", template).lower()).strip(" ?.")


def build(instance: Instance, fills: int, seed: int) -> tuple[list[dict], list[dict], list[str], Counter]:
    """Train rows, validation rows, the rows left out (with why) and what the data covers."""
    rows = [json.loads(x) for x in TEMPLATES.read_text(encoding="utf-8").splitlines() if x.strip()]
    test = {_shape(json.loads(x)["template"]) for path in (TEST_TEMPLATES, TEST_TEMPLATES_V2) if path.exists()
            for x in path.read_text(encoding="utf-8").splitlines() if x.strip()}
    if clash := [r["id"] for r in rows if _shape(r["template"]) in test]:
        raise SystemExit(f"training templates that are test templates (change them): {clash}")
    depts = departments(instance, seed)
    train, val, dropped, cover, seen = [], [], [], Counter(), set()
    for row in rows:
        held_out = int(hashlib.sha1(row["id"].encode()).hexdigest(), 16) % 10 == 0  # about 1 in 10 templates
        for k in range(fills):
            rng = random.Random(f"{seed}-{row['id']}-{k}")
            dept_name, dept = depts[k % len(depts)]
            now, week = a_moment(rng, dept.calendar.weeks)
            filler = TemplateFiller(dept, current_week=week, now=now, semester_start=SEMESTER_START)
            when = rng.choice([("this week", week), ("next week", week + 1), ("", week), ("in week {n}", None)])
            n = rng.choice([w for w in range(1, dept.calendar.weeks + 1) if w not in (week, week + 1)])
            text = row["template"].replace("{when}", when[0].replace("{n}", str(n)))
            row_k = {**row, "template": re.sub(r"\s+", " ", text).replace(" ?", "?").strip()}
            if "{when}" in row["template"]:
                row_k["week"] = when[1] or n
            filled = filler.fill(row_k, rng)
            if filled is None or (dept_name, filled.question) in seen:
                continue
            seen.add((dept_name, filled.question))
            try:
                choice = choice_for(row_k, filled, filler, week)
            except (KeyError, ValueError) as e:
                dropped.append(f"{row['id']} {filled.question!r}: {e}")
                continue
            agent = ToolAgent(dept, FixedModel(choice), filled.asker, week, today=now.date(), now=now,
                              semester_start=SEMESTER_START)
            call: ToolCall | None = agent.read(filled.question)
            if not right(filled, call) or any(t["problem"] for t in agent.trace):
                dropped.append(f"{row['id']} {filled.question!r}: answer {choice.model_dump(exclude_defaults=True)} "
                               f"gives {call}; want {filled.tool} {filled.args} w{filled.week}")
                continue
            out = {"id": f"{row['id']}-{k}", "task": "lookup", "template": row["id"], "group": row.get("group"),
                   "department": dept_name, "messages": [
                       {"role": "system", "content": SYSTEM},
                       {"role": "user", "content": agent._prompt(filled.question)},
                       {"role": "assistant", "content": choice.model_dump_json(exclude_defaults=True)}]}
            (val if held_out else train).append(out)
            cover[f"tool {choice.tool}"] += 1
            cover[f"group {row.get('group')}"] += 1
            cover[f"asker {'class rep' if filled.asker.startswith('ST-') else 'teacher'}"] += 1
            for field in choice.model_dump(exclude_defaults=True):
                cover[f"field {field}"] += 1
    return train, val, dropped, cover


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fills", type=int, default=16, help="questions per template (spread over the departments)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    from web.app import World

    instance = World(parser_mode="offline", seed_history=False, deadline=60).instance
    train, val, dropped, cover = build(instance, a.fills, a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    random.Random(a.seed).shuffle(train)  # the templates mixed, not one after another
    for name, rows in (("train", train), ("val", val)):
        with (a.out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    manifest = {"task": "lookup", "train": len(train), "val": len(val), "fills": a.fills, "seed": a.seed,
                "templates": TEMPLATES.name, "dropped": len(dropped), "cover": dict(sorted(cover.items()))}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{len(train)} train and {len(val)} validation rows in {a.out}; {len(dropped)} left out")
    for k, v in sorted(cover.items()):
        print(f"  {k:28} {v}")
    for d in dropped[:30]:
        print("LEFT OUT", d)


if __name__ == "__main__":
    main()
