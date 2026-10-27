"""nts-lookup-v3's training data (training.lookup_data_v3) and Gemini's rewordings' check (training.lookup_distill_v3).
Small builds, no API calls."""

import json

import pytest

from agents.questions import ToolCall
from agents.tool_agent import SYSTEM_V3, Lookups
from training.lookup_data_v3 import TEMPLATES, TEST_TEMPLATES, build
from training.lookup_distill_v3 import resolved
from web.app import World


@pytest.fixture(scope="module")
def built():
    instance = World(parser_mode="offline", seed_history=False, deadline=60).instance
    return build(instance, fills=1, multi=30, replay=15, seed=0, mixed=10)


def test_every_row_is_v3_and_checked(built):
    train, val, dropped, cover = built
    rows = train + val
    assert rows and len(dropped) <= 3  # a "today" on a weekend has nothing to fill
    for r in rows:
        system, user, answer = r["messages"]
        assert system["content"] == SYSTEM_V3 and "<question>" in user["content"]
        assert 1 <= len(Lookups.model_validate_json(answer["content"]).lookups) <= 5
        assert {"asker", "now", "week"} <= set(r["meta"])
    assert cover["kind new"] and cover["kind multi"] and cover["kind replay"] and cover["kind mixed"]


def test_a_message_of_several_parts_answers_each_in_order(built):
    train, val, _, _ = built
    multi = [r for r in train + val if r["kind"] == "multi"]
    assert multi
    r = multi[0]
    assert len(json.loads(r["messages"][2]["content"])["lookups"]) == len(r["templates"])


def test_a_request_in_a_message_of_questions_is_none(built):
    train, val, _, _ = built
    for r in (r for r in train + val if r["kind"] == "mixed"):
        tools = [x["tool"] for x in json.loads(r["messages"][2]["content"])["lookups"]]
        assert tools.count("none") == 1 and len(tools) == len(r["templates"]) >= 2


def test_each_one_on_its_own_and_together_are_taught_apart(built):
    train, val, _, _ = built
    answers = {r["templates"][0]: json.loads(r["messages"][2]["content"])["lookups"][0]
               for r in train + val if r["kind"] == "new"}
    assert answers["V3-045"]["separately"] and len(answers["V3-045"]["sections"]) == 3  # "when is A, B and C free"
    assert "separately" not in answers["V3-046"]  # "when are A, B and C free"
    assert answers["V3-047"]["sections"][0].startswith("Section ")  # "sections 1, 2 and 3": each in full
    assert len(answers["V3-060"]["sections"]) == 2 and "separately" not in answers["V3-060"]  # classes_of: each


def test_training_and_test_templates_differ():
    from training.lookup_data import _shape

    train = {_shape(json.loads(x)["template"]) for x in TEMPLATES.read_text(encoding="utf-8").splitlines() if x.strip()}
    test = {_shape(json.loads(x)["template"]) for x in TEST_TEMPLATES.read_text(encoding="utf-8").splitlines() if x.strip()}
    assert not train & test


def test_a_rewording_counts_as_the_same_when_it_resolves_the_same():
    a = ToolCall("free_hours", {"teachers": ["F-102", "F-103"], "days": ["Fri"]}, 13)
    b = ToolCall("free_hours", {"days": ["Fri"], "teachers": ["F-103", "F-102"]}, 13)
    assert resolved(a) == resolved(b)
    assert resolved(ToolCall("multi", {"parts": [a, b]}, 13)) == [resolved(a)[0]] * 2
    assert resolved(a) != resolved(ToolCall("free_hours", {"teachers": ["F-102"]}, 13))
    assert resolved(None) == []
