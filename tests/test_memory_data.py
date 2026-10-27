"""nts-memory's data (training.memory_data) and its scorer (evaluation.memory_set). Small builds, no API calls."""

import json

import pytest

from agents.memory import SYSTEM, Rewritten
from evaluation.memory_set import features, right
from training.memory_data import FORMS, HINGLISH, SHARES, build
from web.app import World


@pytest.fixture(scope="module")
def world():
    return World(parser_mode="offline", seed_history=False, deadline=60)


@pytest.fixture(scope="module")
def built(world):
    return build(world.instance, world.store, 60, test=False, seed=0)[0]


def test_every_row_has_a_conversation_and_one_answer(world):
    built = build(world.instance, world.store, 2 * sum(SHARES.values()), test=False, seed=0)[0]
    assert {r["kind"] for r in built} == set(SHARES)  # every kind can be made
    for r in built:
        system, user, answer = r["messages"]
        assert system["content"] == SYSTEM and "<conversation>" in user["content"] and "<message>" in user["content"]
        assert len(json.loads(answer["content"])) == 1  # exactly one of same, message, ask
        assert r["meta"]["history"] and r["meta"]["history"][-1][0] == "assistant"


def test_a_change_keeps_everything_but_the_one_thing(built):
    for r in (r for r in built if r["kind"] == "change"):
        asked = r["meta"]["history"][-2][1]
        full = r["meta"]["answer"]["message"]
        assert full != asked and len(set(full.split()) ^ set(asked.split())) <= 8, (asked, full)


def test_the_test_set_has_its_own_wordings(world):
    test = build(world.instance, world.store, 40, test=True, seed=0)[0]
    assert test and {r["department"] for r in test} <= {"test-renamed-0", "test-renamed-1", "test-renamed-2"}
    held_out = {f for forms in [*FORMS.values(), *HINGLISH.values()] for f in forms[-2:]}
    changes = [r["meta"]["message"] for r in test if r["kind"] == "change"]
    assert changes and all(any(_matches(f, m) for f in held_out) for m in changes)


def _matches(form: str, message: str) -> bool:
    import re

    pattern = re.escape(form).replace(r"\{v\}", ".+").replace(r"\{old\}", ".+")
    return re.fullmatch(pattern, message) is not None


def test_the_scorer_compares_what_two_wordings_name(world):
    i = world.instance
    me = next(f.id for f in i.faculty if f.name == "Dr. Menon")
    f = lambda t: features(t, i, me, 13)  # noqa: E731
    assert f("when is Dr. Iyer free on Friday?") == f("is iyer free friday")
    assert f("is sec 2 free at 3 pm?") == f("is Section 2 free at 15:00")
    assert f("when is Dr. Iyer free on Friday?") != f("when is Dr. Iyer free on Monday?")
    assert f("has R-00021 been approved?") != f("has R-00022 been approved?")
    assert f("move my DS lecture to Thursday 2 pm") != f("is anyone free Thursday 2 pm")  # a request, a question
    row = {"meta": {"answer": {"message": "when is Dr. Iyer free on Friday?"}, "asker": me, "week": 13}}
    assert right(row, Rewritten("is Iyer free on friday", changed=True), i)
    assert not right(row, Rewritten("and Friday?"), i)
    assert right({"meta": {"answer": {"same": True}, "asker": me, "week": 13}}, Rewritten("x"), i)
