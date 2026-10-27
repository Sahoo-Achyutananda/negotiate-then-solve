"""nts-memory's reader (agents.memory): what it sends, what it checks, the chat as turns. No API calls."""

from datetime import date, datetime

import pytest

from agents.memory import HISTORY, Memory, Rewrite, Turn, turns_of
from web.app import World

AT = datetime(2026, 10, 27, 15, 10)


class FakeModel:
    def __init__(self, *answers) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def generate(self, system, prompt, schema):
        self.prompts.append(prompt)
        return self.answers.pop(0)


@pytest.fixture(scope="module")
def instance():
    return World(parser_mode="offline", seed_history=False, deadline=60).instance


def menon(instance):
    return next(f.id for f in instance.faculty if f.name == "Dr. Menon")


def memory(instance, *answers):
    model = FakeModel(*answers)
    return Memory(instance, model, menon(instance), 13, now=AT), model


HISTORY_RAO = [Turn("you", "when is Dr. Iyer free on Monday?", AT), Turn("assistant", "Dr. Iyer is free on Monday "
                                                                                      "9-11 am and 2-4 pm (week 13).", AT)]


def test_no_history_no_call(instance):
    m, model = memory(instance)
    assert m.rewrite("and Friday?", []).text == "and Friday?" and not model.prompts


def test_a_follow_up_is_written_in_full(instance):
    m, model = memory(instance, Rewrite(message="when is Dr. Iyer free on Friday?"))
    out = m.rewrite("and Friday?", HISTORY_RAO)
    assert out.changed and out.text == "when is Dr. Iyer free on Friday?" and out.ask is None
    p = model.prompts[0]
    assert "<message>and Friday?</message>" in p and "You (15:10): when is Dr. Iyer free on Monday?" in p
    assert "Asking: Dr. Menon" in p and "week 13 of" in p


def test_a_message_that_stands_alone_goes_on_unchanged(instance):
    m, _ = memory(instance, Rewrite(same=True))
    out = m.rewrite("how big is Section 4?", HISTORY_RAO)
    assert not out.changed and out.text == "how big is Section 4?"


def test_it_asks_back_when_the_conversation_does_not_have_it(instance):
    m, _ = memory(instance, Rewrite(ask="Which room do you mean?"))
    out = m.rewrite("is that room free at 3?", HISTORY_RAO)
    assert out.ask == "Which room do you mean?" and out.text == "is that room free at 3?"


def test_a_name_from_nowhere_is_sent_back_then_the_message_goes_as_written(instance):
    m, model = memory(instance, Rewrite(message="when is Dr. Pillai free on Friday?"),
                      Rewrite(message="when is Dr. Iyer free on Friday?"))
    out = m.rewrite("and Friday?", HISTORY_RAO)
    assert out.text == "when is Dr. Iyer free on Friday?"
    assert "Dr. Pillai: not in the message or the conversation" in model.prompts[1]
    m, _ = memory(instance, Rewrite(message="when is Lab 3 free?"), Rewrite(message="when is R-00099 free?"))
    out = m.rewrite("and Friday?", HISTORY_RAO)
    assert not out.changed and out.text == "and Friday?" and len(out.trace) == 2


def test_short_forms_in_the_message_count_as_named(instance):
    m, _ = memory(instance, Rewrite(message="when is Dr. Iyer free on Friday, and Section 3?"))
    assert m.rewrite("and sec 3, friday?", HISTORY_RAO).changed  # "Section 3" is in "sec 3"
    history = [Turn("you", "move my Monday class to Tuesday", AT), Turn("assistant", "Request R-00021 sent.", AT)]
    m, _ = memory(instance, Rewrite(message="has R-00021 been approved?"))
    assert m.rewrite("has it been approved?", history).text == "has R-00021 been approved?"


def test_two_answers_at_once_are_sent_back(instance):
    m, model = memory(instance, Rewrite(same=True, message="x"), Rewrite(same=True))
    assert not m.rewrite("and Friday?", HISTORY_RAO).changed and "exactly one" in model.prompts[1]


def test_only_the_last_messages_are_read_and_long_replies_are_cut(instance):
    old = [Turn("you", f"question {k}", AT) for k in range(6)]
    m, model = memory(instance, Rewrite(same=True))
    m.rewrite("hi", old + [Turn("assistant", "word " * 200, AT)])
    p = model.prompts[0]
    assert "question 0" not in p and f"question {6 - HISTORY + 1}" in p and p.count("word") < 100 and "…" in p


def test_the_chat_as_turns(instance):
    chat = [{"n": 1, "at": "2026-10-26T10:00:00", "role": "you", "kind": "text", "text": "yesterday's"},
            {"n": 2, "at": "2026-10-27T15:00:00", "role": "you", "kind": "text", "text": "am I free Friday?"},
            {"n": 3, "at": "2026-10-27T15:00:01", "role": "assistant", "kind": "lookup", "answer": "Yes, 9-11 am."},
            {"n": 4, "at": "2026-10-27T15:01:00", "role": "assistant", "kind": "request", "case": "R-00021",
             "request_text": "move my DS lecture to Friday 9 am"}]
    turns = turns_of(chat, today=date(2026, 10, 27))
    assert [(t.who, t.text) for t in turns] == [
        ("you", "am I free Friday?"), ("assistant", "Yes, 9-11 am."),
        ("assistant", "Request R-00021 sent: “move my DS lecture to Friday 9 am”.")]


def test_a_follow_up_in_the_history_is_as_it_was_read(instance):
    chat = [{"n": 1, "at": "2026-10-27T15:00:00", "role": "you", "kind": "text", "text": "and Friday?",
             "read_as": "when is Dr. Iyer free on Friday?"}]
    assert turns_of(chat)[0].text == "when is Dr. Iyer free on Friday?"  # "and Monday?" next is about Dr. Iyer


def test_the_office_is_named(instance):
    m = Memory(instance, FakeModel(), "C-TT", 13, now=AT)
    assert "Asking: the timetable office." in m.prompt("x", HISTORY_RAO)
