"""The chat (/api/chat): one conversation per person; timetable questions answered in it, everything else
made a request that the chat follows with a card. Offline: the regex reads the questions."""

import pytest
from fastapi.testclient import TestClient

from pipeline.store import Store
from web.app import World, create_web_app


@pytest.fixture(scope="module")
def client():
    world = World(parser_mode="offline", seed_history=False, deadline=60)
    return TestClient(create_web_app(world)), world


def faculty(world, name):
    return next(f.id for f in world.instance.faculty if f.name == name)


def send(c, person, text):
    r = c.post("/api/chat", json={"text": text}, headers={"X-User": person})
    assert r.status_code == 200, r.text
    return r.json()


def chat(c, person, after=0):
    return c.get(f"/api/chat?after={after}", headers={"X-User": person}).json()


def test_a_timetable_question_is_answered_in_the_chat(client):
    c, world = client
    me = faculty(world, "Dr. Menon")
    before = len(world.store.requests())
    mine, reply = send(c, me, "What are my free slots this week?")
    assert (mine["role"], mine["kind"], mine["text"]) == ("you", "text", "What are my free slots this week?")
    assert (reply["role"], reply["kind"]) == ("assistant", "lookup")
    assert reply["answer"].startswith("You are free on") and reply["call"]["tool"] == "free_hours" and reply["free"]
    assert len(world.store.requests()) == before  # a question is not a request


def confirm(c, person, n, as_):
    r = c.post("/api/chat/confirm", json={"n": n, "as": as_}, headers={"X-User": person})
    return r


def test_a_request_is_sent_only_once_confirmed(client):
    c, world = client
    me = faculty(world, "Dr. Menon")
    before = len(world.store.requests())
    text = "I am at a conference on Monday of week 15, please cancel my classes that day."
    _, ask = send(c, me, text)
    assert ask["kind"] == "confirm" and ask["text"] == text
    assert len(world.store.requests()) == before  # nothing sent yet
    card = confirm(c, me, ask["n"], "request").json()[0]
    assert card["kind"] == "request" and world.store.request(card["case"]).sender_id == me
    assert "status" not in card  # only the number: what happens next is on the request's page
    assert confirm(c, me, ask["n"], "request").status_code == 409  # answered once


def test_it_is_a_question_reads_it_again_without_the_request_words(client):
    c, world = client
    me = faculty(world, "Dr. Menon")
    _, ask = send(c, me, "please tell me my free slots this week")  # "please": request words
    assert ask["kind"] == "confirm"
    replies = confirm(c, me, ask["n"], "question").json()
    assert replies[0]["kind"] in ("lookup", "other")  # offline the regex still sees "please"; online the model reads it
    assert c.get("/api/chat", headers={"X-User": me}).json()[-2]["answered"] == "question"


def test_cancel_drops_it(client):
    c, world = client
    me = faculty(world, "Dr. Menon")
    before = len(world.store.requests())
    _, ask = send(c, me, "Please move my Wednesday lecture to Thursday in week 15.")
    assert confirm(c, me, ask["n"], "cancel").json() == [] and len(world.store.requests()) == before


def test_only_your_own_confirm_can_be_answered(client):
    c, world = client
    _, ask = send(c, faculty(world, "Dr. Menon"), "Please cancel my Friday class in week 15.")
    assert confirm(c, faculty(world, "Dr. Rao"), ask["n"], "request").status_code == 404


def test_a_rule_question_is_answered_with_its_rules_and_no_request(client):
    c, world = client
    before = len(world.store.requests())
    _, reply = send(c, faculty(world, "Dr. Menon"), "how many online classes can a course have?")
    assert reply["kind"] == "rule" and reply["rules"] == [{"id": "P-ONLINE", "title": "Online and covered classes"}]
    assert "online" in reply["answer"].lower()
    assert len(world.store.requests()) == before


def test_a_greeting_is_not_a_request(client):
    c, world = client
    _, reply = send(c, faculty(world, "Dr. Menon"), "hii")
    assert reply["kind"] == "other" and "timetable" in reply["text"]


def test_the_chat_is_kept_in_order_and_read_after_a_number(client):
    c, world = client
    me = faculty(world, "Dr. Iyer")
    first = send(c, me, "What are my free slots this week?")
    second = send(c, me, "what do I teach on wednesday?")
    everything = chat(c, me)
    assert [m["n"] for m in everything] == sorted(m["n"] for m in everything)
    assert everything[-4:] == first + second
    assert chat(c, me, after=first[-1]["n"]) == second


def test_nobody_sees_anyone_elses_chat(client):
    c, world = client
    rao = faculty(world, "Dr. Rao")
    send(c, faculty(world, "Dr. Khan"), "What are my free slots this week?")
    assert all(m.get("text") != "What are my free slots this week?" for m in chat(c, rao))


def test_clearing_the_chat_keeps_the_requests(client):
    c, world = client
    me = faculty(world, "Dr. Das")
    _, ask = send(c, me, "Please cancel my classes on Tuesday of week 15, I am away.")
    card = confirm(c, me, ask["n"], "request").json()[0]
    r = c.delete("/api/chat", headers={"X-User": me})
    assert r.json()["cleared"] == 3 and chat(c, me) == []
    assert world.store.request(card["case"]) is not None


def test_a_request_made_in_the_timetable_appears_in_the_chat(client):
    c, world = client
    me = faculty(world, "Dr. Gupta")
    mine = next(s for s in world.instance.sessions if s.faculty == me)
    intent = {"kind": "cancel", "which": {"session": mine.id}, "period": {"weeks": [15]}}
    r = c.post("/api/requests/form", json={"text": "Cancel my class in week 15.", "intent": intent}, headers={"X-User": me})
    assert r.status_code == 200, r.text
    you, card = chat(c, me)[-2:]
    assert you["text"] == "Cancel my class in week 15." and card["kind"] == "request" and card["case"] == r.json()["id"]


def test_a_request_sent_from_the_request_builder_appears_in_the_chat(client):
    c, world = client
    me = faculty(world, "Dr. Iyer")
    text = "Please hold my Compiler Design lecture in a lab (my Compiler Design lecture on Tuesday at 9 am in week 15)."
    rid = c.post("/api/requests", json={"text": text}, headers={"X-User": me}).json()["id"]
    you, card = chat(c, me)[-2:]
    assert you["text"] == text and card["kind"] == "request" and card["case"] == rid


def finished_request(c, world, person):
    """A request of ``person`` that has run its course (a rule question, answered)."""
    rid = c.post("/api/requests", json={"text": "How many online classes can a course have?"},
                 headers={"X-User": person}).json()["id"]
    world.threads[rid].join()
    return rid


def test_a_finished_request_can_be_taken_off_your_list(client):
    c, world = client
    me = faculty(world, "Dr. Pillai")
    rid = finished_request(c, world, me)
    listed = c.get("/api/cases?scope=mine", headers={"X-User": me}).json()
    assert any(x["id"] == rid and "can_withdraw" in x for x in listed)
    assert c.post(f"/api/cases/{rid}/hide", headers={"X-User": faculty(world, "Dr. Rao")}).status_code == 403
    assert c.post(f"/api/cases/{rid}/hide", headers={"X-User": me}).status_code == 200
    assert all(x["id"] != rid for x in c.get("/api/cases?scope=mine", headers={"X-User": me}).json())
    assert world.store.request(rid) is not None  # still in the records


def test_an_unfinished_request_cannot_be_taken_off(client):
    c, world = client
    me = faculty(world, "Dr. Sharma")
    rid = c.post("/api/requests", json={"text": "please move my class"}, headers={"X-User": me}).json()["id"]
    world.threads[rid].join()  # ends asking which class: not finished
    assert world.store.request(rid).status.value == "clarification_requested"
    assert c.post(f"/api/cases/{rid}/hide", headers={"X-User": me}).status_code == 409


def test_a_message_that_waits_on_you_is_in_the_requests_tab_not_the_chat(client):
    from agents.negotiation import Message

    c, world = client
    me, other = faculty(world, "Dr. Khan"), faculty(world, "Dr. Rao")
    rid = c.post("/api/requests", json={"text": "Please move my Friday class to Monday in week 15."},
                 headers={"X-User": other}).json()["id"]
    world.threads[rid].join()
    m = Message(round=1, to=me, mus=[], offers=[], text="Which of these works for you?\n\nOptions:\nA) Monday 9 am",
                explanation_mode="choice", faithfulness=1.0)
    world.inbox.add(rid, me, m)
    assert all(x["kind"] != "notice" for x in chat(c, me))
    listed = c.get("/api/cases?scope=mine", headers={"X-User": me}).json()
    assert next(x for x in listed if x["id"] == rid)["waits_on_you"]  # a colleague's request that asks me
    assert c.get("/api/overview", headers={"X-User": me}).json()["my_waiting"] >= 1


def test_the_chat_is_saved_with_the_store():
    store = Store()
    store.add_chat("F-102", "you", "text", {"text": "hello"})
    copy = Store()
    copy.load(store.db.serialize())
    assert [m["text"] for m in copy.chat("F-102")] == ["hello"]


def test_signing_in_is_required(client):
    c, _ = client
    assert c.get("/api/chat").status_code in (401, 422)
    assert c.post("/api/chat", json={"text": "hi"}, headers={"X-User": "nobody"}).status_code == 401


# -- follow-ups (agents.memory): a stand-in for the model, which the offline world doesn't have ---------------


def test_a_follow_up_is_read_as_it_was_made_to_stand(client, monkeypatch):
    from agents.memory import Rewritten

    c, world = client
    me = faculty(world, "Dr. Iyer")
    seen = []

    def followup(text, sender, history):
        seen.append([t.text for t in history])
        return {"and Friday?": Rewritten("when am I free on Friday?", changed=True),
                "is that room free?": Rewritten("is that room free?", ask="Which room do you mean?"),
                "move my DS lecture there": Rewritten("move my DS lecture to Friday 9 am", changed=True)}.get(text)

    monkeypatch.setattr(world, "followup", followup)
    send(c, me, "when am I free on Monday?")
    mine, reply = send(c, me, "and Friday?")
    assert mine["text"] == "and Friday?" and mine["read_as"] == "when am I free on Friday?"
    assert reply["kind"] == "lookup" and "Friday" in reply["answer"]
    assert seen[-1][-2:][0] == "when am I free on Monday?"  # the chat before it, as turns
    mine, reply = send(c, me, "is that room free?")
    assert "read_as" not in mine and (reply["kind"], reply["text"]) == ("other", "Which room do you mean?")
    _, ask = send(c, me, "move my DS lecture there")
    assert ask["kind"] == "confirm" and ask["text"] == "move my DS lecture to Friday 9 am"  # sent as read
    card = confirm(c, me, ask["n"], "request").json()[0]
    assert card["request_text"] == "move my DS lecture to Friday 9 am"


def test_memory_is_off_offline(client):
    _, world = client
    assert not world.memory_on and world.followup("and Friday?", "F-101", ["x"]) is None


def test_yes_typed_to_send_as_request_sends_it(client, monkeypatch):
    from agents.memory import Rewritten

    c, world = client
    me = faculty(world, "Dr. Pillai")
    text = "please cancel my classes on Monday of week 15, I am at a conference"
    _, ask = send(c, me, text)
    assert ask["kind"] == "confirm"
    monkeypatch.setattr(world, "followup", lambda t, s, h: Rewritten(text, changed=True) if t == "yes" else None)
    before = len(world.store.requests())
    mine, card = send(c, me, "yes")
    assert mine["read_as"] == text and card["kind"] == "request" and len(world.store.requests()) == before + 1
    assert world.store.chat_message(me, ask["n"])["answered"] == "request"
