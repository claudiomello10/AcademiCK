"""Conversation management endpoints."""

from tests.conftest import auth


async def test_conversation_lifecycle(client, guest_token):
    r = await client.post(
        "/api/v1/conversations/new",
        json={"title": "Test Conversation"},
        headers=auth(guest_token),
    )
    assert r.status_code == 200, r.text
    conversation_id = r.json()["conversation_id"]

    r = await client.get("/api/v1/conversations", headers=auth(guest_token))
    assert r.status_code == 200
    listed = {c["id"]: c for c in r.json()["conversations"]}
    assert conversation_id in listed
    assert listed[conversation_id]["title"] == "Test Conversation"

    r = await client.put(
        "/api/v1/conversations/current/title",
        json={"title": "Renamed"},
        headers=auth(guest_token),
    )
    assert r.status_code == 200

    r = await client.get("/api/v1/conversations", headers=auth(guest_token))
    assert next(
        c for c in r.json()["conversations"] if c["id"] == conversation_id
    )["title"] == "Renamed"

    r = await client.delete(
        f"/api/v1/conversations/{conversation_id}", headers=auth(guest_token)
    )
    assert r.status_code == 200

    r = await client.get("/api/v1/conversations", headers=auth(guest_token))
    assert conversation_id not in {c["id"] for c in r.json()["conversations"]}


async def test_resume_conversation_returns_its_messages(client, guest_token):
    r = await client.post(
        "/api/v1/conversations/new",
        json={"title": "Resumable"},
        headers=auth(guest_token),
    )
    conversation_id = r.json()["conversation_id"]

    r = await client.get(
        f"/api/v1/conversations/resume/{conversation_id}", headers=auth(guest_token)
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["conversation_id"] == conversation_id
    assert body["messages"] == []

    await client.delete(
        f"/api/v1/conversations/{conversation_id}", headers=auth(guest_token)
    )


async def test_resume_unknown_conversation_is_404(client, guest_token):
    r = await client.get(
        "/api/v1/conversations/resume/00000000-0000-0000-0000-000000000000",
        headers=auth(guest_token),
    )
    assert r.status_code == 404


async def test_clear_history_empties_the_conversation(
    client, guest_token, guest_classroom, seed_book, fake_llm
):
    from tests.conftest import BOOK_TOPICS

    book = await seed_book()
    await guest_classroom["attach"](book["id"])
    r = await client.post(
        "/api/v1/chat",
        json={"query": BOOK_TOPICS[0]["question"]},
        headers=auth(guest_token),
    )
    assert r.status_code == 200

    r = await client.get("/api/v1/chat/history", headers=auth(guest_token))
    assert len(r.json()["messages"]) == 2

    r = await client.delete("/api/v1/chat/history", headers=auth(guest_token))
    assert r.status_code == 200

    r = await client.get("/api/v1/chat/history", headers=auth(guest_token))
    assert r.json()["messages"] == []
