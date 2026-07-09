"""Chat / RAG pipeline behavior: real routes, real Qdrant, faked LLM and ML."""

import json
from uuid import UUID

from tests.conftest import auth


def parse_sse(text: str) -> list:
    """Parse an SSE body into (event_type, data) tuples."""
    events = []
    for block in text.strip().split("\n\n"):
        etype, data = None, None
        for line in block.split("\n"):
            if line.startswith("event: "):
                etype = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        if etype:
            events.append((etype, data))
    return events


async def test_chat_single_cites_seeded_book(
    client, guest_token, seed_book, fake_ml, fake_llm
):
    book = await seed_book()
    r = await client.post(
        "/api/v1/chat/single",
        json={"query": "How do neural networks learn?"},
        headers=auth(guest_token),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["response"]
    assert body["sources"]
    assert body["sources"][0]["book"] == book["name"]


async def test_chat_book_filter_only_cites_that_book(
    client, guest_token, seed_book, fake_ml, fake_llm
):
    await seed_book(direction=0)
    book_b = await seed_book(direction=1)
    fake_llm["book"] = book_b["name"]

    r = await client.post(
        "/api/v1/chat/single",
        json={"query": "How do neural networks learn?"},
        headers=auth(guest_token),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sources"]
    assert {s["book"] for s in body["sources"]} == {book_b["name"]}


async def test_chat_not_in_kb_returns_answer_without_sources(
    client, guest_token, fake_ml, fake_llm
):
    fake_llm["decision"] = "NOT_IN_KB"
    r = await client.post(
        "/api/v1/chat/single",
        json={"query": "What is the meaning of life?"},
        headers=auth(guest_token),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sources"] == []
    assert body["response"]


async def test_chat_stream_emits_done_and_persists_history(
    client, guest_token, seed_book, fake_ml, fake_llm
):
    book = await seed_book()
    r = await client.post(
        "/api/v1/chat",
        json={"query": "How do neural networks learn?"},
        headers=auth(guest_token),
    )
    assert r.status_code == 200
    events = parse_sse(r.text)
    types = [e for e, _ in events]
    assert "done" in types
    assert "error" not in types

    done = next(data for e, data in events if e == "done")
    assert done["payload"]["response"]
    assert book["name"] in {s["book"] for s in done["payload"]["sources"]}

    r = await client.get("/api/v1/chat/history", headers=auth(guest_token))
    assert r.status_code == 200
    messages = r.json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]


async def test_chat_rejects_when_conversation_full(
    app, client, guest_token, fake_ml, fake_llm
):
    session = await app.state.session_service.get_session(guest_token)
    async with app.state.db_pool.acquire() as conn:
        await conn.execute(
            "UPDATE conversations SET message_count = 50 WHERE id = $1",
            UUID(session["conversation_id"]),
        )

    r = await client.post(
        "/api/v1/chat",
        json={"query": "one message too many"},
        headers=auth(guest_token),
    )
    assert r.status_code == 200
    events = parse_sse(r.text)
    types = [e for e, _ in events]
    assert "done" not in types
    error = next(data for e, data in events if e == "error")
    assert error["code"] == "conversation_full"


async def test_chat_requires_authentication(client):
    r = await client.post("/api/v1/chat/single", json={"query": "hello"})
    assert r.status_code == 401
