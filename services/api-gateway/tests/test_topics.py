"""Topic tree CRUD (embeddings at edit time) and async query classification.

Real embedding service throughout; the chat LLM is faked. Classification
runs as a background task after the SSE stream closes, so assertions on
message_topics poll with a bounded wait.
"""

import asyncio
import uuid
from uuid import UUID

import pytest

from tests.conftest import BOOK_TOPICS, auth


@pytest.fixture
async def classroom(client, make_user):
    professor = await make_user(role="professor")
    r = await client.post(
        "/api/v1/professor/classes",
        json={"name": f"topics-{uuid.uuid4().hex[:6]}", "subject": "Arquivologia"},
        headers=auth(professor["token"]),
    )
    assert r.status_code == 200, r.text
    return {"professor": professor, "class": r.json()}


async def create_topic(client, classroom, name, description=None, parent=None):
    r = await client.post(
        f"/api/v1/professor/classes/{classroom['class']['id']}/topics",
        json={"name": name, "description": description, "parent_topic_id": parent},
        headers=auth(classroom["professor"]["token"]),
    )
    return r


async def test_topic_tree_crud_with_embeddings(app, client, classroom):
    professor = classroom["professor"]
    cls = classroom["class"]

    r = await create_topic(client, classroom, "Consolidação", "Algoritmos de consolidação")
    assert r.status_code == 200, r.text
    topic = r.json()

    r = await create_topic(client, classroom, "Ledgers", parent=topic["id"])
    assert r.status_code == 200, r.text
    subtopic = r.json()

    # One nesting level only
    r = await create_topic(client, classroom, "Too deep", parent=subtopic["id"])
    assert r.status_code == 400

    # Duplicate names at the same level conflict
    r = await create_topic(client, classroom, "Consolidação")
    assert r.status_code == 409

    # Embeddings are stored at edit time
    async with app.state.db_pool.acquire() as conn:
        embedding = await conn.fetchval(
            "SELECT embedding FROM class_topics WHERE id = $1", UUID(topic["id"])
        )
    assert embedding is not None

    # Tree endpoint nests subtopics
    r = await client.get(
        f"/api/v1/professor/classes/{cls['id']}/topics",
        headers=auth(professor["token"]),
    )
    tree = r.json()["topics"]
    assert [t["name"] for t in tree] == ["Consolidação"]
    assert [s["name"] for s in tree[0]["subtopics"]] == ["Ledgers"]

    # Renaming re-embeds
    r = await client.put(
        f"/api/v1/professor/topics/{topic['id']}",
        json={"name": "Consolidação de Evidências"},
        headers=auth(professor["token"]),
    )
    assert r.status_code == 200, r.text
    async with app.state.db_pool.acquire() as conn:
        re_embedded = await conn.fetchval(
            "SELECT embedding FROM class_topics WHERE id = $1", UUID(topic["id"])
        )
    assert re_embedded != embedding

    # Deleting the parent cascades to subtopics
    r = await client.delete(
        f"/api/v1/professor/topics/{topic['id']}",
        headers=auth(professor["token"]),
    )
    assert r.status_code == 200
    r = await client.get(
        f"/api/v1/professor/classes/{cls['id']}/topics",
        headers=auth(professor["token"]),
    )
    assert r.json()["topics"] == []


async def test_foreign_professor_cannot_touch_topics(client, make_user, classroom):
    other = await make_user(role="professor")
    r = await create_topic(client, classroom, "Meu tópico")
    topic = r.json()

    r = await client.post(
        f"/api/v1/professor/classes/{classroom['class']['id']}/topics",
        json={"name": "Invasão"},
        headers=auth(other["token"]),
    )
    assert r.status_code == 403

    r = await client.delete(
        f"/api/v1/professor/topics/{topic['id']}", headers=auth(other["token"])
    )
    assert r.status_code == 403


async def wait_for_assignment(app, class_id: str, content: str, timeout: float = 10.0):
    """Poll message_topics for the row of the user message with `content`."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        async with app.state.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT mt.topic_id, mt.similarity
                FROM message_topics mt
                JOIN messages m ON m.id = mt.message_id
                WHERE mt.class_id = $1 AND m.content = $2 AND m.role = 'user'
                """,
                UUID(class_id), content,
            )
        if row:
            return row
        await asyncio.sleep(0.2)
    return None


async def test_queries_are_classified_against_class_topics(
    app, client, admin_token, guest_token, guest_classroom, seed_book, fake_llm
):
    book = await seed_book()
    await guest_classroom["attach"](book["id"])
    classroom = {"professor": guest_classroom["professor"], "class": {"id": guest_classroom["id"]}}

    r = await create_topic(
        client, classroom,
        BOOK_TOPICS[0]["topic"],
        "How the Zorbite consolidation algorithm merges evidence fragments into ledgers",
    )
    assert r.status_code == 200, r.text
    topic_id = r.json()["id"]

    # On-topic question lands on the topic
    on_topic = f"{BOOK_TOPICS[0]['question']} ({uuid.uuid4().hex[:6]})"
    fake_llm["query"] = on_topic
    r = await client.post(
        "/api/v1/chat", json={"query": on_topic}, headers=auth(guest_token)
    )
    assert r.status_code == 200

    row = await wait_for_assignment(app, guest_classroom["id"], on_topic)
    assert row is not None, "classification task never wrote message_topics"
    assert str(row["topic_id"]) == topic_id
    assert row["similarity"] > 0.4

    # Gibberish stays unclassified (NULL topic)
    gibberish = f"xyzzy plugh brillig slithy toves {uuid.uuid4().hex[:6]}"
    fake_llm["query"] = gibberish
    r = await client.post(
        "/api/v1/chat", json={"query": gibberish}, headers=auth(guest_token)
    )
    assert r.status_code == 200

    row = await wait_for_assignment(app, guest_classroom["id"], gibberish)
    assert row is not None
    assert row["topic_id"] is None


async def test_classifier_failure_does_not_break_chat(
    app, client, guest_token, guest_classroom, seed_book, fake_llm, monkeypatch
):
    book = await seed_book()
    await guest_classroom["attach"](book["id"])

    async def boom(*a, **k):
        raise RuntimeError("embedding service exploded")

    monkeypatch.setattr(
        app.state.embedding_client.__class__, "embed_batch",
        boom, raising=True,
    )

    # Chat still succeeds even though classification will fail...
    # (retrieval also uses embed_batch, so the pipeline degrades to the
    # not-found path — the stream must still complete without error)
    r = await client.post(
        "/api/v1/chat",
        json={"query": f"resiliência {uuid.uuid4().hex[:6]}"},
        headers=auth(guest_token),
    )
    assert r.status_code == 200
    assert "event: done" in r.text
