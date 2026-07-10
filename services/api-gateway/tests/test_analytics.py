"""Professor/manager topic analytics: ranking, drill-down, date filters,
permission boundaries, the env-gated summary, and per-service system status.

message_topics rows are seeded directly — classification itself is covered
in test_topics.py.
"""

import uuid
from datetime import datetime, timezone
from uuid import UUID

import pytest

from app.config import settings
from tests.conftest import auth


@pytest.fixture
async def analytics_classroom(app, client, make_user):
    """A class with a topic tree and seeded query-topic assignments."""
    professor = await make_user(role="professor")
    r = await client.post(
        "/api/v1/professor/classes",
        json={"name": f"analytics-{uuid.uuid4().hex[:6]}", "subject": "Lógica"},
        headers=auth(professor["token"]),
    )
    cls = r.json()

    async def make_topic(name, parent=None):
        r = await client.post(
            f"/api/v1/professor/classes/{cls['id']}/topics",
            json={"name": name, "parent_topic_id": parent},
            headers=auth(professor["token"]),
        )
        assert r.status_code == 200, r.text
        return r.json()

    proofs = await make_topic("Provas")
    induction = await make_topic("Indução", parent=proofs["id"])
    sets = await make_topic("Conjuntos")

    conversations = []

    async def seed_query(topic_id, content, when):
        conv_id, msg_id = uuid.uuid4(), uuid.uuid4()
        conversations.append(conv_id)
        async with app.state.db_pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO conversations (id, subject, title) VALUES ($1, 'Lógica', 't')",
                conv_id,
            )
            await conn.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, created_at)
                VALUES ($1, $2, 'user', $3, $4)
                """,
                msg_id, conv_id, content, when,
            )
            await conn.execute(
                """
                INSERT INTO message_topics (message_id, class_id, topic_id, similarity, created_at)
                VALUES ($1, $2, $3, 0.7, $4)
                """,
                msg_id, UUID(cls["id"]),
                UUID(topic_id) if topic_id else None, when,
            )

    jan = datetime(2026, 1, 10, tzinfo=timezone.utc)
    mar = datetime(2026, 3, 10, tzinfo=timezone.utc)
    await seed_query(proofs["id"], "Como estruturar uma prova direta?", jan)
    await seed_query(induction["id"], "Por que a base da indução importa?", mar)
    await seed_query(induction["id"], "Indução forte é diferente?", mar)
    await seed_query(None, "Qual a senha do wifi?", mar)

    yield {
        "professor": professor,
        "class": cls,
        "topics": {"proofs": proofs, "induction": induction, "sets": sets},
    }

    async with app.state.db_pool.acquire() as conn:
        for conv_id in conversations:
            await conn.execute("DELETE FROM conversations WHERE id = $1", conv_id)


async def test_topic_ranking_rolls_up_and_counts_unclassified(
    client, analytics_classroom
):
    professor = analytics_classroom["professor"]
    cls = analytics_classroom["class"]

    r = await client.get(
        f"/api/v1/professor/classes/{cls['id']}/analytics/topics",
        headers=auth(professor["token"]),
    )
    assert r.status_code == 200, r.text
    body = r.json()

    by_name = {t["name"]: t for t in body["topics"]}
    # Parent rollup: 1 own + 2 from the subtopic
    assert by_name["Provas"]["count"] == 3
    assert by_name["Provas"]["subtopics"][0]["name"] == "Indução"
    assert by_name["Provas"]["subtopics"][0]["count"] == 2
    # Zero-count topics still listed
    assert by_name["Conjuntos"]["count"] == 0
    assert body["unclassified"] == 1
    assert body["total"] == 4
    # Ranking is by count
    assert body["topics"][0]["name"] == "Provas"


async def test_date_range_filters_ranking(client, analytics_classroom):
    professor = analytics_classroom["professor"]
    cls = analytics_classroom["class"]

    r = await client.get(
        f"/api/v1/professor/classes/{cls['id']}/analytics/topics"
        "?from=2026-02-01&to=2026-03-31",
        headers=auth(professor["token"]),
    )
    body = r.json()
    by_name = {t["name"]: t for t in body["topics"]}
    assert by_name["Provas"]["count"] == 2  # January query excluded
    assert body["unclassified"] == 1
    assert body["total"] == 3

    r = await client.get(
        f"/api/v1/professor/classes/{cls['id']}/analytics/topics?from=not-a-date",
        headers=auth(professor["token"]),
    )
    assert r.status_code == 400


async def test_drilldown_returns_query_texts(client, analytics_classroom):
    professor = analytics_classroom["professor"]
    cls = analytics_classroom["class"]
    proofs = analytics_classroom["topics"]["proofs"]

    # Parent drill-down includes subtopic queries
    r = await client.get(
        f"/api/v1/professor/classes/{cls['id']}/analytics/topics/{proofs['id']}/queries",
        headers=auth(professor["token"]),
    )
    assert r.status_code == 200, r.text
    texts = [q["content"] for q in r.json()["queries"]]
    assert len(texts) == 3
    assert "Como estruturar uma prova direta?" in texts
    assert "Indução forte é diferente?" in texts

    r = await client.get(
        f"/api/v1/professor/classes/{cls['id']}/analytics/topics/unclassified/queries",
        headers=auth(professor["token"]),
    )
    assert [q["content"] for q in r.json()["queries"]] == ["Qual a senha do wifi?"]


async def test_analytics_permission_boundaries(
    client, make_user, analytics_classroom
):
    cls = analytics_classroom["class"]
    other_professor = await make_user(role="professor")
    manager = await make_user(role="manager")
    student = await make_user(role="user")

    url = f"/api/v1/professor/classes/{cls['id']}/analytics/topics"
    r = await client.get(url, headers=auth(other_professor["token"]))
    assert r.status_code == 403
    r = await client.get(url, headers=auth(student["token"]))
    assert r.status_code == 403

    # Managers read any class through their own router
    r = await client.get(
        f"/api/v1/manager/classes/{cls['id']}/analytics/topics",
        headers=auth(manager["token"]),
    )
    assert r.status_code == 200
    assert r.json()["total"] == 4


async def test_summary_is_env_gated_and_generates_when_enabled(
    client, analytics_classroom, monkeypatch
):
    professor = analytics_classroom["professor"]
    cls = analytics_classroom["class"]
    url = f"/api/v1/professor/classes/{cls['id']}/analytics/summary"

    # Default: flag off → 403
    r = await client.post(url, json={}, headers=auth(professor["token"]))
    assert r.status_code == 403

    from pydantic_ai.models.test import TestModel

    from app.services import analytics_summary as summary_module

    monkeypatch.setattr(settings, "analytics_summary_enabled", True)
    monkeypatch.setattr(summary_module, "build_model", lambda *a, **k: TestModel())

    r = await client.post(
        url,
        json={"from": "2026-01-01", "to": "2026-12-31"},
        headers=auth(professor["token"]),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["summary"]
    assert body["total_queries"] == 4


async def test_system_status_reports_each_service(client, admin_token, guest_token):
    r = await client.get("/api/v1/admin/system/status", headers=auth(guest_token))
    assert r.status_code == 403

    r = await client.get("/api/v1/admin/system/status", headers=auth(admin_token))
    assert r.status_code == 200, r.text
    body = r.json()
    services = {s["name"]: s for s in body["services"]}
    assert set(services) == {
        "postgres", "redis", "qdrant",
        "pdf-service", "intent-service", "embedding-service",
    }
    # Infra the tests already depend on must be healthy with a latency reading
    for name in ("postgres", "redis", "qdrant", "embedding-service", "intent-service"):
        assert services[name]["status"] == "healthy", services[name]
        assert services[name]["latency_ms"] >= 0
    assert body["overall"] in ("healthy", "degraded")
