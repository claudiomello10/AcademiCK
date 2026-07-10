"""Shared fixtures: real Postgres/Redis/Qdrant/embedding/intent services.

Only the LLM is faked in the default tier (see fake_llm); if the embedding or
intent service is down, this suite fails. The app config validates environment
at import time, so the env check runs before any `app.*` import.
scripts/run-tests.sh derives these variables from the running dev stack; CI
sets them for its service containers.
"""

import os
import uuid

_REQUIRED_ENV = (
    "DATABASE_URL",
    "REDIS_URL",
    "QDRANT_HOST",
    "EMBEDDING_SERVICE_URL",
    "INTENT_SERVICE_URL",
    "SNAPSHOT_DIR",
    "SESSION_SECRET",
    "ADMIN_PASSWORD",
    "GUEST_PASSWORD",
    "AVAILABLE_MODELS",
    "DEFAULT_MODEL_FRONTEND",
)
_missing = [name for name in _REQUIRED_ENV if not os.getenv(name)]
if _missing:
    raise RuntimeError(
        f"Missing environment for tests: {', '.join(_missing)}. "
        "Run via scripts/run-tests.sh, which derives them from the dev stack."
    )

import httpx
import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport
from qdrant_client.models import PointStruct, SparseVector

from app.config import settings
from app.main import app as fastapi_app

# Seeded books use real embeddings of fictional topics, so retrieval quality
# assertions stay coarse ("the book is cited"), never rank- or score-exact.
BOOK_TOPICS = {
    0: {
        "topic": "Zorbite Consolidation",
        "question": "What is the Zorbite consolidation algorithm and what does it do?",
        "chunks": [
            "The Zorbite consolidation algorithm merges overlapping evidence "
            "fragments into a single ranked ledger while preserving the "
            "provenance of every fragment it absorbs.",
            "Because the Zorbite consolidation algorithm compacts its ledger "
            "after every merge, repeated runs produce identical output, which "
            "makes downstream audits straightforward.",
            "Students remember the Zorbite consolidation algorithm because it "
            "trades a small amount of recall for a large improvement in the "
            "precision of the final ledger.",
        ],
    },
    1: {
        "topic": "Quillmark Indexing",
        "question": "How does the Quillmark indexing ritual organize manuscripts?",
        "chunks": [
            "The Quillmark indexing ritual assigns every manuscript a "
            "three-part sigil derived from its opening sentence, its scribe, "
            "and the season of its binding.",
            "Archivists performed the Quillmark indexing ritual at dusk "
            "because candle smoke was believed to fix the sigil ink "
            "permanently into the catalogue page.",
            "Modern libraries simulate the Quillmark indexing ritual in "
            "software, keeping the sigil scheme while discarding the candles "
            "and the dusk requirement.",
        ],
    },
}


async def embed_texts(texts: list) -> dict:
    """Embed through the real embedding service (GPU locally, CPU in CI)."""
    async with httpx.AsyncClient(timeout=300.0) as c:
        r = await c.post(
            f"{settings.embedding_service_url}/embed",
            json={"texts": texts, "return_sparse": True},
        )
        r.raise_for_status()
        return r.json()


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def block_real_llm(request, monkeypatch):
    """Hard guarantee that unmarked tests can never reach a real LLM."""
    if request.node.get_closest_marker("llm") is None:
        import pydantic_ai.models

        monkeypatch.setattr(pydantic_ai.models, "ALLOW_MODEL_REQUESTS", False)


@pytest.fixture(scope="session")
async def app():
    async with LifespanManager(fastapi_app, startup_timeout=60):
        yield fastapi_app


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    # Generous timeout: llm-marked tests run real models through this client.
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", timeout=300.0
    ) as c:
        yield c


@pytest.fixture
async def admin_token(client):
    r = await client.post(
        "/api/v1/login",
        json={"username": "admin", "password": os.environ["ADMIN_PASSWORD"]},
    )
    assert r.status_code == 200, r.text
    return r.json()["session_id"]


@pytest.fixture
async def guest_token(client):
    r = await client.post(
        "/api/v1/login",
        json={"username": "guest", "password": os.environ["GUEST_PASSWORD"]},
    )
    assert r.status_code == 200, r.text
    return r.json()["session_id"]


@pytest.fixture
async def make_user(app, client, admin_token):
    """Factory creating a DB user of any role, logged in; rows removed afterwards.

    Students get a unique registration number automatically (it is mandatory).
    """
    created = []

    async def _make(role: str = "user", **overrides) -> dict:
        username = overrides.get("username", f"test-{role}-{uuid.uuid4().hex[:8]}")
        password = overrides.get("password", f"pw-{uuid.uuid4().hex[:8]}")
        payload = {"username": username, "password": password, "role": role}
        if "registration_number" in overrides:
            payload["registration_number"] = overrides["registration_number"]
        elif role == "user":
            payload["registration_number"] = f"RA{uuid.uuid4().hex[:10]}"

        r = await client.post(
            "/api/v1/admin/users", json=payload, headers=auth(admin_token)
        )
        assert r.status_code == 200, r.text
        created.append(username)
        body = r.json()

        r = await client.post(
            "/api/v1/login", json={"username": username, "password": password}
        )
        assert r.status_code == 200, r.text

        return {
            "id": body["id"],
            "username": username,
            "password": password,
            "role": role,
            "registration_number": body.get("registration_number"),
            "token": r.json()["session_id"],
        }

    yield _make

    async with app.state.db_pool.acquire() as conn:
        for username in created:
            await conn.execute("DELETE FROM users WHERE username = $1", username)


@pytest.fixture
async def seed_book(app):
    """Factory that seeds a uniquely-named book into Postgres and Qdrant.

    Chunks are real bge-m3 embeddings of a fictional topic (see BOOK_TOPICS),
    so semantically matching queries retrieve them without any vector faking.
    Everything created is torn down afterwards.
    """
    created = []

    async def _seed(topic: int = 0) -> dict:
        spec = BOOK_TOPICS[topic]
        n_chunks = len(spec["chunks"])
        name = f"test-book-{uuid.uuid4().hex[:8]}"
        book_id = str(uuid.uuid4())
        chapter_id = str(uuid.uuid4())

        async with app.state.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO books (id, name, file_path, processing_status, total_chunks)
                VALUES ($1, $2, $3, 'completed', $4)
                """,
                book_id, name, f"/tmp/{name}.pdf", n_chunks,
            )
            await conn.execute(
                """
                INSERT INTO chapters (id, book_id, title, chapter_number, chunk_count)
                VALUES ($1, $2, 'Chapter One', 1, $3)
                """,
                chapter_id, book_id, n_chunks,
            )

        embeddings = await embed_texts(spec["chunks"])
        points = []
        async with app.state.db_pool.acquire() as conn:
            for i, text in enumerate(spec["chunks"]):
                chunk_id = str(uuid.uuid4())
                point_id = str(uuid.uuid4())
                await conn.execute(
                    """
                    INSERT INTO chunks (id, book_id, chapter_id, qdrant_point_id,
                                        text, topic, chunk_index, char_count)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    chunk_id, book_id, chapter_id, point_id, text,
                    spec["topic"], i, len(text),
                )
                sparse = embeddings["sparse_embeddings"][i]
                points.append(
                    PointStruct(
                        id=point_id,
                        vector={
                            "dense": embeddings["dense_embeddings"][i],
                            "sparse": SparseVector(
                                indices=[int(k) for k in sparse.keys()],
                                values=[float(v) for v in sparse.values()],
                            ),
                        },
                        payload={
                            "chunk_id": chunk_id,
                            "book_id": book_id,
                            "book_name": name,
                            "chapter_id": chapter_id,
                            "chapter_title": "Chapter One",
                            "topic": spec["topic"],
                            "text": text,
                            "is_introduction": i == 0,
                            "chunk_index": i,
                            "page_number": i + 1,
                        },
                    )
                )

        await app.state.qdrant.client.upsert(
            collection_name=settings.qdrant_collection, points=points, wait=True
        )
        app.state.qdrant.invalidate_catalog_cache()
        created.append(name)
        return {
            "name": name,
            "id": book_id,
            "chapter_id": chapter_id,
            "n_chunks": n_chunks,
        }

    yield _seed

    for name in created:
        try:
            await app.state.qdrant.delete_by_book_name(name)
        except Exception:
            pass
        async with app.state.db_pool.acquire() as conn:
            await conn.execute("DELETE FROM books WHERE name = $1", name)
    app.state.qdrant.invalidate_catalog_cache()
    # Cached search results may cite the books just deleted.
    if created:
        cache_keys = await app.state.redis.keys("search:*")
        if cache_keys:
            await app.state.redis.delete(*cache_keys)


@pytest.fixture
async def guest_classroom(app, client, admin_token, guest_token, make_user):
    """A class the guest is enrolled in and has selected as active.

    Chat and book listings are class-scoped, so tests exercising them enroll
    the guest here and attach seeded books via `attach(book_id)`.
    """
    professor = await make_user(role="professor")
    r = await client.post(
        "/api/v1/professor/classes",
        json={"name": f"guest-class-{uuid.uuid4().hex[:8]}", "subject": "Machine Learning"},
        headers=auth(professor["token"]),
    )
    assert r.status_code == 200, r.text
    cls = r.json()

    async with app.state.db_pool.acquire() as conn:
        guest_id = await conn.fetchval("SELECT id FROM users WHERE username = 'guest'")
    r = await client.post(
        f"/api/v1/admin/classes/{cls['id']}/students",
        json={"user_id": str(guest_id)},
        headers=auth(admin_token),
    )
    assert r.status_code == 200, r.text

    r = await client.post(
        "/api/v1/session/class",
        json={"class_id": cls["id"]},
        headers=auth(guest_token),
    )
    assert r.status_code == 200, r.text

    async def attach(book_id: str) -> None:
        r = await client.post(
            f"/api/v1/admin/classes/{cls['id']}/books/{book_id}/attach",
            headers=auth(admin_token),
        )
        assert r.status_code == 200, r.text

    yield {"id": cls["id"], "join_code": cls["join_code"], "attach": attach,
           "professor": professor}
    # Class rows cascade when make_user removes the professor.


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace both model factories with deterministic test models.

    The curation agent gets a FunctionModel that performs one real `search`
    tool call (hitting the stubbed embedder and the real Qdrant) and then
    approves everything found; the resolver and answer agents get TestModel.
    Returned state dict lets a test scope the search to a book or force the
    NOT_IN_KB decision.
    """
    from pydantic_ai.messages import (
        ModelResponse, ToolCallPart, ToolReturnPart, UserPromptPart,
    )
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.models.test import TestModel

    from app.config import settings as app_settings
    from app.services import rag_orchestrator, reasoning_agent

    state = {
        "book": None,
        "decision": "APPROVE",
        # Unique suffix per test: search results are cached in the shared
        # dev Redis under a hash of the query, so reuse would serve stale books.
        "query": f"{BOOK_TOPICS[0]['question']} ({uuid.uuid4().hex[:8]})",
    }

    def curation_fn(messages, info):
        output_tool = info.output_tools[0].name
        if state["decision"] == "NOT_IN_KB":
            return ModelResponse(
                parts=[ToolCallPart(
                    tool_name=output_tool,
                    args={"action": "NOT_IN_KB", "reasoning": "test", "keep_indices": []},
                )]
            )
        searched = any(
            isinstance(part, ToolReturnPart)
            for m in messages
            for part in getattr(m, "parts", [])
        )
        if not searched:
            return ModelResponse(
                parts=[ToolCallPart(
                    tool_name="search",
                    args={
                        "queries": [{"query": state["query"], "book": state["book"]}],
                        "keep": [],
                    },
                )]
            )
        return ModelResponse(
            parts=[ToolCallPart(
                tool_name=output_tool,
                args={
                    "action": "APPROVE",
                    "reasoning": "test",
                    "keep_indices": list(range(1, 60)),
                },
            )]
        )

    def resolver_fn(messages, info):
        """Echo the raw user query as the resolved query, so downstream
        consumers (topic classification) see real text, not TestModel noise."""
        user_text = ""
        for m in messages:
            for part in getattr(m, "parts", []):
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    user_text = part.content
        return ModelResponse(
            parts=[ToolCallPart(
                tool_name=info.output_tools[0].name,
                args={"resolved_query": user_text},
            )]
        )

    def orchestrator_model(model_name=None, *a, **k):
        if model_name == app_settings.query_enhancement_model:
            return FunctionModel(resolver_fn)
        return TestModel()

    monkeypatch.setattr(
        reasoning_agent, "build_model", lambda *a, **k: FunctionModel(curation_fn)
    )
    monkeypatch.setattr(rag_orchestrator, "build_model", orchestrator_model)
    return state
