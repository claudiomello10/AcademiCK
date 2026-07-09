"""Shared fixtures: real Postgres/Redis/Qdrant, faked LLM and ML services.

The app config validates environment at import time, so the env check runs
before any `app.*` import. scripts/run-tests.sh derives these variables from
the running dev stack; CI sets them for its service containers.
"""

import json
import os
import uuid

_REQUIRED_ENV = (
    "DATABASE_URL",
    "REDIS_URL",
    "QDRANT_HOST",
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
import respx
from asgi_lifespan import LifespanManager
from httpx import ASGITransport
from qdrant_client.models import PointStruct, SparseVector

from app.config import settings
from app.main import app as fastapi_app

DENSE_DIM = 1024
# Sparse term ids: the embed stub and book seeds with direction 0 share id 7,
# so seeded chunks win both the dense and the sparse leg of hybrid search.
SPARSE_IDX_BY_DIRECTION = {0: 7, 1: 8}


def unit_vec(direction: int) -> list:
    vec = [0.0] * DENSE_DIM
    vec[direction] = 1.0
    return vec


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
async def seed_book(app):
    """Factory that seeds a uniquely-named book into Postgres and Qdrant.

    Chunks carry a synthetic dense unit vector along `direction`, so the embed
    stub (which returns direction 0) makes direction-0 books the top hits.
    Everything created is torn down afterwards.
    """
    created = []

    async def _seed(direction: int = 0, n_chunks: int = 3) -> dict:
        name = f"test-book-{uuid.uuid4().hex[:8]}"
        book_id = str(uuid.uuid4())
        chapter_id = str(uuid.uuid4())
        sparse_idx = SPARSE_IDX_BY_DIRECTION[direction]

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

        points = []
        async with app.state.db_pool.acquire() as conn:
            for i in range(n_chunks):
                chunk_id = str(uuid.uuid4())
                point_id = str(uuid.uuid4())
                text = (
                    f"Test content {i} of {name}: neural networks learn layered "
                    f"representations of data through gradient descent."
                )
                await conn.execute(
                    """
                    INSERT INTO chunks (id, book_id, chapter_id, qdrant_point_id,
                                        text, topic, chunk_index, char_count)
                    VALUES ($1, $2, $3, $4, $5, 'Neural Networks', $6, $7)
                    """,
                    chunk_id, book_id, chapter_id, point_id, text, i, len(text),
                )
                points.append(
                    PointStruct(
                        id=point_id,
                        vector={
                            "dense": unit_vec(direction),
                            "sparse": SparseVector(indices=[sparse_idx], values=[1.0]),
                        },
                        payload={
                            "chunk_id": chunk_id,
                            "book_id": book_id,
                            "book_name": name,
                            "chapter_id": chapter_id,
                            "chapter_title": "Chapter One",
                            "topic": "Neural Networks",
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


@pytest.fixture
def fake_ml(app):
    """Stub the embedding and intent HTTP services at the wire with respx.

    Qdrant also speaks HTTP, so its host is passed through untouched.
    """

    def _embed(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["texts"]
        return httpx.Response(
            200,
            json={
                "dense_embeddings": [unit_vec(0)] * len(texts),
                "sparse_embeddings": [{"7": 1.0}] * len(texts),
            },
        )

    with respx.mock(assert_all_called=False) as router:
        router.route(host=settings.qdrant_host).pass_through()
        router.post(f"{settings.embedding_service_url}/embed").mock(side_effect=_embed)
        router.post(f"{settings.intent_service_url}/classify").mock(
            return_value=httpx.Response(
                200, json={"intent": "question_answering", "confidence": 0.99}
            )
        )
        yield router


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace both model factories with deterministic test models.

    The curation agent gets a FunctionModel that performs one real `search`
    tool call (hitting the stubbed embedder and the real Qdrant) and then
    approves everything found; the resolver and answer agents get TestModel.
    Returned state dict lets a test scope the search to a book or force the
    NOT_IN_KB decision.
    """
    from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.models.test import TestModel

    from app.services import rag_orchestrator, reasoning_agent

    state = {
        "book": None,
        "decision": "APPROVE",
        # Unique query text per test: search results are cached in the shared
        # dev Redis under a hash of the query, so reuse would serve stale books.
        "query": f"neural networks {uuid.uuid4().hex[:8]}",
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

    monkeypatch.setattr(
        reasoning_agent, "build_model", lambda *a, **k: FunctionModel(curation_fn)
    )
    monkeypatch.setattr(
        rag_orchestrator, "build_model", lambda *a, **k: TestModel()
    )
    return state
