"""Shared fixtures: real Postgres/Redis/Qdrant, faked chapter LLM and embedder.

The env check runs before any `app.*` import: config requires DATABASE_URL and
REDIS_URL at import time, and the celery app validates the chapter-detection
provider credentials on import. scripts/run-tests.sh derives everything from
the running dev stack; CI sets it for its service containers.
"""

import hashlib
import json
import os
import sys
import uuid

_REQUIRED_ENV = ("DATABASE_URL", "REDIS_URL", "QDRANT_HOST")
_missing = [name for name in _REQUIRED_ENV if not os.getenv(name)]
if _missing:
    raise RuntimeError(
        f"Missing environment for tests: {', '.join(_missing)}. "
        "Run via scripts/run-tests.sh, which derives them from the dev stack."
    )

import asyncpg
import httpx
import pytest
import respx
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.config import settings

DENSE_DIM = 1024

# Long sentences keep the period density far below the 2% skip threshold.
_SENTENCES = [
    "Neural networks learn layered representations of raw data by repeatedly "
    "adjusting millions of connection weights with gradient descent so that "
    "the training loss shrinks a little bit further on every single pass. ",
    "Backpropagation applies the chain rule of calculus to distribute blame "
    "for the observed prediction error across every layer of the model, from "
    "the output units all the way back to the earliest feature detectors. ",
    "Regularization techniques such as weight decay and dropout discourage "
    "the network from memorizing individual training examples and push it "
    "toward simpler functions that generalize to data it has never seen. ",
    "Convolutional architectures exploit the spatial structure of images by "
    "sharing the same small filters across every location, which drastically "
    "reduces the number of parameters the optimizer needs to fit. ",
    "Attention mechanisms let a model weigh the relevance of every element "
    "in its input sequence when computing each output, which proved to be a "
    "remarkably effective inductive bias for language understanding tasks. ",
    "Stochastic optimization with small batches introduces noise into the "
    "gradient estimates, and that noise often helps the model escape sharp "
    "minima of the loss surface in favor of flatter and more robust ones. ",
]

PAGE_TEXT = "".join(_SENTENCES)


@pytest.fixture(scope="session")
def synthetic_pdf(tmp_path_factory):
    """A 6-page PDF with a real table of contents and two chapters."""
    import fitz

    path = tmp_path_factory.mktemp("pdfs") / "synthetic-book.pdf"
    doc = fitz.open()
    for i in range(6):
        page = doc.new_page()
        rect = fitz.Rect(50, 50, page.rect.width - 50, page.rect.height - 50)
        page.insert_textbox(rect, f"Page {i + 1}. {PAGE_TEXT}", fontsize=10)
    doc.set_toc([[1, "Chapter One", 1], [1, "Chapter Two", 4]])
    doc.save(str(path))
    doc.close()
    return str(path)


@pytest.fixture
def stub_chapter_llm(monkeypatch):
    """Fake only the LLM boundary of chapter detection: TOC extraction,
    chapter matching, page-range extraction and chunking all run for real."""
    from app.services.default_pdf_processor import DefaultPDFProcessor

    monkeypatch.setattr(
        DefaultPDFProcessor,
        "get_model_answer_of_chapters",
        lambda self, toc_prompt, model=None: ["Chapter One", "Chapter Two"],
    )


@pytest.fixture
def block_docling(monkeypatch):
    """Keep the docling fallback from importing (it downloads models)."""
    monkeypatch.setitem(sys.modules, "app.services.docling_pdf_processor", None)


@pytest.fixture
def fake_embedding():
    """Stub the embedding service at the wire; Qdrant HTTP passes through."""

    def _vec(text: str) -> list:
        vec = [0.0] * DENSE_DIM
        vec[int(hashlib.md5(text.encode()).hexdigest(), 16) % DENSE_DIM] = 1.0
        return vec

    def _embed(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["texts"]
        return httpx.Response(
            200,
            json={
                "dense_embeddings": [_vec(t) for t in texts],
                "sparse_embeddings": [{"1": 1.0} for _ in texts],
            },
        )

    passthrough_hosts = {settings.qdrant_host, "api.openai.com"}
    for base_url in (
        settings.anthropic_base_url,
        settings.deepseek_base_url,
        settings.local_llm_base_url,
    ):
        if base_url:
            passthrough_hosts.add(httpx.URL(base_url).host)

    with respx.mock(assert_all_called=False) as router:
        # Qdrant speaks HTTP, and llm-marked tests reach real providers.
        for host in passthrough_hosts:
            router.route(host=host).pass_through()
        router.post(f"{settings.embedding_service_url}/embed").mock(side_effect=_embed)
        yield router


class TaskStub:
    """Stands in for the bound celery task; records progress updates."""

    def __init__(self):
        self.states = []

    def update_state(self, state=None, meta=None):
        self.states.append((state, meta))


@pytest.fixture
def task_stub():
    return TaskStub()


@pytest.fixture
async def book_name():
    """Unique book name, cleaned from Postgres and Qdrant afterwards."""
    name = f"test-pdf-book-{uuid.uuid4().hex[:8]}"
    yield name

    conn = await asyncpg.connect(settings.database_url)
    try:
        await conn.execute("DELETE FROM books WHERE name = $1", name)
    finally:
        await conn.close()
    qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    qdrant.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[FieldCondition(key="book_name", match=MatchValue(value=name))]
        ),
    )
    qdrant.close()


async def book_counts(name: str) -> dict:
    """Postgres and Qdrant footprint of one book."""
    conn = await asyncpg.connect(settings.database_url)
    try:
        row = await conn.fetchrow(
            "SELECT id, processing_status, total_chunks FROM books WHERE name = $1",
            name,
        )
        if row is None:
            return {"exists": False}
        chapters = await conn.fetchval(
            "SELECT COUNT(*) FROM chapters WHERE book_id = $1", row["id"]
        )
        chunks = await conn.fetchval(
            "SELECT COUNT(*) FROM chunks WHERE book_id = $1", row["id"]
        )
    finally:
        await conn.close()

    qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    vectors = qdrant.count(
        collection_name=settings.qdrant_collection,
        count_filter=Filter(
            must=[FieldCondition(key="book_name", match=MatchValue(value=name))]
        ),
    ).count
    qdrant.close()

    return {
        "exists": True,
        "status": row["processing_status"],
        "total_chunks": row["total_chunks"],
        "chapters": chapters,
        "chunks": chunks,
        "vectors": vectors,
    }
