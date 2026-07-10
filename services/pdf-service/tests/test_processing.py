"""PDF processing pipeline behavior.

Real Postgres/Qdrant and real bge-m3 embeddings via the embedding service —
only chapter detection (the LLM call) is faked. Fails if the embedding
service is down.
"""

from types import SimpleNamespace

import pytest

from app.workers.tasks import _process_pdf_async
from tests.conftest import book_counts


async def test_processing_stores_chapters_chunks_and_vectors(
    synthetic_pdf, book_name, task_stub, stub_chapter_llm, block_docling
):
    result = await _process_pdf_async(task_stub, synthetic_pdf, book_name)

    assert result["success"] is True
    assert result["processing_method"] == "default"
    assert result["chapters_processed"] == 2
    assert result["chunks_processed"] > 0

    counts = await book_counts(book_name)
    assert counts["status"] == "completed"
    assert counts["chapters"] == 2
    assert counts["chunks"] == result["chunks_processed"]
    assert counts["vectors"] == result["chunks_processed"]
    assert counts["total_chunks"] == result["chunks_processed"]


async def test_reprocessing_replaces_content_without_duplicates(
    synthetic_pdf, book_name, task_stub, stub_chapter_llm, block_docling
):
    first = await _process_pdf_async(task_stub, synthetic_pdf, book_name)
    first_counts = await book_counts(book_name)

    second = await _process_pdf_async(task_stub, synthetic_pdf, book_name)
    second_counts = await book_counts(book_name)

    assert second["success"] is True
    assert second["chunks_processed"] == first["chunks_processed"]
    assert second_counts == first_counts


async def test_processing_with_class_binds_book_to_class_and_owner(
    synthetic_pdf, book_name, task_stub, stub_chapter_llm, block_docling
):
    import uuid as uuid_lib

    import asyncpg

    from app.config import settings

    conn = await asyncpg.connect(settings.database_url)
    professor_id = str(uuid_lib.uuid4())
    class_id = str(uuid_lib.uuid4())
    username = f"test-prof-{uuid_lib.uuid4().hex[:8]}"
    try:
        await conn.execute(
            """
            INSERT INTO users (id, username, password_hash, role)
            VALUES ($1, $2, 'x', 'professor')
            """,
            professor_id, username,
        )
        await conn.execute(
            """
            INSERT INTO classes (id, name, subject, professor_id)
            VALUES ($1, 'pdf-test-class', 'Test', $2)
            """,
            class_id, professor_id,
        )

        result = await _process_pdf_async(
            task_stub, synthetic_pdf, book_name,
            class_id=class_id, owner_user_id=professor_id,
        )
        assert result["success"] is True

        row = await conn.fetchrow(
            """
            SELECT b.owner_user_id, cb.class_id, cb.added_by
            FROM books b
            JOIN class_books cb ON cb.book_id = b.id
            WHERE b.name = $1
            """,
            book_name,
        )
        assert row is not None
        assert str(row["owner_user_id"]) == professor_id
        assert str(row["class_id"]) == class_id
        assert str(row["added_by"]) == professor_id
    finally:
        # users cascade removes the class, class_books row and book ownership
        await conn.execute("DELETE FROM users WHERE id = $1", professor_id)
        await conn.close()


async def test_failure_of_all_methods_marks_book_failed(
    synthetic_pdf, book_name, task_stub, block_docling, monkeypatch
):
    from app.services.default_pdf_processor import DefaultPDFProcessor

    def _boom(self, toc_prompt, model=None):
        raise RuntimeError("chapter detection unavailable")

    monkeypatch.setattr(DefaultPDFProcessor, "get_model_answer_of_chapters", _boom)

    with pytest.raises(RuntimeError):
        await _process_pdf_async(task_stub, synthetic_pdf, book_name)

    counts = await book_counts(book_name)
    assert counts["status"] == "failed"
    assert counts["chunks"] == 0
    assert counts["chapters"] == 0
    assert counts["vectors"] == 0


@pytest.fixture
async def api_client(monkeypatch, tmp_path):
    import httpx

    import app.workers.tasks as tasks_module
    from app.config import settings as pdf_settings
    from app.main import app

    monkeypatch.setattr(pdf_settings, "upload_dir", str(tmp_path / "uploads"))
    monkeypatch.setattr(
        tasks_module.process_pdf_task,
        "delay",
        lambda **kwargs: SimpleNamespace(id="test-job-123"),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_upload_accepts_pdf_and_returns_job(api_client, synthetic_pdf):
    with open(synthetic_pdf, "rb") as f:
        r = await api_client.post(
            "/upload", files={"file": ("book.pdf", f.read(), "application/pdf")}
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job_id"] == "test-job-123"
    assert body["status"] == "pending"


async def test_upload_rejects_non_pdf_extension(api_client):
    r = await api_client.post(
        "/upload", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert r.status_code == 400


async def test_upload_rejects_fake_pdf_content(api_client):
    r = await api_client.post(
        "/upload",
        files={"file": ("fake.pdf", b"this is not a pdf at all", "application/pdf")},
    )
    assert r.status_code == 400


async def test_job_status_for_unknown_job_is_pending(api_client):
    r = await api_client.get("/job/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 200
    assert r.json()["status"] == "pending"
