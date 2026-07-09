"""PDF processing pipeline behavior: real Postgres/Qdrant, faked LLM/embedder."""

from types import SimpleNamespace

import pytest

from app.workers.tasks import _process_pdf_async
from tests.conftest import book_counts


async def test_processing_stores_chapters_chunks_and_vectors(
    synthetic_pdf, book_name, task_stub, stub_chapter_llm, block_docling, fake_embedding
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
    synthetic_pdf, book_name, task_stub, stub_chapter_llm, block_docling, fake_embedding
):
    first = await _process_pdf_async(task_stub, synthetic_pdf, book_name)
    first_counts = await book_counts(book_name)

    second = await _process_pdf_async(task_stub, synthetic_pdf, book_name)
    second_counts = await book_counts(book_name)

    assert second["success"] is True
    assert second["chunks_processed"] == first["chunks_processed"]
    assert second_counts == first_counts


async def test_failure_of_all_methods_marks_book_failed(
    synthetic_pdf, book_name, task_stub, block_docling, fake_embedding, monkeypatch
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
