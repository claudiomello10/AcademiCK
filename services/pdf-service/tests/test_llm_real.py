"""Real-LLM chapter detection against the configured provider.

Local only (pytest -m llm). Uses the real embedding service too — nothing in
the pipeline is faked here.
"""

import pytest

from app.config import settings
from app.services.llm_client import build_chat_client
from tests.conftest import book_counts

pytestmark = pytest.mark.llm


def _require_chapter_model():
    try:
        build_chat_client(settings.pdf_chapter_detection_model)
    except Exception as e:
        pytest.skip(f"chapter model not configured: {e}")


async def test_real_chapter_detection_reads_toc(synthetic_pdf):
    _require_chapter_model()
    from app.services.default_pdf_processor import DefaultPDFProcessor

    summary = DefaultPDFProcessor().get_summary_list_from_PDF(
        synthetic_pdf, "llm-test-book"
    )
    assert summary is not None
    assert [c["Title"] for c in summary] == ["Chapter One", "Chapter Two"]


async def test_full_pipeline_with_real_chapter_detection(
    synthetic_pdf, book_name, task_stub, block_docling
):
    _require_chapter_model()
    from app.workers.tasks import _process_pdf_async

    result = await _process_pdf_async(task_stub, synthetic_pdf, book_name)
    assert result["success"] is True
    assert result["processing_method"] == "default"

    counts = await book_counts(book_name)
    assert counts["status"] == "completed"
    assert counts["chunks"] > 0
    assert counts["vectors"] == counts["chunks"]
