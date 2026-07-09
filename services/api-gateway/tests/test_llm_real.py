"""Real-LLM smoke test: full RAG pipeline with the configured models.

Local only (pytest -m llm). Embedding/intent stay stubbed — the point is to
exercise the resolver, curation agent, and answer generation against the
actual provider configured in .env.
"""

import pytest

from app.config import settings
from app.services.agent_models import build_model
from tests.conftest import auth

pytestmark = pytest.mark.llm


def _require_model(model: str, reasoning: str):
    try:
        build_model(model, reasoning)
    except Exception as e:
        pytest.skip(f"model '{model}' not configured: {e}")


async def test_real_llm_chat_pipeline(client, guest_token, seed_book, fake_ml):
    _require_model(settings.agent_curation_model, settings.agent_curation_reasoning)
    _require_model(settings.query_enhancement_model, settings.query_enhancement_reasoning)

    await seed_book()
    r = await client.post(
        "/api/v1/chat/single",
        json={"query": "Explain how neural networks learn from data."},
        headers=auth(guest_token),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["response"]
    assert body["intent"]
