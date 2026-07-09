"""Real-LLM smoke test: the full RAG pipeline with nothing faked.

Local only (pytest -m llm). Real embedding and intent services plus the
resolver, curation agent, and answer generation against the actual provider
configured in .env.
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


async def test_real_llm_chat_pipeline(client, guest_token, seed_book):
    _require_model(settings.agent_curation_model, settings.agent_curation_reasoning)
    _require_model(settings.query_enhancement_model, settings.query_enhancement_reasoning)

    book = await seed_book()
    r = await client.post(
        "/api/v1/chat/single",
        json={"query": "What is the Zorbite consolidation algorithm and what does it do?"},
        headers=auth(guest_token),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["response"]
    # Sources prove the real agent actually searched and approved the seeded
    # book rather than serving the no-context fallback.
    assert book["name"] in {s["book"] for s in body["sources"]}
