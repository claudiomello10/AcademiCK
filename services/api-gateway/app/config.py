"""API Gateway Configuration

Required environment variables (no defaults — the service will not start without them):
  DATABASE_URL   — e.g. postgresql://user:pass@localhost:5432/academick
  REDIS_URL      — e.g. redis://:password@localhost:6379/0
  SESSION_SECRET — random string for session encryption
  ADMIN_PASSWORD — admin user password
  GUEST_PASSWORD — guest user password

When using docker compose, these are set automatically from .env.
When running standalone, export them in your shell before starting the service.
"""

import json
import os
from pydantic_settings import BaseSettings
from typing import List, Optional


def _require_env(name: str) -> str:
    """Get a required environment variable or raise an error."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Required environment variable '{name}' is not set. "
            f"Set it in .env or export it before starting the service."
        )
    return value


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = _require_env("DATABASE_URL")

    # Redis
    redis_url: str = _require_env("REDIS_URL")

    # Qdrant
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "academick_embeddings")

    # Service URLs
    embedding_service_url: str = os.getenv(
        "EMBEDDING_SERVICE_URL", "http://localhost:8002"
    )
    intent_service_url: str = os.getenv(
        "INTENT_SERVICE_URL", "http://localhost:8001"
    )

    # Session configuration
    session_secret: str = _require_env("SESSION_SECRET")
    session_ttl_minutes: int = int(os.getenv("SESSION_TTL_MINUTES", "30"))

    # Config users for testing
    config_users_enabled: bool = os.getenv("CONFIG_USERS_ENABLED", "true").lower() == "true"
    admin_password: str = _require_env("ADMIN_PASSWORD")
    guest_password: str = _require_env("GUEST_PASSWORD")

    # LLM API Keys
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    deepseek_api_key: Optional[str] = os.getenv("DEEPSEEK_API_KEY")

    # Frontend model selector, served at runtime via GET /api/v1/models.
    available_models_raw: str = _require_env("AVAILABLE_MODELS")
    default_model_frontend: str = _require_env("DEFAULT_MODEL_FRONTEND")

    # Model for query enhancement (fast and cheap, runs on every query)
    query_enhancement_model: str = os.getenv("QUERY_ENHANCEMENT_MODEL", "gpt-5-nano")

    # Reasoning effort per pipeline stage ("none", "low", "medium", "high")
    query_enhancement_reasoning: str = os.getenv("QUERY_ENHANCEMENT_REASONING", "none")
    rag_reasoning: str = os.getenv("RAG_REASONING", "none")

    # Maximum completion tokens for LLM responses (includes reasoning + output)
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "16384"))

    # Top-K retrieval results per intent
    top_k_searching: int = int(os.getenv("TOP_K_SEARCHING", "10"))
    top_k_default: int = int(os.getenv("TOP_K_DEFAULT", "6"))

    # Search weights per intent (dense vs sparse)
    search_weight_qa_dense: float = float(os.getenv("SEARCH_WEIGHT_QA_DENSE", "0.6"))
    search_weight_qa_sparse: float = float(os.getenv("SEARCH_WEIGHT_QA_SPARSE", "0.4"))
    search_weight_summarization_dense: float = float(os.getenv("SEARCH_WEIGHT_SUMMARIZATION_DENSE", "0.7"))
    search_weight_summarization_sparse: float = float(os.getenv("SEARCH_WEIGHT_SUMMARIZATION_SPARSE", "0.3"))
    search_weight_coding_dense: float = float(os.getenv("SEARCH_WEIGHT_CODING_DENSE", "0.4"))
    search_weight_coding_sparse: float = float(os.getenv("SEARCH_WEIGHT_CODING_SPARSE", "0.6"))
    search_weight_searching_dense: float = float(os.getenv("SEARCH_WEIGHT_SEARCHING_DENSE", "0.5"))
    search_weight_searching_sparse: float = float(os.getenv("SEARCH_WEIGHT_SEARCHING_SPARSE", "0.5"))

    # Agentic RAG (context curation)
    agent_enabled: bool = os.getenv("AGENT_ENABLED", "true").lower() == "true"
    agent_max_iterations: int = int(os.getenv("AGENT_MAX_ITERATIONS", "3"))
    agent_curation_model: str = os.getenv("AGENT_CURATION_MODEL", "gpt-5-nano")
    agent_curation_reasoning: str = os.getenv("AGENT_CURATION_REASONING", "low")
    agent_max_context_chunks: int = int(os.getenv("AGENT_MAX_CONTEXT_CHUNKS", "18"))
    reasoning_trace_visible: bool = os.getenv("REASONING_TRACE_VISIBLE", "false").lower() == "true"

    # Default subject for new sessions
    default_subject: str = os.getenv("DEFAULT_SUBJECT", "Machine Learning")

    # Admin feature toggles
    enable_snapshot_management: bool = os.getenv("ENABLE_SNAPSHOT_MANAGEMENT", "true").lower() == "true"
    enable_pdf_upload: bool = os.getenv("ENABLE_PDF_UPLOAD", "true").lower() == "true"

    # API documentation toggle
    docs_enabled: bool = os.getenv("DOCS_ENABLED", "true").lower() == "true"

    # Snapshot storage directory (shared volume with Qdrant)
    snapshot_dir: str = os.getenv("SNAPSHOT_DIR", "/app/snapshots")

    class Config:
        env_file = ".env"


settings = Settings()


def _parse_frontend_models(raw: str, default: str) -> List[dict]:
    """Parse and validate AVAILABLE_MODELS at startup (fail-fast).

    Returns the list of {provider, value, label} dicts. Raises RuntimeError
    with a clear message if the value is malformed or DEFAULT_MODEL_FRONTEND
    is not among the listed models.
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"AVAILABLE_MODELS is not valid JSON: {exc}. "
            f"Expected a JSON array of {{provider, value, label}} objects."
        ) from exc

    if not isinstance(parsed, list) or not parsed:
        raise RuntimeError("AVAILABLE_MODELS must be a non-empty JSON array.")

    for item in parsed:
        if not isinstance(item, dict) or not {"provider", "value", "label"} <= item.keys():
            raise RuntimeError(
                "Each AVAILABLE_MODELS entry must be an object with "
                "'provider', 'value' and 'label' keys."
            )

    if not any(item["value"] == default for item in parsed):
        raise RuntimeError(
            f'DEFAULT_MODEL_FRONTEND="{default}" is not present in AVAILABLE_MODELS.'
        )

    return parsed


# Parsed once at import time so a misconfigured value crashes the service at
# boot rather than on the first request.
AVAILABLE_MODELS: List[dict] = _parse_frontend_models(
    settings.available_models_raw, settings.default_model_frontend
)
