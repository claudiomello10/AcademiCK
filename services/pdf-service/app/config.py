"""PDF Service Configuration

Required environment variables (no defaults — the service will not start without them):
  DATABASE_URL — e.g. postgresql://user:pass@localhost:5432/academick
  REDIS_URL    — e.g. redis://:password@localhost:6379/0

When using docker compose, these are set automatically from .env.
When running standalone, export them in your shell before starting the service.
"""

import os
from pydantic_settings import BaseSettings


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

    # Embedding service
    embedding_service_url: str = os.getenv(
        "EMBEDDING_SERVICE_URL", "http://localhost:8002"
    )

    # OpenAI for TOC analysis
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Processing settings
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "3000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "1000"))
    min_chunk_length: int = int(os.getenv("MIN_CHUNK_LENGTH", "300"))

    # Upload limits
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "100"))

    # Upload directory - use /app/processed since /app/uploads is read-only (contains existing PDFs)
    upload_dir: str = os.getenv("UPLOAD_DIR", "/app/processed/uploads")
    processed_dir: str = os.getenv("PROCESSED_DIR", "/app/processed")

    class Config:
        env_file = ".env"


settings = Settings()
