"""PDF Service Configuration"""

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://academick:academick_secure_password@localhost:5432/academick"
    )

    # Redis
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

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
