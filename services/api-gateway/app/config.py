"""API Gateway Configuration"""

import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://academick:academick_secure_password@localhost:5432/academick"
    )

    # Redis
    redis_url: str = os.getenv("REDIS_URL", "redis://:change_this_redis_password@localhost:6379/0")

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
    session_secret: str = os.getenv("SESSION_SECRET", "change_this_in_production")
    session_ttl_minutes: int = int(os.getenv("SESSION_TTL_MINUTES", "30"))

    # Config users for testing
    config_users_enabled: bool = os.getenv("CONFIG_USERS_ENABLED", "true").lower() == "true"
    admin_password: str = os.getenv("ADMIN_PASSWORD", "admin_secure_password")
    guest_password: str = os.getenv("GUEST_PASSWORD", "guest_password")

    # LLM API Keys
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    deepseek_api_key: Optional[str] = os.getenv("DEEPSEEK_API_KEY")

    # Default LLM model
    default_model: str = os.getenv("DEFAULT_MODEL", "gpt-5-mini")

    # Model for query enhancement (fast and cheap, runs on every query)
    query_enhancement_model: str = os.getenv("QUERY_ENHANCEMENT_MODEL", "gpt-5-nano")

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
