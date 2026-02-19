"""
AcademiCK API Gateway

Main entry point for the API Gateway service.
Handles authentication, routing, and RAG orchestration.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncpg
from redis import asyncio as aioredis
import logging

from app.config import settings
from app.routers import auth, chat, books, admin, health
from app.clients.qdrant_client import QdrantManager
from app.clients.intent_client import IntentClient
from app.clients.embedding_client import EmbeddingClient
from app.services.session_service import SessionService
from app.utils.security import initialize_config_users

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HealthCheckFilter(logging.Filter):
    """Filter out health check endpoint logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        # Filter out health check related logs
        if "/health" in message:
            return False
        if "/exists" in message:
            return False
        return True


# Apply filter to uvicorn access logger and httpx
logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())
logging.getLogger("httpx").addFilter(HealthCheckFilter())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting API Gateway...")

    # Initialize PostgreSQL connection pool
    logger.info("Connecting to PostgreSQL...")
    app.state.db_pool = await asyncpg.create_pool(
        settings.database_url,
        min_size=5,
        max_size=10
    )
    logger.info("PostgreSQL connected")

    # Initialize Redis
    logger.info("Connecting to Redis...")
    app.state.redis = await aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True
    )
    logger.info("Redis connected")

    # Initialize session service with PostgreSQL persistence
    app.state.session_service = SessionService(
        redis=app.state.redis,
        db_pool=app.state.db_pool,
        session_ttl_minutes=settings.session_ttl_minutes
    )

    # Initialize Qdrant client
    logger.info("Connecting to Qdrant...")
    app.state.qdrant = QdrantManager(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection=settings.qdrant_collection
    )
    app.state.qdrant.ensure_collection()
    logger.info("Qdrant connected")

    # Initialize service clients
    app.state.intent_client = IntentClient(settings.intent_service_url)
    app.state.embedding_client = EmbeddingClient(settings.embedding_service_url)

    # Initialize config users if enabled
    if settings.config_users_enabled:
        await initialize_config_users(app.state.db_pool)
        logger.info("Config users initialized")

    logger.info("API Gateway started successfully")

    yield

    # Shutdown
    logger.info("Shutting down API Gateway...")
    await app.state.db_pool.close()
    await app.state.redis.close()
    await app.state.intent_client.close()
    await app.state.embedding_client.close()
    logger.info("API Gateway shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="AcademiCK API Gateway",
    description="RAG-powered educational assistant API",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.docs_enabled else None,
    redoc_url="/redoc" if settings.docs_enabled else None,
    openapi_url="/openapi.json" if settings.docs_enabled else None,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(auth.router, prefix="/api/v1", tags=["Authentication"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(books.router, prefix="/api/v1", tags=["Books"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["Admin"])


@app.get("/")
async def root():
    """Root endpoint."""
    response = {
        "service": "AcademiCK API Gateway",
        "version": "2.0.0",
    }
    if settings.docs_enabled:
        response["docs"] = "/docs"
    return response
