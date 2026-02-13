"""Health check endpoints."""

from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Dict

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    services: Dict[str, bool]


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Check health of all services."""
    services = {
        "database": False,
        "redis": False,
        "qdrant": False,
        "intent_service": False,
        "embedding_service": False
    }

    # Check PostgreSQL
    try:
        async with request.app.state.db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
            services["database"] = True
    except Exception:
        pass

    # Check Redis
    try:
        await request.app.state.redis.ping()
        services["redis"] = True
    except Exception:
        pass

    # Check Qdrant
    try:
        services["qdrant"] = request.app.state.qdrant.health_check()
    except Exception:
        pass

    # Check Intent Service
    try:
        services["intent_service"] = await request.app.state.intent_client.health_check()
    except Exception:
        pass

    # Check Embedding Service
    try:
        services["embedding_service"] = await request.app.state.embedding_client.health_check()
    except Exception:
        pass

    # Overall status
    all_healthy = all(services.values())
    status = "healthy" if all_healthy else "degraded"

    return HealthResponse(status=status, services=services)


@router.get("/ready")
async def readiness_check(request: Request):
    """Readiness probe for Kubernetes."""
    try:
        async with request.app.state.db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return {"ready": True}
    except Exception:
        return {"ready": False}
