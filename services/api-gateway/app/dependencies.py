"""FastAPI dependencies for dependency injection."""

from fastapi import Request, HTTPException
from typing import Optional


async def get_db_pool(request: Request):
    """Get database connection pool."""
    return request.app.state.db_pool


async def get_redis(request: Request):
    """Get Redis client."""
    return request.app.state.redis


async def get_session_service(request: Request):
    """Get session service."""
    return request.app.state.session_service


async def get_qdrant(request: Request):
    """Get Qdrant client."""
    return request.app.state.qdrant


async def get_intent_client(request: Request):
    """Get intent classification client."""
    return request.app.state.intent_client


async def get_embedding_client(request: Request):
    """Get embedding service client."""
    return request.app.state.embedding_client


async def get_valid_session(request: Request, session_id: str):
    """
    Dependency to validate session and return session data.

    Raises HTTPException if session is invalid.
    """
    session = await request.app.state.session_service.get_session(session_id)

    if not session:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired session"
        )

    return session


async def get_admin_session(request: Request, session_id: str):
    """
    Dependency to validate admin session.

    Raises HTTPException if session is invalid or not admin.
    """
    session = await get_valid_session(request, session_id)

    if session.get("role") != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )

    return session
