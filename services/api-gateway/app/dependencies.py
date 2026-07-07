"""FastAPI dependencies for dependency injection."""

from fastapi import Depends, Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

# auto_error=False so a missing header yields our own 401 instead of a 403.
bearer_scheme = HTTPBearer(auto_error=False)


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


async def get_current_session(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> dict:
    """
    Validate the session token from the Authorization: Bearer header.

    Returns the session dict with "session_id" added.
    Raises HTTPException if the header is missing or the session is invalid.
    """
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header"
        )

    session = await request.app.state.session_service.get_session(
        credentials.credentials
    )

    if not session:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired session"
        )

    session["session_id"] = credentials.credentials
    return session


async def get_admin_session(session: dict = Depends(get_current_session)) -> dict:
    """
    Dependency to validate admin session.

    Raises HTTPException if the session is valid but not admin.
    """
    if session.get("role") != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )

    return session
