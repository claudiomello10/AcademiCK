"""Authentication endpoints."""

from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets

from app.models.schemas import (
    LoginRequest, LoginResponse, SessionResponse,
    SetSubjectRequest, SubjectResponse
)
from app.config import settings
from app.utils.security import authenticate_user
from app.routers.chat import track_usage

router = APIRouter()
security = HTTPBasic()


@router.post("/login", response_model=LoginResponse)
async def login(request: Request, credentials: LoginRequest):
    """
    Authenticate user and create a session.

    Returns session_id for subsequent requests.
    """
    # Authenticate user
    user = await authenticate_user(
        request.app.state.db_pool,
        credentials.username,
        credentials.password
    )

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )

    # Create session
    user_id = user.get("id", user["username"])
    session_id = await request.app.state.session_service.create_session(
        user_id=user_id,
        username=user["username"],
        role=user["role"]
    )

    await track_usage(
        db_pool=request.app.state.db_pool,
        user_id=user_id,
        session_id=session_id,
        action_type="login",
        response_time_ms=0,
        model_used="",
        intent=""
    )

    return LoginResponse(
        session_id=session_id,
        username=user["username"],
        role=user["role"]
    )


@router.get("/validate-session/{session_id}", response_model=SessionResponse)
async def validate_session(request: Request, session_id: str):
    """Validate a session and return session info."""
    result = await request.app.state.session_service.validate_session(session_id)
    return SessionResponse(**result)


@router.post("/logout/{session_id}")
async def logout(request: Request, session_id: str):
    """End a user session."""
    session = await request.app.state.session_service.get_session(session_id)

    deleted = await request.app.state.session_service.delete_session(session_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")

    if session:
        await track_usage(
            db_pool=request.app.state.db_pool,
            user_id=session.get("user_id", ""),
            session_id=session_id,
            action_type="logout",
            response_time_ms=0,
            model_used="",
            intent=""
        )

    return {"message": "Logged out successfully"}


@router.post("/session/{session_id}/subject", response_model=SubjectResponse)
async def set_subject(
    request: Request,
    session_id: str,
    subject_request: SetSubjectRequest
):
    """Set the study subject for a session."""
    session = await request.app.state.session_service.get_session(session_id)

    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")

    await request.app.state.session_service.set_subject(
        session_id,
        subject_request.subject
    )

    return SubjectResponse(subject=subject_request.subject)


@router.get("/session/{session_id}/subject", response_model=SubjectResponse)
async def get_subject(request: Request, session_id: str):
    """Get the current study subject for a session."""
    session = await request.app.state.session_service.get_session(session_id)

    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")

    return SubjectResponse(subject=session.get("subject", settings.default_subject))


# Admin authentication endpoints
@router.post("/admin/login", response_model=LoginResponse)
async def admin_login(request: Request, credentials: LoginRequest):
    """
    Authenticate admin user and create a session.
    Only users with 'admin' role can login.
    """
    # Authenticate user
    user = await authenticate_user(
        request.app.state.db_pool,
        credentials.username,
        credentials.password
    )

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )

    # Check if user is admin
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=403,
            detail="Access denied. Admin privileges required."
        )

    # Create session
    user_id = user.get("id", user["username"])
    session_id = await request.app.state.session_service.create_session(
        user_id=user_id,
        username=user["username"],
        role=user["role"]
    )

    await track_usage(
        db_pool=request.app.state.db_pool,
        user_id=user_id,
        session_id=session_id,
        action_type="login",
        response_time_ms=0,
        model_used="",
        intent=""
    )

    return LoginResponse(
        session_id=session_id,
        username=user["username"],
        role=user["role"]
    )


@router.get("/admin/validate-session/{session_id}")
async def validate_admin_session(request: Request, session_id: str):
    """Validate an admin session."""
    session = await request.app.state.session_service.get_session(session_id)

    if not session:
        return {"valid": False, "message": "Session not found"}

    if session.get("role") != "admin":
        return {"valid": False, "message": "Not an admin session"}

    return {
        "valid": True,
        "username": session.get("username"),
        "role": session.get("role")
    }
