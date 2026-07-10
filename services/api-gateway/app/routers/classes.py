"""Student-facing class endpoints: enrollment and active-class selection."""

from fastapi import APIRouter, Depends, Request, HTTPException
import logging

from app.config import settings
from app.dependencies import get_current_session
from app.models.schemas import JoinClassRequest, SetActiveClassRequest
from app.services import class_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/classes/mine")
async def my_classes(request: Request, session: dict = Depends(get_current_session)):
    """Classes the user is enrolled in (or owns, for professors)."""
    async with request.app.state.db_pool.acquire() as conn:
        return {"classes": await class_service.get_user_classes(conn, session["user_id"])}


@router.post("/classes/join")
async def join_class(
    request: Request,
    body: JoinClassRequest,
    session: dict = Depends(get_current_session),
):
    """Enroll in a class with its join code."""
    if not settings.enrollment_join_code_enabled:
        raise HTTPException(status_code=403, detail="Join codes are disabled")

    async with request.app.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""
            SELECT {class_service.CLASS_COLUMNS} FROM classes c
            WHERE c.join_code = $1 AND c.join_code_enabled = true AND c.is_active = true
            """,
            body.join_code.strip().upper(),
        )
        if not row:
            raise HTTPException(status_code=404, detail="Invalid join code")

        await class_service.enroll(conn, str(row["id"]), session["user_id"], "join_code")

        return {
            "class_id": str(row["id"]),
            "name": row["name"],
            "subject": row["subject"],
        }


@router.post("/session/class")
async def set_active_class(
    request: Request,
    body: SetActiveClassRequest,
    session: dict = Depends(get_current_session),
):
    """Scope the session to a class. Members always; owners and admins too."""
    async with request.app.state.db_pool.acquire() as conn:
        cls = await class_service.get_class(conn, body.class_id)
        if not cls["is_active"]:
            raise HTTPException(status_code=404, detail="Class not found")

        allowed = (
            session.get("role") == "admin"
            or str(cls["professor_id"]) == session["user_id"]
            or await class_service.is_member(conn, body.class_id, session["user_id"])
        )
        if not allowed:
            raise HTTPException(status_code=403, detail="Not enrolled in this class")

    conversation_id = await request.app.state.session_service.set_active_class(
        session["session_id"], body.class_id, cls["subject"]
    )

    return {
        "class_id": str(cls["id"]),
        "name": cls["name"],
        "subject": cls["subject"],
        "conversation_id": conversation_id,
    }


@router.get("/session/class")
async def get_active_class(
    request: Request, session: dict = Depends(get_current_session)
):
    """The session's active class, if any."""
    class_id = session.get("active_class_id")
    if not class_id:
        return {"class_id": None}

    async with request.app.state.db_pool.acquire() as conn:
        cls = await class_service.get_class(conn, class_id)

    return {
        "class_id": str(cls["id"]),
        "name": cls["name"],
        "subject": cls["subject"],
    }


@router.get("/features")
async def student_features(session: dict = Depends(get_current_session)):
    """Feature flags the student UI needs."""
    return {"join_code_enabled": settings.enrollment_join_code_enabled}
