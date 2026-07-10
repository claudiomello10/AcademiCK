"""Professor endpoints: own-class management.

Every class-scoped endpoint loads the class through get_owned_class,
which 403s unless the session user owns it (admins own all).
"""

from uuid import UUID

from fastapi import APIRouter, Depends, Request, HTTPException
import logging

from app.config import settings
from app.dependencies import get_professor_session
from app.models.schemas import (
    AddStudentRequest, ClassCreate, ClassUpdate, JoinCodeToggleRequest,
)
from app.services import class_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/classes")
async def list_classes(request: Request, session: dict = Depends(get_professor_session)):
    """Classes owned by the professor (all classes for admins)."""
    async with request.app.state.db_pool.acquire() as conn:
        if session.get("role") == "admin":
            rows = await conn.fetch(
                f"SELECT {class_service.CLASS_COLUMNS} FROM classes c ORDER BY c.name"
            )
        else:
            rows = await conn.fetch(
                f"""
                SELECT {class_service.CLASS_COLUMNS} FROM classes c
                WHERE c.professor_id = $1 ORDER BY c.name
                """,
                UUID(session["user_id"]),
            )
        return {"classes": [class_service.class_response(r) for r in rows]}


@router.post("/classes")
async def create_class(
    request: Request, body: ClassCreate, session: dict = Depends(get_professor_session)
):
    """Create a class owned by the logged-in professor."""
    async with request.app.state.db_pool.acquire() as conn:
        row = await class_service.create_class(
            conn,
            name=body.name,
            subject=body.subject,
            description=body.description,
            professor_id=session["user_id"],
        )
        return class_service.class_response(row)


@router.put("/classes/{class_id}")
async def update_class(
    request: Request,
    class_id: str,
    body: ClassUpdate,
    session: dict = Depends(get_professor_session),
):
    async with request.app.state.db_pool.acquire() as conn:
        await class_service.get_owned_class(conn, class_id, session)
        row = await class_service.update_class(conn, class_id, body.model_dump())
        return class_service.class_response(row)


@router.delete("/classes/{class_id}")
async def delete_class(
    request: Request, class_id: str, session: dict = Depends(get_professor_session)
):
    async with request.app.state.db_pool.acquire() as conn:
        await class_service.get_owned_class(conn, class_id, session)
        await conn.execute("DELETE FROM classes WHERE id = $1", UUID(class_id))
    return {"message": "Class deleted"}


@router.post("/classes/{class_id}/join-code/regenerate")
async def regenerate_join_code(
    request: Request, class_id: str, session: dict = Depends(get_professor_session)
):
    async with request.app.state.db_pool.acquire() as conn:
        await class_service.get_owned_class(conn, class_id, session)
        join_code = await class_service.regenerate_join_code(conn, class_id)
    return {"join_code": join_code}


@router.put("/classes/{class_id}/join-code")
async def toggle_join_code(
    request: Request,
    class_id: str,
    body: JoinCodeToggleRequest,
    session: dict = Depends(get_professor_session),
):
    async with request.app.state.db_pool.acquire() as conn:
        await class_service.get_owned_class(conn, class_id, session)
        await conn.execute(
            "UPDATE classes SET join_code_enabled = $1 WHERE id = $2",
            body.enabled, UUID(class_id),
        )
    return {"join_code_enabled": body.enabled}


@router.get("/classes/{class_id}/students")
async def list_students(
    request: Request, class_id: str, session: dict = Depends(get_professor_session)
):
    async with request.app.state.db_pool.acquire() as conn:
        await class_service.get_owned_class(conn, class_id, session)
        return {"students": await class_service.list_students(conn, class_id)}


@router.post("/classes/{class_id}/students")
async def add_student(
    request: Request,
    class_id: str,
    body: AddStudentRequest,
    session: dict = Depends(get_professor_session),
):
    """Enroll a student by registration number."""
    if not settings.enrollment_by_registration_enabled:
        raise HTTPException(
            status_code=403, detail="Enrollment by registration number is disabled"
        )

    async with request.app.state.db_pool.acquire() as conn:
        await class_service.get_owned_class(conn, class_id, session)
        student = await conn.fetchrow(
            """
            SELECT id, username, registration_number FROM users
            WHERE registration_number = $1 AND role = 'user' AND status = 'active'
            """,
            body.registration_number.strip(),
        )
        if not student:
            raise HTTPException(
                status_code=404,
                detail="No student with this registration number",
            )
        await class_service.enroll(conn, class_id, str(student["id"]), "professor")

    return {
        "id": str(student["id"]),
        "username": student["username"],
        "registration_number": student["registration_number"],
    }


@router.delete("/classes/{class_id}/students/{user_id}")
async def remove_student(
    request: Request,
    class_id: str,
    user_id: str,
    session: dict = Depends(get_professor_session),
):
    async with request.app.state.db_pool.acquire() as conn:
        await class_service.get_owned_class(conn, class_id, session)
        await class_service.remove_student(conn, class_id, user_id)
    return {"message": "Student removed"}


@router.get("/features")
async def professor_features(session: dict = Depends(get_professor_session)):
    """Feature flags the professor UI needs."""
    return {
        "summary_enabled": settings.analytics_summary_enabled,
        "registration_enrollment_enabled": settings.enrollment_by_registration_enabled,
        "join_code_enabled": settings.enrollment_join_code_enabled,
        "book_upload_enabled": settings.professor_book_upload_enabled,
    }
