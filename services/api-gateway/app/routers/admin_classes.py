"""Admin class administration: full control over all classes."""

from uuid import UUID

from fastapi import APIRouter, Depends, Request, HTTPException
import logging

from app.config import settings
from app.dependencies import get_admin_session
from app.models.schemas import AssignStudentRequest, ClassCreateAdmin, ClassUpdate
from app.services import class_service

router = APIRouter()
logger = logging.getLogger(__name__)


async def list_all_classes(db_pool) -> dict:
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT {class_service.CLASS_COLUMNS}, u.username AS professor_username,
                   (SELECT COUNT(*) FROM class_members m WHERE m.class_id = c.id) AS member_count,
                   (SELECT COUNT(*) FROM class_books cb WHERE cb.class_id = c.id) AS book_count
            FROM classes c
            JOIN users u ON u.id = c.professor_id
            ORDER BY c.name
            """
        )
        return {
            "classes": [
                class_service.class_response(
                    r,
                    professor_username=r["professor_username"],
                    member_count=r["member_count"],
                    book_count=r["book_count"],
                )
                for r in rows
            ]
        }


async def create_class_for_professor(db_pool, body: ClassCreateAdmin) -> dict:
    async with db_pool.acquire() as conn:
        row = await class_service.create_class(
            conn,
            name=body.name,
            subject=body.subject,
            description=body.description,
            professor_id=body.professor_id,
        )
        return class_service.class_response(row)


async def assign_student(db_pool, class_id: str, body: AssignStudentRequest) -> dict:
    """Enroll a student named by id or registration number (admin/manager flow)."""
    if not settings.enrollment_admin_assign_enabled:
        raise HTTPException(status_code=403, detail="Admin assignment is disabled")

    async with db_pool.acquire() as conn:
        await class_service.get_class(conn, class_id)

        if body.user_id:
            student = await conn.fetchrow(
                "SELECT id, username, registration_number FROM users WHERE id = $1 AND role = 'user'",
                UUID(body.user_id),
            )
        elif body.registration_number:
            student = await conn.fetchrow(
                "SELECT id, username, registration_number FROM users WHERE registration_number = $1 AND role = 'user'",
                body.registration_number.strip(),
            )
        else:
            raise HTTPException(
                status_code=400, detail="Provide user_id or registration_number"
            )

        if not student:
            raise HTTPException(status_code=404, detail="Student not found")

        await class_service.enroll(conn, class_id, str(student["id"]), "admin")

    return {
        "id": str(student["id"]),
        "username": student["username"],
        "registration_number": student["registration_number"],
    }


async def list_professors(db_pool) -> dict:
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, username FROM users WHERE role = 'professor' AND status = 'active' ORDER BY username"
        )
        return {
            "professors": [
                {"id": str(r["id"]), "username": r["username"]} for r in rows
            ]
        }


@router.get("/classes")
async def admin_list_classes(request: Request, session: dict = Depends(get_admin_session)):
    return await list_all_classes(request.app.state.db_pool)


@router.post("/classes")
async def admin_create_class(
    request: Request, body: ClassCreateAdmin, session: dict = Depends(get_admin_session)
):
    return await create_class_for_professor(request.app.state.db_pool, body)


@router.put("/classes/{class_id}")
async def admin_update_class(
    request: Request,
    class_id: str,
    body: ClassUpdate,
    session: dict = Depends(get_admin_session),
):
    async with request.app.state.db_pool.acquire() as conn:
        await class_service.get_class(conn, class_id)
        row = await class_service.update_class(conn, class_id, body.model_dump())
        return class_service.class_response(row)


@router.delete("/classes/{class_id}")
async def admin_delete_class(
    request: Request, class_id: str, session: dict = Depends(get_admin_session)
):
    async with request.app.state.db_pool.acquire() as conn:
        await class_service.get_class(conn, class_id)
        await conn.execute("DELETE FROM classes WHERE id = $1", UUID(class_id))
    return {"message": "Class deleted"}


@router.post("/classes/{class_id}/students")
async def admin_assign_student(
    request: Request,
    class_id: str,
    body: AssignStudentRequest,
    session: dict = Depends(get_admin_session),
):
    return await assign_student(request.app.state.db_pool, class_id, body)


@router.get("/classes/{class_id}/students")
async def admin_list_students(
    request: Request, class_id: str, session: dict = Depends(get_admin_session)
):
    async with request.app.state.db_pool.acquire() as conn:
        await class_service.get_class(conn, class_id)
        return {"students": await class_service.list_students(conn, class_id)}


@router.delete("/classes/{class_id}/students/{user_id}")
async def admin_remove_student(
    request: Request,
    class_id: str,
    user_id: str,
    session: dict = Depends(get_admin_session),
):
    async with request.app.state.db_pool.acquire() as conn:
        await class_service.get_class(conn, class_id)
        await class_service.remove_student(conn, class_id, user_id)
    return {"message": "Student removed"}


@router.get("/professors")
async def admin_list_professors(
    request: Request, session: dict = Depends(get_admin_session)
):
    return await list_professors(request.app.state.db_pool)


@router.get("/classes/{class_id}/books")
async def admin_list_class_books(
    request: Request, class_id: str, session: dict = Depends(get_admin_session)
):
    async with request.app.state.db_pool.acquire() as conn:
        await class_service.get_class(conn, class_id)
        return {"books": await class_service.list_class_books(conn, class_id)}


@router.post("/classes/{class_id}/books/{book_id}/attach")
async def admin_attach_book(
    request: Request,
    class_id: str,
    book_id: str,
    session: dict = Depends(get_admin_session),
):
    """Attach any book to any class (how legacy/global books reach classes)."""
    async with request.app.state.db_pool.acquire() as conn:
        await class_service.get_class(conn, class_id)
        book = await class_service.attach_book(
            conn, class_id, book_id, session["user_id"]
        )
    await class_service.flush_search_cache(request.app.state.redis)
    return {"message": f"Attached '{book['name']}'", "book": book}


@router.delete("/classes/{class_id}/books/{book_id}/detach")
async def admin_detach_book(
    request: Request,
    class_id: str,
    book_id: str,
    session: dict = Depends(get_admin_session),
):
    async with request.app.state.db_pool.acquire() as conn:
        await class_service.get_class(conn, class_id)
        await class_service.detach_book(conn, class_id, book_id)
    await class_service.flush_search_cache(request.app.state.redis)
    return {"message": "Book detached"}
