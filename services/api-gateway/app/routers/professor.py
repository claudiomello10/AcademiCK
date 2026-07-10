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
from app.services import class_service, pdf_upload

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


# ===========================================
# Books: select from catalog, upload, delete
# ===========================================

@router.get("/books/catalog")
async def book_catalog(
    request: Request,
    class_id: str = None,
    session: dict = Depends(get_professor_session),
):
    """Books the professor may attach: global books plus their own.

    With class_id, marks which are already attached to that class.
    """
    async with request.app.state.db_pool.acquire() as conn:
        attached = set()
        if class_id:
            await class_service.get_owned_class(conn, class_id, session)
            attached = {
                b["id"] for b in await class_service.list_class_books(conn, class_id)
            }

        if session.get("role") == "admin":
            rows = await conn.fetch("""
                SELECT id, name, total_chunks, owner_user_id FROM books
                WHERE processing_status = 'completed'
                ORDER BY name
            """)
        else:
            rows = await conn.fetch("""
                SELECT id, name, total_chunks, owner_user_id FROM books
                WHERE processing_status = 'completed'
                  AND (owner_user_id IS NULL OR owner_user_id = $1)
                ORDER BY name
            """, UUID(session["user_id"]))

        return {
            "books": [
                {
                    "id": str(r["id"]),
                    "name": r["name"],
                    "total_chunks": r["total_chunks"],
                    "owned": str(r["owner_user_id"]) == session["user_id"],
                    "attached": str(r["id"]) in attached,
                }
                for r in rows
            ]
        }


@router.get("/classes/{class_id}/books")
async def list_class_books(
    request: Request, class_id: str, session: dict = Depends(get_professor_session)
):
    async with request.app.state.db_pool.acquire() as conn:
        await class_service.get_owned_class(conn, class_id, session)
        return {"books": await class_service.list_class_books(conn, class_id)}


@router.post("/classes/{class_id}/books/{book_id}/attach")
async def attach_book(
    request: Request,
    class_id: str,
    book_id: str,
    session: dict = Depends(get_professor_session),
):
    """Attach a catalog book (global or own) to an owned class."""
    async with request.app.state.db_pool.acquire() as conn:
        await class_service.get_owned_class(conn, class_id, session)

        if session.get("role") != "admin":
            owner = await conn.fetchval(
                "SELECT owner_user_id FROM books WHERE id = $1", UUID(book_id)
            )
            if owner is not None and str(owner) != session["user_id"]:
                raise HTTPException(
                    status_code=403,
                    detail="This book belongs to another professor",
                )

        book = await class_service.attach_book(
            conn, class_id, book_id, session["user_id"]
        )
    await class_service.flush_search_cache(request.app.state.redis)
    return {"message": f"Attached '{book['name']}'", "book": book}


@router.delete("/classes/{class_id}/books/{book_id}/detach")
async def detach_book(
    request: Request,
    class_id: str,
    book_id: str,
    session: dict = Depends(get_professor_session),
):
    async with request.app.state.db_pool.acquire() as conn:
        await class_service.get_owned_class(conn, class_id, session)
        await class_service.detach_book(conn, class_id, book_id)
    await class_service.flush_search_cache(request.app.state.redis)
    return {"message": "Book detached"}


@router.post("/classes/{class_id}/books/upload")
async def upload_books(
    request: Request, class_id: str, session: dict = Depends(get_professor_session)
):
    """Upload PDFs for an owned class. Gated by PROFESSOR_BOOK_UPLOAD_ENABLED."""
    if not settings.professor_book_upload_enabled:
        raise HTTPException(status_code=403, detail="Professor book upload is disabled")

    async with request.app.state.db_pool.acquire() as conn:
        await class_service.get_owned_class(conn, class_id, session)

    form = await request.form()
    files = form.getlist("files")

    result = await pdf_upload.forward_pdfs(
        files,
        request.app.state.db_pool,
        uploaded_by=session["user_id"],
        class_id=class_id,
        owner_user_id=session["user_id"],
    )

    # Surface a lone name collision as a proper 409 instead of a 200 wrapper
    if not result["jobs"] and len(result["errors"]) == 1 \
            and result["errors"][0].get("status_code") == 409:
        raise HTTPException(status_code=409, detail=result["errors"][0]["error"])

    return result


@router.delete("/classes/{class_id}/books/{book_id}")
async def delete_book(
    request: Request,
    class_id: str,
    book_id: str,
    session: dict = Depends(get_professor_session),
):
    """Fully delete an owned book (content included). Admins may delete any."""
    async with request.app.state.db_pool.acquire() as conn:
        await class_service.get_owned_class(conn, class_id, session)
        book = await conn.fetchrow(
            "SELECT id, name, owner_user_id FROM books WHERE id = $1", UUID(book_id)
        )
        if not book:
            raise HTTPException(status_code=404, detail="Book not found")
        if session.get("role") != "admin" and (
            book["owner_user_id"] is None
            or str(book["owner_user_id"]) != session["user_id"]
        ):
            raise HTTPException(
                status_code=403,
                detail="Only the owning professor can delete this book",
            )

    deleted_count = await request.app.state.qdrant.delete_by_book_name(book["name"])
    async with request.app.state.db_pool.acquire() as conn:
        await conn.execute("DELETE FROM books WHERE id = $1", UUID(book_id))
    await class_service.flush_search_cache(request.app.state.redis)

    return {
        "success": True,
        "book_name": book["name"],
        "vectors_deleted": deleted_count,
    }


@router.get("/pdf-job/{job_id}")
async def pdf_job_status(
    request: Request, job_id: str, session: dict = Depends(get_professor_session)
):
    """Job status, restricted to the professor's own uploads."""
    if session.get("role") != "admin":
        async with request.app.state.db_pool.acquire() as conn:
            _, metadata = await pdf_upload.get_job(conn, job_id)
            if metadata.get("uploaded_by") != session["user_id"]:
                raise HTTPException(status_code=403, detail="Not your upload job")

    return await pdf_upload.sync_job_status(
        request.app.state.db_pool, request.app.state.qdrant, job_id
    )


@router.get("/features")
async def professor_features(session: dict = Depends(get_professor_session)):
    """Feature flags the professor UI needs."""
    return {
        "summary_enabled": settings.analytics_summary_enabled,
        "registration_enrollment_enabled": settings.enrollment_by_registration_enabled,
        "join_code_enabled": settings.enrollment_join_code_enabled,
        "book_upload_enabled": settings.professor_book_upload_enabled,
    }
