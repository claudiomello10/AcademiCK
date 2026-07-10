"""Class domain logic shared by the student, professor, manager and admin routers."""

import secrets
import string
from uuid import UUID

import asyncpg
from fastapi import HTTPException

JOIN_CODE_ALPHABET = string.ascii_uppercase + string.digits
JOIN_CODE_LENGTH = 8

BARE_CLASS_COLUMNS = (
    "id, name, subject, description, professor_id, join_code, "
    "join_code_enabled, is_active, created_at, updated_at"
)
CLASS_COLUMNS = ", ".join(f"c.{col}" for col in BARE_CLASS_COLUMNS.split(", "))


def generate_join_code() -> str:
    return "".join(secrets.choice(JOIN_CODE_ALPHABET) for _ in range(JOIN_CODE_LENGTH))


def class_response(row: asyncpg.Record, **extra) -> dict:
    return {
        "id": str(row["id"]),
        "name": row["name"],
        "subject": row["subject"],
        "description": row["description"],
        "professor_id": str(row["professor_id"]),
        "join_code": row["join_code"],
        "join_code_enabled": row["join_code_enabled"],
        "is_active": row["is_active"],
        "created_at": row["created_at"].isoformat(),
        **extra,
    }


async def get_class(conn, class_id: str) -> asyncpg.Record:
    try:
        class_uuid = UUID(class_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=404, detail="Class not found")

    row = await conn.fetchrow(
        f"SELECT {CLASS_COLUMNS} FROM classes c WHERE c.id = $1", class_uuid
    )
    if not row:
        raise HTTPException(status_code=404, detail="Class not found")
    return row


async def get_owned_class(conn, class_id: str, session: dict) -> asyncpg.Record:
    """Load a class, requiring the session user to own it (admins own all)."""
    row = await get_class(conn, class_id)
    if session.get("role") != "admin" and str(row["professor_id"]) != session.get("user_id"):
        raise HTTPException(status_code=403, detail="Not your class")
    return row


async def is_member(conn, class_id: str, user_id: str) -> bool:
    return await conn.fetchval(
        "SELECT EXISTS(SELECT 1 FROM class_members WHERE class_id = $1 AND user_id = $2)",
        UUID(class_id), UUID(user_id),
    )


async def enroll(conn, class_id: str, user_id: str, via: str) -> None:
    """Insert a class membership; 409 if already enrolled."""
    try:
        await conn.execute(
            """
            INSERT INTO class_members (class_id, user_id, enrolled_via)
            VALUES ($1, $2, $3)
            """,
            UUID(class_id), UUID(user_id), via,
        )
    except asyncpg.UniqueViolationError:
        raise HTTPException(status_code=409, detail="Student is already enrolled in this class")


async def get_user_classes(conn, user_id: str) -> list[dict]:
    """Classes the user is enrolled in, plus owned classes for professors."""
    rows = await conn.fetch(
        f"""
        SELECT DISTINCT {CLASS_COLUMNS}, u.username AS professor_username
        FROM classes c
        JOIN users u ON u.id = c.professor_id
        LEFT JOIN class_members m ON m.class_id = c.id
        WHERE c.is_active = true AND (m.user_id = $1 OR c.professor_id = $1)
        ORDER BY c.name
        """,
        UUID(user_id),
    )
    return [
        class_response(r, professor_username=r["professor_username"]) for r in rows
    ]


async def create_class(
    conn,
    *,
    name: str,
    subject: str,
    description: str | None,
    professor_id: str,
) -> asyncpg.Record:
    """Insert a class owned by `professor_id` with a fresh join code."""
    professor_role = await conn.fetchval(
        "SELECT role FROM users WHERE id = $1", UUID(professor_id)
    )
    if professor_role not in ("professor", "admin"):
        raise HTTPException(status_code=400, detail="Class owner must be a professor")

    while True:
        try:
            return await conn.fetchrow(
                f"""
                INSERT INTO classes (name, subject, description, professor_id, join_code)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING {BARE_CLASS_COLUMNS}
                """,
                name, subject, description, UUID(professor_id), generate_join_code(),
            )
        except asyncpg.UniqueViolationError as exc:
            if "join_code" not in str(exc.constraint_name or ""):
                raise

async def update_class(conn, class_id: str, updates: dict) -> asyncpg.Record:
    fields = []
    values = []
    param = 1
    for column in ("name", "subject", "description", "is_active", "join_code_enabled"):
        if column in updates and updates[column] is not None:
            fields.append(f"{column} = ${param}")
            values.append(updates[column])
            param += 1

    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")

    values.append(UUID(class_id))
    row = await conn.fetchrow(
        f"""
        UPDATE classes SET {', '.join(fields)}
        WHERE id = ${param}
        RETURNING {BARE_CLASS_COLUMNS}
        """,
        *values,
    )
    return row


async def regenerate_join_code(conn, class_id: str) -> str:
    while True:
        try:
            return await conn.fetchval(
                "UPDATE classes SET join_code = $1 WHERE id = $2 RETURNING join_code",
                generate_join_code(), UUID(class_id),
            )
        except asyncpg.UniqueViolationError:
            continue


async def list_students(conn, class_id: str) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT u.id, u.username, u.registration_number, m.enrolled_via, m.enrolled_at
        FROM class_members m
        JOIN users u ON u.id = m.user_id
        WHERE m.class_id = $1
        ORDER BY u.username
        """,
        UUID(class_id),
    )
    return [
        {
            "id": str(r["id"]),
            "username": r["username"],
            "registration_number": r["registration_number"],
            "enrolled_via": r["enrolled_via"],
            "enrolled_at": r["enrolled_at"].isoformat(),
        }
        for r in rows
    ]


async def remove_student(conn, class_id: str, user_id: str) -> None:
    deleted = await conn.execute(
        "DELETE FROM class_members WHERE class_id = $1 AND user_id = $2",
        UUID(class_id), UUID(user_id),
    )
    if deleted == "DELETE 0":
        raise HTTPException(status_code=404, detail="Student is not enrolled in this class")


async def list_class_books(conn, class_id: str) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT b.id, b.name, b.processing_status, b.total_chunks, b.owner_user_id,
               cb.created_at AS attached_at
        FROM class_books cb
        JOIN books b ON b.id = cb.book_id
        WHERE cb.class_id = $1
        ORDER BY b.name
        """,
        UUID(class_id),
    )
    return [
        {
            "id": str(r["id"]),
            "name": r["name"],
            "processing_status": r["processing_status"],
            "total_chunks": r["total_chunks"],
            "owner_user_id": str(r["owner_user_id"]) if r["owner_user_id"] else None,
            "attached_at": r["attached_at"].isoformat(),
        }
        for r in rows
    ]


async def attach_book(conn, class_id: str, book_id: str, added_by: str) -> dict:
    """Attach a book to a class (idempotent). 404 for unknown books."""
    book = await conn.fetchrow(
        "SELECT id, name FROM books WHERE id = $1", UUID(book_id)
    )
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    await conn.execute(
        """
        INSERT INTO class_books (class_id, book_id, added_by)
        VALUES ($1, $2, $3)
        ON CONFLICT DO NOTHING
        """,
        UUID(class_id), UUID(book_id), UUID(added_by),
    )
    return {"id": str(book["id"]), "name": book["name"]}


async def detach_book(conn, class_id: str, book_id: str) -> None:
    deleted = await conn.execute(
        "DELETE FROM class_books WHERE class_id = $1 AND book_id = $2",
        UUID(class_id), UUID(book_id),
    )
    if deleted == "DELETE 0":
        raise HTTPException(status_code=404, detail="Book is not attached to this class")


async def get_allowed_book_names(db_pool, class_id: str) -> list[str]:
    """Names of the class's completed books — the retrieval allowlist."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT b.name
            FROM class_books cb
            JOIN books b ON b.id = cb.book_id
            WHERE cb.class_id = $1 AND b.processing_status = 'completed'
            ORDER BY b.name
            """,
            UUID(class_id),
        )
    return [r["name"] for r in rows]


async def flush_search_cache(redis) -> None:
    """Drop cached search results after the visible book set changes."""
    keys = await redis.keys("search:*")
    if keys:
        await redis.delete(*keys)
