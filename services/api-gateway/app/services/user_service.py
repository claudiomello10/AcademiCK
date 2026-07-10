"""Shared user CRUD logic used by the admin and manager routers."""

import asyncpg
from fastapi import HTTPException

from app.models.schemas import UserCreate, UserUpdate, UserResponse, VALID_ROLES
from app.utils.security import hash_password

USER_COLUMNS = (
    "id, username, email, role, status, is_config_user, "
    "registration_number, created_at, last_active"
)


def user_response(row: asyncpg.Record) -> UserResponse:
    return UserResponse(
        id=str(row["id"]),
        username=row["username"],
        email=row["email"],
        role=row["role"],
        status=row["status"],
        is_config_user=row["is_config_user"],
        registration_number=row["registration_number"],
        created_at=row["created_at"],
        last_active=row["last_active"],
    )


async def list_users(conn) -> list[UserResponse]:
    rows = await conn.fetch(
        f"SELECT {USER_COLUMNS} FROM users ORDER BY created_at DESC"
    )
    return [user_response(r) for r in rows]


async def create_user(conn, user: UserCreate) -> UserResponse:
    """Insert a user. Students (role 'user') must carry a registration number."""
    if user.role not in VALID_ROLES:
        raise HTTPException(status_code=400, detail=f"Invalid role: {user.role}")

    if user.role == "user" and not user.registration_number:
        raise HTTPException(
            status_code=400,
            detail="Registration number is required for students",
        )

    exists = await conn.fetchval(
        "SELECT EXISTS(SELECT 1 FROM users WHERE username = $1)", user.username
    )
    if exists:
        raise HTTPException(status_code=400, detail="Username already exists")

    try:
        row = await conn.fetchrow(
            f"""
            INSERT INTO users (username, email, password_hash, role, registration_number)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING {USER_COLUMNS}
            """,
            user.username,
            user.email,
            hash_password(user.password),
            user.role,
            user.registration_number,
        )
    except asyncpg.UniqueViolationError as exc:
        if "registration" in str(exc.constraint_name or ""):
            raise HTTPException(
                status_code=409,
                detail=f"Registration number already exists: {user.registration_number}",
            )
        raise HTTPException(status_code=400, detail="Username already exists")

    return user_response(row)


async def get_user(conn, user_id: str) -> asyncpg.Record:
    row = await conn.fetchrow(
        f"SELECT {USER_COLUMNS} FROM users WHERE id = $1", user_id
    )
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    return row


async def update_user(conn, user_id: str, updates: UserUpdate) -> UserResponse:
    if updates.role is not None and updates.role not in VALID_ROLES:
        raise HTTPException(status_code=400, detail=f"Invalid role: {updates.role}")

    update_fields = []
    values = []
    param_count = 1

    for field in ("email", "role", "status", "registration_number"):
        value = getattr(updates, field)
        if value is not None:
            update_fields.append(f"{field} = ${param_count}")
            values.append(value)
            param_count += 1

    if not update_fields:
        raise HTTPException(status_code=400, detail="No fields to update")

    values.append(user_id)

    try:
        row = await conn.fetchrow(
            f"""
            UPDATE users
            SET {', '.join(update_fields)}, updated_at = CURRENT_TIMESTAMP
            WHERE id = ${param_count}
            RETURNING {USER_COLUMNS}
            """,
            *values,
        )
    except asyncpg.UniqueViolationError as exc:
        if "registration" in str(exc.constraint_name or ""):
            raise HTTPException(
                status_code=409,
                detail=f"Registration number already exists: {updates.registration_number}",
            )
        raise HTTPException(status_code=400, detail="Email already exists")

    if not row:
        raise HTTPException(status_code=404, detail="User not found")

    return user_response(row)
