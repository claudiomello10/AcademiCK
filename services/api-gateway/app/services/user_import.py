"""Bulk user import from uploaded files.

Parsers are registered per extension so new formats (xlsx, xml) only
need a new entry in PARSERS. Each row is inserted in its own transaction:
one bad row is reported, not the whole batch aborted.
"""

import csv
import io
import os

from fastapi import HTTPException

from app.models.schemas import (
    ImportRowError, UserCreate, UserImportResponse, VALID_ROLES,
)
from app.services import user_service

ROW_FIELDS = ("username", "password", "email", "role", "registration_number")


def parse_csv(content: bytes) -> list[dict]:
    """Rows from a CSV with a header line naming any of ROW_FIELDS."""
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File is not valid UTF-8")

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or "username" not in reader.fieldnames:
        raise HTTPException(
            status_code=400,
            detail="CSV must have a header line including a 'username' column",
        )

    rows = []
    for raw in reader:
        rows.append({
            field: (raw.get(field) or "").strip() or None
            for field in ROW_FIELDS
        })
    return rows


PARSERS = {".csv": parse_csv}


async def import_users(
    db_pool,
    filename: str,
    content: bytes,
    allowed_roles: tuple[str, ...],
    default_password: str | None = None,
) -> UserImportResponse:
    ext = os.path.splitext(filename or "")[1].lower()
    parser = PARSERS.get(ext)
    if parser is None:
        supported = ", ".join(sorted(PARSERS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {supported}",
        )

    rows = parser(content)
    if not rows:
        raise HTTPException(status_code=400, detail="File contains no data rows")

    created = 0
    errors: list[ImportRowError] = []

    for index, row in enumerate(rows, start=2):  # row 1 is the header
        username = row.get("username")
        try:
            role = row.get("role") or "user"
            if role not in VALID_ROLES:
                raise HTTPException(status_code=400, detail=f"Invalid role: {role}")
            if role not in allowed_roles:
                raise HTTPException(
                    status_code=403, detail=f"Not allowed to create role: {role}"
                )
            password = row.get("password") or default_password
            if not username or not password:
                raise HTTPException(
                    status_code=400, detail="Missing username or password"
                )

            user = UserCreate(
                username=username,
                password=password,
                email=row.get("email"),
                role=role,
                registration_number=row.get("registration_number"),
            )
            async with db_pool.acquire() as conn:
                async with conn.transaction():
                    await user_service.create_user(conn, user)
            created += 1
        except HTTPException as exc:
            errors.append(
                ImportRowError(row=index, username=username, error=exc.detail)
            )
        except ValueError as exc:
            errors.append(
                ImportRowError(row=index, username=username, error=str(exc))
            )

    return UserImportResponse(created=created, errors=errors)
