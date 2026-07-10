"""Manager endpoints: user and class administration without system controls.

Managers handle students and professors only — admin and manager accounts
are invisible to and untouchable by this router.
"""

from fastapi import APIRouter, Depends, Request, HTTPException, UploadFile, File, Form
from typing import List, Optional
import logging

from app.dependencies import get_manager_session
from app.models.schemas import (
    UserCreate, UserUpdate, UserResponse, UserImportResponse,
)
from app.services import user_service, user_import

router = APIRouter()
logger = logging.getLogger(__name__)

MANAGED_ROLES = ("user", "professor")


def _require_managed_role(role: str):
    if role not in MANAGED_ROLES:
        raise HTTPException(
            status_code=403,
            detail=f"Managers can only manage roles: {', '.join(MANAGED_ROLES)}",
        )


async def _get_managed_user(conn, user_id: str):
    """Load the target user, refusing admin/manager accounts."""
    row = await user_service.get_user(conn, user_id)
    _require_managed_role(row["role"])
    return row


@router.get("/users", response_model=List[UserResponse])
async def list_users(request: Request, session: dict = Depends(get_manager_session)):
    """List students and professors."""
    async with request.app.state.db_pool.acquire() as conn:
        users = await user_service.list_users(conn)
        return [u for u in users if u.role in MANAGED_ROLES]


@router.post("/users", response_model=UserResponse)
async def create_user(
    request: Request, user: UserCreate, session: dict = Depends(get_manager_session)
):
    """Create a student or professor."""
    _require_managed_role(user.role)
    async with request.app.state.db_pool.acquire() as conn:
        return await user_service.create_user(conn, user)


@router.post("/users/import", response_model=UserImportResponse)
async def import_users(
    request: Request,
    file: UploadFile = File(...),
    default_password: Optional[str] = Form(None),
    session: dict = Depends(get_manager_session),
):
    """Bulk-import students/professors from a file. Reports errors per row."""
    content = await file.read()
    return await user_import.import_users(
        request.app.state.db_pool,
        file.filename,
        content,
        allowed_roles=MANAGED_ROLES,
        default_password=default_password,
    )


@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    request: Request,
    user_id: str,
    updates: UserUpdate,
    session: dict = Depends(get_manager_session),
):
    """Update a student or professor."""
    if updates.role is not None:
        _require_managed_role(updates.role)
    async with request.app.state.db_pool.acquire() as conn:
        await _get_managed_user(conn, user_id)
        return await user_service.update_user(conn, user_id, updates)


@router.put("/users/{user_id}/status")
async def update_user_status(
    request: Request,
    user_id: str,
    status_update: dict,
    session: dict = Depends(get_manager_session),
):
    """Activate/deactivate a student or professor."""
    new_status = status_update.get("status")
    if new_status not in ["active", "inactive"]:
        raise HTTPException(status_code=400, detail="Invalid status. Use 'active' or 'inactive'")

    async with request.app.state.db_pool.acquire() as conn:
        await _get_managed_user(conn, user_id)
        row = await conn.fetchrow(
            """
            UPDATE users
            SET status = $1, updated_at = CURRENT_TIMESTAMP
            WHERE id = $2
            RETURNING id, username, status
            """,
            new_status, user_id,
        )
        return {
            "id": str(row["id"]),
            "username": row["username"],
            "status": row["status"],
            "message": f"User status updated to {new_status}",
        }
