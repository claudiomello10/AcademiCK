"""Security utilities for authentication and password hashing."""

import bcrypt
import secrets
from typing import Optional, Dict
import asyncpg

from app.config import settings


# Config users for testing (loaded from environment)
CONFIG_USERS: Dict[str, Dict] = {}

if settings.config_users_enabled:
    CONFIG_USERS = {
        "admin": {
            "password": settings.admin_password,
            "role": "admin",
            "email": "admin@academick.local"
        },
        "guest": {
            "password": settings.guest_password,
            "role": "user",
            "email": "guest@academick.local"
        }
    }


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode(), salt).decode()


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except Exception:
        return False


def generate_session_token() -> str:
    """Generate a secure session token."""
    return secrets.token_urlsafe(32)


async def initialize_config_users(pool: asyncpg.Pool) -> None:
    """Initialize config users in the database if they don't exist."""
    if not settings.config_users_enabled:
        return

    async with pool.acquire() as conn:
        for username, user_data in CONFIG_USERS.items():
            # Check if user exists
            exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM users WHERE username = $1)",
                username
            )

            if not exists:
                # Create config user
                password_hash = hash_password(user_data["password"])
                await conn.execute("""
                    INSERT INTO users (username, email, password_hash, role, is_config_user)
                    VALUES ($1, $2, $3, $4, true)
                """, username, user_data["email"], password_hash, user_data["role"])


async def authenticate_user(
    pool: asyncpg.Pool,
    username: str,
    password: str
) -> Optional[Dict]:
    """
    Authenticate a user.

    First checks config users (fast path for testing),
    then checks database users.

    Returns user dict if authenticated, None otherwise.
    """
    # Check config users first
    if settings.config_users_enabled and username in CONFIG_USERS:
        if password == CONFIG_USERS[username]["password"]:
            # Fetch the real UUID from the database (created by initialize_config_users)
            async with pool.acquire() as conn:
                user_id = await conn.fetchval(
                    "SELECT id FROM users WHERE username = $1", username
                )
            return {
                "id": str(user_id) if user_id else username,
                "username": username,
                "role": CONFIG_USERS[username]["role"],
                "is_config_user": True
            }

    # Check database users
    async with pool.acquire() as conn:
        user = await conn.fetchrow("""
            SELECT id, username, password_hash, role, status
            FROM users
            WHERE username = $1 AND status = 'active'
        """, username)

        if user and verify_password(password, user["password_hash"]):
            # Update last active
            await conn.execute(
                "UPDATE users SET last_active = CURRENT_TIMESTAMP WHERE id = $1",
                user["id"]
            )

            return {
                "id": str(user["id"]),
                "username": user["username"],
                "role": user["role"],
                "is_config_user": False
            }

    return None
