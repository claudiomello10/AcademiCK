"""User management: the database-user auth path, end to end.

Config users cover the fast path; these tests exercise the bcrypt/DB path —
create a user through the admin API, log in as them, deactivate, and verify
the lockout.
"""

import uuid

import pytest

from tests.conftest import auth


@pytest.fixture
async def temp_user(app, client, admin_token):
    """A freshly created DB user; the row is removed afterwards."""
    username = f"test-user-{uuid.uuid4().hex[:8]}"
    password = "temp-user-password"
    r = await client.post(
        "/api/v1/admin/users",
        json={"username": username, "password": password, "role": "user"},
        headers=auth(admin_token),
    )
    assert r.status_code == 200, r.text
    yield {"id": r.json()["id"], "username": username, "password": password}
    async with app.state.db_pool.acquire() as conn:
        await conn.execute("DELETE FROM users WHERE username = $1", username)


async def test_created_user_can_login_with_db_credentials(client, temp_user):
    r = await client.post(
        "/api/v1/login",
        json={"username": temp_user["username"], "password": temp_user["password"]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["role"] == "user"

    r = await client.get(
        "/api/v1/validate-session", headers=auth(body["session_id"])
    )
    assert r.json()["valid"] is True


async def test_deactivated_user_is_locked_out(client, admin_token, temp_user):
    r = await client.put(
        f"/api/v1/admin/users/{temp_user['id']}/status",
        json={"status": "inactive"},
        headers=auth(admin_token),
    )
    assert r.status_code == 200

    r = await client.post(
        "/api/v1/login",
        json={"username": temp_user["username"], "password": temp_user["password"]},
    )
    assert r.status_code == 401


async def test_duplicate_username_rejected(client, admin_token, temp_user):
    r = await client.post(
        "/api/v1/admin/users",
        json={"username": temp_user["username"], "password": "whatever-else", "role": "user"},
        headers=auth(admin_token),
    )
    assert r.status_code == 400


async def test_role_change_grants_and_revokes_admin_access(
    client, admin_token, temp_user
):
    r = await client.post(
        "/api/v1/login",
        json={"username": temp_user["username"], "password": temp_user["password"]},
    )
    token = r.json()["session_id"]
    r = await client.get("/api/v1/admin/users", headers=auth(token))
    assert r.status_code == 403

    r = await client.put(
        f"/api/v1/admin/users/{temp_user['id']}",
        json={"role": "admin"},
        headers=auth(admin_token),
    )
    assert r.status_code == 200

    # Role lives in the session: a fresh login picks up the new role.
    r = await client.post(
        "/api/v1/login",
        json={"username": temp_user["username"], "password": temp_user["password"]},
    )
    assert r.json()["role"] == "admin"
    r = await client.get(
        "/api/v1/admin/users", headers=auth(r.json()["session_id"])
    )
    assert r.status_code == 200
