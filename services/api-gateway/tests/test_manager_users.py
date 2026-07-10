"""Manager role: portal logins and the user-management permission boundary.

Managers administer students and professors only; admin and manager
accounts must be invisible to and untouchable by manager endpoints.
"""

import uuid

import pytest

from tests.conftest import auth


async def test_professor_portal_login_gates_by_role(client, make_user):
    professor = await make_user(role="professor")
    student = await make_user(role="user")

    r = await client.post(
        "/api/v1/professor/login",
        json={"username": professor["username"], "password": professor["password"]},
    )
    assert r.status_code == 200, r.text
    token = r.json()["session_id"]

    r = await client.get("/api/v1/professor/validate-session", headers=auth(token))
    assert r.json()["valid"] is True
    assert r.json()["role"] == "professor"

    r = await client.post(
        "/api/v1/professor/login",
        json={"username": student["username"], "password": student["password"]},
    )
    assert r.status_code == 403


async def test_manager_portal_login_gates_by_role(client, make_user):
    manager = await make_user(role="manager")
    professor = await make_user(role="professor")

    r = await client.post(
        "/api/v1/manager/login",
        json={"username": manager["username"], "password": manager["password"]},
    )
    assert r.status_code == 200, r.text

    r = await client.post(
        "/api/v1/manager/login",
        json={"username": professor["username"], "password": professor["password"]},
    )
    assert r.status_code == 403


async def test_manager_creates_students_and_professors_only(app, client, make_user):
    manager = await make_user(role="manager")
    created = []

    try:
        for role, expected in (
            ("user", 200), ("professor", 200), ("manager", 403), ("admin", 403),
        ):
            username = f"test-mgr-created-{uuid.uuid4().hex[:8]}"
            payload = {"username": username, "password": "some-password", "role": role}
            if role == "user":
                payload["registration_number"] = f"RA{uuid.uuid4().hex[:10]}"
            r = await client.post(
                "/api/v1/manager/users", json=payload, headers=auth(manager["token"])
            )
            assert r.status_code == expected, f"{role}: {r.text}"
            if r.status_code == 200:
                created.append(username)
    finally:
        async with app.state.db_pool.acquire() as conn:
            for username in created:
                await conn.execute("DELETE FROM users WHERE username = $1", username)


async def test_manager_cannot_touch_admin_or_manager_accounts(client, make_user):
    manager = await make_user(role="manager")
    other_manager = await make_user(role="manager")

    r = await client.put(
        f"/api/v1/manager/users/{other_manager['id']}",
        json={"status": "inactive"},
        headers=auth(manager["token"]),
    )
    assert r.status_code == 403

    r = await client.put(
        f"/api/v1/manager/users/{other_manager['id']}/status",
        json={"status": "inactive"},
        headers=auth(manager["token"]),
    )
    assert r.status_code == 403


async def test_manager_cannot_promote_beyond_professor(client, make_user):
    manager = await make_user(role="manager")
    student = await make_user(role="user")

    r = await client.put(
        f"/api/v1/manager/users/{student['id']}",
        json={"role": "admin"},
        headers=auth(manager["token"]),
    )
    assert r.status_code == 403

    r = await client.put(
        f"/api/v1/manager/users/{student['id']}",
        json={"role": "professor"},
        headers=auth(manager["token"]),
    )
    assert r.status_code == 200


async def test_manager_user_list_hides_admins_and_managers(client, make_user):
    manager = await make_user(role="manager")
    student = await make_user(role="user")

    r = await client.get("/api/v1/manager/users", headers=auth(manager["token"]))
    assert r.status_code == 200
    listed = {u["username"]: u["role"] for u in r.json()}
    assert student["username"] in listed
    assert all(role in ("user", "professor") for role in listed.values())


async def test_student_cannot_use_manager_endpoints(client, make_user):
    student = await make_user(role="user")
    r = await client.get("/api/v1/manager/users", headers=auth(student["token"]))
    assert r.status_code == 403
