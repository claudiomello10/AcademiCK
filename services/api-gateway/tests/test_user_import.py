"""Bulk user import: happy path, per-row error reporting, format rejection."""

import uuid

import pytest

from tests.conftest import auth


@pytest.fixture
async def cleanup_users(app):
    """Collects usernames to delete after the test."""
    usernames = []
    yield usernames
    async with app.state.db_pool.acquire() as conn:
        for username in usernames:
            await conn.execute("DELETE FROM users WHERE username = $1", username)


def csv_upload(content: str, filename: str = "students.csv") -> dict:
    return {"file": (filename, content.encode(), "text/csv")}


async def test_csv_import_creates_users_who_can_login(
    client, admin_token, cleanup_users
):
    u1 = f"test-imp-{uuid.uuid4().hex[:8]}"
    u2 = f"test-imp-{uuid.uuid4().hex[:8]}"
    prof = f"test-imp-{uuid.uuid4().hex[:8]}"
    cleanup_users.extend([u1, u2, prof])

    content = (
        "username,password,email,role,registration_number\n"
        f"{u1},password-one,,user,RA{uuid.uuid4().hex[:10]}\n"
        f"{u2},password-two,{u2}@example.com,user,RA{uuid.uuid4().hex[:10]}\n"
        f"{prof},password-three,,professor,\n"
    )
    r = await client.post(
        "/api/v1/admin/users/import",
        files=csv_upload(content),
        headers=auth(admin_token),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["created"] == 3
    assert body["errors"] == []

    r = await client.post(
        "/api/v1/login", json={"username": u1, "password": "password-one"}
    )
    assert r.status_code == 200


async def test_import_reports_row_errors_without_aborting(
    client, admin_token, cleanup_users
):
    ok_user = f"test-imp-{uuid.uuid4().hex[:8]}"
    dup_user = f"test-imp-{uuid.uuid4().hex[:8]}"
    registration = f"RA{uuid.uuid4().hex[:10]}"
    cleanup_users.extend([ok_user, dup_user])

    content = (
        "username,password,role,registration_number\n"
        f"{ok_user},password-one,user,{registration}\n"
        f"{dup_user},password-two,user,{registration}\n"
        f",password-three,user,RA{uuid.uuid4().hex[:10]}\n"
    )
    r = await client.post(
        "/api/v1/admin/users/import",
        files=csv_upload(content),
        headers=auth(admin_token),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["created"] == 1
    assert len(body["errors"]) == 2
    # Duplicate registration is called out on the failing row (row 3).
    dup_error = next(e for e in body["errors"] if e["row"] == 3)
    assert registration in dup_error["error"]


async def test_import_uses_default_password_when_column_empty(
    client, admin_token, cleanup_users
):
    username = f"test-imp-{uuid.uuid4().hex[:8]}"
    cleanup_users.append(username)

    content = (
        "username,registration_number\n"
        f"{username},RA{uuid.uuid4().hex[:10]}\n"
    )
    r = await client.post(
        "/api/v1/admin/users/import",
        files=csv_upload(content),
        data={"default_password": "shared-start-password"},
        headers=auth(admin_token),
    )
    assert r.status_code == 200, r.text
    assert r.json()["created"] == 1

    r = await client.post(
        "/api/v1/login",
        json={"username": username, "password": "shared-start-password"},
    )
    assert r.status_code == 200


async def test_import_rejects_unsupported_and_malformed_files(client, admin_token):
    r = await client.post(
        "/api/v1/admin/users/import",
        files={"file": ("users.xml", b"<users/>", "application/xml")},
        headers=auth(admin_token),
    )
    assert r.status_code == 400

    r = await client.post(
        "/api/v1/admin/users/import",
        files=csv_upload("name,pass\nsomeone,x\n"),
        headers=auth(admin_token),
    )
    assert r.status_code == 400


async def test_manager_import_cannot_create_admins(
    client, make_user, cleanup_users
):
    manager = await make_user(role="manager")
    student = f"test-imp-{uuid.uuid4().hex[:8]}"
    would_be_admin = f"test-imp-{uuid.uuid4().hex[:8]}"
    cleanup_users.extend([student, would_be_admin])

    content = (
        "username,password,role,registration_number\n"
        f"{student},password-one,user,RA{uuid.uuid4().hex[:10]}\n"
        f"{would_be_admin},password-two,admin,\n"
    )
    r = await client.post(
        "/api/v1/manager/users/import",
        files=csv_upload(content),
        headers=auth(manager["token"]),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["created"] == 1
    assert len(body["errors"]) == 1
    assert body["errors"][0]["username"] == would_be_admin
