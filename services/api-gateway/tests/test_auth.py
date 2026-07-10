"""Auth and session behavior through the real HTTP routes."""

import os

from tests.conftest import auth


async def test_admin_login_returns_admin_session(client):
    r = await client.post(
        "/api/v1/login",
        json={"username": "admin", "password": os.environ["ADMIN_PASSWORD"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["role"] == "admin"
    assert body["session_id"]


async def test_guest_login_returns_user_session(client):
    r = await client.post(
        "/api/v1/login",
        json={"username": "guest", "password": os.environ["GUEST_PASSWORD"]},
    )
    assert r.status_code == 200
    assert r.json()["role"] == "user"


async def test_wrong_password_rejected(client):
    r = await client.post(
        "/api/v1/login",
        json={"username": "admin", "password": "definitely-wrong-password"},
    )
    assert r.status_code == 401


async def test_unknown_user_rejected(client):
    r = await client.post(
        "/api/v1/login",
        json={"username": "no-such-user-xyz", "password": "whatever"},
    )
    assert r.status_code == 401


async def test_protected_route_requires_token(client):
    r = await client.get("/api/v1/books")
    assert r.status_code == 401


async def test_garbage_token_rejected(client):
    r = await client.get("/api/v1/books", headers=auth("not-a-real-token"))
    assert r.status_code == 401


async def test_guest_cannot_reach_admin_routes(client, guest_token):
    r = await client.get("/api/v1/admin/users", headers=auth(guest_token))
    assert r.status_code == 403


async def test_admin_can_reach_admin_routes(client, admin_token):
    r = await client.get("/api/v1/admin/users", headers=auth(admin_token))
    assert r.status_code == 200
    usernames = {u["username"] for u in r.json()}
    assert "admin" in usernames


async def test_guest_admin_login_forbidden(client):
    r = await client.post(
        "/api/v1/admin/login",
        json={"username": "guest", "password": os.environ["GUEST_PASSWORD"]},
    )
    assert r.status_code == 403


async def test_session_persists_across_requests_until_logout(client, guest_token):
    r = await client.get("/api/v1/validate-session", headers=auth(guest_token))
    assert r.status_code == 200
    assert r.json()["valid"] is True

    r = await client.post("/api/v1/logout", headers=auth(guest_token))
    assert r.status_code == 200

    r = await client.get("/api/v1/validate-session", headers=auth(guest_token))
    assert r.json()["valid"] is False


async def test_session_subject_roundtrip(client, guest_token):
    r = await client.post(
        "/api/v1/session/subject",
        json={"subject": "Linear Algebra"},
        headers=auth(guest_token),
    )
    assert r.status_code == 200

    r = await client.get("/api/v1/session/subject", headers=auth(guest_token))
    assert r.status_code == 200
    assert r.json()["subject"] == "Linear Algebra"


async def test_models_endpoint_lists_configured_models(client, guest_token):
    r = await client.get("/api/v1/models", headers=auth(guest_token))
    assert r.status_code == 200
    body = r.json()
    assert body["available"]
    assert body["default"] in {m["value"] for m in body["available"]}
