"""Classes: CRUD, ownership boundary, the three enrollment paths and their
env flags, and active-class selection surviving Redis expiry.

Classes are owned by temp professors from make_user; deleting the professor
row afterwards cascades to classes and memberships.
"""

import uuid

import pytest

from app.config import settings
from tests.conftest import auth


@pytest.fixture
async def professor_with_class(client, make_user):
    professor = await make_user(role="professor")
    r = await client.post(
        "/api/v1/professor/classes",
        json={"name": f"Turma {uuid.uuid4().hex[:6]}", "subject": "Cálculo I"},
        headers=auth(professor["token"]),
    )
    assert r.status_code == 200, r.text
    return {"professor": professor, "class": r.json()}


async def test_professor_creates_and_lists_own_classes(client, professor_with_class):
    professor = professor_with_class["professor"]
    cls = professor_with_class["class"]
    assert cls["join_code"] and len(cls["join_code"]) == 8

    r = await client.get("/api/v1/professor/classes", headers=auth(professor["token"]))
    assert r.status_code == 200
    assert any(c["id"] == cls["id"] for c in r.json()["classes"])


async def test_student_joins_by_code_and_selects_class(
    client, make_user, professor_with_class
):
    student = await make_user(role="user")
    cls = professor_with_class["class"]

    r = await client.post(
        "/api/v1/classes/join",
        json={"join_code": cls["join_code"]},
        headers=auth(student["token"]),
    )
    assert r.status_code == 200, r.text
    assert r.json()["class_id"] == cls["id"]

    # Joining again conflicts
    r = await client.post(
        "/api/v1/classes/join",
        json={"join_code": cls["join_code"]},
        headers=auth(student["token"]),
    )
    assert r.status_code == 409

    r = await client.get("/api/v1/classes/mine", headers=auth(student["token"]))
    assert any(c["id"] == cls["id"] for c in r.json()["classes"])

    r = await client.post(
        "/api/v1/session/class",
        json={"class_id": cls["id"]},
        headers=auth(student["token"]),
    )
    assert r.status_code == 200, r.text
    assert r.json()["subject"] == "Cálculo I"
    assert r.json()["conversation_id"]

    r = await client.get("/api/v1/session/class", headers=auth(student["token"]))
    assert r.json()["class_id"] == cls["id"]


async def test_unknown_and_disabled_join_codes_rejected(
    client, make_user, professor_with_class
):
    student = await make_user(role="user")
    professor = professor_with_class["professor"]
    cls = professor_with_class["class"]

    r = await client.post(
        "/api/v1/classes/join",
        json={"join_code": "NOPE1234"},
        headers=auth(student["token"]),
    )
    assert r.status_code == 404

    r = await client.put(
        f"/api/v1/professor/classes/{cls['id']}/join-code",
        json={"enabled": False},
        headers=auth(professor["token"]),
    )
    assert r.status_code == 200

    r = await client.post(
        "/api/v1/classes/join",
        json={"join_code": cls["join_code"]},
        headers=auth(student["token"]),
    )
    assert r.status_code == 404


async def test_join_code_flag_disables_endpoint(
    client, make_user, professor_with_class, monkeypatch
):
    student = await make_user(role="user")
    monkeypatch.setattr(settings, "enrollment_join_code_enabled", False)
    r = await client.post(
        "/api/v1/classes/join",
        json={"join_code": professor_with_class["class"]["join_code"]},
        headers=auth(student["token"]),
    )
    assert r.status_code == 403


async def test_professor_enrolls_by_registration_number(
    client, make_user, professor_with_class, monkeypatch
):
    student = await make_user(role="user")
    professor = professor_with_class["professor"]
    cls = professor_with_class["class"]

    r = await client.post(
        f"/api/v1/professor/classes/{cls['id']}/students",
        json={"registration_number": student["registration_number"]},
        headers=auth(professor["token"]),
    )
    assert r.status_code == 200, r.text
    assert r.json()["username"] == student["username"]

    # Duplicate enrollment conflicts; unknown registration is a clear 404
    r = await client.post(
        f"/api/v1/professor/classes/{cls['id']}/students",
        json={"registration_number": student["registration_number"]},
        headers=auth(professor["token"]),
    )
    assert r.status_code == 409

    r = await client.post(
        f"/api/v1/professor/classes/{cls['id']}/students",
        json={"registration_number": "RA-DOES-NOT-EXIST"},
        headers=auth(professor["token"]),
    )
    assert r.status_code == 404

    r = await client.get(
        f"/api/v1/professor/classes/{cls['id']}/students",
        headers=auth(professor["token"]),
    )
    roster = r.json()["students"]
    assert [s["username"] for s in roster] == [student["username"]]
    assert roster[0]["enrolled_via"] == "professor"

    # Flag off blocks the path
    monkeypatch.setattr(settings, "enrollment_by_registration_enabled", False)
    r = await client.post(
        f"/api/v1/professor/classes/{cls['id']}/students",
        json={"registration_number": student["registration_number"]},
        headers=auth(professor["token"]),
    )
    assert r.status_code == 403


async def test_admin_and_manager_assign_students(
    client, admin_token, make_user, professor_with_class, monkeypatch
):
    manager = await make_user(role="manager")
    s1 = await make_user(role="user")
    s2 = await make_user(role="user")
    cls = professor_with_class["class"]

    r = await client.post(
        f"/api/v1/admin/classes/{cls['id']}/students",
        json={"user_id": s1["id"]},
        headers=auth(admin_token),
    )
    assert r.status_code == 200, r.text

    r = await client.post(
        f"/api/v1/manager/classes/{cls['id']}/students",
        json={"registration_number": s2["registration_number"]},
        headers=auth(manager["token"]),
    )
    assert r.status_code == 200, r.text

    monkeypatch.setattr(settings, "enrollment_admin_assign_enabled", False)
    r = await client.post(
        f"/api/v1/admin/classes/{cls['id']}/students",
        json={"user_id": s1["id"]},
        headers=auth(admin_token),
    )
    assert r.status_code == 403


async def test_professor_cannot_touch_another_professors_class(
    client, make_user, professor_with_class
):
    other = await make_user(role="professor")
    cls = professor_with_class["class"]

    for method, url, body in (
        ("put", f"/api/v1/professor/classes/{cls['id']}", {"name": "Hijacked"}),
        ("delete", f"/api/v1/professor/classes/{cls['id']}", None),
        ("get", f"/api/v1/professor/classes/{cls['id']}/students", None),
        ("post", f"/api/v1/professor/classes/{cls['id']}/join-code/regenerate", None),
    ):
        kwargs = {"headers": auth(other["token"])}
        if body is not None:
            kwargs["json"] = body
        r = await getattr(client, method)(url, **kwargs)
        assert r.status_code == 403, f"{method} {url}: {r.status_code}"


async def test_non_member_cannot_select_class(client, make_user, professor_with_class):
    outsider = await make_user(role="user")
    r = await client.post(
        "/api/v1/session/class",
        json={"class_id": professor_with_class["class"]["id"]},
        headers=auth(outsider["token"]),
    )
    assert r.status_code == 403

    # The owning professor can select their own class (for testing the student UI)
    professor = professor_with_class["professor"]
    r = await client.post(
        "/api/v1/session/class",
        json={"class_id": professor_with_class["class"]["id"]},
        headers=auth(professor["token"]),
    )
    assert r.status_code == 200


async def test_active_class_survives_redis_expiry(
    app, client, make_user, professor_with_class
):
    student = await make_user(role="user")
    cls = professor_with_class["class"]

    r = await client.post(
        "/api/v1/classes/join",
        json={"join_code": cls["join_code"]},
        headers=auth(student["token"]),
    )
    assert r.status_code == 200

    r = await client.post(
        "/api/v1/session/class",
        json={"class_id": cls["id"]},
        headers=auth(student["token"]),
    )
    assert r.status_code == 200

    # Simulate Redis TTL expiry: the session must rehydrate from Postgres
    await app.state.redis.delete(f"session:{student['token']}")

    r = await client.get("/api/v1/session/class", headers=auth(student["token"]))
    assert r.status_code == 200, r.text
    assert r.json()["class_id"] == cls["id"]


async def test_manager_creates_class_for_professor(client, make_user):
    manager = await make_user(role="manager")
    professor = await make_user(role="professor")
    student = await make_user(role="user")

    r = await client.post(
        "/api/v1/manager/classes",
        json={
            "name": f"Turma {uuid.uuid4().hex[:6]}",
            "subject": "Física II",
            "professor_id": professor["id"],
        },
        headers=auth(manager["token"]),
    )
    assert r.status_code == 200, r.text
    cls = r.json()
    assert cls["professor_id"] == professor["id"]

    # A student cannot own a class
    r = await client.post(
        "/api/v1/manager/classes",
        json={
            "name": "Bad",
            "subject": "X",
            "professor_id": student["id"],
        },
        headers=auth(manager["token"]),
    )
    assert r.status_code == 400

    r = await client.get("/api/v1/manager/classes", headers=auth(manager["token"]))
    listed = next(c for c in r.json()["classes"] if c["id"] == cls["id"])
    assert listed["professor_username"] == professor["username"]

    r = await client.get("/api/v1/manager/professors", headers=auth(manager["token"]))
    assert any(p["id"] == professor["id"] for p in r.json()["professors"])
