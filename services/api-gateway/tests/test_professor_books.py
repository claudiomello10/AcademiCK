"""Professor book management: catalog visibility, attach/detach rules,
env-gated upload, name-collision guard, owned deletion, job restriction.

Selecting (attaching catalog books) and uploading are distinct capabilities;
only uploading is switchable via PROFESSOR_BOOK_UPLOAD_ENABLED.
"""

import json
import uuid
from uuid import UUID

import pytest

from app.config import settings
from tests.conftest import auth


@pytest.fixture
async def professor_with_class(client, make_user):
    professor = await make_user(role="professor")
    r = await client.post(
        "/api/v1/professor/classes",
        json={"name": f"books-{uuid.uuid4().hex[:6]}", "subject": "Química"},
        headers=auth(professor["token"]),
    )
    assert r.status_code == 200, r.text
    return {"professor": professor, "class": r.json()}


async def set_book_owner(app, book_id: str, owner_id: str) -> None:
    async with app.state.db_pool.acquire() as conn:
        await conn.execute(
            "UPDATE books SET owner_user_id = $1 WHERE id = $2",
            UUID(owner_id), UUID(book_id),
        )


async def test_catalog_shows_global_and_own_but_not_foreign_books(
    app, client, make_user, professor_with_class, seed_book
):
    professor = professor_with_class["professor"]
    other = await make_user(role="professor")

    global_book = await seed_book(topic=0)
    own_book = await seed_book(topic=1)
    foreign_book = await seed_book(topic=0)
    await set_book_owner(app, own_book["id"], professor["id"])
    await set_book_owner(app, foreign_book["id"], other["id"])

    r = await client.get(
        "/api/v1/professor/books/catalog", headers=auth(professor["token"])
    )
    assert r.status_code == 200, r.text
    catalog = {b["name"]: b for b in r.json()["books"]}
    assert global_book["name"] in catalog
    assert not catalog[global_book["name"]]["owned"]
    assert own_book["name"] in catalog
    assert catalog[own_book["name"]]["owned"]
    assert foreign_book["name"] not in catalog


async def test_professor_attaches_and_detaches_catalog_books(
    client, professor_with_class, seed_book
):
    professor = professor_with_class["professor"]
    cls = professor_with_class["class"]
    book = await seed_book()

    r = await client.post(
        f"/api/v1/professor/classes/{cls['id']}/books/{book['id']}/attach",
        headers=auth(professor["token"]),
    )
    assert r.status_code == 200, r.text

    r = await client.get(
        f"/api/v1/professor/books/catalog?class_id={cls['id']}",
        headers=auth(professor["token"]),
    )
    attached = {b["name"] for b in r.json()["books"] if b["attached"]}
    assert book["name"] in attached

    r = await client.get(
        f"/api/v1/professor/classes/{cls['id']}/books",
        headers=auth(professor["token"]),
    )
    assert book["name"] in {b["name"] for b in r.json()["books"]}

    r = await client.delete(
        f"/api/v1/professor/classes/{cls['id']}/books/{book['id']}/detach",
        headers=auth(professor["token"]),
    )
    assert r.status_code == 200

    r = await client.get(
        f"/api/v1/professor/classes/{cls['id']}/books",
        headers=auth(professor["token"]),
    )
    assert r.json()["books"] == []


async def test_professor_cannot_attach_foreign_book(
    app, client, make_user, professor_with_class, seed_book
):
    professor = professor_with_class["professor"]
    cls = professor_with_class["class"]
    other = await make_user(role="professor")
    foreign_book = await seed_book()
    await set_book_owner(app, foreign_book["id"], other["id"])

    r = await client.post(
        f"/api/v1/professor/classes/{cls['id']}/books/{foreign_book['id']}/attach",
        headers=auth(professor["token"]),
    )
    assert r.status_code == 403


def pdf_files(name: str) -> dict:
    return {"files": (name, b"%PDF-fake", "application/pdf")}


async def test_upload_gates(client, make_user, professor_with_class, monkeypatch):
    professor = professor_with_class["professor"]
    cls = professor_with_class["class"]

    # Another professor's class
    other = await make_user(role="professor")
    r = await client.post(
        f"/api/v1/professor/classes/{cls['id']}/books/upload",
        files=pdf_files("anything.pdf"),
        headers=auth(other["token"]),
    )
    assert r.status_code == 403

    # Flag off blocks upload entirely
    monkeypatch.setattr(settings, "professor_book_upload_enabled", False)
    r = await client.post(
        f"/api/v1/professor/classes/{cls['id']}/books/upload",
        files=pdf_files("anything.pdf"),
        headers=auth(professor["token"]),
    )
    assert r.status_code == 403
    assert "disabled" in r.json()["detail"]


async def test_upload_name_collision_is_409(
    client, professor_with_class, seed_book
):
    professor = professor_with_class["professor"]
    cls = professor_with_class["class"]
    existing = await seed_book()  # global book, owner NULL

    r = await client.post(
        f"/api/v1/professor/classes/{cls['id']}/books/upload",
        files=pdf_files(f"{existing['name']}.pdf"),
        headers=auth(professor["token"]),
    )
    assert r.status_code == 409
    assert existing["name"] in r.json()["detail"]


async def test_professor_deletes_own_book_but_not_global(
    app, client, professor_with_class, seed_book
):
    professor = professor_with_class["professor"]
    cls = professor_with_class["class"]
    own_book = await seed_book(topic=0)
    global_book = await seed_book(topic=1)
    await set_book_owner(app, own_book["id"], professor["id"])

    r = await client.delete(
        f"/api/v1/professor/classes/{cls['id']}/books/{global_book['id']}",
        headers=auth(professor["token"]),
    )
    assert r.status_code == 403

    r = await client.delete(
        f"/api/v1/professor/classes/{cls['id']}/books/{own_book['id']}",
        headers=auth(professor["token"]),
    )
    assert r.status_code == 200, r.text
    assert r.json()["vectors_deleted"] == own_book["n_chunks"]

    async with app.state.db_pool.acquire() as conn:
        assert not await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM books WHERE id = $1)", UUID(own_book["id"])
        )


async def test_job_status_restricted_to_own_uploads(
    app, client, make_user, professor_with_class
):
    professor = professor_with_class["professor"]
    other = await make_user(role="professor")

    async with app.state.db_pool.acquire() as conn:
        job_id = await conn.fetchval("""
            INSERT INTO processing_jobs (job_type, status, progress, metadata)
            VALUES ('pdf_processing', 'completed', 1, $1)
            RETURNING id
        """, json.dumps({
            "celery_task_id": "irrelevant",
            "filename": "someone-elses.pdf",
            "uploaded_by": other["id"],
        }))

    try:
        r = await client.get(
            f"/api/v1/professor/pdf-job/{job_id}", headers=auth(professor["token"])
        )
        assert r.status_code == 403

        r = await client.get(
            f"/api/v1/professor/pdf-job/{job_id}", headers=auth(other["token"])
        )
        assert r.status_code == 200
        assert r.json()["filename"] == "someone-elses.pdf"
    finally:
        async with app.state.db_pool.acquire() as conn:
            await conn.execute("DELETE FROM processing_jobs WHERE id = $1", job_id)
