"""Class-scoped retrieval isolation.

The invariant under test: a session scoped to a class can never surface
content from books outside that class — not via chat, not via a
book-targeted agent search, not via a stale cross-class search cache.
"""

import uuid

import pytest

from tests.conftest import BOOK_TOPICS, auth


@pytest.fixture
async def classroom_factory(app, client, admin_token, make_user):
    """Creates a class owned by a fresh professor, enrolls a fresh student,
    selects it as the student's active class, and attaches given books."""

    async def _create(book_ids: list) -> dict:
        professor = await make_user(role="professor")
        student = await make_user(role="user")

        r = await client.post(
            "/api/v1/professor/classes",
            json={"name": f"scope-{uuid.uuid4().hex[:8]}", "subject": "Machine Learning"},
            headers=auth(professor["token"]),
        )
        assert r.status_code == 200, r.text
        cls = r.json()

        r = await client.post(
            f"/api/v1/admin/classes/{cls['id']}/students",
            json={"user_id": student["id"]},
            headers=auth(admin_token),
        )
        assert r.status_code == 200, r.text

        for book_id in book_ids:
            r = await client.post(
                f"/api/v1/admin/classes/{cls['id']}/books/{book_id}/attach",
                headers=auth(admin_token),
            )
            assert r.status_code == 200, r.text

        r = await client.post(
            "/api/v1/session/class",
            json={"class_id": cls["id"]},
            headers=auth(student["token"]),
        )
        assert r.status_code == 200, r.text

        return {"class": cls, "student": student, "professor": professor}

    return _create


async def test_chat_cannot_cite_books_outside_the_class(
    client, classroom_factory, seed_book, fake_llm
):
    book_a = await seed_book(topic=0)
    book_b = await seed_book(topic=1)
    room_a = await classroom_factory([book_a["id"]])

    # Ask about book B's topic from inside class A: retrieval is allowlisted
    # to book A, so nothing from book B may appear.
    fake_llm["query"] = f"{BOOK_TOPICS[1]['question']} ({uuid.uuid4().hex[:8]})"
    r = await client.post(
        "/api/v1/chat/single",
        json={"query": BOOK_TOPICS[1]["question"]},
        headers=auth(room_a["student"]["token"]),
    )
    assert r.status_code == 200, r.text
    cited = {s["book"] for s in r.json()["sources"]}
    assert book_b["name"] not in cited


async def test_agent_book_targeted_search_cannot_escape_allowlist(
    client, classroom_factory, seed_book, fake_llm
):
    book_a = await seed_book(topic=0)
    book_b = await seed_book(topic=1)
    room_a = await classroom_factory([book_a["id"]])

    # The curation model explicitly scopes its search to the other class's
    # book. Whatever the fuzzy book matcher does with the name, the
    # server-side MatchAny means only class-A books can come back.
    fake_llm["book"] = book_b["name"]
    fake_llm["query"] = f"{BOOK_TOPICS[1]['question']} ({uuid.uuid4().hex[:8]})"
    r = await client.post(
        "/api/v1/chat/single",
        json={"query": BOOK_TOPICS[1]["question"]},
        headers=auth(room_a["student"]["token"]),
    )
    assert r.status_code == 200, r.text
    cited = {s["book"] for s in r.json()["sources"]}
    assert book_b["name"] not in cited
    assert cited <= {book_a["name"]}


async def test_search_cache_does_not_leak_across_classes(
    client, classroom_factory, seed_book, fake_llm
):
    book_a = await seed_book(topic=0)
    book_b = await seed_book(topic=1)
    room_a = await classroom_factory([book_a["id"]])
    room_b = await classroom_factory([book_b["id"]])

    # Same query text in both classes: identical cache input except the
    # allowlist. Class A caches book-A results first.
    query = BOOK_TOPICS[0]["question"]
    fake_llm["query"] = f"{query} ({uuid.uuid4().hex[:8]})"

    r = await client.post(
        "/api/v1/chat/single",
        json={"query": query},
        headers=auth(room_a["student"]["token"]),
    )
    assert r.status_code == 200, r.text
    assert book_a["name"] in {s["book"] for s in r.json()["sources"]}

    r = await client.post(
        "/api/v1/chat/single",
        json={"query": query},
        headers=auth(room_b["student"]["token"]),
    )
    assert r.status_code == 200, r.text
    cited = {s["book"] for s in r.json()["sources"]}
    assert book_a["name"] not in cited


async def test_admin_detach_hides_book_again(
    app, client, admin_token, guest_token, guest_classroom, seed_book
):
    book = await seed_book()
    await guest_classroom["attach"](book["id"])

    r = await client.get("/api/v1/books/names/list", headers=auth(guest_token))
    assert book["name"] in r.json()["books"]

    r = await client.delete(
        f"/api/v1/admin/classes/{guest_classroom['id']}/books/{book['id']}/detach",
        headers=auth(admin_token),
    )
    assert r.status_code == 200

    r = await client.get("/api/v1/books/names/list", headers=auth(guest_token))
    assert book["name"] not in r.json()["books"]
