"""Book listing and admin book management against real Postgres and Qdrant."""

from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.config import settings
from tests.conftest import auth


async def _qdrant_count(app, book_name: str) -> int:
    result = await app.state.qdrant.client.count(
        collection_name=settings.qdrant_collection,
        count_filter=Filter(
            must=[FieldCondition(key="book_name", match=MatchValue(value=book_name))]
        ),
    )
    return result.count


async def test_attached_book_appears_in_listings(
    client, guest_token, guest_classroom, seed_book
):
    book = await seed_book()
    other = await seed_book(topic=1)
    await guest_classroom["attach"](book["id"])

    r = await client.get("/api/v1/books", headers=auth(guest_token))
    assert r.status_code == 200
    names = {b["name"] for b in r.json()["books"]}
    assert book["name"] in names
    # Books not attached to the active class are invisible
    assert other["name"] not in names

    r = await client.get("/api/v1/books/names/list", headers=auth(guest_token))
    assert r.status_code == 200
    assert r.json()["books"] == [book["name"]]


async def test_book_listing_requires_active_class(client, guest_token, seed_book):
    await seed_book()
    r = await client.get("/api/v1/books", headers=auth(guest_token))
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "no_active_class"


async def test_get_book_details_from_postgres(client, guest_token, seed_book):
    book = await seed_book()
    r = await client.get(f"/api/v1/books/{book['id']}", headers=auth(guest_token))
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == book["name"]
    assert body["total_chunks"] == book["n_chunks"]
    assert len(body["chapters"]) == 1


async def test_get_unknown_book_is_404(client, guest_token):
    r = await client.get(
        "/api/v1/books/00000000-0000-0000-0000-000000000000",
        headers=auth(guest_token),
    )
    assert r.status_code == 404


async def test_admin_delete_removes_book_everywhere(
    app, client, admin_token, guest_token, guest_classroom, seed_book
):
    book = await seed_book()
    await guest_classroom["attach"](book["id"])
    assert await _qdrant_count(app, book["name"]) == book["n_chunks"]

    r = await client.delete(
        f"/api/v1/admin/books/{book['name']}", headers=auth(admin_token)
    )
    assert r.status_code == 200
    body = r.json()
    assert body["vectors_deleted"] == book["n_chunks"]
    assert body["pg_deleted"] is True

    assert await _qdrant_count(app, book["name"]) == 0
    async with app.state.db_pool.acquire() as conn:
        assert not await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM books WHERE name = $1)", book["name"]
        )

    r = await client.get("/api/v1/books", headers=auth(guest_token))
    assert book["name"] not in {b["name"] for b in r.json()["books"]}


async def test_guest_cannot_delete_book(app, client, guest_token, seed_book):
    book = await seed_book()
    r = await client.delete(
        f"/api/v1/admin/books/{book['name']}", headers=auth(guest_token)
    )
    assert r.status_code == 403
    assert await _qdrant_count(app, book["name"]) == book["n_chunks"]


async def test_delete_unknown_book_is_not_a_server_error(client, admin_token):
    r = await client.delete(
        "/api/v1/admin/books/no-such-book-xyz", headers=auth(admin_token)
    )
    assert r.status_code < 500
