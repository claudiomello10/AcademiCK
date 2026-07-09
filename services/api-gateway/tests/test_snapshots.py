"""Snapshot management: a snapshot must bring back exactly what existed.

The round-trip is the disaster-recovery path: create a snapshot, destroy the
data, restore, and verify nothing was lost — every Qdrant point (id and
payload) and the books/chapters rows in Postgres.

Note the restore contract: Qdrant vectors and the books/chapters metadata are
restored; rows in the `chunks` table are not part of the snapshot.
"""

import pytest
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.config import settings
from tests.conftest import auth

pytestmark = pytest.mark.skipif(
    not settings.enable_snapshot_management,
    reason="snapshot management disabled",
)


async def _book_points(app, book_name: str) -> dict:
    """All Qdrant points of a book: id -> the payload fields that matter."""
    points = {}
    offset = None
    while True:
        batch, offset = await app.state.qdrant.client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="book_name", match=MatchValue(value=book_name))]
            ),
            limit=1000,
            offset=offset,
            with_payload=True,
        )
        for p in batch:
            points[str(p.id)] = (
                p.payload["text"],
                p.payload["chapter_title"],
                p.payload["chunk_index"],
            )
        if offset is None:
            break
    return points


async def _collection_count(app) -> int:
    result = await app.state.qdrant.client.count(
        collection_name=settings.qdrant_collection
    )
    return result.count


async def test_snapshot_roundtrip_restores_deleted_book(
    app, client, admin_token, seed_book
):
    book = await seed_book()

    pre_points = await _book_points(app, book["name"])
    pre_total = await _collection_count(app)
    async with app.state.db_pool.acquire() as conn:
        pre_book = dict(await conn.fetchrow(
            "SELECT id, name, total_chunks, processing_status FROM books WHERE name = $1",
            book["name"],
        ))
        pre_chapters = [
            dict(r) for r in await conn.fetch(
                "SELECT id, title, chapter_number FROM chapters "
                "WHERE book_id = $1 ORDER BY chapter_number",
                book["id"],
            )
        ]
    assert pre_points and pre_book["processing_status"] == "completed"

    r = await client.post("/api/v1/admin/snapshots/create", headers=auth(admin_token))
    assert r.status_code == 200, r.text
    snapshot = r.json()["snapshot_name"]

    try:
        r = await client.get("/api/v1/admin/snapshots", headers=auth(admin_token))
        assert r.status_code == 200
        listed = next(s for s in r.json()["snapshots"] if s["name"] == snapshot)
        assert listed["has_metadata"] is True

        # Destroy: the book vanishes from Qdrant and Postgres.
        r = await client.delete(
            f"/api/v1/admin/books/{book['name']}", headers=auth(admin_token)
        )
        assert r.status_code == 200
        assert await _book_points(app, book["name"]) == {}

        r = await client.post(
            f"/api/v1/admin/snapshots/{snapshot}/restore", headers=auth(admin_token)
        )
        assert r.status_code == 200, r.text
        assert r.json()["success"] is True

        # Nothing lost: identical points, identical totals, identical rows.
        assert await _book_points(app, book["name"]) == pre_points
        assert await _collection_count(app) == pre_total
        async with app.state.db_pool.acquire() as conn:
            post_book = dict(await conn.fetchrow(
                "SELECT id, name, total_chunks, processing_status FROM books WHERE name = $1",
                book["name"],
            ))
            post_chapters = [
                dict(r) for r in await conn.fetch(
                    "SELECT id, title, chapter_number FROM chapters "
                    "WHERE book_id = $1 ORDER BY chapter_number",
                    book["id"],
                )
            ]
        assert post_book == pre_book
        assert post_chapters == pre_chapters

        # And the product sees it again.
        r = await client.get("/api/v1/books", headers=auth(admin_token))
        assert book["name"] in {b["name"] for b in r.json()["books"]}

    finally:
        r = await client.delete(
            f"/api/v1/admin/snapshots/{snapshot}", headers=auth(admin_token)
        )
        assert r.status_code == 200

    r = await client.get("/api/v1/admin/snapshots", headers=auth(admin_token))
    assert snapshot not in {s["name"] for s in r.json()["snapshots"]}


async def test_failed_metadata_write_leaves_no_orphan(
    client, admin_token, monkeypatch, tmp_path
):
    """A snapshot without metadata can't be restored, so when the metadata
    write fails the snapshot must be discarded, not left orphaned."""
    readonly = tmp_path / "readonly"
    readonly.mkdir()
    readonly.chmod(0o500)
    monkeypatch.setattr(settings, "snapshot_dir", str(readonly))

    r = await client.get("/api/v1/admin/snapshots", headers=auth(admin_token))
    before = {s["name"] for s in r.json()["snapshots"]}

    try:
        r = await client.post(
            "/api/v1/admin/snapshots/create", headers=auth(admin_token)
        )
        assert r.status_code == 500
        assert "metadata could not be written" in r.json()["detail"]

        r = await client.get("/api/v1/admin/snapshots", headers=auth(admin_token))
        assert {s["name"] for s in r.json()["snapshots"]} == before
    finally:
        readonly.chmod(0o700)


async def test_restore_without_metadata_is_rejected(client, admin_token):
    r = await client.post(
        "/api/v1/admin/snapshots/no-such-snapshot-xyz/restore",
        headers=auth(admin_token),
    )
    assert r.status_code == 400


async def test_snapshot_endpoints_require_admin(client, guest_token):
    r = await client.post("/api/v1/admin/snapshots/create", headers=auth(guest_token))
    assert r.status_code == 403
