"""Helper functions for snapshot metadata export/import."""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from asyncpg import Connection
import logging

logger = logging.getLogger(__name__)


def _parse_datetime(value: Any) -> Optional[datetime]:
    """Parse an ISO format datetime string back to a datetime object."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


async def export_metadata(conn: Connection) -> Dict[str, Any]:
    """
    Export books and chapters metadata to JSON.

    Does NOT include chunks table - chunks will be reconstructed from Qdrant.
    """
    try:
        # Export books
        books_rows = await conn.fetch("""
            SELECT id, name, file_path, file_hash, total_pages, total_chunks,
                   processing_status, metadata, created_at, updated_at, processed_at
            FROM books
            WHERE processing_status = 'completed'
            ORDER BY name
        """)

        books = [
            {
                "id": str(row["id"]),
                "name": row["name"],
                "file_path": row["file_path"],
                "file_hash": row["file_hash"],
                "total_pages": row["total_pages"],
                "total_chunks": row["total_chunks"],
                "processing_status": row["processing_status"],
                "metadata": row["metadata"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
                "processed_at": row["processed_at"].isoformat() if row["processed_at"] else None,
            }
            for row in books_rows
        ]

        # Export chapters
        chapters_rows = await conn.fetch("""
            SELECT c.id, c.book_id, c.title, c.chapter_number,
                   c.start_page, c.end_page, c.chunk_count, c.created_at
            FROM chapters c
            JOIN books b ON c.book_id = b.id
            WHERE b.processing_status = 'completed'
            ORDER BY c.book_id, c.chapter_number
        """)

        chapters = [
            {
                "id": str(row["id"]),
                "book_id": str(row["book_id"]),
                "title": row["title"],
                "chapter_number": row["chapter_number"],
                "start_page": row["start_page"],
                "end_page": row["end_page"],
                "chunk_count": row["chunk_count"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            }
            for row in chapters_rows
        ]

        return {
            "version": "1.0",
            "export_type": "academick_metadata",
            "books": books,
            "chapters": chapters,
            "total_books": len(books),
            "total_chapters": len(chapters),
        }

    except Exception as e:
        logger.error(f"Failed to export metadata: {e}")
        raise


async def import_metadata(conn: Connection, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Import books and chapters metadata from JSON.

    Returns import statistics.
    """
    try:
        # Validate metadata format
        if metadata.get("export_type") != "academick_metadata":
            raise ValueError("Invalid metadata file format")

        books = metadata.get("books", [])
        chapters = metadata.get("chapters", [])

        books_imported = 0
        chapters_imported = 0

        # Import books
        for book in books:
            await conn.execute("""
                INSERT INTO books (id, name, file_path, file_hash, total_pages, total_chunks,
                                   processing_status, metadata, created_at, updated_at, processed_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (name) DO UPDATE SET
                    file_path = EXCLUDED.file_path,
                    file_hash = EXCLUDED.file_hash,
                    total_pages = EXCLUDED.total_pages,
                    total_chunks = EXCLUDED.total_chunks,
                    processing_status = EXCLUDED.processing_status,
                    metadata = EXCLUDED.metadata,
                    updated_at = EXCLUDED.updated_at,
                    processed_at = EXCLUDED.processed_at
            """,
                book["id"], book["name"], book["file_path"], book["file_hash"],
                book["total_pages"], book["total_chunks"], book["processing_status"],
                book["metadata"],
                _parse_datetime(book["created_at"]),
                _parse_datetime(book["updated_at"]),
                _parse_datetime(book["processed_at"])
            )
            books_imported += 1

        # Import chapters
        for chapter in chapters:
            await conn.execute("""
                INSERT INTO chapters (id, book_id, title, chapter_number, start_page, end_page, chunk_count, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title,
                    chapter_number = EXCLUDED.chapter_number,
                    start_page = EXCLUDED.start_page,
                    end_page = EXCLUDED.end_page,
                    chunk_count = EXCLUDED.chunk_count
            """,
                chapter["id"], chapter["book_id"], chapter["title"], chapter["chapter_number"],
                chapter["start_page"], chapter["end_page"], chapter["chunk_count"],
                _parse_datetime(chapter["created_at"])
            )
            chapters_imported += 1

        return {
            "success": True,
            "books_imported": books_imported,
            "chapters_imported": chapters_imported,
        }

    except Exception as e:
        logger.error(f"Failed to import metadata: {e}")
        raise


async def reconstruct_from_qdrant(conn: Connection, qdrant_manager) -> Dict[str, Any]:
    """
    Reconstruct books and chapters from Qdrant payloads (fallback when no metadata JSON).

    Returns reconstruction statistics.
    """
    try:
        import uuid
        from collections import defaultdict

        # Scroll through Qdrant to collect book/chapter info
        books_dict = defaultdict(lambda: {"chapters": set()})
        offset = None
        points_scanned = 0

        while True:
            result = qdrant_manager.client.scroll(
                collection_name=qdrant_manager.collection,
                limit=1000,
                offset=offset,
                with_payload=["book_name", "chapter_title"]
            )

            points, next_offset = result

            for point in points:
                if point.payload:
                    book_name = point.payload.get("book_name", "")
                    chapter_title = point.payload.get("chapter_title", "")

                    if book_name:
                        books_dict[book_name]["chapters"].add(chapter_title)
                        points_scanned += 1

            if next_offset is None:
                break
            offset = next_offset

        # Insert books and chapters
        books_created = 0
        chapters_created = 0

        for book_name, book_data in sorted(books_dict.items()):
            # Create book record
            book_id = str(uuid.uuid4())
            await conn.execute("""
                INSERT INTO books (id, name, processing_status, total_chunks)
                VALUES ($1, $2, 'completed', 0)
                ON CONFLICT (name) DO NOTHING
            """, book_id, book_name)

            # Get the actual book ID (in case of conflict)
            book_row = await conn.fetchrow("SELECT id FROM books WHERE name = $1", book_name)
            book_id = str(book_row["id"])
            books_created += 1

            # Create chapter records
            for idx, chapter_title in enumerate(sorted(book_data["chapters"]), start=1):
                if chapter_title:
                    chapter_id = str(uuid.uuid4())
                    await conn.execute("""
                        INSERT INTO chapters (id, book_id, title, chapter_number)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (id) DO NOTHING
                    """, chapter_id, book_id, chapter_title, idx)
                    chapters_created += 1

        logger.info(f"Reconstructed {books_created} books, {chapters_created} chapters from Qdrant")

        return {
            "success": True,
            "books_created": books_created,
            "chapters_created": chapters_created,
            "points_scanned": points_scanned,
            "warning": "Page numbers and file metadata were not restored (not available in Qdrant)"
        }

    except Exception as e:
        logger.error(f"Failed to reconstruct from Qdrant: {e}")
        raise


def _get_metadata_path(snapshot_name: str, snapshot_dir: str, collection: str) -> Path:
    """Get the path for a snapshot's metadata file."""
    return Path(snapshot_dir) / collection / f"{snapshot_name}.metadata.json"


async def save_metadata_to_file(conn: Connection, snapshot_name: str, snapshot_dir: str, collection: str) -> str:
    """
    Export current metadata and save it alongside the snapshot file.

    Called at snapshot creation time to capture the current state.
    Returns the path of the saved file.
    """
    try:
        metadata = await export_metadata(conn)
        metadata_path = _get_metadata_path(snapshot_name, snapshot_dir, collection)

        # Ensure directory exists
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata to {metadata_path} ({metadata['total_books']} books, {metadata['total_chapters']} chapters)")
        return str(metadata_path)

    except Exception as e:
        logger.error(f"Failed to save metadata file: {e}")
        raise


def load_metadata_from_file(snapshot_name: str, snapshot_dir: str, collection: str) -> Optional[Dict[str, Any]]:
    """
    Load stored metadata for a snapshot from disk.

    Returns None if no metadata file exists.
    """
    try:
        metadata_path = _get_metadata_path(snapshot_name, snapshot_dir, collection)

        if not metadata_path.exists():
            logger.info(f"No metadata file found at {metadata_path}")
            return None

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        logger.info(f"Loaded metadata from {metadata_path}")
        return metadata

    except Exception as e:
        logger.error(f"Failed to load metadata file: {e}")
        return None


def delete_metadata_file(snapshot_name: str, snapshot_dir: str, collection: str) -> bool:
    """Delete the metadata file for a snapshot."""
    try:
        metadata_path = _get_metadata_path(snapshot_name, snapshot_dir, collection)

        if metadata_path.exists():
            metadata_path.unlink()
            logger.info(f"Deleted metadata file: {metadata_path}")
            return True
        return False

    except Exception as e:
        logger.error(f"Failed to delete metadata file: {e}")
        return False
