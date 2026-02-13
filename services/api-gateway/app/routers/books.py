"""Books endpoints."""

from fastapi import APIRouter, Request, HTTPException
from typing import List
import logging

from app.models.schemas import BookInfo, BookListResponse, ChapterInfo

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/books", response_model=BookListResponse)
async def list_books(request: Request):
    """Get list of available books and their chapters from Qdrant."""
    try:
        # Get books with chapters from Qdrant
        books_data = await request.app.state.qdrant.get_books_with_chapters()

        books = []
        total_chunks = 0

        for i, book_data in enumerate(books_data):
            book = BookInfo(
                id=str(i),
                name=book_data["name"],
                total_chunks=0,  # Not tracked in Qdrant payloads
                processing_status="completed",
                chapters=[
                    ChapterInfo(
                        id=str(j),
                        title=ch["title"],
                        chunk_count=0
                    )
                    for j, ch in enumerate(book_data.get("chapters", []))
                ]
            )
            books.append(book)

        return BookListResponse(
            books=books,
            total_books=len(books),
            total_chunks=total_chunks
        )

    except Exception as e:
        logger.error(f"Failed to list books: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/books/{book_id}", response_model=BookInfo)
async def get_book(request: Request, book_id: str):
    """Get details for a specific book."""
    try:
        async with request.app.state.db_pool.acquire() as conn:
            book_data = await conn.fetchrow("""
                SELECT
                    b.id,
                    b.name,
                    b.total_chunks,
                    b.processing_status
                FROM books b
                WHERE b.id = $1
            """, book_id)

            if not book_data:
                raise HTTPException(status_code=404, detail="Book not found")

            chapters_data = await conn.fetch("""
                SELECT id, title, chunk_count
                FROM chapters
                WHERE book_id = $1
                ORDER BY chapter_number
            """, book_id)

            return BookInfo(
                id=str(book_data["id"]),
                name=book_data["name"],
                total_chunks=book_data["total_chunks"] or 0,
                processing_status=book_data["processing_status"],
                chapters=[
                    ChapterInfo(
                        id=str(c["id"]),
                        title=c["title"],
                        chunk_count=c["chunk_count"] or 0
                    )
                    for c in chapters_data
                ]
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/books/names/list")
async def list_book_names(request: Request):
    """Get simple list of book names for filtering."""
    try:
        # Try to get from Qdrant first (faster)
        books = await request.app.state.qdrant.get_books()

        if books:
            return {"books": books}

        # Fallback to database
        async with request.app.state.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT name FROM books
                WHERE processing_status = 'completed'
                ORDER BY name
            """)
            return {"books": [row["name"] for row in rows]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
