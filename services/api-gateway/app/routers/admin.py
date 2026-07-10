"""Admin endpoints."""

from fastapi import APIRouter, Depends, Request, HTTPException, UploadFile, File, Form
from typing import List, Optional
from datetime import datetime, timedelta, timezone
import logging

from app.dependencies import get_admin_session
from app.models.schemas import (
    UserCreate, UserUpdate, UserResponse, UserImportResponse,
    ContentStats, UsageStats, ProcessingJobResponse, VALID_ROLES
)
from app.services import class_service, pdf_upload, user_service, user_import
from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


# ===========================================
# User Management
# ===========================================

@router.get("/users", response_model=List[UserResponse])
async def list_users(request: Request, session: dict = Depends(get_admin_session)):
    """List all users (admin only)."""

    async with request.app.state.db_pool.acquire() as conn:
        return await user_service.list_users(conn)


@router.post("/users", response_model=UserResponse)
async def create_user(request: Request, user: UserCreate, session: dict = Depends(get_admin_session)):
    """Create a new user (admin only)."""

    async with request.app.state.db_pool.acquire() as conn:
        return await user_service.create_user(conn, user)


@router.post("/users/import", response_model=UserImportResponse)
async def import_users(
    request: Request,
    file: UploadFile = File(...),
    default_password: Optional[str] = Form(None),
    session: dict = Depends(get_admin_session),
):
    """Bulk-import users from a file (admin only). Reports errors per row."""
    content = await file.read()
    return await user_import.import_users(
        request.app.state.db_pool,
        file.filename,
        content,
        allowed_roles=VALID_ROLES,
        default_password=default_password,
    )


@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    request: Request,
    user_id: str,
    updates: UserUpdate,
    session: dict = Depends(get_admin_session)
):
    """Update a user (admin only)."""

    async with request.app.state.db_pool.acquire() as conn:
        return await user_service.update_user(conn, user_id, updates)


# ===========================================
# User Status Update (separate endpoint)
# ===========================================

@router.put("/users/{user_id}/status")
async def update_user_status(
    request: Request,
    user_id: str,
    status_update: dict,
    session: dict = Depends(get_admin_session)
):
    """Update a user's status (admin only)."""

    new_status = status_update.get("status")
    if new_status not in ["active", "inactive"]:
        raise HTTPException(status_code=400, detail="Invalid status. Use 'active' or 'inactive'")

    async with request.app.state.db_pool.acquire() as conn:
        row = await conn.fetchrow("""
            UPDATE users
            SET status = $1, updated_at = CURRENT_TIMESTAMP
            WHERE id = $2
            RETURNING id, username, status
        """, new_status, user_id)

        if not row:
            raise HTTPException(status_code=404, detail="User not found")

        return {
            "id": str(row["id"]),
            "username": row["username"],
            "status": row["status"],
            "message": f"User status updated to {new_status}"
        }


# ===========================================
# Statistics
# ===========================================

@router.get("/stats/content", response_model=ContentStats)
async def get_content_stats(request: Request, session: dict = Depends(get_admin_session)):
    """Get content statistics (admin only)."""

    async with request.app.state.db_pool.acquire() as conn:
        stats = await conn.fetchrow("""
            SELECT
                (SELECT COUNT(*) FROM books WHERE processing_status = 'completed') as total_books,
                (SELECT COUNT(*) FROM chapters) as total_chapters,
                (SELECT COUNT(*) FROM chunks) as total_chunks,
                (SELECT COUNT(*) FROM users) as total_users,
                (SELECT COUNT(*) FROM messages) as total_messages
        """)

        return ContentStats(
            total_books=stats["total_books"],
            total_chapters=stats["total_chapters"],
            total_chunks=stats["total_chunks"],
            total_users=stats["total_users"],
            total_messages=stats["total_messages"]
        )


@router.get("/stats/chunks")
async def get_chunk_retrieval_stats(request: Request, range: str = "7d", session: dict = Depends(get_admin_session)):
    """Get chunk retrieval statistics - which content is most accessed (admin only).

    Provides analytics on which books and chapters are most frequently retrieved
    in RAG search results.

    Query params:
        range: "7d" (default), "30d", or "all"
    """

    # Calculate date range
    if range == "7d":
        days = 7
    elif range == "30d":
        days = 30
    else:
        days = 365  # "all" - last year

    async with request.app.state.db_pool.acquire() as conn:
        # Most retrieved books
        books_stats = await conn.fetch("""
            SELECT
                b.name as book_name,
                COUNT(*) as retrieval_count,
                COUNT(DISTINCT cr.chunk_id) as unique_chunks,
                AVG(cr.score) as avg_score
            FROM chunk_retrievals cr
            JOIN books b ON cr.book_id = b.id
            WHERE cr.created_at > NOW() - INTERVAL '%s days'
            GROUP BY b.id, b.name
            ORDER BY retrieval_count DESC
            LIMIT 20
        """ % days)

        # Most retrieved chapters
        chapters_stats = await conn.fetch("""
            SELECT
                b.name as book_name,
                c.title as chapter_title,
                COUNT(*) as retrieval_count,
                COUNT(DISTINCT cr.chunk_id) as unique_chunks,
                AVG(cr.score) as avg_score
            FROM chunk_retrievals cr
            JOIN chapters c ON cr.chapter_id = c.id
            JOIN books b ON cr.book_id = b.id
            WHERE cr.created_at > NOW() - INTERVAL '%s days'
            GROUP BY b.id, b.name, c.id, c.title
            ORDER BY retrieval_count DESC
            LIMIT 30
        """ % days)

        # Most frequently retrieved specific chunks
        top_chunks = await conn.fetch("""
            SELECT
                ch.id as chunk_id,
                SUBSTRING(ch.text, 1, 200) as text_preview,
                ch.topic,
                b.name as book_name,
                c.title as chapter_title,
                COUNT(*) as times_retrieved,
                AVG(cr.score) as avg_score,
                AVG(cr.position) as avg_position
            FROM chunk_retrievals cr
            JOIN chunks ch ON cr.chunk_id = ch.id
            JOIN books b ON cr.book_id = b.id
            LEFT JOIN chapters c ON cr.chapter_id = c.id
            WHERE cr.created_at > NOW() - INTERVAL '%s days'
            GROUP BY ch.id, ch.text, ch.topic, b.id, b.name, c.id, c.title
            ORDER BY times_retrieved DESC
            LIMIT 50
        """ % days)

        # Retrieval trends over time (daily counts)
        daily_trends = await conn.fetch("""
            SELECT
                DATE(created_at) as date,
                COUNT(*) as retrieval_count,
                COUNT(DISTINCT message_id) as unique_queries
            FROM chunk_retrievals
            WHERE created_at > NOW() - INTERVAL '%s days'
            GROUP BY DATE(created_at)
            ORDER BY date DESC
        """ % days)

        # Total stats
        total_retrievals = await conn.fetchval("""
            SELECT COUNT(*) FROM chunk_retrievals
            WHERE created_at > NOW() - INTERVAL '%s days'
        """ % days) or 0

        total_unique_chunks = await conn.fetchval("""
            SELECT COUNT(DISTINCT chunk_id) FROM chunk_retrievals
            WHERE created_at > NOW() - INTERVAL '%s days'
        """ % days) or 0

        return {
            "time_range": range,
            "total_retrievals": total_retrievals,
            "total_unique_chunks": total_unique_chunks,
            "by_book": [
                {
                    "book_name": row["book_name"],
                    "retrieval_count": row["retrieval_count"],
                    "unique_chunks": row["unique_chunks"],
                    "avg_score": float(row["avg_score"]) if row["avg_score"] else 0
                }
                for row in books_stats
            ],
            "by_chapter": [
                {
                    "book_name": row["book_name"],
                    "chapter_title": row["chapter_title"],
                    "retrieval_count": row["retrieval_count"],
                    "unique_chunks": row["unique_chunks"],
                    "avg_score": float(row["avg_score"]) if row["avg_score"] else 0
                }
                for row in chapters_stats
            ],
            "top_chunks": [
                {
                    "chunk_id": str(row["chunk_id"]),
                    "text_preview": row["text_preview"] + "..." if row["text_preview"] and len(row["text_preview"]) == 200 else row["text_preview"],
                    "topic": row["topic"],
                    "book_name": row["book_name"],
                    "chapter_title": row["chapter_title"],
                    "times_retrieved": row["times_retrieved"],
                    "avg_score": float(row["avg_score"]) if row["avg_score"] else 0,
                    "avg_position": float(row["avg_position"]) if row["avg_position"] else 0
                }
                for row in top_chunks
            ],
            "daily_trends": [
                {
                    "date": row["date"].isoformat() if row["date"] else None,
                    "retrieval_count": row["retrieval_count"],
                    "unique_queries": row["unique_queries"]
                }
                for row in daily_trends
            ]
        }


@router.get("/stats/usage", response_model=UsageStats)
async def get_usage_stats(request: Request, session: dict = Depends(get_admin_session)):
    """Get usage statistics (admin only)."""

    today = datetime.now(timezone.utc).date()

    async with request.app.state.db_pool.acquire() as conn:
        # Total queries
        total_queries = await conn.fetchval(
            "SELECT COUNT(*) FROM usage_stats WHERE action_type = 'query'"
        ) or 0

        # Queries today
        queries_today = await conn.fetchval("""
            SELECT COUNT(*) FROM usage_stats
            WHERE action_type = 'query' AND DATE(created_at) = $1
        """, today) or 0

        # Average response time
        avg_time = await conn.fetchval("""
            SELECT AVG(response_time_ms) FROM usage_stats
            WHERE action_type = 'query' AND response_time_ms IS NOT NULL
        """) or 0

        # Queries by intent
        intent_stats = await conn.fetch("""
            SELECT intent, COUNT(*) as count
            FROM usage_stats
            WHERE action_type = 'query' AND intent IS NOT NULL
            GROUP BY intent
        """)
        queries_by_intent = {row["intent"]: row["count"] for row in intent_stats}

        # Active users today
        active_users = await conn.fetchval("""
            SELECT COUNT(DISTINCT user_id) FROM usage_stats
            WHERE DATE(created_at) = $1
        """, today) or 0

        return UsageStats(
            total_queries=total_queries,
            queries_today=queries_today,
            average_response_time_ms=float(avg_time),
            queries_by_intent=queries_by_intent,
            active_users_today=active_users
        )


# ===========================================
# Processing Jobs
# ===========================================

@router.get("/jobs")
async def list_jobs(request: Request, session: dict = Depends(get_admin_session)):
    """List recent processing jobs (admin only).

    Returns jobs from the last 12 hours, active (pending/processing) first so
    a batch upload can't push a running job out of the list — the Celery sync
    below only covers returned jobs. Excludes dismissed jobs.
    """
    import httpx
    import json as json_lib


    async with request.app.state.db_pool.acquire() as conn:
        jobs = await conn.fetch("""
            SELECT
                j.id, j.job_type, j.status, j.progress, j.error_message,
                j.created_at, j.completed_at, j.metadata, b.name as book_name
            FROM processing_jobs j
            LEFT JOIN books b ON j.book_id = b.id
            WHERE j.created_at > NOW() - INTERVAL '12 hours'
              AND (j.metadata->>'dismissed' IS NULL OR j.metadata->>'dismissed' != 'true')
            ORDER BY (j.status IN ('pending', 'processing')) DESC, j.created_at ASC
            LIMIT 50
        """)

        result = []
        async with httpx.AsyncClient(timeout=5.0) as client:
            for j in jobs:
                metadata = j["metadata"] if j["metadata"] else {}
                if isinstance(metadata, str):
                    metadata = json_lib.loads(metadata)

                filename = metadata.get("filename", j["book_name"] or "unknown")
                celery_task_id = metadata.get("celery_task_id")
                status = j["status"]
                progress = j["progress"] or 0
                error = j["error_message"]
                stage = None
                chapters_total = 0
                chapters_processed = 0
                warning = None

                # Sync with Celery for in-progress jobs
                if status in ("pending", "processing") and celery_task_id:
                    try:
                        response = await client.get(f"http://pdf-service:8003/job/{celery_task_id}")
                        if response.status_code == 200:
                            celery_data = response.json()
                            status = celery_data.get("status", status)
                            progress = celery_data.get("progress", progress)

                            # Extract meta info from Celery result
                            celery_result = celery_data.get("result", {}) or {}
                            if isinstance(celery_result, dict):
                                stage = celery_result.get("stage")
                                chapters_total = celery_result.get("chapters_total", 0)
                                chapters_processed = celery_result.get("chapters_processed", 0)
                                warning = celery_result.get("warning")
                            error = celery_data.get("error") or error

                            # Update PostgreSQL if status changed
                            if status != j["status"] or progress != (j["progress"] or 0):
                                if status == "completed":
                                    await conn.execute("""
                                        UPDATE processing_jobs
                                        SET status = $1, progress = $2, completed_at = NOW()
                                        WHERE id = $3
                                    """, status, progress, j["id"])
                                    # New content just landed in Qdrant
                                    request.app.state.qdrant.invalidate_catalog_cache()
                                elif status == "failed":
                                    await conn.execute("""
                                        UPDATE processing_jobs
                                        SET status = $1, progress = $2, error_message = $3, completed_at = NOW()
                                        WHERE id = $4
                                    """, status, progress, error, j["id"])
                                else:
                                    await conn.execute("""
                                        UPDATE processing_jobs
                                        SET status = $1, progress = $2
                                        WHERE id = $3
                                    """, status, progress, j["id"])
                    except Exception:
                        pass  # Use PostgreSQL data if Celery fails

                result.append({
                    "job_id": str(j["id"]),
                    "job_type": j["job_type"],
                    "status": status,
                    "progress": progress,
                    "filename": filename,
                    "stage": stage,
                    "chapters_total": chapters_total,
                    "chapters_processed": chapters_processed,
                    "warning": warning,
                    "error": error,
                    "created_at": j["created_at"].isoformat() if j["created_at"] else None,
                    "completed_at": j["completed_at"].isoformat() if j["completed_at"] else None
                })

        return result


@router.get("/jobs/{job_id}", response_model=ProcessingJobResponse)
async def get_job(request: Request, job_id: str, session: dict = Depends(get_admin_session)):
    """Get status of a specific job (admin only)."""

    async with request.app.state.db_pool.acquire() as conn:
        job = await conn.fetchrow("""
            SELECT
                j.id, j.job_type, j.status, j.progress, j.error_message,
                j.created_at, j.completed_at, b.name as book_name
            FROM processing_jobs j
            LEFT JOIN books b ON j.book_id = b.id
            WHERE j.id = $1
        """, job_id)

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        return ProcessingJobResponse(
            job_id=str(job["id"]),
            job_type=job["job_type"],
            status=job["status"],
            progress=job["progress"],
            book_name=job["book_name"],
            error_message=job["error_message"],
            created_at=job["created_at"],
            completed_at=job["completed_at"]
        )


# ===========================================
# System Info
# ===========================================

@router.get("/system/info")
async def get_system_info(request: Request, session: dict = Depends(get_admin_session)):
    """Get system information (admin only)."""

    # Get Qdrant info
    qdrant_info = await request.app.state.qdrant.get_collection_info()

    # Get active sessions count
    active_sessions = await request.app.state.session_service.get_active_sessions_count()

    return {
        "qdrant": qdrant_info,
        "active_sessions": active_sessions,
        "services": {
            "intent": await request.app.state.intent_client.health_check(),
            "embedding": await request.app.state.embedding_client.health_check()
        }
    }


# ===========================================
# Frontend-compatible endpoint aliases
# ===========================================

@router.get("/content-stats")
async def get_content_stats_alias(request: Request, session: dict = Depends(get_admin_session)):
    """Get content statistics - frontend compatible endpoint."""

    # Get stats from Qdrant since PostgreSQL tables might be empty
    qdrant_info = await request.app.state.qdrant.get_collection_info()
    books = await request.app.state.qdrant.get_books()

    # Handle vectors_count - it might be a dict for named vectors collections
    # or an int for simple collections. For hybrid collections (dense + sparse),
    # use points_count as the embeddings count since each point has one embedding set.
    vectors_count = qdrant_info.get("vectors_count", 0)
    if isinstance(vectors_count, dict):
        # Sum all named vectors counts, or just use points_count
        vectors_count = qdrant_info.get("points_count", 0)
    elif vectors_count is None:
        vectors_count = qdrant_info.get("points_count", 0)

    return {
        "total_books": len(books),
        "total_chunks": qdrant_info.get("points_count", 0),
        "total_embeddings": vectors_count
    }


@router.get("/book-list")
async def get_book_list(request: Request, session: dict = Depends(get_admin_session)):
    """Get list of books with stats from PostgreSQL."""

    async with request.app.state.db_pool.acquire() as conn:
        books = await conn.fetch("""
            SELECT
                b.name,
                COUNT(DISTINCT c.id) as total_chapters,
                COALESCE(b.total_chunks, 0) as total_chunks,
                b.processing_status,
                b.processing_method
            FROM books b
            LEFT JOIN chapters c ON b.id = c.book_id
            GROUP BY b.id, b.name, b.total_chunks, b.processing_status, b.processing_method
            ORDER BY b.name
        """)

        return [
            {
                "name": book["name"],
                "total_chapters": book["total_chapters"],
                "total_chunks": book["total_chunks"],
                "processing_status": book["processing_status"],
                "processing_method": book["processing_method"]
            }
            for book in books
        ]


@router.get("/usage-stats")
async def get_usage_stats_alias(request: Request, range: str = "7d", session: dict = Depends(get_admin_session)):
    """Get usage statistics - frontend compatible endpoint."""

    today = datetime.now(timezone.utc).date()

    # Calculate date range
    if range == "7d":
        start_date = today - timedelta(days=7)
    elif range == "30d":
        start_date = today - timedelta(days=30)
    else:  # "all"
        start_date = None

    async with request.app.state.db_pool.acquire() as conn:
        # Build query based on date range
        if start_date:
            total_queries = await conn.fetchval("""
                SELECT COUNT(*) FROM usage_stats
                WHERE action_type = 'query' AND DATE(created_at) >= $1
            """, start_date) or 0

            active_users = await conn.fetchval("""
                SELECT COUNT(DISTINCT user_id) FROM usage_stats
                WHERE DATE(created_at) >= $1
            """, start_date) or 0
        else:
            total_queries = await conn.fetchval(
                "SELECT COUNT(*) FROM usage_stats WHERE action_type = 'query'"
            ) or 0

            active_users = await conn.fetchval(
                "SELECT COUNT(DISTINCT user_id) FROM usage_stats"
            ) or 0

        # Average response time
        avg_time = await conn.fetchval("""
            SELECT AVG(response_time_ms) FROM usage_stats
            WHERE action_type = 'query' AND response_time_ms IS NOT NULL
        """) or 0

        return {
            "total_queries": total_queries,
            "avg_response_time": round(float(avg_time) / 1000, 2) if avg_time else 0,  # Convert to seconds
            "active_users": active_users,
            "content_access": total_queries,  # Alias for compatibility
            "time_range": range
        }


@router.delete("/books/{book_name}")
async def delete_book(request: Request, book_name: str, session: dict = Depends(get_admin_session)):
    """Delete a book and all its embeddings (admin only)."""

    try:
        # Delete from Qdrant
        deleted_count = await request.app.state.qdrant.delete_by_book_name(book_name)

        # Also try to delete from PostgreSQL if the book exists there
        pg_deleted = False
        async with request.app.state.db_pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM books WHERE name = $1
            """, book_name)
            pg_deleted = result != "DELETE 0"

        # Cached search results may still cite the deleted book
        await class_service.flush_search_cache(request.app.state.redis)

        return {
            "success": True,
            "book_name": book_name,
            "vectors_deleted": deleted_count,
            "pg_deleted": pg_deleted,
            "message": f"Successfully deleted book '{book_name}' with {deleted_count} embeddings"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete book: {str(e)}")


@router.post("/upload-pdfs")
async def upload_pdfs(request: Request, session: dict = Depends(get_admin_session)):
    """Upload PDFs for processing via pdf-service.

    An optional `class_id` form field attaches the resulting books to a
    class at ingest; books stay global (no owner) either way.
    """
    form = await request.form()
    files = form.getlist("files")
    class_id = form.get("class_id") or None

    return await pdf_upload.forward_pdfs(
        files,
        request.app.state.db_pool,
        uploaded_by=session["user_id"],
        class_id=class_id,
    )


@router.get("/pdf-job/{job_id}")
async def get_pdf_job_status(request: Request, job_id: str, session: dict = Depends(get_admin_session)):
    """Get PDF processing job status - syncs PostgreSQL with Celery."""
    return await pdf_upload.sync_job_status(
        request.app.state.db_pool, request.app.state.qdrant, job_id
    )


@router.delete("/jobs/{job_id}")
async def dismiss_job(request: Request, job_id: str, session: dict = Depends(get_admin_session)):
    """Manually dismiss/remove a job from the list (admin only).

    Jobs are not deleted, just marked as dismissed so they won't appear in the list.
    """

    async with request.app.state.db_pool.acquire() as conn:
        # Check if job exists
        job = await conn.fetchrow(
            "SELECT id, metadata FROM processing_jobs WHERE id = $1",
            job_id
        )

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # Mark as dismissed by updating metadata JSONB
        import json as json_lib

        metadata = job["metadata"] if job["metadata"] else {}
        if isinstance(metadata, str):
            metadata = json_lib.loads(metadata)

        metadata["dismissed"] = True

        await conn.execute("""
            UPDATE processing_jobs
            SET metadata = $1
            WHERE id = $2
        """, json_lib.dumps(metadata), job_id)

    return {"success": True, "message": "Job dismissed"}


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(request: Request, job_id: str, session: dict = Depends(get_admin_session)):
    """Cancel an in-progress processing job (admin only).

    Revokes the Celery task and marks the job as cancelled.
    """
    import httpx
    import json as json_lib


    async with request.app.state.db_pool.acquire() as conn:
        # Check if job exists and is in progress
        job = await conn.fetchrow(
            "SELECT id, status, metadata FROM processing_jobs WHERE id = $1",
            job_id
        )

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        if job["status"] not in ("pending", "processing"):
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel job with status '{job['status']}'"
            )

        metadata = job["metadata"] if job["metadata"] else {}
        if isinstance(metadata, str):
            metadata = json_lib.loads(metadata)

        celery_task_id = metadata.get("celery_task_id")

        # Try to revoke the Celery task
        if celery_task_id:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        f"http://pdf-service:8003/job/{celery_task_id}/cancel"
                    )
                    if response.status_code != 200:
                        logger.warning(f"Failed to revoke Celery task: {response.text}")
            except Exception as e:
                logger.warning(f"Error revoking Celery task: {e}")

        # Mark job as cancelled in PostgreSQL
        await conn.execute("""
            UPDATE processing_jobs
            SET status = 'cancelled', completed_at = NOW()
            WHERE id = $1
        """, job_id)

        # The worker creates the book row (named after the file) when the task
        # starts; cancelling mid-run would otherwise leave it stuck in
        # 'processing'. Jobs carry no book_id, so resolve the book by name.
        filename = metadata.get("filename")
        if filename:
            import os
            await conn.execute("""
                UPDATE books
                SET processing_status = 'cancelled', updated_at = NOW()
                WHERE name = $1 AND processing_status = 'processing'
            """, os.path.splitext(filename)[0])

    return {"success": True, "message": "Job cancelled"}


@router.post("/upload-embedding-file")
async def upload_embedding_file(request: Request, session: dict = Depends(get_admin_session)):
    """
    Upload a JSON embedding file and import into Qdrant.

    Expected JSON format:
    {
        "embeddings": [
            {
                "text": "chunk text",
                "book_name": "Book Title",
                "chapter_title": "Chapter 1",
                "topic": "Topic Name",
                "is_introduction": false,
                "dense_embedding": [0.1, 0.2, ...],
                "sparse_embedding": {"indices": [1, 5, 10], "values": [0.5, 0.3, 0.8]}
            },
            ...
        ]
    }
    """
    import json
    from uuid import uuid4
    from qdrant_client.models import PointStruct, SparseVector


    # Get the form data from the request
    form = await request.form()
    file = form.get("file")

    if not file or not hasattr(file, 'read'):
        raise HTTPException(status_code=400, detail="No file provided")

    if not file.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="Only JSON files are supported")

    try:
        # Read and parse JSON
        content = await file.read()
        data = json.loads(content.decode('utf-8'))

        embeddings = data.get("embeddings", [])
        if not embeddings:
            raise HTTPException(status_code=400, detail="No embeddings found in file")

        # Prepare points for Qdrant
        points = []
        for idx, emb in enumerate(embeddings):
            point_id = str(uuid4())

            # Build vector dict
            vectors = {}
            if "dense_embedding" in emb:
                vectors["dense"] = emb["dense_embedding"]

            if "sparse_embedding" in emb:
                sparse = emb["sparse_embedding"]
                vectors["sparse"] = SparseVector(
                    indices=sparse.get("indices", []),
                    values=sparse.get("values", [])
                )

            if not vectors:
                continue

            # Build payload
            payload = {
                "text": emb.get("text", ""),
                "book_name": emb.get("book_name", "Unknown"),
                "chapter_title": emb.get("chapter_title", ""),
                "topic": emb.get("topic", ""),
                "is_introduction": emb.get("is_introduction", False),
                "chunk_id": point_id
            }

            points.append(PointStruct(
                id=point_id,
                vector=vectors,
                payload=payload
            ))

        if not points:
            raise HTTPException(status_code=400, detail="No valid embeddings to import")

        # Upsert to Qdrant
        qdrant = request.app.state.qdrant
        await qdrant.client.upsert(
            collection_name=qdrant.collection,
            points=points
        )
        qdrant.invalidate_catalog_cache()

        return {
            "success": True,
            "message": f"Successfully imported {len(points)} embeddings",
            "count": len(points)
        }

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to import embeddings: {str(e)}")


# =====================================================================
# Qdrant Snapshot Management Endpoints
# =====================================================================

@router.post("/snapshots/create")
async def create_snapshot(request: Request, session: dict = Depends(get_admin_session)):
    """Create a new Qdrant snapshot and store metadata alongside it."""
    if not settings.enable_snapshot_management:
        raise HTTPException(status_code=403, detail="Snapshot management is disabled")


    try:
        from app.utils.snapshot_helpers import save_metadata_to_file

        result = await request.app.state.qdrant.create_snapshot()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # A snapshot without metadata cannot be restored, so a failed metadata
    # write must not leave the snapshot behind.
    try:
        async with request.app.state.db_pool.acquire() as conn:
            await save_metadata_to_file(
                conn,
                result["snapshot_name"],
                settings.snapshot_dir,
                settings.qdrant_collection
            )
    except Exception as e:
        try:
            await request.app.state.qdrant.delete_snapshot(result["snapshot_name"])
        except Exception as cleanup_error:
            logger.error(f"Failed to delete orphaned snapshot: {cleanup_error}")
        raise HTTPException(
            status_code=500,
            detail=(
                f"Snapshot metadata could not be written to "
                f"'{settings.snapshot_dir}' ({e}). The snapshot was discarded "
                f"because it would not be restorable without metadata."
            )
        )

    return result


@router.get("/snapshots")
async def list_snapshots(request: Request, session: dict = Depends(get_admin_session)):
    """List all available Qdrant snapshots with metadata availability."""
    if not settings.enable_snapshot_management:
        raise HTTPException(status_code=403, detail="Snapshot management is disabled")


    try:
        from app.utils.snapshot_helpers import load_metadata_from_file

        snapshots = await request.app.state.qdrant.list_snapshots()

        # Check which snapshots have stored metadata
        for snap in snapshots:
            metadata = load_metadata_from_file(
                snap["name"], settings.snapshot_dir, settings.qdrant_collection
            )
            snap["has_metadata"] = metadata is not None
            if metadata:
                snap["metadata_books"] = metadata.get("total_books", 0)
                snap["metadata_chapters"] = metadata.get("total_chapters", 0)

        return {"snapshots": snapshots, "count": len(snapshots)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/snapshots/{snapshot_name}/restore")
async def restore_snapshot(request: Request, snapshot_name: str, session: dict = Depends(get_admin_session)):
    """Restore Qdrant snapshot using its stored metadata."""
    if not settings.enable_snapshot_management:
        raise HTTPException(status_code=403, detail="Snapshot management is disabled")


    try:
        from app.utils.snapshot_helpers import load_metadata_from_file, import_metadata

        # Check that stored metadata exists
        metadata = load_metadata_from_file(
            snapshot_name, settings.snapshot_dir, settings.qdrant_collection
        )

        if metadata is None:
            raise HTTPException(
                status_code=400,
                detail="No metadata found for this snapshot. Cannot restore without metadata."
            )

        # Restore Qdrant snapshot
        await request.app.state.qdrant.restore_snapshot(snapshot_name)
        request.app.state.qdrant.invalidate_catalog_cache()

        # Import stored metadata
        async with request.app.state.db_pool.acquire() as conn:
            import_result = await import_metadata(conn, metadata)

        return {
            "success": True,
            "message": "Restored from snapshot with metadata",
            "books_imported": import_result.get("books_imported"),
            "chapters_imported": import_result.get("chapters_imported")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to restore snapshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/snapshots/{snapshot_name}")
async def delete_snapshot(request: Request, snapshot_name: str, session: dict = Depends(get_admin_session)):
    """Delete a Qdrant snapshot and its metadata file."""
    if not settings.enable_snapshot_management:
        raise HTTPException(status_code=403, detail="Snapshot management is disabled")


    try:
        from app.utils.snapshot_helpers import delete_metadata_file

        result = await request.app.state.qdrant.delete_snapshot(snapshot_name)

        # Also clean up metadata file
        delete_metadata_file(snapshot_name, settings.snapshot_dir, settings.qdrant_collection)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/snapshots/{snapshot_name}/download")
async def download_snapshot(request: Request, snapshot_name: str, session: dict = Depends(get_admin_session)):
    """Download Qdrant snapshot file via proxy."""
    if not settings.enable_snapshot_management:
        raise HTTPException(status_code=403, detail="Snapshot management is disabled")


    try:
        import httpx
        from fastapi.responses import StreamingResponse

        qdrant_url = request.app.state.qdrant.get_snapshot_url(snapshot_name)

        async with httpx.AsyncClient() as client:
            response = await client.get(qdrant_url, timeout=60.0)
            response.raise_for_status()

            return StreamingResponse(
                iter([response.content]),
                media_type="application/octet-stream",
                headers={"Content-Disposition": f"attachment; filename={snapshot_name}"}
            )
    except Exception as e:
        logger.error(f"Failed to download snapshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/snapshots/{snapshot_name}/metadata")
async def download_snapshot_metadata(request: Request, snapshot_name: str, session: dict = Depends(get_admin_session)):
    """Download the stored metadata JSON for a snapshot."""
    if not settings.enable_snapshot_management:
        raise HTTPException(status_code=403, detail="Snapshot management is disabled")


    try:
        from fastapi.responses import JSONResponse
        from app.utils.snapshot_helpers import load_metadata_from_file

        metadata = load_metadata_from_file(
            snapshot_name, settings.snapshot_dir, settings.qdrant_collection
        )

        if metadata is None:
            raise HTTPException(status_code=404, detail="No metadata found for this snapshot")

        return JSONResponse(
            content=metadata,
            headers={
                "Content-Disposition": f"attachment; filename={snapshot_name}.metadata.json"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/snapshots/upload")
async def upload_snapshot(
    request: Request,
    snapshot_file: UploadFile = File(...),
    metadata_file: UploadFile = File(...),
    session: dict = Depends(get_admin_session)
):
    """Upload an external snapshot and its metadata file."""
    if not settings.enable_snapshot_management:
        raise HTTPException(status_code=403, detail="Snapshot management is disabled")


    try:
        import httpx
        import json
        from pathlib import Path
        from app.utils.snapshot_helpers import import_metadata

        snapshot_content = await snapshot_file.read()
        snapshot_filename = snapshot_file.filename or "uploaded_snapshot.snapshot"

        # Upload snapshot to Qdrant
        qdrant_url = f"http://{request.app.state.qdrant.host}:{request.app.state.qdrant.port}/collections/{request.app.state.qdrant.collection}/snapshots/upload"

        async with httpx.AsyncClient(timeout=120.0) as client:
            upload_response = await client.post(
                qdrant_url,
                files={"snapshot": (snapshot_filename, snapshot_content, "application/octet-stream")}
            )
            upload_response.raise_for_status()

        request.app.state.qdrant.invalidate_catalog_cache()

        # Parse and import metadata into PostgreSQL
        metadata_content = await metadata_file.read()
        metadata = json.loads(metadata_content)

        async with request.app.state.db_pool.acquire() as conn:
            import_result = await import_metadata(conn, metadata)

        # Save metadata file to disk so it's available for future restores.
        # Without it the uploaded snapshot would not be restorable, so drop
        # the snapshot again if the write fails.
        try:
            metadata_dir = Path(settings.snapshot_dir) / settings.qdrant_collection
            metadata_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = metadata_dir / f"{snapshot_filename}.metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            try:
                await request.app.state.qdrant.delete_snapshot(snapshot_filename)
            except Exception as cleanup_error:
                logger.error(f"Failed to delete orphaned snapshot: {cleanup_error}")
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Snapshot metadata could not be written to "
                    f"'{settings.snapshot_dir}' ({e}). The uploaded snapshot "
                    f"was discarded because it would not be restorable."
                )
            )

        return {
            "success": True,
            "message": "Snapshot uploaded with metadata",
            "books_imported": import_result.get("books_imported"),
            "chapters_imported": import_result.get("chapters_imported")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload snapshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features")
async def get_feature_flags(request: Request, session: dict = Depends(get_admin_session)):
    """Get enabled admin features."""

    return {
        "snapshot_management_enabled": settings.enable_snapshot_management,
        "pdf_upload_enabled": settings.enable_pdf_upload
    }
