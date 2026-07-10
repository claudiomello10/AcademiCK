"""PDF upload forwarding to pdf-service, shared by admin and professor routes."""

import json
import os
from typing import Optional
from uuid import UUID

import httpx
from fastapi import HTTPException

PDF_SERVICE_URL = "http://pdf-service:8003"


async def guard_name_collision(
    conn, filename: str, owner_user_id: Optional[str]
) -> None:
    """409 when the derived book name already belongs to someone else.

    Book names are globally unique and Qdrant content is keyed by name, so
    without this a same-named upload would silently overwrite another
    owner's book. Admin uploads (owner None) keep overwrite semantics.
    """
    if owner_user_id is None:
        return

    book_name = os.path.splitext(filename)[0]
    row = await conn.fetchrow(
        "SELECT owner_user_id FROM books WHERE name = $1", book_name
    )
    if row and (row["owner_user_id"] is None
                or str(row["owner_user_id"]) != owner_user_id):
        raise HTTPException(
            status_code=409,
            detail=(
                f"A book named '{book_name}' already exists and belongs to "
                "another owner. Rename the file, or ask an admin to attach "
                "the existing book to your class."
            ),
        )


async def forward_pdfs(
    files: list,
    db_pool,
    uploaded_by: str,
    class_id: Optional[str] = None,
    owner_user_id: Optional[str] = None,
) -> dict:
    """Forward uploaded PDFs to pdf-service and record processing jobs.

    class_id/owner_user_id bind the resulting book to a class and owner
    (professor uploads); admin uploads pass None for global books.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    jobs = []
    errors = []

    async with httpx.AsyncClient(timeout=60.0) as client:
        for file in files:
            if not (hasattr(file, "filename") and hasattr(file, "read")):
                continue
            try:
                async with db_pool.acquire() as conn:
                    await guard_name_collision(conn, file.filename, owner_user_id)

                content = await file.read()

                form_data = {}
                if class_id:
                    form_data["class_id"] = class_id
                if owner_user_id:
                    form_data["owner_user_id"] = owner_user_id

                response = await client.post(
                    f"{PDF_SERVICE_URL}/upload",
                    files={"file": (file.filename, content, "application/pdf")},
                    data=form_data,
                )

                if response.status_code == 200:
                    result = response.json()
                    celery_task_id = result.get("job_id")

                    # Persist job to PostgreSQL for cross-session visibility
                    async with db_pool.acquire() as conn:
                        pg_job_id = await conn.fetchval("""
                            INSERT INTO processing_jobs
                                (job_type, status, progress, metadata, created_at)
                            VALUES ('pdf_processing', 'pending', 0, $1, NOW())
                            RETURNING id
                        """, json.dumps({
                            "celery_task_id": celery_task_id,
                            "filename": file.filename,
                            "uploaded_by": uploaded_by,
                            "class_id": class_id,
                        }))

                    jobs.append({
                        "filename": file.filename,
                        "job_id": str(pg_job_id),
                        "celery_task_id": celery_task_id,
                        "status": "pending"
                    })
                else:
                    errors.append({
                        "filename": file.filename,
                        "error": response.text
                    })
            except HTTPException as e:
                errors.append({
                    "filename": file.filename,
                    "error": e.detail,
                    "status_code": e.status_code,
                })
            except Exception as e:
                errors.append({
                    "filename": file.filename if hasattr(file, "filename") else "unknown",
                    "error": str(e)
                })

    return {
        "message": f"Submitted {len(jobs)} PDF(s) for processing",
        "jobs": jobs,
        "errors": errors,
        "books_processed": len(jobs),
        "total_chunks": 0
    }


async def sync_job_status(db_pool, qdrant, job_id: str) -> dict:
    """Job status from PostgreSQL, refreshed from Celery while in progress."""
    async with db_pool.acquire() as conn:
        job, metadata = await get_job(conn, job_id)

        celery_task_id = metadata.get("celery_task_id")
        filename = metadata.get("filename", "unknown")

        if job["status"] in ("pending", "processing") and celery_task_id:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(
                        f"{PDF_SERVICE_URL}/job/{celery_task_id}"
                    )

                    if response.status_code == 200:
                        celery_data = response.json()
                        new_status = celery_data.get("status", job["status"])
                        new_progress = celery_data.get("progress", job["progress"])

                        celery_result = celery_data.get("result", {}) or {}
                        stage = None
                        chapters_total = 0
                        chapters_processed = 0
                        warning = None

                        if isinstance(celery_result, dict):
                            stage = celery_result.get("stage")
                            chapters_total = celery_result.get("chapters_total", 0)
                            chapters_processed = celery_result.get("chapters_processed", 0)
                            warning = celery_result.get("warning")

                        error = celery_data.get("error")

                        if new_status != job["status"] or new_progress != job["progress"]:
                            if new_status == "completed":
                                await conn.execute("""
                                    UPDATE processing_jobs
                                    SET status = $1, progress = $2, completed_at = NOW()
                                    WHERE id = $3
                                """, new_status, new_progress, job["id"])
                                # New content just landed in Qdrant
                                qdrant.invalidate_catalog_cache()
                            elif new_status == "failed":
                                await conn.execute("""
                                    UPDATE processing_jobs
                                    SET status = $1, progress = $2, error_message = $3, completed_at = NOW()
                                    WHERE id = $4
                                """, new_status, new_progress, error, job["id"])
                            else:
                                await conn.execute("""
                                    UPDATE processing_jobs
                                    SET status = $1, progress = $2
                                    WHERE id = $3
                                """, new_status, new_progress, job["id"])

                        return {
                            "job_id": str(job["id"]),
                            "status": new_status,
                            "progress": new_progress,
                            "filename": filename,
                            "stage": stage,
                            "chapters_total": chapters_total,
                            "chapters_processed": chapters_processed,
                            "warning": warning,
                            "error": error
                        }
            except Exception:
                # If the Celery query fails, fall back to PostgreSQL data
                pass

        return {
            "job_id": str(job["id"]),
            "status": job["status"],
            "progress": job["progress"] or 0,
            "filename": filename,
            "chapters_total": 0,
            "chapters_processed": 0,
            "warning": None,
            "error": job["error_message"]
        }


async def get_job(conn, job_id: str):
    """Load a processing job row with parsed metadata; 404 when missing."""
    try:
        job_uuid = UUID(job_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=404, detail="Job not found")

    job = await conn.fetchrow("""
        SELECT id, status, progress, error_message, metadata, completed_at
        FROM processing_jobs
        WHERE id = $1
    """, job_uuid)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    metadata = job["metadata"] or {}
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return job, metadata
