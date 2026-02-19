"""
PDF Processing Service

Handles PDF upload, text extraction, chunking, and embedding generation.
Uses Celery for async processing.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import io
import os
import shutil
from uuid import uuid4
import logging
import magic
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.config import settings
from app.workers.celery_app import celery_app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="AcademiCK PDF Processing Service",
    description="PDF processing and embedding generation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProcessRequest(BaseModel):
    """Request to process a PDF file."""
    file_path: str
    book_name: str


class ProcessResponse(BaseModel):
    """Response with job ID."""
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    """Job status response."""
    job_id: str
    status: str
    progress: float
    result: Optional[dict] = None
    error: Optional[str] = None


@app.post("/process", response_model=ProcessResponse)
async def process_pdf(request: ProcessRequest):
    """
    Start processing a PDF file.

    Returns a job ID for tracking progress.
    """
    # Verify file exists
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Start Celery task
    from app.workers.tasks import process_pdf_task

    task = process_pdf_task.delay(
        file_path=request.file_path,
        book_name=request.book_name
    )

    return ProcessResponse(
        job_id=task.id,
        status="queued",
        message=f"Processing started for {request.book_name}"
    )


@app.post("/upload", response_model=ProcessResponse)
async def upload_and_process(
    file: UploadFile = File(...),
    book_name: Optional[str] = None
):
    """
    Upload a PDF file and start processing.

    Validates file extension, size, and MIME type before processing.
    """
    # Check file extension and capture filename
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    original_filename = file.filename

    # Read file content, then close the upload handle
    max_size = settings.max_upload_size_mb * 1024 * 1024
    contents = await file.read()
    await file.close()
    if len(contents) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_upload_size_mb}MB"
        )

    # Validate MIME type using magic bytes
    mime_type = magic.from_buffer(contents[:2048], mime=True)
    if mime_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {mime_type}. Only PDF files are supported"
        )

    # Validate PDF structure — catches files with valid magic bytes but invalid content
    try:
        reader = PdfReader(io.BytesIO(contents))
        if len(reader.pages) == 0:
            raise HTTPException(
                status_code=400,
                detail="Invalid PDF: file contains no pages"
            )
    except HTTPException:
        raise
    except PdfReadError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid PDF structure: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not verify PDF integrity: {e}"
        )

    # Generate unique filename
    file_id = str(uuid4())
    filename = f"{file_id}_{original_filename}"
    file_path = os.path.join(settings.upload_dir, filename)

    # Ensure upload directory exists
    os.makedirs(settings.upload_dir, exist_ok=True)

    # Save uploaded file
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Use filename as book name if not provided
    if not book_name:
        book_name = os.path.splitext(original_filename)[0]

    # Start processing
    from app.workers.tasks import process_pdf_task

    task = process_pdf_task.delay(
        file_path=file_path,
        book_name=book_name
    )

    return ProcessResponse(
        job_id=task.id,
        status="queued",
        message=f"Upload complete. Processing started for {book_name}"
    )


@app.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a processing job."""
    task = celery_app.AsyncResult(job_id)

    response = JobStatusResponse(
        job_id=job_id,
        status=task.status,
        progress=0.0
    )

    if task.state == "PENDING":
        response.status = "pending"
    elif task.state == "PROCESSING":
        response.status = "processing"
        if task.info:
            response.progress = task.info.get("progress", 0)
            response.result = task.info  # Include all meta info (stage, chapters, etc.)
    elif task.state == "SUCCESS":
        response.status = "completed"
        response.progress = 1.0
        response.result = task.result
    elif task.state == "FAILURE":
        response.status = "failed"
        response.error = str(task.result)
    elif task.state == "REVOKED":
        response.status = "cancelled"

    return response


@app.post("/job/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a processing job by revoking its Celery task."""
    task = celery_app.AsyncResult(job_id)

    if task.state in ("SUCCESS", "FAILURE"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status '{task.state}'"
        )

    # Revoke the task (terminate=True kills worker process if running)
    task.revoke(terminate=True)

    return {"success": True, "message": "Job cancellation requested"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Check Celery connection
    try:
        celery_app.control.ping(timeout=1)
        celery_status = "healthy"
    except Exception:
        celery_status = "unhealthy"

    return {
        "status": "healthy",
        "celery": celery_status
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "AcademiCK PDF Processing Service",
        "version": "1.0.0",
        "docs": "/docs"
    }
