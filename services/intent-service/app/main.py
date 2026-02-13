"""
Intent Classification Service

Provides endpoints for classifying user query intents
using the AcademiCK-intent-classifier model.

Intents:
- question_answering: Academic Q&A
- summarization: Content summarization
- coding: Code generation/explanation
- searching_for_information: General search/discovery
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List, Optional
import time
import logging

from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global classifier instance
classifier = None


class ClassifyRequest(BaseModel):
    """Request model for intent classification."""
    text: str = Field(..., min_length=1, max_length=10000)


class ClassifyResponse(BaseModel):
    """Response model for intent classification."""
    intent: str
    confidence: float
    processing_time_ms: float


class BatchClassifyRequest(BaseModel):
    """Request model for batch intent classification."""
    texts: List[str] = Field(..., min_length=1, max_length=50)


class BatchClassifyResponse(BaseModel):
    """Response model for batch intent classification."""
    results: List[ClassifyResponse]
    total_processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_name: str
    device: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - load model on startup."""
    global classifier

    logger.info(f"Loading intent classifier: {settings.model_name}")
    logger.info(f"Device: {settings.device}")
    start_time = time.time()

    try:
        from transformers import pipeline

        # Use CPU (device=-1) or GPU (device=0)
        device_id = -1 if settings.device == "cpu" else 0

        classifier = pipeline(
            "text-classification",
            model=settings.model_name,
            device=device_id
        )

        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f}s")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    yield

    # Cleanup
    logger.info("Shutting down intent classification service")
    del classifier


# Create FastAPI application
app = FastAPI(
    title="AcademiCK Intent Classification Service",
    description="Intent classification for RAG query routing",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/classify", response_model=ClassifyResponse)
async def classify_intent(request: ClassifyRequest):
    """
    Classify the intent of a single query.

    Returns:
    - intent: One of [question_answering, summarization, coding, searching_for_information]
    - confidence: Classification confidence score
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        result = classifier(request.text)[0]

        return ClassifyResponse(
            intent=result["label"],
            confidence=result["score"],
            processing_time_ms=(time.time() - start_time) * 1000
        )

    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/batch", response_model=BatchClassifyResponse)
async def batch_classify(request: BatchClassifyRequest):
    """
    Classify intents for multiple queries in batch.

    More efficient than multiple single requests.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        results = classifier(request.texts)

        return BatchClassifyResponse(
            results=[
                ClassifyResponse(
                    intent=r["label"],
                    confidence=r["score"],
                    processing_time_ms=0  # Individual times not tracked in batch
                )
                for r in results
            ],
            total_processing_time_ms=(time.time() - start_time) * 1000
        )

    except Exception as e:
        logger.error(f"Batch classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if classifier is not None else "loading",
        model_loaded=classifier is not None,
        model_name=settings.model_name,
        device=settings.device
    )


@app.get("/intents")
async def list_intents():
    """List available intent categories."""
    return {
        "intents": [
            {
                "name": "question_answering",
                "description": "Academic questions requiring educational explanations"
            },
            {
                "name": "summarization",
                "description": "Requests to summarize topics or concepts"
            },
            {
                "name": "coding",
                "description": "Code generation, explanation, or debugging requests"
            },
            {
                "name": "searching_for_information",
                "description": "General information discovery and search"
            }
        ]
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "AcademiCK Intent Classification Service",
        "model": settings.model_name,
        "version": "1.0.0",
        "docs": "/docs"
    }
