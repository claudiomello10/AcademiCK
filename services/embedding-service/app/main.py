"""
Embedding Service - BGE-M3 Model Serving

Provides endpoints for generating dense and sparse embeddings
using the BAAI/bge-m3 model with GPU acceleration.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import torch
import time
import logging

from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model = None
device = None


class EmbedRequest(BaseModel):
    """Request model for embedding generation."""
    texts: List[str] = Field(..., min_length=1, max_length=100)
    return_sparse: bool = Field(default=False, description="Return sparse/lexical embeddings")
    return_colbert: bool = Field(default=False, description="Return ColBERT token embeddings")


class EmbedResponse(BaseModel):
    """Response model for embedding generation."""
    dense_embeddings: List[List[float]]
    sparse_embeddings: Optional[List[Dict[str, float]]] = None
    colbert_embeddings: Optional[List[List[List[float]]]] = None
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_name: str
    device: str
    gpu_available: bool
    gpu_name: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - load model on startup."""
    global model, device

    # Determine device from configuration
    if settings.device.lower() == "gpu":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Using GPU (configured): {gpu_name}")
        else:
            device = torch.device("cpu")
            logger.warning("GPU configured but CUDA not available - falling back to CPU")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU (configured)")

    # Load BGE-M3 model
    logger.info(f"Loading model: {settings.model_name}")
    start_time = time.time()

    try:
        from FlagEmbedding import BGEM3FlagModel

        model = BGEM3FlagModel(
            settings.model_name,
            device=device.type,  # "cuda" or "cpu"
            use_fp16=settings.use_fp16 and device.type == "cuda"
        )

        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f}s")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    yield

    # Cleanup
    logger.info("Shutting down embedding service")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Create FastAPI application
app = FastAPI(
    title="AcademiCK Embedding Service",
    description="BGE-M3 embedding generation service with GPU acceleration",
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


@app.post("/embed", response_model=EmbedResponse)
async def generate_embeddings(request: EmbedRequest):
    """
    Generate embeddings for a list of texts.

    Returns:
    - dense_embeddings: Semantic embeddings (1024 dimensions)
    - sparse_embeddings: Lexical/keyword embeddings (optional)
    - colbert_embeddings: Token-level embeddings (optional)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    start_time = time.time()

    try:
        # Generate embeddings
        embeddings = model.encode(
            request.texts,
            return_dense=True,
            return_sparse=request.return_sparse,
            return_colbert_vecs=request.return_colbert,
            batch_size=settings.batch_size,
            max_length=settings.max_length
        )

        # Build response
        response = EmbedResponse(
            dense_embeddings=[e.tolist() for e in embeddings["dense_vecs"]],
            processing_time_ms=(time.time() - start_time) * 1000
        )

        # Add sparse embeddings if requested
        if request.return_sparse and "lexical_weights" in embeddings:
            response.sparse_embeddings = [
                {str(k): float(v) for k, v in e.items()}
                for e in embeddings["lexical_weights"]
            ]

        # Add ColBERT embeddings if requested
        if request.return_colbert and "colbert_vecs" in embeddings:
            response.colbert_embeddings = [
                e.tolist() for e in embeddings["colbert_vecs"]
            ]

        return response

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    return HealthResponse(
        status="healthy" if model is not None else "loading",
        model_loaded=model is not None,
        model_name=settings.model_name,
        device=str(device) if device else "unknown",
        gpu_available=torch.cuda.is_available(),
        gpu_name=gpu_name
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "AcademiCK Embedding Service",
        "model": settings.model_name,
        "version": "1.0.0",
        "docs": "/docs"
    }
