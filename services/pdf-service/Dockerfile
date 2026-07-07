# PDF Processing Service Dockerfile
# Uses PyTorch base image (same as embedding-service) to provide PyTorch,
# CUDA support, and core native libs needed by Docling's layout model.

FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

WORKDIR /app

# Install system dependencies for PDF processing and Docling (OpenCV, libmagic, MuPDF)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libmagic1 \
    libmupdf-dev \
    mupdf-tools \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Model/data caches live under /app so they stay readable once the
# container drops to the non-root user.
ENV HF_HOME=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch
ENV NLTK_DATA=/app/nltk_data

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data for text processing
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Pre-download Docling layout models to avoid cold-start delay at runtime
RUN python -c "from docling.document_converter import DocumentConverter; DocumentConverter()" 2>/dev/null || true

# Copy application code
COPY app/ ./app/

# Run as non-root. /app/processed must exist in the image so the
# pdf_uploads named volume is seeded with app ownership on first creation.
RUN groupadd --system --gid 1000 app && useradd --system --uid 1000 --gid app app \
    && mkdir -p /app/processed \
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
