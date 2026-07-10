# Embedding Service Dockerfile
# Uses PyTorch with CUDA for GPU acceleration with BGE-M3 model

FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model (this is cached in the Docker layer)
# Comment out if you want to use volume-mounted cache instead
# RUN python3 -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3')"

# Copy application code
COPY app/ ./app/

# Run as non-root; the model cache lives under /app so the model_cache
# volume is seeded with app ownership on first creation.
ENV HF_HOME=/app/.cache/huggingface
RUN groupadd --system --gid 1000 app && useradd --system --uid 1000 --gid app app \
    && mkdir -p /app/.cache/huggingface \
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
