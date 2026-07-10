# Intent Classification Service Dockerfile
# Uses PyTorch with CUDA (can run on CPU or GPU)

FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
