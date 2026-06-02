# AcademiCK Usage Guide

Detailed documentation for using, configuring, and troubleshooting AcademiCK.

## Table of Contents

- [Authentication](#authentication)
- [API Usage](#api-usage)
- [Processing New PDFs](#processing-new-pdfs)
- [Admin Dashboard](#admin-dashboard)
- [Backup and Restore](#backup-and-restore)
- [Environment Variables](#environment-variables)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

---

## Authentication

The system supports two authentication modes:

### Config Users (Default for Testing)

Pre-configured users loaded from environment variables:
- `guest` / your `GUEST_PASSWORD` - Regular user
- `admin` / your `ADMIN_PASSWORD` - Admin access

Set these passwords in your `.env` file. The system will not start without them.

### Database Users (Production)

Create users via the admin dashboard or directly in PostgreSQL with bcrypt-hashed passwords. Set `CONFIG_USERS_ENABLED=false` in your `.env` to disable config users.

---

## API Usage

The API documentation is available interactively at `http://localhost/docs` (Swagger UI) when `DOCS_ENABLED=true`.

### Login

```bash
curl -X POST http://localhost/api/v1/login \
  -H "Content-Type: application/json" \
  -d '{"username": "guest", "password": "your-guest-password"}'
```

Response:
```json
{
  "session_id": "abc123...",
  "username": "guest",
  "role": "user"
}
```

### Send a Query

The main chat endpoint streams its response as Server-Sent Events. The
stream emits `status` events for each pipeline stage (intent, enhancing,
searching, curating, generating), `token` events for incremental
answer deltas, a final `done` event with the full payload, and `error`
on failure (no messages are persisted in that case).

```bash
curl -N -X POST http://localhost/api/v1/chat/{session_id} \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"query": "What is gradient descent?"}'
```

For one-shot JSON without conversation history (no streaming), use the
`/single` endpoint which still returns a single `ChatResponse`:

```bash
curl -X POST http://localhost/api/v1/chat/{session_id}/single \
  -H "Content-Type: application/json" \
  -d '{"query": "What is gradient descent?"}'
```

### List Available Books

```bash
curl http://localhost/api/v1/books
```

### Admin Endpoints

```bash
# List processing jobs
curl "http://localhost/api/v1/admin/jobs?session_id={admin_session}"

# Get content stats
curl "http://localhost/api/v1/admin/content-stats?session_id={admin_session}"

# Get book list with chunk counts
curl "http://localhost/api/v1/admin/book-list?session_id={admin_session}"

# Delete a book
curl -X DELETE "http://localhost/api/v1/admin/books/{book_name}?session_id={admin_session}"

# Dismiss a job from the list
curl -X DELETE "http://localhost/api/v1/admin/jobs/{job_id}?session_id={admin_session}"

# Get usage statistics
curl "http://localhost/api/v1/admin/usage-stats?range=week&session_id={admin_session}"
```

---

## Processing New PDFs

### Via Admin Dashboard (Recommended)

1. Access the admin dashboard at http://localhost/admin
2. Login with your admin credentials (`admin` / your `ADMIN_PASSWORD`)
3. Go to "Content Management" tab
4. Click "Add New Content" to upload PDF files
5. Monitor processing progress with real-time chapter tracking

### Via API

```bash
# Upload and process PDF
curl -X POST "http://localhost/api/v1/admin/upload-pdfs?session_id={admin_session}" \
  -F "files=@your-book.pdf"
```

Monitor job status:
```bash
curl "http://localhost/api/v1/admin/jobs?session_id={admin_session}"
```

### PDF Processing Methods

The system uses a **two-tier processing cascade**, trying each method in order:

1. **Default: LLM-Based Processing** (`DefaultPDFProcessor`)
   - Uses an LLM to identify chapters and topics from the table of contents
   - Maintains hierarchical structure (Chapter → Topics)
   - NLTK chunking with per-chunk page tracking
   - Quality filters: period ratio filter (>2% = skip), minimum 300 chars
   - Configurable via `PDF_CHAPTER_DETECTION_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `MIN_CHUNK_LENGTH`

2. **Fallback: Layout-Based Processing** (`DoclingPDFProcessor`)
   - Used when LLM processing fails (no API key, no TOC detected, etc.)
   - Docling layout analysis detects headings in the PDF structure
   - LLM classifies detected headings into chapters vs. sub-sections
   - If both methods fail, the job fails with an error

### Progress Tracking Features

- **Chapter-based progress**: Shows "Chapter X/Y" during processing
- **Real-time updates**: Progress bar updates every 3 seconds
- **Fallback warnings**: Yellow alert when the Docling fallback is used
- **Job dismissal**: Manually dismiss completed jobs from the list
- **Persistent jobs**: Jobs are stored in PostgreSQL, visible to all admins
- **Auto-cleanup**: Jobs auto-hide after 12 hours or when exceeding 10 jobs

---

## Admin Dashboard

Access at http://localhost/admin with admin credentials.

### Content Management

- **Upload PDFs**: Drag-and-drop or click to upload PDF files
- **Processing status**: Real-time progress with chapter tracking
- **Book library**: View all processed books with chapter/chunk counts
- **Delete books**: Remove books and their embeddings from the system
- **Upload embeddings**: Import pre-generated embedding JSON files

### User Management

- **View users**: List all registered users
- **Edit users**: Modify username, email, role
- **Toggle status**: Activate/deactivate user accounts
- **Create users**: Add new users with specified roles

### Usage Statistics

- **Query metrics**: Total queries, response times, token usage
- **Time filtering**: View stats by day, week, or month
- **Per-user stats**: Track usage by individual users

---

## Backup and Restore

Snapshots are managed through the admin dashboard or the API. Each snapshot includes the Qdrant vector data and a metadata JSON file with book/chapter information.

### Via Admin Dashboard

1. Go to "Content Management" tab
2. Use the snapshot management buttons to create, restore, download, upload, or delete snapshots

### Via API

```bash
# Create snapshot
curl -X POST "http://localhost/api/v1/admin/snapshots/create?session_id={admin_session}"

# List snapshots
curl "http://localhost/api/v1/admin/snapshots?session_id={admin_session}"

# Restore snapshot
curl -X POST "http://localhost/api/v1/admin/snapshots/{snapshot_name}/restore?session_id={admin_session}"

# Download snapshot file
curl -O "http://localhost/api/v1/admin/snapshots/{snapshot_name}/download?session_id={admin_session}"

# Upload external snapshot with metadata
curl -X POST "http://localhost/api/v1/admin/snapshots/upload?session_id={admin_session}" \
  -F "snapshot_file=@your-snapshot.snapshot" \
  -F "metadata_file=@your-snapshot.metadata.json"

# Delete snapshot
curl -X DELETE "http://localhost/api/v1/admin/snapshots/{snapshot_name}?session_id={admin_session}"
```

---

## Environment Variables

### Required

| Variable | Description |
|----------|-------------|
| `POSTGRES_PASSWORD` | PostgreSQL password |
| `REDIS_PASSWORD` | Redis authentication password |
| `SESSION_SECRET` | Session token encryption key |
| `ADMIN_PASSWORD` | Admin user password |
| `GUEST_PASSWORD` | Guest user password |
| `OPENAI_API_KEY` | OpenAI API key (at least one LLM key required) |
| `ANTHROPIC_API_KEY` | Anthropic API key (at least one LLM key required) |
| `DEEPSEEK_API_KEY` | DeepSeek API key (at least one LLM key required) |

### General

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_SUBJECT` | `Machine Learning` | Default academic subject for new sessions |
| `SESSION_TTL_MINUTES` | `30` | Session expiration time in minutes |
| `DOCS_ENABLED` | `true` | Enable Swagger UI and ReDoc at `/docs` |
| `CONFIG_USERS_ENABLED` | `true` | Enable hardcoded admin/guest users (disable in production) |
| `NEXT_PUBLIC_API_URL` | `http://localhost` | API URL used by the frontend at build time |
| `NEXT_PUBLIC_DEFAULT_SUBJECT` | `Machine Learning` | Default subject shown in the frontend |

### LLM & RAG

| Variable | Default | Description |
|----------|---------|-------------|
| `AVAILABLE_MODELS` | _(required)_ | JSON array of `{provider, value, label}` models offered in the frontend dropdown; served at runtime via `GET /api/v1/models` |
| `DEFAULT_MODEL_FRONTEND` | _(required)_ | Initially-selected model; must match a `value` in `AVAILABLE_MODELS` |
| `QUERY_ENHANCEMENT_MODEL` | `gpt-5-nano` | Model for generating focused search queries (runs on every query) |
| `LLM_MAX_TOKENS` | `16384` | Maximum completion tokens (increase for reasoning models) |
| `TOP_K_SEARCHING` | `10` | Retrieval chunks for `searching_for_information` intent |
| `TOP_K_DEFAULT` | `6` | Retrieval chunks for all other intents |
| `SEARCH_WEIGHT_QA_DENSE` | `0.6` | Dense weight for Q&A queries (sparse = 1 - dense) |
| `SEARCH_WEIGHT_SUMMARIZATION_DENSE` | `0.7` | Dense weight for summarization queries |
| `SEARCH_WEIGHT_CODING_DENSE` | `0.4` | Dense weight for coding queries |
| `SEARCH_WEIGHT_SEARCHING_DENSE` | `0.5` | Dense weight for search queries |
| `QUERY_ENHANCEMENT_REASONING` | `none` | Reasoning effort for query enhancement (`none`, `low`, `medium`, `high`) |
| `RAG_REASONING` | `none` | Reasoning effort for main answer generation |
| `AGENT_CURATION_REASONING` | `low` | Reasoning effort for curation agent |

### Agentic RAG

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_ENABLED` | `true` | Enable the curation agent (disable for single-pass RAG) |
| `AGENT_MAX_ITERATIONS` | `3` | Maximum curation iterations before forcing approval |
| `AGENT_CURATION_MODEL` | `gpt-5-nano` | Model for curation evaluation (should be fast and cheap) |
| `AGENT_CURATION_REASONING` | `none` | Reasoning effort for the curation agent (`none`, `low`, `medium`, `high`) |
| `AGENT_CURATION_TIMEOUT` | `60` | Per-iteration timeout (seconds) for the curation model call; on timeout the agent uses the context gathered so far |
| `AGENT_CURATION_MAX_TOKENS` | `4096` | Max response tokens for the curation model (also scales the Anthropic thinking budget) |
| `AGENT_MAX_CONTEXT_CHUNKS` | `18` | Maximum chunks in the agent's context pool |
| `REASONING_TRACE_VISIBLE` | `false` | Expose the agent's reasoning trace in the chat UI as an expandable "ver raciocínio" toggle |

### Database & Storage

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_USER` | `academick` | PostgreSQL username |
| `POSTGRES_DB` | `academick` | PostgreSQL database name |
| `QDRANT_HOST` | `qdrant` | Qdrant hostname (`qdrant` for Docker, `localhost` for local dev) |
| `QDRANT_PORT` | `6333` | Qdrant HTTP API port |
| `QDRANT_COLLECTION` | `academick_embeddings` | Qdrant collection name for embeddings |

### Embedding Service

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_DEVICE` | `gpu` | Device for embeddings: `gpu` or `cpu` |
| `EMBEDDING_BATCH_SIZE` | `16` | Batch size for embedding generation (1–32) |
| `MODEL_NAME` | `BAAI/bge-m3` | HuggingFace embedding model (**changing requires re-processing all PDFs**) |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device IDs for CUDA (comma-separated for multi-GPU) |
| `MAX_LENGTH` | `8192` | Maximum tokenization input length in tokens |
| `USE_FP16` | `true` | Enable FP16 precision on GPU (reduces memory, slightly faster) |

### Intent Classification Service

| Variable | Default | Description |
|----------|---------|-------------|
| `INTENT_MODEL_NAME` | `claudiomello/AcademiCK-intent-classifier` | HuggingFace intent classifier model |
| `INTENT_DEVICE` | `cpu` | Device for intent classifier (`cpu` or `cuda`) |

### PDF Processing

| Variable | Default | Description |
|----------|---------|-------------|
| `PDF_CHAPTER_DETECTION_MODEL` | `gpt-5-nano` | LLM model for chapter/heading classification during PDF processing |
| `MAX_UPLOAD_SIZE_MB` | `100` | Maximum PDF upload size in megabytes |
| `CHUNK_SIZE` | `3000` | Text chunk size in characters (**changing requires re-processing all PDFs**) |
| `CHUNK_OVERLAP` | `1000` | Character overlap between chunks (must be less than `CHUNK_SIZE`) |
| `MIN_CHUNK_LENGTH` | `300` | Minimum chunk length to keep (shorter chunks are filtered out) |
| `CELERY_WORKER_CONCURRENCY` | `2` | Number of parallel PDF processing workers (higher = more memory) |
| `OMP_NUM_THREADS` | `4` | OpenMP threads for Docling/PyTorch CPU operations |
| `UPLOAD_DIR` | `/app/processed/uploads` | Upload directory inside container (**requires volume mount update**) |
| `PROCESSED_DIR` | `/app/processed` | Processed files directory inside container (**requires volume mount update**) |

### Admin Dashboard

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_SNAPSHOT_MANAGEMENT` | `true` | Show Qdrant snapshot management buttons in admin |
| `ENABLE_PDF_UPLOAD` | `true` | Show PDF upload button in admin |

See [../.env.example](../.env.example) for all options with inline documentation.

---

## Development

### Running Services Individually

```bash
# Start only infrastructure
docker compose up -d postgres redis qdrant

# Run API gateway locally
cd services/api-gateway
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Rebuilding Services

```bash
# Rebuild specific service
docker compose build embedding-service

# Rebuild all
docker compose build

# Rebuild without cache
docker compose build --no-cache embedding-service
```

### Viewing Logs

```bash
# All services
docker compose logs -f

# Specific service
docker logs -f academick-embedding
```

### Database Access

```bash
docker compose exec postgres psql -U academick -d academick
```

---

## Troubleshooting

### GPU Not Detected

```bash
# Verify NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.6-base-ubuntu22.04 nvidia-smi

# Check Docker daemon config
cat /etc/docker/daemon.json
```

If GPU is unavailable, set `EMBEDDING_DEVICE=cpu` in your `.env` to use CPU inference.

### Services Not Starting

```bash
# Check service health
docker compose ps

# View specific service logs
docker compose logs embedding-service

# Restart unhealthy service
docker compose restart embedding-service
```

### Database Connection Issues

```bash
# Check PostgreSQL is running
docker compose exec postgres pg_isready -U academick

# Connect to database
docker compose exec postgres psql -U academick -d academick
```

### Qdrant Snapshot Issues

```bash
# Ensure snapshot directory has correct permissions
mkdir -p data/qdrant_snapshots
chmod 777 data/qdrant_snapshots

# Check Qdrant logs
docker logs academick-qdrant
```

### Embedding Service Slow to Start

The embedding service downloads the BGE-M3 model (~2GB) on first run. This can take 2-5 minutes. Watch the logs:

```bash
docker logs -f academick-embedding
```

### Frontend Health Check Failing

The frontend uses `127.0.0.1` instead of `localhost` in its health check to avoid IPv6 resolution issues in Alpine containers. If the health check fails, check that port 3000 is accessible inside the container.
