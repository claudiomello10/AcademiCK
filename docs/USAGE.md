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

All authenticated endpoints take the session token in the
`Authorization: Bearer` header — it never appears in URLs, so it can't
leak via proxy logs, browser history, or `Referer` headers. The examples
below assume:

```bash
SESSION={session_id from login}
ADMIN_SESSION={session_id from admin login}
```

### Send a Query

The main chat endpoint streams its response as Server-Sent Events. The
stream emits `status` events for each pipeline stage (intent, enhancing,
searching, curating, generating), `token` events for incremental
answer deltas, a final `done` event with the full payload, and `error`
on failure (no messages are persisted in that case).

```bash
curl -N -X POST http://localhost/api/v1/chat \
  -H "Authorization: Bearer $SESSION" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"query": "What is gradient descent?"}'
```

For one-shot JSON without conversation history (no streaming), use the
`/single` endpoint which still returns a single `ChatResponse`:

```bash
curl -X POST http://localhost/api/v1/chat/single \
  -H "Authorization: Bearer $SESSION" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is gradient descent?"}'
```

### List Available Books

```bash
curl -H "Authorization: Bearer $SESSION" http://localhost/api/v1/books
```

### Admin Endpoints

```bash
# List processing jobs
curl -H "Authorization: Bearer $ADMIN_SESSION" "http://localhost/api/v1/admin/jobs"

# Get content stats
curl -H "Authorization: Bearer $ADMIN_SESSION" "http://localhost/api/v1/admin/content-stats"

# Get book list with chunk counts
curl -H "Authorization: Bearer $ADMIN_SESSION" "http://localhost/api/v1/admin/book-list"

# Delete a book
curl -X DELETE -H "Authorization: Bearer $ADMIN_SESSION" "http://localhost/api/v1/admin/books/{book_name}"

# Dismiss a job from the list
curl -X DELETE -H "Authorization: Bearer $ADMIN_SESSION" "http://localhost/api/v1/admin/jobs/{job_id}"

# Get usage statistics
curl -H "Authorization: Bearer $ADMIN_SESSION" "http://localhost/api/v1/admin/usage-stats?range=week"
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
curl -X POST -H "Authorization: Bearer $ADMIN_SESSION" "http://localhost/api/v1/admin/upload-pdfs" \
  -F "files=@your-book.pdf"
```

Monitor job status:
```bash
curl -H "Authorization: Bearer $ADMIN_SESSION" "http://localhost/api/v1/admin/jobs"
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
curl -X POST -H "Authorization: Bearer $ADMIN_SESSION" "http://localhost/api/v1/admin/snapshots/create"

# List snapshots
curl -H "Authorization: Bearer $ADMIN_SESSION" "http://localhost/api/v1/admin/snapshots"

# Restore snapshot
curl -X POST -H "Authorization: Bearer $ADMIN_SESSION" "http://localhost/api/v1/admin/snapshots/{snapshot_name}/restore"

# Download snapshot file
curl -O -H "Authorization: Bearer $ADMIN_SESSION" "http://localhost/api/v1/admin/snapshots/{snapshot_name}/download"

# Upload external snapshot with metadata
curl -X POST -H "Authorization: Bearer $ADMIN_SESSION" "http://localhost/api/v1/admin/snapshots/upload" \
  -F "snapshot_file=@your-snapshot.snapshot" \
  -F "metadata_file=@your-snapshot.metadata.json"

# Delete snapshot
curl -X DELETE -H "Authorization: Bearer $ADMIN_SESSION" "http://localhost/api/v1/admin/snapshots/{snapshot_name}"
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
| `OPENAI_API_KEY` | OpenAI API key (at least one LLM provider required) |
| `ANTHROPIC_API_KEY` | Anthropic API key (at least one LLM provider required) |
| `DEEPSEEK_API_KEY` | DeepSeek API key (at least one LLM provider required) |

> A self-hosted OpenAI-compatible server (vLLM, Ollama, …) counts as an LLM
> provider and needs no API key — see [Local LLMs](local-llms.md).

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

Models are routed to a provider by an explicit `provider/` prefix on the model
name — `openai/`, `anthropic/`, `deepseek/`, or `local/` (an unprefixed name
defaults to OpenAI). Every `*_MODEL` value and every `AVAILABLE_MODELS` entry
should carry this prefix. See [Local LLMs](local-llms.md) for self-hosting and the
full routing rules.

| Variable | Default | Description |
|----------|---------|-------------|
| `AVAILABLE_MODELS` | _(required)_ | JSON array of `{provider, value, label}` models offered in the frontend dropdown; each `value` must start with a `provider/` prefix. Served at runtime via `GET /api/v1/models`; validated at startup |
| `DEFAULT_MODEL_FRONTEND` | _(required)_ | Initially-selected model; must match a `value` in `AVAILABLE_MODELS` |
| `QUERY_ENHANCEMENT_MODEL` | `openai/gpt-5-nano` | Model for generating focused search queries (runs on every query). Uses structured output — **must support tool calling** |
| `LOCAL_LLM_BASE_URL` | `http://host.docker.internal:8000/v1` | OpenAI-compatible base URL for `local/` models (self-hosted server on the same host by default) |
| `LOCAL_LLM_API_KEY` | `EMPTY` | Token for the local LLM server (most ignore it; the client requires a non-empty value) |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | DeepSeek base URL (override for a proxy/gateway) |
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
| `AGENT_MAX_ACTIONS` | `8` | Single shared budget: every tool call (search or navigation) spends one action, since each call grows the context and cost |
| `AGENT_MAX_QUERIES_PER_SEARCH` | `3` | Max queries batched into one `search` call |
| `AGENT_NAV_MAX_ITEMS` | `3` | Max books/chapters per navigation call |
| `AGENT_READ_CHAPTER_FULL_ENABLED` | `false` | Allow `read_chapter(mode="full")` to return whole chapters (token-heavy) |
| `AGENT_CURATION_MODEL` | `openai/gpt-5-nano` | Model for curation. Uses tool calling — **the model must support function calling**; otherwise the agent falls back to single-pass |
| `AGENT_CURATION_REASONING` | `none` | Reasoning effort for the curation agent (`none`, `low`, `medium`, `high`) |
| `AGENT_CURATION_TIMEOUT` | `150` | Overall timeout (seconds) for the curation run (one model round-trip per tool call, so it must cover many rounds); on timeout the request fails (SSE `error` event / HTTP 504 on `/single`) |
| `AGENT_CURATION_MAX_TOKENS` | `8192` | Max response tokens per curation round; tool-calling reasoning needs headroom (also scales the Anthropic thinking budget) |
| `AGENT_MAX_CONTEXT_CHUNKS` | `18` | Maximum chunks in the agent's final curated context |
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
| `PDF_CHAPTER_DETECTION_MODEL` | `openai/gpt-5-nano` | LLM for chapter/heading classification during PDF processing. Accepts any `provider/` prefix (incl. `local/`); plain text, no tool calling required |
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
