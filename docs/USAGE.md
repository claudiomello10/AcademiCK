# AcademiCK Usage Guide

Detailed documentation for using, configuring, and troubleshooting AcademiCK.

## Table of Contents

- [Authentication](#authentication)
- [API Usage](#api-usage)
- [Processing New PDFs](#processing-new-pdfs)
- [Admin Dashboard](#admin-dashboard)
- [Backup and Restore](#backup-and-restore)
- [Data Migration](#data-migration)
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

```bash
curl -X POST http://localhost/api/v1/chat/{session_id} \
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

The system uses a **dual-method processing approach**:

1. **Primary: LLM-Based Processing**
   - Uses GPT-4o-mini to identify chapters from table of contents
   - Maintains hierarchical structure (Chapter -> Topics)
   - NLTK chunking with 3000 char chunks, 1000 char overlap
   - Quality filters: period ratio filter (>2% = skip), min 300 chars
   - Produces higher quality, structured output

2. **Fallback: Programmatic Processing**
   - Used when LLM processing fails (no TOC, API unavailable, etc.)
   - Programmatic chapter detection from TOC
   - Semantic chunking with 512 char chunks, 50 char overlap
   - User is warned when fallback is used

### Progress Tracking Features

- **Chapter-based progress**: Shows "Chapter X/Y" during processing
- **Real-time updates**: Progress bar updates every 3 seconds
- **Fallback warnings**: Yellow alert when fallback processor is used
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

### Create Qdrant Snapshot

```bash
python scripts/export_qdrant_snapshot.py \
  --action create \
  --output-dir ./data/qdrant_snapshots/
```

### Restore from Snapshot

```bash
python scripts/export_qdrant_snapshot.py \
  --action restore \
  --snapshot-path ./data/qdrant_snapshots/your-snapshot.snapshot
```

---

## Data Migration

If you have existing embeddings in SQLite format:

```bash
# Install migration dependencies
pip install asyncpg httpx qdrant-client tqdm

# Run migration (without sparse embeddings - faster)
python scripts/migrate_embeddings.py \
    --sqlite-path data/embeddings/embeddings.db \
    --no-regenerate-sparse \
    --validate \
    --create-snapshot
```

To generate sparse embeddings for hybrid search (requires embedding service):

```bash
# First start embedding service
docker compose up -d embedding-service

# Wait for model to load (~2-5 minutes)
docker logs -f academick-embedding

# Run migration with sparse embedding generation
python scripts/migrate_embeddings.py \
    --sqlite-path data/embeddings/embeddings.db \
    --validate \
    --create-snapshot
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `POSTGRES_PASSWORD` | **Yes** | - | PostgreSQL password |
| `REDIS_PASSWORD` | **Yes** | - | Redis authentication password |
| `SESSION_SECRET` | **Yes** | - | Session token encryption key |
| `ADMIN_PASSWORD` | **Yes** | - | Admin user password |
| `GUEST_PASSWORD` | **Yes** | - | Guest user password |
| `OPENAI_API_KEY` | At least one | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | At least one | - | Anthropic API key |
| `DEEPSEEK_API_KEY` | At least one | - | DeepSeek API key |
| `DEFAULT_SUBJECT` | No | Machine Learning | Default academic subject for new sessions |
| `POSTGRES_USER` | No | academick | PostgreSQL username |
| `POSTGRES_DB` | No | academick | Database name |
| `QDRANT_HOST` | No | qdrant | Qdrant hostname |
| `QDRANT_PORT` | No | 6333 | Qdrant HTTP port |
| `QDRANT_COLLECTION` | No | academick_embeddings | Qdrant collection name |
| `DEFAULT_MODEL` | No | gpt-5-mini | Default LLM model |
| `SESSION_TTL_MINUTES` | No | 30 | Session timeout |
| `CONFIG_USERS_ENABLED` | No | true | Enable config users |
| `DOCS_ENABLED` | No | true | Enable Swagger UI / ReDoc |
| `EMBEDDING_DEVICE` | No | gpu | Device for embeddings (gpu/cpu) |
| `EMBEDDING_BATCH_SIZE` | No | 16 | Embedding batch size |
| `ENABLE_SNAPSHOT_MANAGEMENT` | No | true | Show snapshot management in admin |
| `ENABLE_PDF_UPLOAD` | No | true | Show PDF upload in admin |
| `NEXT_PUBLIC_API_URL` | No | http://localhost | API URL for frontend |
| `NEXT_PUBLIC_DEFAULT_SUBJECT` | No | Machine Learning | Default subject in frontend |

See [.env.example](../.env.example) for the full list with detailed descriptions and advanced options.

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
