<p align="center">
  <img src="services/frontend/public/app_icon.png" alt="AcademiCK Logo" width="200"/>
</p>

# AcademiCK

**A self-hosted, RAG-powered study assistant for any academic subject.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](docker-compose.yml)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Next.js 15](https://img.shields.io/badge/Next.js-15-000000?logo=next.js&logoColor=white)](https://nextjs.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

AcademiCK helps students learn from textbook content using Retrieval-Augmented Generation. Upload your PDF textbooks, and the system processes them into searchable chunks with embeddings. Students can then ask questions and get accurate, citation-backed answers grounded in the source material.

The system is **subject-agnostic** — configure it for Machine Learning, Organic Chemistry, Constitutional Law, or any academic domain. Just set your `DEFAULT_SUBJECT` and upload the relevant textbooks.

## Features

- **Hybrid Vector Search** — Combines dense and sparse embeddings (BGE-M3) with intent-tuned weighted score fusion for accurate retrieval
- **Intent-Aware Queries** — Custom classifier detects query type (Q&A, summarization, coding, search) and adapts behavior
- **Multi-Provider LLM** — Choose between OpenAI, Anthropic, or DeepSeek models per query
- **Agentic Context Curation** — Multi-iteration agent drops noise and fetches missing context before answer generation
- **PDF Processing Pipeline** — Two-tier processing cascade: LLM-based and layout-based (Docling) with per-chunk page tracking
- **Session Management** — Redis-backed sessions with conversation history and context
- **Admin Dashboard** — Content management, user management, and usage statistics
- **Fully Dockerized** — One command to start 11 services with health checks and auto-restart
- **GPU & CPU Support** — GPU-accelerated embeddings with automatic CPU fallback

## Architecture

```
                            +------------------+
                            |      nginx       |
                            |    (port 80)     |
                            +--------+---------+
                                     |
               +---------------------+---------------------+
               |                     |                     |
      +--------v--------+   +--------v---------+   +-------v---------+
      |    Frontend     |   |   API Gateway    |   |   PDF Service   |
      |   (Next.js)     |   | +--------------+ |   |   (FastAPI)     |
      +-----------------+   | |     RAG      | |   +-------+---------+
                            | | Orchestrator | |           |
                            | +------+-------+ |           |
                            |        |         |           |
                            | +------v-------+ |           |
                            | |   Curation   | |           |
                            | |    Agent     | |           |
                            | +--------------+ |           |
                            +------+-----------+           |
                                   |                       |
         +-----------+-------------+------------+----------+--+
         |           |             |            |             |
+--------v--+ +------v-----+ +-----v----+ +-----v----+ +------v-----+
|  Intent   | | Embedding  | |  Qdrant  | |  Redis   | | PostgreSQL |
| Classifier| |  Service   | | Vectors  | |  Cache   | |  Database  |
+-----------+ +------------+ +----------+ +----------+ +------------+

                    LLM APIs: OpenAI / Anthropic / DeepSeek
```

> All internal services communicate over an isolated Docker network. Only nginx is exposed to the host.

### RAG Pipeline

When a user sends a query, the API Gateway orchestrates the following stages:

1. **Intent classification + Query enhancement** (parallel)

   - A fine-tuned classifier detects query type (Q&A, summarization, coding, search)
   - A fast LLM generates up to 3 focused search queries and a resolved query (pronouns/references replaced with actual terms from conversation history)
2. **Hybrid search** with enhanced queries

   - Each query is embedded and matched against textbook chunks in Qdrant
   - Dense and sparse scores are fused via Reciprocal Rank Fusion
   - Results are merged and deduplicated across queries
3. **Agentic context curation** (when `AGENT_ENABLED=true`)

   - A curation agent evaluates retrieved chunks in an iterative loop (up to `AGENT_MAX_ITERATIONS`)
   - Each iteration: **APPROVE** (context is sufficient) or **REFINE** (drop noise, fetch more via new searches)
   - The agent never answers — it only curates
4. **Answer generation** with curated context

   - The main LLM receives curated chunks and generates a citation-backed response, streamed token-by-token

When the curation agent is disabled, the pipeline skips step 3 (single-pass).

All three LLM stages (query enhancement, curation, answer) run through Pydantic AI with typed structured outputs. The chat endpoint (`POST /api/v1/chat/{session_id}`) streams pipeline progress and answer tokens to the UI as Server-Sent Events; the legacy non-streaming response shape is still available at `/single`.

## Quick Start

### Prerequisites

- **Docker** and **Docker Compose** v2.0+
- At least one LLM API key ([OpenAI](https://platform.openai.com/api-keys), [Anthropic](https://console.anthropic.com/), or [DeepSeek](https://platform.deepseek.com/))
- **16GB+ RAM** recommended for smooth performance
- **NVIDIA GPU** with CUDA (recommended) — or set `EMBEDDING_DEVICE=cpu` for CPU-only mode

### 1. Clone and Configure

```bash
git clone https://github.com/claudiomello10/AcademiCK.git
cd AcademiCK

cp .env.example .env
```

Edit `.env` and set the required values:

```env
# Required: Set all passwords (system won't start without them)
POSTGRES_PASSWORD=your-secure-password
REDIS_PASSWORD=your-secure-password
SESSION_SECRET=your-secure-random-string
ADMIN_PASSWORD=your-admin-password
GUEST_PASSWORD=your-guest-password

# Required: At least one LLM provider API key
OPENAI_API_KEY=sk-your-key-here

# (Optional) Change the default subject
DEFAULT_SUBJECT=Machine Learning
```

### 2. Start All Services

```bash
docker compose up -d
```

> **Note:** On the first run, the `intent-service` and `embedding-service` containers will take longer to start because they need to download their ML models (~1-2 GB). Subsequent starts will use the cached models.

Wait for all services to become healthy:

```bash
docker compose ps
```

### 3. Access the Application

| URL                    | Description     |
| ---------------------- | --------------- |
| http://localhost       | Web interface   |
| http://localhost/admin | Admin dashboard |

### 4. Login

Use the credentials you set in `.env`:

- **Guest**: `guest` / your `GUEST_PASSWORD`
- **Admin**: `admin` / your `ADMIN_PASSWORD`

## Configuration

Key settings in `.env` (see [.env.example](.env.example) and [docs/USAGE.md](docs/USAGE.md) for the full list):

| Variable                      | Required     | Description                                            |
| ----------------------------- | ------------ | ------------------------------------------------------ |
| `POSTGRES_PASSWORD`         | Yes          | PostgreSQL password                                    |
| `REDIS_PASSWORD`            | Yes          | Redis authentication password                          |
| `SESSION_SECRET`            | Yes          | Secret key for session encryption                      |
| `ADMIN_PASSWORD`            | Yes          | Admin user password                                    |
| `GUEST_PASSWORD`            | Yes          | Guest user password                                    |
| `OPENAI_API_KEY`            | At least one | OpenAI API key                                         |
| `ANTHROPIC_API_KEY`         | At least one | Anthropic API key                                      |
| `DEEPSEEK_API_KEY`          | At least one | DeepSeek API key                                       |
| `DEFAULT_SUBJECT`           | No           | Academic subject (default: Machine Learning)           |
| `AVAILABLE_MODELS`          | Yes          | JSON array of models offered in the frontend dropdown  |
| `DEFAULT_MODEL_FRONTEND`    | Yes          | Initially-selected model (a `value` in `AVAILABLE_MODELS`) |
| `AGENT_ENABLED`             | No           | Enable curation agent (default: true)                  |
| `REASONING_TRACE_VISIBLE`   | No           | Expose the agent's reasoning trace in the chat UI (default: false) |
| `EMBEDDING_DEVICE`          | No           | Embedding device:`gpu` or `cpu` (default: gpu)     |
| `CHUNK_SIZE`                | No           | Chunk size in chars for PDF processing (default: 3000) |
| `CELERY_WORKER_CONCURRENCY` | No           | Parallel PDF processing workers (default: 2)           |
| `OMP_NUM_THREADS`           | No           | OpenMP threads for Docling/PyTorch (default: 4)        |
| `DOCS_ENABLED`              | No           | Enable Swagger UI (default: true)                      |

## Services

| Service               | Description                            |
| --------------------- | -------------------------------------- |
| `nginx`             | Reverse proxy (only exposed port: 80)  |
| `frontend`          | Next.js web interface                  |
| `api-gateway`       | Main API + RAG orchestration           |
| `intent-service`    | Intent classification                  |
| `embedding-service` | BGE-M3 embeddings (GPU/CPU)            |
| `pdf-service`       | PDF processing                         |
| `pdf-worker`        | Celery worker for async PDF processing |
| `qdrant`            | Vector database                        |
| `postgres`          | PostgreSQL database                    |
| `redis`             | Cache and task queue                   |

## Documentation

| Document                     | Description                                                    |
| ---------------------------- | -------------------------------------------------------------- |
| [Usage Guide](docs/USAGE.md)    | API examples, admin dashboard, PDF processing, troubleshooting |
| [Database Schema](docs/database.md) | PostgreSQL tables, relationships, and analytics            |
| [Vector Store](docs/qdrant.md)  | Qdrant collection, vectors, payload, and retrieval             |
| [Known Issues](docs/KNOWN_ISSUES.md) | Logged problems, root causes, and workarounds            |
| [Security Policy](SECURITY.md)  | Vulnerability reporting and deployment best practices          |
| [Contributing](CONTRIBUTING.md) | Development setup, code style, PR process                      |
| [.env.example](.env.example)    | All configuration options with descriptions                    |

## Tech Stack

| Layer                    | Technology                                   |
| ------------------------ | -------------------------------------------- |
| **Backend**        | FastAPI, Python 3.11+                        |
| **Frontend**       | Next.js 15, React 19, TailwindCSS, shadcn/ui |
| **Vector DB**      | Qdrant (hybrid dense + sparse search)        |
| **Database**       | PostgreSQL 16                                |
| **Cache**          | Redis 7                                      |
| **Embeddings**     | BGE-M3 (BAAI/bge-m3)                         |
| **Task Queue**     | Celery with Redis broker                     |
| **Infrastructure** | Docker, Docker Compose, Nginx                |

## Contributing

Contributions are welcome! Whether it's fixing a bug, adding a feature, or improving documentation, we appreciate your help.

Please read our [Contributing Guide](CONTRIBUTING.md) to get started.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [BGE-M3](https://huggingface.co/BAAI/bge-m3) for multilingual hybrid embeddings
- [Qdrant](https://qdrant.tech/) for vector search
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- [Next.js](https://nextjs.org/) for the frontend framework
- [shadcn/ui](https://ui.shadcn.com/) for UI components
