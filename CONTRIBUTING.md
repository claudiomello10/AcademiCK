# Contributing to AcademiCK

Thank you for your interest in contributing to AcademiCK! Whether it's reporting a bug, suggesting a feature, improving documentation, or writing code, every contribution is valuable.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [How to Contribute](#how-to-contribute)
- [Code Style](#code-style)
- [Pull Request Process](#pull-request-process)
- [Getting Help](#getting-help)

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/your-username/TCC-RAG.git
   cd TCC-RAG
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- **Docker** and **Docker Compose** v2.0+
- **Git**
- At least one LLM API key (OpenAI, Anthropic, or DeepSeek)
- **NVIDIA GPU** with CUDA support (recommended for embedding service, but CPU mode is available)
- **Node.js 18+** (only if developing the frontend locally outside Docker)
- **Python 3.11+** (only if developing backend services locally outside Docker)

### Environment Configuration

```bash
# Copy the environment template
cp .env.example .env

# Edit .env with your settings
# At minimum, set one LLM API key and change SESSION_SECRET
nano .env
```

See [.env.example](.env.example) for all available configuration options with descriptions.

### Running with Docker (Recommended)

```bash
# Start all services
docker compose up -d

# Check service health
docker compose ps

# View logs for a specific service
docker compose logs -f api-gateway
```

### Running Services Locally (for development)

If you want to develop a specific service without rebuilding Docker images:

```bash
# Start infrastructure services
docker compose up -d postgres redis qdrant

# Run API gateway locally
cd services/api-gateway
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Run frontend locally
cd services/academick-frontend
npm install
npm run dev
```

### Rebuilding After Changes

```bash
# Rebuild a specific service
docker compose build api-gateway
docker compose up -d api-gateway

# Rebuild everything
docker compose build
docker compose up -d
```

## Project Structure

```
TCC-RAG/
├── services/                   # Backend microservices
│   ├── api-gateway/            # Main FastAPI application (RAG pipeline, auth, admin)
│   ├── embedding-service/      # BGE-M3 embedding model server
│   ├── intent-service/         # Query intent classification
│   ├── pdf-service/            # PDF processing with Celery workers
│   ├── frontend/               # Next.js Dockerfile
│   └── academick-frontend/     # Next.js frontend source code
├── scripts/                    # Database schema, migration scripts
├── config/                     # Nginx reverse proxy config
├── docs/                       # Detailed documentation
└── docker-compose.yml          # Service orchestration
```

### Key Backend Files

| File | Purpose |
|------|---------|
| `services/api-gateway/app/services/rag_orchestrator.py` | Main RAG pipeline |
| `services/api-gateway/app/services/search_service.py` | Hybrid vector search |
| `services/api-gateway/app/services/prompt_engineering.py` | Prompt templates |
| `services/api-gateway/app/services/llm_service.py` | Multi-provider LLM client |
| `services/api-gateway/app/routers/` | API endpoint handlers |
| `services/api-gateway/app/clients/` | Service-to-service HTTP clients |

### Key Frontend Files

| File | Purpose |
|------|---------|
| `services/academick-frontend/components/StudentHelper.tsx` | Main chat interface |
| `services/academick-frontend/components/AdminDashboard.tsx` | Admin panel |
| `services/academick-frontend/config/constants.ts` | API endpoint configuration |

## How to Contribute

### Reporting Bugs

- Use the [Bug Report](https://github.com/claudiomello10/TCC-RAG/issues/new?template=bug_report.md) issue template
- Include steps to reproduce, expected vs actual behavior, and environment details
- Attach relevant logs (`docker compose logs <service-name>`)

### Suggesting Features

- Use the [Feature Request](https://github.com/claudiomello10/TCC-RAG/issues/new?template=feature_request.md) issue template
- Describe the problem your feature would solve
- Consider how it fits into the existing architecture

### Improving Documentation

Documentation improvements are always welcome! This includes:
- Fixing typos or unclear instructions
- Adding examples or diagrams
- Translating documentation

### Writing Code

1. Check existing [issues](https://github.com/claudiomello10/TCC-RAG/issues) for something to work on
2. Comment on the issue to let others know you're working on it
3. Follow the [Pull Request Process](#pull-request-process) below

## Code Style

### Python (Backend Services)

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions
- Use type hints for function parameters and return values
- Use `async/await` for I/O operations (FastAPI is async-first)
- Keep functions focused and reasonably sized

### TypeScript/React (Frontend)

- Follow the existing patterns in the codebase
- Use functional components with hooks
- Use TypeScript types/interfaces for props and state

### Commit Messages

We encourage [Conventional Commits](https://www.conventionalcommits.org/) format:

```
feat: add book filtering to search results
fix: resolve session timeout on idle connections
docs: update API usage examples
refactor: extract embedding client into separate module
```

## Pull Request Process

1. **Ensure your branch is up to date** with the main branch:
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. **Test your changes**:
   - Start all services and verify they are healthy
   - Test the specific feature or fix you implemented
   - Verify no existing functionality is broken

3. **Create your Pull Request**:
   - Use a clear, descriptive title
   - Fill out the PR template
   - Reference any related issues (e.g., "Closes #42")
   - Describe what you changed and why

4. **Review process**:
   - A maintainer will review your PR
   - Address any requested changes
   - Once approved, a maintainer will merge your PR

### PR Checklist

Before submitting, make sure:

- Your code follows the existing code style
- You've tested your changes locally
- All services start and pass health checks
- You haven't committed any secrets or API keys
- You've updated documentation if needed

## Getting Help

- **Questions about the codebase?** Open a [Discussion](https://github.com/claudiomello10/TCC-RAG/discussions) or an issue
- **Stuck on setup?** Check the [Usage Guide](docs/USAGE.md) for detailed instructions and troubleshooting
- **Found a bug?** [Open an issue](https://github.com/claudiomello10/TCC-RAG/issues/new?template=bug_report.md)

Thank you for contributing!
