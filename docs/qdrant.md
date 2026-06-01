# Qdrant Vector Store

AcademiCK stores chunk embeddings in a single Qdrant collection and uses hybrid
(dense + sparse) retrieval with Reciprocal Rank Fusion. Relational metadata and
the canonical chunk text live in PostgreSQL (see [database.md](database.md));
each Qdrant point corresponds to one row in the `chunks` table.

The collection is created on startup by `QdrantManager.ensure_collection`
([qdrant_client.py](../services/api-gateway/app/clients/qdrant_client.py)).

## Collection

- **Name:** `QDRANT_COLLECTION` env var (default `academick_embeddings`).
- **Point id:** a UUID (`qdrant_point_id`), also stored on the PostgreSQL
  `chunks` row so the two stores can be joined.

### Vectors

The collection uses **named vectors** — every point carries both:

| Name | Kind | Size | Distance | Source |
|------|------|------|----------|--------|
| `dense` | dense | 1024 | Cosine | BGE-M3 dense embedding |
| `sparse` | sparse | — | dot (sparse) | BGE-M3 learned sparse (lexical) weights |

Both come from the embedding service (`BAAI/bge-m3`). Sparse vectors are stored
as `{indices: int[], values: float[]}`. Sparse index is kept in memory
(`on_disk=False`).

## Payload

Each point stores this payload (written in
[tasks.py](../services/pdf-service/app/workers/tasks.py)):

| Field | Type | Notes |
|-------|------|-------|
| `chunk_id` | string (UUID) | matches `chunks.id` in PostgreSQL |
| `book_id` | string (UUID) | |
| `book_name` | string | indexed (keyword) |
| `chapter_id` | string (UUID) | |
| `chapter_title` | string | indexed (keyword) |
| `topic` | string | indexed (keyword) |
| `text` | string | chunk body (copy of `chunks.text`) |
| `is_introduction` | bool | indexed; true for a chapter's first chunk |
| `page_number` | int \| null | source page |
| `created_at` | string (ISO 8601) | |

### Payload indexes

Created so filters and lookups stay fast:

- `book_name` — keyword
- `chapter_title` — keyword
- `topic` — keyword
- `is_introduction` — bool

## Retrieval

`search_hybrid` issues two prefetches (dense and sparse, each `limit * 3`) and
fuses them server-side with `Fusion.RRF`, returning the top `limit` points. A
`book_name` filter narrows search to a single book. `search_dense` is a
dense-only fallback. Per-intent dense/sparse weighting and `top_k` are
configured via environment variables (see [USAGE.md](USAGE.md)).

Search results expose: `id`, `score`, `text`, `book_name`, `chapter_title`,
`topic`, `is_introduction`, `chunk_id`, `page_number`.

## Maintenance

- **Delete a book:** `delete_by_book_name` removes every point whose
  `book_name` matches. PostgreSQL rows are deleted separately via FK cascade.
- **Snapshots:** the collection can be snapshotted, listed, restored, and
  deleted through `QdrantManager` (exposed in the admin dashboard). Snapshots
  are the supported backup/restore path for the vector store.
- **Ingestion:** points are batch-upserted (100 at a time) by the PDF worker
  after embeddings are generated.
