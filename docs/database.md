# PostgreSQL Schema

AcademiCK stores all relational data in PostgreSQL. The schema is created by
[`scripts/init-db.sql`](../scripts/init-db.sql) on first startup. Vectors live
in Qdrant (see [qdrant.md](qdrant.md)); PostgreSQL holds the metadata, the text
of each chunk, conversation history, and analytics.

All primary keys are `UUID` (`gen_random_uuid()`). All timestamps are
`TIMESTAMP WITH TIME ZONE`.

## Entity overview

```
users ──┬─< sessions ──< conversations ──< messages ──< chunk_retrievals
        └─< conversations                                      │
                                                               │
books ──┬─< chapters ──< chunks <─────────────────────────────┘
        ├─< chunks
        └─< processing_jobs

usage_stats   (analytics, references users)
```

## Tables

### users
Account records. Passwords are bcrypt-hashed. Config users (admin/guest) are
created at startup from environment variables and flagged with `is_config_user`.

| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | PK |
| `username` | varchar(100) | unique, not null |
| `email` | varchar(255) | unique |
| `password_hash` | varchar(255) | bcrypt |
| `role` | varchar(50) | `user` \| `admin` |
| `status` | varchar(50) | `active` \| `inactive` \| `suspended` |
| `is_config_user` | bool | true for env-provisioned admin/guest |
| `created_at` / `updated_at` / `last_active` | timestamptz | `updated_at` maintained by trigger |

Indexes: `username`, `email`, `status`.

### sessions
Session metadata persisted for resumption; the live session lives in Redis.

| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | PK |
| `user_id` | UUID | FK → `users` (cascade delete) |
| `session_token` | varchar(255) | unique, not null |
| `subject` | varchar(255) | default `Machine Learning` |
| `expires_at` | timestamptz | not null |
| `is_active` | bool | default true |
| `created_at` / `last_active` | timestamptz | |

Indexes: `user_id`, `session_token`, partial index on `expires_at` where `is_active`.

### books
PDF metadata. One row per ingested book.

| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | PK |
| `name` | varchar(500) | unique, not null |
| `file_path` | varchar(1000) | |
| `file_hash` | varchar(64) | unique when present (dedup) |
| `total_pages` | int | |
| `total_chunks` | int | default 0 |
| `processing_status` | varchar(50) | `pending` \| `processing` \| `completed` \| `failed` |
| `processing_method` | varchar(50) | processor used (e.g. default / docling) |
| `error_message` | text | |
| `created_at` / `updated_at` / `processed_at` | timestamptz | `updated_at` via trigger |
| `metadata` | jsonb | default `{}` |

Indexes: `processing_status`, unique on `file_hash` (where not null).

### chapters
Chapter breakdown of a book.

| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | PK |
| `book_id` | UUID | FK → `books` (cascade) |
| `title` | varchar(500) | not null |
| `chapter_number` | int | |
| `start_page` / `end_page` | int | |
| `chunk_count` | int | default 0 |
| `created_at` | timestamptz | |

Indexes: `book_id`, `(book_id, chapter_number)`.

### chunks
Text segments. Each chunk maps 1:1 to a Qdrant point via `qdrant_point_id`.
The text itself is stored here (Qdrant keeps a copy in its payload).

| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | PK; equals the Qdrant payload `chunk_id` |
| `book_id` | UUID | FK → `books` (cascade) |
| `chapter_id` | UUID | FK → `chapters` (set null on delete) |
| `qdrant_point_id` | UUID | the Qdrant point id (not null) |
| `text` | text | not null |
| `topic` | varchar(500) | |
| `is_introduction` | bool | first chunk of a chapter |
| `page_number` | int | source page |
| `chunk_index` | int | order within book/chapter |
| `char_count` | int | |
| `created_at` | timestamptz | |
| `metadata` | jsonb | default `{}` |

Indexes: `book_id`, `chapter_id`, `qdrant_point_id`, `topic`.

### conversations
Groups messages into a resumable thread.

| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | PK |
| `session_id` | UUID | FK → `sessions` (set null) |
| `user_id` | UUID | FK → `users` (cascade) |
| `subject` | varchar(255) | |
| `title` | varchar(255) | |
| `message_count` | int | default 0 |
| `created_at` / `updated_at` | timestamptz | `updated_at` via trigger |

Indexes: `session_id`, `user_id`.

### messages
Chat history with per-message RAG metadata.

| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | PK |
| `conversation_id` | UUID | FK → `conversations` (cascade) |
| `role` | varchar(20) | `user` \| `assistant` \| `system` |
| `content` | text | not null |
| `intent` | varchar(100) | classified intent |
| `model_used` | varchar(100) | LLM that produced the answer |
| `tokens_used` | int | |
| `response_time_ms` | int | |
| `created_at` | timestamptz | |
| `metadata` | jsonb | default `{}` |

Indexes: `conversation_id`, `created_at`, `intent`.

### chunk_retrievals
Analytics: which chunks were retrieved for a given assistant message.

| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | PK |
| `message_id` | UUID | FK → `messages` (cascade) |
| `chunk_id` | UUID | FK → `chunks` (set null) |
| `book_id` | UUID | FK → `books` (set null) |
| `chapter_id` | UUID | FK → `chapters` (set null) |
| `score` | float | retrieval score |
| `position` | int | rank in the result list |
| `created_at` | timestamptz | |

Indexes: `message_id`, `chunk_id`, `book_id`, `created_at`.

### processing_jobs
Async job tracking for PDF ingestion and maintenance.

| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | PK |
| `job_type` | varchar(50) | `pdf_processing` \| `reindex` \| `migration` \| `embedding_update` |
| `status` | varchar(50) | `pending` \| `processing` \| `completed` \| `failed` \| `cancelled` |
| `book_id` | UUID | FK → `books` (set null) |
| `progress` | float | 0–1 |
| `error_message` | text | |
| `started_at` / `completed_at` / `created_at` | timestamptz | |
| `metadata` | jsonb | default `{}` |

Indexes: `status`, `book_id`, `job_type`.

### usage_stats
Per-action analytics, including agentic-RAG metrics.

| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | PK |
| `user_id` | UUID | FK → `users` (set null) |
| `session_id` | UUID | not a FK |
| `action_type` | varchar(50) | `query` \| `login` \| `logout` \| `pdf_upload` \| `search` \| `chat` |
| `response_time_ms` | int | |
| `model_used` | varchar(100) | |
| `tokens_consumed` | int | |
| `intent` | varchar(100) | |
| `success` | bool | default true |
| `agent_iterations` | int | curation-agent loops |
| `agent_tokens` | int | tokens spent by the agent |
| `agent_searches` | int | extra searches the agent issued |
| `agent_time_ms` | int | agent wall-clock time |
| `created_at` | timestamptz | |
| `metadata` | jsonb | default `{}` |

Indexes: `user_id`, `created_at`, `action_type`.

## Triggers and views

- **`update_updated_at_column()`** — trigger keeping `updated_at` current on
  `users`, `books`, and `conversations`.
- **`book_stats`** — view summarizing each book with its chapter count and
  processing status.

## Relationship to Qdrant

`chunks.qdrant_point_id` is the join key to the Qdrant vector store. Deleting a
book cascades to its chapters and chunks in PostgreSQL; vectors in Qdrant are
removed separately via `delete_by_book_name` (see [qdrant.md](qdrant.md)).
