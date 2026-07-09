# Testing

The test suites verify *behavior at stable boundaries* — HTTP responses, rows
in PostgreSQL, points in Qdrant — never private functions or internal call
patterns. Internals can be rewritten freely (chunking strategy, score fusion,
agent internals); a test only fails when something a user or another service
can observe actually breaks.

Everything runs against **real infrastructure**: PostgreSQL, Redis, Qdrant,
the bge-m3 embedding service, and the intent classifier are all real, both
locally (the dev compose stack) and in CI (service containers + CPU model
servers). The only thing ever faked is the LLM, and only in the default tier.

## Layout and tiers

| Location | Tier | LLM | Runs in CI |
|---|---|---|---|
| `services/api-gateway/tests/` | default | faked (pydantic-ai test models) | yes |
| `services/pdf-service/tests/` | default | faked (patched chapter detection) | yes |
| same dirs, `-m llm` | llm | **real**, models from `.env` | never |
| `tests/e2e/` | e2e | **real**, whole stack through nginx | never |

```bash
scripts/run-tests.sh              # default tier, both service suites
scripts/run-tests.sh --llm       # + real-LLM tests (costs API credits)
scripts/run-tests.sh --e2e       # + full-stack e2e (stack must be up)
scripts/run-tests.sh -k purge -v # extra args go straight to pytest
```

The runner sources `.env`, resolves the infra containers' IPs (only nginx is
port-mapped to the host), exports `DATABASE_URL`/`REDIS_URL`/`QDRANT_HOST`/
`EMBEDDING_SERVICE_URL`/`INTENT_SERVICE_URL`, and runs each suite from its
service directory in a per-service `.venv-test`. Suites refuse to start if the
env is missing, so run them through the script.

Tests write to the dev stack (books, sessions, usage rows). Every seeded
object is uniquely named (`test-book-<hex>`) and deleted on teardown, but
treat the dev data as disposable.

Unmarked tests can never reach a real LLM: an autouse fixture sets pydantic-ai
`ALLOW_MODEL_REQUESTS = False` unless the test carries the `llm` marker.

---

## api-gateway (`services/api-gateway/tests/`)

The suite boots the real app (`LifespanManager` runs the actual startup:
Postgres pool, Redis, Qdrant, config users) and drives it through
`httpx.ASGITransport` — real routes, middleware, and dependencies, no server
process.

### Key fixtures (`conftest.py`)

- `app` / `client` — the real FastAPI app with its lifespan run once per
  session; an async HTTP client against it.
- `admin_token` / `guest_token` — sessions created through the real
  `/api/v1/login` with the config users from `.env`.
- `seed_book(topic=0|1)` — factory that creates a uniquely-named book:
  rows in `books`/`chapters`/`chunks` plus Qdrant points whose vectors are
  **real bge-m3 embeddings** of a fictional topic (`BOOK_TOPICS`): topic 0 is
  the "Zorbite consolidation algorithm", topic 1 the "Quillmark indexing
  ritual". Fictional content guarantees a semantically matching query
  retrieves *these* chunks and not the dev library. Teardown deletes the
  book everywhere and flushes the `search:*` Redis cache (cached results
  could cite deleted books).
- `fake_llm` — replaces both `build_model` factories. The curation agent
  gets a `FunctionModel` that performs one real `search` tool call (real
  embeddings, real Qdrant) then approves everything; resolver/answer agents
  get `TestModel`. The returned dict steers it per test: `state["book"]`
  scopes the search, `state["query"]` sets the search text,
  `state["decision"] = "NOT_IN_KB"` forces the not-found path.

### test_auth.py — auth & sessions

| Test | Verifies |
|---|---|
| `test_admin_login_returns_admin_session` | `/login` with config admin → 200, `role=admin`, token issued |
| `test_guest_login_returns_user_session` | guest login → `role=user` |
| `test_wrong_password_rejected` | bad password → 401 |
| `test_unknown_user_rejected` | unknown username → 401 |
| `test_protected_route_requires_token` | no `Authorization` header → 401 |
| `test_garbage_token_rejected` | invalid bearer token → 401 |
| `test_guest_cannot_reach_admin_routes` | guest on `/admin/users` → 403 |
| `test_admin_can_reach_admin_routes` | admin on `/admin/users` → 200, lists config users |
| `test_guest_admin_login_forbidden` | `/admin/login` with non-admin → 403 |
| `test_session_persists_across_requests_until_logout` | validate → logout → validate flips `valid` true→false |
| `test_session_subject_roundtrip` | subject set via POST is returned by GET |
| `test_models_endpoint_lists_configured_models` | `/models` reflects `AVAILABLE_MODELS`, default is listed |

### test_chat.py — chat / RAG pipeline

| Test | Verifies |
|---|---|
| `test_chat_single_cites_seeded_book` | a query about the seeded topic returns an answer whose sources cite the seeded book (real retrieval end to end) |
| `test_chat_book_filter_only_cites_that_book` | with two seeded books, a search scoped to book B cites **only** B — the Qdrant filter is a hard guarantee |
| `test_chat_not_in_kb_returns_answer_without_sources` | NOT_IN_KB decision → 200, empty sources, non-empty fallback answer |
| `test_chat_stream_emits_done_and_persists_history` | SSE `/chat`: a `done` event with sources, no `error`, and history afterwards is `[user, assistant]` |
| `test_chat_rejects_when_conversation_full` | with `message_count = 50`, `/chat` emits `error` (`conversation_full`) and no `done` |
| `test_chat_requires_authentication` | `/chat/single` without token → 401 |

Assertions are deliberately coarse ("the book is cited"), never rank- or
score-exact — a bge-m3 update or a fusion algorithm change must not break
them. The cost: a subtle ranking-quality regression that still lands the
right chunk in top-k will pass; that is what the e2e tier and real usage
cover.

### test_books_admin.py — book listing & management

| Test | Verifies |
|---|---|
| `test_seeded_book_appears_in_listings` | `/books` (Qdrant catalog) and `/books/names/list` include the seeded book |
| `test_get_book_details_from_postgres` | `/books/{id}` returns name, chunk count, chapters from PG |
| `test_get_unknown_book_is_404` | unknown book id → 404 |
| `test_admin_delete_removes_book_everywhere` | admin DELETE removes the PG row **and** all Qdrant points, and the listing no longer shows it |
| `test_guest_cannot_delete_book` | guest DELETE → 403 and nothing is removed |
| `test_delete_unknown_book_is_not_a_server_error` | deleting a nonexistent book never 500s |

### test_snapshots.py — snapshot management (disaster recovery)

| Test | Verifies |
|---|---|
| `test_snapshot_roundtrip_restores_deleted_book` | create snapshot → delete a seeded book everywhere → restore → **every Qdrant point (id + payload), the collection total, and the books/chapters rows are byte-identical** to before, and the book is listed again |
| `test_restore_without_metadata_is_rejected` | restoring an unknown snapshot → 400, never a blind restore |
| `test_snapshot_endpoints_require_admin` | guest on snapshot endpoints → 403 |

Restore's contract: Qdrant vectors plus books/chapters metadata come back;
rows in the `chunks` table are **not** part of a snapshot (chat works off
Qdrant payloads, so retrieval is unaffected). Tests write snapshot metadata
to `SNAPSHOT_DIR` (a temp dir — the real `data/qdrant_snapshots` mount is
root-owned). Skipped entirely when `ENABLE_SNAPSHOT_MANAGEMENT` is off.

### test_users_admin.py — database users (the non-config auth path)

| Test | Verifies |
|---|---|
| `test_created_user_can_login_with_db_credentials` | user created via admin API can log in — exercises the bcrypt/DB branch of `authenticate_user` |
| `test_deactivated_user_is_locked_out` | status → `inactive` ⇒ login → 401 |
| `test_duplicate_username_rejected` | duplicate username → 400 |
| `test_role_change_grants_and_revokes_admin_access` | promoting to admin takes effect on the next login; before it, admin routes → 403 |

### test_conversations.py — conversation management

| Test | Verifies |
|---|---|
| `test_conversation_lifecycle` | new → listed with title → renamed → deleted → gone |
| `test_resume_conversation_returns_its_messages` | resume loads the conversation into the session |
| `test_resume_unknown_conversation_is_404` | unknown id → 404 |
| `test_clear_history_empties_the_conversation` | after a chat, `DELETE /chat/history` empties it |

### test_llm_real.py — `-m llm`

`test_real_llm_chat_pipeline` runs `/chat/single` with the *real* models from
`.env` (resolver, curation agent, answer generation) against a seeded book
and asserts the sources cite it — proving the real agent searched and
approved instead of serving the no-context fallback. Skips when the
configured provider has no credentials.

---

## pdf-service (`services/pdf-service/tests/`)

### Key fixtures (`conftest.py`)

- `synthetic_pdf` — a 6-page PDF built with PyMuPDF: real table of contents
  (two chapters), long prose sentences tuned to pass the chunker's
  period-density and min-length filters.
- `stub_chapter_llm` — patches only `get_model_answer_of_chapters` (the LLM
  boundary) to return the two chapter titles; TOC extraction, page-range
  slicing, and chunking all run for real.
- `block_docling` — makes the docling fallback unimportable so a bug in the
  default path can't silently trigger model downloads.
- `book_name` — unique name, purged from PG and Qdrant on teardown.
- `book_counts(name)` — helper returning the book's full footprint (status,
  chapter/chunk rows, vector count) for count-based assertions.

### test_processing.py — the pipeline's observable results

| Test | Verifies |
|---|---|
| `test_processing_stores_chapters_chunks_and_vectors` | processing the synthetic PDF → book `completed`, 2 chapter rows, chunk rows == vectors == reported count |
| `test_reprocessing_replaces_content_without_duplicates` | running the same book twice leaves every count identical — the re-upload purge works (issue #24's headline case) |
| `test_failure_of_all_methods_marks_book_failed` | when chapter detection and the fallback both fail → book `failed`, zero orphan chapters/chunks/vectors |
| `test_upload_accepts_pdf_and_returns_job` | `/upload` accepts a valid PDF and returns the enqueued job id (celery `delay` stubbed — no worker) |
| `test_upload_rejects_non_pdf_extension` | `.txt` upload → 400 |
| `test_upload_rejects_fake_pdf_content` | `.pdf` name with non-PDF bytes → 400 (magic-bytes check) |
| `test_job_status_for_unknown_job_is_pending` | `/job/{unknown}` reports `pending`, not an error |

### test_llm_real.py — `-m llm`

| Test | Verifies |
|---|---|
| `test_real_chapter_detection_reads_toc` | the real model reads the synthetic TOC and returns exactly the two chapters |
| `test_full_pipeline_with_real_chapter_detection` | end-to-end processing with real chapter detection → `completed`, vectors == chunks |

---

## e2e (`tests/e2e/`) — full product, local only

Plain HTTP against `http://localhost` (nginx), no `app` imports. Skips
cleanly when the stack isn't up. Real everything: GPU embeddings, celery
worker, real LLMs. The fixture book (`fixtures/academick-e2e-fixture.pdf`)
contains the fictional "Zorbite consolidation algorithm" so the query test
has a unique retrieval target; it is uploaded once per session and deleted
afterwards.

| Test | Verifies |
|---|---|
| `test_processed_book_is_listed_with_content` | upload → poll job → book listed `completed` with chunks and chapters |
| `test_query_answers_from_uploaded_book` | a chat query about the book's content returns an answer citing it |
| `test_reupload_replaces_instead_of_duplicating` | uploading the same PDF again leaves chunk/chapter counts unchanged |
| `test_guest_cannot_upload` | guest on `/admin/upload-pdfs` → 403 |

Override `E2E_BASE_URL` / `E2E_PROCESS_TIMEOUT` (default 300 s) if needed.

---

## CI (`.github/workflows/ci.yml`)

On every PR/push to `main`/`develop`:

- **pytest matrix** (api-gateway, pdf-service): Postgres/Redis/Qdrant service
  containers, schema from `scripts/init-db.sql`, and the real embedding
  service (plus intent for api-gateway) started on CPU in separate venvs —
  they pin conflicting `transformers` versions. HF models and pip packages
  are cached between runs. Fake API keys only; `llm`/e2e tiers never run.
- **docker build**: only the services whose files changed, with GHA layer
  caching.

## Adding tests

1. Pick the boundary: an HTTP route, or a pipeline entry point whose results
   land in PG/Qdrant. If you're importing a private function, reconsider.
2. Seed through `seed_book` / `synthetic_pdf` or add a similar fixture that
   creates uniquely-named data and tears it down.
3. Assert outcomes a user can observe — status codes, response shape, row and
   vector counts, "book X is cited" — never exact scores or rankings.
4. Real LLM needed? Mark it `@pytest.mark.llm` and skip when the provider
   isn't configured (see `_require_model` / `_require_chapter_model`).
5. New service suite? Create `services/<svc>/tests/` + `pytest.ini` +
   `requirements-dev.txt`; the runner and the CI matrix pick it up (add the
   service name to the matrix).

## Verifying the tests themselves

A test is only trustworthy if it fails when its behavior breaks. When adding
or changing tests, spot-check by mutation: plant the bug the test claims to
guard against (delete the purge, skip the role check, drop the filter), run
the suite, and confirm the exact test goes red — then restore. All four core
guarantees (purge, admin gate, book filter, vector deletion) have been
verified this way.
