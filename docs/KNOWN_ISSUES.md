# Known Issues

Newest first.

## Snapshot creation failed with EACCES under non-root containers

Snapshot metadata used to be written into the `./data/qdrant_snapshots` bind
mount, which is root-owned on the host while the api-gateway container runs
as uid 1000 — `POST /snapshots/create` returned 500 (`Permission denied`)
after Qdrant had already created the snapshot, leaving an orphaned snapshot
that could not be restored.

Fixed by giving the gateway its own `snapshot_metadata` named volume: the
gateway never reads snapshot binaries from disk (all snapshot I/O is proxied
over Qdrant's HTTP API), so only the metadata JSONs live there, and a named
volume inherits the image's app-owned directory. Only Qdrant still uses the
host bind mount. Snapshot creation is also atomic now: if the metadata write
fails, the snapshot is deleted instead of being orphaned.

## Migrating an existing deployment to non-root containers

The Python service containers now run as uid 1000 instead of root. Volumes
created by older root images stay root-owned and become read-only for the
services. One-time migration on existing deployments:

```bash
docker compose down
docker volume rm academick_model_cache academick_pdf_uploads   # re-created with correct ownership; models re-download on first start
docker compose up -d --build
```

Fresh installs need no action.

## Curation agent stalls on Anthropic with extended thinking

Structured output (tool calling) + extended thinking stalls Anthropic curation
calls. Fix: `AGENT_CURATION_REASONING=none` (default). On `AGENT_CURATION_TIMEOUT`
the request fails with an error (SSE `error` event / HTTP 504) rather than
answering from uncurated chunks.
