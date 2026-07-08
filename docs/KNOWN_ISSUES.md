# Known Issues

Newest first.

## Migrating an existing deployment to non-root containers

The Python service containers now run as uid 1000 instead of root. Volumes
created by older root images stay root-owned and become read-only for the
services. One-time migration on existing deployments:

```bash
docker compose down
docker volume rm academick_model_cache academick_pdf_uploads   # re-created with correct ownership; models re-download on first start
sudo chown -R 1000 ./data/qdrant_snapshots                     # bind mount shared with Qdrant; api-gateway writes metadata here
docker compose up -d --build
```

Fresh installs need no action.

## Curation agent stalls on Anthropic with extended thinking

Structured output (tool calling) + extended thinking stalls Anthropic curation
calls. Fix: `AGENT_CURATION_REASONING=none` (default). On `AGENT_CURATION_TIMEOUT`
the request fails with an error (SSE `error` event / HTTP 504) rather than
answering from uncurated chunks.
