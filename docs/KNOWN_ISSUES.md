# Known Issues

Newest first.

## Curation agent stalls on Anthropic with extended thinking

Structured output (tool calling) + extended thinking stalls Anthropic curation
calls. Fix: `AGENT_CURATION_REASONING=none` (default). On `AGENT_CURATION_TIMEOUT`
the request fails with an error (SSE `error` event / HTTP 504) rather than
answering from uncurated chunks.
