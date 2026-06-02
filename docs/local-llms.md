# Using Local / Self-Hosted LLMs

AcademiCK can run entirely against a **self-hosted LLM** instead of (or alongside)
OpenAI, Anthropic, and DeepSeek. Any server that exposes an **OpenAI-compatible
`/v1` API** works — including [vLLM](https://docs.vllm.ai/),
[Ollama](https://ollama.com/), [LM Studio](https://lmstudio.ai/),
[llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server`), and
[LocalAI](https://localai.io/).

No extra dependencies are required: the services already use the `openai` SDK to
talk to any compatible endpoint.

## How model routing works

Every model name carries an explicit **`provider/` prefix** that decides where the
request goes. An unprefixed name defaults to OpenAI.

| Prefix       | Routed to                                 | Example                       |
| ------------ | ----------------------------------------- | ----------------------------- |
| `openai/`    | OpenAI                                     | `openai/gpt-5-mini`           |
| `anthropic/` | Anthropic Claude                           | `anthropic/claude-sonnet-4-6` |
| `deepseek/`  | DeepSeek                                    | `deepseek/deepseek-chat`      |
| `local/`     | Your self-hosted endpoint (`LOCAL_LLM_*`)  | `local/Qwen2.5-7B-Instruct`   |

For a `local/` model, everything after the prefix is the **served model name** —
exactly what your server expects (it may itself contain slashes, e.g.
`local/meta-llama/Llama-3.1-8B-Instruct` → served as
`meta-llama/Llama-3.1-8B-Instruct`).

This prefix is used everywhere a model is selected: the frontend dropdown
(`AVAILABLE_MODELS`), query enhancement, the curation agent, and PDF chapter
detection.

## Configuration

Two environment variables point AcademiCK at your server:

| Variable             | Default                              | Description                                                                 |
| -------------------- | ------------------------------------ | --------------------------------------------------------------------------- |
| `LOCAL_LLM_BASE_URL` | `http://host.docker.internal:8000/v1` | OpenAI-compatible base URL. The default reaches a server on the **same host** as Docker (port 8000). |
| `LOCAL_LLM_API_KEY`  | `EMPTY`                              | Token sent to the server. Most local servers ignore it, but the client requires a non-empty value. Set it if your server enforces a key. |

The Compose file already injects these (with the defaults above) and wires
`host.docker.internal` into the relevant services via `extra_hosts`, so a server
running on the same host as Docker on port 8000 works out of the box — you only
need to add the model to `AVAILABLE_MODELS`.

## Quick start (vLLM on the same host)

### 1. Start a vLLM server

```bash
# On a machine with a GPU
pip install vllm
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
```

This exposes an OpenAI-compatible API at `http://localhost:8000/v1` on the host,
which Docker reaches as `http://host.docker.internal:8000/v1` (the default).

### 2. Add the model to the frontend dropdown

`AVAILABLE_MODELS` is a JSON array of `{provider, value, label}`. The `value` must
carry the `local/` prefix; `provider` and `label` are just display text.

```env
AVAILABLE_MODELS=[{"provider":"Local","value":"local/Qwen/Qwen2.5-7B-Instruct","label":"Qwen 2.5 7B (local)"}]
DEFAULT_MODEL_FRONTEND=local/Qwen/Qwen2.5-7B-Instruct
```

You can mix cloud and local entries in the same list:

```env
AVAILABLE_MODELS=[{"provider":"OpenAI","value":"openai/gpt-5-mini","label":"GPT-5 Mini"},{"provider":"Local","value":"local/Qwen/Qwen2.5-7B-Instruct","label":"Qwen 2.5 7B (local)"}]
```

If your server runs on a different port, host, or as a Compose service, override
the base URL — see [Pointing at a different server](#pointing-at-a-different-server).

### 3. Recreate the containers

`AVAILABLE_MODELS` and the `LOCAL_LLM_*` vars are injected from `.env` at container
**creation** time. `docker compose restart` reuses the old environment — you must
recreate:

```bash
docker compose up -d api-gateway pdf-service pdf-worker
```

Hard-refresh the browser so it re-fetches the model list from `GET /api/v1/models`.

## Other servers

The only thing that changes is the served model name (and the port, if not 8000).

| Server     | Default port | Served name (`local/<name>`)                          |
| ---------- | ------------ | ----------------------------------------------------- |
| vLLM       | 8000         | The HF repo you launched, e.g. `Qwen/Qwen2.5-7B-Instruct` |
| Ollama     | 11434        | The Ollama model tag, e.g. `llama3.1`                 |
| LM Studio  | 1234         | The model identifier shown in LM Studio               |
| llama.cpp  | 8080         | Arbitrary (`llama-server` exposes `/v1`)              |
| LocalAI    | 8080         | The configured model name                             |

## Pointing at a different server

Override `LOCAL_LLM_BASE_URL` in `.env`:

- **Different port on the same host** (e.g. Ollama):
  `LOCAL_LLM_BASE_URL=http://host.docker.internal:11434/v1`
- **A remote server on your network**:
  `LOCAL_LLM_BASE_URL=http://192.168.1.50:8000/v1`
- **Another Compose service** — use the service name, e.g.
  `LOCAL_LLM_BASE_URL=http://vllm:8000/v1`. Example service for `docker-compose.yml`:

  ```yaml
  vllm:
    image: vllm/vllm-openai:latest
    command: ["--model", "Qwen/Qwen2.5-7B-Instruct"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  ```

## Per-stage models

You don't have to use the same model everywhere. Each of these accepts a prefixed
model name independently, so you can keep a cheap cloud model for light stages and
a local model for the main answer, or vice versa:

| Variable                      | Stage                                   | Needs tool calling |
| ----------------------------- | --------------------------------------- | ------------------ |
| `AVAILABLE_MODELS` (selected) | Main answer generation                  | No (plain text)    |
| `QUERY_ENHANCEMENT_MODEL`     | Search-query generation (every query)   | **Yes**            |
| `AGENT_CURATION_MODEL`        | Context curation agent                  | **Yes**            |
| `PDF_CHAPTER_DETECTION_MODEL` | Chapter/heading detection during upload | No (plain text)    |

Example — run everything locally:

```env
QUERY_ENHANCEMENT_MODEL=local/Qwen/Qwen2.5-7B-Instruct
AGENT_CURATION_MODEL=local/Qwen/Qwen2.5-7B-Instruct
PDF_CHAPTER_DETECTION_MODEL=local/Qwen/Qwen2.5-7B-Instruct
```

> PDF chapter detection talks to providers through their OpenAI-compatible
> endpoints, so `anthropic/…` and `deepseek/…` work there too (not just `openai/`
> and `local/`).

### Tool calling is required for the agent stages

The query-enhancement and curation stages run through Pydantic AI with **typed
structured outputs**, which are implemented as **tool/function calls**. A local
model used for `QUERY_ENHANCEMENT_MODEL` or `AGENT_CURATION_MODEL` **must support
tool calling**, and your server must expose it through the OpenAI-compatible API
(e.g. vLLM started with `--enable-auto-tool-choice` and a matching
`--tool-call-parser`). If the model can't call tools, these stages fail.

Pick a tool-calling-capable model (e.g. Qwen2.5-Instruct, Llama 3.1 Instruct,
Mistral/Mixtral Instruct, Hermes) for those stages. The **main answer model** and
**PDF chapter detection** generate plain text and do **not** require tool calling —
if your local model lacks it, you can still use it there while pointing the agent
stages at a capable model (local or cloud).

## Reasoning effort

The `*_REASONING` settings (`none`, `low`, `medium`, `high`) are passed to local
models as the OpenAI `reasoning_effort` parameter when not `none`. Most
self-hosted models **do not** support this and will reject the request. Unless your
served model explicitly supports reasoning effort, leave the reasoning settings at
`none` for local models.

## Provider base URL overrides

If you route a cloud provider through a proxy or compatible gateway, you can
override its base URL (defaults shown):

```env
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
ANTHROPIC_BASE_URL=https://api.anthropic.com/v1/   # pdf-service only
```

## Troubleshooting

**`model '<name>' does not exist` / 404 from `api.openai.com`**
The model name reached the backend **without** a `provider/` prefix and fell
through to the OpenAI default. Make sure every `AVAILABLE_MODELS` value and every
`*_MODEL` setting carries a prefix, and that you ran `docker compose up -d` (not
`restart`) so the container picked up the updated values.

**Gateway refuses to start: "AVAILABLE_MODELS value … must start with a provider prefix"**
This is the startup guard working as intended — fix the offending entry to include
a prefix (e.g. `anthropic/claude-haiku-4-5`, not `claude-haiku-4-5`).

**"A 'local/' model was selected but LOCAL_LLM_BASE_URL is not set."**
You explicitly set `LOCAL_LLM_BASE_URL` to an empty value. Remove the override (to
fall back to the default) or set a real URL, then recreate the containers.

**Connection refused / timeout reaching the local server**
The URL is resolved from inside a container, so `localhost` means the container,
not your host. Use `host.docker.internal` (the default), the host's LAN IP, or the
Compose service name. Confirm the server is reachable from a container:
`docker compose exec api-gateway curl -s $LOCAL_LLM_BASE_URL/models`.

**Empty or truncated answers**
Raise `LLM_MAX_TOKENS`, and make sure your server was launched with a large enough
context/max-model-len for the prompt plus the answer.

**Query enhancement or curation fails (tool/function-call errors)**
The model set for `QUERY_ENHANCEMENT_MODEL` / `AGENT_CURATION_MODEL` doesn't
support tool calling, or your server doesn't expose it. Use a tool-calling-capable
model and enable tool support on the server (for vLLM,
`--enable-auto-tool-choice --tool-call-parser <parser>`). See
[Tool calling is required for the agent stages](#tool-calling-is-required-for-the-agent-stages).
