"""OpenAI-compatible chat client factory.

Single place that maps a chapter-detection model name to an OpenAI client.
Routing is by explicit "provider/" name prefix (an unprefixed name defaults to
OpenAI), pointing the client at each provider's OpenAI-compatible endpoint so the
PDF processors keep using a single chat.completions interface:
  - "local/…"     → self-hosted endpoint (vLLM, Ollama, ...)
  - "anthropic/…" → Anthropic OpenAI-compatible endpoint
  - "deepseek/…"  → DeepSeek
  - "openai/…"    → OpenAI
The "provider/" prefix is stripped from the returned model name. Provider base
URLs are configurable via the *_BASE_URL settings.
"""

from openai import OpenAI

from app.config import settings


def build_chat_client(model: str) -> tuple[OpenAI, str]:
    """Return (client, served_model_name) for a model.

    Raises RuntimeError when the selected provider's key or base URL is missing.
    """
    provider, served_name = _resolve_model(model)

    if provider == "local":
        if not settings.local_llm_base_url:
            raise RuntimeError(
                "A 'local/' model was selected but LOCAL_LLM_BASE_URL is not set."
            )
        client = OpenAI(
            base_url=settings.local_llm_base_url,
            api_key=settings.local_llm_api_key,
        )
        return client, served_name

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError(f"Model '{model}' requires ANTHROPIC_API_KEY, which is not set.")
        return OpenAI(base_url=settings.anthropic_base_url, api_key=settings.anthropic_api_key), served_name

    if provider == "deepseek":
        if not settings.deepseek_api_key:
            raise RuntimeError(f"Model '{model}' requires DEEPSEEK_API_KEY, which is not set.")
        return OpenAI(base_url=settings.deepseek_base_url, api_key=settings.deepseek_api_key), served_name

    if not settings.openai_api_key:
        raise RuntimeError(f"Model '{model}' requires OPENAI_API_KEY, which is not set.")
    return OpenAI(api_key=settings.openai_api_key), served_name


_PROVIDER_PREFIXES = {
    "local/": "local",
    "anthropic/": "anthropic",
    "deepseek/": "deepseek",
    "openai/": "openai",
}


def _resolve_model(model: str) -> tuple[str, str]:
    """Map a model name to (provider, served_model_name)."""
    for prefix, provider in _PROVIDER_PREFIXES.items():
        if model.startswith(prefix):
            return provider, model[len(prefix):]
    return "openai", model
