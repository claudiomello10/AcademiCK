"""Model factory for Pydantic AI agents.

Single place where a model name + reasoning effort is translated into a
provider-specific Pydantic AI Model instance with the correct settings.
Every agent in the pipeline obtains its model through `build_model`,
preserving per-stage model selection (config-driven) across OpenAI,
Anthropic, and DeepSeek.
"""

from typing import Optional

from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from app.config import settings, KNOWN_PROVIDER_PREFIXES


# Provider routing by explicit "provider/" name prefix. Using prefixes (rather
# than guessing from the bare model name) keeps routing robust to new model
# names and makes adding a provider a one-line change. An unprefixed name
# defaults to OpenAI. Derived from the canonical KNOWN_PROVIDER_PREFIXES so the
# router and the AVAILABLE_MODELS validator can't drift apart.
_PROVIDER_PREFIXES = {prefix: prefix.rstrip("/") for prefix in KNOWN_PROVIDER_PREFIXES}


def _resolve_model(model_name: str) -> tuple[str, str]:
    """Map a model name to (provider, served_model_name).

    The served name is the model name with its "provider/" prefix removed, i.e.
    what the provider's API actually expects.
    """
    for prefix, provider in _PROVIDER_PREFIXES.items():
        if model_name.startswith(prefix):
            return provider, model_name[len(prefix):]
    return "openai", model_name


def _anthropic_thinking_budget(reasoning_effort: str, max_tokens: int) -> Optional[int]:
    if reasoning_effort == "none":
        return None
    return {
        "low": max(1024, max_tokens // 4),
        "medium": max(2048, max_tokens // 2),
        "high": max(4096, max_tokens),
    }.get(reasoning_effort)


def build_model(
    model_name: str,
    reasoning_effort: str = "none",
    max_tokens: Optional[int] = None,
) -> Model:
    """Build a Pydantic AI Model for the given model name and reasoning effort.

    max_tokens caps the response (defaults to settings.llm_max_tokens) and,
    for Anthropic, scales the thinking budget.

    Routing is by explicit "provider/" name prefix (see _resolve_model); an
    unprefixed name defaults to OpenAI.

    reasoning_effort is one of "none", "low", "medium", "high". The mapping
    to provider-specific knobs is:
      - OpenAI: passed as openai_reasoning_effort (ignored when "none").
      - Anthropic: mapped to thinking budget_tokens (disabled when "none").
      - DeepSeek: passed as openai_reasoning_effort for reasoner models only.
      - local: OpenAI-compatible self-hosted endpoint; reasoning passed as
        openai_reasoning_effort when not "none".
    """
    provider, served_name = _resolve_model(model_name)
    max_tokens = max_tokens or settings.llm_max_tokens

    if provider == "local":
        if not settings.local_llm_base_url:
            raise RuntimeError(
                "A 'local/' model was selected but LOCAL_LLM_BASE_URL is not set."
            )
        provider_obj = OpenAIProvider(
            base_url=settings.local_llm_base_url,
            api_key=settings.local_llm_api_key,
        )
        model_settings = {"max_tokens": max_tokens}
        if reasoning_effort != "none":
            model_settings["openai_reasoning_effort"] = reasoning_effort
        return OpenAIChatModel(
            served_name,
            provider=provider_obj,
            settings=ModelSettings(**model_settings),
        )

    if provider == "anthropic":
        budget = _anthropic_thinking_budget(reasoning_effort, max_tokens)
        model_settings: dict = {"max_tokens": max_tokens}
        if budget is not None:
            model_settings["anthropic_thinking"] = {
                "type": "enabled",
                "budget_tokens": budget,
            }
        return AnthropicModel(served_name, settings=ModelSettings(**model_settings))

    if provider == "deepseek":
        provider_obj = OpenAIProvider(
            base_url=settings.deepseek_base_url,
            api_key=settings.deepseek_api_key,
        )
        model_settings = {"max_tokens": max_tokens}
        if reasoning_effort != "none" and "reasoner" in served_name.lower():
            model_settings["openai_reasoning_effort"] = reasoning_effort
        return OpenAIChatModel(
            served_name,
            provider=provider_obj,
            settings=ModelSettings(**model_settings),
        )

    # OpenAI (default)
    model_settings = {"max_completion_tokens": max_tokens}
    if reasoning_effort != "none":
        model_settings["openai_reasoning_effort"] = reasoning_effort
    return OpenAIChatModel(served_name, settings=ModelSettings(**model_settings))