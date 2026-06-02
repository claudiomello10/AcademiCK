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

from app.config import settings


def _provider_for(model_name: str) -> str:
    name = model_name.lower()
    if name.startswith("claude"):
        return "anthropic"
    if name.startswith("deepseek"):
        return "deepseek"
    return "openai"


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

    reasoning_effort is one of "none", "low", "medium", "high". The mapping
    to provider-specific knobs is:
      - OpenAI: passed as openai_reasoning_effort (ignored when "none").
      - Anthropic: mapped to thinking budget_tokens (disabled when "none").
      - DeepSeek: passed as openai_reasoning_effort for reasoner models only.
    """
    provider = _provider_for(model_name)
    max_tokens = max_tokens or settings.llm_max_tokens

    if provider == "anthropic":
        budget = _anthropic_thinking_budget(reasoning_effort, max_tokens)
        model_settings: dict = {"max_tokens": max_tokens}
        if budget is not None:
            model_settings["anthropic_thinking"] = {
                "type": "enabled",
                "budget_tokens": budget,
            }
        return AnthropicModel(model_name, settings=ModelSettings(**model_settings))

    if provider == "deepseek":
        provider_obj = OpenAIProvider(
            base_url="https://api.deepseek.com/v1",
            api_key=settings.deepseek_api_key,
        )
        model_settings = {"max_tokens": max_tokens}
        if reasoning_effort != "none" and "reasoner" in model_name.lower():
            model_settings["openai_reasoning_effort"] = reasoning_effort
        return OpenAIChatModel(
            model_name,
            provider=provider_obj,
            settings=ModelSettings(**model_settings),
        )

    # OpenAI (default)
    model_settings = {"max_completion_tokens": max_tokens}
    if reasoning_effort != "none":
        model_settings["openai_reasoning_effort"] = reasoning_effort
    return OpenAIChatModel(model_name, settings=ModelSettings(**model_settings))