"""Multi-provider LLM Service."""

from typing import List, Dict, Optional, Any
import logging

from app.config import settings

logger = logging.getLogger(__name__)


def get_provider_for_model(model: str) -> str:
    """Determine the provider for a given model name."""
    model_lower = model.lower()

    if model_lower.startswith("gpt") or model_lower.startswith("text-davinci"):
        return "openai"
    elif model_lower.startswith("claude"):
        return "anthropic"
    elif model_lower.startswith("deepseek"):
        return "deepseek"
    else:
        return "openai"  # Default to OpenAI


class LLMService:
    """Multi-provider LLM inference service."""

    def __init__(self):
        self._openai_client = None
        self._anthropic_client = None

    @property
    def openai_client(self):
        """Lazy load OpenAI client."""
        if self._openai_client is None and settings.openai_api_key:
            from openai import AsyncOpenAI
            self._openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        return self._openai_client

    @property
    def anthropic_client(self):
        """Lazy load Anthropic client."""
        if self._anthropic_client is None and settings.anthropic_api_key:
            from anthropic import AsyncAnthropic
            self._anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        return self._anthropic_client

    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = settings.llm_max_tokens
    ) -> Dict[str, Any]:
        """
        Generate a response using the specified model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name (determines provider)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Dict with 'text', 'total_tokens', 'prompt_tokens', 'completion_tokens'
        """
        model = model or settings.default_model
        provider = get_provider_for_model(model)

        logger.info(f"Generating with model={model}, provider={provider}")

        if provider == "openai":
            return await self._generate_openai(messages, model, temperature, max_tokens)
        elif provider == "anthropic":
            return await self._generate_anthropic(messages, model, temperature, max_tokens)
        elif provider == "deepseek":
            return await self._generate_deepseek(messages, model, temperature, max_tokens)
        else:
            return await self._generate_openai(messages, model, temperature, max_tokens)

    async def _generate_openai(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Generate using OpenAI API."""
        if not self.openai_client:
            raise ValueError("OpenAI API key not configured")

        try:
            response = await self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if not content:
                logger.warning(
                    f"OpenAI returned empty content: model={model}, "
                    f"finish_reason={response.choices[0].finish_reason}, "
                    f"usage={response.usage}"
                )
            usage = response.usage
            return {
                "text": content or "",
                "total_tokens": usage.total_tokens if usage else None,
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
            }

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise

    async def _generate_anthropic(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Generate using Anthropic API."""
        if not self.anthropic_client:
            raise ValueError("Anthropic API key not configured")

        try:
            # Extract system message
            system_content = ""
            user_messages = []

            for msg in messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                else:
                    user_messages.append({"role": msg["role"], "content": msg["content"]})

            response = await self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_content,
                messages=user_messages,
                temperature=temperature
            )
            content = response.content[0].text if response.content else None
            if not content:
                logger.warning(
                    f"Anthropic returned empty content: model={model}, "
                    f"stop_reason={response.stop_reason}, "
                    f"usage={response.usage}"
                )
            usage = response.usage
            return {
                "text": content or "",
                "total_tokens": (usage.input_tokens + usage.output_tokens) if usage else None,
                "prompt_tokens": usage.input_tokens if usage else None,
                "completion_tokens": usage.output_tokens if usage else None,
            }

        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise

    async def _generate_deepseek(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Generate using DeepSeek API (OpenAI-compatible)."""
        if not settings.deepseek_api_key:
            raise ValueError("DeepSeek API key not configured")

        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=settings.deepseek_api_key,
                base_url="https://api.deepseek.com/v1"
            )

            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            content = response.choices[0].message.content
            if not content:
                logger.warning(
                    f"DeepSeek returned empty content: model={model}, "
                    f"finish_reason={response.choices[0].finish_reason}, "
                    f"usage={response.usage}"
                )
            usage = response.usage
            return {
                "text": content or "",
                "total_tokens": usage.total_tokens if usage else None,
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
            }

        except Exception as e:
            logger.error(f"DeepSeek generation failed: {e}")
            raise
