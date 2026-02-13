"""HTTP client for Intent Classification Service."""

import httpx
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class IntentClient:
    """Client for the Intent Classification Service."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=30.0)

    async def classify(self, text: str) -> dict:
        """
        Classify the intent of a query.

        Returns:
            {"intent": str, "confidence": float}
        """
        try:
            response = await self.client.post(
                f"{self.base_url}/classify",
                json={"text": text}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            # Default to question_answering on failure
            return {"intent": "question_answering", "confidence": 0.0}

    async def health_check(self) -> bool:
        """Check if the intent service is healthy."""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
