"""HTTP client for Embedding Service."""

import httpx
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Client for the Embedding Service."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=120.0)  # Longer timeout for batch

    async def embed(
        self,
        text: str,
        return_sparse: bool = True
    ) -> Dict:
        """
        Generate embedding for a single text.

        Returns:
            {"dense": [...], "sparse": {...}}
        """
        return await self.embed_batch([text], return_sparse)

    async def embed_batch(
        self,
        texts: List[str],
        return_sparse: bool = True,
        return_colbert: bool = False
    ) -> Dict:
        """
        Generate embeddings for multiple texts.

        Returns:
            {
                "dense_embeddings": [[...], ...],
                "sparse_embeddings": [{...}, ...] (optional)
            }
        """
        try:
            response = await self.client.post(
                f"{self.base_url}/embed",
                json={
                    "texts": texts,
                    "return_sparse": return_sparse,
                    "return_colbert": return_colbert
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    async def health_check(self) -> bool:
        """Check if the embedding service is healthy."""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
