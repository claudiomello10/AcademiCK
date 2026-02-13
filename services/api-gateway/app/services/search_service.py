"""Search service for hybrid vector search with caching."""

import asyncio
import json
import hashlib
from typing import List, Dict, Optional, Any
from redis import asyncio as aioredis
import logging

from app.clients.qdrant_client import QdrantManager
from app.clients.embedding_client import EmbeddingClient

logger = logging.getLogger(__name__)

# Search weights based on intent
SEARCH_WEIGHTS = {
    "question_answering": {"dense": 0.6, "sparse": 0.4},
    "summarization": {"dense": 0.7, "sparse": 0.3},
    "coding": {"dense": 0.4, "sparse": 0.6},
    "searching_for_information": {"dense": 0.5, "sparse": 0.5},
}


class SearchService:
    """Service for performing hybrid vector search."""

    def __init__(
        self,
        qdrant: QdrantManager,
        embedding_client: EmbeddingClient,
        redis: aioredis.Redis,
        cache_ttl: int = 3600  # 1 hour cache
    ):
        self.qdrant = qdrant
        self.embedding_client = embedding_client
        self.redis = redis
        self.cache_ttl = cache_ttl
        self.cache_prefix = "search:"

    def _cache_key(self, query: str, filters: Dict) -> str:
        """Generate cache key for a search query."""
        data = json.dumps({"query": query, "filters": filters}, sort_keys=True)
        return f"{self.cache_prefix}{hashlib.sha256(data.encode()).hexdigest()[:16]}"

    async def search(
        self,
        query: str,
        intent: str = "question_answering",
        top_k: int = 6,
        book_filter: Optional[str] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search for a query.

        Args:
            query: Search query text
            intent: Query intent for weight selection
            top_k: Number of results to return
            book_filter: Optional book name to filter by
            use_cache: Whether to use Redis cache

        Returns:
            List of search results with scores
        """
        filters = {"book": book_filter, "intent": intent, "top_k": top_k}

        # Check cache
        if use_cache:
            cache_key = self._cache_key(query, filters)
            cached = await self.redis.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return json.loads(cached)

        # Generate embeddings
        try:
            embeddings = await self.embedding_client.embed_batch(
                [query],
                return_sparse=True
            )

            dense_vector = embeddings["dense_embeddings"][0]
            sparse_dict = None

            if embeddings.get("sparse_embeddings"):
                sparse_data = embeddings["sparse_embeddings"][0]
                sparse_dict = {int(k): float(v) for k, v in sparse_data.items()}

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

        # Perform hybrid search
        try:
            results = await self.qdrant.search_hybrid(
                dense_vector=dense_vector,
                sparse_vector=sparse_dict,
                limit=top_k,
                book_filter=book_filter
            )

            # Cache results
            if use_cache and results:
                await self.redis.setex(
                    cache_key,
                    self.cache_ttl,
                    json.dumps(results)
                )

            return results

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Fallback to dense-only search
            return await self.qdrant.search_dense(
                vector=dense_vector,
                limit=top_k,
                book_filter=book_filter
            )

    async def _search_with_embedding(
        self,
        query: str,
        dense_vector: List[float],
        sparse_dict: Optional[Dict[int, float]],
        intent: str,
        top_k: int,
        book_filter: Optional[str],
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search with pre-computed embeddings.

        Args:
            query: Original query text (for cache key)
            dense_vector: Pre-computed dense embedding
            sparse_dict: Pre-computed sparse embedding
            intent: Query intent
            top_k: Number of results
            book_filter: Optional book filter
            use_cache: Whether to use cache

        Returns:
            List of search results
        """
        filters = {"book": book_filter, "intent": intent, "top_k": top_k}

        # Check cache
        if use_cache:
            cache_key = self._cache_key(query, filters)
            cached = await self.redis.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return json.loads(cached)

        # Perform hybrid search with pre-computed embeddings
        try:
            results = await self.qdrant.search_hybrid(
                dense_vector=dense_vector,
                sparse_vector=sparse_dict,
                limit=top_k,
                book_filter=book_filter
            )

            # Cache results
            if use_cache and results:
                cache_key = self._cache_key(query, filters)
                await self.redis.setex(
                    cache_key,
                    self.cache_ttl,
                    json.dumps(results)
                )

            return results

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Fallback to dense-only search
            return await self.qdrant.search_dense(
                vector=dense_vector,
                limit=top_k,
                book_filter=book_filter
            )

    async def search_with_enhanced_queries(
        self,
        queries: List[Dict[str, Any]],
        intent: str,
        top_k: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Search with multiple enhanced queries concurrently and deduplicate results.

        Batches all embeddings in a single call for efficiency.

        Args:
            queries: List of {"query": str, "book": Optional[str]}
            intent: Query intent
            top_k: Results per query

        Returns:
            Deduplicated and ranked results
        """
        if not queries:
            return []

        # Extract all query texts for batch embedding
        query_texts = [q.get("query", "") for q in queries]

        # Batch embed all queries in a single call
        try:
            embeddings = await self.embedding_client.embed_batch(
                query_texts,
                return_sparse=True
            )

            dense_vectors = embeddings["dense_embeddings"]
            sparse_embeddings = embeddings.get("sparse_embeddings", [])

        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise

        # Create search tasks with pre-computed embeddings
        search_tasks = []
        for i, q in enumerate(queries):
            sparse_dict = None
            if sparse_embeddings and i < len(sparse_embeddings):
                sparse_data = sparse_embeddings[i]
                sparse_dict = {int(k): float(v) for k, v in sparse_data.items()}

            search_tasks.append(
                self._search_with_embedding(
                    query=q.get("query", ""),
                    dense_vector=dense_vectors[i],
                    sparse_dict=sparse_dict,
                    intent=intent,
                    top_k=top_k,
                    book_filter=q.get("book")
                )
            )

        # Execute all searches concurrently
        all_search_results = await asyncio.gather(*search_tasks)

        # Deduplicate results
        all_results = []
        seen_texts = set()

        for results in all_search_results:
            for result in results:
                text_hash = hashlib.md5(result["text"].encode()).hexdigest()
                if text_hash not in seen_texts:
                    seen_texts.add(text_hash)
                    all_results.append(result)

        # Sort by score and return top results
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k * 2]

    # Future: ColBERT reranking placeholder
    async def search_with_colbert_rerank(
        self,
        query: str,
        top_k: int = 10,
        use_colbert: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search with optional ColBERT reranking.

        Note: ColBERT reranking is not yet implemented.
        This is a placeholder for future implementation.
        """
        # 1. Dense + Sparse hybrid search
        candidates = await self.search(query, top_k=top_k * 3)

        # 2. Optional ColBERT reranking (future implementation)
        if use_colbert:
            # TODO: Implement ColBERT reranking
            # colbert_scores = await self.colbert_rerank(query, candidates)
            # candidates = self.merge_scores(candidates, colbert_scores)
            pass

        return candidates[:top_k]
