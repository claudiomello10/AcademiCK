"""RAG Orchestrator - Main pipeline for Retrieval-Augmented Generation."""

import asyncio
import time
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Any
import logging

from app.clients.intent_client import IntentClient
from app.clients.embedding_client import EmbeddingClient
from app.clients.qdrant_client import QdrantManager
from app.services.search_service import SearchService
from app.services.llm_service import LLMService
from app.services.prompt_engineering import get_rag_system_prompt, get_enhanced_query_prompt
from redis import asyncio as aioredis
from app.config import settings

logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """Orchestrates the RAG pipeline for answering queries."""

    def __init__(
        self,
        intent_client: IntentClient,
        embedding_client: EmbeddingClient,
        qdrant: QdrantManager,
        redis: aioredis.Redis
    ):
        self.intent_client = intent_client
        self.qdrant = qdrant
        self.search_service = SearchService(
            qdrant=qdrant,
            embedding_client=embedding_client,
            redis=redis
        )
        self.llm_service = LLMService()

    def _match_book_name(
        self,
        book_name: str,
        available_books: List[str],
        threshold: float = 0.6
    ) -> Optional[str]:
        """
        Match an LLM-produced book name to the closest available book
        using character-level similarity.

        Args:
            book_name: The book name produced by the LLM
            available_books: List of actual book names in Qdrant
            threshold: Minimum similarity ratio (0-1) to accept a match

        Returns:
            The best matching book name, or None if no match above threshold
        """
        if not book_name or not available_books:
            return None

        # Exact match first
        if book_name in available_books:
            return book_name

        # Character-level similarity matching
        book_lower = book_name.lower()
        best_match = None
        best_ratio = 0.0

        for book in available_books:
            ratio = SequenceMatcher(None, book_lower, book.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = book

        if best_ratio >= threshold:
            logger.info(
                f"Fuzzy matched book '{book_name}' -> '{best_match}' "
                f"(similarity: {best_ratio:.2f})"
            )
            return best_match

        logger.warning(
            f"No book match found for '{book_name}' "
            f"(best candidate: '{best_match}', similarity: {best_ratio:.2f})"
        )
        return None

    async def _generate_enhanced_queries(
        self,
        query: str,
        subject: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> List[Dict[str, Optional[str]]]:
        """
        Generate multiple focused search queries from the user query.
        Uses a fast model (GPT-5 Nano) for query enhancement.

        Returns:
            List of {"query": str, "book": Optional[str]}
        """
        try:
            # Get available books
            available_books = await self.qdrant.get_books()

            # Generate enhancement prompt
            prompt = get_enhanced_query_prompt(
                query=query,
                subject=subject,
                available_books=available_books,
                conversation_history=conversation_history
            )

            # Call LLM for query enhancement
            messages = [{"role": "user", "content": prompt}]
            llm_result = await self.llm_service.generate(
                messages=messages,
                model=settings.query_enhancement_model,
                temperature=0.3  # Lower temperature for more focused queries
            )
            response = llm_result["text"] or ""

            logger.debug(f"Query enhancement response: {response}")

            # Parse the XML-like response
            retrievals = []
            for i in range(1, 4):
                pattern = f'<retrieval{i} book="([^"]+)">(.*?)</retrieval{i}>'
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    book = match.group(1).strip()
                    retrieval_query = match.group(2).strip()

                    # Convert "all" to None for no book filter
                    if book.lower() == "all":
                        book = None

                    retrievals.append({
                        "query": retrieval_query,
                        "book": book
                    })

            # Validate book names against available books using fuzzy matching
            for retrieval in retrievals:
                if retrieval["book"] is not None:
                    retrieval["book"] = self._match_book_name(
                        retrieval["book"], available_books
                    )

            logger.info(f"Generated {len(retrievals)} enhanced queries: {retrievals}")
            return retrievals if retrievals else [{"query": query, "book": None}]

        except Exception as e:
            logger.warning(f"Query enhancement failed, using original query: {e}")
            return [{"query": query, "book": None}]

    async def process_query(
        self,
        query: str,
        subject: str = settings.default_subject,
        conversation_history: Optional[List[Dict]] = None,
        model: Optional[str] = None,
        book_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a user query through the RAG pipeline.

        Args:
            query: User's question
            subject: Study subject
            conversation_history: Previous messages for context
            model: LLM model to use
            book_filter: Optional book to filter search

        Returns:
            Dict containing response, intent, sources, and metadata
        """
        start_time = time.time()

        # Step 1: Classify intent
        intent_task = asyncio.create_task(
            self.intent_client.classify(query)
        )

        # Step 2: Generate enhanced queries (concurrent with intent)
        enhanced_queries_task = asyncio.create_task(
            self._generate_enhanced_queries(query, subject, conversation_history)
        )

        # Wait for intent classification
        intent_result = await intent_task
        intent = intent_result.get("intent", "question_answering")

        # Adjust top_k based on intent
        top_k = 12 if intent == "searching_for_information" else 6

        # Wait for enhanced queries
        enhanced_queries = await enhanced_queries_task

        # Step 3: Search with enhanced queries
        search_results = await self.search_service.search_with_enhanced_queries(
            queries=enhanced_queries,
            intent=intent,
            top_k=top_k
        )

        # Step 4: Build prompt with context
        system_prompt = get_rag_system_prompt(
            intent=intent,
            subject=subject,
            context_chunks=search_results
        )

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history (last few messages for context)
        if conversation_history:
            for msg in conversation_history[-6:]:  # Last 6 messages
                if msg.get("role") in ["user", "assistant"]:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

        # Add current query
        messages.append({"role": "user", "content": query})

        # Step 5: Generate response
        tokens_used = None
        try:
            llm_result = await self.llm_service.generate(
                messages=messages,
                model=model,
                temperature=0.7
            )
            response = llm_result["text"]
            tokens_used = llm_result.get("total_tokens")

            # Retry once if response is empty
            if not response:
                logger.warning(
                    f"Empty LLM response on first attempt, retrying: "
                    f"model={model}, intent={intent}"
                )
                llm_result = await self.llm_service.generate(
                    messages=messages,
                    model=model,
                    temperature=0.7
                )
                response = llm_result["text"]
                tokens_used = llm_result.get("total_tokens")

            # Fallback if still empty after retry
            if not response:
                logger.error(
                    f"Empty LLM response after retry: "
                    f"model={model}, intent={intent}"
                )
                response = "Desculpe, não consegui gerar uma resposta. Por favor, tente novamente."
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            response = "I apologize, but I encountered an error generating a response. Please try again."

        processing_time = (time.time() - start_time) * 1000

        return {
            "response": response,
            "tokens_used": tokens_used,
            "intent": intent,
            "sources": [
                {
                    "text": chunk["text"][:500] + "..." if len(chunk["text"]) > 500 else chunk["text"],
                    "book": chunk["book_name"],
                    "chapter": chunk["chapter_title"],
                    "topic": chunk.get("topic"),
                    "score": chunk["score"]
                }
                for chunk in search_results
            ],
            # Full search results with IDs for chunk retrieval tracking (analytics)
            "search_results": search_results,
            "model_used": model or "gpt-5-nano",
            "processing_time_ms": processing_time
        }

    async def process_single_query(
        self,
        query: str,
        subject: str = settings.default_subject,
        model: Optional[str] = None,
        book_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a single query without conversation history.

        Simplified version for one-shot queries.
        """
        return await self.process_query(
            query=query,
            subject=subject,
            conversation_history=None,
            model=model,
            book_filter=book_filter
        )