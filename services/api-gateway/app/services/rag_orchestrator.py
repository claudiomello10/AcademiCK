"""RAG Orchestrator - Main pipeline for Retrieval-Augmented Generation."""

import asyncio
import time
import re
from typing import Dict, List, Optional, Any
import logging

from app.clients.intent_client import IntentClient
from app.clients.embedding_client import EmbeddingClient
from app.clients.qdrant_client import QdrantManager
from app.services.search_service import SearchService
from app.services.llm_service import LLMService
from app.services.prompt_engineering import get_rag_system_prompt, get_enhanced_query_prompt
from app.services.reasoning_agent import CurationAgent
from app.utils.matching import match_book_name
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

    async def _generate_enhanced_queries(
        self,
        query: str,
        subject: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Generate multiple focused search queries from the user query.
        Uses a fast model (GPT-5 Nano) for query enhancement.

        Returns:
            Dict with "retrievals" (list of {"query": str, "book": Optional[str]})
            and "resolved_query" (str with references resolved from conversation context)
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
                reasoning_effort=settings.query_enhancement_reasoning
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
                    retrieval["book"] = match_book_name(
                        retrieval["book"], available_books
                    )

            # Extract resolved query (references resolved from conversation context)
            resolved_match = re.search(r'<resolved_query>(.*?)</resolved_query>', response, re.DOTALL)
            resolved_query = resolved_match.group(1).strip() if resolved_match else query

            logger.info(
                f"Generated {len(retrievals)} enhanced queries:\n"
                f"Resolved query: '{resolved_query}'\n"
                f"Retrievals: {retrievals}"
            )
            return {
                "retrievals": retrievals if retrievals else [{"query": query, "book": None}],
                "resolved_query": resolved_query
            }

        except Exception as e:
            logger.warning(f"Query enhancement failed, using original query: {e}")
            return {
                "retrievals": [{"query": query, "book": None}],
                "resolved_query": query
            }

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
        top_k = settings.top_k_searching if intent == "searching_for_information" else settings.top_k_default

        # Wait for enhanced queries
        enhancement_result = await enhanced_queries_task
        enhanced_queries = enhancement_result["retrievals"]
        resolved_query = enhancement_result["resolved_query"]

        # Step 3: Search with enhanced queries
        search_results = await self.search_service.search_with_enhanced_queries(
            queries=enhanced_queries,
            intent=intent,
            top_k=top_k
        )

        # Step 4: Agentic context curation (if enabled)
        agent_result = None
        if settings.agent_enabled and search_results:
            agent = CurationAgent(
                search_service=self.search_service,
                qdrant=self.qdrant
            )
            agent_result = await agent.run(
                query=resolved_query,
                intent=intent,
                subject=subject,
                initial_chunks=search_results,
                top_k=top_k
            )
            curated_chunks = agent_result.final_chunks
        else:
            curated_chunks = search_results

        # Step 5: Build prompt with context
        system_prompt = get_rag_system_prompt(
            intent=intent,
            subject=subject,
            context_chunks=curated_chunks
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
                reasoning_effort=settings.rag_reasoning
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
                    reasoning_effort=settings.rag_reasoning
                )
                response = llm_result["text"]
                tokens_used = llm_result.get("total_tokens")

            # Fallback if still empty after retry
            if not response:
                logger.error(
                    f"Empty LLM response after retry: "
                    f"model={model}, intent={intent}"
                )
                response = "I apologize, but I was unable to generate a response. Please try again."
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
                for chunk in curated_chunks
            ],
            # Full search results with IDs for chunk retrieval tracking (analytics)
            "search_results": search_results,
            "model_used": model or "gpt-5-nano",
            "processing_time_ms": processing_time,
            "agent_iterations": agent_result.iterations_used if agent_result else 0,
            "agent_tokens": agent_result.total_agent_tokens if agent_result else 0,
            "agent_searches": agent_result.total_agent_searches if agent_result else 0,
            "agent_time_ms": agent_result.agent_time_ms if agent_result else 0,
            "reasoning_trace": agent_result.reasoning_trace if agent_result else [],
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