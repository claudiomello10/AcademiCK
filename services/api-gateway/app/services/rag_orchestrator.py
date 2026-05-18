"""RAG Orchestrator - Main pipeline for Retrieval-Augmented Generation."""

import asyncio
import time
from typing import Dict, List, Optional, Any
import logging

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from app.clients.intent_client import IntentClient
from app.clients.embedding_client import EmbeddingClient
from app.clients.qdrant_client import QdrantManager
from app.services.search_service import SearchService
from app.services.agent_models import build_model
from app.services.prompt_engineering import (
    get_rag_system_prompt,
    get_enhancement_system_prompt,
)
from app.services.reasoning_agent import CurationAgent
from app.utils.matching import match_book_name
from redis import asyncio as aioredis
from app.config import settings

logger = logging.getLogger(__name__)


class RetrievalQuery(BaseModel):
    """A single search query generated for retrieval."""

    query: str = Field(description="Declarative textbook-style search query.")
    book: Optional[str] = Field(
        default=None,
        description="Exact book name to target, or null to search all books.",
    )


class EnhancedQueries(BaseModel):
    """Structured output of the query enhancement agent."""

    resolved_query: str = Field(
        description=(
            "Student's question with all pronouns and references resolved "
            "using conversation history. Minimal rewrite — no elaboration."
        )
    )
    retrievals: List[RetrievalQuery] = Field(
        default_factory=list,
        description="Up to 3 focused search queries to issue.",
    )


def _to_pydantic_ai_history(
    messages: Optional[List[Dict]], limit: int = 6
) -> List[ModelMessage]:
    """Convert our list-of-dicts conversation history to Pydantic AI messages."""
    if not messages:
        return []
    history: List[ModelMessage] = []
    for msg in messages[-limit:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "assistant":
            history.append(ModelResponse(parts=[TextPart(content=content)]))
        else:
            history.append(ModelRequest(parts=[UserPromptPart(content=content)]))
    return history


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

    async def _generate_enhanced_queries(
        self,
        query: str,
        subject: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Generate focused search queries from the user query via a typed agent.

        Returns dict with "retrievals" (list of {"query", "book"}) and
        "resolved_query" (string with references resolved from history).
        """
        try:
            available_books = await self.qdrant.get_books()

            agent = Agent(
                model=build_model(
                    settings.query_enhancement_model,
                    settings.query_enhancement_reasoning,
                ),
                output_type=EnhancedQueries,
                system_prompt=get_enhancement_system_prompt(subject, available_books),
            )

            result = await agent.run(
                query,
                message_history=_to_pydantic_ai_history(conversation_history),
            )
            output: EnhancedQueries = result.output

            retrievals = [
                {
                    "query": r.query,
                    "book": match_book_name(r.book, available_books) if r.book else None,
                }
                for r in output.retrievals
            ]

            logger.info(
                f"Generated {len(retrievals)} enhanced queries:\n"
                f"Resolved query: '{output.resolved_query}'\n"
                f"Retrievals: {retrievals}"
            )
            return {
                "retrievals": retrievals or [{"query": query, "book": None}],
                "resolved_query": output.resolved_query or query,
            }

        except Exception as e:
            logger.warning(f"Query enhancement failed, using original query: {e}")
            return {
                "retrievals": [{"query": query, "book": None}],
                "resolved_query": query,
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

        # Step 5: Generate final answer via Pydantic AI
        system_prompt = get_rag_system_prompt(
            intent=intent,
            subject=subject,
            context_chunks=curated_chunks
        )
        model_name = model or settings.default_model
        answer_agent = Agent(
            model=build_model(model_name, settings.rag_reasoning),
            output_type=str,
            system_prompt=system_prompt,
        )
        history = _to_pydantic_ai_history(conversation_history)

        response: str = ""
        tokens_used: Optional[int] = None
        try:
            result = await answer_agent.run(query, message_history=history)
            response = result.output or ""
            usage = result.usage()
            tokens_used = usage.total_tokens if usage else None

            # Retry once if the model returned nothing
            if not response:
                logger.warning(
                    f"Empty LLM response on first attempt, retrying: "
                    f"model={model_name}, intent={intent}"
                )
                result = await answer_agent.run(query, message_history=history)
                response = result.output or ""
                usage = result.usage()
                tokens_used = usage.total_tokens if usage else None

            if not response:
                logger.error(
                    f"Empty LLM response after retry: "
                    f"model={model_name}, intent={intent}"
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
            "model_used": model_name,
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