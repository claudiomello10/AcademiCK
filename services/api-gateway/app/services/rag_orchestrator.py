"""RAG Orchestrator - Main pipeline for Retrieval-Augmented Generation."""

import asyncio
import time
from typing import Awaitable, Callable, Dict, List, Optional, Any
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
from redis import asyncio as aioredis
from app.config import settings

logger = logging.getLogger(__name__)


class ResolvedQuery(BaseModel):
    """Structured output of the query resolution agent."""

    resolved_query: str = Field(
        description=(
            "Student's question with all pronouns and references resolved "
            "using conversation history. Minimal rewrite — no elaboration."
        )
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


# Progress callback type: producers in the pipeline emit small dict events.
# The SSE endpoint provides a queue-feeding callback; non-streaming callers
# pass None and get a no-op.
ProgressCallback = Callable[[Dict[str, Any]], Awaitable[None]]


async def _noop_progress(_: Dict[str, Any]) -> None:
    pass


# Portuguese labels emitted with each stage. The UI just renders these
# verbatim — backend can tweak copy without a frontend ship.
STAGE_LABELS = {
    "intent": "Entendendo pergunta...",
    "enhancing": "Interpretando a pergunta...",
    "generating": "Gerando resposta...",
}

NO_CONTEXT_MESSAGE = (
    "Não encontrei informações relevantes nos livros disponíveis para "
    "responder a sua pergunta. Tente reformular a pergunta ou perguntar "
    "sobre outro tópico."
)

NO_CLASS_BOOKS_MESSAGE = (
    "Esta turma ainda não tem livros disponíveis. Peça ao professor para "
    "adicionar o material da turma."
)


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

    async def _resolve_query(
        self,
        query: str,
        subject: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """Resolve references in the user query (e.g. "that" -> the actual topic)
        using conversation history. Retrieval itself is handled by the curation
        agent's tools, so this only summarizes/resolves the task."""
        try:
            agent = Agent(
                model=build_model(
                    settings.query_enhancement_model,
                    settings.query_enhancement_reasoning,
                ),
                output_type=ResolvedQuery,
                system_prompt=get_enhancement_system_prompt(subject),
            )

            result = await agent.run(
                query,
                message_history=_to_pydantic_ai_history(conversation_history),
            )
            output: ResolvedQuery = result.output
            logger.info(f"Resolved query: '{output.resolved_query}'")
            return output.resolved_query or query

        except Exception as e:
            logger.warning(f"Query resolution failed, using original query: {e}")
            return query

    async def process_query(
        self,
        query: str,
        subject: str = settings.default_subject,
        conversation_history: Optional[List[Dict]] = None,
        model: Optional[str] = None,
        allowed_books: Optional[List[str]] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """Process a user query through the RAG pipeline.

        `allowed_books` is the class scope: every retrieval path (agent tools,
        Qdrant filters) is restricted to these books. None means unscoped;
        an empty list short-circuits with a fixed message.

        If `progress` is provided, the orchestrator awaits it at each
        pipeline boundary with small dict events of the form
        `{"type": "status"|"token", ...}`. Non-streaming callers pass None
        and the callback is a no-op — behavior is otherwise identical.
        """
        emit = progress or _noop_progress
        start_time = time.time()

        if allowed_books is not None and not allowed_books:
            # The class has no completed books yet — nothing to retrieve from.
            await emit({"type": "token", "text": NO_CLASS_BOOKS_MESSAGE})
            return {
                "response": NO_CLASS_BOOKS_MESSAGE,
                "tokens_used": None,
                "intent": "question_answering",
                "resolved_query": query,
                "sources": [],
                "search_results": [],
                "model_used": model or settings.default_model_frontend,
                "processing_time_ms": (time.time() - start_time) * 1000,
                "agent_actions": 0,
                "agent_tool_calls": {},
                "agent_pool_chunks": 0,
                "agent_dropped_chunks": 0,
                "agent_final_chunks": 0,
                "agent_not_in_kb": True,
                "agent_tokens": 0,
                "agent_time_ms": 0.0,
                "reasoning_trace": ["Class has no books — short-circuited."],
            }

        # Step 1+2: intent classification and query resolution run concurrently.
        await emit({"type": "status", "stage": "intent", "label": STAGE_LABELS["intent"]})
        intent_task = asyncio.create_task(self.intent_client.classify(query))

        await emit({"type": "status", "stage": "enhancing", "label": STAGE_LABELS["enhancing"]})
        resolve_task = asyncio.create_task(
            self._resolve_query(query, subject, conversation_history)
        )

        intent_result = await intent_task
        intent = intent_result.get("intent", "question_answering")
        top_k = (
            settings.top_k_searching
            if intent == "searching_for_information"
            else settings.top_k_default
        )

        resolved_query = await resolve_task

        # Step 3: agentic curation. The agent does all retrieval through its
        # tools (search, list_chapters, ...) and forwards its own status events.
        agent = CurationAgent(
            search_service=self.search_service,
            qdrant=self.qdrant,
        )
        agent_result = await agent.run(
            query=resolved_query,
            intent=intent,
            subject=subject,
            initial_chunks=[],
            top_k=top_k,
            allowed_books=allowed_books,
            progress=emit,
        )
        curated_chunks = agent_result.final_chunks

        # Step 5: stream the final answer. Tokens are emitted live so the
        # UI can render the response as it's generated.
        await emit({
            "type": "status",
            "stage": "generating",
            "label": STAGE_LABELS["generating"],
        })

        model_name = model or settings.default_model_frontend
        response: str = ""
        tokens_used: Optional[int] = None

        if not curated_chunks:
            # No grounding context survived retrieval/curation — serve the fixed
            # "not found" message instead of letting the model answer blind.
            logger.info("No context after retrieval/curation — serving not-found message")
            response = NO_CONTEXT_MESSAGE
            await emit({"type": "token", "text": response})
        else:
            system_prompt = get_rag_system_prompt(
                intent=intent,
                subject=subject,
                context_chunks=curated_chunks,
            )
            answer_agent = Agent(
                model=build_model(model_name, settings.rag_reasoning),
                output_type=str,
                system_prompt=system_prompt,
            )
            history = _to_pydantic_ai_history(conversation_history)

            logger.info(
                f"Generating answer (model={model_name}, "
                f"context={len(curated_chunks)} chunks)"
            )
            answer_start = time.time()
            try:
                parts: List[str] = []
                async with answer_agent.run_stream(
                    query, message_history=history
                ) as run:
                    async for delta in run.stream_text(delta=True):
                        parts.append(delta)
                        await emit({"type": "token", "text": delta})
                    usage = run.usage
                    tokens_used = usage.total_tokens if usage else None
                response = "".join(parts)
                logger.info(
                    f"Answer generated in {(time.time() - answer_start) * 1000:.0f}ms "
                    f"({tokens_used} tokens)"
                )

                # If the stream produced nothing, retry once non-streaming.
                if not response:
                    logger.warning(
                        f"Empty streamed response, retrying non-streaming: "
                        f"model={model_name}, intent={intent}"
                    )
                    result = await answer_agent.run(query, message_history=history)
                    response = result.output or ""
                    usage = result.usage
                    tokens_used = usage.total_tokens if usage else None
                    if response:
                        await emit({"type": "token", "text": response})

                if not response:
                    logger.error(
                        f"Empty LLM response after retry: "
                        f"model={model_name}, intent={intent}"
                    )
                    response = "I apologize, but I was unable to generate a response. Please try again."
                    await emit({"type": "token", "text": response})
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                response = "I apologize, but I encountered an error generating a response. Please try again."
                await emit({"type": "token", "text": response})

        processing_time = (time.time() - start_time) * 1000

        return {
            "response": response,
            "tokens_used": tokens_used,
            "intent": intent,
            "resolved_query": resolved_query,
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
            # Curated chunks with IDs for chunk retrieval tracking (analytics).
            # Synthetic outline chunks (no real chunk id) are excluded.
            "search_results": [
                c for c in curated_chunks if c.get("chunk_id") != "synthetic"
            ],
            "model_used": model_name,
            "processing_time_ms": processing_time,
            "agent_actions": agent_result.actions_used,
            "agent_tool_calls": agent_result.tool_calls,
            "agent_pool_chunks": agent_result.pool_chunks,
            "agent_dropped_chunks": agent_result.dropped_chunks,
            "agent_final_chunks": len(curated_chunks),
            "agent_not_in_kb": agent_result.not_found,
            "agent_tokens": agent_result.total_agent_tokens,
            "agent_time_ms": agent_result.agent_time_ms,
            "reasoning_trace": agent_result.reasoning_trace,
        }

    async def process_single_query(
        self,
        query: str,
        subject: str = settings.default_subject,
        model: Optional[str] = None,
        allowed_books: Optional[List[str]] = None
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
            allowed_books=allowed_books
        )