"""Curation Agent for Agentic RAG — evaluates and curates retrieved context."""

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from app.config import settings
from app.services.agent_models import build_model
from app.services.prompt_engineering import (
    get_curation_system_prompt,
    get_curation_user_prompt,
)
from app.utils.matching import match_book_name

logger = logging.getLogger(__name__)


class NewQuery(BaseModel):
    """A follow-up search query the agent wants to issue."""

    query: str = Field(description="Declarative textbook-style search query.")
    book: Optional[str] = Field(
        default=None,
        description="Exact book name to target, or None to search all books.",
    )


class KeepDecision(BaseModel):
    """The curation agent's per-iteration decision."""

    action: Literal["APPROVE", "REFINE"]
    reasoning: str = Field(description="Brief justification for the decision.")
    keep_indices: List[int] = Field(
        default_factory=list,
        description="1-indexed chunk indices to keep. Unlisted chunks are dropped.",
    )
    new_queries: List[NewQuery] = Field(
        default_factory=list,
        description=(
            "Follow-up queries to issue when action is REFINE. "
            "Must be non-empty for REFINE; ignored for APPROVE."
        ),
    )


@dataclass
class AgentResult:
    """Result returned to the orchestrator."""

    final_chunks: List[Dict]
    iterations_used: int
    reasoning_trace: List[str]
    total_agent_tokens: int
    total_agent_searches: int
    agent_time_ms: float


class CurationAgent:
    """Evaluates retrieved chunks, drops noise, optionally issues follow-up
    searches, and approves a curated context for the main LLM."""

    def __init__(self, search_service, qdrant):
        self.search_service = search_service
        self.qdrant = qdrant

    async def run(
        self,
        query: str,
        intent: str,
        subject: str,
        initial_chunks: List[Dict],
        top_k: int,
        progress: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    ) -> AgentResult:
        try:
            return await self._run_loop(
                query=query,
                intent=intent,
                subject=subject,
                initial_chunks=initial_chunks,
                top_k=top_k,
                progress=progress,
            )
        except Exception as e:
            logger.error(f"CurationAgent failed, falling back to single-pass: {e}")
            return AgentResult(
                final_chunks=initial_chunks,
                iterations_used=0,
                reasoning_trace=[f"Agent error: {e}"],
                total_agent_tokens=0,
                total_agent_searches=0,
                agent_time_ms=0.0,
            )

    async def _run_loop(
        self,
        query: str,
        intent: str,
        subject: str,
        initial_chunks: List[Dict],
        top_k: int,
        progress: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    ) -> AgentResult:
        agent_start = time.time()
        context_pool = list(initial_chunks)
        reasoning_trace: List[str] = []
        total_tokens = 0
        total_searches = 0
        max_iterations = settings.agent_max_iterations

        available_books = await self.qdrant.get_books()

        agent = Agent(
            model=build_model(
                settings.agent_curation_model,
                settings.agent_curation_reasoning,
            ),
            output_type=KeepDecision,
            system_prompt=get_curation_system_prompt(subject, available_books),
        )

        iteration = 0
        for iteration in range(1, max_iterations + 1):
            user_prompt = get_curation_user_prompt(
                query=query,
                context_chunks=context_pool,
                iteration=iteration,
                max_iterations=max_iterations,
                previous_reasoning=reasoning_trace,
            )

            result = await agent.run(user_prompt)
            decision: KeepDecision = result.output

            usage = result.usage
            total_tokens += (usage.total_tokens or 0) if usage else 0

            reasoning_trace.append(
                f"[Iter {iteration}] {decision.action}: {decision.reasoning}"
            )
            self._log_decision(iteration, max_iterations, decision)

            if progress is not None:
                if decision.action == "REFINE":
                    label = f"Refinando contexto ({iteration}/{max_iterations})..."
                else:
                    label = f"Curando contexto ({iteration}/{max_iterations})..."
                await progress({
                    "type": "status",
                    "stage": "curating",
                    "label": label,
                    "iteration": iteration,
                    "max_iterations": max_iterations,
                    "action": decision.action,
                    "kept": len(decision.keep_indices),
                    "new_searches": (
                        len(decision.new_queries)
                        if decision.action == "REFINE"
                        else 0
                    ),
                })

            # Apply keep filter (1-indexed). Out-of-range indices silently ignored.
            context_pool = [
                chunk
                for i, chunk in enumerate(context_pool, 1)
                if i in decision.keep_indices
            ]

            # Safety floor: never let the pool go empty if we started with chunks.
            if not context_pool and initial_chunks:
                context_pool = [initial_chunks[0]]
                reasoning_trace.append(
                    f"[Iter {iteration}] Safety floor: kept top initial chunk"
                )

            if decision.action == "APPROVE" or iteration == max_iterations:
                break

            if decision.new_queries:
                resolved_queries = [
                    {
                        "query": nq.query,
                        "book": match_book_name(nq.book, available_books)
                        if nq.book
                        else None,
                    }
                    for nq in decision.new_queries
                ]
                new_results = await self.search_service.search_with_enhanced_queries(
                    queries=resolved_queries,
                    intent=intent,
                    top_k=top_k,
                )
                total_searches += len(resolved_queries)
                context_pool = self._merge_and_deduplicate(context_pool, new_results)

        return AgentResult(
            final_chunks=context_pool,
            iterations_used=iteration,
            reasoning_trace=reasoning_trace,
            total_agent_tokens=total_tokens,
            total_agent_searches=total_searches,
            agent_time_ms=(time.time() - agent_start) * 1000,
        )

    @staticmethod
    def _log_decision(
        iteration: int, max_iterations: int, decision: KeepDecision
    ) -> None:
        if decision.action == "REFINE":
            logger.info(
                f"[CurationAgent] Iter {iteration}/{max_iterations}: REFINE\n"
                f"Reasoning: {decision.reasoning}\n"
                f"Keeping chunks: {decision.keep_indices}\n"
                f"New queries: {[(q.query, q.book) for q in decision.new_queries]}"
            )
        else:
            logger.info(
                f"[CurationAgent] Iter {iteration}/{max_iterations}: "
                f"{decision.action} — keeping {len(decision.keep_indices)} chunks"
            )

    @staticmethod
    def _merge_and_deduplicate(
        existing: List[Dict], new_results: List[Dict]
    ) -> List[Dict]:
        """Merge new search results into the pool, dedup by text hash, cap at max."""
        seen: set = set()
        merged: List[Dict] = []

        for chunk in [*existing, *new_results]:
            text_hash = hashlib.md5(chunk["text"].encode()).hexdigest()
            if text_hash not in seen:
                seen.add(text_hash)
                merged.append(chunk)

        if len(merged) > settings.agent_max_context_chunks:
            merged.sort(key=lambda x: x.get("score", 0), reverse=True)
            merged = merged[: settings.agent_max_context_chunks]

        return merged
