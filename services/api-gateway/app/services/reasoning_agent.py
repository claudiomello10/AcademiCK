"""Curation Agent for Agentic RAG — evaluates and curates retrieved context."""

import re
import hashlib
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Any
import logging

from app.config import settings
from app.services.prompt_engineering import get_curation_evaluation_prompt

logger = logging.getLogger(__name__)


@dataclass
class AgentAction:
    """Parsed result of the curation agent's XML response."""
    type: str  # "APPROVE" or "REFINE"
    reasoning: str
    keep_indices: List[int]  # 1-indexed chunk indices to keep
    new_queries: List[Dict[str, Optional[str]]] = field(default_factory=list)


@dataclass
class AgentResult:
    """Result returned to the orchestrator."""
    final_chunks: List[Dict]
    iterations_used: int
    reasoning_trace: List[str]
    total_agent_tokens: int


def _fail_safe_approve(num_chunks: int, reason: str) -> AgentAction:
    """Return an APPROVE action that keeps all chunks (fail-safe)."""
    return AgentAction(
        type="APPROVE",
        reasoning=reason,
        keep_indices=list(range(1, num_chunks + 1))
    )


def _resolve_action_type(raw_type: str) -> Optional[str]:
    """
    Resolve action type: exact match first, then fuzzy match.
    Returns "APPROVE" or "REFINE", or None if unrecognizable.
    """
    upper = raw_type.strip().upper()
    valid_types = ["APPROVE", "REFINE"]

    # Exact match
    if upper in valid_types:
        return upper

    # Fuzzy match
    best_type = None
    best_ratio = 0.0
    for vt in valid_types:
        ratio = SequenceMatcher(None, upper, vt).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_type = vt

    if best_ratio >= 0.5:
        logger.info(f"[CurationAgent] Fuzzy matched action '{raw_type}' -> '{best_type}' (similarity: {best_ratio:.2f})")
        return best_type

    return None


def _extract_action_type(response_text: str) -> Optional[str]:
    """
    Extract and resolve action type from response text.

    1. Try exact regex for <action type="APPROVE|REFINE">
    2. Try flexible regex for <action type="anything"> + fuzzy match
    3. Scan raw text for APPROVE/REFINE keywords
    Returns resolved action type or None.
    """
    # 1. Exact match
    exact_match = re.search(
        r'<action\s+type="(APPROVE|REFINE)">', response_text, re.IGNORECASE
    )
    if exact_match:
        return exact_match.group(1).upper()

    # 2. Flexible regex + fuzzy match on the attribute value
    flexible_match = re.search(
        r'<action\s+type="([^"]+)">', response_text, re.IGNORECASE
    )
    if flexible_match:
        resolved = _resolve_action_type(flexible_match.group(1))
        if resolved:
            return resolved

    # 3. Scan raw text for keywords as last resort before fail-safe
    text_upper = response_text.upper()
    has_approve = "APPROVE" in text_upper or "APPROV" in text_upper
    has_refine = "REFINE" in text_upper or "REFIN" in text_upper
    if has_approve and not has_refine:
        logger.info("[CurationAgent] Extracted action type from raw text: APPROVE")
        return "APPROVE"
    if has_refine and not has_approve:
        logger.info("[CurationAgent] Extracted action type from raw text: REFINE")
        return "REFINE"

    return None


def parse_agent_action(response_text: str, num_chunks: int) -> AgentAction:
    """
    Parse the curation agent's XML response into an AgentAction.

    Uses a 3-tier strategy to extract action type (exact regex → fuzzy match
    → keyword scan), then falls back to APPROVE keeping all chunks.
    """
    try:
        # Resolve action type (exact → fuzzy → keyword scan)
        action_type = _extract_action_type(response_text)
        if not action_type:
            logger.warning("Could not determine action type from agent response, defaulting to APPROVE")
            return _fail_safe_approve(num_chunks, "Parse failure — keeping all chunks")

        # Extract reasoning
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response_text, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        # Extract keep_chunks
        keep_match = re.search(r'<keep_chunks>\[?(.*?)\]?</keep_chunks>', response_text, re.DOTALL)
        keep_indices = []
        if keep_match:
            keep_str = keep_match.group(1).strip()
            if keep_str:
                keep_indices = [int(x.strip()) for x in keep_str.split(",") if x.strip().isdigit()]

        # Extract new_queries (only for REFINE)
        new_queries = []
        if action_type == "REFINE":
            query_matches = re.finditer(
                r'<query\s+book="([^"]*)">(.*?)</query>', response_text, re.DOTALL
            )
            for match in query_matches:
                book = match.group(1).strip()
                query = match.group(2).strip()
                if book.lower() == "all":
                    book = None
                new_queries.append({"query": query, "book": book})

        return AgentAction(
            type=action_type,
            reasoning=reasoning,
            keep_indices=keep_indices,
            new_queries=new_queries
        )

    except Exception as e:
        logger.warning(f"Failed to parse agent action: {e}, defaulting to APPROVE")
        return _fail_safe_approve(num_chunks, "Parse failure — keeping all chunks")


def _match_book_name(
    book_name: str,
    available_books: List[str],
    threshold: float = 0.6
) -> Optional[str]:
    """
    Match an LLM-produced book name to the closest available book
    using character-level similarity.

    Returns the best matching book name, or None if no match above threshold.
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
            f"[CurationAgent] Fuzzy matched book '{book_name}' -> '{best_match}' "
            f"(similarity: {best_ratio:.2f})"
        )
        return best_match

    logger.warning(
        f"[CurationAgent] No book match for '{book_name}' "
        f"(best: '{best_match}', similarity: {best_ratio:.2f})"
    )
    return None


class CurationAgent:
    """
    Curation agent that evaluates retrieved context,
    drops irrelevant chunks, optionally fetches more,
    then approves the curated context for the main LLM.
    """

    def __init__(self, search_service, llm_service, qdrant):
        self.search_service = search_service
        self.llm_service = llm_service
        self.qdrant = qdrant

    async def run(
        self,
        query: str,
        intent: str,
        subject: str,
        initial_chunks: List[Dict],
        conversation_history: Optional[List[Dict]],
        top_k: int
    ) -> AgentResult:
        """
        Run the curation loop over retrieved chunks.

        Returns curated chunks for the main LLM, plus metadata.
        """
        try:
            return await self._run_loop(
                query=query,
                intent=intent,
                subject=subject,
                initial_chunks=initial_chunks,
                conversation_history=conversation_history,
                top_k=top_k
            )
        except Exception as e:
            logger.error(f"CurationAgent failed, falling back to single-pass: {e}")
            return AgentResult(
                final_chunks=initial_chunks,
                iterations_used=0,
                reasoning_trace=[f"Agent error: {e}"],
                total_agent_tokens=0
            )

    async def _run_loop(
        self,
        query: str,
        intent: str,
        subject: str,
        initial_chunks: List[Dict],
        conversation_history: Optional[List[Dict]],
        top_k: int
    ) -> AgentResult:
        context_pool = list(initial_chunks)
        reasoning_trace = []
        total_agent_tokens = 0
        max_iterations = settings.agent_max_iterations

        # Get available books for the prompt and book name matching
        available_books = await self.qdrant.get_books()

        iteration = 0
        for iteration in range(1, max_iterations + 1):
            # Build evaluation prompt
            eval_prompt = get_curation_evaluation_prompt(
                query=query,
                subject=subject,
                context_chunks=context_pool,
                iteration=iteration,
                max_iterations=max_iterations,
                previous_reasoning=reasoning_trace,
                available_books=available_books
            )

            # Call LLM for evaluation
            eval_result = await self.llm_service.generate(
                messages=[{"role": "user", "content": eval_prompt}],
                model=settings.agent_curation_model,
                temperature=settings.agent_curation_temperature
            )
            total_agent_tokens += eval_result.get("total_tokens", 0)

            # Parse action
            action = parse_agent_action(eval_result.get("text", ""), num_chunks=len(context_pool))
            reasoning_trace.append(
                f"[Iter {iteration}] {action.type}: {action.reasoning}"
            )

            logger.info(
                f"[CurationAgent] Iter {iteration}/{max_iterations}: "
                f"{action.type} — keeping {len(action.keep_indices)} chunks, "
                f"{len(action.new_queries)} new queries"
            )

            # Apply keep_chunks filter (1-indexed)
            context_pool = [
                chunk for i, chunk in enumerate(context_pool, 1)
                if i in action.keep_indices
            ]

            # Safety floor: if everything was dropped, keep highest-scored initial chunk
            if not context_pool and initial_chunks:
                context_pool = [initial_chunks[0]]
                reasoning_trace.append(
                    f"[Iter {iteration}] Safety floor: kept top initial chunk"
                )

            if action.type == "APPROVE" or iteration == max_iterations:
                break

            # REFINE: validate book names via fuzzy matching, then search
            if action.new_queries:
                for q in action.new_queries:
                    if q.get("book") is not None:
                        q["book"] = _match_book_name(q["book"], available_books)

                new_results = await self.search_service.search_with_enhanced_queries(
                    queries=action.new_queries,
                    intent=intent,
                    top_k=top_k
                )
                context_pool = self._merge_and_deduplicate(
                    context_pool, new_results
                )

        return AgentResult(
            final_chunks=context_pool,
            iterations_used=iteration,
            reasoning_trace=reasoning_trace,
            total_agent_tokens=total_agent_tokens
        )

    def _merge_and_deduplicate(
        self,
        existing: List[Dict],
        new_results: List[Dict]
    ) -> List[Dict]:
        """Merge new search results into existing pool, dedup by text hash, cap at max."""
        seen_texts = set()
        merged = []

        # Existing chunks first (preserve order)
        for chunk in existing:
            text_hash = hashlib.md5(chunk["text"].encode()).hexdigest()
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                merged.append(chunk)

        # Add new results
        for chunk in new_results:
            text_hash = hashlib.md5(chunk["text"].encode()).hexdigest()
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                merged.append(chunk)

        # Cap at max context chunks (drop lowest scored)
        if len(merged) > settings.agent_max_context_chunks:
            merged.sort(key=lambda x: x.get("score", 0), reverse=True)
            merged = merged[:settings.agent_max_context_chunks]

        return merged
