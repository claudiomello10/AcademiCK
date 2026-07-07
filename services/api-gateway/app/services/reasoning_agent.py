"""Curation Agent for Agentic RAG — explores the library via tools, curates context."""

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ToolReturnPart

from app.config import settings
from app.services.agent_models import build_model
from app.services.prompt_engineering import (
    get_curation_system_prompt,
    get_curation_user_prompt,
)
from app.utils.matching import match_book_name

logger = logging.getLogger(__name__)


class CurationTimeoutError(Exception):
    """Raised when a curation model call exceeds AGENT_CURATION_TIMEOUT."""


class SearchQuery(BaseModel):
    """One follow-up search query for the search tool."""

    query: str = Field(description="Declarative, textbook-style search statement.")
    book: Optional[str] = Field(
        default=None,
        description="Exact book name to scope to, or null to search all books.",
    )


class KeepDecision(BaseModel):
    """The curation agent's final decision."""

    action: Literal["APPROVE", "NOT_IN_KB"]
    reasoning: str = Field(description="Brief justification for the decision.")
    keep_indices: List[int] = Field(
        default_factory=list,
        description="1-indexed positions in the context list to keep. Unlisted are dropped.",
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
    not_found: bool = False


@dataclass
class CurationDeps:
    """Mutable state shared with the curation tools during a run."""

    search_service: Any
    qdrant: Any
    library_map: Dict[str, Dict[str, List[str]]]
    intent: str
    top_k: int
    progress: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
    context_pool: List[Dict] = field(default_factory=list)
    seen_hashes: set = field(default_factory=set)
    dropped: set = field(default_factory=set)
    reasoning_trace: List[str] = field(default_factory=list)
    counters: Dict[str, int] = field(default_factory=dict)
    total_tool_calls: int = 0

    def add_chunks(self, new_chunks: List[Dict]) -> List[int]:
        """Append new chunks (deduped by text), return their 1-based pool indices."""
        added: List[int] = []
        for ch in new_chunks:
            h = hashlib.md5(ch.get("text", "").encode()).hexdigest()
            if h in self.seen_hashes:
                continue
            self.seen_hashes.add(h)
            self.context_pool.append(ch)
            added.append(len(self.context_pool))
        return added


# --- Tool helpers ----------------------------------------------------------


def _spend_action(deps: CurationDeps, key: str) -> Optional[str]:
    """Spend one action from the single shared budget. Every tool call costs the
    same — it grows the context and makes later calls more expensive. Returns a
    refusal message when the budget is spent, else counts the call and returns None."""
    if deps.total_tool_calls >= settings.agent_max_actions:
        logger.info(
            f"[CurationAgent] {key}: REFUSED — action budget spent "
            f"({deps.total_tool_calls}/{settings.agent_max_actions})"
        )
        return (
            f"Action budget spent ({deps.total_tool_calls}/{settings.agent_max_actions}). "
            "Return your final decision now with the context you have."
        )
    deps.counters[key] = deps.counters.get(key, 0) + 1
    deps.total_tool_calls += 1
    return None


def _live_indices(deps: CurationDeps) -> List[int]:
    return [i for i in range(1, len(deps.context_pool) + 1) if i not in deps.dropped]


def _status(deps: CurationDeps) -> str:
    return (
        f"[actions: {deps.total_tool_calls}/{settings.agent_max_actions} | "
        f"live chunks: {_live_indices(deps)}]"
    )


def _apply_keep(deps: CurationDeps, keep: List[int]) -> int:
    """Retain only the listed live chunks; soft-delete every other currently-live
    chunk. Runs before new chunks are added, so it only affects chunks already in
    context. Indices stay stable (dropped positions are retired, never reused).
    Returns how many were newly dropped."""
    keep_set = {i for i in keep if 1 <= i <= len(deps.context_pool)}
    dropped_now = 0
    for i in range(1, len(deps.context_pool) + 1):
        if i not in deps.dropped and i not in keep_set:
            deps.dropped.add(i)
            dropped_now += 1
    return dropped_now


def _drop_note(n: int) -> str:
    return f"Dropped {n} unkept chunk(s). " if n else ""


_CHUNK_HEADER_RE = re.compile(r"^\[(\d+)\]")


def _strip_dropped_blocks(text: str, dropped: set) -> str:
    """Collapse dropped chunks' preview blocks in a tool-result string to a stub,
    so they stop consuming tokens when the message history is re-sent."""
    lines = text.split("\n")
    out: List[str] = []
    i = 0
    while i < len(lines):
        m = _CHUNK_HEADER_RE.match(lines[i])
        if m:
            idx = int(m.group(1))
            j = i + 1
            while (
                j < len(lines)
                and not _CHUNK_HEADER_RE.match(lines[j])
                and lines[j].startswith("    ")
            ):
                j += 1
            if idx in dropped:
                out.append(f"[{idx}] (dropped)")
            else:
                out.extend(lines[i:j])
            i = j
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def _make_history_processor(deps: CurationDeps):
    """Prune dropped chunks' previews from re-sent tool results each round."""

    def processor(messages: List[ModelMessage]) -> List[ModelMessage]:
        if not deps.dropped:
            return messages
        for msg in messages:
            for part in getattr(msg, "parts", []):
                if isinstance(part, ToolReturnPart) and isinstance(part.content, str):
                    part.content = _strip_dropped_blocks(part.content, deps.dropped)
        return messages

    return processor


def _format_chunks_at(pool: List[Dict], indices: List[int]) -> str:
    if not indices:
        return "(no new chunks added — all were already in context)"
    lines = []
    for i in indices:
        ch = pool[i - 1]
        page = ch.get("page_number")
        page_str = f" - Page: {page}" if page else ""
        text = ch.get("text", "")
        preview = text[:300] + "..." if len(text) > 300 else text
        lines.append(
            f"[{i}] {ch.get('book_name', 'Unknown')} - "
            f"{ch.get('chapter_title', '')} - {ch.get('topic', '')}{page_str}\n    {preview}"
        )
    return "\n\n".join(lines)


def _index_note(indices: List[int]) -> str:
    if not indices:
        return "(already in your context — nothing new added)"
    nums = ", ".join(f"[{i}]" for i in indices)
    return f"Added to your context list as {nums} — include these in keep_indices if they help answer."


def _synthetic_chunk(text: str, book: str = "") -> Dict:
    return {
        "id": "",
        "score": 1.0,
        "text": text,
        "book_name": book,
        "chapter_title": "",
        "chapter_id": "",
        "topic": "",
        "is_introduction": False,
        "chunk_id": "synthetic",
        "chunk_index": None,
        "page_number": None,
    }


def _fuzzy_contains(keyword: str, text: str, threshold: float = 0.6) -> bool:
    k, t = keyword.lower().strip(), text.lower()
    if not k or not t:
        return False
    if k in t:
        return True
    return SequenceMatcher(None, k, t).ratio() >= threshold


async def _emit(deps: CurationDeps, label: str) -> None:
    if deps.progress is not None:
        await deps.progress({"type": "status", "stage": "curating", "label": label})


# --- Tools -----------------------------------------------------------------


async def search(
    ctx: RunContext[CurationDeps],
    queries: List[SearchQuery],
    keep: List[int],
) -> str:
    """Semantic search over the books — your main way to fetch content. Batch up
    to 3 declarative, textbook-style queries into one call (each may target a
    specific book); one call spends one action regardless of the query count.
    keep=[indices] is REQUIRED: list the currently-live chunks worth retaining;
    every live chunk you omit is dropped. Pass keep=[] on your first call."""
    deps = ctx.deps
    refusal = _spend_action(deps, "search")
    if refusal:
        return refusal

    dropped = _apply_keep(deps, keep)
    await _emit(deps, "Buscando nos livros...")
    available_books = list(deps.library_map.keys())
    resolved = [
        {
            "query": q.query,
            "book": match_book_name(q.book, available_books) if q.book else None,
        }
        for q in queries[: settings.agent_max_queries_per_search]
    ]
    results = await deps.search_service.search_with_enhanced_queries(
        queries=resolved, intent=deps.intent, top_k=deps.top_k
    )
    added = deps.add_chunks(results)
    deps.reasoning_trace.append(
        f"search({[r['query'] for r in resolved]}) -> +{len(added)} chunks, -{dropped} dropped"
    )
    logger.info(
        f"[CurationAgent] ACTION search: queries={[r['query'] for r in resolved]}, "
        f"books={[r['book'] for r in resolved]} -> +{len(added)} chunks, "
        f"-{dropped} dropped {_status(deps)}"
    )
    return (
        f"{_drop_note(dropped)}Added {len(added)} new chunk(s).\n\n"
        f"{_format_chunks_at(deps.context_pool, added)}\n\n{_status(deps)}"
    )


async def list_chapters(
    ctx: RunContext[CurationDeps],
    books: List[str],
    keep: List[int],
    include_topics: bool = False,
) -> str:
    """List the chapters of up to 3 books (table of contents). Set
    include_topics=True to also nest each chapter's topics, resolving the outline
    in one call — use this only when you actually need chapter-level detail, not
    by default. keep=[indices] is REQUIRED: currently-live chunks to retain;
    omitted live chunks are dropped. Spends one action."""
    deps = ctx.deps
    refusal = _spend_action(deps, "list_chapters")
    if refusal:
        return refusal

    dropped = _apply_keep(deps, keep)
    await _emit(deps, "Explorando capítulos...")
    available_books = list(deps.library_map.keys())
    sections = []
    added: List[int] = []
    for name in books[: settings.agent_nav_max_items]:
        book = match_book_name(name, available_books)
        if not book:
            sections.append(f"'{name}': no matching book found.")
            continue
        chapter_map = deps.library_map.get(book, {})
        if include_topics:
            lines = []
            for ch, topics in chapter_map.items():
                if not ch:
                    continue
                lines.append(f"- {ch}")
                lines.extend(f"    • {t}" for t in topics)
            outline = f"Chapters of {book}:\n" + "\n".join(lines)
        else:
            chapters = [c for c in chapter_map if c]
            outline = f"Chapters of {book}:\n" + "\n".join(f"- {c}" for c in chapters)
        sections.append(outline)
        added += deps.add_chunks([_synthetic_chunk(outline, book)])

    text = "\n\n".join(sections)
    deps.reasoning_trace.append(
        f"list_chapters({books}, include_topics={include_topics}) -> outline"
    )
    logger.info(
        f"[CurationAgent] ACTION list_chapters: books={books}, "
        f"include_topics={include_topics}, -{dropped} dropped {_status(deps)}"
    )
    return f"{_drop_note(dropped)}{text}\n\n{_index_note(added)}\n\n{_status(deps)}"


async def list_topics(
    ctx: RunContext[CurationDeps], chapters: List[str], keep: List[int]
) -> str:
    """List the topics covered inside up to 3 chapters. keep=[indices] is
    REQUIRED: currently-live chunks to retain; omitted live chunks are dropped.
    Spends one action."""
    deps = ctx.deps
    refusal = _spend_action(deps, "list_topics")
    if refusal:
        return refusal

    dropped = _apply_keep(deps, keep)
    await _emit(deps, "Explorando tópicos...")
    sections = []
    added: List[int] = []
    for name in chapters[: settings.agent_nav_max_items]:
        matches = []
        for book, chs in deps.library_map.items():
            for ch, topics in chs.items():
                if ch and _fuzzy_contains(name, ch):
                    matches.append((book, ch, topics))
        if not matches:
            sections.append(f"'{name}': no matching chapter found.")
            continue
        for book, ch, topics in matches:
            body = "\n".join(f"- {t}" for t in topics) if topics else "(no topics tagged)"
            outline = f"Topics in {ch} ({book}):\n{body}"
            sections.append(outline)
            added += deps.add_chunks([_synthetic_chunk(outline, book)])

    text = "\n\n".join(sections)
    deps.reasoning_trace.append(f"list_topics({chapters}) -> topics")
    logger.info(
        f"[CurationAgent] ACTION list_topics: chapters={chapters}, "
        f"-{dropped} dropped {_status(deps)}"
    )
    return f"{_drop_note(dropped)}{text}\n\n{_index_note(added)}\n\n{_status(deps)}"


async def read_chapter(
    ctx: RunContext[CurationDeps],
    book: str,
    chapter: str,
    keep: List[int],
    mode: str = "intro",
) -> str:
    """Read a chapter's text. mode='intro' returns its introduction;
    mode='full' returns the whole chapter (token-heavy, may be disabled).
    keep=[indices] is REQUIRED: currently-live chunks to retain; omitted live
    chunks are dropped. Spends one action."""
    deps = ctx.deps
    refusal = _spend_action(deps, "read_chapter")
    if refusal:
        return refusal

    dropped = _apply_keep(deps, keep)
    resolved_book = match_book_name(book, list(deps.library_map.keys()))
    if not resolved_book:
        return f"No matching book for '{book}'. {_status(deps)}"
    resolved_chapter = match_book_name(chapter, list(deps.library_map.get(resolved_book, {}).keys()))
    if not resolved_chapter:
        return f"No matching chapter '{chapter}' in {resolved_book}. {_status(deps)}"

    intro_only = mode != "full"
    if mode == "full" and not settings.agent_read_chapter_full_enabled:
        intro_only = True

    await _emit(deps, "Lendo capítulo...")
    chunks = await deps.qdrant.get_chapter_chunks(
        resolved_book, resolved_chapter, intro_only=intro_only
    )
    added = deps.add_chunks(chunks)
    deps.reasoning_trace.append(
        f"read_chapter({resolved_book}/{resolved_chapter}, {'intro' if intro_only else 'full'}) "
        f"-> +{len(added)} chunks"
    )
    logger.info(
        f"[CurationAgent] ACTION read_chapter: book='{resolved_book}', "
        f"chapter='{resolved_chapter}', mode={'intro' if intro_only else 'full'} "
        f"-> +{len(added)} chunks, -{dropped} dropped {_status(deps)}"
    )
    return (
        f"{_drop_note(dropped)}Read {resolved_chapter} ({resolved_book}), "
        f"added {len(added)} chunk(s).\n\n"
        f"{_format_chunks_at(deps.context_pool, added)}\n\n"
        f"{_status(deps)}"
    )


async def expand_context(
    ctx: RunContext[CurationDeps],
    chunk: int,
    keep: List[int],
    window: int = 1,
) -> str:
    """Pull the chunks adjacent to a context chunk (by its 1-indexed position)
    when a passage looks cut off at a boundary. keep=[indices] is REQUIRED:
    currently-live chunks to retain (include the chunk you're expanding); omitted
    live chunks are dropped. Spends one action."""
    deps = ctx.deps
    refusal = _spend_action(deps, "expand_context")
    if refusal:
        return refusal

    dropped = _apply_keep(deps, keep)
    if chunk < 1 or chunk > len(deps.context_pool):
        return f"Index {chunk} is out of range (1..{len(deps.context_pool)}). {_status(deps)}"

    target = deps.context_pool[chunk - 1]
    await _emit(deps, "Expandindo contexto...")
    neighbours = await deps.qdrant.get_adjacent_chunks(
        target.get("chapter_id", ""), target.get("chunk_index"), window=window
    )
    added = deps.add_chunks(neighbours)
    deps.reasoning_trace.append(
        f"expand_context({chunk}) -> +{len(added)} chunks, -{dropped} dropped"
    )
    logger.info(
        f"[CurationAgent] ACTION expand_context: chunk={chunk}, window={window} "
        f"-> +{len(added)} chunks, -{dropped} dropped {_status(deps)}"
    )
    return (
        f"{_drop_note(dropped)}Added {len(added)} adjacent chunk(s).\n\n"
        f"{_format_chunks_at(deps.context_pool, added)}\n\n"
        f"{_status(deps)}"
    )


class CurationAgent:
    """Evaluates retrieved chunks, explores the library via tools, drops noise,
    and approves a curated context for the main LLM."""

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
        except CurationTimeoutError:
            raise
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

        library_map = await self.qdrant.get_library_map()
        deps = CurationDeps(
            search_service=self.search_service,
            qdrant=self.qdrant,
            library_map=library_map,
            intent=intent,
            top_k=top_k,
            progress=progress,
        )
        deps.add_chunks(initial_chunks)

        agent = Agent(
            model=build_model(
                settings.agent_curation_model,
                settings.agent_curation_reasoning,
                max_tokens=settings.agent_curation_max_tokens,
            ),
            deps_type=CurationDeps,
            output_type=KeepDecision,
            system_prompt=get_curation_system_prompt(subject, list(library_map.keys())),
            tools=[search, list_chapters, list_topics, read_chapter, expand_context],
            history_processors=[_make_history_processor(deps)],
        )

        user_prompt = get_curation_user_prompt(query=query, context_chunks=deps.context_pool)

        if progress is not None:
            await progress({"type": "status", "stage": "curating", "label": "Curando contexto..."})

        logger.info(
            f"[CurationAgent] start (model={settings.agent_curation_model}, "
            f"pool={len(deps.context_pool)} chunks, books={len(library_map)})"
        )
        try:
            result = await asyncio.wait_for(
                agent.run(user_prompt, deps=deps),
                timeout=settings.agent_curation_timeout,
            )
        except asyncio.TimeoutError:
            logger.error(
                f"[CurationAgent] timed out after {settings.agent_curation_timeout:.0f}s"
            )
            raise CurationTimeoutError(
                f"A curadoria de contexto excedeu o tempo limite de "
                f"{settings.agent_curation_timeout:.0f}s. Tente novamente."
            )

        decision: KeepDecision = result.output
        usage = result.usage
        total_tokens = (usage.total_tokens or 0) if usage else 0
        searches = deps.counters.get("search", 0)
        reasoning_trace = list(deps.reasoning_trace)
        reasoning_trace.append(f"{decision.action}: {decision.reasoning}")

        logger.info(
            f"[CurationAgent] done: action={decision.action}, "
            f"tool_calls={deps.total_tool_calls} ({dict(deps.counters)}), "
            f"keep_indices={decision.keep_indices}, dropped={len(deps.dropped)}, "
            f"pool={len(deps.context_pool)}, tokens={total_tokens}"
        )
        logger.info(f"[CurationAgent] reasoning: {decision.reasoning}")

        if decision.action == "NOT_IN_KB":
            return AgentResult(
                final_chunks=[],
                iterations_used=searches,
                reasoning_trace=reasoning_trace,
                total_agent_tokens=total_tokens,
                total_agent_searches=searches,
                agent_time_ms=(time.time() - agent_start) * 1000,
                not_found=True,
            )

        keep = set(decision.keep_indices) - deps.dropped
        final_chunks = [c for i, c in enumerate(deps.context_pool, 1) if i in keep]
        if not final_chunks:
            survivors = [
                c for i, c in enumerate(deps.context_pool, 1) if i not in deps.dropped
            ]
            if survivors:
                final_chunks = survivors
                reasoning_trace.append("Empty keep set — fell back to surviving pool.")

        if len(final_chunks) > settings.agent_max_context_chunks:
            final_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
            final_chunks = final_chunks[: settings.agent_max_context_chunks]

        return AgentResult(
            final_chunks=final_chunks,
            iterations_used=searches,
            reasoning_trace=reasoning_trace,
            total_agent_tokens=total_tokens,
            total_agent_searches=searches,
            agent_time_ms=(time.time() - agent_start) * 1000,
        )
