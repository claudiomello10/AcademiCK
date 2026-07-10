"""Chat endpoints for RAG conversations."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.dependencies import get_current_session
from app.models.schemas import (
    ChatRequest, ChatResponse, SourceChunk,
    ConversationHistory, MessageHistory
)
from app.config import settings
from app.services import class_service
from app.services.rag_orchestrator import RAGOrchestrator
from app.services.reasoning_agent import CurationTimeoutError
from app.services.session_service import ConversationFullError


async def resolve_class_scope(request: Request, session: dict) -> tuple[str, List[str]]:
    """The session's active class and its book allowlist; 409 without a class."""
    class_id = session.get("active_class_id")
    if not class_id:
        raise HTTPException(
            status_code=409,
            detail={"code": "no_active_class", "message": "Selecione uma turma antes de conversar."},
        )
    allowed_books = await class_service.get_allowed_book_names(
        request.app.state.db_pool, class_id
    )
    return class_id, allowed_books

router = APIRouter()
logger = logging.getLogger(__name__)


class ConversationInfo(BaseModel):
    id: str
    session_id: Optional[str]
    subject: Optional[str]
    title: str
    message_count: int
    created_at: Optional[str]
    updated_at: Optional[str]


class ConversationListResponse(BaseModel):
    conversations: List[ConversationInfo]


class ConversationDetailResponse(BaseModel):
    conversation_id: str
    subject: Optional[str]
    title: Optional[str]
    message_count: int
    messages: List[MessageHistory]


class NewConversationRequest(BaseModel):
    title: Optional[str] = None


class UpdateTitleRequest(BaseModel):
    title: str


async def track_usage(
    db_pool,
    user_id: str,
    session_id: str,
    action_type: str,
    response_time_ms: int,
    model_used: str,
    intent: str,
    success: bool = True,
    tokens_consumed: Optional[int] = None,
    agent_actions: int = 0,
    agent_tool_calls: Optional[Dict[str, int]] = None,
    agent_pool_chunks: int = 0,
    agent_dropped_chunks: int = 0,
    agent_final_chunks: int = 0,
    agent_not_in_kb: bool = False,
    agent_tokens: int = 0,
    agent_time_ms: int = 0
):
    """Insert usage stats into PostgreSQL."""
    try:
        # Validate user_id is a valid UUID, skip tracking if not
        try:
            from uuid import UUID
            UUID(user_id)
        except (ValueError, TypeError):
            # user_id is not a valid UUID (e.g., 'guest'), skip tracking
            logger.debug(f"Skipping usage tracking for non-UUID user_id: {user_id}")
            return

        tools = agent_tool_calls or {}
        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO usage_stats
                    (user_id, session_id, action_type, response_time_ms, model_used,
                     intent, success, tokens_consumed,
                     agent_actions, agent_search_calls, agent_list_chapters_calls,
                     agent_list_topics_calls, agent_read_chapter_calls,
                     agent_expand_context_calls, agent_pool_chunks,
                     agent_dropped_chunks, agent_final_chunks, agent_not_in_kb,
                     agent_tokens, agent_time_ms)
                VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                        $12, $13, $14, $15, $16, $17, $18, $19, $20)
                """,
                user_id,
                session_id,
                action_type,
                response_time_ms,
                model_used,
                intent,
                success,
                tokens_consumed,
                agent_actions,
                tools.get("search", 0),
                tools.get("list_chapters", 0),
                tools.get("list_topics", 0),
                tools.get("read_chapter", 0),
                tools.get("expand_context", 0),
                agent_pool_chunks,
                agent_dropped_chunks,
                agent_final_chunks,
                agent_not_in_kb,
                agent_tokens,
                agent_time_ms
            )
    except Exception as e:
        logger.warning(f"Failed to track usage stats: {e}")


def _sse(event_type: str, data: Dict[str, Any]) -> str:
    """Format a single Server-Sent Event frame."""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _build_chat_response(result: Dict[str, Any]) -> ChatResponse:
    return ChatResponse(
        response=result["response"],
        intent=result["intent"],
        sources=[SourceChunk(**s) for s in result["sources"]],
        model_used=result["model_used"],
        processing_time_ms=result["processing_time_ms"],
        agent_actions=result.get("agent_actions"),
        agent_tool_calls=result.get("agent_tool_calls"),
        reasoning_trace=(
            result.get("reasoning_trace")
            if settings.reasoning_trace_visible
            else None
        ),
    )


@router.post("/chat")
async def chat(
    request: Request,
    chat_request: ChatRequest,
    session: dict = Depends(get_current_session),
):
    """Send a message and stream a RAG-powered response as SSE.

    Event types emitted in order:
      - status: pipeline stage transitions (intent, enhancing, searching,
                curating, generating).
      - token:  incremental deltas of the final answer.
      - done:   full ChatResponse payload after persistence succeeds.
      - error:  any failure; no messages are persisted in this case.
    """
    session_id = session["session_id"]
    class_id, allowed_books = await resolve_class_scope(request, session)

    orchestrator = RAGOrchestrator(
        intent_client=request.app.state.intent_client,
        embedding_client=request.app.state.embedding_client,
        qdrant=request.app.state.qdrant,
        redis=request.app.state.redis,
    )
    messages = await request.app.state.session_service.get_messages(session_id)

    # Sentinel that terminates the event-drain loop.
    END = object()

    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()

        async def progress(event: Dict[str, Any]) -> None:
            await queue.put(event)

        async def runner() -> None:
            try:
                result = await orchestrator.process_query(
                    query=chat_request.query,
                    subject=session.get("subject", settings.default_subject),
                    conversation_history=messages,
                    model=chat_request.model,
                    allowed_books=allowed_books,
                    progress=progress,
                )

                # Persist after the pipeline succeeds. On any error here,
                # we emit an error event and skip the done event so the
                # client knows nothing was saved.
                try:
                    await request.app.state.session_service.add_message(
                        session_id=session_id,
                        role="user",
                        content=chat_request.query,
                    )
                    await request.app.state.session_service.add_message(
                        session_id=session_id,
                        role="assistant",
                        content=result["response"],
                        intent=result["intent"],
                        model_used=result["model_used"],
                        tokens_used=result.get("tokens_used"),
                        response_time_ms=int(result["processing_time_ms"]),
                        retrieved_chunks=result.get("search_results", []),
                    )
                except ConversationFullError as e:
                    await queue.put({
                        "type": "error",
                        "code": "conversation_full",
                        "message": str(e),
                    })
                    return

                await track_usage(
                    db_pool=request.app.state.db_pool,
                    user_id=session.get("user_id"),
                    session_id=session_id,
                    action_type="query",
                    response_time_ms=result["processing_time_ms"],
                    model_used=result["model_used"],
                    intent=result["intent"],
                    tokens_consumed=result.get("tokens_used"),
                    agent_actions=result.get("agent_actions", 0),
                    agent_tool_calls=result.get("agent_tool_calls"),
                    agent_pool_chunks=result.get("agent_pool_chunks", 0),
                    agent_dropped_chunks=result.get("agent_dropped_chunks", 0),
                    agent_final_chunks=result.get("agent_final_chunks", 0),
                    agent_not_in_kb=result.get("agent_not_in_kb", False),
                    agent_tokens=result.get("agent_tokens", 0),
                    agent_time_ms=int(result.get("agent_time_ms", 0)),
                )

                payload = _build_chat_response(result).model_dump(mode="json")
                await queue.put({"type": "done", "payload": payload})

            except Exception as e:
                logger.exception("chat stream failed")
                await queue.put({"type": "error", "message": str(e)})
            finally:
                await queue.put(END)

        task = asyncio.create_task(runner())
        try:
            while True:
                event = await queue.get()
                if event is END:
                    break
                event_type = event.pop("type")
                yield _sse(event_type, event)
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/chat/single", response_model=ChatResponse)
async def chat_single(
    request: Request,
    chat_request: ChatRequest,
    session: dict = Depends(get_current_session)
):
    """
    Send a single query without using conversation history.

    Useful for one-off questions.
    """
    session_id = session["session_id"]
    class_id, allowed_books = await resolve_class_scope(request, session)

    # Create RAG orchestrator
    orchestrator = RAGOrchestrator(
        intent_client=request.app.state.intent_client,
        embedding_client=request.app.state.embedding_client,
        qdrant=request.app.state.qdrant,
        redis=request.app.state.redis
    )

    # Process query without history
    try:
        result = await orchestrator.process_single_query(
            query=chat_request.query,
            subject=session.get("subject", settings.default_subject),
            model=chat_request.model,
            allowed_books=allowed_books
        )
    except CurationTimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))

    # Track usage stats
    await track_usage(
        db_pool=request.app.state.db_pool,
        user_id=session.get("user_id"),
        session_id=session_id,
        action_type="query",
        response_time_ms=result["processing_time_ms"],
        model_used=result["model_used"],
        intent=result["intent"],
        tokens_consumed=result.get("tokens_used"),
        agent_actions=result.get("agent_actions", 0),
        agent_tool_calls=result.get("agent_tool_calls"),
        agent_pool_chunks=result.get("agent_pool_chunks", 0),
        agent_dropped_chunks=result.get("agent_dropped_chunks", 0),
        agent_final_chunks=result.get("agent_final_chunks", 0),
        agent_not_in_kb=result.get("agent_not_in_kb", False),
        agent_tokens=result.get("agent_tokens", 0),
        agent_time_ms=int(result.get("agent_time_ms", 0)),
    )

    return ChatResponse(
        response=result["response"],
        intent=result["intent"],
        sources=[SourceChunk(**s) for s in result["sources"]],
        model_used=result["model_used"],
        processing_time_ms=result["processing_time_ms"],
        agent_actions=result.get("agent_actions"),
        agent_tool_calls=result.get("agent_tool_calls"),
        reasoning_trace=result.get("reasoning_trace") if settings.reasoning_trace_visible else None,
    )


@router.get("/chat/history", response_model=ConversationHistory)
async def get_chat_history(
    request: Request,
    session: dict = Depends(get_current_session)
):
    """Get conversation history for a session."""
    messages = await request.app.state.session_service.get_messages(session["session_id"])

    return ConversationHistory(
        messages=[
            MessageHistory(
                role=msg["role"],
                content=msg["content"],
                timestamp=msg["timestamp"],
                intent=msg.get("intent")
            )
            for msg in messages
        ],
        subject=session.get("subject")
    )


@router.delete("/chat/history")
async def clear_chat_history(
    request: Request,
    session: dict = Depends(get_current_session)
):
    """Clear conversation history for a session."""
    await request.app.state.session_service.clear_messages(session["session_id"])

    return {"message": "Chat history cleared"}


# ===========================================
# CONVERSATION MANAGEMENT ENDPOINTS
# ===========================================


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    request: Request,
    class_id: Optional[str] = None,
    session: dict = Depends(get_current_session)
):
    """
    List all conversations for the authenticated user.

    Returns conversations sorted by most recently updated,
    optionally scoped to one class.
    """
    user_id = session.get("user_id")

    conversations = await request.app.state.session_service.get_user_conversations(
        user_id, class_id=class_id
    )

    return ConversationListResponse(
        conversations=[
            ConversationInfo(
                id=conv["id"],
                session_id=conv["session_id"],
                subject=conv["subject"],
                title=conv["title"],
                message_count=conv["message_count"],
                created_at=conv["created_at"],
                updated_at=conv["updated_at"]
            )
            for conv in conversations
        ]
    )


@router.get("/conversations/resume/{conversation_id}", response_model=ConversationDetailResponse)
async def resume_conversation(
    request: Request,
    conversation_id: str,
    session: dict = Depends(get_current_session)
):
    """
    Load and resume an old conversation.

    This loads the conversation messages into the current session,
    allowing the user to continue the conversation.
    """
    result = await request.app.state.session_service.load_conversation(
        session_id=session["session_id"],
        conversation_id=conversation_id
    )

    if not result:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationDetailResponse(
        conversation_id=result["conversation_id"],
        subject=result["subject"],
        title=result["title"],
        message_count=result["message_count"],
        messages=[
            MessageHistory(
                role=msg["role"],
                content=msg["content"],
                timestamp=msg["timestamp"],
                intent=msg.get("intent")
            )
            for msg in result["messages"]
        ]
    )


@router.post("/conversations/new")
async def create_new_conversation(
    request: Request,
    body: NewConversationRequest = NewConversationRequest(),
    session: dict = Depends(get_current_session)
):
    """
    Create a new conversation within the current session.

    Use this when the current conversation reaches the 50 message limit,
    or when the user wants to start a fresh conversation on a new topic.
    """
    conversation_id = await request.app.state.session_service.create_new_conversation(
        session_id=session["session_id"],
        title=body.title
    )

    if not conversation_id:
        raise HTTPException(status_code=500, detail="Failed to create new conversation")

    return {
        "conversation_id": conversation_id,
        "title": body.title or "New Conversation",
        "message": "New conversation created successfully"
    }


@router.put("/conversations/current/title")
async def update_conversation_title(
    request: Request,
    body: UpdateTitleRequest,
    session: dict = Depends(get_current_session)
):
    """
    Update the title of the current conversation.
    """
    success = await request.app.state.session_service.update_conversation_title(
        session_id=session["session_id"],
        title=body.title
    )

    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {"success": True, "title": body.title}


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    request: Request,
    conversation_id: str,
    session: dict = Depends(get_current_session)
):
    """
    Delete a conversation and all its messages.

    If the deleted conversation is the current one, a new empty conversation is created.
    """
    success = await request.app.state.session_service.delete_conversation(
        session_id=session["session_id"],
        conversation_id=conversation_id
    )

    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {"success": True, "message": "Conversation deleted"}
