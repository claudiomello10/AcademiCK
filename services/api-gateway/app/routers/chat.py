"""Chat endpoints for RAG conversations."""

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging

from app.models.schemas import (
    ChatRequest, ChatResponse, SourceChunk,
    ConversationHistory, MessageHistory
)
from app.config import settings
from app.services.rag_orchestrator import RAGOrchestrator
from app.services.session_service import ConversationFullError

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
    tokens_consumed: Optional[int] = None
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

        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO usage_stats
                    (user_id, session_id, action_type, response_time_ms, model_used, intent, success, tokens_consumed)
                VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, $7, $8)
                """,
                user_id,
                session_id,
                action_type,
                response_time_ms,
                model_used,
                intent,
                success,
                tokens_consumed
            )
    except Exception as e:
        logger.warning(f"Failed to track usage stats: {e}")


async def get_session_or_error(request: Request, session_id: str):
    """Get session or raise 401 error."""
    session = await request.app.state.session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return session


@router.post("/chat/{session_id}", response_model=ChatResponse)
async def chat(
    request: Request,
    session_id: str,
    chat_request: ChatRequest
):
    """
    Send a message and get a RAG-powered response.

    Uses conversation history for context.
    """
    session = await get_session_or_error(request, session_id)

    # Create RAG orchestrator
    orchestrator = RAGOrchestrator(
        intent_client=request.app.state.intent_client,
        embedding_client=request.app.state.embedding_client,
        qdrant=request.app.state.qdrant,
        redis=request.app.state.redis
    )

    # Get conversation history
    messages = await request.app.state.session_service.get_messages(session_id)

    # Process query
    result = await orchestrator.process_query(
        query=chat_request.query,
        subject=session.get("subject", settings.default_subject),
        conversation_history=messages,
        model=chat_request.model,
        book_filter=chat_request.book_filter
    )

    # Save messages to session with PostgreSQL persistence
    try:
        # Save user message
        await request.app.state.session_service.add_message(
            session_id=session_id,
            role="user",
            content=chat_request.query
        )

        # Save assistant message with metadata and retrieved chunks for analytics
        await request.app.state.session_service.add_message(
            session_id=session_id,
            role="assistant",
            content=result["response"],
            intent=result["intent"],
            model_used=result["model_used"],
            tokens_used=result.get("tokens_used"),
            response_time_ms=int(result["processing_time_ms"]),
            retrieved_chunks=result.get("search_results", [])
        )
    except ConversationFullError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "conversation_full",
                "message": str(e),
                "action": "Please start a new conversation using POST /conversations/{session_id}/new"
            }
        )

    # Track usage stats (use 'query' action_type to match admin stats queries)
    await track_usage(
        db_pool=request.app.state.db_pool,
        user_id=session.get("user_id"),
        session_id=session_id,
        action_type="query",
        response_time_ms=result["processing_time_ms"],
        model_used=result["model_used"],
        intent=result["intent"],
        tokens_consumed=result.get("tokens_used")
    )

    return ChatResponse(
        response=result["response"],
        intent=result["intent"],
        sources=[SourceChunk(**s) for s in result["sources"]],
        model_used=result["model_used"],
        processing_time_ms=result["processing_time_ms"]
    )


@router.post("/chat/{session_id}/single", response_model=ChatResponse)
async def chat_single(
    request: Request,
    session_id: str,
    chat_request: ChatRequest
):
    """
    Send a single query without using conversation history.

    Useful for one-off questions.
    """
    session = await get_session_or_error(request, session_id)

    # Create RAG orchestrator
    orchestrator = RAGOrchestrator(
        intent_client=request.app.state.intent_client,
        embedding_client=request.app.state.embedding_client,
        qdrant=request.app.state.qdrant,
        redis=request.app.state.redis
    )

    # Process query without history
    result = await orchestrator.process_single_query(
        query=chat_request.query,
        subject=session.get("subject", settings.default_subject),
        model=chat_request.model,
        book_filter=chat_request.book_filter
    )

    # Track usage stats
    await track_usage(
        db_pool=request.app.state.db_pool,
        user_id=session.get("user_id"),
        session_id=session_id,
        action_type="query",
        response_time_ms=result["processing_time_ms"],
        model_used=result["model_used"],
        intent=result["intent"],
        tokens_consumed=result.get("tokens_used")
    )

    return ChatResponse(
        response=result["response"],
        intent=result["intent"],
        sources=[SourceChunk(**s) for s in result["sources"]],
        model_used=result["model_used"],
        processing_time_ms=result["processing_time_ms"]
    )


@router.get("/chat/{session_id}/history", response_model=ConversationHistory)
async def get_chat_history(request: Request, session_id: str):
    """Get conversation history for a session."""
    session = await get_session_or_error(request, session_id)

    messages = await request.app.state.session_service.get_messages(session_id)

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


@router.delete("/chat/{session_id}/history")
async def clear_chat_history(request: Request, session_id: str):
    """Clear conversation history for a session."""
    session = await get_session_or_error(request, session_id)

    await request.app.state.session_service.clear_messages(session_id)

    return {"message": "Chat history cleared"}


# ===========================================
# CONVERSATION MANAGEMENT ENDPOINTS
# ===========================================


@router.get("/conversations/{session_id}", response_model=ConversationListResponse)
async def list_conversations(request: Request, session_id: str):
    """
    List all conversations for the authenticated user.

    Returns conversations sorted by most recently updated.
    """
    session = await get_session_or_error(request, session_id)
    user_id = session.get("user_id")

    conversations = await request.app.state.session_service.get_user_conversations(user_id)

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


@router.get("/conversations/{session_id}/resume/{conversation_id}", response_model=ConversationDetailResponse)
async def resume_conversation(request: Request, session_id: str, conversation_id: str):
    """
    Load and resume an old conversation.

    This loads the conversation messages into the current session,
    allowing the user to continue the conversation.
    """
    session = await get_session_or_error(request, session_id)

    result = await request.app.state.session_service.load_conversation(
        session_id=session_id,
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


@router.post("/conversations/{session_id}/new")
async def create_new_conversation(
    request: Request,
    session_id: str,
    body: NewConversationRequest = NewConversationRequest()
):
    """
    Create a new conversation within the current session.

    Use this when the current conversation reaches the 50 message limit,
    or when the user wants to start a fresh conversation on a new topic.
    """
    session = await get_session_or_error(request, session_id)

    conversation_id = await request.app.state.session_service.create_new_conversation(
        session_id=session_id,
        title=body.title
    )

    if not conversation_id:
        raise HTTPException(status_code=500, detail="Failed to create new conversation")

    return {
        "conversation_id": conversation_id,
        "title": body.title or "New Conversation",
        "message": "New conversation created successfully"
    }


@router.put("/conversations/{session_id}/current/title")
async def update_conversation_title(
    request: Request,
    session_id: str,
    body: UpdateTitleRequest
):
    """
    Update the title of the current conversation.
    """
    session = await get_session_or_error(request, session_id)

    success = await request.app.state.session_service.update_conversation_title(
        session_id=session_id,
        title=body.title
    )

    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {"success": True, "title": body.title}


@router.delete("/conversations/{session_id}/{conversation_id}")
async def delete_conversation(
    request: Request,
    session_id: str,
    conversation_id: str
):
    """
    Delete a conversation and all its messages.

    If the deleted conversation is the current one, a new empty conversation is created.
    """
    session = await get_session_or_error(request, session_id)

    success = await request.app.state.session_service.delete_conversation(
        session_id=session_id,
        conversation_id=conversation_id
    )

    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {"success": True, "message": "Conversation deleted"}
