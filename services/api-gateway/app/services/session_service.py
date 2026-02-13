"""Session management service using Redis + PostgreSQL persistence."""

import json
from datetime import datetime, timedelta
from uuid import uuid4, UUID
from typing import Optional, List, Dict, Any
from redis import asyncio as aioredis
import asyncpg
import logging

from app.config import settings

logger = logging.getLogger(__name__)


class ConversationFullError(Exception):
    """Raised when conversation has reached the 50 message limit."""
    pass


class SessionService:
    """Manages user sessions with Redis cache and PostgreSQL persistence."""

    def __init__(
        self,
        redis: aioredis.Redis,
        db_pool: asyncpg.Pool,
        session_ttl_minutes: int = 30
    ):
        self.redis = redis
        self.db_pool = db_pool
        self.session_ttl = timedelta(minutes=session_ttl_minutes)
        self.prefix = "session:"
        self.max_messages = 50  # Max messages per conversation

    async def create_session(
        self,
        user_id: str,
        username: str,
        role: str
    ) -> str:
        """Create a new session and return the session ID.

        Writes to both Redis (cache) and PostgreSQL (persistence).
        """
        session_id = str(uuid4())
        now = datetime.utcnow()
        expires_at = now + self.session_ttl

        # Create session and conversation in PostgreSQL
        async with self.db_pool.acquire() as conn:
            # Insert session record
            await conn.execute("""
                INSERT INTO sessions (id, user_id, session_token, subject, expires_at, is_active, created_at, last_active)
                VALUES ($1, $2, $3, $4, $5, true, $6, $6)
            """, UUID(session_id), UUID(user_id) if self._is_valid_uuid(user_id) else None,
                session_id, settings.default_subject, expires_at, now)

            # Create initial conversation
            conversation_id = str(uuid4())
            await conn.execute("""
                INSERT INTO conversations (id, session_id, user_id, subject, title, message_count, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, 0, $6, $6)
            """, UUID(conversation_id), UUID(session_id),
                UUID(user_id) if self._is_valid_uuid(user_id) else None,
                settings.default_subject, "New Conversation", now)

        # Create Redis cache
        session_data = {
            "user_id": user_id,
            "username": username,
            "role": role,
            "subject": settings.default_subject,
            "conversation_id": conversation_id,
            "messages": [],
            "created_at": now.isoformat(),
            "last_active": now.isoformat()
        }

        await self.redis.setex(
            f"{self.prefix}{session_id}",
            int(self.session_ttl.total_seconds()),
            json.dumps(session_data)
        )

        # Track user's active session
        await self.redis.set(f"user_session:{user_id}", session_id)

        logger.info(f"Created session for user: {username}")
        return session_id

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data from Redis, falling back to PostgreSQL if expired.

        This enables session resumption after Redis TTL expires.
        """
        # Try Redis first (fast path)
        data = await self.redis.get(f"{self.prefix}{session_id}")

        if data:
            session = json.loads(data)
            # Refresh TTL on access
            await self.redis.expire(
                f"{self.prefix}{session_id}",
                int(self.session_ttl.total_seconds())
            )
            return session

        # Fall back to PostgreSQL (session resumption)
        return await self._load_session_from_postgres(session_id)

    async def _load_session_from_postgres(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session from PostgreSQL and rebuild Redis cache."""
        async with self.db_pool.acquire() as conn:
            # Get session with user info
            session_row = await conn.fetchrow("""
                SELECT s.id, s.user_id, s.subject, s.is_active, s.created_at,
                       u.username, u.role
                FROM sessions s
                LEFT JOIN users u ON s.user_id = u.id
                WHERE s.session_token = $1 AND s.is_active = true
            """, session_id)

            if not session_row:
                return None

            # Get the most recent conversation for this session
            conv_row = await conn.fetchrow("""
                SELECT id, subject, title, message_count
                FROM conversations
                WHERE session_id = $1
                ORDER BY updated_at DESC
                LIMIT 1
            """, UUID(session_id))

            if not conv_row:
                # No conversation exists, create one
                conversation_id = str(uuid4())
                await conn.execute("""
                    INSERT INTO conversations (id, session_id, user_id, subject, title, message_count)
                    VALUES ($1, $2, $3, $4, $5, 0)
                """, UUID(conversation_id), UUID(session_id), session_row['user_id'],
                    session_row['subject'], "Resumed Conversation")
                messages = []
            else:
                conversation_id = str(conv_row['id'])
                # Load messages for this conversation
                messages = await self._load_messages_from_postgres(conn, UUID(conversation_id))

            # Update last_active in PostgreSQL
            now = datetime.utcnow()
            await conn.execute("""
                UPDATE sessions SET last_active = $1, expires_at = $2
                WHERE session_token = $3
            """, now, now + self.session_ttl, session_id)

            # Rebuild session data
            session_data = {
                "user_id": str(session_row['user_id']) if session_row['user_id'] else session_row['username'],
                "username": session_row['username'] or "unknown",
                "role": session_row['role'] or "user",
                "subject": session_row['subject'],
                "conversation_id": conversation_id,
                "messages": messages,
                "created_at": session_row['created_at'].isoformat() if session_row['created_at'] else now.isoformat(),
                "last_active": now.isoformat()
            }

            # Re-cache in Redis
            await self.redis.setex(
                f"{self.prefix}{session_id}",
                int(self.session_ttl.total_seconds()),
                json.dumps(session_data)
            )

            logger.info(f"Resumed session from PostgreSQL: {session_id}")
            return session_data

    async def _load_messages_from_postgres(
        self,
        conn: asyncpg.Connection,
        conversation_id: UUID
    ) -> List[Dict[str, Any]]:
        """Load messages from PostgreSQL for a conversation."""
        rows = await conn.fetch("""
            SELECT role, content, intent, created_at
            FROM messages
            WHERE conversation_id = $1
            ORDER BY created_at ASC
            LIMIT $2
        """, conversation_id, self.max_messages)

        return [
            {
                "role": row['role'],
                "content": row['content'],
                "intent": row['intent'],
                "timestamp": row['created_at'].isoformat() if row['created_at'] else datetime.utcnow().isoformat()
            }
            for row in rows
        ]

    async def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Validate a session and return session info."""
        session = await self.get_session(session_id)

        if session:
            return {
                "valid": True,
                "username": session["username"],
                "role": session["role"],
                "subject": session.get("subject", settings.default_subject),
                "conversation_id": session.get("conversation_id")
            }

        return {"valid": False}

    async def update_session(
        self,
        session_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update session data in Redis cache."""
        session = await self.get_session(session_id)

        if not session:
            return False

        session.update(updates)
        session["last_active"] = datetime.utcnow().isoformat()

        await self.redis.setex(
            f"{self.prefix}{session_id}",
            int(self.session_ttl.total_seconds()),
            json.dumps(session)
        )

        return True

    async def set_subject(self, session_id: str, subject: str) -> bool:
        """Set the subject for a session.

        Writes to both Redis (cache) and PostgreSQL (persistence).
        """
        session = await self.get_session(session_id)
        if not session:
            return False

        # Update Redis cache
        success = await self.update_session(session_id, {"subject": subject})
        if not success:
            return False

        # Persist to PostgreSQL
        try:
            now = datetime.utcnow()
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE sessions SET subject = $1, last_active = $2
                    WHERE session_token = $3
                """, subject, now, session_id)

                conversation_id = session.get("conversation_id")
                if conversation_id and self._is_valid_uuid(conversation_id):
                    await conn.execute("""
                        UPDATE conversations SET subject = $1, updated_at = $2
                        WHERE id = $3
                    """, subject, now, UUID(conversation_id))
        except Exception as e:
            logger.warning(f"Failed to persist subject to PostgreSQL: {e}")

        return True

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: Optional[str] = None,
        model_used: Optional[str] = None,
        tokens_used: Optional[int] = None,
        response_time_ms: Optional[int] = None,
        retrieved_chunks: Optional[List[Dict]] = None
    ) -> Optional[str]:
        """Add a message to the session history.

        Writes to both PostgreSQL (persistence) and Redis (cache).
        Returns the message ID if successful.

        Raises:
            ConversationFullError: If conversation has reached 50 message limit.
        """
        session = await self.get_session(session_id)

        if not session:
            return None

        conversation_id = session.get("conversation_id")
        if not conversation_id:
            logger.error(f"No conversation_id found for session {session_id}")
            return None

        now = datetime.utcnow()
        message_id = str(uuid4())

        async with self.db_pool.acquire() as conn:
            # Check message count first
            count = await conn.fetchval("""
                SELECT message_count FROM conversations WHERE id = $1
            """, UUID(conversation_id))

            if count is not None and count >= self.max_messages:
                raise ConversationFullError(
                    f"Conversation has reached the {self.max_messages} message limit. "
                    "Please start a new conversation."
                )

            # Insert message into PostgreSQL
            await conn.execute("""
                INSERT INTO messages (id, conversation_id, role, content, intent, model_used, tokens_used, response_time_ms, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """, UUID(message_id), UUID(conversation_id), role, content,
                intent, model_used, tokens_used, response_time_ms, now)

            # Update conversation message count
            await conn.execute("""
                UPDATE conversations SET message_count = message_count + 1, updated_at = $1
                WHERE id = $2
            """, now, UUID(conversation_id))

            # Auto-name conversation from the first message
            if count is not None and count == 0:
                first_line = content.split('\n')[0]
                auto_title = first_line[:60] + ("..." if len(first_line) > 60 else "")
                await conn.execute("""
                    UPDATE conversations SET title = $1 WHERE id = $2
                """, auto_title, UUID(conversation_id))

            # Track retrieved chunks for assistant messages (analytics)
            if retrieved_chunks and role == "assistant":
                await self._save_chunk_retrievals(conn, UUID(message_id), retrieved_chunks)

        # Update Redis cache
        message = {
            "role": role,
            "content": content,
            "timestamp": now.isoformat()
        }
        if intent:
            message["intent"] = intent

        session["messages"].append(message)

        # Keep only last N messages in Redis cache
        if len(session["messages"]) > self.max_messages:
            session["messages"] = session["messages"][-self.max_messages:]

        await self.update_session(session_id, {"messages": session["messages"]})

        return message_id

    async def _save_chunk_retrievals(
        self,
        conn: asyncpg.Connection,
        message_id: UUID,
        chunks: List[Dict]
    ) -> None:
        """Save chunk retrieval records for analytics."""
        for position, chunk in enumerate(chunks):
            # Get chunk_id from qdrant_point_id if available
            chunk_id = None
            book_id = None
            chapter_id = None

            qdrant_point_id = chunk.get("id") or chunk.get("qdrant_point_id")
            if qdrant_point_id:
                # Look up the chunk by qdrant_point_id
                chunk_row = await conn.fetchrow("""
                    SELECT id, book_id, chapter_id FROM chunks
                    WHERE qdrant_point_id = $1
                """, UUID(qdrant_point_id) if self._is_valid_uuid(qdrant_point_id) else None)

                if chunk_row:
                    chunk_id = chunk_row['id']
                    book_id = chunk_row['book_id']
                    chapter_id = chunk_row['chapter_id']

            await conn.execute("""
                INSERT INTO chunk_retrievals (message_id, chunk_id, book_id, chapter_id, score, position)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, message_id, chunk_id, book_id, chapter_id,
                chunk.get("score"), position)

    async def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Get message history for a session."""
        session = await self.get_session(session_id)

        if session:
            return session.get("messages", [])

        return []

    async def clear_messages(self, session_id: str) -> bool:
        """Clear message history for a session (Redis only, PostgreSQL preserved)."""
        return await self.update_session(session_id, {"messages": []})

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session (marks as inactive in PostgreSQL, removes from Redis)."""
        session = await self.get_session(session_id)

        if session:
            # Mark session as inactive in PostgreSQL (don't delete for history)
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE sessions SET is_active = false, last_active = $1
                    WHERE session_token = $2
                """, datetime.utcnow(), session_id)

            # Remove user session mapping
            user_id = session.get("user_id")
            if user_id:
                await self.redis.delete(f"user_session:{user_id}")

            # Remove from Redis
            await self.redis.delete(f"{self.prefix}{session_id}")
            return True

        return False

    async def get_active_sessions_count(self) -> int:
        """Get count of active sessions."""
        keys = await self.redis.keys(f"{self.prefix}*")
        return len(keys)

    async def create_new_conversation(
        self,
        session_id: str,
        title: Optional[str] = None
    ) -> Optional[str]:
        """Create a new conversation within an existing session.

        Used when a conversation reaches the 50 message limit.
        Returns the new conversation_id.
        """
        session = await self.get_session(session_id)
        if not session:
            return None

        user_id = session.get("user_id")
        subject = session.get("subject", settings.default_subject)
        now = datetime.utcnow()
        conversation_id = str(uuid4())

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO conversations (id, session_id, user_id, subject, title, message_count, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, 0, $6, $6)
            """, UUID(conversation_id), UUID(session_id),
                UUID(user_id) if self._is_valid_uuid(user_id) else None,
                subject, title or "New Conversation", now)

        # Update Redis session with new conversation
        session["conversation_id"] = conversation_id
        session["messages"] = []
        await self.update_session(session_id, {
            "conversation_id": conversation_id,
            "messages": []
        })

        logger.info(f"Created new conversation {conversation_id} for session {session_id}")
        return conversation_id

    async def get_user_conversations(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get all conversations for a user."""
        if not self._is_valid_uuid(user_id):
            return []

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT c.id, c.session_id, c.subject, c.title, c.message_count,
                       c.created_at, c.updated_at,
                       s.session_token
                FROM conversations c
                LEFT JOIN sessions s ON c.session_id = s.id
                WHERE c.user_id = $1
                ORDER BY c.updated_at DESC
                LIMIT $2
            """, UUID(user_id), limit)

            return [
                {
                    "id": str(row['id']),
                    "session_id": row['session_token'],
                    "subject": row['subject'],
                    "title": row['title'] or f"Conversation from {row['created_at'].strftime('%b %d, %Y')}",
                    "message_count": row['message_count'],
                    "created_at": row['created_at'].isoformat() if row['created_at'] else None,
                    "updated_at": row['updated_at'].isoformat() if row['updated_at'] else None
                }
                for row in rows
            ]

    async def load_conversation(
        self,
        session_id: str,
        conversation_id: str
    ) -> Optional[Dict[str, Any]]:
        """Load a specific conversation into the current session."""
        session = await self.get_session(session_id)
        if not session:
            return None

        user_id = session.get("user_id")

        async with self.db_pool.acquire() as conn:
            # Verify ownership
            conv_row = await conn.fetchrow("""
                SELECT id, subject, title, message_count
                FROM conversations
                WHERE id = $1 AND user_id = $2
            """, UUID(conversation_id),
                UUID(user_id) if self._is_valid_uuid(user_id) else None)

            if not conv_row:
                return None

            # Load messages
            messages = await self._load_messages_from_postgres(conn, UUID(conversation_id))

        # Update session to use this conversation
        session["conversation_id"] = conversation_id
        session["subject"] = conv_row['subject']
        session["messages"] = messages

        await self.update_session(session_id, {
            "conversation_id": conversation_id,
            "subject": conv_row['subject'],
            "messages": messages
        })

        return {
            "conversation_id": conversation_id,
            "subject": conv_row['subject'],
            "title": conv_row['title'],
            "message_count": conv_row['message_count'],
            "messages": messages
        }

    async def update_conversation_title(
        self,
        session_id: str,
        title: str
    ) -> bool:
        """Update the title of the current conversation."""
        session = await self.get_session(session_id)
        if not session:
            return False

        conversation_id = session.get("conversation_id")
        if not conversation_id:
            return False

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE conversations SET title = $1, updated_at = $2
                WHERE id = $3
            """, title[:255], datetime.utcnow(), UUID(conversation_id))

        return True

    async def delete_conversation(
        self,
        session_id: str,
        conversation_id: str
    ) -> bool:
        """Delete a conversation and its messages.

        If the deleted conversation is the current one, creates a new conversation.
        """
        session = await self.get_session(session_id)
        if not session:
            return False

        user_id = session.get("user_id")
        if not conversation_id or not self._is_valid_uuid(conversation_id):
            return False

        async with self.db_pool.acquire() as conn:
            # Verify ownership
            owner = await conn.fetchval("""
                SELECT user_id FROM conversations WHERE id = $1
            """, UUID(conversation_id))

            if owner is None:
                return False
            if self._is_valid_uuid(str(user_id)) and owner != UUID(user_id):
                return False

            # Delete messages first (FK constraint)
            await conn.execute("""
                DELETE FROM chunk_retrievals WHERE message_id IN (
                    SELECT id FROM messages WHERE conversation_id = $1
                )
            """, UUID(conversation_id))
            await conn.execute("""
                DELETE FROM messages WHERE conversation_id = $1
            """, UUID(conversation_id))
            await conn.execute("""
                DELETE FROM conversations WHERE id = $1
            """, UUID(conversation_id))

        # If we deleted the current conversation, create a new one
        if session.get("conversation_id") == conversation_id:
            await self.create_new_conversation(session_id, title="Nova Conversa")

        return True

    def _is_valid_uuid(self, value: str) -> bool:
        """Check if a string is a valid UUID."""
        try:
            UUID(value)
            return True
        except (ValueError, TypeError, AttributeError):
            return False
