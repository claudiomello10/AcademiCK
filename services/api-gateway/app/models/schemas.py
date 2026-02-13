"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# ===========================================
# Authentication
# ===========================================

class LoginRequest(BaseModel):
    """Login request."""
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1, max_length=100)


class LoginResponse(BaseModel):
    """Login response."""
    session_id: str
    username: str
    role: str


class SessionResponse(BaseModel):
    """Session validation response."""
    valid: bool
    username: Optional[str] = None
    role: Optional[str] = None
    subject: Optional[str] = None
    conversation_id: Optional[str] = None


# ===========================================
# Chat
# ===========================================

class Intent(str, Enum):
    """Query intent types."""
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"
    CODING = "coding"
    SEARCHING = "searching_for_information"


class ChatRequest(BaseModel):
    """Chat request."""
    query: str = Field(..., min_length=1, max_length=10000)
    model: Optional[str] = None
    book_filter: Optional[str] = None


class SourceChunk(BaseModel):
    """Source chunk from RAG retrieval."""
    text: str
    book: str
    chapter: str
    topic: Optional[str] = None
    score: float


class ChatResponse(BaseModel):
    """Chat response."""
    response: str
    intent: str
    sources: List[SourceChunk]
    model_used: str
    processing_time_ms: float


class MessageHistory(BaseModel):
    """Single message in conversation history."""
    role: str
    content: str
    timestamp: datetime
    intent: Optional[str] = None


class ConversationHistory(BaseModel):
    """Conversation history response."""
    messages: List[MessageHistory]
    subject: Optional[str] = None


# ===========================================
# Books
# ===========================================

class ChapterInfo(BaseModel):
    """Chapter information."""
    id: str
    title: str
    chunk_count: int


class BookInfo(BaseModel):
    """Book information."""
    id: str
    name: str
    total_chunks: int
    chapters: List[ChapterInfo]
    processing_status: str


class BookListResponse(BaseModel):
    """List of available books."""
    books: List[BookInfo]
    total_books: int
    total_chunks: int


# ===========================================
# Admin
# ===========================================

class UserCreate(BaseModel):
    """Create user request."""
    username: str = Field(..., min_length=3, max_length=100)
    email: Optional[str] = None
    password: str = Field(..., min_length=6, max_length=100)
    role: str = Field(default="user")


class UserUpdate(BaseModel):
    """Update user request."""
    email: Optional[str] = None
    role: Optional[str] = None
    status: Optional[str] = None


class UserResponse(BaseModel):
    """User response."""
    id: str
    username: str
    email: Optional[str]
    role: str
    status: str
    is_config_user: bool
    created_at: datetime
    last_active: Optional[datetime]


class ContentStats(BaseModel):
    """Content statistics."""
    total_books: int
    total_chapters: int
    total_chunks: int
    total_users: int
    total_messages: int


class UsageStats(BaseModel):
    """Usage statistics."""
    total_queries: int
    queries_today: int
    average_response_time_ms: float
    queries_by_intent: Dict[str, int]
    active_users_today: int


class ProcessingJobResponse(BaseModel):
    """Processing job status response."""
    job_id: str
    job_type: str
    status: str
    progress: float
    book_name: Optional[str]
    error_message: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]


# ===========================================
# Subject
# ===========================================

class SetSubjectRequest(BaseModel):
    """Set subject request."""
    subject: str = Field(..., min_length=1, max_length=200)


class SubjectResponse(BaseModel):
    """Subject response."""
    subject: str
