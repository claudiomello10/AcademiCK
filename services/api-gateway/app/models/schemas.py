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
    # Agentic RAG metadata (agent_tokens and agent_time_ms stay DB-only for admin analytics)
    agent_actions: Optional[int] = None
    agent_tool_calls: Optional[Dict[str, int]] = None
    reasoning_trace: Optional[List[str]] = None


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

VALID_ROLES = ("user", "professor", "manager", "admin")


class UserCreate(BaseModel):
    """Create user request."""
    username: str = Field(..., min_length=3, max_length=100)
    email: Optional[str] = None
    password: str = Field(..., min_length=6, max_length=100)
    role: str = Field(default="user")
    registration_number: Optional[str] = Field(default=None, max_length=50)


class UserUpdate(BaseModel):
    """Update user request."""
    email: Optional[str] = None
    role: Optional[str] = None
    status: Optional[str] = None
    registration_number: Optional[str] = Field(default=None, max_length=50)


class UserResponse(BaseModel):
    """User response."""
    id: str
    username: str
    email: Optional[str]
    role: str
    status: str
    is_config_user: bool
    registration_number: Optional[str] = None
    created_at: datetime
    last_active: Optional[datetime]


class ImportRowError(BaseModel):
    """A single failed row in a bulk user import."""
    row: int
    username: Optional[str] = None
    error: str


class UserImportResponse(BaseModel):
    """Bulk user import result."""
    created: int
    errors: List[ImportRowError]


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
# Classes
# ===========================================

class ClassCreate(BaseModel):
    """Create class request (professor-owned)."""
    name: str = Field(..., min_length=1, max_length=255)
    subject: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None


class ClassCreateAdmin(ClassCreate):
    """Create class request naming the owning professor (admin/manager)."""
    professor_id: str


class ClassUpdate(BaseModel):
    """Update class request."""
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    subject: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = None
    is_active: Optional[bool] = None


class JoinClassRequest(BaseModel):
    """Student join-by-code request."""
    join_code: str = Field(..., min_length=1, max_length=16)


class JoinCodeToggleRequest(BaseModel):
    """Enable/disable a class join code."""
    enabled: bool


class SetActiveClassRequest(BaseModel):
    """Select the session's active class."""
    class_id: str


class AddStudentRequest(BaseModel):
    """Enroll a student by registration number (professor flow)."""
    registration_number: str = Field(..., min_length=1, max_length=50)


class AssignStudentRequest(BaseModel):
    """Enroll a student by id or registration number (admin/manager flow)."""
    user_id: Optional[str] = None
    registration_number: Optional[str] = None


class TopicCreate(BaseModel):
    """Create a topic or subtopic in a class's topic tree."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    parent_topic_id: Optional[str] = None
    position: int = 0


class TopicUpdate(BaseModel):
    """Update a topic; name/description changes re-embed the topic."""
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = None
    position: Optional[int] = None


# ===========================================
# Subject
# ===========================================

class SetSubjectRequest(BaseModel):
    """Set subject request."""
    subject: str = Field(..., min_length=1, max_length=200)


class SubjectResponse(BaseModel):
    """Subject response."""
    subject: str


# ===========================================
# Models
# ===========================================

class ModelOption(BaseModel):
    """A single model available in the frontend selector."""
    provider: str
    value: str
    label: str


class ModelsResponse(BaseModel):
    """Available models and the default selection for the frontend."""
    available: List[ModelOption]
    default: str
