-- AcademiCK PostgreSQL Database Schema
-- Run this script to initialize the database

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ===========================================
-- USERS AND AUTHENTICATION
-- ===========================================

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'user' CHECK (role IN ('user', 'professor', 'manager', 'admin')),
    status VARCHAR(50) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'suspended')),
    is_config_user BOOLEAN DEFAULT false,
    registration_number VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_status ON users(status);
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_registration_number
    ON users(registration_number) WHERE registration_number IS NOT NULL;

-- ===========================================
-- CLASSES
-- ===========================================

CREATE TABLE IF NOT EXISTS classes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    subject VARCHAR(255) NOT NULL,
    description TEXT,
    professor_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    join_code VARCHAR(16) UNIQUE,
    join_code_enabled BOOLEAN DEFAULT true,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_classes_professor ON classes(professor_id);
CREATE INDEX IF NOT EXISTS idx_classes_join_code ON classes(join_code) WHERE join_code IS NOT NULL;

CREATE TABLE IF NOT EXISTS class_members (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    class_id UUID NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    enrolled_via VARCHAR(20) NOT NULL CHECK (enrolled_via IN ('join_code', 'professor', 'admin')),
    enrolled_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (class_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_class_members_user ON class_members(user_id);

CREATE TABLE IF NOT EXISTS class_topics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    class_id UUID NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
    parent_topic_id UUID REFERENCES class_topics(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    embedding JSONB,
    position INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE NULLS NOT DISTINCT (class_id, parent_topic_id, name)
);

CREATE INDEX IF NOT EXISTS idx_class_topics_class ON class_topics(class_id);

-- ===========================================
-- SESSIONS (Metadata - actual sessions in Redis)
-- ===========================================

CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    subject VARCHAR(255) DEFAULT 'Machine Learning',
    active_class_id UUID REFERENCES classes(id) ON DELETE SET NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at) WHERE is_active = true;

-- ===========================================
-- BOOKS (PDF Metadata)
-- ===========================================

CREATE TABLE IF NOT EXISTS books (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(500) UNIQUE NOT NULL,
    file_path VARCHAR(1000),
    file_hash VARCHAR(64),
    total_pages INTEGER,
    total_chunks INTEGER DEFAULT 0,
    processing_status VARCHAR(50) DEFAULT 'pending'
        CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')),
    processing_method VARCHAR(50) DEFAULT NULL,
    error_message TEXT,
    owner_user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_books_status ON books(processing_status);
CREATE UNIQUE INDEX IF NOT EXISTS idx_books_hash ON books(file_hash) WHERE file_hash IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_books_owner ON books(owner_user_id);

CREATE TABLE IF NOT EXISTS class_books (
    class_id UUID NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
    book_id UUID NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    added_by UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (class_id, book_id)
);

CREATE INDEX IF NOT EXISTS idx_class_books_book ON class_books(book_id);

-- ===========================================
-- CHAPTERS
-- ===========================================

CREATE TABLE IF NOT EXISTS chapters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    book_id UUID REFERENCES books(id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    chapter_number INTEGER,
    start_page INTEGER,
    end_page INTEGER,
    chunk_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chapters_book_id ON chapters(book_id);
CREATE INDEX IF NOT EXISTS idx_chapters_number ON chapters(book_id, chapter_number);

-- ===========================================
-- CHUNKS (Text segments with Qdrant references)
-- ===========================================

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    book_id UUID REFERENCES books(id) ON DELETE CASCADE,
    chapter_id UUID REFERENCES chapters(id) ON DELETE SET NULL,
    qdrant_point_id UUID NOT NULL,
    text TEXT NOT NULL,
    topic VARCHAR(500),
    is_introduction BOOLEAN DEFAULT false,
    page_number INTEGER,
    chunk_index INTEGER,
    char_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_chunks_book_id ON chunks(book_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chapter_id ON chunks(chapter_id);
CREATE INDEX IF NOT EXISTS idx_chunks_qdrant_id ON chunks(qdrant_point_id);
CREATE INDEX IF NOT EXISTS idx_chunks_topic ON chunks(topic);

-- ===========================================
-- CONVERSATIONS
-- ===========================================

CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    class_id UUID REFERENCES classes(id) ON DELETE SET NULL,
    subject VARCHAR(255),
    title VARCHAR(255),
    message_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_class ON conversations(class_id);

-- ===========================================
-- MESSAGES
-- ===========================================

CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    intent VARCHAR(100),
    model_used VARCHAR(100),
    tokens_used INTEGER,
    response_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_messages_intent ON messages(intent);

-- ===========================================
-- MESSAGE TOPICS (Analytics - query-to-topic assignments)
-- ===========================================

CREATE TABLE IF NOT EXISTS message_topics (
    message_id UUID PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
    class_id UUID REFERENCES classes(id) ON DELETE CASCADE,
    topic_id UUID REFERENCES class_topics(id) ON DELETE SET NULL,
    similarity FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_message_topics_class_created ON message_topics(class_id, created_at);
CREATE INDEX IF NOT EXISTS idx_message_topics_topic ON message_topics(topic_id);

-- ===========================================
-- CHUNK RETRIEVALS (Analytics - tracks which chunks are retrieved)
-- ===========================================

CREATE TABLE IF NOT EXISTS chunk_retrievals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id UUID REFERENCES messages(id) ON DELETE CASCADE,
    chunk_id UUID REFERENCES chunks(id) ON DELETE SET NULL,
    book_id UUID REFERENCES books(id) ON DELETE SET NULL,
    chapter_id UUID REFERENCES chapters(id) ON DELETE SET NULL,
    score FLOAT,
    position INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chunk_retrievals_message ON chunk_retrievals(message_id);
CREATE INDEX IF NOT EXISTS idx_chunk_retrievals_chunk ON chunk_retrievals(chunk_id);
CREATE INDEX IF NOT EXISTS idx_chunk_retrievals_book ON chunk_retrievals(book_id);
CREATE INDEX IF NOT EXISTS idx_chunk_retrievals_created ON chunk_retrievals(created_at);

-- ===========================================
-- PROCESSING JOBS
-- ===========================================

CREATE TABLE IF NOT EXISTS processing_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_type VARCHAR(50) NOT NULL CHECK (job_type IN ('pdf_processing', 'reindex', 'migration', 'embedding_update')),
    status VARCHAR(50) DEFAULT 'pending'
        CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')),
    book_id UUID REFERENCES books(id) ON DELETE SET NULL,
    progress FLOAT DEFAULT 0 CHECK (progress >= 0 AND progress <= 1),
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON processing_jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_book ON processing_jobs(book_id);
CREATE INDEX IF NOT EXISTS idx_jobs_type ON processing_jobs(job_type);

-- ===========================================
-- USAGE STATISTICS
-- ===========================================

CREATE TABLE IF NOT EXISTS usage_stats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    session_id UUID,
    action_type VARCHAR(50) NOT NULL
        CHECK (action_type IN ('query', 'login', 'logout', 'pdf_upload', 'search', 'chat')),
    response_time_ms INTEGER,
    model_used VARCHAR(100),
    tokens_consumed INTEGER,
    intent VARCHAR(100),
    success BOOLEAN DEFAULT true,
    agent_actions INTEGER DEFAULT 0,
    agent_search_calls INTEGER DEFAULT 0,
    agent_list_chapters_calls INTEGER DEFAULT 0,
    agent_list_topics_calls INTEGER DEFAULT 0,
    agent_read_chapter_calls INTEGER DEFAULT 0,
    agent_expand_context_calls INTEGER DEFAULT 0,
    agent_pool_chunks INTEGER DEFAULT 0,
    agent_dropped_chunks INTEGER DEFAULT 0,
    agent_final_chunks INTEGER DEFAULT 0,
    agent_not_in_kb BOOLEAN DEFAULT false,
    agent_tokens INTEGER DEFAULT 0,
    agent_time_ms INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_usage_user ON usage_stats(user_id);
CREATE INDEX IF NOT EXISTS idx_usage_created ON usage_stats(created_at);
CREATE INDEX IF NOT EXISTS idx_usage_type ON usage_stats(action_type);

-- ===========================================
-- FUNCTIONS
-- ===========================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_books_updated_at
    BEFORE UPDATE ON books
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_conversations_updated_at
    BEFORE UPDATE ON conversations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_classes_updated_at
    BEFORE UPDATE ON classes
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_class_topics_updated_at
    BEFORE UPDATE ON class_topics
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ===========================================
-- VIEWS
-- ===========================================

-- View for book statistics
CREATE OR REPLACE VIEW book_stats AS
SELECT
    b.id,
    b.name,
    b.processing_status,
    b.total_chunks,
    COUNT(DISTINCT c.id) as chapter_count,
    b.created_at,
    b.processed_at
FROM books b
LEFT JOIN chapters c ON b.id = c.book_id
GROUP BY b.id;

-- ===========================================
-- DEFAULT DATA
-- ===========================================

-- Note: Default users are created by the API gateway on startup
-- based on CONFIG_USERS environment variables

COMMENT ON TABLE users IS 'User accounts with bcrypt password hashing';
COMMENT ON TABLE sessions IS 'Session metadata with PostgreSQL persistence for resumption';
COMMENT ON TABLE books IS 'PDF book metadata';
COMMENT ON TABLE chapters IS 'Book chapter information';
COMMENT ON TABLE chunks IS 'Text chunks with references to Qdrant vectors';
COMMENT ON TABLE conversations IS 'Conversation records linking sessions to message history';
COMMENT ON TABLE messages IS 'Chat message history with intent and model metadata';
COMMENT ON TABLE chunk_retrievals IS 'Analytics tracking which chunks are retrieved in searches';
COMMENT ON TABLE processing_jobs IS 'Async job tracking for PDF processing';
COMMENT ON TABLE usage_stats IS 'User activity and usage analytics';
COMMENT ON TABLE classes IS 'Classes owned by professors, grouping students, books and topics';
COMMENT ON TABLE class_members IS 'Student enrollment in classes with enrollment method';
COMMENT ON TABLE class_books IS 'Books attached to a class (the student-visible library)';
COMMENT ON TABLE class_topics IS 'Per-class topic/subtopic tree with embedding vectors for query matching';
COMMENT ON TABLE message_topics IS 'Async query-to-topic assignments; NULL topic_id = unclassified';
