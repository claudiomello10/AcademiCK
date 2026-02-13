// API configuration
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// API endpoints - matching the new API Gateway routes
export const API_ENDPOINTS = {
    // Authentication
    login: "/api/v1/login",
    logout: (sessionId: string) => `/api/v1/logout/${sessionId}`,
    validateSession: (sessionId: string) => `/api/v1/validate-session/${sessionId}`,
    setSubject: (sessionId: string) => `/api/v1/session/${sessionId}/subject`,
    getSubject: (sessionId: string) => `/api/v1/session/${sessionId}/subject`,

    // Chat
    chat: (sessionId: string) => `/api/v1/chat/${sessionId}`,
    chatSingle: (sessionId: string) => `/api/v1/chat/${sessionId}/single`,
    chatHistory: (sessionId: string) => `/api/v1/chat/${sessionId}/history`,
    clearHistory: (sessionId: string) => `/api/v1/chat/${sessionId}/history`,

    // Conversations
    conversations: (sessionId: string) => `/api/v1/conversations/${sessionId}`,
    resumeConversation: (sessionId: string, conversationId: string) => `/api/v1/conversations/${sessionId}/resume/${conversationId}`,
    newConversation: (sessionId: string) => `/api/v1/conversations/${sessionId}/new`,
    updateConversationTitle: (sessionId: string) => `/api/v1/conversations/${sessionId}/current/title`,

    // Books
    books: "/api/v1/books",
    bookDetails: (bookId: string) => `/api/v1/books/${bookId}`,
    bookNames: "/api/v1/books/names/list",

    // Health
    health: "/health",

    // Admin endpoints
    admin: {
        login: "/api/v1/admin/login",
        validateSession: (sessionId: string) => `/api/v1/admin/validate-session/${sessionId}`,
        users: (sessionId: string) => `/api/v1/admin/users?session_id=${sessionId}`,
        userById: (userId: string, sessionId: string) => `/api/v1/admin/users/${userId}?session_id=${sessionId}`,
        userStatus: (userId: string, sessionId: string) => `/api/v1/admin/users/${userId}/status?session_id=${sessionId}`,
        contentStats: (sessionId: string) => `/api/v1/admin/content-stats?session_id=${sessionId}`,
        bookList: (sessionId: string) => `/api/v1/admin/book-list?session_id=${sessionId}`,
        deleteBook: (bookName: string, sessionId: string) => `/api/v1/admin/books/${encodeURIComponent(bookName)}?session_id=${sessionId}`,
        uploadPdfs: (sessionId: string) => `/api/v1/admin/upload-pdfs?session_id=${sessionId}`,
        usageStats: (timeRange: string, sessionId: string) => `/api/v1/admin/usage-stats?range=${timeRange}&session_id=${sessionId}`,
        pdfJobStatus: (jobId: string, sessionId: string) => `/api/v1/admin/pdf-job/${jobId}?session_id=${sessionId}`,
        jobs: (sessionId: string) => `/api/v1/admin/jobs?session_id=${sessionId}`,
        dismissJob: (jobId: string, sessionId: string) => `/api/v1/admin/jobs/${jobId}?session_id=${sessionId}`,
        cancelJob: (jobId: string, sessionId: string) => `/api/v1/admin/jobs/${jobId}/cancel?session_id=${sessionId}`,

        // Snapshot management endpoints
        snapshots: (sessionId: string) => `/api/v1/admin/snapshots?session_id=${sessionId}`,
        createSnapshot: (sessionId: string) => `/api/v1/admin/snapshots/create?session_id=${sessionId}`,
        uploadSnapshot: (sessionId: string) => `/api/v1/admin/snapshots/upload?session_id=${sessionId}`,
        restoreSnapshot: (snapshotName: string, sessionId: string) => `/api/v1/admin/snapshots/${snapshotName}/restore?session_id=${sessionId}`,
        deleteSnapshot: (snapshotName: string, sessionId: string) => `/api/v1/admin/snapshots/${snapshotName}?session_id=${sessionId}`,
        downloadSnapshot: (snapshotName: string, sessionId: string) => `/api/v1/admin/snapshots/${snapshotName}/download?session_id=${sessionId}`,
        downloadMetadata: (snapshotName: string, sessionId: string) => `/api/v1/admin/snapshots/${snapshotName}/metadata?session_id=${sessionId}`,
        features: (sessionId: string) => `/api/v1/admin/features?session_id=${sessionId}`,
    },
};