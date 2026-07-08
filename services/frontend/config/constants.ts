// API configuration
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// The session token travels in the Authorization header, never in URLs.
export const authHeaders = (sessionId: string): Record<string, string> => ({
    Authorization: `Bearer ${sessionId}`,
});

// API endpoints - matching the new API Gateway routes
export const API_ENDPOINTS = {
    // Authentication
    login: "/api/v1/login",
    logout: "/api/v1/logout",
    validateSession: "/api/v1/validate-session",
    setSubject: "/api/v1/session/subject",
    getSubject: "/api/v1/session/subject",

    // Chat
    chat: "/api/v1/chat",
    chatSingle: "/api/v1/chat/single",
    chatHistory: "/api/v1/chat/history",
    clearHistory: "/api/v1/chat/history",

    // Conversations
    conversations: "/api/v1/conversations",
    resumeConversation: (conversationId: string) => `/api/v1/conversations/resume/${conversationId}`,
    newConversation: "/api/v1/conversations/new",
    updateConversationTitle: "/api/v1/conversations/current/title",

    // Models
    models: "/api/v1/models",

    // Books
    books: "/api/v1/books",
    bookDetails: (bookId: string) => `/api/v1/books/${bookId}`,
    bookNames: "/api/v1/books/names/list",

    // Health
    health: "/health",

    // Admin endpoints
    admin: {
        login: "/api/v1/admin/login",
        validateSession: "/api/v1/admin/validate-session",
        users: "/api/v1/admin/users",
        userById: (userId: string) => `/api/v1/admin/users/${userId}`,
        userStatus: (userId: string) => `/api/v1/admin/users/${userId}/status`,
        contentStats: "/api/v1/admin/content-stats",
        bookList: "/api/v1/admin/book-list",
        deleteBook: (bookName: string) => `/api/v1/admin/books/${encodeURIComponent(bookName)}`,
        uploadPdfs: "/api/v1/admin/upload-pdfs",
        usageStats: (timeRange: string) => `/api/v1/admin/usage-stats?range=${timeRange}`,
        pdfJobStatus: (jobId: string) => `/api/v1/admin/pdf-job/${jobId}`,
        jobs: "/api/v1/admin/jobs",
        dismissJob: (jobId: string) => `/api/v1/admin/jobs/${jobId}`,
        cancelJob: (jobId: string) => `/api/v1/admin/jobs/${jobId}/cancel`,

        // Snapshot management endpoints
        snapshots: "/api/v1/admin/snapshots",
        createSnapshot: "/api/v1/admin/snapshots/create",
        uploadSnapshot: "/api/v1/admin/snapshots/upload",
        restoreSnapshot: (snapshotName: string) => `/api/v1/admin/snapshots/${snapshotName}/restore`,
        deleteSnapshot: (snapshotName: string) => `/api/v1/admin/snapshots/${snapshotName}`,
        downloadSnapshot: (snapshotName: string) => `/api/v1/admin/snapshots/${snapshotName}/download`,
        downloadMetadata: (snapshotName: string) => `/api/v1/admin/snapshots/${snapshotName}/metadata`,
        features: "/api/v1/admin/features",
    },
};
