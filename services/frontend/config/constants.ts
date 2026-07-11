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

    // Classes (student-facing)
    myClasses: "/api/v1/classes/mine",
    joinClass: "/api/v1/classes/join",
    sessionClass: "/api/v1/session/class",
    studentFeatures: "/api/v1/features",

    // Health
    health: "/health",

    // Professor portal
    professor: {
        login: "/api/v1/professor/login",
        validateSession: "/api/v1/professor/validate-session",
        features: "/api/v1/professor/features",
        classes: "/api/v1/professor/classes",
        classById: (classId: string) => `/api/v1/professor/classes/${classId}`,
        regenerateJoinCode: (classId: string) => `/api/v1/professor/classes/${classId}/join-code/regenerate`,
        toggleJoinCode: (classId: string) => `/api/v1/professor/classes/${classId}/join-code`,
        students: (classId: string) => `/api/v1/professor/classes/${classId}/students`,
        removeStudent: (classId: string, userId: string) => `/api/v1/professor/classes/${classId}/students/${userId}`,
        topics: (classId: string) => `/api/v1/professor/classes/${classId}/topics`,
        topicById: (topicId: string) => `/api/v1/professor/topics/${topicId}`,
        bookCatalog: (classId?: string) => classId ? `/api/v1/professor/books/catalog?class_id=${classId}` : "/api/v1/professor/books/catalog",
        classBooks: (classId: string) => `/api/v1/professor/classes/${classId}/books`,
        attachBook: (classId: string, bookId: string) => `/api/v1/professor/classes/${classId}/books/${bookId}/attach`,
        detachBook: (classId: string, bookId: string) => `/api/v1/professor/classes/${classId}/books/${bookId}/detach`,
        deleteBook: (classId: string, bookId: string) => `/api/v1/professor/classes/${classId}/books/${bookId}`,
        uploadBooks: (classId: string) => `/api/v1/professor/classes/${classId}/books/upload`,
        pdfJobStatus: (jobId: string) => `/api/v1/professor/pdf-job/${jobId}`,
        analyticsTopics: (classId: string, params: string = "") => `/api/v1/professor/classes/${classId}/analytics/topics${params}`,
        analyticsQueries: (classId: string, topicId: string, params: string = "") =>
            `/api/v1/professor/classes/${classId}/analytics/topics/${topicId}/queries${params}`,
        analyticsSummary: (classId: string) => `/api/v1/professor/classes/${classId}/analytics/summary`,
    },

    // Manager portal
    manager: {
        login: "/api/v1/manager/login",
        validateSession: "/api/v1/manager/validate-session",
        features: "/api/v1/manager/features",
        users: "/api/v1/manager/users",
        userById: (userId: string) => `/api/v1/manager/users/${userId}`,
        userStatus: (userId: string) => `/api/v1/manager/users/${userId}/status`,
        importUsers: "/api/v1/manager/users/import",
        classes: "/api/v1/manager/classes",
        classById: (classId: string) => `/api/v1/manager/classes/${classId}`,
        students: (classId: string) => `/api/v1/manager/classes/${classId}/students`,
        removeStudent: (classId: string, userId: string) => `/api/v1/manager/classes/${classId}/students/${userId}`,
        professors: "/api/v1/manager/professors",
        analyticsTopics: (classId: string, params: string = "") => `/api/v1/manager/classes/${classId}/analytics/topics${params}`,
        analyticsQueries: (classId: string, topicId: string, params: string = "") =>
            `/api/v1/manager/classes/${classId}/analytics/topics/${topicId}/queries${params}`,
        analyticsSummary: (classId: string) => `/api/v1/manager/classes/${classId}/analytics/summary`,
    },

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
        importUsers: "/api/v1/admin/users/import",
        systemStatus: "/api/v1/admin/system/status",
        classes: "/api/v1/admin/classes",
        classById: (classId: string) => `/api/v1/admin/classes/${classId}`,
        classStudents: (classId: string) => `/api/v1/admin/classes/${classId}/students`,
        removeClassStudent: (classId: string, userId: string) => `/api/v1/admin/classes/${classId}/students/${userId}`,
        classBooks: (classId: string) => `/api/v1/admin/classes/${classId}/books`,
        attachClassBook: (classId: string, bookId: string) => `/api/v1/admin/classes/${classId}/books/${bookId}/attach`,
        detachClassBook: (classId: string, bookId: string) => `/api/v1/admin/classes/${classId}/books/${bookId}/detach`,
        professors: "/api/v1/admin/professors",
    },
};
