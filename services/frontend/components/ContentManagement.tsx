import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from '@/components/ui/alert-dialog';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Progress } from '@/components/ui/progress';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
    Book, Database, FileText, Upload, Download,
    Trash2, AlertCircle, Loader2, FileJson,
    ChevronUp, ChevronDown, CheckCircle2, X, AlertTriangle, StopCircle
} from 'lucide-react';

import { API_BASE_URL, API_ENDPOINTS } from '@/config/constants';

interface ProcessingJob {
    filename: string;
    job_id: string;
    status: string;
    progress: number;
    stage?: string;
    chapters_total?: number;
    chapters_processed?: number;
    warning?: string;
    error?: string;
}

interface BookInfo {
    name: string;
    total_chapters: number;
    total_chunks: number;
    processing_status?: string;
}

interface ExpandedState {
    [key: string]: boolean;
}

const ContentManagement = () => {
    const [files, setFiles] = useState<File[]>([]);
    const [processing, setProcessing] = useState(false);
    const [deleting, setDeleting] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState('');
    const [stats, setStats] = useState({
        total_books: 0,
        total_chunks: 0,
        total_embeddings: 0
    });
    const [books, setBooks] = useState<BookInfo[]>([]);
    const [expanded, setExpanded] = useState<ExpandedState>({});
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [processingJobs, setProcessingJobs] = useState<ProcessingJob[]>([]);
    const [snapshots, setSnapshots] = useState<any[]>([]);
    const [snapshotLoading, setSnapshotLoading] = useState(false);
    const [featureFlags, setFeatureFlags] = useState({
        snapshot_management_enabled: true,
        pdf_upload_enabled: true
    });
    const [snapshotToRestore, setSnapshotToRestore] = useState<string | null>(null);
    const [snapshotToDelete, setSnapshotToDelete] = useState<string | null>(null);
    const [showUploadDialog, setShowUploadDialog] = useState(false);
    const [uploadSnapshotFile, setUploadSnapshotFile] = useState<File | null>(null);
    const [uploadMetadataFile, setUploadMetadataFile] = useState<File | null>(null);
    const pollingRef = useRef<NodeJS.Timeout | null>(null);
    const completedJobsRef = useRef<Set<string>>(new Set());
    const autoDismissTimersRef = useRef<Map<string, NodeJS.Timeout>>(new Map());

    useEffect(() => {
        // Get session ID from localStorage instead of sessionStorage
        const adminSession = localStorage.getItem('adminSession');
        if (adminSession) {
            setSessionId(adminSession);
        }
    }, []);

    // Fetch feature flags and snapshots when sessionId is set
    useEffect(() => {
        if (sessionId) {
            fetchFeatureFlags();
            fetchSnapshots();
        }
    }, [sessionId]);

    // Fetch jobs from PostgreSQL (persisted, visible to all admins)
    const fetchJobs = useCallback(async () => {
        if (!sessionId) return;

        try {
            const response = await fetch(
                `${API_BASE_URL}${API_ENDPOINTS.admin.jobs(sessionId)}`
            );
            if (response.ok) {
                const data = await response.json();
                const jobs: ProcessingJob[] = data.map((job: {
                    job_id: string;
                    filename: string;
                    status: string;
                    progress: number;
                    stage?: string;
                    chapters_total?: number;
                    chapters_processed?: number;
                    warning?: string;
                    error?: string;
                }) => ({
                    job_id: job.job_id,
                    filename: job.filename,
                    status: job.status,
                    progress: job.status === 'completed' ? 100 : (job.progress || 0),
                    stage: job.stage,
                    chapters_total: job.chapters_total || 0,
                    chapters_processed: job.chapters_processed || 0,
                    warning: job.warning,
                    error: job.error
                }));

                // Detect newly completed jobs and refresh books/stats
                const newlyCompleted = jobs.filter(
                    j => (j.status === 'completed' || j.status === 'failed') &&
                         !completedJobsRef.current.has(j.job_id)
                );

                if (newlyCompleted.length > 0) {
                    // Add to completed set
                    newlyCompleted.forEach(j => completedJobsRef.current.add(j.job_id));
                    // Refresh books and stats immediately
                    fetchBooks();
                    fetchStats();

                    // Auto-dismiss only successfully completed jobs after 5 seconds
                    // Failed jobs stay visible so user can see the error
                    newlyCompleted
                        .filter(job => job.status === 'completed')
                        .forEach(job => {
                            const timer = setTimeout(() => {
                                handleDismissJob(job.job_id, job.filename);
                                autoDismissTimersRef.current.delete(job.job_id);
                            }, 5000);
                            autoDismissTimersRef.current.set(job.job_id, timer);
                        });
                }

                setProcessingJobs(jobs);

                // Start polling if there are active jobs
                const hasActiveJobs = jobs.some(
                    j => j.status !== 'completed' && j.status !== 'failed' && j.status !== 'cancelled'
                );
                if (hasActiveJobs && !pollingRef.current) {
                    setProcessing(true);
                    pollingRef.current = setInterval(fetchJobs, 3000);
                } else if (!hasActiveJobs && pollingRef.current) {
                    clearInterval(pollingRef.current);
                    pollingRef.current = null;
                    setProcessing(false);
                }
            }
        } catch (err) {
            console.error('Error fetching jobs:', err);
        }
    }, [sessionId]);

    const fetchStats = useCallback(async () => {
        try {
            if (!sessionId) return;

            const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.admin.contentStats(sessionId)}`);
            if (response.ok) {
                const data = await response.json();
                setStats(data);
            }
        } catch (err) {
            console.error('Error fetching stats:', err);
        }
    }, [sessionId]);

    const fetchBooks = useCallback(async () => {
        try {
            if (!sessionId) return;

            const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.admin.bookList(sessionId)}`);
            if (response.ok) {
                const data = await response.json();
                setBooks(data);
            }
        } catch (err) {
            console.error('Error fetching books:', err);
        }
    }, [sessionId]);

    // Fetch data on mount
    useEffect(() => {
        if (sessionId) {
            fetchStats();
            fetchBooks();
            fetchJobs();
        }
    }, [sessionId, fetchStats, fetchBooks, fetchJobs]);

    // Cleanup polling and timers on unmount
    useEffect(() => {
        return () => {
            if (pollingRef.current) {
                clearInterval(pollingRef.current);
            }
            // Clear all auto-dismiss timers
            autoDismissTimersRef.current.forEach(timer => clearTimeout(timer));
            autoDismissTimersRef.current.clear();
        };
    }, []);

    const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        const uploadedFiles = Array.from(event.target.files || []) as File[];
        setFiles(prevFiles => [...prevFiles, ...uploadedFiles]);
    };

    const removeFile = (index: number) => {
        setFiles(prevFiles => prevFiles.filter((_, i) => i !== index));
    };

    const handleDismissJob = async (jobId: string, filename: string) => {
        if (!sessionId) return;

        // Frontend-only error jobs have no job_id — just remove from local state
        if (!jobId) {
            setProcessingJobs(prev => prev.filter(j => j.filename !== filename));
            return;
        }

        // Clear any pending auto-dismiss timer for this job
        const timer = autoDismissTimersRef.current.get(jobId);
        if (timer) {
            clearTimeout(timer);
            autoDismissTimersRef.current.delete(jobId);
        }

        try {
            const response = await fetch(
                `${API_BASE_URL}${API_ENDPOINTS.admin.dismissJob(jobId, sessionId)}`,
                { method: 'DELETE' }
            );
            if (response.ok) {
                setProcessingJobs(prev => prev.filter(j => j.job_id !== jobId));
            }
        } catch (err) {
            console.error('Error dismissing job:', err);
        }
    };

    const handleCancelJob = async (jobId: string) => {
        if (!sessionId) return;
        try {
            const response = await fetch(
                `${API_BASE_URL}${API_ENDPOINTS.admin.cancelJob(jobId, sessionId)}`,
                { method: 'POST' }
            );
            if (response.ok) {
                // Update job status locally while waiting for next poll
                setProcessingJobs(prev => prev.map(j =>
                    j.job_id === jobId
                        ? { ...j, status: 'cancelled', stage: 'cancelled' }
                        : j
                ));
                setSuccess('Job cancellation requested');
            } else {
                const data = await response.json();
                setError(data.detail || 'Failed to cancel job');
            }
        } catch (err) {
            console.error('Error cancelling job:', err);
            setError('Failed to cancel job');
        }
    };

    const processFiles = async () => {
        if (!sessionId) return;

        setProcessing(true);
        setError(null);
        setSuccess('');
        setProcessingJobs([]);

        const formData = new FormData();
        files.forEach(file => {
            formData.append('files', file);
        });

        try {
            const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.admin.uploadPdfs(sessionId)}`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                // Initialize jobs from response
                const jobs: ProcessingJob[] = data.jobs.map((job: { filename: string; job_id: string; status: string }) => ({
                    filename: job.filename,
                    job_id: job.job_id,
                    status: job.status || 'queued',
                    progress: 0,
                    stage: 'queued'
                }));

                if (data.errors && data.errors.length > 0) {
                    const errorJobs: ProcessingJob[] = data.errors.map((err: { filename: string; error: string }) => ({
                        filename: err.filename,
                        job_id: '',
                        status: 'failed',
                        progress: 0,
                        error: err.error
                    }));
                    jobs.push(...errorJobs);
                }

                setProcessingJobs(jobs);
                setFiles([]);

                // Start polling for job status using the jobs endpoint
                if (jobs.some(j => j.status !== 'failed')) {
                    pollingRef.current = setInterval(() => {
                        fetchJobs();
                    }, 3000);
                } else {
                    setProcessing(false);
                    setError('All uploads failed');
                }
            } else {
                throw new Error(data.detail || 'Error processing PDFs');
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Unknown error');
            setProcessing(false);
        }
    };

    // Snapshot Management Functions
    const fetchFeatureFlags = async () => {
        if (!sessionId) return;
        try {
            const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.admin.features(sessionId)}`);
            if (response.ok) {
                const data = await response.json();
                setFeatureFlags(data);
            }
        } catch (error) {
            console.error('Failed to fetch feature flags:', error);
        }
    };

    const fetchSnapshots = async () => {
        if (!sessionId) return;
        setSnapshotLoading(true);
        try {
            const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.admin.snapshots(sessionId)}`);
            if (response.ok) {
                const data = await response.json();
                setSnapshots(data.snapshots);
            }
        } catch (error) {
            console.error('Failed to fetch snapshots:', error);
            setError('Failed to load snapshots');
        } finally {
            setSnapshotLoading(false);
        }
    };

    const createSnapshot = async () => {
        if (!sessionId) return;
        setProcessing(true);
        setError(null);
        try {
            const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.admin.createSnapshot(sessionId)}`, {
                method: 'POST'
            });
            if (response.ok) {
                const data = await response.json();
                setSuccess(`Snapshot created: ${data.snapshot_name}`);
                await fetchSnapshots();
            } else {
                throw new Error('Failed to create snapshot');
            }
        } catch (error) {
            setError('Failed to create snapshot');
        } finally {
            setProcessing(false);
        }
    };

    const restoreSnapshot = async (snapshotName: string) => {
        if (!sessionId) return;

        setProcessing(true);
        setError(null);
        try {
            const response = await fetch(
                `${API_BASE_URL}${API_ENDPOINTS.admin.restoreSnapshot(snapshotName, sessionId)}`,
                { method: 'POST' }
            );

            if (response.ok) {
                const data = await response.json();
                setSuccess(`Restored: ${data.books_imported} books, ${data.chapters_imported} chapters`);
                await fetchStats();
                await fetchBooks();
            } else {
                const err = await response.json();
                throw new Error(err.detail || 'Failed to restore snapshot');
            }
        } catch (error: any) {
            setError(error.message || 'Failed to restore snapshot');
        } finally {
            setProcessing(false);
            setSnapshotToRestore(null);
        }
    };

    const uploadSnapshot = async () => {
        if (!sessionId || !uploadSnapshotFile || !uploadMetadataFile) return;

        setProcessing(true);
        setError(null);
        try {
            const formData = new FormData();
            formData.append('snapshot_file', uploadSnapshotFile);
            formData.append('metadata_file', uploadMetadataFile);

            const response = await fetch(
                `${API_BASE_URL}${API_ENDPOINTS.admin.uploadSnapshot(sessionId)}`,
                { method: 'POST', body: formData }
            );

            if (response.ok) {
                const data = await response.json();
                setSuccess(`Uploaded: ${data.books_imported} books, ${data.chapters_imported} chapters`);
                await fetchSnapshots();
                await fetchStats();
                await fetchBooks();
            } else {
                const err = await response.json();
                throw new Error(err.detail || 'Failed to upload snapshot');
            }
        } catch (error: any) {
            setError(error.message || 'Failed to upload snapshot');
        } finally {
            setProcessing(false);
            setShowUploadDialog(false);
            setUploadSnapshotFile(null);
            setUploadMetadataFile(null);
        }
    };

    const downloadSnapshot = async (snapshotName: string) => {
        if (!sessionId) return;
        try {
            const url = `${API_BASE_URL}${API_ENDPOINTS.admin.downloadSnapshot(snapshotName, sessionId)}`;
            const link = document.createElement('a');
            link.href = url;
            link.download = snapshotName;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            setSuccess(`Downloading snapshot: ${snapshotName}`);
        } catch (error) {
            setError('Failed to download snapshot');
        }
    };

    const downloadMetadata = async (snapshotName: string) => {
        if (!sessionId) return;
        try {
            const url = `${API_BASE_URL}${API_ENDPOINTS.admin.downloadMetadata(snapshotName, sessionId)}`;
            const link = document.createElement('a');
            link.href = url;
            link.download = `${snapshotName}.metadata.json`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            setSuccess(`Downloading metadata for ${snapshotName}`);
        } catch (error) {
            setError('Failed to download metadata');
        }
    };

    const deleteSnapshot = async (snapshotName: string) => {
        if (!sessionId) return;

        try {
            const response = await fetch(
                `${API_BASE_URL}${API_ENDPOINTS.admin.deleteSnapshot(snapshotName, sessionId)}`,
                { method: 'DELETE' }
            );
            if (response.ok) {
                setSuccess(`Deleted snapshot: ${snapshotName}`);
                await fetchSnapshots();
            } else {
                throw new Error('Failed to delete snapshot');
            }
        } catch (error) {
            setError('Failed to delete snapshot');
        } finally {
            setSnapshotToDelete(null);
        }
    };

    const handleDeleteBook = async (bookName: string) => {
        if (!sessionId) return;

        setDeleting(bookName);
        setError(null);
        setSuccess('');

        try {
            const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.admin.deleteBook(bookName, sessionId)}`, {
                method: 'DELETE'
            });

            const data = await response.json();

            if (response.ok) {
                setSuccess(`Successfully deleted "${bookName}" (${data.vectors_deleted} embeddings removed)`);
                fetchStats();
                fetchBooks();
            } else {
                throw new Error(data.detail || 'Error deleting book');
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Unknown error');
        } finally {
            setDeleting(null);
        }
    };

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <h3 className="text-lg font-medium text-secondary">Content Management</h3>
                {featureFlags.pdf_upload_enabled && (
                    <>
                        <Button
                            className="rounded-xl bg-primary text-primary-foreground hover:bg-primary/90"
                            onClick={() => document.getElementById('file-upload')?.click()}
                        >
                            <Upload className="h-4 w-4 mr-2" />
                            Add New Content
                        </Button>
                        <Input
                            id="file-upload"
                            type="file"
                            multiple
                            accept=".pdf"
                            className="hidden"
                            onChange={handleFileUpload}
                        />
                    </>
                )}
            </div>


            {/* Statistics Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Card className="rounded-xl border border-primary/20">
                    <CardContent className="p-4">
                        <div className="flex items-center gap-3">
                            <Book className="h-5 w-5 text-primary" />
                            <div className='text-secondary'>
                                <p className="text-sm font-medium">Total Books</p>
                                <p className="text-2xl font-bold">{stats.total_books}</p>
                            </div>
                        </div>
                    </CardContent>
                </Card>

                <Card className="rounded-xl border border-primary/20">
                    <CardContent className="p-4">
                        <div className="flex items-center gap-3">
                            <FileText className="h-5 w-5 text-primary" />
                            <div className='text-secondary'>
                                <p className="text-sm font-medium">Total Chunks</p>
                                <p className="text-2xl font-bold">{stats.total_chunks}</p>
                            </div>
                        </div>
                    </CardContent>
                </Card>

                <Card className="rounded-xl border border-primary/20">
                    <CardContent className="p-4">
                        <div className="flex items-center gap-3">
                            <Database className="h-5 w-5 text-primary" />
                            <div className='text-secondary'>
                                <p className="text-sm font-medium">Total Embeddings</p>
                                <p className="text-2xl font-bold">{stats.total_embeddings}</p>
                            </div>
                        </div>
                    </CardContent>
                </Card>
            </div>

            {/* File Upload Section */}
            {featureFlags.pdf_upload_enabled && files.length > 0 && (
                <Card className="rounded-xl">
                    <CardHeader>
                        <CardTitle className='text-primary'>Files to Process</CardTitle>
                        <CardDescription>Review and process selected PDF files</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <ScrollArea className="h-64">
                            <div className="space-y-2">
                                {files.map((file, index) => (
                                    <div key={index} className="flex items-center justify-between bg-primary p-3 rounded-lg">
                                        <div className="flex items-center gap-3">
                                            <FileText className="h-5 w-5" />
                                            <span className="font-medium">{file.name}</span>
                                            <span className="text-sm text-muted-foreground text-black">
                                                ({(file.size / 1024 / 1024).toFixed(2)} MB)
                                            </span>
                                        </div>
                                        <Button
                                            variant="ghost"
                                            size="icon"
                                            className="h-8 w-8 text-destructive bg-white border-2 border-destructive hover:text-white hover:bg-destructive"
                                            onClick={() => removeFile(index)}
                                        >
                                            <Trash2 className="h-4 w-4" />
                                        </Button>
                                    </div>
                                ))}
                            </div>
                        </ScrollArea>

                        <div className="mt-4 flex justify-end">
                            <Button
                                className="rounded-xl bg-primary text-primary-foreground"
                                onClick={processFiles}
                                disabled={processing}
                            >
                                {processing ? (
                                    <>
                                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                        Processing...
                                    </>
                                ) : (
                                    <>
                                        <Database className="h-4 w-4 mr-2" />
                                        Process Files
                                    </>
                                )}
                            </Button>
                        </div>
                    </CardContent>
                </Card>
            )}

            {/* Processing Jobs Progress */}
            {featureFlags.pdf_upload_enabled && processingJobs.length > 0 && (
                <Card className="rounded-xl border border-primary/20">
                    <CardHeader>
                        <CardTitle className="text-primary">Processing Status</CardTitle>
                        <CardDescription>PDF processing progress</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div className="space-y-4">
                            {processingJobs.map((job, index) => (
                                <div key={index} className="space-y-2">
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-2">
                                            {job.status === 'completed' ? (
                                                <CheckCircle2 className="h-4 w-4 text-green-500" />
                                            ) : job.status === 'failed' ? (
                                                <AlertCircle className="h-4 w-4 text-red-500" />
                                            ) : job.status === 'cancelled' ? (
                                                <StopCircle className="h-4 w-4 text-orange-500" />
                                            ) : (
                                                <Loader2 className="h-4 w-4 animate-spin text-primary" />
                                            )}
                                            <span className="font-medium text-secondary">{job.filename}</span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <span className="text-sm text-muted-foreground">
                                                {job.status === 'completed' ? 'Complete' :
                                                 job.status === 'failed' ? 'Failed' :
                                                 job.status === 'cancelled' ? 'Cancelled' :
                                                 job.chapters_total && job.chapters_total > 0 &&
                                                 (job.chapters_processed || 0) < job.chapters_total &&
                                                 !job.stage?.includes('embedding') &&
                                                 !job.stage?.includes('storing')
                                                    ? `Chapter ${Math.min((job.chapters_processed || 0) + 1, job.chapters_total)}/${job.chapters_total}`
                                                    : job.stage ? job.stage.replace(/_/g, ' ') : 'Processing...'}
                                            </span>
                                            {/* Show Cancel button for in-progress jobs, Dismiss button for finished jobs */}
                                            {(job.status === 'queued' || job.status === 'processing') ? (
                                                <Button
                                                    variant="ghost"
                                                    size="icon"
                                                    className="h-6 w-6 text-destructive hover:bg-destructive hover:text-white"
                                                    onClick={() => handleCancelJob(job.job_id)}
                                                    title="Cancel job"
                                                >
                                                    <StopCircle className="h-4 w-4" />
                                                </Button>
                                            ) : (
                                                <Button
                                                    variant="ghost"
                                                    size="icon"
                                                    className="h-6 w-6 text-muted-foreground hover:text-destructive"
                                                    onClick={() => handleDismissJob(job.job_id, job.filename)}
                                                    title="Dismiss job"
                                                >
                                                    <X className="h-4 w-4" />
                                                </Button>
                                            )}
                                        </div>
                                    </div>
                                    <Progress value={job.progress} className="h-2" />
                                    {job.warning && (
                                        <div className="flex items-center gap-2 p-2 rounded bg-yellow-50 border border-yellow-200">
                                            <AlertTriangle className="h-4 w-4 text-yellow-600 flex-shrink-0" />
                                            <span className="text-sm text-yellow-700">{job.warning}</span>
                                        </div>
                                    )}
                                    {job.error && (
                                        <p className="text-sm text-red-500">{job.error}</p>
                                    )}
                                </div>
                            ))}
                        </div>
                    </CardContent>
                </Card>
            )}

            {/* Status Messages */}
            {error && (
                <Alert variant="destructive" className="rounded-xl">
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                </Alert>
            )}

            {success && (
                <Alert className="rounded-xl bg-green-50 border-green-200">
                    <AlertTitle className="text-green-800">Success</AlertTitle>
                    <AlertDescription className="text-green-700">{success}</AlertDescription>
                </Alert>
            )}

            {/* Book List */}
            <Card className="rounded-xl border border-primary/20 text-primary">
                <CardHeader>
                    <CardTitle>Content Library</CardTitle>
                    <CardDescription>Processed books and their statistics</CardDescription>
                </CardHeader>
                <CardContent>
                    <ScrollArea className="h-[400px]">
                        <div className="space-y-2">
                            {books.map((book, index) => (
                                <Card
                                    key={index}
                                    className={`rounded-lg border text-primary ${
                                        book.processing_status === 'processing'
                                            ? 'border-yellow-400 animate-pulse'
                                            : 'border-primary/10'
                                    }`}
                                >
                                    <CardContent className="p-4">
                                        <div className="flex items-center justify-between">
                                            <div className="flex items-center gap-3">
                                                {book.processing_status === 'processing' ? (
                                                    <Loader2 className="h-5 w-5 animate-spin text-yellow-500" />
                                                ) : (
                                                    <Book className="h-5 w-5" />
                                                )}
                                                <div>
                                                    <div className="flex items-center gap-2">
                                                        <h4 className="font-medium">{book.name}</h4>
                                                        {book.processing_status === 'processing' && (
                                                            <span className="text-xs px-2 py-0.5 rounded-full bg-yellow-100 text-yellow-700 font-medium">
                                                                Processing...
                                                            </span>
                                                        )}
                                                    </div>
                                                    <p className="text-sm text-muted-foreground text-secondary">
                                                        {book.total_chapters} chapters, {book.total_chunks} chunks
                                                    </p>
                                                </div>
                                            </div>
                                            <div className="flex items-center gap-2">
                                                <Button
                                                    variant="ghost"
                                                    size="icon"
                                                    className="h-8 w-8"
                                                    onClick={() => setExpanded(prev => ({ ...prev, [book.name]: !prev[book.name] }))}
                                                >
                                                    {expanded[book.name] ? (
                                                        <ChevronUp className="h-4 w-4" />
                                                    ) : (
                                                        <ChevronDown className="h-4 w-4" />
                                                    )}
                                                </Button>
                                                <AlertDialog>
                                                    <AlertDialogTrigger asChild>
                                                        <Button
                                                            variant="ghost"
                                                            size="icon"
                                                            className="h-8 w-8 text-destructive hover:text-white hover:bg-destructive"
                                                            disabled={deleting === book.name || book.processing_status === 'processing'}
                                                        >
                                                            {deleting === book.name ? (
                                                                <Loader2 className="h-4 w-4 animate-spin" />
                                                            ) : (
                                                                <Trash2 className="h-4 w-4" />
                                                            )}
                                                        </Button>
                                                    </AlertDialogTrigger>
                                                    <AlertDialogContent className="bg-muted border-primary/20">
                                                        <AlertDialogHeader>
                                                            <AlertDialogTitle className="text-primary">Delete Book</AlertDialogTitle>
                                                            <AlertDialogDescription className="text-muted-foreground">
                                                                Are you sure you want to delete &quot;{book.name}&quot;? This will remove all embeddings associated with this book. This action cannot be undone.
                                                            </AlertDialogDescription>
                                                        </AlertDialogHeader>
                                                        <AlertDialogFooter>
                                                            <AlertDialogCancel className="border-muted-foreground/30 text-secondary hover:bg-muted-foreground/10 hover:text-secondary">Cancel</AlertDialogCancel>
                                                            <AlertDialogAction
                                                                className="bg-primary text-primary-foreground hover:bg-primary/90"
                                                                onClick={() => handleDeleteBook(book.name)}
                                                            >
                                                                Delete
                                                            </AlertDialogAction>
                                                        </AlertDialogFooter>
                                                    </AlertDialogContent>
                                                </AlertDialog>
                                            </div>
                                        </div>
                                        {expanded[book.name] && (
                                            <div className="mt-4 space-y-2">
                                                <div className="text-sm text-secondary">
                                                    <p><strong>Total Chapters:</strong> {book.total_chapters}</p>
                                                    <p><strong>Total Chunks:</strong> {book.total_chunks}</p>
                                                    <p><strong>Status:</strong> {book.processing_status === 'completed' ? 'Complete' : book.processing_status === 'processing' ? 'Processing...' : book.processing_status || 'Unknown'}</p>
                                                </div>
                                            </div>
                                        )}
                                    </CardContent>
                                </Card>
                            ))}
                        </div>
                    </ScrollArea>
                </CardContent>
            </Card>

            {/* Snapshot Management Section */}
            {featureFlags.snapshot_management_enabled && (
                <Card className="rounded-xl border border-primary/20">
                    <CardHeader>
                        <CardTitle className="text-primary">Qdrant Snapshot Management</CardTitle>
                        <CardDescription>Backup and restore vector database snapshots</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        {/* Create Snapshot Button */}
                        <div className="flex items-center justify-between">
                            <div>
                                <h3 className="text-lg font-medium text-secondary">Create New Snapshot</h3>
                                <p className="text-sm text-muted-foreground">Backup current embeddings</p>
                            </div>
                            <Button
                                className="rounded-xl bg-primary text-primary-foreground hover:bg-primary/90"
                                onClick={createSnapshot}
                                disabled={processing}
                            >
                                {processing ? (
                                    <>
                                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                        Creating...
                                    </>
                                ) : (
                                    <>
                                        <Database className="h-4 w-4 mr-2" />
                                        Create Snapshot
                                    </>
                                )}
                            </Button>
                        </div>

                        {/* Upload Snapshot Button */}
                        <div className="flex items-center justify-between">
                            <div>
                                <h3 className="text-lg font-medium text-secondary">Upload Snapshot</h3>
                                <p className="text-sm text-muted-foreground">Load pre-built snapshot from file</p>
                            </div>
                            <Button
                                className="rounded-xl bg-blue-600 text-white hover:bg-blue-700"
                                onClick={() => setShowUploadDialog(true)}
                                disabled={processing}
                            >
                                <Upload className="h-4 w-4 mr-2" />
                                Upload Snapshot
                            </Button>
                        </div>

                        {/* Snapshot List */}
                        <div className="space-y-2">
                            <h3 className="text-lg font-medium text-secondary">Available Snapshots</h3>
                            {snapshotLoading ? (
                                <div className="flex items-center justify-center py-8">
                                    <Loader2 className="h-6 w-6 animate-spin text-primary" />
                                </div>
                            ) : snapshots.length === 0 ? (
                                <div className="text-center py-8 text-muted-foreground">
                                    No snapshots available
                                </div>
                            ) : (
                                <ScrollArea className="h-[300px] rounded-xl border p-4">
                                    <div className="space-y-2">
                                        {snapshots.map((snapshot) => (
                                            <div
                                                key={snapshot.name}
                                                className="flex items-center justify-between p-3 rounded-lg bg-muted/50 hover:bg-muted transition-colors"
                                            >
                                                <div className="flex-1">
                                                    <p className="font-medium text-secondary">{snapshot.name}</p>
                                                    <p className="text-sm text-muted-foreground">
                                                        {new Date(snapshot.created_at).toLocaleString()} • {(snapshot.size / 1024 / 1024).toFixed(2)} MB
                                                        {snapshot.has_metadata && (
                                                            <span className="ml-2 text-green-600">
                                                                • {snapshot.metadata_books} books, {snapshot.metadata_chapters} chapters
                                                            </span>
                                                        )}
                                                        {!snapshot.has_metadata && (
                                                            <span className="ml-2 text-yellow-600">• No metadata</span>
                                                        )}
                                                    </p>
                                                </div>
                                                <div className="flex gap-2">
                                                    <Button
                                                        size="sm"
                                                        className="rounded-xl bg-blue-600 text-white hover:bg-blue-700"
                                                        onClick={() => setSnapshotToRestore(snapshot.name)}
                                                        disabled={processing || !snapshot.has_metadata}
                                                        title={!snapshot.has_metadata ? 'Metadata required for restore' : ''}
                                                    >
                                                        <Upload className="h-4 w-4 mr-1" />
                                                        Restore
                                                    </Button>
                                                    <Button
                                                        size="sm"
                                                        className="rounded-xl bg-green-600 text-white hover:bg-green-700"
                                                        onClick={() => downloadSnapshot(snapshot.name)}
                                                    >
                                                        <Download className="h-4 w-4 mr-1" />
                                                        Snapshot
                                                    </Button>
                                                    <Button
                                                        size="sm"
                                                        className="rounded-xl bg-green-500 text-white hover:bg-green-600"
                                                        onClick={() => downloadMetadata(snapshot.name)}
                                                        disabled={!snapshot.has_metadata}
                                                    >
                                                        <FileJson className="h-4 w-4 mr-1" />
                                                        Metadata
                                                    </Button>
                                                    <Button
                                                        size="sm"
                                                        variant="destructive"
                                                        className="rounded-xl"
                                                        onClick={() => setSnapshotToDelete(snapshot.name)}
                                                    >
                                                        <Trash2 className="h-4 w-4" />
                                                    </Button>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </ScrollArea>
                            )}
                        </div>
                    </CardContent>
                </Card>
            )}

            {/* Restore Snapshot Dialog */}
            <AlertDialog open={!!snapshotToRestore} onOpenChange={(open: boolean) => !open && setSnapshotToRestore(null)}>
                <AlertDialogContent className="bg-muted border-primary/20">
                    <AlertDialogHeader>
                        <AlertDialogTitle className="text-primary">Restore Snapshot</AlertDialogTitle>
                        <AlertDialogDescription className="text-muted-foreground">
                            This will restore the snapshot &quot;{snapshotToRestore}&quot; and import its stored metadata (books and chapters). Existing data will be overwritten.
                        </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                        <AlertDialogCancel className="border-muted-foreground/30 text-secondary hover:bg-muted-foreground/10 hover:text-secondary">Cancel</AlertDialogCancel>
                        <AlertDialogAction
                            className="bg-blue-600 text-white hover:bg-blue-700"
                            onClick={() => snapshotToRestore && restoreSnapshot(snapshotToRestore)}
                        >
                            Restore
                        </AlertDialogAction>
                    </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialog>

            {/* Delete Snapshot Dialog */}
            <AlertDialog open={!!snapshotToDelete} onOpenChange={(open: boolean) => !open && setSnapshotToDelete(null)}>
                <AlertDialogContent className="bg-muted border-primary/20">
                    <AlertDialogHeader>
                        <AlertDialogTitle className="text-primary">Delete Snapshot</AlertDialogTitle>
                        <AlertDialogDescription className="text-muted-foreground">
                            Are you sure you want to delete the snapshot &quot;{snapshotToDelete}&quot;? This will also remove its metadata file. This action cannot be undone.
                        </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                        <AlertDialogCancel className="border-muted-foreground/30 text-secondary hover:bg-muted-foreground/10 hover:text-secondary">Cancel</AlertDialogCancel>
                        <AlertDialogAction
                            className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                            onClick={() => snapshotToDelete && deleteSnapshot(snapshotToDelete)}
                        >
                            Delete
                        </AlertDialogAction>
                    </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialog>

            {/* Upload Snapshot Dialog */}
            <AlertDialog open={showUploadDialog} onOpenChange={(open: boolean) => {
                if (!open) {
                    setShowUploadDialog(false);
                    setUploadSnapshotFile(null);
                    setUploadMetadataFile(null);
                }
            }}>
                <AlertDialogContent className="bg-muted border-primary/20">
                    <AlertDialogHeader>
                        <AlertDialogTitle className="text-primary">Upload External Snapshot</AlertDialogTitle>
                        <AlertDialogDescription className="text-muted-foreground">
                            Upload a Qdrant snapshot file and its metadata JSON. Both files are required to restore books and chapters data.
                        </AlertDialogDescription>
                    </AlertDialogHeader>
                    <div className="space-y-4 py-4">
                        <div className="space-y-2">
                            <p className="text-sm font-medium text-secondary">Snapshot File (.snapshot)</p>
                            <Input
                                type="file"
                                accept=".snapshot"
                                className="border-muted-foreground/30 text-secondary file:text-secondary file:mr-3"
                                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setUploadSnapshotFile(e.target.files?.[0] || null)}
                            />
                        </div>
                        <div className="space-y-2">
                            <p className="text-sm font-medium text-secondary">Metadata File (.json)</p>
                            <Input
                                type="file"
                                accept=".json"
                                className="border-muted-foreground/30 text-secondary file:text-secondary file:mr-3"
                                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setUploadMetadataFile(e.target.files?.[0] || null)}
                            />
                        </div>
                    </div>
                    <AlertDialogFooter>
                        <AlertDialogCancel className="border-muted-foreground/30 text-secondary hover:bg-muted-foreground/10 hover:text-secondary">Cancel</AlertDialogCancel>
                        <AlertDialogAction
                            className="bg-primary text-primary-foreground hover:bg-primary/90"
                            onClick={uploadSnapshot}
                            disabled={!uploadSnapshotFile || !uploadMetadataFile}
                        >
                            Upload & Restore
                        </AlertDialogAction>
                    </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialog>
        </div>
    );
};

export default ContentManagement;