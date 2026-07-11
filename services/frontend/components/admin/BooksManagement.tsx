'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from '@/components/ui/alert-dialog';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
    Book, Database, FileText, Upload,
    Trash2, AlertCircle, Loader2,
    ChevronUp, ChevronDown, StopCircle
} from 'lucide-react';

import { API_ENDPOINTS } from '@/config/constants';
import { usePortal } from '@/components/portal/PortalShell';

interface BookInfo {
    name: string;
    total_chapters: number;
    total_chunks: number;
    processing_status?: string;
    processing_method?: string;
}

const BooksManagement = () => {
    const { portalFetch } = usePortal();
    const [files, setFiles] = useState<File[]>([]);
    const [uploading, setUploading] = useState(false);
    const [deleting, setDeleting] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState('');
    const [stats, setStats] = useState({ total_books: 0, total_chunks: 0, total_embeddings: 0 });
    const [books, setBooks] = useState<BookInfo[]>([]);
    const [expanded, setExpanded] = useState<Record<string, boolean>>({});
    const [uploadEnabled, setUploadEnabled] = useState(true);

    const fetchStats = useCallback(async () => {
        const response = await portalFetch(API_ENDPOINTS.admin.contentStats);
        if (response.ok) setStats(await response.json());
    }, [portalFetch]);

    const fetchBooks = useCallback(async () => {
        const response = await portalFetch(API_ENDPOINTS.admin.bookList);
        if (response.ok) setBooks(await response.json());
    }, [portalFetch]);

    useEffect(() => {
        fetchStats();
        fetchBooks();
        (async () => {
            const response = await portalFetch(API_ENDPOINTS.admin.features);
            if (response.ok) {
                const flags = await response.json();
                setUploadEnabled(flags.pdf_upload_enabled !== false);
            }
        })();
    }, [portalFetch, fetchStats, fetchBooks]);

    const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        const uploadedFiles = Array.from(event.target.files || []) as File[];
        setFiles(prev => [...prev, ...uploadedFiles]);
    };

    const processFiles = async () => {
        setUploading(true);
        setError(null);
        setSuccess('');

        const formData = new FormData();
        files.forEach(file => formData.append('files', file));

        try {
            const response = await portalFetch(API_ENDPOINTS.admin.uploadPdfs, {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || 'Error processing PDFs');

            const failed = (data.errors || []).length;
            setSuccess(
                `Submitted ${data.jobs.length} PDF(s) for processing` +
                (failed ? ` — ${failed} failed` : '') +
                '. Track progress in the Jobs section.'
            );
            if (failed) {
                setError(data.errors.map((e: { filename: string; error: string }) =>
                    `${e.filename}: ${e.error}`).join('\n'));
            }
            setFiles([]);
            fetchBooks();
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Unknown error');
        } finally {
            setUploading(false);
        }
    };

    const handleDeleteBook = async (bookName: string) => {
        setDeleting(bookName);
        setError(null);
        setSuccess('');
        try {
            const response = await portalFetch(API_ENDPOINTS.admin.deleteBook(bookName), {
                method: 'DELETE',
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || 'Error deleting book');
            setSuccess(`Deleted "${bookName}" (${data.vectors_deleted} embeddings removed)`);
            fetchStats();
            fetchBooks();
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Unknown error');
        } finally {
            setDeleting(null);
        }
    };

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <p className="text-sm text-muted-foreground">
                    Global library. Attach books to classes in the Classes section.
                </p>
                {uploadEnabled && (
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

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {[
                    { icon: Book, label: 'Total Books', value: stats.total_books },
                    { icon: FileText, label: 'Total Chunks', value: stats.total_chunks },
                    { icon: Database, label: 'Total Embeddings', value: stats.total_embeddings },
                ].map(({ icon: Icon, label, value }) => (
                    <Card key={label} className="rounded-xl border border-primary/20">
                        <CardContent className="p-4">
                            <div className="flex items-center gap-3">
                                <Icon className="h-5 w-5 text-primary" />
                                <div className="text-secondary">
                                    <p className="text-sm font-medium">{label}</p>
                                    <p className="text-2xl font-bold">{value}</p>
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                ))}
            </div>

            {uploadEnabled && files.length > 0 && (
                <Card className="rounded-xl">
                    <CardHeader>
                        <CardTitle className="text-primary">Files to Process</CardTitle>
                        <CardDescription>Review and process selected PDF files</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <ScrollArea className="h-64">
                            <div className="space-y-2">
                                {files.map((file, index) => (
                                    <div key={index} className="flex items-center justify-between bg-primary text-primary-foreground p-3 rounded-lg">
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
                                            onClick={() => setFiles(prev => prev.filter((_, i) => i !== index))}
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
                                disabled={uploading}
                            >
                                {uploading ? (
                                    <>
                                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                        Submitting...
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

            {error && (
                <Alert variant="destructive" className="rounded-xl">
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription className="whitespace-pre-line">{error}</AlertDescription>
                </Alert>
            )}

            {success && (
                <Alert className="rounded-xl bg-green-50 border-green-200">
                    <AlertTitle className="text-green-800">Success</AlertTitle>
                    <AlertDescription className="text-green-700">{success}</AlertDescription>
                </Alert>
            )}

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
                                                ) : book.processing_status === 'cancelled' ? (
                                                    <StopCircle className="h-5 w-5 text-orange-500" />
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
                                                        {book.processing_status === 'cancelled' && (
                                                            <span className="text-xs px-2 py-0.5 rounded-full bg-orange-100 text-orange-700 font-medium">
                                                                Cancelled
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
                                                                Are you sure you want to delete &quot;{book.name}&quot;? This will remove all embeddings associated with this book and detach it from every class. This action cannot be undone.
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
                                                    <p><strong>Processing Method:</strong> {book.processing_method === 'default' ? 'Default' : book.processing_method === 'docling' ? 'Docling (layout-based)' : 'Unknown'}</p>
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
        </div>
    );
};

export default BooksManagement;
