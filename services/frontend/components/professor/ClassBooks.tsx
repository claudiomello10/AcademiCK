'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import {
    AlertCircle, Book, CheckCircle2, Library, Loader2, Plus,
    Trash2, Upload, X,
} from 'lucide-react';

import { API_ENDPOINTS } from '@/config/constants';
import { usePortal } from '@/components/portal/PortalShell';

interface ClassInfo { id: string; name: string; subject: string }

interface ClassBook {
    id: string;
    name: string;
    processing_status: string;
    owner_user_id?: string | null;
}

interface CatalogBook {
    id: string;
    name: string;
    owned: boolean;
    attached: boolean;
}

interface UploadJob {
    job_id: string;
    filename: string;
    status: string;
    progress: number;
    error?: string;
}

const ClassBooks = ({ features }: { features: { book_upload_enabled: boolean } }) => {
    const { portalFetch } = usePortal();
    const [classes, setClasses] = useState<ClassInfo[]>([]);
    const [classId, setClassId] = useState('');
    const [books, setBooks] = useState<ClassBook[]>([]);
    const [catalog, setCatalog] = useState<CatalogBook[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState('');
    const [showCatalog, setShowCatalog] = useState(false);
    const [attaching, setAttaching] = useState<string | null>(null);
    const [uploading, setUploading] = useState(false);
    const [jobs, setJobs] = useState<UploadJob[]>([]);
    const pollingRef = useRef<NodeJS.Timeout | null>(null);

    useEffect(() => {
        (async () => {
            const response = await portalFetch(API_ENDPOINTS.professor.classes);
            if (response.ok) {
                const list = (await response.json()).classes;
                setClasses(list);
                if (list.length > 0) setClassId(prev => prev || list[0].id);
            }
        })();
        return () => {
            if (pollingRef.current) clearInterval(pollingRef.current);
        };
    }, [portalFetch]);

    const fetchBooks = useCallback(async () => {
        if (!classId) return;
        setLoading(true);
        try {
            const response = await portalFetch(API_ENDPOINTS.professor.classBooks(classId));
            if (response.ok) setBooks((await response.json()).books);
        } finally {
            setLoading(false);
        }
    }, [portalFetch, classId]);

    useEffect(() => { fetchBooks(); }, [fetchBooks]);

    const openCatalog = async () => {
        setShowCatalog(true);
        const response = await portalFetch(API_ENDPOINTS.professor.bookCatalog(classId));
        if (response.ok) setCatalog((await response.json()).books);
    };

    const attach = async (book: CatalogBook) => {
        setAttaching(book.id);
        setError(null);
        try {
            const response = await portalFetch(API_ENDPOINTS.professor.attachBook(classId, book.id), {
                method: 'POST',
            });
            const data = await response.json();
            if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Falha ao adicionar livro');
            setCatalog(prev => prev.map(b => b.id === book.id ? { ...b, attached: true } : b));
            fetchBooks();
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Falha ao adicionar livro');
        } finally {
            setAttaching(null);
        }
    };

    const detach = async (bookId: string) => {
        const response = await portalFetch(API_ENDPOINTS.professor.detachBook(classId, bookId), {
            method: 'DELETE',
        });
        if (response.ok) {
            setBooks(prev => prev.filter(b => b.id !== bookId));
            setCatalog(prev => prev.map(b => b.id === bookId ? { ...b, attached: false } : b));
        }
    };

    const deleteBook = async (bookId: string) => {
        setError(null);
        const response = await portalFetch(API_ENDPOINTS.professor.deleteBook(classId, bookId), {
            method: 'DELETE',
        });
        const data = await response.json();
        if (!response.ok) {
            setError(typeof data.detail === 'string' ? data.detail : 'Falha ao excluir livro');
            return;
        }
        setSuccess(`Livro "${data.book_name}" excluído`);
        fetchBooks();
    };

    const pollJobs = useCallback((activeJobs: UploadJob[]) => {
        if (pollingRef.current) clearInterval(pollingRef.current);
        pollingRef.current = setInterval(async () => {
            const updated = await Promise.all(activeJobs.map(async job => {
                if (job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') {
                    return job;
                }
                const response = await portalFetch(API_ENDPOINTS.professor.pdfJobStatus(job.job_id));
                if (!response.ok) return job;
                const data = await response.json();
                return {
                    ...job,
                    status: data.status,
                    progress: data.status === 'completed' ? 100 : Math.round((data.progress || 0) * 100),
                    error: data.error,
                };
            }));
            setJobs(updated);
            activeJobs.splice(0, activeJobs.length, ...updated);
            if (updated.every(j => ['completed', 'failed', 'cancelled'].includes(j.status))) {
                if (pollingRef.current) clearInterval(pollingRef.current);
                pollingRef.current = null;
                fetchBooks();
            }
        }, 3000);
    }, [portalFetch, fetchBooks]);

    const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const files = Array.from(event.target.files || []);
        event.target.value = '';
        if (!files.length || !classId) return;

        setUploading(true);
        setError(null);
        setSuccess('');
        const formData = new FormData();
        files.forEach(file => formData.append('files', file));

        try {
            const response = await portalFetch(API_ENDPOINTS.professor.uploadBooks(classId), {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(typeof data.detail === 'string' ? data.detail : 'Falha no envio');
            }
            const newJobs: UploadJob[] = data.jobs.map((job: UploadJob) => ({
                ...job, progress: 0,
            }));
            (data.errors || []).forEach((err: { filename: string; error: string }) => {
                newJobs.push({ job_id: '', filename: err.filename, status: 'failed', progress: 0, error: err.error });
            });
            setJobs(newJobs);
            const active = newJobs.filter(j => j.status !== 'failed');
            if (active.length) pollJobs(newJobs);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Falha no envio');
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="space-y-6">
            <div className="flex items-center gap-3 flex-wrap">
                <Select value={classId} onValueChange={setClassId}>
                    <SelectTrigger className="rounded-xl w-72">
                        <SelectValue placeholder="Selecione a turma" />
                    </SelectTrigger>
                    <SelectContent>
                        {classes.map(cls => (
                            <SelectItem key={cls.id} value={cls.id}>{cls.name}</SelectItem>
                        ))}
                    </SelectContent>
                </Select>

                {classId && (
                    <div className="flex gap-2 ml-auto">
                        <Button variant="outline" className="rounded-xl" onClick={openCatalog}>
                            <Library className="h-4 w-4 mr-2" />
                            Selecionar livros
                        </Button>
                        {features.book_upload_enabled && (
                            <>
                                <Button
                                    className="rounded-xl bg-primary text-primary-foreground hover:bg-primary/90"
                                    onClick={() => document.getElementById('prof-file-upload')?.click()}
                                    disabled={uploading}
                                >
                                    {uploading ? (
                                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                    ) : (
                                        <Upload className="h-4 w-4 mr-2" />
                                    )}
                                    Enviar novo livro
                                </Button>
                                <Input
                                    id="prof-file-upload"
                                    type="file"
                                    multiple
                                    accept=".pdf"
                                    className="hidden"
                                    onChange={handleUpload}
                                />
                            </>
                        )}
                    </div>
                )}
            </div>

            {error && (
                <Alert variant="destructive" className="rounded-xl">
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>Erro</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                </Alert>
            )}
            {success && (
                <Alert className="rounded-xl bg-green-50 border-green-200">
                    <AlertTitle className="text-green-800">Sucesso</AlertTitle>
                    <AlertDescription className="text-green-700">{success}</AlertDescription>
                </Alert>
            )}

            {jobs.length > 0 && (
                <Card className="rounded-xl border border-primary/20">
                    <CardHeader>
                        <CardTitle className="text-primary">Processamento</CardTitle>
                        <CardDescription>Os livros ficam disponíveis para a turma ao concluir</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-3">
                        {jobs.map((job, i) => (
                            <div key={i} className="space-y-1">
                                <div className="flex items-center justify-between text-sm">
                                    <div className="flex items-center gap-2">
                                        {job.status === 'completed' ? (
                                            <CheckCircle2 className="h-4 w-4 text-green-500" />
                                        ) : job.status === 'failed' ? (
                                            <AlertCircle className="h-4 w-4 text-red-500" />
                                        ) : (
                                            <Loader2 className="h-4 w-4 animate-spin text-primary" />
                                        )}
                                        <span className="text-secondary">{job.filename}</span>
                                    </div>
                                    <span className="text-muted-foreground">
                                        {job.status === 'completed' ? 'Concluído' :
                                         job.status === 'failed' ? 'Falhou' : 'Processando...'}
                                    </span>
                                </div>
                                <Progress value={job.status === 'failed' ? 100 : job.progress} className="h-2" />
                                {job.error && (
                                    <p className="text-sm text-red-600">{job.error}</p>
                                )}
                            </div>
                        ))}
                    </CardContent>
                </Card>
            )}

            {classId && (
                <Card className="rounded-xl border border-primary/20">
                    <CardHeader>
                        <CardTitle className="text-primary">Livros da turma</CardTitle>
                        <CardDescription>O que os alunos desta turma podem consultar</CardDescription>
                    </CardHeader>
                    <CardContent>
                        {loading ? (
                            <div className="flex justify-center py-8">
                                <Loader2 className="h-6 w-6 animate-spin text-primary" />
                            </div>
                        ) : books.length === 0 ? (
                            <div className="text-center py-8 text-muted-foreground">
                                Nenhum livro nesta turma. Selecione do catálogo ou envie um novo.
                            </div>
                        ) : (
                            <div className="space-y-2">
                                {books.map(book => (
                                    <div key={book.id} className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                                        <div className="flex items-center gap-3 min-w-0">
                                            <Book className="h-5 w-5 text-primary shrink-0" />
                                            <div className="min-w-0">
                                                <p className="font-medium text-secondary truncate">{book.name}</p>
                                                <p className="text-xs text-muted-foreground">
                                                    {book.processing_status === 'completed' ? 'Disponível' : 'Processando...'}
                                                    {book.owner_user_id ? ' · enviado por você' : ' · catálogo geral'}
                                                </p>
                                            </div>
                                        </div>
                                        <div className="flex gap-1 shrink-0">
                                            <Button
                                                variant="ghost"
                                                size="icon"
                                                className="h-8 w-8 text-muted-foreground hover:text-destructive"
                                                onClick={() => detach(book.id)}
                                                title="Remover da turma (mantém o livro)"
                                            >
                                                <X className="h-4 w-4" />
                                            </Button>
                                            {book.owner_user_id && (
                                                <Button
                                                    variant="ghost"
                                                    size="icon"
                                                    className="h-8 w-8 text-destructive hover:text-white hover:bg-destructive"
                                                    onClick={() => deleteBook(book.id)}
                                                    title="Excluir o livro definitivamente"
                                                >
                                                    <Trash2 className="h-4 w-4" />
                                                </Button>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </CardContent>
                </Card>
            )}

            <Dialog open={showCatalog} onOpenChange={setShowCatalog}>
                <DialogContent className="max-w-lg bg-muted border-primary/20">
                    <DialogHeader>
                        <DialogTitle className="text-primary">Catálogo de livros</DialogTitle>
                        <DialogDescription>
                            Livros do catálogo geral e seus envios. Adicione à turma selecionada.
                        </DialogDescription>
                    </DialogHeader>
                    <div className="space-y-2 max-h-96 overflow-y-auto py-2">
                        {catalog.length === 0 && (
                            <p className="text-sm text-muted-foreground text-center py-4">
                                Nenhum livro disponível no catálogo
                            </p>
                        )}
                        {catalog.map(book => (
                            <div key={book.id} className="flex items-center justify-between p-2 rounded-lg bg-background/50">
                                <div className="min-w-0">
                                    <p className="text-sm font-medium text-secondary truncate">{book.name}</p>
                                    <p className="text-xs text-muted-foreground">
                                        {book.owned ? 'Enviado por você' : 'Catálogo geral'}
                                    </p>
                                </div>
                                {book.attached ? (
                                    <Button
                                        variant="outline"
                                        size="sm"
                                        className="rounded-xl shrink-0"
                                        onClick={() => detach(book.id)}
                                    >
                                        <X className="h-4 w-4 mr-1" /> Remover
                                    </Button>
                                ) : (
                                    <Button
                                        size="sm"
                                        className="rounded-xl bg-primary text-primary-foreground shrink-0"
                                        onClick={() => attach(book)}
                                        disabled={attaching === book.id}
                                    >
                                        {attaching === book.id ? (
                                            <Loader2 className="h-4 w-4 animate-spin" />
                                        ) : (
                                            <><Plus className="h-4 w-4 mr-1" /> Adicionar</>
                                        )}
                                    </Button>
                                )}
                            </div>
                        ))}
                    </div>
                </DialogContent>
            </Dialog>
        </div>
    );
};

export default ClassBooks;
