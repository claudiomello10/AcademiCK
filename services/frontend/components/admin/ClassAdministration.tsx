'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import {
    AlertCircle, Book, ChevronDown, ChevronUp, GraduationCap,
    Loader2, Plus, Trash2, UserPlus, Users,
} from 'lucide-react';

import { API_ENDPOINTS } from '@/config/constants';
import { usePortal } from '@/components/portal/PortalShell';

interface ClassInfo {
    id: string;
    name: string;
    subject: string;
    description?: string;
    professor_id: string;
    professor_username?: string;
    join_code?: string;
    is_active: boolean;
    member_count?: number;
    book_count?: number;
}

interface Student {
    id: string;
    username: string;
    registration_number?: string;
    enrolled_via: string;
}

interface ClassBook {
    id: string;
    name: string;
    processing_status: string;
}

const ClassAdministration = () => {
    const { portalFetch } = usePortal();
    const [classes, setClasses] = useState<ClassInfo[]>([]);
    const [professors, setProfessors] = useState<{ id: string; username: string }[]>([]);
    const [catalog, setCatalog] = useState<{ name: string }[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState('');
    const [expanded, setExpanded] = useState<string | null>(null);
    const [students, setStudents] = useState<Student[]>([]);
    const [classBooks, setClassBooks] = useState<ClassBook[]>([]);
    const [allBooks, setAllBooks] = useState<{ id: string; name: string }[]>([]);
    const [detailLoading, setDetailLoading] = useState(false);

    const [showCreate, setShowCreate] = useState(false);
    const [newClass, setNewClass] = useState({ name: '', subject: '', description: '', professor_id: '' });
    const [creating, setCreating] = useState(false);
    const [assignValue, setAssignValue] = useState('');
    const [attachBookId, setAttachBookId] = useState('');

    const fetchClasses = useCallback(async () => {
        const response = await portalFetch(API_ENDPOINTS.admin.classes);
        if (response.ok) setClasses((await response.json()).classes);
        setLoading(false);
    }, [portalFetch]);

    useEffect(() => {
        fetchClasses();
        (async () => {
            const response = await portalFetch(API_ENDPOINTS.admin.professors);
            if (response.ok) setProfessors((await response.json()).professors);
        })();
    }, [portalFetch, fetchClasses]);

    const openDetail = async (classId: string) => {
        if (expanded === classId) {
            setExpanded(null);
            return;
        }
        setExpanded(classId);
        setDetailLoading(true);
        try {
            const [studentsRes, booksRes, catalogRes] = await Promise.all([
                portalFetch(API_ENDPOINTS.admin.classStudents(classId)),
                portalFetch(API_ENDPOINTS.admin.classBooks(classId)),
                portalFetch(API_ENDPOINTS.admin.bookList),
            ]);
            if (studentsRes.ok) setStudents((await studentsRes.json()).students);
            if (booksRes.ok) setClassBooks((await booksRes.json()).books);
            if (catalogRes.ok) {
                const books = await catalogRes.json();
                setCatalog(books.filter((b: any) => b.processing_status === 'completed'));
            }
        } finally {
            setDetailLoading(false);
        }
    };

    // The admin book list is keyed by name; resolve ids via the professor catalog shape
    useEffect(() => {
        (async () => {
            const response = await portalFetch(API_ENDPOINTS.professor.bookCatalog());
            if (response.ok) setAllBooks((await response.json()).books);
        })();
    }, [portalFetch]);

    const createClass = async () => {
        setCreating(true);
        setError(null);
        try {
            const response = await portalFetch(API_ENDPOINTS.admin.classes, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(newClass),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Failed to create class');
            setSuccess(`Class "${data.name}" created`);
            setShowCreate(false);
            setNewClass({ name: '', subject: '', description: '', professor_id: '' });
            fetchClasses();
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to create class');
        } finally {
            setCreating(false);
        }
    };

    const deleteClass = async (classId: string) => {
        const response = await portalFetch(API_ENDPOINTS.admin.classById(classId), { method: 'DELETE' });
        if (response.ok) {
            setSuccess('Class deleted');
            setExpanded(null);
            fetchClasses();
        }
    };

    const assignStudent = async (classId: string) => {
        setError(null);
        const value = assignValue.trim();
        if (!value) return;
        const response = await portalFetch(API_ENDPOINTS.admin.classStudents(classId), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ registration_number: value }),
        });
        const data = await response.json();
        if (!response.ok) {
            setError(typeof data.detail === 'string' ? data.detail : 'Failed to assign student');
            return;
        }
        setAssignValue('');
        setStudents(prev => [...prev, { ...data, enrolled_via: 'admin' }]);
        fetchClasses();
    };

    const removeStudent = async (classId: string, userId: string) => {
        const response = await portalFetch(API_ENDPOINTS.admin.removeClassStudent(classId, userId), {
            method: 'DELETE',
        });
        if (response.ok) {
            setStudents(prev => prev.filter(s => s.id !== userId));
            fetchClasses();
        }
    };

    const attachBook = async (classId: string) => {
        setError(null);
        if (!attachBookId) return;
        const response = await portalFetch(API_ENDPOINTS.admin.attachClassBook(classId, attachBookId), {
            method: 'POST',
        });
        const data = await response.json();
        if (!response.ok) {
            setError(typeof data.detail === 'string' ? data.detail : 'Failed to attach book');
            return;
        }
        setAttachBookId('');
        setClassBooks(prev => [...prev, { id: data.book.id, name: data.book.name, processing_status: 'completed' }]);
        fetchClasses();
    };

    const detachBook = async (classId: string, bookId: string) => {
        const response = await portalFetch(API_ENDPOINTS.admin.detachClassBook(classId, bookId), {
            method: 'DELETE',
        });
        if (response.ok) {
            setClassBooks(prev => prev.filter(b => b.id !== bookId));
            fetchClasses();
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center py-12">
                <Loader2 className="h-6 w-6 animate-spin text-primary" />
            </div>
        );
    }

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <p className="text-sm text-muted-foreground">
                    All classes across professors: enrollment and book access.
                </p>
                <Button
                    className="rounded-xl bg-primary text-primary-foreground hover:bg-primary/90"
                    onClick={() => setShowCreate(true)}
                >
                    <Plus className="h-4 w-4 mr-2" />
                    New Class
                </Button>
            </div>

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

            <div className="space-y-2">
                {classes.length === 0 && (
                    <div className="text-center py-12 text-muted-foreground">No classes yet</div>
                )}
                {classes.map(cls => (
                    <Card key={cls.id} className="rounded-xl border border-primary/20">
                        <CardContent className="p-4">
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-3 min-w-0">
                                    <GraduationCap className="h-5 w-5 text-primary shrink-0" />
                                    <div className="min-w-0">
                                        <h4 className="font-medium text-secondary truncate">{cls.name}</h4>
                                        <p className="text-sm text-muted-foreground truncate">
                                            {cls.subject} · {cls.professor_username} · {cls.member_count} students · {cls.book_count} books
                                            {cls.join_code && <> · code <span className="font-mono">{cls.join_code}</span></>}
                                        </p>
                                    </div>
                                </div>
                                <div className="flex items-center gap-2 shrink-0">
                                    <Button
                                        variant="ghost"
                                        size="icon"
                                        className="h-8 w-8"
                                        onClick={() => openDetail(cls.id)}
                                    >
                                        {expanded === cls.id ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                                    </Button>
                                    <Button
                                        variant="ghost"
                                        size="icon"
                                        className="h-8 w-8 text-destructive hover:text-white hover:bg-destructive"
                                        onClick={() => deleteClass(cls.id)}
                                    >
                                        <Trash2 className="h-4 w-4" />
                                    </Button>
                                </div>
                            </div>

                            {expanded === cls.id && (
                                <div className="mt-4 pt-4 border-t border-border grid grid-cols-1 md:grid-cols-2 gap-6">
                                    {detailLoading ? (
                                        <div className="col-span-2 flex justify-center py-6">
                                            <Loader2 className="h-5 w-5 animate-spin text-primary" />
                                        </div>
                                    ) : (
                                        <>
                                            <div className="space-y-3">
                                                <h5 className="font-medium text-secondary flex items-center gap-2">
                                                    <Users className="h-4 w-4" /> Students
                                                </h5>
                                                <div className="flex gap-2">
                                                    <Input
                                                        placeholder="Registration number"
                                                        value={assignValue}
                                                        onChange={e => setAssignValue(e.target.value)}
                                                        className="rounded-xl h-9"
                                                    />
                                                    <Button
                                                        size="sm"
                                                        className="rounded-xl bg-primary text-primary-foreground"
                                                        onClick={() => assignStudent(cls.id)}
                                                    >
                                                        <UserPlus className="h-4 w-4" />
                                                    </Button>
                                                </div>
                                                <div className="space-y-1 max-h-56 overflow-y-auto">
                                                    {students.length === 0 && (
                                                        <p className="text-sm text-muted-foreground">No students enrolled</p>
                                                    )}
                                                    {students.map(student => (
                                                        <div key={student.id} className="flex items-center justify-between text-sm bg-muted/50 rounded-lg px-3 py-1.5">
                                                            <span className="text-secondary">
                                                                {student.username}
                                                                {student.registration_number && (
                                                                    <span className="text-muted-foreground ml-2 font-mono text-xs">
                                                                        {student.registration_number}
                                                                    </span>
                                                                )}
                                                            </span>
                                                            <Button
                                                                variant="ghost"
                                                                size="icon"
                                                                className="h-6 w-6 text-muted-foreground hover:text-destructive"
                                                                onClick={() => removeStudent(cls.id, student.id)}
                                                            >
                                                                <Trash2 className="h-3.5 w-3.5" />
                                                            </Button>
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>

                                            <div className="space-y-3">
                                                <h5 className="font-medium text-secondary flex items-center gap-2">
                                                    <Book className="h-4 w-4" /> Books
                                                </h5>
                                                <div className="flex gap-2">
                                                    <Select value={attachBookId} onValueChange={setAttachBookId}>
                                                        <SelectTrigger className="rounded-xl h-9">
                                                            <SelectValue placeholder="Attach a book..." />
                                                        </SelectTrigger>
                                                        <SelectContent>
                                                            {allBooks
                                                                .filter(b => !classBooks.some(cb => cb.id === b.id))
                                                                .map(book => (
                                                                    <SelectItem key={book.id} value={book.id}>
                                                                        {book.name}
                                                                    </SelectItem>
                                                                ))}
                                                        </SelectContent>
                                                    </Select>
                                                    <Button
                                                        size="sm"
                                                        className="rounded-xl bg-primary text-primary-foreground"
                                                        onClick={() => attachBook(cls.id)}
                                                        disabled={!attachBookId}
                                                    >
                                                        <Plus className="h-4 w-4" />
                                                    </Button>
                                                </div>
                                                <div className="space-y-1 max-h-56 overflow-y-auto">
                                                    {classBooks.length === 0 && (
                                                        <p className="text-sm text-muted-foreground">No books attached</p>
                                                    )}
                                                    {classBooks.map(book => (
                                                        <div key={book.id} className="flex items-center justify-between text-sm bg-muted/50 rounded-lg px-3 py-1.5">
                                                            <span className="text-secondary truncate">{book.name}</span>
                                                            <Button
                                                                variant="ghost"
                                                                size="icon"
                                                                className="h-6 w-6 text-muted-foreground hover:text-destructive"
                                                                onClick={() => detachBook(cls.id, book.id)}
                                                            >
                                                                <Trash2 className="h-3.5 w-3.5" />
                                                            </Button>
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        </>
                                    )}
                                </div>
                            )}
                        </CardContent>
                    </Card>
                ))}
            </div>

            <Dialog open={showCreate} onOpenChange={setShowCreate}>
                <DialogContent className="bg-muted border-primary/20">
                    <DialogHeader>
                        <DialogTitle className="text-primary">New Class</DialogTitle>
                        <DialogDescription>Create a class and assign its professor</DialogDescription>
                    </DialogHeader>
                    <div className="space-y-4 py-2">
                        <Input
                            placeholder="Class name"
                            value={newClass.name}
                            onChange={e => setNewClass(prev => ({ ...prev, name: e.target.value }))}
                            className="rounded-xl"
                        />
                        <Input
                            placeholder="Subject"
                            value={newClass.subject}
                            onChange={e => setNewClass(prev => ({ ...prev, subject: e.target.value }))}
                            className="rounded-xl"
                        />
                        <Input
                            placeholder="Description (optional)"
                            value={newClass.description}
                            onChange={e => setNewClass(prev => ({ ...prev, description: e.target.value }))}
                            className="rounded-xl"
                        />
                        <Select
                            value={newClass.professor_id}
                            onValueChange={value => setNewClass(prev => ({ ...prev, professor_id: value }))}
                        >
                            <SelectTrigger className="rounded-xl">
                                <SelectValue placeholder="Professor" />
                            </SelectTrigger>
                            <SelectContent>
                                {professors.map(prof => (
                                    <SelectItem key={prof.id} value={prof.id}>{prof.username}</SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                    </div>
                    <DialogFooter>
                        <Button variant="outline" className="rounded-xl" onClick={() => setShowCreate(false)}>
                            Cancel
                        </Button>
                        <Button
                            className="rounded-xl bg-primary text-primary-foreground"
                            onClick={createClass}
                            disabled={creating || !newClass.name || !newClass.subject || !newClass.professor_id}
                        >
                            {creating ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Create'}
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </div>
    );
};

export default ClassAdministration;
