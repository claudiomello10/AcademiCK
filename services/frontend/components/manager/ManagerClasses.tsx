'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import {
    AlertCircle, ChevronDown, ChevronUp, GraduationCap,
    Loader2, Plus, Trash2, UserPlus, Users,
} from 'lucide-react';

import { API_ENDPOINTS } from '@/config/constants';
import { usePortal } from '@/components/portal/PortalShell';

interface ClassInfo {
    id: string;
    name: string;
    subject: string;
    professor_username?: string;
    member_count?: number;
    book_count?: number;
    join_code?: string;
}

interface Student {
    id: string;
    username: string;
    registration_number?: string;
}

const ManagerClasses = ({ assignEnabled }: { assignEnabled: boolean }) => {
    const { portalFetch } = usePortal();
    const [classes, setClasses] = useState<ClassInfo[]>([]);
    const [professors, setProfessors] = useState<{ id: string; username: string }[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState('');
    const [expanded, setExpanded] = useState<string | null>(null);
    const [students, setStudents] = useState<Student[]>([]);
    const [studentsLoading, setStudentsLoading] = useState(false);

    const [showCreate, setShowCreate] = useState(false);
    const [form, setForm] = useState({ name: '', subject: '', description: '', professor_id: '' });
    const [creating, setCreating] = useState(false);
    const [registration, setRegistration] = useState('');

    const fetchClasses = useCallback(async () => {
        const response = await portalFetch(API_ENDPOINTS.manager.classes);
        if (response.ok) setClasses((await response.json()).classes);
        setLoading(false);
    }, [portalFetch]);

    useEffect(() => {
        fetchClasses();
        (async () => {
            const response = await portalFetch(API_ENDPOINTS.manager.professors);
            if (response.ok) setProfessors((await response.json()).professors);
        })();
    }, [portalFetch, fetchClasses]);

    const openRoster = async (classId: string) => {
        if (expanded === classId) {
            setExpanded(null);
            return;
        }
        setExpanded(classId);
        setStudentsLoading(true);
        try {
            const response = await portalFetch(API_ENDPOINTS.manager.students(classId));
            if (response.ok) setStudents((await response.json()).students);
        } finally {
            setStudentsLoading(false);
        }
    };

    const createClass = async () => {
        setCreating(true);
        setError(null);
        try {
            const response = await portalFetch(API_ENDPOINTS.manager.classes, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(form),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Falha ao criar turma');
            setSuccess(`Turma "${data.name}" criada`);
            setShowCreate(false);
            setForm({ name: '', subject: '', description: '', professor_id: '' });
            fetchClasses();
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Falha ao criar turma');
        } finally {
            setCreating(false);
        }
    };

    const deleteClass = async (classId: string) => {
        const response = await portalFetch(API_ENDPOINTS.manager.classById(classId), { method: 'DELETE' });
        if (response.ok) {
            setSuccess('Turma excluída');
            setExpanded(null);
            fetchClasses();
        }
    };

    const assignStudent = async (classId: string) => {
        setError(null);
        const value = registration.trim();
        if (!value) return;
        const response = await portalFetch(API_ENDPOINTS.manager.students(classId), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ registration_number: value }),
        });
        const data = await response.json();
        if (!response.ok) {
            setError(typeof data.detail === 'string' ? data.detail : 'Falha ao matricular aluno');
            return;
        }
        setRegistration('');
        setStudents(prev => [...prev, data]);
        fetchClasses();
    };

    const removeStudent = async (classId: string, userId: string) => {
        const response = await portalFetch(API_ENDPOINTS.manager.removeStudent(classId, userId), {
            method: 'DELETE',
        });
        if (response.ok) {
            setStudents(prev => prev.filter(s => s.id !== userId));
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
                    Todas as turmas: crie turmas para os professores e matricule alunos.
                </p>
                <Button
                    className="rounded-xl bg-primary text-primary-foreground hover:bg-primary/90"
                    onClick={() => setShowCreate(true)}
                >
                    <Plus className="h-4 w-4 mr-2" />
                    Nova Turma
                </Button>
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

            <div className="space-y-2">
                {classes.length === 0 && (
                    <div className="text-center py-12 text-muted-foreground">Nenhuma turma criada</div>
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
                                            {cls.subject} · Prof. {cls.professor_username} · {cls.member_count} aluno(s) · {cls.book_count} livro(s)
                                        </p>
                                    </div>
                                </div>
                                <div className="flex items-center gap-2 shrink-0">
                                    <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => openRoster(cls.id)}>
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
                                <div className="mt-4 pt-4 border-t border-border space-y-3">
                                    <h5 className="font-medium text-secondary flex items-center gap-2">
                                        <Users className="h-4 w-4" /> Alunos
                                    </h5>
                                    {assignEnabled && (
                                        <div className="flex gap-2">
                                            <Input
                                                placeholder="Matrícula do aluno"
                                                value={registration}
                                                onChange={e => setRegistration(e.target.value)}
                                                onKeyDown={e => e.key === 'Enter' && assignStudent(cls.id)}
                                                className="rounded-xl h-9 max-w-xs"
                                            />
                                            <Button
                                                size="sm"
                                                className="rounded-xl bg-primary text-primary-foreground"
                                                onClick={() => assignStudent(cls.id)}
                                            >
                                                <UserPlus className="h-4 w-4" />
                                            </Button>
                                        </div>
                                    )}
                                    {studentsLoading ? (
                                        <Loader2 className="h-5 w-5 animate-spin text-primary" />
                                    ) : (
                                        <div className="space-y-1 max-h-64 overflow-y-auto">
                                            {students.length === 0 && (
                                                <p className="text-sm text-muted-foreground">Nenhum aluno matriculado</p>
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
                        <DialogTitle className="text-primary">Nova Turma</DialogTitle>
                        <DialogDescription>Crie uma turma e defina o professor responsável</DialogDescription>
                    </DialogHeader>
                    <div className="space-y-4 py-2">
                        <Input
                            placeholder="Nome da turma"
                            value={form.name}
                            onChange={e => setForm(prev => ({ ...prev, name: e.target.value }))}
                            className="rounded-xl"
                        />
                        <Input
                            placeholder="Disciplina"
                            value={form.subject}
                            onChange={e => setForm(prev => ({ ...prev, subject: e.target.value }))}
                            className="rounded-xl"
                        />
                        <Input
                            placeholder="Descrição (opcional)"
                            value={form.description}
                            onChange={e => setForm(prev => ({ ...prev, description: e.target.value }))}
                            className="rounded-xl"
                        />
                        <Select
                            value={form.professor_id}
                            onValueChange={value => setForm(prev => ({ ...prev, professor_id: value }))}
                        >
                            <SelectTrigger className="rounded-xl">
                                <SelectValue placeholder="Professor responsável" />
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
                            Cancelar
                        </Button>
                        <Button
                            className="rounded-xl bg-primary text-primary-foreground"
                            onClick={createClass}
                            disabled={creating || !form.name || !form.subject || !form.professor_id}
                        >
                            {creating ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Criar'}
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </div>
    );
};

export default ManagerClasses;
