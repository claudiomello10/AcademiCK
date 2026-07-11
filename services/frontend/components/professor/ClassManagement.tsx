'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from '@/components/ui/alert-dialog';
import {
    AlertCircle, Check, ChevronDown, ChevronUp, Copy, GraduationCap,
    Loader2, Plus, RefreshCw, Trash2, UserPlus, Users,
} from 'lucide-react';

import { API_ENDPOINTS } from '@/config/constants';
import { usePortal } from '@/components/portal/PortalShell';

interface ClassInfo {
    id: string;
    name: string;
    subject: string;
    description?: string;
    join_code?: string;
    join_code_enabled: boolean;
    is_active: boolean;
}

interface Student {
    id: string;
    username: string;
    registration_number?: string;
    enrolled_via: string;
}

interface ClassManagementProps {
    features: { registration_enrollment_enabled: boolean; join_code_enabled: boolean };
}

const ClassManagement = ({ features }: ClassManagementProps) => {
    const { portalFetch } = usePortal();
    const [classes, setClasses] = useState<ClassInfo[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState('');
    const [expanded, setExpanded] = useState<string | null>(null);
    const [students, setStudents] = useState<Student[]>([]);
    const [studentsLoading, setStudentsLoading] = useState(false);
    const [copied, setCopied] = useState<string | null>(null);

    const [showCreate, setShowCreate] = useState(false);
    const [form, setForm] = useState({ name: '', subject: '', description: '' });
    const [creating, setCreating] = useState(false);
    const [registration, setRegistration] = useState('');
    const [classToDelete, setClassToDelete] = useState<ClassInfo | null>(null);

    const fetchClasses = useCallback(async () => {
        const response = await portalFetch(API_ENDPOINTS.professor.classes);
        if (response.ok) setClasses((await response.json()).classes);
        setLoading(false);
    }, [portalFetch]);

    useEffect(() => { fetchClasses(); }, [fetchClasses]);

    const openRoster = async (classId: string) => {
        if (expanded === classId) {
            setExpanded(null);
            return;
        }
        setExpanded(classId);
        setStudentsLoading(true);
        try {
            const response = await portalFetch(API_ENDPOINTS.professor.students(classId));
            if (response.ok) setStudents((await response.json()).students);
        } finally {
            setStudentsLoading(false);
        }
    };

    const createClass = async () => {
        setCreating(true);
        setError(null);
        try {
            const response = await portalFetch(API_ENDPOINTS.professor.classes, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(form),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Falha ao criar turma');
            setSuccess(`Turma "${data.name}" criada`);
            setShowCreate(false);
            setForm({ name: '', subject: '', description: '' });
            fetchClasses();
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Falha ao criar turma');
        } finally {
            setCreating(false);
        }
    };

    const deleteClass = async (cls: ClassInfo) => {
        const response = await portalFetch(API_ENDPOINTS.professor.classById(cls.id), { method: 'DELETE' });
        if (response.ok) {
            setSuccess(`Turma "${cls.name}" excluída`);
            setExpanded(null);
            fetchClasses();
        }
        setClassToDelete(null);
    };

    const copyJoinCode = (cls: ClassInfo) => {
        if (!cls.join_code) return;
        navigator.clipboard.writeText(cls.join_code);
        setCopied(cls.id);
        setTimeout(() => setCopied(null), 1500);
    };

    const regenerateCode = async (classId: string) => {
        const response = await portalFetch(API_ENDPOINTS.professor.regenerateJoinCode(classId), { method: 'POST' });
        if (response.ok) {
            const data = await response.json();
            setClasses(prev => prev.map(c => c.id === classId ? { ...c, join_code: data.join_code } : c));
        }
    };

    const toggleCode = async (cls: ClassInfo) => {
        const response = await portalFetch(API_ENDPOINTS.professor.toggleJoinCode(cls.id), {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled: !cls.join_code_enabled }),
        });
        if (response.ok) {
            setClasses(prev => prev.map(c =>
                c.id === cls.id ? { ...c, join_code_enabled: !cls.join_code_enabled } : c
            ));
        }
    };

    const addStudent = async (classId: string) => {
        setError(null);
        const value = registration.trim();
        if (!value) return;
        const response = await portalFetch(API_ENDPOINTS.professor.students(classId), {
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
        setStudents(prev => [...prev, { ...data, enrolled_via: 'professor' }]);
    };

    const removeStudent = async (classId: string, userId: string) => {
        const response = await portalFetch(API_ENDPOINTS.professor.removeStudent(classId, userId), {
            method: 'DELETE',
        });
        if (response.ok) setStudents(prev => prev.filter(s => s.id !== userId));
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
                    Suas turmas, códigos de acesso e alunos matriculados.
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

            {classes.length === 0 && (
                <div className="text-center py-12 text-muted-foreground">
                    Você ainda não tem turmas. Crie a primeira!
                </div>
            )}

            <div className="space-y-2">
                {classes.map(cls => (
                    <Card key={cls.id} className="rounded-xl border border-primary/20">
                        <CardContent className="p-4">
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-3 min-w-0">
                                    <GraduationCap className="h-5 w-5 text-primary shrink-0" />
                                    <div className="min-w-0">
                                        <h4 className="font-medium text-secondary truncate">{cls.name}</h4>
                                        <p className="text-sm text-muted-foreground truncate">{cls.subject}</p>
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
                                        onClick={() => setClassToDelete(cls)}
                                    >
                                        <Trash2 className="h-4 w-4" />
                                    </Button>
                                </div>
                            </div>

                            {expanded === cls.id && (
                                <div className="mt-4 pt-4 border-t border-border space-y-4">
                                    {features.join_code_enabled && (
                                        <div className="flex items-center justify-between flex-wrap gap-2 bg-muted/50 rounded-xl p-3">
                                            <div>
                                                <p className="text-sm font-medium text-secondary">Código da turma</p>
                                                <p className="font-mono text-lg text-primary tracking-widest">
                                                    {cls.join_code_enabled ? cls.join_code : '—— desativado ——'}
                                                </p>
                                            </div>
                                            <div className="flex gap-2">
                                                <Button
                                                    variant="outline"
                                                    size="sm"
                                                    className="rounded-xl"
                                                    onClick={() => copyJoinCode(cls)}
                                                    disabled={!cls.join_code_enabled}
                                                >
                                                    {copied === cls.id ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                                                </Button>
                                                <Button
                                                    variant="outline"
                                                    size="sm"
                                                    className="rounded-xl"
                                                    onClick={() => regenerateCode(cls.id)}
                                                    disabled={!cls.join_code_enabled}
                                                    title="Gerar novo código"
                                                >
                                                    <RefreshCw className="h-4 w-4" />
                                                </Button>
                                                <Button
                                                    variant="outline"
                                                    size="sm"
                                                    className="rounded-xl"
                                                    onClick={() => toggleCode(cls)}
                                                >
                                                    {cls.join_code_enabled ? 'Desativar' : 'Ativar'}
                                                </Button>
                                            </div>
                                        </div>
                                    )}

                                    <div className="space-y-3">
                                        <h5 className="font-medium text-secondary flex items-center gap-2">
                                            <Users className="h-4 w-4" /> Alunos
                                        </h5>
                                        {features.registration_enrollment_enabled && (
                                            <div className="flex gap-2">
                                                <Input
                                                    placeholder="Matrícula do aluno"
                                                    value={registration}
                                                    onChange={e => setRegistration(e.target.value)}
                                                    onKeyDown={e => e.key === 'Enter' && addStudent(cls.id)}
                                                    className="rounded-xl h-9"
                                                />
                                                <Button
                                                    size="sm"
                                                    className="rounded-xl bg-primary text-primary-foreground"
                                                    onClick={() => addStudent(cls.id)}
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
                        <DialogDescription>Crie uma turma para seus alunos</DialogDescription>
                    </DialogHeader>
                    <div className="space-y-4 py-2">
                        <Input
                            placeholder="Nome da turma (ex.: Cálculo I — 2026.1)"
                            value={form.name}
                            onChange={e => setForm(prev => ({ ...prev, name: e.target.value }))}
                            className="rounded-xl"
                        />
                        <Input
                            placeholder="Disciplina (ex.: Cálculo I)"
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
                    </div>
                    <DialogFooter>
                        <Button variant="outline" className="rounded-xl" onClick={() => setShowCreate(false)}>
                            Cancelar
                        </Button>
                        <Button
                            className="rounded-xl bg-primary text-primary-foreground"
                            onClick={createClass}
                            disabled={creating || !form.name || !form.subject}
                        >
                            {creating ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Criar'}
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>

            <AlertDialog open={!!classToDelete} onOpenChange={(open: boolean) => !open && setClassToDelete(null)}>
                <AlertDialogContent className="bg-muted border-primary/20">
                    <AlertDialogHeader>
                        <AlertDialogTitle className="text-primary">Excluir turma</AlertDialogTitle>
                        <AlertDialogDescription className="text-muted-foreground">
                            Excluir &quot;{classToDelete?.name}&quot;? As matrículas, tópicos e
                            estatísticas da turma serão removidos. Esta ação não pode ser desfeita.
                        </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                        <AlertDialogCancel className="border-muted-foreground/30 text-secondary hover:bg-muted-foreground/10 hover:text-secondary">Cancelar</AlertDialogCancel>
                        <AlertDialogAction
                            className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                            onClick={() => classToDelete && deleteClass(classToDelete)}
                        >
                            Excluir
                        </AlertDialogAction>
                    </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialog>
        </div>
    );
};

export default ClassManagement;
