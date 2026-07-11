'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from '@/components/ui/alert-dialog';
import {
    AlertCircle, CornerDownRight, Edit, ListTree, Loader2, Plus, Trash2,
} from 'lucide-react';

import { API_ENDPOINTS } from '@/config/constants';
import { usePortal } from '@/components/portal/PortalShell';

interface Topic {
    id: string;
    name: string;
    description?: string | null;
    parent_topic_id?: string | null;
    subtopics?: Topic[];
}

interface ClassInfo { id: string; name: string }

const TopicsEditor = () => {
    const { portalFetch } = usePortal();
    const [classes, setClasses] = useState<ClassInfo[]>([]);
    const [classId, setClassId] = useState('');
    const [topics, setTopics] = useState<Topic[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const [dialogOpen, setDialogOpen] = useState(false);
    const [editing, setEditing] = useState<Topic | null>(null);
    const [parentId, setParentId] = useState<string | null>(null);
    const [form, setForm] = useState({ name: '', description: '' });
    const [saving, setSaving] = useState(false);
    const [topicToDelete, setTopicToDelete] = useState<Topic | null>(null);

    useEffect(() => {
        (async () => {
            const response = await portalFetch(API_ENDPOINTS.professor.classes);
            if (response.ok) {
                const list = (await response.json()).classes;
                setClasses(list);
                if (list.length > 0) setClassId(prev => prev || list[0].id);
            }
        })();
    }, [portalFetch]);

    const fetchTopics = useCallback(async () => {
        if (!classId) return;
        setLoading(true);
        try {
            const response = await portalFetch(API_ENDPOINTS.professor.topics(classId));
            if (response.ok) setTopics((await response.json()).topics);
        } finally {
            setLoading(false);
        }
    }, [portalFetch, classId]);

    useEffect(() => { fetchTopics(); }, [fetchTopics]);

    const openCreate = (parent: string | null) => {
        setEditing(null);
        setParentId(parent);
        setForm({ name: '', description: '' });
        setDialogOpen(true);
    };

    const openEdit = (topic: Topic) => {
        setEditing(topic);
        setParentId(topic.parent_topic_id || null);
        setForm({ name: topic.name, description: topic.description || '' });
        setDialogOpen(true);
    };

    const save = async () => {
        setSaving(true);
        setError(null);
        try {
            let response: Response;
            if (editing) {
                response = await portalFetch(API_ENDPOINTS.professor.topicById(editing.id), {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name: form.name, description: form.description || null }),
                });
            } else {
                response = await portalFetch(API_ENDPOINTS.professor.topics(classId), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name: form.name,
                        description: form.description || null,
                        parent_topic_id: parentId,
                    }),
                });
            }
            const data = await response.json();
            if (!response.ok) {
                throw new Error(typeof data.detail === 'string' ? data.detail : 'Falha ao salvar tópico');
            }
            setDialogOpen(false);
            fetchTopics();
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Falha ao salvar tópico');
        } finally {
            setSaving(false);
        }
    };

    const remove = async (topic: Topic) => {
        const response = await portalFetch(API_ENDPOINTS.professor.topicById(topic.id), {
            method: 'DELETE',
        });
        if (response.ok) fetchTopics();
        setTopicToDelete(null);
    };

    const TopicRow = ({ topic, isSub }: { topic: Topic; isSub: boolean }) => (
        <div className={`flex items-center justify-between p-3 rounded-lg bg-muted/50 ${isSub ? 'ml-8' : ''}`}>
            <div className="flex items-center gap-2 min-w-0">
                {isSub && <CornerDownRight className="h-4 w-4 text-muted-foreground shrink-0" />}
                <div className="min-w-0">
                    <p className="font-medium text-secondary truncate">{topic.name}</p>
                    {topic.description && (
                        <p className="text-xs text-muted-foreground truncate">{topic.description}</p>
                    )}
                </div>
            </div>
            <div className="flex gap-1 shrink-0">
                {!isSub && (
                    <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7"
                        onClick={() => openCreate(topic.id)}
                        title="Adicionar subtópico"
                    >
                        <Plus className="h-4 w-4" />
                    </Button>
                )}
                <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7"
                    onClick={() => openEdit(topic)}
                >
                    <Edit className="h-4 w-4" />
                </Button>
                <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7 text-muted-foreground hover:text-destructive"
                    onClick={() => setTopicToDelete(topic)}
                >
                    <Trash2 className="h-4 w-4" />
                </Button>
            </div>
        </div>
    );

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
                    <Button
                        className="rounded-xl bg-primary text-primary-foreground hover:bg-primary/90 ml-auto"
                        onClick={() => openCreate(null)}
                    >
                        <Plus className="h-4 w-4 mr-2" />
                        Novo Tópico
                    </Button>
                )}
            </div>

            {error && (
                <Alert variant="destructive" className="rounded-xl">
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>Erro</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                </Alert>
            )}

            {classId && (
                <Card className="rounded-xl border border-primary/20">
                    <CardHeader>
                        <CardTitle className="text-primary flex items-center gap-2">
                            <ListTree className="h-5 w-5" />
                            Tópicos da disciplina
                        </CardTitle>
                        <CardDescription>
                            Cada pergunta dos alunos é comparada a estes tópicos. Descrições
                            claras melhoram a classificação.
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        {loading ? (
                            <div className="flex justify-center py-8">
                                <Loader2 className="h-6 w-6 animate-spin text-primary" />
                            </div>
                        ) : topics.length === 0 ? (
                            <div className="text-center py-8 text-muted-foreground">
                                Nenhum tópico definido. Crie a estrutura da sua disciplina!
                            </div>
                        ) : (
                            <div className="space-y-2">
                                {topics.map(topic => (
                                    <div key={topic.id} className="space-y-2">
                                        <TopicRow topic={topic} isSub={false} />
                                        {(topic.subtopics || []).map(sub => (
                                            <TopicRow key={sub.id} topic={sub} isSub />
                                        ))}
                                    </div>
                                ))}
                            </div>
                        )}
                    </CardContent>
                </Card>
            )}

            <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
                <DialogContent className="bg-muted border-primary/20">
                    <DialogHeader>
                        <DialogTitle className="text-primary">
                            {editing ? 'Editar tópico' : parentId ? 'Novo subtópico' : 'Novo tópico'}
                        </DialogTitle>
                        <DialogDescription>
                            A descrição é usada para associar as perguntas dos alunos ao tópico.
                        </DialogDescription>
                    </DialogHeader>
                    <div className="space-y-4 py-2">
                        <div className="space-y-1">
                            <Label>Nome</Label>
                            <Input
                                value={form.name}
                                onChange={e => setForm(prev => ({ ...prev, name: e.target.value }))}
                                className="rounded-xl"
                                placeholder="ex.: Derivadas"
                            />
                        </div>
                        <div className="space-y-1">
                            <Label>Descrição</Label>
                            <Input
                                value={form.description}
                                onChange={e => setForm(prev => ({ ...prev, description: e.target.value }))}
                                className="rounded-xl"
                                placeholder="ex.: Regras de derivação, regra da cadeia, derivadas implícitas"
                            />
                        </div>
                    </div>
                    <DialogFooter>
                        <Button variant="outline" className="rounded-xl" onClick={() => setDialogOpen(false)}>
                            Cancelar
                        </Button>
                        <Button
                            className="rounded-xl bg-primary text-primary-foreground"
                            onClick={save}
                            disabled={saving || !form.name}
                        >
                            {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Salvar'}
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>

            <AlertDialog open={!!topicToDelete} onOpenChange={(open: boolean) => !open && setTopicToDelete(null)}>
                <AlertDialogContent className="bg-muted border-primary/20">
                    <AlertDialogHeader>
                        <AlertDialogTitle className="text-primary">Excluir tópico</AlertDialogTitle>
                        <AlertDialogDescription className="text-muted-foreground">
                            Excluir &quot;{topicToDelete?.name}&quot;
                            {!topicToDelete?.parent_topic_id && ' e seus subtópicos'}? As perguntas
                            já classificadas neste tópico passam a contar como não classificadas.
                        </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                        <AlertDialogCancel className="border-muted-foreground/30 text-secondary hover:bg-muted-foreground/10 hover:text-secondary">Cancelar</AlertDialogCancel>
                        <AlertDialogAction
                            className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                            onClick={() => topicToDelete && remove(topicToDelete)}
                        >
                            Excluir
                        </AlertDialogAction>
                    </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialog>
        </div>
    );
};

export default TopicsEditor;
