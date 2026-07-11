'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import {
    AlertCircle, BarChart3, CornerDownRight, FileText, Loader2, Sparkles,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';

import { usePortal } from '@/components/portal/PortalShell';

// Validated against the dark surface (#121212) — see AnalyticsSection.
const BAR_GREEN = '#66a328';

export interface ClassAnalyticsEndpoints {
    classes: string;
    analyticsTopics: (classId: string, params?: string) => string;
    analyticsQueries: (classId: string, topicId: string, params?: string) => string;
    analyticsSummary: (classId: string) => string;
}

interface TopicCount {
    id: string;
    name: string;
    count: number;
    subtopics: { id: string; name: string; count: number }[];
}

interface QueryEntry {
    content: string;
    created_at: string;
    similarity?: number | null;
}

const ClassAnalytics = ({
    endpoints, summaryEnabled,
}: {
    endpoints: ClassAnalyticsEndpoints;
    summaryEnabled: boolean;
}) => {
    const { portalFetch } = usePortal();
    const [classes, setClasses] = useState<{ id: string; name: string }[]>([]);
    const [classId, setClassId] = useState('');
    const [dateFrom, setDateFrom] = useState('');
    const [dateTo, setDateTo] = useState('');
    const [ranking, setRanking] = useState<{
        topics: TopicCount[]; unclassified: number; total: number;
    } | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const [drilldown, setDrilldown] = useState<{ name: string; queries: QueryEntry[] } | null>(null);
    const [drilldownLoading, setDrilldownLoading] = useState(false);

    const [summary, setSummary] = useState<string | null>(null);
    const [summaryLoading, setSummaryLoading] = useState(false);

    useEffect(() => {
        (async () => {
            const response = await portalFetch(endpoints.classes);
            if (response.ok) {
                const list = (await response.json()).classes;
                setClasses(list);
                if (list.length > 0) setClassId(prev => prev || list[0].id);
            }
        })();
    }, [portalFetch, endpoints.classes]);

    const rangeParams = useCallback(() => {
        const params = new URLSearchParams();
        if (dateFrom) params.set('from', dateFrom);
        if (dateTo) params.set('to', dateTo);
        const qs = params.toString();
        return qs ? `?${qs}` : '';
    }, [dateFrom, dateTo]);

    const fetchRanking = useCallback(async () => {
        if (!classId) return;
        setLoading(true);
        setError(null);
        try {
            const response = await portalFetch(endpoints.analyticsTopics(classId, rangeParams()));
            const data = await response.json();
            if (!response.ok) {
                throw new Error(typeof data.detail === 'string' ? data.detail : 'Falha ao carregar análises');
            }
            setRanking(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Falha ao carregar análises');
        } finally {
            setLoading(false);
        }
    }, [portalFetch, endpoints, classId, rangeParams]);

    useEffect(() => { fetchRanking(); setSummary(null); }, [fetchRanking]);

    const openDrilldown = async (topicId: string, name: string) => {
        setDrilldown({ name, queries: [] });
        setDrilldownLoading(true);
        try {
            const response = await portalFetch(
                endpoints.analyticsQueries(classId, topicId, rangeParams())
            );
            if (response.ok) {
                const data = await response.json();
                setDrilldown({ name, queries: data.queries });
            }
        } finally {
            setDrilldownLoading(false);
        }
    };

    const generateSummary = async () => {
        setSummaryLoading(true);
        setError(null);
        try {
            const response = await portalFetch(endpoints.analyticsSummary(classId), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ date_from: dateFrom || null, date_to: dateTo || null }),
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(typeof data.detail === 'string' ? data.detail : 'Falha ao gerar resumo');
            }
            setSummary(data.summary);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Falha ao gerar resumo');
        } finally {
            setSummaryLoading(false);
        }
    };

    const maxCount = Math.max(
        1,
        ...(ranking?.topics.map(t => t.count) || []),
        ranking?.unclassified || 0,
    );

    const CountBar = ({ label, count, onClick, isSub = false }: {
        label: string; count: number; onClick: () => void; isSub?: boolean;
    }) => (
        <button
            onClick={onClick}
            className={`w-full text-left group ${isSub ? 'pl-8' : ''}`}
            title="Ver perguntas"
        >
            <div className="flex items-center justify-between text-sm mb-1">
                <span className="text-secondary flex items-center gap-1 truncate">
                    {isSub && <CornerDownRight className="h-3.5 w-3.5 text-muted-foreground shrink-0" />}
                    {label}
                </span>
                <span className="text-muted-foreground font-mono">{count}</span>
            </div>
            <div className="h-3 rounded bg-muted/60 overflow-hidden">
                <div
                    className="h-full rounded group-hover:opacity-80 transition-opacity"
                    style={{
                        width: `${(count / maxCount) * 100}%`,
                        backgroundColor: BAR_GREEN,
                        minWidth: count > 0 ? '4px' : 0,
                    }}
                />
            </div>
        </button>
    );

    return (
        <div className="space-y-6">
            <div className="flex items-end gap-3 flex-wrap">
                <div className="space-y-1">
                    <Label className="text-xs text-muted-foreground">Turma</Label>
                    <Select value={classId} onValueChange={setClassId}>
                        <SelectTrigger className="rounded-xl w-64">
                            <SelectValue placeholder="Selecione a turma" />
                        </SelectTrigger>
                        <SelectContent>
                            {classes.map(cls => (
                                <SelectItem key={cls.id} value={cls.id}>{cls.name}</SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                </div>
                <div className="space-y-1">
                    <Label className="text-xs text-muted-foreground">De</Label>
                    <Input
                        type="date"
                        value={dateFrom}
                        onChange={e => setDateFrom(e.target.value)}
                        className="rounded-xl w-40"
                    />
                </div>
                <div className="space-y-1">
                    <Label className="text-xs text-muted-foreground">Até</Label>
                    <Input
                        type="date"
                        value={dateTo}
                        onChange={e => setDateTo(e.target.value)}
                        className="rounded-xl w-40"
                    />
                </div>
                {summaryEnabled && classId && (
                    <Button
                        className="rounded-xl bg-primary text-primary-foreground hover:bg-primary/90 ml-auto"
                        onClick={generateSummary}
                        disabled={summaryLoading}
                    >
                        {summaryLoading ? (
                            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        ) : (
                            <Sparkles className="h-4 w-4 mr-2" />
                        )}
                        Gerar resumo
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

            {summary && (
                <Card className="rounded-xl border border-primary/40">
                    <CardHeader>
                        <CardTitle className="text-primary flex items-center gap-2">
                            <Sparkles className="h-5 w-5" />
                            Resumo do período
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="prose prose-sm prose-invert max-w-none text-secondary">
                        <ReactMarkdown>{summary}</ReactMarkdown>
                    </CardContent>
                </Card>
            )}

            {classId && (
                <Card className="rounded-xl border border-primary/20">
                    <CardHeader>
                        <CardTitle className="text-primary flex items-center gap-2">
                            <BarChart3 className="h-5 w-5" />
                            Dificuldades por tópico
                        </CardTitle>
                        <CardDescription>
                            {ranking ? `${ranking.total} pergunta(s) no período — clique para ver as perguntas` : ''}
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        {loading ? (
                            <div className="flex justify-center py-8">
                                <Loader2 className="h-6 w-6 animate-spin text-primary" />
                            </div>
                        ) : !ranking || (ranking.topics.length === 0 && ranking.unclassified === 0) ? (
                            <div className="text-center py-8 text-muted-foreground">
                                Sem dados no período. Defina tópicos na aba Tópicos e aguarde as
                                perguntas dos alunos.
                            </div>
                        ) : (
                            <div className="space-y-4">
                                {ranking.topics.map(topic => (
                                    <div key={topic.id} className="space-y-2">
                                        <CountBar
                                            label={topic.name}
                                            count={topic.count}
                                            onClick={() => openDrilldown(topic.id, topic.name)}
                                        />
                                        {topic.subtopics.map(sub => (
                                            <CountBar
                                                key={sub.id}
                                                label={sub.name}
                                                count={sub.count}
                                                onClick={() => openDrilldown(sub.id, sub.name)}
                                                isSub
                                            />
                                        ))}
                                    </div>
                                ))}
                                <CountBar
                                    label="Não classificado"
                                    count={ranking.unclassified}
                                    onClick={() => openDrilldown('unclassified', 'Não classificado')}
                                />
                            </div>
                        )}
                    </CardContent>
                </Card>
            )}

            <Dialog open={!!drilldown} onOpenChange={(open) => !open && setDrilldown(null)}>
                <DialogContent className="max-w-2xl bg-muted border-primary/20">
                    <DialogHeader>
                        <DialogTitle className="text-primary flex items-center gap-2">
                            <FileText className="h-5 w-5" />
                            {drilldown?.name}
                        </DialogTitle>
                        <DialogDescription>
                            Perguntas dos alunos neste tópico (mais recentes primeiro)
                        </DialogDescription>
                    </DialogHeader>
                    <div className="max-h-96 overflow-y-auto space-y-2 py-2">
                        {drilldownLoading ? (
                            <div className="flex justify-center py-6">
                                <Loader2 className="h-5 w-5 animate-spin text-primary" />
                            </div>
                        ) : drilldown?.queries.length === 0 ? (
                            <p className="text-sm text-muted-foreground text-center py-4">
                                Nenhuma pergunta neste tópico no período
                            </p>
                        ) : (
                            drilldown?.queries.map((query, i) => (
                                <div key={i} className="p-3 rounded-lg bg-background/50">
                                    <p className="text-sm text-secondary">{query.content}</p>
                                    <p className="text-xs text-muted-foreground mt-1">
                                        {new Date(query.created_at).toLocaleString('pt-BR')}
                                    </p>
                                </div>
                            ))
                        )}
                    </div>
                </DialogContent>
            </Dialog>
        </div>
    );
};

export default ClassAnalytics;
