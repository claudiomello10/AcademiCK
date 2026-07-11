'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Users, Clock, AlertCircle, BookOpen, MessageSquare, Loader2 } from 'lucide-react';
import {
    ResponsiveContainer, LineChart, Line, BarChart, Bar,
    XAxis, YAxis, CartesianGrid, Tooltip, Legend,
} from 'recharts';

import { API_ENDPOINTS } from '@/config/constants';
import { usePortal } from '@/components/portal/PortalShell';

// Validated against the dark surface (#121212): lightness band, chroma,
// CVD separation and contrast all pass — keep in sync if the theme changes.
const CHART_GREEN = '#66a328';
const CHART_BLUE = '#4b90cc';
const GRID_STROKE = 'rgba(255,255,255,0.08)';
const AXIS_INK = 'rgba(255,255,255,0.55)';

const TOOLTIP_STYLE = {
    backgroundColor: '#1c1c1c',
    border: '1px solid rgba(255,255,255,0.15)',
    borderRadius: '0.75rem',
    color: '#eee',
    fontSize: '0.8rem',
};

interface UsageSummary {
    total_queries: number;
    avg_response_time: number;
    active_users: number;
    content_access: number;
}

interface DailyTrend {
    date: string;
    retrieval_count: number;
    unique_queries: number;
}

const INTENT_LABELS: Record<string, string> = {
    question_answering: 'Q&A',
    summarization: 'Summarize',
    coding: 'Coding',
    searching_for_information: 'Search',
};

const AnalyticsSection = () => {
    const { portalFetch } = usePortal();
    const [timeRange, setTimeRange] = useState('7d');
    const [stats, setStats] = useState<UsageSummary | null>(null);
    const [trends, setTrends] = useState<DailyTrend[]>([]);
    const [intents, setIntents] = useState<{ intent: string; count: number }[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const fetchAll = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const [usageRes, chunksRes, intentsRes] = await Promise.all([
                portalFetch(API_ENDPOINTS.admin.usageStats(timeRange)),
                portalFetch(`/api/v1/admin/stats/chunks?range=${timeRange === 'all' ? '90d' : timeRange}`),
                portalFetch('/api/v1/admin/stats/usage'),
            ]);
            if (usageRes.ok) setStats(await usageRes.json());
            if (chunksRes.ok) {
                const data = await chunksRes.json();
                setTrends([...(data.daily_trends || [])].reverse().map((t: DailyTrend) => ({
                    ...t,
                    date: t.date ? t.date.slice(5) : '',
                })));
            }
            if (intentsRes.ok) {
                const data = await intentsRes.json();
                setIntents(Object.entries(data.queries_by_intent || {})
                    .map(([intent, count]) => ({
                        intent: INTENT_LABELS[intent] || intent,
                        count: count as number,
                    }))
                    .sort((a, b) => b.count - a.count));
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load analytics');
        } finally {
            setLoading(false);
        }
    }, [portalFetch, timeRange]);

    useEffect(() => { fetchAll(); }, [fetchAll]);

    const formatNumber = (num: number) =>
        num >= 1000 ? (num / 1000).toFixed(1) + 'k' : num.toString();

    if (loading && !stats) {
        return (
            <div className="flex items-center justify-center h-64">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
        );
    }

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-end gap-2">
                {(['7d', '30d', 'all'] as const).map(range => (
                    <Button
                        key={range}
                        variant={timeRange === range ? 'default' : 'secondary'}
                        className={`rounded-xl ${timeRange !== range ? 'text-black' : ''}`}
                        onClick={() => setTimeRange(range)}
                    >
                        {range === '7d' ? 'Last 7 Days' : range === '30d' ? 'Last 30 Days' : 'All Time'}
                    </Button>
                ))}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                {[
                    { icon: MessageSquare, label: 'Total Queries', value: formatNumber(stats?.total_queries || 0), sub: timeRange === 'all' ? 'All time' : `Last ${timeRange === '7d' ? 7 : 30} days` },
                    { icon: Clock, label: 'Avg Response Time', value: `${stats?.avg_response_time?.toFixed(2) || '0'}s`, sub: 'Average per query' },
                    { icon: Users, label: 'Active Users', value: `${stats?.active_users || 0}`, sub: 'Unique users in period' },
                    { icon: BookOpen, label: 'Content Access', value: formatNumber(stats?.content_access || 0), sub: 'Content retrievals' },
                ].map(({ icon: Icon, label, value, sub }) => (
                    <Card key={label} className="rounded-xl border border-primary/20">
                        <CardContent className="p-4">
                            <div className="flex items-center gap-3">
                                <Icon className="h-5 w-5 text-primary" />
                                <div className="text-secondary">
                                    <p className="text-sm font-medium">{label}</p>
                                    <p className="text-2xl font-bold">{value}</p>
                                    <p className="text-xs text-muted-foreground">{sub}</p>
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                ))}
            </div>

            {error && (
                <Alert variant="destructive" className="rounded-xl">
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                </Alert>
            )}

            <Card className="rounded-xl border border-primary/20">
                <CardHeader>
                    <CardTitle className="text-primary">Retrieval Activity</CardTitle>
                    <CardDescription>Daily content retrievals and unique queries</CardDescription>
                </CardHeader>
                <CardContent>
                    {trends.length === 0 ? (
                        <div className="text-center py-10 text-muted-foreground">No activity in this period</div>
                    ) : (
                        <div className="h-64">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={trends} margin={{ top: 8, right: 16, bottom: 0, left: -16 }}>
                                    <CartesianGrid stroke={GRID_STROKE} vertical={false} />
                                    <XAxis dataKey="date" stroke={AXIS_INK} tickLine={false} axisLine={false} fontSize={12} />
                                    <YAxis stroke={AXIS_INK} tickLine={false} axisLine={false} fontSize={12} allowDecimals={false} />
                                    <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ stroke: AXIS_INK, strokeDasharray: '3 3' }} />
                                    <Legend wrapperStyle={{ fontSize: '0.8rem' }} />
                                    <Line type="monotone" dataKey="retrieval_count" name="Retrievals" stroke={CHART_GREEN} strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
                                    <Line type="monotone" dataKey="unique_queries" name="Unique queries" stroke={CHART_BLUE} strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    )}
                </CardContent>
            </Card>

            <Card className="rounded-xl border border-primary/20">
                <CardHeader>
                    <CardTitle className="text-primary">Queries by Intent</CardTitle>
                    <CardDescription>All-time distribution of query intents</CardDescription>
                </CardHeader>
                <CardContent>
                    {intents.length === 0 ? (
                        <div className="text-center py-10 text-muted-foreground">No queries recorded</div>
                    ) : (
                        <div className="h-56">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={intents} layout="vertical" margin={{ top: 0, right: 24, bottom: 0, left: 8 }}>
                                    <CartesianGrid stroke={GRID_STROKE} horizontal={false} />
                                    <XAxis type="number" stroke={AXIS_INK} tickLine={false} axisLine={false} fontSize={12} allowDecimals={false} />
                                    <YAxis type="category" dataKey="intent" stroke={AXIS_INK} tickLine={false} axisLine={false} fontSize={12} width={90} />
                                    <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
                                    <Bar dataKey="count" name="Queries" fill={CHART_GREEN} radius={[0, 4, 4, 0]} barSize={18} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    )}
                </CardContent>
            </Card>
        </div>
    );
};

export default AnalyticsSection;
