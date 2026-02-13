import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Users, Clock, AlertCircle, BookOpen, MessageSquare, Loader2 } from 'lucide-react';

import { API_BASE_URL, API_ENDPOINTS } from '@/config/constants';

interface UsageStatsData {
    total_queries: number;
    avg_response_time: number;
    active_users: number;
    content_access: number;
    time_range: string;
}

const UsageManagement = () => {
    const [timeRange, setTimeRange] = useState('7d');
    const [stats, setStats] = useState<UsageStatsData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [sessionId, setSessionId] = useState<string | null>(null);

    useEffect(() => {
        // Get session ID from localStorage
        const adminSession = localStorage.getItem('adminSession');
        if (adminSession) {
            setSessionId(adminSession);
        }
    }, []);

    useEffect(() => {
        if (sessionId) {
            fetchUsageStats();
        }
    }, [timeRange, sessionId]);

    const fetchUsageStats = async () => {
        if (!sessionId) return;

        setLoading(true);
        try {
            const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.admin.usageStats(timeRange, sessionId)}`);
            if (response.ok) {
                const data = await response.json();
                setStats(data);
            } else {
                throw new Error('Failed to fetch usage statistics');
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    // Format number for display
    const formatNumber = (num: number) => {
        if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'k';
        }
        return num.toString();
    };

    if (loading && !stats) {
        return (
            <div className="flex items-center justify-center h-64">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
        );
    }

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <h3 className="text-lg font-medium text-primary">Usage Analytics</h3>
                <div className="flex gap-2">
                    <Button
                        variant={timeRange === '7d' ? 'default' : 'secondary'}
                        className={`rounded-xl ${timeRange !== '7d' ? 'text-black' : ''}`}
                        onClick={() => setTimeRange('7d')}
                    >
                        Last 7 Days
                    </Button>
                    <Button
                        variant={timeRange === '30d' ? 'default' : 'secondary'}
                        className={`rounded-xl ${timeRange !== '30d' ? 'text-black' : ''}`}
                        onClick={() => setTimeRange('30d')}
                    >
                        Last 30 Days
                    </Button>
                    <Button
                        variant={timeRange === 'all' ? 'default' : 'secondary'}
                        className={`rounded-xl ${timeRange !== 'all' ? 'text-black' : ''}`}
                        onClick={() => setTimeRange('all')}
                    >
                        All Time
                    </Button>
                </div>
            </div>

            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <Card className="rounded-xl border border-primary/20">
                    <CardContent className="p-4">
                        <div className="flex items-center gap-3">
                            <MessageSquare className="h-5 w-5 text-primary" />
                            <div className="text-secondary">
                                <p className="text-sm font-medium">Total Queries</p>
                                <p className="text-2xl font-bold">{formatNumber(stats?.total_queries || 0)}</p>
                                <p className="text-xs text-muted-foreground">{timeRange === '7d' ? 'Last 7 days' : timeRange === '30d' ? 'Last 30 days' : 'All time'}</p>
                            </div>
                        </div>
                    </CardContent>
                </Card>

                <Card className="rounded-xl border border-primary/20">
                    <CardContent className="p-4">
                        <div className="flex items-center gap-3">
                            <Clock className="h-5 w-5 text-primary" />
                            <div className="text-secondary">
                                <p className="text-sm font-medium">Avg Response Time</p>
                                <p className="text-2xl font-bold">{stats?.avg_response_time?.toFixed(2) || '0'}s</p>
                                <p className="text-xs text-muted-foreground">Average per query</p>
                            </div>
                        </div>
                    </CardContent>
                </Card>

                <Card className="rounded-xl border border-primary/20">
                    <CardContent className="p-4">
                        <div className="flex items-center gap-3">
                            <Users className="h-5 w-5 text-primary" />
                            <div className="text-secondary">
                                <p className="text-sm font-medium">Active Users</p>
                                <p className="text-2xl font-bold">{stats?.active_users || 0}</p>
                                <p className="text-xs text-muted-foreground">Unique users in period</p>
                            </div>
                        </div>
                    </CardContent>
                </Card>

                <Card className="rounded-xl border border-primary/20">
                    <CardContent className="p-4">
                        <div className="flex items-center gap-3">
                            <BookOpen className="h-5 w-5 text-primary" />
                            <div className="text-secondary">
                                <p className="text-sm font-medium">Content Access</p>
                                <p className="text-2xl font-bold">{formatNumber(stats?.content_access || 0)}</p>
                                <p className="text-xs text-muted-foreground">Content retrievals</p>
                            </div>
                        </div>
                    </CardContent>
                </Card>
            </div>

            {/* Usage Summary */}
            <Card className="rounded-xl border border-primary/20">
                <CardHeader>
                    <CardTitle className="text-primary">Usage Summary</CardTitle>
                    <CardDescription>Overview of system usage for the selected time period</CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="text-center py-8 text-muted-foreground">
                        <MessageSquare className="h-12 w-12 mx-auto mb-4 opacity-50" />
                        <p className="text-lg font-medium">Summary Statistics</p>
                        <p className="text-sm mt-2">
                            {stats?.total_queries || 0} total queries from {stats?.active_users || 0} users
                        </p>
                        <p className="text-sm">
                            Average response time: {stats?.avg_response_time?.toFixed(2) || '0'}s
                        </p>
                    </div>
                </CardContent>
            </Card>

            {/* System Status */}
            <Card className="rounded-xl">
                <CardHeader>
                    <CardTitle className="text-primary">System Status</CardTitle>
                    <CardDescription>Current system health</CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="flex items-center justify-center py-4">
                        <div className="flex items-center gap-2">
                            <div className="h-3 w-3 rounded-full bg-green-500 animate-pulse" />
                            <span className="text-sm font-medium text-secondary">All systems operational</span>
                        </div>
                    </div>
                </CardContent>
            </Card>

            {/* Error Display */}
            {error && (
                <Alert variant="destructive" className="rounded-xl">
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                </Alert>
            )}
        </div>
    );
};

export default UsageManagement;