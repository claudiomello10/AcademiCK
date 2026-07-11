'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Button } from '@/components/ui/button';
import {
    AlertCircle, Loader2, CheckCircle2, X, AlertTriangle, StopCircle, RefreshCw
} from 'lucide-react';

import { API_ENDPOINTS } from '@/config/constants';
import { usePortal } from '@/components/portal/PortalShell';

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

const JobsManagement = () => {
    const { portalFetch } = usePortal();
    const [jobs, setJobs] = useState<ProcessingJob[]>([]);
    const [loading, setLoading] = useState(true);
    const pollingRef = useRef<NodeJS.Timeout | null>(null);

    const fetchJobs = useCallback(async () => {
        try {
            const response = await portalFetch(API_ENDPOINTS.admin.jobs);
            if (!response.ok) return;
            const data = await response.json();
            const parsed: ProcessingJob[] = data.map((job: ProcessingJob) => ({
                ...job,
                progress: job.status === 'completed' ? 100 : (job.progress || 0),
            }));
            setJobs(parsed);

            const hasActive = parsed.some(
                j => j.status !== 'completed' && j.status !== 'failed' && j.status !== 'cancelled'
            );
            if (hasActive && !pollingRef.current) {
                pollingRef.current = setInterval(fetchJobs, 3000);
            } else if (!hasActive && pollingRef.current) {
                clearInterval(pollingRef.current);
                pollingRef.current = null;
            }
        } finally {
            setLoading(false);
        }
    }, [portalFetch]);

    useEffect(() => {
        fetchJobs();
        return () => {
            if (pollingRef.current) clearInterval(pollingRef.current);
        };
    }, [fetchJobs]);

    const dismissJob = async (jobId: string) => {
        const response = await portalFetch(API_ENDPOINTS.admin.dismissJob(jobId), { method: 'DELETE' });
        if (response.ok) setJobs(prev => prev.filter(j => j.job_id !== jobId));
    };

    const cancelJob = async (jobId: string) => {
        const response = await portalFetch(API_ENDPOINTS.admin.cancelJob(jobId), { method: 'POST' });
        if (response.ok) {
            setJobs(prev => prev.map(j =>
                j.job_id === jobId ? { ...j, status: 'cancelled', stage: 'cancelled' } : j
            ));
        }
    };

    const statusText = (job: ProcessingJob) =>
        job.status === 'completed' ? 'Complete' :
        job.status === 'failed' ? 'Failed' :
        job.status === 'cancelled' ? 'Cancelled' :
        job.status === 'pending' || job.stage === 'pending' ? 'Waiting for available worker...' :
        job.chapters_total && job.chapters_total > 0 &&
        (job.chapters_processed || 0) < job.chapters_total &&
        !job.stage?.includes('embedding') &&
        !job.stage?.includes('storing')
            ? `Chapter ${Math.min((job.chapters_processed || 0) + 1, job.chapters_total)}/${job.chapters_total}`
            : job.stage ? job.stage.replace(/_/g, ' ') : 'Processing...';

    return (
        <Card className="rounded-xl border border-primary/20">
            <CardHeader className="flex flex-row items-center justify-between space-y-0">
                <div>
                    <CardTitle className="text-primary">Processing Jobs</CardTitle>
                    <CardDescription>PDF ingestion across all portals</CardDescription>
                </div>
                <Button
                    variant="outline"
                    size="sm"
                    className="rounded-xl"
                    onClick={() => { setLoading(true); fetchJobs(); }}
                >
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Refresh
                </Button>
            </CardHeader>
            <CardContent>
                {loading ? (
                    <div className="flex items-center justify-center py-12">
                        <Loader2 className="h-6 w-6 animate-spin text-primary" />
                    </div>
                ) : jobs.length === 0 ? (
                    <div className="text-center py-12 text-muted-foreground">
                        No processing jobs
                    </div>
                ) : (
                    <div className="space-y-4 max-h-[600px] overflow-y-auto pr-2">
                        {jobs.map((job) => (
                            <div key={job.job_id} className="space-y-2">
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
                                        <span className="text-sm text-muted-foreground">{statusText(job)}</span>
                                        {(job.status === 'pending' || job.status === 'processing') ? (
                                            <Button
                                                variant="ghost"
                                                size="icon"
                                                className="h-6 w-6 text-destructive hover:bg-destructive hover:text-white"
                                                onClick={() => cancelJob(job.job_id)}
                                                title="Cancel job"
                                            >
                                                <StopCircle className="h-4 w-4" />
                                            </Button>
                                        ) : (
                                            <Button
                                                variant="ghost"
                                                size="icon"
                                                className="h-6 w-6 text-muted-foreground hover:text-destructive"
                                                onClick={() => dismissJob(job.job_id)}
                                                title="Dismiss job"
                                            >
                                                <X className="h-4 w-4" />
                                            </Button>
                                        )}
                                    </div>
                                </div>
                                <Progress
                                    value={job.status === 'failed' || job.status === 'cancelled' ? 100 : job.progress}
                                    className="h-2"
                                    indicatorClassName={
                                        job.status === 'failed'
                                            ? 'bg-red-500'
                                            : job.status === 'cancelled'
                                                ? 'bg-orange-400'
                                                : undefined
                                    }
                                />
                                {job.warning && (
                                    <div className="flex items-center gap-2 p-2 rounded bg-yellow-50 border border-yellow-200">
                                        <AlertTriangle className="h-4 w-4 text-yellow-600 flex-shrink-0" />
                                        <span className="text-sm text-yellow-700">{job.warning}</span>
                                    </div>
                                )}
                                {job.error && (
                                    <div className="flex items-start gap-2 p-2 rounded bg-red-50 border border-red-200">
                                        <AlertCircle className="h-4 w-4 text-red-600 flex-shrink-0 mt-0.5" />
                                        <span className="text-sm text-red-700 break-words">{job.error}</span>
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                )}
            </CardContent>
        </Card>
    );
};

export default JobsManagement;
