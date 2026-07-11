'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from '@/components/ui/alert-dialog';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
    Database, Upload, Download, Trash2, AlertCircle, Loader2, FileJson, RefreshCw
} from 'lucide-react';
import { cn } from '@/lib/utils';

import { API_ENDPOINTS } from '@/config/constants';
import { usePortal } from '@/components/portal/PortalShell';

interface ServiceStatus {
    name: string;
    status: 'healthy' | 'degraded' | 'down';
    latency_ms: number;
    detail: string;
}

type SnapshotOp = 'restore' | 'delete' | 'download' | 'metadata';

export const ServiceStatusPanel = ({ compact = false }: { compact?: boolean }) => {
    const { portalFetch } = usePortal();
    const [services, setServices] = useState<ServiceStatus[]>([]);
    const [overall, setOverall] = useState('');
    const [loading, setLoading] = useState(true);

    const fetchStatus = useCallback(async () => {
        try {
            const response = await portalFetch(API_ENDPOINTS.admin.systemStatus);
            if (response.ok) {
                const data = await response.json();
                setServices(data.services);
                setOverall(data.overall);
            }
        } finally {
            setLoading(false);
        }
    }, [portalFetch]);

    useEffect(() => {
        fetchStatus();
        const interval = setInterval(fetchStatus, 15000);
        return () => clearInterval(interval);
    }, [fetchStatus]);

    if (loading) {
        return (
            <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin text-primary" />
            </div>
        );
    }

    return (
        <div className="space-y-3">
            {!compact && (
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <span className={cn(
                            'inline-block h-3 w-3 rounded-full',
                            overall === 'healthy' ? 'bg-green-500' : 'bg-yellow-500'
                        )} />
                        <span className="font-medium text-secondary capitalize">{overall}</span>
                    </div>
                    <Button variant="outline" size="sm" className="rounded-xl" onClick={fetchStatus}>
                        <RefreshCw className="h-4 w-4 mr-2" />
                        Refresh
                    </Button>
                </div>
            )}
            <div className={cn(
                'grid gap-3',
                compact ? 'grid-cols-2 md:grid-cols-6' : 'grid-cols-1 md:grid-cols-3'
            )}>
                {services.map(service => (
                    <Card key={service.name} className="rounded-xl border border-primary/20">
                        <CardContent className={compact ? 'p-3' : 'p-4'}>
                            <div className="flex items-center gap-2">
                                <span className={cn(
                                    'inline-block h-2.5 w-2.5 rounded-full shrink-0',
                                    service.status === 'healthy' ? 'bg-green-500' :
                                    service.status === 'degraded' ? 'bg-yellow-500' : 'bg-red-500'
                                )} />
                                <span className="font-medium text-secondary text-sm truncate">{service.name}</span>
                            </div>
                            <p className="text-xs text-muted-foreground mt-1">
                                {service.status === 'healthy'
                                    ? `${service.latency_ms} ms${service.detail ? ` · ${service.detail}` : ''}`
                                    : service.detail || service.status}
                            </p>
                        </CardContent>
                    </Card>
                ))}
            </div>
        </div>
    );
};

const SystemManagement = () => {
    const { portalFetch } = usePortal();
    const [snapshots, setSnapshots] = useState<any[]>([]);
    const [snapshotLoading, setSnapshotLoading] = useState(false);
    const [snapshotEnabled, setSnapshotEnabled] = useState(true);
    const [snapshotToRestore, setSnapshotToRestore] = useState<string | null>(null);
    const [snapshotToDelete, setSnapshotToDelete] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState('');
    const [busy, setBusy] = useState<{ name: string; op: SnapshotOp } | null>(null);
    const [creating, setCreating] = useState(false);
    const [showUploadDialog, setShowUploadDialog] = useState(false);
    const [uploadSnapshotFile, setUploadSnapshotFile] = useState<File | null>(null);
    const [uploadMetadataFile, setUploadMetadataFile] = useState<File | null>(null);

    const isBusy = (name: string, op: SnapshotOp) => busy?.name === name && busy?.op === op;

    const fetchSnapshots = useCallback(async () => {
        setSnapshotLoading(true);
        try {
            const response = await portalFetch(API_ENDPOINTS.admin.snapshots);
            if (response.ok) {
                setSnapshots((await response.json()).snapshots);
                setError(null);
            } else {
                const err = await response.json().catch(() => ({}));
                throw new Error(err.detail || `Failed to load snapshots (HTTP ${response.status})`);
            }
        } catch (err: any) {
            setError(err.message || 'Failed to load snapshots');
        } finally {
            setSnapshotLoading(false);
        }
    }, [portalFetch]);

    useEffect(() => {
        (async () => {
            const response = await portalFetch(API_ENDPOINTS.admin.features);
            if (response.ok) {
                const flags = await response.json();
                setSnapshotEnabled(flags.snapshot_management_enabled !== false);
            }
        })();
        fetchSnapshots();
    }, [portalFetch, fetchSnapshots]);

    const createSnapshot = async () => {
        setCreating(true);
        setError(null);
        setSuccess('');
        try {
            const response = await portalFetch(API_ENDPOINTS.admin.createSnapshot, { method: 'POST' });
            if (response.ok) {
                const data = await response.json();
                setSuccess(`Snapshot created: ${data.snapshot_name}`);
                await fetchSnapshots();
            } else {
                const err = await response.json().catch(() => ({}));
                throw new Error(err.detail || `Failed to create snapshot (HTTP ${response.status})`);
            }
        } catch (err: any) {
            setError(err.message || 'Failed to create snapshot');
        } finally {
            setCreating(false);
        }
    };

    const restoreSnapshot = async (name: string) => {
        setBusy({ name, op: 'restore' });
        setError(null);
        setSuccess('');
        try {
            const response = await portalFetch(API_ENDPOINTS.admin.restoreSnapshot(name), { method: 'POST' });
            if (response.ok) {
                const data = await response.json();
                setSuccess(`Restored: ${data.books_imported} books, ${data.chapters_imported} chapters`);
            } else {
                const err = await response.json().catch(() => ({}));
                throw new Error(err.detail || 'Failed to restore snapshot');
            }
        } catch (err: any) {
            setError(err.message || 'Failed to restore snapshot');
        } finally {
            setBusy(null);
            setSnapshotToRestore(null);
        }
    };

    const uploadSnapshot = async () => {
        if (!uploadSnapshotFile || !uploadMetadataFile) return;
        setCreating(true);
        setError(null);
        setSuccess('');
        try {
            const formData = new FormData();
            formData.append('snapshot_file', uploadSnapshotFile);
            formData.append('metadata_file', uploadMetadataFile);
            const response = await portalFetch(API_ENDPOINTS.admin.uploadSnapshot, {
                method: 'POST', body: formData,
            });
            if (response.ok) {
                const data = await response.json();
                setSuccess(`Uploaded: ${data.books_imported} books, ${data.chapters_imported} chapters`);
                await fetchSnapshots();
            } else {
                const err = await response.json().catch(() => ({}));
                throw new Error(err.detail || 'Failed to upload snapshot');
            }
        } catch (err: any) {
            setError(err.message || 'Failed to upload snapshot');
        } finally {
            setCreating(false);
            setShowUploadDialog(false);
            setUploadSnapshotFile(null);
            setUploadMetadataFile(null);
        }
    };

    // Anchor-href downloads can't carry the Authorization header, so fetch
    // the file as a blob and trigger the download from an object URL.
    const downloadAsBlob = async (path: string, filename: string) => {
        const response = await portalFetch(path);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const blobUrl = URL.createObjectURL(await response.blob());
        const link = document.createElement('a');
        link.href = blobUrl;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
    };

    const downloadSnapshot = async (name: string) => {
        setBusy({ name, op: 'download' });
        setError(null);
        try {
            await downloadAsBlob(API_ENDPOINTS.admin.downloadSnapshot(name), name);
        } catch (err: any) {
            setError(`Failed to download snapshot (${err.message})`);
        } finally {
            setBusy(null);
        }
    };

    const downloadMetadata = async (name: string) => {
        setBusy({ name, op: 'metadata' });
        setError(null);
        try {
            await downloadAsBlob(API_ENDPOINTS.admin.downloadMetadata(name), `${name}.metadata.json`);
        } catch (err: any) {
            setError(`Failed to download metadata (${err.message})`);
        } finally {
            setBusy(null);
        }
    };

    const deleteSnapshot = async (name: string) => {
        setBusy({ name, op: 'delete' });
        setError(null);
        setSuccess('');
        try {
            const response = await portalFetch(API_ENDPOINTS.admin.deleteSnapshot(name), { method: 'DELETE' });
            if (response.ok) {
                setSuccess(`Deleted snapshot: ${name}`);
                await fetchSnapshots();
            } else {
                const err = await response.json().catch(() => ({}));
                throw new Error(err.detail || `Failed to delete snapshot (HTTP ${response.status})`);
            }
        } catch (err: any) {
            setError(err.message || 'Failed to delete snapshot');
        } finally {
            setBusy(null);
            setSnapshotToDelete(null);
        }
    };

    return (
        <div className="space-y-6">
            <Card className="rounded-xl border border-primary/20">
                <CardHeader>
                    <CardTitle className="text-primary">Service Status</CardTitle>
                    <CardDescription>Live health of every service, refreshed every 15s</CardDescription>
                </CardHeader>
                <CardContent>
                    <ServiceStatusPanel />
                </CardContent>
            </Card>

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

            {snapshotEnabled && (
                <Card className="rounded-xl border border-primary/20">
                    <CardHeader>
                        <CardTitle className="text-primary">Qdrant Snapshot Management</CardTitle>
                        <CardDescription>Backup and restore vector database snapshots</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="flex items-center justify-between">
                            <div>
                                <h3 className="text-lg font-medium text-secondary">Create New Snapshot</h3>
                                <p className="text-sm text-muted-foreground">Backup current embeddings</p>
                            </div>
                            <Button
                                className="rounded-xl bg-primary text-primary-foreground hover:bg-primary/90"
                                onClick={createSnapshot}
                                disabled={creating}
                            >
                                {creating ? (
                                    <>
                                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                        Working...
                                    </>
                                ) : (
                                    <>
                                        <Database className="h-4 w-4 mr-2" />
                                        Create Snapshot
                                    </>
                                )}
                            </Button>
                        </div>

                        <div className="flex items-center justify-between">
                            <div>
                                <h3 className="text-lg font-medium text-secondary">Upload Snapshot</h3>
                                <p className="text-sm text-muted-foreground">Load pre-built snapshot from file</p>
                            </div>
                            <Button
                                className="rounded-xl bg-blue-600 text-white hover:bg-blue-700"
                                onClick={() => setShowUploadDialog(true)}
                                disabled={creating}
                            >
                                <Upload className="h-4 w-4 mr-2" />
                                Upload Snapshot
                            </Button>
                        </div>

                        <div className="space-y-2">
                            <h3 className="text-lg font-medium text-secondary">Available Snapshots</h3>
                            {snapshotLoading ? (
                                <div className="flex items-center justify-center py-8">
                                    <Loader2 className="h-6 w-6 animate-spin text-primary" />
                                </div>
                            ) : snapshots.length === 0 ? (
                                <div className="text-center py-8 text-muted-foreground">
                                    No snapshots available
                                </div>
                            ) : (
                                <ScrollArea className="h-[300px] rounded-xl border p-4">
                                    <div className="space-y-2">
                                        {snapshots.map((snapshot) => (
                                            <div
                                                key={snapshot.name}
                                                className="flex items-center justify-between p-3 rounded-lg bg-muted/50 hover:bg-muted transition-colors"
                                            >
                                                <div className="flex-1">
                                                    <p className="font-medium text-secondary">{snapshot.name}</p>
                                                    <p className="text-sm text-muted-foreground">
                                                        {new Date(snapshot.created_at).toLocaleString()} • {(snapshot.size / 1024 / 1024).toFixed(2)} MB
                                                        {snapshot.has_metadata ? (
                                                            <span className="ml-2 text-green-600">
                                                                • {snapshot.metadata_books} books, {snapshot.metadata_chapters} chapters
                                                            </span>
                                                        ) : (
                                                            <span className="ml-2 text-yellow-600">• No metadata</span>
                                                        )}
                                                    </p>
                                                </div>
                                                <div className="flex gap-2">
                                                    <Button
                                                        size="sm"
                                                        className="rounded-xl bg-blue-600 text-white hover:bg-blue-700"
                                                        onClick={() => setSnapshotToRestore(snapshot.name)}
                                                        disabled={creating || !!busy || !snapshot.has_metadata}
                                                        title={!snapshot.has_metadata ? 'Metadata required for restore' : ''}
                                                    >
                                                        {isBusy(snapshot.name, 'restore') ? (
                                                            <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                                                        ) : (
                                                            <Upload className="h-4 w-4 mr-1" />
                                                        )}
                                                        {isBusy(snapshot.name, 'restore') ? 'Restoring...' : 'Restore'}
                                                    </Button>
                                                    <Button
                                                        size="sm"
                                                        className="rounded-xl bg-green-600 text-white hover:bg-green-700"
                                                        onClick={() => downloadSnapshot(snapshot.name)}
                                                        disabled={!!busy}
                                                    >
                                                        {isBusy(snapshot.name, 'download') ? (
                                                            <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                                                        ) : (
                                                            <Download className="h-4 w-4 mr-1" />
                                                        )}
                                                        Snapshot
                                                    </Button>
                                                    <Button
                                                        size="sm"
                                                        className="rounded-xl bg-green-500 text-white hover:bg-green-600"
                                                        onClick={() => downloadMetadata(snapshot.name)}
                                                        disabled={!!busy || !snapshot.has_metadata}
                                                    >
                                                        {isBusy(snapshot.name, 'metadata') ? (
                                                            <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                                                        ) : (
                                                            <FileJson className="h-4 w-4 mr-1" />
                                                        )}
                                                        Metadata
                                                    </Button>
                                                    <Button
                                                        size="sm"
                                                        variant="destructive"
                                                        className="rounded-xl"
                                                        onClick={() => setSnapshotToDelete(snapshot.name)}
                                                        disabled={!!busy}
                                                    >
                                                        {isBusy(snapshot.name, 'delete') ? (
                                                            <Loader2 className="h-4 w-4 animate-spin" />
                                                        ) : (
                                                            <Trash2 className="h-4 w-4" />
                                                        )}
                                                    </Button>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </ScrollArea>
                            )}
                        </div>
                    </CardContent>
                </Card>
            )}

            <AlertDialog open={!!snapshotToRestore} onOpenChange={(open: boolean) => !open && setSnapshotToRestore(null)}>
                <AlertDialogContent className="bg-muted border-primary/20">
                    <AlertDialogHeader>
                        <AlertDialogTitle className="text-primary">Restore Snapshot</AlertDialogTitle>
                        <AlertDialogDescription className="text-muted-foreground">
                            This will restore the snapshot &quot;{snapshotToRestore}&quot; and import its stored metadata (books and chapters). Existing data will be overwritten.
                        </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                        <AlertDialogCancel className="border-muted-foreground/30 text-secondary hover:bg-muted-foreground/10 hover:text-secondary">Cancel</AlertDialogCancel>
                        <AlertDialogAction
                            className="bg-blue-600 text-white hover:bg-blue-700"
                            onClick={() => snapshotToRestore && restoreSnapshot(snapshotToRestore)}
                        >
                            Restore
                        </AlertDialogAction>
                    </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialog>

            <AlertDialog open={!!snapshotToDelete} onOpenChange={(open: boolean) => !open && setSnapshotToDelete(null)}>
                <AlertDialogContent className="bg-muted border-primary/20">
                    <AlertDialogHeader>
                        <AlertDialogTitle className="text-primary">Delete Snapshot</AlertDialogTitle>
                        <AlertDialogDescription className="text-muted-foreground">
                            Are you sure you want to delete the snapshot &quot;{snapshotToDelete}&quot;? This will also remove its metadata file. This action cannot be undone.
                        </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                        <AlertDialogCancel className="border-muted-foreground/30 text-secondary hover:bg-muted-foreground/10 hover:text-secondary">Cancel</AlertDialogCancel>
                        <AlertDialogAction
                            className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                            onClick={() => snapshotToDelete && deleteSnapshot(snapshotToDelete)}
                        >
                            Delete
                        </AlertDialogAction>
                    </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialog>

            <AlertDialog open={showUploadDialog} onOpenChange={(open: boolean) => {
                if (!open) {
                    setShowUploadDialog(false);
                    setUploadSnapshotFile(null);
                    setUploadMetadataFile(null);
                }
            }}>
                <AlertDialogContent className="bg-muted border-primary/20">
                    <AlertDialogHeader>
                        <AlertDialogTitle className="text-primary">Upload External Snapshot</AlertDialogTitle>
                        <AlertDialogDescription className="text-muted-foreground">
                            Upload a Qdrant snapshot file and its metadata JSON. Both files are required to restore books and chapters data.
                        </AlertDialogDescription>
                    </AlertDialogHeader>
                    <div className="space-y-4 py-4">
                        <div className="space-y-2">
                            <p className="text-sm font-medium text-secondary">Snapshot File (.snapshot)</p>
                            <Input
                                type="file"
                                accept=".snapshot"
                                className="border-muted-foreground/30 text-secondary file:text-secondary file:mr-3"
                                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setUploadSnapshotFile(e.target.files?.[0] || null)}
                            />
                        </div>
                        <div className="space-y-2">
                            <p className="text-sm font-medium text-secondary">Metadata File (.json)</p>
                            <Input
                                type="file"
                                accept=".json"
                                className="border-muted-foreground/30 text-secondary file:text-secondary file:mr-3"
                                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setUploadMetadataFile(e.target.files?.[0] || null)}
                            />
                        </div>
                    </div>
                    <AlertDialogFooter>
                        <AlertDialogCancel className="border-muted-foreground/30 text-secondary hover:bg-muted-foreground/10 hover:text-secondary">Cancel</AlertDialogCancel>
                        <AlertDialogAction
                            className="bg-primary text-primary-foreground hover:bg-primary/90"
                            onClick={uploadSnapshot}
                            disabled={!uploadSnapshotFile || !uploadMetadataFile}
                        >
                            Upload & Restore
                        </AlertDialogAction>
                    </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialog>
        </div>
    );
};

export default SystemManagement;
