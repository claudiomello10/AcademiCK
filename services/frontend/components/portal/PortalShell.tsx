'use client';

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { AlertCircle, Loader2, LogOut, Menu, X, LucideIcon } from 'lucide-react';
import { cn } from '@/lib/utils';

import { API_BASE_URL, API_ENDPOINTS, authHeaders } from '@/config/constants';

export interface PortalSection {
    id: string;
    label: string;
    icon: LucideIcon;
    component: React.ComponentType;
}

export interface PortalConfig {
    title: string;
    subtitle: string;
    loginEndpoint: string;
    validateEndpoint: string;
    storageKey: string;
    loginLabels: { username: string; password: string; submit: string };
    sections: PortalSection[];
}

interface PortalContextValue {
    sessionId: string;
    username: string;
    role: string;
    // fetch wrapper: a 401 clears the session and bounces back to login
    portalFetch: (path: string, init?: RequestInit) => Promise<Response>;
    logout: () => void;
}

const PortalContext = createContext<PortalContextValue | null>(null);

export const usePortal = (): PortalContextValue => {
    const ctx = useContext(PortalContext);
    if (!ctx) throw new Error('usePortal must be used inside a PortalShell');
    return ctx;
};

const LoginForm = ({
    config, onLogin, isLoading, error,
}: {
    config: PortalConfig;
    onLogin: (e: React.FormEvent<HTMLFormElement>) => void;
    isLoading: boolean;
    error: string | null;
}) => (
    <div className="min-h-screen flex items-center justify-center p-4">
        <Card className="w-full max-w-md rounded-2xl shadow-lg border border-primary/30">
            <CardHeader className="flex flex-col items-center space-y-2">
                <CardTitle className="text-center text-primary text-2xl">
                    {config.title}
                </CardTitle>
                <CardDescription className="text-center">
                    {config.subtitle}
                </CardDescription>
            </CardHeader>
            <CardContent>
                <form onSubmit={onLogin} className="space-y-6">
                    <div className="space-y-4">
                        <Input
                            type="text"
                            name="username"
                            placeholder={config.loginLabels.username}
                            required
                            autoFocus
                            className="rounded-xl h-12 bg-secondary border-border"
                        />
                        <Input
                            type="password"
                            name="password"
                            placeholder={config.loginLabels.password}
                            required
                            className="rounded-xl h-12 bg-secondary border-border"
                        />
                    </div>
                    {error && (
                        <Alert variant="destructive" className="rounded-xl">
                            <AlertCircle className="h-4 w-4" />
                            <AlertTitle>Erro</AlertTitle>
                            <AlertDescription>{error}</AlertDescription>
                        </Alert>
                    )}
                    <Button
                        type="submit"
                        className="w-full h-12 rounded-xl transition-all duration-200 hover:opacity-90 bg-primary text-primary-foreground"
                        disabled={isLoading}
                    >
                        {isLoading ? (
                            <Loader2 className="h-5 w-5 animate-spin" />
                        ) : config.loginLabels.submit}
                    </Button>
                </form>
            </CardContent>
        </Card>
    </div>
);

const PortalShell = ({ config }: { config: PortalConfig }) => {
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [username, setUsername] = useState('');
    const [role, setRole] = useState('');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [activeSection, setActiveSection] = useState(config.sections[0].id);
    const [sidebarOpen, setSidebarOpen] = useState(false);

    const clearSession = useCallback(() => {
        localStorage.removeItem(config.storageKey);
        setSessionId(null);
        setUsername('');
        setRole('');
    }, [config.storageKey]);

    useEffect(() => {
        const saved = localStorage.getItem(config.storageKey);
        if (!saved) {
            setLoading(false);
            return;
        }
        (async () => {
            try {
                const response = await fetch(
                    `${API_BASE_URL}${config.validateEndpoint}`,
                    { headers: authHeaders(saved) }
                );
                const data = await response.json();
                if (data.valid) {
                    setSessionId(saved);
                    setUsername(data.username || '');
                    setRole(data.role || '');
                } else {
                    localStorage.removeItem(config.storageKey);
                }
            } catch {
                setError('Falha ao validar a sessão');
            } finally {
                setLoading(false);
            }
        })();
    }, [config.storageKey, config.validateEndpoint]);

    const handleLogin = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        setLoading(true);
        setError(null);

        const formData = new FormData(e.currentTarget);
        try {
            const response = await fetch(`${API_BASE_URL}${config.loginEndpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    username: formData.get('username'),
                    password: formData.get('password'),
                }),
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.detail || 'Credenciais inválidas');
            }
            localStorage.setItem(config.storageKey, data.session_id);
            setSessionId(data.session_id);
            setUsername(data.username || '');
            setRole(data.role || '');
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Falha no login');
        } finally {
            setLoading(false);
        }
    };

    const portalFetch = useCallback(async (path: string, init: RequestInit = {}) => {
        const response = await fetch(`${API_BASE_URL}${path}`, {
            ...init,
            headers: { ...authHeaders(sessionId || ''), ...(init.headers || {}) },
        });
        if (response.status === 401) {
            clearSession();
        }
        return response;
    }, [sessionId, clearSession]);

    const logout = useCallback(async () => {
        if (sessionId) {
            try {
                await fetch(`${API_BASE_URL}${API_ENDPOINTS.logout}`, {
                    method: 'POST',
                    headers: authHeaders(sessionId),
                });
            } catch {
                // Session is being discarded either way
            }
        }
        clearSession();
    }, [sessionId, clearSession]);

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-screen">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
        );
    }

    if (!sessionId) {
        return <LoginForm config={config} onLogin={handleLogin} isLoading={loading} error={error} />;
    }

    const active = config.sections.find(s => s.id === activeSection) || config.sections[0];
    const ActiveComponent = active.component;

    const nav = (
        <nav className="flex flex-col gap-1 px-3">
            {config.sections.map(section => {
                const Icon = section.icon;
                return (
                    <button
                        key={section.id}
                        onClick={() => { setActiveSection(section.id); setSidebarOpen(false); }}
                        className={cn(
                            'flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-colors text-left',
                            activeSection === section.id
                                ? 'bg-primary text-primary-foreground'
                                : 'text-secondary hover:bg-muted'
                        )}
                    >
                        <Icon className="h-4 w-4 shrink-0" />
                        {section.label}
                    </button>
                );
            })}
        </nav>
    );

    const sidebarInner = (
        <div className="flex flex-col h-full py-4">
            <div className="px-6 pb-4 border-b border-border mb-4">
                <h1 className="text-lg font-bold text-primary">{config.title}</h1>
                <p className="text-xs text-muted-foreground mt-1 truncate">{username} · {role}</p>
            </div>
            {nav}
            <div className="mt-auto px-3 pt-4 border-t border-border">
                <button
                    onClick={logout}
                    className="flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium text-destructive hover:bg-destructive hover:text-destructive-foreground transition-colors w-full text-left"
                >
                    <LogOut className="h-4 w-4 shrink-0" />
                    {config.loginLabels.submit === 'Login' ? 'Logout' : 'Sair'}
                </button>
            </div>
        </div>
    );

    return (
        <PortalContext.Provider value={{ sessionId, username, role, portalFetch, logout }}>
            <div className="min-h-screen flex">
                {/* Desktop sidebar */}
                <aside className="hidden md:flex w-60 shrink-0 border-r border-border flex-col sticky top-0 h-screen">
                    {sidebarInner}
                </aside>

                {/* Mobile sidebar */}
                {sidebarOpen && (
                    <div className="fixed inset-0 z-50 md:hidden">
                        <div className="absolute inset-0 bg-black/50" onClick={() => setSidebarOpen(false)} />
                        <aside className="absolute left-0 top-0 h-full w-64 bg-background border-r border-border">
                            <button
                                className="absolute right-3 top-3 text-muted-foreground"
                                onClick={() => setSidebarOpen(false)}
                            >
                                <X className="h-5 w-5" />
                            </button>
                            {sidebarInner}
                        </aside>
                    </div>
                )}

                <div className="flex-1 min-w-0">
                    <header className="md:hidden flex items-center gap-3 p-4 border-b border-border sticky top-0 bg-background z-40">
                        <button onClick={() => setSidebarOpen(true)}>
                            <Menu className="h-5 w-5 text-primary" />
                        </button>
                        <h1 className="font-bold text-primary">{config.title}</h1>
                    </header>
                    <main className="p-4 md:p-8 max-w-6xl mx-auto">
                        <h2 className="text-2xl font-bold text-primary mb-6">{active.label}</h2>
                        <ActiveComponent />
                    </main>
                </div>
            </div>
        </PortalContext.Provider>
    );
};

export default PortalShell;
