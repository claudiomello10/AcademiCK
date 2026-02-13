import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { AlertCircle, Loader2 } from 'lucide-react';

import { API_BASE_URL, API_ENDPOINTS } from '@/config/constants';

const AdminLoginForm = ({ onLogin, isLoading, error }) => (
    <Card className="w-full max-w-md mx-auto mt-20 rounded-2xl shadow-lg border border-primary/30">
        <CardHeader className="flex flex-col items-center space-y-4">
            <CardTitle className="text-center text-primary">
                Admin Dashboard
            </CardTitle>
            <CardDescription className="text-center">
                Login to access admin features
            </CardDescription>
        </CardHeader>
        <CardContent>
            <form onSubmit={onLogin} className="space-y-6">
                <div className="space-y-4">
                    <Input
                        type="text"
                        name="username"
                        placeholder="Username"
                        required
                        className="rounded-xl h-12 bg-secondary border-border"
                    />
                    <Input
                        type="password"
                        name="password"
                        placeholder="Password"
                        required
                        className="rounded-xl h-12 bg-secondary border-border"
                    />
                </div>
                {error && (
                    <Alert variant="destructive" className="rounded-xl">
                        <AlertCircle className="h-4 w-4" />
                        <AlertTitle>Error</AlertTitle>
                        <AlertDescription>{error}</AlertDescription>
                    </Alert>
                )}
                <Button
                    type="submit"
                    className="w-full h-12 rounded-xl transition-all duration-200 hover:opacity-90 bg-primary text-primary-foreground"
                    disabled={isLoading}
                >
                    {isLoading ? (
                        <Loader2 className="h-5 w-5 animate-spin mr-2" />
                    ) : 'Login'}
                </Button>
            </form>
        </CardContent>
    </Card>
);

const AdminLayout = ({ children }) => {
    const [sessionId, setSessionId] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const savedSession = localStorage.getItem('adminSession');
        if (savedSession) {
            validateSession(savedSession);
        } else {
            setLoading(false);
        }
    }, []);

    const validateSession = async (sessionId) => {
        try {
            const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.admin.validateSession(sessionId)}`);
            const data = await response.json();
            if (data.valid) {
                setSessionId(sessionId);
            } else {
                localStorage.removeItem('adminSession');
            }
        } catch (err) {
            setError('Failed to validate session');
        } finally {
            setLoading(false);
        }
    };

    const handleLogin = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);

        const formData = new FormData(e.currentTarget);
        const username = formData.get('username');
        const password = formData.get('password');

        try {
            const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.admin.login}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ username, password })
            });

            const data = await response.json();
            if (response.ok) {
                setSessionId(data.session_id);
                localStorage.setItem('adminSession', data.session_id);
            } else {
                throw new Error(data.detail || 'Invalid credentials');
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-screen">
                <Loader2 className="h-8 w-8 animate-spin" />
            </div>
        );
    }

    if (!sessionId) {
        return <AdminLoginForm onLogin={handleLogin} isLoading={loading} error={error} />;
    }

    return children;
};

export default AdminLayout;