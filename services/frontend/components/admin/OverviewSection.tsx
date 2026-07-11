'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Book, GraduationCap, MessageSquare, Users } from 'lucide-react';

import { API_ENDPOINTS } from '@/config/constants';
import { usePortal } from '@/components/portal/PortalShell';
import { ServiceStatusPanel } from '@/components/admin/SystemManagement';

const OverviewSection = () => {
    const { portalFetch } = usePortal();
    const [counts, setCounts] = useState({ users: 0, classes: 0, books: 0, queries: 0 });

    useEffect(() => {
        (async () => {
            const [contentRes, classesRes, usageRes] = await Promise.all([
                portalFetch(API_ENDPOINTS.admin.contentStats),
                portalFetch(API_ENDPOINTS.admin.classes),
                portalFetch(API_ENDPOINTS.admin.usageStats('7d')),
            ]);
            const next = { users: 0, classes: 0, books: 0, queries: 0 };
            if (contentRes.ok) {
                const data = await contentRes.json();
                next.users = data.total_users ?? 0;
                next.books = data.total_books ?? 0;
            }
            if (classesRes.ok) next.classes = (await classesRes.json()).classes.length;
            if (usageRes.ok) next.queries = (await usageRes.json()).total_queries ?? 0;
            setCounts(next);
        })();
    }, [portalFetch]);

    return (
        <div className="space-y-6">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                    { icon: Users, label: 'Users', value: counts.users },
                    { icon: GraduationCap, label: 'Classes', value: counts.classes },
                    { icon: Book, label: 'Books', value: counts.books },
                    { icon: MessageSquare, label: 'Queries (7d)', value: counts.queries },
                ].map(({ icon: Icon, label, value }) => (
                    <Card key={label} className="rounded-xl border border-primary/20">
                        <CardContent className="p-4">
                            <div className="flex items-center gap-3">
                                <Icon className="h-5 w-5 text-primary" />
                                <div className="text-secondary">
                                    <p className="text-sm font-medium">{label}</p>
                                    <p className="text-2xl font-bold">{value}</p>
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                ))}
            </div>

            <Card className="rounded-xl border border-primary/20">
                <CardHeader>
                    <CardTitle className="text-primary">Service Status</CardTitle>
                    <CardDescription>Live health of every service</CardDescription>
                </CardHeader>
                <CardContent>
                    <ServiceStatusPanel compact />
                </CardContent>
            </Card>
        </div>
    );
};

export default OverviewSection;
