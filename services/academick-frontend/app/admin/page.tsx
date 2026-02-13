// app/admin/page.tsx
'use client';

import AdminLayout from '@/components/AdminLayout';
import AdminDashboard from '@/components/AdminDashboard';

export default function AdminPage() {
    return (
        <AdminLayout>
            <main className="min-h-screen p-4">
                <AdminDashboard />
            </main>
        </AdminLayout>
    );
}