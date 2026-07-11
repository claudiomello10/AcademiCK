// app/admin/page.tsx
'use client';

import {
    Activity, Book, GraduationCap, LayoutDashboard, ListChecks, Settings, Users,
} from 'lucide-react';

import PortalShell, { PortalConfig } from '@/components/portal/PortalShell';
import { API_ENDPOINTS } from '@/config/constants';
import OverviewSection from '@/components/admin/OverviewSection';
import UsersSection from '@/components/shared/UsersSection';
import ClassAdministration from '@/components/admin/ClassAdministration';
import BooksManagement from '@/components/admin/BooksManagement';
import JobsManagement from '@/components/admin/JobsManagement';
import SystemManagement from '@/components/admin/SystemManagement';
import AnalyticsSection from '@/components/admin/AnalyticsSection';

const AdminUsers = () => (
    <UsersSection
        endpoints={{
            users: API_ENDPOINTS.admin.users,
            userById: API_ENDPOINTS.admin.userById,
            userStatus: API_ENDPOINTS.admin.userStatus,
            importUsers: API_ENDPOINTS.admin.importUsers,
        }}
        assignableRoles={['user', 'professor', 'manager', 'admin']}
        labels={{
            addUser: 'Add User',
            import: 'Import Users',
            roleNames: { user: 'Student', professor: 'Professor', manager: 'Manager', admin: 'Admin' },
        }}
    />
);

const ADMIN_CONFIG: PortalConfig = {
    title: 'AcademiCK Admin',
    subtitle: 'Login to access admin features',
    loginEndpoint: API_ENDPOINTS.admin.login,
    validateEndpoint: API_ENDPOINTS.admin.validateSession,
    storageKey: 'adminSession',
    loginLabels: { username: 'Username', password: 'Password', submit: 'Login' },
    sections: [
        { id: 'overview', label: 'Overview', icon: LayoutDashboard, component: OverviewSection },
        { id: 'users', label: 'Users', icon: Users, component: AdminUsers },
        { id: 'classes', label: 'Classes', icon: GraduationCap, component: ClassAdministration },
        { id: 'books', label: 'Books', icon: Book, component: BooksManagement },
        { id: 'jobs', label: 'Jobs', icon: ListChecks, component: JobsManagement },
        { id: 'system', label: 'System', icon: Settings, component: SystemManagement },
        { id: 'analytics', label: 'Analytics', icon: Activity, component: AnalyticsSection },
    ],
};

export default function AdminPage() {
    return <PortalShell config={ADMIN_CONFIG} />;
}
