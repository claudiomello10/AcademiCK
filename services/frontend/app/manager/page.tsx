// app/manager/page.tsx
'use client';

import { useState, useEffect } from 'react';
import { BarChart3, GraduationCap, Users } from 'lucide-react';

import PortalShell, { PortalConfig, usePortal } from '@/components/portal/PortalShell';
import { API_ENDPOINTS } from '@/config/constants';
import UsersSection from '@/components/shared/UsersSection';
import ManagerClasses from '@/components/manager/ManagerClasses';
import ClassAnalytics from '@/components/shared/ClassAnalytics';

interface ManagerFeatures {
    summary_enabled: boolean;
    admin_assign_enabled: boolean;
}

const useManagerFeatures = (): ManagerFeatures => {
    const { portalFetch } = usePortal();
    const [features, setFeatures] = useState<ManagerFeatures>({
        summary_enabled: false,
        admin_assign_enabled: true,
    });
    useEffect(() => {
        (async () => {
            const response = await portalFetch(API_ENDPOINTS.manager.features);
            if (response.ok) setFeatures(await response.json());
        })();
    }, [portalFetch]);
    return features;
};

const UsuariosSection = () => (
    <UsersSection
        endpoints={{
            users: API_ENDPOINTS.manager.users,
            userById: API_ENDPOINTS.manager.userById,
            userStatus: API_ENDPOINTS.manager.userStatus,
            importUsers: API_ENDPOINTS.manager.importUsers,
        }}
        assignableRoles={['user', 'professor']}
        labels={{
            addUser: 'Novo Usuário',
            import: 'Importar Alunos',
            roleNames: { user: 'Aluno', professor: 'Professor', manager: 'Coordenador', admin: 'Admin' },
        }}
    />
);

const TurmasSection = () => {
    const features = useManagerFeatures();
    return <ManagerClasses assignEnabled={features.admin_assign_enabled} />;
};

const AnalisesSection = () => {
    const features = useManagerFeatures();
    return (
        <ClassAnalytics
            summaryEnabled={features.summary_enabled}
            endpoints={{
                classes: API_ENDPOINTS.manager.classes,
                analyticsTopics: API_ENDPOINTS.manager.analyticsTopics,
                analyticsQueries: API_ENDPOINTS.manager.analyticsQueries,
                analyticsSummary: API_ENDPOINTS.manager.analyticsSummary,
            }}
        />
    );
};

const MANAGER_CONFIG: PortalConfig = {
    title: 'AcademiCK Coordenação',
    subtitle: 'Gerencie usuários, turmas e acompanhe as análises',
    loginEndpoint: API_ENDPOINTS.manager.login,
    validateEndpoint: API_ENDPOINTS.manager.validateSession,
    storageKey: 'managerSession',
    loginLabels: { username: 'Usuário', password: 'Senha', submit: 'Entrar' },
    sections: [
        { id: 'usuarios', label: 'Usuários', icon: Users, component: UsuariosSection },
        { id: 'turmas', label: 'Turmas', icon: GraduationCap, component: TurmasSection },
        { id: 'analises', label: 'Análises', icon: BarChart3, component: AnalisesSection },
    ],
};

export default function ManagerPage() {
    return <PortalShell config={MANAGER_CONFIG} />;
}
