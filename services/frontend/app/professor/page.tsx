// app/professor/page.tsx
'use client';

import { useState, useEffect } from 'react';
import { BarChart3, Book, GraduationCap, ListTree } from 'lucide-react';

import PortalShell, { PortalConfig, usePortal } from '@/components/portal/PortalShell';
import { API_ENDPOINTS } from '@/config/constants';
import ClassManagement from '@/components/professor/ClassManagement';
import ClassBooks from '@/components/professor/ClassBooks';
import TopicsEditor from '@/components/professor/TopicsEditor';
import ClassAnalytics from '@/components/shared/ClassAnalytics';

interface ProfessorFeatures {
    summary_enabled: boolean;
    registration_enrollment_enabled: boolean;
    join_code_enabled: boolean;
    book_upload_enabled: boolean;
}

const DEFAULT_FEATURES: ProfessorFeatures = {
    summary_enabled: false,
    registration_enrollment_enabled: true,
    join_code_enabled: true,
    book_upload_enabled: true,
};

const useProfessorFeatures = (): ProfessorFeatures => {
    const { portalFetch } = usePortal();
    const [features, setFeatures] = useState<ProfessorFeatures>(DEFAULT_FEATURES);
    useEffect(() => {
        (async () => {
            const response = await portalFetch(API_ENDPOINTS.professor.features);
            if (response.ok) setFeatures(await response.json());
        })();
    }, [portalFetch]);
    return features;
};

const TurmasSection = () => {
    const features = useProfessorFeatures();
    return <ClassManagement features={features} />;
};

const LivrosSection = () => {
    const features = useProfessorFeatures();
    return <ClassBooks features={features} />;
};

const AnalisesSection = () => {
    const features = useProfessorFeatures();
    return (
        <ClassAnalytics
            summaryEnabled={features.summary_enabled}
            endpoints={{
                classes: API_ENDPOINTS.professor.classes,
                analyticsTopics: API_ENDPOINTS.professor.analyticsTopics,
                analyticsQueries: API_ENDPOINTS.professor.analyticsQueries,
                analyticsSummary: API_ENDPOINTS.professor.analyticsSummary,
            }}
        />
    );
};

const PROFESSOR_CONFIG: PortalConfig = {
    title: 'AcademiCK Professor',
    subtitle: 'Gerencie suas turmas, livros e tópicos',
    loginEndpoint: API_ENDPOINTS.professor.login,
    validateEndpoint: API_ENDPOINTS.professor.validateSession,
    storageKey: 'professorSession',
    loginLabels: { username: 'Usuário', password: 'Senha', submit: 'Entrar' },
    sections: [
        { id: 'turmas', label: 'Turmas', icon: GraduationCap, component: TurmasSection },
        { id: 'livros', label: 'Livros', icon: Book, component: LivrosSection },
        { id: 'topicos', label: 'Tópicos', icon: ListTree, component: TopicsEditor },
        { id: 'analises', label: 'Análises', icon: BarChart3, component: AnalisesSection },
    ],
};

export default function ProfessorPage() {
    return <PortalShell config={PROFESSOR_CONFIG} />;
}
