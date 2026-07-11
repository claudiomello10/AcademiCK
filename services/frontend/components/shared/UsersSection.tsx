'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import {
    Users, Shield, Activity, UserPlus, Edit, GraduationCap, Briefcase,
    AlertCircle, Loader2, Check, X, Upload, FileText,
} from 'lucide-react';

import { usePortal } from '@/components/portal/PortalShell';

export interface UsersSectionEndpoints {
    users: string;
    userById: (id: string) => string;
    userStatus: (id: string) => string;
    importUsers: string;
}

interface UsersSectionProps {
    endpoints: UsersSectionEndpoints;
    // Roles this portal may assign (managers cannot create managers/admins)
    assignableRoles: string[];
    labels: {
        addUser: string; import: string; roleNames: Record<string, string>;
    };
}

interface UserInfo {
    id: string;
    username: string;
    email?: string;
    role: string;
    status: string;
    registration_number?: string;
    last_active?: string;
}

interface UserFormData {
    username: string;
    password: string;
    role: string;
    email: string;
    registration_number: string;
}

const EMPTY_FORM: UserFormData = {
    username: '', password: '', role: 'user', email: '', registration_number: '',
};

const ROLE_ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
    admin: Shield,
    manager: Briefcase,
    professor: GraduationCap,
    user: Users,
};

const parseDetail = (data: any, fallback: string): string => {
    if (!data?.detail) return fallback;
    if (Array.isArray(data.detail)) {
        return data.detail.map((err: { msg: string; loc?: string[] }) =>
            err.loc ? `${err.loc.join('.')}: ${err.msg}` : err.msg
        ).join(', ');
    }
    return typeof data.detail === 'string' ? data.detail : fallback;
};

const UserForm = ({
    userForm, setUserForm, onSubmit, isEdit, assignableRoles, roleNames,
}: {
    userForm: UserFormData;
    setUserForm: React.Dispatch<React.SetStateAction<UserFormData>>;
    onSubmit: () => void;
    isEdit: boolean;
    assignableRoles: string[];
    roleNames: Record<string, string>;
}) => (
    <div className="space-y-4 text-secondary">
        <div className="space-y-2">
            <Label htmlFor="username">Username</Label>
            <Input
                id="username"
                value={userForm.username}
                disabled={isEdit}
                onChange={(e) => setUserForm(prev => ({ ...prev, username: e.target.value }))}
            />
        </div>
        <div className="space-y-2">
            <Label htmlFor="registration_number">Registration number {userForm.role === 'user' ? '' : '(optional)'}</Label>
            <Input
                id="registration_number"
                value={userForm.registration_number}
                onChange={(e) => setUserForm(prev => ({ ...prev, registration_number: e.target.value }))}
            />
        </div>
        <div className="space-y-2">
            <Label htmlFor="email">Email (optional)</Label>
            <Input
                id="email"
                type="email"
                value={userForm.email}
                onChange={(e) => setUserForm(prev => ({ ...prev, email: e.target.value }))}
            />
        </div>
        {!isEdit && (
            <div className="space-y-2">
                <Label htmlFor="password">Password</Label>
                <Input
                    id="password"
                    type="password"
                    value={userForm.password}
                    onChange={(e) => setUserForm(prev => ({ ...prev, password: e.target.value }))}
                />
            </div>
        )}
        <div className="space-y-2">
            <Label htmlFor="role">Role</Label>
            <Select
                value={userForm.role}
                onValueChange={(value) => setUserForm(prev => ({ ...prev, role: value }))}
            >
                <SelectTrigger>
                    <SelectValue placeholder="Select role" />
                </SelectTrigger>
                <SelectContent>
                    {assignableRoles.map(role => (
                        <SelectItem key={role} value={role}>{roleNames[role] || role}</SelectItem>
                    ))}
                </SelectContent>
            </Select>
        </div>
        <Button className="w-full bg-primary text-primary-foreground" onClick={onSubmit}>
            {isEdit ? 'Save' : 'Create'}
        </Button>
    </div>
);

const UsersSection = ({ endpoints, assignableRoles, labels }: UsersSectionProps) => {
    const { portalFetch } = usePortal();
    const [users, setUsers] = useState<UserInfo[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [showAddUser, setShowAddUser] = useState(false);
    const [showEditUser, setShowEditUser] = useState(false);
    const [selectedUser, setSelectedUser] = useState<UserInfo | null>(null);
    const [userForm, setUserForm] = useState<UserFormData>(EMPTY_FORM);

    const [showImport, setShowImport] = useState(false);
    const [importFile, setImportFile] = useState<File | null>(null);
    const [importDefaultPassword, setImportDefaultPassword] = useState('');
    const [importing, setImporting] = useState(false);
    const [importResult, setImportResult] = useState<{
        created: number;
        errors: { row: number; username?: string; error: string }[];
    } | null>(null);

    const fetchUsers = useCallback(async () => {
        setLoading(true);
        try {
            const response = await portalFetch(endpoints.users);
            if (!response.ok) throw new Error('Failed to fetch users');
            setUsers(await response.json());
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to fetch users');
        } finally {
            setLoading(false);
        }
    }, [portalFetch, endpoints.users]);

    useEffect(() => { fetchUsers(); }, [fetchUsers]);

    const submitUser = async (isEdit: boolean) => {
        setError(null);
        try {
            const payload: Record<string, string | null> = {
                email: userForm.email || null,
                role: userForm.role,
                registration_number: userForm.registration_number || null,
            };
            let response: Response;
            if (isEdit && selectedUser) {
                response = await portalFetch(endpoints.userById(selectedUser.id), {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
            } else {
                response = await portalFetch(endpoints.users, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ...payload, username: userForm.username, password: userForm.password }),
                });
            }
            if (!response.ok) {
                throw new Error(parseDetail(await response.json(), isEdit ? 'Failed to update user' : 'Failed to add user'));
            }
            setShowAddUser(false);
            setShowEditUser(false);
            setSelectedUser(null);
            setUserForm(EMPTY_FORM);
            fetchUsers();
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Request failed');
        }
    };

    const toggleStatus = async (user: UserInfo) => {
        setError(null);
        const response = await portalFetch(endpoints.userStatus(user.id), {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status: user.status === 'active' ? 'inactive' : 'active' }),
        });
        if (response.ok) {
            fetchUsers();
        } else {
            setError(parseDetail(await response.json().catch(() => ({})), 'Failed to update status'));
        }
    };

    const runImport = async () => {
        if (!importFile) return;
        setImporting(true);
        setImportResult(null);
        setError(null);
        try {
            const formData = new FormData();
            formData.append('file', importFile);
            if (importDefaultPassword) formData.append('default_password', importDefaultPassword);
            const response = await portalFetch(endpoints.importUsers, {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();
            if (!response.ok) throw new Error(parseDetail(data, 'Import failed'));
            setImportResult(data);
            fetchUsers();
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Import failed');
        } finally {
            setImporting(false);
        }
    };

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between flex-wrap gap-2">
                <p className="text-sm text-muted-foreground">
                    Students need a unique registration number.
                </p>
                <div className="flex gap-2">
                    <Button
                        variant="outline"
                        className="rounded-xl"
                        onClick={() => { setShowImport(true); setImportResult(null); setImportFile(null); }}
                    >
                        <Upload className="h-4 w-4 mr-2" />
                        {labels.import}
                    </Button>
                    <Dialog open={showAddUser} onOpenChange={(open) => { setShowAddUser(open); if (open) setUserForm(EMPTY_FORM); }}>
                        <DialogTrigger asChild>
                            <Button className="rounded-xl bg-primary text-primary-foreground">
                                <UserPlus className="h-4 w-4 mr-2" />
                                {labels.addUser}
                            </Button>
                        </DialogTrigger>
                        <DialogContent>
                            <DialogHeader>
                                <DialogTitle>{labels.addUser}</DialogTitle>
                            </DialogHeader>
                            <UserForm
                                userForm={userForm}
                                setUserForm={setUserForm}
                                onSubmit={() => submitUser(false)}
                                isEdit={false}
                                assignableRoles={assignableRoles}
                                roleNames={labels.roleNames}
                            />
                        </DialogContent>
                    </Dialog>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {[
                    { icon: Users, label: 'Total', value: users.length },
                    { icon: GraduationCap, label: labels.roleNames.professor || 'Professors', value: users.filter(u => u.role === 'professor').length },
                    { icon: Activity, label: 'Active', value: users.filter(u => u.status === 'active').length },
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

            {error && (
                <Alert variant="destructive" className="rounded-xl">
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                </Alert>
            )}

            <Card className="rounded-xl border border-primary/20">
                <CardHeader className="text-primary">
                    <CardTitle>Users</CardTitle>
                    <CardDescription>Accounts, roles and registration numbers</CardDescription>
                </CardHeader>
                <CardContent>
                    <ScrollArea className="h-[500px] pr-4">
                        {loading ? (
                            <div className="flex items-center justify-center p-4">
                                <Loader2 className="h-6 w-6 animate-spin" />
                            </div>
                        ) : (
                            <div className="space-y-2">
                                {users.map((user) => {
                                    const RoleIcon = ROLE_ICONS[user.role] || Users;
                                    return (
                                        <div key={user.id} className="p-4 border border-primary/20 rounded-lg bg-secondary hover:bg-secondary/80">
                                            <div className="flex items-center justify-between">
                                                <div className="flex flex-col min-w-0">
                                                    <div className="flex items-center gap-2">
                                                        <RoleIcon className="h-4 w-4 shrink-0" />
                                                        <span className="font-medium truncate">{user.username}</span>
                                                        <span className="text-xs px-2 py-0.5 rounded-full bg-muted">
                                                            {labels.roleNames[user.role] || user.role}
                                                        </span>
                                                    </div>
                                                    <span className="text-sm truncate">
                                                        {user.registration_number && (
                                                            <span className="font-mono mr-2">{user.registration_number}</span>
                                                        )}
                                                        {user.email}
                                                    </span>
                                                </div>
                                                <div className="flex items-center gap-2 shrink-0">
                                                    <span className={`px-2 py-1 rounded-full text-xs font-bold ${user.status === 'active'
                                                        ? 'bg-primary text-black'
                                                        : 'bg-muted text-white'
                                                        }`}>
                                                        {user.status}
                                                    </span>
                                                    <Button
                                                        variant="ghost"
                                                        size="sm"
                                                        className="h-8 w-8 p-0"
                                                        onClick={() => {
                                                            setSelectedUser(user);
                                                            setUserForm({
                                                                username: user.username,
                                                                role: user.role,
                                                                email: user.email || '',
                                                                registration_number: user.registration_number || '',
                                                                password: '',
                                                            });
                                                            setShowEditUser(true);
                                                        }}
                                                    >
                                                        <Edit className="h-4 w-4" />
                                                    </Button>
                                                    <Button
                                                        variant="ghost"
                                                        size="sm"
                                                        className="h-8 w-8 p-0"
                                                        onClick={() => toggleStatus(user)}
                                                    >
                                                        {user.status === 'active' ? (
                                                            <X className="h-4 w-4 text-red-500" />
                                                        ) : (
                                                            <Check className="h-4 w-4 text-green-500" />
                                                        )}
                                                    </Button>
                                                </div>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        )}
                    </ScrollArea>
                </CardContent>
            </Card>

            <Dialog open={showEditUser} onOpenChange={setShowEditUser}>
                <DialogContent>
                    <DialogHeader className="text-primary">
                        <DialogTitle>Edit user</DialogTitle>
                    </DialogHeader>
                    <UserForm
                        userForm={userForm}
                        setUserForm={setUserForm}
                        onSubmit={() => submitUser(true)}
                        isEdit
                        assignableRoles={assignableRoles}
                        roleNames={labels.roleNames}
                    />
                </DialogContent>
            </Dialog>

            <Dialog open={showImport} onOpenChange={setShowImport}>
                <DialogContent className="max-w-lg">
                    <DialogHeader>
                        <DialogTitle className="text-primary">{labels.import}</DialogTitle>
                        <DialogDescription>
                            CSV with a header line. Columns: username, registration_number,
                            password, email, role. Each row is reported individually.
                        </DialogDescription>
                    </DialogHeader>
                    <div className="space-y-4 py-2">
                        <Input
                            type="file"
                            accept=".csv"
                            className="rounded-xl file:mr-3"
                            onChange={(e) => setImportFile(e.target.files?.[0] || null)}
                        />
                        <div className="space-y-1">
                            <Label htmlFor="default-password">Default password (for rows without one)</Label>
                            <Input
                                id="default-password"
                                type="password"
                                value={importDefaultPassword}
                                onChange={(e) => setImportDefaultPassword(e.target.value)}
                                className="rounded-xl"
                            />
                        </div>
                        {importResult && (
                            <div className="space-y-2">
                                <Alert className="rounded-xl bg-green-50 border-green-200">
                                    <AlertTitle className="text-green-800">
                                        {importResult.created} user(s) created
                                    </AlertTitle>
                                </Alert>
                                {importResult.errors.length > 0 && (
                                    <ScrollArea className="max-h-40 rounded-xl border p-2">
                                        {importResult.errors.map((err, i) => (
                                            <div key={i} className="flex items-start gap-2 text-sm py-1">
                                                <FileText className="h-4 w-4 text-red-500 shrink-0 mt-0.5" />
                                                <span className="text-red-600">
                                                    Row {err.row}{err.username ? ` (${err.username})` : ''}: {err.error}
                                                </span>
                                            </div>
                                        ))}
                                    </ScrollArea>
                                )}
                            </div>
                        )}
                    </div>
                    <DialogFooter>
                        <Button variant="outline" className="rounded-xl" onClick={() => setShowImport(false)}>
                            Close
                        </Button>
                        <Button
                            className="rounded-xl bg-primary text-primary-foreground"
                            onClick={runImport}
                            disabled={!importFile || importing}
                        >
                            {importing ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Import'}
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </div>
    );
};

export default UsersSection;
