import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import {
    Users, Shield, Activity, UserPlus, Edit,
    AlertCircle, Loader2, Check, X
} from 'lucide-react';

import { API_BASE_URL, API_ENDPOINTS } from '@/config/constants';

// UserForm types
interface UserFormData {
    username: string;
    password: string;
    role: string;
    email: string;
}

interface UserFormProps {
    userForm: UserFormData;
    setUserForm: React.Dispatch<React.SetStateAction<UserFormData>>;
    onSubmit: () => void;
    title: string;
    submitText: string;
}

// UserForm component defined OUTSIDE UserManagement to prevent re-creation on parent re-renders
const UserForm: React.FC<UserFormProps> = ({ userForm, setUserForm, onSubmit, title, submitText }) => (
    <div className="space-y-4 text-secondary">
        <div className="space-y-2">
            <Label htmlFor="username">Username</Label>
            <Input
                id="username"
                value={userForm.username}
                onChange={(e) => setUserForm(prev => ({ ...prev, username: e.target.value }))}
            />
        </div>
        <div className="space-y-2">
            <Label htmlFor="email">Email</Label>
            <Input
                id="email"
                type="email"
                value={userForm.email}
                onChange={(e) => setUserForm(prev => ({ ...prev, email: e.target.value }))}
            />
        </div>
        <div className="space-y-2">
            <Label htmlFor="password">{title === 'Edit User' ? 'New Password (optional)' : 'Password'}</Label>
            <Input
                id="password"
                type="password"
                value={userForm.password}
                onChange={(e) => setUserForm(prev => ({ ...prev, password: e.target.value }))}
            />
        </div>
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
                    <SelectItem value="admin">Admin</SelectItem>
                    <SelectItem value="user">User</SelectItem>
                </SelectContent>
            </Select>
        </div>
        <Button className="w-full bg-primary text-primary-foreground" onClick={onSubmit}>{submitText}</Button>
    </div>
);

const UserRow = ({ user, onEdit, onToggleStatus }) => (
    <div className="p-4 border border-primary/20 rounded-lg bg-secondary mb-2 hover:bg-secondary/80">
        <div className="flex items-center justify-between">
            <div className="flex flex-col">
                <div className="flex items-center gap-2">
                    {user.role === 'admin' ? (
                        <Shield className="h-4 w-4" />
                    ) : (
                        <Users className="h-4 w-4" />
                    )}
                    <span className="font-medium">{user.username}</span>
                </div>
                <span className="text-sm">{user.email}</span>
            </div>
            <div className="flex items-center gap-2">
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
                    onClick={() => onEdit(user)}
                >
                    <Edit className="h-4 w-4" />
                </Button>
                <Button
                    variant="ghost"
                    size="sm"
                    className="h-8 w-8 p-0"
                    onClick={() => onToggleStatus(user.id, user.status)}
                >
                    {user.status === 'active' ? (
                        <X className="h-4 w-4 text-red-500" />
                    ) : (
                        <Check className="h-4 w-4 text-green-500" />
                    )}
                </Button>
            </div>
        </div>
        <div className="mt-2 text-sm">
            Last active: {new Date(user.lastActive).toLocaleString()}
        </div>
    </div>
);

const UserManagement = () => {
    const [users, setUsers] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [showAddUser, setShowAddUser] = useState(false);
    const [showEditUser, setShowEditUser] = useState(false);
    const [selectedUser, setSelectedUser] = useState(null);
    const [userForm, setUserForm] = useState({
        username: '',
        password: '',
        role: 'user',
        email: ''
    });
    const [sessionId, setSessionId] = useState(null);

    useEffect(() => {
        // Get session ID from localStorage
        const adminSession = localStorage.getItem('adminSession');
        if (adminSession) {
            setSessionId(adminSession);
        }
    }, []);

    useEffect(() => {
        if (sessionId) {
            fetchUsers();
        }
    }, [sessionId]);

    const fetchUsers = async () => {
        if (!sessionId) return;

        setLoading(true);
        try {
            const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.admin.users(sessionId)}`);
            if (response.ok) {
                const data = await response.json();
                setUsers(data);
            } else {
                throw new Error('Failed to fetch users');
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleAddUser = async () => {
        if (!sessionId) return;

        try {
            const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.admin.users(sessionId)}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(userForm),
            });

            if (response.ok) {
                setShowAddUser(false);
                setUserForm({ username: '', password: '', role: 'user', email: '' });
                fetchUsers();
            } else {
                const data = await response.json();
                // FastAPI validation errors come in detail field, sometimes as array
                let errorMsg = 'Failed to add user';
                if (data.detail) {
                    if (Array.isArray(data.detail)) {
                        // Pydantic validation errors: [{loc: [...], msg: "...", type: "..."}]
                        errorMsg = data.detail.map((err: { msg: string; loc?: string[] }) =>
                            err.loc ? `${err.loc.join('.')}: ${err.msg}` : err.msg
                        ).join(', ');
                    } else {
                        errorMsg = data.detail;
                    }
                }
                throw new Error(errorMsg);
            }
        } catch (err) {
            setError(err.message);
        }
    };

    const handleEditUser = async () => {
        if (!sessionId) return;

        try {
            const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.admin.userById(selectedUser.id, sessionId)}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(userForm),
            });

            if (response.ok) {
                setShowEditUser(false);
                setSelectedUser(null);
                setUserForm({ username: '', password: '', role: 'user', email: '' });
                fetchUsers();
            } else {
                const data = await response.json();
                let errorMsg = 'Failed to update user';
                if (data.detail) {
                    if (Array.isArray(data.detail)) {
                        errorMsg = data.detail.map((err: { msg: string; loc?: string[] }) =>
                            err.loc ? `${err.loc.join('.')}: ${err.msg}` : err.msg
                        ).join(', ');
                    } else {
                        errorMsg = data.detail;
                    }
                }
                throw new Error(errorMsg);
            }
        } catch (err) {
            setError(err.message);
        }
    };

    const handleToggleUserStatus = async (userId, currentStatus) => {
        if (!sessionId) return;

        try {
            const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.admin.userStatus(userId, sessionId)}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    status: currentStatus === 'active' ? 'inactive' : 'active'
                }),
            });

            if (response.ok) {
                fetchUsers();
            } else {
                throw new Error('Failed to update user status');
            }
        } catch (err) {
            setError(err.message);
        }
    };

    const startEditUser = (user) => {
        setSelectedUser(user);
        setUserForm({
            username: user.username,
            role: user.role,
            email: user.email,
            password: ''
        });
        setShowEditUser(true);
    };

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <h3 className="text-lg font-medium text-primary">User Management</h3>
                <Dialog open={showAddUser} onOpenChange={setShowAddUser}>
                    <DialogTrigger asChild>
                        <Button className="rounded-xl bg-primary text-primary-foreground">
                            <UserPlus className="h-4 w-4 mr-2" />
                            Add New User
                        </Button>
                    </DialogTrigger>
                    <DialogContent>
                        <DialogHeader>
                            <DialogTitle>Add New User</DialogTitle>
                        </DialogHeader>
                        <UserForm userForm={userForm} setUserForm={setUserForm} onSubmit={handleAddUser} title="Add User" submitText="Add User" />
                    </DialogContent>
                </Dialog>
            </div>

            {/* Statistics Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Card className="rounded-xl border border-primary/20">
                    <CardContent className="p-4">
                        <div className="flex items-center gap-3">
                            <Users className="h-5 w-5 text-primary" />
                            <div className='text-secondary'>
                                <p className="text-sm font-medium">Total Users</p>
                                <p className="text-2xl font-bold">{users.length}</p>
                            </div>
                        </div>
                    </CardContent>
                </Card>

                <Card className="rounded-xl border border-primary/20">
                    <CardContent className="p-4">
                        <div className="flex items-center gap-3">
                            <Shield className="h-5 w-5 text-primary" />
                            <div className='text-secondary'>
                                <p className="text-sm font-medium">Admin Users</p>
                                <p className="text-2xl font-bold">
                                    {users.filter(user => user.role === 'admin').length}
                                </p>
                            </div>
                        </div>
                    </CardContent>
                </Card>

                <Card className="rounded-xl border border-primary/20">
                    <CardContent className="p-4">
                        <div className="flex items-center gap-3">
                            <Activity className="h-5 w-5 text-primary" />
                            <div className='text-secondary'>
                                <p className="text-sm font-medium">Active Users</p>
                                <p className="text-2xl font-bold">
                                    {users.filter(user => user.status === 'active').length}
                                </p>
                            </div>
                        </div>
                    </CardContent>
                </Card>
            </div>

            {/* User List */}
            <Card className="rounded-xl border border-primary/20">
                <CardHeader className='text-primary'>
                    <CardTitle>User List</CardTitle>
                    <CardDescription>Manage system users and their permissions</CardDescription>
                </CardHeader>
                <CardContent>
                    <ScrollArea className="h-[500px] pr-4">
                        {loading ? (
                            <div className="flex items-center justify-center p-4">
                                <Loader2 className="h-6 w-6 animate-spin" />
                            </div>
                        ) : (
                            <div className="space-y-2">
                                {users.map((user) => (
                                    <UserRow
                                        key={user.id}
                                        user={user}
                                        onEdit={startEditUser}
                                        onToggleStatus={handleToggleUserStatus}
                                    />
                                ))}
                            </div>
                        )}
                    </ScrollArea>
                </CardContent>
            </Card>

            {/* Edit User Dialog */}
            <Dialog open={showEditUser} onOpenChange={setShowEditUser}>
                <DialogContent>
                    <DialogHeader className='text-primary'>
                        <DialogTitle>Edit User</DialogTitle>
                    </DialogHeader>
                    <UserForm userForm={userForm} setUserForm={setUserForm} onSubmit={handleEditUser} title="Edit User" submitText="Save Changes" />
                </DialogContent>
            </Dialog>

            {/* Error Alert */}
            {error && (
                <Alert variant="destructive" className="rounded-xl">
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                </Alert>
            )}
        </div>
    );
};

export default UserManagement;