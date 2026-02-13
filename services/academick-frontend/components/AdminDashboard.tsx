import React from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Users, Book, Activity } from 'lucide-react';

import ContentManagement from './ContentManagement';
import UserManagement from './UserManagement';
import UsageManagement from './UsageManagement';

const AdminDashboard = () => {
    return (
        <div className="w-full max-w-7xl mx-auto p-4">
            <Card className="rounded-2xl shadow-lg border border-primary/30">
                <CardHeader className="border-b border-border rounded-t-2xl">
                    <div className="flex items-center justify-between text-primary">
                        <div className="flex-1">
                            <CardTitle>Admin Dashboard</CardTitle>
                            <CardDescription className="mt-2">
                                Manage users, content, and system usage
                            </CardDescription>
                        </div>
                    </div>
                </CardHeader>

                <CardContent className="p-6">
                    <Tabs defaultValue="content" className="space-y-6">
                        <TabsList className="grid grid-cols-3 gap-4 bg-transparent h-auto p-0">
                            <TabsTrigger
                                value="content"
                                className="rounded-xl data-[state=active]:bg-primary data-[state=active]:text-primary-foreground border border-secondary data-[state=active]:border-primary"
                            >
                                <Book className="h-4 w-4 mr-2" />
                                Content
                            </TabsTrigger>
                            <TabsTrigger
                                value="users"
                                className="rounded-xl data-[state=active]:bg-primary data-[state=active]:text-primary-foreground border border-secondary data-[state=active]:border-primary"
                            >
                                <Users className="h-4 w-4 mr-2" />
                                Users
                            </TabsTrigger>
                            <TabsTrigger
                                value="usage"
                                className="rounded-xl data-[state=active]:bg-primary data-[state=active]:text-primary-foreground border border-secondary data-[state=active]:border-primary"
                            >
                                <Activity className="h-4 w-4 mr-2" />
                                Usage
                            </TabsTrigger>
                        </TabsList>

                        <TabsContent value="content" className="m-0">
                            <ContentManagement />
                        </TabsContent>

                        <TabsContent value="users" className="m-0">
                            <UserManagement />
                        </TabsContent>

                        <TabsContent value="usage" className="m-0">
                            <UsageManagement />
                        </TabsContent>
                    </Tabs>
                </CardContent>
            </Card>
        </div>
    );
};

export default AdminDashboard;