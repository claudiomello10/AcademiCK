import React, { useState, useEffect } from 'react';

// Configuração do servidor
import { API_BASE_URL } from '@/config/constants';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { AlertCircle, Send, Book, Loader2 } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';

// Markdown 
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';


interface MessageProps {
    message: string;
    isAI: boolean;
}

const Message = ({ message, isAI }: MessageProps) => (
    <div className={`flex gap-3 ${isAI ? 'bg-secondary/50' : ''} p-4 rounded-lg`}>
        <Avatar className="h-8 w-8">
            {isAI ? (
                <>
                    <AvatarImage src="/api/placeholder/32/32" alt="AI" />
                    <AvatarFallback>AI</AvatarFallback>
                </>
            ) : (
                <>
                    <AvatarImage src="/api/placeholder/32/32" alt="Você" />
                    <AvatarFallback>Você</AvatarFallback>
                </>
            )}
        </Avatar>
        <div className="flex-1">
            <p className="text-base font-medium mb-1">{isAI ? 'Assistente AI' : 'Você'}</p>
            <div className="text-base prose prose-base max-w-none">
                {isAI ? (
                    <ReactMarkdown
                        className="whitespace-pre-wrap"
                        remarkPlugins={[remarkGfm]}
                        components={{
                            code({ className, children, ...props }) {
                                const match = /language-(\w+)/.exec(className || '');
                                return match ? (
                                    <SyntaxHighlighter
                                        style={oneDark}
                                        language={match[1]}
                                        PreTag="div"
                                        customStyle={{ borderRadius: '12px', margin: '0' }}
                                    >
                                        {String(children).replace(/\n$/, '')}
                                    </SyntaxHighlighter>
                                ) : (
                                    <code className={className} {...props}>
                                        {children}
                                    </code>
                                )
                            }
                        }}
                    >
                        {message}
                    </ReactMarkdown>
                ) : (
                    <div className="whitespace-pre-wrap">{message}</div>
                )}
            </div>
        </div>
    </div>
);

const StudentHelper = () => {
    const [sessionId, setSessionId] = useState(null);
    const [subject, setSubject] = useState('');
    const [query, setQuery] = useState('');
    interface ConversationMessage {
        text: string;
        isAI: boolean;
    }

    const [conversation, setConversation] = useState<ConversationMessage[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>('');

    const subjects = [
        "Aprendizado de Máquina",
        "Ciência de Dados",
        "Visão Computacional",
        "Processamento de Linguagem Natural",
        "Aprendizado Profundo"
    ];

    useEffect(() => {
        createSession();
    }, []);

    // Auto-scroll para o final quando a conversa é atualizada
    useEffect(() => {
        const scrollArea = document.getElementById('conversation-scroll');
        if (scrollArea) {
            scrollArea.scrollTop = scrollArea.scrollHeight;
        }
    }, [conversation]);

    const createSession = async () => {
        try {
            const response = await fetch(`http://${API_BASE_URL}/create_session`, {
                method: 'POST'
            });
            const data = await response.json();
            setSessionId(data.session_id);
        } catch {
            setError('Falha ao criar sessão. Por favor, tente novamente.');
        }
    };

    const setStudySubject = async (selectedSubject: string) => {
        try {
            await fetch(`http://${API_BASE_URL}/set_subject/${sessionId}?subject=${encodeURIComponent(selectedSubject)}`, {
                method: 'POST'
            });
            setSubject(selectedSubject);
            // Adicionar mensagem de boas-vindas quando o assunto é selecionado
            setConversation([{
                text: `Bem-vindo! Eu sou o AcademiCK seu assistente de estudos AI para ${selectedSubject}. Como posso ajudar você hoje?`,
                isAI: true
            }]);
        } catch {
            setError('Falha ao definir o assunto. Por favor, tente novamente.');
        }
    };

    const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        if (!query.trim() || !sessionId) return;

        const userQuery = query.trim();
        setQuery('');
        setIsLoading(true);
        setError(null);

        // Adicionar mensagem do usuário imediatamente
        setConversation(prev => [...prev, { text: userQuery, isAI: false }]);

        try {
            const response = await fetch(`http://${API_BASE_URL}/generate_response_conversation/${sessionId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: userQuery,
                    model: 'gpt-5-mini'
                })
            });

            const data = await response.json();
            if (response.ok) {
                setConversation(prev => [...prev, { text: data.response, isAI: true }]);
            } else {
                throw new Error(data.detail || 'Falha ao obter resposta');
            }
        } catch (err) {
            setError((err as Error).message);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="w-full h-screen max-w-6xl mx-auto flex flex-col">
            <Card className="flex-1 flex flex-col">
                <CardHeader className="border-b">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                            <Book className="h-6 w-6" />
                            <CardTitle></CardTitle>
                        </div>
                        <Select onValueChange={setStudySubject} value={subject}>
                            <SelectTrigger className="w-48">
                                <SelectValue placeholder="Escolha o assunto" />
                            </SelectTrigger>
                            <SelectContent>
                                {subjects.map((subj) => (
                                    <SelectItem key={subj} value={subj}>
                                        {subj}
                                    </SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                    </div>
                </CardHeader>
                <CardContent className="flex-1 flex flex-col p-0">
                    {/* Área de Conversação */}
                    <ScrollArea className="flex-1 p-4" id="conversation-scroll">
                        <div className="space-y-4">
                            {conversation.map((msg, idx) => (
                                <Message key={idx} message={msg.text} isAI={msg.isAI} />
                            ))}
                            {isLoading && (
                                <div className="flex items-center gap-2 text-muted-foreground p-4">
                                    <Loader2 className="h-4 w-4 animate-spin" />
                                    <span>Pensando...</span>
                                </div>
                            )}
                        </div>
                    </ScrollArea>

                    {/* Exibição de Erro */}
                    {error && (
                        <div className="px-4">
                            <Alert variant="destructive">
                                <AlertCircle className="h-4 w-4" />
                                <AlertTitle>Erro</AlertTitle>
                                <AlertDescription>{error}</AlertDescription>
                            </Alert>
                        </div>
                    )}

                    {/* Área de Entrada */}
                    <div className="border-t p-4">
                        <form onSubmit={handleSubmit} className="flex gap-2">
                            <Input
                                value={query}
                                onChange={(e) => setQuery(e.target.value)}
                                placeholder={subject ? "Pergunte qualquer coisa sobre o seu assunto..." : "Por favor, selecione um assunto primeiro"}
                                disabled={isLoading || !subject}
                                className="flex-1"
                            />
                            <Button type="submit" disabled={isLoading || !subject || !query.trim()}>
                                {isLoading ? (
                                    <Loader2 className="h-4 w-4 animate-spin" />
                                ) : (
                                    <Send className="h-4 w-4" />
                                )}
                            </Button>
                        </form>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
};

export default StudentHelper;