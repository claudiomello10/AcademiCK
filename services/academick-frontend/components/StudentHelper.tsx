import React, { useState, useEffect } from 'react';

import { API_BASE_URL, API_ENDPOINTS } from '@/config/constants';

import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import Image from 'next/image';
import { AlertCircle, Send, Loader2, LogOut, Book, MessageSquare, Plus, ChevronDown, History, X, Trash2 } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import 'katex/dist/katex.min.css';

// Define interfaces for book data
interface BookChapter {
    name: string;
    chapters: string[];
}

interface BookMention {
    show: boolean;
    position: { top: number; left: number; } | null;
    query: string;
}

interface Conversation {
    id: string;
    session_id: string | null;
    subject: string | null;
    title: string;
    message_count: number;
    created_at: string | null;
    updated_at: string | null;
}

interface MessageProps {
    message: string;
    isAI: boolean;
}

const Message = ({ message, isAI }: MessageProps) => (
    <div className={`flex gap-4 ${isAI ? 'justify-start' : 'justify-end'} mb-4`}>
        {isAI && (
            <Avatar className="rounded-lg shadow-sm flex-shrink-0 mt-1">
                <AvatarImage src="/app_icon_reduced.png" alt="AI" className="rounded-lg" />
                <AvatarFallback className="rounded-lg">AI</AvatarFallback>
            </Avatar>
        )}
        <div className={`flex flex-col ${isAI ? 'items-start' : 'items-end'} max-w-[80%]`}>
            <span className={`text-sm mb-1 ${isAI ? 'text-primary' : 'text-primary'}`}>
                {isAI ? 'AcademiCK' : 'Você'}
            </span>
            <div className={`rounded-3xl p-3 shadow-sm border ${isAI ? 'bg-secondary rounded-tl-none border-primary border-2' : 'bg-primary text-primary-foreground rounded-tr-none border-primary'} max-w-full`}>
                {isAI ? (
                    <ReactMarkdown
                        className="prose prose-base max-w-none text-foreground [&>p]:mb-2 [&>p:last-child]:mb-0"
                        remarkPlugins={[remarkGfm, remarkMath]}
                        rehypePlugins={[rehypeKatex]}
                        components={{
                            code({ className, children, ...props }) {
                                const match = /language-(\w+)/.exec(className || '');
                                return match ? (
                                    <div className="mt-2 mb-2">
                                        <SyntaxHighlighter
                                            style={oneDark}
                                            language={match[1]}
                                            PreTag="div"
                                            customStyle={{ borderRadius: '12px', margin: '0' }}
                                        >
                                            {String(children).replace(/\n$/, '')}
                                        </SyntaxHighlighter>
                                    </div>
                                ) : (
                                    <code className={`${className} rounded-lg px-1`} {...props}>
                                        {children}
                                    </code>
                                );
                            }
                        }}
                    >
                        {message}
                    </ReactMarkdown>
                ) : (
                    <div className="text-base whitespace-pre-wrap break-words">
                        {/* Highlight book references in user messages */}
                        {message.split(/<Book>(.+?)<\/Book>/).map((part, i) =>
                            i % 2 === 1 ? (
                                <span key={i} className="font-semibold bg-primary-foreground/20 px-1 rounded">
                                    {part}
                                </span>
                            ) : (
                                part
                            )
                        )}
                    </div>
                )}
            </div>
        </div>
        {!isAI && (
            <Avatar className="h-8 w-8 rounded-full shadow-sm flex-shrink-0 mt-1">
                <AvatarImage src="/reading-book.png" alt="Você" className="rounded-full" />
                <AvatarFallback className="rounded-full">Você</AvatarFallback>
            </Avatar>
        )}
    </div>
);

interface LoginFormProps {
    onLogin: (e: React.FormEvent<HTMLFormElement>) => void;
    isLoading: boolean;
    error: string | null;
}

const LoginForm = ({ onLogin, isLoading, error }: LoginFormProps) => (
    <Card className="w-full max-w-md mx-auto mt-20 rounded-2xl shadow-lg border border-primary/30">
        <CardHeader className="flex flex-col items-center space-y-4">
            <Image
                src="/app_icon.png"
                alt="AcademiCK"
                width={200}
                height={0}
                style={{ height: 'auto' }}
            />
            <CardDescription className="text-center">
                Auxiliando seus estudos
            </CardDescription>
        </CardHeader>
        <CardContent>
            <form onSubmit={onLogin} className="space-y-6">
                <div className="space-y-4">
                    <Input
                        type="text"
                        name="username"
                        placeholder="Usuário"
                        required
                        className="rounded-xl h-12 text-primary"
                    />
                    <Input
                        type="password"
                        name="password"
                        placeholder="Senha"
                        required
                        className="rounded-xl h-12 text-primary"
                    />
                </div>
                {error && (
                    <Alert variant="destructive" className="rounded-xl">
                        <AlertCircle className="h-4 w-4" />
                        <AlertTitle>Erro</AlertTitle>
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
                    ) : 'Entrar'}
                </Button>
            </form>
        </CardContent>
    </Card>
);

// Modified: Add useRef for textarea and adjust initial height
const StudentHelper = () => {
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [username, setUsername] = useState<string | null>(null);
    const [query, setQuery] = useState('');
    const [conversation, setConversation] = useState<{ text: string; isAI: boolean; }[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [model, setModel] = useState('gpt-5-mini');
    const [subject, setSubject] = useState(process.env.NEXT_PUBLIC_DEFAULT_SUBJECT || 'Machine Learning');
    const textareaRef = React.useRef<HTMLTextAreaElement>(null);

    // New state variables for book selection
    const [books, setBooks] = useState<BookChapter[]>([]);
    const [bookMention, setBookMention] = useState<BookMention>({
        show: false,
        position: null,
        query: ''
    });

    // Conversation history state
    const [conversations, setConversations] = useState<Conversation[]>([]);
    const [showConversations, setShowConversations] = useState(false);
    const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
    const [conversationTitle, setConversationTitle] = useState<string>('Nova Conversa');

    useEffect(() => {
        const savedSession = localStorage.getItem('session');
        const savedUsername = localStorage.getItem('username');
        if (savedSession && savedUsername) {
            fetch(`${API_BASE_URL}${API_ENDPOINTS.validateSession(savedSession)}`)
                .then(response => response.json())
                .then(data => {
                    if (data.valid) {
                        setSessionId(savedSession);
                        setUsername(savedUsername);
                        if (data.conversation_id) {
                            setCurrentConversationId(data.conversation_id);
                        }
                        loadChatHistory(savedSession);
                        loadBooks(); // Load books when session is valid
                        loadConversations(savedSession); // Load conversation history
                    } else {
                        handleLogout();
                        setError('Sessão expirada. Por favor, faça login novamente.');
                    }
                })
                .catch(() => {
                    handleLogout();
                    setError('Erro de conexão. Por favor, faça login novamente.');
                });
        }
    }, []);

    // New function to load books
    const loadBooks = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.books}`);
            if (response.ok) {
                const data = await response.json();
                // Transform response to match expected format
                const bookList = data.books?.map((book: { name: string; chapters: { title: string }[] }) => ({
                    name: book.name,
                    chapters: book.chapters?.map((c: { title: string }) => c.title) || []
                })) || [];
                setBooks(bookList);
            }
        } catch (error) {
            console.error("Error loading books:", error);
        }
    };

    // Load user's conversation history
    const loadConversations = async (sid: string) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/v1/conversations/${sid}`);
            if (response.ok) {
                const data = await response.json();
                setConversations(data.conversations || []);
            }
        } catch (error) {
            console.error("Error loading conversations:", error);
        }
    };

    // Resume a previous conversation
    const resumeConversation = async (conversationId: string) => {
        if (!sessionId) return;

        setIsLoading(true);
        try {
            const response = await fetch(
                `${API_BASE_URL}/api/v1/conversations/${sessionId}/resume/${conversationId}`
            );
            if (response.ok) {
                const data = await response.json();
                if (data.messages && data.messages.length > 0) {
                    setConversation(data.messages.map((msg: { role: string; content: string }) => ({
                        text: msg.content,
                        isAI: msg.role === 'assistant'
                    })));
                } else {
                    setConversation([{
                        text: "Bem-vindo ao AcademiCK!\n\nEstou aqui para ajudá-lo a estudar baseado nos materiais fornecidos.\n\nPosso auxiliar você a encontrar conteúdos específicos dentro do livro, explicar conceitos, fornecer detalhes de implementação e criar exemplos de código.\nFique à vontade para fazer qualquer pergunta sobre o conteúdo!",
                        isAI: true
                    }]);
                }
                setCurrentConversationId(data.conversation_id);
                setConversationTitle(data.title || 'Conversa');
                setShowConversations(false);
            } else {
                setError('Falha ao carregar conversa');
            }
        } catch (error) {
            console.error("Error resuming conversation:", error);
            setError('Erro ao carregar conversa');
        } finally {
            setIsLoading(false);
        }
    };

    // Create a new conversation
    const createNewConversation = async () => {
        if (!sessionId) return;

        setIsLoading(true);
        try {
            const response = await fetch(
                `${API_BASE_URL}/api/v1/conversations/${sessionId}/new`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ title: 'Nova Conversa' })
                }
            );
            if (response.ok) {
                const data = await response.json();
                setCurrentConversationId(data.conversation_id);
                setConversationTitle(data.title || 'Nova Conversa');
                setConversation([{
                    text: "Bem-vindo ao AcademiCK!\n\nEstou aqui para ajudá-lo a estudar baseado nos materiais fornecidos.\n\nPosso auxiliar você a encontrar conteúdos específicos dentro do livro, explicar conceitos, fornecer detalhes de implementação e criar exemplos de código.\nFique à vontade para fazer qualquer pergunta sobre o conteúdo!",
                    isAI: true
                }]);
                setShowConversations(false);
                // Refresh conversation list
                loadConversations(sessionId);
            } else {
                setError('Falha ao criar nova conversa');
            }
        } catch (error) {
            console.error("Error creating conversation:", error);
            setError('Erro ao criar nova conversa');
        } finally {
            setIsLoading(false);
        }
    };

    const deleteConversation = async (conversationId: string, e: React.MouseEvent) => {
        e.stopPropagation();
        if (!sessionId) return;

        if (currentConversationId === conversationId) {
            setError('Para excluir esta conversa, mude para outra primeiro.');
            setTimeout(() => setError(null), 3000);
            return;
        }

        try {
            const response = await fetch(
                `${API_BASE_URL}/api/v1/conversations/${sessionId}/${conversationId}`,
                { method: 'DELETE' }
            );
            if (response.ok) {
                loadConversations(sessionId);
            }
        } catch (error) {
            console.error("Error deleting conversation:", error);
        }
    };

    useEffect(() => {
        const scrollArea = document.getElementById('conversation-scroll');
        if (scrollArea) {
            scrollArea.scrollTop = scrollArea.scrollHeight;
        }
    }, [conversation]);

    // Add useEffect to set initial textarea height
    useEffect(() => {
        if (textareaRef.current) {
            // Set initial height to minHeight on first render
            textareaRef.current.style.height = '96px';
        }
    }, []);

    const loadChatHistory = async (sid: string): Promise<void> => {
        try {
            const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.chatHistory(sid)}`);
            if (response.ok) {
                const data = await response.json();
                if (!data.messages || data.messages.length === 0) {
                    setConversation([{
                        text: "Bem-vindo ao AcademiCK!\n\nEstou aqui para ajudá-lo a estudar baseado nos materiais fornecidos.\n\nPosso auxiliar você a encontrar conteúdos específicos dentro do livro, explicar conceitos, fornecer detalhes de implementação e criar exemplos de código.\nFique à vontade para fazer qualquer pergunta sobre o conteúdo!",
                        isAI: true
                    }]);
                } else {
                    setConversation(data.messages.map((msg: { role: string; content: string }) => ({
                        text: msg.content,
                        isAI: msg.role === 'assistant'
                    })));
                }
            }
        } catch {
            setError('Falha ao carregar histórico do chat.');
        }
    };

    interface LoginResponse {
        session_id: string;
        username: string;
        detail?: string;
    }

    const handleDeleteChat = async () => {
        if (!sessionId) return;

        setIsLoading(true);
        setError(null);

        try {
            const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.clearHistory(sessionId)}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                setConversation([{
                    text: "Bem-vindo ao AcademiCK!\n\nEstou aqui para ajudá-lo a estudar baseado nos materiais fornecidos.\n\nPosso auxiliar você a encontrar conteúdos específicos dentro do livro, explicar conceitos, fornecer detalhes de implementação e criar exemplos de código.\nFique à vontade para fazer qualquer pergunta sobre o conteúdo!",
                    isAI: true
                }]);
            } else {
                throw new Error('Falha ao deletar histórico do chat');
            }
        } catch (err: unknown) {
            setError(err instanceof Error ? err.message : 'Erro desconhecido');
        } finally {
            setIsLoading(false);
        }
    };

    const handleLogin = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        setIsLoading(true);
        setError(null);

        const formData = new FormData(e.currentTarget);
        const username = formData.get('username') as string;
        const password = formData.get('password') as string;

        try {
            const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.login}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ username, password })
            });

            const data: LoginResponse = await response.json();
            if (response.ok) {
                setSessionId(data.session_id);
                setUsername(data.username);
                localStorage.setItem('session', data.session_id);
                localStorage.setItem('username', data.username);

                // Set subject for the session
                await fetch(`${API_BASE_URL}${API_ENDPOINTS.setSubject(data.session_id)}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ subject })
                });

                // Fetch conversation_id from session validation
                const validateRes = await fetch(`${API_BASE_URL}${API_ENDPOINTS.validateSession(data.session_id)}`);
                const validateData = await validateRes.json();
                if (validateData.conversation_id) {
                    setCurrentConversationId(validateData.conversation_id);
                }

                loadChatHistory(data.session_id);
                loadBooks(); // Load books after successful login
                loadConversations(data.session_id); // Load conversation history
            } else {
                throw new Error(data.detail || 'Credenciais inválidas');
            }
        } catch (err: unknown) {
            setError(err instanceof Error ? err.message : 'Falha ao fazer login. Por favor, tente novamente.');
        } finally {
            setIsLoading(false);
        }
    };

    const handleLogout = () => {
        setSessionId(null);
        setUsername(null);
        setConversation([]);
        localStorage.removeItem('session');
        localStorage.removeItem('username');
    };

    interface GenerateResponseBody {
        query: string;
        model: string;
        subject: string; // Add subject to the request body
    }

    interface GenerateResponseData {
        response: string;
        detail?: string;
    }

    // Modified handleInputKeyDown to handle @ symbol
    const handleInputKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        // Handle Enter + Shift for newline
        if (e.key === 'Enter' && e.shiftKey) {
            e.preventDefault();

            // Insert newline at cursor position and force update
            const cursorPosition = e.currentTarget.selectionStart;
            const textBeforeCursor = query.substring(0, cursorPosition);
            const textAfterCursor = query.substring(cursorPosition);
            const newValue = textBeforeCursor + '\n' + textAfterCursor;

            setQuery(newValue);

            // Set cursor position after the newline in the next render cycle
            setTimeout(() => {
                const textarea = document.querySelector('textarea');
                if (textarea) {
                    textarea.selectionStart = textarea.selectionEnd = cursorPosition + 1;
                    // Manually trigger height adjustment
                    adjustTextareaHeight(textarea);
                }
            }, 0);
        }
        // Handle @ for book mention with improved positioning
        else if (e.key === '@') {
            const textarea = e.currentTarget;

            // We need to calculate caret position accurately
            // First, capture the current cursor position in the text
            const cursorPosition = textarea.selectionStart;

            // Create a range to get the caret coordinates
            const createCaretCoordinates = () => {
                // Temporarily create a span at the caret position to find its coordinates
                const tempValue = textarea.value;
                const cursorChar = '|';

                // Insert a special character at cursor position
                textarea.value =
                    tempValue.substring(0, cursorPosition) +
                    cursorChar +
                    tempValue.substring(cursorPosition);

                // Get the position of the textarea
                const textareaRect = textarea.getBoundingClientRect();

                // Create a range to find the position of our special character
                const range = document.createRange();
                const textNode = textarea.firstChild || textarea;

                // Find and select our special character
                const textContent = textarea.value;
                const cursorIndex = textContent.indexOf(cursorChar);

                if (cursorIndex >= 0) {
                    try {
                        range.setStart(textNode, cursorIndex);
                        range.setEnd(textNode, cursorIndex + 1);

                        // Get the bounding rectangle of the range
                        const rangeRect = range.getBoundingClientRect();

                        // Restore the original value
                        textarea.value = tempValue;

                        // Set selection back to original position
                        textarea.setSelectionRange(cursorPosition, cursorPosition);

                        // Return the calculated coordinates
                        return {
                            left: rangeRect.left - textareaRect.left + textarea.scrollLeft,
                            top: rangeRect.top - textareaRect.top + textarea.scrollTop
                        };
                    } catch {
                        // Fallback if range setting fails
                        textarea.value = tempValue;
                        textarea.setSelectionRange(cursorPosition, cursorPosition);
                    }
                } else {
                    // Restore if character not found
                    textarea.value = tempValue;
                    textarea.setSelectionRange(cursorPosition, cursorPosition);
                }

                // Fallback calculation
                return null;
            };

            // Get the textarea's position
            const textareaRect = textarea.getBoundingClientRect();

            // Try to get precise coordinates
            const caretCoords = createCaretCoordinates();

            let dropdownLeft, dropdownTop;

            if (caretCoords) {
                // Use the calculated coordinates if available
                dropdownLeft = textareaRect.left + caretCoords.left + 10;
                dropdownTop = textareaRect.top + caretCoords.top + 20;
            } else {
                // Fallback: position near the textarea
                dropdownLeft = textareaRect.left + 10;
                dropdownTop = textareaRect.top + 40;
            }

            // Set the book mention position
            setBookMention({
                show: true,
                position: { top: dropdownTop, left: dropdownLeft },
                query: ''
            });
        }
    }

    // Improved textarea height adjustment
    const adjustTextareaHeight = (textarea: HTMLTextAreaElement) => {
        // Set a small height first to get the proper scrollHeight
        textarea.style.height = '20px';

        // Limit height between min and max
        const minHeight = 96; // Minimum height in px
        const maxHeight = 300; // Maximum height in px before scrolling

        // Calculate new height based on content
        const newHeight = Math.max(minHeight, Math.min(textarea.scrollHeight, maxHeight));
        textarea.style.height = `${newHeight}px`;
    };

    // Use useEffect to adjust the textarea height on component mount and query changes
    useEffect(() => {
        const textarea = document.querySelector('textarea');
        if (textarea) {
            adjustTextareaHeight(textarea);
        }
    }, [query]);

    // Modified handleInputChange to handle bookMention
    const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        setQuery(e.target.value);
        adjustTextareaHeight(e.target);

        // Check for @ pattern and update book mention if active
        if (bookMention.show) {
            const cursorPosition = e.target.selectionStart;
            const textBeforeCursor = e.target.value.substring(0, cursorPosition);
            const lastAtPos = textBeforeCursor.lastIndexOf('@');

            if (lastAtPos !== -1) {
                const query = textBeforeCursor.substring(lastAtPos + 1);
                setBookMention(prev => ({
                    ...prev,
                    query
                }));
            } else {
                // If there's no @ before cursor, close the mention
                setBookMention({
                    show: false,
                    position: null,
                    query: ''
                });
            }
        }
    };

    const selectBook = (bookName: string) => {
        // Get cursor position
        if (!textareaRef.current) return;

        const textarea = textareaRef.current;
        const cursorPosition = textarea.selectionStart;
        const textBeforeCursor = query.substring(0, cursorPosition);
        const lastAtPos = textBeforeCursor.lastIndexOf('@');
        const textAfterCursor = query.substring(cursorPosition);

        if (lastAtPos !== -1) {
            // Replace the @mention with the book name
            const newText =
                textBeforeCursor.substring(0, lastAtPos) +
                `<Book>${bookName}</Book>` +
                textAfterCursor;

            setQuery(newText);

            // Close the mention dropdown
            setBookMention({
                show: false,
                position: null,
                query: ''
            });

            // Focus back on textarea and set cursor position after the inserted book name
            setTimeout(() => {
                textarea.focus();
                const newPosition = lastAtPos + bookName.length + 15; // 15 for <Book></Book> tags
                textarea.selectionStart = textarea.selectionEnd = newPosition;
                adjustTextareaHeight(textarea);
            }, 0);
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!query.trim() || !sessionId) return;

        const userQuery = query.trim();
        setQuery('');
        setIsLoading(true);
        setError(null);

        setConversation(prev => [...prev, { text: userQuery, isAI: false }]);

        try {
            const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.chat(sessionId)}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: userQuery,
                    model: model,
                    subject: subject // Include subject in the API call
                } as GenerateResponseBody)
            });

            const data: GenerateResponseData = await response.json();
            if (response.ok) {
                setConversation(prev => [...prev, { text: data.response, isAI: true }]);
            } else {
                throw new Error(data.detail || 'Falha ao obter resposta');
            }
        } catch (err: unknown) {
            setError(err instanceof Error ? err.message : 'Erro desconhecido');
        } finally {
            setIsLoading(false);
        }
    };

    // Update the subject when changed
    const handleSubjectChange = async (newSubject: string) => {
        setSubject(newSubject);
        if (sessionId) {
            try {
                await fetch(`${API_BASE_URL}${API_ENDPOINTS.setSubject(sessionId)}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ subject: newSubject })
                });
            } catch (err) {
                console.error("Failed to update subject:", err);
            }
        }
    };

    // Book mention dropdown component
    const BookMentionDropdown = () => {
        if (!bookMention.show || !bookMention.position) return null;

        const filteredBooks = books.filter(book =>
            bookMention.query ? book.name.toLowerCase().includes(bookMention.query.toLowerCase()) : true
        );

        if (filteredBooks.length === 0) {
            return null;
        }

        return (
            <div
                className="fixed z-50 bg-secondary rounded-md shadow-lg border p-1 max-h-60 overflow-auto"
                style={{
                    top: `${bookMention.position.top - 275}px`,
                    left: `${bookMention.position.left}px`,
                    maxWidth: '100%', // Ensure it doesn't leave the window
                }}
            >
                <div className="text-xs font-semibold text-foreground p-2">
                    Selecione um livro
                </div>
                {filteredBooks.map((book, idx) => (
                    <div
                        key={idx}
                        className="flex items-center gap-2 p-1 hover:bg-primary cursor-pointer rounded-sm"
                        onClick={() => selectBook(book.name)}
                    >
                        <Book className="h-5 w-5 text-foreground flex-shrink-0" />
                        <span className="text-sm truncate">{book.name}</span>
                    </div>
                ))}
            </div>
        );
    };

    // Conversation history sidebar component
    const ConversationHistorySidebar = () => {
        if (!showConversations) return null;

        return (
            <div className="fixed inset-0 z-50 flex">
                {/* Backdrop */}
                <div
                    className="absolute inset-0 bg-black/50"
                    onClick={() => setShowConversations(false)}
                />
                {/* Sidebar */}
                <div className="relative w-96 max-w-[85vw] bg-background h-full shadow-xl flex flex-col overflow-hidden">
                    <div className="flex items-center justify-between p-4 border-b">
                        <h2 className="text-lg font-semibold flex items-center gap-2 text-primary">
                            <History className="h-5 w-5 text-primary" />
                            Histórico de Conversas
                        </h2>
                        <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => setShowConversations(false)}
                            className="text-red-500 hover:text-red-700"
                        >
                            <X className="h-5 w-5" />
                        </Button>
                    </div>

                    <div className="p-3">
                        <Button
                            onClick={createNewConversation}
                            className="w-full rounded-xl bg-primary text-primary-foreground"
                            disabled={isLoading}
                        >
                            <Plus className="h-4 w-4 mr-2" />
                            Nova Conversa
                        </Button>
                    </div>

                    <ScrollArea className="flex-1 w-full">
                        {conversations.length === 0 ? (
                            <div className="text-center text-muted-foreground py-8 px-3">
                                Nenhuma conversa anterior
                            </div>
                        ) : (
                            <div className="space-y-2 pb-4 px-3 w-full">
                                {conversations.map((conv) => (
                                    <div
                                        key={conv.id}
                                        onClick={() => resumeConversation(conv.id)}
                                        className={`p-3 rounded-xl border cursor-pointer transition-colors hover:bg-secondary ${currentConversationId === conv.id ? 'border-primary bg-primary/10 ring-1 ring-primary' : 'border-border'
                                            }`}
                                    >
                                        <div className="flex items-center gap-2">
                                            <MessageSquare className="h-4 w-4 text-white flex-shrink-0" />
                                            <div className="flex-1 min-w-0">
                                                <p className="font-medium text-sm truncate text-primary">
                                                    {conv.title}
                                                </p>
                                                <p className="text-xs text-muted-foreground truncate">
                                                    {conv.message_count} mensagens
                                                </p>
                                                {conv.updated_at && (
                                                    <p className="text-xs text-muted-foreground truncate">
                                                        {new Date(conv.updated_at).toLocaleDateString('pt-BR', {
                                                            day: '2-digit',
                                                            month: 'short',
                                                            hour: '2-digit',
                                                            minute: '2-digit'
                                                        })}
                                                    </p>
                                                )}
                                            </div>
                                            {currentConversationId === conv.id ? (
                                                <span className="text-[10px] font-semibold text-primary bg-primary/20 px-2 py-0.5 rounded-full flex-shrink-0 whitespace-nowrap">
                                                    Ativa
                                                </span>
                                            ) : (
                                                <button
                                                    onClick={(e) => deleteConversation(conv.id, e)}
                                                    className="p-1 rounded-md hover:bg-red-500/20 transition-colors flex-shrink-0"
                                                    title="Excluir conversa"
                                                >
                                                    <Trash2 className="h-4 w-4 text-red-500" />
                                                </button>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </ScrollArea>
                </div>
            </div>
        );
    };

    if (!sessionId || !username) {
        return <LoginForm onLogin={handleLogin} isLoading={isLoading} error={error} />;
    }

    return (
        <div className="w-full h-screen max-w-6xl mx-auto flex flex-col">
            {/* Conversation history sidebar */}
            <ConversationHistorySidebar />

            <Card className="flex-1 flex flex-col rounded-2xl shadow-lg border border-primary/30">
                <CardHeader className="border-b rounded-t-2xl border-border p-3 sm:p-6">
                    <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                        <div className="flex-1 w-full">
                            <div className="flex items-center gap-4">
                                <CardTitle className="text-primary">AcademiCK</CardTitle>
                            </div>
                            <CardDescription className="mt-2">
                                Auxiliando seus estudos
                            </CardDescription>
                            <span className="text-xs text-muted-foreground mt-1">
                                Desenvolvido por Cláudio Klautau Mello
                            </span>
                            <div className="flex items-center gap-2 mt-2">
                                <span className="text-sm text-muted-foreground">Assunto:</span>
                                <Input
                                    type="text"
                                    value={subject}
                                    onChange={(e) => handleSubjectChange(e.target.value)}
                                    placeholder="Assunto"
                                    className="rounded-xl h-8 max-w-[180px] text-sm bg-secondary"
                                />
                            </div>
                        </div>

                        <div className="flex flex-row sm:flex-col items-center sm:items-end gap-2 sm:gap-3 w-full sm:w-auto">
                            <Button
                                variant="outline"
                                onClick={() => setShowConversations(true)}
                                className="rounded-xl hover:bg-secondary transition-colors duration-200 text-xs sm:text-sm flex-1 sm:flex-auto w-full sm:w-auto"
                            >
                                <History className="h-4 w-4 text-primary mr-1 sm:mr-2" />
                                <span className="text-primary">Histórico</span>
                            </Button>

                            <Button
                                variant="outline"
                                onClick={handleLogout}
                                className="rounded-xl hover:bg-red-50 transition-colors duration-200 text-xs sm:text-sm flex-1 sm:flex-auto w-full sm:w-auto"
                            >
                                <LogOut className="h-4 w-4 text-red-500 mr-1 sm:mr-2" />
                                <span className="text-red-500">Sair</span>
                            </Button>

                            <div className="text-right hidden sm:block">
                                <span className="text-sm text-muted-foreground block">Logado como:</span>
                                <span className="text-sm font-bold text-primary">{username}</span>
                            </div>
                        </div>
                    </div>

                    <div className="sm:hidden text-center mt-2">
                        <span className="text-xs text-muted-foreground">Logado como: </span>
                        <span className="text-xs font-bold text-primary">{username}</span>
                    </div>
                </CardHeader>
                <CardContent className="flex-1 flex flex-col p-0">
                    <ScrollArea className="flex-1 px-6 py-4" id="conversation-scroll">
                        <div className="space-y-1">
                            {conversation.map((msg, idx) => (
                                <Message key={idx} message={msg.text} isAI={msg.isAI} />
                            ))}
                            {isLoading && (
                                <div className="flex items-center gap-3 text-muted-foreground p-4">
                                    <Loader2 className="h-5 w-5 animate-spin" />
                                    <span>Pensando...</span>
                                </div>
                            )}
                        </div>
                    </ScrollArea>

                    {error && (
                        <div className="px-6 mb-4">
                            <Alert variant="destructive" className="rounded-xl">
                                <AlertCircle className="h-4 w-4" />
                                <AlertTitle>Erro</AlertTitle>
                                <AlertDescription>{error}</AlertDescription>
                            </Alert>
                        </div>
                    )}

                    <div className="border-t border-border p-4 pb-1 pr-2 pl-2">
                        <form onSubmit={(e) => e.preventDefault()} className="flex gap-3">
                            <div className="flex-1 relative">
                                <textarea
                                    ref={textareaRef}
                                    value={query}
                                    onChange={handleInputChange}
                                    onKeyDown={handleInputKeyDown}
                                    placeholder="Faça uma pergunta... (use @ para mencionar um livro)"
                                    disabled={isLoading}
                                    className="w-full rounded-xl p-3 resize-none border border-border bg-secondary focus:ring-0 focus:outline-none focus:border-primary"
                                    style={{
                                        height: '96px',
                                        minHeight: '96px',
                                        maxHeight: '300px',
                                        overflowY: textareaRef.current && textareaRef.current.scrollHeight > 300 ? 'auto' : 'hidden'
                                    }}
                                />
                                {/* Render the book mention dropdown */}
                                <BookMentionDropdown />
                            </div>
                            <div className="flex flex-col space-y-2" style={{ width: '110px', height: '96px' }}>
                                <Button
                                    type="button"
                                    onClick={(e) => handleSubmit(e)}
                                    disabled={isLoading || !query.trim()}
                                    className={`rounded-xl flex-1 h-[56px] w-full transition-all duration-200 ${query.trim()
                                        ? 'bg-primary text-primary-foreground hover:opacity-90'
                                        : 'bg-gray-300 text-gray-500 opacity-50 cursor-not-allowed'
                                        }`}
                                >
                                    {isLoading ? (
                                        <Loader2 className="h-5 w-5 animate-spin" />
                                    ) : (
                                        <Send className="h-5 w-5" />
                                    )}
                                </Button>
                                <select
                                    value={model}
                                    onChange={(e) => setModel(e.target.value)}
                                    className="rounded-lg px-2 py-1 text-xs border border-gray-400 h-[26px] w-full bg-secondary"
                                >
                                    {/* OpenAI Models */}
                                    <optgroup label="OpenAI">
                                        <option value="gpt-4.1-nano">GPT-4.1 Nano</option>
                                        <option value="gpt-4.1-mini">GPT-4.1 Mini</option>
                                        <option value="gpt-4.1">GPT-4.1</option>
                                        <option value="gpt-5-nano">GPT-5 Nano</option>
                                        <option value="gpt-5-mini">GPT-5 Mini</option>
                                        <option value="gpt-5.1">GPT-5.1</option>
                                    </optgroup>

                                    {/* Claude Models */}
                                    <optgroup label="Anthropic Claude">
                                        <option value="claude-haiku-4-5">Claude 4.5 Haiku</option>
                                        <option value="claude-sonnet-4-5">Claude 4.5 Sonnet</option>
                                        <option value="claude-opus-4-5">Claude 4.5 Opus</option>
                                    </optgroup>

                                    {/* DeepSeek Models */}
                                    <optgroup label="DeepSeek">
                                        <option value="deepseek-coder">DeepSeek Coder</option>
                                        <option value="deepseek-chat">DeepSeek Chat</option>
                                    </optgroup>
                                </select>
                            </div>
                        </form>
                    </div>
                </CardContent>
            </Card>
        </div >
    );
};

export default StudentHelper;