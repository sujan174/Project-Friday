import React, { useState, useEffect } from 'react';
import { Toaster } from 'react-hot-toast';
import ChatInterface from './components/ChatInterface';
import Sidebar from './components/Sidebar';
import Settings from './components/Settings';
import LoadingScreen from './components/LoadingScreen';
import orchestratorService, { Agent, Message, Conversation } from './services/orchestratorService';
import './styles/App.css';

function App() {
  const [isInitializing, setIsInitializing] = useState(true);
  const [isInitialized, setIsInitialized] = useState(false);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [sessionId, setSessionId] = useState<string>('');
  const [showSettings, setShowSettings] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [conversationHistory, setConversationHistory] = useState<Conversation[]>([]);

  useEffect(() => {
    // Initialize orchestrator on mount
    initializeOrchestrator();

    // Set up message listener
    const unsubscribe = orchestratorService.onMessage(handleBackendMessage);

    // Load conversation history
    loadHistory();

    return () => {
      unsubscribe();
    };
  }, []);

  const initializeOrchestrator = async () => {
    try {
      setIsInitializing(true);
      await orchestratorService.initialize();
    } catch (error) {
      console.error('Failed to initialize:', error);
      addSystemMessage('Failed to initialize orchestrator. Please check your configuration.');
    }
  };

  const loadHistory = async () => {
    const history = await orchestratorService.getHistory();
    setConversationHistory(history);
  };

  const handleBackendMessage = (data: any) => {
    switch (data.type) {
      case 'ready':
        console.log('Backend ready');
        break;

      case 'status':
        addSystemMessage(data.data.message);
        break;

      case 'initialized':
        setIsInitialized(true);
        setIsInitializing(false);
        setAgents(data.data.agents);
        setSessionId(data.data.session_id);
        addSystemMessage('Aerius initialized successfully! How can I help you today?');
        break;

      case 'processing':
        // Add user message to chat
        addMessage('user', data.data.message);
        break;

      case 'response':
        // Add assistant response
        addMessage('assistant', data.data.text);
        break;

      case 'error':
        addSystemMessage(`Error: ${data.data.message}`, 'error');
        break;

      default:
        console.log('Unknown message type:', data.type);
    }
  };

  const addMessage = (role: 'user' | 'assistant' | 'system', content: string) => {
    const message: Message = {
      id: Date.now().toString() + Math.random(),
      role,
      content,
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev, message]);
  };

  const addSystemMessage = (content: string, type: 'info' | 'error' = 'info') => {
    addMessage('system', content);
  };

  const handleSendMessage = async (text: string) => {
    try {
      await orchestratorService.sendMessage(text);
    } catch (error) {
      addSystemMessage('Failed to send message. Please try again.', 'error');
    }
  };

  const handleNewConversation = async () => {
    // Save current conversation if it has messages
    if (messages.length > 0) {
      const conversation: Conversation = {
        id: Date.now().toString(),
        title: messages.find(m => m.role === 'user')?.content.slice(0, 50) || 'New Conversation',
        messages,
        timestamp: Date.now(),
      };
      await orchestratorService.saveConversation(conversation);
      await loadHistory();
    }

    // Reset session
    setMessages([]);
    setIsInitialized(false);
    setIsInitializing(true);
    await orchestratorService.resetSession();
  };

  const handleLoadConversation = (conversation: Conversation) => {
    setMessages(conversation.messages);
  };

  const handleExportConversation = async () => {
    const conversation: Conversation = {
      id: sessionId,
      title: messages.find(m => m.role === 'user')?.content.slice(0, 50) || 'Conversation',
      messages,
      timestamp: Date.now(),
    };
    const result = await orchestratorService.exportConversation(conversation);
    if (result.success) {
      addSystemMessage(`Conversation exported to ${result.path}`);
    }
  };

  return (
    <div className="app">
      <Toaster position="top-right" />

      {isInitializing && !isInitialized && <LoadingScreen />}

      {!isInitializing && (
        <>
          {sidebarOpen && (
            <Sidebar
              agents={agents}
              conversationHistory={conversationHistory}
              onNewConversation={handleNewConversation}
              onLoadConversation={handleLoadConversation}
              onOpenSettings={() => setShowSettings(true)}
              onClose={() => setSidebarOpen(false)}
            />
          )}

          <ChatInterface
            messages={messages}
            onSendMessage={handleSendMessage}
            onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
            onExport={handleExportConversation}
            sessionId={sessionId}
            sidebarOpen={sidebarOpen}
          />

          {showSettings && (
            <Settings onClose={() => setShowSettings(false)} />
          )}
        </>
      )}
    </div>
  );
}

export default App;
