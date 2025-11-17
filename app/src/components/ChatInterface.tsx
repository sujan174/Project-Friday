import React, { useState, useRef, useEffect } from 'react';
import { PaperAirplaneIcon, Bars3Icon, ArrowDownTrayIcon } from '@heroicons/react/24/outline';
import Message from './Message';
import { Message as MessageType } from '../services/orchestratorService';

interface ChatInterfaceProps {
  messages: MessageType[];
  onSendMessage: (message: string) => void;
  onToggleSidebar: () => void;
  onExport: () => void;
  sessionId: string;
  sidebarOpen: boolean;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
  messages,
  onSendMessage,
  onToggleSidebar,
  onExport,
  sessionId,
  sidebarOpen,
}) => {
  const [input, setInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.style.height = inputRef.current.scrollHeight + 'px';
    }
  }, [input]);

  // Track processing state
  useEffect(() => {
    const lastMessage = messages[messages.length - 1];
    if (lastMessage?.role === 'user') {
      setIsProcessing(true);
    } else if (lastMessage?.role === 'assistant') {
      setIsProcessing(false);
    }
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isProcessing) return;

    onSendMessage(input.trim());
    setInput('');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className={`chat-interface ${sidebarOpen ? 'with-sidebar' : ''}`}>
      {/* Header */}
      <div className="chat-header">
        <div className="chat-header-left">
          {!sidebarOpen && (
            <button
              onClick={onToggleSidebar}
              className="icon-button"
              aria-label="Open sidebar"
            >
              <Bars3Icon className="w-5 h-5" />
            </button>
          )}
          <div className="chat-title">
            <h1>Aerius</h1>
            {sessionId && (
              <span className="session-id">Session: {sessionId.slice(0, 8)}</span>
            )}
          </div>
        </div>
        <div className="chat-header-right">
          {messages.length > 0 && (
            <button
              onClick={onExport}
              className="icon-button"
              aria-label="Export conversation"
            >
              <ArrowDownTrayIcon className="w-5 h-5" />
            </button>
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="messages-container">
        {messages.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">✨</div>
            <h2>How can I help you today?</h2>
            <p>I can help you with tasks across Slack, GitHub, Jira, Notion, and more.</p>
            <div className="example-prompts">
              <button
                onClick={() => setInput('Show my latest GitHub issues')}
                className="example-prompt"
              >
                Show my latest GitHub issues
              </button>
              <button
                onClick={() => setInput('What are my Jira tasks?')}
                className="example-prompt"
              >
                What are my Jira tasks?
              </button>
              <button
                onClick={() => setInput('Send a message to #general on Slack')}
                className="example-prompt"
              >
                Send a message to Slack
              </button>
            </div>
          </div>
        ) : (
          <>
            {messages.map((message) => (
              <Message key={message.id} message={message} />
            ))}
            {isProcessing && (
              <div className="processing-indicator">
                <div className="typing-dots">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
                <span>Aerius is thinking...</span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Input */}
      <div className="input-container">
        <form onSubmit={handleSubmit} className="input-form">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask me anything..."
            rows={1}
            disabled={isProcessing}
            className="message-input"
          />
          <button
            type="submit"
            disabled={!input.trim() || isProcessing}
            className="send-button"
            aria-label="Send message"
          >
            <PaperAirplaneIcon className="w-5 h-5" />
          </button>
        </form>
        <div className="input-footer">
          <span className="input-hint">
            Press <kbd>Enter</kbd> to send, <kbd>Shift + Enter</kbd> for new line
          </span>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
