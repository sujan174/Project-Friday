import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { Message as MessageType } from '../services/orchestratorService';
import { formatDistanceToNow } from 'date-fns';

interface MessageProps {
  message: MessageType;
}

const Message: React.FC<MessageProps> = ({ message }) => {
  const { role, content, timestamp } = message;

  const formatTime = (ts: number) => {
    try {
      return formatDistanceToNow(new Date(ts), { addSuffix: true });
    } catch {
      return '';
    }
  };

  const getMessageClass = () => {
    switch (role) {
      case 'user':
        return 'message-user';
      case 'assistant':
        return 'message-assistant';
      case 'system':
        return 'message-system';
      default:
        return '';
    }
  };

  const getAvatar = () => {
    switch (role) {
      case 'user':
        return '👤';
      case 'assistant':
        return '🤖';
      case 'system':
        return 'ℹ️';
      default:
        return '';
    }
  };

  const getLabel = () => {
    switch (role) {
      case 'user':
        return 'You';
      case 'assistant':
        return 'Aerius';
      case 'system':
        return 'System';
      default:
        return '';
    }
  };

  return (
    <div className={`message ${getMessageClass()}`}>
      <div className="message-avatar">{getAvatar()}</div>
      <div className="message-content-wrapper">
        <div className="message-header">
          <span className="message-label">{getLabel()}</span>
          {timestamp && (
            <span className="message-time">{formatTime(timestamp)}</span>
          )}
        </div>
        <div className="message-content">
          {role === 'system' ? (
            <p>{content}</p>
          ) : (
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeHighlight]}
              components={{
                code({ node, inline, className, children, ...props }) {
                  return inline ? (
                    <code className={className} {...props}>
                      {children}
                    </code>
                  ) : (
                    <code className={className} {...props}>
                      {children}
                    </code>
                  );
                },
              }}
            >
              {content}
            </ReactMarkdown>
          )}
        </div>
      </div>
    </div>
  );
};

export default Message;
