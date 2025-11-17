import React from 'react';
import {
  PlusIcon,
  Cog6ToothIcon,
  XMarkIcon,
  ClockIcon,
} from '@heroicons/react/24/outline';
import AgentStatus from './AgentStatus';
import { Agent, Conversation } from '../services/orchestratorService';

interface SidebarProps {
  agents: Agent[];
  conversationHistory: Conversation[];
  onNewConversation: () => void;
  onLoadConversation: (conversation: Conversation) => void;
  onOpenSettings: () => void;
  onClose: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({
  agents,
  conversationHistory,
  onNewConversation,
  onLoadConversation,
  onOpenSettings,
  onClose,
}) => {
  return (
    <div className="sidebar">
      {/* Header */}
      <div className="sidebar-header">
        <h2 className="sidebar-title">Aerius</h2>
        <button
          onClick={onClose}
          className="icon-button"
          aria-label="Close sidebar"
        >
          <XMarkIcon className="w-5 h-5" />
        </button>
      </div>

      {/* New Conversation Button */}
      <button onClick={onNewConversation} className="new-conversation-button">
        <PlusIcon className="w-5 h-5" />
        <span>New Conversation</span>
      </button>

      {/* Agents Section */}
      <div className="sidebar-section">
        <h3 className="sidebar-section-title">Active Agents</h3>
        <div className="agents-list">
          {agents.length === 0 ? (
            <p className="empty-text">No agents available</p>
          ) : (
            agents.map((agent) => (
              <AgentStatus key={agent.name} agent={agent} />
            ))
          )}
        </div>
      </div>

      {/* Conversation History */}
      <div className="sidebar-section flex-1 overflow-auto">
        <h3 className="sidebar-section-title">
          <ClockIcon className="w-4 h-4 inline mr-1" />
          Recent Conversations
        </h3>
        <div className="conversation-list">
          {conversationHistory.length === 0 ? (
            <p className="empty-text">No conversation history</p>
          ) : (
            conversationHistory.slice(0, 10).map((conversation) => (
              <button
                key={conversation.id}
                onClick={() => onLoadConversation(conversation)}
                className="conversation-item"
              >
                <span className="conversation-title">{conversation.title}</span>
                <span className="conversation-time">
                  {new Date(conversation.timestamp).toLocaleDateString()}
                </span>
              </button>
            ))
          )}
        </div>
      </div>

      {/* Settings Button */}
      <button onClick={onOpenSettings} className="settings-button">
        <Cog6ToothIcon className="w-5 h-5" />
        <span>Settings</span>
      </button>
    </div>
  );
};

export default Sidebar;
