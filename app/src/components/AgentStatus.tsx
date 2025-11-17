import React, { useState } from 'react';
import { ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';
import { Agent } from '../services/orchestratorService';

interface AgentStatusProps {
  agent: Agent;
}

const AgentStatus: React.FC<AgentStatusProps> = ({ agent }) => {
  const [expanded, setExpanded] = useState(false);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'status-healthy';
      case 'degraded':
        return 'status-degraded';
      case 'down':
        return 'status-down';
      default:
        return 'status-unknown';
    }
  };

  const formatAgentName = (name: string) => {
    return name
      .replace(/_/g, ' ')
      .split(' ')
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  return (
    <div className="agent-status">
      <button
        onClick={() => setExpanded(!expanded)}
        className="agent-status-header"
      >
        <div className="agent-status-left">
          <span className={`status-dot ${getStatusColor(agent.status)}`} />
          <span className="agent-name">{formatAgentName(agent.name)}</span>
        </div>
        {expanded ? (
          <ChevronUpIcon className="w-4 h-4" />
        ) : (
          <ChevronDownIcon className="w-4 h-4" />
        )}
      </button>

      {expanded && agent.capabilities.length > 0 && (
        <div className="agent-capabilities">
          <p className="capabilities-title">Capabilities:</p>
          <ul className="capabilities-list">
            {agent.capabilities.slice(0, 5).map((capability, idx) => (
              <li key={idx}>{capability}</li>
            ))}
            {agent.capabilities.length > 5 && (
              <li className="capabilities-more">
                +{agent.capabilities.length - 5} more
              </li>
            )}
          </ul>
        </div>
      )}
    </div>
  );
};

export default AgentStatus;
