"""
Connectors Package - Agent implementations for external services

Provides specialized agents for integrating with:
- Slack: Messaging and communication
- Jira: Issue tracking and project management
- GitHub: Code repositories and pull requests
- Notion: Knowledge base and databases
- Google Calendar: Scheduling and events
- Browser: Web automation
- Scraper: Web content extraction
- Code Reviewer: AI-powered code analysis

Author: AI System
Version: 1.0
"""

from .base_agent import BaseAgent
from .agent_intelligence import (
    ConversationMemory,
    WorkspaceKnowledge,
    SharedContext,
    ProactiveAssistant
)

__all__ = [
    'BaseAgent',
    'ConversationMemory',
    'WorkspaceKnowledge',
    'SharedContext',
    'ProactiveAssistant',
]

__version__ = '1.0.0'
