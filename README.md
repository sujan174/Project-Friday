# Project Friday - AI Workspace Orchestrator

A sophisticated AI-powered multi-agent system that coordinates specialized agents to interact with your workplace tools (Slack, Jira, GitHub, Notion, Google Calendar, and more).

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Agents](#agents)
- [Project Structure](#project-structure)
- [Core Concepts](#core-concepts)
- [Development Guide](#development-guide)
- [Troubleshooting](#troubleshooting)

---

## Overview

Project Friday is a **hub-and-spoke multi-agent orchestration system** that understands natural language commands and routes them to specialized AI agents. It uses a hybrid intelligence system combining fast keyword-based filtering with LLM-powered semantic understanding.

### Key Features

- **8 Specialized Agents**: Slack, Jira, GitHub, Notion, Google Calendar, Code Reviewer, Browser, Scraper
- **Hybrid Intelligence v5.0**: Two-tier decision making (fast filter + LLM)
- **Enterprise-Grade Reliability**: Circuit breaker, smart retries, error classification
- **Undo System**: Reversible operations for critical actions
- **Rich Terminal UI**: Beautiful interface powered by the Rich library
- **Session Logging**: JSONL-based logging for debugging and analytics

### Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.8+ |
| Async Runtime | asyncio |
| LLM Provider | Google Gemini Flash |
| Agent Protocol | Model Context Protocol (MCP) |
| Terminal UI | Rich library |
| Configuration | python-dotenv |
| Persistence | JSON files |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER INTERFACE                        │
│           (Enhanced Terminal UI with Rich)               │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                 ORCHESTRATOR AGENT                       │
│  - Hybrid Intelligence System v5.0 (Fast Filter + LLM)   │
│  - Circuit Breaker Pattern                              │
│  - Smart Retry Management                               │
│  - Undo System                                          │
└────┬────────────┬──────────────┬──────────────┬─────────┘
     │            │              │              │
  ┌──▼──┐    ┌────▼───┐    ┌────▼───┐    ┌────▼───┐
  │Slack│    │ Jira   │    │ GitHub  │    │ Notion │
  │Agent│    │ Agent  │    │ Agent   │    │ Agent  │
  └─────┘    └────────┘    └────────┘    └────────┘
     │            │              │              │
  ┌──▼──┐    ┌────▼───┐    ┌────▼───┐    ┌────▼────┐
  │Google│   │Code    │    │Browser  │    │Scraper  │
  │Cal   │   │Review  │    │Agent    │    │Agent    │
  └──────┘   └────────┘    └────────┘    └─────────┘
```

### Data Flow

1. **User Input** → Input validation and sanitization
2. **Hybrid Intelligence** → Fast filter (free, ~10ms) or LLM classifier (~200ms)
3. **Intent & Entity Extraction** → Understand what the user wants
4. **Agent Selection** → Route to appropriate specialized agent
5. **Execution** → Agent performs action via MCP protocol
6. **Response** → Formatted result returned to user

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/sujan174/Project-Friday.git
cd Project-Friday

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install google-generativeai python-dotenv rich mcp

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 5. Run the application
python main.py
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- API keys for services you want to use

### Required API Keys

| Service | Required For | How to Get |
|---------|-------------|------------|
| Google Gemini | Core AI functionality | [Google AI Studio](https://aistudio.google.com/app/apikey) |
| Notion | Notion agent | [Notion Integrations](https://www.notion.so/profile/integrations) |
| Google OAuth | Calendar agent | [Google Cloud Console](https://console.cloud.google.com) |

### MCP Server Dependencies

Agents communicate via Model Context Protocol (MCP). Install the required MCP servers:

```bash
# Slack MCP server
npx @anthropic/mcp-server-slack

# GitHub MCP server
npx @anthropic/mcp-server-github

# Jira MCP server
npx @anthropic/mcp-server-jira

# And others as needed...
```

---

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Required: Gemini API Key
GOOGLE_API_KEY=your_gemini_api_key_here

# For Notion agent
NOTION_TOKEN=your_notion_integration_token_here

# For Google Calendar agent
GOOGLE_CLIENT_ID=your_google_client_id_here
GOOGLE_CLIENT_SECRET=your_google_client_secret_here
```

### Configuration Categories

#### API Keys
- `GOOGLE_API_KEY` - Gemini API key (required)
- `NOTION_TOKEN` - Notion integration token
- `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` - Google OAuth credentials

#### Confirmation Settings
```bash
CONFIRM_SLACK_MESSAGES=true    # Preview before sending Slack messages
CONFIRM_JIRA_OPERATIONS=true   # Preview before Jira operations
CONFIRM_DELETES=true           # Confirm all delete operations
CONFIRM_BULK_OPERATIONS=true   # Confirm batch operations
```

#### Performance Tuning
```bash
AGENT_TIMEOUT=120.0            # Agent operation timeout (seconds)
LLM_TIMEOUT=30.0               # LLM call timeout (seconds)
MAX_RETRIES=3                  # Retry attempts for failed operations
RETRY_BACKOFF=2.0              # Exponential backoff multiplier
```

#### Logging
```bash
LOG_LEVEL=INFO                 # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_DIR=logs                   # Log directory path
ENABLE_FILE_LOGGING=true       # Write logs to files
ENABLE_JSON_LOGGING=true       # Machine-readable logs
VERBOSE=false                  # Detailed debug output
```

---

## Usage

### Starting the Application

```bash
# Standard mode with enhanced UI
python main.py

# Debug mode with verbose output
python main.py --verbose

# Simple UI fallback
python main.py --simple
```

### Example Commands

```
You: Send a message to #general saying "Hello team!"
You: Create a Jira issue in PROJECT for fixing the login bug
You: List all my GitHub pull requests
You: What meetings do I have tomorrow?
You: Find all Notion pages about onboarding
You: Review the code in my latest PR
```

### Session Viewer

Analyze past sessions:

```bash
python tools/session_viewer.py <session_id>
```

### Exiting

Type `exit`, `quit`, `bye`, or `q` to end the session.

---

## Agents

### Slack Agent
**File**: `connectors/slack_agent.py` (1763 LOC)

Operations:
- Send messages to channels/users
- Search messages
- List channels and users
- Manage channel membership

Features:
- Metadata caching (channels, users)
- Cache-first optimization for channel lists
- Smart message formatting

### Jira Agent
**File**: `connectors/jira_agent.py` (1708 LOC)

Operations:
- Create, update, delete issues
- Transition issue status
- Search with JQL
- Link issues
- Manage sprints

Features:
- Project and issue type caching
- Undo support for most operations
- Custom field handling

### GitHub Agent
**File**: `connectors/github_agent.py` (1673 LOC)

Operations:
- Create pull requests and issues
- Comment on PRs/issues
- Manage repositories
- Create branches
- Merge PRs

Features:
- Repository metadata caching
- Fast capability checking
- Undo support for close operations

### Notion Agent
**File**: `connectors/notion_agent.py` (1570 LOC)

Operations:
- Create and update pages
- Query databases
- Manage database properties
- Search content

Features:
- Database schema caching
- Rich property support
- Undo support for deletions

### Google Calendar Agent
**File**: `connectors/google_calendar_agent.py` (1463 LOC)

Operations:
- Create and update events
- List calendars
- Search events
- Manage attendees

Features:
- OAuth2 authentication
- Calendar metadata caching (30-min TTL)
- Manual auth support for headless environments

### Code Reviewer Agent
**File**: `connectors/code_reviewer_agent.py` (716 LOC)

Operations:
- Review code quality
- Suggest improvements
- Check for common issues

Features:
- AI-powered analysis using Gemini
- Pattern caching for common issues

### Browser Agent
**File**: `connectors/browser_agent.py` (791 LOC)

Operations:
- Navigate to URLs
- Fill forms
- Click elements
- Take screenshots

Features:
- Headless mode support
- Configuration caching

### Scraper Agent
**File**: `connectors/scraper_agent.py` (809 LOC)

Operations:
- Fetch web page content
- Extract structured data
- Handle different content types

Features:
- Charset detection
- Rate limiting
- Retry logic

---

## Project Structure

```
Project-Friday/
├── main.py                      # Entry point
├── orchestrator.py              # Main orchestration engine (1806 LOC)
├── config.py                    # Centralized configuration
├── .env.example                 # Configuration template
│
├── core/                        # Core infrastructure (17 files)
│   ├── logging_config.py        # Enhanced logging system
│   ├── simple_session_logger.py # Session logging
│   ├── error_handler.py         # Error classification
│   ├── retry_manager.py         # Smart retry logic
│   ├── circuit_breaker.py       # Circuit breaker pattern
│   ├── undo_manager.py          # Undo system
│   ├── analytics.py             # Metrics tracking
│   └── ...
│
├── connectors/                  # Agent implementations (12 files)
│   ├── base_agent.py            # Abstract base class
│   ├── agent_intelligence.py    # Shared intelligence
│   ├── slack_agent.py           # Slack integration
│   ├── jira_agent.py            # Jira integration
│   ├── github_agent.py          # GitHub integration
│   └── ...
│
├── intelligence/                # AI intelligence (12 files)
│   ├── base_types.py            # Data structures
│   ├── hybrid_system.py         # Hybrid Intelligence v5.0
│   ├── intent_classifier.py     # Intent classification
│   ├── entity_extractor.py      # Entity extraction
│   ├── task_decomposer.py       # Task decomposition
│   └── ...
│
├── llms/                        # LLM abstraction (2 files)
│   ├── base_llm.py              # Abstract LLM interface
│   └── gemini_flash.py          # Gemini implementation
│
├── ui/                          # User interface (2 files)
│   ├── enhanced_terminal_ui.py  # Rich-based UI
│   └── terminal_ui.py           # Simple fallback UI
│
├── tools/                       # Utilities
│   └── session_viewer.py        # Session log analyzer
│
└── docs/                        # Documentation
    └── *.md                     # Various guides
```

---

## Core Concepts

### Hybrid Intelligence System v5.0

The system uses a two-tier approach for understanding user intent:

1. **Tier 1 - Fast Keyword Filter** (~10ms, free)
   - Pattern-based classification
   - Handles clear, unambiguous requests
   - 35-40% of requests resolved here

2. **Tier 2 - LLM Classifier** (~200ms, paid)
   - Semantic understanding via Gemini
   - Handles complex, ambiguous requests
   - 60-65% of requests need this

### Error Classification

Errors are categorized for appropriate handling:

| Category | Description | Retry? |
|----------|-------------|--------|
| TRANSIENT | Temporary failures (network, timeout) | Yes |
| RATE_LIMIT | API rate limits exceeded | Yes (with backoff) |
| CAPABILITY | Agent can't perform action | No |
| PERMISSION | Authorization issues | No |
| VALIDATION | Invalid input | No |

### Circuit Breaker Pattern

Prevents cascading failures:

- **CLOSED**: Normal operation
- **OPEN**: Failures detected, reject requests
- **HALF_OPEN**: Testing recovery

### Metadata Caching

Agents cache workspace metadata to reduce API calls with advanced features:

#### Cache Configuration by Agent

| Agent | Cache Contents | TTL |
|-------|---------------|-----|
| **Slack** | Channels, users, channel info | 1 hour |
| **Jira** | Projects, issue types, transitions | 1 hour |
| **Jira** | Recent issues (per project) | 15 minutes |
| **GitHub** | Repositories, labels | 1 hour |
| **Notion** | Databases | 1 hour |
| **Calendar** | Calendars, time context | 30 minutes |
| **Calendar** | Free/busy availability | 15 minutes |

#### Cache Features

- **LRU Eviction**: Automatic eviction of oldest entries when max capacity reached
- **Cache-First Optimization**: Slack operations skip API calls for cached data
- **Hit/Miss Metrics**: Track cache performance per agent
- **Conditional Invalidation**: Auto-invalidate on write operations
- **Cache Warming**: Prefetch metadata on agent initialization

#### Cache Metrics

The system tracks cache hit/miss ratios for optimization:

```python
from connectors.agent_intelligence import get_cache_metrics

# Get cache statistics
metrics = get_cache_metrics()
print(metrics.get_stats('slack'))  # {'hits': 45, 'misses': 5, 'hit_rate': '90.0%'}
```

---

## Development Guide

### Adding a New Agent

1. Create `connectors/your_agent.py`:

```python
from connectors.base_agent import BaseAgent

class YourAgent(BaseAgent):
    AGENT_NAME = "your_service"
    DESCRIPTION = "Integration with Your Service"

    async def initialize(self):
        """Connect to external service"""
        # MCP server connection setup
        pass

    def get_capabilities(self) -> list:
        """Return list of capabilities"""
        return ["create_item", "update_item", "delete_item"]

    async def execute(self, instruction: str, context: dict = None) -> str:
        """Execute user instruction"""
        # Parse instruction and call MCP tools
        pass

    async def cleanup(self):
        """Clean up resources"""
        pass
```

2. The orchestrator auto-discovers agents from the `connectors/` directory

### Error Handling Pattern

```python
from core.error_handler import ErrorClassifier

try:
    result = await operation()
except Exception as e:
    classification = ErrorClassifier.classify(str(e), agent_name)
    if classification.is_retryable:
        # Use RetryManager for automatic retry
        pass
    else:
        # Return user-friendly error
        pass
```

### Testing

Currently, there are no automated tests. Testing is done via:

- Session logs (JSONL format)
- Session viewer tool
- Manual verification

To analyze a session:

```bash
python tools/session_viewer.py <session_id>
```

### Logging

Use structured logging throughout:

```python
from core.logger import get_logger

logger = get_logger(__name__)
logger.info("Operation completed", extra={
    "agent": "slack",
    "operation": "send_message",
    "duration_ms": 234
})
```

---

## Troubleshooting

### Common Issues

#### "No agents loaded"

- Check that MCP servers are installed and accessible
- Verify API keys are configured in `.env`
- Run with `--verbose` for detailed error output

#### "Agent timeout"

- Increase `AGENT_TIMEOUT` in `.env`
- Check network connectivity
- Verify external service is accessible

#### "Rate limit exceeded"

- The system will automatically retry with backoff
- Consider reducing request frequency
- Check API quota in service dashboard

#### "Authentication failed"

- Verify API keys are correct
- For Google Calendar, re-run OAuth flow
- Check token expiration

### Debug Mode

Run with verbose output:

```bash
python main.py --verbose
```

### Log Files

Check logs in the `logs/` directory:

- `session_<id>_messages.jsonl` - Message flow
- `session_<id>_intelligence.jsonl` - Intelligence metrics

### Getting Help

- Check existing documentation in `/docs`
- Review session logs for error details
- Use session viewer for debugging

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly (manual testing via sessions)
5. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for public methods
- Use async/await for all I/O operations

---

## License

[Add your license here]

---

## Acknowledgments

- Google Gemini for LLM capabilities
- Model Context Protocol (MCP) for agent communication
- Rich library for beautiful terminal UI
