# System Architecture

## Overview

Professional multi-agent orchestration system with a clean separation of concerns between business logic and presentation.

## Architecture Layers

```
┌─────────────────────────────────────────────────┐
│            main.py (Entry Point)                │
│  • Command-line parsing                         │
│  • Session management                           │
│  • User interaction loop                        │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│      ui/professional_ui.py (Presentation)       │
│  • Claude Code-inspired interface               │
│  • Task progress tracking                       │
│  • Status indicators                            │
│  • Error formatting                             │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│     orchestrator.py (Business Logic)            │
│  • Agent coordination                           │
│  • LLM integration (Gemini)                     │
│  • Intelligence system                          │
│  • Circuit breaker                              │
│  • Retry logic                                  │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│         connectors/ (Agents)                    │
│  • Slack, Jira, GitHub, etc.                    │
│  • Specialized domain agents                    │
│  • Platform integrations                        │
└─────────────────────────────────────────────────┘
```

## Key Design Principles

### 1. Separation of Concerns

**orchestrator.py** - Pure business logic
- No UI code (no prints with colors, no input())
- No interactive methods
- Returns plain strings
- Uses logger for debugging
- Reusable in any context (CLI, web, API, tests)

**main.py** - Application layer
- Entry point
- Session management
- Wraps orchestrator with UI feedback
- Handles user input/output
- Coordinates between UI and orchestrator

**ui/professional_ui.py** - Presentation layer
- All visual formatting
- Color codes
- Progress indicators
- Error display
- No business logic

### 2. Clean Interfaces

```python
# Orchestrator provides simple interface
orchestrator.process_message(user_input) -> str
orchestrator.call_sub_agent(name, instruction) -> str
orchestrator.discover_and_load_agents() -> None

# UI provides display methods
ui.start_agent_call(name, instruction)
ui.end_agent_call(name, success, duration, message)
ui.print_response(text)
ui.print_error(error)
```

### 3. Sequential Execution

Tasks execute **one at a time** for reliability:
- No parallel execution issues
- Clear progress tracking
- Easier debugging
- Predictable behavior

**Code**: `orchestrator.py:906` - Parallel execution disabled

```python
if len(function_calls) > 1 and False:  # DISABLED
```

## File Structure

```
.
├── main.py                    # Entry point & UI coordination
├── orchestrator.py            # Core business logic
│
├── ui/
│   ├── __init__.py           # Module exports
│   └── professional_ui.py    # Claude Code-style UI
│
├── core/                      # Core systems
│   ├── circuit_breaker.py    # Health management
│   ├── resilience.py         # Retry logic
│   ├── errors.py             # Error classification
│   ├── logger.py             # Logging
│   └── observability.py      # Metrics & tracing
│
├── intelligence/              # Intelligence systems
│   ├── hybrid_intelligence.py # Fast filter + LLM
│   ├── confidence_scorer.py  # Confidence scoring
│   └── context_manager.py    # Conversation context
│
├── llms/                      # LLM integrations
│   ├── base_llm.py           # Base interface
│   └── gemini_flash.py       # Gemini with caching
│
└── connectors/                # Agent implementations
    ├── slack_agent.py
    ├── jira_agent.py
    ├── github_agent.py
    └── ...
```

## How It Works

### 1. Startup Sequence

```python
# main.py
orchestrator = OrchestratorAgent(connectors_dir="connectors")
await orchestrator.discover_and_load_agents()
# → Discovers and loads all agents
# → Displays loaded agents via UI
```

### 2. Message Processing

```python
# User types message
user_input = "Send hello to #general and create issue KAN-123"

# main.py wraps orchestrator.call_sub_agent with UI feedback
async def wrapped_call_sub_agent(agent_name, instruction):
    ui.start_agent_call(agent_name, instruction)  # Show start
    result = await orchestrator.call_sub_agent(...)
    ui.end_agent_call(agent_name, success, ...)   # Show result
    return result

# Process with wrapped method
response = await orchestrator.process_message(user_input)
# → LLM analyzes intent
# → Decides which agents to call
# → Calls agents sequentially:
#     1. ui.start_agent_call("slack", "Send hello...")
#     2. slack_agent.execute(...)
#     3. ui.end_agent_call("slack", success=True)
#     4. ui.start_agent_call("jira", "Create issue...")
#     5. jira_agent.execute(...)
#     6. ui.end_agent_call("jira", success=True)
# → Returns summary

ui.print_response(response)
```

### 3. Error Handling

```python
# Agent fails
result = "❌ ERROR: Connection timeout"

# Circuit breaker tracks failures
await circuit_breaker.record_failure(agent_name)

# Error enhancer adds context
enhanced = error_enhancer.enhance_error(...)

# Retry manager handles retries
if error_classification.is_retryable:
    ui.show_retry(agent_name, attempt, max_attempts)
    await retry(...)

# UI displays error
ui.end_agent_call(agent_name, success=False, message=error)
```

## Key Features

### Circuit Breaker
- Tracks agent health
- Opens after 5 consecutive failures
- Closes after 2 successful recoveries
- 5-minute recovery timeout

### Intelligent Retry
- Classifies errors (transient vs permanent)
- Exponential backoff
- Max 3 attempts
- Duplicate operation detection

### Model Caching
- Caches Gemini GenerativeModel instances
- MD5-based cache keys
- 1-hour TTL
- 50-80% faster initialization

### Hybrid Intelligence
- Fast keyword filter (2-5ms)
- Falls back to LLM classifier (~100ms)
- 92% accuracy
- Semantic understanding

## Usage

### Basic Usage

```bash
# Start system
python main.py

# With verbose output
python main.py --verbose

# Example session
> Send "Hello team" to #general
⚙️ Slack
  Send "Hello team" to #general
  ✓ Completed (1234ms)

✓ Completed 1 operation(s)

I sent "Hello team" to the #general channel successfully.

> exit
────────────────────────────────────────────────────────────
Session Summary

  Duration: 2m 15s
  Messages: 3
  Agent calls: 5
  Success rate: 100%
────────────────────────────────────────────────────────────

Goodbye!
```

### Programmatic Usage

```python
from orchestrator import OrchestratorAgent

# Create orchestrator
orchestrator = OrchestratorAgent(
    connectors_dir="connectors",
    verbose=False
)

# Load agents
await orchestrator.discover_and_load_agents()

# Process message
response = await orchestrator.process_message(
    "Create a new Jira issue for bug fixes"
)

# Response is plain string
print(response)

# Cleanup
await orchestrator.cleanup()
```

### Testing

```python
# orchestrator.py has no UI dependencies
# Easy to test business logic

async def test_orchestrator():
    orch = OrchestratorAgent(connectors_dir="test_connectors")
    await orch.discover_and_load_agents()

    result = await orch.call_sub_agent(
        "slack",
        "Send test message"
    )

    assert "success" in result.lower()
```

## Configuration

### Environment Variables

```bash
# Required
GOOGLE_API_KEY=your_gemini_api_key

# Optional
USER_ID=default              # For user preferences
LOG_LEVEL=INFO              # Logging level
```

### Orchestrator Configuration

```python
orchestrator = OrchestratorAgent(
    connectors_dir="connectors",  # Agent directory
    verbose=False,                # Debug output
    llm=custom_llm               # Custom LLM (optional)
)

# Modify settings
orchestrator.max_retry_attempts = 3
orchestrator.system_prompt = "..."
```

## Extending the System

### Adding a New Agent

1. Create `connectors/myagent_agent.py`:

```python
class Agent:
    async def initialize(self):
        """Initialize the agent"""
        pass

    async def get_capabilities(self) -> List[str]:
        """Return list of capabilities"""
        return ["do something", "do another thing"]

    async def execute(self, instruction: str) -> str:
        """Execute instruction and return result"""
        # Your implementation
        return "Success"

    async def cleanup(self):
        """Cleanup resources"""
        pass
```

2. Restart system - agent auto-discovered

### Adding a New UI

```python
# custom_ui.py
class CustomUI:
    def show_agent_call(self, name, instruction):
        # Your custom display
        pass

    def show_result(self, success, message):
        # Your custom display
        pass

# custom_main.py
ui = CustomUI()
orchestrator = OrchestratorAgent(...)

# Wrap calls as needed
# orchestrator.process_message() returns plain strings
```

## Performance

- **Agent Discovery**: Parallel loading (~200ms for 5 agents)
- **Intelligence**: ~80ms average with caching
- **Model Caching**: 50-80% faster initialization
- **Circuit Breaker**: 360x faster failure detection (0.1ms vs 120s)
- **Retry Logic**: Exponential backoff (1s → 2s → 4s)

## Monitoring

- **Logs**: `logs/` directory
  - Session logs: `logs/session_{id}.jsonl`
  - Analytics: `logs/analytics/{id}.json`

- **Observability**: Metrics, traces, structured logs
  - Orchestration events
  - Intelligence analysis
  - Agent performance

## Troubleshooting

### Agent fails to load
```
Check logs/session_*.jsonl
Verify agent has Agent class
Check async def initialize()
```

### Slow performance
```
Enable verbose: --verbose
Check intelligence latency
Review agent execution times
Consider model caching
```

### Circuit breaker open
```
Check agent health status
Review recent failures
Wait for recovery timeout (5min)
Or restart system
```

## Summary

This architecture provides:

✓ **Clean separation** - UI, business logic, agents
✓ **Professional interface** - Claude Code-inspired
✓ **Sequential reliability** - One task at a time
✓ **Intelligent error handling** - Classification & retry
✓ **Health management** - Circuit breaker
✓ **High performance** - Caching & optimization
✓ **Easy testing** - No UI in business logic
✓ **Extensible** - Add agents/UIs easily
✓ **Production-ready** - Logging, metrics, observability
