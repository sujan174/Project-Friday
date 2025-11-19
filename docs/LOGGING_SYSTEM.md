# Unified Session Logging System

## Overview

The Unified Session Logging System creates **exactly 2 files per chat session** to track all conversation activity and intelligence system status.

## Files Created Per Session

### 1. Message Exchange Log
**File**: `logs/session_{session_id}_messages.jsonl`

**Purpose**: Captures detailed message flow between user, orchestrator, and subagents.

**Format**: JSONL (JSON Lines) - one JSON object per line for streaming efficiency

**Content Structure**:
```json
{
  "timestamp": "2025-11-18T10:30:45.123Z",
  "type": "user_message|orchestrator_to_agent|agent_to_orchestrator|assistant_response",
  "from": "user|orchestrator|{agent_name}",
  "to": "orchestrator|{agent_name}|user",
  "content": "message content",
  "metadata": {
    "message_length": 150,
    "duration_ms": 234.5,
    "success": true,
    "error": null
  }
}
```

**Message Types**:
- `user_message`: User -> Orchestrator
- `orchestrator_to_agent`: Orchestrator -> Subagent (task delegation)
- `agent_to_orchestrator`: Subagent -> Orchestrator (task response)
- `assistant_response`: Orchestrator -> User (final response)

### 2. Intelligence Status Log
**File**: `logs/session_{session_id}_intelligence.jsonl`

**Purpose**: Tracks intelligence system processing and status after each conversation turn.

**Format**: JSONL (JSON Lines)

**Content Structure**:
```json
{
  "turn_number": 1,
  "timestamp": "2025-11-18T10:30:45.123Z",
  "user_message": "Create a Jira ticket",
  "intelligence": {
    "path_used": "fast|llm",
    "latency_ms": 12.5,
    "intents": ["create"],
    "entities": [
      {
        "type": "issue_type",
        "value": "ticket",
        "confidence": 0.95
      }
    ],
    "confidence": 0.92,
    "reasoning": "High-confidence keyword match",
    "ambiguities": [],
    "suggested_clarifications": []
  },
  "execution_summary": {
    "tasks": [
      {
        "id": "t1",
        "agent": "jira",
        "action": "create_ticket",
        "estimated_duration_ms": 2500
      }
    ],
    "total_estimated_duration_ms": 2500,
    "estimated_cost_tokens": 1000
  },
  "turn_complete": false
}
```

**Turn Execution Summary** (appended after turn completes):
```json
{
  "type": "turn_execution_summary",
  "data": {
    "turn_number": 1,
    "timestamp": "2025-11-18T10:30:46.456Z",
    "agents_called": ["jira"],
    "agent_count": 1,
    "total_duration_ms": 2354.0,
    "success": true,
    "errors": []
  }
}
```

## Intelligence System Components Tracked

### Hybrid Intelligence System v5.0
- **path_used**: Which tier was used ("fast" for keyword filter, "llm" for semantic analysis)
- **latency_ms**: Processing time in milliseconds
- **confidence**: Overall confidence score (0.0 - 1.0)

### Intent Classification
- **intents**: List of detected user intents (e.g., ["create", "search", "update"])

### Entity Extraction
- **entities**: List of extracted entities with type, value, and confidence
- Common entity types: `issue_type`, `agent_name`, `project`, `priority`, etc.

### Task Decomposition
- **execution_plan**: Breakdown of tasks to be executed
- **tasks**: Individual tasks with agent assignments
- **estimated_duration_ms**: Expected execution time
- **estimated_cost_tokens**: Estimated LLM token usage

### Context & Ambiguity Detection
- **ambiguities**: List of ambiguities detected in user input
- **suggested_clarifications**: Clarification questions to resolve ambiguities

## Usage

### Basic Logging
```python
from core.unified_session_logger import UnifiedSessionLogger

# Initialize logger
logger = UnifiedSessionLogger(session_id="unique-session-id", log_dir="logs")

# Log user message (starts a new turn)
logger.log_user_message("Create a Jira ticket for bug fix")

# Log intelligence processing
logger.log_intelligence_status(
    intelligence_result={...},
    execution_plan={...}
)

# Log orchestrator -> agent communication
logger.log_orchestrator_to_agent(
    agent_name="jira",
    instruction="Create ticket",
    context={}
)

# Log agent -> orchestrator response
logger.log_agent_to_orchestrator(
    agent_name="jira",
    response="Ticket created",
    success=True,
    duration_ms=234.5
)

# Log final assistant response
logger.log_assistant_response("I've created the ticket for you.")
```

### Reading Logs
```python
# Read all messages
messages = logger.read_messages()

# Read all intelligence entries
intelligence_log = logger.read_intelligence_log()

# Get session summary
summary = logger.get_session_summary()
```

## Benefits

### 1. Simplified File Structure
- **Before**: 6+ files per session (session.json, session.txt, orchestration_{timestamp}.json, intelligence_{timestamp}.json, agent logs, etc.)
- **After**: 2 files per session (messages.jsonl, intelligence.jsonl)

### 2. Complete Message Traceability
Every message exchange is logged:
- User -> Orchestrator
- Orchestrator -> Subagent
- Subagent -> Orchestrator
- Orchestrator -> User

### 3. Intelligence System Transparency
Full visibility into:
- Intent classification decisions
- Entity extraction results
- Confidence scoring
- Task decomposition
- Execution planning

### 4. Performance Analysis
Track:
- Intelligence processing latency
- Agent execution duration
- Turn-by-turn performance
- Success/failure rates

### 5. Debugging & Auditing
- Complete conversation history
- Intelligence reasoning traces
- Error tracking
- Agent call traces

## File Format: JSONL

**Why JSONL?**
- **Streaming-friendly**: Append new entries without re-writing entire file
- **Line-by-line processing**: Parse incrementally for large logs
- **Simple structure**: One JSON object per line
- **Tool support**: Many log analysis tools support JSONL

**Reading JSONL**:
```python
import json

with open('logs/session_123_messages.jsonl', 'r') as f:
    for line in f:
        entry = json.loads(line.strip())
        print(entry)
```

## Integration Points

### In Orchestrator (orchestrator.py)

**Initialization** (line ~161):
```python
self.unified_logger = UnifiedSessionLogger(
    session_id=self.session_id,
    log_dir="logs"
)
```

**User Message** (line ~1103):
```python
self.unified_logger.log_user_message(
    message=user_message,
    metadata={'message_length': len(user_message)}
)
```

**Intelligence Processing** (line ~1156):
```python
self.unified_logger.log_intelligence_status(
    intelligence_result={...},
    execution_plan={...}
)
```

**Agent Communication** (lines ~867, ~902):
```python
# Orchestrator -> Agent
self.unified_logger.log_orchestrator_to_agent(
    agent_name=agent_name,
    instruction=full_instruction,
    context=context
)

# Agent -> Orchestrator
self.unified_logger.log_agent_to_orchestrator(
    agent_name=agent_name,
    response=result,
    success=success,
    duration_ms=latency_ms,
    error=error
)
```

**Assistant Response** (line ~1113):
```python
self._log_and_return_response(response)
# Internally calls: self.unified_logger.log_assistant_response(...)
```

## Backward Compatibility

The old logging system (`SessionLogger`, `OrchestrationLogger`, `IntelligenceLogger`) is still active for backward compatibility. Both systems run in parallel during the transition period.

To disable old logging:
1. Remove `self.session_logger` initialization
2. Remove old logging calls
3. Remove old logger imports

## Example Session Logs

See `logs/session_demo-session-123_messages.jsonl` and `logs/session_demo-session-123_intelligence.jsonl` for example output from the demo script.

## Testing

Run the demo script:
```bash
python -m core.unified_session_logger
```

This creates example log files and demonstrates all logging features.
