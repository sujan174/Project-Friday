# Comprehensive Logging System Guide

## Overview

This project now includes an industry-standard observability system with:

- **Distributed Tracing**: OpenTelemetry-inspired trace and span tracking
- **Structured Logging**: JSON logs with context propagation
- **Agent Orchestration Logging**: Track agent lifecycle and state transitions
- **Intelligence Pipeline Logging**: Monitor each stage of intelligence processing
- **Metrics Collection**: Counters, gauges, histograms, and timers
- **Unified API**: Single entry point for all observability needs

## Quick Start

### 1. Initialize the Observability System

At application startup (typically in `main.py` or `orchestrator.py.__init__`):

```python
from core.observability import initialize_observability, get_logger

# Initialize observability system
obs = initialize_observability(
    session_id="unique-session-id",
    service_name="orchestrator",
    log_level="INFO",  # or DEBUG for development
    verbose=True
)

# Get a logger for your module
logger = get_logger(__name__)
logger.info("Application started")
```

### 2. Use Distributed Tracing

Wrap operations in spans to track execution flow:

```python
from core.observability import traced_span, SpanKind

# Trace an operation
with traced_span("user_request_processing", kind=SpanKind.ORCHESTRATION) as span:
    span.set_attribute("user.id", user_id)
    span.set_attribute("request.type", "create_issue")

    # Your code here
    result = process_user_request(user_input)

    span.add_event("request_processed")
    span.set_attribute("result.success", True)
```

### 3. Log Agent Orchestration

Track agent lifecycle and task execution:

```python
# Access orchestration logger
orch_logger = obs.orchestration_logger

# Log agent initialization
orch_logger.log_agent_initialized(
    agent_name="slack_agent",
    capabilities=["send_message", "create_channel"],
    metadata={"version": "1.0"}
)

# Log agent ready
orch_logger.log_agent_ready("slack_agent")

# Log task assignment
task_id = "task-123"
orch_logger.log_task_assigned(
    task_id=task_id,
    task_name="send_slack_message",
    agent_name="slack_agent",
    metadata={"channel": "#engineering"}
)

# Log task start
orch_logger.log_task_started(task_id)

# Log task completion
orch_logger.log_task_completed(
    task_id=task_id,
    success=True,
    metadata={"message_id": "msg-456"}
)

# Log routing decision
orch_logger.log_routing_decision(
    task_name="send_message",
    selected_agent="slack_agent",
    reason="Intent matches Slack capabilities",
    considered_agents=["slack_agent", "email_agent"],
    confidence=0.95
)
```

### 4. Log Intelligence Pipeline

Track each stage of intelligence processing:

```python
# Access intelligence logger
intel_logger = obs.intelligence_logger

# Log message processing
start_time = intel_logger.log_message_processing_start("Create a Jira issue for the bug")

# Log intent classification
intel_logger.log_intent_classification(
    message="Create a Jira issue for the bug",
    detected_intents=["CREATE", "JIRA"],
    confidence_scores={"CREATE": 0.95, "JIRA": 0.90},
    classification_method="keyword",
    duration_ms=5.2,
    cache_hit=False
)

# Log entity extraction
intel_logger.log_entity_extraction(
    message="Create a Jira issue for the bug",
    extracted_entities={
        "action": ["create"],
        "platform": ["jira"],
        "resource": ["issue"],
        "topic": ["bug"]
    },
    entity_relationships=[
        {"type": "action_target", "source": "create", "target": "issue"}
    ],
    confidence=0.92,
    duration_ms=8.5,
    cache_hit=False
)

# Log task decomposition
intel_logger.log_task_decomposition(
    tasks=[
        {"name": "create_jira_issue", "agent": "jira", "priority": 1}
    ],
    dependency_graph={},
    execution_plan="sequential",
    confidence=0.88,
    duration_ms=12.3
)

# Log decision
from core.intelligence_logger import DecisionType

intel_logger.log_decision(
    decision_type=DecisionType.CONFIRM,
    confidence=0.88,
    reasoning="Medium confidence, requires user confirmation",
    factors={
        "intent_confidence": 0.95,
        "entity_confidence": 0.92,
        "task_confidence": 0.88,
        "risk_level": "medium"
    }
)

# Log message processing complete
intel_logger.log_message_processing_complete(start_time, success=True)
```

### 5. Record Metrics

Track performance and system metrics:

```python
from core.observability import get_global_metrics

metrics = get_global_metrics()

# Increment counter
requests_counter = metrics.get_metric("requests_total")
requests_counter.increment()

# Set gauge
active_agents = metrics.get_metric("active_agents")
active_agents.set(5)

# Observe histogram value
duration_histogram = metrics.get_metric("request_duration")
duration_histogram.observe(125.5)  # milliseconds

# Use timer context manager
timer = metrics.get_metric("agent_task_duration")
with timer.time_function():
    # Execute task
    result = execute_agent_task()
```

### 6. Export and Cleanup

At session end:

```python
# Export all observability data
obs.export_all()

# Cleanup resources
obs.cleanup()
```

## Advanced Usage

### Using the Tracing Decorator

```python
from core.observability import trace, SpanKind

@trace("process_user_input", kind=SpanKind.ORCHESTRATION)
def process_user_input(user_input: str):
    # Automatically wrapped in a span
    return parse_input(user_input)
```

### Logging with Context

```python
from core.logging_config import LogContext

# Set context that will appear in all logs
LogContext.set_session("session-123")
LogContext.set_agent("slack_agent")

logger.info("Processing message")  # Will include session and agent in log
```

### Custom Metrics

```python
metrics = get_global_metrics()

# Register custom counter
custom_counter = metrics.register_counter(
    name="api_calls_total",
    description="Total API calls made",
    unit="calls",
    labels={"api": "slack"}
)

# Use it
custom_counter.increment(labels={"endpoint": "send_message"})
```

## Log Output Formats

### Console Output

```
INFO | 2025-01-08 10:30:45 | trace=a1b2c3d4 | span=e5f6g7h8 | session=sess-123 | agent=slack | orchestrator:process_request:450 | Processing user request
```

### JSON Log (for machine parsing)

```json
{
  "timestamp": "2025-01-08T10:30:45.123Z",
  "level": "INFO",
  "logger": "orchestrator",
  "message": "Processing user request",
  "module": "orchestrator",
  "function": "process_request",
  "line": 450,
  "trace_id": "a1b2c3d4e5f6g7h8",
  "span_id": "i9j0k1l2m3n4",
  "session_id": "sess-123",
  "agent_name": "slack"
}
```

### Trace Export (JSON)

```json
{
  "trace_id": "a1b2c3d4e5f6g7h8",
  "start_time_iso": "2025-01-08T10:30:45.000Z",
  "duration_ms": 1234.5,
  "spans": [
    {
      "span_id": "i9j0k1l2m3n4",
      "name": "user_request_processing",
      "kind": "ORCHESTRATION",
      "duration_ms": 1234.5,
      "status": "OK",
      "attributes": {
        "user.id": "user-123",
        "request.type": "create_issue"
      },
      "events": [
        {
          "name": "request_processed",
          "timestamp_iso": "2025-01-08T10:30:46.000Z"
        }
      ]
    }
  ]
}
```

## File Structure

After running, you'll find:

```
logs/
├── traces/
│   └── trace_<trace_id>_<timestamp>.json
├── orchestration/
│   └── orchestration_<session_id>_<timestamp>.json
├── intelligence/
│   └── intelligence_<session_id>_<timestamp>.json
├── metrics/
│   └── metrics_<session_id>.json
└── *.log (standard log files)
    └── *.json.log (JSON structured logs)
```

## Best Practices

### 1. Always Initialize First

```python
# ❌ BAD - Using logger before initialization
logger = get_logger(__name__)
logger.info("Starting")  # Context not set!

# ✅ GOOD - Initialize first
obs = initialize_observability(session_id="sess-123")
logger = get_logger(__name__)
logger.info("Starting")  # Now has proper context
```

### 2. Use Appropriate Span Kinds

```python
# For agent execution
with traced_span("slack_send_message", kind=SpanKind.AGENT):
    pass

# For intelligence processing
with traced_span("intent_classification", kind=SpanKind.INTELLIGENCE):
    pass

# For orchestration logic
with traced_span("route_to_agent", kind=SpanKind.ORCHESTRATION):
    pass

# For internal operations
with traced_span("parse_config", kind=SpanKind.INTERNAL):
    pass
```

### 3. Add Meaningful Attributes

```python
with traced_span("create_issue") as span:
    # ✅ GOOD - Specific, searchable attributes
    span.set_attribute("jira.project", "KAN")
    span.set_attribute("jira.issue_type", "Bug")
    span.set_attribute("jira.priority", "High")

    # ❌ BAD - Too vague
    span.set_attribute("data", "some data")
```

### 4. Log at Appropriate Levels

```python
# DEBUG - Detailed diagnostic info
logger.debug(f"Parsing entity: {entity_data}")

# INFO - General informational messages
logger.info("Task completed successfully")

# WARNING - Something unexpected but not critical
logger.warning("Cache miss, fetching from API")

# ERROR - Error occurred but system can continue
logger.error("Failed to send notification", exc_info=True)

# CRITICAL - Critical failure requiring immediate attention
logger.critical("Database connection lost")
```

### 5. Export Before Exit

```python
import atexit

# Register cleanup on exit
obs = initialize_observability(session_id="sess-123")
atexit.register(obs.cleanup)
```

## Integration with Existing Code

### Minimal Integration (Just Logging)

```python
from core.observability import initialize_observability, get_logger

# In __init__ or main
obs = initialize_observability(session_id=self.session_id, verbose=self.verbose)

# Replace existing logger imports
# OLD: from core.logger import get_logger
# NEW: from core.observability import get_logger

# Everything else works the same!
```

### Full Integration (Tracing + Orchestration + Intelligence)

```python
from core.observability import (
    initialize_observability,
    get_logger,
    traced_span,
    SpanKind
)

# Initialize
obs = initialize_observability(
    session_id=self.session_id,
    verbose=self.verbose
)

# Store for later use
self.observability = obs
self.logger = get_logger(__name__)

# Use throughout your code
with traced_span("process_request", kind=SpanKind.ORCHESTRATION) as span:
    # Log orchestration
    self.observability.orchestration_logger.log_routing_decision(...)

    # Log intelligence
    self.observability.intelligence_logger.log_intent_classification(...)

    # Regular logging
    self.logger.info("Processing complete")
```

## Troubleshooting

### Issue: No trace_id in logs

**Solution**: Make sure to initialize observability and start a trace:

```python
from core.observability import start_trace, end_trace

trace_id = start_trace("main_workflow")
# ... your code ...
end_trace(trace_id)
```

### Issue: Metrics not found

**Solution**: Register metrics before using them:

```python
metrics = get_global_metrics()
metrics.register_counter("my_metric", "My custom metric")
counter = metrics.get_metric("my_metric")
counter.increment()
```

### Issue: Large trace files

**Solution**: Traces are exported per-session. Clean up old traces:

```bash
find logs/traces -mtime +7 -delete  # Delete traces older than 7 days
```

## Performance Considerations

- **Trace Export**: Happens at trace end, non-blocking
- **JSON Logging**: Slight overhead, but enables powerful analysis
- **Metrics**: Very low overhead (in-memory until export)
- **Context Variables**: Thread-safe, minimal overhead

## Summary

The new observability system provides:

✅ **Complete visibility** into system behavior
✅ **Distributed tracing** across agents and operations
✅ **Structured logs** for easy parsing and analysis
✅ **Performance metrics** for monitoring
✅ **Simple API** - easy to integrate
✅ **Industry standards** - OpenTelemetry-inspired

For questions or issues, see the source code in `core/observability.py` or related modules.
