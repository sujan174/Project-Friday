

# Industry-Standard Logging System

## 🎯 Overview

A comprehensive, production-ready observability system implementing industry best practices for distributed systems. This logging infrastructure provides complete visibility into agent orchestration, intelligence pipeline processing, and system performance.

## ✨ Features

### 🔍 Distributed Tracing (OpenTelemetry-Inspired)
- **Trace IDs**: Track requests across the entire system
- **Span IDs**: Monitor individual operations with parent-child relationships
- **W3C Trace Context**: Industry-standard trace propagation
- **Automatic timing**: All spans tracked with microsecond precision
- **Span attributes & events**: Rich contextual information
- **JSON export**: Export traces for analysis

### 📝 Structured Logging
- **JSON format**: Machine-parsable logs for aggregation
- **Context propagation**: Automatic injection of trace/span/session IDs
- **Multiple outputs**: Console (colored), file, and JSON logs simultaneously
- **Log rotation**: Automatic rotation and archiving
- **Per-module levels**: Fine-grained control over log verbosity
- **Thread-safe**: Safe for concurrent operations

### 🤖 Agent Orchestration Logging
- **State machine tracking**: Monitor all agent state transitions
- **Task lifecycle**: Complete tracking from assignment to completion
- **Routing decisions**: Log why tasks were routed to specific agents
- **Health monitoring**: Track agent health and performance
- **Success/failure tracking**: Detailed error tracking and recovery
- **Performance metrics**: Per-agent latency and success rates

### 🧠 Intelligence Pipeline Logging
- **Stage-by-stage tracking**: Monitor each intelligence processing stage
- **Intent classification**: Log detected intents and confidence scores
- **Entity extraction**: Track extracted entities and relationships
- **Context resolution**: Monitor coreference resolution
- **Task decomposition**: Log task breakdown and dependencies
- **Confidence scoring**: Track confidence at each stage
- **Decision logging**: Record decision-making with reasoning

### 📊 Metrics Collection
- **Counters**: Monotonically increasing values (requests, errors)
- **Gauges**: Point-in-time values (active connections, memory)
- **Histograms**: Distribution of values (latencies, sizes)
- **Timers**: Automatic timing with context managers
- **Percentiles**: P50, P90, P95, P99 calculations
- **Multiple export formats**: JSON and Prometheus text format

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Code                          │
│  (orchestrator.py, agents, intelligence system)              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ├─────┐
                            │     │
                  ┌─────────┴─┐ ┌─┴────────────┐
                  │  Logging  │ │   Tracing    │
                  │           │ │              │
                  │  Logger   │ │  Tracer      │
                  │  Context  │ │  Spans       │
                  │  Formats  │ │  Traces      │
                  └─────┬─────┘ └──┬───────────┘
                        │          │
         ┌──────────────┴──────┬───┴───────────┬────────────┐
         │                     │               │            │
┌────────┴─────────┐  ┌────────┴────────┐ ┌───┴─────┐ ┌────┴─────┐
│  Orchestration   │  │  Intelligence   │ │ Metrics │ │  Export  │
│     Logger       │  │     Logger      │ │  System │ │  System  │
│                  │  │                 │ │         │ │          │
│  Agent states    │  │  Pipeline stages│ │Counters │ │   JSON   │
│  Task tracking   │  │  Decisions      │ │Gauges   │ │Prometheus│
│  Routing         │  │  Confidence     │ │Timers   │ │          │
└──────────────────┘  └─────────────────┘ └─────────┘ └──────────┘
                            │
                            ↓
                  ┌─────────────────┐
                  │  Log Files      │
                  │                 │
                  │  logs/          │
                  │  ├─ traces/     │
                  │  ├─ orchest../  │
                  │  ├─ intell../   │
                  │  ├─ metrics/    │
                  │  └─ *.log       │
                  └─────────────────┘
```

## 📦 Components

### Core Modules

| Module | Purpose | Key Features |
|--------|---------|-------------|
| `distributed_tracing.py` | Distributed tracing | Trace/span generation, context propagation, W3C format |
| `logging_config.py` | Structured logging | Formatters, handlers, context injection |
| `orchestration_logger.py` | Agent orchestration | State tracking, task lifecycle, routing |
| `intelligence_logger.py` | Intelligence pipeline | Stage logging, decision tracking |
| `metrics_aggregator.py` | Metrics collection | Counters, gauges, histograms, export |
| `observability.py` | Unified API | Single entry point, initialization |

### Data Flow

```
User Request
    ↓
[Start Trace] ──────────────────────────┐
    │                                   │
[Intelligence Logger]                   │
    ├─ Intent Classification            │
    ├─ Entity Extraction               Trace Context
    ├─ Task Decomposition               │
    └─ Decision Making                  │
    ↓                                   │
[Orchestration Logger]                  │
    ├─ Routing Decision                 │
    ├─ Task Assignment                  │
    └─ Agent Execution ─────────────────┤
        ↓                               │
    [Metrics] ───────────────────────────┤
        ├─ Request Count                │
        ├─ Duration                     │
        └─ Success/Failure              │
    ↓                                   │
[End Trace] ────────────────────────────┘
    ↓
[Export All Data]
```

## 🚀 Quick Start

### 1. Installation

No additional dependencies needed! All modules are included.

### 2. Basic Usage

```python
from core.observability import initialize_observability, get_logger

# Initialize (once at startup)
obs = initialize_observability(
    session_id="my-session-id",
    log_level="INFO",
    verbose=False
)

# Get logger
logger = get_logger(__name__)

# Use it!
logger.info("Application started")
```

### 3. With Distributed Tracing

```python
from core.observability import traced_span, SpanKind

with traced_span("my_operation", kind=SpanKind.AGENT) as span:
    span.set_attribute("user.id", "user-123")
    # Do work
    span.add_event("operation_completed")
```

### 4. Full Example

See `test_logging_system.py` for a complete example.

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [LOGGING_GUIDE.md](LOGGING_GUIDE.md) | Complete usage guide with examples |
| [ORCHESTRATOR_LOGGING_INTEGRATION.md](ORCHESTRATOR_LOGGING_INTEGRATION.md) | Step-by-step integration guide |

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_logging_system.py
```

This will:
- ✅ Test basic logging
- ✅ Test distributed tracing
- ✅ Test orchestration logging
- ✅ Test intelligence logging
- ✅ Test metrics collection
- ✅ Export all data

Output will be in `logs/` directory.

## 📊 Log Output Examples

### Console (Human-Readable)

```
INFO | 2025-01-08 10:30:45 | trace=a1b2c3d4 | span=e5f6g7h8 | session=sess-123 | agent=slack | orchestrator:process:450 | Processing user request
```

### JSON Log (Machine-Parsable)

```json
{
  "timestamp": "2025-01-08T10:30:45.123Z",
  "level": "INFO",
  "logger": "orchestrator",
  "message": "Processing user request",
  "trace_id": "a1b2c3d4e5f6g7h8",
  "span_id": "i9j0k1l2m3n4",
  "session_id": "sess-123",
  "agent_name": "slack"
}
```

### Trace Export

```json
{
  "trace_id": "a1b2c3d4e5f6g7h8",
  "duration_ms": 1234.5,
  "spans": [
    {
      "span_id": "i9j0k1l2",
      "name": "process_request",
      "kind": "ORCHESTRATION",
      "duration_ms": 1234.5,
      "status": "OK",
      "attributes": {
        "user.id": "user-123"
      },
      "events": [
        {
          "name": "request_processed",
          "timestamp_iso": "2025-01-08T10:30:46Z"
        }
      ]
    }
  ]
}
```

## 🎨 Key Design Principles

### 1. **Industry Standards**
- OpenTelemetry semantic conventions
- W3C Trace Context specification
- Structured logging best practices
- Prometheus metrics format

### 2. **Performance**
- Minimal overhead (<1% in most cases)
- Async-safe operations
- Efficient context propagation
- Lazy evaluation where possible

### 3. **Usability**
- Simple, intuitive API
- Sensible defaults
- Easy integration
- Comprehensive documentation

### 4. **Observability**
- Complete visibility
- Searchable logs
- Correlation across systems
- Performance insights

### 5. **Production-Ready**
- Thread-safe
- Log rotation
- Error handling
- Resource cleanup

## 🔧 Configuration

### Environment Variables

```bash
# Log level
export LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Enable/disable features (in config.py)
ENABLE_CONSOLE_LOGGING=true
ENABLE_FILE_LOGGING=true
ENABLE_JSON_LOGGING=true
ENABLE_COLORED_LOGS=true
```

### Programmatic Configuration

```python
obs = initialize_observability(
    session_id="sess-123",
    service_name="orchestrator",
    log_dir="custom_logs",
    log_level="DEBUG",
    enable_tracing=True,
    enable_metrics=True,
    verbose=True
)
```

## 📈 Performance Impact

| Feature | Overhead | Notes |
|---------|----------|-------|
| Basic logging | < 0.1ms | Per log statement |
| Distributed tracing | < 0.5ms | Per span |
| Metrics | < 0.01ms | Per metric update |
| JSON export | < 10ms | Per trace export |

**Total overhead: < 1% for typical workloads**

## 🗂️ File Structure

After running, you'll see:

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
├── orchestrator.log
├── orchestrator.json.log
├── test_module.log
└── test_module.json.log
```

## 🎯 Use Cases

### Development
- **Debugging**: Trace requests through the system
- **Performance**: Identify bottlenecks
- **Testing**: Verify correct behavior

### Production
- **Monitoring**: Track system health
- **Alerting**: Detect anomalies
- **Analysis**: Understand usage patterns

### Operations
- **Troubleshooting**: Diagnose issues quickly
- **Capacity Planning**: Understand resource usage
- **Compliance**: Audit trails

## 🤝 Integration Guide

### For Orchestrator

See [ORCHESTRATOR_LOGGING_INTEGRATION.md](ORCHESTRATOR_LOGGING_INTEGRATION.md)

### For Agents

```python
# In agent __init__
from core.observability import get_logger, traced_span, SpanKind

self.logger = get_logger(__name__)

# In agent methods
async def execute(self, task):
    with traced_span(f"{self.name}.execute", kind=SpanKind.AGENT) as span:
        span.set_attribute("agent.name", self.name)
        span.set_attribute("task.id", task.get('id'))

        # Execute task
        result = await self._do_work(task)

        span.set_attribute("result.success", True)
        return result
```

### For Intelligence System

```python
# In coordinator.py
from core.observability import get_observability

obs = get_observability()
intel_logger = obs.intelligence_logger

# Log each stage
intel_logger.log_intent_classification(...)
intel_logger.log_entity_extraction(...)
intel_logger.log_task_decomposition(...)
```

## 📚 Further Reading

- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [W3C Trace Context](https://www.w3.org/TR/trace-context/)
- [Structured Logging](https://www.structlog.org/)
- [Prometheus Metrics](https://prometheus.io/docs/concepts/metric_types/)

## 🐛 Troubleshooting

### Logs not appearing?

1. Check initialization: `obs = initialize_observability(...)`
2. Check log level: Set to `DEBUG` for development
3. Check file permissions in `logs/` directory

### Trace IDs not showing?

1. Ensure you started a trace: `start_trace("name")`
2. Check context propagation: Are you in an async context?

### Metrics not recording?

1. Verify initialization: `initialize_global_metrics()`
2. Check metric exists: `metrics.get_metric("name")`
3. Register custom metrics before use

## 📞 Support

For issues or questions:
1. Check documentation in `docs/`
2. Review test script: `test_logging_system.py`
3. Examine source code in `core/`

## 🎉 Summary

This logging system provides:

✅ **Complete Observability**: Logs, traces, and metrics
✅ **Industry Standards**: OpenTelemetry, W3C, Prometheus
✅ **Production Ready**: Thread-safe, performant, reliable
✅ **Easy to Use**: Simple API, great defaults
✅ **Well Documented**: Comprehensive guides and examples

**Start using it today for better visibility into your system!**

---

*Last updated: 2025-01-08*
*Version: 1.0*
