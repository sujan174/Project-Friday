# Orchestrator Logging Integration Guide

## Overview

This guide shows exactly how to integrate the new observability system into `orchestrator.py`.

## Step 1: Update Imports

Add these imports to the top of `orchestrator.py`:

```python
# NEW: Import observability system
from core.observability import (
    initialize_observability,
    get_logger as get_observability_logger,
    traced_span,
    SpanKind,
    start_trace,
    end_trace
)
```

## Step 2: Initialize in `__init__`

In the `OrchestratorAgent.__init__` method, add observability initialization:

```python
def __init__(self, connectors_dir: str = "connectors", verbose: bool = False, llm: Optional[BaseLLM] = None):
    # ... existing initialization code ...

    # NEW: Initialize observability system
    self.observability = initialize_observability(
        session_id=self.session_id,
        service_name="orchestrator",
        log_level=Config.LOG_LEVEL if hasattr(Config, 'LOG_LEVEL') else "INFO",
        enable_tracing=True,
        enable_metrics=True,
        verbose=self.verbose
    )

    # Update logger to use observability logger
    # Replace: logger = get_logger(__name__)
    # With:
    self.logger = get_observability_logger(__name__)

    # Store convenience references
    self.orch_logger = self.observability.orchestration_logger
    self.intel_logger = self.observability.intelligence_logger

    # ... rest of initialization ...
```

## Step 3: Add Tracing to Request Processing

When processing user requests, wrap in a trace:

```python
async def run_interactive(self):
    """Run interactive chat session"""
    # ... existing UI setup ...

    while True:
        user_input = await self._get_user_input()

        if user_input in ['exit', 'quit']:
            break

        # NEW: Start a trace for this request
        trace_id = start_trace(
            name="user_request",
            metadata={
                "user_input": user_input[:100],  # First 100 chars
                "session_id": self.session_id
            }
        )

        try:
            # Wrap processing in a span
            with traced_span("process_user_request", kind=SpanKind.ORCHESTRATION) as span:
                span.set_attribute("request.length", len(user_input))
                span.add_event("processing_started")

                # Process the request
                await self._process_request(user_input)

                span.add_event("processing_completed")
                span.set_attribute("request.success", True)

        except Exception as e:
            self.logger.error(f"Request processing failed: {e}", exc_info=True)
        finally:
            # End the trace
            end_trace(trace_id)
```

## Step 4: Log Agent Discovery and Initialization

In the `_discover_and_load_agents` method:

```python
def _discover_and_load_agents(self):
    """Discover and load agent modules"""
    with traced_span("discover_agents", kind=SpanKind.INTERNAL) as span:
        # ... existing discovery code ...

        for agent_name, agent_instance in self.sub_agents.items():
            # NEW: Log agent initialization
            self.orch_logger.log_agent_initialized(
                agent_name=agent_name,
                capabilities=self.agent_capabilities.get(agent_name, []),
                metadata={"type": type(agent_instance).__name__}
            )

            # NEW: Log agent as ready
            self.orch_logger.log_agent_ready(agent_name)

            span.add_event(f"agent_loaded", {
                "agent_name": agent_name,
                "capabilities_count": len(self.agent_capabilities.get(agent_name, []))
            })

        span.set_attribute("agents.count", len(self.sub_agents))
```

## Step 5: Log Intelligence Processing

When using the intelligence system, add logging:

```python
# In the method where you process intelligence
with traced_span("intelligence_pipeline", kind=SpanKind.INTELLIGENCE) as span:
    # Log message processing start
    start_time = self.intel_logger.log_message_processing_start(user_input)

    # Intent classification
    intents = self.intent_classifier.classify(user_input)
    self.intel_logger.log_intent_classification(
        message=user_input,
        detected_intents=intents['intents'],
        confidence_scores=intents['confidence'],
        classification_method=intents.get('method', 'unknown'),
        duration_ms=intents.get('duration_ms', 0),
        cache_hit=intents.get('cache_hit', False)
    )
    span.add_event("intents_classified", {"count": len(intents['intents'])})

    # Entity extraction
    entities = self.entity_extractor.extract(user_input)
    self.intel_logger.log_entity_extraction(
        message=user_input,
        extracted_entities=entities['entities'],
        entity_relationships=entities.get('relationships', []),
        confidence=entities.get('confidence', 0),
        duration_ms=entities.get('duration_ms', 0),
        cache_hit=entities.get('cache_hit', False)
    )
    span.add_event("entities_extracted", {
        "count": sum(len(v) for v in entities['entities'].values())
    })

    # Task decomposition
    tasks = self.task_decomposer.decompose(user_input, intents, entities)
    self.intel_logger.log_task_decomposition(
        tasks=tasks['tasks'],
        dependency_graph=tasks.get('dependencies', {}),
        execution_plan=tasks.get('execution_plan', 'sequential'),
        confidence=tasks.get('confidence', 0),
        duration_ms=tasks.get('duration_ms', 0)
    )
    span.add_event("tasks_decomposed", {"count": len(tasks['tasks'])})

    # Decision
    from core.intelligence_logger import DecisionType
    decision = self._make_decision(confidence_score)

    self.intel_logger.log_decision(
        decision_type=DecisionType.PROCEED if decision == 'proceed' else DecisionType.CONFIRM,
        confidence=confidence_score,
        reasoning=f"Confidence {confidence_score:.2f} {'above' if decision == 'proceed' else 'below'} threshold",
        factors={
            "intent_confidence": intents.get('confidence', {}),
            "entity_confidence": entities.get('confidence', 0),
            "task_confidence": tasks.get('confidence', 0)
        }
    )

    # Log completion
    self.intel_logger.log_message_processing_complete(start_time, success=True)
```

## Step 6: Log Agent Task Execution

When calling an agent to execute a task:

```python
async def _execute_agent_task(self, agent_name: str, task: Dict[str, Any]):
    """Execute a task with an agent"""
    import uuid

    task_id = str(uuid.uuid4())

    # NEW: Log task assignment
    self.orch_logger.log_task_assigned(
        task_id=task_id,
        task_name=task.get('name', 'unknown'),
        agent_name=agent_name,
        metadata=task.get('metadata', {})
    )

    with traced_span(f"execute_task.{agent_name}", kind=SpanKind.AGENT) as span:
        span.set_attribute("agent.name", agent_name)
        span.set_attribute("task.id", task_id)
        span.set_attribute("task.name", task.get('name', 'unknown'))

        # NEW: Log task started
        self.orch_logger.log_task_started(task_id)
        span.add_event("task_started")

        try:
            # Execute the task
            result = await agent.execute(task)

            # NEW: Log task completed successfully
            self.orch_logger.log_task_completed(
                task_id=task_id,
                success=True,
                metadata={"result": result}
            )

            span.add_event("task_completed")
            span.set_attribute("task.success", True)

            # Update metrics
            from core.observability import get_global_metrics
            metrics = get_global_metrics()
            metrics.get_metric("agent_tasks_succeeded").increment()

            return result

        except Exception as e:
            # NEW: Log task failed
            self.orch_logger.log_task_completed(
                task_id=task_id,
                success=False,
                error=str(e),
                metadata={"error_type": type(e).__name__}
            )

            span.set_attribute("task.success", False)
            span.set_attribute("error.message", str(e))

            # Update metrics
            from core.observability import get_global_metrics
            metrics = get_global_metrics()
            metrics.get_metric("agent_tasks_failed").increment()

            raise
```

## Step 7: Log Routing Decisions

When deciding which agent to use:

```python
def _select_agent_for_task(self, task: Dict[str, Any]) -> str:
    """Select the best agent for a task"""
    with traced_span("agent_routing", kind=SpanKind.ORCHESTRATION) as span:
        # ... existing routing logic ...

        # NEW: Log the routing decision
        self.orch_logger.log_routing_decision(
            task_name=task.get('name', 'unknown'),
            selected_agent=selected_agent,
            reason=f"Agent has required capability: {required_capability}",
            considered_agents=list(self.sub_agents.keys()),
            confidence=0.95,  # Or calculate based on your logic
            metadata={
                "required_capability": required_capability,
                "available_agents": len(self.sub_agents)
            }
        )

        span.set_attribute("routing.selected_agent", selected_agent)
        span.add_event("agent_selected")

        return selected_agent
```

## Step 8: Update Metrics

Record metrics throughout execution:

```python
from core.observability import get_global_metrics

def _record_request_metrics(self, success: bool, duration_ms: float):
    """Record request metrics"""
    metrics = get_global_metrics()

    # Increment request counter
    metrics.get_metric("requests_total").increment()

    if success:
        metrics.get_metric("requests_succeeded").increment()
    else:
        metrics.get_metric("requests_failed").increment()

    # Record duration
    metrics.get_metric("request_duration").observe(duration_ms)
```

## Step 9: Cleanup on Exit

In the cleanup/shutdown method:

```python
async def shutdown(self):
    """Shutdown orchestrator and cleanup resources"""
    self.logger.info("Shutting down orchestrator")

    # ... existing cleanup ...

    # NEW: Export observability data and cleanup
    try:
        self.logger.info("Exporting observability data...")
        self.observability.export_all()
        self.observability.cleanup()
        self.logger.info("Observability data exported successfully")
    except Exception as e:
        self.logger.error(f"Failed to export observability data: {e}")
```

## Step 10: Optional - Add Cleanup on Process Exit

In `main.py` or where you initialize the orchestrator:

```python
import atexit

# Create orchestrator
orchestrator = OrchestratorAgent(verbose=args.verbose)

# Register cleanup
atexit.register(lambda: asyncio.run(orchestrator.shutdown()))
```

## Summary of Changes

### What to Add:

1. ✅ Import observability system
2. ✅ Initialize observability in `__init__`
3. ✅ Wrap request processing in traces
4. ✅ Log agent discovery and initialization
5. ✅ Log intelligence pipeline stages
6. ✅ Log task assignments and execution
7. ✅ Log routing decisions
8. ✅ Record metrics
9. ✅ Export data on shutdown

### What to Replace:

- Old logger: `from core.logger import get_logger`
- New logger: `from core.observability import get_logger`

### Benefits:

- 🔍 **Full visibility** into request flow
- 📊 **Performance metrics** for all operations
- 🐛 **Easy debugging** with distributed traces
- 📈 **Production monitoring** with metrics
- 🎯 **Searchable logs** with structured JSON

## Example Full Integration

Here's a minimal example showing all pieces together:

```python
from core.observability import (
    initialize_observability,
    get_logger,
    traced_span,
    SpanKind,
    start_trace,
    end_trace,
    get_global_metrics
)

class OrchestratorAgent:
    def __init__(self, verbose=False):
        self.session_id = str(uuid.uuid4())

        # Initialize observability
        self.observability = initialize_observability(
            session_id=self.session_id,
            verbose=verbose
        )

        self.logger = get_logger(__name__)
        self.orch_logger = self.observability.orchestration_logger
        self.intel_logger = self.observability.intelligence_logger

    async def process_request(self, user_input: str):
        # Start trace
        trace_id = start_trace("user_request")

        try:
            with traced_span("process_request", kind=SpanKind.ORCHESTRATION):
                # Intelligence pipeline
                with traced_span("intelligence", kind=SpanKind.INTELLIGENCE):
                    self.intel_logger.log_message_processing_start(user_input)
                    # ... intelligence processing ...

                # Agent execution
                task_id = str(uuid.uuid4())
                self.orch_logger.log_task_assigned(
                    task_id=task_id,
                    task_name="send_message",
                    agent_name="slack"
                )

                with traced_span("execute_task", kind=SpanKind.AGENT):
                    # ... execute task ...
                    self.orch_logger.log_task_completed(task_id, success=True)

                # Record metrics
                metrics = get_global_metrics()
                metrics.get_metric("requests_succeeded").increment()

        finally:
            end_trace(trace_id)

    def shutdown(self):
        self.observability.export_all()
        self.observability.cleanup()
```

## Next Steps

1. Start with Step 1-2 (imports and initialization)
2. Test that logging still works
3. Gradually add tracing (Step 3)
4. Add intelligence logging (Step 5)
5. Add orchestration logging (Step 6-7)
6. Add metrics (Step 8)
7. Add cleanup (Step 9-10)

Each step is independent and can be tested separately!
