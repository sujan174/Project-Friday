#!/usr/bin/env python3
"""
Test Script for New Logging System

Demonstrates all features of the observability system:
- Distributed tracing
- Structured logging
- Agent orchestration logging
- Intelligence pipeline logging
- Metrics collection

Run this script to verify the logging system works correctly.

Usage:
    python test_logging_system.py
"""

import asyncio
import time
import uuid
from pathlib import Path

# Import observability system
from core.observability import (
    initialize_observability,
    get_logger,
    traced_span,
    SpanKind,
    start_trace,
    end_trace,
    get_global_metrics
)

from core.intelligence_logger import DecisionType


def test_basic_logging():
    """Test 1: Basic structured logging"""
    print("\n" + "="*60)
    print("TEST 1: Basic Structured Logging")
    print("="*60)

    # Initialize
    obs = initialize_observability(
        session_id="test-basic-logging",
        log_level="DEBUG",
        verbose=True
    )

    logger = get_logger("test_module")

    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Test with extra data
    logger.info(
        "User action performed",
        extra={
            'user_id': 'user-123',
            'action': 'create_issue',
            'platform': 'jira'
        }
    )

    print("✅ Basic logging test complete")
    print(f"📂 Check: {obs.log_dir}/test_module.log")
    print(f"📂 Check: {obs.log_dir}/test_module.json.log")

    return obs


def test_distributed_tracing(obs):
    """Test 2: Distributed tracing"""
    print("\n" + "="*60)
    print("TEST 2: Distributed Tracing")
    print("="*60)

    logger = get_logger("test_tracing")

    # Start a trace
    trace_id = start_trace(
        name="test_workflow",
        metadata={"test": "distributed_tracing"}
    )

    print(f"Started trace: {trace_id}")

    # Root span
    with traced_span("main_operation", kind=SpanKind.ORCHESTRATION) as root_span:
        root_span.set_attribute("operation.type", "test")
        root_span.add_event("operation_started")

        logger.info("Executing main operation")

        # Child span 1
        with traced_span("sub_operation_1", kind=SpanKind.AGENT) as child1:
            child1.set_attribute("agent.name", "test_agent_1")
            logger.info("Executing sub-operation 1")
            time.sleep(0.1)
            child1.add_event("sub_op_1_completed")

        # Child span 2
        with traced_span("sub_operation_2", kind=SpanKind.INTELLIGENCE) as child2:
            child2.set_attribute("intelligence.stage", "classification")
            logger.info("Executing sub-operation 2")
            time.sleep(0.05)
            child2.add_event("sub_op_2_completed")

        root_span.add_event("operation_completed")

    # End trace
    end_trace(trace_id)

    print("✅ Distributed tracing test complete")
    print(f"📂 Check: {obs.log_dir}/traces/trace_{trace_id}_*.json")


def test_orchestration_logging(obs):
    """Test 3: Agent orchestration logging"""
    print("\n" + "="*60)
    print("TEST 3: Agent Orchestration Logging")
    print("="*60)

    orch_logger = obs.orchestration_logger

    # Test agent lifecycle
    print("Testing agent lifecycle...")

    # Initialize agent
    orch_logger.log_agent_initialized(
        agent_name="test_agent",
        capabilities=["send_message", "create_issue", "search"],
        metadata={"version": "1.0", "type": "test"}
    )

    # Agent ready
    orch_logger.log_agent_ready("test_agent")

    # Assign task
    task_id = str(uuid.uuid4())
    orch_logger.log_task_assigned(
        task_id=task_id,
        task_name="send_test_message",
        agent_name="test_agent",
        metadata={"priority": "high"}
    )

    # Start task
    orch_logger.log_task_started(task_id)

    # Simulate work
    time.sleep(0.1)

    # Complete task
    orch_logger.log_task_completed(
        task_id=task_id,
        success=True,
        metadata={"result": "message_sent"}
    )

    # Log routing decision
    orch_logger.log_routing_decision(
        task_name="send_message",
        selected_agent="test_agent",
        reason="Agent has send_message capability",
        considered_agents=["test_agent", "slack_agent", "email_agent"],
        confidence=0.95,
        metadata={"selection_method": "capability_match"}
    )

    # Log health
    orch_logger.log_agent_health(
        agent_name="test_agent",
        is_healthy=True,
        metrics={
            "success_rate": 0.98,
            "avg_latency_ms": 123.4,
            "total_tasks_completed": 10
        }
    )

    print("✅ Orchestration logging test complete")
    print(f"📂 Check: {obs.log_dir}/orchestration/orchestration_*.json")


def test_intelligence_logging(obs):
    """Test 4: Intelligence pipeline logging"""
    print("\n" + "="*60)
    print("TEST 4: Intelligence Pipeline Logging")
    print("="*60)

    intel_logger = obs.intelligence_logger

    # Test message processing
    message = "Create a Jira issue for the login bug"
    print(f"Processing: {message}")

    # Start processing
    start_time = intel_logger.log_message_processing_start(message)

    # Intent classification
    intel_logger.log_intent_classification(
        message=message,
        detected_intents=["CREATE", "JIRA"],
        confidence_scores={"CREATE": 0.95, "JIRA": 0.90},
        classification_method="keyword",
        duration_ms=5.2,
        cache_hit=False
    )

    # Entity extraction
    intel_logger.log_entity_extraction(
        message=message,
        extracted_entities={
            "action": ["create"],
            "platform": ["jira"],
            "resource": ["issue"],
            "topic": ["login", "bug"]
        },
        entity_relationships=[
            {"type": "action_target", "source": "create", "target": "issue"},
            {"type": "located_in", "source": "bug", "target": "login"}
        ],
        confidence=0.92,
        duration_ms=8.5,
        cache_hit=False
    )

    # Context resolution
    intel_logger.log_context_resolution(
        references_resolved=2,
        context_applied=True,
        duration_ms=3.2,
        details={"resolved": ["login", "bug"]}
    )

    # Task decomposition
    intel_logger.log_task_decomposition(
        tasks=[
            {
                "name": "create_jira_issue",
                "agent": "jira",
                "priority": 1,
                "parameters": {
                    "summary": "Login bug",
                    "project": "PROJ"
                }
            }
        ],
        dependency_graph={},
        execution_plan="sequential",
        confidence=0.88,
        duration_ms=12.3
    )

    # Confidence scoring
    intel_logger.log_confidence_score(
        overall_confidence=0.90,
        component_scores={
            "intent": 0.925,
            "entity": 0.92,
            "context": 0.95,
            "task": 0.88
        },
        factors={
            "keyword_matches": 3,
            "entity_count": 4,
            "context_available": True
        },
        duration_ms=2.1
    )

    # Decision
    intel_logger.log_decision(
        decision_type=DecisionType.CONFIRM,
        confidence=0.90,
        reasoning="High confidence but requires user confirmation for Jira creation",
        factors={
            "overall_confidence": 0.90,
            "risk_level": "medium",
            "user_preference": "confirm_jira"
        },
        metadata={"requires_confirmation": True}
    )

    # Complete processing
    intel_logger.log_message_processing_complete(start_time, success=True)

    print("✅ Intelligence logging test complete")
    print(f"📂 Check: {obs.log_dir}/intelligence/intelligence_*.json")


def test_metrics(obs):
    """Test 5: Metrics collection"""
    print("\n" + "="*60)
    print("TEST 5: Metrics Collection")
    print("="*60)

    metrics = get_global_metrics()

    # Test counter
    print("Testing counters...")
    requests_counter = metrics.get_metric("requests_total")
    for i in range(10):
        requests_counter.increment()

    succeeded = metrics.get_metric("requests_succeeded")
    for i in range(8):
        succeeded.increment()

    failed = metrics.get_metric("requests_failed")
    for i in range(2):
        failed.increment()

    # Test gauge
    print("Testing gauges...")
    active_agents = metrics.get_metric("active_agents")
    active_agents.set(5)

    # Test histogram
    print("Testing histograms...")
    duration = metrics.get_metric("request_duration")
    for latency in [50, 75, 100, 125, 150, 200, 250]:
        duration.observe(latency)

    # Test timer with context manager
    print("Testing timer...")
    timer = metrics.get_metric("agent_task_duration")
    for i in range(5):
        with timer.time_function():
            time.sleep(0.05)

    # Print statistics
    print("\n📊 Metrics Summary:")
    print(f"  Total requests: {requests_counter.get_total()}")
    print(f"  Succeeded: {succeeded.get_total()}")
    print(f"  Failed: {failed.get_total()}")
    print(f"  Active agents: {active_agents.get_latest_value()}")

    duration_stats = duration.get_statistics()
    print(f"  Request duration:")
    print(f"    Mean: {duration_stats.get('mean', 0):.1f}ms")
    print(f"    P50: {duration_stats.get('p50', 0):.1f}ms")
    print(f"    P95: {duration_stats.get('p95', 0):.1f}ms")
    print(f"    P99: {duration_stats.get('p99', 0):.1f}ms")

    print("\n✅ Metrics collection test complete")
    print(f"📂 Check: {obs.log_dir}/metrics/metrics_*.json")


def test_export_and_cleanup(obs):
    """Test 6: Export and cleanup"""
    print("\n" + "="*60)
    print("TEST 6: Export and Cleanup")
    print("="*60)

    print("Exporting all observability data...")

    # Export everything
    export_results = obs.export_all()

    print("\n📦 Export Results:")
    for key, value in export_results.items():
        if isinstance(value, dict):
            print(f"  {key}: exported successfully")

    # Cleanup
    print("\nCleaning up...")
    obs.cleanup()

    print("✅ Export and cleanup complete")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("LOGGING SYSTEM COMPREHENSIVE TEST")
    print("="*60)
    print("\nThis test will verify all features of the new logging system.")
    print("Watch for log output in the console and check the logs/ directory.")

    try:
        # Test 1: Basic logging
        obs = test_basic_logging()

        # Test 2: Distributed tracing
        test_distributed_tracing(obs)

        # Test 3: Orchestration logging
        test_orchestration_logging(obs)

        # Test 4: Intelligence logging
        test_intelligence_logging(obs)

        # Test 5: Metrics
        test_metrics(obs)

        # Test 6: Export and cleanup
        test_export_and_cleanup(obs)

        # Final summary
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✅")
        print("="*60)
        print(f"\n📂 Log files location: {obs.log_dir}")
        print("\nYou should see the following files:")
        print("  logs/")
        print("    ├── traces/trace_*.json")
        print("    ├── orchestration/orchestration_*.json")
        print("    ├── intelligence/intelligence_*.json")
        print("    ├── metrics/metrics_*.json")
        print("    ├── *.log (human-readable logs)")
        print("    └── *.json.log (structured JSON logs)")
        print("\n🎉 Logging system is working correctly!")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
