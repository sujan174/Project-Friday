"""
Unified Observability System

Single entry point for all logging, tracing, and metrics.
Provides a simple API for the entire observability stack.

Features:
- Distributed tracing
- Structured logging
- Agent orchestration logging
- Intelligence pipeline logging
- Metrics collection
- Unified initialization
- Export and reporting

Usage:
    # Initialize at application startup
    from core.observability import initialize_observability, get_logger, traced_span

    # Initialize system
    initialize_observability(session_id="my-session", verbose=True)

    # Get logger
    logger = get_logger(__name__)
    logger.info("Starting application")

    # Use distributed tracing
    with traced_span("my_operation", kind=SpanKind.AGENT):
        # Do work
        pass

Author: AI System
Version: 1.0
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path

# Import all components
from .distributed_tracing import (
    initialize_global_tracer,
    get_global_tracer,
    traced_span,
    trace,
    Tracer,
    Span,
    SpanKind,
    SpanStatus,
    TraceContext
)

from .logging_config import (
    configure_logging,
    get_logger,
    LogContext,
    operation_context,
    track_performance
)

from .orchestration_logger import (
    OrchestrationLogger,
    AgentState,
    TaskStatus
)

from .intelligence_logger import (
    IntelligenceLogger,
    IntelligenceStage,
    DecisionType
)

from .metrics_aggregator import (
    initialize_global_metrics,
    get_global_metrics,
    MetricsAggregator,
    MetricType,
    Counter,
    Gauge,
    Histogram,
    Timer
)


# ============================================================================
# OBSERVABILITY SYSTEM
# ============================================================================

class ObservabilitySystem:
    """
    Unified observability system

    Provides single point of access to:
    - Distributed tracing
    - Structured logging
    - Agent orchestration logging
    - Intelligence pipeline logging
    - Metrics collection
    """

    def __init__(
        self,
        session_id: str,
        service_name: str = "orchestrator",
        log_dir: str = "logs",
        log_level: str = "INFO",
        enable_tracing: bool = True,
        enable_metrics: bool = True,
        verbose: bool = False
    ):
        """
        Initialize observability system

        Args:
            session_id: Session ID
            service_name: Service name for tracing
            log_dir: Base directory for logs
            log_level: Logging level
            enable_tracing: Enable distributed tracing
            enable_metrics: Enable metrics collection
            verbose: Enable verbose logging
        """
        self.session_id = session_id
        self.service_name = service_name
        self.log_dir = Path(log_dir)
        self.verbose = verbose

        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._init_logging(log_level)

        if enable_tracing:
            self._init_tracing()

        if enable_metrics:
            self._init_metrics()

        # Create specialized loggers (without export directories to avoid folder clutter)
        self.orchestration_logger = OrchestrationLogger(
            session_id=session_id,
            export_dir=None,  # Disabled - using SimpleSessionLogger instead
            verbose=verbose
        )

        self.intelligence_logger = IntelligenceLogger(
            session_id=session_id,
            export_dir=None,  # Disabled - using SimpleSessionLogger instead
            verbose=verbose
        )

        # Set session context
        LogContext.set_session(session_id)

        # Get main logger
        self.logger = get_logger(__name__)
        self.logger.info(f"Observability system initialized for session: {session_id}")

    def _init_logging(self, log_level: str):
        """Initialize logging system"""
        configure_logging({
            'log_level': log_level,
            'log_dir': str(self.log_dir),
            'enable_file_logging': False,  # Disabled - using SimpleSessionLogger instead
            'enable_json_logging': False,  # Disabled - using SimpleSessionLogger instead
            'enable_console': True,
            'enable_colors': True,
            'max_file_size_mb': 10,
            'backup_count': 5
        })

    def _init_tracing(self):
        """Initialize distributed tracing"""
        # Disabled export directory to avoid folder clutter
        initialize_global_tracer(
            service_name=self.service_name,
            export_dir=None  # Disabled - using SimpleSessionLogger instead
        )

    def _init_metrics(self):
        """Initialize metrics collection"""
        # Disabled export directory to avoid folder clutter
        initialize_global_metrics(
            retention_window_seconds=3600,
            export_dir=None,  # Disabled - using SimpleSessionLogger instead
            verbose=self.verbose
        )

    def export_all(self):
        """Export all observability data"""
        results = {}

        # Export orchestration summary (in-memory only since export_dir is disabled)
        try:
            results['orchestration'] = self.orchestration_logger.export_session_summary()
        except Exception as e:
            self.logger.error(f"Failed to export orchestration data: {e}")

        # Export intelligence summary (in-memory only since export_dir is disabled)
        try:
            results['intelligence'] = self.intelligence_logger.export_session_summary()
        except Exception as e:
            self.logger.error(f"Failed to export intelligence data: {e}")

        # Metrics export disabled - using SimpleSessionLogger instead

        return results

    def cleanup(self):
        """Cleanup observability resources"""
        # Export all data
        self.export_all()

        # Cleanup old metric values
        try:
            metrics = get_global_metrics()
            metrics.cleanup_old_values()
        except Exception:
            pass

        # End current trace if any
        try:
            trace_id = TraceContext.get_trace_id()
            if trace_id:
                tracer = get_global_tracer()
                tracer.end_trace(trace_id)
        except Exception:
            pass

        self.logger.info("Observability system cleanup complete")


# ============================================================================
# GLOBAL OBSERVABILITY INSTANCE
# ============================================================================

_global_observability: Optional[ObservabilitySystem] = None


def initialize_observability(
    session_id: str,
    service_name: str = "orchestrator",
    log_dir: str = "logs",
    log_level: Optional[str] = None,
    enable_tracing: bool = True,
    enable_metrics: bool = True,
    verbose: bool = False
) -> ObservabilitySystem:
    """
    Initialize the global observability system

    Args:
        session_id: Session ID
        service_name: Service name
        log_dir: Log directory
        log_level: Log level (defaults to env var or INFO)
        enable_tracing: Enable distributed tracing
        enable_metrics: Enable metrics collection
        verbose: Enable verbose logging

    Returns:
        ObservabilitySystem instance
    """
    global _global_observability

    # Get log level from environment if not provided
    if log_level is None:
        log_level = os.environ.get("LOG_LEVEL", "INFO")

    _global_observability = ObservabilitySystem(
        session_id=session_id,
        service_name=service_name,
        log_dir=log_dir,
        log_level=log_level,
        enable_tracing=enable_tracing,
        enable_metrics=enable_metrics,
        verbose=verbose
    )

    return _global_observability


def get_observability() -> Optional[ObservabilitySystem]:
    """Get the global observability system"""
    return _global_observability


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def start_trace(name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Start a new distributed trace"""
    tracer = get_global_tracer()
    return tracer.start_trace(name, metadata)


def end_trace(trace_id: str):
    """End a distributed trace"""
    tracer = get_global_tracer()
    tracer.end_trace(trace_id)


def record_metric(metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
    """Record a metric value"""
    metrics = get_global_metrics()
    metric = metrics.get_metric(metric_name)

    if metric:
        if isinstance(metric, Counter):
            metric.increment(value, labels)
        elif isinstance(metric, Gauge):
            metric.set(value, labels)
        elif isinstance(metric, (Histogram, Timer)):
            metric.observe(value, labels)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """Example usage of the observability system"""

    # Initialize
    obs = initialize_observability(
        session_id="example-session",
        log_level="DEBUG",
        verbose=True
    )

    # Get logger
    logger = get_logger("example")

    # Start a trace
    trace_id = start_trace("example_workflow")

    # Use structured logging
    logger.info("Starting example workflow")

    # Use distributed tracing
    with traced_span("step_1", kind=SpanKind.AGENT) as span:
        span.set_attribute("agent.name", "example_agent")
        logger.info("Executing step 1")

        # Log agent activity
        obs.orchestration_logger.log_agent_initialized("example_agent", ["capability_1"])
        obs.orchestration_logger.log_agent_ready("example_agent")

        # Simulate some work
        import time
        time.sleep(0.1)

        span.add_event("step_1_completed")

    # Use intelligence logging
    obs.intelligence_logger.log_intent_classification(
        message="Create a Jira issue",
        detected_intents=["CREATE", "JIRA"],
        confidence_scores={"CREATE": 0.95, "JIRA": 0.90},
        classification_method="keyword",
        duration_ms=5.2,
        cache_hit=False
    )

    # Record metrics
    metrics = get_global_metrics()
    counter = metrics.get_metric("requests_total")
    if counter:
        counter.increment()

    # End trace
    end_trace(trace_id)

    # Export all data
    obs.export_all()

    # Cleanup
    obs.cleanup()

    print("\n✓ Observability example complete!")
    print(f"  Check '{obs.log_dir}' for exported data")
