import os
from typing import Optional, Dict, Any
from pathlib import Path

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


class ObservabilitySystem:
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
        self.session_id = session_id
        self.service_name = service_name
        self.log_dir = Path(log_dir)
        self.verbose = verbose

        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._init_logging(log_level)

        if enable_tracing:
            self._init_tracing()

        if enable_metrics:
            self._init_metrics()

        self.orchestration_logger = OrchestrationLogger(
            session_id=session_id,
            export_dir=str(self.log_dir / "orchestration"),
            verbose=verbose
        )

        self.intelligence_logger = IntelligenceLogger(
            session_id=session_id,
            export_dir=str(self.log_dir / "intelligence"),
            verbose=verbose
        )

        LogContext.set_session(session_id)

        self.logger = get_logger(__name__)
        self.logger.info(f"Observability system initialized for session: {session_id}")

    def _init_logging(self, log_level: str):
        configure_logging({
            'log_level': log_level,
            'log_dir': str(self.log_dir),
            'enable_file_logging': True,
            'enable_json_logging': True,
            'enable_console': True,
            'enable_colors': True,
            'max_file_size_mb': 10,
            'backup_count': 5
        })

    def _init_tracing(self):
        trace_dir = self.log_dir / "traces"
        initialize_global_tracer(
            service_name=self.service_name,
            export_dir=str(trace_dir)
        )

    def _init_metrics(self):
        metrics_dir = self.log_dir / "metrics"
        initialize_global_metrics(
            retention_window_seconds=3600,
            export_dir=str(metrics_dir),
            verbose=self.verbose
        )

    def export_all(self):
        results = {}

        try:
            results['orchestration'] = self.orchestration_logger.export_session_summary()
        except Exception as e:
            self.logger.error(f"Failed to export orchestration data: {e}")

        try:
            results['intelligence'] = self.intelligence_logger.export_session_summary()
        except Exception as e:
            self.logger.error(f"Failed to export intelligence data: {e}")

        try:
            metrics = get_global_metrics()
            metrics_path = self.log_dir / "metrics" / f"metrics_{self.session_id}.json"
            results['metrics'] = metrics.export_json(str(metrics_path))
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")

        return results

    def cleanup(self):
        self.export_all()

        try:
            metrics = get_global_metrics()
            metrics.cleanup_old_values()
        except Exception:
            pass

        try:
            trace_id = TraceContext.get_trace_id()
            if trace_id:
                tracer = get_global_tracer()
                tracer.end_trace(trace_id)
        except Exception:
            pass

        self.logger.info("Observability system cleanup complete")


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
    global _global_observability

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
    return _global_observability


def start_trace(name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    tracer = get_global_tracer()
    return tracer.start_trace(name, metadata)


def end_trace(trace_id: str):
    tracer = get_global_tracer()
    tracer.end_trace(trace_id)


def record_metric(metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
    metrics = get_global_metrics()
    metric = metrics.get_metric(metric_name)

    if metric:
        if isinstance(metric, Counter):
            metric.increment(value, labels)
        elif isinstance(metric, Gauge):
            metric.set(value, labels)
        elif isinstance(metric, (Histogram, Timer)):
            metric.observe(value, labels)


if __name__ == "__main__":
    obs = initialize_observability(
        session_id="example-session",
        log_level="DEBUG",
        verbose=True
    )

    logger = get_logger("example")

    trace_id = start_trace("example_workflow")

    logger.info("Starting example workflow")

    with traced_span("step_1", kind=SpanKind.AGENT) as span:
        span.set_attribute("agent.name", "example_agent")
        logger.info("Executing step 1")

        obs.orchestration_logger.log_agent_initialized("example_agent", ["capability_1"])
        obs.orchestration_logger.log_agent_ready("example_agent")

        import time
        time.sleep(0.1)

        span.add_event("step_1_completed")

    obs.intelligence_logger.log_intent_classification(
        message="Create a Jira issue",
        detected_intents=["CREATE", "JIRA"],
        confidence_scores={"CREATE": 0.95, "JIRA": 0.90},
        classification_method="keyword",
        duration_ms=5.2,
        cache_hit=False
    )

    metrics = get_global_metrics()
    counter = metrics.get_metric("requests_total")
    if counter:
        counter.increment()

    end_trace(trace_id)

    obs.export_all()

    obs.cleanup()

    print("\n✓ Observability example complete!")
    print(f"  Check '{obs.log_dir}' for exported data")
