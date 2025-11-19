"""
Metrics Aggregator

Centralized metrics collection, aggregation, and reporting.
Provides real-time and historical metrics for system monitoring.

Features:
- Counter metrics
- Gauge metrics
- Histogram metrics
- Timer metrics
- Percentile calculations
- Metric export (JSON, Prometheus format)
- Time-series data
- Aggregation windows

Author: AI System
Version: 1.0
"""

import time
import json
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections import defaultdict, deque
from enum import Enum
import statistics


# ============================================================================
# METRIC TYPES
# ============================================================================

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"           # Monotonically increasing value
    GAUGE = "gauge"               # Arbitrary value that can go up or down
    HISTOGRAM = "histogram"       # Distribution of values
    TIMER = "timer"               # Time duration measurements


# ============================================================================
# METRIC DATA STRUCTURES
# ============================================================================

@dataclass
class MetricValue:
    """A single metric value with timestamp"""
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Metric:
    """Base metric class"""
    name: str
    type: MetricType
    description: str
    unit: str = ""
    values: List[MetricValue] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)

    def add_value(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Add a value to the metric"""
        metric_value = MetricValue(
            value=value,
            timestamp=time.time(),
            labels={**self.labels, **(labels or {})}
        )
        self.values.append(metric_value)

    def get_latest_value(self) -> Optional[float]:
        """Get the most recent value"""
        if self.values:
            return self.values[-1].value
        return None

    def get_values_in_window(self, window_seconds: float) -> List[MetricValue]:
        """Get values within a time window"""
        cutoff = time.time() - window_seconds
        return [v for v in self.values if v.timestamp >= cutoff]


class Counter(Metric):
    """Counter metric - monotonically increasing"""

    def __init__(self, name: str, description: str, unit: str = "", labels: Optional[Dict[str, str]] = None):
        super().__init__(
            name=name,
            type=MetricType.COUNTER,
            description=description,
            unit=unit,
            labels=labels or {}
        )
        self._current_value = 0.0

    def increment(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment counter"""
        self._current_value += value
        self.add_value(self._current_value, labels)

    def get_total(self) -> float:
        """Get total count"""
        return self._current_value


class Gauge(Metric):
    """Gauge metric - arbitrary value"""

    def __init__(self, name: str, description: str, unit: str = "", labels: Optional[Dict[str, str]] = None):
        super().__init__(
            name=name,
            type=MetricType.GAUGE,
            description=description,
            unit=unit,
            labels=labels or {}
        )

    def set(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Set gauge value"""
        self.add_value(value, labels)


class Histogram(Metric):
    """Histogram metric - distribution of values"""

    def __init__(
        self,
        name: str,
        description: str,
        unit: str = "",
        buckets: Optional[List[float]] = None,
        labels: Optional[Dict[str, str]] = None
    ):
        super().__init__(
            name=name,
            type=MetricType.HISTOGRAM,
            description=description,
            unit=unit,
            labels=labels or {}
        )
        self.buckets = buckets or [0.1, 0.5, 1, 2.5, 5, 10, 25, 50, 100]

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a value"""
        self.add_value(value, labels)

    def get_percentile(self, percentile: float, window_seconds: Optional[float] = None) -> Optional[float]:
        """Calculate percentile"""
        if window_seconds:
            values = [v.value for v in self.get_values_in_window(window_seconds)]
        else:
            values = [v.value for v in self.values]

        if not values:
            return None

        return statistics.quantiles(values, n=100)[int(percentile) - 1] if len(values) > 1 else values[0]

    def get_statistics(self, window_seconds: Optional[float] = None) -> Dict[str, float]:
        """Get statistical summary"""
        if window_seconds:
            values = [v.value for v in self.get_values_in_window(window_seconds)]
        else:
            values = [v.value for v in self.values]

        if not values:
            return {}

        return {
            'count': len(values),
            'sum': sum(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'stddev': statistics.stdev(values) if len(values) > 1 else 0,
            'p50': self.get_percentile(50, window_seconds),
            'p90': self.get_percentile(90, window_seconds),
            'p95': self.get_percentile(95, window_seconds),
            'p99': self.get_percentile(99, window_seconds),
        }


class Timer(Histogram):
    """Timer metric - specialized histogram for durations"""

    def __init__(self, name: str, description: str, labels: Optional[Dict[str, str]] = None):
        super().__init__(
            name=name,
            description=description,
            unit="ms",
            labels=labels
        )

    def time_function(self):
        """Context manager for timing function execution"""
        return TimerContext(self)


class TimerContext:
    """Context manager for timing operations"""

    def __init__(self, timer: Timer, labels: Optional[Dict[str, str]] = None):
        self.timer = timer
        self.labels = labels
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        self.timer.observe(duration_ms, self.labels)
        return False


# ============================================================================
# METRICS REGISTRY
# ============================================================================

class MetricsAggregator:
    """
    Centralized metrics aggregation and reporting

    Features:
    - Register and manage metrics
    - Thread-safe metric updates
    - Metric export (JSON, Prometheus)
    - Real-time aggregation
    - Historical data retention
    """

    def __init__(
        self,
        retention_window_seconds: float = 3600,  # 1 hour
        export_dir: Optional[str] = "logs/metrics",
        verbose: bool = False
    ):
        """
        Initialize metrics aggregator

        Args:
            retention_window_seconds: How long to keep metric values
            export_dir: Directory for metric exports
            verbose: Enable verbose logging
        """
        self.retention_window_seconds = retention_window_seconds
        self.export_dir = Path(export_dir) if export_dir else None
        self.verbose = verbose

        # Metrics storage
        self._metrics: Dict[str, Metric] = {}
        self._lock = threading.Lock()

        # Create export directory
        if self.export_dir:
            self.export_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # METRIC REGISTRATION
    # ========================================================================

    def register_counter(
        self,
        name: str,
        description: str,
        unit: str = "",
        labels: Optional[Dict[str, str]] = None
    ) -> Counter:
        """Register a counter metric"""
        with self._lock:
            if name in self._metrics:
                metric = self._metrics[name]
                if not isinstance(metric, Counter):
                    raise ValueError(f"Metric {name} already exists as {type(metric).__name__}")
                return metric

            counter = Counter(name, description, unit, labels)
            self._metrics[name] = counter
            return counter

    def register_gauge(
        self,
        name: str,
        description: str,
        unit: str = "",
        labels: Optional[Dict[str, str]] = None
    ) -> Gauge:
        """Register a gauge metric"""
        with self._lock:
            if name in self._metrics:
                metric = self._metrics[name]
                if not isinstance(metric, Gauge):
                    raise ValueError(f"Metric {name} already exists as {type(metric).__name__}")
                return metric

            gauge = Gauge(name, description, unit, labels)
            self._metrics[name] = gauge
            return gauge

    def register_histogram(
        self,
        name: str,
        description: str,
        unit: str = "",
        buckets: Optional[List[float]] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> Histogram:
        """Register a histogram metric"""
        with self._lock:
            if name in self._metrics:
                metric = self._metrics[name]
                if not isinstance(metric, Histogram):
                    raise ValueError(f"Metric {name} already exists as {type(metric).__name__}")
                return metric

            histogram = Histogram(name, description, unit, buckets, labels)
            self._metrics[name] = histogram
            return histogram

    def register_timer(
        self,
        name: str,
        description: str,
        labels: Optional[Dict[str, str]] = None
    ) -> Timer:
        """Register a timer metric"""
        with self._lock:
            if name in self._metrics:
                metric = self._metrics[name]
                if not isinstance(metric, Timer):
                    raise ValueError(f"Metric {name} already exists as {type(metric).__name__}")
                return metric

            timer = Timer(name, description, labels)
            self._metrics[name] = timer
            return timer

    # ========================================================================
    # METRIC ACCESS
    # ========================================================================

    def get_metric(self, name: str) -> Optional[Metric]:
        """Get a metric by name"""
        with self._lock:
            return self._metrics.get(name)

    def get_all_metrics(self) -> Dict[str, Metric]:
        """Get all metrics"""
        with self._lock:
            return dict(self._metrics)

    # ========================================================================
    # CLEANUP
    # ========================================================================

    def cleanup_old_values(self):
        """Remove old metric values outside retention window"""
        cutoff = time.time() - self.retention_window_seconds

        with self._lock:
            for metric in self._metrics.values():
                metric.values = [v for v in metric.values if v.timestamp >= cutoff]

    # ========================================================================
    # EXPORT
    # ========================================================================

    def export_json(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """Export metrics to JSON format"""
        export_data = {
            'timestamp': time.time(),
            'timestamp_iso': datetime.now().isoformat(),
            'metrics': {}
        }

        with self._lock:
            for name, metric in self._metrics.items():
                metric_data = {
                    'name': metric.name,
                    'type': metric.type.value,
                    'description': metric.description,
                    'unit': metric.unit,
                    'labels': metric.labels
                }

                if isinstance(metric, Counter):
                    metric_data['value'] = metric.get_total()
                elif isinstance(metric, Gauge):
                    metric_data['value'] = metric.get_latest_value()
                elif isinstance(metric, (Histogram, Timer)):
                    metric_data['statistics'] = metric.get_statistics(window_seconds=300)  # Last 5 minutes

                export_data['metrics'][name] = metric_data

        # Write to file
        if filepath:
            try:
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2)
            except Exception as e:
                print(f"Failed to export metrics: {e}")

        return export_data

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format"""
        lines = []

        with self._lock:
            for name, metric in self._metrics.items():
                # Sanitize metric name for Prometheus
                prom_name = name.replace('.', '_').replace('-', '_')

                # Add HELP and TYPE
                lines.append(f"# HELP {prom_name} {metric.description}")
                lines.append(f"# TYPE {prom_name} {metric.type.value}")

                if isinstance(metric, Counter):
                    value = metric.get_total()
                    labels_str = self._format_prometheus_labels(metric.labels)
                    lines.append(f"{prom_name}{labels_str} {value}")

                elif isinstance(metric, Gauge):
                    value = metric.get_latest_value()
                    if value is not None:
                        labels_str = self._format_prometheus_labels(metric.labels)
                        lines.append(f"{prom_name}{labels_str} {value}")

                elif isinstance(metric, (Histogram, Timer)):
                    stats = metric.get_statistics(window_seconds=300)
                    if stats:
                        labels_str = self._format_prometheus_labels(metric.labels)
                        lines.append(f"{prom_name}_count{labels_str} {stats['count']}")
                        lines.append(f"{prom_name}_sum{labels_str} {stats['sum']}")
                        lines.append(f"{prom_name}_min{labels_str} {stats['min']}")
                        lines.append(f"{prom_name}_max{labels_str} {stats['max']}")

        return '\n'.join(lines)

    def _format_prometheus_labels(self, labels: Dict[str, str]) -> str:
        """Format labels for Prometheus format"""
        if not labels:
            return ""
        label_pairs = [f'{k}="{v}"' for k, v in labels.items()]
        return "{" + ", ".join(label_pairs) + "}"

    # ========================================================================
    # STANDARD METRICS
    # ========================================================================

    def register_standard_metrics(self):
        """Register standard system metrics"""
        # Request metrics
        self.register_counter("requests_total", "Total number of requests")
        self.register_counter("requests_succeeded", "Total successful requests")
        self.register_counter("requests_failed", "Total failed requests")
        self.register_timer("request_duration", "Request duration")

        # Agent metrics
        self.register_counter("agent_tasks_total", "Total agent tasks")
        self.register_counter("agent_tasks_succeeded", "Successful agent tasks")
        self.register_counter("agent_tasks_failed", "Failed agent tasks")
        self.register_timer("agent_task_duration", "Agent task duration")
        self.register_gauge("active_agents", "Number of active agents")

        # Intelligence metrics
        self.register_timer("intelligence_processing_duration", "Intelligence processing duration")
        self.register_counter("intelligence_cache_hits", "Intelligence cache hits")
        self.register_counter("intelligence_cache_misses", "Intelligence cache misses")
        self.register_histogram("intelligence_confidence_score", "Intelligence confidence scores")

        # System metrics
        self.register_gauge("active_sessions", "Number of active sessions")
        self.register_counter("errors_total", "Total errors")


# ============================================================================
# GLOBAL METRICS AGGREGATOR
# ============================================================================

_global_metrics: Optional[MetricsAggregator] = None
_metrics_lock = threading.Lock()


def initialize_global_metrics(
    retention_window_seconds: float = 3600,
    export_dir: Optional[str] = "logs/metrics",
    verbose: bool = False
):
    """Initialize the global metrics aggregator"""
    global _global_metrics

    with _metrics_lock:
        if _global_metrics is None:
            _global_metrics = MetricsAggregator(
                retention_window_seconds=retention_window_seconds,
                export_dir=export_dir,
                verbose=verbose
            )
            _global_metrics.register_standard_metrics()


def get_global_metrics() -> MetricsAggregator:
    """Get or create the global metrics aggregator"""
    global _global_metrics

    if _global_metrics is None:
        initialize_global_metrics()

    return _global_metrics
