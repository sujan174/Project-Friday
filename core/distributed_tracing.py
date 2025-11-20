"""
Distributed Tracing System

OpenTelemetry-inspired distributed tracing for agent orchestration.
Tracks requests across multiple agents and systems with parent-child spans.

Features:
- Trace and Span ID generation
- Parent-child span relationships
- Span attributes and events
- Span status tracking
- Automatic timing
- Context propagation
- Trace export to JSON

Industry Standards:
- OpenTelemetry semantic conventions
- W3C Trace Context specification
- Structured event logging

Author: AI System
Version: 1.0
"""

import time
import uuid
import json
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from contextvars import ContextVar
from pathlib import Path


# ============================================================================
# SPAN STATUS
# ============================================================================

class SpanStatus(Enum):
    """Span execution status"""
    UNSET = "UNSET"
    OK = "OK"
    ERROR = "ERROR"


class SpanKind(Enum):
    """Type of span operation"""
    INTERNAL = "INTERNAL"           # Internal operation
    SERVER = "SERVER"               # Server-side handler
    CLIENT = "CLIENT"               # Client-side call
    PRODUCER = "PRODUCER"           # Message producer
    CONSUMER = "CONSUMER"           # Message consumer
    AGENT = "AGENT"                 # Agent execution
    INTELLIGENCE = "INTELLIGENCE"   # Intelligence processing
    ORCHESTRATION = "ORCHESTRATION" # Orchestration logic


# ============================================================================
# SPAN EVENT
# ============================================================================

@dataclass
class SpanEvent:
    """Event that occurred during span execution"""
    name: str
    timestamp: float
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'timestamp': self.timestamp,
            'timestamp_iso': datetime.fromtimestamp(self.timestamp).isoformat(),
            'attributes': self.attributes
        }


# ============================================================================
# SPAN
# ============================================================================

@dataclass
class Span:
    """
    Represents a unit of work in distributed tracing

    Attributes:
        trace_id: ID of the entire trace
        span_id: Unique ID of this span
        parent_span_id: ID of parent span (None for root)
        name: Human-readable operation name
        kind: Type of span
        start_time: When span started (epoch seconds)
        end_time: When span ended
        status: Execution status
        attributes: Key-value metadata
        events: List of events during execution
        errors: List of errors
    """
    trace_id: str
    span_id: str
    name: str
    kind: SpanKind
    start_time: float
    parent_span_id: Optional[str] = None
    end_time: Optional[float] = None
    status: SpanStatus = SpanStatus.UNSET
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def duration_ms(self) -> Optional[float]:
        """Duration in milliseconds"""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None

    @property
    def is_active(self) -> bool:
        """Is span still active"""
        return self.end_time is None

    def set_attribute(self, key: str, value: Any):
        """Set a span attribute"""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add an event to the span"""
        event = SpanEvent(
            name=name,
            timestamp=time.time(),
            attributes=attributes or {}
        )
        self.events.append(event)

    def set_status(self, status: SpanStatus, error: Optional[str] = None):
        """Set span status"""
        self.status = status
        if error:
            self.errors.append(error)

    def end(self, status: Optional[SpanStatus] = None):
        """End the span"""
        self.end_time = time.time()
        if status:
            self.status = status
        elif self.status == SpanStatus.UNSET:
            # Auto-set to OK if not explicitly set
            self.status = SpanStatus.OK if not self.errors else SpanStatus.ERROR

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for export"""
        return {
            'trace_id': self.trace_id,
            'span_id': self.span_id,
            'parent_span_id': self.parent_span_id,
            'name': self.name,
            'kind': self.kind.value,
            'start_time': self.start_time,
            'start_time_iso': datetime.fromtimestamp(self.start_time).isoformat(),
            'end_time': self.end_time,
            'end_time_iso': datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            'duration_ms': self.duration_ms,
            'status': self.status.value,
            'attributes': self.attributes,
            'events': [e.to_dict() for e in self.events],
            'errors': self.errors,
            'is_active': self.is_active
        }


# ============================================================================
# TRACE
# ============================================================================

@dataclass
class Trace:
    """Collection of spans that form a complete operation trace"""
    trace_id: str
    spans: List[Span] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    def add_span(self, span: Span):
        """Add a span to the trace"""
        self.spans.append(span)

    def get_root_span(self) -> Optional[Span]:
        """Get the root span (no parent)"""
        for span in self.spans:
            if span.parent_span_id is None:
                return span
        return None

    def get_child_spans(self, parent_span_id: str) -> List[Span]:
        """Get child spans of a parent"""
        return [s for s in self.spans if s.parent_span_id == parent_span_id]

    def end(self):
        """Mark trace as ended"""
        self.end_time = time.time()

    @property
    def duration_ms(self) -> Optional[float]:
        """Total trace duration"""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Export trace to dictionary"""
        return {
            'trace_id': self.trace_id,
            'start_time': self.start_time,
            'start_time_iso': datetime.fromtimestamp(self.start_time).isoformat(),
            'end_time': self.end_time,
            'end_time_iso': datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            'duration_ms': self.duration_ms,
            'metadata': self.metadata,
            'spans': [s.to_dict() for s in self.spans],
            'span_count': len(self.spans)
        }


# ============================================================================
# TRACE CONTEXT (Thread-safe context propagation)
# ============================================================================

# Context variables for current trace and span
_current_trace_id: ContextVar[Optional[str]] = ContextVar('trace_id', default=None)
_current_span_id: ContextVar[Optional[str]] = ContextVar('span_id', default=None)


class TraceContext:
    """Manages trace context propagation"""

    @staticmethod
    def set_trace_id(trace_id: str):
        """Set current trace ID"""
        _current_trace_id.set(trace_id)

    @staticmethod
    def get_trace_id() -> Optional[str]:
        """Get current trace ID"""
        return _current_trace_id.get()

    @staticmethod
    def set_span_id(span_id: str):
        """Set current span ID"""
        _current_span_id.set(span_id)

    @staticmethod
    def get_span_id() -> Optional[str]:
        """Get current span ID"""
        return _current_span_id.get()

    @staticmethod
    def clear():
        """Clear trace context"""
        _current_trace_id.set(None)
        _current_span_id.set(None)

    @staticmethod
    def get_context() -> Dict[str, Optional[str]]:
        """Get current context"""
        return {
            'trace_id': _current_trace_id.get(),
            'span_id': _current_span_id.get()
        }


# ============================================================================
# TRACER
# ============================================================================

class Tracer:
    """
    Main tracer for creating and managing traces and spans

    Thread-safe tracer that manages trace lifecycle and exports.
    """

    def __init__(self, service_name: str = "orchestrator", export_dir: Optional[str] = None):
        """
        Initialize tracer

        Args:
            service_name: Name of the service being traced
            export_dir: Directory to export traces (None = no export)
        """
        self.service_name = service_name
        self.export_dir = Path(export_dir) if export_dir else None

        # Thread-safe trace storage
        self._traces: Dict[str, Trace] = {}
        self._spans: Dict[str, Span] = {}  # span_id -> Span
        self._lock = threading.Lock()

        # Create export directory
        if self.export_dir:
            self.export_dir.mkdir(parents=True, exist_ok=True)

    def start_trace(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start a new trace

        Args:
            name: Trace name
            metadata: Optional metadata

        Returns:
            Trace ID
        """
        trace_id = self._generate_trace_id()

        trace = Trace(
            trace_id=trace_id,
            metadata=metadata or {}
        )
        trace.metadata['service.name'] = self.service_name
        trace.metadata['trace.name'] = name

        with self._lock:
            self._traces[trace_id] = trace

        # Set in context
        TraceContext.set_trace_id(trace_id)

        return trace_id

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Span:
        """
        Start a new span

        Args:
            name: Span operation name
            kind: Type of span
            parent_span_id: Parent span ID (None = use current context)
            attributes: Initial attributes

        Returns:
            Started span
        """
        trace_id = TraceContext.get_trace_id()
        if not trace_id:
            # Auto-start trace if not in context
            trace_id = self.start_trace(name)

        # Use provided parent or current context
        if parent_span_id is None:
            parent_span_id = TraceContext.get_span_id()

        span_id = self._generate_span_id()

        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            name=name,
            kind=kind,
            start_time=time.time(),
            attributes=attributes or {}
        )

        # Add service name
        span.set_attribute('service.name', self.service_name)

        with self._lock:
            self._spans[span_id] = span
            if trace_id in self._traces:
                self._traces[trace_id].add_span(span)

        # Set current span in context
        TraceContext.set_span_id(span_id)

        return span

    def end_span(self, span: Span, status: Optional[SpanStatus] = None):
        """End a span"""
        span.end(status)

        # Clear from context if it's the current span
        if TraceContext.get_span_id() == span.span_id:
            # Restore parent span to context
            if span.parent_span_id:
                TraceContext.set_span_id(span.parent_span_id)
            else:
                TraceContext.set_span_id(None)

    def end_trace(self, trace_id: str):
        """
        End a trace and export it

        Args:
            trace_id: Trace to end
        """
        with self._lock:
            if trace_id not in self._traces:
                return

            trace = self._traces[trace_id]
            trace.end()

        # Export trace
        if self.export_dir:
            self._export_trace(trace)

        # Clear context if it's the current trace
        if TraceContext.get_trace_id() == trace_id:
            TraceContext.clear()

    def get_span(self, span_id: str) -> Optional[Span]:
        """Get a span by ID"""
        with self._lock:
            return self._spans.get(span_id)

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get a trace by ID"""
        with self._lock:
            return self._traces.get(trace_id)

    def _export_trace(self, trace: Trace):
        """Export trace to JSON file"""
        if not self.export_dir:
            return

        filename = f"trace_{trace.trace_id}_{int(trace.start_time)}.json"
        filepath = self.export_dir / filename

        try:
            with open(filepath, 'w') as f:
                json.dump(trace.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Failed to export trace {trace.trace_id}: {e}")

    def _generate_trace_id(self) -> str:
        """Generate a unique trace ID (W3C format: 32 hex chars)"""
        return uuid.uuid4().hex

    def _generate_span_id(self) -> str:
        """Generate a unique span ID (W3C format: 16 hex chars)"""
        return uuid.uuid4().hex[:16]


# ============================================================================
# SPAN CONTEXT MANAGER
# ============================================================================

class traced_span:
    """
    Context manager for automatic span lifecycle management

    Usage:
        tracer = get_global_tracer()

        with traced_span("operation_name", kind=SpanKind.AGENT) as span:
            span.set_attribute("agent.name", "slack")
            span.add_event("processing_started")
            # Do work
            span.add_event("processing_completed")
    """

    def __init__(
        self,
        name: str,
        tracer: Optional[Tracer] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.tracer = tracer or get_global_tracer()
        self.kind = kind
        self.attributes = attributes
        self.span: Optional[Span] = None

    def __enter__(self) -> Span:
        """Start span"""
        self.span = self.tracer.start_span(
            name=self.name,
            kind=self.kind,
            attributes=self.attributes
        )
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End span"""
        if self.span:
            if exc_type:
                self.span.set_status(
                    SpanStatus.ERROR,
                    error=f"{exc_type.__name__}: {exc_val}"
                )
            else:
                self.span.set_status(SpanStatus.OK)

            self.tracer.end_span(self.span)

        # Don't suppress exceptions
        return False


# ============================================================================
# GLOBAL TRACER
# ============================================================================

_global_tracer: Optional[Tracer] = None
_tracer_lock = threading.Lock()


def initialize_global_tracer(
    service_name: str = "orchestrator",
    export_dir: Optional[str] = "logs/traces"
):
    """Initialize the global tracer"""
    global _global_tracer

    with _tracer_lock:
        if _global_tracer is None:
            _global_tracer = Tracer(
                service_name=service_name,
                export_dir=export_dir
            )


def get_global_tracer() -> Tracer:
    """Get or create the global tracer"""
    global _global_tracer

    if _global_tracer is None:
        initialize_global_tracer()

    return _global_tracer


# ============================================================================
# DECORATOR FOR AUTOMATIC TRACING
# ============================================================================

def trace(name: Optional[str] = None, kind: SpanKind = SpanKind.INTERNAL):
    """
    Decorator to automatically trace function execution

    Usage:
        @trace("my_operation", kind=SpanKind.AGENT)
        def my_function(x, y):
            return x + y
    """
    def decorator(func):
        import functools

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            span_name = name or f"{func.__module__}.{func.__name__}"

            with traced_span(span_name, kind=kind) as span:
                # Add function metadata
                span.set_attribute("code.function", func.__name__)
                span.set_attribute("code.module", func.__module__)

                return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            span_name = name or f"{func.__module__}.{func.__name__}"

            with traced_span(span_name, kind=kind) as span:
                # Add function metadata
                span.set_attribute("code.function", func.__name__)
                span.set_attribute("code.module", func.__module__)

                return await func(*args, **kwargs)

        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
