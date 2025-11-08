"""
Agent Orchestration Logger

Specialized logging for agent orchestration and coordination.
Tracks agent lifecycle, state transitions, and inter-agent communication.

Features:
- Agent state machine logging
- Agent health monitoring
- Task routing decisions
- Inter-agent messages
- Resource allocation
- Performance metrics per agent
- Failure tracking and recovery

Author: AI System
Version: 1.0
"""

import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path

from .distributed_tracing import get_global_tracer, traced_span, SpanKind, SpanStatus
from .logging_config import get_logger


# ============================================================================
# AGENT STATES
# ============================================================================

class AgentState(Enum):
    """Agent lifecycle states"""
    UNINITIALIZED = "UNINITIALIZED"
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    BUSY = "BUSY"
    IDLE = "IDLE"
    ERROR = "ERROR"
    RECOVERING = "RECOVERING"
    SHUTTING_DOWN = "SHUTTING_DOWN"
    TERMINATED = "TERMINATED"


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "PENDING"
    ASSIGNED = "ASSIGNED"
    EXECUTING = "EXECUTING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    RETRYING = "RETRYING"
    CANCELLED = "CANCELLED"


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class AgentStateTransition:
    """Records an agent state transition"""
    agent_name: str
    from_state: AgentState
    to_state: AgentState
    timestamp: float
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'agent_name': self.agent_name,
            'from_state': self.from_state.value,
            'to_state': self.to_state.value,
            'timestamp': self.timestamp,
            'timestamp_iso': datetime.fromtimestamp(self.timestamp).isoformat(),
            'reason': self.reason,
            'metadata': self.metadata
        }


@dataclass
class TaskAssignment:
    """Records a task assignment to an agent"""
    task_id: str
    task_name: str
    assigned_agent: str
    status: TaskStatus
    assigned_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def duration_ms(self) -> Optional[float]:
        """Task duration in milliseconds"""
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at) * 1000
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'task_id': self.task_id,
            'task_name': self.task_name,
            'assigned_agent': self.assigned_agent,
            'status': self.status.value,
            'assigned_at': self.assigned_at,
            'assigned_at_iso': datetime.fromtimestamp(self.assigned_at).isoformat(),
            'started_at': self.started_at,
            'started_at_iso': datetime.fromtimestamp(self.started_at).isoformat() if self.started_at else None,
            'completed_at': self.completed_at,
            'completed_at_iso': datetime.fromtimestamp(self.completed_at).isoformat() if self.completed_at else None,
            'duration_ms': self.duration_ms,
            'metadata': self.metadata,
            'errors': self.errors
        }


@dataclass
class AgentRoutingDecision:
    """Records a routing decision"""
    timestamp: float
    task_name: str
    selected_agent: str
    reason: str
    considered_agents: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'timestamp_iso': datetime.fromtimestamp(self.timestamp).isoformat(),
            'task_name': self.task_name,
            'selected_agent': self.selected_agent,
            'reason': self.reason,
            'considered_agents': self.considered_agents,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


@dataclass
class AgentHealthRecord:
    """Health status of an agent"""
    agent_name: str
    timestamp: float
    state: AgentState
    is_healthy: bool
    success_rate: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    active_tasks: int = 0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'agent_name': self.agent_name,
            'timestamp': self.timestamp,
            'timestamp_iso': datetime.fromtimestamp(self.timestamp).isoformat(),
            'state': self.state.value,
            'is_healthy': self.is_healthy,
            'success_rate': self.success_rate,
            'avg_latency_ms': self.avg_latency_ms,
            'active_tasks': self.active_tasks,
            'total_tasks_completed': self.total_tasks_completed,
            'total_tasks_failed': self.total_tasks_failed,
            'errors': self.errors,
            'warnings': self.warnings
        }


# ============================================================================
# ORCHESTRATION LOGGER
# ============================================================================

class OrchestrationLogger:
    """
    Specialized logger for agent orchestration

    Tracks:
    - Agent state transitions
    - Task assignments and execution
    - Routing decisions
    - Agent health
    - Inter-agent communication
    - Resource allocation
    """

    def __init__(
        self,
        session_id: str,
        export_dir: Optional[str] = "logs/orchestration",
        verbose: bool = False
    ):
        """
        Initialize orchestration logger

        Args:
            session_id: Session ID
            export_dir: Directory for exports
            verbose: Enable verbose logging
        """
        self.session_id = session_id
        self.export_dir = Path(export_dir) if export_dir else None
        self.verbose = verbose

        # Get standard logger
        self.logger = get_logger(__name__)

        # Get tracer
        self.tracer = get_global_tracer()

        # Create export directory
        if self.export_dir:
            self.export_dir.mkdir(parents=True, exist_ok=True)

        # State tracking
        self.agent_states: Dict[str, AgentState] = {}
        self.state_transitions: List[AgentStateTransition] = []
        self.task_assignments: Dict[str, TaskAssignment] = {}
        self.routing_decisions: List[AgentRoutingDecision] = []
        self.health_records: Dict[str, AgentHealthRecord] = {}

        # Metrics
        self.start_time = time.time()

    # ========================================================================
    # AGENT LIFECYCLE LOGGING
    # ========================================================================

    def log_agent_initialized(
        self,
        agent_name: str,
        capabilities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log agent initialization"""
        with traced_span(
            f"agent.initialize.{agent_name}",
            kind=SpanKind.AGENT
        ) as span:
            span.set_attribute("agent.name", agent_name)
            span.set_attribute("agent.state", AgentState.INITIALIZING.value)

            if capabilities:
                span.set_attribute("agent.capabilities", capabilities)

            self._transition_state(
                agent_name,
                AgentState.UNINITIALIZED,
                AgentState.INITIALIZING,
                "Agent initialization started",
                metadata or {}
            )

            self.logger.info(
                f"Agent initialized: {agent_name}",
                extra={
                    'agent_name': agent_name,
                    'capabilities': capabilities,
                    'metadata': metadata
                }
            )

    def log_agent_ready(self, agent_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Log agent ready state"""
        with traced_span(
            f"agent.ready.{agent_name}",
            kind=SpanKind.AGENT
        ) as span:
            span.set_attribute("agent.name", agent_name)
            span.set_attribute("agent.state", AgentState.READY.value)

            self._transition_state(
                agent_name,
                AgentState.INITIALIZING,
                AgentState.READY,
                "Agent ready to accept tasks",
                metadata or {}
            )

            span.add_event("agent_ready")

            self.logger.info(
                f"Agent ready: {agent_name}",
                extra={'agent_name': agent_name, 'metadata': metadata}
            )

    def log_agent_error(
        self,
        agent_name: str,
        error: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log agent error"""
        with traced_span(
            f"agent.error.{agent_name}",
            kind=SpanKind.AGENT
        ) as span:
            span.set_attribute("agent.name", agent_name)
            span.set_attribute("agent.state", AgentState.ERROR.value)
            span.set_attribute("error.message", error)
            span.set_status(SpanStatus.ERROR, error)

            current_state = self.agent_states.get(agent_name, AgentState.UNINITIALIZED)
            self._transition_state(
                agent_name,
                current_state,
                AgentState.ERROR,
                f"Error: {error}",
                metadata or {}
            )

            self.logger.error(
                f"Agent error: {agent_name} - {error}",
                extra={
                    'agent_name': agent_name,
                    'error': error,
                    'metadata': metadata
                }
            )

    # ========================================================================
    # TASK LIFECYCLE LOGGING
    # ========================================================================

    def log_task_assigned(
        self,
        task_id: str,
        task_name: str,
        agent_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log task assignment"""
        with traced_span(
            f"task.assign.{task_name}",
            kind=SpanKind.ORCHESTRATION
        ) as span:
            span.set_attribute("task.id", task_id)
            span.set_attribute("task.name", task_name)
            span.set_attribute("agent.name", agent_name)
            span.set_attribute("task.status", TaskStatus.ASSIGNED.value)

            assignment = TaskAssignment(
                task_id=task_id,
                task_name=task_name,
                assigned_agent=agent_name,
                status=TaskStatus.ASSIGNED,
                assigned_at=time.time(),
                metadata=metadata or {}
            )

            self.task_assignments[task_id] = assignment

            # Update agent state
            self._transition_state(
                agent_name,
                self.agent_states.get(agent_name, AgentState.IDLE),
                AgentState.BUSY,
                f"Task assigned: {task_name}",
                {'task_id': task_id}
            )

            self.logger.info(
                f"Task assigned: {task_name} -> {agent_name}",
                extra={
                    'task_id': task_id,
                    'task_name': task_name,
                    'agent_name': agent_name,
                    'metadata': metadata
                }
            )

    def log_task_started(self, task_id: str):
        """Log task execution started"""
        if task_id not in self.task_assignments:
            self.logger.warning(f"Task not found: {task_id}")
            return

        assignment = self.task_assignments[task_id]
        assignment.status = TaskStatus.EXECUTING
        assignment.started_at = time.time()

        with traced_span(
            f"task.execute.{assignment.task_name}",
            kind=SpanKind.AGENT
        ) as span:
            span.set_attribute("task.id", task_id)
            span.set_attribute("task.name", assignment.task_name)
            span.set_attribute("agent.name", assignment.assigned_agent)
            span.set_attribute("task.status", TaskStatus.EXECUTING.value)

            self.logger.info(
                f"Task started: {assignment.task_name}",
                extra={'task_id': task_id, 'agent_name': assignment.assigned_agent}
            )

    def log_task_completed(
        self,
        task_id: str,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log task completion"""
        if task_id not in self.task_assignments:
            self.logger.warning(f"Task not found: {task_id}")
            return

        assignment = self.task_assignments[task_id]
        assignment.completed_at = time.time()
        assignment.status = TaskStatus.SUCCEEDED if success else TaskStatus.FAILED

        if error:
            assignment.errors.append(error)

        if metadata:
            assignment.metadata.update(metadata)

        with traced_span(
            f"task.complete.{assignment.task_name}",
            kind=SpanKind.AGENT
        ) as span:
            span.set_attribute("task.id", task_id)
            span.set_attribute("task.name", assignment.task_name)
            span.set_attribute("agent.name", assignment.assigned_agent)
            span.set_attribute("task.status", assignment.status.value)
            span.set_attribute("task.duration_ms", assignment.duration_ms)
            span.set_attribute("task.success", success)

            if error:
                span.set_status(SpanStatus.ERROR, error)
            else:
                span.set_status(SpanStatus.OK)

            # Update agent state back to IDLE
            self._transition_state(
                assignment.assigned_agent,
                AgentState.BUSY,
                AgentState.IDLE,
                f"Task completed: {assignment.task_name}",
                {'success': success, 'duration_ms': assignment.duration_ms}
            )

            log_msg = f"Task {'succeeded' if success else 'failed'}: {assignment.task_name} ({assignment.duration_ms:.1f}ms)"

            if success:
                self.logger.info(
                    log_msg,
                    extra={
                        'task_id': task_id,
                        'agent_name': assignment.assigned_agent,
                        'duration_ms': assignment.duration_ms,
                        'metadata': metadata
                    }
                )
            else:
                self.logger.error(
                    log_msg,
                    extra={
                        'task_id': task_id,
                        'agent_name': assignment.assigned_agent,
                        'error': error,
                        'duration_ms': assignment.duration_ms,
                        'metadata': metadata
                    }
                )

    # ========================================================================
    # ROUTING DECISIONS
    # ========================================================================

    def log_routing_decision(
        self,
        task_name: str,
        selected_agent: str,
        reason: str,
        considered_agents: Optional[List[str]] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log an agent routing decision"""
        with traced_span(
            f"routing.decision.{task_name}",
            kind=SpanKind.ORCHESTRATION
        ) as span:
            span.set_attribute("task.name", task_name)
            span.set_attribute("routing.selected_agent", selected_agent)
            span.set_attribute("routing.reason", reason)
            span.set_attribute("routing.confidence", confidence or 0.0)

            decision = AgentRoutingDecision(
                timestamp=time.time(),
                task_name=task_name,
                selected_agent=selected_agent,
                reason=reason,
                considered_agents=considered_agents or [],
                confidence=confidence,
                metadata=metadata or {}
            )

            self.routing_decisions.append(decision)

            span.add_event("routing_decision_made", {
                'selected_agent': selected_agent,
                'confidence': confidence
            })

            self.logger.info(
                f"Routing decision: {task_name} -> {selected_agent} (reason: {reason})",
                extra={
                    'task_name': task_name,
                    'selected_agent': selected_agent,
                    'reason': reason,
                    'considered_agents': considered_agents,
                    'confidence': confidence,
                    'metadata': metadata
                }
            )

    # ========================================================================
    # HEALTH MONITORING
    # ========================================================================

    def log_agent_health(
        self,
        agent_name: str,
        is_healthy: bool,
        metrics: Optional[Dict[str, Any]] = None
    ):
        """Log agent health status"""
        health_record = AgentHealthRecord(
            agent_name=agent_name,
            timestamp=time.time(),
            state=self.agent_states.get(agent_name, AgentState.UNINITIALIZED),
            is_healthy=is_healthy,
            **(metrics or {})
        )

        self.health_records[agent_name] = health_record

        if self.verbose or not is_healthy:
            status = "healthy" if is_healthy else "unhealthy"
            self.logger.info(
                f"Agent health: {agent_name} is {status}",
                extra=health_record.to_dict()
            )

    # ========================================================================
    # INTERNAL METHODS
    # ========================================================================

    def _transition_state(
        self,
        agent_name: str,
        from_state: AgentState,
        to_state: AgentState,
        reason: str,
        metadata: Dict[str, Any]
    ):
        """Record a state transition"""
        transition = AgentStateTransition(
            agent_name=agent_name,
            from_state=from_state,
            to_state=to_state,
            timestamp=time.time(),
            reason=reason,
            metadata=metadata
        )

        self.state_transitions.append(transition)
        self.agent_states[agent_name] = to_state

        if self.verbose:
            self.logger.debug(
                f"State transition: {agent_name} {from_state.value} -> {to_state.value}",
                extra=transition.to_dict()
            )

    # ========================================================================
    # EXPORT & REPORTING
    # ========================================================================

    def export_session_summary(self) -> Dict[str, Any]:
        """Export orchestration summary"""
        summary = {
            'session_id': self.session_id,
            'start_time': self.start_time,
            'start_time_iso': datetime.fromtimestamp(self.start_time).isoformat(),
            'duration_ms': (time.time() - self.start_time) * 1000,
            'state_transitions': [t.to_dict() for t in self.state_transitions],
            'task_assignments': {tid: t.to_dict() for tid, t in self.task_assignments.items()},
            'routing_decisions': [d.to_dict() for d in self.routing_decisions],
            'health_records': {name: h.to_dict() for name, h in self.health_records.items()},
            'statistics': self._calculate_statistics()
        }

        # Export to file
        if self.export_dir:
            filename = f"orchestration_{self.session_id}_{int(self.start_time)}.json"
            filepath = self.export_dir / filename

            try:
                with open(filepath, 'w') as f:
                    json.dump(summary, f, indent=2)
            except Exception as e:
                self.logger.error(f"Failed to export orchestration summary: {e}")

        return summary

    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate orchestration statistics"""
        total_tasks = len(self.task_assignments)
        succeeded_tasks = sum(1 for t in self.task_assignments.values() if t.status == TaskStatus.SUCCEEDED)
        failed_tasks = sum(1 for t in self.task_assignments.values() if t.status == TaskStatus.FAILED)

        durations = [t.duration_ms for t in self.task_assignments.values() if t.duration_ms is not None]
        avg_duration = sum(durations) / len(durations) if durations else 0

        return {
            'total_tasks': total_tasks,
            'succeeded_tasks': succeeded_tasks,
            'failed_tasks': failed_tasks,
            'success_rate': succeeded_tasks / total_tasks if total_tasks > 0 else 0,
            'avg_task_duration_ms': avg_duration,
            'total_state_transitions': len(self.state_transitions),
            'total_routing_decisions': len(self.routing_decisions),
            'agents_count': len(self.agent_states)
        }
