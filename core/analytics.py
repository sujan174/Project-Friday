"""
Usage Analytics and Performance Monitoring

Tracks and analyzes:
- Agent performance (success rate, latency, errors)
- User behavior patterns
- System health metrics
- Usage trends

Author: AI System
Version: 1.0
"""

import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import statistics


@dataclass
class AgentMetrics:
    """Performance metrics for a single agent"""
    agent_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_latency_ms: float = 0.0
    error_counts: Dict[str, int] = field(default_factory=dict)

    # Latency percentiles (tracked separately)
    latencies: List[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 - 1.0)"""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate (0.0 - 1.0)"""
        return 1.0 - self.success_rate

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency in milliseconds"""
        if self.total_calls == 0:
            return 0.0
        return self.total_latency_ms / self.total_calls

    @property
    def p50_latency_ms(self) -> float:
        """Get 50th percentile (median) latency"""
        if not self.latencies:
            return 0.0
        return statistics.median(self.latencies)

    @property
    def p95_latency_ms(self) -> float:
        """Get 95th percentile latency"""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        index = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]

    @property
    def p99_latency_ms(self) -> float:
        """Get 99th percentile latency"""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        index = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]


@dataclass
class SessionMetrics:
    """Metrics for a user session"""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    user_messages: int = 0
    agent_calls: int = 0
    errors_encountered: int = 0
    successful_operations: int = 0

    @property
    def duration_seconds(self) -> float:
        """Get session duration in seconds"""
        end = self.end_time or time.time()
        return end - self.start_time


class AnalyticsCollector:
    """
    Collects and analyzes usage metrics.

    Features:
    - Real-time metric collection
    - Agent performance tracking
    - Session analytics
    - Trend analysis
    - Health monitoring
    """

    def __init__(
        self,
        session_id: str,
        max_latency_samples: int = 1000,
        verbose: bool = False
    ):
        """
        Initialize analytics collector.

        Args:
            session_id: Current session ID
            max_latency_samples: Max latency samples to keep per agent
            verbose: Enable detailed logging
        """
        self.session_id = session_id
        self.max_latency_samples = max_latency_samples
        self.verbose = verbose

        # Agent metrics
        self.agent_metrics: Dict[str, AgentMetrics] = {}

        # Session tracking
        self.current_session = SessionMetrics(
            session_id=session_id,
            start_time=time.time()
        )
        self.session_history: List[SessionMetrics] = []

        # Usage patterns
        self.hourly_usage: Dict[int, int] = defaultdict(int)  # hour -> count
        self.daily_usage: Dict[str, int] = defaultdict(int)   # date -> count

        # Tool/operation tracking
        self.operation_counts: Counter = Counter()
        self.operation_success_counts: Counter = Counter()

        # Error tracking
        self.error_patterns: Counter = Counter()
        self.agent_error_patterns: Dict[str, Counter] = defaultdict(Counter)

    # =========================================================================
    # AGENT METRICS
    # =========================================================================

    def record_agent_call(
        self,
        agent_name: str,
        success: bool,
        latency_ms: float,
        error_message: Optional[str] = None
    ):
        """
        Record an agent call with performance metrics.

        Args:
            agent_name: Name of the agent
            success: Whether call succeeded
            latency_ms: Call duration in milliseconds
            error_message: Error message if failed
        """
        # Initialize metrics for agent if needed
        if agent_name not in self.agent_metrics:
            self.agent_metrics[agent_name] = AgentMetrics(agent_name=agent_name)

        metrics = self.agent_metrics[agent_name]

        # Update counts
        metrics.total_calls += 1
        if success:
            metrics.successful_calls += 1
            self.current_session.successful_operations += 1
        else:
            metrics.failed_calls += 1
            self.current_session.errors_encountered += 1

        # Update latency
        metrics.total_latency_ms += latency_ms
        metrics.latencies.append(latency_ms)

        # Trim latencies to max samples
        if len(metrics.latencies) > self.max_latency_samples:
            metrics.latencies = metrics.latencies[-self.max_latency_samples:]

        # Track errors
        if error_message:
            # Classify error type
            error_type = self._classify_error(error_message)
            metrics.error_counts[error_type] = metrics.error_counts.get(error_type, 0) + 1

            # Track error patterns
            self.error_patterns[error_type] += 1
            self.agent_error_patterns[agent_name][error_type] += 1

        # Update session metrics
        self.current_session.agent_calls += 1

        if self.verbose:
            status = "✓" if success else "✗"
            print(f"[ANALYTICS] {status} {agent_name}: {latency_ms:.0f}ms")

    def _classify_error(self, error_message: str) -> str:
        """Classify error into categories"""
        error_lower = error_message.lower()

        if 'timeout' in error_lower or 'timed out' in error_lower:
            return 'timeout'
        elif 'permission' in error_lower or 'forbidden' in error_lower or '403' in error_lower:
            return 'permission'
        elif 'not found' in error_lower or '404' in error_lower:
            return 'not_found'
        elif 'rate limit' in error_lower or '429' in error_lower:
            return 'rate_limit'
        elif 'network' in error_lower or 'connection' in error_lower:
            return 'network'
        elif 'validation' in error_lower or 'invalid' in error_lower:
            return 'validation'
        else:
            return 'other'

    def get_agent_metrics(self, agent_name: str) -> Optional[AgentMetrics]:
        """Get metrics for a specific agent"""
        return self.agent_metrics.get(agent_name)

    def get_all_agent_metrics(self) -> Dict[str, AgentMetrics]:
        """Get metrics for all agents"""
        return self.agent_metrics.copy()

    # =========================================================================
    # SESSION TRACKING
    # =========================================================================

    def record_user_message(self):
        """Record a user message"""
        self.current_session.user_messages += 1

        # Track usage by hour
        current_hour = datetime.now().hour
        self.hourly_usage[current_hour] += 1

        # Track usage by day
        today = datetime.now().strftime("%Y-%m-%d")
        self.daily_usage[today] += 1

    def record_operation(self, operation_type: str, success: bool):
        """Record an operation execution"""
        self.operation_counts[operation_type] += 1
        if success:
            self.operation_success_counts[operation_type] += 1

    def end_session(self):
        """End current session and archive metrics"""
        self.current_session.end_time = time.time()
        self.session_history.append(self.current_session)

        # Start new session
        self.current_session = SessionMetrics(
            session_id=f"{self.session_id}_cont",
            start_time=time.time()
        )

        if self.verbose:
            print(f"[ANALYTICS] Session ended: {self.current_session.duration_seconds:.0f}s")

    # =========================================================================
    # ANALYTICS AND REPORTING
    # =========================================================================

    def get_agent_ranking(self) -> List[tuple]:
        """
        Get agents ranked by success rate.

        Returns:
            List of (agent_name, success_rate, total_calls) sorted by success rate
        """
        rankings = [
            (
                agent_name,
                metrics.success_rate,
                metrics.total_calls
            )
            for agent_name, metrics in self.agent_metrics.items()
            if metrics.total_calls > 0
        ]

        # Sort by success rate (descending), then by call count
        rankings.sort(key=lambda x: (-x[1], -x[2]))

        return rankings

    def get_slowest_agents(self, limit: int = 5) -> List[tuple]:
        """
        Get slowest agents by average latency.

        Returns:
            List of (agent_name, avg_latency_ms, call_count)
        """
        slowest = [
            (
                agent_name,
                metrics.avg_latency_ms,
                metrics.total_calls
            )
            for agent_name, metrics in self.agent_metrics.items()
            if metrics.total_calls > 0
        ]

        slowest.sort(key=lambda x: -x[1])

        return slowest[:limit]

    def get_most_used_operations(self, limit: int = 10) -> List[tuple]:
        """Get most frequently used operations"""
        return self.operation_counts.most_common(limit)

    def get_most_common_errors(self, limit: int = 10) -> List[tuple]:
        """Get most common error types"""
        return self.error_patterns.most_common(limit)

    def get_usage_by_hour(self) -> Dict[int, int]:
        """Get usage distribution by hour of day"""
        return dict(self.hourly_usage)

    def get_peak_usage_hours(self, top_n: int = 3) -> List[int]:
        """Get peak usage hours"""
        sorted_hours = sorted(
            self.hourly_usage.items(),
            key=lambda x: -x[1]
        )
        return [hour for hour, _ in sorted_hours[:top_n]]

    def get_health_score(self) -> float:
        """
        Calculate overall system health score (0.0 - 1.0).

        Factors:
        - Overall success rate
        - Agent availability
        - Error rate
        """
        if not self.agent_metrics:
            return 1.0

        # Calculate weighted metrics
        total_calls = sum(m.total_calls for m in self.agent_metrics.values())
        if total_calls == 0:
            return 1.0

        # Success rate score (70% weight)
        total_success = sum(m.successful_calls for m in self.agent_metrics.values())
        success_score = total_success / total_calls

        # Agent availability score (20% weight)
        # Agents with 0 calls are considered unavailable
        available_agents = sum(1 for m in self.agent_metrics.values() if m.total_calls > 0)
        total_agents = len(self.agent_metrics)
        availability_score = available_agents / total_agents if total_agents > 0 else 1.0

        # Error diversity score (10% weight)
        # Lower is better - many different errors suggests systemic issues
        error_types = len(self.error_patterns)
        error_diversity_score = 1.0 - min(error_types / 10, 1.0)

        health_score = (
            success_score * 0.7 +
            availability_score * 0.2 +
            error_diversity_score * 0.1
        )

        return health_score

    # =========================================================================
    # REPORTING
    # =========================================================================

    def generate_summary_report(self) -> str:
        """Generate human-readable summary report"""
        lines = [
            "📊 **System Analytics Summary**\n",
            f"Session: {self.session_id}",
            f"Duration: {self.current_session.duration_seconds:.0f}s",
            f"Health Score: {self.get_health_score():.1%}\n"
        ]

        # Agent performance
        lines.append("**Agent Performance:**")
        for agent_name, success_rate, calls in self.get_agent_ranking():
            metrics = self.agent_metrics[agent_name]
            lines.append(
                f"  • {agent_name}: {success_rate:.1%} success "
                f"({calls} calls, {metrics.avg_latency_ms:.0f}ms avg)"
            )
        lines.append("")

        # Usage statistics
        lines.append("**Usage Statistics:**")
        lines.append(f"  • User messages: {self.current_session.user_messages}")
        lines.append(f"  • Agent calls: {self.current_session.agent_calls}")
        lines.append(f"  • Successful operations: {self.current_session.successful_operations}")
        lines.append(f"  • Errors: {self.current_session.errors_encountered}")
        lines.append("")

        # Top operations
        if self.operation_counts:
            lines.append("**Most Used Operations:**")
            for op, count in self.get_most_used_operations(5):
                success = self.operation_success_counts[op]
                success_rate = success / count if count > 0 else 0
                lines.append(f"  • {op}: {count} times ({success_rate:.1%} success)")
            lines.append("")

        # Common errors
        if self.error_patterns:
            lines.append("**Common Errors:**")
            for error_type, count in self.get_most_common_errors(5):
                lines.append(f"  • {error_type}: {count} occurrences")
            lines.append("")

        # Peak usage
        peak_hours = self.get_peak_usage_hours(3)
        if peak_hours:
            lines.append(f"**Peak Usage Hours:** {', '.join(f'{h}:00' for h in peak_hours)}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Export all metrics as dictionary"""
        return {
            'session_id': self.session_id,
            'current_session': asdict(self.current_session),
            'agent_metrics': {
                name: {
                    **asdict(metrics),
                    'latencies': metrics.latencies[-100:]  # Keep last 100 only
                }
                for name, metrics in self.agent_metrics.items()
            },
            'hourly_usage': dict(self.hourly_usage),
            'daily_usage': dict(self.daily_usage),
            'operation_counts': dict(self.operation_counts),
            'error_patterns': dict(self.error_patterns),
            'health_score': self.get_health_score(),
            'timestamp': time.time()
        }

    def save_to_file(self, filepath: str):
        """Save analytics to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        if self.verbose:
            print(f"[ANALYTICS] Saved to {filepath}")
