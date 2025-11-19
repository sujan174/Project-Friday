"""
Agent Logger - Session-Based Logging System

Provides comprehensive logging for agent operations with:
- Session-based file logging (one file per session, rewritten each time)
- Message tracking between orchestrator and agents
- Token consumption metrics
- Tool call tracking
- ASCII visualization of agent interactions
- Condensed but informative output

Author: AI System
Version: 1.0
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict


@dataclass
class ToolCall:
    """Record of a single tool call"""
    tool_name: str
    timestamp: float
    duration: Optional[float] = None
    success: bool = True
    error: Optional[str] = None


@dataclass
class MessageLog:
    """Record of a message between orchestrator and agent"""
    timestamp: float
    direction: str  # "to_agent" or "from_agent"
    agent_name: str
    message_preview: str  # First 100 chars
    message_length: int
    has_function_calls: bool = False
    function_call_count: int = 0


@dataclass
class AgentMetrics:
    """Metrics for a single agent"""
    agent_name: str
    messages_received: int = 0
    messages_sent: int = 0
    tools_called: int = 0
    successful_tools: int = 0
    failed_tools: int = 0
    total_duration: float = 0.0
    tool_calls: List[ToolCall] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class SessionLogger:
    """
    Main session logger for agent operations

    Creates a fresh log file for each session with condensed,
    informative logging of all agent interactions.
    """

    def __init__(self, log_dir: str = "logs", session_id: Optional[str] = None):
        """
        Initialize session logger

        Args:
            log_dir: Directory to store log files
            session_id: Optional session ID (generates one if not provided)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start = time.time()

        # Log file path - one per session, overwrites on restart
        self.log_file = self.log_dir / f"session_{self.session_id}.log"

        # Metrics tracking
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        self.message_log: List[MessageLog] = []
        self.total_tokens: Dict[str, int] = defaultdict(int)  # agent_name -> tokens

        # Initialize log file
        self._initialize_log_file()

    def _initialize_log_file(self):
        """Initialize the log file with session header"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("AGENT SESSION LOG\n")
            f.write("=" * 80 + "\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Log File: {self.log_file}\n")
            f.write("=" * 80 + "\n\n")

    def log_message_to_agent(
        self,
        agent_name: str,
        message: str,
        has_function_calls: bool = False,
        function_call_count: int = 0
    ):
        """
        Log a message sent to an agent

        Args:
            agent_name: Name of the agent receiving the message
            message: The message content
            has_function_calls: Whether message contains function calls
            function_call_count: Number of function calls in message
        """
        # Ensure agent exists in metrics
        if agent_name not in self.agent_metrics:
            self.agent_metrics[agent_name] = AgentMetrics(agent_name=agent_name)

        # Create message log entry
        msg_log = MessageLog(
            timestamp=time.time(),
            direction="to_agent",
            agent_name=agent_name,
            message_preview=message[:100],
            message_length=len(message),
            has_function_calls=has_function_calls,
            function_call_count=function_call_count
        )
        self.message_log.append(msg_log)
        self.agent_metrics[agent_name].messages_received += 1

        # Write to log file
        self._append_to_log(
            f"[{self._format_time(msg_log.timestamp)}] → {agent_name}\n"
            f"  Message: {message[:80]}{'...' if len(message) > 80 else ''}\n"
            f"  Length: {len(message)} chars"
            + (f", Function calls: {function_call_count}" if has_function_calls else "")
            + "\n"
        )

    def log_message_from_agent(
        self,
        agent_name: str,
        message: str,
        success: bool = True,
        error: Optional[str] = None
    ):
        """
        Log a message received from an agent

        Args:
            agent_name: Name of the agent sending the message
            message: The message content
            success: Whether the agent operation was successful
            error: Optional error message
        """
        # Ensure agent exists in metrics
        if agent_name not in self.agent_metrics:
            self.agent_metrics[agent_name] = AgentMetrics(agent_name=agent_name)

        # Create message log entry
        msg_log = MessageLog(
            timestamp=time.time(),
            direction="from_agent",
            agent_name=agent_name,
            message_preview=message[:100],
            message_length=len(message)
        )
        self.message_log.append(msg_log)
        self.agent_metrics[agent_name].messages_sent += 1

        if error:
            self.agent_metrics[agent_name].errors.append(error)

        # Write to log file
        status = "✓" if success else "✗"
        self._append_to_log(
            f"[{self._format_time(msg_log.timestamp)}] ← {agent_name} {status}\n"
            f"  Response: {message[:80]}{'...' if len(message) > 80 else ''}\n"
            f"  Length: {len(message)} chars"
            + (f"\n  Error: {error}" if error else "")
            + "\n"
        )

    def log_tool_call(
        self,
        agent_name: str,
        tool_name: str,
        duration: Optional[float] = None,
        success: bool = True,
        error: Optional[str] = None
    ):
        """
        Log a tool call by an agent

        Args:
            agent_name: Name of the agent calling the tool
            tool_name: Name of the tool being called
            duration: Time taken in seconds
            success: Whether the call succeeded
            error: Optional error message
        """
        # Ensure agent exists in metrics
        if agent_name not in self.agent_metrics:
            self.agent_metrics[agent_name] = AgentMetrics(agent_name=agent_name)

        # Create tool call record
        tool_call = ToolCall(
            tool_name=tool_name,
            timestamp=time.time(),
            duration=duration,
            success=success,
            error=error
        )

        metrics = self.agent_metrics[agent_name]
        metrics.tool_calls.append(tool_call)
        metrics.tools_called += 1

        if success:
            metrics.successful_tools += 1
        else:
            metrics.failed_tools += 1
            if error:
                metrics.errors.append(f"Tool {tool_name}: {error}")

        if duration:
            metrics.total_duration += duration

        # Write to log file
        status = "✓" if success else "✗"
        duration_str = f" ({duration:.2f}s)" if duration else ""
        self._append_to_log(
            f"[{self._format_time(tool_call.timestamp)}] 🔧 {agent_name} → {tool_name} {status}{duration_str}\n"
            + (f"  Error: {error}\n" if error else "")
        )

    def log_tokens(self, agent_name: str, tokens: int, operation: str = ""):
        """
        Log token consumption

        Args:
            agent_name: Name of the agent
            tokens: Number of tokens consumed
            operation: Optional description of operation
        """
        self.total_tokens[agent_name] += tokens

        op_str = f" ({operation})" if operation else ""
        self._append_to_log(
            f"[{self._format_time(time.time())}] 📊 {agent_name}: {tokens:,} tokens{op_str}\n"
        )

    def generate_summary(self):
        """Generate and write session summary to log file"""
        session_duration = time.time() - self.session_start

        summary = [
            "\n" + "=" * 80,
            "SESSION SUMMARY",
            "=" * 80,
            f"Duration: {self._format_duration(session_duration)}",
            f"Total Messages: {len(self.message_log)}",
            ""
        ]

        # Agent metrics
        if self.agent_metrics:
            summary.append("AGENT METRICS:")
            summary.append("-" * 80)

            for agent_name in sorted(self.agent_metrics.keys()):
                metrics = self.agent_metrics[agent_name]
                tokens = self.total_tokens.get(agent_name, 0)

                success_rate = (
                    (metrics.successful_tools / metrics.tools_called * 100)
                    if metrics.tools_called > 0 else 0
                )

                summary.extend([
                    f"\n{agent_name.upper()}:",
                    f"  Messages: {metrics.messages_received} received, {metrics.messages_sent} sent",
                    f"  Tools: {metrics.tools_called} called ({metrics.successful_tools} ✓, {metrics.failed_tools} ✗)",
                    f"  Success Rate: {success_rate:.1f}%",
                    f"  Duration: {self._format_duration(metrics.total_duration)}",
                ])

                if tokens > 0:
                    summary.append(f"  Tokens: {tokens:,}")

                if metrics.errors:
                    summary.append(f"  Errors: {len(metrics.errors)}")

        # Total tokens
        total_tokens = sum(self.total_tokens.values())
        if total_tokens > 0:
            summary.extend([
                "",
                "TOKEN CONSUMPTION:",
                "-" * 80,
                f"Total: {total_tokens:,} tokens",
            ])
            for agent_name, tokens in sorted(self.total_tokens.items(), key=lambda x: x[1], reverse=True):
                percentage = (tokens / total_tokens * 100) if total_tokens > 0 else 0
                summary.append(f"  {agent_name}: {tokens:,} ({percentage:.1f}%)")

        # Message flow visualization
        if self.message_log:
            summary.extend([
                "",
                "MESSAGE FLOW:",
                "-" * 80,
                self._generate_message_flow_graph()
            ])

        # Tool usage summary
        if any(m.tools_called > 0 for m in self.agent_metrics.values()):
            summary.extend([
                "",
                "TOOL USAGE:",
                "-" * 80,
            ])

            tool_counts = defaultdict(int)
            for metrics in self.agent_metrics.values():
                for tool_call in metrics.tool_calls:
                    tool_counts[tool_call.tool_name] += 1

            for tool_name, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True):
                summary.append(f"  {tool_name}: {count} call(s)")

        summary.append("\n" + "=" * 80 + "\n")

        # Write to log
        self._append_to_log("\n".join(summary))

    def _generate_message_flow_graph(self) -> str:
        """Generate ASCII visualization of message flow"""
        if not self.message_log:
            return "  No messages"

        lines = []
        agent_positions = {}
        agents = sorted(set(msg.agent_name for msg in self.message_log))

        # Create header
        header = "  Time      "
        for i, agent in enumerate(agents):
            agent_positions[agent] = i
            header += f"  {agent[:10]:<10}"
        lines.append(header)
        lines.append("  " + "-" * (10 + len(agents) * 12))

        # Add message flow (condensed - max 20 entries)
        messages_to_show = self.message_log[-20:] if len(self.message_log) > 20 else self.message_log

        for msg in messages_to_show:
            time_str = self._format_time(msg.timestamp, short=True)
            line = f"  {time_str}  "

            # Create visual representation
            for agent in agents:
                if agent == msg.agent_name:
                    symbol = "→" if msg.direction == "to_agent" else "←"
                    line += f"  {symbol:<10}"
                else:
                    line += "  " + " " * 10

            lines.append(line)

        if len(self.message_log) > 20:
            lines.append(f"  ... ({len(self.message_log) - 20} earlier messages)")

        return "\n".join(lines)

    def _append_to_log(self, content: str):
        """Append content to log file"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(content)

    def _format_time(self, timestamp: float, short: bool = False) -> str:
        """Format timestamp relative to session start"""
        elapsed = timestamp - self.session_start
        if short:
            return f"{elapsed:6.1f}s"
        return f"+{elapsed:.2f}s"

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins}m {secs:.1f}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"

    def get_log_path(self) -> str:
        """Get the path to the current log file"""
        return str(self.log_file)

    def close(self):
        """Close the logger and write final summary"""
        self.generate_summary()


class AgentLoggerMixin:
    """
    Mixin class to add logging capabilities to agents

    Usage:
        class MyAgent(BaseAgent, AgentLoggerMixin):
            def __init__(self, ...):
                self.logger = session_logger  # Shared session logger
                self.agent_name = "my_agent"
    """

    def log_received_message(self, message: str, has_function_calls: bool = False, function_call_count: int = 0):
        """Log a message received from orchestrator"""
        if hasattr(self, 'logger') and self.logger:
            self.logger.log_message_to_agent(
                self.agent_name,
                message,
                has_function_calls,
                function_call_count
            )

    def log_sent_message(self, message: str, success: bool = True, error: Optional[str] = None):
        """Log a message sent to orchestrator"""
        if hasattr(self, 'logger') and self.logger:
            self.logger.log_message_from_agent(
                self.agent_name,
                message,
                success,
                error
            )

    def log_tool_execution(self, tool_name: str, duration: Optional[float] = None, success: bool = True, error: Optional[str] = None):
        """Log a tool execution"""
        if hasattr(self, 'logger') and self.logger:
            self.logger.log_tool_call(
                self.agent_name,
                tool_name,
                duration,
                success,
                error
            )

    def log_token_usage(self, tokens: int, operation: str = ""):
        """Log token consumption"""
        if hasattr(self, 'logger') and self.logger:
            self.logger.log_tokens(
                self.agent_name,
                tokens,
                operation
            )
