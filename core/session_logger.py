"""
Comprehensive Session Logger - Production Grade

Complete logging of ALL session activities to both JSON and text formats.
Every decision, every action, every error logged for debugging and analysis.

Features:
- Dual format logging (JSON + Text)
- All log entry types from documentation
- Session statistics and summaries
- Thread-safe operations
- Rich metadata tracking

Author: AI System (Senior Developer)
Version: 2.0 - Complete Implementation
"""

import os
import json
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum


class LogEntryType(Enum):
    """Types of log entries"""
    USER_MESSAGE = "user_message"
    ASSISTANT_RESPONSE = "assistant_response"
    INTELLIGENCE_CLASSIFICATION = "intelligence_classification"
    CONFIDENCE_SCORING = "confidence_scoring"
    RISK_ASSESSMENT = "risk_assessment"
    CONTEXT_RESOLUTION = "context_resolution"
    AGENT_CALL = "agent_call"
    FUNCTION_CALL = "function_call"
    MEMORY_UPDATE = "memory_update"
    CONTEXT_UPDATE = "context_update"
    INTELLIGENCE_PROCESSING = "intelligence_processing"
    ERROR = "error"
    SYSTEM = "system"


@dataclass
class LogEntry:
    """Single log entry"""
    timestamp: float
    entry_type: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'entry_type': self.entry_type,
            'data': self.data,
            'metadata': self.metadata
        }


class SessionLogger:
    """
    Comprehensive session logger with dual-format output.

    Logs ALL session activities to:
    - logs/session_{session_id}.json (machine-readable)
    - logs/session_{session_id}.txt (human-readable)

    This is the most detailed logging component - everything in one place per session.
    """

    def __init__(self, log_dir: str = "logs", session_id: Optional[str] = None):
        """
        Initialize comprehensive session logger

        Args:
            log_dir: Directory to store log files
            session_id: Optional session ID (generates one if not provided)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)

        self.session_id = session_id or self._generate_session_id()
        self.session_start = time.time()

        # Log file paths
        self.json_log_file = self.log_dir / f"session_{self.session_id}.json"
        self.text_log_file = self.log_dir / f"session_{self.session_id}.txt"

        # In-memory entries buffer
        self.entries: List[LogEntry] = []
        self._lock = threading.RLock()

        # Statistics
        self.entry_counts: Dict[str, int] = {}

        # Initialize log files
        self._initialize_text_log()

    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    def _initialize_text_log(self):
        """Initialize text log file with header"""
        header = f"""{'='*80}
SESSION LOG: {self.session_id}
Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

"""
        with open(self.text_log_file, 'w', encoding='utf-8') as f:
            f.write(header)

    def _add_entry(self, entry_type: LogEntryType, data: Dict[str, Any], metadata: Optional[Dict] = None):
        """Add log entry to buffer"""
        with self._lock:
            entry = LogEntry(
                timestamp=time.time(),
                entry_type=entry_type.value,
                data=data,
                metadata=metadata or {}
            )
            self.entries.append(entry)

            # Update statistics
            self.entry_counts[entry_type.value] = self.entry_counts.get(entry_type.value, 0) + 1

            # Write to text log immediately
            self._write_text_entry(entry)

    def _write_text_entry(self, entry: LogEntry):
        """Write entry to text log file"""
        time_str = datetime.fromtimestamp(entry.timestamp).strftime('%H:%M:%S')

        # Format based on entry type
        formatted = self._format_text_entry(entry)

        with open(self.text_log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n[{time_str}] {formatted}\n")

    def _format_text_entry(self, entry: LogEntry) -> str:
        """Format entry for text log"""
        entry_type = entry.entry_type
        data = entry.data

        if entry_type == LogEntryType.USER_MESSAGE.value:
            return f"""USER_MESSAGE
{'─'*80}
USER: {data.get('message', '')[:200]}{'...' if len(data.get('message', '')) > 200 else ''}
Length: {data.get('length', 0)} chars"""

        elif entry_type == LogEntryType.ASSISTANT_RESPONSE.value:
            return f"""ASSISTANT_RESPONSE
{'─'*80}
ASSISTANT: {data.get('response', '')[:200]}{'...' if len(data.get('response', '')) > 200 else ''}
Length: {data.get('length', 0)} chars"""

        elif entry_type == LogEntryType.INTELLIGENCE_CLASSIFICATION.value:
            return f"""INTELLIGENCE_CLASSIFICATION
{'─'*80}
🧠 INTELLIGENCE CLASSIFICATION:
  Path Used: {data.get('path_used', 'unknown')}
  Latency: {data.get('latency_ms', 0):.1f}ms
  Confidence: {data.get('confidence', 0):.2f}
  Intents: {data.get('intents', [])}
  Entities: {data.get('entity_count', 0)} found
  Reasoning: {data.get('reasoning', 'N/A')}"""

        elif entry_type == LogEntryType.CONFIDENCE_SCORING.value:
            return f"""CONFIDENCE_SCORING
{'─'*80}
📊 CONFIDENCE SCORING:
  Overall Score: {data.get('overall_score', 0):.2f}
  Intent Clarity: {data.get('intent_clarity', 0):.2f}
  Entity Clarity: {data.get('entity_clarity', 0):.2f}
  Ambiguity: {data.get('ambiguity', 0):.2f}
  Recommendation: {data.get('recommendation', 'unknown')}"""

        elif entry_type == LogEntryType.RISK_ASSESSMENT.value:
            return f"""RISK_ASSESSMENT
{'─'*80}
⚠️  RISK ASSESSMENT:
  Risk Level: {data.get('risk_level', 'UNKNOWN')}
  Reason: {data.get('reason', 'N/A')}"""

        elif entry_type == LogEntryType.AGENT_CALL.value:
            success_icon = "✅" if data.get('success', False) else "❌"
            return f"""AGENT_CALL
{'─'*80}
{success_icon} AGENT CALL: {data.get('agent_name', 'unknown')}
  Instruction: {data.get('instruction', '')[:100]}...
  Duration: {data.get('duration_ms', 0):.1f}ms
  Success: {data.get('success', False)}
  {'Error: ' + data.get('error', '') if not data.get('success', False) else ''}"""

        elif entry_type == LogEntryType.FUNCTION_CALL.value:
            return f"""FUNCTION_CALL
{'─'*80}
🔧 {data.get('function_name', 'unknown')}
  Arguments: {json.dumps(data.get('arguments', {}), indent=2)[:200]}
  Success: {data.get('success', False)}
  Duration: {data.get('duration_ms', 0):.1f}ms"""

        elif entry_type == LogEntryType.ERROR.value:
            return f"""ERROR
{'─'*80}
❌ ERROR: {data.get('error_type', 'unknown')}
  Message: {data.get('error', 'N/A')}
  Context: {json.dumps(data.get('context', {}), indent=2)[:200]}"""

        elif entry_type == LogEntryType.SYSTEM.value:
            return f"""SYSTEM
{'─'*80}
⚙️  {data.get('event', 'unknown event')}
  {json.dumps({k: v for k, v in data.items() if k != 'event'}, indent=2)[:200]}"""

        else:
            # Generic format
            return f"""{entry_type.upper()}
{'─'*80}
{json.dumps(data, indent=2)[:300]}"""

    # ==========================================================================
    # PUBLIC LOGGING METHODS
    # ==========================================================================

    def log_user_message(self, message: str, metadata: Optional[Dict] = None):
        """Log user input message"""
        self._add_entry(
            LogEntryType.USER_MESSAGE,
            {'message': message, 'length': len(message)},
            metadata
        )

    def log_assistant_response(self, response: str, metadata: Optional[Dict] = None):
        """Log assistant output"""
        self._add_entry(
            LogEntryType.ASSISTANT_RESPONSE,
            {'response': response, 'length': len(response)},
            metadata
        )

    def log_intelligence_classification(
        self,
        path_used: str,
        latency_ms: float,
        confidence: float,
        intents: List[Any],
        entities: List[Any],
        reasoning: str,
        metadata: Optional[Dict] = None
    ):
        """Log hybrid intelligence classification results"""
        self._add_entry(
            LogEntryType.INTELLIGENCE_CLASSIFICATION,
            {
                'path_used': path_used,
                'latency_ms': latency_ms,
                'confidence': confidence,
                'intents': [str(i) for i in intents],
                'entity_count': len(entities),
                'entities': [str(e) for e in entities][:10],  # First 10
                'reasoning': reasoning
            },
            metadata
        )

    def log_confidence_scoring(
        self,
        overall_score: float,
        intent_clarity: float,
        entity_clarity: float,
        ambiguity: float,
        recommendation: str,
        details: Optional[Dict] = None
    ):
        """Log confidence scoring details"""
        data = {
            'overall_score': overall_score,
            'intent_clarity': intent_clarity,
            'entity_clarity': entity_clarity,
            'ambiguity': ambiguity,
            'recommendation': recommendation
        }
        if details:
            data.update(details)

        self._add_entry(LogEntryType.CONFIDENCE_SCORING, data)

    def log_risk_assessment(
        self,
        risk_level: str,
        reason: str,
        intents: Optional[List] = None
    ):
        """Log risk assessment results"""
        self._add_entry(
            LogEntryType.RISK_ASSESSMENT,
            {
                'risk_level': risk_level,
                'reason': reason,
                'intents': [str(i) for i in (intents or [])]
            }
        )

    def log_context_resolution(self, resolutions: List[Dict]):
        """Log reference resolutions from context manager"""
        self._add_entry(
            LogEntryType.CONTEXT_RESOLUTION,
            {'resolutions': resolutions}
        )

    def log_agent_call(
        self,
        agent_name: str,
        instruction: str,
        response: str,
        duration: float,
        success: bool,
        error: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """Log agent execution details"""
        self._add_entry(
            LogEntryType.AGENT_CALL,
            {
                'agent_name': agent_name,
                'instruction': instruction,
                'instruction_length': len(instruction),
                'response': response,
                'response_length': len(response),
                'duration_ms': duration * 1000,
                'success': success,
                'error': error
            },
            metadata
        )

    def log_function_call(
        self,
        function_name: str,
        arguments: Dict,
        result: Any,
        success: bool,
        duration: float,
        error: Optional[str] = None
    ):
        """Log LLM function calling details"""
        self._add_entry(
            LogEntryType.FUNCTION_CALL,
            {
                'function_name': function_name,
                'arguments': arguments,
                'result': str(result)[:200] if result else None,
                'success': success,
                'duration_ms': duration * 1000,
                'error': error
            }
        )

    def log_memory_update(self, key: str, value: Any, operation: str):
        """Log changes to workspace knowledge"""
        self._add_entry(
            LogEntryType.MEMORY_UPDATE,
            {
                'key': key,
                'value': str(value)[:200],
                'operation': operation
            }
        )

    def log_context_update(self, context_data: Dict):
        """Log changes to conversation context"""
        self._add_entry(
            LogEntryType.CONTEXT_UPDATE,
            context_data
        )

    def log_intelligence_processing(
        self,
        stage: str,
        input_data: Dict,
        output_data: Dict,
        duration_ms: float,
        success: bool
    ):
        """Log generic intelligence stage processing"""
        self._add_entry(
            LogEntryType.INTELLIGENCE_PROCESSING,
            {
                'stage': stage,
                'input': str(input_data)[:200],
                'output': str(output_data)[:200],
                'duration_ms': duration_ms,
                'success': success
            }
        )

    def log_error(
        self,
        error: str,
        error_type: str,
        context: Optional[Dict] = None,
        traceback: Optional[str] = None
    ):
        """Log error occurred during processing"""
        self._add_entry(
            LogEntryType.ERROR,
            {
                'error': error,
                'error_type': error_type,
                'context': context or {},
                'traceback': traceback
            }
        )

    def log_system_event(self, event: str, **kwargs):
        """Log system events"""
        data = {'event': event}
        data.update(kwargs)
        self._add_entry(LogEntryType.SYSTEM, data)

    # Alias methods for compatibility with existing code
    def log_message_to_agent(self, agent_name: str, message: str):
        """Compatibility method"""
        self.log_system_event(
            'message_to_agent',
            agent_name=agent_name,
            message_preview=message[:100],
            message_length=len(message)
        )

    def log_message_from_agent(
        self,
        agent_name: str,
        message: str,
        success: bool = True,
        error: Optional[str] = None
    ):
        """Compatibility method"""
        self.log_system_event(
            'message_from_agent',
            agent_name=agent_name,
            message_preview=message[:100],
            message_length=len(message),
            success=success,
            error=error
        )

    # ==========================================================================
    # SESSION SUMMARY & CLOSE
    # ==========================================================================

    def get_summary(self) -> Dict[str, Any]:
        """Get session statistics"""
        duration = time.time() - self.session_start

        return {
            'session_id': self.session_id,
            'started_at': self.session_start,
            'duration_seconds': duration,
            'entry_count': len(self.entries),
            'entry_types': dict(self.entry_counts),
            'files': {
                'json': str(self.json_log_file),
                'text': str(self.text_log_file)
            }
        }

    def close(self):
        """Close logger, write summary, and save JSON"""
        with self._lock:
            # Write session summary to text log
            self._write_summary()

            # Write all entries to JSON
            self._write_json_log()

    def _write_summary(self):
        """Write session summary to text log"""
        summary = self.get_summary()
        duration = summary['duration_seconds']

        summary_text = f"""

{'='*80}
SESSION SUMMARY
{'='*80}
Duration: {duration:.1f}s
Total Entries: {summary['entry_count']}

Entry Types:
"""

        for entry_type, count in sorted(summary['entry_types'].items()):
            summary_text += f"  {entry_type}: {count}\n"

        # Extract agent calls
        agent_calls = {}
        for entry in self.entries:
            if entry.entry_type == LogEntryType.AGENT_CALL.value:
                agent_name = entry.data.get('agent_name', 'unknown')
                agent_calls[agent_name] = agent_calls.get(agent_name, 0) + 1

        if agent_calls:
            summary_text += "\nAgent Calls:\n"
            for agent, count in sorted(agent_calls.items()):
                summary_text += f"  {agent}: {count}\n"

        # Error count
        error_count = self.entry_counts.get(LogEntryType.ERROR.value, 0)
        summary_text += f"\nErrors: {error_count}\n"
        summary_text += "="*80 + "\n"

        with open(self.text_log_file, 'a', encoding='utf-8') as f:
            f.write(summary_text)

    def _write_json_log(self):
        """Write complete session to JSON file"""
        summary = self.get_summary()

        json_data = {
            'metadata': {
                'session_id': self.session_id,
                'started_at': self.session_start,
                'ended_at': time.time(),
                'duration_seconds': summary['duration_seconds'],
                'entry_count': len(self.entries)
            },
            'entries': [entry.to_dict() for entry in self.entries]
        }

        with open(self.json_log_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, default=str)

    def get_log_path(self) -> str:
        """Get path to text log file"""
        return str(self.text_log_file)

    def get_json_log_path(self) -> str:
        """Get path to JSON log file"""
        return str(self.json_log_file)
