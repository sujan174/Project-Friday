"""
Simple Session Logger - One file per session containing everything

This module provides a simple logging system where each chat session has ONE file
that contains all calls, responses, memory updates, and everything else.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class LogEntry:
    """A single log entry"""
    timestamp: float
    timestamp_iso: str
    type: str  # 'user_message', 'assistant_response', 'agent_call', 'function_call', 'memory_update', 'error', 'system'
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SessionLogger:
    """
    Simple session logger that writes everything to one JSON file per session.

    Each session file contains:
    - All user messages
    - All assistant responses
    - All agent calls and responses
    - All function calls
    - All memory updates
    - All errors
    - Session metadata
    """

    def __init__(self, session_id: str, log_dir: str = "logs"):
        """
        Initialize session logger

        Args:
            session_id: Unique session identifier
            log_dir: Directory to store log files
        """
        self.session_id = session_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)

        self.log_file = self.log_dir / f"session_{session_id}.json"
        self.entries: List[LogEntry] = []
        self.session_start = time.time()

        # Session metadata
        self.metadata = {
            'session_id': session_id,
            'started_at': self.session_start,
            'started_at_iso': datetime.fromtimestamp(self.session_start).isoformat(),
        }

        # Load existing log if it exists
        if self.log_file.exists():
            self._load_existing_log()
        else:
            self._write_initial_log()

    def _write_initial_log(self):
        """Write initial log file structure"""
        initial_data = {
            'metadata': self.metadata,
            'entries': []
        }
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(initial_data, f, indent=2)

    def _load_existing_log(self):
        """Load existing log file"""
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.metadata = data.get('metadata', self.metadata)
                entries_data = data.get('entries', [])
                self.entries = [
                    LogEntry(
                        timestamp=e['timestamp'],
                        timestamp_iso=e['timestamp_iso'],
                        type=e['type'],
                        data=e['data']
                    ) for e in entries_data
                ]
        except Exception:
            # If loading fails, start fresh
            self.entries = []

    def _add_entry(self, entry_type: str, data: Dict[str, Any]):
        """Add a log entry"""
        now = time.time()
        entry = LogEntry(
            timestamp=now,
            timestamp_iso=datetime.fromtimestamp(now).isoformat(),
            type=entry_type,
            data=data
        )
        self.entries.append(entry)
        self._save()

    def _save(self):
        """Save log file"""
        try:
            # Update metadata
            self.metadata['last_updated'] = time.time()
            self.metadata['last_updated_iso'] = datetime.now().isoformat()
            self.metadata['entry_count'] = len(self.entries)
            self.metadata['duration_seconds'] = time.time() - self.session_start

            # Write to file
            log_data = {
                'metadata': self.metadata,
                'entries': [e.to_dict() for e in self.entries]
            }

            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            # Fail silently to avoid breaking the application
            print(f"Warning: Failed to save session log: {e}")

    def log_user_message(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log a user message"""
        self._add_entry('user_message', {
            'message': message,
            'length': len(message),
            **(metadata or {})
        })

    def log_assistant_response(self, response: str, metadata: Optional[Dict[str, Any]] = None):
        """Log an assistant response"""
        self._add_entry('assistant_response', {
            'response': response,
            'length': len(response),
            **(metadata or {})
        })

    def log_agent_call(
        self,
        agent_name: str,
        instruction: str,
        response: Optional[str] = None,
        duration: Optional[float] = None,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log an agent call"""
        data = {
            'agent_name': agent_name,
            'instruction': instruction,
            'instruction_length': len(instruction),
            'success': success,
        }

        if response:
            data['response'] = response
            data['response_length'] = len(response)

        if duration is not None:
            data['duration_ms'] = duration * 1000 if duration < 1000 else duration

        if error:
            data['error'] = error

        if metadata:
            data.update(metadata)

        self._add_entry('agent_call', data)

    def log_function_call(
        self,
        function_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        result: Optional[Any] = None,
        success: bool = True,
        error: Optional[str] = None,
        duration: Optional[float] = None
    ):
        """Log a function call"""
        data = {
            'function_name': function_name,
            'success': success,
        }

        if arguments:
            data['arguments'] = arguments

        if result is not None:
            data['result'] = str(result) if not isinstance(result, (dict, list, str, int, float, bool)) else result

        if error:
            data['error'] = error

        if duration is not None:
            data['duration_ms'] = duration * 1000 if duration < 1000 else duration

        self._add_entry('function_call', data)

    def log_memory_update(self, key: str, value: Any, operation: str = 'set'):
        """Log a memory update"""
        self._add_entry('memory_update', {
            'key': key,
            'value': str(value) if not isinstance(value, (dict, list, str, int, float, bool)) else value,
            'operation': operation
        })

    def log_context_update(self, context_data: Dict[str, Any]):
        """Log a context update"""
        self._add_entry('context_update', context_data)

    def log_intelligence_processing(
        self,
        stage: str,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        duration: Optional[float] = None,
        success: bool = True
    ):
        """Log intelligence processing stage"""
        data = {
            'stage': stage,
            'success': success,
        }

        if input_data:
            data['input'] = input_data

        if output_data:
            data['output'] = output_data

        if duration is not None:
            data['duration_ms'] = duration * 1000 if duration < 1000 else duration

        self._add_entry('intelligence_processing', data)

    def log_error(
        self,
        error: str,
        error_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        traceback: Optional[str] = None
    ):
        """Log an error"""
        data = {
            'error': error,
        }

        if error_type:
            data['error_type'] = error_type

        if context:
            data['context'] = context

        if traceback:
            data['traceback'] = traceback

        self._add_entry('error', data)

    def log_system_event(self, event: str, data: Optional[Dict[str, Any]] = None):
        """Log a system event"""
        self._add_entry('system', {
            'event': event,
            **(data or {})
        })

    def get_log_path(self) -> str:
        """Get the path to the log file"""
        return str(self.log_file)

    def get_entries(self, entry_type: Optional[str] = None) -> List[LogEntry]:
        """Get all entries, optionally filtered by type"""
        if entry_type:
            return [e for e in self.entries if e.type == entry_type]
        return self.entries

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the session"""
        # Count entries by type
        type_counts = {}
        for entry in self.entries:
            type_counts[entry.type] = type_counts.get(entry.type, 0) + 1

        # Count agent calls
        agent_calls = {}
        for entry in self.entries:
            if entry.type == 'agent_call':
                agent_name = entry.data.get('agent_name', 'unknown')
                agent_calls[agent_name] = agent_calls.get(agent_name, 0) + 1

        # Count errors
        error_count = type_counts.get('error', 0)

        return {
            'session_id': self.session_id,
            'duration_seconds': time.time() - self.session_start,
            'total_entries': len(self.entries),
            'entry_types': type_counts,
            'agent_calls': agent_calls,
            'error_count': error_count,
            'log_file': str(self.log_file)
        }

    def close(self):
        """Close the logger and write final summary"""
        self.metadata['ended_at'] = time.time()
        self.metadata['ended_at_iso'] = datetime.now().isoformat()
        self.metadata['duration_seconds'] = time.time() - self.session_start
        self.metadata['summary'] = self.get_summary()
        self._save()


# Global session logger instance
_global_logger: Optional[SessionLogger] = None


def get_session_logger() -> Optional[SessionLogger]:
    """Get the global session logger instance"""
    return _global_logger


def set_session_logger(logger: SessionLogger):
    """Set the global session logger instance"""
    global _global_logger
    _global_logger = logger


def init_session_logger(session_id: str, log_dir: str = "logs") -> SessionLogger:
    """Initialize and set the global session logger"""
    logger = SessionLogger(session_id, log_dir)
    set_session_logger(logger)
    return logger
