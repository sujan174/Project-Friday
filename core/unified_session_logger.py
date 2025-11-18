"""
Unified Session Logger - Simplified 2-File Logging System

Creates only 2 files per chat session:
1. messages.jsonl - All message exchanges (user <-> orchestrator <-> subagents)
2. intelligence.jsonl - Intelligence system status after each conversation turn
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class MessageLogEntry:
    """Represents a single message in the conversation flow"""
    timestamp: str
    type: str  # 'user_message', 'orchestrator_to_agent', 'agent_to_orchestrator', 'assistant_response'
    from_entity: str  # 'user', 'orchestrator', or agent name
    to_entity: str  # 'orchestrator', agent name, or 'user'
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary"""
        return {
            'timestamp': self.timestamp,
            'type': self.type,
            'from': self.from_entity,
            'to': self.to_entity,
            'content': self.content,
            'metadata': self.metadata
        }


@dataclass
class IntelligenceLogEntry:
    """Represents intelligence system status for a conversation turn"""
    turn_number: int
    timestamp: str
    user_message: str
    intelligence: Dict[str, Any]
    execution_summary: Dict[str, Any] = field(default_factory=dict)
    turn_complete: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary"""
        return {
            'turn_number': self.turn_number,
            'timestamp': self.timestamp,
            'user_message': self.user_message,
            'intelligence': self.intelligence,
            'execution_summary': self.execution_summary,
            'turn_complete': self.turn_complete
        }


class UnifiedSessionLogger:
    """
    Unified session logger that creates exactly 2 files per chat session:
    - session_{session_id}_messages.jsonl: All message exchanges
    - session_{session_id}_intelligence.jsonl: Intelligence system status
    """

    def __init__(self, session_id: str, log_dir: str = "logs"):
        """
        Initialize the unified session logger

        Args:
            session_id: Unique session identifier
            log_dir: Directory to store log files
        """
        self.session_id = session_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Define the two log files
        self.messages_file = self.log_dir / f"session_{session_id}_messages.jsonl"
        self.intelligence_file = self.log_dir / f"session_{session_id}_intelligence.jsonl"

        # Track conversation turn number
        self.turn_number = 0
        self.current_turn_start = None
        self.current_turn_user_message = None

        # Initialize files with metadata
        self._initialize_files()

    def _initialize_files(self):
        """Initialize log files with session metadata"""
        session_metadata = {
            'session_id': self.session_id,
            'started_at': datetime.now().isoformat(),
            'log_version': '2.0',
            'description': 'Unified session logging - 2 files per session'
        }

        # Initialize messages file
        with open(self.messages_file, 'w') as f:
            f.write(json.dumps({
                'type': 'session_metadata',
                'data': session_metadata
            }) + '\n')

        # Initialize intelligence file
        with open(self.intelligence_file, 'w') as f:
            f.write(json.dumps({
                'type': 'session_metadata',
                'data': session_metadata
            }) + '\n')

    def _write_message(self, entry: MessageLogEntry):
        """Write a message entry to the messages log file"""
        with open(self.messages_file, 'a') as f:
            f.write(json.dumps(entry.to_dict()) + '\n')

    def _write_intelligence(self, entry: IntelligenceLogEntry):
        """Write an intelligence entry to the intelligence log file"""
        with open(self.intelligence_file, 'a') as f:
            f.write(json.dumps(entry.to_dict()) + '\n')

    # ==================== MESSAGE LOGGING ====================

    def log_user_message(self, message: str, metadata: Optional[Dict] = None):
        """
        Log a message from the user to the orchestrator

        Args:
            message: User's message content
            metadata: Optional additional metadata
        """
        # Start a new conversation turn
        self.turn_number += 1
        self.current_turn_start = datetime.now().isoformat()
        self.current_turn_user_message = message

        entry = MessageLogEntry(
            timestamp=self.current_turn_start,
            type='user_message',
            from_entity='user',
            to_entity='orchestrator',
            content=message,
            metadata=metadata or {}
        )
        self._write_message(entry)

    def log_orchestrator_to_agent(
        self,
        agent_name: str,
        instruction: str,
        context: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Log a message from orchestrator to a subagent

        Args:
            agent_name: Name of the target agent
            instruction: Instruction/task for the agent
            context: Optional context provided to agent
            metadata: Optional additional metadata
        """
        entry_metadata = metadata or {}
        entry_metadata['instruction_length'] = len(instruction)
        if context:
            entry_metadata['has_context'] = True
            entry_metadata['context_keys'] = list(context.keys()) if isinstance(context, dict) else None

        entry = MessageLogEntry(
            timestamp=datetime.now().isoformat(),
            type='orchestrator_to_agent',
            from_entity='orchestrator',
            to_entity=agent_name,
            content=instruction,
            metadata=entry_metadata
        )
        self._write_message(entry)

    def log_agent_to_orchestrator(
        self,
        agent_name: str,
        response: str,
        success: bool = True,
        duration_ms: Optional[float] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Log a response from a subagent back to orchestrator

        Args:
            agent_name: Name of the responding agent
            response: Agent's response content
            success: Whether the agent execution was successful
            duration_ms: Execution duration in milliseconds
            error: Error message if execution failed
            metadata: Optional additional metadata
        """
        entry_metadata = metadata or {}
        entry_metadata['success'] = success
        entry_metadata['response_length'] = len(response)
        if duration_ms is not None:
            entry_metadata['duration_ms'] = duration_ms
        if error:
            entry_metadata['error'] = error

        entry = MessageLogEntry(
            timestamp=datetime.now().isoformat(),
            type='agent_to_orchestrator',
            from_entity=agent_name,
            to_entity='orchestrator',
            content=response,
            metadata=entry_metadata
        )
        self._write_message(entry)

    def log_assistant_response(self, response: str, metadata: Optional[Dict] = None):
        """
        Log the final response from orchestrator back to user

        Args:
            response: Orchestrator's response to user
            metadata: Optional additional metadata
        """
        entry = MessageLogEntry(
            timestamp=datetime.now().isoformat(),
            type='assistant_response',
            from_entity='orchestrator',
            to_entity='user',
            content=response,
            metadata=metadata or {}
        )
        self._write_message(entry)

    # ==================== INTELLIGENCE LOGGING ====================

    def log_intelligence_status(
        self,
        intelligence_result: Dict[str, Any],
        execution_plan: Optional[Dict] = None,
        turn_complete: bool = False
    ):
        """
        Log the intelligence system status after processing a user message

        Args:
            intelligence_result: Result from hybrid intelligence system containing:
                - path_used: 'fast' or 'llm'
                - latency_ms: Processing time
                - intents: List of detected intents
                - entities: List of extracted entities
                - confidence: Overall confidence score
                - reasoning: Intelligence system reasoning
                - ambiguities: List of ambiguities detected
                - suggested_clarifications: Clarification suggestions
            execution_plan: Optional execution plan with tasks
            turn_complete: Whether this conversation turn is complete
        """
        entry = IntelligenceLogEntry(
            turn_number=self.turn_number,
            timestamp=datetime.now().isoformat(),
            user_message=self.current_turn_user_message or "",
            intelligence=intelligence_result,
            execution_summary=execution_plan or {},
            turn_complete=turn_complete
        )
        self._write_intelligence(entry)

    def log_turn_execution_summary(
        self,
        agents_called: List[str],
        total_duration_ms: float,
        success: bool,
        errors: Optional[List[str]] = None
    ):
        """
        Log execution summary for the current conversation turn

        Args:
            agents_called: List of agent names that were invoked
            total_duration_ms: Total execution time
            success: Whether the turn completed successfully
            errors: Any errors encountered
        """
        summary = {
            'turn_number': self.turn_number,
            'timestamp': datetime.now().isoformat(),
            'agents_called': agents_called,
            'agent_count': len(agents_called),
            'total_duration_ms': total_duration_ms,
            'success': success,
            'errors': errors or []
        }

        # Append to last intelligence entry if exists
        with open(self.intelligence_file, 'a') as f:
            f.write(json.dumps({
                'type': 'turn_execution_summary',
                'data': summary
            }) + '\n')

    def complete_turn(self):
        """Mark the current conversation turn as complete"""
        # This can be called to explicitly mark a turn as complete
        pass

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the session

        Returns:
            Dictionary containing session statistics
        """
        return {
            'session_id': self.session_id,
            'total_turns': self.turn_number,
            'messages_file': str(self.messages_file),
            'intelligence_file': str(self.intelligence_file),
            'files_exist': {
                'messages': self.messages_file.exists(),
                'intelligence': self.intelligence_file.exists()
            }
        }

    def read_messages(self) -> List[Dict[str, Any]]:
        """
        Read all messages from the messages log file

        Returns:
            List of message entries
        """
        messages = []
        if self.messages_file.exists():
            with open(self.messages_file, 'r') as f:
                for line in f:
                    messages.append(json.loads(line.strip()))
        return messages

    def read_intelligence_log(self) -> List[Dict[str, Any]]:
        """
        Read all intelligence entries from the intelligence log file

        Returns:
            List of intelligence entries
        """
        entries = []
        if self.intelligence_file.exists():
            with open(self.intelligence_file, 'r') as f:
                for line in f:
                    entries.append(json.loads(line.strip()))
        return entries


# Convenience functions for testing
def demo_logging():
    """Demonstrate the unified logging system"""
    logger = UnifiedSessionLogger(session_id="demo-session-123")

    # Simulate a conversation turn
    logger.log_user_message("Create a Jira ticket for bug fix")

    # Log intelligence processing
    logger.log_intelligence_status(
        intelligence_result={
            'path_used': 'fast',
            'latency_ms': 12.5,
            'intents': ['create'],
            'entities': [
                {'type': 'issue_type', 'value': 'ticket', 'confidence': 0.95},
                {'type': 'purpose', 'value': 'bug fix', 'confidence': 0.88}
            ],
            'confidence': 0.92,
            'reasoning': 'High-confidence keyword match on "create" and "Jira ticket"',
            'ambiguities': [],
            'suggested_clarifications': []
        },
        execution_plan={
            'tasks': [
                {
                    'id': 't1',
                    'agent': 'jira',
                    'action': 'create_ticket',
                    'estimated_duration_ms': 2500
                }
            ],
            'total_estimated_duration_ms': 2500,
            'estimated_cost_tokens': 1000
        }
    )

    # Log orchestrator -> agent communication
    logger.log_orchestrator_to_agent(
        agent_name='jira',
        instruction='Create a new ticket with title "Bug fix" and type "Bug"',
        metadata={'priority': 'high'}
    )

    # Log agent -> orchestrator communication
    logger.log_agent_to_orchestrator(
        agent_name='jira',
        response='Successfully created ticket PROJ-123: Bug fix',
        success=True,
        duration_ms=2341.5
    )

    # Log final response
    logger.log_assistant_response(
        "I've created Jira ticket PROJ-123 for the bug fix as requested."
    )

    # Log execution summary
    logger.log_turn_execution_summary(
        agents_called=['jira'],
        total_duration_ms=2354.0,
        success=True
    )

    print("Demo logging complete!")
    print(f"Messages log: {logger.messages_file}")
    print(f"Intelligence log: {logger.intelligence_file}")
    print(f"\nSession summary: {json.dumps(logger.get_session_summary(), indent=2)}")


if __name__ == "__main__":
    demo_logging()
