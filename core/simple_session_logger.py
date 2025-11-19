"""
Simple Session Logger - 2 Text Files Per Session

Creates exactly 2 human-readable text files per session:
1. session_{id}_conversations.txt - All messages between user, orchestrator, and agents
2. session_{id}_intelligence.txt - Intelligence system decisions, actions, and storage

Author: AI System
Version: 1.0
"""

import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class SimpleSessionLogger:
    """
    Simple session logger that creates exactly 2 text files:
    - Conversations: All message exchanges
    - Intelligence: All AI decisions and actions
    """

    def __init__(self, session_id: str, log_dir: str = "logs"):
        """
        Initialize the simple session logger.

        Args:
            session_id: Unique session identifier
            log_dir: Base directory to store log files
        """
        self.session_id = session_id
        self.base_log_dir = Path(log_dir)

        # Create session-specific folder
        self.session_dir = self.base_log_dir / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Two log files inside session folder
        self.conversations_file = self.session_dir / "conversations.txt"
        self.intelligence_file = self.session_dir / "intelligence.txt"

        # Thread safety
        self._lock = threading.Lock()

        # Session start time
        self.session_start = datetime.now()

        # Initialize files
        self._initialize_files()

    def _initialize_files(self):
        """Initialize both log files with headers"""
        # Conversations file header
        with open(self.conversations_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("SESSION CONVERSATIONS LOG\n")
            f.write("=" * 80 + "\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Started: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

        # Intelligence file header
        with open(self.intelligence_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("SESSION INTELLIGENCE LOG\n")
            f.write("=" * 80 + "\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Started: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

    def _get_timestamp(self) -> str:
        """Get formatted timestamp"""
        return datetime.now().strftime('%H:%M:%S')

    # =========================================================================
    # CONVERSATION LOGGING
    # =========================================================================

    def log_user_message(self, message: str):
        """Log a message from user to orchestrator"""
        with self._lock:
            with open(self.conversations_file, 'a', encoding='utf-8') as f:
                f.write(f"[{self._get_timestamp()}] USER -> ORCHESTRATOR\n")
                f.write("-" * 80 + "\n")
                f.write(f"{message}\n")
                f.write("\n")

    def log_orchestrator_response(self, response: str):
        """Log orchestrator's response to user"""
        with self._lock:
            with open(self.conversations_file, 'a', encoding='utf-8') as f:
                f.write(f"[{self._get_timestamp()}] ORCHESTRATOR -> USER\n")
                f.write("-" * 80 + "\n")
                f.write(f"{response}\n")
                f.write("\n")

    def log_orchestrator_to_agent(self, agent_name: str, instruction: str):
        """Log orchestrator sending instruction to agent"""
        with self._lock:
            with open(self.conversations_file, 'a', encoding='utf-8') as f:
                f.write(f"[{self._get_timestamp()}] ORCHESTRATOR -> {agent_name.upper()}\n")
                f.write("-" * 80 + "\n")
                f.write(f"{instruction}\n")
                f.write("\n")

    def log_agent_to_orchestrator(
        self,
        agent_name: str,
        response: str,
        success: bool = True,
        error: Optional[str] = None,
        duration_ms: Optional[float] = None
    ):
        """Log agent's response to orchestrator"""
        with self._lock:
            with open(self.conversations_file, 'a', encoding='utf-8') as f:
                status = "SUCCESS" if success else "FAILED"
                duration_str = f" ({duration_ms:.0f}ms)" if duration_ms else ""

                f.write(f"[{self._get_timestamp()}] {agent_name.upper()} -> ORCHESTRATOR [{status}]{duration_str}\n")
                f.write("-" * 80 + "\n")
                f.write(f"{response}\n")
                if error:
                    f.write(f"\nError: {error}\n")
                f.write("\n")

    def log_function_call(
        self,
        agent_name: str,
        function_name: str,
        arguments: Dict[str, Any],
        result: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None
    ):
        """Log a function/tool call by an agent"""
        with self._lock:
            with open(self.conversations_file, 'a', encoding='utf-8') as f:
                status = "SUCCESS" if success else "FAILED"
                f.write(f"[{self._get_timestamp()}] {agent_name.upper()} TOOL CALL [{status}]\n")
                f.write("-" * 80 + "\n")
                f.write(f"Function: {function_name}\n")
                f.write(f"Arguments: {self._format_dict(arguments)}\n")
                if result:
                    f.write(f"Result: {result[:500]}{'...' if len(result) > 500 else ''}\n")
                if error:
                    f.write(f"Error: {error}\n")
                f.write("\n")

    # =========================================================================
    # INTELLIGENCE LOGGING
    # =========================================================================

    def log_intelligence_start(self, user_message: str):
        """Log start of intelligence processing for a user message"""
        with self._lock:
            with open(self.intelligence_file, 'a', encoding='utf-8') as f:
                f.write(f"[{self._get_timestamp()}] PROCESSING USER MESSAGE\n")
                f.write("=" * 80 + "\n")
                f.write(f"Input: {user_message[:200]}{'...' if len(user_message) > 200 else ''}\n")
                f.write("\n")

    def log_intent_classification(
        self,
        intents: List[str],
        confidence: float,
        method: str,
        reasoning: str
    ):
        """Log intent classification results"""
        with self._lock:
            with open(self.intelligence_file, 'a', encoding='utf-8') as f:
                f.write(f"[{self._get_timestamp()}] INTENT CLASSIFICATION\n")
                f.write("-" * 80 + "\n")
                f.write(f"Method: {method}\n")
                f.write(f"Detected Intents: {', '.join(intents)}\n")
                f.write(f"Confidence: {confidence:.2f}\n")
                f.write(f"Reasoning: {reasoning}\n")
                f.write("\n")

    def log_entity_extraction(
        self,
        entities: List[Dict[str, Any]],
        confidence: float
    ):
        """Log entity extraction results"""
        with self._lock:
            with open(self.intelligence_file, 'a', encoding='utf-8') as f:
                f.write(f"[{self._get_timestamp()}] ENTITY EXTRACTION\n")
                f.write("-" * 80 + "\n")
                f.write(f"Confidence: {confidence:.2f}\n")
                f.write(f"Entities Found: {len(entities)}\n")
                for entity in entities[:10]:  # Show first 10
                    entity_type = entity.get('type', 'unknown')
                    entity_value = entity.get('value', 'unknown')
                    entity_conf = entity.get('confidence', 0)
                    f.write(f"  - {entity_type}: {entity_value} (conf: {entity_conf:.2f})\n")
                if len(entities) > 10:
                    f.write(f"  ... and {len(entities) - 10} more\n")
                f.write("\n")

    def log_context_resolution(self, resolutions: List[Dict[str, Any]]):
        """Log context/reference resolution"""
        with self._lock:
            with open(self.intelligence_file, 'a', encoding='utf-8') as f:
                f.write(f"[{self._get_timestamp()}] CONTEXT RESOLUTION\n")
                f.write("-" * 80 + "\n")
                f.write(f"References Resolved: {len(resolutions)}\n")
                for res in resolutions:
                    original = res.get('original', 'unknown')
                    resolved = res.get('resolved', 'unknown')
                    f.write(f"  - '{original}' -> '{resolved}'\n")
                f.write("\n")

    def log_task_decomposition(
        self,
        tasks: List[Dict[str, Any]],
        execution_plan: str
    ):
        """Log task decomposition and planning"""
        with self._lock:
            with open(self.intelligence_file, 'a', encoding='utf-8') as f:
                f.write(f"[{self._get_timestamp()}] TASK DECOMPOSITION\n")
                f.write("-" * 80 + "\n")
                f.write(f"Execution Plan: {execution_plan}\n")
                f.write(f"Tasks: {len(tasks)}\n")
                for i, task in enumerate(tasks, 1):
                    task_name = task.get('name', task.get('action', 'unknown'))
                    agent = task.get('agent', 'unknown')
                    f.write(f"  {i}. {task_name} -> {agent}\n")
                f.write("\n")

    def log_agent_selection(
        self,
        selected_agent: str,
        reason: str,
        considered_agents: Optional[List[str]] = None
    ):
        """Log agent selection decision"""
        with self._lock:
            with open(self.intelligence_file, 'a', encoding='utf-8') as f:
                f.write(f"[{self._get_timestamp()}] AGENT SELECTION\n")
                f.write("-" * 80 + "\n")
                f.write(f"Selected: {selected_agent}\n")
                f.write(f"Reason: {reason}\n")
                if considered_agents:
                    f.write(f"Considered: {', '.join(considered_agents)}\n")
                f.write("\n")

    def log_confidence_score(
        self,
        overall_score: float,
        components: Dict[str, float],
        recommendation: str
    ):
        """Log confidence scoring"""
        with self._lock:
            with open(self.intelligence_file, 'a', encoding='utf-8') as f:
                f.write(f"[{self._get_timestamp()}] CONFIDENCE SCORING\n")
                f.write("-" * 80 + "\n")
                f.write(f"Overall Score: {overall_score:.2f}\n")
                f.write("Components:\n")
                for component, score in components.items():
                    f.write(f"  - {component}: {score:.2f}\n")
                f.write(f"Recommendation: {recommendation}\n")
                f.write("\n")

    def log_decision(
        self,
        decision_type: str,
        action: str,
        reasoning: str
    ):
        """Log a decision made by the intelligence system"""
        with self._lock:
            with open(self.intelligence_file, 'a', encoding='utf-8') as f:
                f.write(f"[{self._get_timestamp()}] DECISION: {decision_type.upper()}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Action: {action}\n")
                f.write(f"Reasoning: {reasoning}\n")
                f.write("\n")

    def log_memory_storage(
        self,
        storage_type: str,
        key: str,
        value: Any,
        operation: str = "stored"
    ):
        """Log when something is stored in memory/context"""
        with self._lock:
            with open(self.intelligence_file, 'a', encoding='utf-8') as f:
                f.write(f"[{self._get_timestamp()}] MEMORY {operation.upper()}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Type: {storage_type}\n")
                f.write(f"Key: {key}\n")
                value_str = str(value)
                f.write(f"Value: {value_str[:200]}{'...' if len(value_str) > 200 else ''}\n")
                f.write("\n")

    def log_context_update(
        self,
        context_type: str,
        updates: Dict[str, Any]
    ):
        """Log context updates"""
        with self._lock:
            with open(self.intelligence_file, 'a', encoding='utf-8') as f:
                f.write(f"[{self._get_timestamp()}] CONTEXT UPDATE: {context_type.upper()}\n")
                f.write("-" * 80 + "\n")
                for key, value in updates.items():
                    value_str = str(value)
                    f.write(f"  {key}: {value_str[:100]}{'...' if len(value_str) > 100 else ''}\n")
                f.write("\n")

    def log_hybrid_intelligence_result(
        self,
        path_used: str,
        latency_ms: float,
        intents: List[str],
        entities: List[Any],
        confidence: float,
        reasoning: str
    ):
        """Log complete hybrid intelligence processing result"""
        with self._lock:
            with open(self.intelligence_file, 'a', encoding='utf-8') as f:
                f.write(f"[{self._get_timestamp()}] HYBRID INTELLIGENCE RESULT\n")
                f.write("-" * 80 + "\n")
                f.write(f"Path Used: {path_used} ({latency_ms:.1f}ms)\n")
                f.write(f"Intents: {', '.join(str(i) for i in intents)}\n")
                f.write(f"Entities: {len(entities)} found\n")
                f.write(f"Confidence: {confidence:.2f}\n")
                f.write(f"Reasoning: {reasoning}\n")
                f.write("\n")

    def log_risk_assessment(
        self,
        risk_level: str,
        reason: str,
        requires_confirmation: bool = False
    ):
        """Log risk assessment"""
        with self._lock:
            with open(self.intelligence_file, 'a', encoding='utf-8') as f:
                f.write(f"[{self._get_timestamp()}] RISK ASSESSMENT\n")
                f.write("-" * 80 + "\n")
                f.write(f"Risk Level: {risk_level}\n")
                f.write(f"Reason: {reason}\n")
                if requires_confirmation:
                    f.write("Requires User Confirmation: Yes\n")
                f.write("\n")

    def log_error(self, error_type: str, error_message: str, context: Optional[str] = None):
        """Log an error in the intelligence system"""
        with self._lock:
            with open(self.intelligence_file, 'a', encoding='utf-8') as f:
                f.write(f"[{self._get_timestamp()}] ERROR: {error_type.upper()}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Message: {error_message}\n")
                if context:
                    f.write(f"Context: {context}\n")
                f.write("\n")

    def log_intelligence_complete(self, total_duration_ms: float, success: bool):
        """Log completion of intelligence processing"""
        with self._lock:
            with open(self.intelligence_file, 'a', encoding='utf-8') as f:
                status = "SUCCESS" if success else "FAILED"
                f.write(f"[{self._get_timestamp()}] PROCESSING COMPLETE [{status}]\n")
                f.write(f"Total Duration: {total_duration_ms:.0f}ms\n")
                f.write("=" * 80 + "\n\n")

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _format_dict(self, d: Dict[str, Any], max_length: int = 200) -> str:
        """Format dictionary for logging"""
        result = []
        for key, value in d.items():
            value_str = str(value)
            if len(value_str) > 50:
                value_str = value_str[:50] + "..."
            result.append(f"{key}={value_str}")
        output = ", ".join(result)
        if len(output) > max_length:
            output = output[:max_length] + "..."
        return output

    def get_session_dir(self) -> str:
        """Get path to session directory"""
        return str(self.session_dir)

    def get_conversation_log_path(self) -> str:
        """Get path to conversations log file"""
        return str(self.conversations_file)

    def get_intelligence_log_path(self) -> str:
        """Get path to intelligence log file"""
        return str(self.intelligence_file)

    def close(self):
        """Close the logger and write session summary"""
        duration = datetime.now() - self.session_start

        # Write summary to both files
        summary = f"\n{'=' * 80}\n"
        summary += f"SESSION ENDED\n"
        summary += f"Duration: {duration}\n"
        summary += f"{'=' * 80}\n"

        with self._lock:
            with open(self.conversations_file, 'a', encoding='utf-8') as f:
                f.write(summary)
            with open(self.intelligence_file, 'a', encoding='utf-8') as f:
                f.write(summary)


# Convenience function
def create_session_logger(session_id: str, log_dir: str = "logs") -> SimpleSessionLogger:
    """Create a new simple session logger"""
    return SimpleSessionLogger(session_id, log_dir)


# Demo/test
if __name__ == "__main__":
    # Create logger
    logger = SimpleSessionLogger("demo-123")

    # Log a conversation
    logger.log_user_message("Create a Jira ticket for the login bug")

    # Log intelligence processing
    logger.log_intelligence_start("Create a Jira ticket for the login bug")

    logger.log_intent_classification(
        intents=["create"],
        confidence=0.95,
        method="fast_filter",
        reasoning="Keyword 'create' detected with high confidence"
    )

    logger.log_entity_extraction(
        entities=[
            {"type": "platform", "value": "jira", "confidence": 0.98},
            {"type": "issue_type", "value": "ticket", "confidence": 0.95},
            {"type": "description", "value": "login bug", "confidence": 0.90}
        ],
        confidence=0.94
    )

    logger.log_task_decomposition(
        tasks=[
            {"name": "create_issue", "agent": "jira", "priority": 1}
        ],
        execution_plan="sequential"
    )

    logger.log_agent_selection(
        selected_agent="jira",
        reason="Task requires Jira issue creation",
        considered_agents=["jira", "github"]
    )

    logger.log_decision(
        decision_type="proceed",
        action="Execute Jira create_issue",
        reasoning="High confidence (0.95), clear intent, no ambiguity"
    )

    logger.log_intelligence_complete(total_duration_ms=45.2, success=True)

    # Log orchestrator to agent
    logger.log_orchestrator_to_agent(
        agent_name="jira",
        instruction="Create a new ticket with title 'Login Bug' and type 'Bug'"
    )

    # Log agent response
    logger.log_agent_to_orchestrator(
        agent_name="jira",
        response="Successfully created ticket PROJ-456: Login Bug",
        success=True,
        duration_ms=1234.5
    )

    # Log memory storage
    logger.log_memory_storage(
        storage_type="workspace_knowledge",
        key="last_created_ticket",
        value="PROJ-456"
    )

    # Log orchestrator response
    logger.log_orchestrator_response(
        "I've created Jira ticket PROJ-456 for the login bug."
    )

    # Close logger
    logger.close()

    print(f"Demo complete!")
    print(f"Conversations: {logger.get_conversation_log_path()}")
    print(f"Intelligence: {logger.get_intelligence_log_path()}")
