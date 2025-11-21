"""
Memory Manager - Unified Interface for All Memory Operations.

This is the main entry point for the memory system. It provides:
- Single interface for all memory operations
- Automatic learning from interactions
- Memory context injection for prompts
- Memory commands (remember, forget, recall)
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .memory_types import (
    ActionOutcome,
    ActionRecord,
    FeedbackType,
    ImportanceLevel,
    Instruction,
    MemoryContext,
    MemorySource,
    UserPreference
)
from .semantic_memory import SemanticMemory
from .episodic_memory import EpisodicMemory
from .procedural_memory import ProceduralMemory
from .memory_retrieval import MemoryRetrieval
from .memory_consolidation import MemoryConsolidation

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Unified interface for all memory operations.

    This is the main class that orchestrators and agents should use.
    It coordinates all three memory types and provides high-level operations.
    """

    def __init__(self, storage_dir: str = "memory"):
        """
        Initialize the memory manager.

        Args:
            storage_dir: Directory for persistent storage
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Initialize memory subsystems
        self.semantic = SemanticMemory(storage_dir)
        self.episodic = EpisodicMemory(storage_dir)
        self.procedural = ProceduralMemory(storage_dir)

        # Initialize retrieval and consolidation
        self.retrieval = MemoryRetrieval(
            self.semantic, self.episodic, self.procedural
        )
        self.consolidation = MemoryConsolidation(
            self.semantic, self.episodic, self.procedural
        )

        # Current session tracking
        self.current_session_id: Optional[str] = None
        self.session_actions: List[str] = []

        logger.info(f"Memory manager initialized with storage: {storage_dir}")

    # =========================================================================
    # Session Management
    # =========================================================================

    def start_session(self, session_id: str) -> None:
        """
        Start a new session.

        Args:
            session_id: Unique session identifier
        """
        self.current_session_id = session_id
        self.session_actions = []
        self.episodic.start_conversation(session_id)
        logger.info(f"Started session: {session_id}")

    def end_session(self) -> None:
        """End the current session and consolidate memories."""
        if not self.current_session_id:
            return

        # End conversation
        self.episodic.end_conversation(
            self.current_session_id,
            summary=f"Session with {len(self.session_actions)} actions",
            outcome="completed"
        )

        # Learn from conversation
        self.consolidation.learn_from_conversation(
            self.current_session_id,
            [],  # Messages not tracked here
            self.session_actions
        )

        logger.info(f"Ended session: {self.current_session_id}")
        self.current_session_id = None
        self.session_actions = []

    # =========================================================================
    # Context Retrieval (Main API for Orchestrator)
    # =========================================================================

    def get_context(
        self,
        message: str,
        entities: Optional[List[str]] = None,
        intent: Optional[str] = None,
        agent: Optional[str] = None
    ) -> MemoryContext:
        """
        Get relevant memory context for a message.

        This is the main method for injecting memory into prompts.

        Args:
            message: User message or query
            entities: Extracted entities
            intent: Detected intent
            agent: Target agent

        Returns:
            MemoryContext with relevant memories
        """
        return self.retrieval.get_relevant_context(
            message, entities, intent, agent
        )

    def get_context_text(
        self,
        message: str,
        entities: Optional[List[str]] = None,
        intent: Optional[str] = None,
        agent: Optional[str] = None
    ) -> str:
        """
        Get memory context as formatted text for prompt injection.

        Args:
            message: User message
            entities: Extracted entities
            intent: Detected intent
            agent: Target agent

        Returns:
            Formatted text suitable for prompt injection
        """
        context = self.get_context(message, entities, intent, agent)
        return context.to_prompt_text()

    # =========================================================================
    # Preference Management
    # =========================================================================

    def remember_preference(
        self,
        key: str,
        value: Any,
        category: str = "general",
        context: str = "global"
    ) -> UserPreference:
        """
        Remember a user preference.

        Args:
            key: Preference key (e.g., "timezone")
            value: Preference value (e.g., "America/New_York")
            category: Category (e.g., "format", "style")
            context: When this applies

        Returns:
            The stored preference
        """
        pref = self.semantic.set_preference(
            category=category,
            key=key,
            value=value,
            context=context,
            source=MemorySource.EXPLICIT,
            importance=ImportanceLevel.HIGH.value
        )
        logger.info(f"Remembered preference: {key} = {value}")
        return pref

    def get_preference(
        self,
        key: str,
        context: str = "global",
        default: Any = None
    ) -> Any:
        """
        Get a preference value.

        Args:
            key: Preference key
            context: Context to check
            default: Default if not found

        Returns:
            Preference value or default
        """
        value = self.semantic.get_preference(key, context)
        return value if value is not None else default

    def list_preferences(self, context: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all preferences.

        Args:
            context: Optional context filter

        Returns:
            List of preference dictionaries
        """
        prefs = self.semantic.get_all_preferences(context)
        return [
            {
                "key": p.key,
                "value": p.value,
                "category": p.category,
                "context": p.context
            }
            for p in prefs
        ]

    # =========================================================================
    # Instruction Management
    # =========================================================================

    def remember_instruction(
        self,
        instruction: str,
        context: str = "global",
        priority: int = 5
    ) -> Instruction:
        """
        Remember an explicit instruction.

        Args:
            instruction: The instruction text
            context: When this applies
            priority: Importance (1-10)

        Returns:
            The stored instruction
        """
        instr = self.semantic.add_instruction(
            instruction=instruction,
            context=context,
            priority=priority,
            source=MemorySource.EXPLICIT
        )
        logger.info(f"Remembered instruction: {instruction[:50]}...")
        return instr

    def get_instructions(self, context: str = "global") -> List[str]:
        """
        Get all active instructions for a context.

        Args:
            context: Context to filter

        Returns:
            List of instruction texts
        """
        instructions = self.semantic.get_instructions(context)
        return [i.instruction for i in instructions]

    # =========================================================================
    # Entity/Fact Management
    # =========================================================================

    def remember_entity(
        self,
        entity_type: str,
        entity_id: str,
        entity_name: str,
        properties: Optional[Dict[str, Any]] = None,
        notes: str = ""
    ) -> None:
        """
        Remember information about an entity.

        Args:
            entity_type: Type (person, project, etc.)
            entity_id: Identifier
            entity_name: Display name
            properties: Entity properties
            notes: Free-form notes
        """
        self.semantic.add_entity(
            entity_type=entity_type,
            entity_id=entity_id,
            entity_name=entity_name,
            properties=properties,
            notes=notes,
            source=MemorySource.EXPLICIT
        )
        logger.info(f"Remembered entity: {entity_type}:{entity_name}")

    def remember_fact(
        self,
        subject: str,
        predicate: str,
        obj: str
    ) -> None:
        """
        Remember a fact (subject-predicate-object triple).

        Args:
            subject: What this is about
            predicate: The relationship
            obj: The value
        """
        self.semantic.add_fact(
            subject=subject,
            predicate=predicate,
            obj=obj,
            source=MemorySource.EXPLICIT
        )
        logger.info(f"Remembered fact: {subject} {predicate} {obj}")

    # =========================================================================
    # Action Recording
    # =========================================================================

    def record_action(
        self,
        action_type: str,
        agent: str,
        input_summary: str,
        output_summary: str,
        success: bool = True,
        duration_ms: float = 0.0,
        entities: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> ActionRecord:
        """
        Record an action that was taken.

        Args:
            action_type: Type of action
            agent: Which agent performed it
            input_summary: What was requested
            output_summary: What was the result
            success: Whether it succeeded
            duration_ms: How long it took
            entities: Entities involved
            parameters: Action parameters

        Returns:
            The action record
        """
        # Record in episodic memory
        action = self.episodic.record_action(
            action_type=action_type,
            agent=agent,
            input_summary=input_summary,
            output_summary=output_summary,
            outcome=ActionOutcome.SUCCESS if success else ActionOutcome.FAILURE,
            duration_ms=duration_ms,
            entities=entities
        )

        # Record tool usage pattern
        if parameters:
            self.procedural.record_tool_usage(
                agent=agent,
                action=action_type,
                parameters=parameters,
                success=success
            )

        # Track for session
        self.session_actions.append(action_type)

        # Update conversation
        if self.current_session_id:
            self.episodic.update_conversation(
                self.current_session_id,
                actions=[action_type],
                entities=entities
            )

        return action

    def record_feedback(
        self,
        action_id: str,
        positive: bool,
        details: str = ""
    ) -> None:
        """
        Record user feedback on an action.

        Args:
            action_id: ID of the action
            positive: Whether feedback was positive
            details: Additional details
        """
        feedback = FeedbackType.POSITIVE if positive else FeedbackType.NEGATIVE
        self.episodic.record_feedback(action_id, feedback, details)

        # Learn from feedback
        for action in self.episodic.actions:
            if action.id == action_id:
                self.consolidation.learn_from_feedback(action, feedback, details)
                break

    # =========================================================================
    # Memory Commands (Natural Language Interface)
    # =========================================================================

    def process_memory_command(self, message: str) -> Optional[str]:
        """
        Process a potential memory command from user message.

        Handles commands like:
        - "Remember that I prefer UTC timezone"
        - "Forget my timezone preference"
        - "What do you remember about timezone?"

        Args:
            message: User message

        Returns:
            Response if it was a memory command, None otherwise
        """
        message_lower = message.lower().strip()

        # Remember commands
        remember_patterns = [
            r"remember (?:that )?(.+)",
            r"save (?:that )?(.+)",
            r"note (?:that )?(.+)",
            r"keep in mind (?:that )?(.+)"
        ]

        for pattern in remember_patterns:
            match = re.match(pattern, message_lower)
            if match:
                content = match.group(1)
                return self._handle_remember(content)

        # Forget commands
        forget_patterns = [
            r"forget (?:about )?(.+)",
            r"delete (?:my )?(.+)",
            r"remove (?:my )?(.+)"
        ]

        for pattern in forget_patterns:
            match = re.match(pattern, message_lower)
            if match:
                content = match.group(1)
                return self._handle_forget(content)

        # Recall commands
        recall_patterns = [
            r"what do you (?:remember|know) (?:about )?(.+)",
            r"recall (.+)",
            r"show (?:me )?(?:my )?(.+) (?:preferences?|settings?)",
            r"list (?:my )?(.+)"
        ]

        for pattern in recall_patterns:
            match = re.match(pattern, message_lower)
            if match:
                content = match.group(1)
                return self._handle_recall(content)

        return None

    def _handle_remember(self, content: str) -> str:
        """Handle a remember command."""
        # Try to parse preference
        pref_match = re.match(
            r"(?:i |my )?(?:prefer|like|want|use) (.+?) (?:to be |as |is )?(.+)",
            content,
            re.IGNORECASE
        )

        if pref_match:
            key = pref_match.group(1).strip()
            value = pref_match.group(2).strip()
            self.remember_preference(key, value)
            return f"I'll remember that your {key} is {value}."

        # Try to parse instruction
        if any(word in content.lower() for word in ["always", "never", "don't", "do not"]):
            self.remember_instruction(content, priority=7)
            return f"I'll remember: {content}"

        # Store as general fact
        self.remember_instruction(content, priority=5)
        return f"I'll remember: {content}"

    def _handle_forget(self, content: str) -> str:
        """Handle a forget command."""
        content_lower = content.lower()

        # Try to find matching preference
        for pref_id, pref in list(self.semantic.preferences.items()):
            if (pref.key.lower() in content_lower or
                content_lower in pref.key.lower()):
                del self.semantic.preferences[pref_id]
                self.semantic._save()
                return f"I've forgotten your {pref.key} preference."

        # Try to find matching instruction
        for instr_id, instr in list(self.semantic.instructions.items()):
            if content_lower in instr.instruction.lower():
                self.semantic.deactivate_instruction(instr_id)
                return f"I've forgotten that instruction."

        return f"I couldn't find anything matching '{content}' to forget."

    def _handle_recall(self, content: str) -> str:
        """Handle a recall command."""
        content_lower = content.lower()

        # Check for specific preference
        value = self.semantic.get_preference(content_lower)
        if value:
            return f"Your {content} is set to: {value}"

        # Search all memories
        results = self.semantic.search(content, max_results=5)
        if results:
            lines = ["Here's what I remember:"]
            for mem in results:
                if isinstance(mem, UserPreference):
                    lines.append(f"- {mem.key}: {mem.value}")
                elif isinstance(mem, Instruction):
                    lines.append(f"- Instruction: {mem.instruction}")
            return "\n".join(lines)

        return f"I don't have any memories about '{content}'."

    # =========================================================================
    # Workflow Management
    # =========================================================================

    def add_workflow(
        self,
        name: str,
        description: str,
        steps: List[Dict[str, Any]],
        trigger: str = ""
    ) -> None:
        """
        Add a workflow.

        Args:
            name: Workflow name
            description: What it does
            steps: List of steps
            trigger: Trigger phrase
        """
        self.procedural.add_workflow(
            workflow_name=name,
            description=description,
            steps=steps,
            trigger_phrase=trigger,
            source=MemorySource.EXPLICIT
        )

    def find_workflow(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Find a workflow matching a query.

        Args:
            query: Search query

        Returns:
            Workflow dict or None
        """
        workflow = self.procedural.find_workflow(query)
        if workflow:
            return {
                "name": workflow.workflow_name,
                "description": workflow.description,
                "steps": workflow.steps
            }
        return None

    # =========================================================================
    # Statistics & Management
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return {
            "semantic": self.semantic.get_stats(),
            "episodic": self.episodic.get_stats(),
            "procedural": self.procedural.get_stats(),
            "learning": self.consolidation.get_learning_stats()
        }

    def consolidate(self) -> Dict[str, int]:
        """
        Run memory consolidation.

        Should be called periodically or at end of session.
        """
        return self.consolidation.consolidate()

    def clear_all(self) -> None:
        """Clear all memories. Use with caution!"""
        self.semantic.clear()
        self.episodic.clear()
        self.procedural.clear()
        logger.warning("Cleared all memories")

    def export_memories(self) -> Dict[str, Any]:
        """Export all memories for backup."""
        return {
            "semantic": {
                "preferences": [p.to_dict() for p in self.semantic.preferences.values()],
                "instructions": [i.to_dict() for i in self.semantic.instructions.values()],
                "facts": [f.to_dict() for f in self.semantic.facts.values()],
                "entities": [e.to_dict() for e in self.semantic.entities.values()]
            },
            "episodic": {
                "conversations": [c.to_dict() for c in self.episodic.conversations.values()],
                "actions": [a.to_dict() for a in self.episodic.actions],
                "corrections": [c.to_dict() for c in self.episodic.corrections]
            },
            "procedural": {
                "patterns": [p.to_dict() for p in self.procedural.patterns.values()],
                "workflows": [w.to_dict() for w in self.procedural.workflows.values()],
                "tool_patterns": [t.to_dict() for t in self.procedural.tool_patterns.values()]
            }
        }


# Convenience function for creating a memory manager
def create_memory_manager(storage_dir: str = "memory") -> MemoryManager:
    """
    Create and return a memory manager instance.

    Args:
        storage_dir: Directory for persistent storage

    Returns:
        Configured MemoryManager instance
    """
    return MemoryManager(storage_dir)
