"""
Episodic Memory - Interaction History and Experience Storage.

This module manages the "what happened" aspect of memory:
- Conversation summaries
- Actions taken and their outcomes
- User corrections and feedback
"""

import json
import logging
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from .memory_types import (
    ActionOutcome,
    ActionRecord,
    ConversationSummary,
    FeedbackType,
    ImportanceLevel,
    Memory,
    MemorySource,
    UserCorrection
)

logger = logging.getLogger(__name__)


class EpisodicMemory:
    """
    Manages episodic memory - what happened in past interactions.

    This is the "what happened" memory that stores:
    - Conversation summaries
    - Actions taken and their outcomes
    - User corrections and feedback
    - Error patterns
    """

    def __init__(
        self,
        storage_dir: str = "memory",
        max_recent_actions: int = 100,
        max_conversations: int = 50
    ):
        """
        Initialize episodic memory.

        Args:
            storage_dir: Directory for persistent storage
            max_recent_actions: Maximum recent actions to keep in memory
            max_conversations: Maximum conversation summaries to keep
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.max_recent_actions = max_recent_actions
        self.max_conversations = max_conversations

        # In-memory storage
        self.conversations: Dict[str, ConversationSummary] = {}
        self.actions: deque = deque(maxlen=max_recent_actions)
        self.corrections: List[UserCorrection] = []
        self.action_index: Dict[str, List[str]] = {}  # action_type -> [action_ids]

        # Load from disk
        self._load()

    # =========================================================================
    # Conversation Management
    # =========================================================================

    def start_conversation(self, session_id: str) -> ConversationSummary:
        """
        Start tracking a new conversation.

        Args:
            session_id: Unique session identifier

        Returns:
            The conversation summary object
        """
        summary = ConversationSummary(
            session_id=session_id,
            source=MemorySource.SYSTEM,
            importance=ImportanceLevel.MEDIUM.value
        )
        self.conversations[session_id] = summary
        logger.debug(f"Started conversation: {session_id}")
        return summary

    def update_conversation(
        self,
        session_id: str,
        topics: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
        actions: Optional[List[str]] = None
    ) -> None:
        """
        Update conversation with new information.

        Args:
            session_id: Session identifier
            topics: Topics discussed
            entities: Entities mentioned
            actions: Actions taken
        """
        if session_id not in self.conversations:
            self.start_conversation(session_id)

        conv = self.conversations[session_id]
        if topics:
            conv.topics.extend(t for t in topics if t not in conv.topics)
        if entities:
            conv.entities_mentioned.extend(e for e in entities if e not in conv.entities_mentioned)
        if actions:
            conv.actions_taken.extend(a for a in actions if a not in conv.actions_taken)
        conv.message_count += 1

    def end_conversation(
        self,
        session_id: str,
        summary: str = "",
        outcome: str = ""
    ) -> None:
        """
        End a conversation and compute final summary.

        Args:
            session_id: Session identifier
            summary: Summary of the conversation
            outcome: Overall outcome
        """
        if session_id not in self.conversations:
            return

        conv = self.conversations[session_id]
        conv.summary = summary
        conv.outcome = outcome
        conv.duration_seconds = (datetime.utcnow() - conv.created_at).total_seconds()

        # Compute importance based on content
        if len(conv.actions_taken) > 5:
            conv.importance = ImportanceLevel.HIGH.value
        elif len(conv.actions_taken) > 0:
            conv.importance = ImportanceLevel.MEDIUM.value

        self._save()
        logger.info(f"Ended conversation {session_id}: {len(conv.actions_taken)} actions")

    def get_conversation(self, session_id: str) -> Optional[ConversationSummary]:
        """Get a conversation by session ID."""
        return self.conversations.get(session_id)

    def get_recent_conversations(
        self,
        limit: int = 10,
        min_importance: float = 0.0
    ) -> List[ConversationSummary]:
        """
        Get recent conversations.

        Args:
            limit: Maximum number to return
            min_importance: Minimum importance threshold

        Returns:
            List of conversations sorted by recency
        """
        convs = [c for c in self.conversations.values()
                 if c.importance >= min_importance]
        convs.sort(key=lambda x: x.created_at, reverse=True)
        return convs[:limit]

    # =========================================================================
    # Action Recording
    # =========================================================================

    def record_action(
        self,
        action_type: str,
        agent: str,
        input_summary: str,
        output_summary: str,
        outcome: ActionOutcome = ActionOutcome.SUCCESS,
        duration_ms: float = 0.0,
        entities: Optional[List[str]] = None,
        error_message: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> ActionRecord:
        """
        Record an action that was taken.

        Args:
            action_type: Type of action (e.g., "create_issue")
            agent: Which agent performed it
            input_summary: What was requested
            output_summary: What was the result
            outcome: Success/failure status
            duration_ms: How long it took
            entities: Entities involved
            error_message: Error message if failed
            metadata: Additional metadata

        Returns:
            The action record
        """
        action = ActionRecord(
            action_type=action_type,
            agent=agent,
            input_summary=input_summary,
            output_summary=output_summary,
            outcome=outcome,
            duration_ms=duration_ms,
            entities_involved=entities or [],
            error_message=error_message,
            source=MemorySource.SYSTEM,
            metadata=metadata or {},
            importance=ImportanceLevel.MEDIUM.value if outcome == ActionOutcome.SUCCESS else ImportanceLevel.HIGH.value
        )

        self.actions.append(action)

        # Update index
        if action_type not in self.action_index:
            self.action_index[action_type] = []
        self.action_index[action_type].append(action.id)

        logger.debug(f"Recorded action: {action_type} [{outcome.value}]")
        self._save()
        return action

    def record_feedback(
        self,
        action_id: str,
        feedback: FeedbackType,
        details: str = ""
    ) -> bool:
        """
        Record user feedback on an action.

        Args:
            action_id: ID of the action
            feedback: Type of feedback
            details: Additional details

        Returns:
            True if feedback was recorded
        """
        for action in self.actions:
            if action.id == action_id:
                action.feedback = feedback
                action.feedback_details = details

                # Increase importance if correction
                if feedback == FeedbackType.CORRECTION:
                    action.importance = ImportanceLevel.HIGH.value

                self._save()
                logger.info(f"Recorded feedback for action {action_id}: {feedback.value}")
                return True

        return False

    def get_recent_actions(
        self,
        limit: int = 10,
        action_type: Optional[str] = None,
        agent: Optional[str] = None,
        outcome: Optional[ActionOutcome] = None
    ) -> List[ActionRecord]:
        """
        Get recent actions with optional filtering.

        Args:
            limit: Maximum number to return
            action_type: Filter by action type
            agent: Filter by agent
            outcome: Filter by outcome

        Returns:
            List of action records
        """
        results = []
        for action in reversed(self.actions):
            if action_type and action.action_type != action_type:
                continue
            if agent and action.agent != agent:
                continue
            if outcome and action.outcome != outcome:
                continue

            results.append(action)
            if len(results) >= limit:
                break

        return results

    def get_actions_by_type(
        self,
        action_type: str,
        limit: int = 10
    ) -> List[ActionRecord]:
        """Get actions of a specific type."""
        action_ids = self.action_index.get(action_type, [])
        results = []

        for action in reversed(self.actions):
            if action.id in action_ids:
                results.append(action)
                if len(results) >= limit:
                    break

        return results

    def get_actions_involving_entity(
        self,
        entity: str,
        limit: int = 10
    ) -> List[ActionRecord]:
        """Get actions involving a specific entity."""
        entity_lower = entity.lower()
        results = []

        for action in reversed(self.actions):
            if any(entity_lower in e.lower() for e in action.entities_involved):
                results.append(action)
                if len(results) >= limit:
                    break

        return results

    def get_failed_actions(
        self,
        limit: int = 10,
        since: Optional[datetime] = None
    ) -> List[ActionRecord]:
        """Get recent failed actions."""
        results = []

        for action in reversed(self.actions):
            if action.outcome == ActionOutcome.FAILURE:
                if since and action.created_at < since:
                    continue
                results.append(action)
                if len(results) >= limit:
                    break

        return results

    # =========================================================================
    # Correction Management
    # =========================================================================

    def record_correction(
        self,
        original_output: str,
        corrected_output: str,
        correction_type: str,
        context: str = "",
        lesson_learned: str = ""
    ) -> UserCorrection:
        """
        Record a user correction.

        Args:
            original_output: What we produced
            corrected_output: What user wanted
            correction_type: Type of correction (format, content, etc.)
            context: What was happening
            lesson_learned: What we should do differently

        Returns:
            The correction record
        """
        correction = UserCorrection(
            original_output=original_output,
            corrected_output=corrected_output,
            correction_type=correction_type,
            context=context,
            lesson_learned=lesson_learned,
            source=MemorySource.EXPLICIT,
            importance=ImportanceLevel.HIGH.value
        )

        self.corrections.append(correction)
        logger.info(f"Recorded correction: {correction_type}")

        self._save()
        return correction

    def get_corrections(
        self,
        correction_type: Optional[str] = None,
        limit: int = 10
    ) -> List[UserCorrection]:
        """
        Get user corrections.

        Args:
            correction_type: Filter by type
            limit: Maximum to return

        Returns:
            List of corrections
        """
        results = []
        for corr in reversed(self.corrections):
            if correction_type and corr.correction_type != correction_type:
                continue
            results.append(corr)
            if len(results) >= limit:
                break

        return results

    def get_lessons_learned(self) -> List[str]:
        """Get all lessons learned from corrections."""
        return [c.lesson_learned for c in self.corrections if c.lesson_learned]

    # =========================================================================
    # Analysis & Statistics
    # =========================================================================

    def get_action_success_rate(
        self,
        action_type: Optional[str] = None,
        agent: Optional[str] = None,
        days: int = 7
    ) -> float:
        """
        Calculate success rate for actions.

        Args:
            action_type: Filter by type
            agent: Filter by agent
            days: Look back period

        Returns:
            Success rate (0.0 to 1.0)
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        total = 0
        successful = 0

        for action in self.actions:
            if action.created_at < cutoff:
                continue
            if action_type and action.action_type != action_type:
                continue
            if agent and action.agent != agent:
                continue

            total += 1
            if action.outcome == ActionOutcome.SUCCESS:
                successful += 1

        return successful / total if total > 0 else 1.0

    def get_common_errors(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get most common error patterns."""
        error_counts: Dict[str, int] = {}

        for action in self.actions:
            if action.outcome == ActionOutcome.FAILURE and action.error_message:
                # Normalize error message
                error_key = action.error_message[:100]
                error_counts[error_key] = error_counts.get(error_key, 0) + 1

        # Sort by count
        sorted_errors = sorted(error_counts.items(), key=lambda x: -x[1])
        return [{"error": e, "count": c} for e, c in sorted_errors[:limit]]

    def get_stats(self) -> Dict[str, Any]:
        """Get episodic memory statistics."""
        total_actions = len(self.actions)
        successful = sum(1 for a in self.actions if a.outcome == ActionOutcome.SUCCESS)

        return {
            "conversations": len(self.conversations),
            "total_actions": total_actions,
            "successful_actions": successful,
            "success_rate": successful / total_actions if total_actions > 0 else 1.0,
            "corrections": len(self.corrections),
            "action_types": len(self.action_index)
        }

    # =========================================================================
    # Persistence
    # =========================================================================

    def _save(self) -> None:
        """Save episodic memory to disk."""
        try:
            data = {
                "conversations": {k: v.to_dict() for k, v in self.conversations.items()},
                "actions": [a.to_dict() for a in self.actions],
                "corrections": [c.to_dict() for c in self.corrections],
                "action_index": self.action_index
            }

            filepath = self.storage_dir / "episodic_memory.json"
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug(f"Saved episodic memory to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save episodic memory: {e}")

    def _load(self) -> None:
        """Load episodic memory from disk."""
        filepath = self.storage_dir / "episodic_memory.json"
        if not filepath.exists():
            logger.info("No existing episodic memory found, starting fresh")
            return

        try:
            with open(filepath) as f:
                data = json.load(f)

            # Load conversations
            for key, conv_data in data.get("conversations", {}).items():
                self.conversations[key] = ConversationSummary.from_dict(conv_data)

            # Load actions
            for action_data in data.get("actions", []):
                action = ActionRecord.from_dict(action_data)
                self.actions.append(action)

            # Load corrections
            for corr_data in data.get("corrections", []):
                self.corrections.append(UserCorrection.from_dict(corr_data))

            # Load action index
            self.action_index = data.get("action_index", {})

            logger.info(
                f"Loaded episodic memory: {len(self.conversations)} conversations, "
                f"{len(self.actions)} actions, {len(self.corrections)} corrections"
            )
        except Exception as e:
            logger.error(f"Failed to load episodic memory: {e}")

    def clear(self) -> None:
        """Clear all episodic memory."""
        self.conversations.clear()
        self.actions.clear()
        self.corrections.clear()
        self.action_index.clear()
        self._save()
        logger.info("Cleared all episodic memory")

    def prune_old_memories(self, days: int = 30) -> int:
        """
        Remove memories older than specified days.

        Args:
            days: Age threshold

        Returns:
            Number of memories removed
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        removed = 0

        # Prune conversations
        old_convs = [k for k, v in self.conversations.items()
                    if v.created_at < cutoff and v.importance < ImportanceLevel.HIGH.value]
        for k in old_convs:
            del self.conversations[k]
            removed += 1

        # Prune corrections (keep high importance)
        self.corrections = [c for c in self.corrections
                          if c.created_at >= cutoff or c.importance >= ImportanceLevel.HIGH.value]

        if removed > 0:
            self._save()
            logger.info(f"Pruned {removed} old memories")

        return removed
