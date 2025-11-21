"""
Memory Consolidation - Learning and Importance Decay.

This module handles:
- Pattern detection from episodic memories
- Short-term to long-term memory transfer
- Importance decay over time
- Memory pruning
"""

import logging
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .memory_types import (
    ActionOutcome,
    ActionRecord,
    BehavioralPattern,
    FeedbackType,
    ImportanceLevel,
    MemorySource,
    UserCorrection
)
from .semantic_memory import SemanticMemory
from .episodic_memory import EpisodicMemory
from .procedural_memory import ProceduralMemory

logger = logging.getLogger(__name__)


class MemoryConsolidation:
    """
    Handles memory consolidation - learning from experience.

    Features:
    - Pattern detection from repeated actions
    - Preference inference from behavior
    - Memory importance decay
    - Old memory pruning
    """

    def __init__(
        self,
        semantic_memory: SemanticMemory,
        episodic_memory: EpisodicMemory,
        procedural_memory: ProceduralMemory
    ):
        """
        Initialize memory consolidation.

        Args:
            semantic_memory: Semantic memory instance
            episodic_memory: Episodic memory instance
            procedural_memory: Procedural memory instance
        """
        self.semantic = semantic_memory
        self.episodic = episodic_memory
        self.procedural = procedural_memory

        # Configuration
        self.min_pattern_occurrences = 3
        self.pattern_confidence_threshold = 0.6
        self.decay_rate = 0.05
        self.max_memory_age_days = 90

    def consolidate(self) -> Dict[str, int]:
        """
        Run full memory consolidation.

        This should be called periodically (e.g., at end of session).

        Returns:
            Statistics about what was consolidated
        """
        stats = {
            "patterns_detected": 0,
            "preferences_inferred": 0,
            "memories_decayed": 0,
            "memories_pruned": 0
        }

        # Detect patterns from actions
        patterns = self._detect_action_patterns()
        stats["patterns_detected"] = len(patterns)

        # Infer preferences from behavior
        inferred = self._infer_preferences()
        stats["preferences_inferred"] = inferred

        # Decay old memories
        decayed = self._apply_decay()
        stats["memories_decayed"] = decayed

        # Prune very old or low-importance memories
        pruned = self._prune_memories()
        stats["memories_pruned"] = pruned

        logger.info(f"Consolidation complete: {stats}")
        return stats

    # =========================================================================
    # Pattern Detection
    # =========================================================================

    def _detect_action_patterns(self) -> List[BehavioralPattern]:
        """
        Detect behavioral patterns from action history.

        Looks for:
        - Repeated sequences of actions
        - Time-based patterns
        - Context-based patterns
        """
        patterns = []

        # Get all actions
        actions = list(self.episodic.actions)
        if len(actions) < self.min_pattern_occurrences:
            return patterns

        # Detect sequence patterns
        sequence_patterns = self._find_sequence_patterns(actions)
        patterns.extend(sequence_patterns)

        # Detect time-based patterns
        time_patterns = self._find_time_patterns(actions)
        patterns.extend(time_patterns)

        return patterns

    def _find_sequence_patterns(
        self,
        actions: List[ActionRecord]
    ) -> List[BehavioralPattern]:
        """Find repeated action sequences."""
        patterns = []

        # Extract action sequences
        action_types = [a.action_type for a in actions]

        # Look for patterns of length 2-4
        for length in range(2, 5):
            if len(action_types) < length:
                continue

            # Count sequences
            sequences: Counter = Counter()
            for i in range(len(action_types) - length + 1):
                seq = tuple(action_types[i:i + length])
                sequences[seq] += 1

            # Create patterns from frequent sequences
            for seq, count in sequences.items():
                if count >= self.min_pattern_occurrences:
                    pattern_name = f"seq_{'_'.join(seq)}"

                    # Check if pattern already exists
                    existing = self.procedural.get_pattern(pattern_name)
                    if existing:
                        # Reinforce existing pattern
                        for _ in range(count):
                            self.procedural.observe_pattern(
                                pattern_name,
                                list(seq)
                            )
                    else:
                        # Create new pattern
                        pattern = self.procedural.observe_pattern(
                            pattern_name,
                            list(seq),
                            frequency="on_demand"
                        )
                        # Boost confidence based on occurrence count
                        pattern.confidence = min(0.9, 0.3 + (count * 0.1))
                        pattern.observation_count = count
                        patterns.append(pattern)

                        logger.info(
                            f"Detected sequence pattern: {seq} "
                            f"(count: {count}, confidence: {pattern.confidence:.2f})"
                        )

        return patterns

    def _find_time_patterns(
        self,
        actions: List[ActionRecord]
    ) -> List[BehavioralPattern]:
        """Find time-based action patterns."""
        patterns = []

        # Group actions by hour
        hourly_actions: Dict[int, Counter] = {}
        for action in actions:
            hour = action.created_at.hour
            if hour not in hourly_actions:
                hourly_actions[hour] = Counter()
            hourly_actions[hour][action.action_type] += 1

        # Find actions that happen at specific times
        for hour, action_counts in hourly_actions.items():
            for action_type, count in action_counts.items():
                if count >= self.min_pattern_occurrences:
                    pattern_name = f"time_{hour:02d}_{action_type}"

                    existing = self.procedural.get_pattern(pattern_name)
                    if not existing:
                        pattern = self.procedural.observe_pattern(
                            pattern_name,
                            [action_type],
                            typical_time=f"{hour:02d}:00",
                            frequency="daily"
                        )
                        pattern.confidence = min(0.8, 0.3 + (count * 0.05))
                        pattern.observation_count = count
                        patterns.append(pattern)

                        logger.info(
                            f"Detected time pattern: {action_type} at {hour}:00 "
                            f"(count: {count})"
                        )

        return patterns

    # =========================================================================
    # Preference Inference
    # =========================================================================

    def _infer_preferences(self) -> int:
        """
        Infer preferences from behavior.

        Looks at:
        - Common parameter values
        - Format preferences from corrections
        - Success patterns
        """
        inferred = 0

        # Infer from tool usage patterns
        for pattern in self.procedural.tool_patterns.values():
            if pattern.usage_count < self.min_pattern_occurrences:
                continue

            # Get common parameters
            common_params = self.procedural.get_common_parameters(
                pattern.agent, pattern.action
            )

            for param, value in common_params.items():
                # Check if this is a meaningful preference
                if self._is_meaningful_preference(param, value):
                    existing = self.semantic.get_preference(
                        param,
                        context=pattern.agent
                    )

                    if not existing:
                        self.semantic.set_preference(
                            category="inferred",
                            key=f"{pattern.agent}_{param}",
                            value=value,
                            context=pattern.agent,
                            source=MemorySource.OBSERVED,
                            importance=ImportanceLevel.MEDIUM.value
                        )
                        inferred += 1
                        logger.debug(
                            f"Inferred preference: {pattern.agent}_{param}={value}"
                        )

        # Infer from corrections
        corrections = self.episodic.get_corrections(limit=50)
        correction_lessons = self._analyze_corrections(corrections)
        inferred += len(correction_lessons)

        return inferred

    def _is_meaningful_preference(self, param: str, value: str) -> bool:
        """Check if a parameter value represents a meaningful preference."""
        # Skip empty or generic values
        if not value or value.lower() in ["none", "null", "default", ""]:
            return False

        # Skip very long values (likely content, not preferences)
        if len(value) > 100:
            return False

        # Meaningful parameter names
        meaningful_params = [
            "format", "style", "priority", "project", "channel",
            "assignee", "type", "status", "label", "timezone"
        ]

        return any(mp in param.lower() for mp in meaningful_params)

    def _analyze_corrections(
        self,
        corrections: List[UserCorrection]
    ) -> List[Dict[str, Any]]:
        """Analyze corrections to learn preferences."""
        lessons = []

        # Group corrections by type
        by_type: Dict[str, List[UserCorrection]] = {}
        for corr in corrections:
            if corr.correction_type not in by_type:
                by_type[corr.correction_type] = []
            by_type[corr.correction_type].append(corr)

        # Analyze each type
        for corr_type, corrs in by_type.items():
            if len(corrs) >= 2:  # Multiple corrections of same type
                # Store as a preference/instruction
                lesson = corrs[-1].lesson_learned  # Most recent
                if lesson:
                    self.semantic.add_instruction(
                        instruction=lesson,
                        context="global",
                        priority=7,
                        conditions=[corr_type],
                        source=MemorySource.DERIVED
                    )
                    lessons.append({
                        "type": corr_type,
                        "lesson": lesson,
                        "count": len(corrs)
                    })
                    logger.info(f"Learned from corrections: {lesson}")

        return lessons

    # =========================================================================
    # Memory Decay
    # =========================================================================

    def _apply_decay(self) -> int:
        """
        Apply importance decay to old memories.

        Returns:
            Number of memories affected
        """
        decayed = 0
        now = datetime.utcnow()

        # Decay preferences
        for pref in self.semantic.preferences.values():
            days_old = (now - pref.last_accessed).days
            if days_old > 7 and pref.source != MemorySource.EXPLICIT:
                decay = self.decay_rate * (days_old / 7)
                pref.importance = max(0.1, pref.importance - decay)
                decayed += 1

        # Decay patterns
        for pattern in self.procedural.patterns.values():
            days_old = (now - pattern.last_accessed).days
            if days_old > 14:
                decay = self.decay_rate * (days_old / 14)
                pattern.confidence = max(0.1, pattern.confidence - decay)
                decayed += 1

        # Decay facts
        for fact in self.semantic.facts.values():
            days_old = (now - fact.last_accessed).days
            if days_old > 30:
                decay = self.decay_rate * (days_old / 30)
                fact.importance = max(0.1, fact.importance - decay)
                decayed += 1

        if decayed > 0:
            self.semantic._save()
            self.procedural._save()
            logger.debug(f"Applied decay to {decayed} memories")

        return decayed

    # =========================================================================
    # Memory Pruning
    # =========================================================================

    def _prune_memories(self) -> int:
        """
        Remove old or low-importance memories.

        Returns:
            Number of memories removed
        """
        pruned = 0

        # Prune old episodic memories
        pruned += self.episodic.prune_old_memories(
            days=self.max_memory_age_days
        )

        # Prune low-confidence patterns
        pruned += self.procedural.decay_unused_patterns(
            days_threshold=30,
            min_confidence=0.2
        )

        # Prune low-importance facts
        pruned += self._prune_low_importance_facts()

        return pruned

    def _prune_low_importance_facts(self) -> int:
        """Remove very low importance facts."""
        to_remove = []
        now = datetime.utcnow()

        for fact_id, fact in self.semantic.facts.items():
            # Keep if recently accessed
            if (now - fact.last_accessed).days < 7:
                continue
            # Keep if high importance
            if fact.importance >= ImportanceLevel.MEDIUM.value:
                continue
            # Keep if explicitly stored
            if fact.source == MemorySource.EXPLICIT:
                continue

            # Remove low-value facts
            if fact.importance < 0.2:
                to_remove.append(fact_id)

        for fact_id in to_remove:
            del self.semantic.facts[fact_id]

        if to_remove:
            self.semantic._save()
            logger.debug(f"Pruned {len(to_remove)} low-importance facts")

        return len(to_remove)

    # =========================================================================
    # Learning from Feedback
    # =========================================================================

    def learn_from_feedback(
        self,
        action: ActionRecord,
        feedback: FeedbackType,
        details: str = ""
    ) -> None:
        """
        Learn from user feedback on an action.

        Args:
            action: The action that was performed
            feedback: User's feedback
            details: Additional feedback details
        """
        if feedback == FeedbackType.POSITIVE:
            # Reinforce the pattern
            self.procedural.update_pattern_success(
                action.action_type, success=True
            )

            # Boost tool usage pattern
            pattern = self.procedural.get_tool_pattern(
                action.agent, action.action_type
            )
            if pattern:
                pattern.success_rate = min(1.0, pattern.success_rate + 0.05)

        elif feedback == FeedbackType.NEGATIVE:
            # Reduce confidence in pattern
            self.procedural.update_pattern_success(
                action.action_type, success=False
            )

        elif feedback == FeedbackType.CORRECTION:
            # Record the correction for learning
            self.episodic.record_correction(
                original_output=action.output_summary,
                corrected_output=details,
                correction_type=action.action_type,
                context=action.agent,
                lesson_learned=""  # Will be filled by analysis
            )

    def learn_from_conversation(
        self,
        session_id: str,
        messages: List[Dict[str, str]],
        actions_taken: List[str]
    ) -> None:
        """
        Learn from a completed conversation.

        Args:
            session_id: Session identifier
            messages: Conversation messages
            actions_taken: Actions taken during conversation
        """
        # Update conversation summary
        self.episodic.end_conversation(
            session_id,
            summary=f"Conversation with {len(messages)} messages, {len(actions_taken)} actions",
            outcome="completed"
        )

        # Look for patterns in action sequence
        if len(actions_taken) >= 2:
            # Record as potential pattern
            pattern_name = f"conv_{session_id[:8]}"
            self.procedural.observe_pattern(
                pattern_name,
                actions_taken,
                frequency="on_demand"
            )

        # Run consolidation
        self.consolidate()

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about learning progress."""
        return {
            "total_patterns": len(self.procedural.patterns),
            "high_confidence_patterns": sum(
                1 for p in self.procedural.patterns.values()
                if p.confidence >= self.pattern_confidence_threshold
            ),
            "inferred_preferences": sum(
                1 for p in self.semantic.preferences.values()
                if p.source == MemorySource.OBSERVED
            ),
            "corrections_analyzed": len(self.episodic.corrections),
            "lessons_learned": len(self.episodic.get_lessons_learned())
        }
