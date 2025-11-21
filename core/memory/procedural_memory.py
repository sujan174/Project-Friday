"""
Procedural Memory - Patterns, Workflows, and Habits Storage.

This module manages the "how to do things" aspect of memory:
- Behavioral patterns learned from observation
- Workflows for multi-step tasks
- Tool usage patterns
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from .memory_types import (
    BehavioralPattern,
    ImportanceLevel,
    MemorySource,
    ToolUsagePattern,
    Workflow
)

logger = logging.getLogger(__name__)


class ProceduralMemory:
    """
    Manages procedural memory - patterns, workflows, and habits.

    This is the "how to do things" memory that stores:
    - Behavioral patterns (observed sequences)
    - Workflows (multi-step procedures)
    - Tool usage patterns (common parameters, contexts)
    """

    def __init__(self, storage_dir: str = "memory"):
        """
        Initialize procedural memory.

        Args:
            storage_dir: Directory for persistent storage
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # In-memory storage
        self.patterns: Dict[str, BehavioralPattern] = {}
        self.workflows: Dict[str, Workflow] = {}
        self.tool_patterns: Dict[str, ToolUsagePattern] = {}

        # Load from disk
        self._load()

    # =========================================================================
    # Behavioral Pattern Management
    # =========================================================================

    def observe_pattern(
        self,
        pattern_name: str,
        action_sequence: List[str],
        trigger_conditions: Optional[List[str]] = None,
        typical_time: str = "",
        frequency: str = "on_demand"
    ) -> BehavioralPattern:
        """
        Record an observed behavioral pattern.

        If pattern exists, reinforces it. Otherwise creates new.

        Args:
            pattern_name: Identifier for this pattern
            action_sequence: Sequence of actions
            trigger_conditions: What triggers this pattern
            typical_time: When this usually happens
            frequency: How often (daily, weekly, etc.)

        Returns:
            The pattern object
        """
        if pattern_name in self.patterns:
            pattern = self.patterns[pattern_name]
            pattern.observation_count += 1
            pattern.confidence = min(1.0, pattern.confidence + 0.05)
            pattern.last_accessed = datetime.utcnow()

            # Update if sequence changed
            if action_sequence != pattern.action_sequence:
                # Blend sequences or update
                pattern.action_sequence = action_sequence

            logger.debug(f"Reinforced pattern '{pattern_name}' (count: {pattern.observation_count})")
        else:
            pattern = BehavioralPattern(
                pattern_name=pattern_name,
                description=f"Observed pattern: {' -> '.join(action_sequence)}",
                trigger_conditions=trigger_conditions or [],
                action_sequence=action_sequence,
                typical_time=typical_time,
                frequency=frequency,
                confidence=0.3,  # Start with low confidence
                observation_count=1,
                source=MemorySource.OBSERVED,
                importance=ImportanceLevel.MEDIUM.value
            )
            self.patterns[pattern_name] = pattern
            logger.info(f"New pattern observed: {pattern_name}")

        self._save()
        return pattern

    def get_pattern(self, pattern_name: str) -> Optional[BehavioralPattern]:
        """Get a pattern by name."""
        pattern = self.patterns.get(pattern_name)
        if pattern:
            pattern.touch()
        return pattern

    def get_patterns_for_context(
        self,
        context: str,
        time_of_day: Optional[str] = None,
        min_confidence: float = 0.5
    ) -> List[BehavioralPattern]:
        """
        Get patterns applicable to a context.

        Args:
            context: Current context/trigger
            time_of_day: Current time (HH:MM)
            min_confidence: Minimum confidence threshold

        Returns:
            List of applicable patterns
        """
        context_lower = context.lower()
        results = []

        for pattern in self.patterns.values():
            if pattern.confidence < min_confidence:
                continue

            # Check trigger conditions
            matches_context = any(
                cond.lower() in context_lower or context_lower in cond.lower()
                for cond in pattern.trigger_conditions
            )

            # Check time if specified
            matches_time = True
            if time_of_day and pattern.typical_time:
                # Simple time matching (within 2 hours)
                try:
                    pattern_hour = int(pattern.typical_time.split(":")[0])
                    current_hour = int(time_of_day.split(":")[0])
                    matches_time = abs(pattern_hour - current_hour) <= 2
                except:
                    matches_time = True

            if matches_context and matches_time:
                pattern.touch()
                results.append(pattern)

        # Sort by confidence
        return sorted(results, key=lambda x: -x.confidence)

    def update_pattern_success(
        self,
        pattern_name: str,
        success: bool
    ) -> None:
        """
        Update pattern success rate after execution.

        Args:
            pattern_name: Pattern identifier
            success: Whether execution was successful
        """
        if pattern_name not in self.patterns:
            return

        pattern = self.patterns[pattern_name]

        # Update success rate with exponential moving average
        alpha = 0.1
        pattern.success_rate = (1 - alpha) * pattern.success_rate + alpha * (1.0 if success else 0.0)

        # Adjust confidence based on success rate
        if success:
            pattern.confidence = min(1.0, pattern.confidence + 0.02)
        else:
            pattern.confidence = max(0.1, pattern.confidence - 0.05)

        self._save()

    def delete_pattern(self, pattern_name: str) -> bool:
        """Delete a pattern."""
        if pattern_name in self.patterns:
            del self.patterns[pattern_name]
            self._save()
            return True
        return False

    # =========================================================================
    # Workflow Management
    # =========================================================================

    def add_workflow(
        self,
        workflow_name: str,
        description: str,
        steps: List[Dict[str, Any]],
        trigger_phrase: str = "",
        source: MemorySource = MemorySource.EXPLICIT
    ) -> Workflow:
        """
        Add or update a workflow.

        Args:
            workflow_name: Identifier
            description: What this workflow does
            steps: List of step definitions
            trigger_phrase: How user initiates this
            source: How this was created

        Returns:
            The workflow object
        """
        if workflow_name in self.workflows:
            workflow = self.workflows[workflow_name]
            workflow.description = description
            workflow.steps = steps
            workflow.trigger_phrase = trigger_phrase
            workflow.touch()
        else:
            workflow = Workflow(
                workflow_name=workflow_name,
                description=description,
                steps=steps,
                trigger_phrase=trigger_phrase,
                source=source,
                importance=ImportanceLevel.HIGH.value if source == MemorySource.EXPLICIT else ImportanceLevel.MEDIUM.value
            )
            self.workflows[workflow_name] = workflow
            logger.info(f"Added workflow: {workflow_name}")

        self._save()
        return workflow

    def get_workflow(self, workflow_name: str) -> Optional[Workflow]:
        """Get a workflow by name."""
        workflow = self.workflows.get(workflow_name)
        if workflow:
            workflow.touch()
        return workflow

    def find_workflow(self, query: str) -> Optional[Workflow]:
        """
        Find a workflow matching a query.

        Args:
            query: Search query (matches trigger phrase or name)

        Returns:
            Best matching workflow or None
        """
        query_lower = query.lower()

        for workflow in self.workflows.values():
            # Check trigger phrase
            if workflow.trigger_phrase and workflow.trigger_phrase.lower() in query_lower:
                workflow.touch()
                return workflow

            # Check name
            if workflow.workflow_name.lower() in query_lower:
                workflow.touch()
                return workflow

        return None

    def record_workflow_completion(
        self,
        workflow_name: str,
        duration_ms: float
    ) -> None:
        """
        Record completion of a workflow.

        Args:
            workflow_name: Workflow identifier
            duration_ms: How long it took
        """
        if workflow_name not in self.workflows:
            return

        workflow = self.workflows[workflow_name]
        workflow.completion_count += 1

        # Update average duration
        if workflow.average_duration_ms == 0:
            workflow.average_duration_ms = duration_ms
        else:
            # Exponential moving average
            alpha = 0.3
            workflow.average_duration_ms = (
                (1 - alpha) * workflow.average_duration_ms + alpha * duration_ms
            )

        workflow.importance = min(
            ImportanceLevel.CRITICAL.value,
            workflow.importance + 0.05 * workflow.completion_count
        )

        self._save()

    def get_all_workflows(self) -> List[Workflow]:
        """Get all workflows sorted by usage."""
        workflows = list(self.workflows.values())
        return sorted(workflows, key=lambda x: -x.completion_count)

    def delete_workflow(self, workflow_name: str) -> bool:
        """Delete a workflow."""
        if workflow_name in self.workflows:
            del self.workflows[workflow_name]
            self._save()
            return True
        return False

    # =========================================================================
    # Tool Usage Pattern Management
    # =========================================================================

    def record_tool_usage(
        self,
        agent: str,
        action: str,
        parameters: Dict[str, Any],
        context: str = "",
        success: bool = True
    ) -> ToolUsagePattern:
        """
        Record tool/agent usage to learn patterns.

        Args:
            agent: Which agent
            action: What action
            parameters: Parameters used
            context: Context of usage
            success: Whether it succeeded

        Returns:
            The usage pattern
        """
        key = f"{agent}:{action}"

        if key in self.tool_patterns:
            pattern = self.tool_patterns[key]
            pattern.usage_count += 1
            pattern.last_used = datetime.utcnow()

            # Update common parameters (keep most frequent values)
            for param_key, param_value in parameters.items():
                if param_key not in pattern.common_parameters:
                    pattern.common_parameters[param_key] = {}
                if not isinstance(pattern.common_parameters[param_key], dict):
                    pattern.common_parameters[param_key] = {}

                # Track parameter value frequency
                value_str = str(param_value)
                param_counts = pattern.common_parameters[param_key]
                param_counts[value_str] = param_counts.get(value_str, 0) + 1

            # Update success rate
            alpha = 0.1
            pattern.success_rate = (
                (1 - alpha) * pattern.success_rate + alpha * (1.0 if success else 0.0)
            )
        else:
            pattern = ToolUsagePattern(
                agent=agent,
                action=action,
                common_parameters={k: {str(v): 1} for k, v in parameters.items()},
                typical_context=context,
                usage_count=1,
                success_rate=1.0 if success else 0.0,
                source=MemorySource.OBSERVED,
                importance=ImportanceLevel.LOW.value
            )
            self.tool_patterns[key] = pattern

        self._save()
        return pattern

    def get_tool_pattern(
        self,
        agent: str,
        action: str
    ) -> Optional[ToolUsagePattern]:
        """
        Get usage pattern for a tool.

        Args:
            agent: Agent name
            action: Action name

        Returns:
            Usage pattern or None
        """
        key = f"{agent}:{action}"
        return self.tool_patterns.get(key)

    def get_common_parameters(
        self,
        agent: str,
        action: str
    ) -> Dict[str, Any]:
        """
        Get most common parameters for a tool.

        Args:
            agent: Agent name
            action: Action name

        Returns:
            Dictionary of parameter -> most common value
        """
        pattern = self.get_tool_pattern(agent, action)
        if not pattern:
            return {}

        result = {}
        for param_key, value_counts in pattern.common_parameters.items():
            if isinstance(value_counts, dict) and value_counts:
                # Get most common value
                most_common = max(value_counts.items(), key=lambda x: x[1])
                result[param_key] = most_common[0]

        return result

    def get_agent_patterns(self, agent: str) -> List[ToolUsagePattern]:
        """Get all patterns for an agent."""
        return [p for p in self.tool_patterns.values() if p.agent == agent]

    # =========================================================================
    # Pattern Analysis
    # =========================================================================

    def detect_sequence_pattern(
        self,
        recent_actions: List[str],
        min_occurrences: int = 3
    ) -> Optional[str]:
        """
        Detect if recent actions match a known pattern.

        Args:
            recent_actions: List of recent action types
            min_occurrences: Minimum pattern occurrences to consider

        Returns:
            Pattern name if found
        """
        for pattern in self.patterns.values():
            if pattern.observation_count < min_occurrences:
                continue

            seq = pattern.action_sequence
            if len(seq) > len(recent_actions):
                continue

            # Check if recent actions end with this sequence
            if recent_actions[-len(seq):] == seq:
                return pattern.pattern_name

        return None

    def suggest_next_action(
        self,
        recent_actions: List[str],
        context: str = ""
    ) -> Optional[str]:
        """
        Suggest next action based on patterns.

        Args:
            recent_actions: List of recent action types
            context: Current context

        Returns:
            Suggested next action or None
        """
        best_suggestion = None
        best_confidence = 0.0

        for pattern in self.patterns.values():
            if pattern.confidence < 0.5:
                continue

            seq = pattern.action_sequence
            if len(seq) < 2:
                continue

            # Check if recent actions match beginning of pattern
            for i in range(1, len(seq)):
                if recent_actions[-i:] == seq[:i]:
                    # We're in the middle of this pattern
                    next_action = seq[i]
                    if pattern.confidence > best_confidence:
                        best_confidence = pattern.confidence
                        best_suggestion = next_action
                    break

        return best_suggestion

    def get_stats(self) -> Dict[str, Any]:
        """Get procedural memory statistics."""
        return {
            "patterns": len(self.patterns),
            "workflows": len(self.workflows),
            "tool_patterns": len(self.tool_patterns),
            "high_confidence_patterns": sum(
                1 for p in self.patterns.values() if p.confidence > 0.7
            ),
            "total_workflow_completions": sum(
                w.completion_count for w in self.workflows.values()
            )
        }

    # =========================================================================
    # Persistence
    # =========================================================================

    def _save(self) -> None:
        """Save procedural memory to disk."""
        try:
            data = {
                "patterns": {k: v.to_dict() for k, v in self.patterns.items()},
                "workflows": {k: v.to_dict() for k, v in self.workflows.items()},
                "tool_patterns": {k: v.to_dict() for k, v in self.tool_patterns.items()}
            }

            filepath = self.storage_dir / "procedural_memory.json"
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug(f"Saved procedural memory to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save procedural memory: {e}")

    def _load(self) -> None:
        """Load procedural memory from disk."""
        filepath = self.storage_dir / "procedural_memory.json"
        if not filepath.exists():
            logger.info("No existing procedural memory found, starting fresh")
            return

        try:
            with open(filepath) as f:
                data = json.load(f)

            # Load patterns
            for key, pattern_data in data.get("patterns", {}).items():
                self.patterns[key] = BehavioralPattern.from_dict(pattern_data)

            # Load workflows
            for key, workflow_data in data.get("workflows", {}).items():
                self.workflows[key] = Workflow.from_dict(workflow_data)

            # Load tool patterns
            for key, tool_data in data.get("tool_patterns", {}).items():
                self.tool_patterns[key] = ToolUsagePattern.from_dict(tool_data)

            logger.info(
                f"Loaded procedural memory: {len(self.patterns)} patterns, "
                f"{len(self.workflows)} workflows, {len(self.tool_patterns)} tool patterns"
            )
        except Exception as e:
            logger.error(f"Failed to load procedural memory: {e}")

    def clear(self) -> None:
        """Clear all procedural memory."""
        self.patterns.clear()
        self.workflows.clear()
        self.tool_patterns.clear()
        self._save()
        logger.info("Cleared all procedural memory")

    def decay_unused_patterns(
        self,
        days_threshold: int = 30,
        min_confidence: float = 0.3
    ) -> int:
        """
        Decay confidence of unused patterns.

        Args:
            days_threshold: Days of inactivity before decay
            min_confidence: Minimum confidence before removal

        Returns:
            Number of patterns removed
        """
        cutoff = datetime.utcnow() - timedelta(days=days_threshold)
        removed = 0

        to_remove = []
        for name, pattern in self.patterns.items():
            if pattern.last_accessed < cutoff:
                # Decay confidence
                pattern.confidence *= 0.9
                if pattern.confidence < min_confidence:
                    to_remove.append(name)

        for name in to_remove:
            del self.patterns[name]
            removed += 1

        if removed > 0:
            self._save()
            logger.info(f"Removed {removed} low-confidence patterns")

        return removed
