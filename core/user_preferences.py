"""
User Preference Learning System

Learns from user behavior to personalize the experience:
- Confirmation preferences (when to ask vs auto-execute)
- Preferred agents for tasks
- Common task patterns
- Communication style preferences
- Working hours and timing

Author: AI System
Version: 1.0
"""

import json
import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, time as dt_time
from collections import Counter, defaultdict


@dataclass
class ConfirmationPreference:
    """User's learned preference for confirmations"""
    operation_pattern: str  # e.g., "jira_create", "slack_message", "github_delete"
    always_confirm: bool
    auto_execute: bool
    confidence: float = 0.0  # 0.0 - 1.0
    sample_count: int = 0  # How many times we've observed this


@dataclass
class AgentPreference:
    """User's preferred agent for a task type"""
    task_pattern: str  # e.g., "create ticket", "send message"
    preferred_agent: str
    confidence: float = 0.0
    usage_count: int = 0


@dataclass
class CommunicationStyle:
    """User's communication preferences"""
    prefers_verbose: bool = False  # Detailed explanations vs concise
    prefers_technical: bool = True  # Technical details vs simplified
    prefers_emojis: bool = False  # Use emojis in responses

    sample_count: int = 0
    confidence: float = 0.0


@dataclass
class WorkingHours:
    """User's working hours pattern"""
    typical_start_hour: int = 9  # 24-hour format
    typical_end_hour: int = 17
    active_days: Set[int] = field(default_factory=lambda: {0, 1, 2, 3, 4})  # Mon-Fri
    timezone_offset: int = 0  # UTC offset

    sample_count: int = 0
    confidence: float = 0.0


class UserPreferenceManager:
    """
    Learns and manages user preferences over time.

    Features:
    - Implicit learning from user behavior
    - Explicit preference settings
    - Confidence-based recommendations
    - Persistent storage
    """

    def __init__(
        self,
        user_id: str = "default",
        min_confidence_threshold: float = 0.7,
        verbose: bool = False
    ):
        """
        Initialize preference manager.

        Args:
            user_id: Unique identifier for this user
            min_confidence_threshold: Minimum confidence to apply learned preferences
            verbose: Enable detailed logging
        """
        self.user_id = user_id
        self.min_confidence_threshold = min_confidence_threshold
        self.verbose = verbose

        # Learned preferences
        self.confirmation_prefs: Dict[str, ConfirmationPreference] = {}
        self.agent_prefs: Dict[str, AgentPreference] = {}
        self.communication_style = CommunicationStyle()
        self.working_hours = WorkingHours()

        # Interaction history
        self.interaction_history: List[Dict[str, Any]] = []
        self.task_patterns: Dict[str, List[str]] = defaultdict(list)  # task_type -> [agent_used]

        # Statistics
        self.total_interactions = 0
        self.confirmations_given = 0
        self.confirmations_rejected = 0
        self.auto_executions = 0

    # =========================================================================
    # CONFIRMATION PREFERENCES
    # =========================================================================

    def record_confirmation_decision(
        self,
        operation_pattern: str,
        user_confirmed: bool,
        had_chance_to_edit: bool = False
    ):
        """
        Record user's decision on a confirmation prompt.

        Args:
            operation_pattern: Type of operation (e.g., "jira_delete", "slack_post")
            user_confirmed: Whether user confirmed or rejected
            had_chance_to_edit: Whether user edited parameters before confirming
        """
        if operation_pattern not in self.confirmation_prefs:
            self.confirmation_prefs[operation_pattern] = ConfirmationPreference(
                operation_pattern=operation_pattern,
                always_confirm=True,  # Start conservative
                auto_execute=False,
                confidence=0.0,
                sample_count=0
            )

        pref = self.confirmation_prefs[operation_pattern]
        pref.sample_count += 1

        if user_confirmed:
            self.confirmations_given += 1
        else:
            self.confirmations_rejected += 1

        # Update preferences based on pattern
        # If user consistently confirms without edits, maybe auto-execute is OK
        if pref.sample_count >= 5:
            confirm_rate = self.confirmations_given / (self.confirmations_given + self.confirmations_rejected)

            if confirm_rate > 0.9 and not had_chance_to_edit:
                # User almost always confirms - consider auto-execute
                pref.auto_execute = True
                pref.always_confirm = False
                pref.confidence = min(0.9, pref.sample_count / 20)
            elif confirm_rate < 0.5:
                # User often rejects - always confirm
                pref.always_confirm = True
                pref.auto_execute = False
                pref.confidence = min(0.9, pref.sample_count / 20)

        if self.verbose:
            print(f"[PREFS] Recorded confirmation: {operation_pattern} -> {'confirmed' if user_confirmed else 'rejected'}")

    def should_auto_execute(self, operation_pattern: str) -> bool:
        """Check if operation can be auto-executed based on learned preferences"""
        if operation_pattern not in self.confirmation_prefs:
            return False

        pref = self.confirmation_prefs[operation_pattern]

        # Only auto-execute if confident
        if pref.confidence >= self.min_confidence_threshold and pref.auto_execute:
            return True

        return False

    def should_always_confirm(self, operation_pattern: str) -> bool:
        """Check if operation should always be confirmed"""
        if operation_pattern not in self.confirmation_prefs:
            return True  # Default to safe side

        pref = self.confirmation_prefs[operation_pattern]
        return pref.always_confirm

    # =========================================================================
    # AGENT PREFERENCES
    # =========================================================================

    def record_agent_usage(
        self,
        task_pattern: str,
        agent_used: str,
        was_successful: bool
    ):
        """
        Record which agent was used for a task.

        Args:
            task_pattern: Type of task (e.g., "create_ticket", "send_message")
            agent_used: Agent that was used
            was_successful: Whether the task succeeded
        """
        # Track patterns
        if was_successful:
            self.task_patterns[task_pattern].append(agent_used)

        # Update agent preference
        if task_pattern not in self.agent_prefs:
            self.agent_prefs[task_pattern] = AgentPreference(
                task_pattern=task_pattern,
                preferred_agent=agent_used,
                confidence=0.0,
                usage_count=0
            )

        pref = self.agent_prefs[task_pattern]
        pref.usage_count += 1

        # Calculate most common agent for this task
        if len(self.task_patterns[task_pattern]) >= 3:
            agent_counts = Counter(self.task_patterns[task_pattern])
            most_common_agent, count = agent_counts.most_common(1)[0]

            pref.preferred_agent = most_common_agent
            pref.confidence = min(0.95, count / len(self.task_patterns[task_pattern]))

        if self.verbose:
            print(f"[PREFS] Recorded agent usage: {task_pattern} -> {agent_used} ({'success' if was_successful else 'failed'})")

    def get_preferred_agent(self, task_pattern: str) -> Optional[str]:
        """Get user's preferred agent for a task pattern"""
        if task_pattern not in self.agent_prefs:
            return None

        pref = self.agent_prefs[task_pattern]

        if pref.confidence >= self.min_confidence_threshold:
            return pref.preferred_agent

        return None

    # =========================================================================
    # COMMUNICATION STYLE
    # =========================================================================

    def record_interaction_style(
        self,
        user_message: str,
        user_requested_verbose: bool = False,
        user_requested_technical: bool = False
    ):
        """
        Learn from user's communication style.

        Args:
            user_message: User's message
            user_requested_verbose: Whether user asked for detailed explanations
            user_requested_technical: Whether user asked for technical details
        """
        self.communication_style.sample_count += 1

        # Update preferences based on explicit requests
        if user_requested_verbose:
            self.communication_style.prefers_verbose = True

        if user_requested_technical:
            self.communication_style.prefers_technical = True

        # Detect emojis in user messages
        if any(char in user_message for char in "😀😃😄😁😆😅🤣😂🙂🙃😉😊"):
            self.communication_style.prefers_emojis = True

        # Update confidence
        if self.communication_style.sample_count >= 5:
            self.communication_style.confidence = min(0.9, self.communication_style.sample_count / 20)

    def get_communication_preferences(self) -> Dict[str, bool]:
        """Get user's communication preferences"""
        if self.communication_style.confidence < self.min_confidence_threshold:
            # Use defaults
            return {
                'verbose': False,
                'technical': True,
                'emojis': False
            }

        return {
            'verbose': self.communication_style.prefers_verbose,
            'technical': self.communication_style.prefers_technical,
            'emojis': self.communication_style.prefers_emojis
        }

    # =========================================================================
    # WORKING HOURS
    # =========================================================================

    def record_interaction_time(self, timestamp: Optional[float] = None):
        """Record time of user interaction to learn working hours"""
        if timestamp is None:
            timestamp = time.time()

        dt = datetime.fromtimestamp(timestamp)
        hour = dt.hour
        weekday = dt.weekday()  # 0 = Monday

        self.working_hours.sample_count += 1

        # Update typical start/end hours (weighted average)
        if self.working_hours.sample_count == 1:
            self.working_hours.typical_start_hour = hour
            self.working_hours.typical_end_hour = hour
        else:
            # Simple running average
            if hour < 12:  # Morning - likely start
                self.working_hours.typical_start_hour = int(
                    (self.working_hours.typical_start_hour + hour) / 2
                )
            elif hour > 15:  # Afternoon/evening - likely end
                self.working_hours.typical_end_hour = int(
                    (self.working_hours.typical_end_hour + hour) / 2
                )

        # Track active days
        self.working_hours.active_days.add(weekday)

        # Update confidence
        if self.working_hours.sample_count >= 10:
            self.working_hours.confidence = min(0.9, self.working_hours.sample_count / 50)

    def is_during_working_hours(self, timestamp: Optional[float] = None) -> bool:
        """Check if timestamp is during user's typical working hours"""
        if self.working_hours.confidence < self.min_confidence_threshold:
            return True  # Assume always OK if not confident

        if timestamp is None:
            timestamp = time.time()

        dt = datetime.fromtimestamp(timestamp)
        hour = dt.hour
        weekday = dt.weekday()

        # Check if it's an active day
        if weekday not in self.working_hours.active_days:
            return False

        # Check if it's during working hours
        return self.working_hours.typical_start_hour <= hour <= self.working_hours.typical_end_hour

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert preferences to dictionary for serialization"""
        return {
            'user_id': self.user_id,
            'confirmation_prefs': {
                k: asdict(v) for k, v in self.confirmation_prefs.items()
            },
            'agent_prefs': {
                k: asdict(v) for k, v in self.agent_prefs.items()
            },
            'communication_style': asdict(self.communication_style),
            'working_hours': {
                **asdict(self.working_hours),
                'active_days': list(self.working_hours.active_days)  # Convert set to list
            },
            'statistics': {
                'total_interactions': self.total_interactions,
                'confirmations_given': self.confirmations_given,
                'confirmations_rejected': self.confirmations_rejected,
                'auto_executions': self.auto_executions
            },
            'saved_at': time.time()
        }

    def save_to_file(self, filepath: str):
        """Save preferences to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        if self.verbose:
            print(f"[PREFS] Saved preferences to {filepath}")

    def load_from_file(self, filepath: str):
        """Load preferences from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Restore confirmation preferences
            self.confirmation_prefs = {
                k: ConfirmationPreference(**v)
                for k, v in data.get('confirmation_prefs', {}).items()
            }

            # Restore agent preferences
            self.agent_prefs = {
                k: AgentPreference(**v)
                for k, v in data.get('agent_prefs', {}).items()
            }

            # Restore communication style
            comm_data = data.get('communication_style', {})
            self.communication_style = CommunicationStyle(**comm_data)

            # Restore working hours
            hours_data = data.get('working_hours', {})
            # Convert active_days list back to set
            if 'active_days' in hours_data:
                hours_data['active_days'] = set(hours_data['active_days'])
            self.working_hours = WorkingHours(**hours_data)

            # Restore statistics
            stats = data.get('statistics', {})
            self.total_interactions = stats.get('total_interactions', 0)
            self.confirmations_given = stats.get('confirmations_given', 0)
            self.confirmations_rejected = stats.get('confirmations_rejected', 0)
            self.auto_executions = stats.get('auto_executions', 0)

            if self.verbose:
                print(f"[PREFS] Loaded preferences from {filepath}")

        except Exception as e:
            if self.verbose:
                print(f"[PREFS] Failed to load preferences: {e}")

    # =========================================================================
    # ANALYTICS
    # =========================================================================

    def get_summary(self) -> str:
        """Get human-readable summary of learned preferences"""
        lines = [
            "📊 **Learned Preferences Summary**\n",
            f"User: {self.user_id}",
            f"Total Interactions: {self.total_interactions}\n"
        ]

        # Confirmation preferences
        if self.confirmation_prefs:
            lines.append("**Confirmation Preferences**:")
            for pattern, pref in self.confirmation_prefs.items():
                status = "Auto-execute" if pref.auto_execute else "Always confirm"
                lines.append(f"  • {pattern}: {status} (confidence: {pref.confidence:.0%})")
            lines.append("")

        # Agent preferences
        if self.agent_prefs:
            lines.append("**Preferred Agents**:")
            for pattern, pref in self.agent_prefs.items():
                if pref.confidence >= self.min_confidence_threshold:
                    lines.append(f"  • {pattern} → {pref.preferred_agent} (confidence: {pref.confidence:.0%})")
            lines.append("")

        # Communication style
        if self.communication_style.confidence >= self.min_confidence_threshold:
            lines.append("**Communication Style**:")
            style = self.get_communication_preferences()
            lines.append(f"  • Verbose: {style['verbose']}")
            lines.append(f"  • Technical: {style['technical']}")
            lines.append(f"  • Emojis: {style['emojis']}")
            lines.append("")

        # Working hours
        if self.working_hours.confidence >= self.min_confidence_threshold:
            lines.append("**Typical Working Hours**:")
            lines.append(f"  • {self.working_hours.typical_start_hour}:00 - {self.working_hours.typical_end_hour}:00")
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            active = ', '.join(days[d] for d in sorted(self.working_hours.active_days))
            lines.append(f"  • Active days: {active}")
            lines.append("")

        return "\n".join(lines)
