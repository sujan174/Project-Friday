"""
Core data structures for the unified memory system.

This module defines the foundational types used across all memory components:
- Episodic Memory: What happened (conversations, actions, outcomes)
- Semantic Memory: What we know (facts, preferences, knowledge)
- Procedural Memory: How to do things (patterns, workflows, habits)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class MemoryType(Enum):
    """Types of memory in the system."""
    EPISODIC = "episodic"      # What happened
    SEMANTIC = "semantic"      # What we know
    PROCEDURAL = "procedural"  # How to do things


class ImportanceLevel(Enum):
    """Importance levels for memory prioritization."""
    CRITICAL = 1.0    # Never forget (explicit user instruction)
    HIGH = 0.8        # Important preference or fact
    MEDIUM = 0.5      # Useful but not critical
    LOW = 0.3         # Nice to have
    TRIVIAL = 0.1     # Can be forgotten easily


class MemorySource(Enum):
    """How the memory was acquired."""
    EXPLICIT = "explicit"      # User explicitly told us
    OBSERVED = "observed"      # Inferred from behavior
    DERIVED = "derived"        # Computed from other memories
    SYSTEM = "system"          # System-generated


class ActionOutcome(Enum):
    """Outcome of an action."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


class FeedbackType(Enum):
    """Type of user feedback."""
    POSITIVE = "positive"       # User was satisfied
    NEGATIVE = "negative"       # User was not satisfied
    CORRECTION = "correction"   # User corrected the output
    NEUTRAL = "neutral"         # No feedback


# =============================================================================
# Base Memory Class
# =============================================================================

@dataclass
class Memory:
    """Base class for all memory types."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    importance: float = 0.5
    source: MemorySource = MemorySource.OBSERVED
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        """Update access timestamp and count."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "importance": self.importance,
            "source": self.source.value,
            "tags": self.tags,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        """Create from dictionary."""
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["last_accessed"] = datetime.fromisoformat(data["last_accessed"])
        data["source"] = MemorySource(data["source"])
        return cls(**data)


# =============================================================================
# Semantic Memory Types (Facts, Preferences, Knowledge)
# =============================================================================

@dataclass
class UserPreference(Memory):
    """A user preference or setting."""
    category: str = ""          # e.g., "timezone", "format", "style"
    key: str = ""               # e.g., "timezone", "bullet_points"
    value: Any = None           # e.g., "America/New_York", True
    context: str = "global"     # When this applies: "global", "slack", "jira"
    confidence: float = 1.0     # How confident we are in this preference

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "type": "user_preference",
            "category": self.category,
            "key": self.key,
            "value": self.value,
            "context": self.context,
            "confidence": self.confidence
        })
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPreference":
        data = data.copy()
        data.pop("type", None)
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["last_accessed"] = datetime.fromisoformat(data["last_accessed"])
        data["source"] = MemorySource(data["source"])
        return cls(**data)


@dataclass
class Instruction(Memory):
    """An explicit instruction from the user."""
    instruction: str = ""       # The instruction text
    context: str = "global"     # When this applies
    priority: int = 1           # Higher = more important (1-10)
    active: bool = True         # Whether this instruction is active
    conditions: List[str] = field(default_factory=list)  # When to apply

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "type": "instruction",
            "instruction": self.instruction,
            "context": self.context,
            "priority": self.priority,
            "active": self.active,
            "conditions": self.conditions
        })
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Instruction":
        data = data.copy()
        data.pop("type", None)
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["last_accessed"] = datetime.fromisoformat(data["last_accessed"])
        data["source"] = MemorySource(data["source"])
        return cls(**data)


@dataclass
class Fact(Memory):
    """A piece of knowledge about the domain."""
    subject: str = ""           # What this fact is about
    predicate: str = ""         # The relationship or property
    object: str = ""            # The value or related entity
    confidence: float = 1.0     # How confident we are
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "type": "fact",
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None
        })
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Fact":
        data = data.copy()
        data.pop("type", None)
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["last_accessed"] = datetime.fromisoformat(data["last_accessed"])
        data["source"] = MemorySource(data["source"])
        if data.get("valid_from"):
            data["valid_from"] = datetime.fromisoformat(data["valid_from"])
        if data.get("valid_until"):
            data["valid_until"] = datetime.fromisoformat(data["valid_until"])
        return cls(**data)


@dataclass
class EntityKnowledge(Memory):
    """Knowledge about a specific entity (person, project, etc.)."""
    entity_type: str = ""       # "person", "project", "channel", etc.
    entity_id: str = ""         # Identifier
    entity_name: str = ""       # Display name
    properties: Dict[str, Any] = field(default_factory=dict)
    relationships: List[Dict[str, str]] = field(default_factory=list)
    notes: str = ""             # Free-form notes about this entity

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "type": "entity_knowledge",
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "entity_name": self.entity_name,
            "properties": self.properties,
            "relationships": self.relationships,
            "notes": self.notes
        })
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntityKnowledge":
        data = data.copy()
        data.pop("type", None)
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["last_accessed"] = datetime.fromisoformat(data["last_accessed"])
        data["source"] = MemorySource(data["source"])
        return cls(**data)


# =============================================================================
# Episodic Memory Types (What Happened)
# =============================================================================

@dataclass
class ConversationSummary(Memory):
    """Summary of a conversation or session."""
    session_id: str = ""
    summary: str = ""
    topics: List[str] = field(default_factory=list)
    entities_mentioned: List[str] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    outcome: str = ""
    duration_seconds: float = 0.0
    message_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "type": "conversation_summary",
            "session_id": self.session_id,
            "summary": self.summary,
            "topics": self.topics,
            "entities_mentioned": self.entities_mentioned,
            "actions_taken": self.actions_taken,
            "outcome": self.outcome,
            "duration_seconds": self.duration_seconds,
            "message_count": self.message_count
        })
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationSummary":
        data = data.copy()
        data.pop("type", None)
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["last_accessed"] = datetime.fromisoformat(data["last_accessed"])
        data["source"] = MemorySource(data["source"])
        return cls(**data)


@dataclass
class ActionRecord(Memory):
    """Record of an action taken."""
    action_type: str = ""       # "create_issue", "send_message", etc.
    agent: str = ""             # Which agent performed it
    input_summary: str = ""     # What was requested
    output_summary: str = ""    # What was the result
    outcome: ActionOutcome = ActionOutcome.SUCCESS
    feedback: FeedbackType = FeedbackType.NEUTRAL
    feedback_details: str = ""  # Any corrections or comments
    duration_ms: float = 0.0
    error_message: str = ""
    entities_involved: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "type": "action_record",
            "action_type": self.action_type,
            "agent": self.agent,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "outcome": self.outcome.value,
            "feedback": self.feedback.value,
            "feedback_details": self.feedback_details,
            "duration_ms": self.duration_ms,
            "error_message": self.error_message,
            "entities_involved": self.entities_involved
        })
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionRecord":
        data = data.copy()
        data.pop("type", None)
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["last_accessed"] = datetime.fromisoformat(data["last_accessed"])
        data["source"] = MemorySource(data["source"])
        data["outcome"] = ActionOutcome(data["outcome"])
        data["feedback"] = FeedbackType(data["feedback"])
        return cls(**data)


@dataclass
class UserCorrection(Memory):
    """Record of a user correction."""
    original_output: str = ""
    corrected_output: str = ""
    correction_type: str = ""   # "format", "content", "action", etc.
    context: str = ""           # What was the user doing
    lesson_learned: str = ""    # What should we do differently

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "type": "user_correction",
            "original_output": self.original_output,
            "corrected_output": self.corrected_output,
            "correction_type": self.correction_type,
            "context": self.context,
            "lesson_learned": self.lesson_learned
        })
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserCorrection":
        data = data.copy()
        data.pop("type", None)
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["last_accessed"] = datetime.fromisoformat(data["last_accessed"])
        data["source"] = MemorySource(data["source"])
        return cls(**data)


# =============================================================================
# Procedural Memory Types (How to Do Things)
# =============================================================================

@dataclass
class BehavioralPattern(Memory):
    """A learned behavioral pattern."""
    pattern_name: str = ""      # Identifier for this pattern
    description: str = ""       # Human-readable description
    trigger_conditions: List[str] = field(default_factory=list)
    action_sequence: List[str] = field(default_factory=list)
    typical_time: str = ""      # When this usually happens (e.g., "09:00")
    frequency: str = ""         # "daily", "weekly", "on_demand"
    confidence: float = 0.5     # How confident we are in this pattern
    observation_count: int = 0  # How many times observed
    success_rate: float = 1.0   # How often it succeeds

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "type": "behavioral_pattern",
            "pattern_name": self.pattern_name,
            "description": self.description,
            "trigger_conditions": self.trigger_conditions,
            "action_sequence": self.action_sequence,
            "typical_time": self.typical_time,
            "frequency": self.frequency,
            "confidence": self.confidence,
            "observation_count": self.observation_count,
            "success_rate": self.success_rate
        })
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BehavioralPattern":
        data = data.copy()
        data.pop("type", None)
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["last_accessed"] = datetime.fromisoformat(data["last_accessed"])
        data["source"] = MemorySource(data["source"])
        return cls(**data)


@dataclass
class Workflow(Memory):
    """A multi-step workflow."""
    workflow_name: str = ""
    description: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    # Each step: {"name": str, "agent": str, "action": str, "params": dict}
    trigger_phrase: str = ""    # How user typically initiates this
    completion_count: int = 0
    average_duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "type": "workflow",
            "workflow_name": self.workflow_name,
            "description": self.description,
            "steps": self.steps,
            "trigger_phrase": self.trigger_phrase,
            "completion_count": self.completion_count,
            "average_duration_ms": self.average_duration_ms
        })
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Workflow":
        data = data.copy()
        data.pop("type", None)
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["last_accessed"] = datetime.fromisoformat(data["last_accessed"])
        data["source"] = MemorySource(data["source"])
        return cls(**data)


@dataclass
class ToolUsagePattern(Memory):
    """Pattern of how a tool/agent is typically used."""
    agent: str = ""
    action: str = ""
    common_parameters: Dict[str, Any] = field(default_factory=dict)
    typical_context: str = ""
    success_rate: float = 1.0
    usage_count: int = 0
    last_used: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "type": "tool_usage_pattern",
            "agent": self.agent,
            "action": self.action,
            "common_parameters": self.common_parameters,
            "typical_context": self.typical_context,
            "success_rate": self.success_rate,
            "usage_count": self.usage_count,
            "last_used": self.last_used.isoformat()
        })
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolUsagePattern":
        data = data.copy()
        data.pop("type", None)
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["last_accessed"] = datetime.fromisoformat(data["last_accessed"])
        data["source"] = MemorySource(data["source"])
        data["last_used"] = datetime.fromisoformat(data["last_used"])
        return cls(**data)


# =============================================================================
# Memory Query & Context Types
# =============================================================================

@dataclass
class MemoryQuery:
    """A query to retrieve memories."""
    query_text: str = ""
    memory_types: List[MemoryType] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    min_importance: float = 0.0
    max_results: int = 10
    include_expired: bool = False
    context: str = ""           # Additional context for semantic search


@dataclass
class MemoryContext:
    """Context retrieved from memory for injection into prompts."""
    preferences: List[UserPreference] = field(default_factory=list)
    instructions: List[Instruction] = field(default_factory=list)
    relevant_facts: List[Fact] = field(default_factory=list)
    relevant_entities: List[EntityKnowledge] = field(default_factory=list)
    recent_actions: List[ActionRecord] = field(default_factory=list)
    applicable_patterns: List[BehavioralPattern] = field(default_factory=list)
    applicable_workflows: List[Workflow] = field(default_factory=list)

    def to_prompt_text(self) -> str:
        """Convert context to text suitable for prompt injection."""
        sections = []

        # User preferences
        if self.preferences:
            prefs = []
            for p in self.preferences:
                prefs.append(f"- {p.key}: {p.value}")
            sections.append("## User Preferences\n" + "\n".join(prefs))

        # Active instructions
        if self.instructions:
            instr = []
            for i in sorted(self.instructions, key=lambda x: -x.priority):
                instr.append(f"- {i.instruction}")
            sections.append("## Instructions\n" + "\n".join(instr))

        # Relevant knowledge
        if self.relevant_facts or self.relevant_entities:
            knowledge = []
            for f in self.relevant_facts:
                knowledge.append(f"- {f.subject} {f.predicate} {f.object}")
            for e in self.relevant_entities:
                knowledge.append(f"- {e.entity_name}: {e.notes}")
            sections.append("## Relevant Knowledge\n" + "\n".join(knowledge))

        # Recent context
        if self.recent_actions:
            actions = []
            for a in self.recent_actions[:3]:
                actions.append(f"- {a.action_type}: {a.output_summary}")
            sections.append("## Recent Actions\n" + "\n".join(actions))

        return "\n\n".join(sections) if sections else ""

    def is_empty(self) -> bool:
        """Check if context has any content."""
        return not any([
            self.preferences,
            self.instructions,
            self.relevant_facts,
            self.relevant_entities,
            self.recent_actions,
            self.applicable_patterns,
            self.applicable_workflows
        ])


# =============================================================================
# Memory Update Events
# =============================================================================

@dataclass
class MemoryEvent:
    """An event that triggers memory update."""
    event_type: str             # "user_message", "action_completed", "correction"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)


# Type mapping for deserialization
MEMORY_TYPE_MAP = {
    "user_preference": UserPreference,
    "instruction": Instruction,
    "fact": Fact,
    "entity_knowledge": EntityKnowledge,
    "conversation_summary": ConversationSummary,
    "action_record": ActionRecord,
    "user_correction": UserCorrection,
    "behavioral_pattern": BehavioralPattern,
    "workflow": Workflow,
    "tool_usage_pattern": ToolUsagePattern
}


def memory_from_dict(data: Dict[str, Any]) -> Memory:
    """Deserialize a memory from dictionary based on type."""
    memory_type = data.get("type")
    if memory_type not in MEMORY_TYPE_MAP:
        raise ValueError(f"Unknown memory type: {memory_type}")
    return MEMORY_TYPE_MAP[memory_type].from_dict(data)
