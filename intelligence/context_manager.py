"""
Enhanced Context Manager

Maintains deep understanding of conversation and workspace context:
- Multi-turn conversation tracking
- Entity tracking across messages
- Resource relationship graphs
- Temporal context
- Reference resolution

Author: AI System
Version: 2.0
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime

from .base_types import (
    ConversationTurn, TrackedEntity, Entity, EntityType,
    Intent, Pattern
)


class ConversationContextManager:
    """
    Track and maintain conversation context

    Capabilities:
    - Remember conversation history
    - Track entities mentioned across turns
    - Resolve coreferences ("it", "that", "the issue")
    - Maintain topic focus
    - Understand temporal context
    """

    def __init__(self, session_id: str, verbose: bool = False):
        self.session_id = session_id
        self.verbose = verbose

        # Conversation history
        self.turns: List[ConversationTurn] = []

        # Entity tracking
        self.tracked_entities: Dict[str, TrackedEntity] = {}  # entity_id -> TrackedEntity

        # Current focus
        self.current_topic: Optional[str] = None
        self.focused_entities: List[str] = []  # Recently mentioned entity IDs

        # Temporal context
        self.current_project: Optional[str] = None
        self.current_repository: Optional[str] = None
        self.current_branch: Optional[str] = None

        # Learned patterns
        self.patterns: List[Pattern] = []

    def add_turn(
        self,
        role: str,
        message: str,
        intents: Optional[List[Intent]] = None,
        entities: Optional[List[Entity]] = None,
        tasks_executed: Optional[List[str]] = None
    ):
        """
        Add a conversation turn

        Args:
            role: 'user' or 'assistant'
            message: The message
            intents: Detected intents (if any)
            entities: Extracted entities (if any)
            tasks_executed: Task IDs executed (if any)
        """
        turn = ConversationTurn(
            role=role,
            message=message,
            timestamp=datetime.now(),
            intents=intents or [],
            entities=entities or [],
            tasks_executed=tasks_executed or []
        )

        self.turns.append(turn)

        # Track entities from this turn
        if entities:
            self._track_entities(entities)

        # Update focus
        if role == 'user':
            self._update_focus(message, entities or [])

        if self.verbose:
            print(f"[CONTEXT] Added {role} turn: {message[:50]}...")
            if entities:
                print(f"  Entities: {[str(e) for e in entities[:3]]}")

    def get_recent_turns(self, count: int = 5) -> List[ConversationTurn]:
        """Get recent conversation turns"""
        return self.turns[-count:] if self.turns else []

    def resolve_reference(self, phrase: str) -> Optional[Tuple[str, Entity]]:
        """
        Resolve ambiguous references like 'it', 'that', 'the issue'

        Args:
            phrase: The phrase to resolve

        Returns:
            (entity_id, entity) tuple if resolved, None otherwise
        """
        phrase_lower = phrase.lower().strip()

        # Exact ambiguous references
        exact_refs = {
            'it', 'that', 'this', 'them', 'those',
            'the issue', 'the ticket', 'the pr',
            'the pull request', 'the page'
        }

        if phrase_lower in exact_refs:
            # Return most recently mentioned entity
            if self.focused_entities:
                entity_id = self.focused_entities[-1]
                if entity_id in self.tracked_entities:
                    tracked = self.tracked_entities[entity_id]
                    return (entity_id, tracked.entity)

        # Type-specific references
        if phrase_lower in ['the issue', 'the ticket']:
            return self._get_most_recent_by_type(EntityType.ISSUE)

        elif phrase_lower in ['the pr', 'the pull request']:
            return self._get_most_recent_by_type(EntityType.PR)

        elif phrase_lower in ['the channel']:
            return self._get_most_recent_by_type(EntityType.CHANNEL)

        return None

    def get_relevant_context(self, current_message: str) -> Dict:
        """
        Get relevant context for current message

        Returns:
            Dictionary with relevant context information (JSON-serializable)
        """
        # Convert recent turns to JSON-serializable dicts
        recent_turns = self.get_recent_turns(3)
        recent_turns_dict = [turn.to_dict() for turn in recent_turns]

        context = {
            'recent_turns': recent_turns_dict,  # Now JSON-serializable
            'current_project': self.current_project,
            'current_repository': self.current_repository,
            'focused_entities': self._get_focused_entities(),
            'recent_tasks': self._get_recent_tasks(),
            'temporal': {
                'project': self.current_project,
                'repository': self.current_repository,
                'branch': self.current_branch
            }
        }

        return context

    def _track_entities(self, entities: List[Entity]):
        """Track entities across conversation"""
        now = datetime.now()

        for entity in entities:
            # Create entity ID
            entity_id = f"{entity.type.value}:{entity.value}"

            if entity_id in self.tracked_entities:
                # Update existing entity
                tracked = self.tracked_entities[entity_id]
                tracked.last_referenced = now
                tracked.mention_count += 1

            else:
                # Create new tracked entity
                tracked = TrackedEntity(
                    entity=entity,
                    first_mentioned=now,
                    last_referenced=now,
                    mention_count=1
                )
                self.tracked_entities[entity_id] = tracked

            # Add to focus
            if entity_id not in self.focused_entities:
                self.focused_entities.append(entity_id)

        # Keep focus list bounded
        if len(self.focused_entities) > 10:
            self.focused_entities = self.focused_entities[-10:]

        # Update temporal context
        self._update_temporal_context(entities)

    def _update_focus(self, message: str, entities: List[Entity]):
        """Update current focus based on message"""
        # Detect topic changes
        topic_keywords = {
            'authentication': ['auth', 'login', 'password', 'security'],
            'bugs': ['bug', 'issue', 'problem', 'error'],
            'features': ['feature', 'enhancement', 'new'],
            'deployment': ['deploy', 'release', 'production'],
        }

        message_lower = message.lower()
        for topic, keywords in topic_keywords.items():
            if any(kw in message_lower for kw in keywords):
                if self.current_topic != topic:
                    if self.verbose:
                        print(f"[CONTEXT] Topic changed: {self.current_topic} → {topic}")
                    self.current_topic = topic
                break

    def _update_temporal_context(self, entities: List[Entity]):
        """Update temporal context (current project, repo, etc.)"""
        for entity in entities:
            if entity.type == EntityType.PROJECT and entity.confidence > 0.8:
                self.current_project = entity.value

            elif entity.type == EntityType.REPOSITORY and entity.confidence > 0.8:
                self.current_repository = entity.value

    def _get_focused_entities(self) -> List[Dict]:
        """Get currently focused entities with details (JSON-serializable)"""
        focused = []

        for entity_id in reversed(self.focused_entities[-5:]):  # Last 5 focused
            if entity_id in self.tracked_entities:
                tracked = self.tracked_entities[entity_id]

                # Only include recent entities (within last 5 minutes)
                if tracked.is_recent(max_age_seconds=300):
                    focused.append({
                        'type': tracked.entity.type.value,
                        'value': tracked.entity.value,
                        'mentions': tracked.mention_count,
                        'last_seen': tracked.last_referenced.isoformat() if tracked.last_referenced else None  # Convert datetime to ISO string
                    })

        return focused

    def _get_recent_tasks(self) -> List[str]:
        """Get recently executed tasks"""
        tasks = []
        for turn in reversed(self.turns[-5:]):
            if turn.tasks_executed:
                tasks.extend(turn.tasks_executed)
        return tasks

    def _get_most_recent_by_type(self, entity_type: EntityType) -> Optional[Tuple[str, Entity]]:
        """Get most recently mentioned entity of specific type"""
        candidates = []

        for entity_id, tracked in self.tracked_entities.items():
            if tracked.entity.type == entity_type and tracked.is_recent():
                candidates.append((tracked.last_referenced, entity_id, tracked.entity))

        if candidates:
            # Sort by most recent
            candidates.sort(reverse=True)
            _, entity_id, entity = candidates[0]
            return (entity_id, entity)

        return None

    def add_entity_relationship(
        self,
        from_entity_id: str,
        relation_type: str,
        to_entity_id: str
    ):
        """
        Add relationship between entities

        Args:
            from_entity_id: Source entity ID
            relation_type: Type of relationship (e.g., 'assigned_to', 'linked_to')
            to_entity_id: Target entity ID
        """
        if from_entity_id in self.tracked_entities:
            tracked = self.tracked_entities[from_entity_id]
            tracked.relationships.append((relation_type, to_entity_id))

            if self.verbose:
                print(f"[CONTEXT] Relationship: {from_entity_id} --{relation_type}-> {to_entity_id}")

    def get_related_entities(self, entity_id: str) -> List[Tuple[str, str, Entity]]:
        """
        Get entities related to given entity

        Args:
            entity_id: Entity to find relationships for

        Returns:
            List of (relation_type, related_entity_id, related_entity) tuples
        """
        if entity_id not in self.tracked_entities:
            return []

        tracked = self.tracked_entities[entity_id]
        related = []

        for relation_type, related_id in tracked.relationships:
            if related_id in self.tracked_entities:
                related_entity = self.tracked_entities[related_id].entity
                related.append((relation_type, related_id, related_entity))

        return related

    def learn_pattern(
        self,
        pattern_type: str,
        pattern_data: Dict,
        success: bool = True
    ):
        """
        Learn a pattern from user behavior

        Args:
            pattern_type: Type of pattern (e.g., 'issue_creation', 'assignment')
            pattern_data: Pattern details
            success: Whether this pattern led to success
        """
        # Check if pattern exists
        existing = None
        for pattern in self.patterns:
            if pattern.pattern_type == pattern_type:
                # Check if data is similar
                if self._patterns_match(pattern.pattern_data, pattern_data):
                    existing = pattern
                    break

        if existing:
            # Update existing pattern
            existing.occurrence_count += 1
            if success:
                existing.success_count += 1
            existing.last_seen = datetime.now()

        else:
            # Create new pattern
            pattern = Pattern(
                pattern_type=pattern_type,
                pattern_data=pattern_data,
                confidence=0.5,
                occurrence_count=1,
                success_count=1 if success else 0,
                last_seen=datetime.now()
            )
            self.patterns.append(pattern)

        if self.verbose:
            print(f"[CONTEXT] Learned pattern: {pattern_type} (occurrences: {existing.occurrence_count if existing else 1})")

    def get_learned_patterns(self, pattern_type: Optional[str] = None) -> List[Pattern]:
        """Get learned patterns, optionally filtered by type"""
        if pattern_type:
            return [p for p in self.patterns if p.pattern_type == pattern_type]
        return self.patterns

    def _patterns_match(self, pattern1: Dict, pattern2: Dict, threshold: float = 0.7) -> bool:
        """Check if two pattern data dictionaries match sufficiently"""
        # Simple matching: Check if most keys match
        keys1 = set(pattern1.keys())
        keys2 = set(pattern2.keys())

        if not keys1 or not keys2:
            return False

        overlap = keys1 & keys2
        match_ratio = len(overlap) / max(len(keys1), len(keys2))

        if match_ratio < threshold:
            return False

        # Check if values for overlapping keys are similar
        matching_values = sum(
            1 for key in overlap
            if str(pattern1.get(key)) == str(pattern2.get(key))
        )

        value_match_ratio = matching_values / len(overlap) if overlap else 0
        return value_match_ratio >= threshold

    def get_context_summary(self) -> str:
        """Get human-readable summary of current context"""
        lines = []
        lines.append(f"Session: {self.session_id}")
        lines.append(f"Turns: {len(self.turns)}")

        if self.current_project:
            lines.append(f"Current Project: {self.current_project}")

        if self.current_repository:
            lines.append(f"Current Repository: {self.current_repository}")

        if self.current_topic:
            lines.append(f"Current Topic: {self.current_topic}")

        focused = self._get_focused_entities()
        if focused:
            lines.append(f"Focused Entities: {len(focused)}")
            for entity in focused[:3]:
                lines.append(f"  - {entity['type']}: {entity['value']}")

        if self.patterns:
            lines.append(f"Learned Patterns: {len(self.patterns)}")

        return "\n".join(lines)
