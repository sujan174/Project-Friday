"""
Semantic Memory - Facts, Preferences, and Knowledge Storage.

This module manages the "what we know" aspect of memory:
- User preferences (timezone, style, formats)
- Explicit instructions
- Domain knowledge (facts, entities, relationships)
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .memory_types import (
    EntityKnowledge,
    Fact,
    ImportanceLevel,
    Instruction,
    Memory,
    MemorySource,
    UserPreference,
    memory_from_dict
)

logger = logging.getLogger(__name__)


class SemanticMemory:
    """
    Manages semantic memory - facts, preferences, and knowledge.

    This is the "what we know" memory that stores:
    - User preferences and settings
    - Explicit instructions from the user
    - Facts about the domain
    - Entity knowledge (people, projects, etc.)
    """

    def __init__(self, storage_dir: str = "memory"):
        """
        Initialize semantic memory.

        Args:
            storage_dir: Directory for persistent storage
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # In-memory storage
        self.preferences: Dict[str, UserPreference] = {}
        self.instructions: Dict[str, Instruction] = {}
        self.facts: Dict[str, Fact] = {}
        self.entities: Dict[str, EntityKnowledge] = {}

        # Load from disk
        self._load()

    # =========================================================================
    # Preference Management
    # =========================================================================

    def set_preference(
        self,
        category: str,
        key: str,
        value: Any,
        context: str = "global",
        source: MemorySource = MemorySource.EXPLICIT,
        importance: float = ImportanceLevel.HIGH.value
    ) -> UserPreference:
        """
        Set a user preference.

        Args:
            category: Category of preference (e.g., "format", "timezone")
            key: Preference key
            value: Preference value
            context: Where this applies ("global", "slack", etc.)
            source: How this preference was learned
            importance: How important this preference is

        Returns:
            The created/updated preference
        """
        pref_id = f"{context}:{category}:{key}"

        # Check if updating existing
        if pref_id in self.preferences:
            pref = self.preferences[pref_id]
            pref.value = value
            pref.last_accessed = datetime.utcnow()
            pref.access_count += 1
            # Increase importance if explicitly set again
            if source == MemorySource.EXPLICIT:
                pref.importance = min(1.0, pref.importance + 0.1)
                pref.confidence = 1.0
            logger.debug(f"Updated preference: {pref_id} = {value}")
        else:
            pref = UserPreference(
                category=category,
                key=key,
                value=value,
                context=context,
                source=source,
                importance=importance,
                confidence=1.0 if source == MemorySource.EXPLICIT else 0.7
            )
            self.preferences[pref_id] = pref
            logger.info(f"New preference: {pref_id} = {value}")

        self._save()
        return pref

    def get_preference(
        self,
        key: str,
        context: str = "global",
        category: Optional[str] = None
    ) -> Optional[Any]:
        """
        Get a preference value.

        Args:
            key: Preference key
            context: Context to check (will fall back to global)
            category: Optional category filter

        Returns:
            The preference value or None
        """
        # Try specific context first, then global
        contexts_to_check = [context] if context == "global" else [context, "global"]

        for ctx in contexts_to_check:
            if category:
                pref_id = f"{ctx}:{category}:{key}"
                if pref_id in self.preferences:
                    pref = self.preferences[pref_id]
                    pref.touch()
                    return pref.value
            else:
                # Search all categories
                for pref_id, pref in self.preferences.items():
                    if pref_id.endswith(f":{key}") and pref_id.startswith(f"{ctx}:"):
                        pref.touch()
                        return pref.value

        return None

    def get_all_preferences(self, context: Optional[str] = None) -> List[UserPreference]:
        """
        Get all preferences, optionally filtered by context.

        Args:
            context: Optional context filter

        Returns:
            List of preferences
        """
        if context:
            return [p for p in self.preferences.values()
                    if p.context == context or p.context == "global"]
        return list(self.preferences.values())

    # =========================================================================
    # Instruction Management
    # =========================================================================

    def add_instruction(
        self,
        instruction: str,
        context: str = "global",
        priority: int = 5,
        conditions: Optional[List[str]] = None,
        source: MemorySource = MemorySource.EXPLICIT
    ) -> Instruction:
        """
        Add an explicit instruction.

        Args:
            instruction: The instruction text
            context: When this applies
            priority: Importance (1-10, higher = more important)
            conditions: When to apply this instruction
            source: How this was learned

        Returns:
            The created instruction
        """
        instr = Instruction(
            instruction=instruction,
            context=context,
            priority=priority,
            conditions=conditions or [],
            source=source,
            importance=ImportanceLevel.CRITICAL.value if source == MemorySource.EXPLICIT else ImportanceLevel.HIGH.value
        )

        self.instructions[instr.id] = instr
        logger.info(f"Added instruction: {instruction[:50]}...")

        self._save()
        return instr

    def get_instructions(
        self,
        context: str = "global",
        active_only: bool = True
    ) -> List[Instruction]:
        """
        Get instructions for a context.

        Args:
            context: Context to filter by
            active_only: Only return active instructions

        Returns:
            List of instructions sorted by priority
        """
        instructions = []
        for instr in self.instructions.values():
            if active_only and not instr.active:
                continue
            if instr.context == "global" or instr.context == context:
                instr.touch()
                instructions.append(instr)

        return sorted(instructions, key=lambda x: -x.priority)

    def deactivate_instruction(self, instruction_id: str) -> bool:
        """Deactivate an instruction by ID."""
        if instruction_id in self.instructions:
            self.instructions[instruction_id].active = False
            self._save()
            return True
        return False

    # =========================================================================
    # Fact Management
    # =========================================================================

    def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        source: MemorySource = MemorySource.EXPLICIT,
        tags: Optional[List[str]] = None
    ) -> Fact:
        """
        Add a fact (subject-predicate-object triple).

        Args:
            subject: What this fact is about
            predicate: The relationship/property
            obj: The value/related entity
            confidence: How confident we are
            source: How this was learned
            tags: Tags for categorization

        Returns:
            The created fact
        """
        fact = Fact(
            subject=subject,
            predicate=predicate,
            object=obj,
            confidence=confidence,
            source=source,
            tags=tags or [],
            importance=ImportanceLevel.MEDIUM.value
        )

        self.facts[fact.id] = fact
        logger.debug(f"Added fact: {subject} {predicate} {obj}")

        self._save()
        return fact

    def query_facts(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Fact]:
        """
        Query facts by subject, predicate, object, or tags.

        Args:
            subject: Filter by subject
            predicate: Filter by predicate
            obj: Filter by object
            tags: Filter by tags

        Returns:
            List of matching facts
        """
        results = []
        for fact in self.facts.values():
            if subject and subject.lower() not in fact.subject.lower():
                continue
            if predicate and predicate.lower() not in fact.predicate.lower():
                continue
            if obj and obj.lower() not in fact.object.lower():
                continue
            if tags and not any(t in fact.tags for t in tags):
                continue

            fact.touch()
            results.append(fact)

        return results

    # =========================================================================
    # Entity Knowledge Management
    # =========================================================================

    def add_entity(
        self,
        entity_type: str,
        entity_id: str,
        entity_name: str,
        properties: Optional[Dict[str, Any]] = None,
        relationships: Optional[List[Dict[str, str]]] = None,
        notes: str = "",
        source: MemorySource = MemorySource.OBSERVED
    ) -> EntityKnowledge:
        """
        Add or update entity knowledge.

        Args:
            entity_type: Type of entity (person, project, etc.)
            entity_id: Unique identifier
            entity_name: Display name
            properties: Entity properties
            relationships: Entity relationships
            notes: Free-form notes
            source: How this was learned

        Returns:
            The created/updated entity
        """
        key = f"{entity_type}:{entity_id}"

        if key in self.entities:
            entity = self.entities[key]
            entity.entity_name = entity_name
            if properties:
                entity.properties.update(properties)
            if relationships:
                entity.relationships.extend(relationships)
            if notes:
                entity.notes = notes
            entity.touch()
        else:
            entity = EntityKnowledge(
                entity_type=entity_type,
                entity_id=entity_id,
                entity_name=entity_name,
                properties=properties or {},
                relationships=relationships or [],
                notes=notes,
                source=source,
                importance=ImportanceLevel.MEDIUM.value
            )
            self.entities[key] = entity
            logger.debug(f"Added entity: {entity_type}:{entity_name}")

        self._save()
        return entity

    def get_entity(
        self,
        entity_type: str,
        entity_id: str
    ) -> Optional[EntityKnowledge]:
        """
        Get entity knowledge by type and ID.

        Args:
            entity_type: Type of entity
            entity_id: Entity identifier

        Returns:
            Entity knowledge or None
        """
        key = f"{entity_type}:{entity_id}"
        if key in self.entities:
            entity = self.entities[key]
            entity.touch()
            return entity
        return None

    def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None
    ) -> List[EntityKnowledge]:
        """
        Search entities by name or notes.

        Args:
            query: Search query
            entity_type: Optional type filter

        Returns:
            List of matching entities
        """
        query_lower = query.lower()
        results = []

        for entity in self.entities.values():
            if entity_type and entity.entity_type != entity_type:
                continue

            # Search in name and notes
            if (query_lower in entity.entity_name.lower() or
                query_lower in entity.notes.lower()):
                entity.touch()
                results.append(entity)

        return results

    def get_related_entities(
        self,
        entity_type: str,
        entity_id: str
    ) -> List[EntityKnowledge]:
        """
        Get entities related to a given entity.

        Args:
            entity_type: Type of source entity
            entity_id: Source entity ID

        Returns:
            List of related entities
        """
        source_key = f"{entity_type}:{entity_id}"
        source = self.entities.get(source_key)
        if not source:
            return []

        related = []
        for rel in source.relationships:
            rel_type = rel.get("type", "")
            rel_id = rel.get("entity_id", "")
            if rel_type and rel_id:
                entity = self.entities.get(f"{rel_type}:{rel_id}")
                if entity:
                    related.append(entity)

        return related

    # =========================================================================
    # Semantic Search
    # =========================================================================

    def search(
        self,
        query: str,
        max_results: int = 10
    ) -> List[Memory]:
        """
        Semantic search across all semantic memories.

        This is a simple keyword-based search. For more sophisticated
        semantic search, use the MemoryRetrieval module.

        Args:
            query: Search query
            max_results: Maximum results to return

        Returns:
            List of matching memories
        """
        query_lower = query.lower()
        query_terms = query_lower.split()
        results = []

        # Score and collect all memories
        all_memories: List[tuple] = []

        # Search preferences
        for pref in self.preferences.values():
            score = self._score_match(query_terms, [
                pref.key, pref.category, str(pref.value)
            ])
            if score > 0:
                all_memories.append((score, pref))

        # Search instructions
        for instr in self.instructions.values():
            score = self._score_match(query_terms, [
                instr.instruction, instr.context
            ])
            if score > 0:
                all_memories.append((score, instr))

        # Search facts
        for fact in self.facts.values():
            score = self._score_match(query_terms, [
                fact.subject, fact.predicate, fact.object
            ])
            if score > 0:
                all_memories.append((score, fact))

        # Search entities
        for entity in self.entities.values():
            score = self._score_match(query_terms, [
                entity.entity_name, entity.notes, entity.entity_type
            ])
            if score > 0:
                all_memories.append((score, entity))

        # Sort by score and return top results
        all_memories.sort(key=lambda x: -x[0])
        return [m for _, m in all_memories[:max_results]]

    def _score_match(self, query_terms: List[str], fields: List[str]) -> float:
        """Score how well fields match query terms."""
        score = 0.0
        text = " ".join(str(f).lower() for f in fields)

        for term in query_terms:
            if term in text:
                score += 1.0

        return score

    # =========================================================================
    # Persistence
    # =========================================================================

    def _save(self) -> None:
        """Save all semantic memory to disk."""
        try:
            data = {
                "preferences": {k: v.to_dict() for k, v in self.preferences.items()},
                "instructions": {k: v.to_dict() for k, v in self.instructions.items()},
                "facts": {k: v.to_dict() for k, v in self.facts.items()},
                "entities": {k: v.to_dict() for k, v in self.entities.items()}
            }

            filepath = self.storage_dir / "semantic_memory.json"
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug(f"Saved semantic memory to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save semantic memory: {e}")

    def _load(self) -> None:
        """Load semantic memory from disk."""
        filepath = self.storage_dir / "semantic_memory.json"
        if not filepath.exists():
            logger.info("No existing semantic memory found, starting fresh")
            return

        try:
            with open(filepath) as f:
                data = json.load(f)

            # Load preferences
            for key, pref_data in data.get("preferences", {}).items():
                self.preferences[key] = UserPreference.from_dict(pref_data)

            # Load instructions
            for key, instr_data in data.get("instructions", {}).items():
                self.instructions[key] = Instruction.from_dict(instr_data)

            # Load facts
            for key, fact_data in data.get("facts", {}).items():
                self.facts[key] = Fact.from_dict(fact_data)

            # Load entities
            for key, entity_data in data.get("entities", {}).items():
                self.entities[key] = EntityKnowledge.from_dict(entity_data)

            logger.info(
                f"Loaded semantic memory: {len(self.preferences)} preferences, "
                f"{len(self.instructions)} instructions, {len(self.facts)} facts, "
                f"{len(self.entities)} entities"
            )
        except Exception as e:
            logger.error(f"Failed to load semantic memory: {e}")

    def clear(self) -> None:
        """Clear all semantic memory."""
        self.preferences.clear()
        self.instructions.clear()
        self.facts.clear()
        self.entities.clear()
        self._save()
        logger.info("Cleared all semantic memory")

    def get_stats(self) -> Dict[str, int]:
        """Get memory statistics."""
        return {
            "preferences": len(self.preferences),
            "instructions": len(self.instructions),
            "facts": len(self.facts),
            "entities": len(self.entities)
        }
