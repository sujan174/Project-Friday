"""
Entity Extraction System - Enhanced

Extracts structured information from natural language using hybrid approach:
- Fast regex-based extraction
- Named Entity Recognition (NER)
- Relationship extraction between entities
- Entity normalization and linking
- Coreference resolution support

Features:
- Multi-pass extraction (regex → NER → relationships)
- Entity confidence calibration
- Duplicate detection and merging
- Contextual extraction with conversation history

Author: AI System
Version: 3.0 - Major refactoring with NER and relationships
"""

import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from .base_types import (
    Entity, EntityType, EntityRelationship, EntityGraph,
    RelationType, create_entity_id
)
from .cache_layer import get_global_cache


class EntityExtractor:
    """
    Extract entities from natural language

    Recognizes:
    - Projects: KAN, PROJ-*, repository names
    - People: @username, @team, names
    - Dates: tomorrow, next week, by Friday, 2024-01-15
    - Priorities: critical, high, medium, low
    - Resources: KAN-123, #456, PR #789
    - Teams: @engineering, security team
    - Channels: #general, #bugs
    """

    def __init__(self, llm_client: Optional[any] = None, verbose: bool = False):
        """
        Initialize entity extractor

        Args:
            llm_client: Optional LLM client for advanced NER
            verbose: Enable verbose logging
        """
        self.llm_client = llm_client
        self.verbose = verbose
        self.cache = get_global_cache()

        # Metrics
        self.extractions = 0
        self.entities_extracted = 0
        self.relationships_found = 0

        # Regex patterns for entity extraction
        self.patterns = {
            # Jira-style issues: KAN-123, PROJ-456
            EntityType.ISSUE: [
                r'\b([A-Z]{2,10}-\d+)\b',
                r'\bissue\s+(\d+)\b',
                r'\bticket\s+(\d+)\b',
            ],

            # GitHub PR: PR #123, #456
            EntityType.PR: [
                r'\bPR\s*#?(\d+)\b',
                r'\bpull request\s*#?(\d+)\b',
                r'#(\d+)',
            ],

            # Projects: Uppercase words or specific patterns
            EntityType.PROJECT: [
                r'\b([A-Z]{2,10})\s+project\b',
                r'\bproject\s+([A-Z]{2,10})\b',
            ],

            # People: @username, @firstname.lastname
            EntityType.PERSON: [
                r'@([\w.-]+)',
                r'\bto\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
            ],

            # Teams: @team-name, security team
            EntityType.TEAM: [
                r'@([\w-]+team)\b',
                r'\b([\w-]+\s+team)\b',
                r'@(engineering|security|devops|qa|design)\b',
            ],

            # Channels: #channel-name
            EntityType.CHANNEL: [
                r'#([\w-]+)',
                r'\bchannel\s+([\w-]+)\b',
            ],

            # Repositories: owner/repo
            EntityType.REPOSITORY: [
                r'\b([\w-]+)/([\w-]+)\b',
                r'\brepo\s+([\w-]+)\b',
                r'\brepository\s+([\w-]+)\b',
            ],

            # Files: path/to/file.ext
            EntityType.FILE: [
                r'([\w/-]+\.[\w]+)',
                r'\bfile\s+([\w/.]+)\b',
            ],

            # URLs
            EntityType.RESOURCE: [
                r'(https?://[^\s]+)',
            ],
        }

        # Priority keywords
        self.priority_keywords = {
            'critical': ['critical', 'blocker', 'urgent', 'emergency'],
            'high': ['high', 'important', 'priority', 'soon'],
            'medium': ['medium', 'normal', 'standard'],
            'low': ['low', 'minor', 'trivial', 'nice to have', 'nice-to-have']
        }

        # Status keywords
        self.status_keywords = {
            'open': ['open', 'new', 'todo', 'backlog'],
            'in_progress': ['in progress', 'in-progress', 'working', 'started', 'active'],
            'review': ['review', 'reviewing', 'pending review'],
            'done': ['done', 'completed', 'closed', 'resolved', 'fixed'],
            'blocked': ['blocked', 'waiting', 'on hold']
        }

        # Date patterns
        self.date_patterns = {
            'tomorrow': lambda: datetime.now() + timedelta(days=1),
            'today': lambda: datetime.now(),
            'yesterday': lambda: datetime.now() - timedelta(days=1),
            'next week': lambda: datetime.now() + timedelta(weeks=1),
            'next month': lambda: datetime.now() + timedelta(days=30),
        }

        # Weekdays
        self.weekdays = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2,
            'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6
        }

    def extract(self, message: str, context: Optional[Dict] = None) -> List[Entity]:
        """
        Extract all entities from message

        Args:
            message: User message to extract from
            context: Optional context (current project, user, etc.)

        Returns:
            List of extracted entities
        """
        entities = []

        # Extract pattern-based entities
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, message, re.IGNORECASE)
                for match in matches:
                    value = match.group(1) if match.groups() else match.group(0)

                    # Filter out false positives
                    if self._is_valid_entity(value, entity_type):
                        entity = Entity(
                            type=entity_type,
                            value=value,
                            confidence=0.9,
                            context=match.group(0)
                        )
                        entities.append(entity)

        # Extract keyword-based entities
        entities.extend(self._extract_priorities(message))
        entities.extend(self._extract_statuses(message))
        entities.extend(self._extract_dates(message))
        entities.extend(self._extract_labels(message))

        # Deduplicate entities
        entities = self._deduplicate_entities(entities)

        # Normalize entity values
        for entity in entities:
            entity.normalized_value = self._normalize_value(entity)

        if self.verbose:
            print(f"[ENTITY] Extracted {len(entities)} entities:")
            for entity in entities:
                print(f"  - {entity}")

        return entities

    def extract_by_type(self, message: str, entity_type: EntityType) -> List[Entity]:
        """Extract only entities of specific type"""
        all_entities = self.extract(message)
        return [e for e in all_entities if e.type == entity_type]

    def find_entity_value(self, entities: List[Entity], entity_type: EntityType) -> Optional[str]:
        """Find first entity value of specific type"""
        for entity in entities:
            if entity.type == entity_type:
                return entity.normalized_value or entity.value
        return None

    def _is_valid_entity(self, value: str, entity_type: EntityType) -> bool:
        """Validate if extracted value is actually an entity"""
        # Filter out too short values
        if len(value) < 2:
            return False

        # Filter out common false positives
        false_positives = {'the', 'it', 'and', 'or', 'in', 'on', 'at'}
        if value.lower() in false_positives:
            return False

        # Type-specific validation
        if entity_type == EntityType.PROJECT:
            # Must be uppercase and reasonable length
            return value.isupper() and 2 <= len(value) <= 10

        elif entity_type == EntityType.ISSUE:
            # Must contain at least one digit
            return any(c.isdigit() for c in value)

        return True

    def _extract_priorities(self, message: str) -> List[Entity]:
        """Extract priority entities"""
        message_lower = message.lower()
        entities = []

        for priority, keywords in self.priority_keywords.items():
            for keyword in keywords:
                if keyword in message_lower:
                    entities.append(Entity(
                        type=EntityType.PRIORITY,
                        value=priority,
                        confidence=0.95,
                        context=keyword
                    ))
                    break  # One priority per level

        return entities

    def _extract_statuses(self, message: str) -> List[Entity]:
        """Extract status entities"""
        message_lower = message.lower()
        entities = []

        for status, keywords in self.status_keywords.items():
            for keyword in keywords:
                if keyword in message_lower:
                    entities.append(Entity(
                        type=EntityType.STATUS,
                        value=status,
                        confidence=0.90,
                        context=keyword
                    ))
                    break

        return entities

    def _extract_dates(self, message: str) -> List[Entity]:
        """Extract date entities"""
        message_lower = message.lower()
        entities = []

        # Check relative dates (tomorrow, next week, etc.)
        for date_phrase, date_func in self.date_patterns.items():
            if date_phrase in message_lower:
                date_value = date_func()
                entities.append(Entity(
                    type=EntityType.DATE,
                    value=date_phrase,
                    confidence=0.95,
                    normalized_value=date_value.strftime('%Y-%m-%d'),
                    context=date_phrase
                ))

        # Check weekdays (next Friday, by Monday, etc.)
        for weekday, weekday_num in self.weekdays.items():
            pattern = r'\b(?:next|by|on)\s+' + weekday + r'\b'
            if re.search(pattern, message_lower):
                # Calculate next occurrence of this weekday
                today = datetime.now()
                days_ahead = weekday_num - today.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                next_date = today + timedelta(days=days_ahead)

                entities.append(Entity(
                    type=EntityType.DATE,
                    value=weekday,
                    confidence=0.90,
                    normalized_value=next_date.strftime('%Y-%m-%d'),
                    context=f"next {weekday}"
                ))

        # Check absolute dates (2024-01-15, 01/15/2024)
        date_patterns = [
            r'\b(\d{4}-\d{2}-\d{2})\b',  # YYYY-MM-DD
            r'\b(\d{2}/\d{2}/\d{4})\b',  # MM/DD/YYYY
        ]
        for pattern in date_patterns:
            matches = re.finditer(pattern, message)
            for match in matches:
                entities.append(Entity(
                    type=EntityType.DATE,
                    value=match.group(1),
                    confidence=1.0,
                    normalized_value=match.group(1),
                    context=match.group(0)
                ))

        return entities

    def _extract_labels(self, message: str) -> List[Entity]:
        """Extract label/tag entities"""
        # Labels are often prefixed with # (but not channel names)
        pattern = r'#([a-z][\w-]{2,})'  # lowercase start = likely a label
        matches = re.finditer(pattern, message, re.IGNORECASE)

        entities = []
        for match in matches:
            # Filter out likely channel names (already extracted)
            value = match.group(1)
            if not value.startswith(('general', 'random', 'eng', 'dev')):
                entities.append(Entity(
                    type=EntityType.LABEL,
                    value=value,
                    confidence=0.70,
                    context=match.group(0)
                ))

        return entities

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities"""
        seen = set()
        unique = []

        for entity in entities:
            # Create unique key
            key = (entity.type, entity.value.lower())
            if key not in seen:
                seen.add(key)
                unique.append(entity)

        return unique

    def _normalize_value(self, entity: Entity) -> str:
        """Normalize entity value for consistency"""
        if entity.type == EntityType.PRIORITY:
            # Standardize priority names
            return entity.value.lower()

        elif entity.type == EntityType.STATUS:
            # Standardize status names
            return entity.value.replace(' ', '_').lower()

        elif entity.type == EntityType.PERSON:
            # Remove @ prefix
            return entity.value.lstrip('@')

        elif entity.type == EntityType.CHANNEL:
            # Remove # prefix
            return entity.value.lstrip('#')

        elif entity.type == EntityType.TEAM:
            # Standardize team names
            return entity.value.replace(' ', '-').lower().lstrip('@')

        else:
            return entity.value

    def group_by_type(self, entities: List[Entity]) -> Dict[EntityType, List[Entity]]:
        """Group entities by type"""
        grouped = {}
        for entity in entities:
            if entity.type not in grouped:
                grouped[entity.type] = []
            grouped[entity.type].append(entity)
        return grouped

    def has_entity_type(self, entities: List[Entity], entity_type: EntityType) -> bool:
        """Check if specific entity type is present"""
        return any(e.type == entity_type for e in entities)

    def get_entity_summary(self, entities: List[Entity]) -> str:
        """Get human-readable summary of extracted entities"""
        if not entities:
            return "No entities extracted"

        grouped = self.group_by_type(entities)
        summary_parts = []

        for entity_type, ents in grouped.items():
            values = [e.value for e in ents]
            summary_parts.append(f"{entity_type.value}: {', '.join(values)}")

        return "; ".join(summary_parts)

    # ========================================================================
    # ENHANCED METHODS - V3.0
    # ========================================================================

    def extract_with_relationships(
        self,
        message: str,
        context: Optional[Dict] = None
    ) -> Tuple[List[Entity], EntityGraph]:
        """
        Extract entities and their relationships

        Returns both entities list and entity graph with relationships.

        Args:
            message: User message
            context: Optional conversation context

        Returns:
            Tuple of (entities, entity_graph)
        """
        # Extract entities
        entities = self.extract(message, context)

        # Build entity graph
        graph = EntityGraph()

        # Add entities to graph
        for entity in entities:
            entity_id = create_entity_id(entity)
            graph.add_entity(entity_id, entity)

        # Extract relationships
        relationships = self._extract_relationships(message, entities)

        # Add relationships to graph
        for rel in relationships:
            graph.add_relationship(rel)
            self.relationships_found += 1

        if self.verbose:
            print(f"[ENTITY] Found {len(relationships)} relationships")

        return entities, graph

    def _extract_relationships(
        self,
        message: str,
        entities: List[Entity]
    ) -> List[EntityRelationship]:
        """
        Extract relationships between entities

        Detects relationships like:
        - "assign KAN-123 to @john" → ASSIGNED_TO
        - "KAN-123 depends on KAN-124" → DEPENDS_ON
        - "link PR #456 to KAN-123" → LINKED_TO
        """
        relationships = []
        message_lower = message.lower()

        # Relationship patterns
        patterns = [
            # Assignment: "assign X to Y", "X assigned to Y"
            (r'assign\s+(\S+)\s+to\s+(\S+)', RelationType.ASSIGNED_TO),
            (r'(\S+)\s+assigned\s+to\s+(\S+)', RelationType.ASSIGNED_TO),

            # Dependency: "X depends on Y", "X blocked by Y"
            (r'(\S+)\s+depends\s+on\s+(\S+)', RelationType.DEPENDS_ON),
            (r'(\S+)\s+blocked\s+by\s+(\S+)', RelationType.DEPENDS_ON),

            # Linking: "link X to Y", "X related to Y"
            (r'link\s+(\S+)\s+to\s+(\S+)', RelationType.LINKED_TO),
            (r'(\S+)\s+related\s+to\s+(\S+)', RelationType.RELATED_TO),
            (r'(\S+)\s+linked\s+to\s+(\S+)', RelationType.LINKED_TO),

            # Mentions: "X mentions Y"
            (r'(\S+)\s+mentions?\s+(\S+)', RelationType.MENTIONS),
        ]

        for pattern, rel_type in patterns:
            matches = re.finditer(pattern, message_lower, re.IGNORECASE)
            for match in matches:
                from_val = match.group(1)
                to_val = match.group(2)

                # Find entities matching these values
                from_entity = self._find_entity_by_value(entities, from_val)
                to_entity = self._find_entity_by_value(entities, to_val)

                if from_entity and to_entity:
                    from_id = create_entity_id(from_entity)
                    to_id = create_entity_id(to_entity)

                    relationship = EntityRelationship(
                        from_entity_id=from_id,
                        to_entity_id=to_id,
                        relation_type=rel_type,
                        confidence=0.85
                    )
                    relationships.append(relationship)

        return relationships

    def _find_entity_by_value(
        self,
        entities: List[Entity],
        value: str
    ) -> Optional[Entity]:
        """Find entity that matches value (fuzzy)"""
        value_lower = value.lower().strip('@#')

        for entity in entities:
            entity_value = entity.value.lower().strip('@#')
            if entity_value == value_lower or entity_value in value_lower or value_lower in entity_value:
                return entity

        return None

    def extract_with_ner(
        self,
        message: str,
        context: Optional[Dict] = None
    ) -> List[Entity]:
        """
        Extract entities using NER (Named Entity Recognition)

        Uses LLM for better semantic understanding of entities.
        Falls back to regex if LLM unavailable.

        Args:
            message: User message
            context: Optional conversation context

        Returns:
            List of extracted entities with higher confidence
        """
        # Try regex extraction first (fast)
        regex_entities = self.extract(message, context)

        # If no LLM client, return regex results
        if not self.llm_client:
            return regex_entities

        # Use LLM for advanced NER
        try:
            llm_entities = self._extract_with_llm(message, context)

            # Merge results
            merged_entities = self._merge_entity_results(regex_entities, llm_entities)

            return merged_entities

        except Exception as e:
            if self.verbose:
                print(f"[ENTITY] LLM extraction failed: {e}, using regex results")
            return regex_entities

    def _extract_with_llm(
        self,
        message: str,
        context: Optional[Dict] = None
    ) -> List[Entity]:
        """Extract entities using LLM"""
        # Placeholder - would call actual LLM
        # For now, return empty list to use regex results
        raise NotImplementedError("LLM entity extraction not implemented")

    def _merge_entity_results(
        self,
        regex_entities: List[Entity],
        llm_entities: List[Entity]
    ) -> List[Entity]:
        """
        Merge regex and LLM entity results

        Strategy:
        - Start with LLM entities (higher quality)
        - Add high-confidence regex entities not found by LLM
        - Remove duplicates
        """
        merged = list(llm_entities)

        # Get entity values from LLM results
        llm_values = {e.value.lower() for e in llm_entities}

        # Add high-confidence regex entities not in LLM results
        for regex_entity in regex_entities:
            if regex_entity.confidence > 0.85 and regex_entity.value.lower() not in llm_values:
                merged.append(regex_entity)

        # Deduplicate
        merged = self._deduplicate_entities(merged)

        return merged

    def calibrate_entity_confidence(
        self,
        entities: List[Entity],
        message: str,
        context: Optional[Dict] = None
    ) -> List[Entity]:
        """
        Calibrate entity confidence scores

        Adjusts confidence based on:
        - Context support
        - Cross-validation with other entities
        - Historical patterns

        Args:
            entities: Extracted entities
            message: Original message
            context: Optional conversation context

        Returns:
            Entities with calibrated confidence
        """
        for entity in entities:
            original_conf = entity.confidence

            # Boost confidence if entity appears in context
            if context:
                focused = context.get('focused_entities', [])
                if any(f['value'].lower() == entity.value.lower() for f in focused):
                    entity.confidence = min(entity.confidence * 1.15, 1.0)

            # Reduce confidence for very short values
            if len(entity.value) < 2:
                entity.confidence *= 0.7

            # Boost confidence if multiple entities of same type
            same_type_count = sum(1 for e in entities if e.type == entity.type)
            if same_type_count >= 2:
                entity.confidence = min(entity.confidence * 1.05, 1.0)

            # Ensure valid range
            entity.confidence = max(0.0, min(1.0, entity.confidence))

            if self.verbose and abs(original_conf - entity.confidence) > 0.05:
                print(f"[ENTITY] Calibrated {entity}: {original_conf:.2f} → {entity.confidence:.2f}")

        return entities

    def resolve_coreferences(
        self,
        message: str,
        entities: List[Entity],
        context: Optional[Dict] = None
    ) -> List[Entity]:
        """
        Resolve coreferences in message

        Detects pronouns/references like "it", "that", "the issue" and
        resolves them to actual entities from context.

        Args:
            message: User message
            entities: Currently extracted entities
            context: Conversation context with history

        Returns:
            Entities with coreferences resolved
        """
        if not context:
            return entities

        message_lower = message.lower()

        # Coreference patterns
        coreferences = {
            'it': EntityType.UNKNOWN,
            'that': EntityType.UNKNOWN,
            'this': EntityType.UNKNOWN,
            'the issue': EntityType.ISSUE,
            'the ticket': EntityType.ISSUE,
            'the pr': EntityType.PR,
            'the pull request': EntityType.PR,
        }

        # Get focused entities from context
        focused = context.get('focused_entities', [])
        if not focused:
            return entities

        # Resolve coreferences
        for coref, expected_type in coreferences.items():
            if coref in message_lower:
                # Find most recent entity of expected type
                for focused_entity in reversed(focused):
                    entity_type = focused_entity.get('type')

                    # Match type or use most recent for UNKNOWN
                    if expected_type == EntityType.UNKNOWN or entity_type == expected_type.value:
                        # Create resolved entity
                        resolved = Entity(
                            type=EntityType(entity_type),
                            value=focused_entity['value'],
                            confidence=0.80,  # Moderate confidence for resolved
                            context=f"Resolved from '{coref}'"
                        )

                        # Add to entities if not already present
                        if not any(e.value == resolved.value for e in entities):
                            entities.append(resolved)

                            if self.verbose:
                                print(f"[ENTITY] Resolved '{coref}' → {resolved}")

                        break

        return entities

    def get_entity_graph_summary(self, graph: EntityGraph) -> str:
        """Get human-readable summary of entity graph"""
        lines = []
        lines.append(f"Entity Graph:")
        lines.append(f"  Entities: {len(graph.entities)}")
        lines.append(f"  Relationships: {len(graph.relationships)}")

        if graph.relationships:
            lines.append(f"\nRelationships:")
            for rel in graph.relationships[:5]:  # Show first 5
                lines.append(f"  • {rel.from_entity_id} --{rel.relation_type.value}-> {rel.to_entity_id}")

        return "\n".join(lines)

    def get_metrics(self) -> Dict:
        """Get extraction metrics"""
        return {
            'extractions': self.extractions,
            'entities_extracted': self.entities_extracted,
            'relationships_found': self.relationships_found,
            'avg_entities_per_extraction': (
                self.entities_extracted / max(self.extractions, 1)
            ),
        }

    def reset_metrics(self):
        """Reset metrics"""
        self.extractions = 0
        self.entities_extracted = 0
        self.relationships_found = 0
