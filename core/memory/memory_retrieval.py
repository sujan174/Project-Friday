"""
Memory Retrieval - Semantic Search and Relevance Scoring.

This module handles intelligent retrieval of memories:
- Semantic similarity search
- Relevance scoring
- Context-aware filtering
- Memory ranking
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from .memory_types import (
    ActionRecord,
    BehavioralPattern,
    EntityKnowledge,
    Fact,
    Instruction,
    Memory,
    MemoryContext,
    MemoryQuery,
    MemoryType,
    UserPreference,
    Workflow
)
from .semantic_memory import SemanticMemory
from .episodic_memory import EpisodicMemory
from .procedural_memory import ProceduralMemory

logger = logging.getLogger(__name__)


class MemoryRetrieval:
    """
    Handles intelligent retrieval of memories.

    Features:
    - Semantic similarity search
    - Multi-factor relevance scoring
    - Context-aware filtering
    - Memory importance weighting
    """

    def __init__(
        self,
        semantic_memory: SemanticMemory,
        episodic_memory: EpisodicMemory,
        procedural_memory: ProceduralMemory
    ):
        """
        Initialize memory retrieval.

        Args:
            semantic_memory: Semantic memory instance
            episodic_memory: Episodic memory instance
            procedural_memory: Procedural memory instance
        """
        self.semantic = semantic_memory
        self.episodic = episodic_memory
        self.procedural = procedural_memory

        # Keyword weights for different contexts
        self.context_keywords = {
            "slack": ["message", "channel", "post", "slack", "dm", "thread"],
            "jira": ["issue", "ticket", "jira", "sprint", "epic", "story", "bug"],
            "github": ["pr", "pull request", "commit", "branch", "github", "repo"],
            "notion": ["page", "database", "notion", "doc", "wiki"],
            "calendar": ["meeting", "event", "calendar", "schedule", "appointment"]
        }

    def get_relevant_context(
        self,
        query: str,
        entities: Optional[List[str]] = None,
        intent: Optional[str] = None,
        agent: Optional[str] = None,
        max_memories: int = 20
    ) -> MemoryContext:
        """
        Get relevant context for a query.

        This is the main method for injecting memory into prompts.

        Args:
            query: User query or current context
            entities: Entities mentioned in query
            intent: Detected intent type
            agent: Target agent (for context filtering)
            max_memories: Maximum memories to include

        Returns:
            MemoryContext with relevant memories
        """
        context = MemoryContext()

        # Detect context from query
        detected_context = self._detect_context(query)
        if agent:
            detected_context = agent

        # Get relevant preferences
        context.preferences = self._get_relevant_preferences(
            query, detected_context
        )

        # Get applicable instructions
        context.instructions = self._get_relevant_instructions(
            query, detected_context, intent
        )

        # Get relevant facts and entities
        if entities:
            context.relevant_facts = self._get_facts_for_entities(entities)
            context.relevant_entities = self._get_entity_knowledge(entities)
        else:
            # Extract entities from query
            extracted = self._extract_entities_from_query(query)
            if extracted:
                context.relevant_facts = self._get_facts_for_entities(extracted)
                context.relevant_entities = self._get_entity_knowledge(extracted)

        # Get recent relevant actions
        context.recent_actions = self._get_relevant_actions(
            query, entities, detected_context
        )

        # Get applicable patterns and workflows
        context.applicable_patterns = self._get_applicable_patterns(
            query, detected_context
        )
        context.applicable_workflows = self._get_applicable_workflows(query)

        logger.debug(
            f"Retrieved context: {len(context.preferences)} prefs, "
            f"{len(context.instructions)} instructions, "
            f"{len(context.relevant_entities)} entities"
        )

        return context

    def search(
        self,
        query: MemoryQuery
    ) -> List[Memory]:
        """
        Search memories based on query.

        Args:
            query: Memory query specification

        Returns:
            List of matching memories sorted by relevance
        """
        results: List[Tuple[float, Memory]] = []

        # Search based on memory types
        if not query.memory_types or MemoryType.SEMANTIC in query.memory_types:
            semantic_results = self.semantic.search(
                query.query_text,
                max_results=query.max_results * 2
            )
            for memory in semantic_results:
                score = self._score_memory(memory, query)
                if score >= query.min_importance:
                    results.append((score, memory))

        if not query.memory_types or MemoryType.EPISODIC in query.memory_types:
            # Search actions
            for action in self.episodic.get_recent_actions(limit=query.max_results * 2):
                if self._matches_query(action, query):
                    score = self._score_memory(action, query)
                    if score >= query.min_importance:
                        results.append((score, action))

        if not query.memory_types or MemoryType.PROCEDURAL in query.memory_types:
            # Search patterns
            for pattern in self.procedural.patterns.values():
                if self._matches_query(pattern, query):
                    score = self._score_memory(pattern, query)
                    if score >= query.min_importance:
                        results.append((score, pattern))

        # Sort by score and return top results
        results.sort(key=lambda x: -x[0])
        return [m for _, m in results[:query.max_results]]

    # =========================================================================
    # Private Methods - Preference Retrieval
    # =========================================================================

    def _get_relevant_preferences(
        self,
        query: str,
        context: str
    ) -> List[UserPreference]:
        """Get preferences relevant to the query and context."""
        all_prefs = self.semantic.get_all_preferences(context)

        # Always include critical preferences
        critical = [p for p in all_prefs if p.importance >= 0.9]

        # Add context-specific preferences
        relevant = []
        query_lower = query.lower()

        for pref in all_prefs:
            if pref in critical:
                continue

            # Check if preference is relevant to query
            if (pref.key.lower() in query_lower or
                query_lower in pref.key.lower() or
                str(pref.value).lower() in query_lower):
                relevant.append(pref)
            # Check category relevance
            elif pref.category.lower() in query_lower:
                relevant.append(pref)

        return critical + relevant[:10]

    def _get_relevant_instructions(
        self,
        query: str,
        context: str,
        intent: Optional[str]
    ) -> List[Instruction]:
        """Get instructions relevant to the query."""
        all_instructions = self.semantic.get_instructions(context)

        relevant = []
        query_lower = query.lower()

        for instr in all_instructions:
            # Always include high-priority global instructions
            if instr.priority >= 8 and instr.context == "global":
                relevant.append(instr)
                continue

            # Check if instruction applies to current situation
            if self._instruction_applies(instr, query_lower, intent):
                relevant.append(instr)

        # Sort by priority
        return sorted(relevant, key=lambda x: -x.priority)[:10]

    def _instruction_applies(
        self,
        instr: Instruction,
        query: str,
        intent: Optional[str]
    ) -> bool:
        """Check if an instruction applies to current situation."""
        # Check conditions
        if instr.conditions:
            for condition in instr.conditions:
                cond_lower = condition.lower()
                if cond_lower in query:
                    return True
                if intent and intent.lower() in cond_lower:
                    return True
            return False

        # Check instruction text for relevance
        instr_lower = instr.instruction.lower()
        query_words = set(query.split())
        instr_words = set(instr_lower.split())

        # At least 2 word overlap
        overlap = query_words & instr_words
        return len(overlap) >= 2

    # =========================================================================
    # Private Methods - Entity/Fact Retrieval
    # =========================================================================

    def _get_facts_for_entities(self, entities: List[str]) -> List[Fact]:
        """Get facts related to entities."""
        facts = []
        for entity in entities:
            entity_facts = self.semantic.query_facts(subject=entity)
            facts.extend(entity_facts)
            # Also check as object
            entity_facts = self.semantic.query_facts(obj=entity)
            facts.extend(entity_facts)

        # Deduplicate
        seen = set()
        unique_facts = []
        for fact in facts:
            if fact.id not in seen:
                seen.add(fact.id)
                unique_facts.append(fact)

        return unique_facts[:10]

    def _get_entity_knowledge(self, entities: List[str]) -> List[EntityKnowledge]:
        """Get knowledge about entities."""
        knowledge = []
        for entity in entities:
            # Search by name
            results = self.semantic.search_entities(entity)
            knowledge.extend(results)

        # Deduplicate
        seen = set()
        unique = []
        for ent in knowledge:
            key = f"{ent.entity_type}:{ent.entity_id}"
            if key not in seen:
                seen.add(key)
                unique.append(ent)

        return unique[:10]

    def _extract_entities_from_query(self, query: str) -> List[str]:
        """Extract potential entity names from query."""
        entities = []

        # Extract quoted strings
        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)

        # Extract capitalized words (potential names)
        words = query.split()
        for word in words:
            if word[0].isupper() and len(word) > 2:
                clean = re.sub(r'[^\w]', '', word)
                if clean:
                    entities.append(clean)

        # Extract project-like patterns (e.g., PROJ-123)
        projects = re.findall(r'\b([A-Z]+-\d+)\b', query)
        entities.extend(projects)

        return list(set(entities))

    # =========================================================================
    # Private Methods - Action Retrieval
    # =========================================================================

    def _get_relevant_actions(
        self,
        query: str,
        entities: Optional[List[str]],
        context: str
    ) -> List[ActionRecord]:
        """Get recent actions relevant to the query."""
        actions = []

        # Get actions by context/agent
        if context:
            recent = self.episodic.get_recent_actions(
                limit=20,
                agent=context if context in ["slack", "jira", "github", "notion", "calendar"] else None
            )
            actions.extend(recent)

        # Get actions involving entities
        if entities:
            for entity in entities[:3]:  # Limit entity search
                entity_actions = self.episodic.get_actions_involving_entity(
                    entity, limit=5
                )
                actions.extend(entity_actions)

        # Deduplicate and sort by recency
        seen = set()
        unique = []
        for action in actions:
            if action.id not in seen:
                seen.add(action.id)
                unique.append(action)

        unique.sort(key=lambda x: x.created_at, reverse=True)
        return unique[:5]

    # =========================================================================
    # Private Methods - Pattern/Workflow Retrieval
    # =========================================================================

    def _get_applicable_patterns(
        self,
        query: str,
        context: str
    ) -> List[BehavioralPattern]:
        """Get patterns that might apply to current situation."""
        # Get time of day
        now = datetime.now()
        time_str = now.strftime("%H:%M")

        patterns = self.procedural.get_patterns_for_context(
            context or query,
            time_of_day=time_str,
            min_confidence=0.5
        )

        return patterns[:5]

    def _get_applicable_workflows(self, query: str) -> List[Workflow]:
        """Get workflows that might apply to current query."""
        workflows = []

        # Check if query matches a workflow trigger
        workflow = self.procedural.find_workflow(query)
        if workflow:
            workflows.append(workflow)

        return workflows

    # =========================================================================
    # Private Methods - Scoring & Matching
    # =========================================================================

    def _detect_context(self, query: str) -> str:
        """Detect context from query keywords."""
        query_lower = query.lower()

        scores = {}
        for context, keywords in self.context_keywords.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[context] = score

        if scores:
            return max(scores, key=scores.get)
        return "global"

    def _score_memory(
        self,
        memory: Memory,
        query: MemoryQuery
    ) -> float:
        """Score a memory for relevance to query."""
        score = 0.0

        # Base importance
        score += memory.importance * 0.3

        # Recency factor
        age_days = (datetime.utcnow() - memory.last_accessed).days
        recency = max(0, 1 - (age_days / 30))
        score += recency * 0.2

        # Access frequency
        freq = min(1.0, memory.access_count / 10)
        score += freq * 0.1

        # Query match
        match_score = self._text_match_score(memory, query.query_text)
        score += match_score * 0.4

        return min(1.0, score)

    def _text_match_score(self, memory: Memory, query: str) -> float:
        """Score text match between memory and query."""
        query_terms = set(query.lower().split())

        # Get text content based on memory type
        memory_text = ""
        if isinstance(memory, UserPreference):
            memory_text = f"{memory.key} {memory.value} {memory.category}"
        elif isinstance(memory, Instruction):
            memory_text = memory.instruction
        elif isinstance(memory, Fact):
            memory_text = f"{memory.subject} {memory.predicate} {memory.object}"
        elif isinstance(memory, EntityKnowledge):
            memory_text = f"{memory.entity_name} {memory.notes}"
        elif isinstance(memory, ActionRecord):
            memory_text = f"{memory.action_type} {memory.input_summary} {memory.output_summary}"
        elif isinstance(memory, BehavioralPattern):
            memory_text = f"{memory.pattern_name} {memory.description}"

        memory_terms = set(memory_text.lower().split())
        overlap = query_terms & memory_terms

        if not query_terms:
            return 0.0
        return len(overlap) / len(query_terms)

    def _matches_query(
        self,
        memory: Memory,
        query: MemoryQuery
    ) -> bool:
        """Check if memory matches query criteria."""
        # Check importance threshold
        if memory.importance < query.min_importance:
            return False

        # Check tags
        if query.tags:
            if not any(t in memory.tags for t in query.tags):
                return False

        # Check text match
        if query.query_text:
            score = self._text_match_score(memory, query.query_text)
            if score < 0.1:
                return False

        return True

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of all memories."""
        return {
            "semantic": self.semantic.get_stats(),
            "episodic": self.episodic.get_stats(),
            "procedural": self.procedural.get_stats()
        }

    def find_similar_memories(
        self,
        memory: Memory,
        limit: int = 5
    ) -> List[Memory]:
        """Find memories similar to a given memory."""
        # Get text representation
        if isinstance(memory, Instruction):
            query = memory.instruction
        elif isinstance(memory, UserPreference):
            query = f"{memory.key} {memory.value}"
        elif isinstance(memory, Fact):
            query = f"{memory.subject} {memory.predicate} {memory.object}"
        else:
            query = str(memory.metadata)

        # Search all memories
        memory_query = MemoryQuery(
            query_text=query,
            max_results=limit + 1  # +1 to exclude self
        )

        results = self.search(memory_query)
        return [m for m in results if m.id != memory.id][:limit]
