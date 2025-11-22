"""
Unified Memory System

Consolidates all memory systems into a single, intelligent memory layer with:
- Tiered injection (Always/Sometimes/On-demand)
- Memory intent detection
- Importance scoring
- Session consolidation to facts

Author: AI System
Version: 1.0
"""

import time
import json
import hashlib
import re
import math
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from enum import Enum

import chromadb
from chromadb.config import Settings
import google.generativeai as genai

from config import Config


# Constants for memory decay
DECAY_RATE = 0.02  # Halves importance every ~35 hours
SIMILARITY_THRESHOLD = 0.85  # For semantic deduplication
MAX_RECENT_ENTITIES = 20  # For coreference resolution


class MemoryTier(Enum):
    """Memory injection tiers"""
    ALWAYS = "always"         # Always inject (timezone, name, project)
    SOMETIMES = "sometimes"   # Inject when relevant (last session)
    ON_DEMAND = "on_demand"   # Only on explicit recall queries


class MemoryIntentType(Enum):
    """Types of memory-related intents"""
    RECALL = "recall"             # "what did we discuss", "last time"
    REFERENCE = "reference"       # "that ticket", "the one we talked about"
    NONE = "none"                 # No memory intent


@dataclass
class CoreFact:
    """A core fact about the user (always injected)"""
    key: str
    value: str
    category: str  # identity, preference, default
    importance: float  # 0-1
    updated_at: float
    source: str  # how we learned this (explicit, inferred)


@dataclass
class ConsolidatedFact:
    """A fact consolidated from session patterns"""
    fact: str
    confidence: float
    evidence_count: int  # how many sessions support this
    created_at: float
    last_updated: float


@dataclass
class MemoryQuery:
    """Result of memory intent detection"""
    intent_type: MemoryIntentType
    query_text: str
    time_reference: Optional[str]  # "yesterday", "last week"
    entity_references: List[str]
    confidence: float


class UnifiedMemory:
    """
    Single unified memory system with tiered injection.

    Tiers:
    - ALWAYS: Core facts (timezone, name, default project)
    - SOMETIMES: Last session context (if relevant)
    - ON_DEMAND: Full semantic search (for recall queries)
    """

    EMBEDDING_MODEL = "models/text-embedding-004"

    def __init__(
        self,
        llm_client,
        storage_dir: str = "data/unified_memory",
        verbose: bool = False
    ):
        self.llm = llm_client
        self.storage_dir = storage_dir
        self.verbose = verbose

        # Ensure directory exists
        Path(storage_dir).mkdir(parents=True, exist_ok=True)

        # Core facts (always injected)
        self.core_facts: Dict[str, CoreFact] = {}
        self._load_core_facts()

        # Consolidated facts (learned from patterns)
        self.consolidated_facts: List[ConsolidatedFact] = []
        self._load_consolidated_facts()

        # Session storage
        self.sessions: List[Dict] = []
        self._load_sessions()

        # Entity store
        self.entities: Dict[str, Dict] = {}
        self._load_entities()

        # Current session
        self.current_session: Optional[Dict] = None

        # ChromaDB for semantic search
        self._init_vector_store()

        # Embedding cache
        self._embedding_cache: Dict[str, List[float]] = {}

        # Stats
        self.stats = {
            'queries': 0,
            'cache_hits': 0,
            'consolidations': 0,
            'merges': 0,  # Semantic deduplication merges
            'coreferences_resolved': 0  # Entity coreference resolutions
        }

        if self.verbose:
            print(f"[UNIFIED MEMORY] Initialized: {len(self.core_facts)} core facts, "
                  f"{len(self.sessions)} sessions, {len(self.entities)} entities")

    def _init_vector_store(self):
        """Initialize ChromaDB for semantic search"""
        self.chroma_client = chromadb.PersistentClient(
            path=f"{self.storage_dir}/chroma",
            settings=Settings(anonymized_telemetry=False)
        )

        # Episodes collection
        self.episodes = self.chroma_client.get_or_create_collection(
            name="episodes",
            metadata={"hnsw:space": "cosine"}
        )

        # Facts collection (for semantic search on consolidated facts)
        self.facts_collection = self.chroma_client.get_or_create_collection(
            name="facts",
            metadata={"hnsw:space": "cosine"}
        )

    # =========================================================================
    # CORE FACTS (Always Tier)
    # =========================================================================

    def set_core_fact(
        self,
        key: str,
        value: str,
        category: str = "preference",
        source: str = "explicit"
    ):
        """
        Set a core fact about the user.

        These are always injected into prompts.
        """
        # Calculate importance based on category
        importance_map = {
            "identity": 1.0,     # name, role
            "preference": 0.9,   # timezone, format
            "default": 0.8,      # default project, assignee
            "context": 0.7,      # current focus
        }

        importance = importance_map.get(category, 0.7)

        self.core_facts[key] = CoreFact(
            key=key,
            value=value,
            category=category,
            importance=importance,
            updated_at=time.time(),
            source=source
        )

        self._save_core_facts()

        if self.verbose:
            print(f"[CORE FACT] Set {key} = {value} ({category}, {source})")

    def get_core_fact(self, key: str) -> Optional[str]:
        """Get a core fact value"""
        if key in self.core_facts:
            return self.core_facts[key].value
        return None

    def get_always_context(self) -> str:
        """
        Get context that should ALWAYS be injected.

        This is the minimal, essential context.
        """
        if not self.core_facts:
            return ""

        lines = ["# User Context", ""]

        # Group by category
        by_category = defaultdict(list)
        for key, fact in self.core_facts.items():
            by_category[fact.category].append(fact)

        # Sort categories by importance
        category_order = ["identity", "preference", "default", "context"]

        for category in category_order:
            if category in by_category:
                facts = by_category[category]
                for fact in sorted(facts, key=lambda f: -f.importance):
                    lines.append(f"- **{fact.key}**: {fact.value}")

        return "\n".join(lines)

    # =========================================================================
    # MEMORY INTENT DETECTION
    # =========================================================================

    def detect_memory_intent(self, message: str) -> MemoryQuery:
        """
        Detect if message contains memory-related intent.

        This determines if we need to do expensive retrieval.
        Also resolves entity coreferences ("that ticket" → "KAN-123").
        """
        message_lower = message.lower()

        # Recall patterns - user is asking about past interactions
        recall_patterns = [
            (r'\bwhat did we (discuss|talk about|do)\b', 'RECALL'),
            (r'\blast (time|session|conversation)\b', 'RECALL'),
            (r'\byesterday\b.*\b(we|i)\b', 'RECALL'),
            (r'\bremember when\b', 'RECALL'),
            (r'\bearlier (today|we)\b', 'RECALL'),
            (r'\bwhat was (that|the)\b', 'RECALL'),
            (r'\bcan you recall\b', 'RECALL'),
            (r'\bwe (discussed|talked about|mentioned)\b', 'RECALL'),
        ]

        # Reference patterns - user is referring to something from context
        reference_patterns = [
            (r'\bthat (ticket|issue|pr|project|one)\b', 'REFERENCE'),
            (r'\bthe (ticket|issue|pr|project)\b', 'REFERENCE'),
            (r'\bthe one (we|i)\b', 'REFERENCE'),
            (r'\bthe same\b', 'REFERENCE'),
            (r'\b(this|that) again\b', 'REFERENCE'),
            (r'\bit\b', 'REFERENCE'),  # Pronoun reference
        ]

        # Check recall patterns
        for pattern, intent_type in recall_patterns:
            if re.search(pattern, message_lower):
                # Extract time reference
                time_ref = self._extract_time_reference(message_lower)

                return MemoryQuery(
                    intent_type=MemoryIntentType.RECALL,
                    query_text=message,
                    time_reference=time_ref,
                    entity_references=[],
                    confidence=0.85
                )

        # Check reference patterns and resolve coreferences
        for pattern, intent_type in reference_patterns:
            match = re.search(pattern, message_lower)
            if match:
                # Resolve the reference to actual entities
                resolved_entities = self._resolve_coreference(message_lower, match)

                return MemoryQuery(
                    intent_type=MemoryIntentType.REFERENCE,
                    query_text=message,
                    time_reference=None,
                    entity_references=resolved_entities,
                    confidence=0.75 if resolved_entities else 0.6
                )

        # No memory intent
        return MemoryQuery(
            intent_type=MemoryIntentType.NONE,
            query_text=message,
            time_reference=None,
            entity_references=[],
            confidence=0.95
        )

    def _resolve_coreference(self, message: str, match: re.Match) -> List[str]:
        """
        Resolve coreferences like "that ticket", "the project", "it" to actual entities.

        Uses recent entities from current session and entity store.
        """
        resolved = []
        matched_text = match.group(0)

        # Determine what type of entity is being referenced
        entity_type_hints = {
            'ticket': ['ticket', 'issue', 'bug'],
            'pr': ['pr', 'pull request'],
            'project': ['project'],
            'channel': ['channel'],
        }

        target_type = None
        for etype, keywords in entity_type_hints.items():
            if any(kw in matched_text for kw in keywords):
                target_type = etype
                break

        # Get recent entities from current session
        if self.current_session:
            session_entities = list(self.current_session.get('entities', set()))

            # Filter by type if we know what we're looking for
            if target_type == 'ticket':
                # Look for Jira-style tickets
                tickets = [e for e in session_entities if re.match(r'^[A-Z]+-\d+$', e)]
                if tickets:
                    resolved.extend(tickets[-MAX_RECENT_ENTITIES:])
            elif target_type == 'project':
                # Look for uppercase project names
                projects = [e for e in session_entities if e.isupper() and len(e) <= 5]
                if projects:
                    resolved.extend(projects[-MAX_RECENT_ENTITIES:])
            elif target_type == 'channel':
                # Look for channel names
                channels = [e for e in session_entities if e.startswith('#')]
                if channels:
                    resolved.extend(channels[-MAX_RECENT_ENTITIES:])
            else:
                # No specific type, return most recent entities
                resolved.extend(session_entities[-MAX_RECENT_ENTITIES:])

        # If nothing found in session, check entity store
        if not resolved and self.entities:
            # Get most recently used entities
            recent_entities = sorted(
                self.entities.items(),
                key=lambda x: x[1].get('last_seen', 0),
                reverse=True
            )[:MAX_RECENT_ENTITIES]

            for entity_id, data in recent_entities:
                entity_value = data.get('value', entity_id)

                # Filter by type if known
                if target_type == 'ticket':
                    if re.match(r'^[A-Z]+-\d+$', entity_value):
                        resolved.append(entity_value)
                elif target_type == 'project':
                    if entity_value.isupper() and len(entity_value) <= 5:
                        resolved.append(entity_value)
                elif target_type == 'channel':
                    if entity_value.startswith('#'):
                        resolved.append(entity_value)
                else:
                    resolved.append(entity_value)

        if resolved:
            self.stats['coreferences_resolved'] += 1
            if self.verbose:
                print(f"[COREFERENCE] Resolved '{matched_text}' → {resolved[:3]}")

        return resolved[:5]  # Return top 5 matches

    def _extract_time_reference(self, message: str) -> Optional[str]:
        """Extract time reference from message"""
        time_patterns = {
            'yesterday': r'\byesterday\b',
            'last week': r'\blast week\b',
            'earlier today': r'\bearlier (today)?\b',
            'last time': r'\blast time\b',
            'last session': r'\blast (session|conversation)\b',
        }

        for ref, pattern in time_patterns.items():
            if re.search(pattern, message):
                return ref

        return None

    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================

    def start_session(self, session_id: str):
        """Start a new conversation session"""
        self.current_session = {
            'session_id': session_id,
            'start_time': time.time(),
            'messages': [],
            'entities': set(),
            'intents': [],
            'agents_used': set(),
            'importance_score': 0.5
        }

        if self.verbose:
            print(f"[SESSION] Started {session_id[:8]}...")

    async def add_message(
        self,
        user_message: str,
        response: str,
        agents_used: List[str] = None,
        intent_type: str = "unknown"
    ):
        """Add a message to the current session"""
        if not self.current_session:
            return

        # Store message
        self.current_session['messages'].append({
            'user': user_message[:500],
            'response': response[:500],
            'timestamp': time.time()
        })

        # Track agents
        if agents_used:
            self.current_session['agents_used'].update(agents_used)

        # Track intent
        if intent_type and intent_type not in self.current_session['intents']:
            self.current_session['intents'].append(intent_type)

        # Extract entities
        entities = self._extract_entities(user_message)
        self.current_session['entities'].update(entities)

        # Update entity store
        for entity in entities:
            self._update_entity(entity)

        # Calculate importance score
        self.current_session['importance_score'] = self._calculate_importance(
            self.current_session
        )

        # Store as episode for semantic search
        await self._store_episode(user_message, response, agents_used, intent_type)

    async def end_session(self):
        """End the current session and consolidate"""
        if not self.current_session or not self.current_session['messages']:
            return

        # Create session summary
        session_data = {
            'session_id': self.current_session['session_id'],
            'start_time': self.current_session['start_time'],
            'end_time': time.time(),
            'message_count': len(self.current_session['messages']),
            'entities': list(self.current_session['entities']),
            'intents': self.current_session['intents'],
            'agents_used': list(self.current_session['agents_used']),
            'importance_score': self.current_session['importance_score'],
            'user_messages': [m['user'] for m in self.current_session['messages'][:5]],
            'summary': await self._create_summary(self.current_session)
        }

        self.sessions.append(session_data)

        # Keep last 10 sessions
        if len(self.sessions) > 10:
            self.sessions = self.sessions[-10:]

        self._save_sessions()

        # Consolidate patterns into facts
        await self._consolidate_sessions()

        if self.verbose:
            print(f"[SESSION] Ended with {session_data['message_count']} messages, "
                  f"importance: {session_data['importance_score']:.2f}")

        self.current_session = None

    def _calculate_importance(self, session: Dict, apply_decay: bool = False) -> float:
        """
        Calculate importance score for a session with optional time decay.

        Factors:
        - Number of messages
        - Variety of intents
        - Entities involved
        - Agents used
        - Time decay (exponential)
        """
        score = 0.5  # Base score

        # More messages = more important
        msg_count = len(session.get('messages', []))
        score += min(0.2, msg_count * 0.02)

        # Variety of intents
        intent_count = len(session.get('intents', []))
        score += min(0.15, intent_count * 0.05)

        # Entities involved
        entity_count = len(session.get('entities', set()))
        score += min(0.1, entity_count * 0.02)

        # Agents used = productive session
        agent_count = len(session.get('agents_used', set()))
        score += min(0.05, agent_count * 0.02)

        base_score = min(1.0, score)

        # Apply time decay if requested
        if apply_decay and 'start_time' in session:
            age_hours = (time.time() - session['start_time']) / 3600
            decay_factor = math.exp(-DECAY_RATE * age_hours)
            return base_score * decay_factor

        return base_score

    def get_decayed_importance(self, session: Dict) -> float:
        """Get importance score with time decay applied."""
        return self._calculate_importance(session, apply_decay=True)

    async def _create_summary(self, session: Dict) -> str:
        """Create a concise summary of the session"""
        messages = session.get('messages', [])
        if not messages:
            return "Empty session"

        # Build excerpt with more content and include responses
        excerpt = []
        for msg in messages[:5]:
            user_msg = msg['user'][:300]
            response_msg = msg.get('response', '')[:200]
            excerpt.append(f"User: {user_msg}")
            if response_msg:
                excerpt.append(f"Assistant: {response_msg}")

        prompt = f"""Summarize this conversation in 2-3 sentences.
Focus on: what was requested, what was done, include specific details like names, emails, project names, ticket IDs.

Messages:
{chr(10).join(excerpt)}

Keep under 75 words. Include specific identifiers mentioned. Just the summary."""

        try:
            response = await self.llm.generate(prompt)
            return response.text.strip() if hasattr(response, 'text') else str(response).strip()
        except:
            # Fallback
            return f"{len(messages)} messages about {', '.join(list(session.get('entities', []))[:3])}"

    # =========================================================================
    # SOMETIMES TIER - Session Context
    # =========================================================================

    def get_sometimes_context(self, query: str) -> str:
        """
        Get context that should SOMETIMES be injected.

        Returns last session context if it seems relevant to the query.
        """
        if not self.sessions:
            return ""

        # Get last session
        last_session = self.sessions[-1]

        # Check relevance
        if not self._is_session_relevant(last_session, query):
            return ""

        # Format last session
        date = datetime.fromtimestamp(last_session['start_time']).strftime("%Y-%m-%d %H:%M")

        lines = [
            "# Previous Session Context",
            "",
            f"**Last session** ({date}, {last_session['message_count']} messages):",
            f"  {last_session.get('summary', 'No summary')}",
            ""
        ]

        # Show key messages for context (not just first message)
        if last_session.get('user_messages'):
            lines.append("  **Key messages from session:**")
            for i, msg in enumerate(last_session['user_messages'][:5]):
                # Show more of each message to capture important details
                msg_preview = msg[:300] if len(msg) <= 300 else f"{msg[:300]}..."
                lines.append(f"  - {msg_preview}")
            lines.append("")

        if last_session.get('entities'):
            lines.append(f"  _Entities: {', '.join(last_session['entities'][:10])}_")

        return "\n".join(lines)

    def _is_session_relevant(self, session: Dict, query: str) -> bool:
        """Check if a session is relevant to the current query"""
        query_lower = query.lower()

        # Check entity overlap
        session_entities = set(e.lower() for e in session.get('entities', []))
        query_entities = set(self._extract_entities(query))

        if session_entities & query_entities:
            return True

        # Check topic overlap
        session_topics = set(session.get('intents', []))

        # Simple topic detection in query
        topic_keywords = {
            'ticket': ['ticket', 'issue', 'bug', 'jira'],
            'calendar': ['calendar', 'meeting', 'schedule', 'event'],
            'slack': ['slack', 'message', 'channel'],
            'github': ['github', 'pr', 'pull request', 'commit'],
        }

        for topic, keywords in topic_keywords.items():
            if any(kw in query_lower for kw in keywords):
                if topic in session_topics:
                    return True

        # Recent session (within 1 hour) is always relevant
        if time.time() - session['end_time'] < 3600:
            return True

        return False

    # =========================================================================
    # ON-DEMAND TIER - Full Semantic Search
    # =========================================================================

    async def get_ondemand_context(
        self,
        query: str,
        n_results: int = 5,
        min_similarity: float = 0.7
    ) -> str:
        """
        Get context from full semantic search.

        Only called when memory intent is detected.
        """
        self.stats['queries'] += 1

        # Get query embedding
        query_embedding = await self._get_embedding(query)

        # Search episodes
        results = self.episodes.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        if not results or not results['ids'] or not results['ids'][0]:
            return ""

        lines = [
            "# Relevant Past Interactions",
            "",
        ]

        for i, episode_id in enumerate(results['ids'][0]):
            distance = results['distances'][0][i]
            similarity = 1 - distance

            if similarity < min_similarity:
                continue

            metadata = results['metadatas'][0][i]
            document = results['documents'][0][i]

            date = metadata.get('date', 'Unknown')

            lines.append(f"**{i+1}. [{date}]** (relevance: {similarity:.0%})")
            lines.append(f"   {document}")

            if metadata.get('agents'):
                lines.append(f"   _Agents: {metadata['agents']}_")
            lines.append("")

        if len(lines) <= 2:
            return ""

        return "\n".join(lines)

    async def _store_episode(
        self,
        user_message: str,
        response: str,
        agents_used: List[str],
        intent_type: str
    ):
        """Store an episode for semantic search with deduplication"""
        # Create summary for embedding - use more chars to capture full details
        summary = f"User asked: {user_message[:500]}. Response: {response[:500]}"

        # Get embedding
        embedding = await self._get_embedding(summary)

        # Check for semantic duplicates before storing
        if self.episodes.count() > 0:
            similar = self.episodes.query(
                query_embeddings=[embedding],
                n_results=1,
                include=["distances", "metadatas", "documents"]
            )

            if similar and similar['ids'] and similar['ids'][0]:
                distance = similar['distances'][0][0]
                similarity = 1 - distance

                # If very similar to existing memory, merge instead of adding
                if similarity >= SIMILARITY_THRESHOLD:
                    existing_id = similar['ids'][0][0]
                    existing_meta = similar['metadatas'][0][0]
                    existing_doc = similar['documents'][0][0]

                    # Update existing episode with merged info
                    merged_summary = f"{existing_doc} | Also: {user_message[:300]}"
                    merged_agents = existing_meta.get('agents', '')
                    if agents_used:
                        new_agents = ",".join(agents_used)
                        if merged_agents:
                            merged_agents = f"{merged_agents},{new_agents}"
                        else:
                            merged_agents = new_agents

                    # Update the existing episode
                    self.episodes.update(
                        ids=[existing_id],
                        documents=[merged_summary[:1000]],
                        metadatas=[{
                            **existing_meta,
                            "agents": merged_agents,
                            "merged_count": existing_meta.get('merged_count', 1) + 1,
                            "last_merged": time.time()
                        }]
                    )

                    self.stats['merges'] += 1
                    if self.verbose:
                        print(f"[MEMORY] Merged similar episode (similarity: {similarity:.2f})")
                    return

        # Generate ID for new episode
        episode_id = hashlib.sha256(
            f"{user_message}{time.time()}".encode()
        ).hexdigest()[:16]

        # Metadata
        metadata = {
            "timestamp": time.time(),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "agents": ",".join(agents_used) if agents_used else "",
            "intent": intent_type,
            "user_message": user_message[:500],
            "merged_count": 1
        }

        # Store new episode
        self.episodes.add(
            ids=[episode_id],
            embeddings=[embedding],
            documents=[summary],
            metadatas=[metadata]
        )

        if self.verbose:
            print(f"[MEMORY] Stored new episode {episode_id[:8]}...")

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding with caching"""
        cache_key = hashlib.md5(text.encode()).hexdigest()

        if cache_key in self._embedding_cache:
            self.stats['cache_hits'] += 1
            return self._embedding_cache[cache_key]

        try:
            result = genai.embed_content(
                model=self.EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_document"
            )
            embedding = result['embedding']

            # Cache
            self._embedding_cache[cache_key] = embedding

            # Limit cache
            if len(self._embedding_cache) > 500:
                keys = list(self._embedding_cache.keys())
                for key in keys[:50]:
                    del self._embedding_cache[key]

            return embedding
        except Exception as e:
            if self.verbose:
                print(f"[EMBEDDING] Failed: {e}")
            return [0.0] * 768

    # =========================================================================
    # CONSOLIDATION - Sessions to Facts
    # =========================================================================

    async def _consolidate_sessions(self):
        """
        Consolidate patterns from sessions into facts using LLM.

        This compresses old sessions into actionable knowledge.
        """
        if len(self.sessions) < 3:
            return

        self.stats['consolidations'] += 1

        # First do pattern-based consolidation
        patterns = self._analyze_patterns()

        for pattern_type, pattern_data in patterns.items():
            if pattern_data['count'] >= 3:
                fact = ConsolidatedFact(
                    fact=pattern_data['description'],
                    confidence=min(0.95, pattern_data['count'] * 0.15),
                    evidence_count=pattern_data['count'],
                    created_at=time.time(),
                    last_updated=time.time()
                )

                # Check if we already have this fact
                existing = next(
                    (f for f in self.consolidated_facts if f.fact == fact.fact),
                    None
                )

                if existing:
                    existing.evidence_count = pattern_data['count']
                    existing.last_updated = time.time()
                    existing.confidence = fact.confidence
                else:
                    self.consolidated_facts.append(fact)

                    if self.verbose:
                        print(f"[CONSOLIDATION] Pattern fact: {fact.fact}")

        # Now use LLM for deeper insight extraction
        await self._llm_consolidate()

        self._save_consolidated_facts()

    async def _llm_consolidate(self):
        """Use LLM to extract meaningful facts from session summaries."""
        if len(self.sessions) < 5:
            return

        # Build session summaries for LLM
        session_summaries = []
        for session in self.sessions[-10:]:  # Last 10 sessions
            summary = session.get('summary', '')
            entities = ', '.join(session.get('entities', [])[:5])
            intents = ', '.join(session.get('intents', []))
            session_summaries.append(
                f"- {summary} [Entities: {entities}] [Actions: {intents}]"
            )

        prompt = f"""Analyze these conversation session summaries and extract 3-5 key facts about the user's work patterns, preferences, or common tasks.

Sessions:
{chr(10).join(session_summaries)}

Extract facts that would be useful to remember for future conversations. Focus on:
- Projects or tools they work with regularly
- Common workflows or task patterns
- Preferences in how they work

Return ONLY a JSON array of strings, each being a single concise fact:
["fact 1", "fact 2", "fact 3"]

Keep each fact under 15 words. Be specific and actionable."""

        try:
            response = await self.llm.generate(prompt)
            text = response.text if hasattr(response, 'text') else str(response)

            # Parse JSON
            start = text.find('[')
            end = text.rfind(']') + 1
            if start >= 0 and end > start:
                facts = json.loads(text[start:end])

                for fact_text in facts[:5]:  # Max 5 facts
                    if not isinstance(fact_text, str):
                        continue

                    # Check if this fact is already known
                    is_duplicate = any(
                        f.fact.lower() == fact_text.lower() or
                        fact_text.lower() in f.fact.lower() or
                        f.fact.lower() in fact_text.lower()
                        for f in self.consolidated_facts
                    )

                    if not is_duplicate:
                        new_fact = ConsolidatedFact(
                            fact=fact_text,
                            confidence=0.75,  # LLM-derived facts start at 0.75
                            evidence_count=len(self.sessions),
                            created_at=time.time(),
                            last_updated=time.time()
                        )
                        self.consolidated_facts.append(new_fact)

                        if self.verbose:
                            print(f"[CONSOLIDATION] LLM fact: {fact_text}")

        except Exception as e:
            if self.verbose:
                print(f"[CONSOLIDATION] LLM consolidation failed: {e}")

    def _analyze_patterns(self) -> Dict[str, Dict]:
        """Analyze patterns across sessions"""
        patterns = {}

        # Entity frequency
        entity_counts = defaultdict(int)
        for session in self.sessions:
            for entity in session.get('entities', []):
                entity_counts[entity] += 1

        for entity, count in entity_counts.items():
            if count >= 3:
                patterns[f"entity_{entity}"] = {
                    'description': f"User frequently works with {entity}",
                    'count': count
                }

        # Intent patterns
        intent_counts = defaultdict(int)
        for session in self.sessions:
            for intent in session.get('intents', []):
                intent_counts[intent] += 1

        for intent, count in intent_counts.items():
            if count >= 3:
                patterns[f"intent_{intent}"] = {
                    'description': f"User often performs {intent} actions",
                    'count': count
                }

        # Agent preferences
        agent_counts = defaultdict(int)
        for session in self.sessions:
            for agent in session.get('agents_used', []):
                agent_counts[agent] += 1

        for agent, count in agent_counts.items():
            if count >= 3:
                patterns[f"agent_{agent}"] = {
                    'description': f"User regularly uses {agent}",
                    'count': count
                }

        return patterns

    def get_consolidated_facts_context(self) -> str:
        """Get consolidated facts for context injection"""
        if not self.consolidated_facts:
            return ""

        lines = [
            "# Learned Patterns",
            "",
        ]

        # Sort by confidence
        sorted_facts = sorted(
            self.consolidated_facts,
            key=lambda f: f.confidence,
            reverse=True
        )[:5]

        for fact in sorted_facts:
            lines.append(f"- {fact.fact} (confidence: {fact.confidence:.0%})")

        return "\n".join(lines)

    # =========================================================================
    # ENTITY MANAGEMENT
    # =========================================================================

    def _extract_entities(self, message: str) -> List[str]:
        """Extract entities from message"""
        entities = []

        # Jira issues
        jira = re.findall(r'\b([A-Z]{2,10}-\d+)\b', message)
        entities.extend(jira)

        # Mentions
        mentions = re.findall(r'@([\w.-]+)', message)
        entities.extend([f"@{m}" for m in mentions])

        # Channels
        channels = re.findall(r'#([\w-]+)', message)
        entities.extend([f"#{c}" for c in channels])

        # Projects (uppercase)
        projects = re.findall(r'\b([A-Z]{2,5})\b(?!\s*-\d)', message)
        entities.extend(projects)

        return entities

    def _update_entity(self, entity: str):
        """Update entity in store"""
        entity_lower = entity.lower()

        if entity_lower in self.entities:
            self.entities[entity_lower]['mention_count'] += 1
            self.entities[entity_lower]['last_seen'] = time.time()
        else:
            self.entities[entity_lower] = {
                'value': entity,
                'first_seen': time.time(),
                'last_seen': time.time(),
                'mention_count': 1
            }

        self._save_entities()

    def get_entity_context(self) -> str:
        """Get frequently used entities for context"""
        if not self.entities:
            return ""

        # Sort by frequency
        sorted_entities = sorted(
            self.entities.items(),
            key=lambda x: x[1]['mention_count'],
            reverse=True
        )[:10]

        if not sorted_entities:
            return ""

        lines = [
            "# Known Entities",
            "",
        ]

        for entity_id, data in sorted_entities:
            lines.append(f"- {data['value']} (used {data['mention_count']}x)")

        return "\n".join(lines)

    # =========================================================================
    # MAIN CONTEXT RETRIEVAL
    # =========================================================================

    async def get_context(self, message: str) -> Tuple[str, MemoryQuery]:
        """
        Get the appropriate context for a message.

        This is the main entry point for the orchestrator.

        Returns:
            Tuple of (context_string, memory_query)
        """
        # Detect memory intent (includes coreference resolution)
        memory_query = self.detect_memory_intent(message)

        context_parts = []

        # ALWAYS tier - Core facts
        always_context = self.get_always_context()
        if always_context:
            context_parts.append(always_context)

        # SOMETIMES tier - Last session (if relevant)
        sometimes_context = self.get_sometimes_context(message)
        if sometimes_context:
            context_parts.append(sometimes_context)

        # Consolidated facts
        facts_context = self.get_consolidated_facts_context()
        if facts_context:
            context_parts.append(facts_context)

        # ON-DEMAND tier - Only if recall intent detected
        if memory_query.intent_type == MemoryIntentType.RECALL:
            ondemand_context = await self.get_ondemand_context(message)
            if ondemand_context:
                context_parts.append(ondemand_context)

        # Entity context (for reference intents)
        if memory_query.intent_type == MemoryIntentType.REFERENCE:
            # Add resolved entity references to context
            if memory_query.entity_references:
                resolved_context = self._format_resolved_entities(memory_query.entity_references)
                if resolved_context:
                    context_parts.append(resolved_context)
            else:
                # Fall back to general entity context
                entity_context = self.get_entity_context()
                if entity_context:
                    context_parts.append(entity_context)

        return "\n\n".join(context_parts), memory_query

    def _format_resolved_entities(self, entities: List[str]) -> str:
        """Format resolved entity references for context injection."""
        if not entities:
            return ""

        lines = [
            "# Resolved References",
            "",
            "Based on recent context, these entities are likely being referenced:",
            ""
        ]

        for entity in entities[:5]:
            # Get additional info from entity store
            entity_lower = entity.lower()
            if entity_lower in self.entities:
                data = self.entities[entity_lower]
                lines.append(f"- **{entity}** (used {data.get('mention_count', 1)}x)")
            else:
                lines.append(f"- **{entity}**")

        return "\n".join(lines)

    def get_statistics(self) -> Dict:
        """Get memory statistics"""
        return {
            'core_facts': len(self.core_facts),
            'consolidated_facts': len(self.consolidated_facts),
            'sessions': len(self.sessions),
            'entities': len(self.entities),
            'episodes': self.episodes.count(),
            'queries': self.stats['queries'],
            'cache_hits': self.stats['cache_hits'],
            'consolidations': self.stats['consolidations'],
            'merges': self.stats['merges'],
            'coreferences_resolved': self.stats['coreferences_resolved']
        }

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def _save_core_facts(self):
        """Save core facts to disk"""
        try:
            data = {key: asdict(fact) for key, fact in self.core_facts.items()}
            with open(f"{self.storage_dir}/core_facts.json", 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            if self.verbose:
                print(f"[SAVE] Core facts error: {e}")

    def _load_core_facts(self):
        """Load core facts from disk"""
        try:
            path = f"{self.storage_dir}/core_facts.json"
            if Path(path).exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                for key, fact_data in data.items():
                    self.core_facts[key] = CoreFact(**fact_data)
        except Exception as e:
            if self.verbose:
                print(f"[LOAD] Core facts error: {e}")

    def _save_consolidated_facts(self):
        """Save consolidated facts to disk"""
        try:
            data = [asdict(fact) for fact in self.consolidated_facts]
            with open(f"{self.storage_dir}/consolidated_facts.json", 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            if self.verbose:
                print(f"[SAVE] Consolidated facts error: {e}")

    def _load_consolidated_facts(self):
        """Load consolidated facts from disk"""
        try:
            path = f"{self.storage_dir}/consolidated_facts.json"
            if Path(path).exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                self.consolidated_facts = [ConsolidatedFact(**d) for d in data]
        except Exception as e:
            if self.verbose:
                print(f"[LOAD] Consolidated facts error: {e}")

    def _save_sessions(self):
        """Save sessions to disk"""
        try:
            with open(f"{self.storage_dir}/sessions.json", 'w') as f:
                json.dump({'sessions': self.sessions}, f, indent=2)
        except Exception as e:
            if self.verbose:
                print(f"[SAVE] Sessions error: {e}")

    def _load_sessions(self):
        """Load sessions from disk"""
        try:
            path = f"{self.storage_dir}/sessions.json"
            if Path(path).exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                self.sessions = data.get('sessions', [])
        except Exception as e:
            if self.verbose:
                print(f"[LOAD] Sessions error: {e}")

    def _save_entities(self):
        """Save entities to disk"""
        try:
            with open(f"{self.storage_dir}/entities.json", 'w') as f:
                json.dump(self.entities, f, indent=2)
        except Exception as e:
            if self.verbose:
                print(f"[SAVE] Entities error: {e}")

    def _load_entities(self):
        """Load entities from disk"""
        try:
            path = f"{self.storage_dir}/entities.json"
            if Path(path).exists():
                with open(path, 'r') as f:
                    self.entities = json.load(f)
        except Exception as e:
            if self.verbose:
                print(f"[LOAD] Entities error: {e}")
