# Intelligence Pipeline - Unified Processing Module

import re
import math
from typing import List, Dict, Set, Optional, Any, Tuple
from datetime import datetime, timedelta

from .base_types import (
    Intent, IntentType, Entity, EntityType, EntityRelationship, EntityGraph,
    RelationType, create_entity_id, Task, ExecutionPlan, DependencyGraph,
    Confidence, ConfidenceLevel
)
from .system import get_global_cache, CacheKeyBuilder


class IntentClassifier:

    def __init__(self, llm_client: Optional[Any] = None, use_llm: bool = False, verbose: bool = False):
        self.llm_client = llm_client
        self.use_llm = use_llm
        self.verbose = verbose
        self.cache = get_global_cache()

        self.keyword_classifications = 0
        self.llm_classifications = 0
        self.cache_hits = 0

        self.intent_keywords = {
            IntentType.CREATE: {
                'primary': ['create', 'make', 'add', 'new', 'start', 'open', 'initialize', 'build', 'generate'],
                'secondary': ['set up', 'spin up', 'kick off'],
                'modifiers': ['issue', 'ticket', 'pr', 'page', 'task', 'project']
            },
            IntentType.READ: {
                'primary': ['show', 'get', 'find', 'search', 'list', 'what', 'where', 'display', 'fetch', 'retrieve'],
                'secondary': ['look up', 'check out', 'pull up'],
                'modifiers': ['status', 'details', 'info', 'information']
            },
            IntentType.UPDATE: {
                'primary': ['update', 'change', 'modify', 'edit', 'fix', 'correct', 'adjust', 'set', 'put', 'mark'],
                'secondary': ['move to', 'transition', 'reassign', 'mark as', 'put as', 'set to', 'change to', 'move it to'],
                'modifiers': ['priority', 'status', 'assignee', 'description', 'in review', 'in progress', 'done', 'blocked']
            },
            IntentType.DELETE: {
                'primary': ['delete', 'remove', 'close', 'archive', 'cancel', 'drop'],
                'secondary': ['get rid of', 'clean up'],
                'modifiers': []
            },
            IntentType.ANALYZE: {
                'primary': ['analyze', 'review', 'check', 'inspect', 'examine', 'evaluate', 'assess', 'audit'],
                'secondary': ['look at', 'take a look'],
                'modifiers': ['code', 'security', 'performance', 'quality']
            },
            IntentType.COORDINATE: {
                'primary': ['notify', 'tell', 'inform', 'alert', 'message', 'ping', 'send', 'post'],
                'secondary': ['let know', 'reach out'],
                'modifiers': ['team', 'channel', 'person', 'slack', 'email']
            },
            IntentType.SEARCH: {
                'primary': ['search', 'find', 'lookup', 'query', 'locate'],
                'secondary': ['look for', 'hunt for'],
                'modifiers': ['for', 'about', 'related to']
            },
            IntentType.WORKFLOW: {
                'primary': ['when', 'if', 'then', 'automate', 'trigger', 'schedule'],
                'secondary': ['set up automation', 'create workflow'],
                'modifiers': []
            }
        }

        self.implicit_patterns = {
            'urgency': {
                'critical': ['urgent', 'critical', 'asap', 'immediately', 'emergency', 'blocker'],
                'high': ['important', 'soon', 'quickly', 'high priority'],
                'normal': []
            },
            'scope': {
                'single': ['the', 'this', 'that', 'one'],
                'multiple': ['all', 'every', 'each', 'multiple'],
                'batch': ['batch', 'bulk', 'mass']
            },
            'visibility': {
                'public': ['public', 'everyone', 'all', 'team'],
                'private': ['private', 'just me', 'personal']
            }
        }

    def classify(self, message: str) -> List[Intent]:
        message_lower = message.lower()
        detected_intents = []

        for intent_type, keywords in self.intent_keywords.items():
            confidence = self._calculate_intent_confidence(message_lower, keywords)

            if confidence > 0.3:
                intent = Intent(
                    type=intent_type,
                    confidence=confidence,
                    raw_indicators=self._extract_indicators(message_lower, keywords)
                )
                detected_intents.append(intent)

        detected_intents.sort(key=lambda x: x.confidence, reverse=True)

        implicit_reqs = self._detect_implicit_requirements(message_lower)
        for intent in detected_intents:
            intent.implicit_requirements = implicit_reqs

        if not detected_intents:
            detected_intents.append(Intent(
                type=IntentType.UNKNOWN,
                confidence=0.5,
                implicit_requirements=implicit_reqs
            ))

        if self.verbose:
            print(f"[INTENT] Detected {len(detected_intents)} intents:")
            for intent in detected_intents:
                print(f"  - {intent}")

        return detected_intents

    def get_primary_intent(self, intents: List[Intent]) -> Intent:
        return intents[0] if intents else Intent(type=IntentType.UNKNOWN, confidence=0.0)

    def has_intent_type(self, intents: List[Intent], intent_type: IntentType) -> bool:
        return any(i.type == intent_type for i in intents)

    def _calculate_intent_confidence(self, message: str, keywords: Dict[str, List[str]]) -> float:
        score = 0.0

        for keyword in keywords.get('primary', []):
            if keyword in message:
                position = message.find(keyword)
                position_factor = 1.0 - (position / len(message)) * 0.2
                score = max(score, 0.9 * position_factor)
                break

        for keyword in keywords.get('secondary', []):
            if keyword in message:
                score = max(score, 0.7)
                break

        modifiers_found = sum(1 for mod in keywords.get('modifiers', []) if mod in message)
        if modifiers_found > 0:
            score = min(score + (modifiers_found * 0.1), 1.0)

        return score

    def _extract_indicators(self, message: str, keywords: Dict[str, List[str]]) -> List[str]:
        indicators = []
        for keyword in keywords.get('primary', []) + keywords.get('secondary', []):
            if keyword in message:
                indicators.append(keyword)
        return indicators

    def _detect_implicit_requirements(self, message: str) -> List[str]:
        requirements = []

        for level, keywords in self.implicit_patterns['urgency'].items():
            for keyword in keywords:
                if keyword in message:
                    if level == 'critical':
                        requirements.append('priority:critical')
                        requirements.append('urgent:true')
                    elif level == 'high':
                        requirements.append('priority:high')
                    break

        for scope, keywords in self.implicit_patterns['scope'].items():
            for keyword in keywords:
                if keyword in message:
                    requirements.append(f'scope:{scope}')
                    if scope in ['multiple', 'batch']:
                        requirements.append('batch_operation:true')
                    break

        for visibility, keywords in self.implicit_patterns['visibility'].items():
            for keyword in keywords:
                if keyword in message:
                    requirements.append(f'visibility:{visibility}')
                    break

        security_keywords = ['security', 'secure', 'auth', 'authentication', 'authorization', 'permission']
        if any(kw in message for kw in security_keywords):
            requirements.append('security_sensitive:true')

        performance_keywords = ['performance', 'slow', 'fast', 'optimize', 'speed']
        if any(kw in message for kw in performance_keywords):
            requirements.append('performance_related:true')

        return requirements

    def is_multi_intent(self, intents: List[Intent]) -> bool:
        high_confidence_intents = [i for i in intents if i.confidence > 0.6]
        return len(high_confidence_intents) > 1

    def suggest_clarifications(self, intents: List[Intent]) -> List[str]:
        clarifications = []
        primary = self.get_primary_intent(intents)

        if primary.type == IntentType.CREATE:
            clarifications.extend([
                "What should be created? (issue, PR, page, etc.)",
                "Which project/repository?",
                "Any specific details or description?"
            ])
        elif primary.type == IntentType.READ:
            clarifications.extend([
                "What information are you looking for?",
                "Which project/repository?"
            ])
        elif primary.type == IntentType.UPDATE:
            clarifications.extend([
                "Which resource to update?",
                "What changes should be made?"
            ])
        elif primary.type == IntentType.COORDINATE:
            clarifications.extend([
                "Who should be notified?",
                "What message to send?"
            ])

        return clarifications

    def extract_action_target(self, message: str) -> Optional[str]:
        message_lower = message.lower()
        targets = [
            'issue', 'ticket', 'pr', 'pull request', 'page', 'task',
            'project', 'repository', 'repo', 'file', 'code',
            'message', 'channel', 'comment', 'branch'
        ]

        for target in targets:
            if target in message_lower:
                return target

        return None

    def detect_conditional_logic(self, message: str) -> bool:
        message_lower = message.lower()
        conditional_patterns = [
            r'\bwhen\b.*\b(then|do|notify|create)',
            r'\bif\b.*\b(then|do|notify|create)',
            r'\bwhenever\b.*\b(then|do|notify|create)',
        ]
        return any(re.search(pattern, message_lower) for pattern in conditional_patterns)

    def classify_with_cache(self, message: str) -> List[Intent]:
        cache_key = CacheKeyBuilder.for_intent_classification(message)

        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self.cache_hits += 1
            if self.verbose:
                print(f"[INTENT] Cache hit for message")
            return cached_result

        intents = self.classify(message)
        self.cache.set(cache_key, intents, ttl_seconds=300)

        return intents

    def classify_with_llm(self, message: str, context: Optional[Dict] = None) -> List[Intent]:
        if not self.llm_client:
            if self.verbose:
                print("[INTENT] No LLM client available, falling back to keywords")
            return self.classify(message)

        self.llm_classifications += 1

        prompt = self._build_intent_classification_prompt(message, context)

        try:
            response = self._call_llm_for_intent(prompt)
            intents = self._parse_llm_intent_response(response)

            if self.verbose:
                print(f"[INTENT] LLM classified {len(intents)} intents")

            return intents

        except Exception as e:
            if self.verbose:
                print(f"[INTENT] LLM classification failed: {e}, falling back to keywords")
            return self.classify(message)

    def classify_hybrid(self, message: str, context: Optional[Dict] = None) -> List[Intent]:
        keyword_intents = self.classify(message)
        needs_disambiguation = self._needs_disambiguation(keyword_intents, message)

        if not needs_disambiguation or not self.use_llm:
            return keyword_intents

        if self.verbose:
            print("[INTENT] Low confidence, using LLM for disambiguation")

        llm_intents = self.classify_with_llm(message, context)
        merged_intents = self._merge_intent_results(keyword_intents, llm_intents)

        return merged_intents

    def _needs_disambiguation(self, intents: List[Intent], message: str) -> bool:
        if not intents:
            return True

        high_conf_intents = [i for i in intents if i.confidence > 0.8]
        if not high_conf_intents:
            return True

        similar_conf_intents = [
            i for i in intents
            if i.confidence > 0.6 and abs(i.confidence - intents[0].confidence) < 0.15
        ]
        if len(similar_conf_intents) > 2:
            return True

        ambiguous_words = ['maybe', 'might', 'could', 'should', 'possibly', 'perhaps']
        if any(word in message.lower() for word in ambiguous_words):
            return True

        if len(re.split(r'[.!?;]', message)) > 3:
            return True

        return False

    def _merge_intent_results(
        self,
        keyword_intents: List[Intent],
        llm_intents: List[Intent]
    ) -> List[Intent]:
        merged = list(llm_intents)
        llm_intent_types = {i.type for i in llm_intents}

        for keyword_intent in keyword_intents:
            if keyword_intent.confidence > 0.85 and keyword_intent.type not in llm_intent_types:
                merged.append(keyword_intent)

        merged.sort(key=lambda x: x.confidence, reverse=True)
        return merged

    def _build_intent_classification_prompt(
        self,
        message: str,
        context: Optional[Dict] = None
    ) -> str:
        return f"""Classify the user's intent(s) from this message.

Message: "{message}"

Available intent types:
- CREATE: Create something new (issue, PR, page, task, project)
- READ: Get or retrieve information (show, find, search, list)
- UPDATE: Modify existing resource (update, change, edit, fix)
- DELETE: Remove or close something (delete, remove, close, archive)
- ANALYZE: Review or analyze (review code, check quality, assess)
- COORDINATE: Notify or communicate (tell someone, send message, post)
- SEARCH: Search for information
- WORKFLOW: Automation or conditional logic (if/when/then)
- UNKNOWN: Cannot determine intent

For each detected intent, provide:
1. Intent type
2. Confidence score (0.0 to 1.0)
3. Key indicators from message

Format response as JSON array:
[{{"intent": "CREATE", "confidence": 0.95, "indicators": ["create", "new issue"]}}]

Response:"""

    def _call_llm_for_intent(self, prompt: str) -> str:
        raise NotImplementedError("LLM client integration not implemented")

    def _parse_llm_intent_response(self, response: str) -> List[Intent]:
        import json

        try:
            data = json.loads(response)
            intents = []
            for item in data:
                intent_type_str = item.get('intent', 'UNKNOWN')
                confidence = item.get('confidence', 0.5)
                indicators = item.get('indicators', [])

                try:
                    intent_type = IntentType(intent_type_str.lower())
                except ValueError:
                    intent_type = IntentType.UNKNOWN

                intent = Intent(
                    type=intent_type,
                    confidence=confidence,
                    raw_indicators=indicators
                )
                intents.append(intent)

            return intents

        except json.JSONDecodeError:
            if self.verbose:
                print("[INTENT] Failed to parse LLM response as JSON")
            return []

    def calibrate_confidence(self, intents: List[Intent], message: str) -> List[Intent]:
        for intent in intents:
            original_conf = intent.confidence

            if len(message.split()) < 3:
                intent.confidence *= 0.9

            vague_words = ['thing', 'stuff', 'something', 'anything']
            if any(word in message.lower() for word in vague_words):
                intent.confidence *= 0.85

            specific_indicators = ['#', '@', 'http', '://', '-']
            specificity = sum(1 for ind in specific_indicators if ind in message)
            if specificity >= 2:
                intent.confidence = min(intent.confidence * 1.1, 1.0)

            intent.confidence = max(0.0, min(1.0, intent.confidence))

            if self.verbose and abs(original_conf - intent.confidence) > 0.05:
                print(f"[INTENT] Calibrated {intent.type.value}: {original_conf:.2f} → {intent.confidence:.2f}")

        return intents

    def get_intent_hierarchy(self, intent_type: IntentType) -> List[IntentType]:
        hierarchies = {
            IntentType.CREATE: [IntentType.UPDATE, IntentType.COORDINATE],
            IntentType.ANALYZE: [IntentType.CREATE, IntentType.COORDINATE],
            IntentType.SEARCH: [IntentType.READ],
            IntentType.READ: [IntentType.ANALYZE],
        }
        return hierarchies.get(intent_type, [])

    def get_metrics(self) -> Dict[str, Any]:
        return {
            'keyword_classifications': self.keyword_classifications,
            'llm_classifications': self.llm_classifications,
            'cache_hits': self.cache_hits,
            'total_classifications': self.keyword_classifications + self.llm_classifications,
        }

    def reset_metrics(self):
        self.keyword_classifications = 0
        self.llm_classifications = 0
        self.cache_hits = 0


class EntityExtractor:

    def __init__(self, llm_client: Optional[any] = None, verbose: bool = False):
        self.llm_client = llm_client
        self.verbose = verbose
        self.cache = get_global_cache()

        self.extractions = 0
        self.entities_extracted = 0
        self.relationships_found = 0

        self.patterns = {
            EntityType.ISSUE: [
                r'\b([A-Z]{2,10}-\d+)\b',
                r'\bissue\s+(\d+)\b',
                r'\bticket\s+(\d+)\b',
            ],
            EntityType.PR: [
                r'\bPR\s*#?(\d+)\b',
                r'\bpull request\s*#?(\d+)\b',
                r'#(\d+)',
            ],
            EntityType.PROJECT: [
                r'\b([A-Z]{2,10})\s+project\b',
                r'\bproject\s+([A-Z]{2,10})\b',
            ],
            EntityType.PERSON: [
                r'@([\w.-]+)',
                r'\bto\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
            ],
            EntityType.TEAM: [
                r'@([\w-]+team)\b',
                r'\b([\w-]+\s+team)\b',
                r'@(engineering|security|devops|qa|design)\b',
            ],
            EntityType.CHANNEL: [
                r'#([\w-]+)',
                r'\bchannel\s+([\w-]+)\b',
            ],
            EntityType.REPOSITORY: [
                r'\b([\w-]+)/([\w-]+)\b',
                r'\brepo\s+([\w-]+)\b',
                r'\brepository\s+([\w-]+)\b',
            ],
            EntityType.FILE: [
                r'([\w/-]+\.[\w]+)',
                r'\bfile\s+([\w/.]+)\b',
            ],
            EntityType.RESOURCE: [
                r'(https?://[^\s]+)',
            ],
        }

        self.priority_keywords = {
            'critical': ['critical', 'blocker', 'urgent', 'emergency'],
            'high': ['high', 'important', 'priority', 'soon'],
            'medium': ['medium', 'normal', 'standard'],
            'low': ['low', 'minor', 'trivial', 'nice to have', 'nice-to-have']
        }

        self.status_keywords = {
            'open': ['open', 'new', 'todo', 'backlog'],
            'in_progress': ['in progress', 'in-progress', 'working', 'started', 'active'],
            'review': ['review', 'reviewing', 'pending review'],
            'done': ['done', 'completed', 'closed', 'resolved', 'fixed'],
            'blocked': ['blocked', 'waiting', 'on hold']
        }

        self.date_patterns = {
            'tomorrow': lambda: datetime.now() + timedelta(days=1),
            'today': lambda: datetime.now(),
            'yesterday': lambda: datetime.now() - timedelta(days=1),
            'next week': lambda: datetime.now() + timedelta(weeks=1),
            'next month': lambda: datetime.now() + timedelta(days=30),
        }

        self.weekdays = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2,
            'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6
        }

    def extract(self, message: str, context: Optional[Dict] = None) -> List[Entity]:
        entities = []

        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, message, re.IGNORECASE)
                for match in matches:
                    value = match.group(1) if match.groups() else match.group(0)

                    if self._is_valid_entity(value, entity_type):
                        entity = Entity(
                            type=entity_type,
                            value=value,
                            confidence=0.9,
                            context=match.group(0)
                        )
                        entities.append(entity)

        entities.extend(self._extract_priorities(message))
        entities.extend(self._extract_statuses(message))
        entities.extend(self._extract_dates(message))
        entities.extend(self._extract_labels(message))

        entities = self._deduplicate_entities(entities)

        for entity in entities:
            entity.normalized_value = self._normalize_value(entity)

        if self.verbose:
            print(f"[ENTITY] Extracted {len(entities)} entities:")
            for entity in entities:
                print(f"  - {entity}")

        return entities

    def extract_by_type(self, message: str, entity_type: EntityType) -> List[Entity]:
        all_entities = self.extract(message)
        return [e for e in all_entities if e.type == entity_type]

    def find_entity_value(self, entities: List[Entity], entity_type: EntityType) -> Optional[str]:
        for entity in entities:
            if entity.type == entity_type:
                return entity.normalized_value or entity.value
        return None

    def _is_valid_entity(self, value: str, entity_type: EntityType) -> bool:
        if len(value) < 2:
            return False

        false_positives = {'the', 'it', 'and', 'or', 'in', 'on', 'at'}
        if value.lower() in false_positives:
            return False

        if entity_type == EntityType.PROJECT:
            return value.isupper() and 2 <= len(value) <= 10
        elif entity_type == EntityType.ISSUE:
            return any(c.isdigit() for c in value)

        return True

    def _extract_priorities(self, message: str) -> List[Entity]:
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
                    break

        return entities

    def _extract_statuses(self, message: str) -> List[Entity]:
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
        message_lower = message.lower()
        entities = []

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

        for weekday, weekday_num in self.weekdays.items():
            pattern = r'\b(?:next|by|on)\s+' + weekday + r'\b'
            if re.search(pattern, message_lower):
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

        date_patterns = [
            r'\b(\d{4}-\d{2}-\d{2})\b',
            r'\b(\d{2}/\d{2}/\d{4})\b',
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
        pattern = r'#([a-z][\w-]{2,})'
        matches = re.finditer(pattern, message, re.IGNORECASE)

        entities = []
        for match in matches:
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
        seen = set()
        unique = []

        for entity in entities:
            key = (entity.type, entity.value.lower())
            if key not in seen:
                seen.add(key)
                unique.append(entity)

        return unique

    def _normalize_value(self, entity: Entity) -> str:
        if entity.type == EntityType.PRIORITY:
            return entity.value.lower()
        elif entity.type == EntityType.STATUS:
            return entity.value.replace(' ', '_').lower()
        elif entity.type == EntityType.PERSON:
            return entity.value.lstrip('@')
        elif entity.type == EntityType.CHANNEL:
            return entity.value.lstrip('#')
        elif entity.type == EntityType.TEAM:
            return entity.value.replace(' ', '-').lower().lstrip('@')
        else:
            return entity.value

    def group_by_type(self, entities: List[Entity]) -> Dict[EntityType, List[Entity]]:
        grouped = {}
        for entity in entities:
            if entity.type not in grouped:
                grouped[entity.type] = []
            grouped[entity.type].append(entity)
        return grouped

    def has_entity_type(self, entities: List[Entity], entity_type: EntityType) -> bool:
        return any(e.type == entity_type for e in entities)

    def get_entity_summary(self, entities: List[Entity]) -> str:
        if not entities:
            return "No entities extracted"

        grouped = self.group_by_type(entities)
        summary_parts = []

        for entity_type, ents in grouped.items():
            values = [e.value for e in ents]
            summary_parts.append(f"{entity_type.value}: {', '.join(values)}")

        return "; ".join(summary_parts)

    def extract_with_relationships(
        self,
        message: str,
        context: Optional[Dict] = None
    ) -> Tuple[List[Entity], EntityGraph]:
        entities = self.extract(message, context)

        graph = EntityGraph()

        for entity in entities:
            entity_id = create_entity_id(entity)
            graph.add_entity(entity_id, entity)

        relationships = self._extract_relationships(message, entities)

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
        relationships = []
        message_lower = message.lower()

        patterns = [
            (r'assign\s+(\S+)\s+to\s+(\S+)', RelationType.ASSIGNED_TO),
            (r'(\S+)\s+assigned\s+to\s+(\S+)', RelationType.ASSIGNED_TO),
            (r'(\S+)\s+depends\s+on\s+(\S+)', RelationType.DEPENDS_ON),
            (r'(\S+)\s+blocked\s+by\s+(\S+)', RelationType.DEPENDS_ON),
            (r'link\s+(\S+)\s+to\s+(\S+)', RelationType.LINKED_TO),
            (r'(\S+)\s+related\s+to\s+(\S+)', RelationType.RELATED_TO),
            (r'(\S+)\s+linked\s+to\s+(\S+)', RelationType.LINKED_TO),
            (r'(\S+)\s+mentions?\s+(\S+)', RelationType.MENTIONS),
        ]

        for pattern, rel_type in patterns:
            matches = re.finditer(pattern, message_lower, re.IGNORECASE)
            for match in matches:
                from_val = match.group(1)
                to_val = match.group(2)

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
        regex_entities = self.extract(message, context)

        if not self.llm_client:
            return regex_entities

        try:
            llm_entities = self._extract_with_llm(message, context)
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
        raise NotImplementedError("LLM entity extraction not implemented")

    def _merge_entity_results(
        self,
        regex_entities: List[Entity],
        llm_entities: List[Entity]
    ) -> List[Entity]:
        merged = list(llm_entities)
        llm_values = {e.value.lower() for e in llm_entities}

        for regex_entity in regex_entities:
            if regex_entity.confidence > 0.85 and regex_entity.value.lower() not in llm_values:
                merged.append(regex_entity)

        merged = self._deduplicate_entities(merged)
        return merged

    def calibrate_entity_confidence(
        self,
        entities: List[Entity],
        message: str,
        context: Optional[Dict] = None
    ) -> List[Entity]:
        for entity in entities:
            original_conf = entity.confidence

            if context:
                focused = context.get('focused_entities', [])
                if any(f['value'].lower() == entity.value.lower() for f in focused):
                    entity.confidence = min(entity.confidence * 1.15, 1.0)

            if len(entity.value) < 2:
                entity.confidence *= 0.7

            same_type_count = sum(1 for e in entities if e.type == entity.type)
            if same_type_count >= 2:
                entity.confidence = min(entity.confidence * 1.05, 1.0)

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
        if not context:
            return entities

        message_lower = message.lower()

        coreferences = {
            'it': EntityType.UNKNOWN,
            'that': EntityType.UNKNOWN,
            'this': EntityType.UNKNOWN,
            'the issue': EntityType.ISSUE,
            'the ticket': EntityType.ISSUE,
            'the pr': EntityType.PR,
            'the pull request': EntityType.PR,
        }

        focused = context.get('focused_entities', [])
        if not focused:
            return entities

        for coref, expected_type in coreferences.items():
            if coref in message_lower:
                for focused_entity in reversed(focused):
                    entity_type = focused_entity.get('type')

                    if expected_type == EntityType.UNKNOWN or entity_type == expected_type.value:
                        resolved = Entity(
                            type=EntityType(entity_type),
                            value=focused_entity['value'],
                            confidence=0.80,
                            context=f"Resolved from '{coref}'"
                        )

                        if not any(e.value == resolved.value for e in entities):
                            entities.append(resolved)

                            if self.verbose:
                                print(f"[ENTITY] Resolved '{coref}' → {resolved}")

                        break

        return entities

    def get_entity_graph_summary(self, graph: EntityGraph) -> str:
        lines = []
        lines.append(f"Entity Graph:")
        lines.append(f"  Entities: {len(graph.entities)}")
        lines.append(f"  Relationships: {len(graph.relationships)}")

        if graph.relationships:
            lines.append(f"\nRelationships:")
            for rel in graph.relationships[:5]:
                lines.append(f"  • {rel.from_entity_id} --{rel.relation_type.value}-> {rel.to_entity_id}")

        return "\n".join(lines)

    def get_metrics(self) -> Dict:
        return {
            'extractions': self.extractions,
            'entities_extracted': self.entities_extracted,
            'relationships_found': self.relationships_found,
            'avg_entities_per_extraction': (
                self.entities_extracted / max(self.extractions, 1)
            ),
        }

    def reset_metrics(self):
        self.extractions = 0
        self.entities_extracted = 0
        self.relationships_found = 0


class TaskDecomposer:

    def __init__(self, agent_capabilities: Optional[Dict[str, List[str]]] = None, verbose: bool = False):
        self.agent_capabilities = agent_capabilities or {}
        self.verbose = verbose
        self.task_counter = 0

        self.intent_actions = {
            IntentType.CREATE: ['create', 'build', 'generate'],
            IntentType.READ: ['get', 'fetch', 'list', 'search'],
            IntentType.UPDATE: ['update', 'modify', 'change', 'set'],
            IntentType.DELETE: ['delete', 'remove', 'close'],
            IntentType.ANALYZE: ['review', 'analyze', 'check'],
            IntentType.COORDINATE: ['notify', 'send', 'post'],
            IntentType.SEARCH: ['search', 'find', 'query'],
        }

        self.entity_agent_hints = {
            EntityType.ISSUE: ['jira', 'github'],
            EntityType.PR: ['github'],
            EntityType.PROJECT: ['jira'],
            EntityType.CHANNEL: ['slack'],
            EntityType.FILE: ['github', 'browser', 'scraper'],
            EntityType.CODE: ['code_reviewer', 'github'],
        }

    def decompose(
        self,
        message: str,
        intents: List[Intent],
        entities: List[Entity],
        context: Optional[Dict] = None
    ) -> ExecutionPlan:
        if not intents:
            return ExecutionPlan()

        tasks = []
        for intent in intents:
            task = self._intent_to_task(intent, entities, message, context)
            if task:
                tasks.append(task)

        dependency_graph = self._build_dependency_graph(tasks, intents)
        self._detect_conditional_tasks(tasks, message)
        self._estimate_costs(tasks)

        plan = ExecutionPlan(
            tasks=tasks,
            dependency_graph=dependency_graph,
            estimated_duration=sum(t.estimated_duration for t in tasks),
            estimated_cost=sum(t.estimated_cost for t in tasks)
        )

        plan.risks = self._identify_risks(plan)

        if self.verbose:
            print(f"[DECOMPOSE] Created plan with {len(tasks)} tasks")
            for task in tasks:
                print(f"  - {task}")
            if dependency_graph.edges:
                print(f"[DECOMPOSE] Dependencies: {dependency_graph.edges}")

        return plan

    def _intent_to_task(
        self,
        intent: Intent,
        entities: List[Entity],
        message: str,
        context: Optional[Dict]
    ) -> Optional[Task]:
        self.task_counter += 1
        task_id = f"task_{self.task_counter}"

        action = self._get_action_for_intent(intent)
        suggested_agent = self._suggest_agent_for_intent(intent, entities)
        task_entities = self._filter_entities_for_intent(intent, entities)
        inputs = self._build_task_inputs(intent, task_entities, message)
        outputs = self._determine_task_outputs(intent, task_entities)

        task = Task(
            id=task_id,
            action=action,
            agent=suggested_agent,
            inputs=inputs,
            outputs=outputs,
            metadata={
                'intent': str(intent.type),
                'confidence': intent.confidence,
                'entities': [str(e) for e in task_entities]
            }
        )

        return task

    def _get_action_for_intent(self, intent: Intent) -> str:
        actions = self.intent_actions.get(intent.type, ['execute'])
        return actions[0]

    def _suggest_agent_for_intent(self, intent: Intent, entities: List[Entity]) -> Optional[str]:
        agent_scores = {}

        for entity in entities:
            suggested_agents = self.entity_agent_hints.get(entity.type, [])
            for agent in suggested_agents:
                agent_scores[agent] = agent_scores.get(agent, 0) + entity.confidence

        if intent.type == IntentType.ANALYZE:
            if any(e.type == EntityType.CODE for e in entities):
                agent_scores['code_reviewer'] = agent_scores.get('code_reviewer', 0) + 1.0

        elif intent.type == IntentType.COORDINATE:
            if any(e.type == EntityType.CHANNEL for e in entities):
                agent_scores['slack'] = agent_scores.get('slack', 0) + 1.0

        if agent_scores:
            return max(agent_scores.items(), key=lambda x: x[1])[0]

        return None

    def _filter_entities_for_intent(self, intent: Intent, entities: List[Entity]) -> List[Entity]:
        return entities

    def _build_task_inputs(
        self,
        intent: Intent,
        entities: List[Entity],
        message: str
    ) -> Dict[str, any]:
        inputs = {
            'message': message,
            'intent_type': intent.type.value,
        }

        for entity in entities:
            key = entity.type.value
            value = entity.normalized_value or entity.value

            if key in inputs:
                if not isinstance(inputs[key], list):
                    inputs[key] = [inputs[key]]
                inputs[key].append(value)
            else:
                inputs[key] = value

        for req in intent.implicit_requirements:
            if ':' in req:
                key, value = req.split(':', 1)
                inputs[key] = value

        return inputs

    def _determine_task_outputs(self, intent: Intent, entities: List[Entity]) -> List[str]:
        outputs = []

        if intent.type == IntentType.CREATE:
            for entity in entities:
                if entity.type in [EntityType.ISSUE, EntityType.PR, EntityType.PROJECT]:
                    outputs.append(f"{entity.type.value}_id")
                    outputs.append(f"{entity.type.value}_url")

            if not outputs:
                outputs = ['resource_id', 'resource_url']

        elif intent.type == IntentType.READ:
            outputs = ['data', 'results']

        elif intent.type == IntentType.ANALYZE:
            outputs = ['analysis', 'findings', 'issues']

        return outputs

    def _build_dependency_graph(self, tasks: List[Task], intents: List[Intent]) -> DependencyGraph:
        graph = DependencyGraph()

        for task in tasks:
            graph.add_task(task)

        for i, task in enumerate(tasks):
            for j in range(i + 1, len(tasks)):
                dependent_task = tasks[j]

                if self._needs_output_from(dependent_task, task):
                    graph.add_dependency(task.id, dependent_task.id)
                    dependent_task.dependencies.append(task.id)

        return graph

    def _needs_output_from(self, dependent_task: Task, provider_task: Task) -> bool:
        for output in provider_task.outputs:
            for input_key in dependent_task.inputs.keys():
                if output.replace('_', '') in input_key.replace('_', ''):
                    return True

        action_dependencies = {
            'create': ['update', 'notify', 'set'],
            'analyze': ['create', 'notify'],
            'get': ['create', 'update', 'notify'],
            'search': ['create', 'update'],
        }

        provider_action = provider_task.action
        dependent_action = dependent_task.action

        for key_action, dependent_actions in action_dependencies.items():
            if key_action in provider_action and dependent_action in dependent_actions:
                return True

        return False

    def _detect_conditional_tasks(self, tasks: List[Task], message: str):
        message_lower = message.lower()

        if 'if' in message_lower or 'when' in message_lower:
            for task in tasks:
                if task.action in ['create', 'notify', 'update']:
                    task.conditions = "conditional:check_condition"
                    task.metadata['conditional'] = True

    def _estimate_costs(self, tasks: List[Task]):
        for task in tasks:
            if task.action == 'review' or task.action == 'analyze':
                task.estimated_duration = 5.0
                task.estimated_cost = 500.0

            elif task.action == 'create':
                task.estimated_duration = 2.0
                task.estimated_cost = 100.0

            elif task.action in ['get', 'fetch', 'list', 'search']:
                task.estimated_duration = 1.5
                task.estimated_cost = 50.0

            else:
                task.estimated_duration = 2.0
                task.estimated_cost = 100.0

    def _identify_risks(self, plan: ExecutionPlan) -> List[str]:
        risks = []

        if plan.dependency_graph and plan.dependency_graph.has_cycle():
            risks.append("⚠️ CRITICAL: Circular dependencies detected")

        if plan.estimated_cost > 1000:
            risks.append(f"⚠️ HIGH: Estimated cost is high ({plan.estimated_cost:.0f} tokens)")

        if plan.estimated_duration > 30:
            risks.append(f"⚠️ MEDIUM: Estimated duration is long ({plan.estimated_duration:.1f}s)")

        if len(plan.tasks) > 10:
            risks.append(f"⚠️ MEDIUM: Many tasks ({len(plan.tasks)})")

        conditional_tasks = [t for t in plan.tasks if t.conditions]
        if conditional_tasks:
            risks.append(f"ℹ️ INFO: {len(conditional_tasks)} conditional tasks")

        return risks

    def get_parallel_tasks(self, plan: ExecutionPlan) -> List[List[Task]]:
        if not plan.dependency_graph:
            return [[task] for task in plan.tasks]

        levels = []
        processed = set()
        remaining = set(plan.tasks)

        while remaining:
            current_level = []
            for task in remaining:
                deps_processed = all(
                    dep_id in processed or dep_id not in [t.id for t in plan.tasks]
                    for dep_id in task.dependencies
                )
                if deps_processed:
                    current_level.append(task)

            if not current_level:
                break

            levels.append(current_level)
            for task in current_level:
                processed.add(task.id)
                remaining.remove(task)

        return levels

    def optimize_plan(self, plan: ExecutionPlan) -> ExecutionPlan:
        return plan

    def explain_plan(self, plan: ExecutionPlan) -> str:
        lines = []
        lines.append(f"Execution Plan ({len(plan.tasks)} tasks):")
        lines.append("")

        ordered_tasks = plan.get_execution_order()

        for i, task in enumerate(ordered_tasks, 1):
            agent = task.agent or "?"
            action = task.action
            dependencies = f" (after {', '.join(task.dependencies)})" if task.dependencies else ""
            conditional = " [conditional]" if task.conditions else ""

            lines.append(f"{i}. [{agent}] {action}{dependencies}{conditional}")

        lines.append("")
        lines.append(f"Estimated duration: {plan.estimated_duration:.1f}s")
        lines.append(f"Estimated cost: {plan.estimated_cost:.0f} tokens")

        if plan.risks:
            lines.append("")
            lines.append("Risks:")
            for risk in plan.risks:
                lines.append(f"  {risk}")

        return "\n".join(lines)


class ConfidenceScorer:

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        self.required_entities = {
            'create': ['issue', 'pr', 'project', 'repository'],
            'update': ['issue', 'pr', 'resource'],
            'coordinate': ['channel', 'person', 'team'],
            'analyze': ['code', 'file', 'repository'],
        }

    def score_overall(
        self,
        message: str,
        intents: List[Intent],
        entities: List[Entity],
        plan: Optional[ExecutionPlan] = None
    ) -> Confidence:
        factors = {}

        factors['intent_clarity'] = self._score_intent_clarity(message, intents)
        factors['entity_completeness'] = self._score_entity_completeness(intents, entities)
        factors['message_clarity'] = self._score_message_clarity(message)

        if plan:
            factors['plan_quality'] = self._score_plan_quality(plan)
        else:
            factors['plan_quality'] = 0.5

        weights = {
            'intent_clarity': 0.3,
            'entity_completeness': 0.3,
            'message_clarity': 0.2,
            'plan_quality': 0.2
        }

        total_score = sum(factors[k] * weights[k] for k in factors)

        uncertainties = self._identify_uncertainties(message, intents, entities, factors)
        assumptions = self._identify_assumptions(message, intents, entities)

        confidence = Confidence.from_score(total_score, factors)
        confidence.uncertainties = uncertainties
        confidence.assumptions = assumptions

        if self.verbose:
            print(f"[CONFIDENCE] Overall score: {confidence}")
            print(f"  Factors: {factors}")
            if uncertainties:
                print(f"  Uncertainties: {len(uncertainties)}")
            if assumptions:
                print(f"  Assumptions: {len(assumptions)}")

        return confidence

    def _score_intent_clarity(self, message: str, intents: List[Intent]) -> float:
        if not intents:
            return 0.2

        high_conf_intents = [i for i in intents if i.confidence > 0.8]
        if not high_conf_intents:
            return 0.4

        if len(high_conf_intents) == 1:
            return min(high_conf_intents[0].confidence, 0.95)

        if len(high_conf_intents) <= 3:
            avg_confidence = sum(i.confidence for i in high_conf_intents) / len(high_conf_intents)
            return avg_confidence * 0.9

        return 0.6

    def _score_entity_completeness(self, intents: List[Intent], entities: List[Entity]) -> float:
        if not intents:
            return 0.0

        primary_intent = intents[0]
        intent_type = primary_intent.type.value

        required = self.required_entities.get(intent_type, [])

        if not required:
            return 0.8

        entity_types = [e.type.value for e in entities]
        has_required = any(req in entity_types for req in required)

        if has_required:
            high_conf_entities = [e for e in entities if e.confidence > 0.8]
            if len(high_conf_entities) >= 2:
                return 0.95
            elif len(high_conf_entities) == 1:
                return 0.80
            else:
                return 0.60
        else:
            return 0.3

    def _score_message_clarity(self, message: str) -> float:
        message_lower = message.lower()
        words = message_lower.split()
        word_count = len(words)

        score = 0.5

        if 5 <= word_count <= 30:
            score += 0.2
        elif word_count < 5:
            score -= 0.2
        elif word_count > 50:
            score -= 0.1

        question_words = ['what', 'how', 'why', 'when', 'where', 'which']
        question_count = sum(1 for qw in question_words if qw in message_lower)
        if question_count > 2:
            score -= 0.2

        specific_indicators = ['#', '@', '-', '/', 'http']
        specificity = sum(1 for ind in specific_indicators if ind in message)
        score += min(specificity * 0.05, 0.2)

        return max(0.0, min(1.0, score))

    def _score_plan_quality(self, plan: ExecutionPlan) -> float:
        score = 0.8

        if plan.risks:
            critical_risks = [r for r in plan.risks if 'CRITICAL' in r]
            if critical_risks:
                score -= 0.5
            else:
                score -= 0.1 * len(plan.risks)

        tasks_without_agents = [t for t in plan.tasks if not t.agent]
        if tasks_without_agents:
            score -= 0.1 * len(tasks_without_agents) / len(plan.tasks)

        if len(plan.tasks) == 0:
            score = 0.0
        elif len(plan.tasks) > 15:
            score -= 0.1

        return max(0.0, min(1.0, score))

    def _identify_uncertainties(
        self,
        message: str,
        intents: List[Intent],
        entities: List[Entity],
        factors: Dict[str, float]
    ) -> List[str]:
        uncertainties = []

        if factors.get('intent_clarity', 0) < 0.6:
            uncertainties.append("Unclear what action to take")

        if factors.get('entity_completeness', 0) < 0.6:
            if intents:
                primary_intent = intents[0].type.value
                uncertainties.append(f"Missing information for {primary_intent} action")

        ambiguous_words = ['it', 'that', 'this', 'them', 'those']
        message_lower = message.lower()
        has_ambiguous = any(word in message_lower.split() for word in ambiguous_words)

        if has_ambiguous and len(entities) == 0:
            uncertainties.append("Ambiguous references without context")

        if intents:
            high_conf = [i for i in intents if i.confidence > 0.7]
            if len(high_conf) > 3:
                uncertainties.append(f"Multiple actions requested ({len(high_conf)})")

        if intents and intents[0].type.value in ['create', 'update']:
            has_project = any(e.type.value in ['project', 'repository'] for e in entities)
            if not has_project:
                uncertainties.append("Project/repository not specified")

        return uncertainties

    def _identify_assumptions(
        self,
        message: str,
        intents: List[Intent],
        entities: List[Entity]
    ) -> List[str]:
        assumptions = []

        if intents and intents[0].type.value in ['create', 'update']:
            has_explicit_project = any(e.type.value in ['project', 'repository'] for e in entities)
            if not has_explicit_project:
                assumptions.append("Using current/default project")

        has_priority = any(e.type.value == 'priority' for e in entities)
        if not has_priority and intents and intents[0].type.value == 'create':
            assumptions.append("Using default priority (medium)")

        has_assignee = any(e.type.value == 'person' for e in entities)
        if not has_assignee and intents and intents[0].type.value == 'create':
            assumptions.append("Leaving unassigned")

        return assumptions

    def suggest_clarifications(self, confidence: Confidence, intents: List[Intent]) -> List[str]:
        questions = []

        if "Unclear what action" in str(confidence.uncertainties):
            questions.append("What would you like me to do?")

        if "Missing information" in str(confidence.uncertainties):
            if intents:
                primary = intents[0].type.value
                if primary == 'create':
                    questions.append("What should I create? (issue, PR, page, etc.)")
                elif primary == 'update':
                    questions.append("What should I update?")
                elif primary == 'coordinate':
                    questions.append("Who should I notify?")

        if "Project/repository not specified" in str(confidence.uncertainties):
            questions.append("Which project or repository?")

        if "Ambiguous references" in str(confidence.uncertainties):
            questions.append("Can you clarify what 'it' or 'that' refers to?")

        return questions

    def should_proceed_automatically(self, confidence: Confidence) -> bool:
        return confidence.should_proceed()

    def should_review_with_user(self, confidence: Confidence) -> bool:
        return confidence.should_review()

    def should_ask_clarifying_questions(self, confidence: Confidence) -> bool:
        return confidence.should_clarify()

    def get_action_recommendation(self, confidence: Confidence) -> Tuple[str, str]:
        if self.should_proceed_automatically(confidence):
            return ('proceed', f"High confidence ({confidence.score:.2f}) - proceeding automatically")

        elif self.should_review_with_user(confidence):
            return ('review', f"Medium confidence ({confidence.score:.2f}) - reviewing plan with user")

        else:
            return ('clarify', f"Low confidence ({confidence.score:.2f}) - asking clarifying questions")

    def score_bayesian(
        self,
        message: str,
        intents: List[Intent],
        entities: List[Entity],
        plan: Optional[ExecutionPlan] = None,
        prior_confidence: float = 0.5
    ) -> Confidence:
        posterior = prior_confidence

        intent_likelihood = self._likelihood_from_intent_clarity(message, intents)
        posterior = self._bayesian_update(posterior, intent_likelihood)

        entity_likelihood = self._likelihood_from_entity_completeness(intents, entities)
        posterior = self._bayesian_update(posterior, entity_likelihood)

        message_likelihood = self._likelihood_from_message_clarity(message)
        posterior = self._bayesian_update(posterior, message_likelihood)

        if plan:
            plan_likelihood = self._likelihood_from_plan_quality(plan)
            posterior = self._bayesian_update(posterior, plan_likelihood)

        factors = {
            'intent_likelihood': intent_likelihood,
            'entity_likelihood': entity_likelihood,
            'message_likelihood': message_likelihood,
            'prior': prior_confidence,
            'posterior': posterior
        }

        uncertainties = self._identify_uncertainties(message, intents, entities, factors)
        assumptions = self._identify_assumptions(message, intents, entities)

        confidence = Confidence.from_score(posterior, factors)
        confidence.uncertainties = uncertainties
        confidence.assumptions = assumptions

        if self.verbose:
            print(f"[CONFIDENCE] Bayesian score: {posterior:.3f}")
            print(f"  Prior: {prior_confidence:.3f}")

        return confidence

    def _bayesian_update(self, prior: float, likelihood: float) -> float:
        prior = max(0.01, min(0.99, prior))
        likelihood = max(0.01, min(0.99, likelihood))

        likelihood_ratio = likelihood / (1 - likelihood)
        prior_odds = prior / (1 - prior)

        posterior_odds = likelihood_ratio * prior_odds
        posterior = posterior_odds / (1 + posterior_odds)

        return max(0.0, min(1.0, posterior))

    def _likelihood_from_intent_clarity(self, message: str, intents: List[Intent]) -> float:
        if not intents:
            return 0.2

        max_conf = max(i.confidence for i in intents)
        high_conf_count = sum(1 for i in intents if i.confidence > 0.8)

        if high_conf_count == 1:
            return max_conf
        elif high_conf_count == 0:
            return 0.4
        else:
            return max_conf * 0.85

    def _likelihood_from_entity_completeness(
        self,
        intents: List[Intent],
        entities: List[Entity]
    ) -> float:
        if not entities:
            return 0.3

        avg_conf = sum(e.confidence for e in entities) / len(entities)
        high_conf_entities = sum(1 for e in entities if e.confidence > 0.8)

        if high_conf_entities >= 2:
            return min(avg_conf * 1.1, 1.0)
        elif high_conf_entities == 1:
            return avg_conf
        else:
            return avg_conf * 0.8

    def _likelihood_from_message_clarity(self, message: str) -> float:
        words = message.split()
        word_count = len(words)

        if 5 <= word_count <= 30:
            length_score = 0.9
        elif word_count < 5:
            length_score = 0.5
        else:
            length_score = 0.7

        ambiguous = ['maybe', 'might', 'could', 'should', 'possibly', 'perhaps', 'probably']
        ambiguous_count = sum(1 for word in words if word.lower() in ambiguous)

        if ambiguous_count == 0:
            ambiguity_score = 0.9
        elif ambiguous_count == 1:
            ambiguity_score = 0.7
        else:
            ambiguity_score = 0.5

        return (length_score + ambiguity_score) / 2

    def _likelihood_from_plan_quality(self, plan: ExecutionPlan) -> float:
        if not plan.tasks:
            return 0.3

        critical_risks = sum(1 for r in plan.risks if 'CRITICAL' in r)
        if critical_risks > 0:
            return 0.3

        unassigned = sum(1 for t in plan.tasks if not t.agent)
        assignment_ratio = 1.0 - (unassigned / len(plan.tasks))

        if 1 <= len(plan.tasks) <= 10:
            task_count_score = 0.9
        elif len(plan.tasks) > 15:
            task_count_score = 0.6
        else:
            task_count_score = 0.8

        return (assignment_ratio + task_count_score) / 2

    def calibrate_with_history(
        self,
        confidence: Confidence,
        historical_accuracy: Optional[Dict[str, float]] = None
    ) -> Confidence:
        if not historical_accuracy:
            return confidence

        conf_range = self._get_confidence_range(confidence.score)
        historical_acc = historical_accuracy.get(conf_range, confidence.score)

        calibration_factor = 0.3
        calibrated_score = (
            confidence.score * (1 - calibration_factor) +
            historical_acc * calibration_factor
        )

        calibrated = Confidence.from_score(calibrated_score, confidence.factors)
        calibrated.uncertainties = confidence.uncertainties
        calibrated.assumptions = confidence.assumptions

        if self.verbose:
            print(f"[CONFIDENCE] Calibrated: {confidence.score:.3f} → {calibrated_score:.3f}")

        return calibrated

    def _get_confidence_range(self, score: float) -> str:
        if score >= 0.9:
            return "0.9-1.0"
        elif score >= 0.8:
            return "0.8-0.9"
        elif score >= 0.6:
            return "0.6-0.8"
        elif score >= 0.4:
            return "0.4-0.6"
        else:
            return "0.0-0.4"

    def compute_entropy(self, intents: List[Intent]) -> float:
        if not intents:
            return 0.0

        total_conf = sum(i.confidence for i in intents)
        if total_conf == 0:
            return 0.0

        probs = [i.confidence / total_conf for i in intents]

        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log2(p)

        if self.verbose:
            print(f"[CONFIDENCE] Entropy: {entropy:.3f}")

        return entropy

    def should_ask_for_clarification_bayesian(
        self,
        confidence: Confidence,
        entropy: float,
        cost_of_error: float = 0.5
    ) -> bool:
        p_correct = confidence.score
        p_wrong = 1 - p_correct

        benefit_correct = 1.0
        eu_proceed = p_correct * benefit_correct - p_wrong * cost_of_error

        p_get_answer = 0.8
        cost_of_asking = 0.1

        eu_clarify = p_get_answer * benefit_correct - cost_of_asking

        if entropy > 2.0:
            eu_clarify += 0.2

        should_clarify = eu_clarify > eu_proceed

        if self.verbose:
            print(f"[CONFIDENCE] Decision theory:")
            print(f"  EU(proceed): {eu_proceed:.3f}")
            print(f"  EU(clarify): {eu_clarify:.3f}")
            print(f"  Decision: {'CLARIFY' if should_clarify else 'PROCEED'}")

        return should_clarify

    def get_metrics(self) -> Dict:
        return {
            'verbose': self.verbose,
        }

    def reset_metrics(self):
        pass
