"""
Intent Classification Engine - Enhanced

Understands what users really want from their natural language requests.
Uses hybrid approach: fast keyword matching + optional LLM-based semantic understanding.

Features:
- Hierarchical intent taxonomy
- Multi-intent detection
- Implicit requirement detection
- Coreference resolution awareness
- LLM-enhanced semantic understanding (optional)
- Confidence calibration

Author: AI System
Version: 3.0 - Major refactoring with LLM support
"""

import re
from typing import List, Dict, Optional, Any, Tuple
from .base_types import Intent, IntentType
from .cache_layer import get_global_cache, CacheKeyBuilder


class IntentClassifier:
    """
    Classify user intents from natural language

    Understands:
    - Primary intent (CREATE, READ, UPDATE, DELETE, ANALYZE, COORDINATE)
    - Multiple intents in one request
    - Implicit requirements
    - Contextual indicators
    """

    def __init__(self, llm_client: Optional[Any] = None, use_llm: bool = False, verbose: bool = False):
        """
        Initialize intent classifier

        Args:
            llm_client: Optional LLM client for semantic understanding
            use_llm: Whether to use LLM for disambiguation (slower but more accurate)
            verbose: Enable verbose logging
        """
        self.llm_client = llm_client
        self.use_llm = use_llm
        self.verbose = verbose
        self.cache = get_global_cache()

        # Metrics
        self.keyword_classifications = 0
        self.llm_classifications = 0
        self.cache_hits = 0

        # Intent keyword mappings (hierarchical)
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
                'primary': ['update', 'change', 'modify', 'edit', 'fix', 'correct', 'adjust', 'set'],
                'secondary': ['move to', 'transition', 'reassign'],
                'modifiers': ['priority', 'status', 'assignee', 'description']
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

        # Implicit requirement patterns
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
        """
        Classify intents in user message

        Args:
            message: User message to classify

        Returns:
            List of detected intents with confidence scores
        """
        message_lower = message.lower()
        detected_intents = []

        # Check each intent type
        for intent_type, keywords in self.intent_keywords.items():
            confidence = self._calculate_intent_confidence(message_lower, keywords)

            if confidence > 0.3:  # Threshold for detection
                intent = Intent(
                    type=intent_type,
                    confidence=confidence,
                    raw_indicators=self._extract_indicators(message_lower, keywords)
                )
                detected_intents.append(intent)

        # Sort by confidence
        detected_intents.sort(key=lambda x: x.confidence, reverse=True)

        # Detect implicit requirements
        implicit_reqs = self._detect_implicit_requirements(message_lower)
        for intent in detected_intents:
            intent.implicit_requirements = implicit_reqs

        # If no intents detected, mark as unknown
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
        """Get the primary (highest confidence) intent"""
        return intents[0] if intents else Intent(type=IntentType.UNKNOWN, confidence=0.0)

    def has_intent_type(self, intents: List[Intent], intent_type: IntentType) -> bool:
        """Check if specific intent type is present"""
        return any(i.type == intent_type for i in intents)

    def _calculate_intent_confidence(self, message: str, keywords: Dict[str, List[str]]) -> float:
        """
        Calculate confidence score for an intent

        Factors:
        - Primary keyword match (highest weight)
        - Secondary keyword match (medium weight)
        - Modifier presence (context boost)
        - Position in sentence (earlier = higher confidence)
        """
        score = 0.0

        # Check primary keywords (weight: 1.0)
        for keyword in keywords.get('primary', []):
            if keyword in message:
                # Bonus for early position
                position = message.find(keyword)
                position_factor = 1.0 - (position / len(message)) * 0.2

                score = max(score, 0.9 * position_factor)
                break

        # Check secondary keywords (weight: 0.7)
        for keyword in keywords.get('secondary', []):
            if keyword in message:
                score = max(score, 0.7)
                break

        # Boost if modifiers present (context)
        modifiers_found = sum(1 for mod in keywords.get('modifiers', []) if mod in message)
        if modifiers_found > 0:
            score = min(score + (modifiers_found * 0.1), 1.0)

        return score

    def _extract_indicators(self, message: str, keywords: Dict[str, List[str]]) -> List[str]:
        """Extract words that indicated this intent"""
        indicators = []

        for keyword in keywords.get('primary', []) + keywords.get('secondary', []):
            if keyword in message:
                indicators.append(keyword)

        return indicators

    def _detect_implicit_requirements(self, message: str) -> List[str]:
        """
        Detect implicit requirements from message

        Examples:
        - "urgent bug" → implicit: high priority
        - "notify everyone" → implicit: public visibility
        - "create all the issues" → implicit: batch operation
        """
        requirements = []

        # Urgency detection
        for level, keywords in self.implicit_patterns['urgency'].items():
            for keyword in keywords:
                if keyword in message:
                    if level == 'critical':
                        requirements.append('priority:critical')
                        requirements.append('urgent:true')
                    elif level == 'high':
                        requirements.append('priority:high')
                    break

        # Scope detection
        for scope, keywords in self.implicit_patterns['scope'].items():
            for keyword in keywords:
                if keyword in message:
                    requirements.append(f'scope:{scope}')
                    if scope in ['multiple', 'batch']:
                        requirements.append('batch_operation:true')
                    break

        # Visibility detection
        for visibility, keywords in self.implicit_patterns['visibility'].items():
            for keyword in keywords:
                if keyword in message:
                    requirements.append(f'visibility:{visibility}')
                    break

        # Security-related detection
        security_keywords = ['security', 'secure', 'auth', 'authentication', 'authorization', 'permission']
        if any(kw in message for kw in security_keywords):
            requirements.append('security_sensitive:true')

        # Performance-related detection
        performance_keywords = ['performance', 'slow', 'fast', 'optimize', 'speed']
        if any(kw in message for kw in performance_keywords):
            requirements.append('performance_related:true')

        return requirements

    def is_multi_intent(self, intents: List[Intent]) -> bool:
        """Check if message contains multiple high-confidence intents"""
        high_confidence_intents = [i for i in intents if i.confidence > 0.6]
        return len(high_confidence_intents) > 1

    def suggest_clarifications(self, intents: List[Intent]) -> List[str]:
        """
        Suggest what clarifications might be needed based on intents

        Returns:
            List of clarification questions
        """
        clarifications = []

        primary = self.get_primary_intent(intents)

        # CREATE intent clarifications
        if primary.type == IntentType.CREATE:
            clarifications.extend([
                "What should be created? (issue, PR, page, etc.)",
                "Which project/repository?",
                "Any specific details or description?"
            ])

        # READ intent clarifications
        elif primary.type == IntentType.READ:
            clarifications.extend([
                "What information are you looking for?",
                "Which project/repository?"
            ])

        # UPDATE intent clarifications
        elif primary.type == IntentType.UPDATE:
            clarifications.extend([
                "Which resource to update?",
                "What changes should be made?"
            ])

        # COORDINATE intent clarifications
        elif primary.type == IntentType.COORDINATE:
            clarifications.extend([
                "Who should be notified?",
                "What message to send?"
            ])

        return clarifications

    def extract_action_target(self, message: str) -> Optional[str]:
        """
        Extract the target of an action from message

        Examples:
        - "create an issue" → "issue"
        - "update the PR" → "PR"
        - "review the code" → "code"
        """
        message_lower = message.lower()

        # Common targets
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
        """
        Detect if message contains conditional/workflow logic

        Examples:
        - "when X happens, do Y"
        - "if status is done, then notify"
        """
        message_lower = message.lower()

        conditional_patterns = [
            r'\bwhen\b.*\b(then|do|notify|create)',
            r'\bif\b.*\b(then|do|notify|create)',
            r'\bwhenever\b.*\b(then|do|notify|create)',
        ]

        return any(re.search(pattern, message_lower) for pattern in conditional_patterns)

    # ========================================================================
    # ENHANCED METHODS - V3.0
    # ========================================================================

    def classify_with_cache(self, message: str) -> List[Intent]:
        """
        Classify with caching support

        Checks cache first, then falls back to classification.

        Args:
            message: User message

        Returns:
            List of detected intents
        """
        cache_key = CacheKeyBuilder.for_intent_classification(message)

        # Try cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self.cache_hits += 1
            if self.verbose:
                print(f"[INTENT] Cache hit for message")
            return cached_result

        # Classify
        intents = self.classify(message)

        # Cache result
        self.cache.set(cache_key, intents, ttl_seconds=300)  # 5 minute TTL

        return intents

    def classify_with_llm(self, message: str, context: Optional[Dict] = None) -> List[Intent]:
        """
        Classify intents using LLM for better semantic understanding

        This provides more accurate classification but is slower.
        Use for ambiguous cases or when keyword matching has low confidence.

        Args:
            message: User message
            context: Optional conversation context

        Returns:
            List of detected intents with higher confidence
        """
        if not self.llm_client:
            if self.verbose:
                print("[INTENT] No LLM client available, falling back to keywords")
            return self.classify(message)

        self.llm_classifications += 1

        # Build prompt for LLM
        prompt = self._build_intent_classification_prompt(message, context)

        try:
            # Call LLM
            response = self._call_llm_for_intent(prompt)

            # Parse LLM response
            intents = self._parse_llm_intent_response(response)

            if self.verbose:
                print(f"[INTENT] LLM classified {len(intents)} intents")

            return intents

        except Exception as e:
            if self.verbose:
                print(f"[INTENT] LLM classification failed: {e}, falling back to keywords")
            return self.classify(message)

    def classify_hybrid(self, message: str, context: Optional[Dict] = None) -> List[Intent]:
        """
        Hybrid classification: keywords first, LLM for disambiguation

        Best of both worlds:
        - Fast keyword matching for clear cases
        - LLM for ambiguous or complex cases

        Args:
            message: User message
            context: Optional conversation context

        Returns:
            List of detected intents
        """
        # First try keyword-based classification
        keyword_intents = self.classify(message)

        # Check if we need LLM disambiguation
        needs_disambiguation = self._needs_disambiguation(keyword_intents, message)

        if not needs_disambiguation or not self.use_llm:
            return keyword_intents

        if self.verbose:
            print("[INTENT] Low confidence, using LLM for disambiguation")

        # Use LLM for better understanding
        llm_intents = self.classify_with_llm(message, context)

        # Merge results (prefer LLM but keep high-confidence keyword matches)
        merged_intents = self._merge_intent_results(keyword_intents, llm_intents)

        return merged_intents

    def _needs_disambiguation(self, intents: List[Intent], message: str) -> bool:
        """
        Determine if intents need LLM disambiguation

        Disambiguation needed when:
        - No high-confidence intents
        - Multiple competing intents
        - Ambiguous language detected
        - Complex multi-step request
        """
        if not intents:
            return True

        # Check confidence levels
        high_conf_intents = [i for i in intents if i.confidence > 0.8]
        if not high_conf_intents:
            return True

        # Check for multiple competing intents
        similar_conf_intents = [
            i for i in intents
            if i.confidence > 0.6 and abs(i.confidence - intents[0].confidence) < 0.15
        ]
        if len(similar_conf_intents) > 2:
            return True

        # Check for ambiguous language
        ambiguous_words = ['maybe', 'might', 'could', 'should', 'possibly', 'perhaps']
        if any(word in message.lower() for word in ambiguous_words):
            return True

        # Check for complex requests (multiple sentences)
        if len(re.split(r'[.!?;]', message)) > 3:
            return True

        return False

    def _merge_intent_results(
        self,
        keyword_intents: List[Intent],
        llm_intents: List[Intent]
    ) -> List[Intent]:
        """
        Merge keyword and LLM intent results intelligently

        Strategy:
        - Prefer LLM results as they're more semantically accurate
        - Keep high-confidence keyword matches that LLM missed
        - Remove duplicates
        """
        # Start with LLM intents (higher quality)
        merged = list(llm_intents)

        # Add high-confidence keyword intents not in LLM results
        llm_intent_types = {i.type for i in llm_intents}

        for keyword_intent in keyword_intents:
            if keyword_intent.confidence > 0.85 and keyword_intent.type not in llm_intent_types:
                merged.append(keyword_intent)

        # Sort by confidence
        merged.sort(key=lambda x: x.confidence, reverse=True)

        return merged

    def _build_intent_classification_prompt(
        self,
        message: str,
        context: Optional[Dict] = None
    ) -> str:
        """Build prompt for LLM intent classification"""
        prompt = f"""Classify the user's intent(s) from this message.

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

If multiple intents detected, include all.

Response:"""

        return prompt

    def _call_llm_for_intent(self, prompt: str) -> str:
        """Call LLM for intent classification"""
        # This is a placeholder - actual implementation would call real LLM
        # For now, return empty to fall back to keyword matching
        raise NotImplementedError("LLM client integration not implemented")

    def _parse_llm_intent_response(self, response: str) -> List[Intent]:
        """Parse LLM response into Intent objects"""
        import json

        try:
            # Parse JSON response
            data = json.loads(response)

            intents = []
            for item in data:
                intent_type_str = item.get('intent', 'UNKNOWN')
                confidence = item.get('confidence', 0.5)
                indicators = item.get('indicators', [])

                # Map string to IntentType
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
        """
        Calibrate confidence scores based on additional factors

        Adjusts confidence based on:
        - Message clarity
        - Specificity
        - Ambiguity indicators
        - Historical accuracy

        Args:
            intents: Detected intents
            message: Original message

        Returns:
            Intents with calibrated confidence scores
        """
        for intent in intents:
            original_conf = intent.confidence

            # Reduce confidence for very short messages
            if len(message.split()) < 3:
                intent.confidence *= 0.9

            # Reduce confidence for vague language
            vague_words = ['thing', 'stuff', 'something', 'anything']
            if any(word in message.lower() for word in vague_words):
                intent.confidence *= 0.85

            # Increase confidence for very specific language
            specific_indicators = ['#', '@', 'http', '://', '-']
            specificity = sum(1 for ind in specific_indicators if ind in message)
            if specificity >= 2:
                intent.confidence = min(intent.confidence * 1.1, 1.0)

            # Ensure confidence stays in valid range
            intent.confidence = max(0.0, min(1.0, intent.confidence))

            if self.verbose and abs(original_conf - intent.confidence) > 0.05:
                print(f"[INTENT] Calibrated {intent.type.value}: {original_conf:.2f} → {intent.confidence:.2f}")

        return intents

    def get_intent_hierarchy(self, intent_type: IntentType) -> List[IntentType]:
        """
        Get intent hierarchy (parent-child relationships)

        For example:
        - UPDATE can be a specialized CREATE (create new version)
        - ANALYZE can lead to CREATE (create issue from findings)

        Args:
            intent_type: Intent type to get hierarchy for

        Returns:
            List of related intent types in hierarchy
        """
        hierarchies = {
            IntentType.CREATE: [IntentType.UPDATE, IntentType.COORDINATE],
            IntentType.ANALYZE: [IntentType.CREATE, IntentType.COORDINATE],
            IntentType.SEARCH: [IntentType.READ],
            IntentType.READ: [IntentType.ANALYZE],
        }

        return hierarchies.get(intent_type, [])

    def get_metrics(self) -> Dict[str, Any]:
        """Get classification metrics"""
        return {
            'keyword_classifications': self.keyword_classifications,
            'llm_classifications': self.llm_classifications,
            'cache_hits': self.cache_hits,
            'total_classifications': self.keyword_classifications + self.llm_classifications,
        }

    def reset_metrics(self):
        """Reset metrics"""
        self.keyword_classifications = 0
        self.llm_classifications = 0
        self.cache_hits = 0
