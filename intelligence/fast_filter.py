"""
Fast Keyword Filter - Tier 1 Intelligence

Ultra-fast pattern matching for obvious, high-confidence requests.
This is the first line of defense in the hybrid intelligence system.

Performance Targets:
- Latency: < 10ms per classification
- Cost: $0 (no API calls)
- Coverage: 35-40% of requests
- Accuracy: 95% for covered patterns

Author: AI System (Senior Developer)
Version: 5.0 - Production Implementation
"""

import re
import time
from typing import Dict, List, Optional, Tuple, Set

from .base_types import Intent, IntentType, Entity, EntityType


class FastKeywordFilter:
    """
    Tier 1: Fast keyword-based intent classification.

    Uses regex patterns and keyword matching to quickly identify
    obvious intents with high confidence (0.85-0.95).

    This is designed for SPEED - handles simple, clear requests instantly.
    """

    # Intent patterns with confidence thresholds
    PATTERNS = {
        IntentType.CREATE: {
            'keywords': ['create', 'make', 'add', 'new', 'start', 'open', 'build', 'generate', 'initialize'],
            'confidence_threshold': 0.90,
            'entity_hints': ['issue', 'pr', 'pull request', 'channel', 'page', 'task', 'ticket'],
            'bonus_phrases': ['create new', 'make a', 'add a', 'start a']
        },
        IntentType.DELETE: {
            'keywords': ['delete', 'remove', 'destroy', 'drop', 'cancel'],
            'confidence_threshold': 0.95,  # High confidence required for DELETE
            'entity_hints': ['issue', 'pr', 'channel', 'file', 'task'],
            'bonus_phrases': ['delete the', 'remove the', 'get rid of']
        },
        IntentType.UPDATE: {
            'keywords': ['update', 'modify', 'change', 'edit', 'set', 'fix', 'adjust', 'revise'],
            'confidence_threshold': 0.85,
            'entity_hints': ['status', 'title', 'description', 'assignee', 'priority', 'label'],
            'bonus_phrases': ['update the', 'change the', 'set the', 'modify the']
        },
        IntentType.READ: {
            'keywords': ['show', 'get', 'fetch', 'find', 'list', 'display', 'view', 'retrieve', 'see'],
            'confidence_threshold': 0.85,
            'entity_hints': ['issue', 'pr', 'file', 'channel', 'status', 'details'],
            'bonus_phrases': ['show me', 'get me', 'what is', 'display the']
        },
        IntentType.SEARCH: {
            'keywords': ['search', 'find', 'lookup', 'query', 'locate', 'hunt'],
            'confidence_threshold': 0.90,
            'entity_hints': ['for', 'with', 'containing', 'matching', 'about'],
            'bonus_phrases': ['search for', 'find all', 'look for']
        },
        IntentType.ANALYZE: {
            'keywords': ['analyze', 'review', 'check', 'inspect', 'examine', 'evaluate', 'assess', 'audit'],
            'confidence_threshold': 0.88,
            'entity_hints': ['code', 'pr', 'pull request', 'quality', 'security', 'performance'],
            'bonus_phrases': ['review the', 'check the', 'analyze the']
        },
        IntentType.COORDINATE: {
            'keywords': ['notify', 'tell', 'inform', 'alert', 'message', 'ping', 'send', 'post', 'share'],
            'confidence_threshold': 0.87,
            'entity_hints': ['team', 'channel', 'person', 'slack', 'email', 'everyone'],
            'bonus_phrases': ['notify the', 'tell the', 'send to', 'post to']
        }
    }

    def __init__(self, verbose: bool = False):
        """
        Initialize fast filter

        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose

        # Performance metrics
        self.total_classifications = 0
        self.fast_hits = 0
        self.total_latency_ms = 0.0

        # Pre-compile regex patterns for speed
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for faster matching"""
        self.compiled_patterns = {}

        for intent_type, pattern_data in self.PATTERNS.items():
            # Compile keyword patterns with word boundaries
            keyword_regex = r'\b(' + '|'.join(re.escape(kw) for kw in pattern_data['keywords']) + r')\b'
            self.compiled_patterns[intent_type] = {
                'keyword_regex': re.compile(keyword_regex, re.IGNORECASE),
                'entity_hints': pattern_data['entity_hints'],
                'bonus_phrases': pattern_data['bonus_phrases'],
                'confidence_threshold': pattern_data['confidence_threshold']
            }

    def classify_fast(self, message: str) -> Optional[Tuple[IntentType, float, List[str]]]:
        """
        Attempt fast classification using keyword patterns.

        Args:
            message: User message to classify

        Returns:
            (IntentType, confidence, indicators) if high-confidence match found
            None if unable to classify with confidence

        Performance: < 10ms on average
        """
        start_time = time.time()
        self.total_classifications += 1

        message_lower = message.lower()
        message_words = set(message_lower.split())

        best_match = None
        best_confidence = 0.0
        best_indicators = []

        # Check each intent type
        for intent_type, pattern in self.compiled_patterns.items():
            confidence, indicators = self._calculate_confidence(
                message_lower,
                message_words,
                pattern
            )

            threshold = self.PATTERNS[intent_type]['confidence_threshold']

            # Check if exceeds threshold
            if confidence >= threshold and confidence > best_confidence:
                best_match = intent_type
                best_confidence = confidence
                best_indicators = indicators

        # Record latency
        latency_ms = (time.time() - start_time) * 1000
        self.total_latency_ms += latency_ms

        if best_match:
            self.fast_hits += 1
            if self.verbose:
                print(f"[FAST FILTER] Match: {best_match.value} ({best_confidence:.2f}) in {latency_ms:.1f}ms")
            return (best_match, best_confidence, best_indicators)

        if self.verbose:
            print(f"[FAST FILTER] No high-confidence match in {latency_ms:.1f}ms")

        return None

    def _calculate_confidence(
        self,
        message: str,
        message_words: Set[str],
        pattern: Dict
    ) -> Tuple[float, List[str]]:
        """
        Calculate confidence score for an intent pattern

        Factors:
        - Primary keyword match (highest weight)
        - Bonus phrase match (high weight)
        - Entity hints present (context boost)
        - Position in sentence (earlier = higher confidence)
        """
        score = 0.0
        indicators = []

        # 1. Check primary keywords (weight: up to 0.9)
        keyword_match = pattern['keyword_regex'].search(message)
        if keyword_match:
            matched_keyword = keyword_match.group(0)
            indicators.append(matched_keyword)

            # Position factor (earlier in message = higher confidence)
            position = keyword_match.start()
            position_factor = 1.0 - (position / max(len(message), 1)) * 0.15

            score = 0.80 * position_factor

        # 2. Check bonus phrases (weight: +0.15)
        for phrase in pattern['bonus_phrases']:
            if phrase in message:
                score += 0.15
                indicators.append(phrase)
                break  # Only count one bonus phrase

        # 3. Check entity hints (context boost: +0.05 per hint, max 0.15)
        entity_matches = 0
        for hint in pattern['entity_hints']:
            if hint in message:
                entity_matches += 1
                if entity_matches <= 3:  # Max 3 hints
                    score += 0.05
                    indicators.append(f"entity:{hint}")

        # Cap at 0.99 (never 100% certain with keywords alone)
        score = min(score, 0.99)

        return score, indicators

    def classify_with_entities(
        self,
        message: str
    ) -> Optional[Tuple[IntentType, float, List[str], List[Entity]]]:
        """
        Fast classification with basic entity extraction

        Args:
            message: User message

        Returns:
            (intent_type, confidence, indicators, entities) or None
        """
        # First get intent
        result = self.classify_fast(message)
        if not result:
            return None

        intent_type, confidence, indicators = result

        # Extract basic entities (fast patterns)
        entities = self._extract_fast_entities(message)

        return (intent_type, confidence, indicators, entities)

    def _extract_fast_entities(self, message: str) -> List[Entity]:
        """
        Extract obvious entities using fast pattern matching

        Extracts:
        - Issue IDs (e.g., KAN-123, PROJ-456)
        - PR numbers (e.g., #123, PR-456)
        - Channel names (e.g., #general, @channel)
        - Users (e.g., @username)
        - URLs
        """
        entities = []

        # Issue IDs: PROJECT-123 pattern
        issue_pattern = r'\b([A-Z]{2,10}-\d+)\b'
        for match in re.finditer(issue_pattern, message):
            entities.append(Entity(
                type=EntityType.ISSUE,
                value=match.group(1),
                confidence=0.95,
                context=message[max(0, match.start()-20):match.end()+20]
            ))

        # PR numbers: #123 or PR-123
        pr_pattern = r'(?:#|PR-?)(\d+)\b'
        for match in re.finditer(pr_pattern, message, re.IGNORECASE):
            entities.append(Entity(
                type=EntityType.PR,
                value=match.group(1),
                confidence=0.90,
                context=message[max(0, match.start()-20):match.end()+20]
            ))

        # Slack channels: #channel-name
        channel_pattern = r'#([a-z0-9\-_]+)'
        for match in re.finditer(channel_pattern, message, re.IGNORECASE):
            entities.append(Entity(
                type=EntityType.CHANNEL,
                value=match.group(1),
                confidence=0.92,
                context=message[max(0, match.start()-20):match.end()+20]
            ))

        # Users: @username
        user_pattern = r'@([a-zA-Z0-9\-_]+)'
        for match in re.finditer(user_pattern, message):
            entities.append(Entity(
                type=EntityType.PERSON,
                value=match.group(1),
                confidence=0.88,
                context=message[max(0, match.start()-20):match.end()+20]
            ))

        # Priority keywords
        priority_keywords = {
            'critical': 'critical',
            'urgent': 'critical',
            'high': 'high',
            'medium': 'medium',
            'low': 'low'
        }
        message_lower = message.lower()
        for keyword, priority in priority_keywords.items():
            if keyword in message_lower:
                entities.append(Entity(
                    type=EntityType.PRIORITY,
                    value=priority,
                    confidence=0.85
                ))
                break

        return entities

    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        hit_rate = (self.fast_hits / self.total_classifications * 100) if self.total_classifications > 0 else 0
        avg_latency = (self.total_latency_ms / self.total_classifications) if self.total_classifications > 0 else 0

        return {
            'total_classifications': self.total_classifications,
            'fast_hits': self.fast_hits,
            'fast_hit_rate': f"{hit_rate:.1f}%",
            'avg_latency_ms': f"{avg_latency:.2f}",
            'target_coverage': '35-40%',
            'actual_coverage': f"{hit_rate:.1f}%"
        }

    def to_legacy_intent(
        self,
        intent_type: IntentType,
        confidence: float,
        message: str,
        indicators: List[str]
    ) -> Intent:
        """
        Convert fast filter result to legacy Intent object

        Args:
            intent_type: Detected intent type
            confidence: Confidence score
            message: Original message
            indicators: Keywords/phrases that triggered match

        Returns:
            Intent object
        """
        return Intent(
            type=intent_type,
            confidence=confidence,
            raw_indicators=indicators
        )

    def reset_statistics(self):
        """Reset performance metrics"""
        self.total_classifications = 0
        self.fast_hits = 0
        self.total_latency_ms = 0.0
