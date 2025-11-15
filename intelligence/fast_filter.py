"""Fast Keyword Pre-Filter for Obvious Intent Classification

Handles simple, unambiguous requests with < 10ms latency and $0 cost.
Routes complex requests to LLM for accurate classification.
"""

from typing import Optional, Tuple
from intelligence.base_types import IntentType, Intent, Entity, EntityType
import re


class FastKeywordFilter:
    """
    Lightning-fast keyword-based pre-filter for obvious intents.

    Handles ~30-40% of simple requests instantly (< 10ms).
    Returns None for complex cases that need LLM classification.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.fast_classifications = 0
        self.passed_to_llm = 0

        # High-confidence keyword patterns for each intent
        self.intent_patterns = {
            IntentType.CREATE: {
                'keywords': ['create', 'add', 'new', 'make', 'open'],
                'patterns': [
                    r'\bcreate\s+(a|an|new)?\s*(issue|pr|ticket|page|channel)',
                    r'\badd\s+(a|an|new)?\s*(issue|task|item)',
                    r'\bopen\s+(a|an|new)?\s*(issue|pr|ticket)',
                ],
                'min_confidence': 0.9
            },
            IntentType.DELETE: {
                'keywords': ['delete', 'remove', 'close', 'archive'],
                'patterns': [
                    r'\bdelete\s+(the|this|that)?\s*(issue|pr|message|page)',
                    r'\bremove\s+(the|this|that)?\s*(issue|pr|file)',
                    r'\bclose\s+(the|this|that)?\s*(issue|pr|ticket)',
                ],
                'min_confidence': 0.95  # Higher confidence for destructive ops
            },
            IntentType.READ: {
                'keywords': ['show', 'get', 'list', 'display', 'view', 'fetch', 'find'],
                'patterns': [
                    r'\bshow\s+me',
                    r'\bget\s+(the|all|my)?\s*(issues|prs|messages|tasks?|tickets?)',
                    r'\blist\s+(all\s+)?(my\s+)?(issues|prs|channels|tasks?|tickets?)',
                    r'\bfetch\s+(the|all|my)?\s*(issues|prs|tasks?|tickets?)',
                    r'\b(my|the)\s+(issues|tasks?|tickets?)\s+(from|in)',
                ],
                'min_confidence': 0.85
            },
            IntentType.UPDATE: {
                'keywords': ['update', 'edit', 'modify', 'change'],
                'patterns': [
                    r'\bupdate\s+(the|this|that)?\s*(issue|pr|page)',
                    r'\bedit\s+(the|this|that)?\s*(description|title)',
                    r'\bchange\s+(the|this|that)?\s*(status|priority)',
                ],
                'min_confidence': 0.9
            },
            IntentType.SEARCH: {
                'keywords': ['search', 'find', 'query', 'lookup'],
                'patterns': [
                    r'\bsearch\s+for',
                    r'\bfind\s+(all|any)?\s*(issues|prs|files)',
                ],
                'min_confidence': 0.85
            },
        }

    def classify_fast(self, message: str) -> Optional[Tuple[IntentType, float]]:
        """
        Attempt fast classification based on obvious keywords.

        Args:
            message: User's message

        Returns:
            (IntentType, confidence) if certain, None otherwise
        """
        message_lower = message.lower().strip()

        # Check each intent pattern
        for intent_type, config in self.intent_patterns.items():
            confidence = 0.0

            # Check regex patterns (strongest signal)
            for pattern in config['patterns']:
                if re.search(pattern, message_lower):
                    confidence = max(confidence, config['min_confidence'])
                    break

            # Check keywords (weaker signal)
            if confidence < config['min_confidence']:
                for keyword in config['keywords']:
                    if f' {keyword} ' in f' {message_lower} ':
                        # Keyword match but not pattern - lower confidence
                        confidence = max(confidence, config['min_confidence'] - 0.15)
                        break

            # If confident enough, return immediately
            if confidence >= config['min_confidence']:
                self.fast_classifications += 1
                if self.verbose:
                    print(f"[FAST] Classified as {intent_type.value} (conf={confidence:.2f})")
                return (intent_type, confidence)

        # No confident classification - pass to LLM
        self.passed_to_llm += 1
        return None

    def is_certain(self, message: str) -> bool:
        """Check if message can be classified with high certainty"""
        result = self.classify_fast(message)
        return result is not None

    def to_legacy_intent(self, intent_type: IntentType, confidence: float, message: str) -> Intent:
        """Convert to legacy Intent format"""
        # Simple entity extraction for common patterns
        entities = []
        message_lower = message.lower()

        # Extract issue keys (KAN-123, PROJ-456, etc.)
        issue_matches = re.findall(r'\b([A-Z]+-\d+)\b', message)
        for issue_key in issue_matches:
            entities.append(Entity(
                type=EntityType.ISSUE,
                value=issue_key,
                confidence=0.95
            ))

        # Extract project names (uppercase words, 2-10 chars)
        # Look for patterns like "from KAN", "in JIRA", "project ABC"
        project_matches = re.findall(r'\b(?:from|in|project|to)\s+([A-Z]{2,10})\b', message)
        for project in project_matches:
            # Avoid duplicates from issue keys
            if not any(e.value == project for e in entities):
                entities.append(Entity(
                    type=EntityType.PROJECT,
                    value=project,
                    confidence=0.8
                ))

        # Extract service/platform keywords (jira, github, slack, notion)
        service_keywords = {
            'jira': EntityType.PROJECT,
            'github': EntityType.REPOSITORY,
            'slack': EntityType.CHANNEL,
            'notion': EntityType.RESOURCE
        }
        for keyword, entity_type in service_keywords.items():
            if keyword in message_lower:
                entities.append(Entity(
                    type=entity_type,
                    value=keyword,
                    confidence=0.7
                ))

        # Extract PR numbers (#123)
        pr_matches = re.findall(r'#(\d+)', message)
        for pr_num in pr_matches:
            entities.append(Entity(
                type=EntityType.PR,
                value=pr_num,
                confidence=0.9
            ))

        # Extract Slack channels (#channel-name)
        channel_matches = re.findall(r'#([a-z0-9-]+)', message)
        for channel in channel_matches:
            if not channel.isdigit():  # Exclude PR numbers
                entities.append(Entity(
                    type=EntityType.CHANNEL,
                    value=channel,
                    confidence=0.85
                ))

        return Intent(
            type=intent_type,
            confidence=confidence,
            entities=entities,
            implicit_requirements=[],
            raw_indicators=["fast_filter"]
        )

    def get_statistics(self) -> dict:
        """Get filter statistics"""
        total = self.fast_classifications + self.passed_to_llm
        fast_rate = self.fast_classifications / total if total > 0 else 0

        return {
            'fast_classifications': self.fast_classifications,
            'passed_to_llm': self.passed_to_llm,
            'fast_path_rate': fast_rate,
            'total_requests': total
        }
