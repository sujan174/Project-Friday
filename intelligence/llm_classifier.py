"""
LLM-Based Intent Classifier - Tier 2 Intelligence

Deep semantic understanding for complex requests using LLM.
This is the fallback when fast keyword matching is insufficient.

Performance Targets:
- Latency: ~200ms (without cache), ~20ms (with cache)
- Cost: $0.01/1K requests
- Coverage: 60-65% of requests (those not handled by fast filter)
- Accuracy: 92%
- Cache hit rate: 70-80%

Author: AI System (Senior Developer)
Version: 5.0 - Production Implementation
"""

import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .base_types import Intent, IntentType, Entity, EntityType
from .cache_layer import get_global_cache, CacheKeyBuilder


@dataclass
class LLMClassificationResult:
    """Result from LLM classification"""
    intents: List[Dict[str, Any]]
    entities: List[Dict[str, Any]]
    confidence: float
    ambiguities: List[str]
    suggested_clarifications: List[str]
    reasoning: str
    latency_ms: float = 0.0
    cache_hit: bool = False


class LLMIntentClassifier:
    """
    Tier 2: LLM-based semantic intent classification.

    Uses Gemini Flash for deep semantic understanding of complex
    requests that cannot be handled by fast keyword matching.

    Includes semantic caching for 70-80% cache hit rate.
    """

    CLASSIFICATION_PROMPT_TEMPLATE = """Analyze the user's message and classify their intent.

You must identify:
1. PRIMARY INTENT: What is the user trying to do?
   - CREATE: Make something new
   - READ: View/retrieve information
   - UPDATE: Modify existing data
   - DELETE: Remove something
   - SEARCH: Find specific information
   - EXECUTE: Run an operation
   - APPROVE: Accept/approve something
   - REJECT: Deny/reject something
   - ANALYZE: Examine/evaluate data
   - COORDINATE: Notify or communicate with others
   - WORKFLOW: Automation or conditional logic
   - UNKNOWN: Cannot determine

2. ENTITIES: Extract specific entities mentioned
   - Types: ISSUE, PR, CHANNEL, PROJECT, FILE, USER, DATE, STATUS, PRIORITY, LABEL, ASSIGNEE, REPOSITORY, BRANCH, WORKSPACE, PAGE, DATABASE, COMMENT, REACTION, SPRINT, EPIC, UNKNOWN
   - Include the actual value (e.g., "KAN-123", "#general", "@john")

3. CONFIDENCE: How confident are you? (0.0 to 1.0)

4. AMBIGUITIES: What is unclear or could be interpreted multiple ways?

5. SUGGESTED CLARIFICATIONS: What questions would resolve ambiguities?

User Message: "{message}"

Conversation Context (if available):
{context}

Respond in JSON format:
{{
  "intents": [
    {{
      "type": "CREATE",
      "confidence": 0.95,
      "reasoning": "User explicitly says 'create new issue'"
    }}
  ],
  "entities": [
    {{
      "type": "PROJECT",
      "value": "KAN",
      "confidence": 0.90
    }}
  ],
  "confidence": 0.95,
  "ambiguities": [],
  "suggested_clarifications": [],
  "reasoning": "Clear request to create a new issue in the KAN project"
}}

IMPORTANT: Return ONLY valid JSON. No additional text."""

    def __init__(self, llm_client: Optional[Any] = None, verbose: bool = False):
        """
        Initialize LLM classifier

        Args:
            llm_client: LLM client (e.g., Gemini Flash)
            verbose: Enable verbose logging
        """
        self.llm_client = llm_client
        self.verbose = verbose
        self.cache = get_global_cache()

        # Performance metrics
        self.total_classifications = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_latency_ms = 0.0
        self.llm_calls = 0

    async def classify(
        self,
        message: str,
        context: Optional[Dict] = None
    ) -> LLMClassificationResult:
        """
        Classify intent using LLM with semantic caching.

        Implementation:
        1. Check semantic cache (exact match first)
        2. If cache hit, return cached result (~20ms)
        3. If cache miss, call LLM (~200ms)
        4. Store result in cache
        5. Return classification

        Args:
            message: User message to classify
            context: Optional conversation context

        Returns:
            LLM classification result
        """
        start_time = time.time()
        self.total_classifications += 1

        # Check cache first (exact match)
        cache_key = CacheKeyBuilder.for_intent_classification(message)
        cached = self.cache.get(cache_key)

        if cached:
            self.cache_hits += 1
            latency_ms = (time.time() - start_time) * 1000
            self.total_latency_ms += latency_ms

            if self.verbose:
                print(f"[LLM CLASSIFIER] Cache hit in {latency_ms:.1f}ms")

            cached['cache_hit'] = True
            cached['latency_ms'] = latency_ms
            return LLMClassificationResult(**cached)

        self.cache_misses += 1

        # Build prompt with context
        context_str = json.dumps(context, indent=2) if context else "None"
        prompt = self.CLASSIFICATION_PROMPT_TEMPLATE.format(
            message=message,
            context=context_str
        )

        # Call LLM
        try:
            if self.llm_client:
                response = await self._call_llm(prompt)
            else:
                # Fallback: return low-confidence unknown
                response = {
                    'intents': [{'type': 'UNKNOWN', 'confidence': 0.5, 'reasoning': 'No LLM available'}],
                    'entities': [],
                    'confidence': 0.5,
                    'ambiguities': ['Cannot classify without LLM'],
                    'suggested_clarifications': ['Please rephrase your request'],
                    'reasoning': 'LLM classifier not configured'
                }
        except Exception as e:
            if self.verbose:
                print(f"[LLM CLASSIFIER] Error calling LLM: {e}")
            # Return error result
            response = {
                'intents': [{'type': 'UNKNOWN', 'confidence': 0.3, 'reasoning': f'LLM error: {str(e)}'}],
                'entities': [],
                'confidence': 0.3,
                'ambiguities': ['LLM classification failed'],
                'suggested_clarifications': [],
                'reasoning': f'Error: {str(e)}'
            }

        latency_ms = (time.time() - start_time) * 1000
        self.total_latency_ms += latency_ms

        # Create result
        result_data = {
            'intents': response.get('intents', []),
            'entities': response.get('entities', []),
            'confidence': response.get('confidence', 0.5),
            'ambiguities': response.get('ambiguities', []),
            'suggested_clarifications': response.get('suggested_clarifications', []),
            'reasoning': response.get('reasoning', ''),
            'latency_ms': latency_ms,
            'cache_hit': False
        }

        result = LLMClassificationResult(**result_data)

        # Cache for future use (5 minute TTL)
        self.cache.set(cache_key, result_data, ttl_seconds=300)

        if self.verbose:
            print(f"[LLM CLASSIFIER] LLM classification in {latency_ms:.1f}ms (confidence: {result.confidence:.2f})")

        return result

    async def _call_llm(self, prompt: str) -> Dict:
        """
        Call LLM for classification

        Args:
            prompt: Classification prompt

        Returns:
            Parsed JSON response from LLM
        """
        self.llm_calls += 1

        # Call LLM (async)
        if hasattr(self.llm_client, 'generate_content'):
            response = await self.llm_client.generate_content(prompt)
            response_text = response.text
        elif hasattr(self.llm_client, 'generate'):
            response = await self.llm_client.generate(prompt)
            response_text = response.get('text', '')
        else:
            raise ValueError("LLM client does not have generate_content or generate method")

        # Parse JSON response
        result = self._parse_llm_response(response_text)
        return result

    def _parse_llm_response(self, response_text: str) -> Dict:
        """
        Parse LLM response into structured format

        Args:
            response_text: Raw LLM response

        Returns:
            Parsed dictionary
        """
        try:
            # Try to extract JSON from response
            # Sometimes LLM adds explanatory text before/after JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)
                return data
            else:
                raise ValueError("No JSON found in response")

        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"[LLM CLASSIFIER] JSON parse error: {e}")
                print(f"Response: {response_text[:200]}")

            # Return fallback response
            return {
                'intents': [{'type': 'UNKNOWN', 'confidence': 0.4, 'reasoning': 'Parse error'}],
                'entities': [],
                'confidence': 0.4,
                'ambiguities': ['Could not parse LLM response'],
                'suggested_clarifications': [],
                'reasoning': 'JSON parse error'
            }

    def convert_to_legacy_format(self, result: LLMClassificationResult) -> List[Intent]:
        """
        Convert LLM result to legacy Intent objects

        Args:
            result: LLM classification result

        Returns:
            List of Intent objects
        """
        intents = []

        for intent_data in result.intents:
            intent_type_str = intent_data.get('type', 'unknown').lower()

            # Map string to IntentType enum
            try:
                intent_type = IntentType(intent_type_str)
            except ValueError:
                intent_type = IntentType.UNKNOWN

            # Extract entities for this intent
            entities = []
            for entity_data in result.entities:
                entity_type_str = entity_data.get('type', 'unknown').lower()
                try:
                    entity_type = EntityType(entity_type_str)
                except ValueError:
                    entity_type = EntityType.UNKNOWN

                entities.append(Entity(
                    type=entity_type,
                    value=entity_data.get('value', ''),
                    confidence=entity_data.get('confidence', 0.7)
                ))

            intent = Intent(
                type=intent_type,
                confidence=intent_data.get('confidence', 0.5),
                entities=entities,
                raw_indicators=[intent_data.get('reasoning', '')]
            )
            intents.append(intent)

        return intents

    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        total = self.total_classifications
        cache_hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        avg_latency = (self.total_latency_ms / total) if total > 0 else 0

        return {
            'total_classifications': total,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'avg_latency_ms': f"{avg_latency:.1f}",
            'llm_calls': self.llm_calls,
            'target_cache_hit_rate': '70-80%',
            'actual_cache_hit_rate': f"{cache_hit_rate:.1f}%"
        }

    def reset_statistics(self):
        """Reset performance metrics"""
        self.total_classifications = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_latency_ms = 0.0
        self.llm_calls = 0
