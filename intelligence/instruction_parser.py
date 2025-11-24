"""
Intelligent Instruction Parser

Uses LLM to semantically understand and extract user instructions/preferences
from natural language. Much more robust than regex patterns.

Author: AI System
Version: 1.0
"""

import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict


@dataclass
class ParsedInstruction:
    """Represents a parsed instruction from user input"""
    is_instruction: bool  # Whether this message contains an instruction
    category: str  # timezone, default, behavior, formatting, notification, etc.
    key: str  # The setting key (e.g., "timezone", "default_project")
    value: str  # The value to set
    original_message: str  # Original user message
    confidence: float  # How confident we are this is an instruction
    reasoning: str  # Why we think this is/isn't an instruction


class IntelligentInstructionParser:
    """
    LLM-powered instruction parser that understands natural language instructions.

    Uses semantic understanding instead of rigid regex patterns.
    Maintains a cache for efficiency.
    """

    # Categories of instructions we support
    # Note: 'identity' and 'timezone' are now handled by orchestrator's update_user_fact tool
    INSTRUCTION_CATEGORIES = [
        'default',       # Default values (project, assignee, channel)
        'behavior',      # Behavioral rules (always/never do X)
        'formatting',    # Output formatting preferences
        'notification',  # Notification preferences
        'style',         # Communication style
        'workflow',      # Workflow automation preferences
    ]

    # Known instruction keys for validation
    KNOWN_KEYS = {
        'default': ['default_project', 'default_assignee', 'default_channel', 'default_priority', 'default_repository'],
        'behavior': ['always', 'never', 'confirm_before', 'auto_approve'],
        'formatting': ['verbosity', 'format_style', 'use_emojis', 'language'],
        'notification': ['notification_channel', 'notify_on', 'quiet_hours'],
        'style': ['tone', 'detail_level', 'technical_level'],
        'workflow': ['auto_assign', 'auto_label', 'default_workflow'],
    }

    def __init__(self, llm_client=None, verbose: bool = False):
        """
        Initialize the instruction parser.

        Args:
            llm_client: LLM client for semantic parsing (optional, will use simple heuristics if None)
            verbose: Enable verbose logging
        """
        self.llm = llm_client
        self.verbose = verbose

        # Simple cache for parsed instructions (avoid re-parsing similar messages)
        self._cache: Dict[str, Any] = {}

        # Statistics
        self.total_parses = 0
        self.instructions_found = 0
        self.cache_hits = 0
        self.fast_path_skips = 0  # Skipped LLM via pre-filter
        self.llm_calls = 0  # Actual LLM calls made

    async def parse(self, message: str) -> ParsedInstruction:
        """
        Parse a message to detect and extract any instruction.

        Uses cheap keyword pre-filter first, only calls LLM if needed.
        This is the Google/top-company approach: minimize LLM calls.

        Args:
            message: User message to parse

        Returns:
            ParsedInstruction with detection results
        """
        start_time = time.time()
        self.total_parses += 1

        # Check cache first (O(1) lookup)
        cache_key = f"instruction:{hash(message.lower().strip())}"
        cached = self._cache.get(cache_key)
        if cached:
            self.cache_hits += 1
            return cached

        # STEP 1: Cheap keyword pre-filter (< 1ms)
        # Skip LLM entirely for 90%+ of messages
        if not self._should_parse_with_llm(message):
            self.fast_path_skips += 1
            result = ParsedInstruction(
                is_instruction=False,
                category='',
                key='',
                value='',
                original_message=message,
                confidence=0.95,
                reasoning="No instruction keywords detected (fast path)"
            )
            self._cache[cache_key] = result
            return result

        # STEP 2: Only call LLM for likely instructions (~5-10% of messages)
        self.llm_calls += 1
        if self.llm:
            result = await self._parse_with_llm(message)
        else:
            result = self._parse_with_heuristics(message)

        # Update statistics
        if result.is_instruction:
            self.instructions_found += 1

        # Cache the result
        self._cache[cache_key] = result

        if self.verbose:
            latency = (time.time() - start_time) * 1000
            status = "✓ INSTRUCTION" if result.is_instruction else "✗ Not instruction"
            print(f"[INSTRUCTION PARSER] {status} ({latency:.1f}ms)")
            if result.is_instruction:
                print(f"  Category: {result.category}, Key: {result.key}, Value: {result.value}")

        return result

    def _should_parse_with_llm(self, message: str) -> bool:
        """
        Cheap pre-filter to determine if LLM parsing is needed.

        Returns True only for messages likely to be instructions.
        This saves 90%+ of LLM calls.

        Performance: < 1ms (just string operations)
        """
        message_lower = message.lower()

        # Skip patterns handled by update_user_fact tool - don't waste LLM calls
        # These include: identity, timezone, defaults
        skip_patterns = ['default project', 'default assignee', 'my name is', 'i am ',
                        'my email', 'my mail', 'timezone', 'time zone',
                        'use est', 'use pst', 'use ist', 'use utc', 'use gmt']
        if any(pattern in message_lower for pattern in skip_patterns):
            return False

        # Strong instruction indicators - behavioral/workflow patterns only
        # Note: Identity, timezone, and defaults are handled by orchestrator's update_user_fact tool
        strong_indicators = [
            'from now on', 'from now', 'remember that', 'always use', 'never use',
            'my preference', 'always confirm', 'never send', 'be concise', 'be verbose',
        ]

        if any(indicator in message_lower for indicator in strong_indicators):
            return True

        # Weak indicators - need multiple to trigger
        weak_indicators = ['remember', 'always', 'never', 'prefer']

        # Behavioral/formatting words (not identity/defaults)
        behavior_words = ['verbose', 'concise', 'format', 'style', 'confirm', 'notify']

        weak_count = sum(1 for w in weak_indicators if w in message_lower)
        behavior_count = sum(1 for w in behavior_words if w in message_lower)

        # Need at least one weak indicator AND one behavior word
        if weak_count >= 1 and behavior_count >= 1:
            return True

        # Quick exclusions - definitely not instructions
        if message_lower.startswith(('show ', 'list ', 'get ', 'find ', 'search ',
                                     'create ', 'delete ', 'update ', 'what ', 'how ')):
            return False

        if '?' in message:
            return False

        return False

    async def _parse_with_llm(self, message: str) -> ParsedInstruction:
        """Parse instruction using LLM for semantic understanding"""

        prompt = f"""Analyze this message. Is the user giving a permanent instruction/preference to remember?

Message: "{message}"

INSTRUCTIONS (remember permanently):
- Defaults: "default project is X", "always assign to Y"
- Rules: "always X", "never Y", "confirm before deleting"
- Preferences: "be concise", "use emojis", "verbose responses"
- Workflow: "auto-approve PRs", "notify me on failures"

NOT INSTRUCTIONS (don't remember):
- Questions: "what time?", "show me X"
- Actions: "create issue", "send message"
- One-time requests
- Identity info (name, email, timezone) - these are handled separately

JSON response only:
{{"is_instruction":true/false,"category":"default|behavior|formatting|workflow","key":"setting_name","value":"setting_value","confidence":0.0-1.0,"reasoning":"brief"}}"""

        try:
            # Use the LLM to parse
            response = await self.llm.generate(prompt)

            # Extract JSON from response
            response_text = response.text if hasattr(response, 'text') else str(response)

            # Find JSON in response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                parsed = json.loads(json_str)

                return ParsedInstruction(
                    is_instruction=parsed.get('is_instruction', False),
                    category=parsed.get('category', ''),
                    key=parsed.get('key', ''),
                    value=parsed.get('value', ''),
                    original_message=message,
                    confidence=parsed.get('confidence', 0.5),
                    reasoning=parsed.get('reasoning', '')
                )
        except Exception as e:
            if self.verbose:
                print(f"[INSTRUCTION PARSER] LLM parse error: {e}")

        # Fallback to heuristics if LLM fails
        return self._parse_with_heuristics(message)

    def _parse_with_heuristics(self, message: str) -> ParsedInstruction:
        """
        Minimal fallback when LLM is unavailable.
        CONSERVATIVE: Default to NOT storing as instruction when uncertain.
        Only handles the most obvious patterns - LLM should be primary.
        """
        import re
        message_lower = message.lower().strip()

        # Strong instruction indicators that are unambiguous
        # Note: Identity (name, email), timezone, and defaults are handled by orchestrator's update_user_fact tool
        # Only behavioral/workflow patterns should be handled here
        behavioral_patterns = [
            'from now on', 'remember that', 'always use', 'never use',
            'always confirm', 'never send', 'be concise', 'be verbose',
        ]

        has_behavioral_indicator = any(pattern in message_lower for pattern in behavioral_patterns)

        # Exclude if it's clearly a question or action
        is_question = '?' in message
        is_action = any(message_lower.startswith(p) for p in
                       ['check ', 'create ', 'send ', 'get ', 'show ', 'list ', 'schedule '])

        if is_action or is_question:
            return ParsedInstruction(
                is_instruction=False,
                category='',
                key='',
                value='',
                original_message=message,
                confidence=0.9,
                reasoning="Detected as action/question, not instruction (heuristic)"
            )

        # Skip patterns handled by update_user_fact tool
        # These include: default project, default assignee, timezone, name, email
        skip_patterns = ['default project', 'default assignee', 'my name is', 'my email',
                        'timezone', 'time zone', 'use est', 'use pst', 'use ist', 'use utc']
        if any(pattern in message_lower for pattern in skip_patterns):
            return ParsedInstruction(
                is_instruction=False,
                category='',
                key='',
                value='',
                original_message=message,
                confidence=0.8,
                reasoning="Handled by update_user_fact tool, not instruction parser (heuristic)"
            )

        # Only store behavioral/workflow instructions
        if has_behavioral_indicator:
            # Try to extract a meaningful key
            key = 'behavior'
            category = 'behavior'

            if 'concise' in message_lower:
                key = 'response_style'
                category = 'formatting'
            elif 'verbose' in message_lower:
                key = 'response_style'
                category = 'formatting'
            elif 'confirm' in message_lower:
                key = 'confirm_before_action'
                category = 'behavior'

            return ParsedInstruction(
                is_instruction=True,
                category=category,
                key=key,
                value=message,
                original_message=message,
                confidence=0.6,
                reasoning="Detected behavioral instruction pattern (heuristic)"
            )

        # DEFAULT: Be conservative - don't store as instruction when uncertain
        return ParsedInstruction(
            is_instruction=False,
            category='',
            key='',
            value='',
            original_message=message,
            confidence=0.7,
            reasoning="Uncertain - defaulting to not storing (heuristic fallback)"
        )

    def get_statistics(self) -> Dict:
        """Get parser statistics"""
        total = max(self.total_parses, 1)
        return {
            'total_parses': self.total_parses,
            'instructions_found': self.instructions_found,
            'cache_hits': self.cache_hits,
            'fast_path_skips': self.fast_path_skips,
            'llm_calls': self.llm_calls,
            'cache_hit_rate': f"{(self.cache_hits / total * 100):.1f}%",
            'fast_path_rate': f"{(self.fast_path_skips / total * 100):.1f}%",
            'llm_call_rate': f"{(self.llm_calls / total * 100):.1f}%",
            'instruction_rate': f"{(self.instructions_found / total * 100):.1f}%"
        }


class InstructionMemory:
    """
    Maintains an up-to-date record of all user instructions.

    Features:
    - Persistent storage
    - Conflict resolution (newer instructions override older)
    - Category-based organization
    - History tracking
    """

    def __init__(self, storage_path: str = None, verbose: bool = False):
        """
        Initialize instruction memory.

        Args:
            storage_path: Path to JSON file for persistence
            verbose: Enable verbose logging
        """
        self.storage_path = storage_path
        self.verbose = verbose

        # Current active instructions (key -> instruction)
        self.instructions: Dict[str, Dict[str, Any]] = {}

        # History of all instructions (for audit/undo)
        self.history: List[Dict[str, Any]] = []

        # Load from storage if exists
        if storage_path:
            self._load()

    def add(self, instruction: ParsedInstruction) -> bool:
        """
        Add or update an instruction.

        Args:
            instruction: Parsed instruction to add

        Returns:
            True if added/updated, False if invalid
        """
        if not instruction.is_instruction or not instruction.key:
            return False

        # Normalize value for certain categories
        value = instruction.value
        if instruction.category == 'timezone' and value:
            value = value.upper()

        # Create instruction record
        record = {
            'category': instruction.category,
            'key': instruction.key,
            'value': value,
            'original_message': instruction.original_message,
            'confidence': instruction.confidence,
            'created_at': time.time(),
            'is_active': True
        }

        # Check if updating existing
        was_update = instruction.key in self.instructions

        # Store/update instruction
        self.instructions[instruction.key] = record

        # Add to history
        self.history.append({
            **record,
            'action': 'update' if was_update else 'add'
        })

        # Persist
        if self.storage_path:
            self._save()

        if self.verbose:
            action = "Updated" if was_update else "Added"
            print(f"[INSTRUCTION MEMORY] {action}: {instruction.key} = {instruction.value}")

        return True

    def get(self, key: str) -> Optional[str]:
        """Get the value for an instruction key"""
        if key in self.instructions and self.instructions[key]['is_active']:
            return self.instructions[key]['value']
        return None

    def get_all(self, category: str = None) -> Dict[str, str]:
        """
        Get all active instructions.

        Args:
            category: Optional filter by category

        Returns:
            Dict of key -> value for active instructions
        """
        result = {}
        for key, record in self.instructions.items():
            if record['is_active']:
                if category is None or record['category'] == category:
                    result[key] = record['value']
        return result

    def remove(self, key: str) -> bool:
        """Remove an instruction by key"""
        if key in self.instructions:
            self.instructions[key]['is_active'] = False
            self.history.append({
                'key': key,
                'action': 'remove',
                'created_at': time.time()
            })
            if self.storage_path:
                self._save()
            return True
        return False

    def format_for_prompt(self) -> str:
        """Format active instructions for system prompt injection"""
        active = self.get_all()
        if not active:
            return ""

        # Group by category
        by_category: Dict[str, List[tuple]] = {}
        for key, value in active.items():
            if key in self.instructions:
                cat = self.instructions[key]['category']
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append((key, value))

        lines = [
            "# User's Explicit Instructions",
            "",
            "The following preferences MUST be applied:"
        ]

        category_labels = {
            'timezone': 'Time & Timezone',
            'default': 'Default Values',
            'behavior': 'Behavior Rules',
            'formatting': 'Formatting',
            'notification': 'Notifications',
            'style': 'Communication Style',
            'workflow': 'Workflow'
        }

        for cat, items in by_category.items():
            label = category_labels.get(cat, cat.title())
            lines.append(f"\n**{label}:**")
            for key, value in items:
                lines.append(f"- {key}: {value}")

        return "\n".join(lines)

    def _save(self):
        """Save instructions to storage"""
        try:
            import json
            from pathlib import Path

            Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)

            data = {
                'instructions': self.instructions,
                'history': self.history[-100:],  # Keep last 100 history items
                'saved_at': time.time()
            }

            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            if self.verbose:
                print(f"[INSTRUCTION MEMORY] Save error: {e}")

    def _load(self):
        """Load instructions from storage"""
        try:
            import json
            from pathlib import Path

            if not Path(self.storage_path).exists():
                return

            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            self.instructions = data.get('instructions', {})
            self.history = data.get('history', [])

            if self.verbose:
                active_count = sum(1 for i in self.instructions.values() if i['is_active'])
                print(f"[INSTRUCTION MEMORY] Loaded {active_count} active instructions")

        except Exception as e:
            if self.verbose:
                print(f"[INSTRUCTION MEMORY] Load error: {e}")

    def get_summary(self) -> str:
        """Get human-readable summary of instructions"""
        active = self.get_all()
        if not active:
            return "No active instructions."

        lines = [f"**{len(active)} Active Instructions:**"]
        for key, value in active.items():
            lines.append(f"  • {key}: {value}")

        return "\n".join(lines)
