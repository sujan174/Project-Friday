"""
Instruction Memory System

A unified system for storing, retrieving, and injecting user instructions into the orchestrator.
Uses semantic embeddings (RAG) for intelligent retrieval of relevant instructions.

## Architecture

Two types of instructions:
1. **Default Instructions** - ALWAYS injected (timezone, response style, formatting)
2. **Contextual Instructions** - Matched via semantic similarity to current message

## Usage Guidelines for the Orchestrator

### WHEN TO STORE INSTRUCTIONS

Store an instruction when the user:
1. Explicitly says to remember something: "Remember that...", "From now on...", "Always..."
2. Sets a preference: "Use EST timezone", "My default project is X", "I prefer concise responses"
3. Defines a rule: "Never send messages without confirmation", "Always assign tickets to me"
4. Specifies defaults: "Default assignee is John", "Use #general for notifications"

### WHEN NOT TO STORE

Do NOT store:
1. One-time requests: "Send this message to #general" (not "always send to #general")
2. Questions: "What's my timezone?"
3. Actions: "Create a ticket", "List my PRs"
4. Temporary preferences: "For this task, use verbose output"

### HOW TO STORE

Call store_instruction() with:
- content: The instruction text
- category: 'default' (always apply) or 'contextual' (apply when relevant)
- instruction_type: More specific type (timezone, formatting, behavior, etc.)
- context_keywords: For contextual instructions, keywords that trigger this instruction

Author: AI System
Version: 1.0
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np


class InstructionCategory(Enum):
    """Categories for instruction routing"""
    DEFAULT = "default"  # Always injected
    CONTEXTUAL = "contextual"  # Matched via RAG


class InstructionType(Enum):
    """Specific types of instructions"""
    # Default types (always applied)
    TIMEZONE = "timezone"
    RESPONSE_STYLE = "response_style"
    FORMATTING = "formatting"
    COMMUNICATION = "communication"

    # Contextual types (applied when relevant)
    BEHAVIOR = "behavior"
    WORKFLOW = "workflow"
    AGENT_SPECIFIC = "agent_specific"
    PROJECT_SPECIFIC = "project_specific"
    NOTIFICATION = "notification"


@dataclass
class StoredInstruction:
    """A stored user instruction"""
    id: str
    content: str  # The actual instruction
    category: str  # 'default' or 'contextual'
    instruction_type: str  # More specific type
    embedding: Optional[List[float]]  # Vector embedding for semantic search
    context_keywords: List[str]  # Keywords that trigger this instruction
    created_at: float
    updated_at: float
    access_count: int = 0
    is_active: bool = True
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class InstructionMemorySystem:
    """
    Unified instruction memory system with semantic retrieval.

    Features:
    - Stores instructions with embeddings for semantic search
    - Separates defaults (always applied) from contextual (matched per message)
    - Provides formatted output for prompt injection
    - Persistent storage
    - Clear usage guidelines
    """

    # Default instructions that should be set if not present
    RECOMMENDED_DEFAULTS = {
        'timezone': 'Set your preferred timezone (e.g., EST, PST, IST, GMT)',
        'response_style': 'Set preferred response verbosity (concise/detailed)',
        'formatting': 'Set formatting preferences (technical level, use of emojis)',
    }

    def __init__(
        self,
        storage_path: str = None,
        embedding_fn=None,
        verbose: bool = False
    ):
        """
        Initialize instruction memory system.

        Args:
            storage_path: Path to JSON file for persistence
            embedding_fn: Async function to generate embeddings (text -> List[float])
            verbose: Enable verbose logging
        """
        self.storage_path = storage_path
        self.embedding_fn = embedding_fn
        self.verbose = verbose

        # Storage
        self.instructions: Dict[str, StoredInstruction] = {}

        # Statistics
        self.total_stores = 0
        self.total_retrievals = 0
        self.semantic_matches = 0

        # Load from disk
        if storage_path:
            self._load()

    def _generate_id(self, content: str) -> str:
        """Generate unique ID for instruction"""
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    # =========================================================================
    # STORAGE OPERATIONS
    # =========================================================================

    async def store_instruction(
        self,
        content: str,
        category: str = "contextual",
        instruction_type: str = "behavior",
        context_keywords: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Store a user instruction.

        This is the main tool for the orchestrator to use when the user
        provides an instruction to remember.

        Args:
            content: The instruction text (e.g., "Always use EST timezone")
            category: 'default' (always apply) or 'contextual' (apply when relevant)
            instruction_type: Type of instruction (timezone, behavior, workflow, etc.)
            context_keywords: Keywords that trigger this instruction (for contextual)
            metadata: Additional metadata

        Returns:
            Instruction ID

        Example usage:
            # User says: "Remember to always use EST timezone"
            await store_instruction(
                content="Use EST timezone for all time-related operations",
                category="default",
                instruction_type="timezone"
            )

            # User says: "When creating Jira tickets, always assign to me"
            await store_instruction(
                content="Assign all new Jira tickets to current user",
                category="contextual",
                instruction_type="agent_specific",
                context_keywords=["jira", "ticket", "create", "issue"]
            )
        """
        self.total_stores += 1

        # Generate ID
        instruction_id = self._generate_id(content)

        # Check if updating existing
        is_update = instruction_id in self.instructions

        # ===================================================================
        # IMPORTANT: For default instructions, deactivate existing ones of the same type
        # This prevents conflicts like having both "timezone: EST" and "timezone: IST"
        # ===================================================================
        if category == "default":
            deactivated_count = 0
            for inst in self.instructions.values():
                if (inst.is_active and
                    inst.category == "default" and
                    inst.instruction_type == instruction_type and
                    inst.id != instruction_id):
                    inst.is_active = False
                    deactivated_count += 1
                    if self.verbose:
                        print(f"[INSTRUCTION MEMORY] Deactivated old {instruction_type}: {inst.content[:50]}...")

            if deactivated_count > 0 and self.verbose:
                print(f"[INSTRUCTION MEMORY] Replaced {deactivated_count} existing {instruction_type} instruction(s)")

        # Generate embedding for semantic search
        embedding = None
        if self.embedding_fn:
            try:
                # Combine content and keywords for better embedding
                embed_text = content
                if context_keywords:
                    embed_text += " " + " ".join(context_keywords)
                embedding = await self.embedding_fn(embed_text)
            except Exception as e:
                if self.verbose:
                    print(f"[INSTRUCTION MEMORY] Embedding generation failed: {e}")

        # Create instruction
        now = time.time()
        instruction = StoredInstruction(
            id=instruction_id,
            content=content,
            category=category,
            instruction_type=instruction_type,
            embedding=embedding,
            context_keywords=context_keywords or [],
            created_at=self.instructions[instruction_id].created_at if is_update else now,
            updated_at=now,
            access_count=self.instructions[instruction_id].access_count if is_update else 0,
            is_active=True,
            metadata=metadata or {}
        )

        # Store
        self.instructions[instruction_id] = instruction

        # Persist
        if self.storage_path:
            self._save()

        if self.verbose:
            action = "Updated" if is_update else "Stored"
            print(f"[INSTRUCTION MEMORY] {action}: {content[:50]}... (category: {category}, type: {instruction_type})")

        return instruction_id

    def remove_instruction(self, instruction_id: str) -> bool:
        """Remove an instruction by ID"""
        if instruction_id in self.instructions:
            self.instructions[instruction_id].is_active = False
            if self.storage_path:
                self._save()
            return True
        return False

    def clear_instructions(self, category: str = None, instruction_type: str = None):
        """Clear instructions by category and/or type"""
        for instruction in self.instructions.values():
            if category and instruction.category != category:
                continue
            if instruction_type and instruction.instruction_type != instruction_type:
                continue
            instruction.is_active = False

        if self.storage_path:
            self._save()

    # =========================================================================
    # RETRIEVAL OPERATIONS
    # =========================================================================

    def get_default_instructions(self) -> List[StoredInstruction]:
        """
        Get all default instructions (always applied).

        Returns:
            List of default instructions
        """
        return [
            inst for inst in self.instructions.values()
            if inst.is_active and inst.category == InstructionCategory.DEFAULT.value
        ]

    async def get_relevant_instructions(
        self,
        message: str,
        top_k: int = 5,
        threshold: float = 0.4
    ) -> List[Tuple[StoredInstruction, float]]:
        """
        Get contextual instructions relevant to the current message.

        Uses semantic similarity to match instructions to the message context.

        Args:
            message: Current user message
            top_k: Maximum number of instructions to return
            threshold: Minimum similarity threshold

        Returns:
            List of (instruction, similarity_score) tuples
        """
        self.total_retrievals += 1

        # Get all active contextual instructions
        contextual = [
            inst for inst in self.instructions.values()
            if inst.is_active and inst.category == InstructionCategory.CONTEXTUAL.value
        ]

        if not contextual:
            return []

        # First, check keyword matches (fast path)
        keyword_matches = []
        message_lower = message.lower()
        for inst in contextual:
            if inst.context_keywords:
                keyword_score = sum(1 for kw in inst.context_keywords if kw.lower() in message_lower)
                if keyword_score > 0:
                    # Normalize score
                    score = min(1.0, keyword_score / len(inst.context_keywords))
                    keyword_matches.append((inst, score))

        # If we have keyword matches and no embedding function, return those
        if keyword_matches and not self.embedding_fn:
            keyword_matches.sort(key=lambda x: x[1], reverse=True)
            return keyword_matches[:top_k]

        # Semantic search with embeddings
        if self.embedding_fn:
            try:
                query_embedding = await self.embedding_fn(message)

                results = []
                for inst in contextual:
                    if inst.embedding:
                        similarity = self._cosine_similarity(query_embedding, inst.embedding)

                        # Boost score if keywords also match
                        if inst.context_keywords:
                            keyword_boost = sum(0.1 for kw in inst.context_keywords if kw.lower() in message_lower)
                            similarity = min(1.0, similarity + keyword_boost)

                        if similarity >= threshold:
                            results.append((inst, similarity))
                            inst.access_count += 1
                            self.semantic_matches += 1

                results.sort(key=lambda x: x[1], reverse=True)
                return results[:top_k]

            except Exception as e:
                if self.verbose:
                    print(f"[INSTRUCTION MEMORY] Semantic search failed: {e}")

        # Fallback to keyword matches
        keyword_matches.sort(key=lambda x: x[1], reverse=True)
        return keyword_matches[:top_k]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        a = np.array(vec1)
        b = np.array(vec2)

        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot / (norm_a * norm_b))

    # =========================================================================
    # PROMPT FORMATTING
    # =========================================================================

    async def get_instructions_for_message(self, message: str) -> str:
        """
        Get formatted instructions to inject into the prompt for this message.

        This is the main method to call on each user message.
        It returns:
        1. All default instructions (always)
        2. Relevant contextual instructions (matched via RAG)

        Args:
            message: Current user message

        Returns:
            Formatted string for prompt injection
        """
        # Get defaults
        defaults = self.get_default_instructions()

        # Get relevant contextual instructions
        relevant = await self.get_relevant_instructions(message)

        if not defaults and not relevant:
            return ""

        lines = [
            "# User Instructions & Preferences",
            "",
            "Apply these instructions to your response:",
            ""
        ]

        # Format defaults
        if defaults:
            lines.append("## Always Apply")
            for inst in defaults:
                lines.append(f"- **{inst.instruction_type}**: {inst.content}")
            lines.append("")

        # Format contextual
        if relevant:
            lines.append("## Relevant to This Request")
            for inst, score in relevant:
                lines.append(f"- {inst.content}")
            lines.append("")

        return "\n".join(lines)

    def format_defaults_only(self) -> str:
        """
        Format only default instructions (for system prompt).

        Returns:
            Formatted string of default instructions
        """
        defaults = self.get_default_instructions()

        if not defaults:
            return ""

        lines = [
            "# User Default Settings",
            "",
            "These preferences MUST be applied to ALL responses:",
            ""
        ]

        # Group by type
        by_type: Dict[str, List[str]] = {}
        for inst in defaults:
            if inst.instruction_type not in by_type:
                by_type[inst.instruction_type] = []
            by_type[inst.instruction_type].append(inst.content)

        type_labels = {
            'timezone': 'Timezone',
            'response_style': 'Response Style',
            'formatting': 'Formatting',
            'communication': 'Communication',
        }

        for inst_type, contents in by_type.items():
            label = type_labels.get(inst_type, inst_type.title())
            lines.append(f"**{label}:**")
            for content in contents:
                lines.append(f"- {content}")
            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_all_instructions(self, active_only: bool = True) -> List[StoredInstruction]:
        """Get all instructions"""
        instructions = list(self.instructions.values())
        if active_only:
            instructions = [inst for inst in instructions if inst.is_active]
        return sorted(instructions, key=lambda x: x.updated_at, reverse=True)

    def get_instruction(self, instruction_id: str) -> Optional[StoredInstruction]:
        """Get instruction by ID"""
        inst = self.instructions.get(instruction_id)
        if inst and inst.is_active:
            return inst
        return None

    def get_by_type(self, instruction_type: str) -> List[StoredInstruction]:
        """Get instructions by type"""
        return [
            inst for inst in self.instructions.values()
            if inst.is_active and inst.instruction_type == instruction_type
        ]

    def has_default(self, instruction_type: str) -> bool:
        """Check if a default instruction of this type exists"""
        return any(
            inst.is_active and
            inst.category == InstructionCategory.DEFAULT.value and
            inst.instruction_type == instruction_type
            for inst in self.instructions.values()
        )

    def get_missing_defaults(self) -> List[str]:
        """Get list of recommended defaults that are not set"""
        missing = []
        for default_type in ['timezone', 'response_style', 'formatting']:
            if not self.has_default(default_type):
                missing.append(default_type)
        return missing

    def deduplicate_defaults(self) -> int:
        """
        Clean up duplicate default instructions, keeping only the most recent per type.

        This fixes conflicts like having both "timezone: EST" and "timezone: IST".
        Only the most recently updated instruction of each type is kept.

        Returns:
            Number of duplicates removed
        """
        # Group defaults by type
        defaults_by_type: Dict[str, List[StoredInstruction]] = {}
        for inst in self.instructions.values():
            if inst.is_active and inst.category == InstructionCategory.DEFAULT.value:
                if inst.instruction_type not in defaults_by_type:
                    defaults_by_type[inst.instruction_type] = []
                defaults_by_type[inst.instruction_type].append(inst)

        # Deactivate all but the most recent per type
        deactivated = 0
        for inst_type, instructions in defaults_by_type.items():
            if len(instructions) > 1:
                # Sort by updated_at descending (most recent first)
                instructions.sort(key=lambda x: x.updated_at, reverse=True)
                # Keep the first (most recent), deactivate the rest
                for inst in instructions[1:]:
                    inst.is_active = False
                    deactivated += 1
                    if self.verbose:
                        print(f"[INSTRUCTION MEMORY] Deduped: deactivated old {inst_type}: {inst.content[:50]}...")

        if deactivated > 0:
            if self.storage_path:
                self._save()
            if self.verbose:
                print(f"[INSTRUCTION MEMORY] Removed {deactivated} duplicate default instruction(s)")

        return deactivated

    def get_statistics(self) -> Dict:
        """Get system statistics"""
        active = [inst for inst in self.instructions.values() if inst.is_active]
        defaults = [inst for inst in active if inst.category == InstructionCategory.DEFAULT.value]
        contextual = [inst for inst in active if inst.category == InstructionCategory.CONTEXTUAL.value]

        return {
            'total_instructions': len(active),
            'default_instructions': len(defaults),
            'contextual_instructions': len(contextual),
            'total_stores': self.total_stores,
            'total_retrievals': self.total_retrievals,
            'semantic_matches': self.semantic_matches,
            'has_embeddings': any(inst.embedding for inst in active),
            'missing_defaults': self.get_missing_defaults()
        }

    def get_summary(self) -> str:
        """Get human-readable summary"""
        stats = self.get_statistics()

        lines = [
            "## Instruction Memory Summary",
            "",
            f"**Total Instructions:** {stats['total_instructions']}",
            f"- Defaults (always applied): {stats['default_instructions']}",
            f"- Contextual (matched per message): {stats['contextual_instructions']}",
            "",
        ]

        # Show defaults
        defaults = self.get_default_instructions()
        if defaults:
            lines.append("**Default Instructions:**")
            for inst in defaults:
                lines.append(f"- {inst.instruction_type}: {inst.content}")
            lines.append("")

        # Show missing defaults
        if stats['missing_defaults']:
            lines.append("**Recommended to Set:**")
            for missing in stats['missing_defaults']:
                lines.append(f"- {missing}: {self.RECOMMENDED_DEFAULTS.get(missing, 'Not set')}")
            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def _save(self):
        """Save instructions to disk"""
        try:
            Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)

            data = {
                'instructions': {
                    k: {
                        'id': v.id,
                        'content': v.content,
                        'category': v.category,
                        'instruction_type': v.instruction_type,
                        'embedding': v.embedding,
                        'context_keywords': v.context_keywords,
                        'created_at': v.created_at,
                        'updated_at': v.updated_at,
                        'access_count': v.access_count,
                        'is_active': v.is_active,
                        'metadata': v.metadata
                    }
                    for k, v in self.instructions.items()
                },
                'statistics': {
                    'total_stores': self.total_stores,
                    'total_retrievals': self.total_retrievals,
                    'semantic_matches': self.semantic_matches
                },
                'saved_at': time.time()
            }

            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            if self.verbose:
                print(f"[INSTRUCTION MEMORY] Save error: {e}")

    def _load(self):
        """Load instructions from disk"""
        try:
            if not Path(self.storage_path).exists():
                return

            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            for instruction_id, inst_data in data.get('instructions', {}).items():
                self.instructions[instruction_id] = StoredInstruction(
                    id=inst_data['id'],
                    content=inst_data['content'],
                    category=inst_data['category'],
                    instruction_type=inst_data['instruction_type'],
                    embedding=inst_data.get('embedding'),
                    context_keywords=inst_data.get('context_keywords', []),
                    created_at=inst_data['created_at'],
                    updated_at=inst_data['updated_at'],
                    access_count=inst_data.get('access_count', 0),
                    is_active=inst_data.get('is_active', True),
                    metadata=inst_data.get('metadata', {})
                )

            # Restore statistics
            stats = data.get('statistics', {})
            self.total_stores = stats.get('total_stores', 0)
            self.total_retrievals = stats.get('total_retrievals', 0)
            self.semantic_matches = stats.get('semantic_matches', 0)

            if self.verbose:
                active = sum(1 for inst in self.instructions.values() if inst.is_active)
                print(f"[INSTRUCTION MEMORY] Loaded {active} instructions")

        except Exception as e:
            if self.verbose:
                print(f"[INSTRUCTION MEMORY] Load error: {e}")


# =============================================================================
# USAGE GUIDELINES FOR THE ORCHESTRATOR
# =============================================================================

INSTRUCTION_STORAGE_GUIDELINES = """
## When to Store Instructions

### STORE when the user:
1. **Explicitly asks to remember**:
   - "Remember that I prefer..."
   - "From now on, always..."
   - "Keep in mind that..."

2. **Sets preferences/defaults**:
   - "Use EST timezone"
   - "My default project is KAN"
   - "Default assignee should be John"

3. **Defines behavioral rules**:
   - "Always confirm before sending"
   - "Never delete without asking"
   - "Automatically assign to me"

4. **Specifies style preferences**:
   - "Be concise in your responses"
   - "Use technical language"
   - "Include emojis"

### DO NOT STORE when the user:
1. Makes a one-time request (no "always", "from now on", etc.)
2. Asks a question
3. Requests an action (create, send, delete)
4. Specifies something for "just this time"

## How to Categorize

### Default (category='default') - Always injected:
- Timezone settings
- Response style (verbose/concise)
- Formatting preferences
- Communication style

### Contextual (category='contextual') - Matched per message:
- Agent-specific rules (Jira, Slack, GitHub)
- Project-specific instructions
- Workflow automation rules
- Conditional behaviors

## Example Calls

```python
# Default: timezone
await instruction_memory.store_instruction(
    content="Use IST (Indian Standard Time) for all timestamps",
    category="default",
    instruction_type="timezone"
)

# Default: response style
await instruction_memory.store_instruction(
    content="Provide concise responses without unnecessary detail",
    category="default",
    instruction_type="response_style"
)

# Contextual: Jira-specific
await instruction_memory.store_instruction(
    content="Always assign new Jira tickets to me unless specified otherwise",
    category="contextual",
    instruction_type="agent_specific",
    context_keywords=["jira", "ticket", "issue", "create", "assign"]
)

# Contextual: GitHub-specific
await instruction_memory.store_instruction(
    content="Add 'needs-review' label to all new PRs",
    category="contextual",
    instruction_type="agent_specific",
    context_keywords=["github", "pr", "pull request", "create"]
)

# Contextual: Project-specific
await instruction_memory.store_instruction(
    content="For project KAN, always use high priority",
    category="contextual",
    instruction_type="project_specific",
    context_keywords=["KAN", "project", "priority"]
)
```
"""
