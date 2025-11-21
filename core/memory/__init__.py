"""
Memory System - Unified Memory Management for Project Friday.

This package provides a comprehensive memory system with three types:
- Episodic Memory: What happened (conversations, actions, outcomes)
- Semantic Memory: What we know (facts, preferences, knowledge)
- Procedural Memory: How to do things (patterns, workflows, habits)

Usage:
    from core.memory import MemoryManager

    memory = MemoryManager(storage_dir="memory")

    # Remember preferences
    memory.remember_preference("timezone", "America/New_York")

    # Remember instructions
    memory.remember_instruction("Always confirm before sending to public channels")

    # Get context for prompts
    context = memory.get_context("Create a Jira issue")
    context_text = context.to_prompt_text()

    # Record actions
    memory.record_action(
        action_type="create_issue",
        agent="jira",
        input_summary="Create bug for login failure",
        output_summary="Created PROJ-123",
        success=True
    )

    # Process memory commands
    response = memory.process_memory_command("Remember that I prefer UTC")
"""

from .memory_types import (
    MemoryType,
    ImportanceLevel,
    MemorySource,
    ActionOutcome,
    FeedbackType,
    Memory,
    UserPreference,
    Instruction,
    Fact,
    EntityKnowledge,
    ConversationSummary,
    ActionRecord,
    UserCorrection,
    BehavioralPattern,
    Workflow,
    ToolUsagePattern,
    MemoryQuery,
    MemoryContext
)

from .semantic_memory import SemanticMemory
from .episodic_memory import EpisodicMemory
from .procedural_memory import ProceduralMemory
from .memory_retrieval import MemoryRetrieval
from .memory_consolidation import MemoryConsolidation
from .memory_manager import MemoryManager, create_memory_manager

__all__ = [
    # Main interface
    "MemoryManager",
    "create_memory_manager",

    # Memory subsystems
    "SemanticMemory",
    "EpisodicMemory",
    "ProceduralMemory",
    "MemoryRetrieval",
    "MemoryConsolidation",

    # Types
    "MemoryType",
    "ImportanceLevel",
    "MemorySource",
    "ActionOutcome",
    "FeedbackType",
    "Memory",
    "UserPreference",
    "Instruction",
    "Fact",
    "EntityKnowledge",
    "ConversationSummary",
    "ActionRecord",
    "UserCorrection",
    "BehavioralPattern",
    "Workflow",
    "ToolUsagePattern",
    "MemoryQuery",
    "MemoryContext"
]
