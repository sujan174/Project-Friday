"""
Intelligence Module - Advanced AI Orchestration Intelligence

This module provides sophisticated intelligence components that transform
the orchestrator from a simple delegation system into an intelligent,
adaptive, learning coordination engine.

Components:
- Intent Classification: Understand what users really want
- Entity Extraction: Extract structured information from natural language
- Task Decomposition: Break complex tasks into optimal execution plans
- Confidence Scoring: Make decisions with awareness of certainty
- Context Management: Deep conversation and workspace understanding
- Agent Selection: Intelligently match tasks to optimal agents
- Error Intelligence: Predict and prevent errors
- Learning & Adaptation: Improve from experience
- Optimization: Cost and efficiency intelligence

Author: AI System
Version: 2.0
"""

from .base_types import (
    Intent, IntentType, Entity, EntityType,
    Task, ExecutionPlan, DependencyGraph,
    Confidence, ConfidenceLevel
)

from .intent_classifier import IntentClassifier
from .entity_extractor import EntityExtractor
from .task_decomposer import TaskDecomposer
from .confidence_scorer import ConfidenceScorer
from .context_manager import ConversationContextManager

__all__ = [
    # Base types
    'Intent', 'IntentType', 'Entity', 'EntityType',
    'Task', 'ExecutionPlan', 'DependencyGraph',
    'Confidence', 'ConfidenceLevel',

    # Components
    'IntentClassifier',
    'EntityExtractor',
    'TaskDecomposer',
    'ConfidenceScorer',
    'ConversationContextManager',
]

__version__ = '2.0.0'
