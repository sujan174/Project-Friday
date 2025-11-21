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

Enhanced Components (v6.0):
- ReAct Engine: Reasoning-Action-Observation loop
- Critic Model: Self-reflection and evaluation
- LLM Planner: LLM-based task planning
- Uncertainty Quantifier: Calibrated confidence scores
- Tool Reasoner: Reasoned tool selection

Author: AI System
Version: 6.0
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

# Hybrid Intelligence System
from .hybrid_system import (
    HybridIntelligenceSystem,
    HybridIntelligenceResult,
    EnhancedHybridIntelligence,
    EnhancedIntelligenceResult
)

# Enhanced Components (v6.0)
from .react_types import (
    Thought, ThoughtType, Action, ActionType, Observation, ObservationType,
    ReActStep, ReActTrace, ExecutionContext, ExecutionResult,
    Critique, CritiqueType, Reflection, RefinementResult,
    PlanCandidate, PlanStep, PlanScore, PlanStatus,
    CalibratedConfidence, UncertaintyAnalysis, UncertaintySource,
    ToolReasoning, ToolCapability, ToolComposition
)

from .react_engine import ReActEngine
from .critic import CriticModel
from .llm_planner import LLMPlanner
from .uncertainty import UncertaintyQuantifier
from .tool_reasoner import ToolReasoner

__all__ = [
    # Base types
    'Intent', 'IntentType', 'Entity', 'EntityType',
    'Task', 'ExecutionPlan', 'DependencyGraph',
    'Confidence', 'ConfidenceLevel',

    # Original Components
    'IntentClassifier',
    'EntityExtractor',
    'TaskDecomposer',
    'ConfidenceScorer',
    'ConversationContextManager',

    # Hybrid Intelligence System
    'HybridIntelligenceSystem',
    'HybridIntelligenceResult',
    'EnhancedHybridIntelligence',
    'EnhancedIntelligenceResult',

    # ReAct Types
    'Thought', 'ThoughtType', 'Action', 'ActionType',
    'Observation', 'ObservationType',
    'ReActStep', 'ReActTrace', 'ExecutionContext', 'ExecutionResult',

    # Reflection Types
    'Critique', 'CritiqueType', 'Reflection', 'RefinementResult',

    # Planning Types
    'PlanCandidate', 'PlanStep', 'PlanScore', 'PlanStatus',

    # Uncertainty Types
    'CalibratedConfidence', 'UncertaintyAnalysis', 'UncertaintySource',

    # Tool Types
    'ToolReasoning', 'ToolCapability', 'ToolComposition',

    # Enhanced Components
    'ReActEngine',
    'CriticModel',
    'LLMPlanner',
    'UncertaintyQuantifier',
    'ToolReasoner',
]

__version__ = '6.0.0'
