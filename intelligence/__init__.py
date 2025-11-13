from .base_types import (
    Intent, IntentType, Entity, EntityType,
    Task, ExecutionPlan, DependencyGraph,
    Confidence, ConfidenceLevel
)

from .pipeline import (
    IntentClassifier,
    EntityExtractor,
    TaskDecomposer,
    ConfidenceScorer
)

from .system import (
    ConversationContextManager,
    IntelligentCache,
    CacheKeyBuilder,
    IntelligenceCoordinator,
    get_global_cache,
    configure_global_cache
)

__all__ = [
    'Intent', 'IntentType', 'Entity', 'EntityType',
    'Task', 'ExecutionPlan', 'DependencyGraph',
    'Confidence', 'ConfidenceLevel',
    'IntentClassifier',
    'EntityExtractor',
    'TaskDecomposer',
    'ConfidenceScorer',
    'ConversationContextManager',
    'IntelligentCache',
    'CacheKeyBuilder',
    'IntelligenceCoordinator',
    'get_global_cache',
    'configure_global_cache',
]

__version__ = '4.0.0'
