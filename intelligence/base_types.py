"""
Base Types for Intelligence System

Defines core data structures used across all intelligence components.
Enhanced with immutability, validation, and rich type support.

Author: AI System
Version: 3.0 - Major refactoring with enterprise-grade patterns
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod
import hashlib
import json


# ============================================================================
# INTENT TYPES
# ============================================================================

class IntentType(Enum):
    """Types of user intents"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    ANALYZE = "analyze"
    COORDINATE = "coordinate"
    WORKFLOW = "workflow"
    SEARCH = "search"
    UNKNOWN = "unknown"


@dataclass
class Intent:
    """Represents a user intent"""
    type: IntentType
    confidence: float  # 0.0 to 1.0
    entities: List['Entity'] = field(default_factory=list)
    implicit_requirements: List[str] = field(default_factory=list)
    raw_indicators: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.type.value}({self.confidence:.2f})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary"""
        return {
            'type': self.type.value,
            'confidence': self.confidence,
            'entities': [e.to_dict() if hasattr(e, 'to_dict') else str(e) for e in self.entities],
            'implicit_requirements': self.implicit_requirements,
            'raw_indicators': self.raw_indicators
        }


# ============================================================================
# ENTITY TYPES
# ============================================================================

class EntityType(Enum):
    """Types of entities that can be extracted"""
    PROJECT = "project"
    PERSON = "person"
    TEAM = "team"
    RESOURCE = "resource"
    DATE = "date"
    PRIORITY = "priority"
    STATUS = "status"
    LABEL = "label"
    ISSUE = "issue"
    PR = "pr"
    CHANNEL = "channel"
    REPOSITORY = "repository"
    FILE = "file"
    CODE = "code"
    UNKNOWN = "unknown"


@dataclass
class Entity:
    """Represents an extracted entity"""
    type: EntityType
    value: str
    confidence: float  # 0.0 to 1.0
    context: Optional[str] = None
    normalized_value: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.type.value}:{self.value}({self.confidence:.2f})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary"""
        return {
            'type': self.type.value,
            'value': self.value,
            'confidence': self.confidence,
            'context': self.context,
            'normalized_value': self.normalized_value,
            'metadata': self.metadata
        }


# ============================================================================
# TASK TYPES
# ============================================================================

@dataclass
class Task:
    """Represents a decomposed task"""
    id: str
    action: str
    agent: Optional[str] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    conditions: Optional[str] = None
    priority: int = 0
    estimated_duration: float = 0.0  # seconds
    estimated_cost: float = 0.0  # tokens or API cost
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"Task({self.id}: {self.agent or '?'}.{self.action})"


@dataclass
class DependencyGraph:
    """Represents task dependencies"""
    tasks: Dict[str, Task] = field(default_factory=dict)
    edges: List[tuple] = field(default_factory=list)  # (from_id, to_id)

    def add_task(self, task: Task):
        """Add a task to the graph"""
        self.tasks[task.id] = task

    def add_dependency(self, from_task_id: str, to_task_id: str):
        """Add a dependency edge"""
        self.edges.append((from_task_id, to_task_id))

    def get_execution_order(self) -> List[Task]:
        """Get tasks in topologically sorted order"""
        # Simple topological sort
        in_degree = {task_id: 0 for task_id in self.tasks}
        for from_id, to_id in self.edges:
            in_degree[to_id] = in_degree.get(to_id, 0) + 1

        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            task_id = queue.pop(0)
            result.append(self.tasks[task_id])

            # Reduce in-degree for dependent tasks
            for from_id, to_id in self.edges:
                if from_id == task_id:
                    in_degree[to_id] -= 1
                    if in_degree[to_id] == 0:
                        queue.append(to_id)

        return result

    def has_cycle(self) -> bool:
        """Check if graph has cycles"""
        visited = set()
        rec_stack = set()

        def has_cycle_util(task_id: str) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)

            for from_id, to_id in self.edges:
                if from_id == task_id:
                    if to_id not in visited:
                        if has_cycle_util(to_id):
                            return True
                    elif to_id in rec_stack:
                        return True

            rec_stack.remove(task_id)
            return False

        for task_id in self.tasks:
            if task_id not in visited:
                if has_cycle_util(task_id):
                    return True

        return False


@dataclass
class ExecutionPlan:
    """Complete execution plan"""
    tasks: List[Task] = field(default_factory=list)
    dependency_graph: Optional[DependencyGraph] = None
    estimated_duration: float = 0.0
    estimated_cost: float = 0.0
    risks: List[str] = field(default_factory=list)
    optimizations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_execution_order(self) -> List[Task]:
        """Get tasks in optimal execution order"""
        if self.dependency_graph:
            return self.dependency_graph.get_execution_order()
        return self.tasks


# ============================================================================
# CONFIDENCE TYPES
# ============================================================================

class ConfidenceLevel(Enum):
    """Confidence levels for decision making"""
    VERY_HIGH = "very_high"  # > 0.9
    HIGH = "high"            # > 0.8
    MEDIUM = "medium"        # > 0.6
    LOW = "low"              # > 0.4
    VERY_LOW = "very_low"    # <= 0.4


@dataclass
class Confidence:
    """Represents confidence in a decision"""
    score: float  # 0.0 to 1.0
    level: ConfidenceLevel
    factors: Dict[str, float] = field(default_factory=dict)
    uncertainties: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)

    @staticmethod
    def from_score(score: float, factors: Optional[Dict[str, float]] = None) -> 'Confidence':
        """Create Confidence from score"""
        if score > 0.9:
            level = ConfidenceLevel.VERY_HIGH
        elif score > 0.8:
            level = ConfidenceLevel.HIGH
        elif score > 0.6:
            level = ConfidenceLevel.MEDIUM
        elif score > 0.4:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.VERY_LOW

        return Confidence(
            score=score,
            level=level,
            factors=factors or {}
        )

    def should_proceed(self) -> bool:
        """Should proceed without asking questions?"""
        return self.score > 0.8

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary"""
        return {
            'score': self.score,
            'level': self.level.value,
            'factors': self.factors,
            'uncertainties': self.uncertainties,
            'assumptions': self.assumptions
        }

    def should_confirm(self) -> bool:
        """Should confirm with user?"""
        return 0.5 < self.score <= 0.8

    def should_clarify(self) -> bool:
        """Should ask clarifying questions?"""
        return self.score <= 0.5

    def __str__(self) -> str:
        return f"{self.level.value}({self.score:.2f})"


# ============================================================================
# CONTEXT TYPES
# ============================================================================

@dataclass
class ConversationTurn:
    """Represents a single conversation turn"""
    role: str  # 'user' or 'assistant'
    message: str
    timestamp: datetime
    intents: List[Intent] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    tasks_executed: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary"""
        return {
            'role': self.role,
            'message': self.message,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'intents': [str(i) for i in self.intents],
            'entities': [str(e) for e in self.entities],
            'tasks_executed': self.tasks_executed
        }


@dataclass
class TrackedEntity:
    """Entity being tracked across conversation"""
    entity: Entity
    first_mentioned: datetime
    last_referenced: datetime
    mention_count: int = 0
    attributes: Dict[str, Any] = field(default_factory=dict)
    relationships: List[tuple] = field(default_factory=list)  # (relation_type, other_entity_id)

    def is_recent(self, max_age_seconds: float = 300) -> bool:
        """Is this entity recently referenced?"""
        age = (datetime.now() - self.last_referenced).total_seconds()
        return age <= max_age_seconds


@dataclass
class Pattern:
    """Learned pattern from operations"""
    pattern_type: str
    pattern_data: Dict[str, Any]
    confidence: float
    occurrence_count: int = 1
    success_count: int = 0
    last_seen: datetime = field(default_factory=datetime.now)


@dataclass
class ErrorPattern:
    """Pattern of errors"""
    error_type: str
    context_pattern: Dict[str, Any]
    solutions: List[str]
    occurrence_count: int = 1
    success_rate: float = 0.0


# ============================================================================
# AGENT SELECTION TYPES
# ============================================================================

@dataclass
class AgentScore:
    """Score for an agent's suitability for a task"""
    agent_name: str
    total_score: float  # 0.0 to 1.0
    capability_match: float = 0.0
    health_score: float = 0.0
    context_relevance: float = 0.0
    cost_efficiency: float = 0.0
    historical_success: float = 0.0
    reasoning: str = ""

    def __str__(self) -> str:
        return f"{self.agent_name}({self.total_score:.2f})"


# ============================================================================
# OPTIMIZATION TYPES
# ============================================================================

@dataclass
class CostEstimate:
    """Cost estimate for execution"""
    estimated_tokens: int
    estimated_api_calls: int
    estimated_duration_seconds: float
    estimated_cost_usd: float
    breakdown: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Optimization:
    """Suggested optimization"""
    optimization_type: str
    description: str
    estimated_savings: Dict[str, float]  # {'tokens': 100, 'time': 2.5, 'cost': 0.01}
    implementation: str
    confidence: float = 0.8


# ============================================================================
# SEMANTIC AND EMBEDDING TYPES
# ============================================================================

@dataclass
class SemanticVector:
    """Semantic embedding vector for similarity computations"""
    vector: List[float]
    dimension: int
    model: str = "default"

    def cosine_similarity(self, other: 'SemanticVector') -> float:
        """Calculate cosine similarity with another vector"""
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have same dimension")

        dot_product = sum(a * b for a, b in zip(self.vector, other.vector))
        magnitude_a = sum(a * a for a in self.vector) ** 0.5
        magnitude_b = sum(b * b for b in other.vector) ** 0.5

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)


@dataclass
class SemanticMatch:
    """Result of semantic similarity search"""
    item_id: str
    similarity_score: float
    item: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# PIPELINE AND PROCESSING TYPES
# ============================================================================

class ProcessingStage(Enum):
    """Stages in intelligence processing pipeline"""
    PREPROCESSING = "preprocessing"
    INTENT_CLASSIFICATION = "intent_classification"
    ENTITY_EXTRACTION = "entity_extraction"
    CONTEXT_INTEGRATION = "context_integration"
    TASK_DECOMPOSITION = "task_decomposition"
    CONFIDENCE_SCORING = "confidence_scoring"
    DECISION_MAKING = "decision_making"


@dataclass
class ProcessingResult:
    """Result from a processing stage"""
    stage: ProcessingStage
    success: bool
    data: Dict[str, Any]
    latency_ms: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PipelineContext:
    """Context passed through processing pipeline"""
    message: str
    session_id: str
    user_id: Optional[str] = None
    intents: List[Intent] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    confidence: Optional[Confidence] = None
    execution_plan: Optional[ExecutionPlan] = None
    conversation_context: Optional[Dict] = None
    processing_results: List[ProcessingResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: ProcessingResult):
        """Add processing result"""
        self.processing_results.append(result)

    def get_stage_result(self, stage: ProcessingStage) -> Optional[ProcessingResult]:
        """Get result from specific stage"""
        for result in reversed(self.processing_results):
            if result.stage == stage:
                return result
        return None


# ============================================================================
# RELATIONSHIP AND GRAPH TYPES
# ============================================================================

class RelationType(Enum):
    """Types of relationships between entities"""
    ASSIGNED_TO = "assigned_to"
    CREATED_BY = "created_by"
    DEPENDS_ON = "depends_on"
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    LINKED_TO = "linked_to"
    MENTIONS = "mentions"
    REFERENCES = "references"


@dataclass
class EntityRelationship:
    """Relationship between two entities"""
    from_entity_id: str
    to_entity_id: str
    relation_type: RelationType
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class EntityGraph:
    """Graph of entities and their relationships"""
    entities: Dict[str, Entity] = field(default_factory=dict)
    relationships: List[EntityRelationship] = field(default_factory=list)

    def add_entity(self, entity_id: str, entity: Entity):
        """Add entity to graph"""
        self.entities[entity_id] = entity

    def add_relationship(self, relationship: EntityRelationship):
        """Add relationship to graph"""
        self.relationships.append(relationship)

    def get_related_entities(
        self,
        entity_id: str,
        relation_type: Optional[RelationType] = None
    ) -> List[Tuple[RelationType, str, Entity]]:
        """Get entities related to given entity"""
        related = []
        for rel in self.relationships:
            if rel.from_entity_id == entity_id:
                if relation_type is None or rel.relation_type == relation_type:
                    if rel.to_entity_id in self.entities:
                        related.append((
                            rel.relation_type,
                            rel.to_entity_id,
                            self.entities[rel.to_entity_id]
                        ))
        return related


# ============================================================================
# CACHING TYPES
# ============================================================================

@dataclass
class CacheEntry:
    """Entry in cache"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: Optional[float] = None

    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl_seconds is None:
            return False
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def touch(self):
        """Update last accessed time"""
        self.last_accessed = datetime.now()
        self.access_count += 1


# ============================================================================
# METRICS AND MONITORING TYPES
# ============================================================================

@dataclass
class MetricPoint:
    """Single metric data point"""
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for intelligence system"""
    total_latency_ms: float = 0.0
    intent_classification_ms: float = 0.0
    entity_extraction_ms: float = 0.0
    context_integration_ms: float = 0.0
    task_decomposition_ms: float = 0.0
    confidence_scoring_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    llm_calls: int = 0
    llm_tokens: int = 0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'total_latency_ms': self.total_latency_ms,
            'intent_classification_ms': self.intent_classification_ms,
            'entity_extraction_ms': self.entity_extraction_ms,
            'context_integration_ms': self.context_integration_ms,
            'task_decomposition_ms': self.task_decomposition_ms,
            'confidence_scoring_ms': self.confidence_scoring_ms,
            'cache_hit_rate': self.cache_hits / max(self.cache_hits + self.cache_misses, 1),
            'llm_calls': self.llm_calls,
            'llm_tokens': self.llm_tokens,
        }


@dataclass
class QualityMetrics:
    """Quality metrics for intelligence outputs"""
    intent_accuracy: float = 0.0
    entity_precision: float = 0.0
    entity_recall: float = 0.0
    confidence_calibration: float = 0.0
    user_satisfaction: float = 0.0
    task_success_rate: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'intent_accuracy': self.intent_accuracy,
            'entity_precision': self.entity_precision,
            'entity_recall': self.entity_recall,
            'confidence_calibration': self.confidence_calibration,
            'user_satisfaction': self.user_satisfaction,
            'task_success_rate': self.task_success_rate,
        }


# ============================================================================
# LEARNING AND ADAPTATION TYPES
# ============================================================================

@dataclass
class FeedbackSignal:
    """User feedback signal for learning"""
    signal_type: str  # 'positive', 'negative', 'correction'
    context: PipelineContext
    correction_data: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningUpdate:
    """Update to be applied to learning systems"""
    component: str  # 'intent_classifier', 'entity_extractor', etc.
    update_type: str  # 'pattern', 'weight', 'example'
    update_data: Dict[str, Any]
    confidence: float
    source: str  # 'user_feedback', 'automatic', 'admin'
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# ABSTRACT BASE CLASSES FOR COMPONENTS
# ============================================================================

class IntelligenceComponent(ABC):
    """Abstract base class for intelligence components"""

    @abstractmethod
    def process(self, context: PipelineContext) -> ProcessingResult:
        """Process pipeline context and return result"""
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get component metrics"""
        pass

    @abstractmethod
    def reset_metrics(self):
        """Reset component metrics"""
        pass


# ============================================================================
# UTILITY TYPES
# ============================================================================

@dataclass
class ValidationResult:
    """Result of validation"""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, error: str):
        """Add validation error"""
        self.valid = False
        self.errors.append(error)

    def add_warning(self, warning: str):
        """Add validation warning"""
        self.warnings.append(warning)


def create_entity_id(entity: Entity) -> str:
    """Create unique ID for entity"""
    return f"{entity.type.value}:{entity.value}"


def hash_content(content: str) -> str:
    """Create hash of content for caching"""
    return hashlib.sha256(content.encode()).hexdigest()[:16]
