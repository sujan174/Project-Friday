"""Base Types for Intelligence System"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod
import hashlib
import json


class IntentType(Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    ANALYZE = "analyze"
    COORDINATE = "coordinate"
    WORKFLOW = "workflow"
    SEARCH = "search"
    UNKNOWN = "unknown"


class RiskLevel(Enum):
    """Risk level for operations - used for confidence-based autonomy"""
    LOW = "low"          # Read-only operations - auto-execute
    MEDIUM = "medium"    # Write operations - confirm if confidence < threshold
    HIGH = "high"        # Destructive operations - always confirm


@dataclass
class Intent:
    type: IntentType
    confidence: float
    entities: List['Entity'] = field(default_factory=list)
    implicit_requirements: List[str] = field(default_factory=list)
    raw_indicators: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.type.value}({self.confidence:.2f})"


class EntityType(Enum):
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
    type: EntityType
    value: str
    confidence: float
    context: Optional[str] = None
    normalized_value: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.type.value}:{self.value}({self.confidence:.2f})"


@dataclass
class Task:
    id: str
    action: str
    agent: Optional[str] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    conditions: Optional[str] = None
    priority: int = 0
    estimated_duration: float = 0.0
    estimated_cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"Task({self.id}: {self.agent or '?'}.{self.action})"


@dataclass
class DependencyGraph:
    tasks: Dict[str, Task] = field(default_factory=dict)
    edges: List[tuple] = field(default_factory=list)

    def add_task(self, task: Task):
        self.tasks[task.id] = task

    def add_dependency(self, from_task_id: str, to_task_id: str):
        self.edges.append((from_task_id, to_task_id))

    def get_execution_order(self) -> List[Task]:
        in_degree = {task_id: 0 for task_id in self.tasks}
        for from_id, to_id in self.edges:
            in_degree[to_id] = in_degree.get(to_id, 0) + 1

        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            task_id = queue.pop(0)
            result.append(self.tasks[task_id])

            for from_id, to_id in self.edges:
                if from_id == task_id:
                    in_degree[to_id] -= 1
                    if in_degree[to_id] == 0:
                        queue.append(to_id)

        return result

    def has_cycle(self) -> bool:
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
    tasks: List[Task] = field(default_factory=list)
    dependency_graph: Optional[DependencyGraph] = None
    estimated_duration: float = 0.0
    estimated_cost: float = 0.0
    risks: List[str] = field(default_factory=list)
    optimizations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_execution_order(self) -> List[Task]:
        if self.dependency_graph:
            return self.dependency_graph.get_execution_order()
        return self.tasks


class ConfidenceLevel(Enum):
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class Confidence:
    score: float
    level: ConfidenceLevel
    factors: Dict[str, float] = field(default_factory=dict)
    uncertainties: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)

    @staticmethod
    def from_score(score: float, factors: Optional[Dict[str, float]] = None) -> 'Confidence':
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
        return self.score > 0.8

    def should_review(self) -> bool:
        return 0.4 < self.score <= 0.8

    def should_clarify(self) -> bool:
        return self.score <= 0.4

    def __str__(self) -> str:
        return f"{self.level.value}({self.score:.2f})"


@dataclass
class ConversationTurn:
    role: str
    message: str
    timestamp: datetime
    intents: List[Intent] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    tasks_executed: List[str] = field(default_factory=list)


@dataclass
class TrackedEntity:
    entity: Entity
    first_mentioned: datetime
    last_referenced: datetime
    mention_count: int = 0
    attributes: Dict[str, Any] = field(default_factory=dict)
    relationships: List[tuple] = field(default_factory=list)

    def is_recent(self, max_age_seconds: float = 300) -> bool:
        age = (datetime.now() - self.last_referenced).total_seconds()
        return age <= max_age_seconds


@dataclass
class Pattern:
    pattern_type: str
    pattern_data: Dict[str, Any]
    confidence: float
    occurrence_count: int = 1
    success_count: int = 0
    last_seen: datetime = field(default_factory=datetime.now)


@dataclass
class ErrorPattern:
    error_type: str
    context_pattern: Dict[str, Any]
    solutions: List[str]
    occurrence_count: int = 1
    success_rate: float = 0.0


@dataclass
class AgentScore:
    agent_name: str
    total_score: float
    capability_match: float = 0.0
    health_score: float = 0.0
    context_relevance: float = 0.0
    cost_efficiency: float = 0.0
    historical_success: float = 0.0
    reasoning: str = ""

    def __str__(self) -> str:
        return f"{self.agent_name}({self.total_score:.2f})"


@dataclass
class CostEstimate:
    estimated_tokens: int
    estimated_api_calls: int
    estimated_duration_seconds: float
    estimated_cost_usd: float
    breakdown: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Optimization:
    optimization_type: str
    description: str
    estimated_savings: Dict[str, float]
    implementation: str
    confidence: float = 0.8


@dataclass
class SemanticVector:
    """Semantic embedding vector for similarity computations"""
    vector: List[float]
    dimension: int
    model: str = "default"

    def cosine_similarity(self, other: 'SemanticVector') -> float:
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
    item_id: str
    similarity_score: float
    item: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProcessingStage(Enum):
    PREPROCESSING = "preprocessing"
    INTENT_CLASSIFICATION = "intent_classification"
    ENTITY_EXTRACTION = "entity_extraction"
    CONTEXT_INTEGRATION = "context_integration"
    TASK_DECOMPOSITION = "task_decomposition"
    CONFIDENCE_SCORING = "confidence_scoring"
    DECISION_MAKING = "decision_making"


@dataclass
class ProcessingResult:
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
        self.processing_results.append(result)

    def get_stage_result(self, stage: ProcessingStage) -> Optional[ProcessingResult]:
        for result in reversed(self.processing_results):
            if result.stage == stage:
                return result
        return None


@dataclass
class CacheEntry:
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: Optional[float] = None

    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def touch(self):
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class MetricPoint:
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
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
    intent_accuracy: float = 0.0
    entity_precision: float = 0.0
    entity_recall: float = 0.0
    confidence_calibration: float = 0.0
    user_satisfaction: float = 0.0
    task_success_rate: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            'intent_accuracy': self.intent_accuracy,
            'entity_precision': self.entity_precision,
            'entity_recall': self.entity_recall,
            'confidence_calibration': self.confidence_calibration,
            'user_satisfaction': self.user_satisfaction,
            'task_success_rate': self.task_success_rate,
        }


@dataclass
class FeedbackSignal:
    """User feedback signal for learning"""
    signal_type: str
    context: PipelineContext
    correction_data: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningUpdate:
    """Update to be applied to learning systems"""
    component: str
    update_type: str
    update_data: Dict[str, Any]
    confidence: float
    source: str
    timestamp: datetime = field(default_factory=datetime.now)


class IntelligenceComponent(ABC):
    """Abstract base class for intelligence components"""

    @abstractmethod
    def process(self, context: PipelineContext) -> ProcessingResult:
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def reset_metrics(self):
        pass


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, error: str):
        self.valid = False
        self.errors.append(error)

    def add_warning(self, warning: str):
        self.warnings.append(warning)


class OperationRiskClassifier:
    """
    Classifies operation risk for confidence-based autonomy.

    Rules:
    - READ, SEARCH, ANALYZE = LOW risk → auto-execute
    - CREATE, UPDATE, COORDINATE, WORKFLOW = MEDIUM risk → confirm if confidence < 0.75
    - DELETE = HIGH risk → always confirm
    """

    @staticmethod
    def classify_risk(intents: List[Intent]) -> RiskLevel:
        """Classify risk level based on primary intent"""
        if not intents:
            return RiskLevel.MEDIUM

        # Get highest confidence intent
        primary_intent = max(intents, key=lambda i: i.confidence)

        # Classify based on intent type
        if primary_intent.type == IntentType.DELETE:
            return RiskLevel.HIGH

        elif primary_intent.type in [IntentType.READ, IntentType.SEARCH, IntentType.ANALYZE]:
            return RiskLevel.LOW

        elif primary_intent.type in [IntentType.CREATE, IntentType.UPDATE,
                                      IntentType.COORDINATE, IntentType.WORKFLOW]:
            return RiskLevel.MEDIUM

        else:  # UNKNOWN or other
            return RiskLevel.MEDIUM

    @staticmethod
    def should_confirm(risk_level: RiskLevel, confidence: float) -> Tuple[bool, str]:
        """
        Determine if user confirmation is needed.

        Returns:
            (needs_confirmation, reason)
        """
        if risk_level == RiskLevel.HIGH:
            return (True, "Destructive operation requires confirmation")

        elif risk_level == RiskLevel.MEDIUM:
            if confidence < 0.75:
                return (True, f"Medium risk operation with moderate confidence ({confidence:.2f})")
            else:
                return (False, f"Medium risk operation with high confidence ({confidence:.2f})")

        else:  # LOW risk
            return (False, "Read-only operation - safe to auto-execute")


def hash_content(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]
