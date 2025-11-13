import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from .distributed_tracing import get_global_tracer, traced_span, SpanKind, SpanStatus
from .logging_config import get_logger


class IntelligenceStage(Enum):
    PREPROCESSING = "PREPROCESSING"
    INTENT_CLASSIFICATION = "INTENT_CLASSIFICATION"
    ENTITY_EXTRACTION = "ENTITY_EXTRACTION"
    CONTEXT_INTEGRATION = "CONTEXT_INTEGRATION"
    TASK_DECOMPOSITION = "TASK_DECOMPOSITION"
    CONFIDENCE_SCORING = "CONFIDENCE_SCORING"
    DECISION_MAKING = "DECISION_MAKING"


class DecisionType(Enum):
    PROCEED = "PROCEED"
    CONFIRM = "CONFIRM"
    CLARIFY = "CLARIFY"
    REJECT = "REJECT"
    DELEGATE = "DELEGATE"


@dataclass
class StageResult:
    stage: IntelligenceStage
    success: bool
    duration_ms: float
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'stage': self.stage.value,
            'success': self.success,
            'duration_ms': self.duration_ms,
            'timestamp': self.timestamp,
            'timestamp_iso': datetime.fromtimestamp(self.timestamp).isoformat(),
            'data': self.data,
            'metrics': self.metrics,
            'errors': self.errors,
            'warnings': self.warnings
        }


@dataclass
class IntentClassificationResult:
    timestamp: float
    message: str
    detected_intents: List[str]
    confidence_scores: Dict[str, float]
    classification_method: str
    duration_ms: float
    cache_hit: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'timestamp_iso': datetime.fromtimestamp(self.timestamp).isoformat(),
            'user_message': self.message,
            'detected_intents': self.detected_intents,
            'confidence_scores': self.confidence_scores,
            'classification_method': self.classification_method,
            'duration_ms': self.duration_ms,
            'cache_hit': self.cache_hit
        }


@dataclass
class EntityExtractionResult:
    timestamp: float
    message: str
    extracted_entities: Dict[str, List[str]]
    entity_relationships: List[Dict[str, Any]]
    confidence: float
    duration_ms: float
    cache_hit: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'timestamp_iso': datetime.fromtimestamp(self.timestamp).isoformat(),
            'user_message': self.message,
            'extracted_entities': self.extracted_entities,
            'entity_relationships': self.entity_relationships,
            'confidence': self.confidence,
            'duration_ms': self.duration_ms,
            'cache_hit': self.cache_hit
        }


@dataclass
class TaskDecompositionResult:
    timestamp: float
    tasks: List[Dict[str, Any]]
    dependency_graph: Dict[str, List[str]]
    execution_plan: str
    confidence: float
    duration_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'timestamp_iso': datetime.fromtimestamp(self.timestamp).isoformat(),
            'tasks': self.tasks,
            'dependency_graph': self.dependency_graph,
            'execution_plan': self.execution_plan,
            'confidence': self.confidence,
            'duration_ms': self.duration_ms
        }


@dataclass
class DecisionRecord:
    timestamp: float
    decision_type: DecisionType
    confidence: float
    reasoning: str
    factors: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'timestamp_iso': datetime.fromtimestamp(self.timestamp).isoformat(),
            'decision_type': self.decision_type.value,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'factors': self.factors,
            'metadata': self.metadata
        }


class IntelligenceLogger:
    def __init__(
        self,
        session_id: str,
        export_dir: Optional[str] = "logs/intelligence",
        verbose: bool = False
    ):
        self.session_id = session_id
        self.export_dir = Path(export_dir) if export_dir else None
        self.verbose = verbose

        self.logger = get_logger(__name__)
        self.tracer = get_global_tracer()

        if self.export_dir:
            self.export_dir.mkdir(parents=True, exist_ok=True)

        self.stage_results: List[StageResult] = []
        self.intent_classifications: List[IntentClassificationResult] = []
        self.entity_extractions: List[EntityExtractionResult] = []
        self.task_decompositions: List[TaskDecompositionResult] = []
        self.decisions: List[DecisionRecord] = []

        self.start_time = time.time()
        self.total_messages_processed = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def log_stage_start(
        self,
        stage: IntelligenceStage,
        input_data: Optional[Dict[str, Any]] = None
    ) -> float:
        with traced_span(
            f"intelligence.{stage.value.lower()}",
            kind=SpanKind.INTELLIGENCE
        ) as span:
            span.set_attribute("intelligence.stage", stage.value)
            span.add_event("stage_started", input_data or {})

            if self.verbose:
                self.logger.debug(
                    f"Intelligence stage started: {stage.value}",
                    extra={
                        'stage': stage.value,
                        'input_data': input_data
                    }
                )

        return time.time()

    def log_stage_complete(
        self,
        stage: IntelligenceStage,
        start_time: float,
        success: bool,
        data: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None
    ):
        duration_ms = (time.time() - start_time) * 1000

        result = StageResult(
            stage=stage,
            success=success,
            duration_ms=duration_ms,
            timestamp=time.time(),
            data=data or {},
            metrics=metrics or {},
            errors=errors or [],
            warnings=warnings or []
        )

        self.stage_results.append(result)

        with traced_span(
            f"intelligence.{stage.value.lower()}.complete",
            kind=SpanKind.INTELLIGENCE
        ) as span:
            span.set_attribute("intelligence.stage", stage.value)
            span.set_attribute("intelligence.success", success)
            span.set_attribute("intelligence.duration_ms", duration_ms)

            if not success:
                span.set_status(SpanStatus.ERROR, "; ".join(errors or []))

            span.add_event("stage_completed", {
                'success': success,
                'duration_ms': duration_ms
            })

            level = "info" if success else "error"
            msg = f"Intelligence stage {'succeeded' if success else 'failed'}: {stage.value} ({duration_ms:.1f}ms)"

            getattr(self.logger, level)(
                msg,
                extra=result.to_dict()
            )

    def log_intent_classification(
        self,
        message: str,
        detected_intents: List[str],
        confidence_scores: Dict[str, float],
        classification_method: str,
        duration_ms: float,
        cache_hit: bool = False
    ):
        result = IntentClassificationResult(
            timestamp=time.time(),
            message=message,
            detected_intents=detected_intents,
            confidence_scores=confidence_scores,
            classification_method=classification_method,
            duration_ms=duration_ms,
            cache_hit=cache_hit
        )

        self.intent_classifications.append(result)

        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        with traced_span(
            "intelligence.intent_classification",
            kind=SpanKind.INTELLIGENCE
        ) as span:
            span.set_attribute("intent.method", classification_method)
            span.set_attribute("intent.count", len(detected_intents))
            span.set_attribute("intent.cache_hit", cache_hit)
            span.set_attribute("intent.duration_ms", duration_ms)

            for intent in detected_intents:
                span.add_event("intent_detected", {
                    'intent': intent,
                    'confidence': confidence_scores.get(intent, 0.0)
                })

            self.logger.info(
                f"Intents classified: {', '.join(detected_intents)} "
                f"(method: {classification_method}, cache: {cache_hit})",
                extra=result.to_dict()
            )

    def log_entity_extraction(
        self,
        message: str,
        extracted_entities: Dict[str, List[str]],
        entity_relationships: List[Dict[str, Any]],
        confidence: float,
        duration_ms: float,
        cache_hit: bool = False
    ):
        result = EntityExtractionResult(
            timestamp=time.time(),
            message=message,
            extracted_entities=extracted_entities,
            entity_relationships=entity_relationships,
            confidence=confidence,
            duration_ms=duration_ms,
            cache_hit=cache_hit
        )

        self.entity_extractions.append(result)

        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        with traced_span(
            "intelligence.entity_extraction",
            kind=SpanKind.INTELLIGENCE
        ) as span:
            entity_count = sum(len(entities) for entities in extracted_entities.values())

            span.set_attribute("entity.count", entity_count)
            span.set_attribute("entity.types", len(extracted_entities))
            span.set_attribute("entity.relationships", len(entity_relationships))
            span.set_attribute("entity.confidence", confidence)
            span.set_attribute("entity.cache_hit", cache_hit)
            span.set_attribute("entity.duration_ms", duration_ms)

            for entity_type, entities in extracted_entities.items():
                span.add_event("entities_extracted", {
                    'type': entity_type,
                    'count': len(entities),
                    'entities': entities
                })

            self.logger.info(
                f"Entities extracted: {entity_count} entities of {len(extracted_entities)} types "
                f"(confidence: {confidence:.2f}, cache: {cache_hit})",
                extra=result.to_dict()
            )

    def log_task_decomposition(
        self,
        tasks: List[Dict[str, Any]],
        dependency_graph: Dict[str, List[str]],
        execution_plan: str,
        confidence: float,
        duration_ms: float
    ):
        result = TaskDecompositionResult(
            timestamp=time.time(),
            tasks=tasks,
            dependency_graph=dependency_graph,
            execution_plan=execution_plan,
            confidence=confidence,
            duration_ms=duration_ms
        )

        self.task_decompositions.append(result)

        with traced_span(
            "intelligence.task_decomposition",
            kind=SpanKind.INTELLIGENCE
        ) as span:
            span.set_attribute("task.count", len(tasks))
            span.set_attribute("task.execution_plan", execution_plan)
            span.set_attribute("task.confidence", confidence)
            span.set_attribute("task.duration_ms", duration_ms)

            for i, task in enumerate(tasks):
                span.add_event("task_identified", {
                    'task_index': i,
                    'task_name': task.get('name', 'unknown'),
                    'agent': task.get('agent', 'unknown'),
                    'priority': task.get('priority', 0)
                })

            self.logger.info(
                f"Tasks decomposed: {len(tasks)} tasks "
                f"(plan: {execution_plan}, confidence: {confidence:.2f})",
                extra=result.to_dict()
            )

    def log_context_resolution(
        self,
        references_resolved: int,
        context_applied: bool,
        duration_ms: float,
        details: Optional[Dict[str, Any]] = None
    ):
        with traced_span(
            "intelligence.context_resolution",
            kind=SpanKind.INTELLIGENCE
        ) as span:
            span.set_attribute("context.references_resolved", references_resolved)
            span.set_attribute("context.applied", context_applied)
            span.set_attribute("context.duration_ms", duration_ms)

            if details:
                for key, value in details.items():
                    span.set_attribute(f"context.{key}", value)

            self.logger.info(
                f"Context resolved: {references_resolved} references resolved",
                extra={
                    'references_resolved': references_resolved,
                    'context_applied': context_applied,
                    'duration_ms': duration_ms,
                    'details': details
                }
            )

    def log_confidence_score(
        self,
        overall_confidence: float,
        component_scores: Dict[str, float],
        factors: Dict[str, Any],
        duration_ms: float
    ):
        with traced_span(
            "intelligence.confidence_scoring",
            kind=SpanKind.INTELLIGENCE
        ) as span:
            span.set_attribute("confidence.overall", overall_confidence)
            span.set_attribute("confidence.duration_ms", duration_ms)

            for component, score in component_scores.items():
                span.set_attribute(f"confidence.{component}", score)

            span.add_event("confidence_calculated", {
                'overall_confidence': overall_confidence,
                'component_scores': component_scores
            })

            self.logger.info(
                f"Confidence scored: {overall_confidence:.2f}",
                extra={
                    'overall_confidence': overall_confidence,
                    'component_scores': component_scores,
                    'factors': factors,
                    'duration_ms': duration_ms
                }
            )

    def log_decision(
        self,
        decision_type: DecisionType,
        confidence: float,
        reasoning: str,
        factors: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        decision = DecisionRecord(
            timestamp=time.time(),
            decision_type=decision_type,
            confidence=confidence,
            reasoning=reasoning,
            factors=factors,
            metadata=metadata or {}
        )

        self.decisions.append(decision)

        with traced_span(
            "intelligence.decision",
            kind=SpanKind.INTELLIGENCE
        ) as span:
            span.set_attribute("decision.type", decision_type.value)
            span.set_attribute("decision.confidence", confidence)
            span.set_attribute("decision.reasoning", reasoning)

            for key, value in factors.items():
                span.set_attribute(f"decision.factor.{key}", value)

            span.add_event("decision_made", {
                'decision_type': decision_type.value,
                'confidence': confidence
            })

            self.logger.info(
                f"Decision: {decision_type.value} (confidence: {confidence:.2f}) - {reasoning}",
                extra=decision.to_dict()
            )

    def log_message_processing_start(self, message: str) -> float:
        self.total_messages_processed += 1

        with traced_span(
            "intelligence.process_message",
            kind=SpanKind.INTELLIGENCE
        ) as span:
            span.set_attribute("message.length", len(message))
            span.set_attribute("message.number", self.total_messages_processed)
            span.add_event("processing_started")

            self.logger.info(
                f"Processing message #{self.total_messages_processed}",
                extra={'user_message': message[:100]}
            )

        return time.time()

    def log_message_processing_complete(
        self,
        start_time: float,
        success: bool,
        error: Optional[str] = None
    ):
        duration_ms = (time.time() - start_time) * 1000

        with traced_span(
            "intelligence.process_message.complete",
            kind=SpanKind.INTELLIGENCE
        ) as span:
            span.set_attribute("processing.success", success)
            span.set_attribute("processing.duration_ms", duration_ms)

            if error:
                span.set_status(SpanStatus.ERROR, error)

            span.add_event("processing_completed", {
                'success': success,
                'duration_ms': duration_ms
            })

            level = "info" if success else "error"
            msg = f"Message processing {'succeeded' if success else 'failed'} ({duration_ms:.1f}ms)"

            getattr(self.logger, level)(
                msg,
                extra={
                    'success': success,
                    'duration_ms': duration_ms,
                    'error': error
                }
            )

    def export_session_summary(self) -> Dict[str, Any]:
        summary = {
            'session_id': self.session_id,
            'start_time': self.start_time,
            'start_time_iso': datetime.fromtimestamp(self.start_time).isoformat(),
            'duration_ms': (time.time() - self.start_time) * 1000,
            'total_messages_processed': self.total_messages_processed,
            'cache_statistics': {
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
            },
            'stage_results': [r.to_dict() for r in self.stage_results],
            'intent_classifications': [i.to_dict() for i in self.intent_classifications],
            'entity_extractions': [e.to_dict() for e in self.entity_extractions],
            'task_decompositions': [t.to_dict() for t in self.task_decompositions],
            'decisions': [d.to_dict() for d in self.decisions],
            'statistics': self._calculate_statistics()
        }

        if self.export_dir:
            filename = f"intelligence_{self.session_id}_{int(self.start_time)}.json"
            filepath = self.export_dir / filename

            try:
                with open(filepath, 'w') as f:
                    json.dump(summary, f, indent=2)
            except Exception as e:
                self.logger.error(f"Failed to export intelligence summary: {e}")

        return summary

    def _calculate_statistics(self) -> Dict[str, Any]:
        stage_durations = {}
        for result in self.stage_results:
            stage = result.stage.value
            if stage not in stage_durations:
                stage_durations[stage] = []
            stage_durations[stage].append(result.duration_ms)

        avg_durations = {
            stage: sum(durations) / len(durations)
            for stage, durations in stage_durations.items()
        }

        return {
            'total_stages_executed': len(self.stage_results),
            'avg_stage_durations_ms': avg_durations,
            'total_intents_classified': len(self.intent_classifications),
            'total_entities_extracted': sum(
                sum(len(entities) for entities in e.extracted_entities.values())
                for e in self.entity_extractions
            ),
            'total_tasks_decomposed': sum(len(t.tasks) for t in self.task_decompositions),
            'total_decisions_made': len(self.decisions),
            'decision_types': {
                dt.value: sum(1 for d in self.decisions if d.decision_type == dt)
                for dt in DecisionType
            },
            'avg_confidence': sum(d.confidence for d in self.decisions) / len(self.decisions) if self.decisions else 0
        }
