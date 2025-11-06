"""
Intelligence Coordinator

Central orchestration for the entire intelligence system.
Coordinates all processing stages in a pipeline architecture.

Features:
- Pipeline-based processing
- Metrics collection
- Error handling and recovery
- Performance optimization
- Component coordination

Author: AI System
Version: 3.0
"""

from typing import Optional, Dict, List, Any
from datetime import datetime
import time

from .base_types import (
    PipelineContext, ProcessingStage, ProcessingResult,
    PerformanceMetrics, QualityMetrics, Intent, Entity,
    Confidence, ExecutionPlan, IntelligenceComponent
)
from .cache_layer import get_global_cache, IntelligentCache
from .intent_classifier import IntentClassifier
from .entity_extractor import EntityExtractor
from .context_manager import ConversationContextManager
from .task_decomposer import TaskDecomposer
from .confidence_scorer import ConfidenceScorer


class IntelligenceCoordinator:
    """
    Coordinates all intelligence components in a pipeline

    Pipeline Stages:
    1. Preprocessing - Normalize and validate input
    2. Intent Classification - Understand user intent
    3. Entity Extraction - Extract structured information
    4. Context Integration - Integrate conversation context
    5. Task Decomposition - Break down into executable tasks
    6. Confidence Scoring - Score confidence in understanding
    7. Decision Making - Decide on action (proceed/confirm/clarify)

    Features:
    - Caching of expensive operations
    - Metrics collection at each stage
    - Error handling and graceful degradation
    - Performance optimization
    """

    def __init__(
        self,
        session_id: str,
        agent_capabilities: Optional[Dict[str, List[str]]] = None,
        cache: Optional[IntelligentCache] = None,
        verbose: bool = False
    ):
        """
        Initialize intelligence coordinator

        Args:
            session_id: Session ID for context management
            agent_capabilities: Map of agent names to capabilities
            cache: Cache instance (uses global if None)
            verbose: Enable verbose logging
        """
        self.session_id = session_id
        self.agent_capabilities = agent_capabilities or {}
        self.cache = cache or get_global_cache()
        self.verbose = verbose

        # Initialize components
        self.intent_classifier = IntentClassifier(verbose=verbose)
        self.entity_extractor = EntityExtractor(verbose=verbose)
        self.context_manager = ConversationContextManager(session_id, verbose=verbose)
        self.task_decomposer = TaskDecomposer(agent_capabilities, verbose=verbose)
        self.confidence_scorer = ConfidenceScorer(verbose=verbose)

        # Metrics
        self.performance_metrics = PerformanceMetrics()
        self.quality_metrics = QualityMetrics()

        # Processing history
        self.processing_history: List[PipelineContext] = []

        if self.verbose:
            print(f"[COORDINATOR] Initialized for session: {session_id}")

    def process(self, message: str, user_id: Optional[str] = None) -> PipelineContext:
        """
        Process user message through intelligence pipeline

        Args:
            message: User message to process
            user_id: Optional user ID

        Returns:
            PipelineContext with all processing results
        """
        start_time = time.time()

        # Create pipeline context
        context = PipelineContext(
            message=message,
            session_id=self.session_id,
            user_id=user_id
        )

        try:
            # Stage 1: Preprocessing
            self._stage_preprocessing(context)

            # Stage 2: Intent Classification
            self._stage_intent_classification(context)

            # Stage 3: Entity Extraction
            self._stage_entity_extraction(context)

            # Stage 4: Context Integration
            self._stage_context_integration(context)

            # Stage 5: Task Decomposition
            self._stage_task_decomposition(context)

            # Stage 6: Confidence Scoring
            self._stage_confidence_scoring(context)

            # Stage 7: Decision Making
            self._stage_decision_making(context)

        except Exception as e:
            # Handle pipeline errors gracefully
            error_result = ProcessingResult(
                stage=ProcessingStage.DECISION_MAKING,
                success=False,
                data={},
                latency_ms=0.0,
                errors=[f"Pipeline error: {str(e)}"]
            )
            context.add_result(error_result)

            if self.verbose:
                print(f"[COORDINATOR] Pipeline error: {e}")

        # Update total latency
        total_latency = (time.time() - start_time) * 1000
        self.performance_metrics.total_latency_ms += total_latency

        # Add to history
        self.processing_history.append(context)

        # Keep history bounded
        if len(self.processing_history) > 100:
            self.processing_history = self.processing_history[-100:]

        if self.verbose:
            print(f"[COORDINATOR] Processing complete in {total_latency:.1f}ms")
            self._print_summary(context)

        return context

    def _stage_preprocessing(self, context: PipelineContext):
        """Stage 1: Preprocessing"""
        start_time = time.time()

        # Normalize message
        normalized_message = context.message.strip()

        # Validate message
        errors = []
        warnings = []

        if not normalized_message:
            errors.append("Empty message")

        if len(normalized_message) > 2000:
            warnings.append("Message very long (>2000 chars)")

        result = ProcessingResult(
            stage=ProcessingStage.PREPROCESSING,
            success=len(errors) == 0,
            data={'normalized_message': normalized_message},
            latency_ms=(time.time() - start_time) * 1000,
            errors=errors,
            warnings=warnings
        )

        context.add_result(result)
        context.message = normalized_message

    def _stage_intent_classification(self, context: PipelineContext):
        """Stage 2: Intent Classification"""
        start_time = time.time()

        try:
            # Classify intents
            intents = self.intent_classifier.classify(context.message)

            result = ProcessingResult(
                stage=ProcessingStage.INTENT_CLASSIFICATION,
                success=True,
                data={
                    'intents': intents,
                    'primary_intent': intents[0] if intents else None
                },
                latency_ms=(time.time() - start_time) * 1000
            )

            context.intents = intents
            context.add_result(result)

            # Update metrics
            latency = (time.time() - start_time) * 1000
            self.performance_metrics.intent_classification_ms += latency

        except Exception as e:
            result = ProcessingResult(
                stage=ProcessingStage.INTENT_CLASSIFICATION,
                success=False,
                data={},
                latency_ms=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )
            context.add_result(result)

    def _stage_entity_extraction(self, context: PipelineContext):
        """Stage 3: Entity Extraction"""
        start_time = time.time()

        try:
            # Get conversation context for entity extraction
            conv_context = self.context_manager.get_relevant_context(context.message)

            # Extract entities
            entities = self.entity_extractor.extract(context.message, conv_context)

            result = ProcessingResult(
                stage=ProcessingStage.ENTITY_EXTRACTION,
                success=True,
                data={
                    'entities': entities,
                    'entity_count': len(entities)
                },
                latency_ms=(time.time() - start_time) * 1000
            )

            context.entities = entities
            context.add_result(result)

            # Update metrics
            latency = (time.time() - start_time) * 1000
            self.performance_metrics.entity_extraction_ms += latency

        except Exception as e:
            result = ProcessingResult(
                stage=ProcessingStage.ENTITY_EXTRACTION,
                success=False,
                data={},
                latency_ms=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )
            context.add_result(result)

    def _stage_context_integration(self, context: PipelineContext):
        """Stage 4: Context Integration"""
        start_time = time.time()

        try:
            # Add turn to context manager
            self.context_manager.add_turn(
                role='user',
                message=context.message,
                intents=context.intents,
                entities=context.entities
            )

            # Get relevant context
            conversation_context = self.context_manager.get_relevant_context(context.message)

            result = ProcessingResult(
                stage=ProcessingStage.CONTEXT_INTEGRATION,
                success=True,
                data={
                    'conversation_context': conversation_context,
                    'recent_turns': len(conversation_context.get('recent_turns', [])),
                    'focused_entities': len(conversation_context.get('focused_entities', []))
                },
                latency_ms=(time.time() - start_time) * 1000
            )

            context.conversation_context = conversation_context
            context.add_result(result)

            # Update metrics
            latency = (time.time() - start_time) * 1000
            self.performance_metrics.context_integration_ms += latency

        except Exception as e:
            result = ProcessingResult(
                stage=ProcessingStage.CONTEXT_INTEGRATION,
                success=False,
                data={},
                latency_ms=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )
            context.add_result(result)

    def _stage_task_decomposition(self, context: PipelineContext):
        """Stage 5: Task Decomposition"""
        start_time = time.time()

        try:
            # Decompose into tasks
            execution_plan = self.task_decomposer.decompose(
                message=context.message,
                intents=context.intents,
                entities=context.entities,
                context=context.conversation_context
            )

            result = ProcessingResult(
                stage=ProcessingStage.TASK_DECOMPOSITION,
                success=True,
                data={
                    'execution_plan': execution_plan,
                    'task_count': len(execution_plan.tasks),
                    'estimated_duration': execution_plan.estimated_duration,
                    'estimated_cost': execution_plan.estimated_cost
                },
                latency_ms=(time.time() - start_time) * 1000
            )

            context.execution_plan = execution_plan
            context.add_result(result)

            # Update metrics
            latency = (time.time() - start_time) * 1000
            self.performance_metrics.task_decomposition_ms += latency

        except Exception as e:
            result = ProcessingResult(
                stage=ProcessingStage.TASK_DECOMPOSITION,
                success=False,
                data={},
                latency_ms=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )
            context.add_result(result)

    def _stage_confidence_scoring(self, context: PipelineContext):
        """Stage 6: Confidence Scoring"""
        start_time = time.time()

        try:
            # Score confidence
            confidence = self.confidence_scorer.score_overall(
                message=context.message,
                intents=context.intents,
                entities=context.entities,
                plan=context.execution_plan
            )

            result = ProcessingResult(
                stage=ProcessingStage.CONFIDENCE_SCORING,
                success=True,
                data={
                    'confidence': confidence,
                    'confidence_score': confidence.score,
                    'confidence_level': confidence.level.value,
                    'uncertainties': confidence.uncertainties,
                    'assumptions': confidence.assumptions
                },
                latency_ms=(time.time() - start_time) * 1000
            )

            context.confidence = confidence
            context.add_result(result)

            # Update metrics
            latency = (time.time() - start_time) * 1000
            self.performance_metrics.confidence_scoring_ms += latency

        except Exception as e:
            result = ProcessingResult(
                stage=ProcessingStage.CONFIDENCE_SCORING,
                success=False,
                data={},
                latency_ms=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )
            context.add_result(result)

    def _stage_decision_making(self, context: PipelineContext):
        """Stage 7: Decision Making"""
        start_time = time.time()

        try:
            confidence = context.confidence

            if not confidence:
                decision = 'clarify'
                reasoning = "No confidence score available"
            elif self.confidence_scorer.should_proceed_automatically(confidence):
                decision = 'proceed'
                reasoning = f"High confidence ({confidence.score:.2f})"
            elif self.confidence_scorer.should_confirm_with_user(confidence):
                decision = 'confirm'
                reasoning = f"Medium confidence ({confidence.score:.2f})"
            else:
                decision = 'clarify'
                reasoning = f"Low confidence ({confidence.score:.2f})"

            # Get clarification questions if needed
            clarifications = []
            if decision == 'clarify' and confidence:
                clarifications = self.confidence_scorer.suggest_clarifications(
                    confidence,
                    context.intents
                )

            result = ProcessingResult(
                stage=ProcessingStage.DECISION_MAKING,
                success=True,
                data={
                    'decision': decision,
                    'reasoning': reasoning,
                    'clarifications': clarifications
                },
                latency_ms=(time.time() - start_time) * 1000
            )

            context.add_result(result)

        except Exception as e:
            result = ProcessingResult(
                stage=ProcessingStage.DECISION_MAKING,
                success=False,
                data={'decision': 'clarify', 'reasoning': 'Error in decision making'},
                latency_ms=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )
            context.add_result(result)

    def _print_summary(self, context: PipelineContext):
        """Print processing summary"""
        print("\n" + "="*70)
        print("INTELLIGENCE PROCESSING SUMMARY")
        print("="*70)

        # Intents
        if context.intents:
            print(f"\nIntents ({len(context.intents)}):")
            for intent in context.intents[:3]:
                print(f"  • {intent}")

        # Entities
        if context.entities:
            print(f"\nEntities ({len(context.entities)}):")
            for entity in context.entities[:5]:
                print(f"  • {entity}")

        # Execution Plan
        if context.execution_plan:
            plan = context.execution_plan
            print(f"\nExecution Plan:")
            print(f"  • Tasks: {len(plan.tasks)}")
            print(f"  • Duration: {plan.estimated_duration:.1f}s")
            print(f"  • Cost: {plan.estimated_cost:.0f} tokens")
            if plan.risks:
                print(f"  • Risks: {len(plan.risks)}")

        # Confidence
        if context.confidence:
            conf = context.confidence
            print(f"\nConfidence: {conf.level.value.upper()} ({conf.score:.2f})")
            if conf.uncertainties:
                print(f"  • Uncertainties: {len(conf.uncertainties)}")
            if conf.assumptions:
                print(f"  • Assumptions: {len(conf.assumptions)}")

        # Decision
        decision_result = context.get_stage_result(ProcessingStage.DECISION_MAKING)
        if decision_result and decision_result.success:
            decision = decision_result.data.get('decision')
            reasoning = decision_result.data.get('reasoning')
            print(f"\nDecision: {decision.upper()}")
            print(f"  • {reasoning}")

        # Performance
        total_latency = sum(r.latency_ms for r in context.processing_results)
        print(f"\nPerformance:")
        print(f"  • Total latency: {total_latency:.1f}ms")
        for result in context.processing_results:
            print(f"  • {result.stage.value}: {result.latency_ms:.1f}ms")

        print("="*70 + "\n")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.performance_metrics.to_dict()

    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get quality metrics"""
        return self.quality_metrics.to_dict()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()

    def reset_metrics(self):
        """Reset all metrics"""
        self.performance_metrics = PerformanceMetrics()
        self.quality_metrics = QualityMetrics()
        self.cache.reset_stats()

        if self.verbose:
            print("[COORDINATOR] Metrics reset")

    def get_context_manager(self) -> ConversationContextManager:
        """Get context manager instance"""
        return self.context_manager

    def get_processing_history(self, count: int = 10) -> List[PipelineContext]:
        """Get recent processing history"""
        return self.processing_history[-count:]
