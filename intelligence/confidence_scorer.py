"""
Confidence Scoring System - Enhanced

Probabilistic confidence scoring with Bayesian estimation.

Features:
- Bayesian confidence estimation
- Multi-factor probability combination
- Confidence calibration against historical data
- Uncertainty quantification
- Decision theory for proceed/confirm/clarify

Author: AI System
Version: 3.0 - Major refactoring with probabilistic methods
"""

from typing import List, Dict, Optional, Tuple
import math
from .base_types import (
    Confidence, ConfidenceLevel, Intent, Entity,
    ExecutionPlan, Task
)


class ConfidenceScorer:
    """
    Score confidence in understanding and decisions

    Confidence factors:
    - Intent clarity (how clear is what user wants)
    - Entity completeness (do we have all needed info)
    - Task decomposition quality
    - Agent selection certainty
    - Historical success patterns
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        # Minimum required entities for each intent type
        self.required_entities = {
            'create': ['issue', 'pr', 'project', 'repository'],  # At least one
            'update': ['issue', 'pr', 'resource'],               # At least one
            'coordinate': ['channel', 'person', 'team'],         # At least one
            'analyze': ['code', 'file', 'repository'],           # At least one
        }

    def score_overall(
        self,
        message: str,
        intents: List[Intent],
        entities: List[Entity],
        plan: Optional[ExecutionPlan] = None
    ) -> Confidence:
        """
        Score overall confidence in understanding and plan

        Args:
            message: User message
            intents: Detected intents
            entities: Extracted entities
            plan: Execution plan (if created)

        Returns:
            Confidence score with factors
        """
        factors = {}

        # Factor 1: Intent clarity (0-1.0)
        factors['intent_clarity'] = self._score_intent_clarity(message, intents)

        # Factor 2: Entity completeness (0-1.0)
        factors['entity_completeness'] = self._score_entity_completeness(intents, entities)

        # Factor 3: Message ambiguity (0-1.0, lower is more ambiguous)
        factors['message_clarity'] = self._score_message_clarity(message)

        # Factor 4: Plan quality (if available)
        if plan:
            factors['plan_quality'] = self._score_plan_quality(plan)
        else:
            factors['plan_quality'] = 0.5

        # Calculate weighted average
        weights = {
            'intent_clarity': 0.3,
            'entity_completeness': 0.3,
            'message_clarity': 0.2,
            'plan_quality': 0.2
        }

        total_score = sum(factors[k] * weights[k] for k in factors)

        # Identify uncertainties
        uncertainties = self._identify_uncertainties(message, intents, entities, factors)

        # Identify assumptions
        assumptions = self._identify_assumptions(message, intents, entities)

        confidence = Confidence.from_score(total_score, factors)
        confidence.uncertainties = uncertainties
        confidence.assumptions = assumptions

        if self.verbose:
            print(f"[CONFIDENCE] Overall score: {confidence}")
            print(f"  Factors: {factors}")
            if uncertainties:
                print(f"  Uncertainties: {len(uncertainties)}")
            if assumptions:
                print(f"  Assumptions: {len(assumptions)}")

        return confidence

    def _score_intent_clarity(self, message: str, intents: List[Intent]) -> float:
        """
        Score how clear the user's intent is

        Factors:
        - Number of intents (1 is clearest)
        - Intent confidence scores
        - Presence of action words
        """
        if not intents:
            return 0.2  # Very unclear

        # High confidence intents
        high_conf_intents = [i for i in intents if i.confidence > 0.8]
        if not high_conf_intents:
            return 0.4  # Low confidence in all intents

        # Single clear intent is best
        if len(high_conf_intents) == 1:
            return min(high_conf_intents[0].confidence, 0.95)

        # Multiple clear intents is okay but slightly lower confidence
        if len(high_conf_intents) <= 3:
            avg_confidence = sum(i.confidence for i in high_conf_intents) / len(high_conf_intents)
            return avg_confidence * 0.9

        # Too many intents - might be unclear
        return 0.6

    def _score_entity_completeness(self, intents: List[Intent], entities: List[Entity]) -> float:
        """
        Score whether we have all needed entities for the intents

        Checks if we have the minimum required entities
        """
        if not intents:
            return 0.0

        primary_intent = intents[0]
        intent_type = primary_intent.type.value

        # Check if we have required entities
        required = self.required_entities.get(intent_type, [])

        if not required:
            # No specific requirements
            return 0.8

        # Check if we have at least one required entity
        entity_types = [e.type.value for e in entities]
        has_required = any(req in entity_types for req in required)

        if has_required:
            # Have at least one required entity
            # Score based on number of high-confidence entities
            high_conf_entities = [e for e in entities if e.confidence > 0.8]
            if len(high_conf_entities) >= 2:
                return 0.95
            elif len(high_conf_entities) == 1:
                return 0.80
            else:
                return 0.60
        else:
            # Missing required entities
            return 0.3

    def _score_message_clarity(self, message: str) -> float:
        """
        Score clarity of message itself

        Factors:
        - Length (too short or too long is unclear)
        - Question words (many questions = seeking info)
        - Specificity (specific details = clearer)
        """
        message_lower = message.lower()
        words = message_lower.split()
        word_count = len(words)

        score = 0.5  # Base score

        # Length scoring
        if 5 <= word_count <= 30:
            score += 0.2  # Good length
        elif word_count < 5:
            score -= 0.2  # Too short
        elif word_count > 50:
            score -= 0.1  # Too long

        # Question words reduce clarity for action requests
        question_words = ['what', 'how', 'why', 'when', 'where', 'which']
        question_count = sum(1 for qw in question_words if qw in message_lower)
        if question_count > 2:
            score -= 0.2  # Many questions = uncertain

        # Specific details increase clarity
        specific_indicators = ['#', '@', '-', '/', 'http']
        specificity = sum(1 for ind in specific_indicators if ind in message)
        score += min(specificity * 0.05, 0.2)

        return max(0.0, min(1.0, score))

    def _score_plan_quality(self, plan: ExecutionPlan) -> float:
        """
        Score quality of execution plan

        Factors:
        - No circular dependencies
        - Reasonable task count
        - Clear agent assignments
        - No high risks
        """
        score = 0.8  # Base score

        # Check for critical risks
        if plan.risks:
            critical_risks = [r for r in plan.risks if 'CRITICAL' in r]
            if critical_risks:
                score -= 0.5  # Major issue
            else:
                score -= 0.1 * len(plan.risks)

        # Check agent assignments
        tasks_without_agents = [t for t in plan.tasks if not t.agent]
        if tasks_without_agents:
            score -= 0.1 * len(tasks_without_agents) / len(plan.tasks)

        # Check task count
        if len(plan.tasks) == 0:
            score = 0.0
        elif len(plan.tasks) > 15:
            score -= 0.1  # Too many tasks

        return max(0.0, min(1.0, score))

    def _identify_uncertainties(
        self,
        message: str,
        intents: List[Intent],
        entities: List[Entity],
        factors: Dict[str, float]
    ) -> List[str]:
        """
        Identify specific uncertainties that should be clarified

        Returns list of uncertainty descriptions
        """
        uncertainties = []

        # Low intent clarity
        if factors.get('intent_clarity', 0) < 0.6:
            uncertainties.append("Unclear what action to take")

        # Low entity completeness
        if factors.get('entity_completeness', 0) < 0.6:
            if intents:
                primary_intent = intents[0].type.value
                uncertainties.append(f"Missing information for {primary_intent} action")

        # Ambiguous references
        ambiguous_words = ['it', 'that', 'this', 'them', 'those']
        message_lower = message.lower()
        has_ambiguous = any(word in message_lower.split() for word in ambiguous_words)

        if has_ambiguous and len(entities) == 0:
            uncertainties.append("Ambiguous references without context")

        # Multiple high-confidence intents
        if intents:
            high_conf = [i for i in intents if i.confidence > 0.7]
            if len(high_conf) > 3:
                uncertainties.append(f"Multiple actions requested ({len(high_conf)})")

        # No project/resource specified for create/update
        if intents and intents[0].type.value in ['create', 'update']:
            has_project = any(e.type.value in ['project', 'repository'] for e in entities)
            if not has_project:
                uncertainties.append("Project/repository not specified")

        return uncertainties

    def _identify_assumptions(
        self,
        message: str,
        intents: List[Intent],
        entities: List[Entity]
    ) -> List[str]:
        """
        Identify assumptions we're making

        Important to communicate these to user
        """
        assumptions = []

        # Assumption about which project
        if intents and intents[0].type.value in ['create', 'update']:
            has_explicit_project = any(e.type.value in ['project', 'repository'] for e in entities)
            if not has_explicit_project:
                assumptions.append("Using current/default project")

        # Assumption about priority
        has_priority = any(e.type.value == 'priority' for e in entities)
        if not has_priority and intents and intents[0].type.value == 'create':
            assumptions.append("Using default priority (medium)")

        # Assumption about assignee
        has_assignee = any(e.type.value == 'person' for e in entities)
        if not has_assignee and intents and intents[0].type.value == 'create':
            assumptions.append("Leaving unassigned")

        return assumptions

    def suggest_clarifications(self, confidence: Confidence, intents: List[Intent]) -> List[str]:
        """
        Suggest what clarifying questions to ask

        Args:
            confidence: Confidence score
            intents: Detected intents

        Returns:
            List of clarifying questions to ask user
        """
        questions = []

        # Ask based on uncertainties
        if "Unclear what action" in str(confidence.uncertainties):
            questions.append("What would you like me to do?")

        if "Missing information" in str(confidence.uncertainties):
            if intents:
                primary = intents[0].type.value
                if primary == 'create':
                    questions.append("What should I create? (issue, PR, page, etc.)")
                elif primary == 'update':
                    questions.append("What should I update?")
                elif primary == 'coordinate':
                    questions.append("Who should I notify?")

        if "Project/repository not specified" in str(confidence.uncertainties):
            questions.append("Which project or repository?")

        if "Ambiguous references" in str(confidence.uncertainties):
            questions.append("Can you clarify what 'it' or 'that' refers to?")

        return questions

    def should_proceed_automatically(self, confidence: Confidence) -> bool:
        """Should we proceed without asking user?"""
        return confidence.should_proceed()

    def should_confirm_with_user(self, confidence: Confidence) -> bool:
        """Should we confirm plan with user before executing?"""
        return confidence.should_confirm()

    def should_ask_clarifying_questions(self, confidence: Confidence) -> bool:
        """Should we ask clarifying questions?"""
        return confidence.should_clarify()

    def get_action_recommendation(self, confidence: Confidence) -> Tuple[str, str]:
        """
        Get recommended action based on confidence

        Returns:
            (action, explanation) tuple
            action: 'proceed', 'confirm', or 'clarify'
        """
        if self.should_proceed_automatically(confidence):
            return ('proceed', f"High confidence ({confidence.score:.2f}) - proceeding automatically")

        elif self.should_confirm_with_user(confidence):
            return ('confirm', f"Medium confidence ({confidence.score:.2f}) - confirming plan with user")

        else:
            return ('clarify', f"Low confidence ({confidence.score:.2f}) - asking clarifying questions")

    # ========================================================================
    # ENHANCED METHODS - V3.0: Bayesian and Probabilistic
    # ========================================================================

    def score_bayesian(
        self,
        message: str,
        intents: List[Intent],
        entities: List[Entity],
        plan: Optional[ExecutionPlan] = None,
        prior_confidence: float = 0.5
    ) -> Confidence:
        """
        Bayesian confidence scoring

        Uses Bayes' theorem to combine multiple evidence sources:
        P(correct|evidence) = P(evidence|correct) * P(correct) / P(evidence)

        Args:
            message: User message
            intents: Detected intents
            entities: Extracted entities
            plan: Execution plan
            prior_confidence: Prior probability (default 0.5)

        Returns:
            Confidence with Bayesian estimation
        """
        # Start with prior
        posterior = prior_confidence

        # Evidence 1: Intent clarity
        intent_likelihood = self._likelihood_from_intent_clarity(message, intents)
        posterior = self._bayesian_update(posterior, intent_likelihood)

        # Evidence 2: Entity completeness
        entity_likelihood = self._likelihood_from_entity_completeness(intents, entities)
        posterior = self._bayesian_update(posterior, entity_likelihood)

        # Evidence 3: Message clarity
        message_likelihood = self._likelihood_from_message_clarity(message)
        posterior = self._bayesian_update(posterior, message_likelihood)

        # Evidence 4: Plan quality (if available)
        if plan:
            plan_likelihood = self._likelihood_from_plan_quality(plan)
            posterior = self._bayesian_update(posterior, plan_likelihood)

        # Build confidence object
        factors = {
            'intent_likelihood': intent_likelihood,
            'entity_likelihood': entity_likelihood,
            'message_likelihood': message_likelihood,
            'prior': prior_confidence,
            'posterior': posterior
        }

        # Identify uncertainties and assumptions
        uncertainties = self._identify_uncertainties(message, intents, entities, factors)
        assumptions = self._identify_assumptions(message, intents, entities)

        confidence = Confidence.from_score(posterior, factors)
        confidence.uncertainties = uncertainties
        confidence.assumptions = assumptions

        if self.verbose:
            print(f"[CONFIDENCE] Bayesian score: {posterior:.3f}")
            print(f"  Prior: {prior_confidence:.3f}")
            print(f"  Likelihood factors: intent={intent_likelihood:.3f}, entity={entity_likelihood:.3f}, message={message_likelihood:.3f}")

        return confidence

    def _bayesian_update(self, prior: float, likelihood: float) -> float:
        """
        Bayesian update: P(H|E) = P(E|H) * P(H) / P(E)

        Simplified version using likelihood ratio.

        Args:
            prior: Prior probability P(H)
            likelihood: Likelihood P(E|H)

        Returns:
            Updated posterior probability
        """
        # Avoid division by zero
        prior = max(0.01, min(0.99, prior))
        likelihood = max(0.01, min(0.99, likelihood))

        # Compute likelihood ratio
        # Assuming P(E|¬H) is inversely related to P(E|H)
        likelihood_ratio = likelihood / (1 - likelihood)
        prior_odds = prior / (1 - prior)

        # Posterior odds = likelihood ratio * prior odds
        posterior_odds = likelihood_ratio * prior_odds

        # Convert back to probability
        posterior = posterior_odds / (1 + posterior_odds)

        return max(0.0, min(1.0, posterior))

    def _likelihood_from_intent_clarity(self, message: str, intents: List[Intent]) -> float:
        """Compute likelihood from intent clarity"""
        if not intents:
            return 0.2

        # Use highest confidence intent
        max_conf = max(i.confidence for i in intents)

        # Number of high-confidence intents
        high_conf_count = sum(1 for i in intents if i.confidence > 0.8)

        if high_conf_count == 1:
            return max_conf
        elif high_conf_count == 0:
            return 0.4
        else:
            # Multiple intents reduce clarity
            return max_conf * 0.85

    def _likelihood_from_entity_completeness(
        self,
        intents: List[Intent],
        entities: List[Entity]
    ) -> float:
        """Compute likelihood from entity completeness"""
        if not entities:
            return 0.3

        # Average entity confidence
        avg_conf = sum(e.confidence for e in entities) / len(entities)

        # Boost for multiple high-confidence entities
        high_conf_entities = sum(1 for e in entities if e.confidence > 0.8)

        if high_conf_entities >= 2:
            return min(avg_conf * 1.1, 1.0)
        elif high_conf_entities == 1:
            return avg_conf
        else:
            return avg_conf * 0.8

    def _likelihood_from_message_clarity(self, message: str) -> float:
        """Compute likelihood from message clarity"""
        words = message.split()
        word_count = len(words)

        # Optimal length is 5-30 words
        if 5 <= word_count <= 30:
            length_score = 0.9
        elif word_count < 5:
            length_score = 0.5
        else:
            length_score = 0.7

        # Check for ambiguous words
        ambiguous = ['maybe', 'might', 'could', 'should', 'possibly', 'perhaps', 'probably']
        ambiguous_count = sum(1 for word in words if word.lower() in ambiguous)

        if ambiguous_count == 0:
            ambiguity_score = 0.9
        elif ambiguous_count == 1:
            ambiguity_score = 0.7
        else:
            ambiguity_score = 0.5

        # Combine
        return (length_score + ambiguity_score) / 2

    def _likelihood_from_plan_quality(self, plan: ExecutionPlan) -> float:
        """Compute likelihood from plan quality"""
        if not plan.tasks:
            return 0.3

        # Check for critical risks
        critical_risks = sum(1 for r in plan.risks if 'CRITICAL' in r)
        if critical_risks > 0:
            return 0.3

        # Check agent assignments
        unassigned = sum(1 for t in plan.tasks if not t.agent)
        assignment_ratio = 1.0 - (unassigned / len(plan.tasks))

        # Reasonable task count (1-10 is good)
        if 1 <= len(plan.tasks) <= 10:
            task_count_score = 0.9
        elif len(plan.tasks) > 15:
            task_count_score = 0.6
        else:
            task_count_score = 0.8

        return (assignment_ratio + task_count_score) / 2

    def calibrate_with_history(
        self,
        confidence: Confidence,
        historical_accuracy: Optional[Dict[str, float]] = None
    ) -> Confidence:
        """
        Calibrate confidence against historical accuracy

        Adjusts confidence based on past performance.

        Args:
            confidence: Current confidence
            historical_accuracy: Map of confidence ranges to actual accuracy

        Returns:
            Calibrated confidence
        """
        if not historical_accuracy:
            return confidence

        # Find historical accuracy for this confidence range
        conf_range = self._get_confidence_range(confidence.score)
        historical_acc = historical_accuracy.get(conf_range, confidence.score)

        # Calibrate: move towards historical accuracy
        calibration_factor = 0.3  # Weight of historical data
        calibrated_score = (
            confidence.score * (1 - calibration_factor) +
            historical_acc * calibration_factor
        )

        # Create calibrated confidence
        calibrated = Confidence.from_score(calibrated_score, confidence.factors)
        calibrated.uncertainties = confidence.uncertainties
        calibrated.assumptions = confidence.assumptions

        if self.verbose:
            print(f"[CONFIDENCE] Calibrated: {confidence.score:.3f} → {calibrated_score:.3f} (historical: {historical_acc:.3f})")

        return calibrated

    def _get_confidence_range(self, score: float) -> str:
        """Get confidence range bucket"""
        if score >= 0.9:
            return "0.9-1.0"
        elif score >= 0.8:
            return "0.8-0.9"
        elif score >= 0.6:
            return "0.6-0.8"
        elif score >= 0.4:
            return "0.4-0.6"
        else:
            return "0.0-0.4"

    def compute_entropy(self, intents: List[Intent]) -> float:
        """
        Compute entropy of intent distribution

        High entropy = uncertain (many competing intents)
        Low entropy = certain (one clear intent)

        Args:
            intents: List of intents with confidences

        Returns:
            Entropy value (0 = certain, higher = uncertain)
        """
        if not intents:
            return 0.0

        # Normalize confidences to probabilities
        total_conf = sum(i.confidence for i in intents)
        if total_conf == 0:
            return 0.0

        probs = [i.confidence / total_conf for i in intents]

        # Compute Shannon entropy: H = -Σ p(x) * log2(p(x))
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log2(p)

        if self.verbose:
            print(f"[CONFIDENCE] Entropy: {entropy:.3f} ({'high uncertainty' if entropy > 1.5 else 'low uncertainty'})")

        return entropy

    def should_ask_for_clarification_bayesian(
        self,
        confidence: Confidence,
        entropy: float,
        cost_of_error: float = 0.5
    ) -> bool:
        """
        Bayesian decision theory for clarification

        Uses expected utility to decide if asking for clarification is worth it.

        Args:
            confidence: Confidence score
            entropy: Entropy of intent distribution
            cost_of_error: Cost of making wrong decision (0-1)

        Returns:
            True if should ask for clarification
        """
        # Expected utility of proceeding
        # U(proceed) = P(correct) * benefit - P(wrong) * cost
        p_correct = confidence.score
        p_wrong = 1 - p_correct

        benefit_correct = 1.0
        eu_proceed = p_correct * benefit_correct - p_wrong * cost_of_error

        # Expected utility of clarifying
        # U(clarify) = P(get_answer) * benefit_correct - small_cost_of_asking
        p_get_answer = 0.8  # Assume 80% chance user provides good answer
        cost_of_asking = 0.1

        eu_clarify = p_get_answer * benefit_correct - cost_of_asking

        # Also consider entropy
        if entropy > 2.0:  # High uncertainty
            eu_clarify += 0.2  # Boost clarification value

        should_clarify = eu_clarify > eu_proceed

        if self.verbose:
            print(f"[CONFIDENCE] Decision theory:")
            print(f"  EU(proceed): {eu_proceed:.3f}")
            print(f"  EU(clarify): {eu_clarify:.3f}")
            print(f"  Decision: {'CLARIFY' if should_clarify else 'PROCEED'}")

        return should_clarify

    def get_metrics(self) -> Dict:
        """Get scorer metrics"""
        return {
            'verbose': self.verbose,
        }

    def reset_metrics(self):
        """Reset metrics"""
        pass
