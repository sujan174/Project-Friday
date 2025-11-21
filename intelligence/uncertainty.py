"""
Uncertainty Quantification System

Implements proper uncertainty quantification for agentic AI:
- Temperature-scaled confidence calibration
- Entropy-based uncertainty measurement
- Clarification threshold detection
- Multiple uncertainty source tracking

Author: AI System
Version: 1.0
"""

import math
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from .react_types import (
    CalibratedConfidence, UncertaintyAnalysis, UncertaintySource
)
from .base_types import Intent, Entity, Confidence


class UncertaintyQuantifier:
    """
    Uncertainty Quantification System

    Provides properly calibrated confidence scores:
    - Temperature scaling for calibration
    - Entropy computation for uncertainty
    - Multi-source uncertainty tracking
    - Clarification threshold detection
    """

    def __init__(
        self,
        temperature: float = 1.5,
        clarify_threshold: float = 0.6,
        confirm_threshold: float = 0.8,
        verbose: bool = False
    ):
        """
        Initialize Uncertainty Quantifier

        Args:
            temperature: Temperature for confidence calibration (>1 = less confident)
            clarify_threshold: Uncertainty threshold for asking clarification
            confirm_threshold: Confidence threshold for proceeding without confirmation
            verbose: Enable verbose logging
        """
        self.temperature = temperature
        self.clarify_threshold = clarify_threshold
        self.confirm_threshold = confirm_threshold
        self.verbose = verbose

        # Calibration statistics (could be learned from data)
        self.calibration_offset = 0.0  # Adjustment based on historical accuracy

        # Historical accuracy tracking for proper calibration
        # Maps confidence bins to (total_predictions, correct_predictions)
        self.historical_accuracy: Dict[str, List[int]] = {
            "0.0-0.4": [0, 0],
            "0.4-0.6": [0, 0],
            "0.6-0.8": [0, 0],
            "0.8-0.9": [0, 0],
            "0.9-1.0": [0, 0]
        }

        # Statistics
        self.total_quantifications = 0
        self.clarifications_triggered = 0

    def quantify(
        self,
        raw_confidence: float,
        intents: List[Intent],
        entities: List[Entity],
        message: str
    ) -> UncertaintyAnalysis:
        """
        Quantify uncertainty for a classification

        Args:
            raw_confidence: Raw confidence score from classifier
            intents: Detected intents
            entities: Extracted entities
            message: Original message

        Returns:
            Complete uncertainty analysis
        """
        self.total_quantifications += 1

        # Calibrate the raw confidence
        calibrated = self._calibrate_confidence(raw_confidence)

        # Compute entropy from intent distribution
        entropy = self._compute_entropy(intents)

        # Identify uncertainty sources
        sources = self._identify_uncertainty_sources(intents, entities, message)

        # Compute confidence interval
        interval = self._compute_confidence_interval(calibrated, entropy)

        # Determine if clarification is needed
        should_clarify = self._should_clarify(calibrated, entropy, sources)

        # Generate clarification questions if needed
        questions = []
        if should_clarify:
            questions = self._generate_clarification_questions(intents, entities, message)
            self.clarifications_triggered += 1

        # Create calibrated confidence
        confidence = CalibratedConfidence(
            raw_score=raw_confidence,
            calibrated_score=calibrated,
            entropy=entropy,
            confidence_interval=interval,
            uncertainty_sources=sources,
            should_clarify=should_clarify,
            clarification_questions=questions
        )

        # Compute overall uncertainty (inverse of confidence)
        overall_uncertainty = 1.0 - calibrated

        # Adjust for entropy
        overall_uncertainty = min(1.0, overall_uncertainty + (entropy * 0.2))

        # Build analysis
        analysis = UncertaintyAnalysis(
            overall_uncertainty=overall_uncertainty,
            confidence=confidence,
            breakdown={
                'raw_confidence': raw_confidence,
                'calibrated': calibrated,
                'entropy': entropy,
                'ambiguity': len(sources) * 0.1
            },
            recommendations=self._generate_recommendations(confidence, sources)
        )

        if self.verbose:
            print(f"[UNCERTAINTY] Raw: {raw_confidence:.2f} -> Calibrated: {calibrated:.2f}")
            print(f"[UNCERTAINTY] Entropy: {entropy:.2f}, Sources: {[s.value for s in sources]}")
            if should_clarify:
                print(f"[UNCERTAINTY] Clarification needed: {questions}")

        return analysis

    def _calibrate_confidence(self, raw_confidence: float) -> float:
        """
        Apply temperature scaling and historical calibration

        Uses both temperature scaling and historical accuracy data
        to produce properly calibrated confidence scores.
        """
        # Temperature scaling
        calibrated = raw_confidence ** (1.0 / self.temperature)

        # Apply learned calibration offset
        calibrated = calibrated + self.calibration_offset

        # Use historical accuracy if we have enough data
        bin_key = self._get_confidence_bin(raw_confidence)
        if bin_key in self.historical_accuracy:
            total, correct = self.historical_accuracy[bin_key]
            if total >= 10:  # Need at least 10 samples
                historical_acc = correct / total
                # Blend: 70% temperature-scaled, 30% historical
                calibrated = 0.7 * calibrated + 0.3 * historical_acc

        # Clamp to valid range
        return max(0.0, min(1.0, calibrated))

    def _compute_entropy(self, intents: List[Intent]) -> float:
        """
        Compute entropy from intent confidence distribution

        High entropy = high uncertainty (multiple competing intents)
        Low entropy = low uncertainty (single clear intent)
        """
        if not intents:
            return 1.0  # Maximum uncertainty

        # Get confidence values
        confidences = [i.confidence for i in intents]

        # Normalize to probability distribution
        total = sum(confidences)
        if total == 0:
            return 1.0

        probabilities = [c / total for c in confidences]

        # Compute Shannon entropy
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize by maximum possible entropy
        max_entropy = math.log2(len(intents)) if len(intents) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return normalized_entropy

    def _identify_uncertainty_sources(
        self,
        intents: List[Intent],
        entities: List[Entity],
        message: str
    ) -> List[UncertaintySource]:
        """Identify sources of uncertainty"""
        sources = []

        # Check for ambiguous input
        ambiguous_words = ['maybe', 'might', 'could', 'should', 'possibly', 'perhaps', 'or']
        if any(word in message.lower() for word in ambiguous_words):
            sources.append(UncertaintySource.INPUT)

        # Check for competing intents (epistemic uncertainty)
        if len(intents) > 1:
            top_two = sorted(intents, key=lambda i: i.confidence, reverse=True)[:2]
            if len(top_two) == 2 and abs(top_two[0].confidence - top_two[1].confidence) < 0.15:
                sources.append(UncertaintySource.EPISTEMIC)

        # Check for missing entities (model uncertainty)
        if not entities:
            sources.append(UncertaintySource.MODEL)

        # Check for vague language (aleatoric uncertainty)
        vague_words = ['thing', 'stuff', 'something', 'anything', 'whatever']
        if any(word in message.lower() for word in vague_words):
            sources.append(UncertaintySource.ALEATORIC)

        # Short messages have higher uncertainty
        if len(message.split()) < 3:
            if UncertaintySource.INPUT not in sources:
                sources.append(UncertaintySource.INPUT)

        return sources

    def _compute_confidence_interval(
        self,
        calibrated: float,
        entropy: float
    ) -> Tuple[float, float]:
        """
        Compute confidence interval based on calibration and entropy

        Returns (lower_bound, upper_bound)
        """
        # Base uncertainty from entropy
        uncertainty = entropy * 0.3

        # Asymmetric bounds (can be more wrong than right)
        lower = max(0.0, calibrated - uncertainty - 0.1)
        upper = min(1.0, calibrated + uncertainty * 0.5)

        return (lower, upper)

    def _should_clarify(
        self,
        calibrated: float,
        entropy: float,
        sources: List[UncertaintySource]
    ) -> bool:
        """Determine if clarification is needed"""
        # Low calibrated confidence
        if calibrated < self.clarify_threshold:
            return True

        # High entropy (competing intents)
        if entropy > 0.7:
            return True

        # Multiple uncertainty sources
        if len(sources) >= 2:
            return True

        # Specific uncertainty sources that warrant clarification
        critical_sources = [UncertaintySource.INPUT, UncertaintySource.EPISTEMIC]
        if any(s in sources for s in critical_sources) and calibrated < 0.75:
            return True

        return False

    def _generate_clarification_questions(
        self,
        intents: List[Intent],
        entities: List[Entity],
        message: str
    ) -> List[str]:
        """Generate relevant clarification questions"""
        questions = []

        # Multiple competing intents
        if len(intents) > 1:
            top_intents = sorted(intents, key=lambda i: i.confidence, reverse=True)[:2]
            if len(top_intents) == 2:
                intent1 = top_intents[0].type.value
                intent2 = top_intents[1].type.value
                questions.append(f"Did you want to {intent1} or {intent2}?")

        # Missing entities
        if not entities:
            primary_intent = intents[0] if intents else None
            if primary_intent:
                if primary_intent.type.value == 'create':
                    questions.append("What would you like to create?")
                elif primary_intent.type.value == 'search':
                    questions.append("What are you searching for?")
                elif primary_intent.type.value == 'update':
                    questions.append("What would you like to update?")

        # Generic clarification for very short messages
        if len(message.split()) < 3 and not questions:
            questions.append("Could you provide more details about what you'd like to do?")

        # Limit number of questions
        return questions[:2]

    def _generate_recommendations(
        self,
        confidence: CalibratedConfidence,
        sources: List[UncertaintySource]
    ) -> List[str]:
        """Generate recommendations based on uncertainty"""
        recommendations = []

        if confidence.should_clarify:
            recommendations.append("Ask user for clarification before proceeding")

        if UncertaintySource.INPUT in sources:
            recommendations.append("Request more specific input from user")

        if UncertaintySource.EPISTEMIC in sources:
            recommendations.append("Consider multiple interpretations of the request")

        if confidence.calibrated_score < 0.5:
            recommendations.append("High uncertainty - confirm before taking action")

        if not recommendations:
            recommendations.append("Confidence is acceptable - proceed with task")

        return recommendations

    def update_calibration(self, predicted: float, actual: float):
        """
        Update calibration based on feedback

        Args:
            predicted: Predicted confidence
            actual: Actual outcome (1.0 = correct, 0.0 = incorrect)
        """
        # Simple exponential moving average update
        error = actual - predicted
        self.calibration_offset = 0.9 * self.calibration_offset + 0.1 * error

        # Track historical accuracy per bin
        bin_key = self._get_confidence_bin(predicted)
        if bin_key in self.historical_accuracy:
            self.historical_accuracy[bin_key][0] += 1  # Total predictions
            if actual >= 0.5:  # Consider success
                self.historical_accuracy[bin_key][1] += 1  # Correct predictions

    def _get_confidence_bin(self, score: float) -> str:
        """Get the confidence bin for a score"""
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

    def get_historical_accuracy_for_bin(self, bin_key: str) -> float:
        """Get historical accuracy for a confidence bin"""
        if bin_key not in self.historical_accuracy:
            return 0.5  # Default

        total, correct = self.historical_accuracy[bin_key]
        if total == 0:
            return 0.5  # No data, return neutral

        return correct / total

    def get_calibration_report(self) -> Dict[str, Any]:
        """Get calibration report showing expected vs actual accuracy"""
        report = {}
        for bin_key, (total, correct) in self.historical_accuracy.items():
            actual_acc = correct / total if total > 0 else None
            # Expected accuracy is midpoint of bin
            if bin_key == "0.9-1.0":
                expected = 0.95
            elif bin_key == "0.8-0.9":
                expected = 0.85
            elif bin_key == "0.6-0.8":
                expected = 0.7
            elif bin_key == "0.4-0.6":
                expected = 0.5
            else:
                expected = 0.2

            report[bin_key] = {
                "expected_accuracy": expected,
                "actual_accuracy": actual_acc,
                "total_samples": total,
                "is_calibrated": abs(expected - (actual_acc or expected)) < 0.1 if actual_acc else None
            }
        return report

    def get_decision_threshold(self) -> Dict[str, float]:
        """Get current decision thresholds"""
        return {
            'clarify_below': self.clarify_threshold,
            'confirm_above': self.confirm_threshold,
            'proceed_above': 0.85
        }

    def adjust_thresholds(
        self,
        clarify: Optional[float] = None,
        confirm: Optional[float] = None
    ):
        """
        Adjust decision thresholds

        Args:
            clarify: New clarification threshold
            confirm: New confirmation threshold
        """
        if clarify is not None:
            self.clarify_threshold = clarify
        if confirm is not None:
            self.confirm_threshold = confirm

    def get_statistics(self) -> Dict[str, Any]:
        """Get quantifier statistics"""
        clarification_rate = (
            self.clarifications_triggered / self.total_quantifications * 100
            if self.total_quantifications > 0 else 0
        )

        return {
            'total_quantifications': self.total_quantifications,
            'clarifications_triggered': self.clarifications_triggered,
            'clarification_rate': f"{clarification_rate:.1f}%",
            'calibration_offset': f"{self.calibration_offset:.3f}",
            'temperature': self.temperature
        }

    def reset_statistics(self):
        """Reset statistics"""
        self.total_quantifications = 0
        self.clarifications_triggered = 0


def compute_confidence_from_intents(intents: List[Intent]) -> Confidence:
    """
    Helper function to compute Confidence object from intents

    Args:
        intents: List of detected intents

    Returns:
        Confidence object with computed score
    """
    if not intents:
        return Confidence.from_score(0.0)

    # Primary confidence from top intent
    primary_conf = intents[0].confidence

    # Penalty for multiple competing intents
    if len(intents) > 1:
        gap = intents[0].confidence - intents[1].confidence
        if gap < 0.2:
            primary_conf *= 0.9  # Reduce confidence for ambiguity

    # Factors breakdown
    factors = {
        'primary_intent': intents[0].confidence,
        'num_intents': len(intents),
        'intent_gap': gap if len(intents) > 1 else 1.0
    }

    confidence = Confidence.from_score(primary_conf, factors)

    # Add uncertainties
    if len(intents) > 1 and gap < 0.2:
        confidence.uncertainties.append("Multiple competing intents detected")

    return confidence
