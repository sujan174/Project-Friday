"""
Error Classification and Handling System

Provides intelligent error categorization to distinguish between:
- Transient errors (retry)
- Permanent errors (stop)
- Capability gaps (inform user)
- Permission issues (require user action)
- Rate limiting (backoff and retry)

This enables smarter recovery and better user messaging.

Author: AI System
Version: 1.0
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib


class ErrorCategory(str, Enum):
    """Categories of errors with different handling strategies"""
    TRANSIENT = "transient"          # Temporary - retry with backoff
    RATE_LIMIT = "rate_limit"        # API rate limited - retry with longer delay
    CAPABILITY = "capability"        # API doesn't support this - don't retry
    PERMISSION = "permission"        # Access denied - require user action
    VALIDATION = "validation"        # Invalid input - don't retry
    UNKNOWN = "unknown"              # Unknown - assume retryable but inform


@dataclass
class ErrorClassification:
    """Complete error classification with recovery suggestions"""
    category: ErrorCategory
    is_retryable: bool
    explanation: str  # What happened in simple terms
    technical_details: Optional[str] = None  # Full error message
    suggestions: List[str] = None  # What user can do
    retry_delay_seconds: int = 0  # Backoff delay for rate limits

    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []


class ErrorClassifier:
    """
    Intelligent error classification system.

    Analyzes error messages to determine:
    - Root cause category
    - Whether to retry
    - How to inform the user
    - What to suggest next
    """

    # Error patterns for each category
    CAPABILITY_PATTERNS = [
        'does not support',
        'cannot fetch',
        'not available',
        'api does not',
        'is not supported',
        'not implemented',
        'unsupported operation',
        'cannot provide',
        'unable to retrieve',
    ]

    PERMISSION_PATTERNS = [
        'permission denied',
        'forbidden',
        'unauthorized',
        '401',
        '403',
        'access denied',
        'private repository',
        'not found',
        '404',
        'insufficient permissions',
        'access token',
    ]

    RATE_LIMIT_PATTERNS = [
        'rate limit',
        'rate limited',
        'too many requests',
        'quota exceeded',
        '429',
        '503',
        'throttled',
        'back off',
    ]

    TRANSIENT_PATTERNS = [
        'timeout',
        'timed out',
        'connection',
        'network',
        'temporarily',
        '502',
        '504',
        'gateway',
        'temporary',
        'service unavailable',
    ]

    VALIDATION_PATTERNS = [
        'invalid input',
        'invalid parameter',
        'required field',
        'bad request',
        '400',
        'validation error',
        'malformed',
    ]

    @staticmethod
    def classify(error_msg: str, agent_name: Optional[str] = None) -> ErrorClassification:
        """
        Classify an error message and return handling strategy.

        Args:
            error_msg: The error message string
            agent_name: Optional agent name for context-specific classification

        Returns:
            ErrorClassification with category, retry decision, and suggestions
        """
        error_lower = error_msg.lower()

        # Check each category in priority order
        # (more specific categories first)

        # 1. Check for CAPABILITY errors (permanent - don't retry)
        if ErrorClassifier._matches_patterns(error_lower, ErrorClassifier.CAPABILITY_PATTERNS):
            return ErrorClassification(
                category=ErrorCategory.CAPABILITY,
                is_retryable=False,
                explanation="The underlying API or agent does not support this operation",
                technical_details=error_msg,
                suggestions=[
                    "• Try a different approach or agent",
                    "• Check the agent's stated capabilities",
                    "• Look for an alternative tool that supports this operation",
                ]
            )

        # 2. Check for PERMISSION errors (permanent - don't retry)
        if ErrorClassifier._matches_patterns(error_lower, ErrorClassifier.PERMISSION_PATTERNS):
            return ErrorClassification(
                category=ErrorCategory.PERMISSION,
                is_retryable=False,
                explanation="Access denied - check permissions and resource status",
                technical_details=error_msg,
                suggestions=[
                    "• Verify the resource exists and is accessible",
                    "• Check if the repository/resource is private",
                    "• Verify API token has required permissions",
                    "• Confirm you have access to this resource",
                ]
            )

        # 3. Check for VALIDATION errors (permanent - don't retry)
        if ErrorClassifier._matches_patterns(error_lower, ErrorClassifier.VALIDATION_PATTERNS):
            return ErrorClassification(
                category=ErrorCategory.VALIDATION,
                is_retryable=False,
                explanation="Invalid input provided - request cannot be processed",
                technical_details=error_msg,
                suggestions=[
                    "• Check the instruction syntax",
                    "• Verify all required parameters are provided",
                    "• Ensure values are in the correct format",
                ]
            )

        # 4. Check for RATE_LIMIT errors (retryable with backoff)
        if ErrorClassifier._matches_patterns(error_lower, ErrorClassifier.RATE_LIMIT_PATTERNS):
            return ErrorClassification(
                category=ErrorCategory.RATE_LIMIT,
                is_retryable=True,
                explanation="API rate limit reached - will retry with delay",
                technical_details=error_msg,
                suggestions=[
                    "• Operation will be retried automatically",
                    "• Consider spacing out large operations",
                ],
                retry_delay_seconds=10  # Wait 10 seconds before retry
            )

        # 5. Check for TRANSIENT errors (retryable)
        if ErrorClassifier._matches_patterns(error_lower, ErrorClassifier.TRANSIENT_PATTERNS):
            return ErrorClassification(
                category=ErrorCategory.TRANSIENT,
                is_retryable=True,
                explanation="Temporary network or service issue - will retry",
                technical_details=error_msg,
                suggestions=[
                    "• Operation will be retried automatically",
                    "• If it persists, the service may be down",
                ],
                retry_delay_seconds=2
            )

        # 6. Default to UNKNOWN (assume retryable but flag)
        return ErrorClassification(
            category=ErrorCategory.UNKNOWN,
            is_retryable=True,
            explanation="An unexpected error occurred",
            technical_details=error_msg,
            suggestions=[
                "• Check agent logs for more details",
                "• Verify the instruction format",
                "• Try with a simpler or more specific request",
            ]
        )

    @staticmethod
    def _matches_patterns(text: str, patterns: List[str]) -> bool:
        """Check if text matches any pattern (case-insensitive)"""
        return any(pattern in text for pattern in patterns)


def format_error_for_user(
    classification: ErrorClassification,
    agent_name: str,
    instruction: str,
    attempt_number: int = 1,
    max_attempts: int = 3
) -> str:
    """
    Format an error classification into a user-friendly message.

    Args:
        classification: The error classification
        agent_name: Name of the agent that failed
        instruction: The original instruction
        attempt_number: Current attempt number
        max_attempts: Maximum retry attempts

    Returns:
        Formatted error message for user
    """

    # Build error header based on category
    if classification.category == ErrorCategory.CAPABILITY:
        header = "❌ **Cannot perform this operation**"
    elif classification.category == ErrorCategory.PERMISSION:
        header = "🔐 **Access Denied**"
    elif classification.category == ErrorCategory.RATE_LIMIT:
        header = "⏳ **Rate Limited - Retrying**"
    elif classification.category == ErrorCategory.TRANSIENT:
        header = "⏳ **Temporary Issue - Retrying**"
    elif classification.category == ErrorCategory.VALIDATION:
        header = "❌ **Invalid Input**"
    else:
        header = "⚠️ **Error**"

    # Build message
    message = f"{header}\n\n"
    message += f"**What happened**: {classification.explanation}\n\n"

    # Add retry context if applicable
    if classification.is_retryable and attempt_number > 1:
        message += f"**Attempt**: {attempt_number}/{max_attempts}\n\n"

    # Add suggestions
    if classification.suggestions:
        message += "**What you can try**:\n"
        for suggestion in classification.suggestions:
            message += f"{suggestion}\n"
        message += "\n"

    # Add technical details if verbose or UNKNOWN
    if classification.category == ErrorCategory.UNKNOWN:
        message += f"**Technical details**: {classification.technical_details}\n"

    return message.strip()


class DuplicateOperationDetector:
    """
    Detects when the same operation is being retried multiple times with identical failures.

    This catches situations like:
    - Agent keeps trying the same impossible operation
    - Agent forgets it already executed something
    - Stuck in retry loop for operation that will never succeed

    Example from real session:
    - Notion agent adds tasks successfully
    - Then retries adding same tasks 10+ times
    - Each time gives different/conflicting responses
    - User keeps confirming thinking they're different operations
    """

    def __init__(self, window_size: int = 5, similarity_threshold: float = 0.8):
        """
        Initialize duplicate detector.

        Args:
            window_size: Number of recent operations to track
            similarity_threshold: How similar errors need to be (0.0-1.0)
        """
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.operation_history: Dict[str, List[Dict[str, Any]]] = {}

    def _create_operation_hash(self, agent_name: str, instruction: str) -> str:
        """Create a hash of the operation (agent + instruction)"""
        key = f"{agent_name}:{instruction[:100]}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def _similarity_score(self, error1: str, error2: str) -> float:
        """
        Calculate similarity between two error messages.

        Returns: 0.0 (completely different) to 1.0 (identical)
        """
        if error1 == error2:
            return 1.0

        # Normalize to lowercase
        e1 = error1.lower()
        e2 = error2.lower()

        # Check if one contains the other (high similarity)
        if e1 in e2 or e2 in e1:
            return 0.9

        # Count matching words
        words1 = set(e1.split())
        words2 = set(e2.split())

        if not words1 or not words2:
            return 0.0

        # Calculate Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def track_operation(
        self,
        agent_name: str,
        instruction: str,
        error: str,
        success: bool = False
    ) -> None:
        """
        Track an operation attempt.

        Args:
            agent_name: Name of the agent
            instruction: The instruction being executed
            error: Error message (or empty if success)
            success: Whether the operation succeeded
        """
        op_hash = self._create_operation_hash(agent_name, instruction)

        if op_hash not in self.operation_history:
            self.operation_history[op_hash] = []

        # Add to history
        self.operation_history[op_hash].append({
            'error': error,
            'success': success
        })

        # Keep only recent attempts
        if len(self.operation_history[op_hash]) > self.window_size:
            self.operation_history[op_hash] = self.operation_history[op_hash][-self.window_size:]

    def detect_duplicate_failure(
        self,
        agent_name: str,
        instruction: str,
        current_error: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect if this operation has failed multiple times identically or similarly.

        Returns:
            (is_duplicate, explanation)
            - is_duplicate: True if this looks like a stuck operation
            - explanation: Description of what's happening
        """
        op_hash = self._create_operation_hash(agent_name, instruction)

        if op_hash not in self.operation_history:
            return False, None

        history = self.operation_history[op_hash]

        if len(history) < 2:
            return False, None

        # Get recent failed attempts
        recent_failures = [h for h in history if not h['success']]

        if len(recent_failures) < 2:
            return False, None

        # Check if errors are similar
        error_messages = [f['error'] for f in recent_failures]

        # Compare last error with previous ones
        similar_count = 0
        for prev_error in error_messages[:-1]:
            similarity = self._similarity_score(current_error, prev_error)
            if similarity >= self.similarity_threshold:
                similar_count += 1

        # If 2+ similar errors, it's a duplicate failure pattern
        if similar_count >= 1:
            num_attempts = len(recent_failures)
            return True, (
                f"This operation has been attempted {num_attempts} times with similar failures. "
                f"The {agent_name} agent appears to be stuck or unable to complete this task."
            )

        # Check for conflicting responses (different errors for same operation)
        if len(set(error_messages)) > 1 and len(recent_failures) >= 3:
            return True, (
                f"The {agent_name} agent is giving conflicting responses for the same operation. "
                f"This suggests it's not properly tracking state or has a capability limitation."
            )

        return False, None

    def detect_inconsistent_responses(
        self,
        agent_name: str,
        instruction: str
    ) -> Tuple[bool, Optional[List[str]]]:
        """
        Detect if agent is giving inconsistent responses (success/failure alternating).

        Returns:
            (is_inconsistent, response_pattern)
            - is_inconsistent: True if responses are contradictory
            - response_pattern: List of recent responses (success/failed)
        """
        op_hash = self._create_operation_hash(agent_name, instruction)

        if op_hash not in self.operation_history:
            return False, None

        history = self.operation_history[op_hash]

        if len(history) < 3:
            return False, None

        # Get recent attempts
        recent = history[-5:]  # Last 5
        pattern = ['success' if h['success'] else 'failed' for h in recent]

        # Check for alternating or chaotic pattern
        success_count = sum(1 for h in recent if h['success'])
        failed_count = len(recent) - success_count

        # If both successes AND failures in recent attempts, that's suspicious
        if success_count > 0 and failed_count > 0 and len(recent) >= 3:
            return True, pattern

        return False, None

    def get_duplicate_summary(
        self,
        agent_name: str,
        instruction: str
    ) -> Optional[str]:
        """
        Get a summary of duplicate/stuck operations for this agent+instruction.
        """
        op_hash = self._create_operation_hash(agent_name, instruction)

        if op_hash not in self.operation_history:
            return None

        history = self.operation_history[op_hash]
        total_attempts = len(history)
        success_count = sum(1 for h in history if h['success'])
        failed_count = total_attempts - success_count

        if total_attempts < 2:
            return None

        return (
            f"Attempt history: {total_attempts} total attempts "
            f"({success_count} successful, {failed_count} failed)\n"
            f"Status: This operation appears to be stuck or unstable"
        )
