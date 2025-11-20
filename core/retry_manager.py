"""
Advanced Retry Management System

Provides intelligent retry logic with:
- Exponential backoff with jitter
- Retry budget tracking
- Progress feedback
- Learning from past failures

Author: AI System
Version: 2.0
"""

import asyncio
import time
import random
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime

from core.error_handler import ErrorClassifier, ErrorClassification


@dataclass
class RetryAttempt:
    """Records a single retry attempt"""
    attempt_number: int
    timestamp: float
    error: Optional[str] = None
    success: bool = False
    delay_seconds: float = 0.0


@dataclass
class RetryContext:
    """Tracks retry state for an operation"""
    operation_key: str
    agent_name: str
    instruction: str
    max_retries: int
    attempts: List[RetryAttempt] = field(default_factory=list)
    first_attempt_time: float = field(default_factory=time.time)
    last_classification: Optional[ErrorClassification] = None

    @property
    def current_attempt(self) -> int:
        """Get current attempt number (1-indexed)"""
        return len(self.attempts) + 1

    @property
    def should_retry(self) -> bool:
        """Check if we should retry based on attempts and last error"""
        if len(self.attempts) >= self.max_retries:
            return False

        if self.last_classification and not self.last_classification.is_retryable:
            return False

        return True

    @property
    def total_elapsed_time(self) -> float:
        """Total time spent on this operation"""
        return time.time() - self.first_attempt_time


class RetryManager:
    """
    Intelligent retry management with progress feedback.

    Features:
    - Smart exponential backoff with jitter
    - Progress callbacks for UI updates
    - Learning from error patterns
    - Retry budget management
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        verbose: bool = False
    ):
        """
        Initialize retry manager.

        Args:
            max_retries: Maximum retry attempts per operation
            base_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            backoff_factor: Multiplier for exponential backoff
            jitter: Add randomness to delays to avoid thundering herd
            verbose: Enable detailed logging
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.verbose = verbose

        # Track all retry contexts
        self.contexts: Dict[str, RetryContext] = {}

        # Global retry budget (prevent runaway retries)
        self.total_retry_budget = 50
        self.retries_used = 0

    def get_or_create_context(
        self,
        operation_key: str,
        agent_name: str,
        instruction: str
    ) -> RetryContext:
        """Get existing retry context or create new one"""
        if operation_key not in self.contexts:
            self.contexts[operation_key] = RetryContext(
                operation_key=operation_key,
                agent_name=agent_name,
                instruction=instruction,
                max_retries=self.max_retries
            )
        return self.contexts[operation_key]

    def calculate_delay(
        self,
        attempt_number: int,
        classification: Optional[ErrorClassification] = None
    ) -> float:
        """
        Calculate retry delay with exponential backoff and jitter.

        Args:
            attempt_number: Current attempt (1-indexed)
            classification: Error classification (may specify delay)

        Returns:
            Delay in seconds
        """
        # Use classification's recommended delay if available
        if classification and classification.retry_delay_seconds > 0:
            base = float(classification.retry_delay_seconds)
        else:
            base = self.base_delay

        # Exponential backoff: delay = base * (backoff_factor ^ (attempt - 1))
        delay = base * (self.backoff_factor ** (attempt_number - 1))

        # Cap at max delay
        delay = min(delay, self.max_delay)

        # Add jitter to avoid thundering herd
        if self.jitter:
            # Add ±20% random variation
            jitter_amount = delay * 0.2
            delay = delay + random.uniform(-jitter_amount, jitter_amount)
            delay = max(0.1, delay)  # Ensure positive

        return delay

    async def execute_with_retry(
        self,
        operation_key: str,
        agent_name: str,
        instruction: str,
        operation: Callable,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Any:
        """
        Execute an operation with intelligent retry logic.

        Args:
            operation_key: Unique key for this operation
            agent_name: Name of the agent
            instruction: Instruction being executed
            operation: Async callable to execute
            progress_callback: Optional callback for progress updates

        Returns:
            Result from operation

        Raises:
            Last exception if all retries exhausted
        """
        context = self.get_or_create_context(operation_key, agent_name, instruction)

        last_error = None

        while context.should_retry:
            attempt = context.current_attempt

            try:
                # Notify progress
                if progress_callback:
                    if attempt == 1:
                        progress_callback(f"Executing {agent_name} agent...", attempt, self.max_retries)
                    else:
                        progress_callback(f"Retrying {agent_name} agent (attempt {attempt}/{self.max_retries})...", attempt, self.max_retries)

                # Execute operation
                result = await operation()

                # Success! Record it
                context.attempts.append(RetryAttempt(
                    attempt_number=attempt,
                    timestamp=time.time(),
                    success=True
                ))

                return result

            except Exception as e:
                last_error = e
                error_msg = str(e)

                # Classify the error
                classification = ErrorClassifier.classify(error_msg, agent_name)
                context.last_classification = classification

                # Check if we should retry
                if not classification.is_retryable:
                    # Non-retryable error - stop immediately
                    context.attempts.append(RetryAttempt(
                        attempt_number=attempt,
                        timestamp=time.time(),
                        error=error_msg,
                        success=False
                    ))

                    if self.verbose:
                        print(f"[RETRY] Non-retryable error ({classification.category.value}): {error_msg}")

                    raise

                # Check retry budget
                if self.retries_used >= self.total_retry_budget:
                    if self.verbose:
                        print(f"[RETRY] Global retry budget exhausted ({self.retries_used}/{self.total_retry_budget})")
                    raise

                # Calculate delay
                delay = self.calculate_delay(attempt, classification)

                # Record attempt
                context.attempts.append(RetryAttempt(
                    attempt_number=attempt,
                    timestamp=time.time(),
                    error=error_msg,
                    success=False,
                    delay_seconds=delay
                ))

                self.retries_used += 1

                # Check if this is the last retry
                if not context.should_retry:
                    if self.verbose:
                        print(f"[RETRY] Max retries ({self.max_retries}) reached for {agent_name}")
                    raise

                # Notify about retry
                if progress_callback:
                    reason = classification.explanation
                    progress_callback(
                        f"⏳ {reason} - Waiting {delay:.1f}s before retry {attempt + 1}/{self.max_retries}...",
                        attempt,
                        self.max_retries
                    )

                if self.verbose:
                    print(f"[RETRY] Attempt {attempt} failed: {classification.category.value}")
                    print(f"[RETRY] Waiting {delay:.1f}s before retry...")

                # Wait with backoff
                await asyncio.sleep(delay)

        # All retries exhausted
        if last_error:
            raise last_error

    def get_statistics(self) -> Dict[str, Any]:
        """Get retry statistics for monitoring"""
        total_operations = len(self.contexts)
        total_attempts = sum(len(ctx.attempts) for ctx in self.contexts.values())
        successful_ops = sum(1 for ctx in self.contexts.values() if ctx.attempts and ctx.attempts[-1].success)
        failed_ops = total_operations - successful_ops

        # Average retries per operation
        avg_retries = (total_attempts - total_operations) / total_operations if total_operations > 0 else 0

        return {
            'total_operations': total_operations,
            'successful': successful_ops,
            'failed': failed_ops,
            'total_attempts': total_attempts,
            'avg_retries_per_operation': round(avg_retries, 2),
            'retry_budget_used': self.retries_used,
            'retry_budget_remaining': self.total_retry_budget - self.retries_used
        }

    def get_operation_summary(self, operation_key: str) -> Optional[str]:
        """Get human-readable summary of an operation's retry history"""
        if operation_key not in self.contexts:
            return None

        ctx = self.contexts[operation_key]

        if not ctx.attempts:
            return f"Operation pending (no attempts yet)"

        last_attempt = ctx.attempts[-1]

        if last_attempt.success:
            if len(ctx.attempts) == 1:
                return f"✓ Succeeded on first attempt"
            else:
                return f"✓ Succeeded after {len(ctx.attempts)} attempts ({ctx.total_elapsed_time:.1f}s total)"
        else:
            return f"✗ Failed after {len(ctx.attempts)} attempts - {last_attempt.error[:100]}"

    def reset_context(self, operation_key: str):
        """Reset retry context for an operation"""
        if operation_key in self.contexts:
            del self.contexts[operation_key]

    def reset_all(self):
        """Reset all retry contexts"""
        self.contexts.clear()
        self.retries_used = 0
