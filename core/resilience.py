import asyncio
import time
import random
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime

from core.errors import ErrorClassifier, ErrorClassification, ErrorCategory


@dataclass
class RetryAttempt:
    attempt_number: int
    timestamp: float
    error: Optional[str] = None
    success: bool = False
    delay_seconds: float = 0.0


@dataclass
class RetryContext:
    operation_key: str
    agent_name: str
    instruction: str
    max_retries: int
    attempts: List[RetryAttempt] = field(default_factory=list)
    first_attempt_time: float = field(default_factory=time.time)
    last_classification: Optional[ErrorClassification] = None

    @property
    def current_attempt(self) -> int:
        return len(self.attempts) + 1

    @property
    def should_retry(self) -> bool:
        if len(self.attempts) >= self.max_retries:
            return False

        if self.last_classification and not self.last_classification.is_retryable:
            return False

        return True

    @property
    def total_elapsed_time(self) -> float:
        return time.time() - self.first_attempt_time


class RetryManager:
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        verbose: bool = False
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.verbose = verbose

        self.contexts: Dict[str, RetryContext] = {}
        self.total_retry_budget = 50
        self.retries_used = 0

    def get_or_create_context(
        self,
        operation_key: str,
        agent_name: str,
        instruction: str
    ) -> RetryContext:
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
        if classification and classification.retry_delay_seconds > 0:
            base = float(classification.retry_delay_seconds)
        else:
            base = self.base_delay

        delay = base * (self.backoff_factor ** (attempt_number - 1))
        delay = min(delay, self.max_delay)

        if self.jitter:
            jitter_amount = delay * 0.2
            delay = delay + random.uniform(-jitter_amount, jitter_amount)
            delay = max(0.1, delay)

        return delay

    async def execute_with_retry(
        self,
        operation_key: str,
        agent_name: str,
        instruction: str,
        operation: Callable,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Any:
        context = self.get_or_create_context(operation_key, agent_name, instruction)
        last_error = None

        while context.should_retry:
            attempt = context.current_attempt

            try:
                if progress_callback:
                    if attempt == 1:
                        progress_callback(f"Executing {agent_name} agent...", attempt, self.max_retries)
                    else:
                        progress_callback(f"Retrying {agent_name} agent (attempt {attempt}/{self.max_retries})...", attempt, self.max_retries)

                result = await operation()

                context.attempts.append(RetryAttempt(
                    attempt_number=attempt,
                    timestamp=time.time(),
                    success=True
                ))

                return result

            except Exception as e:
                last_error = e
                error_msg = str(e)

                classification = ErrorClassifier.classify(error_msg, agent_name)
                context.last_classification = classification

                if not classification.is_retryable:
                    context.attempts.append(RetryAttempt(
                        attempt_number=attempt,
                        timestamp=time.time(),
                        error=error_msg,
                        success=False
                    ))

                    if self.verbose:
                        print(f"[RETRY] Non-retryable error ({classification.category.value}): {error_msg}")

                    raise

                if self.retries_used >= self.total_retry_budget:
                    if self.verbose:
                        print(f"[RETRY] Global retry budget exhausted ({self.retries_used}/{self.total_retry_budget})")
                    raise

                delay = self.calculate_delay(attempt, classification)

                context.attempts.append(RetryAttempt(
                    attempt_number=attempt,
                    timestamp=time.time(),
                    error=error_msg,
                    success=False,
                    delay_seconds=delay
                ))

                self.retries_used += 1

                if not context.should_retry:
                    if self.verbose:
                        print(f"[RETRY] Max retries ({self.max_retries}) reached for {agent_name}")
                    raise

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

                await asyncio.sleep(delay)

        if last_error:
            raise last_error

    def get_statistics(self) -> Dict[str, Any]:
        total_operations = len(self.contexts)
        total_attempts = sum(len(ctx.attempts) for ctx in self.contexts.values())
        successful_ops = sum(1 for ctx in self.contexts.values() if ctx.attempts and ctx.attempts[-1].success)
        failed_ops = total_operations - successful_ops

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
        if operation_key in self.contexts:
            del self.contexts[operation_key]

    def reset_all(self):
        self.contexts.clear()
        self.retries_used = 0
