"""
Confirmation queue: Manages batches of actions waiting for user confirmation.

This module handles:
- Queuing actions for confirmation
- Deciding when to batch (time/size triggers)
- Managing batch lifecycle
- Tracking user decisions and edits
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import uuid
import asyncio
from .action_model import Action, ActionStatus
from config import Config
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ConfirmationBatch:
    """A single batch of actions ready for user review"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    actions: List[Action] = field(default_factory=list)

    # User decisions
    presented_at: Optional[datetime] = None
    decisions_received_at: Optional[datetime] = None

    def add_action(self, action: Action) -> None:
        """Add action to this batch"""
        action.batch_id = self.id
        self.actions.append(action)

    def get_pending_actions(self) -> List[Action]:
        """Get actions still waiting for decision"""
        return [a for a in self.actions if a.status == ActionStatus.PENDING]

    def mark_presented(self) -> None:
        """Mark this batch as shown to user"""
        self.presented_at = datetime.now()

    def mark_decisions_received(self) -> None:
        """Mark this batch as having decisions"""
        self.decisions_received_at = datetime.now()

    def is_empty(self) -> bool:
        """No actions to confirm?"""
        return len(self.get_pending_actions()) == 0


class ConfirmationQueue:
    """
    Manages pending confirmations and batching.
    Groups independent actions for efficient user review.
    """

    def __init__(
        self,
        batch_timeout_ms: Optional[int] = None,
        max_batch_size: Optional[int] = None,
        verbose: bool = False
    ):
        self.pending_actions: List[Action] = []
        self.current_batch: Optional[ConfirmationBatch] = None
        self.batch_timeout_ms = batch_timeout_ms or Config.BATCH_TIMEOUT_MS
        self.max_batch_size = max_batch_size or Config.MAX_BATCH_SIZE
        self.completed_batches: List[ConfirmationBatch] = []
        self.verbose = verbose

        # Thread-safety lock for concurrent queue operations
        self._lock = asyncio.Lock()
        self._max_pending = Config.MAX_PENDING_ACTIONS

    async def queue_action(self, action: Action) -> None:
        """Add an action to the queue (thread-safe)"""
        async with self._lock:
            # Check if queue is full
            if len(self.pending_actions) >= self._max_pending:
                error_msg = f"Queue full: {len(self.pending_actions)}/{self._max_pending} actions pending"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            self.pending_actions.append(action)
            if self.verbose:
                logger.info(f"Action queued: {action.agent_name}.{action.action_type.value}")

    def should_batch_now(self) -> bool:
        """
        Should we present the batch to user now?
        Conditions:
        1. Max batch size reached
        2. Oldest action in queue exceeds timeout
        """
        if not self.pending_actions:
            return False

        # Condition 1: Max size reached
        if len(self.pending_actions) >= self.max_batch_size:
            if self.verbose:
                print(f"[QUEUE] Batch size ({len(self.pending_actions)}) reached max")
            return True

        # Condition 2: Timeout exceeded
        oldest_action = self.pending_actions[0]
        age_ms = (datetime.now() - oldest_action.created_at).total_seconds() * 1000
        if age_ms >= self.batch_timeout_ms:
            if self.verbose:
                print(f"[QUEUE] Batch timeout exceeded ({age_ms:.0f}ms >= {self.batch_timeout_ms}ms)")
            return True

        return False

    async def prepare_batch(self) -> ConfirmationBatch:
        """
        Create batch from pending actions (thread-safe).
        Takes up to max_batch_size actions from queue.
        """
        async with self._lock:
            if not self.pending_actions:
                return ConfirmationBatch()

            # Take up to max_batch_size actions
            batch_actions = self.pending_actions[:self.max_batch_size]
            self.pending_actions = self.pending_actions[self.max_batch_size:]

            batch = ConfirmationBatch()
            for action in batch_actions:
                batch.add_action(action)

            self.current_batch = batch
            if self.verbose:
                logger.info(f"Batch prepared with {len(batch.actions)} action(s)")
            return batch

    def get_pending_count(self) -> int:
        """How many actions waiting to be batched?"""
        return len(self.pending_actions)

    def get_current_batch(self) -> Optional[ConfirmationBatch]:
        """Get the batch currently being reviewed"""
        return self.current_batch

    def archive_batch(self) -> None:
        """Move current batch to history"""
        if self.current_batch:
            self.completed_batches.append(self.current_batch)
            if self.verbose:
                print(f"[QUEUE] Batch {self.current_batch.id[:8]}... archived")
            self.current_batch = None

    def get_batch_history(self) -> List[ConfirmationBatch]:
        """Get completed batches for audit"""
        return self.completed_batches

    def get_stats(self) -> Dict[str, any]:
        """Get queue statistics"""
        total_actions = len(self.pending_actions) + sum(
            len(b.actions) for b in self.completed_batches
        )
        return {
            'pending_actions': len(self.pending_actions),
            'completed_batches': len(self.completed_batches),
            'total_actions': total_actions,
            'current_batch': self.current_batch.id if self.current_batch else None,
        }
