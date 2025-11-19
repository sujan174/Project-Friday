"""
Undo System for Destructive Operations

Provides ability to undo:
- Jira issue deletions/closures
- Slack message deletions
- GitHub PR closures
- Notion page deletions
- File deletions

Author: AI System
Version: 1.0
"""

import json
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta


class UndoableOperationType(str, Enum):
    """Types of operations that can be undone"""
    JIRA_DELETE_ISSUE = "jira_delete_issue"
    JIRA_CLOSE_ISSUE = "jira_close_issue"
    JIRA_TRANSITION = "jira_transition"

    SLACK_DELETE_MESSAGE = "slack_delete_message"
    SLACK_ARCHIVE_CHANNEL = "slack_archive_channel"

    GITHUB_CLOSE_PR = "github_close_pr"
    GITHUB_CLOSE_ISSUE = "github_close_issue"
    GITHUB_DELETE_BRANCH = "github_delete_branch"

    NOTION_DELETE_PAGE = "notion_delete_page"
    NOTION_ARCHIVE_PAGE = "notion_archive_page"

    CUSTOM = "custom"


@dataclass
class UndoSnapshot:
    """Snapshot of state before a destructive operation"""
    operation_id: str
    operation_type: UndoableOperationType
    agent_name: str
    timestamp: float
    description: str  # Human-readable description

    # State before operation
    before_state: Dict[str, Any]

    # Information needed to undo
    undo_params: Dict[str, Any]

    # Result of the operation (for display)
    operation_result: Optional[str] = None

    # Undo function identifier
    undo_handler: Optional[str] = None

    # TTL (Time To Live) - how long undo is available
    ttl_seconds: int = 3600  # 1 hour default

    # Whether this has been undone
    undone: bool = False
    undone_at: Optional[float] = None

    @property
    def is_expired(self) -> bool:
        """Check if undo window has expired"""
        return time.time() - self.timestamp > self.ttl_seconds

    @property
    def can_undo(self) -> bool:
        """Check if this operation can still be undone"""
        return not self.undone and not self.is_expired

    @property
    def age_minutes(self) -> float:
        """Get age of operation in minutes"""
        return (time.time() - self.timestamp) / 60

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


class UndoManager:
    """
    Manages undo operations for destructive actions.

    Features:
    - Records state before destructive operations
    - Provides undo capability within time window
    - Persists undo history
    - Integrates with agent operations
    """

    def __init__(
        self,
        max_undo_history: int = 20,
        default_ttl_seconds: int = 3600,
        verbose: bool = False
    ):
        """
        Initialize undo manager.

        Args:
            max_undo_history: Maximum number of undo operations to keep
            default_ttl_seconds: Default TTL for undo operations
            verbose: Enable detailed logging
        """
        self.max_undo_history = max_undo_history
        self.default_ttl_seconds = default_ttl_seconds
        self.verbose = verbose

        # Undo history (most recent first)
        self.undo_stack: List[UndoSnapshot] = []

        # Undo handlers (operation_type -> handler function)
        self.undo_handlers: Dict[str, Callable] = {}

    def record_operation(
        self,
        operation_type: UndoableOperationType,
        agent_name: str,
        description: str,
        before_state: Dict[str, Any],
        undo_params: Dict[str, Any],
        operation_result: Optional[str] = None,
        ttl_seconds: Optional[int] = None
    ) -> str:
        """
        Record a destructive operation for potential undo.

        Args:
            operation_type: Type of operation
            agent_name: Agent that performed the operation
            description: Human-readable description
            before_state: State before operation (for display)
            undo_params: Parameters needed to undo
            operation_result: Result message from operation
            ttl_seconds: Custom TTL (uses default if None)

        Returns:
            operation_id for referencing this undo
        """
        operation_id = f"{agent_name}_{int(time.time() * 1000)}"

        snapshot = UndoSnapshot(
            operation_id=operation_id,
            operation_type=operation_type,
            agent_name=agent_name,
            timestamp=time.time(),
            description=description,
            before_state=before_state,
            undo_params=undo_params,
            operation_result=operation_result,
            ttl_seconds=ttl_seconds or self.default_ttl_seconds
        )

        # Add to stack (most recent first)
        self.undo_stack.insert(0, snapshot)

        # Trim history
        if len(self.undo_stack) > self.max_undo_history:
            self.undo_stack = self.undo_stack[:self.max_undo_history]

        if self.verbose:
            print(f"[UNDO] Recorded: {description} (ID: {operation_id})")

        return operation_id

    def register_undo_handler(
        self,
        operation_type: UndoableOperationType,
        handler: Callable
    ):
        """
        Register a handler function for undoing a specific operation type.

        Handler signature: async def handler(undo_params: Dict) -> str
        """
        self.undo_handlers[operation_type.value] = handler

        if self.verbose:
            print(f"[UNDO] Registered handler for {operation_type.value}")

    async def undo_operation(self, operation_id: str) -> str:
        """
        Undo a specific operation.

        Args:
            operation_id: ID of operation to undo

        Returns:
            Result message

        Raises:
            ValueError: If operation not found or cannot be undone
        """
        # Find the operation
        snapshot = None
        for snap in self.undo_stack:
            if snap.operation_id == operation_id:
                snapshot = snap
                break

        if not snapshot:
            raise ValueError(f"Operation {operation_id} not found in undo history")

        if not snapshot.can_undo:
            if snapshot.undone:
                raise ValueError(f"Operation {operation_id} has already been undone")
            if snapshot.is_expired:
                raise ValueError(
                    f"Operation {operation_id} undo window expired "
                    f"({snapshot.ttl_seconds}s TTL, {snapshot.age_minutes:.1f}m old)"
                )

        # Get handler
        handler = self.undo_handlers.get(snapshot.operation_type.value)
        if not handler:
            raise ValueError(f"No undo handler registered for {snapshot.operation_type.value}")

        # Execute undo
        try:
            result = await handler(snapshot.undo_params)

            # Mark as undone
            snapshot.undone = True
            snapshot.undone_at = time.time()

            if self.verbose:
                print(f"[UNDO] Successfully undid: {snapshot.description}")

            return result

        except Exception as e:
            raise RuntimeError(f"Failed to undo operation: {e}")

    def undo_last(self) -> Optional[str]:
        """
        Get operation ID of the most recent undoable operation.

        Returns:
            operation_id or None if no undoable operations
        """
        for snapshot in self.undo_stack:
            if snapshot.can_undo:
                return snapshot.operation_id
        return None

    def get_undoable_operations(
        self,
        agent_name: Optional[str] = None,
        operation_type: Optional[UndoableOperationType] = None,
        limit: int = 10
    ) -> List[UndoSnapshot]:
        """
        Get list of operations that can be undone.

        Args:
            agent_name: Filter by agent
            operation_type: Filter by operation type
            limit: Maximum number to return

        Returns:
            List of undoable operations (most recent first)
        """
        results = []

        for snapshot in self.undo_stack:
            if not snapshot.can_undo:
                continue

            if agent_name and snapshot.agent_name != agent_name:
                continue

            if operation_type and snapshot.operation_type != operation_type:
                continue

            results.append(snapshot)

            if len(results) >= limit:
                break

        return results

    def get_undo_summary(self, operation_id: str) -> Optional[str]:
        """Get human-readable summary of an undo operation"""
        for snapshot in self.undo_stack:
            if snapshot.operation_id == operation_id:
                status = "✓ Undone" if snapshot.undone else "Available"
                age = f"{snapshot.age_minutes:.1f}m ago"
                expires = f"expires in {(snapshot.ttl_seconds - (time.time() - snapshot.timestamp)) / 60:.0f}m"

                summary = f"{status} | {snapshot.description}\n"
                summary += f"  Agent: {snapshot.agent_name}\n"
                summary += f"  Time: {age} | {expires}\n"

                if snapshot.operation_result:
                    summary += f"  Result: {snapshot.operation_result[:100]}\n"

                return summary

        return None

    def format_undo_list(self, limit: int = 10) -> str:
        """Format list of undoable operations for display"""
        operations = self.get_undoable_operations(limit=limit)

        if not operations:
            return "No recent operations can be undone."

        lines = ["📋 **Recent Undoable Operations**:\n"]

        for i, snapshot in enumerate(operations, 1):
            age = f"{snapshot.age_minutes:.1f}m ago"
            lines.append(f"{i}. [{snapshot.agent_name}] {snapshot.description} ({age})")
            lines.append(f"   ID: `{snapshot.operation_id}`\n")

        lines.append("\nTo undo an operation, use: `/undo <operation_id>`")

        return "\n".join(lines)

    def cleanup_expired(self) -> int:
        """Remove expired undo operations from history"""
        original_count = len(self.undo_stack)

        self.undo_stack = [
            snap for snap in self.undo_stack
            if not snap.is_expired or snap.undone
        ]

        removed = original_count - len(self.undo_stack)

        if self.verbose and removed > 0:
            print(f"[UNDO] Cleaned up {removed} expired operation(s)")

        return removed

    def get_statistics(self) -> Dict[str, Any]:
        """Get undo statistics for monitoring"""
        total = len(self.undo_stack)
        undone = sum(1 for snap in self.undo_stack if snap.undone)
        available = sum(1 for snap in self.undo_stack if snap.can_undo)
        expired = sum(1 for snap in self.undo_stack if snap.is_expired and not snap.undone)

        # Count by operation type
        by_type = {}
        for snap in self.undo_stack:
            type_name = snap.operation_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1

        return {
            'total_operations': total,
            'undone': undone,
            'available_for_undo': available,
            'expired': expired,
            'by_type': by_type
        }

    def save_to_file(self, filepath: str):
        """Persist undo history to file"""
        data = {
            'snapshots': [snap.to_dict() for snap in self.undo_stack],
            'saved_at': time.time()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        if self.verbose:
            print(f"[UNDO] Saved {len(self.undo_stack)} operation(s) to {filepath}")

    def load_from_file(self, filepath: str):
        """Load undo history from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Reconstruct snapshots
            self.undo_stack = []
            for snap_dict in data.get('snapshots', []):
                # Convert string back to enum
                snap_dict['operation_type'] = UndoableOperationType(snap_dict['operation_type'])
                snapshot = UndoSnapshot(**snap_dict)
                self.undo_stack.append(snapshot)

            if self.verbose:
                print(f"[UNDO] Loaded {len(self.undo_stack)} operation(s) from {filepath}")

        except Exception as e:
            if self.verbose:
                print(f"[UNDO] Failed to load from {filepath}: {e}")
