"""
Message Confirmation System

Mandatory human-in-the-loop for Slack messages and Notion content.
Users can:
- Preview the message/content
- Edit it manually
- Ask agent to modify it
- Approve or reject

Author: AI System
Version: 1.0
"""

import sys
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Import config for toggleable confirmations
try:
    from config import Config
except ImportError:
    # Fallback if config not available
    class Config:
        CONFIRM_SLACK_MESSAGES = True
        CONFIRM_JIRA_OPERATIONS = True


class ConfirmationDecision(str, Enum):
    """User's decision on a message/content"""
    APPROVED = "approved"
    REJECTED = "rejected"
    EDIT_MANUAL = "edit_manual"  # User will edit manually
    EDIT_AI = "edit_ai"  # Ask AI to modify


@dataclass
class MessagePreview:
    """Preview of a message/content before sending"""
    agent_name: str  # slack, notion, etc.
    operation_type: str  # "send_message", "create_page", etc.
    destination: str  # channel name, page title, etc.
    content: str  # The actual message/content
    metadata: Dict[str, Any]  # Additional context

    def format_preview(self) -> str:
        """Format for display to user"""
        lines = [
            f"\n{'='*70}",
            f"📝 **{self.agent_name.upper()} - {self.operation_type}**",
            f"{'='*70}\n",
            f"**Destination:** {self.destination}\n",
            f"**Content Preview:**",
            f"{'-'*70}",
            self.content,
            f"{'-'*70}\n"
        ]

        # Add metadata if present
        if self.metadata:
            lines.append("**Additional Info:**")
            for key, value in self.metadata.items():
                lines.append(f"  • {key}: {value}")
            lines.append("")

        return "\n".join(lines)


class MessageConfirmation:
    """
    Handles mandatory confirmation for Slack/Notion operations.

    Features:
    - Shows preview of message/content
    - Allows manual editing
    - Allows AI-assisted editing
    - Enforces human approval
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def confirm_slack_message(
        self,
        channel: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[ConfirmationDecision, Optional[str]]:
        """
        Confirm Slack message before sending.

        Args:
            channel: Target Slack channel
            message: Message content
            metadata: Additional context (thread_ts, etc.)

        Returns:
            (decision, modified_message)
        """
        preview = MessagePreview(
            agent_name="Slack",
            operation_type="Send Message",
            destination=channel,
            content=message,
            metadata=metadata or {}
        )

        return self._confirm_with_edit(preview)

    def confirm_notion_operation(
        self,
        operation_type: str,
        page_title: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[ConfirmationDecision, Optional[str]]:
        """
        Confirm Notion operation before executing.

        Args:
            operation_type: "Create Page", "Update Page", etc.
            page_title: Title of the page
            content: Page content
            metadata: Additional context

        Returns:
            (decision, modified_content)
        """
        preview = MessagePreview(
            agent_name="Notion",
            operation_type=operation_type,
            destination=page_title,
            content=content,
            metadata=metadata or {}
        )

        return self._confirm_with_edit(preview)

    def _confirm_with_edit(
        self,
        preview: MessagePreview
    ) -> Tuple[ConfirmationDecision, Optional[str]]:
        """
        Main confirmation flow with edit capability.

        Returns:
            (decision, modified_content)
        """
        modified_content = preview.content

        while True:
            # Show preview
            print(preview.format_preview())

            # Show options
            print("**Options:**")
            print("  [a] Approve and send")
            print("  [e] Edit manually")
            print("  [m] Ask AI to modify")
            print("  [r] Reject (don't send)")
            print()

            try:
                choice = input("Your decision [a/e/m/r]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n❌ Cancelled by user")
                return ConfirmationDecision.REJECTED, None

            if choice == 'a':
                # Approved
                print("✅ Approved!")
                return ConfirmationDecision.APPROVED, modified_content

            elif choice == 'e':
                # Manual edit
                print("\n📝 Edit the message below. Press Ctrl+D (Unix) or Ctrl+Z then Enter (Windows) when done:")
                print(f"{'-'*70}")
                print(modified_content)
                print(f"{'-'*70}")

                # Multi-line input
                edited_lines = []
                try:
                    while True:
                        line = input()
                        edited_lines.append(line)
                except (EOFError, KeyboardInterrupt):
                    pass

                if edited_lines:
                    modified_content = '\n'.join(edited_lines)
                    preview.content = modified_content
                    print("\n✅ Message updated! Review below:\n")
                    # Loop back to show updated preview
                else:
                    print("⚠️ No changes made")

            elif choice == 'm':
                # AI-assisted modification
                print("\n🤖 What changes would you like the AI to make?")
                modification_request = input("Your request: ").strip()

                if modification_request:
                    # Return for AI to process
                    return ConfirmationDecision.EDIT_AI, modification_request
                else:
                    print("⚠️ No modification requested")

            elif choice == 'r':
                # Rejected
                print("❌ Rejected - message will not be sent")
                return ConfirmationDecision.REJECTED, None

            else:
                print("⚠️ Invalid choice. Please enter a, e, m, or r")

    def confirm_bulk_messages(
        self,
        messages: List[Tuple[str, str, Dict]]  # (channel, message, metadata)
    ) -> List[Tuple[str, str, bool]]:
        """
        Confirm multiple messages at once.

        Args:
            messages: List of (channel, message, metadata) tuples

        Returns:
            List of (channel, final_message, approved) tuples
        """
        results = []

        print(f"\n🔔 **{len(messages)} messages require confirmation**\n")

        for i, (channel, message, metadata) in enumerate(messages, 1):
            print(f"\n{'='*70}")
            print(f"Message {i}/{len(messages)}")
            print(f"{'='*70}")

            decision, modified = self.confirm_slack_message(
                channel=channel,
                message=message,
                metadata=metadata
            )

            if decision == ConfirmationDecision.APPROVED:
                results.append((channel, modified, True))
            elif decision == ConfirmationDecision.EDIT_AI:
                # AI modification requested - mark for reprocessing
                results.append((channel, modified, False))  # False = needs reprocessing
            else:
                # Rejected
                results.append((channel, message, False))

        return results


class MandatoryConfirmationEnforcer:
    """
    Enforces mandatory confirmation for Slack and Notion.

    This is a safety layer that ensures NO Slack message or Notion
    operation executes without human approval.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.confirmer = MessageConfirmation(verbose=verbose)

        # Track operations that require confirmation
        self.pending_confirmations: List[Dict[str, Any]] = []

    def requires_confirmation(self, agent_name: str, instruction: str) -> bool:
        """
        Check if this operation requires confirmation.

        Args:
            agent_name: Name of the agent
            instruction: The instruction being executed

        Returns:
            True if confirmation is required
        """
        agent_lower = agent_name.lower()

        # Slack: Only confirm WRITE operations (sending messages, not reading)
        if agent_lower == 'slack' and Config.CONFIRM_SLACK_MESSAGES:
            # Write operations (need confirmation)
            write_keywords = ['send', 'post', 'message to', 'notify', 'announce', 'reply', 'react', 'delete']
            # Read operations (no confirmation needed)
            read_keywords = ['list', 'get', 'search', 'find', 'show', 'view', 'read', 'channels', 'users']

            # If it's a read operation, don't confirm
            if any(read_kw in instruction.lower() for read_kw in read_keywords):
                return False

            # If it's a write operation, confirm
            if any(write_kw in instruction.lower() for write_kw in write_keywords):
                return True

        # Notion: Only confirm WRITE operations (not reads)
        if agent_lower == 'notion':
            # Write operations (need confirmation)
            write_keywords = ['create', 'add', 'update', 'write', 'insert', 'delete', 'edit']
            # Read operations (no confirmation needed)
            read_keywords = ['get', 'search', 'list', 'find', 'show', 'view', 'read', 'pages', 'database']

            # If it's a read operation, don't confirm
            if any(read_kw in instruction.lower() for read_kw in read_keywords):
                return False

            # If it's a write operation, confirm
            if any(write_kw in instruction.lower() for write_kw in write_keywords):
                return True

        # Jira: Only confirm WRITE operations (not reads)
        if agent_lower == 'jira' and Config.CONFIRM_JIRA_OPERATIONS:
            # Only confirm write operations
            write_keywords = ['create', 'update', 'delete', 'transition', 'assign', 'add comment', 'close', 'edit']
            # Exclude read operations
            read_keywords = ['get', 'search', 'list', 'find', 'show', 'view', 'assigned to me', 'my tasks']

            # If it's a read operation, don't confirm
            if any(read_kw in instruction.lower() for read_kw in read_keywords):
                return False

            # If it's a write operation, confirm
            if any(write_kw in instruction.lower() for write_kw in write_keywords):
                return True

        return False

    def extract_message_content(
        self,
        agent_name: str,
        instruction: str
    ) -> Optional[Tuple[str, str, Dict]]:
        """
        Extract message content from instruction.

        Args:
            agent_name: Name of the agent
            instruction: The instruction

        Returns:
            (destination, content, metadata) or None
        """
        agent_lower = agent_name.lower()
        instruction_lower = instruction.lower()

        if agent_lower == 'slack':
            # Try to extract channel and message
            import re

            # Pattern: "send/post ... to #channel"
            channel_match = re.search(r'#([\w\-]+)', instruction)
            channel = channel_match.group(1) if channel_match else "unknown"

            # Extract message content (everything after keywords)
            message_patterns = [
                r'message[:\s]+"([^"]+)"',
                r'message[:\s]+\'([^\']+)\'',
                r'send[:\s]+"([^"]+)"',
                r'send[:\s]+\'([^\']+)\'',
            ]

            message = None
            for pattern in message_patterns:
                match = re.search(pattern, instruction, re.IGNORECASE)
                if match:
                    message = match.group(1)
                    break

            if not message:
                # Fallback: use instruction as message
                message = instruction

            return (channel, message, {})

        elif agent_lower == 'notion':
            # Try to extract page title and content
            import re

            # Pattern: "create page ... titled/named ..."
            title_match = re.search(r'titled?[:\s]+"([^"]+)"', instruction, re.IGNORECASE)
            if not title_match:
                title_match = re.search(r'named?[:\s]+"([^"]+)"', instruction, re.IGNORECASE)

            title = title_match.group(1) if title_match else "New Page"

            # Content is harder to extract, use full instruction
            content = instruction

            return (title, content, {})

        return None

    def confirm_before_execution(
        self,
        agent_name: str,
        instruction: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Show confirmation prompt before execution.

        Args:
            agent_name: Name of the agent
            instruction: The instruction to execute

        Returns:
            (should_execute, modified_instruction)
        """
        agent_lower = agent_name.lower()

        # Extract message content
        extracted = self.extract_message_content(agent_name, instruction)
        if not extracted:
            # Couldn't extract, show generic confirmation
            return self._generic_confirmation(agent_name, instruction)

        destination, content, metadata = extracted

        # Show confirmation based on agent type
        if agent_lower == 'slack':
            decision, modified = self.confirmer.confirm_slack_message(
                channel=destination,
                message=content,
                metadata=metadata
            )

        elif agent_lower == 'notion':
            operation = "Create Page" if "create" in instruction.lower() else "Update Page"
            decision, modified = self.confirmer.confirm_notion_operation(
                operation_type=operation,
                page_title=destination,
                content=content,
                metadata=metadata
            )

        else:
            return self._generic_confirmation(agent_name, instruction)

        # Process decision
        if decision == ConfirmationDecision.APPROVED:
            # Rebuild instruction with modified content if changed
            if modified and modified != content:
                modified_instruction = self._rebuild_instruction(
                    agent_name, instruction, content, modified
                )
                return True, modified_instruction
            return True, instruction

        elif decision == ConfirmationDecision.EDIT_AI:
            # User wants AI to modify - return modification request
            # The orchestrator should handle this by asking the agent to revise
            return False, f"[AI_MODIFICATION_REQUESTED] {modified}"

        else:
            # Rejected
            return False, None

    def _rebuild_instruction(
        self,
        agent_name: str,
        original_instruction: str,
        old_content: str,
        new_content: str
    ) -> str:
        """Rebuild instruction with modified content"""
        # Simple replacement
        return original_instruction.replace(old_content, new_content)

    def _generic_confirmation(
        self,
        agent_name: str,
        instruction: str
    ) -> Tuple[bool, Optional[str]]:
        """Generic confirmation when content can't be extracted"""
        print(f"\n{'='*70}")
        print(f"⚠️ **{agent_name.upper()} OPERATION REQUIRES CONFIRMATION**")
        print(f"{'='*70}")
        print(f"\n**Instruction:**")
        print(instruction)
        print()

        try:
            choice = input("Approve this operation? [y/n]: ").strip().lower()
            if choice == 'y':
                print("✅ Approved!")
                return True, instruction
            else:
                print("❌ Rejected!")
                return False, None
        except (EOFError, KeyboardInterrupt):
            print("\n❌ Cancelled")
            return False, None
