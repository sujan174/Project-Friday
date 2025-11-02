"""
Interactive Action Editor - Rich editing experience for proposed actions

Provides an intuitive interface for users to:
- Review what will be executed
- Edit any parameters before execution
- Preview changes in real-time
- Validate edits before confirming
"""

import sys
from typing import Dict, List, Optional, Any
from orchestration.action_model import Action, ActionType, RiskLevel, FieldInfo


class Colors:
    """ANSI color codes"""
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'


class InteractiveActionEditor:
    """
    Rich interactive editor for action parameters.

    Features:
    - Multi-line text editing
    - Real-time validation
    - Clear before/after comparison
    - Helpful hints and examples
    - Undo edits
    """

    def __init__(self):
        self.colors_enabled = sys.stdout.isatty()

    def c(self, color: str, text: str) -> str:
        """Conditionally apply color"""
        if self.colors_enabled:
            return f"{color}{text}{Colors.ENDC}"
        return text

    def edit_action(self, action: Action) -> Optional[Dict[str, Any]]:
        """
        Main editing interface for an action.

        Shows:
        1. Action summary
        2. Current parameters
        3. Interactive editor for each field
        4. Confirmation of changes

        Returns:
            Dict of edits, or None if cancelled
        """
        self._clear_screen()
        self._show_action_header(action)

        # Collect editable fields
        editable_fields = {
            fname: finfo for fname, finfo in action.field_info.items()
            if finfo.editable
        }

        if not editable_fields:
            print(self.c(Colors.YELLOW, "\n⚠ This action has no editable fields\n"))
            input(self.c(Colors.DIM, "Press Enter to continue..."))
            return None

        edits = {}

        # Show current parameters first
        self._show_current_parameters(action, editable_fields)

        # Ask if user wants to edit
        print(f"\n{self.c(Colors.BOLD, 'Options:')}")
        print(f"  {self.c(Colors.CYAN, '[e]')} Edit parameters")
        print(f"  {self.c(Colors.GREEN, '[a]')} Approve as-is")
        print(f"  {self.c(Colors.RED, '[c]')} Cancel this action")

        choice = input(f"\n{self.c(Colors.BOLD, 'Your choice:')} ").strip().lower()

        if choice == 'a':
            return {}  # Approve with no edits
        elif choice == 'c':
            return None  # Cancel
        elif choice != 'e':
            return None  # Unknown, treat as cancel

        # Interactive field-by-field editor
        for field_name, field_info in editable_fields.items():
            edit_result = self._edit_field(field_name, field_info, action)

            if edit_result is not None:
                edits[field_name] = edit_result

        # Show summary of changes
        if edits:
            self._show_edit_summary(action, edits)

            confirm = input(
                f"\n{self.c(Colors.BOLD, 'Apply these changes?')} [y/n]: "
            ).strip().lower()

            if confirm == 'y':
                return edits

        return None

    def _edit_field(
        self,
        field_name: str,
        field_info: FieldInfo,
        action: Action
    ) -> Optional[Any]:
        """
        Edit a single field with rich interface.

        Returns:
            New value, or None if skipped
        """
        self._clear_screen()

        print(self.c(Colors.BOLD + Colors.CYAN, f"\n✏️  Editing: {field_info.display_label}"))
        print(self.c(Colors.DIM, "─" * 70))

        # Show description
        print(f"\n{self.c(Colors.BOLD, 'Description:')}")
        print(f"  {field_info.description}")

        # Show current value
        print(f"\n{self.c(Colors.BOLD, 'Current value:')}")
        current_val = field_info.current_value

        # Handle different field types
        if field_info.field_type == 'text':
            # Multi-line text (show first 3 lines)
            lines = str(current_val).split('\n')
            for i, line in enumerate(lines[:3]):
                print(f"  {self.c(Colors.DIM, str(i+1)+'.')} {line}")
            if len(lines) > 3:
                print(f"  {self.c(Colors.DIM, f'  ... ({len(lines)-3} more lines)')}")
        else:
            # Single line
            print(f"  {self.c(Colors.CYAN, str(current_val))}")

        # Show constraints/hints
        if field_info.constraints.allowed_values:
            print(f"\n{self.c(Colors.BOLD, 'Allowed values:')}")
            for val in field_info.constraints.allowed_values:
                print(f"  • {val}")

        if field_info.constraints.min_length or field_info.constraints.max_length:
            print(f"\n{self.c(Colors.BOLD, 'Length:')}")
            if field_info.constraints.min_length:
                print(f"  Minimum: {field_info.constraints.min_length} characters")
            if field_info.constraints.max_length:
                print(f"  Maximum: {field_info.constraints.max_length} characters")

        if field_info.examples:
            print(f"\n{self.c(Colors.BOLD, 'Examples:')}")
            for ex in field_info.examples[:3]:
                print(f"  • {ex}")

        # Get new value
        print(f"\n{self.c(Colors.DIM, '─' * 70)}")

        if field_info.field_type == 'text':
            print(self.c(Colors.YELLOW, "\nEnter new value (type 'END' on a new line when done, or 'SKIP' to keep current):"))
            new_value = self._read_multiline()

            if new_value == 'SKIP':
                return None
        else:
            new_value = input(
                f"{self.c(Colors.BOLD, 'New value')} (or Enter to skip): "
            ).strip()

            if not new_value:
                return None

        # Validate
        is_valid, error = field_info.constraints.validate(new_value)

        if not is_valid:
            print(f"\n{self.c(Colors.RED, '✗ Validation error:')} {error}")
            retry = input(f"{self.c(Colors.YELLOW, 'Try again?')} [y/n]: ").strip().lower()

            if retry == 'y':
                return self._edit_field(field_name, field_info, action)
            else:
                return None

        # Confirm change
        print(f"\n{self.c(Colors.GREEN, '✓ Valid!')}")
        print(f"\n{self.c(Colors.BOLD, 'Preview:')}")
        print(f"  Old: {self.c(Colors.DIM, str(current_val)[:100])}")
        print(f"  New: {self.c(Colors.CYAN, str(new_value)[:100])}")

        confirm = input(f"\n{self.c(Colors.BOLD, 'Keep this change?')} [y/n]: ").strip().lower()

        if confirm == 'y':
            return new_value
        else:
            return None

    def _read_multiline(self) -> str:
        """Read multi-line input until user types 'END'"""
        lines = []

        while True:
            try:
                line = input()
                if line == 'END':
                    break
                if line == 'SKIP':
                    return 'SKIP'
                lines.append(line)
            except EOFError:
                break

        return '\n'.join(lines)

    def _show_action_header(self, action: Action):
        """Show action overview"""
        risk_icon = {
            RiskLevel.HIGH: '🔴',
            RiskLevel.MEDIUM: '🟡',
            RiskLevel.LOW: '🟢'
        }.get(action.risk_level, '⚪')

        print("\n" + self.c(Colors.BOLD + Colors.CYAN, "=" * 70))
        print(self.c(
            Colors.BOLD,
            f"  {risk_icon} ACTION REVIEW: {action.action_type.value.upper()}"
        ))
        print(self.c(Colors.BOLD + Colors.CYAN, "=" * 70))

        print(f"\n{self.c(Colors.BOLD, 'Agent:')} {action.agent_name}")
        print(f"{self.c(Colors.BOLD, 'Type:')} {action.action_type.value}")
        print(f"{self.c(Colors.BOLD, 'Risk:')} {action.risk_level.value}")

        if action.reason_for_confirmation:
            print(f"\n{self.c(Colors.YELLOW, '⚠ Reason:')}")
            print(f"  {action.reason_for_confirmation}")

        # Show enriched details if available
        if hasattr(action, 'details') and action.details:
            details = action.details

            if 'description' in details:
                print(f"\n{self.c(Colors.BOLD, 'What will happen:')}")
                print(f"  {details['description']}")

            if 'channel' in details:
                print(f"  Channel: {self.c(Colors.CYAN, details['channel'])}")

            if 'project' in details:
                print(f"  Project: {self.c(Colors.CYAN, details['project'])}")

        print(self.c(Colors.DIM, "\n" + "─" * 70))

    def _show_current_parameters(self, action: Action, editable_fields: Dict):
        """Show all current parameter values"""
        print(f"\n{self.c(Colors.BOLD, 'Current Parameters:')}\n")

        for fname, finfo in editable_fields.items():
            value = finfo.current_value

            # Truncate long values
            if isinstance(value, str) and len(value) > 60:
                display_val = value[:57] + "..."
            else:
                display_val = str(value)

            print(f"  {self.c(Colors.BOLD, finfo.display_label + ':')} {display_val}")

        print()

    def _show_edit_summary(self, action: Action, edits: Dict[str, Any]):
        """Show summary of all changes"""
        self._clear_screen()

        print(self.c(Colors.BOLD + Colors.GREEN, "\n✓ EDIT SUMMARY"))
        print(self.c(Colors.DIM, "─" * 70))

        for field_name, new_value in edits.items():
            field_info = action.field_info[field_name]
            old_value = field_info.current_value

            print(f"\n{self.c(Colors.BOLD, field_info.display_label + ':')}")
            print(f"  {self.c(Colors.DIM, 'Was:')} {str(old_value)[:60]}")
            print(f"  {self.c(Colors.CYAN, 'Now:')} {str(new_value)[:60]}")

    def _clear_screen(self):
        """Clear terminal screen (optional)"""
        # Don't actually clear - just add spacing for better UX
        print("\n" * 2)


class SimpleReviewUI:
    """
    Simplified review interface for actions that show you exactly what will happen.
    """

    def __init__(self):
        self.editor = InteractiveActionEditor()

    def review_action(self, action: Action) -> tuple:
        """
        Review a single action with option to edit.

        Returns:
            (approved: bool, edits: Optional[Dict])
        """
        # Show the action in detail
        self._display_action_detail(action)

        # Get user decision
        while True:
            print(f"\n{self.editor.c(Colors.BOLD, 'What would you like to do?')}")
            print(f"  {self.editor.c(Colors.GREEN, '[a]')} Approve and execute")
            print(f"  {self.editor.c(Colors.CYAN, '[e]')} Edit before executing")
            print(f"  {self.editor.c(Colors.RED, '[c]')} Cancel this action")

            choice = input(f"\n{self.editor.c(Colors.BOLD, 'Choice:')} ").strip().lower()

            if choice == 'a':
                return True, None

            elif choice == 'e':
                edits = self.editor.edit_action(action)
                if edits is not None:
                    return True, edits
                # If edit was cancelled, return to menu
                self._display_action_detail(action)

            elif choice == 'c':
                return False, None

            else:
                print(self.editor.c(Colors.YELLOW, "Invalid choice. Please use a, e, or c"))

    def _display_action_detail(self, action: Action):
        """Display full action details in readable format"""
        self.editor._clear_screen()
        self.editor._show_action_header(action)

        print(f"\n{self.editor.c(Colors.BOLD, '📋 FULL DETAILS:')}\n")

        # Show all parameters (editable and non-editable)
        for fname, finfo in action.field_info.items():
            editable_marker = " ✏️" if finfo.editable else " 📌"

            print(f"{self.editor.c(Colors.BOLD, finfo.display_label + editable_marker)}")

            value = finfo.current_value

            # Format multi-line values
            if isinstance(value, str) and '\n' in value:
                print(f"  {self.editor.c(Colors.DIM, '─' * 60)}")
                for line in value.split('\n')[:10]:  # Show first 10 lines
                    print(f"  {line}")
                if value.count('\n') > 10:
                    print(f"  {self.editor.c(Colors.DIM, f'... ({value.count(chr(10)) - 10} more lines)')}")
                print(f"  {self.editor.c(Colors.DIM, '─' * 60)}")
            else:
                print(f"  {self.editor.c(Colors.CYAN, str(value))}")

            # Show description
            if finfo.description:
                print(f"  {self.editor.c(Colors.DIM, '(' + finfo.description + ')')}")

            print()

        print(self.editor.c(Colors.DIM, "─" * 70))
