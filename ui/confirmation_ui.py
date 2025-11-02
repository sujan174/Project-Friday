"""
Confirmation UI: Terminal interface for batch action confirmation.

Presents multiple pending actions for user review, approval, rejection, and editing.
"""

from typing import Dict, List, Optional, Tuple
from orchestration.action_model import Action, ActionStatus, RiskLevel
from ui.interactive_editor import InteractiveActionEditor


# ANSI Color codes
class Colors:
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'


class ConfirmationUI:
    """
    Rich terminal UI for confirming batches of actions.
    Supports viewing, editing, and decision-making.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.editor = InteractiveActionEditor()  # Rich interactive editor

    def present_batch(self, actions: List[Action]) -> None:
        """
        Display a batch of actions waiting for confirmation.
        Shows all actions with their details and edit options.
        """
        if not actions:
            return

        print(f"\n{Colors.YELLOW}{'='*70}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.CYAN}📋 REVIEW PHASE - {len(actions)} action(s) pending{Colors.ENDC}")
        print(f"{Colors.YELLOW}{'='*70}{Colors.ENDC}\n")

        for i, action in enumerate(actions, 1):
            self._display_action_summary(action, i, len(actions))

        print(f"{Colors.YELLOW}{'='*70}{Colors.ENDC}")

    def _display_action_summary(self, action: Action, index: int, total: int) -> None:
        """Display summary of a single action with enriched context"""

        risk_color = self._risk_color(action.risk_level)

        print(f"{Colors.BOLD}[{index}/{total}] {action.action_type.value.upper()}{Colors.ENDC}")
        print(f"  Agent: {action.agent_name}")
        print(f"  Risk: {risk_color}{action.risk_level.value.upper()}{Colors.ENDC}")
        print(f"  {action.reason_for_confirmation}")

        # Show enriched context details if available
        if hasattr(action, 'details') and action.details:
            print(f"\n  {Colors.CYAN}📋 What will happen:{Colors.ENDC}")
            details = action.details

            # Show description
            if 'description' in details:
                print(f"     {details['description']}")

            # Show issue keys for delete operations
            if 'issue_keys' in details:
                keys_str = ', '.join(details['issue_keys'][:5])  # Show first 5
                if len(details['issue_keys']) > 5:
                    keys_str += f" + {len(details['issue_keys']) - 5} more"
                print(f"     Issues: {keys_str}")

            # Show message preview for Slack
            if 'message_preview' in details:
                print(f"     Message: {details['message_preview']}")

            # Show channel
            if 'channel' in details:
                print(f"     Channel: {details['channel']}")

            # Show project for Jira
            if 'project' in details:
                print(f"     Project: {details['project']}")

            # Show summary/title
            if 'summary' in details:
                print(f"     Summary: {details['summary']}")

        # Show editable fields if any
        editable_fields = [f for f in action.field_info if action.field_info[f].editable]
        if editable_fields:
            print(f"\n  {Colors.CYAN}✏️  Can edit:{Colors.ENDC}")
            for field_name in editable_fields:
                field_info = action.field_info[field_name]
                current_val = field_info.current_value
                if isinstance(current_val, str) and len(str(current_val)) > 40:
                    current_val = str(current_val)[:37] + "..."
                print(f"     • {field_info.display_label}: {current_val}")

        # Show read-only fields
        readonly_fields = [f for f in action.field_info if not action.field_info[f].editable]
        if readonly_fields:
            print(f"\n  {Colors.YELLOW}📌 Fixed:{Colors.ENDC}")
            for field_name in readonly_fields:
                field_info = action.field_info[field_name]
                print(f"     • {field_info.display_label}: {field_info.current_value}")

        print()

    def collect_decisions(self, actions: List[Action]) -> Dict[str, any]:
        """
        Interactive decision collection for batch.
        Returns decisions dict with confirmations, rejections, and edits.
        """

        decisions = {
            'confirmed': {},      # action_id -> True
            'rejected': {},       # action_id -> reason
            'edited': {}          # action_id -> {field: new_value}
        }

        while True:
            try:
                user_input = input(
                    f"{Colors.BOLD}Command: [a]pprove-all | [e]dit #N | [r]eview | [c]ancel{Colors.ENDC}\n> "
                ).strip().lower()

                if user_input == 'a':
                    # Approve all
                    for action in actions:
                        decisions['confirmed'][action.id] = True
                    print(f"\n{Colors.GREEN}✓ All actions approved!{Colors.ENDC}\n")
                    break

                elif user_input == 'r':
                    # Show batch again
                    self.present_batch(actions)

                elif user_input.startswith('e'):
                    # Edit specific action: "e 1" or just "e" to get prompt
                    parts = user_input.split()
                    action_num = None

                    if len(parts) > 1:
                        try:
                            action_num = int(parts[1])
                        except ValueError:
                            print(f"{Colors.YELLOW}Please use 'e 1' or 'e 2'{Colors.ENDC}")
                            continue
                    else:
                        # Ask which action to edit
                        try:
                            action_num = int(input(f"{Colors.BOLD}Edit action #: {Colors.ENDC}"))
                        except ValueError:
                            print(f"{Colors.YELLOW}Please enter a number{Colors.ENDC}")
                            continue

                    if 1 <= action_num <= len(actions):
                        edits = self._edit_action_interactive(actions[action_num - 1])
                        if edits:
                            decisions['edited'][actions[action_num - 1].id] = edits
                            print(f"{Colors.GREEN}✓ Action #{action_num} updated{Colors.ENDC}\n")
                    else:
                        print(f"{Colors.YELLOW}Invalid action number (1-{len(actions)}){Colors.ENDC}")

                elif user_input == 'c':
                    # Cancel all
                    for action in actions:
                        decisions['rejected'][action.id] = 'User cancelled'
                    print(f"\n{Colors.RED}✗ All actions cancelled.{Colors.ENDC}\n")
                    break

                else:
                    print(f"{Colors.YELLOW}Unknown command. Use: a, e, r, or c{Colors.ENDC}")

            except KeyboardInterrupt:
                print(f"\n{Colors.RED}✗ Cancelled by user (Ctrl+C){Colors.ENDC}\n")
                for action in actions:
                    decisions['rejected'][action.id] = 'User cancelled (Ctrl+C)'
                break

        return decisions

    def _edit_action_interactive(self, action: Action) -> Optional[Dict[str, any]]:
        """
        Let user edit a single action's parameters using the rich interactive editor.
        Returns dict of field edits, or None if cancelled/skipped.
        """
        # Use the rich interactive editor
        return self.editor.edit_action(action)

    def show_action_with_edits(self, action: Action) -> None:
        """Show an action with user's edits highlighted"""
        print(f"\n{Colors.CYAN}{'─'*60}{Colors.ENDC}")
        print(f"{Colors.BOLD}{action.action_type.value.upper()} (Agent: {action.agent_name}){Colors.ENDC}")

        if action.user_edits:
            print(f"{Colors.GREEN}✏️  Your edits:{Colors.ENDC}")
            for field_name, old_val in action.field_info.items():
                if field_name in action.user_edits:
                    new_val = action.user_edits[field_name]
                    print(f"  {old_val.display_label}:")
                    print(f"    Was: {old_val.current_value}")
                    print(f"    Now: {new_val}")

        print(f"{Colors.CYAN}{'─'*60}{Colors.ENDC}\n")

    def _risk_color(self, risk_level: RiskLevel) -> str:
        """Get color for risk level"""
        if risk_level == RiskLevel.HIGH:
            return Colors.RED
        elif risk_level == RiskLevel.MEDIUM:
            return Colors.YELLOW
        else:
            return Colors.GREEN


class ConfirmationModal:
    """Simpler UI for single confirmation (for low-risk operations)"""

    @staticmethod
    def ask(message: str, action_type: str = "Proceed") -> bool:
        """Simple yes/no confirmation"""
        while True:
            response = input(f"\n{message}\n{Colors.BOLD}{action_type}? [y/n]: {Colors.ENDC}").strip().lower()
            if response in ['yes', 'y']:
                return True
            elif response in ['no', 'n']:
                return False
            else:
                print(f"{Colors.YELLOW}Please answer y or n{Colors.ENDC}")
