"""
Professional Claude Code-inspired UI for the Multi-Agent Orchestration System

Features:
- Clean, minimal design
- Task-based progress tracking
- Real-time status updates
- Professional typography and spacing
- Subtle, purposeful use of color
"""

import sys
from datetime import datetime
from typing import Optional, Dict, List
from enum import Enum


class Status(Enum):
    """Task status indicators"""
    PENDING = "⏳"
    RUNNING = "⚙️"
    SUCCESS = "✓"
    ERROR = "✗"
    WARNING = "⚠"
    INFO = "ℹ"


class Color:
    """ANSI color codes - subtle and professional"""
    # Subtle colors
    GRAY = '\033[90m'
    DIM = '\033[2m'

    # Status colors
    GREEN = '\033[32m'
    RED = '\033[31m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    CYAN = '\033[36m'

    # Typography
    BOLD = '\033[1m'
    RESET = '\033[0m'

    @staticmethod
    def strip():
        """Remove all color codes from output"""
        return True  # Placeholder for future no-color mode


class ClaudeCodeUI:
    """
    Professional UI inspired by Claude Code's clean interface.

    Design principles:
    - Clarity over decoration
    - Progressive disclosure
    - Immediate feedback
    - Visual hierarchy
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.agent_call_count = 0
        self.active_tasks: Dict[str, dict] = {}
        self.indent_level = 0

    def _write(self, text: str, end: str = '\n'):
        """Write text to stdout"""
        sys.stdout.write(text + end)
        sys.stdout.flush()

    def _indent(self) -> str:
        """Get current indentation"""
        return "  " * self.indent_level

    def _timestamp(self) -> str:
        """Get formatted timestamp"""
        return datetime.now().strftime("%H:%M:%S")

    # ============================================================================
    # Session Management
    # ============================================================================

    def print_welcome(self, session_id: str):
        """Display welcome message"""
        self._write("")
        self._write(f"{Color.BOLD}Multi-Agent Orchestration System{Color.RESET}")
        self._write(f"{Color.GRAY}Session {session_id[:8]}...{Color.RESET}")
        self._write("")

    def print_agents_loaded(self, agents: List[dict]):
        """Display loaded agents in a clean format"""
        if not agents:
            return

        self._write(f"{Color.GREEN}{Status.SUCCESS.value} {len(agents)} agents ready{Color.RESET}")
        self._write("")

    def print_agents_failed(self, failed_count: int):
        """Display failed agents count"""
        if failed_count > 0:
            self._write(f"{Color.YELLOW}{Status.WARNING.value} {failed_count} agents unavailable{Color.RESET}")
            self._write("")

    # ============================================================================
    # User Interaction
    # ============================================================================

    def print_prompt(self):
        """Display input prompt"""
        self._write(f"{Color.BOLD}>{Color.RESET} ", end='')

    def print_divider(self, char: str = "─", length: int = 60):
        """Print a subtle divider"""
        self._write(f"{Color.GRAY}{char * length}{Color.RESET}")

    # ============================================================================
    # Task Progress
    # ============================================================================

    def start_thinking(self):
        """Indicate that processing has started"""
        # Silent - no output needed
        pass

    def start_agent_call(self, agent_name: str, instruction: str):
        """Start tracking an agent call"""
        self.agent_call_count += 1

        # Clean agent name
        display_name = agent_name.replace('_', ' ').title()

        # Truncate instruction if too long
        display_instruction = instruction[:70] + "..." if len(instruction) > 70 else instruction

        self._write("")
        self._write(f"{Color.BLUE}{Status.RUNNING.value}{Color.RESET} {Color.BOLD}{display_name}{Color.RESET}")
        self._write(f"  {Color.GRAY}{display_instruction}{Color.RESET}")

        # Store start time for duration calculation
        self.active_tasks[agent_name] = {
            'start': datetime.now(),
            'instruction': instruction
        }

    def end_agent_call(self, agent_name: str, success: bool, duration_ms: Optional[float] = None, message: Optional[str] = None):
        """Complete an agent call"""
        # Calculate duration if not provided
        if duration_ms is None and agent_name in self.active_tasks:
            start = self.active_tasks[agent_name]['start']
            duration_ms = (datetime.now() - start).total_seconds() * 1000

        # Clean up task
        if agent_name in self.active_tasks:
            del self.active_tasks[agent_name]

        # Display result
        if success:
            status_icon = f"{Color.GREEN}{Status.SUCCESS.value}{Color.RESET}"
            duration_str = f"{Color.GRAY}({duration_ms:.0f}ms){Color.RESET}" if duration_ms else ""
            self._write(f"  {status_icon} Completed {duration_str}")

            # Show success message if provided
            if message and self.verbose:
                # Truncate message
                display_msg = message[:100] + "..." if len(message) > 100 else message
                self._write(f"    {Color.GRAY}{display_msg}{Color.RESET}")
        else:
            status_icon = f"{Color.RED}{Status.ERROR.value}{Color.RESET}"
            self._write(f"  {status_icon} Failed")

            # Show error message
            if message:
                # Split into lines and indent
                for line in message.split('\n')[:3]:  # Only show first 3 lines
                    if line.strip():
                        self._write(f"    {Color.RED}{line}{Color.RESET}")

    def show_retry(self, agent_name: str, attempt: int, max_attempts: int):
        """Show retry indicator"""
        display_name = agent_name.replace('_', ' ').title()
        self._write(f"  {Color.YELLOW}{Status.WARNING.value}{Color.RESET} Retrying {display_name} ({attempt}/{max_attempts})")

    # ============================================================================
    # Response Display
    # ============================================================================

    def print_response(self, response: str):
        """Display the orchestrator's response with proper markdown formatting"""
        self._write("")

        # Parse and format the response
        lines = response.split('\n')
        in_list = False

        for line in lines:
            stripped = line.strip()

            # Skip empty lines at start
            if not stripped:
                if in_list:
                    in_list = False
                self._write("")
                continue

            # Headers (## Header)
            if stripped.startswith('##'):
                header_text = stripped[2:].strip()
                self._write(f"{Color.BOLD}{header_text}{Color.RESET}")
                self._write("")
                continue

            # Bold text (**text**)
            if '**' in line:
                # Replace **text** with bold formatting
                import re
                line = re.sub(r'\*\*([^*]+)\*\*', f'{Color.BOLD}\\1{Color.RESET}', line)

            # List items (- item or * item)
            if stripped.startswith('- ') or stripped.startswith('* '):
                in_list = True
                # Check indentation level
                indent = len(line) - len(line.lstrip())
                list_content = stripped[2:]

                # Clean up bold markers if still present
                list_content = list_content.replace('**', '')

                if indent == 0:
                    self._write(f"  • {list_content}")
                else:
                    spaces = "  " * (indent // 2 + 1)
                    self._write(f"{spaces}• {list_content}")
                continue

            # Status lines with icons - keep as is
            if any(icon in line for icon in ['✓', '✗', '⚠', '●', '⚙️']):
                self._write(line)
                continue

            # Regular paragraphs
            # Clean up any remaining markdown
            text = stripped.replace('**', '').replace('`', '')
            if text:
                self._write(text)

        self._write("")

    # ============================================================================
    # Error Handling
    # ============================================================================

    def print_error(self, error: str):
        """Display an error message"""
        self._write("")
        self._write(f"{Color.RED}{Status.ERROR.value} Error{Color.RESET}")

        # Format error message
        lines = error.split('\n')
        for line in lines[:5]:  # Limit to 5 lines
            if line.strip():
                self._write(f"  {Color.RED}{line}{Color.RESET}")

        if len(lines) > 5:
            self._write(f"  {Color.GRAY}... ({len(lines) - 5} more lines){Color.RESET}")

        self._write("")

    def print_warning(self, warning: str):
        """Display a warning message"""
        self._write(f"{Color.YELLOW}{Status.WARNING.value} {warning}{Color.RESET}")

    # ============================================================================
    # Session End
    # ============================================================================

    def print_session_summary(self, stats: dict):
        """Display session summary"""
        self._write("")
        self.print_divider()
        self._write(f"{Color.BOLD}Session Summary{Color.RESET}")
        self._write("")

        # Format stats
        if 'duration' in stats:
            self._write(f"  Duration: {Color.GRAY}{stats['duration']}{Color.RESET}")

        if 'message_count' in stats:
            self._write(f"  Messages: {Color.GRAY}{stats['message_count']}{Color.RESET}")

        if 'agent_calls' in stats:
            self._write(f"  Agent calls: {Color.GRAY}{stats['agent_calls']}{Color.RESET}")

        if 'success_rate' in stats and stats['success_rate'] != "N/A":
            rate = stats['success_rate']
            color = Color.GREEN if rate > 80 else Color.YELLOW if rate > 50 else Color.RED
            self._write(f"  Success rate: {color}{rate}%{Color.RESET}")

        self.print_divider()
        self._write("")

    def print_goodbye(self):
        """Display goodbye message"""
        self._write("")
        self._write(f"{Color.GRAY}Goodbye!{Color.RESET}")
        self._write("")

    # ============================================================================
    # Intelligence/Debug Info
    # ============================================================================

    def print_intelligence_info(self, info: dict):
        """Display intelligence analysis info (only in verbose mode)"""
        if not self.verbose:
            return

        self._write(f"{Color.GRAY}Intelligence: {info.get('method', 'hybrid')} "
                   f"({info.get('latency_ms', 0):.0f}ms, "
                   f"confidence: {info.get('confidence', 0):.2f}){Color.RESET}")

    def print_debug(self, message: str):
        """Print debug message (only in verbose mode)"""
        if self.verbose:
            self._write(f"{Color.GRAY}[DEBUG] {message}{Color.RESET}")

    # ============================================================================
    # Progress Indicators
    # ============================================================================

    def show_progress(self, message: str):
        """Show a progress message"""
        self._write(f"{Color.GRAY}⋯ {message}{Color.RESET}")

    def show_operation_complete(self, count: int):
        """Show operations completed"""
        if count > 0:
            self._write("")
            self._write(f"{Color.GREEN}{Status.SUCCESS.value} Completed {count} operation(s){Color.RESET}")
