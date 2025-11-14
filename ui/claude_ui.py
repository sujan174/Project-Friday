"""
Claude Code-style Terminal UI

Elegant, minimal interface that looks EXACTLY like Claude Code.
No boxes, no panels, just clean formatted text.
"""

import sys
import re
from typing import Optional, List
from datetime import datetime


class Colors:
    """ANSI color codes - very subtle"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    # Subtle colors for minimal UI
    GRAY = '\033[90m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'


class ClaudeUI:
    """
    Elegant terminal UI inspired by Claude Code.

    Design Philosophy:
    - Minimal decoration
    - Clean typography
    - Subtle colors
    - Maximum clarity
    - No boxes or panels
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.agent_call_count = 0

    def _write(self, text: str = "", end: str = '\n'):
        """Write to stdout"""
        sys.stdout.write(text + end)
        sys.stdout.flush()

    # ============================================================================
    # Markdown Rendering (Clean, No Boxes)
    # ============================================================================

    def print_markdown(self, text: str):
        """
        Render markdown in a clean, minimal style.
        No boxes, no panels - just formatted text like Claude Code.
        """
        lines = text.split('\n')
        in_code_block = False
        code_language = None

        i = 0
        while i < len(lines):
            line = lines[i]

            # Code blocks
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                if in_code_block:
                    code_language = line.strip()[3:].strip()
                    self._write()  # Blank line before code
                else:
                    self._write()  # Blank line after code
                    code_language = None
                i += 1
                continue

            if in_code_block:
                # Render code with subtle indentation and dim color
                self._write(f"{Colors.DIM}  {line}{Colors.RESET}")
                i += 1
                continue

            # Skip empty lines but preserve spacing
            if not line.strip():
                self._write()
                i += 1
                continue

            # Headers
            if line.startswith('# '):
                self._write()
                self._write(f"{Colors.BOLD}{line[2:]}{Colors.RESET}")
                self._write()
                i += 1
                continue

            if line.startswith('## '):
                self._write()
                self._write(f"{Colors.BOLD}{line[3:]}{Colors.RESET}")
                i += 1
                continue

            if line.startswith('### '):
                self._write(f"{Colors.BOLD}{line[4:]}{Colors.RESET}")
                i += 1
                continue

            # Lists
            if line.strip().startswith('- ') or line.strip().startswith('* '):
                indent = len(line) - len(line.lstrip())
                content = line.strip()[2:]
                # Format inline elements
                content = self._format_inline(content)
                spaces = ' ' * indent
                self._write(f"{spaces}• {content}")
                i += 1
                continue

            # Numbered lists
            if re.match(r'^\d+\.\s', line.strip()):
                content = re.sub(r'^\d+\.\s', '', line.strip())
                content = self._format_inline(content)
                number = re.match(r'^(\d+)\.', line.strip()).group(1)
                self._write(f"{number}. {content}")
                i += 1
                continue

            # Blockquotes
            if line.strip().startswith('>'):
                content = line.strip()[1:].strip()
                content = self._format_inline(content)
                self._write(f"{Colors.DIM}  {content}{Colors.RESET}")
                i += 1
                continue

            # Regular paragraphs
            formatted = self._format_inline(line.strip())
            self._write(formatted)
            i += 1

        self._write()  # Final blank line

    def _format_inline(self, text: str) -> str:
        """Format inline markdown elements (bold, italic, code)"""
        # Bold: **text**
        text = re.sub(
            r'\*\*([^\*]+)\*\*',
            f'{Colors.BOLD}\\1{Colors.RESET}',
            text
        )

        # Italic: *text* or _text_
        text = re.sub(
            r'(?<!\*)\*(?!\*)([^\*]+)\*(?!\*)',
            f'{Colors.DIM}\\1{Colors.RESET}',
            text
        )
        text = re.sub(
            r'_([^_]+)_',
            f'{Colors.DIM}\\1{Colors.RESET}',
            text
        )

        # Inline code: `code`
        text = re.sub(
            r'`([^`]+)`',
            f'{Colors.GRAY}\\1{Colors.RESET}',
            text
        )

        return text

    # ============================================================================
    # Session Management
    # ============================================================================

    def print_welcome(self, session_id: str):
        """Display minimal welcome"""
        self._write()
        self._write(f"{Colors.BOLD}Multi-Agent Orchestration System{Colors.RESET}")
        self._write(f"{Colors.GRAY}Session {session_id[:8]}{Colors.RESET}")
        self._write()

    def print_agents_loaded(self, agents: List[dict]):
        """Display loaded agents count"""
        if not agents:
            return
        self._write(f"{Colors.GREEN}✓{Colors.RESET} {len(agents)} agents ready")
        self._write()

    def print_agents_failed(self, failed_count: int):
        """Display failed agents count"""
        if failed_count > 0:
            self._write(f"{Colors.YELLOW}⚠{Colors.RESET} {failed_count} agents unavailable")
            self._write()

    # ============================================================================
    # User Interaction
    # ============================================================================

    def print_prompt(self):
        """Display clean prompt"""
        self._write(f"{Colors.BOLD}{Colors.CYAN}>{Colors.RESET} ", end='')

    def print_divider(self):
        """Print subtle divider"""
        self._write(f"{Colors.GRAY}{'─' * 60}{Colors.RESET}")

    # ============================================================================
    # Agent Operations
    # ============================================================================

    def start_thinking(self):
        """Silent - no visual indicator needed"""
        pass

    def start_agent_call(self, agent_name: str, instruction: str):
        """Show agent starting - very minimal"""
        self.agent_call_count += 1

        # Only show in verbose mode, and very subtly
        if self.verbose:
            display_name = agent_name.replace('_', ' ').title()
            self._write()
            self._write(f"{Colors.GRAY}→ {display_name}...{Colors.RESET}")

    def end_agent_call(
        self,
        agent_name: str,
        success: bool,
        duration_ms: Optional[float] = None,
        message: Optional[str] = None
    ):
        """
        Show agent completion - very minimal.
        In non-verbose mode, show nothing (let the response speak for itself).
        In verbose mode, show a subtle completion message.
        """
        if not self.verbose:
            return

        if success:
            duration = f" {duration_ms:.0f}ms" if duration_ms else ""
            self._write(f"{Colors.GRAY}  ✓{duration}{Colors.RESET}")
        else:
            self._write(f"{Colors.RED}  ✗ Failed{Colors.RESET}")
            if message:
                # Show first line of error only
                first_line = message.split('\n')[0]
                if first_line.strip():
                    self._write(f"{Colors.RED}  {first_line[:80]}{Colors.RESET}")

    def show_retry(self, agent_name: str, attempt: int, max_attempts: int):
        """Show retry - only in verbose mode"""
        if self.verbose:
            self._write(f"{Colors.GRAY}  ↻ Retry {attempt}/{max_attempts}{Colors.RESET}")

    # ============================================================================
    # Response Display
    # ============================================================================

    def print_response(self, response: str):
        """Display response with clean markdown"""
        self._write()
        self.print_markdown(response)

    # ============================================================================
    # Messages
    # ============================================================================

    def print_error(self, error: str):
        """Display error"""
        self._write()
        self._write(f"{Colors.RED}✗ Error{Colors.RESET}")

        lines = error.split('\n')[:5]
        for line in lines:
            if line.strip():
                self._write(f"  {line}")

        if len(error.split('\n')) > 5:
            remaining = len(error.split('\n')) - 5
            self._write(f"{Colors.GRAY}  ... {remaining} more lines{Colors.RESET}")

        self._write()

    def print_warning(self, warning: str):
        """Display warning"""
        self._write(f"{Colors.YELLOW}⚠{Colors.RESET} {warning}")

    def print_info(self, info: str):
        """Display info"""
        self._write(f"{Colors.BLUE}ℹ{Colors.RESET} {info}")

    def print_success(self, message: str):
        """Display success"""
        self._write(f"{Colors.GREEN}✓{Colors.RESET} {message}")

    # ============================================================================
    # Session End
    # ============================================================================

    def print_session_summary(self, stats: dict):
        """Display minimal session summary"""
        self._write()
        self._write(f"{Colors.BOLD}Session Summary{Colors.RESET}")
        self._write()

        if 'duration' in stats:
            self._write(f"  Duration: {Colors.GRAY}{stats['duration']}{Colors.RESET}")

        if 'message_count' in stats:
            self._write(f"  Messages: {Colors.GRAY}{stats['message_count']}{Colors.RESET}")

        if 'agent_calls' in stats:
            self._write(f"  Agent calls: {Colors.GRAY}{stats['agent_calls']}{Colors.RESET}")

        self._write()

    def print_goodbye(self):
        """Display goodbye"""
        self._write()
        self._write(f"{Colors.GRAY}Goodbye{Colors.RESET}")
        self._write()

    # ============================================================================
    # Debug/Verbose
    # ============================================================================

    def print_debug(self, message: str):
        """Print debug (verbose only)"""
        if self.verbose:
            self._write(f"{Colors.GRAY}[DEBUG] {message}{Colors.RESET}")

    def print_intelligence_info(self, info: dict):
        """Display intelligence info (verbose only)"""
        if not self.verbose:
            return

        method = info.get('method', 'hybrid')
        latency = info.get('latency_ms', 0)
        confidence = info.get('confidence', 0)

        self._write(
            f"{Colors.GRAY}Intelligence: {method} "
            f"({latency:.0f}ms, {confidence:.2f}){Colors.RESET}"
        )

    # ============================================================================
    # Progress
    # ============================================================================

    def show_progress(self, message: str):
        """Show progress (minimal)"""
        if self.verbose:
            self._write(f"{Colors.GRAY}⋯ {message}{Colors.RESET}")

    def show_operation_complete(self, count: int):
        """Show completion"""
        if count > 0 and self.verbose:
            self._write()
            self._write(f"{Colors.GREEN}✓{Colors.RESET} Completed {count} operation(s)")

    # ============================================================================
    # Help
    # ============================================================================

    def print_help(self, orchestrator):
        """Display help"""
        self._write()
        self._write(f"{Colors.BOLD}Available Commands{Colors.RESET}")
        self._write()
        self._write("  help  - Show this help")
        self._write("  exit  - Exit the system")
        self._write()

        if hasattr(orchestrator, 'agent_capabilities') and orchestrator.agent_capabilities:
            self._write(f"{Colors.BOLD}Available Agents{Colors.RESET}")
            self._write()

            for agent_name, capabilities in orchestrator.agent_capabilities.items():
                display_name = agent_name.replace('_', ' ').title()
                status = orchestrator.agent_health.get(agent_name, {}).get('status', 'unknown')

                if status == 'healthy':
                    icon = f"{Colors.GREEN}●{Colors.RESET}"
                elif status == 'degraded':
                    icon = f"{Colors.YELLOW}●{Colors.RESET}"
                else:
                    icon = f"{Colors.RED}●{Colors.RESET}"

                self._write(f"  {icon} {display_name}")

                if self.verbose and capabilities:
                    for cap in capabilities[:3]:
                        self._write(f"{Colors.GRAY}     • {cap}{Colors.RESET}")
                    if len(capabilities) > 3:
                        self._write(f"{Colors.GRAY}     ... and {len(capabilities) - 3} more{Colors.RESET}")

        self._write()
