"""
Claude Code-style Terminal UI

A clean, minimal interface inspired by Claude Code with proper markdown support.
"""

import sys
from datetime import datetime
from typing import Optional, Dict, List
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.spinner import Spinner
from rich.live import Live
from rich.theme import Theme


# Custom theme matching Claude Code's aesthetic
CLAUDE_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red",
    "success": "green",
    "muted": "dim",
    "prompt": "bold cyan",
})


class ClaudeUI:
    """
    Clean, minimal UI inspired by Claude Code.

    Features:
    - Proper markdown rendering
    - Code blocks with syntax highlighting
    - Clean visual hierarchy
    - Subtle, professional colors
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.console = Console(theme=CLAUDE_THEME, highlight=False)
        self.agent_call_count = 0
        self.current_spinner = None

    # ============================================================================
    # Core Output Methods
    # ============================================================================

    def print(self, text: str = "", style: Optional[str] = None):
        """Print text with optional style"""
        if style:
            self.console.print(text, style=style)
        else:
            self.console.print(text)

    def print_markdown(self, markdown_text: str):
        """Print markdown with proper formatting"""
        # Use monokai theme for code blocks, but keep headers simple
        md = Markdown(
            markdown_text,
            code_theme="monokai",
            inline_code_theme="monokai",
            inline_code_lexer=None
        )
        self.console.print(md)
        self.console.print()

    # ============================================================================
    # Session Management
    # ============================================================================

    def print_welcome(self, session_id: str):
        """Display welcome message"""
        self.console.print()
        self.console.print("Multi-Agent Orchestration System", style="bold")
        self.console.print(f"Session {session_id[:8]}...", style="muted")
        self.console.print()

    def print_agents_loaded(self, agents: List[dict]):
        """Display loaded agents"""
        if not agents:
            return

        self.console.print(f"✓ {len(agents)} agents ready", style="success")
        self.console.print()

    def print_agents_failed(self, failed_count: int):
        """Display failed agents count"""
        if failed_count > 0:
            self.console.print(f"⚠ {failed_count} agents unavailable", style="warning")
            self.console.print()

    # ============================================================================
    # User Interaction
    # ============================================================================

    def print_prompt(self):
        """Display input prompt"""
        self.console.print("> ", style="prompt", end="")

    def print_divider(self):
        """Print a subtle divider"""
        self.console.print("─" * 60, style="muted")

    # ============================================================================
    # Agent Operations
    # ============================================================================

    def start_thinking(self):
        """Show thinking indicator"""
        # Silent - no spinner needed for Claude Code style
        pass

    def start_agent_call(self, agent_name: str, instruction: str):
        """Indicate agent call started"""
        self.agent_call_count += 1

        # Clean display
        display_name = agent_name.replace('_', ' ').title()
        preview = instruction[:60] + "..." if len(instruction) > 60 else instruction

        self.console.print()
        self.console.print(f"⚙  {display_name}", style="info")
        if self.verbose:
            self.console.print(f"   {preview}", style="muted")

    def end_agent_call(
        self,
        agent_name: str,
        success: bool,
        duration_ms: Optional[float] = None,
        message: Optional[str] = None
    ):
        """Show agent call completion"""
        if success:
            duration_str = f" ({duration_ms:.0f}ms)" if duration_ms else ""
            self.console.print(f"   ✓ Completed{duration_str}", style="success")

            if message and self.verbose:
                preview = message[:80] + "..." if len(message) > 80 else message
                self.console.print(f"   {preview}", style="muted")
        else:
            self.console.print(f"   ✗ Failed", style="error")

            if message:
                # Show first few lines of error
                lines = message.split('\n')[:3]
                for line in lines:
                    if line.strip():
                        self.console.print(f"   {line}", style="error")

    def show_retry(self, agent_name: str, attempt: int, max_attempts: int):
        """Show retry indicator"""
        display_name = agent_name.replace('_', ' ').title()
        self.console.print(
            f"   ⟳ Retrying {display_name} ({attempt}/{max_attempts})",
            style="warning"
        )

    # ============================================================================
    # Response Display
    # ============================================================================

    def print_response(self, response: str):
        """Display assistant response with markdown formatting"""
        self.console.print()

        # Use Rich's markdown renderer for perfect formatting
        self.print_markdown(response)

    # ============================================================================
    # Messages & Notifications
    # ============================================================================

    def print_error(self, error: str):
        """Display error message"""
        self.console.print()
        self.console.print("✗ Error", style="bold error")

        # Format error
        lines = error.split('\n')[:5]
        for line in lines:
            if line.strip():
                self.console.print(f"  {line}", style="error")

        if len(error.split('\n')) > 5:
            remaining = len(error.split('\n')) - 5
            self.console.print(f"  ... ({remaining} more lines)", style="muted")

        self.console.print()

    def print_warning(self, warning: str):
        """Display warning message"""
        self.console.print(f"⚠ {warning}", style="warning")

    def print_info(self, info: str):
        """Display info message"""
        self.console.print(f"ℹ {info}", style="info")

    def print_success(self, message: str):
        """Display success message"""
        self.console.print(f"✓ {message}", style="success")

    # ============================================================================
    # Session End
    # ============================================================================

    def print_session_summary(self, stats: dict):
        """Display session summary"""
        self.console.print()
        self.print_divider()
        self.console.print("Session Summary", style="bold")
        self.console.print()

        if 'duration' in stats:
            self.console.print(f"  Duration: {stats['duration']}", style="muted")

        if 'message_count' in stats:
            self.console.print(f"  Messages: {stats['message_count']}", style="muted")

        if 'agent_calls' in stats:
            self.console.print(f"  Agent calls: {stats['agent_calls']}", style="muted")

        if 'success_rate' in stats and stats['success_rate'] != "N/A":
            rate = stats['success_rate']
            if rate > 80:
                style = "success"
            elif rate > 50:
                style = "warning"
            else:
                style = "error"
            self.console.print(f"  Success rate: {rate}%", style=style)

        self.print_divider()
        self.console.print()

    def print_goodbye(self):
        """Display goodbye message"""
        self.console.print()
        self.console.print("Goodbye!", style="muted")
        self.console.print()

    # ============================================================================
    # Debug/Verbose Output
    # ============================================================================

    def print_debug(self, message: str):
        """Print debug message (verbose mode only)"""
        if self.verbose:
            self.console.print(f"[DEBUG] {message}", style="muted")

    def print_intelligence_info(self, info: dict):
        """Display intelligence info (verbose mode only)"""
        if not self.verbose:
            return

        method = info.get('method', 'hybrid')
        latency = info.get('latency_ms', 0)
        confidence = info.get('confidence', 0)

        self.console.print(
            f"Intelligence: {method} ({latency:.0f}ms, confidence: {confidence:.2f})",
            style="muted"
        )

    # ============================================================================
    # Progress & Status
    # ============================================================================

    def show_progress(self, message: str):
        """Show progress message"""
        self.console.print(f"⋯ {message}", style="muted")

    def show_operation_complete(self, count: int):
        """Show completion message"""
        if count > 0:
            self.console.print()
            self.console.print(f"✓ Completed {count} operation(s)", style="success")

    # ============================================================================
    # Help & Information
    # ============================================================================

    def print_help(self, orchestrator):
        """Display help information"""
        help_md = """
# Available Commands

- `help` - Show this help
- `exit` - Exit the system

"""

        if hasattr(orchestrator, 'agent_capabilities') and orchestrator.agent_capabilities:
            help_md += "# Available Agents\n\n"

            for agent_name, capabilities in orchestrator.agent_capabilities.items():
                display_name = agent_name.replace('_', ' ').title()
                status = orchestrator.agent_health.get(agent_name, {}).get('status', 'unknown')

                if status == 'healthy':
                    status_icon = "🟢"
                elif status == 'degraded':
                    status_icon = "🟡"
                else:
                    status_icon = "🔴"

                help_md += f"- {status_icon} **{display_name}**\n"

                if self.verbose and capabilities:
                    for cap in capabilities[:3]:
                        help_md += f"  - {cap}\n"
                    if len(capabilities) > 3:
                        help_md += f"  - ... and {len(capabilities) - 3} more\n"
                help_md += "\n"

        self.console.print()
        self.print_markdown(help_md)
