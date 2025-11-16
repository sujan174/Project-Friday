#!/usr/bin/env python3
"""
Terminal UI for Project Aerius Multi-Agent Orchestration System

A beautiful, production-grade terminal interface that combines the polish of modern
CLI tools with the elegance of minimal design. Features rich formatting, live updates,
progress indicators, and graceful fallbacks.

Design Philosophy:
- Clean and professional aesthetics
- Rich formatting with syntax highlighting
- Smooth animations and transitions
- Informative without clutter
- Graceful degradation when Rich is unavailable
"""

import sys
import time
import re
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import contextmanager

# Try to import Rich for enhanced terminal UI
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.live import Live
    from rich.text import Text
    from rich.layout import Layout
    from rich import box
    from rich.style import Style
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# ============================================================================
# ANSI Colors for Fallback Mode
# ============================================================================

class Colors:
    """ANSI color codes for when Rich is not available"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    # Colors
    GRAY = '\033[90m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'


# ============================================================================
# Main Terminal UI Class
# ============================================================================

class TerminalUI:
    """
    Production-grade terminal UI for Project Aerius.

    Automatically uses Rich library when available, falls back to ANSI codes
    when not. Provides a consistent interface regardless of the backend.

    Features:
    - Welcome screens and session management
    - Agent status tracking and display
    - Beautiful markdown rendering
    - Syntax-highlighted code blocks
    - Progress bars and spinners
    - Error/warning/info/success messages
    - Session statistics and summaries
    - Help screens with agent listings
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the terminal UI

        Args:
            verbose: Enable verbose output with detailed operation logging
        """
        self.verbose = verbose
        self.agent_call_count = 0
        self.message_count = 0
        self.session_start = datetime.now()
        self.active_spinners = {}

        if RICH_AVAILABLE:
            # Initialize Rich console with custom theme
            self.console = Console(
                highlight=True,
                markup=True,
                emoji=True,
                soft_wrap=True,
            )

            # Professional color scheme
            self.colors = {
                'primary': '#00A8E8',      # Bright blue
                'success': '#00C853',      # Green
                'warning': '#FFB300',      # Amber
                'error': '#FF1744',        # Red
                'muted': '#78909C',        # Blue grey
                'accent': '#7C4DFF',       # Deep purple
                'dim': '#546E7A',          # Dim grey
            }
        else:
            self.console = None
            self.colors = None


    # ========================================================================
    # Core Output Methods
    # ========================================================================

    def _write(self, text: str = "", end: str = '\n'):
        """Write text to stdout (fallback mode)"""
        sys.stdout.write(text + end)
        sys.stdout.flush()

    def _print_rich(self, *args, **kwargs):
        """Print using Rich if available"""
        if RICH_AVAILABLE and self.console:
            self.console.print(*args, **kwargs)


    # ========================================================================
    # Session Management
    # ========================================================================

    def print_welcome(self, session_id: str):
        """Display welcome screen with session information"""
        if RICH_AVAILABLE:
            self.console.clear()
            self.console.print()

            # Create beautiful welcome panel
            welcome_text = Text()
            welcome_text.append("Project Aerius\n", style=f"bold {self.colors['primary']}")
            welcome_text.append("Multi-Agent Orchestration System\n\n", style=f"{self.colors['muted']}")
            welcome_text.append(f"Session: {session_id[:12]}", style=f"{self.colors['dim']}")

            panel = Panel(
                welcome_text,
                border_style=self.colors['primary'],
                box=box.DOUBLE,
                padding=(1, 2),
            )
            self.console.print(panel)
            self.console.print()
        else:
            # Fallback mode
            self._write()
            self._write(f"{Colors.BOLD}Project Aerius - Multi-Agent Orchestration System{Colors.RESET}")
            self._write(f"{Colors.GRAY}Session {session_id[:8]}{Colors.RESET}")
            self._write()

    def print_agents_loaded(self, agents: List[dict]):
        """Display loaded agents count and details"""
        if not agents:
            return

        if RICH_AVAILABLE:
            self.console.print(
                f"[{self.colors['success']}]✓[/] [bold]{len(agents)}[/] agents initialized",
                style=f"{self.colors['muted']}"
            )

            # Show agent list in verbose mode
            if self.verbose and len(agents) > 0:
                agent_names = [
                    agent.get('name', 'Unknown').replace('_', ' ').title()
                    for agent in agents[:5]
                ]
                self.console.print(
                    f"[{self.colors['dim']}]  • {', '.join(agent_names)}[/]"
                )
                if len(agents) > 5:
                    self.console.print(
                        f"[{self.colors['dim']}]  • ... and {len(agents) - 5} more[/]"
                    )

            self.console.print()
        else:
            # Fallback mode
            self._write(f"{Colors.GREEN}✓{Colors.RESET} {len(agents)} agents ready")
            self._write()

    def print_agents_failed(self, failed_count: int):
        """Display failed agents count"""
        if failed_count <= 0:
            return

        if RICH_AVAILABLE:
            self.console.print(
                f"[{self.colors['warning']}]⚠[/] {failed_count} agent(s) unavailable",
                style=f"{self.colors['muted']}"
            )
            self.console.print()
        else:
            # Fallback mode
            self._write(f"{Colors.YELLOW}⚠{Colors.RESET} {failed_count} agents unavailable")
            self._write()

    def print_goodbye(self):
        """Display goodbye message"""
        if RICH_AVAILABLE:
            self.console.print()
            self.console.print(
                f"[{self.colors['dim']}]Goodbye ✨[/]",
                justify="left"
            )
            self.console.print()
        else:
            # Fallback mode
            self._write()
            self._write(f"{Colors.GRAY}Goodbye{Colors.RESET}")
            self._write()


    # ========================================================================
    # User Interaction
    # ========================================================================

    def print_prompt(self):
        """Display user input prompt"""
        self.message_count += 1

        if RICH_AVAILABLE:
            prompt_text = Text()
            prompt_text.append("You", style=f"bold {self.colors['primary']}")
            prompt_text.append("\n❯ ", style=f"{self.colors['muted']}")
            self.console.print(prompt_text, end="")
        else:
            # Fallback mode
            self._write(f"{Colors.BOLD}{Colors.CYAN}>{Colors.RESET} ", end='')


    # ========================================================================
    # Agent Operations & Progress Indicators
    # ========================================================================

    def start_thinking(self):
        """Start thinking indicator (for backward compatibility)"""
        pass

    @contextmanager
    def thinking_spinner(self, message: str = "Thinking"):
        """Context manager for displaying thinking spinner"""
        if RICH_AVAILABLE and self.verbose:
            with self.console.status(
                f"[{self.colors['dim']}]{message}...[/]",
                spinner="dots",
                spinner_style=self.colors['primary']
            ) as status:
                yield status
        else:
            yield

    @contextmanager
    def agent_operation(self, agent_name: str, instruction: str = ""):
        """Context manager for agent operations with spinner and timing"""
        self.agent_call_count += 1
        display_name = agent_name.replace('_', ' ').title()
        start_time = time.time()

        if RICH_AVAILABLE and self.verbose:
            self.console.print()
            with self.console.status(
                f"[{self.colors['dim']}]→ {display_name}...[/]",
                spinner="arc",
                spinner_style=self.colors['accent']
            ) as status:
                try:
                    yield status
                    duration_ms = (time.time() - start_time) * 1000
                    self.console.print(
                        f"[{self.colors['success']}]  ✓[/] "
                        f"[{self.colors['dim']}]{duration_ms:.0f}ms[/]"
                    )
                except Exception as e:
                    self.console.print(
                        f"[{self.colors['error']}]  ✗ Failed[/]"
                    )
                    raise
        else:
            yield None

    def start_agent_call(self, agent_name: str, instruction: str):
        """Show agent starting (for backward compatibility)"""
        if not RICH_AVAILABLE and self.verbose:
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
        """Show agent completion (for backward compatibility)"""
        if not RICH_AVAILABLE and self.verbose:
            if success:
                duration = f" {duration_ms:.0f}ms" if duration_ms else ""
                self._write(f"{Colors.GRAY}  ✓{duration}{Colors.RESET}")
            else:
                self._write(f"{Colors.RED}  ✗ Failed{Colors.RESET}")
                if message:
                    first_line = message.split('\n')[0]
                    if first_line.strip():
                        self._write(f"{Colors.RED}  {first_line[:80]}{Colors.RESET}")

    def show_retry(self, agent_name: str, attempt: int, max_attempts: int):
        """Show retry attempt indicator"""
        if not self.verbose:
            return

        if RICH_AVAILABLE:
            self.console.print(
                f"[{self.colors['warning']}]  ↻[/] "
                f"[{self.colors['dim']}]Retry {attempt}/{max_attempts}[/]"
            )
        else:
            self._write(f"{Colors.GRAY}  ↻ Retry {attempt}/{max_attempts}{Colors.RESET}")

    @contextmanager
    def progress_bar(self, description: str, total: int):
        """Context manager for progress bars"""
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(spinner_name="dots", style=self.colors['primary']),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(
                    style=self.colors['primary'],
                    complete_style=self.colors['success'],
                    finished_style=self.colors['success'],
                ),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console,
                transient=True,
            ) as progress:
                task = progress.add_task(description, total=total)
                yield progress, task
        else:
            yield None, None


    # ========================================================================
    # Response Display & Markdown Rendering
    # ========================================================================

    def print_response(self, response: str):
        """Display agent response with beautiful formatting"""
        if RICH_AVAILABLE:
            self.console.print()

            # Use Rich markdown renderer
            md = Markdown(
                response,
                code_theme="monokai",
                inline_code_lexer="python",
            )

            # Wrap in panel like modern CLI tools
            panel = Panel(
                md,
                border_style=self.colors['primary'],
                box=box.ROUNDED,
                padding=(1, 2),
                title="[bold]Aerius[/bold]",
                title_align="left",
            )
            self.console.print(panel)
            self.console.print()
        else:
            # Fallback to basic markdown rendering
            self._write()
            self._render_markdown_fallback(response)

    def _render_markdown_fallback(self, text: str):
        """Render markdown in fallback mode with ANSI codes"""
        lines = text.split('\n')
        in_code_block = False

        i = 0
        while i < len(lines):
            line = lines[i]

            # Code blocks
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                if in_code_block:
                    self._write()  # Blank line before code
                else:
                    self._write()  # Blank line after code
                i += 1
                continue

            if in_code_block:
                self._write(f"{Colors.DIM}  {line}{Colors.RESET}")
                i += 1
                continue

            # Empty lines
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
                content = self._format_inline_markdown(content)
                spaces = ' ' * indent
                self._write(f"{spaces}• {content}")
                i += 1
                continue

            # Numbered lists
            if re.match(r'^\d+\.\s', line.strip()):
                content = re.sub(r'^\d+\.\s', '', line.strip())
                content = self._format_inline_markdown(content)
                number = re.match(r'^(\d+)\.', line.strip()).group(1)
                self._write(f"{number}. {content}")
                i += 1
                continue

            # Blockquotes
            if line.strip().startswith('>'):
                content = line.strip()[1:].strip()
                content = self._format_inline_markdown(content)
                self._write(f"{Colors.DIM}  {content}{Colors.RESET}")
                i += 1
                continue

            # Regular paragraphs
            formatted = self._format_inline_markdown(line.strip())
            self._write(formatted)
            i += 1

        self._write()  # Final blank line

    def _format_inline_markdown(self, text: str) -> str:
        """Format inline markdown elements (bold, italic, code)"""
        # Bold: **text**
        text = re.sub(
            r'\*\*([^\*]+)\*\*',
            f'{Colors.BOLD}\\1{Colors.RESET}',
            text
        )

        # Italic: *text*
        text = re.sub(
            r'(?<!\*)\*(?!\*)([^\*]+)\*(?!\*)',
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

    def print_code(self, code: str, language: str = "python", line_numbers: bool = True):
        """Display syntax-highlighted code"""
        if RICH_AVAILABLE:
            syntax = Syntax(
                code,
                language,
                theme="monokai",
                line_numbers=line_numbers,
                word_wrap=True,
            )
            self.console.print(syntax)
        else:
            # Fallback: just print with indentation
            self._write()
            for line in code.split('\n'):
                self._write(f"{Colors.DIM}  {line}{Colors.RESET}")
            self._write()


    # ========================================================================
    # Messages (Error, Warning, Info, Success)
    # ========================================================================

    def print_error(self, error: str):
        """Display error message"""
        if RICH_AVAILABLE:
            self.console.print()
            error_text = Text(error)
            panel = Panel(
                error_text,
                title="[bold]Error[/bold]",
                border_style=self.colors['error'],
                box=box.ROUNDED,
                padding=(0, 1),
            )
            self.console.print(panel)
            self.console.print()
        else:
            # Fallback mode
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
        """Display warning message"""
        if RICH_AVAILABLE:
            self.console.print(f"[{self.colors['warning']}]⚠[/] {warning}")
        else:
            self._write(f"{Colors.YELLOW}⚠{Colors.RESET} {warning}")

    def print_info(self, info: str):
        """Display info message"""
        if RICH_AVAILABLE:
            self.console.print(f"[{self.colors['primary']}]ℹ[/] {info}")
        else:
            self._write(f"{Colors.BLUE}ℹ{Colors.RESET} {info}")

    def print_success(self, message: str):
        """Display success message"""
        if RICH_AVAILABLE:
            self.console.print(f"[{self.colors['success']}]✓[/] {message}")
        else:
            self._write(f"{Colors.GREEN}✓{Colors.RESET} {message}")


    # ========================================================================
    # Session Statistics & Summaries
    # ========================================================================

    def print_session_summary(self, stats: dict):
        """Display session summary with statistics"""
        if RICH_AVAILABLE:
            self.console.print()

            # Calculate duration
            duration = datetime.now() - self.session_start
            hours, remainder = divmod(duration.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)

            if hours > 0:
                duration_str = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                duration_str = f"{minutes}m {seconds}s"
            else:
                duration_str = f"{seconds}s"

            # Create statistics table
            table = Table(
                show_header=False,
                box=None,
                padding=(0, 2),
                show_edge=False,
            )

            table.add_column("Metric", style=f"{self.colors['dim']}")
            table.add_column("Value", style="bold")

            table.add_row("Duration", duration_str)
            table.add_row("Messages", str(stats.get('message_count', self.message_count)))
            table.add_row("Agent Calls", str(stats.get('agent_calls', self.agent_call_count)))

            # Add agent-specific stats if available
            if 'agent_stats' in stats and stats['agent_stats']:
                self.console.print(f"[bold {self.colors['primary']}]Session Summary[/]")
                self.console.print(table)
                self.console.print()

                # Agent performance table
                agent_table = Table(
                    title="Agent Performance",
                    box=box.MINIMAL,
                    show_header=True,
                    header_style=f"bold {self.colors['primary']}",
                )

                agent_table.add_column("Agent", style=f"{self.colors['muted']}")
                agent_table.add_column("Calls", justify="right")
                agent_table.add_column("Avg Time", justify="right")
                agent_table.add_column("Success", justify="right")

                for agent_name, agent_stat in stats['agent_stats'].items():
                    display_name = agent_name.replace('_', ' ').title()
                    calls = agent_stat.get('calls', 0)
                    avg_time = agent_stat.get('avg_time_ms', 0)
                    success_rate = agent_stat.get('success_rate', 100)

                    # Color code success rate
                    if success_rate >= 95:
                        success_color = self.colors['success']
                    elif success_rate >= 80:
                        success_color = self.colors['warning']
                    else:
                        success_color = self.colors['error']

                    agent_table.add_row(
                        display_name,
                        str(calls),
                        f"{avg_time:.0f}ms",
                        f"[{success_color}]{success_rate:.0f}%[/]"
                    )

                self.console.print(agent_table)
            else:
                self.console.print(f"[bold {self.colors['primary']}]Session Summary[/]")
                self.console.print(table)

            self.console.print()
        else:
            # Fallback mode
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


    # ========================================================================
    # Help & Agent Listings
    # ========================================================================

    def print_help(self, orchestrator):
        """Display help with available commands and agents"""
        if RICH_AVAILABLE:
            self.console.print()
            self.console.print(f"[bold {self.colors['primary']}]Available Commands[/]")
            self.console.print()

            # Commands table
            cmd_table = Table(
                show_header=False,
                box=None,
                padding=(0, 2),
            )
            cmd_table.add_column("Command", style=f"bold {self.colors['accent']}")
            cmd_table.add_column("Description", style=f"{self.colors['muted']}")

            cmd_table.add_row("help", "Show this help message")
            cmd_table.add_row("exit", "Exit the system")
            cmd_table.add_row("stats", "Show session statistics")
            cmd_table.add_row("agents", "List all available agents")

            self.console.print(cmd_table)
            self.console.print()

            # Agents table
            if hasattr(orchestrator, 'agent_capabilities') and orchestrator.agent_capabilities:
                self.console.print(f"[bold {self.colors['primary']}]Available Agents[/]")
                self.console.print()

                agent_table = Table(
                    show_header=True,
                    box=box.MINIMAL,
                    header_style=f"bold {self.colors['primary']}",
                )

                agent_table.add_column("Status", width=3)
                agent_table.add_column("Agent")
                agent_table.add_column("Capabilities", style=f"{self.colors['dim']}")

                for agent_name, capabilities in orchestrator.agent_capabilities.items():
                    display_name = agent_name.replace('_', ' ').title()
                    status = orchestrator.agent_health.get(agent_name, {}).get('status', 'unknown')

                    # Status indicator
                    if status == 'healthy':
                        status_icon = f"[{self.colors['success']}]●[/]"
                    elif status == 'degraded':
                        status_icon = f"[{self.colors['warning']}]●[/]"
                    else:
                        status_icon = f"[{self.colors['error']}]●[/]"

                    # Capabilities preview
                    if capabilities:
                        cap_preview = ', '.join(capabilities[:2])
                        if len(capabilities) > 2:
                            cap_preview += f" +{len(capabilities) - 2} more"
                    else:
                        cap_preview = "No capabilities listed"

                    agent_table.add_row(
                        status_icon,
                        display_name,
                        cap_preview
                    )

                self.console.print(agent_table)

            self.console.print()
        else:
            # Fallback mode
            self._write()
            self._write(f"{Colors.BOLD}Available Commands{Colors.RESET}")
            self._write()
            self._write("  help  - Show this help")
            self._write("  exit  - Exit the system")
            self._write("  stats - Show session statistics")
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


    # ========================================================================
    # Debug & Verbose Output
    # ========================================================================

    def print_debug(self, message: str):
        """Print debug message (verbose mode only)"""
        if not self.verbose:
            return

        if RICH_AVAILABLE:
            self.console.print(f"[{self.colors['dim']}][DEBUG] {message}[/]")
        else:
            self._write(f"{Colors.GRAY}[DEBUG] {message}{Colors.RESET}")

    def print_intelligence_info(self, info: dict):
        """Display intelligence routing information (verbose mode only)"""
        if not self.verbose:
            return

        method = info.get('method', 'hybrid')
        latency = info.get('latency_ms', 0)
        confidence = info.get('confidence', 0)

        if RICH_AVAILABLE:
            self.console.print(
                f"[{self.colors['dim']}]Intelligence: {method} "
                f"({latency:.0f}ms, confidence: {confidence:.2f})[/]"
            )
        else:
            self._write(
                f"{Colors.GRAY}Intelligence: {method} "
                f"({latency:.0f}ms, {confidence:.2f}){Colors.RESET}"
            )


    # ========================================================================
    # Utility Methods
    # ========================================================================

    def print_divider(self):
        """Print a subtle divider line"""
        if RICH_AVAILABLE:
            self.console.print(f"[{self.colors['dim']}]{'─' * 60}[/]")
        else:
            self._write(f"{Colors.GRAY}{'─' * 60}{Colors.RESET}")

    def show_progress(self, message: str):
        """Show progress message (verbose mode only)"""
        if not self.verbose:
            return

        if RICH_AVAILABLE:
            self.console.print(f"[{self.colors['dim']}]⋯ {message}[/]")
        else:
            self._write(f"{Colors.GRAY}⋯ {message}{Colors.RESET}")

    def clear_screen(self):
        """Clear the terminal screen"""
        if RICH_AVAILABLE and self.console:
            self.console.clear()
        else:
            # Fallback: ANSI clear screen
            sys.stdout.write('\033[2J\033[H')
            sys.stdout.flush()

    def print_banner(self, text: str, style: str = "primary"):
        """Print a banner message"""
        if RICH_AVAILABLE:
            panel = Panel(
                Text(text, justify="center"),
                border_style=self.colors.get(style, self.colors['primary']),
                box=box.DOUBLE,
            )
            self.console.print(panel)
        else:
            # Fallback: simple banner
            self._write()
            self._write(f"{Colors.BOLD}{text}{Colors.RESET}")
            self._write()

    def show_operation_complete(self, count: int):
        """Show operation completion message"""
        if count > 0 and self.verbose:
            if RICH_AVAILABLE:
                self.console.print()
                self.console.print(
                    f"[{self.colors['success']}]✓[/] Completed {count} operation(s)"
                )
            else:
                self._write()
                self._write(f"{Colors.GREEN}✓{Colors.RESET} Completed {count} operation(s)")
