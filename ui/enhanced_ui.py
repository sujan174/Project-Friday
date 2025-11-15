"""
Production-Grade Terminal UI for Project Aerius

Combines the minimal aesthetic of Claude Code with the polish of modern CLI tools.
Features:
- Animated spinners during operations
- Syntax-highlighted code blocks
- Progress indicators for multi-step operations
- Beautiful error messages
- Session statistics with tables
- Live status updates
"""

import sys
import time
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from contextlib import contextmanager

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
    # Fallback to basic UI if Rich is not installed
    from .claude_ui import ClaudeUI


class EnhancedUI:
    """
    Production-grade terminal UI with Rich library support.

    Design Philosophy:
    - Minimal yet polished
    - Informative without being cluttered
    - Smooth animations and transitions
    - Professional color scheme
    - Clear visual hierarchy
    """

    def __init__(self, verbose: bool = False):
        """Initialize the UI with Rich console"""
        self.verbose = verbose
        self.agent_call_count = 0
        self.session_start = datetime.now()
        self.message_count = 0
        self.active_spinners = {}

        if not RICH_AVAILABLE:
            # Fallback to basic UI
            self.fallback_ui = ClaudeUI(verbose=verbose)
            self.console = None
            return

        # Initialize Rich console with custom theme
        self.console = Console(
            highlight=True,
            markup=True,
            emoji=True,
            soft_wrap=True,
        )

        # Color scheme - minimal and professional
        self.colors = {
            'primary': '#00A8E8',      # Bright blue
            'success': '#00C853',      # Green
            'warning': '#FFB300',      # Amber
            'error': '#FF1744',        # Red
            'muted': '#78909C',        # Blue grey
            'accent': '#7C4DFF',       # Deep purple
            'dim': '#546E7A',          # Dim grey
        }

    def _fallback_method(self, method_name: str, *args, **kwargs):
        """Call fallback UI method if Rich is not available"""
        if not RICH_AVAILABLE and hasattr(self.fallback_ui, method_name):
            return getattr(self.fallback_ui, method_name)(*args, **kwargs)

    # ============================================================================
    # Session Management
    # ============================================================================

    def print_welcome(self, session_id: str):
        """Display beautiful welcome message"""
        if not RICH_AVAILABLE:
            return self._fallback_method('print_welcome', session_id)

        self.console.print()

        # Create title with gradient effect
        title = Text()
        title.append("Project ", style="bold")
        title.append("Aerius", style=f"bold {self.colors['primary']}")

        self.console.print(title, justify="left")
        self.console.print(
            f"[{self.colors['muted']}]Multi-Agent Orchestration System[/]",
            justify="left"
        )
        self.console.print(
            f"[{self.colors['dim']}]Session {session_id[:12]}[/]",
            justify="left"
        )
        self.console.print()

    def print_agents_loaded(self, agents: List[dict]):
        """Display loaded agents with beautiful formatting"""
        if not RICH_AVAILABLE:
            return self._fallback_method('print_agents_loaded', agents)

        if not agents:
            return

        # Success message with icon
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

    def print_agents_failed(self, failed_count: int):
        """Display failed agents count"""
        if not RICH_AVAILABLE:
            return self._fallback_method('print_agents_failed', failed_count)

        if failed_count > 0:
            self.console.print(
                f"[{self.colors['warning']}]⚠[/] {failed_count} agent(s) unavailable",
                style=f"{self.colors['muted']}"
            )
            self.console.print()

    # ============================================================================
    # User Interaction
    # ============================================================================

    def print_prompt(self):
        """Display clean, professional prompt"""
        if not RICH_AVAILABLE:
            return self._fallback_method('print_prompt')

        self.message_count += 1
        prompt_text = Text()
        prompt_text.append("❯ ", style=f"bold {self.colors['primary']}")
        self.console.print(prompt_text, end="")

    # ============================================================================
    # Agent Operations with Spinners
    # ============================================================================

    @contextmanager
    def thinking_spinner(self, message: str = "Thinking"):
        """Context manager for thinking spinner"""
        if not RICH_AVAILABLE or not self.verbose:
            yield
            return

        with self.console.status(
            f"[{self.colors['dim']}]{message}...[/]",
            spinner="dots",
            spinner_style=self.colors['primary']
        ) as status:
            yield status

    @contextmanager
    def agent_operation(self, agent_name: str, instruction: str = ""):
        """Context manager for agent operations with spinner"""
        if not RICH_AVAILABLE:
            yield
            return

        self.agent_call_count += 1
        display_name = agent_name.replace('_', ' ').title()

        start_time = time.time()

        if self.verbose:
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
        # This is handled by agent_operation context manager now
        # But kept for backward compatibility
        pass

    def end_agent_call(
        self,
        agent_name: str,
        success: bool,
        duration_ms: Optional[float] = None,
        message: Optional[str] = None
    ):
        """Show agent completion (for backward compatibility)"""
        # This is handled by agent_operation context manager now
        # But kept for backward compatibility
        pass

    def show_retry(self, agent_name: str, attempt: int, max_attempts: int):
        """Show retry with spinner"""
        if not RICH_AVAILABLE:
            return

        if self.verbose:
            self.console.print(
                f"[{self.colors['warning']}]  ↻[/] "
                f"[{self.colors['dim']}]Retry {attempt}/{max_attempts}[/]"
            )

    # ============================================================================
    # Response Display with Rich Markdown
    # ============================================================================

    def print_response(self, response: str):
        """Display response with beautiful markdown rendering"""
        if not RICH_AVAILABLE:
            return self._fallback_method('print_response', response)

        self.console.print()

        # Use Rich's markdown renderer for beautiful output
        md = Markdown(
            response,
            code_theme="monokai",
            inline_code_lexer="python",
        )
        self.console.print(md)
        self.console.print()

    def print_code(self, code: str, language: str = "python", line_numbers: bool = True):
        """Display syntax-highlighted code"""
        if not RICH_AVAILABLE:
            self.console.print(code)
            return

        syntax = Syntax(
            code,
            language,
            theme="monokai",
            line_numbers=line_numbers,
            word_wrap=True,
        )
        self.console.print(syntax)

    # ============================================================================
    # Messages with Icons and Colors
    # ============================================================================

    def print_error(self, error: str):
        """Display beautiful error message"""
        if not RICH_AVAILABLE:
            return self._fallback_method('print_error', error)

        self.console.print()

        # Create error panel
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

    def print_warning(self, warning: str):
        """Display warning"""
        if not RICH_AVAILABLE:
            return self._fallback_method('print_warning', warning)

        self.console.print(
            f"[{self.colors['warning']}]⚠[/] {warning}"
        )

    def print_info(self, info: str):
        """Display info"""
        if not RICH_AVAILABLE:
            return self._fallback_method('print_info', info)

        self.console.print(
            f"[{self.colors['primary']}]ℹ[/] {info}"
        )

    def print_success(self, message: str):
        """Display success"""
        if not RICH_AVAILABLE:
            return self._fallback_method('print_success', message)

        self.console.print(
            f"[{self.colors['success']}]✓[/] {message}"
        )

    # ============================================================================
    # Progress Bars
    # ============================================================================

    @contextmanager
    def progress_bar(self, description: str, total: int):
        """Context manager for progress bars"""
        if not RICH_AVAILABLE:
            yield None
            return

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

    # ============================================================================
    # Session Statistics with Tables
    # ============================================================================

    def print_session_summary(self, stats: dict):
        """Display beautiful session summary with table"""
        if not RICH_AVAILABLE:
            return self._fallback_method('print_session_summary', stats)

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
            self.console.print(
                f"[bold {self.colors['primary']}]Session Summary[/]"
            )
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
            self.console.print(
                f"[bold {self.colors['primary']}]Session Summary[/]"
            )
            self.console.print(table)

        self.console.print()

    def print_goodbye(self):
        """Display goodbye"""
        if not RICH_AVAILABLE:
            return self._fallback_method('print_goodbye')

        self.console.print()
        self.console.print(
            f"[{self.colors['dim']}]Goodbye ✨[/]",
            justify="left"
        )
        self.console.print()

    # ============================================================================
    # Debug/Verbose
    # ============================================================================

    def print_debug(self, message: str):
        """Print debug (verbose only)"""
        if not RICH_AVAILABLE:
            return self._fallback_method('print_debug', message)

        if self.verbose:
            self.console.print(
                f"[{self.colors['dim']}][DEBUG] {message}[/]"
            )

    def print_intelligence_info(self, info: dict):
        """Display intelligence info (verbose only)"""
        if not RICH_AVAILABLE:
            return self._fallback_method('print_intelligence_info', info)

        if not self.verbose:
            return

        method = info.get('method', 'hybrid')
        latency = info.get('latency_ms', 0)
        confidence = info.get('confidence', 0)

        self.console.print(
            f"[{self.colors['dim']}]Intelligence: {method} "
            f"({latency:.0f}ms, confidence: {confidence:.2f})[/]"
        )

    # ============================================================================
    # Help with Beautiful Tables
    # ============================================================================

    def print_help(self, orchestrator):
        """Display beautiful help with tables"""
        if not RICH_AVAILABLE:
            return self._fallback_method('print_help', orchestrator)

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

    # ============================================================================
    # Additional Utility Methods
    # ============================================================================

    def print_divider(self):
        """Print subtle divider"""
        if not RICH_AVAILABLE:
            return self._fallback_method('print_divider')

        self.console.print(
            f"[{self.colors['dim']}]{'─' * 60}[/]"
        )

    def show_progress(self, message: str):
        """Show progress (minimal)"""
        if not RICH_AVAILABLE:
            return self._fallback_method('show_progress', message)

        if self.verbose:
            self.console.print(
                f"[{self.colors['dim']}]⋯ {message}[/]"
            )

    def clear_screen(self):
        """Clear the terminal screen"""
        if not RICH_AVAILABLE:
            return

        self.console.clear()

    def print_banner(self, text: str, style: str = "primary"):
        """Print a banner message"""
        if not RICH_AVAILABLE:
            print(text)
            return

        panel = Panel(
            Text(text, justify="center"),
            border_style=self.colors.get(style, self.colors['primary']),
            box=box.DOUBLE,
        )
        self.console.print(panel)
