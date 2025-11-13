import sys
import time
import asyncio
from typing import Optional, List, Dict, Any, Generator
from datetime import datetime, timedelta
from contextlib import contextmanager

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TaskID
from rich.syntax import Syntax
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich.rule import Rule
from rich.columns import Columns
from rich import box
from rich.align import Align
from rich.padding import Padding
from rich.tree import Tree
from rich.status import Status

from ui.design_system import ds, build_status_text, build_key_value, build_divider


class EnhancedTerminalUI:

    def __init__(self, verbose: bool = False):
        self.console = ds.get_console()
        self.verbose = verbose

        self.session_start = datetime.now()
        self.message_count = 0
        self.agent_calls = {}
        self.session_stats = {
            'successes': 0,
            'errors': 0,
            'agent_calls_by_type': {}
        }

        self.show_help = False
        self.current_operation = None
        self.last_notification = None

    def clear_screen(self):
        self.console.clear()

    def print_header(self, session_id: str):
        self.clear_screen()

        banner = Text()
        banner.append("\n")
        banner.append("     █████╗ ██╗    ██╗ ██████╗ ██████╗  ██████╗██╗  ██╗███████╗\n", style=f"bold {ds.colors.primary_500}")
        banner.append("    ██╔══██╗██║    ██║██╔═══██╗██╔══██╗██╔════╝██║  ██║██╔════╝\n", style=f"bold {ds.colors.primary_500}")
        banner.append("    ███████║██║ █╗ ██║██║   ██║██████╔╝██║     ███████║█████╗\n", style=f"{ds.colors.primary_600}")
        banner.append("    ██╔══██║██║███╗██║██║   ██║██╔══██╗██║     ██╔══██║██╔══╝\n", style=f"{ds.colors.primary_700}")
        banner.append("    ██║  ██║╚███╔███╔╝╚██████╔╝██║  ██║╚██████╗██║  ██║███████╗\n", style=f"{ds.colors.primary_800}")
        banner.append("    ╚═╝  ╚═╝ ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝\n", style=f"dim {ds.colors.primary_900}")
        banner.append("\n")

        tagline = Text()
        tagline.append("                 ", style="")
        tagline.append("✨ ", style=ds.colors.accent_teal)
        tagline.append("Your Intelligent Multi-Agent Workspace Assistant", style=f"italic {ds.colors.text_secondary}")
        tagline.append(" ✨", style=ds.colors.accent_teal)

        header_content = Group(
            banner,
            Align.center(tagline),
        )

        self.console.print(header_content)
        self.console.print()

        session_content = Text()
        session_content.append(f"{ds.icons.rocket} ", style=ds.colors.accent_purple)
        session_content.append("Session: ", style=ds.semantic.component['label'])
        session_content.append(f"{session_id[:12]}...\n", style=ds.colors.accent_teal)

        session_content.append(f"{ds.icons.clock} ", style=ds.colors.accent_purple)
        session_content.append("Started: ", style=ds.semantic.component['label'])
        session_content.append(f"{self.session_start.strftime('%I:%M:%S %p')}\n", style=ds.colors.text_primary)

        session_content.append(f"{ds.icons.info} ", style=ds.colors.accent_purple)
        session_content.append("Quick Tips: ", style=ds.semantic.component['label'])
        session_content.append("Type ", style=ds.colors.text_secondary)
        session_content.append("help", style=f"bold {ds.colors.accent_amber}")
        session_content.append(" for keyboard shortcuts, ", style=ds.colors.text_secondary)
        session_content.append("exit", style=f"bold {ds.colors.error}")
        session_content.append(" to quit", style=ds.colors.text_secondary)

        info_panel = Panel(
            session_content,
            border_style=ds.colors.border_bright,
            box=ds.box_styles.panel_default,
            padding=ds.spacing.padding_md
        )
        self.console.print(Padding(info_panel, (0, 2)))
        self.console.print()

    def print_agent_discovery_start(self):
        self.console.print()

        header = Text()
        header.append(f"{ds.icons.wrench} ", style=ds.colors.accent_purple)
        header.append("Discovering Agents...", style=f"bold {ds.colors.primary_500}")

        self.console.print(header)
        self.console.print()

    def print_agent_table(self, loaded_agents: List[Dict[str, Any]], failed_agents: List[str] = None):
        table = Table(
            show_header=True,
            header_style=f"bold {ds.colors.text_primary}",
            border_style=ds.colors.border_bright,
            box=ds.box_styles.table_default,
            padding=(0, 2),
            expand=False
        )

        table.add_column(f"{ds.icons.agent} Agent", style=f"bold {ds.colors.accent_purple}", width=18)
        table.add_column(f"{ds.icons.gear} Capabilities", style=ds.colors.text_secondary, width=50)
        table.add_column("Count", justify="right", style=ds.colors.accent_teal, width=8)

        for agent in loaded_agents:
            name = agent['name'].replace('_', ' ').title()
            caps = agent.get('capabilities', [])

            if len(caps) > 3:
                caps_text = ", ".join(caps[:3])
                caps_text = f"{caps_text}..."
            else:
                caps_text = ", ".join(caps) if caps else "General purpose"

            agent_icon = self._get_agent_icon(agent['name'])

            table.add_row(
                f"{agent_icon} {name}",
                caps_text,
                f"{len(caps)}"
            )

        self.console.print(Padding(table, (0, 2)))

        if failed_agents and len(failed_agents) > 0:
            warning = Text()
            warning.append(f"\n{ds.icons.warning} ", style=ds.colors.warning)
            warning.append(f"Skipped {len(failed_agents)} agent(s): ", style=ds.colors.warning)
            warning.append(", ".join(failed_agents), style=ds.colors.text_tertiary)
            self.console.print(warning)

        self.console.print()

    def print_loaded_summary(self, loaded_count: int, failed_count: int):
        summary = Text()
        summary.append(f"\n{ds.icons.success} ", style=ds.colors.success)
        summary.append(f"Loaded ", style=ds.colors.success)
        summary.append(f"{loaded_count}", style=f"bold {ds.colors.success}")
        summary.append(f" agent(s) successfully", style=ds.colors.success)

        if failed_count > 0:
            summary.append(f"  {ds.icons.warning} ", style=ds.colors.warning)
            summary.append(f"{failed_count} skipped", style=ds.colors.warning)

        panel = Panel(
            Align.center(summary),
            border_style=ds.colors.success,
            box=ds.box_styles.panel_emphasis,
            padding=ds.spacing.padding_sm
        )

        self.console.print(panel)
        self.console.print()

    def _get_agent_icon(self, agent_name: str) -> str:
        icon_map = {
            'slack': '💬',
            'jira': '📊',
            'github': '🐙',
            'notion': '📝',
            'browser': '🌐',
            'scraper': '🕷',
            'code_reviewer': '👁',
            'email': '📧',
        }

        for key, icon in icon_map.items():
            if key in agent_name.lower():
                return icon

        return ds.icons.agent

    def print_prompt(self):
        prompt_text = Text()
        prompt_text.append("\n┃ ", style=f"bold {ds.colors.border_bright}")
        prompt_text.append(f"{ds.icons.user} ", style=ds.colors.accent_teal)
        prompt_text.append("You", style=f"bold {ds.colors.text_primary}")
        prompt_text.append("  ", style="")
        prompt_text.append("›", style=f"dim {ds.colors.text_tertiary}")
        prompt_text.append(" ", style="")

        self.console.print(prompt_text, end="")

    def print_thinking(self):
        thinking = Text()
        thinking.append("┃ ", style=f"bold {ds.colors.border_bright}")
        thinking.append(f"{ds.icons.thinking} ", style=ds.colors.accent_purple)
        thinking.append("Assistant", style=f"bold {ds.colors.primary_500}")
        thinking.append(" is processing", style=ds.colors.text_secondary)
        thinking.append("...", style=f"dim italic {ds.colors.text_tertiary}")

        self.console.print(thinking)

    def print_assistant_header(self):
        header = Text()
        header.append("\n┃ ", style=f"bold {ds.colors.border_bright}")
        header.append(f"{ds.icons.sparkle} ", style=ds.colors.accent_purple)
        header.append("Assistant", style=f"bold {ds.colors.primary_500}")

        self.console.print(header)

    def print_response(self, response: str):
        self.print_assistant_header()

        cleaned_response = response.strip()

        lines = cleaned_response.split('\n')
        normalized_lines = []

        for line in lines:
            if line.startswith('     '):
                normalized_lines.append('  ' + line.lstrip())
            else:
                normalized_lines.append(line)

        cleaned_response = '\n'.join(normalized_lines)

        md = Markdown(
            cleaned_response,
            code_theme="monokai",
            inline_code_lexer="python",
            inline_code_theme="monokai",
            justify="left"
        )

        response_panel = Panel(
            md,
            border_style=ds.colors.border,
            box=ds.box_styles.panel_subtle,
            padding=ds.spacing.padding_lg,
            title=f"[{ds.colors.text_tertiary}]{ds.icons.arrow_right} Response[/]",
            title_align="left"
        )

        self.console.print(response_panel)
        self.console.print()

        self.session_stats['successes'] += 1

    def print_streaming_response(self, response_generator):
        full_response = ""
        for chunk in response_generator:
            full_response += chunk

        self.print_response(full_response)

    def print_tool_call(self, agent_name: str, tool_name: str):
        self.agent_calls[agent_name] = self.agent_calls.get(agent_name, 0) + 1
        self.session_stats['agent_calls_by_type'][agent_name] = \
            self.session_stats['agent_calls_by_type'].get(agent_name, 0) + 1

        status = Text()
        status.append("┃  ", style=f"bold {ds.colors.border_bright}")
        status.append(f"{ds.icons.lightning} ", style=ds.colors.accent_amber)
        status.append("Calling ", style=ds.colors.text_tertiary)
        status.append(agent_name.replace('_', ' ').title(), style=f"bold {ds.colors.accent_purple}")

        if tool_name and tool_name != "processing...":
            status.append(f" {ds.icons.arrow_right} ", style=ds.colors.text_tertiary)
            status.append(tool_name, style=ds.colors.text_secondary)

        self.console.print(status)

    def print_tool_result(self, success: bool, message: Optional[str] = None):
        status = Text()
        status.append("┃  ", style=f"bold {ds.colors.border_bright}")

        if success:
            status.append(f"{ds.icons.success} ", style=ds.colors.success)
            status.append("Success", style=f"bold {ds.colors.success}")
            if message:
                truncated = message[:60] + "..." if len(message) > 60 else message
                status.append(f" {ds.icons.bullet} ", style=ds.colors.text_tertiary)
                status.append(truncated, style=ds.colors.text_secondary)
            self.session_stats['successes'] += 1
        else:
            status.append(f"{ds.icons.error} ", style=ds.colors.error)
            status.append("Failed", style=f"bold {ds.colors.error}")
            if message:
                truncated = message[:60] + "..." if len(message) > 60 else message
                status.append(f" {ds.icons.bullet} ", style=ds.colors.text_tertiary)
                status.append(truncated, style=ds.colors.error_light)
            self.session_stats['errors'] += 1

        self.console.print(status)

    def print_error(self, error: str, traceback_str: Optional[str] = None):
        self.console.print()
        self.session_stats['errors'] += 1

        error_content = Text()
        error_content.append(f"{ds.icons.error} Error\n\n", style=f"bold {ds.colors.error}")
        error_content.append(error, style=ds.colors.error_light)

        error_panel = Panel(
            error_content,
            title=f"[{ds.colors.error}]⚠ Error Occurred[/]",
            border_style=ds.colors.error,
            box=ds.box_styles.panel_emphasis,
            padding=ds.spacing.padding_lg
        )
        self.console.print(error_panel)

        if traceback_str and self.verbose:
            self.console.print(f"\n[{ds.colors.text_tertiary}]Stack Trace:[/]")
            syntax = Syntax(
                traceback_str,
                "python",
                theme="monokai",
                line_numbers=False,
                background_color=ds.colors.background
            )
            self.console.print(Padding(syntax, (0, 2)))

        self.console.print()

    def show_notification(self, message: str, type: str = "info"):
        type_config = {
            'success': (ds.icons.success, ds.colors.success),
            'error': (ds.icons.error, ds.colors.error),
            'warning': (ds.icons.warning, ds.colors.warning),
            'info': (ds.icons.info, ds.colors.info),
        }

        icon, color = type_config.get(type, (ds.icons.info, ds.colors.info))

        notification = Text()
        notification.append(f"{icon} ", style=color)
        notification.append(message, style=color)

        panel = Panel(
            notification,
            border_style=color,
            box=ds.box_styles.minimal,
            padding=ds.spacing.padding_sm
        )

        self.console.print(panel)
        self.console.print()

    def print_session_stats(self, stats: Optional[Dict[str, Any]] = None):
        self.console.print()

        duration = datetime.now() - self.session_start
        hours = int(duration.total_seconds() // 3600)
        minutes = int((duration.total_seconds() % 3600) // 60)
        seconds = int(duration.total_seconds() % 60)

        if hours > 0:
            duration_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            duration_str = f"{minutes}m {seconds}s"
        else:
            duration_str = f"{seconds}s"

        header = Text()
        header.append(f"{ds.icons.calendar} ", style=ds.colors.accent_purple)
        header.append("Session Summary", style=f"bold {ds.colors.primary_500}")

        self.console.rule(header, style=ds.colors.primary_500)
        self.console.print()

        stats_table = Table(
            show_header=False,
            border_style=ds.colors.border,
            box=ds.box_styles.panel_default,
            padding=(0, 3),
            expand=True
        )

        stats_table.add_row(
            f"[{ds.colors.text_secondary}]{ds.icons.clock} Duration[/]",
            f"[bold {ds.colors.text_primary}]{duration_str}[/]"
        )
        stats_table.add_row(
            f"[{ds.colors.text_secondary}]{ds.icons.user} Messages[/]",
            f"[bold {ds.colors.text_primary}]{self.message_count}[/]"
        )
        stats_table.add_row(
            f"[{ds.colors.text_secondary}]{ds.icons.agent} Agent Calls[/]",
            f"[bold {ds.colors.text_primary}]{sum(self.agent_calls.values())}[/]"
        )

        total_ops = self.session_stats['successes'] + self.session_stats['errors']
        if total_ops > 0:
            success_rate = (self.session_stats['successes'] / total_ops) * 100
            success_color = ds.colors.success if success_rate > 80 else ds.colors.warning if success_rate > 50 else ds.colors.error
            stats_table.add_row(
                f"[{ds.colors.text_secondary}]{ds.icons.star} Success Rate[/]",
                f"[bold {success_color}]{success_rate:.1f}%[/]"
            )

        self.console.print(Padding(stats_table, (0, 2)))

        if self.agent_calls:
            self.console.print()
            self.console.print(f"[{ds.colors.text_secondary}]Most Used Agents:[/]")

            sorted_agents = sorted(self.agent_calls.items(), key=lambda x: x[1], reverse=True)
            for agent, count in sorted_agents[:5]:
                agent_name = agent.replace('_', ' ').title()
                icon = self._get_agent_icon(agent)
                self.console.print(f"  [{ds.colors.accent_purple}]{icon} {agent_name}[/]: [{ds.colors.accent_teal}]{count}[/]")

        self.console.print()

    def print_goodbye(self):
        self.console.print()

        goodbye_text = Text()
        goodbye_text.append(f"{ds.icons.wave} ", style="")
        goodbye_text.append("Goodbye! ", style=f"bold {ds.colors.text_primary}")
        goodbye_text.append("Thanks for using the AI Workspace Orchestrator.", style=ds.colors.text_secondary)
        goodbye_text.append("\n")
        goodbye_text.append(f"{ds.icons.sparkle} ", style=ds.colors.accent_teal)
        goodbye_text.append("Have a great day!", style=f"italic {ds.colors.accent_teal}")

        panel = Panel(
            Align.center(goodbye_text),
            border_style=ds.colors.primary_500,
            box=ds.box_styles.panel_default,
            padding=ds.spacing.padding_md
        )

        self.console.print(panel)
        self.console.print()

    def print_help(self):
        self.console.clear()

        header = Text()
        header.append(f"{ds.icons.info} ", style=ds.colors.info)
        header.append("Help & Keyboard Shortcuts", style=f"bold {ds.colors.primary_500}")

        self.console.rule(header, style=ds.colors.primary_500)
        self.console.print()

        commands_table = Table(
            title=f"[bold {ds.colors.accent_purple}]{ds.icons.gear} Available Commands[/]",
            show_header=True,
            header_style=f"bold {ds.colors.text_primary}",
            border_style=ds.colors.border,
            box=ds.box_styles.table_default,
            padding=(0, 2)
        )

        commands_table.add_column("Command", style=f"bold {ds.colors.accent_teal}")
        commands_table.add_column("Description", style=ds.colors.text_secondary)

        commands_table.add_row("help", "Show this help screen")
        commands_table.add_row("stats", "Display session statistics")
        commands_table.add_row("agents", "List all available agents")
        commands_table.add_row("clear", "Clear the screen")
        commands_table.add_row("exit / quit", "Exit the application")

        self.console.print(commands_table)
        self.console.print()

        tips_panel = Panel(
            f"[{ds.colors.text_secondary}]{ds.icons.sparkle} Tip: Use natural language to interact with agents\n"
            f"{ds.icons.lightning} Example: \"Create a Jira ticket for the login bug and notify the team on Slack\"\n"
            f"{ds.icons.info} The system will automatically route your request to the appropriate agents[/]",
            title=f"[{ds.colors.accent_amber}]{ds.icons.star} Pro Tips[/]",
            border_style=ds.colors.accent_amber,
            box=ds.box_styles.panel_default,
            padding=ds.spacing.padding_md
        )
        self.console.print(tips_panel)
        self.console.print()

        Prompt.ask(f"\n[{ds.colors.text_tertiary}]Press Enter to continue[/]", default="")

    @contextmanager
    def show_status(self, message: str):
        with self.console.status(
            f"[{ds.colors.primary_500}]{message}...[/]",
            spinner="dots",
            spinner_style=ds.colors.primary_500
        ) as status:
            yield status

    def show_progress(self, description: str, total: Optional[int] = None):
        return Progress(
            SpinnerColumn(spinner_name="dots", style=ds.colors.primary_500),
            TextColumn("[{task.description}]", style=ds.colors.text_secondary),
            BarColumn(
                complete_style=ds.colors.success,
                finished_style=ds.colors.success,
                bar_width=40
            ),
            TextColumn("[{task.percentage:>3.0f}%]", style=ds.colors.text_primary),
            TimeElapsedColumn(),
            console=self.console
        )

    def print_divider(self, text: str = ""):
        if text:
            self.console.rule(f"[{ds.colors.text_tertiary}]{text}[/]", style=ds.colors.border)
        else:
            self.console.print(f"[{ds.colors.border}]{'─' * 70}[/]")


enhanced_ui = EnhancedTerminalUI()
