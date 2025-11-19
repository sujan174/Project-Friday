"""
Terminal UI - Professional Interface for Agent System

Provides a clean, branded terminal interface similar to Claude Code
with smooth animations, status indicators, and beautiful formatting.

Author: AI System
Version: 1.0
"""

import sys
import time
import asyncio
from typing import Optional
from datetime import datetime


class Colors:
    """ANSI color codes for terminal styling"""
    # Brand colors
    PRIMARY = '\033[38;5;141m'      # Purple (brand)
    SECONDARY = '\033[38;5;75m'     # Blue
    SUCCESS = '\033[38;5;120m'      # Green
    WARNING = '\033[38;5;214m'      # Orange
    ERROR = '\033[38;5;203m'        # Red
    MUTED = '\033[38;5;245m'        # Gray

    # Text styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'

    # Reset
    RESET = '\033[0m'

    # Background
    BG_PRIMARY = '\033[48;5;141m'
    BG_DARK = '\033[48;5;235m'


class Icons:
    """Unicode icons for terminal UI"""
    # Status
    SUCCESS = '✓'
    ERROR = '✗'
    WARNING = '⚠'
    INFO = 'ℹ'

    # Progress
    SPINNER = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    DOTS = ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷']

    # Shapes
    ARROW = '→'
    BULLET = '•'
    CIRCLE = '○'
    FILLED_CIRCLE = '●'

    # Special
    ROBOT = '🤖'
    SPARKLES = '✨'
    THINKING = '💭'
    WRITING = '✍️'
    SEARCH = '🔍'


class TerminalUI:
    """
    Professional terminal UI for agent system

    Provides a clean, branded interface with smooth animations
    and beautiful formatting.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.spinner_index = 0
        self.last_spinner_update = 0

    def clear_screen(self):
        """Clear the terminal screen"""
        print('\033[2J\033[H', end='', flush=True)

    def clear_line(self):
        """Clear the current line"""
        print('\r\033[K', end='', flush=True)

    def move_cursor_up(self, lines: int = 1):
        """Move cursor up by n lines"""
        print(f'\033[{lines}A', end='', flush=True)

    def hide_cursor(self):
        """Hide the cursor"""
        print('\033[?25l', end='', flush=True)

    def show_cursor(self):
        """Show the cursor"""
        print('\033[?25h', end='', flush=True)

    def print_header(self, session_id: str):
        """Print beautiful branded header"""
        print()
        print(f"{Colors.PRIMARY}{Colors.BOLD}╭{'─' * 58}╮{Colors.RESET}")
        print(f"{Colors.PRIMARY}{Colors.BOLD}│{Colors.RESET}  {Colors.PRIMARY}{Icons.ROBOT}  {Colors.BOLD}Multi-Agent Orchestration System{Colors.RESET}                 {Colors.PRIMARY}{Colors.BOLD}│{Colors.RESET}")
        print(f"{Colors.PRIMARY}{Colors.BOLD}╰{'─' * 58}╮{Colors.RESET}")
        print(f"{Colors.MUTED}  Session: {session_id[:8]}...  {Icons.SPARKLES} Ready to assist{Colors.RESET}")
        print()

    def print_prompt(self):
        """Print user input prompt"""
        print(f"{Colors.PRIMARY}{Colors.BOLD}You{Colors.RESET} {Colors.MUTED}{Icons.ARROW}{Colors.RESET} ", end='', flush=True)

    def print_thinking(self):
        """Print thinking indicator"""
        print(f"\n{Colors.SECONDARY}{Colors.BOLD}Assistant{Colors.RESET} {Colors.MUTED}{Icons.THINKING}  Thinking...{Colors.RESET}")

    def print_agent_status(self, agent_name: str, status: str = "working"):
        """Print agent status"""
        icon = Icons.FILLED_CIRCLE
        color = Colors.SECONDARY

        if status == "success":
            icon = Icons.SUCCESS
            color = Colors.SUCCESS
        elif status == "error":
            icon = Icons.ERROR
            color = Colors.ERROR

        agent_display = agent_name.replace('_', ' ').title()
        print(f"{Colors.MUTED}  {color}{icon}{Colors.RESET}  {Colors.DIM}{agent_display}{Colors.RESET}", flush=True)

    def print_response(self, text: str):
        """Print assistant response with nice formatting"""
        print(f"\n{Colors.PRIMARY}{Colors.BOLD}Assistant{Colors.RESET} {Colors.MUTED}{Icons.ARROW}{Colors.RESET}")
        print()

        # Format the response
        lines = text.split('\n')
        for line in lines:
            if line.strip():
                # Detect markdown headers
                if line.startswith('# '):
                    print(f"{Colors.BOLD}{line[2:]}{Colors.RESET}")
                elif line.startswith('## '):
                    print(f"{Colors.BOLD}{line[3:]}{Colors.RESET}")
                elif line.startswith('###'):
                    print(f"{Colors.DIM}{line[4:]}{Colors.RESET}")
                # Detect code blocks
                elif line.startswith('```'):
                    print(f"{Colors.MUTED}{line}{Colors.RESET}")
                # Detect bullet points
                elif line.strip().startswith(('- ', '* ', '• ')):
                    bullet_text = line.strip()[2:]
                    indent = len(line) - len(line.lstrip())
                    print(f"{' ' * indent}{Colors.SECONDARY}{Icons.BULLET}{Colors.RESET} {bullet_text}")
                else:
                    print(line)
            else:
                print()

        print()

    def print_error(self, message: str):
        """Print error message"""
        print(f"\n{Colors.ERROR}{Icons.ERROR}  {Colors.BOLD}Error{Colors.RESET}")
        print(f"{Colors.ERROR}{message}{Colors.RESET}")
        print()

    def print_warning(self, message: str):
        """Print warning message"""
        print(f"{Colors.WARNING}{Icons.WARNING}  {message}{Colors.RESET}")

    def print_success(self, message: str):
        """Print success message"""
        print(f"{Colors.SUCCESS}{Icons.SUCCESS}  {message}{Colors.RESET}")

    def print_info(self, message: str):
        """Print info message"""
        print(f"{Colors.MUTED}{Icons.INFO}  {message}{Colors.RESET}")

    def print_divider(self, char: str = '─', length: int = 60):
        """Print a divider line"""
        print(f"{Colors.MUTED}{char * length}{Colors.RESET}")

    def start_progress(self, message: str) -> 'ProgressIndicator':
        """Start a progress indicator"""
        return ProgressIndicator(message, self.verbose)

    def print_agent_list(self, agents: list):
        """Print list of available agents"""
        print(f"\n{Colors.MUTED}Available agents:{Colors.RESET}")
        for agent in agents:
            print(f"  {Colors.SECONDARY}{Icons.BULLET}{Colors.RESET} {Colors.DIM}{agent}{Colors.RESET}")
        print()

    def print_goodbye(self):
        """Print goodbye message"""
        print()
        print(f"{Colors.PRIMARY}{Icons.SPARKLES}  Thanks for using the orchestration system!{Colors.RESET}")
        print(f"{Colors.MUTED}  Session saved. Goodbye!{Colors.RESET}")
        print()


class ProgressIndicator:
    """
    Simple progress indicator with spinner

    Usage:
        progress = ui.start_progress("Loading agents")
        # ... do work ...
        progress.stop("Done")
    """

    def __init__(self, message: str, verbose: bool = False):
        self.message = message
        self.verbose = verbose
        self.spinner_chars = Icons.DOTS
        self.spinner_index = 0

        if not verbose:
            self.hide_cursor()

    def update(self, message: str):
        """Update the progress message"""
        self.message = message

    def stop(self, final_message: Optional[str] = None, success: bool = True):
        """Stop the spinner and show final message"""
        if not self.verbose:
            self.clear_line()

            if final_message:
                icon = Icons.SUCCESS if success else Icons.ERROR
                color = Colors.SUCCESS if success else Colors.ERROR
                print(f"{color}{icon}{Colors.RESET}  {Colors.DIM}{final_message}{Colors.RESET}")

            self.show_cursor()

    def clear_line(self):
        """Clear the current line"""
        print('\r\033[K', end='', flush=True)

    def hide_cursor(self):
        """Hide the cursor"""
        print('\033[?25l', end='', flush=True)

    def show_cursor(self):
        """Show the cursor"""
        print('\033[?25h', end='', flush=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class StatusBar:
    """
    Bottom status bar showing current state

    Usage:
        status = StatusBar()
        status.update("Processing...", agent="github")
    """

    def __init__(self):
        self.current_status = ""
        self.current_agent = None

    def update(self, status: str, agent: Optional[str] = None):
        """Update the status bar"""
        self.current_status = status
        self.current_agent = agent
        self._render()

    def _render(self):
        """Render the status bar"""
        # Save cursor position
        print('\033[s', end='')

        # Move to bottom of screen
        print('\033[999;0H', end='')

        # Clear line
        print('\033[K', end='')

        # Print status
        status_text = f"{Colors.BG_DARK} {self.current_status} {Colors.RESET}"
        if self.current_agent:
            status_text += f" {Colors.MUTED}[{self.current_agent}]{Colors.RESET}"

        print(status_text, end='', flush=True)

        # Restore cursor position
        print('\033[u', end='', flush=True)

    def clear(self):
        """Clear the status bar"""
        print('\033[s\033[999;0H\033[K\033[u', end='', flush=True)


# Create global UI instance
ui = TerminalUI()
