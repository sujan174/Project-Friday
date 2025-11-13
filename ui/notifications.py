from typing import Literal, Optional
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.padding import Padding
from datetime import datetime

from ui.design_system import ds


class NotificationManager:

    def __init__(self, console: Optional[Console] = None):
        self.console = console or ds.get_console()
        self.notification_history = []

    def success(self, message: str, title: Optional[str] = None, dismissible: bool = False):
        self._show_notification(
            message=message,
            title=title or "Success",
            icon=ds.icons.success,
            color=ds.colors.success,
            type="success",
            dismissible=dismissible
        )

    def error(self, message: str, title: Optional[str] = None, dismissible: bool = False):
        self._show_notification(
            message=message,
            title=title or "Error",
            icon=ds.icons.error,
            color=ds.colors.error,
            type="error",
            dismissible=dismissible
        )

    def warning(self, message: str, title: Optional[str] = None, dismissible: bool = False):
        self._show_notification(
            message=message,
            title=title or "Warning",
            icon=ds.icons.warning,
            color=ds.colors.warning,
            type="warning",
            dismissible=dismissible
        )

    def info(self, message: str, title: Optional[str] = None, dismissible: bool = False):
        self._show_notification(
            message=message,
            title=title or "Info",
            icon=ds.icons.info,
            color=ds.colors.info,
            type="info",
            dismissible=dismissible
        )

    def custom(
        self,
        message: str,
        title: str,
        icon: str,
        color: str,
        dismissible: bool = False
    ):
        self._show_notification(
            message=message,
            title=title,
            icon=icon,
            color=color,
            type="custom",
            dismissible=dismissible
        )

    def _show_notification(
        self,
        message: str,
        title: str,
        icon: str,
        color: str,
        type: str,
        dismissible: bool = False
    ):
        content = Text()
        content.append(f"{icon} ", style=color)
        content.append(message, style=color)

        panel = Panel(
            Align.center(content),
            title=f"[{color}]{title}[/]" if title else None,
            border_style=color,
            box=ds.box_styles.minimal,
            padding=ds.spacing.padding_sm
        )

        self.console.print()
        self.console.print(panel)

        if dismissible:
            dismiss_text = Text()
            dismiss_text.append("Press Enter to dismiss", style=f"dim {ds.colors.text_tertiary}")
            self.console.print(Align.center(dismiss_text))
            input()

        self.console.print()

        self.notification_history.append({
            'timestamp': datetime.now(),
            'type': type,
            'title': title,
            'message': message
        })

    def progress_start(self, message: str) -> 'ProgressNotification':
        return ProgressNotification(self.console, message)

    def toast(self, message: str, duration: float = 2.0):
        content = Text()
        content.append(f"{ds.icons.sparkle} ", style=ds.colors.accent_teal)
        content.append(message, style=ds.colors.text_secondary)

        self.console.print()
        self.console.print(content)
        self.console.print()


class ProgressNotification:

    def __init__(self, console: Console, initial_message: str):
        self.console = console
        self.start_time = datetime.now()

        self.current_message = initial_message
        self._show()

    def update(self, message: str):
        self.current_message = message
        self._show()

    def complete(self, message: Optional[str] = None):
        if message:
            self.current_message = message

        completion = Text()
        completion.append(f"{ds.icons.success} ", style=ds.colors.success)
        completion.append(self.current_message, style=ds.colors.success)

        duration = (datetime.now() - self.start_time).total_seconds()
        completion.append(f" ({duration:.1f}s)", style=f"dim {ds.colors.text_tertiary}")

        self.console.print()
        self.console.print(completion)
        self.console.print()

    def fail(self, message: Optional[str] = None):
        if message:
            self.current_message = message

        failure = Text()
        failure.append(f"{ds.icons.error} ", style=ds.colors.error)
        failure.append(self.current_message, style=ds.colors.error)

        self.console.print()
        self.console.print(failure)
        self.console.print()

    def _show(self):
        progress = Text()
        progress.append(f"{ds.icons.lightning} ", style=ds.colors.primary_500)
        progress.append(self.current_message, style=ds.colors.text_secondary)
        progress.append("...", style=f"dim {ds.colors.text_tertiary}")

        self.console.print(progress)


notifications = NotificationManager()
