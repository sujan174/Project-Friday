from dataclasses import dataclass
from typing import Dict, Literal
from rich.console import Console
from rich.theme import Theme
from rich import box
from rich.style import Style


@dataclass
class ColorPalette:

    primary_50: str = "#E0F7FF"
    primary_100: str = "#B3EDFF"
    primary_200: str = "#80E1FF"
    primary_300: str = "#4DD5FF"
    primary_400: str = "#1ACAFF"
    primary_500: str = "#00BFF0"
    primary_600: str = "#00A8D6"
    primary_700: str = "#0090BD"
    primary_800: str = "#0078A3"
    primary_900: str = "#005B7A"

    accent_purple: str = "#A78BFA"
    accent_pink: str = "#F472B6"
    accent_teal: str = "#2DD4BF"
    accent_amber: str = "#FBBF24"

    success_light: str = "#6EE7B7"
    success: str = "#10B981"
    success_dark: str = "#059669"

    warning_light: str = "#FCD34D"
    warning: str = "#F59E0B"
    warning_dark: str = "#D97706"

    error_light: str = "#FCA5A5"
    error: str = "#EF4444"
    error_dark: str = "#DC2626"

    info_light: str = "#93C5FD"
    info: str = "#3B82F6"
    info_dark: str = "#2563EB"

    gray_50: str = "#F9FAFB"
    gray_100: str = "#F3F4F6"
    gray_200: str = "#E5E7EB"
    gray_300: str = "#D1D5DB"
    gray_400: str = "#9CA3AF"
    gray_500: str = "#6B7280"
    gray_600: str = "#4B5563"
    gray_700: str = "#374151"
    gray_800: str = "#1F2937"
    gray_900: str = "#111827"

    background: str = "#0A0E27"
    background_light: str = "#0F1419"
    surface: str = "#161B22"
    surface_hover: str = "#1C2128"

    text_primary: str = "#F0F6FC"
    text_secondary: str = "#B1BAC4"
    text_tertiary: str = "#7D8590"
    text_disabled: str = "#484F58"

    border: str = "#30363D"
    border_bright: str = "#444C56"

    syntax_keyword: str = "#FF7B72"
    syntax_string: str = "#A5D6FF"
    syntax_function: str = "#D2A8FF"
    syntax_variable: str = "#FFA657"
    syntax_comment: str = "#8B949E"


@dataclass
class Typography:

    mono: str = "monospace"

    size_xs: str = "dim"
    size_sm: str = "default"
    size_base: str = "default"
    size_lg: str = "bold"
    size_xl: str = "bold"
    size_2xl: str = "bold"
    size_3xl: str = "bold"

    weight_normal: str = ""
    weight_medium: str = ""
    weight_semibold: str = "bold"
    weight_bold: str = "bold"

    leading_tight: int = 1
    leading_normal: int = 1
    leading_relaxed: int = 2


@dataclass
class Spacing:

    xs: int = 1
    sm: int = 2
    md: int = 3
    lg: int = 4
    xl: int = 6
    xxl: int = 8
    xxxl: int = 12

    padding_none: tuple = (0, 0)
    padding_sm: tuple = (0, 1)
    padding_md: tuple = (1, 2)
    padding_lg: tuple = (1, 3)
    padding_xl: tuple = (2, 4)


class BoxStyles:

    default = box.ROUNDED
    heavy = box.HEAVY
    double = box.DOUBLE
    minimal = box.MINIMAL
    simple = box.SIMPLE

    panel_default = box.ROUNDED
    panel_emphasis = box.DOUBLE
    panel_subtle = box.MINIMAL
    table_default = box.ROUNDED
    container = box.SQUARE


@dataclass
class Animation:

    instant: float = 0.0
    fast: float = 0.1
    normal: float = 0.2
    slow: float = 0.3
    slower: float = 0.5

    spinner_default: str = "dots"
    spinner_processing: str = "arc"
    spinner_loading: str = "line"
    spinner_bounce: str = "bouncingBall"


class Icons:

    success = "✓"
    error = "✗"
    warning = "⚠"
    info = "ℹ"

    send = "↗"
    receive = "↙"
    edit = "✎"
    delete = "⌫"
    copy = "⎘"
    save = "💾"

    chevron_right = "›"
    chevron_down = "⌄"
    bullet = "•"
    arrow_right = "→"
    arrow_left = "←"

    user = "👤"
    agent = "🤖"
    file = "📄"
    folder = "📁"
    link = "🔗"

    sparkle = "✨"
    rocket = "🚀"
    fire = "🔥"
    star = "⭐"
    clock = "⏱"
    calendar = "📅"

    wave = "👋"
    party = "🎉"
    thinking = "💭"
    lightning = "⚡"

    wrench = "🔧"
    gear = "⚙"
    magnifying_glass = "🔍"

    risk_low = "🟢"
    risk_medium = "🟡"
    risk_high = "🔴"


class Semantic:

    def __init__(self, palette: ColorPalette):
        self.palette = palette

        self.text = {
            'primary': f"bold {palette.text_primary}",
            'secondary': palette.text_secondary,
            'tertiary': f"dim {palette.text_tertiary}",
            'disabled': f"dim {palette.text_disabled}",
            'brand': f"bold {palette.primary_500}",
        }

        self.status = {
            'success': f"bold {palette.success}",
            'warning': f"bold {palette.warning}",
            'error': f"bold {palette.error}",
            'info': f"bold {palette.info}",
        }

        self.component = {
            'header': f"bold {palette.primary_500}",
            'subheader': palette.text_secondary,
            'label': f"bold {palette.text_secondary}",
            'value': palette.text_primary,
            'code': f"{palette.accent_purple}",
            'link': f"underline {palette.info}",
        }

        self.border = {
            'default': palette.border,
            'bright': palette.border_bright,
            'success': palette.success,
            'warning': palette.warning,
            'error': palette.error,
            'info': palette.info,
            'primary': palette.primary_500,
        }


def create_rich_theme() -> Theme:

    palette = ColorPalette()

    return Theme({
        "success": f"bold {palette.success}",
        "warning": f"bold {palette.warning}",
        "error": f"bold {palette.error}",
        "info": f"bold {palette.info}",

        "primary": f"bold {palette.primary_500}",
        "accent": f"bold {palette.accent_purple}",
        "muted": f"dim {palette.text_tertiary}",
        "brand": f"bold {palette.primary_500}",

        "panel.header": f"bold {palette.primary_500}",
        "panel.border": palette.border,
        "table.header": f"bold {palette.text_primary}",
        "code": palette.syntax_keyword,

        "status.loading": f"{palette.info}",
        "status.success": f"bold {palette.success}",
        "status.error": f"bold {palette.error}",

        "bar.complete": palette.success,
        "bar.finished": palette.success,
        "bar.pulse": palette.primary_500,
    })


class DesignSystem:

    def __init__(self):
        self.colors = ColorPalette()
        self.typography = Typography()
        self.spacing = Spacing()
        self.box_styles = BoxStyles()
        self.animation = Animation()
        self.icons = Icons()
        self.semantic = Semantic(self.colors)
        self.theme = create_rich_theme()

    def get_console(self, **kwargs) -> Console:
        return Console(theme=self.theme, **kwargs)


ds = DesignSystem()


def build_status_text(status: Literal['success', 'error', 'warning', 'info'], message: str) -> str:

    icon_map = {
        'success': (ds.icons.success, ds.colors.success),
        'error': (ds.icons.error, ds.colors.error),
        'warning': (ds.icons.warning, ds.colors.warning),
        'info': (ds.icons.info, ds.colors.info),
    }

    icon, color = icon_map[status]
    return f"[{color}]{icon}[/] [{color}]{message}[/]"


def build_badge(text: str, style: Literal['success', 'error', 'warning', 'info', 'neutral'] = 'neutral') -> str:

    style_map = {
        'success': (ds.colors.success, ds.colors.background),
        'error': (ds.colors.error, ds.colors.background),
        'warning': (ds.colors.warning, ds.colors.background),
        'info': (ds.colors.info, ds.colors.background),
        'neutral': (ds.colors.gray_500, ds.colors.background),
    }

    fg, bg = style_map[style]
    return f"[{fg} on {bg}] {text} [/]"


def build_divider(text: str = "", style: str = None) -> str:

    if not style:
        style = ds.colors.border

    if text:
        return f"[{style}]{'─' * 3}[/] [{ds.colors.text_secondary}]{text}[/] [{style}]{'─' * 40}[/]"
    else:
        return f"[{style}]{'─' * 70}[/]"


def build_key_value(key: str, value: str, key_width: int = 20) -> str:

    key_part = f"[{ds.semantic.component['label']}]{key.ljust(key_width)}[/]"
    value_part = f"[{ds.semantic.component['value']}]{value}[/]"

    return f"{key_part} {value_part}"


class Layout:

    @staticmethod
    def center_text(text: str, width: int = 70) -> str:
        padding = (width - len(text)) // 2
        return " " * padding + text

    @staticmethod
    def truncate(text: str, max_length: int = 60, suffix: str = "...") -> str:
        if len(text) <= max_length:
            return text
        return text[:max_length - len(suffix)] + suffix

    @staticmethod
    def wrap_text(text: str, width: int = 70) -> list:
        import textwrap
        return textwrap.wrap(text, width=width)
