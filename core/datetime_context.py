"""
DateTime Context Utility

Provides consistent datetime and timezone context for all agents and the orchestrator.
This ensures all components have access to the current time and can apply user timezone preferences.

Author: AI System
Version: 1.0
"""

import os
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
import zoneinfo

from config import Config


class DateTimeContext:
    """
    Provides datetime context that can be injected into agent instructions.

    Supports:
    - System timezone detection
    - User timezone preferences (from explicit instructions)
    - Multiple format outputs (ISO, human-readable, etc.)
    - Relative time calculations
    """

    # Common timezone abbreviations and their full names
    TIMEZONE_ABBREVIATIONS = {
        # US Timezones
        'EST': 'America/New_York',
        'EDT': 'America/New_York',
        'CST': 'America/Chicago',
        'CDT': 'America/Chicago',
        'MST': 'America/Denver',
        'MDT': 'America/Denver',
        'PST': 'America/Los_Angeles',
        'PDT': 'America/Los_Angeles',
        'AKST': 'America/Anchorage',
        'AKDT': 'America/Anchorage',
        'HST': 'Pacific/Honolulu',

        # European Timezones
        'GMT': 'Europe/London',
        'BST': 'Europe/London',
        'CET': 'Europe/Paris',
        'CEST': 'Europe/Paris',
        'EET': 'Europe/Helsinki',
        'EEST': 'Europe/Helsinki',

        # Asian Timezones
        'IST': 'Asia/Kolkata',
        'JST': 'Asia/Tokyo',
        'KST': 'Asia/Seoul',
        'CST_ASIA': 'Asia/Shanghai',
        'HKT': 'Asia/Hong_Kong',
        'SGT': 'Asia/Singapore',

        # Australian Timezones
        'AEST': 'Australia/Sydney',
        'AEDT': 'Australia/Sydney',
        'ACST': 'Australia/Adelaide',
        'ACDT': 'Australia/Adelaide',
        'AWST': 'Australia/Perth',

        # Other
        'UTC': 'UTC',
        'NZST': 'Pacific/Auckland',
        'NZDT': 'Pacific/Auckland',
    }

    def __init__(self, user_timezone: Optional[str] = None, verbose: bool = False):
        """
        Initialize datetime context.

        Args:
            user_timezone: User's preferred timezone (abbreviation or full name)
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self._user_timezone = None

        if user_timezone:
            self.set_user_timezone(user_timezone)

    def set_user_timezone(self, timezone_str: str) -> bool:
        """
        Set user's preferred timezone.

        Args:
            timezone_str: Timezone abbreviation (EST, PST) or full name (America/New_York)

        Returns:
            True if timezone was set successfully, False otherwise
        """
        # Normalize to uppercase for abbreviation lookup
        tz_upper = timezone_str.upper().strip()

        # Check if it's an abbreviation
        if tz_upper in self.TIMEZONE_ABBREVIATIONS:
            tz_name = self.TIMEZONE_ABBREVIATIONS[tz_upper]
        else:
            # Assume it's a full timezone name
            tz_name = timezone_str

        # Validate timezone
        try:
            zoneinfo.ZoneInfo(tz_name)
            self._user_timezone = tz_name

            # Also update Config.USER_TIMEZONE so all agents use the same timezone
            Config.USER_TIMEZONE = tz_name

            if self.verbose:
                print(f"[DATETIME] Set user timezone: {timezone_str} -> {tz_name}")

            return True
        except Exception as e:
            if self.verbose:
                print(f"[DATETIME] Invalid timezone '{timezone_str}': {e}")
            return False

    def _detect_system_timezone(self) -> str:
        """Detect system timezone"""
        # Try to get from TZ environment variable
        tz_name = os.environ.get('TZ', '')
        if tz_name:
            return tz_name

        # Try to detect on Linux
        if os.path.exists('/etc/timezone'):
            try:
                with open('/etc/timezone', 'r') as f:
                    return f.read().strip()
            except:
                pass

        # Try to detect from /etc/localtime symlink
        if os.path.exists('/etc/localtime'):
            try:
                result = subprocess.run(
                    ['readlink', '-f', '/etc/localtime'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    path = result.stdout.strip()
                    if 'zoneinfo/' in path:
                        return path.split('zoneinfo/')[-1]
            except:
                pass

        # Fallback to UTC
        return 'UTC'

    def get_timezone(self) -> zoneinfo.ZoneInfo:
        """Get the current timezone object (user timezone or system default)"""
        if self._user_timezone:
            tz_name = self._user_timezone
        else:
            tz_name = self._detect_system_timezone()

        try:
            return zoneinfo.ZoneInfo(tz_name)
        except:
            return zoneinfo.ZoneInfo('UTC')

    def get_timezone_name(self) -> str:
        """Get the current timezone name"""
        if self._user_timezone:
            return self._user_timezone
        return self._detect_system_timezone()

    def get_current_time_context(self) -> Dict:
        """
        Get comprehensive current time context.

        Returns a dict with:
        - current_time: ISO format datetime
        - current_date: YYYY-MM-DD format
        - timezone: Full timezone name
        - timezone_abbr: Timezone abbreviation
        - timezone_offset: UTC offset string
        - day_of_week: Full day name
        - formatted_time: Human-readable time (e.g., "3:30 PM")
        - formatted_date: Human-readable date (e.g., "January 15, 2025")
        - formatted_datetime: Combined human-readable (e.g., "Wednesday, January 15, 2025 at 3:30 PM")
        - unix_timestamp: Unix timestamp
        - is_weekend: Boolean
        - is_business_hours: Boolean (9 AM - 5 PM on weekdays)
        """
        tz = self.get_timezone()
        now = datetime.now(tz)

        # Calculate UTC offset
        offset = now.utcoffset()
        if offset:
            total_seconds = int(offset.total_seconds())
            hours, remainder = divmod(abs(total_seconds), 3600)
            minutes = remainder // 60
            sign = '+' if total_seconds >= 0 else '-'
            offset_str = f"UTC{sign}{hours:02d}:{minutes:02d}"
        else:
            offset_str = "UTC+00:00"

        # Get timezone abbreviation
        tz_name = self.get_timezone_name()
        tz_abbr = now.strftime('%Z') or tz_name.split('/')[-1]

        # Determine business hours and weekend
        is_weekend = now.weekday() >= 5
        is_business_hours = not is_weekend and 9 <= now.hour < 17

        return {
            'current_time': now.isoformat(),
            'current_date': now.strftime('%Y-%m-%d'),
            'timezone': tz_name,
            'timezone_abbr': tz_abbr,
            'timezone_offset': offset_str,
            'day_of_week': now.strftime('%A'),
            'formatted_time': now.strftime('%I:%M %p').lstrip('0'),
            'formatted_date': now.strftime('%B %d, %Y'),
            'formatted_datetime': now.strftime('%A, %B %d, %Y at %I:%M %p').replace(' 0', ' '),
            'unix_timestamp': int(now.timestamp()),
            'is_weekend': is_weekend,
            'is_business_hours': is_business_hours,
            'year': now.year,
            'month': now.month,
            'day': now.day,
            'hour': now.hour,
            'minute': now.minute
        }

    def format_for_prompt(self) -> str:
        """
        Format datetime context for injection into system prompts.

        Returns:
            Formatted string suitable for inclusion in agent instructions
        """
        ctx = self.get_current_time_context()

        lines = [
            "# Current Date & Time Context",
            "",
            f"**Current Time:** {ctx['formatted_time']} {ctx['timezone_abbr']}",
            f"**Current Date:** {ctx['formatted_date']} ({ctx['day_of_week']})",
            f"**Timezone:** {ctx['timezone']} ({ctx['timezone_offset']})",
            "",
            "Use this information when:",
            "- Scheduling events (e.g., 'tomorrow at 2pm' = day after current date)",
            "- Understanding relative time references",
            "- Converting between timezones",
            "- Determining business hours or availability"
        ]

        return "\n".join(lines)

    def format_for_instruction(self) -> str:
        """
        Format datetime context for inline injection into agent instructions.

        Returns:
            Compact formatted string for instruction injection
        """
        ctx = self.get_current_time_context()

        return (
            f"[Current Time: {ctx['formatted_time']} on {ctx['day_of_week']}, "
            f"{ctx['formatted_date']} ({ctx['timezone_abbr']})]"
        )

    def get_relative_date(self, days_offset: int) -> Dict:
        """
        Get date information for a relative offset from today.

        Args:
            days_offset: Number of days from today (positive = future, negative = past)

        Returns:
            Dict with date information
        """
        tz = self.get_timezone()
        target = datetime.now(tz) + timedelta(days=days_offset)

        return {
            'date': target.strftime('%Y-%m-%d'),
            'day_of_week': target.strftime('%A'),
            'formatted_date': target.strftime('%B %d, %Y'),
            'is_weekend': target.weekday() >= 5
        }


# Global instance for easy access
_global_datetime_context: Optional[DateTimeContext] = None


def get_datetime_context() -> DateTimeContext:
    """
    Get the global datetime context instance.

    Returns:
        DateTimeContext instance
    """
    global _global_datetime_context
    if _global_datetime_context is None:
        _global_datetime_context = DateTimeContext()
    return _global_datetime_context


def set_user_timezone(timezone_str: str) -> bool:
    """
    Set the user's preferred timezone globally.

    Args:
        timezone_str: Timezone abbreviation (EST, PST) or full name

    Returns:
        True if successful
    """
    return get_datetime_context().set_user_timezone(timezone_str)


def get_current_time_context() -> Dict:
    """
    Get current time context from global instance.

    Returns:
        Dict with comprehensive time information
    """
    return get_datetime_context().get_current_time_context()


def format_datetime_for_prompt() -> str:
    """
    Format datetime context for system prompt injection.

    Returns:
        Formatted string for prompts
    """
    return get_datetime_context().format_for_prompt()


def format_datetime_for_instruction() -> str:
    """
    Format datetime context for instruction injection.

    Returns:
        Compact formatted string
    """
    return get_datetime_context().format_for_instruction()
