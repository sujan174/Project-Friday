"""
Input validation and sanitization for production safety.
Prevents injection attacks and malformed data.
"""

import re
from typing import Tuple, Optional
from config import Config

class InputValidator:
    """Validates and sanitizes user inputs."""

    @staticmethod
    def validate_instruction(instruction: str) -> Tuple[bool, Optional[str]]:
        """
        Validate instruction string.
        Returns: (is_valid, error_message)
        """
        if not instruction:
            return False, "Instruction cannot be empty"

        if not isinstance(instruction, str):
            return False, f"Instruction must be string, got {type(instruction)}"

        if len(instruction) > Config.MAX_INSTRUCTION_LENGTH:
            return False, f"Instruction exceeds max length of {Config.MAX_INSTRUCTION_LENGTH}"

        # Check for null bytes
        if '\x00' in instruction:
            return False, "Instruction contains null bytes"

        # Check for excessive special characters (potential injection)
        # Use a more lenient threshold that scales with instruction length
        # to allow legitimate code snippets, file paths, and quoted text
        special_count = sum(1 for c in instruction if c in ';\'"\\')
        max_allowed_special_chars = max(100, len(instruction) // 5)  # At least 100, or 20% of length
        if special_count > max_allowed_special_chars:
            return False, "Instruction contains excessive special characters"

        return True, None

    @staticmethod
    def validate_parameter(param_name: str, param_value: any) -> Tuple[bool, Optional[str]]:
        """
        Validate a single parameter.
        Returns: (is_valid, error_message)
        """
        if param_value is None:
            return True, None  # None is valid

        param_str = str(param_value)

        if len(param_str) > Config.MAX_PARAMETER_VALUE_LENGTH:
            return False, f"Parameter '{param_name}' exceeds max length"

        # Check for null bytes
        if '\x00' in param_str:
            return False, f"Parameter '{param_name}' contains null bytes"

        return True, None

    @staticmethod
    def sanitize_for_regex(text: str) -> str:
        """
        Escape text for safe use in regex patterns.
        Prevents ReDoS attacks and regex injection.
        """
        if not isinstance(text, str):
            return ""

        if len(text) > Config.MAX_REGEX_PATTERN_LENGTH:
            text = text[:Config.MAX_REGEX_PATTERN_LENGTH]

        # Use re.escape to safely escape all special regex characters
        return re.escape(text)

    @staticmethod
    def validate_regex_pattern(pattern: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a regex pattern for safety.
        Returns: (is_valid, error_message)
        """
        if len(pattern) > Config.MAX_REGEX_PATTERN_LENGTH:
            return False, f"Regex pattern exceeds max length of {Config.MAX_REGEX_PATTERN_LENGTH}"

        # Check for known dangerous patterns (catastrophic backtracking)
        dangerous_patterns = [
            r'(.*)*',
            r'(.+)*',
            r'(a|a)*',
            r'(a|ab)*',
            r'(a|a?)*',
            r'(.*|.*)*',
        ]

        for dangerous in dangerous_patterns:
            if dangerous in pattern:
                return False, f"Regex pattern contains known dangerous pattern: {dangerous}"

        # Try to compile to catch syntax errors
        try:
            re.compile(pattern)
            return True, None
        except re.error as e:
            return False, f"Invalid regex pattern: {str(e)}"

    @staticmethod
    def extract_jira_keys_safe(text: str) -> list:
        """
        Safely extract Jira issue keys without regex injection.
        Uses pre-compiled, tested pattern.
        """
        if not isinstance(text, str):
            return []

        if len(text) > Config.MAX_INSTRUCTION_LENGTH:
            text = text[:Config.MAX_INSTRUCTION_LENGTH]

        # Pre-compiled pattern - immune to user input
        pattern = re.compile(r'\b([A-Z]+-\d+)\b')
        try:
            matches = pattern.findall(text)
            # Additional validation: keys should be reasonable length
            return [k for k in matches if len(k) < 50]
        except Exception:
            return []

    @staticmethod
    def extract_slack_channel_safe(text: str) -> Optional[str]:
        """
        Safely extract Slack channel name.
        """
        if not isinstance(text, str):
            return None

        # Pre-compiled pattern - handles #channel and @user
        pattern = re.compile(r'(?:to|in|channel)[:\s]+([#@\w\-]+)')
        try:
            match = pattern.search(text)
            if match:
                channel = match.group(1)
                # Validate channel name
                if len(channel) < 100 and channel.replace('#', '').replace('@', '').replace('_', '').isalnum():
                    return channel
        except Exception:
            pass

        return None
