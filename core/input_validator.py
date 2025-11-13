import re
from typing import Tuple, Optional
from config import Config

class InputValidator:
    @staticmethod
    def validate_instruction(instruction: str) -> Tuple[bool, Optional[str]]:
        if not instruction:
            return False, "Instruction cannot be empty"

        if not isinstance(instruction, str):
            return False, f"Instruction must be string, got {type(instruction)}"

        if len(instruction) > Config.MAX_INSTRUCTION_LENGTH:
            return False, f"Instruction exceeds max length of {Config.MAX_INSTRUCTION_LENGTH}"

        if '\x00' in instruction:
            return False, "Instruction contains null bytes"

        special_count = sum(1 for c in instruction if c in ';\'"\\')
        max_allowed_special_chars = max(100, len(instruction) // 5)
        if special_count > max_allowed_special_chars:
            return False, "Instruction contains excessive special characters"

        return True, None

    @staticmethod
    def validate_parameter(param_name: str, param_value: any) -> Tuple[bool, Optional[str]]:
        if param_value is None:
            return True, None

        if isinstance(param_value, str):
            if len(param_value) > Config.MAX_PARAMETER_VALUE_LENGTH:
                return False, f"Parameter '{param_name}' exceeds max length"

            if '\x00' in param_value:
                return False, f"Parameter '{param_name}' contains null bytes"

        return True, None

    @staticmethod
    def sanitize_regex(pattern: str) -> Tuple[str, bool]:
        if len(pattern) > Config.MAX_REGEX_PATTERN_LENGTH:
            return pattern[:Config.MAX_REGEX_PATTERN_LENGTH], True

        dangerous_patterns = [
            r'\(\?\#',
            r'\(\?\!',
            r'\(\?\<\!',
            r'\(\?\=',
            r'\(\?\<\=',
        ]

        for dangerous in dangerous_patterns:
            if re.search(dangerous, pattern):
                pattern = re.sub(dangerous, '', pattern)
                return pattern, True

        return pattern, False

    @staticmethod
    def validate_agent_name(agent_name: str) -> Tuple[bool, Optional[str]]:
        if not agent_name:
            return False, "Agent name cannot be empty"

        if not isinstance(agent_name, str):
            return False, "Agent name must be a string"

        if not re.match(r'^[a-zA-Z0-9_-]+$', agent_name):
            return False, "Agent name can only contain alphanumeric, underscore, hyphen"

        if len(agent_name) > 50:
            return False, "Agent name too long (max 50 characters)"

        return True, None
