"""Base Agent - Abstract Foundation for All Specialized Agents"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import traceback


def safe_extract_response_text(response: Any) -> str:
    try:
        return response.text
    except Exception as e:
        try:
            if hasattr(response, 'candidates') and response.candidates:
                text_parts = []
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)

                if text_parts:
                    return '\n'.join(text_parts)
        except Exception:
            pass

        return f"⚠️ Response received but could not extract text. Error: {str(e)}"


class BaseAgent(ABC):
    """Abstract base class for all specialized agents"""

    def __init__(self):
        self.name = self.__class__.__name__.lower().replace("agent", "")
        self.initialized = False

    @abstractmethod
    async def initialize(self):
        pass

    @abstractmethod
    async def get_capabilities(self) -> List[str]:
        pass

    @abstractmethod
    async def execute(self, instruction: str) -> str:
        pass

    async def cleanup(self):
        pass

    async def get_action_schema(self) -> Dict[str, Any]:
        return {}

    async def apply_parameter_edits(
        self,
        instruction: str,
        parameter_edits: Dict[str, Any]
    ) -> str:
        updated = instruction
        for param_name, new_value in parameter_edits.items():
            import re
            pattern = rf"{param_name}[:\s]*['\"]([^'\"]+)['\"]"
            updated = re.sub(pattern, f"{param_name}: '{new_value}'", updated)

        return updated

    async def can_edit_parameter(
        self,
        action_type: str,
        parameter_name: str
    ) -> bool:
        sensitive = {'api_key', 'token', 'password', 'secret', 'auth'}

        for sensitive_word in sensitive:
            if sensitive_word in parameter_name.lower():
                return False

        return True

    async def validate_operation(self, instruction: str) -> Dict[str, Any]:
        return {
            'valid': True,
            'missing': [],
            'warnings': [],
            'confidence': 1.0
        }

    def _format_error(self, error: Exception) -> str:
        error_msg = str(error)
        formatted = f"❌ Error in {self.name}: {error_msg}"
        return formatted

    def _format_success(self, message: str, details: Optional[Dict[str, Any]] = None) -> str:
        formatted = f"✓ {message}"

        if details:
            for key, value in details.items():
                formatted += f"\n  {key.title()}: {value}"

        return formatted

    def _get_error_context(self, error: Exception) -> str:
        error_type = type(error).__name__
        error_msg = str(error)
        error_trace = ''.join(traceback.format_tb(error.__traceback__))

        return (
            f"Exception Type: {error_type}\n"
            f"Message: {error_msg}\n"
            f"Traceback:\n{error_trace}"
        )

    def is_initialized(self) -> bool:
        return self.initialized

    def get_name(self) -> str:
        return self.name

    def __repr__(self) -> str:
        status = "initialized" if self.initialized else "not initialized"
        return f"<{self.__class__.__name__} ({self.name}): {status}>"


class AgentError(Exception):
    """Base exception for all agent-related errors"""
    pass


class AgentNotInitializedError(AgentError):
    def __init__(self, agent_name: str):
        super().__init__(f"Agent '{agent_name}' is not initialized. Call initialize() first.")
        self.agent_name = agent_name


class AgentConnectionError(AgentError):
    def __init__(self, agent_name: str, service: str, details: str):
        super().__init__(
            f"Agent '{agent_name}' failed to connect to {service}: {details}"
        )
        self.agent_name = agent_name
        self.service = service


class AgentAuthenticationError(AgentError):
    def __init__(self, agent_name: str, details: str):
        super().__init__(
            f"Authentication failed for agent '{agent_name}': {details}"
        )
        self.agent_name = agent_name


class AgentExecutionError(AgentError):
    def __init__(self, agent_name: str, instruction: str, details: str):
        super().__init__(
            f"Agent '{agent_name}' failed to execute: {details}\n"
            f"Instruction: {instruction}"
        )
        self.agent_name = agent_name
        self.instruction = instruction
