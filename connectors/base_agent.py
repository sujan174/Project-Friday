"""
Base Agent - Abstract Foundation for All Specialized Agents

This module defines the base contract that all specialized agents must implement.
It provides the foundation for a consistent, predictable agent architecture across
the entire multi-agent orchestration system.

Design Philosophy:
- Abstract interface enforces consistency
- Minimal required implementation burden
- Extensible for agent-specific needs
- Clear lifecycle management (init → execute → cleanup)

Author: AI System
Version: 2.0
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import traceback


def safe_extract_response_text(response: Any) -> str:
    """
    Safely extract text from Gemini response objects.

    Handles cases where response.text quick accessor fails due to function calls.
    This is a common issue across all agents using the Gemini API.

    Args:
        response: A Gemini response object

    Returns:
        str: Extracted text or error message
    """
    try:
        # Try the quick accessor first (fast path)
        return response.text
    except Exception as e:
        # If that fails, try manual extraction from response parts
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

        # Graceful degradation: return error message instead of crashing
        return f"⚠️ Response received but could not extract text. Error: {str(e)}"


# ============================================================================
# BASE AGENT ABSTRACT CLASS
# ============================================================================

class BaseAgent(ABC):
    """
    Abstract base class for all specialized agents in the orchestration system

    This class defines the core contract that all agents must implement. It ensures
    consistency across different agent types while allowing flexibility for
    agent-specific implementations.

    ## Agent Lifecycle

    1. **Initialization**: `__init__()` + `initialize()`
       - Constructor sets up basic properties
       - initialize() connects to external services, loads tools, etc.

    2. **Discovery**: `get_capabilities()`
       - Returns list of what the agent can do
       - Used by orchestrator to route requests appropriately

    3. **Execution**: `execute(instruction)`
       - Receives natural language instructions
       - Performs actions using underlying tools/APIs
       - Returns natural language results

    4. **Cleanup**: `cleanup()`
       - Disconnects from services
       - Releases resources
       - Called when agent is no longer needed

    ## Implementation Guide

    To create a new agent, inherit from BaseAgent and implement:

    ```python
    from connectors.base_agent import BaseAgent

    class MyAgent(BaseAgent):
        def __init__(self, verbose=False):
            super().__init__()
            # Agent-specific initialization
            self.verbose = verbose

        async def initialize(self):
            # Connect to services, load tools
            self.initialized = True

        async def get_capabilities(self) -> List[str]:
            return ["Capability 1", "Capability 2"]

        async def execute(self, instruction: str) -> str:
            if not self.initialized:
                return self._format_error(Exception("Agent not initialized"))
            # Execute instruction
            return "Result"

        async def cleanup(self):
            # Release resources
            pass
    ```

    Attributes:
        name (str): Human-readable name of the agent (auto-derived from class name)
        initialized (bool): Whether the agent has been successfully initialized
    """

    def __init__(self):
        """
        Initialize base agent properties

        This is called by child classes via super().__init__(). It sets up
        the basic properties all agents need.
        """
        # Derive agent name from class name (e.g., "JiraAgent" -> "jira")
        self.name = self.__class__.__name__.lower().replace("agent", "")

        # Track initialization state
        self.initialized = False

    # ========================================================================
    # ABSTRACT METHODS - Must be implemented by child classes
    # ========================================================================

    @abstractmethod
    async def initialize(self):
        """
        Initialize the agent and connect to external services

        This method is called once after the agent is instantiated. It should:
        - Connect to required external services (APIs, databases, etc.)
        - Load available tools or capabilities
        - Initialize AI models if needed
        - Set self.initialized = True on success

        Raises:
            RuntimeError: If initialization fails
            ValueError: If required environment variables are missing

        Example:
            async def initialize(self):
                api_key = os.environ.get("SERVICE_API_KEY")
                if not api_key:
                    raise ValueError("SERVICE_API_KEY must be set")

                await self.connect_to_service(api_key)
                self.initialized = True
        """
        pass

    @abstractmethod
    async def get_capabilities(self) -> List[str]:
        """
        Return a list of capabilities this agent provides

        This is used by the orchestrator to understand what the agent can do
        and decide when to delegate tasks to it. Capabilities should be:
        - Human-readable descriptions
        - Specific enough to be actionable
        - Broad enough to cover the agent's domain

        Returns:
            List[str]: User-friendly capability descriptions

        Example:
            async def get_capabilities(self) -> List[str]:
                return [
                    "Create and update Jira tickets",
                    "Search Jira issues using JQL",
                    "Add comments and manage workflows",
                    "Generate reports from Jira data"
                ]
        """
        pass

    @abstractmethod
    async def execute(self, instruction: str) -> str:
        """
        Execute a task based on natural language instruction

        This is the main entry point for agent execution. The agent receives
        a natural language instruction and should:
        - Interpret the instruction using AI or rule-based logic
        - Execute appropriate actions using available tools
        - Handle errors gracefully
        - Return a natural language result

        Args:
            instruction: Natural language instruction describing the task
                        Examples: "Create a ticket for bug X"
                                "Find all issues assigned to John"
                                "Update issue ABC-123 to Done"

        Returns:
            str: Natural language response describing what was done
                 Examples: "Created ticket ABC-456: Bug X"
                          "Found 5 issues assigned to John: [list]"
                          "Updated ABC-123 to Done status"

        Notes:
            - Always check self.initialized before executing
            - Use self._format_error() for error messages
            - Include specific details (IDs, links, etc.) in responses
            - Suggest next steps when appropriate

        Example:
            async def execute(self, instruction: str) -> str:
                if not self.initialized:
                    return self._format_error(Exception("Agent not initialized"))

                try:
                    # Parse instruction and execute
                    result = await self.perform_action(instruction)
                    return f"Successfully completed: {result}"
                except Exception as e:
                    return self._format_error(e)
        """
        pass

    # ========================================================================
    # OPTIONAL METHODS - Can be overridden by child classes
    # ========================================================================

    async def cleanup(self):
        """
        Clean up resources and disconnect from services

        This method is called when the agent is being shut down. It should:
        - Close connections to external services
        - Release file handles or network resources
        - Cancel pending operations if possible
        - NOT raise exceptions (suppress them gracefully)

        The default implementation does nothing. Override if your agent
        needs cleanup logic.

        Example:
            async def cleanup(self):
                try:
                    if self.session:
                        await self.session.close()
                except Exception:
                    pass  # Suppress cleanup errors
        """
        pass

    async def get_action_schema(self) -> Dict[str, Any]:
        """
        OPTIONAL: Return schema describing possible actions this agent can perform.

        Default implementation returns empty dict (agent doesn't support schema).

        Expected format:
        {
            'create': {
                'action_type': 'create',
                'risk_level': 'medium',
                'parameters': {
                    'title': {
                        'type': 'string',
                        'display_label': 'Issue Title',
                        'editable': True,
                        'required': True,
                        'constraints': {
                            'min_length': 5,
                            'max_length': 255
                        }
                    },
                    'project': {
                        'type': 'string',
                        'editable': False,  # Can't change which project
                        'required': True
                    }
                }
            }
        }

        Returns:
            Dict[str, Any]: Schema describing agent's actions and parameters
        """
        return {}

    async def apply_parameter_edits(
        self,
        instruction: str,
        parameter_edits: Dict[str, Any]
    ) -> str:
        """
        OPTIONAL: Apply user's edited parameters back into the instruction.

        The orchestrator will call this when a user has edited parameters
        before confirming. You need to merge the edits back into the instruction
        in a format your agent's execute() method can understand.

        Args:
            instruction: Original instruction from LLM
            parameter_edits: {field_name: new_value} from user

        Returns:
            str: Modified instruction with edits applied

        Example for Jira:
            Original: "Create issue with title 'Bug' description 'System crash'"
            Edits: {'title': 'Critical: System crash'}
            Return: "Create issue with title 'Critical: System crash' description 'System crash'"
        """
        # Default: Simple string replacement
        updated = instruction
        for param_name, new_value in parameter_edits.items():
            # Try to find and replace parameter values in instruction
            # This is simplistic - agents should override for complex logic
            import re
            pattern = rf"{param_name}[:\s]*['\"]([^'\"]+)['\"]"
            updated = re.sub(pattern, f"{param_name}: '{new_value}'", updated)

        return updated

    async def can_edit_parameter(
        self,
        action_type: str,
        parameter_name: str
    ) -> bool:
        """
        OPTIONAL: Return whether a parameter can be edited by user.

        Override if your agent has constraints on which parameters
        users can modify. Default is to allow editing of all non-sensitive parameters.

        Args:
            action_type: Type of action ('create', 'update', etc.)
            parameter_name: Name of the parameter

        Returns:
            bool: True if parameter can be edited, False if read-only
        """
        # Don't allow editing of sensitive data
        sensitive = {'api_key', 'token', 'password', 'secret', 'auth'}

        for sensitive_word in sensitive:
            if sensitive_word in parameter_name.lower():
                return False

        return True

    async def validate_operation(self, instruction: str) -> Dict[str, Any]:
        """
        Validate if an operation can be performed before execution (Feature #14)

        This method performs quick pre-flight checks to catch errors early:
        - Are required parameters available?
        - Do we have proper authentication?
        - Does the target resource exist?

        The default implementation returns valid=True. Override in child classes
        for agent-specific validation logic.

        Args:
            instruction: The instruction to validate

        Returns:
            Dict with structure:
            {
                'valid': bool,              # Can this operation proceed?
                'missing': List[str],       # Missing required information
                'warnings': List[str],      # Non-blocking warnings
                'confidence': float         # 0.0-1.0, how confident we are
            }

        Example:
            async def validate_operation(self, instruction: str) -> Dict:
                if 'create issue' in instruction.lower():
                    if not self.metadata_cache.get('projects'):
                        return {
                            'valid': False,
                            'missing': ['project information'],
                            'warnings': [],
                            'confidence': 0.0
                        }
                return {'valid': True, 'missing': [], 'warnings': [], 'confidence': 1.0}
        """
        # Default: assume operation is valid
        return {
            'valid': True,
            'missing': [],
            'warnings': [],
            'confidence': 1.0
        }

    # ========================================================================
    # HELPER METHODS - Available to all child classes
    # ========================================================================

    def _format_error(self, error: Exception) -> str:
        """
        Format an error message in a consistent, user-friendly way

        This helper formats errors for user consumption. It includes the
        agent name and error details in a clear format.

        Args:
            error: The exception that occurred

        Returns:
            str: Formatted error message

        Example:
            try:
                result = risky_operation()
            except Exception as e:
                return self._format_error(e)
            # Output: "❌ Error in jira: Connection timeout"
        """
        error_msg = str(error)

        # Add context about which agent failed
        formatted = f"❌ Error in {self.name}: {error_msg}"

        return formatted

    def _format_success(self, message: str, details: Optional[Dict[str, Any]] = None) -> str:
        """
        Format a success message in a consistent way

        This helper formats success messages with optional structured details.

        Args:
            message: Main success message
            details: Optional dictionary of additional details to include

        Returns:
            str: Formatted success message

        Example:
            return self._format_success(
                "Created issue ABC-123",
                {"url": "https://...", "status": "Open"}
            )
            # Output: "✓ Created issue ABC-123
            #          URL: https://...
            #          Status: Open"
        """
        formatted = f"✓ {message}"

        if details:
            for key, value in details.items():
                formatted += f"\n  {key.title()}: {value}"

        return formatted

    def _get_error_context(self, error: Exception) -> str:
        """
        Get detailed context about an error for debugging

        This includes the full traceback, which is useful for verbose mode
        or logging, but not for user-facing messages.

        Args:
            error: The exception that occurred

        Returns:
            str: Detailed error context including traceback

        Example:
            except Exception as e:
                if self.verbose:
                    print(self._get_error_context(e))
                return self._format_error(e)
        """
        error_type = type(error).__name__
        error_msg = str(error)
        error_trace = ''.join(traceback.format_tb(error.__traceback__))

        return (
            f"Exception Type: {error_type}\n"
            f"Message: {error_msg}\n"
            f"Traceback:\n{error_trace}"
        )

    # ========================================================================
    # STATUS AND INFORMATION METHODS
    # ========================================================================

    def is_initialized(self) -> bool:
        """
        Check if the agent has been successfully initialized

        Returns:
            bool: True if initialized, False otherwise
        """
        return self.initialized

    def get_name(self) -> str:
        """
        Get the human-readable name of this agent

        Returns:
            str: Agent name
        """
        return self.name

    def __repr__(self) -> str:
        """
        String representation for debugging

        Returns:
            str: String representation
        """
        status = "initialized" if self.initialized else "not initialized"
        return f"<{self.__class__.__name__} ({self.name}): {status}>"


# ============================================================================
# AGENT EXCEPTIONS - Specialized exceptions for common agent errors
# ============================================================================

class AgentError(Exception):
    """Base exception for all agent-related errors"""
    pass


class AgentNotInitializedError(AgentError):
    """Raised when trying to use an agent that hasn't been initialized"""
    def __init__(self, agent_name: str):
        super().__init__(f"Agent '{agent_name}' is not initialized. Call initialize() first.")
        self.agent_name = agent_name


class AgentConnectionError(AgentError):
    """Raised when an agent fails to connect to external service"""
    def __init__(self, agent_name: str, service: str, details: str):
        super().__init__(
            f"Agent '{agent_name}' failed to connect to {service}: {details}"
        )
        self.agent_name = agent_name
        self.service = service


class AgentAuthenticationError(AgentError):
    """Raised when agent authentication fails"""
    def __init__(self, agent_name: str, details: str):
        super().__init__(
            f"Authentication failed for agent '{agent_name}': {details}"
        )
        self.agent_name = agent_name


class AgentExecutionError(AgentError):
    """Raised when agent fails to execute a task"""
    def __init__(self, agent_name: str, instruction: str, details: str):
        super().__init__(
            f"Agent '{agent_name}' failed to execute: {details}\n"
            f"Instruction: {instruction}"
        )
        self.agent_name = agent_name
        self.instruction = instruction
