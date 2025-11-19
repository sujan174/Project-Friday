"""
Base LLM - Abstract Foundation for All LLM Providers

This module defines the universal interface that all LLM providers must implement.
It ensures complete abstraction so any agent can work with any LLM provider.

Design Philosophy:
- Provider-agnostic interface
- Supports function calling / tool use
- Async-first design
- MCP tool compatibility
- Easy to swap providers

Author: AI System
Version: 1.0
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """Configuration for LLM models"""
    model_name: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 0.95
    top_k: int = 40
    stop_sequences: Optional[List[str]] = None
    system_instruction: Optional[str] = None


@dataclass
class ChatMessage:
    """Universal chat message format"""
    role: str  # 'user', 'assistant', 'system'
    content: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class FunctionCall:
    """Universal function call format"""
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    """Universal LLM response format"""
    text: Optional[str] = None
    function_calls: Optional[List[FunctionCall]] = None
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def has_function_calls(self) -> bool:
        """Check if response contains function calls"""
        return self.function_calls is not None and len(self.function_calls) > 0


class ChatSession(ABC):
    """Abstract chat session interface"""

    @abstractmethod
    async def send_message(self, message: str) -> LLMResponse:
        """
        Send a message and get response

        Args:
            message: User message

        Returns:
            LLMResponse with text or function calls
        """
        pass

    @abstractmethod
    async def send_message_with_functions(
        self,
        message: str,
        function_result: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """
        Send message with function call results

        Args:
            message: User message
            function_result: Results from previous function call

        Returns:
            LLMResponse
        """
        pass

    @abstractmethod
    def get_history(self) -> List[ChatMessage]:
        """Get chat history"""
        pass


class BaseLLM(ABC):
    """
    Abstract base class for all LLM providers

    This class defines the universal interface that all LLM implementations
    must follow. It ensures that agents can work with any LLM provider
    without modification.

    Supported Providers:
    - Google Gemini (2.5 Flash, 2.5 Pro)
    - Anthropic Claude (Haiku, Sonnet, Opus)
    - OpenAI (GPT-4, GPT-3.5)
    - Others (easily extensible)

    Core Methods:
    - generate_content(): Simple text generation
    - start_chat(): Start conversational session
    - build_function_declaration(): Convert MCP tools to LLM format
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize base LLM

        Args:
            config: LLM configuration
        """
        self.config = config
        self.provider_name = "unknown"
        self.supports_function_calling = False

    @abstractmethod
    async def generate_content(self, prompt: str) -> LLMResponse:
        """
        Generate content from prompt (simple, one-shot generation)

        Args:
            prompt: Text prompt

        Returns:
            LLMResponse with generated text
        """
        pass

    @abstractmethod
    def start_chat(
        self,
        history: Optional[List[ChatMessage]] = None,
        enable_function_calling: bool = False
    ) -> ChatSession:
        """
        Start a chat session

        Args:
            history: Optional chat history to restore
            enable_function_calling: Enable function/tool calling

        Returns:
            ChatSession instance
        """
        pass

    @abstractmethod
    def build_function_declaration(self, tool: Any) -> Any:
        """
        Convert MCP tool schema to LLM-specific function declaration format

        This is critical for MCP compatibility. Each LLM provider has its own
        format for function/tool declarations:
        - Gemini uses protos.FunctionDeclaration
        - Claude uses tool schemas
        - OpenAI uses function schemas

        Args:
            tool: MCP tool object with name, description, inputSchema

        Returns:
            Provider-specific function declaration
        """
        pass

    @abstractmethod
    def build_function_response(
        self,
        function_name: str,
        result: Dict[str, Any]
    ) -> Any:
        """
        Build provider-specific function response format

        After calling an MCP tool, results need to be sent back to the LLM
        in the provider's expected format.

        Args:
            function_name: Name of the function that was called
            result: Result from the function execution

        Returns:
            Provider-specific function response format
        """
        pass

    @abstractmethod
    def extract_function_calls(self, response: Any) -> List[FunctionCall]:
        """
        Extract function calls from provider-specific response

        Each provider returns function calls in different formats.
        This method normalizes them to our universal FunctionCall format.

        Args:
            response: Provider-specific response object

        Returns:
            List of FunctionCall objects
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model

        Returns:
            Dict with model details
        """
        return {
            'provider': self.provider_name,
            'model': self.config.model_name,
            'temperature': self.config.temperature,
            'supports_functions': self.supports_function_calling
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model_name}, provider={self.provider_name})"


# Utility functions for common conversions

def clean_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean and normalize JSON schema for LLM compatibility

    Some providers are strict about schema format. This utility
    removes unsupported fields and normalizes types.

    Args:
        schema: Raw JSON schema

    Returns:
        Cleaned schema
    """
    cleaned = {}

    # Copy basic fields
    if 'type' in schema:
        cleaned['type'] = schema['type']
    if 'description' in schema:
        cleaned['description'] = schema['description']
    if 'enum' in schema:
        cleaned['enum'] = schema['enum']

    # Recursively clean properties
    if 'properties' in schema and isinstance(schema['properties'], dict):
        cleaned['properties'] = {}
        for prop_name, prop_schema in schema['properties'].items():
            cleaned['properties'][prop_name] = clean_json_schema(prop_schema)

    # Handle items for arrays
    if 'items' in schema:
        cleaned['items'] = clean_json_schema(schema['items'])

    # Copy required fields
    if 'required' in schema:
        cleaned['required'] = schema['required']

    return cleaned


def convert_proto_args(value: Any) -> Any:
    """
    Recursively convert protobuf types to standard Python types

    Gemini uses protobuf, which needs conversion. Other providers
    may have similar needs.

    Args:
        value: Value to convert (could be protobuf object, dict, list, etc.)

    Returns:
        Standard Python type
    """
    type_str = str(type(value))

    # Protobuf map type
    if "MapComposite" in type_str:
        return {k: convert_proto_args(v) for k, v in value.items()}

    # Protobuf repeated type
    elif "RepeatedComposite" in type_str:
        return [convert_proto_args(item) for item in value]

    # Regular dict
    elif isinstance(value, dict):
        return {k: convert_proto_args(v) for k, v in value.items()}

    # Regular list
    elif isinstance(value, list):
        return [convert_proto_args(item) for item in value]

    # Primitive type
    else:
        return value
