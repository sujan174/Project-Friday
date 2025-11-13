from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass


@dataclass
class LLMConfig:
    model_name: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 0.95
    top_k: int = 40
    stop_sequences: Optional[List[str]] = None
    system_instruction: Optional[str] = None


@dataclass
class ChatMessage:
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class FunctionCall:
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    text: Optional[str] = None
    function_calls: Optional[List[FunctionCall]] = None
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def has_function_calls(self) -> bool:
        return self.function_calls is not None and len(self.function_calls) > 0


class ChatSession(ABC):
    @abstractmethod
    async def send_message(self, message: str) -> LLMResponse:
        pass

    @abstractmethod
    async def send_message_with_functions(
        self,
        message: str,
        function_result: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        pass

    @abstractmethod
    def get_history(self) -> List[ChatMessage]:
        pass


class BaseLLM(ABC):
    """Abstract base class for all LLM providers"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider_name = "unknown"
        self.supports_function_calling = False

    @abstractmethod
    async def generate_content(self, prompt: str) -> LLMResponse:
        pass

    @abstractmethod
    def start_chat(
        self,
        history: Optional[List[ChatMessage]] = None,
        enable_function_calling: bool = False
    ) -> ChatSession:
        pass

    @abstractmethod
    def build_function_declaration(self, tool: Any) -> Any:
        """Convert MCP tool schema to LLM-specific function declaration"""
        pass

    @abstractmethod
    def build_function_response(
        self,
        function_name: str,
        result: Dict[str, Any]
    ) -> Any:
        """Build provider-specific function response format"""
        pass

    @abstractmethod
    def extract_function_calls(self, response: Any) -> List[FunctionCall]:
        """Extract function calls from provider-specific response"""
        pass

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'provider': self.provider_name,
            'model': self.config.model_name,
            'temperature': self.config.temperature,
            'supports_functions': self.supports_function_calling
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model_name}, provider={self.provider_name})"


def clean_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Remove unsupported fields and normalize schema for LLM compatibility"""
    cleaned = {}

    if 'type' in schema:
        cleaned['type'] = schema['type']
    if 'description' in schema:
        cleaned['description'] = schema['description']
    if 'enum' in schema:
        cleaned['enum'] = schema['enum']

    if 'properties' in schema and isinstance(schema['properties'], dict):
        cleaned['properties'] = {}
        for prop_name, prop_schema in schema['properties'].items():
            cleaned['properties'][prop_name] = clean_json_schema(prop_schema)

    if 'items' in schema:
        cleaned['items'] = clean_json_schema(schema['items'])

    if 'required' in schema:
        cleaned['required'] = schema['required']

    return cleaned


def convert_proto_args(value: Any) -> Any:
    """Recursively convert protobuf types to standard Python types"""
    type_str = str(type(value))

    if "MapComposite" in type_str:
        return {k: convert_proto_args(v) for k, v in value.items()}
    elif "RepeatedComposite" in type_str:
        return [convert_proto_args(item) for item in value]
    elif isinstance(value, dict):
        return {k: convert_proto_args(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [convert_proto_args(item) for item in value]
    else:
        return value
