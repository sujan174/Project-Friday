"""
LLMs Package - Language model abstractions

Provides LLM abstraction layer:
- BaseLLM: Abstract base class for LLM implementations
- GeminiFlash: Google Gemini 2.5 Flash implementation

Author: AI System
Version: 1.0
"""

from .base_llm import (
    BaseLLM,
    LLMConfig,
    LLMResponse,
    ChatSession,
    ChatMessage,
    FunctionCall
)
from .gemini_flash import GeminiFlash

__all__ = [
    'BaseLLM',
    'LLMConfig',
    'LLMResponse',
    'ChatSession',
    'ChatMessage',
    'FunctionCall',
    'GeminiFlash',
]

__version__ = '1.0.0'
