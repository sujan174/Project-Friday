"""
Core Package - Infrastructure and utilities

Provides core infrastructure components:
- Logging: Session and intelligence logging
- Error handling: Classification and recovery
- Retry management: Exponential backoff
- Circuit breaker: Failure isolation
- Analytics: Performance tracking
- Observability: Tracing and metrics

Author: AI System
Version: 1.0
"""

from .logger import get_logger
from .error_handler import ErrorClassifier, ErrorClassification
from .retry_manager import RetryManager
from .circuit_breaker import CircuitBreaker
from .input_validator import InputValidator
from .undo_manager import UndoManager

__all__ = [
    'get_logger',
    'ErrorClassifier',
    'ErrorClassification',
    'RetryManager',
    'CircuitBreaker',
    'InputValidator',
    'UndoManager',
]

__version__ = '1.0.0'
