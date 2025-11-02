"""
Enhanced Production Logging System

Features:
- Structured logging with context
- Multiple output formats (console, file, JSON)
- Performance tracking and metrics
- Request/operation tracing
- Log rotation and archiving
- Error tracking with stack traces
- Configurable log levels per module
- Thread-safe operation
"""

import logging
import logging.handlers
import json
import sys
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
from contextvars import ContextVar
import threading
from dataclasses import dataclass, asdict


# ============================================================================
# CONTEXT TRACKING (for tracing operations across components)
# ============================================================================

# Context variables for tracking current operation
current_session_id: ContextVar[Optional[str]] = ContextVar('session_id', default=None)
current_operation_id: ContextVar[Optional[str]] = ContextVar('operation_id', default=None)
current_agent_name: ContextVar[Optional[str]] = ContextVar('agent_name', default=None)


class LogContext:
    """Manages logging context for request tracing"""

    @staticmethod
    def set_session(session_id: str):
        """Set current session ID"""
        current_session_id.set(session_id)

    @staticmethod
    def set_operation(operation_id: str):
        """Set current operation ID"""
        current_operation_id.set(operation_id)

    @staticmethod
    def set_agent(agent_name: str):
        """Set current agent name"""
        current_agent_name.set(agent_name)

    @staticmethod
    def clear():
        """Clear all context"""
        current_session_id.set(None)
        current_operation_id.set(None)
        current_agent_name.set(None)

    @staticmethod
    def get_context() -> Dict[str, Any]:
        """Get current context as dict"""
        return {
            'session_id': current_session_id.get(),
            'operation_id': current_operation_id.get(),
            'agent_name': current_agent_name.get()
        }


# ============================================================================
# PERFORMANCE TRACKING
# ============================================================================

@dataclass
class PerformanceMetrics:
    """Track performance metrics for operations"""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def finish(self, success: bool = True, error_type: Optional[str] = None):
        """Mark operation as finished"""
        import time
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.success = success
        self.error_type = error_type

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for logging"""
        return {k: v for k, v in asdict(self).items() if v is not None}


# ============================================================================
# CUSTOM LOG FORMATTERS
# ============================================================================

class ContextFormatter(logging.Formatter):
    """Formatter that includes context variables"""

    def format(self, record):
        # Add context to record
        context = LogContext.get_context()
        for key, value in context.items():
            if value:
                setattr(record, key, value)
            else:
                setattr(record, key, '-')

        return super().format(record)


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured logging"""

    def format(self, record):
        log_data = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add context
        context = LogContext.get_context()
        log_data.update({k: v for k, v in context.items() if v})

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }

        # Add extra fields
        if hasattr(record, 'extra'):
            log_data.update(record.extra)

        return json.dumps(log_data)


class ColoredFormatter(ContextFormatter):
    """Colored console output for better readability"""

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'

    def format(self, record):
        # Add color
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"

        formatted = super().format(record)

        # Reset levelname for other formatters
        record.levelname = levelname

        return formatted


# ============================================================================
# ENHANCED LOGGER CLASS
# ============================================================================

class EnhancedLogger:
    """
    Production-grade logger with context, performance tracking, and structured logging.

    Features:
    - Automatic context injection (session_id, operation_id, agent_name)
    - Performance metrics tracking
    - Structured logging (JSON + human-readable)
    - Log rotation
    - Per-module log levels
    - Thread-safe operations
    """

    # Class-level logger cache
    _loggers: Dict[str, logging.Logger] = {}
    _lock = threading.Lock()
    _initialized = False
    _config = None

    @classmethod
    def initialize(cls, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the logging system (call once at startup).

        Args:
            config: Configuration dict with keys:
                - log_level: Default log level (DEBUG, INFO, WARNING, ERROR)
                - log_dir: Directory for log files
                - enable_file_logging: Write logs to files
                - enable_json_logging: Write JSON logs
                - enable_console: Show logs in console
                - enable_colors: Use colored console output
                - max_file_size_mb: Max size before rotation
                - backup_count: Number of backup files to keep
                - per_module_levels: Dict of module-specific levels
        """
        with cls._lock:
            if cls._initialized:
                return

            # Default configuration
            cls._config = {
                'log_level': 'INFO',
                'log_dir': 'logs',
                'enable_file_logging': True,
                'enable_json_logging': True,
                'enable_console': True,
                'enable_colors': True,
                'max_file_size_mb': 10,
                'backup_count': 5,
                'per_module_levels': {}
            }

            # Override with provided config
            if config:
                cls._config.update(config)

            # Create log directory
            log_dir = Path(cls._config['log_dir'])
            log_dir.mkdir(exist_ok=True)

            cls._initialized = True

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get or create a logger for a module.

        Args:
            name: Module name (usually __name__)

        Returns:
            Configured logger instance
        """
        # Initialize if not done
        if not cls._initialized:
            cls.initialize()

        # Return cached logger if exists
        if name in cls._loggers:
            return cls._loggers[name]

        with cls._lock:
            # Double-check after acquiring lock
            if name in cls._loggers:
                return cls._loggers[name]

            # Create new logger
            logger = logging.getLogger(name)

            # Determine log level
            per_module = cls._config.get('per_module_levels', {})
            if name in per_module:
                level = getattr(logging, per_module[name].upper(), logging.INFO)
            else:
                level = getattr(logging, cls._config['log_level'].upper(), logging.INFO)

            logger.setLevel(level)

            # Prevent duplicate handlers
            if logger.handlers:
                logger.handlers.clear()

            # Prevent propagation to root logger
            logger.propagate = False

            # Add handlers
            cls._add_handlers(logger, name)

            # Cache and return
            cls._loggers[name] = logger
            return logger

    @classmethod
    def _add_handlers(cls, logger: logging.Logger, name: str):
        """Add configured handlers to logger"""
        log_dir = Path(cls._config['log_dir'])

        # Console handler
        if cls._config['enable_console']:
            console_handler = logging.StreamHandler(sys.stdout)

            if cls._config['enable_colors']:
                console_format = (
                    '%(levelname)s | %(asctime)s | '
                    '%(session_id)s | %(agent_name)s | '
                    '%(name)s:%(funcName)s:%(lineno)d | '
                    '%(message)s'
                )
                console_handler.setFormatter(ColoredFormatter(
                    console_format,
                    datefmt='%Y-%m-%d %H:%M:%S'
                ))
            else:
                console_format = (
                    '%(levelname)s | %(asctime)s | '
                    '%(session_id)s | %(agent_name)s | '
                    '%(name)s:%(funcName)s:%(lineno)d | '
                    '%(message)s'
                )
                console_handler.setFormatter(ContextFormatter(
                    console_format,
                    datefmt='%Y-%m-%d %H:%M:%S'
                ))

            logger.addHandler(console_handler)

        # File handler (human-readable)
        if cls._config['enable_file_logging']:
            file_path = log_dir / f"{name.replace('.', '_')}.log"
            max_bytes = cls._config['max_file_size_mb'] * 1024 * 1024

            file_handler = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=max_bytes,
                backupCount=cls._config['backup_count']
            )

            file_format = (
                '%(asctime)s | %(levelname)-8s | '
                'session=%(session_id)s | op=%(operation_id)s | agent=%(agent_name)s | '
                '%(name)s:%(funcName)s:%(lineno)d | '
                '%(message)s'
            )
            file_handler.setFormatter(ContextFormatter(
                file_format,
                datefmt='%Y-%m-%d %H:%M:%S'
            ))

            logger.addHandler(file_handler)

        # JSON handler (structured logging)
        if cls._config['enable_json_logging']:
            json_path = log_dir / f"{name.replace('.', '_')}.json.log"
            max_bytes = cls._config['max_file_size_mb'] * 1024 * 1024

            json_handler = logging.handlers.RotatingFileHandler(
                json_path,
                maxBytes=max_bytes,
                backupCount=cls._config['backup_count']
            )

            json_handler.setFormatter(JSONFormatter())
            logger.addHandler(json_handler)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_logger(module_name: str) -> logging.Logger:
    """
    Get a logger for a module (convenience function).

    Usage:
        from logging_config import get_logger
        logger = get_logger(__name__)
        logger.info("Message")
    """
    return EnhancedLogger.get_logger(module_name)


def configure_logging(config: Optional[Dict[str, Any]] = None):
    """
    Configure the logging system (call once at startup).

    Usage:
        from logging_config import configure_logging

        configure_logging({
            'log_level': 'DEBUG',
            'enable_colors': True,
            'per_module_levels': {
                'orchestrator': 'DEBUG',
                'connectors.slack_agent': 'INFO'
            }
        })
    """
    EnhancedLogger.initialize(config)


# ============================================================================
# CONTEXT MANAGERS FOR OPERATIONS
# ============================================================================

class operation_context:
    """
    Context manager for tracking operation performance and logging.

    Usage:
        with operation_context('slack_send_message', agent='slack'):
            # Do work
            pass

        # Automatically logs start, end, duration, and any errors
    """

    def __init__(
        self,
        operation_name: str,
        logger: Optional[logging.Logger] = None,
        agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.operation_name = operation_name
        self.logger = logger or get_logger('operations')
        self.agent = agent
        self.metadata = metadata or {}
        self.metrics: Optional[PerformanceMetrics] = None

    def __enter__(self):
        import time
        import uuid

        # Set context
        operation_id = str(uuid.uuid4())[:8]
        LogContext.set_operation(operation_id)

        if self.agent:
            LogContext.set_agent(self.agent)

        # Start tracking
        self.metrics = PerformanceMetrics(
            operation_name=self.operation_name,
            start_time=time.time(),
            metadata=self.metadata
        )

        self.logger.info(
            f"Operation started: {self.operation_name}",
            extra={'metadata': self.metadata}
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Finish tracking
        if exc_type:
            self.metrics.finish(success=False, error_type=exc_type.__name__)
            self.logger.error(
                f"Operation failed: {self.operation_name} "
                f"({self.metrics.duration_ms:.2f}ms)",
                exc_info=(exc_type, exc_val, exc_tb),
                extra={'metrics': self.metrics.to_dict()}
            )
        else:
            self.metrics.finish(success=True)
            self.logger.info(
                f"Operation completed: {self.operation_name} "
                f"({self.metrics.duration_ms:.2f}ms)",
                extra={'metrics': self.metrics.to_dict()}
            )

        # Clear operation context
        LogContext.set_operation(None)
        if self.agent:
            LogContext.set_agent(None)

        # Don't suppress exceptions
        return False


# ============================================================================
# PERFORMANCE TRACKING DECORATOR
# ============================================================================

def track_performance(operation_name: Optional[str] = None):
    """
    Decorator to automatically track function performance.

    Usage:
        @track_performance('user_request_processing')
        def process_request(user_input):
            # Do work
            pass
    """
    def decorator(func):
        import functools

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            logger = get_logger(func.__module__)

            with operation_context(op_name, logger=logger):
                return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            logger = get_logger(func.__module__)

            with operation_context(op_name, logger=logger):
                return await func(*args, **kwargs)

        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Configure logging at startup
    configure_logging({
        'log_level': 'DEBUG',
        'enable_colors': True,
        'enable_json_logging': True,
        'per_module_levels': {
            'test_module': 'DEBUG'
        }
    })

    # Get logger
    logger = get_logger('test_module')

    # Set session context
    LogContext.set_session('session-123')

    # Test different log levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    # Test operation tracking
    with operation_context('test_operation', logger=logger, agent='test_agent'):
        logger.info("Doing some work")
        import time
        time.sleep(0.1)

    # Test error tracking
    try:
        with operation_context('failing_operation', logger=logger):
            raise ValueError("Something went wrong")
    except ValueError:
        pass

    # Test performance decorator
    @track_performance('example_function')
    def example_function(x):
        import time
        time.sleep(0.05)
        return x * 2

    result = example_function(5)
    logger.info(f"Result: {result}")

    print("\n✓ Logging examples complete. Check logs/ directory for output files.")
