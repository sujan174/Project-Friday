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

current_session_id: ContextVar[Optional[str]] = ContextVar('session_id', default=None)
current_operation_id: ContextVar[Optional[str]] = ContextVar('operation_id', default=None)
current_agent_name: ContextVar[Optional[str]] = ContextVar('agent_name', default=None)

try:
    from .distributed_tracing import TraceContext
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False
    TraceContext = None


class LogContext:
    @staticmethod
    def set_session(session_id: str):
        current_session_id.set(session_id)

    @staticmethod
    def set_operation(operation_id: str):
        current_operation_id.set(operation_id)

    @staticmethod
    def set_agent(agent_name: str):
        current_agent_name.set(agent_name)

    @staticmethod
    def clear():
        current_session_id.set(None)
        current_operation_id.set(None)
        current_agent_name.set(None)

    @staticmethod
    def get_context() -> Dict[str, Any]:
        context = {
            'session_id': current_session_id.get(),
            'operation_id': current_operation_id.get(),
            'agent_name': current_agent_name.get()
        }

        if TRACING_AVAILABLE and TraceContext:
            trace_ctx = TraceContext.get_context()
            context['trace_id'] = trace_ctx.get('trace_id')
            context['span_id'] = trace_ctx.get('span_id')

        return context


@dataclass
class PerformanceMetrics:
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def finish(self, success: bool = True, error_type: Optional[str] = None):
        import time
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.success = success
        self.error_type = error_type

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class ContextFormatter(logging.Formatter):
    def format(self, record):
        context = LogContext.get_context()
        for key, value in context.items():
            if value:
                setattr(record, key, value)
            else:
                setattr(record, key, '-')

        return super().format(record)


class JSONFormatter(logging.Formatter):
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

        context = LogContext.get_context()
        log_data.update({k: v for k, v in context.items() if v})

        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }

        if hasattr(record, 'extra'):
            log_data.update(record.extra)

        return json.dumps(log_data)


class ColoredFormatter(ContextFormatter):
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"

        formatted = super().format(record)
        record.levelname = levelname

        return formatted


class EnhancedLogger:
    _loggers: Dict[str, logging.Logger] = {}
    _lock = threading.Lock()
    _initialized = False
    _config = None

    @classmethod
    def initialize(cls, config: Optional[Dict[str, Any]] = None):
        with cls._lock:
            if cls._initialized:
                return

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

            if config:
                cls._config.update(config)

            log_dir = Path(cls._config['log_dir'])
            log_dir.mkdir(exist_ok=True)

            cls._initialized = True

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        if not cls._initialized:
            cls.initialize()

        if name in cls._loggers:
            return cls._loggers[name]

        with cls._lock:
            if name in cls._loggers:
                return cls._loggers[name]

            logger = logging.getLogger(name)

            per_module = cls._config.get('per_module_levels', {})
            if name in per_module:
                level = getattr(logging, per_module[name].upper(), logging.INFO)
            else:
                level = getattr(logging, cls._config['log_level'].upper(), logging.INFO)

            logger.setLevel(level)

            if logger.handlers:
                logger.handlers.clear()

            logger.propagate = False

            cls._add_handlers(logger, name)

            cls._loggers[name] = logger
            return logger

    @classmethod
    def _add_handlers(cls, logger: logging.Logger, name: str):
        log_dir = Path(cls._config['log_dir'])

        if cls._config['enable_console']:
            console_handler = logging.StreamHandler(sys.stdout)

            if cls._config['enable_colors']:
                console_format = (
                    '%(levelname)s | %(asctime)s | '
                    'trace=%(trace_id)s | span=%(span_id)s | '
                    'session=%(session_id)s | agent=%(agent_name)s | '
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
                    'trace=%(trace_id)s | span=%(span_id)s | '
                    'session=%(session_id)s | agent=%(agent_name)s | '
                    '%(name)s:%(funcName)s:%(lineno)d | '
                    '%(message)s'
                )
                console_handler.setFormatter(ContextFormatter(
                    console_format,
                    datefmt='%Y-%m-%d %H:%M:%S'
                ))

            logger.addHandler(console_handler)

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
                'trace=%(trace_id)s | span=%(span_id)s | '
                'session=%(session_id)s | op=%(operation_id)s | agent=%(agent_name)s | '
                '%(name)s:%(funcName)s:%(lineno)d | '
                '%(message)s'
            )
            file_handler.setFormatter(ContextFormatter(
                file_format,
                datefmt='%Y-%m-%d %H:%M:%S'
            ))

            logger.addHandler(file_handler)

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


def get_logger(module_name: str) -> logging.Logger:
    return EnhancedLogger.get_logger(module_name)


def configure_logging(config: Optional[Dict[str, Any]] = None):
    EnhancedLogger.initialize(config)


class operation_context:
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

        operation_id = str(uuid.uuid4())[:8]
        LogContext.set_operation(operation_id)

        if self.agent:
            LogContext.set_agent(self.agent)

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

        LogContext.set_operation(None)
        if self.agent:
            LogContext.set_agent(None)

        return False


def track_performance(operation_name: Optional[str] = None):
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

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


if __name__ == "__main__":
    configure_logging({
        'log_level': 'DEBUG',
        'enable_colors': True,
        'enable_json_logging': True,
        'per_module_levels': {
            'test_module': 'DEBUG'
        }
    })

    logger = get_logger('test_module')
    LogContext.set_session('session-123')

    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    with operation_context('test_operation', logger=logger, agent='test_agent'):
        logger.info("Doing some work")
        import time
        time.sleep(0.1)

    try:
        with operation_context('failing_operation', logger=logger):
            raise ValueError("Something went wrong")
    except ValueError:
        pass

    @track_performance('example_function')
    def example_function(x):
        import time
        time.sleep(0.05)
        return x * 2

    result = example_function(5)
    logger.info(f"Result: {result}")

    print("\n✓ Logging examples complete. Check logs/ directory for output files.")
