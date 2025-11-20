"""
Structured logging for production debugging and monitoring.

This is a compatibility wrapper around the enhanced logging system.
For new code, import from logging_config instead.
"""

import logging
from core.logging_config import configure_logging, get_logger as _get_logger

# For backwards compatibility
class Logger:
    """Centralized logging with consistent formatting (legacy wrapper)."""

    _loggers = {}

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get or create a logger with standard configuration.

        This method now uses the enhanced logging system under the hood.
        """
        return _get_logger(name)


def get_logger(module_name: str) -> logging.Logger:
    """
    Convenience function to get a logger for a module.

    Usage:
        from logger import get_logger
        logger = get_logger(__name__)
        logger.info("Message")
    """
    return _get_logger(module_name)


# Initialize logging system with defaults if not already configured
try:
    from config import Config
    configure_logging({
        'log_level': getattr(Config, 'LOG_LEVEL', 'INFO'),
        'log_dir': getattr(Config, 'LOG_DIR', 'logs'),
        'enable_file_logging': False,  # Disabled - using SimpleSessionLogger instead
        'enable_json_logging': False,  # Disabled - using SimpleSessionLogger instead
        'enable_console': getattr(Config, 'ENABLE_CONSOLE_LOGGING', True),
        'enable_colors': getattr(Config, 'ENABLE_COLORED_LOGS', True),
    })
except ImportError:
    # Config not available, use defaults with file logging disabled
    configure_logging({
        'enable_file_logging': False,
        'enable_json_logging': False
    })
