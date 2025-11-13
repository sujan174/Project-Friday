import logging
from core.logging_config import EnhancedLogger, configure_logging, get_logger as _get_logger


class Logger:
    _loggers = {}

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        return _get_logger(name)


def get_logger(module_name: str) -> logging.Logger:
    return _get_logger(module_name)


try:
    from config import Config
    configure_logging({
        'log_level': getattr(Config, 'LOG_LEVEL', 'INFO'),
        'log_dir': getattr(Config, 'LOG_DIR', 'logs'),
        'enable_file_logging': getattr(Config, 'ENABLE_FILE_LOGGING', True),
        'enable_json_logging': getattr(Config, 'ENABLE_JSON_LOGGING', True),
        'enable_console': getattr(Config, 'ENABLE_CONSOLE_LOGGING', True),
        'enable_colors': getattr(Config, 'ENABLE_COLORED_LOGS', True),
    })
except ImportError:
    configure_logging()
