"""
Production configuration for the orchestration system.
All hardcoded values moved here for easy tuning.
"""

import os
from typing import Dict, Any

class Config:
    """Central configuration for all system components."""

    # Agent Operation Timeouts (seconds)
    AGENT_OPERATION_TIMEOUT = float(os.getenv('AGENT_TIMEOUT', '120.0'))
    ENRICHMENT_TIMEOUT = float(os.getenv('ENRICHMENT_TIMEOUT', '5.0'))
    LLM_OPERATION_TIMEOUT = float(os.getenv('LLM_TIMEOUT', '30.0'))

    # Retry Configuration
    MAX_RETRY_ATTEMPTS = int(os.getenv('MAX_RETRIES', '3'))
    RETRY_BACKOFF_FACTOR = float(os.getenv('RETRY_BACKOFF', '2.0'))
    INITIAL_RETRY_DELAY = float(os.getenv('INITIAL_RETRY_DELAY', '1.0'))

    # Input Validation
    MAX_INSTRUCTION_LENGTH = int(os.getenv('MAX_INSTRUCTION_LENGTH', '10000'))
    MAX_PARAMETER_VALUE_LENGTH = int(os.getenv('MAX_PARAM_LENGTH', '5000'))

    # Enrichment Settings
    REQUIRE_ENRICHMENT_FOR_HIGH_RISK = os.getenv('REQUIRE_ENRICHMENT_HIGH_RISK', 'true').lower() == 'true'
    FAIL_OPEN_ON_ENRICHMENT_ERROR = os.getenv('FAIL_OPEN_ENRICHMENT', 'false').lower() == 'true'

    # Security Settings
    ENABLE_INPUT_SANITIZATION = os.getenv('ENABLE_SANITIZATION', 'true').lower() == 'true'
    MAX_REGEX_PATTERN_LENGTH = int(os.getenv('MAX_REGEX_LENGTH', '1000'))

    # Timezone Configuration
    USER_TIMEZONE = os.getenv('USER_TIMEZONE', os.getenv('TZ', 'Asia/Kolkata'))  # Default to IST

    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'WARNING')  # Suppress INFO logs by default
    LOG_DIR = os.getenv('LOG_DIR', 'logs')
    ENABLE_FILE_LOGGING = os.getenv('ENABLE_FILE_LOGGING', 'true').lower() == 'true'
    ENABLE_JSON_LOGGING = os.getenv('ENABLE_JSON_LOGGING', 'true').lower() == 'true'
    ENABLE_CONSOLE_LOGGING = os.getenv('ENABLE_CONSOLE_LOGGING', 'false').lower() == 'true'  # Disable console logs by default
    ENABLE_COLORED_LOGS = os.getenv('ENABLE_COLORED_LOGS', 'true').lower() == 'true'
    MAX_LOG_FILE_SIZE_MB = int(os.getenv('MAX_LOG_FILE_SIZE_MB', '10'))
    LOG_BACKUP_COUNT = int(os.getenv('LOG_BACKUP_COUNT', '5'))
    VERBOSE = os.getenv('VERBOSE', 'false').lower() == 'true'

    # Per-module log levels (can be overridden in environment)
    PER_MODULE_LOG_LEVELS = {
        'orchestrator': os.getenv('LOG_LEVEL_ORCHESTRATOR', LOG_LEVEL),
        'connectors.slack_agent': os.getenv('LOG_LEVEL_SLACK', LOG_LEVEL),
        'connectors.jira_agent': os.getenv('LOG_LEVEL_JIRA', LOG_LEVEL),
        'connectors.github_agent': os.getenv('LOG_LEVEL_GITHUB', LOG_LEVEL),
        'connectors.notion_agent': os.getenv('LOG_LEVEL_NOTION', LOG_LEVEL),
        'error_handler': os.getenv('LOG_LEVEL_ERROR_HANDLER', LOG_LEVEL),
        'intelligence': os.getenv('LOG_LEVEL_INTELLIGENCE', LOG_LEVEL),
    }

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Return all config values as a dictionary."""
        return {
            'agent_timeout': cls.AGENT_OPERATION_TIMEOUT,
            'enrichment_timeout': cls.ENRICHMENT_TIMEOUT,
            'llm_timeout': cls.LLM_OPERATION_TIMEOUT,
            'max_retry_attempts': cls.MAX_RETRY_ATTEMPTS,
            'max_instruction_length': cls.MAX_INSTRUCTION_LENGTH,
            'verbose': cls.VERBOSE,
        }
