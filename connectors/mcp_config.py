"""
MCP Configuration - Shared Settings for All MCP Agents

This module provides centralized configuration for Model Context Protocol (MCP)
agents to ensure consistent timeout, retry, and error handling behavior.

Author: AI System
Version: 1.0
"""

from typing import List


class MCPTimeouts:
    """Timeout configuration for MCP operations"""

    # Connection timeouts
    INITIAL_CONNECTION = 30.0  # seconds - initial server connection
    SESSION_INIT = 10.0        # seconds - session initialization
    TOOL_LIST = 10.0           # seconds - listing available tools

    # Operation timeouts
    TOOL_EXECUTION = 60.0      # seconds - individual tool execution
    SEARCH_OPERATION = 45.0    # seconds - search/query operations
    CREATE_OPERATION = 30.0    # seconds - create operations
    UPDATE_OPERATION = 30.0    # seconds - update operations
    DELETE_OPERATION = 20.0    # seconds - delete operations


class MCPRetryConfig:
    """Retry configuration for MCP operations"""

    MAX_RETRIES = 3
    INITIAL_DELAY = 1.0        # seconds
    MAX_DELAY = 10.0           # seconds
    BACKOFF_FACTOR = 2.0       # exponential backoff multiplier

    # Connection retry settings
    CONNECTION_MAX_RETRIES = 2
    CONNECTION_INITIAL_DELAY = 1.0


class MCPRetryableErrors:
    """Error patterns that should trigger automatic retry"""

    # Network and connection errors
    NETWORK_ERRORS = [
        "timeout",
        "connection",
        "network",
        "ECONNRESET",           # TCP connection reset
        "ETIMEDOUT",            # Connection timeout
        "ECONNREFUSED",         # Connection refused
        "ENETUNREACH",          # Network unreachable
    ]

    # SSE-specific errors (Server-Sent Events)
    SSE_ERRORS = [
        "sse error",
        "sseerror",
        "body timeout",
        "terminated",
        "stream closed",
        "connection closed",
        "fetch failed",
        "stream error",
    ]

    # HTTP errors
    HTTP_ERRORS = [
        "503",                  # Service Unavailable
        "502",                  # Bad Gateway
        "504",                  # Gateway Timeout
        "429",                  # Too Many Requests (but also check rate limit)
    ]

    # Rate limiting
    RATE_LIMIT_ERRORS = [
        "rate_limited",
        "rate limit",
        "too many requests",
        "quota exceeded",
    ]

    # Server errors
    SERVER_ERRORS = [
        "internal server error",
        "service unavailable",
        "bad gateway",
    ]

    @classmethod
    def get_all_retryable_errors(cls) -> List[str]:
        """Get complete list of all retryable error patterns"""
        return (
            cls.NETWORK_ERRORS +
            cls.SSE_ERRORS +
            cls.HTTP_ERRORS +
            cls.RATE_LIMIT_ERRORS +
            cls.SERVER_ERRORS
        )

    @classmethod
    def is_retryable(cls, error_message: str) -> bool:
        """
        Check if an error message indicates a retryable error

        Args:
            error_message: Error message to check

        Returns:
            bool: True if error is retryable
        """
        error_lower = error_message.lower()
        return any(
            pattern in error_lower
            for pattern in cls.get_all_retryable_errors()
        )


class MCPErrorMessages:
    """Standardized error messages for MCP operations"""

    @staticmethod
    def connection_failed(server_name: str, error: str, attempt: int) -> str:
        """Format connection failure message"""
        return (
            f"Failed to connect to {server_name} MCP server after {attempt} attempt(s): {error}\n"
            "Troubleshooting steps:\n"
            "1. Check your internet connection\n"
            "2. Verify the MCP server is available\n"
            "3. Ensure required npm packages are installed\n"
            "4. Check for any authentication requirements\n"
            "5. Try again later if the service is temporarily unavailable"
        )

    @staticmethod
    def operation_timeout(operation: str, timeout: float) -> str:
        """Format operation timeout message"""
        return (
            f"Operation '{operation}' timed out after {timeout}s. "
            "The MCP server may be slow or experiencing issues."
        )

    @staticmethod
    def max_retries_exceeded(operation: str, retries: int) -> str:
        """Format max retries exceeded message"""
        return (
            f"Operation '{operation}' failed after {retries} retries. "
            "The service may be unavailable. Please try again later."
        )

    @staticmethod
    def sse_error(details: str) -> str:
        """Format SSE-specific error message"""
        return (
            f"SSE connection error: {details}\n"
            "This is likely a temporary issue with the server-sent events connection. "
            "The operation will be retried automatically."
        )


# Verbose mode helpers
class MCPVerboseLogger:
    """Helper for consistent verbose logging across MCP agents"""

    @staticmethod
    def log_connection_attempt(agent_name: str, server: str, attempt: int, max_attempts: int):
        """Log connection attempt"""
        retry_msg = f" (attempt {attempt}/{max_attempts})" if attempt > 1 else ""
        print(f"[{agent_name}] Connecting to {server}{retry_msg}")

    @staticmethod
    def log_connection_success(agent_name: str, tools_count: int):
        """Log successful connection"""
        print(f"[{agent_name}] Connected successfully. {tools_count} tools available.")

    @staticmethod
    def log_retry(agent_name: str, operation: str, retry_num: int, max_retries: int, delay: float):
        """Log retry attempt"""
        print(f"[{agent_name}] Retrying {operation} ({retry_num}/{max_retries}) in {delay:.1f}s...")

    @staticmethod
    def log_operation(agent_name: str, tool_name: str, retry_count: int = 0):
        """Log tool operation"""
        retry_info = f" (retry {retry_count})" if retry_count > 0 else ""
        print(f"[{agent_name}] Calling tool: {tool_name}{retry_info}")
