"""
Base Connector Class - Anthropic-Grade Architecture

A connector is a lightweight wrapper around an MCP server that:
1. Manages the MCP connection lifecycle
2. Exposes tool definitions to the orchestrator
3. Handles tool execution
4. Prefetches metadata for intelligent context
5. NO LLM - tools are called directly by the main orchestrator

Author: AI System
Version: 2.0 (Single-Agent Architecture)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ToolDefinition:
    """Structured tool definition for the orchestrator"""
    name: str
    description: str
    parameters: Dict[str, Any]
    connector_name: str  # Which connector owns this tool


@dataclass
class ConnectorMetadata:
    """Metadata about a connector's capabilities"""
    name: str
    domain: str  # e.g., "communication", "project_management", "code"
    capabilities: List[str]
    system_instructions: str  # Domain expertise to inject
    keywords: List[str]  # For intent detection
    prefetch_data: Optional[Dict[str, Any]] = None


class BaseConnector(ABC):
    """
    Base class for all MCP connectors.

    Connectors are stateless tool providers that:
    - Connect to MCP servers
    - Expose tools to the orchestrator
    - Execute tools when called
    - Provide domain expertise via system instructions
    """

    def __init__(
        self,
        verbose: bool = False,
        shared_context: Optional[Any] = None,
        knowledge_base: Optional[Any] = None,
        session_logger: Optional[Any] = None
    ):
        self.verbose = verbose
        self.shared_context = shared_context
        self.knowledge_base = knowledge_base
        self.session_logger = session_logger

        # Connection state
        self.initialized = False
        self.mcp_client = None
        self.mcp_tools = []

        # Metadata cache
        self.metadata_cache: Dict[str, Any] = {}

        # Performance tracking
        self.operation_count = 0
        self.success_count = 0
        self.failure_count = 0

    @abstractmethod
    async def initialize(self):
        """
        Initialize the connector.
        - Connect to MCP server
        - Prefetch metadata
        - Cache tool definitions
        """
        pass

    @abstractmethod
    async def get_metadata(self) -> ConnectorMetadata:
        """
        Return connector metadata for intelligent tool selection.

        Returns:
            ConnectorMetadata with domain expertise and capabilities
        """
        pass

    @abstractmethod
    async def get_tool_definitions(self) -> List[ToolDefinition]:
        """
        Get all tool definitions from this connector.

        Returns:
            List of ToolDefinition objects
        """
        pass

    @abstractmethod
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool call.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        pass

    async def prefetch_metadata(self) -> Dict[str, Any]:
        """
        Prefetch metadata for intelligent context.
        Override in subclasses to prefetch domain-specific data.

        Returns:
            Dict of prefetched metadata
        """
        return {}

    async def cleanup(self):
        """Cleanup connector resources"""
        if self.verbose:
            print(f"[{self.get_name()}] Cleaning up. "
                  f"Operations: {self.operation_count} total, "
                  f"{self.success_count} successful, "
                  f"{self.failure_count} failed "
                  f"({self.success_count/max(1, self.operation_count)*100:.1f}% success rate)")

        self.initialized = False

    def get_name(self) -> str:
        """Get connector name from class name"""
        return self.__class__.__name__.replace('Connector', '').lower()

    def track_operation(self, success: bool):
        """Track operation for analytics"""
        self.operation_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

    def get_system_instructions(self) -> str:
        """
        Get domain-specific instructions to inject into orchestrator context.
        Override in subclasses.
        """
        return ""

    def is_initialized(self) -> bool:
        """Check if connector is initialized"""
        return self.initialized

    def _log(self, message: str, level: str = "info"):
        """Log message if verbose"""
        if self.verbose:
            prefix = f"[{self.get_name().upper()}]"
            print(f"{prefix} {message}")
