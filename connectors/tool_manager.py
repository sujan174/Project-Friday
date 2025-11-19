"""
Intelligent Tool Manager - Dynamic Tool Loading System

Responsibilities:
1. Analyze user intent to determine needed connectors
2. Dynamically load/unload tools based on conversation
3. Inject domain-specific context
4. Manage connector lifecycle
5. Optimize token usage by loading only relevant tools

Author: AI System
Version: 2.0
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import asyncio
from dataclasses import dataclass
import re

from connectors.base_connector import BaseConnector, ConnectorMetadata, ToolDefinition


@dataclass
class LoadedContext:
    """Represents currently loaded tools and context"""
    active_connectors: Set[str]  # Currently loaded connector names
    loaded_tools: List[ToolDefinition]  # Currently available tools
    injected_instructions: str  # Combined system instructions
    last_update: float  # Timestamp of last update


class IntelligentToolManager:
    """
    Manages dynamic tool loading based on conversation context.

    Strategy:
    - Start with minimal/most common tools (Slack, Jira)
    - Expand tool set based on detected intent
    - Inject domain expertise as needed
    - Unload unused tools to save tokens
    """

    def __init__(
        self,
        connectors: Dict[str, BaseConnector],
        intent_classifier: Optional[Any] = None,
        verbose: bool = False
    ):
        self.connectors = connectors  # All available connectors
        self.intent_classifier = intent_classifier
        self.verbose = verbose

        # Connector metadata cache
        self.metadata_cache: Dict[str, ConnectorMetadata] = {}

        # Currently loaded context
        self.loaded_context: Optional[LoadedContext] = None

        # Load history for smart prefetching
        self.usage_history: Dict[str, int] = {}

        # Always-load connectors (most frequently used)
        self.core_connectors = {'slack', 'jira'}

    async def initialize(self):
        """Initialize tool manager and prefetch metadata"""
        if self.verbose:
            print("[TOOL MANAGER] Initializing...")

        # Fetch metadata from all connectors
        for name, connector in self.connectors.items():
            if connector.is_initialized():
                metadata = await connector.get_metadata()
                self.metadata_cache[name] = metadata

        if self.verbose:
            print(f"[TOOL MANAGER] Cached metadata for {len(self.metadata_cache)} connectors")

    async def analyze_and_load_tools(
        self,
        user_message: str,
        conversation_context: Optional[List[str]] = None
    ) -> Tuple[List[ToolDefinition], str]:
        """
        Analyze user intent and load relevant tools.

        Args:
            user_message: Current user message
            conversation_context: Recent conversation history

        Returns:
            (tool_definitions, system_instructions)
        """
        # Detect which connectors are needed
        needed_connectors = await self._detect_needed_connectors(
            user_message,
            conversation_context
        )

        # Always include core connectors
        needed_connectors.update(self.core_connectors)

        if self.verbose:
            print(f"[TOOL MANAGER] Loading connectors: {sorted(needed_connectors)}")

        # Load tools from needed connectors
        tools = []
        instructions_parts = []

        for connector_name in needed_connectors:
            if connector_name in self.connectors:
                connector = self.connectors[connector_name]

                # Get tool definitions
                connector_tools = await connector.get_tool_definitions()
                tools.extend(connector_tools)

                # Get domain instructions
                metadata = self.metadata_cache.get(connector_name)
                if metadata and metadata.system_instructions:
                    instructions_parts.append(
                        f"# {metadata.domain.upper()} ({connector_name})\n"
                        f"{metadata.system_instructions}\n"
                    )

                # Track usage
                self.usage_history[connector_name] = \
                    self.usage_history.get(connector_name, 0) + 1

        # Combine instructions
        combined_instructions = "\n".join(instructions_parts)

        # Update loaded context
        self.loaded_context = LoadedContext(
            active_connectors=needed_connectors,
            loaded_tools=tools,
            injected_instructions=combined_instructions,
            last_update=asyncio.get_event_loop().time()
        )

        if self.verbose:
            print(f"[TOOL MANAGER] Loaded {len(tools)} tools from "
                  f"{len(needed_connectors)} connectors")

        return tools, combined_instructions

    async def _detect_needed_connectors(
        self,
        user_message: str,
        conversation_context: Optional[List[str]] = None
    ) -> Set[str]:
        """
        Detect which connectors are needed based on message analysis.

        Strategy:
        1. Keyword matching (fast)
        2. Intent classification (if available)
        3. Conversation context analysis
        """
        needed = set()
        message_lower = user_message.lower()

        # Keyword-based detection (fast path)
        for connector_name, metadata in self.metadata_cache.items():
            if any(keyword in message_lower for keyword in metadata.keywords):
                needed.add(connector_name)

        # Intent-based detection (if classifier available)
        if self.intent_classifier:
            try:
                intents = self.intent_classifier.classify(user_message)
                # Map intents to connectors
                for intent in intents[:3]:  # Top 3 intents
                    connector = self._intent_to_connector(intent)
                    if connector:
                        needed.add(connector)
            except Exception as e:
                if self.verbose:
                    print(f"[TOOL MANAGER] Intent classification failed: {e}")

        # Context-based detection
        if conversation_context:
            # If recent messages mention a connector, keep it loaded
            recent_text = " ".join(conversation_context[-3:]).lower()
            for connector_name, metadata in self.metadata_cache.items():
                if any(keyword in recent_text for keyword in metadata.keywords):
                    needed.add(connector_name)

        # Smart prefetching: If user mentions "and", might need multiple
        if " and " in message_lower or "also " in message_lower:
            # Add frequently used connectors
            top_used = sorted(
                self.usage_history.items(),
                key=lambda x: x[1],
                reverse=True
            )[:2]
            for conn_name, _ in top_used:
                needed.add(conn_name)

        return needed

    def _intent_to_connector(self, intent: Any) -> Optional[str]:
        """Map an intent to a connector name"""
        # This would use the intent type to determine connector
        # For now, simple string matching
        intent_str = str(intent).lower()

        mapping = {
            'slack': 'slack',
            'message': 'slack',
            'send': 'slack',
            'notify': 'slack',
            'jira': 'jira',
            'issue': 'jira',
            'ticket': 'jira',
            'bug': 'jira',
            'github': 'github',
            'repo': 'github',
            'pr': 'github',
            'pull request': 'github',
            'notion': 'notion',
            'page': 'notion',
            'document': 'notion',
            'scrape': 'scraper',
            'web': 'scraper',
            'browse': 'browser',
            'navigate': 'browser',
        }

        for keyword, connector in mapping.items():
            if keyword in intent_str:
                return connector

        return None

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """
        Execute a tool call through the appropriate connector.

        Args:
            tool_name: Tool to execute
            arguments: Tool arguments

        Returns:
            Tool result
        """
        # Find which connector owns this tool
        connector_name = self._find_tool_connector(tool_name)

        if not connector_name:
            raise ValueError(f"Tool '{tool_name}' not found in any connector")

        if connector_name not in self.connectors:
            raise ValueError(f"Connector '{connector_name}' not available")

        connector = self.connectors[connector_name]

        # Execute through connector
        try:
            result = await connector.execute_tool(tool_name, arguments)
            connector.track_operation(success=True)
            return result
        except Exception as e:
            connector.track_operation(success=False)
            raise

    def _find_tool_connector(self, tool_name: str) -> Optional[str]:
        """Find which connector owns a tool"""
        if not self.loaded_context:
            return None

        for tool_def in self.loaded_context.loaded_tools:
            if tool_def.name == tool_name:
                return tool_def.connector_name

        return None

    def get_current_context(self) -> Optional[LoadedContext]:
        """Get currently loaded context"""
        return self.loaded_context

    def get_usage_stats(self) -> Dict[str, int]:
        """Get connector usage statistics"""
        return self.usage_history.copy()
