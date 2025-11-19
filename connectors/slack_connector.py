"""
Slack Connector - Lightweight MCP Wrapper

Provides direct access to Slack tools without an intermediate LLM.
Domain expertise is injected into the main orchestrator context.

Author: AI System
Version: 2.0 (Single-Agent Architecture)
"""

import os
import asyncio
from typing import Any, Dict, List, Optional
import google.generativeai.protos as protos
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from connectors.base_connector import BaseConnector, ConnectorMetadata, ToolDefinition


class SlackConnector(BaseConnector):
    """
    Slack connector - manages Slack MCP server and provides tools.

    No LLM - tools are called directly by orchestrator.
    System instructions are injected into orchestrator context.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.server_params = StdioServerParameters(
            command="npx",
            args=[
                "-y",
                "@modelcontextprotocol/server-slack",
                os.environ.get("SLACK_TEAM_ID"),
                os.environ.get("SLACK_BOT_TOKEN")
            ],
            env=None
        )

        self.session: Optional[ClientSession] = None
        self.read_stream = None
        self.write_stream = None
        self.stdio_context = None

    async def initialize(self):
        """Initialize Slack MCP connection"""
        self._log("Initializing connection to Slack")

        try:
            # Connect to MCP server
            self.stdio_context = stdio_client(self.server_params)
            self.read_stream, self.write_stream = await self.stdio_context.__aenter__()
            self.session = await ClientSession(
                self.read_stream,
                self.write_stream
            ).__aenter__()

            # Initialize session
            await self.session.initialize()

            # List available tools
            response = await self.session.list_tools()
            self.mcp_tools = response.tools

            self._log(f"Connected successfully. {len(self.mcp_tools)} tools available")

            # Prefetch metadata
            await self._prefetch_slack_metadata()

            self.initialized = True

        except Exception as e:
            self._log(f"Initialization failed: {e}", "error")
            raise

    async def _prefetch_slack_metadata(self):
        """Prefetch Slack workspace metadata for intelligent context"""
        try:
            # Get channel list
            channels_result = await self.session.call_tool(
                "slack_list_channels",
                arguments={}
            )

            if channels_result.content:
                import json
                channels_data = json.loads(channels_result.content[0].text)
                self.metadata_cache['channels'] = channels_data.get('channels', [])

                self._log(f"Cached {len(self.metadata_cache.get('channels', []))} channels")

        except Exception as e:
            self._log(f"Metadata prefetch failed: {e}", "warning")
            # Non-fatal - continue without metadata

    async def get_metadata(self) -> ConnectorMetadata:
        """Return Slack connector metadata"""
        return ConnectorMetadata(
            name="slack",
            domain="communication",
            capabilities=[
                "send messages to channels",
                "send direct messages",
                "search messages and content",
                "list channels and users",
                "manage threads",
                "add reactions",
                "list conversations"
            ],
            system_instructions=self._get_slack_expertise(),
            keywords=[
                'slack', 'message', 'send', 'post', 'channel', 'dm', 'notify',
                'announce', 'thread', 'reply', 'conversation', 'chat', 'team'
            ],
            prefetch_data=self.metadata_cache
        )

    def _get_slack_expertise(self) -> str:
        """Slack domain expertise to inject into orchestrator"""
        return """
# SLACK COMMUNICATION EXPERTISE

You are an elite team communication specialist. When working with Slack:

**Channel Selection**:
- Public channels for transparency
- Private groups for sensitive topics
- DMs for 1:1 conversations
- Always use threads to keep channels organized

**Message Formatting**:
- Lead with the key point
- Use *bold* for emphasis
- Use `code` for technical terms
- Use bullets for lists
- Keep messages concise and scannable

**Notification Strategy**:
- @channel/@here: Team-wide urgency ONLY
- @username: When specific input needed
- No mention: Informational, async-friendly

**Search Intelligence**:
- Use temporal filters: `in:channel after:2025-01-01`
- Use author filters: `from:username`
- Use boolean: `"exact phrase" AND keyword`

**Tools Available**:
- slack_post_message: Send message to channel
- slack_list_channels: List all channels
- slack_search_messages: Search for content
- slack_add_reaction: React to messages
- slack_list_users: List team members

**Channel Context** (prefetched):
{cached_channels}

Always validate channel names before posting.
""".format(
            cached_channels=self._format_cached_channels()
        )

    def _format_cached_channels(self) -> str:
        """Format cached channel data for context"""
        channels = self.metadata_cache.get('channels', [])
        if not channels:
            return "No channels cached yet"

        # Format top 10 most relevant channels
        channel_list = []
        for ch in channels[:10]:
            name = ch.get('name', 'unknown')
            purpose = ch.get('purpose', {}).get('value', '')
            channel_list.append(f"  - #{name}: {purpose[:50]}")

        return "\n".join(channel_list)

    async def get_tool_definitions(self) -> List[ToolDefinition]:
        """Convert MCP tools to ToolDefinitions"""
        tool_defs = []

        for mcp_tool in self.mcp_tools:
            # Convert MCP tool schema to Gemini format
            tool_def = ToolDefinition(
                name=mcp_tool.name,
                description=mcp_tool.description or f"Slack: {mcp_tool.name}",
                parameters=self._convert_mcp_schema(mcp_tool.inputSchema),
                connector_name="slack"
            )
            tool_defs.append(tool_def)

        return tool_defs

    def _convert_mcp_schema(self, input_schema: Dict) -> Dict[str, Any]:
        """Convert MCP input schema to Gemini parameters format"""
        if not input_schema:
            return {"type": "object", "properties": {}}

        # MCP uses JSON Schema, which is compatible with Gemini
        # Just ensure we have the right structure
        return {
            "type": input_schema.get("type", "object"),
            "properties": input_schema.get("properties", {}),
            "required": input_schema.get("required", [])
        }

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a Slack tool via MCP"""
        if not self.initialized:
            raise RuntimeError("Slack connector not initialized")

        self._log(f"Executing tool: {tool_name}")

        try:
            # Call MCP tool
            result = await self.session.call_tool(tool_name, arguments=arguments)

            # Extract result
            if result.content:
                response_text = result.content[0].text
                self._log(f"Tool succeeded: {tool_name}")
                return response_text
            else:
                return "Tool executed successfully (no output)"

        except Exception as e:
            self._log(f"Tool failed: {tool_name} - {e}", "error")
            raise

    async def cleanup(self):
        """Cleanup Slack MCP connection"""
        await super().cleanup()

        if self.session:
            try:
                await self.session.__aexit__(None, None, None)
            except Exception as e:
                self._log(f"Session cleanup error: {e}", "warning")

        if self.stdio_context:
            try:
                await self.stdio_context.__aexit__(None, None, None)
            except Exception as e:
                self._log(f"Stdio cleanup error: {e}", "warning")
