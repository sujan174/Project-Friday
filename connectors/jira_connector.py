"""
Jira Connector - Lightweight MCP Wrapper

Provides direct access to Jira tools without an intermediate LLM.
Domain expertise is injected into the main orchestrator context.

Author: AI System
Version: 2.0 (Single-Agent Architecture)
"""

import os
import asyncio
from typing import Any, Dict, List, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from connectors.base_connector import BaseConnector, ConnectorMetadata, ToolDefinition


class JiraConnector(BaseConnector):
    """
    Jira connector - manages Jira MCP server and provides tools.

    Prefetches project/sprint/issue metadata for intelligent context.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.server_params = StdioServerParameters(
            command="npx",
            args=[
                "-y",
                "@modelcontextprotocol/server-atlassian"
            ],
            env={
                **os.environ,
                "ATLASSIAN_INSTANCE_URL": os.environ.get("JIRA_INSTANCE_URL", ""),
                "ATLASSIAN_EMAIL": os.environ.get("JIRA_EMAIL", ""),
                "ATLASSIAN_API_TOKEN": os.environ.get("JIRA_API_TOKEN", "")
            }
        )

        self.session: Optional[ClientSession] = None
        self.read_stream = None
        self.write_stream = None
        self.stdio_context = None

    async def initialize(self):
        """Initialize Jira MCP connection"""
        instance_url = os.environ.get("JIRA_INSTANCE_URL", "")
        self._log(f"Initializing connection to {instance_url}")

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

            # Prefetch Jira metadata
            await self._prefetch_jira_metadata()

            self.initialized = True

        except Exception as e:
            self._log(f"Initialization failed: {e}", "error")
            raise

    async def _prefetch_jira_metadata(self):
        """Prefetch Jira projects, sprints, and common issues"""
        try:
            # Get current user info
            user_result = await self.session.call_tool(
                "jira_get_current_user",
                arguments={}
            )
            if user_result.content:
                import json
                user_data = json.loads(user_result.content[0].text)
                self.metadata_cache['current_user'] = user_data

            self._log("Cached user metadata")

        except Exception as e:
            self._log(f"Metadata prefetch failed: {e}", "warning")

    async def get_metadata(self) -> ConnectorMetadata:
        """Return Jira connector metadata"""
        return ConnectorMetadata(
            name="jira",
            domain="project_management",
            capabilities=[
                "create and update issues",
                "search for issues and tasks",
                "manage sprints and backlogs",
                "transition issue status",
                "assign tasks to team members",
                "add comments to issues"
            ],
            system_instructions=self._get_jira_expertise(),
            keywords=[
                'jira', 'issue', 'ticket', 'bug', 'task', 'story', 'epic',
                'sprint', 'backlog', 'assign', 'status', 'project', 'kanban'
            ],
            prefetch_data=self.metadata_cache
        )

    def _get_jira_expertise(self) -> str:
        """Jira domain expertise to inject into orchestrator"""
        current_user = self.metadata_cache.get('current_user', {})
        user_email = current_user.get('emailAddress', 'unknown')

        return f"""
# JIRA PROJECT MANAGEMENT EXPERTISE

You are an expert Agile project manager. When working with Jira:

**Issue Creation Best Practices**:
- Always include clear summary and description
- Set appropriate issue type (Bug, Task, Story, Epic)
- Add labels for categorization
- Set priority based on urgency/impact
- Assign to appropriate team member

**Search Intelligence**:
- Use JQL for complex queries: `project = KAN AND status = "In Progress"`
- Filter by sprint: `sprint = "Sprint 1"`
- Find bugs: `type = Bug AND priority = High`
- Recent updates: `updated >= -7d`

**Status Transitions**:
- Understand workflow: To Do → In Progress → Done
- Use proper transitions (don't force invalid states)
- Add comments when transitioning to explain

**Current Context**:
- Logged in as: {user_email}
- Default project: Use KAN if not specified

**Tools Available**:
- jira_create_issue: Create new issue/task
- jira_search_issues: Search with JQL
- jira_update_issue: Update existing issue
- jira_add_comment: Comment on issues
- jira_transition_issue: Change issue status
- jira_assign_issue: Assign to user

Always confirm issue keys (like KAN-123) before operations.
"""

    async def get_tool_definitions(self) -> List[ToolDefinition]:
        """Convert MCP tools to ToolDefinitions"""
        tool_defs = []

        for mcp_tool in self.mcp_tools:
            tool_def = ToolDefinition(
                name=mcp_tool.name,
                description=mcp_tool.description or f"Jira: {mcp_tool.name}",
                parameters=self._convert_mcp_schema(mcp_tool.inputSchema),
                connector_name="jira"
            )
            tool_defs.append(tool_def)

        return tool_defs

    def _convert_mcp_schema(self, input_schema: Dict) -> Dict[str, Any]:
        """Convert MCP input schema to Gemini parameters format"""
        if not input_schema:
            return {"type": "object", "properties": {}}

        return {
            "type": input_schema.get("type", "object"),
            "properties": input_schema.get("properties", {}),
            "required": input_schema.get("required", [])
        }

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a Jira tool via MCP"""
        if not self.initialized:
            raise RuntimeError("Jira connector not initialized")

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
                return "Tool executed successfully"

        except Exception as e:
            self._log(f"Tool failed: {tool_name} - {e}", "error")
            raise

    async def cleanup(self):
        """Cleanup Jira MCP connection"""
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
