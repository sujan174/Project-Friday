import os
import sys
import json
from typing import Any, Dict, List

import google.generativeai as genai
import google.generativeai.protos as protos
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Add parent directory to path to import base_agent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from connectors.base_agent import BaseAgent


class Agent(BaseAgent):
    """Specialized agent for Jira operations via MCP using sooperset/mcp-atlassian"""
    
    def __init__(self, verbose: bool = False):
        super().__init__()
        self.session: ClientSession = None
        self.stdio_context = None
        self.model = None
        self.available_tools = []
        self.verbose = verbose  # Enable verbose logging
        
        self.schema_type_map = {
            "string": protos.Type.STRING,
            "number": protos.Type.NUMBER,
            "integer": protos.Type.INTEGER,
            "boolean": protos.Type.BOOLEAN,
            "object": protos.Type.OBJECT,
            "array": protos.Type.ARRAY,
        }
        
        self.system_prompt = """You are a specialized Jira agent with deep expertise in project management, issue tracking, and Atlassian workflows. Your purpose is to help users efficiently manage their work in Jira through intelligent automation and precise execution.

# Your Capabilities

You have access to Jira's full functionality through specialized tools:
- **Issue Management**: Create, update, transition, and delete issues
- **Search & Query**: Execute JQL queries to find and filter issues
- **Collaboration**: Add comments, mentions, and attachments
- **Information Retrieval**: Get detailed issue information, project metadata, and workflows
- **Bulk Operations**: Handle multiple issues efficiently when needed

# Core Principles

**Precision First**: Jira is a structured system with specific formats and requirements. Always:
- Use exact issue keys (format: PROJECT-123)
- Respect required fields for issue creation (project, issue type, summary at minimum)
- Follow proper JQL syntax for searches
- Validate project keys and issue types before operations

**Context Awareness**: Understand the business context behind requests:
- A "bug" isn't just an issue type—it represents a problem that needs tracking and resolution
- Sprint planning involves understanding capacity, priorities, and dependencies
- Status transitions must follow the workflow rules of each project

**Proactive Intelligence**: Don't just execute commands—think ahead:
- If creating an issue requires information not provided, identify what's missing
- When searching, consider what filters would give the most useful results
- Before bulk operations, verify the scope makes sense
- Suggest related actions that might be helpful

# Execution Guidelines

**When Creating Issues**:
1. Ensure you have: project key, issue type, and summary at minimum
2. Add relevant fields: description, priority, assignee, labels, components
3. Consider linking to related issues if context suggests relationships
4. Return the created issue key clearly for reference

**When Searching**:
1. Construct precise JQL based on the user's intent, not just keywords
2. Common patterns:
   - "my open bugs" → `assignee = currentUser() AND type = Bug AND status != Done`
   - "recent updates" → `updated >= -7d ORDER BY updated DESC`
   - "sprint issues" → `sprint = ACTIVE ORDER BY priority DESC`
3. Limit results appropriately—ask if more results are needed

**When Updating Issues**:
1. Confirm the issue key exists and is accessible
2. Specify exactly what fields are changing
3. For status transitions, use the proper transition name (not just the target status)
4. Provide clear confirmation of what changed

**When Adding Comments**:
1. Format comments clearly and professionally
2. Use mentions (@username) when directing comments to specific people
3. Keep comments concise but informative

# Error Handling

If you encounter errors:
- **Authentication issues**: Explain clearly that Jira authentication may need refresh
- **Permission errors**: Inform that the user may lack permissions for the requested action
- **Invalid issue keys**: Suggest using search to find the correct issue
- **Required field errors**: List exactly what fields are missing and their expected format
- **JQL syntax errors**: Explain what's wrong and provide a corrected version

# Output Format

Always structure your responses clearly:
1. **Action Summary**: What you did in plain language
2. **Key Details**: Issue keys, links, or important data
3. **Next Steps**: Suggest logical follow-up actions when appropriate

Example:
"I've created a new bug issue PROJ-456: 'Login button not responsive on mobile'.
- Priority: High
- Assigned to: @john.doe
- Added to Sprint 23

The issue is now in the 'To Do' status and visible in the sprint board."

# Best Practices

- **Be efficient**: Don't make unnecessary API calls. If information was just retrieved, use it.
- **Be accurate**: Double-check issue keys and field names before operations
- **Be helpful**: Provide context and suggestions, not just raw data
- **Be clear**: Use natural language to explain what happened, avoiding Jira jargon when possible
- **Be safe**: For destructive operations, describe what will happen before execution

# CRITICAL: Always Verify Your Actions

**This is extremely important**: After performing ANY update, transition, or modification operation, you MUST verify the change was successful by:

1. **Immediately search or retrieve the affected issues** to confirm the changes
2. **Check the actual current state** matches what you attempted to change
3. **Report discrepancies** if the verification shows the action didn't complete as expected
4. **Never assume success** - always verify with a follow-up query

Example workflow:
- User asks: "Mark KAN-1, KAN-2, KAN-3 as Done"
- You execute: Transition each issue to Done
- You MUST verify: Search for those issues and check their current status
- You report: "Verified: KAN-1 and KAN-2 are now Done. However, KAN-3 is still In Progress - let me retry..."

This verification step is NOT optional. Users depend on accurate information about what actually happened.

# Understanding User Intent

Common request patterns and how to handle them:
- "Create a ticket for X" → Infer appropriate issue type (bug, task, story) from description
- "What's blocking us?" → Search for issues with 'Blocked' status or blocker links
- "Update the status" → Use proper workflow transitions, not direct status changes
- "Show me my work" → Filter for assignee=currentUser with relevant status filters

Remember: You're not just executing commands—you're helping users manage their work more effectively. Think about what they're trying to accomplish and help them get there efficiently."""
    
    async def initialize(self):
        """Connect to the Jira MCP server using sooperset/mcp-atlassian"""
        try:
            # Required environment variables for sooperset/mcp-atlassian
            jira_url = os.environ.get("JIRA_URL")
            jira_username = os.environ.get("JIRA_USERNAME")
            jira_api_token = os.environ.get("JIRA_API_TOKEN")

            if not all([jira_url, jira_username, jira_api_token]):
                raise ValueError(
                    "Missing required environment variables.\n"
                    "Please set the following:\n"
                    "  JIRA_URL - Your full Jira URL (e.g., 'https://mycompany.atlassian.net')\n"
                    "  JIRA_USERNAME - Your Atlassian account email\n"
                    "  JIRA_API_TOKEN - Your Jira API token\n\n"
                    "To create an API token:\n"
                    "1. Go to https://id.atlassian.com/manage-profile/security/api-tokens\n"
                    "2. Click 'Create API token'\n"
                    "3. Give it a label and copy the token\n"
                    "4. Set it in your environment"
                )

            # Remove trailing slash from URL if present
            jira_url = jira_url.rstrip('/')

            # Use the sooperset/mcp-atlassian Docker image
            server_params = StdioServerParameters(
                command="docker",
                args=[
                    "run", "-i", "--rm",
                    "-e", "JIRA_URL",
                    "-e", "JIRA_USERNAME",
                    "-e", "JIRA_API_TOKEN",
                    "ghcr.io/sooperset/mcp-atlassian:latest"
                ],
                env={
                    **os.environ,
                    "JIRA_URL": jira_url,
                    "JIRA_USERNAME": jira_username,
                    "JIRA_API_TOKEN": jira_api_token,
                }
            )
        
            self.stdio_context = stdio_client(server_params)
            stdio, write = await self.stdio_context.__aenter__()
            self.session = ClientSession(stdio, write)
            
            await self.session.__aenter__()
            await self.session.initialize()
            
            # Load tools
            tools_list = await self.session.list_tools()
            self.available_tools = tools_list.tools
            
            if not self.available_tools:
                raise RuntimeError("No tools available from Atlassian MCP server")
            
            # Convert to Gemini format
            gemini_tools = [self._build_function_declaration(tool) for tool in self.available_tools]
            
            # Create model with improved configuration
            self.model = genai.GenerativeModel(
                'models/gemini-2.5-flash',
                system_instruction=self.system_prompt,
                tools=gemini_tools
            )
            
            self.initialized = True
            
        except Exception as e:
            error_msg = str(e)
            troubleshooting = [
                "\nTroubleshooting steps:",
                "1. Ensure Docker is installed and running:",
                "   docker --version",
                "   docker ps",
                "",
                "2. Verify environment variables are set correctly:",
                "   - JIRA_URL (full URL like https://mycompany.atlassian.net)",
                "   - JIRA_USERNAME (your Atlassian account email)",
                "   - JIRA_API_TOKEN (from https://id.atlassian.com/manage-profile/security/api-tokens)",
                "",
                "3. Test your credentials manually:",
                "   curl -u YOUR_EMAIL:YOUR_API_TOKEN https://your-domain.atlassian.net/rest/api/3/myself",
                "",
                "4. Pull the Docker image manually:",
                "   docker pull ghcr.io/sooperset/mcp-atlassian:latest",
                "",
                "5. Test the MCP server directly:",
                "   docker run -i --rm -e JIRA_URL -e JIRA_USERNAME -e JIRA_API_TOKEN ghcr.io/sooperset/mcp-atlassian:latest"
            ]

            raise RuntimeError(
                f"Failed to initialize Jira agent: {error_msg}\n" +
                "\n".join(troubleshooting)
            )
    
    async def get_capabilities(self) -> List[str]:
        """Return Jira capabilities in user-friendly format"""
        if not self.available_tools:
            return ["Jira operations (initializing...)"]
        
        # Group capabilities by category for better readability
        capabilities = []
        for tool in self.available_tools:
            description = tool.description or tool.name
            # Clean up technical tool names for user presentation
            if description:
                capabilities.append(description)
        
        # If we have many tools, provide a summary instead of listing all
        if len(capabilities) > 10:
            return [
                "Create and manage Jira issues",
                "Search issues using JQL",
                "Add comments and collaborate",
                "Manage workflows and transitions",
                f"...and {len(capabilities) - 4} more Jira operations"
            ]
        
        return capabilities
    
    async def execute(self, instruction: str) -> str:
        """Execute a Jira task with enhanced error handling and context awareness"""
        if not self.initialized:
            return self._format_error(Exception("Jira agent not initialized. Please restart the system."))
        
        try:
            chat = self.model.start_chat()
            response = await chat.send_message_async(instruction)
            
            # Handle function calling loop with better tracking
            max_iterations = 10
            iteration = 0
            actions_taken = []
            
            while iteration < max_iterations:
                parts = response.candidates[0].content.parts
                has_function_call = any(
                    hasattr(part, 'function_call') and part.function_call 
                    for part in parts
                )
                
                if not has_function_call:
                    break
                
                # Get function call
                function_call = None
                for part in parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_call = part.function_call
                        break
                
                if not function_call:
                    break
                
                tool_name = function_call.name
                tool_args = self._deep_convert_proto_args(function_call.args)

                # Track action for debugging/logging
                actions_taken.append(tool_name)

                # Log tool call if verbose
                if self.verbose:
                    print(f"\n[JIRA AGENT] Calling tool: {tool_name}")
                    print(f"[JIRA AGENT] Arguments: {json.dumps(tool_args, indent=2)[:500]}")

                # Call the tool via MCP with enhanced error handling
                try:
                    tool_result = await self.session.call_tool(tool_name, tool_args)
                    
                    result_content = []
                    for content in tool_result.content:
                        if hasattr(content, 'text'):
                            result_content.append(content.text)
                    
                    result_text = "\n".join(result_content)
                    if not result_text:
                        result_text = json.dumps(tool_result.content, default=str)

                    # Log tool result if verbose
                    if self.verbose:
                        print(f"[JIRA AGENT] Result: {result_text[:500]}")
                        if tool_result.isError:
                            print(f"[JIRA AGENT] ERROR FLAG SET: {tool_result.isError}")

                    # Send result back
                    response = await chat.send_message_async(
                        genai.protos.Content(
                            parts=[genai.protos.Part(
                                function_response=genai.protos.FunctionResponse(
                                    name=tool_name,
                                    response={"result": result_text}
                                )
                            )]
                        )
                    )
                    
                except Exception as e:
                    # Provide more helpful error messages
                    error_msg = self._format_tool_error(tool_name, str(e), tool_args)

                    # Log error if verbose
                    if self.verbose:
                        print(f"[JIRA AGENT] ERROR calling {tool_name}: {error_msg}")

                    response = await chat.send_message_async(
                        genai.protos.Content(
                            parts=[genai.protos.Part(
                                function_response=genai.protos.FunctionResponse(
                                    name=tool_name,
                                    response={"error": error_msg}
                                )
                            )]
                        )
                    )
                
                iteration += 1
            
            if iteration >= max_iterations:
                return (
                    f"{response.text}\n\n"
                    "Note: Reached maximum operation limit. The task may be incomplete. "
                    "Consider breaking this into smaller requests."
                )
            
            return response.text
            
        except Exception as e:
            return self._format_error(e)
    
    def _format_tool_error(self, tool_name: str, error: str, args: Dict) -> str:
        """Format tool errors with helpful context"""
        error_lower = error.lower()
        
        # Provide specific guidance based on error type
        if "authentication" in error_lower or "unauthorized" in error_lower:
            return (
                f"Authentication error when calling {tool_name}. "
                "Your Jira API token may be invalid or expired. Please verify your JIRA_API_TOKEN."
            )
        elif "permission" in error_lower or "forbidden" in error_lower:
            return (
                f"Permission denied for {tool_name}. "
                "You may not have the required permissions for this operation."
            )
        elif "not found" in error_lower or "404" in error_lower:
            issue_key = args.get('issueKey') or args.get('issueIdOrKey') or args.get('issueId')
            if issue_key:
                return (
                    f"Issue '{issue_key}' not found. "
                    "Please verify the issue key is correct and you have access to it."
                )
            return f"Resource not found when calling {tool_name}. Please check your input."
        elif "required" in error_lower:
            return (
                f"Missing required fields for {tool_name}. "
                f"Error details: {error}"
            )
        else:
            return f"Error calling {tool_name}: {error}"
    
    async def cleanup(self):
        """Disconnect from Jira with proper cleanup"""
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
        except Exception:
            pass  # Suppress cleanup errors
        
        try:
            if self.stdio_context:
                await self.stdio_context.__aexit__(None, None, None)
        except Exception:
            pass  # Suppress cleanup errors
    
    def _build_function_declaration(self, tool: Any) -> protos.FunctionDeclaration:
        """Convert MCP tool to Gemini function declaration"""
        parameters_schema = protos.Schema(type_=protos.Type.OBJECT)
        
        if tool.inputSchema:
            schema = tool.inputSchema
            if "properties" in schema:
                for prop_name, prop_schema in schema["properties"].items():
                    parameters_schema.properties[prop_name] = self._clean_schema(prop_schema)
            
            if "required" in schema:
                parameters_schema.required.extend(schema["required"])
        
        return protos.FunctionDeclaration(
            name=tool.name,
            description=tool.description or "",
            parameters=parameters_schema
        )
    
    def _clean_schema(self, schema: Dict) -> protos.Schema:
        """Convert JSON schema to protobuf schema"""
        schema_pb = protos.Schema()
        if "type" in schema:
            schema_pb.type_ = self.schema_type_map.get(schema["type"], protos.Type.TYPE_UNSPECIFIED)
        if "description" in schema:
            schema_pb.description = schema["description"]
        if "enum" in schema:
            schema_pb.enum.extend(schema["enum"])
        if "items" in schema and isinstance(schema["items"], dict):
            schema_pb.items = self._clean_schema(schema["items"])
        if "properties" in schema and isinstance(schema["properties"], dict):
            for prop_name, prop_schema in schema["properties"].items():
                schema_pb.properties[prop_name] = self._clean_schema(prop_schema)
        if "required" in schema:
            schema_pb.required.extend(schema["required"])
        return schema_pb
    
    def _deep_convert_proto_args(self, value: Any) -> Any:
        """Convert protobuf types to Python types"""
        type_str = str(type(value))
        if "MapComposite" in type_str:
            return {k: self._deep_convert_proto_args(v) for k, v in value.items()}
        elif "RepeatedComposite" in type_str:
            return [self._deep_convert_proto_args(item) for item in value]
        else:
            return value