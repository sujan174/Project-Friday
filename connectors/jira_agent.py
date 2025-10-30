"""
Jira Agent - Production-Ready Connector for Atlassian Jira

This module provides a robust, intelligent agent for interacting with Jira through
the Model Context Protocol (MCP). It uses the sooperset/mcp-atlassian Docker image
for reliable connectivity and includes comprehensive error handling, retry logic,
and verification workflows.

Key Features:
- Automatic retry with exponential backoff for transient failures
- Verification of all state-changing operations
- Comprehensive error handling with context-aware messages
- Operation tracking and statistics
- Verbose logging for debugging
- Smart result validation and parsing

Author: AI System
Version: 2.0
"""

import os
import sys
import json
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import google.generativeai as genai
import google.generativeai.protos as protos
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Add parent directory to path to import base_agent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from connectors.base_agent import BaseAgent


# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

class RetryConfig:
    """Configuration for retry logic with exponential backoff"""
    MAX_RETRIES = 3
    INITIAL_DELAY = 1.0  # seconds
    MAX_DELAY = 10.0     # seconds
    BACKOFF_FACTOR = 2.0

    # Error types that should trigger a retry
    RETRYABLE_ERRORS = [
        "timeout",
        "connection",
        "network",
        "rate limit",
        "too many requests",
        "503",
        "502",
        "504"
    ]


class ErrorType(Enum):
    """Classification of error types for better handling"""
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    NOT_FOUND = "not_found"
    VALIDATION = "validation"
    RATE_LIMIT = "rate_limit"
    NETWORK = "network"
    UNKNOWN = "unknown"


@dataclass
class OperationStats:
    """Track statistics for agent operations"""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    retries: int = 0
    tools_called: Dict[str, int] = field(default_factory=dict)

    def record_operation(self, tool_name: str, success: bool, retry_count: int = 0):
        """Record an operation for statistics tracking"""
        self.total_operations += 1
        self.tools_called[tool_name] = self.tools_called.get(tool_name, 0) + 1

        if success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1

        self.retries += retry_count

    def get_summary(self) -> str:
        """Get a human-readable summary of operations"""
        success_rate = (self.successful_operations / self.total_operations * 100) if self.total_operations > 0 else 0
        return (
            f"Operations: {self.total_operations} total, "
            f"{self.successful_operations} successful, "
            f"{self.failed_operations} failed "
            f"({success_rate:.1f}% success rate), "
            f"{self.retries} retries"
        )


# ============================================================================
# MAIN AGENT CLASS
# ============================================================================

class Agent(BaseAgent):
    """
    Specialized agent for Jira operations via MCP using sooperset/mcp-atlassian

    This agent provides intelligent, reliable interaction with Jira through:
    - Automatic verification of state-changing operations
    - Retry logic for transient failures
    - Comprehensive error handling and reporting
    - Operation tracking and statistics

    Usage:
        agent = Agent(verbose=True)
        await agent.initialize()
        result = await agent.execute("Mark issue KAN-1 as Done")
        await agent.cleanup()
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the Jira agent

        Args:
            verbose: Enable detailed logging for debugging (default: False)
        """
        super().__init__()

        # MCP Connection Components
        self.session: ClientSession = None
        self.stdio_context = None
        self.model = None
        self.available_tools = []

        # Configuration
        self.verbose = verbose
        self.stats = OperationStats()

        # Schema type mapping for Gemini
        self.schema_type_map = {
            "string": protos.Type.STRING,
            "number": protos.Type.NUMBER,
            "integer": protos.Type.INTEGER,
            "boolean": protos.Type.BOOLEAN,
            "object": protos.Type.OBJECT,
            "array": protos.Type.ARRAY,
        }

        # System prompt - defines agent behavior and intelligence
        self.system_prompt = self._build_system_prompt()

    # ========================================================================
    # SYSTEM PROMPT - Agent Intelligence and Behavior
    # ========================================================================

    def _build_system_prompt(self) -> str:
        """
        Build the comprehensive system prompt that defines agent behavior

        This prompt is the core of the agent's intelligence, defining how it
        should think, act, and respond to user requests.
        """
        return """You are a specialized Jira agent with deep expertise in project management, issue tracking, and Atlassian workflows. Your purpose is to help users efficiently manage their work in Jira through intelligent automation and precise execution.

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
5. VERIFY the issue was created by retrieving it

**When Searching**:
1. Construct precise JQL based on the user's intent, not just keywords
2. Common patterns:
   - "my open bugs" → `assignee = currentUser() AND type = Bug AND status != Done`
   - "recent updates" → `updated >= -7d ORDER BY updated DESC`
   - "sprint issues" → `sprint = ACTIVE ORDER BY priority DESC`
3. Limit results appropriately—ask if more results are needed

**When Updating or Transitioning Issues**:
1. Confirm the issue key exists and is accessible FIRST
2. Specify exactly what fields are changing
3. For status transitions, use the proper transition name (not just the target status)
4. **CRITICAL**: ALWAYS verify after the operation by retrieving the issue(s) again
5. Compare the actual state with the expected state
6. If discrepancies exist, report them clearly and retry

**When Adding Comments**:
1. Format comments clearly and professionally
2. Use mentions (@username) when directing comments to specific people
3. Keep comments concise but informative

# Error Handling and Resilience

If you encounter errors:
- **Authentication issues**: Explain clearly that Jira authentication may need refresh
- **Permission errors**: Inform that the user may lack permissions for the requested action
- **Invalid issue keys**: Suggest using search to find the correct issue
- **Required field errors**: List exactly what fields are missing and their expected format
- **JQL syntax errors**: Explain what's wrong and provide a corrected version
- **Transient errors**: The system will automatically retry, but inform the user if multiple retries fail
- **Partial failures**: If some operations succeed and others fail, report EXACTLY which succeeded and which failed

## CRITICAL: Smart Error Recovery

**When you encounter "valid project is required" or "project not found" errors**:
1. **Immediately search for available projects** using search/list project tools
2. **Check project key case sensitivity** (KAN vs KAn vs kan - Jira is case-sensitive!)
3. **Suggest correct project key** to user if multiple matches found
4. **Retry with correct project key** once discovered
5. **DO NOT give up** after one failed attempt - this is solvable!

Example flow:
```
Error: "valid project is required"
→ Search for projects matching the pattern
→ Find "KAN" exists (not "KAn")
→ Retry with correct case
```

**When you encounter "Specify a valid issue type" errors**:
1. **This means the project exists but the issue type is wrong!**
2. **Immediately get project metadata** or search for available issue types for that project
3. **Common issue types**: Task, Story, Bug, Epic, Subtask (vary by project!)
4. **Use the first available issue type** if the requested one doesn't exist
5. **Inform the user** what issue type you used
6. **Retry immediately** with valid issue type - DO NOT ask user for input!

Example flow:
```
Error: "Specify a valid issue type" for "Bug" in project "KAN"
→ Get project "KAN" metadata to discover available issue types
→ Find available types: ["Task", "Story", "Epic"]
→ Retry with "Task" (first available)
→ Report: "Created as Task since Bug type not available in KAN project"
```

**Key Principles for Error Recovery**:
- **Errors are information, not failures** - use them to discover the correct parameters
- **Projects and issue types are discoverable** - you have tools to find them
- **Most errors are recoverable** - try at least 2-3 approaches before giving up
- **Inform users of what you learned** - "Project is 'KAN' not 'KAn'" helps them understand
- **Complete the task** - if user wanted a bug created, create it as Task if Bug doesn't exist

**YOU MUST NEVER**:
- Give up after one "valid project" or "valid issue type" error
- Ask user for information you can discover yourself (project keys, issue types)
- Fail silently without attempting recovery
- Report failure without explaining what went wrong AND what you tried

# CRITICAL: Mandatory Verification Protocol

**This is your most important responsibility**: After performing ANY update, transition, or modification operation, you MUST verify the change was successful.

## Verification Workflow (NON-NEGOTIABLE):

1. **Execute the operation** (e.g., transition issue, update field, create issue)
2. **Immediately retrieve the affected issue(s)** using search or direct get
3. **Compare actual state vs expected state**:
   - For transitions: Check the status field matches the target status
   - For updates: Check the updated fields contain the new values
   - For creations: Check the issue exists and has correct values
4. **Report results accurately**:
   - ✓ Success: "Verified: KAN-1 is now in 'Done' status"
   - ✗ Failure: "Attempted to mark KAN-1 as Done, but verification shows it's still 'In Progress'. Retrying..."
5. **Retry on failure**: If verification fails, retry the operation once, then verify again
6. **Never claim success without verification**: Users depend on accurate information

## Example Verification Workflow:

User request: "Mark KAN-1, KAN-2, KAN-3, KAN-4 as Done"

Your process:
1. Transition KAN-1 to Done
2. Transition KAN-2 to Done
3. Transition KAN-3 to Done
4. Transition KAN-4 to Done
5. **VERIFY**: Search for "key in (KAN-1, KAN-2, KAN-3, KAN-4)" and check status of each
6. **REPORT**:
   - If all verified: "Successfully marked all 4 issues as Done. Verified: KAN-1, KAN-2, KAN-3, KAN-4 are all in 'Done' status."
   - If partial: "Marked KAN-1 and KAN-2 as Done (verified). However, KAN-3 and KAN-4 are still 'In Progress'. Let me retry those..."
7. **RETRY** failed ones and verify again
8. **FINAL REPORT**: Confirm final state of all issues

This verification step is NOT optional. It is a core requirement of every state-changing operation.

# Output Format

Always structure your responses clearly:
1. **Action Summary**: What you did in plain language
2. **Verification Results**: Confirmed state of affected resources
3. **Key Details**: Issue keys, links, or important data
4. **Discrepancies**: Any differences between expected and actual state
5. **Next Steps**: Suggest logical follow-up actions when appropriate

Example:
"I've created a new bug issue PROJ-456: 'Login button not responsive on mobile'.
- Priority: High
- Assigned to: @john.doe
- Added to Sprint 23

Verified: Issue PROJ-456 is now in 'To Do' status and visible in the sprint board."

# Best Practices

- **Be efficient**: Don't make unnecessary API calls, but ALWAYS verify state changes
- **Be accurate**: Double-check issue keys and field names before operations
- **Be helpful**: Provide context and suggestions, not just raw data
- **Be clear**: Use natural language to explain what happened, avoiding Jira jargon when possible
- **Be safe**: For destructive operations, describe what will happen before execution
- **Be honest**: If you can't verify something succeeded, say so explicitly

# Understanding User Intent

Common request patterns and how to handle them:
- "Create a ticket for X" → Infer appropriate issue type (bug, task, story) from description
- "What's blocking us?" → Search for issues with 'Blocked' status or blocker links
- "Update the status" → Use proper workflow transitions, not direct status changes
- "Show me my work" → Filter for assignee=currentUser with relevant status filters
- "Mark X as done" → Transition to Done, then VERIFY it actually transitioned

Remember: You're not just executing commands—you're helping users manage their work more effectively. Think about what they're trying to accomplish, execute it reliably, verify it succeeded, and help them get there efficiently."""

    # ========================================================================
    # INITIALIZATION AND CONNECTION
    # ========================================================================

    async def initialize(self):
        """
        Connect to the Jira MCP server using sooperset/mcp-atlassian

        This method establishes the connection to Jira through Docker and
        initializes the AI model with available tools.

        Raises:
            ValueError: If required environment variables are missing
            RuntimeError: If connection or initialization fails
        """
        try:
            # Validate environment variables
            jira_url, jira_username, jira_api_token = self._get_credentials()

            if self.verbose:
                print(f"[JIRA AGENT] Initializing connection to {jira_url}")

            # Configure Docker-based MCP server
            server_params = self._create_server_params(jira_url, jira_username, jira_api_token)

            # Establish MCP connection
            await self._connect_to_mcp(server_params)

            # Load available tools from server
            await self._load_tools()

            # Initialize AI model
            self._initialize_model()

            self.initialized = True

            if self.verbose:
                print(f"[JIRA AGENT] Initialization complete. {len(self.available_tools)} tools available.")

        except Exception as e:
            error_msg = str(e)
            troubleshooting = self._get_troubleshooting_guide()

            raise RuntimeError(
                f"Failed to initialize Jira agent: {error_msg}\n" +
                "\n".join(troubleshooting)
            )

    def _get_credentials(self) -> Tuple[str, str, str]:
        """
        Retrieve and validate Jira credentials from environment

        Returns:
            Tuple of (jira_url, jira_username, jira_api_token)

        Raises:
            ValueError: If any required credentials are missing
        """
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

        # Clean up URL
        jira_url = jira_url.rstrip('/')

        return jira_url, jira_username, jira_api_token

    def _create_server_params(self, jira_url: str, jira_username: str, jira_api_token: str) -> StdioServerParameters:
        """
        Create MCP server parameters for Docker connection

        Args:
            jira_url: Full Jira instance URL
            jira_username: Atlassian account email
            jira_api_token: Jira API token

        Returns:
            Configured StdioServerParameters for Docker
        """
        return StdioServerParameters(
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

    async def _connect_to_mcp(self, server_params: StdioServerParameters):
        """
        Establish connection to MCP server

        Args:
            server_params: Server configuration parameters
        """
        self.stdio_context = stdio_client(server_params)
        stdio, write = await self.stdio_context.__aenter__()
        self.session = ClientSession(stdio, write)

        await self.session.__aenter__()
        await self.session.initialize()

    async def _load_tools(self):
        """
        Load available tools from MCP server

        Raises:
            RuntimeError: If no tools are available
        """
        tools_list = await self.session.list_tools()
        self.available_tools = tools_list.tools

        if not self.available_tools:
            raise RuntimeError("No tools available from Atlassian MCP server")

    def _initialize_model(self):
        """
        Initialize the Gemini AI model with available tools
        """
        # Convert MCP tools to Gemini format
        gemini_tools = [self._build_function_declaration(tool) for tool in self.available_tools]

        # Create model with configuration
        self.model = genai.GenerativeModel(
            'models/gemini-2.5-flash',
            system_instruction=self.system_prompt,
            tools=gemini_tools
        )

    def _get_troubleshooting_guide(self) -> List[str]:
        """
        Get troubleshooting steps for initialization failures

        Returns:
            List of troubleshooting steps
        """
        return [
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

    # ========================================================================
    # CORE EXECUTION ENGINE
    # ========================================================================

    async def execute(self, instruction: str) -> str:
        """
        Execute a Jira task with enhanced error handling and verification

        This is the main entry point for task execution. It handles:
        - LLM-based instruction interpretation
        - Tool calling loop with retry logic
        - Result validation and verification
        - Comprehensive error handling

        Args:
            instruction: Natural language instruction for the agent

        Returns:
            Natural language response with results
        """
        if not self.initialized:
            return self._format_error(Exception("Jira agent not initialized. Please restart the system."))

        try:
            # Start conversation with LLM
            chat = self.model.start_chat()
            response = await chat.send_message_async(instruction)

            # Handle function calling loop with retry logic
            max_iterations = 15  # Increased for verification steps
            iteration = 0
            actions_taken = []

            while iteration < max_iterations:
                # Check if LLM wants to call a tool
                function_call = self._extract_function_call(response)

                if not function_call:
                    # No more function calls, we're done
                    break

                tool_name = function_call.name
                tool_args = self._deep_convert_proto_args(function_call.args)

                # Track action for debugging
                actions_taken.append(tool_name)

                # Execute tool with retry logic
                result_text, error_msg = await self._execute_tool_with_retry(
                    tool_name,
                    tool_args
                )

                # Send result back to LLM
                response = await self._send_function_response(
                    chat,
                    tool_name,
                    result_text,
                    error_msg
                )

                iteration += 1

            # Check if we hit iteration limit
            if iteration >= max_iterations:
                return (
                    f"{response.text}\n\n"
                    "⚠ Note: Reached maximum operation limit. The task may be incomplete. "
                    "Consider breaking this into smaller requests."
                )

            # Return final response
            final_response = response.text

            if self.verbose:
                print(f"\n[JIRA AGENT] Execution complete. {self.stats.get_summary()}")

            return final_response

        except Exception as e:
            return self._format_error(e)

    def _extract_function_call(self, response) -> Optional[Any]:
        """
        Extract function call from LLM response

        Args:
            response: LLM response object

        Returns:
            Function call object or None if no call present
        """
        parts = response.candidates[0].content.parts
        has_function_call = any(
            hasattr(part, 'function_call') and part.function_call
            for part in parts
        )

        if not has_function_call:
            return None

        # Find and return the function call
        for part in parts:
            if hasattr(part, 'function_call') and part.function_call:
                return part.function_call

        return None

    async def _execute_tool_with_retry(
        self,
        tool_name: str,
        tool_args: Dict
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Execute a tool with automatic retry on transient failures

        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments for the tool

        Returns:
            Tuple of (result_text, error_message). One will be None.
        """
        retry_count = 0
        delay = RetryConfig.INITIAL_DELAY

        while retry_count <= RetryConfig.MAX_RETRIES:
            try:
                # Log tool call
                if self.verbose or retry_count > 0:
                    retry_info = f" (retry {retry_count}/{RetryConfig.MAX_RETRIES})" if retry_count > 0 else ""
                    print(f"\n[JIRA AGENT] Calling tool: {tool_name}{retry_info}")
                    if self.verbose:
                        print(f"[JIRA AGENT] Arguments: {json.dumps(tool_args, indent=2)[:500]}")

                # Execute tool call
                tool_result = await self.session.call_tool(tool_name, tool_args)

                # Extract result text
                result_content = []
                for content in tool_result.content:
                    if hasattr(content, 'text'):
                        result_content.append(content.text)

                result_text = "\n".join(result_content)
                if not result_text:
                    result_text = json.dumps(tool_result.content, default=str)

                # Log result
                if self.verbose:
                    print(f"[JIRA AGENT] Result: {result_text[:500]}")
                    if hasattr(tool_result, 'isError') and tool_result.isError:
                        print(f"[JIRA AGENT] ERROR FLAG SET: {tool_result.isError}")

                # Record success
                self.stats.record_operation(tool_name, True, retry_count)

                return result_text, None

            except Exception as e:
                error_str = str(e).lower()
                is_retryable = any(
                    retryable in error_str
                    for retryable in RetryConfig.RETRYABLE_ERRORS
                )

                # Log error
                if self.verbose or retry_count > 0:
                    print(f"[JIRA AGENT] Error calling {tool_name}: {str(e)}")

                # Check if we should retry
                if is_retryable and retry_count < RetryConfig.MAX_RETRIES:
                    retry_count += 1

                    if self.verbose:
                        print(f"[JIRA AGENT] Retrying in {delay:.1f}s...")

                    await asyncio.sleep(delay)
                    delay = min(delay * RetryConfig.BACKOFF_FACTOR, RetryConfig.MAX_DELAY)
                    continue
                else:
                    # No more retries or non-retryable error
                    error_msg = self._format_tool_error(tool_name, str(e), tool_args)

                    # Record failure
                    self.stats.record_operation(tool_name, False, retry_count)

                    return None, error_msg

        # Should never reach here, but just in case
        return None, f"Max retries exceeded for {tool_name}"

    async def _send_function_response(
        self,
        chat,
        tool_name: str,
        result_text: Optional[str],
        error_msg: Optional[str]
    ):
        """
        Send function call result back to LLM

        Args:
            chat: Chat session
            tool_name: Name of the tool that was called
            result_text: Successful result (if any)
            error_msg: Error message (if any)

        Returns:
            LLM response
        """
        if result_text is not None:
            # Success case
            response_data = {"result": result_text}
        else:
            # Error case
            response_data = {"error": error_msg}

        return await chat.send_message_async(
            genai.protos.Content(
                parts=[genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=tool_name,
                        response=response_data
                    )
                )]
            )
        )

    # ========================================================================
    # ERROR HANDLING AND FORMATTING
    # ========================================================================

    def _classify_error(self, error: str) -> ErrorType:
        """
        Classify an error into a specific type for better handling

        Args:
            error: Error message string

        Returns:
            ErrorType classification
        """
        error_lower = error.lower()

        if "authentication" in error_lower or "unauthorized" in error_lower:
            return ErrorType.AUTHENTICATION
        elif "permission" in error_lower or "forbidden" in error_lower:
            return ErrorType.PERMISSION
        elif "not found" in error_lower or "404" in error_lower:
            return ErrorType.NOT_FOUND
        elif "required" in error_lower or "validation" in error_lower:
            return ErrorType.VALIDATION
        elif "rate limit" in error_lower or "too many requests" in error_lower:
            return ErrorType.RATE_LIMIT
        elif any(net in error_lower for net in ["timeout", "connection", "network"]):
            return ErrorType.NETWORK
        else:
            return ErrorType.UNKNOWN

    def _format_tool_error(self, tool_name: str, error: str, args: Dict) -> str:
        """
        Format tool errors with helpful, context-aware messages

        Args:
            tool_name: Name of the tool that failed
            error: Error message
            args: Arguments that were passed to the tool

        Returns:
            Formatted, user-friendly error message
        """
        error_type = self._classify_error(error)

        # Generate context-aware error message based on type
        if error_type == ErrorType.AUTHENTICATION:
            return (
                f"🔐 Authentication error when calling {tool_name}. "
                "Your Jira API token may be invalid or expired. "
                "Please verify your JIRA_API_TOKEN environment variable."
            )

        elif error_type == ErrorType.PERMISSION:
            return (
                f"🚫 Permission denied for {tool_name}. "
                "You may not have the required permissions for this operation. "
                "Check your Jira role and project permissions."
            )

        elif error_type == ErrorType.NOT_FOUND:
            issue_key = args.get('issueKey') or args.get('issueIdOrKey') or args.get('issueId')
            if issue_key:
                return (
                    f"🔍 Issue '{issue_key}' not found. "
                    "Please verify the issue key is correct and you have access to it. "
                    "Try searching for the issue first."
                )
            return (
                f"🔍 Resource not found when calling {tool_name}. "
                "Please check your input parameters."
            )

        elif error_type == ErrorType.VALIDATION:
            return (
                f"⚠️ Validation error for {tool_name}. "
                f"Missing or invalid required fields. "
                f"Error details: {error}"
            )

        elif error_type == ErrorType.RATE_LIMIT:
            return (
                f"⏳ Rate limit exceeded for {tool_name}. "
                "Too many requests to Jira API. "
                "The system will automatically retry after a delay."
            )

        elif error_type == ErrorType.NETWORK:
            return (
                f"🌐 Network error when calling {tool_name}: {error}. "
                "The system will automatically retry. "
                "If this persists, check your internet connection and Jira availability."
            )

        else:
            return f"❌ Error calling {tool_name}: {error}"

    # ========================================================================
    # CAPABILITIES AND INFORMATION
    # ========================================================================

    async def get_capabilities(self) -> List[str]:
        """
        Return Jira capabilities in user-friendly format

        Returns:
            List of capability descriptions
        """
        if not self.available_tools:
            return ["Jira operations (initializing...)"]

        # Group capabilities by category for better readability
        capabilities = []
        for tool in self.available_tools:
            description = tool.description or tool.name
            if description:
                capabilities.append(description)

        # If we have many tools, provide a summary instead of listing all
        if len(capabilities) > 10:
            return [
                "✓ Create and manage Jira issues",
                "✓ Search issues using JQL",
                "✓ Add comments and collaborate",
                "✓ Manage workflows and transitions",
                f"✓ ...and {len(capabilities) - 4} more Jira operations"
            ]

        return capabilities

    def get_stats(self) -> str:
        """
        Get operation statistics summary

        Returns:
            Human-readable statistics summary
        """
        return self.stats.get_summary()

    # ========================================================================
    # CLEANUP AND RESOURCE MANAGEMENT
    # ========================================================================

    async def cleanup(self):
        """
        Disconnect from Jira and clean up resources

        This method ensures all connections are properly closed and
        resources are released.
        """
        try:
            if self.verbose:
                print(f"\n[JIRA AGENT] Cleaning up. {self.stats.get_summary()}")

            if self.session:
                await self.session.__aexit__(None, None, None)
        except Exception as e:
            if self.verbose:
                print(f"[JIRA AGENT] Error closing session: {e}")

        try:
            if self.stdio_context:
                await self.stdio_context.__aexit__(None, None, None)
        except Exception as e:
            if self.verbose:
                print(f"[JIRA AGENT] Error closing stdio context: {e}")

    # ========================================================================
    # SCHEMA CONVERSION HELPERS
    # ========================================================================

    def _build_function_declaration(self, tool: Any) -> protos.FunctionDeclaration:
        """
        Convert MCP tool schema to Gemini function declaration

        Args:
            tool: MCP tool object

        Returns:
            Gemini-compatible function declaration
        """
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
        """
        Convert JSON schema to protobuf schema recursively

        Args:
            schema: JSON schema dictionary

        Returns:
            Protobuf schema object
        """
        schema_pb = protos.Schema()

        # Map type
        if "type" in schema:
            schema_pb.type_ = self.schema_type_map.get(
                schema["type"],
                protos.Type.TYPE_UNSPECIFIED
            )

        # Copy description
        if "description" in schema:
            schema_pb.description = schema["description"]

        # Handle enum values
        if "enum" in schema:
            schema_pb.enum.extend(schema["enum"])

        # Handle array items
        if "items" in schema and isinstance(schema["items"], dict):
            schema_pb.items = self._clean_schema(schema["items"])

        # Handle object properties (recursive)
        if "properties" in schema and isinstance(schema["properties"], dict):
            for prop_name, prop_schema in schema["properties"].items():
                schema_pb.properties[prop_name] = self._clean_schema(prop_schema)

        # Handle required fields
        if "required" in schema:
            schema_pb.required.extend(schema["required"])

        return schema_pb

    def _deep_convert_proto_args(self, value: Any) -> Any:
        """
        Recursively convert protobuf types to standard Python types

        This is needed because Gemini returns protobuf composite types
        that need to be converted to dicts/lists for JSON serialization.

        Args:
            value: Protobuf value (possibly nested)

        Returns:
            Standard Python dict, list, or primitive
        """
        type_str = str(type(value))

        if "MapComposite" in type_str:
            # Convert protobuf map to dict
            return {k: self._deep_convert_proto_args(v) for k, v in value.items()}
        elif "RepeatedComposite" in type_str:
            # Convert protobuf repeated field to list
            return [self._deep_convert_proto_args(item) for item in value]
        else:
            # Return primitive as-is
            return value
