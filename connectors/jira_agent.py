# Jira agent connector using MCP with Docker integration

import os
import sys
import json
import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import google.generativeai as genai
import google.generativeai.protos as protos
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from connectors.mcp_stdio_wrapper import quiet_stdio_client

# Add parent directory to path to import base_agent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from connectors.base_agent import BaseAgent, safe_extract_response_text
from connectors.agent_intelligence import (
    ConversationMemory,
    WorkspaceKnowledge,
    SharedContext,
    ProactiveAssistant
)
from core.agent_error_handling import CredentialValidator, AgentErrorHandler


class RetryConfig:
    # Retry configuration with exponential backoff
    MAX_RETRIES = 3
    INITIAL_DELAY = 1.0
    MAX_DELAY = 10.0
    BACKOFF_FACTOR = 2.0
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
    # Error type classification
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    NOT_FOUND = "not_found"
    VALIDATION = "validation"
    RATE_LIMIT = "rate_limit"
    NETWORK = "network"
    UNKNOWN = "unknown"


@dataclass
class OperationStats:
    # Track statistics for agent operations
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    retries: int = 0
    suggestions_offered: int = 0
    tools_called: Dict[str, int] = field(default_factory=dict)

    def record_operation(self, tool_name: str, success: bool, retry_count: int = 0):
        # Record operation statistics
        self.total_operations += 1
        self.tools_called[tool_name] = self.tools_called.get(tool_name, 0) + 1

        if success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1

        self.retries += retry_count

    def get_summary(self) -> str:
        # Get human-readable summary
        success_rate = (self.successful_operations / self.total_operations * 100) if self.total_operations > 0 else 0
        return (
            f"Operations: {self.total_operations} total, "
            f"{self.successful_operations} successful, "
            f"{self.failed_operations} failed "
            f"({success_rate:.1f}% success rate), "
            f"{self.retries} retries"
        )


class Agent(BaseAgent):
    # Specialized agent for Jira operations via MCP

    def __init__(
        self,
        verbose: bool = False,
        shared_context: Optional[SharedContext] = None,
        knowledge_base: Optional[WorkspaceKnowledge] = None
    ,
        session_logger=None
    ):
        # Initialize the Jira agent
        super().__init__()

        self.logger = session_logger
        self.agent_name = "jira"

        self.session: ClientSession = None
        self.session_entered = False
        self.stdio_context = None
        self.stdio_context_entered = False
        self.model = None
        self.available_tools = []

        self.verbose = verbose
        self.stats = OperationStats()

        self.memory = ConversationMemory()
        self.knowledge = knowledge_base or WorkspaceKnowledge()
        self.shared_context = shared_context
        self.proactive = ProactiveAssistant('jira', verbose)

        self.metadata_cache = {}

        self.schema_type_map = {
            "string": protos.Type.STRING,
            "number": protos.Type.NUMBER,
            "integer": protos.Type.INTEGER,
            "boolean": protos.Type.BOOLEAN,
            "object": protos.Type.OBJECT,
            "array": protos.Type.ARRAY,
        }

        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        # Build system prompt defining agent behavior
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

**CRITICAL: Response Formatting After Successful Recovery**:
When you encounter an error but then successfully recover and complete the task:
- **YOUR FINAL RESPONSE MUST FOCUS ON THE SUCCESS, NOT THE ERROR**
- Lead with what you accomplished: "✓ Created issue KAN-18: 'buffer overflow issue' with High priority"
- Optionally mention the recovery: "(Note: Created as Task since Bug type is not available in this project)"
- **NEVER say the operation failed if your retry succeeded** - always check your most recent tool call results

Example CORRECT final response after error recovery:
```
✓ Created issue KAN-18: "buffer overflow issue" with High priority

Summary: buffer overflow issue
Type: Task (Bug type not available in KAN project)
Priority: High
Status: To Do

Link: https://your-domain.atlassian.net/browse/KAN-18
```

Example WRONG final response (DO NOT DO THIS):
```
I wasn't able to create the issue. There was an error with the issue type.
```
^^ This is WRONG even if you successfully created it after retrying!

**YOU MUST NEVER**:
- Give up after one "valid project" or "valid issue type" error
- Ask user for information you can discover yourself (project keys, issue types)
- Fail silently without attempting recovery
- Report failure without explaining what went wrong AND what you tried
- **Report failure when your retry actually succeeded** - always verify your final tool call result!

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

    async def initialize(self):
        # Connect to Jira MCP server and initialize model
        try:
            jira_url, jira_username, jira_api_token = self._get_credentials()

            if self.verbose:
                print(f"[JIRA AGENT] Initializing connection to {jira_url}")

            server_params = self._create_server_params(jira_url, jira_username, jira_api_token)
            await self._connect_to_mcp(server_params)
            await self._load_tools()
            self._initialize_model()
            self.initialized = True

            # Prefetch metadata in background (non-blocking, with timeout)
            try:
                await asyncio.wait_for(self._prefetch_metadata(), timeout=10.0)
            except asyncio.TimeoutError:
                if self.verbose:
                    print(f"[JIRA AGENT] Metadata prefetch timed out (continuing without cache)")
                self.metadata_cache = {'projects': {}}
            except Exception as e:
                if self.verbose:
                    print(f"[JIRA AGENT] Metadata prefetch failed: {str(e)[:100]}")
                self.metadata_cache = {'projects': {}}

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
        # Get Jira credentials from environment with validation
        required_vars = {
            'JIRA_URL': 'Jira URL',
            'JIRA_USERNAME': 'Jira Username',
            'JIRA_API_TOKEN': 'Jira API Token'
        }

        credential_check = CredentialValidator.check_credentials('Jira', required_vars)

        if not credential_check.valid:
            raise ValueError(
                credential_check.error_message + "\n" +
                credential_check.setup_instructions
            )

        jira_url = os.environ.get("JIRA_URL").rstrip('/')
        jira_username = os.environ.get("JIRA_USERNAME")
        jira_api_token = os.environ.get("JIRA_API_TOKEN")

        return jira_url, jira_username, jira_api_token

    def _create_server_params(self, jira_url: str, jira_username: str, jira_api_token: str) -> StdioServerParameters:
        # Create MCP server parameters for Docker
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
        # Establish connection to MCP server
        try:
            self.stdio_context = quiet_stdio_client(server_params)
            stdio, write = await self.stdio_context.__aenter__()
            self.stdio_context_entered = True

            self.session = ClientSession(stdio, write)
            await self.session.__aenter__()
            self.session_entered = True

            await self.session.initialize()
        except Exception as e:
            await self._cleanup_connection()
            raise

    async def _load_tools(self):
        # Load available tools from MCP server
        tools_list = await self.session.list_tools()
        self.available_tools = tools_list.tools

        if not self.available_tools:
            raise RuntimeError("No tools available from Atlassian MCP server")

    def _initialize_model(self):
        # Initialize Gemini AI model with tools
        gemini_tools = [self._build_function_declaration(tool) for tool in self.available_tools]
        self.model = genai.GenerativeModel(
            'models/gemini-2.5-flash',
            system_instruction=self.system_prompt,
            tools=gemini_tools
        )

    def _get_troubleshooting_guide(self) -> List[str]:
        # Get troubleshooting steps for initialization
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

    async def _prefetch_metadata(self):
        """Prefetch metadata - with timeout protection to avoid blocking initialization"""
        try:
            # Check cache first
            cached = self.knowledge.get_metadata_cache('jira')
            if cached:
                self.metadata_cache = cached
                if self.verbose:
                    print(f"[JIRA AGENT] Loaded metadata from cache ({len(cached.get('projects', {}))} projects)")
                return

            if self.verbose:
                print(f"[JIRA AGENT] Prefetching metadata...")

            # Fetch projects only (skip detailed metadata to speed up init)
            projects = await self._fetch_all_projects()

            self.metadata_cache = {
                'projects': projects,
                'fetched_at': asyncio.get_event_loop().time()
            }

            self.knowledge.save_metadata_cache('jira', self.metadata_cache, ttl_seconds=3600)

            if self.verbose:
                print(f"[JIRA AGENT] Cached metadata for {len(projects)} projects")

        except Exception as e:
            # Silently fail - metadata is optional for agent operation
            if self.verbose:
                print(f"[JIRA AGENT] Metadata prefetch skipped: {str(e)[:100]}")
            # Set empty cache to avoid repeated attempts
            self.metadata_cache = {'projects': {}}

    async def _fetch_all_projects(self) -> Dict:
        # Fetch all accessible projects
        try:
            result = await self.session.call_tool("jira_get_projects", {})

            projects = {}
            if hasattr(result, 'content') and result.content:
                content = result.content[0].text if result.content else "{}"
                data = json.loads(content) if isinstance(content, str) else content

                if isinstance(data, list):
                    for proj in data:
                        key = proj.get('key', '')
                        projects[key] = {
                            'id': proj.get('id', ''),
                            'name': proj.get('name', ''),
                            'key': key
                        }

            return projects
        except:
            return {}

    async def _fetch_project_issue_types(self, project_key: str) -> List[str]:
        # Fetch available issue types for project
        try:
            result = await self.session.call_tool("jira_get_issue_types", {"project": project_key})

            if hasattr(result, 'content') and result.content:
                content = result.content[0].text if result.content else "[]"
                data = json.loads(content) if isinstance(content, str) else content

                if isinstance(data, list):
                    return [t.get('name', '') for t in data if t.get('name')]

            return []
        except:
            return []

    async def _fetch_project_fields(self, project_key: str) -> Dict:
        # Fetch custom fields for project
        return {}

    async def execute(self, instruction: str) -> str:
        # Execute Jira task with error handling
        if not self.initialized:
            return self._format_error(Exception("Jira agent not initialized. Please restart the system."))

        try:
            resolved_instruction = self._resolve_references(instruction)

            if resolved_instruction != instruction and self.verbose:
                print(f"[JIRA AGENT] Resolved instruction: {resolved_instruction}")

            context_from_other_agents = self._get_cross_agent_context()
            if context_from_other_agents and self.verbose:
                print(f"[JIRA AGENT] Found context from other agents: {len(context_from_other_agents)} resources")

            chat = self.model.start_chat()

            if context_from_other_agents:
                resolved_instruction += f"\n\n[Additional context from other agents: {context_from_other_agents}]"

            response = await chat.send_message_async(resolved_instruction)

            max_iterations = 15
            iteration = 0
            actions_taken = []

            while iteration < max_iterations:
                function_call = self._extract_function_call(response)

                if not function_call:
                    break

                tool_name = function_call.name
                tool_args = self._deep_convert_proto_args(function_call.args)

                actions_taken.append(tool_name)

                result_text, error_msg = await self._execute_tool_with_retry(
                    tool_name,
                    tool_args
                )

                response = await self._send_function_response(
                    chat,
                    tool_name,
                    result_text,
                    error_msg
                )

                iteration += 1

            if iteration >= max_iterations:
                return (
                    f"{safe_extract_response_text(response)}\n\n"
                    "⚠ Note: Reached maximum operation limit. The task may be incomplete. "
                    "Consider breaking this into smaller requests."
                )

            final_response = safe_extract_response_text(response)
            self._remember_created_resources(final_response, resolved_instruction)

            operation_type = self._infer_operation_type(resolved_instruction)
            context = {'instruction': resolved_instruction, 'response': final_response}
            suggestions = self.proactive.suggest_next_steps(operation_type, context)

            if suggestions:
                self.stats.suggestions_offered += 1
                final_response += "\n\n**💡 Suggested next steps:**\n" + "\n".join(f"  • {s}" for s in suggestions)

            if self.verbose:
                print(f"\n[JIRA AGENT] Execution complete. {self.stats.get_summary()}")

            return final_response

        except Exception as e:
            return self._format_error(e)

    def _resolve_references(self, instruction: str) -> str:
        # Resolve ambiguous references using memory
        ambiguous_terms = ['it', 'that', 'this', 'the issue', 'the ticket']

        for term in ambiguous_terms:
            if term in instruction.lower():
                reference = self.memory.resolve_reference(term)
                if reference:
                    instruction = instruction.replace(term, reference)
                    instruction = instruction.replace(term.capitalize(), reference)
                    if self.verbose:
                        print(f"[JIRA AGENT] Resolved '{term}' → {reference}")
                    break

        return instruction

    def _get_cross_agent_context(self) -> str:
        # Get context from other agents
        if not self.shared_context:
            return ""

        recent_resources = self.shared_context.get_recent_resources(limit=5)

        if not recent_resources:
            return ""

        context_parts = []
        for resource in recent_resources:
            if resource['agent'] != 'jira':
                context_parts.append(
                    f"{resource['agent'].capitalize()} {resource['type']}: {resource['id']} ({resource['url']})"
                )

        return "; ".join(context_parts) if context_parts else ""

    def _remember_created_resources(self, response: str, instruction: str):
        # Extract and remember created resources from response
        import re

        issue_pattern = r'\b([A-Z][A-Z0-9]+-\d+)\b'
        matches = re.findall(issue_pattern, response)

        if matches:
            issue_key = matches[-1]

            operation_type = 'create_issue' if 'creat' in instruction.lower() else 'update_issue'

            self.memory.remember(
                operation_type,
                issue_key,
                {'instruction': instruction[:100]}
            )

            if self.shared_context:
                jira_url = os.environ.get('JIRA_URL', 'https://your-domain.atlassian.net')
                issue_url = f"{jira_url}/browse/{issue_key}"

                self.shared_context.share_resource(
                    'jira',
                    'ticket',
                    issue_key,
                    issue_url,
                    {'created_by': 'jira_agent'}
                )

                if self.verbose:
                    print(f"[JIRA AGENT] Shared {issue_key} with other agents")

    def _infer_operation_type(self, instruction: str) -> str:
        # Infer operation type from instruction
        instruction_lower = instruction.lower()

        if 'create' in instruction_lower or 'new' in instruction_lower:
            return 'create_issue'
        elif 'update' in instruction_lower or 'edit' in instruction_lower or 'change' in instruction_lower:
            return 'update_issue'
        elif 'transition' in instruction_lower or 'mark' in instruction_lower or 'move' in instruction_lower:
            return 'transition_issue'
        elif 'comment' in instruction_lower:
            return 'add_comment'
        elif 'search' in instruction_lower or 'find' in instruction_lower or 'list' in instruction_lower:
            return 'search'
        else:
            return 'unknown'

    def _extract_function_call(self, response) -> Optional[Any]:
        # Extract function call from LLM response
        parts = response.candidates[0].content.parts
        has_function_call = any(
            hasattr(part, 'function_call') and part.function_call
            for part in parts
        )

        if not has_function_call:
            return None

        for part in parts:
            if hasattr(part, 'function_call') and part.function_call:
                return part.function_call

        return None

    async def _execute_tool_with_retry(
        self,
        tool_name: str,
        tool_args: Dict
    ) -> Tuple[Optional[str], Optional[str]]:
        # Execute tool with automatic retry on failures
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
                # Log tool call start

                start_time = time.time()


                tool_result = await self.session.call_tool(tool_name, tool_args)


                # Log tool call completion

                duration = time.time() - start_time

                if self.logger:

                    self.logger.log_tool_call(self.agent_name, tool_name, duration, success=True)


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

                    # Log tool call failure
                    if self.logger:
                        self.logger.log_tool_call(self.agent_name, tool_name, None, success=False, error=str(e))

                    return None, error_msg

        # Log max retries exceeded
        if self.logger:
            self.logger.log_tool_call(self.agent_name, tool_name, None, success=False, error="Max retries exceeded")

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
                    session_logger: Optional session logger for tracking operations
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
                    session_logger: Optional session logger for tracking operations
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
                    session_logger: Optional session logger for tracking operations
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
        Return Jira capabilities in user-friendly format with limitations

        Returns:
            List of capability descriptions with clear limitations
        """
        if not self.available_tools:
            return ["Jira operations (initializing...)"]

        # Return curated list with clear capabilities and limitations
        return [
            "✓ Create and manage Jira issues",
            "✓ Search issues using JQL (Jira Query Language)",
            "✓ Update issue fields (status, assignee, labels, custom fields)",
            "✓ Add comments and collaborate on issues",
            "✓ Manage workflows and transitions",
            "✓ Create sub-tasks and link issues",
            "✗ Cannot: Delete issues (archived only)",
            "✗ Cannot: Manage project settings or permissions",
            "✗ Cannot: Create or delete projects (admin only)",
            "✗ Cannot: Modify issue history or audit logs",
        ]

    async def get_action_schema(self) -> Dict[str, Any]:
        """
        Return schema describing editable parameters for Jira actions.

        This enables rich interactive editing of Jira issues before creation/update.

        Returns:
            Dict mapping action types to their parameter schemas
        """
        # Get available projects and issue types from cache
        project_keys = []
        issue_types = ['Bug', 'Task', 'Story', 'Epic', 'Subtask']
        priorities = ['Highest', 'High', 'Medium', 'Low', 'Lowest']

        if self.metadata_cache.get('projects'):
            project_keys = list(self.metadata_cache['projects'].keys())

        if self.metadata_cache.get('issue_types'):
            issue_types = self.metadata_cache['issue_types']

        return {
            'create': {
                'parameters': {
                    'summary': {
                        'display_label': 'Summary',
                        'description': 'Brief title/summary of the issue',
                        'type': 'string',
                        'editable': True,
                        'required': True,
                        'constraints': {
                            'min_length': 5,
                            'max_length': 255,
                        },
                        'examples': [
                            'Fix login page timeout issue',
                            'Add user profile export feature',
                            'Update API documentation'
                        ]
                    },
                    'description': {
                        'display_label': 'Description',
                        'description': 'Detailed description of the issue',
                        'type': 'text',  # Multi-line
                        'editable': True,
                        'required': False,
                        'constraints': {
                            'max_length': 10000,
                        },
                        'examples': [
                            'Users are experiencing timeouts when logging in during peak hours...',
                            'We need to allow users to export their profile data in CSV format...'
                        ]
                    },
                    'project': {
                        'display_label': 'Project',
                        'description': 'Jira project key (e.g., KAN, PROJ)',
                        'type': 'string',
                        'editable': True,
                        'required': True,
                        'constraints': {
                            'allowed_values': project_keys if project_keys else None,
                        },
                        'examples': ['KAN', 'PROJ', 'DEV']
                    },
                    'issue_type': {
                        'display_label': 'Issue Type',
                        'description': 'Type of issue (Bug, Task, Story, etc.)',
                        'type': 'string',
                        'editable': True,
                        'required': True,
                        'constraints': {
                            'allowed_values': issue_types,
                        }
                    },
                    'priority': {
                        'display_label': 'Priority',
                        'description': 'Priority level for the issue',
                        'type': 'string',
                        'editable': True,
                        'required': False,
                        'constraints': {
                            'allowed_values': priorities,
                        }
                    },
                    'assignee': {
                        'display_label': 'Assignee',
                        'description': 'User to assign the issue to',
                        'type': 'string',
                        'editable': True,
                        'required': False,
                        'examples': ['john@company.com', 'unassigned']
                    },
                    'labels': {
                        'display_label': 'Labels',
                        'description': 'Labels/tags for the issue (comma-separated)',
                        'type': 'string',
                        'editable': True,
                        'required': False,
                        'examples': ['backend,urgent', 'frontend,ui']
                    }
                }
            },
            'update': {
                'parameters': {
                    'issue_key': {
                        'display_label': 'Issue Key',
                        'description': 'The issue to update (e.g., KAN-123)',
                        'type': 'string',
                        'editable': False,  # Can't change which issue we're updating
                        'required': True
                    },
                    'summary': {
                        'display_label': 'Summary',
                        'description': 'New summary/title for the issue',
                        'type': 'string',
                        'editable': True,
                        'required': False,
                        'constraints': {
                            'min_length': 5,
                            'max_length': 255,
                        }
                    },
                    'description': {
                        'display_label': 'Description',
                        'description': 'New description for the issue',
                        'type': 'text',
                        'editable': True,
                        'required': False,
                        'constraints': {
                            'max_length': 10000,
                        }
                    },
                    'priority': {
                        'display_label': 'Priority',
                        'description': 'New priority level',
                        'type': 'string',
                        'editable': True,
                        'required': False,
                        'constraints': {
                            'allowed_values': priorities,
                        }
                    },
                    'assignee': {
                        'display_label': 'Assignee',
                        'description': 'New assignee for the issue',
                        'type': 'string',
                        'editable': True,
                        'required': False
                    }
                }
            }
        }

    async def validate_operation(self, instruction: str) -> Dict[str, Any]:
        """
        Validate if a Jira operation can be performed (Feature #14)

        Uses cached metadata to quickly check if the operation is likely to succeed.

        Args:
            instruction: The instruction to validate

        Returns:
            Dict with validation results
                    session_logger: Optional session logger for tracking operations
        """
        result = {
            'valid': True,
            'missing': [],
            'warnings': [],
            'confidence': 1.0
        }

        instruction_lower = instruction.lower()

        # Check if we're creating an issue
        if any(word in instruction_lower for word in ['create', 'new']) and 'issue' in instruction_lower:
            # Check if we have project metadata
            if not self.metadata_cache.get('projects'):
                result['warnings'].append("No project metadata cached - operation may be slow")
                result['confidence'] = 0.7

            # Try to extract project key from instruction
            projects = self.metadata_cache.get('projects', {})
            if projects:
                # Check if instruction mentions a known project
                mentioned_project = None
                for proj_key in projects.keys():
                    if proj_key.lower() in instruction_lower:
                        mentioned_project = proj_key
                        break

                if not mentioned_project:
                    result['missing'].append("project name or key")
                    result['valid'] = False
                    result['confidence'] = 0.3

        # Check if agent is initialized
        if not self.initialized:
            result['valid'] = False
            result['missing'].append("agent initialization")
            result['confidence'] = 0.0

        return result

    def get_stats(self) -> str:
        """
        Get operation statistics summary

        Returns:
            Human-readable statistics summary
        """
        return self.stats.get_summary()

    async def apply_parameter_edits(
        self,
        instruction: str,
        parameter_edits: Dict[str, Any]
    ) -> str:
        """
        Apply user's edited parameters back into the instruction for Jira actions.

        Args:
            instruction: Original instruction from LLM
            parameter_edits: {field_name: new_value} from user

        Returns:
            Modified instruction with edits applied

        Example:
            Original: "Create issue in KAN with summary 'Bug fix' description 'System crash'"
            Edits: {'summary': 'Critical: System crash', 'priority': 'High'}
            Return: "Create issue in KAN with summary 'Critical: System crash' description 'System crash' priority 'High'"
        """
        import re

        modified = instruction

        # Apply summary/title edits
        if 'summary' in parameter_edits or 'title' in parameter_edits:
            new_summary = parameter_edits.get('summary') or parameter_edits.get('title')
            patterns = [
                (r"summary\s+['\"]([^'\"]+)['\"]", f"summary '{new_summary}'"),
                (r"title\s+['\"]([^'\"]+)['\"]", f"title '{new_summary}'"),
            ]

            for pattern, replacement in patterns:
                if re.search(pattern, modified, re.IGNORECASE):
                    modified = re.sub(pattern, replacement, modified, count=1, flags=re.IGNORECASE)
                    break

        # Apply description edits
        if 'description' in parameter_edits:
            new_desc = parameter_edits['description']
            pattern = r"description\s+['\"]([^'\"]+)['\"]"
            if re.search(pattern, modified, re.IGNORECASE):
                modified = re.sub(pattern, f"description '{new_desc}'", modified, count=1, flags=re.IGNORECASE)
            else:
                # Add description if not present
                modified += f" description '{new_desc}'"

        # Apply project edits
        if 'project' in parameter_edits:
            new_project = parameter_edits['project']
            patterns = [
                (r"in\s+([A-Z]{2,10})", f"in {new_project}"),
                (r"project\s+([A-Z]{2,10})", f"project {new_project}"),
            ]

            for pattern, replacement in patterns:
                if re.search(pattern, modified):
                    modified = re.sub(pattern, replacement, modified, count=1)
                    break

        # Apply priority edits
        if 'priority' in parameter_edits:
            new_priority = parameter_edits['priority']
            pattern = r"priority\s+\w+"
            if re.search(pattern, modified, re.IGNORECASE):
                modified = re.sub(pattern, f"priority {new_priority}", modified, count=1, flags=re.IGNORECASE)
            else:
                # Add priority if not present
                modified += f" priority {new_priority}"

        # Apply assignee edits
        if 'assignee' in parameter_edits:
            new_assignee = parameter_edits['assignee']
            pattern = r"assign(ee)?\s+(to\s+)?['\"]?([^'\"]+)['\"]?"
            if re.search(pattern, modified, re.IGNORECASE):
                modified = re.sub(pattern, f"assign to {new_assignee}", modified, count=1, flags=re.IGNORECASE)
            else:
                # Add assignee if not present
                modified += f" assign to {new_assignee}"

        # Apply issue_type edits
        if 'issue_type' in parameter_edits:
            new_type = parameter_edits['issue_type']
            pattern = r"(type|issue_type)\s+\w+"
            if re.search(pattern, modified, re.IGNORECASE):
                modified = re.sub(pattern, f"type {new_type}", modified, count=1, flags=re.IGNORECASE)
            else:
                # Add type if not present
                modified += f" type {new_type}"

        # Apply labels edits
        if 'labels' in parameter_edits:
            new_labels = parameter_edits['labels']
            pattern = r"labels?\s+['\"]?([^'\"]+)['\"]?"
            if re.search(pattern, modified, re.IGNORECASE):
                modified = re.sub(pattern, f"labels {new_labels}", modified, count=1, flags=re.IGNORECASE)
            else:
                # Add labels if not present
                modified += f" labels {new_labels}"

        return modified

    # ========================================================================
    # CLEANUP AND RESOURCE MANAGEMENT
    # ========================================================================

    async def cleanup(self):
        """
        Disconnect from Jira and clean up resources

        This method ensures all connections are properly closed and
        resources are released.
        """
        if self.verbose:
            print(f"\n[JIRA AGENT] Cleaning up. {self.stats.get_summary()}")

        await self._cleanup_connection()

    async def _cleanup_connection(self):
        """Internal cleanup helper for MCP connection resources"""
        # Close session if it was successfully entered
        if self.session and self.session_entered:
            try:
                await self.session.__aexit__(None, None, None)
            except Exception as e:
                # Suppress all cleanup errors to prevent cascading failures
                if self.verbose:
                    print(f"[JIRA AGENT] Suppressed session cleanup error: {e}")
            finally:
                self.session = None
                self.session_entered = False

        # Close stdio context if it was successfully entered
        # IMPORTANT: The MCP stdio_client uses anyio which requires context managers
        # to be entered/exited in the same task. If cancelled, this may fail.
        if self.stdio_context and self.stdio_context_entered:
            try:
                await self.stdio_context.__aexit__(None, None, None)
            except RuntimeError as e:
                # Specifically suppress "cancel scope in different task" errors
                # This happens when the agent is cancelled/timed out
                if "cancel scope" in str(e).lower() or "different task" in str(e).lower():
                    # This is expected during cancellation, silently ignore
                    pass
                elif self.verbose:
                    print(f"[JIRA AGENT] Suppressed stdio cleanup error: {e}")
            except Exception as e:
                # Suppress all other cleanup errors
                if self.verbose:
                    print(f"[JIRA AGENT] Suppressed stdio cleanup error: {e}")
            finally:
                self.stdio_context = None
                self.stdio_context_entered = False

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
                    session_logger: Optional session logger for tracking operations
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
                    session_logger: Optional session logger for tracking operations
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
                    session_logger: Optional session logger for tracking operations
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
