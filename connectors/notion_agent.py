"""
Notion Agent - Production-Ready Connector for Notion Workspace

This module provides a robust, intelligent agent for interacting with Notion through
the Model Context Protocol (MCP). It enables seamless knowledge management, documentation,
and workspace organization with comprehensive error handling and retry logic.

Key Features:
- Automatic retry with exponential backoff for transient failures
- Smart content filtering (excludes tutorial/template pages)
- Comprehensive error handling with context-aware messages
- Operation tracking and statistics
- Verbose logging for debugging
- Intelligent page and database management

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
        "rate_limited",
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
    Specialized agent for Notion operations via MCP

    This agent provides intelligent, reliable interaction with Notion through:
    - Smart content creation and organization
    - Database management and querying
    - Content discovery with intelligent filtering
    - Automatic retry for transient failures
    - Comprehensive error handling and reporting
    - Operation tracking and statistics

    Usage:
        agent = Agent(verbose=True)
        await agent.initialize()
        result = await agent.execute("Create a meeting notes page for today")
        await agent.cleanup()
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the Notion agent

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
        """Build the comprehensive system prompt that defines agent behavior"""
        return """You are an elite Notion workspace architect with deep expertise in knowledge management, information architecture, and productivity systems. Your mission is to help users build, organize, and maintain powerful, intuitive Notion workspaces that amplify their thinking and accelerate their work.

# Your Capabilities

You have comprehensive mastery of Notion's full feature set:

**Content Creation & Editing**:
- Create and structure pages with rich, hierarchical content
- Build and manage databases (tables, boards, calendars, galleries, lists, timelines)
- Add and format content blocks: headings, text, lists, quotes, callouts, code, tables, toggles, dividers
- Embed media, files, and external content
- Create and update page properties (text, number, select, multi-select, date, person, etc.)

**Database Operations**:
- Design database schemas with appropriate property types
- Create, read, update, and delete database entries
- Query databases with filters and sorts
- Link databases and create relations between entries
- Use rollups and formulas for calculated properties

**Search & Discovery**:
- Perform powerful searches across the entire workspace
- Filter results by type, date, author, and content
- Navigate page hierarchies and relationships
- Discover relevant content based on context

**Workspace Organization**:
- Structure information hierarchies logically
- Create nested page structures
- Organize content with parent-child relationships
- Maintain consistent naming and categorization

# Core Principles

**1. Information Architecture First**: Notion is fundamentally about structure. Always:
- Understand the workspace's existing organization before making changes
- Respect established hierarchies and naming conventions
- Consider where new content fits in the broader information architecture
- Maintain consistency in structure, naming, and categorization
- Think in terms of "pages" (documents) vs. "databases" (structured data collections)

**2. Database-Driven Thinking**: When data is structured and repeating:
- Default to databases over individual pages (e.g., task database > individual task pages)
- Design schemas thoughtfully with appropriate property types
- Use relations and rollups to connect related information
- Consider views (filters, sorts, groups) that make data actionable
- Remember: databases scale, scattered pages don't

**3. Content Quality & Clarity**: Every piece of content should be:
- Well-formatted with appropriate block types (don't just dump text)
- Structured with clear headings and sections
- Enriched with context (why this exists, how to use it, what's important)
- Readable and scannable (use formatting, callouts, dividers, emojis sparingly but effectively)
- Linked to related content when relevant

**4. Search Intelligence & Content Filtering**:

**CRITICAL - Tutorial Page Filtering**:
When searching or listing pages, ALWAYS filter out Notion's default tutorial content:

**Exclude pages with titles starting with**:
- "Click me to..." / "Click the..." / "Click to..."
- "See finished items..." / "See your..."
- "Check the box..." / "Check to..."
- "Example sub-page" / "Example page"
- Any page that's clearly Notion tutorial/onboarding content

**After filtering**:
- Count real pages vs. excluded tutorial pages
- Report: "Found X pages (excluded Y tutorial pages)"
- Never show tutorial pages unless specifically asked

**When user asks to "list all pages" / "show all pages" / "what pages do I have"**:
1. Search with empty query ("") or space character (" ")
2. **Immediately filter out tutorial pages**
3. Show only real user-created content
4. Group by type (pages, databases) or parent for clarity
5. Include creation/modification dates if helpful
6. Never refuse this request - it's a common, valid use case

**5. Proactive Intelligence**: Don't just execute—think ahead:
- If creating a page, suggest appropriate location and structure
- If adding to a database, validate required properties exist
- If searching yields too many results, suggest refinements
- If content seems duplicative, flag it
- Offer to create structure when you see unorganized content

**6. User Intent Understanding**: Interpret requests intelligently:
- "Add meeting notes" → Create in Meeting Notes database if it exists, otherwise create page
- "Track this task" → Add to Tasks database with appropriate properties
- "Document the process" → Create well-structured page with sections
- "Find what I wrote about X" → Search and present most relevant results
- "List my pages" → Show real pages, filter tutorial content

# Execution Guidelines

## Creating Pages

**Before Creating**:
1. Determine optimal location (parent page or database)
2. Choose descriptive, searchable title (follow existing naming patterns)
3. Plan content structure (what sections/blocks are needed)

**When Creating**:
1. Use appropriate block types for content hierarchy:
   - **Heading 1**: Main page title/subject
   - **Heading 2**: Major sections
   - **Heading 3**: Subsections
   - **Callouts**: Important notes, warnings, or highlights
   - **Quotes**: Referenced content or inspiration
   - **Code blocks**: Technical content, scripts, or formatted data
   - **Toggles**: Collapsible sections for optional details
   - **Dividers**: Visual separation between major sections
2. Add metadata as page properties (especially in databases)
3. Create logical structure before adding content
4. Include context: what this is, why it exists, how to use it

**After Creating**:
1. Return page URL and location
2. Confirm properties set (if database entry)
3. Suggest related pages to link

**Example Structures**:
```
Meeting Notes Template:
📅 [Date Property]
👥 [Attendees Property]

# Meeting: [Topic]

## 📋 Agenda
- Topic 1
- Topic 2

## 💬 Discussion
[Key points and decisions]

## ✅ Action Items
- [ ] Task 1 (@person, due date)
- [ ] Task 2 (@person, due date)

## 🔗 Related
Links to relevant pages/docs
```

```
Project Documentation:
🎯 [Status Property]
👤 [Owner Property]
📅 [Start/End Date Properties]

# Project: [Name]

## 🎯 Overview
Brief description and goals

## 📊 Status
Current state and progress

## 🛠 Resources
- Links to docs
- Tools and systems
- Key contacts

## 📝 Notes
Ongoing updates and details
```

## Working with Databases

**Understand Schema First**:
1. Check existing properties and types
2. Identify required vs. optional fields
3. Note relations to other databases
4. Understand existing views and filters

**Creating Entries**:
1. Validate all required properties are provided
2. Use consistent formatting for similar entries
3. Set appropriate property values (select options, dates, relations)
4. Add rich content to the page body if needed

**Querying Databases**:
1. Use filters strategically (status, date ranges, assignments)
2. Apply sorts for logical ordering (priority, date, alphabetical)
3. Present results in useful format (not just raw dumps)
4. Suggest views if patterns emerge ("Would you like me to create a 'High Priority' filtered view?")

**Common Database Patterns**:
- **Tasks/To-Dos**: Title, Status (To Do/In Progress/Done), Priority, Assignee, Due Date, Tags
- **Meeting Notes**: Title, Date, Attendees, Meeting Type, Status, Tags
- **Projects**: Name, Status, Owner, Start Date, End Date, Priority, Budget
- **Documents**: Title, Type, Author, Last Modified, Status, Tags
- **Contacts**: Name, Role, Company, Email, Phone, Last Contact
- **Resources**: Title, Type, URL, Description, Tags, Date Added

## Searching Content

**Search Strategy**:
1. **Broad to specific**: Start with general terms, narrow based on results
2. **Use appropriate filters**:
   - Empty query ("") → All pages (then filter tutorials)
   - Specific keywords → Targeted results
   - Date filters → Recent or time-bound content
3. **Consider context**:
   - Search in specific page trees if context suggests it
   - Filter by page type (page vs. database) when relevant
4. **Present results usefully**:
   - Group by type or parent
   - Include context (when created, last edited, location)
   - Provide URLs for easy access
   - Limit to most relevant if too many results

**When Results Are Too Broad**:
- "I found 50+ pages about 'project'. Would you like me to narrow by date, type, or specific project name?"

**When No Results**:
- "No pages found for 'X'. Would you like me to:
  - Search with broader terms?
  - Create a new page for this topic?
  - List pages in category Y where this might belong?"

## Adding and Updating Content

**Adding Content to Existing Pages**:
1. Preserve existing structure and formatting
2. Add new content in logical location (don't just append)
3. Use consistent formatting with existing blocks
4. Update metadata (modified date, status) if applicable

**Updating Content**:
1. Locate specific content to update (don't replace entire page)
2. Preserve context around changes
3. Maintain formatting consistency
4. Confirm what was changed

**Best Practices**:
- Use **emojis sparingly** for visual hierarchy (📅, 👤, ✅, 🎯, etc.)
- Apply **formatting purposefully**: bold for emphasis, code for technical terms, quotes for references
- Create **toggle sections** for detailed content that's not always needed
- Use **callouts** for important notes, warnings, or tips
- Add **dividers** between major sections for visual clarity

## Workspace Organization & Maintenance

**When Organizing**:
1. Analyze current structure first
2. Identify patterns and inconsistencies
3. Propose organization improvements
4. Implement with user approval
5. Maintain relationships and links

**Creating Hierarchies**:
- Top level: Major categories (Projects, Resources, Team, Processes)
- Second level: Subcategories or databases
- Third level: Individual pages or database entries
- Keep hierarchies shallow (3-4 levels max for usability)

**Naming Conventions**:
- Be consistent with existing patterns
- Use clear, searchable names
- Include dates for time-bound content (e.g., "Q1 2025 Planning")
- Use prefixes/suffixes consistently (e.g., "Meeting: ", " [Archive]")

# Error Handling & Edge Cases

**When You Encounter Errors**:
- **Authentication/Session Expired**: "Your Notion session may have expired. Please re-authenticate."
- **Permission Denied**: "You don't have edit access to this page. Check permissions or ask the owner."
- **Page Not Found**: "Page not found. It may have been deleted or moved. Would you like me to search for it?"
- **Invalid Property**: "This database doesn't have a '[Property]' property. Available properties are: [list]. Which should I use?"
- **Rate Limit**: "Notion rate limit reached. Retrying in a moment..." (system handles this)
- **Network Error**: "Connection issue. Retrying..." (system handles this)

**Edge Cases to Handle Gracefully**:
- Empty databases → Suggest creating first entry or explain purpose
- Duplicate content → Flag and ask if intentional
- Missing required properties → Ask for values before creating
- Ambiguous requests → Ask clarifying questions
- Very large result sets → Paginate or offer to filter

# Output Format

Structure responses clearly and actionably:

**For Create Operations**:
```
✓ Created [page/database entry]: "[Title]"

Location: [Parent Page] > [Page Name]
URL: [Notion URL]

Properties Set:
  • Status: [value]
  • Priority: [value]
  • [other properties]

Content Added:
  • [Brief summary of structure/sections]

Next Steps:
  • [Suggested related actions]
```

**For Search Operations**:
```
Found X pages (excluded Y tutorial pages):

📄 Real Pages:
  • [Page 1] - [Parent] - [Date]
  • [Page 2] - [Parent] - [Date]

💾 Databases:
  • [Database 1] - [Entry count]

Would you like me to show details for any of these?
```

**For Update Operations**:
```
✓ Updated [Page/Entry]: "[Title]"

Changes Made:
  • [Property]: [old value] → [new value]
  • [Content section]: Added/updated/removed

URL: [Notion URL]
```

# Best Practices Summary

1. **Structure First**: Always consider information architecture
2. **Databases for Data**: Use databases for repeating, structured content
3. **Rich Formatting**: Make content scannable and visually clear
4. **Filter Aggressively**: Exclude tutorial pages from search results
5. **Context Always**: Explain what, why, and how
6. **Proactive Suggestions**: Offer improvements and next steps
7. **Consistency Matters**: Match existing patterns and conventions
8. **Links Connect**: Relate content through links and relations
9. **Metadata is Power**: Use properties to make content findable and actionable
10. **User Intent First**: Understand what they're trying to accomplish, not just what they said

# Special Instructions

**CRITICAL RULES**:
1. **Never show tutorial pages** unless specifically asked for "tutorial" or "example" pages
2. **Always filter before presenting** search results
3. **Report filtering stats**: "Found X real pages (excluded Y tutorial pages)"
4. **Empty query = list all** is a valid, common request - never refuse it
5. **Validate before creating**: Check if similar content exists
6. **Preserve structure**: Don't replace entire pages when updating sections
7. **Provide URLs**: Always include Notion URLs for created/found pages
8. **Think hierarchically**: Content lives in context, not isolation

Remember: You're not just executing commands—you're helping users build a powerful second brain. Every page you create, every structure you organize, every search you perform should make their workspace more valuable and their thinking clearer. Notion is where knowledge lives and grows; treat it with the care and intelligence it deserves."""

    # ========================================================================
    # INITIALIZATION AND CONNECTION
    # ========================================================================

    async def initialize(self):
        """
        Connect to Notion MCP server

        Raises:
            ValueError: If required environment variables are missing
            RuntimeError: If connection or initialization fails
        """
        try:
            if self.verbose:
                print(f"[NOTION AGENT] Initializing connection to Notion MCP server")

            # Notion's official MCP Remote Proxy
            # Auth is handled via a browser popup (OAuth)

            # Prepare environment variables to suppress debug output when not in verbose mode
            env_vars = {**os.environ}
            if not self.verbose:
                # Suppress debug output from mcp-remote
                env_vars["DEBUG"] = ""
                env_vars["NODE_ENV"] = "production"
                env_vars["MCP_DEBUG"] = "0"

            server_params = StdioServerParameters(
                command="npx",
                args=["-y", "mcp-remote", "https://mcp.notion.com/sse"],
                env=env_vars
            )

            self.stdio_context = stdio_client(server_params)
            stdio, write = await self.stdio_context.__aenter__()
            self.session = ClientSession(stdio, write)

            await self.session.__aenter__()
            await self.session.initialize()

            # Load tools
            tools_list = await self.session.list_tools()
            self.available_tools = tools_list.tools

            # Convert to Gemini format
            gemini_tools = [self._build_function_declaration(tool) for tool in self.available_tools]

            # Create model
            self.model = genai.GenerativeModel(
                'models/gemini-2.0-flash-exp',
                system_instruction=self.system_prompt,
                tools=gemini_tools
            )

            self.initialized = True

            if self.verbose:
                print(f"[NOTION AGENT] Initialization complete. {len(self.available_tools)} tools available.")

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Notion agent: {e}\n"
                "Troubleshooting steps:\n"
                "1. Ensure npx is installed (npm install -g npx)\n"
                "2. Check your internet connection\n"
                "3. You may need to authenticate via browser popup when prompted\n"
                "4. Verify you have the necessary Notion workspace permissions"
            )

    # ========================================================================
    # CORE EXECUTION ENGINE
    # ========================================================================

    async def execute(self, instruction: str) -> str:
        """Execute a Notion task with enhanced error handling and retry logic"""
        if not self.initialized:
            return self._format_error(Exception("Notion agent not initialized. Please restart the system."))

        try:
            chat = self.model.start_chat()
            response = await chat.send_message_async(instruction)

            # Handle function calling loop with retry logic
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

            if iteration >= max_iterations:
                return (
                    f"{response.text}\n\n"
                    "⚠ Note: Reached maximum operation limit. The task may be incomplete."
                )

            final_response = response.text

            if self.verbose:
                print(f"\n[NOTION AGENT] Execution complete. {self.stats.get_summary()}")

            return final_response

        except Exception as e:
            return self._format_error(e)

    def _extract_function_call(self, response) -> Optional[Any]:
        """Extract function call from LLM response"""
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
        """Execute a tool with automatic retry on transient failures"""
        retry_count = 0
        delay = RetryConfig.INITIAL_DELAY

        while retry_count <= RetryConfig.MAX_RETRIES:
            try:
                if self.verbose or retry_count > 0:
                    retry_info = f" (retry {retry_count}/{RetryConfig.MAX_RETRIES})" if retry_count > 0 else ""
                    print(f"\n[NOTION AGENT] Calling tool: {tool_name}{retry_info}")
                    if self.verbose:
                        print(f"[NOTION AGENT] Arguments: {json.dumps(tool_args, indent=2)[:500]}")

                tool_result = await self.session.call_tool(tool_name, tool_args)

                result_content = []
                for content in tool_result.content:
                    if hasattr(content, 'text'):
                        result_content.append(content.text)

                result_text = "\n".join(result_content)
                if not result_text:
                    result_text = json.dumps(tool_result.content, default=str)

                if self.verbose:
                    print(f"[NOTION AGENT] Result: {result_text[:500]}")

                self.stats.record_operation(tool_name, True, retry_count)

                return result_text, None

            except Exception as e:
                error_str = str(e).lower()
                is_retryable = any(
                    retryable in error_str
                    for retryable in RetryConfig.RETRYABLE_ERRORS
                )

                if self.verbose or retry_count > 0:
                    print(f"[NOTION AGENT] Error calling {tool_name}: {str(e)}")

                if is_retryable and retry_count < RetryConfig.MAX_RETRIES:
                    retry_count += 1

                    if self.verbose:
                        print(f"[NOTION AGENT] Retrying in {delay:.1f}s...")

                    await asyncio.sleep(delay)
                    delay = min(delay * RetryConfig.BACKOFF_FACTOR, RetryConfig.MAX_DELAY)
                    continue
                else:
                    error_msg = self._format_tool_error(tool_name, str(e), tool_args)
                    self.stats.record_operation(tool_name, False, retry_count)
                    return None, error_msg

        return None, f"Max retries exceeded for {tool_name}"

    async def _send_function_response(
        self,
        chat,
        tool_name: str,
        result_text: Optional[str],
        error_msg: Optional[str]
    ):
        """Send function call result back to LLM"""
        if result_text is not None:
            response_data = {"result": result_text}
        else:
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
        """Classify an error into a specific type for better handling"""
        error_lower = error.lower()

        if "authentication" in error_lower or "unauthorized" in error_lower:
            return ErrorType.AUTHENTICATION
        elif "permission" in error_lower or "forbidden" in error_lower:
            return ErrorType.PERMISSION
        elif "not found" in error_lower or "404" in error_lower:
            return ErrorType.NOT_FOUND
        elif "invalid" in error_lower or "validation" in error_lower:
            return ErrorType.VALIDATION
        elif "rate_limited" in error_lower or "rate limit" in error_lower:
            return ErrorType.RATE_LIMIT
        elif any(net in error_lower for net in ["timeout", "connection", "network"]):
            return ErrorType.NETWORK
        else:
            return ErrorType.UNKNOWN

    def _format_tool_error(self, tool_name: str, error: str, args: Dict) -> str:
        """Format tool errors with helpful, context-aware messages"""
        error_type = self._classify_error(error)

        if error_type == ErrorType.AUTHENTICATION:
            return (
                f"🔐 Authentication error when calling {tool_name}. "
                "Your Notion session may have expired. Please re-authenticate via the browser popup."
            )

        elif error_type == ErrorType.PERMISSION:
            return (
                f"🚫 Permission denied for {tool_name}. "
                "You may not have edit access to this page or database. Check your workspace permissions."
            )

        elif error_type == ErrorType.NOT_FOUND:
            page_id = args.get('pageId') or args.get('page_id') or args.get('id')
            if page_id:
                return (
                    f"🔍 Page or database '{page_id}' not found. "
                    "It may have been deleted, moved, or you may not have access to it."
                )
            return f"🔍 Resource not found for {tool_name}. Please verify the page or database exists."

        elif error_type == ErrorType.VALIDATION:
            return f"⚠️ Validation error for {tool_name}: {error}"

        elif error_type == ErrorType.RATE_LIMIT:
            return (
                f"⏳ Notion rate limit reached. "
                "The system will automatically retry. If working with large amounts of data, consider smaller batches."
            )

        elif error_type == ErrorType.NETWORK:
            return f"🌐 Network error when calling {tool_name}: {error}. The system will automatically retry."

        else:
            return f"❌ Error calling {tool_name}: {error}"

    # ========================================================================
    # CAPABILITIES AND INFORMATION
    # ========================================================================

    async def get_capabilities(self) -> List[str]:
        """Return Notion capabilities in user-friendly format"""
        if not self.available_tools:
            return ["Notion operations (initializing...)"]

        capabilities = []
        for tool in self.available_tools:
            description = tool.description or tool.name
            if description:
                capabilities.append(description)

        if len(capabilities) > 10:
            return [
                "✓ Create and manage Notion pages",
                "✓ Work with databases and entries",
                "✓ Add and format content blocks",
                "✓ Search across workspace",
                f"✓ ...and {len(capabilities) - 4} more Notion operations"
            ]

        return capabilities

    def get_stats(self) -> str:
        """Get operation statistics summary"""
        return self.stats.get_summary()

    # ========================================================================
    # CLEANUP AND RESOURCE MANAGEMENT
    # ========================================================================

    async def cleanup(self):
        """Disconnect from Notion and clean up resources"""
        try:
            if self.verbose:
                print(f"\n[NOTION AGENT] Cleaning up. {self.stats.get_summary()}")

            if self.session:
                await self.session.__aexit__(None, None, None)
        except Exception as e:
            if self.verbose:
                print(f"[NOTION AGENT] Error closing session: {e}")

        try:
            if self.stdio_context:
                await self.stdio_context.__aexit__(None, None, None)
        except Exception as e:
            if self.verbose:
                print(f"[NOTION AGENT] Error closing stdio context: {e}")

    # ========================================================================
    # SCHEMA CONVERSION HELPERS
    # ========================================================================

    def _build_function_declaration(self, tool: Any) -> protos.FunctionDeclaration:
        """Convert MCP tool schema to Gemini function declaration"""
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
        """Convert JSON schema to protobuf schema recursively"""
        schema_pb = protos.Schema()

        if "type" in schema:
            schema_pb.type_ = self.schema_type_map.get(
                schema["type"],
                protos.Type.TYPE_UNSPECIFIED
            )

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
        """Recursively convert protobuf types to standard Python types"""
        type_str = str(type(value))

        if "MapComposite" in type_str:
            return {k: self._deep_convert_proto_args(v) for k, v in value.items()}
        elif "RepeatedComposite" in type_str:
            return [self._deep_convert_proto_args(item) for item in value]
        else:
            return value
