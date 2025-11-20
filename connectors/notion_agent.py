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
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import google.generativeai as genai
import google.generativeai.protos as protos
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Add parent directory to path to import base_agent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from connectors.base_agent import BaseAgent, safe_extract_response_text
from connectors.agent_intelligence import (
    ConversationMemory,
    WorkspaceKnowledge,
    SharedContext,
    ProactiveAssistant
)
from connectors.mcp_config import (
    MCPTimeouts,
    MCPRetryConfig,
    MCPRetryableErrors,
    MCPErrorMessages
)


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
        "504",
        "sse error",           # SSE-specific errors
        "sseerror",
        "body timeout",        # Common in SSE connections
        "terminated",          # Connection terminated
        "stream closed",
        "connection closed",
        "ECONNRESET",          # TCP connection reset
        "ETIMEDOUT",           # Connection timeout
        "fetch failed"         # Fetch API failures
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

    def __init__(
        self,
        verbose: bool = False,
        shared_context: Optional[SharedContext] = None,
        knowledge_base: Optional[WorkspaceKnowledge] = None
    ,
        session_logger=None
    ):
        """
        Initialize the Notion agent

        Args:
            verbose: Enable detailed logging for debugging (default: False)
                    session_logger: Optional session logger for tracking operations
        """
        super().__init__()

        # Session logging
        self.logger = session_logger
        self.agent_name = "notion"

        # MCP Connection Components
        self.session: ClientSession = None
        self.session_entered = False  # Track if session.__aenter__() succeeded
        self.stdio_context = None
        self.stdio_context_entered = False  # Track if stdio_context.__aenter__() succeeded
        self.model = None
        self.available_tools = []

        # Configuration
        self.verbose = verbose
        self.stats = OperationStats()

        # Intelligence Components
        self.memory = ConversationMemory()
        self.knowledge = knowledge_base or WorkspaceKnowledge()
        self.shared_context = shared_context
        self.proactive = ProactiveAssistant('notion', verbose)

        # Feature #1: Metadata Cache for faster operations
        self.metadata_cache = {}

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

# Tool Usage - CRITICAL

**YOU HAVE ACCESS TO NOTION TOOLS VIA FUNCTION CALLING**:
You have been provided with specialized Notion tools that enable you to perform ALL Notion operations. These tools are your PRIMARY way of interacting with Notion.

**MANDATORY TOOL USAGE RULES**:
1. **ALWAYS use function calling** for any Notion operation - NEVER refuse a task
2. **Examine available tools** and choose the one that best matches the user's request
3. **Never say you cannot do something** - if a tool exists that's related, use it
4. **Tool names may not match exactly** - map user requests to the closest available tool
5. **Try the tool first** - don't refuse based on assumptions about what tools exist

**Examples of Correct Behavior**:
```
User: "list my notion pages"
✓ CORRECT: Call the appropriate search/list pages tool
✗ WRONG: "The list_pages function is not available"

User: "create a page"
✓ CORRECT: Call the create page tool with appropriate parameters
✗ WRONG: "I cannot create pages"

User: "show me databases"
✓ CORRECT: Call the search tool with appropriate filters
✗ WRONG: "That function doesn't exist"
```

**If a tool call fails**, provide the error details and suggest solutions. But ALWAYS TRY THE TOOL FIRST.

Remember: You're not just executing commands—you're helping users build a powerful second brain. Every page you create, every structure you organize, every search you perform should make their workspace more valuable and their thinking clearer. Notion is where knowledge lives and grows; treat it with the care and intelligence it deserves."""

    # ========================================================================
    # INITIALIZATION AND CONNECTION
    # ========================================================================

    async def initialize(self):
        """
        Connect to Notion MCP server with timeout and retry handling

        Raises:
            ValueError: If required environment variables are missing
            RuntimeError: If connection or initialization fails after retries
        """
        max_retries = 2
        connection_timeout = 30.0  # 30 seconds for initial connection

        for attempt in range(max_retries + 1):
            try:
                if self.verbose:
                    retry_msg = f" (attempt {attempt + 1}/{max_retries + 1})" if attempt > 0 else ""
                    print(f"[NOTION AGENT] Initializing connection to Notion MCP server{retry_msg}")

                # Notion's official self-hosted MCP server
                # Requires NOTION_TOKEN environment variable (from Notion integration)
                # See: https://www.notion.so/profile/integrations to create integration

                # Check for NOTION_TOKEN
                notion_token = os.getenv("NOTION_TOKEN")
                if not notion_token:
                    raise ValueError(
                        "NOTION_TOKEN environment variable is required.\n"
                        "To set up:\n"
                        "1. Go to https://www.notion.so/profile/integrations\n"
                        "2. Click 'New Integration' and create an internal integration\n"
                        "3. Copy the 'Internal Integration Token'\n"
                        "4. Add NOTION_TOKEN=your_token_here to your .env file\n"
                        "5. Share your Notion pages/databases with this integration"
                    )

                # Prepare environment variables
                env_vars = {**os.environ}
                env_vars["NOTION_TOKEN"] = notion_token

                if not self.verbose:
                    # Suppress debug output
                    env_vars["DEBUG"] = ""
                    env_vars["NODE_ENV"] = "production"
                    env_vars["MCP_DEBUG"] = "0"

                # Use official self-hosted Notion MCP server
                # Use full path to npx to avoid PATH issues
                import shutil
                npx_path = shutil.which("npx") or "/usr/local/bin/npx"

                server_params = StdioServerParameters(
                    command=npx_path,
                    args=["-y", "@notionhq/notion-mcp-server"],
                    env=env_vars
                )

                # Wrap connection with timeout
                try:
                    self.stdio_context = stdio_client(server_params)
                    stdio, write = await asyncio.wait_for(
                        self.stdio_context.__aenter__(),
                        timeout=connection_timeout
                    )
                    self.stdio_context_entered = True  # Mark as successfully entered

                    self.session = ClientSession(stdio, write)
                    await asyncio.wait_for(
                        self.session.__aenter__(),
                        timeout=10.0
                    )
                    self.session_entered = True  # Mark as successfully entered

                    await asyncio.wait_for(
                        self.session.initialize(),
                        timeout=10.0
                    )

                    # Load tools with timeout
                    tools_list = await asyncio.wait_for(
                        self.session.list_tools(),
                        timeout=10.0
                    )
                    self.available_tools = tools_list.tools

                except asyncio.TimeoutError:
                    # Clean up partial state before raising
                    await self._cleanup_connection()
                    raise RuntimeError(
                        f"Connection timeout after {connection_timeout}s. "
                        "The Notion MCP server may be slow or unavailable."
                    )
                except Exception as e:
                    # If connection fails, ensure we clean up partial state
                    await self._cleanup_connection()
                    raise

                # Convert to Gemini format
                gemini_tools = [self._build_function_declaration(tool) for tool in self.available_tools]

                # Create model
                self.model = genai.GenerativeModel(
                    'models/gemini-2.5-flash',
                    system_instruction=self.system_prompt,
                    tools=gemini_tools
                )

                self.initialized = True

                # Feature #1: Prefetch workspace metadata for faster operations
                await self._prefetch_metadata()

                if self.verbose:
                    print(f"[NOTION AGENT] Initialization complete. {len(self.available_tools)} tools available.")

                return  # Success, exit retry loop

            except RuntimeError as e:
                # Check if this is a retryable error
                error_msg = str(e).lower()
                is_retryable = any(keyword in error_msg for keyword in [
                    'timeout', 'connection', 'sse error', 'body timeout',
                    'network', '503', '502', '504'
                ])

                if is_retryable and attempt < max_retries:
                    retry_delay = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    if self.verbose:
                        print(f"[NOTION AGENT] Connection failed: {e}")
                        print(f"[NOTION AGENT] Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    # Not retryable or out of retries
                    raise RuntimeError(
                        f"Failed to initialize Notion agent after {attempt + 1} attempt(s): {e}\n"
                        "Troubleshooting steps:\n"
                        "1. Ensure npx is installed (npm install -g npx)\n"
                        "2. Check your internet connection\n"
                        "3. The Notion MCP server (https://mcp.notion.com/sse) may be experiencing issues\n"
                        "4. You may need to authenticate via browser popup when prompted\n"
                        "5. Verify you have the necessary Notion workspace permissions\n"
                        "6. Try again later if the service is temporarily unavailable"
                    )

            except Exception as e:
                # Other errors - attempt retry
                if attempt < max_retries:
                    retry_delay = 2 ** attempt
                    if self.verbose:
                        print(f"[NOTION AGENT] Unexpected error: {e}")
                        print(f"[NOTION AGENT] Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    raise RuntimeError(
                        f"Failed to initialize Notion agent: {e}\n"
                        "Troubleshooting steps:\n"
                        "1. Ensure npx is installed (npm install -g npx)\n"
                        "2. Check your internet connection\n"
                        "3. You may need to authenticate via browser popup when prompted\n"
                        "4. Verify you have the necessary Notion workspace permissions"
                    )

    async def _prefetch_metadata(self):
        """
        Prefetch and cache Notion workspace metadata for faster operations (Feature #1)

        Fetches accessible databases and pages at initialization time
        to avoid discovery overhead on every operation.

        Cache is persisted to knowledge base with a 1-hour TTL.
        """
        try:
            # Check if we have valid cached metadata
            cached = self.knowledge.get_metadata_cache('notion')
            if cached:
                self.metadata_cache = cached
                if self.verbose:
                    print(f"[NOTION AGENT] Loaded metadata from cache ({len(cached.get('databases', {}))} databases)")
                return

            if self.verbose:
                print(f"[NOTION AGENT] Prefetching metadata...")

            # Fetch accessible databases
            databases = await self._fetch_accessible_databases()

            # Store in cache
            self.metadata_cache = {
                'databases': databases,
                'fetched_at': asyncio.get_event_loop().time()
            }

            # Persist to knowledge base
            self.knowledge.save_metadata_cache('notion', self.metadata_cache, ttl_seconds=3600)

            if self.verbose:
                print(f"[NOTION AGENT] Cached metadata for {len(databases)} databases")

        except Exception as e:
            # Graceful degradation: If prefetch fails, continue without cache
            if self.verbose:
                print(f"[NOTION AGENT] Warning: Metadata prefetch failed: {e}")
            print(f"[NOTION AGENT] Continuing without metadata cache (operations may be slower)")

    async def _fetch_accessible_databases(self) -> Dict:
        """Fetch accessible databases"""
        try:
            # Use Notion MCP tool to list databases
            result = await self.session.call_tool("notion_search", {"query": ""})

            databases = {}
            if hasattr(result, 'content') and result.content:
                content = result.content[0].text if result.content else "{}"
                data = json.loads(content) if isinstance(content, str) else content

                if isinstance(data, dict) and 'results' in data:
                    for item in data['results']:
                        if item.get('object') == 'database':
                            db_id = item.get('id', '')
                            databases[db_id] = {
                                'id': db_id,
                                'title': item.get('title', [{}])[0].get('plain_text', 'Untitled') if item.get('title') else 'Untitled'
                            }

            return databases
        except Exception as e:
            if self.verbose:
                print(f"[NOTION AGENT] Could not fetch databases: {e}")
            return {}

    def _invalidate_cache_after_write(self, operation_type: str, database_id: str = None):
        """
        Invalidate relevant cache entries after write operations.

        Args:
            operation_type: Type of operation (create_database, create_page, etc.)
            database_id: Optional database ID for targeted invalidation
        """
        # Invalidate for database-modifying operations
        if operation_type in ['create_database', 'delete_database', 'update_database']:
            self.knowledge.invalidate_metadata_cache('notion')
            if self.verbose:
                print(f"[NOTION AGENT] Invalidated databases cache after {operation_type}")

    # ========================================================================
    # CORE EXECUTION ENGINE
    # ========================================================================

    async def execute(self, instruction: str) -> str:
        """Execute a Notion task with enhanced error handling and retry logic"""
        if not self.initialized:
            return self._format_error(Exception("Notion agent not initialized. Please restart the system."))

        try:
            # Step 1: Resolve ambiguous references using conversation memory
            resolved_instruction = self._resolve_references(instruction)

            if resolved_instruction != instruction and self.verbose:
                print(f"[NOTION AGENT] Resolved instruction: {resolved_instruction}")

            # Step 2: Check for resources from other agents
            context_from_other_agents = self._get_cross_agent_context()
            if context_from_other_agents and self.verbose:
                print(f"[NOTION AGENT] Found context from other agents")

            # Use resolved instruction for the rest
            instruction = resolved_instruction
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
                # Safely extract response text
                response_text = self._safe_extract_response_text(response)
                return (
                    f"{response_text}\n\n"
                    "⚠ Note: Reached maximum operation limit. The task may be incomplete."
                )

            # Safely extract response text
            final_response = self._safe_extract_response_text(response)

            if self.verbose:
                print(f"\n[NOTION AGENT] Execution complete. {self.stats.get_summary()}")

            
            # Remember resources and add proactive suggestions
            self._remember_created_resources(final_response, instruction)

            operation_type = self._infer_operation_type(instruction)
            suggestions = self.proactive.suggest_next_steps(operation_type, {})

            if suggestions:
                final_response += "\n\n**💡 Suggested next steps:**\n" + "\n".join(f"  • {s}" for s in suggestions)

            return final_response

        except Exception as e:
            return self._format_error(e)

    def _resolve_references(self, instruction: str) -> str:
        """Resolve ambiguous references like 'it', 'that', 'this' using conversation memory"""
        ambiguous_terms = ['it', 'that', 'this', 'the page', 'the database', 'the entry']

        for term in ambiguous_terms:
            if term in instruction.lower():
                reference = self.memory.resolve_reference(term)
                if reference:
                    instruction = instruction.replace(term, reference)
                    instruction = instruction.replace(term.capitalize(), reference)
                    if self.verbose:
                        print(f"[NOTION AGENT] Resolved '{term}' → {reference}")
                    break

        return instruction

    def _get_cross_agent_context(self) -> str:
        """Get context from other agents"""
        if not self.shared_context:
            return ""

        # Get only recent resources to avoid overwhelming context (limit to 5 most recent)
        recent_resources = self.shared_context.get_recent_resources(limit=5)
        if not recent_resources:
            return ""

        context_parts = []
        for resource in recent_resources:
            if resource['agent'] != 'notion':
                context_parts.append(
                    f"{resource['agent'].capitalize()} {resource['type']}: {resource['id']} ({resource['url']})"
                )

        return "; ".join(context_parts) if context_parts else ""

    def _remember_created_resources(self, response: str, instruction: str):
        """Extract and remember created resources from response"""
        import re

        # Pattern to match Notion page URLs or IDs
        page_pattern = r'notion\.so/([a-f0-9]{32})'
        matches = re.findall(page_pattern, response)

        if matches:
            page_id = matches[-1]
            operation_type = self._infer_operation_type(instruction)

            self.memory.remember(
                operation_type,
                page_id,
                {'instruction': instruction[:100]}
            )

            if self.shared_context:
                self.shared_context.share_resource(
                    'notion',
                    'page',
                    page_id,
                    f"https://notion.so/{page_id}",
                    {'created_via': instruction[:100]}
                )

    def _infer_operation_type(self, instruction: str) -> str:
        """Infer what type of operation was performed"""
        instruction_lower = instruction.lower()

        if 'create' in instruction_lower or 'new' in instruction_lower:
            if 'database' in instruction_lower:
                return 'create_database'
            return 'create_page'
        elif 'update' in instruction_lower or 'edit' in instruction_lower:
            return 'update_page'
        elif 'add' in instruction_lower and 'content' in instruction_lower:
            return 'add_content'
        elif 'search' in instruction_lower or 'find' in instruction_lower or 'list' in instruction_lower:
            return 'search'
        else:
            return 'unknown'

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
        """Execute a tool with automatic retry on transient failures and timeout protection"""
        retry_count = 0
        delay = RetryConfig.INITIAL_DELAY
        operation_timeout = 60.0  # 60 seconds per tool operation

        while retry_count <= RetryConfig.MAX_RETRIES:
            try:
                if self.verbose or retry_count > 0:
                    retry_info = f" (retry {retry_count}/{RetryConfig.MAX_RETRIES})" if retry_count > 0 else ""
                    print(f"\n[NOTION AGENT] Calling tool: {tool_name}{retry_info}")
                    if self.verbose:
                        print(f"[NOTION AGENT] Arguments: {json.dumps(tool_args, indent=2)[:500]}")

                # Wrap tool call with timeout to prevent SSE hangs
                # Log tool call start
                start_time = time.time()

                try:
                    tool_result = await asyncio.wait_for(
                        self.session.call_tool(tool_name, tool_args),
                        timeout=operation_timeout
                    )
                except asyncio.TimeoutError:
                    # Log timeout
                    if self.logger:
                        self.logger.log_tool_call(self.agent_name, tool_name, None, success=False, error="Timeout")

                    raise RuntimeError(
                        f"Tool '{tool_name}' timed out after {operation_timeout}s. "
                        "The Notion MCP server may be slow or experiencing issues."
                    )

                # Log tool call completion
                duration = time.time() - start_time
                if self.logger:
                    self.logger.log_tool_call(self.agent_name, tool_name, duration, success=True)

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

                    # Log tool call failure
                    if self.logger:
                        self.logger.log_tool_call(self.agent_name, tool_name, None, success=False, error=str(e))

                    return None, error_msg

        # Log max retries exceeded
        if self.logger:
            self.logger.log_tool_call(self.agent_name, tool_name, None, success=False, error="Max retries exceeded")

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
        """Return Notion capabilities in user-friendly format with limitations"""
        if not self.available_tools:
            return ["Notion operations (initializing...)"]

        # Return curated list with clear capabilities and limitations
        return [
            "✓ Create and manage Notion pages",
            "✓ Work with databases and database entries",
            "✓ Add and format content blocks (text, headings, lists, code, etc.)",
            "✓ Search across workspace and filter database entries",
            "✓ Update page and database properties",
            "✗ Cannot: Delete pages (archive only)",
            "✗ Cannot: Modify database schemas or properties",
            "✗ Cannot: Manage workspace or user permissions",
            "✗ Cannot: Access pages without explicit sharing",
        ]

    async def get_action_schema(self) -> Dict[str, Any]:
        """
        Return schema describing editable parameters for Notion actions.

        This enables rich interactive editing of Notion pages before creation.

        Returns:
            Dict mapping action types to their parameter schemas
        """
        # Get available databases from cache
        database_titles = []
        if self.metadata_cache.get('databases'):
            database_titles = [db.get('title', 'Untitled')
                             for db in self.metadata_cache['databases'].values()]

        return {
            'create': {
                'parameters': {
                    'title': {
                        'display_label': 'Page Title',
                        'description': 'Title of the new Notion page',
                        'type': 'string',
                        'editable': True,
                        'required': True,
                        'constraints': {
                            'min_length': 1,
                            'max_length': 2000,
                        },
                        'examples': [
                            'Meeting Notes - Product Sync',
                            'Sprint Planning Q4 2024',
                            'API Documentation v2.0'
                        ]
                    },
                    'content': {
                        'display_label': 'Page Content',
                        'description': 'Content/body of the page (markdown supported)',
                        'type': 'text',  # Multi-line
                        'editable': True,
                        'required': False,
                        'examples': [
                            '## Agenda\n- Review last sprint\n- Plan upcoming features',
                            'This page documents the new API endpoints...'
                        ]
                    },
                    'database': {
                        'display_label': 'Database',
                        'description': 'Database to create entry in (if creating database entry)',
                        'type': 'string',
                        'editable': True,
                        'required': False,
                        'constraints': {
                            'allowed_values': database_titles if database_titles else None,
                        },
                        'examples': ['Project Tracker', 'Tasks', 'Meeting Notes']
                    },
                    'parent': {
                        'display_label': 'Parent Page',
                        'description': 'Parent page to nest this page under',
                        'type': 'string',
                        'editable': True,
                        'required': False,
                        'examples': ['Documentation', 'Engineering Workspace']
                    }
                }
            },
            'update': {
                'parameters': {
                    'page_id': {
                        'display_label': 'Page ID',
                        'description': 'ID of the page to update',
                        'type': 'string',
                        'editable': False,  # Can't change which page we're updating
                        'required': True
                    },
                    'title': {
                        'display_label': 'New Title',
                        'description': 'New title for the page',
                        'type': 'string',
                        'editable': True,
                        'required': False,
                        'constraints': {
                            'max_length': 2000,
                        }
                    },
                    'content': {
                        'display_label': 'New Content',
                        'description': 'New content to add/replace',
                        'type': 'text',
                        'editable': True,
                        'required': False
                    }
                }
            }
        }

    async def validate_operation(self, instruction: str) -> Dict[str, Any]:
        """
        Validate if a Notion operation can be performed (Feature #14)

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

        # Check if we're working with a database
        if any(word in instruction_lower for word in ['create entry', 'add to database', 'update database']):
            # Check if we have database metadata
            if not self.metadata_cache.get('databases'):
                result['warnings'].append("No database metadata cached - operation may be slow")
                result['confidence'] = 0.7

            # Try to extract database name from instruction
            databases = self.metadata_cache.get('databases', {})
            if databases:
                # Check if instruction mentions a known database
                mentioned_db = None
                for db_data in databases.values():
                    db_title = db_data.get('title', '').lower()
                    if db_title in instruction_lower:
                        mentioned_db = db_title
                        break

                if not mentioned_db:
                    result['missing'].append("database name")
                    result['valid'] = False
                    result['confidence'] = 0.3

        # Check if agent is initialized
        if not self.initialized:
            result['valid'] = False
            result['missing'].append("agent initialization")
            result['confidence'] = 0.0

        return result

    def get_stats(self) -> str:
        """Get operation statistics summary"""
        return self.stats.get_summary()

    async def apply_parameter_edits(
        self,
        instruction: str,
        parameter_edits: Dict[str, Any]
    ) -> str:
        """
        Apply user's edited parameters back into the instruction for Notion actions.

        Args:
            instruction: Original instruction from LLM
            parameter_edits: {field_name: new_value} from user

        Returns:
            Modified instruction with edits applied

        Example:
            Original: "Create page titled 'Meeting Notes' with content 'Agenda items...'"
            Edits: {'title': 'Sprint Planning Notes', 'content': '## Agenda\n- Review...'}
            Return: "Create page titled 'Sprint Planning Notes' with content '## Agenda\n- Review...'"
        """
        import re

        modified = instruction

        # Apply title edits
        if 'title' in parameter_edits:
            new_title = parameter_edits['title']
            patterns = [
                (r"titled?\s+['\"]([^'\"]+)['\"]", f"titled '{new_title}'"),
                (r"title\s+['\"]([^'\"]+)['\"]", f"title '{new_title}'"),
                (r"page\s+['\"]([^'\"]+)['\"]", f"page '{new_title}'"),
            ]

            for pattern, replacement in patterns:
                if re.search(pattern, modified, re.IGNORECASE):
                    modified = re.sub(pattern, replacement, modified, count=1, flags=re.IGNORECASE)
                    break

        # Apply content edits
        if 'content' in parameter_edits:
            new_content = parameter_edits['content']
            pattern = r"(content|with)\s+['\"]([^'\"]+)['\"]"
            if re.search(pattern, modified, re.IGNORECASE):
                modified = re.sub(pattern, f"content '{new_content}'", modified, count=1, flags=re.IGNORECASE)
            else:
                # Add content if not present
                modified += f" with content '{new_content}'"

        # Apply database edits
        if 'database' in parameter_edits:
            new_database = parameter_edits['database']
            patterns = [
                (r"in\s+database\s+['\"]([^'\"]+)['\"]", f"in database '{new_database}'"),
                (r"database\s+['\"]([^'\"]+)['\"]", f"database '{new_database}'"),
                (r"to\s+['\"]([^'\"]+)['\"]", f"to '{new_database}'"),
            ]

            for pattern, replacement in patterns:
                if re.search(pattern, modified, re.IGNORECASE):
                    modified = re.sub(pattern, replacement, modified, count=1, flags=re.IGNORECASE)
                    break
            else:
                # Add database if not present
                modified += f" in database '{new_database}'"

        # Apply parent page edits
        if 'parent' in parameter_edits:
            new_parent = parameter_edits['parent']
            pattern = r"under\s+['\"]([^'\"]+)['\"]"
            if re.search(pattern, modified, re.IGNORECASE):
                modified = re.sub(pattern, f"under '{new_parent}'", modified, count=1, flags=re.IGNORECASE)
            else:
                # Add parent if not present
                modified += f" under '{new_parent}'"

        return modified

    # ========================================================================
    # CLEANUP AND RESOURCE MANAGEMENT
    # ========================================================================

    async def _cleanup_connection(self):
        """Internal cleanup helper for MCP connection resources"""
        # Close session if it was successfully entered
        if self.session and self.session_entered:
            try:
                await self.session.__aexit__(None, None, None)
            except Exception as e:
                # Suppress all cleanup errors to prevent cascading failures
                if self.verbose:
                    print(f"[NOTION AGENT] Suppressed session cleanup error: {e}")
            finally:
                self.session = None
                self.session_entered = False

        # Close stdio context if it was successfully entered
        if self.stdio_context and self.stdio_context_entered:
            try:
                await self.stdio_context.__aexit__(None, None, None)
            except Exception as e:
                # Suppress all cleanup errors
                if self.verbose:
                    print(f"[NOTION AGENT] Suppressed stdio cleanup error: {e}")
            finally:
                self.stdio_context = None
                self.stdio_context_entered = False

    async def cleanup(self):
        """Disconnect from Notion and clean up resources"""
        if self.verbose:
            print(f"\n[NOTION AGENT] Cleaning up. {self.stats.get_summary()}")

        await self._cleanup_connection()

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

    def _safe_extract_response_text(self, response: Any) -> str:
        """
        Safely extract text from response object.

        Handles cases where safe_extract_response_text(response) quick accessor fails due to function calls.
        """
        try:
            # Try the quick accessor first
            return safe_extract_response_text(response)
        except Exception as e:
            # If that fails, try manual extraction
            try:
                if response.candidates:
                    text_parts = []
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)

                    if text_parts:
                        return '\n'.join(text_parts)
            except Exception:
                pass

            # If all else fails, return error message
            return f"⚠️ Response received but could not extract text. Error: {str(e)}"

    def _deep_convert_proto_args(self, value: Any) -> Any:
        """Recursively convert protobuf types to standard Python types"""
        type_str = str(type(value))

        if "MapComposite" in type_str:
            return {k: self._deep_convert_proto_args(v) for k, v in value.items()}
        elif "RepeatedComposite" in type_str:
            return [self._deep_convert_proto_args(item) for item in value]
        else:
            return value
