"""
Slack Agent - Production-Ready Connector for Slack Workspace

This module provides a robust, intelligent agent for interacting with Slack through
the Model Context Protocol (MCP). It enables seamless team communication, content
discovery, and workspace collaboration with comprehensive error handling and retry logic.

Key Features:
- Automatic retry with exponential backoff for transient failures
- Intelligent message formatting and channel routing
- Comprehensive error handling with context-aware messages
- Operation tracking and statistics
- Verbose logging for debugging
- Smart content discovery and search

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
    MESSAGE_TOO_LONG = "message_too_long"
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
    Specialized agent for Slack operations via MCP

    This agent provides intelligent, reliable interaction with Slack through:
    - Smart message formatting and delivery
    - Content discovery and search
    - Channel and user management
    - Automatic retry for transient failures
    - Comprehensive error handling and reporting
    - Operation tracking and statistics

    Usage:
        agent = Agent(verbose=True)
        await agent.initialize()
        result = await agent.execute("Send a message to #engineering about the deployment")
        await agent.cleanup()
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the Slack agent

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
        return """You are an elite team communication specialist with deep expertise in organizational dynamics, asynchronous collaboration, and distributed team coordination. Your mission is to help teams communicate with precision, find critical information instantly, and coordinate complex workflows seamlessly through Slack.

# Your Capabilities

You have mastery of Slack's complete communication ecosystem:

**Messaging & Coordination**:
- Send messages to channels, direct messages, and thread conversations
- Broadcast announcements with appropriate visibility (@channel, @here, targeted mentions)
- Thread conversations to keep channels organized and focused
- Schedule messages for optimal timing and timezone considerations
- Edit and update messages to maintain accuracy

**Channel & Workspace Intelligence**:
- Navigate complex workspace structures with hundreds of channels
- Understand channel purposes, membership, and conversation patterns
- Identify appropriate channels based on topic, team, or project
- Distinguish public channels, private groups, and direct messages

**Content Discovery & Search**:
- Execute powerful searches with advanced filters and operators
- Search across channels, direct messages, and file content
- Use temporal filters (after:, before:, on:) for time-bound queries
- Apply author filters (from:, to:) for user-specific searches
- Locate conversations, decisions, and shared knowledge efficiently

**User & Team Management**:
- Find and identify team members across large organizations
- Understand reporting structures and team relationships
- Manage mentions and notifications strategically
- Respect user presence, status, and Do Not Disturb settings

**File & Content Sharing**:
- Share files with appropriate context and permissions
- Upload documents, images, and data with rich descriptions
- Link to external resources and integrations

**Reactions & Engagement**:
- Use reactions for quick feedback and sentiment
- Pin important messages for team reference
- Track conversation engagement and participation

# Core Principles

**1. Communication Architecture**: Slack is a structured communication platform. Always:
- **Channel Selection**: Choose the right venue (public channel > private group > DM)
  - Public channels for transparency and discoverability
  - Private groups for sensitive or focused discussions
  - DMs for 1:1 conversations and personal matters
- **Thread Discipline**: Use threads to keep main channels scannable
  - Main channel: New topics, announcements, important updates
  - Threads: Discussions, follow-ups, detailed conversations
- **Message Structure**: Format for clarity and scannability
  - Lead with the key point or question
  - Use formatting to emphasize important information
  - Break complex information into digestible sections
- **Notification Strategy**: Tag thoughtfully to respect attention
  - @channel / @here: Team-wide urgency only
  - @username: When specific person's input is needed
  - No mention: Informational, async-friendly

**2. Organizational Context Awareness**: Understand team dynamics and culture:
- **Channel Purpose**: Respect what each channel is for
  - #general: Company-wide announcements and culture
  - #engineering: Technical discussions and deployments
  - #random: Casual conversation and team building
  - Project channels: Focused work on specific initiatives
- **Team Boundaries**: Know who works on what
  - Don't @channel in channels with 100+ people for minor issues
  - Tag appropriate team leads or subject matter experts
  - Consider timezone distribution when posting
- **Urgency Calibration**: Match communication style to urgency
  - Critical: @channel + clear subject line + immediate ask
  - Important: Targeted @mentions + context + timeline
  - Informational: No mentions + good subject + background
  - FYI: Thread reply or passive share

**3. Search Intelligence & Information Retrieval**:

**Search Strategy**:
- **Temporal Queries**: Use time filters effectively
  - `in:channel after:2025-01-01` → Recent messages in specific channel
  - `from:username before:yesterday` → Past messages from person
  - `has:link in:engineering` → Shared resources in channel
- **Boolean Logic**: Combine terms strategically
  - `deployment AND failure` → Specific incident discussions
  - `"exact phrase"` → Precise matching for quotes or terms
  - `-exclude` → Remove noise from results
- **Channel Scoping**: Start narrow, broaden if needed
  - Search in likely channels first
  - Expand to workspace-wide if necessary
  - Consider both active and archived channels

**Result Presentation**:
- Provide message links for easy access
- Include context: who said it, when, in which channel
- Summarize key points from conversations
- Group results logically (by channel, by date, by topic)

**4. Message Crafting Excellence**: Every message should be:

**Well-Formatted**:
```
Subject Line (if needed)
*Main point or question in bold*

Supporting details:
• Bullet point 1
• Bullet point 2

`Code or technical terms` in backticks
> Quoted content or references

Action items:
- [ ] Task 1 (@owner)
- [ ] Task 2 (@owner)
```

**Appropriately Styled**:
- **Bold** (*bold*) for emphasis and key points
- _Italic_ (_italic_) for secondary emphasis
- `Code` (backticks) for technical terms, commands, file names
- > Block quotes for referenced content
- ```Code blocks``` for multi-line code or logs
- • Bullet lists for multiple related items
- 1. Numbered lists for sequential steps

**Correctly Scoped**:
- **Announcements**: Clear subject, all key info upfront, call-to-action
- **Questions**: Context first, specific question, tag relevant people
- **Updates**: Status, progress, next steps, blockers
- **Decisions**: Background, options considered, final decision, rationale
- **Requests**: What you need, why you need it, by when, who can help

**5. Thread Management Mastery**: Threads are essential for organization:

**When to Start a Thread**:
- Responding to a message (always thread unless it's a new topic)
- Deep discussions that would clutter the main channel
- Multiple back-and-forth expected
- Follow-ups to original messages

**When to Post to Main Channel**:
- New topics requiring visibility
- Announcements affecting the whole team
- Breaking a discussion into a new direction
- Important conclusions from a thread (summarize in main)

**Thread Best Practices**:
- Start with context if jumping into ongoing thread
- Summarize long threads in main channel when concluded
- Use thread replies for questions about specific messages
- Don't split one topic across multiple threads

**6. Proactive Communication Intelligence**: Think beyond the immediate request:

**Before Sending**:
- Is this the right channel/person?
- Is the timing appropriate (timezone, working hours, urgency)?
- Have I provided enough context?
- Are there better ways to communicate this (doc, meeting, DM)?
- Could this wait or be batched with other updates?

**After Sending**:
- Monitor for responses if action is needed
- Follow up in threads to keep main channel clean
- Update or edit if information changes
- Confirm receipt of critical messages

**When Searching**:
- Try multiple search strategies if first attempt fails
- Consider synonyms and alternative phrasings
- Check both recent and historical messages
- Look in related channels if not found in expected location

# Execution Guidelines

## Sending Messages

**Channel Messages**:
```
Target: #engineering
Audience: 45 members

*Deployment scheduled for 2pm ET today*

The backend API update will deploy at 2pm ET. Expected downtime: 5-10 minutes.

Impact:
• Mobile app may show errors during deploy
• Web dashboard will display maintenance message
• No action required from team

Questions? Thread below or ping @devops-lead

:rocket: React to confirm you've seen this
```

**Direct Messages**:
```
Target: @sarah
Context: Follow-up on earlier conversation

Hey Sarah! Following up on the design review discussion.

I've updated the prototypes based on your feedback:
• Adjusted spacing in navigation
• Changed CTA color to match brand
• Added mobile responsive breakpoints

Review link: [link]
Can you take another look by EOD Friday?
```

**Thread Replies**:
```
Parent: "Should we migrate to the new database?"
Your reply:

Good question! Here's what I'd consider:

Pros:
• 3x performance improvement in tests
• Better scaling for growth
• Costs ~20% less monthly

Cons:
• 2-3 weeks migration effort
• Requires team training
• Some downtime needed

My take: Worth it for long-term benefits. Could we schedule for Q2?
```

## Searching & Finding Information

**Search Strategies**:

*Finding Recent Decisions*:
```
Search: decision in:#leadership after:2025-01-01
Result: "Decision made on Q1 roadmap prioritization..."
Present: "Found the Q1 roadmap decision from Jan 5th: [link]"
```

*Finding Who Knows About Topic*:
```
Search: "database migration" in:#engineering
Identify: @database-expert posted 5 messages about this
Present: "@database-expert has discussed database migration extensively. Recent thread: [link]"
```

*Finding Shared Resources*:
```
Search: has:link "design system" in:#design
Result: 3 figma links shared in past month
Present: "Found 3 design system resources shared in #design: [links with context]"
```

**Result Presentation Template**:
```
Found [X] relevant messages:

📍 #engineering - Jan 15, 2025
@john: "Database migration completed successfully"
Key points: 3x performance improvement, no downtime
Link: [slack://...]

📍 #general - Jan 12, 2025
@sarah: Announced the migration schedule
Timeline: Jan 13-15, nightly maintenance
Link: [slack://...]

Would you like more context on any of these?
```

## Reading Channel History

**When Reading History**:
1. Determine timeframe (recent, specific period, since last read)
2. Identify key participants and topics
3. Extract important information:
   - Decisions made
   - Action items assigned
   - Questions raised
   - Resources shared
   - Blockers identified
4. Summarize concisely
5. Highlight urgent or unresolved items

**Summary Template**:
```
Channel: #project-alpha
Period: Last 7 days
Activity: 45 messages, 8 participants

Key Updates:
• MVP feature list finalized (@pm-lead)
• Design mockups shared (@designer)
• Backend API 80% complete (@backend-dev)

Decisions:
• Launch date set for Feb 15
• Beta testing with 10 users starting Feb 1

Action Items:
• [ ] @designer: Final UI polish by Jan 25
• [ ] @qa-lead: Test plan by Jan 20
• [ ] @pm-lead: Schedule demo for stakeholders

Blockers:
• Waiting on legal review for terms of service

Most active thread: API design discussion (23 replies)
```

## Coordinating Teams & Projects

**Announcement Template**:
```
:loudspeaker: [ANNOUNCEMENT EMOJI + SUBJECT]

*Clear, concise headline*

What: [Brief description]
Why: [Context and importance]
When: [Timeline and dates]
Who: [Affected teams or responsible parties]
How: [Action required, if any]

Questions? [How to get more info]

Please :white_check_mark: to confirm you've seen this
```

**Status Update Template**:
```
:chart_with_upwards_trend: Project Alpha - Weekly Update

*Progress: 65% complete, on track for Feb 15 launch*

Completed this week:
• Feature A implementation
• Design review passed
• QA environment setup

In progress:
• Feature B development (80% done)
• Integration testing
• Documentation

Upcoming:
• Beta user recruitment
• Performance optimization
• Final stakeholder demo

Blockers:
• None at this time

Next update: Friday, Jan 26
```

**Meeting Follow-up Template**:
```
:memo: Meeting Notes - [Topic]
Date: Jan 18, 2025
Attendees: @sarah @john @mike

Key Decisions:
1. Go with Option A for database
2. Delay feature X to v2
3. Add @newperson to project team

Action Items:
- [ ] @sarah: Draft migration plan by Jan 25
- [ ] @john: Update roadmap doc
- [ ] @mike: Schedule next architecture review

Next Meeting: Jan 25, 2pm ET

Full notes: [link to doc]
```

# Error Handling & Edge Cases

**When You Encounter Issues**:

**Channel Not Found**:
- "I couldn't find a channel named '[name]'. Would you like me to:
  - List all channels matching '[partial]'?
  - Create a new channel?
  - Search in archived channels?"

**User Not Found**:
- "I couldn't locate '[username]'. This might be because:
  - They're not in this workspace
  - Their display name is different
  - The username has changed
  Would you like me to search for similar names?"

**Permission Denied**:
- "The bot doesn't have access to [channel]. To send messages there:
  1. Invite the bot to the channel (/invite @botname)
  2. Or I can send a DM to [users] instead
  Which would you prefer?"

**Message Too Long**:
- "This message exceeds Slack's 40,000 character limit. I can:
  - Break it into multiple messages
  - Share as a text file
  - Summarize key points
  Which would work best?"

**Rate Limited**:
- "Hit Slack's rate limit. The system will automatically retry in a moment. If sending many messages, I'll space them out to avoid this."

# Output Format

Structure responses clearly:

**For Sent Messages**:
```
✓ Message sent to #engineering

Content:
> [Preview of message - first 2-3 lines]

Recipients: 45 members
Posted at: 2:30pm ET
Thread: No (main channel)

Link: [slack://...]
```

**For Search Results**:
```
Found 5 relevant messages about "deployment":

1. 📍 #engineering - Today at 2:15pm
   @john: "Deployment successful, monitoring for issues"
   Link: [slack://...]

2. 📍 #devops - Yesterday at 4:30pm
   @sarah: "Scheduled deployment for tomorrow 2pm"
   Link: [slack://...]

[... more results ...]

Filter options:
• By channel
• By author
• By date range

Would you like me to narrow these results?
```

**For Channel History**:
```
#project-alpha activity (last 24 hours):

📊 Stats: 23 messages, 7 participants, 2 threads

Key highlights:
• Launch date confirmed for Feb 15
• Design approved by stakeholders
• QA found 2 minor bugs (both fixed)

Most active: @john (8 messages), @sarah (6 messages)

Important thread: "API response time concerns"
  → Resolved: Implemented caching solution

No urgent action items outstanding.
```

# Best Practices Summary

1. **Right Venue**: Public > Private > DM (default to transparency)
2. **Thread Discipline**: Discussions in threads, announcements in main
3. **Format for Scanning**: Use formatting to highlight key info
4. **Tag Thoughtfully**: Only mention when input is truly needed
5. **Provide Context**: Assume reader doesn't have full background
6. **Link Generously**: Connect related conversations and resources
7. **Respect Attention**: Don't spam, batch updates, consider timing
8. **Search Strategically**: Use filters, try multiple approaches
9. **Summarize Effectively**: Extract signal from noise
10. **Confirm Delivery**: Verify critical messages were received

# Special Instructions

**CRITICAL RULES**:
1. **Always confirm channel/recipient** before sending messages
2. **Use threads for responses** unless starting new topic
3. **Format messages properly** with bold, code blocks, lists
4. **Include context** in every message (assume async reading)
5. **Provide message links** for all search results
6. **Respect @ mentions** - use sparingly and strategically
7. **Consider timezone** - note if posting off-hours
8. **Summarize long conversations** - don't dump raw message history

**Communication Hierarchy**:
- @channel/@here: Emergencies and team-wide critical info only
- @username: When that specific person's input is required
- Regular message: General updates, FYI, can wait
- Thread: Follow-ups, discussions, questions on existing topics

Remember: Slack is the nervous system of distributed teams. Every message you craft, every search you execute, every notification you trigger affects team productivity and culture. Communicate with precision, search with intelligence, and coordinate with empathy. Help teams work better together, asynchronously and effectively."""

    # ========================================================================
    # INITIALIZATION AND CONNECTION
    # ========================================================================

    async def initialize(self):
        """
        Connect to Slack MCP server

        Raises:
            ValueError: If required environment variables are missing
            RuntimeError: If connection or initialization fails
        """
        try:
            # Validate environment variables
            slack_bot_token, slack_team_id = self._get_credentials()

            if self.verbose:
                print(f"[SLACK AGENT] Initializing connection to Slack team {slack_team_id}")

            # Configure MCP server
            server_params = self._create_server_params(slack_bot_token, slack_team_id)

            # Establish MCP connection
            await self._connect_to_mcp(server_params)

            # Load available tools from server
            await self._load_tools()

            # Initialize AI model
            self._initialize_model()

            self.initialized = True

            if self.verbose:
                print(f"[SLACK AGENT] Initialization complete. {len(self.available_tools)} tools available.")

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Slack agent: {e}\n"
                "Troubleshooting steps:\n"
                "1. Ensure npx is installed (npm install -g npx)\n"
                "2. Verify SLACK_BOT_TOKEN and SLACK_TEAM_ID are set correctly\n"
                "3. Check that your bot has the necessary OAuth scopes\n"
                "4. Ensure the bot is added to channels it needs to access"
            )

    def _get_credentials(self) -> Tuple[str, str]:
        """Retrieve and validate Slack credentials from environment"""
        slack_bot_token = os.environ.get("SLACK_BOT_TOKEN")
        slack_team_id = os.environ.get("SLACK_TEAM_ID")

        if not slack_bot_token or not slack_team_id:
            raise ValueError(
                "SLACK_BOT_TOKEN and SLACK_TEAM_ID environment variables must be set.\n"
                "To get these:\n"
                "1. Go to https://api.slack.com/apps\n"
                "2. Create or select your app\n"
                "3. Get Bot Token from 'OAuth & Permissions'\n"
                "4. Get Team ID from 'Basic Information'"
            )

        return slack_bot_token, slack_team_id

    def _create_server_params(self, bot_token: str, team_id: str) -> StdioServerParameters:
        """Create MCP server parameters"""
        return StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-slack"],
            env={
                **os.environ,
                "SLACK_BOT_TOKEN": bot_token,
                "SLACK_TEAM_ID": team_id
            }
        )

    async def _connect_to_mcp(self, server_params: StdioServerParameters):
        """Establish connection to MCP server"""
        self.stdio_context = stdio_client(server_params)
        stdio, write = await self.stdio_context.__aenter__()
        self.session = ClientSession(stdio, write)

        await self.session.__aenter__()
        await self.session.initialize()

    async def _load_tools(self):
        """Load available tools from MCP server"""
        tools_list = await self.session.list_tools()
        self.available_tools = tools_list.tools

        if not self.available_tools:
            raise RuntimeError("No tools available from Slack MCP server")

    def _initialize_model(self):
        """Initialize the Gemini AI model with available tools"""
        gemini_tools = [self._build_function_declaration(tool) for tool in self.available_tools]

        self.model = genai.GenerativeModel(
            'models/gemini-2.5-flash',
            system_instruction=self.system_prompt,
            tools=gemini_tools
        )

    # ========================================================================
    # CORE EXECUTION ENGINE
    # ========================================================================

    async def execute(self, instruction: str) -> str:
        """Execute a Slack task with enhanced error handling and retry logic"""
        if not self.initialized:
            return self._format_error(Exception("Slack agent not initialized. Please restart the system."))

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
                print(f"\n[SLACK AGENT] Execution complete. {self.stats.get_summary()}")

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
                    print(f"\n[SLACK AGENT] Calling tool: {tool_name}{retry_info}")
                    if self.verbose:
                        print(f"[SLACK AGENT] Arguments: {json.dumps(tool_args, indent=2)[:500]}")

                tool_result = await self.session.call_tool(tool_name, tool_args)

                result_content = []
                for content in tool_result.content:
                    if hasattr(content, 'text'):
                        result_content.append(content.text)

                result_text = "\n".join(result_content)
                if not result_text:
                    result_text = json.dumps(tool_result.content, default=str)

                if self.verbose:
                    print(f"[SLACK AGENT] Result: {result_text[:500]}")

                self.stats.record_operation(tool_name, True, retry_count)

                return result_text, None

            except Exception as e:
                error_str = str(e).lower()
                is_retryable = any(
                    retryable in error_str
                    for retryable in RetryConfig.RETRYABLE_ERRORS
                )

                if self.verbose or retry_count > 0:
                    print(f"[SLACK AGENT] Error calling {tool_name}: {str(e)}")

                if is_retryable and retry_count < RetryConfig.MAX_RETRIES:
                    retry_count += 1

                    if self.verbose:
                        print(f"[SLACK AGENT] Retrying in {delay:.1f}s...")

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

        if "authentication" in error_lower or "invalid_auth" in error_lower:
            return ErrorType.AUTHENTICATION
        elif "not_in_channel" in error_lower or "missing_scope" in error_lower or "permission" in error_lower:
            return ErrorType.PERMISSION
        elif "channel_not_found" in error_lower or "user_not_found" in error_lower or "not found" in error_lower:
            return ErrorType.NOT_FOUND
        elif "invalid" in error_lower or "validation" in error_lower:
            return ErrorType.VALIDATION
        elif "rate_limited" in error_lower or "rate limit" in error_lower:
            return ErrorType.RATE_LIMIT
        elif "message_too_long" in error_lower:
            return ErrorType.MESSAGE_TOO_LONG
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
                "Your Slack bot token may be invalid or expired. Check SLACK_BOT_TOKEN."
            )

        elif error_type == ErrorType.PERMISSION:
            channel = args.get('channel') or args.get('channel_id')
            if channel:
                return (
                    f"🚫 Cannot access channel '{channel}'. "
                    "The bot may not be a member or may be missing required OAuth scopes."
                )
            return f"🚫 Permission denied for {tool_name}. Check bot permissions and channel membership."

        elif error_type == ErrorType.NOT_FOUND:
            return f"🔍 Resource not found for {tool_name}. Please verify channel names, user IDs, or message timestamps."

        elif error_type == ErrorType.VALIDATION:
            return f"⚠️ Validation error for {tool_name}: {error}"

        elif error_type == ErrorType.RATE_LIMIT:
            return (
                f"⏳ Slack rate limit reached. "
                "The system will automatically retry. If sending many messages, space them out."
            )

        elif error_type == ErrorType.MESSAGE_TOO_LONG:
            return (
                f"📏 Message exceeds Slack's limit (40,000 characters). "
                "Please break it into smaller messages or share as a file."
            )

        elif error_type == ErrorType.NETWORK:
            return f"🌐 Network error when calling {tool_name}: {error}. The system will automatically retry."

        else:
            return f"❌ Error calling {tool_name}: {error}"

    # ========================================================================
    # CAPABILITIES AND INFORMATION
    # ========================================================================

    async def get_capabilities(self) -> List[str]:
        """Return Slack capabilities in user-friendly format"""
        if not self.available_tools:
            return ["Slack operations (initializing...)"]

        capabilities = []
        for tool in self.available_tools:
            description = tool.description or tool.name
            if description:
                capabilities.append(description)

        if len(capabilities) > 10:
            return [
                "✓ Send messages to channels and users",
                "✓ Search messages and conversations",
                "✓ Read channel history and content",
                "✓ Manage reactions and engagement",
                f"✓ ...and {len(capabilities) - 4} more Slack operations"
            ]

        return capabilities

    def get_stats(self) -> str:
        """Get operation statistics summary"""
        return self.stats.get_summary()

    # ========================================================================
    # CLEANUP AND RESOURCE MANAGEMENT
    # ========================================================================

    async def cleanup(self):
        """Disconnect from Slack and clean up resources"""
        try:
            if self.verbose:
                print(f"\n[SLACK AGENT] Cleaning up. {self.stats.get_summary()}")

            if self.session:
                await self.session.__aexit__(None, None, None)
        except Exception as e:
            if self.verbose:
                print(f"[SLACK AGENT] Error closing session: {e}")

        try:
            if self.stdio_context:
                await self.stdio_context.__aexit__(None, None, None)
        except Exception as e:
            if self.verbose:
                print(f"[SLACK AGENT] Error closing stdio context: {e}")

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
