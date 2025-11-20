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

    def __init__(
        self,
        verbose: bool = False,
        shared_context: Optional[SharedContext] = None,
        knowledge_base: Optional[WorkspaceKnowledge] = None
    ,
        session_logger=None
    ):
        """
        Initialize the Slack agent

        Args:
            verbose: Enable detailed logging for debugging (default: False)
                    session_logger: Optional session logger for tracking operations
        """
        super().__init__()

        # Session logging
        self.logger = session_logger
        self.agent_name = "slack"

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
        self.proactive = ProactiveAssistant('slack', verbose)

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

            # Feature #1: Prefetch workspace metadata for faster operations
            await self._prefetch_metadata()

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
        try:
            self.stdio_context = stdio_client(server_params)
            stdio, write = await self.stdio_context.__aenter__()
            self.stdio_context_entered = True  # Mark as successfully entered

            self.session = ClientSession(stdio, write)
            await self.session.__aenter__()
            self.session_entered = True  # Mark as successfully entered

            await self.session.initialize()
        except Exception as e:
            # If connection fails, ensure we clean up partial state
            await self._cleanup_connection()
            raise

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

    async def _prefetch_metadata(self):
        """
        Prefetch and cache Slack workspace metadata for faster operations (Feature #1)

        Fetches channels, users, and workspace info at initialization time
        to avoid discovery overhead on every operation.

        Cache is persisted to knowledge base with a 1-hour TTL.
        """
        try:
            # Check if we have valid cached metadata
            cached = self.knowledge.get_metadata_cache('slack')
            if cached:
                self.metadata_cache = cached
                if self.verbose:
                    print(f"[SLACK AGENT] Loaded metadata from cache ({len(cached.get('channels', {}))} channels, {len(cached.get('users', {}))} users)")
                return

            if self.verbose:
                print(f"[SLACK AGENT] Prefetching metadata...")

            # Fetch all channels (public + user's private)
            channels = await self._fetch_all_channels()
            if self.verbose:
                print(f"[SLACK AGENT] Prefetched {len(channels)} channels")

            # Fetch all users (limit to active users to avoid huge lists)
            users = await self._fetch_all_users()
            if self.verbose:
                print(f"[SLACK AGENT] Prefetched {len(users)} users")

            # Store in cache
            self.metadata_cache = {
                'channels': channels,
                'users': users,
                'fetched_at': asyncio.get_event_loop().time()
            }

            # Persist to knowledge base
            self.knowledge.save_metadata_cache('slack', self.metadata_cache, ttl_seconds=3600)

            if self.verbose:
                print(f"[SLACK AGENT] Cached metadata: {len(channels)} channels, {len(users)} users")

        except Exception as e:
            # Graceful degradation: If prefetch fails, continue without cache
            if self.verbose:
                print(f"[SLACK AGENT] Warning: Metadata prefetch failed: {e}")
            print(f"[SLACK AGENT] Continuing without metadata cache (operations may be slower)")

    async def _fetch_all_channels(self) -> Dict:
        """Fetch all accessible channels"""
        try:
            # Use Slack MCP tool to list channels
            result = await self.session.call_tool("slack_list_channels", {})

            channels = {}
            if hasattr(result, 'content') and result.content:
                content = result.content[0].text if result.content else "{}"
                data = json.loads(content) if isinstance(content, str) else content

                if isinstance(data, dict) and 'channels' in data:
                    for ch in data['channels']:
                        channel_id = ch.get('id', '')
                        channels[channel_id] = {
                            'id': channel_id,
                            'name': ch.get('name', ''),
                            'is_private': ch.get('is_private', False),
                            'is_archived': ch.get('is_archived', False),
                            'num_members': ch.get('num_members', 0)
                        }
                elif isinstance(data, list):
                    for ch in data:
                        channel_id = ch.get('id', '')
                        channels[channel_id] = {
                            'id': channel_id,
                            'name': ch.get('name', ''),
                            'is_private': ch.get('is_private', False),
                            'is_archived': ch.get('is_archived', False),
                            'num_members': ch.get('num_members', 0)
                        }

            return channels
        except Exception as e:
            if self.verbose:
                print(f"[SLACK AGENT] Could not fetch channels: {e}")
            return {}

    async def _fetch_all_users(self) -> Dict:
        """Fetch all users (limit to active to avoid huge lists)"""
        try:
            # Use Slack MCP tool to list users
            result = await self.session.call_tool("slack_list_users", {})

            users = {}
            if hasattr(result, 'content') and result.content:
                content = result.content[0].text if result.content else "{}"
                data = json.loads(content) if isinstance(content, str) else content

                if isinstance(data, dict) and 'members' in data:
                    user_list = data['members']
                elif isinstance(data, list):
                    user_list = data
                else:
                    user_list = []

                # Limit to first 100 active users to avoid huge cache
                for user in user_list[:100]:
                    if not user.get('deleted', False) and not user.get('is_bot', False):
                        user_id = user.get('id', '')
                        users[user_id] = {
                            'id': user_id,
                            'name': user.get('name', ''),
                            'real_name': user.get('real_name', ''),
                            'display_name': user.get('profile', {}).get('display_name', '')
                        }

            return users
        except Exception as e:
            if self.verbose:
                print(f"[SLACK AGENT] Could not fetch users: {e}")
            return {}

    def get_cached_channels(self) -> Dict[str, Any]:
        """
        Get all prefetched channels from cache.

        Returns:
            Dict mapping channel ID to channel info (id, name, is_private, num_members)
        """
        return self.metadata_cache.get('channels', {})

    def get_cached_channel_by_name(self, channel_name: str) -> Optional[Dict[str, Any]]:
        """
        Look up a channel by name from cache (fast lookup).

        Args:
            channel_name: Name of the channel (with or without #)

        Returns:
            Channel info dict or None if not found
        """
        # Normalize channel name (remove # if present)
        name = channel_name.lstrip('#').lower()

        # Search in cached channels
        for channel_id, channel_info in self.get_cached_channels().items():
            if channel_info['name'].lower() == name:
                return channel_info

        return None

    def list_available_channels(self) -> List[str]:
        """
        Get list of all available channel names for user reference.

        Returns:
            List of channel names
        """
        channels = self.get_cached_channels()
        return sorted([ch['name'] for ch in channels.values() if not ch.get('is_archived', False)])

    def _get_cached_channels_response(self) -> Optional[str]:
        """
        Get cached channels formatted as JSON response (cache-first optimization).

        This method returns the cached channels in the same format that the
        slack_list_channels API would return, allowing us to avoid API calls
        for frequently used channel listings.

        Returns:
            JSON string with channels data, or None if cache is empty
        """
        cached_channels = self.get_cached_channels()

        # If cache is empty, return None to fall back to API call
        if not cached_channels:
            return None

        # Convert cached channels to the format expected by the LLM
        channels_list = []
        for channel_id, channel_info in cached_channels.items():
            channels_list.append({
                'id': channel_info['id'],
                'name': channel_info['name'],
                'is_private': channel_info.get('is_private', False),
                'is_archived': channel_info.get('is_archived', False),
                'num_members': channel_info.get('num_members', 0)
            })

        # Format as JSON response
        response_data = {
            'channels': channels_list,
            'cached': True,  # Mark that this came from cache
            'timestamp': time.time()
        }

        return json.dumps(response_data)

    def _get_cached_users_response(self) -> Optional[str]:
        """
        Get cached users formatted as JSON response (cache-first optimization).

        This method returns the cached users in the same format that the
        slack_list_users API would return, allowing us to avoid API calls
        for frequently used user listings.

        Returns:
            JSON string with users data, or None if cache is empty
        """
        cached_users = self.metadata_cache.get('users', {})

        # If cache is empty, return None to fall back to API call
        if not cached_users:
            return None

        # Convert cached users to the format expected by the LLM
        users_list = []
        for user_id, user_info in cached_users.items():
            users_list.append({
                'id': user_info.get('id', user_id),
                'name': user_info.get('name', ''),
                'real_name': user_info.get('real_name', ''),
                'is_bot': user_info.get('is_bot', False),
                'is_active': user_info.get('is_active', True)
            })

        # Format as JSON response
        response_data = {
            'members': users_list,
            'cached': True,  # Mark that this came from cache
            'timestamp': time.time()
        }

        return json.dumps(response_data)

    def _get_cached_channel_info_response(self, channel_name: str) -> Optional[str]:
        """
        Get cached channel info formatted as JSON response (cache-first optimization).

        This method returns the cached channel info in the same format that the
        slack_get_channel_info API would return.

        Args:
            channel_name: Name of the channel (with or without #)

        Returns:
            JSON string with channel info, or None if not cached
        """
        # Clean channel name
        clean_name = channel_name.lstrip('#').lower()

        # Try to find channel by name
        cached_channel = self.get_cached_channel_by_name(clean_name)

        if not cached_channel:
            return None

        # Format as JSON response
        response_data = {
            'channel': {
                'id': cached_channel.get('id', ''),
                'name': cached_channel.get('name', ''),
                'is_private': cached_channel.get('is_private', False),
                'is_archived': cached_channel.get('is_archived', False),
                'num_members': cached_channel.get('num_members', 0),
                'topic': cached_channel.get('topic', ''),
                'purpose': cached_channel.get('purpose', '')
            },
            'cached': True,
            'timestamp': time.time()
        }

        return json.dumps(response_data)

    def _invalidate_cache_after_write(self, operation_type: str, channel_name: str = None):
        """
        Invalidate relevant cache entries after write operations.

        Args:
            operation_type: Type of operation (create_channel, archive_channel, etc.)
            channel_name: Optional channel name for targeted invalidation
        """
        # Invalidate channels cache for channel-modifying operations
        if operation_type in ['create_channel', 'archive_channel', 'unarchive_channel', 'rename_channel']:
            self.knowledge.invalidate_metadata_cache('slack')
            if self.verbose:
                print(f"[SLACK AGENT] Invalidated channels cache after {operation_type}")

    # ========================================================================
    # CORE EXECUTION ENGINE
    # ========================================================================

    async def execute(self, instruction: str) -> str:
        """Execute a Slack task with enhanced error handling and retry logic"""
        if not self.initialized:
            return self._format_error(Exception("Slack agent not initialized. Please restart the system."))

        try:
            # Step 1: Resolve ambiguous references using conversation memory
            resolved_instruction = self._resolve_references(instruction)

            if resolved_instruction != instruction and self.verbose:
                print(f"[SLACK AGENT] Resolved instruction: {resolved_instruction}")

            # Step 2: Check for resources from other agents
            context_from_other_agents = self._get_cross_agent_context()
            if context_from_other_agents and self.verbose:
                print(f"[SLACK AGENT] Found context from other agents")

            # Step 3: OPTIMIZATION - Inject available channels into instruction
            # This eliminates unnecessary slack_list_channels tool calls and prevents
            # false positives where the LLM claims to complete actions without calling tools
            instruction = self._inject_channel_context(resolved_instruction)

            if instruction != resolved_instruction and self.verbose:
                print(f"[SLACK AGENT] Injected channel context into instruction")

            # Use instruction with injected context for the rest
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
                response_text = safe_extract_response_text(response)
                return (
                    f"{response_text}\n\n"
                    "⚠ Note: Reached maximum operation limit. The task may be incomplete."
                )

            final_response = safe_extract_response_text(response)

            if self.verbose:
                print(f"\n[SLACK AGENT] Execution complete. {self.stats.get_summary()}")

            
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
        ambiguous_terms = ['it', 'that', 'this', 'the channel', 'the message', 'the thread']

        for term in ambiguous_terms:
            if term in instruction.lower():
                reference = self.memory.resolve_reference(term)
                if reference:
                    instruction = instruction.replace(term, reference)
                    instruction = instruction.replace(term.capitalize(), reference)
                    if self.verbose:
                        print(f"[SLACK AGENT] Resolved '{term}' → {reference}")
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
            if resource['agent'] != 'slack':
                context_parts.append(
                    f"{resource['agent'].capitalize()} {resource['type']}: {resource['id']} ({resource['url']})"
                )

        return "; ".join(context_parts) if context_parts else ""

    def _inject_channel_context(self, instruction: str) -> str:
        """
        Inject available Slack channels directly into the instruction.

        This optimization prevents unnecessary slack_list_channels tool calls and
        false positives where the LLM claims to complete actions without actually
        calling the required tools. By providing channel information upfront, the LLM
        can make better decisions about which channels exist and proceed directly to
        action execution.

        Args:
            instruction: Original instruction from user

        Returns:
            Instruction with injected channel context
        """
        available_channels = self.list_available_channels()

        # Only inject if we have cached channels
        if not available_channels:
            return instruction

        # Build channel list for injection
        channel_list = ", ".join(f"#{ch}" for ch in available_channels)

        # Inject channel context into instruction
        context = (
            f"\n\n[Available Slack Channels: {channel_list}]\n"
            f"[Note: Use these channels directly - no need to list channels first]"
        )

        return instruction + context

    def _remember_created_resources(self, response: str, instruction: str):
        """Extract and remember created resources from response"""
        import re

        # Pattern to match Slack channel names or message timestamps
        channel_pattern = r'#([a-z0-9-]+)'
        matches = re.findall(channel_pattern, response)

        if matches:
            resource_id = f"#{matches[-1]}"
            operation_type = self._infer_operation_type(instruction)

            self.memory.remember(
                operation_type,
                resource_id,
                {'instruction': instruction[:100]}
            )

            if self.shared_context:
                self.shared_context.share_resource(
                    'slack',
                    'message',
                    resource_id,
                    f"https://slack.com/app_redirect?channel={matches[-1]}",
                    {'created_via': instruction[:100]}
                )

    def _infer_operation_type(self, instruction: str) -> str:
        """Infer what type of operation was performed"""
        instruction_lower = instruction.lower()

        if 'post' in instruction_lower or 'send' in instruction_lower or 'message' in instruction_lower:
            return 'send_message'
        elif 'reply' in instruction_lower or 'thread' in instruction_lower:
            return 'reply_thread'
        elif 'react' in instruction_lower or 'emoji' in instruction_lower:
            return 'add_reaction'
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
        """Execute a tool with automatic retry on transient failures"""
        retry_count = 0
        delay = RetryConfig.INITIAL_DELAY

        # OPTIMIZATION: Check cache first to avoid unnecessary API calls
        if tool_name == "slack_list_channels":
            cached_result = self._get_cached_channels_response()
            if cached_result is not None:
                if self.verbose:
                    print(f"\n[SLACK AGENT] Using cached channels instead of API call")
                self.stats.record_operation(tool_name, True, 0)
                return cached_result, None

        # OPTIMIZATION: Cache-first for slack_list_users
        if tool_name == "slack_list_users":
            cached_result = self._get_cached_users_response()
            if cached_result is not None:
                if self.verbose:
                    print(f"\n[SLACK AGENT] Using cached users instead of API call")
                self.stats.record_operation(tool_name, True, 0)
                return cached_result, None

        # OPTIMIZATION: Cache-first for slack_get_channel_info
        if tool_name == "slack_get_channel_info":
            channel_name = tool_args.get('channel', '') or tool_args.get('channel_id', '')
            if channel_name:
                cached_result = self._get_cached_channel_info_response(channel_name)
                if cached_result is not None:
                    if self.verbose:
                        print(f"\n[SLACK AGENT] Using cached channel info for {channel_name} instead of API call")
                    self.stats.record_operation(tool_name, True, 0)
                    return cached_result, None

        while retry_count <= RetryConfig.MAX_RETRIES:
            try:
                if self.verbose or retry_count > 0:
                    retry_info = f" (retry {retry_count}/{RetryConfig.MAX_RETRIES})" if retry_count > 0 else ""
                    print(f"\n[SLACK AGENT] Calling tool: {tool_name}{retry_info}")
                    if self.verbose:
                        print(f"[SLACK AGENT] Arguments: {json.dumps(tool_args, indent=2)[:500]}")

                # Log tool call start


                start_time = time.time()



                tool_result = await self.session.call_tool(tool_name, tool_args)



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
        """Return Slack capabilities in user-friendly format with limitations"""
        if not self.available_tools:
            return ["Slack operations (initializing...)"]

        # Return curated list with clear capabilities and limitations
        return [
            "✓ Send messages to channels and direct messages",
            "✓ Search messages and conversations",
            "✓ Read channel history and content",
            "✓ Manage reactions and thread replies",
            "✓ List channels and get user information",
            "✗ Cannot: Delete messages (only admins)",
            "✗ Cannot: Manage channel settings (members, permissions, etc.)",
            "✗ Cannot: Create or delete channels (limited to admin workspace)",
            "✗ Cannot: Access private channels without membership",
        ]

    async def get_action_schema(self) -> Dict[str, Any]:
        """
        Return schema describing editable parameters for Slack actions.

        This enables rich interactive editing of Slack messages before sending.

        Returns:
            Dict mapping action types to their parameter schemas
        """
        # Get list of available channels for validation
        channel_names = []
        if self.metadata_cache.get('channels'):
            channel_names = [f"#{ch['name']}" for ch in self.metadata_cache['channels'].values()
                           if not ch.get('is_archived', False)]

        return {
            'send': {
                'parameters': {
                    'message': {
                        'display_label': 'Message',
                        'description': 'The message content to send',
                        'type': 'text',  # Multi-line text
                        'editable': True,
                        'required': True,
                        'constraints': {
                            'min_length': 1,
                            'max_length': 4000,  # Slack message limit
                        },
                        'examples': [
                            'Team meeting at 3pm today',
                            'Deployment completed successfully ✓',
                            'Bug fix is ready for review: https://github.com/...'
                        ]
                    },
                    'channel': {
                        'display_label': 'Channel',
                        'description': 'The channel or user to send the message to',
                        'type': 'string',
                        'editable': True,
                        'required': True,
                        'constraints': {
                            'allowed_values': channel_names if channel_names else None,
                        },
                        'examples': ['#engineering', '#general', '@john']
                    },
                    'thread_ts': {
                        'display_label': 'Thread Timestamp',
                        'description': 'Reply in thread (timestamp of parent message)',
                        'type': 'string',
                        'editable': False,  # Thread context is usually determined by context
                        'required': False
                    }
                }
            },
            'notify': {
                'parameters': {
                    'message': {
                        'display_label': 'Notification Message',
                        'description': 'The notification message to broadcast',
                        'type': 'text',
                        'editable': True,
                        'required': True,
                        'constraints': {
                            'min_length': 1,
                            'max_length': 4000,
                        }
                    },
                    'channel': {
                        'display_label': 'Channel',
                        'description': 'Channel to notify',
                        'type': 'string',
                        'editable': True,
                        'required': True,
                        'constraints': {
                            'allowed_values': channel_names if channel_names else None,
                        }
                    },
                    'mention_type': {
                        'display_label': 'Mention Type',
                        'description': 'Who to mention (@channel, @here, or none)',
                        'type': 'string',
                        'editable': True,
                        'required': False,
                        'constraints': {
                            'allowed_values': ['@channel', '@here', 'none']
                        }
                    }
                }
            }
        }

    async def validate_operation(self, instruction: str) -> Dict[str, Any]:
        """
        Validate if a Slack operation can be performed (Feature #14)

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

        # Check if we're sending a message to a channel
        if any(word in instruction_lower for word in ['send', 'post', 'message']) and 'channel' in instruction_lower:
            # Check if we have channel metadata
            if not self.metadata_cache.get('channels'):
                result['warnings'].append("No channel metadata cached - operation may be slow")
                result['confidence'] = 0.7

            # Try to extract channel name from instruction
            channels = self.metadata_cache.get('channels', {})
            if channels:
                # Check if instruction mentions a known channel
                mentioned_channel = None
                for channel_data in channels.values():
                    channel_name = channel_data.get('name', '')
                    if f"#{channel_name}" in instruction_lower or channel_name in instruction_lower:
                        mentioned_channel = channel_name
                        break

                if not mentioned_channel:
                    result['missing'].append("channel name")
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
        Apply user's edited parameters back into the instruction for Slack actions.

        Args:
            instruction: Original instruction from LLM
            parameter_edits: {field_name: new_value} from user

        Returns:
            Modified instruction with edits applied

        Example:
            Original: "Send message 'Hello team' to #engineering"
            Edits: {'message': 'Hello team! Deployment is complete.', 'channel': '#general'}
            Return: "Send message 'Hello team! Deployment is complete.' to #general"
        """
        import re

        modified = instruction

        # Apply message edits
        if 'message' in parameter_edits:
            new_message = parameter_edits['message']
            # Find message in quotes and replace
            patterns = [
                (r"message\s+['\"]([^'\"]+)['\"]", f"message '{new_message}'"),
                (r"['\"]([^'\"]{10,})['\"]", f"'{new_message}'"),  # Long quoted strings are likely messages
            ]

            for pattern, replacement in patterns:
                if re.search(pattern, modified, re.IGNORECASE):
                    modified = re.sub(pattern, replacement, modified, count=1, flags=re.IGNORECASE)
                    break

        # Apply channel edits
        if 'channel' in parameter_edits:
            new_channel = parameter_edits['channel']
            # Ensure channel has # prefix
            if not new_channel.startswith('#') and not new_channel.startswith('@'):
                new_channel = f"#{new_channel}"

            patterns = [
                (r"to\s+[#@]?\w+", f"to {new_channel}"),
                (r"channel\s+[#@]?\w+", f"channel {new_channel}"),
                (r"in\s+[#@]?\w+", f"in {new_channel}"),
            ]

            for pattern, replacement in patterns:
                if re.search(pattern, modified, re.IGNORECASE):
                    modified = re.sub(pattern, replacement, modified, count=1, flags=re.IGNORECASE)
                    break

        # Apply mention_type edits
        if 'mention_type' in parameter_edits:
            mention = parameter_edits['mention_type']
            if mention and mention != 'none':
                # Add mention to message if not already present
                if mention not in modified:
                    modified = modified.replace("'", f"' with {mention}", 1)

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
                    print(f"[SLACK AGENT] Suppressed session cleanup error: {e}")
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
                    print(f"[SLACK AGENT] Suppressed stdio cleanup error: {e}")
            finally:
                self.stdio_context = None
                self.stdio_context_entered = False

    async def cleanup(self):
        """Disconnect from Slack and clean up resources"""
        if self.verbose:
            print(f"\n[SLACK AGENT] Cleaning up. {self.stats.get_summary()}")

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

    def _deep_convert_proto_args(self, value: Any) -> Any:
        """Recursively convert protobuf types to standard Python types"""
        type_str = str(type(value))

        if "MapComposite" in type_str:
            return {k: self._deep_convert_proto_args(v) for k, v in value.items()}
        elif "RepeatedComposite" in type_str:
            return [self._deep_convert_proto_args(item) for item in value]
        else:
            return value
