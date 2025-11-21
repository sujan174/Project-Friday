"""
Google Calendar Agent - Production-Ready Connector for Google Calendar

This module provides a robust, intelligent agent for interacting with Google Calendar through
the Model Context Protocol (MCP). It enables seamless calendar management, event scheduling,
and meeting coordination with comprehensive error handling and retry logic.

Key Features:
- Automatic retry with exponential backoff for transient failures
- Intelligent event scheduling and conflict detection
- Comprehensive error handling with context-aware messages
- Operation tracking and statistics
- Verbose logging for debugging
- Smart scheduling with natural language understanding

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
from datetime import datetime, timedelta, timezone
import zoneinfo

import google.generativeai as genai
import google.generativeai.protos as protos
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Add parent directory to path to import base_agent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from connectors.base_agent import BaseAgent, safe_extract_response_text
from config import Config
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
        "504",
        "quotaExceeded",
        "userRateLimitExceeded",
        "backendError"
    ]


class ErrorType(Enum):
    """Classification of error types for better handling"""
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    NOT_FOUND = "not_found"
    VALIDATION = "validation"
    RATE_LIMIT = "rate_limit"
    NETWORK = "network"
    CONFLICT = "conflict"
    QUOTA_EXCEEDED = "quota_exceeded"
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
    Specialized agent for Google Calendar operations via MCP

    This agent provides intelligent, reliable interaction with Google Calendar through:
    - Event creation, modification, and deletion
    - Smart scheduling with conflict detection
    - Calendar querying and availability checking
    - Automatic retry for transient failures
    - Comprehensive error handling and reporting
    - Operation tracking and statistics

    Usage:
        agent = Agent(verbose=True)
        await agent.initialize()
        result = await agent.execute("Schedule a team meeting tomorrow at 2pm")
        await agent.cleanup()
    """

    def __init__(
        self,
        verbose: bool = False,
        shared_context: Optional[SharedContext] = None,
        knowledge_base: Optional[WorkspaceKnowledge] = None,
        session_logger=None
    ):
        """
        Initialize the Google Calendar agent

        Args:
            verbose: Enable detailed logging for debugging (default: False)
            shared_context: Optional shared context for cross-agent coordination
            knowledge_base: Optional workspace knowledge base
            session_logger: Optional session logger for tracking operations
        """
        super().__init__()

        # Session logging
        self.logger = session_logger
        self.agent_name = "google_calendar"

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
        self.proactive = ProactiveAssistant('google_calendar', verbose)

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
        return """You are an elite personal scheduling and time management specialist with deep expertise in calendar coordination, meeting orchestration, and productivity optimization. Your mission is to help users manage their time effectively, schedule meetings seamlessly, avoid conflicts, and maintain optimal work-life balance through intelligent Google Calendar management.

# Your Capabilities

You have complete mastery of Google Calendar's scheduling ecosystem:

**Event Management**:
- Create events with precise timing, titles, descriptions, and locations
- Schedule single events, recurring meetings, and all-day events
- Support multiple time zones and daylight saving transitions
- Add attendees with automatic email invitations
- Set event visibility (public, private, default)
- Configure reminders and notifications
- Add conferencing details (Google Meet, Zoom, etc.)
- Attach files and add rich descriptions with formatting

**Smart Scheduling**:
- Parse natural language time expressions ("tomorrow at 2pm", "next Tuesday", "in 3 hours")
- Detect scheduling conflicts and suggest alternatives
- Find optimal meeting times across multiple attendees
- Check availability and free/busy status
- Propose meeting times based on preferences
- Handle recurring event patterns intelligently
- Account for timezone differences automatically

**Calendar Querying**:
- List events for specific date ranges
- Search events by title, description, attendees, or location
- Find upcoming meetings and deadlines
- Check daily, weekly, or monthly schedules
- Query multiple calendars simultaneously
- Filter by calendar type, visibility, or status

**Event Modification**:
- Update event details (time, title, description, location)
- Add or remove attendees
- Reschedule events with conflict detection
- Cancel events and notify attendees
- Modify recurring event instances or series
- Change event reminders and notifications

**Availability Management**:
- Check free/busy status for specific time ranges
- Find next available time slot
- Block time for focus work or personal time
- Manage working hours and availability preferences
- Coordinate across multiple time zones

**Calendar Organization**:
- Work with multiple calendars (personal, work, shared)
- Manage calendar permissions and sharing
- Color-code events for visual organization
- Set calendar-specific default settings
- Handle calendar subscriptions

# Core Principles

**1. Time is Sacred**: Calendar management is about respecting time. Always:
- **Confirm Before Scheduling**: Verify time, date, duration, and attendees
  - Check for conflicts before creating events
  - Propose alternatives if conflicts exist
  - Respect existing commitments
  - Consider time zones for remote attendees
- **Be Precise with Timing**: Never be ambiguous about when
  - Always include timezone information
  - Clarify AM/PM when needed
  - Account for daylight saving time changes
  - Round to standard increments (30 min, 1 hour) unless specified
- **Minimize Context Switching**: Batch similar meetings, avoid fragmentation
  - Group related meetings together
  - Respect focus time blocks
  - Leave buffer time between back-to-back meetings
- **Protect Personal Time**: Understand work-life boundaries
  - Don't schedule over lunch without permission
  - Respect after-hours and weekend boundaries
  - Honor "do not disturb" time blocks
  - Maintain healthy meeting cadence

**2. Natural Language Understanding**: Parse time expressions intelligently:
- **Relative Times**:
  - "tomorrow" = next calendar day at specified time
  - "next week" = same weekday next week
  - "in 3 days" = current time + 72 hours
  - "end of month" = last day of current month
- **Recurring Patterns**:
  - "every Monday at 10am" = weekly recurring on Mondays
  - "bi-weekly" = every 2 weeks
  - "first Friday of every month" = monthly on 1st Friday
  - "daily standup" = every weekday (Mon-Fri)
- **Duration Inference**:
  - "quick sync" = 15-30 minutes
  - "meeting" = 30-60 minutes (default 30)
  - "workshop" = 2-4 hours
  - "all-day event" = midnight to midnight
- **Timezone Awareness**:
  - "2pm PT" = 2pm Pacific Time (convert to user's timezone)
  - "9am in Tokyo" = handle timezone conversion
  - "EOD" = end of working day in user's timezone
  - Default to user's local timezone if not specified

**3. Conflict Detection & Resolution**:

**Before Creating Events**:
- **Check Existing Calendar**: Always verify no conflicts
  - Query calendar for the proposed time slot
  - Look for overlapping events
  - Consider buffer time needed between meetings
  - Check attendee availability if possible
- **Conflict Resolution Strategies**:
  - **Hard Conflicts**: Events that cannot overlap
    - Propose alternative time slots (before/after, next day)
    - Ask user to choose between conflicting events
    - Suggest rescheduling the less important event
  - **Soft Conflicts**: Events that could potentially overlap
    - Lunch meetings (if lunch block exists but is flexible)
    - Optional events vs. required events
    - Internal meetings vs. external commitments
  - **Buffer Conflicts**: Back-to-back meetings with no transition time
    - Suggest 5-15 minute buffer between meetings
    - Flag if travel time is needed between locations
    - Warn about meeting fatigue from consecutive calls

**Smart Suggestions**:
```
User: "Schedule marketing sync with Sarah"
Agent checks calendar, finds conflicts

CONFLICT DETECTED:
× Proposed: Tuesday 2pm (30 min)
× Existing: Client call Tuesday 1:30-2:30pm

SUGGESTED ALTERNATIVES:
✓ Tuesday 3pm-3:30pm (right after client call)
✓ Wednesday 10am-10:30am (morning slot)
✓ Thursday 2pm-2:30pm (same time, different day)

Which works best for you?
```

**4. Meeting Crafting Excellence**: Every event should be:

**Well-Structured**:
```
Title: Clear and descriptive
Example: "Q1 Planning - Marketing Team" not just "Meeting"

Description:
*Agenda:*
1. Review Q4 performance
2. Set Q1 goals and KPIs
3. Budget allocation discussion
4. Action items and next steps

*Pre-read:* [link to doc]
*Prepared by:* @owner

Location: Conference Room B or Google Meet link

Attendees:
• @sarah (required)
• @john (required)
• @mike (optional)

Reminders: 10 min before
```

**Appropriately Scoped**:
- **1:1s**: 30 minutes, recurring weekly or bi-weekly
- **Team Standups**: 15 minutes daily
- **Planning Sessions**: 60-90 minutes with clear agenda
- **Workshops**: 2-4 hours with breaks built in
- **Social Events**: Clearly marked as optional, outside core hours
- **Focus Time**: Blocks of 2-4 hours for deep work (no meetings)

**Properly Configured**:
- **Working Hours**: Schedule within 9am-6pm local time (unless specified)
- **Recurring Events**: Set proper recurrence pattern and end date
- **Conferencing**: Add video link for remote meetings
- **Reminders**: Default 10 min before, adjust for important events
- **Visibility**: Private for personal, public for team events
- **Time Zone**: Always explicit, especially for international meetings

**5. Attendee Management**:

**Adding Attendees**:
- Use email addresses for external attendees
- Use names for internal team members (resolve to email)
- Mark required vs. optional attendees
- Include attendees' timezones in description for global teams
- Send calendar invitations automatically
- Allow attendees to propose new times

**Communication Best Practices**:
- Include clear meeting purpose in description
- Add preparation materials or pre-reads
- Set expectations for attendance (required/optional)
- Include dial-in or video conferencing details
- Note if meeting will be recorded
- Specify who's leading/facilitating

**6. Recurring Events Intelligence**:

**Creating Recurring Events**:
- **Daily**: Standups, exercise, morning routines
- **Weekly**: 1:1s, team meetings, recurring check-ins
- **Bi-weekly**: Sprint planning, retrospectives
- **Monthly**: All-hands, board meetings, monthly reviews
- **Custom**: First Monday of month, last Friday, etc.

**Managing Recurring Events**:
- Understand "this instance" vs. "all future" vs. "entire series"
- Preserve exceptions when modifying series
- Handle holiday exceptions intelligently
- Suggest review/cleanup of stale recurring events
- Track attendance patterns for optimization

**Pattern Examples**:
```
"Every Monday at 10am"
→ Weekly recurring, Mondays, 10:00am, no end date

"Bi-weekly team sync starting next week"
→ Every 2 weeks, same day of week, start date = next week

"First Friday of each month at 9am for 6 months"
→ Monthly, 1st Friday, 9:00am, ends after 6 occurrences

"Daily standup on weekdays at 9:30am"
→ Daily, Monday-Friday, 9:30am, no end date
```

# Execution Guidelines

## Creating Events

**Event Creation Template**:
```
Title: [Clear, descriptive name]
When: [Start date/time with timezone]
Duration: [Length in hours/minutes]
Location: [Physical location or video link]
Description:
  [Meeting purpose and agenda]
  [Preparation materials]
  [Expected outcomes]
Attendees: [email1@example.com, email2@example.com]
Reminders: [10 minutes before]
Recurring: [if applicable: daily/weekly/monthly pattern]
```

**Natural Language Parsing Examples**:
```
User: "Schedule team sync tomorrow at 2pm"
Parse:
  - Title: Team Sync
  - Date: Tomorrow (current date + 1 day)
  - Time: 2:00 PM (14:00)
  - Duration: 30 minutes (default for "sync")
  - Create event

User: "Book a 1-hour workshop on Friday afternoon"
Parse:
  - Title: Workshop
  - Date: Next Friday
  - Time: 2:00 PM (afternoon default)
  - Duration: 1 hour (specified)
  - Create event

User: "Set up weekly 1:1 with Sarah every Monday at 10am starting next week"
Parse:
  - Title: 1:1 with Sarah
  - Date: Next Monday
  - Time: 10:00 AM
  - Duration: 30 minutes (default 1:1)
  - Recurring: Weekly on Mondays
  - Attendees: sarah@example.com
  - Create event
```

## Querying Events

**Search Strategies**:
```
User: "What meetings do I have tomorrow?"
Query: List events from start of tomorrow to end of tomorrow
Present: Chronological list with times and titles

User: "Find my meeting with John about the project"
Query: Search events containing "John" and "project"
Present: Matching events with dates and details

User: "Am I free Friday afternoon?"
Query: Check events from Friday 12pm-5pm
Present: Free slots or list of existing events

User: "Show me all recurring meetings"
Query: Filter events with recurrence rules
Present: List of recurring events with patterns
```

**Result Presentation**:
```
Your schedule for tomorrow (Tuesday, Jan 16):

09:00 - 09:30  Daily Standup (Google Meet)
10:00 - 11:00  Project Review with Sarah
               Location: Conference Room A
               Agenda: Q4 deliverables review

12:00 - 01:00  Lunch Block (Personal time)

02:00 - 03:30  Client Presentation
               Location: https://zoom.us/j/123456
               ⚠️ Important: Prepare slides

04:00 - 05:00  Team Planning Session
               Attendees: 5 people
               Recurring: Weekly on Tuesdays

Available slots: 11:00am-12:00pm, 1:00pm-2:00pm
```

## Modifying Events

**Update Scenarios**:
```
Reschedule:
  User: "Move my 2pm meeting to 3pm"
  Actions:
    1. Find event at 2pm today
    2. Check for conflicts at 3pm
    3. Update start time to 3pm
    4. Notify attendees of change

Change Title:
  User: "Rename 'Team Meeting' to 'Q1 Planning Session'"
  Actions:
    1. Find event titled "Team Meeting"
    2. Update title
    3. Keep all other details intact

Add Attendee:
  User: "Add mike@company.com to tomorrow's planning session"
  Actions:
    1. Find planning session tomorrow
    2. Add Mike to attendees
    3. Send calendar invitation to Mike
```

## Handling Conflicts

**Conflict Workflow**:
```
1. User requests new event
2. Check calendar for time slot
3. If conflict exists:
   a. Identify conflicting event(s)
   b. Assess conflict severity (hard vs soft)
   c. Propose alternatives
   d. Wait for user decision
4. If no conflict:
   a. Create event
   b. Confirm with details
```

**Example Dialogue**:
```
User: "Schedule code review session tomorrow at 10am"

Agent: "I found a conflict at 10am tomorrow:
  × Existing: Client demo (9:30am - 11:00am)

  Here are some alternative times:
  ✓ Tomorrow 11:30am - 12:30pm
  ✓ Tomorrow 2:00pm - 3:00pm
  ✓ Thursday 10:00am - 11:00am (original time slot)

  Which would you prefer?"

User: "Tomorrow at 2pm works"

Agent: "✓ Created: Code Review Session
  When: Tomorrow 2:00pm - 3:00pm
  [Event details and link]"
```

# Error Handling & Edge Cases

**When You Encounter Issues**:

**Authentication Required**:
```
"Google Calendar authentication needed. The system requires:
 1. Google OAuth 2.0 credentials configured
 2. Calendar API access enabled
 3. Redirect URI configured: http://localhost:4153/oauth2callback

 Please check GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET environment variables.
 A browser window should open for you to authenticate when the agent starts."
```

**Permission Denied**:
```
"You don't have permission to modify this calendar:
 • Calendar may be read-only
 • Shared calendar with restricted access
 • Corporate policy prevents modifications

 Contact the calendar owner or your G Suite admin for access."
```

**Event Not Found**:
```
"Couldn't find that event. This could mean:
 • Event was already deleted
 • Event is on a different calendar
 • You may not have access to view it

 Try searching by title or checking your other calendars."
```

**Time Zone Confusion**:
```
"Detected potential timezone ambiguity:
 • You mentioned '2pm' but didn't specify timezone
 • Your local time: PST (GMT-8)
 • Calendar default: EST (GMT-5)

 I'll use PST unless you specify otherwise. Confirm: 2pm PST?"
```

**Quota Exceeded**:
```
"Google Calendar API quota exceeded:
 • Daily limit: 1,000,000 requests
 • Per-user limit: 100 requests/100 seconds

 Quota resets at midnight Pacific Time.
 The system will automatically retry with exponential backoff."
```

# Output Format

Structure responses clearly:

**For Event Creation**:
```
✓ Created: Team Standup

When: Monday, Jan 15, 2025 @ 9:00 AM PST
Duration: 15 minutes
Location: Google Meet (link: https://meet.google.com/xxx)

Attendees:
• sarah@company.com (organizer)
• john@company.com
• mike@company.com

Recurring: Daily (weekdays)
Reminders: 5 minutes before

Link: https://calendar.google.com/event?eid=xxx
```

**For Event Query**:
```
Found 3 events matching "planning":

1. 📅 Q1 Planning Session
   Tuesday, Jan 16 @ 2:00 PM - 3:30 PM
   Location: Conference Room B
   Attendees: 5 people
   Link: [calendar link]

2. 📅 Sprint Planning
   Thursday, Jan 18 @ 10:00 AM - 12:00 PM
   Location: Google Meet
   Recurring: Bi-weekly
   Link: [calendar link]

3. 📅 Annual Planning Workshop
   Friday, Jan 26 @ 9:00 AM - 4:00 PM
   Location: Off-site (Building C)
   All-day event
   Link: [calendar link]
```

**For Availability Check**:
```
Your availability for Friday, Jan 19:

Free slots:
✓ 9:00 AM - 10:00 AM (1 hour)
✓ 11:30 AM - 12:30 PM (1 hour)
✓ 3:00 PM - 5:00 PM (2 hours)

Busy times:
× 10:00 AM - 11:30 AM: Team Meeting
× 12:30 PM - 1:30 PM: Lunch with Client
× 1:30 PM - 3:00 PM: Project Review

Best times for scheduling:
→ Morning: 9:00 AM - 10:00 AM
→ Afternoon: 3:00 PM - 5:00 PM
```

# Best Practices Summary

1. **Always Check for Conflicts**: Query calendar before creating events
2. **Parse Time Naturally**: Understand "tomorrow", "next week", relative times
3. **Be Timezone Aware**: Always include timezone, especially for remote teams
4. **Provide Context**: Event descriptions should include agenda and purpose
5. **Respect Work-Life Balance**: Don't schedule over personal time without asking
6. **Communicate Changes**: Notify attendees when rescheduling or canceling
7. **Use Proper Recurrence**: Set end dates for recurring meetings when appropriate
8. **Optimize Meeting Length**: Default to 25/50 minutes to allow transition time
9. **Include Conferencing Links**: Add video links for remote meetings automatically
10. **Batch Similar Events**: Group related meetings to minimize context switching

# Special Instructions

**CRITICAL RULES**:
1. **Always check for conflicts** before creating events
2. **ALWAYS include explicit timezone** when creating events - use the timezone from the context (shown in Current Time). NEVER create events without timezone.
3. **Use user's timezone from context** - Look at the Current Time context for the user's configured timezone (e.g., Asia/Kolkata for IST)
4. **When user specifies timezone (IST, PST, etc.)** - Map to proper IANA timezone:
   - IST = Asia/Kolkata (NOT converted to UTC)
   - PST = America/Los_Angeles
   - EST = America/New_York
   - GMT/UTC = UTC
5. **Parse natural language carefully** - "tomorrow" depends on current date
6. **Include meeting context** - titles, agendas, locations, conferencing
7. **Respect working hours** - don't schedule outside 9am-6pm without permission
8. **Handle recurring events precisely** - understand "this instance" vs "all events"
9. **Notify attendees** - send invitations when adding people to events
10. **Be proactive about conflicts** - suggest alternatives, don't just report errors

**Scheduling Philosophy**:
- Protect focus time and deep work blocks
- Minimize meeting fragmentation
- Respect personal boundaries and work-life balance
- Optimize for productivity, not just availability
- Consider energy levels (avoid too many back-to-back meetings)

**Natural Language Excellence**:
- "tomorrow" = next calendar day
- "next week" = same day of following week
- "Monday" (if today is Wed) = next Monday (not this past Monday)
- "end of day" = 5pm or 6pm depending on context
- "morning" = 9am-12pm, "afternoon" = 1pm-5pm, "evening" = 6pm-9pm

Remember: Calendar management is about respecting time - the most finite resource. Every event you create, every meeting you schedule, every conflict you resolve affects someone's productivity and wellbeing. Schedule with precision, communicate with clarity, and optimize for human energy and effectiveness."""

    # ========================================================================
    # INITIALIZATION AND CONNECTION
    # ========================================================================

    async def initialize(self):
        """
        Connect to Google Calendar MCP server

        Raises:
            ValueError: If required environment variables are missing
            RuntimeError: If connection or initialization fails
        """
        try:
            if self.verbose:
                print(f"[GOOGLE CALENDAR AGENT] Initializing connection to Google Calendar MCP server")

            # Google Calendar credentials should be in environment
            # This MCP server uses a credentials JSON file instead of individual env vars
            credentials_path = os.environ.get("GOOGLE_OAUTH_CREDENTIALS", "")

            # Fall back to individual env vars for backwards compatibility
            client_id = os.environ.get("GOOGLE_CLIENT_ID", "")
            client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "")

            # Check if credentials file exists
            if credentials_path and os.path.exists(credentials_path):
                if self.verbose:
                    print(f"[GOOGLE CALENDAR AGENT] Using credentials file: {credentials_path}")
            elif client_id and client_secret:
                # Create a temporary credentials file from env vars
                import tempfile
                import json
                creds_data = {
                    "installed": {
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "redirect_uris": ["http://localhost"],
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token"
                    }
                }
                # Write to a temp file
                temp_creds = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
                json.dump(creds_data, temp_creds)
                temp_creds.close()
                credentials_path = temp_creds.name
                if self.verbose:
                    print(f"[GOOGLE CALENDAR AGENT] Created temporary credentials file from env vars")
            else:
                # Check if credentials look like placeholders
                is_placeholder_id = (not client_id or
                                   "your_google_client_id_here" in client_id or
                                   len(client_id.strip()) < 10)
                is_placeholder_secret = (not client_secret or
                                        "your_google_client_secret_here" in client_secret or
                                        len(client_secret.strip()) < 10)

                if is_placeholder_id or is_placeholder_secret:
                    print(f"[GOOGLE CALENDAR AGENT] ❌ Client ID: {'PLACEHOLDER or EMPTY' if is_placeholder_id else 'SET'}")
                    print(f"[GOOGLE CALENDAR AGENT] ❌ Client Secret: {'PLACEHOLDER or EMPTY' if is_placeholder_secret else 'SET'}")

                raise ValueError(
                    "Google Calendar authentication required. Please set ONE of:\n\n"
                    "Option 1 - Credentials JSON file (recommended):\n"
                    "  - GOOGLE_OAUTH_CREDENTIALS=/path/to/credentials.json\n"
                    "  - Download from Google Cloud Console → Credentials → Download JSON\n\n"
                    "Option 2 - Individual environment variables:\n"
                    "  - GOOGLE_CLIENT_ID\n"
                    "  - GOOGLE_CLIENT_SECRET\n\n"
                    "To obtain credentials:\n"
                    "1. Go to https://console.cloud.google.com\n"
                    "2. Create a new project or select existing\n"
                    "3. Enable Google Calendar API\n"
                    "4. Create OAuth 2.0 credentials (Desktop app)\n"
                    "5. Download the JSON file or copy Client ID/Secret\n\n"
                    "First-time setup: Run 'npx @cocal/google-calendar-mcp auth' to authenticate"
                )

            if self.verbose:
                print(f"[GOOGLE CALENDAR AGENT] Credentials path: {credentials_path}")

            # Prepare environment variables
            env_vars = {
                **os.environ,
                "GOOGLE_OAUTH_CREDENTIALS": credentials_path,
                # Prevent automatic browser opening for OAuth - user will authenticate manually
                "BROWSER": "none"
            }
            if not self.verbose:
                # Suppress debug output from MCP server
                env_vars["DEBUG"] = ""
                env_vars["NODE_ENV"] = "production"

            server_params = StdioServerParameters(
                command="npx",
                args=["-y", "@cocal/google-calendar-mcp"],
                env=env_vars
            )

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

            # Load tools
            tools_list = await self.session.list_tools()
            self.available_tools = tools_list.tools

            if not self.available_tools:
                raise RuntimeError("No tools available from Google Calendar MCP server")

            if self.verbose:
                print(f"[GOOGLE CALENDAR AGENT] Available tools:")
                for tool in self.available_tools:
                    print(f"  - {tool.name}")

            # Convert to Gemini format
            gemini_tools = [self._build_function_declaration(tool) for tool in self.available_tools]

            # Create model
            self.model = genai.GenerativeModel(
                'models/gemini-2.5-flash',
                system_instruction=self.system_prompt,
                tools=gemini_tools
            )

            self.initialized = True

            # Feature #1: Prefetch calendar metadata for faster operations
            await self._prefetch_metadata()

            if self.verbose:
                print(f"[GOOGLE CALENDAR AGENT] Initialization complete. {len(self.available_tools)} tools available.")

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Google Calendar agent: {e}\n"
                "Troubleshooting steps:\n"
                "1. Ensure npx is installed (npm install -g npx)\n"
                "2. Verify Google OAuth credentials are set correctly\n"
                "3. Check that Google Calendar API is enabled in your GCP project\n"
                "4. Ensure refresh token hasn't expired"
            )

    async def _prefetch_metadata(self):
        """
        Prefetch and cache Google Calendar metadata for faster operations

        Fetches calendars list and recent events at initialization time
        to avoid discovery overhead on every operation.

        Cache is persisted to knowledge base with a 30-minute TTL.
        Also caches current time context and free/busy information.
        """
        try:
            # Always update current time context (this is cheap)
            time_context = self._get_current_time_context()

            # Check if we have valid cached metadata
            cached = self.knowledge.get_metadata_cache('google_calendar')
            if cached:
                self.metadata_cache = cached
                # Update time context in memory
                self.metadata_cache['time_context'] = time_context
                if self.verbose:
                    print(f"[GOOGLE CALENDAR AGENT] Loaded metadata from cache")

                # Check for free/busy cache
                free_busy_cache = self.knowledge.get_metadata_cache('google_calendar', sub_key='free_busy')
                if free_busy_cache:
                    self.metadata_cache['free_busy'] = free_busy_cache
                    if self.verbose:
                        print(f"[GOOGLE CALENDAR AGENT] Loaded free/busy cache")
                return

            if self.verbose:
                print(f"[GOOGLE CALENDAR AGENT] Prefetching metadata...")

            # Store in cache with current time context
            self.metadata_cache = {
                'fetched_at': asyncio.get_event_loop().time(),
                'calendars': [],
                'upcoming_events_count': 0,
                'time_context': time_context,
                'free_busy': {}  # Will be populated on demand
            }

            # Persist to knowledge base with 30-minute TTL
            self.knowledge.save_metadata_cache('google_calendar', self.metadata_cache, ttl_seconds=1800)

            if self.verbose:
                print(f"[GOOGLE CALENDAR AGENT] Metadata cache initialized with time context: {time_context['timezone']}")

        except Exception as e:
            # Graceful degradation: If prefetch fails, continue without cache
            if self.verbose:
                print(f"[GOOGLE CALENDAR AGENT] Warning: Metadata prefetch failed: {e}")
            print(f"[GOOGLE CALENDAR AGENT] Continuing without metadata cache")

    def _get_current_time_context(self) -> Dict:
        """
        Get current time, date, and timezone information.

        Returns a dict with:
        - current_time: ISO format datetime
        - current_date: YYYY-MM-DD format
        - timezone: System timezone name
        - timezone_offset: UTC offset string
        - day_of_week: Full day name
        """
        # Get timezone from Config (defaults to Asia/Kolkata for IST)
        try:
            tz_name = Config.USER_TIMEZONE
            tz = zoneinfo.ZoneInfo(tz_name)
        except Exception:
            # Fallback to UTC if configured timezone is invalid
            tz = timezone.utc
            tz_name = 'UTC'

        now = datetime.now(tz)

        # Calculate UTC offset
        offset = now.utcoffset()
        if offset:
            total_seconds = int(offset.total_seconds())
            hours, remainder = divmod(abs(total_seconds), 3600)
            minutes = remainder // 60
            sign = '+' if total_seconds >= 0 else '-'
            offset_str = f"UTC{sign}{hours:02d}:{minutes:02d}"
        else:
            offset_str = "UTC+00:00"

        return {
            'current_time': now.isoformat(),
            'current_date': now.strftime('%Y-%m-%d'),
            'timezone': tz_name,
            'timezone_offset': offset_str,
            'day_of_week': now.strftime('%A'),
            'formatted_time': now.strftime('%I:%M %p'),
            'formatted_date': now.strftime('%B %d, %Y')
        }

    async def cache_free_busy(self, free_busy_data: Dict, days: int = 7):
        """
        Cache free/busy information for faster scheduling.

        Args:
            free_busy_data: Dict with calendars and their busy slots
            days: Number of days of data cached (default 7)
        """
        cache_entry = {
            'data': free_busy_data,
            'days': days,
            'cached_at': asyncio.get_event_loop().time()
        }

        self.metadata_cache['free_busy'] = cache_entry

        # Save with 15-minute TTL (free/busy changes frequently)
        self.knowledge.save_metadata_cache(
            'google_calendar',
            cache_entry,
            ttl_seconds=900,  # 15 minutes
            sub_key='free_busy'
        )

        if self.verbose:
            print(f"[GOOGLE CALENDAR AGENT] Cached free/busy data for {days} days")

    def get_cached_free_busy(self) -> Optional[Dict]:
        """
        Get cached free/busy information if still valid.

        Returns:
            Free/busy data dict or None if not cached/expired
        """
        # Check memory cache first
        cached = self.metadata_cache.get('free_busy', {})
        if cached:
            cached_at = cached.get('cached_at', 0)
            # Check if still valid (15 minutes)
            if asyncio.get_event_loop().time() - cached_at < 900:
                return cached.get('data')

        # Try knowledge base cache
        return self.knowledge.get_metadata_cache('google_calendar', sub_key='free_busy')

    def get_time_context(self) -> Dict:
        """
        Get current time context.

        Returns:
            Dict with current time, date, timezone information
        """
        # Always return fresh time context
        return self._get_current_time_context()

    def _invalidate_cache_after_write(self, operation_type: str):
        """
        Invalidate relevant cache entries after write operations.

        Args:
            operation_type: Type of operation (create_event, update_event, delete_event)
        """
        # Invalidate free/busy cache since schedule has changed
        if operation_type in ['create_event', 'update_event', 'delete_event']:
            self.knowledge.invalidate_metadata_cache('google_calendar', sub_key='free_busy')
            if 'free_busy' in self.metadata_cache:
                del self.metadata_cache['free_busy']

            if self.verbose:
                print(f"[GOOGLE CALENDAR AGENT] Invalidated free/busy cache after {operation_type}")

    def _inject_calendar_context(self, instruction: str) -> str:
        """
        Inject calendar time context and available calendars into the instruction.

        This optimization provides the LLM with current time/date information
        and available calendars, enabling better scheduling decisions.

        Args:
            instruction: Original instruction from user

        Returns:
            Instruction with injected calendar context
        """
        context_parts = []

        # Add time context (very useful for calendar operations)
        time_context = self.metadata_cache.get('time_context', {})
        if time_context:
            tz = time_context.get('timezone', 'Unknown')
            context_parts.append(
                f"Current Time: {time_context.get('formatted_time', 'Unknown')} on "
                f"{time_context.get('day_of_week', 'Unknown')}, "
                f"{time_context.get('formatted_date', 'Unknown')} "
                f"| User Timezone: {tz} - USE THIS TIMEZONE for all events unless user specifies otherwise"
            )

        # Add calendars if available
        calendars = self.metadata_cache.get('calendars', [])
        if calendars:
            cal_names = [cal.get('summary', 'Untitled') for cal in calendars[:5]]
            context_parts.append(f"Available Calendars: {', '.join(cal_names)}")

        if not context_parts:
            return instruction

        # Inject calendar context into instruction
        context = "\n\n[" + " | ".join(context_parts) + "]"

        return instruction + context

    # ========================================================================
    # CORE EXECUTION ENGINE
    # ========================================================================

    async def execute(self, instruction: str) -> str:
        """Execute a Google Calendar task with enhanced error handling and retry logic"""
        if not self.initialized:
            return self._format_error(Exception("Google Calendar agent not initialized. Please restart the system."))

        try:
            # Step 1: Resolve ambiguous references using conversation memory
            resolved_instruction = self._resolve_references(instruction)

            if resolved_instruction != instruction and self.verbose:
                print(f"[GOOGLE CALENDAR AGENT] Resolved instruction: {resolved_instruction}")

            # Step 2: Check for resources from other agents
            context_from_other_agents = self._get_cross_agent_context()
            if context_from_other_agents and self.verbose:
                print(f"[GOOGLE CALENDAR AGENT] Found context from other agents")

            # Step 3: OPTIMIZATION - Inject calendar context (time, calendars)
            # This provides current time context for scheduling operations
            resolved_instruction = self._inject_calendar_context(resolved_instruction)

            if self.verbose:
                print(f"[GOOGLE CALENDAR AGENT] Injected calendar context into instruction")

            # Enhance instruction with cross-agent context if available
            if context_from_other_agents:
                resolved_instruction += f"\n\n[Additional context from other agents: {context_from_other_agents}]"

            # Step 4: Start conversation with LLM
            chat = self.model.start_chat()
            response = await chat.send_message_async(resolved_instruction)

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
                    f"{safe_extract_response_text(response)}\n\n"
                    "⚠ Note: Reached maximum operation limit. The task may be incomplete."
                )

            final_response = safe_extract_response_text(response)

            if self.verbose:
                print(f"\n[GOOGLE CALENDAR AGENT] Execution complete. {self.stats.get_summary()}")

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
        ambiguous_terms = ['it', 'that', 'this', 'the event', 'the meeting', 'the calendar']

        for term in ambiguous_terms:
            if term in instruction.lower():
                reference = self.memory.resolve_reference(term)
                if reference:
                    instruction = instruction.replace(term, reference)
                    instruction = instruction.replace(term.capitalize(), reference)
                    if self.verbose:
                        print(f"[GOOGLE CALENDAR AGENT] Resolved '{term}' → {reference}")
                    break

        return instruction

    def _get_cross_agent_context(self) -> str:
        """Get context from other agents"""
        if not self.shared_context:
            return ""

        # Get only recent resources to avoid overwhelming context
        recent_resources = self.shared_context.get_recent_resources(limit=5)
        if not recent_resources:
            return ""

        context_parts = []
        for resource in recent_resources:
            if resource['agent'] != 'google_calendar':
                context_parts.append(
                    f"{resource['agent'].capitalize()} {resource['type']}: {resource['id']} ({resource['url']})"
                )

        return "; ".join(context_parts) if context_parts else ""

    def _remember_created_resources(self, response: str, instruction: str):
        """Extract and remember created resources from response"""
        import re

        # Pattern to match event IDs or titles in response
        event_pattern = r'Created:\s+([^\n]+)'
        matches = re.findall(event_pattern, response)

        if matches:
            resource_id = matches[0].strip()
            operation_type = self._infer_operation_type(instruction)

            self.memory.remember(
                operation_type,
                resource_id,
                {'instruction': instruction[:100]}
            )

            if self.shared_context:
                self.shared_context.share_resource(
                    'google_calendar',
                    'event',
                    resource_id,
                    f"https://calendar.google.com",
                    {'created_via': instruction[:100]}
                )

    def _infer_operation_type(self, instruction: str) -> str:
        """Infer what type of operation was performed"""
        instruction_lower = instruction.lower()

        if 'create' in instruction_lower or 'schedule' in instruction_lower or 'add' in instruction_lower:
            return 'create_event'
        elif 'update' in instruction_lower or 'modify' in instruction_lower or 'change' in instruction_lower:
            return 'update_event'
        elif 'delete' in instruction_lower or 'cancel' in instruction_lower or 'remove' in instruction_lower:
            return 'delete_event'
        elif 'list' in instruction_lower or 'show' in instruction_lower or 'find' in instruction_lower:
            return 'query_events'
        elif 'free' in instruction_lower or 'available' in instruction_lower or 'busy' in instruction_lower:
            return 'check_availability'
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

        while retry_count <= RetryConfig.MAX_RETRIES:
            try:
                if self.verbose or retry_count > 0:
                    retry_info = f" (retry {retry_count}/{RetryConfig.MAX_RETRIES})" if retry_count > 0 else ""
                    print(f"\n[GOOGLE CALENDAR AGENT] Calling tool: {tool_name}{retry_info}")
                    if self.verbose:
                        print(f"[GOOGLE CALENDAR AGENT] Arguments: {json.dumps(tool_args, indent=2)[:500]}")

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
                    print(f"[GOOGLE CALENDAR AGENT] Result: {result_text[:500]}")

                self.stats.record_operation(tool_name, True, retry_count)

                return result_text, None

            except Exception as e:
                error_str = str(e).lower()
                is_retryable = any(
                    retryable in error_str
                    for retryable in RetryConfig.RETRYABLE_ERRORS
                )

                if self.verbose or retry_count > 0:
                    print(f"[GOOGLE CALENDAR AGENT] Error calling {tool_name}: {str(e)}")

                if is_retryable and retry_count < RetryConfig.MAX_RETRIES:
                    retry_count += 1

                    if self.verbose:
                        print(f"[GOOGLE CALENDAR AGENT] Retrying in {delay:.1f}s...")

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

        if "authentication" in error_lower or "unauthorized" in error_lower or "401" in error:
            return ErrorType.AUTHENTICATION
        elif "not found" in error_lower or "404" in error:
            return ErrorType.NOT_FOUND
        elif "forbidden" in error_lower or "403" in error or "permission" in error_lower:
            return ErrorType.PERMISSION
        elif "validation" in error_lower or "invalid" in error_lower:
            return ErrorType.VALIDATION
        elif "rate limit" in error_lower or "429" in error or "quota" in error_lower:
            return ErrorType.RATE_LIMIT
        elif "conflict" in error_lower or "already exists" in error_lower:
            return ErrorType.CONFLICT
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
                "Your Google OAuth tokens may be invalid or expired. "
                "Check GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, and GOOGLE_REFRESH_TOKEN. "
                "You may need to regenerate the refresh token."
            )

        elif error_type == ErrorType.NOT_FOUND:
            return (
                f"🔍 Event or calendar not found for {tool_name}. "
                "The event may have been deleted, or you may not have access to it. "
                "Verify the event ID or calendar ID is correct."
            )

        elif error_type == ErrorType.PERMISSION:
            return (
                f"🚫 Permission denied for {tool_name}. "
                "You may not have write access to this calendar, or the calendar may be read-only. "
                "Check calendar permissions in Google Calendar settings."
            )

        elif error_type == ErrorType.VALIDATION:
            return (
                f"⚠️ Validation error for {tool_name}. "
                f"The provided data may not match Google Calendar's requirements. Details: {error}"
            )

        elif error_type == ErrorType.RATE_LIMIT:
            return (
                f"⏳ Google Calendar API rate limit or quota exceeded. "
                "Please wait before making more requests. Quotas reset daily. "
                "The system will automatically retry with exponential backoff."
            )

        elif error_type == ErrorType.CONFLICT:
            return (
                f"⚠️ Scheduling conflict detected for {tool_name}. "
                "There may be overlapping events at this time. "
                "Check the calendar and consider alternative time slots."
            )

        elif error_type == ErrorType.NETWORK:
            return f"🌐 Network error when calling {tool_name}: {error}. The system will automatically retry."

        else:
            return f"❌ Error calling {tool_name}: {error}"

    # ========================================================================
    # CAPABILITIES AND INFORMATION
    # ========================================================================

    async def get_capabilities(self) -> List[str]:
        """Return Google Calendar capabilities in user-friendly format"""
        if not self.available_tools:
            return ["Google Calendar operations (initializing...)"]

        # Return curated list with clear capabilities
        return [
            "✓ Create calendar events with dates, times, and attendees",
            "✓ Schedule recurring meetings (daily, weekly, monthly)",
            "✓ List and search calendar events",
            "✓ Update and reschedule existing events",
            "✓ Delete and cancel events",
            "✓ Check availability and free/busy status",
            "✓ Add Google Meet links to events",
            "✓ Set reminders and notifications",
            "✓ Manage multiple calendars",
            "✓ Parse natural language time expressions"
        ]

    async def validate_operation(self, instruction: str) -> Dict[str, Any]:
        """
        Validate if a Google Calendar operation can be performed

        Args:
            instruction: The instruction to validate

        Returns:
            Dict with validation results
        """
        result = {
            'valid': True,
            'missing': [],
            'warnings': [],
            'confidence': 1.0
        }

        instruction_lower = instruction.lower()

        # Check if we're creating an event
        if any(word in instruction_lower for word in ['create', 'schedule', 'add event']):
            # Check for time information
            time_indicators = ['at', 'tomorrow', 'today', 'next', 'on', 'pm', 'am']
            has_time = any(indicator in instruction_lower for indicator in time_indicators)

            if not has_time:
                result['warnings'].append("No time specified - may need clarification")
                result['confidence'] = 0.6

        # Check if agent is initialized
        if not self.initialized:
            result['valid'] = False
            result['missing'].append("agent initialization")
            result['confidence'] = 0.0

        return result

    def get_stats(self) -> str:
        """Get operation statistics summary"""
        return self.stats.get_summary()

    # ========================================================================
    # CLEANUP AND RESOURCE MANAGEMENT
    # ========================================================================

    async def _cleanup_connection(self):
        """Internal cleanup helper for MCP connection resources"""
        # Close session if it was successfully entered
        if self.session and self.session_entered:
            try:
                await self.session.__aexit__(None, None, None)
            except BaseException as e:
                # Suppress all cleanup errors to prevent cascading failures
                # Use BaseException to catch BaseExceptionGroup from anyio
                if self.verbose:
                    print(f"[GOOGLE CALENDAR AGENT] Suppressed session cleanup error: {e}")
            finally:
                self.session = None
                self.session_entered = False

        # Close stdio context if it was successfully entered
        if self.stdio_context and self.stdio_context_entered:
            try:
                await self.stdio_context.__aexit__(None, None, None)
            except BaseException as e:
                # Suppress all cleanup errors (including BaseExceptionGroup from anyio)
                if self.verbose:
                    print(f"[GOOGLE CALENDAR AGENT] Suppressed stdio cleanup error: {e}")
            finally:
                self.stdio_context = None
                self.stdio_context_entered = False

    async def cleanup(self):
        """Disconnect from Google Calendar and clean up resources"""
        if self.verbose:
            print(f"\n[GOOGLE CALENDAR AGENT] Cleaning up. {self.stats.get_summary()}")

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
