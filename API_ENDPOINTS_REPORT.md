# PROJECT-FRIDAY: COMPREHENSIVE API ENDPOINTS & ROUTES REPORT

**Report Generated:** 2025-11-20
**Project Type:** AI Workspace Orchestrator (Multi-Agent System)
**Version:** 3.0
**Architecture:** Event-Driven Agent-Based Architecture with MCP (Model Context Protocol)

---

## EXECUTIVE SUMMARY

Project-Friday is NOT a traditional REST API or web service. Instead, it implements a **multi-agent orchestration system** where:

- **Primary Interface:** Async Python CLI (main.py) with terminal UI
- **Communication Protocol:** Model Context Protocol (MCP) for agent-to-service communication
- **Agent Routing:** LLM-based (Gemini 2.5 Flash) with intelligent function calling
- **Architecture:** Hub-and-spoke with orchestrator coordinating specialized agents

The system exposes **8 major specialized agents** as "endpoints" in terms of functional capabilities, each with its own toolset that communicates with external APIs.

---

## SECTION 1: AGENT ENDPOINTS (PRIMARY "ROUTES")

### 1.1 AGENT DISCOVERY MECHANISM

The orchestrator dynamically discovers and registers agents through:
```
Location: /home/user/Project-Friday/orchestrator.py (Lines: 400-600)
Method: async def discover_and_load_agents()
```

**Agent Registration Pattern:**
```python
# Each agent must implement:
class BaseAgent(ABC):
    async def initialize()      # Connect to services
    async def get_capabilities()  # Advertise capabilities
    async def execute(instruction) # Execute tasks
    async def cleanup()          # Release resources
```

---

## SECTION 2: COMPLETE AGENT ENDPOINT CATALOG

### 2.1 SLACK AGENT
**File:** `/home/user/Project-Friday/connectors/slack_agent.py`

**Capabilities (get_capabilities - Lines 1434-1450):**
```
✓ Send messages to channels and direct messages
✓ Search messages and conversations
✓ Read channel history and content
✓ Manage reactions and thread replies
✓ List channels and get user information
✗ Cannot: Delete messages (only admins)
✗ Cannot: Manage channel settings
✗ Cannot: Create or delete channels
✗ Cannot: Access private channels without membership
```

**Execution Method:**
```
Entry Point: async def execute(instruction: str) -> str
Location: Lines 1023-1430
Protocol: MCP (Model Context Protocol)
```

**MCP Tools Available (via ClientSession):**
- slack_send_message
- slack_search_messages
- slack_read_channel
- slack_list_channels
- slack_get_user_info
- slack_add_reaction
- slack_post_thread_reply

**Authentication:**
```
Method: OAuth 2.0 Token
Environment Variable: SLACK_BOT_TOKEN
Source: https://api.slack.com/apps
Configuration: .env.example (lines 43-50)
```

**Request Format Example:**
```python
instruction = "Send message 'Hello team' to #general channel"
result = await slack_agent.execute(instruction)
```

**Response Format:**
```
Natural language confirmation:
"✓ Sent message to #general channel"
or
"⚠️ Error: Cannot access private channel #hr-only"
```

---

### 2.2 JIRA AGENT
**File:** `/home/user/Project-Friday/connectors/jira_agent.py`

**Capabilities (get_capabilities - Lines 1227-1249):**
```
✓ Create and manage Jira issues
✓ Search issues using JQL (Jira Query Language)
✓ Update issue fields (status, assignee, labels, custom fields)
✓ Add comments and collaborate on issues
✓ Manage workflows and transitions
✓ Create sub-tasks and link issues
✗ Cannot: Delete issues (archived only)
✗ Cannot: Manage project settings or permissions
✗ Cannot: Create or delete projects (admin only)
✗ Cannot: Modify issue history or audit logs
```

**Execution Method:**
```
Entry Point: async def execute(instruction: str) -> str
Location: Lines 713-1227
Protocol: MCP (Model Context Protocol)
```

**MCP Tools Available:**
- jira_get_projects
- jira_get_issue_types
- jira_create_issue
- jira_search_issues
- jira_update_issue
- jira_add_comment
- jira_transition_issue
- jira_assign_issue

**Authentication:**
```
Method: Basic Auth (Email + API Token)
Environment Variables:
  - JIRA_URL: Full Jira instance URL (e.g., https://mycompany.atlassian.net)
  - JIRA_USERNAME: Atlassian account email
  - JIRA_API_TOKEN: API token from https://id.atlassian.com/manage-profile/security/api-tokens

MCP Server Parameters: Lines 505-535
```

**Request Format Example:**
```python
instruction = "Create a bug ticket for login timeout issue"
result = await jira_agent.execute(instruction)
```

**Response Format:**
```
"✓ Created issue KAN-456: Login page timeout issue
Link: https://your-domain.atlassian.net/browse/KAN-456"
```

**Action Schema (for pre-execution editing):**
- create: summary, description, project, issue_type, priority, assignee
- update: issue_key, fields (status, assignee, labels, etc.)

---

### 2.3 GITHUB AGENT
**File:** `/home/user/Project-Friday/connectors/github_agent.py`

**Capabilities (get_capabilities - Lines 1502-1519):**
```
✓ Manage issues and pull requests
✓ Perform code reviews and analyze PRs
✓ Search code and commits
✓ Read repository files and structure
✓ Work with branches and commits
✓ Get commit metadata (SHA, author, message, date)
✗ Cannot fetch: Raw diff/patch content
✗ Cannot: Modify repository settings or access controls
✗ Cannot: Create or delete repositories
✗ Cannot: Manage webhooks or integrations
```

**Execution Method:**
```
Entry Point: async def execute(instruction: str) -> str
Location: Lines 1135-1502
Protocol: MCP (Model Context Protocol)
```

**MCP Tools Available:**
- github_create_issue
- github_create_pr
- github_search_code
- github_search_commits
- github_read_file
- github_list_branches
- github_get_commit_info
- github_review_pr

**Authentication:**
```
Method: Personal Access Token (PAT)
Environment Variable: GITHUB_TOKEN
Source: https://github.com/settings/tokens
Scopes Required:
  - repo (full control of private repositories)
  - read:org (read org data)
```

**Request Format Example:**
```python
instruction = "Create an issue in owner/repo for feature request: Add dark mode"
result = await github_agent.execute(instruction)
```

**Response Format:**
```
"✓ Created issue #247 in owner/repo: Add dark mode feature
Link: https://github.com/owner/repo/issues/247"
```

---

### 2.4 NOTION AGENT
**File:** `/home/user/Project-Friday/connectors/notion_agent.py`

**Capabilities (get_capabilities - Lines 1199-1215):**
```
✓ Create and manage Notion pages
✓ Work with databases and database entries
✓ Add and format content blocks (text, headings, lists, code, etc.)
✓ Search across workspace and filter database entries
✓ Update page and database properties
✗ Cannot: Delete pages (archive only)
✗ Cannot: Modify database schemas or properties
✗ Cannot: Manage workspace or user permissions
✗ Cannot: Access pages without explicit sharing
```

**Execution Method:**
```
Entry Point: async def execute(instruction: str) -> str
Location: Lines 836-1199
Protocol: MCP (Model Context Protocol)
```

**MCP Tools Available:**
- notion_create_page
- notion_create_database_entry
- notion_search_pages
- notion_update_page
- notion_add_blocks
- notion_filter_database

**Authentication:**
```
Method: Internal Integration Token
Environment Variable: NOTION_TOKEN
Source: https://www.notion.so/profile/integrations
Token Format: Starts with "secret_"
Setup:
  1. Create new internal integration
  2. Grant required capabilities
  3. Share pages/databases with integration
```

**Request Format Example:**
```python
instruction = "Create a Notion page titled 'Sprint Planning Q4 2024' with meeting agenda"
result = await notion_agent.execute(instruction)
```

**Response Format:**
```
"✓ Created Notion page: Sprint Planning Q4 2024
Link: https://notion.so/page_id_here"
```

---

### 2.5 GOOGLE CALENDAR AGENT
**File:** `/home/user/Project-Friday/connectors/google_calendar_agent.py`

**Capabilities (get_capabilities - Lines 1306-1323):**
```
✓ Create calendar events with dates, times, and attendees
✓ Schedule recurring meetings (daily, weekly, monthly)
✓ List and search calendar events
✓ Update and reschedule existing events
✓ Delete and cancel events
✓ Check availability and free/busy status
✓ Add Google Meet links to events
✓ Set reminders and notifications
✓ Manage multiple calendars
✓ Parse natural language time expressions
```

**Execution Method:**
```
Entry Point: async def execute(instruction: str) -> str
Location: Lines 948-1306
Protocol: MCP (Model Context Protocol)
```

**MCP Tools Available:**
- calendar_create_event
- calendar_list_events
- calendar_update_event
- calendar_delete_event
- calendar_check_availability
- calendar_add_attendees

**Authentication:**
```
Method: OAuth 2.0 (Desktop App Flow)
Environment Variables:
  - GOOGLE_CLIENT_ID: OAuth 2.0 Client ID
  - GOOGLE_CLIENT_SECRET: OAuth 2.0 Client Secret
  - GOOGLE_REDIRECT_URI: (optional) Custom redirect URI, default: http://localhost:4153/oauth2callback

Setup:
  1. Create project in https://console.cloud.google.com
  2. Enable Google Calendar API
  3. Create OAuth 2.0 Desktop credentials
  4. Configure redirect URI: http://localhost:4153/oauth2callback
  5. Add required scopes:
     - https://www.googleapis.com/auth/calendar
```

**Request Format Example:**
```python
instruction = "Schedule a meeting with John tomorrow at 2pm for 1 hour to discuss Q4 roadmap"
result = await google_calendar_agent.execute(instruction)
```

**Response Format:**
```
"✓ Created event: Q4 Roadmap Discussion (tomorrow, 2:00 PM)
Location: Google Meet (link: https://meet.google.com/xxx)
Attendees: john@company.com
Link: https://calendar.google.com/event?eid=xxx"
```

---

### 2.6 BROWSER AGENT
**File:** `/home/user/Project-Friday/connectors/browser_agent.py`

**Capabilities (get_capabilities - Lines 412-430):**
```
✓ Navigate to websites and URLs
✓ Click buttons and interactive elements
✓ Fill forms and input fields
✓ Extract data and text from pages
✓ Take screenshots and generate PDFs
✓ Execute JavaScript in page context
✓ Handle authentication and cookies
✓ Automate complex workflows
✓ Scrape dynamic content
✓ Handle file uploads and downloads
```

**Execution Method:**
```
Entry Point: async def execute(instruction: str) -> str
Location: Lines 432-750
Protocol: LLM Function Calling (Gemini 2.5 Flash)
```

**Tools Available (Gemini Function Definitions):**
- navigate_url
- click_element
- type_text
- extract_content
- screenshot
- execute_javascript
- handle_dialog
- upload_file

**Authentication:**
```
Method: Session/Cookie Based
No explicit credentials needed - uses built-in browser automation
Supported via: Playwright/Puppeteer integration
```

**Request Format Example:**
```python
instruction = "Go to example.com, click the login button, and fill in the form"
result = await browser_agent.execute(instruction)
```

**Response Format:**
```
"✓ Navigated to example.com and completed login form"
```

---

### 2.7 SCRAPER AGENT
**File:** `/home/user/Project-Friday/connectors/scraper_agent.py`

**Capabilities (get_capabilities - Lines 429-447):**
```
✓ Scrape single web pages
✓ Crawl entire websites
✓ Extract structured data with AI
✓ Handle JavaScript-rendered content
✓ Convert pages to Markdown
✓ Extract specific data fields
✓ Follow links and pagination
✓ Respect robots.txt rules
✓ Export data in multiple formats
✓ Parse HTML and extract elements
```

**Execution Method:**
```
Entry Point: async def execute(instruction: str) -> str
Location: Lines 449-750
Protocol: MCP (Model Context Protocol) + Firecrawl API
```

**External API Integration:**
```
Service: Firecrawl (https://firecrawl.dev)
API Endpoint: https://api.firecrawl.dev/v0/scrape
Method: POST
```

**Authentication:**
```
Method: API Key
Environment Variable: FIRECRAWL_API_KEY
Source: https://firecrawl.dev (sign up for API key)
Configuration: Passed via environment in MCP server setup
```

**Request Format Example:**
```python
instruction = "Scrape the pricing table from example.com and extract all features"
result = await scraper_agent.execute(instruction)
```

**Response Format:**
```
"✓ Extracted data from example.com:
- Features: [list]
- Pricing: [table]
- Format: Markdown"
```

---

### 2.8 CODE REVIEWER AGENT
**File:** `/home/user/Project-Friday/connectors/code_reviewer_agent.py`

**Capabilities (get_capabilities - Lines 467-485):**
```
✓ Analyze code for security vulnerabilities
✓ Detect performance issues and bottlenecks
✓ Review code quality and maintainability
✓ Validate best practices and patterns
✓ Check error handling and edge cases
✓ Assess test coverage and quality
✓ Review architecture and design
✓ Multi-language support (Python, JS, Java, Go, etc.)
✓ Provide actionable improvement suggestions
✓ Generate structured review reports
```

**Execution Method:**
```
Entry Point: async def execute(instruction: str) -> str
Location: Lines 487-750
Protocol: Direct LLM Analysis (Gemini 2.5 Flash)
```

**Analysis Categories:**
- Security Vulnerabilities
- Performance Issues
- Code Quality
- Best Practices
- Error Handling
- Test Coverage
- Architecture & Design

**Authentication:**
```
Method: GOOGLE_API_KEY (shared with orchestrator)
Uses: Gemini API for analysis
```

**Request Format Example:**
```python
instruction = """Review this Python code for security issues:
def login(username, password):
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    return db.execute(query)
"""
result = await code_reviewer_agent.execute(instruction)
```

**Response Format:**
```
"Security Review Results:
1. CRITICAL: SQL Injection vulnerability in query building
   - Use parameterized queries instead of string interpolation
   
2. HIGH: Password stored in plain text
   - Implement proper password hashing (bcrypt, argon2)
..."
```

---

## SECTION 3: REQUEST/RESPONSE ROUTING ARCHITECTURE

### 3.1 MAIN ENTRY POINT: process_message()
**File:** `/home/user/Project-Friday/orchestrator.py`
**Location:** Lines 1136-1400

```python
async def process_message(user_message: str) -> str:
    """
    Main request routing function
    
    Flow:
    1. Validate input with InputValidator
    2. Process with Hybrid Intelligence System
    3. Extract intents, entities, confidence
    4. Send to LLM with agent tools
    5. Handle function calling loop
    6. Route to appropriate agent(s)
    7. Collect and format responses
    """
```

### 3.2 AGENT INVOCATION: call_sub_agent()
**Location:** Lines 725-813

```python
async def call_sub_agent(
    agent_name: str,
    instruction: str,
    context: Any = None
) -> str:
    """
    Routes requests to specific agents with:
    - Circuit Breaker protection
    - Retry management
    - Analytics tracking
    - Health checking
    """
```

### 3.3 REQUEST VALIDATION PIPELINE
**File:** `/home/user/Project-Friday/core/input_validator.py`

**Validation Steps:**
```
1. InputValidator.validate_instruction(instruction)
   - Check empty/null
   - Verify string type
   - Enforce max length (default: 10,000 chars)
   - Scan for null bytes
   - Check for injection patterns (excessive special chars)
   
2. InputValidator.validate_parameter(param_name, param_value)
   - Null check
   - Length check (default: 5,000 chars)
   - Null byte detection
   
3. InputValidator.sanitize_for_regex(text)
   - Escape special regex characters
   - Prevent ReDoS attacks
   - Limit pattern length (default: 1,000 chars)
```

### 3.4 REQUEST FLOW DIAGRAM

```
USER INPUT
    ↓
[InputValidator.validate_instruction()]
    ↓
[HybridIntelligenceSystem v5.0]
├─→ Tier 1: Fast Filter (regex patterns)
└─→ Tier 2: LLM Classifier (semantic analysis)
    ↓
[Intent/Entity/Task Extraction]
    ↓
[Confidence Scoring]
    ↓
[LLM Function Calling with Agent Tools]
    ↓
[call_sub_agent(agent_name, instruction, context)]
    ↓
[CircuitBreaker.can_execute(agent_name)?]
    ↓
[RetryManager.execute_with_retry()]
    ↓
[Agent.execute(instruction)]
    ↓
[Response Formatting + Logging]
    ↓
USER RESPONSE
```

---

## SECTION 4: MIDDLEWARE & INTERCEPTORS

### 4.1 CIRCUIT BREAKER PATTERN
**File:** `/home/user/Project-Friday/core/circuit_breaker.py`

**Purpose:** Prevent cascading failures by temporarily disabling failing agents

**State Machine:**
```
CLOSED (normal) → OPEN (failing) → HALF_OPEN (testing) → CLOSED
```

**Configuration (orchestrator.py, Lines 210-219):**
```python
CircuitBreaker(
    config=CircuitConfig(
        failure_threshold=5,        # Open after 5 consecutive failures
        success_threshold=2,        # Close after 2 consecutive successes
        timeout_seconds=300.0,      # 5 minute recovery waiting period
        half_open_timeout=10.0      # 10 second test period
    )
)
```

**Middleware Check (call_sub_agent, Lines 735-740):**
```python
allowed, reason = await self.circuit_breaker.can_execute(agent_name)
if not allowed:
    return f"⚠️ {reason}"  # Block request with descriptive error
```

---

### 4.2 RETRY MANAGER
**File:** `/home/user/Project-Friday/core/retry_manager.py`

**Purpose:** Intelligent retry with exponential backoff and jitter

**Configuration (orchestrator.py, Lines 200-208):**
```python
RetryManager(
    max_retries=3,              # 3 attempts max
    base_delay=1.0,             # Start with 1 second
    max_delay=30.0,             # Cap at 30 seconds
    backoff_factor=2.0,         # Exponential: 1s, 2s, 4s, 8s...
    jitter=True                 # Add randomness to prevent thundering herd
)
```

**Retry Strategy:**
- Attempt 1: Immediate
- Attempt 2: 1 second delay (+ jitter)
- Attempt 3: 2 second delay (+ jitter)
- Attempt 4: 4 second delay (+ jitter)
- Then fail with descriptive error

---

### 4.3 UNDO MANAGER
**File:** `/home/user/Project-Friday/core/undo_manager.py`

**Purpose:** Allow reversal of destructive operations

**Supported Operations:**
- Slack message deletion
- Jira issue creation (can transition to archived)
- Notion page deletion (can restore from archive)
- GitHub PR closing (can reopen)

**Configuration (orchestrator.py, Lines 221-226):**
```python
UndoManager(
    max_undo_history=20,           # Keep last 20 undoable operations
    default_ttl_seconds=3600,      # 1 hour TTL for undo
)
```

---

### 4.4 USER PREFERENCE MANAGER
**File:** `/home/user/Project-Friday/core/user_preferences.py`

**Purpose:** Learn from user behavior and adjust responses

**Tracked Metrics:**
- Interaction times (working hours learning)
- Communication style (formal vs casual)
- Agent preferences (which agents user favors)
- Confirmation preferences (skip confirmations for trusted actions)

**Storage:** `data/preferences/{user_id}.json`

---

### 4.5 ANALYTICS COLLECTOR
**File:** `/home/user/Project-Friday/core/analytics.py`

**Purpose:** Track performance metrics

**Tracked Metrics:**
- Message latency
- Agent execution times
- Error rates
- Agent usage frequency
- Token consumption (for LLM billing)

---

## SECTION 5: ERROR HANDLING & CATEGORIZATION

### 5.1 ERROR CLASSIFICATION SYSTEM
**File:** `/home/user/Project-Friday/core/error_handler.py`

**Error Categories:**

| Category | Pattern | Action | Retryable |
|----------|---------|--------|-----------|
| TRANSIENT | timeout, connection, network | Retry with backoff | Yes |
| RATE_LIMIT | rate limit, quota exceeded, 429, 503 | Retry with longer delay | Yes |
| CAPABILITY | does not support, not available, unsupported | Inform user, no retry | No |
| PERMISSION | forbidden, 401, 403, access denied, unauthorized | Require user action | No |
| VALIDATION | invalid input, malformed data | Clarify input, no retry | No |
| UNKNOWN | other errors | Retry once, then inform | Yes |

**Implementation:**
```python
classifier = ErrorClassifier()
classification = classifier.classify(error_message)
# Returns: ErrorClassification(
#   category: ErrorCategory,
#   is_retryable: bool,
#   explanation: str,
#   suggestions: List[str],
#   retry_delay_seconds: int
# )
```

### 5.2 ERROR MESSAGE ENHANCEMENT
**File:** `/home/user/Project-Friday/core/error_messaging.py`

**Purpose:** Convert technical errors to user-friendly messages

**Error Message Format:**
```
"❌ [Agent Name]: [User-friendly explanation]
   What went wrong: [Technical details]
   What to try next: [Suggested actions]"
```

**Example:**
```
"❌ Jira: Unable to create ticket
   What went wrong: You don't have permission to create issues in this project
   What to try next:
   1. Request project access from your Jira admin
   2. Try creating in a different project
   3. Ask someone with permission to create it for you"
```

---

## SECTION 6: AUTHENTICATION & AUTHORIZATION

### 6.1 AUTHENTICATION METHODS BY AGENT

**Slack Agent:**
- Method: OAuth 2.0 Token
- Scope: Read + Write
- Environment Variable: SLACK_BOT_TOKEN
- Renewal: Token-based (no expiration if properly configured)

**Jira Agent:**
- Method: HTTP Basic Auth
- Credentials: Email + API Token
- Environment Variables: JIRA_URL, JIRA_USERNAME, JIRA_API_TOKEN
- Renewal: Manual token rotation

**GitHub Agent:**
- Method: Personal Access Token (PAT)
- Scopes: repo, read:org
- Environment Variable: GITHUB_TOKEN
- Renewal: Manual token rotation

**Notion Agent:**
- Method: Internal Integration Token
- Scope: Configurable per integration
- Environment Variable: NOTION_TOKEN
- Renewal: Manual token rotation

**Google Calendar Agent:**
- Method: OAuth 2.0 (3-legged flow)
- Scopes: https://www.googleapis.com/auth/calendar
- Environment Variables: GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT_URI
- Renewal: Automatic (refresh tokens)

**Scraper Agent:**
- Method: API Key
- Service: Firecrawl
- Environment Variable: FIRECRAWL_API_KEY
- Renewal: Manual

**Code Reviewer Agent:**
- Method: Google API Key (shared)
- Environment Variable: GOOGLE_API_KEY
- Service: Gemini API
- Renewal: Manual

### 6.2 SECURITY FEATURES

**Input Sanitization:**
```python
# Enabled by default (ENABLE_SANITIZATION=true)
# Prevents injection attacks:
- Null byte injection: '\x00' checks
- SQL/Command injection: Special character limits
- Regex DoS: Pattern length limits + dangerous pattern detection
- Path traversal: Input validation
```

**Sensitive Data Handling:**
```python
# From base_agent.py, Line 354:
sensitive = {'api_key', 'token', 'password', 'secret', 'auth'}
# These fields are masked in logs
```

**MCP Configuration Timeout:**
```python
# From mcp_config.py
MCPTimeouts.INITIAL_CONNECTION = 30.0 seconds
MCPTimeouts.TOOL_EXECUTION = 60.0 seconds
MCPTimeouts.SESSION_INIT = 10.0 seconds
```

---

## SECTION 7: API VERSIONING & COMPATIBILITY

### 7.1 VERSIONING SCHEME

**System Version:** 3.0
**Location:** orchestrator.py docstring

**Component Versions:**
```
BaseAgent v2.0                  (stable, abstract interface)
OrchestratorAgent (no version)  (core routing, continuously improved)
Circuit Breaker v2.0            (production grade)
MCP Config v1.0                 (standard MCP)
Retry Manager (no version)      (production grade)
Hybrid Intelligence v5.0        (latest tier-based system)
```

### 7.2 BREAKING CHANGES

**Removed (v3.0):**
- Old SessionLogger (replaced with SimpleSessionLogger)
- Old UnifiedSessionLogger (replaced with SimpleSessionLogger)
- Separate intent_classifier/entity_extractor (merged into HybridIntelligenceSystem)

**Kept for Backward Compatibility:**
- Legacy intent_classifier (lines 182)
- Legacy entity_extractor (lines 183)
- Terminal UI v1 (as fallback)

### 7.3 CONFIGURATION VERSIONING

**Configuration Format:** Environment Variables
**Location:** `.env` file (copy from `.env.example`)
**Version Control:** Values can be overridden per-module

**Per-Module Log Level Configuration:**
```
LOG_LEVEL_ORCHESTRATOR
LOG_LEVEL_SLACK
LOG_LEVEL_JIRA
LOG_LEVEL_GITHUB
LOG_LEVEL_NOTION
LOG_LEVEL_ERROR_HANDLER
LOG_LEVEL_INTELLIGENCE
```

---

## SECTION 8: RESPONSE FORMATS

### 8.1 SUCCESS RESPONSE FORMAT

**Standard Success Response:**
```
"✓ [Action completed successfully]
[Details about what was created/updated]
Link: [URL to resource]"
```

**Example Slack Success:**
```
"✓ Sent message to #general channel
Message: 'Team standup in 5 minutes'
Reactions: 1 person reacted"
```

**Example Jira Success:**
```
"✓ Created issue KAN-456: Login page timeout issue
Project: KAN
Issue Type: Bug
Priority: High
Assignee: John Smith
Link: https://your-domain.atlassian.net/browse/KAN-456"
```

### 8.2 ERROR RESPONSE FORMAT

**Standard Error Response:**
```
"⚠️ [Error Type]: [User-friendly message]
Technical Details: [What failed]
Suggestion: [What to try next]"
```

**Example:**
```
"⚠️ Jira: Authentication Failed
Technical Details: Invalid API token or expired credentials
Suggestion: Check your JIRA_API_TOKEN in .env file at https://id.atlassian.com/manage-profile/security/api-tokens"
```

### 8.3 PARTIAL SUCCESS RESPONSE

**Format:**
```
"⚠️ Partial Success: [What succeeded] but [what failed]
Completed: [list]
Failed: [list with reasons]"
```

**Example:**
```
"⚠️ Partial Success: Sent 8 of 10 messages
Completed: [msg1, msg2, ..., msg8]
Failed: 
- msg9: Channel #archive is archived (read-only)
- msg10: User @bot-user doesn't have write access to #internal"
```

---

## SECTION 9: REQUEST/RESPONSE EXAMPLES BY AGENT

### 9.1 SLACK AGENT REQUEST/RESPONSE

**Request:**
```json
{
  "agent": "slack",
  "instruction": "Send a message to #announcements saying 'Version 3.0 released!' with a rocket emoji",
  "context": {
    "channel_preference": "#announcements",
    "urgency": "high"
  }
}
```

**Response:**
```
✓ Posted message to #announcements
Message: "Version 3.0 released! 🚀"
Timestamp: 2025-11-20T15:30:45Z
Reactions: 🚀 (1), 🎉 (2)
```

### 9.2 JIRA AGENT REQUEST/RESPONSE

**Request:**
```json
{
  "agent": "jira",
  "instruction": "Create a critical bug ticket: 'Database connection timeout after 5 minutes of inactivity. Affects production users.' with priority High and assign to Sarah",
  "context": {
    "project": "BACKEND",
    "component": "database"
  }
}
```

**Response:**
```
✓ Created issue BACKEND-1823: Database connection timeout
Description: "Database connection timeout after 5 minutes of inactivity. Affects production users."
Priority: High
Assignee: Sarah Johnson
Status: To Do
Components: database
Link: https://mycompany.atlassian.net/browse/BACKEND-1823

Auto-transitioned to In Progress in 2 minutes
```

### 9.3 GITHUB AGENT REQUEST/RESPONSE

**Request:**
```json
{
  "agent": "github",
  "instruction": "Review the pull request #245 in Facebook/React for code quality and security issues",
  "context": {
    "repository": "facebook/react",
    "pr_number": 245
  }
}
```

**Response:**
```
✓ Reviewed PR #245: Add concurrent rendering support

Code Quality Review:
- ✓ Proper error handling
- ⚠️ 2 console.log() statements left in (should use logger)
- ✓ Good test coverage (92%)

Security Review:
- ✓ No obvious injection vulnerabilities
- ✓ Input properly validated
- ✓ No hardcoded secrets

Performance:
- ✓ Algorithm complexity acceptable
- ⚠️ Memory usage increased by ~2MB in worst case

Suggestions:
1. Remove debug console.log statements
2. Add memory usage benchmarks
3. Update documentation for new parameters

Link: https://github.com/facebook/react/pull/245
```

### 9.4 NOTION AGENT REQUEST/RESPONSE

**Request:**
```json
{
  "agent": "notion",
  "instruction": "Create a new meeting notes page titled 'Team Sync - 2025-11-20' with agenda items: 1) Project status, 2) Blockers, 3) Next week plan",
  "context": {
    "parent_page": "Meetings Archive",
    "add_to_database": "Meeting Notes"
  }
}
```

**Response:**
```
✓ Created Notion page: Team Sync - 2025-11-20

Blocks Added:
- Heading: "Agenda"
- Numbered List: "Project status"
- Numbered List: "Blockers"
- Numbered List: "Next week plan"

Database Entry: Added to "Meeting Notes" with:
- Date: 2025-11-20
- Status: In Progress
- Attendees: (empty - ready for manual entry)

Link: https://notion.so/page_id_abcdef123456
```

### 9.5 GOOGLE CALENDAR AGENT REQUEST/RESPONSE

**Request:**
```json
{
  "agent": "google_calendar",
  "instruction": "Schedule a weekly team standup meeting every Monday at 9:00 AM for 30 minutes starting next week with the team@company.com distribution list",
  "context": {
    "calendar": "Work Calendar",
    "location": "Conference Room B or Google Meet"
  }
}
```

**Response:**
```
✓ Created recurring event: Team Standup

Details:
- Time: Every Monday at 9:00 AM (30 minutes)
- Start Date: 2025-11-24
- Recurrence: Weekly (no end date)
- Location: Google Meet (link: https://meet.google.com/abc-defg-hij)
- Attendees: team@company.com (invited)
- Reminders: 15 minutes before, 1 day before

Next Occurrence: Monday, 2025-11-24 at 9:00 AM
Event Link: https://calendar.google.com/event?eid=abc123

Calendar Updated: 1 event created with 1 recurring rule
```

### 9.6 BROWSER AGENT REQUEST/RESPONSE

**Request:**
```json
{
  "agent": "browser",
  "instruction": "Navigate to https://example.com/login, log in with username 'test@example.com' and password (from secure store), then take a screenshot of the dashboard",
  "context": {
    "timeout": 30000
  }
}
```

**Response:**
```
✓ Completed browser automation successfully

Actions Performed:
1. ✓ Navigated to https://example.com/login (loaded in 1.2s)
2. ✓ Filled username field with 'test@example.com'
3. ✓ Filled password field (credentials from secure store)
4. ✓ Clicked "Login" button
5. ✓ Waited for navigation (loaded dashboard in 0.8s)
6. ✓ Took screenshot of dashboard (1920x1080, 324KB)

Screenshot Location: /tmp/screenshot-2025-11-20-1530.png
Dashboard Title: "Welcome back, Test User"
Visible Elements: Navigation, Widgets, Recent Activity
```

### 9.7 SCRAPER AGENT REQUEST/RESPONSE

**Request:**
```json
{
  "agent": "scraper",
  "instruction": "Scrape the pricing page from https://example.com/pricing and extract all plan names, prices, features, and billing cycles",
  "context": {
    "format": "markdown",
    "include_html": false
  }
}
```

**Response:**
```
✓ Extracted data from https://example.com/pricing (loaded in 2.1s)

Pricing Plans Found: 3

## Plan 1: Starter
- Price: $29/month
- Billing: Monthly or Annual ($290/year)
- Features:
  - Up to 5 users
  - Basic support
  - 10GB storage
  - Email integrations
  - API access

## Plan 2: Professional
- Price: $99/month
- Billing: Monthly or Annual ($990/year)
- Features:
  - Up to 50 users
  - Priority support
  - 1TB storage
  - Advanced integrations
  - Webhooks
  - Custom fields

## Plan 3: Enterprise
- Price: Custom (contact sales)
- Billing: Annual
- Features:
  - Unlimited users
  - Dedicated support
  - Unlimited storage
  - Custom integrations
  - Advanced security
  - SLA guarantee

Full Report: markdown_file_path_here
```

### 9.8 CODE REVIEWER AGENT REQUEST/RESPONSE

**Request:**
```json
{
  "agent": "code_reviewer",
  "instruction": "Review this Python function for security, performance, and best practices:\n\ndef process_user_input(user_data):\n    result = {}\n    for key, value in user_data.items():\n        result[key.replace('_', ' ').title()] = value\n    return result",
  "context": {
    "language": "python",
    "focus": ["security", "performance"]
  }
}
```

**Response:**
```
✓ Code Review Complete

Security Issues: 1
- LOW: No input validation on keys/values
  → Could process unexpected data types
  → Recommendation: Add type hints and validation

Performance Issues: 1
- LOW: String operations in loop (replace, title)
  → Recommendation: Pre-compile transformations or use map()

Code Quality Issues: 1
- MEDIUM: No error handling for malformed input
  → Recommendation: Add try/except block

Best Practices: 2
- MISSING: No docstring
  → Add docstring with examples and parameters
- MISSING: No type hints
  → Add type hints: def process_user_input(user_data: Dict[str, Any]) -> Dict[str, Any]:

Improved Version:
```python
def process_user_input(user_data: Dict[str, Any]) -> Dict[str, Any]:
    '''
    Transform user data keys to title case.
    
    Args:
        user_data: Dictionary of user-provided data
        
    Returns:
        Dictionary with transformed keys
    '''
    if not isinstance(user_data, dict):
        raise TypeError("user_data must be a dictionary")
    
    return {
        key.replace('_', ' ').title(): value
        for key, value in user_data.items()
    }
```

Overall Score: 7/10 (Good - minor improvements recommended)
```

---

## SECTION 10: MCP (MODEL CONTEXT PROTOCOL) CONFIGURATION

### 10.1 MCP TIMEOUTS

**File:** `/home/user/Project-Friday/connectors/mcp_config.py`

| Operation | Timeout | Purpose |
|-----------|---------|---------|
| INITIAL_CONNECTION | 30.0s | Initial server connection |
| SESSION_INIT | 10.0s | Session initialization |
| TOOL_LIST | 10.0s | List available tools |
| TOOL_EXECUTION | 60.0s | Execute individual tool |
| SEARCH_OPERATION | 45.0s | Search/query operations |
| CREATE_OPERATION | 30.0s | Create new resources |
| UPDATE_OPERATION | 30.0s | Update existing resources |
| DELETE_OPERATION | 20.0s | Delete resources |

### 10.2 MCP RETRY CONFIGURATION

```python
MCPRetryConfig.MAX_RETRIES = 3
MCPRetryConfig.INITIAL_DELAY = 1.0 seconds
MCPRetryConfig.MAX_DELAY = 10.0 seconds
MCPRetryConfig.BACKOFF_FACTOR = 2.0
MCPRetryConfig.CONNECTION_MAX_RETRIES = 2
```

### 10.3 MCP RETRYABLE ERRORS

**Network Errors:**
- timeout, connection, network
- ECONNRESET, ETIMEDOUT, ECONNREFUSED, ENETUNREACH

**SSE Errors:**
- sse error, sseerror, body timeout, terminated
- stream closed, connection closed, fetch failed, stream error

**HTTP Errors:**
- 503 (Service Unavailable)
- 502 (Bad Gateway)
- 504 (Gateway Timeout)
- 429 (Too Many Requests)

**Rate Limiting:**
- rate_limited, rate limit, too many requests, quota exceeded

**Server Errors:**
- internal server error, service unavailable, bad gateway

---

## SECTION 11: CONFIGURATION & DEPLOYMENT

### 11.1 ENVIRONMENT VARIABLES

**Critical (Required):**
```
GOOGLE_API_KEY = Gemini API key for orchestrator
```

**Agent-Specific (Conditional):**
```
# Slack
SLACK_BOT_TOKEN (required for Slack agent)

# Jira
JIRA_URL (required)
JIRA_USERNAME (required)
JIRA_API_TOKEN (required)

# GitHub
GITHUB_TOKEN (required)

# Notion
NOTION_TOKEN (required)

# Google Calendar
GOOGLE_CLIENT_ID (required)
GOOGLE_CLIENT_SECRET (required)
GOOGLE_REDIRECT_URI (optional, defaults to http://localhost:4153/oauth2callback)

# Scraper
FIRECRAWL_API_KEY (required)
```

**Configuration Variables:**
```
# Timeouts
AGENT_TIMEOUT = 120.0 seconds
ENRICHMENT_TIMEOUT = 5.0 seconds
LLM_TIMEOUT = 30.0 seconds

# Retry
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0
INITIAL_RETRY_DELAY = 1.0

# Input Validation
MAX_INSTRUCTION_LENGTH = 10000
MAX_PARAM_LENGTH = 5000
MAX_REGEX_LENGTH = 1000

# Logging
LOG_LEVEL = INFO
LOG_DIR = logs
ENABLE_FILE_LOGGING = true
ENABLE_JSON_LOGGING = true
ENABLE_CONSOLE_LOGGING = true
ENABLE_COLORED_LOGS = true
MAX_LOG_FILE_SIZE_MB = 10
LOG_BACKUP_COUNT = 5
VERBOSE = false

# Security
ENABLE_SANITIZATION = true

# Enrichment
REQUIRE_ENRICHMENT_HIGH_RISK = true
FAIL_OPEN_ENRICHMENT = false
```

### 11.2 SESSION LOGGING

**Location:** `logs/{session_id}/`

**Files Generated:**
1. `{session_id}_summary.txt` - Human-readable summary
2. `{session_id}_detailed.txt` - Detailed execution log
3. `{session_id}_metrics.json` - Performance metrics (optional)

**Logged Information:**
- User messages
- Agent selections
- Tool executions
- Response summaries
- Errors and warnings
- Performance metrics

---

## SECTION 12: PRODUCTION CONSIDERATIONS

### 12.1 SECURITY CHECKLIST

- [x] Input validation enabled by default
- [x] Sensitive data masked in logs
- [x] API keys stored in environment variables
- [x] No hardcoded credentials in code
- [x] Circuit breaker for fault isolation
- [x] Rate limiting via MCP timeouts
- [x] Error messages don't expose sensitive info
- [x] Null injection prevention
- [x] SQL injection prevention (parameterized queries)
- [x] XSS prevention (not applicable for CLI)
- [x] CSRF prevention (not applicable for CLI)

### 12.2 PERFORMANCE CHARACTERISTICS

**Latency (typical):**
- Simple message routing: 100-500ms
- Agent tool execution: 1-5 seconds
- LLM function calling loop: 2-10 seconds
- End-to-end request: 3-15 seconds

**Throughput:**
- Sequential processing only (not parallel)
- Single user interactive session
- Max 30 concurrent function calling iterations

**Resource Usage:**
- Memory: ~200MB baseline + agent allocations
- Disk: ~1MB per session log
- Network: Depends on agent integrations

### 12.3 ERROR RECOVERY

**Automatic:**
- Transient errors: Retried automatically (3x)
- Rate limited: Retried with exponential backoff
- Connection failures: Retried (2x for connection, 3x for operations)
- Circuit breaker: Automatically tests recovery after 5 minutes

**Manual:**
- Capability gaps: User must choose alternative
- Permission errors: User must grant access
- Validation errors: User must provide correct input

---

## SECTION 13: TESTING & DEBUGGING

### 13.1 VERBOSE MODE

**Enable:**
```bash
python main.py --verbose
```

**Shows:**
- Agent discovery process
- Tool selection logic
- Intelligence system decisions
- Retry attempts
- Circuit breaker state changes
- Full error traces

### 13.2 LOG LEVELS

**DEBUG:** Detailed internal state
**INFO:** User-level information (default)
**WARNING:** Potentially problematic situations
**ERROR:** Error conditions
**CRITICAL:** System failures

**Per-Module Override:**
```
LOG_LEVEL_ORCHESTRATOR=DEBUG
LOG_LEVEL_SLACK=DEBUG
LOG_LEVEL_JIRA=INFO
LOG_LEVEL_GITHUB=INFO
LOG_LEVEL_NOTION=DEBUG
LOG_LEVEL_ERROR_HANDLER=WARNING
LOG_LEVEL_INTELLIGENCE=DEBUG
```

### 13.3 SESSION VIEWER

**Tool:** `tools/session_viewer.py`

**Shows:**
- Session summary
- Message exchange
- Agent selections
- Tool executions
- Performance metrics
- Errors and warnings

---

## SECTION 14: LIMITS & CONSTRAINTS

### 14.1 REQUEST/RESPONSE LIMITS

| Limit | Value | Purpose |
|-------|-------|---------|
| Max instruction length | 10,000 characters | Prevent token explosion |
| Max parameter length | 5,000 characters | Input validation |
| Max regex pattern | 1,000 characters | DoS prevention |
| Max batch size | 10 actions | Prevent overwhelming services |
| Max pending queue | 100 actions | Memory management |
| Function call iterations | 30 per request | Prevent infinite loops |
| Undo history | 20 operations | Memory management |

### 14.2 API RATE LIMITS

**Managed by:**
- External service rate limits
- MCP timeout configuration
- Retry backoff strategy
- Circuit breaker

**Typical Rate Limits:**
- Slack: 3,600 requests/hour
- Jira: 1,000 requests/hour (varies by plan)
- GitHub: 5,000 requests/hour (authenticated)
- Notion: 3 requests/second
- Google Calendar: 10 million requests/day
- Firecrawl: Depends on API plan

### 14.3 TIMEOUT LIMITS

| Operation | Timeout | Retry |
|-----------|---------|-------|
| LLM response | 30 seconds | Yes (3x) |
| Agent execution | 120 seconds | Yes (3x) |
| MCP tool call | 60 seconds | Yes (3x) |
| Network connection | 30 seconds | Yes (2x) |
| Total request | (sum of above) | No |

---

## SECTION 15: KNOWN LIMITATIONS

### 15.1 Agent Capabilities Gaps

**Slack:**
- Cannot delete messages (only admins can)
- Cannot manage channel settings
- Cannot create/delete channels
- Cannot access private channels without membership

**Jira:**
- Cannot delete issues (only archive)
- Cannot modify project settings
- Cannot create/delete projects (admin only)
- Cannot modify audit logs

**GitHub:**
- Cannot fetch raw diff/patch content
- Cannot modify repository settings
- Cannot manage webhooks
- Cannot delete repositories

**Notion:**
- Cannot delete pages (only archive)
- Cannot modify database schemas
- Cannot manage workspace permissions
- Cannot access unpermitted pages

**Browser:**
- Cannot interact with external authentication (OAuth, SAML)
- Cannot bypass security headers
- Limited to public/accessible content

**Scraper:**
- Cannot bypass robots.txt restrictions
- Cannot handle complex JavaScript SPAs well
- Cannot interact with authenticated endpoints

---

## SECTION 16: API DOCUMENTATION RESOURCES

### 16.1 External Service Documentation

**Slack API:**
- https://api.slack.com/docs
- Web API Reference: https://api.slack.com/methods
- OAuth: https://api.slack.com/authentication/oauth-v2

**Jira API:**
- https://developer.atlassian.com/cloud/jira/rest/v3
- JQL Documentation: https://support.atlassian.com/jira-software-cloud/articles/advanced-searching-using-jql

**GitHub API:**
- https://docs.github.com/en/rest
- GraphQL: https://docs.github.com/en/graphql

**Notion API:**
- https://developers.notion.com
- Database API: https://developers.notion.com/reference/database

**Google Calendar API:**
- https://developers.google.com/calendar/api
- OAuth Setup: https://developers.google.com/calendar/api/quickstart/python

**Firecrawl:**
- https://docs.firecrawl.dev
- API Reference: https://docs.firecrawl.dev/api-reference

### 16.2 Internal Documentation

**Session Viewer Guide:**
- `/home/user/Project-Friday/docs/SESSION_VIEWER_GUIDE.md`

**Logging System:**
- `/home/user/Project-Friday/docs/LOGGING_SYSTEM.md`
- `/home/user/Project-Friday/docs/LOGGING_GUIDE.md`

**Code Review Report:**
- `/home/user/Project-Friday/CODE_REVIEW_REPORT.md`

---

## SECTION 17: SUMMARY & KEY TAKEAWAYS

### 17.1 Architecture Overview

Project-Friday is a **multi-agent orchestration system** NOT a traditional REST API:

1. **Entry Point:** Async CLI (main.py) → process_message()
2. **Request Routing:** LLM-based function calling with Gemini 2.5 Flash
3. **Agent Communication:** MCP (Model Context Protocol)
4. **Request Validation:** InputValidator with injection prevention
5. **Failure Handling:** CircuitBreaker + RetryManager
6. **Middleware:** Undo, Preferences, Analytics, UI
7. **Response Format:** Natural language with resource links

### 17.2 Agent Endpoints (8 Total)

| Agent | Protocol | Auth | External API |
|-------|----------|------|--------------|
| Slack | MCP | OAuth Token | Slack API |
| Jira | MCP | Basic Auth | Jira REST API |
| GitHub | MCP | PAT | GitHub API |
| Notion | MCP | Internal Token | Notion API |
| Google Calendar | MCP | OAuth 2.0 | Google Calendar API |
| Browser | LLM Functions | None | (Local automation) |
| Scraper | MCP + Firecrawl | API Key | Firecrawl API |
| Code Reviewer | LLM | Shared Key | Gemini API |

### 17.3 Request/Response Flow

```
User Input
  ↓ Validate (InputValidator)
  ↓ Enhance (HybridIntelligence v5.0)
  ↓ Route (LLM function calling)
  ↓ Execute (Agent via MCP)
  ↓ Handle Errors (ErrorClassifier)
  ↓ Log & Format (SimpleSessionLogger)
  ↓ User Output
```

### 17.4 Production Features

- **Fault Tolerance:** Circuit breaker pattern
- **Reliability:** Exponential backoff retry
- **Observability:** Distributed tracing + analytics
- **Security:** Input validation + sensitive data masking
- **User Experience:** Undo manager + preference learning
- **Performance:** Token optimization + caching

---

## APPENDIX A: FILE LOCATIONS & REFERENCES

**Core Files:**
- Orchestrator: `/home/user/Project-Friday/orchestrator.py`
- Main Entry: `/home/user/Project-Friday/main.py`
- Configuration: `/home/user/Project-Friday/config.py`

**Agents (connectors/):**
- Base: `base_agent.py`
- Slack: `slack_agent.py`
- Jira: `jira_agent.py`
- GitHub: `github_agent.py`
- Notion: `notion_agent.py`
- Google Calendar: `google_calendar_agent.py`
- Browser: `browser_agent.py`
- Scraper: `scraper_agent.py`
- Code Reviewer: `code_reviewer_agent.py`

**Infrastructure (core/):**
- Input Validator: `input_validator.py`
- Error Handler: `error_handler.py`
- Circuit Breaker: `circuit_breaker.py`
- Retry Manager: `retry_manager.py`
- Undo Manager: `undo_manager.py`
- Analytics: `analytics.py`
- Logging: `logging_config.py`, `simple_session_logger.py`

**Intelligence (intelligence/):**
- Hybrid System: `hybrid_system.py`
- Intent Classifier: `intent_classifier.py`
- Entity Extractor: `entity_extractor.py`
- Task Decomposer: `task_decomposer.py`

---

## APPENDIX B: CHANGELOG

**Version 3.0 (Latest):**
- New HybridIntelligenceSystem v5.0 (two-tier: fast filter + LLM)
- Circuit breaker for fault isolation
- Enhanced error classification
- Simple session logger (replaces old logging)
- Undo manager for reversible operations
- User preference learning
- Analytics collector

**Version 2.x:**
- Base agent abstraction
- MCP integration
- Individual agent implementations

**Version 1.x:**
- Initial multi-agent system
- Basic orchestration

---

**END OF REPORT**

Report compiled by: Claude Code
Date: 2025-11-20
Project: Project-Friday v3.0
Total Agents: 8
Total Capabilities: 50+
Total Configuration Variables: 30+
