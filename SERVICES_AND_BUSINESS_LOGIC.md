# PROJECT-FRIDAY: COMPREHENSIVE SERVICES AND BUSINESS LOGIC ANALYSIS

## EXECUTIVE SUMMARY

Project-Friday is a sophisticated AI-powered workspace orchestration system that coordinates specialized agents across multiple platforms (Jira, Slack, GitHub, Notion, Google Calendar, etc.). It features a two-tier hybrid intelligence system combining fast keyword filtering with LLM-based semantic understanding, comprehensive observability, and advanced resilience patterns.

---

## 1. CORE SERVICES ARCHITECTURE

### 1.1 ORCHESTRATION LAYER (orchestrator.py - 1812 lines)

**Primary Responsibility**: Main coordination hub that orchestrates all specialized agents and intelligence components.

**Key Features**:
- **Agent Discovery & Loading**: Dynamically loads agents from `*_agent.py` files with intelligent fallback mechanisms
- **Message Processing Pipeline**: Routes user messages through intelligence system to determine optimal agent(s)
- **Task Execution**: Calls sub-agents with retry logic, circuit breaker protection, and analytics
- **Session Management**: Maintains session context, conversation history, and user preferences
- **Error Handling**: Comprehensive error classification and recovery strategies

**Core Components Initialized**:
```
- LLM Abstraction (Gemini Flash)
- Hybrid Intelligence System (Fast Filter + LLM Classifier)
- Retry Manager (exponential backoff with jitter)
- Circuit Breaker (per-agent failure tracking)
- Undo Manager (reversible operations)
- Analytics Collector (performance metrics)
- User Preference Manager (behavioral learning)
- Observability System (distributed tracing, metrics)
- Simple Session Logger (human-readable logs)
```

**Data Flow**:
```
User Message
    ↓
Input Validation
    ↓
Hybrid Intelligence (Intent/Entity Extraction)
    ↓
Task Decomposition
    ↓
Confidence Scoring
    ↓
Agent Selection & Execution
    ↓
Result Aggregation & Formatting
    ↓
User Response
```

---

## 2. INTELLIGENCE SYSTEM (Tier 1 & 2)

### 2.1 HYBRID INTELLIGENCE SYSTEM v5.0 (intelligence/hybrid_system.py)

**Architecture**: Two-tier system optimizing speed vs. accuracy

**Tier 1: Fast Keyword Filter** (10ms, cost-free)
- Pattern matching for obvious, high-confidence requests
- Coverage: 35-40% of requests
- Accuracy: 95% for covered patterns
- File: `intelligence/fast_filter.py`

**Tier 2: LLM Classifier** (~200ms, $0.01/1K requests)
- Deep semantic understanding for complex requests
- Coverage: 60-65% of remaining requests
- Accuracy: 92%
- Semantic caching: 70-80% cache hit rate
- File: `intelligence/llm_classifier.py`

**Performance Targets**:
- Overall accuracy: 92%
- Average latency: 80ms
- Cost: $0.0065/1K requests

### 2.2 INTENT CLASSIFIER (intelligence/intent_classifier.py)

**Capabilities**:
- Detects 7 primary intent types:
  - CREATE (make something new)
  - READ (view/retrieve information)
  - UPDATE (modify existing data)
  - DELETE (remove something)
  - ANALYZE (examine/evaluate data)
  - COORDINATE (notify/communicate)
  - SEARCH (find specific information)
- Multi-intent detection (multiple intents in one request)
- Implicit requirement detection
- Coreference resolution support
- LLM-enhanced semantic understanding (optional)

**Methods**:
- `classify(message: str) → List[Intent]`
- `classify_with_context(message: str, context: Dict) → List[Intent]`

### 2.3 ENTITY EXTRACTOR (intelligence/entity_extractor.py)

**Entity Types Recognized**:
- ISSUE (Jira: KAN-123, GitHub: #456)
- PR (Pull Request: PR #789)
- PROJECT (project names)
- CHANNEL (#slack-channel)
- PERSON (@username)
- DATE (tomorrow, next week, by Friday)
- PRIORITY (critical, high, medium, low)
- FILE (file paths/names)
- CODE (code snippets)

**Features**:
- Multi-pass extraction (regex → NER → relationships)
- Entity confidence calibration
- Duplicate detection and merging
- Contextual extraction with conversation history
- Named Entity Recognition (NER)
- Relationship extraction between entities

### 2.4 TASK DECOMPOSER (intelligence/task_decomposer.py)

**Responsibilities**:
- Break complex requests into executable tasks
- Detect task dependencies
- Optimize execution order
- Identify parallelizable tasks
- Build execution plans with data flow
- Map entities to best-fit agents

**Methods**:
- `decompose(message, intents, entities, context) → ExecutionPlan`

### 2.5 CONFIDENCE SCORER (intelligence/confidence_scorer.py)

**Scoring Factors**:
- Intent clarity (0-1.0)
- Entity completeness (0-1.0)
- Message clarity (0-1.0)
- Plan quality (0-1.0)
- Bayesian probability combination

**Uses Bayesian Estimation** for probabilistic confidence scoring

**Methods**:
- `score_overall(message, intents, entities, plan) → Confidence`
- `should_confirm(confidence) → bool`

### 2.6 CONTEXT MANAGER (intelligence/context_manager.py)

**Capabilities**:
- Remember conversation history (multi-turn)
- Track entities across messages
- Resolve coreferences ("it", "that", "the issue")
- Maintain topic focus
- Understand temporal context
- Learn patterns from interaction

**Tracked Data**:
- Conversation turns (user/assistant messages)
- Entity tracking (with mention counts)
- Current focus (recent entities)
- Temporal context (project, repository, branch)
- Learned patterns

### 2.7 CACHE LAYER (intelligence/cache_layer.py)

**Features**:
- LRU cache with TTL support (5 min default)
- Thread-safe operations
- Statistics tracking (hits, misses, evictions)
- Automatic cleanup
- Global cache instance

**Performance Impact**: 70-80% cache hit rate on LLM classifications

---

## 3. LLM INTEGRATION LAYER

### 3.1 BASE LLM ABSTRACTION (llms/base_llm.py)

**Design**: Provider-agnostic interface supporting any LLM provider

**Supported Capabilities**:
- Chat completions
- Function calling / tool use
- Streaming responses
- Async-first design
- MCP tool compatibility

**Key Classes**:
- `BaseLLM`: Abstract base for all LLM providers
- `ChatSession`: Manages multi-turn conversations
- `LLMResponse`: Universal response format
- `FunctionCall`: Universal function call format
- `LLMConfig`: Configuration for all models

**Configuration Parameters**:
```
- model_name: string
- temperature: float (0.0-2.0)
- max_tokens: optional int
- top_p: float (0.0-1.0)
- top_k: int
- stop_sequences: optional list
- system_instruction: optional string
```

### 3.2 GEMINI FLASH IMPLEMENTATION (llms/gemini_flash.py)

**Provider**: Google Gemini 2.5 Flash

**Features**:
- Fast inference (optimized for speed)
- Function calling / tool use support
- Async-first design
- MCP tool schema conversion
- Chat session management
- Response extraction handling

**API Integration**:
- Uses `google.generativeai` SDK
- Handles complex response structures
- Graceful degradation for errors

**Used By**:
- Hybrid Intelligence System (Tier 2 LLM Classifier)
- Code Reviewer Agent (code analysis)
- Most agents with LLM capabilities

---

## 4. CONNECTOR/AGENT SERVICES

### 4.1 BASE AGENT ARCHITECTURE (connectors/base_agent.py)

**Design Pattern**: Abstract base class for all specialized agents

**Agent Lifecycle**:
1. **Initialization**: Constructor + `initialize()`
2. **Discovery**: `get_capabilities()` returns list of actions
3. **Execution**: `execute(instruction)` performs actions
4. **Cleanup**: `cleanup()` releases resources

**Key Features**:
- Safe response text extraction from Gemini API
- Error handling with graceful degradation
- Connection management
- Tool initialization

---

### 4.2 CONNECTOR AGENTS (MCP-based)

#### **JIRA AGENT** (connectors/jira_agent.py)
**Purpose**: Project management and issue tracking

**Capabilities**:
- Create/update/delete issues
- Manage issue workflows (transitions, status changes)
- Assign issues and set priorities
- Search and filter issues
- Link issues together
- Add comments and attachments
- Sprint management

**Integration**:
- MCP: sooperset/mcp-atlassian Docker image
- Authentication: API tokens
- Retry logic: 3 retries with exponential backoff (1s → 10s)

**Error Types Handled**:
- Authentication errors
- Permission denied
- Rate limiting
- Not found (invalid issue keys)
- Validation errors (invalid data)

#### **SLACK AGENT** (connectors/slack_agent.py)
**Purpose**: Team communication and collaboration

**Capabilities**:
- Send messages to channels
- Create threads and reply to messages
- Search conversations and channels
- Manage channel topics and descriptions
- Post formatted messages with blocks
- User and channel management
- Emoji reactions

**Integration**:
- MCP: Slack API via MCP server
- Authentication: Slack tokens/webhooks
- Message limits: Respects rate limiting (429 responses)

**Special Handling**: Message length validation (Slack limits)

#### **GITHUB AGENT** (connectors/github_agent.py)
**Purpose**: Code collaboration and development workflow

**Capabilities**:
- Create/update/close issues and PRs
- Manage branches and commits
- Review code and create comments
- Manage labels, milestones, and assignees
- Repository management
- Release management
- Workflow automation

**Integration**:
- MCP: GitHub API via MCP server
- Authentication: GitHub tokens
- Branch protection handling

**Error Types**: Branch protection, invalid repo format, permission issues

#### **NOTION AGENT** (connectors/notion_agent.py)
**Purpose**: Knowledge management and documentation

**Capabilities**:
- Create/update/delete pages and databases
- Query and search content
- Manage properties and relations
- Block management (text, images, tables)
- User and permission management
- Archive and restore pages

**Integration**:
- MCP: Notion API via MCP server (SSE-based)
- Authentication: Notion API tokens
- Special handling: SSE connection errors, body timeouts

**Smart Filtering**: Excludes tutorial/template pages

#### **GOOGLE CALENDAR AGENT** (connectors/google_calendar_agent.py)
**Purpose**: Calendar management and meeting scheduling

**Capabilities**:
- Create/update/delete events
- List and search events
- Manage attendees and invitations
- Set reminders and notifications
- Manage calendar properties
- Availability checking
- Conflict detection

**Integration**:
- MCP: Google Calendar API via MCP server
- Authentication: Google OAuth tokens
- Rate limits: Quota management

**Error Handling**: Quota exceeded, rate limits, timezone issues

#### **CODE REVIEWER AGENT** (connectors/code_reviewer_agent.py)
**Purpose**: Intelligent code analysis and review

**Capabilities**:
- Code quality assessment
- Security vulnerability detection
- Performance issue identification
- Best practices validation
- Architecture review
- Multi-language support (Python, JavaScript, Java, Go, etc.)

**Integration**:
- LLM-based: Uses Gemini Flash for analysis
- No external API (pure analysis)
- Can analyze GitHub PRs, local files, or code snippets

**Analysis Types**:
- Security vulnerabilities
- Performance bottlenecks
- Code quality issues
- Best practice violations
- Architecture patterns

#### **BROWSER AGENT** (connectors/browser_agent.py)
**Purpose**: Web automation and data extraction

**Capabilities**:
- Navigate websites
- Click elements and fill forms
- Extract structured data
- Capture screenshots and PDFs
- JavaScript execution
- Handle dynamic content
- Cookie and session management

**Integration**:
- MCP: Microsoft Playwright MCP
- Headless browser automation
- Screenshot capture

**Features**:
- Intelligent retry and error handling
- Metadata caching
- Proactive suggestions

#### **WEB SCRAPER AGENT** (connectors/scraper_agent.py)
**Purpose**: Advanced web scraping and data extraction

**Capabilities**:
- Crawl entire websites or single pages
- Extract structured data with AI
- Handle JavaScript-rendered content
- Convert pages to Markdown, HTML, or JSON
- Document analysis
- Content extraction

**Integration**:
- MCP: Firecrawl MCP
- Authentication: Firecrawl API keys
- Retry logic: 3 retries with exponential backoff

**Supported Formats**: Markdown, HTML, Structured JSON

---

### 4.3 AGENT INTELLIGENCE COMPONENTS (connectors/agent_intelligence.py)

**Shared Infrastructure** used by all agents:

#### **ConversationMemory**
- Remember recent operations (10-item history default)
- Resolve ambiguous references ("it", "that", "the issue")
- Track context across operations
- Natural conversation flow

#### **WorkspaceKnowledge**
- Persistent learning about workspace
- Entity relationships
- User patterns
- Saved configurations

#### **SharedContext**
- Cross-agent coordination
- Session-wide state
- Shared data structures
- Context propagation

#### **ProactiveAssistant**
- Suggest next steps
- Validate operations before execution
- Identify potential issues
- Recommend alternatives

---

## 5. CORE SERVICES (Infrastructure & Quality)

### 5.1 ERROR HANDLING (core/error_handler.py)

**Error Classification System**:

**Categories**:
- TRANSIENT: Temporary failures (retry with backoff)
- RATE_LIMIT: API rate limited (retry with longer delay)
- CAPABILITY: API doesn't support operation (don't retry)
- PERMISSION: Access denied (require user action)
- VALIDATION: Invalid input (don't retry)
- UNKNOWN: Unknown error (assume retryable)

**ErrorClassifier Features**:
- Pattern-based classification using regex
- Root cause analysis
- User-friendly explanations
- Actionable suggestions
- Retry delay recommendations
- Similar alternative suggestions

**Methods**:
- `classify_error(error_message) → ErrorClassification`

### 5.2 RETRY MANAGER (core/retry_manager.py)

**Strategy**: Intelligent exponential backoff with jitter

**Configuration**:
- Max retries: 3 (configurable)
- Initial delay: 1 second
- Max delay: 30 seconds
- Backoff factor: 2.0
- Jitter: Randomization to prevent thundering herd

**Features**:
- Smart exponential backoff
- Retry budget tracking
- Progress callbacks for UI
- Learning from past failures
- Error classification integration

**Methods**:
- `execute_with_retry(operation_key, agent_name, instruction, operation) → Any`
- `get_retry_history(operation_key) → RetryContext`

### 5.3 CIRCUIT BREAKER (core/circuit_breaker.py)

**Pattern**: Classic three-state machine to prevent cascading failures

**States**:
- CLOSED: Normal operation, requests flow through
- OPEN: Agent failing, block all requests (prevents hammering)
- HALF_OPEN: Testing recovery, allow limited requests

**State Transitions**:
- CLOSED → OPEN: After N consecutive failures (default: 5)
- OPEN → HALF_OPEN: After timeout period (default: 5 minutes)
- HALF_OPEN → CLOSED: After N consecutive successes (default: 2)
- HALF_OPEN → OPEN: If any request fails

**Configuration**:
```
- failure_threshold: 5 (consecutive failures to open)
- success_threshold: 2 (consecutive successes to close)
- timeout_seconds: 300.0 (wait before testing recovery)
- half_open_timeout: 10.0 (max time in half-open state)
```

**Per-Agent Tracking**:
- Separate circuit state for each agent
- Detailed statistics and state history

### 5.4 INPUT VALIDATOR (core/input_validator.py)

**Security Features**:
- Length validation (configurable limits)
- Null byte detection
- Special character limit (prevents injection)
- Regex pattern validation
- SQL/command injection prevention

**Validates**:
- Instructions (max 10,000 chars)
- Parameters (max 5,000 chars)
- Regex patterns (max 1,000 chars)

**Methods**:
- `validate_instruction(instruction) → (bool, optional error)`
- `validate_parameter(param_name, param_value) → (bool, optional error)`
- `validate_regex_pattern(pattern) → (bool, optional error)`
- `sanitize_for_regex(text) → str`

### 5.5 ERROR MESSAGING (core/error_messaging.py)

**Purpose**: Transform technical errors into helpful, actionable messages

**EnhancedError Structure**:
```
- agent_name: which agent failed
- error_type: category (not_found, permission, validation, etc.)
- what_failed: user-friendly description
- why_failed: root cause explanation
- suggestions: 1-3 actionable fixes
- alternatives: similar options found
```

**Methods**:
- `enhance_jira_error(error, context) → EnhancedError`
- `enhance_github_error(error, context) → EnhancedError`
- `enhance_slack_error(error, context) → EnhancedError`

### 5.6 UNDO MANAGER (core/undo_manager.py)

**Supported Operations**:
- JIRA_DELETE_ISSUE, JIRA_CLOSE_ISSUE, JIRA_TRANSITION
- SLACK_DELETE_MESSAGE, SLACK_ARCHIVE_CHANNEL
- GITHUB_CLOSE_PR, GITHUB_CLOSE_ISSUE, GITHUB_DELETE_BRANCH
- NOTION_DELETE_PAGE, NOTION_ARCHIVE_PAGE
- CUSTOM (user-defined)

**Features**:
- 1-hour TTL by default (configurable)
- 20-item history per session
- Undo handlers per operation type
- State snapshots
- Verification before execution

**Methods**:
- `record_operation(operation_type, agent_name, description, before_state, undo_params)`
- `undo(operation_id) → str`
- `register_undo_handler(operation_type, async_handler)`
- `get_recent_operations(n=5) → List[UndoSnapshot]`

### 5.7 USER PREFERENCES (core/user_preferences.py)

**Learning Dimensions**:

**Agent Preferences**:
- Preferred agent for specific task patterns
- Usage history
- Confidence scoring

**Communication Style**:
- Verbose vs. concise
- Technical vs. simplified
- Emoji usage preference

**Working Hours**:
- Typical start/end times
- Active days
- Timezone offset

**Methods**:
- `record_usage(task_pattern, agent_name, success)`
- `get_recommended_agent(task_pattern) → (agent_name, confidence)`
- `update_communication_style(verbose, technical, emojis)`
- `load_from_file(path)`, `save_to_file(path)`

### 5.8 ANALYTICS COLLECTOR (core/analytics.py)

**Metrics Tracked**:

**Agent Metrics**:
- total_calls, successful_calls, failed_calls
- success_rate, failure_rate
- total_latency_ms, avg_latency_ms
- Percentiles: p50, p95, p99 latency
- Error counts by type

**Session Metrics**:
- session_id, duration_seconds
- user_messages, agent_calls
- errors_encountered
- successful_operations

**Methods**:
- `record_agent_call(agent_name, latency_ms, success, error=None)`
- `get_agent_metrics(agent_name) → AgentMetrics`
- `get_session_metrics() → SessionMetrics`
- `generate_summary() → str`

### 5.9 OBSERVABILITY SYSTEM (core/observability.py)

**Components**:

1. **Distributed Tracing** (core/distributed_tracing.py)
   - OpenTelemetry-inspired tracing
   - Trace and Span ID generation
   - Parent-child span relationships
   - Span attributes and events
   - Automatic timing
   - Context propagation

2. **Structured Logging** (core/logging_config.py)
   - Per-module log levels
   - JSON format support
   - Console and file logging
   - Color-coded output
   - Performance tracking context

3. **Orchestration Logger** (core/orchestration_logger.py)
   - Agent state tracking
   - Task assignment logging
   - Orchestration flow logging
   - Task execution timeline

4. **Intelligence Logger** (core/intelligence_logger.py)
   - Intent classification logging
   - Entity extraction logging
   - Task decomposition logging
   - Decision tracking

5. **Metrics Aggregation** (core/metrics_aggregator.py)
   - Counter, Gauge, Histogram, Timer metrics
   - Percentile calculations
   - Time-series data
   - Metric export (JSON, Prometheus)

**Initialization**:
```python
initialize_observability(
    session_id="...",
    service_name="orchestrator",
    log_level="INFO",
    enable_tracing=True,
    enable_metrics=True
)
```

---

## 6. TOOL MANAGEMENT

### 6.1 INTELLIGENT TOOL MANAGER (connectors/tool_manager.py)

**Responsibilities**:
1. Analyze user intent to determine needed connectors
2. Dynamically load/unload tools based on conversation
3. Inject domain-specific context
4. Manage connector lifecycle
5. Optimize token usage by loading only relevant tools

**Strategy**:
- Start with minimal/most common tools (Slack, Jira)
- Expand tool set based on detected intent
- Inject domain expertise as needed
- Unload unused tools to save tokens

**Methods**:
- `analyze_and_load_tools(user_message, conversation_context) → (tools, system_instructions)`
- `_detect_needed_connectors(user_message, context) → Set[str]`

**Core Connectors** (always loaded):
- slack
- jira

---

## 7. MCP (Model Context Protocol) CONFIGURATION

### 7.1 MCP CONFIGURATION (connectors/mcp_config.py)

**Standardized Configuration** for all MCP-based agents:

**Timeouts**:
```
- Initial connection: 30 seconds
- Session initialization: 10 seconds
- Tool listing: 10 seconds
- Tool execution: 60 seconds
- Search operations: 45 seconds
- CRUD operations: 20-30 seconds each
```

**Retry Configuration**:
```
- Max retries: 3
- Initial delay: 1 second
- Max delay: 10 seconds
- Backoff factor: 2.0
```

**Retryable Error Patterns**:
- Network: timeout, connection, ECONNRESET, ETIMEDOUT, ECONNREFUSED
- SSE (Server-Sent Events): sse error, body timeout, stream closed
- HTTP: 502, 503, 504, 429
- Rate limiting: rate_limited, quota exceeded

---

## 8. DATA FLOW PATTERNS

### 8.1 MESSAGE PROCESSING FLOW

```
┌─────────────────────────────────────────────────────────────────────┐
│ USER MESSAGE                                                          │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ INPUT VALIDATION                                                      │
│ - Length check, null bytes, special characters                       │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ CONTEXT MANAGER                                                       │
│ - Add to conversation history                                        │
│ - Update entity tracking                                             │
│ - Resolve coreferences                                               │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ HYBRID INTELLIGENCE SYSTEM                                            │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ TIER 1: Fast Keyword Filter (~10ms)                            │ │
│ │ - Pattern matching                                             │ │
│ │ - High confidence? → RETURN with intents & entities            │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                              ↓ (if low confidence)                   │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ TIER 2: LLM Classifier (~200ms, with caching ~20ms)            │ │
│ │ - Semantic understanding                                       │ │
│ │ - Deep analysis                                                │ │
│ │ - RETURN intents & entities with confidence                    │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ OUTPUT: HybridIntelligenceResult                                     │
│ - intents: List[Intent]                                             │
│ - entities: List[Entity]                                            │
│ - confidence: float                                                 │
│ - path_used: 'fast' or 'llm'                                        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ CONFIDENCE SCORER                                                     │
│ - Score overall confidence in understanding                          │
│ - Return: Confidence with decision (PROCEED, CONFIRM, CLARIFY)       │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ TASK DECOMPOSER                                                       │
│ - Break into subtasks                                                │
│ - Detect dependencies                                                │
│ - Optimize execution order                                           │
│ - Return: ExecutionPlan with Task objects                            │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ INTELLIGENT TOOL MANAGER                                              │
│ - Analyze needed connectors                                          │
│ - Load relevant tools dynamically                                    │
│ - Inject domain expertise                                            │
│ - Return: (Tool definitions, System instructions)                    │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ ORCHESTRATOR AGENT SELECTION & EXECUTION                              │
│ - Route to best-fit agent(s)                                         │
│ - For each task:                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ PRE-EXECUTION:                                              │   │
│   │ - Check circuit breaker status                             │   │
│   │ - Check agent health                                       │   │
│   │ - Log agent selection                                      │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                        │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ EXECUTION WITH RETRY (RetryManager):                        │   │
│   │ - Execute operation                                         │   │
│   │ - On failure: Classify error                                │   │
│   │ - If retryable: Exponential backoff & retry                 │   │
│   │ - Max 3 attempts                                            │   │
│   │ - Progress callbacks for UI                                 │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                        │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ POST-EXECUTION:                                             │   │
│   │ - Record result in analytics                                │   │
│   │ - Update circuit breaker state                              │   │
│   │ - Log task completion                                       │   │
│   │ - Record in undo history (if applicable)                    │   │
│   │ - Update user preferences                                   │   │
│   └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ RESULT AGGREGATION                                                    │
│ - Combine results from multiple agents                               │
│ - Extract key information                                            │
│ - Format for user comprehension                                      │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ USER RESPONSE                                                         │
│ - Natural language summary                                           │
│ - Formatted results                                                  │
│ - Next step suggestions                                              │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 AGENT EXECUTION FLOW

```
ORCHESTRATOR CALLS AGENT
        ↓
1. INPUT VALIDATION
   - Check instruction length
   - Sanitize for security
        ↓
2. CIRCUIT BREAKER CHECK
   - Is agent's circuit open?
   - If OPEN → return error
   - If HALF_OPEN → record attempt
        ↓
3. RETRY MANAGER SETUP
   - Create operation key
   - Setup error classifier
   - Setup retry callback
        ↓
4. EXECUTE OPERATION
   - Call agent.execute(instruction, context)
   - Measure latency
   - Capture result
        ↓
5. ERROR HANDLING
   - Classify error (if failed)
   - Is retryable?
   - Exponential backoff
   - Retry up to 3 times
        ↓
6. RECORD RESULTS
   - Circuit breaker state update
   - Analytics recording
   - Session logging
   - Preference learning
        ↓
7. RETURN RESULT TO ORCHESTRATOR
   - Result string or error message
```

### 8.3 INTELLIGENCE PROCESSING FLOW

```
MESSAGE: "Create a Jira issue for the bug found in my code, assign to John, and notify the team in Slack"

STEP 1: CONTEXT TRACKING
- Add to conversation history
- Track entities: {ISSUE: none yet, PERSON: John, CHANNEL: team}

STEP 2: HYBRID INTELLIGENCE
Path: Fast filter recognizes "Create", "Jira", "assign"
Intents: [CREATE, COORDINATE]
Entities: [ISSUE(type=bug), PERSON(name=John), CHANNEL(name=team)]
Confidence: 0.92

STEP 3: CONFIDENCE SCORING
- Intent clarity: 0.95 (clear what to do)
- Entity completeness: 0.85 (have most needed info)
- Overall: 0.90 → PROCEED (no confirmation needed)

STEP 4: TASK DECOMPOSITION
Task 1: Create Jira issue
  - Input: bug description
  - Output: issue_key (e.g., "KAN-123")
  - Agent: jira

Task 2: Assign to John
  - Input: issue_key, assignee
  - Dependencies: [Task 1]
  - Agent: jira

Task 3: Notify Slack
  - Input: message_text with issue_key
  - Dependencies: [Task 1]
  - Agent: slack

STEP 5: EXECUTION PLAN
[Task 1] → [Task 2, Task 3] (Task 2&3 can run in parallel after Task 1)

STEP 6: TOOL MANAGER
Needed connectors: jira, slack
Load tools for both
Inject domain expertise for each

STEP 7: EXECUTE TASKS
Task 1: jira.execute("Create issue for bug...") → "Created KAN-123"
Task 2: jira.execute("Assign KAN-123 to John") → "Success"
Task 3: slack.execute("Notify team about KAN-123") → "Message posted"

STEP 8: RESULT
"Created issue KAN-123 (assigned to John) and notified the team in Slack"
```

---

## 9. SERVICE DEPENDENCIES MAP

```
ORCHESTRATOR (Center)
├── INTELLIGENCE SYSTEM
│   ├── Hybrid Intelligence
│   │   ├── Fast Keyword Filter
│   │   └── LLM Classifier (Gemini Flash)
│   │       └── LLM Abstraction
│   ├── Intent Classifier
│   ├── Entity Extractor
│   ├── Task Decomposer
│   ├── Confidence Scorer
│   ├── Context Manager
│   └── Cache Layer
│
├── CORE SERVICES
│   ├── Error Handler
│   ├── Input Validator
│   ├── Retry Manager
│   │   └── Error Classifier
│   ├── Circuit Breaker
│   ├── Error Message Enhancer
│   ├── Undo Manager
│   ├── Analytics Collector
│   ├── User Preferences
│   └── Observability System
│       ├── Distributed Tracing
│       ├── Logging Config
│       ├── Orchestration Logger
│       ├── Intelligence Logger
│       └── Metrics Aggregator
│
├── AGENT INFRASTRUCTURE
│   ├── Tool Manager
│   ├── MCP Configuration
│   └── Agent Intelligence Components
│       ├── Conversation Memory
│       ├── Workspace Knowledge
│       ├── Shared Context
│       └── Proactive Assistant
│
└── SPECIALIZED AGENTS (9 agents)
    ├── Jira Agent (MCP)
    ├── Slack Agent (MCP)
    ├── GitHub Agent (MCP)
    ├── Notion Agent (MCP)
    ├── Google Calendar Agent (MCP)
    ├── Browser Agent (MCP - Playwright)
    ├── Web Scraper Agent (MCP - Firecrawl)
    ├── Code Reviewer Agent (LLM-based)
    └── Base Agent (Abstract)

EXTERNAL INTEGRATIONS:
├── Google Gemini API (LLM)
├── Jira Cloud API (via MCP)
├── Slack API (via MCP)
├── GitHub API (via MCP)
├── Notion API (via MCP)
├── Google Calendar API (via MCP)
├── Microsoft Playwright (via MCP)
├── Firecrawl (via MCP)
└── .env Configuration
```

---

## 10. ERROR HANDLING PATTERNS

### 10.1 Error Classification Strategy

```
Error Message
        ↓
ErrorClassifier.classify_error()
        ↓
Pattern Matching
├── Capability Patterns: "does not support", "cannot fetch"
├── Permission Patterns: "forbidden", "unauthorized", "401", "403"
├── Rate Limit Patterns: "too many requests", "rate limit"
├── Validation Patterns: "invalid", "malformed"
├── Network Patterns: "timeout", "connection", "network"
└── Unknown → assume retryable
        ↓
ErrorClassification
├── category: ErrorCategory enum
├── is_retryable: bool
├── explanation: str (user-friendly)
├── suggestions: List[str] (actionable fixes)
└── retry_delay_seconds: int
        ↓
Decision
├── If retryable → RetryManager handles retry
├── If not retryable → Stop and report error
└── If transient → Quick exponential backoff
```

### 10.2 Retry Strategy

```
OPERATION FAILS
        ↓
ErrorClassifier.classify()
        ↓
is_retryable?
├── YES → Continue to retry logic
└── NO → Stop, report error immediately
        ↓
RetryManager.execute_with_retry()
├── Calculate backoff delay: backoff_factor^attempt + jitter
├── Delay: min(initial_delay * backoff, max_delay)
├── Attempt < max_retries?
│   ├── YES → Retry with new delay
│   │   └── Repeat classification
│   └── NO → Return error
└── Success? → Return result
```

### 10.3 Circuit Breaker Pattern

```
Agent Request
        ↓
CircuitBreaker.can_execute(agent_name)?
        ↓
Check State
├── CLOSED (normal operation)
│   ├── Execute request
│   ├── Success? → Count success, continue
│   └── Failure? → Increment failure_count
│       ├── failure_count >= threshold?
│       │   └── Open circuit → OPEN state
│       └── Continue
├── OPEN (preventing hammering)
│   ├── Reject request immediately
│   ├── Wait timeout_seconds
│   └── Transition to HALF_OPEN
└── HALF_OPEN (testing recovery)
    ├── Allow limited requests
    ├── Success? → success_count++
    │   └── success_count >= threshold?
    │       └── Close circuit → CLOSED state
    └── Failure? → Reopen circuit → OPEN state
```

---

## 11. CONFIGURATION & ENVIRONMENT

### 11.1 Environment Variables (from config.py)

```
TIMEOUTS:
- AGENT_TIMEOUT (default: 120.0s)
- ENRICHMENT_TIMEOUT (default: 5.0s)
- LLM_TIMEOUT (default: 30.0s)

RETRY:
- MAX_RETRIES (default: 3)
- RETRY_BACKOFF (default: 2.0)
- INITIAL_RETRY_DELAY (default: 1.0s)

INPUT VALIDATION:
- MAX_INSTRUCTION_LENGTH (default: 10,000)
- MAX_PARAMETER_VALUE_LENGTH (default: 5,000)
- ENABLE_INPUT_SANITIZATION (default: true)
- MAX_REGEX_PATTERN_LENGTH (default: 1,000)

ENRICHMENT:
- REQUIRE_ENRICHMENT_FOR_HIGH_RISK (default: true)
- FAIL_OPEN_ON_ENRICHMENT_ERROR (default: false)

LOGGING:
- LOG_LEVEL (default: INFO)
- LOG_DIR (default: logs/)
- ENABLE_FILE_LOGGING (default: true)
- ENABLE_JSON_LOGGING (default: true)
- ENABLE_CONSOLE_LOGGING (default: true)
- ENABLE_COLORED_LOGS (default: true)
- MAX_LOG_FILE_SIZE_MB (default: 10)
- LOG_BACKUP_COUNT (default: 5)
- VERBOSE (default: false)

PER-MODULE LOG LEVELS:
- LOG_LEVEL_ORCHESTRATOR
- LOG_LEVEL_SLACK
- LOG_LEVEL_JIRA
- LOG_LEVEL_GITHUB
- LOG_LEVEL_NOTION
- LOG_LEVEL_ERROR_HANDLER
- LOG_LEVEL_INTELLIGENCE

GOOGLE API:
- GOOGLE_API_KEY (required)

USER ID:
- USER_ID (default: "default")
```

---

## 12. AI/ML & LLM INTEGRATIONS SUMMARY

### 12.1 LLM Usage

**Provider**: Google Gemini 2.5 Flash

**Used For**:
1. **Tier 2 Intent Classification** (Hybrid Intelligence System)
   - Semantic understanding of complex requests
   - Disambiguation of ambiguous requests
   - Cost: ~$0.01/1K requests
   - Latency: ~200ms (20ms with cache)

2. **Code Reviewer Agent**
   - Code quality analysis
   - Security vulnerability detection
   - Performance optimization recommendations
   - Pure analysis (no external API calls)

3. **Optional in Other Agents**
   - Some agents support LLM enhancement
   - Can be disabled for cost savings

### 12.2 AI/ML Capabilities

1. **Intent Understanding**
   - 7 primary intent types
   - Multi-intent detection
   - Implicit requirement detection
   - Confidence scoring with Bayesian estimation

2. **Entity Recognition**
   - 10+ entity types
   - Multi-pass extraction (regex → NER → relationships)
   - Named Entity Recognition (NER)
   - Relationship extraction

3. **Task Intelligence**
   - Dependency detection
   - Execution order optimization
   - Data flow tracking
   - Parallelization identification

4. **Contextual Understanding**
   - Multi-turn conversation tracking
   - Coreference resolution
   - Entity tracking across messages
   - Pattern learning

5. **Confidence Estimation**
   - Bayesian probability combination
   - Multi-factor scoring
   - Uncertainty quantification
   - Decision recommendations (PROCEED, CONFIRM, CLARIFY)

---

## 13. SERVICE INTERACTION PATTERNS

### 13.1 Synchronization Points

**Sequential Execution Points**:
1. Task 1 completion → Extract results → Pass to Task 2
2. Error classification → Decide retry strategy
3. Circuit breaker state change → Update health status
4. Agent failure → Update analytics → Update preferences

**Parallel Execution Points**:
- Multiple agent calls (when no dependencies)
- Task 2 & Task 3 can run simultaneously
- Metric recording (separate thread)
- Async logging

### 13.2 Data Consistency

**Session State**:
- Maintained in orchestrator memory
- Persisted to session logger (2 files)
- Conversation history tracked in context manager
- User preferences persisted to disk

**Agent State**:
- Per-agent circuit breaker state
- Per-agent health status
- Per-agent analytics metrics
- Per-agent operation history

### 13.3 Failure Propagation

```
Agent Failure
        ↓
Error logged to session
        ↓
ErrorClassifier analyzes
        ↓
If retryable → RetryManager handles
        ↓
Circuit breaker updated
        ↓
Analytics recorded
        ↓
If final failure → User notified
        ↓
Undo handler available (if applicable)
```

---

## 14. PERFORMANCE CHARACTERISTICS

### 14.1 Latency Profile

```
User Input → Response Timeline:

1. Input Validation: ~1ms
2. Context Manager: ~2ms
3. Hybrid Intelligence:
   - Fast Path (35-40%): ~10ms
   - LLM Path (60-65%): ~200ms (or ~20ms with cache)
   - Average: ~80ms
4. Confidence Scoring: ~2ms
5. Task Decomposition: ~3ms
6. Tool Manager: ~5ms
7. Agent Execution: 100-10,000ms (depends on operation)
8. Result Aggregation: ~5ms

Total P50: ~200-300ms (without agent execution)
Total P95: ~500-1000ms (with agent execution)
Total P99: ~2000-5000ms (slow agent operations)
```

### 14.2 Resource Usage

**Memory**:
- Conversation history: ~10 turns (in context manager)
- Cache: 1000 entries LRU (intelligence cache)
- Circuit breaker state: ~100 bytes per agent (9 agents)
- Undo history: 20 snapshots per session

**Network**:
- Hybrid intelligence: 1-2 API calls per message (1 to Gemini)
- Agent operations: 1+ calls per agent depending on task
- MCP connections: 1 persistent connection per agent

**CPU**:
- Fast filter: <1% on modern CPU
- Cache lookup: O(1) hash table
- Regex matching: Linear in message length
- JSON parsing: Linear in response size

---

## 15. SUMMARY: KEY TAKEAWAYS

### Services Overview
- **9 Specialized Agents**: Jira, Slack, GitHub, Notion, Google Calendar, Browser, Scraper, Code Reviewer, Base Agent
- **2-Tier Intelligence**: Fast filtering (10ms) + LLM (200ms) for 92% accuracy at 80ms avg latency
- **10+ Core Services**: Error handling, retry, circuit breaker, undo, analytics, preferences, observability
- **Comprehensive Resilience**: Retry manager, circuit breaker, error classification, fallback strategies
- **Deep Context Awareness**: Multi-turn conversation tracking, entity tracking, coreference resolution

### External Integrations
- **LLM**: Google Gemini 2.5 Flash
- **Project Management**: Jira Cloud API
- **Communication**: Slack API
- **Development**: GitHub API
- **Knowledge**: Notion API
- **Calendar**: Google Calendar API
- **Web**: Playwright (browser), Firecrawl (scraper)

### Data Flow Highlights
- **Message → Intelligence → Task Decomposition → Agent Selection → Execution → Analytics**
- **Error classification drives retry decisions and user messaging**
- **Circuit breaker prevents cascading failures**
- **Observability tracks every operation with distributed tracing**

### Quality Features
- **Production-ready error handling with intelligent recovery**
- **Observability at every layer (logs, traces, metrics)**
- **User learning through preference management**
- **Undo capability for destructive operations**
- **Session persistence for audit trails**
