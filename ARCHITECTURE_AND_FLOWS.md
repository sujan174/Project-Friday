# PROJECT-FRIDAY: ARCHITECTURE AND FLOWS

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE LAYER                              │
│                     (Terminal UI / Enhanced Terminal UI)                     │
└─────────────────────┬───────────────────────────────────────────────────────┘
                      │
                      │ User Input
                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATION LAYER (Main Hub)                       │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ ORCHESTRATOR AGENT                                                  │   │
│  │ - Agent Discovery & Loading                                        │   │
│  │ - Message Routing                                                  │   │
│  │ - Task Execution & Coordination                                    │   │
│  │ - Session Management                                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────┬───────────────────────┬──────────────────────────┬────────────────────┘
      │                       │                          │
      ↓                       ↓                          ↓
┌──────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│ INTELLIGENCE │    │  CORE SERVICES      │    │   LLM SERVICES      │
│  SYSTEM      │    │                     │    │                     │
└──────────────┘    └─────────────────────┘    └─────────────────────┘
      │                       │                          │
      │                       │                          │
      ↓                       ↓                          ↓
┌─────────────────┐  ┌──────────────────────┐  ┌─────────────────────┐
│ Hybrid Intel    │  │ Error Handler        │  │ Base LLM (Abstract) │
│ Fast Filter     │  │ Retry Manager        │  │                     │
│ LLM Classifier  │  │ Circuit Breaker      │  │ Gemini Flash        │
│ Intent Parsing  │  │ Input Validator      │  │ (Google)            │
│ Entity Extract  │  │ Error Messaging      │  └─────────────────────┘
│ Task Decompose  │  │ Undo Manager         │          │
│ Confidence Scor │  │ User Preferences     │          │
│ Context Manager │  │ Analytics Collector  │          │
│ Cache Layer     │  │ Observability System │          │
└────────┬────────┘  │ Distributed Tracing  │          │
         │           │ Logging/Metrics      │          │
         │           └──────────────────────┘          │
         │                                              │
         └──────────────────────┬──────────────────────┘
                                │
         ┌──────────────────────┴──────────────────────┐
         │                                              │
         ↓                                              ↓
    ┌─────────────────────────────┐    ┌──────────────────────────┐
    │   TOOL MANAGER              │    │   AGENT INFRASTRUCTURE   │
    │                             │    │                          │
    │ - Dynamic Tool Loading      │    │ - Conversation Memory    │
    │ - Connector Analysis        │    │ - Workspace Knowledge    │
    │ - Domain Expert Injection   │    │ - Shared Context         │
    │ - Token Optimization        │    │ - Proactive Assistant    │
    │ - MCP Configuration         │    │                          │
    └──────────┬──────────────────┘    └──────────────────────────┘
               │                                │
               └────────────────┬───────────────┘
                                │
                                ↓
                   ┌────────────────────────┐
                   │  SPECIALIZED AGENTS    │
                   │  (9 Agents Total)      │
                   └────────┬───────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ↓                   ↓                   ↓
    ┌────────────┐   ┌────────────┐   ┌─────────────────┐
    │ MCP Agents │   │ LLM Agents │   │  Base/Support   │
    ├────────────┤   ├────────────┤   │                 │
    │ - Jira     │   │ - Code     │   │ - Base Agent    │
    │ - Slack    │   │   Reviewer │   │ - Agent Logger  │
    │ - GitHub   │   └────────────┘   │                 │
    │ - Notion   │                    └─────────────────┘
    │ - Calendar │
    │ - Browser  │
    │ - Scraper  │
    └────────────┘
         │
         ↓
┌──────────────────────────────────────────────┐
│     EXTERNAL SERVICES & APIs                  │
│                                              │
│  LLM:           Google Gemini 2.5 Flash     │
│  Project Mgmt:  Jira Cloud API              │
│  Communication: Slack API                   │
│  Code:          GitHub API                  │
│  Knowledge:     Notion API (SSE)            │
│  Calendar:      Google Calendar API         │
│  Web:           Microsoft Playwright        │
│  Web:           Firecrawl                   │
└──────────────────────────────────────────────┘
```

---

## Intelligence Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ USER MESSAGE: "Create a Jira bug ticket and notify Slack"      │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────┐
        │ INPUT VALIDATION              │
        │ - Length checks               │
        │ - Null bytes detection        │
        │ - Special character limits    │
        │ - Security checks             │
        └──────────────┬────────────────┘
                       │
                       ↓
        ┌──────────────────────────────┐
        │ CONTEXT MANAGER               │
        │ - Add to conversation history │
        │ - Update entity tracking      │
        │ - Track recent operations     │
        │ - Resolve coreferences        │
        └──────────────┬────────────────┘
                       │
                       ↓
    ┌──────────────────────────────────────┐
    │ HYBRID INTELLIGENCE SYSTEM (v5.0)    │
    │                                      │
    │  ┌──────────────────────────────┐  │
    │  │ TIER 1: Fast Keyword Filter   │  │
    │  │ Pattern: "create" + "jira"    │  │
    │  │ Time: <10ms                   │  │
    │  │ Confidence: 0.92              │  │
    │  │ → MATCH! Return intents       │  │
    │  └──────────────────────────────┘  │
    │                                      │
    │  Intents: [CREATE, COORDINATE]     │
    │  Entities: [ISSUE(bug), CHANNEL]   │
    │  Confidence: 0.92                  │
    │  Path: 'fast' (10ms)               │
    └──────────────┬──────────────────────┘
                   │
                   ↓
    ┌──────────────────────────────────┐
    │ CONFIDENCE SCORER                 │
    │ - Intent clarity: 0.95            │
    │ - Entity completeness: 0.85       │
    │ - Message clarity: 0.90           │
    │ Decision: PROCEED (confidence ok) │
    └──────────────┬──────────────────┘
                   │
                   ↓
    ┌──────────────────────────────────┐
    │ TASK DECOMPOSER                   │
    │                                  │
    │ Task 1: Create Jira issue        │
    │   Agent: jira                    │
    │                                  │
    │ Task 2: Notify Slack             │
    │   Agent: slack                   │
    │   Dependencies: [Task 1]          │
    │                                  │
    │ Execution Plan:                  │
    │ [Task 1] → [Task 2]              │
    │ (sequential)                     │
    └──────────────┬──────────────────┘
                   │
                   ↓
    ┌──────────────────────────────────┐
    │ INTELLIGENT TOOL MANAGER          │
    │ - Detect needed connectors        │
    │ - Load: jira, slack               │
    │ - Inject domain expertise         │
    │ - Prepare tool definitions        │
    └──────────────┬──────────────────┘
                   │
                   ↓ (to orchestrator)
         Intelligence Results Ready
```

---

## Agent Execution Flow

```
┌────────────────────────────────────────────┐
│ ORCHESTRATOR: call_sub_agent()             │
│ Agent: jira                                │
│ Instruction: "Create Jira bug..."          │
└────────────┬───────────────────────────────┘
             │
             ↓
┌────────────────────────────────────────────┐
│ 1. CREATE OPERATION KEY                    │
│    MD5(instruction[:100])[:8]              │
│    operation_key: "jira_ab12cd34"          │
└────────────┬───────────────────────────────┘
             │
             ↓
┌────────────────────────────────────────────┐
│ 2. CIRCUIT BREAKER CHECK                   │
│    can_execute("jira")?                    │
│    Status: CLOSED (normal operation)       │
│    → PROCEED                               │
└────────────┬───────────────────────────────┘
             │
             ↓
┌────────────────────────────────────────────┐
│ 3. PREPARE EXECUTION                       │
│    - Log task assignment                   │
│    - Record operation start                │
│    - Setup error classifier                │
│    - Setup retry callback                  │
└────────────┬───────────────────────────────┘
             │
             ↓
┌────────────────────────────────────────────┐
│ 4. RETRY MANAGER EXECUTION LOOP            │
│                                            │
│  Attempt 1:                                │
│  ├─ Execute: jira.execute(instruction)    │
│  ├─ Latency: 2500ms                       │
│  └─ Result: "❌ Network timeout"           │
│     │                                      │
│     ↓ Classify error                       │
│     ErrorClassification:                   │
│     - Category: TRANSIENT (retryable)      │
│     - Retry delay: 1s                      │
│     - Is retryable: true                   │
│                                            │
│  Attempt 2 (after 1s + jitter):           │
│  ├─ Execute: jira.execute(instruction)    │
│  ├─ Latency: 2300ms                       │
│  └─ Result: "✓ Created KAN-123"           │
│     → SUCCESS! Return result               │
│                                            │
└────────────┬───────────────────────────────┘
             │
             ↓
┌────────────────────────────────────────────┐
│ 5. RECORD RESULTS                          │
│    - Update circuit breaker (success)      │
│    - Record analytics (2300ms, success)    │
│    - Log task completion                   │
│    - Record in undo history                │
│    - Update user preferences               │
│    - Publish metrics                       │
└────────────┬───────────────────────────────┘
             │
             ↓
        Result: "✓ Created KAN-123"
```

---

## Error Classification & Recovery

```
┌──────────────────────────────────┐
│ ERROR OCCURS                     │
│ Message: "Connection timeout"    │
└────────────┬─────────────────────┘
             │
             ↓
    ┌─────────────────────┐
    │ ErrorClassifier     │
    │ .classify_error()   │
    └────────┬────────────┘
             │
         ┌───┴───┬──────┬────────┬──────┐
         ↓       ↓      ↓        ↓      ↓
    Pattern Matching:
    - "connection"  → Match!
    - Category: TRANSIENT
         │
         ↓
    ┌──────────────────────────────────┐
    │ ErrorClassification              │
    │ - category: TRANSIENT            │
    │ - is_retryable: true             │
    │ - retry_delay_seconds: 1         │
    │ - explanation: "Temporary network│
    │   connection issue"              │
    │ - suggestions: [                 │
    │     "Check network connectivity",│
    │     "Try again in a few seconds" │
    │   ]                              │
    └────────────┬─────────────────────┘
                 │
                 ↓
         ┌───────────────┐
         │ Decision Tree │
         └───┬───────────┘
             │
         is_retryable?
         ├─ YES → RetryManager handles retry
         │        └─ Exponential backoff
         │        └─ Max 3 attempts
         │        └─ Report progress
         │
         └─ NO → Stop immediately
              └─ Report error to user
              └─ Suggest alternatives

                     ↓

        ┌─────────────────────────┐
        │ USER-FRIENDLY MESSAGE   │
        │                         │
        │ ❌ Jira Connection Error│
        │                         │
        │ What failed:            │
        │ Could not connect to    │
        │ Jira API                │
        │                         │
        │ Why:                    │
        │ Temporary network issue │
        │                         │
        │ How to fix:             │
        │ • Check internet        │
        │ • Try again in a moment │
        │ • Check Jira status     │
        └─────────────────────────┘
```

---

## Circuit Breaker State Machine

```
                         Failure Count = 0
                         Success Count = 0
                         ↓
                    ┌─────────────┐
                    │   CLOSED    │ ← Normal Operation
                    │  (Requests  │   All requests pass through
                    │   allowed)  │
                    └──────┬──────┘
                           │
                    ┌──────┴─────────┐
                    │ Success        │ Failure
                    │ Success++      │ Failure++
                    ↓                │
            Success < 2             Failure >= 5?
            ├─ YES → Stay CLOSED    ├─ YES → OPEN ↓
            └─ NO                   └─ NO → Stay CLOSED
                                            
                                    ┌─────────────┐
                                    │    OPEN     │ ← Circuit Broken
                                    │ (Requests   │   Reject all requests
                                    │  blocked)   │   to prevent hammering
                                    └──────┬──────┘
                                           │
                                    Wait 5 minutes
                                    timeout_seconds
                                    ↓
                                    ┌──────────────┐
                                    │  HALF_OPEN  │ ← Testing Recovery
                                    │ (Limited req)│   Allow test requests
                                    └──────┬───────┘
                                           │
                                    ┌──────┴──────────┐
                                    │ Success         │ Failure
                                    │ Success++       │ Go back OPEN
                                    ↓                │
                                Success >= 2?      Retry later
                                ├─ YES → CLOSED ↓
                                └─ NO → Stay HALF_OPEN

Configuration:
- failure_threshold: 5 (consecutive failures)
- success_threshold: 2 (consecutive successes)
- timeout_seconds: 300 (5 minutes)
- half_open_timeout: 10 (seconds)
```

---

## Observability & Monitoring Architecture

```
┌─────────────────────────────────────────────────────────────┐
│           UNIFIED OBSERVABILITY SYSTEM                      │
└──────────────┬──────────────────────────────────────────────┘
               │
        ┌──────┴──────┬──────────────┬──────────┬──────────┐
        ↓             ↓              ↓          ↓          ↓
    ┌────────┐  ┌────────┐  ┌────────────┐ ┌──────┐ ┌─────────┐
    │Logging │  │Tracing │  │Orchestr.   │ │Intel │ │Metrics  │
    │        │  │        │  │Logger      │ │Logger│ │         │
    └────┬───┘  └───┬────┘  └────┬───────┘ └──┬───┘ └────┬────┘
         │          │            │            │         │
         ↓          ↓            ↓            ↓         ↓
    ┌─────────────────────────────────────────────────────────┐
    │                    TRACES & SPANS                       │
    │                                                         │
    │  Trace: user-message-12345                             │
    │  ├─ Span: input-validation (2ms)                       │
    │  ├─ Span: hybrid-intelligence (80ms)                   │
    │  │  ├─ Span: fast-filter (10ms) - COMPLETED            │
    │  │  └─ Event: HIGH_CONFIDENCE - no LLM needed          │
    │  ├─ Span: confidence-scorer (2ms)                      │
    │  ├─ Span: task-decomposer (3ms)                        │
    │  ├─ Span: tool-manager (5ms)                           │
    │  ├─ Span: jira-agent-execution (2500ms)                │
    │  │  ├─ Event: ATTEMPT_1 - failed                       │
    │  │  ├─ Span: retry-sleep (1000ms)                      │
    │  │  ├─ Event: ATTEMPT_2 - success                      │
    │  │  └─ Tag: agent=jira, operation=create               │
    │  └─ Span: result-aggregation (5ms)                     │
    │                                                         │
    │  Total Duration: ~2600ms                               │
    │  Status: SUCCESS                                       │
    └────────────┬────────────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        ↓                 ↓
    ┌──────────────┐  ┌──────────────┐
    │ JSON Export  │  │ Log Files    │
    │              │  │              │
    │ {            │  │ 2024-11-20   │
    │  "trace_id": │  │ 10:30:45     │
    │  "spans": [] │  │ [INFO] User  │
    │ }            │  │ input        │
    └──────────────┘  └──────────────┘
```

---

## Data Dependencies & State Flow

```
┌─────────────────────────────────────────┐
│ Session State (In Memory)               │
├─────────────────────────────────────────┤
│                                         │
│ Conversation History:                   │
│  - Turn 1: User input                   │
│  - Turn 2: Assistant response           │
│  - Turn 3: User clarification           │
│    └─ Entities tracked across turns     │
│                                         │
│ Circuit Breaker State (Per Agent):      │
│  - jira: CLOSED, failures: 0            │
│  - slack: HALF_OPEN, successes: 1       │
│  - github: OPEN, last_attempt: 5m ago   │
│                                         │
│ Undo History (Last 20 ops):             │
│  - Op 1: created_jira_KAN-123           │
│  - Op 2: assigned_to_john               │
│  - Op 3: sent_slack_message             │
│    └─ All can be undone (TTL < 1h)      │
│                                         │
│ Metrics Snapshot:                       │
│  - jira: success_rate=98%, p95=2.5s     │
│  - slack: success_rate=100%, p95=1.2s   │
│  - github: success_rate=95%, p95=3.1s   │
│                                         │
│ User Preferences:                       │
│  - Preferred_agent(create): jira (0.92) │
│  - Communication: technical=true        │
│  - Working_hours: 9-17 UTC              │
│                                         │
└─────────────────────────────────────────┘
         │                      │
         │ (persisted at end)   │ (loaded at start)
         ↓                      ↓
┌────────────────────────────────────────────┐
│ Persistent Storage                         │
├────────────────────────────────────────────┤
│                                            │
│ Session Logs: logs/{session_id}/           │
│  - messages.txt (human-readable)           │
│  - operations.json (detailed)              │
│                                            │
│ User Preferences: data/preferences/        │
│  - {user_id}.json (learned preferences)    │
│                                            │
│ Traces & Metrics: logs/traces/             │
│  - {session_id}_traces.json                │
│                                            │
└────────────────────────────────────────────┘
```

---

## Service Communication Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                    SYNCHRONOUS CALLS                        │
│                  (Direct function calls)                    │
└─────────────────────────────────────────────────────────────┘

Orchestrator
  ├─→ Input Validator.validate() → bool
  │
  ├─→ Context Manager.add_turn() → None
  │
  ├─→ Hybrid Intelligence.classify() → HybridIntelligenceResult
  │    ├─→ Fast Filter.classify() → (intents, entities, confidence)
  │    ├─→ [if low confidence]
  │    │    └─→ LLM Classifier.classify() → LLMClassificationResult
  │    │         └─→ Gemini Flash.classify() → LLMResponse
  │    │              └─→ Google Gemini API (HTTP)
  │    └─→ Cache Layer.get() / put() → cached results
  │
  ├─→ Confidence Scorer.score() → Confidence
  │
  ├─→ Task Decomposer.decompose() → ExecutionPlan
  │
  ├─→ Tool Manager.analyze_and_load_tools() → (tools, instructions)
  │
  └─→ call_sub_agent(agent_name, instruction)
       │
       ├─→ Circuit Breaker.can_execute() → (bool, reason)
       │
       ├─→ Retry Manager.execute_with_retry()
       │    │
       │    ├─→ Agent.execute(instruction)
       │    │    ├─→ MCP Client.call_tool() → response
       │    │    │    └─→ External API (MCP server)
       │    │    └─→ Return result
       │    │
       │    ├─→ [on failure]
       │    │    └─→ Error Handler.classify() → ErrorClassification
       │    │         ├─→ [if retryable & attempts < max]
       │    │         │    └─→ sleep + retry
       │    │         └─→ [if max attempts or not retryable]
       │    │              └─→ raise exception
       │    │
       │    └─→ Return result (success or final error)
       │
       ├─→ Analytics.record_agent_call() → None
       │
       ├─→ Undo Manager.record_operation() → None
       │
       └─→ User Preferences.record_usage() → None

┌─────────────────────────────────────────────────────────────┐
│                  ASYNCHRONOUS PUBLISHING                    │
│              (Fire-and-forget, background tasks)           │
└─────────────────────────────────────────────────────────────┘

After operation completion:
  ├─→ Metrics Aggregator.publish() [thread-safe]
  │
  ├─→ Session Logger.log_operation() [async write]
  │
  ├─→ Distributed Tracer.export_trace() [batched]
  │
  └─→ Preference Manager.save_to_disk() [async]
```

---

## Failure Recovery Strategies

```
TRANSIENT ERROR (Network Timeout)
  ↓
ErrorClassifier → Category: TRANSIENT, is_retryable: true
  ↓
Retry Manager → Execute with exponential backoff
  ├─ Attempt 1: failed
  ├─ Delay: 1s + jitter
  ├─ Attempt 2: failed
  ├─ Delay: 2s + jitter
  ├─ Attempt 3: SUCCESS ✓
  └─ Return result
                                                        
PERMISSION ERROR (Unauthorized)
  ↓
ErrorClassifier → Category: PERMISSION, is_retryable: false
  ↓
Retry Manager → Skip retries, fail immediately
  ↓
Error Message Enhancer → Generate user-friendly message
  └─ Explain: Need higher permissions
     Suggest: Contact admin to grant access
                                                        
RATE LIMIT ERROR (429 Too Many Requests)
  ↓
ErrorClassifier → Category: RATE_LIMIT, retry_delay: 60s
  ↓
Retry Manager → Single exponential backoff attempt
  ├─ Delay: 60s (much longer than transient)
  └─ Retry with longer wait
                                                        
CAPABILITY ERROR (API doesn't support operation)
  ↓
ErrorClassifier → Category: CAPABILITY, is_retryable: false
  ↓
Retry Manager → Skip, fail immediately
  ↓
Error Message Enhancer → Generate message
  └─ Explain: Operation not supported
     Suggest: Use alternative agent or method
                                                        
CASCADING FAILURE (Agent repeatedly failing)
  ↓
Circuit Breaker → After 5 consecutive failures
  ├─ State: CLOSED → OPEN
  └─ Block all requests to prevent hammering
     ↓
  After 5 minute timeout:
  ├─ State: OPEN → HALF_OPEN
  └─ Allow test requests to check if recovered
     ├─ If success → CLOSED (normal)
     └─ If failure → OPEN (still broken)
```

---

## Performance Optimization Strategies

```
1. CACHING (Hybrid Intelligence)
   ├─ LLM Classifications cached for 5 minutes
   ├─ Cache hit rate: 70-80%
   └─ Savings: ~20ms per cache hit

2. TWO-TIER INTELLIGENCE
   ├─ Fast Filter handles 35-40% of requests (~10ms)
   ├─ LLM handles complex cases (~200ms)
   └─ Average: 80ms (vs 200ms if all LLM)
   └─ Cost: $0.0065/1K (vs $0.01/1K if all LLM)

3. DYNAMIC TOOL LOADING
   ├─ Load only needed connectors per request
   ├─ Reduces token usage in LLM calls
   └─ Faster context injection

4. ASYNC OPERATIONS
   ├─ Logging done async (doesn't block user)
   ├─ Metrics publishing batched
   ├─ Preferences saved async
   └─ Total async overhead: ~5ms

5. PARALLEL EXECUTION (When possible)
   ├─ Multiple independent tasks run concurrently
   │  Example: Task 2 and Task 3 run together
   ├─ Dependency-aware ordering prevents race conditions
   └─ Measured with ExecutionPlan dependency graph

6. CONNECTION POOLING (MCP)
   ├─ Persistent connections to each agent
   ├─ Reuse connections across requests
   └─ Saves handshake time (~500ms per connection)

7. METRICS SAMPLING
   ├─ Track latency percentiles (P50, P95, P99)
   ├─ Avoid full histogram tracking
   └─ Calculate percentiles on demand
```

---

## Deployment Architecture

```
┌──────────────────────────────────────────────┐
│      DOCKER CONTAINERS (Production)          │
├──────────────────────────────────────────────┤
│                                              │
│ Container 1: Project-Friday (Main)           │
│  - orchestrator.py                           │
│  - intelligence system                       │
│  - core services                             │
│  - python:3.11+                              │
│                                              │
│ Container 2: MCP Atlassian (Optional)        │
│  - sooperset/mcp-atlassian                   │
│  - Jira API wrapper                          │
│  - Shared volume: config                     │
│                                              │
│ Container 3: MCP Playwright (Optional)       │
│  - Browser automation via MCP                │
│  - Headless browser instance                 │
│                                              │
│ [Other MCP containers as needed]             │
│                                              │
└──────────────────────────────────────────────┘
         │                    │
         │ Network            │ Volume
         ├── Stdio comm ──────┤ shared configs
         └── Environment var  │
```

---

## Summary: Key Architectural Decisions

| Decision | Rationale | Benefit |
|----------|-----------|---------|
| Two-tier intelligence | Speed vs accuracy tradeoff | 80ms avg vs 200ms |
| Circuit breaker per agent | Isolate failures | Prevent cascading |
| Exponential backoff | Avoid thundering herd | Smooth retry pattern |
| Semantic caching | LLM responses similar | 70-80% cache hit |
| Dynamic tool loading | Token efficiency | Reduces context size |
| Observability everywhere | Debugging & monitoring | Complete visibility |
| Async logging | Non-blocking I/O | Lower latency |
| MCP abstraction | Future flexibility | Easy to add services |
| Session persistence | Audit trail | Accountability |
| Undo capability | User confidence | Reversible operations |

