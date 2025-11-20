# PROJECT-FRIDAY: QUICK REFERENCE GUIDE

## Service Catalog

### Intelligence Services (9 files)
| Service | Purpose | Key Features |
|---------|---------|--------------|
| **Hybrid Intelligence System** | Two-tier intent/entity understanding | Fast filter (10ms) + LLM (200ms), 92% accuracy |
| **Intent Classifier** | Detects user intent | 7 types: CREATE, READ, UPDATE, DELETE, ANALYZE, COORDINATE, SEARCH |
| **Entity Extractor** | Extracts entities | ISSUE, PR, PROJECT, CHANNEL, PERSON, DATE, PRIORITY, FILE |
| **Task Decomposer** | Breaks complex requests | Dependency detection, execution order optimization |
| **Confidence Scorer** | Scores understanding quality | Bayesian estimation, PROCEED/CONFIRM/CLARIFY decisions |
| **Context Manager** | Tracks conversation context | Multi-turn tracking, coreference resolution, entity tracking |
| **Fast Keyword Filter** | Tier 1 (fast) classification | <10ms, pattern matching, 95% accuracy |
| **LLM Classifier** | Tier 2 semantic classification | ~200ms (20ms cached), 92% accuracy, semantic caching |
| **Cache Layer** | Intelligent caching | LRU with TTL, 70-80% hit rate on LLM |

### LLM Services (2 files)
| Service | Purpose | Provider |
|---------|---------|----------|
| **Base LLM** | Provider-agnostic abstraction | Abstract interface |
| **Gemini Flash** | Production LLM implementation | Google Gemini 2.5 Flash |

### Agent Services (9 agents)
| Agent | Platform | Purpose | Key Integration |
|-------|----------|---------|-----------------|
| **Jira Agent** | Jira Cloud | Issue tracking, workflow management | MCP (sooperset/mcp-atlassian) |
| **Slack Agent** | Slack | Team communication | MCP (Slack API) |
| **GitHub Agent** | GitHub | Code collaboration, development | MCP (GitHub API) |
| **Notion Agent** | Notion | Knowledge management, documentation | MCP (Notion API, SSE-based) |
| **Google Calendar Agent** | Google Calendar | Calendar management, scheduling | MCP (Google Calendar API) |
| **Browser Agent** | Web | Browser automation | MCP (Microsoft Playwright) |
| **Web Scraper Agent** | Web | Web scraping, data extraction | MCP (Firecrawl) |
| **Code Reviewer Agent** | Multi-source | Code quality analysis | LLM-based (pure analysis) |
| **Base Agent** | Abstract | Foundation for all agents | Abstract base class |

### Core Services (10 files)
| Service | Purpose | Key Feature |
|---------|---------|------------|
| **Retry Manager** | Smart retry logic | Exponential backoff, jitter, error classification |
| **Circuit Breaker** | Prevent cascading failures | 3-state machine (CLOSED, OPEN, HALF_OPEN) |
| **Error Handler** | Classify errors | 6 categories, actionable suggestions |
| **Input Validator** | Security & validation | Length checks, injection prevention |
| **Error Messaging** | User-friendly errors | Root cause + suggestions + alternatives |
| **Undo Manager** | Reversible operations | 1-hour TTL, 20-item history per session |
| **User Preferences** | Behavioral learning | Agent preferences, communication style, working hours |
| **Analytics Collector** | Performance metrics | Agent metrics, session metrics, percentiles |
| **Observability System** | Tracing & monitoring | Distributed tracing, structured logging, metrics |
| **Tool Manager** | Dynamic tool loading | Intent-aware connector activation |

### Supporting Services (3 files)
| Service | Purpose |
|---------|---------|
| **Agent Intelligence** | Conversation memory, workspace knowledge, shared context, proactive assistant |
| **MCP Configuration** | Standardized timeouts and retry config for all MCP agents |
| **Distributed Tracing** | OpenTelemetry-inspired tracing with span tracking |

---

## External Integrations

### LLM APIs
- **Google Gemini 2.5 Flash** → Intent/entity classification, code review

### Workspace APIs (via MCP)
- **Jira Cloud** → Issue management
- **Slack** → Team communication
- **GitHub** → Code collaboration
- **Notion** → Knowledge management
- **Google Calendar** → Calendar/scheduling

### Web Services (via MCP)
- **Microsoft Playwright** → Browser automation
- **Firecrawl** → Web scraping

---

## Data Flow Summary

```
User Input
  ↓ (validate)
Conversation Context
  ↓ (track)
Hybrid Intelligence (Intent + Entity)
  ↓ (score)
Confidence Assessment
  ↓ (decompose)
Task Plan with Dependencies
  ↓ (select tools)
Agent Execution (with retry + circuit breaker)
  ↓ (track)
Analytics + Undo History + Preferences
  ↓ (aggregate)
User Response
```

---

## Key Patterns

### Resilience
- **Retry Manager**: 3 attempts with exponential backoff (1s → 30s)
- **Circuit Breaker**: Stops hammering failing agents
- **Error Classification**: Smart retry decisions based on error type
- **Undo System**: Reversible operations with 1-hour window

### Performance
- **Hybrid Intelligence**: 80ms avg latency (35% fast path, 65% LLM)
- **Caching**: 70-80% hit rate on LLM classifications
- **Tool Manager**: Dynamic loading saves tokens
- **Percentile Tracking**: P50, P95, P99 latencies per agent

### Context Awareness
- **Conversation Memory**: Multi-turn tracking with coreference resolution
- **Entity Tracking**: Mentions count, relationships
- **Temporal Context**: Project, repository, branch awareness
- **Pattern Learning**: User behavior understanding

### Quality
- **Observability**: Distributed tracing + structured logging
- **User Learning**: Preference manager learns from behavior
- **Error Recovery**: Multi-level fallback strategies
- **Analytics**: Real-time performance metrics

---

## Configuration

### Timeouts
- Agent operations: 120 seconds
- LLM operations: 30 seconds
- Enrichment: 5 seconds
- MCP operations: 20-60 seconds per type

### Retries
- Max attempts: 3
- Backoff factor: 2.0
- Initial delay: 1 second
- Max delay: 30 seconds

### Circuit Breaker
- Failure threshold: 5 consecutive failures
- Success threshold: 2 consecutive successes
- Timeout before recovery: 300 seconds (5 minutes)

### Input Limits
- Instruction length: 10,000 characters
- Parameter length: 5,000 characters
- Regex pattern length: 1,000 characters

### Cache
- Max size: 1000 entries
- Default TTL: 5 minutes
- Eviction: LRU policy

---

## Service Dependencies

```
ORCHESTRATOR (Hub)
├── Intelligence System
├── Core Services (Retry, Circuit Breaker, Error, Analytics, etc.)
├── LLM Abstraction
├── Tool Manager
└── 9 Specialized Agents
    └── Agent Intelligence Components
    └── MCP Connections
```

**Key Dependency**: Every service depends on error_handler for consistent error classification.

---

## API Endpoints & Methods (Key Functions)

### Orchestrator
- `process_message(user_message)` → str
- `call_sub_agent(agent_name, instruction, context)` → str
- `discover_and_load_agents()` → None

### Hybrid Intelligence
- `classify_intent(message, context)` → HybridIntelligenceResult

### Error Handling
- `ErrorClassifier.classify_error(error)` → ErrorClassification
- `RetryManager.execute_with_retry(operation_key, agent_name, instruction, operation)` → Any
- `CircuitBreaker.can_execute(agent_name)` → (bool, str)

### Analytics
- `AnalyticsCollector.record_agent_call(agent_name, latency_ms, success)` → None
- `AnalyticsCollector.get_agent_metrics(agent_name)` → AgentMetrics

### Preferences
- `UserPreferenceManager.record_usage(task_pattern, agent_name, success)` → None
- `UserPreferenceManager.get_recommended_agent(task_pattern)` → (str, float)

---

## Logging & Monitoring

### Log Files
- **Session logs**: `logs/{session_id}/messages.txt` (human-readable)
- **Structure**: Conversation flow, agent selections, results

### Traces
- OpenTelemetry-compatible spans with parent-child relationships
- Distributed tracing across agents

### Metrics
- Counter: total requests
- Gauge: active agents
- Histogram: latency distribution
- Timer: operation durations
- Percentiles: P50, P95, P99

---

## File Structure

```
Project-Friday/
├── orchestrator.py (1812 lines) - Main orchestration hub
├── main.py - Entry point with CLI
├── config.py - Configuration management
│
├── intelligence/ (9 services)
│   ├── hybrid_system.py - Two-tier intelligence
│   ├── intent_classifier.py - Intent detection
│   ├── entity_extractor.py - Entity recognition
│   ├── task_decomposer.py - Task planning
│   ├── confidence_scorer.py - Confidence estimation
│   ├── context_manager.py - Context tracking
│   ├── fast_filter.py - Tier 1 (fast)
│   ├── llm_classifier.py - Tier 2 (semantic)
│   ├── cache_layer.py - Intelligent caching
│   └── base_types.py - Type definitions
│
├── llms/ (2 services)
│   ├── base_llm.py - Provider abstraction
│   └── gemini_flash.py - Google Gemini implementation
│
├── connectors/ (9+ agents)
│   ├── base_agent.py - Base class
│   ├── *_agent.py (8 agents)
│   ├── agent_intelligence.py - Shared components
│   ├── tool_manager.py - Dynamic tool loading
│   ├── mcp_config.py - MCP configuration
│   └── base_connector.py - Connector base
│
├── core/ (10+ services)
│   ├── error_handler.py - Error classification
│   ├── retry_manager.py - Intelligent retry
│   ├── circuit_breaker.py - Failure prevention
│   ├── input_validator.py - Security & validation
│   ├── error_messaging.py - User-friendly errors
│   ├── undo_manager.py - Reversible operations
│   ├── user_preferences.py - Behavioral learning
│   ├── analytics.py - Performance metrics
│   ├── observability.py - Unified monitoring
│   ├── distributed_tracing.py - Trace tracking
│   ├── metrics_aggregator.py - Metric collection
│   └── [logging, orchestration, intelligence loggers]
│
├── ui/ - User interface
│   ├── terminal_ui.py - Simple terminal
│   └── enhanced_terminal_ui.py - Rich UI
│
└── tools/ - Utility tools
    └── session_viewer.py - Session analysis
```

---

## Performance Profile

### Latency
- Input validation: 1ms
- Context manager: 2ms
- Hybrid intelligence: 80ms avg (10ms fast, 200ms LLM)
- Confidence scoring: 2ms
- Task decomposition: 3ms
- Tool manager: 5ms
- **Agent execution: 100-10,000ms** (depends on operation)

### Total Response Time
- P50 (without agent): 100-300ms
- P95 (with typical agent): 500-1000ms
- P99 (slow operations): 2000-5000ms

### Resource Efficiency
- Cache hit rate: 70-80%
- Fast path coverage: 35-40%
- LLM cost: $0.0065/1K requests
- Memory: ~100MB baseline

---

## Quick Start Commands

```bash
# Run with enhanced UI
python main.py

# Run with verbose logging
python main.py --verbose

# Run with simple UI
python main.py --simple

# View recent session logs
ls logs/*/messages.txt | tail -1 | xargs cat
```

---

Generated: Project-Friday Services Analysis
**Total Services**: 35+ interconnected services
**Lines of Code**: 56 Python files, ~15,000+ LOC
**External Integrations**: 8+ platforms via MCP + Google API
**Agents**: 9 specialized agents (8 MCP-based, 1 LLM-based)
