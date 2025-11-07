# System Improvements Guide

This document describes the new features added to enhance the Multi-Agent Orchestration System.

## 🎯 Overview

We've added 4 major improvements to make the system more intelligent, user-friendly, and production-ready:

1. **Advanced Retry Management** - Smarter retries with exponential backoff and progress feedback
2. **Undo System** - Ability to undo destructive operations
3. **User Preference Learning** - System learns from user behavior
4. **Analytics & Monitoring** - Comprehensive performance metrics

---

## 1. 🔄 Advanced Retry Management

**Location:** `core/retry_manager.py`

### Features
- ✅ Exponential backoff with jitter (avoids thundering herd)
- ✅ Intelligent retry decisions based on error classification
- ✅ Progress callbacks for UI updates
- ✅ Retry budget management (prevents runaway retries)
- ✅ Per-operation retry tracking

### Usage

```python
from core.retry_manager import RetryManager

# Initialize
retry_manager = RetryManager(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    verbose=True
)

# Execute with retry
async def my_operation():
    return await agent.execute(instruction)

result = await retry_manager.execute_with_retry(
    operation_key="jira_create_123",
    agent_name="jira",
    instruction="Create issue",
    operation=my_operation,
    progress_callback=lambda msg, attempt, max_attempts: print(msg)
)

# Get statistics
stats = retry_manager.get_statistics()
print(f"Total operations: {stats['total_operations']}")
print(f"Success rate: {stats['successful']}/{stats['total_operations']}")
print(f"Average retries: {stats['avg_retries_per_operation']}")
```

### Integration with Orchestrator

Add to `orchestrator.py`:

```python
from core.retry_manager import RetryManager

class OrchestratorAgent:
    def __init__(self, ...):
        # ... existing code ...
        self.retry_manager = RetryManager(
            max_retries=self.max_retry_attempts,
            verbose=self.verbose
        )

    async def call_sub_agent(self, agent_name: str, instruction: str, context: Any = None):
        """Execute with retry management"""
        operation_key = self._get_operation_key(agent_name, instruction)

        async def operation():
            return await self._execute_agent(agent_name, instruction, context)

        return await self.retry_manager.execute_with_retry(
            operation_key=operation_key,
            agent_name=agent_name,
            instruction=instruction,
            operation=operation,
            progress_callback=self._on_retry_progress
        )
```

---

## 2. ↩️ Undo System

**Location:** `core/undo_manager.py`

### Features
- ✅ Record state before destructive operations
- ✅ Time-based undo windows (default 1 hour)
- ✅ Support for multiple operation types
- ✅ Persistent undo history
- ✅ Extensible handler system

### Supported Operations

- **Jira:** Delete/close issues, transitions
- **Slack:** Delete messages, archive channels
- **GitHub:** Close PRs/issues, delete branches
- **Notion:** Delete/archive pages
- **Custom:** Add your own via handlers

### Usage

```python
from core.undo_manager import UndoManager, UndoableOperationType

# Initialize
undo_manager = UndoManager(
    max_undo_history=20,
    default_ttl_seconds=3600,  # 1 hour
    verbose=True
)

# Register undo handlers
async def undo_jira_delete(undo_params: Dict) -> str:
    issue_key = undo_params['issue_key']
    # Restore the issue
    return f"Restored {issue_key}"

undo_manager.register_undo_handler(
    UndoableOperationType.JIRA_DELETE_ISSUE,
    undo_jira_delete
)

# Record a destructive operation
operation_id = undo_manager.record_operation(
    operation_type=UndoableOperationType.JIRA_DELETE_ISSUE,
    agent_name="jira",
    description="Deleted issue ABC-123",
    before_state={"key": "ABC-123", "summary": "Bug fix"},
    undo_params={"issue_key": "ABC-123", "project": "ABC"},
    operation_result="Successfully deleted ABC-123"
)

# List undoable operations
operations = undo_manager.get_undoable_operations(limit=10)
print(undo_manager.format_undo_list())

# Undo an operation
result = await undo_manager.undo_operation(operation_id)
print(result)

# Or undo last operation
last_op_id = undo_manager.undo_last()
if last_op_id:
    await undo_manager.undo_operation(last_op_id)
```

### Integration with Agents

Add to your agent's destructive operations:

```python
class JiraAgent(BaseAgent):
    def __init__(self, ..., undo_manager: Optional[UndoManager] = None):
        self.undo_manager = undo_manager

    async def execute(self, instruction: str) -> str:
        # ... parse instruction ...

        if action == "delete_issue":
            # Record before deleting
            if self.undo_manager:
                operation_id = self.undo_manager.record_operation(
                    operation_type=UndoableOperationType.JIRA_DELETE_ISSUE,
                    agent_name="jira",
                    description=f"Deleted issue {issue_key}",
                    before_state={"key": issue_key, ...},
                    undo_params={"issue_key": issue_key, ...}
                )

            # Perform deletion
            result = await self._delete_issue(issue_key)
            return result
```

---

## 3. 🧠 User Preference Learning

**Location:** `core/user_preferences.py`

### Features
- ✅ Learns confirmation preferences (auto-execute vs always confirm)
- ✅ Learns preferred agents for task types
- ✅ Adapts to communication style (verbose, technical, emojis)
- ✅ Learns working hours patterns
- ✅ Confidence-based recommendations

### Usage

```python
from core.user_preferences import UserPreferenceManager

# Initialize
prefs = UserPreferenceManager(
    user_id="user@example.com",
    min_confidence_threshold=0.7,
    verbose=True
)

# Record user behavior
prefs.record_confirmation_decision("jira_delete", user_confirmed=True)
prefs.record_agent_usage("create_ticket", "jira", was_successful=True)
prefs.record_interaction_style(user_message, user_requested_verbose=False)
prefs.record_interaction_time()

# Check learned preferences
if prefs.should_auto_execute("jira_create"):
    # Auto-execute without confirmation
    result = await agent.execute(instruction)
else:
    # Show confirmation
    if confirm_with_user():
        result = await agent.execute(instruction)

# Get preferred agent
preferred = prefs.get_preferred_agent("create_ticket")
if preferred:
    agent = self.sub_agents[preferred]

# Get communication preferences
style = prefs.get_communication_preferences()
if style['verbose']:
    # Provide detailed explanations
    pass

# Check working hours
if prefs.is_during_working_hours():
    # Send notifications
    pass

# Save preferences
prefs.save_to_file("data/preferences/user123.json")

# Get summary
print(prefs.get_summary())
```

### Integration with Orchestrator

```python
class OrchestratorAgent:
    def __init__(self, ...):
        # ... existing code ...
        self.user_prefs = UserPreferenceManager(
            user_id=os.environ.get("USER_ID", "default"),
            verbose=self.verbose
        )

        # Load existing preferences
        prefs_file = f"data/preferences/{self.user_prefs.user_id}.json"
        if os.path.exists(prefs_file):
            self.user_prefs.load_from_file(prefs_file)

    async def process_message(self, user_message: str) -> str:
        # Record interaction
        self.user_prefs.record_interaction_time()
        self.user_prefs.record_interaction_style(user_message)

        # ... existing logic ...

        # Check if operation needs confirmation
        operation_pattern = f"{agent_name}_{action_type}"

        if self.user_prefs.should_auto_execute(operation_pattern):
            # Auto-execute based on learned preference
            result = await self.call_sub_agent(agent_name, instruction)
            self.user_prefs.record_confirmation_decision(operation_pattern, True)
        else:
            # Show confirmation
            confirmed = self._ask_confirmation(...)
            self.user_prefs.record_confirmation_decision(operation_pattern, confirmed)

            if confirmed:
                result = await self.call_sub_agent(agent_name, instruction)

        # Save preferences periodically
        self.user_prefs.save_to_file(prefs_file)
```

---

## 4. 📊 Analytics & Monitoring

**Location:** `core/analytics.py`

### Features
- ✅ Agent performance tracking (success rate, latency)
- ✅ Latency percentiles (P50, P95, P99)
- ✅ Error classification and tracking
- ✅ Usage patterns (hourly, daily)
- ✅ Operation tracking
- ✅ Health score calculation
- ✅ Session metrics

### Usage

```python
from core.analytics import AnalyticsCollector

# Initialize
analytics = AnalyticsCollector(
    session_id="session_123",
    max_latency_samples=1000,
    verbose=True
)

# Record agent calls
start = time.time()
try:
    result = await agent.execute(instruction)
    latency_ms = (time.time() - start) * 1000
    analytics.record_agent_call("jira", success=True, latency_ms=latency_ms)
except Exception as e:
    latency_ms = (time.time() - start) * 1000
    analytics.record_agent_call("jira", success=False, latency_ms=latency_ms, error_message=str(e))

# Record other events
analytics.record_user_message()
analytics.record_confirmation(accepted=True)
analytics.record_operation("jira_create", success=True)

# Get metrics
agent_metrics = analytics.get_agent_metrics("jira")
print(f"Success rate: {agent_metrics.success_rate:.1%}")
print(f"Avg latency: {agent_metrics.avg_latency_ms:.0f}ms")
print(f"P95 latency: {agent_metrics.p95_latency_ms:.0f}ms")

# Get rankings
rankings = analytics.get_agent_ranking()
for agent_name, success_rate, calls in rankings[:5]:
    print(f"{agent_name}: {success_rate:.1%} ({calls} calls)")

# Get slowest agents
slowest = analytics.get_slowest_agents(limit=5)
for agent_name, avg_latency, calls in slowest:
    print(f"{agent_name}: {avg_latency:.0f}ms avg")

# Check system health
health = analytics.get_health_score()
if health < 0.8:
    print(f"⚠️ System health degraded: {health:.1%}")

# Generate report
print(analytics.generate_summary_report())

# Save analytics
analytics.save_to_file(f"logs/analytics_{session_id}.json")
```

### Integration with Orchestrator

```python
class OrchestratorAgent:
    def __init__(self, ...):
        # ... existing code ...
        self.analytics = AnalyticsCollector(
            session_id=self.session_id,
            verbose=self.verbose
        )

    async def call_sub_agent(self, agent_name: str, instruction: str, context: Any = None):
        # Track timing
        start_time = time.time()

        try:
            result = await self._execute_agent(agent_name, instruction, context)

            # Record success
            latency_ms = (time.time() - start_time) * 1000
            self.analytics.record_agent_call(agent_name, True, latency_ms)

            return result

        except Exception as e:
            # Record failure
            latency_ms = (time.time() - start_time) * 1000
            self.analytics.record_agent_call(agent_name, False, latency_ms, str(e))
            raise

    async def process_message(self, user_message: str) -> str:
        # Track user message
        self.analytics.record_user_message()

        # ... existing logic ...

        # At end of session
        self.analytics.end_session()

        # Save analytics
        self.analytics.save_to_file(f"logs/analytics_{self.session_id}.json")
```

---

## 📈 Monitoring Dashboard

Create a simple monitoring dashboard:

```python
# monitoring_dashboard.py
from core.analytics import AnalyticsCollector

def print_dashboard(analytics: AnalyticsCollector):
    print("\n" + "="*60)
    print("📊 SYSTEM DASHBOARD")
    print("="*60)

    # Health score
    health = analytics.get_health_score()
    health_emoji = "🟢" if health > 0.9 else "🟡" if health > 0.7 else "🔴"
    print(f"\n{health_emoji} Health Score: {health:.1%}")

    # Top agents
    print("\n🏆 Top Performing Agents:")
    for i, (agent, success_rate, calls) in enumerate(analytics.get_agent_ranking()[:5], 1):
        print(f"  {i}. {agent}: {success_rate:.1%} ({calls} calls)")

    # Slowest agents
    print("\n🐌 Slowest Agents:")
    for agent, latency, calls in analytics.get_slowest_agents(3):
        print(f"  • {agent}: {latency:.0f}ms avg ({calls} calls)")

    # Most used operations
    print("\n📊 Most Used Operations:")
    for op, count in analytics.get_most_used_operations(5):
        success = analytics.operation_success_counts[op]
        print(f"  • {op}: {count} times ({success/count:.0%} success)")

    # Common errors
    if analytics.error_patterns:
        print("\n⚠️  Common Errors:")
        for error_type, count in analytics.get_most_common_errors(5):
            print(f"  • {error_type}: {count} times")

    # Peak hours
    peak_hours = analytics.get_peak_usage_hours(3)
    print(f"\n⏰ Peak Usage: {', '.join(f'{h}:00' for h in peak_hours)}")

    print("\n" + "="*60 + "\n")
```

---

## 🚀 Quick Start Integration

### Step 1: Add to Orchestrator

```python
# orchestrator.py
from core.retry_manager import RetryManager
from core.undo_manager import UndoManager
from core.user_preferences import UserPreferenceManager
from core.analytics import AnalyticsCollector

class OrchestratorAgent:
    def __init__(self, ...):
        # ... existing code ...

        # Add new systems
        self.retry_manager = RetryManager(max_retries=3, verbose=self.verbose)
        self.undo_manager = UndoManager(verbose=self.verbose)
        self.user_prefs = UserPreferenceManager(user_id="default", verbose=self.verbose)
        self.analytics = AnalyticsCollector(session_id=self.session_id, verbose=self.verbose)

        # Register undo handlers
        self._register_undo_handlers()

        # Load user preferences
        self._load_user_preferences()
```

### Step 2: Update Agent Calls

Replace direct agent calls with managed calls:

```python
# Before:
result = await agent.execute(instruction)

# After:
result = await self._execute_with_management(agent_name, instruction, context)
```

### Step 3: Add Management Wrapper

```python
async def _execute_with_management(self, agent_name: str, instruction: str, context: Any = None):
    """Execute agent with all management features"""

    # Create operation key
    operation_key = self._get_operation_key(agent_name, instruction)

    # Check user preferences
    operation_pattern = self._get_operation_pattern(agent_name, instruction)

    # Execute with retry
    async def operation():
        start_time = time.time()

        try:
            result = await self.call_sub_agent(agent_name, instruction, context)

            # Record success
            latency_ms = (time.time() - start_time) * 1000
            self.analytics.record_agent_call(agent_name, True, latency_ms)
            self.user_prefs.record_agent_usage(operation_pattern, agent_name, True)

            return result

        except Exception as e:
            # Record failure
            latency_ms = (time.time() - start_time) * 1000
            self.analytics.record_agent_call(agent_name, False, latency_ms, str(e))
            raise

    return await self.retry_manager.execute_with_retry(
        operation_key=operation_key,
        agent_name=agent_name,
        instruction=instruction,
        operation=operation
    )
```

---

## 💡 Best Practices

### 1. Gradual Rollout
- Start with analytics (passive monitoring)
- Add retry management (improves reliability)
- Enable user preferences (after collecting data)
- Add undo last (requires handler implementation)

### 2. Configuration
```python
# config.py
class ImprovedConfig:
    # Retry settings
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 1.0
    RETRY_MAX_DELAY = 30.0

    # Undo settings
    UNDO_TTL_SECONDS = 3600  # 1 hour
    UNDO_MAX_HISTORY = 20

    # Preference learning
    PREFERENCE_CONFIDENCE_THRESHOLD = 0.7

    # Analytics
    MAX_LATENCY_SAMPLES = 1000
    ANALYTICS_SAVE_INTERVAL = 300  # 5 minutes
```

### 3. Persistence
```bash
# Directory structure
data/
  preferences/
    user123.json
  undo/
    session_abc.json
logs/
  analytics/
    session_abc.json
  sessions/
    session_abc.log
```

### 4. Monitoring
- Check health score regularly
- Alert if health < 0.8
- Review error patterns daily
- Analyze slow agents weekly

---

## 🎓 Examples

See `examples/` directory for:
- `retry_example.py` - Retry management usage
- `undo_example.py` - Undo system usage
- `preferences_example.py` - Preference learning
- `analytics_example.py` - Analytics and monitoring

---

## 📝 Next Steps

1. **Test the fixes** - Run the orchestrator and verify stability
2. **Add integration** - Integrate these systems into orchestrator.py
3. **Create handlers** - Implement undo handlers for each agent
4. **Monitor** - Set up analytics dashboard
5. **Optimize** - Use analytics to identify bottlenecks

---

**Questions?** Check the code comments or create an issue!
