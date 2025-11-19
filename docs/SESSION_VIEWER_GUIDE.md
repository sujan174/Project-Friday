# Session Viewer Guide

Professional tool for visualizing complete session logs in a beautiful, readable format.

## 🎯 What It Does

The Session Viewer loads all log data for a specific session and presents it in a professional format:

- **Terminal View**: Beautiful colored output in your terminal
- **HTML Report**: Professional web-based report that opens in your browser
- **Timeline View**: See the complete flow of events chronologically
- **Agent Activity**: What each agent did and when
- **Intelligence Insights**: Intent classification, entity extraction, decisions
- **Performance Metrics**: Latency, success rates, resource usage
- **Distributed Traces**: Complete request flow with parent-child relationships

## 🚀 Quick Start

### List All Available Sessions

```bash
python tools/session_viewer.py --list
```

Output:
```
Available Sessions:

  • test-basic-logging
  • my-session-id
  • abc123def456
```

### View a Session in Terminal

```bash
python tools/session_viewer.py <session-id>
```

Example:
```bash
python tools/session_viewer.py test-basic-logging
```

### Generate HTML Report

```bash
python tools/session_viewer.py <session-id> --html
```

This will:
1. Generate a beautiful HTML report
2. Save it as `session_report_<session-id>.html`
3. Automatically open it in your browser

### Both Terminal and HTML

```bash
python tools/session_viewer.py <session-id> --format both
```

## 📊 What You'll See

### 1. Agent Orchestration Section

Shows you:
- **Overview Stats**: Total tasks, success/failure counts, success rate, average duration
- **Agent Lifecycle**: When agents were initialized, became ready, or encountered errors
- **Task Execution**: Complete task history with:
  - Task name and description
  - Which agent handled it
  - Success/failure status
  - Duration in milliseconds
  - Error messages if failed
- **Routing Decisions**: Why each task was routed to a specific agent
  - Confidence level
  - Reasoning
  - Considered alternatives

### 2. Intelligence Processing Section

Shows you:
- **Overview Stats**: Messages processed, cache hit rate, intents/entities/decisions
- **Intent Classifications**: What intents were detected in each message
  - Detected intent types
  - Confidence scores
  - Classification method (keyword vs LLM)
- **Entity Extractions**: What entities were found
  - Entity types (action, platform, resource, etc.)
  - Entity values
  - Confidence scores
  - Relationships between entities
- **Decisions Made**: What the system decided to do
  - Decision type (PROCEED, CONFIRM, CLARIFY, etc.)
  - Confidence level
  - Reasoning behind the decision

### 3. Performance Metrics Section

Shows you:
- **Request Metrics**:
  - Total requests
  - Succeeded/failed counts
  - Success rate percentage
- **Latency Stats**:
  - Mean latency
  - Median (P50)
  - P95 and P99 percentiles
- **Agent Stats**:
  - Number of active agents
  - Task counts per agent

### 4. Distributed Traces Section

Shows you:
- **Trace Overview**: Trace ID, total duration, span count
- **Span Tree**: Visual hierarchy showing:
  - Parent-child relationships
  - Operation names
  - Duration of each operation
  - Success/failure status
  - Events that occurred during execution

## 🎨 Terminal View Example

```
================================================================================
                              SESSION VIEWER
================================================================================

Session ID: test-basic-logging

────────────────────────────────────────────────────────────────────────────────
🤖 AGENT ORCHESTRATION
────────────────────────────────────────────────────────────────────────────────

Overview:
  Total Tasks: 3
  Succeeded: 3
  Failed: 0
  Success Rate: 100.0%
  Avg Duration: 102.5ms
  Total Agents: 1

Agent Lifecycle:
  2025-11-08T18:19:42 | test_agent: UNINITIALIZED → INITIALIZING
    Reason: Agent initialization started
  2025-11-08T18:19:42 | test_agent: INITIALIZING → READY
    Reason: Agent ready to accept tasks

Task Execution:

  send_test_message
    Agent: test_agent
    Status: SUCCEEDED
    Duration: 105.1ms

────────────────────────────────────────────────────────────────────────────────
🧠 INTELLIGENCE PROCESSING
────────────────────────────────────────────────────────────────────────────────

Overview:
  Messages Processed: 1
  Cache Hit Rate: 0.0%
  Total Intents Classified: 1
  Total Entities Extracted: 5
  Total Decisions Made: 1
  Avg Confidence: 90%

Intent Classifications:

  Message 1: Create a Jira issue for the login bug...
    Intents: CREATE, JIRA
    Method: keyword

Decisions:

  CONFIRM
    Confidence: 90%
    Reasoning: High confidence but requires user confirmation for Jira creation
```

## 📄 HTML Report Example

The HTML report provides the same information in a beautiful web interface:

- **Modern, gradient header** with session information
- **Color-coded stats cards** for quick insights
- **Timeline view** with visual indicators
- **Expandable task cards** with detailed information
- **Professional styling** suitable for sharing with team members
- **Responsive design** that works on all screen sizes

## 🔧 Advanced Usage

### Custom Output Path

```bash
python tools/session_viewer.py <session-id> --html --output /path/to/report.html
```

### Custom Logs Directory

```bash
python tools/session_viewer.py <session-id> --logs-dir /custom/logs/path
```

### Save Terminal Output to File

```bash
python tools/session_viewer.py <session-id> > session_analysis.txt
```

## 💡 Use Cases

### 1. Debugging Issues

When something goes wrong:
```bash
python tools/session_viewer.py <failed-session-id>
```

Instantly see:
- Which agent failed
- What error occurred
- How many retries were attempted
- What led up to the failure

### 2. Performance Analysis

Find bottlenecks:
```bash
python tools/session_viewer.py <session-id> --html
```

Look at:
- Which operations took the longest
- P95/P99 latencies
- Success rates by agent
- Cache hit rates

### 3. Understanding User Interactions

See how the system interpreted user requests:
```bash
python tools/session_viewer.py <session-id>
```

Review:
- What intents were detected
- What entities were extracted
- Why certain agents were selected
- What decisions were made

### 4. Team Sharing

Generate professional reports for your team:
```bash
python tools/session_viewer.py <session-id> --html --output weekly_report.html
```

Share the HTML file via:
- Email
- Slack
- Internal wiki
- Bug tracker

### 5. Training Data

Use session logs to:
- Improve intent classification
- Refine entity extraction
- Train new team members
- Document system behavior

## 🎯 Tips & Tricks

### Filter Sessions by Date

```bash
# List sessions and grep for date
python tools/session_viewer.py --list | grep "2025-11"
```

### Compare Multiple Sessions

```bash
# Generate HTML for multiple sessions
for session in session1 session2 session3; do
    python tools/session_viewer.py $session --html --output "${session}.html"
done
```

### Quick Health Check

```bash
# Check if any sessions had failures
python tools/session_viewer.py <session-id> | grep -i "failed"
```

### Export to JSON

The viewer reads from JSON files, so you can also:
```bash
# Direct JSON inspection
cat logs/orchestration/orchestration_<session-id>_*.json | jq .
```

## 📊 Understanding the Data

### Session ID Format

Session IDs are UUIDs generated at orchestrator startup:
- Format: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`
- Example: `test-basic-logging`

### File Locations

The viewer reads from:
```
logs/
├── orchestration/orchestration_<session-id>_<timestamp>.json
├── intelligence/intelligence_<session-id>_<timestamp>.json
├── metrics/metrics_<session-id>.json
└── traces/trace_<trace-id>_<timestamp>.json
```

### Timestamps

All timestamps are in ISO 8601 format:
- `2025-11-08T18:19:42.123456Z`
- UTC timezone
- Microsecond precision

## 🐛 Troubleshooting

### "No sessions found"

**Cause**: No log files in the logs directory

**Solution**:
```bash
# Check logs directory exists
ls -la logs/

# Run orchestrator to generate logs
python orchestrator.py

# Try again
python tools/session_viewer.py --list
```

### "Error loading session"

**Cause**: Session ID doesn't match any log files

**Solution**:
```bash
# List available sessions
python tools/session_viewer.py --list

# Use exact session ID
python tools/session_viewer.py <exact-session-id>
```

### HTML report doesn't open

**Cause**: Browser security restrictions

**Solution**:
```bash
# Manually open the file
open session_report_<session-id>.html  # macOS
xdg-open session_report_<session-id>.html  # Linux
start session_report_<session-id>.html  # Windows
```

## 🚀 Integration Ideas

### CI/CD Pipeline

```bash
# In your test script
SESSION_ID=$(grep "session_id" test_output.log | cut -d: -f2)
python tools/session_viewer.py $SESSION_ID --html --output test_report.html

# Upload to S3/artifact storage
aws s3 cp test_report.html s3://my-bucket/test-reports/
```

### Monitoring Dashboard

```python
# Monitor script
import subprocess
import time

while True:
    # Get latest session
    result = subprocess.run([
        'python', 'tools/session_viewer.py', '--list'
    ], capture_output=True, text=True)

    sessions = result.stdout.strip().split('\n')
    latest = sessions[-1].strip()

    # Generate report
    subprocess.run([
        'python', 'tools/session_viewer.py',
        latest, '--html', '--output', 'latest.html'
    ])

    time.sleep(300)  # Every 5 minutes
```

### Slack Bot Integration

```python
# Send session reports to Slack
import requests

session_id = "abc123"
subprocess.run([
    'python', 'tools/session_viewer.py',
    session_id, '--html'
])

# Upload to Slack
files = {'file': open(f'session_report_{session_id}.html')}
requests.post(
    'https://slack.com/api/files.upload',
    headers={'Authorization': f'Bearer {SLACK_TOKEN}'},
    files=files
)
```

## 📚 Further Reading

- [LOGGING_SYSTEM_README.md](LOGGING_SYSTEM_README.md) - Overview of the logging system
- [LOGGING_GUIDE.md](LOGGING_GUIDE.md) - How to add logging to your code
- [ORCHESTRATOR_LOGGING_INTEGRATION.md](ORCHESTRATOR_LOGGING_INTEGRATION.md) - Integration guide

## 🎉 Summary

The Session Viewer is your window into understanding exactly what happened during any orchestrator session:

✅ **Beautiful terminal output** for quick checks
✅ **Professional HTML reports** for sharing
✅ **Complete timeline** of all events
✅ **Agent activity tracking** with full details
✅ **Intelligence insights** into decision-making
✅ **Performance metrics** for optimization
✅ **Distributed traces** showing request flow

**Use it every time you need to understand what the system did!**
