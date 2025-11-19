#!/usr/bin/env python3
"""
Session Log Viewer

Professional visualization tool for viewing complete session logs.
Supports both terminal output and HTML report generation.

Usage:
    python tools/session_viewer.py <session_id>                    # Terminal view
    python tools/session_viewer.py <session_id> --html             # Generate HTML report
    python tools/session_viewer.py <session_id> --format both      # Both terminal and HTML
    python tools/session_viewer.py --list                          # List all sessions
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from html import escape as html_escape


# ============================================================================
# ANSI COLOR CODES FOR TERMINAL
# ============================================================================

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    DIM = '\033[2m'


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class SessionData:
    """Complete session data loaded from all sources"""
    session_id: str
    orchestration: Optional[Dict] = None
    intelligence: Optional[Dict] = None
    metrics: Optional[Dict] = None
    traces: List[Dict] = None

    def __post_init__(self):
        if self.traces is None:
            self.traces = []


# ============================================================================
# SESSION DATA LOADER
# ============================================================================

class SessionLoader:
    """Load session data from log files"""

    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)

    def list_sessions(self) -> List[str]:
        """List all available session IDs"""
        sessions = set()

        # Check orchestration logs
        orch_dir = self.logs_dir / "orchestration"
        if orch_dir.exists():
            for file in orch_dir.glob("orchestration_*.json"):
                # Extract session ID from filename
                parts = file.stem.split('_')
                if len(parts) >= 2:
                    sessions.add(parts[1])

        # Check intelligence logs
        intel_dir = self.logs_dir / "intelligence"
        if intel_dir.exists():
            for file in intel_dir.glob("intelligence_*.json"):
                parts = file.stem.split('_')
                if len(parts) >= 2:
                    sessions.add(parts[1])

        return sorted(list(sessions))

    def load_session(self, session_id: str) -> SessionData:
        """Load all data for a session"""
        data = SessionData(session_id=session_id)

        # Load orchestration data
        orch_files = list((self.logs_dir / "orchestration").glob(f"orchestration_{session_id}_*.json"))
        if orch_files:
            try:
                with open(orch_files[0], 'r') as f:
                    data.orchestration = json.load(f)
            except json.JSONDecodeError as e:
                print(f"{Colors.YELLOW}⚠ Warning: Skipping malformed orchestration file: {e}{Colors.ENDC}")

        # Load intelligence data
        intel_files = list((self.logs_dir / "intelligence").glob(f"intelligence_{session_id}_*.json"))
        if intel_files:
            try:
                with open(intel_files[0], 'r') as f:
                    data.intelligence = json.load(f)
            except json.JSONDecodeError as e:
                print(f"{Colors.YELLOW}⚠ Warning: Skipping malformed intelligence file: {e}{Colors.ENDC}")

        # Load metrics data
        metrics_files = list((self.logs_dir / "metrics").glob(f"metrics_{session_id}.json"))
        if metrics_files:
            try:
                with open(metrics_files[0], 'r') as f:
                    data.metrics = json.load(f)
            except json.JSONDecodeError as e:
                print(f"{Colors.YELLOW}⚠ Warning: Skipping malformed metrics file: {e}{Colors.ENDC}")

        # Load traces (skip malformed files)
        trace_dir = self.logs_dir / "traces"
        if trace_dir.exists():
            for trace_file in trace_dir.glob("trace_*.json"):
                try:
                    with open(trace_file, 'r') as f:
                        trace = json.load(f)
                        data.traces.append(trace)
                except json.JSONDecodeError as e:
                    # Silently skip malformed trace files (common during crashes)
                    pass

        return data


# ============================================================================
# TERMINAL VIEWER
# ============================================================================

class TerminalViewer:
    """Beautiful terminal output for session data"""

    @staticmethod
    def print_header(session_id: str):
        """Print session header"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'  SESSION VIEWER':^80}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.ENDC}")
        print(f"\n{Colors.BOLD}Session ID:{Colors.ENDC} {session_id}\n")

    @staticmethod
    def print_section(title: str):
        """Print section header"""
        print(f"\n{Colors.BOLD}{Colors.YELLOW}{'─'*80}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.YELLOW}{title}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.YELLOW}{'─'*80}{Colors.ENDC}\n")

    @staticmethod
    def print_orchestration(data: Dict):
        """Print orchestration summary"""
        TerminalViewer.print_section("🤖 AGENT ORCHESTRATION")

        stats = data.get('statistics', {})

        # Summary stats
        print(f"{Colors.BOLD}Overview:{Colors.ENDC}")
        print(f"  Total Tasks: {stats.get('total_tasks', 0)}")
        print(f"  Succeeded: {Colors.GREEN}{stats.get('succeeded_tasks', 0)}{Colors.ENDC}")
        print(f"  Failed: {Colors.RED}{stats.get('failed_tasks', 0)}{Colors.ENDC}")
        print(f"  Success Rate: {stats.get('success_rate', 0)*100:.1f}%")
        print(f"  Avg Duration: {stats.get('avg_task_duration_ms', 0):.1f}ms")
        print(f"  Total Agents: {stats.get('agents_count', 0)}")

        # Agent state transitions
        transitions = data.get('state_transitions', [])
        if transitions:
            print(f"\n{Colors.BOLD}Agent Lifecycle:{Colors.ENDC}")
            for t in transitions:
                time_str = t.get('timestamp_iso', '')
                agent = t.get('agent_name', 'unknown')
                from_state = t.get('from_state', '')
                to_state = t.get('to_state', '')
                reason = t.get('reason', '')

                color = Colors.GREEN if to_state == 'READY' else Colors.YELLOW
                print(f"  {Colors.DIM}{time_str}{Colors.ENDC} | {color}{agent}{Colors.ENDC}: {from_state} → {to_state}")
                if reason:
                    print(f"    {Colors.DIM}Reason: {reason}{Colors.ENDC}")

        # Task assignments
        tasks = data.get('task_assignments', {})
        if tasks:
            print(f"\n{Colors.BOLD}Task Execution:{Colors.ENDC}")
            for task_id, task in tasks.items():
                agent = task.get('assigned_agent', 'unknown')
                task_name = task.get('task_name', 'unknown')
                status = task.get('status', 'unknown')
                duration = task.get('duration_ms', 0)

                status_color = Colors.GREEN if status == 'SUCCEEDED' else Colors.RED
                print(f"\n  {Colors.BOLD}{task_name[:60]}{Colors.ENDC}")
                print(f"    Agent: {Colors.CYAN}{agent}{Colors.ENDC}")
                print(f"    Status: {status_color}{status}{Colors.ENDC}")
                print(f"    Duration: {duration:.1f}ms")

                if task.get('errors'):
                    print(f"    {Colors.RED}Errors:{Colors.ENDC}")
                    for error in task['errors']:
                        print(f"      • {error[:100]}")

        # Routing decisions
        routing = data.get('routing_decisions', [])
        if routing:
            print(f"\n{Colors.BOLD}Routing Decisions:{Colors.ENDC}")
            for decision in routing[:5]:  # Show first 5
                task_name = decision.get('task_name', 'unknown')
                selected = decision.get('selected_agent', 'unknown')
                reason = decision.get('reason', '')
                confidence = decision.get('confidence', 0)

                print(f"\n  {Colors.BOLD}{task_name}{Colors.ENDC}")
                print(f"    Selected: {Colors.CYAN}{selected}{Colors.ENDC}")
                print(f"    Confidence: {confidence*100:.0f}%")
                print(f"    Reason: {Colors.DIM}{reason}{Colors.ENDC}")

    @staticmethod
    def print_intelligence(data: Dict):
        """Print intelligence summary"""
        TerminalViewer.print_section("🧠 INTELLIGENCE PROCESSING")

        stats = data.get('statistics', {})
        cache_stats = data.get('cache_statistics', {})

        # Summary
        print(f"{Colors.BOLD}Overview:{Colors.ENDC}")
        print(f"  Messages Processed: {data.get('total_messages_processed', 0)}")
        print(f"  Cache Hit Rate: {cache_stats.get('cache_hit_rate', 0)*100:.1f}%")
        print(f"  Total Intents Classified: {stats.get('total_intents_classified', 0)}")
        print(f"  Total Entities Extracted: {stats.get('total_entities_extracted', 0)}")
        print(f"  Total Decisions Made: {stats.get('total_decisions_made', 0)}")
        print(f"  Avg Confidence: {stats.get('avg_confidence', 0)*100:.0f}%")

        # Intent classifications
        intents = data.get('intent_classifications', [])
        if intents:
            print(f"\n{Colors.BOLD}Intent Classifications:{Colors.ENDC}")
            for i, intent in enumerate(intents, 1):
                msg = intent.get('user_message', '')[:60]
                detected = intent.get('detected_intents', [])
                method = intent.get('classification_method', 'unknown')

                print(f"\n  {Colors.BOLD}Message {i}:{Colors.ENDC} {msg}...")
                print(f"    Intents: {Colors.CYAN}{', '.join(detected[:3])}{Colors.ENDC}")
                print(f"    Method: {method}")

        # Entity extractions
        entities = data.get('entity_extractions', [])
        if entities:
            print(f"\n{Colors.BOLD}Entity Extractions:{Colors.ENDC}")
            for i, extraction in enumerate(entities, 1):
                extracted = extraction.get('extracted_entities', {})
                total = sum(len(v) for v in extracted.values())
                confidence = extraction.get('confidence', 0)

                print(f"\n  {Colors.BOLD}Extraction {i}:{Colors.ENDC}")
                print(f"    Total Entities: {total}")
                print(f"    Confidence: {confidence*100:.0f}%")
                for entity_type, values in list(extracted.items())[:3]:
                    print(f"    {entity_type}: {Colors.CYAN}{', '.join(map(str, values[:3]))}{Colors.ENDC}")

        # Decisions
        decisions = data.get('decisions', [])
        if decisions:
            print(f"\n{Colors.BOLD}Decisions:{Colors.ENDC}")
            for decision in decisions:
                decision_type = decision.get('decision_type', 'unknown')
                confidence = decision.get('confidence', 0)
                reasoning = decision.get('reasoning', '')

                print(f"\n  {Colors.BOLD}{decision_type}{Colors.ENDC}")
                print(f"    Confidence: {confidence*100:.0f}%")
                print(f"    Reasoning: {Colors.DIM}{reasoning}{Colors.ENDC}")

    @staticmethod
    def print_metrics(data: Dict):
        """Print metrics summary"""
        TerminalViewer.print_section("📊 PERFORMANCE METRICS")

        metrics = data.get('metrics', {})

        if not metrics:
            print(f"  {Colors.DIM}No metrics available{Colors.ENDC}")
            return

        # Request metrics
        total_requests = metrics.get('requests_total', {}).get('value', 0)
        succeeded = metrics.get('requests_succeeded', {}).get('value', 0)
        failed = metrics.get('requests_failed', {}).get('value', 0)

        if total_requests:
            print(f"{Colors.BOLD}Requests:{Colors.ENDC}")
            print(f"  Total: {total_requests}")
            print(f"  Succeeded: {Colors.GREEN}{succeeded}{Colors.ENDC}")
            print(f"  Failed: {Colors.RED}{failed}{Colors.ENDC}")
            print(f"  Success Rate: {(succeeded/total_requests)*100:.1f}%")

        # Latency metrics
        duration = metrics.get('request_duration', {}).get('statistics', {})
        if duration:
            print(f"\n{Colors.BOLD}Latency:{Colors.ENDC}")
            print(f"  Mean: {duration.get('mean', 0):.1f}ms")
            print(f"  Median: {duration.get('median', 0):.1f}ms")
            print(f"  P95: {duration.get('p95', 0):.1f}ms")
            print(f"  P99: {duration.get('p99', 0):.1f}ms")

        # Agent metrics
        active_agents = metrics.get('active_agents', {}).get('value', 0)
        if active_agents:
            print(f"\n{Colors.BOLD}Agents:{Colors.ENDC}")
            print(f"  Active: {active_agents}")

    @staticmethod
    def print_traces(traces: List[Dict]):
        """Print distributed traces"""
        TerminalViewer.print_section("🔍 DISTRIBUTED TRACES")

        if not traces:
            print(f"  {Colors.DIM}No traces available{Colors.ENDC}")
            return

        for i, trace in enumerate(traces, 1):
            trace_id = trace.get('trace_id', 'unknown')
            duration = trace.get('duration_ms', 0)
            span_count = trace.get('span_count', 0)

            print(f"\n{Colors.BOLD}Trace {i}:{Colors.ENDC} {trace_id}")
            print(f"  Duration: {duration:.1f}ms")
            print(f"  Spans: {span_count}")

            # Show span tree
            spans = trace.get('spans', [])
            root_spans = [s for s in spans if s.get('parent_span_id') is None]

            for root in root_spans:
                TerminalViewer._print_span_tree(root, spans, indent=2)

    @staticmethod
    def _print_span_tree(span: Dict, all_spans: List[Dict], indent: int = 0):
        """Recursively print span tree"""
        prefix = "  " * indent
        name = span.get('name', 'unknown')
        duration = span.get('duration_ms', 0)
        status = span.get('status', 'UNSET')

        status_color = Colors.GREEN if status == 'OK' else Colors.RED
        print(f"{prefix}├─ {Colors.BOLD}{name}{Colors.ENDC} ({duration:.1f}ms) {status_color}[{status}]{Colors.ENDC}")

        # Show events
        events = span.get('events', [])
        if events:
            for event in events[:2]:  # Show first 2 events
                event_name = event.get('name', 'unknown')
                print(f"{prefix}│  └─ {Colors.DIM}{event_name}{Colors.ENDC}")

        # Recursively print children
        span_id = span.get('span_id')
        children = [s for s in all_spans if s.get('parent_span_id') == span_id]
        for child in children:
            TerminalViewer._print_span_tree(child, all_spans, indent + 1)


# ============================================================================
# HTML REPORT GENERATOR
# ============================================================================

class HTMLReportGenerator:
    """Generate professional HTML reports"""

    @staticmethod
    def generate(session_data: SessionData, output_path: str):
        """Generate HTML report"""
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Session Report - {session_data.session_id}</title>
    <style>
        {HTMLReportGenerator._get_css()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Session Report</h1>
            <div class="session-id">Session: {session_data.session_id}</div>
            <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </header>

        {HTMLReportGenerator._generate_orchestration_section(session_data.orchestration)}
        {HTMLReportGenerator._generate_intelligence_section(session_data.intelligence)}
        {HTMLReportGenerator._generate_metrics_section(session_data.metrics)}
        {HTMLReportGenerator._generate_traces_section(session_data.traces)}

        <footer>
            <p>Generated by Session Viewer | Lazy Devs Orchestrator</p>
        </footer>
    </div>
</body>
</html>"""

        with open(output_path, 'w') as f:
            f.write(html_content)

    @staticmethod
    def _get_css() -> str:
        """Get CSS styles"""
        return """
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .session-id { font-size: 1.2em; opacity: 0.9; }
        .timestamp { font-size: 0.9em; opacity: 0.7; margin-top: 5px; }

        .section {
            padding: 30px 40px;
            border-bottom: 1px solid #eee;
        }
        .section:last-of-type { border-bottom: none; }
        .section h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.8em;
            display: flex;
            align-items: center;
        }
        .section h2::before {
            content: '';
            display: inline-block;
            width: 4px;
            height: 30px;
            background: #667eea;
            margin-right: 15px;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .stat-label {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }
        .stat-value.success { color: #28a745; }
        .stat-value.error { color: #dc3545; }

        .timeline {
            position: relative;
            padding-left: 30px;
        }
        .timeline::before {
            content: '';
            position: absolute;
            left: 10px;
            top: 0;
            bottom: 0;
            width: 2px;
            background: #ddd;
        }
        .timeline-item {
            position: relative;
            padding: 15px;
            margin-bottom: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        .timeline-item::before {
            content: '';
            position: absolute;
            left: -24px;
            top: 20px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #667eea;
            border: 3px solid white;
        }
        .timeline-item.success::before { background: #28a745; }
        .timeline-item.error::before { background: #dc3545; }

        .task-card {
            background: #f8f9fa;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .task-card.success { border-left-color: #28a745; }
        .task-card.error { border-left-color: #dc3545; }
        .task-header {
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }
        .task-meta {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            font-size: 0.9em;
            color: #666;
        }
        .task-meta span {
            display: flex;
            align-items: center;
        }
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 500;
        }
        .badge.success { background: #d4edda; color: #155724; }
        .badge.error { background: #f8d7da; color: #721c24; }
        .badge.info { background: #d1ecf1; color: #0c5460; }

        footer {
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            color: #666;
            font-size: 0.9em;
        }

        .chart { margin: 20px 0; }
        .no-data {
            text-align: center;
            padding: 40px;
            color: #999;
            font-style: italic;
        }
        """

    @staticmethod
    def _generate_orchestration_section(data: Optional[Dict]) -> str:
        """Generate orchestration section"""
        if not data:
            return '<div class="section"><h2>🤖 Agent Orchestration</h2><div class="no-data">No data available</div></div>'

        stats = data.get('statistics', {})

        html = '<div class="section"><h2>🤖 Agent Orchestration</h2>'

        # Stats cards
        html += '<div class="stats-grid">'
        html += f'<div class="stat-card"><div class="stat-label">Total Tasks</div><div class="stat-value">{stats.get("total_tasks", 0)}</div></div>'
        html += f'<div class="stat-card"><div class="stat-label">Succeeded</div><div class="stat-value success">{stats.get("succeeded_tasks", 0)}</div></div>'
        html += f'<div class="stat-card"><div class="stat-label">Failed</div><div class="stat-value error">{stats.get("failed_tasks", 0)}</div></div>'
        html += f'<div class="stat-card"><div class="stat-label">Avg Duration</div><div class="stat-value">{stats.get("avg_task_duration_ms", 0):.0f}ms</div></div>'
        html += '</div>'

        # Tasks
        tasks = data.get('task_assignments', {})
        if tasks:
            html += '<h3>Task Execution</h3>'
            for task_id, task in tasks.items():
                status_class = 'success' if task.get('status') == 'SUCCEEDED' else 'error'
                html += f'<div class="task-card {status_class}">'
                html += f'<div class="task-header">{html_escape(task.get("task_name", "Unknown")[:80])}</div>'
                html += f'<div class="task-meta">'
                html += f'<span><strong>Agent:</strong> {task.get("assigned_agent", "unknown")}</span>'
                html += f'<span><strong>Status:</strong> <span class="badge {status_class}">{task.get("status", "unknown")}</span></span>'
                html += f'<span><strong>Duration:</strong> {task.get("duration_ms", 0):.1f}ms</span>'
                html += f'</div>'
                html += '</div>'

        html += '</div>'
        return html

    @staticmethod
    def _generate_intelligence_section(data: Optional[Dict]) -> str:
        """Generate intelligence section"""
        if not data:
            return '<div class="section"><h2>🧠 Intelligence Processing</h2><div class="no-data">No data available</div></div>'

        stats = data.get('statistics', {})
        cache_stats = data.get('cache_statistics', {})

        html = '<div class="section"><h2>🧠 Intelligence Processing</h2>'

        # Stats
        html += '<div class="stats-grid">'
        html += f'<div class="stat-card"><div class="stat-label">Messages Processed</div><div class="stat-value">{data.get("total_messages_processed", 0)}</div></div>'
        html += f'<div class="stat-card"><div class="stat-label">Cache Hit Rate</div><div class="stat-value">{cache_stats.get("cache_hit_rate", 0)*100:.0f}%</div></div>'
        html += f'<div class="stat-card"><div class="stat-label">Intents Classified</div><div class="stat-value">{stats.get("total_intents_classified", 0)}</div></div>'
        html += f'<div class="stat-card"><div class="stat-label">Decisions Made</div><div class="stat-value">{stats.get("total_decisions_made", 0)}</div></div>'
        html += '</div>'

        html += '</div>'
        return html

    @staticmethod
    def _generate_metrics_section(data: Optional[Dict]) -> str:
        """Generate metrics section"""
        if not data:
            return '<div class="section"><h2>📊 Performance Metrics</h2><div class="no-data">No data available</div></div>'

        metrics = data.get('metrics', {})

        html = '<div class="section"><h2>📊 Performance Metrics</h2>'

        # Request metrics
        total = metrics.get('requests_total', {}).get('value', 0)
        succeeded = metrics.get('requests_succeeded', {}).get('value', 0)
        failed = metrics.get('requests_failed', {}).get('value', 0)

        html += '<div class="stats-grid">'
        html += f'<div class="stat-card"><div class="stat-label">Total Requests</div><div class="stat-value">{total}</div></div>'
        html += f'<div class="stat-card"><div class="stat-label">Succeeded</div><div class="stat-value success">{succeeded}</div></div>'
        html += f'<div class="stat-card"><div class="stat-label">Failed</div><div class="stat-value error">{failed}</div></div>'
        if total > 0:
            html += f'<div class="stat-card"><div class="stat-label">Success Rate</div><div class="stat-value">{(succeeded/total)*100:.0f}%</div></div>'
        html += '</div>'

        html += '</div>'
        return html

    @staticmethod
    def _generate_traces_section(traces: List[Dict]) -> str:
        """Generate traces section"""
        if not traces:
            return '<div class="section"><h2>🔍 Distributed Traces</h2><div class="no-data">No traces available</div></div>'

        html = '<div class="section"><h2>🔍 Distributed Traces</h2>'

        for trace in traces:
            trace_id = trace.get('trace_id', 'unknown')
            duration = trace.get('duration_ms', 0)
            span_count = trace.get('span_count', 0)

            html += f'<div class="task-card info">'
            html += f'<div class="task-header">Trace: {trace_id[:16]}...</div>'
            html += f'<div class="task-meta">'
            html += f'<span><strong>Duration:</strong> {duration:.1f}ms</span>'
            html += f'<span><strong>Spans:</strong> {span_count}</span>'
            html += f'</div>'
            html += '</div>'

        html += '</div>'
        return html


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Professional Session Log Viewer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/session_viewer.py abc123def                 # View in terminal
  python tools/session_viewer.py abc123def --html          # Generate HTML report
  python tools/session_viewer.py abc123def --format both   # Both terminal and HTML
  python tools/session_viewer.py --list                    # List all sessions
        """
    )

    parser.add_argument('session_id', nargs='?', help='Session ID to view')
    parser.add_argument('--list', action='store_true', help='List all available sessions')
    parser.add_argument('--html', action='store_true', help='Generate HTML report')
    parser.add_argument('--format', choices=['terminal', 'html', 'both'], default='terminal',
                       help='Output format (default: terminal)')
    parser.add_argument('--output', help='Output file path for HTML report')
    parser.add_argument('--logs-dir', default='logs', help='Logs directory (default: logs)')

    args = parser.parse_args()

    loader = SessionLoader(logs_dir=args.logs_dir)

    # List sessions
    if args.list:
        sessions = loader.list_sessions()
        print(f"\n{Colors.BOLD}{Colors.CYAN}Available Sessions:{Colors.ENDC}\n")
        if sessions:
            for session in sessions:
                print(f"  • {session}")
        else:
            print(f"  {Colors.DIM}No sessions found in {args.logs_dir}/{Colors.ENDC}")
        print()
        return

    # Check session ID provided
    if not args.session_id:
        parser.print_help()
        return

    # Load session data
    try:
        data = loader.load_session(args.session_id)
    except Exception as e:
        print(f"{Colors.RED}✗ Error loading session: {e}{Colors.ENDC}")
        return

    # Determine format
    if args.html:
        output_format = 'html'
    else:
        output_format = args.format

    # Terminal output
    if output_format in ['terminal', 'both']:
        TerminalViewer.print_header(args.session_id)

        if data.orchestration:
            TerminalViewer.print_orchestration(data.orchestration)

        if data.intelligence:
            TerminalViewer.print_intelligence(data.intelligence)

        if data.metrics:
            TerminalViewer.print_metrics(data.metrics)

        if data.traces:
            TerminalViewer.print_traces(data.traces)

        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.ENDC}\n")

    # HTML output
    if output_format in ['html', 'both']:
        output_path = args.output or f"session_report_{args.session_id}.html"
        try:
            HTMLReportGenerator.generate(data, output_path)
            print(f"{Colors.GREEN}✓ HTML report generated: {output_path}{Colors.ENDC}")

            # Try to open in browser
            import webbrowser
            import os
            abs_path = os.path.abspath(output_path)
            webbrowser.open(f'file://{abs_path}')
            print(f"{Colors.CYAN}  Opening in browser...{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.RED}✗ Error generating HTML: {e}{Colors.ENDC}")


if __name__ == "__main__":
    main()
