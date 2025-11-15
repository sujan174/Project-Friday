#!/usr/bin/env python3
"""
Quick test script for the Enhanced UI

This script demonstrates the UI features without requiring full agent setup.
"""

import sys
import time
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from ui.enhanced_ui import EnhancedUI
    print("✓ Enhanced UI imported successfully")
    UI_AVAILABLE = True
except ImportError as e:
    print(f"✗ Enhanced UI not available: {e}")
    print("  Run: pip install rich")
    UI_AVAILABLE = False
    sys.exit(1)


def test_basic_ui():
    """Test basic UI features"""
    print("\n" + "="*60)
    print("Testing Enhanced UI - Basic Features")
    print("="*60 + "\n")

    ui = EnhancedUI(verbose=True)

    # Test welcome
    print("1. Testing welcome screen...")
    ui.print_welcome("test-session-12345678")
    time.sleep(0.5)

    # Test agents loaded
    print("2. Testing agents loaded...")
    test_agents = [
        {'name': 'slack_agent', 'capabilities': ['send_message', 'read_channel']},
        {'name': 'jira_agent', 'capabilities': ['create_issue', 'update_issue']},
        {'name': 'github_agent', 'capabilities': ['create_pr', 'merge_pr']},
        {'name': 'notion_agent', 'capabilities': ['create_page', 'search_db']},
        {'name': 'browser_agent', 'capabilities': ['navigate', 'click']},
    ]
    ui.print_agents_loaded(test_agents)
    time.sleep(0.5)

    # Test messages
    print("3. Testing message types...")
    ui.print_success("This is a success message")
    time.sleep(0.3)
    ui.print_info("This is an info message")
    time.sleep(0.3)
    ui.print_warning("This is a warning message")
    time.sleep(0.3)

    # Test markdown
    print("\n4. Testing markdown rendering...")
    markdown_text = """
# Test Response

This is a **bold** statement with some *italic* text and `inline code`.

## Features

- First feature with **emphasis**
- Second feature with `code`
- Third feature

### Code Block

```python
def hello_world():
    print("Hello from Project Aerius!")
    return True
```

Regular paragraph with more text to show rendering.
"""
    ui.print_response(markdown_text)
    time.sleep(0.5)

    # Test error
    print("\n5. Testing error display...")
    ui.print_error("This is a test error message\nWith multiple lines\nTo show formatting")
    time.sleep(0.5)

    print("\n" + "="*60)
    print("All basic tests passed! ✓")
    print("="*60 + "\n")


async def test_advanced_ui():
    """Test advanced UI features with async"""
    print("\n" + "="*60)
    print("Testing Enhanced UI - Advanced Features")
    print("="*60 + "\n")

    ui = EnhancedUI(verbose=True)

    # Test thinking spinner
    print("1. Testing thinking spinner...")
    with ui.thinking_spinner("Analyzing request"):
        await asyncio.sleep(2)
    time.sleep(0.5)

    # Test agent operation
    print("\n2. Testing agent operation spinner...")
    with ui.agent_operation("slack_agent", "Send message to #general"):
        await asyncio.sleep(1.5)
    time.sleep(0.5)

    # Test progress bar
    print("\n3. Testing progress bar...")
    with ui.progress_bar("Processing files", total=20) as result:
        if result:
            progress, task = result
            for i in range(20):
                await asyncio.sleep(0.1)
                progress.update(task, advance=1)
    time.sleep(0.5)

    # Test session summary
    print("\n4. Testing session summary...")
    test_stats = {
        'message_count': 25,
        'agent_calls': 42,
        'agent_stats': {
            'slack_agent': {
                'calls': 15,
                'avg_time_ms': 234,
                'success_rate': 100
            },
            'jira_agent': {
                'calls': 18,
                'avg_time_ms': 456,
                'success_rate': 94.4
            },
            'github_agent': {
                'calls': 9,
                'avg_time_ms': 678,
                'success_rate': 88.9
            }
        }
    }
    ui.print_session_summary(test_stats)

    print("\n" + "="*60)
    print("All advanced tests passed! ✓")
    print("="*60 + "\n")


async def test_help_display():
    """Test help display with mock orchestrator"""
    print("\n" + "="*60)
    print("Testing Enhanced UI - Help Display")
    print("="*60 + "\n")

    ui = EnhancedUI(verbose=True)

    # Create mock orchestrator
    class MockOrchestrator:
        agent_capabilities = {
            'slack_agent': ['send_message', 'read_channel', 'list_channels'],
            'jira_agent': ['create_issue', 'update_issue', 'search_issues'],
            'github_agent': ['create_pr', 'merge_pr', 'list_repos'],
            'notion_agent': ['create_page', 'search_database'],
            'browser_agent': ['navigate', 'click', 'extract_text'],
        }
        agent_health = {
            'slack_agent': {'status': 'healthy'},
            'jira_agent': {'status': 'healthy'},
            'github_agent': {'status': 'degraded'},
            'notion_agent': {'status': 'healthy'},
            'browser_agent': {'status': 'unavailable'},
        }

    orchestrator = MockOrchestrator()
    ui.print_help(orchestrator)

    print("Help display test completed! ✓\n")


def main():
    """Run all tests"""
    print("\n╔════════════════════════════════════════════════════════╗")
    print("║         Project Aerius - Enhanced UI Test Suite       ║")
    print("╚════════════════════════════════════════════════════════╝")

    try:
        # Basic tests
        test_basic_ui()
        time.sleep(1)

        # Advanced tests (async)
        asyncio.run(test_advanced_ui())
        time.sleep(1)

        # Help display test
        asyncio.run(test_help_display())

        # Final message
        ui = EnhancedUI(verbose=False)
        ui.print_goodbye()

        print("\n✓ All UI tests completed successfully!\n")
        return 0

    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
