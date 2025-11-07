#!/usr/bin/env python3
"""
Test the confirmation system for Slack messages
"""

import asyncio
import os

# Set confirmation to ON for testing
os.environ['CONFIRM_SLACK_MESSAGES'] = 'true'

async def test_slack_confirmation():
    """Test Slack message confirmation"""
    from orchestrator import OrchestratorAgent

    print("="*70)
    print("TESTING SLACK MESSAGE CONFIRMATION")
    print("="*70)
    print()
    print("This will test sending a Slack message with confirmation enabled.")
    print("You should see:")
    print("  1. Message preview")
    print("  2. Options to: [a]pprove, [e]dit manually, [m]odify with AI, [r]eject")
    print("  3. Ability to edit the message before sending")
    print()
    print("="*70)
    print()

    orchestrator = OrchestratorAgent(connectors_dir="connectors", verbose=True)

    try:
        # This should trigger confirmation
        response = await orchestrator.process_message(
            "send a message to #dev-opps saying 'Test message - please ignore'"
        )
        print(f"\n\nOrchestrator Response: {response}")
    finally:
        await orchestrator.cleanup()

if __name__ == "__main__":
    asyncio.run(test_slack_confirmation())
