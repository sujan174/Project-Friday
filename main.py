#!/usr/bin/env python3
"""
Main entry point for the Multi-Agent Orchestration System

A professional, Claude Code-inspired interface for managing and coordinating
specialized AI agents across multiple platforms.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import OrchestratorAgent
from ui.claude_ui import ClaudeUI


async def main():
    """Main entry point"""
    # Parse command-line arguments
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    simple_mode = "--simple" in sys.argv

    # Initialize UI
    ui = ClaudeUI(verbose=verbose)

    # Initialize orchestrator
    orchestrator = OrchestratorAgent(
        connectors_dir="connectors",
        verbose=verbose
    )

    try:
        # Display welcome
        ui.print_welcome(orchestrator.session_id)

        # Discover and load agents (silent)
        await orchestrator.discover_and_load_agents()

        # Display loaded agents
        loaded_agents = []
        failed_agents = []

        for agent_name, health in orchestrator.agent_health.items():
            if health['status'] == 'healthy':
                caps = orchestrator.agent_capabilities.get(agent_name, [])
                loaded_agents.append({
                    'name': agent_name,
                    'capabilities': caps[:3],
                    'total': len(caps)
                })
            else:
                failed_agents.append(agent_name)

        ui.print_agents_loaded(loaded_agents)

        if failed_agents:
            ui.print_agents_failed(len(failed_agents))

        # Start interactive session
        await run_interactive_session(orchestrator, ui)

    except KeyboardInterrupt:
        ui.print_goodbye()
    except Exception as e:
        ui.print_error(str(e))
        if verbose:
            import traceback
            traceback.print_exc()
    finally:
        await orchestrator.cleanup()


async def run_interactive_session(orchestrator: OrchestratorAgent, ui: ClaudeUI):
    """
    Run the main interactive session loop

    Args:
        orchestrator: The orchestration agent
        ui: The user interface
    """
    message_count = 0
    session_start = time.time()

    while True:
        try:
            # Display prompt
            ui.print_prompt()
            user_input = input().strip()

            # Handle exit commands
            if user_input.lower() in ['exit', 'quit', 'bye', 'q']:
                # Calculate session stats
                duration = int(time.time() - session_start)
                stats = {
                    'duration': format_duration(duration),
                    'message_count': message_count,
                    'agent_calls': ui.agent_call_count,
                    'success_rate': "N/A"  # Can be calculated from analytics
                }
                ui.print_session_summary(stats)
                ui.print_goodbye()
                break

            # Skip empty input
            if not user_input:
                continue

            # Handle help command
            if user_input.lower() == 'help':
                show_help(orchestrator, ui)
                continue

            # Process the message
            message_count += 1
            ui.start_thinking()

            response = await process_with_ui(orchestrator, user_input, ui)

            # Display response
            ui.print_response(response)

        except KeyboardInterrupt:
            ui.print_goodbye()
            break
        except Exception as e:
            ui.print_error(str(e))
            if ui.verbose:
                import traceback
                traceback.print_exc()


async def process_with_ui(
    orchestrator: OrchestratorAgent,
    user_message: str,
    ui: ClaudeUI
) -> str:
    """
    Process a user message with UI feedback

    This wraps the orchestrator's call_sub_agent method to provide
    real-time UI updates as agents are called.

    Args:
        orchestrator: The orchestration agent
        user_message: The user's message
        ui: The user interface

    Returns:
        The orchestrator's response
    """
    # Store original method
    original_call_sub_agent = orchestrator.call_sub_agent

    # Track retry attempts
    retry_counts: dict = {}

    async def wrapped_call_sub_agent(agent_name: str, instruction: str, context: dict = None):
        """Wrapped version that provides UI feedback"""

        # Track retries
        retry_key = f"{agent_name}:{instruction[:50]}"
        retry_counts[retry_key] = retry_counts.get(retry_key, 0) + 1

        # Show retry if this is not the first attempt
        if retry_counts[retry_key] > 1:
            ui.show_retry(agent_name, retry_counts[retry_key], orchestrator.max_retry_attempts)

        # Show agent call start
        if retry_counts[retry_key] == 1:
            ui.start_agent_call(agent_name, instruction)

        # Call the actual agent
        start_time = time.time()
        try:
            result = await original_call_sub_agent(agent_name, instruction, context)

            # Check if successful
            success = not (
                result.startswith("Error") or
                result.startswith("⚠️") or
                result.startswith("❌")
            )

            duration_ms = (time.time() - start_time) * 1000

            # Show completion
            if retry_counts[retry_key] == 1 or success:
                ui.end_agent_call(
                    agent_name,
                    success=success,
                    duration_ms=duration_ms,
                    message=result if success else result[:200]
                )

            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            ui.end_agent_call(
                agent_name,
                success=False,
                duration_ms=duration_ms,
                message=str(e)
            )
            raise

    # Replace the method temporarily
    orchestrator.call_sub_agent = wrapped_call_sub_agent

    try:
        # Process the message
        response = await orchestrator.process_message(user_message)

        return response

    finally:
        # Restore original method
        orchestrator.call_sub_agent = original_call_sub_agent


def show_help(orchestrator: OrchestratorAgent, ui: ClaudeUI):
    """Display help information"""
    ui.print_help(orchestrator)


def format_duration(seconds: int) -> str:
    """Format duration in seconds to human-readable string"""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}m {secs}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"


# Clean imports - no need for Color anymore


if __name__ == "__main__":
    asyncio.run(main())
