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

# Try to import enhanced UI, fall back to basic UI if Rich is not available
try:
    from ui.enhanced_ui import EnhancedUI
    UI_CLASS = EnhancedUI
except ImportError:
    from ui.claude_ui import ClaudeUI
    UI_CLASS = ClaudeUI
    print("Note: Install 'rich' for enhanced UI experience (pip install -r requirements.txt)")


async def _async_cleanup_background_tasks(verbose: bool = False):
    """
    Asynchronously clean up background tasks with proper await.
    Protected by multiple layers of error handling to prevent cancellation.

    IMPORTANT: This should ONLY be called during:
    - Application startup (after agent loading)
    - Application shutdown
    NOT during active message processing!
    """
    import time

    current_task = asyncio.current_task()
    # Filter out tasks, excluding current task and any tasks that look like main processing
    remaining = []
    for t in asyncio.all_tasks():
        if t is current_task or t.done():
            continue
        # Don't cancel tasks that look like they're part of main processing
        task_name = t.get_name()
        if 'process_message' in task_name.lower() or 'process_with_ui' in task_name.lower():
            if verbose:
                print(f"[CLEANUP] Skipping main processing task: {task_name}")
            continue
        remaining.append(t)

    if remaining:
        if verbose:
            print(f"\n{'='*80}")
            print(f"[CLEANUP] Found {len(remaining)} background tasks to clean up")
            print(f"[CLEANUP] Current task: {current_task.get_name()}")
            for i, task in enumerate(remaining, 1):
                coro_name = "unknown"
                try:
                    if hasattr(task, 'get_coro'):
                        coro = task.get_coro()
                        coro_name = f"{coro.__name__}" if hasattr(coro, '__name__') else str(coro)
                    elif hasattr(task, '_coro'):
                        coro_name = str(task._coro)
                except:
                    pass
                print(f"[CLEANUP]   Task {i}: {task.get_name()} | Coro: {coro_name} | Done: {task.done()}")
            print(f"{'='*80}\n")

        # Cancel all tasks with shield to prevent our cancellation from affecting us
        for task in remaining:
            if not task.done():
                if verbose:
                    print(f"[CLEANUP] Cancelling task: {task.get_name()}")
                try:
                    task.cancel()
                except Exception as e:
                    if verbose:
                        print(f"[CLEANUP] Failed to cancel {task.get_name()}: {e}")

        # Actually wait for them to finish cancelling (with timeout and shield)
        if verbose:
            print(f"[CLEANUP] Waiting for tasks to complete cancellation (2s timeout)...")

        try:
            # Use shield to protect the gather from being cancelled by the tasks we're cleaning
            await asyncio.shield(
                asyncio.wait_for(
                    asyncio.gather(*remaining, return_exceptions=True),
                    timeout=2.0
                )
            )
            if verbose:
                print(f"[CLEANUP] All tasks completed cancellation successfully")
        except asyncio.TimeoutError:
            if verbose:
                print(f"[CLEANUP] Cleanup timed out after 2s, using sync fallback")
            time.sleep(1.0)  # Fallback to sync sleep
        except asyncio.CancelledError:
            # Even with shield, we got cancelled - use sync sleep as last resort
            if verbose:
                print(f"[CLEANUP] ⚠️  Cleanup itself was cancelled! Using sync fallback")
            time.sleep(1.0)
        except Exception as e:
            if verbose:
                print(f"[CLEANUP] Cleanup error: {e}, using sync fallback")
            time.sleep(1.0)

        if verbose:
            current_task = asyncio.current_task()
            still_remaining = [t for t in asyncio.all_tasks() if t is not current_task and not t.done()]
            if still_remaining:
                print(f"\n[CLEANUP] ⚠️  WARNING: {len(still_remaining)} tasks still active after cleanup")
                for i, task in enumerate(still_remaining, 1):
                    coro_name = "unknown"
                    try:
                        if hasattr(task, 'get_coro'):
                            coro = task.get_coro()
                            coro_name = f"{coro.__name__}" if hasattr(coro, '__name__') else str(coro)
                        elif hasattr(task, '_coro'):
                            coro_name = str(task._coro)
                    except:
                        pass
                    print(f"[CLEANUP]   Remaining Task {i}: {task.get_name()} | Coro: {coro_name}")
                print()
            else:
                print(f"[CLEANUP] ✓ All background tasks cleaned up successfully\n")


async def main():
    """Main entry point"""
    # Parse command-line arguments
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    simple_mode = "--simple" in sys.argv

    # Initialize UI (use simple UI if requested, otherwise use enhanced)
    print("Initializing UI...", flush=True)
    if simple_mode:
        from ui.claude_ui import ClaudeUI
        ui = ClaudeUI(verbose=verbose)
    else:
        ui = UI_CLASS(verbose=verbose)

    # Initialize orchestrator
    print("Initializing orchestrator...", flush=True)
    orchestrator = OrchestratorAgent(
        connectors_dir="connectors",
        verbose=verbose
    )
    print("Orchestrator initialized.", flush=True)

    try:
        # Display welcome
        ui.print_welcome(orchestrator.session_id)

        # Discover and load agents
        # Wrap in try/except to handle cancellations during agent loading
        if verbose:
            print(f"\n[MAIN] Starting agent discovery and loading...")
            current_task = asyncio.current_task()
            all_tasks = [t for t in asyncio.all_tasks() if t is not current_task]
            print(f"[MAIN] Background tasks before agent loading: {len(all_tasks)}\n")

        try:
            await orchestrator.discover_and_load_agents()
        except asyncio.CancelledError:
            if verbose:
                print("[MAIN] ⚠️  Agent loading was cancelled, cleaning up...")
            # Don't re-raise - continue with whatever agents loaded successfully
        except Exception as e:
            if verbose:
                print(f"[MAIN] Error during agent loading: {e}")
            # Continue anyway

        if verbose:
            current_task = asyncio.current_task()
            all_tasks = [t for t in asyncio.all_tasks() if t is not current_task]
            print(f"\n[MAIN] Agent loading complete")
            print(f"[MAIN] Background tasks after agent loading: {len(all_tasks)}")

        # CRITICAL: Aggressively cleanup ALL background tasks before interactive session
        # This is essential to prevent zombie tasks (like Task-22 async_generator_athrow)
        # from interfering with message processing
        if verbose:
            print("[MAIN] Aggressively cleaning up background tasks from agent initialization...\n")

        # Do multiple cleanup passes to ensure all tasks are terminated
        for cleanup_pass in range(3):
            if verbose:
                print(f"[MAIN] Cleanup pass {cleanup_pass + 1}/3")

            try:
                await _async_cleanup_background_tasks(verbose)
            except Exception as e:
                if verbose:
                    print(f"[MAIN] Cleanup exception: {e}, continuing anyway")

            # Wait a bit for tasks to fully terminate
            import time
            time.sleep(0.3)

            # Check if any tasks remain
            current_task = asyncio.current_task()
            remaining = [t for t in asyncio.all_tasks() if t is not current_task and not t.done()]
            if not remaining:
                if verbose:
                    print(f"[MAIN] ✓ All background tasks cleaned up after pass {cleanup_pass + 1}\n")
                break
            elif verbose:
                print(f"[MAIN] {len(remaining)} tasks still active, continuing cleanup...\n")

        # Final verification
        current_task = asyncio.current_task()
        final_remaining = [t for t in asyncio.all_tasks() if t is not current_task and not t.done()]
        if final_remaining and verbose:
            print(f"[MAIN] ⚠️  WARNING: {len(final_remaining)} background tasks still active after cleanup")
            for i, task in enumerate(final_remaining, 1):
                print(f"[MAIN]   Task {i}: {task.get_name()}")
            print()

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
        # Wrap in try/except to handle cancellations during session
        try:
            await run_interactive_session(orchestrator, ui)
        except asyncio.CancelledError:
            if verbose:
                print("[MAIN] Session was cancelled")
            ui.print_goodbye()
        except Exception as e:
            if verbose:
                print(f"[MAIN] Session error: {e}")
            ui.print_error(str(e))

    except KeyboardInterrupt:
        ui.print_goodbye()
    except asyncio.CancelledError:
        if verbose:
            print("[MAIN] Main function was cancelled")
        ui.print_goodbye()
    except Exception as e:
        ui.print_error(str(e))
        if verbose:
            import traceback
            traceback.print_exc()
    finally:
        # Clean up orchestrator and any remaining tasks
        try:
            await orchestrator.cleanup()
        except asyncio.CancelledError:
            if verbose:
                print("[MAIN] Orchestrator cleanup was cancelled, using async cleanup")
            try:
                await _async_cleanup_background_tasks(verbose)
            except Exception:
                pass  # Best effort cleanup
        except Exception as e:
            if verbose:
                print(f"[MAIN] Error during orchestrator cleanup: {e}")


async def run_interactive_session(orchestrator: OrchestratorAgent, ui):
    """
    Run the main interactive session loop

    Args:
        orchestrator: The orchestration agent
        ui: The user interface
    """
    if ui.verbose:
        print(f"\n[SESSION] Starting interactive session")
        current_task = asyncio.current_task()
        all_tasks = [t for t in asyncio.all_tasks() if t is not current_task]
        print(f"[SESSION] Background tasks at session start: {len(all_tasks)}")
        if all_tasks:
            for i, task in enumerate(all_tasks, 1):
                coro_name = "unknown"
                try:
                    if hasattr(task, 'get_coro'):
                        coro = task.get_coro()
                        coro_name = f"{coro.__name__}" if hasattr(coro, '__name__') else str(coro)
                    elif hasattr(task, '_coro'):
                        coro_name = str(task._coro)
                except:
                    pass
                print(f"[SESSION]   Task {i}: {task.get_name()} | Coro: {coro_name}")
        print(f"[SESSION] Ready for user input\n")

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

            # Handle stats command
            if user_input.lower() in ['stats', 'statistics', 'status']:
                show_stats_command(orchestrator, ui)
                continue

            # Handle agents command
            if user_input.lower() == 'agents':
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
        except asyncio.CancelledError:
            # Handle cancellation gracefully - DO NOT cleanup background tasks here!
            # Cleanup during active session can cause more cancellations
            if ui.verbose:
                print("[SESSION] Operation cancelled, continuing session without cleanup")
            ui.print_error("Operation was cancelled.")

            # Just continue the session - background tasks will be cleaned up at session end
            # Aggressive cleanup here causes cascading cancellations
            continue
        except Exception as e:
            ui.print_error(str(e))
            if ui.verbose:
                import traceback
                traceback.print_exc()


async def process_with_ui(
    orchestrator: OrchestratorAgent,
    user_message: str,
    ui
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
        # Simple direct call - no isolation, no shields
        # MCP connections stay open, managed by agents themselves
        response = await orchestrator.process_message(user_message)
        return response

    except asyncio.CancelledError:
        # Handle cancellation gracefully
        if ui.verbose:
            print(f"[PROCESS] Operation was cancelled")
        return "⚠️ Operation was cancelled."

    except Exception as e:
        # Handle any other errors
        if ui.verbose:
            print(f"[PROCESS] Error during processing: {e}")
            import traceback
            traceback.print_exc()
        return f"⚠️ An error occurred: {str(e)}"

    finally:
        # Restore original method
        orchestrator.call_sub_agent = original_call_sub_agent


def show_help(orchestrator: OrchestratorAgent, ui):
    """Display help information"""
    ui.print_help(orchestrator)


def show_stats_command(orchestrator: OrchestratorAgent, ui):
    """Display session statistics"""
    # Get analytics from orchestrator if available
    stats = {
        'message_count': getattr(ui, 'message_count', 0),
        'agent_calls': getattr(ui, 'agent_call_count', 0),
    }

    # Add agent-specific stats if available
    if hasattr(orchestrator, 'analytics_collector') and orchestrator.analytics_collector:
        agent_stats = {}
        for agent_name in orchestrator.agent_capabilities.keys():
            metrics = orchestrator.analytics_collector.get_agent_metrics(agent_name)
            if metrics and metrics.total_calls > 0:
                agent_stats[agent_name] = {
                    'calls': metrics.total_calls,
                    'avg_time_ms': metrics.p50_latency_ms,
                    'success_rate': (metrics.successes / metrics.total_calls * 100) if metrics.total_calls > 0 else 0
                }

        if agent_stats:
            stats['agent_stats'] = agent_stats

    ui.print_session_summary(stats)


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


if __name__ == "__main__":
    asyncio.run(main())
