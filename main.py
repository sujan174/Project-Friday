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
    """
    import time

    current_task = asyncio.current_task()
    remaining = [t for t in asyncio.all_tasks() if t is not current_task and not t.done()]

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

        # Cancel all tasks
        for task in remaining:
            if not task.done():
                if verbose:
                    print(f"[CLEANUP] Cancelling task: {task.get_name()}")
                task.cancel()

        # Actually wait for them to finish cancelling (with timeout)
        if verbose:
            print(f"[CLEANUP] Waiting for tasks to complete cancellation (2s timeout)...")

        try:
            # Use wait_for with timeout to prevent hanging
            await asyncio.wait_for(
                asyncio.gather(*remaining, return_exceptions=True),
                timeout=2.0
            )
            if verbose:
                print(f"[CLEANUP] All tasks completed cancellation successfully")
        except asyncio.TimeoutError:
            if verbose:
                print(f"[CLEANUP] Cleanup timed out after 2s, using sync fallback")
            time.sleep(1.0)  # Fallback to sync sleep
        except asyncio.CancelledError:
            # Even cleanup is being cancelled - use sync sleep as last resort
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

        # CRITICAL: Cleanup of background tasks with proper await
        # Protected by error handling to prevent cancellation from affecting cleanup
        if verbose:
            print("[MAIN] Cleaning up background tasks from agent initialization...\n")

        try:
            await _async_cleanup_background_tasks(verbose)
        except Exception as e:
            if verbose:
                print(f"[MAIN] Cleanup exception: {e}, continuing anyway")

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
            # Handle cancellation gracefully
            ui.print_error("Operation was cancelled. Cleaning up...")
            # Clean up background tasks
            try:
                await _async_cleanup_background_tasks(ui.verbose)
            except Exception:
                pass  # Best effort cleanup
            # Continue the session instead of crashing
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
        # Clean up any lingering background tasks before processing
        if ui.verbose:
            print(f"\n[PROCESS] Starting message processing: '{user_message[:50]}...'")
            current_task = asyncio.current_task()
            all_tasks = [t for t in asyncio.all_tasks() if t is not current_task]
            print(f"[PROCESS] Background tasks before cleanup: {len(all_tasks)}")

        try:
            await _async_cleanup_background_tasks(ui.verbose)
        except Exception as e:
            if ui.verbose:
                print(f"[PROCESS] Cleanup exception (continuing): {e}")

        if ui.verbose:
            current_task = asyncio.current_task()
            all_tasks = [t for t in asyncio.all_tasks() if t is not current_task]
            print(f"[PROCESS] Background tasks after cleanup: {len(all_tasks)}")
            print(f"[PROCESS] Creating processing task...")

        # Create the processing task
        processing_task = asyncio.create_task(orchestrator.process_message(user_message))

        if ui.verbose:
            print(f"[PROCESS] Processing task created: {processing_task.get_name()}")
            print(f"[PROCESS] Awaiting response (300s timeout)...\n")

        # Process with timeout protection
        try:
            response = await asyncio.wait_for(processing_task, timeout=300.0)  # 5 minute timeout
            if ui.verbose:
                print(f"\n[PROCESS] ✓ Processing completed successfully")
            return response

        except asyncio.TimeoutError:
            if ui.verbose:
                print("[MAIN] Processing timed out, cancelling...")
            processing_task.cancel()
            try:
                await processing_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            return "⚠️ Operation timed out after 5 minutes. Please try a simpler request."

        except asyncio.CancelledError as ce:
            # Processing was cancelled by background task
            if ui.verbose:
                print(f"\n{'!'*80}")
                print(f"[PROCESS] ⚠️  CANCELLATION DETECTED!")
                print(f"[PROCESS] CancelledError: {ce}")

                # Show current task state
                current_task = asyncio.current_task()
                print(f"[PROCESS] Current task: {current_task.get_name()}")
                print(f"[PROCESS] Processing task done: {processing_task.done()}")
                if processing_task.done():
                    try:
                        exc = processing_task.exception()
                        print(f"[PROCESS] Processing task exception: {exc}")
                    except:
                        pass

                # Show all background tasks
                all_tasks = [t for t in asyncio.all_tasks() if t is not current_task and not t.done()]
                print(f"[PROCESS] Active background tasks: {len(all_tasks)}")
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
                    print(f"[PROCESS]   Task {i}: {task.get_name()} | Coro: {coro_name}")
                print(f"{'!'*80}\n")

            ui.print_error("Operation was cancelled. Please try again.")

            # Clean up background tasks
            if ui.verbose:
                print(f"[PROCESS] Attempting cleanup after cancellation...")
            try:
                await _async_cleanup_background_tasks(ui.verbose)
            except Exception as e:
                if ui.verbose:
                    print(f"[PROCESS] Cleanup after cancellation failed: {e}")

            # Return error message instead of raising
            return "⚠️ Operation was cancelled due to background task interference. Please try again."

        except Exception as e:
            # Handle any other errors normally
            if ui.verbose:
                print(f"[MAIN] Processing error: {e}")
            raise

    except Exception as e:
        # Outer catch-all to ensure we always restore the original method
        if ui.verbose:
            print(f"[MAIN] Outer exception: {e}")
        raise

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
