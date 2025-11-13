#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import OrchestratorAgent
from ui.enhanced_terminal_ui import EnhancedTerminalUI


async def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    simple_ui = "--simple" in sys.argv

    if simple_ui:
        orchestrator = OrchestratorAgent(connectors_dir="connectors", verbose=verbose)
        await orchestrator.run_interactive()
        return

    ui = EnhancedTerminalUI(verbose=verbose)

    try:
        orchestrator = OrchestratorAgent(connectors_dir="connectors", verbose=False)
        ui.print_header(orchestrator.session_id)
        ui.console.print("[bold cyan]🔌 Discovering Agents...[/bold cyan]\n")
        await orchestrator.discover_and_load_agents()

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

        if loaded_agents:
            from rich.table import Table
            from rich import box

            table = Table(
                show_header=True,
                header_style="bold cyan",
                border_style="dim",
                box=box.ROUNDED
            )
            table.add_column("Agent", style="white")
            table.add_column("Capabilities", style="dim")

            for agent in loaded_agents:
                caps_text = ", ".join(agent['capabilities'])
                if agent['total'] > 3:
                    caps_text += f" (+{agent['total']-3} more)"
                table.add_row(agent['name'].title(), caps_text)

            ui.console.print(table)
            ui.console.print()

        ui.print_loaded_summary(len(loaded_agents), len(failed_agents))
        await run_interactive_session(orchestrator, ui)

    except KeyboardInterrupt:
        ui.print_goodbye()
    except Exception as e:
        ui.print_error(str(e))
        if verbose:
            import traceback
            ui.print_error(str(e), traceback.format_exc())
    finally:
        if 'orchestrator' in locals():
            await orchestrator.cleanup()


async def run_interactive_session(orchestrator: OrchestratorAgent, ui: EnhancedTerminalUI):
    message_count = 0

    while True:
        try:
            ui.print_prompt()
            user_input = input().strip()

            if user_input.lower() in ['exit', 'quit', 'bye', 'q']:
                import time
                duration = time.time() - orchestrator.analytics.start_time if hasattr(orchestrator, 'analytics') else 0
                stats = {
                    'duration': f"{int(duration)}s" if duration else "N/A",
                    'message_count': message_count,
                    'agent_calls': sum(ui.agent_calls.values()),
                    'success_rate': "N/A"
                }
                ui.print_session_stats(stats)
                ui.print_goodbye()
                break

            if not user_input:
                continue

            message_count += 1
            ui.print_thinking()
            response = await process_with_ui(orchestrator, user_input, ui)
            ui.print_response(response)

        except KeyboardInterrupt:
            ui.print_goodbye()
            break
        except Exception as e:
            ui.print_error(str(e))
            if ui.verbose:
                import traceback
                ui.print_error(str(e), traceback.format_exc())


async def process_with_ui(
    orchestrator: OrchestratorAgent,
    user_message: str,
    ui: EnhancedTerminalUI
) -> str:
    original_call_sub_agent = orchestrator.call_sub_agent

    async def wrapped_call_sub_agent(agent_name: str, instruction: str, context: dict = None):
        ui.print_tool_call(agent_name, "processing...")

        try:
            result = await original_call_sub_agent(agent_name, instruction, context)
            success = not (result.startswith("Error") or result.startswith("⚠️"))
            ui.print_tool_result(success, result[:50] if success else result[:100])
            return result
        except Exception as e:
            ui.print_tool_result(False, str(e)[:100])
            raise

    orchestrator.call_sub_agent = wrapped_call_sub_agent

    try:
        response = await orchestrator.process_message(user_message)
        return response
    finally:
        orchestrator.call_sub_agent = original_call_sub_agent


if __name__ == "__main__":
    asyncio.run(main())
