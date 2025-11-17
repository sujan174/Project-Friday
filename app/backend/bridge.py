"""
Python Backend Bridge for Aerius Desktop App

This bridge connects the Electron frontend to the Project Aerius orchestrator.
Communication happens via JSON messages over stdin/stdout.
"""

import sys
import json
import asyncio
import os
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path to import from Project-Aerius
# The app folder is inside Project-Aerius, so go up 3 levels to get to Project-Aerius root
project_aerius_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_aerius_path))

try:
    from orchestrator import MultiAgentOrchestrator
    from config import Config
except ImportError as e:
    print(json.dumps({
        "type": "error",
        "message": f"Failed to import Project Aerius: {str(e)}",
        "details": "Make sure Project-Aerius is in the parent directory"
    }), flush=True)
    sys.exit(1)


class DesktopBridge:
    """Bridge between Electron frontend and Python orchestrator"""

    def __init__(self):
        self.orchestrator = None
        self.session_id = None

    def send_message(self, msg_type: str, data: Any = None):
        """Send JSON message to Electron frontend"""
        message = {
            "type": msg_type,
            "data": data
        }
        print(json.dumps(message), flush=True)

    async def initialize_orchestrator(self):
        """Initialize the orchestrator"""
        try:
            self.send_message("status", {"message": "Initializing orchestrator..."})

            # Initialize orchestrator without UI
            self.orchestrator = MultiAgentOrchestrator(verbose=False, use_ui=False)
            await self.orchestrator.initialize()

            self.session_id = self.orchestrator.session_id

            # Get agent capabilities
            agents = []
            for agent_name, capabilities in self.orchestrator.agent_capabilities.items():
                agents.append({
                    "name": agent_name,
                    "capabilities": capabilities,
                    "status": self.orchestrator.agent_health.get(agent_name, {}).get("status", "unknown")
                })

            self.send_message("initialized", {
                "session_id": self.session_id,
                "agents": agents
            })

        except Exception as e:
            self.send_message("error", {
                "message": "Failed to initialize orchestrator",
                "details": str(e)
            })
            raise

    async def process_message(self, user_message: str):
        """Process user message through orchestrator"""
        try:
            if not self.orchestrator:
                self.send_message("error", {"message": "Orchestrator not initialized"})
                return

            self.send_message("processing", {"message": user_message})

            # Send message to orchestrator
            response = await self.orchestrator.process_message(user_message)

            self.send_message("response", {
                "text": response,
                "timestamp": asyncio.get_event_loop().time()
            })

        except Exception as e:
            self.send_message("error", {
                "message": "Failed to process message",
                "details": str(e)
            })

    async def get_session_stats(self):
        """Get current session statistics"""
        try:
            if not self.orchestrator:
                self.send_message("error", {"message": "Orchestrator not initialized"})
                return

            stats = {
                "session_id": self.session_id,
                "message_count": len(self.orchestrator.conversation_history),
                "agent_calls": sum(1 for msg in self.orchestrator.conversation_history
                                  if msg.get("role") == "function"),
            }

            self.send_message("stats", stats)

        except Exception as e:
            self.send_message("error", {
                "message": "Failed to get stats",
                "details": str(e)
            })

    async def reset_session(self):
        """Reset the current session"""
        try:
            if self.orchestrator:
                await self.orchestrator.cleanup()

            await self.initialize_orchestrator()
            self.send_message("session_reset", {"session_id": self.session_id})

        except Exception as e:
            self.send_message("error", {
                "message": "Failed to reset session",
                "details": str(e)
            })

    async def handle_command(self, command: Dict[str, Any]):
        """Handle incoming command from frontend"""
        cmd_type = command.get("type")
        data = command.get("data", {})

        if cmd_type == "initialize":
            await self.initialize_orchestrator()
        elif cmd_type == "message":
            await self.process_message(data.get("text", ""))
        elif cmd_type == "stats":
            await self.get_session_stats()
        elif cmd_type == "reset":
            await self.reset_session()
        elif cmd_type == "shutdown":
            if self.orchestrator:
                await self.orchestrator.cleanup()
            sys.exit(0)
        else:
            self.send_message("error", {"message": f"Unknown command: {cmd_type}"})

    async def run(self):
        """Main run loop - read commands from stdin"""
        self.send_message("ready", {"message": "Bridge ready"})

        # Read from stdin in a non-blocking way
        loop = asyncio.get_event_loop()

        while True:
            try:
                # Read line from stdin
                line = await loop.run_in_executor(None, sys.stdin.readline)

                if not line:
                    break

                line = line.strip()
                if not line:
                    continue

                # Parse JSON command
                try:
                    command = json.loads(line)
                    await self.handle_command(command)
                except json.JSONDecodeError as e:
                    self.send_message("error", {
                        "message": "Invalid JSON",
                        "details": str(e)
                    })

            except Exception as e:
                self.send_message("error", {
                    "message": "Bridge error",
                    "details": str(e)
                })
                break


async def main():
    """Main entry point"""
    bridge = DesktopBridge()
    try:
        await bridge.run()
    except KeyboardInterrupt:
        if bridge.orchestrator:
            await bridge.orchestrator.cleanup()
    except Exception as e:
        print(json.dumps({
            "type": "fatal_error",
            "message": str(e)
        }), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    # Suppress output from orchestrator
    os.environ["AERIUS_DESKTOP_MODE"] = "1"

    asyncio.run(main())
