import os
import json
import sys  # Added for verbose flag
import asyncio
import traceback
from typing import Any, Dict, List, Optional
import google.generativeai as genai
import google.generativeai.protos as protos
from dotenv import load_dotenv
from pathlib import Path
import importlib.util

load_dotenv()

# === ANSI COLOR CODES ===
class C:
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

# Configure Gemini
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY environment variable")

genai.configure(api_key=GOOGLE_API_KEY)


class OrchestratorAgent:
    """Main orchestration agent that coordinates specialized sub-agents"""
    
    def __init__(self, connectors_dir: str = "connectors", verbose: bool = False):
        self.connectors_dir = Path(connectors_dir)
        self.sub_agents: Dict[str, Any] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}
        self.verbose = verbose  # Set to True for detailed logging
        
        self.system_prompt = """You are an AI orchestration system that coordinates specialized agents to help users accomplish complex tasks across multiple platforms and tools.

Your core purpose is to be a highly capable, reliable workspace assistant that understands user intent, breaks down complex requests into actionable steps, and seamlessly coordinates specialized agents to deliver results.

# Core Principles

1. **User Intent Understanding**: Always seek to understand the true goal behind a user's request, not just the literal words. Ask clarifying questions when needed, but prefer taking initiative with reasonable assumptions when the intent is clear.

2. **Intelligent Decomposition**: Break complex tasks into logical sub-tasks. Consider dependencies, required context flow, and optimal sequencing. Some tasks can be parallelized conceptually, but you must execute them sequentially.

3. **Contextual Awareness**: Maintain awareness of conversation history and context. When delegating to agents, provide them with relevant context from previous steps or earlier in the conversation.

4. **Proactive Problem Solving**: Anticipate potential issues before they occur. If a task might fail due to missing information, gather that information first rather than attempting and failing.

5. **Clear Communication**: Provide users with clear, concise updates about what you're doing and what you've accomplished. Avoid unnecessary technical jargon.

# Working with Specialized Agents

You coordinate specialized agents, each with their own domain expertise. Available agents and their capabilities will be provided to you as tools.

When delegating to an agent:
- **Be specific and complete**: Provide clear, unambiguous instructions that include all necessary context
- **Pass relevant context**: If information from a previous step is needed, explicitly include it in the context parameter
- **Set clear expectations**: Tell the agent exactly what output format or specific actions you need
- **Handle errors gracefully**: If an agent reports an error, try alternative approaches or gather missing information

# Task Execution Patterns

**Sequential Dependencies**: When task B requires output from task A:
1. Execute task A first
2. Extract relevant information from A's result
3. Pass that information as context to task B

**Information Gathering Then Action**: For tasks requiring confirmation or specific details:
1. First, gather all necessary information (search, list, query)
2. Present findings to user if confirmation is needed
3. Then execute the action with complete information

**Multi-Platform Coordination**: When working across multiple platforms:
1. Consider the logical flow across platforms
2. Maintain consistency in naming, formatting, and references
3. Provide a unified summary that connects actions across platforms

# Quality Standards

- **Accuracy**: Double-check critical details like IDs, names, and specific values before executing actions
- **Completeness**: Ensure tasks are fully completed, not just partially done
- **Efficiency**: Minimize unnecessary agent calls while ensuring thoroughness
- **Transparency**: Keep users informed of progress, especially for multi-step operations
- **Error Recovery**: When errors occur, explain what went wrong clearly and suggest solutions

# User Interaction Guidelines

- Respond naturally and conversationally, avoiding robotic or overly formal language
- Show your reasoning for important decisions, but keep explanations concise
- When you've completed a task successfully, provide a clear summary of what was done
- If you need more information, ask specific questions rather than generic ones
- Respect user preferences and working styles as they emerge in conversation

# Safety and Limitations

- Always confirm before taking irreversible actions (deletions, major changes, public posts)
- If a request is ambiguous and could have significant consequences, seek clarification
- Stay within the capabilities of your available agents - don't promise what you cannot deliver
- If you encounter repeated failures, explain the situation clearly rather than continuing to retry
- Be honest about limitations and uncertainties

Remember: Your goal is to be genuinely helpful, making users more productive and their work across platforms smoother and more connected. Think carefully, act decisively, and always keep the user's ultimate goal in mind."""
        
        self.model = None
        self.chat = None
        self.conversation_history = []
        
        self.schema_type_map = {
            "string": protos.Type.STRING,
            "number": protos.Type.NUMBER,
            "integer": protos.Type.INTEGER,
            "boolean": protos.Type.BOOLEAN,
            "object": protos.Type.OBJECT,
            "array": protos.Type.ARRAY,
        }

    async def _spinner(self, task: asyncio.Task, message: str):
        """Display an animated spinner while a task is running."""
        # If in verbose mode, just await the task and let the logs print
        if self.verbose:
            await task  # This will still raise exceptions if the task fails
            return

        spinner_chars = ['|', '/', '─', '\\']
        start_time = asyncio.get_event_loop().time()
        
        # Hide cursor
        print(f"\033[?25l{C.CYAN}   {spinner_chars[0]} {message}...{C.ENDC}", end='', flush=True)
        
        i = 0
        while not task.done():
            await asyncio.sleep(0.1)
            i = (i + 1) % len(spinner_chars)
            elapsed = asyncio.get_event_loop().time() - start_time
            # Move cursor to beginning of line and print spinner
            print(f"\r{C.CYAN}   {spinner_chars[i]} {message}... ({elapsed:.1f}s){C.ENDC}", end='', flush=True)
        
        # Task is done, check for exceptions
        try:
            await task  # Re-await to raise exception if one occurred
        finally:
            # Clear the line and show cursor
            print(f"\r{' ' * 80}\r", end='', flush=True)
            print("\033[?25h", end='', flush=True)
    
    async def _load_single_agent(self, connector_file: Path) -> Optional[tuple]:
        """Load a single agent connector. Returns (agent_name, agent_instance, capabilities) or None on failure."""
        agent_name = connector_file.stem.replace("_agent", "")
        
        # Skip base_agent.py
        if agent_name == "base":
            return None
            
        if self.verbose:
            print(f"{C.CYAN}📦 Loading: {C.BOLD}{agent_name}{C.ENDC}{C.CYAN} agent...{C.ENDC}")
        
        try:
            # Dynamically import the module
            spec = importlib.util.spec_from_file_location(
                f"connectors.{agent_name}_agent",
                connector_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for AgentClass in the module
            if not hasattr(module, 'Agent'):
                print(f"{C.RED}  ✗ No 'Agent' class found in {connector_file}{C.ENDC}")
                return None
            
            agent_class = module.Agent

            # Try to initialize with verbose flag if the agent supports it
            try:
                agent_instance = agent_class(verbose=self.verbose)
            except TypeError:
                # Agent doesn't support verbose parameter
                agent_instance = agent_class()

            # Set verbose as an attribute for agents that support it
            if hasattr(agent_instance, 'verbose'):
                agent_instance.verbose = self.verbose

            # Initialize the agent
            try:
                await agent_instance.initialize()
                
                # Get agent capabilities
                capabilities = await agent_instance.get_capabilities()
                
                if self.verbose:
                    print(f"{C.GREEN}  ✓ Loaded {agent_name} with {len(capabilities)} capabilities{C.ENDC}")
                    for cap in capabilities[:3]:  # Show first 3
                        print(f"{C.GREEN}    - {cap}{C.ENDC}")
                    if len(capabilities) > 3:
                        print(f"{C.GREEN}    ... and {len(capabilities) - 3} more{C.ENDC}")
                
                return (agent_name, agent_instance, capabilities)
                
            except Exception as init_error:
                print(f"{C.RED}  ✗ Failed to initialize {agent_name}: {init_error}{C.ENDC}")
                if self.verbose:
                    traceback.print_exc()
                # Clean up the failed agent
                try:
                    if hasattr(agent_instance, 'cleanup'):
                        await agent_instance.cleanup()
                except:
                    pass
                return None
                
        except Exception as e:
            # Always print errors
            print(f"{C.RED}  ✗ Failed to load {agent_name}: {e}{C.ENDC}")
            if self.verbose:
                traceback.print_exc()
            return None
    
    async def discover_and_load_agents(self):
        """Automatically discover and load all agent connectors"""
        if self.verbose:
            print(f"\n{C.YELLOW}{'='*60}{C.ENDC}")
            print(f"{C.BOLD}{C.CYAN}🔍 Discovering Agent Connectors...{C.ENDC}")
            print(f"{C.YELLOW}{'='*60}{C.ENDC}\n")
        
        if not self.connectors_dir.exists():
            print(f"{C.RED}✗ Connectors directory '{self.connectors_dir}' not found!{C.ENDC}")
            print(f"{C.YELLOW}Creating directory...{C.ENDC}")
            self.connectors_dir.mkdir(parents=True, exist_ok=True)
            return
        
        # Find all Python files in connectors directory
        connector_files = list(self.connectors_dir.glob("*_agent.py"))
        
        if not connector_files:
            print(f"{C.YELLOW}⚠ No agent connectors found in '{self.connectors_dir}'{C.ENDC}")
            print(f"{C.YELLOW}  Expected files matching pattern: *_agent.py{C.ENDC}")
            return
        
        # Load agents sequentially to avoid concurrent initialization issues
        for connector_file in connector_files:
            result = await self._load_single_agent(connector_file)
            if result:
                agent_name, agent_instance, capabilities = result
                self.sub_agents[agent_name] = agent_instance
                self.agent_capabilities[agent_name] = capabilities
        
        print(f"\n{C.GREEN}✓ Loaded {len(self.sub_agents)} agent(s) successfully.{C.ENDC}")
        if self.verbose:
            print(f"{C.YELLOW}{'='*60}{C.ENDC}\n")
    
    def _create_agent_tools(self) -> List[protos.FunctionDeclaration]:
        """Convert sub-agents into Gemini tools"""
        tools = []
        
        for agent_name, capabilities in self.agent_capabilities.items():
            # Create a tool for each agent
            tool = protos.FunctionDeclaration(
                name=f"use_{agent_name}_agent",
                description=f"""Use the {agent_name} agent to perform tasks. 
                
Capabilities: {', '.join(capabilities)}

This agent can handle complex requests related to {agent_name}. 
Provide a clear instruction describing what you want to accomplish.""",
                parameters=protos.Schema(
                    type_=protos.Type.OBJECT,
                    properties={
                        "instruction": protos.Schema(
                            type_=protos.Type.STRING,
                            description=f"Clear instruction for the {agent_name} agent"
                        ),
                        "context": protos.Schema(
                            # Context can be any type, so we allow object/array/string
                            type_=protos.Type.OBJECT, 
                            description="Optional context or data from previous steps"
                        )
                    },
                    required=["instruction"]
                )
            )
            tools.append(tool)
        
        return tools
    
    async def call_sub_agent(self, agent_name: str, instruction: str, context: Any = None) -> str:
        """Execute a task using a specialized sub-agent"""
        if agent_name not in self.sub_agents:
            return f"Error: Agent '{agent_name}' not found"
        
        if self.verbose:
            print(f"\n{C.MAGENTA}{'─'*60}{C.ENDC}")
            print(f"{C.MAGENTA}🤖 Delegating to {C.BOLD}{agent_name}{C.ENDC}{C.MAGENTA} agent{C.ENDC}")
            print(f"{C.MAGENTA}{'─'*60}{C.ENDC}")
            print(f"{C.CYAN}Instruction: {instruction}{C.ENDC}")
        
        # --- START FIX ---
        context_str = ""
        if context:
            # If context is a dict/object, JSON-serialize it. Otherwise, cast to string.
            if isinstance(context, (dict, list)): 
                context_str = json.dumps(context, indent=2)
            else:
                context_str = str(context)
            
            if self.verbose:
                print(f"{C.CYAN}Context: {context_str[:200]}...{C.ENDC}")
        # --- END FIX ---
        
        try:
            agent = self.sub_agents[agent_name]
            
            # Build the full prompt with context_str
            full_instruction = instruction
            if context_str:
                full_instruction = f"Context from previous steps:\n{context_str}\n\nTask: {instruction}"
            
            # Execute the agent
            result = await agent.execute(full_instruction)
            
            if self.verbose:
                print(f"{C.GREEN}✓ {agent_name} completed successfully{C.ENDC}")
                print(f"{C.MAGENTA}{'─'*60}{C.ENDC}\n")
            
            return result
            
        except Exception as e:
            error_msg = f"Error executing {agent_name} agent: {str(e)}"
            print(f"{C.RED}✗ {error_msg}{C.ENDC}")
            if self.verbose:
                traceback.print_exc()
            return error_msg
    
    async def process_message(self, user_message: str) -> str:
        """Process a user message with orchestration"""
        
        # Initialize on first message
        if not self.chat:
            # Run the discovery task with a spinner
            discover_task = asyncio.create_task(self.discover_and_load_agents())
            await self._spinner(discover_task, "Discovering agents")
            
            if not self.sub_agents:
                return "No agents available. Please add agent connectors to the 'connectors' directory."
            
            # Create model with agent tools
            agent_tools = self._create_agent_tools()
            
            self.model = genai.GenerativeModel(
                'models/gemini-2.5-pro',
                system_instruction=self.system_prompt,
                tools=agent_tools
            )
            
            self.chat = self.model.start_chat(history=self.conversation_history)
        
        # Create and run the initial send task with a spinner
        send_task = asyncio.create_task(self.chat.send_message_async(user_message))
        await self._spinner(send_task, "Thinking")
        response = send_task.result()
        
        # Handle function calling loop
        max_iterations = 15
        iteration = 0
        
        while iteration < max_iterations:
            parts = response.candidates[0].content.parts
            has_function_call = any(
                hasattr(part, 'function_call') and part.function_call 
                for part in parts
            )
            
            if not has_function_call:
                break
            
            # Get the function call
            function_call = None
            for part in parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_call = part.function_call
                    break
            
            if not function_call:
                break
            
            tool_name = function_call.name
            
            # Extract agent name from tool name (use_X_agent -> X)
            if tool_name.startswith("use_") and tool_name.endswith("_agent"):
                agent_name = tool_name[4:-6]  # Remove "use_" and "_agent"
            else:
                agent_name = tool_name
            
            # Convert protobuf args to dict
            args = self._deep_convert_proto_args(function_call.args)
            
            instruction = args.get("instruction", "")
            context = args.get("context", "") # This will be a dict/list if passed by LLM
            
            # Call the sub-agent with a spinner
            agent_task = asyncio.create_task(
                self.call_sub_agent(agent_name, instruction, context)
            )
            await self._spinner(agent_task, f"Running {agent_name} agent")
            result = agent_task.result()
            
            # Send result back to orchestrator with a spinner
            response_task = asyncio.create_task(
                self.chat.send_message_async(
                    genai.protos.Content(
                        parts=[genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=tool_name,
                                response={"result": result}
                            )
                        )]
                    )
                )
            )
            await self._spinner(response_task, "Synthesizing results")
            response = response_task.result()
            
            iteration += 1
        
        if iteration >= max_iterations:
            print(f"{C.YELLOW}⚠ Warning: Reached maximum orchestration iterations{C.ENDC}")
        
        self.conversation_history = self.chat.history
        return response.text
    
    def _deep_convert_proto_args(self, value: Any) -> Any:
        """Recursively converts Protobuf composite types into standard Python dicts/lists"""
        type_str = str(type(value))
        if "MapComposite" in type_str:
            return {k: self._deep_convert_proto_args(v) for k, v in value.items()}
        elif "RepeatedComposite" in type_str:
            return [self._deep_convert_proto_args(item) for item in value]
        else:
            return value
    
    async def cleanup(self):
        """Cleanup all sub-agents"""
        if self.verbose:
            print(f"\n{C.YELLOW}Shutting down agents...{C.ENDC}")
            
        for agent_name, agent in list(self.sub_agents.items()):
            try:
                if hasattr(agent, 'cleanup'):
                    await agent.cleanup()
                if self.verbose:
                    print(f"{C.GREEN}  ✓ {agent_name} shut down{C.ENDC}")
            except Exception as e:
                # Always print errors
                print(f"{C.RED}  ✗ Error shutting down {agent_name}: {e}{C.ENDC}")
    
    async def run_interactive(self):
        """Run interactive chat session"""
        print(f"\n{C.YELLOW}{'='*60}{C.ENDC}")
        print(f"{C.BOLD}{C.CYAN}🎭 Multi-Agent Orchestration System{C.ENDC}")
        print(f"{C.YELLOW}Mode: {'Verbose' if self.verbose else 'Clean'}{C.ENDC}")
        print(f"{C.YELLOW}{'='*60}{C.ENDC}\n")
        
        try:
            while True:
                user_input = input(f"{C.BOLD}{C.BLUE}You: {C.ENDC}").strip()
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print(f"\n{C.GREEN}Goodbye! 👋{C.ENDC}")
                    break
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                try:
                    # The spinners will play here
                    response = await self.process_message(user_input)
                    # The final response is printed clearly
                    print(f"\n{C.BOLD}{C.GREEN}🎭 Orchestrator:{C.ENDC}\n{response}\n")
                    
                except Exception as e:
                    # Errors are always printed
                    print(f"\n{C.RED}✗ An error occurred: {str(e)}{C.ENDC}")
                    if self.verbose:
                        traceback.print_exc()
        
        finally:
            await self.cleanup()
    
    def _show_help(self):
        """Display help information"""
        print(f"\n{C.CYAN}{'='*60}{C.ENDC}")
        print(f"{C.BOLD}Available Agents:{C.ENDC}")
        for agent_name, capabilities in self.agent_capabilities.items():
            print(f"\n{C.GREEN}{agent_name.upper()}{C.ENDC}")
            for cap in capabilities:
                print(f"  • {cap}")
        print(f"\n{C.CYAN}{'='*60}{C.ENDC}")
        print(f"{C.YELLOW}Commands:{C.ENDC}")
        print(f"  help  - Show this help")
        print(f"  exit  - Exit the system")
        print(f"{C.CYAN}{'='*60}{C.ENDC}\n")


async def main():
    """Main entry point"""
    # Check for --verbose flag
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    
    orchestrator = OrchestratorAgent(connectors_dir="connectors", verbose=verbose)
    await orchestrator.run_interactive()


if __name__ == "__main__":
    asyncio.run(main())