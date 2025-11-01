import os
import json
import sys  # Added for verbose flag
import asyncio
import traceback
import uuid
import hashlib  # For operation key hashing
from typing import Any, Dict, List, Optional
import google.generativeai as genai
import google.generativeai.protos as protos
from dotenv import load_dotenv
from pathlib import Path
import importlib.util

# Import LLM abstraction layer
from llms.base_llm import BaseLLM, LLMConfig
from llms.gemini_flash import GeminiFlash

# Import intelligence components for smart agents
from connectors.agent_intelligence import WorkspaceKnowledge, SharedContext

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

    def __init__(self, connectors_dir: str = "connectors", verbose: bool = False, llm: Optional[BaseLLM] = None):
        self.connectors_dir = Path(connectors_dir)
        self.sub_agents: Dict[str, Any] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}
        self.agent_health: Dict[str, Dict[str, Any]] = {}  # Track agent health status
        self.verbose = verbose  # Set to True for detailed logging

        # LLM abstraction - use provided LLM or default to Gemini Flash
        if llm is None:
            # Default LLM: Gemini 2.5 Flash
            self.llm = GeminiFlash(LLMConfig(
                model_name='models/gemini-2.5-flash',
                temperature=0.7
            ))
        else:
            self.llm = llm

        if self.verbose:
            print(f"{C.CYAN}🧠 Using LLM: {self.llm}{C.ENDC}")

        # Intelligence components for smart cross-agent coordination
        self.knowledge_base = WorkspaceKnowledge()
        self.session_id = str(uuid.uuid4())
        self.shared_context = SharedContext(self.session_id)

        # Feature #20: Confirmation preferences
        self.confirmation_prefs = {
            'always_confirm_deletes': True,
            'always_confirm_bulk': True,
            'bulk_threshold': 5,
            'confirm_public_posts': True,
            'confirm_ambiguous': True
        }

        # Feature #8: Intelligent Retry tracking
        self.retry_tracker: Dict[str, Dict[str, Any]] = {}  # Track retry attempts per operation
        self.max_retry_attempts = 3  # Maximum retries before suggesting alternative

        # Feature #11: Simple progress tracking for streaming
        self.operation_count = 0  # Track number of operations in current request

        if self.verbose:
            print(f"{C.CYAN}🧠 Intelligence enabled: Session {self.session_id[:8]}...{C.ENDC}")
        
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
- **Trust final agent status**: Agents may encounter errors and successfully retry. Focus on the agent's final result, not interim errors during execution.
- **"✓ agent completed successfully" means SUCCESS**: If you see this message, the agent accomplished the task even if there were errors during execution. Parse the agent's response for created resources (like "Created KAN-20" or "issue KAN-18").
- **Error recovery is normal**: Agents are designed to handle errors intelligently through retry logic. Don't interpret retry attempts or interim errors as failures if the agent ultimately succeeds.

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
        """Load a single agent connector. Returns (agent_name, agent_instance, capabilities, messages) or None on failure."""
        agent_name = connector_file.stem.replace("_agent", "")
        messages = []  # Buffer messages for this agent

        # Skip base_agent.py and intelligence module
        if agent_name in ["base", "agent_intelligence"]:
            return None

        if self.verbose:
            messages.append(f"{C.CYAN}📦 Loading: {C.BOLD}{agent_name}{C.ENDC}{C.CYAN} agent...{C.ENDC}")

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
                messages.append(f"{C.RED}  ✗ No 'Agent' class found in {connector_file}{C.ENDC}")
                return None

            agent_class = module.Agent

            # Try to initialize with intelligence components (smart agents) or fallback to legacy
            try:
                # Try with full intelligence support + LLM abstraction
                agent_instance = agent_class(
                    verbose=self.verbose,
                    shared_context=self.shared_context,
                    knowledge_base=self.knowledge_base,
                    llm=self.llm
                )
            except TypeError:
                # Fallback: Try without LLM
                try:
                    agent_instance = agent_class(
                        verbose=self.verbose,
                        shared_context=self.shared_context,
                        knowledge_base=self.knowledge_base
                    )
                except TypeError:
                    # Fallback: Try with just verbose
                    try:
                        agent_instance = agent_class(verbose=self.verbose)
                    except TypeError:
                        # Fallback: No parameters (legacy agent)
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
                    messages.append(f"{C.GREEN}  ✓ Loaded {agent_name} with {len(capabilities)} capabilities{C.ENDC}")
                    for cap in capabilities[:3]:  # Show first 3
                        messages.append(f"{C.GREEN}    - {cap}{C.ENDC}")
                    if len(capabilities) > 3:
                        messages.append(f"{C.GREEN}    ... and {len(capabilities) - 3} more{C.ENDC}")

                return (agent_name, agent_instance, capabilities, messages)

            except Exception as init_error:
                messages.append(f"{C.RED}  ✗ Failed to initialize {agent_name}: {init_error}{C.ENDC}")
                if self.verbose:
                    messages.append(f"{C.RED}    {traceback.format_exc()}{C.ENDC}")
                # Clean up the failed agent
                try:
                    if hasattr(agent_instance, 'cleanup'):
                        await agent_instance.cleanup()
                except:
                    pass
                return (agent_name, None, None, messages)

        except Exception as e:
            # Buffer errors instead of printing
            messages.append(f"{C.RED}  ✗ Failed to load {agent_name}: {e}{C.ENDC}")
            if self.verbose:
                messages.append(f"{C.RED}    {traceback.format_exc()}{C.ENDC}")
            return (agent_name, None, None, messages)
    
    async def discover_and_load_agents(self):
        """Automatically discover and load all agent connectors in parallel"""
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

        # Load all agents in parallel using asyncio.gather
        print(f"{C.CYAN}Loading {len(connector_files)} agent(s) in parallel...{C.ENDC}\n")

        load_tasks = [self._load_single_agent(f) for f in connector_files]
        results = await asyncio.gather(*load_tasks, return_exceptions=True)

        # Process results and print buffered messages
        successful = 0
        failed = 0

        for result in results:
            if result is None:
                continue

            if isinstance(result, Exception):
                failed += 1
                print(f"{C.RED}✗ Exception during loading: {result}{C.ENDC}")
                continue

            agent_name, agent_instance, capabilities, messages = result

            # Print buffered messages
            for msg in messages:
                print(msg)

            # Store agent if successfully loaded
            if agent_instance is not None and capabilities is not None:
                self.sub_agents[agent_name] = agent_instance
                self.agent_capabilities[agent_name] = capabilities
                self.agent_health[agent_name] = {
                    'status': 'healthy',
                    'last_success': asyncio.get_event_loop().time(),
                    'error_count': 0
                }
                successful += 1
            else:
                # Agent failed to load - mark as unavailable (Feature #15: Graceful Degradation)
                self.agent_health[agent_name] = {
                    'status': 'unavailable',
                    'last_failure': asyncio.get_event_loop().time(),
                    'error_count': 1,
                    'error_message': 'Failed to initialize'
                }
                failed += 1

        # Summary
        print(f"\n{C.GREEN}✓ Loaded {successful} agent(s) successfully.{C.ENDC}")
        if failed > 0:
            print(f"{C.YELLOW}⚠ {failed} agent(s) failed to load but system will continue.{C.ENDC}")

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
        """Execute a task using a specialized sub-agent with health checking"""
        if agent_name not in self.sub_agents:
            return f"Error: Agent '{agent_name}' not found"

        # Feature #15: Graceful Degradation - Check agent health before calling
        if agent_name in self.agent_health:
            health = self.agent_health[agent_name]
            if health['status'] == 'unavailable':
                error_msg = f"⚠️ {agent_name} agent is currently unavailable: {health.get('error_message', 'Unknown error')}"
                print(f"{C.YELLOW}{error_msg}{C.ENDC}")
                return error_msg

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

            # Update health status on success
            if agent_name in self.agent_health:
                self.agent_health[agent_name]['status'] = 'healthy'
                self.agent_health[agent_name]['last_success'] = asyncio.get_event_loop().time()
                self.agent_health[agent_name]['error_count'] = 0

            if self.verbose:
                print(f"{C.GREEN}✓ {agent_name} completed successfully{C.ENDC}")
                print(f"{C.MAGENTA}{'─'*60}{C.ENDC}\n")

            return result

        except Exception as e:
            # Update health status on failure
            if agent_name in self.agent_health:
                self.agent_health[agent_name]['error_count'] = self.agent_health[agent_name].get('error_count', 0) + 1
                self.agent_health[agent_name]['last_failure'] = asyncio.get_event_loop().time()
                self.agent_health[agent_name]['error_message'] = str(e)

                # Mark as degraded if error count exceeds threshold
                if self.agent_health[agent_name]['error_count'] >= 3:
                    self.agent_health[agent_name]['status'] = 'degraded'
                    print(f"{C.YELLOW}⚠️ {agent_name} agent marked as degraded after 3 failures{C.ENDC}")

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

            # Create model with agent tools using LLM abstraction
            agent_tools = self._create_agent_tools()

            # Set the system instruction on the LLM config
            self.llm.config.system_instruction = self.system_prompt

            # Set tools on the LLM (for Gemini, these are FunctionDeclarations)
            self.llm.set_tools(agent_tools)

            # Start chat session with function calling enabled
            self.chat = self.llm.start_chat(
                history=None,
                enable_function_calling=True
            )

        # Reset operation counter for this request
        self.operation_count = 0

        # Create and run the initial send task with a spinner
        send_task = asyncio.create_task(self.chat.send_message(user_message))
        await self._spinner(send_task, "Thinking")
        llm_response = send_task.result()

        # Get the raw response object for compatibility
        response = llm_response.metadata.get('response_object') if llm_response.metadata else None

        # Handle function calling loop
        # Increased from 15 to 30 for batch operations
        max_iterations = 30
        iteration = 0

        while iteration < max_iterations:
            # Ensure we have a valid response object
            if not response or not hasattr(response, 'candidates') or not response.candidates:
                break

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

            # Feature #20: Check if operation needs confirmation
            risk_info = self._detect_risky_operation(agent_name, instruction)

            if risk_info.get('needs_confirmation'):
                # Handle ambiguous operations (resolve before confirming)
                if risk_info['risk_type'] == 'ambiguous':
                    enhanced_instruction = self._resolve_ambiguity(agent_name, instruction, risk_info)
                    if enhanced_instruction is None:
                        # User cancelled during ambiguity resolution
                        result = "⚠️ Operation cancelled by user"
                        # Skip to next iteration
                        function_result = {'name': tool_name, 'result': result}
                        response_task = asyncio.create_task(
                            self.chat.send_message_with_functions("", function_result)
                        )
                        await self._spinner(response_task, "Updating")
                        llm_response = response_task.result()
                        response = llm_response.metadata.get('response_object') if llm_response.metadata else None
                        iteration += 1
                        continue
                    instruction = enhanced_instruction

                # Ask for confirmation for other risky operations
                if risk_info['risk_type'] != 'ambiguous':
                    confirmed = self._ask_confirmation(risk_info, instruction)
                    if not confirmed:
                        result = "⚠️ Operation cancelled by user"
                        # Skip to next iteration
                        function_result = {'name': tool_name, 'result': result}
                        response_task = asyncio.create_task(
                            self.chat.send_message_with_functions("", function_result)
                        )
                        await self._spinner(response_task, "Updating")
                        llm_response = response_task.result()
                        response = llm_response.metadata.get('response_object') if llm_response.metadata else None
                        iteration += 1
                        continue

            # Feature #8: Intelligent Retry - Track and enhance retries
            operation_key = self._get_operation_key(agent_name, instruction)
            retry_context = self._get_retry_context(operation_key)

            # Track this attempt (before execution)
            self._track_retry_attempt(operation_key, agent_name, instruction)

            # If this is a retry, query knowledge base for solutions
            known_solutions = []
            if retry_context and retry_context['previous_errors']:
                last_error_msg = retry_context['previous_errors'][-1]['message']
                known_solutions = self._query_error_solutions(agent_name, last_error_msg)

                if self.verbose and known_solutions:
                    print(f"{C.CYAN}💡 Found {len(known_solutions)} known solution(s) for this error{C.ENDC}")

            # Call the sub-agent with a spinner and timeout
            agent_task = asyncio.create_task(
                self.call_sub_agent(agent_name, instruction, context)
            )

            # Feature #11: Simple operation tracking
            self.operation_count += 1
            spinner_msg = f"Running {agent_name} agent"

            try:
                # Add 120-second timeout for agent operations
                await asyncio.wait_for(
                    self._spinner(agent_task, spinner_msg),
                    timeout=120.0
                )
                result = agent_task.result()

                # Feature #11: Show completion (simple checkmark)
                if self.verbose:
                    print(f"{C.GREEN}✓ {agent_name} completed{C.ENDC}")

                # Feature #8: Mark success if this was a retry
                if retry_context:
                    self._mark_retry_success(operation_key, solution_used="retry with same parameters")

            except asyncio.TimeoutError:
                error_msg = f"⚠️ {agent_name} agent timed out after 120 seconds. Operation may have completed but response was not received."
                result = error_msg
                print(f"{C.YELLOW}⚠ {agent_name} agent operation timed out{C.ENDC}")

                # Feature #8: Track this error
                self._track_retry_attempt(operation_key, agent_name, instruction, error=error_msg)

                # Add retry context to error if this is a retry
                if retry_context:
                    attempt_num = retry_context['attempt_number']
                    result += f"\n\n📊 Retry Info: This was attempt #{attempt_num}."

                    if attempt_num >= self.max_retry_attempts:
                        result += f"\n⚠️ Reached maximum retry attempts ({self.max_retry_attempts}). Consider:\n"
                        result += f"  • Breaking this into smaller steps\n"
                        result += f"  • Trying a different approach\n"
                        result += f"  • Checking if the service is available"

            except Exception as e:
                error_msg = f"⚠️ {agent_name} agent error: {str(e)}"
                result = error_msg
                print(f"{C.RED}✗ {agent_name} agent failed: {e}{C.ENDC}")

                # Feature #8: Track this error
                self._track_retry_attempt(operation_key, agent_name, instruction, error=str(e))

                # Add retry context and suggestions to error
                if retry_context:
                    attempt_num = retry_context['attempt_number']
                    result += f"\n\n📊 Retry Info: This was attempt #{attempt_num}."

                    # Add known solutions if available
                    if known_solutions:
                        result += f"\n\n💡 Known solutions for similar errors:\n"
                        for i, solution in enumerate(known_solutions[:3], 1):
                            result += f"  {i}. {solution}\n"

                    # Suggest giving up if too many retries
                    if attempt_num >= self.max_retry_attempts:
                        result += f"\n⚠️ Reached maximum retry attempts ({self.max_retry_attempts}). Suggesting:\n"
                        result += f"  • Try a different approach or tool\n"
                        result += f"  • Simplify the request\n"
                        result += f"  • Ask user for clarification"
                else:
                    # First attempt - add basic guidance
                    if "not found" in str(e).lower() or "invalid" in str(e).lower():
                        result += f"\n\n💡 Tip: Try using discovery/validation before this operation"
            
            # Send result back to orchestrator with a spinner
            function_result = {
                'name': tool_name,
                'result': result
            }

            response_task = asyncio.create_task(
                self.chat.send_message_with_functions("", function_result)
            )
            await self._spinner(response_task, "Synthesizing results")
            llm_response = response_task.result()

            # Get raw response for next iteration
            response = llm_response.metadata.get('response_object') if llm_response.metadata else None
            
            iteration += 1
        
        if iteration >= max_iterations:
            print(f"{C.YELLOW}⚠ Warning: Reached maximum orchestration iterations{C.ENDC}")
            print(f"{C.YELLOW}💡 Tip: Break complex tasks into smaller steps{C.ENDC}")

        # Feature #11: Simple completion indicator
        if self.operation_count > 0 and self.verbose:
            print(f"\n{C.GREEN}✅ Completed {self.operation_count} operation(s){C.ENDC}\n")

        # Update conversation history
        try:
            self.conversation_history = self.chat.get_history()
        except Exception as e:
            if self.verbose:
                print(f"{C.YELLOW}⚠ Could not update conversation history: {e}{C.ENDC}")

        # Extract text from final LLM response
        if llm_response and llm_response.text:
            return llm_response.text

        # Fallback: Try to extract from raw response object
        if response and hasattr(response, 'candidates') and response.candidates:
            try:
                # Try to get text property (may fail if there are function_call parts)
                return response.text
            except Exception:
                # Manual extraction from parts
                text_parts = []
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)

                if text_parts:
                    return '\n'.join(text_parts)

        # If we still don't have text, return a generic message
        return "⚠️ Task completed but response formatting failed. The operations were executed successfully."
    
    def _deep_convert_proto_args(self, value: Any) -> Any:
        """Recursively converts Protobuf composite types into standard Python dicts/lists"""
        type_str = str(type(value))
        if "MapComposite" in type_str:
            return {k: self._deep_convert_proto_args(v) for k, v in value.items()}
        elif "RepeatedComposite" in type_str:
            return [self._deep_convert_proto_args(item) for item in value]
        else:
            return value

    def _detect_risky_operation(self, agent_name: str, instruction: str) -> Dict[str, Any]:
        """
        Detect if an operation is risky and needs confirmation (Feature #20)

        Returns:
            Dict with:
                needs_confirmation: bool
                risk_type: str (delete, bulk, public, ambiguous)
                reason: str (explanation)
                suggestions: List[str] (alternatives or clarifications needed)
        """
        instruction_lower = instruction.lower()

        # IMPORTANT: Read-only operations NEVER need confirmation
        read_only_keywords = [
            'get', 'list', 'show', 'find', 'search', 'fetch', 'read', 'view',
            'display', 'query', 'retrieve', 'check', 'lookup', 'see'
        ]

        # Check if this is a read-only operation
        first_word = instruction_lower.split()[0] if instruction_lower.split() else ''
        if first_word in read_only_keywords or any(word in instruction_lower.split()[:3] for word in read_only_keywords):
            # This is a read operation - no confirmation needed
            return {'needs_confirmation': False}

        # Destructive operations
        destructive_keywords = ['delete', 'remove', 'close', 'archive', 'drop', 'destroy']
        if any(word in instruction_lower for word in destructive_keywords):
            if self.confirmation_prefs['always_confirm_deletes']:
                return {
                    'needs_confirmation': True,
                    'risk_type': 'destructive',
                    'reason': 'This operation will permanently delete or close items',
                    'suggestions': []
                }

        # Bulk operations
        bulk_indicators = ['all', 'every', 'each', 'multiple']
        if any(word in instruction_lower for word in bulk_indicators):
            if self.confirmation_prefs['always_confirm_bulk']:
                return {
                    'needs_confirmation': True,
                    'risk_type': 'bulk',
                    'reason': f'This appears to be a bulk operation affecting multiple items',
                    'suggestions': ['Consider processing items one at a time for safety']
                }

        # Public/broadcast operations
        public_indicators = ['everyone', 'all users', '@channel', '@here', 'company-wide']
        if any(word in instruction_lower for word in public_indicators):
            if self.confirmation_prefs['confirm_public_posts']:
                return {
                    'needs_confirmation': True,
                    'risk_type': 'public',
                    'reason': 'This will notify many people or post publicly',
                    'suggestions': ['Consider using a more targeted audience']
                }

        # Ambiguous operations (Notion database case) - only for write operations
        write_to_notion_keywords = ['add to', 'put in', 'create in', 'insert into', 'save to']
        if agent_name == 'notion' and any(phrase in instruction_lower for phrase in write_to_notion_keywords):
            # Check if we have metadata about databases
            if hasattr(self.sub_agents.get('notion'), 'metadata_cache'):
                databases = self.sub_agents['notion'].metadata_cache.get('databases', {})
                if len(databases) > 1:
                    return {
                        'needs_confirmation': True,
                        'risk_type': 'ambiguous',
                        'reason': f'Found {len(databases)} databases. Which one should I use?',
                        'suggestions': [f"- {db.get('title', 'Untitled')}" for db in databases.values()]
                    }

        return {'needs_confirmation': False}

    def _ask_confirmation(self, risk_info: Dict[str, Any], instruction: str) -> bool:
        """
        Ask user for confirmation before proceeding (Feature #20)

        Returns:
            bool: True if user confirmed, False if cancelled
        """
        print(f"\n{C.YELLOW}⚠️  CONFIRMATION REQUIRED{C.ENDC}")
        print(f"{C.YELLOW}{'─' * 60}{C.ENDC}")
        print(f"{C.CYAN}Operation: {instruction}{C.ENDC}")
        print(f"{C.YELLOW}Risk Type: {risk_info['risk_type'].upper()}{C.ENDC}")
        print(f"{C.YELLOW}Reason: {risk_info['reason']}{C.ENDC}")

        if risk_info.get('suggestions'):
            print(f"\n{C.CYAN}💡 Options:{C.ENDC}")
            for suggestion in risk_info['suggestions']:
                print(f"  {suggestion}")

        print(f"{C.YELLOW}{'─' * 60}{C.ENDC}")

        while True:
            response = input(f"{C.BOLD}Proceed? (yes/no): {C.ENDC}").strip().lower()
            if response in ['yes', 'y']:
                return True
            elif response in ['no', 'n']:
                print(f"{C.RED}✗ Operation cancelled by user{C.ENDC}")
                return False
            else:
                print(f"{C.YELLOW}Please answer 'yes' or 'no'{C.ENDC}")

    def _resolve_ambiguity(self, agent_name: str, instruction: str, risk_info: Dict[str, Any]) -> str:
        """
        Help user resolve ambiguous operations (Feature #20)

        For Notion: Ask which database to use
        For others: Ask for clarification

        Returns:
            str: Enhanced instruction with resolved ambiguity
        """
        if risk_info['risk_type'] == 'ambiguous' and agent_name == 'notion':
            print(f"\n{C.CYAN}🤔 Which database should I use?{C.ENDC}")

            suggestions = risk_info.get('suggestions', [])
            for i, suggestion in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion}")
            print(f"  0. Cancel")

            while True:
                try:
                    choice = input(f"{C.BOLD}Enter number: {C.ENDC}").strip()
                    choice_num = int(choice)

                    if choice_num == 0:
                        print(f"{C.RED}✗ Operation cancelled{C.ENDC}")
                        return None

                    if 1 <= choice_num <= len(suggestions):
                        selected = suggestions[choice_num - 1]
                        # Extract database name from suggestion
                        db_name = selected.replace('- ', '')
                        # Enhance instruction with specific database
                        enhanced = instruction.replace('my TODO', f'the "{db_name}" database')
                        enhanced = enhanced.replace('TODO', f'"{db_name}" database')
                        print(f"{C.GREEN}✓ Using database: {db_name}{C.ENDC}")
                        return enhanced
                    else:
                        print(f"{C.YELLOW}Invalid choice. Please try again.{C.ENDC}")
                except ValueError:
                    print(f"{C.YELLOW}Please enter a number.{C.ENDC}")

        return instruction

    def _get_operation_key(self, agent_name: str, instruction: str) -> str:
        """
        Generate unique key for an operation (Feature #8)

        Args:
            agent_name: Name of the agent
            instruction: Instruction being executed

        Returns:
            str: Hash key for tracking this operation
        """
        # Create hash from agent + instruction (first 100 chars)
        key_string = f"{agent_name}:{instruction[:100]}"
        return hashlib.md5(key_string.encode()).hexdigest()[:16]

    def _track_retry_attempt(self, operation_key: str, agent_name: str, instruction: str, error: str = None):
        """
        Track a retry attempt for an operation (Feature #8)

        Args:
            operation_key: Unique operation key
            agent_name: Agent being called
            instruction: Instruction being executed
            error: Error message if this attempt failed
        """
        if operation_key not in self.retry_tracker:
            self.retry_tracker[operation_key] = {
                'agent': agent_name,
                'instruction': instruction,
                'attempts': 0,
                'errors': [],
                'solutions_tried': [],
                'first_attempt': asyncio.get_event_loop().time()
            }

        self.retry_tracker[operation_key]['attempts'] += 1

        if error:
            self.retry_tracker[operation_key]['errors'].append({
                'message': error,
                'timestamp': asyncio.get_event_loop().time()
            })

    def _get_retry_context(self, operation_key: str) -> Optional[Dict[str, Any]]:
        """
        Get retry context for an operation (Feature #8)

        Returns None if this is first attempt, otherwise returns retry context
        """
        if operation_key not in self.retry_tracker:
            return None

        tracker = self.retry_tracker[operation_key]

        if tracker['attempts'] == 0:
            return None

        return {
            'attempt_number': tracker['attempts'] + 1,
            'previous_errors': tracker['errors'],
            'solutions_tried': tracker['solutions_tried']
        }

    def _query_error_solutions(self, agent_name: str, error_message: str) -> List[str]:
        """
        Query knowledge base for known solutions to this error (Feature #8)

        Args:
            agent_name: Agent that encountered the error
            error_message: Error message text

        Returns:
            List of suggested solutions
        """
        # Query knowledge base for error solutions
        error_solutions = self.knowledge_base.data.get('error_solutions', {})
        agent_solutions = error_solutions.get(agent_name, {})

        # Look for matching error patterns
        solutions = []
        for error_pattern, solution_info in agent_solutions.items():
            if error_pattern.lower() in error_message.lower():
                solutions.append(solution_info.get('solution', ''))

        return solutions

    def _mark_retry_success(self, operation_key: str, solution_used: str = None):
        """
        Mark a retry as successful and learn from it (Feature #8)

        Args:
            operation_key: Operation that succeeded
            solution_used: Description of what fixed the issue
        """
        if operation_key not in self.retry_tracker:
            return

        tracker = self.retry_tracker[operation_key]

        # If this was a retry (not first attempt) and it succeeded, learn from it
        if tracker['attempts'] > 1 and solution_used:
            # Extract the error pattern from previous errors
            if tracker['errors']:
                last_error = tracker['errors'][-1]['message']

                # Save this solution to knowledge base for future reference
                # This would be implemented in the knowledge base class
                # For now, just log it
                if self.verbose:
                    print(f"{C.GREEN}✓ Learned: {solution_used} fixed the issue{C.ENDC}")

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