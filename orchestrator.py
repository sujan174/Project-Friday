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
from connectors.agent_logger import SessionLogger

# Import advanced intelligence system
from intelligence import (
    IntentClassifier, EntityExtractor, TaskDecomposer,
    ConfidenceScorer, ConversationContextManager
)

# Import terminal UI
from ui.terminal_ui import TerminalUI, Colors as C_NEW, Icons

# Old confirmation system removed - using MandatoryConfirmationEnforcer instead

# Import production utilities
from config import Config
from core.logger import get_logger
from core.input_validator import InputValidator
from core.error_handler import ErrorClassifier, format_error_for_user, DuplicateOperationDetector

# Import new enhancement systems
from core.retry_manager import RetryManager
from core.undo_manager import UndoManager, UndoableOperationType
from core.user_preferences import UserPreferenceManager
from core.analytics import AnalyticsCollector
from core.error_messaging import ErrorMessageEnhancer
from core.message_confirmation import MandatoryConfirmationEnforcer

logger = get_logger(__name__)
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

    @staticmethod
    def _safe_get_response_object(llm_response) -> Optional[Any]:
        """Safely extract response object from LLM response with null checks"""
        try:
            if not llm_response:
                return None
            if not hasattr(llm_response, 'metadata') or not llm_response.metadata:
                logger.warning("LLM response missing metadata")
                return None
            response_obj = llm_response.metadata.get('response_object')
            if not response_obj:
                logger.warning("Response object not in metadata")
                return None
            # Validate response structure
            if not hasattr(response_obj, 'candidates') or not response_obj.candidates:
                logger.warning("Response object missing candidates")
                return None
            return response_obj
        except AttributeError as e:
            logger.error(f"Error accessing response metadata: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error extracting response: {e}", exc_info=True)
            return None

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

        # Feature #20: Confirmation preferences (loaded from Config)
        # NOTE: Confirmations now handled by MandatoryConfirmationEnforcer for Slack/Notion
        self.confirmation_prefs = {
            'always_confirm_deletes': Config.CONFIRM_DELETES,
            'always_confirm_bulk': Config.CONFIRM_BULK_OPERATIONS,
            'bulk_threshold': 5,
            'confirm_public_posts': Config.CONFIRM_PUBLIC_POSTS,
            'confirm_ambiguous': True,
            'confirm_slack_messages': Config.CONFIRM_SLACK_MESSAGES,  # Always confirm Slack messages (send/notify actions)
            'confirm_jira_operations': Config.CONFIRM_JIRA_OPERATIONS,  # Always confirm Jira operations (create/update/delete)
        }

        # Feature #8: Intelligent Retry tracking
        self.retry_tracker: Dict[str, Dict[str, Any]] = {}  # Track retry attempts per operation
        self.max_retry_attempts = 3  # Maximum retries before suggesting alternative

        # Feature #21: Duplicate Operation Detection
        self.duplicate_detector = DuplicateOperationDetector(window_size=5, similarity_threshold=0.8)

        # Feature #11: Simple progress tracking for streaming
        self.operation_count = 0  # Track number of operations in current request

        # Session logging
        self.session_logger = SessionLogger(log_dir="logs", session_id=self.session_id)

        # Terminal UI
        self.ui = TerminalUI(verbose=self.verbose)

        # Advanced Intelligence System (Phase 1)
        self.intent_classifier = IntentClassifier(verbose=self.verbose)
        self.entity_extractor = EntityExtractor(verbose=self.verbose)
        self.task_decomposer = TaskDecomposer(
            agent_capabilities=self.agent_capabilities,
            verbose=self.verbose
        )
        self.confidence_scorer = ConfidenceScorer(verbose=self.verbose)
        self.context_manager = ConversationContextManager(
            session_id=self.session_id,
            verbose=self.verbose
        )

        # ===================================================================
        # NEW ENHANCEMENT SYSTEMS (Retry, Undo, Preferences, Analytics)
        # ===================================================================

        # 1. Retry Manager - Smart exponential backoff
        self.retry_manager = RetryManager(
            max_retries=self.max_retry_attempts,
            base_delay=1.0,
            max_delay=30.0,
            backoff_factor=2.0,
            jitter=True,
            verbose=self.verbose
        )

        # 2. Undo Manager - Undo destructive operations
        self.undo_manager = UndoManager(
            max_undo_history=20,
            default_ttl_seconds=3600,  # 1 hour
            verbose=self.verbose
        )

        # 3. User Preferences - Learn from user behavior
        user_id = os.environ.get("USER_ID", "default")
        self.user_prefs = UserPreferenceManager(
            user_id=user_id,
            min_confidence_threshold=0.7,
            verbose=self.verbose
        )

        # 4. Analytics - Performance monitoring
        self.analytics = AnalyticsCollector(
            session_id=self.session_id,
            max_latency_samples=1000,
            verbose=self.verbose
        )

        # 5. Error Message Enhancer - Better error messages
        self.error_enhancer = ErrorMessageEnhancer(verbose=self.verbose)

        # 6. Mandatory Confirmation - Human-in-the-loop for Slack/Notion
        self.message_confirmer = MandatoryConfirmationEnforcer(verbose=self.verbose)

        # Load user preferences from file if exists
        self.prefs_file = Path(f"data/preferences/{user_id}.json")
        self.prefs_file.parent.mkdir(parents=True, exist_ok=True)
        if self.prefs_file.exists():
            try:
                self.user_prefs.load_from_file(str(self.prefs_file))
                if self.verbose:
                    print(f"{C.CYAN}📊 Loaded user preferences from {self.prefs_file}{C.ENDC}")
            except Exception as e:
                logger.warning(f"Failed to load preferences: {e}")

        # Register undo handlers
        self._register_undo_handlers()

        # ===================================================================

        if self.verbose:
            print(f"{C.CYAN}🧠 Intelligence enabled: Session {self.session_id[:8]}...{C.ENDC}")
            print(f"{C.CYAN}📊 Advanced Intelligence: Intent, Entity, Task, Confidence, Context{C.ENDC}")
            print(f"{C.CYAN}📝 Logging to: {self.session_logger.get_log_path()}{C.ENDC}")
            print(f"{C.CYAN}🔄 Retry, 📊 Analytics, 🧠 Preferences, ↩️  Undo - All enabled{C.ENDC}")

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

    # =========================================================================
    # UNDO HANDLER REGISTRATION
    # =========================================================================

    def _register_undo_handlers(self):
        """Register undo handlers for all agent types"""

        # Jira undo handlers
        async def undo_jira_delete(undo_params: Dict) -> str:
            """Undo Jira issue deletion (note: actual deletion is usually permanent)"""
            issue_key = undo_params.get('issue_key')
            return f"⚠️ Cannot restore deleted issue {issue_key} - deletion is permanent. Consider recreating it manually."

        async def undo_jira_transition(undo_params: Dict) -> str:
            """Undo Jira issue transition by reverting to previous status"""
            issue_key = undo_params.get('issue_key')
            previous_status = undo_params.get('previous_status')
            agent = self.sub_agents.get('jira')

            if agent:
                instruction = f"Transition {issue_key} back to '{previous_status}' status"
                result = await agent.execute(instruction)
                return f"✓ Reverted {issue_key} to {previous_status}: {result}"
            return f"❌ Jira agent not available"

        # Slack undo handlers
        async def undo_slack_delete_message(undo_params: Dict) -> str:
            """Undo Slack message deletion (note: usually permanent)"""
            channel = undo_params.get('channel')
            message_text = undo_params.get('message_text', '')
            return f"⚠️ Cannot restore deleted message in {channel} - deletion is permanent. Original message: {message_text[:100]}"

        # GitHub undo handlers
        async def undo_github_close_pr(undo_params: Dict) -> str:
            """Undo GitHub PR closure by reopening"""
            pr_number = undo_params.get('pr_number')
            repo = undo_params.get('repo')
            agent = self.sub_agents.get('github')

            if agent:
                instruction = f"Reopen pull request #{pr_number} in {repo}"
                result = await agent.execute(instruction)
                return f"✓ Reopened PR #{pr_number}: {result}"
            return f"❌ GitHub agent not available"

        async def undo_github_close_issue(undo_params: Dict) -> str:
            """Undo GitHub issue closure by reopening"""
            issue_number = undo_params.get('issue_number')
            repo = undo_params.get('repo')
            agent = self.sub_agents.get('github')

            if agent:
                instruction = f"Reopen issue #{issue_number} in {repo}"
                result = await agent.execute(instruction)
                return f"✓ Reopened issue #{issue_number}: {result}"
            return f"❌ GitHub agent not available"

        # Notion undo handlers
        async def undo_notion_delete_page(undo_params: Dict) -> str:
            """Undo Notion page deletion (note: usually permanent)"""
            page_title = undo_params.get('page_title', 'Unknown')
            return f"⚠️ Cannot restore deleted Notion page '{page_title}' - deletion is permanent."

        # Register all handlers
        self.undo_manager.register_undo_handler(UndoableOperationType.JIRA_DELETE_ISSUE, undo_jira_delete)
        self.undo_manager.register_undo_handler(UndoableOperationType.JIRA_TRANSITION, undo_jira_transition)
        self.undo_manager.register_undo_handler(UndoableOperationType.SLACK_DELETE_MESSAGE, undo_slack_delete_message)
        self.undo_manager.register_undo_handler(UndoableOperationType.GITHUB_CLOSE_PR, undo_github_close_pr)
        self.undo_manager.register_undo_handler(UndoableOperationType.GITHUB_CLOSE_ISSUE, undo_github_close_issue)
        self.undo_manager.register_undo_handler(UndoableOperationType.NOTION_DELETE_PAGE, undo_notion_delete_page)

        if self.verbose:
            print(f"{C.CYAN}↩️  Registered {len(self.undo_manager.undo_handlers)} undo handlers{C.ENDC}")

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    async def _spinner(self, task: asyncio.Task, message: str):
        """Display an animated spinner while a task is running."""
        # If in verbose mode, just await the task and let the logs print
        if self.verbose:
            await task  # This will still raise exceptions if the task fails
            return

        # Use new UI progress indicator
        progress = self.ui.start_progress(message)

        try:
            await task  # This will raise exceptions if the task fails
        finally:
            progress.stop()
    
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
                # Try with full intelligence support + LLM abstraction + logging
                agent_instance = agent_class(
                    verbose=self.verbose,
                    shared_context=self.shared_context,
                    knowledge_base=self.knowledge_base,
                    llm=self.llm,
                    session_logger=self.session_logger
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
                except Exception as cleanup_err:
                    logger.error(f"Failed to cleanup {agent_name}: {cleanup_err}", exc_info=True)
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

            # Check if result is an exception (including BaseException subclasses like CancelledError)
            if isinstance(result, BaseException):
                failed += 1
                print(f"{C.RED}✗ Exception during loading: {result}{C.ENDC}")
                if self.verbose:
                    print(f"{C.RED}    Type: {type(result).__name__}{C.ENDC}")
                continue

            # Validate result is a tuple before unpacking
            if not isinstance(result, tuple) or len(result) != 4:
                failed += 1
                print(f"{C.RED}✗ Invalid result from agent loading: {result}{C.ENDC}")
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
        """
        Execute a task using a specialized sub-agent with:
        - Retry management (smart exponential backoff)
        - Analytics tracking (performance metrics)
        - Health checking
        """
        if agent_name not in self.sub_agents:
            return f"Error: Agent '{agent_name}' not found"

        # Check agent health before calling
        if agent_name in self.agent_health:
            health = self.agent_health[agent_name]
            if health['status'] == 'unavailable':
                error_msg = f"⚠️ {agent_name} agent is currently unavailable: {health.get('error_message', 'Unknown error')}"
                print(f"{C.YELLOW}{error_msg}{C.ENDC}")
                return error_msg

        # ===== MANDATORY CONFIRMATION FOR SLACK & NOTION =====
        # Check if this operation requires human approval
        if self.message_confirmer.requires_confirmation(agent_name, instruction):
            if self.verbose:
                print(f"\n{C.CYAN}📋 This operation requires human confirmation{C.ENDC}")

            should_execute, modified_instruction = self.message_confirmer.confirm_before_execution(
                agent_name=agent_name,
                instruction=instruction
            )

            if not should_execute:
                # User rejected or wants AI modification
                if modified_instruction and modified_instruction.startswith("[AI_MODIFICATION_REQUESTED]"):
                    # User wants AI to modify the message/content
                    modification_request = modified_instruction.replace("[AI_MODIFICATION_REQUESTED] ", "")

                    # Return a special response that tells the orchestrator to regenerate
                    # The main intelligence layer will see this and ask Claude to revise
                    return f"🔄 User requested modification: {modification_request}\n\nPlease revise the {agent_name} operation based on this feedback and try again."
                else:
                    # User rejected - don't execute
                    return "❌ Operation cancelled by user"
            else:
                # User approved (possibly with manual edits)
                # Use the modified instruction if provided
                if modified_instruction:
                    instruction = modified_instruction
                    print(f"{C.GREEN}✅ Approved! Executing...{C.ENDC}")

        # Create operation key for retry/analytics tracking
        operation_key = f"{agent_name}_{hashlib.md5(instruction[:100].encode()).hexdigest()[:8]}"

        # Define the operation to execute
        async def execute_operation():
            return await self._execute_agent_direct(agent_name, instruction, context)

        # Define progress callback for retry manager
        def progress_callback(message: str, attempt: int, max_attempts: int):
            if attempt > 1:
                print(f"{C.YELLOW}🔄 {message}{C.ENDC}")

        # Execute with retry management
        try:
            result = await self.retry_manager.execute_with_retry(
                operation_key=operation_key,
                agent_name=agent_name,
                instruction=instruction,
                operation=execute_operation,
                progress_callback=progress_callback
            )
            return result

        except Exception as e:
            # Final failure after all retries
            error_msg = f"Error executing {agent_name} agent: {str(e)}"
            print(f"{C.RED}✗ {error_msg}{C.ENDC}")
            if self.verbose:
                traceback.print_exc()
            return error_msg

    async def _execute_agent_direct(self, agent_name: str, instruction: str, context: Any = None) -> str:
        """Direct agent execution with analytics tracking (called by retry manager)"""

        if self.verbose:
            print(f"\n{C.MAGENTA}{'─'*60}{C.ENDC}")
            print(f"{C.MAGENTA}🤖 Delegating to {C.BOLD}{agent_name}{C.ENDC}{C.MAGENTA} agent{C.ENDC}")
            print(f"{C.MAGENTA}{'─'*60}{C.ENDC}")
            print(f"{C.CYAN}Instruction: {instruction}{C.ENDC}")

        # Prepare context
        context_str = ""
        if context:
            if isinstance(context, (dict, list)):
                context_str = json.dumps(context, indent=2)
            else:
                context_str = str(context)

            if self.verbose:
                print(f"{C.CYAN}Context: {context_str[:200]}...{C.ENDC}")

        # Start timing for analytics
        import time
        start_time = time.time()

        try:
            agent = self.sub_agents[agent_name]

            # Build full instruction with context
            full_instruction = instruction
            if context_str:
                full_instruction = f"Context from previous steps:\n{context_str}\n\nTask: {instruction}"

            # Log message to agent
            self.session_logger.log_message_to_agent(agent_name, full_instruction)

            # Execute the agent
            result = await agent.execute(full_instruction)

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Determine success
            success = not result.startswith("⚠️") and not result.startswith("Error")
            error = result if not success else None

            # Record analytics
            self.analytics.record_agent_call(
                agent_name=agent_name,
                success=success,
                latency_ms=latency_ms,
                error_message=error
            )

            # Log response from agent
            self.session_logger.log_message_from_agent(agent_name, result, success, error)

            # Update health status on success
            if success and agent_name in self.agent_health:
                self.agent_health[agent_name]['status'] = 'healthy'
                self.agent_health[agent_name]['last_success'] = asyncio.get_event_loop().time()
                self.agent_health[agent_name]['error_count'] = 0

            if self.verbose:
                status = "✓" if success else "✗"
                print(f"{C.GREEN if success else C.RED}{status} {agent_name} completed ({latency_ms:.0f}ms){C.ENDC}")
                print(f"{C.MAGENTA}{'─'*60}{C.ENDC}\n")

            # If failed, raise exception for retry manager
            if not success:
                # Enhance error message before raising
                enhanced = self.error_enhancer.enhance_error(
                    agent_name=agent_name,
                    error=RuntimeError(result),
                    instruction=instruction,
                    context=context
                )
                enhanced_msg = enhanced.format()

                if self.verbose:
                    print(f"{C.YELLOW}{enhanced_msg}{C.ENDC}")

                raise RuntimeError(enhanced_msg)

            return result

        except Exception as e:
            # Calculate latency even on failure
            latency_ms = (time.time() - start_time) * 1000

            # Enhance error message
            enhanced = self.error_enhancer.enhance_error(
                agent_name=agent_name,
                error=e,
                instruction=instruction,
                context=context
            )
            enhanced_msg = enhanced.format()

            # Record analytics
            self.analytics.record_agent_call(
                agent_name=agent_name,
                success=False,
                latency_ms=latency_ms,
                error_message=enhanced_msg
            )

            # Update health status on failure
            if agent_name in self.agent_health:
                self.agent_health[agent_name]['error_count'] = self.agent_health[agent_name].get('error_count', 0) + 1
                self.agent_health[agent_name]['last_failure'] = asyncio.get_event_loop().time()
                self.agent_health[agent_name]['error_message'] = enhanced_msg

                # Mark as degraded if error count exceeds threshold
                if self.agent_health[agent_name]['error_count'] >= 3:
                    self.agent_health[agent_name]['status'] = 'degraded'
                    print(f"{C.YELLOW}⚠️ {agent_name} agent marked as degraded after 3 failures{C.ENDC}")

            # Re-raise enhanced error for retry manager
            raise RuntimeError(enhanced_msg)

    def _process_with_intelligence(self, user_message: str) -> Dict:
        """
        Process user message with advanced intelligence

        Returns intelligence analysis including intents, entities, confidence, etc.
        """
        # 1. Classify intent
        intents = self.intent_classifier.classify(user_message)

        # 2. Extract entities
        context_dict = self.context_manager.get_relevant_context(user_message)
        entities = self.entity_extractor.extract(user_message, context=context_dict)

        # 3. Score confidence
        confidence = self.confidence_scorer.score_overall(
            message=user_message,
            intents=intents,
            entities=entities
        )

        # 4. Update conversation context
        self.context_manager.add_turn(
            role='user',
            message=user_message,
            intents=intents,
            entities=entities
        )

        # 5. Resolve references if needed (but don't modify message for now)
        # Note: Reference resolution is tracked in context but we let the LLM
        # handle the actual interpretation with context provided
        resolved_message = user_message

        # Log resolved references for debugging
        if self.verbose:
            for word in user_message.split():
                resolution = self.context_manager.resolve_reference(word)
                if resolution:
                    entity_id, entity = resolution
                    print(f"[INTELLIGENCE] Can resolve '{word}' → {entity.value}")

        # 6. Build intelligence summary
        intelligence = {
            'intents': intents,
            'entities': entities,
            'confidence': confidence,
            'resolved_message': resolved_message,
            'context': context_dict,
            'primary_intent': self.intent_classifier.get_primary_intent(intents),
            'action_recommendation': self.confidence_scorer.get_action_recommendation(confidence)
        }

        # 7. Log intelligence insights
        if self.verbose:
            print(f"\n{C.CYAN}🧠 Intelligence Analysis:{C.ENDC}")
            print(f"  Intents: {[str(i) for i in intents[:3]]}")
            print(f"  Entities: {len(entities)} found")
            print(f"  Confidence: {confidence}")
            print(f"  Recommendation: {intelligence['action_recommendation'][0]}")

        return intelligence

    async def process_message(self, user_message: str) -> str:
        """Process a user message with orchestration"""

        # ===================================================================
        # USER PREFERENCES & ANALYTICS TRACKING
        # ===================================================================

        # Record user message for analytics
        self.analytics.record_user_message()

        # Record interaction time for working hours learning
        self.user_prefs.record_interaction_time()

        # Learn communication style from user message
        self.user_prefs.record_interaction_style(user_message)

        # ===================================================================

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

        # Process with intelligence
        intelligence = self._process_with_intelligence(user_message)

        # Use resolved message if references were resolved
        message_to_send = intelligence.get('resolved_message', user_message)

        # Check confidence and handle accordingly
        action, explanation = intelligence['action_recommendation']

        if action == 'clarify' and not self.verbose:
            # Low confidence - ask clarifying questions
            clarifications = self.confidence_scorer.suggest_clarifications(
                intelligence['confidence'],
                intelligence['intents']
            )
            if clarifications:
                self.ui.print_response("\n".join(clarifications[:2]))
                return "I need more information to proceed. " + clarifications[0]

        # Log intelligence insights
        if self.verbose:
            print(f"{C.CYAN}📊 Using intelligence: {explanation}{C.ENDC}")

        # Create and run the initial send task with a spinner
        send_task = asyncio.create_task(self.chat.send_message(message_to_send))
        await self._spinner(send_task, "Thinking")
        llm_response = send_task.result()

        # Get the raw response object for compatibility (with safety checks)
        response = self._safe_get_response_object(llm_response)

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

            # SECURITY: Validate instruction before processing
            is_valid, validation_error = InputValidator.validate_instruction(instruction)
            if not is_valid:
                logger.error(f"Invalid instruction rejected: {validation_error}")
                function_call_result = {
                    'agent': agent_name,
                    'status': 'error',
                    'error': validation_error
                }
                continue

            # NOTE: Old confirmation systems removed - now using MandatoryConfirmationEnforcer
            # The new system confirms operations inline in call_sub_agent() for Slack/Notion

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

                # Feature #21: Track successful operation in duplicate detector
                self.duplicate_detector.track_operation(agent_name, instruction, "", success=True)

                # Feature #8: Mark success if this was a retry
                if retry_context:
                    self._mark_retry_success(operation_key, solution_used="retry with same parameters")

            except asyncio.TimeoutError:
                error_msg = f"Agent timed out after 120 seconds. Operation may have completed but response was not received."

                # Classify timeout as transient error
                error_classification = ErrorClassifier.classify(error_msg, agent_name)
                self._track_retry_attempt(operation_key, agent_name, instruction, error=error_msg)

                # Get retry context
                retry_ctx = self._get_retry_context(operation_key)
                attempt_num = retry_ctx['attempt_number'] if retry_ctx else 1

                # Format error with intelligent messaging
                result = format_error_for_user(
                    error_classification,
                    agent_name,
                    instruction,
                    attempt_num,
                    self.max_retry_attempts
                )

                # Add max retries message if exceeded
                if retry_ctx and attempt_num >= self.max_retry_attempts:
                    result += f"\n\n**Note**: This error appears to be transient (temporary network issue). If it persists:\n"
                    result += f"  • Check your internet connection\n"
                    result += f"  • Try again in a few moments\n"
                    result += f"  • Break the operation into smaller steps"

                print(f"{C.YELLOW}⚠ {agent_name} agent operation timed out{C.ENDC}")

            except Exception as e:
                error_str = str(e)

                # Track in duplicate detector
                self.duplicate_detector.track_operation(agent_name, instruction, error_str, success=False)

                # Classify the error intelligently
                error_classification = ErrorClassifier.classify(error_str, agent_name)

                # Track the error
                self._track_retry_attempt(operation_key, agent_name, instruction, error=error_str)

                # Get retry context
                retry_ctx = self._get_retry_context(operation_key)
                attempt_num = retry_ctx['attempt_number'] if retry_ctx else 1

                # Check for duplicate/stuck operations
                is_duplicate, dup_explanation = self.duplicate_detector.detect_duplicate_failure(
                    agent_name, instruction, error_str
                )

                # Check for inconsistent responses
                is_inconsistent, dup_pattern = self.duplicate_detector.detect_inconsistent_responses(
                    agent_name, instruction
                )

                # Format error message with intelligent suggestions
                result = format_error_for_user(
                    error_classification,
                    agent_name,
                    instruction,
                    attempt_num,
                    self.max_retry_attempts
                )

                # Add duplicate/stuck operation warning
                if is_duplicate and dup_explanation:
                    result += f"\n\n⚠️ **DUPLICATE OPERATION DETECTED**\n{dup_explanation}"
                    result += f"\n\nThis operation appears stuck. It will NOT be retried further."
                    # Force stop retrying for duplicate operations
                    error_classification.is_retryable = False

                # Add inconsistent response warning
                if is_inconsistent and dup_pattern:
                    result += f"\n\n⚠️ **INCONSISTENT RESPONSES DETECTED**\n"
                    result += f"Response pattern: {' → '.join(dup_pattern[-5:])}\n"
                    result += f"The agent is giving conflicting results. Please verify manually."

                # Add context-specific guidance
                if not error_classification.is_retryable and attempt_num == 1:
                    # Non-retryable error on first attempt - explain clearly
                    result += f"\n\n⚠️ **This operation will not be retried** because it's a {error_classification.category.value} error."

                if error_classification.is_retryable and retry_ctx:
                    # Show retry info for retryable errors
                    if attempt_num >= self.max_retry_attempts:
                        result += f"\n\n**Note**: Maximum retry attempts ({self.max_retry_attempts}) reached. This appears to be a persistent issue."
                    else:
                        result += f"\n\n**Next step**: The system will automatically retry this operation."

                # Log the classification for debugging
                if self.verbose:
                    print(f"{C.CYAN}[ERROR CLASSIFICATION] Category: {error_classification.category.value}, Retryable: {error_classification.is_retryable}{C.ENDC}")
                    if is_duplicate:
                        print(f"{C.YELLOW}[DUPLICATE DETECTED] {dup_explanation}{C.ENDC}")
                    if is_inconsistent:
                        print(f"{C.YELLOW}[INCONSISTENT RESPONSES] Pattern: {dup_pattern}{C.ENDC}")

                print(f"{C.RED}✗ {agent_name} agent failed: {e}{C.ENDC}")
            
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

    def _should_retry_operation(self, error_str: str, operation_key: str) -> bool:
        """
        Determine if we should retry an operation based on error classification.

        Args:
            error_str: The error message
            operation_key: The operation key for tracking

        Returns:
            bool: True if we should retry, False if we should give up
        """
        # Classify the error
        error_classification = ErrorClassifier.classify(error_str)

        # Non-retryable errors - never retry
        if not error_classification.is_retryable:
            if self.verbose:
                print(f"{C.CYAN}[RETRY DECISION] Non-retryable error ({error_classification.category.value}) - will NOT retry{C.ENDC}")
            return False

        # Check retry attempt count
        retry_context = self._get_retry_context(operation_key)
        if retry_context:
            attempt_num = retry_context['attempt_number']
            if attempt_num >= self.max_retry_attempts:
                if self.verbose:
                    print(f"{C.CYAN}[RETRY DECISION] Max attempts ({self.max_retry_attempts}) reached - will NOT retry{C.ENDC}")
                return False

        # Retryable error and within limits - retry
        if self.verbose:
            category = error_classification.category.value
            print(f"{C.CYAN}[RETRY DECISION] Retryable error ({category}) - will retry{C.ENDC}")

        return True

    def _deep_convert_proto_args(self, value: Any) -> Any:
        """Recursively converts Protobuf composite types into standard Python dicts/lists"""
        type_str = str(type(value))
        if "MapComposite" in type_str:
            return {k: self._deep_convert_proto_args(v) for k, v in value.items()}
        elif "RepeatedComposite" in type_str:
            return [self._deep_convert_proto_args(item) for item in value]
        else:
            return value

    def _create_operation_signature(self, agent_name: str, instruction: str, risk_type: str) -> str:
        """
        Create a unique signature for an operation to track confirmations.

        This prevents asking for confirmation multiple times for the same operation.
        The signature is based on agent, action type, and key entities (not exact instruction).

        Args:
            agent_name: Name of the agent
            instruction: The instruction text
            risk_type: Type of risk (destructive, bulk, public, ambiguous)

        Returns:
            A signature string for tracking this operation
        """
        import hashlib

        # Normalize instruction to extract intent
        instruction_lower = instruction.lower().strip()

        # Extract action verb (first meaningful word)
        action_words = ['create', 'delete', 'update', 'remove', 'add', 'send', 'notify',
                       'close', 'archive', 'change', 'modify', 'edit']
        action = 'unknown'
        for word in action_words:
            if word in instruction_lower:
                action = word
                break

        # Create signature from: agent + action + risk_type
        # This allows same action to be confirmed once, but different actions to be asked separately
        sig_base = f"{agent_name}:{action}:{risk_type}"

        # For more specific tracking, include a hash of key instruction parts
        # This prevents "delete KAN-20" and "delete KAN-30" from being treated as same
        # But allows repeated calls with exact same instruction to skip confirmation
        instruction_hash = hashlib.md5(instruction_lower.encode()).hexdigest()[:8]

        signature = f"{sig_base}:{instruction_hash}"

        return signature

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
        import re

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

        # Destructive operations - use word boundaries to avoid false matches
        # (e.g., "disclose" should not match "close")
        destructive_keywords = ['delete', 'remove', 'close', 'archive', 'drop', 'destroy']

        # Check for whole words only using word boundaries
        destructive_pattern = r'\b(' + '|'.join(destructive_keywords) + r')\b'
        if re.search(destructive_pattern, instruction_lower):
            if self.confirmation_prefs['always_confirm_deletes']:
                return {
                    'needs_confirmation': True,
                    'risk_type': 'destructive',
                    'reason': 'This operation will permanently delete or close items',
                    'suggestions': []
                }

        # Bulk operations - but be smarter about detection
        # Only consider it bulk if it's actually acting on multiple items
        bulk_action_patterns = [
            'delete all', 'remove all', 'update all', 'change all',
            'create multiple', 'add multiple', 'send to all', 'notify everyone'
        ]

        is_bulk = any(pattern in instruction_lower for pattern in bulk_action_patterns)

        # Additional check: if instruction contains numbers > 5, might be bulk
        numbers = re.findall(r'\b(\d+)\b', instruction_lower)
        if numbers and any(int(n) > 5 for n in numbers if n.isdigit()):
            # Check if it's about creating/updating that many items
            if any(word in instruction_lower for word in ['create', 'delete', 'update', 'add', 'remove']):
                is_bulk = True

        # Skip confirmation for bulk READ operations
        if is_bulk:
            bulk_read_keywords = ['review', 'analyze', 'check', 'scan', 'inspect', 'examine', 'list', 'get', 'show']
            is_bulk_read = any(word in instruction_lower for word in bulk_read_keywords)

            if is_bulk_read:
                return {'needs_confirmation': False}

            # Bulk write operation - needs confirmation
            if self.confirmation_prefs['always_confirm_bulk']:
                return {
                    'needs_confirmation': True,
                    'risk_type': 'bulk',
                    'reason': f'This will affect multiple items',
                    'suggestions': ['Consider processing items one at a time for safety']
                }

        # Public/broadcast operations - use word boundaries for exact matches
        # Note: @channel and @here are exact matches, not word patterns
        public_indicators = ['everyone', 'all users', '@channel', '@here', 'company-wide']

        # Check each indicator (special handling for @ symbols)
        is_public = False
        for indicator in public_indicators:
            if indicator.startswith('@'):
                # For @mentions, do exact substring match
                if indicator in instruction_lower:
                    is_public = True
                    break
            else:
                # For phrases with spaces, escape words individually and join with \s+
                if ' ' in indicator:
                    words = indicator.split()
                    escaped_words = [re.escape(word) for word in words]
                    pattern = r'\b' + r'\s+'.join(escaped_words) + r'\b'
                else:
                    # Single word, just escape and add word boundaries
                    pattern = r'\b' + re.escape(indicator) + r'\b'

                if re.search(pattern, instruction_lower):
                    is_public = True
                    break

        if is_public:
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

        # Slack message confirmation (if enabled)
        if agent_name == 'slack' and self.confirmation_prefs.get('confirm_slack_messages', False):
            # Check if this is a send/notify operation
            slack_action_keywords = ['send', 'post', 'message', 'notify', 'broadcast', 'share']
            if any(word in instruction_lower for word in slack_action_keywords):
                return {
                    'needs_confirmation': True,
                    'risk_type': 'slack_message',
                    'reason': 'Review and edit message before sending',
                    'suggestions': []
                }

        # Jira operation confirmation (if enabled)
        if agent_name == 'jira' and self.confirmation_prefs.get('confirm_jira_operations', False):
            # Check if this is a create/update/delete operation
            jira_action_keywords = ['create', 'update', 'modify', 'delete', 'transition', 'assign', 'add', 'edit']
            if any(word in instruction_lower for word in jira_action_keywords):
                return {
                    'needs_confirmation': True,
                    'risk_type': 'jira_operation',
                    'reason': 'Review and edit Jira operation before executing',
                    'suggestions': []
                }

        return {'needs_confirmation': False}

    def _ask_confirmation(self, risk_info: Dict[str, Any], instruction: str) -> bool:
        """
        Ask user for confirmation before proceeding (Feature #20)

        Returns:
            bool: True if user confirmed, False if cancelled
        """
        # Shorten instruction for display - don't show full code
        display_instruction = instruction[:200]
        if len(instruction) > 200:
            display_instruction += "... (truncated)"

        print(f"\n{C.YELLOW}⚠️  CONFIRMATION REQUIRED{C.ENDC}")
        print(f"{C.YELLOW}{'─' * 60}{C.ENDC}")
        print(f"{C.CYAN}Operation: {display_instruction}{C.ENDC}")
        print(f"{C.YELLOW}Risk Type: {risk_info['risk_type'].upper()}{C.ENDC}")
        print(f"{C.YELLOW}Reason: {risk_info['reason']}{C.ENDC}")

        if risk_info.get('suggestions'):
            print(f"\n{C.CYAN}💡 Options:{C.ENDC}")
            for suggestion in risk_info['suggestions'][:5]:  # Max 5 suggestions
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
        """Cleanup all sub-agents and close session logger"""
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

        # ===================================================================
        # SAVE ANALYTICS AND PREFERENCES
        # ===================================================================

        # End analytics session
        if hasattr(self, 'analytics'):
            try:
                self.analytics.end_session()

                # Save analytics
                analytics_dir = Path("logs/analytics")
                analytics_dir.mkdir(parents=True, exist_ok=True)
                analytics_file = analytics_dir / f"{self.session_id}.json"
                self.analytics.save_to_file(str(analytics_file))

                if self.verbose:
                    print(f"{C.GREEN}  ✓ Analytics saved: {analytics_file}{C.ENDC}")
                    print(f"{C.CYAN}    {self.analytics.generate_summary_report()}{C.ENDC}")
            except Exception as e:
                logger.warning(f"Failed to save analytics: {e}")

        # Save user preferences
        if hasattr(self, 'user_prefs') and hasattr(self, 'prefs_file'):
            try:
                self.user_prefs.save_to_file(str(self.prefs_file))
                if self.verbose:
                    print(f"{C.GREEN}  ✓ Preferences saved: {self.prefs_file}{C.ENDC}")
            except Exception as e:
                logger.warning(f"Failed to save preferences: {e}")

        # Display retry statistics
        if hasattr(self, 'retry_manager') and self.verbose:
            stats = self.retry_manager.get_statistics()
            if stats['total_operations'] > 0:
                print(f"{C.CYAN}  📊 Retry stats: {stats['total_operations']} ops, "
                      f"{stats['avg_retries_per_operation']:.1f} avg retries{C.ENDC}")

        # Display undo statistics
        if hasattr(self, 'undo_manager') and self.verbose:
            undo_stats = self.undo_manager.get_statistics()
            if undo_stats['total_operations'] > 0:
                print(f"{C.CYAN}  ↩️  Undo: {undo_stats['available_for_undo']} operations available{C.ENDC}")

        # ===================================================================

        # Close session logger and generate summary
        if hasattr(self, 'session_logger'):
            self.session_logger.close()
            if self.verbose:
                print(f"{C.GREEN}  ✓ Session log saved: {self.session_logger.get_log_path()}{C.ENDC}")
    
    async def run_interactive(self):
        """Run interactive chat session"""
        # Use new UI for non-verbose mode
        if not self.verbose:
            self.ui.print_header(self.session_id)
        else:
            # Old verbose header
            print(f"\n{C.YELLOW}{'='*60}{C.ENDC}")
            print(f"{C.BOLD}{C.CYAN}🎭 Multi-Agent Orchestration System{C.ENDC}")
            print(f"{C.YELLOW}Mode: Verbose{C.ENDC}")
            print(f"{C.YELLOW}{'='*60}{C.ENDC}\n")

        try:
            while True:
                # Print prompt based on mode
                if not self.verbose:
                    self.ui.print_prompt()
                    user_input = input().strip()
                else:
                    user_input = input(f"{C.BOLD}{C.BLUE}You: {C.ENDC}").strip()

                if user_input.lower() in ['exit', 'quit', 'bye']:
                    if not self.verbose:
                        self.ui.print_goodbye()
                    else:
                        print(f"\n{C.GREEN}Goodbye! 👋{C.ENDC}")
                    break

                if not user_input:
                    continue

                if user_input.lower() == 'help':
                    self._show_help()
                    continue

                try:
                    # Show thinking indicator in non-verbose mode
                    if not self.verbose:
                        self.ui.print_thinking()

                    # The spinners will play here
                    response = await self.process_message(user_input)

                    # Print response based on mode
                    if not self.verbose:
                        self.ui.print_response(response)
                    else:
                        # Old verbose response
                        print(f"\n{C.BOLD}{C.GREEN}🎭 Orchestrator:{C.ENDC}\n{response}\n")

                except Exception as e:
                    # Errors are always printed
                    if not self.verbose:
                        self.ui.print_error(str(e))
                    else:
                        print(f"\n{C.RED}✗ An error occurred: {str(e)}{C.ENDC}")
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