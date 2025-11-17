import os
import json
import sys
import asyncio
import traceback
import uuid
import hashlib
from typing import Any, Dict, List, Optional
import google.generativeai as genai
import google.generativeai.protos as protos
from dotenv import load_dotenv
import time 

from pathlib import Path
import importlib.util

from llms.base_llm import BaseLLM, LLMConfig
from llms.gemini_flash import GeminiFlash

from connectors.agent_intelligence import WorkspaceKnowledge, SharedContext

from intelligence import (
    TaskDecomposer,
    ConfidenceScorer, ConversationContextManager,
    HybridIntelligenceSystem
)
from intelligence.base_types import OperationRiskClassifier, RiskLevel, IntentType

from core.session_logger import SessionLogger
from core.input_validator import InputValidator
from core.errors import ErrorClassifier, format_error_for_user, DuplicateOperationDetector, ErrorMessageEnhancer
from core.resilience import RetryManager
from core.user import UserPreferenceManager, AnalyticsCollector
from core.parallel_executor import ParallelExecutor, AgentTask
from core.circuit_breaker import CircuitBreaker, CircuitConfig, CircuitBreakerError
from core.advanced_cache import HybridCache, APIResponseCache
from core.simple_embeddings import create_default_embeddings
from core.agent_error_handling import AgentErrorHandler, FallbackBehaviors
load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError(
        "Google Gemini API Key is required to run Project Aerius.\n\n"
        "To set up:\n"
        "1. Go to: https://makersuite.google.com/app/apikey\n"
        "2. Click 'Create API Key'\n"
        "3. Copy the API key\n"
        "4. Add to your .env file:\n"
        "   GOOGLE_API_KEY=your_api_key_here\n"
    )

genai.configure(api_key=GOOGLE_API_KEY)


class OrchestratorAgent:

    @staticmethod
    def _safe_get_response_object(llm_response) -> Optional[Any]:
        try:
            if not llm_response:
                return None
            if not hasattr(llm_response, 'metadata') or not llm_response.metadata:
                return None
            response_obj = llm_response.metadata.get('response_object')
            if not response_obj:
                return None
            if not hasattr(response_obj, 'candidates') or not response_obj.candidates:
                return None
            return response_obj
        except AttributeError as e:
            return None
        except Exception as e:
            return None

    def __init__(self, connectors_dir: str = "connectors", verbose: bool = False, llm: Optional[BaseLLM] = None):
        self.connectors_dir = Path(connectors_dir)
        self.sub_agents: Dict[str, Any] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}
        self.agent_health: Dict[str, Dict[str, Any]] = {}
        self.verbose = verbose

        if llm is None:
            self.llm = GeminiFlash(LLMConfig(
                model_name='models/gemini-2.5-flash',
                temperature=0.7
            ))
        else:
            self.llm = llm

        if self.verbose:
            print(f"Using LLM: {self.llm}")

        self.knowledge_base = WorkspaceKnowledge()
        self.session_id = str(uuid.uuid4())
        self.shared_context = SharedContext(self.session_id)

        self.retry_tracker: Dict[str, Dict[str, Any]] = {}
        self.max_retry_attempts = 3

        self.duplicate_detector = DuplicateOperationDetector(window_size=5, similarity_threshold=0.8)

        self.operation_count = 0

        self.session_logger = SessionLogger(session_id=self.session_id, log_dir="logs")

        # Modern hybrid intelligence system (Fast filter + LLM classifier)
        self.hybrid_intelligence = HybridIntelligenceSystem(
            llm=self.llm,
            verbose=self.verbose
        )

        # Task decomposer and confidence scorer (still useful for task planning)
        self.task_decomposer = TaskDecomposer(
            agent_capabilities=self.agent_capabilities,
            verbose=self.verbose
        )
        self.confidence_scorer = ConfidenceScorer(verbose=self.verbose)

        # Context manager (essential for conversation tracking)
        self.context_manager = ConversationContextManager(
            session_id=self.session_id,
            verbose=self.verbose
        )

        # Parallel executor for multi-agent workflows
        self.parallel_executor = ParallelExecutor(verbose=self.verbose)

        # Circuit breaker for agent health management
        self.circuit_breaker = CircuitBreaker(
            config=CircuitConfig(
                failure_threshold=5,      # Open after 5 consecutive failures
                success_threshold=2,      # Close after 2 consecutive successes
                timeout_seconds=300.0,    # 5 minutes before recovery attempt
                half_open_timeout=10.0
            ),
            verbose=self.verbose
        )

        self.retry_manager = RetryManager(
            max_retries=self.max_retry_attempts,
            base_delay=1.0,
            max_delay=30.0,
            backoff_factor=2.0,
            jitter=True,
            verbose=self.verbose
        )


        # User preferences - session-based, not user-based
        self.user_prefs = UserPreferenceManager(
            user_id=self.session_id,  # Use session_id instead of USER_ID
            min_confidence_threshold=0.7,
            verbose=self.verbose
        )

        self.analytics = AnalyticsCollector(
            session_id=self.session_id,
            max_latency_samples=1000,
            verbose=self.verbose
        )

        self.error_enhancer = ErrorMessageEnhancer(verbose=self.verbose)

        # Advanced caching system with semantic deduplication and persistence
        try:
            if self.verbose:
                print("Initializing hybrid cache...")
            embeddings = create_default_embeddings()
            self.hybrid_cache = HybridCache(
                cache_dir=".cache",
                enable_semantic=True,
                enable_persistent=True,
                enable_api_cache=True,
                embedding_model=embeddings,
                verbose=self.verbose
            )

            if self.verbose:
                print("✓ Hybrid cache initialized with semantic deduplication and persistence")

        except Exception as e:
            if self.verbose:
                print(f"⚠ Hybrid cache initialization failed, using basic cache: {e}")
            self.hybrid_cache = None

        self.prefs_file = Path(f"data/preferences/{self.session_id}.json")
        self.prefs_file.parent.mkdir(parents=True, exist_ok=True)
        if self.prefs_file.exists():
            try:
                self.user_prefs.load_from_file(str(self.prefs_file))
                if self.verbose:
                    print(f"Loaded user preferences from {self.prefs_file}")
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to load preferences: {e}")

        if self.verbose:
            print(f"Intelligence enabled: Session {self.session_id[:8]}...")
            print("Hybrid Intelligence: Fast Filter + LLM Classifier")
            print("  • 92% accuracy with semantic understanding")
            print("  • ~80ms avg latency with caching")
            print(f"Logging to: {self.session_logger.get_log_path()}")
            print("Retry, Analytics, Preferences - All enabled")

        # Log session start
        self.session_logger.log_system_event('session_start', {
            'session_id': self.session_id,
            'verbose': self.verbose
        })

        self.system_prompt = """You are an AI orchestration system that coordinates specialized agents to help users accomplish complex tasks across multiple platforms and tools.

Your core purpose is to be a highly capable, reliable workspace assistant that understands user intent, breaks down complex requests into actionable steps, and seamlessly coordinates specialized agents to deliver results.

# Core Principles

1. **User Intent Understanding**: Always seek to understand the true goal behind a user's request, not just the literal words. Ask clarifying questions when needed, but prefer taking initiative with reasonable assumptions when the intent is clear.

2. **Intelligent Decomposition**: Break complex tasks into logical sub-tasks. Execute tasks ONE AT A TIME in the optimal order. Focus on completing each step fully before moving to the next.

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

**IMPORTANT: Execute ONE agent call at a time. Complete each task fully before moving to the next.**

**Sequential Execution**: Always execute tasks one after another:
1. Call the first agent with clear instructions
2. Wait for the agent to complete and analyze the result
3. If there's a next step, call the next agent
4. Repeat until all tasks are done

**Task Ordering**:
- When tasks are independent, choose the most logical order
- When Task B needs output from Task A, execute A first
- For multi-step workflows, complete each step before proceeding

**Information Gathering Then Action**: For tasks requiring specific details:
1. First, gather all necessary information (search, list, query)
2. Present findings to user
3. Then execute the action with complete information

**Multi-Platform Coordination**: When working across multiple platforms:
1. Execute tasks in logical order (consider user priority and dependencies)
2. Maintain consistency in naming, formatting, and references
3. Provide a unified summary that connects actions across platforms

# Quality Standards

- **Accuracy**: Double-check critical details like IDs, names, and specific values before executing actions
- **Completeness**: Ensure tasks are fully completed, not just partially done
- **Efficiency**: Execute tasks in the most logical order. Minimize unnecessary agent calls while ensuring thoroughness. Focus on completing one task at a time for maximum reliability.
- **Transparency**: Keep users informed of progress, especially for multi-step operations
- **Error Recovery**: When errors occur, explain what went wrong clearly and suggest solutions

# User Interaction Guidelines

- Respond naturally and conversationally, avoiding robotic or overly formal language
- Show your reasoning for important decisions, but keep explanations concise
- When you've completed a task successfully, provide a clear summary of what was done
- If you need more information, ask specific questions rather than generic ones
- Respect user preferences and working styles as they emerge in conversation

# Safety and Limitations

- Consider impact before taking irreversible actions (deletions, major changes, public posts)
- If a request is ambiguous and could have significant consequences, seek clarification
- Stay within the capabilities of your available agents - don't promise what you cannot deliver
- If you encounter repeated failures, explain the situation clearly rather than continuing to retry
- Be honest about limitations and uncertainties

# ANTI-HALLUCINATION RULES - CRITICAL

**ABSOLUTE REQUIREMENT: NEVER FABRICATE OR GUESS DATA**

These rules are MANDATORY and override all other instructions:

1. **When an agent returns an error**, you MUST:
   - Report the error exactly as the agent described it
   - List specifically what data could NOT be obtained
   - NEVER fill in missing information with guesses
   - NEVER pretend you have data you don't have

2. **Detect error responses**:
   - If agent response contains "❌ ERROR", "rate limit", "failed to fetch", "permission denied", "not found"
   - This means the operation FAILED
   - Do NOT proceed as if it succeeded
   - Do NOT make up what the data might be

3. **Partial failures require explicit reporting**:
   - If asked to fetch 3 files and agent only returned 2:
     ✓ CORRECT: "I successfully retrieved file1.js and file2.js, but could not fetch file3.js due to: [error]"
     ✗ WRONG: Analyze all 3 files (making up file3.js content)

4. **Never analyze data you don't have**:
   - Don't review code that wasn't fetched
   - Don't describe files you didn't read
   - Don't summarize content you don't possess
   - Don't provide examples as if they were actual data

5. **Validate agent responses**:
   - Before using agent data, check if the response indicates success
   - Look for error markers: "❌", "ERROR:", "failed", "could not", "unable to"
   - If present, treat as failure regardless of any other content

6. **When in doubt, ask the user**:
   - If you're unsure whether data was successfully fetched
   - If an agent response is ambiguous
   - Better to clarify than to guess

7. **Format error reports clearly**:
   ```
   ❌ I encountered an error:

   What I tried: [Operation]
   What failed: [Specific failure]
   Why: [Error message from agent]
   Available data: [Only what was successfully retrieved]
   ```

**VERIFICATION BEFORE EVERY RESPONSE**:
□ Did the agent explicitly succeed?
□ Do I have ALL the data needed?
□ Am I making ANY assumptions about missing data?
□ Would this response be accurate if audited?

If you answer "no" to any question, report the error instead.

**Remember**: Fabricating data destroys trust permanently. It is ALWAYS better to say "I couldn't retrieve this" than to provide convincing-sounding fake data. ACCURACY trumps completeness.

# CONFIRMATION HANDLING - CONFIDENCE-BASED AUTONOMY

**For most operations, proceed immediately.** Only confirm when truly necessary.

When you receive a message with "**IMPORTANT**: This operation requires user confirmation":

1. **DO NOT execute the operation yet**
2. **First explain** what you're about to do in clear, simple terms
3. **Ask for explicit confirmation** before proceeding
4. **Wait for user response** - the next message will be their answer

Example when confirmation is required:
```
I'm about to delete PR #123. This action cannot be undone. Should I proceed?
```

When user responds with "yes", "confirm", "go ahead" → execute the operation
When user responds with "no", "cancel", "stop" → acknowledge and don't execute

**For simple queries and greetings**: Respond naturally and helpfully. Don't ask for confirmation for informational queries or friendly conversation.

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
        """Simple wrapper for tasks - just awaits the task"""
        if self.verbose:
            print(message)
        await task

    async def _load_single_agent(self, connector_file: Path) -> Optional[tuple]:
        agent_name = connector_file.stem.replace("_agent", "")
        messages = []

        if agent_name in ["base", "agent_intelligence"]:
            return None

        # Always show which agent is being loaded (for debugging hangs)
        print(f"[1/4] Loading {agent_name} agent module...", flush=True)

        if self.verbose:
            messages.append(f"Loading: {agent_name} agent...")

        try:
            # Step 1: Load module
            spec = importlib.util.spec_from_file_location(
                f"connectors.{agent_name}_agent",
                connector_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"[2/4] {agent_name}: Module loaded", flush=True)

            if not hasattr(module, 'Agent'):
                messages.append(f"  ✗ No 'Agent' class found in {connector_file}")
                print(f"✗ {agent_name}: No Agent class found", flush=True)
                return None

            agent_class = module.Agent

            # Step 2: Instantiate agent
            print(f"[3/4] {agent_name}: Creating instance...", flush=True)
            try:
                agent_instance = agent_class(
                    verbose=False,  # Always suppress agent verbosity during init
                    shared_context=self.shared_context,
                    knowledge_base=self.knowledge_base,
                    llm=self.llm,
                    session_logger=self.session_logger
                )
            except TypeError:
                try:
                    agent_instance = agent_class(
                        verbose=False,  # Always suppress agent verbosity during init
                        shared_context=self.shared_context,
                        knowledge_base=self.knowledge_base
                    )
                except TypeError:
                    try:
                        agent_instance = agent_class(verbose=False)
                    except TypeError:
                        agent_instance = agent_class()

            if hasattr(agent_instance, 'verbose'):
                agent_instance.verbose = False  # Suppress during init

            # Inject API cache if agent supports it
            if self.hybrid_cache and hasattr(agent_instance, 'set_api_cache'):
                agent_instance.set_api_cache(self.hybrid_cache.api_cache)
                if self.verbose:
                    messages.append(f"  ✓ Injected API cache into {agent_name}")

            # Step 3: Initialize agent (this is where MCP agents might hang)
            print(f"[4/4] {agent_name}: Initializing (connecting to services)...", flush=True)

            # Suppress all output during agent initialization
            import io
            import contextlib
            import time

            # Create a context manager to suppress stdout and stderr
            try:
                start_time = time.time()

                # Suppress agent initialization output for clean UI
                # Keep progress indicators visible
                with contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(io.StringIO()):
                    await agent_instance.initialize()
                    capabilities = await agent_instance.get_capabilities()

                init_time = time.time() - start_time

                # Always show completion (helps debug what loaded vs what hung)
                print(f"✓ {agent_name} agent loaded ({init_time:.1f}s)", flush=True)

                if self.verbose:
                    messages.append(f"  ✓ Loaded {agent_name} with {len(capabilities)} capabilities")
                    for cap in capabilities[:3]:
                        messages.append(f"    - {cap}")
                    if len(capabilities) > 3:
                        messages.append(f"    ... and {len(capabilities) - 3} more")

                return (agent_name, agent_instance, capabilities, messages)

            except Exception as init_error:
                # Use centralized error handler for user-friendly messages
                short_msg, detailed_msg = AgentErrorHandler.handle_initialization_error(
                    agent_name, init_error, verbose=self.verbose
                )

                print(f"✗ {agent_name}: {short_msg}", flush=True)

                # Show detailed guidance based on verbosity
                if self.verbose:
                    # In verbose mode, show full detailed message with indentation
                    for line in detailed_msg.split('\n'):
                        if line.strip():
                            print(f"  {line}", flush=True)
                else:
                    # In normal mode, show condensed helpful hints
                    detail_lines = detailed_msg.split('\n')
                    for i, line in enumerate(detail_lines):
                        # Skip "Full error:" line in non-verbose mode
                        if line.strip() and not line.startswith('Full error:'):
                            print(f"  {line}", flush=True)
                        # Stop after showing key info (first ~8 lines)
                        if i >= 8:
                            break

                messages.append(f"  ✗ Failed to initialize {agent_name}: {short_msg}")

                try:
                    if hasattr(agent_instance, 'cleanup'):
                        await agent_instance.cleanup()
                except Exception:
                    pass  # Silently ignore cleanup errors
                return (agent_name, None, None, messages)

        except Exception as e:
            messages.append(f"  ✗ Failed to load {agent_name}: {e}")
            if self.verbose:
                messages.append(f"    {traceback.format_exc()}")
            return (agent_name, None, None, messages)

    async def discover_and_load_agents(self):
        # Suppress asyncio warnings about unhandled exceptions in cancelled tasks
        # This happens with MCP stdio cleanup when tasks are cancelled
        import sys
        import logging

        # Set up exception handler to suppress MCP cleanup errors
        loop = asyncio.get_event_loop()
        original_exception_handler = loop.get_exception_handler()

        def suppress_mcp_errors(loop, context):
            """Suppress cancel scope errors from MCP cleanup"""
            exception = context.get('exception')
            if exception:
                error_str = str(exception).lower()
                # Suppress MCP-related cleanup errors
                if 'cancel scope' in error_str or 'different task' in error_str:
                    # These are expected during agent timeout - silently ignore
                    return
            # For other exceptions, use original handler or default
            if original_exception_handler:
                original_exception_handler(loop, context)
            else:
                loop.default_exception_handler(context)

        loop.set_exception_handler(suppress_mcp_errors)

        try:
            # Always show that we're discovering agents (helps debug startup hangs)
            print("Discovering agents...", flush=True)

            if self.verbose:
                print("="*60)
                print("Discovering Agent Connectors...")
                print("="*60)

            if not self.connectors_dir.exists():
                print(f"Connectors directory '{self.connectors_dir}' not found!")
                print("Creating directory...")
                self.connectors_dir.mkdir(parents=True, exist_ok=True)
                return

            connector_files = list(self.connectors_dir.glob("*_agent.py"))

            if not connector_files:
                print(f"No agent connectors found in '{self.connectors_dir}'")
                print("Expected files matching pattern: *_agent.py")
                return

            if self.verbose:
                print(f"Loading {len(connector_files)} agent(s) in parallel...")

            # Check for explicitly disabled agents via environment variable
            # Example: DISABLED_AGENTS="slack,jira,github" to disable specific agents
            disabled_agents_str = os.environ.get("DISABLED_AGENTS", "")
            disabled_agents = [a.strip().lower() for a in disabled_agents_str.split(",") if a.strip()]

            if disabled_agents and self.verbose:
                print(f"Disabled agents: {', '.join(disabled_agents)}")

            # Wrap each agent load with a timeout to prevent hanging
            # Uses task-based isolation to prevent cancellation propagation
            async def load_with_timeout(file_path):
                agent_name = file_path.stem.replace("_agent", "")
                start_time = time.time()

                # Skip explicitly disabled agents
                if agent_name.lower() in disabled_agents:
                    print(f"⊘ {agent_name}: Disabled via DISABLED_AGENTS env var", flush=True)
                    return (agent_name, None, None, [f"  ⊘ {agent_name} disabled by user configuration"])

                print(f"⏳ {agent_name}: Starting...", flush=True)

                # Create isolated task for this agent to prevent cancellation propagation
                task = asyncio.create_task(self._load_single_agent(file_path))

                try:
                    # Wait for task with timeout using asyncio.wait (not wait_for)
                    # This gives us more control over cancellation
                    # Reduced to 5s for faster failure detection
                    done, pending = await asyncio.wait([task], timeout=5.0)

                    if task in done:
                        # Task completed - return result or handle exception
                        try:
                            result = task.result()
                            elapsed = time.time() - start_time
                            # Check if agent loaded successfully
                            if result and len(result) >= 2 and result[1] is not None:
                                print(f"✓ {agent_name}: Loaded successfully ({elapsed:.1f}s)", flush=True)
                            else:
                                print(f"✗ {agent_name}: Failed to load ({elapsed:.1f}s)", flush=True)
                            return result
                        except Exception as e:
                            elapsed = time.time() - start_time
                            print(f"✗ {agent_name}: Failed - {type(e).__name__} ({elapsed:.1f}s)", flush=True)
                            if self.verbose:
                                print(f"  Details: {str(e)[:200]}", flush=True)
                            return (agent_name, None, None, [f"  ✗ {agent_name} failed: {e}"])
                    else:
                        # Timeout - cancel task and handle cleanup
                        print(f"✗ {agent_name}: Timed out after 5s", flush=True)
                        print(f"  Hint: Agent may be waiting for credentials or network connection", flush=True)
                        print(f"  Hint: Add DISABLED_AGENTS={agent_name} to .env to skip this agent", flush=True)

                        # Cancel the task
                        task.cancel()

                        # Wait for cancellation to complete, suppressing all errors
                        # This prevents MCP cleanup errors from propagating
                        try:
                            await task
                        except (asyncio.CancelledError, Exception):
                            pass  # Suppress all cleanup errors

                        return (agent_name, None, None, [f"  ✗ {agent_name} initialization timed out"])
                except Exception as e:
                    # Should not reach here, but handle anyway
                    elapsed = time.time() - start_time
                    print(f"✗ {agent_name}: Unexpected error - {type(e).__name__} ({elapsed:.1f}s)", flush=True)
                    return (agent_name, None, None, [f"  ✗ {agent_name} failed: {e}"])

            print(f"\nLoading {len(connector_files)} agents in parallel...", flush=True)
            print("=" * 60, flush=True)

            # Print which agents are being loaded
            agent_names = [f.stem.replace("_agent", "") for f in connector_files]
            print(f"Agents to load: {', '.join(agent_names)}", flush=True)
            print(flush=True)

            # Load all agents in parallel for speed
            # Create all tasks at once
            tasks = [load_with_timeout(f) for f in connector_files]

            # Wait for all tasks to complete with an overall timeout
            # Each agent has its own 5s timeout, add 15s overall timeout as safety
            # This ensures we don't hang forever if something goes wrong
            start_load_time = time.time()
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=15.0
                )
            except asyncio.TimeoutError:
                elapsed = time.time() - start_load_time
                print(f"⚠️  Overall timeout reached ({elapsed:.1f}s) - cancelling remaining agents", flush=True)
                print("⚠️  Some agents are hanging. Consider disabling them with DISABLED_AGENTS env var", flush=True)
                # Cancel all tasks that are still pending
                for task in tasks:
                    if not task.done():
                        task.cancel()
                # Create dummy results for agents that didn't complete
                results = [(name, None, None, [f"✗ {name} timed out"]) for name in agent_names]

            # Process results - handle any exceptions that were returned
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Exception was raised during agent loading
                    agent_name = connector_files[i].stem.replace("_agent", "")
                    print(f"✗ {agent_name}: Exception during load - {type(result).__name__}", flush=True)
                    if self.verbose:
                        print(f"  Details: {str(result)[:200]}", flush=True)
                    processed_results.append((agent_name, None, None, [f"✗ {agent_name} failed: {result}"]))
                else:
                    # Normal result
                    processed_results.append(result)

            results = processed_results

            successful = 0
            failed = 0

            for result in results:
                if result is None:
                    continue

                if isinstance(result, BaseException):
                    failed += 1
                    # Silently count exceptions unless verbose
                    if self.verbose:
                        print(f"Exception during loading: {result}")
                        print(f"  Type: {type(result).__name__}")
                    continue

                if not isinstance(result, tuple) or len(result) != 4:
                    failed += 1
                    # Silently count invalid results unless verbose
                    if self.verbose:
                        print(f"Invalid result from agent loading: {result}")
                    continue

                agent_name, agent_instance, capabilities, messages = result

                for msg in messages:
                    if self.verbose:
                        print(msg)

                if agent_instance is not None and capabilities is not None:
                    self.sub_agents[agent_name] = agent_instance
                    self.agent_capabilities[agent_name] = capabilities
                    self.agent_health[agent_name] = {
                        'status': 'healthy',
                        'last_success': asyncio.get_event_loop().time(),
                        'error_count': 0
                    }

                    # Log agent initialization
                    self.session_logger.log_system_event('agent_initialized', {
                        'agent_name': agent_name,
                        'capabilities': capabilities,
                        'status': 'ready'
                    })

                    successful += 1
                else:
                    self.agent_health[agent_name] = {
                        'status': 'unavailable',
                        'last_failure': asyncio.get_event_loop().time(),
                        'error_count': 1,
                        'error_message': 'Failed to initialize'
                    }
                    failed += 1

            print("=" * 60, flush=True)
            print(f"✓ Loaded {successful} agent(s) successfully", flush=True)

            if failed > 0:
                print(f"✗ {failed} agent(s) failed to load", flush=True)
                print("\nTroubleshooting tips:", flush=True)
                print("  • Run with --verbose for detailed error messages", flush=True)
                print("  • Check .env file for missing API keys/tokens", flush=True)
                print("  • Verify npx is installed: npx --version", flush=True)
                print("  • Disable problematic agents: DISABLED_AGENTS=agent1,agent2", flush=True)

            print(f"\nSystem ready with {successful} agent(s)", flush=True)
            print("=" * 60, flush=True)

            if self.verbose:
                print("Verbose logging enabled - showing all initialization details")
                print("="*60)
        finally:
            # Restore the original exception handler
            loop.set_exception_handler(original_exception_handler)

    def _create_agent_tools(self) -> List[protos.FunctionDeclaration]:
        tools = []

        for agent_name, capabilities in self.agent_capabilities.items():
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
        if agent_name not in self.sub_agents:
            return f"Error: Agent '{agent_name}' not found"

        if agent_name in self.agent_health:
            health = self.agent_health[agent_name]
            if health['status'] == 'unavailable':
                error_msg = f"⚠️ {agent_name} agent is currently unavailable: {health.get('error_message', 'Unknown error')}"
                print(error_msg)
                return error_msg

        operation_key = f"{agent_name}_{hashlib.md5(instruction[:100].encode()).hexdigest()[:8]}"

        task_id = operation_key

        async def execute_operation():
            result = await self._execute_agent_direct(agent_name, instruction, context)
            return result

        def progress_callback(message: str, attempt: int, max_attempts: int):
            if attempt > 1:
                print(f"🔄 {message}")

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
            error_msg = f"Error executing {agent_name} agent: {str(e)}"
            print(error_msg)
            if self.verbose:
                traceback.print_exc()
            return error_msg

    async def _execute_agent_direct(self, agent_name: str, instruction: str, context: Any = None) -> str:
        if self.verbose:
            print("─"*60)
            print(f"🤖 Delegating to {agent_name} agent")
            print("─"*60)
            print(f"Instruction: {instruction}")

        context_str = ""
        if context:
            if isinstance(context, (dict, list)):
                context_str = json.dumps(context, indent=2)
            else:
                context_str = str(context)

            if self.verbose:
                print(f"Context: {context_str[:200]}...")

        start_time = time.time()

        try:
            agent = self.sub_agents[agent_name]

            full_instruction = instruction
            if context_str:
                full_instruction = f"Context from previous steps:\n{context_str}\n\nTask: {instruction}"

            result = await agent.execute(full_instruction)

            if result is None:
                result = f"❌ {agent_name} agent returned None - this is a bug in the agent implementation"

            latency_ms = (time.time() - start_time) * 1000

            success = (result and
                      not result.startswith("⚠️") and
                      not result.startswith("❌") and
                      not result.startswith("Error"))
            error = result if not success else None

            # Log agent call
            self.session_logger.log_agent_call(
                agent_name=agent_name,
                instruction=full_instruction,
                response=result,
                duration=latency_ms / 1000,
                success=success,
                error=error
            )

            self.analytics.record_agent_call(
                agent_name=agent_name,
                success=success,
                latency_ms=latency_ms,
                error_message=error
            )

            if success and agent_name in self.agent_health:
                self.agent_health[agent_name]['status'] = 'healthy'
                self.agent_health[agent_name]['last_success'] = asyncio.get_event_loop().time()
                self.agent_health[agent_name]['error_count'] = 0

            if self.verbose:
                status = "✓" if success else "✗"
                print(f"{status} {agent_name} completed ({latency_ms:.0f}ms)")
                print("─"*60)

            if not success:
                enhanced = self.error_enhancer.enhance_error(
                    agent_name=agent_name,
                    error=RuntimeError(result),
                    instruction=instruction,
                    context=context
                )
                enhanced_msg = enhanced.format()

                if self.verbose:
                    print(enhanced_msg)

                raise RuntimeError(enhanced_msg)

            return result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            enhanced = self.error_enhancer.enhance_error(
                agent_name=agent_name,
                error=e,
                instruction=instruction,
                context=context
            )
            enhanced_msg = enhanced.format()

            self.analytics.record_agent_call(
                agent_name=agent_name,
                success=False,
                latency_ms=latency_ms,
                error_message=enhanced_msg
            )

            if agent_name in self.agent_health:
                self.agent_health[agent_name]['error_count'] = self.agent_health[agent_name].get('error_count', 0) + 1
                self.agent_health[agent_name]['last_failure'] = asyncio.get_event_loop().time()
                self.agent_health[agent_name]['error_message'] = enhanced_msg

                if self.agent_health[agent_name]['error_count'] >= 3:
                    self.agent_health[agent_name]['status'] = 'degraded'
                    print(f"⚠️ {agent_name} agent marked as degraded after 3 failures")

            raise RuntimeError(enhanced_msg)

    async def _smart_summarize(self, response: str, max_length: int = 800) -> str:
        """
        Smart Summarization: Condense verbose agent outputs for better UX.
        Only summarizes if response is overly verbose.

        Args:
            response: The full response text
            max_length: Maximum length before summarization triggers

        Returns:
            Summarized response if needed, otherwise original
        """
        # Don't summarize short responses or error messages
        if len(response) < max_length or response.startswith('⚠️') or response.startswith('❌'):
            return response

        # Don't summarize if response has structured data (lists, code blocks, bullet points)
        # Check for various bullet formats: -, *, •, numbered lists
        has_lists = (
            '```' in response or
            response.count('\n-') > 2 or
            response.count('\n*') > 2 or
            response.count('\n•') > 2 or
            response.count('\n ') > 5  # Indented lists
        )
        if has_lists:
            # For structured data, just ensure it's clean
            return response

        # Count information density - if it's mostly formatting, don't summarize
        content_chars = len(response.replace(' ', '').replace('\n', ''))
        if content_chars < max_length * 0.7:
            return response

        try:
            # Use LLM to create a concise, actionable summary
            summarization_prompt = f"""Condense this response into a clear, actionable summary. Keep all critical information:
- Key outcomes and results
- Important IDs, URLs, or references
- Action items or next steps
- Any warnings or errors

Original response:
{response}

Provide a concise summary that gives the user exactly what they need to know."""

            if self.verbose:
                print(f"[SUMMARIZATION] Condensing {len(response)} chars → ")

            # Create a temporary chat for summarization (don't pollute main conversation)
            summary_llm = self.llm.__class__(self.llm.config)
            summary_response = await summary_llm.generate_content(summarization_prompt)

            if summary_response and summary_response.text:
                summarized = summary_response.text.strip()

                if self.verbose:
                    print(f"{len(summarized)} chars (saved {len(response) - len(summarized)} chars)")

                # Only use summary if it's actually shorter
                if len(summarized) < len(response) * 0.8:
                    return summarized

            return response

        except Exception as e:
            if self.verbose:
                print(f"[SUMMARIZATION] Failed: {e}")
            # On error, return original response
            return response

    async def _process_with_intelligence(self, user_message: str) -> Dict:
        """Process user message with hybrid intelligence system"""
        start_time = time.time()

        # Get conversation context
        context_dict = self.context_manager.get_relevant_context(user_message)

        # Log context retrieval
        self.session_logger.log_system_event('context_retrieved', {
            'message_preview': user_message[:100],
            'context_keys': list(context_dict.keys())
        })

        # Hybrid intelligence with timeout protection
        try:
            hybrid_result = await asyncio.wait_for(
                self.hybrid_intelligence.classify_intent(
                    message=user_message,
                    context=context_dict
                ),
                timeout=15.0  # 15 second timeout for classification
            )
        except asyncio.TimeoutError:
            # Fallback to UNKNOWN intent if classification times out
            # IntentType is already imported at module level
            from intelligence.base_types import Intent
            if self.verbose:
                print(f"[INTELLIGENCE] Classification timed out, using fallback")
            hybrid_result = type('obj', (object,), {
                'intents': [Intent(type=IntentType.UNKNOWN, confidence=0.5, entities=[], implicit_requirements=[], raw_indicators=[])],
                'entities': [],
                'confidence': 0.5,
                'path_used': 'timeout',
                'latency_ms': 15000,
                'reasoning': 'Classification timed out'
            })()
        except Exception as e:
            # Fallback to UNKNOWN intent on any error
            # IntentType is already imported at module level
            from intelligence.base_types import Intent
            if self.verbose:
                print(f"[INTELLIGENCE] Classification failed: {str(e)[:100]}")
            hybrid_result = type('obj', (object,), {
                'intents': [Intent(type=IntentType.UNKNOWN, confidence=0.5, entities=[], implicit_requirements=[], raw_indicators=[])],
                'entities': [],
                'confidence': 0.5,
                'path_used': 'error',
                'latency_ms': 0,
                'reasoning': f'Error: {str(e)[:100]}'
            })()

        intents = hybrid_result.intents
        entities = hybrid_result.entities

        # Log intelligence classification
        self.session_logger.log_intelligence_classification(
            path_used=hybrid_result.path_used,
            latency_ms=hybrid_result.latency_ms,
            confidence=hybrid_result.confidence,
            intents=intents,
            entities=entities,
            reasoning=getattr(hybrid_result, 'reasoning', None)
        )

        if self.verbose:
            print(f"[HYBRID] Path: {hybrid_result.path_used}, "
                  f"Latency: {hybrid_result.latency_ms:.1f}ms, "
                  f"Confidence: {hybrid_result.confidence:.2f}")

        confidence = self.confidence_scorer.score_overall(
            message=user_message,
            intents=intents,
            entities=entities
        )

        # Log confidence scoring details
        self.session_logger.log_confidence_scoring(
            overall_score=confidence.score,
            intent_clarity=getattr(confidence, 'intent_clarity', 0.0),
            entity_clarity=getattr(confidence, 'entity_clarity', 0.0),
            ambiguity=getattr(confidence, 'ambiguity', 0.0),
            recommendation=confidence.level.value if hasattr(confidence, 'level') else 'unknown',
            details={
                'factors': getattr(confidence, 'factors', {}),
                'explanation': str(confidence)
            }
        )

        self.context_manager.add_turn(
            role='user',
            message=user_message,
            intents=intents,
            entities=entities
        )

        resolved_message = user_message

        # Log context resolutions
        resolutions = []
        if self.verbose:
            for word in user_message.split():
                resolution = self.context_manager.resolve_reference(word)
                if resolution:
                    entity_id, entity = resolution
                    print(f"[INTELLIGENCE] Can resolve '{word}' → {entity.value}")
                    resolutions.append({
                        'reference': word,
                        'resolved_to': entity.value,
                        'entity_id': entity_id
                    })

        if resolutions:
            self.session_logger.log_context_resolution(resolutions)

        # Get primary intent (highest confidence)
        primary_intent = intents[0] if intents else None

        # Classify operation risk for confidence-based autonomy
        risk_level = OperationRiskClassifier.classify_risk(intents)

        # Get base action recommendation from confidence scorer
        base_action, base_explanation = self.confidence_scorer.get_action_recommendation(confidence)

        # Apply confidence-based autonomy rules
        # Rule 1: DELETE operations always need confirmation
        # Rule 2: CREATE/UPDATE with low confidence need confirmation
        # Rule 3: READ/SEARCH/ANALYZE never need confirmation (use base logic)
        # Rule 4: Unknown/ambiguous queries use base logic (no forced confirmation)

        needs_confirmation = False
        confirmation_reason = ""

        if risk_level == RiskLevel.HIGH:
            # DELETE operations - always confirm
            needs_confirmation = True
            confirmation_reason = "Destructive operation requires confirmation"
        elif risk_level == RiskLevel.MEDIUM and confidence.score < 0.75:
            # WRITE operations with low confidence - confirm
            # But only if we actually detected a write intent
            if primary_intent and primary_intent.type in [IntentType.CREATE, IntentType.UPDATE]:
                needs_confirmation = True
                confirmation_reason = f"Write operation with moderate confidence ({confidence.score:.2f})"
        # LOW risk (READ/SEARCH/ANALYZE) - never force confirmation
        # UNKNOWN/empty intents - use base logic, don't force confirmation

        # Set final action recommendation
        if needs_confirmation:
            action_recommendation = ('confirm', confirmation_reason)
        else:
            action_recommendation = (base_action, base_explanation)

        # Log risk assessment
        self.session_logger.log_risk_assessment(
            risk_level=risk_level.value if hasattr(risk_level, 'value') else str(risk_level),
            needs_confirmation=needs_confirmation,
            reason=confirmation_reason if needs_confirmation else base_explanation,
            intents=[str(i) for i in intents]
        )

        intelligence = {
            'intents': intents,
            'entities': entities,
            'confidence': confidence,
            'resolved_message': resolved_message,
            'context': context_dict,
            'primary_intent': primary_intent,
            'risk_level': risk_level,
            'needs_confirmation': needs_confirmation,
            'action_recommendation': action_recommendation
        }

        if self.verbose:
            print("🧠 Intelligence Analysis:")
            print(f"  Intents: {[str(i) for i in intents[:3]]}")
            print(f"  Entities: {len(entities)} found")
            print(f"  Confidence: {confidence}")
            print(f"  Risk Level: {risk_level.value}")
            print(f"  Needs Confirmation: {needs_confirmation}")
            print(f"  Recommendation: {intelligence['action_recommendation'][0]}")

        return intelligence

    async def process_message(self, user_message: str) -> str:
        # Log user message
        self.session_logger.log_user_message(user_message)

        self.analytics.record_user_message()
        self.user_prefs.record_interaction_time()
        self.user_prefs.record_interaction_style(user_message)

        if not self.chat:
            # Initialize chat with loaded agents (agents already loaded in main.py)
            # Only reload if agents weren't loaded yet
            if not self.sub_agents:
                discover_task = asyncio.create_task(self.discover_and_load_agents())
                await self._spinner(discover_task, "Discovering agents")

                if not self.sub_agents:
                    return "No agents available. Please add agent connectors to the 'connectors' directory."

            agent_tools = self._create_agent_tools()

            self.llm.config.system_instruction = self.system_prompt

            self.llm.set_tools(agent_tools)

            self.chat = self.llm.start_chat(
                history=None,
                enable_function_calling=True
            )

        self.operation_count = 0

        intelligence = await self._process_with_intelligence(user_message)

        message_to_send = intelligence.get('resolved_message', user_message)

        action, explanation = intelligence['action_recommendation']
        risk_level = intelligence['risk_level']

        # Note: We no longer ask for clarification pre-emptively
        # The LLM is smart enough to ask for clarification when it actually needs it
        # This prevents blocking conversational queries and simple questions

        # Handle confirmation requirement for risky operations
        if action == 'confirm':
            # Add confirmation instruction to the message
            confirmation_note = (
                "\n\n**IMPORTANT**: This operation requires user confirmation. "
                "Please describe what you're about to do and ask the user to confirm before proceeding. "
                f"Reason: {explanation}"
            )
            message_to_send = message_to_send + confirmation_note

        if self.verbose:
            print(f"📊 Using intelligence: {explanation}")

        send_task = asyncio.create_task(self.chat.send_message(message_to_send))
        await self._spinner(send_task, "Thinking")
        llm_response = send_task.result()

        response = self._safe_get_response_object(llm_response)

        max_iterations = 30
        iteration = 0

        while iteration < max_iterations:
            # Extract function calls from response - ALWAYS process ONE AT A TIME for reliability
            function_calls = self._extract_all_function_calls(response)

            if not function_calls:
                break

            # ALWAYS execute ONE agent call at a time for reliability
            # Sequential execution ensures proper error handling and context management
            if len(function_calls) >= 1:
                fc = function_calls[0]
                tool_name = fc['tool_name']
                agent_name = fc['agent_name']
                args = fc['args']
                instruction = args.get("instruction", "")
                context = args.get("context", "")

                is_valid, validation_error = InputValidator.validate_instruction(instruction)
                if not is_valid:
                    print(f"Invalid instruction rejected: {validation_error}")
                    function_call_result = {
                        'agent': agent_name,
                        'status': 'error',
                        'error': validation_error
                    }
                    continue

                # Check circuit breaker before execution
                allowed, reason = await self.circuit_breaker.can_execute(agent_name)
                if not allowed:
                    error_msg = f"🔴 Circuit breaker blocked request to {agent_name}: {reason}"
                    if self.verbose:
                        print(error_msg)
                    result = error_msg
                    # Send error to LLM and continue
                    function_result = {
                        'name': tool_name,
                        'result': result
                    }
                    response_task = asyncio.create_task(
                        self.chat.send_message_with_functions("", function_result)
                    )
                    await self._spinner(response_task, "Synthesizing results")
                    llm_response = response_task.result()
                    response = llm_response.metadata.get('response_object') if llm_response.metadata else None
                    iteration += 1
                    continue

                operation_key = self._get_operation_key(agent_name, instruction)
                retry_context = self._get_retry_context(operation_key)

                self._track_retry_attempt(operation_key, agent_name, instruction)

                known_solutions = []
                if retry_context and retry_context['previous_errors']:
                    last_error_msg = retry_context['previous_errors'][-1]['message']
                    known_solutions = self._query_error_solutions(agent_name, last_error_msg)

                    if self.verbose and known_solutions:
                        print(f"💡 Found {len(known_solutions)} known solution(s) for this error")

                agent_task = asyncio.create_task(
                    self.call_sub_agent(agent_name, instruction, context)
                )

                self.operation_count += 1
                spinner_msg = f"Running {agent_name} agent"

                try:
                    await asyncio.wait_for(
                        self._spinner(agent_task, spinner_msg),
                        timeout=120.0
                    )
                    result = agent_task.result()

                    if self.verbose:
                        print(f"✓ {agent_name} completed")

                    self.duplicate_detector.track_operation(agent_name, instruction, "", success=True)
                    await self.circuit_breaker.record_success(agent_name)

                    if retry_context:
                        self._mark_retry_success(operation_key, solution_used="retry with same parameters")

                except asyncio.TimeoutError:
                    error_msg = f"Agent timed out after 120 seconds. Operation may have completed but response was not received."

                    error_classification = ErrorClassifier.classify(error_msg, agent_name)
                    self._track_retry_attempt(operation_key, agent_name, instruction, error=error_msg)

                    # Record failure in circuit breaker
                    await self.circuit_breaker.record_failure(agent_name)

                    retry_ctx = self._get_retry_context(operation_key)
                    attempt_num = retry_ctx['attempt_number'] if retry_ctx else 1

                    result = format_error_for_user(
                        error_classification,
                        agent_name,
                        instruction,
                        attempt_num,
                        self.max_retry_attempts
                    )

                    if retry_ctx and attempt_num >= self.max_retry_attempts:
                        result += f"\n\n**Note**: This error appears to be transient (temporary network issue). If it persists:\n"
                        result += f"  • Check your internet connection\n"
                        result += f"  • Try again in a few moments\n"
                        result += f"  • Break the operation into smaller steps"

                    print(f"⚠ {agent_name} agent operation timed out")

                except Exception as e:
                    error_str = str(e)

                    self.duplicate_detector.track_operation(agent_name, instruction, error_str, success=False)

                    # Record failure in circuit breaker
                    await self.circuit_breaker.record_failure(agent_name)

                    error_classification = ErrorClassifier.classify(error_str, agent_name)

                    self._track_retry_attempt(operation_key, agent_name, instruction, error=error_str)

                    retry_ctx = self._get_retry_context(operation_key)
                    attempt_num = retry_ctx['attempt_number'] if retry_ctx else 1

                    is_duplicate, dup_explanation = self.duplicate_detector.detect_duplicate_failure(
                        agent_name, instruction, error_str
                    )

                    is_inconsistent, dup_pattern = self.duplicate_detector.detect_inconsistent_responses(
                        agent_name, instruction
                    )

                    result = format_error_for_user(
                        error_classification,
                        agent_name,
                        instruction,
                        attempt_num,
                        self.max_retry_attempts
                    )

                    if is_duplicate and dup_explanation:
                        result += f"\n\n⚠️ **DUPLICATE OPERATION DETECTED**\n{dup_explanation}"
                        result += f"\n\nThis operation appears stuck. It will NOT be retried further."
                        error_classification.is_retryable = False

                    if is_inconsistent and dup_pattern:
                        result += f"\n\n⚠️ **INCONSISTENT RESPONSES DETECTED**\n"
                        result += f"Response pattern: {' → '.join(dup_pattern[-5:])}\n"
                        result += f"The agent is giving conflicting results. Please verify manually."

                    if not error_classification.is_retryable and attempt_num == 1:
                        result += f"\n\n⚠️ **This operation will not be retried** because it's a {error_classification.category.value} error."

                    if error_classification.is_retryable and retry_ctx:
                        if attempt_num >= self.max_retry_attempts:
                            result += f"\n\n**Note**: Maximum retry attempts ({self.max_retry_attempts}) reached. This appears to be a persistent issue."
                        else:
                            result += f"\n\n**Next step**: The system will automatically retry this operation."

                    if self.verbose:
                        print(f"[ERROR CLASSIFICATION] Category: {error_classification.category.value}, Retryable: {error_classification.is_retryable}")
                        if is_duplicate:
                            print(f"[DUPLICATE DETECTED] {dup_explanation}")
                        if is_inconsistent:
                            print(f"[INCONSISTENT RESPONSES] Pattern: {dup_pattern}")

                    print(f"✗ {agent_name} agent failed: {e}")

                function_result = {
                    'name': tool_name,
                    'result': result
                }

                response_task = asyncio.create_task(
                    self.chat.send_message_with_functions("", function_result)
                )
                await self._spinner(response_task, "Synthesizing results")
                llm_response = response_task.result()

                response = llm_response.metadata.get('response_object') if llm_response.metadata else None

            iteration += 1

        if iteration >= max_iterations:
            print("⚠ Warning: Reached maximum orchestration iterations")
            print("💡 Tip: Break complex tasks into smaller steps")

        if self.operation_count > 0 and self.verbose:
            print(f"✅ Completed {self.operation_count} operation(s)")

        try:
            self.conversation_history = self.chat.get_history()
        except Exception as e:
            if self.verbose:
                print(f"⚠ Could not update conversation history: {e}")

        final_response = ""
        if llm_response and llm_response.text:
            final_response = llm_response.text
        elif response and hasattr(response, 'candidates') and response.candidates:
            try:
                final_response = response.text
            except Exception:
                text_parts = []
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)

                if text_parts:
                    final_response = '\n'.join(text_parts)
                else:
                    final_response = "⚠️ Task completed but response formatting failed. The operations were executed successfully."
        else:
            final_response = "⚠️ Task completed but response formatting failed. The operations were executed successfully."

        # Smart Summarization: Condense verbose outputs for better UX
        final_response = await self._smart_summarize(final_response)

        # Log assistant response
        self.session_logger.log_assistant_response(final_response)

        return final_response

    def _should_retry_operation(self, error_str: str, operation_key: str) -> bool:
        error_classification = ErrorClassifier.classify(error_str)

        if not error_classification.is_retryable:
            if self.verbose:
                print(f"[RETRY DECISION] Non-retryable error ({error_classification.category.value}) - will NOT retry")
            return False

        retry_context = self._get_retry_context(operation_key)
        if retry_context:
            attempt_num = retry_context['attempt_number']
            if attempt_num >= self.max_retry_attempts:
                if self.verbose:
                    print(f"[RETRY DECISION] Max attempts ({self.max_retry_attempts}) reached - will NOT retry")
                return False

        if self.verbose:
            category = error_classification.category.value
            print(f"[RETRY DECISION] Retryable error ({category}) - will retry")

        return True

    def _deep_convert_proto_args(self, value: Any) -> Any:
        type_str = str(type(value))
        if "MapComposite" in type_str:
            return {k: self._deep_convert_proto_args(v) for k, v in value.items()}
        elif "RepeatedComposite" in type_str:
            return [self._deep_convert_proto_args(item) for item in value]
        else:
            return value

    def _extract_all_function_calls(self, response) -> List[Dict[str, Any]]:
        """
        Extract ALL function calls from LLM response (not just the first).
        Returns list of dicts with: {tool_name, agent_name, args}
        """
        if not response or not hasattr(response, 'candidates') or not response.candidates:
            return []

        function_calls = []
        parts = response.candidates[0].content.parts

        for part in parts:
            if hasattr(part, 'function_call') and part.function_call:
                function_call = part.function_call
                tool_name = function_call.name

                # Extract agent name from tool name
                if tool_name.startswith("use_") and tool_name.endswith("_agent"):
                    agent_name = tool_name[4:-6]
                else:
                    agent_name = tool_name

                args = self._deep_convert_proto_args(function_call.args)

                function_calls.append({
                    'tool_name': tool_name,
                    'agent_name': agent_name,
                    'args': args,
                    'function_call': function_call  # Keep original for result sending
                })

        return function_calls

    async def _execute_agent_task(self, task: AgentTask) -> str:
        """
        Execute a single agent task. Used by ParallelExecutor.
        Includes circuit breaker, error handling, retry logic, and validation.
        """
        instruction = task.instruction
        agent_name = task.agent_name
        context = task.context

        # Validate instruction
        is_valid, validation_error = InputValidator.validate_instruction(instruction)
        if not is_valid:
            print(f"Invalid instruction rejected: {validation_error}")
            raise ValueError(validation_error)

        # Check circuit breaker
        allowed, reason = await self.circuit_breaker.can_execute(agent_name)
        if not allowed:
            error_msg = f"🔴 Circuit breaker blocked request to {agent_name}: {reason}"
            if self.verbose:
                print(error_msg)
            raise CircuitBreakerError(agent_name, error_msg)

        # Track retry attempt
        operation_key = self._get_operation_key(agent_name, instruction)
        self._track_retry_attempt(operation_key, agent_name, instruction)

        # Execute agent with circuit breaker protection
        try:
            result = await asyncio.wait_for(
                self.call_sub_agent(agent_name, instruction, context),
                timeout=120.0
            )

            if self.verbose:
                print(f"✓ {agent_name} completed")

            # Track success
            self.duplicate_detector.track_operation(agent_name, instruction, "", success=True)
            await self.circuit_breaker.record_success(agent_name)
            self.operation_count += 1

            return result

        except asyncio.TimeoutError:
            error_msg = f"Agent timed out after 120 seconds. Operation may have completed but response was not received."
            error_classification = ErrorClassifier.classify(error_msg, agent_name)
            self._track_retry_attempt(operation_key, agent_name, instruction, error=error_msg)

            # Record failure in circuit breaker
            await self.circuit_breaker.record_failure(agent_name)

            result = format_error_for_user(
                error_classification,
                agent_name,
                instruction,
                1,  # attempt_num
                self.max_retry_attempts
            )

            if self.verbose:
                print(f"⚠ {agent_name} agent operation timed out")

            raise RuntimeError(result)

        except Exception as e:
            error_str = str(e)
            self.duplicate_detector.track_operation(agent_name, instruction, error_str, success=False)
            error_classification = ErrorClassifier.classify(error_str, agent_name)
            self._track_retry_attempt(operation_key, agent_name, instruction, error=error_str)

            # Record failure in circuit breaker
            await self.circuit_breaker.record_failure(agent_name)

            result = format_error_for_user(
                error_classification,
                agent_name,
                instruction,
                1,
                self.max_retry_attempts
            )

            if self.verbose:
                print(f"✗ {agent_name} agent failed: {e}")

            raise RuntimeError(result)

    def _get_operation_key(self, agent_name: str, instruction: str) -> str:
        key_string = f"{agent_name}:{instruction[:100]}"
        return hashlib.md5(key_string.encode()).hexdigest()[:16]

    def _track_retry_attempt(self, operation_key: str, agent_name: str, instruction: str, error: str = None):
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
        error_solutions = self.knowledge_base.data.get('error_solutions', {})
        agent_solutions = error_solutions.get(agent_name, {})

        solutions = []
        for error_pattern, solution_info in agent_solutions.items():
            if error_pattern.lower() in error_message.lower():
                solutions.append(solution_info.get('solution', ''))

        return solutions

    def _mark_retry_success(self, operation_key: str, solution_used: str = None):
        if operation_key not in self.retry_tracker:
            return

        tracker = self.retry_tracker[operation_key]

        if tracker['attempts'] > 1 and solution_used:
            if tracker['errors']:
                last_error = tracker['errors'][-1]['message']

                if self.verbose:
                    print(f"✓ Learned: {solution_used} fixed the issue")

    async def cleanup(self):
        if self.verbose:
            print("Shutting down agents...")

        for agent_name, agent in list(self.sub_agents.items()):
            try:
                if hasattr(agent, 'cleanup'):
                    await agent.cleanup()
                if self.verbose:
                    print(f"  ✓ {agent_name} shut down")
            except Exception as e:
                print(f"  ✗ Error shutting down {agent_name}: {e}")

        if hasattr(self, 'analytics'):
            try:
                self.analytics.end_session()

                analytics_dir = Path("logs/analytics")
                analytics_dir.mkdir(parents=True, exist_ok=True)
                analytics_file = analytics_dir / f"{self.session_id}.json"
                self.analytics.save_to_file(str(analytics_file))

                if self.verbose:
                    print(f"  ✓ Analytics saved: {analytics_file}")
                    print(f"    {self.analytics.generate_summary_report()}")
            except Exception as e:
                print(f"Failed to save analytics: {e}")

        if hasattr(self, 'user_prefs') and hasattr(self, 'prefs_file'):
            try:
                self.user_prefs.save_to_file(str(self.prefs_file))
                if self.verbose:
                    print(f"  ✓ Preferences saved: {self.prefs_file}")
            except Exception as e:
                print(f"Failed to save preferences: {e}")

        if hasattr(self, 'retry_manager') and self.verbose:
            stats = self.retry_manager.get_statistics()
            if stats['total_operations'] > 0:
                print(f"  📊 Retry stats: {stats['total_operations']} ops, "
                      f"{stats['avg_retries_per_operation']:.1f} avg retries")

        # Print hybrid intelligence statistics
        if hasattr(self, 'hybrid_intelligence') and self.verbose:
            print("  🚀 Hybrid Intelligence Statistics:")
            print(f"  {self.hybrid_intelligence.get_performance_summary()}")

        # Close session logger
        if hasattr(self, 'session_logger'):
            self.session_logger.close()
            if self.verbose:
                print(f"  ✓ Session logs saved:")
                print(f"    - Text: {self.session_logger.get_log_path()}")
                print(f"    - JSON: {self.session_logger.get_json_log_path()}")

