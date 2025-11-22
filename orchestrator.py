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
from core.simple_session_logger import SimpleSessionLogger

# Import advanced intelligence system
from intelligence import (
    TaskDecomposer,
    ConfidenceScorer, ConversationContextManager
)

# Import Hybrid Intelligence System v5.0 (Two-tier: Fast Filter + LLM)
from intelligence.hybrid_system import HybridIntelligenceSystem, HybridIntelligenceResult

# Import terminal UI
from ui.terminal_ui import TerminalUI

# Import production utilities
from config import Config
from core.logger import get_logger
from core.input_validator import InputValidator
from core.error_handler import ErrorClassifier, format_error_for_user, DuplicateOperationDetector

# Import observability system
from core.observability import initialize_observability, get_observability

# Import new enhancement systems
from core.retry_manager import RetryManager
from core.circuit_breaker import CircuitBreaker, CircuitConfig
from core.undo_manager import UndoManager, UndoableOperationType
from core.user_preferences import UserPreferenceManager
from core.analytics import AnalyticsCollector
from core.error_messaging import ErrorMessageEnhancer
from core.datetime_context import (
    get_datetime_context, set_user_timezone, format_datetime_for_prompt,
    format_datetime_for_instruction
)

# Import intelligent instruction system
from intelligence.instruction_parser import (
    IntelligentInstructionParser, ParsedInstruction
)

# Import unified memory system (consolidates all memory systems)
from core.unified_memory import UnifiedMemory, MemoryIntentType

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
    """
    Main orchestration agent that coordinates specialized sub-agents.

    Features:
    - Hybrid Intelligence System v5.0 (Fast Filter + LLM Classifier)
    - Circuit Breaker Pattern (prevents cascading failures)
    - Smart Retry Management (exponential backoff with jitter)
    - Undo System (reversible operations)
    - User Preference Learning
    - Comprehensive Analytics & Observability
    """

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

    def _extract_token_usage(self, llm_response) -> Dict[str, int]:
        """Extract token usage from LLM response metadata."""
        tokens = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        try:
            response = self._safe_get_response_object(llm_response)
            if response and hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                tokens['prompt_tokens'] = getattr(usage, 'prompt_token_count', 0) or 0
                tokens['completion_tokens'] = getattr(usage, 'candidates_token_count', 0) or 0
                tokens['total_tokens'] = getattr(usage, 'total_token_count', 0) or 0
        except Exception as e:
            if self.verbose:
                print(f"{C.YELLOW}⚠ Could not extract token usage: {e}{C.ENDC}")
        return tokens

    def _track_tokens(self, llm_response):
        """Track token usage from an LLM response."""
        tokens = self._extract_token_usage(llm_response)
        # Update cumulative totals
        self.token_usage['prompt_tokens'] += tokens['prompt_tokens']
        self.token_usage['completion_tokens'] += tokens['completion_tokens']
        self.token_usage['total_tokens'] += tokens['total_tokens']
        # Update last message totals
        self.last_message_tokens['prompt_tokens'] += tokens['prompt_tokens']
        self.last_message_tokens['completion_tokens'] += tokens['completion_tokens']
        self.last_message_tokens['total_tokens'] += tokens['total_tokens']

    def get_token_usage(self) -> Dict[str, int]:
        """Get cumulative token usage for this session."""
        return self.token_usage.copy()

    def get_last_message_tokens(self) -> Dict[str, int]:
        """Get token usage for the last processed message."""
        return self.last_message_tokens.copy()

    def reset_token_usage(self):
        """Reset all token usage counters."""
        self.token_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        self.last_message_tokens = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

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

        # Feature #8: Intelligent Retry tracking
        self.retry_tracker: Dict[str, Dict[str, Any]] = {}  # Track retry attempts per operation
        self.max_retry_attempts = 3  # Maximum retries before suggesting alternative

        # Feature #21: Duplicate Operation Detection
        self.duplicate_detector = DuplicateOperationDetector(window_size=5, similarity_threshold=0.8)

        # Feature #11: Simple progress tracking for streaming
        self.operation_count = 0  # Track number of operations in current request

        # Token usage tracking
        self.token_usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }
        self.last_message_tokens = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }

        # SIMPLE SESSION LOGGER - 2 human-readable text files per session
        # This replaces the old SessionLogger and UnifiedSessionLogger
        self.simple_logger = SimpleSessionLogger(session_id=self.session_id, log_dir="logs")

        # Initialize observability system (tracing, metrics, specialized loggers)
        self.observability = initialize_observability(
            session_id=self.session_id,
            service_name="orchestrator",
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            enable_tracing=True,
            enable_metrics=True,
            verbose=self.verbose
        )

        # Get specialized loggers
        self.orch_logger = self.observability.orchestration_logger
        self.intel_logger = self.observability.intelligence_logger

        # Terminal UI
        self.ui = TerminalUI(verbose=self.verbose)

        # Advanced Intelligence System (Phase 1)
        # HYBRID INTELLIGENCE v5.0: Two-tier system (Fast Filter + LLM)
        # Replaces separate intent_classifier and entity_extractor with unified hybrid system
        self.hybrid_intelligence = HybridIntelligenceSystem(
            llm_client=self.llm,  # Pass LLM for Tier 2 semantic analysis
            verbose=self.verbose
        )

        # Other intelligence components
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

        # 2. Circuit Breaker - Prevent cascading failures
        self.circuit_breaker = CircuitBreaker(
            config=CircuitConfig(
                failure_threshold=5,        # Open circuit after 5 consecutive failures
                success_threshold=2,        # Close circuit after 2 consecutive successes
                timeout_seconds=300.0,      # Wait 5 minutes before testing recovery
                half_open_timeout=10.0      # Max 10 seconds in half-open state
            ),
            verbose=self.verbose
        )

        # 3. Undo Manager - Undo destructive operations
        self.undo_manager = UndoManager(
            max_undo_history=20,
            default_ttl_seconds=3600,  # 1 hour
            verbose=self.verbose
        )

        # 4. User Preferences - Learn from user behavior
        user_id = os.environ.get("USER_ID", "default")
        self.user_prefs = UserPreferenceManager(
            user_id=user_id,
            min_confidence_threshold=0.7,
            verbose=self.verbose
        )

        # 5. Analytics - Performance monitoring
        self.analytics = AnalyticsCollector(
            session_id=self.session_id,
            max_latency_samples=1000,
            verbose=self.verbose
        )

        # 6. Error Message Enhancer - Better error messages
        self.error_enhancer = ErrorMessageEnhancer(verbose=self.verbose)

        # ===================================================================
        # INTELLIGENT INSTRUCTION SYSTEM
        # ===================================================================

        # Initialize intelligent instruction parser (uses LLM for semantic understanding)
        self.instruction_parser = IntelligentInstructionParser(
            llm_client=self.llm,
            verbose=self.verbose
        )

        # Use instruction memory from user_prefs (single source of truth with persistence)
        self.instruction_memory = self.user_prefs.instruction_memory

        # ===================================================================
        # EPISODIC MEMORY STORE
        # ===================================================================

        # ===================================================================
        # UNIFIED MEMORY SYSTEM v1.0
        # Consolidates: EpisodicMemory + SessionStore + IntelligentMemory
        # Features: Tiered injection, memory intent detection, consolidation
        # ===================================================================

        self.unified_memory = UnifiedMemory(
            llm_client=self.llm,
            storage_dir="data/unified_memory",
            verbose=self.verbose
        )
        self.unified_memory.start_session(self.session_id)

        # Core facts are loaded from unified_memory's own persistence
        # Don't copy from instruction_memory as it may have incorrectly parsed data

        # Track current turn data for episode storage
        self._current_turn_data = {
            'user_message': '',
            'agents_used': [],
            'responses': [],
            'intent': '',
            'entities': []
        }

        # ===================================================================

        # Register undo handlers
        self._register_undo_handlers()

        # ===================================================================

        # Load saved timezone - check unified memory first, then instruction memory
        saved_timezone = None

        # 1. Check unified memory core facts first (most reliable)
        if hasattr(self, 'unified_memory'):
            saved_timezone = self.unified_memory.get_core_fact('timezone')

        # 2. Fall back to instruction memory
        if not saved_timezone:
            saved_timezone = self.instruction_memory.get('timezone')

        # 3. Try to extract timezone from instruction values
        if not saved_timezone:
            for key, value in self.instruction_memory.get_all().items():
                # Check if value contains timezone abbreviations
                import re
                tz_match = re.search(r'\b(IST|EST|PST|CST|MST|UTC|GMT)\b', str(value).upper())
                if tz_match:
                    saved_timezone = tz_match.group(1)
                    break

        if saved_timezone:
            set_user_timezone(saved_timezone)
            # Also save to unified memory for future sessions
            if hasattr(self, 'unified_memory'):
                self.unified_memory.set_core_fact('timezone', saved_timezone, 'preference', 'loaded')
            if self.verbose:
                print(f"{C.CYAN}🕐 Loaded timezone from preferences: {saved_timezone}{C.ENDC}")
        else:
            set_user_timezone('IST')
            # Save default to unified memory
            if hasattr(self, 'unified_memory'):
                self.unified_memory.set_core_fact('timezone', 'IST', 'preference', 'default')
            if self.verbose:
                print(f"{C.CYAN}🕐 Default timezone: IST (Asia/Kolkata){C.ENDC}")

        # ===================================================================

        if self.verbose:
            print(f"{C.CYAN}🧠 Intelligence enabled: Session {self.session_id[:8]}...{C.ENDC}")
            print(f"{C.CYAN}📊 Advanced Intelligence: Intent, Entity, Task, Confidence, Context{C.ENDC}")
            print(f"{C.CYAN}📝 Logging to: {self.simple_logger.get_session_dir()}{C.ENDC}")
            print(f"{C.CYAN}🔄 Retry, 📊 Analytics, 🧠 Preferences, ↩️  Undo - All enabled{C.ENDC}")

        # Track current intent for agent usage recording
        self._current_intent_type = None

        self._base_system_prompt = """You are an AI orchestration system that coordinates specialized agents to help users accomplish complex tasks across multiple platforms and tools.

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

**Information Gathering Then Action**: For tasks requiring specific details:
1. First, gather all necessary information (search, list, query)
2. Present findings to user if needed
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

# Multi-Step Operation Reporting

When executing tasks that require multiple operations (e.g., "check calendar, create ticket, and notify channel"):

1. **Always report ALL results**: Don't focus only on failures - report successes too
2. **Structure your response clearly**:
   - Start with what was successfully completed
   - Then explain what failed and why
   - End with suggested next steps

**Example format for partial success**:
```
I've completed part of your request:

✅ **Completed:**
- Checked your calendar: You're free at 10am on Friday
- Created Jira task KAN-42: "Weekly Sync"

❌ **Could not complete:**
- Notify #general channel: This channel is not available

Would you like me to notify one of these available channels instead?
- #dev-opps
- #team-updates
```

**Never** report only the failure when other operations succeeded. Users need to know what was accomplished.

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

    def _build_dynamic_system_prompt(self, memory_context: str = "") -> str:
        """
        Build system prompt with dynamic user preferences and agent health status.

        This enhances the base prompt with:
        - Current datetime context
        - Episodic memory context
        - User communication style preferences
        - Agent health/availability status
        - Learned user patterns
        """
        prompt = self._base_system_prompt

        # Add current datetime context (important for scheduling, relative times)
        datetime_context = format_datetime_for_prompt()
        prompt += "\n\n" + datetime_context

        # Add unified memory context (core facts + sessions + episodic)
        if memory_context:
            prompt += "\n\n" + memory_context

        # Add explicit user instructions from intelligent instruction memory (highest priority)
        if hasattr(self, 'instruction_memory'):
            instruction_prompt = self.instruction_memory.format_for_prompt()
            if instruction_prompt:
                prompt += "\n\n" + instruction_prompt

        # Also add legacy user preferences instructions (for backward compatibility)
        explicit_instructions = self.user_prefs.get_instructions_for_prompt()
        if explicit_instructions:
            prompt += "\n\n" + explicit_instructions

        # Add user communication preferences
        comm_prefs = self.user_prefs.get_communication_preferences()
        if self.user_prefs.communication_style.confidence >= self.user_prefs.min_confidence_threshold:
            style_section = f"""

# User Communication Preferences (Learned)

Based on past interactions, this user prefers:
- Verbose explanations: {"Yes" if comm_prefs['verbose'] else "No, keep responses concise"}
- Technical details: {"Yes, include technical specifics" if comm_prefs['technical'] else "No, use simplified language"}
- Emojis in responses: {"Yes" if comm_prefs['emojis'] else "No, avoid emojis"}

Adjust your communication style accordingly."""
            prompt += style_section

        # Add agent health status
        unhealthy_agents = []
        for agent_name in self.sub_agents.keys():
            if hasattr(self, 'circuit_breaker'):
                health = self.circuit_breaker.get_health_status(agent_name)
                state = health.get('state', 'closed')
                if state and state != 'closed':
                    unhealthy_agents.append(f"{agent_name} ({state})")

        if unhealthy_agents:
            health_section = f"""

# Agent Availability Warning

The following agents are currently experiencing issues and may be unavailable:
{', '.join(unhealthy_agents)}

Prefer using healthy agents when possible. If a user specifically requests an unhealthy agent, explain that it may be temporarily unavailable."""
            prompt += health_section

        return prompt

    # =========================================================================
    # UNIFIED MEMORY METHODS
    # =========================================================================

    async def _retrieve_memory_context(self, user_message: str) -> str:
        """
        Retrieve relevant memory context using unified memory system.

        Uses tiered injection:
        - ALWAYS: Core facts (timezone, name, defaults)
        - SOMETIMES: Last session (if relevant)
        - ON-DEMAND: Full semantic search (for recall queries)

        Called at start of process_message(), before intelligence analysis.
        """
        try:
            context, memory_query = await self.unified_memory.get_context(user_message)

            # Log memory intent for debugging
            if self.verbose and memory_query.intent_type != MemoryIntentType.NONE:
                print(f"[MEMORY] Detected intent: {memory_query.intent_type.value}")

            return context

        except Exception as e:
            if self.verbose:
                print(f"[ORCHESTRATOR] Unified memory failed: {e}")
            return ""

    async def _store_in_memory(
        self,
        user_message: str,
        final_response: str,
        agents_used: List[str],
        intent_type: str
    ):
        """
        Store interaction in unified memory.

        Called at end of process_message() after generating response.
        """
        try:
            await self.unified_memory.add_message(
                user_message=user_message,
                response=final_response[:500],
                agents_used=agents_used,
                intent_type=intent_type
            )
        except Exception as e:
            if self.verbose:
                print(f"[ORCHESTRATOR] Memory storage failed: {e}")

    # =========================================================================
    # CORRECTION DETECTION AND LEARNING
    # =========================================================================

    async def _detect_and_handle_correction(self, user_message: str) -> bool:
        """
        Detect if user is correcting previous behavior.

        Patterns:
        - "No, use X instead"
        - "That's wrong, it should be Y"
        - "I said X, not Y"

        Returns True if correction detected and handled.
        """
        import re

        correction_patterns = [
            r"no,?\s+(use|do|make|set)\s+(.+)",
            r"that'?s\s+wrong",
            r"i\s+said\s+(.+),?\s+not\s+(.+)",
            r"don'?t\s+(use|do)\s+(.+)",
            r"actually,?\s+(use|do|make|set)\s+(.+)"
        ]

        message_lower = user_message.lower()

        for pattern in correction_patterns:
            match = re.search(pattern, message_lower)
            if match:
                # This is a correction - analyze with LLM
                correction_analysis = await self._analyze_correction(user_message)

                if correction_analysis:
                    key = correction_analysis.get('key')
                    new_value = correction_analysis.get('value')

                    if key and new_value:
                        self.user_prefs.flip_preference(
                            key=key,
                            new_value=new_value,
                            source="user_correction"
                        )

                        if self.verbose:
                            print(f"[ORCHESTRATOR] Learned from correction: {key} = {new_value}")

                        return True

        return False

    async def _analyze_correction(self, message: str) -> Optional[Dict]:
        """Use LLM to extract what preference is being corrected"""
        # Get recent context
        recent_context = ""
        if hasattr(self, 'context_manager'):
            recent_turns = self.context_manager.get_relevant_context(message)
            if recent_turns:
                recent_context = str(recent_turns)[:500]

        prompt = f"""The user is correcting previous behavior. Extract what setting they want changed.

Message: "{message}"

Recent context: {recent_context}

JSON response only:
{{"is_correction": true/false, "key": "preference_key", "value": "new_value", "confidence": 0.0-1.0}}

Examples:
- "No, use EST timezone" -> {{"is_correction": true, "key": "timezone", "value": "EST", "confidence": 0.95}}
- "Don't assign to me" -> {{"is_correction": true, "key": "default_assignee", "value": "none", "confidence": 0.9}}
"""

        try:
            response = await self.llm.generate(prompt)
            text = response.text if hasattr(response, 'text') else str(response)

            # Parse JSON
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except Exception as e:
            if self.verbose:
                print(f"[ORCHESTRATOR] Correction analysis failed: {e}")

        return None

    # =========================================================================
    # EXPLICIT INSTRUCTION DETECTION
    # =========================================================================

    def _detect_and_store_explicit_instruction(self, message: str) -> Optional[str]:
        """
        Detect and store explicit user instructions from a message.

        Patterns detected:
        - "from now on use X" / "from now on always X"
        - "always X" / "never X"
        - "remember that X" / "remember to X"
        - "my X is Y" / "my timezone is EST"
        - "default X is Y" / "set default X to Y"
        - "use X for Y"

        Args:
            message: User message to analyze

        Returns:
            Confirmation message if instruction was stored, None otherwise
        """
        import re

        message_lower = message.lower().strip()

        # Define patterns and their extractors
        patterns = [
            # Timezone patterns - enhanced for "change timezone to EST" style
            # Note: Use [a-zA-Z] to match both cases since re.IGNORECASE doesn't affect character classes
            (r'(?:change|set|switch)\s+(?:my\s+)?timezone\s+to\s+([a-zA-Z]{2,5})',
             'timezone', 'timezone'),
            (r'(?:from now on\s+)?(?:use|set|switch to)\s+([a-zA-Z]{2,5})\s*(?:time(?:zone)?)?',
             'timezone', 'timezone'),
            (r'my\s+timezone\s+is\s+([a-zA-Z]{2,5})',
             'timezone', 'timezone'),
            (r'(?:from now on\s+)?(?:use|set)\s+([a-zA-Z]{2,5})\s+(?:for\s+)?(?:all\s+)?times?',
             'timezone', 'timezone'),
            (r'timezone\s*[=:]\s*([a-zA-Z]{2,5})',
             'timezone', 'timezone'),

            # Default project patterns
            (r'(?:my\s+)?default\s+project\s+(?:is|should be)\s+([A-Z0-9_-]+)',
             'default', 'default_project'),
            (r'(?:from now on\s+)?use\s+([A-Z0-9_-]+)\s+(?:as\s+)?(?:the\s+)?default\s+project',
             'default', 'default_project'),
            (r'always\s+(?:use|create\s+(?:tickets?|issues?)\s+in)\s+([A-Z0-9_-]+)',
             'default', 'default_project'),

            # User name patterns
            (r'(?:my\s+name\s+is|i\s+am|call\s+me|i\'m)\s+([A-Za-z][A-Za-z\s]{1,30})',
             'preference', 'user_name'),

            # Default assignee patterns
            (r'(?:always\s+)?assign\s+(?:tickets?|issues?|tasks?)?\s*to\s+([A-Za-z0-9_@.-]+)',
             'default', 'default_assignee'),
            (r'default\s+assignee\s+(?:is|should be)\s+([A-Za-z0-9_@.-]+)',
             'default', 'default_assignee'),

            # Notification channel patterns
            (r'(?:send|post)\s+(?:all\s+)?notifications?\s+(?:to|in)\s+#?([A-Za-z0-9_-]+)',
             'default', 'notification_channel'),
            (r'default\s+(?:slack\s+)?channel\s+(?:is|should be)\s+#?([A-Za-z0-9_-]+)',
             'default', 'default_channel'),

            # Formatting preferences
            (r'(?:always\s+)?(?:use|prefer)\s+(bullet\s*points?|numbered\s*lists?|markdown)',
             'formatting', 'list_format'),
            (r'(?:be\s+)?(?:more\s+)?(concise|verbose|detailed|brief)',
             'formatting', 'verbosity'),

            # Behavior patterns
            (r'never\s+(.+?)(?:\s+unless|\s+except|\.|$)',
             'behavior', 'never'),
            (r'always\s+(.+?)(?:\s+when|\s+for|\.|$)',
             'behavior', 'always'),

            # Remember patterns (generic)
            (r'remember\s+(?:that\s+)?(?:my\s+)?([a-z_]+)\s+is\s+([A-Za-z0-9_@.-]+)',
             'preference', None),  # Special handling for key-value
        ]

        for pattern, category, key in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                # Extract value(s)
                groups = match.groups()

                if key is None and len(groups) >= 2:
                    # Handle "remember that X is Y" pattern
                    extracted_key = groups[0].strip().replace(' ', '_')
                    value = groups[1].strip()
                elif len(groups) >= 1:
                    extracted_key = key
                    value = groups[0].strip()
                else:
                    continue

                # Clean up value
                value = value.strip('.,;!?')

                # Special handling for certain categories
                if category == 'timezone':
                    value = value.upper()
                    # Also set the global timezone for datetime context
                    if set_user_timezone(value):
                        if self.verbose:
                            print(f"{C.CYAN}🕐 Global timezone set to: {value}{C.ENDC}")
                        # Store in unified memory as core fact
                        if hasattr(self, 'unified_memory'):
                            self.unified_memory.set_core_fact('timezone', value, 'preference', 'explicit')
                elif category == 'formatting' and extracted_key == 'verbosity':
                    # Normalize verbosity values
                    if value in ['concise', 'brief']:
                        value = 'concise'
                    elif value in ['verbose', 'detailed']:
                        value = 'verbose'

                # Store the instruction
                was_new = self.user_prefs.add_explicit_instruction(
                    instruction=message,
                    category=category,
                    key=extracted_key,
                    value=value
                )

                # Generate confirmation
                if was_new:
                    confirmation = f"Got it! I'll remember that {extracted_key} = {value}."
                else:
                    confirmation = f"Updated: {extracted_key} is now {value}."

                if self.verbose:
                    print(f"{C.CYAN}📝 Stored instruction: {extracted_key} = {value} ({category}){C.ENDC}")

                return confirmation

        return None

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
                    simple_logger=self.simple_logger
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
        if self.verbose:
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

            # Print buffered messages only in verbose mode
            if self.verbose:
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

                # Log agent initialization in orchestration logger
                if hasattr(self, 'orch_logger'):
                    self.orch_logger.log_agent_initialized(
                        agent_name=agent_name,
                        capabilities=capabilities,
                        metadata={'loaded_at': asyncio.get_event_loop().time()}
                    )
                    self.orch_logger.log_agent_ready(agent_name)

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

        # Summary (only in verbose mode)
        if self.verbose:
            print(f"\n{C.GREEN}✓ Loaded {successful} agent(s) successfully.{C.ENDC}")
            if failed > 0:
                print(f"{C.YELLOW}⚠ {failed} agent(s) failed to load but system will continue.{C.ENDC}")
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

        # Check circuit breaker before attempting execution
        allowed, reason = await self.circuit_breaker.can_execute(agent_name)
        if not allowed:
            error_msg = f"⚠️ {reason}"
            print(f"{C.YELLOW}{error_msg}{C.ENDC}")
            return error_msg

        # Legacy health check for initialization failures
        if agent_name in self.agent_health:
            health = self.agent_health[agent_name]
            if health['status'] == 'unavailable':
                error_msg = f"⚠️ {agent_name} agent is currently unavailable: {health.get('error_message', 'Unknown error')}"
                print(f"{C.YELLOW}{error_msg}{C.ENDC}")
                return error_msg

        # Create operation key for retry/analytics tracking
        operation_key = f"{agent_name}_{hashlib.md5(instruction[:100].encode()).hexdigest()[:8]}"

        # Generate task ID and log task assignment
        task_id = operation_key
        if hasattr(self, 'orch_logger'):
            self.orch_logger.log_task_assigned(
                task_id=task_id,
                task_name=instruction[:100],
                agent_name=agent_name,
                metadata={'operation_key': operation_key}
            )

        # SIMPLE LOGGING - Log agent selection
        available_agents = list(self.sub_agents.keys())
        self.simple_logger.log_agent_selection(
            selected_agent=agent_name,
            reason=f"Task requires {agent_name} capabilities",
            considered_agents=available_agents
        )

        # Define the operation to execute
        async def execute_operation():
            # Log task started
            if hasattr(self, 'orch_logger'):
                self.orch_logger.log_task_started(task_id)

            result = await self._execute_agent_direct(agent_name, instruction, context)

            # Log task completed
            if hasattr(self, 'orch_logger'):
                success = result and not result.startswith("⚠️") and not result.startswith("❌")
                self.orch_logger.log_task_completed(
                    task_id=task_id,
                    success=success,
                    error=result if not success else None
                )

            return result

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

            # Inject datetime context for all agents
            datetime_info = format_datetime_for_instruction()
            full_instruction = f"{datetime_info}\n\n{full_instruction}"

            if context_str:
                full_instruction = f"Context from previous steps:\n{context_str}\n\nTask: {full_instruction}"

            # SIMPLE LOGGING - Log orchestrator -> agent
            self.simple_logger.log_orchestrator_to_agent(agent_name, full_instruction)

            # Execute the agent
            result = await agent.execute(full_instruction)

            # Handle None result (shouldn't happen but defensive programming)
            if result is None:
                result = f"❌ {agent_name} agent returned None - this is a bug in the agent implementation"

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Determine success (safely handle potential None)
            success = (result and
                      not result.startswith("⚠️") and
                      not result.startswith("❌") and
                      not result.startswith("Error"))
            error = result if not success else None

            # Record analytics
            self.analytics.record_agent_call(
                agent_name=agent_name,
                success=success,
                latency_ms=latency_ms,
                error_message=error
            )

            # SIMPLE LOGGING - Log agent -> orchestrator
            self.simple_logger.log_agent_to_orchestrator(
                agent_name=agent_name,
                response=result,
                success=success,
                error=error,
                duration_ms=latency_ms
            )

            # Update health status on success
            if success and agent_name in self.agent_health:
                self.agent_health[agent_name]['status'] = 'healthy'
                self.agent_health[agent_name]['last_success'] = asyncio.get_event_loop().time()
                self.agent_health[agent_name]['error_count'] = 0

            # Record to circuit breaker
            if success:
                await self.circuit_breaker.record_success(agent_name)

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

            # Record failure to circuit breaker (replaces old "degraded" logic)
            await self.circuit_breaker.record_failure(agent_name, e)

            # Re-raise enhanced error for retry manager
            raise RuntimeError(enhanced_msg)

    async def _process_with_intelligence(self, user_message: str) -> Dict:
        """
        Process user message with advanced Hybrid Intelligence System v5.0

        Uses two-tier approach:
        - Tier 1: Fast keyword filter (~10ms, free)
        - Tier 2: LLM semantic analysis (~200ms, with caching)

        Returns intelligence analysis including intents, entities, confidence, etc.
        """
        # Log start of intelligence processing
        if hasattr(self, 'intel_logger'):
            start_time = self.intel_logger.log_message_processing_start(user_message)
        else:
            import time
            start_time = time.time()

        # SIMPLE LOGGING - Log intelligence processing start
        self.simple_logger.log_intelligence_start(user_message)

        # Get conversation context for better understanding
        context_dict = self.context_manager.get_relevant_context(user_message)

        # 1. HYBRID INTELLIGENCE: Classify intent + Extract entities (unified)
        # This replaces separate intent_classifier and entity_extractor calls
        hybrid_result: HybridIntelligenceResult = await self.hybrid_intelligence.classify_intent(
            message=user_message,
            context=context_dict
        )

        # Extract components from hybrid result
        intents = hybrid_result.intents
        entities = hybrid_result.entities

        # Log intent classification with hybrid intelligence metadata
        if hasattr(self, 'intel_logger'):
            intent_names = [str(i) for i in intents]
            confidence_scores = {str(i): getattr(i, 'confidence', 0.8) for i in intents}
            self.intel_logger.log_intent_classification(
                message=user_message,
                detected_intents=intent_names[:5],
                confidence_scores=confidence_scores,
                classification_method=f"hybrid_{hybrid_result.path_used}",  # 'hybrid_fast' or 'hybrid_llm'
                duration_ms=hybrid_result.latency_ms,
                cache_hit=(hybrid_result.path_used == 'fast')  # Fast path is effectively cached
            )

        # SIMPLE LOGGING - Log intent classification
        self.simple_logger.log_intent_classification(
            intents=[str(i) for i in intents],
            confidence=hybrid_result.confidence,
            method=f"hybrid_{hybrid_result.path_used}",
            reasoning=hybrid_result.reasoning
        )

        # Log entity extraction with hybrid intelligence metadata
        if hasattr(self, 'intel_logger') and entities:
            entity_dict = {}
            for ent in entities:
                ent_type = getattr(ent, 'type', 'unknown')
                ent_value = getattr(ent, 'value', str(ent))
                if ent_type not in entity_dict:
                    entity_dict[ent_type] = []
                entity_dict[ent_type].append(ent_value)

            self.intel_logger.log_entity_extraction(
                message=user_message,
                extracted_entities=entity_dict,
                entity_relationships=[],
                confidence=hybrid_result.confidence,  # Use hybrid result confidence
                duration_ms=hybrid_result.latency_ms,  # Already included in classification time
                cache_hit=(hybrid_result.path_used == 'fast')
            )

        # SIMPLE LOGGING - Log entity extraction
        if entities:
            entity_list = []
            for ent in entities:
                entity_list.append({
                    'type': str(getattr(ent, 'type', 'unknown').value) if hasattr(getattr(ent, 'type', None), 'value') else str(getattr(ent, 'type', 'unknown')),
                    'value': str(getattr(ent, 'value', str(ent))),
                    'confidence': float(getattr(ent, 'confidence', 0.0))
                })
            self.simple_logger.log_entity_extraction(
                entities=entity_list,
                confidence=hybrid_result.confidence
            )

        # 2. Score confidence (combines hybrid result with other factors)
        confidence = self.confidence_scorer.score_overall(
            message=user_message,
            intents=intents,
            entities=entities
        )

        # Log confidence scoring
        if hasattr(self, 'intel_logger'):
            # Extract numeric score from Confidence object
            confidence_value = confidence.score if hasattr(confidence, 'score') else 0.5
            self.intel_logger.log_confidence_score(
                overall_confidence=confidence_value,
                component_scores={
                    'intent': 0.85,
                    'entity': 0.80,
                    'context': 0.90
                },
                factors={'message_length': len(user_message)},
                duration_ms=1.0
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

        # 6. Build intelligence summary with hybrid intelligence metadata
        intelligence = {
            'intents': intents,
            'entities': entities,
            'confidence': confidence,
            'resolved_message': resolved_message,
            'context': context_dict,
            'primary_intent': intents[0] if intents else None,  # Get first intent
            'action_recommendation': self.confidence_scorer.get_action_recommendation(confidence),

            # Hybrid Intelligence v5.0 metadata
            'hybrid_path_used': hybrid_result.path_used,  # 'fast' or 'llm'
            'hybrid_latency_ms': hybrid_result.latency_ms,
            'hybrid_reasoning': hybrid_result.reasoning,
            'hybrid_confidence': hybrid_result.confidence,
            'ambiguities': hybrid_result.ambiguities,
            'suggested_clarifications': hybrid_result.suggested_clarifications
        }

        # 7. Log intelligence insights with hybrid system stats
        if self.verbose:
            print(f"\n{C.CYAN}🧠 Hybrid Intelligence Analysis v5.0:{C.ENDC}")
            print(f"  Path: {C.BOLD}{hybrid_result.path_used.upper()}{C.ENDC} ({hybrid_result.latency_ms:.1f}ms)")
            print(f"  Intents: {[str(i) for i in intents[:3]]}")
            print(f"  Entities: {len(entities)} found")
            print(f"  Confidence: {confidence}")
            print(f"  Reasoning: {hybrid_result.reasoning[:80]}...")
            print(f"  Recommendation: {intelligence['action_recommendation'][0]}")
            if hybrid_result.ambiguities:
                print(f"  {C.YELLOW}Ambiguities: {', '.join(hybrid_result.ambiguities[:2])}{C.ENDC}")

        # SIMPLE LOGGING - Log decision based on action recommendation
        action_rec, explanation = intelligence['action_recommendation']
        self.simple_logger.log_decision(
            decision_type=action_rec,
            action=f"Process with {hybrid_result.path_used} intelligence path",
            reasoning=explanation
        )

        # SIMPLE LOGGING - Log intelligence processing complete
        import time
        total_duration = (time.time() - start_time) * 1000
        confidence_score = confidence.score if hasattr(confidence, 'score') else 0.5
        self.simple_logger.log_intelligence_complete(
            total_duration_ms=total_duration,
            success=(confidence_score > 0.3)
        )

        return intelligence

    def _log_and_return_response(self, response: str) -> str:
        """Helper method to log assistant response and return it"""
        # SIMPLE LOGGING - Log orchestrator response
        self.simple_logger.log_orchestrator_response(response)
        return response

    async def process_message(self, user_message: str) -> str:
        """Process a user message with orchestration"""

        # SIMPLE SESSION LOGGING - Log user message
        self.simple_logger.log_user_message(user_message)

        # Reset per-message token tracking
        self.last_message_tokens = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

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
        # CORRECTION DETECTION
        # ===================================================================

        # Check for corrections first (before other processing)
        is_correction = await self._detect_and_handle_correction(user_message)

        # ===================================================================
        # EPISODIC MEMORY RETRIEVAL
        # ===================================================================

        # Retrieve relevant episodic memory context
        memory_context = await self._retrieve_memory_context(user_message)

        # ===================================================================
        # INTELLIGENT INSTRUCTION DETECTION
        # ===================================================================

        # Use intelligent instruction parser (LLM-powered semantic understanding)
        instruction_confirmation = None
        try:
            parsed_instruction = await self.instruction_parser.parse(user_message)

            if parsed_instruction.is_instruction and parsed_instruction.confidence >= 0.5:
                # Apply specific instruction types
                if parsed_instruction.category == 'timezone' and parsed_instruction.value:
                    # Force consistent key for timezone instructions
                    parsed_instruction.key = 'timezone'
                    if set_user_timezone(parsed_instruction.value):
                        if self.verbose:
                            print(f"{C.CYAN}🕐 Global timezone set to: {parsed_instruction.value}{C.ENDC}")
                        # Store in unified memory as core fact
                        if hasattr(self, 'unified_memory'):
                            self.unified_memory.set_core_fact('timezone', parsed_instruction.value, 'preference', 'explicit')
                        # Clear confirmation for timezone - will be shown via context
                        instruction_confirmation = f"🕐 Timezone set to {parsed_instruction.value}"
                        self.instruction_memory.add(parsed_instruction)
                elif parsed_instruction.category == 'identity' and parsed_instruction.value:
                    # Store identity information (name, role, etc.)
                    if hasattr(self, 'unified_memory'):
                        key = parsed_instruction.key or 'user_name'
                        self.unified_memory.set_core_fact(key, parsed_instruction.value, 'identity', 'explicit')
                        instruction_confirmation = f"👤 {key.replace('_', ' ').title()} set to {parsed_instruction.value}"
                        if self.verbose:
                            print(f"{C.CYAN}👤 Identity saved: {key} = {parsed_instruction.value}{C.ENDC}")
                    self.instruction_memory.add(parsed_instruction)
                elif parsed_instruction.category == 'default' and parsed_instruction.key and parsed_instruction.value:
                    # Store default values (project, assignee, etc.)
                    if hasattr(self, 'unified_memory'):
                        self.unified_memory.set_core_fact(parsed_instruction.key, parsed_instruction.value, 'default', 'explicit')
                        if self.verbose:
                            print(f"{C.CYAN}📋 Default saved: {parsed_instruction.key} = {parsed_instruction.value}{C.ENDC}")
                    self.instruction_memory.add(parsed_instruction)
                else:
                    # Store in instruction memory for other instructions
                    self.instruction_memory.add(parsed_instruction)

                    # Generate confirmation for other instruction types
                    instruction_confirmation = (
                        f"✓ Noted: {parsed_instruction.key} = {parsed_instruction.value}"
                    )

                if self.verbose:
                    print(f"{C.GREEN}{instruction_confirmation}{C.ENDC}")

        except Exception as e:
            logger.warning(f"Instruction parsing error: {e}")
            # Fall back to old regex method if intelligent parsing fails
            instruction_confirmation = self._detect_and_store_explicit_instruction(user_message)

        # Pattern-based fallback detection for miscategorized instructions
        # This catches cases where the parser returns wrong category (e.g., 'default' instead of 'timezone')
        import re

        # Fallback timezone detection - always check for timezone patterns
        # This catches cases where instruction parser miscategorizes timezone updates
        if hasattr(self, 'unified_memory'):
            # Look for timezone patterns in user message (always check, even if timezone exists)
            tz_patterns = [
                r'\b(?:use|set|switch to|change to)\s+(\w+)\s+(?:timezone|time\s*zone|from now|going forward)',
                r'\b(IST|EST|PST|CST|MST|UTC|GMT|PDT|EDT|CDT|MDT)\b.*(?:from now|timezone|time)',
                r'(?:timezone|time\s*zone).*\b(IST|EST|PST|CST|MST|UTC|GMT|PDT|EDT|CDT|MDT)\b',
            ]
            for pattern in tz_patterns:
                match = re.search(pattern, user_message, re.IGNORECASE)
                if match:
                    detected_tz = match.group(1).upper()
                    if detected_tz in ['IST', 'EST', 'PST', 'CST', 'MST', 'UTC', 'GMT', 'PDT', 'EDT', 'CDT', 'MDT']:
                        current_tz = self.unified_memory.get_core_fact('timezone')
                        # Only update if different from current or no current set
                        if not current_tz or current_tz != detected_tz:
                            if set_user_timezone(detected_tz):
                                self.unified_memory.set_core_fact('timezone', detected_tz, 'preference', 'pattern_fallback')
                                if self.verbose:
                                    print(f"{C.CYAN}🕐 Timezone detected (fallback): {detected_tz}{C.ENDC}")
                                if not instruction_confirmation:
                                    instruction_confirmation = f"🕐 Timezone set to {detected_tz}"
                        break

        # Fallback name/identity detection
        if hasattr(self, 'unified_memory'):
            current_name = self.unified_memory.get_core_fact('user_name')
            if not current_name:
                # Look for name patterns in user message
                name_patterns = [
                    r"(?:my name is|i'm|i am|call me|name's)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
                    r"(?:this is|it's)\s+([A-Z][a-z]+)(?:\s+here)?",
                ]
                for pattern in name_patterns:
                    match = re.search(pattern, user_message, re.IGNORECASE)
                    if match:
                        detected_name = match.group(1).strip().title()
                        if len(detected_name) >= 2 and len(detected_name) <= 50:
                            self.unified_memory.set_core_fact('user_name', detected_name, 'identity', 'pattern_fallback')
                            if self.verbose:
                                print(f"{C.CYAN}👤 Name detected (fallback): {detected_name}{C.ENDC}")
                            if not instruction_confirmation:
                                instruction_confirmation = f"👤 Name set to {detected_name}"
                            break

        # ===================================================================

        # Initialize on first message
        if not self.chat:
            # Only discover agents if not already loaded (main.py may have loaded them)
            if not self.sub_agents:
                discover_task = asyncio.create_task(self.discover_and_load_agents())
                await self._spinner(discover_task, "Discovering agents")

            if not self.sub_agents:
                return self._log_and_return_response("No agents available. Please add agent connectors to the 'connectors' directory.")

            # Create model with agent tools using LLM abstraction
            agent_tools = self._create_agent_tools()

            # Set the system instruction on the LLM config (with dynamic preferences and memory)
            self.llm.config.system_instruction = self._build_dynamic_system_prompt(memory_context=memory_context)

            # Set tools on the LLM (for Gemini, these are FunctionDeclarations)
            self.llm.set_tools(agent_tools)

            # Start chat session with function calling enabled
            self.chat = self.llm.start_chat(
                history=None,
                enable_function_calling=True
            )

        # Reset operation counter for this request
        self.operation_count = 0

        # Reset current turn data for episode tracking
        self._current_turn_data = {
            'user_message': user_message,
            'agents_used': [],
            'responses': [],
            'intent': '',
            'entities': []
        }

        # Process with Hybrid Intelligence System v5.0 (async)
        intelligence = await self._process_with_intelligence(user_message)

        # Update turn data with intelligence results
        self._current_turn_data['intent'] = self._current_intent_type if hasattr(self, '_current_intent_type') else ''
        self._current_turn_data['entities'] = [
            {'type': str(getattr(e, 'type', 'unknown').value) if hasattr(getattr(e, 'type', None), 'value') else str(getattr(e, 'type', 'unknown')),
             'value': str(getattr(e, 'value', str(e)))}
            for e in intelligence.get('entities', [])
        ]

        # Store current intent for agent usage tracking
        primary_intent = intelligence.get('primary_intent')
        if primary_intent and hasattr(primary_intent, 'type'):
            self._current_intent_type = str(primary_intent.type.value) if hasattr(primary_intent.type, 'value') else str(primary_intent.type)
        else:
            self._current_intent_type = 'unknown'

        # Use resolved message if references were resolved
        message_to_send = intelligence.get('resolved_message', user_message)

        # Add agent preference hints based on learned patterns
        if self._current_intent_type and self._current_intent_type != 'unknown':
            preferred_agent = self.user_prefs.get_preferred_agent(self._current_intent_type)
            if preferred_agent:
                message_to_send += f"\n\n[User Preference: For {self._current_intent_type} tasks, this user typically prefers using the {preferred_agent} agent]"
                if self.verbose:
                    print(f"{C.CYAN}📊 User prefers {preferred_agent} for {self._current_intent_type} tasks{C.ENDC}")

        # Add explicit instructions to message for immediate application
        # This ensures instructions apply to the current message even if stored mid-session
        active_instructions = self.user_prefs.get_explicit_instructions()
        if active_instructions:
            instruction_hints = []
            for inst in active_instructions:
                instruction_hints.append(f"{inst.key}: {inst.value}")
            if instruction_hints:
                message_to_send += f"\n\n[User's Explicit Instructions: {'; '.join(instruction_hints)}]"
                if self.verbose:
                    print(f"{C.CYAN}📝 Applying {len(active_instructions)} explicit instruction(s){C.ENDC}")

        # Add current datetime to each message (ensures time is always accurate)
        # This overrides the stale datetime in system prompt
        current_time_context = format_datetime_for_instruction()
        message_to_send += f"\n\n{current_time_context}"

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
                return self._log_and_return_response("I need more information to proceed. " + clarifications[0])

        # Log intelligence insights
        if self.verbose:
            print(f"{C.CYAN}📊 Using intelligence: {explanation}{C.ENDC}")

        # Create and run the initial send task with a spinner
        send_task = asyncio.create_task(self.chat.send_message(message_to_send))
        await self._spinner(send_task, "Thinking")
        llm_response = send_task.result()

        # Track token usage
        self._track_tokens(llm_response)

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

                # Track agent for episode storage
                if agent_name not in self._current_turn_data['agents_used']:
                    self._current_turn_data['agents_used'].append(agent_name)

                # Feature #21: Track successful operation in duplicate detector
                self.duplicate_detector.track_operation(agent_name, instruction, "", success=True)

                # Record agent usage for preference learning
                if self._current_intent_type and self._current_intent_type != 'unknown':
                    self.user_prefs.record_agent_usage(
                        task_pattern=self._current_intent_type,
                        agent_used=agent_name,
                        was_successful=True
                    )
                    if self.verbose:
                        print(f"{C.CYAN}📊 Recorded: {agent_name} used for {self._current_intent_type}{C.ENDC}")

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

            # Track token usage
            self._track_tokens(llm_response)

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
        final_response = None
        if llm_response and llm_response.text:
            final_response = llm_response.text
        elif response and hasattr(response, 'candidates') and response.candidates:
            try:
                # Try to get text property (may fail if there are function_call parts)
                final_response = response.text
            except Exception:
                # Manual extraction from parts
                text_parts = []
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)

                if text_parts:
                    final_response = '\n'.join(text_parts)

        # Default response if nothing found
        if not final_response:
            final_response = "Task completed but response formatting failed. The operations were executed successfully."

        # Prepend instruction confirmation if we stored a new instruction
        if instruction_confirmation:
            final_response = f"{instruction_confirmation}\n\n{final_response}"

        # ===================================================================
        # EPISODIC MEMORY STORAGE
        # ===================================================================

        # Store this interaction as episodic memory
        # ===================================================================
        # UNIFIED MEMORY STORAGE
        # ===================================================================

        # Store interaction in unified memory (handles everything)
        await self._store_in_memory(
            user_message=user_message,
            final_response=final_response,
            agents_used=self._current_turn_data['agents_used'],
            intent_type=self._current_turn_data.get('intent', 'unknown')
        )

        # Reinforce preferences if this wasn't a correction (implicit acceptance)
        if not is_correction:
            # Get recently used preference keys and reinforce them
            for key in list(self.user_prefs.instruction_memory.instructions.keys())[:5]:
                self.user_prefs.reinforce_preference(key, boost=0.05)

        return self._log_and_return_response(final_response)

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

                # Display analytics summary (no separate folder - using session logging)
                if self.verbose:
                    print(f"{C.CYAN}    {self.analytics.generate_summary_report()}{C.ENDC}")
            except Exception as e:
                logger.warning(f"Failed to process analytics: {e}")

        # Display retry statistics
        if hasattr(self, 'retry_manager') and self.verbose:
            stats = self.retry_manager.get_statistics()
            if stats['total_operations'] > 0:
                print(f"{C.CYAN}  📊 Retry stats: {stats['total_operations']} ops, "
                      f"{stats['avg_retries_per_operation']:.1f} avg retries{C.ENDC}")

        # Display circuit breaker statistics
        if hasattr(self, 'circuit_breaker'):
            cb_stats = self.circuit_breaker.get_statistics()
            if cb_stats['total_circuits'] > 0:
                if self.verbose or cb_stats['open_circuits'] > 0:
                    # Always show if there are open circuits (important!)
                    print(f"{C.CYAN}  🔌 Circuit Breaker: {cb_stats['closed_circuits']}/{cb_stats['total_circuits']} healthy{C.ENDC}")
                    if cb_stats['open_circuits'] > 0:
                        print(f"{C.RED}     ⚠️  {cb_stats['open_circuits']} circuit(s) OPEN (failing agents){C.ENDC}")
                    if cb_stats['total_blocked_requests'] > 0:
                        print(f"{C.YELLOW}     Blocked {cb_stats['total_blocked_requests']} requests (fail-fast){C.ENDC}")

        # Display undo statistics
        if hasattr(self, 'undo_manager') and self.verbose:
            undo_stats = self.undo_manager.get_statistics()
            if undo_stats['total_operations'] > 0:
                print(f"{C.CYAN}  ↩️  Undo: {undo_stats['available_for_undo']} operations available{C.ENDC}")

        # Display instruction memory statistics
        if hasattr(self, 'instruction_memory') and self.verbose:
            active_instructions = self.instruction_memory.get_all()
            if active_instructions:
                print(f"{C.CYAN}  📝 Instructions: {len(active_instructions)} active{C.ENDC}")
                for key, value in list(active_instructions.items())[:3]:
                    print(f"{C.CYAN}     • {key}: {value}{C.ENDC}")

        # Display instruction parser statistics
        if hasattr(self, 'instruction_parser') and self.verbose:
            stats = self.instruction_parser.get_statistics()
            if stats['total_parses'] > 0:
                print(f"{C.CYAN}  🔍 Instruction Parser: {stats['instructions_found']}/{stats['total_parses']} detected{C.ENDC}")
                print(f"{C.CYAN}     Fast path: {stats['fast_path_rate']} | LLM calls: {stats['llm_call_rate']} | Cache: {stats['cache_hit_rate']}{C.ENDC}")

        # End unified memory session and display statistics
        if hasattr(self, 'unified_memory'):
            try:
                await self.unified_memory.end_session()
                if self.verbose:
                    stats = self.unified_memory.get_statistics()
                    print(f"{C.CYAN}  🧠 Unified Memory Statistics:{C.ENDC}")
                    print(f"{C.CYAN}     Core facts: {stats['core_facts']} | Consolidated: {stats['consolidated_facts']}{C.ENDC}")
                    print(f"{C.CYAN}     Sessions: {stats['sessions']} | Entities: {stats['entities']}{C.ENDC}")
                    print(f"{C.CYAN}     Episodes: {stats['episodes']} | Queries: {stats['queries']}{C.ENDC}")
                    if stats['consolidations'] > 0:
                        print(f"{C.GREEN}     ✓ {stats['consolidations']} consolidations performed{C.ENDC}")
            except Exception as e:
                logger.warning(f"Failed to end unified memory session: {e}")

        # Display Hybrid Intelligence statistics
        if hasattr(self, 'hybrid_intelligence') and self.verbose:
            try:
                stats = self.hybrid_intelligence.get_statistics()
                if stats['total_requests'] > 0:
                    print(f"\n{C.BOLD}{C.CYAN}🧠 Hybrid Intelligence System v5.0 - Session Statistics{C.ENDC}")
                    print(f"{C.CYAN}  Total Requests: {stats['total_requests']}{C.ENDC}")
                    print(f"{C.CYAN}  Fast Path: {stats['fast_path_count']} ({stats['fast_path_rate']}){C.ENDC}")
                    print(f"{C.CYAN}  LLM Path: {stats['llm_path_count']} ({stats['llm_path_rate']}){C.ENDC}")
                    print(f"{C.CYAN}  Avg Latency: {stats['avg_latency_ms']}ms{C.ENDC}")
                    print(f"{C.GREEN}  ✓ Performance target: 92% accuracy, 80ms latency{C.ENDC}")
            except Exception as e:
                logger.warning(f"Failed to display hybrid intelligence stats: {e}")

        # ===================================================================

        # Export observability data
        if hasattr(self, 'observability'):
            try:
                self.observability.export_all()
                self.observability.cleanup()
                if self.verbose:
                    print(f"{C.GREEN}  ✓ Observability data exported{C.ENDC}")
            except Exception as e:
                logger.warning(f"Failed to export observability data: {e}")

        # Close simple session logger
        if hasattr(self, 'simple_logger'):
            self.simple_logger.close()
            if self.verbose:
                print(f"{C.GREEN}  ✓ Simple session logs saved to: {self.simple_logger.get_session_dir()}{C.ENDC}")
                print(f"{C.GREEN}    - conversations.txt{C.ENDC}")
                print(f"{C.GREEN}    - intelligence.txt{C.ENDC}")
    
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