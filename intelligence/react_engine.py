"""
ReAct Engine - Reasoning and Action Loop Implementation

Implements the ReAct (Reasoning + Acting) pattern for agentic AI:
- Interleaved reasoning and action
- Observation feedback integration
- Adaptive execution based on intermediate results
- Self-correction capabilities

Reference: "ReAct: Synergizing Reasoning and Acting in Language Models"

Author: AI System
Version: 1.0
"""

import asyncio
import time
import json
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime

import google.generativeai as genai

from .react_types import (
    Thought, ThoughtType, Action, ActionType, Observation, ObservationType,
    ReActStep, ReActTrace, ExecutionContext, ExecutionResult
)


class ReActEngine:
    """
    ReAct Engine - Core reasoning-action-observation loop

    This engine implements the ReAct pattern where the agent:
    1. THINKS about the current situation
    2. ACTS based on that thinking
    3. OBSERVES the result
    4. Repeats until task is complete or max iterations reached

    Key features:
    - LLM-based reasoning with chain-of-thought
    - Adaptive execution based on observations
    - Error recovery and self-correction
    - Early termination when task is complete
    """

    def __init__(
        self,
        llm_model: str = 'models/gemini-2.5-flash',
        verbose: bool = False
    ):
        """
        Initialize ReAct Engine

        Args:
            llm_model: Gemini model to use for reasoning
            verbose: Enable verbose logging
        """
        self.llm_model = llm_model
        self.verbose = verbose

        # Action executors (registered by agents)
        self.action_executors: Dict[str, Callable] = {}

        # Statistics
        self.total_executions = 0
        self.total_iterations = 0
        self.successful_executions = 0

    def register_action_executor(self, action_name: str, executor: Callable):
        """
        Register an action executor

        Args:
            action_name: Name of the action
            executor: Async function to execute the action
        """
        self.action_executors[action_name] = executor

    async def execute(
        self,
        context: ExecutionContext,
        initial_observation: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute ReAct loop for a task

        Args:
            context: Execution context with task, goal, constraints
            initial_observation: Optional initial observation to start with

        Returns:
            ExecutionResult with trace and final result
        """
        start_time = time.time()
        self.total_executions += 1

        # Initialize trace
        trace = ReActTrace(start_time=datetime.now())

        # Build system prompt for ReAct
        system_prompt = self._build_system_prompt(context)

        # Create chat session
        model = genai.GenerativeModel(
            self.llm_model,
            system_instruction=system_prompt
        )
        chat = model.start_chat()

        # Initialize conversation with task
        conversation_history = []
        if initial_observation:
            conversation_history.append(f"Initial observation: {initial_observation}")

        # Main ReAct loop
        iteration = 0
        final_result = None
        terminated_reason = ""

        while iteration < context.max_iterations:
            iteration += 1
            self.total_iterations += 1

            if self.verbose:
                print(f"\n[REACT] Iteration {iteration}/{context.max_iterations}")

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > context.timeout_seconds:
                terminated_reason = "timeout"
                break

            # THINK: Generate reasoning
            thought = await self._think(chat, context, conversation_history, trace)

            if self.verbose:
                print(f"[REACT] Thought: {thought.content[:200]}...")

            # Check if we should terminate
            if self._should_terminate(thought):
                terminated_reason = "task_complete"
                final_result = self._extract_final_answer(thought)
                break

            # ACT: Decide on action
            action = await self._decide_action(chat, thought, context)

            if self.verbose:
                print(f"[REACT] Action: {action.name} ({action.type.value})")

            # Check for termination action
            if action.type == ActionType.TERMINATE:
                terminated_reason = "explicit_termination"
                final_result = action.parameters.get('result', '')
                break

            # Execute action and get observation
            observation = await self._execute_action(action, context)

            if self.verbose:
                print(f"[REACT] Observation: {observation.content[:200]}...")

            # Record step
            step = ReActStep(
                step_number=iteration,
                thought=thought,
                action=action,
                observation=observation
            )
            trace.add_step(step)

            # Update conversation history
            conversation_history.append(f"Thought: {thought.content}")
            conversation_history.append(f"Action: {action.name}({json.dumps(action.parameters)})")
            conversation_history.append(f"Observation: {observation.content}")

            # Check if observation indicates completion
            if observation.type == ObservationType.SUCCESS and self._is_task_complete(observation, context):
                terminated_reason = "task_complete"
                final_result = observation.content
                break

            # Check if we need to handle errors
            if observation.type in [ObservationType.ERROR, ObservationType.FAILURE]:
                # Let the model reason about the error in next iteration
                pass

        # Finalize trace
        trace.end_time = datetime.now()
        trace.final_result = final_result
        trace.terminated_reason = terminated_reason or "max_iterations"

        # Determine success
        success = terminated_reason in ["task_complete", "explicit_termination"]
        if success:
            self.successful_executions += 1

        total_duration = time.time() - start_time

        return ExecutionResult(
            success=success,
            result=final_result or "Task incomplete",
            trace=trace,
            total_iterations=iteration,
            total_duration_seconds=total_duration
        )

    def _build_system_prompt(self, context: ExecutionContext) -> str:
        """Build system prompt for ReAct reasoning"""
        tools_desc = ", ".join(context.available_tools) if context.available_tools else "various tools"

        return f"""You are an AI agent using the ReAct (Reasoning + Acting) framework.

TASK: {context.task}
GOAL: {context.goal}
AVAILABLE TOOLS: {tools_desc}
CONSTRAINTS: {', '.join(context.constraints) if context.constraints else 'None'}

For each step, you must:
1. THINK: Analyze the current situation and reason about what to do next
2. ACT: Choose an action to take
3. Then you will receive an OBSERVATION with the result

FORMAT YOUR RESPONSES AS:

**Thought**: [Your reasoning about the current situation and what to do]

**Action**: [action_name]
**Parameters**: {{"param1": "value1", "param2": "value2"}}

Available actions:
- tool_call: Call an external tool (specify tool name in parameters)
- respond: Provide final response to user
- clarify: Ask user for clarification
- terminate: End the task (use when complete)

IMPORTANT:
- Think step by step before acting
- If an action fails, analyze why and try a different approach
- When the task is complete, use the 'terminate' action with the result
- Be concise but thorough in your reasoning
"""

    async def _think(
        self,
        chat: Any,
        context: ExecutionContext,
        history: List[str],
        trace: ReActTrace
    ) -> Thought:
        """
        Generate a reasoning thought

        Args:
            chat: Active chat session
            context: Execution context
            history: Conversation history
            trace: Current trace

        Returns:
            Thought object with reasoning
        """
        # Build prompt for thinking
        if history:
            history_text = "\n".join(history[-6:])  # Last 3 steps (thought/action/observation each)
            prompt = f"""Previous steps:
{history_text}

Based on the above, what should we do next? Provide your thought process."""
        else:
            prompt = f"""Task: {context.task}
Goal: {context.goal}

Let's begin. What's your first thought about how to approach this task?"""

        # Get LLM response
        response = await chat.send_message_async(prompt)
        response_text = response.text if hasattr(response, 'text') else str(response)

        # Parse thought from response
        thought_content = self._extract_thought(response_text)

        # Determine thought type
        thought_type = self._classify_thought_type(thought_content)

        return Thought(
            type=thought_type,
            content=thought_content,
            confidence=0.8,  # Default confidence
            timestamp=datetime.now()
        )

    async def _decide_action(
        self,
        chat: Any,
        thought: Thought,
        context: ExecutionContext
    ) -> Action:
        """
        Decide on an action based on the thought

        Args:
            chat: Active chat session
            thought: Current thought
            context: Execution context

        Returns:
            Action to execute
        """
        # Ask LLM to decide on action
        prompt = f"""Based on your thought: "{thought.content}"

What action should we take? Respond with:
**Action**: [action_name]
**Parameters**: {{"key": "value"}}
**Expected Outcome**: [what you expect to happen]

Remember: Use 'terminate' when the task is complete."""

        response = await chat.send_message_async(prompt)
        response_text = response.text if hasattr(response, 'text') else str(response)

        # Parse action from response
        action_name, parameters, expected = self._extract_action(response_text)

        # Determine action type
        action_type = self._classify_action_type(action_name)

        return Action(
            type=action_type,
            name=action_name,
            parameters=parameters,
            reasoning=thought.content,
            expected_outcome=expected,
            confidence=thought.confidence,
            timestamp=datetime.now()
        )

    async def _execute_action(
        self,
        action: Action,
        context: ExecutionContext
    ) -> Observation:
        """
        Execute an action and return observation

        Args:
            action: Action to execute
            context: Execution context

        Returns:
            Observation with result
        """
        try:
            # Check if we have an executor for this action
            if action.name in self.action_executors:
                executor = self.action_executors[action.name]
                result = await executor(action.parameters)

                return Observation(
                    type=ObservationType.SUCCESS,
                    content=str(result),
                    raw_result=result,
                    matches_expectation=True,
                    timestamp=datetime.now()
                )

            # Handle built-in actions
            if action.type == ActionType.TOOL_CALL:
                tool_name = action.parameters.get('tool', action.name)
                if tool_name in self.action_executors:
                    executor = self.action_executors[tool_name]
                    result = await executor(action.parameters)
                    return Observation(
                        type=ObservationType.SUCCESS,
                        content=str(result),
                        raw_result=result,
                        matches_expectation=True
                    )
                else:
                    return Observation(
                        type=ObservationType.ERROR,
                        content=f"Unknown tool: {tool_name}. Available: {list(self.action_executors.keys())}",
                        error_message=f"Tool '{tool_name}' not registered"
                    )

            elif action.type == ActionType.CLARIFY:
                return Observation(
                    type=ObservationType.SUCCESS,
                    content=f"Clarification needed: {action.parameters.get('question', 'Please clarify your request')}",
                    matches_expectation=True
                )

            elif action.type == ActionType.RESPOND:
                return Observation(
                    type=ObservationType.SUCCESS,
                    content=action.parameters.get('response', ''),
                    matches_expectation=True
                )

            else:
                return Observation(
                    type=ObservationType.ERROR,
                    content=f"Unknown action type: {action.type}",
                    error_message=f"Cannot execute action type: {action.type}"
                )

        except Exception as e:
            return Observation(
                type=ObservationType.ERROR,
                content=f"Error executing {action.name}: {str(e)}",
                error_message=str(e),
                matches_expectation=False
            )

    def _should_terminate(self, thought: Thought) -> bool:
        """Check if thought indicates task completion"""
        completion_indicators = [
            "task is complete",
            "task has been completed",
            "successfully completed",
            "goal achieved",
            "finished",
            "done with the task"
        ]
        thought_lower = thought.content.lower()
        return any(indicator in thought_lower for indicator in completion_indicators)

    def _is_task_complete(self, observation: Observation, context: ExecutionContext) -> bool:
        """Check if observation indicates task completion"""
        if observation.type != ObservationType.SUCCESS:
            return False

        # Check for completion indicators in observation
        completion_indicators = ["completed", "success", "done", "finished", "created", "sent"]
        obs_lower = observation.content.lower()
        return any(indicator in obs_lower for indicator in completion_indicators)

    def _extract_thought(self, response: str) -> str:
        """Extract thought content from LLM response"""
        # Try to find thought section
        if "**Thought**:" in response:
            parts = response.split("**Thought**:")
            if len(parts) > 1:
                thought_part = parts[1].split("**Action**")[0] if "**Action**" in parts[1] else parts[1]
                return thought_part.strip()

        # Try alternative format
        if "Thought:" in response:
            parts = response.split("Thought:")
            if len(parts) > 1:
                thought_part = parts[1].split("Action:")[0] if "Action:" in parts[1] else parts[1]
                return thought_part.strip()

        # Return full response as thought
        return response.strip()

    def _extract_action(self, response: str) -> tuple:
        """Extract action, parameters, and expected outcome from response"""
        action_name = "respond"
        parameters = {}
        expected = ""

        # Extract action name
        if "**Action**:" in response:
            action_line = response.split("**Action**:")[1].split("\n")[0]
            action_name = action_line.strip()
        elif "Action:" in response:
            action_line = response.split("Action:")[1].split("\n")[0]
            action_name = action_line.strip()

        # Extract parameters
        if "**Parameters**:" in response:
            params_section = response.split("**Parameters**:")[1]
            params_end = params_section.find("**Expected")
            if params_end == -1:
                params_end = params_section.find("\n\n")
            params_text = params_section[:params_end] if params_end > 0 else params_section

            try:
                # Find JSON in params text
                start = params_text.find("{")
                end = params_text.rfind("}") + 1
                if start >= 0 and end > start:
                    parameters = json.loads(params_text[start:end])
            except json.JSONDecodeError:
                pass

        # Extract expected outcome
        if "**Expected Outcome**:" in response:
            expected = response.split("**Expected Outcome**:")[1].split("\n")[0].strip()

        return action_name, parameters, expected

    def _classify_thought_type(self, content: str) -> ThoughtType:
        """Classify the type of thought"""
        content_lower = content.lower()

        if any(word in content_lower for word in ["analyze", "examining", "looking at"]):
            return ThoughtType.ANALYSIS
        elif any(word in content_lower for word in ["plan", "will", "should", "next"]):
            return ThoughtType.PLANNING
        elif any(word in content_lower for word in ["error", "failed", "wrong", "issue"]):
            return ThoughtType.ERROR_ANALYSIS
        elif any(word in content_lower for word in ["decide", "choose", "select"]):
            return ThoughtType.DECISION
        else:
            return ThoughtType.ANALYSIS

    def _classify_action_type(self, action_name: str) -> ActionType:
        """Classify the type of action"""
        action_lower = action_name.lower()

        if action_lower in ["terminate", "finish", "complete", "done"]:
            return ActionType.TERMINATE
        elif action_lower in ["clarify", "ask", "question"]:
            return ActionType.CLARIFY
        elif action_lower in ["respond", "answer", "reply"]:
            return ActionType.RESPOND
        elif action_lower in ["wait", "pause"]:
            return ActionType.WAIT
        else:
            return ActionType.TOOL_CALL

    def _extract_final_answer(self, thought: Thought) -> str:
        """Extract final answer from completion thought"""
        return thought.content

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics"""
        success_rate = (
            self.successful_executions / self.total_executions * 100
            if self.total_executions > 0 else 0
        )
        avg_iterations = (
            self.total_iterations / self.total_executions
            if self.total_executions > 0 else 0
        )

        return {
            'total_executions': self.total_executions,
            'successful_executions': self.successful_executions,
            'success_rate': f"{success_rate:.1f}%",
            'total_iterations': self.total_iterations,
            'avg_iterations': f"{avg_iterations:.1f}"
        }

    def reset_statistics(self):
        """Reset engine statistics"""
        self.total_executions = 0
        self.total_iterations = 0
        self.successful_executions = 0
