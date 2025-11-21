"""
LLM-Based Planning System

Replaces rule-based task decomposition with LLM reasoning:
- Generate multiple plan candidates
- Score and rank plans
- Detect dependencies through understanding
- Dynamic replanning on failures

Author: AI System
Version: 1.0
"""

import json
import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime

import google.generativeai as genai

from .react_types import (
    PlanCandidate, PlanStep, PlanScore, PlanStatus,
    PlanExecutionState, ExecutionContext
)
from .base_types import Intent, Entity, ExecutionPlan, Task
from .critic import CriticModel


class LLMPlanner:
    """
    LLM-Based Planner

    Uses LLM reasoning for intelligent task planning:
    - Semantic understanding of task requirements
    - Multiple plan generation and scoring
    - Dependency detection through reasoning
    - Adaptive replanning on failures
    """

    def __init__(
        self,
        llm_model: str = 'models/gemini-2.5-flash',
        verbose: bool = False
    ):
        """
        Initialize LLM Planner

        Args:
            llm_model: Gemini model to use
            verbose: Enable verbose logging
        """
        self.llm_model = llm_model
        self.verbose = verbose

        # Critic for plan evaluation
        self.critic = CriticModel(llm_model, verbose)

        # Agent capabilities (populated by orchestrator)
        self.agent_capabilities: Dict[str, List[str]] = {}

        # Statistics
        self.plans_generated = 0
        self.replans_performed = 0

    def set_agent_capabilities(self, capabilities: Dict[str, List[str]]):
        """
        Set available agent capabilities

        Args:
            capabilities: Map of agent names to capability lists
        """
        self.agent_capabilities = capabilities

    async def create_plan(
        self,
        message: str,
        intents: List[Intent],
        entities: List[Entity],
        context: Optional[ExecutionContext] = None
    ) -> PlanCandidate:
        """
        Create an execution plan using LLM reasoning

        Args:
            message: User message
            intents: Detected intents
            entities: Extracted entities
            context: Optional execution context

        Returns:
            Best plan candidate
        """
        self.plans_generated += 1

        # Dynamic candidate count based on task complexity
        # Simple (1 intent): 1 candidate, Moderate (2-3): 2 candidates, Complex (4+): 3 candidates
        num_intents = len(intents) if intents else 1
        if num_intents <= 1:
            num_candidates = 1  # Simple task, single plan
        elif num_intents <= 3:
            num_candidates = 2  # Moderate task
        else:
            num_candidates = 3  # Complex task

        # Generate plan candidates
        candidates = await self._generate_plan_candidates(
            message, intents, entities, context, num_candidates=num_candidates
        )

        if not candidates:
            # Fallback: create simple single-step plan
            return self._create_fallback_plan(message, intents, entities)

        # For single candidate (simple tasks), skip expensive critic evaluation
        if len(candidates) == 1:
            best_plan = candidates[0]
            # Quick default score without LLM call
            best_plan.score = PlanScore(
                feasibility=0.8, efficiency=0.8,
                completeness=0.8, robustness=0.7
            )
            best_plan.score.compute_total()
            best_plan.status = PlanStatus.VALIDATED

            if self.verbose:
                print(f"[PLANNER] Single plan (fast): {best_plan.score.total_score:.2f}")
            return best_plan

        # Score and rank multiple candidates
        scored_candidates = []
        for candidate in candidates:
            exec_context = context or ExecutionContext(
                task=message,
                goal=self._infer_goal(intents),
                available_tools=list(self.agent_capabilities.keys())
            )
            score = await self.critic.evaluate_plan(candidate, exec_context)
            candidate.score = score
            scored_candidates.append(candidate)

        # Sort by score
        scored_candidates.sort(key=lambda c: c.score.total_score, reverse=True)

        best_plan = scored_candidates[0]
        best_plan.status = PlanStatus.VALIDATED

        if self.verbose:
            print(f"[PLANNER] Generated {len(candidates)} plans, best score: {best_plan.score.total_score:.2f}")

        return best_plan

    async def _generate_plan_candidates(
        self,
        message: str,
        intents: List[Intent],
        entities: List[Entity],
        context: Optional[ExecutionContext],
        num_candidates: int = 3
    ) -> List[PlanCandidate]:
        """Generate multiple plan candidates"""
        # Build planning prompt
        prompt = self._build_planning_prompt(message, intents, entities, context)

        model = genai.GenerativeModel(self.llm_model)
        candidates = []

        for i in range(num_candidates):
            try:
                # Add variation for different candidates
                variation_prompt = prompt
                if i > 0:
                    variation_prompt += f"\n\nGenerate a DIFFERENT approach (variation {i+1})."

                response = await model.generate_content_async(variation_prompt)
                response_text = response.text if hasattr(response, 'text') else str(response)

                # Parse plan from response
                plan = self._parse_plan_response(response_text, message)
                if plan:
                    candidates.append(plan)

            except Exception as e:
                if self.verbose:
                    print(f"[PLANNER] Error generating candidate {i+1}: {e}")

        return candidates

    def _build_planning_prompt(
        self,
        message: str,
        intents: List[Intent],
        entities: List[Entity],
        context: Optional[ExecutionContext]
    ) -> str:
        """Build the planning prompt"""
        # Format intents
        intents_text = ", ".join(f"{i.type.value}({i.confidence:.2f})" for i in intents)

        # Format entities
        entities_text = ", ".join(f"{e.type.value}:{e.value}" for e in entities)

        # Format agent capabilities
        agents_text = ""
        for agent, caps in self.agent_capabilities.items():
            caps_short = caps[:5] if len(caps) > 5 else caps
            agents_text += f"- {agent}: {', '.join(caps_short)}\n"

        if not agents_text:
            agents_text = "- Various agents available for different tasks\n"

        constraints = ""
        if context and context.constraints:
            constraints = f"\nCONSTRAINTS: {', '.join(context.constraints)}"

        return f"""Create an execution plan for this request:

USER REQUEST: {message}

DETECTED INTENTS: {intents_text}
EXTRACTED ENTITIES: {entities_text}
{constraints}

AVAILABLE AGENTS:
{agents_text}

Create a step-by-step plan. For each step, specify:
1. What to do (action)
2. Which agent should do it
3. What inputs are needed
4. What output is expected
5. Dependencies (which steps must complete first)

Respond in JSON format:
{{
    "goal": "what this plan achieves",
    "steps": [
        {{
            "step_id": "step_1",
            "description": "what this step does",
            "action": "action_name",
            "agent": "agent_name",
            "inputs": {{"key": "value"}},
            "expected_output": "what this produces",
            "dependencies": [],
            "is_critical": true
        }}
    ]
}}

IMPORTANT:
- Order steps logically with correct dependencies
- Use the most appropriate agent for each step
- Include all necessary steps to complete the task
- Be specific about inputs and expected outputs"""

    def _parse_plan_response(self, response: str, original_message: str) -> Optional[PlanCandidate]:
        """Parse plan from LLM response"""
        try:
            # Find JSON in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start < 0 or end <= start:
                return None

            data = json.loads(response[start:end])

            # Create plan steps
            steps = []
            for step_data in data.get('steps', []):
                step = PlanStep(
                    step_id=step_data.get('step_id', f'step_{len(steps)+1}'),
                    description=step_data.get('description', ''),
                    action=step_data.get('action', ''),
                    agent=step_data.get('agent'),
                    inputs=step_data.get('inputs', {}),
                    expected_output=step_data.get('expected_output', ''),
                    dependencies=step_data.get('dependencies', []),
                    is_critical=step_data.get('is_critical', False)
                )
                steps.append(step)

            if not steps:
                return None

            # Create plan candidate
            plan = PlanCandidate(
                plan_id=f"plan_{datetime.now().strftime('%H%M%S')}",
                goal=data.get('goal', original_message),
                steps=steps,
                status=PlanStatus.DRAFT
            )

            return plan

        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"[PLANNER] JSON parse error: {e}")
            return None

    def _create_fallback_plan(
        self,
        message: str,
        intents: List[Intent],
        entities: List[Entity]
    ) -> PlanCandidate:
        """Create a simple fallback plan"""
        # Determine primary intent
        primary_intent = intents[0] if intents else None
        action = "execute"
        if primary_intent:
            action = primary_intent.type.value

        # Create single step
        step = PlanStep(
            step_id="step_1",
            description=f"Execute: {message}",
            action=action,
            inputs={'message': message},
            expected_output="Task result",
            is_critical=True
        )

        return PlanCandidate(
            plan_id=f"fallback_{datetime.now().strftime('%H%M%S')}",
            goal=message,
            steps=[step],
            status=PlanStatus.DRAFT
        )

    async def replan(
        self,
        original_plan: PlanCandidate,
        execution_state: PlanExecutionState,
        failure_reason: str,
        context: ExecutionContext
    ) -> PlanCandidate:
        """
        Create a new plan after a failure

        Args:
            original_plan: The plan that failed
            execution_state: Current execution state
            failure_reason: Why the plan failed
            context: Execution context

        Returns:
            New plan to continue execution
        """
        self.replans_performed += 1

        # Build replanning prompt
        completed = ", ".join(execution_state.completed_steps) if execution_state.completed_steps else "None"
        failed = ", ".join(execution_state.failed_steps) if execution_state.failed_steps else "None"

        prompt = f"""The current plan has failed and needs replanning.

ORIGINAL GOAL: {original_plan.goal}

COMPLETED STEPS: {completed}
FAILED STEPS: {failed}
FAILURE REASON: {failure_reason}

REMAINING GOAL: What still needs to be accomplished to complete the original goal.

Create a NEW plan to complete the remaining goal. Consider:
1. What was already accomplished (don't repeat)
2. Why the previous approach failed
3. Alternative approaches to achieve the goal

Respond in JSON format (same as before):
{{
    "goal": "remaining goal",
    "steps": [...]
}}"""

        model = genai.GenerativeModel(self.llm_model)
        response = await model.generate_content_async(prompt)
        response_text = response.text if hasattr(response, 'text') else str(response)

        # Parse new plan
        new_plan = self._parse_plan_response(response_text, original_plan.goal)

        if new_plan:
            new_plan.status = PlanStatus.VALIDATED
            new_plan.metadata['is_replan'] = True
            new_plan.metadata['original_plan_id'] = original_plan.plan_id
            new_plan.metadata['failure_reason'] = failure_reason

            if self.verbose:
                print(f"[PLANNER] Replanned with {len(new_plan.steps)} steps")
        else:
            # Fallback
            new_plan = self._create_fallback_plan(
                original_plan.goal,
                [],  # No intents for replan
                []   # No entities for replan
            )

        return new_plan

    async def detect_dependencies(
        self,
        steps: List[PlanStep]
    ) -> List[PlanStep]:
        """
        Use LLM to detect dependencies between steps

        Args:
            steps: List of plan steps

        Returns:
            Steps with updated dependencies
        """
        if len(steps) <= 1:
            return steps

        # Build dependency detection prompt
        steps_desc = "\n".join(
            f"{i+1}. {step.step_id}: {step.description} (produces: {step.expected_output})"
            for i, step in enumerate(steps)
        )

        prompt = f"""Analyze these steps and determine dependencies:

{steps_desc}

For each step, determine which previous steps MUST complete before it can start.
Consider data dependencies (step needs output from previous step).

Respond in JSON:
{{
    "dependencies": {{
        "step_id": ["list of step_ids this depends on"],
        ...
    }}
}}"""

        model = genai.GenerativeModel(self.llm_model)
        response = await model.generate_content_async(prompt)
        response_text = response.text if hasattr(response, 'text') else str(response)

        try:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response_text[start:end])
                deps = data.get('dependencies', {})

                # Update step dependencies
                for step in steps:
                    if step.step_id in deps:
                        step.dependencies = deps[step.step_id]

        except json.JSONDecodeError:
            pass

        return steps

    def convert_to_execution_plan(self, plan: PlanCandidate) -> ExecutionPlan:
        """
        Convert PlanCandidate to legacy ExecutionPlan format

        Args:
            plan: Plan candidate

        Returns:
            ExecutionPlan for backward compatibility
        """
        tasks = []

        for step in plan.steps:
            task = Task(
                id=step.step_id,
                action=step.action,
                agent=step.agent,
                inputs=step.inputs,
                outputs=[step.expected_output] if step.expected_output else [],
                dependencies=step.dependencies,
                metadata={
                    'description': step.description,
                    'is_critical': step.is_critical
                }
            )
            tasks.append(task)

        exec_plan = ExecutionPlan(
            tasks=tasks,
            estimated_duration=sum(s.estimated_duration for s in plan.steps),
            metadata={
                'plan_id': plan.plan_id,
                'goal': plan.goal,
                'score': plan.score.total_score if plan.score else 0.0
            }
        )

        return exec_plan

    def _infer_goal(self, intents: List[Intent]) -> str:
        """Infer goal from intents"""
        if not intents:
            return "Complete the requested task"

        primary = intents[0]
        goal_map = {
            'create': 'Create the requested resource',
            'read': 'Retrieve the requested information',
            'update': 'Update the specified resource',
            'delete': 'Delete the specified resource',
            'analyze': 'Analyze and provide insights',
            'coordinate': 'Coordinate and notify as requested',
            'search': 'Find the requested information'
        }

        return goal_map.get(primary.type.value, "Complete the requested task")

    def get_statistics(self) -> Dict[str, Any]:
        """Get planner statistics"""
        return {
            'plans_generated': self.plans_generated,
            'replans_performed': self.replans_performed,
            'critic_stats': self.critic.get_statistics()
        }

    def reset_statistics(self):
        """Reset statistics"""
        self.plans_generated = 0
        self.replans_performed = 0
        self.critic.reset_statistics()
