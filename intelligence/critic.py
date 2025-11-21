"""
Critic System - Self-Reflection and Evaluation

Implements a critic model for evaluating and improving agent outputs:
- Plan evaluation before execution
- Output verification after execution
- Iterative refinement loops
- Quality scoring and improvement suggestions

Author: AI System
Version: 1.0
"""

import json
from typing import Any, Dict, List, Optional
from datetime import datetime

import google.generativeai as genai

from .react_types import (
    Critique, CritiqueType, Reflection, RefinementResult,
    PlanCandidate, PlanScore, ExecutionContext
)
from .base_types import ExecutionPlan, Task


class CriticModel:
    """
    Critic Model for Self-Reflection

    Evaluates plans, actions, and outputs to ensure quality:
    - Score plans before execution
    - Verify outputs match expectations
    - Suggest improvements
    - Enable iterative refinement
    """

    def __init__(
        self,
        llm_model: str = 'models/gemini-2.5-flash',
        verbose: bool = False
    ):
        """
        Initialize Critic Model

        Args:
            llm_model: Gemini model to use for evaluation
            verbose: Enable verbose logging
        """
        self.llm_model = llm_model
        self.verbose = verbose

        # Statistics
        self.total_evaluations = 0
        self.total_refinements = 0

    async def evaluate_plan(
        self,
        plan: PlanCandidate,
        context: ExecutionContext
    ) -> PlanScore:
        """
        Evaluate a plan before execution

        Args:
            plan: Plan to evaluate
            context: Execution context

        Returns:
            PlanScore with detailed evaluation
        """
        self.total_evaluations += 1

        # Build evaluation prompt
        prompt = self._build_plan_evaluation_prompt(plan, context)

        # Get LLM evaluation
        model = genai.GenerativeModel(self.llm_model)
        response = await model.generate_content_async(prompt)
        response_text = response.text if hasattr(response, 'text') else str(response)

        # Parse scores from response
        score = self._parse_plan_score(response_text)

        if self.verbose:
            print(f"[CRITIC] Plan score: {score.total_score:.2f}")
            print(f"[CRITIC] Reasoning: {score.reasoning[:200]}...")

        return score

    async def evaluate_output(
        self,
        task: str,
        output: str,
        expected: Optional[str] = None
    ) -> Reflection:
        """
        Evaluate an output for quality

        Args:
            task: Original task
            output: Output to evaluate
            expected: Optional expected outcome

        Returns:
            Reflection with critiques and suggestions
        """
        self.total_evaluations += 1

        # Build evaluation prompt
        prompt = self._build_output_evaluation_prompt(task, output, expected)

        # Get LLM evaluation
        model = genai.GenerativeModel(self.llm_model)
        response = await model.generate_content_async(prompt)
        response_text = response.text if hasattr(response, 'text') else str(response)

        # Parse reflection from response
        reflection = self._parse_reflection(response_text)

        if self.verbose:
            print(f"[CRITIC] Output score: {reflection.overall_score:.2f}")
            if reflection.needs_revision:
                print(f"[CRITIC] Needs revision: {reflection.revision_suggestions}")

        return reflection

    async def evaluate_action(
        self,
        action_name: str,
        parameters: Dict[str, Any],
        context: str
    ) -> Critique:
        """
        Evaluate a single action before execution

        Args:
            action_name: Name of the action
            parameters: Action parameters
            context: Current context

        Returns:
            Critique of the action
        """
        prompt = f"""Evaluate this action:

Action: {action_name}
Parameters: {json.dumps(parameters, indent=2)}
Context: {context}

Evaluate on these criteria (score 0.0 to 1.0):
1. Is this the right action for the context?
2. Are the parameters correct and complete?
3. Is this action safe to execute?
4. Is this efficient (not redundant)?

Respond in JSON:
{{
    "score": 0.85,
    "is_appropriate": true,
    "issues": ["any issues found"],
    "suggestions": ["any improvements"],
    "reasoning": "why this score"
}}"""

        model = genai.GenerativeModel(self.llm_model)
        response = await model.generate_content_async(prompt)
        response_text = response.text if hasattr(response, 'text') else str(response)

        # Parse critique
        try:
            # Find JSON in response
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response_text[start:end])
                return Critique(
                    type=CritiqueType.CORRECTNESS,
                    score=data.get('score', 0.7),
                    issues=data.get('issues', []),
                    suggestions=data.get('suggestions', []),
                    reasoning=data.get('reasoning', '')
                )
        except json.JSONDecodeError:
            pass

        # Default critique
        return Critique(
            type=CritiqueType.CORRECTNESS,
            score=0.7,
            reasoning="Could not parse evaluation"
        )

    async def refine_output(
        self,
        task: str,
        output: str,
        reflection: Reflection,
        max_iterations: int = 3
    ) -> RefinementResult:
        """
        Iteratively refine output based on reflection

        Args:
            task: Original task
            output: Current output
            reflection: Reflection with critiques
            max_iterations: Maximum refinement iterations

        Returns:
            RefinementResult with refined output
        """
        self.total_refinements += 1

        current_output = output
        current_reflection = reflection
        iteration = 0

        model = genai.GenerativeModel(self.llm_model)

        while iteration < max_iterations and current_reflection.needs_revision:
            iteration += 1

            if self.verbose:
                print(f"[CRITIC] Refinement iteration {iteration}")

            # Build refinement prompt
            prompt = self._build_refinement_prompt(
                task, current_output, current_reflection
            )

            # Get refined output
            response = await model.generate_content_async(prompt)
            refined_output = response.text if hasattr(response, 'text') else str(response)

            # Re-evaluate
            new_reflection = await self.evaluate_output(task, refined_output)

            # Check for improvement
            improvement = new_reflection.overall_score - current_reflection.overall_score

            if self.verbose:
                print(f"[CRITIC] Improvement: {improvement:+.2f}")

            # Update for next iteration
            current_output = refined_output
            current_reflection = new_reflection

            # Check if we've converged
            if new_reflection.overall_score >= 0.85 or improvement < 0.05:
                break

        return RefinementResult(
            iteration=iteration,
            original_output=output,
            refined_output=current_output,
            reflection=current_reflection,
            improvement_score=current_reflection.overall_score - reflection.overall_score,
            converged=(current_reflection.overall_score >= 0.85)
        )

    async def verify_completion(
        self,
        task: str,
        result: str,
        success_criteria: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Verify that a task was successfully completed

        Args:
            task: Original task
            result: Execution result
            success_criteria: Optional specific criteria to check

        Returns:
            Verification result with details
        """
        criteria_text = ""
        if success_criteria:
            criteria_text = f"\nSuccess criteria:\n" + "\n".join(f"- {c}" for c in success_criteria)

        prompt = f"""Verify task completion:

Task: {task}
Result: {result}
{criteria_text}

Determine:
1. Was the task completed successfully?
2. What was accomplished?
3. What (if anything) was not accomplished?
4. Confidence in completion (0.0 to 1.0)

Respond in JSON:
{{
    "completed": true,
    "confidence": 0.95,
    "accomplished": ["list of accomplishments"],
    "not_accomplished": ["list of missing items"],
    "reasoning": "explanation"
}}"""

        model = genai.GenerativeModel(self.llm_model)
        response = await model.generate_content_async(prompt)
        response_text = response.text if hasattr(response, 'text') else str(response)

        try:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response_text[start:end])
        except json.JSONDecodeError:
            pass

        return {
            "completed": True,
            "confidence": 0.7,
            "accomplished": [],
            "not_accomplished": [],
            "reasoning": "Could not parse verification"
        }

    def _build_plan_evaluation_prompt(
        self,
        plan: PlanCandidate,
        context: ExecutionContext
    ) -> str:
        """Build prompt for plan evaluation"""
        steps_desc = "\n".join(
            f"{i+1}. {step.description} (agent: {step.agent or 'any'})"
            for i, step in enumerate(plan.steps)
        )

        return f"""Evaluate this execution plan:

GOAL: {plan.goal}
TASK: {context.task}
CONSTRAINTS: {', '.join(context.constraints) if context.constraints else 'None'}

PLAN STEPS:
{steps_desc}

Evaluate the plan on these criteria (score 0.0 to 1.0):

1. FEASIBILITY: Can this plan be executed successfully?
   - Are all steps achievable?
   - Are dependencies correct?

2. EFFICIENCY: Is this plan efficient?
   - Minimal steps to achieve goal?
   - No redundant actions?

3. COMPLETENESS: Does this plan fully achieve the goal?
   - All requirements covered?
   - No missing steps?

4. ROBUSTNESS: How well does it handle failures?
   - Has fallbacks?
   - Handles edge cases?

Respond in JSON:
{{
    "feasibility": 0.85,
    "efficiency": 0.80,
    "completeness": 0.90,
    "robustness": 0.75,
    "issues": ["list any issues"],
    "suggestions": ["list improvements"],
    "reasoning": "overall assessment"
}}"""

    def _build_output_evaluation_prompt(
        self,
        task: str,
        output: str,
        expected: Optional[str]
    ) -> str:
        """Build prompt for output evaluation"""
        expected_text = f"\nExpected outcome: {expected}" if expected else ""

        return f"""Evaluate this output:

TASK: {task}
{expected_text}

OUTPUT:
{output}

Evaluate on these criteria (score 0.0 to 1.0):

1. CORRECTNESS: Is the output correct?
2. COMPLETENESS: Does it fully address the task?
3. RELEVANCE: Is everything relevant to the task?
4. SAFETY: Is the output safe (no harmful content)?

Respond in JSON:
{{
    "correctness": 0.85,
    "completeness": 0.80,
    "relevance": 0.90,
    "safety": 1.0,
    "overall_score": 0.85,
    "needs_revision": false,
    "issues": ["list any issues"],
    "revision_suggestions": ["how to improve"],
    "reasoning": "overall assessment"
}}"""

    def _build_refinement_prompt(
        self,
        task: str,
        output: str,
        reflection: Reflection
    ) -> str:
        """Build prompt for output refinement"""
        issues = "\n".join(f"- {issue}" for issue in reflection.get_primary_issues())
        suggestions = "\n".join(f"- {s}" for s in reflection.revision_suggestions)

        return f"""Refine this output based on the feedback:

TASK: {task}

CURRENT OUTPUT:
{output}

ISSUES IDENTIFIED:
{issues if issues else "None"}

SUGGESTIONS FOR IMPROVEMENT:
{suggestions if suggestions else "None"}

Please provide an improved version that addresses these issues while maintaining what was correct in the original. Only output the refined result, nothing else."""

    def _parse_plan_score(self, response: str) -> PlanScore:
        """Parse plan score from LLM response"""
        score = PlanScore()

        try:
            # Find JSON in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])

                score.feasibility = data.get('feasibility', 0.7)
                score.efficiency = data.get('efficiency', 0.7)
                score.completeness = data.get('completeness', 0.7)
                score.robustness = data.get('robustness', 0.7)
                score.reasoning = data.get('reasoning', '')

                score.compute_total()

        except json.JSONDecodeError:
            # Default scores if parsing fails
            score.feasibility = 0.7
            score.efficiency = 0.7
            score.completeness = 0.7
            score.robustness = 0.7
            score.compute_total()
            score.reasoning = "Could not parse evaluation response"

        return score

    def _parse_reflection(self, response: str) -> Reflection:
        """Parse reflection from LLM response"""
        reflection = Reflection()

        try:
            # Find JSON in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])

                # Create critiques for each criterion
                for crit_type, key in [
                    (CritiqueType.CORRECTNESS, 'correctness'),
                    (CritiqueType.COMPLETENESS, 'completeness'),
                    (CritiqueType.RELEVANCE, 'relevance'),
                    (CritiqueType.SAFETY, 'safety')
                ]:
                    if key in data:
                        reflection.critiques.append(Critique(
                            type=crit_type,
                            score=data[key],
                            issues=data.get('issues', []) if data[key] < 0.7 else [],
                            suggestions=data.get('revision_suggestions', []) if data[key] < 0.7 else []
                        ))

                reflection.overall_score = data.get('overall_score', 0.7)
                reflection.needs_revision = data.get('needs_revision', False)
                reflection.revision_suggestions = data.get('revision_suggestions', [])

        except json.JSONDecodeError:
            # Default reflection if parsing fails
            reflection.overall_score = 0.7
            reflection.needs_revision = False

        return reflection

    def get_statistics(self) -> Dict[str, Any]:
        """Get critic statistics"""
        return {
            'total_evaluations': self.total_evaluations,
            'total_refinements': self.total_refinements
        }

    def reset_statistics(self):
        """Reset statistics"""
        self.total_evaluations = 0
        self.total_refinements = 0
