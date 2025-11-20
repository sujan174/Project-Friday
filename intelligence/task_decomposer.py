"""
Task Decomposition Engine

Breaks complex requests into optimal execution plans with dependencies.
Understands task relationships, data flow, and execution order.

Author: AI System
Version: 2.0
"""

from typing import List, Dict, Optional, Tuple
from .base_types import (
    Task, Intent, IntentType, Entity, EntityType,
    ExecutionPlan, DependencyGraph
)


class TaskDecomposer:
    """
    Decompose complex requests into executable tasks

    Capabilities:
    - Break multi-intent requests into tasks
    - Detect task dependencies
    - Optimize execution order
    - Identify parallelizable tasks
    - Build execution plans with data flow
    """

    def __init__(self, agent_capabilities: Optional[Dict[str, List[str]]] = None, verbose: bool = False):
        """
        Initialize task decomposer

        Args:
            agent_capabilities: Map of agent names to their capabilities
            verbose: Enable verbose logging
        """
        self.agent_capabilities = agent_capabilities or {}
        self.verbose = verbose
        self.task_counter = 0

        # Intent-to-action mapping
        self.intent_actions = {
            IntentType.CREATE: ['create', 'build', 'generate'],
            IntentType.READ: ['get', 'fetch', 'list', 'search'],
            IntentType.UPDATE: ['update', 'modify', 'change', 'set'],
            IntentType.DELETE: ['delete', 'remove', 'close'],
            IntentType.ANALYZE: ['review', 'analyze', 'check'],
            IntentType.COORDINATE: ['notify', 'send', 'post'],
            IntentType.SEARCH: ['search', 'find', 'query'],
        }

        # Entity-to-agent mapping (guesses for agent selection)
        self.entity_agent_hints = {
            EntityType.ISSUE: ['jira', 'github'],
            EntityType.PR: ['github'],
            EntityType.PROJECT: ['jira'],
            EntityType.CHANNEL: ['slack'],
            EntityType.FILE: ['github', 'browser', 'scraper'],
            EntityType.CODE: ['code_reviewer', 'github'],
        }

    def decompose(
        self,
        message: str,
        intents: List[Intent],
        entities: List[Entity],
        context: Optional[Dict] = None
    ) -> ExecutionPlan:
        """
        Decompose message into execution plan

        Args:
            message: Original user message
            intents: Detected intents
            entities: Extracted entities
            context: Optional conversation context

        Returns:
            ExecutionPlan with tasks and dependencies
        """
        if not intents:
            return ExecutionPlan()

        # Generate tasks from intents
        tasks = []
        for intent in intents:
            task = self._intent_to_task(intent, entities, message, context)
            if task:
                tasks.append(task)

        # Detect dependencies between tasks
        dependency_graph = self._build_dependency_graph(tasks, intents)

        # Detect conditional tasks
        self._detect_conditional_tasks(tasks, message)

        # Estimate costs
        self._estimate_costs(tasks)

        # Build execution plan
        plan = ExecutionPlan(
            tasks=tasks,
            dependency_graph=dependency_graph,
            estimated_duration=sum(t.estimated_duration for t in tasks),
            estimated_cost=sum(t.estimated_cost for t in tasks)
        )

        # Detect potential issues
        plan.risks = self._identify_risks(plan)

        if self.verbose:
            print(f"[DECOMPOSE] Created plan with {len(tasks)} tasks")
            for task in tasks:
                print(f"  - {task}")
            if dependency_graph.edges:
                print(f"[DECOMPOSE] Dependencies: {dependency_graph.edges}")

        return plan

    def _intent_to_task(
        self,
        intent: Intent,
        entities: List[Entity],
        message: str,
        context: Optional[Dict]
    ) -> Optional[Task]:
        """Convert an intent to a task"""
        self.task_counter += 1
        task_id = f"task_{self.task_counter}"

        # Determine action from intent
        action = self._get_action_for_intent(intent)

        # Determine suggested agent from entities
        suggested_agent = self._suggest_agent_for_intent(intent, entities)

        # Extract relevant entities for this intent
        task_entities = self._filter_entities_for_intent(intent, entities)

        # Build task inputs from entities
        inputs = self._build_task_inputs(intent, task_entities, message)

        # Determine what this task outputs (for dependency detection)
        outputs = self._determine_task_outputs(intent, task_entities)

        task = Task(
            id=task_id,
            action=action,
            agent=suggested_agent,
            inputs=inputs,
            outputs=outputs,
            metadata={
                'intent': str(intent.type),
                'confidence': intent.confidence,
                'entities': [str(e) for e in task_entities]
            }
        )

        return task

    def _get_action_for_intent(self, intent: Intent) -> str:
        """Get action name for intent"""
        actions = self.intent_actions.get(intent.type, ['execute'])
        return actions[0]

    def _suggest_agent_for_intent(self, intent: Intent, entities: List[Entity]) -> Optional[str]:
        """
        Suggest which agent should handle this task

        Based on:
        - Intent type
        - Entities involved
        - Agent capabilities
        """
        # Get hints from entities
        agent_scores = {}

        for entity in entities:
            suggested_agents = self.entity_agent_hints.get(entity.type, [])
            for agent in suggested_agents:
                agent_scores[agent] = agent_scores.get(agent, 0) + entity.confidence

        # Intent-specific suggestions
        if intent.type == IntentType.ANALYZE:
            if any(e.type == EntityType.CODE for e in entities):
                agent_scores['code_reviewer'] = agent_scores.get('code_reviewer', 0) + 1.0

        elif intent.type == IntentType.COORDINATE:
            if any(e.type == EntityType.CHANNEL for e in entities):
                agent_scores['slack'] = agent_scores.get('slack', 0) + 1.0

        # Return highest scoring agent
        if agent_scores:
            return max(agent_scores.items(), key=lambda x: x[1])[0]

        return None

    def _filter_entities_for_intent(self, intent: Intent, entities: List[Entity]) -> List[Entity]:
        """Filter entities relevant to this specific intent"""
        # For now, include all entities (can be more sophisticated)
        return entities

    def _build_task_inputs(
        self,
        intent: Intent,
        entities: List[Entity],
        message: str
    ) -> Dict[str, any]:
        """
        Build task inputs from intent and entities

        Extracts structured data for the task
        """
        inputs = {
            'message': message,
            'intent_type': intent.type.value,
        }

        # Add entity values as inputs
        for entity in entities:
            key = entity.type.value
            value = entity.normalized_value or entity.value

            # Handle multiple entities of same type
            if key in inputs:
                # Convert to list
                if not isinstance(inputs[key], list):
                    inputs[key] = [inputs[key]]
                inputs[key].append(value)
            else:
                inputs[key] = value

        # Add implicit requirements
        for req in intent.implicit_requirements:
            if ':' in req:
                key, value = req.split(':', 1)
                inputs[key] = value

        return inputs

    def _determine_task_outputs(self, intent: Intent, entities: List[Entity]) -> List[str]:
        """
        Determine what outputs this task will produce

        Used for dependency detection
        """
        outputs = []

        # CREATE intents produce resources
        if intent.type == IntentType.CREATE:
            # Will create an issue, PR, page, etc.
            for entity in entities:
                if entity.type in [EntityType.ISSUE, EntityType.PR, EntityType.PROJECT]:
                    outputs.append(f"{entity.type.value}_id")
                    outputs.append(f"{entity.type.value}_url")

            # Generic outputs
            if not outputs:
                outputs = ['resource_id', 'resource_url']

        # READ intents produce data
        elif intent.type == IntentType.READ:
            outputs = ['data', 'results']

        # ANALYZE intents produce findings
        elif intent.type == IntentType.ANALYZE:
            outputs = ['analysis', 'findings', 'issues']

        return outputs

    def _build_dependency_graph(self, tasks: List[Task], intents: List[Intent]) -> DependencyGraph:
        """
        Build dependency graph for tasks

        Detects dependencies based on:
        - Task outputs needed by other tasks
        - Sequential intent order
        - Data flow requirements
        """
        graph = DependencyGraph()

        # Add all tasks
        for task in tasks:
            graph.add_task(task)

        # Detect dependencies
        for i, task in enumerate(tasks):
            # Sequential dependency: Later tasks may depend on earlier ones
            for j in range(i + 1, len(tasks)):
                dependent_task = tasks[j]

                # Check if dependent task needs outputs from this task
                if self._needs_output_from(dependent_task, task):
                    graph.add_dependency(task.id, dependent_task.id)
                    dependent_task.dependencies.append(task.id)

        return graph

    def _needs_output_from(self, dependent_task: Task, provider_task: Task) -> bool:
        """
        Check if dependent_task needs outputs from provider_task

        Heuristics:
        - If provider creates a resource, dependent might need its ID
        - If provider analyzes, dependent might need findings
        - Sequential tasks with related entities
        """
        # Check for output-input matching
        for output in provider_task.outputs:
            # Check if any input in dependent task matches this output pattern
            for input_key in dependent_task.inputs.keys():
                if output.replace('_', '') in input_key.replace('_', ''):
                    return True

        # Check for action relationships
        action_dependencies = {
            'create': ['update', 'notify', 'set'],  # Create must happen before update/notify
            'analyze': ['create', 'notify'],        # Analyze before create issue/notify
            'get': ['create', 'update', 'notify'],  # Fetch before action
            'search': ['create', 'update'],         # Search before action
        }

        provider_action = provider_task.action
        dependent_action = dependent_task.action

        for key_action, dependent_actions in action_dependencies.items():
            if key_action in provider_action and dependent_action in dependent_actions:
                return True

        return False

    def _detect_conditional_tasks(self, tasks: List[Task], message: str):
        """
        Detect if tasks are conditional

        Examples:
        - "create issue if there are bugs"
        - "notify team when done"
        """
        message_lower = message.lower()

        # Conditional keywords
        if 'if' in message_lower or 'when' in message_lower:
            for task in tasks:
                if task.action in ['create', 'notify', 'update']:
                    # Mark as conditional
                    task.conditions = "conditional:check_condition"
                    task.metadata['conditional'] = True

    def _estimate_costs(self, tasks: List[Task]):
        """Estimate duration and cost for each task"""
        for task in tasks:
            # Basic estimates (can be more sophisticated)
            if task.action == 'review' or task.action == 'analyze':
                task.estimated_duration = 5.0  # 5 seconds
                task.estimated_cost = 500.0    # 500 tokens

            elif task.action == 'create':
                task.estimated_duration = 2.0  # 2 seconds
                task.estimated_cost = 100.0    # 100 tokens

            elif task.action in ['get', 'fetch', 'list', 'search']:
                task.estimated_duration = 1.5  # 1.5 seconds
                task.estimated_cost = 50.0     # 50 tokens

            else:
                task.estimated_duration = 2.0  # default
                task.estimated_cost = 100.0    # default

    def _identify_risks(self, plan: ExecutionPlan) -> List[str]:
        """Identify potential risks in execution plan"""
        risks = []

        # Check for circular dependencies
        if plan.dependency_graph and plan.dependency_graph.has_cycle():
            risks.append("⚠️ CRITICAL: Circular dependencies detected")

        # Check for high cost
        if plan.estimated_cost > 1000:
            risks.append(f"⚠️ HIGH: Estimated cost is high ({plan.estimated_cost:.0f} tokens)")

        # Check for long duration
        if plan.estimated_duration > 30:
            risks.append(f"⚠️ MEDIUM: Estimated duration is long ({plan.estimated_duration:.1f}s)")

        # Check for many tasks
        if len(plan.tasks) > 10:
            risks.append(f"⚠️ MEDIUM: Many tasks ({len(plan.tasks)})")

        # Check for conditional tasks
        conditional_tasks = [t for t in plan.tasks if t.conditions]
        if conditional_tasks:
            risks.append(f"ℹ️ INFO: {len(conditional_tasks)} conditional tasks")

        return risks

    def get_parallel_tasks(self, plan: ExecutionPlan) -> List[List[Task]]:
        """
        Get groups of tasks that can be executed in parallel

        Returns:
            List of task groups that can run in parallel
        """
        if not plan.dependency_graph:
            # No dependencies, but execute sequentially for safety
            return [[task] for task in plan.tasks]

        # Group tasks by dependency level
        levels = []
        processed = set()
        remaining = set(plan.tasks)

        while remaining:
            # Find tasks with no unprocessed dependencies
            current_level = []
            for task in remaining:
                deps_processed = all(
                    dep_id in processed or dep_id not in [t.id for t in plan.tasks]
                    for dep_id in task.dependencies
                )
                if deps_processed:
                    current_level.append(task)

            if not current_level:
                # No progress possible - circular dependency or error
                break

            levels.append(current_level)
            for task in current_level:
                processed.add(task.id)
                remaining.remove(task)

        return levels

    def optimize_plan(self, plan: ExecutionPlan) -> ExecutionPlan:
        """
        Optimize execution plan

        Optimizations:
        - Remove redundant tasks
        - Merge similar tasks
        - Reorder for efficiency
        """
        # TODO: Implement optimizations
        # For now, return as-is
        return plan

    def explain_plan(self, plan: ExecutionPlan) -> str:
        """
        Generate human-readable explanation of execution plan

        Returns:
            Explanation string
        """
        lines = []
        lines.append(f"Execution Plan ({len(plan.tasks)} tasks):")
        lines.append("")

        # Get execution order
        ordered_tasks = plan.get_execution_order()

        for i, task in enumerate(ordered_tasks, 1):
            agent = task.agent or "?"
            action = task.action
            dependencies = f" (after {', '.join(task.dependencies)})" if task.dependencies else ""
            conditional = " [conditional]" if task.conditions else ""

            lines.append(f"{i}. [{agent}] {action}{dependencies}{conditional}")

        lines.append("")
        lines.append(f"Estimated duration: {plan.estimated_duration:.1f}s")
        lines.append(f"Estimated cost: {plan.estimated_cost:.0f} tokens")

        if plan.risks:
            lines.append("")
            lines.append("Risks:")
            for risk in plan.risks:
                lines.append(f"  {risk}")

        return "\n".join(lines)
