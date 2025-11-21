"""
ReAct Types - Data structures for Reasoning-Action-Observation Loop

Defines core primitives for modern agentic AI architecture:
- ReAct loop types (Thought, Action, Observation)
- Self-reflection types (Critique, Reflection)
- Planning types (PlanCandidate, PlanScore)
- Uncertainty types (CalibratedConfidence)
- Tool reasoning types (ToolReasoning)

Author: AI System
Version: 1.0
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime


# ============================================================================
# REACT LOOP TYPES
# ============================================================================

class ThoughtType(Enum):
    """Types of reasoning thoughts"""
    ANALYSIS = "analysis"           # Analyzing the current situation
    PLANNING = "planning"           # Planning next steps
    REFLECTION = "reflection"       # Reflecting on progress
    HYPOTHESIS = "hypothesis"       # Forming hypotheses
    DECISION = "decision"           # Making decisions
    ERROR_ANALYSIS = "error_analysis"  # Analyzing errors


@dataclass
class Thought:
    """Represents a reasoning step in the ReAct loop"""
    type: ThoughtType
    content: str
    reasoning_chain: List[str] = field(default_factory=list)
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"Thought({self.type.value}): {self.content[:100]}..."


class ActionType(Enum):
    """Types of actions the agent can take"""
    TOOL_CALL = "tool_call"         # Call an external tool
    QUERY = "query"                 # Query for information
    RESPOND = "respond"             # Respond to user
    CLARIFY = "clarify"             # Ask for clarification
    DELEGATE = "delegate"           # Delegate to another agent
    WAIT = "wait"                   # Wait for external event
    TERMINATE = "terminate"         # End the loop


@dataclass
class Action:
    """Represents an action in the ReAct loop"""
    type: ActionType
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""             # Why this action was chosen
    expected_outcome: str = ""      # What we expect to happen
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"Action({self.type.value}): {self.name}"


class ObservationType(Enum):
    """Types of observations"""
    SUCCESS = "success"             # Action succeeded
    FAILURE = "failure"             # Action failed
    PARTIAL = "partial"             # Partial success
    UNEXPECTED = "unexpected"       # Unexpected result
    TIMEOUT = "timeout"             # Action timed out
    ERROR = "error"                 # Error occurred


@dataclass
class Observation:
    """Represents an observation after an action"""
    type: ObservationType
    content: str
    raw_result: Any = None
    matches_expectation: bool = True
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"Observation({self.type.value}): {self.content[:100]}..."


@dataclass
class ReActStep:
    """A single step in the ReAct loop (Thought -> Action -> Observation)"""
    step_number: int
    thought: Thought
    action: Action
    observation: Optional[Observation] = None

    def is_complete(self) -> bool:
        return self.observation is not None


@dataclass
class ReActTrace:
    """Complete trace of a ReAct execution"""
    steps: List[ReActStep] = field(default_factory=list)
    final_result: Optional[str] = None
    total_iterations: int = 0
    terminated_reason: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    def add_step(self, step: ReActStep):
        self.steps.append(step)
        self.total_iterations = len(self.steps)

    def get_last_observation(self) -> Optional[Observation]:
        if self.steps and self.steps[-1].observation:
            return self.steps[-1].observation
        return None


# ============================================================================
# SELF-REFLECTION TYPES
# ============================================================================

class CritiqueType(Enum):
    """Types of critiques"""
    CORRECTNESS = "correctness"     # Is the output correct?
    COMPLETENESS = "completeness"   # Is the output complete?
    RELEVANCE = "relevance"         # Is the output relevant?
    EFFICIENCY = "efficiency"       # Is the approach efficient?
    SAFETY = "safety"               # Is the output safe?


@dataclass
class Critique:
    """Critique of an output or plan"""
    type: CritiqueType
    score: float                    # 0.0 to 1.0
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def is_acceptable(self, threshold: float = 0.7) -> bool:
        return self.score >= threshold


@dataclass
class Reflection:
    """Self-reflection on execution"""
    critiques: List[Critique] = field(default_factory=list)
    overall_score: float = 0.0
    needs_revision: bool = False
    revision_suggestions: List[str] = field(default_factory=list)
    learned_patterns: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def get_primary_issues(self) -> List[str]:
        """Get the most critical issues across all critiques"""
        all_issues = []
        for critique in self.critiques:
            if critique.score < 0.6:
                all_issues.extend(critique.issues)
        return all_issues


@dataclass
class RefinementResult:
    """Result of a refinement iteration"""
    iteration: int
    original_output: str
    refined_output: str
    reflection: Reflection
    improvement_score: float        # How much better is refined vs original
    converged: bool = False         # Did refinement converge?


# ============================================================================
# PLANNING TYPES
# ============================================================================

class PlanStatus(Enum):
    """Status of a plan"""
    DRAFT = "draft"
    VALIDATED = "validated"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    REPLANNING = "replanning"


@dataclass
class PlanStep:
    """A single step in a plan"""
    step_id: str
    description: str
    action: str
    agent: Optional[str] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    expected_output: str = ""
    dependencies: List[str] = field(default_factory=list)
    is_critical: bool = False       # If this fails, should we abort?
    fallback_action: Optional[str] = None
    estimated_duration: float = 0.0

    def __str__(self) -> str:
        return f"Step({self.step_id}): {self.description}"


@dataclass
class PlanScore:
    """Score for a plan candidate"""
    feasibility: float = 0.0        # Can this plan be executed?
    efficiency: float = 0.0         # How efficient is this plan?
    completeness: float = 0.0       # Does it cover all requirements?
    robustness: float = 0.0         # How well does it handle failures?
    total_score: float = 0.0
    reasoning: str = ""

    def compute_total(self, weights: Optional[Dict[str, float]] = None):
        """Compute weighted total score"""
        if weights is None:
            weights = {
                'feasibility': 0.3,
                'efficiency': 0.2,
                'completeness': 0.3,
                'robustness': 0.2
            }

        self.total_score = (
            self.feasibility * weights.get('feasibility', 0.25) +
            self.efficiency * weights.get('efficiency', 0.25) +
            self.completeness * weights.get('completeness', 0.25) +
            self.robustness * weights.get('robustness', 0.25)
        )


@dataclass
class PlanCandidate:
    """A candidate plan for execution"""
    plan_id: str
    goal: str
    steps: List[PlanStep] = field(default_factory=list)
    score: Optional[PlanScore] = None
    status: PlanStatus = PlanStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_execution_order(self) -> List[PlanStep]:
        """Get steps in execution order (respecting dependencies)"""
        # Simple topological sort
        executed = set()
        result = []
        remaining = list(self.steps)

        while remaining:
            for step in remaining[:]:
                deps_satisfied = all(d in executed for d in step.dependencies)
                if deps_satisfied:
                    result.append(step)
                    executed.add(step.step_id)
                    remaining.remove(step)

            if remaining and len(result) == len(self.steps) - len(remaining):
                # No progress made - circular dependency
                break

        return result


@dataclass
class PlanExecutionState:
    """State of plan execution"""
    plan: PlanCandidate
    current_step_index: int = 0
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    step_results: Dict[str, Any] = field(default_factory=dict)
    needs_replan: bool = False
    replan_reason: str = ""


# ============================================================================
# UNCERTAINTY TYPES
# ============================================================================

class UncertaintySource(Enum):
    """Sources of uncertainty"""
    ALEATORIC = "aleatoric"         # Inherent randomness in data
    EPISTEMIC = "epistemic"         # Lack of knowledge
    MODEL = "model"                 # Model limitations
    INPUT = "input"                 # Ambiguous input


@dataclass
class CalibratedConfidence:
    """Properly calibrated confidence score"""
    raw_score: float                # Original model confidence
    calibrated_score: float         # After temperature scaling
    entropy: float                  # Information-theoretic uncertainty
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    uncertainty_sources: List[UncertaintySource] = field(default_factory=list)
    should_clarify: bool = False
    clarification_questions: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"Confidence: {self.calibrated_score:.2f} (raw: {self.raw_score:.2f}, entropy: {self.entropy:.2f})"


@dataclass
class UncertaintyAnalysis:
    """Complete uncertainty analysis"""
    overall_uncertainty: float      # 0.0 (certain) to 1.0 (uncertain)
    confidence: CalibratedConfidence
    breakdown: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def should_proceed(self) -> bool:
        """Should we proceed without clarification?"""
        return self.overall_uncertainty < 0.3

    def should_confirm(self) -> bool:
        """Should we confirm with user?"""
        return 0.3 <= self.overall_uncertainty < 0.6

    def should_clarify(self) -> bool:
        """Should we ask for clarification?"""
        return self.overall_uncertainty >= 0.6


# ============================================================================
# TOOL REASONING TYPES
# ============================================================================

@dataclass
class ToolCapability:
    """Description of what a tool can do"""
    tool_name: str
    description: str
    capabilities: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    typical_use_cases: List[str] = field(default_factory=list)
    cost_estimate: float = 0.0      # Relative cost (0-1)
    latency_estimate: float = 0.0   # Relative latency (0-1)


@dataclass
class ToolReasoning:
    """Reasoning about tool selection"""
    selected_tool: str
    reasoning_chain: List[str] = field(default_factory=list)
    alternative_tools: List[str] = field(default_factory=list)
    why_not_alternatives: Dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    requires_composition: bool = False
    composition_plan: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"ToolReasoning: {self.selected_tool} (confidence: {self.confidence:.2f})"


@dataclass
class ToolComposition:
    """Plan for composing multiple tools"""
    tools: List[str]
    sequence: List[Tuple[str, str]]  # [(tool, purpose), ...]
    data_flow: Dict[str, str] = field(default_factory=dict)  # output -> input mapping
    reasoning: str = ""


# ============================================================================
# EXECUTION CONTEXT TYPES
# ============================================================================

@dataclass
class ExecutionContext:
    """Context for ReAct execution"""
    task: str
    goal: str
    constraints: List[str] = field(default_factory=list)
    available_tools: List[str] = field(default_factory=list)
    max_iterations: int = 10
    timeout_seconds: float = 120.0
    require_confirmation: bool = False
    allow_replanning: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of ReAct execution"""
    success: bool
    result: str
    trace: ReActTrace
    reflection: Optional[Reflection] = None
    uncertainty: Optional[UncertaintyAnalysis] = None
    total_iterations: int = 0
    total_duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_summary(self) -> str:
        """Get execution summary"""
        status = "SUCCESS" if self.success else "FAILED"
        return f"{status}: {self.total_iterations} iterations, {self.total_duration_seconds:.2f}s"
