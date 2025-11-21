"""
Tool Reasoning System

Implements reasoned tool selection for agentic AI:
- LLM-based reasoning about tool choice
- Tool composition planning
- Alternative tool suggestions
- Justification for selections

Author: AI System
Version: 1.0
"""

import json
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import google.generativeai as genai

from .react_types import ToolReasoning, ToolCapability, ToolComposition
from .base_types import Intent, Entity


class ToolReasoner:
    """
    Tool Reasoning System

    Uses LLM to reason about tool selection:
    - Analyze task requirements
    - Match against tool capabilities
    - Generate reasoning chain
    - Suggest alternatives
    - Plan tool compositions
    """

    def __init__(
        self,
        llm_model: str = 'models/gemini-2.5-flash',
        verbose: bool = False
    ):
        """
        Initialize Tool Reasoner

        Args:
            llm_model: Gemini model to use
            verbose: Enable verbose logging
        """
        self.llm_model = llm_model
        self.verbose = verbose

        # Tool registry
        self.tools: Dict[str, ToolCapability] = {}

        # Cache for tool reasoning results (5-minute TTL)
        self._cache: Dict[str, Tuple[ToolReasoning, float]] = {}
        self._cache_ttl = 300  # 5 minutes

        # Statistics
        self.total_reasonings = 0
        self.compositions_planned = 0
        self.cache_hits = 0

    def register_tool(self, tool: ToolCapability):
        """
        Register a tool with its capabilities

        Args:
            tool: Tool capability description
        """
        self.tools[tool.tool_name] = tool

    def register_agent_as_tool(
        self,
        agent_name: str,
        description: str,
        capabilities: List[str],
        limitations: Optional[List[str]] = None
    ):
        """
        Register an agent as a tool

        Args:
            agent_name: Name of the agent
            description: What the agent does
            capabilities: List of capabilities
            limitations: Optional limitations
        """
        tool = ToolCapability(
            tool_name=agent_name,
            description=description,
            capabilities=capabilities,
            limitations=limitations or []
        )
        self.register_tool(tool)

    async def reason_about_tools(
        self,
        task: str,
        intents: List[Intent],
        entities: List[Entity],
        context: Optional[str] = None
    ) -> ToolReasoning:
        """
        Reason about which tool to use for a task

        Uses caching to avoid redundant LLM calls for similar task patterns.

        Args:
            task: Task description
            intents: Detected intents
            entities: Extracted entities
            context: Additional context

        Returns:
            ToolReasoning with selected tool and justification
        """
        self.total_reasonings += 1

        # Generate cache key from intent types (pattern-based caching)
        cache_key = self._generate_cache_key(intents, entities)

        # Check cache
        if cache_key in self._cache:
            cached_result, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                self.cache_hits += 1
                if self.verbose:
                    print(f"[TOOL_REASONER] Cache hit for pattern: {cache_key[:20]}...")
                return cached_result

        # Build reasoning prompt
        prompt = self._build_reasoning_prompt(task, intents, entities, context)

        # Get LLM reasoning
        model = genai.GenerativeModel(self.llm_model)
        response = await model.generate_content_async(prompt)
        response_text = response.text if hasattr(response, 'text') else str(response)

        # Parse reasoning
        reasoning = self._parse_reasoning_response(response_text)

        # Cache the result
        self._cache[cache_key] = (reasoning, time.time())

        if self.verbose:
            print(f"[TOOL_REASONER] Selected: {reasoning.selected_tool}")
            print(f"[TOOL_REASONER] Confidence: {reasoning.confidence:.2f}")

        return reasoning

    def _generate_cache_key(self, intents: List[Intent], entities: List[Entity]) -> str:
        """Generate cache key based on intent and entity patterns"""
        # Key is based on intent types and entity types (not values)
        intent_types = sorted([i.type.value for i in intents]) if intents else []
        entity_types = sorted([e.type.value for e in entities]) if entities else []

        pattern = f"{','.join(intent_types)}|{','.join(entity_types)}"
        return hashlib.md5(pattern.encode()).hexdigest()

    async def plan_tool_composition(
        self,
        task: str,
        required_capabilities: List[str]
    ) -> ToolComposition:
        """
        Plan a composition of multiple tools

        Args:
            task: Task requiring multiple tools
            required_capabilities: Capabilities needed

        Returns:
            ToolComposition with sequence and data flow
        """
        self.compositions_planned += 1

        # Build composition prompt
        tools_desc = self._format_tools_description()

        prompt = f"""Plan a composition of tools for this task:

TASK: {task}

REQUIRED CAPABILITIES:
{chr(10).join(f'- {cap}' for cap in required_capabilities)}

AVAILABLE TOOLS:
{tools_desc}

Create a sequence of tools that together can accomplish the task.
For each tool, specify:
1. Which tool to use
2. What it should do
3. What data it receives from previous tools
4. What data it outputs for next tools

Respond in JSON:
{{
    "tools": ["tool1", "tool2"],
    "sequence": [
        ["tool1", "purpose of tool1"],
        ["tool2", "purpose of tool2"]
    ],
    "data_flow": {{
        "tool1_output": "tool2_input"
    }},
    "reasoning": "why this composition"
}}"""

        model = genai.GenerativeModel(self.llm_model)
        response = await model.generate_content_async(prompt)
        response_text = response.text if hasattr(response, 'text') else str(response)

        # Parse composition
        return self._parse_composition_response(response_text)

    async def validate_tool_choice(
        self,
        tool_name: str,
        task: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate that a tool choice is appropriate

        Args:
            tool_name: Selected tool
            task: Task to perform
            parameters: Tool parameters

        Returns:
            Validation result
        """
        if tool_name not in self.tools:
            return {
                'valid': False,
                'reason': f"Unknown tool: {tool_name}",
                'suggestions': list(self.tools.keys())[:3]
            }

        tool = self.tools[tool_name]

        prompt = f"""Validate this tool selection:

TASK: {task}
SELECTED TOOL: {tool_name}
TOOL DESCRIPTION: {tool.description}
TOOL CAPABILITIES: {', '.join(tool.capabilities)}
TOOL LIMITATIONS: {', '.join(tool.limitations) if tool.limitations else 'None'}
PARAMETERS: {json.dumps(parameters)}

Is this the right tool for the task?
Are the parameters correct?

Respond in JSON:
{{
    "valid": true,
    "confidence": 0.9,
    "issues": [],
    "suggestions": [],
    "reasoning": "why valid or not"
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
            'valid': True,
            'confidence': 0.7,
            'reasoning': 'Could not validate'
        }

    def _build_reasoning_prompt(
        self,
        task: str,
        intents: List[Intent],
        entities: List[Entity],
        context: Optional[str]
    ) -> str:
        """Build prompt for tool reasoning"""
        # Format intents
        intents_text = ", ".join(f"{i.type.value}({i.confidence:.2f})" for i in intents)

        # Format entities
        entities_text = ", ".join(f"{e.type.value}:{e.value}" for e in entities)

        # Format available tools
        tools_desc = self._format_tools_description()

        context_text = f"\nCONTEXT: {context}" if context else ""

        return f"""Select the best tool for this task:

TASK: {task}
INTENTS: {intents_text}
ENTITIES: {entities_text}
{context_text}

AVAILABLE TOOLS:
{tools_desc}

Think step by step:
1. What does this task require?
2. Which tool has the right capabilities?
3. Are there any limitations to consider?
4. What alternatives exist?

Respond in JSON:
{{
    "selected_tool": "tool_name",
    "reasoning_chain": [
        "step 1 of reasoning",
        "step 2 of reasoning"
    ],
    "confidence": 0.9,
    "alternative_tools": ["tool2", "tool3"],
    "why_not_alternatives": {{
        "tool2": "reason not to use",
        "tool3": "reason not to use"
    }},
    "requires_composition": false
}}"""

    def _format_tools_description(self) -> str:
        """Format tool descriptions for prompts"""
        if not self.tools:
            return "No tools registered"

        lines = []
        for name, tool in self.tools.items():
            caps = ", ".join(tool.capabilities[:5])
            limitations = f" (Limitations: {', '.join(tool.limitations[:2])})" if tool.limitations else ""
            lines.append(f"- {name}: {tool.description}\n  Capabilities: {caps}{limitations}")

        return "\n".join(lines)

    def _parse_reasoning_response(self, response: str) -> ToolReasoning:
        """Parse tool reasoning from LLM response"""
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])

                return ToolReasoning(
                    selected_tool=data.get('selected_tool', ''),
                    reasoning_chain=data.get('reasoning_chain', []),
                    alternative_tools=data.get('alternative_tools', []),
                    why_not_alternatives=data.get('why_not_alternatives', {}),
                    confidence=data.get('confidence', 0.7),
                    requires_composition=data.get('requires_composition', False),
                    composition_plan=data.get('composition_plan', [])
                )

        except json.JSONDecodeError:
            pass

        # Fallback: try to extract tool name from response
        for tool_name in self.tools.keys():
            if tool_name in response.lower():
                return ToolReasoning(
                    selected_tool=tool_name,
                    reasoning_chain=["Could not parse full reasoning"],
                    confidence=0.5
                )

        # Default to first tool if parsing fails
        default_tool = list(self.tools.keys())[0] if self.tools else "unknown"
        return ToolReasoning(
            selected_tool=default_tool,
            reasoning_chain=["Fallback selection - could not parse response"],
            confidence=0.3
        )

    def _parse_composition_response(self, response: str) -> ToolComposition:
        """Parse tool composition from LLM response"""
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])

                return ToolComposition(
                    tools=data.get('tools', []),
                    sequence=data.get('sequence', []),
                    data_flow=data.get('data_flow', {}),
                    reasoning=data.get('reasoning', '')
                )

        except json.JSONDecodeError:
            pass

        return ToolComposition(
            tools=[],
            sequence=[],
            reasoning="Could not parse composition"
        )

    def suggest_tool_for_intent(self, intent: Intent) -> Optional[str]:
        """
        Quick suggestion based on intent type (for fast path)

        Args:
            intent: Detected intent

        Returns:
            Suggested tool name or None
        """
        # Simple intent-to-tool mapping for fast path
        intent_tool_map = {
            'create': ['jira', 'github', 'notion'],
            'read': ['jira', 'github', 'notion', 'slack'],
            'update': ['jira', 'github', 'notion'],
            'delete': ['jira', 'github', 'notion'],
            'analyze': ['code_reviewer', 'github'],
            'coordinate': ['slack', 'calendar'],
            'search': ['jira', 'github', 'notion', 'slack'],
        }

        candidates = intent_tool_map.get(intent.type.value, [])

        # Return first available tool from candidates
        for tool_name in candidates:
            if tool_name in self.tools:
                return tool_name

        return None

    def get_tool_capabilities(self, tool_name: str) -> Optional[ToolCapability]:
        """Get capabilities for a specific tool"""
        return self.tools.get(tool_name)

    def list_tools_for_capability(self, capability: str) -> List[str]:
        """
        Find tools that have a specific capability

        Args:
            capability: Capability to search for

        Returns:
            List of tool names
        """
        matching = []
        capability_lower = capability.lower()

        for name, tool in self.tools.items():
            for cap in tool.capabilities:
                if capability_lower in cap.lower():
                    matching.append(name)
                    break

        return matching

    def get_statistics(self) -> Dict[str, Any]:
        """Get reasoner statistics"""
        cache_hit_rate = (
            self.cache_hits / self.total_reasonings * 100
            if self.total_reasonings > 0 else 0
        )
        return {
            'total_reasonings': self.total_reasonings,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'compositions_planned': self.compositions_planned,
            'registered_tools': len(self.tools),
            'cached_patterns': len(self._cache)
        }

    def reset_statistics(self):
        """Reset statistics"""
        self.total_reasonings = 0
        self.compositions_planned = 0
        self.cache_hits = 0

    def clear_cache(self):
        """Clear the reasoning cache"""
        self._cache.clear()
