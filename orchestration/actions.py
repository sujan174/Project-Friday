# Action management system - models, parsing, and enrichment

import re
import uuid
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
from datetime import datetime

from config import Config
from core.input_validator import InputValidator

# logger removed


class ActionType(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    SEND = "send"
    NOTIFY = "notify"
    ARCHIVE = "archive"
    EXECUTE = "execute"
    READ = "read"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ActionStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class FieldConstraint:
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    forbidden_values: Optional[List[Any]] = None

    def validate(self, value: Any) -> tuple:
        if isinstance(value, str):
            if self.min_length and len(value) < self.min_length:
                return False, f"Must be at least {self.min_length} characters"
            if self.max_length and len(value) > self.max_length:
                return False, f"Must be at most {self.max_length} characters"
            if self.pattern:
                if not re.match(self.pattern, value):
                    return False, f"Must match pattern: {self.pattern}"

        if isinstance(value, (int, float)):
            if self.min_value is not None and value < self.min_value:
                return False, f"Must be at least {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Must be at most {self.max_value}"

        if self.allowed_values and value not in self.allowed_values:
            return False, f"Must be one of: {self.allowed_values}"

        if self.forbidden_values and value in self.forbidden_values:
            return False, f"Cannot be: {self.forbidden_values}"

        return True, None


@dataclass
class FieldInfo:
    display_label: str
    description: str
    field_type: str
    current_value: Any
    editable: bool = True
    required: bool = True
    constraints: FieldConstraint = field(default_factory=FieldConstraint)
    examples: Optional[List[str]] = None


@dataclass
class Action:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)

    agent_name: str = ""
    action_type: ActionType = ActionType.EXECUTE
    risk_level: RiskLevel = RiskLevel.MEDIUM

    instruction: str = ""
    context: Optional[Any] = None

    parameters: Dict[str, Any] = field(default_factory=dict)
    field_info: Dict[str, FieldInfo] = field(default_factory=dict)

    status: ActionStatus = ActionStatus.PENDING
    user_edits: Dict[str, Any] = field(default_factory=dict)
    user_decision_at: Optional[datetime] = None
    user_decision_note: Optional[str] = None

    executed_at: Optional[datetime] = None
    execution_result: Optional[str] = None
    execution_error: Optional[str] = None

    batch_id: Optional[str] = None

    details: Dict[str, Any] = field(default_factory=dict)

    def apply_edits(self) -> Dict[str, Any]:
        merged = self.parameters.copy()
        merged.update(self.user_edits)
        return merged

    def mark_ready(self) -> None:
        self.status = ActionStatus.CONFIRMED
        self.user_decision_at = datetime.now()

    def mark_rejected(self, reason: str = "") -> None:
        self.status = ActionStatus.REJECTED
        self.user_decision_at = datetime.now()
        self.user_decision_note = reason

    def mark_executing(self) -> None:
        self.status = ActionStatus.EXECUTING

    def mark_succeeded(self, result: str) -> None:
        self.status = ActionStatus.SUCCEEDED
        self.executed_at = datetime.now()
        self.execution_result = result

    def mark_failed(self, error: str) -> None:
        self.status = ActionStatus.FAILED
        self.executed_at = datetime.now()
        self.execution_error = error

    def is_ready_for_execution(self) -> bool:
        return self.status == ActionStatus.CONFIRMED

    def can_edit(self, field_name: str) -> bool:
        if field_name not in self.field_info:
            return False
        return self.field_info[field_name].editable

    def validate_edits(self) -> tuple:
        errors = []

        for field_name, new_value in self.user_edits.items():
            if field_name not in self.field_info:
                errors.append(f"Unknown field: {field_name}")
                continue

            field = self.field_info[field_name]

            if not field.editable:
                errors.append(f"Field '{field_name}' is not editable")
                continue

            is_valid, error_msg = field.constraints.validate(new_value)
            if not is_valid:
                errors.append(f"{field_name}: {error_msg}")

        return len(errors) == 0, errors


class ActionParser:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    async def parse_instruction(
        self,
        agent_name: str,
        instruction: str,
        agent: Optional[Any] = None,
        context: Optional[Any] = None
    ) -> Action:
        action = Action(
            agent_name=agent_name,
            instruction=instruction,
            context=context
        )

        action.action_type = self._detect_action_type(instruction)
        action.risk_level = self._detect_risk_level(agent_name, instruction, action.action_type)

        self._parse_generic_action(action)

        if agent and hasattr(agent, 'get_action_schema'):
            try:
                schema = await agent.get_action_schema()
                await self._enrich_with_agent_schema(action, schema)
            except Exception as e:
                if self.verbose:
                    print(f"[PARSER] Could not get agent schema: {e}")

        return action

    def _detect_action_type(self, instruction: str) -> ActionType:
        instruction_lower = instruction.lower()

        type_map = {
            ActionType.CREATE: ['create', 'add', 'make', 'new', 'write', 'generate'],
            ActionType.UPDATE: ['update', 'modify', 'change', 'edit', 'set', 'revise'],
            ActionType.DELETE: ['delete', 'remove', 'destroy', 'drop', 'purge'],
            ActionType.SEND: ['send', 'post', 'share', 'broadcast', 'message', 'reply'],
            ActionType.NOTIFY: ['notify', 'alert', 'announce', 'inform'],
            ActionType.ARCHIVE: ['archive', 'close', 'close out'],
        }

        for action_type, keywords in type_map.items():
            if any(kw in instruction_lower for kw in keywords):
                return action_type

        return ActionType.EXECUTE

    def _detect_risk_level(
        self,
        agent_name: str,
        instruction: str,
        action_type: ActionType
    ) -> RiskLevel:
        if action_type in [ActionType.DELETE, ActionType.ARCHIVE]:
            return RiskLevel.HIGH

        if action_type == ActionType.SEND:
            if any(word in instruction.lower() for word in ['everyone', '@channel', '@here']):
                return RiskLevel.HIGH

        if action_type in [ActionType.CREATE, ActionType.UPDATE]:
            return RiskLevel.MEDIUM

        return RiskLevel.LOW

    def _parse_generic_action(self, action: Action) -> None:
        instruction = action.instruction

        quoted_pattern = r'["\']([^"\']+)["\']'
        quoted_values = re.findall(quoted_pattern, instruction)

        kv_pattern = r'(\w+)[:\s]+["\']?([^"\';\n]+)["\']?'
        matches = re.finditer(kv_pattern, instruction)

        for match in matches:
            key, value = match.groups()
            key_lower = key.lower().strip()

            skip_words = {'the', 'with', 'in', 'to', 'for', 'from', 'and', 'or', 'at', 'by'}
            if key_lower in skip_words:
                continue

            value = value.strip()

            if key_lower not in action.parameters:
                action.parameters[key_lower] = value

                action.field_info[key_lower] = FieldInfo(
                    display_label=key.replace('_', ' ').title(),
                    description=f"Parameter: {key}",
                    field_type='string',
                    current_value=value,
                    editable=True
                )

    async def _enrich_with_agent_schema(
        self,
        action: Action,
        agent_schema: Dict[str, Any]
    ) -> None:
        action_type_key = str(action.action_type.value)
        if action_type_key not in agent_schema:
            return

        schema_for_type = agent_schema[action_type_key]
        schema_params = schema_for_type.get('parameters', {})

        for param_name, param_schema in schema_params.items():
            if param_name in action.parameters:
                editable = param_schema.get('editable', True)

                field_info = FieldInfo(
                    display_label=param_schema.get('display_label', param_name),
                    description=param_schema.get('description', ''),
                    field_type=param_schema.get('type', 'string'),
                    current_value=action.parameters[param_name],
                    editable=editable,
                    required=param_schema.get('required', True)
                )

                constraints_schema = param_schema.get('constraints', {})
                if constraints_schema:
                    field_info.constraints = FieldConstraint(
                        min_length=constraints_schema.get('min_length'),
                        max_length=constraints_schema.get('max_length'),
                        min_value=constraints_schema.get('min_value'),
                        max_value=constraints_schema.get('max_value'),
                        pattern=constraints_schema.get('pattern'),
                        allowed_values=constraints_schema.get('allowed_values'),
                        forbidden_values=constraints_schema.get('forbidden_values')
                    )

                action.field_info[param_name] = field_info


class ActionEnricher:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    async def enrich_action(self, action: Action, agent: Optional[Any] = None) -> None:
        try:
            is_valid, error = InputValidator.validate_instruction(action.instruction)
            if not is_valid:
                # warning(f"Invalid instruction: {error}")
                action.details = {'error': f'Invalid instruction: {error}'}
                return

            try:
                await asyncio.wait_for(
                    self._enrich_action_impl(action, agent),
                    timeout=Config.ENRICHMENT_TIMEOUT
                )
            except asyncio.TimeoutError:
                # warning(f"Enrichment timeout for {action.agent_name}")
                action.details = {'error': f'Enrichment timed out (>{Config.ENRICHMENT_TIMEOUT}s)'}
                if Config.REQUIRE_ENRICHMENT_FOR_HIGH_RISK and action.risk_level.value == 'high':
                    # error(f"Blocking HIGH-RISK action due to failed enrichment: {action.id}")

        except Exception as e:
            # error(f"Unexpected error during enrichment: {str(e)}", exc_info=True)
            action.details = {'error': f'Enrichment error: {str(e)}'}

    async def _enrich_action_impl(self, action: Action, agent: Optional[Any] = None) -> None:
        if action.agent_name == 'jira':
            await self._enrich_jira_action(action, agent)
        elif action.agent_name == 'slack':
            await self._enrich_slack_action(action, agent)
        elif action.agent_name == 'github':
            await self._enrich_github_action(action, agent)
        else:
            await self._enrich_generic_action(action, agent)

    async def _enrich_jira_action(self, action: Action, agent: Optional[Any] = None) -> None:
        if action.action_type == ActionType.DELETE:
            issue_keys = self._extract_jira_keys(action.instruction)
            if issue_keys:
                action.details = {
                    'issue_keys': issue_keys,
                    'count': len(issue_keys),
                    'description': f"Will permanently delete {len(issue_keys)} Jira issue(s)"
                }

                if agent and hasattr(agent, 'execute'):
                    try:
                        fetch_instruction = f"Get details for issues: {', '.join(issue_keys)}"
                        details_response = await agent.execute(fetch_instruction)
                        action.details['full_details'] = details_response
                    except Exception as e:
                        if self.verbose:
                            print(f"[ENRICHER] Could not fetch Jira details: {e}")

        elif action.action_type == ActionType.CREATE:
            action.details = {
                'description': "Will create a new Jira issue",
                'summary': action.parameters.get('title', 'Untitled'),
                'project': action.parameters.get('project', 'Unknown')
            }

        elif action.action_type == ActionType.UPDATE:
            issue_key = self._extract_jira_keys(action.instruction)
            action.details = {
                'description': f"Will update Jira issue(s): {', '.join(issue_key) if issue_key else 'Unknown'}",
                'changes': action.parameters
            }

    async def _enrich_slack_action(self, action: Action, agent: Optional[Any] = None) -> None:
        if action.action_type == ActionType.SEND:
            channel = InputValidator.extract_slack_channel_safe(action.instruction)
            if not channel:
                channel = 'Unknown'

            message_match = re.search(r'(?:message|text)[:\s]*["\']?([^"\']+)', action.instruction, re.IGNORECASE)
            message = message_match.group(1) if message_match else action.parameters.get('message', '')

            action.details = {
                'channel': channel,
                'message_preview': message[:100] + ('...' if len(message) > 100 else ''),
                'full_message': message,
                'description': f"Will send message to {channel}"
            }

    async def _enrich_github_action(self, action: Action, agent: Optional[Any] = None) -> None:
        if action.action_type == ActionType.SEND:
            action.details = {
                'type': 'comment',
                'description': 'Will post a comment',
                'preview': action.parameters.get('comment', '')[:100]
            }

        elif action.action_type == ActionType.DELETE:
            action.details = {
                'description': 'Will delete GitHub resource',
                'target': action.parameters.get('target', 'Unknown')
            }

    async def _enrich_generic_action(self, action: Action, agent: Optional[Any] = None) -> None:
        action.details = {
            'description': f"{action.action_type.value.upper()} operation on {action.agent_name}",
            'parameters': action.parameters
        }

    def _extract_jira_keys(self, text: str) -> List[str]:
        return InputValidator.extract_jira_keys_safe(text)

    def get_action_summary(self, action: Action) -> str:
        if not hasattr(action, 'details') or not action.details:
            return f"Action type: {action.action_type.value}"

        details = action.details
        return details.get('description', f"Action type: {action.action_type.value}")

    def get_action_context_lines(self, action: Action) -> List[str]:
        lines = []

        if not hasattr(action, 'details') or not action.details:
            return lines

        details = action.details

        if 'description' in details:
            lines.append(f"  {details['description']}")

        if 'issue_keys' in details:
            keys_str = ', '.join(details['issue_keys'])
            lines.append(f"  Issues: {keys_str}")

        if 'message_preview' in details:
            lines.append(f"  Message: {details['message_preview']}")

        if 'channel' in details:
            lines.append(f"  Channel: {details['channel']}")

        if 'full_details' in details:
            lines.append(f"\n  Details:")
            detail_text = details['full_details']
            if isinstance(detail_text, str):
                detail_lines = detail_text.split('\n')[:3]
                for line in detail_lines:
                    if line.strip():
                        lines.append(f"    {line.strip()}")

        return lines
