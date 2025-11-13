"""Agent Intelligence Components - Shared Intelligence Infrastructure"""

import os
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


class ConversationMemory:
    """Remember recent operations and resolve ambiguous references"""

    def __init__(self, max_history: int = 10):
        self.recent_operations = []
        self.max_history = max_history
        self.current_context = {}

    def remember(self, operation_type: str, resource_id: str, details: Dict):
        self.recent_operations.append({
            'type': operation_type,
            'id': resource_id,
            'details': details,
            'timestamp': datetime.now()
        })

        if len(self.recent_operations) > self.max_history:
            self.recent_operations.pop(0)

        self.current_context = {
            'last_resource': resource_id,
            'last_operation': operation_type,
            'last_details': details
        }

    def resolve_reference(self, phrase: str) -> Optional[str]:
        phrase_lower = phrase.lower().strip()

        ambiguous_terms = [
            'it', 'that', 'this', 'the issue', 'the ticket',
            'the pr', 'the page', 'the message', 'them'
        ]

        if phrase_lower in ambiguous_terms:
            if self.recent_operations:
                return self.recent_operations[-1]['id']

        return None

    def get_recent(self, count: int = 5) -> List[Dict]:
        return self.recent_operations[-count:] if self.recent_operations else []

    def get_last_of_type(self, operation_type: str) -> Optional[Dict]:
        for op in reversed(self.recent_operations):
            if op['type'] == operation_type:
                return op
        return None


class WorkspaceKnowledge:
    """Persistent workspace-specific knowledge base"""

    def __init__(self, knowledge_file: str = 'data/workspace_knowledge.json'):
        self.knowledge_file = knowledge_file
        self.data = self._load()

    def _load(self) -> Dict:
        try:
            if os.path.exists(self.knowledge_file):
                with open(self.knowledge_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"[KNOWLEDGE] Warning: Could not load knowledge file: {e}")

        return {
            'projects': {},
            'user_preferences': {},
            'error_solutions': {},
            'patterns': {},
            'metadata_caches': {},
            'version': '1.0'
        }

    def _save(self):
        try:
            with open(self.knowledge_file, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"[KNOWLEDGE] Warning: Could not save knowledge: {e}")

    def learn_project_config(self, project: str, key: str, value: Any):
        if project not in self.data['projects']:
            self.data['projects'][project] = {}

        self.data['projects'][project][key] = value
        self._save()

        print(f"[KNOWLEDGE] Learned: {project}.{key} = {value}")

    def get_project_config(self, project: str, key: str, default=None) -> Any:
        return self.data['projects'].get(project, {}).get(key, default)

    def learn_error_solution(self, error_type: str, context: Dict, solution: Dict):
        error_key = f"{error_type}:{context.get('operation', 'unknown')}"

        self.data['error_solutions'][error_key] = {
            'error_type': error_type,
            'context': context,
            'solution': solution,
            'learned_at': datetime.now().isoformat(),
            'success_count': self.data['error_solutions'].get(error_key, {}).get('success_count', 0) + 1
        }

        self._save()

    def get_error_solution(self, error_type: str, context: Dict) -> Optional[Dict]:
        error_key = f"{error_type}:{context.get('operation', 'unknown')}"
        return self.data['error_solutions'].get(error_key)

    def learn_user_preference(self, key: str, value: Any):
        self.data['user_preferences'][key] = value
        self._save()

    def get_user_preference(self, key: str, default=None) -> Any:
        return self.data['user_preferences'].get(key, default)

    def save_metadata_cache(self, agent_name: str, metadata: Dict, ttl_seconds: int = 3600):
        self.data['metadata_caches'][agent_name] = {
            'data': metadata,
            'cached_at': datetime.now().isoformat(),
            'ttl_seconds': ttl_seconds
        }
        self._save()

        if hasattr(self, 'verbose') and self.verbose:
            print(f"[KNOWLEDGE] Cached metadata for {agent_name} (TTL: {ttl_seconds}s)")

    def get_metadata_cache(self, agent_name: str) -> Optional[Dict]:
        if agent_name not in self.data['metadata_caches']:
            return None

        cache_entry = self.data['metadata_caches'][agent_name]
        cached_at = datetime.fromisoformat(cache_entry['cached_at'])
        ttl = cache_entry.get('ttl_seconds', 3600)

        age_seconds = (datetime.now() - cached_at).total_seconds()
        if age_seconds > ttl:
            if hasattr(self, 'verbose') and self.verbose:
                print(f"[KNOWLEDGE] Metadata cache for {agent_name} expired (age: {age_seconds:.0f}s)")
            return None

        return cache_entry['data']

    def invalidate_metadata_cache(self, agent_name: str):
        if agent_name in self.data['metadata_caches']:
            del self.data['metadata_caches'][agent_name]
            self._save()


@dataclass
class SharedContext:
    """Shared context between agents during orchestration session"""

    session_id: str
    created_resources: Dict[str, Dict] = field(default_factory=dict)
    agent_messages: List[Dict] = field(default_factory=list)

    def share_resource(
        self,
        agent_name: str,
        resource_type: str,
        resource_id: str,
        url: str,
        details: Optional[Dict] = None
    ):
        key = f"{agent_name}:{resource_type}:{resource_id}"

        self.created_resources[key] = {
            'agent': agent_name,
            'type': resource_type,
            'id': resource_id,
            'url': url,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        }

    def get_resources_by_type(self, resource_type: str) -> List[Dict]:
        return [
            r for key, r in self.created_resources.items()
            if r['type'] == resource_type
        ]

    def get_resources_by_agent(self, agent_name: str) -> List[Dict]:
        return [
            r for key, r in self.created_resources.items()
            if r['agent'] == agent_name
        ]

    def get_all_resources(self) -> List[Dict]:
        return list(self.created_resources.values())

    def get_recent_resources(self, limit: int = 10) -> List[Dict]:
        all_resources = list(self.created_resources.values())

        all_resources.sort(
            key=lambda r: r.get('timestamp', ''),
            reverse=True
        )

        return all_resources[:limit]

    def add_message(self, agent_name: str, message: str):
        self.agent_messages.append({
            'agent': agent_name,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })


class ProactiveAssistant:
    """Proactive suggestions and validation"""

    def __init__(self, agent_name: str, verbose: bool = False):
        self.agent_name = agent_name
        self.verbose = verbose

    def suggest_next_steps(self, operation: str, context: Dict) -> List[str]:
        suggestions = []

        if self.agent_name == 'jira':
            if operation == 'create_issue':
                suggestions.extend([
                    "Assign to a team member?",
                    "Add to current sprint?",
                    "Link to related issues?"
                ])

                if context.get('priority') in ['Critical', 'High']:
                    suggestions.append("⚠️ Notify team about critical issue?")

            elif operation == 'transition_issue':
                if context.get('target_status') == 'Done':
                    suggestions.append("Create release notes entry?")

        elif self.agent_name == 'github':
            if operation == 'create_pr':
                suggestions.extend([
                    "Request review from team lead?",
                    "Link to related issues?"
                ])

                if not context.get('has_tests'):
                    suggestions.append("⚠️ Consider adding tests to this PR")

            elif operation == 'create_issue':
                suggestions.append("Create feature branch for this issue?")

        elif self.agent_name == 'slack':
            if operation == 'post_message':
                if context.get('is_announcement'):
                    suggestions.append("Pin this message for visibility?")

        elif self.agent_name == 'notion':
            if operation == 'create_page':
                suggestions.extend([
                    "Share with team members?",
                    "Add to relevant database?"
                ])

        return suggestions

    def validate_before_execute(self, operation: str, context: Dict) -> Optional[str]:
        if operation in ['delete', 'remove', 'close']:
            return "⚠️ This is a destructive operation. Are you sure?"

        if self.agent_name == 'github':
            if operation == 'merge_pr':
                if not context.get('has_reviews'):
                    return "⚠️ PR has no reviews. Merge anyway?"

                if context.get('has_failing_checks'):
                    return "⚠️ Some checks are failing. Merging is not recommended."

        elif self.agent_name == 'jira':
            if operation == 'transition_issue':
                required_fields = context.get('required_fields', [])
                if required_fields:
                    return f"⚠️ This transition may require: {', '.join(required_fields)}"

        return None

    def check_for_duplicates(self, resource_type: str, query: str) -> Optional[str]:
        return None
