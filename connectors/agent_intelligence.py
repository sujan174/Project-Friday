"""
Agent Intelligence Components - Shared Intelligence Infrastructure

This module provides intelligence components that make agents smarter:
- ConversationMemory: Remember recent operations and resolve references
- WorkspaceKnowledge: Learn and persist workspace-specific knowledge
- SharedContext: Enable cross-agent coordination
- ProactiveAssistant: Suggest next steps and validate operations

All production agents use these components to provide intelligent,
context-aware assistance.

Author: AI System
Version: 1.0
"""

import os
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import OrderedDict


# ============================================================================
# CACHE METRICS
# ============================================================================

class CacheMetrics:
    """
    Track cache hit/miss statistics for performance optimization.

    Logs metrics to intelligence logs for analysis.
    """

    def __init__(self):
        self.hits: Dict[str, int] = {}
        self.misses: Dict[str, int] = {}
        self.evictions: Dict[str, int] = {}
        self.invalidations: Dict[str, int] = {}

    def record_hit(self, agent_name: str):
        """Record a cache hit"""
        self.hits[agent_name] = self.hits.get(agent_name, 0) + 1

    def record_miss(self, agent_name: str):
        """Record a cache miss"""
        self.misses[agent_name] = self.misses.get(agent_name, 0) + 1

    def record_eviction(self, agent_name: str):
        """Record a cache eviction"""
        self.evictions[agent_name] = self.evictions.get(agent_name, 0) + 1

    def record_invalidation(self, agent_name: str):
        """Record a cache invalidation"""
        self.invalidations[agent_name] = self.invalidations.get(agent_name, 0) + 1

    def get_hit_rate(self, agent_name: str) -> float:
        """Get cache hit rate for an agent"""
        hits = self.hits.get(agent_name, 0)
        misses = self.misses.get(agent_name, 0)
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0

    def get_stats(self, agent_name: str = None) -> Dict:
        """Get cache statistics"""
        if agent_name:
            return {
                'hits': self.hits.get(agent_name, 0),
                'misses': self.misses.get(agent_name, 0),
                'evictions': self.evictions.get(agent_name, 0),
                'invalidations': self.invalidations.get(agent_name, 0),
                'hit_rate': f"{self.get_hit_rate(agent_name):.1f}%"
            }
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'invalidations': self.invalidations
        }

    def to_dict(self) -> Dict:
        """Convert metrics to dict for logging"""
        return {
            'hits': dict(self.hits),
            'misses': dict(self.misses),
            'evictions': dict(self.evictions),
            'invalidations': dict(self.invalidations),
            'hit_rates': {
                agent: f"{self.get_hit_rate(agent):.1f}%"
                for agent in set(list(self.hits.keys()) + list(self.misses.keys()))
            }
        }


# Global cache metrics instance
_cache_metrics = CacheMetrics()

def get_cache_metrics() -> CacheMetrics:
    """Get the global cache metrics instance"""
    return _cache_metrics


# ============================================================================
# CONVERSATION MEMORY
# ============================================================================

class ConversationMemory:
    """
    Remember recent operations and resolve ambiguous references

    Enables natural conversation flow where users can say:
    "Create an issue... assign it to John... add it to sprint 5"
    without repeating the issue key each time.
    """

    def __init__(self, max_history: int = 10):
        """
        Initialize conversation memory

        Args:
            max_history: Maximum number of operations to remember (default: 10)
        """
        self.recent_operations = []  # List of recent operations
        self.max_history = max_history
        self.current_context = {}  # Current active context

    def remember(self, operation_type: str, resource_id: str, details: Dict):
        """
        Remember an operation that just happened

        Args:
            operation_type: Type of operation (e.g., 'create_issue', 'create_pr')
            resource_id: ID of the resource (e.g., 'KAN-50', 'PR #123')
            details: Additional details about the operation
        """
        self.recent_operations.append({
            'type': operation_type,
            'id': resource_id,
            'details': details,
            'timestamp': datetime.now()
        })

        # Keep only recent history
        if len(self.recent_operations) > self.max_history:
            self.recent_operations.pop(0)

        # Update current context
        self.current_context = {
            'last_resource': resource_id,
            'last_operation': operation_type,
            'last_details': details
        }

    def resolve_reference(self, phrase: str) -> Optional[str]:
        """
        Resolve ambiguous references like 'it', 'that', 'this'

        Args:
            phrase: The phrase to resolve

        Returns:
            The resource ID being referenced, or None if can't resolve
        """
        phrase_lower = phrase.lower().strip()

        # Common ambiguous references
        ambiguous_terms = [
            'it', 'that', 'this', 'the issue', 'the ticket',
            'the pr', 'the page', 'the message', 'them'
        ]

        if phrase_lower in ambiguous_terms:
            if self.recent_operations:
                # Return most recent resource
                return self.recent_operations[-1]['id']

        return None

    def get_recent(self, count: int = 5) -> List[Dict]:
        """
        Get recent operations

        Args:
            count: Number of recent operations to return

        Returns:
            List of recent operations
        """
        return self.recent_operations[-count:] if self.recent_operations else []

    def get_last_of_type(self, operation_type: str) -> Optional[Dict]:
        """
        Get the most recent operation of a specific type

        Args:
            operation_type: Type of operation to find

        Returns:
            The operation dict, or None if not found
        """
        for op in reversed(self.recent_operations):
            if op['type'] == operation_type:
                return op
        return None


# ============================================================================
# WORKSPACE KNOWLEDGE
# ============================================================================

class WorkspaceKnowledge:
    """
    Persistent workspace-specific knowledge base

    Learns from operations and errors to become smarter over time:
    - "KAN project uses 'Task' not 'Bug'"
    - "Security issues always assigned to @security-team"
    - "Critical bugs get #critical-bugs notification"

    Features LRU eviction and cache metrics tracking.
    """

    # Default max entries per cache type for LRU eviction
    DEFAULT_MAX_CACHE_ENTRIES = {
        'jira': 100,
        'slack': 200,
        'github': 100,
        'notion': 100,
        'google_calendar': 50,
        'browser': 20,
        'scraper': 50,
        'code_reviewer': 30,
        'default': 50
    }

    def __init__(self, knowledge_file: str = 'data/workspace_knowledge.json', verbose: bool = False):
        """
        Initialize workspace knowledge

        Args:
            knowledge_file: Path to persistent knowledge file
            verbose: Enable verbose logging
        """
        self.knowledge_file = knowledge_file
        self.verbose = verbose
        self.data = self._load()
        self.metrics = get_cache_metrics()

    def _load(self) -> Dict:
        """Load knowledge from disk"""
        try:
            if os.path.exists(self.knowledge_file):
                with open(self.knowledge_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"[KNOWLEDGE] Warning: Could not load knowledge file: {e}")

        # Return default structure
        return {
            'projects': {},
            'user_preferences': {},
            'error_solutions': {},
            'patterns': {},
            'metadata_caches': {},  # Feature #1: Store agent metadata caches
            'version': '1.0'
        }

    def _save(self):
        """Persist knowledge to disk"""
        try:
            # Create parent directory if it doesn't exist
            parent_dir = os.path.dirname(self.knowledge_file)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

            with open(self.knowledge_file, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"[KNOWLEDGE] Warning: Could not save knowledge: {e}")

    def learn_project_config(self, project: str, key: str, value: Any):
        """
        Learn project-specific configuration

        Args:
            project: Project identifier (e.g., 'KAN')
            key: Configuration key (e.g., 'valid_issue_types')
            value: Configuration value
        """
        if project not in self.data['projects']:
            self.data['projects'][project] = {}

        self.data['projects'][project][key] = value
        self._save()

        print(f"[KNOWLEDGE] Learned: {project}.{key} = {value}")

    def get_project_config(self, project: str, key: str, default=None) -> Any:
        """
        Get learned project configuration

        Args:
            project: Project identifier
            key: Configuration key
            default: Default value if not found

        Returns:
            The configuration value, or default if not found
        """
        return self.data['projects'].get(project, {}).get(key, default)

    def learn_error_solution(self, error_type: str, context: Dict, solution: Dict):
        """
        Learn solution for a specific error

        Args:
            error_type: Type of error encountered
            context: Context when error occurred
            solution: Solution that worked
        """
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
        """
        Get known solution for an error

        Args:
            error_type: Type of error
            context: Current context

        Returns:
            Known solution, or None if not found
        """
        error_key = f"{error_type}:{context.get('operation', 'unknown')}"
        return self.data['error_solutions'].get(error_key)

    def learn_user_preference(self, key: str, value: Any):
        """
        Learn user preference

        Args:
            key: Preference key
            value: Preference value
        """
        self.data['user_preferences'][key] = value
        self._save()

    def get_user_preference(self, key: str, default=None) -> Any:
        """Get user preference"""
        return self.data['user_preferences'].get(key, default)

    def save_metadata_cache(self, agent_name: str, metadata: Dict, ttl_seconds: int = 3600, sub_key: str = None):
        """
        Save agent metadata cache with TTL and LRU eviction (Feature #1)

        Args:
            agent_name: Name of the agent (e.g., 'jira', 'slack')
            metadata: Metadata dictionary to cache
            ttl_seconds: Time-to-live in seconds (default: 1 hour)
            sub_key: Optional sub-key for partial caching (e.g., 'transitions', 'recent_issues')
        """
        # Initialize agent cache if doesn't exist
        if agent_name not in self.data['metadata_caches']:
            self.data['metadata_caches'][agent_name] = {}

        # Determine cache key
        cache_key = sub_key if sub_key else '_main'

        # Get max entries for this agent
        max_entries = self.DEFAULT_MAX_CACHE_ENTRIES.get(
            agent_name,
            self.DEFAULT_MAX_CACHE_ENTRIES['default']
        )

        # Check if we need LRU eviction
        agent_cache = self.data['metadata_caches'][agent_name]
        if isinstance(agent_cache, dict) and len(agent_cache) >= max_entries:
            # Evict oldest entry (by cached_at timestamp)
            oldest_key = None
            oldest_time = None
            for key, value in agent_cache.items():
                if isinstance(value, dict) and 'cached_at' in value:
                    cached_time = value['cached_at']
                    if oldest_time is None or cached_time < oldest_time:
                        oldest_time = cached_time
                        oldest_key = key

            if oldest_key:
                del agent_cache[oldest_key]
                self.metrics.record_eviction(agent_name)
                if self.verbose:
                    print(f"[KNOWLEDGE] LRU eviction: removed {oldest_key} from {agent_name} cache")

        # Store the cache entry
        self.data['metadata_caches'][agent_name][cache_key] = {
            'data': metadata,
            'cached_at': datetime.now().isoformat(),
            'ttl_seconds': ttl_seconds
        }
        self._save()

        if self.verbose:
            print(f"[KNOWLEDGE] Cached metadata for {agent_name}/{cache_key} (TTL: {ttl_seconds}s)")

    def get_metadata_cache(self, agent_name: str, sub_key: str = None) -> Optional[Dict]:
        """
        Get cached metadata for an agent if still valid (Feature #1)

        Args:
            agent_name: Name of the agent
            sub_key: Optional sub-key for partial cache retrieval

        Returns:
            Cached metadata dict, or None if expired/not found
        """
        if agent_name not in self.data['metadata_caches']:
            self.metrics.record_miss(agent_name)
            return None

        agent_cache = self.data['metadata_caches'][agent_name]

        # Handle legacy format (direct data storage)
        if 'data' in agent_cache and 'cached_at' in agent_cache:
            # Old format - convert to new format
            cache_entry = agent_cache
        else:
            # New format with sub-keys
            cache_key = sub_key if sub_key else '_main'
            if cache_key not in agent_cache:
                self.metrics.record_miss(agent_name)
                return None
            cache_entry = agent_cache[cache_key]

        # Validate cache entry structure
        if not isinstance(cache_entry, dict) or 'cached_at' not in cache_entry:
            self.metrics.record_miss(agent_name)
            return None

        cached_at = datetime.fromisoformat(cache_entry['cached_at'])
        ttl = cache_entry.get('ttl_seconds', 3600)

        # Check if cache is still valid
        age_seconds = (datetime.now() - cached_at).total_seconds()
        if age_seconds > ttl:
            # Cache expired
            self.metrics.record_miss(agent_name)
            if self.verbose:
                print(f"[KNOWLEDGE] Metadata cache for {agent_name} expired (age: {age_seconds:.0f}s)")
            return None

        # Cache hit!
        self.metrics.record_hit(agent_name)
        return cache_entry['data']

    def invalidate_metadata_cache(self, agent_name: str, sub_key: str = None):
        """
        Invalidate (delete) cached metadata for an agent

        Args:
            agent_name: Name of the agent
            sub_key: Optional sub-key to invalidate only a portion of cache
        """
        if agent_name not in self.data['metadata_caches']:
            return

        if sub_key:
            # Partial invalidation
            agent_cache = self.data['metadata_caches'][agent_name]
            if isinstance(agent_cache, dict) and sub_key in agent_cache:
                del agent_cache[sub_key]
                self.metrics.record_invalidation(agent_name)
                if self.verbose:
                    print(f"[KNOWLEDGE] Invalidated {agent_name}/{sub_key} cache")
        else:
            # Full invalidation
            del self.data['metadata_caches'][agent_name]
            self.metrics.record_invalidation(agent_name)
            if self.verbose:
                print(f"[KNOWLEDGE] Invalidated all {agent_name} cache")

        self._save()

    def get_cache_stats(self, agent_name: str = None) -> Dict:
        """
        Get cache statistics for monitoring

        Args:
            agent_name: Optional agent name to filter stats

        Returns:
            Dict with cache statistics
        """
        return self.metrics.get_stats(agent_name)


# ============================================================================
# SHARED CONTEXT (Cross-Agent Coordination)
# ============================================================================

@dataclass
class SharedContext:
    """
    Shared context between agents during orchestration session

    Enables cross-agent coordination:
    - GitHub creates issue #123
    - Jira sees it and auto-links in ticket description
    - Slack sees both and includes links in message
    - Notion creates incident page with all references
    """

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
        """
        Agent shares a resource it created

        Args:
            agent_name: Name of the agent (e.g., 'github', 'jira')
            resource_type: Type of resource (e.g., 'issue', 'ticket', 'page')
            resource_id: Resource identifier (e.g., '#123', 'KAN-50')
            url: URL to the resource
            details: Additional details about the resource
        """
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
        """
        Get all resources of specific type from this session

        Args:
            resource_type: Type of resource to find

        Returns:
            List of matching resources
        """
        return [
            r for key, r in self.created_resources.items()
            if r['type'] == resource_type
        ]

    def get_resources_by_agent(self, agent_name: str) -> List[Dict]:
        """
        Get all resources created by specific agent

        Args:
            agent_name: Name of the agent

        Returns:
            List of resources created by that agent
        """
        return [
            r for key, r in self.created_resources.items()
            if r['agent'] == agent_name
        ]

    def get_all_resources(self) -> List[Dict]:
        """Get all resources created in this session"""
        return list(self.created_resources.values())

    def get_recent_resources(self, limit: int = 10) -> List[Dict]:
        """
        Get most recent resources from this session

        Args:
            limit: Maximum number of recent resources to return (default: 10)

        Returns:
            List of most recent resources, sorted by timestamp
        """
        all_resources = list(self.created_resources.values())

        # Sort by timestamp (most recent first)
        all_resources.sort(
            key=lambda r: r.get('timestamp', ''),
            reverse=True
        )

        # Return only the most recent N resources
        return all_resources[:limit]

    def add_message(self, agent_name: str, message: str):
        """Record a message from an agent"""
        self.agent_messages.append({
            'agent': agent_name,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })


# ============================================================================
# PROACTIVE ASSISTANT
# ============================================================================

class ProactiveAssistant:
    """
    Proactive suggestions and validation

    Helps users by:
    - Suggesting next steps after operations
    - Validating operations before execution
    - Warning about potential issues
    """

    def __init__(self, agent_name: str, verbose: bool = False):
        """
        Initialize proactive assistant

        Args:
            agent_name: Name of the agent this assistant helps
            verbose: Enable verbose logging
        """
        self.agent_name = agent_name
        self.verbose = verbose

    def suggest_next_steps(self, operation: str, context: Dict) -> List[str]:
        """
        Suggest logical next steps after an operation

        Args:
            operation: Operation that was just completed
            context: Context of the operation

        Returns:
            List of suggested next steps
        """
        suggestions = []

        # Jira-specific suggestions
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

        # GitHub-specific suggestions
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

        # Slack-specific suggestions
        elif self.agent_name == 'slack':
            if operation == 'post_message':
                if context.get('is_announcement'):
                    suggestions.append("Pin this message for visibility?")

        # Notion-specific suggestions
        elif self.agent_name == 'notion':
            if operation == 'create_page':
                suggestions.extend([
                    "Share with team members?",
                    "Add to relevant database?"
                ])

        return suggestions

    def validate_before_execute(self, operation: str, context: Dict) -> Optional[str]:
        """
        Validate operation before execution

        Args:
            operation: Operation about to be executed
            context: Operation context

        Returns:
            Warning message if issues found, None otherwise
        """
        # Common validations
        if operation in ['delete', 'remove', 'close']:
            return "⚠️ This is a destructive operation. Are you sure?"

        # GitHub-specific validations
        if self.agent_name == 'github':
            if operation == 'merge_pr':
                if not context.get('has_approvals'):
                    return "⚠️ PR has no approvals. Merge anyway?"

                if context.get('has_failing_checks'):
                    return "⚠️ Some checks are failing. Merging is not recommended."

        # Jira-specific validations
        elif self.agent_name == 'jira':
            if operation == 'transition_issue':
                required_fields = context.get('required_fields', [])
                if required_fields:
                    return f"⚠️ This transition may require: {', '.join(required_fields)}"

        return None

    def check_for_duplicates(self, resource_type: str, query: str) -> Optional[str]:
        """
        Warn if similar resources might exist

        Args:
            resource_type: Type of resource being created
            query: Search query to check for duplicates

        Returns:
            Warning message if potential duplicates found
        """
        # This is a placeholder - actual implementation would search
        # for similar resources and warn the user
        return None
