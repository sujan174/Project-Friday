"""Error Classification and Enhanced Messaging System"""

from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib
import re


class ErrorCategory(str, Enum):
    """Categories of errors with different handling strategies"""
    TRANSIENT = "transient"
    RATE_LIMIT = "rate_limit"
    CAPABILITY = "capability"
    PERMISSION = "permission"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


@dataclass
class ErrorClassification:
    """Complete error classification with recovery suggestions"""
    category: ErrorCategory
    is_retryable: bool
    explanation: str
    technical_details: Optional[str] = None
    suggestions: List[str] = None
    retry_delay_seconds: int = 0

    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []


class ErrorClassifier:
    """Intelligent error classification system"""

    CAPABILITY_PATTERNS = [
        'does not support',
        'cannot fetch',
        'not available',
        'api does not',
        'is not supported',
        'not implemented',
        'unsupported operation',
        'cannot provide',
        'unable to retrieve',
    ]

    PERMISSION_PATTERNS = [
        'permission denied',
        'forbidden',
        'unauthorized',
        '401',
        '403',
        'access denied',
        'private repository',
        'not found',
        '404',
        'insufficient permissions',
        'access token',
    ]

    RATE_LIMIT_PATTERNS = [
        'rate limit',
        'rate limited',
        'too many requests',
        'quota exceeded',
        '429',
        '503',
        'throttled',
        'back off',
    ]

    TRANSIENT_PATTERNS = [
        'timeout',
        'timed out',
        'connection',
        'network',
        'temporarily',
        '502',
        '504',
        'gateway',
        'temporary',
        'service unavailable',
    ]

    VALIDATION_PATTERNS = [
        'invalid input',
        'invalid parameter',
        'required field',
        'bad request',
        '400',
        'validation error',
        'malformed',
    ]

    @staticmethod
    def classify(error_msg: str, agent_name: Optional[str] = None) -> ErrorClassification:
        """Classify an error message and return handling strategy"""
        error_lower = error_msg.lower()

        if ErrorClassifier._matches_patterns(error_lower, ErrorClassifier.CAPABILITY_PATTERNS):
            return ErrorClassification(
                category=ErrorCategory.CAPABILITY,
                is_retryable=False,
                explanation="The underlying API or agent does not support this operation",
                technical_details=error_msg,
                suggestions=[
                    "• Try a different approach or agent",
                    "• Check the agent's stated capabilities",
                    "• Look for an alternative tool that supports this operation",
                ]
            )

        if ErrorClassifier._matches_patterns(error_lower, ErrorClassifier.PERMISSION_PATTERNS):
            return ErrorClassification(
                category=ErrorCategory.PERMISSION,
                is_retryable=False,
                explanation="Access denied - check permissions and resource status",
                technical_details=error_msg,
                suggestions=[
                    "• Verify the resource exists and is accessible",
                    "• Check if the repository/resource is private",
                    "• Verify API token has required permissions",
                    "• Confirm you have access to this resource",
                ]
            )

        if ErrorClassifier._matches_patterns(error_lower, ErrorClassifier.VALIDATION_PATTERNS):
            return ErrorClassification(
                category=ErrorCategory.VALIDATION,
                is_retryable=False,
                explanation="Invalid input provided - request cannot be processed",
                technical_details=error_msg,
                suggestions=[
                    "• Check the instruction syntax",
                    "• Verify all required parameters are provided",
                    "• Ensure values are in the correct format",
                ]
            )

        if ErrorClassifier._matches_patterns(error_lower, ErrorClassifier.RATE_LIMIT_PATTERNS):
            return ErrorClassification(
                category=ErrorCategory.RATE_LIMIT,
                is_retryable=True,
                explanation="API rate limit reached - will retry with delay",
                technical_details=error_msg,
                suggestions=[
                    "• Operation will be retried automatically",
                    "• Consider spacing out large operations",
                ],
                retry_delay_seconds=10
            )

        if ErrorClassifier._matches_patterns(error_lower, ErrorClassifier.TRANSIENT_PATTERNS):
            return ErrorClassification(
                category=ErrorCategory.TRANSIENT,
                is_retryable=True,
                explanation="Temporary network or service issue - will retry",
                technical_details=error_msg,
                suggestions=[
                    "• Operation will be retried automatically",
                    "• If it persists, the service may be down",
                ],
                retry_delay_seconds=2
            )

        return ErrorClassification(
            category=ErrorCategory.UNKNOWN,
            is_retryable=True,
            explanation="An unexpected error occurred",
            technical_details=error_msg,
            suggestions=[
                "• Check agent logs for more details",
                "• Verify the instruction format",
                "• Try with a simpler or more specific request",
            ]
        )

    @staticmethod
    def _matches_patterns(text: str, patterns: List[str]) -> bool:
        """Check if text matches any pattern"""
        return any(pattern in text for pattern in patterns)


def format_error_for_user(
    classification: ErrorClassification,
    agent_name: str,
    instruction: str,
    attempt_number: int = 1,
    max_attempts: int = 3
) -> str:
    """Format an error classification into a user-friendly message"""

    if classification.category == ErrorCategory.CAPABILITY:
        header = "❌ **Cannot perform this operation**"
    elif classification.category == ErrorCategory.PERMISSION:
        header = "🔐 **Access Denied**"
    elif classification.category == ErrorCategory.RATE_LIMIT:
        header = "⏳ **Rate Limited - Retrying**"
    elif classification.category == ErrorCategory.TRANSIENT:
        header = "⏳ **Temporary Issue - Retrying**"
    elif classification.category == ErrorCategory.VALIDATION:
        header = "❌ **Invalid Input**"
    else:
        header = "⚠️ **Error**"

    message = f"{header}\n\n"
    message += f"**What happened**: {classification.explanation}\n\n"

    if classification.is_retryable and attempt_number > 1:
        message += f"**Attempt**: {attempt_number}/{max_attempts}\n\n"

    if classification.suggestions:
        message += "**What you can try**:\n"
        for suggestion in classification.suggestions:
            message += f"{suggestion}\n"
        message += "\n"

    if classification.category == ErrorCategory.UNKNOWN:
        message += f"**Technical details**: {classification.technical_details}\n"

    return message.strip()


class DuplicateOperationDetector:
    """Detects when the same operation is being retried multiple times with identical failures"""

    def __init__(self, window_size: int = 5, similarity_threshold: float = 0.8):
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.operation_history: Dict[str, List[Dict[str, Any]]] = {}

    def _create_operation_hash(self, agent_name: str, instruction: str) -> str:
        """Create a hash of the operation"""
        key = f"{agent_name}:{instruction[:100]}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def _similarity_score(self, error1: str, error2: str) -> float:
        """Calculate similarity between two error messages (0.0 to 1.0)"""
        if error1 == error2:
            return 1.0

        e1 = error1.lower()
        e2 = error2.lower()

        if e1 in e2 or e2 in e1:
            return 0.9

        words1 = set(e1.split())
        words2 = set(e2.split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def track_operation(
        self,
        agent_name: str,
        instruction: str,
        error: str,
        success: bool = False
    ) -> None:
        """Track an operation attempt"""
        op_hash = self._create_operation_hash(agent_name, instruction)

        if op_hash not in self.operation_history:
            self.operation_history[op_hash] = []

        self.operation_history[op_hash].append({
            'error': error,
            'success': success
        })

        if len(self.operation_history[op_hash]) > self.window_size:
            self.operation_history[op_hash] = self.operation_history[op_hash][-self.window_size:]

    def detect_duplicate_failure(
        self,
        agent_name: str,
        instruction: str,
        current_error: str
    ) -> Tuple[bool, Optional[str]]:
        """Detect if this operation has failed multiple times identically or similarly"""
        op_hash = self._create_operation_hash(agent_name, instruction)

        if op_hash not in self.operation_history:
            return False, None

        history = self.operation_history[op_hash]

        if len(history) < 2:
            return False, None

        recent_failures = [h for h in history if not h['success']]

        if len(recent_failures) < 2:
            return False, None

        error_messages = [f['error'] for f in recent_failures]

        similar_count = 0
        for prev_error in error_messages[:-1]:
            similarity = self._similarity_score(current_error, prev_error)
            if similarity >= self.similarity_threshold:
                similar_count += 1

        if similar_count >= 1:
            num_attempts = len(recent_failures)
            return True, (
                f"This operation has been attempted {num_attempts} times with similar failures. "
                f"The {agent_name} agent appears to be stuck or unable to complete this task."
            )

        if len(set(error_messages)) > 1 and len(recent_failures) >= 3:
            return True, (
                f"The {agent_name} agent is giving conflicting responses for the same operation. "
                f"This suggests it's not properly tracking state or has a capability limitation."
            )

        return False, None

    def detect_inconsistent_responses(
        self,
        agent_name: str,
        instruction: str
    ) -> Tuple[bool, Optional[List[str]]]:
        """Detect if agent is giving inconsistent responses"""
        op_hash = self._create_operation_hash(agent_name, instruction)

        if op_hash not in self.operation_history:
            return False, None

        history = self.operation_history[op_hash]

        if len(history) < 3:
            return False, None

        recent = history[-5:]
        pattern = ['success' if h['success'] else 'failed' for h in recent]

        success_count = sum(1 for h in recent if h['success'])
        failed_count = len(recent) - success_count

        if success_count > 0 and failed_count > 0 and len(recent) >= 3:
            return True, pattern

        return False, None

    def get_duplicate_summary(
        self,
        agent_name: str,
        instruction: str
    ) -> Optional[str]:
        """Get a summary of duplicate/stuck operations"""
        op_hash = self._create_operation_hash(agent_name, instruction)

        if op_hash not in self.operation_history:
            return None

        history = self.operation_history[op_hash]
        total_attempts = len(history)
        success_count = sum(1 for h in history if h['success'])
        failed_count = total_attempts - success_count

        if total_attempts < 2:
            return None

        return (
            f"Attempt history: {total_attempts} total attempts "
            f"({success_count} successful, {failed_count} failed)\n"
            f"Status: This operation appears to be stuck or unstable"
        )


@dataclass
class EnhancedError:
    """Enhanced error message with context and suggestions"""
    agent_name: str
    error_type: str
    what_failed: str
    why_failed: str
    what_was_tried: Optional[str] = None
    suggestions: List[str] = None
    alternatives: List[str] = None

    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []
        if self.alternatives is None:
            self.alternatives = []

    def format(self) -> str:
        """Format as user-friendly message"""
        lines = [f"❌ **{self.agent_name.title()} Error**\n"]

        lines.append(f"**What failed:** {self.what_failed}")
        lines.append(f"**Why:** {self.why_failed}")

        if self.what_was_tried:
            lines.append(f"**Attempted:** `{self.what_was_tried}`")

        if self.alternatives:
            lines.append(f"\n**💡 Did you mean:**")
            for alt in self.alternatives[:5]:
                lines.append(f"  • `{alt}`")

        if self.suggestions:
            lines.append(f"\n**🔧 How to fix:**")
            for suggestion in self.suggestions:
                lines.append(f"  • {suggestion}")

        return "\n".join(lines)


class ErrorMessageEnhancer:
    """Enhances error messages from agents to be more helpful"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def enhance_github_error(
        self,
        error: Exception,
        instruction: str,
        available_context: Optional[Dict[str, Any]] = None
    ) -> EnhancedError:
        """Enhance GitHub agent errors"""
        error_str = str(error).lower()

        if 'not found' in error_str or '404' in error_str:
            path_match = re.search(r'([\w\-_/]+/[\w\-_/]+)', str(error))
            attempted_path = path_match.group(1) if path_match else "unknown"

            repo_match = re.search(r'([\w\-_]+/[\w\-_]+)', attempted_path)
            repo = repo_match.group(1) if repo_match else None

            if '/' in attempted_path and attempted_path.count('/') > 1:
                path_parts = attempted_path.split('/')
                wrong_path = '/'.join(path_parts[2:])

                return EnhancedError(
                    agent_name="GitHub",
                    error_type="path_not_found",
                    what_failed=f"Access folder/file in repository",
                    why_failed=f"The path `{wrong_path}` doesn't exist in this repository",
                    what_was_tried=attempted_path,
                    suggestions=[
                        "Check if the folder/file name is spelled correctly",
                        "Use `/list <repo>` to see available files and folders",
                        f"Try: list contents of {repo} first",
                        "Folder names are case-sensitive"
                    ],
                    alternatives=self._suggest_github_paths(wrong_path, available_context)
                )
            else:
                return EnhancedError(
                    agent_name="GitHub",
                    error_type="repo_not_found",
                    what_failed="Access GitHub repository",
                    why_failed=f"Repository `{attempted_path}` doesn't exist or is private",
                    what_was_tried=attempted_path,
                    suggestions=[
                        "Check the repository name (format: owner/repo)",
                        "Verify the repository is public or you have access",
                        "Check for typos in the owner or repo name"
                    ]
                )

        elif 'permission' in error_str or 'forbidden' in error_str or '403' in error_str:
            return EnhancedError(
                agent_name="GitHub",
                error_type="permission",
                what_failed="Access GitHub resource",
                why_failed="Your access token doesn't have the required permissions",
                suggestions=[
                    "Check if your GITHUB_PERSONAL_ACCESS_TOKEN has the right scopes",
                    "For private repos, token needs 'repo' scope",
                    "For public repos, token needs at least 'public_repo' scope",
                    "Regenerate token at: https://github.com/settings/tokens"
                ]
            )

        elif 'rate limit' in error_str or '429' in error_str:
            return EnhancedError(
                agent_name="GitHub",
                error_type="rate_limit",
                what_failed="Make GitHub API request",
                why_failed="GitHub API rate limit exceeded",
                suggestions=[
                    "Wait 60 seconds and try again",
                    "GitHub has a limit of 5000 requests/hour for authenticated users",
                    "Check current rate limit status at: https://api.github.com/rate_limit"
                ]
            )

        return EnhancedError(
            agent_name="GitHub",
            error_type="unknown",
            what_failed="Complete GitHub operation",
            why_failed=str(error),
            suggestions=[
                "Check the GitHub agent logs for more details",
                "Verify your GitHub token is valid",
                "Try the operation again"
            ]
        )

    def _suggest_github_paths(self, wrong_path: str, context: Optional[Dict] = None) -> List[str]:
        """Suggest similar GitHub paths based on common patterns"""
        suggestions = []

        corrections = {
            'middlewares': 'middleware',
            'controller': 'controllers',
            'model': 'models',
            'view': 'views',
            'util': 'utils',
            'helper': 'helpers',
            'lib': 'libs',
            'test': 'tests',
            'spec': 'specs',
            'doc': 'docs',
        }

        wrong_lower = wrong_path.lower()
        for wrong, correct in corrections.items():
            if wrong in wrong_lower:
                suggested = wrong_path.lower().replace(wrong, correct)
                suggestions.append(suggested)

        if context and 'available_paths' in context:
            available = context['available_paths']
            for path in available:
                if self._similarity(wrong_path.lower(), path.lower()) > 0.6:
                    suggestions.append(path)

        return suggestions[:5]

    def _similarity(self, a: str, b: str) -> float:
        """Calculate simple string similarity (0.0 - 1.0)"""
        if a == b:
            return 1.0

        if a in b or b in a:
            return 0.8

        matches = sum(1 for c1, c2 in zip(a, b) if c1 == c2)
        max_len = max(len(a), len(b))

        return matches / max_len if max_len > 0 else 0.0

    def enhance_jira_error(self, error: Exception, instruction: str) -> EnhancedError:
        """Enhance Jira agent errors"""
        error_str = str(error).lower()

        if 'not found' in error_str or '404' in error_str:
            issue_match = re.search(r'([A-Z]+-\d+)', str(error), re.IGNORECASE)
            issue_key = issue_match.group(1) if issue_match else "unknown"

            return EnhancedError(
                agent_name="Jira",
                error_type="not_found",
                what_failed="Find Jira issue",
                why_failed=f"Issue `{issue_key}` doesn't exist or you don't have access",
                what_was_tried=issue_key,
                suggestions=[
                    f"Check if {issue_key} is the correct issue key",
                    "Verify you have access to this Jira project",
                    "Use: search for issues in [project] to find correct key"
                ]
            )

        elif 'authentication' in error_str or 'unauthorized' in error_str:
            return EnhancedError(
                agent_name="Jira",
                error_type="authentication",
                what_failed="Authenticate with Jira",
                why_failed="Invalid Jira credentials or token expired",
                suggestions=[
                    "Check your JIRA_API_TOKEN in .env",
                    "Verify JIRA_EMAIL matches your Atlassian account",
                    "Generate new API token at: https://id.atlassian.com/manage-profile/security/api-tokens"
                ]
            )

        return EnhancedError(
            agent_name="Jira",
            error_type="unknown",
            what_failed="Complete Jira operation",
            why_failed=str(error),
            suggestions=["Check Jira agent logs", "Verify your Jira credentials"]
        )

    def enhance_slack_error(self, error: Exception, instruction: str) -> EnhancedError:
        """Enhance Slack agent errors"""
        error_str = str(error).lower()

        if 'channel_not_found' in error_str:
            channel_match = re.search(r'#([\w\-]+)', instruction)
            channel = channel_match.group(1) if channel_match else "unknown"

            return EnhancedError(
                agent_name="Slack",
                error_type="not_found",
                what_failed="Find Slack channel",
                why_failed=f"Channel `#{channel}` doesn't exist or bot isn't a member",
                what_was_tried=f"#{channel}",
                suggestions=[
                    "Check if channel name is spelled correctly",
                    f"Invite bot to #{channel} first",
                    "Use: list channels to see available channels"
                ]
            )

        elif 'not_in_channel' in error_str:
            return EnhancedError(
                agent_name="Slack",
                error_type="permission",
                what_failed="Access Slack channel",
                why_failed="Bot is not a member of this channel",
                suggestions=[
                    "Invite the bot to the channel first: /invite @bot-name",
                    "Or make the channel public and try again"
                ]
            )

        return EnhancedError(
            agent_name="Slack",
            error_type="unknown",
            what_failed="Complete Slack operation",
            why_failed=str(error),
            suggestions=["Check Slack bot permissions", "Verify bot token is valid"]
        )

    def enhance_notion_error(self, error: Exception, instruction: str) -> EnhancedError:
        """Enhance Notion agent errors"""
        error_str = str(error).lower()

        if 'not found' in error_str or '404' in error_str:
            return EnhancedError(
                agent_name="Notion",
                error_type="not_found",
                what_failed="Find Notion page or database",
                why_failed="Page/database doesn't exist or integration doesn't have access",
                suggestions=[
                    "Share the page/database with your Notion integration",
                    "Check if the page ID is correct",
                    "Use: list pages to see accessible pages"
                ]
            )

        elif 'unauthorized' in error_str:
            return EnhancedError(
                agent_name="Notion",
                error_type="permission",
                what_failed="Access Notion resource",
                why_failed="Notion integration doesn't have required permissions",
                suggestions=[
                    "Share the page with your integration",
                    "Check NOTION_API_KEY is valid",
                    "Regenerate token at: https://www.notion.so/my-integrations"
                ]
            )

        return EnhancedError(
            agent_name="Notion",
            error_type="unknown",
            what_failed="Complete Notion operation",
            why_failed=str(error),
            suggestions=["Check Notion integration settings", "Verify API key is valid"]
        )

    def enhance_error(
        self,
        agent_name: str,
        error: Exception,
        instruction: str,
        context: Optional[Dict[str, Any]] = None
    ) -> EnhancedError:
        """Main method to enhance any agent error"""
        agent_lower = agent_name.lower()

        if agent_lower == 'github':
            return self.enhance_github_error(error, instruction, context)
        elif agent_lower == 'jira':
            return self.enhance_jira_error(error, instruction)
        elif agent_lower == 'slack':
            return self.enhance_slack_error(error, instruction)
        elif agent_lower == 'notion':
            return self.enhance_notion_error(error, instruction)
        else:
            return EnhancedError(
                agent_name=agent_name.title(),
                error_type="unknown",
                what_failed=f"Complete {agent_name} operation",
                why_failed=str(error),
                suggestions=[
                    f"Check {agent_name} agent logs for details",
                    "Verify your credentials and permissions",
                    "Try the operation again"
                ]
            )
