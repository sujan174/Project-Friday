"""
Enhanced Error Messaging System

Provides clear, actionable error messages with:
- Root cause explanation
- What actually happened vs what was expected
- Suggestions for fixing the issue
- Similar/alternative options when available

Author: AI System
Version: 1.0
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import re


@dataclass
class EnhancedError:
    """Enhanced error message with context and suggestions"""
    agent_name: str
    error_type: str  # 'not_found', 'permission', 'validation', etc.
    what_failed: str  # What the user tried to do
    why_failed: str  # Root cause explanation
    what_was_tried: Optional[str] = None  # What specific value/path was tried
    suggestions: List[str] = None  # Actionable suggestions
    alternatives: List[str] = None  # Alternative options found

    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []
        if self.alternatives is None:
            self.alternatives = []

    def format(self) -> str:
        """Format as user-friendly message"""
        lines = [f"❌ **{self.agent_name.title()} Error**\n"]

        # What failed
        lines.append(f"**What failed:** {self.what_failed}")

        # Why it failed
        lines.append(f"**Why:** {self.why_failed}")

        # What was tried (if available)
        if self.what_was_tried:
            lines.append(f"**Attempted:** `{self.what_was_tried}`")

        # Alternatives (if found)
        if self.alternatives:
            lines.append(f"\n**💡 Did you mean:**")
            for alt in self.alternatives[:5]:  # Show top 5
                lines.append(f"  • `{alt}`")

        # Suggestions
        if self.suggestions:
            lines.append(f"\n**🔧 How to fix:**")
            for suggestion in self.suggestions:
                lines.append(f"  • {suggestion}")

        return "\n".join(lines)


class ErrorMessageEnhancer:
    """
    Enhances error messages from agents to be more helpful.

    Features:
    - Extracts key information from raw errors
    - Suggests similar alternatives
    - Provides actionable fixes
    - Detects common issues
    """

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

        # Pattern: Directory/file not found
        if 'not found' in error_str or '404' in error_str:
            # Extract what was being searched for
            path_match = re.search(r'([\w\-_/]+/[\w\-_/]+)', str(error))
            attempted_path = path_match.group(1) if path_match else "unknown"

            # Extract repo and path
            repo_match = re.search(r'([\w\-_]+/[\w\-_]+)', attempted_path)
            repo = repo_match.group(1) if repo_match else None

            # Determine if it's a repo or path issue
            if '/' in attempted_path and attempted_path.count('/') > 1:
                # It's a path within a repo
                path_parts = attempted_path.split('/')
                wrong_path = '/'.join(path_parts[2:])  # After owner/repo

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
                # It's a repo not found
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

        # Pattern: Permission denied
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

        # Pattern: Rate limit
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

        # Default generic error
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

        # Common corrections
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

        # If context provided with actual paths, suggest those
        if context and 'available_paths' in context:
            available = context['available_paths']
            # Find similar paths using simple string matching
            for path in available:
                if self._similarity(wrong_path.lower(), path.lower()) > 0.6:
                    suggestions.append(path)

        return suggestions[:5]  # Top 5

    def _similarity(self, a: str, b: str) -> float:
        """Calculate simple string similarity (0.0 - 1.0)"""
        if a == b:
            return 1.0

        # Check if one contains the other
        if a in b or b in a:
            return 0.8

        # Count matching characters
        matches = sum(1 for c1, c2 in zip(a, b) if c1 == c2)
        max_len = max(len(a), len(b))

        return matches / max_len if max_len > 0 else 0.0

    def enhance_jira_error(self, error: Exception, instruction: str) -> EnhancedError:
        """Enhance Jira agent errors"""
        error_str = str(error).lower()

        if 'not found' in error_str or '404' in error_str:
            # Extract issue key if present
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
        """
        Main method to enhance any agent error.

        Routes to agent-specific enhancer.
        """
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
            # Generic enhancer
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
