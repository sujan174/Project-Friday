"""
Comprehensive Error Handling and Fallbacks for Agents

This module provides robust error handling utilities to ensure a smooth
user experience even when things go wrong.
"""

import os
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels"""
    INFO = "info"           # Informational, no action needed
    WARNING = "warning"     # Something's not ideal but system continues
    ERROR = "error"         # Agent can't function, but system continues
    CRITICAL = "critical"   # System-level issue


@dataclass
class CredentialCheck:
    """Result of credential validation"""
    valid: bool
    missing_vars: list
    error_message: str
    setup_instructions: str


class CredentialValidator:
    """Validates agent credentials with helpful error messages"""

    # Service-specific setup instructions
    SETUP_INSTRUCTIONS = {
        'JIRA_URL': """
To set up Jira:
1. Get your Jira instance URL (e.g., https://yourcompany.atlassian.net)
2. Generate an API token:
   - Go to: https://id.atlassian.com/manage-profile/security/api-tokens
   - Click "Create API token"
   - Copy the token
3. Add to your .env file:
   JIRA_URL=https://yourcompany.atlassian.net
   JIRA_USERNAME=your.email@company.com
   JIRA_API_TOKEN=your_token_here
""",
        'GITHUB_TOKEN': """
To set up GitHub:
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Select scopes: repo, read:org, read:user
4. Copy the token
5. Add to your .env file:
   GITHUB_TOKEN=ghp_your_token_here
""",
        'SLACK_TOKEN': """
To set up Slack:
1. Go to: https://api.slack.com/apps
2. Create a new app or select existing
3. Go to "OAuth & Permissions"
4. Add required scopes (channels:read, chat:write, users:read, etc.)
5. Install app to workspace
6. Copy "Bot User OAuth Token"
7. Add to your .env file:
   SLACK_TOKEN=xoxb-your-token-here
""",
        'NOTION_TOKEN': """
To set up Notion:
1. Go to: https://www.notion.so/profile/integrations
2. Click "New Integration"
3. Give it a name and workspace
4. Copy the "Internal Integration Token"
5. Share your pages/databases with this integration
6. Add to your .env file:
   NOTION_TOKEN=secret_your_token_here
""",
        'GEMINI_API_KEY': """
To set up Google Gemini:
1. Go to: https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy the API key
4. Add to your .env file:
   GEMINI_API_KEY=your_api_key_here
"""
    }

    @staticmethod
    def check_credentials(
        agent_name: str,
        required_vars: Dict[str, str]
    ) -> CredentialCheck:
        """
        Check if all required credentials are present

        Args:
            agent_name: Name of the agent (for error messages)
            required_vars: Dict of {env_var_name: friendly_name}

        Returns:
            CredentialCheck with validation results
        """
        missing = []

        for var_name, friendly_name in required_vars.items():
            value = os.getenv(var_name)
            if not value or value.strip() == "":
                missing.append((var_name, friendly_name))

        if not missing:
            return CredentialCheck(
                valid=True,
                missing_vars=[],
                error_message="",
                setup_instructions=""
            )

        # Build friendly error message
        missing_list = ", ".join([f[1] for f in missing])
        error_msg = (
            f"{agent_name} agent cannot start: Missing credentials for {missing_list}\n\n"
            f"To fix this:\n"
        )

        # Add setup instructions for each missing credential
        instructions = []
        for var_name, friendly_name in missing:
            if var_name in CredentialValidator.SETUP_INSTRUCTIONS:
                instructions.append(CredentialValidator.SETUP_INSTRUCTIONS[var_name])
            else:
                instructions.append(f"\nAdd {var_name} to your .env file")

        setup_instructions = "\n".join(instructions)

        # Add disable hint
        error_msg += f"\nOR disable this agent: Add DISABLED_AGENTS={agent_name.lower()} to .env"

        return CredentialCheck(
            valid=False,
            missing_vars=[v[0] for v in missing],
            error_message=error_msg,
            setup_instructions=setup_instructions
        )


class AgentErrorHandler:
    """Handles agent errors gracefully with fallbacks"""

    @staticmethod
    def handle_initialization_error(
        agent_name: str,
        error: Exception,
        verbose: bool = False
    ) -> Tuple[str, str]:
        """
        Convert agent initialization errors to user-friendly messages

        Returns:
            (short_message, detailed_message)
        """
        error_str = str(error).lower()
        error_type = type(error).__name__

        # Connection/Network errors
        if any(word in error_str for word in ['timeout', 'connection', 'network', 'econnrefused', 'unreachable']):
            short = f"Network connection issue"
            detailed = (
                f"{agent_name} couldn't connect to its service.\n"
                f"This usually means:\n"
                f"  • Network connectivity issues\n"
                f"  • Service is temporarily down\n"
                f"  • Firewall/proxy blocking the connection\n\n"
                f"Try:\n"
                f"  • Check your internet connection\n"
                f"  • Verify the service is accessible\n"
                f"  • Try again in a few moments\n"
                f"  • Disable with: DISABLED_AGENTS={agent_name.lower()}"
            )
            return short, detailed

        # Authentication errors
        if any(word in error_str for word in ['authentication', 'unauthorized', '401', '403', 'forbidden', 'token', 'api key']):
            short = f"Invalid or expired credentials"
            detailed = (
                f"{agent_name} credentials are invalid or expired.\n\n"
                f"Try:\n"
                f"  • Check your credentials in .env file\n"
                f"  • Regenerate API token/key\n"
                f"  • Verify token has correct permissions\n"
                f"  • Disable with: DISABLED_AGENTS={agent_name.lower()}"
            )
            return short, detailed

        # Rate limiting
        if any(word in error_str for word in ['rate limit', '429', 'too many requests', 'throttle']):
            short = f"Rate limit exceeded"
            detailed = (
                f"{agent_name} hit API rate limits.\n\n"
                f"Try:\n"
                f"  • Wait a few minutes and try again\n"
                f"  • Check your API usage quota\n"
                f"  • Disable temporarily: DISABLED_AGENTS={agent_name.lower()}"
            )
            return short, detailed

        # MCP/Server errors
        if any(word in error_str for word in ['mcp', 'npx', 'server', 'stdio']):
            short = f"MCP server issue"
            detailed = (
                f"{agent_name} MCP server couldn't start.\n\n"
                f"Try:\n"
                f"  • Ensure Node.js is installed: node --version\n"
                f"  • Check npx is available: npx --version\n"
                f"  • Try: npm cache clean --force\n"
                f"  • Disable with: DISABLED_AGENTS={agent_name.lower()}"
            )
            return short, detailed

        # Permission errors
        if any(word in error_str for word in ['permission', 'access denied', 'not allowed']):
            short = f"Permission denied"
            detailed = (
                f"{agent_name} doesn't have required permissions.\n\n"
                f"Try:\n"
                f"  • Check API token has necessary scopes\n"
                f"  • Verify workspace/organization access\n"
                f"  • Regenerate token with correct permissions\n"
                f"  • Disable with: DISABLED_AGENTS={agent_name.lower()}"
            )
            return short, detailed

        # Generic fallback
        short = f"Initialization failed ({error_type})"
        detailed = (
            f"{agent_name} failed to initialize.\n"
            f"Error: {str(error)[:200]}\n\n"
            f"Try:\n"
            f"  • Check logs for details\n"
            f"  • Verify configuration in .env\n"
            f"  • Disable with: DISABLED_AGENTS={agent_name.lower()}"
        )

        if verbose:
            detailed += f"\n\nFull error: {str(error)}"

        return short, detailed

    @staticmethod
    def handle_execution_error(
        agent_name: str,
        instruction: str,
        error: Exception,
        verbose: bool = False
    ) -> str:
        """
        Convert agent execution errors to user-friendly messages

        Returns:
            User-friendly error message
        """
        error_str = str(error).lower()

        # Network/timeout during execution
        if any(word in error_str for word in ['timeout', 'timed out']):
            return (
                f"⏱️ {agent_name} operation timed out.\n\n"
                f"The operation took too long to complete. This might mean:\n"
                f"  • The service is slow to respond\n"
                f"  • The operation is complex and needs more time\n"
                f"  • Network connectivity issues\n\n"
                f"Try: Simplify the request or try again later"
            )

        # Not found errors
        if any(word in error_str for word in ['not found', '404', 'does not exist', 'cannot find']):
            return (
                f"❌ {agent_name} couldn't find the requested resource.\n\n"
                f"Instruction: {instruction[:100]}...\n\n"
                f"Try:\n"
                f"  • Check the name/ID is correct\n"
                f"  • Verify you have access to the resource\n"
                f"  • List available resources first"
            )

        # Permission during execution
        if any(word in error_str for word in ['permission', 'forbidden', '403']):
            return (
                f"🔒 {agent_name} doesn't have permission for this operation.\n\n"
                f"Instruction: {instruction[:100]}...\n\n"
                f"You may need to:\n"
                f"  • Update API token permissions\n"
                f"  • Request access from admin\n"
                f"  • Check workspace settings"
            )

        # Generic execution error
        message = (
            f"❌ {agent_name} encountered an error:\n"
            f"{str(error)[:300]}\n\n"
            f"Instruction: {instruction[:100]}..."
        )

        if verbose:
            message += f"\n\nFull error: {str(error)}"

        return message


class FallbackBehaviors:
    """Defines fallback behaviors when agents fail"""

    @staticmethod
    def get_graceful_degradation_message(agent_name: str, requested_action: str) -> str:
        """
        Generate a helpful message when an agent is unavailable
        """
        return (
            f"ℹ️ {agent_name} agent is currently unavailable.\n\n"
            f"You requested: {requested_action[:100]}...\n\n"
            f"What you can do:\n"
            f"  • Set up the agent (see error messages above)\n"
            f"  • Use an alternative agent if available\n"
            f"  • Perform this action manually\n"
            f"  • Skip this step for now"
        )

    @staticmethod
    def should_retry(error: Exception, attempt: int, max_attempts: int) -> bool:
        """
        Determine if an operation should be retried
        """
        if attempt >= max_attempts:
            return False

        error_str = str(error).lower()

        # Retry on transient errors
        retryable_keywords = [
            'timeout', 'connection', 'network',
            'rate limit', '429', '503', '502', '504',
            'temporarily', 'unavailable', 'try again'
        ]

        return any(keyword in error_str for keyword in retryable_keywords)

    @staticmethod
    def get_retry_delay(attempt: int, base_delay: float = 1.0) -> float:
        """
        Calculate retry delay with exponential backoff
        """
        return min(base_delay * (2 ** attempt), 30.0)  # Max 30 seconds


def format_user_friendly_error(
    error: Exception,
    context: str = "",
    verbose: bool = False
) -> str:
    """
    Format any exception into a user-friendly message

    Args:
        error: The exception
        context: Additional context about what was being done
        verbose: Include technical details

    Returns:
        Formatted error message
    """
    error_type = type(error).__name__
    error_msg = str(error)

    message = f"❌ Something went wrong: {error_type}\n"

    if context:
        message += f"\nWhile: {context}\n"

    # Extract useful info from error message
    if len(error_msg) > 0:
        # Truncate very long error messages
        if len(error_msg) > 200 and not verbose:
            message += f"\nError: {error_msg[:200]}...\n"
        else:
            message += f"\nError: {error_msg}\n"

    if verbose:
        import traceback
        message += f"\nFull traceback:\n{traceback.format_exc()}"

    return message
