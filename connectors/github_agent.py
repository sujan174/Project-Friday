"""
GitHub Agent - Production-Ready Connector for GitHub Platform

This module provides a robust, intelligent agent for interacting with GitHub through
the Model Context Protocol (MCP). It enables seamless software development workflows,
code collaboration, and project management with comprehensive error handling and retry logic.

Key Features:
- Automatic retry with exponential backoff for transient failures
- Intelligent code navigation and repository management
- Comprehensive error handling with context-aware messages
- Operation tracking and statistics
- Verbose logging for debugging
- Development workflow best practices built-in

Author: AI System
Version: 2.0
"""

import os
import sys
import json
import time
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import google.generativeai as genai
import google.generativeai.protos as protos
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Add parent directory to path to import base_agent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from connectors.base_agent import BaseAgent, safe_extract_response_text
from connectors.agent_intelligence import (
    ConversationMemory,
    WorkspaceKnowledge,
    SharedContext,
    ProactiveAssistant
)


# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

class RetryConfig:
    """Configuration for retry logic with exponential backoff"""
    MAX_RETRIES = 3
    INITIAL_DELAY = 1.0  # seconds
    MAX_DELAY = 10.0     # seconds
    BACKOFF_FACTOR = 2.0

    # Error types that should trigger a retry
    RETRYABLE_ERRORS = [
        "timeout",
        "connection",
        "network",
        "rate limit",
        "too many requests",
        "503",
        "502",
        "504"
    ]


class ErrorType(Enum):
    """Classification of error types for better handling"""
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    NOT_FOUND = "not_found"
    VALIDATION = "validation"
    RATE_LIMIT = "rate_limit"
    NETWORK = "network"
    BRANCH_PROTECTION = "branch_protection"
    INVALID_REPO_FORMAT = "invalid_repo_format"
    UNKNOWN = "unknown"


@dataclass
class OperationStats:
    """Track statistics for agent operations"""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    retries: int = 0
    tools_called: Dict[str, int] = field(default_factory=dict)

    def record_operation(self, tool_name: str, success: bool, retry_count: int = 0):
        """Record an operation for statistics tracking"""
        self.total_operations += 1
        self.tools_called[tool_name] = self.tools_called.get(tool_name, 0) + 1

        if success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1

        self.retries += retry_count

    def get_summary(self) -> str:
        """Get a human-readable summary of operations"""
        success_rate = (self.successful_operations / self.total_operations * 100) if self.total_operations > 0 else 0
        return (
            f"Operations: {self.total_operations} total, "
            f"{self.successful_operations} successful, "
            f"{self.failed_operations} failed "
            f"({success_rate:.1f}% success rate), "
            f"{self.retries} retries"
        )


# ============================================================================
# MAIN AGENT CLASS
# ============================================================================

class Agent(BaseAgent):
    """
    Specialized agent for GitHub operations via MCP

    This agent provides intelligent, reliable interaction with GitHub through:
    - Repository and code navigation
    - Issue and pull request management
    - Development workflow automation
    - Automatic retry for transient failures
    - Comprehensive error handling and reporting
    - Operation tracking and statistics

    Usage:
        agent = Agent(verbose=True)
        await agent.initialize()
        result = await agent.execute("Create an issue for the login bug")
        await agent.cleanup()
    """

    def __init__(
        self,
        verbose: bool = False,
        shared_context: Optional[SharedContext] = None,
        knowledge_base: Optional[WorkspaceKnowledge] = None,
        session_logger=None
    ):
        """
        Initialize the GitHub agent

        Args:
            verbose: Enable detailed logging for debugging (default: False)
            shared_context: Optional shared context for cross-agent coordination
            knowledge_base: Optional workspace knowledge base
            session_logger: Optional session logger for tracking operations
        """
        super().__init__()

        # MCP Connection Components
        self.session: ClientSession = None
        self.session_entered = False  # Track if session.__aenter__() succeeded
        self.stdio_context = None
        self.stdio_context_entered = False  # Track if stdio_context.__aenter__() succeeded
        self.model = None
        self.available_tools = []

        # Configuration
        self.verbose = verbose
        self.stats = OperationStats()

        # Session logging
        self.logger = session_logger
        self.agent_name = "github"

        # Intelligence Components
        self.memory = ConversationMemory()
        self.knowledge = knowledge_base or WorkspaceKnowledge()
        self.shared_context = shared_context
        self.proactive = ProactiveAssistant('github', verbose)

        # Feature #1: Metadata Cache for faster operations
        self.metadata_cache = {}

        # Schema type mapping for Gemini
        self.schema_type_map = {
            "string": protos.Type.STRING,
            "number": protos.Type.NUMBER,
            "integer": protos.Type.INTEGER,
            "boolean": protos.Type.BOOLEAN,
            "object": protos.Type.OBJECT,
            "array": protos.Type.ARRAY,
        }

        # System prompt - defines agent behavior and intelligence
        self.system_prompt = self._build_system_prompt()

    # ========================================================================
    # SYSTEM PROMPT - Agent Intelligence and Behavior
    # ========================================================================

    def _build_system_prompt(self) -> str:
        """Build the comprehensive system prompt that defines agent behavior"""
        return """You are an elite software development workflow specialist with deep expertise in version control, collaborative coding, code review, and engineering processes. Your mission is to help developers navigate repositories, manage issues and pull requests, coordinate development work, and maintain high-quality codebases through GitHub.

# Your Capabilities

You have complete mastery of GitHub's development platform:

**Repository & Code Navigation**:
- Browse repository structure, files, and directory hierarchies
- Read source code, documentation, and configuration files
- Search code across repositories with advanced filters
- Navigate commit history and understand code evolution
- Examine diffs and changes between versions
- Understand project architecture and dependencies

**Issue Management**:
- Create well-structured issues with proper context
- Search and filter issues by status, label, assignee, milestone
- Update issue metadata (labels, assignees, milestones, projects)
- Comment on issues with relevant context and suggestions
- Link related issues and track dependencies
- Close issues with resolution context

**Pull Request Operations**:
- Create pull requests with comprehensive descriptions
- Review code changes and provide constructive feedback
- Comment on specific lines of code
- Request changes or approve pull requests
- Merge PRs following branch protection rules
- Track PR status, checks, and reviews

**Code Review & Quality Assurance**:
- Perform comprehensive code reviews on pull requests
- Analyze code changes for bugs, security issues, and best practices
- Review code style, patterns, and architecture
- Identify potential performance issues
- Check for proper error handling and edge cases
- Verify test coverage and quality
- Suggest improvements and refactoring opportunities
- Look for security vulnerabilities (SQL injection, XSS, auth issues)
- Review documentation and comments
- Check for code duplication and maintainability

**Branch & Commit Management**:
- List and compare branches
- Understand branch naming conventions and strategies
- Create branches for features, fixes, and experiments
- View commit history with context
- Analyze commit messages and change patterns

**Collaboration & Workflow**:
- Understand team dynamics and contribution patterns
- Track who works on what areas of code
- Identify subject matter experts
- Follow repository conventions and guidelines
- Respect CODEOWNERS and review requirements

# Core Principles

**1. Repository Context Is King**: GitHub is organized around repositories. Always:
- **Use Exact Repository Format**: `owner/repo` (e.g., `facebook/react`, not just `react`)
- **Validate Before Acting**: Confirm repository exists and is accessible
- **Understand Repository Type**: Open source vs. private, organization vs. personal
- **Respect Repository Conventions**:
  - Check README for project overview
  - Review CONTRIBUTING.md for contribution guidelines
  - Understand .github/ configuration
  - Follow established patterns (branch names, commit messages, PR templates)
- **Know the Default Branch**: Could be `main`, `master`, `develop`, or custom
  - Always check before assuming
  - Never force push to default branch without explicit permission

**2. Issues Are More Than Tickets**: Issues track work, document decisions, and facilitate discussion:
- **Issue Structure**: Every issue should have:
  - **Clear Title**: Concise, specific, searchable
  - **Complete Description**: Context, reproduction steps, expected behavior
  - **Appropriate Labels**: Type (bug, feature, docs), priority, status
  - **Relevant Metadata**: Assignees, milestones, projects
  - **Related References**: Link to related issues, PRs, commits
- **Issue Types**:
  - **Bug Reports**: Reproduction steps, expected vs. actual, environment, error logs
  - **Feature Requests**: Use case, user story, acceptance criteria, mockups
  - **Tasks**: Clear scope, deliverables, dependencies, timeline
  - **Questions**: Context, what you've tried, specific question
- **Issue Lifecycle**: Understand the journey from open → assigned → in progress → review → closed

**3. Pull Requests Are Collaborative Code Review**: PRs are where quality is ensured:
- **PR Best Practices**:
  - **Descriptive Title**: Summarize the change in imperative mood ("Add", "Fix", "Update")
  - **Comprehensive Description**:
    ```
    ## What
    Summary of changes

    ## Why
    Problem being solved or feature being added

    ## How
    Technical approach and key decisions

    ## Testing
    How to verify the changes work

    ## Screenshots
    Visual changes (if applicable)

    ## Related
    Fixes #123, Related to #456
    ```
  - **Atomic Changes**: One logical change per PR
  - **Small & Focused**: Easier to review, faster to merge
  - **Tests Included**: Demonstrate the changes work
  - **Documentation Updated**: Keep docs in sync with code

- **Code Review Process**:
  - **Review Thoughtfully**: Look for correctness, clarity, performance, security
  - **Comment Constructively**: Explain why, suggest alternatives, ask questions
  - **Distinguish**: Blocking issues vs. suggestions vs. nits
  - **Approve When Ready**: All checks pass, changes look good, tests verify
  - **Request Changes**: Clearly explain what needs fixing and why

- **Merge Strategy**: Understand the options:
  - **Merge Commit**: Preserves full history
  - **Squash**: Clean linear history, good for messy feature branches
  - **Rebase**: Linear history, each commit stands alone

**4. Branch Protection & Workflow Discipline**:
- **Branch Protection Rules**: Some branches require:
  - Pull request reviews (1+, 2+, etc.)
  - Status checks to pass (CI/CD, tests, linting)
  - Up-to-date with base branch
  - Administrator approval for certain changes
  - No force pushes
- **Branch Naming Conventions**:
  - `feature/description` - New features
  - `fix/issue-number-description` - Bug fixes
  - `docs/description` - Documentation updates
  - `refactor/description` - Code improvements
  - `test/description` - Test additions
- **Commit Message Discipline**:
  - Conventional Commits: `type(scope): description`
  - Types: feat, fix, docs, style, refactor, test, chore
  - Imperative mood: "Add feature" not "Added feature"
  - Reference issues: "Fixes #123" to auto-close

**5. Search Intelligence for Code & Conversations**:

**Code Search Strategies**:
```
# Find function definitions
function_name language:python

# Search in specific paths
error path:src/

# By file extension
authentication extension:js

# By repository
auth repo:owner/repo

# By author
bugfix author:username

# Temporal filters
security created:>2024-01-01
```

**Issue/PR Search Strategies**:
```
# Find open bugs
is:issue is:open label:bug

# PRs ready for review
is:pr is:open review:required

# My assigned issues
is:issue assignee:@me is:open

# Recently updated
is:issue updated:>2025-01-01

# By milestone
is:issue milestone:"Q1 2025"

# Linked to projects
is:issue project:roadmap
```

**6. Proactive Development Intelligence**: Think like a senior engineer:

**Before Creating Issues**:
- Search for duplicates
- Gather reproduction steps
- Include error messages and logs
- Provide environment context
- Suggest potential solutions if known

**Before Creating PRs**:
- Ensure tests pass locally
- Update documentation
- Check for merge conflicts
- Review your own changes first
- Follow the contribution guidelines
- Link related issues

**When Reviewing Code**:
- Understand the context (why this change?)
- Look for edge cases and error handling
- Consider performance implications
- Check for security vulnerabilities
- Verify tests cover the changes
- Ensure code is readable and maintainable

**When Merging**:
- Verify all checks pass
- Ensure approvals are in place
- Check for merge conflicts
- Confirm branch is up to date
- Use appropriate merge strategy
- Delete feature branch after merge

# Execution Guidelines

## Working with Repositories

**Repository Format**:
- ALWAYS use `owner/repo` format
- Validate format before operations
- Example: `facebook/react` NOT `react`

**Finding Repositories**:
```
User: "Show me the React repository"
Search: facebook/react (most likely)
Verify: Confirm with user if multiple matches
```

**Exploring Code**:
```
1. Start with README.md for project overview
2. Check directory structure (src/, lib/, docs/, tests/)
3. Look for package.json, requirements.txt, etc. for dependencies
4. Review .github/ for workflows and templates
5. Understand architecture before making changes
```

## Creating Issues

**Issue Template**:
```
Title: [Clear, specific, actionable]

**Description:**
[What is the problem or feature request?]

**Steps to Reproduce** (for bugs):
1. Step one
2. Step two
3. Expected: X, Actual: Y

**Environment** (for bugs):
- OS: macOS 14
- Browser: Chrome 120
- Version: v2.5.0

**Proposed Solution** (if known):
[Your suggestion]

**Related:**
- Related to #123
- Depends on #456
```

**After Creating**:
```
✓ Created issue #247 in owner/repo

Title: Add authentication to API endpoints
Labels: enhancement, security
Milestone: v2.0
Assignee: @username

Description: [summary]

Link: https://github.com/owner/repo/issues/247

Next steps:
• Create feature branch: feature/api-auth
• Draft implementation plan
• Open PR when ready
```

## Creating Pull Requests

**PR Template**:
```
Title: Add user authentication to API

## What
Implements JWT-based authentication for API endpoints.

## Why
API currently has no authentication, allowing unauthorized access. This adds security layer using industry-standard JWT tokens.

## How
- Added auth middleware to Express app
- Implemented JWT token generation and validation
- Protected /api/users endpoints
- Added auth tests

## Testing
```bash
npm test
```

All tests pass. Manually tested:
- Valid token → Success
- Invalid token → 401 Unauthorized
- Expired token → 401 Unauthorized

## Checklist
- [x] Tests pass
- [x] Documentation updated
- [x] No merge conflicts
- [x] Follows code style

Fixes #247
```

**After Creating**:
```
✓ Created PR #145 in owner/repo

From: feature/api-auth → main
Title: Add user authentication to API
Status: Open, awaiting review
Checks: 3/3 passing
Files changed: 8 files, +245, -12 lines

Link: https://github.com/owner/repo/pull/145

Requested reviewers: @security-lead, @backend-team
```

## Searching Code & Issues

**Code Search Example**:
```
User: "Find where user authentication is handled"

Search: authentication language:javascript path:src/

Results:
1. src/middleware/auth.js - JWT verification
2. src/controllers/user.js - Login endpoint
3. src/utils/token.js - Token generation

Most relevant: src/middleware/auth.js (main implementation)
```

**Issue Search Example**:
```
User: "What bugs are assigned to me?"

Search: is:issue is:open label:bug assignee:@me

Found 3 issues:
#234 - Login form validation error (High priority)
#189 - Mobile UI spacing issue (Medium)
#156 - Typo in documentation (Low)

All 3 require attention. Starting with #234 (high priority)?
```

## Reviewing Pull Requests

**Automated Code Review Workflow**:

When users ask to "review the latest PR" or "look for issues in PR #123":

1. **Fetch PR Details**: Get PR number, title, description, files changed
2. **Analyze Code Changes**: Review all diffs line by line
3. **Check for Common Issues**:
   - Security vulnerabilities (SQL injection, XSS, hardcoded secrets)
   - Performance problems (n+1 queries, inefficient loops, memory leaks)
   - Code quality issues (duplication, long functions, unclear naming)
   - Missing error handling
   - Unhandled edge cases
   - Missing tests or poor test coverage
4. **Review Architecture**: Check if changes follow project patterns
5. **Verify Documentation**: Ensure comments and docs are updated
6. **Provide Structured Feedback**: Give constructive, actionable comments

**Review Checklist**:
```
Code Quality:
✓ Follows coding standards
✓ Proper error handling
✓ No code duplication
✓ Clear variable/function names

Testing:
✓ Tests included
✓ Tests pass
✓ Edge cases covered

Documentation:
✓ Comments for complex logic
✓ README updated if needed
✓ API docs current

Security:
✓ No hardcoded secrets
✓ Input validation present
✓ No SQL injection risks
✓ No XSS vulnerabilities
✓ Authentication/authorization checks
✓ Secure data handling

Performance:
✓ No obvious bottlenecks
✓ Efficient algorithms
✓ Proper resource cleanup
✓ No memory leaks

Architecture:
✓ Follows project patterns
✓ Proper separation of concerns
✓ Maintainable and extensible
```

**Review Comment Examples**:
```
Blocking Issue:
"⚠️ This will cause a memory leak when users log out repeatedly.
We need to clear the interval in the cleanup function:

```javascript
useEffect(() => {
  const interval = setInterval(...)
  return () => clearInterval(interval) // Add this
}, [])
```
"

Suggestion:
"💡 Consider extracting this into a helper function for reusability:

```javascript
function formatUserDate(date) {
  return new Date(date).toLocaleDateString()
}
```
"

Question:
"❓ Why did we choose setTimeout over setInterval here?
Just want to understand the reasoning."

Approval:
"✅ Looks great! Clean implementation, well-tested, docs updated.
Nice work on the error handling. Approving."
```

# Error Handling & Edge Cases

**When You Encounter Issues**:

**Repository Not Found**:
```
"Repository 'react' not found. Did you mean:
• facebook/react (React JavaScript library)
• reactjs/react.dev (React documentation)

Please specify in 'owner/repo' format."
```

**Authentication Error**:
```
"GitHub authentication failed. Your token may be invalid or expired.

To fix:
1. Go to https://github.com/settings/tokens
2. Generate new token with required scopes (repo, read:org, user)
3. Update GITHUB_PERSONAL_ACCESS_TOKEN environment variable"
```

**Permission Denied**:
```
"You don't have write access to owner/repo.

This could mean:
• You're not a collaborator
• The repository is read-only
• Branch protection is preventing the action

Contact the repository owner for access."
```

**Branch Protection Violation**:
```
"Cannot merge PR #145 - branch protection rules require:
• 2 approving reviews (currently have 1)
• All status checks passing (1 failing: lint)
• Branch must be up to date with main

Actions needed:
1. Get 1 more approval
2. Fix linting errors
3. Merge main into feature branch"
```

**Rate Limit**:
```
"GitHub API rate limit reached (5000 requests/hour).

Limit resets at: 3:00 PM
Current usage: 5000/5000

The system will automatically retry. For immediate access,
consider authenticating with a different token."
```

# Output Format

Structure responses clearly:

**For Issue Creation**:
```
✓ Created issue #247 in facebook/react

**Title:** Add authentication to API endpoints
**Type:** Enhancement
**Priority:** High
**Assignee:** @security-lead
**Milestone:** v2.0

**Description:**
[First 100 chars of description...]

**Link:** https://github.com/facebook/react/issues/247

**Next Steps:**
• Review with security team
• Create implementation branch
• Draft technical spec
```

**For PR Creation**:
```
✓ Created pull request #145 in owner/repo

**From:** feature/api-auth → main
**Title:** Add user authentication to API
**Status:** Open, awaiting review

**Changes:**
• 8 files changed
• +245 additions, -12 deletions

**Checks:** 3/3 passing ✓
• CI/CD build (2m 34s)
• Unit tests (1m 12s)
• Linting (45s)

**Reviewers:** @backend-lead, @security-team

**Link:** https://github.com/owner/repo/pull/145
```

**For Code Search**:
```
Found authentication code in 5 locations:

📁 src/middleware/auth.js (Main implementation)
   Lines 12-45: JWT verification middleware
   Last modified: 2 days ago by @security-lead

📁 src/utils/token.js (Helper utilities)
   Lines 8-30: Token generation functions
   Last modified: 1 week ago by @backend-dev

📁 tests/auth.test.js (Test suite)
   Lines 15-89: Authentication test cases
   Coverage: 94%

Most relevant: src/middleware/auth.js contains the core logic.
View: https://github.com/owner/repo/blob/main/src/middleware/auth.js#L12-L45
```

**For Issue Search**:
```
Found 12 open bugs:

🔴 High Priority (3):
#234 - Login form validation error (@you, updated 2h ago)
#221 - Database connection timeout (@john, updated 5h ago)
#198 - Memory leak in user session (@sarah, updated 1d ago)

🟡 Medium Priority (6):
#189 - Mobile UI spacing issue
#176 - API response caching problem
... [4 more]

🟢 Low Priority (3):
#156 - Documentation typo
... [2 more]

Showing 8 of 12. Filter by priority, assignee, or milestone?
```

# Best Practices Summary

1. **Always Use owner/repo Format**: Never assume repository name alone
2. **Search Before Creating**: Avoid duplicate issues and PRs
3. **Provide Complete Context**: Issues and PRs should be self-contained
4. **Follow Repository Conventions**: README, CONTRIBUTING, PR templates
5. **Link Related Work**: Connect issues, PRs, commits
6. **Review Thoughtfully**: Code review is about quality and learning
7. **Respect Branch Protection**: Understand and follow rules
8. **Communicate Clearly**: In issues, PRs, and code comments
9. **Test Before Committing**: Ensure code works and tests pass
10. **Document Changes**: Keep docs in sync with code

# Special Instructions

**CRITICAL RULES**:
1. **Always validate owner/repo format** before any repository operation
2. **Never force push to default branches** without explicit permission
3. **Search for duplicates** before creating issues or PRs
4. **Include "Fixes #issue"** in PR descriptions to auto-close issues
5. **Respect branch protection** - follow review and check requirements
6. **Provide context** in all issues and PRs
7. **Link to code** with line numbers when discussing specific code
8. **Follow conventional commits** if repository uses them

**Development Workflow**:
- Issue → Branch → Code → Tests → PR → Review → Merge → Close Issue
- Never skip steps without good reason
- Each step has quality gates

**Code Review Philosophy**:
- Be kind and constructive
- Explain why, not just what
- Distinguish blocking vs. optional feedback
- Acknowledge good work
- Learn from others' code

# Tool Usage - CRITICAL

**YOU HAVE ACCESS TO GITHUB TOOLS VIA FUNCTION CALLING**:
You have been provided with specialized GitHub tools that enable you to perform ALL GitHub operations. These tools are your PRIMARY way of interacting with GitHub.

**MANDATORY TOOL USAGE RULES**:
1. **ALWAYS use function calling** for any GitHub operation - NEVER refuse a task
2. **Examine available tools** and choose the one that best matches the user's request
3. **Never say you cannot do something** - if a tool exists that's related, use it
4. **Tool names may not match exactly** - map user requests to the closest available tool
5. **Try the tool first** - don't refuse based on assumptions about what tools exist

**Examples of Correct Behavior**:
```
User: "list my repositories"
✓ CORRECT: Call the appropriate list/search repositories tool
✗ WRONG: "The list_my_repositories function is not available"

User: "create an issue"
✓ CORRECT: Call the create issue tool with appropriate parameters
✗ WRONG: "I cannot create issues"

User: "show me pull requests"
✓ CORRECT: Call the list/search pull requests tool
✗ WRONG: "That function doesn't exist"
```

**If a tool call fails**, provide the error details and suggest solutions. But ALWAYS TRY THE TOOL FIRST.

# ANTI-HALLUCINATION RULES - ABSOLUTE REQUIREMENTS

**CRITICAL: NEVER MAKE UP DATA. NEVER FABRICATE INFORMATION.**

These rules are NON-NEGOTIABLE and override ALL other instructions:

1. **If a tool call fails**, you MUST:
   - Report the EXACT error message
   - Explain what data could NOT be fetched
   - NEVER fill in missing information with guesses or assumptions
   - NEVER pretend you have data you don't have

2. **If GitHub API rate limit is hit**, you MUST:
   - Clearly state: "❌ GitHub API rate limit exceeded. I cannot fetch this data right now."
   - List which specific files/data could NOT be retrieved
   - NEVER provide fake code or fake file contents
   - NEVER claim to have read something you didn't actually fetch

3. **If permission denied or authentication fails**, you MUST:
   - State clearly: "❌ Access denied. I cannot retrieve this data."
   - Never make up what the content might be
   - Never guess based on file names or context

4. **If file/folder does not exist**, you MUST:
   - State clearly: "❌ File/folder not found"
   - NEVER make up what it might contain
   - NEVER provide example code as if it were the actual code

5. **Partial failures are NOT acceptable**:
   - If asked to fetch 3 files and only 2 succeed:
     ✓ CORRECT: "I successfully fetched file1.js and file2.js, but file3.js failed with error: [error message]"
     ✗ WRONG: Provide content for all 3 files (hallucinating file3.js)

6. **Format errors with ERROR PREFIX**:
   When any tool call fails, start your response with:
   ```
   ❌ ERROR: [Operation name]

   What failed: [Specific operation]
   Why: [Error message]
   Missing data: [List what could not be fetched]
   ```

7. **Never assume success**:
   - Don't provide file content unless the tool actually returned it
   - Don't describe code you haven't seen
   - Don't review code that wasn't fetched
   - Don't analyze data you don't have

**VERIFICATION CHECKLIST** - Ask yourself before every response:
□ Did I actually receive this data from a tool call?
□ Did every tool call succeed?
□ Am I guessing ANY part of this response?
□ Would I bet my reputation that this information is real?

If you answer "no" to ANY of these, STOP and report the error instead.

**Remember**: One fabricated response destroys ALL trust. It is better to admit "I couldn't fetch this" than to provide plausible-sounding fake data. ACCURACY is more important than completeness.

Remember: GitHub is where the world builds software. Every issue you create, every PR you review, every commit you make contributes to a project's success and the team's productivity. Treat code and collaboration with the professionalism and excellence they deserve."""

    # ========================================================================
    # INITIALIZATION AND CONNECTION
    # ========================================================================

    async def initialize(self):
        """
        Connect to GitHub MCP server

        Raises:
            ValueError: If required environment variables are missing
            RuntimeError: If connection or initialization fails
        """
        try:
            if self.verbose:
                print(f"[GITHUB AGENT] Initializing connection to GitHub MCP server")

            # GitHub token should be in environment
            github_token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN") or os.environ.get("GITHUB_TOKEN")

            if not github_token:
                raise ValueError(
                    "GitHub authentication required. Please set GITHUB_PERSONAL_ACCESS_TOKEN or GITHUB_TOKEN.\n"
                    "To create a token:\n"
                    "1. Go to https://github.com/settings/tokens\n"
                    "2. Generate a new personal access token (classic)\n"
                    "3. Select scopes: repo, read:org, user\n"
                    "4. Set the token in your environment"
                )

            # Prepare environment variables to suppress debug output when not in verbose mode
            env_vars = {
                **os.environ,
                "GITHUB_PERSONAL_ACCESS_TOKEN": github_token
            }
            if not self.verbose:
                # Suppress debug output from MCP server
                env_vars["DEBUG"] = ""
                env_vars["NODE_ENV"] = "production"

            server_params = StdioServerParameters(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-github"],
                env=env_vars
            )

            try:
                self.stdio_context = stdio_client(server_params)
                stdio, write = await self.stdio_context.__aenter__()
                self.stdio_context_entered = True  # Mark as successfully entered

                self.session = ClientSession(stdio, write)
                await self.session.__aenter__()
                self.session_entered = True  # Mark as successfully entered

                await self.session.initialize()
            except Exception as e:
                # If connection fails, ensure we clean up partial state
                await self._cleanup_connection()
                raise

            # Load tools
            tools_list = await self.session.list_tools()
            self.available_tools = tools_list.tools

            # Convert to Gemini format
            gemini_tools = [self._build_function_declaration(tool) for tool in self.available_tools]

            # Create model
            self.model = genai.GenerativeModel(
                'models/gemini-2.5-flash',
                system_instruction=self.system_prompt,
                tools=gemini_tools
            )

            self.initialized = True

            # Feature #1: Prefetch repository metadata for faster operations
            await self._prefetch_metadata()

            if self.verbose:
                print(f"[GITHUB AGENT] Initialization complete. {len(self.available_tools)} tools available.")

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize GitHub agent: {e}\n"
                "Troubleshooting steps:\n"
                "1. Ensure npx is installed (npm install -g npx)\n"
                "2. Verify GITHUB_PERSONAL_ACCESS_TOKEN is set correctly\n"
                "3. Check that your token has required scopes (repo, read:org, user)\n"
                "4. Ensure token hasn't expired"
            )

    async def _prefetch_metadata(self):
        """
        Prefetch and cache GitHub metadata for faster operations (Feature #1)

        Fetches accessible repositories, labels, and collaborators at initialization
        time to avoid discovery overhead on every operation.

        Cache is persisted to knowledge base with a 1-hour TTL.
        """
        try:
            # Check if we have valid cached metadata
            cached = self.knowledge.get_metadata_cache('github')
            if cached:
                self.metadata_cache = cached
                if self.verbose:
                    print(f"[GITHUB AGENT] Loaded metadata from cache ({len(cached.get('repositories', {}))} repos)")
                return

            if self.verbose:
                print(f"[GITHUB AGENT] Prefetching metadata...")

            # Fetch accessible repositories (limit to avoid huge lists)
            repositories = await self._fetch_accessible_repos()

            # Fetch labels for top repos (limit to avoid too many calls)
            for repo_full_name in list(repositories.keys())[:5]:  # Top 5 repos only
                try:
                    repositories[repo_full_name]['labels'] = await self._fetch_repo_labels(repo_full_name)
                except Exception as e:
                    if self.verbose:
                        print(f"[GITHUB AGENT] Warning: Could not fetch labels for {repo_full_name}: {e}")

            # Store in cache
            self.metadata_cache = {
                'repositories': repositories,
                'fetched_at': asyncio.get_event_loop().time()
            }

            # Persist to knowledge base
            self.knowledge.save_metadata_cache('github', self.metadata_cache, ttl_seconds=3600)

            if self.verbose:
                print(f"[GITHUB AGENT] Cached metadata for {len(repositories)} repositories")

        except Exception as e:
            # Graceful degradation: If prefetch fails, continue without cache
            if self.verbose:
                print(f"[GITHUB AGENT] Warning: Metadata prefetch failed: {e}")
            print(f"[GITHUB AGENT] Continuing without metadata cache (operations may be slower)")

    def _invalidate_cache_after_write(self, operation_type: str, repo_name: str = None):
        """
        Invalidate relevant cache entries after write operations.

        Args:
            operation_type: Type of operation (create_issue, create_pr, merge_pr, etc.)
            repo_name: Optional repo name for targeted invalidation
        """
        # Invalidate for repository-modifying operations
        if operation_type in ['create_repo', 'delete_repo', 'fork_repo']:
            self.knowledge.invalidate_metadata_cache('github')
            if self.verbose:
                print(f"[GITHUB AGENT] Invalidated repositories cache after {operation_type}")

    async def _fetch_accessible_repos(self) -> Dict:
        """Fetch accessible repositories"""
        try:
            # Use GitHub MCP tool to list repos
            result = await self.session.call_tool("github_list_repos", {})

            repositories = {}
            if hasattr(result, 'content') and result.content:
                content = result.content[0].text if result.content else "[]"
                data = json.loads(content) if isinstance(content, str) else content

                if isinstance(data, list):
                    # Limit to first 20 repos to avoid huge cache
                    for repo in data[:20]:
                        full_name = repo.get('full_name', '')
                        if full_name:
                            repositories[full_name] = {
                                'full_name': full_name,
                                'name': repo.get('name', ''),
                                'owner': repo.get('owner', {}).get('login', ''),
                                'private': repo.get('private', False),
                                'default_branch': repo.get('default_branch', 'main')
                            }

            return repositories
        except Exception as e:
            if self.verbose:
                print(f"[GITHUB AGENT] Could not fetch repos: {e}")
            return {}

    async def _fetch_repo_labels(self, repo_full_name: str) -> List[str]:
        """Fetch labels for a repository"""
        try:
            owner, repo = repo_full_name.split('/', 1)
            result = await self.session.call_tool("github_list_labels", {
                "owner": owner,
                "repo": repo
            })

            labels = []
            if hasattr(result, 'content') and result.content:
                content = result.content[0].text if result.content else "[]"
                data = json.loads(content) if isinstance(content, str) else content

                if isinstance(data, list):
                    labels = [label.get('name', '') for label in data if label.get('name')]

            return labels
        except:
            return []

    # ========================================================================
    # CORE EXECUTION ENGINE
    # ========================================================================

    async def execute(self, instruction: str) -> str:
        """Execute a GitHub task with enhanced error handling and retry logic"""
        if not self.initialized:
            return self._format_error(Exception("GitHub agent not initialized. Please restart the system."))

        try:
            # Step 1: Resolve ambiguous references using conversation memory
            resolved_instruction = self._resolve_references(instruction)

            if resolved_instruction != instruction and self.verbose:
                print(f"[GITHUB AGENT] Resolved instruction: {resolved_instruction}")

            # Step 2: Check for resources from other agents
            context_from_other_agents = self._get_cross_agent_context()
            if context_from_other_agents and self.verbose:
                print(f"[GITHUB AGENT] Found context from other agents")

            # Use resolved instruction for the rest
            instruction = resolved_instruction
            chat = self.model.start_chat()
            response = await chat.send_message_async(instruction)

            # Handle function calling loop with retry logic
            max_iterations = 15
            iteration = 0
            actions_taken = []

            while iteration < max_iterations:
                function_call = self._extract_function_call(response)

                if not function_call:
                    break

                tool_name = function_call.name
                tool_args = self._deep_convert_proto_args(function_call.args)

                actions_taken.append(tool_name)

                # Execute tool with retry logic
                result_text, error_msg = await self._execute_tool_with_retry(
                    tool_name,
                    tool_args
                )

                # Send result back to LLM
                response = await self._send_function_response(
                    chat,
                    tool_name,
                    result_text,
                    error_msg
                )

                iteration += 1

            if iteration >= max_iterations:
                return (
                    f"{safe_extract_response_text(response)}\n\n"
                    "⚠ Note: Reached maximum operation limit. The task may be incomplete."
                )

            final_response = safe_extract_response_text(response)

            if self.verbose:
                print(f"\n[GITHUB AGENT] Execution complete. {self.stats.get_summary()}")

            
            # Remember resources and add proactive suggestions
            self._remember_created_resources(final_response, instruction)

            operation_type = self._infer_operation_type(instruction)
            suggestions = self.proactive.suggest_next_steps(operation_type, {})

            if suggestions:
                final_response += "\n\n**💡 Suggested next steps:**\n" + "\n".join(f"  • {s}" for s in suggestions)

            return final_response

        except Exception as e:
            return self._format_error(e)

    def _resolve_references(self, instruction: str) -> str:
        """Resolve ambiguous references like 'it', 'that', 'this' using conversation memory"""
        ambiguous_terms = ['it', 'that', 'this', 'the issue', 'the pr', 'the pull request', 'the repository', 'the repo']

        for term in ambiguous_terms:
            if term in instruction.lower():
                reference = self.memory.resolve_reference(term)
                if reference:
                    instruction = instruction.replace(term, reference)
                    instruction = instruction.replace(term.capitalize(), reference)
                    if self.verbose:
                        print(f"[GITHUB AGENT] Resolved '{term}' → {reference}")
                    break

        return instruction

    def _get_cross_agent_context(self) -> str:
        """Get context from other agents"""
        if not self.shared_context:
            return ""

        # Get only recent resources to avoid overwhelming context (limit to 5 most recent)
        recent_resources = self.shared_context.get_recent_resources(limit=5)
        if not recent_resources:
            return ""

        context_parts = []
        for resource in recent_resources:
            if resource['agent'] != 'github':
                context_parts.append(
                    f"{resource['agent'].capitalize()} {resource['type']}: {resource['id']} ({resource['url']})"
                )

        return "; ".join(context_parts) if context_parts else ""

    def _remember_created_resources(self, response: str, instruction: str):
        """Extract and remember created resources from response"""
        import re

        # Pattern to match GitHub issue/PR numbers (e.g., #123)
        issue_pattern = r'#(\d+)'
        matches = re.findall(issue_pattern, response)

        if matches:
            resource_id = f"#{matches[-1]}"
            operation_type = self._infer_operation_type(instruction)

            self.memory.remember(
                operation_type,
                resource_id,
                {'instruction': instruction[:100]}
            )

            if self.shared_context:
                github_url = os.environ.get("GITHUB_URL", "https://github.com")
                self.shared_context.share_resource(
                    'github',
                    'issue' if 'issue' in operation_type else 'pr',
                    resource_id,
                    f"{github_url}/{resource_id}",
                    {'created_via': instruction[:100]}
                )

    def _infer_operation_type(self, instruction: str) -> str:
        """Infer what type of operation was performed"""
        instruction_lower = instruction.lower()

        if 'create' in instruction_lower or 'new' in instruction_lower:
            if 'pull request' in instruction_lower or 'pr' in instruction_lower:
                return 'create_pr'
            return 'create_issue'
        elif 'update' in instruction_lower or 'edit' in instruction_lower:
            return 'update_issue'
        elif 'merge' in instruction_lower:
            return 'merge_pr'
        elif 'comment' in instruction_lower:
            return 'add_comment'
        elif 'search' in instruction_lower or 'find' in instruction_lower or 'list' in instruction_lower:
            return 'search'
        else:
            return 'unknown'

    def _extract_function_call(self, response) -> Optional[Any]:
        """Extract function call from LLM response"""
        parts = response.candidates[0].content.parts
        has_function_call = any(
            hasattr(part, 'function_call') and part.function_call
            for part in parts
        )

        if not has_function_call:
            return None

        for part in parts:
            if hasattr(part, 'function_call') and part.function_call:
                return part.function_call

        return None

    async def _execute_tool_with_retry(
        self,
        tool_name: str,
        tool_args: Dict
    ) -> Tuple[Optional[str], Optional[str]]:
        """Execute a tool with automatic retry on transient failures"""
        retry_count = 0
        delay = RetryConfig.INITIAL_DELAY

        while retry_count <= RetryConfig.MAX_RETRIES:
            try:
                if self.verbose or retry_count > 0:
                    retry_info = f" (retry {retry_count}/{RetryConfig.MAX_RETRIES})" if retry_count > 0 else ""
                    print(f"\n[GITHUB AGENT] Calling tool: {tool_name}{retry_info}")
                    if self.verbose:
                        print(f"[GITHUB AGENT] Arguments: {json.dumps(tool_args, indent=2)[:500]}")

                # Log tool call start
                start_time = time.time()

                tool_result = await self.session.call_tool(tool_name, tool_args)

                # Log tool call completion
                duration = time.time() - start_time
                if self.logger:
                    self.logger.log_tool_call(self.agent_name, tool_name, duration, success=True)

                result_content = []
                for content in tool_result.content:
                    if hasattr(content, 'text'):
                        result_content.append(content.text)

                result_text = "\n".join(result_content)
                if not result_text:
                    result_text = json.dumps(tool_result.content, default=str)

                if self.verbose:
                    print(f"[GITHUB AGENT] Result: {result_text[:500]}")

                self.stats.record_operation(tool_name, True, retry_count)

                return result_text, None

            except Exception as e:
                error_str = str(e).lower()
                is_retryable = any(
                    retryable in error_str
                    for retryable in RetryConfig.RETRYABLE_ERRORS
                )

                if self.verbose or retry_count > 0:
                    print(f"[GITHUB AGENT] Error calling {tool_name}: {str(e)}")

                if is_retryable and retry_count < RetryConfig.MAX_RETRIES:
                    retry_count += 1

                    if self.verbose:
                        print(f"[GITHUB AGENT] Retrying in {delay:.1f}s...")

                    await asyncio.sleep(delay)
                    delay = min(delay * RetryConfig.BACKOFF_FACTOR, RetryConfig.MAX_DELAY)
                    continue
                else:
                    error_msg = self._format_tool_error(tool_name, str(e), tool_args)
                    self.stats.record_operation(tool_name, False, retry_count)

                    # Log tool call failure
                    if self.logger:
                        self.logger.log_tool_call(self.agent_name, tool_name, None, success=False, error=str(e))

                    return None, error_msg

        # Log max retries exceeded
        if self.logger:
            self.logger.log_tool_call(self.agent_name, tool_name, None, success=False, error="Max retries exceeded")

        return None, f"Max retries exceeded for {tool_name}"

    async def _send_function_response(
        self,
        chat,
        tool_name: str,
        result_text: Optional[str],
        error_msg: Optional[str]
    ):
        """Send function call result back to LLM"""
        if result_text is not None:
            response_data = {"result": result_text}
        else:
            response_data = {"error": error_msg}

        return await chat.send_message_async(
            genai.protos.Content(
                parts=[genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=tool_name,
                        response=response_data
                    )
                )]
            )
        )

    # ========================================================================
    # ERROR HANDLING AND FORMATTING
    # ========================================================================

    def _classify_error(self, error: str) -> ErrorType:
        """Classify an error into a specific type for better handling"""
        error_lower = error.lower()

        if "authentication" in error_lower or "unauthorized" in error_lower or "401" in error:
            return ErrorType.AUTHENTICATION
        elif "not found" in error_lower or "404" in error:
            return ErrorType.NOT_FOUND
        elif "forbidden" in error_lower or "403" in error:
            return ErrorType.PERMISSION
        elif "validation" in error_lower or "422" in error:
            return ErrorType.VALIDATION
        elif "rate limit" in error_lower or "429" in error:
            return ErrorType.RATE_LIMIT
        elif "branch" in error_lower and "protected" in error_lower:
            return ErrorType.BRANCH_PROTECTION
        elif "invalid" in error_lower and ("owner/repo" in error_lower or "repository" in error_lower):
            return ErrorType.INVALID_REPO_FORMAT
        elif any(net in error_lower for net in ["timeout", "connection", "network"]):
            return ErrorType.NETWORK
        else:
            return ErrorType.UNKNOWN

    def _format_tool_error(self, tool_name: str, error: str, args: Dict) -> str:
        """Format tool errors with helpful, context-aware messages"""
        error_type = self._classify_error(error)

        if error_type == ErrorType.AUTHENTICATION:
            return (
                f"🔐 Authentication error when calling {tool_name}. "
                "Your GitHub token may be invalid, expired, or missing required scopes. "
                "Check GITHUB_PERSONAL_ACCESS_TOKEN and verify it has 'repo', 'read:org', and 'user' scopes."
            )

        elif error_type == ErrorType.NOT_FOUND:
            repo = args.get('repo') or args.get('repository') or args.get('owner')
            if repo:
                return (
                    f"🔍 Repository or resource '{repo}' not found. "
                    "Verify the repository name is correct (owner/repo format) and you have access to it."
                )
            return f"🔍 Resource not found for {tool_name}. Please verify the repository, issue, or PR exists and is accessible."

        elif error_type == ErrorType.PERMISSION:
            return (
                f"🚫 Permission denied for {tool_name}. "
                "You may not have write access to this repository or the resource is restricted. "
                "Check your token scopes and repository permissions."
            )

        elif error_type == ErrorType.VALIDATION:
            return f"⚠️ Validation error for {tool_name}. The provided data may not match GitHub's requirements. Details: {error}"

        elif error_type == ErrorType.RATE_LIMIT:
            return (
                f"⏳ GitHub API rate limit reached. "
                "Please wait before making more requests. Authenticated requests have higher limits. "
                "Consider reducing the frequency of operations."
            )

        elif error_type == ErrorType.BRANCH_PROTECTION:
            return (
                f"🛡️ Branch protection rules prevent this operation. "
                "The branch may require pull request reviews, status checks, or administrator privileges."
            )

        elif error_type == ErrorType.INVALID_REPO_FORMAT:
            return (
                f"📝 Invalid repository format. "
                "Use 'owner/repo' format (e.g., 'facebook/react', not just 'react')."
            )

        elif error_type == ErrorType.NETWORK:
            return f"🌐 Network error when calling {tool_name}: {error}. The system will automatically retry."

        else:
            return f"❌ Error calling {tool_name}: {error}"

    # ========================================================================
    # CAPABILITIES AND INFORMATION
    # ========================================================================

    async def get_capabilities(self) -> List[str]:
        """Return GitHub capabilities in user-friendly format with limitations"""
        if not self.available_tools:
            return ["GitHub operations (initializing...)"]

        # Return curated list with clear capabilities and limitations
        return [
            "✓ Manage issues and pull requests",
            "✓ Perform code reviews and analyze PRs",
            "✓ Search code and commits",
            "✓ Read repository files and structure",
            "✓ Work with branches and commits",
            "✓ Get commit metadata (SHA, author, message, date)",
            "✗ Cannot fetch: Raw diff/patch content (use GitHub UI or API directly)",
            "✗ Cannot: Modify repository settings or access controls",
            "✗ Cannot: Create or delete repositories",
            "✗ Cannot: Manage webhooks or integrations",
        ]

    async def validate_operation(self, instruction: str) -> Dict[str, Any]:
        """
        Validate if a GitHub operation can be performed (Feature #14)

        Uses cached metadata to quickly check if the operation is likely to succeed.

        Args:
            instruction: The instruction to validate

        Returns:
            Dict with validation results
        """
        result = {
            'valid': True,
            'missing': [],
            'warnings': [],
            'confidence': 1.0
        }

        instruction_lower = instruction.lower()

        # Check if we're working with a specific repository
        if any(word in instruction_lower for word in ['create issue', 'create pr', 'pull request']):
            # Check if we have repository metadata
            if not self.metadata_cache.get('repositories'):
                result['warnings'].append("No repository metadata cached - operation may be slow")
                result['confidence'] = 0.7

            # Try to extract repo name from instruction
            repositories = self.metadata_cache.get('repositories', {})
            if repositories:
                # Check if instruction mentions a known repo
                mentioned_repo = None
                for repo_full_name in repositories.keys():
                    if repo_full_name.lower() in instruction_lower:
                        mentioned_repo = repo_full_name
                        break

                if not mentioned_repo:
                    result['missing'].append("repository name (owner/repo)")
                    result['valid'] = False
                    result['confidence'] = 0.3

        # Check if agent is initialized
        if not self.initialized:
            result['valid'] = False
            result['missing'].append("agent initialization")
            result['confidence'] = 0.0

        return result

    def get_stats(self) -> str:
        """Get operation statistics summary"""
        return self.stats.get_summary()

    # ========================================================================
    # CLEANUP AND RESOURCE MANAGEMENT
    # ========================================================================

    async def _cleanup_connection(self):
        """Internal cleanup helper for MCP connection resources"""
        # Close session if it was successfully entered
        if self.session and self.session_entered:
            try:
                await self.session.__aexit__(None, None, None)
            except Exception as e:
                # Suppress all cleanup errors to prevent cascading failures
                if self.verbose:
                    print(f"[GITHUB AGENT] Suppressed session cleanup error: {e}")
            finally:
                self.session = None
                self.session_entered = False

        # Close stdio context if it was successfully entered
        if self.stdio_context and self.stdio_context_entered:
            try:
                await self.stdio_context.__aexit__(None, None, None)
            except Exception as e:
                # Suppress all cleanup errors
                if self.verbose:
                    print(f"[GITHUB AGENT] Suppressed stdio cleanup error: {e}")
            finally:
                self.stdio_context = None
                self.stdio_context_entered = False

    async def cleanup(self):
        """Disconnect from GitHub and clean up resources"""
        if self.verbose:
            print(f"\n[GITHUB AGENT] Cleaning up. {self.stats.get_summary()}")

        await self._cleanup_connection()

    # ========================================================================
    # SCHEMA CONVERSION HELPERS
    # ========================================================================

    def _build_function_declaration(self, tool: Any) -> protos.FunctionDeclaration:
        """Convert MCP tool schema to Gemini function declaration"""
        parameters_schema = protos.Schema(type_=protos.Type.OBJECT)

        if tool.inputSchema:
            schema = tool.inputSchema
            if "properties" in schema:
                for prop_name, prop_schema in schema["properties"].items():
                    parameters_schema.properties[prop_name] = self._clean_schema(prop_schema)

            if "required" in schema:
                parameters_schema.required.extend(schema["required"])

        return protos.FunctionDeclaration(
            name=tool.name,
            description=tool.description or "",
            parameters=parameters_schema
        )

    def _clean_schema(self, schema: Dict) -> protos.Schema:
        """Convert JSON schema to protobuf schema recursively"""
        schema_pb = protos.Schema()

        if "type" in schema:
            schema_pb.type_ = self.schema_type_map.get(
                schema["type"],
                protos.Type.TYPE_UNSPECIFIED
            )

        if "description" in schema:
            schema_pb.description = schema["description"]

        if "enum" in schema:
            schema_pb.enum.extend(schema["enum"])

        if "items" in schema and isinstance(schema["items"], dict):
            schema_pb.items = self._clean_schema(schema["items"])

        if "properties" in schema and isinstance(schema["properties"], dict):
            for prop_name, prop_schema in schema["properties"].items():
                schema_pb.properties[prop_name] = self._clean_schema(prop_schema)

        if "required" in schema:
            schema_pb.required.extend(schema["required"])

        return schema_pb

    def _deep_convert_proto_args(self, value: Any) -> Any:
        """Recursively convert protobuf types to standard Python types"""
        type_str = str(type(value))

        if "MapComposite" in type_str:
            return {k: self._deep_convert_proto_args(v) for k, v in value.items()}
        elif "RepeatedComposite" in type_str:
            return [self._deep_convert_proto_args(item) for item in value]
        else:
            return value
