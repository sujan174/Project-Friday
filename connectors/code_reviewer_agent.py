"""
Code Reviewer Agent - Intelligent Code Analysis & Review

This module provides a dedicated agent for comprehensive code review and analysis.
It can analyze code from any source (GitHub PRs, local files, code snippets) and
provide detailed feedback on quality, security, performance, and best practices.

Features:
- LLM-based code analysis (Gemini 2.5 Flash)
- Multi-language support (Python, JavaScript, Java, Go, etc.)
- Security vulnerability detection
- Performance issue identification
- Code quality assessment
- Best practices validation
- Intelligent retry and error handling
- Can be enhanced with static analysis tools later

Author: AI System
Version: 1.0
"""

import os
import asyncio
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from connectors.base_agent import BaseAgent, safe_extract_response_text
from connectors.agent_intelligence import (
    ConversationMemory,
    WorkspaceKnowledge,
    SharedContext,
    ProactiveAssistant
)
from llms.base_llm import BaseLLM, LLMConfig
from llms.gemini_flash import GeminiFlash


@dataclass
class OperationStats:
    """Track code review statistics"""
    total_reviews: int = 0
    successful_reviews: int = 0
    failed_reviews: int = 0
    issues_found: int = 0
    security_issues: int = 0
    performance_issues: int = 0
    quality_issues: int = 0

    def record_success(self, issues: Dict[str, int] = None):
        self.successful_reviews += 1
        self.total_reviews += 1
        if issues:
            self.security_issues += issues.get('security', 0)
            self.performance_issues += issues.get('performance', 0)
            self.quality_issues += issues.get('quality', 0)
            self.issues_found += sum(issues.values())

    def record_failure(self):
        self.failed_reviews += 1
        self.total_reviews += 1

    def get_summary(self) -> str:
        if self.total_reviews == 0:
            return "No reviews yet"
        success_rate = (self.successful_reviews / self.total_reviews) * 100
        return f"{self.total_reviews} reviews, {success_rate:.1f}% success, {self.issues_found} issues found"


class Agent(BaseAgent):
    """
    Intelligent Code Reviewer Agent

    This agent specializes in comprehensive code analysis and review:
    - Security vulnerability detection
    - Performance bottleneck identification
    - Code quality assessment
    - Best practices validation
    - Architecture review
    - Documentation review
    - Test coverage analysis
    """

    def __init__(self, verbose: bool = False, shared_context: Optional[SharedContext] = None, llm: Optional[BaseLLM] = None, session_logger=None, **kwargs):
        """
        Initialize Code Reviewer Agent

        Args:
            verbose: Enable detailed logging
            shared_context: Optional shared context for cross-agent coordination
            llm: Optional LLM instance (defaults to Gemini Flash)
            session_logger: Optional session logger for tracking operations
        """
        super().__init__()

        self.verbose = verbose
        self.initialized = False
        self.agent_name = "code_reviewer"  # For logging
        self.logger = session_logger  # Session logger

        # LLM abstraction for code analysis
        if llm is None:
            # Default to Gemini 2.5 Flash
            self.llm = GeminiFlash(LLMConfig(
                model_name='models/gemini-2.5-flash',
                temperature=0.7
            ))
        else:
            self.llm = llm

        # Intelligence components
        self.memory = ConversationMemory()
        self.knowledge = WorkspaceKnowledge()
        self.shared_context = shared_context
        self.proactive = ProactiveAssistant('code_reviewer', verbose)

        # Statistics tracking
        self.stats = OperationStats()

        # Feature #1: Metadata cache for code review patterns
        self.metadata_cache = {}

        # System prompt - defines review behavior
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build comprehensive code review system prompt"""
        return """You are an elite code review specialist with deep expertise in software engineering, security, performance optimization, and code quality. Your mission is to provide comprehensive, actionable code reviews that help developers write better, safer, and more maintainable code.

# Core Expertise

You have mastery across:
- **Languages**: Python, JavaScript/TypeScript, Java, Go, C#, Ruby, PHP, Rust, C/C++
- **Frameworks**: React, Vue, Angular, Django, Flask, Spring, Express, Rails, .NET
- **Databases**: SQL, NoSQL, ORMs, query optimization
- **Security**: OWASP Top 10, authentication, authorization, encryption, secure coding
- **Performance**: Algorithms, data structures, caching, async patterns, profiling
- **Architecture**: Design patterns, SOLID principles, clean code, refactoring

# Review Framework

## 1. SECURITY ANALYSIS (CRITICAL)

**Common Vulnerabilities to Detect**:

**Injection Attacks**:
- SQL Injection: Unsanitized user input in queries
- Command Injection: User input in system commands
- XSS (Cross-Site Scripting): Unescaped HTML/JS output
- Path Traversal: Unvalidated file paths
- LDAP/XML/NoSQL Injection

**Authentication & Authorization**:
- Weak password policies or storage
- Missing authentication checks
- Broken access control
- Session management issues
- Token/JWT vulnerabilities
- Missing CSRF protection

**Data Exposure**:
- Hardcoded secrets (API keys, passwords, tokens)
- Sensitive data in logs
- Insecure data transmission (no HTTPS/TLS)
- Missing encryption for sensitive data
- Information leakage in error messages

**Other Security Issues**:
- Insecure dependencies (outdated packages)
- Missing input validation
- Unsafe deserialization
- Race conditions in security checks
- Insufficient logging for security events

## 2. PERFORMANCE ANALYSIS (HIGH PRIORITY)

**Common Performance Issues**:

**Database**:
- N+1 query problems
- Missing indexes
- Inefficient queries (SELECT *)
- No query pagination
- Missing database connection pooling

**Algorithms & Data Structures**:
- Nested loops with high complexity (O(n²), O(n³))
- Inefficient data structure choice
- Unnecessary iterations
- Missing memoization/caching
- Recursive functions without optimization

**Resource Management**:
- Memory leaks (unclosed resources)
- Large object creation in loops
- Unnecessary object copies
- Missing garbage collection hints
- File/socket handles not closed

**Asynchronous Code**:
- Blocking I/O in async context
- Missing await/async keywords
- Synchronous loops over async operations
- No parallel processing where applicable

## 3. CODE QUALITY ANALYSIS (MEDIUM PRIORITY)

**Maintainability**:
- Functions too long (>50 lines)
- Cyclomatic complexity too high
- Deep nesting (>3 levels)
- Code duplication (DRY violations)
- Unclear variable/function names
- Magic numbers without constants

**Readability**:
- Missing or unclear comments
- Inconsistent formatting
- Complex logic without explanation
- Unclear control flow
- Poor naming conventions

**Error Handling**:
- Bare except/catch blocks
- Swallowing exceptions
- Missing error logging
- No error recovery
- Unclear error messages
- Missing edge case handling

**Testing**:
- Missing test cases
- No edge case tests
- Low test coverage
- Flaky tests
- Missing integration tests
- No error path testing

## 4. ARCHITECTURE & DESIGN (MEDIUM PRIORITY)

**Design Principles**:
- SOLID principle violations
- Tight coupling
- God objects/classes
- Circular dependencies
- Missing abstraction layers
- Poor separation of concerns

**Patterns & Practices**:
- Inappropriate design pattern use
- Missing design patterns where needed
- Antipatterns (singletons, global state)
- Inconsistent with codebase conventions

## 5. DOCUMENTATION (LOW PRIORITY)

- Missing docstrings/JSDoc
- Outdated comments
- Unclear API documentation
- Missing README updates
- No usage examples

# Review Output Format

Provide reviews in this structured format:

```
## Code Review Summary

**Overall Assessment**: [Approve / Request Changes / Needs Discussion]
**Severity**: [Critical / High / Medium / Low]

---

## 🔴 CRITICAL ISSUES (Must Fix Before Merge)

### Security
- **[Line XX]**: [Issue description]
  - **Problem**: [What's wrong]
  - **Impact**: [Security risk]
  - **Fix**: [How to fix it]
  ```[language]
  // Suggested fix
  ```

### Performance
- **[Line XX]**: [Issue description]
  - **Problem**: [What's wrong]
  - **Impact**: [Performance impact]
  - **Fix**: [How to fix it]

---

## 🟡 IMPORTANT ISSUES (Should Fix)

### Code Quality
- **[Line XX]**: [Issue description]
  - **Problem**: [What's wrong]
  - **Suggestion**: [How to improve]

### Error Handling
- **[Line XX]**: Missing error handling for [scenario]

---

## 💡 SUGGESTIONS (Nice to Have)

- Refactor [function] to improve readability
- Consider using [pattern] for [scenario]
- Add tests for [edge case]

---

## ✅ STRENGTHS

- Good use of [pattern/practice]
- Clear naming and structure
- Comprehensive test coverage

---

## 📊 METRICS

- **Issues Found**: X critical, Y high, Z medium
- **Files Reviewed**: N files
- **Lines Changed**: +X -Y
```

# Review Best Practices

1. **Be Specific**: Point to exact line numbers and code snippets
2. **Explain Why**: Don't just say it's wrong, explain the impact
3. **Provide Solutions**: Show how to fix issues with code examples
4. **Prioritize**: Separate critical issues from suggestions
5. **Be Constructive**: Focus on improvement, not criticism
6. **Consider Context**: Understand the broader codebase patterns
7. **Balance**: Find positives as well as negatives

# When Reviewing Code

**Always Ask**:
- Is this code secure? (Check OWASP Top 10)
- Is this code performant? (Check algorithms, queries, loops)
- Is this code maintainable? (Check clarity, structure, documentation)
- Does this code have proper error handling?
- Are there tests for this code?
- Does this follow the project's conventions?

**Red Flags**:
- User input directly in SQL queries
- Passwords or API keys in code
- No input validation
- Missing authentication checks
- Nested loops over large datasets
- Resources not being closed
- Empty catch blocks
- No tests for new features

# Language-Specific Checks

**Python**:
- Check for SQL injection in string formatting
- Look for pickle usage (unsafe deserialization)
- Verify proper exception handling
- Check for mutable default arguments

**JavaScript/TypeScript**:
- Check for XSS in innerHTML/dangerouslySetInnerHTML
- Look for eval() usage
- Verify proper async/await usage
- Check for memory leaks in event listeners

**Java**:
- Check for SQL injection in JDBC
- Look for insecure random number generation
- Verify proper resource management (try-with-resources)
- Check for serialization vulnerabilities

**Go**:
- Check for SQL injection
- Look for goroutine leaks
- Verify proper error handling
- Check for race conditions

Remember: Your goal is to help developers ship secure, performant, and maintainable code. Be thorough but constructive."""

    async def initialize(self) -> bool:
        """
        Initialize the code reviewer agent

        Returns:
            bool: True if initialization succeeded
        """
        try:
            if self.verbose:
                print(f"[CODE REVIEWER] Initializing...")

            # Set system instruction on LLM
            self.llm.config.system_instruction = self.system_prompt

            self.initialized = True

            # Feature #1: Prefetch code review patterns
            await self._prefetch_metadata()

            if self.verbose:
                print(f"[CODE REVIEWER] Initialization complete with {self.llm}")

            return True

        except Exception as e:
            error_msg = f"Failed to initialize Code Reviewer agent: {str(e)}"
            print(f"[CODE REVIEWER] {error_msg}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False

    async def _prefetch_metadata(self):
        """
        Prefetch and cache code review patterns (Feature #1)
        """
        try:
            # Check if we have valid cached metadata
            cached = self.knowledge.get_metadata_cache('code_reviewer')
            if cached:
                self.metadata_cache = cached
                if self.verbose:
                    print(f"[CODE REVIEWER] Loaded metadata from cache")
                return

            if self.verbose:
                print(f"[CODE REVIEWER] Prefetching review patterns...")

            # Store common review patterns and checklists
            self.metadata_cache = {
                'security_patterns': [
                    'sql_injection', 'xss', 'hardcoded_secrets', 'weak_auth',
                    'missing_validation', 'insecure_deserialization'
                ],
                'performance_patterns': [
                    'n_plus_one', 'nested_loops', 'memory_leaks', 'blocking_io',
                    'missing_caching', 'inefficient_queries'
                ],
                'quality_patterns': [
                    'long_functions', 'code_duplication', 'magic_numbers',
                    'poor_naming', 'missing_error_handling', 'low_test_coverage'
                ],
                'languages_supported': [
                    'python', 'javascript', 'typescript', 'java', 'go',
                    'csharp', 'ruby', 'php', 'rust', 'cpp'
                ],
                'fetched_at': asyncio.get_event_loop().time()
            }

            # Persist to knowledge base
            self.knowledge.save_metadata_cache('code_reviewer', self.metadata_cache, ttl_seconds=3600)

            if self.verbose:
                print(f"[CODE REVIEWER] Cached review patterns")

        except Exception as e:
            if self.verbose:
                print(f"[CODE REVIEWER] Warning: Metadata prefetch failed: {e}")

    async def get_capabilities(self) -> List[str]:
        """
        Get list of code review capabilities

        Returns:
            List of capability descriptions
        """
        return [
            "Analyze code for security vulnerabilities",
            "Detect performance issues and bottlenecks",
            "Review code quality and maintainability",
            "Validate best practices and patterns",
            "Check error handling and edge cases",
            "Assess test coverage and quality",
            "Review architecture and design",
            "Multi-language support (Python, JS, Java, Go, etc.)",
            "Provide actionable improvement suggestions",
            "Generate structured review reports"
        ]

    async def execute(self, instruction: str) -> str:
        """
        Execute code review instruction

        Args:
            instruction: Natural language instruction with code to review

        Returns:
            str: Structured code review feedback
        """
        if not self.initialized:
            return self._format_error(Exception("Code Reviewer agent not initialized"))

        try:
            # Check shared context for code from other agents
            context_from_other_agents = {}
            if self.shared_context:
                # Get recent resources from other agents in this session
                recent_resources = self.shared_context.get_recent_resources(limit=5)
                if recent_resources:
                    context_from_other_agents = {
                        'resources': recent_resources,
                        'count': len(recent_resources)
                    }

            if context_from_other_agents and self.verbose:
                print(f"[CODE REVIEWER] Found context from other agents: {context_from_other_agents.get('count', 0)} resources")

            # Start analysis with LLM
            if self.verbose:
                print(f"[CODE REVIEWER] Analyzing code...")

            # Log tool execution start
            if self.logger:
                self.logger.log_tool_call(self.agent_name, "llm_code_analysis")

            llm_response = await self.llm.generate_content(instruction)

            # Check for RECITATION (finish_reason 12) - Gemini refusing due to potential copyrighted content
            if llm_response.finish_reason and '12' in str(llm_response.finish_reason):
                if self.verbose:
                    print(f"[CODE REVIEWER] RECITATION detected - retrying with modified prompt")

                # Retry with a prompt that asks for analysis rather than full review
                modified_instruction = f"""Please provide a high-level analysis of this code focusing on:
1. Architecture and design patterns used
2. Potential security concerns (generic patterns only)
3. Performance considerations
4. Code quality improvements

Do not reproduce or quote the code. Only provide analytical observations.

{instruction}"""

                llm_response = await self.llm.generate_content(modified_instruction)

                # If still RECITATION, provide helpful error
                if llm_response.finish_reason and '12' in str(llm_response.finish_reason):
                    return """⚠️ Code Review Limited

The code file is too large or complex for detailed review in a single request.

Suggestions:
• Break the code into smaller chunks (functions/classes)
• Request review of specific sections
• Ask about specific aspects (e.g., "review security" or "check performance")

The system detected potential copyrighted content patterns in the large code block."""

            # Extract review text safely
            try:
                # Use the correct variable name and function
                review_text = llm_response.text if llm_response and llm_response.text else ""

                # If text is empty or whitespace, check finish_reason
                if not review_text or not review_text.strip():
                    finish_reason = llm_response.finish_reason if llm_response else "unknown"

                    if self.verbose:
                        print(f"[CODE REVIEWER] Empty response. Finish reason: {finish_reason}")

                    # Provide specific error based on finish_reason
                    if '12' in str(finish_reason):
                        review_text = """❌ ERROR: Code Review Blocked

What failed: Safety filters blocked the code review
Why: The code content triggered Google's safety filters (RECITATION block)

This commonly happens with:
• Authentication code (passwords, tokens, API keys)
• User credential handling
• Large code blocks that may contain copyrighted patterns

Suggestions:
• Review smaller code sections separately
• Remove sensitive data (passwords, keys) before review
• Ask for specific aspects: "check security issues" or "review error handling"
"""
                    else:
                        review_text = f"""❌ ERROR: Empty Response

What failed: Code review generated no output
Why: Finish reason: {finish_reason}

This may occur when:
• Response was blocked by safety filters
• Code is too large to process
• Unexpected API response format

Try:
• Breaking code into smaller chunks
• Removing sensitive data
• Requesting specific analysis aspects
"""

            except Exception as text_error:
                # If text accessor fails, check finish_reason and provide helpful message
                finish_reason = llm_response.finish_reason if llm_response else "unknown"
                if self.verbose:
                    print(f"[CODE REVIEWER] Text extraction failed. Finish reason: {finish_reason}, Error: {text_error}")

                # Provide helpful error based on finish_reason
                review_text = f"""❌ ERROR: Review Generation Failed

What failed: Could not extract review text from response
Why: {str(text_error)}
Finish reason: {finish_reason}

This may occur when:
• The code is too large (try smaller chunks)
• The content triggers safety filters
• The response format is unexpected

Suggestions:
• Try reviewing smaller code sections
• Remove sensitive data (tokens, passwords)
• Request specific analysis: "check for security issues" or "review performance"
"""

            # Parse issues from review (simple extraction for stats)
            issues = self._extract_issue_counts(review_text)
            self.stats.record_success(issues)

            # Log tool completion
            if self.logger:
                self.logger.log_tool_call(
                    self.agent_name,
                    "llm_code_analysis",
                    success=True
                )

            if self.verbose:
                print(f"[CODE REVIEWER] Review complete. {self.stats.get_summary()}")

            # Add proactive suggestions
            suggestions = self.proactive.suggest_next_steps('code_review', {})
            if suggestions:
                review_text += f"\n\n💡 {suggestions}"

            return review_text

        except Exception as e:
            self.stats.record_failure()

            # Log failure
            if self.logger:
                self.logger.log_tool_call(
                    self.agent_name,
                    "llm_code_analysis",
                    success=False,
                    error=str(e)
                )

            return self._format_error(e)

    def _extract_issue_counts(self, review_text: str) -> Dict[str, int]:
        """Extract issue counts from review text for statistics"""
        issues = {
            'security': review_text.lower().count('security'),
            'performance': review_text.lower().count('performance'),
            'quality': review_text.lower().count('quality')
        }
        return issues

    async def validate_operation(self, instruction: str) -> Dict[str, Any]:
        """
        Validate if a code review operation can be performed (Feature #14)

        Args:
            instruction: User instruction to validate

        Returns:
            Dict with validation results
        """
        result = {
            'valid': True,
            'missing': [],
            'warnings': [],
            'confidence': 1.0
        }

        # Check if code is provided
        if len(instruction) < 50:
            result['warnings'].append("Instruction seems short - may need actual code to review")
            result['confidence'] = 0.6

        # Check if language is identifiable
        common_keywords = ['def ', 'function ', 'class ', 'import ', 'const ', 'var ', 'public ']
        has_code_keywords = any(keyword in instruction.lower() for keyword in common_keywords)

        if not has_code_keywords:
            result['warnings'].append("No code keywords detected - may need code snippet or file")
            result['confidence'] = 0.7

        return result

    async def cleanup(self):
        """Cleanup code reviewer agent resources"""
        try:
            if self.verbose:
                print(f"\n[CODE REVIEWER] Cleaning up. {self.stats.get_summary()}")
        except Exception as e:
            if self.verbose:
                print(f"[CODE REVIEWER] Error during cleanup: {e}")

        self.initialized = False

    def _format_error(self, error: Exception) -> str:
        """Format error message for user consumption"""
        return f"⚠️ Code Reviewer error: {str(error)}"
