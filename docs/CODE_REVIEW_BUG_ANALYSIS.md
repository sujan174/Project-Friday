# Code Review Bug Analysis

## Issue Report

**User Request:** "review the codes in the upscale folder"

**What Should Have Happened:**
1. GitHub agent fetches all files in upscale folder
2. Code Reviewer agent analyzes each file
3. Comprehensive review presented to user

**What Actually Happened:**
1. GitHub agent fetched only 1 file (handleResult.js) ✅
2. LLM hallucinated the content of handleUpload.js ❌
3. Code Reviewer agent was NEVER called ❌
4. Main orchestrator LLM analyzed code directly ❌

---

## Root Causes

### 1. **Task Decomposer Not Used**

**Location:** `orchestrator.py:156-159`

```python
self.task_decomposer = TaskDecomposer(
    agent_capabilities=self.agent_capabilities,
    verbose=self.verbose
)
```

**Problem:** Task decomposer is initialized but **NEVER called**. No usage found in entire orchestrator.

**Impact:** Multi-step operations like "fetch code then review it" are not coordinated. The orchestrator relies on Gemini's function calling, which doesn't understand dependencies between tasks.

**What Task Decomposer Would Do:**
```python
# Input: "review the codes in the upscale folder"

# Task Decomposer Output:
Task 1: Fetch all files in upscale folder
  Agent: github
  Dependencies: None

Task 2: Review fetched code
  Agent: code_reviewer
  Dependencies: [Task 1]  # Must wait for files
  Input: Output from Task 1
```

### 2. **LLM Hallucination**

**Evidence from logs** (`logs/session_8abd8434-aead-4e07-b8a9-72a6e3aa3c58.log`):

```
[+12072.40s] → github
  Message: Get the content of the file 'controllers/upscale/handleResult.js'
[+12076.72s] ← github ✓
  Response: The content of `controllers/upscale/handleResult.js`...

# Log ends here - handleUpload.js was NEVER fetched!
```

But the LLM response included:
```json
{"use_github_agent_response": {"result": "The content of controllers/upscale/handleUpload.js is:\n\njavascript\n// Developed by Sujan H\n..."}}
```

**Problem:** When asked about multiple files, if only some are fetched, the LLM fills in the gaps with plausible-sounding but completely fabricated code.

### 3. **Code Reviewer Agent Never Invoked**

**Evidence:** Session log shows only GitHub agent calls:
```
AGENT METRICS:
GITHUB: Messages: 3 received, 3 sent
SLACK:  Messages: 2 received, 2 sent
# code_reviewer: NOT IN LOG!
```

**Why:** Gemini's function calling chose GitHub agent to fulfill the request, and after getting file contents, the main orchestrator LLM analyzed them directly instead of routing to code_reviewer.

**Agent Routing Logic** (`orchestrator.py:592-610`):
```python
for agent_name, capabilities in self.agent_capabilities.items():
    tool = protos.FunctionDeclaration(
        name=f"use_{agent_name}_agent",
        description=f"""Use the {agent_name} agent to perform tasks.

Capabilities: {', '.join(capabilities)}
"""
    )
```

Gemini saw:
- `use_github_agent` - "Read GitHub files, PRs, repos"
- `use_code_reviewer_agent` - "Analyze code for security, quality"

It chose GitHub to "read" the code, but never realized it should also call code_reviewer to "analyze" it.

---

## Fixes Required

### Fix 1: Enable Task Decomposer for Complex Requests

**Location:** `orchestrator.py:process_message()`

**Current Flow:**
```python
# Line 929
intelligence = self._process_with_intelligence(user_message)
message_to_send = intelligence.get('resolved_message', user_message)

# Line 937 - Send directly to LLM
response = self.chat.send_message(message_to_send)
```

**Proposed Fix:**
```python
# After intelligence analysis
intelligence = self._process_with_intelligence(user_message)

# NEW: Check if task decomposition needed
if len(intelligence['intents']) > 1 or self._is_complex_request(intelligence):
    # Use task decomposer for multi-step operations
    execution_plan = self.task_decomposer.decompose(
        message=user_message,
        intents=intelligence['intents'],
        entities=intelligence['entities'],
        context=intelligence['context']
    )

    if execution_plan.tasks:
        # Execute tasks in dependency order
        return await self._execute_plan(execution_plan)

# Fall back to normal LLM routing for simple requests
response = self.chat.send_message(message_to_send)
```

**Add Helper:**
```python
def _is_complex_request(self, intelligence: Dict) -> bool:
    """
    Detect if request requires task decomposition

    Complex requests include:
    - Multiple intents (e.g., "fetch and review")
    - Analyze intent with entities (e.g., "review code")
    - Multi-agent operations
    """
    intents = intelligence['intents']

    # Multiple intents = complex
    if len(intents) > 1:
        return True

    # ANALYZE intent usually needs prior READ
    if any(i.type == IntentType.ANALYZE for i in intents):
        entities = intelligence['entities']
        # Analyzing code/files requires fetching first
        if any(e.type in [EntityType.CODE, EntityType.FILE] for e in entities):
            return True

    return False
```

### Fix 2: Implement Task Execution Pipeline

**Location:** New method in `orchestrator.py`

```python
async def _execute_plan(self, execution_plan: ExecutionPlan) -> str:
    """
    Execute multi-task plan with dependencies

    Args:
        execution_plan: Execution plan from task decomposer

    Returns:
        Final result after executing all tasks
    """
    results = {}  # Store task results by ID

    # Get tasks in dependency order
    ordered_tasks = execution_plan.get_execution_order()

    for task in ordered_tasks:
        if self.verbose:
            print(f"\n{C.CYAN}📋 Executing Task: {task.description}{C.ENDC}")
            print(f"  Agent: {task.agent_name}")
            if task.dependencies:
                print(f"  Depends on: {task.dependencies}")

        # Gather dependencies' results
        dependency_data = {}
        for dep_id in task.dependencies:
            if dep_id in results:
                dependency_data[dep_id] = results[dep_id]

        # Build instruction with dependency context
        instruction = task.instruction
        if dependency_data:
            # Inject previous results into instruction
            context_str = "\n\n".join([
                f"Result from previous step:\n{res}"
                for res in dependency_data.values()
            ])
            instruction = f"{context_str}\n\n{instruction}"

        # Execute task with appropriate agent
        agent = self.sub_agents.get(task.agent_name)
        if not agent:
            results[task.id] = f"Error: Agent {task.agent_name} not available"
            continue

        try:
            result = await agent.execute(instruction, context=dependency_data)
            results[task.id] = result

            if self.verbose:
                print(f"  {C.GREEN}✓ Completed{C.ENDC}")

        except Exception as e:
            error_msg = f"Error executing task: {e}"
            results[task.id] = error_msg

            if self.verbose:
                print(f"  {C.RED}✗ Failed: {error_msg}{C.ENDC}")

            # Check if failure blocks other tasks
            if task.is_critical:
                return f"Critical task failed: {error_msg}"

    # Return final result (from last task or aggregated)
    final_task = ordered_tasks[-1]
    return results.get(final_task.id, "Execution completed")
```

### Fix 3: Improve Agent Descriptions for Better Routing

**Location:** `orchestrator.py:_create_agent_tools()`

**Current:**
```python
description=f"""Use the {agent_name} agent to perform tasks.

Capabilities: {', '.join(capabilities)}
```

**Improved:**
```python
# Enhanced descriptions
AGENT_DESCRIPTIONS = {
    'github': """Fetch code, files, PRs, and repositories from GitHub.

    Use this agent to READ/FETCH content. If you need to ANALYZE the fetched content,
    you must call another agent (like code_reviewer) after getting the data.
    """,

    'code_reviewer': """Analyze and review code for quality, security, performance.

    Use this agent to REVIEW code that has already been fetched. Pass the code content
    to this agent for comprehensive analysis.

    IMPORTANT: If code needs to be fetched first, call the appropriate fetch agent
    (github/browser) before calling this agent.
    """,
}

description = AGENT_DESCRIPTIONS.get(agent_name,
    f"Use the {agent_name} agent to perform tasks.\n\nCapabilities: {', '.join(capabilities)}"
)
```

### Fix 4: Add Explicit Multi-Step Prompting

**Location:** `orchestrator.py:system_prompt`

**Add to system prompt:**
```python
self.system_prompt += """

MULTI-STEP OPERATION RULES:

1. When user asks to ANALYZE/REVIEW code or files:
   a. FIRST: Fetch the content using appropriate agent (github/browser/scraper)
   b. THEN: Call code_reviewer agent with the fetched content
   c. NEVER analyze code yourself - always use code_reviewer agent

2. When user asks to "review code in [location]":
   - Step 1: use_github_agent to fetch files from [location]
   - Step 2: use_code_reviewer_agent with fetched content

3. ALWAYS call specialized agents instead of providing your own analysis.

Examples:
- "review the upscale folder" → use_github_agent (fetch) → use_code_reviewer_agent (analyze)
- "analyze PR #123" → use_github_agent (fetch PR) → use_code_reviewer_agent (review changes)
"""
```

### Fix 5: Prevent Hallucination with Explicit Fetching

**Location:** `connectors/github_agent.py`

**Add method to fetch multiple files:**
```python
async def fetch_all_files_in_folder(self, repo: str, folder_path: str) -> List[Dict]:
    """
    Fetch contents of all files in a folder

    Returns:
        List of {filename, content, path} dicts
    """
    # List files
    files = await self.list_folder_contents(repo, folder_path)

    # Fetch each file
    results = []
    for file in files:
        if file['type'] == 'file':
            content = await self.get_file_content(repo, file['path'])
            results.append({
                'filename': file['name'],
                'path': file['path'],
                'content': content
            })

    return results
```

**Update tool registration:**
```python
tools.append(protos.Tool(
    function_declarations=[
        protos.FunctionDeclaration(
            name="github_fetch_folder_files",
            description="Fetch contents of ALL files in a folder (not just list)",
            parameters=protos.Schema(
                type=protos.Type.OBJECT,
                properties={
                    'repo': protos.Schema(type=protos.Type.STRING),
                    'folder_path': protos.Schema(type=protos.Type.STRING),
                }
            )
        )
    ]
))
```

---

## Testing Plan

### Test Case 1: Multi-File Code Review
```
Input: "review the codes in the upscale folder"

Expected Behavior:
1. GitHub agent fetches ALL files in folder
2. Code reviewer agent analyzes EACH file
3. Comprehensive review presented

Success Criteria:
- All files fetched (no hallucination)
- Code reviewer agent invoked
- Detailed review for each file
```

### Test Case 2: PR Review
```
Input: "review PR #123 in my repo"

Expected:
1. GitHub agent fetches PR diff
2. Code reviewer analyzes changes
3. Review with suggestions presented
```

### Test Case 3: Simple Request (No Decomposition)
```
Input: "list my Jira issues"

Expected:
- Single Jira agent call
- No task decomposition overhead
```

---

## Priority

**P0 (Critical):**
- Fix 4: Add multi-step prompting to system prompt
- Fix 5: Prevent hallucination by fetching all files

**P1 (High):**
- Fix 1: Enable task decomposer for complex requests
- Fix 2: Implement execution pipeline

**P2 (Medium):**
- Fix 3: Improve agent descriptions

---

## Summary

The current system has three major issues:

1. **Task decomposer exists but is unused** → multi-step operations not coordinated
2. **LLM hallucinates missing data** → reliability issues
3. **Specialized agents not invoked** → incorrect routing

Quick fix: Add explicit multi-step rules to system prompt.
Long-term fix: Implement task decomposition and execution pipeline.
