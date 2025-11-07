# Confirmation Policy - Read vs Write Operations

## Philosophy

**Only confirm operations that CHANGE something.**

Reading data is safe and shouldn't interrupt the user. Writing, modifying, or deleting requires human approval.

## Confirmation Rules by Agent

### ✅ Slack

**REQUIRES Confirmation** (Write Operations):
- ✉️ `send` message
- 📤 `post` message
- 💬 `notify` users
- 📣 `announce` to channel
- ↩️ `reply` in thread
- 👍 `react` to message
- 🗑️ `delete` message

**NO Confirmation** (Read Operations):
- 📋 `list` channels
- 🔍 `search` messages
- 👥 `get` users
- 📖 `view` messages
- 👀 `show` conversations
- 🔎 `find` content

### ✅ Jira

**REQUIRES Confirmation** (Write Operations):
- ➕ `create` issue
- ✏️ `update` issue
- 🗑️ `delete` issue
- 🔄 `transition` status
- 👤 `assign` task
- 💬 `add comment`
- ❌ `close` issue
- 📝 `edit` description

**NO Confirmation** (Read Operations):
- 🔍 `search` issues
- 📋 `list` issues
- 🔎 `find` tasks
- 👀 `get` issue details
- 📊 `view` board
- 📈 `show` sprint
- 👤 `assigned to me`
- 📝 `my tasks`

### ✅ Notion

**REQUIRES Confirmation** (Write Operations):
- ➕ `create` page
- 📝 `add` content
- ✏️ `update` page
- 🖊️ `write` to database
- 📥 `insert` block
- 🗑️ `delete` page
- 📝 `edit` content

**NO Confirmation** (Read Operations):
- 📋 `list` pages
- 🔍 `search` content
- 🔎 `find` database entries
- 👀 `get` page
- 📖 `view` workspace
- 📊 `show` database
- 📚 `read` content

### ✅ GitHub

**REQUIRES Confirmation** (Write Operations):
- ➕ `create` PR
- ➕ `create` issue
- 🔀 `merge` PR
- ❌ `close` PR/issue
- 💬 `comment` on PR
- ✏️ `edit` issue
- 🏷️ `add` label

**NO Confirmation** (Read Operations):
- 📋 `list` PRs
- 🔍 `search` code
- 👀 `view` repository
- 📊 `show` status
- 🔎 `find` issues
- 📖 `get` file content

## Implementation

The confirmation system uses **keyword detection** to distinguish read vs write:

```python
# Example: Jira
write_keywords = ['create', 'update', 'delete', 'transition', 'assign']
read_keywords = ['get', 'search', 'list', 'find', 'show', 'view']

# If read keyword found → NO confirmation
if any(read_kw in instruction.lower() for read_kw in read_keywords):
    return False

# If write keyword found → REQUIRE confirmation
if any(write_kw in instruction.lower() for write_kw in write_keywords):
    return True
```

## User Experience

### Read Operation (Fast & Smooth)
```
You: Get my Jira tasks
Assistant: [immediately executes, no confirmation]

Here are your tasks:
1. KAN-123: Fix login bug
2. KAN-124: Update documentation
```

No interruption, instant results! ✨

### Write Operation (Confirmed & Safe)
```
You: Create a Jira issue about the bug
Assistant: [shows confirmation]

══════════════════════════════════════════
⚠️ JIRA OPERATION REQUIRES CONFIRMATION
══════════════════════════════════════════

Operation: Create Issue
Project: KAN
Summary: Fix login bug

Approve this operation? [y/n]: y
✅ Approved!

Issue KAN-125 created successfully.
```

Safe, reviewed, confirmed! 🛡️

## Benefits

### For Users
✅ **Fast reads** - No interruptions for viewing data
✅ **Safe writes** - Always review before changes
✅ **Clear distinction** - Know what's safe vs risky
✅ **Better UX** - Smooth flow for common tasks

### For Safety
✅ **Prevent accidents** - Can't accidentally delete/send
✅ **Review content** - Check message before sending
✅ **Catch mistakes** - Wrong channel? Wrong assignee? Fix before sending
✅ **Audit trail** - Know every write was approved

## Configuration

You can toggle confirmations in `.env`:

```bash
# Slack confirmations (default: true)
CONFIRM_SLACK_MESSAGES=true

# Jira confirmations (default: true)
CONFIRM_JIRA_OPERATIONS=true
```

**Note**: Even with confirmations enabled, READ operations never prompt.

## Examples

### ✅ No Confirmation Needed
```
- "List Slack channels"
- "Search Jira for bug issues"
- "Get my assigned tasks"
- "Show Notion pages"
- "Find messages in #dev-opps"
- "View GitHub PRs"
```

### ⚠️ Confirmation Required
```
- "Send message to #general"
- "Create Jira issue"
- "Close GitHub PR"
- "Delete Notion page"
- "Update Jira status"
- "Post announcement"
```

## Edge Cases

### Ambiguous Operations
If an instruction could be read OR write:
- System errs on the side of caution
- Shows confirmation if unclear
- User can clarify in the prompt

### Bulk Operations
Reading 100 items? ✅ No confirmation
Creating 10 issues? ⚠️ Batch confirmation

### Combined Operations
"Search Jira and create issue from results"
- First part (search): No confirmation
- Second part (create): Shows confirmation

## Summary

**Golden Rule**: If it changes data, confirm it. If it just reads, let it flow.

This creates the perfect balance:
- 🚀 Speed for common read operations
- 🛡️ Safety for all write operations
- 😊 Great user experience overall