# Entity Relationship Graph & Multi-Turn Context - Implementation Plan

## Executive Summary

Building a "workspace memory" system that:
1. **Remembers relationships** between Jira issues, GitHub PRs, and Slack threads
2. **Enables natural follow-ups** like "Create a PR for that bug" without repeating context
3. **Links work across platforms** automatically

## Current State Analysis

### ✅ What We Have
- `ConversationContextManager` - Tracks entities and turns
- Reference resolution ("it", "that") - Works within session
- Entity tracking - Mentions and recency
- Focused entities - Last 10 referenced entities

### ❌ What's Missing
- **No persistent relationships** - Can't link "Jira issue KAN-42 → PR #123"
- **No cross-platform memory** - If you mention an issue in Jira, then say "create PR", it doesn't know which issue
- **No relationship queries** - Can't ask "Which PR fixes this bug?"
- **Session-only memory** - Everything lost on restart

## Proposed Architecture

### Component 1: Entity Relationship Graph (SQLite)

**Schema Design** (Simple to start, extensible later):

```sql
-- Core entities table
CREATE TABLE entities (
    id TEXT PRIMARY KEY,              -- e.g., "jira:KAN-42", "github:pr:123"
    platform TEXT NOT NULL,           -- jira, github, slack, notion
    entity_type TEXT NOT NULL,        -- issue, pr, thread, page
    entity_value TEXT NOT NULL,       -- KAN-42, #123, C12345
    title TEXT,                       -- Human-readable title
    url TEXT,                         -- Direct link to entity
    status TEXT,                      -- open, closed, in_progress
    metadata JSON,                    -- Platform-specific data
    created_at TIMESTAMP,
    last_accessed TIMESTAMP,
    access_count INTEGER DEFAULT 1
);

-- Relationships between entities
CREATE TABLE relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,          -- Entity that originates relationship
    target_id TEXT NOT NULL,          -- Entity being related to
    relationship_type TEXT NOT NULL,  -- fixes, implements, discusses, blocks
    created_by TEXT,                  -- user or agent that created link
    created_at TIMESTAMP,
    metadata JSON,                    -- Additional context
    FOREIGN KEY (source_id) REFERENCES entities(id),
    FOREIGN KEY (target_id) REFERENCES entities(id)
);

-- Conversation context (for multi-turn)
CREATE TABLE conversation_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    role TEXT NOT NULL,               -- user, assistant
    message TEXT NOT NULL,
    entities_mentioned JSON,          -- List of entity IDs
    relationships_created JSON,       -- Relationships established
    timestamp TIMESTAMP
);

-- Indexes for fast queries
CREATE INDEX idx_entities_platform ON entities(platform);
CREATE INDEX idx_entities_type ON entities(entity_type);
CREATE INDEX idx_relationships_source ON relationships(source_id);
CREATE INDEX idx_relationships_target ON relationships(target_id);
CREATE INDEX idx_conversations_session ON conversation_turns(session_id);
```

**Relationship Types**:
- `fixes` - PR fixes Jira issue
- `implements` - PR implements feature from issue
- `discusses` - Slack thread discusses Jira issue/PR
- `blocks` - Issue blocks another issue
- `duplicates` - Duplicate issues
- `references` - General mention
- `documents` - Notion page documents feature/issue

### Component 2: Enhanced Multi-Turn Context

**Memory Layers**:

```
Layer 1: Working Memory (In-Session)
└─ ConversationContextManager (existing)
   └─ Current turn, last 10 entities, focused context

Layer 2: Short-Term Memory (SQLite)
└─ EntityRelationshipGraph
   └─ Last 100 turns, entity relationships, links

Layer 3: Long-Term Memory (Future: Vector DB)
└─ Semantic search over all conversations
   └─ "What bugs did we fix last week?"
```

### Component 3: Natural Language Reference Resolution

**Enhanced resolution chain**:

```python
User: "Create a PR for that bug"

Step 1: Parse "that bug"
  → Reference detected

Step 2: Check working memory
  → Last mentioned issue: KAN-42

Step 3: Query relationship graph
  → KAN-42 is a Jira issue, type=bug, status=open

Step 4: Resolve full context
  → "Create PR to fix Jira issue KAN-42: 'Login fails on Safari'"

Step 5: Execute with full context
  → GitHub agent creates PR with:
    - Title: "Fix: Login fails on Safari (KAN-42)"
    - Body: "Fixes KAN-42"
    - Auto-link created: PR#123 → fixes → KAN-42
```

## Implementation Plan

### Phase 1: Entity Graph Foundation (Week 1)

**Files to Create**:
```
intelligence/
└── memory/
    ├── __init__.py
    ├── entity_graph.py          # Main SQLite interface
    ├── schema.sql               # Database schema
    └── relationship_types.py    # Relationship type definitions
```

**Core Classes**:

```python
class EntityRelationshipGraph:
    """SQLite-based entity relationship graph"""

    def add_entity(self, entity_id, platform, entity_type, title, url, metadata)
    def add_relationship(self, source_id, target_id, rel_type, metadata)
    def get_entity(self, entity_id)
    def find_related(self, entity_id, rel_type=None) → List[Entity]
    def query_path(self, from_id, to_id) → List[Relationship]

class RelationshipType(Enum):
    FIXES = "fixes"
    IMPLEMENTS = "implements"
    DISCUSSES = "discusses"
    BLOCKS = "blocks"
    DUPLICATES = "duplicates"
    REFERENCES = "references"
```

### Phase 2: Multi-Turn Context Enhancement (Week 1-2)

**Enhance ConversationContextManager**:

```python
class EnhancedConversationContext(ConversationContextManager):
    """Adds entity graph integration"""

    def __init__(self, session_id, entity_graph):
        super().__init__(session_id)
        self.entity_graph = entity_graph

    def resolve_reference_with_graph(self, phrase):
        """Uses both in-memory + SQLite graph"""

        # Try in-memory first (fast)
        if result := super().resolve_reference(phrase):
            return result

        # Query entity graph (relationships)
        if phrase in ["that bug", "the bug"]:
            # Find most recent bug-type issue
            return self.entity_graph.find_recent(
                entity_type="issue",
                filters={"type": "bug"}
            )

    def create_relationship_on_action(self, action, source_entity, target_entity):
        """Auto-link entities when actions occur"""

        # Example: When PR is created from issue
        if action == "create_pr" and source_entity.type == "issue":
            self.entity_graph.add_relationship(
                source_id=target_entity.id,  # PR
                target_id=source_entity.id,  # Issue
                rel_type="fixes"
            )
```

### Phase 3: Agent Integration (Week 2)

**Auto-Linking on Actions**:

```python
class JiraAgent:
    async def create_issue(self, title, description):
        # Create issue
        issue = await self.jira_api.create(title, description)

        # Register in entity graph
        self.entity_graph.add_entity(
            entity_id=f"jira:{issue.key}",
            platform="jira",
            entity_type="issue",
            title=issue.title,
            url=issue.url,
            metadata={"project": issue.project, "type": issue.type}
        )

        return issue

class GitHubAgent:
    async def create_pr(self, title, body, issue_ref=None):
        # Create PR
        pr = await self.github_api.create_pr(title, body)

        # Register in entity graph
        pr_id = f"github:pr:{pr.number}"
        self.entity_graph.add_entity(
            entity_id=pr_id,
            platform="github",
            entity_type="pr",
            title=pr.title,
            url=pr.url,
            metadata={"repo": pr.repo, "branch": pr.branch}
        )

        # Auto-link if issue reference found
        if issue_ref:
            self.entity_graph.add_relationship(
                source_id=pr_id,
                target_id=issue_ref,
                rel_type="fixes"
            )

        return pr
```

### Phase 4: Natural Multi-Turn Workflows (Week 2-3)

**Example Workflows**:

**Workflow 1: Bug → PR → Deployment**
```
User: "Show me bug KAN-42"
  → Graph: Store entity jira:KAN-42

User: "Create a PR to fix it"
  → Resolve: "it" = KAN-42
  → Create: PR #123
  → Link: PR#123 fixes KAN-42

User: "Review the PR"
  → Resolve: "the PR" = #123 (most recent)
  → Action: Code review on PR#123

User: "Merge it and deploy"
  → Resolve: "it" = PR#123
  → Action: Merge + deploy
  → Update: KAN-42 status = resolved
```

**Workflow 2: Multi-Platform Discussion**
```
User: "Create issue: Login broken in Safari"
  → Create: KAN-45
  → Graph: jira:KAN-45

User: "Discuss this in #engineering"
  → Resolve: "this" = KAN-45
  → Create: Slack thread in #engineering
  → Link: thread:C123 discusses KAN-45

User: "Document the fix in Notion"
  → Resolve: "the fix" = KAN-45
  → Create: Notion page
  → Link: page:P456 documents KAN-45

User: "Show me everything related to that login bug"
  → Query graph: KAN-45
  → Result:
    - Jira issue: KAN-45
    - PR #123 (fixes)
    - Slack thread C123 (discusses)
    - Notion page P456 (documents)
```

## Performance Considerations

### Database Size
- **Entities**: ~1000 per month → 12K/year
- **Relationships**: ~2000 per month → 24K/year
- **Conversations**: ~5000 turns/month → 60K/year
- **Total DB size**: ~50MB/year (easily manageable)

### Query Performance
- Indexed lookups: <1ms
- Relationship traversal (1 hop): <5ms
- Path finding (2-3 hops): <20ms

### Memory Usage
- SQLite in-memory cache: ~10MB
- Working memory: ~5MB
- Total: ~15MB overhead

## ROI Analysis

### Before (Current State)
```
User: "Show Jira ticket KAN-42"
Assistant: [Shows ticket]

User: "Create a PR for it"
Assistant: "Which issue would you like to create a PR for?" ❌
User: "KAN-42" (has to repeat!)
```

### After (With Entity Graph)
```
User: "Show Jira ticket KAN-42"
Assistant: [Shows ticket + stores in graph]

User: "Create a PR for it"
Assistant: [Creates PR, auto-links to KAN-42] ✅

User: "What's the status?"
Assistant: "KAN-42 is open. PR #123 is in review." ✅
```

**User Experience Improvements**:
- ✅ No repetition needed
- ✅ Natural conversation flow
- ✅ Cross-platform awareness
- ✅ Relationship tracking
- ✅ Context preservation

**Productivity Gains**:
- 50% reduction in clarification questions
- 30% faster workflows
- Better context for agents

## Technical Risks & Mitigation

### Risk 1: Graph Gets Out of Sync
**Mitigation**:
- Periodic sync jobs
- Webhook updates from platforms
- Cleanup stale entities (30 days inactive)

### Risk 2: Performance Degrades
**Mitigation**:
- Aggressive indexing
- Query result caching
- Pagination for large result sets

### Risk 3: Wrong Relationships Created
**Mitigation**:
- Confidence scoring on auto-links
- User confirmation for ambiguous cases
- Relationship editing/deletion

## Migration Path

### Phase 1: Foundation (This Week)
- ✅ Create entity_graph.py
- ✅ SQLite schema
- ✅ Basic CRUD operations
- ✅ Unit tests

### Phase 2: Integration (Next Week)
- ✅ Enhance ConversationContextManager
- ✅ Agent auto-linking
- ✅ Reference resolution with graph

### Phase 3: Polish (Week 3)
- ✅ Query optimization
- ✅ Relationship confidence
- ✅ UI for viewing graph

### Phase 4: Advanced (Future)
- Vector database for semantic search
- Graph visualization
- Predictive linking
- Auto-cleanup and archiving

## Success Metrics

**Week 1**:
- Entity graph stores 100+ entities
- 50+ relationships created
- Basic queries working (<10ms)

**Month 1**:
- 80% of follow-up questions resolved without clarification
- Users reference "it"/"that" successfully 70% of time
- Average conversation length increases (more multi-turn)

**Quarter 1**:
- 1000+ entities tracked
- Cross-platform workflows common
- Users report "feels like it remembers"

## Next Steps

1. **Get approval** on architecture
2. **Create intelligence/memory/** directory
3. **Implement EntityRelationshipGraph**
4. **Enhance ConversationContextManager**
5. **Add agent auto-linking**
6. **Test multi-turn workflows**

---

## Summary

This implementation transforms the agent system from **stateless & transactional** to **stateful & conversational**.

**Core Innovation**: SQLite-based entity graph that links work across Jira, GitHub, Slack, and Notion.

**User Impact**: "Create a PR for that bug" just works, without repeating context.

**Technical Complexity**: Medium - SQLite is simple, schema is straightforward, integration points are clear.

**Timeline**: 2-3 weeks for full implementation, 1 week for MVP.

**ROI**: Massive - transforms UX from frustrating to delightful.
