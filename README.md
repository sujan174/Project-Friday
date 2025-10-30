# 🎭 Multi-Agent Orchestration System

**A production-ready, intelligent orchestration system for coordinating specialized AI agents across Jira, Slack, GitHub, and Notion.**

---

## 🌟 **Overview**

This system provides a powerful framework for building and orchestrating specialized AI agents that can seamlessly interact with multiple platforms. Each agent is an expert in its domain, and the orchestrator intelligently routes tasks to the appropriate agent based on natural language instructions.

### **Key Features**

- **🤖 Intelligent Orchestration**: Smart routing of tasks to specialized agents
- **🔄 Automatic Retry Logic**: Exponential backoff for transient failures
- **✅ Verification Workflows**: Mandatory verification of state-changing operations
- **📊 Operation Tracking**: Comprehensive statistics and reporting
- **🔍 Verbose Logging**: Detailed debugging information when needed
- **⚡ Production-Ready**: Comprehensive error handling and resilience
- **🎯 Type-Safe**: Full type hints throughout the codebase
- **📚 Well-Documented**: Extensive docstrings and inline comments

---

## 🏗️ **Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                      Orchestrator Agent                      │
│  (Gemini 2.5 Pro - Intelligent Task Routing & Coordination) │
└────────────┬───────────────────────────────────────┬────────┘
             │                                       │
    ┌────────▼────────┐                    ┌────────▼────────┐
    │   Jira Agent    │                    │   Slack Agent   │
    │ (Gemini Flash)  │                    │ (Gemini Flash)  │
    └────────┬────────┘                    └────────┬────────┘
             │                                       │
       ┌─────▼──────┐                         ┌─────▼──────┐
       │ MCP Server │                         │ MCP Server │
       │  (Docker)  │                         │   (NPX)    │
       └─────┬──────┘                         └─────┬──────┘
             │                                       │
        ┌────▼────┐                             ┌───▼────┐
        │  Jira   │                             │ Slack  │
        │   API   │                             │  API   │
        └─────────┘                             └────────┘

    ┌────────────────┐                    ┌─────────────────┐
    │  GitHub Agent  │                    │  Notion Agent   │
    │ (Gemini Flash) │                    │ (Gemini Flash)  │
    └────────┬───────┘                    └────────┬────────┘
             │                                      │
       ┌─────▼──────┐                        ┌─────▼──────┐
       │ MCP Server │                        │ MCP Server │
       │   (NPX)    │                        │   (NPX)    │
       └─────┬──────┘                        └─────┬──────┘
             │                                      │
        ┌────▼─────┐                          ┌────▼─────┐
        │  GitHub  │                          │  Notion  │
        │    API   │                          │    API   │
        └──────────┘                          └──────────┘
```

### **Component Descriptions**

#### **Orchestrator (orchestrator.py)**
- **Model**: Gemini 2.5 Pro (most capable for reasoning)
- **Role**: Understands user intent, breaks down complex tasks, routes to agents
- **Capabilities**: Multi-step planning, context management, result synthesis

#### **Specialized Agents** (connectors/)
- **Jira Agent**: Issue management, JQL search, workflow automation
- **Slack Agent**: Messaging, search, channel management
- **GitHub Agent**: Repository management, issues, PRs, code search
- **Notion Agent**: Knowledge management, databases, content creation

#### **Base Agent** (connectors/base_agent.py)
- Abstract interface enforcing consistency
- Common helper methods for error formatting
- Custom exception types for better error handling

#### **MCP (Model Context Protocol)**
- Standardized way for AI models to interact with external tools
- Each agent connects to its own MCP server
- MCP servers handle authentication and API communication

---

## 🚀 **Quick Start**

### **Prerequisites**

- **Python 3.10+**
- **Docker** (for Jira agent)
- **Node.js 18+** (for Slack, GitHub, Notion agents via npx)
- **API Credentials** for the services you want to use

### **Installation**

1. **Clone the repository**
```bash
cd /path/to/project
```

2. **Install Python dependencies**
```bash
pip install google-generativeai python-mcp python-dotenv
```

3. **Install Docker** (if not already installed)
```bash
# macOS
brew install docker

# Linux
sudo apt-get install docker.io
```

4. **Verify Node.js and npx**
```bash
node --version  # Should be 18+
npx --version
```

### **Configuration**

Create a `.env` file in the project root:

```bash
# Google AI (Required for all agents)
GOOGLE_API_KEY="your-google-ai-api-key"

# Jira Configuration
JIRA_URL="https://your-domain.atlassian.net"
JIRA_USERNAME="your-email@company.com"
JIRA_API_TOKEN="your-jira-api-token"

# Slack Configuration
SLACK_BOT_TOKEN="xoxb-your-slack-bot-token"
SLACK_TEAM_ID="T0123456789"

# GitHub Configuration
GITHUB_PERSONAL_ACCESS_TOKEN="ghp_your-github-token"

# Notion Configuration
# (OAuth handled via browser popup - no env vars needed)
```

#### **Getting API Credentials**

**Jira:**
1. Go to https://id.atlassian.com/manage-profile/security/api-tokens
2. Click "Create API token"
3. Copy the token and add to `.env`

**Slack:**
1. Go to https://api.slack.com/apps
2. Create/select your app
3. Get Bot Token from "OAuth & Permissions"
4. Get Team ID from "Basic Information"

**GitHub:**
1. Go to https://github.com/settings/tokens
2. Generate new token (classic)
3. Select scopes: `repo`, `read:org`, `user`

### **Running the System**

**Basic Mode:**
```bash
python orchestrator.py
```

**Verbose Mode** (shows detailed agent operations):
```bash
python orchestrator.py --verbose
```

**Example Session:**
```
🎭 Multi-Agent Orchestration System
Mode: Clean
============================================================

You: Create a Jira ticket for the deployment bug and notify #engineering on Slack

   ⠋ Discovering agents...
   ✓ Loaded 4 agent(s) successfully.

   ⠋ Running jira agent... (5.2s)
   ⠋ Running slack agent... (2.1s)

🎭 Orchestrator:
I've created Jira issue PROJ-123: "Deployment Bug" with priority High.
I've also sent a notification to #engineering on Slack informing the team.

Issue URL: https://your-domain.atlassian.net/browse/PROJ-123
Slack Message: Posted in #engineering at 2:45 PM

You: exit
```

---

## 📂 **Project Structure**

```
Lazy devs backone RnD/
├── orchestrator.py          # Main orchestrator agent
├── connectors/              # Specialized agents
│   ├── base_agent.py       # Abstract base class
│   ├── jira_agent.py       # Jira integration (PRODUCTION-READY ✅)
│   ├── slack_agent.py      # Slack integration (PRODUCTION-READY ✅)
│   ├── github_agent.py     # GitHub integration
│   └── notion_agent.py     # Notion integration
├── .env                     # Environment variables
├── .gitignore              # Git ignore patterns
└── README.md               # This file
```

---

## 🛠️ **Advanced Features**

### **1. Verification Protocol** (Jira Agent)

The Jira agent includes a **mandatory verification protocol** that ensures state-changing operations are successful:

```python
# User: "Mark KAN-1, KAN-2, KAN-3 as Done"

# Agent Process:
# 1. Transitions each issue to Done
# 2. IMMEDIATELY verifies by searching for the issues
# 3. Checks actual status vs expected status
# 4. Reports discrepancies: "KAN-1, KAN-2 verified Done. KAN-3 still In Progress. Retrying..."
# 5. Retries failed operations
# 6. Final verification and report
```

This prevents the common problem of agents claiming success without verification.

### **2. Retry Logic with Exponential Backoff**

All production-ready agents include automatic retry for transient failures:

```python
RetryConfig:
- MAX_RETRIES: 3
- INITIAL_DELAY: 1.0s
- MAX_DELAY: 10.0s
- BACKOFF_FACTOR: 2.0

Retry Sequence: 1s → 2s → 4s → 8s (capped at 10s)
```

**Retryable Errors:**
- Network timeouts
- Connection errors
- Rate limiting (429, 503, 502, 504)
- Temporary service unavailability

### **3. Operation Statistics**

Track performance and reliability:

```python
agent.get_stats()
# Output: "Operations: 25 total, 23 successful, 2 failed (92.0% success rate), 5 retries"
```

### **4. Verbose Logging**

Enable detailed logging for debugging:

```bash
python orchestrator.py --verbose
```

Output shows:
- Agent initialization steps
- Tool calls with arguments
- Results from each operation
- Retry attempts
- Error details

---

## 🎯 **Production-Ready Status**

### **Fully Production-Ready ✅**

| Component | Status | Features |
|-----------|--------|----------|
| **base_agent.py** | ✅ Complete | Comprehensive docs, helper methods, custom exceptions |
| **jira_agent.py** | ✅ Complete | Retry logic, verification, stats, error handling, verbose logging |
| **slack_agent.py** | ✅ Complete | Retry logic, stats, error handling, verbose logging |

### **Ready with Minor Improvements Needed 🟡**

| Component | Status | Notes |
|-----------|--------|-------|
| **github_agent.py** | 🟡 Good | Apply same pattern as slack_agent.py (retry, stats, error classification) |
| **notion_agent.py** | 🟡 Good | Apply same pattern as slack_agent.py (retry, stats, error classification) |
| **orchestrator.py** | 🟡 Good | Currently functional; could benefit from better organization |

---

## 🔧 **Configuration Options**

### **Retry Configuration**

Edit `RetryConfig` class in agent files:

```python
class RetryConfig:
    MAX_RETRIES = 3              # Number of retry attempts
    INITIAL_DELAY = 1.0          # Initial delay in seconds
    MAX_DELAY = 10.0             # Maximum delay in seconds
    BACKOFF_FACTOR = 2.0         # Exponential backoff multiplier
```

### **Agent Behavior**

Each agent has a comprehensive **system prompt** that defines its behavior. You can customize these prompts in the agent files to adjust:
- Tone and style
- Error handling approach
- Output format
- Verification requirements

### **Orchestrator Settings**

In `orchestrator.py`:

```python
max_iterations = 15  # Maximum tool calls per request
model = 'gemini-2.5-pro'  # Orchestrator model
agent_model = 'gemini-2.5-flash'  # Agent model (faster, cheaper)
```

---

## 📊 **Performance Considerations**

### **Model Selection**

- **Orchestrator**: Gemini 2.5 Pro
  - More capable reasoning for complex task decomposition
  - Higher cost but better for coordination

- **Agents**: Gemini 2.5 Flash
  - Faster response times
  - Lower cost for straightforward tool execution
  - Sufficient for specialized tasks

### **Cost Optimization**

1. **Use verbose mode only when debugging** (extra logging = extra token usage)
2. **Agents use cheaper Flash model** for most operations
3. **Retry logic prevents unnecessary failed operations**
4. **Efficient function calling** reduces redundant API calls

### **Latency**

- **Typical task**: 2-8 seconds
- **Complex multi-agent tasks**: 10-30 seconds
- **Network-dependent**: Jira Docker startup adds 1-2s on first call

---

## 🐛 **Troubleshooting**

### **"Agent not initialized" errors**

**Cause**: Environment variables missing or incorrect

**Solution**:
1. Check `.env` file exists and has correct values
2. Verify API tokens are valid and not expired
3. Run with `--verbose` to see initialization details

### **Docker-related errors (Jira)**

**Symptoms**: "Docker not found" or "Cannot connect to Docker daemon"

**Solution**:
```bash
# Check Docker is running
docker ps

# Pull the image manually
docker pull ghcr.io/sooperset/mcp-atlassian:latest

# Test Docker connection
docker run hello-world
```

### **Rate limiting errors**

**Symptoms**: "Rate limit exceeded" or "429 Too Many Requests"

**Solution**:
- The system automatically retries with exponential backoff
- Space out requests if sending many messages/updates
- Check your API rate limits for each service

### **"No agents available"**

**Cause**: All agents failed to initialize

**Solution**:
1. Run with `--verbose` to see which agents are failing
2. Check environment variables for failed agents
3. Verify API credentials are correct
4. Check network connectivity

---

## 🧪 **Testing**

### **Test Individual Agents**

```bash
# Test with verbose logging
python orchestrator.py --verbose

# Then try simple commands:
You: List all my open Jira issues
You: Send "Test message" to #general on Slack
You: Search GitHub for "authentication" in myorg/myrepo
```

### **Test Error Handling**

```python
# Intentionally cause errors to test recovery:
You: Update Jira issue FAKE-999 to Done  # Should handle gracefully
You: Send message to #nonexistent-channel  # Should report error clearly
```

### **Test Verification (Jira)**

```python
# Test the verification protocol:
You: Mark issues KAN-1, KAN-2, KAN-3 as Done

# Agent should:
# 1. Attempt to transition all
# 2. Verify each one
# 3. Report which succeeded/failed
# 4. Retry failures
# 5. Final verification
```

---

## 🔐 **Security Best Practices**

1. **Never commit `.env` file** to version control (it's in `.gitignore`)
2. **Use environment variables** for all credentials
3. **Rotate API tokens regularly**
4. **Use least-privilege access** for bot accounts
5. **Review agent actions** before granting write permissions
6. **Monitor usage** through operation statistics

---

## 📈 **Extending the System**

### **Adding a New Agent**

1. **Create agent file**:
```python
# connectors/newservice_agent.py
from connectors.base_agent import BaseAgent

class Agent(BaseAgent):
    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose
        # ... (follow jira_agent.py pattern)

    async def initialize(self):
        # Connect to service
        self.initialized = True

    async def get_capabilities(self) -> List[str]:
        return ["Capability 1", "Capability 2"]

    async def execute(self, instruction: str) -> str:
        # Execute task
        return "Result"
```

2. **Add environment variables** to `.env`

3. **Restart orchestrator** - agent is auto-discovered!

### **Customizing Agent Behavior**

Edit the **system prompt** in the agent's `_build_system_prompt()` method:

```python
def _build_system_prompt(self) -> str:
    return """You are a specialized agent for...

    # Add your custom instructions here
    # Define behavior, output format, error handling, etc.
    """
```

---

## 📚 **Code Quality**

### **Standards**

- ✅ **Type Hints**: All functions have type annotations
- ✅ **Docstrings**: Comprehensive documentation for all classes/methods
- ✅ **Error Handling**: Try-catch blocks with specific error types
- ✅ **Code Organization**: Logical sections with visual separators
- ✅ **DRY Principle**: No code duplication
- ✅ **Single Responsibility**: Each method has one clear purpose

### **Naming Conventions**

- **Classes**: `PascalCase` (e.g., `BaseAgent`, `RetryConfig`)
- **Functions/Methods**: `snake_case` (e.g., `execute_task`, `get_capabilities`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_RETRIES`, `INITIAL_DELAY`)
- **Private Methods**: `_leading_underscore` (e.g., `_format_error`, `_classify_error`)

---

## 🤝 **Contributing**

When contributing improvements:

1. **Follow the established patterns** (see `jira_agent.py` for reference)
2. **Add comprehensive docstrings** for all new functions/classes
3. **Include error handling** with specific error types
4. **Add retry logic** for network operations
5. **Update README** with new features/changes
6. **Test thoroughly** with both success and error cases

---

## 📝 **License**

This is an internal project. Add appropriate license information as needed.

---

## 🙏 **Acknowledgments**

- **Model Context Protocol (MCP)**: Anthropic's standard for AI-tool integration
- **Google Gemini**: Powering the AI reasoning and execution
- **MCP Servers**: @modelcontextprotocol for Slack/GitHub, sooperset for Jira/Atlassian

---

## 📞 **Support**

For issues or questions:
1. Check this README first
2. Run with `--verbose` flag to debug
3. Review error messages carefully (they include troubleshooting steps)
4. Check API service status pages if errors persist

---

**Built with ❤️ for seamless multi-platform automation**
