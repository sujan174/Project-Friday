# Troubleshooting Guide

## Agent Loading Issues

### Understanding the Loading Process

When you run the system, agents load in 4 stages:

```
[1/4] Loading {agent} agent module...     <- Python module loading
[2/4] {agent}: Module loaded              <- Import successful
[3/4] {agent}: Creating instance...       <- Instantiating agent class
[4/4] {agent}: Initializing...            <- Connecting to external services (MOST LIKELY TO HANG)
✓ {agent} agent loaded (X.Xs)             <- Success!
```

### Common Issues and Solutions

#### Issue: Agent Hangs at [4/4] Initializing

**Symptoms:**
```
[4/4] slack: Initializing (connecting to services)...
[stuck here for 10+ seconds]
✗ slack: Timed out after 10s
```

**Why it happens:**
- Stage 4 is where MCP agents spawn subprocess servers using `npx`
- They try to connect to external services (Slack API, Jira, etc.)
- Missing credentials or network issues cause hanging

**Solutions:**

1. **Check .env file:**
   ```bash
   # Make sure you have the required keys
   cat .env | grep SLACK_BOT_TOKEN
   cat .env | grep GOOGLE_API_KEY
   ```

2. **Verify credentials are correct:**
   - Slack: `SLACK_BOT_TOKEN` and `SLACK_TEAM_ID`
   - Jira: `JIRA_URL` and `JIRA_API_TOKEN`
   - GitHub: `GITHUB_TOKEN`
   - Notion: `NOTION_TOKEN`

3. **Check network connectivity:**
   ```bash
   curl -I https://api.slack.com
   curl -I https://api.github.com
   ```

4. **Disable the problematic agent:**
   ```bash
   # In .env file
   DISABLED_AGENTS=slack,jira
   ```

#### Issue: npx/npm not installed

**Symptoms:**
```
✗ slack: npx/npm not installed
  Hint: Install Node.js from https://nodejs.org/
```

**Solution:**
1. Install Node.js (includes npm and npx): https://nodejs.org/
2. Verify installation:
   ```bash
   node --version
   npm --version
   npx --version
   ```

#### Issue: Missing Environment Variables

**Symptoms:**
```
✗ slack: Missing required environment variables
  Hint: Check .env file for required keys
```

**Solution:**
1. Create .env file if it doesn't exist:
   ```bash
   cp .env.example .env
   ```

2. Add required keys to .env:
   ```bash
   # Required for all agents
   GOOGLE_API_KEY=your_key_here

   # Required for specific agents
   SLACK_BOT_TOKEN=xoxb-...
   SLACK_TEAM_ID=T...
   ```

3. Restart the system

#### Issue: Global Timeout Reached

**Symptoms:**
```
✗ Global timeout reached (60s), stopping agent loading
```

**Why it happens:**
- Multiple agents are hanging
- Total loading time exceeded 60 seconds
- System gives up to prevent infinite hanging

**Solutions:**

1. **Disable slow/failing agents:**
   ```bash
   # In .env file
   DISABLED_AGENTS=slack,jira,github
   ```

2. **Check which agents are slow:**
   ```bash
   python main.py --verbose 2>&1 | grep "agent loaded"
   # Look for agents taking >3 seconds
   ```

3. **Fix underlying issues first:**
   - Add missing credentials
   - Fix network connectivity
   - Install missing dependencies

### Verbose Mode

For detailed debugging information:

```bash
python main.py --verbose
```

**What you'll see:**
- Full error stack traces
- Detailed initialization steps
- Agent capability lists
- Timing information for each stage
- Complete error messages

### Quick Diagnostic Commands

**1. Check environment setup:**
```bash
# Check if .env exists
ls -la .env

# Check required keys are present
grep "GOOGLE_API_KEY" .env
grep "SLACK_BOT_TOKEN" .env
```

**2. Check dependencies:**
```bash
# Check Python packages
pip list | grep google-generativeai
pip list | grep mcp

# Check Node.js
node --version
npx --version
```

**3. Test network connectivity:**
```bash
# Test API endpoints
curl -I https://generativelanguage.googleapis.com
curl -I https://api.slack.com
curl -I https://api.github.com
```

**4. Check for common errors:**
```bash
# Run with verbose mode and save log
python main.py --verbose 2>&1 | tee startup.log

# Search for common error patterns
grep "Missing required" startup.log
grep "Timed out" startup.log
grep "Failed to initialize" startup.log
```

### Performance Notes

**Normal Loading Times:**
- code_reviewer: ~0.5-1s (no external dependencies)
- MCP agents (slack, jira, github): ~2-5s each (spawns subprocess)
- Browser/Scraper agents: ~3-7s (heavy MCP servers)

**If loading takes longer:**
- First time: npm packages download (can be slow)
- Network issues: Connection timeouts
- Missing credentials: Agents retry connections
- System load: Other processes competing for resources

### Still Having Issues?

1. **Read the full error messages:**
   ```bash
   python main.py --verbose 2>&1 | less
   ```

2. **Check the log files:**
   ```bash
   ls -la logs/
   tail -f logs/session_*.log
   ```

3. **Try loading agents individually:**
   ```bash
   # Disable all except one
   DISABLED_AGENTS=slack,jira,github,notion,browser,scraper python main.py
   # Only code_reviewer will load

   # Then enable one at a time
   DISABLED_AGENTS=jira,github,notion,browser,scraper python main.py
   # slack + code_reviewer will load
   ```

4. **Check system resources:**
   ```bash
   # Check if system is overloaded
   top
   # Look for high CPU/memory usage

   # Check disk space
   df -h
   ```

5. **Review AGENTS_SETUP.md:**
   - Complete setup instructions
   - Agent-specific requirements
   - Configuration examples

### Common Environment Variables

```bash
# Core configuration
GOOGLE_API_KEY=your_gemini_api_key

# Agent-specific
SLACK_BOT_TOKEN=xoxb-...
SLACK_TEAM_ID=T...
JIRA_URL=https://your-domain.atlassian.net
JIRA_API_TOKEN=your_token
GITHUB_TOKEN=ghp_...
NOTION_TOKEN=secret_...

# Optional controls
DISABLED_AGENTS=agent1,agent2,agent3
LOG_LEVEL=INFO
ENABLE_FILE_LOGGING=true
```

### Debug Checklist

- [ ] .env file exists and has all required keys
- [ ] GOOGLE_API_KEY is set (required for all operations)
- [ ] Node.js/npm/npx are installed
- [ ] Network connectivity works
- [ ] No typos in environment variable names
- [ ] API tokens are valid (not expired)
- [ ] Firewall allows outbound connections
- [ ] System has enough resources (CPU, RAM, disk)
- [ ] Run with --verbose to see detailed errors
- [ ] Check logs in logs/ directory

### Getting Help

When reporting issues, include:
1. Full output from `python main.py --verbose`
2. Contents of .env file (REDACT sensitive tokens!)
3. System info: OS, Python version, Node version
4. What stage each agent fails at [1/4, 2/4, 3/4, or 4/4]
5. Any error messages or stack traces
