# Quick Start Guide

## ⚡ Get Running in 3 Steps

### Step 1: Add Your Google API Key

I've created a `.env` file for you. **Edit it now:**

```bash
nano .env
# OR
vi .env
# OR use any text editor
```

**Replace this line:**
```bash
GOOGLE_API_KEY=your_gemini_api_key_here_PLEASE_REPLACE_THIS
```

**With your real key:**
```bash
GOOGLE_API_KEY=AIza...your_actual_key_here
```

**Get a key here:** https://makersuite.google.com/app/apikey

### Step 2: Run the System

```bash
python main.py
```

### Step 3: Watch the Detailed Output

You'll now see **exactly where each agent is** during loading:

```
[1/4] Loading code_reviewer agent module...
[2/4] code_reviewer: Module loaded
[3/4] code_reviewer: Creating instance...
[4/4] code_reviewer: Initializing (connecting to services)...
      → code_reviewer: Calling initialize()...
      [any output from the agent shows here]
      → code_reviewer: Getting capabilities...
✓ code_reviewer agent loaded (0.8s)
```

If an agent hangs, you'll see exactly which step it's stuck on!

---

## 🔍 What We Fixed

### Issue 1: Slack Agent Error (FALSE ALARM)
**You saw:** `✗ slack: npx/npm not installed`
**Reality:** npx IS installed (version 10.9.4 at `/opt/node22/bin/npx`)
**Real issue:** Missing `SLACK_BOT_TOKEN` and `SLACK_TEAM_ID` in .env file

**Fix:** The error message was misleading. If you want to use Slack agent, uncomment these lines in `.env`:
```bash
SLACK_BOT_TOKEN=xoxb-your-token
SLACK_TEAM_ID=T-your-team-id
```

### Issue 2: System Hanging (NOW VISIBLE)
**Before:** Silent hang after browser agent loads
**After:** You'll see exactly where it's stuck:
- Which agent is loading (probably code_reviewer)
- Which step it's on (initialize or get_capabilities)
- Any error output from the agent itself

---

## 📊 What You Should See Now

### If Code Reviewer Hangs:
```
[4/4] code_reviewer: Initializing (connecting to services)...
      → code_reviewer: Calling initialize()...
      [stuck here? Shows what it's trying to do]
```

**Possible causes:**
- GOOGLE_API_KEY is invalid or expired
- Network connectivity issues
- Rate limiting from Google API

### If All Agents Load Successfully:
```
============================================================
✓ Loaded 7 agent(s) successfully

System ready with 7 agent(s)
============================================================

[beautiful UI with prompt]
```

---

## 🚨 Common Issues

### "GOOGLE_API_KEY environment variable not set"
**Cause:** Forgot to edit .env file
**Fix:** Edit `.env` and add your real Google API key

### Slack/Jira/GitHub agents fail with "Missing required environment variables"
**Cause:** These agents need their own API keys
**Fix:** Either:
1. Add the required keys to `.env` (see `.env.example` for all options)
2. OR disable them: `DISABLED_AGENTS=slack,jira,github`

### Agent times out after 10 seconds
**Cause:** Agent can't connect (bad credentials, network issue, etc.)
**Fix:**
1. Check the detailed output to see what it was trying to do
2. Verify credentials are correct
3. Check network connectivity
4. OR disable the problematic agent

---

## 🎯 Next Steps

### 1. Run Now
```bash
python main.py
```

### 2. Share the Output
Copy and share the **full output** (especially the Stage 4 details) so I can see:
- Which agent is hanging
- At which exact step
- What errors/output it's producing

### 3. Add More Agent Credentials (Optional)
If you want to use other agents (Slack, Jira, GitHub, etc.), edit `.env` and add their keys:
```bash
# Slack
SLACK_BOT_TOKEN=xoxb-...
SLACK_TEAM_ID=T...

# GitHub
GITHUB_TOKEN=ghp_...

# etc.
```

---

## 📝 Summary of Changes

1. **Created `.env` file** with GOOGLE_API_KEY placeholder
2. **Disabled output suppression** so you can see what's happening
3. **Added detailed Stage 4 logging** showing:
   - When initialize() is called
   - Actual output from agents
   - When get_capabilities() is called

This is temporary for debugging. Once agents load successfully, we'll re-enable clean output.

---

## ⚙️ Debugging Commands

### Check your API key is set:
```bash
grep GOOGLE_API_KEY .env
```

### Check npx is really installed:
```bash
which npx
npx --version
```

### Test with just code_reviewer (fastest):
```bash
DISABLED_AGENTS=slack,browser,jira,github,notion,scraper python main.py
```

### Run with maximum verbosity:
```bash
python main.py --verbose
```

---

**Now run `python main.py` and let me know what you see!** 🚀
