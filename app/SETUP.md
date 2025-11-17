# Aerius Desktop - Setup Guide

Complete setup instructions for getting Aerius Desktop running on your machine.

## System Requirements

- **Operating System**: macOS 10.13+, Windows 10+, or Linux (Ubuntu 18.04+)
- **Node.js**: v18.0.0 or higher
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Disk Space**: 500MB for app + dependencies

## Step-by-Step Setup

### 1. Verify Prerequisites

```bash
# Check Node.js version
node --version  # Should be v18.0.0 or higher

# Check Python version
python3 --version  # Should be 3.8 or higher

# Check npm version
npm --version
```

If any are missing:
- **Node.js**: Download from [nodejs.org](https://nodejs.org/)
- **Python**: Download from [python.org](https://www.python.org/)

### 2. Clone or Verify Project Structure

Ensure your directory structure looks like this:

```
/home/user/
├── Project-Aerius/          # Main orchestrator (must exist!)
│   ├── orchestrator.py
│   ├── config.py
│   ├── requirements.txt
│   └── .env
└── Aerius-Desktop/          # Desktop app (this project)
    ├── package.json
    ├── electron/
    ├── backend/
    └── src/
```

### 3. Install Project Aerius Dependencies

First, set up the backend orchestrator:

```bash
# Navigate to Project-Aerius
cd /home/user/Project-Aerius

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### 4. Configure API Keys

Copy the example environment file:

```bash
cd /home/user/Project-Aerius
cp .env.example .env
```

Edit `.env` and add your API keys:

```bash
# Required - Get from https://makersuite.google.com/app/apikey
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional - Add only the services you plan to use
SLACK_BOT_TOKEN=xoxb-your-slack-token
SLACK_TEAM_ID=your-slack-team-id

GITHUB_PERSONAL_ACCESS_TOKEN=ghp_your_github_token

JIRA_URL=https://your-domain.atlassian.net
JIRA_USERNAME=your-email@example.com
JIRA_API_TOKEN=your_jira_token

NOTION_TOKEN=secret_your_notion_token
```

**Getting API Keys:**

- **Gemini (Required)**: https://makersuite.google.com/app/apikey
- **Slack**: https://api.slack.com/apps → Your App → OAuth & Permissions
- **GitHub**: https://github.com/settings/tokens → Generate new token (classic)
- **Jira**: https://id.atlassian.com/manage-profile/security/api-tokens
- **Notion**: https://www.notion.so/my-integrations

### 5. Install Desktop App Dependencies

```bash
cd /home/user/Aerius-Desktop

# Install all npm dependencies
npm install
```

This will install:
- Electron and build tools
- React and related packages
- All UI dependencies

**Expected installation time**: 2-5 minutes

### 6. Test Project Aerius CLI

Before running the desktop app, verify the backend works:

```bash
cd /home/user/Project-Aerius
python3 main.py
```

You should see:
- Agent initialization messages
- Welcome screen
- Prompt for input

Type `help` to see available commands, then `exit` to quit.

If you see errors:
- Check that API keys are correct in `.env`
- Verify all Python packages installed
- Review `Project-Aerius/TROUBLESHOOTING.md`

### 7. Run Aerius Desktop

```bash
cd /home/user/Aerius-Desktop
npm start
```

This will:
1. Start the React development server (port 3000)
2. Launch the Electron window
3. Start the Python backend bridge
4. Initialize the orchestrator

**First launch may take 30-60 seconds** as agents initialize.

### 8. Verify Everything Works

In the desktop app:

1. **Check Agent Status**: Look at the sidebar - agents should show green dots
2. **Test a Simple Query**: Type "hello" and press Enter
3. **Try an Agent**: "Show my GitHub repositories" (if GitHub token configured)
4. **Check Settings**: Click the settings button to verify preferences load

## Common Setup Issues

### Issue: "Failed to import Project Aerius"

**Cause**: Desktop app can't find the Python orchestrator

**Fix**:
```bash
# Verify directory structure
ls /home/user/Project-Aerius/orchestrator.py

# If path is different, edit backend/bridge.py line 14:
# project_aerius_path = Path(__file__).parent.parent.parent / "Project-Aerius"
```

### Issue: "No module named 'google.generativeai'"

**Cause**: Python dependencies not installed

**Fix**:
```bash
cd /home/user/Project-Aerius
pip install -r requirements.txt
```

### Issue: "Electron app won't start"

**Cause**: Port 3000 might be in use

**Fix**:
```bash
# Check what's using port 3000
lsof -i :3000  # macOS/Linux
netstat -ano | findstr :3000  # Windows

# Kill the process or change the port in package.json
```

### Issue: "Agents showing as unavailable"

**Cause**: Missing API keys or incorrect configuration

**Fix**:
1. Check `.env` file has correct keys
2. Verify keys are valid (test in Project-Aerius CLI first)
3. Review agent initialization logs in terminal

### Issue: "Python process exited with code 1"

**Cause**: Python error during startup

**Fix**:
1. Check terminal for Python error messages
2. Test `python3 backend/bridge.py` directly
3. Verify Python version is 3.8+

## Building for Production

### Create Distributable Package

```bash
cd /home/user/Aerius-Desktop

# Build the React app
npm run build

# Package for your platform
npm run package
```

Installers will be created in `dist/`:
- **macOS**: `Aerius-1.0.0.dmg`
- **Windows**: `Aerius Setup 1.0.0.exe`
- **Linux**: `Aerius-1.0.0.AppImage`

### Cross-Platform Builds

To build for multiple platforms from one machine:

```bash
# Build for macOS
npm run package -- --mac

# Build for Windows
npm run package -- --win

# Build for Linux
npm run package -- --linux
```

**Note**: Building for macOS requires macOS, Windows builds need Windows or Wine.

## Development Mode

### Run Components Separately

For debugging, you can run each part independently:

```bash
# Terminal 1: React dev server
cd /home/user/Aerius-Desktop
npm run start:react

# Terminal 2: Electron app
cd /home/user/Aerius-Desktop
npm run start:electron

# Terminal 3: Python backend (manual testing)
cd /home/user/Aerius-Desktop
python3 backend/bridge.py
```

### Enable DevTools

DevTools automatically open in development mode. To open in production:

Edit `electron/main.js` line 34:
```javascript
// Change:
if (isDev) {
  mainWindow.webContents.openDevTools();
}

// To:
mainWindow.webContents.openDevTools(); // Always open
```

## Next Steps

After successful setup:

1. **Explore Features**: Try the example prompts in the chat interface
2. **Configure Agents**: Add API keys for agents you want to use
3. **Customize UI**: Edit `src/styles/App.css` for theme changes
4. **Read Documentation**: Review `README.md` for full feature list
5. **Join Community**: Report issues and request features

## Getting Help

If you encounter issues not covered here:

1. **Check Logs**:
   - Electron console (View → Toggle Developer Tools)
   - Python output in terminal
   - Project-Aerius logs in `~/.aerius/sessions/`

2. **Review Documentation**:
   - `README.md` - Feature overview
   - `Project-Aerius/TROUBLESHOOTING.md` - Backend issues
   - `Project-Aerius/QUICKSTART.md` - Agent setup

3. **Common Solutions**:
   - Restart the app
   - Clear cache: Delete `node_modules` and `npm install`
   - Reset config: Delete `~/Library/Application Support/aerius-desktop` (macOS)
   - Verify API keys are correct

4. **Report Issues**:
   - Include error messages
   - Describe steps to reproduce
   - Share relevant logs (remove API keys!)

## Success Checklist

- [ ] Node.js v18+ installed
- [ ] Python 3.8+ installed
- [ ] Project-Aerius in correct location
- [ ] Python dependencies installed
- [ ] API keys configured in `.env`
- [ ] Desktop app dependencies installed (`npm install`)
- [ ] Project-Aerius CLI tested and working
- [ ] Desktop app launches without errors
- [ ] Agents show as healthy in sidebar
- [ ] Can send and receive messages
- [ ] Settings panel opens and saves

---

**Setup complete!** 🎉 You're ready to use Aerius Desktop for multi-agent orchestration.
