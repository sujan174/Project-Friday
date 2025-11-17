# Aerius Desktop

A professional-grade desktop application for the Project Aerius Multi-Agent Orchestration System.

## 🌟 Features

- **Beautiful Modern UI**: Clean, professional interface built with React and Tailwind CSS
- **Multi-Agent Orchestration**: Seamlessly coordinate tasks across Slack, GitHub, Jira, Notion, and more
- **Real-time Communication**: Instant feedback from AI agents through IPC
- **Session Management**: Save, load, and export conversation history
- **Agent Monitoring**: Live status indicators for all connected agents
- **Cross-Platform**: Works on macOS, Windows, and Linux
- **Markdown Support**: Full markdown rendering with syntax highlighting
- **Dark Mode**: Eye-friendly dark theme optimized for long sessions

## 📋 Prerequisites

Before running Aerius Desktop, ensure you have:

1. **Node.js** (v18 or higher)
2. **Python 3.8+**
3. **Project Aerius** installed in the parent directory
4. All required API keys configured (see Project Aerius documentation)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
npm install
```

### 2. Set Up Python Environment

The desktop app uses the Python backend from Project Aerius. Make sure:

- Project-Aerius is in `/home/user/Project-Aerius` (or adjust the path in `backend/bridge.py`)
- All Python dependencies are installed:

```bash
cd ../Project-Aerius
pip install -r requirements.txt
```

### 3. Configure Environment

Copy the `.env.example` from Project-Aerius and configure your API keys:

```bash
cp ../Project-Aerius/.env.example ../Project-Aerius/.env
```

Edit `.env` and add your API keys:
- `GOOGLE_API_KEY` (required)
- `SLACK_BOT_TOKEN` (optional)
- `GITHUB_PERSONAL_ACCESS_TOKEN` (optional)
- `JIRA_URL`, `JIRA_USERNAME`, `JIRA_API_TOKEN` (optional)
- `NOTION_TOKEN` (optional)

### 4. Run the App

Development mode:
```bash
npm start
```

This will:
1. Start the React development server on `http://localhost:3000`
2. Launch the Electron app
3. Start the Python backend bridge

## 📦 Building

Create a distributable package:

```bash
npm run package
```

This will create platform-specific installers in the `dist/` directory:
- **macOS**: `.dmg` file
- **Windows**: `.exe` installer
- **Linux**: `.AppImage` file

## 🏗️ Architecture

```
Aerius Desktop
├── Electron Main Process (electron/main.js)
│   ├── Window management
│   ├── Python process spawning
│   └── IPC communication
│
├── Python Backend (backend/bridge.py)
│   ├── Orchestrator initialization
│   ├── Message processing
│   └── Agent coordination
│
└── React Frontend (src/)
    ├── Chat Interface
    ├── Agent Status
    ├── Settings
    └── Session Management
```

### Communication Flow

1. **User Input** → React UI
2. **IPC Call** → Electron Main Process
3. **JSON Message** → Python Backend (stdin)
4. **Orchestrator** → Process through agents
5. **JSON Response** → Python Backend (stdout)
6. **IPC Event** → React UI
7. **Display** → User sees response

## 🎨 UI Components

### ChatInterface
- Main conversation view
- Message input with auto-resize
- Markdown rendering with syntax highlighting
- Real-time typing indicators

### Sidebar
- Agent status monitoring
- Conversation history
- New conversation button
- Settings access

### Settings
- Theme preferences
- Font size adjustment
- Notification settings
- Data management

### LoadingScreen
- Beautiful initialization animation
- Loading progress indicators
- System status updates

## 🛠️ Development

### Project Structure

```
Aerius-Desktop/
├── electron/              # Electron main process
│   ├── main.js           # Main Electron logic
│   └── preload.js        # IPC bridge
│
├── backend/              # Python backend
│   └── bridge.py         # Orchestrator bridge
│
├── src/                  # React frontend
│   ├── components/       # UI components
│   ├── services/         # Business logic
│   ├── styles/           # CSS styles
│   └── types/            # TypeScript types
│
├── public/               # Static assets
├── package.json          # Dependencies
└── tsconfig.json         # TypeScript config
```

### Key Technologies

- **Frontend**: React 18, TypeScript, Tailwind CSS
- **Desktop**: Electron 28
- **Backend**: Python 3.8+, Project Aerius
- **Communication**: IPC (Inter-Process Communication)
- **State Management**: React Hooks
- **Persistence**: electron-store
- **Markdown**: react-markdown, rehype-highlight

## 📝 Scripts

- `npm start` - Start development mode
- `npm run build` - Build React app
- `npm run package` - Create distributable
- `npm test` - Run tests
- `npm run start:react` - Start React only
- `npm run start:electron` - Start Electron only

## 🔧 Configuration

### Electron Settings

Modify `electron/main.js` for:
- Window size and behavior
- DevTools access
- IPC handlers

### UI Customization

Edit `src/styles/App.css` for:
- Color scheme
- Typography
- Layout adjustments

### Backend Configuration

Adjust `backend/bridge.py` for:
- Project Aerius path
- Orchestrator settings
- Error handling

## 🐛 Troubleshooting

### App won't start
- Check that Python 3.8+ is installed
- Verify Project-Aerius is in the correct location
- Ensure all dependencies are installed

### Python backend errors
- Check Python console output
- Verify API keys in `.env`
- Test Project-Aerius CLI first

### Agents not loading
- Check agent configuration in Project-Aerius
- Verify MCP servers are installed
- Review agent initialization logs

### UI issues
- Clear browser cache (Cmd/Ctrl + Shift + R)
- Check console for errors
- Verify React build is up to date

## 📚 Documentation

For more information about the orchestration system:
- [Project Aerius Documentation](../Project-Aerius/README.md)
- [Agent Configuration](../Project-Aerius/QUICKSTART.md)
- [Troubleshooting Guide](../Project-Aerius/TROUBLESHOOTING.md)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Project Aerius orchestration engine
- Electron framework
- React and the React community
- All open-source dependencies

## 💡 Tips

### Keyboard Shortcuts
- `Cmd/Ctrl + N` - New conversation
- `Enter` - Send message
- `Shift + Enter` - New line
- `Cmd/Ctrl + ,` - Open settings
- `Cmd/Ctrl + E` - Export conversation

### Best Practices
- Keep conversations focused on specific tasks
- Use the example prompts for common operations
- Monitor agent status before complex operations
- Export important conversations regularly
- Clear history periodically for better performance

## 🚀 Next Steps

1. **Explore Agents**: Check what each agent can do in the sidebar
2. **Try Examples**: Use the example prompts to get started
3. **Configure Settings**: Customize the app to your preferences
4. **Save Sessions**: Export important conversations
5. **Provide Feedback**: Report issues and suggest features

---

**Built with ❤️ for productive multi-agent workflows**
