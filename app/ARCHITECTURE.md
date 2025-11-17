# Aerius Desktop - Architecture Documentation

Technical architecture and design decisions for the Aerius Desktop application.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Aerius Desktop App                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │            React Frontend (Renderer)                │    │
│  │  ┌──────────────────────────────────────────────┐  │    │
│  │  │  Components                                   │  │    │
│  │  │  • ChatInterface  • Settings                  │  │    │
│  │  │  • Sidebar        • Message                   │  │    │
│  │  │  • AgentStatus    • LoadingScreen            │  │    │
│  │  └──────────────────────────────────────────────┘  │    │
│  │  ┌──────────────────────────────────────────────┐  │    │
│  │  │  Services                                     │  │    │
│  │  │  • OrchestratorService (IPC communication)    │  │    │
│  │  └──────────────────────────────────────────────┘  │    │
│  └────────────────────────────────────────────────────┘    │
│                           ↕ IPC                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │         Electron Main Process (Node.js)            │    │
│  │  • Window management                               │    │
│  │  • Python process spawning                         │    │
│  │  • IPC message routing                             │    │
│  │  • Settings persistence (electron-store)           │    │
│  └────────────────────────────────────────────────────┘    │
│                      ↕ stdin/stdout                          │
│  ┌────────────────────────────────────────────────────┐    │
│  │          Python Backend Bridge                      │    │
│  │  • JSON message protocol                           │    │
│  │  • Command handling                                │    │
│  │  • Session management                              │    │
│  └────────────────────────────────────────────────────┘    │
│                           ↕                                  │
│  ┌────────────────────────────────────────────────────┐    │
│  │         Project Aerius Orchestrator                │    │
│  │  • Multi-agent coordination                        │    │
│  │  • LLM integration (Gemini)                        │    │
│  │  • Agent management                                │    │
│  │  • Task execution                                  │    │
│  └────────────────────────────────────────────────────┘    │
│                           ↕                                  │
│         ┌──────────┬──────────┬──────────┬──────────┐      │
│         │  Slack   │  GitHub  │   Jira   │  Notion  │      │
│         │  Agent   │  Agent   │  Agent   │  Agent   │      │
│         └──────────┴──────────┴──────────┴──────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Technology Stack

### Frontend Layer

**React 18 + TypeScript**
- **Why React**: Component-based architecture, rich ecosystem, performance
- **Why TypeScript**: Type safety, better IDE support, fewer runtime errors
- **State Management**: React Hooks (useState, useEffect, useRef)
- **Styling**: Tailwind CSS + Custom CSS for precise control
- **Markdown**: react-markdown with syntax highlighting (Prism.js)

### Desktop Layer

**Electron 28**
- **Why Electron**: Cross-platform, web technologies, large community
- **Main Process**: Window management, system integration, IPC
- **Renderer Process**: React app with security isolation
- **Preload Script**: Safe IPC bridge with contextBridge
- **Storage**: electron-store for persistent settings

### Backend Layer

**Python 3.8+**
- **Bridge Pattern**: Isolates desktop app from orchestrator complexity
- **Communication**: JSON over stdin/stdout (simple, reliable)
- **Async/Await**: Non-blocking orchestrator operations
- **Error Handling**: Graceful degradation, detailed error messages

### Build & Development

- **Bundler**: React Scripts (webpack under the hood)
- **Package Manager**: npm
- **Builder**: electron-builder (creates installers)
- **Dev Tools**: Concurrently, wait-on for parallel processes

## Communication Protocol

### IPC Flow (Frontend ↔ Electron)

```typescript
// Frontend sends command
window.electronAPI.sendMessage("Hello, agents!")

// Electron receives via IPC
ipcMain.handle('send-message', async (event, message) => {
  sendToPython({ type: 'message', data: { text: message } })
})

// Frontend receives responses
window.electronAPI.onPythonMessage((data) => {
  // Handle: initialized, processing, response, error, etc.
})
```

### JSON Protocol (Electron ↔ Python)

**Command Format** (Electron → Python):
```json
{
  "type": "message|initialize|stats|reset|shutdown",
  "data": {
    "text": "User message here"
  }
}
```

**Response Format** (Python → Electron):
```json
{
  "type": "ready|status|initialized|processing|response|error",
  "data": {
    "message": "Response text",
    "session_id": "abc123",
    "agents": [...]
  }
}
```

## Component Architecture

### Frontend Components

```
App.tsx (Root)
├── LoadingScreen (initialization)
├── Sidebar
│   ├── AgentStatus (per agent)
│   └── ConversationItem (history)
├── ChatInterface
│   ├── Message (per message)
│   └── Input
└── Settings (modal)
```

**Component Responsibilities**:

1. **App.tsx**
   - Application state management
   - Backend message handling
   - Route between loading/main/settings

2. **ChatInterface**
   - Message display
   - User input handling
   - Auto-scroll, typing indicators
   - Export functionality

3. **Message**
   - Markdown rendering
   - Syntax highlighting
   - Role-based styling
   - Timestamp formatting

4. **Sidebar**
   - Agent status display
   - Conversation history
   - Navigation controls
   - Settings access

5. **Settings**
   - Preference management
   - Theme selection
   - Data controls
   - About information

### Service Layer

**OrchestratorService** (Singleton)
- Manages all backend communication
- Handles IPC message routing
- Provides async API for components
- Implements observer pattern for real-time updates

```typescript
orchestratorService.onMessage((data) => {
  // React components subscribe to backend events
})

await orchestratorService.sendMessage("text")
await orchestratorService.resetSession()
```

## Data Flow

### User Message Flow

1. **User types** → `ChatInterface` input
2. **Enter pressed** → `onSendMessage` callback
3. **Service call** → `orchestratorService.sendMessage(text)`
4. **IPC** → `window.electronAPI.sendMessage(text)`
5. **Main process** → `ipcMain.handle('send-message')`
6. **Python stdin** → `{ type: 'message', data: { text } }`
7. **Bridge** → `handle_command('message')`
8. **Orchestrator** → `process_message(text)`
9. **Agents** → Execute tasks
10. **Response** → Bubble back up through layers
11. **UI update** → Message appears in chat

### State Management

**React State**:
```typescript
const [messages, setMessages] = useState<Message[]>([])
const [agents, setAgents] = useState<Agent[]>([])
const [sessionId, setSessionId] = useState<string>('')
```

**Persistent State** (electron-store):
- User settings (theme, fontSize, notifications)
- Conversation history (last 50 sessions)
- Window position and size

**Session State** (Python):
- Current conversation context
- Agent health status
- Active session ID

## Security Model

### Context Isolation

```javascript
webPreferences: {
  nodeIntegration: false,        // No direct Node.js access
  contextIsolation: true,        // Isolated context
  preload: path.join(__dirname, 'preload.js')
}
```

### Preload Script

```javascript
contextBridge.exposeInMainWorld('electronAPI', {
  // Only expose specific, safe methods
  sendMessage: (msg) => ipcRenderer.invoke('send-message', msg),
  // No direct access to Node.js or Electron internals
})
```

### Input Validation

- Frontend: TypeScript type checking
- Electron: Parameter validation in IPC handlers
- Python: JSON schema validation, error handling

### API Keys

- Stored in `.env` (not committed to git)
- Never exposed to frontend
- Handled only in Python backend
- electron-store encrypts sensitive data

## Performance Optimizations

### Frontend

1. **React.memo**: Prevent unnecessary re-renders
2. **Lazy Loading**: Code splitting for settings modal
3. **Virtual Scrolling**: Planned for long message lists
4. **Debouncing**: Input resize, search filtering

### Backend

1. **Async Operations**: Non-blocking Python async/await
2. **Agent Caching**: Reuse initialized agents
3. **Smart Summarization**: Condense verbose responses
4. **Connection Pooling**: MCP agent connections

### Communication

1. **Batching**: Multiple Python messages in single IPC event
2. **Streaming**: Planned for real-time agent responses
3. **Compression**: Planned for large data transfers

## Error Handling Strategy

### Frontend Errors

```typescript
try {
  await orchestratorService.sendMessage(text)
} catch (error) {
  toast.error('Failed to send message')
  console.error(error)
}
```

### Backend Errors

```python
try:
    response = await self.orchestrator.process_message(user_message)
    self.send_message("response", {"text": response})
except Exception as e:
    self.send_message("error", {
        "message": "Failed to process message",
        "details": str(e)
    })
```

### Python Process Crash

```javascript
pythonProcess.on('close', (code) => {
  console.error(`Python exited: ${code}`)
  mainWindow.webContents.send('python-error', {
    message: 'Backend disconnected. Please restart.'
  })
})
```

## Build Process

### Development Build

```bash
npm start
```

1. `concurrently` starts both:
   - `react-scripts start` (port 3000)
   - `electron .` (after React ready)
2. Electron loads `http://localhost:3000`
3. Hot reload enabled
4. DevTools open

### Production Build

```bash
npm run package
```

1. `react-scripts build` creates optimized bundle
2. `electron-builder` packages:
   - Minified JavaScript
   - Optimized assets
   - Platform-specific installer
3. Output: `dist/Aerius-1.0.0.[dmg|exe|AppImage]`

## Deployment Considerations

### Platform Differences

**macOS**:
- Code signing required for distribution
- Notarization needed for Gatekeeper
- DMG with drag-to-Applications

**Windows**:
- NSIS installer with registry entries
- Code signing recommended
- Auto-update support

**Linux**:
- AppImage (portable)
- .deb and .rpm options
- Desktop entry files

### Auto-Updates

Planned integration with `electron-updater`:
```javascript
autoUpdater.checkForUpdatesAndNotify()
```

### Analytics

Privacy-respecting analytics planned:
- Usage statistics (opt-in)
- Error reporting (anonymized)
- Feature usage tracking

## Future Enhancements

### Short Term

1. **Streaming Responses**: Real-time agent output
2. **Voice Input**: Speech-to-text integration
3. **Keyboard Shortcuts**: Power user features
4. **Themes**: Light mode, custom themes

### Medium Term

1. **Plugins**: Third-party agent extensions
2. **Cloud Sync**: Cross-device conversation history
3. **Collaboration**: Shared sessions
4. **Mobile App**: iOS/Android companion

### Long Term

1. **Local LLM**: Offline mode with local models
2. **Custom Agents**: Visual agent builder
3. **Workflow Automation**: Scheduled tasks
4. **Enterprise Features**: SSO, audit logs

## Testing Strategy

### Unit Tests

```typescript
describe('OrchestratorService', () => {
  it('should send message via IPC', async () => {
    await orchestratorService.sendMessage('test')
    expect(ipcRenderer.invoke).toHaveBeenCalled()
  })
})
```

### Integration Tests

```typescript
describe('Chat Flow', () => {
  it('should display response after sending message', async () => {
    render(<App />)
    fireEvent.change(input, { target: { value: 'Hello' } })
    fireEvent.submit(form)
    await waitFor(() => {
      expect(screen.getByText(/Hello/)).toBeInTheDocument()
    })
  })
})
```

### E2E Tests

Planned with Playwright:
```typescript
test('complete conversation flow', async ({ page }) => {
  await page.goto('http://localhost:3000')
  await page.fill('textarea', 'Show my tasks')
  await page.keyboard.press('Enter')
  await expect(page.locator('.message-assistant')).toBeVisible()
})
```

## Development Guidelines

### Code Style

- **TypeScript**: Strict mode enabled
- **React**: Functional components with hooks
- **CSS**: BEM-like naming convention
- **Formatting**: Prettier + ESLint

### Commit Messages

```
type(scope): subject

- feat: New feature
- fix: Bug fix
- docs: Documentation
- style: Formatting
- refactor: Code restructure
- test: Tests
- chore: Maintenance
```

### Branch Strategy

- `main`: Production-ready code
- `develop`: Integration branch
- `feature/*`: New features
- `fix/*`: Bug fixes

---

**This architecture provides**:
- ✅ Separation of concerns
- ✅ Maintainability and extensibility
- ✅ Performance and responsiveness
- ✅ Security and reliability
- ✅ Cross-platform compatibility
