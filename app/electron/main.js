/**
 * Electron Main Process for Aerius Desktop
 *
 * Manages the application window, Python backend bridge, and IPC communication
 */

const { app, BrowserWindow, ipcMain } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const Store = require('electron-store');

// Initialize electron-store for persistent settings
const store = new Store();

let mainWindow = null;
let pythonProcess = null;

// Check if running in development mode
const isDev = !app.isPackaged;

/**
 * Create the main application window
 */
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 600,
    backgroundColor: '#0f1419',
    titleBarStyle: 'hiddenInset',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    },
    show: false, // Don't show until ready
  });

  // Load the React app
  const startUrl = isDev
    ? 'http://localhost:3000'
    : `file://${path.join(__dirname, '../build/index.html')}`;

  mainWindow.loadURL(startUrl);

  // Show window when ready
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  // Open DevTools in development
  if (isDev) {
    mainWindow.webContents.openDevTools();
  }

  // Handle window close
  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

/**
 * Start the Python backend bridge
 */
function startPythonBackend() {
  const pythonPath = 'python3'; // or 'python' on Windows
  const scriptPath = path.join(__dirname, '../backend/bridge.py');

  pythonProcess = spawn(pythonPath, [scriptPath], {
    stdio: ['pipe', 'pipe', 'pipe'],
  });

  // Handle Python stdout (messages from backend)
  pythonProcess.stdout.on('data', (data) => {
    const messages = data.toString().split('\n').filter(line => line.trim());

    messages.forEach(message => {
      try {
        const parsed = JSON.parse(message);
        // Forward to renderer process
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.webContents.send('python-message', parsed);
        }
      } catch (e) {
        console.error('Failed to parse Python message:', message);
      }
    });
  });

  // Handle Python stderr (errors and logs)
  pythonProcess.stderr.on('data', (data) => {
    console.error('Python stderr:', data.toString());
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('python-error', {
        type: 'error',
        message: data.toString(),
      });
    }
  });

  // Handle Python process exit
  pythonProcess.on('close', (code) => {
    console.log(`Python process exited with code ${code}`);
    pythonProcess = null;
  });

  return pythonProcess;
}

/**
 * Send command to Python backend
 */
function sendToPython(command) {
  if (pythonProcess && pythonProcess.stdin.writable) {
    pythonProcess.stdin.write(JSON.stringify(command) + '\n');
  } else {
    console.error('Python process not available');
  }
}

/**
 * App lifecycle events
 */
app.whenReady().then(() => {
  createWindow();
  startPythonBackend();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  // Cleanup Python process
  if (pythonProcess) {
    sendToPython({ type: 'shutdown' });
    pythonProcess.kill();
  }
});

/**
 * IPC Handlers - Communication between renderer and main process
 */

// Initialize orchestrator
ipcMain.handle('initialize', async () => {
  sendToPython({ type: 'initialize' });
});

// Send message to orchestrator
ipcMain.handle('send-message', async (event, message) => {
  sendToPython({ type: 'message', data: { text: message } });
});

// Get session statistics
ipcMain.handle('get-stats', async () => {
  sendToPython({ type: 'stats' });
});

// Reset session
ipcMain.handle('reset-session', async () => {
  sendToPython({ type: 'reset' });
});

// Get stored settings
ipcMain.handle('get-settings', async () => {
  return {
    theme: store.get('theme', 'dark'),
    fontSize: store.get('fontSize', 'medium'),
    notifications: store.get('notifications', true),
  };
});

// Save settings
ipcMain.handle('save-settings', async (event, settings) => {
  store.set('theme', settings.theme);
  store.set('fontSize', settings.fontSize);
  store.set('notifications', settings.notifications);
  return true;
});

// Get conversation history
ipcMain.handle('get-history', async () => {
  return store.get('conversationHistory', []);
});

// Save conversation
ipcMain.handle('save-conversation', async (event, conversation) => {
  const history = store.get('conversationHistory', []);
  history.unshift(conversation);
  // Keep only last 50 conversations
  store.set('conversationHistory', history.slice(0, 50));
  return true;
});

// Clear history
ipcMain.handle('clear-history', async () => {
  store.set('conversationHistory', []);
  return true;
});

// Export conversation
ipcMain.handle('export-conversation', async (event, conversation) => {
  const { dialog } = require('electron');
  const fs = require('fs');

  const { filePath } = await dialog.showSaveDialog(mainWindow, {
    title: 'Export Conversation',
    defaultPath: `aerius-conversation-${Date.now()}.json`,
    filters: [
      { name: 'JSON', extensions: ['json'] },
      { name: 'Text', extensions: ['txt'] },
      { name: 'Markdown', extensions: ['md'] },
    ],
  });

  if (filePath) {
    let content = '';
    const ext = path.extname(filePath);

    if (ext === '.json') {
      content = JSON.stringify(conversation, null, 2);
    } else if (ext === '.md') {
      content = `# Aerius Conversation - ${conversation.title}\n\n`;
      conversation.messages.forEach(msg => {
        content += `## ${msg.role === 'user' ? 'You' : 'Aerius'}\n\n${msg.content}\n\n`;
      });
    } else {
      conversation.messages.forEach(msg => {
        content += `${msg.role === 'user' ? 'You' : 'Aerius'}: ${msg.content}\n\n`;
      });
    }

    fs.writeFileSync(filePath, content);
    return { success: true, path: filePath };
  }

  return { success: false };
});
