/**
 * Electron Preload Script
 *
 * Exposes safe IPC methods to the renderer process
 */

const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // Orchestrator commands
  initialize: () => ipcRenderer.invoke('initialize'),
  sendMessage: (message) => ipcRenderer.invoke('send-message', message),
  getStats: () => ipcRenderer.invoke('get-stats'),
  resetSession: () => ipcRenderer.invoke('reset-session'),

  // Settings
  getSettings: () => ipcRenderer.invoke('get-settings'),
  saveSettings: (settings) => ipcRenderer.invoke('save-settings', settings),

  // Conversation history
  getHistory: () => ipcRenderer.invoke('get-history'),
  saveConversation: (conversation) => ipcRenderer.invoke('save-conversation', conversation),
  clearHistory: () => ipcRenderer.invoke('clear-history'),
  exportConversation: (conversation) => ipcRenderer.invoke('export-conversation', conversation),

  // Listen for Python backend messages
  onPythonMessage: (callback) => {
    ipcRenderer.on('python-message', (event, data) => callback(data));
  },

  onPythonError: (callback) => {
    ipcRenderer.on('python-error', (event, data) => callback(data));
  },

  // Remove listeners
  removeAllListeners: (channel) => {
    ipcRenderer.removeAllListeners(channel);
  },
});
