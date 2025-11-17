/**
 * TypeScript definitions for Electron API exposed through preload
 */

interface ElectronAPI {
  initialize: () => Promise<void>;
  sendMessage: (message: string) => Promise<void>;
  getStats: () => Promise<void>;
  resetSession: () => Promise<void>;
  getSettings: () => Promise<any>;
  saveSettings: (settings: any) => Promise<boolean>;
  getHistory: () => Promise<any[]>;
  saveConversation: (conversation: any) => Promise<boolean>;
  clearHistory: () => Promise<boolean>;
  exportConversation: (conversation: any) => Promise<{ success: boolean; path?: string }>;
  onPythonMessage: (callback: (data: any) => void) => void;
  onPythonError: (callback: (data: any) => void) => void;
  removeAllListeners: (channel: string) => void;
}

declare global {
  interface Window {
    electronAPI: ElectronAPI;
  }
}

export {};
