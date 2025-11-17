/**
 * Orchestrator Service
 *
 * Handles all communication with the Python backend through Electron IPC
 */

export interface Agent {
  name: string;
  capabilities: string[];
  status: 'healthy' | 'degraded' | 'down' | 'unknown';
}

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
}

export interface SessionStats {
  session_id: string;
  message_count: number;
  agent_calls: number;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  timestamp: number;
}

type MessageHandler = (message: any) => void;

class OrchestratorService {
  private messageHandlers: Set<MessageHandler> = new Set();
  private initialized: boolean = false;

  constructor() {
    // Set up Python message listener
    if (window.electronAPI) {
      window.electronAPI.onPythonMessage((data: any) => {
        this.handlePythonMessage(data);
      });

      window.electronAPI.onPythonError((data: any) => {
        this.notifyHandlers({
          type: 'error',
          data: {
            message: 'Python backend error',
            details: data.message,
          },
        });
      });
    }
  }

  /**
   * Add a message handler
   */
  onMessage(handler: MessageHandler): () => void {
    this.messageHandlers.add(handler);
    return () => this.messageHandlers.delete(handler);
  }

  /**
   * Notify all handlers of a message
   */
  private notifyHandlers(message: any): void {
    this.messageHandlers.forEach((handler) => {
      try {
        handler(message);
      } catch (error) {
        console.error('Error in message handler:', error);
      }
    });
  }

  /**
   * Handle messages from Python backend
   */
  private handlePythonMessage(data: any): void {
    this.notifyHandlers(data);
  }

  /**
   * Initialize the orchestrator
   */
  async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }

    try {
      await window.electronAPI.initialize();
      this.initialized = true;
    } catch (error) {
      console.error('Failed to initialize:', error);
      throw error;
    }
  }

  /**
   * Send a message to the orchestrator
   */
  async sendMessage(message: string): Promise<void> {
    if (!this.initialized) {
      throw new Error('Orchestrator not initialized');
    }

    try {
      await window.electronAPI.sendMessage(message);
    } catch (error) {
      console.error('Failed to send message:', error);
      throw error;
    }
  }

  /**
   * Get session statistics
   */
  async getStats(): Promise<void> {
    try {
      await window.electronAPI.getStats();
    } catch (error) {
      console.error('Failed to get stats:', error);
      throw error;
    }
  }

  /**
   * Reset the current session
   */
  async resetSession(): Promise<void> {
    try {
      await window.electronAPI.resetSession();
      this.initialized = false;
    } catch (error) {
      console.error('Failed to reset session:', error);
      throw error;
    }
  }

  /**
   * Get settings
   */
  async getSettings(): Promise<any> {
    try {
      return await window.electronAPI.getSettings();
    } catch (error) {
      console.error('Failed to get settings:', error);
      throw error;
    }
  }

  /**
   * Save settings
   */
  async saveSettings(settings: any): Promise<boolean> {
    try {
      return await window.electronAPI.saveSettings(settings);
    } catch (error) {
      console.error('Failed to save settings:', error);
      throw error;
    }
  }

  /**
   * Get conversation history
   */
  async getHistory(): Promise<Conversation[]> {
    try {
      return await window.electronAPI.getHistory();
    } catch (error) {
      console.error('Failed to get history:', error);
      return [];
    }
  }

  /**
   * Save conversation
   */
  async saveConversation(conversation: Conversation): Promise<boolean> {
    try {
      return await window.electronAPI.saveConversation(conversation);
    } catch (error) {
      console.error('Failed to save conversation:', error);
      return false;
    }
  }

  /**
   * Clear conversation history
   */
  async clearHistory(): Promise<boolean> {
    try {
      return await window.electronAPI.clearHistory();
    } catch (error) {
      console.error('Failed to clear history:', error);
      return false;
    }
  }

  /**
   * Export conversation
   */
  async exportConversation(conversation: Conversation): Promise<{ success: boolean; path?: string }> {
    try {
      return await window.electronAPI.exportConversation(conversation);
    } catch (error) {
      console.error('Failed to export conversation:', error);
      return { success: false };
    }
  }
}

// Create singleton instance
const orchestratorService = new OrchestratorService();

export default orchestratorService;
