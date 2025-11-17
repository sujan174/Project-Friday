import React, { useState, useEffect } from 'react';
import { XMarkIcon } from '@heroicons/react/24/outline';
import orchestratorService from '../services/orchestratorService';
import toast from 'react-hot-toast';

interface SettingsProps {
  onClose: () => void;
}

const Settings: React.FC<SettingsProps> = ({ onClose }) => {
  const [theme, setTheme] = useState<'light' | 'dark'>('dark');
  const [fontSize, setFontSize] = useState<'small' | 'medium' | 'large'>('medium');
  const [notifications, setNotifications] = useState(true);

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      const settings = await orchestratorService.getSettings();
      setTheme(settings.theme || 'dark');
      setFontSize(settings.fontSize || 'medium');
      setNotifications(settings.notifications !== false);
    } catch (error) {
      console.error('Failed to load settings:', error);
    }
  };

  const handleSave = async () => {
    try {
      await orchestratorService.saveSettings({
        theme,
        fontSize,
        notifications,
      });
      toast.success('Settings saved successfully!');
      onClose();
    } catch (error) {
      toast.error('Failed to save settings');
    }
  };

  const handleClearHistory = async () => {
    if (window.confirm('Are you sure you want to clear all conversation history?')) {
      try {
        await orchestratorService.clearHistory();
        toast.success('Conversation history cleared');
      } catch (error) {
        toast.error('Failed to clear history');
      }
    }
  };

  return (
    <div className="settings-overlay" onClick={onClose}>
      <div className="settings-modal" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="settings-header">
          <h2>Settings</h2>
          <button onClick={onClose} className="icon-button" aria-label="Close settings">
            <XMarkIcon className="w-6 h-6" />
          </button>
        </div>

        {/* Content */}
        <div className="settings-content">
          {/* Appearance */}
          <div className="settings-section">
            <h3>Appearance</h3>

            <div className="setting-item">
              <label htmlFor="theme">Theme</label>
              <select
                id="theme"
                value={theme}
                onChange={(e) => setTheme(e.target.value as 'light' | 'dark')}
                className="setting-select"
              >
                <option value="light">Light</option>
                <option value="dark">Dark</option>
              </select>
            </div>

            <div className="setting-item">
              <label htmlFor="fontSize">Font Size</label>
              <select
                id="fontSize"
                value={fontSize}
                onChange={(e) => setFontSize(e.target.value as 'small' | 'medium' | 'large')}
                className="setting-select"
              >
                <option value="small">Small</option>
                <option value="medium">Medium</option>
                <option value="large">Large</option>
              </select>
            </div>
          </div>

          {/* Notifications */}
          <div className="settings-section">
            <h3>Notifications</h3>
            <div className="setting-item">
              <label htmlFor="notifications" className="checkbox-label">
                <input
                  type="checkbox"
                  id="notifications"
                  checked={notifications}
                  onChange={(e) => setNotifications(e.target.checked)}
                  className="setting-checkbox"
                />
                Enable desktop notifications
              </label>
            </div>
          </div>

          {/* Data */}
          <div className="settings-section">
            <h3>Data</h3>
            <div className="setting-item">
              <button onClick={handleClearHistory} className="danger-button">
                Clear Conversation History
              </button>
              <p className="setting-description">
                This will permanently delete all saved conversations.
              </p>
            </div>
          </div>

          {/* About */}
          <div className="settings-section">
            <h3>About</h3>
            <p className="about-text">
              <strong>Aerius Desktop</strong>
              <br />
              Version 1.0.0
              <br />
              Multi-Agent Orchestration System
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="settings-footer">
          <button onClick={onClose} className="secondary-button">
            Cancel
          </button>
          <button onClick={handleSave} className="primary-button">
            Save Changes
          </button>
        </div>
      </div>
    </div>
  );
};

export default Settings;
