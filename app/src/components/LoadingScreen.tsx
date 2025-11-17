import React from 'react';

const LoadingScreen: React.FC = () => {
  return (
    <div className="loading-screen">
      <div className="loading-content">
        <div className="loading-logo">
          <div className="loading-spinner">
            <div className="spinner-ring"></div>
            <div className="spinner-ring"></div>
            <div className="spinner-ring"></div>
          </div>
        </div>
        <h1 className="loading-title">Aerius</h1>
        <p className="loading-text">Initializing multi-agent orchestration system...</p>
        <div className="loading-steps">
          <div className="loading-step">
            <div className="step-dot"></div>
            <span>Loading agents</span>
          </div>
          <div className="loading-step">
            <div className="step-dot"></div>
            <span>Establishing connections</span>
          </div>
          <div className="loading-step">
            <div className="step-dot"></div>
            <span>Preparing workspace</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoadingScreen;
