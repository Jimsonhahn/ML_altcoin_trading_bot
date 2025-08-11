/**
 * Enhanced Connection Status Component
 * Shows real-time API and WebSocket connection status with fallback indicators
 */

import React, { useState, useEffect } from 'react';
import robustAPI from '../services/robustApi';

const ConnectionStatus = ({ isConnected = false }) => {
  const [apiStatus, setApiStatus] = useState('unknown');
  const [lastCheck, setLastCheck] = useState(null);
  const [retryCount, setRetryCount] = useState(0);
  const [fallbackMode, setFallbackMode] = useState(false);

  useEffect(() => {
    checkApiStatus();
    const interval = setInterval(checkApiStatus, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const checkApiStatus = async () => {
    try {
      const health = await robustAPI.fetchWithRetry('/health');
      if (health && health.status === 'healthy') {
        setApiStatus('connected');
        setRetryCount(0);
      } else {
        setApiStatus('degraded');
      }
      setFallbackMode(robustAPI.isInFallbackMode());
      setLastCheck(new Date());
    } catch (error) {
      setApiStatus('disconnected');
      setRetryCount(prev => prev + 1);
      setFallbackMode(true);
      setLastCheck(new Date());
    }
  };

  const getStatusColor = () => {
    if (fallbackMode) return 'bg-yellow-500';
    switch (apiStatus) {
      case 'connected': return 'bg-green-500';
      case 'degraded': return 'bg-yellow-500';
      case 'disconnected': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusText = () => {
    if (fallbackMode) return 'Offline Mode';
    switch (apiStatus) {
      case 'connected': return 'Connected';
      case 'degraded': return 'Degraded';
      case 'disconnected': return 'Disconnected';
      default: return 'Unknown';
    }
  };

  const getWebSocketStatus = () => {
    return isConnected ? 'Connected' : 'Disconnected';
  };

  const getWebSocketColor = () => {
    return isConnected ? 'bg-green-500' : 'bg-red-500';
  };

  return (
    <div className="flex items-center space-x-4">
      {/* API Status */}
      <div className="flex items-center space-x-2">
        <div className="flex items-center space-x-1">
          <div className={`w-2 h-2 rounded-full ${getStatusColor()}`}></div>
          <span className="text-sm text-gray-600 dark:text-gray-300">
            API: {getStatusText()}
          </span>
        </div>
        {fallbackMode && (
          <div className="flex items-center space-x-1">
            <svg className="w-4 h-4 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
            <span className="text-xs text-yellow-600 dark:text-yellow-400">
              Using cached data
            </span>
          </div>
        )}
      </div>

      {/* WebSocket Status */}
      <div className="flex items-center space-x-1">
        <div className={`w-2 h-2 rounded-full ${getWebSocketColor()}`}></div>
        <span className="text-sm text-gray-600 dark:text-gray-300">
          WS: {getWebSocketStatus()}
        </span>
      </div>

      {/* Retry Count */}
      {retryCount > 0 && (
        <div className="flex items-center space-x-1">
          <svg className="w-4 h-4 text-orange-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          <span className="text-xs text-orange-600 dark:text-orange-400">
            Retries: {retryCount}
          </span>
        </div>
      )}

      {/* Refresh Button */}
      <button
        onClick={checkApiStatus}
        className="p-1 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 rounded transition-colors"
        title="Check API Status"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
        </svg>
      </button>

      {/* Last Check Time */}
      {lastCheck && (
        <span className="text-xs text-gray-500 dark:text-gray-400">
          {lastCheck.toLocaleTimeString()}
        </span>
      )}
    </div>
  );
};

export default ConnectionStatus;