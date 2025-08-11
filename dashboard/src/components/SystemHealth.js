/**
 * SystemHealth Component
 * Displays system health metrics and alerts
 */

import React, { useState, useEffect } from 'react';
import { useSystemStatus, useAlerts } from '../hooks/useWebSocket';
import robustAPI from '../services/robustApi';

const SystemHealth = ({ className = '' }) => {
  const [systemMetrics, setSystemMetrics] = useState({});
  const [systemHealth, setSystemHealth] = useState('unknown');
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(new Date());
  const { systemStatus, systemHealth: wsSystemHealth } = useSystemStatus();
  const { alerts: wsAlerts } = useAlerts();

  useEffect(() => {
    loadSystemData();
    const interval = setInterval(loadSystemData, 30000); // Update every 30 seconds
    
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (wsSystemHealth) {
      setSystemHealth(wsSystemHealth);
    }
  }, [wsSystemHealth]);

  useEffect(() => {
    if (wsAlerts && wsAlerts.length > 0) {
      setAlerts(wsAlerts);
    }
  }, [wsAlerts]);

  const loadSystemData = async () => {
    try {
      console.log('🔄 Loading system health data...');
      
      const [healthResponse, metricsResponse] = await Promise.all([
        robustAPI.fetchWithRetry('/api/v1/monitoring/health'),
        robustAPI.fetchWithRetry('/api/v1/monitoring/metrics')
      ]);

      console.log('💚 System health data loaded:', { healthResponse, metricsResponse });

      setSystemHealth(healthResponse.status || 'unknown');
      setSystemMetrics(healthResponse.metrics || metricsResponse || {});
      setAlerts(healthResponse.alerts || []);
      setLastUpdated(new Date());
      setError(null);

      if (robustAPI.isInFallbackMode()) {
        setError('Using cached system data - API temporarily unavailable');
      }
    } catch (err) {
      setError(`Failed to load system data: ${err.message}`);
      console.error('❌ Error loading system data:', err);
    } finally {
      setLoading(false);
    }
  };

  const getHealthColor = (status) => {
    switch (status) {
      case 'healthy':
        return 'text-success-600 dark:text-success-400 bg-success-100 dark:bg-success-900';
      case 'degraded':
        return 'text-warning-600 dark:text-warning-400 bg-warning-100 dark:bg-warning-900';
      case 'unhealthy':
        return 'text-danger-600 dark:text-danger-400 bg-danger-100 dark:bg-danger-900';
      default:
        return 'text-gray-600 dark:text-gray-400 bg-gray-100 dark:bg-gray-700';
    }
  };

  const getHealthIcon = (status) => {
    switch (status) {
      case 'healthy':
        return (
          <svg className="w-5 h-5 text-success-600 dark:text-success-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        );
      case 'degraded':
        return (
          <svg className="w-5 h-5 text-warning-600 dark:text-warning-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
        );
      case 'unhealthy':
        return (
          <svg className="w-5 h-5 text-danger-600 dark:text-danger-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        );
      default:
        return (
          <svg className="w-5 h-5 text-gray-600 dark:text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        );
    }
  };

  const getMetricColor = (value, thresholds) => {
    if (value >= thresholds.critical) {
      return 'text-danger-600 dark:text-danger-400';
    } else if (value >= thresholds.warning) {
      return 'text-warning-600 dark:text-warning-400';
    }
    return 'text-success-600 dark:text-success-400';
  };

  const getAlertIcon = (severity) => {
    switch (severity) {
      case 'critical':
        return (
          <svg className="w-4 h-4 text-danger-600 dark:text-danger-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
        );
      case 'warning':
        return (
          <svg className="w-4 h-4 text-warning-600 dark:text-warning-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
        );
      default:
        return (
          <svg className="w-4 h-4 text-primary-600 dark:text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        );
    }
  };

  const formatBytes = (bytes) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatUptime = (seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  if (loading) {
    return (
      <div className={`card ${className}`}>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
            System Health
          </h2>
          <div className="loading-spinner"></div>
        </div>
        <div className="space-y-4">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="animate-pulse">
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-full"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`card ${className}`}>
        <div className="text-center py-8">
          <div className="text-danger-600 dark:text-danger-400 mb-4">
            <svg className="w-12 h-12 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
          </div>
          <p className="text-gray-600 dark:text-gray-400 mb-4">{error}</p>
          <button
            onClick={loadSystemData}
            className="btn-primary"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`card ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
            System Health
          </h2>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Last updated: {lastUpdated.toLocaleTimeString()}
          </p>
        </div>
        
        <div className="flex items-center space-x-2">
          <div className={`status-indicator ${getHealthColor(systemHealth)}`}>
            {getHealthIcon(systemHealth)}
            <span className="ml-2 capitalize">{systemHealth}</span>
          </div>
          <button
            onClick={loadSystemData}
            className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </button>
        </div>
      </div>

      <div className="space-y-6">
        {/* System Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* CPU Usage */}
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-600 dark:text-gray-400">CPU Usage</span>
              <span className={`text-lg font-semibold ${getMetricColor(systemMetrics.cpu?.percent || 0, { warning: 70, critical: 90 })}`}>
                {(systemMetrics.cpu?.percent || 0).toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all duration-300 ${
                  (systemMetrics.cpu?.percent || 0) >= 90 ? 'bg-danger-500' :
                  (systemMetrics.cpu?.percent || 0) >= 70 ? 'bg-warning-500' : 'bg-success-500'
                }`}
                style={{ width: `${systemMetrics.cpu?.percent || 0}%` }}
              ></div>
            </div>
          </div>

          {/* Memory Usage */}
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Memory Usage</span>
              <span className={`text-lg font-semibold ${getMetricColor(systemMetrics.memory?.percent || 0, { warning: 70, critical: 90 })}`}>
                {(systemMetrics.memory?.percent || 0).toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all duration-300 ${
                  (systemMetrics.memory?.percent || 0) >= 90 ? 'bg-danger-500' :
                  (systemMetrics.memory?.percent || 0) >= 70 ? 'bg-warning-500' : 'bg-success-500'
                }`}
                style={{ width: `${systemMetrics.memory?.percent || 0}%` }}
              ></div>
            </div>
            <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
              <span>{formatBytes(systemMetrics.memory?.used || 0)}</span>
              <span>{formatBytes(systemMetrics.memory?.total || 0)}</span>
            </div>
          </div>

          {/* Disk Usage */}
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Disk Usage</span>
              <span className={`text-lg font-semibold ${getMetricColor(systemMetrics.disk?.percent || 0, { warning: 80, critical: 95 })}`}>
                {(systemMetrics.disk?.percent || 0).toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all duration-300 ${
                  (systemMetrics.disk?.percent || 0) >= 95 ? 'bg-danger-500' :
                  (systemMetrics.disk?.percent || 0) >= 80 ? 'bg-warning-500' : 'bg-success-500'
                }`}
                style={{ width: `${systemMetrics.disk?.percent || 0}%` }}
              ></div>
            </div>
            <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
              <span>{formatBytes(systemMetrics.disk?.used || 0)}</span>
              <span>{formatBytes(systemMetrics.disk?.total || 0)}</span>
            </div>
          </div>
        </div>

        {/* Process Information */}
        {systemMetrics.process && (
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
            <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-3">Process Information</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <span className="text-xs text-gray-500 dark:text-gray-400">PID</span>
                <p className="text-sm font-medium text-gray-900 dark:text-white">{systemMetrics.process.pid}</p>
              </div>
              <div>
                <span className="text-xs text-gray-500 dark:text-gray-400">Uptime</span>
                <p className="text-sm font-medium text-gray-900 dark:text-white">
                  {formatUptime(systemMetrics.process.uptime)}
                </p>
              </div>
              <div>
                <span className="text-xs text-gray-500 dark:text-gray-400">Threads</span>
                <p className="text-sm font-medium text-gray-900 dark:text-white">{systemMetrics.process.threads}</p>
              </div>
              <div>
                <span className="text-xs text-gray-500 dark:text-gray-400">Memory</span>
                <p className="text-sm font-medium text-gray-900 dark:text-white">
                  {formatBytes(systemMetrics.process.memory_mb * 1024 * 1024)}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Active Alerts */}
        <div>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium text-gray-900 dark:text-white">Active Alerts</h3>
            <span className="text-sm text-gray-500 dark:text-gray-400">
              {(alerts || []).length} alert{(alerts || []).length !== 1 ? 's' : ''}
            </span>
          </div>
          
          {(alerts || []).length === 0 ? (
            <div className="text-center py-4">
              <div className="text-success-600 dark:text-success-400 mb-2">
                <svg className="w-8 h-8 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">No active alerts</p>
            </div>
          ) : (
            <div className="space-y-2">
              {(alerts || []).slice(0, 5).map((alert, index) => (
                <div
                  key={alert.id || index}
                  className={`flex items-center space-x-3 p-3 rounded-lg ${
                    alert.severity === 'critical' ? 'bg-danger-50 dark:bg-danger-900/20' :
                    alert.severity === 'warning' ? 'bg-warning-50 dark:bg-warning-900/20' :
                    'bg-primary-50 dark:bg-primary-900/20'
                  }`}
                >
                  {getAlertIcon(alert.severity)}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                      {alert.message}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      {new Date(alert.timestamp).toLocaleString()}
                    </p>
                  </div>
                  <span className={`status-indicator ${
                    alert.severity === 'critical' ? 'bg-danger-100 text-danger-800 dark:bg-danger-900 dark:text-danger-100' :
                    alert.severity === 'warning' ? 'bg-warning-100 text-warning-800 dark:bg-warning-900 dark:text-warning-100' :
                    'bg-primary-100 text-primary-800 dark:bg-primary-900 dark:text-primary-100'
                  }`}>
                    {alert.severity}
                  </span>
                </div>
              ))}
              
              {(alerts || []).length > 5 && (
                <div className="text-center py-2">
                  <button className="text-sm text-primary-600 dark:text-primary-400 hover:text-primary-800 dark:hover:text-primary-300">
                    View {(alerts || []).length - 5} more alerts
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SystemHealth;