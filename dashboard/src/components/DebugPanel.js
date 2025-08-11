/**
 * Debug Panel Component
 * Provides real-time debugging information for dashboard issues
 */

import React, { useState, useEffect } from 'react';
import robustAPI from '../services/robustApi';

const DebugPanel = ({ isVisible = false, onClose }) => {
  const [debugInfo, setDebugInfo] = useState({});
  const [apiTests, setApiTests] = useState({});
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (isVisible) {
      runDiagnostics();
    }
  }, [isVisible]);

  const runDiagnostics = async () => {
    setLoading(true);
    const info = {
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href,
      cacheStats: robustAPI.getCacheStats(),
      fallbackMode: robustAPI.isInFallbackMode()
    };

    // Test all critical API endpoints
    const endpoints = [
      { name: 'Health Check', path: '/health' },
      { name: 'Trading Status', path: '/api/v1/trading/status' },
      { name: 'Performance Data', path: '/api/v1/trading/performance' },
      { name: 'Positions', path: '/api/v1/trading/positions' },
      { name: 'Market Regime', path: '/api/v1/market/regime' },
      { name: 'Position Sizing', path: '/api/v1/risk/position-sizing' },
      { name: 'ML Analysis', path: '/api/v1/ml/analysis' },
      { name: 'Advanced Metrics', path: '/api/v1/analytics/advanced' },
      { name: 'Strategies List', path: '/api/v1/strategies/list' },
      { name: 'System Health', path: '/api/v1/monitoring/health' }
    ];

    const tests = {};
    
    for (const endpoint of endpoints) {
      try {
        const startTime = performance.now();
        const result = await robustAPI.fetchWithRetry(endpoint.path);
        const endTime = performance.now();
        
        tests[endpoint.name] = {
          status: 'success',
          responseTime: Math.round(endTime - startTime),
          dataSize: JSON.stringify(result).length,
          cached: result.cached || false,
          sample: JSON.stringify(result).substring(0, 100) + '...'
        };
      } catch (error) {
        tests[endpoint.name] = {
          status: 'error',
          error: error.message,
          responseTime: 0
        };
      }
    }

    setDebugInfo(info);
    setApiTests(tests);
    setLoading(false);
  };

  const clearCache = () => {
    robustAPI.clearCache();
    runDiagnostics();
  };

  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black bg-opacity-50 flex items-center justify-center p-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-4xl w-full max-h-screen overflow-y-auto">
        <div className="flex justify-between items-center p-6 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white">
            🔧 Dashboard Debug Panel
          </h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="p-6 space-y-6">
          {/* System Information */}
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
              📊 System Information
            </h3>
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 space-y-2">
              <div><strong>Timestamp:</strong> {debugInfo.timestamp}</div>
              <div><strong>Current URL:</strong> {debugInfo.url}</div>
              <div><strong>Fallback Mode:</strong> {debugInfo.fallbackMode ? '🟡 Yes' : '🟢 No'}</div>
              <div><strong>Cache Size:</strong> {debugInfo.cacheStats?.size || 0} entries</div>
            </div>
          </div>

          {/* API Endpoint Tests */}
          <div>
            <div className="flex justify-between items-center mb-3">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                🌐 API Endpoint Tests
              </h3>
              <div className="space-x-2">
                <button
                  onClick={runDiagnostics}
                  disabled={loading}
                  className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
                >
                  {loading ? 'Testing...' : 'Retest All'}
                </button>
                <button
                  onClick={clearCache}
                  className="px-3 py-1 bg-orange-500 text-white rounded hover:bg-orange-600"
                >
                  Clear Cache
                </button>
              </div>
            </div>

            <div className="space-y-3">
              {Object.entries(apiTests).map(([name, test]) => (
                <div
                  key={name}
                  className={`p-4 rounded-lg border ${
                    test.status === 'success' 
                      ? 'bg-green-50 border-green-200 dark:bg-green-900/20 dark:border-green-800'
                      : 'bg-red-50 border-red-200 dark:bg-red-900/20 dark:border-red-800'
                  }`}
                >
                  <div className="flex justify-between items-start">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2">
                        <span className={`w-3 h-3 rounded-full ${
                          test.status === 'success' ? 'bg-green-500' : 'bg-red-500'
                        }`}></span>
                        <strong className="text-gray-900 dark:text-white">{name}</strong>
                        {test.cached && (
                          <span className="text-xs bg-yellow-100 text-yellow-800 px-2 py-1 rounded">
                            CACHED
                          </span>
                        )}
                      </div>

                      {test.status === 'success' ? (
                        <div className="mt-2 text-sm text-gray-600 dark:text-gray-400 space-y-1">
                          <div>Response Time: {test.responseTime}ms</div>
                          <div>Data Size: {test.dataSize} bytes</div>
                          <details className="mt-2">
                            <summary className="cursor-pointer text-blue-600 dark:text-blue-400">
                              Show Sample Data
                            </summary>
                            <pre className="mt-2 text-xs bg-gray-100 dark:bg-gray-800 p-2 rounded overflow-x-auto">
                              {test.sample}
                            </pre>
                          </details>
                        </div>
                      ) : (
                        <div className="mt-2 text-sm text-red-600 dark:text-red-400">
                          Error: {test.error}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Common Solutions */}
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
              💡 Common Solutions
            </h3>
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 space-y-2 text-sm">
              <div><strong>If API endpoints are failing:</strong></div>
              <ul className="list-disc ml-6 space-y-1">
                <li>Check if API is running: <code>curl http://localhost:5001/health</code></li>
                <li>Verify API URL in .env file: <code>REACT_APP_API_URL=http://localhost:5001</code></li>
                <li>Check browser console for CORS errors</li>
                <li>Restart both API and dashboard</li>
              </ul>
              
              <div className="mt-4"><strong>If dashboard shows cached data:</strong></div>
              <ul className="list-disc ml-6 space-y-1">
                <li>Click "Clear Cache" button above</li>
                <li>Hard refresh browser (Ctrl+F5 / Cmd+Shift+R)</li>
                <li>Check network connectivity</li>
              </ul>
            </div>
          </div>

          {/* Cache Information */}
          {debugInfo.cacheStats && (
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
                💾 Cache Information
              </h3>
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <div><strong>Cached Endpoints:</strong></div>
                <ul className="mt-2 text-sm space-y-1">
                  {debugInfo.cacheStats.keys?.map((key, index) => (
                    <li key={index} className="font-mono text-xs">{key}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DebugPanel;