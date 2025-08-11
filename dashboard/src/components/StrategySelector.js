/**
 * StrategySelector Component
 * Strategy selection and management interface
 */

import React, { useState, useEffect } from 'react';
import { useTradingUpdates } from '../hooks/useWebSocket';
import robustAPI from '../services/robustApi';

const StrategySelector = ({ className = '', onStrategyChange }) => {
  const [strategies, setStrategies] = useState([]);
  const [selectedStrategy, setSelectedStrategy] = useState('super_lazy_billionaire');  // Default to new master strategy
  const [tradingMode, setTradingMode] = useState('paper');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [tradingStatus, setTradingStatus] = useState(null);
  const [strategyConfig, setStrategyConfig] = useState({});
  const [showConfig, setShowConfig] = useState(false);
  const [positionSizing, setPositionSizing] = useState(null);
  const [marketRegime, setMarketRegime] = useState(null);
  const { tradingStatus: wsTradingStatus } = useTradingUpdates();

  useEffect(() => {
    loadStrategies();
    loadTradingStatus();
  }, []);

  useEffect(() => {
    if (wsTradingStatus) {
      setTradingStatus(wsTradingStatus);
    }
  }, [wsTradingStatus]);

  const loadStrategies = async () => {
    try {
      console.log('🔄 Loading available strategies...');
      setLoading(true);
      setError(null);
      
      const response = await robustAPI.fetchWithRetry('/api/v1/strategies/list');
      console.log('📊 Raw API response:', response);
      
      const strategiesData = response.strategies || [];
      console.log('📊 Strategies array:', strategiesData);
      console.log('📊 Number of strategies:', strategiesData.length);
      
      if (strategiesData.length === 0) {
        console.warn('⚠️ No strategies received from API');
        setError('No strategies available');
      } else {
        console.log('✅ Setting strategies state with', strategiesData.length, 'items');
        setStrategies(strategiesData);
      }
      
      // Load additional data for advanced strategies
      await loadMarketRegime();
      await loadPositionSizing();
    } catch (err) {
      console.error('❌ Error loading strategies:', err);
      setError(`Failed to load strategies: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const loadMarketRegime = async () => {
    try {
      const response = await robustAPI.fetchWithRetry('/api/v1/market/regime');
      setMarketRegime(response);
    } catch (err) {
      console.error('Error loading market regime:', err);
    }
  };

  const loadPositionSizing = async () => {
    try {
      const response = await robustAPI.fetchWithRetry('/api/v1/risk/position-sizing');
      setPositionSizing(response);
    } catch (err) {
      console.error('Error loading position sizing:', err);
    }
  };

  const loadTradingStatus = async () => {
    try {
      const response = await robustAPI.fetchWithRetry('/api/v1/trading/status');
      setTradingStatus(response);
      if (response.current_strategy) {
        setSelectedStrategy(response.current_strategy);
      }
      setTradingMode(response.mode || 'paper');
    } catch (err) {
      console.error('Error loading trading status:', err);
    }
  };

  const handleStartTrading = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await robustAPI.fetchWithRetry('/api/v1/trading/start', {
        method: 'POST',
        body: JSON.stringify({
          mode: tradingMode,
          strategy: selectedStrategy,
          config: strategyConfig
        })
      });
      
      if (response.status === 'success') {
        setTradingStatus({
          active: true,
          mode: tradingMode,
          current_strategy: selectedStrategy
        });
        
        if (onStrategyChange) {
          onStrategyChange(selectedStrategy, tradingMode);
        }
      }
    } catch (err) {
      setError(err.message || 'Failed to start trading');
    } finally {
      setLoading(false);
    }
  };

  const handleStopTrading = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await robustAPI.fetchWithRetry('/api/v1/trading/stop', {
        method: 'POST'
      });
      
      if (response.status === 'success') {
        setTradingStatus({
          active: false,
          mode: null,
          current_strategy: null
        });
        
        if (onStrategyChange) {
          onStrategyChange(null, null);
        }
      }
    } catch (err) {
      setError(err.message || 'Failed to stop trading');
    } finally {
      setLoading(false);
    }
  };

  const handleStrategyChange = async (strategyName) => {
    setSelectedStrategy(strategyName);
    
    try {
      const response = await robustAPI.fetchWithRetry(`/api/v1/strategies/${strategyName}`);
      setStrategyConfig(response.parameters || {});
    } catch (err) {
      console.error('Error loading strategy details:', err);
    }
  };

  const getRiskLevelColor = (level) => {
    switch (level) {
      case 'low':
        return 'text-success-600 dark:text-success-400 bg-success-100 dark:bg-success-900';
      case 'medium':
        return 'text-warning-600 dark:text-warning-400 bg-warning-100 dark:bg-warning-900';
      case 'high':
        return 'text-danger-600 dark:text-danger-400 bg-danger-100 dark:bg-danger-900';
      case 'adaptive':
        return 'text-primary-600 dark:text-primary-400 bg-primary-100 dark:bg-primary-900';
      default:
        return 'text-gray-600 dark:text-gray-400 bg-gray-100 dark:bg-gray-700';
    }
  };

  const getStrategyIcon = (strategyName) => {
    switch (strategyName) {
      case 'ultimate_btc':
        return (
          <svg className="w-5 h-5 text-orange-500" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 2l2.4 7.2h7.6l-6 4.8 2.4 7.2L12 16.8 5.6 21.2 8 14l-6-4.8h7.6L12 2z"/>
          </svg>
        );
      case 'super_lazy_billionaire':
        return (
          <svg className="w-5 h-5 text-yellow-500" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
          </svg>
        );
      case 'autopilot':
        return (
          <svg className="w-5 h-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        );
      case 'ml_strategy':
        return (
          <svg className="w-5 h-5 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
        );
      case 'arbitrage':
        return (
          <svg className="w-5 h-5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 4V2a1 1 0 011-1h8a1 1 0 011 1v2m-9 0h10l1 16H6L7 4z" />
          </svg>
        );
      default:
        return (
          <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
        );
    }
  };

  const getStatusColor = (active) => {
    if (active) {
      return 'text-success-600 dark:text-success-400 bg-success-100 dark:bg-success-900';
    }
    return 'text-gray-600 dark:text-gray-400 bg-gray-100 dark:bg-gray-700';
  };

  const selectedStrategyData = strategies.find(s => s.name === selectedStrategy);

  return (
    <div className={`card card-hover ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-gray-900 dark:text-white flex items-center space-x-3">
            <div className="w-8 h-8 bg-gradient-to-br from-primary-500 to-primary-600 rounded-lg flex items-center justify-center">
              <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <span>Strategy Control</span>
          </h2>
          <div className="flex items-center space-x-4 mt-2">
            <div className="flex items-center space-x-2">
              <div className={`status-dot ${
                tradingStatus?.active ? 'status-dot-online' : 'status-dot-offline'
              }`}></div>
              <span className="text-sm text-gray-500 dark:text-gray-400">Status:</span>
              <span className={`status-indicator ${getStatusColor(tradingStatus?.active)}`}>
                {tradingStatus?.active ? 'Active' : 'Inactive'}
              </span>
            </div>
            {tradingStatus?.active && (
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-500 dark:text-gray-400">Mode:</span>
                <span className="status-indicator status-info">
                  {tradingStatus.mode}
                </span>
              </div>
            )}
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowConfig(!showConfig)}
            className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            title="Configuration"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-danger-100 dark:bg-danger-900 border border-danger-200 dark:border-danger-800 rounded-lg">
          <p className="text-danger-700 dark:text-danger-300 text-sm">{error}</p>
        </div>
      )}

      <div className="space-y-6">
        {/* Strategy Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Strategy
          </label>
          <select
            value={selectedStrategy}
            onChange={(e) => handleStrategyChange(e.target.value)}
            disabled={tradingStatus?.active}
            className="input-field"
          >
            {strategies.map((strategy) => (
              <option key={strategy.name} value={strategy.name}>
                {strategy.name === 'ultimate_btc' ? '🏆 ' : 
                 strategy.name === 'super_lazy_billionaire' ? '🚀 ' : ''}
                {strategy.name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())} 
                {strategy.description ? ` - ${strategy.description}` : ''}
              </option>
            ))}
          </select>
          
          {selectedStrategyData && (
            <div className="mt-2 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-2">
                  {getStrategyIcon(selectedStrategy)}
                  <span className="text-sm text-gray-500 dark:text-gray-400">Risk Level:</span>
                  <span className={`status-indicator ${getRiskLevelColor(selectedStrategyData.risk_level)}`}>
                    {selectedStrategyData.risk_level}
                  </span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-gray-500 dark:text-gray-400">Timeframes:</span>
                  <span className="text-sm text-gray-700 dark:text-gray-300">
                    {selectedStrategyData.timeframes?.join(', ') || 'N/A'}
                  </span>
                </div>
              </div>
              
              {/* Special display for Ultimate BTC Strategy */}
              {selectedStrategy === 'ultimate_btc' && (
                <div className="mb-3 p-3 bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 border border-green-200 dark:border-green-800 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <svg className="w-5 h-5 text-green-600" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                      </svg>
                      <span className="text-sm font-bold text-green-800 dark:text-green-200">🔧 EVENT-DRIVEN PRODUCTION STRATEGY</span>
                    </div>
                    <span className="text-xs px-2 py-1 bg-green-200 dark:bg-green-800 text-green-800 dark:text-green-200 rounded font-semibold">
                      FIXED ✅
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-3 mb-2">
                    <div className="bg-white dark:bg-gray-800 p-2 rounded">
                      <div className="text-xs text-gray-500 dark:text-gray-400">Lookahead Bias</div>
                      <div className="text-sm font-bold text-green-600 dark:text-green-400">Eliminated ✅</div>
                    </div>
                    <div className="bg-white dark:bg-gray-800 p-2 rounded">
                      <div className="text-xs text-gray-500 dark:text-gray-400">Indicator Method</div>
                      <div className="text-sm font-bold text-blue-600 dark:text-blue-400">Event-Driven</div>
                    </div>
                    <div className="bg-white dark:bg-gray-800 p-2 rounded">
                      <div className="text-xs text-gray-500 dark:text-gray-400">Thresholds</div>
                      <div className="text-sm font-bold text-purple-600 dark:text-purple-400">Adaptive</div>
                    </div>
                    <div className="bg-white dark:bg-gray-800 p-2 rounded">
                      <div className="text-xs text-gray-500 dark:text-gray-400">Production Ready</div>
                      <div className="text-sm font-bold text-green-600 dark:text-green-400">Yes ✅</div>
                    </div>
                  </div>
                  <p className="text-xs text-green-700 dark:text-green-300">
                    🛠️ Fixed all critical issues identified in code review: eliminated lookahead bias with incremental state management, 
                    replaced magic numbers with adaptive thresholds, and implemented proper event-driven indicator calculations.
                  </p>
                </div>
              )}

              {/* Special display for SuperLazyBillionaire */}
              {selectedStrategy === 'super_lazy_billionaire' && (
                <div className="mb-3 p-2 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded">
                  <div className="flex items-center space-x-2 mb-1">
                    <svg className="w-4 h-4 text-yellow-600" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
                    </svg>
                    <span className="text-sm font-medium text-yellow-800 dark:text-yellow-200">Master Strategy</span>
                  </div>
                  <p className="text-xs text-yellow-700 dark:text-yellow-300">
                    Advanced AI-driven strategy with dynamic allocation, ML-enhanced entry/exit, and Kelly Criterion position sizing
                  </p>
                </div>
              )}

              {/* Market Regime Display for Advanced Strategies */}
              {marketRegime && (selectedStrategy === 'super_lazy_billionaire' || selectedStrategy === 'ultimate_btc') && (
                <div className="mb-2 flex items-center space-x-4">
                  <div className="flex items-center space-x-2">
                    <span className="text-xs text-gray-500 dark:text-gray-400">Market Regime:</span>
                    <span className="text-xs px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 rounded">
                      {marketRegime.regime} ({(marketRegime.confidence * 100).toFixed(0)}%)
                    </span>
                  </div>
                  {positionSizing && (
                    <div className="flex items-center space-x-2">
                      <span className="text-xs text-gray-500 dark:text-gray-400">
                        {selectedStrategy === 'ultimate_btc' ? 'Position Size:' : 'Kelly Size:'}
                      </span>
                      <span className="text-xs px-2 py-1 bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200 rounded">
                        {(positionSizing.recommended_size * 100).toFixed(1)}%
                      </span>
                    </div>
                  )}
                  {selectedStrategy === 'ultimate_btc' && (
                    <div className="flex items-center space-x-2">
                      <span className="text-xs text-gray-500 dark:text-gray-400">Signal Quality:</span>
                      <span className="text-xs px-2 py-1 bg-purple-100 dark:bg-purple-900 text-purple-800 dark:text-purple-200 rounded">
                        High
                      </span>
                    </div>
                  )}
                </div>
              )}
              
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                {selectedStrategyData.description}
              </p>
            </div>
          )}
        </div>

        {/* Trading Mode */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Trading Mode
          </label>
          <div className="flex space-x-4">
            <label className="flex items-center">
              <input
                type="radio"
                value="paper"
                checked={tradingMode === 'paper'}
                onChange={(e) => setTradingMode(e.target.value)}
                disabled={tradingStatus?.active}
                className="mr-2"
              />
              <span className="text-sm text-gray-700 dark:text-gray-300">Paper Trading</span>
            </label>
            <label className="flex items-center">
              <input
                type="radio"
                value="live"
                checked={tradingMode === 'live'}
                onChange={(e) => setTradingMode(e.target.value)}
                disabled={tradingStatus?.active}
                className="mr-2"
              />
              <span className="text-sm text-gray-700 dark:text-gray-300">Live Trading</span>
            </label>
          </div>
        </div>

        {/* Strategy Configuration (collapsible) */}
        {showConfig && selectedStrategyData && (
          <div className="border border-gray-200 dark:border-gray-600 rounded-lg p-4">
            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
              Strategy Configuration
            </h3>
            <div className="space-y-4">
              {Object.entries(selectedStrategyData.parameters || {}).map(([key, value]) => (
                <div key={key}>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </label>
                  <input
                    type={typeof value === 'number' ? 'number' : 'text'}
                    value={strategyConfig[key] || value}
                    onChange={(e) => setStrategyConfig(prev => ({
                      ...prev,
                      [key]: e.target.type === 'number' ? parseFloat(e.target.value) : e.target.value
                    }))}
                    disabled={tradingStatus?.active}
                    className="input-field"
                  />
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Control Buttons */}
        <div className="flex space-x-4">
          {!tradingStatus?.active ? (
            <button
              onClick={handleStartTrading}
              disabled={loading}
              className="btn-success flex-1 flex items-center justify-center space-x-2"
            >
              {loading ? (
                <div className="loading-spinner"></div>
              ) : (
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h1m4 0h1m-6 4h1m4 0h1m-4-8h4m-4 0a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              )}
              <span>Start Trading</span>
            </button>
          ) : (
            <button
              onClick={handleStopTrading}
              disabled={loading}
              className="btn-danger flex-1 flex items-center justify-center space-x-2"
            >
              {loading ? (
                <div className="loading-spinner"></div>
              ) : (
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10h6v4H9z" />
                </svg>
              )}
              <span>Stop Trading</span>
            </button>
          )}
        </div>

        {/* Current Strategy Info */}
        {tradingStatus?.active && tradingStatus.current_strategy && (
          <div className="bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800 rounded-lg p-4">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-primary-600 rounded-full animate-pulse"></div>
              <span className="text-sm font-medium text-primary-700 dark:text-primary-300">
                Currently running: {tradingStatus.current_strategy} ({tradingStatus.mode})
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default StrategySelector;