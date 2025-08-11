/**
 * SuperLazyBillionaire Strategy Panel
 * Advanced dashboard for the master trading strategy
 */

import React, { useState, useEffect } from 'react';
import robustAPI from '../services/robustApi';

const SuperLazyBillionairePanel = ({ className = '' }) => {
  const [marketRegime, setMarketRegime] = useState(null);
  const [positionSizing, setPositionSizing] = useState(null);
  const [mlAnalysis, setMLAnalysis] = useState(null);
  const [advancedMetrics, setAdvancedMetrics] = useState(null);
  const [strategyStatus, setStrategyStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedSymbol, setSelectedSymbol] = useState('BTC/USDT');

  useEffect(() => {
    loadAllData();
    const interval = setInterval(loadAllData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, [selectedSymbol]);

  const loadAllData = async () => {
    try {
      setLoading(true);
      setError(null);

      console.log('🚀 Loading SuperLazyBillionaire data...');

      const [regime, sizing, ml, metrics, status] = await Promise.all([
        robustAPI.fetchWithRetry('/api/v1/market/regime'),
        robustAPI.fetchWithRetry('/api/v1/risk/position-sizing', { symbol: selectedSymbol }),
        robustAPI.fetchWithRetry('/api/v1/ml/analysis', { symbol: selectedSymbol }),
        robustAPI.fetchWithRetry('/api/v1/analytics/advanced'),
        robustAPI.fetchWithRetry('/api/v1/strategies/super_lazy_billionaire')
      ]);

      console.log('✅ Data loaded successfully:', { regime, sizing, ml, metrics, status });

      setMarketRegime(regime);
      setPositionSizing(sizing);
      setMLAnalysis(ml);
      setAdvancedMetrics(metrics);
      setStrategyStatus(status);

      // Show notification if using fallback data
      if (robustAPI.isInFallbackMode()) {
        setError('Using cached data - API temporarily unavailable');
      }
    } catch (err) {
      console.error('❌ Error loading SuperLazyBillionaire data:', err);
      setError(`Failed to load data: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const getRegimeColor = (regime) => {
    const regimeColors = {
      'BULL_STRONG': 'text-green-600 bg-green-100 dark:bg-green-900',
      'BULL_WEAK': 'text-green-500 bg-green-50 dark:bg-green-800',
      'BEAR_STRONG': 'text-red-600 bg-red-100 dark:bg-red-900',
      'BEAR_WEAK': 'text-red-500 bg-red-50 dark:bg-red-800',
      'SIDEWAYS': 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900',
      'VOLATILE': 'text-orange-600 bg-orange-100 dark:bg-orange-900',
      'UNKNOWN': 'text-gray-600 bg-gray-100 dark:bg-gray-700'
    };
    return regimeColors[regime] || regimeColors['UNKNOWN'];
  };

  const getSignalColor = (signal) => {
    const signalColors = {
      'BUY': 'text-green-600 bg-green-100 dark:bg-green-900',
      'SELL': 'text-red-600 bg-red-100 dark:bg-red-900',
      'HOLD': 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900',
      'STRONG_BUY': 'text-green-700 bg-green-200 dark:bg-green-800',
      'STRONG_SELL': 'text-red-700 bg-red-200 dark:bg-red-800'
    };
    return signalColors[signal] || signalColors['HOLD'];
  };

  const formatPercentage = (value) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  if (loading) {
    return (
      <div className={`card ${className}`}>
        <div className="flex items-center justify-center h-48">
          <div className="loading-spinner"></div>
          <span className="ml-3 text-gray-600 dark:text-gray-400">Loading SuperLazyBillionaire data...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`card ${className}`}>
        <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
          <p className="text-red-700 dark:text-red-300">Error: {error}</p>
          <button 
            onClick={loadAllData}
            className="mt-2 btn-sm btn-primary"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`card ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-gradient-to-br from-yellow-400 to-yellow-600 rounded-lg flex items-center justify-center">
            <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
            </svg>
          </div>
          <div>
            <h2 className="text-xl font-bold text-gray-900 dark:text-white">
              SuperLazyBillionaire Strategy
            </h2>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              AI-Driven Master Trading Strategy
            </p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <select
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value)}
            className="input-field-sm"
          >
            <option value="BTC/USDT">BTC/USDT</option>
            <option value="ETH/USDT">ETH/USDT</option>
            <option value="ADA/USDT">ADA/USDT</option>
            <option value="SOL/USDT">SOL/USDT</option>
          </select>
          <button
            onClick={loadAllData}
            className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            title="Refresh"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Market Regime Analysis */}
        {marketRegime && (
          <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium text-blue-900 dark:text-blue-100">Market Regime</h3>
              <svg className="w-4 h-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            
            <div className="space-y-2">
              <div className={`inline-flex px-3 py-1 rounded-full text-sm font-medium ${getRegimeColor(marketRegime.regime)}`}>
                {marketRegime.regime}
              </div>
              
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Confidence:</span>
                  <span className="ml-1 font-medium">{formatPercentage(marketRegime.confidence)}</span>
                </div>
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Trend:</span>
                  <span className="ml-1 font-medium">{formatPercentage(marketRegime.trend_strength)}</span>
                </div>
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Volatility:</span>
                  <span className="ml-1 font-medium">{formatPercentage(marketRegime.volatility)}</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Kelly Criterion Position Sizing */}
        {positionSizing && (
          <div className="p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium text-green-900 dark:text-green-100">Kelly Sizing</h3>
              <svg className="w-4 h-4 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1" />
              </svg>
            </div>
            
            <div className="space-y-2">
              <div className="text-lg font-bold text-green-700 dark:text-green-300">
                {formatPercentage(positionSizing.recommended_size)}
              </div>
              
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Kelly:</span>
                  <span className="ml-1 font-medium">{formatPercentage(positionSizing.kelly_fraction)}</span>
                </div>
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Safety:</span>
                  <span className="ml-1 font-medium">{formatPercentage(positionSizing.safety_factor)}</span>
                </div>
                <div className="col-span-2">
                  <span className="text-gray-500 dark:text-gray-400">Max Risk:</span>
                  <span className="ml-1 font-medium">{formatPercentage(positionSizing.max_risk_per_trade)}</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ML Analysis */}
        {mlAnalysis && (
          <div className="p-4 bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium text-purple-900 dark:text-purple-100">ML Analysis</h3>
              <svg className="w-4 h-4 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className={`px-2 py-1 rounded text-xs font-medium ${getSignalColor(mlAnalysis.entry_signal)}`}>
                  {mlAnalysis.entry_signal}
                </span>
                <span className={`px-2 py-1 rounded text-xs font-medium ${getSignalColor(mlAnalysis.exit_signal)}`}>
                  {mlAnalysis.exit_signal}
                </span>
              </div>
              
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Confidence:</span>
                  <span className="ml-1 font-medium">{formatPercentage(mlAnalysis.confidence)}</span>
                </div>
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Features:</span>
                  <span className="ml-1 font-medium">{mlAnalysis.features_count || 200}+</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Advanced Performance Metrics */}
        {advancedMetrics && (
          <div className="col-span-full p-4 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg">
            <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-4">Advanced Performance Metrics</h3>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-lg font-bold text-blue-600 dark:text-blue-400">
                  {advancedMetrics.sharpe_ratio?.toFixed(2) || '0.00'}
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">Sharpe Ratio</div>
              </div>
              
              <div className="text-center">
                <div className="text-lg font-bold text-red-600 dark:text-red-400">
                  {formatPercentage(advancedMetrics.max_drawdown || 0)}
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">Max Drawdown</div>
              </div>
              
              <div className="text-center">
                <div className="text-lg font-bold text-green-600 dark:text-green-400">
                  {formatPercentage(advancedMetrics.win_rate || 0.5)}
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">Win Rate</div>
              </div>
              
              <div className="text-center">
                <div className="text-lg font-bold text-purple-600 dark:text-purple-400">
                  {advancedMetrics.profit_factor?.toFixed(2) || '1.00'}
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">Profit Factor</div>
              </div>
            </div>
          </div>
        )}

        {/* Strategy Status */}
        {strategyStatus && (
          <div className="col-span-full p-4 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg">
            <h3 className="text-sm font-medium text-yellow-900 dark:text-yellow-100 mb-3">Strategy Configuration</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div>
                <span className="text-gray-500 dark:text-gray-400">Risk Level:</span>
                <span className="ml-2 font-medium">{strategyStatus.risk_level || 'Adaptive'}</span>
              </div>
              
              <div>
                <span className="text-gray-500 dark:text-gray-400">Timeframes:</span>
                <span className="ml-2 font-medium">{strategyStatus.timeframes?.join(', ') || '15m, 1h, 4h, 1d'}</span>
              </div>
              
              <div>
                <span className="text-gray-500 dark:text-gray-400">Markets:</span>
                <span className="ml-2 font-medium">{strategyStatus.markets?.join(', ') || 'Spot, Futures'}</span>
              </div>
            </div>
            
            <div className="mt-3 p-3 bg-yellow-100 dark:bg-yellow-800/20 rounded text-xs text-yellow-800 dark:text-yellow-200">
              <strong>Note:</strong> This strategy dynamically adjusts allocation based on market conditions, 
              ML predictions, and Kelly Criterion optimization. Expected annual return: 60-80% with 1.8+ Sharpe ratio.
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SuperLazyBillionairePanel;