import React, { useState, useEffect } from 'react';
import robustAPI from '../services/robustApi';
import StrategyOverviewPanel from './StrategyOverviewPanel';
import PerformanceMetricsGrid from './PerformanceMetricsGrid';
import LivePositionTracker from './LivePositionTracker';
import DecisionLog from './DecisionLog';
import CapitalAllocationChart from './CapitalAllocationChart';
import RiskManagementPanel from './RiskManagementPanel';
import { 
  ChartBarIcon, 
  CogIcon, 
  HeartIcon,
  ExclamationTriangleIcon,
  CurrencyDollarIcon,
  ArrowTrendingUpIcon,
  BeakerIcon,
  ArrowPathIcon,
  PlayIcon,
  StopIcon,
  PowerIcon
} from '@heroicons/react/24/outline';

const OrchestratorDashboard = () => {
  const [orchestratorData, setOrchestratorData] = useState({
    status: 'stopped',
    mode: 'paper',
    discoveredStrategies: 8,
    portfolio: {
      total_value: 10523.45,
      cash_balance: 5234.12,
      positions_value: 5289.33,
      total_pnl: 523.45,
      total_pnl_percent: 5.23,
      win_rate: 0.67,
      sharpe_ratio: 1.34,
      max_drawdown: 0.08,
      total_positions: 8
    },
    strategies: [
      {
        name: 'momentum_strategy',
        risk_level: 'moderate',
        timeframe: 'intraday',
        signal_sources: ['technical', 'volume', 'momentum'],
        expected_win_rate: 0.65,
        sharpe_estimate: 1.4,
        cooperation_score: 7.5,
        conflict_strategies: ['mean_reversion'],
        code_metrics: {
          total_lines: 245,
          complexity_score: 6.7,
          dependencies: ['pandas', 'numpy', 'talib']
        }
      },
      {
        name: 'mean_reversion',
        risk_level: 'conservative',
        timeframe: 'swing',
        signal_sources: ['technical', 'statistical'],
        expected_win_rate: 0.58,
        sharpe_estimate: 1.1,
        cooperation_score: 8.2,
        conflict_strategies: ['momentum_strategy'],
        code_metrics: {
          total_lines: 189,
          complexity_score: 4.3,
          dependencies: ['pandas', 'numpy', 'scipy']
        }
      },
      {
        name: 'arbitrage',
        risk_level: 'conservative',
        timeframe: 'scalping',
        signal_sources: ['price_difference', 'exchange_data'],
        expected_win_rate: 0.82,
        sharpe_estimate: 2.1,
        cooperation_score: 9.1,
        conflict_strategies: [],
        code_metrics: {
          total_lines: 156,
          complexity_score: 3.8,
          dependencies: ['pandas', 'numpy', 'ccxt']
        }
      },
      {
        name: 'high_risk_daily',
        risk_level: 'extreme',
        timeframe: 'intraday',
        signal_sources: ['volume_spike', 'social_sentiment', 'technical'],
        expected_win_rate: 0.55,
        sharpe_estimate: 0.9,
        cooperation_score: 4.2,
        conflict_strategies: ['mean_reversion', 'conservative_long'],
        code_metrics: {
          total_lines: 312,
          complexity_score: 8.9,
          dependencies: ['pandas', 'numpy', 'tweepy', 'textblob']
        }
      },
      {
        name: 'grid_trading',
        risk_level: 'moderate',
        timeframe: 'position',
        signal_sources: ['range_detection', 'volatility'],
        expected_win_rate: 0.72,
        sharpe_estimate: 1.6,
        cooperation_score: 8.7,
        conflict_strategies: ['trend_following'],
        code_metrics: {
          total_lines: 278,
          complexity_score: 7.1,
          dependencies: ['pandas', 'numpy']
        }
      },
      {
        name: 'defi_yield',
        risk_level: 'aggressive',
        timeframe: 'position',
        signal_sources: ['yield_rates', 'liquidity_pool', 'gas_prices'],
        expected_win_rate: 0.61,
        sharpe_estimate: 1.3,
        cooperation_score: 6.8,
        conflict_strategies: [],
        code_metrics: {
          total_lines: 423,
          complexity_score: 9.2,
          dependencies: ['web3', 'pandas', 'numpy', 'requests']
        }
      },
      {
        name: 'copy_trading',
        risk_level: 'moderate',
        timeframe: 'intraday',
        signal_sources: ['social_trading', 'leader_performance'],
        expected_win_rate: 0.63,
        sharpe_estimate: 1.2,
        cooperation_score: 7.9,
        conflict_strategies: [],
        code_metrics: {
          total_lines: 201,
          complexity_score: 5.4,
          dependencies: ['pandas', 'numpy', 'requests']
        }
      },
      {
        name: 'liquidation',
        risk_level: 'aggressive',
        timeframe: 'scalping',
        signal_sources: ['liquidation_data', 'funding_rates'],
        expected_win_rate: 0.59,
        sharpe_estimate: 1.8,
        cooperation_score: 5.6,
        conflict_strategies: ['conservative_long'],
        code_metrics: {
          total_lines: 167,
          complexity_score: 6.2,
          dependencies: ['pandas', 'numpy', 'websocket']
        }
      }
    ],
    healthMetrics: {},
    marketAnalysis: {
      market_regime: 'bull',
      volatility_level: 'moderate',
      analysis_time: new Date().toISOString()
    },
    abTests: {
      total_active_tests: 2,
      completed_tests: 15
    }
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedStrategy, setSelectedStrategy] = useState(null);
  const [isStarting, setIsStarting] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [selectedStartMode, setSelectedStartMode] = useState('paper');

  useEffect(() => {
    // Simulate data loading
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
    }, 1000);
  }, []);

  // Separate useEffect for real-time updates based on orchestrator status
  useEffect(() => {
    let interval;
    if (orchestratorData.status === 'active') {
      console.log(`🔄 Starting real-time updates in ${orchestratorData.mode} mode`);
      interval = setInterval(() => {
        setOrchestratorData(prev => ({
          ...prev,
          portfolio: {
            ...prev.portfolio,
            total_value: prev.portfolio.total_value + (Math.random() - 0.5) * 100,
            total_pnl: prev.portfolio.total_pnl + (Math.random() - 0.5) * 50,
          }
        }));
      }, 5000); // Update every 5 seconds for demo
    } else {
      console.log('⏸️ Real-time updates stopped');
    }

    return () => {
      if (interval) {
        clearInterval(interval);
        console.log('🛑 Cleaned up real-time updates');
      }
    };
  }, [orchestratorData.status]); // Trigger when status changes

  const switchMode = async (newMode) => {
    try {
      // Demo mode - simulate mode switch
      await new Promise(resolve => setTimeout(resolve, 500));
      setOrchestratorData(prev => ({ ...prev, mode: newMode }));
      console.log(`🔄 Switched to ${newMode} mode`);
    } catch (err) {
      setError(`Failed to switch mode: ${err.message}`);
    }
  };

  const startOrchestrator = async () => {
    setIsStarting(true);
    setError(null);
    
    try {
      // Demo mode - simulate orchestrator startup
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      setOrchestratorData(prev => ({
        ...prev,
        status: 'active',
        mode: selectedStartMode
      }));
      
      console.log(`✅ Orchestrator started in ${selectedStartMode} mode`);
    } catch (err) {
      setError(`Failed to start orchestrator: ${err.message}`);
    } finally {
      setIsStarting(false);
    }
  };

  const stopOrchestrator = async () => {
    setIsStopping(true);
    setError(null);
    
    try {
      // Demo mode - simulate orchestrator shutdown
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setOrchestratorData(prev => ({
        ...prev,
        status: 'stopped'
      }));
      
      console.log('🛑 Orchestrator stopped');
    } catch (err) {
      setError(`Failed to stop orchestrator: ${err.message}`);
    } finally {
      setIsStopping(false);
    }
  };

  const getModeColor = (mode) => {
    switch (mode) {
      case 'paper': return 'text-blue-600 bg-blue-100';
      case 'live': return 'text-green-600 bg-green-100';
      case 'hybrid': return 'text-purple-600 bg-purple-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getRegimeColor = (regime) => {
    switch (regime?.toLowerCase()) {
      case 'bull': return 'text-green-600 bg-green-100';
      case 'bear': return 'text-red-600 bg-red-100';
      case 'volatile': return 'text-orange-600 bg-orange-100';
      case 'crash': return 'text-red-800 bg-red-200';
      case 'euphoria': return 'text-yellow-600 bg-yellow-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getRiskLevelColor = (riskLevel) => {
    switch (riskLevel?.toLowerCase()) {
      case 'conservative': return 'text-green-600 bg-green-100';
      case 'moderate': return 'text-blue-600 bg-blue-100';
      case 'aggressive': return 'text-orange-600 bg-orange-100';
      case 'extreme': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getStatusColor = (status) => {
    switch (status?.toLowerCase()) {
      case 'active': return 'text-green-600 bg-green-100';
      case 'stopped': return 'text-red-600 bg-red-100';
      case 'starting': return 'text-yellow-600 bg-yellow-100';
      case 'stopping': return 'text-orange-600 bg-orange-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  if (loading) {
    return (
      <div className="p-6">
        <div className="flex items-center justify-center h-64">
          <ArrowPathIcon className="h-8 w-8 animate-spin text-blue-500" />
          <span className="ml-2 text-lg">Loading Orchestrator...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">Strategy Orchestrator</h1>
          <p className="text-gray-600 dark:text-gray-400">Self-discovering intelligent strategy management</p>
        </div>
        
        <div className="flex items-center space-x-4">
          {/* Demo Mode Indicator */}
          <div className="px-3 py-1 rounded-full text-sm bg-blue-100 text-blue-800">
            Demo Mode
          </div>
          
          {/* Status */}
          <div className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(orchestratorData.status)}`}>
            {orchestratorData.status.toUpperCase()}
          </div>
          
          {/* Trading Mode */}
          {orchestratorData.status === 'active' && (
            <div className={`px-3 py-1 rounded-full text-sm font-medium ${getModeColor(orchestratorData.mode)}`}>
              {orchestratorData.mode.toUpperCase()} MODE
            </div>
          )}
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-md p-4">
          <div className="flex">
            <ExclamationTriangleIcon className="h-5 w-5 text-red-400" />
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">Error</h3>
              <p className="text-sm text-red-700 mt-1">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Orchestrator Controls */}
      {orchestratorData.status === 'stopped' && (
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">Start Strategy Orchestrator</h2>
          <div className="flex items-center space-x-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Trading Mode
              </label>
              <select
                value={selectedStartMode}
                onChange={(e) => setSelectedStartMode(e.target.value)}
                className="block w-32 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="paper">Paper Trading</option>
                <option value="live">Live Trading</option>
                <option value="hybrid">Hybrid Mode</option>
              </select>
            </div>
            <div className="pt-6">
              <button
                onClick={startOrchestrator}
                disabled={isStarting}
                className="flex items-center px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white rounded-md font-medium transition-colors"
              >
                {isStarting ? (
                  <>
                    <ArrowPathIcon className="h-4 w-4 mr-2 animate-spin" />
                    Starting...
                  </>
                ) : (
                  <>
                    <PlayIcon className="h-4 w-4 mr-2" />
                    Start Orchestrator
                  </>
                )}
              </button>
            </div>
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-3">
            Der Orchestrator wird alle Strategien automatisch entdecken und intelligent verwalten.
          </p>
        </div>
      )}
      
      {orchestratorData.status === 'active' && (
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">🎯 Orchestrator Active</h2>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Running in <span className={`font-semibold ${getModeColor(orchestratorData.mode)} px-2 py-1 rounded`}>
                  {orchestratorData.mode.toUpperCase()}
                </span> mode
                {orchestratorData.mode === 'paper' && (
                  <span className="ml-2 text-blue-600">📊 Demo-Handel (kein echtes Geld)</span>
                )}
              </p>
            </div>
            <button
              onClick={stopOrchestrator}
              disabled={isStopping}
              className="flex items-center px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-400 text-white rounded-md font-medium transition-colors"
            >
              {isStopping ? (
                <>
                  <ArrowPathIcon className="h-4 w-4 mr-2 animate-spin" />
                  Stopping...
                </>
              ) : (
                <>
                  <StopIcon className="h-4 w-4 mr-2" />
                  Stop Orchestrator
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* Overview Cards */}
      {orchestratorData.status === 'active' && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <div className="flex items-center">
            <CogIcon className="h-8 w-8 text-blue-500" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Discovered Strategies</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-gray-100">{orchestratorData.discoveredStrategies}</p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <div className="flex items-center">
            <CurrencyDollarIcon className="h-8 w-8 text-green-500" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">
                Portfolio Value {orchestratorData.mode === 'paper' && <span className="text-blue-600">(Demo)</span>}
              </p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-gray-100">${orchestratorData.portfolio.total_value?.toLocaleString()}</p>
              <p className={`text-sm ${orchestratorData.portfolio.total_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {orchestratorData.portfolio.total_pnl >= 0 ? '+' : ''}${orchestratorData.portfolio.total_pnl?.toFixed(2)} 
                ({orchestratorData.portfolio.total_pnl_percent >= 0 ? '+' : ''}{orchestratorData.portfolio.total_pnl_percent?.toFixed(1)}%)
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <div className="flex items-center">
            <ArrowTrendingUpIcon className="h-8 w-8 text-purple-500" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Win Rate</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-gray-100">{(orchestratorData.portfolio.win_rate * 100)?.toFixed(1)}%</p>
              <p className="text-sm text-gray-500 dark:text-gray-400">Sharpe: {orchestratorData.portfolio.sharpe_ratio?.toFixed(2)}</p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <div className="flex items-center">
            <BeakerIcon className="h-8 w-8 text-orange-500" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">A/B Tests</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-gray-100">{orchestratorData.abTests.total_active_tests}</p>
              <p className="text-sm text-gray-500 dark:text-gray-400">{orchestratorData.abTests.completed_tests} completed</p>
            </div>
          </div>
        </div>
        </div>
      )}

      {/* Market Analysis */}
      {orchestratorData.status === 'active' && (
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">Market Analysis</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Market Regime</p>
            <span className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${getRegimeColor(orchestratorData.marketAnalysis.market_regime)}`}>
              {orchestratorData.marketAnalysis.market_regime?.toUpperCase() || 'UNKNOWN'}
            </span>
          </div>
          <div>
            <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Volatility</p>
            <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">{orchestratorData.marketAnalysis.volatility_level?.toUpperCase()}</p>
          </div>
          <div>
            <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Last Update</p>
            <p className="text-sm text-gray-600 dark:text-gray-400">{new Date(orchestratorData.marketAnalysis.analysis_time || Date.now()).toLocaleTimeString()}</p>
          </div>
        </div>
        </div>
      )}

      {/* Advanced Dashboard Components - Only when Active */}
      {orchestratorData.status === 'active' && (
        <div className="space-y-6">
          {/* Strategy Overview Panel */}
          <StrategyOverviewPanel strategies={orchestratorData.strategies} />
          
          {/* Performance Metrics Grid and Capital Allocation */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <PerformanceMetricsGrid metrics={orchestratorData.portfolio} />
            <CapitalAllocationChart allocations={{
              momentum_strategy: 25,
              mean_reversion: 20,
              arbitrage: 15,
              high_risk_daily: 10,
              candle_momentum: 30
            }} />
          </div>
          
          {/* Live Position Tracker */}
          <LivePositionTracker />
          
          {/* Risk Management Panel */}
          <RiskManagementPanel riskData={{
            dailyLossLimit: 500,
            dailyLossUsed: 30,
            openRisk: 127,
            maxPositionSize: 100,
            correlationRisk: 'Low',
            emergencyStopArmed: true,
            drawdownLimit: 0.15,
            currentDrawdown: 0.032,
            riskPerTrade: 0.02
          }} />
          
          {/* Decision Log */}
          <DecisionLog maxEntries={50} />
          
          {/* Mode Switch Panel */}
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">Trading Mode Control</h3>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Current Mode: 
                  <span className={`ml-2 px-3 py-1 rounded-full text-sm font-medium ${getModeColor(orchestratorData.mode)}`}>
                    {orchestratorData.mode.toUpperCase()}
                  </span>
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400">Switch between Paper Trading, Live Trading, or Hybrid mode</p>
              </div>
              <div className="flex space-x-2">
                {['paper', 'live', 'hybrid'].map(mode => (
                  <button
                    key={mode}
                    onClick={() => switchMode(mode)}
                    disabled={orchestratorData.mode === mode}
                    className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                      orchestratorData.mode === mode
                        ? getModeColor(mode) + ' cursor-not-allowed'
                        : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-600'
                    }`}
                  >
                    {mode.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Strategy Details Modal */}
      {selectedStrategy && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg max-w-2xl w-full mx-4 max-h-96 overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">{selectedStrategy.name}</h3>
              <button
                onClick={() => setSelectedStrategy(null)}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
              >
                ✕
              </button>
            </div>
            
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Risk Level</p>
                  <span className={`px-2 py-1 rounded text-sm ${getRiskLevelColor(selectedStrategy.risk_level)}`}>
                    {selectedStrategy.risk_level}
                  </span>
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Timeframe</p>
                  <p className="font-medium text-gray-900 dark:text-gray-100">{selectedStrategy.timeframe}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Expected Win Rate</p>
                  <p className="font-medium text-gray-900 dark:text-gray-100">{(selectedStrategy.expected_win_rate * 100).toFixed(1)}%</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Cooperation Score</p>
                  <p className="font-medium text-gray-900 dark:text-gray-100">{selectedStrategy.cooperation_score}/10</p>
                </div>
              </div>
              
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Signal Sources</p>
                <div className="flex flex-wrap gap-1">
                  {selectedStrategy.signal_sources?.map((source, idx) => (
                    <span key={idx} className="px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 text-xs rounded">
                      {source}
                    </span>
                  ))}
                </div>
              </div>

              {selectedStrategy.conflict_strategies?.length > 0 && (
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Conflicts With</p>
                  <div className="flex flex-wrap gap-1">
                    {selectedStrategy.conflict_strategies.map((conflict, idx) => (
                      <span key={idx} className="px-2 py-1 bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 text-xs rounded">
                        {conflict}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Code Metrics</p>
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="text-gray-500 dark:text-gray-400">Lines:</span>
                    <span className="ml-1 font-medium text-gray-900 dark:text-gray-100">{selectedStrategy.code_metrics?.total_lines}</span>
                  </div>
                  <div>
                    <span className="text-gray-500 dark:text-gray-400">Complexity:</span>
                    <span className="ml-1 font-medium text-gray-900 dark:text-gray-100">{selectedStrategy.code_metrics?.complexity_score}</span>
                  </div>
                  <div>
                    <span className="text-gray-500 dark:text-gray-400">Dependencies:</span>
                    <span className="ml-1 font-medium text-gray-900 dark:text-gray-100">{selectedStrategy.code_metrics?.dependencies?.length}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default OrchestratorDashboard;