import React, { useState, useEffect } from 'react';
import {
  CpuChipIcon,
  ChartBarIcon,
  ArrowPathIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon,
  CurrencyDollarIcon,
  ArrowTrendingUpIcon
} from '@heroicons/react/24/outline';

const DecisionLog = ({ maxEntries = 50 }) => {
  const [decisions, setDecisions] = useState([]);
  const [autoScroll, setAutoScroll] = useState(true);

  // Demo decision log entries
  const demoDecisions = [
    {
      id: 1,
      timestamp: new Date(Date.now() - 2 * 60 * 1000),
      type: 'market_shift',
      icon: ChartBarIcon,
      title: 'Market shift detected',
      message: 'Volatility increased to HIGH (75/100)',
      action: 'Reducing mean_reversion allocation from 30% to 15%',
      severity: 'warning',
      strategy: 'mean_reversion'
    },
    {
      id: 2,
      timestamp: new Date(Date.now() - 3 * 60 * 1000),
      type: 'rebalance',
      icon: ArrowPathIcon,
      title: 'Portfolio rebalancing',
      message: 'Risk-adjusted allocation update',
      action: 'Increasing high_risk_daily allocation to 15%',
      severity: 'info',
      strategy: 'high_risk_daily'
    },
    {
      id: 3,
      timestamp: new Date(Date.now() - 5 * 60 * 1000),
      type: 'position',
      icon: CurrencyDollarIcon,
      title: 'New position opened',
      message: 'Strong momentum signal detected',
      action: 'BTC/USDT LONG @ $45,750 via momentum_strategy',
      severity: 'success',
      strategy: 'momentum_strategy'
    },
    {
      id: 4,
      timestamp: new Date(Date.now() - 8 * 60 * 1000),
      type: 'health_alert',
      icon: ExclamationTriangleIcon,
      title: 'Strategy health alert',
      message: 'Win rate dropping below threshold',
      action: 'ml_strategy win rate: 45% (threshold: 50%)',
      severity: 'warning',
      strategy: 'ml_strategy'
    },
    {
      id: 5,
      timestamp: new Date(Date.now() - 12 * 60 * 1000),
      type: 'opportunity',
      icon: ArrowTrendingUpIcon,
      title: 'High-confidence opportunity',
      message: 'Multiple strategies converging on signal',
      action: 'SOL/USDT setup rated 9.2/10 confidence',
      severity: 'success',
      strategy: 'multiple'
    },
    {
      id: 6,
      timestamp: new Date(Date.now() - 15 * 60 * 1000),
      type: 'risk_check',
      icon: ExclamationTriangleIcon,
      title: 'Risk limit check',
      message: 'Daily loss approaching 60% of limit',
      action: 'Reducing position sizes by 25%',
      severity: 'error',
      strategy: 'risk_manager'
    },
    {
      id: 7,
      timestamp: new Date(Date.now() - 18 * 60 * 1000),
      type: 'learning',
      icon: CpuChipIcon,
      title: 'Pattern recognition',
      message: 'New market pattern identified',
      action: 'Volume spike + RSI divergence = 82% win rate',
      severity: 'info',
      strategy: 'orchestrator'
    },
    {
      id: 8,
      timestamp: new Date(Date.now() - 22 * 60 * 1000),
      type: 'position_close',
      icon: CheckCircleIcon,
      title: 'Position closed',
      message: 'Take profit target reached',
      action: 'ETH/USDT LONG closed @ +3.4% profit',
      severity: 'success',
      strategy: 'candle_momentum'
    }
  ];

  useEffect(() => {
    setDecisions(demoDecisions);

    // Simulate new decisions coming in
    const interval = setInterval(() => {
      const newDecision = {
        id: Date.now(),
        timestamp: new Date(),
        type: ['market_shift', 'rebalance', 'position', 'health_alert', 'opportunity'][Math.floor(Math.random() * 5)],
        icon: [ChartBarIcon, ArrowPathIcon, CurrencyDollarIcon, ExclamationTriangleIcon, ArrowTrendingUpIcon][Math.floor(Math.random() * 5)],
        title: 'Real-time update',
        message: 'Live market data processed',
        action: `${Math.random() > 0.5 ? 'BTC/USDT' : 'ETH/USDT'} signal analyzed`,
        severity: ['info', 'success', 'warning'][Math.floor(Math.random() * 3)],
        strategy: ['momentum_strategy', 'mean_reversion', 'arbitrage'][Math.floor(Math.random() * 3)]
      };

      setDecisions(prev => [newDecision, ...prev.slice(0, maxEntries - 1)]);
    }, 30000); // New decision every 30 seconds

    return () => clearInterval(interval);
  }, [maxEntries]);

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'success':
        return 'bg-green-100 border-green-200 text-green-800 dark:bg-green-900 dark:border-green-800 dark:text-green-200';
      case 'warning':
        return 'bg-yellow-100 border-yellow-200 text-yellow-800 dark:bg-yellow-900 dark:border-yellow-800 dark:text-yellow-200';
      case 'error':
        return 'bg-red-100 border-red-200 text-red-800 dark:bg-red-900 dark:border-red-800 dark:text-red-200';
      default:
        return 'bg-blue-100 border-blue-200 text-blue-800 dark:bg-blue-900 dark:border-blue-800 dark:text-blue-200';
    }
  };

  const getIconColor = (severity) => {
    switch (severity) {
      case 'success':
        return 'text-green-600';
      case 'warning':
        return 'text-yellow-600';
      case 'error':
        return 'text-red-600';
      default:
        return 'text-blue-600';
    }
  };

  const formatTimestamp = (timestamp) => {
    const now = new Date();
    const diff = now - timestamp;
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(minutes / 60);
    
    if (hours > 0) {
      return `${hours}h ${minutes % 60}m ago`;
    } else if (minutes > 0) {
      return `${minutes}m ago`;
    } else {
      return 'Just now';
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
      <div className="flex justify-between items-center mb-4">
        <div className="flex items-center">
          <CpuChipIcon className="h-6 w-6 text-purple-600 mr-2" />
          <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">
            Orchestrator Decision Log
          </h2>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setAutoScroll(!autoScroll)}
            className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
              autoScroll 
                ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                : 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
            }`}
          >
            Auto-scroll {autoScroll ? 'ON' : 'OFF'}
          </button>
          <span className="text-sm text-gray-500 dark:text-gray-400">
            {decisions.length} entries
          </span>
        </div>
      </div>

      <div className="space-y-3 max-h-96 overflow-y-auto">
        {decisions.map((decision) => {
          const IconComponent = decision.icon;
          return (
            <div
              key={decision.id}
              className={`border rounded-lg p-3 ${getSeverityColor(decision.severity)}`}
            >
              <div className="flex items-start space-x-3">
                <div className="flex-shrink-0">
                  <IconComponent className={`h-5 w-5 ${getIconColor(decision.severity)}`} />
                </div>
                <div className="flex-grow min-w-0">
                  <div className="flex items-center justify-between">
                    <p className="text-sm font-semibold truncate">
                      {decision.title}
                    </p>
                    <div className="flex items-center space-x-2">
                      <span className="text-xs font-medium bg-white bg-opacity-50 px-2 py-1 rounded">
                        {decision.strategy}
                      </span>
                      <span className="text-xs text-gray-600 dark:text-gray-300">
                        {formatTimestamp(decision.timestamp)}
                      </span>
                    </div>
                  </div>
                  <p className="text-sm mt-1 opacity-90">
                    {decision.message}
                  </p>
                  <p className="text-sm font-medium mt-2 bg-white bg-opacity-30 rounded p-2">
                    → {decision.action}
                  </p>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Quick Stats */}
      <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
        <div className="grid grid-cols-4 gap-4 text-center">
          <div>
            <p className="text-xs text-gray-500 dark:text-gray-400">Today's Decisions</p>
            <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              {decisions.filter(d => {
                const today = new Date();
                return d.timestamp.toDateString() === today.toDateString();
              }).length}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-500 dark:text-gray-400">Rebalances</p>
            <p className="text-lg font-semibold text-blue-600">
              {decisions.filter(d => d.type === 'rebalance').length}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-500 dark:text-gray-400">Positions</p>
            <p className="text-lg font-semibold text-green-600">
              {decisions.filter(d => d.type === 'position' || d.type === 'position_close').length}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-500 dark:text-gray-400">Alerts</p>
            <p className="text-lg font-semibold text-yellow-600">
              {decisions.filter(d => d.type === 'health_alert' || d.severity === 'warning').length}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DecisionLog;