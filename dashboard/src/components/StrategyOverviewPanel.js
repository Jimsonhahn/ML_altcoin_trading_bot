import React from 'react';
import {
  CheckCircleIcon,
  XCircleIcon,
  ExclamationCircleIcon,
  PauseCircleIcon
} from '@heroicons/react/24/solid';

const StrategyOverviewPanel = ({ strategies = [] }) => {
  const getStatusIcon = (status) => {
    switch (status?.toLowerCase()) {
      case 'active':
        return <CheckCircleIcon className="h-5 w-5 text-green-500" />;
      case 'error':
        return <XCircleIcon className="h-5 w-5 text-red-500" />;
      case 'warning':
        return <ExclamationCircleIcon className="h-5 w-5 text-yellow-500" />;
      case 'paused':
        return <PauseCircleIcon className="h-5 w-5 text-gray-500" />;
      default:
        return <CheckCircleIcon className="h-5 w-5 text-gray-400" />;
    }
  };

  const getHealthColor = (score) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-yellow-600';
    if (score >= 40) return 'text-orange-600';
    return 'text-red-600';
  };

  const getPnLColor = (pnl) => {
    return pnl >= 0 ? 'text-green-600' : 'text-red-600';
  };

  // Demo data if no strategies provided
  const demoStrategies = [
    {
      name: 'momentum_strategy',
      displayName: 'Momentum',
      status: 'active',
      allocation: 25,
      currentPnL: 3.4,
      currentPnLAmount: 340,
      openPositions: 3,
      winRate: 67,
      healthScore: 85,
      riskLevel: 'moderate',
      lastSignal: '2 min ago'
    },
    {
      name: 'mean_reversion',
      displayName: 'Mean Reversion',
      status: 'active',
      allocation: 20,
      currentPnL: -1.2,
      currentPnLAmount: -120,
      openPositions: 2,
      winRate: 58,
      healthScore: 72,
      riskLevel: 'moderate',
      lastSignal: '15 min ago'
    },
    {
      name: 'arbitrage',
      displayName: 'Arbitrage',
      status: 'warning',
      allocation: 15,
      currentPnL: 0.8,
      currentPnLAmount: 80,
      openPositions: 1,
      winRate: 82,
      healthScore: 65,
      riskLevel: 'conservative',
      lastSignal: '1 hour ago'
    },
    {
      name: 'high_risk_daily',
      displayName: 'High Risk Daily',
      status: 'active',
      allocation: 10,
      currentPnL: 8.2,
      currentPnLAmount: 820,
      openPositions: 1,
      winRate: 55,
      healthScore: 78,
      riskLevel: 'extreme',
      lastSignal: 'Just now'
    },
    {
      name: 'candle_momentum',
      displayName: 'Candle Momentum',
      status: 'active',
      allocation: 30,
      currentPnL: 2.1,
      currentPnLAmount: 210,
      openPositions: 4,
      winRate: 71,
      healthScore: 92,
      riskLevel: 'aggressive',
      lastSignal: '5 min ago'
    }
  ];

  const activeStrategies = strategies.length > 0 ? strategies : demoStrategies;

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
      <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-4">
        Strategy Overview
      </h2>
      
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
          <thead>
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Strategy
              </th>
              <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Status
              </th>
              <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Allocation
              </th>
              <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Current P&L
              </th>
              <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Positions
              </th>
              <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Win Rate
              </th>
              <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Health
              </th>
              <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Last Signal
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
            {activeStrategies.map((strategy, index) => (
              <tr key={strategy.name || index} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                <td className="px-4 py-3 whitespace-nowrap">
                  <div>
                    <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                      {strategy.displayName}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      {strategy.riskLevel}
                    </div>
                  </div>
                </td>
                <td className="px-4 py-3 whitespace-nowrap text-center">
                  <div className="flex items-center justify-center">
                    {getStatusIcon(strategy.status)}
                  </div>
                </td>
                <td className="px-4 py-3 whitespace-nowrap text-center">
                  <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                    {strategy.allocation}%
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mt-1">
                    <div 
                      className="bg-blue-600 h-2 rounded-full"
                      style={{ width: `${strategy.allocation}%` }}
                    ></div>
                  </div>
                </td>
                <td className="px-4 py-3 whitespace-nowrap text-center">
                  <div className={`text-sm font-medium ${getPnLColor(strategy.currentPnL)}`}>
                    {strategy.currentPnL >= 0 ? '+' : ''}{strategy.currentPnL}%
                  </div>
                  <div className={`text-xs ${getPnLColor(strategy.currentPnLAmount)}`}>
                    ${Math.abs(strategy.currentPnLAmount)}
                  </div>
                </td>
                <td className="px-4 py-3 whitespace-nowrap text-center">
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
                    {strategy.openPositions}
                  </span>
                </td>
                <td className="px-4 py-3 whitespace-nowrap text-center">
                  <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                    {strategy.winRate}%
                  </div>
                </td>
                <td className="px-4 py-3 whitespace-nowrap text-center">
                  <div className={`text-sm font-bold ${getHealthColor(strategy.healthScore)}`}>
                    {strategy.healthScore}/100
                  </div>
                </td>
                <td className="px-4 py-3 whitespace-nowrap text-center">
                  <span className="text-xs text-gray-500 dark:text-gray-400">
                    {strategy.lastSignal}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Summary Stats */}
      <div className="mt-4 grid grid-cols-4 gap-4 pt-4 border-t border-gray-200 dark:border-gray-700">
        <div className="text-center">
          <p className="text-xs text-gray-500 dark:text-gray-400">Active Strategies</p>
          <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            {activeStrategies.filter(s => s.status === 'active').length}/{activeStrategies.length}
          </p>
        </div>
        <div className="text-center">
          <p className="text-xs text-gray-500 dark:text-gray-400">Total Allocation</p>
          <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            {activeStrategies.reduce((sum, s) => sum + s.allocation, 0)}%
          </p>
        </div>
        <div className="text-center">
          <p className="text-xs text-gray-500 dark:text-gray-400">Avg Win Rate</p>
          <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            {Math.round(activeStrategies.reduce((sum, s) => sum + s.winRate, 0) / activeStrategies.length)}%
          </p>
        </div>
        <div className="text-center">
          <p className="text-xs text-gray-500 dark:text-gray-400">Avg Health</p>
          <p className={`text-lg font-semibold ${getHealthColor(
            activeStrategies.reduce((sum, s) => sum + s.healthScore, 0) / activeStrategies.length
          )}`}>
            {Math.round(activeStrategies.reduce((sum, s) => sum + s.healthScore, 0) / activeStrategies.length)}/100
          </p>
        </div>
      </div>
    </div>
  );
};

export default StrategyOverviewPanel;