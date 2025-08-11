import React, { useState } from 'react';
import {
  ArrowUpIcon,
  ArrowDownIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon
} from '@heroicons/react/24/outline';

const LivePositionTracker = ({ positions = [] }) => {
  const [sortBy, setSortBy] = useState('pnl');
  const [sortOrder, setSortOrder] = useState('desc');

  // Demo positions if none provided
  const demoPositions = [
    {
      id: 'pos_1',
      symbol: 'BTC/USDT',
      strategy: 'momentum_strategy',
      side: 'long',
      entryPrice: 44500.0,
      currentPrice: 45750.0,
      quantity: 0.15,
      positionValue: 6862.50,
      unrealizedPnL: 187.50,
      pnlPercent: 2.81,
      entryTime: '2h 15m ago',
      isPaper: true,
      stopLoss: 43560.0,
      takeProfit: 46725.0,
      riskRewardRatio: 2.1
    },
    {
      id: 'pos_2',
      symbol: 'ETH/USDT',
      strategy: 'mean_reversion',
      side: 'long',
      entryPrice: 2650.0,
      currentPrice: 2589.0,
      quantity: 2.5,
      positionValue: 6472.50,
      unrealizedPnL: -152.50,
      pnlPercent: -2.30,
      entryTime: '45m ago',
      isPaper: true,
      stopLoss: 2597.0,
      takeProfit: 2782.5,
      riskRewardRatio: 2.5
    },
    {
      id: 'pos_3',
      symbol: 'SOL/USDT',
      strategy: 'high_risk_daily',
      side: 'long',
      entryPrice: 98.20,
      currentPrice: 103.50,
      quantity: 15.0,
      positionValue: 1552.50,
      unrealizedPnL: 79.50,
      pnlPercent: 5.39,
      entryTime: '1h 30m ago',
      isPaper: true,
      stopLoss: 96.22,
      takeProfit: 108.02,
      riskRewardRatio: 5.0
    },
    {
      id: 'pos_4',
      symbol: 'ADA/USDT',
      strategy: 'candle_momentum',
      side: 'short',
      entryPrice: 0.545,
      currentPrice: 0.531,
      quantity: 1000,
      positionValue: 531.0,
      unrealizedPnL: 14.0,
      pnlPercent: 2.57,
      entryTime: '3h 45m ago',
      isPaper: true,
      stopLoss: 0.560,
      takeProfit: 0.518,
      riskRewardRatio: 1.8
    },
    {
      id: 'pos_5',
      symbol: 'MATIC/USDT',
      strategy: 'arbitrage',
      side: 'long',
      entryPrice: 0.845,
      currentPrice: 0.832,
      quantity: 500,
      positionValue: 416.0,
      unrealizedPnL: -6.50,
      pnlPercent: -1.54,
      entryTime: '25m ago',
      isPaper: true,
      stopLoss: 0.827,
      takeProfit: 0.887,
      riskRewardRatio: 2.3
    }
  ];

  const activePositions = positions.length > 0 ? positions : demoPositions;

  const handleSort = (field) => {
    if (sortBy === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field);
      setSortOrder('desc');
    }
  };

  const sortedPositions = [...activePositions].sort((a, b) => {
    let aVal = a[sortBy];
    let bVal = b[sortBy];
    
    if (typeof aVal === 'string') {
      aVal = aVal.toLowerCase();
      bVal = bVal.toLowerCase();
    }
    
    if (sortOrder === 'asc') {
      return aVal > bVal ? 1 : -1;
    } else {
      return aVal < bVal ? 1 : -1;
    }
  });

  const getSideIcon = (side) => {
    return side.toLowerCase() === 'long' ? (
      <ArrowUpIcon className="h-4 w-4 text-green-600" />
    ) : (
      <ArrowDownIcon className="h-4 w-4 text-red-600" />
    );
  };

  const getPnLColor = (pnl) => {
    return pnl >= 0 ? 'text-green-600' : 'text-red-600';
  };

  const getPnLBgColor = (pnl) => {
    return pnl >= 0 ? 'bg-green-100 dark:bg-green-900' : 'bg-red-100 dark:bg-red-900';
  };

  const getRiskIcon = (ratio) => {
    if (ratio >= 3) return <CheckCircleIcon className="h-4 w-4 text-green-500" />;
    if (ratio >= 2) return <ClockIcon className="h-4 w-4 text-yellow-500" />;
    return <ExclamationTriangleIcon className="h-4 w-4 text-red-500" />;
  };

  const totalUnrealizedPnL = activePositions.reduce((sum, pos) => sum + pos.unrealizedPnL, 0);
  const totalPositionValue = activePositions.reduce((sum, pos) => sum + pos.positionValue, 0);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">
          Live Position Tracker
        </h2>
        <div className="flex items-center space-x-4 text-sm">
          <span className="text-gray-500 dark:text-gray-400">
            Total Value: ${totalPositionValue.toLocaleString()}
          </span>
          <span className={`font-semibold ${getPnLColor(totalUnrealizedPnL)}`}>
            P&L: {totalUnrealizedPnL >= 0 ? '+' : ''}${totalUnrealizedPnL.toFixed(2)}
          </span>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
          <thead>
            <tr>
              <th 
                className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700"
                onClick={() => handleSort('symbol')}
              >
                Symbol {sortBy === 'symbol' && (sortOrder === 'asc' ? '↑' : '↓')}
              </th>
              <th 
                className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700"
                onClick={() => handleSort('strategy')}
              >
                Strategy {sortBy === 'strategy' && (sortOrder === 'asc' ? '↑' : '↓')}
              </th>
              <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Side
              </th>
              <th 
                className="px-4 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700"
                onClick={() => handleSort('entryPrice')}
              >
                Entry {sortBy === 'entryPrice' && (sortOrder === 'asc' ? '↑' : '↓')}
              </th>
              <th 
                className="px-4 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700"
                onClick={() => handleSort('currentPrice')}
              >
                Current {sortBy === 'currentPrice' && (sortOrder === 'asc' ? '↑' : '↓')}
              </th>
              <th 
                className="px-4 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700"
                onClick={() => handleSort('unrealizedPnL')}
              >
                P&L {sortBy === 'unrealizedPnL' && (sortOrder === 'asc' ? '↑' : '↓')}
              </th>
              <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Risk/Reward
              </th>
              <th 
                className="px-4 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700"
                onClick={() => handleSort('entryTime')}
              >
                Time {sortBy === 'entryTime' && (sortOrder === 'asc' ? '↑' : '↓')}
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
            {sortedPositions.map((position) => (
              <tr key={position.id} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                <td className="px-4 py-3 whitespace-nowrap">
                  <div className="flex items-center">
                    <div>
                      <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                        {position.symbol}
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        {position.quantity} units
                      </div>
                    </div>
                  </div>
                </td>
                <td className="px-4 py-3 whitespace-nowrap">
                  <div className="text-sm text-gray-900 dark:text-gray-100">
                    {position.strategy.replace('_strategy', '')}
                  </div>
                  {position.isPaper && (
                    <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
                      Paper
                    </span>
                  )}
                </td>
                <td className="px-4 py-3 whitespace-nowrap text-center">
                  <div className="flex items-center justify-center">
                    {getSideIcon(position.side)}
                    <span className={`ml-1 text-sm font-medium ${
                      position.side.toLowerCase() === 'long' ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {position.side.toUpperCase()}
                    </span>
                  </div>
                </td>
                <td className="px-4 py-3 whitespace-nowrap text-right">
                  <div className="text-sm text-gray-900 dark:text-gray-100">
                    ${position.entryPrice.toLocaleString()}
                  </div>
                </td>
                <td className="px-4 py-3 whitespace-nowrap text-right">
                  <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                    ${position.currentPrice.toLocaleString()}
                  </div>
                </td>
                <td className="px-4 py-3 whitespace-nowrap text-right">
                  <div className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getPnLBgColor(position.unrealizedPnL)}`}>
                    <div className={`${getPnLColor(position.unrealizedPnL)}`}>
                      <div className="font-bold">
                        {position.unrealizedPnL >= 0 ? '+' : ''}${position.unrealizedPnL.toFixed(2)}
                      </div>
                      <div className="text-xs">
                        ({position.pnlPercent >= 0 ? '+' : ''}{position.pnlPercent.toFixed(2)}%)
                      </div>
                    </div>
                  </div>
                </td>
                <td className="px-4 py-3 whitespace-nowrap text-center">
                  <div className="flex items-center justify-center">
                    {getRiskIcon(position.riskRewardRatio)}
                    <span className="ml-1 text-xs text-gray-500 dark:text-gray-400">
                      1:{position.riskRewardRatio}
                    </span>
                  </div>
                </td>
                <td className="px-4 py-3 whitespace-nowrap text-right">
                  <span className="text-xs text-gray-500 dark:text-gray-400">
                    {position.entryTime}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Summary Row */}
      <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
        <div className="grid grid-cols-4 gap-4 text-center">
          <div>
            <p className="text-xs text-gray-500 dark:text-gray-400">Total Positions</p>
            <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              {activePositions.length}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-500 dark:text-gray-400">Long/Short</p>
            <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              {activePositions.filter(p => p.side.toLowerCase() === 'long').length}/
              {activePositions.filter(p => p.side.toLowerCase() === 'short').length}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-500 dark:text-gray-400">Avg R/R</p>
            <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              1:{(activePositions.reduce((sum, p) => sum + p.riskRewardRatio, 0) / activePositions.length).toFixed(1)}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-500 dark:text-gray-400">Paper/Live</p>
            <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              {activePositions.filter(p => p.isPaper).length}/{activePositions.filter(p => !p.isPaper).length}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LivePositionTracker;