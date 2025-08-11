import React from 'react';
import {
  CurrencyDollarIcon,
  ArrowTrendingUpIcon,
  ChartBarIcon,
  ExclamationTriangleIcon,
  CalendarIcon,
  ClockIcon
} from '@heroicons/react/24/outline';

const PerformanceMetricsGrid = ({ metrics = {} }) => {
  // Default demo metrics
  const defaultMetrics = {
    totalPnL: 12.5,
    totalPnLAmount: 1250,
    todayPnL: 2.3,
    todayPnLAmount: 230,
    winRate: 64,
    totalTrades: 145,
    sharpeRatio: 1.85,
    maxDrawdown: -8.2,
    currentDrawdown: -3.2,
    avgWinTrade: 45.20,
    avgLossTrade: -28.50,
    profitFactor: 2.14,
    calmarRatio: 5.91,
    sortinoRatio: 2.45,
    consecutiveWins: 7,
    consecutiveLosses: 0,
    bestDay: 450,
    worstDay: -180,
    tradingDays: 45,
    uptimeDays: 47
  };

  const data = { ...defaultMetrics, ...metrics };

  const MetricCard = ({ icon: Icon, title, value, subValue, trend, color = 'text-gray-900' }) => (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center">
          <Icon className={`h-5 w-5 ${color} mr-2`} />
          <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">{title}</h3>
        </div>
        {trend && (
          <ArrowTrendingUpIcon className={`h-4 w-4 ${trend > 0 ? 'text-green-500' : 'text-red-500'}`} />
        )}
      </div>
      <div className="mt-2">
        <p className={`text-2xl font-bold ${color} dark:text-gray-100`}>{value}</p>
        {subValue && (
          <p className="text-sm text-gray-500 dark:text-gray-400">{subValue}</p>
        )}
      </div>
    </div>
  );

  const formatPnL = (value) => {
    return value >= 0 ? `+${value}%` : `${value}%`;
  };

  const formatCurrency = (value) => {
    return value >= 0 ? `+$${Math.abs(value)}` : `-$${Math.abs(value)}`;
  };

  return (
    <div className="space-y-6">
      {/* Main Performance Header */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg shadow-lg p-6 text-white">
        <h2 className="text-2xl font-bold mb-4">GESAMT-PERFORMANCE</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <p className="text-sm opacity-90">Total P&L</p>
            <p className="text-3xl font-bold">{formatPnL(data.totalPnL)}</p>
            <p className="text-sm opacity-75">{formatCurrency(data.totalPnLAmount)}</p>
          </div>
          <div>
            <p className="text-sm opacity-90">Today's P&L</p>
            <p className="text-3xl font-bold">{formatPnL(data.todayPnL)}</p>
            <p className="text-sm opacity-75">{formatCurrency(data.todayPnLAmount)}</p>
          </div>
          <div>
            <p className="text-sm opacity-90">Win Rate</p>
            <p className="text-3xl font-bold">{data.winRate}%</p>
            <p className="text-sm opacity-75">{data.totalTrades} trades</p>
          </div>
          <div>
            <p className="text-sm opacity-90">Sharpe Ratio</p>
            <p className="text-3xl font-bold">{data.sharpeRatio}</p>
            <p className="text-sm opacity-75">Risk-adjusted</p>
          </div>
        </div>
      </div>

      {/* Detailed Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          icon={ExclamationTriangleIcon}
          title="Max Drawdown"
          value={`${data.maxDrawdown}%`}
          subValue={`Current: ${data.currentDrawdown}%`}
          color={data.currentDrawdown < -5 ? 'text-red-600' : 'text-orange-600'}
        />
        
        <MetricCard
          icon={ChartBarIcon}
          title="Profit Factor"
          value={data.profitFactor}
          subValue="Win/Loss Ratio"
          color="text-green-600"
        />
        
        <MetricCard
          icon={CurrencyDollarIcon}
          title="Avg Win/Loss"
          value={`$${data.avgWinTrade}`}
          subValue={`Loss: $${Math.abs(data.avgLossTrade)}`}
          color="text-blue-600"
        />
        
        <MetricCard
          icon={ArrowTrendingUpIcon}
          title="Win Streak"
          value={data.consecutiveWins}
          subValue={`Losses: ${data.consecutiveLosses}`}
          color={data.consecutiveWins > 5 ? 'text-green-600' : 'text-gray-600'}
        />
      </div>

      {/* Advanced Ratios */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
          Advanced Metrics
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <p className="text-sm text-gray-500 dark:text-gray-400">Calmar Ratio</p>
            <p className="text-xl font-bold text-gray-900 dark:text-gray-100">{data.calmarRatio}</p>
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-500 dark:text-gray-400">Sortino Ratio</p>
            <p className="text-xl font-bold text-gray-900 dark:text-gray-100">{data.sortinoRatio}</p>
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-500 dark:text-gray-400">Best Day</p>
            <p className="text-xl font-bold text-green-600">{formatCurrency(data.bestDay)}</p>
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-500 dark:text-gray-400">Worst Day</p>
            <p className="text-xl font-bold text-red-600">{formatCurrency(data.worstDay)}</p>
          </div>
        </div>
      </div>

      {/* Time Stats */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
          <div className="flex items-center">
            <CalendarIcon className="h-5 w-5 text-blue-600 mr-2" />
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Trading Days</p>
              <p className="text-xl font-bold text-gray-900 dark:text-gray-100">{data.tradingDays}</p>
            </div>
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
          <div className="flex items-center">
            <ClockIcon className="h-5 w-5 text-green-600 mr-2" />
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">System Uptime</p>
              <p className="text-xl font-bold text-gray-900 dark:text-gray-100">{data.uptimeDays} days</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PerformanceMetricsGrid;