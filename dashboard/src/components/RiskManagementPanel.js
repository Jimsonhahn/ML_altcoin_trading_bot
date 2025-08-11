import React from 'react';
import {
  ShieldCheckIcon,
  ExclamationTriangleIcon,
  FireIcon,
  ClockIcon,
  CurrencyDollarIcon,
  ChartBarIcon
} from '@heroicons/react/24/outline';

const RiskManagementPanel = ({ riskData = {} }) => {
  // Default risk management data
  const defaultRiskData = {
    dailyLossLimit: 500,
    dailyLossUsed: 30,
    openRisk: 127,
    maxPositionSize: 100,
    correlationRisk: 'Low',
    emergencyStopArmed: true,
    drawdownLimit: 0.15,
    currentDrawdown: 0.032,
    riskPerTrade: 0.02,
    exposureByExchange: {
      'Binance': 60,
      'Coinbase': 25,
      'Kraken': 15
    },
    riskEvents: [
      { time: '2 min ago', message: 'Position size reduced due to correlation', severity: 'warning' },
      { time: '15 min ago', message: 'Daily loss limit check passed', severity: 'success' },
      { time: '1 hour ago', message: 'High volatility detected - reducing exposure', severity: 'warning' }
    ]
  };

  const data = { ...defaultRiskData, ...riskData };

  const getRiskColor = (level) => {
    switch (level?.toLowerCase()) {
      case 'low':
        return 'text-green-600 bg-green-100 dark:bg-green-900 dark:text-green-200';
      case 'medium':
        return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900 dark:text-yellow-200';
      case 'high':
        return 'text-red-600 bg-red-100 dark:bg-red-900 dark:text-red-200';
      default:
        return 'text-gray-600 bg-gray-100 dark:bg-gray-900 dark:text-gray-200';
    }
  };

  const getDrawdownSeverity = (current, limit) => {
    const ratio = current / limit;
    if (ratio < 0.5) return 'low';
    if (ratio < 0.8) return 'medium';
    return 'high';
  };

  const getLossUsageColor = (used, limit) => {
    const percentage = (used / limit) * 100;
    if (percentage < 20) return 'bg-green-500';
    if (percentage < 60) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  const RiskMetric = ({ icon: Icon, title, value, subValue, status, color = 'text-gray-900' }) => (
    <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center">
          <Icon className={`h-5 w-5 ${color} mr-2`} />
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">{title}</h4>
        </div>
        {status && (
          <span className={`px-2 py-1 text-xs font-medium rounded-full ${getRiskColor(status)}`}>
            {status.toUpperCase()}
          </span>
        )}
      </div>
      <div>
        <p className={`text-lg font-bold ${color} dark:text-gray-100`}>{value}</p>
        {subValue && (
          <p className="text-xs text-gray-500 dark:text-gray-400">{subValue}</p>
        )}
      </div>
    </div>
  );

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">
          Risk Management
        </h2>
        <div className="flex items-center">
          {data.emergencyStopArmed ? (
            <div className="flex items-center text-green-600">
              <ShieldCheckIcon className="h-5 w-5 mr-1" />
              <span className="text-sm font-medium">Emergency Stop Armed</span>
            </div>
          ) : (
            <div className="flex items-center text-red-600">
              <ExclamationTriangleIcon className="h-5 w-5 mr-1" />
              <span className="text-sm font-medium">Emergency Stop Disabled</span>
            </div>
          )}
        </div>
      </div>

      {/* Daily Loss Limit Progress */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">Daily Loss Limit</h3>
          <span className="text-sm text-gray-500 dark:text-gray-400">
            ${data.dailyLossUsed}/${data.dailyLossLimit} ({((data.dailyLossUsed / data.dailyLossLimit) * 100).toFixed(1)}%)
          </span>
        </div>
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
          <div 
            className={`h-3 rounded-full ${getLossUsageColor(data.dailyLossUsed, data.dailyLossLimit)}`}
            style={{ width: `${Math.min((data.dailyLossUsed / data.dailyLossLimit) * 100, 100)}%` }}
          />
        </div>
      </div>

      {/* Risk Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
        <RiskMetric
          icon={CurrencyDollarIcon}
          title="Open Risk"
          value={`$${data.openRisk}`}
          subValue="Across all positions"
          color="text-blue-600"
        />
        
        <RiskMetric
          icon={ChartBarIcon}
          title="Max Position Size"
          value={`$${data.maxPositionSize}`}
          subValue="Per single trade"
          color="text-purple-600"
        />
        
        <RiskMetric
          icon={FireIcon}
          title="Correlation Risk"
          value={data.correlationRisk}
          status={data.correlationRisk}
          color="text-orange-600"
        />
      </div>

      {/* Drawdown Monitoring */}
      <div className="mb-6 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Drawdown Monitoring</h3>
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-gray-600 dark:text-gray-400">Current Drawdown</p>
            <p className={`text-lg font-bold ${
              getDrawdownSeverity(data.currentDrawdown, data.drawdownLimit) === 'low' ? 'text-green-600' :
              getDrawdownSeverity(data.currentDrawdown, data.drawdownLimit) === 'medium' ? 'text-yellow-600' : 'text-red-600'
            }`}>
              {(data.currentDrawdown * 100).toFixed(1)}%
            </p>
          </div>
          <div>
            <p className="text-sm text-gray-600 dark:text-gray-400">Limit</p>
            <p className="text-lg font-bold text-gray-900 dark:text-gray-100">
              {(data.drawdownLimit * 100).toFixed(1)}%
            </p>
          </div>
          <div>
            <p className="text-sm text-gray-600 dark:text-gray-400">Buffer</p>
            <p className="text-lg font-bold text-green-600">
              {((data.drawdownLimit - data.currentDrawdown) * 100).toFixed(1)}%
            </p>
          </div>
        </div>
        <div className="mt-2 w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
          <div 
            className={`h-2 rounded-full ${
              getDrawdownSeverity(data.currentDrawdown, data.drawdownLimit) === 'low' ? 'bg-green-500' :
              getDrawdownSeverity(data.currentDrawdown, data.drawdownLimit) === 'medium' ? 'bg-yellow-500' : 'bg-red-500'
            }`}
            style={{ width: `${(data.currentDrawdown / data.drawdownLimit) * 100}%` }}
          />
        </div>
      </div>

      {/* Exchange Exposure */}
      <div className="mb-6">
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Exchange Exposure</h3>
        <div className="space-y-2">
          {Object.entries(data.exposureByExchange).map(([exchange, percentage]) => (
            <div key={exchange} className="flex items-center justify-between">
              <span className="text-sm text-gray-600 dark:text-gray-400">{exchange}</span>
              <div className="flex items-center">
                <div className="w-24 bg-gray-200 dark:bg-gray-600 rounded-full h-2 mr-2">
                  <div 
                    className="bg-blue-500 h-2 rounded-full"
                    style={{ width: `${percentage}%` }}
                  />
                </div>
                <span className="text-sm font-medium text-gray-900 dark:text-gray-100 w-10">
                  {percentage}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Recent Risk Events */}
      <div>
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Recent Risk Events</h3>
        <div className="space-y-2 max-h-32 overflow-y-auto">
          {data.riskEvents.map((event, index) => (
            <div key={index} className={`p-2 rounded text-xs ${
              event.severity === 'success' ? 'bg-green-50 dark:bg-green-900 text-green-800 dark:text-green-200' :
              event.severity === 'warning' ? 'bg-yellow-50 dark:bg-yellow-900 text-yellow-800 dark:text-yellow-200' :
              'bg-red-50 dark:bg-red-900 text-red-800 dark:text-red-200'
            }`}>
              <div className="flex items-center justify-between">
                <span>{event.message}</span>
                <span className="text-xs opacity-75">{event.time}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default RiskManagementPanel;