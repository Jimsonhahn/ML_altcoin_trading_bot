import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';

const CapitalAllocationChart = ({ allocations = {} }) => {
  // Default allocation data
  const defaultAllocations = {
    momentum_strategy: 25,
    mean_reversion: 20,
    arbitrage: 15,
    high_risk_daily: 10,
    candle_momentum: 30
  };

  const data = Object.keys(allocations).length > 0 ? allocations : defaultAllocations;

  // Convert to chart format
  const chartData = Object.entries(data).map(([strategy, percentage]) => ({
    name: strategy.replace('_strategy', '').replace('_', ' '),
    value: percentage,
    displayName: strategy.replace('_strategy', '').replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())
  }));

  // Colors for different strategies
  const COLORS = [
    '#3B82F6', // Blue
    '#10B981', // Green
    '#F59E0B', // Yellow
    '#EF4444', // Red
    '#8B5CF6', // Purple
    '#06B6D4', // Cyan
    '#F97316', // Orange
    '#84CC16'  // Lime
  ];

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0];
      return (
        <div className="bg-white dark:bg-gray-800 p-3 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
          <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
            {data.payload.displayName}
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Allocation: <span className="font-semibold">{data.value}%</span>
          </p>
        </div>
      );
    }
    return null;
  };

  const CustomLegend = ({ payload }) => {
    return (
      <div className="grid grid-cols-2 gap-2 mt-4">
        {payload.map((entry, index) => (
          <div key={index} className="flex items-center">
            <div 
              className="w-3 h-3 rounded-full mr-2"
              style={{ backgroundColor: entry.color }}
            />
            <span className="text-xs text-gray-600 dark:text-gray-400 truncate">
              {entry.payload.displayName}: {entry.payload.value}%
            </span>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
      <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-4">
        Capital Allocation
      </h2>
      
      <div className="flex flex-col lg:flex-row items-center">
        <div className="w-full lg:w-2/3">
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={chartData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={120}
                dataKey="value"
                animationBegin={0}
                animationDuration={800}
              >
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
            </PieChart>
          </ResponsiveContainer>
        </div>
        
        <div className="w-full lg:w-1/3 lg:pl-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-3">
            Strategy Breakdown
          </h3>
          <div className="space-y-2">
            {chartData.map((entry, index) => (
              <div key={index} className="flex items-center justify-between">
                <div className="flex items-center">
                  <div 
                    className="w-4 h-4 rounded-full mr-3"
                    style={{ backgroundColor: COLORS[index % COLORS.length] }}
                  />
                  <span className="text-sm text-gray-700 dark:text-gray-300">
                    {entry.displayName}
                  </span>
                </div>
                <div className="text-right">
                  <div className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                    {entry.value}%
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    ${(entry.value * 105).toFixed(0)} {/* Assuming $10.5k total */}
                  </div>
                </div>
              </div>
            ))}
          </div>
          
          {/* Allocation Stats */}
          <div className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-700">
            <div className="grid grid-cols-2 gap-4 text-center">
              <div>
                <p className="text-xs text-gray-500 dark:text-gray-400">Total Allocated</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  {chartData.reduce((sum, item) => sum + item.value, 0)}%
                </p>
              </div>
              <div>
                <p className="text-xs text-gray-500 dark:text-gray-400">Strategies</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  {chartData.length}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Rebalancing Info */}
      <div className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-gray-600 dark:text-gray-400">Last Rebalancing</p>
            <p className="text-xs text-gray-500 dark:text-gray-500">2 hours ago</p>
          </div>
          <div>
            <p className="text-sm text-gray-600 dark:text-gray-400">Next Check</p>
            <p className="text-xs text-gray-500 dark:text-gray-500">In 4 hours</p>
          </div>
          <div>
            <p className="text-sm text-gray-600 dark:text-gray-400">Rebalance Trigger</p>
            <p className="text-xs text-gray-500 dark:text-gray-500">±5% deviation</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CapitalAllocationChart;