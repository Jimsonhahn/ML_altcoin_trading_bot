import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell, Area, AreaChart } from 'recharts';
import { TrendingUp, TrendingDown, DollarSign, Target, Calendar, Award, AlertCircle, Activity } from 'lucide-react';
import apiService from '../services/api';
import { useTheme } from '../hooks/useTheme';

const PerformanceAnalytics = ({ className = '' }) => {
  const { isDark } = useTheme();
  const [analytics, setAnalytics] = useState({
    summary: {
      totalPnL: 0,
      totalTrades: 0,
      winRate: 0,
      avgWin: 0,
      avgLoss: 0,
      maxDrawdown: 0,
      sharpeRatio: 0,
      profitFactor: 0
    },
    dailyPnL: [],
    monthlyPnL: [],
    strategyPerformance: [],
    symbolPerformance: [],
    drawdownHistory: [],
    tradeDistribution: []
  });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedPeriod, setSelectedPeriod] = useState('30d');
  const [selectedMetric, setSelectedMetric] = useState('pnl');

  useEffect(() => {
    loadAnalytics();
  }, [selectedPeriod]);

  const loadAnalytics = async () => {
    setIsLoading(true);
    setError('');

    try {
      const response = await apiService.getPerformanceAnalytics({
        period: selectedPeriod,
        includeDrawdown: true,
        includeStrategyBreakdown: true
      });
      setAnalytics(response.analytics || analytics);
    } catch (err) {
      setError(err.message || 'Fehler beim Laden der Performance-Daten');
    } finally {
      setIsLoading(false);
    }
  };

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('de-DE', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(amount);
  };

  const formatPercentage = (value) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  const getPnLColor = (value) => {
    if (value > 0) return 'text-green-600 dark:text-green-400';
    if (value < 0) return 'text-red-600 dark:text-red-400';
    return 'text-gray-600 dark:text-gray-400';
  };

  const getMetricColor = (metric, value) => {
    switch (metric) {
      case 'winRate':
        return value >= 0.6 ? 'text-green-600 dark:text-green-400' : 
               value >= 0.4 ? 'text-yellow-600 dark:text-yellow-400' : 
               'text-red-600 dark:text-red-400';
      case 'sharpeRatio':
        return value >= 1.5 ? 'text-green-600 dark:text-green-400' : 
               value >= 1.0 ? 'text-yellow-600 dark:text-yellow-400' : 
               'text-red-600 dark:text-red-400';
      case 'profitFactor':
        return value >= 1.5 ? 'text-green-600 dark:text-green-400' : 
               value >= 1.0 ? 'text-yellow-600 dark:text-yellow-400' : 
               'text-red-600 dark:text-red-400';
      default:
        return getPnLColor(value);
    }
  };

  const pieColors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#06B6D4'];

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg p-3">
          <p className="text-sm font-medium text-gray-900 dark:text-white">{label}</p>
          {payload.map((entry, index) => (
            <p key={index} className="text-sm" style={{ color: entry.color }}>
              {entry.dataKey}: {entry.dataKey.includes('pnl') ? formatCurrency(entry.value) : entry.value}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  const summaryCards = [
    {
      title: 'Gesamt P&L',
      value: analytics.summary.totalPnL,
      format: 'currency',
      icon: <DollarSign className="w-5 h-5" />,
      color: getPnLColor(analytics.summary.totalPnL)
    },
    {
      title: 'Gesamt Trades',
      value: analytics.summary.totalTrades,
      format: 'number',
      icon: <Activity className="w-5 h-5" />,
      color: 'text-blue-600 dark:text-blue-400'
    },
    {
      title: 'Win Rate',
      value: analytics.summary.winRate,
      format: 'percentage',
      icon: <Target className="w-5 h-5" />,
      color: getMetricColor('winRate', analytics.summary.winRate)
    },
    {
      title: 'Avg Win',
      value: analytics.summary.avgWin,
      format: 'currency',
      icon: <TrendingUp className="w-5 h-5" />,
      color: 'text-green-600 dark:text-green-400'
    },
    {
      title: 'Avg Loss',
      value: analytics.summary.avgLoss,
      format: 'currency',
      icon: <TrendingDown className="w-5 h-5" />,
      color: 'text-red-600 dark:text-red-400'
    },
    {
      title: 'Max Drawdown',
      value: analytics.summary.maxDrawdown,
      format: 'percentage',
      icon: <AlertCircle className="w-5 h-5" />,
      color: 'text-red-600 dark:text-red-400'
    },
    {
      title: 'Sharpe Ratio',
      value: analytics.summary.sharpeRatio,
      format: 'number',
      icon: <Award className="w-5 h-5" />,
      color: getMetricColor('sharpeRatio', analytics.summary.sharpeRatio)
    },
    {
      title: 'Profit Factor',
      value: analytics.summary.profitFactor,
      format: 'number',
      icon: <TrendingUp className="w-5 h-5" />,
      color: getMetricColor('profitFactor', analytics.summary.profitFactor)
    }
  ];

  const formatValue = (value, format) => {
    switch (format) {
      case 'currency':
        return formatCurrency(value);
      case 'percentage':
        return formatPercentage(value);
      case 'number':
        return value.toFixed(2);
      default:
        return value;
    }
  };

  if (isLoading) {
    return (
      <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6 ${className}`}>
        <div className="flex items-center justify-center h-96">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6 ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
          Performance Analytics
        </h2>
        <div className="flex items-center gap-2">
          <select
            value={selectedPeriod}
            onChange={(e) => setSelectedPeriod(e.target.value)}
            className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
          >
            <option value="7d">7 Tage</option>
            <option value="30d">30 Tage</option>
            <option value="90d">90 Tage</option>
            <option value="1y">1 Jahr</option>
            <option value="all">Alle Zeit</option>
          </select>
        </div>
      </div>

      {error && (
        <div className="mb-6 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
          <span className="text-sm text-red-600 dark:text-red-400">{error}</span>
        </div>
      )}

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {summaryCards.map((card, index) => (
          <div key={index} className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-500 dark:text-gray-400">{card.title}</span>
              <div className="text-gray-400 dark:text-gray-500">
                {card.icon}
              </div>
            </div>
            <div className={`text-lg font-semibold ${card.color}`}>
              {formatValue(card.value, card.format)}
            </div>
          </div>
        ))}
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Daily P&L Chart */}
        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
          <h3 className="text-base font-medium text-gray-900 dark:text-white mb-4">
            Täglicher P&L
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={analytics.dailyPnL}>
              <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#374151' : '#E5E7EB'} />
              <XAxis 
                dataKey="date" 
                tick={{ fontSize: 12 }}
                stroke={isDark ? '#9CA3AF' : '#6B7280'}
              />
              <YAxis 
                tick={{ fontSize: 12 }}
                stroke={isDark ? '#9CA3AF' : '#6B7280'}
              />
              <Tooltip content={<CustomTooltip />} />
              <Area 
                type="monotone" 
                dataKey="pnl" 
                stroke="#3B82F6" 
                fill="#3B82F6" 
                fillOpacity={0.1}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Drawdown Chart */}
        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
          <h3 className="text-base font-medium text-gray-900 dark:text-white mb-4">
            Drawdown History
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={analytics.drawdownHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#374151' : '#E5E7EB'} />
              <XAxis 
                dataKey="date" 
                tick={{ fontSize: 12 }}
                stroke={isDark ? '#9CA3AF' : '#6B7280'}
              />
              <YAxis 
                tick={{ fontSize: 12 }}
                stroke={isDark ? '#9CA3AF' : '#6B7280'}
              />
              <Tooltip content={<CustomTooltip />} />
              <Line 
                type="monotone" 
                dataKey="drawdown" 
                stroke="#EF4444" 
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Strategy Performance */}
        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
          <h3 className="text-base font-medium text-gray-900 dark:text-white mb-4">
            Strategie Performance
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={analytics.strategyPerformance}>
              <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#374151' : '#E5E7EB'} />
              <XAxis 
                dataKey="strategy" 
                tick={{ fontSize: 12 }}
                stroke={isDark ? '#9CA3AF' : '#6B7280'}
              />
              <YAxis 
                tick={{ fontSize: 12 }}
                stroke={isDark ? '#9CA3AF' : '#6B7280'}
              />
              <Tooltip content={<CustomTooltip />} />
              <Bar 
                dataKey="pnl" 
                fill="#3B82F6"
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Symbol Performance */}
        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
          <h3 className="text-base font-medium text-gray-900 dark:text-white mb-4">
            Symbol Performance
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={analytics.symbolPerformance}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                paddingAngle={5}
                dataKey="pnl"
              >
                {analytics.symbolPerformance.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={pieColors[index % pieColors.length]} />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
            </PieChart>
          </ResponsiveContainer>
          <div className="mt-4 grid grid-cols-2 gap-2">
            {analytics.symbolPerformance.slice(0, 6).map((item, index) => (
              <div key={index} className="flex items-center text-sm">
                <div 
                  className="w-3 h-3 rounded-full mr-2" 
                  style={{ backgroundColor: pieColors[index % pieColors.length] }}
                />
                <span className="text-gray-600 dark:text-gray-400 truncate">
                  {item.symbol}: {formatCurrency(item.pnl)}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Monthly Performance Table */}
      <div className="mt-6 bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
        <h3 className="text-base font-medium text-gray-900 dark:text-white mb-4">
          Monatliche Performance
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-600">
                <th className="text-left py-2 px-4 text-gray-600 dark:text-gray-400">Monat</th>
                <th className="text-right py-2 px-4 text-gray-600 dark:text-gray-400">P&L</th>
                <th className="text-right py-2 px-4 text-gray-600 dark:text-gray-400">Trades</th>
                <th className="text-right py-2 px-4 text-gray-600 dark:text-gray-400">Win Rate</th>
                <th className="text-right py-2 px-4 text-gray-600 dark:text-gray-400">Avg Win</th>
                <th className="text-right py-2 px-4 text-gray-600 dark:text-gray-400">Avg Loss</th>
              </tr>
            </thead>
            <tbody>
              {analytics.monthlyPnL.map((month, index) => (
                <tr key={index} className="border-b border-gray-100 dark:border-gray-600">
                  <td className="py-2 px-4 text-gray-900 dark:text-white">
                    {new Date(month.month).toLocaleDateString('de-DE', { year: 'numeric', month: 'long' })}
                  </td>
                  <td className={`py-2 px-4 text-right font-medium ${getPnLColor(month.pnl)}`}>
                    {formatCurrency(month.pnl)}
                  </td>
                  <td className="py-2 px-4 text-right text-gray-900 dark:text-white">
                    {month.trades}
                  </td>
                  <td className="py-2 px-4 text-right text-gray-900 dark:text-white">
                    {formatPercentage(month.winRate)}
                  </td>
                  <td className="py-2 px-4 text-right text-green-600 dark:text-green-400">
                    {formatCurrency(month.avgWin)}
                  </td>
                  <td className="py-2 px-4 text-right text-red-600 dark:text-red-400">
                    {formatCurrency(month.avgLoss)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default PerformanceAnalytics;