/**
 * PnLChart Component
 * Real-time P&L chart with Chart.js
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  Filler,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import 'chartjs-adapter-date-fns';
import { usePerformanceUpdates } from '../hooks/useWebSocket';
import { useTheme } from '../hooks/useTheme';
import robustAPI from '../services/robustApi';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  Filler
);

const PnLChart = ({ className = '', period = 'day' }) => {
  const [chartData, setChartData] = useState({ labels: [], datasets: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedPeriod, setSelectedPeriod] = useState(period);
  const { performance, performanceHistory } = usePerformanceUpdates();
  const { isDark } = useTheme();
  const chartRef = useRef(null);

  useEffect(() => {
    loadPerformanceData();
  }, [selectedPeriod]);

  useEffect(() => {
    // Update chart with real-time data
    if (performanceHistory && performanceHistory.length > 0) {
      updateChartData(performanceHistory);
    }
  }, [performanceHistory, isDark]);

  const loadPerformanceData = async () => {
    try {
      setLoading(true);
      const response = await robustAPI.fetchWithRetry(`/api/v1/trading/performance?period=${selectedPeriod}`);
      
      // Convert to chart data format
      const history = response.history || [];
      updateChartData(history);
      
      setError(null);
    } catch (err) {
      setError('Failed to load performance data');
      console.error('Error loading performance:', err);
    } finally {
      setLoading(false);
    }
  };

  const updateChartData = (history) => {
    const labels = history.map(item => new Date(item.timestamp));
    const pnlData = history.map(item => item.total_pnl || 0);
    const returnsData = history.map(item => (item.total_return || 0) * 100);

    const gridColor = isDark ? 'rgb(75, 85, 99)' : 'rgb(229, 231, 235)';
    const textColor = isDark ? 'rgb(156, 163, 175)' : 'rgb(75, 85, 99)';

    setChartData({
      labels,
      datasets: [
        {
          label: 'P&L (USD)',
          data: pnlData,
          borderColor: pnlData && pnlData.length > 0 && pnlData[pnlData.length - 1] >= 0 ? 'rgb(34, 197, 94)' : 'rgb(239, 68, 68)',
          backgroundColor: pnlData && pnlData.length > 0 && pnlData[pnlData.length - 1] >= 0 
            ? 'rgba(34, 197, 94, 0.1)' 
            : 'rgba(239, 68, 68, 0.1)',
          borderWidth: 2,
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          pointHoverRadius: 6,
          pointHoverBorderWidth: 2,
          pointHoverBorderColor: '#ffffff',
        },
        {
          label: 'Return (%)',
          data: returnsData,
          borderColor: 'rgb(59, 130, 246)',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          borderWidth: 2,
          fill: false,
          tension: 0.4,
          pointRadius: 0,
          pointHoverRadius: 6,
          pointHoverBorderWidth: 2,
          pointHoverBorderColor: '#ffffff',
          yAxisID: 'y1',
        },
      ],
    });
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      title: {
        display: true,
        text: `P&L Performance (${selectedPeriod})`,
        color: isDark ? 'rgb(156, 163, 175)' : 'rgb(75, 85, 99)',
        font: {
          size: 16,
          weight: 'bold',
        },
      },
      legend: {
        display: true,
        position: 'top',
        labels: {
          color: isDark ? 'rgb(156, 163, 175)' : 'rgb(75, 85, 99)',
          usePointStyle: true,
          padding: 20,
        },
      },
      tooltip: {
        backgroundColor: isDark ? 'rgb(17, 24, 39)' : 'rgb(255, 255, 255)',
        titleColor: isDark ? 'rgb(243, 244, 246)' : 'rgb(17, 24, 39)',
        bodyColor: isDark ? 'rgb(209, 213, 219)' : 'rgb(75, 85, 99)',
        borderColor: isDark ? 'rgb(75, 85, 99)' : 'rgb(229, 231, 235)',
        borderWidth: 1,
        callbacks: {
          label: (context) => {
            const label = context.dataset.label || '';
            const value = context.parsed.y;
            if (label.includes('P&L')) {
              return `${label}: $${value.toFixed(2)}`;
            } else if (label.includes('Return')) {
              return `${label}: ${value.toFixed(2)}%`;
            }
            return `${label}: ${value}`;
          },
        },
      },
    },
    scales: {
      x: {
        type: 'time',
        time: {
          displayFormats: {
            hour: 'HH:mm',
            day: 'MMM dd',
            week: 'MMM dd',
            month: 'MMM yyyy',
          },
        },
        grid: {
          color: isDark ? 'rgb(75, 85, 99)' : 'rgb(229, 231, 235)',
        },
        ticks: {
          color: isDark ? 'rgb(156, 163, 175)' : 'rgb(75, 85, 99)',
        },
      },
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        title: {
          display: true,
          text: 'P&L (USD)',
          color: isDark ? 'rgb(156, 163, 175)' : 'rgb(75, 85, 99)',
        },
        grid: {
          color: isDark ? 'rgb(75, 85, 99)' : 'rgb(229, 231, 235)',
        },
        ticks: {
          color: isDark ? 'rgb(156, 163, 175)' : 'rgb(75, 85, 99)',
          callback: (value) => `$${value.toFixed(0)}`,
        },
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        title: {
          display: true,
          text: 'Return (%)',
          color: isDark ? 'rgb(156, 163, 175)' : 'rgb(75, 85, 99)',
        },
        grid: {
          drawOnChartArea: false,
        },
        ticks: {
          color: isDark ? 'rgb(156, 163, 175)' : 'rgb(75, 85, 99)',
          callback: (value) => `${value.toFixed(1)}%`,
        },
      },
    },
  };

  const periods = [
    { value: 'day', label: '1D' },
    { value: 'week', label: '1W' },
    { value: 'month', label: '1M' },
    { value: 'all', label: 'All' },
  ];

  const currentPnL = performance?.total_pnl || 0;
  const currentReturn = (performance?.total_return || 0) * 100;

  if (loading) {
    return (
      <div className={`card ${className}`}>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
            P&L Chart
          </h2>
          <div className="loading-spinner"></div>
        </div>
        <div className="h-96 bg-gray-100 dark:bg-gray-700 rounded-lg animate-pulse flex items-center justify-center">
          <div className="text-gray-500 dark:text-gray-400">Loading chart...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`card ${className}`}>
        <div className="text-center py-8">
          <div className="text-danger-600 dark:text-danger-400 mb-4">
            <svg className="w-12 h-12 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
          </div>
          <p className="text-gray-600 dark:text-gray-400 mb-4">{error}</p>
          <button
            onClick={loadPerformanceData}
            className="btn-primary"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`card ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
            P&L Chart
          </h2>
          <div className="flex items-center space-x-4 mt-2">
            <div className="flex items-center space-x-2">
              <span className="text-sm text-gray-500 dark:text-gray-400">P&L:</span>
              <span className={`font-medium ${
                currentPnL >= 0 
                  ? 'text-success-600 dark:text-success-400' 
                  : 'text-danger-600 dark:text-danger-400'
              }`}>
                ${currentPnL.toFixed(2)}
              </span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-sm text-gray-500 dark:text-gray-400">Return:</span>
              <span className={`font-medium ${
                currentReturn >= 0 
                  ? 'text-success-600 dark:text-success-400' 
                  : 'text-danger-600 dark:text-danger-400'
              }`}>
                {currentReturn.toFixed(2)}%
              </span>
            </div>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <div className="flex bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
            {periods.map(({ value, label }) => (
              <button
                key={value}
                onClick={() => setSelectedPeriod(value)}
                className={`px-3 py-1 text-sm font-medium rounded-md transition-colors ${
                  selectedPeriod === value
                    ? 'bg-primary-600 text-white'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          <button
            onClick={loadPerformanceData}
            className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </button>
        </div>
      </div>

      <div className="chart-container">
        <Line ref={chartRef} data={chartData} options={chartOptions} />
      </div>
    </div>
  );
};

export default PnLChart;