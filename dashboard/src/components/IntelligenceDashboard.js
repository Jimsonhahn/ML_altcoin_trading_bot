import React, { useState, useEffect, useRef } from 'react';
import { Chart, registerables } from 'chart.js';
import 'chartjs-adapter-date-fns';

Chart.register(...registerables);

const IntelligenceDashboard = () => {
    const [data, setData] = useState({
        insights: [],
        patterns: [],
        performance: [],
        recommendations: [],
        stats: { insights: {}, patterns: {} }
    });
    const [loading, setLoading] = useState(true);
    const [selectedPeriod, setSelectedPeriod] = useState('7d');
    const [selectedPattern, setSelectedPattern] = useState(null);
    const [lastUpdate, setLastUpdate] = useState(new Date());
    
    const performanceChartRef = useRef();
    const backtestChartRef = useRef();
    const performanceChart = useRef();
    const backtestChart = useRef();

    // API base URL
    const API_BASE = 'http://localhost:8001/api';

    // Fetch data from API
    const fetchData = async () => {
        try {
            const [insightsRes, patternsRes, performanceRes, recommendationsRes, insightsStatsRes, patternsStatsRes] = 
                await Promise.all([
                    fetch(`${API_BASE}/insights/latest?limit=20`),
                    fetch(`${API_BASE}/patterns/discovered?limit=10`),
                    fetch(`${API_BASE}/performance/evolution?period=${selectedPeriod}`),
                    fetch(`${API_BASE}/recommendations?limit=10`),
                    fetch(`${API_BASE}/insights/stats`),
                    fetch(`${API_BASE}/patterns/stats`)
                ]);

            const [insights, patterns, performance, recommendations, insightsStats, patternsStats] = 
                await Promise.all([
                    insightsRes.json(),
                    patternsRes.json(),
                    performanceRes.json(),
                    recommendationsRes.json(),
                    insightsStatsRes.json(),
                    patternsStatsRes.json()
                ]);

            setData({
                insights,
                patterns,
                performance,
                recommendations,
                stats: { insights: insightsStats, patterns: patternsStats }
            });
            setLastUpdate(new Date());
        } catch (error) {
            console.error('Failed to fetch data:', error);
        } finally {
            setLoading(false);
        }
    };

    // Update performance chart
    const updatePerformanceChart = () => {
        if (!performanceChartRef.current || !data.performance.length) return;

        const ctx = performanceChartRef.current.getContext('2d');
        
        if (performanceChart.current) {
            performanceChart.current.destroy();
        }

        // Group data by strategy
        const strategies = [...new Set(data.performance.map(d => d.strategy_name))];
        const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

        const datasets = strategies.map((strategy, index) => {
            const strategyData = data.performance
                .filter(d => d.strategy_name === strategy)
                .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

            return {
                label: strategy.replace('_', ' ').toUpperCase(),
                data: strategyData.map(d => ({
                    x: new Date(d.timestamp),
                    y: d.cumulative_return
                })),
                borderColor: colors[index % colors.length],
                backgroundColor: colors[index % colors.length] + '20',
                fill: false,
                tension: 0.1
            };
        });

        performanceChart.current = new Chart(ctx, {
            type: 'line',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: selectedPeriod === '7d' ? 'day' : 'week'
                        },
                        title: {
                            display: true,
                            text: 'Zeit'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Kumulativer Return (%)'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: `Performance Evolution (${selectedPeriod})`
                    },
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    };

    // Update backtest chart
    const updateBacktestChart = () => {
        if (!backtestChartRef.current) return;

        const ctx = backtestChartRef.current.getContext('2d');
        
        if (backtestChart.current) {
            backtestChart.current.destroy();
        }

        const labels = ['Baseline', 'ML Enhancement', 'Risk Optimization', 'Combined'];
        const chartData = [100, 115, 108, 123];

        backtestChart.current = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Return (%)',
                    data: chartData,
                    backgroundColor: [
                        '#6b7280',
                        '#3b82f6',
                        '#10b981',
                        '#f59e0b'
                    ],
                    borderColor: [
                        '#4b5563',
                        '#1d4ed8',
                        '#059669',
                        '#d97706'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Backtest Vergleich'
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Return (%)'
                        }
                    }
                }
            }
        });
    };

    // Format time ago
    const formatTimeAgo = (timestamp) => {
        const now = new Date();
        const time = new Date(timestamp);
        const diffMs = now - time;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMins / 60);
        const diffDays = Math.floor(diffHours / 24);
        
        if (diffMins < 1) return 'gerade eben';
        if (diffMins < 60) return `vor ${diffMins}m`;
        if (diffHours < 24) return `vor ${diffHours}h`;
        return `vor ${diffDays}d`;
    };

    // Get priority class
    const getPriorityClass = (priority) => {
        const classes = {
            'high': 'bg-red-100 text-red-800 border-red-200',
            'medium': 'bg-yellow-100 text-yellow-800 border-yellow-200',
            'low': 'bg-blue-100 text-blue-800 border-blue-200'
        };
        return classes[priority] || 'bg-gray-100 text-gray-800 border-gray-200';
    };

    // Get pattern type class
    const getPatternTypeClass = (type) => {
        const classes = {
            'success_pattern': 'bg-green-100 text-green-800',
            'dangerous_condition': 'bg-red-100 text-red-800',
            'strategy_synergy': 'bg-blue-100 text-blue-800'
        };
        return classes[type] || 'bg-gray-100 text-gray-800';
    };

    // Format pattern type
    const formatPatternType = (type) => {
        const names = {
            'success_pattern': 'Success Pattern',
            'dangerous_condition': 'Risk Alert',
            'strategy_synergy': 'Strategy Synergy'
        };
        return names[type] || type;
    };

    // Calculate performance boost
    const calculatePerformanceBoost = () => {
        if (!data.stats.patterns.avg_success_rate) return 0;
        return ((data.stats.patterns.avg_success_rate * 100) - 50).toFixed(1);
    };

    // Effects
    useEffect(() => {
        fetchData();
        const interval = setInterval(fetchData, 30000); // Auto-refresh every 30 seconds
        return () => clearInterval(interval);
    }, [selectedPeriod]);

    useEffect(() => {
        if (data.performance.length > 0) {
            updatePerformanceChart();
        }
    }, [data.performance, selectedPeriod]);

    useEffect(() => {
        updateBacktestChart();
    }, []);

    if (loading) {
        return (
            <div className="flex items-center justify-center h-96">
                <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-500"></div>
            </div>
        );
    }

    return (
        <div className="p-6 space-y-6">
            {/* Header */}
            <div className="flex justify-between items-center">
                <div className="flex items-center space-x-3">
                    <div className="text-2xl">🧠</div>
                    <h1 className="text-2xl font-bold text-gray-900">Intelligence Dashboard</h1>
                </div>
                <div className="flex items-center space-x-4 text-sm text-gray-500">
                    <span>Letztes Update: {lastUpdate.toLocaleTimeString()}</span>
                    <div className="flex items-center">
                        <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                        <span>Live</span>
                    </div>
                </div>
            </div>

            {/* Quick Stats */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div className="bg-white rounded-lg shadow p-6">
                    <div className="flex items-center">
                        <div className="text-yellow-500 text-2xl mr-4">💡</div>
                        <div>
                            <div className="text-sm font-medium text-gray-500">Heutige Insights</div>
                            <div className="text-2xl font-bold text-gray-900">
                                {data.stats.insights.total_insights_today || '--'}
                            </div>
                        </div>
                    </div>
                </div>

                <div className="bg-white rounded-lg shadow p-6">
                    <div className="flex items-center">
                        <div className="text-blue-500 text-2xl mr-4">🔍</div>
                        <div>
                            <div className="text-sm font-medium text-gray-500">Neue Muster</div>
                            <div className="text-2xl font-bold text-gray-900">
                                {data.stats.patterns.new_patterns_today || '--'}
                            </div>
                        </div>
                    </div>
                </div>

                <div className="bg-white rounded-lg shadow p-6">
                    <div className="flex items-center">
                        <div className="text-green-500 text-2xl mr-4">📈</div>
                        <div>
                            <div className="text-sm font-medium text-gray-500">Performance Boost</div>
                            <div className="text-2xl font-bold text-green-600">
                                +{calculatePerformanceBoost()}%
                            </div>
                        </div>
                    </div>
                </div>

                <div className="bg-white rounded-lg shadow p-6">
                    <div className="flex items-center">
                        <div className="text-purple-500 text-2xl mr-4">⚙️</div>
                        <div>
                            <div className="text-sm font-medium text-gray-500">Empfehlungen</div>
                            <div className="text-2xl font-bold text-gray-900">
                                {data.recommendations.length || '--'}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Main Dashboard Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                
                {/* Left Column */}
                <div className="lg:col-span-2 space-y-6">
                    
                    {/* Performance Evolution Chart */}
                    <div className="bg-white rounded-lg shadow p-6">
                        <div className="flex items-center justify-between mb-4">
                            <h2 className="text-lg font-semibold text-gray-900">
                                📊 Performance Evolution
                            </h2>
                            <div className="flex space-x-2">
                                {['7d', '30d', '90d'].map(period => (
                                    <button
                                        key={period}
                                        className={`px-3 py-1 text-xs rounded-md ${
                                            selectedPeriod === period
                                                ? 'bg-blue-100 text-blue-700'
                                                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                                        }`}
                                        onClick={() => setSelectedPeriod(period)}
                                    >
                                        {period.toUpperCase()}
                                    </button>
                                ))}
                            </div>
                        </div>
                        <div className="h-80">
                            <canvas ref={performanceChartRef}></canvas>
                        </div>
                    </div>

                    {/* Discovered Patterns */}
                    <div className="bg-white rounded-lg shadow p-6">
                        <h2 className="text-lg font-semibold text-gray-900 mb-4">
                            🧩 Entdeckte Muster
                        </h2>
                        <div className="space-y-4">
                            {data.patterns.map(pattern => (
                                <div
                                    key={pattern.id}
                                    className="border rounded-lg p-4 hover:bg-gray-50 cursor-pointer transition-colors"
                                    onClick={() => setSelectedPattern(pattern)}
                                >
                                    <div className="flex items-center justify-between mb-2">
                                        <h3 className="text-sm font-semibold text-gray-900">{pattern.name}</h3>
                                        <span className={`px-2 py-1 text-xs rounded-full ${getPatternTypeClass(pattern.pattern_type)}`}>
                                            {formatPatternType(pattern.pattern_type)}
                                        </span>
                                    </div>
                                    <p className="text-xs text-gray-600 mb-3">{pattern.description}</p>
                                    <div className="grid grid-cols-2 gap-2 text-xs">
                                        <div>
                                            <span className="text-gray-500">Confidence:</span>
                                            <span className="font-medium ml-1">{(pattern.confidence * 100).toFixed(0)}%</span>
                                        </div>
                                        <div>
                                            <span className="text-gray-500">Success Rate:</span>
                                            <span className="font-medium ml-1">{(pattern.success_rate * 100).toFixed(0)}%</span>
                                        </div>
                                        <div>
                                            <span className="text-gray-500">Frequency:</span>
                                            <span className="font-medium ml-1">{pattern.frequency}x</span>
                                        </div>
                                        <div>
                                            <span className="text-gray-500">Risk:</span>
                                            <span className="font-medium ml-1 capitalize">{pattern.risk_level}</span>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Backtest Results */}
                    <div className="bg-white rounded-lg shadow p-6">
                        <h2 className="text-lg font-semibold text-gray-900 mb-4">
                            🧪 Backtest Resultate
                        </h2>
                        <div className="h-80">
                            <canvas ref={backtestChartRef}></canvas>
                        </div>
                    </div>
                </div>

                {/* Right Column */}
                <div className="space-y-6">
                    
                    {/* Latest Insights */}
                    <div className="bg-white rounded-lg shadow p-6">
                        <h2 className="text-lg font-semibold text-gray-900 mb-4">
                            🧠 Neueste Insights
                        </h2>
                        <div className="space-y-4">
                            {data.insights.slice(0, 5).map(insight => (
                                <div key={insight.id} className="border-l-4 border-blue-400 pl-4">
                                    <div className="flex items-start space-x-3">
                                        <div className="flex-shrink-0 w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                                            <div className="text-blue-600 text-xs">💡</div>
                                        </div>
                                        <div className="flex-1 min-w-0">
                                            <p className="text-sm font-medium text-gray-900">{insight.title}</p>
                                            <p className="text-xs text-gray-500 mt-1">{insight.description}</p>
                                            <div className="flex items-center mt-2 space-x-2">
                                                <div className="w-16 h-2 bg-gray-200 rounded-full">
                                                    <div 
                                                        className="h-2 bg-gradient-to-r from-red-400 via-yellow-400 to-green-400 rounded-full"
                                                        style={{ width: `${insight.confidence * 100}%` }}
                                                    ></div>
                                                </div>
                                                <span className="text-xs text-gray-500">{(insight.confidence * 100).toFixed(0)}%</span>
                                                <span className="text-xs text-gray-400">{formatTimeAgo(insight.timestamp)}</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Recommended Actions */}
                    <div className="bg-white rounded-lg shadow p-6">
                        <h2 className="text-lg font-semibold text-gray-900 mb-4">
                            💡 Empfohlene Aktionen
                        </h2>
                        <div className="space-y-3">
                            {data.recommendations.slice(0, 5).map(rec => (
                                <div key={rec.id} className={`border-l-4 bg-gray-50 p-3 rounded-r ${
                                    rec.priority === 'high' ? 'border-red-400' : 
                                    rec.priority === 'medium' ? 'border-yellow-400' : 'border-blue-400'
                                }`}>
                                    <div className="flex items-center justify-between mb-1">
                                        <h3 className="text-sm font-medium text-gray-900">{rec.title}</h3>
                                        <span className={`px-2 py-1 text-xs rounded ${getPriorityClass(rec.priority)}`}>
                                            {rec.priority.toUpperCase()}
                                        </span>
                                    </div>
                                    <p className="text-xs text-gray-600 mb-2">{rec.description}</p>
                                    <div className="flex items-center justify-between text-xs">
                                        <span className="text-gray-500">
                                            Confidence: {(rec.confidence * 100).toFixed(0)}%
                                        </span>
                                        <span className="text-green-600 font-medium">
                                            +{(rec.expected_improvement * 100).toFixed(0)}%
                                        </span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>

            {/* Pattern Detail Modal */}
            {selectedPattern && (
                <div className="fixed inset-0 bg-gray-600 bg-opacity-50 z-50 flex items-center justify-center p-4">
                    <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-96 overflow-auto">
                        <div className="p-6">
                            <div className="flex justify-between items-center mb-4">
                                <h3 className="text-lg font-semibold">{selectedPattern.name}</h3>
                                <button 
                                    className="text-gray-400 hover:text-gray-600"
                                    onClick={() => setSelectedPattern(null)}
                                >
                                    ✕
                                </button>
                            </div>
                            <div className="space-y-4">
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <h4 className="font-medium text-gray-900 mb-2">Pattern Details</h4>
                                        <div className="space-y-1 text-sm">
                                            <div><span className="text-gray-500">Type:</span> {formatPatternType(selectedPattern.pattern_type)}</div>
                                            <div><span className="text-gray-500">Confidence:</span> {(selectedPattern.confidence * 100).toFixed(1)}%</div>
                                            <div><span className="text-gray-500">Success Rate:</span> {(selectedPattern.success_rate * 100).toFixed(1)}%</div>
                                            <div><span className="text-gray-500">Frequency:</span> {selectedPattern.frequency} occurrences</div>
                                            <div><span className="text-gray-500">Risk Level:</span> {selectedPattern.risk_level}</div>
                                        </div>
                                    </div>
                                    <div>
                                        <h4 className="font-medium text-gray-900 mb-2">Performance</h4>
                                        <div className="space-y-1 text-sm">
                                            <div><span className="text-gray-500">Profit Potential:</span> {(selectedPattern.profit_potential * 100).toFixed(1)}%</div>
                                            <div><span className="text-gray-500">Discovered:</span> {new Date(selectedPattern.discovered_at).toLocaleDateString()}</div>
                                            <div><span className="text-gray-500">Last Seen:</span> {formatTimeAgo(selectedPattern.last_seen)}</div>
                                        </div>
                                    </div>
                                </div>
                                <div>
                                    <h4 className="font-medium text-gray-900 mb-2">Description</h4>
                                    <p className="text-sm text-gray-600">{selectedPattern.description}</p>
                                </div>
                                <div>
                                    <h4 className="font-medium text-gray-900 mb-2">Strategies Involved</h4>
                                    <div className="flex flex-wrap gap-2">
                                        {selectedPattern.strategies_involved.map(strategy => 
                                            <span key={strategy} className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded">
                                                {strategy}
                                            </span>
                                        )}
                                    </div>
                                </div>
                                {selectedPattern.market_conditions && (
                                    <div>
                                        <h4 className="font-medium text-gray-900 mb-2">Market Conditions</h4>
                                        <div className="bg-gray-50 p-3 rounded text-sm">
                                            <pre className="text-xs whitespace-pre-wrap">
                                                {JSON.stringify(selectedPattern.market_conditions, null, 2)}
                                            </pre>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default IntelligenceDashboard;