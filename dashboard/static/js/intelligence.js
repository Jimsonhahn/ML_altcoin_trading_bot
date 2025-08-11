/**
 * Intelligence Dashboard Frontend Logic
 * 
 * Features:
 * - Auto-refresh alle 30 Sekunden
 * - Interaktive Charts für Performance-Evolution
 * - Pattern-Visualisierung
 * - Insight-Timeline
 * - Mobile-responsive Design
 */

class IntelligenceDashboard {
    constructor() {
        this.apiBaseUrl = '/api';
        this.refreshInterval = 30000; // 30 seconds
        this.charts = {};
        this.refreshTimer = null;
        this.isLoading = false;
        
        // Initialize dashboard
        this.init();
    }
    
    async init() {
        console.log('🧠 Initializing Intelligence Dashboard...');
        
        try {
            // Setup event listeners
            this.setupEventListeners();
            
            // Initial data load
            await this.loadAllData();
            
            // Start auto-refresh
            this.startAutoRefresh();
            
            console.log('✅ Intelligence Dashboard initialized successfully');
        } catch (error) {
            console.error('❌ Failed to initialize dashboard:', error);
            this.showError('Failed to initialize dashboard');
        }
    }
    
    setupEventListeners() {
        // Performance chart timeframe buttons
        document.querySelectorAll('[data-period]').forEach(button => {
            button.addEventListener('click', (e) => {
                e.preventDefault();
                this.updatePerformanceChart(e.target.dataset.period);
                
                // Update button states
                document.querySelectorAll('[data-period]').forEach(btn => {
                    btn.classList.remove('active-timeframe', 'bg-blue-100', 'text-blue-700');
                    btn.classList.add('bg-gray-100', 'text-gray-700');
                });
                e.target.classList.remove('bg-gray-100', 'text-gray-700');
                e.target.classList.add('active-timeframe', 'bg-blue-100', 'text-blue-700');
            });
        });
        
        // Pattern modal events
        const modal = document.getElementById('patternModal');
        const closeModal = document.getElementById('closeModal');
        
        if (closeModal) {
            closeModal.addEventListener('click', () => {
                modal.classList.add('hidden');
            });
        }
        
        // Close modal on outside click
        if (modal) {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    modal.classList.add('hidden');
                }
            });
        }
        
        // Escape key to close modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && modal && !modal.classList.contains('hidden')) {
                modal.classList.add('hidden');
            }
        });
        
        // Manual refresh on header click
        const headerTitle = document.querySelector('h1');
        if (headerTitle) {
            headerTitle.style.cursor = 'pointer';
            headerTitle.addEventListener('click', () => {
                this.loadAllData();
            });
        }
    }
    
    async loadAllData() {
        if (this.isLoading) return;
        
        this.isLoading = true;
        this.updateConnectionStatus('loading');
        
        try {
            // Load all data in parallel
            const [
                insights,
                patterns,
                performance,
                recommendations,
                insightsStats,
                patternsStats
            ] = await Promise.all([
                this.fetchInsights(),
                this.fetchPatterns(),
                this.fetchPerformanceData('7d'),
                this.fetchRecommendations(),
                this.fetchInsightsStats(),
                this.fetchPatternsStats()
            ]);
            
            // Update UI components
            this.updateQuickStats(insightsStats, patternsStats);
            this.updateInsightsTimeline(insights);
            this.updatePatternsDisplay(patterns);
            this.updatePerformanceChart('7d', performance);
            this.updateRecommendations(recommendations);
            this.updateBacktestChart();
            this.updateStrategyHealth();
            
            // Update last update time
            this.updateLastUpdateTime();
            this.updateConnectionStatus('connected');
            
        } catch (error) {
            console.error('Failed to load dashboard data:', error);
            this.updateConnectionStatus('error');
            this.showError('Failed to load dashboard data');
        } finally {
            this.isLoading = false;
        }
    }
    
    // API Methods
    async fetchInsights(limit = 20, hours = 24) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/insights/latest?limit=${limit}&hours=${hours}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('Failed to fetch insights:', error);
            return this.getMockInsights();
        }
    }
    
    async fetchPatterns(limit = 10, minConfidence = 0.5) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/patterns/discovered?limit=${limit}&min_confidence=${minConfidence}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('Failed to fetch patterns:', error);
            return this.getMockPatterns();
        }
    }
    
    async fetchPerformanceData(period = '7d', strategies = null) {
        try {
            let url = `${this.apiBaseUrl}/performance/evolution?period=${period}`;
            if (strategies) {
                url += `&strategies=${strategies.join(',')}`;
            }
            const response = await fetch(url);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('Failed to fetch performance data:', error);
            return this.getMockPerformanceData(period);
        }
    }
    
    async fetchRecommendations(priority = null, limit = 10) {
        try {
            let url = `${this.apiBaseUrl}/recommendations?limit=${limit}`;
            if (priority) url += `&priority=${priority}`;
            const response = await fetch(url);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('Failed to fetch recommendations:', error);
            return this.getMockRecommendations();
        }
    }
    
    async fetchInsightsStats() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/insights/stats`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('Failed to fetch insights stats:', error);
            return {
                total_insights_today: 12,
                high_confidence_insights: 8,
                actionable_insights: 10,
                avg_confidence: 0.83
            };
        }
    }
    
    async fetchPatternsStats() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/patterns/stats`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('Failed to fetch patterns stats:', error);
            return {
                total_patterns: 23,
                new_patterns_today: 3,
                high_confidence_patterns: 15,
                avg_success_rate: 0.68
            };
        }
    }
    
    // UI Update Methods
    updateQuickStats(insightsStats, patternsStats) {
        const elements = {
            todayInsights: document.getElementById('todayInsights'),
            newPatterns: document.getElementById('newPatterns'),
            performanceBoost: document.getElementById('performanceBoost'),
            recommendations: document.getElementById('recommendations')
        };
        
        if (elements.todayInsights) {
            elements.todayInsights.textContent = insightsStats.total_insights_today || '--';
        }
        if (elements.newPatterns) {
            elements.newPatterns.textContent = patternsStats.new_patterns_today || '--';
        }
        if (elements.performanceBoost) {
            const boost = ((patternsStats.avg_success_rate || 0.68) * 100 - 50);
            elements.performanceBoost.textContent = `+${boost.toFixed(1)}%`;
        }
        if (elements.recommendations) {
            elements.recommendations.textContent = insightsStats.actionable_insights || '--';
        }
    }
    
    updateInsightsTimeline(insights) {
        const container = document.getElementById('insightsTimeline');
        if (!container) return;
        
        if (!insights || insights.length === 0) {
            container.innerHTML = '<p class="text-gray-500 text-sm">Keine Insights verfügbar</p>';
            return;
        }
        
        container.innerHTML = insights.slice(0, 5).map(insight => `
            <div class="timeline-item">
                <div class="flex items-start space-x-3">
                    <div class="flex-shrink-0">
                        <div class="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                            <i class="fas fa-lightbulb text-blue-600 text-xs"></i>
                        </div>
                    </div>
                    <div class="flex-1 min-w-0">
                        <p class="text-sm font-medium text-gray-900">${insight.title}</p>
                        <p class="text-xs text-gray-500 mt-1">${insight.description}</p>
                        <div class="flex items-center mt-2 space-x-2">
                            <div class="confidence-bar w-16 relative">
                                <div class="confidence-indicator" style="width: ${insight.confidence * 100}%"></div>
                            </div>
                            <span class="text-xs text-gray-500">${(insight.confidence * 100).toFixed(0)}%</span>
                            <span class="text-xs text-gray-400">${this.formatTimeAgo(insight.timestamp)}</span>
                        </div>
                    </div>
                </div>
            </div>
        `).join('');
    }
    
    updatePatternsDisplay(patterns) {
        const container = document.getElementById('patternsContainer');
        if (!container) return;
        
        if (!patterns || patterns.length === 0) {
            container.innerHTML = '<p class="text-gray-500 text-sm">Keine Muster gefunden</p>';
            return;
        }
        
        container.innerHTML = patterns.map(pattern => `
            <div class="border rounded-lg p-4 hover:bg-gray-50 cursor-pointer pattern-card" 
                 data-pattern-id="${pattern.id}"
                 onclick="dashboard.showPatternDetails('${pattern.id}', ${JSON.stringify(pattern).replace(/"/g, '&quot;')})">
                <div class="flex items-center justify-between mb-2">
                    <h3 class="text-sm font-semibold text-gray-900">${pattern.name}</h3>
                    <span class="px-2 py-1 text-xs rounded-full ${this.getPatternTypeClass(pattern.pattern_type)}">
                        ${this.formatPatternType(pattern.pattern_type)}
                    </span>
                </div>
                <p class="text-xs text-gray-600 mb-3">${pattern.description}</p>
                <div class="grid grid-cols-2 gap-2 text-xs">
                    <div>
                        <span class="text-gray-500">Confidence:</span>
                        <span class="font-medium">${(pattern.confidence * 100).toFixed(0)}%</span>
                    </div>
                    <div>
                        <span class="text-gray-500">Success Rate:</span>
                        <span class="font-medium">${(pattern.success_rate * 100).toFixed(0)}%</span>
                    </div>
                    <div>
                        <span class="text-gray-500">Frequency:</span>
                        <span class="font-medium">${pattern.frequency}x</span>
                    </div>
                    <div>
                        <span class="text-gray-500">Risk:</span>
                        <span class="font-medium capitalize">${pattern.risk_level}</span>
                    </div>
                </div>
            </div>
        `).join('');
    }
    
    async updatePerformanceChart(period, data = null) {
        if (!data) {
            data = await this.fetchPerformanceData(period);
        }
        
        const ctx = document.getElementById('performanceChart');
        if (!ctx) return;
        
        // Destroy existing chart
        if (this.charts.performance) {
            this.charts.performance.destroy();
        }
        
        // Process data for Chart.js
        const strategies = [...new Set(data.map(d => d.strategy_name))];
        const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];
        
        const datasets = strategies.map((strategy, index) => {
            const strategyData = data.filter(d => d.strategy_name === strategy);
            return {
                label: strategy.replace('_', ' ').toUpperCase(),
                data: strategyData.map(d => ({
                    x: d.timestamp,
                    y: d.cumulative_return
                })),
                borderColor: colors[index % colors.length],
                backgroundColor: colors[index % colors.length] + '20',
                fill: false,
                tension: 0.1
            };
        });
        
        this.charts.performance = new Chart(ctx, {
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
                            unit: period === '7d' ? 'day' : 'week'
                        },
                        display: true,
                        title: {
                            display: true,
                            text: 'Zeit'
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Kumulativer Return (%)'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: `Performance Evolution (${period})`
                    },
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }
    
    updateBacktestChart() {
        const ctx = document.getElementById('backtestChart');
        if (!ctx) return;
        
        // Destroy existing chart
        if (this.charts.backtest) {
            this.charts.backtest.destroy();
        }
        
        // Mock backtest data
        const labels = ['Baseline', 'ML Enhancement', 'Risk Optimization', 'Combined'];
        const data = [100, 115, 108, 123];
        
        this.charts.backtest = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Return (%)',
                    data: data,
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
    }
    
    updateRecommendations(recommendations) {
        const container = document.getElementById('recommendationsContainer');
        if (!container) return;
        
        if (!recommendations || recommendations.length === 0) {
            container.innerHTML = '<p class="text-gray-500 text-sm">Keine Empfehlungen verfügbar</p>';
            return;
        }
        
        container.innerHTML = recommendations.slice(0, 5).map(rec => `
            <div class="border-l-4 ${this.getPriorityBorderClass(rec.priority)} bg-gray-50 p-3 rounded-r">
                <div class="flex items-center justify-between mb-1">
                    <h3 class="text-sm font-medium text-gray-900">${rec.title}</h3>
                    <span class="px-2 py-1 text-xs rounded ${this.getPriorityClass(rec.priority)}">
                        ${rec.priority.toUpperCase()}
                    </span>
                </div>
                <p class="text-xs text-gray-600 mb-2">${rec.description}</p>
                <div class="flex items-center justify-between text-xs">
                    <span class="text-gray-500">
                        Confidence: ${(rec.confidence * 100).toFixed(0)}%
                    </span>
                    <span class="text-green-600 font-medium">
                        +${(rec.expected_improvement * 100).toFixed(0)}%
                    </span>
                </div>
            </div>
        `).join('');
    }
    
    updateStrategyHealth() {
        const container = document.getElementById('strategyHealth');
        if (!container) return;
        
        // Mock strategy health data
        const strategies = [
            { name: 'Momentum', health: 0.92, status: 'excellent' },
            { name: 'Mean Reversion', health: 0.78, status: 'good' },
            { name: 'Volatility', health: 0.65, status: 'warning' },
            { name: 'Arbitrage', health: 0.45, status: 'poor' }
        ];
        
        container.innerHTML = strategies.map(strategy => `
            <div class="flex items-center justify-between p-2 border rounded">
                <span class="text-sm font-medium">${strategy.name}</span>
                <div class="flex items-center space-x-2">
                    <div class="w-16 h-2 bg-gray-200 rounded-full">
                        <div class="h-2 rounded-full ${this.getHealthColor(strategy.health)}" 
                             style="width: ${strategy.health * 100}%"></div>
                    </div>
                    <span class="text-xs text-gray-500">${(strategy.health * 100).toFixed(0)}%</span>
                </div>
            </div>
        `).join('');
    }
    
    showPatternDetails(patternId, patternData) {
        const modal = document.getElementById('patternModal');
        const modalTitle = document.getElementById('modalTitle');
        const modalContent = document.getElementById('modalContent');
        
        if (!modal || !modalTitle || !modalContent) return;
        
        modalTitle.textContent = patternData.name;
        modalContent.innerHTML = `
            <div class="space-y-4">
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <h4 class="font-medium text-gray-900 mb-2">Pattern Details</h4>
                        <div class="space-y-1 text-sm">
                            <div><span class="text-gray-500">Type:</span> ${this.formatPatternType(patternData.pattern_type)}</div>
                            <div><span class="text-gray-500">Confidence:</span> ${(patternData.confidence * 100).toFixed(1)}%</div>
                            <div><span class="text-gray-500">Success Rate:</span> ${(patternData.success_rate * 100).toFixed(1)}%</div>
                            <div><span class="text-gray-500">Frequency:</span> ${patternData.frequency} occurrences</div>
                            <div><span class="text-gray-500">Risk Level:</span> ${patternData.risk_level}</div>
                        </div>
                    </div>
                    <div>
                        <h4 class="font-medium text-gray-900 mb-2">Performance</h4>
                        <div class="space-y-1 text-sm">
                            <div><span class="text-gray-500">Profit Potential:</span> ${(patternData.profit_potential * 100).toFixed(1)}%</div>
                            <div><span class="text-gray-500">Discovered:</span> ${this.formatDate(patternData.discovered_at)}</div>
                            <div><span class="text-gray-500">Last Seen:</span> ${this.formatTimeAgo(patternData.last_seen)}</div>
                        </div>
                    </div>
                </div>
                <div>
                    <h4 class="font-medium text-gray-900 mb-2">Description</h4>
                    <p class="text-sm text-gray-600">${patternData.description}</p>
                </div>
                <div>
                    <h4 class="font-medium text-gray-900 mb-2">Strategies Involved</h4>
                    <div class="flex flex-wrap gap-2">
                        ${patternData.strategies_involved.map(strategy => 
                            `<span class="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded">${strategy}</span>`
                        ).join('')}
                    </div>
                </div>
                ${patternData.market_conditions ? `
                <div>
                    <h4 class="font-medium text-gray-900 mb-2">Market Conditions</h4>
                    <div class="bg-gray-50 p-3 rounded text-sm">
                        <pre class="text-xs">${JSON.stringify(patternData.market_conditions, null, 2)}</pre>
                    </div>
                </div>
                ` : ''}
            </div>
        `;
        
        modal.classList.remove('hidden');
    }
    
    // Utility Methods
    startAutoRefresh() {
        this.refreshTimer = setInterval(() => {
            this.loadAllData();
        }, this.refreshInterval);
    }
    
    stopAutoRefresh() {
        if (this.refreshTimer) {
            clearInterval(this.refreshTimer);
            this.refreshTimer = null;
        }
    }
    
    updateConnectionStatus(status) {
        const indicator = document.getElementById('connectionStatus');
        if (!indicator) return;
        
        indicator.className = 'status-indicator ' + 
            (status === 'connected' ? 'status-active' : 
             status === 'loading' ? 'status-warning' : 'status-error');
    }
    
    updateLastUpdateTime() {
        const element = document.getElementById('lastUpdate');
        if (element) {
            element.textContent = new Date().toLocaleTimeString();
        }
        
        const lastAnalysisElement = document.getElementById('lastAnalysis');
        if (lastAnalysisElement) {
            lastAnalysisElement.textContent = 'vor 15 Minuten';
        }
    }
    
    showError(message) {
        // Simple error display - in production würde ein Toast/Notification System verwendet
        console.error('Dashboard Error:', message);
    }
    
    // Helper Methods for CSS Classes and Formatting
    getPatternTypeClass(type) {
        const classes = {
            'success_pattern': 'bg-green-100 text-green-800',
            'dangerous_condition': 'bg-red-100 text-red-800',
            'strategy_synergy': 'bg-blue-100 text-blue-800'
        };
        return classes[type] || 'bg-gray-100 text-gray-800';
    }
    
    formatPatternType(type) {
        const names = {
            'success_pattern': 'Success Pattern',
            'dangerous_condition': 'Risk Alert',
            'strategy_synergy': 'Strategy Synergy'
        };
        return names[type] || type;
    }
    
    getPriorityClass(priority) {
        const classes = {
            'high': 'bg-red-100 text-red-800',
            'medium': 'bg-yellow-100 text-yellow-800',
            'low': 'bg-blue-100 text-blue-800'
        };
        return classes[priority] || 'bg-gray-100 text-gray-800';
    }
    
    getPriorityBorderClass(priority) {
        const classes = {
            'high': 'border-red-400',
            'medium': 'border-yellow-400',
            'low': 'border-blue-400'
        };
        return classes[priority] || 'border-gray-400';
    }
    
    getHealthColor(health) {
        if (health >= 0.8) return 'bg-green-500';
        if (health >= 0.6) return 'bg-yellow-500';
        return 'bg-red-500';
    }
    
    formatTimeAgo(timestamp) {
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
    }
    
    formatDate(timestamp) {
        return new Date(timestamp).toLocaleDateString('de-DE', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
    }
    
    // Mock Data Methods (für Entwicklung/Testing)
    getMockInsights() {
        return [
            {
                id: 'mock_1',
                timestamp: new Date(Date.now() - 30 * 60000).toISOString(),
                type: 'optimization',
                title: 'Momentum Strategy Optimization',
                description: 'Detected opportunity to improve entry timing by 12%',
                confidence: 0.87,
                impact_score: 0.85,
                category: 'optimization'
            },
            {
                id: 'mock_2',
                timestamp: new Date(Date.now() - 45 * 60000).toISOString(),
                type: 'risk',
                title: 'Correlation Risk Alert',
                description: 'High correlation detected between strategies',
                confidence: 0.91,
                impact_score: 0.78,
                category: 'risk'
            }
        ];
    }
    
    getMockPatterns() {
        return [
            {
                id: 'pattern_1',
                pattern_type: 'success_pattern',
                name: 'High Volatility Momentum',
                description: 'Strong momentum signals during high volatility periods',
                confidence: 0.89,
                frequency: 47,
                success_rate: 0.73,
                profit_potential: 0.23,
                risk_level: 'medium',
                strategies_involved: ['momentum_strategy'],
                market_conditions: { volatility: 'high', volume: 'above_average' },
                discovered_at: new Date(Date.now() - 2 * 24 * 60 * 60000).toISOString(),
                last_seen: new Date(Date.now() - 3 * 60 * 60000).toISOString()
            }
        ];
    }
    
    getMockPerformanceData(period) {
        const days = { '7d': 7, '30d': 30, '90d': 90 }[period];
        const strategies = ['momentum_strategy', 'mean_reversion', 'volatility_strategy'];
        const data = [];
        
        for (let i = 0; i < days; i++) {
            const date = new Date(Date.now() - (days - i) * 24 * 60 * 60000);
            strategies.forEach(strategy => {
                const baseReturn = (Math.sin(i / 10) + Math.random() - 0.5) * 2;
                const cumulativeReturn = baseReturn * (i + 1) / 10;
                
                data.push({
                    timestamp: date.toISOString(),
                    strategy_name: strategy,
                    cumulative_return: cumulativeReturn,
                    daily_return: baseReturn
                });
            });
        }
        
        return data;
    }
    
    getMockRecommendations() {
        return [
            {
                id: 'rec_1',
                title: 'Increase Momentum Allocation',
                description: 'Current market conditions favor momentum strategies',
                priority: 'high',
                category: 'allocation',
                confidence: 0.87,
                expected_improvement: 0.15,
                implementation_effort: 'low',
                risk_level: 'low'
            }
        ];
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new IntelligenceDashboard();
});

// Handle page visibility change (pause/resume auto-refresh)
document.addEventListener('visibilitychange', () => {
    if (window.dashboard) {
        if (document.hidden) {
            window.dashboard.stopAutoRefresh();
        } else {
            window.dashboard.startAutoRefresh();
            window.dashboard.loadAllData(); // Refresh immediately when page becomes visible
        }
    }
});