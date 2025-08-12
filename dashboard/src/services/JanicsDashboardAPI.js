/**
 * Janics Dashboard API Client
 * ===========================
 * 
 * Complete API client for connecting existing dashboard UI to backend functionality.
 */

class JanicsDashboardAPI {
    constructor() {
        this.baseURL = process.env.REACT_APP_API_URL || 'http://localhost:5000';
        this.dashboardPrefix = '/api/v1/dashboard';
        this.authToken = null;
    }

    /**
     * Set authentication token
     */
    setAuthToken(token) {
        this.authToken = token;
    }

    /**
     * Make authenticated API request
     */
    async makeRequest(endpoint, options = {}) {
        const url = `${this.baseURL}${this.dashboardPrefix}${endpoint}`;
        
        const headers = {
            'Content-Type': 'application/json',
            ...options.headers
        };

        if (this.authToken) {
            headers['Authorization'] = `Bearer ${this.authToken}`;
        }

        try {
            const response = await fetch(url, {
                ...options,
                headers,
                credentials: 'include'  // For session-based auth
            });

            if (!response.ok) {
                throw new Error(`API Error: ${response.status} ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('API Request failed:', error);
            throw error;
        }
    }

    // ====================================
    // HEADER STATUS METHODS
    // ====================================

    /**
     * Get header status for Factory Online, Server, Bot, System indicators
     */
    async getHeaderStatus() {
        return this.makeRequest('/status/header');
    }

    /**
     * Get detailed system metrics
     */
    async getSystemMetrics() {
        return this.makeRequest('/status/system');
    }

    // ====================================
    // BOT CONTROL METHODS
    // ====================================

    /**
     * Start the trading bot
     */
    async startBot(options = {}) {
        const { mode = 'live', strategy, profile } = options;
        return this.makeRequest('/bot/start', {
            method: 'POST',
            body: JSON.stringify({ mode, strategy, profile })
        });
    }

    /**
     * Stop the trading bot
     */
    async stopBot() {
        return this.makeRequest('/bot/stop', {
            method: 'POST'
        });
    }

    /**
     * Restart the trading bot
     */
    async restartBot(options = {}) {
        const { mode = 'live', strategy, profile } = options;
        return this.makeRequest('/bot/restart', {
            method: 'POST',
            body: JSON.stringify({ mode, strategy, profile })
        });
    }

    /**
     * Get current bot process status
     */
    async getBotProcessStatus() {
        return this.makeRequest('/bot/status');
    }

    /**
     * Get bot logs
     */
    async getBotLogs(lines = 50) {
        return this.makeRequest(`/bot/logs?lines=${lines}`);
    }

    // ====================================
    // TRADES METHODS
    // ====================================

    /**
     * Get active trades for "Offene Trades - Live Positions" panel
     */
    async getActiveTrades() {
        return this.makeRequest('/trades/active');
    }

    /**
     * Get trade history
     */
    async getTradeHistory(limit = 50) {
        return this.makeRequest(`/trades/history?limit=${limit}`);
    }

    /**
     * Close a specific trade
     */
    async closeTrade(tradeId) {
        return this.makeRequest(`/trades/${tradeId}/close`, {
            method: 'POST'
        });
    }

    // ====================================
    // PORTFOLIO METHODS
    // ====================================

    /**
     * Get wealth data for "Wealth Accumulator" panel
     */
    async getWealthData() {
        return this.makeRequest('/portfolio/wealth');
    }

    /**
     * Get detailed portfolio breakdown
     */
    async getPortfolioBreakdown() {
        return this.makeRequest('/portfolio/breakdown');
    }

    // ====================================
    // BOT INTELLIGENCE METHODS
    // ====================================

    /**
     * Get bot intelligence status for "Janics Bot Status" panel
     */
    async getBotIntelligence() {
        return this.makeRequest('/bot/intelligence');
    }

    /**
     * Get bot learning history
     */
    async getLearningHistory(limit = 10) {
        return this.makeRequest(`/bot/learning/history?limit=${limit}`);
    }

    /**
     * Trigger a learning cycle
     */
    async triggerLearning() {
        return this.makeRequest('/bot/learning/trigger', {
            method: 'POST'
        });
    }

    // ====================================
    // STRATEGY SUPERMIX METHODS
    // ====================================

    /**
     * Get strategy supermix status for "Risk-Tiered Strategy Supermix" panel
     */
    async getStrategySupermix() {
        return this.makeRequest('/strategies/supermix');
    }

    /**
     * Start strategies in a risk tier (high/medium/low)
     */
    async startRiskTier(riskLevel) {
        return this.makeRequest(`/strategies/tier/${riskLevel}/start`, {
            method: 'POST'
        });
    }

    /**
     * Stop strategies in a risk tier
     */
    async stopRiskTier(riskLevel) {
        return this.makeRequest(`/strategies/tier/${riskLevel}/stop`, {
            method: 'POST'
        });
    }

    /**
     * Adjust allocation for a risk tier
     */
    async adjustTierAllocation(riskLevel, allocation) {
        return this.makeRequest(`/strategies/tier/${riskLevel}/allocation`, {
            method: 'PUT',
            body: JSON.stringify({ allocation })
        });
    }

    /**
     * Get detailed information about a specific strategy
     */
    async getStrategyDetails(strategyName) {
        return this.makeRequest(`/strategies/${strategyName}/details`);
    }

    // ====================================
    // AI ANALYTICS METHODS
    // ====================================

    /**
     * Get AI performance insights
     */
    async getAIInsights() {
        return this.makeRequest('/ai/insights');
    }

    /**
     * Get market intelligence data
     */
    async getMarketIntelligence() {
        return this.makeRequest('/ai/market-intelligence');
    }

    /**
     * Get market sentiment intelligence
     */
    async getSentimentIntelligence() {
        return this.makeRequest('/ai/sentiment');
    }

    // ====================================
    // DASHBOARD SUMMARY METHOD
    // ====================================

    /**
     * Get complete dashboard data in one API call (performance optimization)
     */
    async getDashboardSummary() {
        return this.makeRequest('/dashboard/summary');
    }

    // ====================================
    // UTILITY METHODS
    // ====================================

    /**
     * Format currency values
     */
    formatCurrency(value, currency = 'USD') {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: currency
        }).format(value);
    }

    /**
     * Format percentage values
     */
    formatPercentage(value, decimals = 2) {
        return `${value >= 0 ? '+' : ''}${value.toFixed(decimals)}%`;
    }

    /**
     * Format duration
     */
    formatDuration(seconds) {
        if (seconds < 60) return `${seconds}s`;
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
        if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
        return `${Math.floor(seconds / 86400)}d ${Math.floor((seconds % 86400) / 3600)}h`;
    }

    /**
     * Get status color based on value
     */
    getStatusColor(status) {
        const colors = {
            'running': '#10B981',      // Green
            'stopped': '#EF4444',      // Red
            'connecting': '#F59E0B',   // Yellow
            'operational': '#10B981',  // Green
            'warning': '#F59E0B',      // Yellow
            'error': '#EF4444'         // Red
        };
        return colors[status?.toLowerCase()] || '#6B7280'; // Gray default
    }

    /**
     * WebSocket connection for real-time updates
     */
    connectWebSocket(onMessage, onError = null) {
        const wsUrl = this.baseURL.replace('http', 'ws');
        const ws = new WebSocket(wsUrl);

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                onMessage(data);
            } catch (error) {
                console.error('WebSocket message parsing error:', error);
            }
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            if (onError) onError(error);
        };

        ws.onclose = () => {
            console.log('WebSocket connection closed');
            // Attempt to reconnect after 5 seconds
            setTimeout(() => this.connectWebSocket(onMessage, onError), 5000);
        };

        return ws;
    }
}

// Export singleton instance
const janicsDashboardAPI = new JanicsDashboardAPI();
export default janicsDashboardAPI;

// Also export the class for testing
export { JanicsDashboardAPI };