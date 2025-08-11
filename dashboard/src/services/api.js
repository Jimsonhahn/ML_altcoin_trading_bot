/**
 * API Service for Trading Bot Dashboard
 * Handles all HTTP requests to the Flask API
 */

class ApiService {
  constructor() {
    this.baseURL = process.env.REACT_APP_API_URL || 'http://localhost:5000';
    this.token = localStorage.getItem('auth_token') || 'test_token_for_development';
  }

  // Authentication methods
  async login(username, password) {
    try {
      const response = await fetch(`${this.baseURL}/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ username, password }),
      });

      if (!response.ok) {
        throw new Error('Login failed');
      }

      const data = await response.json();
      this.token = data.access_token;
      localStorage.setItem('auth_token', this.token);
      
      // Trigger authentication change event
      window.dispatchEvent(new CustomEvent('authChange'));
      
      return data;
    } catch (error) {
      console.error('Login error:', error);
      throw error;
    }
  }

  async logout() {
    try {
      await fetch(`${this.baseURL}/auth/logout`, {
        method: 'POST',
        headers: this.getAuthHeaders(),
        credentials: 'include',
      });
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      this.token = null;
      localStorage.removeItem('auth_token');
      
      // Trigger authentication change event
      window.dispatchEvent(new CustomEvent('authChange'));
    }
  }

  async refreshToken() {
    try {
      const refreshToken = localStorage.getItem('refresh_token');
      const response = await fetch(`${this.baseURL}/auth/refresh`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${refreshToken}`,
          'Content-Type': 'application/json',
        },
        credentials: 'include',
      });

      if (!response.ok) {
        throw new Error('Token refresh failed');
      }

      const data = await response.json();
      this.token = data.access_token;
      localStorage.setItem('auth_token', this.token);
      return data;
    } catch (error) {
      console.error('Token refresh error:', error);
      this.logout();
      throw error;
    }
  }

  // Helper methods
  getAuthHeaders() {
    return {
      'Authorization': `Bearer ${this.token}`,
      'Content-Type': 'application/json',
    };
  }

  async handleResponse(response) {
    if (response.status === 401) {
      // Token expired, try to refresh
      await this.refreshToken();
      return null; // Indicate retry needed
    }

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || 'API request failed');
    }

    return response.json();
  }

  async apiCall(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const defaultOptions = {
      headers: this.getAuthHeaders(),
      credentials: 'include',
      ...options,
    };

    try {
      let response = await fetch(url, defaultOptions);
      let data = await this.handleResponse(response);

      // If handleResponse returned null (token refresh), retry the request
      if (data === null) {
        response = await fetch(url, {
          ...defaultOptions,
          headers: this.getAuthHeaders(),
        });
        data = await this.handleResponse(response);
      }

      return data;
    } catch (error) {
      console.error(`API call failed for ${endpoint}:`, error);
      throw error;
    }
  }

  // Trading API methods
  async getTradingStatus() {
    return this.apiCall('/api/v1/trading/status');
  }

  async getBotStatus() {
    return this.apiCall('/api/v1/trading/status');
  }

  async getDetailedBotStatus() {
    return this.apiCall('/api/v1/trading/detailed-status');
  }

  async startBot(config) {
    return this.apiCall('/api/v1/trading/start', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  async stopBot() {
    return this.apiCall('/api/v1/trading/stop', {
      method: 'POST',
    });
  }

  async restartBot(config = null) {
    return this.apiCall('/api/v1/trading/restart', {
      method: 'POST',
      body: config ? JSON.stringify(config) : undefined,
    });
  }

  async forceStopBot() {
    return this.apiCall('/api/v1/trading/force-stop', {
      method: 'POST',
    });
  }

  // Legacy methods (for backward compatibility)
  async startTrading(mode, strategy, config = {}) {
    const botConfig = {
      mode,
      strategy,
      symbol: config.symbol || 'BTC/USDT',
      capital: config.capital || 10000,
      risk_per_trade: config.risk_per_trade || 0.02,
      strategy_params: config.strategy_params || {}
    };
    return this.startBot(botConfig);
  }

  async stopTrading() {
    return this.stopBot();
  }

  async getPositions() {
    return this.apiCall('/api/v1/trading/positions');
  }

  async getOrders(limit = 50, symbol = null) {
    const params = new URLSearchParams({ limit });
    if (symbol) params.append('symbol', symbol);
    return this.apiCall(`/api/v1/trading/orders?${params}`);
  }

  async createManualOrder(orderData) {
    return this.apiCall('/api/v1/trading/manual-order', {
      method: 'POST',
      body: JSON.stringify(orderData),
    });
  }

  async cancelOrder(orderId, symbol) {
    return this.apiCall(`/api/v1/trading/cancel-order/${orderId}`, {
      method: 'POST',
      body: JSON.stringify({ symbol }),
    });
  }

  async getPerformance(period = 'all') {
    return this.apiCall(`/api/v1/trading/performance?period=${period}`);
  }

  async runBacktest(backtestData) {
    return this.apiCall('/api/v1/trading/backtest', {
      method: 'POST',
      body: JSON.stringify(backtestData),
    });
  }

  // Strategy API methods
  async getStrategies() {
    return this.apiCall('/api/v1/strategies/list');
  }

  async getStrategyDetails(strategyName) {
    return this.apiCall(`/api/v1/strategies/${strategyName}`);
  }

  async validateStrategyConfig(strategyName, config) {
    return this.apiCall(`/api/v1/strategies/${strategyName}/validate`, {
      method: 'POST',
      body: JSON.stringify({ config }),
    });
  }

  async getStrategySignal(strategyName, symbol, timeframe = '1h', config = {}) {
    return this.apiCall(`/api/v1/strategies/${strategyName}/signal`, {
      method: 'POST',
      body: JSON.stringify({ symbol, timeframe, config }),
    });
  }

  async backtestStrategy(strategyName, backtestData) {
    return this.apiCall(`/api/v1/strategies/${strategyName}/backtest`, {
      method: 'POST',
      body: JSON.stringify(backtestData),
    });
  }

  async getActiveStrategies() {
    return this.apiCall('/api/v1/strategies/active');
  }

  async getStrategyPerformance(strategy = null, period = 'all') {
    const params = new URLSearchParams({ period });
    if (strategy) params.append('strategy', strategy);
    return this.apiCall(`/api/v1/strategies/performance?${params}`);
  }

  // Advanced ML and Risk Management endpoints
  async getMarketRegime() {
    try {
      return await this.apiCall('/api/v1/market/regime');
    } catch (error) {
      console.warn('Market regime endpoint not available:', error);
      return {
        regime: 'UNKNOWN',
        confidence: 0.5,
        trend_strength: 0.5,
        volatility: 0.3
      };
    }
  }

  async getPositionSizing(symbol = 'BTC/USDT') {
    try {
      return await this.apiCall(`/api/v1/risk/position-sizing?symbol=${symbol}`);
    } catch (error) {
      console.warn('Position sizing endpoint not available:', error);
      return {
        recommended_size: 0.05,
        kelly_fraction: 0.10,
        safety_factor: 0.5,
        max_risk_per_trade: 0.02
      };
    }
  }

  async getMLAnalysis(symbol = 'BTC/USDT') {
    try {
      return await this.apiCall(`/api/v1/ml/analysis?symbol=${symbol}`);
    } catch (error) {
      console.warn('ML analysis endpoint not available:', error);
      return {
        entry_signal: 'HOLD',
        exit_signal: 'HOLD',
        confidence: 0.5,
        features_count: 0
      };
    }
  }

  async getAdvancedMetrics() {
    try {
      return await this.apiCall('/api/v1/analytics/advanced');
    } catch (error) {
      console.warn('Advanced metrics endpoint not available:', error);
      return {
        sharpe_ratio: 0,
        max_drawdown: 0,
        win_rate: 0.5,
        profit_factor: 1.0
      };
    }
  }

  async getKellyOptimization(symbol = 'BTC/USDT') {
    try {
      return await this.apiCall(`/api/v1/optimization/kelly?symbol=${symbol}`);
    } catch (error) {
      console.warn('Kelly optimization endpoint not available:', error);
      return {
        optimal_size: 0.05,
        expected_return: 0.01,
        win_probability: 0.55
      };
    }
  }

  // Monitoring API methods
  async getSystemHealth() {
    return this.apiCall('/api/v1/monitoring/health');
  }

  async getSystemMetrics() {
    return this.apiCall('/api/v1/monitoring/metrics');
  }

  async getSystemLogs(level = 'INFO', limit = 100, module = null) {
    const params = new URLSearchParams({ level, limit });
    if (module) params.append('module', module);
    return this.apiCall(`/api/v1/monitoring/logs?${params}`);
  }

  async getSystemErrors(limit = 50, category = null) {
    const params = new URLSearchParams({ limit });
    if (category) params.append('category', category);
    return this.apiCall(`/api/v1/monitoring/errors?${params}`);
  }

  async getErrorStatistics() {
    return this.apiCall('/api/v1/monitoring/error-stats');
  }

  async getSafetyStatus() {
    return this.apiCall('/api/v1/monitoring/safety-status');
  }

  async getSystemAlerts() {
    return this.apiCall('/api/v1/monitoring/alerts');
  }

  async getSystemInfo() {
    return this.apiCall('/api/v1/monitoring/system-info');
  }

  // Position Management methods
  async addToPosition(symbol, amount) {
    return this.apiCall('/api/v1/trading/add-position', {
      method: 'POST',
      body: JSON.stringify({ symbol, amount }),
    });
  }

  async reducePosition(symbol, amount) {
    return this.apiCall('/api/v1/trading/reduce-position', {
      method: 'POST',
      body: JSON.stringify({ symbol, amount }),
    });
  }

  async closePosition(symbol) {
    return this.apiCall('/api/v1/trading/close-position', {
      method: 'POST',
      body: JSON.stringify({ symbol }),
    });
  }

  // Risk Management methods
  async getRiskSettings() {
    return this.apiCall('/api/v1/trading/risk-settings');
  }

  async updateRiskSettings(settings) {
    return this.apiCall('/api/v1/trading/risk-settings', {
      method: 'PUT',
      body: JSON.stringify({ settings }),
    });
  }

  // Trade History methods
  async getTradeHistory(filters = {}) {
    const params = new URLSearchParams();
    Object.entries(filters).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== '') {
        params.append(key, value);
      }
    });
    return this.apiCall(`/api/v1/trading/history?${params}`);
  }

  async exportTradeHistory(options = {}) {
    return this.apiCall('/api/v1/trading/export-history', {
      method: 'POST',
      body: JSON.stringify(options),
    });
  }

  // Performance Analytics methods
  async getPerformanceAnalytics(options = {}) {
    return this.apiCall('/api/v1/trading/analytics', {
      method: 'POST',
      body: JSON.stringify(options),
    });
  }

  // Alert Management methods
  async getAlerts() {
    return this.apiCall('/api/v1/alerts');
  }

  async getAlertSettings() {
    return this.apiCall('/api/v1/alerts/settings');
  }

  async updateAlertSettings(settings) {
    return this.apiCall('/api/v1/alerts/settings', {
      method: 'PUT',
      body: JSON.stringify(settings),
    });
  }

  async createAlert(alertData) {
    return this.apiCall('/api/v1/alerts', {
      method: 'POST',
      body: JSON.stringify(alertData),
    });
  }

  async updateAlert(alertId, updates) {
    return this.apiCall(`/api/v1/alerts/${alertId}`, {
      method: 'PUT',
      body: JSON.stringify(updates),
    });
  }

  async deleteAlert(alertId) {
    return this.apiCall(`/api/v1/alerts/${alertId}`, {
      method: 'DELETE',
    });
  }

  async acknowledgeAlert(alertId) {
    return this.apiCall(`/api/v1/alerts/${alertId}/acknowledge`, {
      method: 'POST',
    });
  }

  async dismissAlert(alertId) {
    return this.apiCall(`/api/v1/alerts/${alertId}/dismiss`, {
      method: 'POST',
    });
  }

  async testAlert(alertId) {
    return this.apiCall(`/api/v1/alerts/${alertId}/test`, {
      method: 'POST',
    });
  }

  async pauseTrading() {
    return this.apiCall('/api/v1/trading/pause', {
      method: 'POST',
    });
  }

  // Utility methods
  isAuthenticated() {
    return !!this.token;
  }

  getToken() {
    return this.token;
  }

  setToken(token) {
    this.token = token;
    localStorage.setItem('auth_token', token);
    
    // Trigger authentication change event
    window.dispatchEvent(new CustomEvent('authChange'));
  }
}

export default new ApiService();