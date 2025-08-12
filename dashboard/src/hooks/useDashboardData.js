/**
 * Dashboard Data Hook
 * ===================
 * 
 * React hook for integrating existing dashboard components with real backend data.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import janicsDashboardAPI from '../services/JanicsDashboardAPI';

/**
 * Main dashboard data hook
 */
export const useDashboardData = (options = {}) => {
    const {
        refreshInterval = 5000,  // 5 seconds
        enableWebSocket = true,
        autoStart = true
    } = options;

    const [data, setData] = useState({
        headerStatus: {},
        activeTrades: { trades: [], active_count: 0 },
        wealthData: {},
        botIntelligence: {},
        strategySupermix: {},
        aiInsights: { ai_insights: [] }
    });

    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [lastUpdate, setLastUpdate] = useState(null);

    const wsRef = useRef(null);
    const intervalRef = useRef(null);

    /**
     * Load all dashboard data
     */
    const loadDashboardData = useCallback(async () => {
        try {
            setError(null);
            
            // Use summary endpoint for better performance
            const summary = await janicsDashboardAPI.getDashboardSummary();
            
            setData({
                headerStatus: summary.header_status || {},
                activeTrades: summary.active_trades || { trades: [], active_count: 0 },
                wealthData: summary.wealth_data || {},
                botIntelligence: summary.bot_intelligence || {},
                strategySupermix: summary.strategy_supermix || {},
                aiInsights: summary.ai_insights || { ai_insights: [] }
            });

            setLastUpdate(new Date());
            setLoading(false);

        } catch (err) {
            console.error('Failed to load dashboard data:', err);
            setError(err.message);
            setLoading(false);
        }
    }, []);

    /**
     * Start periodic data refresh
     */
    const startPeriodicRefresh = useCallback(() => {
        if (intervalRef.current) return;
        
        intervalRef.current = setInterval(loadDashboardData, refreshInterval);
    }, [loadDashboardData, refreshInterval]);

    /**
     * Stop periodic data refresh
     */
    const stopPeriodicRefresh = useCallback(() => {
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
    }, []);

    /**
     * Connect WebSocket for real-time updates
     */
    const connectWebSocket = useCallback(() => {
        if (!enableWebSocket) return;

        try {
            wsRef.current = janicsDashboardAPI.connectWebSocket(
                (wsData) => {
                    // Update specific parts of data based on WebSocket message type
                    if (wsData.type === 'status_update') {
                        setData(prev => ({
                            ...prev,
                            headerStatus: { ...prev.headerStatus, ...wsData.data }
                        }));
                    } else if (wsData.type === 'trades_update') {
                        setData(prev => ({
                            ...prev,
                            activeTrades: wsData.data
                        }));
                    } else if (wsData.type === 'portfolio_update') {
                        setData(prev => ({
                            ...prev,
                            wealthData: { ...prev.wealthData, ...wsData.data }
                        }));
                    }
                    
                    setLastUpdate(new Date());
                },
                (error) => {
                    console.warn('WebSocket error, falling back to polling:', error);
                    // Fall back to polling if WebSocket fails
                    startPeriodicRefresh();
                }
            );
        } catch (error) {
            console.warn('WebSocket connection failed, using polling:', error);
            startPeriodicRefresh();
        }
    }, [enableWebSocket, startPeriodicRefresh]);

    // Initialize on mount
    useEffect(() => {
        if (autoStart) {
            loadDashboardData();
            
            if (enableWebSocket) {
                connectWebSocket();
            } else {
                startPeriodicRefresh();
            }
        }

        return () => {
            stopPeriodicRefresh();
            if (wsRef.current) {
                wsRef.current.close();
            }
        };
    }, [autoStart, enableWebSocket, connectWebSocket, startPeriodicRefresh, stopPeriodicRefresh, loadDashboardData]);

    return {
        data,
        loading,
        error,
        lastUpdate,
        refresh: loadDashboardData,
        startRefresh: startPeriodicRefresh,
        stopRefresh: stopPeriodicRefresh
    };
};

/**
 * Hook for header status data
 */
export const useHeaderStatus = (refreshInterval = 5000) => {
    const [headerStatus, setHeaderStatus] = useState({});
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const loadHeaderStatus = useCallback(async () => {
        try {
            setError(null);
            const status = await janicsDashboardAPI.getHeaderStatus();
            setHeaderStatus(status);
            setLoading(false);
        } catch (err) {
            setError(err.message);
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        loadHeaderStatus();
        const interval = setInterval(loadHeaderStatus, refreshInterval);
        return () => clearInterval(interval);
    }, [loadHeaderStatus, refreshInterval]);

    return { headerStatus, loading, error, refresh: loadHeaderStatus };
};

/**
 * Hook for bot control actions
 */
export const useBotControl = () => {
    const [actionLoading, setActionLoading] = useState(false);
    const [actionError, setActionError] = useState(null);

    const startBot = useCallback(async (options = {}) => {
        setActionLoading(true);
        setActionError(null);
        try {
            const result = await janicsDashboardAPI.startBot(options);
            setActionLoading(false);
            return result;
        } catch (error) {
            setActionError(error.message);
            setActionLoading(false);
            throw error;
        }
    }, []);

    const stopBot = useCallback(async () => {
        setActionLoading(true);
        setActionError(null);
        try {
            const result = await janicsDashboardAPI.stopBot();
            setActionLoading(false);
            return result;
        } catch (error) {
            setActionError(error.message);
            setActionLoading(false);
            throw error;
        }
    }, []);

    const restartBot = useCallback(async (options = {}) => {
        setActionLoading(true);
        setActionError(null);
        try {
            const result = await janicsDashboardAPI.restartBot(options);
            setActionLoading(false);
            return result;
        } catch (error) {
            setActionError(error.message);
            setActionLoading(false);
            throw error;
        }
    }, []);

    return {
        startBot,
        stopBot,
        restartBot,
        actionLoading,
        actionError
    };
};

/**
 * Hook for active trades
 */
export const useActiveTrades = (refreshInterval = 10000) => {
    const [activeTrades, setActiveTrades] = useState({ trades: [], active_count: 0 });
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const loadActiveTrades = useCallback(async () => {
        try {
            setError(null);
            const trades = await janicsDashboardAPI.getActiveTrades();
            setActiveTrades(trades);
            setLoading(false);
        } catch (err) {
            setError(err.message);
            setLoading(false);
        }
    }, []);

    const closeTrade = useCallback(async (tradeId) => {
        try {
            const result = await janicsDashboardAPI.closeTrade(tradeId);
            if (result.success) {
                // Refresh trades after closing
                await loadActiveTrades();
            }
            return result;
        } catch (error) {
            console.error('Failed to close trade:', error);
            throw error;
        }
    }, [loadActiveTrades]);

    useEffect(() => {
        loadActiveTrades();
        const interval = setInterval(loadActiveTrades, refreshInterval);
        return () => clearInterval(interval);
    }, [loadActiveTrades, refreshInterval]);

    return { 
        activeTrades, 
        loading, 
        error, 
        refresh: loadActiveTrades,
        closeTrade 
    };
};

/**
 * Hook for wealth/portfolio data
 */
export const useWealthData = (refreshInterval = 5000) => {
    const [wealthData, setWealthData] = useState({});
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const loadWealthData = useCallback(async () => {
        try {
            setError(null);
            const wealth = await janicsDashboardAPI.getWealthData();
            setWealthData(wealth);
            setLoading(false);
        } catch (err) {
            setError(err.message);
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        loadWealthData();
        const interval = setInterval(loadWealthData, refreshInterval);
        return () => clearInterval(interval);
    }, [loadWealthData, refreshInterval]);

    return { wealthData, loading, error, refresh: loadWealthData };
};

/**
 * Hook for bot intelligence data
 */
export const useBotIntelligence = (refreshInterval = 10000) => {
    const [botIntelligence, setBotIntelligence] = useState({});
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const loadBotIntelligence = useCallback(async () => {
        try {
            setError(null);
            const intelligence = await janicsDashboardAPI.getBotIntelligence();
            setBotIntelligence(intelligence);
            setLoading(false);
        } catch (err) {
            setError(err.message);
            setLoading(false);
        }
    }, []);

    const triggerLearning = useCallback(async () => {
        try {
            const result = await janicsDashboardAPI.triggerLearning();
            if (result.success) {
                // Refresh intelligence data after triggering learning
                await loadBotIntelligence();
            }
            return result;
        } catch (error) {
            console.error('Failed to trigger learning:', error);
            throw error;
        }
    }, [loadBotIntelligence]);

    useEffect(() => {
        loadBotIntelligence();
        const interval = setInterval(loadBotIntelligence, refreshInterval);
        return () => clearInterval(interval);
    }, [loadBotIntelligence, refreshInterval]);

    return { 
        botIntelligence, 
        loading, 
        error, 
        refresh: loadBotIntelligence,
        triggerLearning
    };
};

/**
 * Hook for strategy supermix data
 */
export const useStrategySupermix = (refreshInterval = 15000) => {
    const [strategySupermix, setStrategySupermix] = useState({});
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const loadStrategySupermix = useCallback(async () => {
        try {
            setError(null);
            const supermix = await janicsDashboardAPI.getStrategySupermix();
            setStrategySupermix(supermix);
            setLoading(false);
        } catch (err) {
            setError(err.message);
            setLoading(false);
        }
    }, []);

    const controlRiskTier = useCallback(async (riskLevel, action) => {
        try {
            let result;
            if (action === 'start') {
                result = await janicsDashboardAPI.startRiskTier(riskLevel);
            } else if (action === 'stop') {
                result = await janicsDashboardAPI.stopRiskTier(riskLevel);
            }
            
            if (result.success) {
                // Refresh data after control action
                await loadStrategySupermix();
            }
            return result;
        } catch (error) {
            console.error(`Failed to ${action} risk tier:`, error);
            throw error;
        }
    }, [loadStrategySupermix]);

    useEffect(() => {
        loadStrategySupermix();
        const interval = setInterval(loadStrategySupermix, refreshInterval);
        return () => clearInterval(interval);
    }, [loadStrategySupermix, refreshInterval]);

    return { 
        strategySupermix, 
        loading, 
        error, 
        refresh: loadStrategySupermix,
        controlRiskTier
    };
};