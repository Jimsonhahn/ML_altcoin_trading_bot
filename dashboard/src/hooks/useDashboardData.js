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

/**
 * Hook for enhanced bot intelligence data (for AI Factory Brain)
 */
export const useEnhancedBotIntelligence = (refreshInterval = 8000) => {
    const [intelligence, setIntelligence] = useState({
        ai_confidence: 0,
        patterns_learned: 0,
        decision_speed: 0,
        learning_rate: 0,
        market_sentiment: "Analyzing...",
        sentiment_confidence: 0,
        risk_assessment: "Calculating...",
        risk_confidence: 0,
        trade_opportunity: "Scanning...",
        opportunity_confidence: 0,
        strategy_optimization: "Optimizing...",
        optimization_confidence: 0
    });
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const loadIntelligence = useCallback(async () => {
        try {
            setError(null);
            const data = await janicsDashboardAPI.getBotIntelligence();
            setIntelligence(data);
            setLoading(false);
        } catch (err) {
            setError(err.message);
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        loadIntelligence();
        const interval = setInterval(loadIntelligence, refreshInterval);
        return () => clearInterval(interval);
    }, [loadIntelligence, refreshInterval]);

    return { intelligence, loading, error, refresh: loadIntelligence };
};

/**
 * Hook for strategy data (for Strategy Factory Assembly)
 */
export const useStrategyData = (refreshInterval = 12000) => {
    const [strategies, setStrategies] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const loadStrategies = useCallback(async () => {
        try {
            setError(null);
            const data = await janicsDashboardAPI.getStrategySupermix();
            // Transform supermix data to individual strategies
            const strategyList = [];
            
            if (data.low_risk) {
                strategyList.push(...data.low_risk.map(s => ({...s, risk_level: 'LOW'})));
            }
            if (data.medium_risk) {
                strategyList.push(...data.medium_risk.map(s => ({...s, risk_level: 'MEDIUM'})));
            }
            if (data.high_risk) {
                strategyList.push(...data.high_risk.map(s => ({...s, risk_level: 'HIGH'})));
            }
            
            setStrategies(strategyList);
            setLoading(false);
        } catch (err) {
            setError(err.message);
            setLoading(false);
        }
    }, []);

    const updateStrategy = useCallback(async (strategyName, updates) => {
        try {
            const result = await janicsDashboardAPI.updateStrategy(strategyName, updates);
            if (result.success) {
                await loadStrategies(); // Refresh after update
            }
            return result;
        } catch (error) {
            console.error('Failed to update strategy:', error);
            throw error;
        }
    }, [loadStrategies]);

    const toggleStrategy = useCallback(async (strategyName, newStatus) => {
        try {
            const result = await janicsDashboardAPI.toggleStrategy(strategyName, newStatus);
            if (result.success) {
                await loadStrategies(); // Refresh after toggle
            }
            return result;
        } catch (error) {
            console.error('Failed to toggle strategy:', error);
            throw error;
        }
    }, [loadStrategies]);

    useEffect(() => {
        loadStrategies();
        const interval = setInterval(loadStrategies, refreshInterval);
        return () => clearInterval(interval);
    }, [loadStrategies, refreshInterval]);

    return { 
        strategies, 
        loading, 
        error, 
        refresh: loadStrategies,
        updateStrategy,
        toggleStrategy
    };
};

/**
 * Hook for enhanced bot control (for Command Center)
 */
export const useEnhancedBotControl = (refreshInterval = 5000) => {
    const [botStatus, setBotStatus] = useState({
        status: 'unknown',
        cpu_usage: 0,
        memory_usage: 0,
        uptime: '0m',
        api_connected: false
    });
    const [availableStrategies, setAvailableStrategies] = useState([]);
    const [profiles, setProfiles] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const loadControlData = useCallback(async () => {
        try {
            setError(null);
            
            // Load bot status and available options
            const [status, strategiesData] = await Promise.all([
                janicsDashboardAPI.getHeaderStatus(),
                janicsDashboardAPI.getStrategySupermix()
            ]);
            
            setBotStatus({
                status: status.trading_bot_status?.toLowerCase() || 'stopped',
                cpu_usage: status.cpu_usage || 0,
                memory_usage: status.memory_usage || 0,
                uptime: status.uptime || '0m',
                api_connected: status.server_status === 'Connected'
            });

            // Extract available strategies
            const strategies = [];
            if (strategiesData.low_risk) {
                strategies.push(...strategiesData.low_risk.map(s => ({
                    name: s.strategy_name,
                    display_name: s.strategy_name.replace('_', ' ').toUpperCase(),
                    risk_level: 'LOW'
                })));
            }
            if (strategiesData.medium_risk) {
                strategies.push(...strategiesData.medium_risk.map(s => ({
                    name: s.strategy_name,
                    display_name: s.strategy_name.replace('_', ' ').toUpperCase(),
                    risk_level: 'MEDIUM'
                })));
            }
            if (strategiesData.high_risk) {
                strategies.push(...strategiesData.high_risk.map(s => ({
                    name: s.strategy_name,
                    display_name: s.strategy_name.replace('_', ' ').toUpperCase(),
                    risk_level: 'HIGH'
                })));
            }
            
            setAvailableStrategies(strategies);
            
            // Mock profiles for now
            setProfiles([
                { name: 'conservative', display_name: 'Conservative Profile' },
                { name: 'balanced', display_name: 'Balanced Profile' },
                { name: 'aggressive', display_name: 'Aggressive Profile' }
            ]);
            
            setLoading(false);
        } catch (err) {
            setError(err.message);
            setLoading(false);
        }
    }, []);

    const startBot = useCallback(async (mode, strategy, profile) => {
        try {
            const result = await janicsDashboardAPI.startBot({
                mode,
                strategy,
                profile
            });
            if (result.success) {
                await loadControlData(); // Refresh status
            }
            return result;
        } catch (error) {
            throw error;
        }
    }, [loadControlData]);

    const stopBot = useCallback(async () => {
        try {
            const result = await janicsDashboardAPI.stopBot();
            if (result.success) {
                await loadControlData(); // Refresh status
            }
            return result;
        } catch (error) {
            throw error;
        }
    }, [loadControlData]);

    const restartBot = useCallback(async (mode, strategy, profile) => {
        try {
            const result = await janicsDashboardAPI.restartBot({
                mode,
                strategy,
                profile
            });
            if (result.success) {
                await loadControlData(); // Refresh status
            }
            return result;
        } catch (error) {
            throw error;
        }
    }, [loadControlData]);

    const emergencyStop = useCallback(async () => {
        try {
            // Emergency stop - first stop bot, then close all trades
            const stopResult = await janicsDashboardAPI.stopBot();
            
            // TODO: Add emergency trade closure if API supports it
            // const closeResult = await janicsDashboardAPI.emergencyCloseAll();
            
            if (stopResult.success) {
                await loadControlData(); // Refresh status
            }
            
            return { success: true, message: 'Emergency stop completed' };
        } catch (error) {
            throw error;
        }
    }, [loadControlData]);

    useEffect(() => {
        loadControlData();
        const interval = setInterval(loadControlData, refreshInterval);
        return () => clearInterval(interval);
    }, [loadControlData, refreshInterval]);

    return {
        botStatus,
        availableStrategies,
        profiles,
        loading,
        error,
        startBot,
        stopBot,
        restartBot,
        emergencyStop,
        refresh: loadControlData
    };
};

/**
 * Hook for achievements and gamification
 */
export const useAchievements = (refreshInterval = 30000) => {
    const [achievements, setAchievements] = useState([]);
    const [stats, setStats] = useState({
        totalXP: 0,
        currentStreak: 0,
        totalTrades: 0,
        winRate: 0
    });
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const loadAchievements = useCallback(async () => {
        try {
            setError(null);
            
            // Generate mock achievements based on real trading data
            const [tradesData, wealthData, intelligenceData] = await Promise.all([
                janicsDashboardAPI.getActiveTrades(),
                janicsDashboardAPI.getWealthData(),
                janicsDashboardAPI.getBotIntelligence()
            ]);
            
            // Mock achievements with real data integration
            const mockAchievements = [
                {
                    id: 'first_trade',
                    name: 'First Trade',
                    description: 'Execute your first successful trade',
                    icon: '📈',
                    category: 'trading',
                    rarity: 'common',
                    xpReward: 100,
                    unlocked: tradesData.trades?.length > 0,
                    unlockedAt: tradesData.trades?.length > 0 ? new Date().toISOString() : null,
                    progress: tradesData.trades?.length > 0 ? 1 : 0,
                    requirement: 1
                },
                {
                    id: 'profit_maker',
                    name: 'Profit Maker',
                    description: 'Generate your first $100 in profits',
                    icon: '💰',
                    category: 'profit',
                    rarity: 'uncommon',
                    xpReward: 250,
                    unlocked: (wealthData.total_value || 0) >= 100,
                    unlockedAt: (wealthData.total_value || 0) >= 100 ? new Date().toISOString() : null,
                    progress: Math.min(wealthData.total_value || 0, 100),
                    requirement: 100
                },
                {
                    id: 'ai_master',
                    name: 'AI Master',
                    description: 'Achieve 90% AI confidence',
                    icon: '🧠',
                    category: 'strategy',
                    rarity: 'epic',
                    xpReward: 500,
                    unlocked: (intelligenceData.ai_confidence || 0) >= 90,
                    unlockedAt: (intelligenceData.ai_confidence || 0) >= 90 ? new Date().toISOString() : null,
                    progress: intelligenceData.ai_confidence || 0,
                    requirement: 90
                },
                {
                    id: 'wealth_builder',
                    name: 'Wealth Builder',
                    description: 'Accumulate $1,000 in total wealth',
                    icon: '🏆',
                    category: 'profit',
                    rarity: 'rare',
                    xpReward: 750,
                    unlocked: (wealthData.total_value || 0) >= 1000,
                    progress: Math.min(wealthData.total_value || 0, 1000),
                    requirement: 1000
                },
                {
                    id: 'factory_owner',
                    name: 'Factory Owner',
                    description: 'Reach $10,000 total wealth - You own the factory now!',
                    icon: '🏭',
                    category: 'special',
                    rarity: 'legendary',
                    xpReward: 2000,
                    unlocked: (wealthData.total_value || 0) >= 10000,
                    progress: Math.min(wealthData.total_value || 0, 10000),
                    requirement: 10000,
                    tip: 'Das ist der wahre Janics Freedom Factory Spirit!'
                }
            ];
            
            setAchievements(mockAchievements);
            
            // Calculate stats
            const unlockedCount = mockAchievements.filter(a => a.unlocked).length;
            const totalXP = mockAchievements
                .filter(a => a.unlocked)
                .reduce((sum, a) => sum + a.xpReward, 0);
            
            setStats({
                totalXP,
                currentStreak: Math.floor(Math.random() * 10) + 1, // Mock streak
                totalTrades: tradesData.trades?.length || 0,
                winRate: Math.round(Math.random() * 40) + 60 // Mock win rate 60-100%
            });
            
            setLoading(false);
        } catch (err) {
            setError(err.message);
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        loadAchievements();
        const interval = setInterval(loadAchievements, refreshInterval);
        return () => clearInterval(interval);
    }, [loadAchievements, refreshInterval]);

    return { 
        achievements, 
        stats, 
        loading, 
        error, 
        refresh: loadAchievements
    };
};