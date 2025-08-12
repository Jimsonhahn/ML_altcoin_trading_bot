/**
 * Integrated Dashboard Example
 * ============================
 * 
 * Example showing how to integrate existing dashboard UI with real backend functionality.
 */

import React from 'react';
import { 
    useDashboardData, 
    useHeaderStatus, 
    useBotControl, 
    useActiveTrades,
    useWealthData,
    useBotIntelligence,
    useStrategySupermix 
} from '../hooks/useDashboardData';

/**
 * Enhanced Header Component with Real Status
 */
export const EnhancedHeader = () => {
    const { headerStatus, loading, error } = useHeaderStatus();

    if (loading) return <div>Loading header status...</div>;
    if (error) return <div>Error: {error}</div>;

    return (
        <header className="dashboard-header">
            {/* Factory Online Indicator */}
            <div className="status-indicator">
                <span className={`status-dot ${headerStatus.factory_online ? 'online' : 'offline'}`}></span>
                <span>Factory Online</span>
            </div>

            {/* Server Status */}
            <div className="status-indicator">
                <span>Server: </span>
                <span className={`status-text ${headerStatus.server_status?.toLowerCase()}`}>
                    {headerStatus.server_status || 'Unknown'}
                </span>
            </div>

            {/* Trading Bot Status */}
            <div className="status-indicator">
                <span>Trading Bot: </span>
                <span className={`status-text ${headerStatus.trading_bot_status?.toLowerCase()}`}>
                    {headerStatus.trading_bot_status || 'Unknown'}
                </span>
            </div>

            {/* System Status */}
            <div className="status-indicator">
                <span>System: </span>
                <span className={`status-text ${headerStatus.system_status?.toLowerCase()}`}>
                    {headerStatus.system_status || 'Unknown'}
                </span>
            </div>

            {/* Last Update */}
            <div className="last-update">
                Last Update: {headerStatus.last_update || '--:--:--'}
            </div>
        </header>
    );
};

/**
 * Enhanced Bot Control Panel with Real Functionality
 */
export const EnhancedBotControl = () => {
    const { startBot, stopBot, restartBot, actionLoading, actionError } = useBotControl();

    const handleStartBot = async () => {
        try {
            const result = await startBot({ mode: 'live' });
            if (result.success) {
                alert('Bot started successfully!');
            } else {
                alert(`Failed to start bot: ${result.message}`);
            }
        } catch (error) {
            alert(`Error: ${error.message}`);
        }
    };

    const handleStopBot = async () => {
        try {
            const result = await stopBot();
            if (result.success) {
                alert('Bot stopped successfully!');
            } else {
                alert(`Failed to stop bot: ${result.message}`);
            }
        } catch (error) {
            alert(`Error: ${error.message}`);
        }
    };

    const handleRestartBot = async () => {
        try {
            const result = await restartBot({ mode: 'live' });
            if (result.success) {
                alert('Bot restarted successfully!');
            } else {
                alert(`Failed to restart bot: ${result.message}`);
            }
        } catch (error) {
            alert(`Error: ${error.message}`);
        }
    };

    return (
        <div className="bot-control-panel">
            <h3>Bot Controls</h3>
            
            {actionError && (
                <div className="error-message">{actionError}</div>
            )}

            <div className="control-buttons">
                <button 
                    onClick={handleStartBot} 
                    disabled={actionLoading}
                    className="btn btn-success"
                >
                    {actionLoading ? 'Starting...' : 'Start Bot'}
                </button>

                <button 
                    onClick={handleStopBot} 
                    disabled={actionLoading}
                    className="btn btn-danger"
                >
                    {actionLoading ? 'Stopping...' : 'Stop Bot'}
                </button>

                <button 
                    onClick={handleRestartBot} 
                    disabled={actionLoading}
                    className="btn btn-warning"
                >
                    {actionLoading ? 'Restarting...' : 'Restart Bot'}
                </button>
            </div>
        </div>
    );
};

/**
 * Enhanced Active Trades Panel with Real Data
 */
export const EnhancedActiveTrades = () => {
    const { activeTrades, loading, error, closeTrade } = useActiveTrades();

    if (loading) return <div>Loading trades...</div>;
    if (error) return <div>Error loading trades: {error}</div>;

    const handleCloseTrade = async (tradeId) => {
        if (window.confirm('Are you sure you want to close this trade?')) {
            try {
                const result = await closeTrade(tradeId);
                if (result.success) {
                    alert('Trade closed successfully!');
                } else {
                    alert(`Failed to close trade: ${result.message}`);
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        }
    };

    return (
        <div className="active-trades-panel">
            <h3>Offene Trades - Live Positions</h3>
            <div className="trades-summary">
                <span className="active-count">{activeTrades.active_count} Active</span>
                {activeTrades.summary && (
                    <span className="total-pnl">
                        Total P&L: ${activeTrades.summary.total_pnl?.toFixed(2) || '0.00'}
                    </span>
                )}
            </div>

            <div className="trades-list">
                {activeTrades.trades?.length > 0 ? (
                    activeTrades.trades.map((trade) => (
                        <div key={trade.id} className="trade-item">
                            <div className="trade-symbol">{trade.symbol}</div>
                            <div className="trade-side">{trade.side}</div>
                            <div className="trade-size">{trade.size}</div>
                            <div className="trade-entry">${trade.entry_price}</div>
                            <div className="trade-current">${trade.current_price}</div>
                            <div className={`trade-pnl ${trade.pnl >= 0 ? 'positive' : 'negative'}`}>
                                ${trade.pnl?.toFixed(2)} ({trade.pnl_percentage?.toFixed(2)}%)
                            </div>
                            <div className="trade-strategy">{trade.strategy}</div>
                            <div className="trade-duration">{trade.duration_formatted}</div>
                            <button 
                                onClick={() => handleCloseTrade(trade.id)}
                                className="btn btn-sm btn-danger"
                            >
                                Close
                            </button>
                        </div>
                    ))
                ) : (
                    <div className="no-trades">No active trades</div>
                )}
            </div>
        </div>
    );
};

/**
 * Enhanced Wealth Accumulator with Real Data
 */
export const EnhancedWealthAccumulator = () => {
    const { wealthData, loading, error } = useWealthData();

    if (loading) return <div>Loading wealth data...</div>;
    if (error) return <div>Error loading wealth data: {error}</div>;

    return (
        <div className="wealth-accumulator-panel">
            <h3>Wealth Accumulator</h3>
            
            {/* Total Portfolio Value */}
            <div className="total-value">
                <div className="value-amount">
                    {wealthData.total_value_formatted || '$0.00'}
                </div>
                <div className="value-label">Total Portfolio</div>
            </div>

            {/* Today's Performance */}
            <div className="daily-performance">
                <div className="daily-pnl">
                    <span className="label">Today's Freedom:</span>
                    <span className={`amount ${wealthData.daily_pnl >= 0 ? 'positive' : 'negative'}`}>
                        {wealthData.daily_pnl_formatted || '$0.00'}
                    </span>
                    <span className="percentage">
                        ({wealthData.daily_pnl_percentage?.toFixed(2)}%)
                    </span>
                </div>
            </div>

            {/* Profit Streak */}
            {wealthData.profit_streak_hours > 0 && (
                <div className="profit-streak">
                    <span>Profit Streak: {wealthData.profit_streak_hours}h</span>
                    <span>Confidence Boost: {wealthData.confidence_boost_formatted}</span>
                </div>
            )}

            {/* Progress Rings */}
            <div className="progress-rings">
                <div className="progress-ring">
                    <div className="ring-label">Daily</div>
                    <div className="ring-progress">
                        <div className="progress-percentage">
                            {wealthData.daily_progress?.percentage?.toFixed(0)}%
                        </div>
                        <div className="progress-amount">
                            {wealthData.daily_progress?.formatted}
                        </div>
                    </div>
                </div>

                <div className="progress-ring">
                    <div className="ring-label">Weekly</div>
                    <div className="ring-progress">
                        <div className="progress-percentage">
                            {wealthData.weekly_progress?.percentage?.toFixed(0)}%
                        </div>
                        <div className="progress-amount">
                            {wealthData.weekly_progress?.formatted}
                        </div>
                    </div>
                </div>

                <div className="progress-ring">
                    <div className="ring-label">Monthly</div>
                    <div className="ring-progress">
                        <div className="progress-percentage">
                            {wealthData.monthly_progress?.percentage?.toFixed(0)}%
                        </div>
                        <div className="progress-amount">
                            {wealthData.monthly_progress?.formatted}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

/**
 * Enhanced Bot Intelligence Panel
 */
export const EnhancedBotIntelligence = () => {
    const { botIntelligence, loading, error, triggerLearning } = useBotIntelligence();

    if (loading) return <div>Loading bot intelligence...</div>;
    if (error) return <div>Error loading bot intelligence: {error}</div>;

    const handleTriggerLearning = async () => {
        try {
            const result = await triggerLearning();
            if (result.success) {
                alert('Learning cycle triggered successfully!');
            }
        } catch (error) {
            alert(`Error: ${error.message}`);
        }
    };

    return (
        <div className="bot-intelligence-panel">
            <h3>Janics Bot Status</h3>

            {/* Bot Avatar & Mode */}
            <div className="bot-avatar">
                <div className="avatar-image">🤖</div>
                <div className="bot-mode">{botIntelligence.mode || 'IDLE MODE'}</div>
                <div className="bot-activity">{botIntelligence.activity}</div>
            </div>

            {/* Confidence Metrics */}
            <div className="confidence-metrics">
                <div className="metric">
                    <div className="metric-label">Technical Analysis</div>
                    <div className="metric-bar">
                        <div 
                            className="metric-fill" 
                            style={{width: `${botIntelligence.technical_analysis || 0}%`}}
                        ></div>
                    </div>
                    <div className="metric-value">{botIntelligence.technical_analysis || 0}%</div>
                </div>

                <div className="metric">
                    <div className="metric-label">Market Sentiment</div>
                    <div className="metric-bar">
                        <div 
                            className="metric-fill" 
                            style={{width: `${botIntelligence.market_sentiment || 0}%`}}
                        ></div>
                    </div>
                    <div className="metric-value">{botIntelligence.market_sentiment || 0}%</div>
                </div>

                <div className="metric">
                    <div className="metric-label">Volume Analysis</div>
                    <div className="metric-bar">
                        <div 
                            className="metric-fill" 
                            style={{width: `${botIntelligence.volume_analysis || 0}%`}}
                        ></div>
                    </div>
                    <div className="metric-value">{botIntelligence.volume_analysis || 0}%</div>
                </div>
            </div>

            {/* Learning Status */}
            <div className="learning-status">
                <div className="status-text">{botIntelligence.learning_status}</div>
                <div className="last-learning">{botIntelligence.last_learning_event}</div>
                <button onClick={handleTriggerLearning} className="btn btn-sm btn-primary">
                    Trigger Learning
                </button>
            </div>
        </div>
    );
};

/**
 * Enhanced Strategy Supermix Panel
 */
export const EnhancedStrategySupermix = () => {
    const { strategySupermix, loading, error, controlRiskTier } = useStrategySupermix();

    if (loading) return <div>Loading strategy data...</div>;
    if (error) return <div>Error loading strategy data: {error}</div>;

    const handleTierControl = async (riskLevel, action) => {
        try {
            const result = await controlRiskTier(riskLevel, action);
            if (result.success) {
                alert(`${action} ${riskLevel} risk tier successfully!`);
            }
        } catch (error) {
            alert(`Error: ${error.message}`);
        }
    };

    return (
        <div className="strategy-supermix-panel">
            <h3>Risk-Tiered Strategy Supermix</h3>
            
            <div className="supermix-status">
                <div className="status-indicator">
                    Status: <span className={strategySupermix.status?.toLowerCase()}>{strategySupermix.status}</span>
                </div>
                <div className="execution-mode">{strategySupermix.execution_mode}</div>
                <div className="total-pnl">
                    Total P&L: {strategySupermix.total_pnl_formatted || '$0.00'}
                </div>
            </div>

            {/* Risk Tiers */}
            <div className="risk-tiers">
                {['high_risk', 'medium_risk', 'low_risk'].map((tierKey) => {
                    const tier = strategySupermix[tierKey];
                    if (!tier) return null;

                    return (
                        <div key={tierKey} className="risk-tier">
                            <div className="tier-header">
                                <h4>{tier.risk_level} Risk ({tier.allocation})</h4>
                                <div className="tier-pnl">{tier.current_pnl_formatted}</div>
                            </div>
                            
                            <div className="tier-strategies">
                                <div className="active-count">
                                    {tier.strategies_count} Active Strategies
                                </div>
                                
                                <div className="strategy-list">
                                    {tier.active_strategies?.map((strategy, index) => (
                                        <div key={index} className="strategy-item">
                                            <span>{strategy.name}</span>
                                            <span>{strategy.pnl_formatted}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            <div className="tier-controls">
                                <button 
                                    onClick={() => handleTierControl(tierKey.replace('_risk', ''), 'start')}
                                    className="btn btn-sm btn-success"
                                >
                                    Start
                                </button>
                                <button 
                                    onClick={() => handleTierControl(tierKey.replace('_risk', ''), 'stop')}
                                    className="btn btn-sm btn-danger"
                                >
                                    Stop
                                </button>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

/**
 * Complete Integrated Dashboard
 */
export const IntegratedDashboard = () => {
    const { data, loading, error, lastUpdate, refresh } = useDashboardData();

    if (loading) return <div className="dashboard-loading">Loading Janics Freedom Factory...</div>;
    if (error) return <div className="dashboard-error">Error: {error}</div>;

    return (
        <div className="integrated-dashboard">
            {/* Header with real status */}
            <EnhancedHeader />

            <div className="dashboard-content">
                {/* Bot Controls */}
                <div className="dashboard-section">
                    <EnhancedBotControl />
                </div>

                {/* Main Dashboard Panels */}
                <div className="dashboard-panels">
                    <EnhancedActiveTrades />
                    <EnhancedWealthAccumulator />
                    <EnhancedBotIntelligence />
                    <EnhancedStrategySupermix />
                </div>

                {/* Factory Controls */}
                <div className="factory-controls">
                    <button onClick={refresh} className="btn btn-primary">
                        🔄 Refresh Factory
                    </button>
                    <div className="last-update">
                        Last Update: {lastUpdate?.toLocaleTimeString() || 'Never'}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default IntegratedDashboard;