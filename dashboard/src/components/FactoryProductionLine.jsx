/**
 * 🏭 FACTORY PRODUCTION LINE 🏭
 * Live trading positions displayed as factory production units!
 */

import React, { useState, useEffect } from 'react';
import { useActiveTrades } from '../hooks/useDashboardData';
import '../styles/JanicsFactory.css';

const FactoryProductionLine = () => {
    const { activeTrades, loading, error, closeTrade } = useActiveTrades();
    const [conveyorSpeed, setConveyorSpeed] = useState(1);
    const [productionSteam, setProductionSteam] = useState([]);

    // Generate steam effects for active production
    useEffect(() => {
        if (activeTrades.active_count > 0) {
            const steam = Array.from({ length: activeTrades.active_count * 2 }, (_, i) => ({
                id: i,
                x: Math.random() * 100,
                delay: Math.random() * 2000,
                size: Math.random() * 30 + 20
            }));
            setProductionSteam(steam);
        }
    }, [activeTrades.active_count]);

    // Adjust conveyor speed based on profit performance
    useEffect(() => {
        const totalPnl = activeTrades.summary?.total_pnl || 0;
        if (totalPnl > 50) {
            setConveyorSpeed(2);
        } else if (totalPnl > 20) {
            setConveyorSpeed(1.5);
        } else {
            setConveyorSpeed(1);
        }
    }, [activeTrades.summary?.total_pnl]);

    const handleCloseProduction = async (tradeId, symbol) => {
        if (window.confirm(`🏭 Stop production for ${symbol}?`)) {
            try {
                const result = await closeTrade(tradeId);
                if (result.success) {
                    // Show success notification
                    showProductionAlert('success', `✅ Production stopped for ${symbol}`);
                } else {
                    showProductionAlert('error', `❌ Failed to stop production: ${result.message}`);
                }
            } catch (error) {
                showProductionAlert('error', `⚠️ Production stop error: ${error.message}`);
            }
        }
    };

    const showProductionAlert = (type, message) => {
        // Create temporary alert element
        const alert = document.createElement('div');
        alert.className = `production-alert ${type}`;
        alert.textContent = message;
        document.body.appendChild(alert);
        
        setTimeout(() => {
            alert.remove();
        }, 3000);
    };

    if (loading) {
        return (
            <div className="production-line loading">
                <div className="loading-spinner"></div>
                <div className="loading-text">Starting Production Lines...</div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="production-line error">
                <div className="error-icon">🏭💥</div>
                <div className="error-text">Production Line Error: {error}</div>
            </div>
        );
    }

    return (
        <div className="factory-production-line">
            {/* Production Steam Effects */}
            <div className="production-steam-container">
                {productionSteam.map(steam => (
                    <div
                        key={steam.id}
                        className="production-steam-particle"
                        style={{
                            left: `${steam.x}%`,
                            animationDelay: `${steam.delay}ms`,
                            width: `${steam.size}px`,
                            height: `${steam.size}px`
                        }}
                    />
                ))}
            </div>

            {/* Production Line Header */}
            <div className="production-line-header">
                <div className="header-title">
                    <span className="title-icon">🔧</span>
                    <span className="neon-text">ACTIVE PRODUCTION LINES</span>
                    <span className="title-icon">🔧</span>
                </div>
                
                <div className="production-status">
                    <div className="status-indicator">
                        <span className="status-count">{activeTrades.active_count}</span>
                        <span className="status-label">Lines Running</span>
                    </div>
                    <div className="status-indicator">
                        <span className="status-speed" style={{ animation: `pulse-green ${2/conveyorSpeed}s infinite` }}>
                            {conveyorSpeed.toFixed(1)}x
                        </span>
                        <span className="status-label">Production Speed</span>
                    </div>
                </div>
            </div>

            {/* Main Production Conveyor */}
            <div className="production-conveyor-container">
                {activeTrades.trades && activeTrades.trades.length > 0 ? (
                    <div className="production-conveyor">
                        {activeTrades.trades.map((trade, index) => (
                            <ProductionUnit
                                key={trade.id}
                                trade={trade}
                                conveyorSpeed={conveyorSpeed}
                                onClose={() => handleCloseProduction(trade.id, trade.symbol)}
                                animationDelay={index * 0.2}
                            />
                        ))}
                    </div>
                ) : (
                    <div className="no-production">
                        <div className="no-production-icon">🏭</div>
                        <div className="no-production-text">
                            <div className="main-text">Factory Ready for Production</div>
                            <div className="sub-text">No active production lines running</div>
                        </div>
                    </div>
                )}
            </div>

            {/* Production Summary Dashboard */}
            <div className="production-summary-dashboard">
                <div className="summary-title">
                    <span className="title-icon">📊</span>
                    <span>PRODUCTION SUMMARY</span>
                </div>
                
                <div className="summary-metrics">
                    <ProductionMetric
                        icon="💰"
                        label="Total Output"
                        value={`$${(activeTrades.summary?.total_pnl || 0).toFixed(2)}`}
                        color={activeTrades.summary?.total_pnl >= 0 ? '#00ff88' : '#ff4757'}
                        trend={activeTrades.summary?.total_pnl >= 0 ? 'up' : 'down'}
                    />
                    
                    <ProductionMetric
                        icon="🎯"
                        label="Success Rate"
                        value={`${activeTrades.summary?.winning_trades || 0}/${activeTrades.active_count}`}
                        color="#ffa500"
                        trend="up"
                    />
                    
                    <ProductionMetric
                        icon="📈"
                        label="Performance"
                        value={`${(activeTrades.summary?.total_pnl_percentage || 0).toFixed(2)}%`}
                        color={activeTrades.summary?.total_pnl_percentage >= 0 ? '#00ff88' : '#ff4757'}
                        trend={activeTrades.summary?.total_pnl_percentage >= 0 ? 'up' : 'down'}
                    />
                    
                    <ProductionMetric
                        icon="⚡"
                        label="Efficiency"
                        value={activeTrades.active_count > 0 ? 'High' : 'Idle'}
                        color={activeTrades.active_count > 0 ? '#00ff88' : '#6c757d'}
                        trend={activeTrades.active_count > 0 ? 'up' : 'neutral'}
                    />
                </div>
            </div>
        </div>
    );
};

/**
 * Individual Production Unit Component
 */
const ProductionUnit = ({ trade, conveyorSpeed, onClose, animationDelay }) => {
    const [isHovered, setIsHovered] = useState(false);
    
    const getProfitColor = () => {
        return trade.pnl >= 0 ? '#00ff88' : '#ff4757';
    };

    const getProductionStatus = () => {
        if (trade.pnl > 0) return 'profit';
        if (trade.pnl < 0) return 'loss';
        return 'neutral';
    };

    const formatDuration = (seconds) => {
        if (seconds < 60) return `${seconds}s`;
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
        return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
    };

    return (
        <div 
            className={`production-unit ${getProductionStatus()}`}
            style={{
                animationDelay: `${animationDelay}s`,
                animationDuration: `${3 / conveyorSpeed}s`
            }}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
        >
            {/* Conveyor Belt */}
            <div 
                className="conveyor-belt"
                style={{
                    animationDuration: `${2 / conveyorSpeed}s`
                }}
            >
                {/* Production Item */}
                <div className="production-item">
                    {/* Product Symbol */}
                    <div className="product-header">
                        <div className="product-symbol">{trade.symbol}</div>
                        <div className={`product-side ${trade.side.toLowerCase()}`}>
                            {trade.side}
                        </div>
                    </div>

                    {/* Production Display */}
                    <div className="production-display">
                        <div className="production-visual">
                            {trade.pnl >= 0 ? '💰' : '⚠️'}
                        </div>
                        
                        <div className="production-profit">
                            <span 
                                className="profit-amount"
                                style={{ color: getProfitColor() }}
                            >
                                {trade.pnl >= 0 ? '+' : ''}${trade.pnl.toFixed(2)}
                            </span>
                            <span className="profit-percentage">
                                ({trade.pnl_percentage >= 0 ? '+' : ''}{trade.pnl_percentage.toFixed(2)}%)
                            </span>
                        </div>
                    </div>

                    {/* Production Details */}
                    <div className="production-details">
                        <div className="detail-row">
                            <span className="detail-label">Entry:</span>
                            <span className="detail-value">${trade.entry_price.toFixed(2)}</span>
                        </div>
                        <div className="detail-row">
                            <span className="detail-label">Current:</span>
                            <span className="detail-value">${trade.current_price.toFixed(2)}</span>
                        </div>
                        <div className="detail-row">
                            <span className="detail-label">Size:</span>
                            <span className="detail-value">{trade.size}</span>
                        </div>
                        <div className="detail-row">
                            <span className="detail-label">Runtime:</span>
                            <span className="detail-value">{formatDuration(trade.duration)}</span>
                        </div>
                    </div>

                    {/* Factory Strategy */}
                    <div className="production-strategy">
                        <span className="strategy-icon">🏭</span>
                        <span className="strategy-name">{trade.strategy.replace('_', ' ').toUpperCase()}</span>
                    </div>

                    {/* Production Controls */}
                    {isHovered && (
                        <div className="production-controls">
                            <button 
                                className="factory-button danger small"
                                onClick={onClose}
                                title="Stop Production"
                            >
                                🛑 STOP
                            </button>
                        </div>
                    )}
                </div>
            </div>

            {/* Production Glow Effect */}
            <div 
                className={`production-glow ${getProductionStatus()}`}
                style={{ opacity: isHovered ? 1 : 0.5 }}
            />
        </div>
    );
};

/**
 * Production Metric Component
 */
const ProductionMetric = ({ icon, label, value, color, trend }) => {
    const getTrendIcon = () => {
        switch (trend) {
            case 'up': return '📈';
            case 'down': return '📉';
            default: return '➡️';
        }
    };

    return (
        <div className="production-metric">
            <div className="metric-icon">{icon}</div>
            <div className="metric-content">
                <div 
                    className="metric-value"
                    style={{ color, textShadow: `0 0 10px ${color}` }}
                >
                    {value}
                </div>
                <div className="metric-label">{label}</div>
            </div>
            <div className="metric-trend">{getTrendIcon()}</div>
        </div>
    );
};

export default FactoryProductionLine;