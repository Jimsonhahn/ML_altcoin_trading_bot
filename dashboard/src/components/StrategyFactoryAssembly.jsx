/**
 * 🏭 STRATEGY FACTORY ASSEMBLY LINE 🏭
 * Advanced strategy management with risk zones and performance tracking!
 */

import React, { useState, useEffect } from 'react';
import { useStrategyData } from '../hooks/useDashboardData';
import '../styles/JanicsFactory.css';

const StrategyFactoryAssembly = () => {
    const { strategies, loading, error, updateStrategy, toggleStrategy } = useStrategyData();
    const [assemblySpeed, setAssemblySpeed] = useState(1);
    const [strategyParticles, setStrategyParticles] = useState([]);
    const [riskAlerts, setRiskAlerts] = useState([]);

    // Generate strategy particles for active strategies
    useEffect(() => {
        const activeStrategies = strategies.filter(s => s.status === 'active');
        if (activeStrategies.length > 0) {
            const particles = Array.from({ length: activeStrategies.length * 3 }, (_, i) => ({
                id: i,
                x: Math.random() * 100,
                y: Math.random() * 100,
                delay: Math.random() * 2000,
                symbol: ['⚙️', '🔧', '🎯', '🚀'][Math.floor(Math.random() * 4)],
                size: Math.random() * 15 + 15
            }));
            setStrategyParticles(particles);
        }
    }, [strategies]);

    // Monitor risk levels and generate alerts
    useEffect(() => {
        const highRiskStrategies = strategies.filter(s => s.risk_level === 'HIGH' && s.status === 'active');
        if (highRiskStrategies.length > 0) {
            const alerts = highRiskStrategies.map(s => ({
                id: s.strategy_name,
                message: `⚠️ HIGH RISK: ${s.strategy_name} - ${s.performance}% return`,
                severity: 'warning',
                timestamp: new Date().toLocaleTimeString()
            }));
            setRiskAlerts(alerts);
        }
    }, [strategies]);

    // Adjust assembly speed based on performance
    useEffect(() => {
        const avgPerformance = strategies.reduce((acc, s) => acc + (s.performance || 0), 0) / strategies.length;
        if (avgPerformance > 15) {
            setAssemblySpeed(2);
        } else if (avgPerformance > 5) {
            setAssemblySpeed(1.5);
        } else {
            setAssemblySpeed(1);
        }
    }, [strategies]);

    const handleStrategyToggle = async (strategyName, currentStatus) => {
        try {
            const result = await toggleStrategy(strategyName, currentStatus === 'active' ? 'paused' : 'active');
            if (result.success) {
                showAssemblyNotification('success', `✅ ${strategyName} ${currentStatus === 'active' ? 'paused' : 'activated'}`);
            } else {
                showAssemblyNotification('error', `❌ Failed to update ${strategyName}: ${result.message}`);
            }
        } catch (error) {
            showAssemblyNotification('error', `⚠️ Error: ${error.message}`);
        }
    };

    const handleStrategyUpdate = async (strategyName, updates) => {
        try {
            const result = await updateStrategy(strategyName, updates);
            if (result.success) {
                showAssemblyNotification('success', `✅ ${strategyName} updated successfully`);
            } else {
                showAssemblyNotification('error', `❌ Update failed: ${result.message}`);
            }
        } catch (error) {
            showAssemblyNotification('error', `⚠️ Update error: ${error.message}`);
        }
    };

    const showAssemblyNotification = (type, message) => {
        const notification = document.createElement('div');
        notification.className = `assembly-notification ${type}`;
        notification.textContent = message;
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 4000);
    };

    if (loading) {
        return (
            <div className="strategy-assembly loading">
                <div className="loading-spinner assembly-spinner"></div>
                <div className="loading-text">Initializing Strategy Assembly Lines...</div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="strategy-assembly error">
                <div className="error-icon">🏭💥</div>
                <div className="error-text">Strategy Assembly Error: {error}</div>
            </div>
        );
    }

    return (
        <div className="strategy-factory-assembly">
            {/* Strategy Particles */}
            <div className="strategy-particles-container">
                {strategyParticles.map(particle => (
                    <div
                        key={particle.id}
                        className="strategy-particle"
                        style={{
                            left: `${particle.x}%`,
                            top: `${particle.y}%`,
                            animationDelay: `${particle.delay}ms`,
                            fontSize: `${particle.size}px`
                        }}
                    >
                        {particle.symbol}
                    </div>
                ))}
            </div>

            {/* Risk Alert Banner */}
            {riskAlerts.length > 0 && (
                <div className="risk-alert-banner">
                    <div className="alert-icon pulsing-red">⚠️</div>
                    <div className="alert-content">
                        <div className="alert-title">RISK MANAGEMENT ALERT</div>
                        <div className="alert-messages">
                            {riskAlerts.map(alert => (
                                <div key={alert.id} className="alert-message">
                                    {alert.message}
                                </div>
                            ))}
                        </div>
                    </div>
                    <div className="alert-dismiss" onClick={() => setRiskAlerts([])}>
                        ✕
                    </div>
                </div>
            )}

            {/* Assembly Line Header */}
            <div className="assembly-header">
                <div className="header-title">
                    <span className="title-icon">🏭</span>
                    <span className="neon-text">STRATEGY FACTORY ASSEMBLY</span>
                    <span className="title-icon">🏭</span>
                </div>
                
                <div className="assembly-status">
                    <div className="status-metric">
                        <span className="metric-value">{strategies.filter(s => s.status === 'active').length}</span>
                        <span className="metric-label">Active Lines</span>
                    </div>
                    <div className="status-metric">
                        <span 
                            className="metric-value"
                            style={{ animation: `pulse-green ${2/assemblySpeed}s infinite` }}
                        >
                            {assemblySpeed.toFixed(1)}x
                        </span>
                        <span className="metric-label">Assembly Speed</span>
                    </div>
                    <div className="status-metric">
                        <span className="metric-value">
                            {(strategies.reduce((acc, s) => acc + (s.performance || 0), 0) / strategies.length).toFixed(1)}%
                        </span>
                        <span className="metric-label">Avg Performance</span>
                    </div>
                </div>
            </div>

            {/* Risk Zone Classifications */}
            <div className="risk-zones-container">
                <div className="zones-title">
                    <span className="zones-icon">🎯</span>
                    <span>STRATEGY RISK ZONES</span>
                </div>
                
                <div className="risk-zones">
                    <RiskZone
                        title="SAFE ZONE"
                        color="#00ff88"
                        icon="🛡️"
                        strategies={strategies.filter(s => s.risk_level === 'LOW')}
                        description="Conservative strategies with stable returns"
                    />
                    
                    <RiskZone
                        title="GROWTH ZONE"
                        color="#ffa500"
                        icon="📈"
                        strategies={strategies.filter(s => s.risk_level === 'MEDIUM')}
                        description="Balanced risk-reward strategies"
                    />
                    
                    <RiskZone
                        title="POWER ZONE"
                        color="#ff4757"
                        icon="🚀"
                        strategies={strategies.filter(s => s.risk_level === 'HIGH')}
                        description="High-risk, high-reward strategies"
                    />
                </div>
            </div>

            {/* Main Assembly Conveyor */}
            <div className="main-assembly-conveyor">
                <div className="conveyor-title">
                    <span className="conveyor-icon">🔧</span>
                    <span>ACTIVE ASSEMBLY LINES</span>
                </div>
                
                <div className="strategy-assembly-grid">
                    {strategies.map((strategy, index) => (
                        <StrategyAssemblyUnit
                            key={strategy.strategy_name}
                            strategy={strategy}
                            assemblySpeed={assemblySpeed}
                            onToggle={() => handleStrategyToggle(strategy.strategy_name, strategy.status)}
                            onUpdate={(updates) => handleStrategyUpdate(strategy.strategy_name, updates)}
                            animationDelay={index * 0.3}
                        />
                    ))}
                </div>
            </div>

            {/* Assembly Performance Dashboard */}
            <div className="assembly-performance-dashboard">
                <div className="dashboard-title">
                    <span className="dashboard-icon">📊</span>
                    <span>ASSEMBLY PERFORMANCE METRICS</span>
                </div>
                
                <div className="performance-metrics-grid">
                    <AssemblyMetric
                        icon="🎯"
                        label="Success Rate"
                        value={`${((strategies.filter(s => s.performance > 0).length / strategies.length) * 100).toFixed(1)}%`}
                        color="#00ff88"
                        trend="up"
                    />
                    
                    <AssemblyMetric
                        icon="⚡"
                        label="Total Strategies"
                        value={strategies.length}
                        color="#3742fa"
                        trend="neutral"
                    />
                    
                    <AssemblyMetric
                        icon="🚀"
                        label="Best Performer"
                        value={strategies.length > 0 ? 
                            `${Math.max(...strategies.map(s => s.performance)).toFixed(1)}%` : 
                            'N/A'
                        }
                        color="#ff6b35"
                        trend="up"
                    />
                    
                    <AssemblyMetric
                        icon="⚠️"
                        label="Risk Distribution"
                        value={`${strategies.filter(s => s.risk_level === 'HIGH').length}H / ${strategies.filter(s => s.risk_level === 'MEDIUM').length}M / ${strategies.filter(s => s.risk_level === 'LOW').length}L`}
                        color="#ffa500"
                        trend="neutral"
                    />
                </div>
            </div>
        </div>
    );
};

/**
 * Risk Zone Component
 */
const RiskZone = ({ title, color, icon, strategies, description }) => (
    <div className="risk-zone" style={{ borderColor: color }}>
        <div className="zone-header">
            <span className="zone-icon" style={{ color }}>{icon}</span>
            <span className="zone-title" style={{ color }}>{title}</span>
        </div>
        
        <div className="zone-content">
            <div className="zone-description">{description}</div>
            <div className="zone-stats">
                <div className="stat-item">
                    <span className="stat-value" style={{ color }}>{strategies.length}</span>
                    <span className="stat-label">Strategies</span>
                </div>
                <div className="stat-item">
                    <span className="stat-value" style={{ color }}>
                        {strategies.filter(s => s.status === 'active').length}
                    </span>
                    <span className="stat-label">Active</span>
                </div>
                <div className="stat-item">
                    <span className="stat-value" style={{ color }}>
                        {strategies.length > 0 ? 
                            (strategies.reduce((acc, s) => acc + s.performance, 0) / strategies.length).toFixed(1) : 
                            '0'
                        }%
                    </span>
                    <span className="stat-label">Avg Return</span>
                </div>
            </div>
        </div>
    </div>
);

/**
 * Strategy Assembly Unit Component
 */
const StrategyAssemblyUnit = ({ strategy, assemblySpeed, onToggle, onUpdate, animationDelay }) => {
    const [isExpanded, setIsExpanded] = useState(false);
    const [editMode, setEditMode] = useState(false);
    const [editValues, setEditValues] = useState({
        risk_percent: strategy.risk_percent,
        target_profit: strategy.target_profit,
        stop_loss: strategy.stop_loss
    });

    const getStatusColor = () => {
        switch (strategy.status) {
            case 'active': return '#00ff88';
            case 'paused': return '#ffa500';
            case 'error': return '#ff4757';
            default: return '#6c757d';
        }
    };

    const getRiskColor = () => {
        switch (strategy.risk_level) {
            case 'LOW': return '#00ff88';
            case 'MEDIUM': return '#ffa500';
            case 'HIGH': return '#ff4757';
            default: return '#6c757d';
        }
    };

    const getPerformanceIcon = () => {
        if (strategy.performance > 10) return '🚀';
        if (strategy.performance > 0) return '📈';
        if (strategy.performance === 0) return '➡️';
        return '📉';
    };

    const handleSaveEdit = () => {
        onUpdate(editValues);
        setEditMode(false);
    };

    return (
        <div 
            className={`strategy-assembly-unit ${strategy.status}`}
            style={{
                animationDelay: `${animationDelay}s`,
                animationDuration: `${4 / assemblySpeed}s`
            }}
        >
            {/* Assembly Conveyor Belt */}
            <div 
                className="assembly-belt"
                style={{
                    animationDuration: `${3 / assemblySpeed}s`
                }}
            >
                {/* Strategy Card */}
                <div className="strategy-card">
                    {/* Header */}
                    <div className="strategy-header">
                        <div className="strategy-info">
                            <div className="strategy-name">{strategy.strategy_name}</div>
                            <div className="strategy-type">{strategy.type}</div>
                        </div>
                        
                        <div className="strategy-controls">
                            <button
                                className={`strategy-toggle ${strategy.status}`}
                                onClick={onToggle}
                                style={{ color: getStatusColor() }}
                            >
                                {strategy.status === 'active' ? '⏸️' : '▶️'}
                            </button>
                            
                            <button
                                className="strategy-expand"
                                onClick={() => setIsExpanded(!isExpanded)}
                            >
                                {isExpanded ? '🔽' : '🔼'}
                            </button>
                        </div>
                    </div>

                    {/* Performance Display */}
                    <div className="strategy-performance">
                        <div className="performance-visual">
                            <span className="performance-icon">
                                {getPerformanceIcon()}
                            </span>
                            <div className="performance-value">
                                <span 
                                    className="performance-number"
                                    style={{ 
                                        color: strategy.performance >= 0 ? '#00ff88' : '#ff4757'
                                    }}
                                >
                                    {strategy.performance >= 0 ? '+' : ''}{strategy.performance.toFixed(2)}%
                                </span>
                                <span className="performance-period">
                                    {strategy.time_period}
                                </span>
                            </div>
                        </div>

                        <div className="risk-indicator">
                            <div 
                                className="risk-badge"
                                style={{ 
                                    backgroundColor: getRiskColor(),
                                    color: '#000'
                                }}
                            >
                                {strategy.risk_level}
                            </div>
                        </div>
                    </div>

                    {/* Expanded Details */}
                    {isExpanded && (
                        <div className="strategy-details">
                            <div className="details-tabs">
                                <button
                                    className={`tab ${!editMode ? 'active' : ''}`}
                                    onClick={() => setEditMode(false)}
                                >
                                    📊 Stats
                                </button>
                                <button
                                    className={`tab ${editMode ? 'active' : ''}`}
                                    onClick={() => setEditMode(true)}
                                >
                                    ⚙️ Settings
                                </button>
                            </div>

                            {editMode ? (
                                <div className="edit-panel">
                                    <div className="edit-group">
                                        <label>Risk %:</label>
                                        <input
                                            type="number"
                                            value={editValues.risk_percent}
                                            onChange={(e) => setEditValues(prev => ({
                                                ...prev,
                                                risk_percent: parseFloat(e.target.value)
                                            }))}
                                            min="0.1"
                                            max="10"
                                            step="0.1"
                                        />
                                    </div>
                                    
                                    <div className="edit-group">
                                        <label>Target Profit %:</label>
                                        <input
                                            type="number"
                                            value={editValues.target_profit}
                                            onChange={(e) => setEditValues(prev => ({
                                                ...prev,
                                                target_profit: parseFloat(e.target.value)
                                            }))}
                                            min="1"
                                            max="100"
                                            step="0.5"
                                        />
                                    </div>
                                    
                                    <div className="edit-group">
                                        <label>Stop Loss %:</label>
                                        <input
                                            type="number"
                                            value={editValues.stop_loss}
                                            onChange={(e) => setEditValues(prev => ({
                                                ...prev,
                                                stop_loss: parseFloat(e.target.value)
                                            }))}
                                            min="1"
                                            max="50"
                                            step="0.5"
                                        />
                                    </div>
                                    
                                    <div className="edit-actions">
                                        <button 
                                            className="factory-button success"
                                            onClick={handleSaveEdit}
                                        >
                                            💾 Save
                                        </button>
                                        <button 
                                            className="factory-button secondary"
                                            onClick={() => setEditMode(false)}
                                        >
                                            ❌ Cancel
                                        </button>
                                    </div>
                                </div>
                            ) : (
                                <div className="stats-panel">
                                    <div className="stat-row">
                                        <span className="stat-label">Total Trades:</span>
                                        <span className="stat-value">{strategy.total_trades}</span>
                                    </div>
                                    <div className="stat-row">
                                        <span className="stat-label">Win Rate:</span>
                                        <span className="stat-value">{strategy.win_rate}%</span>
                                    </div>
                                    <div className="stat-row">
                                        <span className="stat-label">Risk Per Trade:</span>
                                        <span className="stat-value">{strategy.risk_percent}%</span>
                                    </div>
                                    <div className="stat-row">
                                        <span className="stat-label">Target Profit:</span>
                                        <span className="stat-value">{strategy.target_profit}%</span>
                                    </div>
                                    <div className="stat-row">
                                        <span className="stat-label">Stop Loss:</span>
                                        <span className="stat-value">{strategy.stop_loss}%</span>
                                    </div>
                                    <div className="stat-row">
                                        <span className="stat-label">Last Update:</span>
                                        <span className="stat-value">{strategy.last_update}</span>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    {/* Status Indicator */}
                    <div 
                        className="strategy-status-indicator"
                        style={{ backgroundColor: getStatusColor() }}
                    />
                </div>
            </div>

            {/* Assembly Glow Effect */}
            <div 
                className={`assembly-glow ${strategy.status}`}
                style={{ 
                    boxShadow: `0 0 20px ${getStatusColor()}`,
                    opacity: strategy.status === 'active' ? 0.6 : 0.3
                }}
            />
        </div>
    );
};

/**
 * Assembly Metric Component
 */
const AssemblyMetric = ({ icon, label, value, color, trend }) => {
    const getTrendIcon = () => {
        switch (trend) {
            case 'up': return '📈';
            case 'down': return '📉';
            default: return '➡️';
        }
    };

    return (
        <div className="assembly-metric">
            <div className="metric-icon" style={{ color }}>{icon}</div>
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

export default StrategyFactoryAssembly;