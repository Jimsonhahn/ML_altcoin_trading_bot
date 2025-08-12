/**
 * 🎮 FACTORY CONTROL COMMAND CENTER 🎮
 * The ultimate bot control interface - start, stop, configure everything!
 */

import React, { useState, useEffect } from 'react';
import { useEnhancedBotControl as useBotControl } from '../hooks/useDashboardData';
import '../styles/JanicsFactory.css';

const FactoryCommandCenter = () => {
    const { 
        botStatus, 
        availableStrategies, 
        profiles, 
        loading, 
        error, 
        startBot, 
        stopBot, 
        restartBot,
        emergencyStop 
    } = useBotControl();
    
    const [selectedStrategy, setSelectedStrategy] = useState('');
    const [selectedProfile, setSelectedProfile] = useState('');
    const [botMode, setBotMode] = useState('live');
    const [commandHistory, setCommandHistory] = useState([]);
    const [emergencyMode, setEmergencyMode] = useState(false);
    const [controlParticles, setControlParticles] = useState([]);

    // Generate control particles for active commands
    useEffect(() => {
        if (botStatus.status === 'running') {
            const particles = Array.from({ length: 10 }, (_, i) => ({
                id: i,
                x: Math.random() * 100,
                y: Math.random() * 100,
                delay: Math.random() * 1000,
                symbol: ['⚡', '🔥', '💎', '🚀', '⭐'][Math.floor(Math.random() * 5)],
                duration: Math.random() * 3000 + 2000
            }));
            setControlParticles(particles);
        } else {
            setControlParticles([]);
        }
    }, [botStatus.status]);

    const addToCommandHistory = (command, status, message) => {
        const timestamp = new Date().toLocaleTimeString();
        const newEntry = {
            id: Date.now(),
            timestamp,
            command,
            status,
            message
        };
        
        setCommandHistory(prev => [newEntry, ...prev.slice(0, 9)]); // Keep last 10 entries
    };

    const handleStartBot = async () => {
        if (!selectedStrategy) {
            showCommandAlert('warning', '⚠️ Please select a strategy first');
            return;
        }

        try {
            addToCommandHistory('START', 'executing', 'Initializing trading bot...');
            const result = await startBot(botMode, selectedStrategy, selectedProfile);
            
            if (result.success) {
                addToCommandHistory('START', 'success', `✅ Bot started successfully - ${selectedStrategy}`);
                showCommandAlert('success', `🚀 Trading Bot Started: ${selectedStrategy}`);
            } else {
                addToCommandHistory('START', 'error', `❌ Start failed: ${result.message}`);
                showCommandAlert('error', `❌ Failed to start bot: ${result.message}`);
            }
        } catch (error) {
            addToCommandHistory('START', 'error', `⚠️ Error: ${error.message}`);
            showCommandAlert('error', `⚠️ Start error: ${error.message}`);
        }
    };

    const handleStopBot = async () => {
        try {
            addToCommandHistory('STOP', 'executing', 'Stopping trading bot...');
            const result = await stopBot();
            
            if (result.success) {
                addToCommandHistory('STOP', 'success', '✅ Bot stopped successfully');
                showCommandAlert('success', '🛑 Trading Bot Stopped Successfully');
            } else {
                addToCommandHistory('STOP', 'error', `❌ Stop failed: ${result.message}`);
                showCommandAlert('error', `❌ Failed to stop bot: ${result.message}`);
            }
        } catch (error) {
            addToCommandHistory('STOP', 'error', `⚠️ Error: ${error.message}`);
            showCommandAlert('error', `⚠️ Stop error: ${error.message}`);
        }
    };

    const handleRestartBot = async () => {
        if (!selectedStrategy) {
            showCommandAlert('warning', '⚠️ Please select a strategy for restart');
            return;
        }

        try {
            addToCommandHistory('RESTART', 'executing', 'Restarting trading bot...');
            const result = await restartBot(botMode, selectedStrategy, selectedProfile);
            
            if (result.success) {
                addToCommandHistory('RESTART', 'success', `✅ Bot restarted - ${selectedStrategy}`);
                showCommandAlert('success', `🔄 Bot Restarted: ${selectedStrategy}`);
            } else {
                addToCommandHistory('RESTART', 'error', `❌ Restart failed: ${result.message}`);
                showCommandAlert('error', `❌ Restart failed: ${result.message}`);
            }
        } catch (error) {
            addToCommandHistory('RESTART', 'error', `⚠️ Error: ${error.message}`);
            showCommandAlert('error', `⚠️ Restart error: ${error.message}`);
        }
    };

    const handleEmergencyStop = async () => {
        if (!window.confirm('🚨 EMERGENCY STOP - This will immediately stop all trading operations. Are you sure?')) {
            return;
        }

        setEmergencyMode(true);
        try {
            addToCommandHistory('EMERGENCY', 'executing', '🚨 EMERGENCY STOP initiated');
            const result = await emergencyStop();
            
            if (result.success) {
                addToCommandHistory('EMERGENCY', 'success', '🚨 Emergency stop completed');
                showCommandAlert('success', '🚨 EMERGENCY STOP - All Operations Halted');
            } else {
                addToCommandHistory('EMERGENCY', 'error', `❌ Emergency stop failed: ${result.message}`);
                showCommandAlert('error', `❌ Emergency stop failed: ${result.message}`);
            }
        } catch (error) {
            addToCommandHistory('EMERGENCY', 'error', `⚠️ Emergency error: ${error.message}`);
            showCommandAlert('error', `⚠️ Emergency stop error: ${error.message}`);
        } finally {
            setTimeout(() => setEmergencyMode(false), 5000); // Reset after 5 seconds
        }
    };

    const showCommandAlert = (type, message) => {
        const alert = document.createElement('div');
        alert.className = `command-alert ${type}`;
        alert.textContent = message;
        document.body.appendChild(alert);
        
        setTimeout(() => {
            alert.remove();
        }, 5000);
    };

    const getStatusColor = () => {
        switch (botStatus.status) {
            case 'running': return '#00ff88';
            case 'stopped': return '#6c757d';
            case 'error': return '#ff4757';
            case 'starting': return '#ffa500';
            case 'stopping': return '#ffa500';
            default: return '#ffffff';
        }
    };

    const getStatusIcon = () => {
        switch (botStatus.status) {
            case 'running': return '🚀';
            case 'stopped': return '🛑';
            case 'error': return '⚠️';
            case 'starting': return '⏳';
            case 'stopping': return '⏸️';
            default: return '❓';
        }
    };

    if (loading) {
        return (
            <div className="command-center loading">
                <div className="loading-spinner command-spinner"></div>
                <div className="loading-text">Initializing Command Center...</div>
            </div>
        );
    }

    return (
        <div className="factory-command-center">
            {/* Control Particles */}
            <div className="control-particles-container">
                {controlParticles.map(particle => (
                    <div
                        key={particle.id}
                        className="control-particle"
                        style={{
                            left: `${particle.x}%`,
                            top: `${particle.y}%`,
                            animationDelay: `${particle.delay}ms`,
                            animationDuration: `${particle.duration}ms`
                        }}
                    >
                        {particle.symbol}
                    </div>
                ))}
            </div>

            {/* Emergency Mode Overlay */}
            {emergencyMode && (
                <div className="emergency-overlay">
                    <div className="emergency-content">
                        <div className="emergency-icon pulsing-red">🚨</div>
                        <div className="emergency-text">EMERGENCY MODE ACTIVE</div>
                        <div className="emergency-subtitle">All systems stopping...</div>
                    </div>
                </div>
            )}

            {/* Command Center Header */}
            <div className="command-center-header">
                <div className="header-title">
                    <span className="title-icon pulsing-blue">🎮</span>
                    <span className="neon-text">FACTORY COMMAND CENTER</span>
                    <span className="title-icon pulsing-blue">🎮</span>
                </div>
                
                <div className="system-status">
                    <div className="status-display">
                        <span className="status-icon" style={{ color: getStatusColor() }}>
                            {getStatusIcon()}
                        </span>
                        <div className="status-info">
                            <div 
                                className="status-value"
                                style={{ color: getStatusColor() }}
                            >
                                {botStatus.status?.toUpperCase() || 'UNKNOWN'}
                            </div>
                            <div className="status-label">System Status</div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Main Control Panel */}
            <div className="main-control-panel">
                {/* Bot Configuration */}
                <div className="bot-configuration-section">
                    <div className="config-title">
                        <span className="config-icon">⚙️</span>
                        <span>BOT CONFIGURATION</span>
                    </div>
                    
                    <div className="configuration-grid">
                        {/* Mode Selection */}
                        <div className="config-group">
                            <label className="config-label">Trading Mode</label>
                            <select 
                                className="factory-select"
                                value={botMode}
                                onChange={(e) => setBotMode(e.target.value)}
                                disabled={botStatus.status === 'running'}
                            >
                                <option value="live">🚀 LIVE TRADING</option>
                                <option value="paper">📋 PAPER TRADING</option>
                                <option value="backtest">📊 BACKTEST</option>
                            </select>
                        </div>

                        {/* Strategy Selection */}
                        <div className="config-group">
                            <label className="config-label">Strategy</label>
                            <select 
                                className="factory-select"
                                value={selectedStrategy}
                                onChange={(e) => setSelectedStrategy(e.target.value)}
                                disabled={botStatus.status === 'running'}
                            >
                                <option value="">Select Strategy...</option>
                                {availableStrategies.map(strategy => (
                                    <option key={strategy.name} value={strategy.name}>
                                        {strategy.display_name} ({strategy.risk_level})
                                    </option>
                                ))}
                            </select>
                        </div>

                        {/* Profile Selection */}
                        <div className="config-group">
                            <label className="config-label">Profile</label>
                            <select 
                                className="factory-select"
                                value={selectedProfile}
                                onChange={(e) => setSelectedProfile(e.target.value)}
                                disabled={botStatus.status === 'running'}
                            >
                                <option value="">Default Profile</option>
                                {profiles.map(profile => (
                                    <option key={profile.name} value={profile.name}>
                                        {profile.display_name}
                                    </option>
                                ))}
                            </select>
                        </div>
                    </div>
                </div>

                {/* Control Buttons */}
                <div className="control-buttons-section">
                    <div className="buttons-title">
                        <span className="buttons-icon">🕹️</span>
                        <span>BOT CONTROLS</span>
                    </div>
                    
                    <div className="control-buttons-grid">
                        <button
                            className="factory-button success large"
                            onClick={handleStartBot}
                            disabled={botStatus.status === 'running' || loading}
                        >
                            <span className="button-icon">🚀</span>
                            <span className="button-text">START BOT</span>
                        </button>

                        <button
                            className="factory-button warning large"
                            onClick={handleStopBot}
                            disabled={botStatus.status !== 'running' || loading}
                        >
                            <span className="button-icon">🛑</span>
                            <span className="button-text">STOP BOT</span>
                        </button>

                        <button
                            className="factory-button info large"
                            onClick={handleRestartBot}
                            disabled={loading}
                        >
                            <span className="button-icon">🔄</span>
                            <span className="button-text">RESTART</span>
                        </button>

                        <button
                            className="factory-button danger large emergency"
                            onClick={handleEmergencyStop}
                            disabled={loading || emergencyMode}
                        >
                            <span className="button-icon">🚨</span>
                            <span className="button-text">EMERGENCY</span>
                        </button>
                    </div>
                </div>

                {/* System Monitoring */}
                <div className="system-monitoring-section">
                    <div className="monitoring-title">
                        <span className="monitoring-icon">📊</span>
                        <span>SYSTEM MONITORING</span>
                    </div>
                    
                    <div className="monitoring-grid">
                        <SystemMetric
                            icon="🖥️"
                            label="CPU Usage"
                            value={`${botStatus.cpu_usage || 0}%`}
                            color={botStatus.cpu_usage > 80 ? "#ff4757" : botStatus.cpu_usage > 60 ? "#ffa500" : "#00ff88"}
                        />
                        
                        <SystemMetric
                            icon="💾"
                            label="Memory"
                            value={`${botStatus.memory_usage || 0}%`}
                            color={botStatus.memory_usage > 80 ? "#ff4757" : botStatus.memory_usage > 60 ? "#ffa500" : "#00ff88"}
                        />
                        
                        <SystemMetric
                            icon="⏱️"
                            label="Uptime"
                            value={botStatus.uptime || '0m'}
                            color="#3742fa"
                        />
                        
                        <SystemMetric
                            icon="🔗"
                            label="API Status"
                            value={botStatus.api_connected ? 'Connected' : 'Disconnected'}
                            color={botStatus.api_connected ? "#00ff88" : "#ff4757"}
                        />
                    </div>
                </div>
            </div>

            {/* Command History */}
            <div className="command-history-section">
                <div className="history-title">
                    <span className="history-icon">📜</span>
                    <span>COMMAND HISTORY</span>
                    <button 
                        className="clear-history-btn"
                        onClick={() => setCommandHistory([])}
                    >
                        🗑️ Clear
                    </button>
                </div>
                
                <div className="command-history-feed">
                    {commandHistory.length > 0 ? (
                        commandHistory.map(entry => (
                            <CommandHistoryItem
                                key={entry.id}
                                timestamp={entry.timestamp}
                                command={entry.command}
                                status={entry.status}
                                message={entry.message}
                            />
                        ))
                    ) : (
                        <div className="no-history">
                            <div className="no-history-icon">📝</div>
                            <div className="no-history-text">No commands executed yet</div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

/**
 * System Metric Component
 */
const SystemMetric = ({ icon, label, value, color }) => (
    <div className="system-metric">
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
    </div>
);

/**
 * Command History Item Component
 */
const CommandHistoryItem = ({ timestamp, command, status, message }) => {
    const getStatusIcon = () => {
        switch (status) {
            case 'success': return '✅';
            case 'error': return '❌';
            case 'executing': return '⏳';
            default: return '📝';
        }
    };

    const getStatusColor = () => {
        switch (status) {
            case 'success': return '#00ff88';
            case 'error': return '#ff4757';
            case 'executing': return '#ffa500';
            default: return '#ffffff';
        }
    };

    return (
        <div className={`command-history-item ${status}`}>
            <div className="history-timestamp">{timestamp}</div>
            <div className="history-command">
                <span 
                    className="command-icon"
                    style={{ color: getStatusColor() }}
                >
                    {getStatusIcon()}
                </span>
                <span className="command-name">{command}</span>
            </div>
            <div className="history-message">{message}</div>
        </div>
    );
};

export default FactoryCommandCenter;