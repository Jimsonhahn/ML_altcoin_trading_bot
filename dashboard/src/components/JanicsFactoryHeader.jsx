/**
 * 🏭 JANICS FREEDOM FACTORY HEADER 🏭
 * The most spectacular trading dashboard header ever created!
 */

import React, { useState, useEffect } from 'react';
import { useHeaderStatus } from '../hooks/useDashboardData';
import '../styles/JanicsFactory.css';

const JanicsFactoryHeader = () => {
    const { headerStatus, loading, error } = useHeaderStatus();
    const [currentTime, setCurrentTime] = useState(new Date());
    const [factoryParticles, setFactoryParticles] = useState([]);

    // Update time every second
    useEffect(() => {
        const timer = setInterval(() => setCurrentTime(new Date()), 1000);
        return () => clearInterval(timer);
    }, []);

    // Generate floating money particles
    useEffect(() => {
        const particles = Array.from({ length: 20 }, (_, i) => ({
            id: i,
            x: Math.random() * 100,
            delay: Math.random() * 3,
            symbol: ['$', '€', '¥', '₿'][Math.floor(Math.random() * 4)]
        }));
        setFactoryParticles(particles);
    }, []);

    if (loading) {
        return (
            <header className="factory-header loading">
                <div className="loading-spinner"></div>
                <div className="neon-text">INITIALIZING FACTORY...</div>
            </header>
        );
    }

    return (
        <header className="factory-header holographic">
            {/* Floating Money Particles */}
            <div className="money-particles-container">
                {factoryParticles.map(particle => (
                    <div
                        key={particle.id}
                        className="floating-money-particle"
                        style={{
                            left: `${particle.x}%`,
                            animationDelay: `${particle.delay}s`
                        }}
                    >
                        {particle.symbol}
                    </div>
                ))}
            </div>

            {/* Factory Steam Effects */}
            <div className="factory-steam" style={{ left: '20%', animationDelay: '0s' }}></div>
            <div className="factory-steam" style={{ left: '80%', animationDelay: '2s' }}></div>

            {/* Main Factory Title */}
            <div className="factory-title-section">
                <div className="factory-logo-container">
                    <div className="factory-icon pulsing-green">🏭</div>
                    <div className="factory-title">
                        <h1 className="neon-text main-title">JANICS FREEDOM FACTORY</h1>
                        <div className="factory-subtitle">
                            AUTONOMOUS WEALTH GENERATION TERMINAL
                        </div>
                        <div className="factory-version">
                            v2.0 - QUANTUM MONEY PRINTER EDITION
                        </div>
                    </div>
                    <div className="factory-icon pulsing-green">🏭</div>
                </div>

                {/* Live Factory Time */}
                <div className="factory-time">
                    <div className="time-label">FACTORY TIME</div>
                    <div className="current-time">
                        {currentTime.toLocaleTimeString('en-US', { 
                            hour12: false,
                            timeZone: 'UTC'
                        })} UTC
                    </div>
                </div>
            </div>

            {/* Factory Status Grid */}
            <div className="factory-status-grid">
                <FactoryStatusIndicator
                    icon="🏭"
                    label="FACTORY ONLINE"
                    value={headerStatus.factory_online ? "OPERATIONAL" : "OFFLINE"}
                    status={headerStatus.factory_online ? "success" : "danger"}
                    animation="pulsing-green"
                />
                
                <FactoryStatusIndicator
                    icon="💰"
                    label="MONEY PRINTER"
                    value={headerStatus.trading_bot_status === "Running" ? "PRINTING" : "READY"}
                    status={headerStatus.trading_bot_status === "Running" ? "success" : "warning"}
                    animation="money-printing"
                />
                
                <FactoryStatusIndicator
                    icon="🧠"
                    label="AI BRAIN"
                    value="LEARNING"
                    status="success"
                    animation="brain-pulse"
                />
                
                <FactoryStatusIndicator
                    icon="📈"
                    label="PROFIT RATE"
                    value="+2.18%/day"
                    status="success"
                    animation="victory-glow"
                />
                
                <FactoryStatusIndicator
                    icon="🌐"
                    label="SERVER"
                    value={headerStatus.server_status}
                    status={headerStatus.server_status === "Connected" ? "success" : "warning"}
                    animation="pulsing-orange"
                />
                
                <FactoryStatusIndicator
                    icon="⚙️"
                    label="SYSTEM"
                    value={headerStatus.system_status}
                    status={headerStatus.system_status === "Operational" ? "success" : "info"}
                    animation="pulsing-green"
                />
            </div>

            {/* Factory Performance Banner */}
            <div className="factory-performance-banner">
                <div className="performance-item">
                    <span className="performance-icon">🎯</span>
                    <span className="performance-value">89.2%</span>
                    <span className="performance-label">AI Confidence</span>
                </div>
                <div className="performance-item">
                    <span className="performance-icon">🚀</span>
                    <span className="performance-value">$27,543</span>
                    <span className="performance-label">Total Generated</span>
                </div>
                <div className="performance-item">
                    <span className="performance-icon">⚡</span>
                    <span className="performance-value">5</span>
                    <span className="performance-label">Strategies Active</span>
                </div>
                <div className="performance-item">
                    <span className="performance-icon">🏆</span>
                    <span className="performance-value">24h</span>
                    <span className="performance-label">Profit Streak</span>
                </div>
            </div>

            {/* Last Update Indicator */}
            <div className="factory-last-update">
                <span className="update-icon">🔄</span>
                <span className="update-text">
                    Last Update: {headerStatus.last_update || '--:--:--'}
                </span>
            </div>
        </header>
    );
};

/**
 * Individual Factory Status Indicator Component
 */
const FactoryStatusIndicator = ({ icon, label, value, status, animation }) => {
    const getStatusClass = () => {
        switch (status) {
            case 'success': return 'status-success';
            case 'warning': return 'status-warning';  
            case 'danger': return 'status-danger';
            case 'info': return 'status-info';
            default: return 'status-neutral';
        }
    };

    return (
        <div className={`factory-status-indicator ${getStatusClass()} ${animation}`}>
            <div className="status-icon">{icon}</div>
            <div className="status-content">
                <div className="status-label">{label}</div>
                <div className="status-value">{value}</div>
            </div>
            <div className="status-glow"></div>
        </div>
    );
};

export default JanicsFactoryHeader;