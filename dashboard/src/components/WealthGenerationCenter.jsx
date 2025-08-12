/**
 * 💰 WEALTH GENERATION CENTER 💰
 * The ultimate money display with holographic effects and real trading data!
 */

import React, { useState, useEffect, useRef } from 'react';
import { useWealthData } from '../hooks/useDashboardData';
import '../styles/JanicsFactory.css';

const WealthGenerationCenter = () => {
    const { wealthData, loading, error } = useWealthData();
    const [animatingValue, setAnimatingValue] = useState(0);
    const [dailyPnlAnimating, setDailyPnlAnimating] = useState(0);
    const [moneyParticles, setMoneyParticles] = useState([]);
    const intervalRef = useRef(null);

    // Animate total wealth counter
    useEffect(() => {
        if (wealthData.total_value && wealthData.total_value !== animatingValue) {
            const startValue = animatingValue;
            const endValue = wealthData.total_value;
            const duration = 2000; // 2 seconds
            const startTime = Date.now();

            const animate = () => {
                const elapsedTime = Date.now() - startTime;
                const progress = Math.min(elapsedTime / duration, 1);
                
                // Easing function for smooth animation
                const easeOutQuart = 1 - Math.pow(1 - progress, 4);
                const currentValue = startValue + (endValue - startValue) * easeOutQuart;
                
                setAnimatingValue(currentValue);

                if (progress < 1) {
                    requestAnimationFrame(animate);
                }
            };

            animate();
        }
    }, [wealthData.total_value]);

    // Animate daily P&L
    useEffect(() => {
        if (wealthData.daily_pnl && wealthData.daily_pnl !== dailyPnlAnimating) {
            const startValue = dailyPnlAnimating;
            const endValue = wealthData.daily_pnl;
            const duration = 1500;
            const startTime = Date.now();

            const animate = () => {
                const elapsedTime = Date.now() - startTime;
                const progress = Math.min(elapsedTime / duration, 1);
                const easeOutCubic = 1 - Math.pow(1 - progress, 3);
                const currentValue = startValue + (endValue - startValue) * easeOutCubic;
                
                setDailyPnlAnimating(currentValue);

                if (progress < 1) {
                    requestAnimationFrame(animate);
                }
            };

            animate();
        }
    }, [wealthData.daily_pnl]);

    // Generate money particles when profit is made
    useEffect(() => {
        if (wealthData.daily_pnl > 0) {
            const generateParticles = () => {
                const newParticles = Array.from({ length: 8 }, (_, i) => ({
                    id: Date.now() + i,
                    x: Math.random() * 100,
                    delay: Math.random() * 1000,
                    symbol: ['💰', '💎', '🤑', '💵', '💸', '🪙'][Math.floor(Math.random() * 6)],
                    size: Math.random() * 20 + 20
                }));
                
                setMoneyParticles(prev => [...prev, ...newParticles]);
                
                // Remove particles after animation
                setTimeout(() => {
                    setMoneyParticles(prev => 
                        prev.filter(p => !newParticles.some(np => np.id === p.id))
                    );
                }, 4000);
            };

            // Generate particles every 3 seconds when profitable
            intervalRef.current = setInterval(generateParticles, 3000);
            generateParticles(); // Initial burst

            return () => {
                if (intervalRef.current) clearInterval(intervalRef.current);
            };
        }
    }, [wealthData.daily_pnl]);

    if (loading) {
        return (
            <div className="wealth-center loading">
                <div className="loading-spinner"></div>
                <div className="loading-text">Loading Wealth Generation Systems...</div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="wealth-center error">
                <div className="error-icon">⚠️</div>
                <div className="error-text">Wealth Systems Offline: {error}</div>
            </div>
        );
    }

    return (
        <div className="wealth-generation-center">
            {/* Money Particles */}
            <div className="money-particles-wealth">
                {moneyParticles.map(particle => (
                    <div
                        key={particle.id}
                        className="wealth-money-particle"
                        style={{
                            left: `${particle.x}%`,
                            animationDelay: `${particle.delay}ms`,
                            fontSize: `${particle.size}px`
                        }}
                    >
                        {particle.symbol}
                    </div>
                ))}
            </div>

            {/* Main Wealth Display */}
            <div className="main-wealth-display holographic">
                <div className="wealth-title">
                    <span className="title-icon">🏭</span>
                    <span className="neon-text">WEALTH GENERATION CENTER</span>
                    <span className="title-icon">🏭</span>
                </div>

                {/* Total Wealth */}
                <div className="total-wealth-container">
                    <div className="wealth-label">TOTAL WEALTH GENERATED</div>
                    <div className="total-wealth-amount">
                        <AnimatedCounter 
                            value={animatingValue}
                            prefix="$"
                            decimals={2}
                            size="massive"
                            glowColor={wealthData.daily_pnl >= 0 ? "#00ff88" : "#ff4757"}
                        />
                    </div>
                    <div className="wealth-subtitle">
                        Autonomous Money Printing Since Genesis Block
                    </div>
                </div>

                {/* Today's Generation */}
                <div className="daily-generation-section">
                    <div className="generation-header">
                        <div className="generation-flame">🔥</div>
                        <div className="generation-title">TODAY'S FREEDOM GENERATION</div>
                        <div className="generation-flame">🔥</div>
                    </div>

                    <div className="generation-main">
                        <div className="generation-amount">
                            <AnimatedCounter 
                                value={dailyPnlAnimating}
                                prefix={wealthData.daily_pnl >= 0 ? "+$" : "-$"}
                                decimals={2}
                                size="large"
                                color={wealthData.daily_pnl >= 0 ? "#ff6b35" : "#ff4757"}
                                animation="victory-glow"
                            />
                        </div>

                        <div className="generation-details">
                            <div className="generation-percentage">
                                <span className={`percentage-value ${wealthData.daily_pnl >= 0 ? 'positive' : 'negative'}`}>
                                    {wealthData.daily_pnl >= 0 ? '+' : ''}{wealthData.daily_pnl_percentage?.toFixed(2)}%
                                </span>
                                <span className="percentage-label">Daily Return</span>
                            </div>

                            {wealthData.profit_streak_hours > 0 && (
                                <div className="profit-streak">
                                    <div className="streak-icon">⚡</div>
                                    <div className="streak-info">
                                        <div className="streak-value">{wealthData.profit_streak_hours}h</div>
                                        <div className="streak-label">Profit Streak</div>
                                    </div>
                                    <div className="confidence-boost">
                                        <span className="boost-value">{wealthData.confidence_boost_formatted}</span>
                                        <span className="boost-label">Confidence</span>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* Holographic Progress Rings */}
            <div className="progress-holo-rings">
                <HolographicProgressRing
                    title="Daily Factory Output"
                    current={wealthData.daily_progress?.current || 0}
                    target={wealthData.daily_progress?.target || 500}
                    percentage={wealthData.daily_progress?.percentage || 0}
                    color="#00ff88"
                    icon="📈"
                    animation={wealthData.daily_progress?.percentage >= 100 ? "victory-pulse" : "pulsing-green"}
                />

                <HolographicProgressRing
                    title="Weekly Production"
                    current={wealthData.weekly_progress?.current || 0}
                    target={wealthData.weekly_progress?.target || 3500}
                    percentage={wealthData.weekly_progress?.percentage || 0}
                    color="#ffa500"
                    icon="📊"
                    animation="pulsing-orange"
                />

                <HolographicProgressRing
                    title="Monthly Goal"
                    current={wealthData.monthly_progress?.current || 0}
                    target={wealthData.monthly_progress?.target || 15000}
                    percentage={wealthData.monthly_progress?.percentage || 0}
                    color="#ff1493"
                    icon="🎯"
                    animation={wealthData.monthly_progress?.percentage >= 95 ? "victory-pulse" : "pulsing-red"}
                />
            </div>

            {/* Wealth Statistics */}
            <div className="wealth-statistics">
                <WealthStatistic
                    icon="🎯"
                    label="Win Rate"
                    value={`${wealthData.win_rate || 0}%`}
                    trend="up"
                />
                <WealthStatistic
                    icon="📈"
                    label="Total Trades"
                    value={wealthData.total_trades || 0}
                    trend="neutral"
                />
                <WealthStatistic
                    icon="🏆"
                    label="Best Performer"
                    value={wealthData.best_performer || 'N/A'}
                    trend="up"
                />
                <WealthStatistic
                    icon="⚠️"
                    label="Risk Level"
                    value={wealthData.risk_level || 'Unknown'}
                    trend={wealthData.risk_level === 'Low' ? 'down' : wealthData.risk_level === 'High' ? 'up' : 'neutral'}
                />
            </div>
        </div>
    );
};

/**
 * Animated Counter Component
 */
const AnimatedCounter = ({ 
    value, 
    prefix = '', 
    suffix = '', 
    decimals = 0, 
    size = 'normal',
    color = '#00ff88',
    glowColor = null,
    animation = ''
}) => {
    const formatValue = (val) => {
        if (isNaN(val)) return '0';
        return val.toLocaleString('en-US', { 
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals 
        });
    };

    const getSizeClass = () => {
        switch (size) {
            case 'massive': return 'counter-massive';
            case 'large': return 'counter-large';
            case 'small': return 'counter-small';
            default: return 'counter-normal';
        }
    };

    return (
        <span 
            className={`animated-counter ${getSizeClass()} ${animation}`}
            style={{ 
                color,
                textShadow: glowColor ? `0 0 20px ${glowColor}` : `0 0 10px ${color}`
            }}
        >
            {prefix}{formatValue(Math.abs(value))}{suffix}
        </span>
    );
};

/**
 * Holographic Progress Ring Component
 */
const HolographicProgressRing = ({ title, current, target, percentage, color, icon, animation }) => {
    const radius = 45;
    const circumference = 2 * Math.PI * radius;
    const strokeDashoffset = circumference - (percentage / 100) * circumference;

    return (
        <div className={`holographic-progress-ring ${animation}`}>
            <div className="ring-icon">{icon}</div>
            
            <div className="progress-ring-container">
                <svg className="progress-ring-svg" width="120" height="120">
                    {/* Background ring */}
                    <circle
                        className="progress-ring-background"
                        cx="60"
                        cy="60"
                        r={radius}
                        fill="transparent"
                        stroke="rgba(255, 255, 255, 0.1)"
                        strokeWidth="4"
                    />
                    
                    {/* Progress ring */}
                    <circle
                        className="progress-ring-progress"
                        cx="60"
                        cy="60"
                        r={radius}
                        fill="transparent"
                        stroke={color}
                        strokeWidth="4"
                        strokeLinecap="round"
                        style={{
                            strokeDasharray: circumference,
                            strokeDashoffset: strokeDashoffset,
                            filter: `drop-shadow(0 0 10px ${color})`
                        }}
                        transform="rotate(-90 60 60)"
                    />
                </svg>
                
                <div className="progress-percentage" style={{ color }}>
                    {percentage.toFixed(0)}%
                </div>
            </div>

            <div className="progress-details">
                <div className="progress-title">{title}</div>
                <div className="progress-values">
                    <span className="current-value">${current.toLocaleString()}</span>
                    <span className="target-divider">/</span>
                    <span className="target-value">${target.toLocaleString()}</span>
                </div>
            </div>
        </div>
    );
};

/**
 * Wealth Statistic Component
 */
const WealthStatistic = ({ icon, label, value, trend }) => {
    const getTrendIcon = () => {
        switch (trend) {
            case 'up': return '📈';
            case 'down': return '📉';
            default: return '➡️';
        }
    };

    const getTrendColor = () => {
        switch (trend) {
            case 'up': return 'var(--factory-success)';
            case 'down': return 'var(--factory-danger)';
            default: return 'var(--factory-info)';
        }
    };

    return (
        <div className="wealth-statistic">
            <div className="stat-icon">{icon}</div>
            <div className="stat-content">
                <div className="stat-value" style={{ color: getTrendColor() }}>
                    {value}
                </div>
                <div className="stat-label">{label}</div>
            </div>
            <div className="stat-trend">{getTrendIcon()}</div>
        </div>
    );
};

export default WealthGenerationCenter;