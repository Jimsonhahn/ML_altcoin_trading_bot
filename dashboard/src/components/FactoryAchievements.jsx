/**
 * 🏆 FACTORY ACHIEVEMENTS & GAMIFICATION 🏆
 * Ultimate achievement system to motivate trading excellence!
 */

import React, { useState, useEffect } from 'react';
import { useAchievements } from '../hooks/useDashboardData';
import factorySoundSystem from '../utils/FactorySoundSystem';
import '../styles/JanicsFactory.css';

const FactoryAchievements = () => {
    const { achievements, stats, loading, error } = useAchievements();
    const [selectedCategory, setSelectedCategory] = useState('all');
    const [newAchievements, setNewAchievements] = useState([]);
    const [showCelebration, setShowCelebration] = useState(false);

    // Check for new achievements
    useEffect(() => {
        const previousAchievements = JSON.parse(localStorage.getItem('factoryAchievements') || '[]');
        const current = achievements.filter(a => a.unlocked);
        const newOnes = current.filter(a => !previousAchievements.find(p => p.id === a.id));
        
        if (newOnes.length > 0) {
            setNewAchievements(newOnes);
            setShowCelebration(true);
            
            // Play celebration sound
            factorySoundSystem.playSound('profitBell');
            
            // Save current achievements
            localStorage.setItem('factoryAchievements', JSON.stringify(current));
            
            // Hide celebration after 5 seconds
            setTimeout(() => {
                setShowCelebration(false);
                setNewAchievements([]);
            }, 5000);
        }
    }, [achievements]);

    const categories = [
        { id: 'all', name: 'All Achievements', icon: '🏆' },
        { id: 'trading', name: 'Trading Mastery', icon: '📈' },
        { id: 'profit', name: 'Profit Milestones', icon: '💰' },
        { id: 'strategy', name: 'Strategy Expert', icon: '🎯' },
        { id: 'consistency', name: 'Consistency Awards', icon: '⭐' },
        { id: 'risk', name: 'Risk Management', icon: '🛡️' },
        { id: 'special', name: 'Special Events', icon: '🎊' }
    ];

    const filteredAchievements = selectedCategory === 'all' 
        ? achievements 
        : achievements.filter(a => a.category === selectedCategory);

    const getAchievementRarity = (achievement) => {
        const rarityColors = {
            common: '#6c757d',
            uncommon: '#28a745',
            rare: '#007bff',
            epic: '#6f42c1',
            legendary: '#fd7e14',
            mythic: '#dc3545'
        };
        
        return {
            color: rarityColors[achievement.rarity] || '#6c757d',
            name: achievement.rarity?.toUpperCase() || 'COMMON'
        };
    };

    const calculateProgress = (achievement) => {
        if (achievement.unlocked) return 100;
        if (!achievement.progress || !achievement.requirement) return 0;
        return Math.min((achievement.progress / achievement.requirement) * 100, 100);
    };

    const getLevelInfo = (xp) => {
        const level = Math.floor(xp / 1000) + 1;
        const currentLevelXP = xp % 1000;
        const nextLevelXP = 1000;
        const progress = (currentLevelXP / nextLevelXP) * 100;
        
        return { level, progress, currentLevelXP, nextLevelXP };
    };

    if (loading) {
        return (
            <div className="achievements loading">
                <div className="loading-spinner achievement-spinner"></div>
                <div className="loading-text">Loading Achievement System...</div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="achievements error">
                <div className="error-icon">🏆💥</div>
                <div className="error-text">Achievement System Error: {error}</div>
            </div>
        );
    }

    const levelInfo = getLevelInfo(stats.totalXP || 0);

    return (
        <div className="factory-achievements">
            {/* Achievement Celebration Overlay */}
            {showCelebration && newAchievements.length > 0 && (
                <div className="achievement-celebration-overlay">
                    <div className="celebration-content">
                        <div className="celebration-title">🎉 ACHIEVEMENT UNLOCKED! 🎉</div>
                        <div className="new-achievements">
                            {newAchievements.map(achievement => (
                                <div key={achievement.id} className="new-achievement">
                                    <div className="achievement-icon">{achievement.icon}</div>
                                    <div className="achievement-info">
                                        <div className="achievement-name">{achievement.name}</div>
                                        <div className="achievement-description">{achievement.description}</div>
                                        <div className="achievement-reward">+{achievement.xpReward} XP</div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            {/* Achievements Header */}
            <div className="achievements-header">
                <div className="header-title">
                    <span className="title-icon pulsing-gold">🏆</span>
                    <span className="neon-text">FACTORY ACHIEVEMENTS</span>
                    <span className="title-icon pulsing-gold">🏆</span>
                </div>
                
                {/* Player Level & Progress */}
                <div className="player-level-section">
                    <div className="level-display">
                        <div className="level-icon">⭐</div>
                        <div className="level-info">
                            <div className="level-number">Level {levelInfo.level}</div>
                            <div className="level-title">Factory Commander</div>
                        </div>
                    </div>
                    
                    <div className="xp-progress">
                        <div className="xp-bar">
                            <div 
                                className="xp-fill"
                                style={{ width: `${levelInfo.progress}%` }}
                            />
                        </div>
                        <div className="xp-text">
                            {levelInfo.currentLevelXP} / {levelInfo.nextLevelXP} XP
                        </div>
                    </div>
                </div>
            </div>

            {/* Achievement Statistics */}
            <div className="achievement-stats">
                <AchievementStat
                    icon="🏆"
                    label="Total Unlocked"
                    value={`${achievements.filter(a => a.unlocked).length}/${achievements.length}`}
                    color="#ffd700"
                />
                <AchievementStat
                    icon="⚡"
                    label="Total XP"
                    value={stats.totalXP || 0}
                    color="#00ff88"
                />
                <AchievementStat
                    icon="📈"
                    label="Completion Rate"
                    value={`${Math.round((achievements.filter(a => a.unlocked).length / achievements.length) * 100)}%`}
                    color="#3742fa"
                />
                <AchievementStat
                    icon="🔥"
                    label="Current Streak"
                    value={`${stats.currentStreak || 0} days`}
                    color="#ff6b35"
                />
            </div>

            {/* Category Filter */}
            <div className="category-filter">
                <div className="filter-title">Categories</div>
                <div className="category-buttons">
                    {categories.map(category => (
                        <button
                            key={category.id}
                            className={`category-button ${selectedCategory === category.id ? 'active' : ''}`}
                            onClick={() => setSelectedCategory(category.id)}
                        >
                            <span className="category-icon">{category.icon}</span>
                            <span className="category-name">{category.name}</span>
                        </button>
                    ))}
                </div>
            </div>

            {/* Achievements Grid */}
            <div className="achievements-grid">
                {filteredAchievements.map(achievement => (
                    <AchievementCard
                        key={achievement.id}
                        achievement={achievement}
                        progress={calculateProgress(achievement)}
                        rarity={getAchievementRarity(achievement)}
                        onClaim={() => {
                            factorySoundSystem.playUISound('success');
                        }}
                    />
                ))}
            </div>
        </div>
    );
};

/**
 * Achievement Statistic Component
 */
const AchievementStat = ({ icon, label, value, color }) => (
    <div className="achievement-stat">
        <div className="stat-icon" style={{ color }}>{icon}</div>
        <div className="stat-content">
            <div className="stat-value" style={{ color }}>{value}</div>
            <div className="stat-label">{label}</div>
        </div>
    </div>
);

/**
 * Achievement Card Component
 */
const AchievementCard = ({ achievement, progress, rarity, onClaim }) => {
    const [isHovered, setIsHovered] = useState(false);
    
    const handleClick = () => {
        if (achievement.unlocked && !achievement.claimed) {
            onClaim();
        }
        factorySoundSystem.playUISound('click');
    };

    return (
        <div 
            className={`achievement-card ${achievement.unlocked ? 'unlocked' : 'locked'} ${achievement.rarity}`}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
            onClick={handleClick}
        >
            {/* Achievement Glow Effect */}
            <div 
                className="achievement-glow"
                style={{ 
                    backgroundColor: rarity.color,
                    opacity: achievement.unlocked ? 0.3 : 0.1 
                }}
            />
            
            {/* Rarity Border */}
            <div 
                className="rarity-border"
                style={{ borderColor: rarity.color }}
            />
            
            {/* Achievement Icon */}
            <div className="achievement-icon-container">
                <div 
                    className={`achievement-icon ${achievement.unlocked ? 'unlocked' : 'locked'}`}
                    style={{ 
                        filter: achievement.unlocked ? 'none' : 'grayscale(100%)',
                        opacity: achievement.unlocked ? 1 : 0.5
                    }}
                >
                    {achievement.icon}
                </div>
                
                {!achievement.unlocked && (
                    <div className="lock-overlay">🔒</div>
                )}
                
                {achievement.unlocked && !achievement.claimed && (
                    <div className="claim-indicator">!</div>
                )}
            </div>
            
            {/* Achievement Info */}
            <div className="achievement-info">
                <div className="achievement-name">
                    {achievement.name}
                </div>
                
                <div className="achievement-description">
                    {achievement.unlocked ? achievement.description : achievement.hiddenDescription || '???'}
                </div>
                
                <div className="achievement-details">
                    <div className="rarity-badge" style={{ backgroundColor: rarity.color }}>
                        {rarity.name}
                    </div>
                    
                    {achievement.unlocked ? (
                        <div className="reward-info">
                            +{achievement.xpReward} XP
                        </div>
                    ) : (
                        <div className="progress-info">
                            {achievement.progress || 0} / {achievement.requirement || 1}
                        </div>
                    )}
                </div>
                
                {/* Progress Bar */}
                {!achievement.unlocked && (
                    <div className="achievement-progress">
                        <div className="progress-bar">
                            <div 
                                className="progress-fill"
                                style={{ 
                                    width: `${progress}%`,
                                    backgroundColor: rarity.color
                                }}
                            />
                        </div>
                        <div className="progress-text">{Math.round(progress)}%</div>
                    </div>
                )}
                
                {/* Unlock Date */}
                {achievement.unlocked && achievement.unlockedAt && (
                    <div className="unlock-date">
                        Unlocked: {new Date(achievement.unlockedAt).toLocaleDateString()}
                    </div>
                )}
            </div>
            
            {/* Hover Effects */}
            {isHovered && achievement.unlocked && (
                <div className="achievement-tooltip">
                    <div className="tooltip-title">{achievement.name}</div>
                    <div className="tooltip-description">{achievement.description}</div>
                    {achievement.tip && (
                        <div className="tooltip-tip">💡 {achievement.tip}</div>
                    )}
                </div>
            )}
        </div>
    );
};

export default FactoryAchievements;