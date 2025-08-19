/**
 * 🚀 THE ULTIMATE JANICS FREEDOM FACTORY DASHBOARD 🚀
 * The most spectacular trading terminal ever created!
 * Das ultimative Trading Terminal der Welt - Sci-Fi meets $100M Hedge Fund!
 */

import React, { useState, useEffect, useRef } from 'react';
import JanicsFactoryHeader from './JanicsFactoryHeader';
import WealthGenerationCenter from './WealthGenerationCenter';
import FactoryProductionLine from './FactoryProductionLine';
import AIFactoryBrain from './AIFactoryBrain';
import StrategyFactoryAssembly from './StrategyFactoryAssembly';
import FactoryCommandCenter from './FactoryCommandCenter';
import FactoryAchievements from './FactoryAchievements';
import TradingModeManager from './TradingModeManager';
import factorySoundSystem from '../utils/FactorySoundSystem';
import ParticleSystem from '../utils/ParticleSystem';

// Import all styles
import '../styles/JanicsFactory.css';
import '../styles/FactoryHeader.css';
import '../styles/WealthCenter.css';
import '../styles/AIFactoryBrain.css';
import '../styles/StrategyAssembly.css';
import '../styles/CommandCenter.css';
import '../styles/Achievements.css';

const RevolutionaryJanicsDashboard = () => {
    const [activeSection, setActiveSection] = useState('overview');
    const [soundEnabled, setSoundEnabled] = useState(true);
    const [particlesEnabled, setParticlesEnabled] = useState(true);
    const [showSettings, setShowSettings] = useState(false);
    const [dashboardTheme, setDashboardTheme] = useState('factory');
    const particleSystemRef = useRef(null);
    const dashboardRef = useRef(null);

    // Initialize systems on mount
    useEffect(() => {
        // Initialize particle system
        if (particlesEnabled && dashboardRef.current) {
            particleSystemRef.current = new ParticleSystem(dashboardRef.current);
            particleSystemRef.current.start();
            
            // Start ambient effects
            particleSystemRef.current.startMoneyRain(0.1);
            particleSystemRef.current.startElectricCurrent();
        }

        // Initialize sound system
        if (soundEnabled) {
            factorySoundSystem.unmute();
            factorySoundSystem.startAmbientSound();
            
            // Welcome sound
            setTimeout(() => {
                factorySoundSystem.playBotStatusSound('starting');
            }, 1000);
        } else {
            factorySoundSystem.mute();
        }

        // Cleanup
        return () => {
            if (particleSystemRef.current) {
                particleSystemRef.current.destroy();
            }
        };
    }, [soundEnabled, particlesEnabled]);

    // Navigation sections (erweitert um Trading Mode Manager)
    const sections = [
        { id: 'overview', name: 'Factory Overview', icon: '🏭', component: 'overview' },
        { id: 'wealth', name: 'Wealth Center', icon: '💰', component: WealthGenerationCenter },
        { id: 'production', name: 'Production Line', icon: '🔧', component: FactoryProductionLine },
        { id: 'brain', name: 'AI Brain', icon: '🧠', component: AIFactoryBrain },
        { id: 'strategy', name: 'Strategy Assembly', icon: '⚙️', component: StrategyFactoryAssembly },
        { id: 'control', name: 'Command Center', icon: '🎮', component: FactoryCommandCenter },
        { id: 'trading-modes', name: 'Trading Modes', icon: '🔄', component: TradingModeManager },
        { id: 'achievements', name: 'Achievements', icon: '🏆', component: FactoryAchievements }
    ];

    const handleSectionChange = (sectionId) => {
        setActiveSection(sectionId);
        
        if (soundEnabled) {
            factorySoundSystem.playUISound('click');
        }
        
        if (particleSystemRef.current) {
            const element = document.querySelector(`[data-section="${sectionId}"]`);
            if (element) {
                const rect = element.getBoundingClientRect();
                particleSystemRef.current.createSuccessRipple(
                    rect.left + rect.width / 2,
                    rect.top + rect.height / 2
                );
            }
        }
    };

    const toggleSound = () => {
        setSoundEnabled(!soundEnabled);
        if (!soundEnabled) {
            factorySoundSystem.unmute();
            factorySoundSystem.playUISound('success');
        } else {
            factorySoundSystem.mute();
        }
    };

    const toggleParticles = () => {
        setParticlesEnabled(!particlesEnabled);
        
        if (!particlesEnabled && dashboardRef.current) {
            // Start particles
            particleSystemRef.current = new ParticleSystem(dashboardRef.current);
            particleSystemRef.current.start();
            particleSystemRef.current.startMoneyRain(0.1);
        } else if (particleSystemRef.current) {
            // Stop particles
            particleSystemRef.current.destroy();
            particleSystemRef.current = null;
        }
    };

    const renderActiveSection = () => {
        const section = sections.find(s => s.id === activeSection);
        
        if (!section) return null;
        
        if (section.component === 'overview') {
            return (
                <div className="dashboard-overview">
                    {/* Overview combines multiple key components */}
                    <div className="overview-grid">
                        <div className="overview-section">
                            <WealthGenerationCenter />
                        </div>
                        
                        <div className="overview-section">
                            <FactoryProductionLine />
                        </div>
                        
                        <div className="overview-section">
                            <AIFactoryBrain />
                        </div>
                        
                        <div className="overview-section">
                            <FactoryCommandCenter />
                        </div>
                    </div>
                </div>
            );
        }
        
        const Component = section.component;
        return <Component />;
    };

    return (
        <div 
            ref={dashboardRef}
            className={`revolutionary-janics-dashboard ${dashboardTheme}-theme`}
            data-sound={soundEnabled}
            data-particles={particlesEnabled}
        >
            {/* Spectacular Factory Header - Always Visible */}
            <JanicsFactoryHeader />
            
            {/* Navigation Bar */}
            <div className="dashboard-navigation">
                <div className="nav-container">
                    <div className="nav-sections">
                        {sections.map(section => (
                            <button
                                key={section.id}
                                className={`nav-section ${activeSection === section.id ? 'active' : ''}`}
                                onClick={() => handleSectionChange(section.id)}
                                data-section={section.id}
                                onMouseEnter={() => {
                                    if (soundEnabled) {
                                        factorySoundSystem.playUISound('hover');
                                    }
                                }}
                            >
                                <span className="nav-icon">{section.icon}</span>
                                <span className="nav-label">{section.name}</span>
                            </button>
                        ))}
                    </div>
                    
                    {/* Dashboard Controls */}
                    <div className="dashboard-controls">
                        <button
                            className={`control-button ${soundEnabled ? 'active' : ''}`}
                            onClick={toggleSound}
                            title="Toggle Sound Effects"
                        >
                            <span className="control-icon">{soundEnabled ? '🔊' : '🔇'}</span>
                        </button>
                        
                        <button
                            className={`control-button ${particlesEnabled ? 'active' : ''}`}
                            onClick={toggleParticles}
                            title="Toggle Particle Effects"
                        >
                            <span className="control-icon">{particlesEnabled ? '✨' : '💫'}</span>
                        </button>
                        
                        <button
                            className="control-button"
                            onClick={() => setShowSettings(!showSettings)}
                            title="Dashboard Settings"
                        >
                            <span className="control-icon">⚙️</span>
                        </button>
                    </div>
                </div>
            </div>

            {/* Settings Panel */}
            {showSettings && (
                <div className="settings-panel">
                    <div className="settings-content">
                        <div className="settings-header">
                            <h3>🎛️ Factory Settings</h3>
                            <button 
                                className="close-settings"
                                onClick={() => setShowSettings(false)}
                            >
                                ✕
                            </button>
                        </div>
                        
                        <div className="settings-options">
                            <div className="setting-group">
                                <label>Dashboard Theme</label>
                                <select 
                                    value={dashboardTheme}
                                    onChange={(e) => setDashboardTheme(e.target.value)}
                                >
                                    <option value="factory">🏭 Factory Theme</option>
                                    <option value="cyberpunk">🌃 Cyberpunk</option>
                                    <option value="matrix">💚 Matrix</option>
                                    <option value="neon">🌈 Neon Dreams</option>
                                </select>
                            </div>
                            
                            <div className="setting-group">
                                <label>Sound Volume</label>
                                <input 
                                    type="range"
                                    min="0"
                                    max="100"
                                    defaultValue="30"
                                    onChange={(e) => factorySoundSystem.setVolume(e.target.value / 100)}
                                />
                            </div>
                            
                            <div className="setting-group">
                                <label>Visual Effects</label>
                                <div className="checkbox-group">
                                    <label>
                                        <input 
                                            type="checkbox" 
                                            checked={particlesEnabled}
                                            onChange={toggleParticles}
                                        />
                                        Particle Effects
                                    </label>
                                    <label>
                                        <input 
                                            type="checkbox" 
                                            checked={soundEnabled}
                                            onChange={toggleSound}
                                        />
                                        Sound Effects
                                    </label>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Main Content Area */}
            <div className="dashboard-content">
                <div className="content-container">
                    {renderActiveSection()}
                </div>
            </div>

            {/* Factory Status Bar */}
            <div className="factory-status-bar">
                <div className="status-left">
                    <div className="status-item">
                        <span className="status-icon">🏭</span>
                        <span className="status-text">Factory Online</span>
                        <span className="status-indicator active"></span>
                    </div>
                    
                    <div className="status-item">
                        <span className="status-icon">💰</span>
                        <span className="status-text">Money Printer: ACTIVE</span>
                        <span className="status-indicator printing"></span>
                    </div>
                    
                    <div className="status-item">
                        <span className="status-icon">🧠</span>
                        <span className="status-text">AI Brain: LEARNING</span>
                        <span className="status-indicator thinking"></span>
                    </div>
                </div>
                
                <div className="status-right">
                    <div className="version-info">
                        <span className="version-text">JANICS FREEDOM FACTORY v2.0</span>
                        <span className="version-subtitle">QUANTUM MONEY PRINTER EDITION</span>
                    </div>
                </div>
            </div>

            {/* Copyright & Credits */}
            <div className="dashboard-footer">
                <div className="footer-content">
                    <div className="footer-left">
                        <span className="copyright">© 2024 Janics Freedom Factory</span>
                        <span className="subtitle">Das ultimative Trading Terminal der Welt</span>
                    </div>
                    
                    <div className="footer-right">
                        <span className="powered-by">Powered by Quantum AI & Pure German Engineering</span>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default RevolutionaryJanicsDashboard;