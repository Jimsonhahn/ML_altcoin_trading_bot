/**
 * 🧠 AI FACTORY BRAIN CENTER 🧠
 * The neural core of the Janics Freedom Factory - pure AI intelligence visualization!
 */

import React, { useState, useEffect, useRef } from 'react';
import { useEnhancedBotIntelligence as useBotIntelligence } from '../hooks/useDashboardData';
import '../styles/JanicsFactory.css';

const AIFactoryBrain = () => {
    const { intelligence, loading, error } = useBotIntelligence();
    const [neuralPulses, setNeuralPulses] = useState([]);
    const [brainActivity, setBrainActivity] = useState(0);
    const [synapseConnections, setSynapseConnections] = useState([]);
    const [dataStreams, setDataStreams] = useState([]);
    const intervalRef = useRef(null);

    // Generate neural pulse effects
    useEffect(() => {
        if (intelligence.ai_confidence > 70) {
            const generatePulses = () => {
                const newPulses = Array.from({ length: 6 }, (_, i) => ({
                    id: Date.now() + i,
                    x: Math.random() * 90 + 5,
                    y: Math.random() * 90 + 5,
                    intensity: Math.random() * 0.5 + 0.5,
                    duration: Math.random() * 2000 + 1000
                }));
                
                setNeuralPulses(prev => [...prev, ...newPulses]);
                
                setTimeout(() => {
                    setNeuralPulses(prev => 
                        prev.filter(p => !newPulses.some(np => np.id === p.id))
                    );
                }, 3000);
            };

            intervalRef.current = setInterval(generatePulses, 2000);
            generatePulses();

            return () => {
                if (intervalRef.current) clearInterval(intervalRef.current);
            };
        }
    }, [intelligence.ai_confidence]);

    // Animate brain activity based on learning patterns
    useEffect(() => {
        if (intelligence.patterns_learned) {
            const activity = Math.min((intelligence.patterns_learned / 100) * 100, 100);
            setBrainActivity(activity);
        }
    }, [intelligence.patterns_learned]);

    // Generate synapse connections
    useEffect(() => {
        const connections = Array.from({ length: 12 }, (_, i) => ({
            id: i,
            x1: Math.random() * 100,
            y1: Math.random() * 100,
            x2: Math.random() * 100,
            y2: Math.random() * 100,
            intensity: Math.random() * 0.8 + 0.2,
            delay: Math.random() * 3000
        }));
        setSynapseConnections(connections);
    }, []);

    // Generate data streams
    useEffect(() => {
        const streams = Array.from({ length: 8 }, (_, i) => ({
            id: i,
            side: i % 2 === 0 ? 'left' : 'right',
            speed: Math.random() * 2 + 1,
            delay: Math.random() * 1000,
            data: ['01010101', '11001100', '10101010', '01100110'][Math.floor(Math.random() * 4)]
        }));
        setDataStreams(streams);
    }, []);

    if (loading) {
        return (
            <div className="ai-brain loading">
                <div className="loading-spinner brain-spinner"></div>
                <div className="loading-text">Initializing AI Neural Networks...</div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="ai-brain error">
                <div className="error-icon">🧠💥</div>
                <div className="error-text">AI Brain Disconnected: {error}</div>
            </div>
        );
    }

    return (
        <div className="ai-factory-brain">
            {/* Neural Pulse Effects */}
            <div className="neural-pulse-container">
                {neuralPulses.map(pulse => (
                    <div
                        key={pulse.id}
                        className="neural-pulse"
                        style={{
                            left: `${pulse.x}%`,
                            top: `${pulse.y}%`,
                            animationDuration: `${pulse.duration}ms`,
                            opacity: pulse.intensity
                        }}
                    />
                ))}
            </div>

            {/* Data Streams */}
            <div className="data-streams-container">
                {dataStreams.map(stream => (
                    <div
                        key={stream.id}
                        className={`data-stream ${stream.side}`}
                        style={{
                            animationDuration: `${stream.speed}s`,
                            animationDelay: `${stream.delay}ms`
                        }}
                    >
                        {stream.data}
                    </div>
                ))}
            </div>

            {/* Main AI Brain Display */}
            <div className="main-brain-display holographic">
                <div className="brain-title">
                    <div className="brain-icon pulsing-blue">🧠</div>
                    <div className="neon-text">AI FACTORY BRAIN</div>
                    <div className="brain-icon pulsing-blue">🧠</div>
                </div>

                {/* Core Intelligence Display */}
                <div className="core-intelligence-display">
                    <div className="intelligence-visual">
                        {/* Central Brain Animation */}
                        <div className="brain-core">
                            <div className="brain-visualization">
                                <svg width="200" height="200" className="brain-svg">
                                    {/* Brain Outline */}
                                    <path
                                        d="M50 100 Q50 50, 100 50 Q150 50, 150 100 Q150 150, 100 150 Q50 150, 50 100"
                                        fill="none"
                                        stroke="rgba(55, 66, 250, 0.8)"
                                        strokeWidth="3"
                                        className="brain-outline"
                                    />
                                    
                                    {/* Neural Connections */}
                                    {synapseConnections.map(conn => (
                                        <line
                                            key={conn.id}
                                            x1={`${conn.x1 * 1.5 + 25}%`}
                                            y1={`${conn.y1 * 1.5 + 25}%`}
                                            x2={`${conn.x2 * 1.5 + 25}%`}
                                            y2={`${conn.y2 * 1.5 + 25}%`}
                                            stroke="rgba(0, 255, 136, 0.6)"
                                            strokeWidth="1"
                                            className="synapse-connection"
                                            style={{
                                                animationDelay: `${conn.delay}ms`,
                                                opacity: conn.intensity
                                            }}
                                        />
                                    ))}
                                    
                                    {/* Activity Nodes */}
                                    <circle cx="100" cy="80" r="8" fill="#ff6b35" className="activity-node" />
                                    <circle cx="80" cy="120" r="6" fill="#00ff88" className="activity-node" />
                                    <circle cx="120" cy="120" r="7" fill="#3742fa" className="activity-node" />
                                </svg>

                                {/* Activity Level Ring */}
                                <div className="brain-activity-ring">
                                    <svg width="220" height="220">
                                        <circle
                                            cx="110"
                                            cy="110"
                                            r="100"
                                            fill="none"
                                            stroke="rgba(255, 255, 255, 0.1)"
                                            strokeWidth="3"
                                        />
                                        <circle
                                            cx="110"
                                            cy="110"
                                            r="100"
                                            fill="none"
                                            stroke="rgba(55, 66, 250, 0.8)"
                                            strokeWidth="3"
                                            strokeLinecap="round"
                                            style={{
                                                strokeDasharray: `${2 * Math.PI * 100}`,
                                                strokeDashoffset: `${2 * Math.PI * 100 * (1 - brainActivity / 100)}`,
                                                filter: 'drop-shadow(0 0 10px rgba(55, 66, 250, 0.8))'
                                            }}
                                            transform="rotate(-90 110 110)"
                                        />
                                    </svg>
                                    <div className="activity-percentage">
                                        {brainActivity.toFixed(0)}%
                                    </div>
                                </div>
                            </div>

                            {/* Intelligence Stats */}
                            <div className="intelligence-stats">
                                <IntelligenceMetric
                                    label="AI Confidence"
                                    value={`${intelligence.ai_confidence || 0}%`}
                                    color={intelligence.ai_confidence >= 80 ? "#00ff88" : intelligence.ai_confidence >= 60 ? "#ffa500" : "#ff4757"}
                                    icon="🎯"
                                />
                                <IntelligenceMetric
                                    label="Patterns Learned"
                                    value={intelligence.patterns_learned || 0}
                                    color="#3742fa"
                                    icon="🧩"
                                />
                                <IntelligenceMetric
                                    label="Decision Speed"
                                    value={`${intelligence.decision_speed || 0}ms`}
                                    color="#ff6b35"
                                    icon="⚡"
                                />
                                <IntelligenceMetric
                                    label="Learning Rate"
                                    value={`${intelligence.learning_rate || 0}x`}
                                    color="#e056fd"
                                    icon="📚"
                                />
                            </div>
                        </div>
                    </div>
                </div>

                {/* AI Analysis Dashboard */}
                <div className="ai-analysis-dashboard">
                    <div className="analysis-title">
                        <span className="analysis-icon">📊</span>
                        <span className="neon-text">REAL-TIME AI ANALYSIS</span>
                    </div>

                    <div className="analysis-grid">
                        <AIAnalysisPanel
                            title="Market Sentiment"
                            value={intelligence.market_sentiment || "Analyzing..."}
                            confidence={intelligence.sentiment_confidence || 0}
                            icon="📈"
                            color="#00ff88"
                        />
                        
                        <AIAnalysisPanel
                            title="Risk Assessment"
                            value={intelligence.risk_assessment || "Calculating..."}
                            confidence={intelligence.risk_confidence || 0}
                            icon="⚠️"
                            color="#ffa500"
                        />
                        
                        <AIAnalysisPanel
                            title="Trade Opportunity"
                            value={intelligence.trade_opportunity || "Scanning..."}
                            confidence={intelligence.opportunity_confidence || 0}
                            icon="💎"
                            color="#3742fa"
                        />
                        
                        <AIAnalysisPanel
                            title="Strategy Optimization"
                            value={intelligence.strategy_optimization || "Optimizing..."}
                            confidence={intelligence.optimization_confidence || 0}
                            icon="🔧"
                            color="#e056fd"
                        />
                    </div>
                </div>

                {/* Neural Activity Feed */}
                <div className="neural-activity-feed">
                    <div className="feed-header">
                        <span className="feed-icon">🔬</span>
                        <span className="feed-title">NEURAL ACTIVITY FEED</span>
                    </div>
                    
                    <div className="activity-feed-content">
                        <NeuralActivityItem
                            timestamp="12:34:56"
                            activity="Pattern recognition enhanced"
                            type="learning"
                            confidence={94}
                        />
                        <NeuralActivityItem
                            timestamp="12:34:45"
                            activity="Market correlation identified"
                            type="analysis"
                            confidence={87}
                        />
                        <NeuralActivityItem
                            timestamp="12:34:32"
                            activity="Risk parameters optimized"
                            type="optimization"
                            confidence={91}
                        />
                        <NeuralActivityItem
                            timestamp="12:34:21"
                            activity="Trading signal generated"
                            type="signal"
                            confidence={89}
                        />
                    </div>
                </div>
            </div>
        </div>
    );
};

/**
 * Intelligence Metric Component
 */
const IntelligenceMetric = ({ label, value, color, icon }) => (
    <div className="intelligence-metric">
        <div className="metric-icon" style={{ color }}>{icon}</div>
        <div className="metric-content">
            <div className="metric-value" style={{ color, textShadow: `0 0 10px ${color}` }}>
                {value}
            </div>
            <div className="metric-label">{label}</div>
        </div>
    </div>
);

/**
 * AI Analysis Panel Component
 */
const AIAnalysisPanel = ({ title, value, confidence, icon, color }) => {
    const getConfidenceClass = () => {
        if (confidence >= 80) return 'high-confidence';
        if (confidence >= 60) return 'medium-confidence';
        return 'low-confidence';
    };

    return (
        <div className={`ai-analysis-panel ${getConfidenceClass()}`}>
            <div className="panel-header">
                <span className="panel-icon" style={{ color }}>{icon}</span>
                <span className="panel-title">{title}</span>
            </div>
            
            <div className="panel-content">
                <div className="analysis-value" style={{ color }}>{value}</div>
                <div className="confidence-bar">
                    <div className="confidence-label">Confidence</div>
                    <div className="confidence-progress">
                        <div 
                            className="confidence-fill"
                            style={{ 
                                width: `${confidence}%`,
                                backgroundColor: color,
                                boxShadow: `0 0 10px ${color}`
                            }}
                        />
                    </div>
                    <div className="confidence-value">{confidence}%</div>
                </div>
            </div>
        </div>
    );
};

/**
 * Neural Activity Item Component
 */
const NeuralActivityItem = ({ timestamp, activity, type, confidence }) => {
    const getTypeIcon = () => {
        switch (type) {
            case 'learning': return '🧠';
            case 'analysis': return '🔍';
            case 'optimization': return '⚙️';
            case 'signal': return '📡';
            default: return '💭';
        }
    };

    const getTypeColor = () => {
        switch (type) {
            case 'learning': return '#3742fa';
            case 'analysis': return '#00ff88';
            case 'optimization': return '#ffa500';
            case 'signal': return '#ff6b35';
            default: return '#ffffff';
        }
    };

    return (
        <div className="neural-activity-item">
            <div className="activity-timestamp">{timestamp}</div>
            <div className="activity-content">
                <span 
                    className="activity-icon"
                    style={{ color: getTypeColor() }}
                >
                    {getTypeIcon()}
                </span>
                <span className="activity-text">{activity}</span>
            </div>
            <div className="activity-confidence">
                <span className="confidence-value">{confidence}%</span>
            </div>
        </div>
    );
};

export default AIFactoryBrain;