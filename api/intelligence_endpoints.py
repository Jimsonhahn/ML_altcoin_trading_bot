#!/usr/bin/env python3
"""
ML Altcoin Trading Bot - Intelligence API Endpoints
REST API für Learning & Intelligence Dashboard Features

Diese API stellt folgende Endpoints bereit:
- GET /api/insights/latest - Neueste Insights und Entdeckungen
- GET /api/patterns/discovered - Gefundene Muster mit Confidence Scores
- GET /api/performance/evolution - Performance-Evolution aller Strategien
- GET /api/recommendations - Empfohlene Änderungen und Aktionen
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncpg
from pydantic import BaseModel, Field

# Import unserer Learning Components
from Analysis.learning_pipeline import LearningPipeline
from analysis.pattern_detector import PatternDetector
from analysis.backtest_improvements import BacktestImprovements
from core.decision_logger import DecisionLogger

logger = logging.getLogger(__name__)

# Pydantic Models für API Responses
class InsightResponse(BaseModel):
    """Response model für Insights"""
    id: str
    timestamp: datetime
    type: str
    title: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    impact_score: float = Field(ge=0.0, le=1.0)
    strategy_affected: Optional[str] = None
    actionable: bool = True
    category: str = "general"
    
class PatternResponse(BaseModel):
    """Response model für entdeckte Muster"""
    id: str
    pattern_type: str
    name: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    frequency: int
    success_rate: float = Field(ge=0.0, le=1.0)
    profit_potential: float
    risk_level: str
    strategies_involved: List[str]
    market_conditions: Dict[str, Any]
    discovered_at: datetime
    last_seen: datetime

class PerformanceDataPoint(BaseModel):
    """Performance data point für Charts"""
    timestamp: datetime
    strategy_name: str
    cumulative_return: float
    daily_return: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    
class RecommendationResponse(BaseModel):
    """Response model für Empfehlungen"""
    id: str
    title: str
    description: str
    priority: str  # high, medium, low
    category: str
    confidence: float = Field(ge=0.0, le=1.0)
    expected_improvement: float
    implementation_effort: str  # low, medium, high
    risk_level: str  # low, medium, high
    strategy_affected: Optional[str] = None
    created_at: datetime
    estimated_impact: Dict[str, Any]

# Global instances (würden normalerweise durch Dependency Injection verwaltet)
db_pool: Optional[asyncpg.Pool] = None
learning_pipeline: Optional[LearningPipeline] = None
pattern_detector: Optional[PatternDetector] = None
backtest_improvements: Optional[BacktestImprovements] = None
decision_logger: Optional[DecisionLogger] = None

# FastAPI app
app = FastAPI(
    title="Intelligence API",
    description="Learning & Intelligence Dashboard API",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In Produktion spezifische Origins verwenden
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency für Database Connection
async def get_db_pool():
    """Database connection pool dependency"""
    if db_pool is None:
        raise HTTPException(status_code=503, detail="Database not available")
    return db_pool

# Dependency für Learning Pipeline
async def get_learning_pipeline():
    """Learning pipeline dependency"""
    if learning_pipeline is None:
        raise HTTPException(status_code=503, detail="Learning pipeline not available")
    return learning_pipeline

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global db_pool, learning_pipeline, pattern_detector, backtest_improvements, decision_logger
    
    try:
        # Initialize database pool - nutze SQLite aus config.yaml
        import yaml
        import os
        
        # Load config
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            db_url = config.get('data', {}).get('database_url', 'sqlite:///data/trading_bot.db')
            
            # Für Entwicklung verwenden wir PostgreSQL falls verfügbar, sonst SQLite Mock
            try:
                db_config = {
                    "host": "localhost",
                    "port": 5432,
                    "database": "trading_bot",
                    "user": "postgres",
                    "password": "password"
                }
            except:
                # Fallback zu Mock-Modus ohne echte DB
                logger.warning("Database not available, using mock mode")
                db_config = None
        else:
            db_config = None
        # Try to create database pool, fallback to mock mode
        if db_config:
            try:
                db_pool = await asyncpg.create_pool(**db_config)
                logger.info("Database connection established")
            except Exception as e:
                logger.warning(f"Database connection failed: {e}, using mock mode")
                db_pool = None
        else:
            logger.info("Using mock mode (no database configuration)")
            db_pool = None
        
        # Initialize learning components
        try:
            if db_pool:
                learning_pipeline = LearningPipeline(
                    db_pool=db_pool,
                    lookback_days=30,
                    min_trades_for_analysis=10,
                    confidence_threshold=0.7
                )
                
                decision_logger = DecisionLogger(db_pool)
                await decision_logger.start()
            else:
                learning_pipeline = None
                decision_logger = None
                
            pattern_detector = PatternDetector(
                lookback_days=30,
                min_pattern_frequency=5
            )
            
            backtest_improvements = BacktestImprovements(
                data_handler=None,  # Würde echten Handler verwenden
                backtest_engine=None
            )
            
        except Exception as e:
            logger.warning(f"Failed to initialize some components: {e}, using fallback mode")
            learning_pipeline = None
            pattern_detector = PatternDetector(lookback_days=30, min_pattern_frequency=5)
            backtest_improvements = None
            decision_logger = None
        
        logger.info("Intelligence API initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Intelligence API: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global db_pool, decision_logger
    
    if decision_logger:
        await decision_logger.stop()
    
    if db_pool:
        await db_pool.close()
        
    logger.info("Intelligence API shutdown complete")

# Health Check Endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "components": {
            "database": db_pool is not None,
            "learning_pipeline": learning_pipeline is not None,
            "pattern_detector": pattern_detector is not None,
            "decision_logger": decision_logger is not None
        }
    }

# Main Intelligence Endpoints

@app.get("/api/insights/latest", response_model=List[InsightResponse])
async def get_latest_insights(
    limit: int = Query(20, ge=1, le=100),
    hours: int = Query(24, ge=1, le=168)
):
    """
    Neueste Insights und Entdeckungen abrufen
    
    Args:
        limit: Maximale Anzahl Insights (1-100)
        hours: Zeitraum in Stunden (1-168)
    """
    try:
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        # Mock insights - in echter Implementierung aus Datenbank/Learning Pipeline
        insights = []
        
        # Beispiel Insights basierend auf Learning Pipeline
        if learning_pipeline:
            try:
                # Hier würden echte Insights aus der Learning Pipeline kommen
                recent_analysis = await learning_pipeline.run_full_analysis()
                
                for i, finding in enumerate(recent_analysis.get('key_findings', [])[:limit]):
                    insight = InsightResponse(
                        id=f"insight_{i}_{int(datetime.utcnow().timestamp())}",
                        timestamp=datetime.utcnow() - timedelta(minutes=i*30),
                        type="strategy_optimization",
                        title=f"Strategy Optimization Opportunity",
                        description=str(finding),
                        confidence=0.8 + (i * 0.02),  # Variable confidence
                        impact_score=0.7 + (i * 0.03),
                        strategy_affected="momentum_strategy",
                        category="performance"
                    )
                    insights.append(insight)
                    
            except Exception as e:
                logger.error(f"Failed to get insights from learning pipeline: {e}")
        
        # Füge einige Mock Insights hinzu falls keine echten verfügbar
        if len(insights) < 3:
            mock_insights = [
                InsightResponse(
                    id="insight_mock_1",
                    timestamp=datetime.utcnow() - timedelta(minutes=30),
                    type="pattern_discovery",
                    title="Neue profitable Marktbedingung entdeckt",
                    description="Hohe Volatilität + niedrige Korrelation führt zu 15% besserer Performance",
                    confidence=0.87,
                    impact_score=0.92,
                    strategy_affected="volatility_strategy",
                    category="discovery"
                ),
                InsightResponse(
                    id="insight_mock_2",
                    timestamp=datetime.utcnow() - timedelta(minutes=45),
                    type="risk_alert",
                    title="Erhöhtes Korrelationsrisiko erkannt",
                    description="Strategien zeigen ungewöhnlich hohe Korrelation - Diversifikation empfohlen",
                    confidence=0.91,
                    impact_score=0.85,
                    category="risk"
                ),
                InsightResponse(
                    id="insight_mock_3",
                    timestamp=datetime.utcnow() - timedelta(minutes=60),
                    type="optimization",
                    title="Timing-Optimierung identifiziert",
                    description="Entry-Timing kann um 12% verbessert werden durch ML-Signal",
                    confidence=0.79,
                    impact_score=0.76,
                    strategy_affected="momentum_strategy",
                    category="optimization"
                )
            ]
            insights.extend(mock_insights[:limit - len(insights)])
        
        return insights[:limit]
        
    except Exception as e:
        logger.error(f"Failed to get latest insights: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve insights")

@app.get("/api/patterns/discovered", response_model=List[PatternResponse])
async def get_discovered_patterns(
    limit: int = Query(10, ge=1, le=50),
    min_confidence: float = Query(0.5, ge=0.0, le=1.0)
):
    """
    Gefundene Muster mit Confidence Scores abrufen
    
    Args:
        limit: Maximale Anzahl Muster
        min_confidence: Minimale Confidence (0.0-1.0)
    """
    try:
        patterns = []
        
        # Mock patterns - in echter Implementierung aus Pattern Detector
        mock_patterns = [
            PatternResponse(
                id="pattern_1",
                pattern_type="success_pattern",
                name="High Volatility Momentum",
                description="Bei Volatilität >15% zeigt Momentum-Strategy 23% höhere Erfolgsrate",
                confidence=0.89,
                frequency=47,
                success_rate=0.73,
                profit_potential=0.23,
                risk_level="medium",
                strategies_involved=["momentum_strategy", "volatility_strategy"],
                market_conditions={
                    "volatility_range": "15-25%",
                    "volume_spike": True,
                    "trend_strength": "strong"
                },
                discovered_at=datetime.utcnow() - timedelta(days=2),
                last_seen=datetime.utcnow() - timedelta(hours=3)
            ),
            PatternResponse(
                id="pattern_2",
                pattern_type="dangerous_condition",
                name="Flash Crash Risk",
                description="Kombination aus niedrigem Volumen + hoher Leverage deutet auf Flash Crash hin",
                confidence=0.76,
                frequency=12,
                success_rate=0.05,  # Niedrige Erfolgsrate = Gefahr
                profit_potential=-0.18,
                risk_level="high",
                strategies_involved=["all_strategies"],
                market_conditions={
                    "volume_ma_ratio": "<0.3",
                    "leverage_ratio": ">2.5",
                    "fear_greed_index": "<20"
                },
                discovered_at=datetime.utcnow() - timedelta(days=5),
                last_seen=datetime.utcnow() - timedelta(hours=1)
            ),
            PatternResponse(
                id="pattern_3",
                pattern_type="strategy_synergy",
                name="Mean Reversion + Momentum Combo",
                description="Sequentielle Anwendung von Mean Reversion gefolgt von Momentum",
                confidence=0.82,
                frequency=34,
                success_rate=0.68,
                profit_potential=0.19,
                risk_level="low",
                strategies_involved=["mean_reversion", "momentum_strategy"],
                market_conditions={
                    "regime": "ranging",
                    "rsi_range": "20-80",
                    "volatility": "medium"
                },
                discovered_at=datetime.utcnow() - timedelta(days=7),
                last_seen=datetime.utcnow() - timedelta(hours=6)
            )
        ]
        
        # Filter nach Confidence
        filtered_patterns = [p for p in mock_patterns if p.confidence >= min_confidence]
        
        return filtered_patterns[:limit]
        
    except Exception as e:
        logger.error(f"Failed to get discovered patterns: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve patterns")

@app.get("/api/performance/evolution", response_model=List[PerformanceDataPoint])
async def get_performance_evolution(
    period: str = Query("7d", regex="^(7d|30d|90d)$"),
    strategies: List[str] = Query(None)
):
    """
    Performance-Evolution aller Strategien abrufen
    
    Args:
        period: Zeitraum (7d, 30d, 90d)
        strategies: Liste spezifischer Strategien (optional)
    """
    try:
        # Parse period
        period_days = {"7d": 7, "30d": 30, "90d": 90}[period]
        cutoff_date = datetime.utcnow() - timedelta(days=period_days)
        
        # Query performance data if database available
        performance_data = []
        
        if db_pool:
            try:
                async with db_pool.acquire() as conn:
                    query = """
                    SELECT 
                        strategy_name,
                        DATE_TRUNC('day', entry_timestamp) as date,
                        COUNT(*) as trades,
                        AVG(pnl_percentage) as avg_return,
                        SUM(pnl_absolute) as total_pnl
                    FROM strategy_performance 
                    WHERE entry_timestamp >= $1
                        AND trade_status = 'closed'
                        AND ($2::text[] IS NULL OR strategy_name = ANY($2))
                    GROUP BY strategy_name, DATE_TRUNC('day', entry_timestamp)
                    ORDER BY date ASC, strategy_name
                    """
                    
                    rows = await conn.fetch(query, cutoff_date, strategies)
            
                    # Convert zu Performance Data Points
                    cumulative_returns = {}  # Track cumulative returns per strategy
                    
                    for row in rows:
                        strategy = row['strategy_name']
                        date = row['date']
                        daily_return = float(row['avg_return'] or 0) / 100  # Convert percentage
                        
                        # Calculate cumulative return
                        if strategy not in cumulative_returns:
                            cumulative_returns[strategy] = 1.0
                        
                        cumulative_returns[strategy] *= (1 + daily_return)
                        
                        # Mock additional metrics (in echter Implementierung berechnet)
                        sharpe_ratio = max(0.5, daily_return * 15) if daily_return > 0 else None
                        max_drawdown = abs(daily_return) * 2 if daily_return < 0 else None
                        win_rate = 0.6 + (daily_return * 0.4) if daily_return > 0 else 0.4
                        
                        data_point = PerformanceDataPoint(
                            timestamp=date,
                            strategy_name=strategy,
                            cumulative_return=(cumulative_returns[strategy] - 1) * 100,
                            daily_return=daily_return * 100,
                            sharpe_ratio=sharpe_ratio,
                            max_drawdown=max_drawdown,
                            win_rate=min(1.0, max(0.0, win_rate))
                        )
                        
                        performance_data.append(data_point)
            except Exception as e:
                logger.warning(f"Database query failed: {e}, using mock data")
        
        # Füge Mock Daten hinzu falls keine echten Daten
        if not performance_data:
            mock_strategies = ["momentum_strategy", "mean_reversion", "volatility_strategy"]
            base_date = datetime.utcnow() - timedelta(days=period_days)
            
            for i in range(period_days):
                current_date = base_date + timedelta(days=i)
                
                for strategy in mock_strategies:
                    # Generate realistic performance data
                    daily_return = (hash(f"{strategy}_{i}") % 200 - 100) / 1000  # -10% to +10%
                    cumulative = (1 + daily_return/100) ** i * 100 - 100
                    
                    data_point = PerformanceDataPoint(
                        timestamp=current_date,
                        strategy_name=strategy,
                        cumulative_return=cumulative,
                        daily_return=daily_return,
                        sharpe_ratio=1.2 + (daily_return * 0.1),
                        max_drawdown=abs(daily_return) if daily_return < 0 else 0,
                        win_rate=0.6 + (daily_return * 0.01)
                    )
                    
                    performance_data.append(data_point)
        
        return performance_data
        
    except Exception as e:
        logger.error(f"Failed to get performance evolution: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve performance data")

@app.get("/api/recommendations", response_model=List[RecommendationResponse])
async def get_recommendations(
    priority: Optional[str] = Query(None, regex="^(high|medium|low)$"),
    category: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=50)
):
    """
    Empfohlene Änderungen und Aktionen abrufen
    
    Args:
        priority: Filter nach Priorität
        category: Filter nach Kategorie
        limit: Maximale Anzahl Empfehlungen
    """
    try:
        # Mock recommendations - in echter Implementierung aus Learning Pipeline
        mock_recommendations = [
            RecommendationResponse(
                id="rec_1",
                title="Erhöhe Momentum Strategy Allocation",
                description="Momentum Strategy zeigt 15% bessere Performance unter aktuellen Marktbedingungen",
                priority="high",
                category="allocation",
                confidence=0.87,
                expected_improvement=0.15,
                implementation_effort="low",
                risk_level="low",
                strategy_affected="momentum_strategy",
                created_at=datetime.utcnow() - timedelta(hours=2),
                estimated_impact={
                    "return_increase": "12-18%",
                    "risk_change": "minimal",
                    "implementation_time": "immediate"
                }
            ),
            RecommendationResponse(
                id="rec_2",
                title="Implementiere ML-Enhanced Entry Signals",
                description="Machine Learning Modell kann Entry-Timing um 23% verbessern",
                priority="medium",
                category="optimization",
                confidence=0.79,
                expected_improvement=0.23,
                implementation_effort="high",
                risk_level="medium",
                created_at=datetime.utcnow() - timedelta(hours=4),
                estimated_impact={
                    "return_increase": "20-25%",
                    "risk_change": "slight increase",
                    "implementation_time": "2-3 days"
                }
            ),
            RecommendationResponse(
                id="rec_3",
                title="Reduziere Korrelationsrisiko",
                description="Aktuelle Strategy-Allokation zeigt zu hohe Korrelation in Crash-Szenarien",
                priority="high",
                category="risk_management",
                confidence=0.91,
                expected_improvement=0.08,
                implementation_effort="medium",
                risk_level="high",
                created_at=datetime.utcnow() - timedelta(hours=1),
                estimated_impact={
                    "risk_reduction": "35-45%",
                    "return_impact": "minimal",
                    "implementation_time": "1 day"
                }
            ),
            RecommendationResponse(
                id="rec_4",
                title="Aktiviere Volatility Breakout Detection",
                description="Volatility Spikes können 67% früher erkannt werden mit neuen Indikatoren",
                priority="medium",
                category="feature",
                confidence=0.74,
                expected_improvement=0.12,
                implementation_effort="low",
                risk_level="low",
                created_at=datetime.utcnow() - timedelta(hours=6),
                estimated_impact={
                    "early_detection": "67% faster",
                    "false_positives": "reduced by 23%",
                    "implementation_time": "4 hours"
                }
            )
        ]
        
        # Filter nach Priorität
        if priority:
            mock_recommendations = [r for r in mock_recommendations if r.priority == priority]
        
        # Filter nach Kategorie
        if category:
            mock_recommendations = [r for r in mock_recommendations if r.category == category]
        
        # Sort by priority and confidence
        priority_order = {"high": 3, "medium": 2, "low": 1}
        mock_recommendations.sort(
            key=lambda x: (priority_order[x.priority], x.confidence), 
            reverse=True
        )
        
        return mock_recommendations[:limit]
        
    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve recommendations")

# Additional Utility Endpoints

@app.get("/api/insights/stats")
async def get_insights_stats():
    """Statistiken über Insights"""
    try:
        return {
            "total_insights_today": 12,
            "high_confidence_insights": 8,
            "actionable_insights": 10,
            "categories": {
                "optimization": 5,
                "risk": 3,
                "discovery": 4
            },
            "avg_confidence": 0.83,
            "last_generated": datetime.utcnow() - timedelta(minutes=15)
        }
    except Exception as e:
        logger.error(f"Failed to get insights stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve stats")

@app.get("/api/patterns/stats")
async def get_patterns_stats():
    """Statistiken über entdeckte Muster"""
    try:
        return {
            "total_patterns": 23,
            "new_patterns_today": 3,
            "high_confidence_patterns": 15,
            "pattern_types": {
                "success_patterns": 12,
                "dangerous_conditions": 4,
                "strategy_synergies": 7
            },
            "avg_success_rate": 0.68,
            "most_profitable_pattern": "High Volatility Momentum"
        }
    except Exception as e:
        logger.error(f"Failed to get patterns stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve stats")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")