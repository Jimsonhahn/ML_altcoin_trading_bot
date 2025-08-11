#!/usr/bin/env python3
"""
Test script for Intelligence API
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
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

# FastAPI app
app = FastAPI(
    title="Intelligence API Test",
    description="Learning & Intelligence Dashboard API - Test Version",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health Check Endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "components": {
            "database": False,  # Mock mode
            "learning_pipeline": False,
            "pattern_detector": True,
            "decision_logger": False
        }
    }

# Main Intelligence Endpoints

@app.get("/api/insights/latest", response_model=List[InsightResponse])
async def get_latest_insights(
    limit: int = Query(20, ge=1, le=100),
    hours: int = Query(24, ge=1, le=168)
):
    """Neueste Insights und Entdeckungen abrufen"""
    
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
    
    return mock_insights[:limit]

@app.get("/api/patterns/discovered", response_model=List[PatternResponse])
async def get_discovered_patterns(
    limit: int = Query(10, ge=1, le=50),
    min_confidence: float = Query(0.5, ge=0.0, le=1.0)
):
    """Gefundene Muster mit Confidence Scores abrufen"""
    
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
            success_rate=0.05,
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
        )
    ]
    
    filtered_patterns = [p for p in mock_patterns if p.confidence >= min_confidence]
    return filtered_patterns[:limit]

@app.get("/api/performance/evolution", response_model=List[PerformanceDataPoint])
async def get_performance_evolution(
    period: str = Query("7d", regex="^(7d|30d|90d)$"),
    strategies: List[str] = Query(None)
):
    """Performance-Evolution aller Strategien abrufen"""
    
    period_days = {"7d": 7, "30d": 30, "90d": 90}[period]
    mock_strategies = ["momentum_strategy", "mean_reversion", "volatility_strategy"]
    base_date = datetime.utcnow() - timedelta(days=period_days)
    performance_data = []
    
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

@app.get("/api/recommendations", response_model=List[RecommendationResponse])
async def get_recommendations(
    priority: Optional[str] = Query(None, regex="^(high|medium|low)$"),
    category: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=50)
):
    """Empfohlene Änderungen und Aktionen abrufen"""
    
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
        )
    ]
    
    # Filter nach Priorität
    if priority:
        mock_recommendations = [r for r in mock_recommendations if r.priority == priority]
    
    # Filter nach Kategorie
    if category:
        mock_recommendations = [r for r in mock_recommendations if r.category == category]
    
    return mock_recommendations[:limit]

@app.get("/api/insights/stats")
async def get_insights_stats():
    """Statistiken über Insights"""
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

@app.get("/api/patterns/stats")
async def get_patterns_stats():
    """Statistiken über entdeckte Muster"""
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

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Intelligence API Test Server...")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")