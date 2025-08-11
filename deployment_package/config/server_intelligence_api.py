#!/usr/bin/env python3
"""
Production Intelligence API Configuration for Server Deployment
Optimized for server environment with proper error handling and logging
"""

import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/trading-bot/logs/intelligence_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Pydantic Models (same as before but with enhanced validation)
class InsightResponse(BaseModel):
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
    timestamp: datetime
    strategy_name: str
    cumulative_return: float
    daily_return: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None

class RecommendationResponse(BaseModel):
    id: str
    title: str
    description: str
    priority: str
    category: str
    confidence: float = Field(ge=0.0, le=1.0)
    expected_improvement: float
    implementation_effort: str
    risk_level: str
    strategy_affected: Optional[str] = None
    created_at: datetime
    estimated_impact: Dict[str, Any]

# FastAPI app with production configuration
app = FastAPI(
    title="Intelligence API - Production",
    description="Learning & Intelligence Dashboard API for Trading Bot",
    version="1.0.0",
    docs_url="/api/docs" if os.getenv("DEBUG", "false").lower() == "true" else None,
    redoc_url="/api/redoc" if os.getenv("DEBUG", "false").lower() == "true" else None
)

# Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("INTELLIGENCE_API_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.utcnow()
    response = await call_next(request)
    process_time = (datetime.utcnow() - start_time).total_seconds()
    
    logger.info(
        f"{request.client.host} - {request.method} {request.url.path} - "
        f"{response.status_code} - {process_time:.3f}s"
    )
    return response

# Health check with detailed system status
@app.get("/api/health")
async def health_check():
    """Enhanced health check for production monitoring"""
    try:
        # Check disk space
        disk_usage = os.statvfs("/opt/trading-bot")
        disk_free_gb = (disk_usage.f_frsize * disk_usage.f_bavail) / (1024**3)
        
        # Check log file sizes
        log_files = {
            "intelligence_api": "/opt/trading-bot/logs/intelligence_api.log",
            "trading_bot": "/opt/trading-bot/logs/trading_bot.log",
            "error": "/opt/trading-bot/logs/error.log"
        }
        
        log_sizes = {}
        for name, path in log_files.items():
            try:
                log_sizes[name] = os.path.getsize(path) / (1024**2) if os.path.exists(path) else 0
            except:
                log_sizes[name] = 0
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "server": {
                "host": os.getenv("SERVER_HOST", "localhost"),
                "port": os.getenv("INTELLIGENCE_API_PORT", "8001"),
                "environment": os.getenv("ENVIRONMENT", "development")
            },
            "system": {
                "disk_free_gb": round(disk_free_gb, 2),
                "log_sizes_mb": log_sizes
            },
            "components": {
                "database": False,  # Will be True when DB is connected
                "learning_pipeline": False,
                "pattern_detector": True,
                "decision_logger": False,
                "redis_cache": False
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

# Enhanced endpoints with production features
@app.get("/api/insights/latest", response_model=List[InsightResponse])
async def get_latest_insights(
    request: Request,
    limit: int = Query(20, ge=1, le=100),
    hours: int = Query(24, ge=1, le=168)
):
    """Get latest insights with request tracking"""
    try:
        logger.info(f"Insights request from {request.client.host}: limit={limit}, hours={hours}")
        
        # In production, this would query the actual database
        # For now, return enhanced mock data
        mock_insights = [
            InsightResponse(
                id=f"insight_prod_{i}",
                timestamp=datetime.utcnow() - timedelta(minutes=i*30),
                type="pattern_discovery",
                title=f"Market Pattern #{i+1} Detected",
                description=f"AI detected profitable pattern with {85+i}% confidence",
                confidence=0.85 + (i * 0.02),
                impact_score=0.80 + (i * 0.03),
                strategy_affected="momentum_strategy",
                category="discovery"
            ) for i in range(min(limit, 10))
        ]
        
        return mock_insights[:limit]
        
    except Exception as e:
        logger.error(f"Failed to get insights: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve insights")

# Similar enhanced endpoints for patterns, performance, and recommendations
@app.get("/api/patterns/discovered", response_model=List[PatternResponse])
async def get_discovered_patterns(
    request: Request,
    limit: int = Query(10, ge=1, le=50),
    min_confidence: float = Query(0.5, ge=0.0, le=1.0)
):
    """Get discovered patterns with enhanced logging"""
    try:
        logger.info(f"Patterns request from {request.client.host}: limit={limit}, confidence>={min_confidence}")
        
        mock_patterns = [
            PatternResponse(
                id="pattern_prod_1",
                pattern_type="success_pattern",
                name="Server High-Frequency Pattern",
                description="High-frequency pattern detected in server environment",
                confidence=0.92,
                frequency=156,
                success_rate=0.78,
                profit_potential=0.28,
                risk_level="medium",
                strategies_involved=["momentum_strategy", "volatility_strategy"],
                market_conditions={"volatility": "high", "volume": "above_average", "server": True},
                discovered_at=datetime.utcnow() - timedelta(days=1),
                last_seen=datetime.utcnow() - timedelta(minutes=15)
            )
        ]
        
        filtered = [p for p in mock_patterns if p.confidence >= min_confidence]
        return filtered[:limit]
        
    except Exception as e:
        logger.error(f"Failed to get patterns: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve patterns")

# Server monitoring endpoint
@app.get("/api/server/stats")
async def get_server_stats():
    """Server-specific statistics"""
    try:
        import psutil
        
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "uptime_hours": (datetime.utcnow() - datetime.fromtimestamp(psutil.boot_time())).total_seconds() / 3600,
            "active_connections": len(psutil.net_connections()),
            "server_ip": "85.215.183.30",
            "timestamp": datetime.utcnow()
        }
    except ImportError:
        return {"message": "psutil not installed, server stats unavailable"}
    except Exception as e:
        logger.error(f"Failed to get server stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve server stats")

if __name__ == "__main__":
    # Production server configuration
    host = os.getenv("INTELLIGENCE_API_HOST", "0.0.0.0")
    port = int(os.getenv("INTELLIGENCE_API_PORT", "8001"))
    
    logger.info(f"🚀 Starting Intelligence API Server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
        reload=False,  # No reload in production
        workers=1 if os.getenv("DEBUG") else 4  # Multiple workers in production
    )