#!/usr/bin/env python3
"""
Intelligence API für Windows Server 2022
Optimiert für Windows-Umgebung mit erweiterten Features
"""

import os
import sys
import logging
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Windows-spezifische Imports
import psutil
import winsound  # Für Windows-Benachrichtigungen

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel, Field

# Windows Logging Setup
log_dir = Path("C:/TradingBot/logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'intelligence_api.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Pydantic Models (gleich wie vorher, aber mit Windows-Optimierungen)
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

# FastAPI App mit Windows-Konfiguration
app = FastAPI(
    title="Intelligence API - Windows Server 2022",
    description="Learning & Intelligence Dashboard API für Trading Bot",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In Produktion spezielle Origins verwenden
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Windows-spezifisches Request Logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.utcnow()
    response = await call_next(request)
    process_time = (datetime.utcnow() - start_time).total_seconds()
    
    # Windows Event Log könnte hier hinzugefügt werden
    logger.info(
        f"{request.client.host} - {request.method} {request.url.path} - "
        f"{response.status_code} - {process_time:.3f}s"
    )
    return response

# Windows System Health Check
@app.get("/api/health")
async def health_check():
    """Windows-spezifischer Health Check"""
    try:
        # Windows System Info
        system_info = {
            "os": f"{psutil.sys.platform} {psutil.version_info}",
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "memory_percent": psutil.virtual_memory().percent
        }
        
        # Disk Info für C:\ Laufwerk
        disk_usage = psutil.disk_usage('C:\\')
        disk_info = {
            "total_gb": round(disk_usage.total / (1024**3), 2),
            "free_gb": round(disk_usage.free / (1024**3), 2),
            "used_percent": round((disk_usage.used / disk_usage.total) * 100, 2)
        }
        
        # Windows Services Status
        services_status = {}
        try:
            # Prüfe ob IIS läuft
            for service in psutil.win_service_iter():
                if service.name() in ['W3SVC', 'WAS']:  # IIS Services
                    services_status[service.name()] = service.status()
        except:
            services_status = {"note": "Service check requires admin rights"}
        
        # Log Files Check
        log_files = {}
        log_dir = Path("C:/TradingBot/logs")
        if log_dir.exists():
            for log_file in log_dir.glob("*.log"):
                try:
                    size_mb = log_file.stat().st_size / (1024**2)
                    log_files[log_file.name] = round(size_mb, 2)
                except:
                    log_files[log_file.name] = "access_denied"
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "server": {
                "platform": "Windows Server 2022",
                "host": "85.215.183.30",
                "port": os.getenv("INTELLIGENCE_API_PORT", "8001")
            },
            "system": system_info,
            "disk": disk_info,
            "services": services_status,
            "logs": log_files,
            "components": {
                "database": False,
                "learning_pipeline": False,
                "pattern_detector": True,
                "decision_logger": False,
                "windows_services": True
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

# Windows-optimierte Insights
@app.get("/api/insights/latest", response_model=List[InsightResponse])
async def get_latest_insights(
    request: Request,
    limit: int = Query(20, ge=1, le=100),
    hours: int = Query(24, ge=1, le=168)
):
    """Windows-optimierte Insights mit Enhanced Logging"""
    try:
        logger.info(f"Insights request from {request.client.host}: limit={limit}, hours={hours}")
        
        # Windows-spezifische Insights
        mock_insights = [
            InsightResponse(
                id="windows_insight_1",
                timestamp=datetime.utcnow() - timedelta(minutes=15),
                type="system_optimization",
                title="Windows Server Performance Optimierung",
                description="IIS und .NET Optimierungen können Trading Performance um 18% steigern",
                confidence=0.89,
                impact_score=0.85,
                strategy_affected="all_strategies",
                category="system"
            ),
            InsightResponse(
                id="windows_insight_2",
                timestamp=datetime.utcnow() - timedelta(minutes=30),
                type="pattern_discovery",
                title="High-Frequency Pattern auf Windows",
                description="Optimierte Thread-Pool Konfiguration verbessert Latenz um 23%",
                confidence=0.92,
                impact_score=0.88,
                strategy_affected="momentum_strategy",
                category="performance"
            ),
            InsightResponse(
                id="windows_insight_3",
                timestamp=datetime.utcnow() - timedelta(minutes=45),
                type="resource_monitoring",
                title="Memory Management Optimierung",
                description="Windows Memory Compression kann Trading Bot Effizienz steigern",
                confidence=0.76,
                impact_score=0.71,
                category="optimization"
            )
        ]
        
        return mock_insights[:limit]
        
    except Exception as e:
        logger.error(f"Failed to get insights: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve insights")

# Windows Performance Patterns
@app.get("/api/patterns/discovered", response_model=List[PatternResponse])
async def get_discovered_patterns(
    request: Request,
    limit: int = Query(10, ge=1, le=50),
    min_confidence: float = Query(0.5, ge=0.0, le=1.0)
):
    """Windows-spezifische Pattern Detection"""
    try:
        logger.info(f"Patterns request from {request.client.host}: limit={limit}, confidence>={min_confidence}")
        
        mock_patterns = [
            PatternResponse(
                id="windows_pattern_1",
                pattern_type="success_pattern",
                name="Windows IIS High-Performance Pattern",
                description="IIS Application Pool Optimierung für Trading APIs",
                confidence=0.94,
                frequency=234,
                success_rate=0.82,
                profit_potential=0.31,
                risk_level="low",
                strategies_involved=["momentum_strategy", "arbitrage_strategy"],
                market_conditions={
                    "platform": "Windows Server 2022",
                    "iis_version": "10.0",
                    "dotnet_optimization": True,
                    "memory_management": "optimized"
                },
                discovered_at=datetime.utcnow() - timedelta(days=1),
                last_seen=datetime.utcnow() - timedelta(minutes=12)
            ),
            PatternResponse(
                id="windows_pattern_2",
                pattern_type="performance_pattern",
                name="Windows Task Scheduler Integration",
                description="Automatisierte Rebalancing über Windows Task Scheduler",
                confidence=0.87,
                frequency=67,
                success_rate=0.75,
                profit_potential=0.19,
                risk_level="medium",
                strategies_involved=["rebalancing_strategy"],
                market_conditions={
                    "scheduler": "Windows Task Scheduler",
                    "automation_level": "high",
                    "execution_time": "optimal"
                },
                discovered_at=datetime.utcnow() - timedelta(days=3),
                last_seen=datetime.utcnow() - timedelta(minutes=28)
            )
        ]
        
        filtered = [p for p in mock_patterns if p.confidence >= min_confidence]
        return filtered[:limit]
        
    except Exception as e:
        logger.error(f"Failed to get patterns: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve patterns")

# Performance Evolution (gleich wie Linux Version)
@app.get("/api/performance/evolution", response_model=List[PerformanceDataPoint])
async def get_performance_evolution(
    request: Request,
    period: str = Query("7d", regex="^(7d|30d|90d)$"),
    strategies: List[str] = Query(None)
):
    """Performance Evolution mit Windows-optimierten Metriken"""
    try:
        period_days = {"7d": 7, "30d": 30, "90d": 90}[period]
        mock_strategies = ["momentum_strategy", "mean_reversion", "volatility_strategy", "windows_optimized_strategy"]
        base_date = datetime.utcnow() - timedelta(days=period_days)
        performance_data = []
        
        for i in range(period_days):
            current_date = base_date + timedelta(days=i)
            
            for strategy in mock_strategies:
                # Windows-optimierte Performance-Simulation
                daily_return = (hash(f"{strategy}_{i}_windows") % 200 - 100) / 1000
                if strategy == "windows_optimized_strategy":
                    daily_return *= 1.15  # Windows-Optimierung Bonus
                
                cumulative = (1 + daily_return/100) ** i * 100 - 100
                
                data_point = PerformanceDataPoint(
                    timestamp=current_date,
                    strategy_name=strategy,
                    cumulative_return=cumulative,
                    daily_return=daily_return,
                    sharpe_ratio=1.3 + (daily_return * 0.1),
                    max_drawdown=abs(daily_return) if daily_return < 0 else 0,
                    win_rate=0.65 + (daily_return * 0.01)
                )
                
                performance_data.append(data_point)
        
        return performance_data
        
    except Exception as e:
        logger.error(f"Failed to get performance evolution: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve performance data")

# Windows-spezifische Recommendations
@app.get("/api/recommendations", response_model=List[RecommendationResponse])
async def get_recommendations(
    request: Request,
    priority: Optional[str] = Query(None, regex="^(high|medium|low)$"),
    category: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=50)
):
    """Windows-optimierte Empfehlungen"""
    try:
        mock_recommendations = [
            RecommendationResponse(
                id="windows_rec_1",
                title="IIS Application Pool Optimierung",
                description="Konfiguriere IIS App Pool für bessere API Performance",
                priority="high",
                category="system_optimization",
                confidence=0.91,
                expected_improvement=0.22,
                implementation_effort="medium",
                risk_level="low",
                strategy_affected="all_apis",
                created_at=datetime.utcnow() - timedelta(hours=1),
                estimated_impact={
                    "api_response_time": "22% faster",
                    "concurrent_requests": "40% more",
                    "memory_usage": "15% less"
                }
            ),
            RecommendationResponse(
                id="windows_rec_2",
                title="Windows Performance Toolkit Setup",
                description="Nutze Windows Performance Counter für besseres Monitoring",
                priority="medium",
                category="monitoring",
                confidence=0.84,
                expected_improvement=0.16,
                implementation_effort="low",
                risk_level="low",
                created_at=datetime.utcnow() - timedelta(hours=3),
                estimated_impact={
                    "monitoring_accuracy": "35% better",
                    "issue_detection": "50% faster",
                    "system_insights": "enhanced"
                }
            ),
            RecommendationResponse(
                id="windows_rec_3",
                title="Windows Task Scheduler Integration",
                description="Automatisiere Backups und Maintenance mit Task Scheduler",
                priority="medium",
                category="automation",
                confidence=0.78,
                expected_improvement=0.12,
                implementation_effort="low",
                risk_level="low",
                created_at=datetime.utcnow() - timedelta(hours=5),
                estimated_impact={
                    "automation_level": "90% automated",
                    "maintenance_time": "60% reduced",
                    "reliability": "significantly improved"
                }
            )
        ]
        
        # Filter und Sort
        if priority:
            mock_recommendations = [r for r in mock_recommendations if r.priority == priority]
        if category:
            mock_recommendations = [r for r in mock_recommendations if r.category == category]
        
        priority_order = {"high": 3, "medium": 2, "low": 1}
        mock_recommendations.sort(
            key=lambda x: (priority_order[x.priority], x.confidence), 
            reverse=True
        )
        
        return mock_recommendations[:limit]
        
    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve recommendations")

# Windows System Stats
@app.get("/api/server/windows-stats")
async def get_windows_stats():
    """Windows-spezifische System-Statistiken"""
    try:
        # Windows Services Status
        services = {}
        try:
            for service in psutil.win_service_iter():
                if service.name() in ['W3SVC', 'WAS', 'MSSQLSERVER', 'SQLSERVERAGENT']:
                    services[service.name()] = {
                        "status": service.status(),
                        "start_type": service.start_type()
                    }
        except:
            services = {"note": "Admin rights required for service details"}
        
        # Windows Performance Counters
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = datetime.now() - boot_time
        
        return {
            "platform": "Windows Server 2022",
            "server_ip": "85.215.183.30",
            "uptime_hours": round(uptime.total_seconds() / 3600, 2),
            "boot_time": boot_time.isoformat(),
            "cpu": {
                "percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
                "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            },
            "memory": {
                "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                "percent": psutil.virtual_memory().percent
            },
            "disk": {
                "total_gb": round(psutil.disk_usage('C:\\').total / (1024**3), 2),
                "free_gb": round(psutil.disk_usage('C:\\').free / (1024**3), 2),
                "percent": round((psutil.disk_usage('C:\\').used / psutil.disk_usage('C:\\').total) * 100, 2)
            },
            "network": {
                "connections": len(psutil.net_connections()),
                "io_counters": psutil.net_io_counters()._asdict()
            },
            "services": services,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Failed to get Windows stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve Windows stats")

# Stats Endpoints (gleich wie vorher)
@app.get("/api/insights/stats")
async def get_insights_stats():
    return {
        "total_insights_today": 15,
        "high_confidence_insights": 12,
        "actionable_insights": 13,
        "categories": {
            "system_optimization": 6,
            "performance": 4,
            "optimization": 3,
            "resource_monitoring": 2
        },
        "avg_confidence": 0.87,
        "last_generated": datetime.utcnow() - timedelta(minutes=12),
        "platform": "Windows Server 2022"
    }

@app.get("/api/patterns/stats")
async def get_patterns_stats():
    return {
        "total_patterns": 28,
        "new_patterns_today": 4,
        "high_confidence_patterns": 19,
        "pattern_types": {
            "success_patterns": 15,
            "performance_patterns": 8,
            "dangerous_conditions": 3,
            "strategy_synergies": 2
        },
        "avg_success_rate": 0.73,
        "most_profitable_pattern": "Windows IIS High-Performance Pattern",
        "platform_optimized": True
    }

if __name__ == "__main__":
    # Windows Production Server Configuration
    host = os.getenv("INTELLIGENCE_API_HOST", "0.0.0.0")
    port = int(os.getenv("INTELLIGENCE_API_PORT", "8001"))
    
    logger.info(f"🚀 Starting Intelligence API on Windows Server - {host}:{port}")
    
    # Windows Sound Notification beim Start
    try:
        winsound.MessageBeep(winsound.MB_ICONINFORMATION)
    except:
        pass
    
    # Uvicorn Server mit Windows-Optimierungen
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
        reload=False,
        workers=1,  # Auf Windows oft besser mit 1 Worker
        loop="asyncio",
        ws_ping_interval=20,
        ws_ping_timeout=10
    )