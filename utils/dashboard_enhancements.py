#!/usr/bin/env python3
"""
📊 Dashboard Enhancement Module
Live Bot-Health Indicators, Strategy Orchestra, Performance Tracking & Notifications
"""

import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import time

logger = logging.getLogger(__name__)

class BotHealthStatus(Enum):
    """Bot health status levels"""
    EXCELLENT = "excellent"      # 95-100% performance
    GOOD = "good"               # 80-95% performance  
    WARNING = "warning"         # 60-80% performance
    CRITICAL = "critical"       # 40-60% performance
    OFFLINE = "offline"         # <40% or no connection

class NotificationLevel(Enum):
    """Notification severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class BotHealthMetrics:
    """Comprehensive bot health metrics"""
    overall_health: float = 0.0              # 0-100 health score
    status: BotHealthStatus = BotHealthStatus.OFFLINE
    uptime_hours: float = 0.0
    connection_quality: float = 0.0          # 0-100 connection score
    api_response_time: float = 0.0           # Average response time in ms
    error_rate: float = 0.0                  # Error rate percentage
    memory_usage: float = 0.0                # Memory usage percentage
    cpu_usage: float = 0.0                   # CPU usage percentage
    active_strategies: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    total_pnl: float = 0.0
    last_update: datetime = None
    
    def __post_init__(self):
        if self.last_update is None:
            self.last_update = datetime.now()

@dataclass
class StrategyOrchestraItem:
    """Individual strategy in the orchestra display"""
    name: str
    status: str                    # active, paused, error, idle
    performance_score: float       # 0-100 performance rating
    current_signal: str           # BUY, SELL, HOLD
    confidence: float             # 0-1 confidence level
    trades_today: int
    pnl_today: float
    risk_level: str               # HIGH, MEDIUM, LOW
    execution_time_ms: float      # Average execution time
    last_action: str
    last_action_time: datetime
    
    def __post_init__(self):
        if isinstance(self.last_action_time, str):
            self.last_action_time = datetime.fromisoformat(self.last_action_time)

@dataclass
class DashboardNotification:
    """Dashboard notification system"""
    id: str
    title: str
    message: str
    level: NotificationLevel
    timestamp: datetime
    category: str                 # system, trading, security, performance
    action_required: bool = False
    auto_dismiss: bool = True
    dismiss_after: int = 10       # Seconds
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)

class DashboardEnhancementManager:
    """
    📊 Enhanced Dashboard Management System
    
    Features:
    - Real-time bot health monitoring with pulse animations
    - Strategy orchestra visualization 
    - Performance tracking and alerts
    - Instant failure notifications
    - Connection stability monitoring
    """
    
    def __init__(self):
        self.bot_health = BotHealthMetrics()
        self.strategy_orchestra: List[StrategyOrchestraItem] = []
        self.notifications: List[DashboardNotification] = []
        self.performance_history: List[Dict] = []
        self.connection_history: List[Dict] = []
        
        # Health monitoring configuration
        self.health_thresholds = {
            'excellent': 95,
            'good': 80,
            'warning': 60,
            'critical': 40
        }
        
        # Performance tracking
        self.performance_metrics = {
            'response_times': [],
            'error_counts': [],
            'success_rates': [],
            'connection_tests': []
        }
        
        logger.info("📊 Dashboard Enhancement Manager initialized")
    
    def update_bot_health(self, metrics: Dict) -> BotHealthMetrics:
        """Update comprehensive bot health metrics"""
        try:
            # Calculate overall health score
            health_factors = {
                'connection_quality': metrics.get('connection_quality', 0) * 0.25,
                'api_response': self._calculate_response_score(metrics.get('api_response_time', 1000)) * 0.20,
                'error_rate': (100 - metrics.get('error_rate', 100)) * 0.20,
                'system_resources': self._calculate_resource_score(
                    metrics.get('cpu_usage', 100), 
                    metrics.get('memory_usage', 100)
                ) * 0.15,
                'trading_performance': metrics.get('trading_success_rate', 0) * 0.20
            }
            
            overall_health = sum(health_factors.values())
            
            # Determine health status
            if overall_health >= self.health_thresholds['excellent']:
                status = BotHealthStatus.EXCELLENT
            elif overall_health >= self.health_thresholds['good']:
                status = BotHealthStatus.GOOD
            elif overall_health >= self.health_thresholds['warning']:
                status = BotHealthStatus.WARNING
            elif overall_health >= self.health_thresholds['critical']:
                status = BotHealthStatus.CRITICAL
            else:
                status = BotHealthStatus.OFFLINE
            
            # Update bot health
            self.bot_health = BotHealthMetrics(
                overall_health=overall_health,
                status=status,
                uptime_hours=metrics.get('uptime_hours', 0),
                connection_quality=metrics.get('connection_quality', 0),
                api_response_time=metrics.get('api_response_time', 0),
                error_rate=metrics.get('error_rate', 0),
                memory_usage=metrics.get('memory_usage', 0),
                cpu_usage=metrics.get('cpu_usage', 0),
                active_strategies=metrics.get('active_strategies', 0),
                successful_trades=metrics.get('successful_trades', 0),
                failed_trades=metrics.get('failed_trades', 0),
                total_pnl=metrics.get('total_pnl', 0),
                last_update=datetime.now()
            )
            
            # Add to history
            self.performance_history.append({
                'timestamp': datetime.now().isoformat(),
                'health_score': overall_health,
                'status': status.value,
                'response_time': metrics.get('api_response_time', 0)
            })
            
            # Keep only last 100 entries
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
            
            # Check for alerts
            self._check_health_alerts(status, overall_health)
            
            return self.bot_health
            
        except Exception as e:
            logger.error(f"Error updating bot health: {e}")
            self.bot_health.status = BotHealthStatus.OFFLINE
            return self.bot_health
    
    def _calculate_response_score(self, response_time_ms: float) -> float:
        """Calculate response time score (0-100)"""
        if response_time_ms <= 100:
            return 100
        elif response_time_ms <= 500:
            return 100 - ((response_time_ms - 100) / 400 * 30)  # 100-70
        elif response_time_ms <= 2000:
            return 70 - ((response_time_ms - 500) / 1500 * 50)   # 70-20
        else:
            return max(0, 20 - ((response_time_ms - 2000) / 3000 * 20))  # 20-0
    
    def _calculate_resource_score(self, cpu_usage: float, memory_usage: float) -> float:
        """Calculate system resource score (0-100)"""
        avg_usage = (cpu_usage + memory_usage) / 2
        if avg_usage <= 50:
            return 100
        elif avg_usage <= 80:
            return 100 - ((avg_usage - 50) / 30 * 50)  # 100-50
        else:
            return max(0, 50 - ((avg_usage - 80) / 20 * 50))  # 50-0
    
    def update_strategy_orchestra(self, strategies: List[Dict]) -> List[StrategyOrchestraItem]:
        """Update strategy orchestra display"""
        try:
            orchestra_items = []
            
            for strategy in strategies:
                item = StrategyOrchestraItem(
                    name=strategy.get('name', 'Unknown'),
                    status=strategy.get('status', 'idle'),
                    performance_score=strategy.get('performance_score', 0),
                    current_signal=strategy.get('current_signal', 'HOLD'),
                    confidence=strategy.get('confidence', 0),
                    trades_today=strategy.get('trades_today', 0),
                    pnl_today=strategy.get('pnl_today', 0),
                    risk_level=strategy.get('risk_level', 'MEDIUM'),
                    execution_time_ms=strategy.get('execution_time_ms', 0),
                    last_action=strategy.get('last_action', 'No action'),
                    last_action_time=datetime.fromisoformat(
                        strategy.get('last_action_time', datetime.now().isoformat())
                    )
                )
                orchestra_items.append(item)
            
            self.strategy_orchestra = sorted(
                orchestra_items, 
                key=lambda x: (x.status == 'active', x.performance_score),
                reverse=True
            )
            
            return self.strategy_orchestra
            
        except Exception as e:
            logger.error(f"Error updating strategy orchestra: {e}")
            return self.strategy_orchestra
    
    def add_notification(self, title: str, message: str, level: NotificationLevel, 
                        category: str = "system", action_required: bool = False) -> str:
        """Add new dashboard notification"""
        notification_id = f"notif_{int(time.time() * 1000)}"
        
        notification = DashboardNotification(
            id=notification_id,
            title=title,
            message=message,
            level=level,
            timestamp=datetime.now(),
            category=category,
            action_required=action_required,
            auto_dismiss=level != NotificationLevel.EMERGENCY,
            dismiss_after=30 if level == NotificationLevel.CRITICAL else 10
        )
        
        self.notifications.insert(0, notification)  # Add to beginning
        
        # Keep only last 50 notifications
        if len(self.notifications) > 50:
            self.notifications = self.notifications[:50]
        
        logger.info(f"📢 Dashboard notification added: {title} ({level.value})")
        return notification_id
    
    def _check_health_alerts(self, status: BotHealthStatus, health_score: float):
        """Check for health-related alerts"""
        
        # Critical health alert
        if status == BotHealthStatus.CRITICAL:
            self.add_notification(
                "🚨 CRITICAL BOT HEALTH",
                f"Bot health critically low: {health_score:.1f}%. Immediate attention required!",
                NotificationLevel.CRITICAL,
                "system",
                action_required=True
            )
        
        # Offline alert
        elif status == BotHealthStatus.OFFLINE:
            self.add_notification(
                "🔴 BOT OFFLINE",
                "Trading bot appears to be offline or unresponsive!",
                NotificationLevel.EMERGENCY,
                "system", 
                action_required=True
            )
        
        # Warning health alert
        elif status == BotHealthStatus.WARNING:
            self.add_notification(
                "⚠️ Bot Performance Warning",
                f"Bot health at {health_score:.1f}%. Monitoring recommended.",
                NotificationLevel.WARNING,
                "performance"
            )
        
        # High error rate alert
        if self.bot_health.error_rate > 10:
            self.add_notification(
                "📈 High Error Rate Detected",
                f"Error rate: {self.bot_health.error_rate:.1f}%. Check system logs.",
                NotificationLevel.WARNING,
                "trading"
            )
        
        # Slow response time alert  
        if self.bot_health.api_response_time > 2000:
            self.add_notification(
                "🐌 Slow API Response",
                f"API response time: {self.bot_health.api_response_time:.0f}ms",
                NotificationLevel.WARNING,
                "performance"
            )
    
    def get_dashboard_data(self) -> Dict:
        """Get comprehensive dashboard data"""
        return {
            'bot_health': asdict(self.bot_health),
            'strategy_orchestra': [asdict(item) for item in self.strategy_orchestra],
            'notifications': [asdict(notif) for notif in self.notifications[:10]], # Last 10
            'performance_history': self.performance_history[-20:],  # Last 20 points
            'summary_stats': {
                'total_active_strategies': len([s for s in self.strategy_orchestra if s.status == 'active']),
                'total_trades_today': sum(s.trades_today for s in self.strategy_orchestra),
                'total_pnl_today': sum(s.pnl_today for s in self.strategy_orchestra),
                'avg_confidence': sum(s.confidence for s in self.strategy_orchestra) / max(1, len(self.strategy_orchestra)),
                'health_trend': self._calculate_health_trend()
            }
        }
    
    def _calculate_health_trend(self) -> str:
        """Calculate health trend direction"""
        if len(self.performance_history) < 2:
            return "stable"
        
        recent_scores = [h['health_score'] for h in self.performance_history[-5:]]
        if len(recent_scores) < 2:
            return "stable"
        
        trend = recent_scores[-1] - recent_scores[0]
        if trend > 5:
            return "improving"
        elif trend < -5:
            return "declining"
        else:
            return "stable"
    
    def dismiss_notification(self, notification_id: str) -> bool:
        """Dismiss a notification"""
        for i, notif in enumerate(self.notifications):
            if notif.id == notification_id:
                self.notifications.pop(i)
                return True
        return False
    
    def get_bot_health_json(self) -> str:
        """Get bot health as JSON string"""
        return json.dumps(asdict(self.bot_health), default=str)
    
    def get_strategy_orchestra_json(self) -> str:
        """Get strategy orchestra as JSON string"""
        return json.dumps([asdict(item) for item in self.strategy_orchestra], default=str)
    
    def simulate_demo_data(self):
        """Generate demo data for testing"""
        # Demo bot health
        demo_health = {
            'connection_quality': 85,
            'api_response_time': 150,
            'error_rate': 2.3,
            'cpu_usage': 45,
            'memory_usage': 60,
            'trading_success_rate': 78,
            'uptime_hours': 24.5,
            'active_strategies': 8,
            'successful_trades': 145,
            'failed_trades': 12,
            'total_pnl': 2847.93
        }
        self.update_bot_health(demo_health)
        
        # Demo strategy orchestra
        demo_strategies = [
            {
                'name': 'Momentum V2.1',
                'status': 'active',
                'performance_score': 87,
                'current_signal': 'BUY',
                'confidence': 0.92,
                'trades_today': 23,
                'pnl_today': 1247.33,
                'risk_level': 'MEDIUM',
                'execution_time_ms': 145,
                'last_action': 'Opened BTC position',
                'last_action_time': datetime.now().isoformat()
            },
            {
                'name': 'Mean Reversion',
                'status': 'active', 
                'performance_score': 76,
                'current_signal': 'HOLD',
                'confidence': 0.65,
                'trades_today': 8,
                'pnl_today': 433.21,
                'risk_level': 'LOW',
                'execution_time_ms': 89,
                'last_action': 'Position monitoring',
                'last_action_time': (datetime.now() - timedelta(minutes=5)).isoformat()
            },
            {
                'name': 'ML Alpha Strategy',
                'status': 'active',
                'performance_score': 94,
                'current_signal': 'SELL',
                'confidence': 0.88,
                'trades_today': 41,
                'pnl_today': 1167.39,
                'risk_level': 'HIGH',
                'execution_time_ms': 234,
                'last_action': 'Pattern detection: Descending triangle',
                'last_action_time': (datetime.now() - timedelta(minutes=2)).isoformat()
            },
            {
                'name': 'Arbitrage Pro',
                'status': 'paused',
                'performance_score': 45,
                'current_signal': 'HOLD',
                'confidence': 0.22,
                'trades_today': 2,
                'pnl_today': -23.11,
                'risk_level': 'LOW',
                'execution_time_ms': 567,
                'last_action': 'Insufficient spread detected',
                'last_action_time': (datetime.now() - timedelta(hours=2)).isoformat()
            }
        ]
        self.update_strategy_orchestra(demo_strategies)
        
        # Demo notifications
        self.add_notification(
            "🚀 Trading Opportunity",
            "High-probability breakout pattern detected on BTC/USDT",
            NotificationLevel.INFO,
            "trading"
        )

# Global instance
dashboard_manager = DashboardEnhancementManager()