# utils/exchange_monitor.py
"""
Exchange Monitor with Health Check, Failover, and Performance Metrics
Monitors exchange connectivity, performance, and automatically handles failover
"""

import asyncio
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import time
import statistics
from threading import Lock
import json

# Try to import notifier
try:
    from utils.notifier import send_info, send_warning, send_error, send_critical
    NOTIFIER_AVAILABLE = True
except ImportError:
    NOTIFIER_AVAILABLE = False

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status of exchange"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    timestamp: datetime
    latency_ms: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class ExchangeHealth:
    """Exchange health information"""
    exchange_name: str
    status: HealthStatus
    last_check: datetime
    latency_ms: Optional[float] = None
    success_rate_24h: float = 0.0
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    uptime_percent_24h: float = 0.0
    
    # Performance metrics
    avg_latency_1h: float = 0.0
    avg_latency_24h: float = 0.0
    min_latency_24h: float = 0.0
    max_latency_24h: float = 0.0
    
    # Counters
    total_requests_24h: int = 0
    successful_requests_24h: int = 0
    failed_requests_24h: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'exchange_name': self.exchange_name,
            'status': self.status.value,
            'last_check': self.last_check.isoformat(),
            'latency_ms': self.latency_ms,
            'success_rate_24h': self.success_rate_24h,
            'consecutive_failures': self.consecutive_failures,
            'last_error': self.last_error,
            'uptime_percent_24h': self.uptime_percent_24h,
            'avg_latency_1h': self.avg_latency_1h,
            'avg_latency_24h': self.avg_latency_24h,
            'min_latency_24h': self.min_latency_24h,
            'max_latency_24h': self.max_latency_24h,
            'total_requests_24h': self.total_requests_24h,
            'successful_requests_24h': self.successful_requests_24h,
            'failed_requests_24h': self.failed_requests_24h
        }


@dataclass
class FailoverEvent:
    """Failover event record"""
    timestamp: datetime
    from_exchange: str
    to_exchange: str
    reason: str
    duration_seconds: Optional[float] = None
    success: bool = True


class ExchangeMonitor:
    """
    Advanced Exchange Monitor with health checking, performance metrics, and failover
    """
    
    def __init__(self, exchange_manager, config: Dict[str, Any]):
        self.exchange_manager = exchange_manager
        self.config = config
        
        # Configuration
        self.check_interval = config.get('health_check_interval_seconds', 60)
        self.max_consecutive_failures = config.get('max_consecutive_failures', 3)
        self.failure_timeout_minutes = config.get('failure_timeout_minutes', 5)
        self.auto_reconnect = config.get('auto_reconnect', True)
        self.reconnect_delay = config.get('reconnect_delay_seconds', 30)
        
        # Health check configuration
        self.health_check_symbols = config.get('health_check_symbols', ['BTC/USDT', 'ETH/USDT'])
        self.latency_threshold_ms = config.get('performance_thresholds', {}).get('max_latency_ms', 500)
        self.min_success_rate = config.get('performance_thresholds', {}).get('min_success_rate_percent', 95)
        
        # Failover configuration
        self.failover_enabled = config.get('failover_enabled', True)
        self.fallback_order = config.get('fallback_order', [])
        
        # State
        self.exchange_health: Dict[str, ExchangeHealth] = {}
        self.performance_history: Dict[str, List[PerformanceMetric]] = {}
        self.failover_history: List[FailoverEvent] = []
        self.active_primary = None
        
        # Threading
        self.monitoring_active = False
        self.monitor_task = None
        self.lock = Lock()
        
        # Callbacks
        self.health_change_callbacks: List[Callable] = []
        self.failover_callbacks: List[Callable] = []
        
        # Initialize
        self._initialize_health_tracking()
        
        logger.info("ExchangeMonitor initialized")
    
    def _initialize_health_tracking(self):
        """Initialize health tracking for all exchanges"""
        for exchange_name in self.exchange_manager.exchanges.keys():
            self.exchange_health[exchange_name] = ExchangeHealth(
                exchange_name=exchange_name,
                status=HealthStatus.OFFLINE,
                last_check=datetime.now()
            )
            self.performance_history[exchange_name] = []
    
    async def start_monitoring(self):
        """Start background monitoring"""
        if self.monitoring_active:
            logger.warning("Exchange monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        
        # Set initial primary exchange
        self.active_primary = self.exchange_manager.primary_exchange
        
        logger.info("Started exchange monitoring")
        
        if NOTIFIER_AVAILABLE:
            send_info("🔍 Exchange monitoring started")
    
    async def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped exchange monitoring")
        
        if NOTIFIER_AVAILABLE:
            send_info("⏹️ Exchange monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Perform health checks
                await self._perform_health_checks()
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Check for failover conditions
                await self._check_failover_conditions()
                
                # Cleanup old data
                self._cleanup_old_data()
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _perform_health_checks(self):
        """Perform health checks on all exchanges"""
        tasks = []
        
        for exchange_name, exchange in self.exchange_manager.exchanges.items():
            task = self._check_exchange_health(exchange_name, exchange)
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_exchange_health(self, exchange_name: str, exchange):
        """Check health of specific exchange"""
        try:
            start_time = time.time()
            
            # Try to fetch ticker for health check
            test_symbol = self.health_check_symbols[0]
            
            try:
                ticker = await exchange.fetch_ticker(test_symbol)
                latency = (time.time() - start_time) * 1000  # Convert to ms
                
                # Health check passed
                metric = PerformanceMetric(
                    timestamp=datetime.now(),
                    latency_ms=latency,
                    success=True
                )
                
                # Update health status
                with self.lock:
                    health = self.exchange_health[exchange_name]
                    health.last_check = datetime.now()
                    health.latency_ms = latency
                    health.consecutive_failures = 0
                    
                    # Determine status based on latency
                    if latency <= self.latency_threshold_ms:
                        new_status = HealthStatus.HEALTHY
                    elif latency <= self.latency_threshold_ms * 2:
                        new_status = HealthStatus.DEGRADED
                    else:
                        new_status = HealthStatus.UNHEALTHY
                    
                    # Check for status change
                    if health.status != new_status:
                        old_status = health.status
                        health.status = new_status
                        await self._notify_health_change(exchange_name, old_status, new_status)
                    
                    # Store performance metric
                    self.performance_history[exchange_name].append(metric)
                
                logger.debug(f"Health check passed for {exchange_name}: {latency:.0f}ms")
                
            except Exception as e:
                # Health check failed
                latency = (time.time() - start_time) * 1000
                
                metric = PerformanceMetric(
                    timestamp=datetime.now(),
                    latency_ms=latency,
                    success=False,
                    error_message=str(e)
                )
                
                # Update health status
                with self.lock:
                    health = self.exchange_health[exchange_name]
                    health.last_check = datetime.now()
                    health.consecutive_failures += 1
                    health.last_error = str(e)
                    
                    # Determine status based on failures
                    if health.consecutive_failures >= self.max_consecutive_failures:
                        new_status = HealthStatus.OFFLINE
                    elif health.consecutive_failures >= 1:
                        new_status = HealthStatus.UNHEALTHY
                    else:
                        new_status = HealthStatus.DEGRADED
                    
                    # Check for status change
                    if health.status != new_status:
                        old_status = health.status
                        health.status = new_status
                        await self._notify_health_change(exchange_name, old_status, new_status)
                    
                    # Store performance metric
                    self.performance_history[exchange_name].append(metric)
                
                logger.warning(f"Health check failed for {exchange_name}: {e}")
                
        except Exception as e:
            logger.error(f"Error checking health for {exchange_name}: {e}")
    
    async def _notify_health_change(self, exchange_name: str, old_status: HealthStatus, new_status: HealthStatus):
        """Notify about health status change"""
        logger.info(f"Exchange {exchange_name} status changed: {old_status.value} -> {new_status.value}")
        
        # Send notification
        if NOTIFIER_AVAILABLE:
            if new_status == HealthStatus.OFFLINE:
                send_critical(f"🚨 Exchange {exchange_name} is OFFLINE!")
            elif new_status == HealthStatus.UNHEALTHY:
                send_error(f"❌ Exchange {exchange_name} is UNHEALTHY")
            elif new_status == HealthStatus.DEGRADED:
                send_warning(f"⚠️ Exchange {exchange_name} performance DEGRADED")
            elif new_status == HealthStatus.HEALTHY and old_status in [HealthStatus.OFFLINE, HealthStatus.UNHEALTHY]:
                send_info(f"✅ Exchange {exchange_name} is back online and HEALTHY")
        
        # Call registered callbacks
        for callback in self.health_change_callbacks:
            try:
                await callback(exchange_name, old_status, new_status)
            except Exception as e:
                logger.error(f"Error in health change callback: {e}")
    
    def _update_performance_metrics(self):
        """Update performance metrics for all exchanges"""
        with self.lock:
            cutoff_24h = datetime.now() - timedelta(hours=24)
            cutoff_1h = datetime.now() - timedelta(hours=1)
            
            for exchange_name, health in self.exchange_health.items():
                metrics = self.performance_history.get(exchange_name, [])
                
                # Filter metrics by time
                metrics_24h = [m for m in metrics if m.timestamp >= cutoff_24h]
                metrics_1h = [m for m in metrics if m.timestamp >= cutoff_1h]
                
                if metrics_24h:
                    # 24-hour metrics
                    successful_24h = [m for m in metrics_24h if m.success]
                    latencies_24h = [m.latency_ms for m in successful_24h]
                    
                    health.total_requests_24h = len(metrics_24h)
                    health.successful_requests_24h = len(successful_24h)
                    health.failed_requests_24h = len(metrics_24h) - len(successful_24h)
                    health.success_rate_24h = len(successful_24h) / len(metrics_24h) * 100
                    
                    if latencies_24h:
                        health.avg_latency_24h = statistics.mean(latencies_24h)
                        health.min_latency_24h = min(latencies_24h)
                        health.max_latency_24h = max(latencies_24h)
                    
                    # Calculate uptime
                    health.uptime_percent_24h = health.success_rate_24h
                
                if metrics_1h:
                    # 1-hour metrics
                    successful_1h = [m for m in metrics_1h if m.success]
                    latencies_1h = [m.latency_ms for m in successful_1h]
                    
                    if latencies_1h:
                        health.avg_latency_1h = statistics.mean(latencies_1h)
    
    async def _check_failover_conditions(self):
        """Check if failover is needed"""
        if not self.failover_enabled or not self.active_primary:
            return
        
        with self.lock:
            primary_health = self.exchange_health.get(self.active_primary)
            
            if not primary_health:
                return
            
            # Check if primary exchange needs failover
            needs_failover = (
                primary_health.status == HealthStatus.OFFLINE or
                (primary_health.status == HealthStatus.UNHEALTHY and 
                 primary_health.consecutive_failures >= self.max_consecutive_failures) or
                (primary_health.success_rate_24h < self.min_success_rate and 
                 primary_health.total_requests_24h > 10)
            )
            
            if needs_failover:
                await self._perform_failover()
    
    async def _perform_failover(self):
        """Perform failover to backup exchange"""
        try:
            # Find best fallback exchange
            fallback_exchange = self._find_best_fallback()
            
            if not fallback_exchange:
                logger.error("No healthy fallback exchange available")
                if NOTIFIER_AVAILABLE:
                    send_critical("🚨 CRITICAL: No healthy exchanges available for failover!")
                return
            
            old_primary = self.active_primary
            
            # Record failover event
            failover_event = FailoverEvent(
                timestamp=datetime.now(),
                from_exchange=old_primary,
                to_exchange=fallback_exchange,
                reason=f"Primary exchange {old_primary} health issues"
            )
            
            # Perform failover
            self.active_primary = fallback_exchange
            self.exchange_manager.primary_exchange = fallback_exchange
            
            # Update event
            failover_event.success = True
            self.failover_history.append(failover_event)
            
            logger.critical(f"FAILOVER: Switched from {old_primary} to {fallback_exchange}")
            
            if NOTIFIER_AVAILABLE:
                send_critical(f"🔄 FAILOVER EXECUTED!\n"
                            f"From: {old_primary}\n"
                            f"To: {fallback_exchange}\n"
                            f"Reason: Health issues")
            
            # Call failover callbacks
            for callback in self.failover_callbacks:
                try:
                    await callback(old_primary, fallback_exchange, failover_event.reason)
                except Exception as e:
                    logger.error(f"Error in failover callback: {e}")
            
            # Schedule reconnection attempt for failed exchange
            if self.auto_reconnect:
                asyncio.create_task(self._attempt_reconnect(old_primary))
            
        except Exception as e:
            logger.error(f"Error performing failover: {e}")
            if NOTIFIER_AVAILABLE:
                send_critical(f"🚨 FAILOVER FAILED: {e}")
    
    def _find_best_fallback(self) -> Optional[str]:
        """Find best available fallback exchange"""
        # Check fallback order first
        for exchange_name in self.fallback_order:
            if exchange_name in self.exchange_health:
                health = self.exchange_health[exchange_name]
                if health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
                    return exchange_name
        
        # Find any healthy exchange
        for exchange_name, health in self.exchange_health.items():
            if (exchange_name != self.active_primary and 
                health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]):
                return exchange_name
        
        return None
    
    async def _attempt_reconnect(self, exchange_name: str):
        """Attempt to reconnect failed exchange"""
        try:
            await asyncio.sleep(self.reconnect_delay)
            
            logger.info(f"Attempting to reconnect {exchange_name}")
            
            exchange = self.exchange_manager.get_exchange(exchange_name)
            if exchange:
                # Try to reconnect
                success = await exchange.connect()
                
                if success:
                    logger.info(f"Successfully reconnected {exchange_name}")
                    if NOTIFIER_AVAILABLE:
                        send_info(f"✅ {exchange_name} reconnected successfully")
                    
                    # Reset health status
                    with self.lock:
                        health = self.exchange_health[exchange_name]
                        health.consecutive_failures = 0
                        health.status = HealthStatus.HEALTHY
                else:
                    logger.warning(f"Failed to reconnect {exchange_name}")
            
        except Exception as e:
            logger.error(f"Error attempting to reconnect {exchange_name}: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old performance data"""
        cutoff_time = datetime.now() - timedelta(days=7)  # Keep 7 days of data
        
        with self.lock:
            for exchange_name in self.performance_history:
                self.performance_history[exchange_name] = [
                    metric for metric in self.performance_history[exchange_name]
                    if metric.timestamp >= cutoff_time
                ]
            
            # Clean up old failover events
            self.failover_history = [
                event for event in self.failover_history
                if event.timestamp >= cutoff_time
            ]
    
    def get_exchange_health(self, exchange_name: str) -> Optional[ExchangeHealth]:
        """Get health status for specific exchange"""
        with self.lock:
            return self.exchange_health.get(exchange_name)
    
    def get_all_health_status(self) -> Dict[str, ExchangeHealth]:
        """Get health status for all exchanges"""
        with self.lock:
            return self.exchange_health.copy()
    
    def get_healthy_exchanges(self) -> List[str]:
        """Get list of healthy exchanges"""
        with self.lock:
            return [
                name for name, health in self.exchange_health.items()
                if health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
            ]
    
    def get_primary_exchange(self) -> Optional[str]:
        """Get current primary exchange"""
        return self.active_primary
    
    def force_failover(self, target_exchange: str = None) -> bool:
        """Force failover to specific exchange or best available"""
        try:
            if target_exchange:
                if target_exchange in self.exchange_health:
                    health = self.exchange_health[target_exchange]
                    if health.status != HealthStatus.OFFLINE:
                        old_primary = self.active_primary
                        self.active_primary = target_exchange
                        self.exchange_manager.primary_exchange = target_exchange
                        
                        # Record event
                        failover_event = FailoverEvent(
                            timestamp=datetime.now(),
                            from_exchange=old_primary,
                            to_exchange=target_exchange,
                            reason="Manual failover",
                            success=True
                        )
                        self.failover_history.append(failover_event)
                        
                        logger.info(f"Manual failover: {old_primary} -> {target_exchange}")
                        return True
            else:
                # Find best available
                best_exchange = self._find_best_fallback()
                if best_exchange:
                    return self.force_failover(best_exchange)
            
            return False
            
        except Exception as e:
            logger.error(f"Error in forced failover: {e}")
            return False
    
    def add_health_change_callback(self, callback: Callable):
        """Add callback for health status changes"""
        self.health_change_callbacks.append(callback)
    
    def add_failover_callback(self, callback: Callable):
        """Add callback for failover events"""
        self.failover_callbacks.append(callback)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        with self.lock:
            report = {
                'timestamp': datetime.now().isoformat(),
                'active_primary': self.active_primary,
                'monitoring_active': self.monitoring_active,
                'exchanges': {},
                'summary': {
                    'total_exchanges': len(self.exchange_health),
                    'healthy_exchanges': len([h for h in self.exchange_health.values() 
                                            if h.status == HealthStatus.HEALTHY]),
                    'degraded_exchanges': len([h for h in self.exchange_health.values() 
                                             if h.status == HealthStatus.DEGRADED]),
                    'unhealthy_exchanges': len([h for h in self.exchange_health.values() 
                                              if h.status == HealthStatus.UNHEALTHY]),
                    'offline_exchanges': len([h for h in self.exchange_health.values() 
                                            if h.status == HealthStatus.OFFLINE])
                },
                'failover_history': [event.__dict__ for event in self.failover_history[-10:]]  # Last 10
            }
            
            # Add detailed exchange data
            for name, health in self.exchange_health.items():
                report['exchanges'][name] = health.to_dict()
            
            return report
    
    def export_metrics(self, filepath: str = None) -> str:
        """Export performance metrics to file"""
        try:
            if filepath is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filepath = f"exchange_metrics_{timestamp}.json"
            
            report = self.get_performance_report()
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Exchange metrics exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return ""


# Factory function
def create_exchange_monitor(exchange_manager, config: Dict[str, Any]) -> ExchangeMonitor:
    """Create and return ExchangeMonitor instance"""
    return ExchangeMonitor(exchange_manager, config)