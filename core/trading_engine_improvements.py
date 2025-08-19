"""
Trading Engine Improvements
===========================

Erweitert das Multi-Mode Trading System mit Production-Ready Features:
- Besseres Error Handling
- Performance Optimierungen
- Security Verbesserungen
- Resource Management
"""

import asyncio
import logging
import traceback
import time
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from functools import wraps
import uuid

logger = logging.getLogger(__name__)


class TradingEventBus:
    """Event Bus für Trading Events mit Publish/Subscribe Pattern"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._event_history: List[Dict] = []
        self._max_history = 1000
    
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to trading events"""
        self.subscribers[event_type].append(callback)
        logger.debug(f"Subscribed to event: {event_type}")
    
    async def publish(self, event_type: str, data: Dict[str, Any]):
        """Publish trading event to all subscribers"""
        try:
            event = {
                'id': str(uuid.uuid4()),
                'type': event_type,
                'data': data,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Store in history
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]
            
            # Publish to subscribers
            for callback in self.subscribers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Error in event callback for {event_type}: {e}")
            
            logger.debug(f"Published event: {event_type}")
            
        except Exception as e:
            logger.error(f"Failed to publish event {event_type}: {e}")
    
    def get_event_history(self, event_type: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get event history"""
        if event_type:
            events = [e for e in self._event_history if e['type'] == event_type]
        else:
            events = self._event_history
        
        return events[-limit:]


def retry_async_operation(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator für Retry-Logic bei async Operationen"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (ConnectionError, TimeoutError, Exception) as e:
                    last_exception = e
                    
                    if attempt == max_retries - 1:
                        logger.error(f"Operation {func.__name__} failed after {max_retries} attempts: {e}")
                        raise last_exception
                    
                    wait_time = delay * (backoff ** attempt)
                    logger.warning(f"Operation {func.__name__} failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
            
            raise last_exception
        
        return wrapper
    return decorator


class APIResponse:
    """Standardized API Response Format"""
    
    @staticmethod
    def success(data: Any = None, message: str = None, code: int = 200) -> tuple:
        """Create success response"""
        response = {
            'success': True,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        if data is not None:
            response['data'] = data
        if message:
            response['message'] = message
        
        return response, code
    
    @staticmethod
    def error(message: str, code: int = 400, details: Any = None) -> tuple:
        """Create error response"""
        response = {
            'success': False,
            'error': {
                'message': message,
                'code': code
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        if details:
            response['error']['details'] = details
        
        return response, code
    
    @staticmethod
    def validation_error(field: str, message: str) -> tuple:
        """Create validation error response"""
        return APIResponse.error(
            message=f"Validation error: {message}",
            code=422,
            details={'field': field, 'message': message}
        )


class ResourceManager:
    """Resource Management für Trading Engine"""
    
    def __init__(self, max_memory_mb: int = 500, max_history_days: int = 30):
        self.max_memory_mb = max_memory_mb
        self.max_history_days = max_history_days
        self.last_cleanup = datetime.now(timezone.utc)
        self._resource_stats = {
            'memory_usage': 0,
            'connections': 0,
            'cleanup_count': 0
        }
    
    async def cleanup_old_data(self, trade_history: List, performance_history: List) -> Dict[str, int]:
        """Cleanup old data to prevent memory leaks"""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.max_history_days)
            
            # Count items before cleanup
            trades_before = len(trade_history)
            perf_before = len(performance_history)
            
            # Filter old data
            trade_history[:] = [
                trade for trade in trade_history 
                if hasattr(trade, 'timestamp') and trade.timestamp > cutoff_date
            ]
            
            performance_history[:] = [
                perf for perf in performance_history
                if perf.get('timestamp', datetime.min.replace(tzinfo=timezone.utc)) > cutoff_date
            ]
            
            # Update stats
            trades_removed = trades_before - len(trade_history)
            perf_removed = perf_before - len(performance_history)
            self._resource_stats['cleanup_count'] += 1
            self.last_cleanup = datetime.now(timezone.utc)
            
            logger.info(f"Cleanup completed: {trades_removed} trades, {perf_removed} performance records removed")
            
            return {
                'trades_removed': trades_removed,
                'performance_removed': perf_removed,
                'timestamp': self.last_cleanup.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return {'error': str(e)}
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get current resource usage statistics"""
        return {
            **self._resource_stats,
            'last_cleanup': self.last_cleanup.isoformat(),
            'max_memory_mb': self.max_memory_mb,
            'max_history_days': self.max_history_days
        }


class HealthChecker:
    """Health Check System für Exchange Connections"""
    
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.health_status = {
            'last_check': None,
            'status': 'unknown',
            'consecutive_failures': 0,
            'uptime_start': datetime.now(timezone.utc)
        }
        self._running = False
        self._health_task: Optional[asyncio.Task] = None
    
    async def start_health_monitoring(self, exchange_client):
        """Start continuous health monitoring"""
        self._running = True
        self._health_task = asyncio.create_task(self._health_check_loop(exchange_client))
        logger.info("Health monitoring started")
    
    async def stop_health_monitoring(self):
        """Stop health monitoring"""
        self._running = False
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")
    
    async def _health_check_loop(self, exchange_client):
        """Main health check loop"""
        while self._running:
            try:
                await self._perform_health_check(exchange_client)
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(self.check_interval)
    
    @retry_async_operation(max_retries=2, delay=5.0)
    async def _perform_health_check(self, exchange_client):
        """Perform actual health check"""
        if not exchange_client:
            self.health_status.update({
                'status': 'no_client',
                'last_check': datetime.now(timezone.utc).isoformat()
            })
            return
        
        try:
            # Try simple API call
            await exchange_client.fetch_time()
            
            self.health_status.update({
                'status': 'healthy',
                'last_check': datetime.now(timezone.utc).isoformat(),
                'consecutive_failures': 0
            })
            
        except Exception as e:
            self.health_status['consecutive_failures'] += 1
            self.health_status.update({
                'status': 'unhealthy',
                'last_check': datetime.now(timezone.utc).isoformat(),
                'last_error': str(e)
            })
            
            logger.warning(f"Health check failed ({self.health_status['consecutive_failures']} consecutive): {e}")
            
            # If too many failures, consider reconnection
            if self.health_status['consecutive_failures'] >= 3:
                logger.error("Multiple consecutive health check failures - consider reconnection")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        uptime = datetime.now(timezone.utc) - self.health_status['uptime_start']
        return {
            **self.health_status,
            'uptime_seconds': int(uptime.total_seconds()),
            'is_running': self._running
        }


class SecurityAuditor:
    """Security Audit Logging für kritische Operationen"""
    
    def __init__(self, log_file: str = "security_audit.log"):
        self.log_file = log_file
        self.audit_logger = self._setup_audit_logger()
    
    def _setup_audit_logger(self):
        """Setup dedicated audit logger"""
        audit_logger = logging.getLogger('security_audit')
        audit_logger.setLevel(logging.INFO)
        
        # Create file handler if not exists
        handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        if not audit_logger.handlers:
            audit_logger.addHandler(handler)
        
        return audit_logger
    
    def log_mode_switch(self, user: str, old_mode: str, new_mode: str, success: bool):
        """Log trading mode switches"""
        self.audit_logger.info(f"MODE_SWITCH - User: {user}, {old_mode} -> {new_mode}, Success: {success}")
    
    def log_credential_change(self, user: str, exchange: str, action: str):
        """Log credential changes"""
        self.audit_logger.info(f"CREDENTIAL_CHANGE - User: {user}, Exchange: {exchange}, Action: {action}")
    
    def log_trade_execution(self, user: str, symbol: str, side: str, size: float, mode: str, success: bool):
        """Log trade executions"""
        self.audit_logger.info(f"TRADE_EXECUTION - User: {user}, {symbol} {side} {size}, Mode: {mode}, Success: {success}")
    
    def log_account_reset(self, user: str, mode: str):
        """Log account resets"""
        self.audit_logger.warning(f"ACCOUNT_RESET - User: {user}, Mode: {mode}")


class PerformanceMonitor:
    """Performance Monitoring für Trading Operations"""
    
    def __init__(self, max_metrics: int = 1000):
        self.metrics = []
        self.max_metrics = max_metrics
        self._operation_stats = defaultdict(list)
    
    def record_operation(self, operation: str, duration_ms: float, success: bool, details: Dict = None):
        """Record operation performance"""
        metric = {
            'operation': operation,
            'duration_ms': duration_ms,
            'success': success,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'details': details or {}
        }
        
        self.metrics.append(metric)
        self._operation_stats[operation].append(duration_ms)
        
        # Keep only latest metrics
        if len(self.metrics) > self.max_metrics:
            self.metrics = self.metrics[-self.max_metrics:]
    
    def get_performance_stats(self, operation: str = None) -> Dict[str, Any]:
        """Get performance statistics"""
        if operation and operation in self._operation_stats:
            durations = self._operation_stats[operation]
            return {
                'operation': operation,
                'count': len(durations),
                'avg_duration_ms': sum(durations) / len(durations) if durations else 0,
                'min_duration_ms': min(durations) if durations else 0,
                'max_duration_ms': max(durations) if durations else 0
            }
        else:
            # Overall stats
            all_ops = {}
            for op, durations in self._operation_stats.items():
                all_ops[op] = {
                    'count': len(durations),
                    'avg_duration_ms': sum(durations) / len(durations) if durations else 0
                }
            return all_ops
    
    def performance_decorator(self, operation_name: str):
        """Decorator to measure operation performance"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                success = False
                error = None
                
                try:
                    result = await func(*args, **kwargs)
                    success = True
                    return result
                except Exception as e:
                    error = str(e)
                    raise
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    self.record_operation(
                        operation_name, 
                        duration_ms, 
                        success,
                        {'error': error} if error else None
                    )
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                success = False
                error = None
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                    return result
                except Exception as e:
                    error = str(e)
                    raise
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    self.record_operation(
                        operation_name, 
                        duration_ms, 
                        success,
                        {'error': error} if error else None
                    )
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator


# Shared instances für das gesamte System
event_bus = TradingEventBus()
resource_manager = ResourceManager()
security_auditor = SecurityAuditor()
performance_monitor = PerformanceMonitor()


def get_request_id() -> str:
    """Generate unique request ID for tracing"""
    return str(uuid.uuid4())[:8]


def log_with_context(logger_instance, level: str, message: str, **context):
    """Log with additional context"""
    extra_context = {
        'request_id': context.pop('request_id', 'unknown'),
        'timestamp': datetime.now(timezone.utc).isoformat(),
        **context
    }
    
    getattr(logger_instance, level)(message, extra=extra_context)