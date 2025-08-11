#!/usr/bin/env python3
"""
📊 Dashboard Enhancement Module
Live Bot-Health Indicators, Strategy Orchestra, Performance Tracking & Notifications
"""

import json
import logging
import asyncio
import os
import psutil
import subprocess
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
    
    # Bot Process Status
    trading_bot_running: bool = False        # Is main.py running?
    trading_bot_pid: Optional[int] = None    # Process ID if running
    trading_bot_uptime: float = 0.0          # Bot uptime in hours
    intelligence_api_running: bool = False   # Is Intelligence API running?
    
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

class BotProcessMonitor:
    """
    🤖 Trading Bot Process Monitor
    
    Detects if the main trading bot (main.py) is actually running
    """
    
    def __init__(self):
        self.known_bot_processes = []
        self.process_patterns = [
            'main.py',
            'python main.py',
            'python3 main.py',
            'trading_bot.py',
            'altcoin_trading_bot'
        ]
        
    def check_bot_processes(self) -> Dict[str, Any]:
        """Check if trading bot processes are running"""
        try:
            running_bots = []
            total_cpu = 0
            total_memory = 0
            oldest_start_time = None
            
            # Check all running processes
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'cpu_percent', 'memory_percent']):
                try:
                    pinfo = proc.info
                    cmdline = ' '.join(pinfo['cmdline']) if pinfo['cmdline'] else ''
                    
                    # Check if this looks like our trading bot
                    is_trading_bot = any(pattern in cmdline.lower() or pattern in pinfo['name'].lower() 
                                       for pattern in self.process_patterns)
                    
                    # Exclude the Intelligence API itself
                    if 'run_intelligence_api.py' in cmdline or 'intelligence_api' in cmdline:
                        continue
                        
                    if is_trading_bot:
                        bot_info = {
                            'pid': pinfo['pid'],
                            'name': pinfo['name'],
                            'cmdline': cmdline,
                            'start_time': datetime.fromtimestamp(pinfo['create_time']),
                            'cpu_percent': pinfo.get('cpu_percent', 0),
                            'memory_percent': pinfo.get('memory_percent', 0)
                        }
                        
                        running_bots.append(bot_info)
                        total_cpu += bot_info['cpu_percent']
                        total_memory += bot_info['memory_percent']
                        
                        if oldest_start_time is None or bot_info['start_time'] < oldest_start_time:
                            oldest_start_time = bot_info['start_time']
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            # Calculate uptime
            uptime_hours = 0
            if oldest_start_time:
                uptime_hours = (datetime.now() - oldest_start_time).total_seconds() / 3600
            
            return {
                'trading_bot_running': len(running_bots) > 0,
                'bot_count': len(running_bots),
                'processes': running_bots,
                'total_cpu_usage': total_cpu,
                'total_memory_usage': total_memory,
                'uptime_hours': uptime_hours,
                'main_pid': running_bots[0]['pid'] if running_bots else None
            }
            
        except Exception as e:
            logger.error(f"Error checking bot processes: {e}")
            return {
                'trading_bot_running': False,
                'bot_count': 0,
                'processes': [],
                'total_cpu_usage': 0,
                'total_memory_usage': 0,
                'uptime_hours': 0,
                'main_pid': None,
                'error': str(e)
            }
    
    def start_trading_bot(self, bot_script: str = "main.py") -> Dict[str, Any]:
        """Start the trading bot process"""
        try:
            # Check if bot is already running
            current_status = self.check_bot_processes()
            if current_status['trading_bot_running']:
                return {
                    'success': False,
                    'message': f'Trading bot is already running (PID: {current_status["main_pid"]})',
                    'error': 'BOT_ALREADY_RUNNING'
                }
            
            # Start the bot process
            import subprocess
            import sys
            
            # Try different possible bot locations - extensive search
            possible_locations = [
                bot_script,
                f"./{bot_script}",
                f"../{bot_script}",
                "main.py",
                "./main.py",
                "../main.py",
                "trading_bot.py",
                "./trading_bot.py",
                "core/trading_bot.py",
                "./core/trading_bot.py",
                "main_fixed.py",
                "./main_fixed.py"
            ]
            
            bot_path = None
            for location in possible_locations:
                if os.path.exists(location):
                    bot_path = location
                    break
            
            if not bot_path:
                # List available Python files for debugging
                try:
                    available_files = [f for f in os.listdir('.') if f.endswith('.py')]
                    return {
                        'success': False,
                        'message': f'Could not find {bot_script}. Available .py files: {available_files[:10]}',
                        'error': 'BOT_SCRIPT_NOT_FOUND',
                        'available_files': available_files[:10],
                        'current_dir': os.getcwd()
                    }
                except:
                    return {
                        'success': False,
                        'message': f'Could not find {bot_script} to start',
                        'error': 'BOT_SCRIPT_NOT_FOUND'
                    }
            
            # Start bot process in background
            process = subprocess.Popen([
                sys.executable, bot_path
            ], 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            start_new_session=True  # Detach from parent
            )
            
            # Wait briefly to check if it started successfully
            import time
            time.sleep(2)
            
            if process.poll() is None:  # Process is still running
                return {
                    'success': True,
                    'message': f'Trading bot started successfully (PID: {process.pid})',
                    'pid': process.pid,
                    'script_path': bot_path
                }
            else:
                # Process died immediately, get error
                stdout, stderr = process.communicate()
                return {
                    'success': False,
                    'message': f'Trading bot failed to start: {stderr.decode()[:200]}',
                    'error': 'BOT_START_FAILED',
                    'stderr': stderr.decode()
                }
                
        except Exception as e:
            logger.error(f"Error starting trading bot: {e}")
            return {
                'success': False,
                'message': f'Failed to start trading bot: {str(e)}',
                'error': 'EXCEPTION'
            }
    
    def stop_trading_bot(self, pid: int = None) -> Dict[str, Any]:
        """Stop the trading bot process"""
        try:
            current_status = self.check_bot_processes()
            
            if not current_status['trading_bot_running']:
                return {
                    'success': False,
                    'message': 'No trading bot processes found to stop',
                    'error': 'BOT_NOT_RUNNING'
                }
            
            stopped_processes = []
            failed_processes = []
            
            # Stop all bot processes or specific PID
            for process_info in current_status['processes']:
                try:
                    if pid and process_info['pid'] != pid:
                        continue
                        
                    process = psutil.Process(process_info['pid'])
                    process_name = process_info['name']
                    process_pid = process_info['pid']
                    
                    # Try graceful shutdown first
                    process.terminate()
                    
                    # Wait for graceful shutdown
                    import time
                    time.sleep(3)
                    
                    if process.is_running():
                        # Force kill if still running
                        process.kill()
                        time.sleep(1)
                    
                    if not process.is_running():
                        stopped_processes.append({
                            'pid': process_pid,
                            'name': process_name
                        })
                    else:
                        failed_processes.append({
                            'pid': process_pid,
                            'name': process_name,
                            'error': 'Still running after kill'
                        })
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    # Process already stopped or no access
                    stopped_processes.append({
                        'pid': process_info['pid'],
                        'name': process_info['name'],
                        'note': 'Already stopped or no access'
                    })
                except Exception as e:
                    failed_processes.append({
                        'pid': process_info['pid'],
                        'name': process_info['name'],
                        'error': str(e)
                    })
            
            if stopped_processes and not failed_processes:
                return {
                    'success': True,
                    'message': f'Successfully stopped {len(stopped_processes)} trading bot process(es)',
                    'stopped_processes': stopped_processes
                }
            elif stopped_processes and failed_processes:
                return {
                    'success': True,
                    'message': f'Stopped {len(stopped_processes)} processes, {len(failed_processes)} failed',
                    'stopped_processes': stopped_processes,
                    'failed_processes': failed_processes
                }
            else:
                return {
                    'success': False,
                    'message': f'Failed to stop {len(failed_processes)} trading bot process(es)',
                    'error': 'STOP_FAILED',
                    'failed_processes': failed_processes
                }
                
        except Exception as e:
            logger.error(f"Error stopping trading bot: {e}")
            return {
                'success': False,
                'message': f'Failed to stop trading bot: {str(e)}',
                'error': 'EXCEPTION'
            }
    
    def restart_trading_bot(self, bot_script: str = "main.py") -> Dict[str, Any]:
        """Restart the trading bot (stop + start)"""
        # Stop first
        stop_result = self.stop_trading_bot()
        
        if not stop_result['success'] and stop_result['error'] != 'BOT_NOT_RUNNING':
            return {
                'success': False,
                'message': f'Failed to stop bot before restart: {stop_result["message"]}',
                'error': 'RESTART_STOP_FAILED'
            }
        
        # Wait a moment
        import time
        time.sleep(2)
        
        # Start again
        start_result = self.start_trading_bot(bot_script)
        
        if start_result['success']:
            return {
                'success': True,
                'message': f'Trading bot restarted successfully (PID: {start_result["pid"]})',
                'pid': start_result['pid']
            }
        else:
            return {
                'success': False,
                'message': f'Bot stopped but failed to restart: {start_result["message"]}',
                'error': 'RESTART_START_FAILED'
            }
    
    def check_intelligence_api(self) -> bool:
        """Check if Intelligence API is running (this process)"""
        return True  # We're running if we can execute this code
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        bot_status = self.check_bot_processes()
        
        return {
            'trading_bot': bot_status,
            'intelligence_api': {
                'running': self.check_intelligence_api(),
                'pid': os.getpid(),
                'uptime_hours': (datetime.now() - datetime.fromtimestamp(psutil.Process().create_time())).total_seconds() / 3600
            },
            'system': {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent
            }
        }

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
        
        # Bot Process Monitoring
        self.process_monitor = BotProcessMonitor()
        self.last_bot_check = None
        
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
            # Check trading bot process status first
            system_status = self.process_monitor.get_system_status()
            trading_bot_status = system_status['trading_bot']
            
            # Calculate overall health score with bot status factor
            health_factors = {
                'connection_quality': metrics.get('connection_quality', 0) * 0.20,
                'api_response': self._calculate_response_score(metrics.get('api_response_time', 1000)) * 0.15,
                'error_rate': (100 - metrics.get('error_rate', 100)) * 0.15,
                'system_resources': self._calculate_resource_score(
                    system_status['system']['cpu_percent'], 
                    system_status['system']['memory_percent']
                ) * 0.15,
                'trading_performance': metrics.get('trading_success_rate', 0) * 0.15,
                'bot_process_status': (100 if trading_bot_status['trading_bot_running'] else 0) * 0.20  # Major factor!
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
            
            # Update bot health with process information
            self.bot_health = BotHealthMetrics(
                overall_health=overall_health,
                status=status,
                uptime_hours=metrics.get('uptime_hours', 0),
                connection_quality=metrics.get('connection_quality', 0),
                api_response_time=metrics.get('api_response_time', 0),
                error_rate=metrics.get('error_rate', 0),
                memory_usage=system_status['system']['memory_percent'],
                cpu_usage=system_status['system']['cpu_percent'],
                active_strategies=metrics.get('active_strategies', 0),
                successful_trades=metrics.get('successful_trades', 0),
                failed_trades=metrics.get('failed_trades', 0),
                total_pnl=metrics.get('total_pnl', 0),
                last_update=datetime.now(),
                
                # Bot Process Status
                trading_bot_running=trading_bot_status['trading_bot_running'],
                trading_bot_pid=trading_bot_status.get('main_pid'),
                trading_bot_uptime=trading_bot_status.get('uptime_hours', 0),
                intelligence_api_running=system_status['intelligence_api']['running']
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
            
            # Check for alerts including bot process alerts
            self._check_health_alerts(status, overall_health)
            self._check_bot_process_alerts(trading_bot_status)
            
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
    
    def _check_bot_process_alerts(self, bot_status: Dict[str, Any]):
        """Check for bot process-specific alerts"""
        
        # Trading Bot offline alert
        if not bot_status['trading_bot_running']:
            self.add_notification(
                "🚨 TRADING BOT OFFLINE",
                "Main trading bot is not running! Start main.py to begin trading.",
                NotificationLevel.EMERGENCY,
                "system",
                action_required=True
            )
        
        # Multiple bot instances warning
        elif bot_status['bot_count'] > 1:
            self.add_notification(
                "⚠️ Multiple Trading Bots Detected",
                f"Found {bot_status['bot_count']} trading bot instances running. This may cause conflicts.",
                NotificationLevel.WARNING,
                "system"
            )
        
        # Bot high resource usage
        if bot_status['total_cpu_usage'] > 80:
            self.add_notification(
                "💻 High Bot CPU Usage",
                f"Trading bot using {bot_status['total_cpu_usage']:.1f}% CPU. Check for performance issues.",
                NotificationLevel.WARNING,
                "performance"
            )
        
        # Bot recently restarted
        if bot_status['uptime_hours'] > 0 and bot_status['uptime_hours'] < 0.5:  # Less than 30 minutes
            self.add_notification(
                "🔄 Trading Bot Recently Started",
                f"Trading bot has been running for {bot_status['uptime_hours']:.1f} hours.",
                NotificationLevel.INFO,
                "system"
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
        # Demo bot health (system stats will be real, trading metrics simulated)
        demo_health = {
            'connection_quality': 85,
            'api_response_time': 150,
            'error_rate': 2.3,
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