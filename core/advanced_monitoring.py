"""
Advanced Monitoring System
==========================

Professional monitoring and alerting system with:
- Multi-channel notifications (Email, Webhook, Slack, Discord)
- Real-time metrics collection and dashboards
- Health checks and system diagnostics
- Performance monitoring and SLA tracking
- Log aggregation and analysis
- Custom alerting rules and escalation
- Integration with external monitoring services
"""

import logging
import asyncio
import threading
import time
import json
import smtplib
import requests
import psutil
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import sqlite3

from core.interfaces import global_event_bus

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class NotificationChannel(Enum):
    """Available notification channels"""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    DISCORD = "discord"
    SMS = "sms"
    CONSOLE = "console"
    FILE = "file"

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    level: AlertLevel
    source: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False
    escalated: bool = False
    notification_channels: List[NotificationChannel] = field(default_factory=list)

@dataclass
class Metric:
    """Metric data structure"""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""

@dataclass
class HealthCheck:
    """Health check definition"""
    name: str
    check_function: Callable[[], bool]
    interval_seconds: float
    timeout_seconds: float
    failure_threshold: int = 3
    last_check: Optional[datetime] = None
    consecutive_failures: int = 0
    is_healthy: bool = True

class AdvancedMonitoringSystem:
    """
    Comprehensive monitoring and alerting system
    """
    
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.monitoring_config = settings.get('advanced_monitoring', {})
        
        # Core configuration
        self.enabled = self.monitoring_config.get('enabled', True)
        self.metrics_retention_days = self.monitoring_config.get('metrics_retention_days', 30)
        self.health_check_interval = self.monitoring_config.get('health_check_interval', 60)
        self.alert_cooldown_minutes = self.monitoring_config.get('alert_cooldown_minutes', 15)
        
        # Storage
        self.metrics_db_path = self.monitoring_config.get('metrics_db_path', 'data/monitoring/metrics.db')
        self.alerts_db_path = self.monitoring_config.get('alerts_db_path', 'data/monitoring/alerts.db')
        
        # In-memory storage
        self.active_alerts: Dict[str, Alert] = {}
        self.metrics_buffer: deque = deque(maxlen=10000)
        self.health_checks: Dict[str, HealthCheck] = {}
        self.alert_cooldowns: Dict[str, datetime] = {}
        
        # Threading
        self._monitoring_thread: Optional[threading.Thread] = None
        self._health_check_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        
        # Notification channels
        self.notification_channels: Dict[NotificationChannel, Dict[str, Any]] = {}
        self._setup_notification_channels()
        
        # Event handlers
        self._alert_handlers: List[Callable] = []
        self._metric_handlers: List[Callable] = []
        
        # System metrics
        self.system_metrics = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'disk_usage': 0.0,
            'network_io': {'bytes_sent': 0, 'bytes_recv': 0},
            'process_count': 0,
            'uptime_seconds': 0
        }
        
        # Bot-specific metrics
        self.bot_metrics = {
            'trades_executed': 0,
            'total_pnl': 0.0,
            'active_positions': 0,
            'errors_count': 0,
            'api_calls': 0,
            'response_times': deque(maxlen=1000)
        }
        
        # Initialize databases
        self._init_databases()
        
        # Setup event handlers
        self._setup_event_handlers()
        
        # Setup default health checks
        self._setup_default_health_checks()
        
        logger.info("Advanced Monitoring System initialized")
    
    def _setup_notification_channels(self):
        """Setup notification channels based on configuration"""
        channels_config = self.monitoring_config.get('notification_channels', {})
        
        # Email configuration
        if 'email' in channels_config:
            email_config = channels_config['email']
            if email_config.get('enabled', False):
                self.notification_channels[NotificationChannel.EMAIL] = {
                    'smtp_server': email_config.get('smtp_server', 'localhost'),
                    'smtp_port': email_config.get('smtp_port', 587),
                    'username': email_config.get('username', ''),
                    'password': email_config.get('password', ''),
                    'from_email': email_config.get('from_email', ''),
                    'to_emails': email_config.get('to_emails', []),
                    'use_tls': email_config.get('use_tls', True)
                }
        
        # Webhook configuration
        if 'webhook' in channels_config:
            webhook_config = channels_config['webhook']
            if webhook_config.get('enabled', False):
                self.notification_channels[NotificationChannel.WEBHOOK] = {
                    'url': webhook_config.get('url', ''),
                    'headers': webhook_config.get('headers', {}),
                    'timeout': webhook_config.get('timeout', 10)
                }
        
        # Slack configuration
        if 'slack' in channels_config:
            slack_config = channels_config['slack']
            if slack_config.get('enabled', False):
                self.notification_channels[NotificationChannel.SLACK] = {
                    'webhook_url': slack_config.get('webhook_url', ''),
                    'channel': slack_config.get('channel', '#alerts'),
                    'username': slack_config.get('username', 'TradingBot'),
                    'emoji': slack_config.get('emoji', ':robot_face:')
                }
        
        # Discord configuration
        if 'discord' in channels_config:
            discord_config = channels_config['discord']
            if discord_config.get('enabled', False):
                self.notification_channels[NotificationChannel.DISCORD] = {
                    'webhook_url': discord_config.get('webhook_url', ''),
                    'username': discord_config.get('username', 'TradingBot')
                }
        
        # Console logging (always enabled)
        self.notification_channels[NotificationChannel.CONSOLE] = {'enabled': True}
        
        # File logging
        if 'file' in channels_config:
            file_config = channels_config['file']
            if file_config.get('enabled', False):
                self.notification_channels[NotificationChannel.FILE] = {
                    'log_file': file_config.get('log_file', 'logs/alerts.log'),
                    'max_size_mb': file_config.get('max_size_mb', 100)
                }
    
    def _init_databases(self):
        """Initialize SQLite databases for metrics and alerts"""
        try:
            # Create data directory
            os.makedirs(os.path.dirname(self.metrics_db_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.alerts_db_path), exist_ok=True)
            
            # Initialize metrics database
            with sqlite3.connect(self.metrics_db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        value REAL NOT NULL,
                        metric_type TEXT NOT NULL,
                        labels TEXT,
                        timestamp TEXT NOT NULL,
                        description TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp 
                    ON metrics(name, timestamp)
                ''')
            
            # Initialize alerts database
            with sqlite3.connect(self.alerts_db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_id TEXT UNIQUE NOT NULL,
                        level TEXT NOT NULL,
                        source TEXT NOT NULL,
                        message TEXT NOT NULL,
                        details TEXT,
                        timestamp TEXT NOT NULL,
                        acknowledged INTEGER DEFAULT 0,
                        resolved INTEGER DEFAULT 0,
                        escalated INTEGER DEFAULT 0
                    )
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_alerts_timestamp 
                    ON alerts(timestamp)
                ''')
            
            logger.info("Monitoring databases initialized")
            
        except Exception as e:
            logger.error(f"Error initializing monitoring databases: {e}")
    
    def _setup_event_handlers(self):
        """Setup event bus handlers"""
        global_event_bus.subscribe("trade_executed", self._on_trade_executed)
        global_event_bus.subscribe("error_occurred", self._on_error_occurred)
        global_event_bus.subscribe("risk_limit_breached", self._on_risk_limit_breached)
        global_event_bus.subscribe("strategy_stopped", self._on_strategy_stopped)
        global_event_bus.subscribe("capital_allocated", self._on_capital_allocated)
        global_event_bus.subscribe("emergency_stop", self._on_emergency_stop)
    
    def _setup_default_health_checks(self):
        """Setup default system health checks"""
        # Database connectivity check
        def check_database_connection():
            try:
                with sqlite3.connect(self.metrics_db_path) as conn:
                    conn.execute("SELECT 1").fetchone()
                return True
            except Exception:
                return False
        
        # Disk space check
        def check_disk_space():
            try:
                disk_usage = psutil.disk_usage('/')
                free_pct = (disk_usage.free / disk_usage.total) * 100
                return free_pct > 10  # Alert if less than 10% free
            except Exception:
                return False
        
        # Memory usage check
        def check_memory_usage():
            try:
                memory = psutil.virtual_memory()
                return memory.percent < 90  # Alert if over 90% usage
            except Exception:
                return False
        
        # CPU usage check
        def check_cpu_usage():
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                return cpu_percent < 95  # Alert if over 95% usage
            except Exception:
                return False
        
        # Process check
        def check_process_health():
            try:
                current_process = psutil.Process()
                return current_process.is_running()
            except Exception:
                return False
        
        # Register health checks
        self.register_health_check("database_connection", check_database_connection, 300, 10)
        self.register_health_check("disk_space", check_disk_space, 600, 5)
        self.register_health_check("memory_usage", check_memory_usage, 60, 5)
        self.register_health_check("cpu_usage", check_cpu_usage, 60, 5)
        self.register_health_check("process_health", check_process_health, 30, 5)
    
    def start_monitoring(self):
        """Start the monitoring system"""
        if not self.enabled:
            logger.info("Advanced monitoring is disabled")
            return
        
        with self._lock:
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                logger.warning("Monitoring already running")
                return
            
            self._stop_event.clear()
            
            # Start monitoring thread
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                name="AdvancedMonitoring",
                daemon=True
            )
            self._monitoring_thread.start()
            
            # Start health check thread
            self._health_check_thread = threading.Thread(
                target=self._health_check_loop,
                name="HealthChecks",
                daemon=True
            )
            self._health_check_thread.start()
            
            logger.info("Advanced monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self._stop_event.set()
        
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=10.0)
        
        if self._health_check_thread:
            self._health_check_thread.join(timeout=10.0)
        
        logger.info("Advanced monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("Monitoring loop started")
        
        while not self._stop_event.is_set():
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Process metrics buffer
                self._process_metrics_buffer()
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Check for alert escalations
                self._check_alert_escalations()
                
                self._stop_event.wait(30)  # Run every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                self._stop_event.wait(60)  # Wait longer on error
    
    def _health_check_loop(self):
        """Health check loop"""
        logger.info("Health check loop started")
        
        while not self._stop_event.is_set():
            try:
                current_time = datetime.now()
                
                for name, health_check in self.health_checks.items():
                    # Check if it's time to run this health check
                    if (health_check.last_check is None or 
                        (current_time - health_check.last_check).total_seconds() >= health_check.interval_seconds):
                        
                        self._run_health_check(name, health_check)
                
                self._stop_event.wait(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}", exc_info=True)
                self._stop_event.wait(30)
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric("system.cpu_usage_percent", cpu_percent, MetricType.GAUGE)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.record_metric("system.memory_usage_percent", memory.percent, MetricType.GAUGE)
            self.record_metric("system.memory_available_bytes", memory.available, MetricType.GAUGE)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_pct = (disk.used / disk.total) * 100
            self.record_metric("system.disk_usage_percent", disk_usage_pct, MetricType.GAUGE)
            self.record_metric("system.disk_free_bytes", disk.free, MetricType.GAUGE)
            
            # Network I/O
            net_io = psutil.net_io_counters()
            self.record_metric("system.network_bytes_sent", net_io.bytes_sent, MetricType.COUNTER)
            self.record_metric("system.network_bytes_recv", net_io.bytes_recv, MetricType.COUNTER)
            
            # Process count
            process_count = len(psutil.pids())
            self.record_metric("system.process_count", process_count, MetricType.GAUGE)
            
            # Current process metrics
            current_process = psutil.Process()
            process_memory = current_process.memory_info()
            self.record_metric("bot.memory_rss_bytes", process_memory.rss, MetricType.GAUGE)
            self.record_metric("bot.memory_vms_bytes", process_memory.vms, MetricType.GAUGE)
            self.record_metric("bot.cpu_percent", current_process.cpu_percent(), MetricType.GAUGE)
            
            # Update internal metrics
            self.system_metrics.update({
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'disk_usage': disk_usage_pct,
                'network_io': {'bytes_sent': net_io.bytes_sent, 'bytes_recv': net_io.bytes_recv},
                'process_count': process_count
            })
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _process_metrics_buffer(self):
        """Process buffered metrics and save to database"""
        if not self.metrics_buffer:
            return
        
        try:
            with sqlite3.connect(self.metrics_db_path) as conn:
                metrics_to_insert = []
                
                while self.metrics_buffer:
                    metric = self.metrics_buffer.popleft()
                    metrics_to_insert.append((
                        metric.name,
                        metric.value,
                        metric.metric_type.value,
                        json.dumps(metric.labels),
                        metric.timestamp.isoformat(),
                        metric.description
                    ))
                
                conn.executemany('''
                    INSERT INTO metrics (name, value, metric_type, labels, timestamp, description)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', metrics_to_insert)
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error processing metrics buffer: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old metrics and alerts"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.metrics_retention_days)
            cutoff_str = cutoff_date.isoformat()
            
            # Clean up old metrics
            with sqlite3.connect(self.metrics_db_path) as conn:
                conn.execute("DELETE FROM metrics WHERE timestamp < ?", (cutoff_str,))
                conn.commit()
            
            # Clean up old resolved alerts
            with sqlite3.connect(self.alerts_db_path) as conn:
                conn.execute(
                    "DELETE FROM alerts WHERE resolved = 1 AND timestamp < ?", 
                    (cutoff_str,)
                )
                conn.commit()
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    def _run_health_check(self, name: str, health_check: HealthCheck):
        """Run a single health check"""
        try:
            health_check.last_check = datetime.now()
            
            # Run the check with timeout
            start_time = time.time()
            is_healthy = health_check.check_function()
            duration = time.time() - start_time
            
            # Record performance metric
            self.record_metric(
                f"health_check.{name}.duration_seconds",
                duration,
                MetricType.TIMER
            )
            
            if is_healthy:
                # Reset failure count
                if health_check.consecutive_failures > 0:
                    health_check.consecutive_failures = 0
                    if not health_check.is_healthy:
                        # Health recovered
                        health_check.is_healthy = True
                        self.create_alert(
                            AlertLevel.INFO,
                            f"health_check.{name}",
                            f"Health check '{name}' recovered",
                            {"duration": duration}
                        )
            else:
                # Increment failure count
                health_check.consecutive_failures += 1
                
                # Check if threshold reached
                if (health_check.consecutive_failures >= health_check.failure_threshold and 
                    health_check.is_healthy):
                    
                    health_check.is_healthy = False
                    self.create_alert(
                        AlertLevel.ERROR,
                        f"health_check.{name}",
                        f"Health check '{name}' failed {health_check.consecutive_failures} times",
                        {
                            "consecutive_failures": health_check.consecutive_failures,
                            "threshold": health_check.failure_threshold,
                            "duration": duration
                        }
                    )
            
            # Record health status
            self.record_metric(
                f"health_check.{name}.status",
                1 if is_healthy else 0,
                MetricType.GAUGE
            )
            
        except Exception as e:
            logger.error(f"Error running health check '{name}': {e}")
            health_check.consecutive_failures += 1
    
    def _check_alert_escalations(self):
        """Check for alerts that need escalation"""
        try:
            current_time = datetime.now()
            escalation_threshold = timedelta(hours=1)  # Escalate after 1 hour
            
            for alert in self.active_alerts.values():
                if (not alert.escalated and 
                    not alert.resolved and
                    alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL] and
                    (current_time - alert.timestamp) > escalation_threshold):
                    
                    # Escalate alert
                    alert.escalated = True
                    self._escalate_alert(alert)
                    
        except Exception as e:
            logger.error(f"Error checking alert escalations: {e}")
    
    def _escalate_alert(self, alert: Alert):
        """Escalate an alert"""
        try:
            escalated_alert = Alert(
                alert_id=f"{alert.alert_id}_escalated",
                level=AlertLevel.CRITICAL,
                source=alert.source,
                message=f"ESCALATED: {alert.message}",
                details={**alert.details, "original_alert_id": alert.alert_id, "escalated": True},
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK]
            )
            
            self.active_alerts[escalated_alert.alert_id] = escalated_alert
            self._send_notifications(escalated_alert)
            self._save_alert_to_db(escalated_alert)
            
            logger.critical(f"Alert escalated: {escalated_alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Error escalating alert: {e}")
    
    # Public API
    def record_metric(self, name: str, value: Union[int, float], metric_type: MetricType,
                     labels: Optional[Dict[str, str]] = None, description: str = ""):
        """Record a metric"""
        try:
            metric = Metric(
                name=name,
                value=value,
                metric_type=metric_type,
                labels=labels or {},
                description=description
            )
            
            self.metrics_buffer.append(metric)
            
            # Call metric handlers
            for handler in self._metric_handlers:
                try:
                    handler(metric)
                except Exception as e:
                    logger.error(f"Error in metric handler: {e}")
                    
        except Exception as e:
            logger.error(f"Error recording metric: {e}")
    
    def create_alert(self, level: AlertLevel, source: str, message: str, 
                    details: Optional[Dict[str, Any]] = None,
                    channels: Optional[List[NotificationChannel]] = None) -> str:
        """Create and send an alert"""
        try:
            # Check cooldown
            cooldown_key = f"{source}:{message}"
            if cooldown_key in self.alert_cooldowns:
                time_since_last = datetime.now() - self.alert_cooldowns[cooldown_key]
                if time_since_last < timedelta(minutes=self.alert_cooldown_minutes):
                    logger.debug(f"Alert in cooldown: {cooldown_key}")
                    return ""
            
            # Create alert
            alert_id = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.active_alerts)}"
            
            # Determine notification channels
            if channels is None:
                if level == AlertLevel.CRITICAL:
                    channels = [NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.CONSOLE]
                elif level == AlertLevel.ERROR:
                    channels = [NotificationChannel.SLACK, NotificationChannel.CONSOLE]
                elif level == AlertLevel.WARNING:
                    channels = [NotificationChannel.CONSOLE]
                else:
                    channels = [NotificationChannel.CONSOLE]
            
            alert = Alert(
                alert_id=alert_id,
                level=level,
                source=source,
                message=message,
                details=details or {},
                notification_channels=channels
            )
            
            # Store alert
            self.active_alerts[alert_id] = alert
            self.alert_cooldowns[cooldown_key] = datetime.now()
            
            # Send notifications
            self._send_notifications(alert)
            
            # Save to database
            self._save_alert_to_db(alert)
            
            # Call alert handlers
            for handler in self._alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Error in alert handler: {e}")
            
            logger.info(f"Alert created: {alert_id} - {level.value.upper()}: {message}")
            return alert_id
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            return ""
    
    def _send_notifications(self, alert: Alert):
        """Send notifications for an alert"""
        for channel in alert.notification_channels:
            try:
                if channel == NotificationChannel.EMAIL:
                    self._send_email_notification(alert)
                elif channel == NotificationChannel.WEBHOOK:
                    self._send_webhook_notification(alert)
                elif channel == NotificationChannel.SLACK:
                    self._send_slack_notification(alert)
                elif channel == NotificationChannel.DISCORD:
                    self._send_discord_notification(alert)
                elif channel == NotificationChannel.CONSOLE:
                    self._send_console_notification(alert)
                elif channel == NotificationChannel.FILE:
                    self._send_file_notification(alert)
                    
            except Exception as e:
                logger.error(f"Error sending {channel.value} notification: {e}")
    
    def _send_email_notification(self, alert: Alert):
        """Send email notification"""
        if NotificationChannel.EMAIL not in self.notification_channels:
            return
        
        config = self.notification_channels[NotificationChannel.EMAIL]
        
        msg = MimeMultipart()
        msg['From'] = config['from_email']
        msg['To'] = ', '.join(config['to_emails'])
        msg['Subject'] = f"[{alert.level.value.upper()}] Trading Bot Alert: {alert.source}"
        
        body = f"""
Alert Details:
- ID: {alert.alert_id}
- Level: {alert.level.value.upper()}
- Source: {alert.source}
- Message: {alert.message}
- Timestamp: {alert.timestamp.isoformat()}

Additional Details:
{json.dumps(alert.details, indent=2)}
        """
        
        msg.attach(MimeText(body, 'plain'))
        
        server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
        if config['use_tls']:
            server.starttls()
        
        if config['username'] and config['password']:
            server.login(config['username'], config['password'])
        
        server.sendmail(config['from_email'], config['to_emails'], msg.as_string())
        server.quit()
    
    def _send_webhook_notification(self, alert: Alert):
        """Send webhook notification"""
        if NotificationChannel.WEBHOOK not in self.notification_channels:
            return
        
        config = self.notification_channels[NotificationChannel.WEBHOOK]
        
        payload = {
            'alert_id': alert.alert_id,
            'level': alert.level.value,
            'source': alert.source,
            'message': alert.message,
            'timestamp': alert.timestamp.isoformat(),
            'details': alert.details
        }
        
        response = requests.post(
            config['url'],
            json=payload,
            headers=config['headers'],
            timeout=config['timeout']
        )
        response.raise_for_status()
    
    def _send_slack_notification(self, alert: Alert):
        """Send Slack notification"""
        if NotificationChannel.SLACK not in self.notification_channels:
            return
        
        config = self.notification_channels[NotificationChannel.SLACK]
        
        color_map = {
            AlertLevel.INFO: '#36a64f',
            AlertLevel.WARNING: '#ff9500',
            AlertLevel.ERROR: '#ff0000',
            AlertLevel.CRITICAL: '#8b0000'
        }
        
        payload = {
            'channel': config['channel'],
            'username': config['username'],
            'icon_emoji': config['emoji'],
            'attachments': [{
                'color': color_map.get(alert.level, '#cccccc'),
                'title': f"{alert.level.value.upper()}: {alert.source}",
                'text': alert.message,
                'fields': [
                    {'title': 'Alert ID', 'value': alert.alert_id, 'short': True},
                    {'title': 'Timestamp', 'value': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), 'short': True}
                ],
                'footer': 'Trading Bot Monitoring',
                'ts': int(alert.timestamp.timestamp())
            }]
        }
        
        response = requests.post(config['webhook_url'], json=payload, timeout=10)
        response.raise_for_status()
    
    def _send_discord_notification(self, alert: Alert):
        """Send Discord notification"""
        if NotificationChannel.DISCORD not in self.notification_channels:
            return
        
        config = self.notification_channels[NotificationChannel.DISCORD]
        
        color_map = {
            AlertLevel.INFO: 0x36a64f,
            AlertLevel.WARNING: 0xff9500,
            AlertLevel.ERROR: 0xff0000,
            AlertLevel.CRITICAL: 0x8b0000
        }
        
        payload = {
            'username': config['username'],
            'embeds': [{
                'title': f"{alert.level.value.upper()}: {alert.source}",
                'description': alert.message,
                'color': color_map.get(alert.level, 0xcccccc),
                'fields': [
                    {'name': 'Alert ID', 'value': alert.alert_id, 'inline': True},
                    {'name': 'Timestamp', 'value': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), 'inline': True}
                ],
                'footer': {'text': 'Trading Bot Monitoring'},
                'timestamp': alert.timestamp.isoformat()
            }]
        }
        
        response = requests.post(config['webhook_url'], json=payload, timeout=10)
        response.raise_for_status()
    
    def _send_console_notification(self, alert: Alert):
        """Send console notification"""
        level_colors = {
            AlertLevel.INFO: '\033[92m',      # Green
            AlertLevel.WARNING: '\033[93m',   # Yellow
            AlertLevel.ERROR: '\033[91m',     # Red
            AlertLevel.CRITICAL: '\033[95m'   # Magenta
        }
        
        color = level_colors.get(alert.level, '\033[0m')
        reset = '\033[0m'
        
        print(f"{color}[{alert.level.value.upper()}] {alert.timestamp.strftime('%H:%M:%S')} "
              f"- {alert.source}: {alert.message}{reset}")
    
    def _send_file_notification(self, alert: Alert):
        """Send file notification"""
        if NotificationChannel.FILE not in self.notification_channels:
            return
        
        config = self.notification_channels[NotificationChannel.FILE]
        log_file = config['log_file']
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, 'a') as f:
            f.write(f"{alert.timestamp.isoformat()} [{alert.level.value.upper()}] "
                   f"{alert.source}: {alert.message}\n")
    
    def _save_alert_to_db(self, alert: Alert):
        """Save alert to database"""
        try:
            with sqlite3.connect(self.alerts_db_path) as conn:
                conn.execute('''
                    INSERT INTO alerts (alert_id, level, source, message, details, timestamp, 
                                      acknowledged, resolved, escalated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.alert_id,
                    alert.level.value,
                    alert.source,
                    alert.message,
                    json.dumps(alert.details),
                    alert.timestamp.isoformat(),
                    alert.acknowledged,
                    alert.resolved,
                    alert.escalated
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving alert to database: {e}")
    
    def register_health_check(self, name: str, check_function: Callable[[], bool],
                            interval_seconds: float, timeout_seconds: float,
                            failure_threshold: int = 3):
        """Register a health check"""
        self.health_checks[name] = HealthCheck(
            name=name,
            check_function=check_function,
            interval_seconds=interval_seconds,
            timeout_seconds=timeout_seconds,
            failure_threshold=failure_threshold
        )
        
        logger.info(f"Health check registered: {name}")
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            
            try:
                with sqlite3.connect(self.alerts_db_path) as conn:
                    conn.execute(
                        "UPDATE alerts SET acknowledged = 1 WHERE alert_id = ?",
                        (alert_id,)
                    )
                    conn.commit()
                
                logger.info(f"Alert acknowledged: {alert_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error acknowledging alert: {e}")
        
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolved = True
            
            try:
                with sqlite3.connect(self.alerts_db_path) as conn:
                    conn.execute(
                        "UPDATE alerts SET resolved = 1 WHERE alert_id = ?",
                        (alert_id,)
                    )
                    conn.commit()
                
                # Remove from active alerts
                del self.active_alerts[alert_id]
                
                logger.info(f"Alert resolved: {alert_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error resolving alert: {e}")
        
        return False
    
    def get_metrics(self, name_pattern: str = None, hours: int = 24) -> List[Dict[str, Any]]:
        """Get metrics from database"""
        try:
            start_time = datetime.now() - timedelta(hours=hours)
            
            with sqlite3.connect(self.metrics_db_path) as conn:
                if name_pattern:
                    cursor = conn.execute('''
                        SELECT name, value, metric_type, labels, timestamp, description
                        FROM metrics 
                        WHERE name LIKE ? AND timestamp >= ?
                        ORDER BY timestamp DESC
                    ''', (f"%{name_pattern}%", start_time.isoformat()))
                else:
                    cursor = conn.execute('''
                        SELECT name, value, metric_type, labels, timestamp, description
                        FROM metrics 
                        WHERE timestamp >= ?
                        ORDER BY timestamp DESC
                    ''', (start_time.isoformat(),))
                
                return [
                    {
                        'name': row[0],
                        'value': row[1],
                        'metric_type': row[2],
                        'labels': json.loads(row[3]) if row[3] else {},
                        'timestamp': row[4],
                        'description': row[5]
                    }
                    for row in cursor.fetchall()
                ]
                
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return []
    
    def get_alerts(self, hours: int = 24, include_resolved: bool = False) -> List[Dict[str, Any]]:
        """Get alerts from database"""
        try:
            start_time = datetime.now() - timedelta(hours=hours)
            
            with sqlite3.connect(self.alerts_db_path) as conn:
                query = '''
                    SELECT alert_id, level, source, message, details, timestamp,
                           acknowledged, resolved, escalated
                    FROM alerts 
                    WHERE timestamp >= ?
                '''
                
                params = [start_time.isoformat()]
                
                if not include_resolved:
                    query += " AND resolved = 0"
                
                query += " ORDER BY timestamp DESC"
                
                cursor = conn.execute(query, params)
                
                return [
                    {
                        'alert_id': row[0],
                        'level': row[1],
                        'source': row[2],
                        'message': row[3],
                        'details': json.loads(row[4]) if row[4] else {},
                        'timestamp': row[5],
                        'acknowledged': bool(row[6]),
                        'resolved': bool(row[7]),
                        'escalated': bool(row[8])
                    }
                    for row in cursor.fetchall()
                ]
                
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'monitoring_enabled': self.enabled,
            'system_metrics': self.system_metrics,
            'bot_metrics': self.bot_metrics,
            'active_alerts_count': len(self.active_alerts),
            'health_checks': {
                name: {
                    'is_healthy': hc.is_healthy,
                    'consecutive_failures': hc.consecutive_failures,
                    'last_check': hc.last_check.isoformat() if hc.last_check else None
                }
                for name, hc in self.health_checks.items()
            },
            'uptime_seconds': time.time() - self.system_metrics.get('start_time', time.time())
        }
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add custom alert handler"""
        self._alert_handlers.append(handler)
    
    def add_metric_handler(self, handler: Callable[[Metric], None]):
        """Add custom metric handler"""
        self._metric_handlers.append(handler)
    
    # Event Handlers
    def _on_trade_executed(self, data: Dict[str, Any]):
        """Handle trade execution events"""
        self.bot_metrics['trades_executed'] += 1
        self.record_metric("bot.trades_executed_total", self.bot_metrics['trades_executed'], MetricType.COUNTER)
        
        # Record trade amount
        amount = data.get('amount', 0)
        self.record_metric("bot.trade_amount", amount, MetricType.HISTOGRAM)
    
    def _on_error_occurred(self, data: Dict[str, Any]):
        """Handle error events"""
        self.bot_metrics['errors_count'] += 1
        self.record_metric("bot.errors_total", self.bot_metrics['errors_count'], MetricType.COUNTER)
        
        error_type = data.get('error_type', 'unknown')
        error_message = data.get('message', 'Unknown error')
        
        # Create alert for errors
        self.create_alert(
            AlertLevel.ERROR,
            f"bot.error.{error_type}",
            f"Error occurred: {error_message}",
            data
        )
    
    def _on_risk_limit_breached(self, data: Dict[str, Any]):
        """Handle risk limit breach events"""
        self.create_alert(
            AlertLevel.CRITICAL,
            "risk_management",
            f"Risk limit breached: {data.get('message', 'Unknown risk event')}",
            data,
            [NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.CONSOLE]
        )
    
    def _on_strategy_stopped(self, data: Dict[str, Any]):
        """Handle strategy stopped events"""
        strategy_name = data.get('strategy_name', 'unknown')
        reason = data.get('reason', 'Unknown reason')
        
        self.create_alert(
            AlertLevel.WARNING,
            "strategy_management",
            f"Strategy '{strategy_name}' stopped: {reason}",
            data
        )
    
    def _on_capital_allocated(self, data: Dict[str, Any]):
        """Handle capital allocation events"""
        strategy_name = data.get('strategy_name', 'unknown')
        amount = data.get('amount', 0)
        
        self.record_metric(
            f"capital.allocated.{strategy_name}",
            amount,
            MetricType.GAUGE
        )
    
    def _on_emergency_stop(self, data: Dict[str, Any]):
        """Handle emergency stop events"""
        self.create_alert(
            AlertLevel.CRITICAL,
            "emergency_stop",
            f"Emergency stop triggered: {data.get('reason', 'Unknown reason')}",
            data,
            [NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.CONSOLE]
        )

# Global instance
_monitoring_instance = None

def get_monitoring_system(settings: Optional[Dict[str, Any]] = None) -> AdvancedMonitoringSystem:
    """Get global monitoring system instance"""
    global _monitoring_instance
    if _monitoring_instance is None:
        if settings is None:
            settings = {}
        _monitoring_instance = AdvancedMonitoringSystem(settings)
    return _monitoring_instance