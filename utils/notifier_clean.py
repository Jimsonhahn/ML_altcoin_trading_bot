# utils/notifier.py
"""
Clean Telegram-Only Notification Manager for Trading Bot
Handles alerts for strategy changes, market phases, drawdowns, API errors, and bot crashes
Supports only Telegram notifications (Email functionality removed)
"""

import logging
import requests
import traceback
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import json
import os
from threading import Lock

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AlertType(Enum):
    """Types of alerts"""
    STRATEGY_CHANGE = "STRATEGY_CHANGE"
    MARKET_PHASE_CHANGE = "MARKET_PHASE_CHANGE"
    DRAWDOWN = "DRAWDOWN"
    API_ERROR = "API_ERROR"
    BOT_CRASH = "BOT_CRASH"
    TRADE_EXECUTED = "TRADE_EXECUTED"
    SYSTEM_STATUS = "SYSTEM_STATUS"
    PORTFOLIO_UPDATE = "PORTFOLIO_UPDATE"


@dataclass
class Alert:
    """Alert data structure"""
    message: str
    level: AlertLevel
    alert_type: AlertType
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'message': self.message,
            'level': self.level.value,
            'type': self.alert_type.value,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata or {}
        }


class NotificationManager:
    """
    Clean Telegram-Only Notification Manager
    """
    
    def __init__(self, settings: Optional[Dict] = None):
        self.settings = settings or {}
        
        # Configuration
        self.telegram_config = self.settings.get('notifications', {}).get('telegram', {})
        self.alert_config = self.settings.get('notifications', {}).get('alerts', {})
        
        # Rate limiting
        self.rate_limit_window = timedelta(minutes=5)
        self.max_alerts_per_window = 10
        self.alert_history: List[Alert] = []
        self.lock = Lock()
        
        # Alert filtering
        self.min_level = AlertLevel(self.alert_config.get('min_level', 'INFO'))
        self.enabled_types = set(self.alert_config.get('enabled_types', [t.value for t in AlertType]))
        
        # Initialize Telegram
        self.telegram_enabled = False
        self._load_credentials()
        self._initialize_telegram()
        
        # Startup notification
        if self.telegram_enabled:
            self.send_alert("Trading Bot notification system initialized", AlertLevel.INFO, AlertType.SYSTEM_STATUS)
    
    def _load_credentials(self):
        """Load Telegram credentials from various sources"""
        # Try environment variables first
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        # Fallback to config file
        if not self.telegram_bot_token:
            self.telegram_bot_token = self.telegram_config.get('bot_token')
        if not self.telegram_chat_id:
            self.telegram_chat_id = self.telegram_config.get('chat_id')
        
        logger.info("Telegram credentials loaded")
    
    def _initialize_telegram(self):
        """Initialize Telegram notifications"""
        if self.telegram_config.get('enabled', False):
            if self.telegram_bot_token and self.telegram_chat_id:
                self.telegram_enabled = True
                logger.info("Telegram notifications enabled")
            else:
                missing = []
                if not self.telegram_bot_token:
                    missing.append("bot_token")
                if not self.telegram_chat_id:
                    missing.append("chat_id")
                logger.warning(f"Telegram disabled - missing: {', '.join(missing)}")
        
        if not self.telegram_enabled:
            logger.warning("Telegram notifications disabled")
    
    def send_alert(self, message: str, level: AlertLevel = AlertLevel.INFO, 
                  alert_type: AlertType = AlertType.SYSTEM_STATUS, 
                  metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send alert through Telegram
        
        Args:
            message: Alert message
            level: Alert severity level
            alert_type: Type of alert
            metadata: Additional data
            
        Returns:
            True if alert was sent successfully
        """
        try:
            # Create alert object
            alert = Alert(
                message=message,
                level=level,
                alert_type=alert_type,
                timestamp=datetime.now(),
                metadata=metadata
            )
            
            # Check if alert should be sent
            if not self._should_send_alert(alert):
                return False
            
            # Add to history for rate limiting
            with self.lock:
                self.alert_history.append(alert)
                self._cleanup_old_alerts()
            
            # Log the alert
            log_level = {
                AlertLevel.INFO: logging.INFO,
                AlertLevel.WARNING: logging.WARNING,
                AlertLevel.ERROR: logging.ERROR,
                AlertLevel.CRITICAL: logging.CRITICAL
            }.get(level, logging.INFO)
            
            logger.log(log_level, f"ALERT [{alert_type.value}]: {message}")
            
            # Send through Telegram
            if self.telegram_enabled:
                return self._send_telegram(alert)
            
            return False
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
            return False
    
    def _should_send_alert(self, alert: Alert) -> bool:
        """Check if alert should be sent based on filters and rate limits"""
        # Check alert level
        level_order = {AlertLevel.INFO: 0, AlertLevel.WARNING: 1, AlertLevel.ERROR: 2, AlertLevel.CRITICAL: 3}
        if level_order[alert.level] < level_order[self.min_level]:
            return False
        
        # Check alert type
        if alert.alert_type.value not in self.enabled_types:
            return False
        
        # Check rate limiting
        with self.lock:
            recent_alerts = [a for a in self.alert_history 
                           if datetime.now() - a.timestamp < self.rate_limit_window]
            
            if len(recent_alerts) >= self.max_alerts_per_window:
                # Allow critical alerts even if rate limited
                if alert.level != AlertLevel.CRITICAL:
                    logger.warning(f"Rate limit exceeded, dropping {alert.level.value} alert")
                    return False
        
        return True
    
    def _cleanup_old_alerts(self):
        """Remove old alerts from history"""
        cutoff = datetime.now() - self.rate_limit_window
        self.alert_history = [a for a in self.alert_history if a.timestamp > cutoff]
    
    def _send_telegram(self, alert: Alert) -> bool:
        """Send alert via Telegram"""
        try:
            # Format message
            emoji_map = {
                AlertLevel.INFO: "ℹ️",
                AlertLevel.WARNING: "⚠️", 
                AlertLevel.ERROR: "❌",
                AlertLevel.CRITICAL: "🚨"
            }
            
            emoji = emoji_map.get(alert.level, "📢")
            formatted_message = f"{emoji} *{alert.level.value}* - {alert.alert_type.value}\n\n{alert.message}"
            
            # Add metadata if present
            if alert.metadata:
                formatted_message += f"\n\n📊 *Details:*"
                for key, value in alert.metadata.items():
                    formatted_message += f"\n• {key}: `{value}`"
            
            formatted_message += f"\n\n🕐 {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Send to Telegram
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': formatted_message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            logger.debug("Telegram alert sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False
    
    def send_strategy_change_alert(self, old_strategy: str, new_strategy: str, reason: str = ""):
        """Send alert for strategy change"""
        message = f"Strategy changed from '{old_strategy}' to '{new_strategy}'"
        if reason:
            message += f". Reason: {reason}"
        
        metadata = {
            'old_strategy': old_strategy,
            'new_strategy': new_strategy,
            'reason': reason
        }
        
        self.send_alert(message, AlertLevel.INFO, AlertType.STRATEGY_CHANGE, metadata)
    
    def send_market_phase_change_alert(self, old_phase: str, new_phase: str, confidence: float = None):
        """Send alert for market phase change"""
        message = f"Market phase changed from '{old_phase}' to '{new_phase}'"
        if confidence:
            message += f" (confidence: {confidence:.2%})"
        
        metadata = {
            'old_phase': old_phase,
            'new_phase': new_phase,
            'confidence': confidence
        }
        
        self.send_alert(message, AlertLevel.INFO, AlertType.MARKET_PHASE_CHANGE, metadata)
    
    def send_drawdown_alert(self, current_drawdown: float, max_drawdown: float, portfolio_value: float):
        """Send alert for significant drawdown"""
        message = f"Portfolio drawdown: {current_drawdown:.2%} (max: {max_drawdown:.2%}). Current value: ${portfolio_value:,.2f}"
        
        level = AlertLevel.WARNING if current_drawdown < 0.1 else AlertLevel.ERROR
        if current_drawdown > 0.2:  # 20% drawdown
            level = AlertLevel.CRITICAL
        
        metadata = {
            'current_drawdown': current_drawdown,
            'max_drawdown': max_drawdown,
            'portfolio_value': portfolio_value
        }
        
        self.send_alert(message, level, AlertType.DRAWDOWN, metadata)
    
    def send_api_error_alert(self, api_name: str, error_message: str, error_count: int = 1):
        """Send alert for API errors"""
        message = f"API Error in {api_name}: {error_message}"
        if error_count > 1:
            message += f" (occurred {error_count} times)"
        
        level = AlertLevel.WARNING if error_count < 5 else AlertLevel.ERROR
        
        metadata = {
            'api_name': api_name,
            'error_message': error_message,
            'error_count': error_count
        }
        
        self.send_alert(message, level, AlertType.API_ERROR, metadata)
    
    def send_bot_crash_alert(self, error_message: str, stack_trace: str = None):
        """Send critical alert for bot crashes"""
        message = f"Trading bot crashed: {error_message}"
        
        metadata = {
            'error_message': error_message,
            'stack_trace': stack_trace or traceback.format_exc()
        }
        
        self.send_alert(message, AlertLevel.CRITICAL, AlertType.BOT_CRASH, metadata)
    
    def send_trade_alert(self, action: str, symbol: str, quantity: float, price: float, strategy: str):
        """Send alert for trade execution"""
        message = f"Trade executed: {action} {quantity} {symbol} at ${price:.4f} (strategy: {strategy})"
        
        metadata = {
            'action': action,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'strategy': strategy
        }
        
        self.send_alert(message, AlertLevel.INFO, AlertType.TRADE_EXECUTED, metadata)
    
    def send_portfolio_update_alert(self, total_value: float, pnl_24h: float, top_performers: List[Dict] = None):
        """Send periodic portfolio update"""
        message = f"Portfolio Update: ${total_value:,.2f} (24h P&L: {pnl_24h:+.2f})"
        
        metadata = {
            'total_value': total_value,
            'pnl_24h': pnl_24h,
            'top_performers': top_performers or []
        }
        
        self.send_alert(message, AlertLevel.INFO, AlertType.PORTFOLIO_UPDATE, metadata)
    
    def get_alert_history(self, hours: int = 24) -> List[Dict]:
        """Get recent alert history"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [alert.to_dict() for alert in self.alert_history if alert.timestamp > cutoff]
    
    def test_notification(self) -> bool:
        """Test Telegram notification"""
        if self.telegram_enabled:
            test_alert = Alert(
                message="This is a test notification from your Trading Bot",
                level=AlertLevel.INFO,
                alert_type=AlertType.SYSTEM_STATUS,
                timestamp=datetime.now(),
                metadata={'test': True}
            )
            return self._send_telegram(test_alert)
        return False


# Global notifier instance
_global_notifier: Optional[NotificationManager] = None


def initialize_notifier(settings: Optional[Dict] = None) -> NotificationManager:
    """Initialize global notifier instance"""
    global _global_notifier
    _global_notifier = NotificationManager(settings)
    return _global_notifier


def get_notifier() -> Optional[NotificationManager]:
    """Get global notifier instance"""
    return _global_notifier


def send_alert(message: str, level: AlertLevel = AlertLevel.INFO, 
               alert_type: AlertType = AlertType.SYSTEM_STATUS) -> bool:
    """
    Simple interface to send alerts
    
    Args:
        message: Alert message
        level: Alert severity level  
        alert_type: Type of alert
        
    Returns:
        True if alert was sent successfully
    """
    notifier = get_notifier()
    if notifier:
        return notifier.send_alert(message, level, alert_type)
    else:
        logger.warning(f"No notifier initialized: {message}")
        return False


# Convenience functions
def send_info(message: str) -> bool:
    """Send info level alert"""
    return send_alert(message, AlertLevel.INFO)


def send_warning(message: str) -> bool:
    """Send warning level alert"""
    return send_alert(message, AlertLevel.WARNING)


def send_error(message: str) -> bool:
    """Send error level alert"""
    return send_alert(message, AlertLevel.ERROR)


def send_critical(message: str) -> bool:
    """Send critical level alert"""
    return send_alert(message, AlertLevel.CRITICAL)