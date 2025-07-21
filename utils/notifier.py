# utils/notifier.py
"""
Final Clean Telegram-Only Notification Manager for Trading Bot
Email functionality completely removed - Telegram only
"""

import logging
import requests
import traceback
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from enum import Enum
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
        self.alert_count = 0
        self.last_reset = datetime.now()
        self.lock = Lock()
        
        # Alert filtering
        try:
            self.min_level = AlertLevel(self.alert_config.get('min_level', 'INFO'))
        except ValueError:
            self.min_level = AlertLevel.INFO
            
        self.enabled_types = set(self.alert_config.get('enabled_types', [t.value for t in AlertType]))
        
        # Initialize Telegram
        self.telegram_enabled = False
        self._load_credentials()
        self._initialize_telegram()
        
        logger.info("Clean Telegram-Only NotificationManager initialized")
    
    def _load_credentials(self):
        """Load Telegram credentials"""
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        # Fallback to config
        if not self.telegram_bot_token:
            self.telegram_bot_token = self.telegram_config.get('bot_token')
        if not self.telegram_chat_id:
            self.telegram_chat_id = self.telegram_config.get('chat_id')
    
    def _initialize_telegram(self):
        """Initialize Telegram notifications"""
        if self.telegram_config.get('enabled', True):  # Default enabled
            if self.telegram_bot_token and self.telegram_chat_id:
                self.telegram_enabled = True
                logger.info("Telegram notifications enabled")
            else:
                logger.warning("Telegram disabled - missing credentials")
        else:
            logger.info("Telegram notifications disabled in config")
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        with self.lock:
            now = datetime.now()
            if now - self.last_reset > self.rate_limit_window:
                self.alert_count = 0
                self.last_reset = now
            
            return self.alert_count < self.max_alerts_per_window
    
    def _increment_rate_limit(self):
        """Increment rate limit counter"""
        with self.lock:
            self.alert_count += 1
    
    def send_alert(self, message: str, level: AlertLevel = AlertLevel.INFO, 
                  alert_type: AlertType = AlertType.SYSTEM_STATUS) -> bool:
        """
        Send alert through Telegram
        """
        try:
            # Check if alert should be sent
            if not self._should_send_alert(level, alert_type):
                return False
            
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
                return self._send_telegram(message, level, alert_type)
            
            return False
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
            return False
    
    def _should_send_alert(self, level: AlertLevel, alert_type: AlertType) -> bool:
        """Check if alert should be sent"""
        # Check alert level
        level_order = {AlertLevel.INFO: 0, AlertLevel.WARNING: 1, AlertLevel.ERROR: 2, AlertLevel.CRITICAL: 3}
        if level_order[level] < level_order[self.min_level]:
            return False
        
        # Check alert type
        if alert_type.value not in self.enabled_types:
            return False
        
        # Check rate limiting (allow critical alerts through)
        if level != AlertLevel.CRITICAL and not self._check_rate_limit():
            logger.warning(f"Rate limit exceeded, dropping {level.value} alert")
            return False
        
        return True
    
    def _send_telegram(self, message: str, level: AlertLevel, alert_type: AlertType) -> bool:
        """Send alert via Telegram - simple text only"""
        try:
            # Emoji for level
            emoji_map = {
                AlertLevel.INFO: "ℹ️",
                AlertLevel.WARNING: "⚠️", 
                AlertLevel.ERROR: "❌",
                AlertLevel.CRITICAL: "🚨"
            }
            
            emoji = emoji_map.get(level, "📢")
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            # Simple text formatting (no Markdown)
            formatted_message = f"{emoji} {level.value} - {alert_type.value}\n\n{message}\n\n🕐 {timestamp}"
            
            # Send to Telegram
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': formatted_message
            }
            
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            self._increment_rate_limit()
            logger.debug("Telegram alert sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False
    
    # Convenience methods
    def send_strategy_change_alert(self, old_strategy: str, new_strategy: str, reason: str = ""):
        """Send alert for strategy change"""
        message = f"Strategy: {old_strategy} → {new_strategy}"
        if reason:
            message += f"\nReason: {reason}"
        return self.send_alert(message, AlertLevel.INFO, AlertType.STRATEGY_CHANGE)
    
    def send_market_phase_change_alert(self, old_phase: str, new_phase: str, confidence: float = None):
        """Send alert for market phase change"""
        message = f"Market Phase: {old_phase} → {new_phase}"
        if confidence:
            message += f"\nConfidence: {confidence:.1%}"
        return self.send_alert(message, AlertLevel.INFO, AlertType.MARKET_PHASE_CHANGE)
    
    def send_drawdown_alert(self, current_drawdown: float, max_drawdown: float, portfolio_value: float):
        """Send alert for significant drawdown"""
        message = f"Drawdown: {current_drawdown:.1%} (max: {max_drawdown:.1%})\nPortfolio: ${portfolio_value:,.2f}"
        
        level = AlertLevel.WARNING if current_drawdown < 0.1 else AlertLevel.ERROR
        if current_drawdown > 0.2:
            level = AlertLevel.CRITICAL
        
        return self.send_alert(message, level, AlertType.DRAWDOWN)
    
    def send_api_error_alert(self, api_name: str, error_message: str):
        """Send alert for API errors"""
        message = f"API Error: {api_name}\n{error_message}"
        return self.send_alert(message, AlertLevel.ERROR, AlertType.API_ERROR)
    
    def send_bot_crash_alert(self, error_message: str):
        """Send critical alert for bot crashes"""
        message = f"Bot crashed: {error_message}"
        return self.send_alert(message, AlertLevel.CRITICAL, AlertType.BOT_CRASH)
    
    def send_trade_alert(self, action: str, symbol: str, quantity: float, price: float, strategy: str):
        """Send alert for trade execution"""
        message = f"Trade: {action} {quantity} {symbol}\nPrice: ${price:.4f}\nStrategy: {strategy}"
        return self.send_alert(message, AlertLevel.INFO, AlertType.TRADE_EXECUTED)
    
    def send_portfolio_update_alert(self, total_value: float, pnl_24h: float):
        """Send periodic portfolio update"""
        message = f"Portfolio: ${total_value:,.2f}\n24h P&L: {pnl_24h:+.2f}"
        return self.send_alert(message, AlertLevel.INFO, AlertType.PORTFOLIO_UPDATE)
    
    def test_notification(self) -> bool:
        """Test Telegram notification"""
        if self.telegram_enabled:
            return self.send_alert("Test notification - Clean Telegram-Only system working!", AlertLevel.INFO, AlertType.SYSTEM_STATUS)
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
    """Simple interface to send alerts"""
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