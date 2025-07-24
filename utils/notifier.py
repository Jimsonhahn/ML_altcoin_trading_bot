"""
Notifier - Konsolidierte Version für Benachrichtigungen
Saubere Telegram-Integration mit Error-Handling
"""

import asyncio
import logging
import os
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from enum import Enum
from dataclasses import dataclass, asdict

import aiohttp

from config.settings import Settings

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alarm-Level für verschiedene Benachrichtigungstypen"""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Alarm-Typen für Kategorisierung"""
    TRADE_EXECUTED = "trade_executed"
    TRADE_REJECTED = "trade_rejected"
    SYSTEM_ERROR = "system_error"
    PRICE_ALERT = "price_alert"
    PERFORMANCE_UPDATE = "performance_update"
    REGIME_CHANGE = "regime_change"
    SAFETY_TRIGGER = "safety_trigger"
    BOT_STATUS = "bot_status"


@dataclass
class Alert:
    """Strukturierte Alarm-Nachricht"""
    level: AlertLevel
    alert_type: AlertType
    title: str
    message: str
    timestamp: datetime
    symbol: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TelegramNotifier:
    """
    Telegram-Notifier für Trading Bot Benachrichtigungen
    """
    
    def __init__(self, bot_token: str, chat_id: str, settings: Settings):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.settings = settings
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
        # Rate Limiting
        self.last_message_time = {}
        self.rate_limit_window = timedelta(seconds=30)  # 30 Sekunden zwischen gleichen Nachrichten
        
        # Message Queue für Batch-Nachrichten
        self.message_queue = []
        self.batch_size = settings.get('notifications.batch_size', 5)
        self.batch_timeout = timedelta(seconds=settings.get('notifications.batch_timeout', 60))
        
        # Aktivierte Benachrichtigungstypen
        self.enabled_alerts = set(settings.get('notifications.enabled_alerts', [
            AlertType.TRADE_EXECUTED.value,
            AlertType.SYSTEM_ERROR.value,
            AlertType.SAFETY_TRIGGER.value,
            AlertType.BOT_STATUS.value
        ]))
        
        # Mindest-Level für Benachrichtigungen
        self.min_level = AlertLevel(settings.get('notifications.min_level', AlertLevel.INFO.value))
        
        logger.info(f"Telegram Notifier initialisiert für Chat {chat_id}")
    
    async def send_alert(self, alert: Alert) -> bool:
        """
        Sendet eine strukturierte Alarm-Nachricht
        
        Args:
            alert: Alert-Objekt mit allen Informationen
            
        Returns:
            True wenn erfolgreich gesendet, False sonst
        """
        # Prüfe ob Alert-Typ aktiviert ist
        if alert.alert_type.value not in self.enabled_alerts:
            logger.debug(f"Alert-Typ {alert.alert_type.value} ist deaktiviert")
            return False
        
        # Prüfe Mindest-Level
        if not self._should_send_alert(alert.level):
            logger.debug(f"Alert-Level {alert.level.value} unter Mindest-Level")
            return False
        
        # Rate Limiting prüfen
        if self._is_rate_limited(alert):
            logger.debug(f"Alert ist rate-limited: {alert.title}")
            return False
        
        # Nachricht formatieren
        message = self._format_alert_message(alert)
        
        # Senden
        success = await self._send_telegram_message(message)
        
        if success:
            self._update_rate_limit(alert)
            logger.debug(f"Alert erfolgreich gesendet: {alert.title}")
        else:
            logger.warning(f"Fehler beim Senden des Alerts: {alert.title}")
        
        return success
    
    def _should_send_alert(self, level: AlertLevel) -> bool:
        """Prüft ob Alert basierend auf Level gesendet werden soll"""
        level_hierarchy = {
            AlertLevel.INFO: 0,
            AlertLevel.WARNING: 1,
            AlertLevel.ERROR: 2,
            AlertLevel.CRITICAL: 3
        }
        
        return level_hierarchy[level] >= level_hierarchy[self.min_level]
    
    def _is_rate_limited(self, alert: Alert) -> bool:
        """Prüft Rate Limiting für ähnliche Nachrichten"""
        # Rate Limit Key basierend auf Alert-Typ und Titel
        rate_key = f"{alert.alert_type.value}:{alert.title[:50]}"
        
        last_time = self.last_message_time.get(rate_key)
        if last_time is None:
            return False
        
        return datetime.now() - last_time < self.rate_limit_window
    
    def _update_rate_limit(self, alert: Alert):
        """Aktualisiert Rate Limit Timing"""
        rate_key = f"{alert.alert_type.value}:{alert.title[:50]}"
        self.last_message_time[rate_key] = datetime.now()
    
    def _format_alert_message(self, alert: Alert) -> str:
        """Formatiert Alert-Nachricht für Telegram"""
        # Emoji basierend auf Level
        level_emojis = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨"
        }
        
        # Emoji basierend auf Alert-Typ
        type_emojis = {
            AlertType.TRADE_EXECUTED: "💰",
            AlertType.TRADE_REJECTED: "🚫",
            AlertType.SYSTEM_ERROR: "🔧",
            AlertType.PRICE_ALERT: "📈",
            AlertType.PERFORMANCE_UPDATE: "📊",
            AlertType.REGIME_CHANGE: "🔄",
            AlertType.SAFETY_TRIGGER: "🛡️",
            AlertType.BOT_STATUS: "🤖"
        }
        
        level_emoji = level_emojis.get(alert.level, "")
        type_emoji = type_emojis.get(alert.alert_type, "")
        
        # Header
        header = f"{level_emoji} {type_emoji} *{alert.title}*"
        
        # Timestamp
        time_str = alert.timestamp.strftime("%H:%M:%S")
        
        # Symbol falls vorhanden
        symbol_str = f" ({alert.symbol})" if alert.symbol else ""
        
        # Basis-Nachricht
        message_parts = [
            header,
            f"⏰ {time_str}{symbol_str}",
            "",
            alert.message
        ]
        
        # Metadata falls vorhanden
        if alert.metadata:
            message_parts.append("")
            message_parts.append("📋 *Details:*")
            for key, value in alert.metadata.items():
                if isinstance(value, (int, float)):
                    if key.endswith('_percent') or 'percent' in key.lower():
                        message_parts.append(f"• {key}: {value:.2%}")
                    elif isinstance(value, float):
                        message_parts.append(f"• {key}: {value:.4f}")
                    else:
                        message_parts.append(f"• {key}: {value}")
                else:
                    message_parts.append(f"• {key}: {value}")
        
        return "\n".join(message_parts)
    
    async def _send_telegram_message(self, message: str) -> bool:
        """Sendet Nachricht über Telegram API"""
        url = f"{self.base_url}/sendMessage"
        
        payload = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': 'Markdown',
            'disable_web_page_preview': True
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as response:
                    if response.status == 200:
                        return True
                    else:
                        response_text = await response.text()
                        logger.error(f"Telegram API Fehler {response.status}: {response_text}")
                        return False
                        
        except asyncio.TimeoutError:
            logger.error("Timeout beim Senden der Telegram-Nachricht")
            return False
        except Exception as e:
            logger.error(f"Fehler beim Senden der Telegram-Nachricht: {e}")
            return False
    
    async def send_batch_summary(self, alerts: List[Alert]) -> bool:
        """Sendet eine Zusammenfassung mehrerer Alerts"""
        if not alerts:
            return True
        
        # Gruppiere Alerts nach Level
        level_groups = {}
        for alert in alerts:
            level = alert.level
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(alert)
        
        # Erstelle Zusammenfassung
        summary_parts = ["📊 *Alert-Zusammenfassung*", ""]
        
        for level, level_alerts in level_groups.items():
            level_emojis = {
                AlertLevel.INFO: "ℹ️",
                AlertLevel.WARNING: "⚠️", 
                AlertLevel.ERROR: "❌",
                AlertLevel.CRITICAL: "🚨"
            }
            
            emoji = level_emojis.get(level, "")
            summary_parts.append(f"{emoji} *{level.value.upper()}* ({len(level_alerts)})")
            
            for alert in level_alerts[:3]:  # Nur ersten 3 zeigen
                summary_parts.append(f"• {alert.title}")
            
            if len(level_alerts) > 3:
                summary_parts.append(f"• ... und {len(level_alerts) - 3} weitere")
            
            summary_parts.append("")
        
        # Zeitraum
        first_time = min(alert.timestamp for alert in alerts)
        last_time = max(alert.timestamp for alert in alerts)
        summary_parts.append(f"⏰ {first_time.strftime('%H:%M')} - {last_time.strftime('%H:%M')}")
        
        message = "\n".join(summary_parts)
        return await self._send_telegram_message(message)


class Notifier:
    """
    Hauptklasse für alle Benachrichtigungen
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.telegram_notifier = None
        
        # Telegram-Notifier initialisieren wenn konfiguriert
        self._init_telegram_notifier()
        
        # Alert-Queue für Batch-Processing
        self.alert_queue = []
        self.last_batch_time = datetime.now()
        
        logger.info("Notifier initialisiert")
    
    def _init_telegram_notifier(self):
        """Initialisiert Telegram-Notifier falls konfiguriert"""
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if bot_token and chat_id:
            try:
                self.telegram_notifier = TelegramNotifier(bot_token, chat_id, self.settings)
                logger.info("Telegram-Notifier erfolgreich initialisiert")
            except Exception as e:
                logger.error(f"Fehler beim Initialisieren des Telegram-Notifiers: {e}")
        else:
            logger.warning("Telegram-Konfiguration fehlt - Benachrichtigungen deaktiviert")
    
    async def send_notification(
        self,
        message: str,
        alert_type: str,
        level: str = AlertLevel.INFO.value,
        symbol: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Vereinfachte Methode zum Senden von Benachrichtigungen
        
        Args:
            message: Nachrichtentext
            alert_type: Typ der Benachrichtigung
            level: Alert-Level
            symbol: Optionales Symbol
            metadata: Optionale Metadaten
            
        Returns:
            True wenn erfolgreich gesendet
        """
        try:
            alert = Alert(
                level=AlertLevel(level),
                alert_type=AlertType(alert_type),
                title=alert_type.replace('_', ' ').title(),
                message=message,
                timestamp=datetime.now(),
                symbol=symbol,
                metadata=metadata
            )
            
            return await self.send_alert(alert)
            
        except ValueError as e:
            logger.error(f"Ungültiger Alert-Level oder -Typ: {e}")
            return False
        except Exception as e:
            logger.error(f"Fehler beim Senden der Benachrichtigung: {e}")
            return False
    
    async def send_alert(self, alert: Alert) -> bool:
        """
        Sendet einen strukturierten Alert
        
        Args:
            alert: Alert-Objekt
            
        Returns:
            True wenn erfolgreich gesendet
        """
        if not self.telegram_notifier:
            logger.debug("Kein Telegram-Notifier verfügbar")
            return False
        
        return await self.telegram_notifier.send_alert(alert)
    
    async def send_trade_notification(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        strategy: str = "unknown"
    ) -> bool:
        """
        Sendet Trade-Benachrichtigung
        
        Args:
            symbol: Trading-Symbol
            side: buy/sell
            amount: Menge
            price: Preis
            strategy: Verwendete Strategie
        """
        message = f"{side.upper()} {amount:.4f} {symbol} @ {price:.4f}"
        
        metadata = {
            'side': side,
            'amount': amount,
            'price': price,
            'strategy': strategy,
            'value': amount * price
        }
        
        return await self.send_notification(
            message=message,
            alert_type=AlertType.TRADE_EXECUTED.value,
            level=AlertLevel.INFO.value,
            symbol=symbol,
            metadata=metadata
        )
    
    async def send_error_notification(
        self,
        error_message: str,
        error_type: str = "system_error",
        symbol: Optional[str] = None
    ) -> bool:
        """
        Sendet Fehler-Benachrichtigung
        
        Args:
            error_message: Fehlermeldung
            error_type: Typ des Fehlers
            symbol: Optionales Symbol
        """
        return await self.send_notification(
            message=error_message,
            alert_type=AlertType.SYSTEM_ERROR.value,
            level=AlertLevel.ERROR.value,
            symbol=symbol,
            metadata={'error_type': error_type}
        )
    
    async def send_performance_update(
        self,
        total_return: float,
        daily_return: float,
        drawdown: float,
        total_trades: int
    ) -> bool:
        """
        Sendet Performance-Update
        
        Args:
            total_return: Gesamtrendite
            daily_return: Tägliche Rendite
            drawdown: Aktueller Drawdown
            total_trades: Anzahl Trades
        """
        message = f"Performance-Update: {total_return:.2%} total, {daily_return:.2%} heute"
        
        metadata = {
            'total_return_percent': total_return,
            'daily_return_percent': daily_return,
            'drawdown_percent': drawdown,
            'total_trades': total_trades
        }
        
        return await self.send_notification(
            message=message,
            alert_type=AlertType.PERFORMANCE_UPDATE.value,
            level=AlertLevel.INFO.value,
            metadata=metadata
        )
    
    async def send_safety_alert(
        self,
        alert_message: str,
        drawdown: float,
        action_taken: str
    ) -> bool:
        """
        Sendet Safety-Alert
        
        Args:
            alert_message: Alert-Nachricht
            drawdown: Aktueller Drawdown
            action_taken: Ergriffene Maßnahme
        """
        metadata = {
            'drawdown_percent': drawdown,
            'action_taken': action_taken
        }
        
        return await self.send_notification(
            message=alert_message,
            alert_type=AlertType.SAFETY_TRIGGER.value,
            level=AlertLevel.WARNING.value,
            metadata=metadata
        )
    
    def is_enabled(self) -> bool:
        """Prüft ob Notifier aktiviert ist"""
        return self.telegram_notifier is not None