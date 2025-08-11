#!/usr/bin/env python3
"""
Ready-to-use Telegram Integration für Ihren Trading Bot
Kopieren Sie diese Klasse in Ihren Trading Bot Code!
"""

import requests
from datetime import datetime
from enum import Enum

class AlertLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING" 
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class TradingBotNotifier:
    """
    Ready-to-use Telegram Notifier für Trading Bot
    
    Usage:
        notifier = TradingBotNotifier()
        notifier.send_alert("Portfolio value: $125,000")
        notifier.strategy_change("momentum", "arbitrage", "market volatility")
        notifier.market_phase_change("bull", "sideways", 0.85)
    """
    
    def __init__(self):
        # Ihre Bot Credentials (bereits konfiguriert)
        self.bot_token = "8153474335:AAFGE6YOGfUTYJbcYvGynKqb3ApoxdMVCds"
        self.chat_id = "6942445141"
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        
        # Alert Einstellungen
        self.min_level = AlertLevel.INFO  # Minimum alert level
        self.enabled = True
        
        print("✅ TradingBotNotifier initialisiert")
    
    def send_message(self, text):
        """Sendet eine einfache Nachricht"""
        if not self.enabled:
            return False
            
        url = f"{self.base_url}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': text
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data['ok']
        except:
            return False
    
    def send_alert(self, message, level=AlertLevel.INFO):
        """Sendet einen Alert mit Level"""
        if not self.enabled:
            return False
            
        # Level Check
        level_order = {AlertLevel.INFO: 0, AlertLevel.WARNING: 1, AlertLevel.ERROR: 2, AlertLevel.CRITICAL: 3}
        if level_order[level] < level_order[self.min_level]:
            return False
        
        # Emoji für Level
        emoji_map = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌", 
            AlertLevel.CRITICAL: "🚨"
        }
        
        emoji = emoji_map[level]
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted_message = f"{emoji} {level.value}: {message}\n🕐 {timestamp}"
        
        return self.send_message(formatted_message)
    
    def strategy_change(self, old_strategy, new_strategy, reason=""):
        """Alert für Strategy Change"""
        message = f"Strategy: {old_strategy} → {new_strategy}"
        if reason:
            message += f"\nReason: {reason}"
        return self.send_alert(message, AlertLevel.INFO)
    
    def market_phase_change(self, old_phase, new_phase, confidence=None):
        """Alert für Market Phase Change"""
        message = f"Market Phase: {old_phase} → {new_phase}"
        if confidence:
            message += f"\nConfidence: {confidence:.1%}"
        return self.send_alert(message, AlertLevel.INFO)
    
    def portfolio_update(self, value, pnl_24h=None):
        """Alert für Portfolio Update"""
        message = f"Portfolio: ${value:,.2f}"
        if pnl_24h is not None:
            message += f"\n24h P&L: {pnl_24h:+.2f}"
        return self.send_alert(message, AlertLevel.INFO)
    
    def drawdown_alert(self, drawdown_pct, portfolio_value):
        """Alert für Drawdown"""
        level = AlertLevel.WARNING if drawdown_pct < 10 else AlertLevel.ERROR
        if drawdown_pct > 20:
            level = AlertLevel.CRITICAL
            
        message = f"Drawdown: {drawdown_pct:.1f}%\nPortfolio: ${portfolio_value:,.2f}"
        return self.send_alert(message, level)
    
    def trade_executed(self, action, symbol, quantity, price):
        """Alert für Trade Execution"""
        message = f"Trade: {action} {quantity} {symbol}\nPrice: ${price:.4f}"
        return self.send_alert(message, AlertLevel.INFO)
    
    def api_error(self, exchange, error_msg):
        """Alert für API Errors"""
        message = f"API Error: {exchange}\n{error_msg}"
        return self.send_alert(message, AlertLevel.ERROR)
    
    def bot_started(self):
        """Alert beim Bot Start"""
        message = "Trading Bot started successfully!"
        return self.send_alert(message, AlertLevel.INFO)
    
    def bot_stopped(self, reason=""):
        """Alert beim Bot Stop"""
        message = "Trading Bot stopped"
        if reason:
            message += f"\nReason: {reason}"
        return self.send_alert(message, AlertLevel.WARNING)
    
    def critical_error(self, error_msg):
        """Alert für kritische Fehler"""
        message = f"CRITICAL ERROR: {error_msg}"
        return self.send_alert(message, AlertLevel.CRITICAL)

# Beispiel-Usage
def demo_usage():
    """Zeigt wie der Notifier verwendet wird"""
    print("🎯 Demo: Trading Bot Notifier Usage")
    print("=" * 40)
    
    # Notifier initialisieren
    notifier = TradingBotNotifier()
    
    # Verschiedene Alerts senden
    print("\n📤 Sende Demo Alerts...")
    
    # Bot started
    notifier.bot_started()
    
    # Portfolio update
    notifier.portfolio_update(125000.50, 2500.75)
    
    # Strategy change
    notifier.strategy_change("momentum", "arbitrage", "High volatility detected")
    
    # Market phase change
    notifier.market_phase_change("bull", "sideways", 0.78)
    
    # Trade executed
    notifier.trade_executed("BUY", "BTC/USDT", 0.1, 45250.00)
    
    # Drawdown alert
    notifier.drawdown_alert(8.5, 115000.00)
    
    # API error
    notifier.api_error("Binance", "Rate limit exceeded")
    
    print("✅ Demo Alerts gesendet!")
    print("📱 Überprüfen Sie Telegram!")

if __name__ == "__main__":
    demo_usage()