#!/usr/bin/env python3
"""
Test Script für Telegram Integration
"""

import os
import sys
from utils.notifier import NotificationManager, AlertLevel, AlertType

def test_telegram_integration():
    """Testet die Telegram Integration"""
    print("🧪 Testing Telegram Integration...")
    
    # Check environment variables
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    print(f"\n📋 Environment Check:")
    print(f"   Bot Token: {'✅ Set' if bot_token else '❌ Missing'}")
    print(f"   Chat ID: {'✅ Set' if chat_id else '❌ Missing'}")
    
    if not bot_token or not chat_id:
        print("\n❌ Telegram Credentials fehlen!")
        print("Bitte führen Sie zuerst das Setup aus:")
        print("   python3 setup_telegram.py")
        print("   source .env_telegram")
        return False
    
    # Test notification system
    settings = {
        'notifications': {
            'telegram': {
                'enabled': True
            },
            'email': {
                'enabled': False
            },
            'alerts': {
                'min_level': 'INFO',
                'enabled_types': ['SYSTEM_STATUS', 'STRATEGY_CHANGE', 'MARKET_PHASE_CHANGE', 'DRAWDOWN']
            }
        }
    }
    
    try:
        # Initialize notifier
        notifier = NotificationManager(settings)
        
        if not notifier.telegram_enabled:
            print("❌ Telegram nicht aktiviert!")
            return False
        
        print("\n✅ NotificationManager initialisiert")
        
        # Send test alerts
        print("\n📤 Sende Test Alerts...")
        
        # Test basic alert
        notifier.send_alert(
            "🎉 Trading Bot Telegram Integration erfolgreich!", 
            AlertLevel.INFO, 
            AlertType.SYSTEM_STATUS
        )
        
        # Test strategy change
        notifier.send_strategy_change_alert(
            old_strategy="momentum",
            new_strategy="arbitrage", 
            reason="Market conditions changed"
        )
        
        # Test market phase change
        notifier.send_market_phase_change_alert(
            old_phase="bull",
            new_phase="sideways",
            confidence=0.78
        )
        
        # Test drawdown alert
        notifier.send_drawdown_alert(
            current_drawdown=0.08,
            max_drawdown=0.15,
            portfolio_value=125000.50
        )
        
        print("✅ Test Alerts gesendet!")
        print("\n📱 Überprüfen Sie Telegram für die Nachrichten!")
        
        # Test notification channels
        test_results = notifier.test_notifications()
        print(f"\n🧪 Channel Tests:")
        for channel, success in test_results.items():
            status = "✅" if success else "❌"
            print(f"   {channel}: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fehler beim Test: {e}")
        return False

if __name__ == "__main__":
    success = test_telegram_integration()
    if success:
        print("\n🎉 Telegram Integration erfolgreich!")
        print("Ihr Trading Bot kann jetzt Alerts senden!")
    else:
        print("\n❌ Integration fehlgeschlagen!")
        sys.exit(1)