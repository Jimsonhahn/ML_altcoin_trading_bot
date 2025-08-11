#!/usr/bin/env python3
"""
Test der bereinigten Telegram-Only Notifier
"""

import os
from utils.notifier import (
    NotificationManager, AlertLevel, AlertType,
    initialize_notifier, send_alert, send_info, send_warning, send_error, send_critical
)

def test_clean_notifier():
    """Test der bereinigten Notifier ohne Email"""
    print("🧹 Test: Bereinigte Telegram-Only Notifier")
    print("=" * 45)
    
    # Set credentials
    os.environ['TELEGRAM_BOT_TOKEN'] = "8153474335:AAFGE6YOGfUTYJbcYvGynKqb3ApoxdMVCds"
    os.environ['TELEGRAM_CHAT_ID'] = "6942445141"
    
    # Configuration (nur Telegram)
    settings = {
        'notifications': {
            'telegram': {
                'enabled': True
            },
            'alerts': {
                'min_level': 'INFO',
                'enabled_types': [
                    'SYSTEM_STATUS', 
                    'STRATEGY_CHANGE', 
                    'MARKET_PHASE_CHANGE',
                    'DRAWDOWN',
                    'TRADE_EXECUTED'
                ]
            }
        }
    }
    
    # Initialize notifier
    notifier = initialize_notifier(settings)
    
    print(f"✅ Notifier initialisiert")
    print(f"   Telegram enabled: {notifier.telegram_enabled}")
    print(f"   Alert history: {len(notifier.alert_history)} alerts")
    
    # Test simple interface
    print("\n📤 Teste Simple Interface...")
    
    success1 = send_info("📈 Portfolio reached new high: $150,000")
    print(f"   send_info(): {'✅' if success1 else '❌'}")
    
    success2 = send_warning("⚠️ High volatility detected in BTC/USDT")
    print(f"   send_warning(): {'✅' if success2 else '❌'}")
    
    success3 = send_error("❌ API connection timeout - retrying...")
    print(f"   send_error(): {'✅' if success3 else '❌'}")
    
    # Test specific alert methods
    print("\n🎯 Teste Spezielle Alert-Methoden...")
    
    success4 = notifier.send_strategy_change_alert(
        "momentum", "arbitrage", "Market conditions favor arbitrage opportunities"
    )
    print(f"   Strategy Change: {'✅' if success4 else '❌'}")
    
    success5 = notifier.send_market_phase_change_alert(
        "bull", "sideways", 0.82
    )
    print(f"   Market Phase Change: {'✅' if success5 else '❌'}")
    
    success6 = notifier.send_trade_alert(
        "BUY", "ETH/USDT", 2.5, 2850.75, "arbitrage"
    )
    print(f"   Trade Alert: {'✅' if success6 else '❌'}")
    
    success7 = notifier.send_drawdown_alert(0.085, 0.15, 137500.50)
    print(f"   Drawdown Alert: {'✅' if success7 else '❌'}")
    
    # Test notification
    print("\n🧪 Teste Notification Function...")
    success8 = notifier.test_notification()
    print(f"   Test Notification: {'✅' if success8 else '❌'}")
    
    # Final summary
    all_success = all([success1, success2, success3, success4, success5, success6, success7, success8])
    
    print(f"\n📊 Ergebnis: {sum([success1, success2, success3, success4, success5, success6, success7, success8])}/8 erfolgreich")
    
    if all_success:
        print("🎉 ALLE TESTS ERFOLGREICH!")
        print("📱 Überprüfen Sie Telegram für die Nachrichten!")
        
        # Send success confirmation
        send_critical("🎉 CLEAN NOTIFIER SETUP COMPLETE! Alle Email-Funktionen entfernt, nur Telegram aktiv!")
        
    else:
        print("⚠️ Einige Tests fehlgeschlagen, aber Grundfunktion arbeitet")
    
    # Show alert history
    history = notifier.get_alert_history(1)
    print(f"\n📋 Alert History: {len(history)} alerts in der letzten Stunde")
    
    return all_success

if __name__ == "__main__":
    test_clean_notifier()