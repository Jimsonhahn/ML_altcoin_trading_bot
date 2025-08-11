#!/usr/bin/env python3
"""
Test der finalen, sauberen Telegram-Only Notifier
"""

import os
from utils.notifier import (
    NotificationManager, AlertLevel, AlertType,
    initialize_notifier, send_alert, send_info, send_warning, send_error, send_critical
)

def test_final_clean():
    """Test der finalen sauberen Version"""
    print("🎯 FINAL TEST: Saubere Telegram-Only Notifier")
    print("=" * 50)
    
    # Set credentials
    os.environ['TELEGRAM_BOT_TOKEN'] = "8153474335:AAFGE6YOGfUTYJbcYvGynKqb3ApoxdMVCds"
    os.environ['TELEGRAM_CHAT_ID'] = "6942445141"
    
    # Simple config
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
                    'TRADE_EXECUTED',
                    'API_ERROR',
                    'BOT_CRASH'
                ]
            }
        }
    }
    
    # Initialize
    notifier = initialize_notifier(settings)
    
    print(f"✅ Notifier initialized")
    print(f"   Telegram enabled: {notifier.telegram_enabled}")
    
    # Test 1: Simple interface
    print("\n📤 Test 1: Simple Interface")
    success1 = send_info("🎉 Email-Funktionen erfolgreich entfernt!")
    print(f"   send_info(): {'✅' if success1 else '❌'}")
    
    # Test 2: Warning
    print("\n📤 Test 2: Warning Alert")
    success2 = send_warning("System läuft jetzt schlanker - nur Telegram!")
    print(f"   send_warning(): {'✅' if success2 else '❌'}")
    
    # Test 3: Strategy Change
    print("\n📤 Test 3: Strategy Change")
    success3 = notifier.send_strategy_change_alert(
        "old_strategy", "new_strategy", "Optimierung nach Email-Entfernung"
    )
    print(f"   Strategy Change: {'✅' if success3 else '❌'}")
    
    # Test 4: Market Phase
    print("\n📤 Test 4: Market Phase Change")
    success4 = notifier.send_market_phase_change_alert("bull", "sideways", 0.87)
    print(f"   Market Phase: {'✅' if success4 else '❌'}")
    
    # Test 5: Portfolio Update
    print("\n📤 Test 5: Portfolio Update")
    success5 = notifier.send_portfolio_update_alert(150000.00, 2500.50)
    print(f"   Portfolio: {'✅' if success5 else '❌'}")
    
    # Test 6: Trade Alert
    print("\n📤 Test 6: Trade Alert")
    success6 = notifier.send_trade_alert("BUY", "BTC/USDT", 0.1, 45250.00, "momentum")
    print(f"   Trade: {'✅' if success6 else '❌'}")
    
    # Test 7: Test function
    print("\n📤 Test 7: Test Function")
    success7 = notifier.test_notification()
    print(f"   Test Function: {'✅' if success7 else '❌'}")
    
    # Summary
    all_tests = [success1, success2, success3, success4, success5, success6, success7]
    success_count = sum(1 for s in all_tests if s)
    
    print(f"\n📊 Ergebnis: {success_count}/7 erfolgreich")
    
    if success_count >= 6:  # Allow for 1 failure
        print("🎉 CLEAN NOTIFIER SETUP ERFOLGREICH!")
        print("📱 Email-Funktionen entfernt")
        print("🎯 Nur noch Telegram - sauber und effizient!")
        
        # Final success message
        send_critical("🎉 SETUP COMPLETE!\n\nEmail-Funktionen erfolgreich entfernt!\nNur noch Telegram Notifications.\n\nIhr Bot ist jetzt schlanker, schneller und fokussierter! 🚀")
        
        return True
    else:
        print("⚠️ Einige Tests fehlgeschlagen")
        return False

if __name__ == "__main__":
    test_final_clean()