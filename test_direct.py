#!/usr/bin/env python3
"""
Direkter Test der Telegram Integration
"""

import os
import requests
from utils.notifier import NotificationManager, AlertLevel, AlertType

def test_direct():
    # Set environment variables directly
    os.environ['TELEGRAM_BOT_TOKEN'] = "8153474335:AAFGE6YOGfUTYJbcYvGynKqb3ApoxdMVCds"
    os.environ['TELEGRAM_CHAT_ID'] = "6942445141"
    
    print("🧪 Direkter Telegram Test")
    print("=" * 30)
    
    # Test direct API call first
    bot_token = os.environ['TELEGRAM_BOT_TOKEN']
    chat_id = os.environ['TELEGRAM_CHAT_ID']
    
    print(f"Bot Token: {bot_token[:15]}...")
    print(f"Chat ID: {chat_id}")
    
    # Direct API test
    print("\n📤 Teste direkte API...")
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': '🧪 *Direkter API Test*\n\nDies ist ein direkter Test der Telegram API.\n\n✅ Bot Token funktioniert\n✅ Chat ID funktioniert\n✅ API Verbindung erfolgreich',
        'parse_mode': 'Markdown'
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data['ok']:
            print("✅ Direkte API erfolgreich!")
        else:
            print(f"❌ API Fehler: {data}")
            return False
            
    except Exception as e:
        print(f"❌ API Fehler: {e}")
        return False
    
    # Test NotificationManager
    print("\n🔧 Teste NotificationManager...")
    
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
                'enabled_types': ['SYSTEM_STATUS', 'STRATEGY_CHANGE', 'MARKET_PHASE_CHANGE']
            }
        }
    }
    
    try:
        notifier = NotificationManager(settings)
        
        print(f"Telegram enabled: {notifier.telegram_enabled}")
        print(f"Email enabled: {notifier.email_enabled}")
        
        if notifier.telegram_enabled:
            print("\n📨 Sende Test Alerts...")
            
            # Test verschiedene Alert-Typen
            notifier.send_alert(
                "🎉 NotificationManager Test erfolgreich!",
                AlertLevel.INFO,
                AlertType.SYSTEM_STATUS
            )
            
            notifier.send_strategy_change_alert(
                "momentum", "arbitrage", "Test der Strategy Change Alerts"
            )
            
            notifier.send_market_phase_change_alert(
                "bull", "sideways", 0.76
            )
            
            print("✅ Alle Test-Alerts gesendet!")
            print("📱 Überprüfen Sie Telegram für die Nachrichten!")
            
            return True
        else:
            print("❌ Telegram nicht aktiviert!")
            return False
            
    except Exception as e:
        print(f"❌ NotificationManager Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_direct()
    if success:
        print("\n🎉 Telegram Integration vollständig funktionsfähig!")
        print("Ihr Trading Bot kann jetzt Alerts senden!")
    else:
        print("\n❌ Test fehlgeschlagen!")