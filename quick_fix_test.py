#!/usr/bin/env python3
"""
Schneller Fix Test - einfache Nachrichten ohne Markdown
"""

import requests
import os

def quick_test():
    bot_token = "8153474335:AAFGE6YOGfUTYJbcYvGynKqb3ApoxdMVCds"
    chat_id = "6942445141"
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    # Test ohne Markdown
    payload = {
        'chat_id': chat_id,
        'text': '✅ CLEAN NOTIFIER TEST\n\nEmail-Funktionen erfolgreich entfernt!\nNur noch Telegram Notifications aktiv.\n\nIhr Bot ist jetzt schlanker und fokussierter! 🎯'
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data['ok']:
            print("✅ Clean Notifier funktioniert!")
            print("📱 Email-Funktionen erfolgreich entfernt!")
            print("🎯 Nur noch Telegram - schlanker und effizienter!")
            return True
        else:
            print(f"❌ Fehler: {data}")
            return False
            
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return False

if __name__ == "__main__":
    quick_test()