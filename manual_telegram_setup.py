#!/usr/bin/env python3
"""
Manuelles Telegram Setup mit Ihrem Bot Token
"""

import requests
import json
import os

def get_chat_id_manual(bot_token):
    """Findet die Chat ID für den Bot"""
    print("\n🔍 Suche nach Chat ID...")
    print("📱 WICHTIG: Bitte führen Sie diese Schritte aus:")
    print("   1. Öffnen Sie Telegram")
    print("   2. Suchen Sie nach Ihrem Bot (der Username den Sie erstellt haben)")
    print("   3. Starten Sie einen Chat mit dem Bot")
    print("   4. Senden Sie eine Nachricht wie: 'Hallo Bot' oder '/start'")
    print()
    print("🔄 Ich suche jetzt nach Ihrer Nachricht...")
    
    # Get updates from Telegram
    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data['ok']:
            print(f"❌ API Fehler: {data.get('description', 'Unbekannter Fehler')}")
            return None
        
        updates = data['result']
        if not updates:
            print("❌ Keine Nachrichten gefunden!")
            print("   Stellen Sie sicher, dass Sie eine Nachricht an den Bot gesendet haben.")
            print("   Dann führen Sie dieses Script nochmal aus.")
            return None
        
        print(f"✅ {len(updates)} Nachricht(en) gefunden!")
        
        # Get the most recent chat ID
        latest_update = updates[-1]
        chat_id = latest_update['message']['chat']['id']
        chat_type = latest_update['message']['chat']['type']
        
        if 'first_name' in latest_update['message']['chat']:
            name = latest_update['message']['chat']['first_name']
            print(f"✅ Chat ID gefunden: {chat_id}")
            print(f"   Name: {name}")
            print(f"   Type: {chat_type}")
        else:
            print(f"✅ Chat ID gefunden: {chat_id}")
            print(f"   Type: {chat_type}")
        
        return str(chat_id)
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Netzwerk Fehler: {e}")
        return None
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return None

def test_bot_manual(bot_token, chat_id):
    """Testet den Bot mit einer Nachricht"""
    print("\n🧪 Teste Bot...")
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': '🎉 *Trading Bot Setup erfolgreich!*\n\nIhr Telegram Bot ist jetzt konfiguriert und ready für Alerts!\n\n✅ Bot Token: Gültig\n✅ Chat ID: Gefunden\n✅ Verbindung: Funktioniert\n\nSie erhalten ab jetzt Trading Alerts hier! 📈',
        'parse_mode': 'Markdown'
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data['ok']:
            print("✅ Test erfolgreich! Sie sollten eine Nachricht erhalten haben.")
            return True
        else:
            print(f"❌ Test fehlgeschlagen: {data.get('description')}")
            return False
            
    except Exception as e:
        print(f"❌ Test Fehler: {e}")
        return False

def save_credentials_manual(bot_token, chat_id):
    """Speichert die Credentials"""
    print("\n💾 Speichere Credentials...")
    
    # Create environment file
    env_content = f"""# Telegram Bot Credentials für Trading Bot
export TELEGRAM_BOT_TOKEN="{bot_token}"
export TELEGRAM_CHAT_ID="{chat_id}"

# Optional: Email Credentials
# export EMAIL_USERNAME="your-email@example.com"
# export EMAIL_PASSWORD="your-app-password"
"""
    
    try:
        with open(".env_telegram", 'w') as f:
            f.write(env_content)
        
        print(f"✅ Credentials gespeichert in: .env_telegram")
        
        # Create shell script
        script_content = f"""#!/bin/bash
# Telegram Bot Environment Setup
export TELEGRAM_BOT_TOKEN="{bot_token}"
export TELEGRAM_CHAT_ID="{chat_id}"

echo "✅ Telegram Bot Umgebungsvariablen gesetzt"
echo "Bot Token: {bot_token[:15]}..."
echo "Chat ID: {chat_id}"
"""
        
        with open("setup_telegram_env.sh", 'w') as f:
            f.write(script_content)
        
        os.chmod("setup_telegram_env.sh", 0o755)
        print(f"✅ Setup Script erstellt: setup_telegram_env.sh")
        
        return True
        
    except Exception as e:
        print(f"❌ Fehler beim Speichern: {e}")
        return False

def main():
    bot_token = "8153474335:AAFGE6YOGfUTYJbcYvGynKqb3ApoxdMVCds"
    
    print("🤖 Trading Bot - Telegram Setup")
    print("=" * 40)
    print(f"✅ Bot Token: {bot_token[:15]}...")
    
    # Get chat ID
    chat_id = get_chat_id_manual(bot_token)
    
    if not chat_id:
        print("\n❌ Setup fehlgeschlagen!")
        print("\n🔄 Bitte:")
        print("1. Gehen Sie zu Telegram")
        print("2. Suchen Sie nach Ihrem Bot")
        print("3. Senden Sie eine Nachricht")
        print("4. Führen Sie dieses Script nochmal aus")
        return False
    
    # Test the bot
    if test_bot_manual(bot_token, chat_id):
        # Save credentials
        if save_credentials_manual(bot_token, chat_id):
            print("\n🎉 Setup erfolgreich abgeschlossen!")
            print("\n📋 Nächste Schritte:")
            print("1. source .env_telegram")
            print("2. python3 test_telegram.py")
            print("\n📱 Ihr Bot sendet jetzt Alerts an Telegram!")
            return True
    
    print("\n❌ Setup fehlgeschlagen!")
    return False

if __name__ == "__main__":
    main()