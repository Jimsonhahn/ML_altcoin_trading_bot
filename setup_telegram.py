#!/usr/bin/env python3
"""
Telegram Bot Setup Helper
Hilft beim Einrichten der Telegram Integration
"""

import requests
import json
import sys
import os

def get_chat_id(bot_token):
    """Findet die Chat ID für den Bot"""
    print("\n🔍 Suche nach Chat ID...")
    print("📱 Bitte senden Sie eine Nachricht an Ihren Bot in Telegram!")
    print("   (z.B. 'Hallo Bot' oder '/start')")
    input("   Drücken Sie Enter, wenn Sie die Nachricht gesendet haben...")
    
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
            return None
        
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

def test_bot(bot_token, chat_id):
    """Testet den Bot mit einer Nachricht"""
    print("\n🧪 Teste Bot...")
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': '🎉 *Trading Bot Setup erfolgreich!*\n\nIhr Telegram Bot ist jetzt konfiguriert und ready für Alerts!',
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

def save_credentials(bot_token, chat_id):
    """Speichert die Credentials als Umgebungsvariablen"""
    print("\n💾 Speichere Credentials...")
    
    # Create environment file
    env_content = f"""# Telegram Bot Credentials
export TELEGRAM_BOT_TOKEN="{bot_token}"
export TELEGRAM_CHAT_ID="{chat_id}"

# Email Credentials (optional)
# export EMAIL_USERNAME="your-email@example.com"
# export EMAIL_PASSWORD="your-app-password"
"""
    
    env_file = ".env_telegram"
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print(f"✅ Credentials gespeichert in: {env_file}")
        print(f"   Führen Sie aus: source {env_file}")
        
        # Also save to a shell script for convenience
        script_content = f"""#!/bin/bash
# Telegram Bot Environment Setup
export TELEGRAM_BOT_TOKEN="{bot_token}"
export TELEGRAM_CHAT_ID="{chat_id}"

echo "✅ Telegram Bot Umgebungsvariablen gesetzt"
echo "Bot Token: {bot_token[:10]}..."
echo "Chat ID: {chat_id}"
"""
        
        script_file = "setup_telegram_env.sh"
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_file, 0o755)  # Make executable
        print(f"✅ Setup Script erstellt: {script_file}")
        
    except Exception as e:
        print(f"❌ Fehler beim Speichern: {e}")

def main():
    print("🤖 Trading Bot - Telegram Setup")
    print("=" * 40)
    
    # Get bot token
    print("\n📝 Schritt 1: Bot Token eingeben")
    bot_token = input("Bot Token (von @BotFather): ").strip()
    
    if not bot_token:
        print("❌ Bot Token ist erforderlich!")
        sys.exit(1)
    
    # Validate token format
    if not bot_token.count(':') == 1 or len(bot_token) < 20:
        print("❌ Bot Token Format scheint ungültig zu sein!")
        print("   Format sollte sein: 123456789:ABCdefGHIjklMNOpqrSTUvwxyz")
        sys.exit(1)
    
    # Get chat ID
    print(f"\n📝 Schritt 2: Chat ID ermitteln")
    chat_id = get_chat_id(bot_token)
    
    if not chat_id:
        print("\n❌ Setup fehlgeschlagen!")
        print("Mögliche Lösungen:")
        print("1. Stellen Sie sicher, dass Sie eine Nachricht an den Bot gesendet haben")
        print("2. Überprüfen Sie den Bot Token")
        print("3. Stellen Sie sicher, dass Sie Internet haben")
        sys.exit(1)
    
    # Test the bot
    if test_bot(bot_token, chat_id):
        # Save credentials
        save_credentials(bot_token, chat_id)
        
        print("\n🎉 Setup erfolgreich abgeschlossen!")
        print("\nNächste Schritte:")
        print("1. Führen Sie aus: source .env_telegram")
        print("2. Oder: ./setup_telegram_env.sh")
        print("3. Starten Sie Ihren Trading Bot")
        print("\n📱 Ihr Bot wird jetzt Alerts an Telegram senden!")
        
    else:
        print("\n❌ Setup fehlgeschlagen beim Test!")

if __name__ == "__main__":
    main()