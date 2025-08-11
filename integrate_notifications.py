#!/usr/bin/env python3
"""
Integration Script für Notifications in den Trading Bot
Zeigt wie die Notifications in bestehende Dateien integriert werden
"""

def show_integration_examples():
    """Zeigt Integrations-Beispiele"""
    
    print("🔧 Trading Bot Notification Integration")
    print("=" * 50)
    
    print("\n📝 1. In main.py hinzufügen:")
    print("""
# Am Anfang der Datei
from utils.notifier import initialize_notifier, send_alert, AlertLevel, AlertType

# In der main() Funktion
def main():
    # ... bestehender Code ...
    
    # Notifications initialisieren
    notifier = initialize_notifier(settings)
    send_alert("Trading Bot gestartet", AlertLevel.INFO, AlertType.SYSTEM_STATUS)
    
    try:
        # ... Bot Logic ...
        
    except Exception as e:
        # Critical Alert bei Crash
        from utils.notifier import send_critical
        send_critical(f"Bot Crash: {str(e)}")
        raise
""")
    
    print("\n📝 2. In trading_bot.py für Strategy Changes:")
    print("""
# Import hinzufügen
from utils.notifier import get_notifier

class TradingBot:
    def __init__(self, ...):
        # ... bestehender Code ...
        self.notifier = get_notifier()
        self.current_strategy = None
    
    def switch_strategy(self, new_strategy, reason=""):
        old_strategy = self.current_strategy
        self.current_strategy = new_strategy
        
        # Notification senden
        if self.notifier and old_strategy:
            self.notifier.send_strategy_change_alert(
                old_strategy=old_strategy,
                new_strategy=new_strategy,
                reason=reason
            )
""")
    
    print("\n📝 3. In market_regime.py für Phase Changes:")
    print("""
# In MarketRegimeDetector class
def detect_market_phase(self, data):
    new_phase = # ... phase detection logic ...
    
    # Check for phase change
    if hasattr(self, 'last_detected_phase') and self.last_detected_phase != new_phase:
        from utils.notifier import get_notifier
        notifier = get_notifier()
        if notifier:
            notifier.send_market_phase_change_alert(
                old_phase=self.last_detected_phase,
                new_phase=new_phase,
                confidence=getattr(self, 'last_confidence', None)
            )
    
    self.last_detected_phase = new_phase
    return new_phase
""")
    
    print("\n📝 4. Für Portfolio/Drawdown Monitoring:")
    print("""
# In portfolio oder risk management code
def check_drawdown(self, current_value, peak_value):
    drawdown = (peak_value - current_value) / peak_value
    
    # Alert bei signifikantem Drawdown
    if drawdown > 0.05:  # 5% Drawdown
        from utils.notifier import get_notifier
        notifier = get_notifier()
        if notifier:
            notifier.send_drawdown_alert(
                current_drawdown=drawdown,
                max_drawdown=self.max_drawdown,
                portfolio_value=current_value
            )
""")
    
    print("\n📝 5. Für API Error Handling:")
    print("""
# In API client code
def api_call(self, endpoint, params):
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        # API Error Alert
        from utils.notifier import get_notifier
        notifier = get_notifier()
        if notifier:
            notifier.send_api_error_alert(
                api_name="Binance",
                error_message=str(e),
                error_count=getattr(self, 'error_count', 1)
            )
        raise
""")
    
    print("\n📝 6. Settings.py erweitern:")
    print("""
# In config/settings.py hinzufügen:
NOTIFICATIONS = {
    'notifications': {
        'telegram': {
            'enabled': True,
            # Credentials werden aus Umgebungsvariablen geladen
        },
        'email': {
            'enabled': False,  # Erstmal nur Telegram
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'sender_email': 'your-bot@example.com',
            'recipient_email': 'alerts@example.com'
        },
        'alerts': {
            'min_level': 'WARNING',  # Nur Warning+ senden
            'enabled_types': [
                'STRATEGY_CHANGE',
                'MARKET_PHASE_CHANGE',
                'DRAWDOWN',
                'API_ERROR',
                'BOT_CRASH'
            ]
        }
    }
}

# Settings erweitern
settings.update(NOTIFICATIONS)
""")

def create_integration_checklist():
    """Erstellt eine Integrations-Checkliste"""
    checklist = """
# 🔧 Telegram Integration Checklist

## ✅ Setup (Bereits erledigt)
- [ ] Bot bei @BotFather erstellt
- [ ] Bot Token erhalten
- [ ] Chat ID ermittelt
- [ ] Credentials als Umgebungsvariablen gesetzt

## 📝 Code Integration
- [ ] notifications import in main.py hinzugefügt
- [ ] initialize_notifier() in main() aufgerufen
- [ ] Strategy change alerts in trading_bot.py
- [ ] Market phase alerts in market_regime.py  
- [ ] Drawdown alerts in risk management
- [ ] API error alerts in exchange clients
- [ ] Bot crash alerts in main exception handler

## 🧪 Testing
- [ ] test_telegram.py erfolgreich ausgeführt
- [ ] Test-Nachrichten in Telegram empfangen
- [ ] Verschiedene Alert-Typen getestet
- [ ] Rate limiting getestet

## 🚀 Production
- [ ] Settings für Production angepasst
- [ ] Min alert level konfiguriert (WARNING/ERROR)
- [ ] Nur wichtige Alert-Typen aktiviert
- [ ] Bot mit Notifications gestartet

## 🛠️ Optional
- [ ] Email Notifications konfiguriert
- [ ] Custom Alert-Templates erstellt
- [ ] Alert-Scheduling implementiert
- [ ] Dashboard für Alert-History
"""
    
    with open("telegram_integration_checklist.md", "w") as f:
        f.write(checklist)
    
    print("📋 Checklist erstellt: telegram_integration_checklist.md")

if __name__ == "__main__":
    show_integration_examples()
    create_integration_checklist()
    
    print("\n🎯 Nächste Schritte:")
    print("1. python3 setup_telegram.py  # Setup ausführen")
    print("2. source .env_telegram       # Umgebung laden")
    print("3. python3 test_telegram.py   # Integration testen")
    print("4. Code nach obigen Beispielen anpassen")
    print("5. Trading Bot starten")
    print("\n📱 Danach erhalten Sie Alerts in Telegram!")