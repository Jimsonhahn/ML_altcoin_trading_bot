# Altcoin Trading Bot

Ein automatisierter Trading Bot für Kryptowährungen mit Fokus auf Altcoins. Der Bot kombiniert technische Analyse, optionale Social Media Sentiment-Analyse und Machine Learning, um Handelssignale zu generieren.

## Features

- **Mehrere Trading-Modi**: Live Trading, Paper Trading und Backtesting
- **Multiple Strategien**: Momentum, Mean Reversion und ML-basierte Strategien
- **Risikomanagement**: Stop-Loss, Take-Profit und Trailing-Stop
- **Modular erweiterbar**: Einfache Integration neuer Strategien und Datenquellen
- **Datenvisualisierung**: Übersichtliche Darstellung der Performance und Trades

## Voraussetzungen

- Python 3.8 oder höher
- Binance-Konto (für Live und Paper Trading)
- API-Schlüssel für Binance und optional für Social Media Plattformen

## Installation

1. Repository klonen oder Dateien in ein neues Verzeichnis kopieren

2. Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt
   ```
   
3. TA-Lib installieren (kann je nach Betriebssystem komplizierter sein):
   - **Windows**: Vorcompilierte Binaries von [hier](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib) herunterladen
   - **Linux**: `apt-get install ta-lib` oder kompilieren aus den Quellen
   - **macOS**: `brew install ta-lib`

4. `.env`-Datei im Hauptverzeichnis erstellen und API-Schlüssel konfigurieren:
   ```
   # Binance Live API Keys
   LIVE_API_KEY=dein_binance_live_api_key
   LIVE_SECRET=dein_binance_live_secret
   
   # Binance Testnet API Keys
   TESTNET_API_KEY=dein_binance_testnet_api_key
   TESTNET_SECRET=dein_binance_testnet_secret
   
   # Optional: Social Media API Keys
   REDDIT_CLIENT_ID=dein_reddit_client_id
   REDDIT_CLIENT_SECRET=dein_reddit_client_secret
   REDDIT_USER_AGENT=dein_reddit_user_agent
   
   TWITTER_CONSUMER_KEY=dein_twitter_consumer_key
   TWITTER_CONSUMER_SECRET=dein_twitter_consumer_secret
   TWITTER_ACCESS_TOKEN=dein_twitter_access_token
   TWITTER_ACCESS_SECRET=dein_twitter_access_secret
   
   YOUTUBE_API_KEY=dein_youtube_api_key
   ```

## Verwendung

### Backtesting

Um eine Strategie auf historischen Daten zu testen:

```bash
python main.py --mode=backtest --strategy=momentum
```

Backtesting-Ergebnisse werden im `results`-Verzeichnis als Grafiken gespeichert.

### Paper Trading

Um den Bot im Paper Trading-Modus zu starten (simulierter Handel mit Echtzeit-Daten):

```bash
python main.py --mode=paper --strategy=momentum
```

### Live Trading

**Achtung**: Dies führt zu echten Trades mit echtem Geld!

```bash
python main.py --mode=live --strategy=momentum
```

### Konfigurationsprofil verwenden

```bash
python main.py --mode=paper --strategy=momentum --config=aggressive
```

### Debug-Modus aktivieren

```bash
python main.py --mode=paper --strategy=momentum --debug
```

## Konfiguration

Die Bot-Konfiguration erfolgt über Profile im Verzeichnis `config/profiles/`. Ein Beispiel-Profil (`default.json`) könnte so aussehen:

```json
{
  "trading_pairs": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"],
  "risk": {
    "position_size": 0.05,
    "stop_loss": 0.03,
    "take_profit": 0.06
  },
  "technical": {
    "rsi": {
      "period": 14,
      "oversold": 30,
      "overbought": 70
    }
  }
}
```

## Eigene Strategien erstellen

1. Erstelle eine neue Python-Datei in `strategies/`, z.B. `my_strategy.py`
2. Erweitere die `Strategy`-Basisklasse und implementiere alle erforderlichen Methoden
3. Importiere deine Strategie in `core/trading.py` und füge sie zur Strategie-Initialisierung hinzu

Beispiel für eine eigene Strategie:

```python
from strategies.strategy_base import Strategy

class MyStrategy(Strategy):
    def __init__(self, settings):
        super().__init__(settings)
        self.name = "my_strategy"
        # Eigene Parameter initialisieren
        
    def prepare_data(self, df):
        # Daten vorbereiten und Indikatoren berechnen
        return df
        
    def generate_signal(self, df, symbol, current_position):
        # Trading-Logik implementieren
        return "HOLD", {"signal": "HOLD", "confidence": 0.0}
```

## Projektstruktur

```
altcoin_trading_bot/
│
├── config/               # Konfigurationsdateien
│   ├── settings.py       # Konfigurationsmanagement
│   ├── api_keys.py       # API-Schlüssel Template
│   └── profiles/         # Konfigurationsprofile
│
├── core/                 # Kernfunktionalität
│   ├── exchange.py       # Exchange-Anbindung
│   ├── trading.py        # Haupttrading-Logik
│   ├── position.py       # Positionsmanagement
│   └── backtesting.py    # Backtesting-Engine
│
├── analysis/             # Analysemodule
│   ├── technical.py      # Technische Analyse
│   ├── social_media.py   # Social Media Analyse
│   ├── on_chain.py       # On-Chain-Analyse
│   └── machine_learning.py # ML-Modelle
│
├── strategies/           # Trading-Strategien
│   ├── strategy_base.py  # Basisklasse für Strategien
│   ├── momentum.py       # Momentum-Strategie
│   ├── mean_reversion.py # Mean Reversion Strategie
│   └── ml_strategy.py    # ML-basierte Strategie
│
├── utils/                # Hilfsfunktionen
│   ├── logger.py         # Logging
│   └── helpers.py        # Sonstige Hilfsfunktionen
│
├── models/               # Gespeicherte ML-Modelle
├── logs/                 # Log-Dateien
├── results/              # Backtesting-Ergebnisse
├── .env                  # API-Schlüssel (nicht committen!)
├── requirements.txt      # Abhängigkeiten
├── main.py               # Haupteinstiegspunkt
└── README.md             # Diese Datei
```

## Risikomanagement

**Wichtiger Hinweis**: Trading mit Kryptowährungen ist hochriskant. Verwende diesen Bot auf eigene Gefahr und starte mit kleinen Beträgen. Der Bot enthält zwar Risikomanagement-Funktionen, kann aber dennoch zu Verlusten führen.

## Fortgeschrittene Nutzung

### ML-Modell trainieren

```bash
python -c "from strategies.ml_strategy import MLStrategy; from config.settings import Settings; s = Settings(); m = MLStrategy(s); m.train(df, 'BTC/USDT')"
```

### Strategie-Parameter optimieren

```bash
python -c "from strategies.momentum import MomentumStrategy; from config.settings import Settings; s = Settings(); m = MomentumStrategy(s); m.optimize(df, 'BTC/USDT')"
```

## Fehlerbehebung

### Verbindungsprobleme mit Binance

- Prüfe, ob deine API-Schlüssel korrekt sind
- Stelle sicher, dass dein Netzwerk keine Verbindung zu Binance blockiert
- Prüfe, ob Trading-Rechte für die API-Schlüssel aktiviert sind

### TA-Lib Installation schlägt fehl

- Versuche, vorcompilierte Binaries zu verwenden
- Auf Linux/macOS: Installiere die Abhängigkeiten vor dem pip-Install:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install build-essential
  # macOS
  brew install pkg-config
  ```

## Weiterentwicklung

Ideen für zukünftige Verbesserungen:
- Integration weiterer Exchanges (Coinbase, Kraken, etc.)
- Erweiterung der Social Media Analyse
- Integration von On-Chain-Daten
- Verbesserung der ML-Modelle
- Benachrichtigungssystem (E-Mail, Telegram, etc.)
- Web-Interface zur Überwachung und Konfiguration

## Lizenz

Dieses Projekt ist für den persönlichen Gebrauch bestimmt. Alle Rechte vorbehalten.