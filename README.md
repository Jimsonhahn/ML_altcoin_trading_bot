# Altcoin Trading Bot

Ein automatisierter Trading Bot für Kryptowährungen mit Fokus auf Altcoins. Der Bot kombiniert technische Analyse, optionale Social Media Sentiment-Analyse und Machine Learning, um Handelssignale zu generieren und bietet umfangreiche Backtesting-Funktionen.

## Features

- **Mehrere Trading-Modi**: Live Trading, Paper Trading und erweitertes Backtesting
- **Multiple Strategien**: Momentum, Mean Reversion und ML-basierte Strategien
- **Risikomanagement**: Stop-Loss, Take-Profit, Trailing-Stop und dynamisches Position Sizing
- **Modular erweiterbar**: Einfache Integration neuer Strategien und Datenquellen
- **Datenvisualisierung**: Übersichtliche Darstellung der Performance und Trades
- **Erweiterte Backtesting-Engine**: Umfassende Analyse und Optimierung von Handelsstrategien
- **Backtest-Tracking**: Zentrale Verfolgung und Vergleich von Backtest-Ergebnissen
- **Parallel-Verarbeitung**: Effiziente Verarbeitung mehrerer Trading-Paare gleichzeitig
- **Monte Carlo Simulation**: Fortschrittliche Risikoanalyse für Handelsstrategien
- **Verbesserte Berichterstattung**: Detaillierte HTML/Markdown/JSON-Berichte mit interaktiven Dashboards
- **Fehlerresistenz**: Automatisches Speichern und Wiederherstellen von Bot-Zuständen

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

### Trading-Modi

#### Backtesting

Um eine Strategie auf historischen Daten zu testen:

```bash
python main.py --mode=backtest --strategy=momentum
```

Für erweiterte Backtest-Funktionen mit Monte Carlo Simulation:

```bash
python main.py --mode=backtest --strategy=momentum --config=config/examples/momentum_standard.json
```

Backtesting-Ergebnisse werden im `data/backtest_results/`-Verzeichnis als Grafiken und Daten gespeichert. Weitere Details zum Backtesting finden Sie in der [Backtesting-Dokumentation](docs/backtesting.md).

#### Paper Trading

Um den Bot im Paper Trading-Modus zu starten (simulierter Handel mit Echtzeit-Daten):

```bash
python main.py --mode=paper --strategy=momentum
```

#### Live Trading

**Achtung**: Dies führt zu echten Trades mit echtem Geld!

```bash
python main.py --mode=live --strategy=momentum
```

### Erweiterte Kommandozeilenoptionen

```bash
# Konfigurationsprofil verwenden
python main.py --mode=paper --strategy=momentum --config=config/profiles/aggressive.json

# Nur bestimmte Trading-Paare handeln
python main.py --mode=paper --strategy=momentum --pairs="BTC/USDT,ETH/USDT,SOL/USDT"

# Debug-Modus mit erhöhter Ausführlichkeit
python main.py --mode=paper --strategy=momentum --verbose --verbose

# Marktanalyse durchführen
python main.py --analyze-market --pairs="BTC/USDT,ETH/USDT,SOL/USDT,DOGE/USDT"

# Strategie-Parameter optimieren
python main.py --optimize --strategy=momentum --param-grid=config/param_grids/momentum_grid.json

# Zustand speichern und laden
python main.py --mode=paper --strategy=momentum --save-state=bot_state.json
python main.py --mode=paper --strategy=momentum --load-state=bot_state.json

# Trading-Bericht generieren
python main.py --report --report-days=30 --report-format=html
```

Vollständige Liste der Kommandozeilenoptionen:

```bash
python main.py --help
```

## Konfiguration

Die Bot-Konfiguration erfolgt über Profile im Verzeichnis `config/profiles/`. Ein Beispiel-Profil (`default.json`) könnte so aussehen:

```json
{
  "trading_pairs": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"],
  "risk": {
    "position_size": 0.05,
    "stop_loss": 0.03,
    "take_profit": 0.06,
    "max_open_positions": 5,
    "dynamic_position_sizing": true,
    "adjust_to_min_size": true
  },
  "technical": {
    "rsi": {
      "period": 14,
      "oversold": 30,
      "overbought": 70
    }
  },
  "system": {
    "parallel_data_updates": true,
    "parallel_signal_processing": true,
    "max_workers": 8,
    "auto_save_interval": 30,
    "save_directory": "data/states",
    "save_on_exit": true,
    "exit_save_path": "data/states/last_state.json",
    "api_error_window": 300,
    "max_api_errors": 5
  },
  "timeframes": {
    "check_interval": 300,
    "analysis": "1h"
  },
  "backtest": {
    "monte_carlo": true,
    "monte_carlo_iterations": 1000,
    "monte_carlo_confidence": 0.95,
    "generate_plots": true,
    "export_results": true,
    "export_format": "csv"
  },
  "optimization": {
    "parallel": true,
    "max_workers": 8,
    "apply_best": false
  },
  "reports": {
    "auto_generate_on_stop": true,
    "default_days": 30,
    "default_format": "html"
  }
}
```

Eine vollständige Liste aller verfügbaren Parameter finden Sie im [Parameter-Katalog](docs/parameter_catalog.md).

## Backtesting

Der Bot verfügt über ein leistungsstarkes Backtesting-System zum Testen und Optimieren von Handelsstrategien.

### Backtesting-Workflow

Der typische Backtesting-Workflow besteht aus:

1. **Strategie auswählen** - Wählen Sie die zu testende Strategie
2. **Parameter konfigurieren** - Definieren Sie Testparameter
3. **Backtest ausführen** - Testen Sie die Strategie auf historischen Daten
4. **Ergebnisse analysieren** - Bewerten Sie Performance-Metriken und Visualisierungen
5. **Parameter optimieren** - Verfeinern Sie die Parameter basierend auf den Ergebnissen
6. **Monte Carlo Simulation** - Bewerten Sie das Risiko der Strategie
7. **In Produktion bringen** - Führen Sie die optimierte Strategie im Paper- oder Live-Modus aus

### Backtest-Management

Sie können Ihre Backtests mit dem `run_backtests.py`-Tool verwalten:

```bash
# Backtest ausführen
python run_backtests.py run --config config/examples/momentum_standard.json

# Backtests auflisten
python run_backtests.py list --type backtests --metric sharpe_ratio --top 10

# Backtests vergleichen
python run_backtests.py compare --tests "test1,test2" --metrics "total_return,sharpe_ratio"

# Ergebnisse visualisieren
python run_backtests.py visualize --metric total_return --top 10
```

### Verfügbare Backtesting-Tools

Der Bot bietet mehrere spezialisierte Tools für das Backtesting:

1. **Backtest-Helper (`backtest_helper.py`)**: Programmatisches Ausführen und Verwalten von Backtests
2. **Enhanced Backtester (`core/enhanced_backtesting.py`)**: Hauptklasse für detailliertes Backtesting
3. **Visualisierungstool (`backtest_visualizer.py`)**: Erweiterte Visualisierungen der Backtest-Ergebnisse
4. **Backtest-Dashboard (`backtest_dashboard.py`)**: Interaktives Dashboard für Backtest-Vergleiche
5. **Aufräum-Tool (`cleanup.py`)**: Bereinigung redundanter Dateien und Verzeichnisse

Detaillierte Informationen zu diesen Tools und ihrer Verwendung finden Sie in der [Backtesting-Dokumentation](docs/backtesting.md).

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
        # Optional: Thread-Lock für Parallelverarbeitung
        self.lock = threading.Lock()
        
    def prepare_data(self, df):
        # Daten vorbereiten und Indikatoren berechnen
        return df
        
    def generate_signal(self, df, symbol, current_position):
        # Trading-Logik implementieren
        return "HOLD", {"signal": "HOLD", "confidence": 0.0}
        
    # Optional: State-Management für Wiederherstellung
    def get_state(self):
        # Strategie-Zustand speichern
        return {"name": self.name, "parameters": {...}}
        
    def set_state(self, state):
        # Strategie-Zustand wiederherstellen
        # ...
```

## Leistungsoptimierung

### Parallele Verarbeitung

Der Bot kann mehrere Trading-Paare parallel verarbeiten, was die Effizienz erheblich verbessert.

Aktivieren Sie die Parallelverarbeitung in der Konfiguration:

```json
{
  "system": {
    "parallel_data_updates": true,
    "parallel_signal_processing": true,
    "max_workers": 8
  }
}
```

### Caching

Der Bot verwendet einen intelligenten Daten-Cache, um wiederholte API-Anfragen zu reduzieren:

```json
{
  "data": {
    "source": "exchange",
    "min_candles": 50,
    "source_name": "binance"
  }
}
```

## Trading-Berichte

Der Bot kann detaillierte Berichte über Ihre Handelsaktivitäten generieren:

```bash
python main.py --report --report-days=30 --report-format=html
```

Berichte enthalten:
- Performance-Übersicht
- Gewinn/Verlust-Visualisierungen
- Handelsaktivität nach Stunde und Tag
- Symbol-spezifische Statistiken
- Detaillierte Auflistung aller Trades

## Fehlerresistenz und Sicherheit

### Zustandsmanagement

Der Bot kann seinen Status speichern und wiederherstellen, was bei Neustarts oder Unterbrechungen hilft:

```bash
# Status speichern
python main.py --save-state=bot_state.json

# Status laden
python main.py --load-state=bot_state.json
```

Aktivieren Sie automatisches Speichern in der Konfiguration:

```json
{
  "system": {
    "auto_save_interval": 30,
    "save_directory": "data/states",
    "save_on_exit": true
  }
}
```

### API-Fehlerbehandlung

Der Bot überwacht API-Fehler und fährt bei zu vielen Fehlern automatisch herunter, um weiteren Schaden zu vermeiden:

```json
{
  "system": {
    "api_error_window": 300,
    "max_api_errors": 5
  }
}
```

## Projektstruktur

```
altcoin_trading_bot/
│
├── config/               # Konfigurationsdateien
│   ├── settings.py       # Konfigurationsmanagement
│   ├── api_keys.py       # API-Schlüssel Template
│   ├── profiles/         # Konfigurationsprofile
│   ├── param_grids/      # Parameter-Grids für Optimierung
│   └── examples/         # Beispielkonfigurationen für Backtests
│
├── core/                 # Kernfunktionalität
│   ├── exchange.py       # Exchange-Anbindung
│   ├── trading_bot.py    # Haupttrading-Logik
│   ├── position.py       # Positionsmanagement
│   ├── enhanced_backtesting.py # Erweiterte Backtesting-Engine
│   └── data_sources/     # Datenquellen-Module
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
├── backtest_helper.py    # Hilfsfunktionen für Backtesting
├── run_backtests.py      # Kommandozeilentool für Backtests
├── backtest_visualizer.py # Tool für erweiterte Visualisierungen
├── backtest_dashboard.py # Interaktives Backtest-Dashboard
├── cleanup.py            # Tool zum Bereinigen des Projekts
│
├── data/                 # Daten und Bot-Status
│   ├── backtest_registry.json # Zentrale Backtest-Datenbank
│   ├── backtest_results/ # Alle Backtest-Ergebnisse
│   │   └── [test_name]/  # Ergebnisse für spezifische Tests
│   ├── visualizations/   # Allgemeine Visualisierungen
│   ├── dashboard/        # Interaktives Dashboard
│   ├── market_data/      # Historische Marktdaten
│   │   ├── binance/      # Nach Quelle gruppiert
│   │   └── coingecko/
│   ├── cache/            # Daten-Cache
│   └── states/           # Gespeicherte Bot-Zustände
│
├── models/               # Gespeicherte ML-Modelle
├── logs/                 # Log-Dateien
├── reports/              # Generierte Trading-Berichte
│
├── docs/                 # Dokumentation
│   ├── parameter_catalog.md # Vollständiger Parameter-Katalog
│   └── backtesting.md    # Backtesting-Dokumentation
│
├── .env                  # API-Schlüssel (nicht committen!)
├── requirements.txt      # Abhängigkeiten
├── main.py               # Haupteinstiegspunkt
└── README.md             # Diese Datei
```

## Fehlerbehebung

### Allgemeine Probleme

#### Verbindungsprobleme mit Binance

- Prüfe, ob deine API-Schlüssel korrekt sind
- Stelle sicher, dass dein Netzwerk keine Verbindung zu Binance blockiert
- Prüfe, ob Trading-Rechte für die API-Schlüssel aktiviert sind
- Prüfe die API-Fehlerprotokolle im Logs-Verzeichnis

#### TA-Lib Installation schlägt fehl

- Versuche, vorcompilierte Binaries zu verwenden
- Auf Linux/macOS: Installiere die Abhängigkeiten vor dem pip-Install:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install build-essential
  # macOS
  brew install pkg-config
  ```

### Backtest-Probleme

- Stelle sicher, dass die Daten für den angegebenen Zeitraum verfügbar sind
- Überprüfe die Parametereinstellungen in der Konfigurationsdatei
- Achte auf ausreichend Speicherplatz für große Datensätze
- Bei Out-of-Memory-Fehlern: Reduziere die Anzahl der Trading-Paare oder den Zeitraum
- Weitere detaillierte Hinweise zur Fehlerbehebung bei Backtests finden Sie in der [Backtesting-Dokumentation](docs/backtesting.md)

### Parallelverarbeitungsprobleme

- Prüfe die CPU-Auslastung während der Ausführung
- Reduziere `max_workers` in den Systemeinstellungen, falls der Bot zu viel CPU-Last verursacht
- Bei Problemen mit Daten-Inkonsistenz: Deaktiviere die Parallelverarbeitung vorübergehend

## Risikomanagement

**Wichtiger Hinweis**: Trading mit Kryptowährungen ist hochriskant. Verwende diesen Bot auf eigene Gefahr und starte mit kleinen Beträgen. Der Bot enthält zwar Risikomanagement-Funktionen, kann aber dennoch zu Verlusten führen.

### Erweiterte Risikomanagement-Funktionen

Der Bot bietet folgende Risikomanagement-Funktionen:

1. **Stop-Loss**: Automatisches Schließen von Positionen bei einem bestimmten Verlustprozentsatz
2. **Take-Profit**: Automatisches Gewinnmitnehmen bei einem bestimmten Gewinnprozentsatz
3. **Trailing-Stop**: Dynamischer Stop-Loss, der mit dem Kurs steigt
4. **Dynamisches Position Sizing**: Anpassung der Positionsgröße basierend auf der Volatilität
5. **Maximale offene Positionen**: Begrenzung der Anzahl gleichzeitiger Trades
6. **Monte Carlo Simulation**: Risikobewertung durch stochastische Analyse

## Fortgeschrittene Nutzung

### ML-Modell trainieren

```bash
python -c "from strategies.ml_strategy import MLStrategy; from config.settings import Settings; s = Settings(); m = MLStrategy(s); m.train(df, 'BTC/USDT')"
```

### Strategie-Parameter optimieren

```bash
# Über die Kommandozeile
python main.py --optimize --strategy=momentum --param-grid=config/param_grids/momentum_grid.json

# Oder programmatisch
python -c "from core.trading_bot import TradingBot; from config.settings import Settings; s = Settings(); bot = TradingBot(mode='backtest', strategy_name='momentum', settings=s); bot.optimize_strategy()"
```

### Marktanalyse durchführen

```bash
# Vollständige Marktanalyse über alle konfigurierten Paare
python main.py --analyze-market

# Ergebnisse werden in market_analysis.json gespeichert
```

## Weiterentwicklung

Ideen für zukünftige Verbesserungen:
- Integration weiterer Exchanges (Coinbase, Kraken, etc.)
- Erweiterung der Social Media Analyse
- Integration von On-Chain-Daten
- Verbesserung der ML-Modelle mit Deep Learning
- Benachrichtigungssystem (E-Mail, Telegram, etc.)
- Web-Interface zur Überwachung und Konfiguration
- Docker-Container für einfache Bereitstellung
- Distributed Computing für leistungshungrige Operationen
- MetaTrader-Integration für traditionelle Märkte

## Lizenz

Dieses Projekt ist für den persönlichen Gebrauch bestimmt. Alle Rechte vorbehalten.