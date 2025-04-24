# Backtesting-Dokumentation

Diese Dokumentation erklärt den Backtesting-Prozess für den Trading Bot und wie Sie die verschiedenen Tools verwenden können, um Ihre Handelsstrategien zu testen, zu optimieren und zu verfolgen.

## Inhaltsverzeichnis

1. [Grundlagen des Backtestings](#grundlagen-des-backtestings)
2. [Backtesting-Workflow](#backtesting-workflow)
3. [Verfügbare Tools](#verfügbare-tools)
4. [Parameter und Konfigurationen](#parameter-und-konfigurationen)
5. [Ergebnisse interpretieren](#ergebnisse-interpretieren)
6. [Visualisierung und Dashboards](#visualisierung-und-dashboards)
7. [Best Practices](#best-practices)
8. [Fehlerbehebung](#fehlerbehebung)
9. [Projektstruktur und Aufräumen](#projektstruktur-und-aufräumen)

## Grundlagen des Backtestings

Backtesting ist der Prozess, bei dem eine Handelsstrategie auf historischen Daten getestet wird, um ihre Wirksamkeit zu bewerten, bevor echtes Geld riskiert wird. Der Trading Bot verwendet einen fortschrittlichen Backtester, der folgende Funktionen bietet:

- Realistische Simulation von Handelsaktivitäten
- Berücksichtigung von Kommissionen und Slippage
- Umfangreiche Performance-Metriken
- Visualisierungen und Exportfunktionen
- Risikomanagement-Evaluierung

## Backtesting-Workflow

Der empfohlene Workflow für das Backtesting besteht aus folgenden Schritten:

1. **Strategie auswählen** - Entscheiden Sie, welche Strategie Sie testen möchten (Momentum, Mean Reversion, ML)
2. **Parameter konfigurieren** - Wählen Sie die Parameter, die Sie testen möchten
3. **Backtest ausführen** - Führen Sie den Backtest mit den gewählten Parametern aus
4. **Ergebnisse analysieren** - Bewerten Sie die Performance-Metriken und Visualisierungen
5. **Parameter optimieren** - Passen Sie die Parameter an und führen Sie weitere Tests durch
6. **Ergebnisse dokumentieren** - Speichern Sie die besten Ergebnisse in der Backtest-Registry
7. **Visualisierungen erstellen** - Nutzen Sie die Visualisierungstools für detaillierte Grafiken
8. **Dashboard aktualisieren** - Halten Sie Ihr Backtest-Dashboard auf dem neuesten Stand

## Verfügbare Tools

Der Trading Bot bietet mehrere Tools für das Backtesting:

### 1. Backtest-Helper (`backtest_helper.py`)

Ein Hilfsprogramm zum Ausführen und Verwalten von Backtests:

```python
from backtest_helper import run_backtest, BacktestRegistry

# Backtest ausführen
params = {
    "backtest.start_date": "2023-01-01",
    "backtest.end_date": "2023-12-31",
    "backtest.initial_balance": 10000,
    "risk.position_size": 0.1,
    # Weitere Parameter...
}

results, output_dir = run_backtest(
    strategy_name="momentum",
    params=params,
    test_name="momentum_test_1",
    tags=["test", "momentum"]
)

# Registry abfragen
registry = BacktestRegistry()
best_backtest = registry.get_best_backtest(metric="sharpe_ratio")
```

### 2. Kommandozeilenwerkzeug (`run_backtests.py`)

Ein Befehlszeilentool zum Ausführen und Verwalten von Backtests:

```bash
# Backtest ausführen
python run_backtests.py run --config config/examples/momentum_standard.json --name "momentum_test_1" --tags "test,momentum" --debug

# Backtests auflisten
python run_backtests.py list --type backtests --metric sharpe_ratio --top 10

# Backtests vergleichen
python run_backtests.py compare --tests "momentum_test_1,momentum_test_2" --metrics "total_return,sharpe_ratio,win_rate"

# Ergebnisse exportieren
python run_backtests.py export --format csv --output data/backtest_results/backtest_summary.csv

# Ergebnisse visualisieren
python run_backtests.py visualize --metric total_return --top 10 --output data/visualizations/comparison.png
```

### 3. Enhanced Backtester (`core/enhanced_backtesting.py`)

Die Hauptklasse für das Backtesting:

```python
from config.settings import Settings
from strategies.momentum import MomentumStrategy
from core.enhanced_backtesting import EnhancedBacktester

# Einstellungen initialisieren
settings = Settings()
settings.set('backtest.start_date', '2023-01-01')
# Weitere Parameter...

# Strategie initialisieren
strategy = MomentumStrategy(settings)

# Backtester initialisieren
backtester = EnhancedBacktester(settings, strategy)

# Backtest ausführen
results = backtester.run(
    symbols=['BTC/USDT', 'ETH/USDT'],
    source='binance',
    timeframe='1h',
    use_cache=True
)

# Ergebnisse visualisieren
backtester.plot_results(output_dir='data/backtest_results/my_test')

# Ergebnisse exportieren
backtester.export_results(output_dir='data/backtest_results/my_test', format='excel')
```

### 4. Visualisierungstool (`backtest_visualizer.py`)

Ein spezielles Tool für erweiterte Visualisierungen der Backtest-Ergebnisse:

```bash
# Visualisierungen für einen bestimmten Backtest erstellen
python backtest_visualizer.py --dir data/backtest_results/momentum_test_1
```

Dieses Tool erstellt folgende Visualisierungen:
- Erweiterte Equity-Kurve mit Drawdown
- Detaillierte Trade-Analyse
- Performance-Übersicht
- Parameter-Übersicht
- Gewinn/Verlust-Verteilung
- Interaktives HTML-Dashboard

### 5. Backtest-Dashboard (`backtest_dashboard.py`)

Ein interaktives Dashboard zur Übersicht und zum Vergleich aller Backtests:

```bash
# Dashboard erstellen
python backtest_dashboard.py

# Dashboard mit benutzerdefinierter Registry-Datei
python backtest_dashboard.py --registry data/custom_registry.json
```

Das Dashboard bietet:
- Übersicht der besten Backtests nach verschiedenen Metriken
- Strategie-Vergleiche
- Interaktive Tabellen
- Übersichtliche Statistiken

### 6. Aufräum-Tool (`cleanup.py`)

Ein Werkzeug zum Bereinigen des Projekts und Entfernen redundanter Dateien:

```bash
# Vorschau der zu löschenden Dateien anzeigen
python cleanup.py

# Dateien tatsächlich löschen
python cleanup.py --execute

# Auch Python-Cache und doppelte Code-Dateien entfernen
python cleanup.py --execute --pycache --duplicate-code
```

## Parameter und Konfigurationen

Die Backtest-Parameter können auf verschiedene Arten konfiguriert werden:

### 1. Konfigurationsdateien

Speichern Sie Parametersätze in JSON-Dateien im Verzeichnis `config/examples/`:

```json
{
  "strategy": "momentum",
  "backtest": {
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "initial_balance": 10000
  },
  "risk": {
    "position_size": 0.1,
    "stop_loss": 0.03,
    "take_profit": 0.06
  },
  "momentum": {
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30
  }
}
```

### 2. Programmatisch

Setzen Sie Parameter direkt im Code:

```python
settings = Settings()
settings.set('backtest.start_date', '2023-01-01')
settings.set('backtest.end_date', '2023-12-31')
settings.set('backtest.initial_balance', 10000)
settings.set('risk.position_size', 0.1)
settings.set('momentum.rsi_period', 14)
# Weitere Parameter...
```

## Ergebnisse interpretieren

Der Backtester erzeugt verschiedene Metriken zur Bewertung der Strategie-Performance:

### Grundlegende Metriken

- **Total Return**: Gesamtrendite in Prozent
- **Win Rate**: Prozentsatz erfolgreicher Trades
- **Number of Trades**: Gesamtzahl der Trades

### Risikometriken

- **Sharpe Ratio**: Rendite im Verhältnis zur Volatilität
- **Sortino Ratio**: Rendite im Verhältnis zur negativen Volatilität
- **Max Drawdown**: Maximaler Rückgang vom Höchststand
- **Profit Factor**: Verhältnis von Gewinnen zu Verlusten
- **Calmar Ratio**: Annualisierte Rendite geteilt durch den maximalen Drawdown

## Visualisierung und Dashboards

### Standardvisualisierungen

Der Backtester erzeugt automatisch folgende Visualisierungen:

1. **Equity-Kurve mit Drawdown**: Zeigt die Entwicklung des Portfoliowerts und Drawdowns
2. **Trade-Verteilung**: Histogramm der Trade-Ergebnisse
3. **Symbol-Performance**: Performance nach Handelssymbol
4. **Monatliche Returns**: Heatmap der monatlichen Renditen
5. **Performance-Kennzahlen**: Übersicht der wichtigsten Kennzahlen

### Erweiterte Visualisierungen

Mit dem `backtest_visualizer.py` Tool können zusätzliche, detailliertere Visualisierungen erstellt werden:

1. **Erweiterte Equity-Kurve**: Mit Drawdown, offenen Positionen und Balance
2. **Detaillierte Trade-Analyse**: Mit kumulativer P/L, Verteilung und Erfolgsrate
3. **Performance-Übersicht**: Tabellarische Darstellung aller Metriken
4. **Parameter-Übersicht**: Visualisierung aller verwendeten Parameter
5. **Gewinn/Verlust-Verteilung**: Mit statistischer Analyse
6. **HTML-Dashboard**: Ein interaktives Dashboard für jeden Backtest

### Backtest-Dashboard

Das `backtest_dashboard.py` Tool erstellt ein zentrales Dashboard für alle Backtests:

1. **Übersicht**: Gesamtanzahl der Backtests, Strategien und beste Ergebnisse
2. **Strategie-Vergleich**: Vergleichende Grafiken der verschiedenen Strategien
3. **Top-Performer**: Tabellen mit den besten Backtests nach verschiedenen Metriken
4. **Interaktive Elemente**: Filter und Sortierfunktionen

Alle Visualisierungen werden standardmäßig im Verzeichnis `data/visualizations/` gespeichert. Dashboard-Dateien werden unter `data/dashboard/` abgelegt.

## Best Practices

Hier sind einige Empfehlungen für effektives Backtesting:

1. **Systematischer Ansatz**:
   - Dokumentieren Sie alle Änderungen und Parameter
   - Verwenden Sie aussagekräftige Testnamen
   - Kategorisieren Sie Tests mit Tags

2. **Vermeidung von Overfitting**:
   - Testen Sie auf verschiedenen Zeiträumen
   - Teilen Sie den Datensatz in In-Sample und Out-of-Sample
   - Begrenzen Sie die Anzahl der getesteten Parameter

3. **Realistische Annahmen**:
   - Berücksichtigen Sie realistische Kommissionen
   - Berücksichtigen Sie Slippage bei volatilen Märkten
   - Testen Sie verschiedene Marktbedingungen

4. **Diversifizierte Bewertung**:
   - Verlassen Sie sich nicht nur auf eine Metrik
   - Bewerten Sie Rendite und Risiko
   - Berücksichtigen Sie die Stabilität der Strategie

5. **Ergebnisse visualisieren**:
   - Nutzen Sie die erweiterten Visualisierungstools
   - Vergleichen Sie Strategien im Dashboard
   - Achten Sie auf Muster in den Handelsergebnissen

## Fehlerbehebung

### Häufige Probleme und Lösungen

1. **Keine Daten verfügbar**
   - Überprüfen Sie den Zeitraum
   - Überprüfen Sie, ob Daten für die ausgewählten Symbole existieren
   - Prüfen Sie die Cache-Einstellungen

2. **Keine Trades generiert**
   - Überprüfen Sie die Strategie-Parameter
   - Prüfen Sie, ob die Signale zu restriktiv sind
   - Erhöhen Sie den Testzeitraum

3. **Unrealistische Ergebnisse**
   - Überprüfen Sie die Kommissionseinstellungen
   - Prüfen Sie, ob das Risikomanagement korrekt konfiguriert ist
   - Überprüfen Sie die Handelsvolumen

4. **Fehler bei der Ausführung**
   - Überprüfen Sie das Log auf spezifische Fehlermeldungen
   - Stellen Sie sicher, dass alle erforderlichen Parameter gesetzt sind
   - Prüfen Sie, ob alle Abhängigkeiten installiert sind
   - Verwenden Sie den `--debug` Parameter bei `run_backtests.py`

5. **Dateien werden im falschen Verzeichnis gespeichert**
   - Überprüfen Sie, ob die neue Verzeichnisstruktur korrekt eingerichtet ist
   - Stellen Sie sicher, dass `data/backtest_results/` und `data/visualizations/` existieren
   - Führen Sie `cleanup.py` aus, um redundante Dateien zu entfernen

## Projektstruktur und Aufräumen

Die Backtest-Ergebnisse und -Visualisierungen werden in einer strukturierten Verzeichnishierarchie gespeichert:

```
altcoin_trading_bot/
├── data/
│   ├── backtest_registry.json       # Zentrale Registry aller Backtests
│   ├── backtest_results/            # Alle Backtest-Ergebnisse
│   │   ├── test_1/                  # Ergebnisse für einen bestimmten Test
│   │   │   ├── config.json          # Konfiguration des Tests
│   │   │   ├── summary.json         # Zusammenfassung der Ergebnisse
│   │   │   ├── *.xlsx               # Excel-Exporte
│   │   │   └── *.png                # Visualisierungen
│   │   └── ...
│   ├── visualizations/              # Allgemeine Visualisierungen
│   │   ├── strategy_comparison.png  # Strategie-Vergleich
│   │   └── ...
│   ├── dashboard/                   # Interaktives Dashboard
│   │   └── index.html               # Haupt-Dashboard-Datei
│   ├── market_data/                 # Marktdaten
│   │   ├── binance/                 # Nach Quelle gruppiert
│   │   └── coingecko/
│   └── states/                      # Bot-Zustandsdateien
├── core/                            # Kernfunktionalität
│   ├── data_sources/                # Datenquellen
│   └── ...
└── ...
```

### Aufräumen des Projekts

Verwenden Sie das `cleanup.py` Tool, um redundante Dateien und Verzeichnisse zu entfernen:

```bash
# Vorschau der zu entfernenden Dateien
python cleanup.py

# Tatsächliches Aufräumen
python cleanup.py --execute
```

Das Tool erkennt und bereinigt:
- Verschachtelte `data`-Verzeichnisse
- Doppelte Registry-Dateien
- Alte Backtest-Verzeichnisse
- Python-Cache-Dateien (`__pycache__`)
- Doppelte Code-Dateien

### Dateimanagement

- **Registry-Datei**: Die zentrale Datenbank aller Backtests wird unter `data/backtest_registry.json` gespeichert und automatisch vom System verwaltet.
- **Backtest-Ergebnisse**: Jeder Backtest bekommt ein eigenes Verzeichnis unter `data/backtest_results/`.
- **Visualisierungen**: Grafiken werden im `data/visualizations/`-Verzeichnis gespeichert.
- **Dashboard**: Das interaktive Dashboard wird unter `data/dashboard/` abgelegt.

Diese Struktur sollte nicht manuell verändert werden. Verwenden Sie stattdessen die bereitgestellten Tools zum Verwalten von Backtests und deren Ergebnissen.