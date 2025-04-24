# Parameter-Katalog

Dieses Dokument enthält eine vollständige Liste aller verfügbaren Parameter für den Trading Bot. Parameter können über die `settings.set()` Methode oder in einer Konfigurationsdatei definiert werden.

## Inhaltsverzeichnis

- [Allgemeine Einstellungen](#allgemeine-einstellungen)
- [Backtest-Parameter](#backtest-parameter)
- [Risikomanagement](#risikomanagement)
- [Zeitrahmen](#zeitrahmen)
- [Datenquellen](#datenquellen)
- [Momentum-Strategie Parameter](#momentum-strategie-parameter)
- [Mean Reversion-Strategie Parameter](#mean-reversion-strategie-parameter)
- [ML-Strategie Parameter](#ml-strategie-parameter)
- [Logging und Benachrichtigungen](#logging-und-benachrichtigungen)

## Allgemeine Einstellungen

| Parameter | Typ | Standardwert | Beschreibung |
|-----------|-----|--------------|-------------|
| `trading_pairs` | Liste | `["BTC/USDT"]` | Liste der Handelssymbole |
| `exchange.name` | String | `"binance"` | Name der zu verwendenden Börse |
| `exchange.api_key` | String | `""` | API-Schlüssel für die Börse |
| `exchange.api_secret` | String | `""` | API-Secret für die Börse |
| `system.status_update_interval` | Integer | `60` | Intervall für Statusaktualisierungen (Sekunden) |

## Backtest-Parameter

| Parameter | Typ | Standardwert | Beschreibung |
|-----------|-----|--------------|-------------|
| `backtest.start_date` | String | `"2023-01-01"` | Startdatum für den Backtest (Format: YYYY-MM-DD) |
| `backtest.end_date` | String | `"2023-12-31"` | Enddatum für den Backtest (Format: YYYY-MM-DD) |
| `backtest.initial_balance` | Float | `10000` | Anfangskapital für den Backtest |
| `backtest.commission` | Float | `0.001` | Kommissionsgebühr (0.001 = 0.1%) |
| `backtest.generate_plots` | Boolean | `true` | Generierung von Visualisierungen |
| `backtest.export_results` | Boolean | `true` | Ergebnisse exportieren |
| `backtest.export_format` | String | `"csv"` | Exportformat (`csv`, `json`, `excel`) |

## Risikomanagement

| Parameter | Typ | Standardwert | Empfohlener Bereich | Beschreibung |
|-----------|-----|--------------|---------------------|--------------------|
| `risk.position_size` | Float | `0.05` | `0.01` - `0.2` | Anteil des Kapitals pro Trade (0.05 = 5%) |
| `risk.stop_loss` | Float | `0.03` | `0.01` - `0.1` | Stop-Loss in Prozent (0.03 = 3%) |
| `risk.take_profit` | Float | `0.06` | `0.02` - `0.2` | Take-Profit in Prozent (0.06 = 6%) |
| `risk.max_open_positions` | Integer | `5` | `1` - `10` | Maximale Anzahl gleichzeitiger Positionen |
| `risk.min_confidence` | Float | `0.6` | `0.5` - `0.9` | Mindestkonfidenz für Signalausführung |
| `risk.dynamic_position_sizing` | Boolean | `false` | - | Dynamische Positionsgrößenbestimmung |

## Zeitrahmen

| Parameter | Typ | Standardwert | Beschreibung |
|-----------|-----|--------------|-------------|
| `timeframes.check_interval` | Integer | `300` | Intervall für die Überprüfung von Signalen (Sekunden) |
| `timeframes.analysis` | String | `"1h"` | Zeitrahmen für die Analyse (`1m`, `5m`, `15m`, `1h`, `4h`, `1d`) |

## Datenquellen

| Parameter | Typ | Standardwert | Beschreibung |
|-----------|-----|--------------|-------------|
| `data.source` | String | `"exchange"` | Datenquelle (`exchange`, `coingecko`, `csv`) |
| `data.source_name` | String | `"binance"` | Name der Datenquelle |
| `data.cache_dir` | String | `"data/cache"` | Verzeichnis für zwischengespeicherte Daten |
| `data.use_cache` | Boolean | `true` | Zwischengespeicherte Daten verwenden, wenn verfügbar |

## Momentum-Strategie Parameter

| Parameter | Typ | Standardwert | Empfohlener Bereich | Beschreibung |
|-----------|-----|--------------|---------------------|--------------|
| `momentum.rsi_period` | Integer | `14` | `7` - `21` | RSI-Periode |
| `momentum.rsi_overbought` | Float | `70` | `65` - `80` | RSI-Überkauft-Schwelle |
| `momentum.rsi_oversold` | Float | `30` | `20` - `35` | RSI-Überverkauft-Schwelle |
| `momentum.ema_short` | Integer | `9` | `5` - `20` | Kurzfristiger EMA |
| `momentum.ema_long` | Integer | `21` | `15` - `50` | Langfristiger EMA |
| `momentum.macd_fast` | Integer | `12` | `8` - `24` | MACD schnelle Periode |
| `momentum.macd_slow` | Integer | `26` | `16` - `52` | MACD langsame Periode |
| `momentum.macd_signal` | Integer | `9` | `5` - `15` | MACD Signal-Periode |
| `momentum.use_volume_filter` | Boolean | `true` | - | Volumenfilter verwenden |
| `momentum.volume_threshold` | Float | `1.5` | `1.1` - `3.0` | Volumenschwelle (Verhältnis zum Durchschnitt) |

## Mean Reversion-Strategie Parameter

| Parameter | Typ | Standardwert | Empfohlener Bereich | Beschreibung |
|-----------|-----|--------------|---------------------|--------------|
| `mean_reversion.bollinger_period` | Integer | `20` | `10` - `30` | Bollinger-Bänder-Periode |
| `mean_reversion.bollinger_std` | Float | `2.0` | `1.5` - `3.0` | Bollinger-Bänder-Standardabweichung |
| `mean_reversion.mean_period` | Integer | `50` | `20` - `100` | Periode für Mittelwertberechnung |
| `mean_reversion.max_deviation` | Float | `0.05` | `0.02` - `0.1` | Maximale Abweichung vom Mittelwert |
| `mean_reversion.atr_period` | Integer | `14` | `7` - `21` | ATR-Periode für Volatilitätsberechnung |
| `mean_reversion.use_rsi_filter` | Boolean | `true` | - | RSI-Filter verwenden |

## ML-Strategie Parameter

| Parameter | Typ | Standardwert | Beschreibung |
|-----------|-----|--------------|-------------|
| `ml.model_type` | String | `"random_forest"` | ML-Modelltyp (`random_forest`, `xgboost`, `lstm`) |
| `ml.features` | Liste | `["rsi", "macd", "ema"]` | Zu verwendende Features |
| `ml.train_size` | Float | `0.7` | Anteil der Daten für Training (0.7 = 70%) |
| `ml.prediction_threshold` | Float | `0.6` | Schwellenwert für Prognosen |
| `ml.retrain_interval` | Integer | `7` | Intervall für Neutraining (Tage) |
| `ml.model_dir` | String | `"models/"` | Verzeichnis für gespeicherte Modelle |

## Logging und Benachrichtigungen

| Parameter | Typ | Standardwert | Beschreibung |
|-----------|-----|--------------|-------------|
| `logging.level` | String | `"INFO"` | Log-Level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `logging.file` | String | `"logs/trading_bot.log"` | Pfad zur Log-Datei |
| `logging.console` | Boolean | `true` | Logs in Konsole ausgeben |
| `notifications.enabled` | Boolean | `false` | Benachrichtigungen aktivieren |
| `notifications.email` | Boolean | `false` | E-Mail-Benachrichtigungen aktivieren |
| `notifications.email_address` | String | `""` | E-Mail-Adresse für Benachrichtigungen |
| `notifications.telegram` | Boolean | `false` | Telegram-Benachrichtigungen aktivieren |
| `notifications.telegram_token` | String | `""` | Telegram-Bot-Token |
| `notifications.telegram_chat_id` | String | `""` | Telegram-Chat-ID |