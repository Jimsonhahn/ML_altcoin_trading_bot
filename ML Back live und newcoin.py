#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Live Trading und Neue Coin Analyse Erweiterung für das ML-Backtest Framework.
Diese Erweiterung ermöglicht den Einsatz der ML-Komponenten im Paper und Live Trading
sowie die automatische Erkennung und Analyse neuer Coins.
"""

import os
import json
import time
import threading
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from collections import defaultdict
import schedule

# Trading Bot Komponenten
from config.settings import Settings
from core.trading_bot import TradingBot
from core.position import Position

# Import des ML-Backtest-Frameworks
from ml_backtest_framework import (
    MLBacktestFramework,
    MarketRegimeDetector,
    AssetClusterAnalyzer,
    MLBacktestOptimizer
)

# Logging einrichten
logger = logging.getLogger("LiveTradingExtension")


class MLTradingMonitor:
    """
    Überwacht kontinuierlich den Markt und erkennt Regimewechsel
    für den Einsatz der ML-Komponenten im Live Trading.
    """

    def __init__(self, base_config_path: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 monitoring_interval: int = 3600,  # Alle 1 Stunde
                 regime_check_interval: int = 21600,  # Alle 6 Stunden
                 new_coin_check_interval: int = 43200):  # Alle 12 Stunden
        """
        Initialisiert den ML Trading Monitor.

        Args:
            base_config_path: Pfad zur Basiskonfiguration
            output_dir: Ausgabeverzeichnis für Analysen und Logs
            monitoring_interval: Intervall für Marktüberwachung in Sekunden
            regime_check_interval: Intervall für Regime-Erkennung in Sekunden
            new_coin_check_interval: Intervall für Überprüfung neuer Coins in Sekunden
        """
        # Datum für Ausgabeverzeichnis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Basis-Konfiguration laden oder neue erstellen
        if base_config_path and os.path.exists(base_config_path):
            self.base_settings = Settings(base_config_path)
            logger.info(f"Basis-Konfiguration aus {base_config_path} geladen")
        else:
            self.base_settings = Settings()
            logger.info("Neue Basis-Konfiguration erstellt")

        # Ausgabeverzeichnis
        self.output_dir = output_dir or f"data/live_trading_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Ausgabeverzeichnis: {self.output_dir}")

        # ML Framework initialisieren
        self.ml_framework = MLBacktestFramework(base_config_path, self.output_dir)

        # Trading Bot Referenz (wird später gesetzt)
        self.trading_bot = None

        # Monitor-Einstellungen
        self.monitoring_interval = monitoring_interval
        self.regime_check_interval = regime_check_interval
        self.new_coin_check_interval = new_coin_check_interval

        # Monitor-Status
        self.running = False
        self.monitoring_thread = None
        self.last_regime_check = datetime.now() - timedelta(hours=24)  # Erzwinge ersten Check
        self.last_new_coin_check = datetime.now() - timedelta(hours=24)  # Erzwinge ersten Check

        # Marktdaten
        self.current_regime = None
        self.regime_history = []
        self.current_clusters = None
        self.coins_under_monitoring = {}  # Symbol -> Monitoring-Info
        self.new_coins_watchlist = {}  # Symbol -> Analyse-Info

        # Liste bekannter Coins (wird regelmäßig aktualisiert)
        self.known_coins = set()

        # Status-Datei-Pfad
        self.status_file = os.path.join(self.output_dir, "monitor_status.json")

        # Trading-Parameter
        self.optimized_params = {}  # Regime -> Parameter-Set
        self.parameter_update_time = datetime.now()

        # Initiale Regime-Daten laden, falls vorhanden
        self._load_regime_models()

        logger.info("ML Trading Monitor initialisiert")

    def connect_trading_bot(self, trading_bot: TradingBot) -> None:
        """
        Verbindet den Trading Bot mit dem ML Monitor.

        Args:
            trading_bot: TradingBot-Instanz
        """
        self.trading_bot = trading_bot

        # Callbacks registrieren
        trading_bot.add_status_update_callback(self._on_bot_status_update)
        trading_bot.add_trade_callback(self._on_bot_trade)
        trading_bot.add_error_callback(self._on_bot_error)

        logger.info("Trading Bot mit ML Monitor verbunden")

    def start(self) -> None:
        """
        Startet den ML-Trading-Monitor.
        """
        if self.running:
            logger.warning("ML Trading Monitor läuft bereits")
            return

        # Monitoring-Status setzen
        self.running = True

        # Initiale Marktanalyse
        logger.info("Führe initiale Marktanalyse durch...")
        try:
            # Bekannte Coins laden
            self._update_known_coins()

            # Marktregime erkennen
            self._check_market_regime()

            # Initial die Asset-Cluster analysieren
            self._analyze_asset_clusters()

        except Exception as e:
            logger.error(f"Fehler bei der initialen Marktanalyse: {e}")

        # Monitor-Thread starten
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="ML-Monitor-Thread",
            daemon=True
        )
        self.monitoring_thread.start()

        logger.info("ML Trading Monitor gestartet")

        # Status speichern
        self._save_status()

    def stop(self) -> None:
        """
        Stoppt den ML-Trading-Monitor.
        """
        if not self.running:
            logger.warning("ML Trading Monitor läuft nicht")
            return

        self.running = False

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)

        logger.info("ML Trading Monitor gestoppt")

        # Status speichern
        self._save_status()

    def _monitoring_loop(self) -> None:
        """
        Hauptüberwachungsschleife des ML Trading Monitors.
        """
        while self.running:
            try:
                # Aktuelle Zeit
                current_time = datetime.now()

                # 1. Prüfen, ob Regime-Check fällig ist
                if (current_time - self.last_regime_check).total_seconds() >= self.regime_check_interval:
                    logger.info("Führe geplanten Regime-Check durch...")
                    self._check_market_regime()
                    self.last_regime_check = current_time

                # 2. Prüfen, ob Neue-Coin-Check fällig ist
                if (current_time - self.last_new_coin_check).total_seconds() >= self.new_coin_check_interval:
                    logger.info("Führe geplante Überprüfung auf neue Coins durch...")
                    self._check_for_new_coins()
                    self.last_new_coin_check = current_time

                # 3. Coins unter Beobachtung aktualisieren
                self._update_monitored_coins()

                # 4. Status speichern
                self._save_status()

                # Pause bis zum nächsten Check
                time.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Fehler in der Monitoring-Schleife: {e}")
                # Kurze Pause und weitermachen
                time.sleep(60)

    def _check_market_regime(self) -> int:
        """
        Erkennt das aktuelle Marktregime und aktualisiert die Parameter.

        Returns:
            Aktuelle Regime-ID oder -1 bei Fehler
        """
        try:
            # Standard-Symbole für Regime-Erkennung
            symbols = self.base_settings.get('trading_pairs', ["BTC/USDT", "ETH/USDT"])

            # Falls wir einen Trading Bot haben, dessen Trading-Paare verwenden
            if self.trading_bot:
                symbols = self.trading_bot.trading_pairs

            # Regime-Erkennung durchführen
            logger.info(f"Starte Regime-Erkennung mit {len(symbols)} Symbolen...")

            # Wenn bereits ein Regime-Detektor trainiert wurde, kann dieser verwendet werden
            if hasattr(self.ml_framework.regime_detector,
                       'model_trained') and self.ml_framework.regime_detector.model_trained:
                # Marktdaten laden
                self.ml_framework.regime_detector.load_market_data(
                    symbols=symbols,
                    timeframe="1d",
                    start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                    end_date=datetime.now().strftime('%Y-%m-%d')
                )

                # Features für die aktuelle Marktlage extrahieren
                features_df = self.ml_framework.regime_detector.extract_market_features()

                if not features_df.empty:
                    # Aktuelles Regime vorhersagen (letzte Zeile des Feature-DataFrame)
                    latest_features = features_df.iloc[-1:].copy()
                    regime_id = self.ml_framework.regime_detector.predict_regime(latest_features)

                    if regime_id >= 0:
                        # Neues Regime erkannt
                        regime_label = self.ml_framework.regime_detector.regime_labels.get(regime_id,
                                                                                           f"Regime {regime_id}")

                        # Regime-Wechsel protokollieren
                        if self.current_regime != regime_id:
                            logger.info(
                                f"Regime-Wechsel erkannt: {self.current_regime} -> {regime_id} ({regime_label})")

                            # Regime-Historie aktualisieren
                            self.regime_history.append({
                                'from_regime': self.current_regime,
                                'to_regime': regime_id,
                                'timestamp': datetime.now().isoformat(),
                                'regime_label': regime_label
                            })

                            # Trading-Parameter aktualisieren
                            self._update_trading_parameters(regime_id)

                        # Aktuelles Regime aktualisieren
                        self.current_regime = regime_id
                        return regime_id

            # Wenn kein Modell trainiert wurde oder Vorhersage fehlgeschlagen ist,
            # vollständige Regime-Erkennung durchführen
            regimes = self.ml_framework.detect_market_regimes(
                n_regimes=3,
                symbols=symbols,
                timeframe="1d",
                start_date=(datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
                end_date=datetime.now().strftime('%Y-%m-%d')
            )

            if regimes:
                # Aktuelles Regime abrufen
                self.current_regime = self.ml_framework.current_regime

                if self.current_regime is not None:
                    # Trading-Parameter aktualisieren
                    self._update_trading_parameters(self.current_regime)

                    # Trading-Regeln extrahieren
                    trading_rules = self.ml_framework.regime_detector.extract_trading_rules()

                    # Regime-Info protokollieren
                    regime_label = regimes.get(self.current_regime, f"Regime {self.current_regime}")
                    logger.info(f"Aktuelles Marktregime: {self.current_regime} ({regime_label})")

                    # Regime-History aktualisieren
                    self.regime_history.append({
                        'from_regime': None,  # Kein vorheriges Regime bei Neutraining
                        'to_regime': self.current_regime,
                        'timestamp': datetime.now().isoformat(),
                        'regime_label': regime_label
                    })

                    return self.current_regime

            logger.error("Fehler bei der Regime-Erkennung")
            return -1

        except Exception as e:
            logger.error(f"Fehler bei der Überprüfung des Marktregimes: {e}")
            return -1

    def _analyze_asset_clusters(self) -> pd.DataFrame:
        """
        Analysiert die Asset-Cluster.

        Returns:
            DataFrame mit Cluster-Zuordnungen oder leerer DataFrame bei Fehler
        """
        try:
            # Trading-Paare
            symbols = self.base_settings.get('trading_pairs', ["BTC/USDT", "ETH/USDT"])

            # Falls wir einen Trading Bot haben, dessen Trading-Paare verwenden
            if self.trading_bot:
                all_symbols = []
                all_symbols.extend(self.trading_bot.trading_pairs)

                # Zusätzlich die Symbole aus der Watchlist hinzufügen, falls vorhanden
                watchlist = self.base_settings.get('watchlist', [])
                if watchlist:
                    all_symbols.extend(watchlist)

                # Neue Coins aus der Überwachung hinzufügen
                all_symbols.extend(list(self.new_coins_watchlist.keys()))

                # Duplikate entfernen
                symbols = list(set(all_symbols))

            # Asset-Clustering durchführen
            logger.info(f"Starte Asset-Cluster-Analyse mit {len(symbols)} Symbolen...")

            clusters = self.ml_framework.analyze_asset_clusters(
                n_clusters=None,  # Automatisch bestimmen
                symbols=symbols,
                timeframe="1d",
                start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                end_date=datetime.now().strftime('%Y-%m-%d')
            )

            if not clusters.empty:
                # Aktualisiere die Cluster-Informationen
                self.current_clusters = clusters

                # Portfolio-Empfehlung für aktuelles Regime
                if self.current_regime is not None:
                    # Empfohlene Assets basierend auf Clustering und aktuellem Regime
                    self._generate_portfolio_recommendation()

                logger.info(
                    f"Asset-Cluster-Analyse abgeschlossen: {len(clusters)} Assets in {len(clusters['cluster'].unique())} Clustern")
                return clusters

            logger.error("Asset-Cluster-Analyse fehlgeschlagen")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Fehler bei der Asset-Cluster-Analyse: {e}")
            return pd.DataFrame()

    def _update_known_coins(self) -> bool:
        """
        Aktualisiert die Liste bekannter Coins.

        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            # Daten-Verzeichnis durchsuchen
            binance_dir = os.path.join("data/market_data/binance")

            if not os.path.exists(binance_dir):
                logger.warning(f"Verzeichnis {binance_dir} existiert nicht")
                return False

            # Bekannte Coins aus Dateinamen extrahieren
            known_coins = set()

            for filename in os.listdir(binance_dir):
                if filename.endswith('.csv'):
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        symbol = f"{parts[0]}/{parts[1]}"
                        known_coins.add(symbol)

            # Bei Binance-API nach allen verfügbaren Symbolen fragen
            try:
                response = requests.get('https://api.binance.com/api/v3/exchangeInfo')
                if response.status_code == 200:
                    exchange_info = response.json()
                    symbols = exchange_info.get('symbols', [])

                    for symbol_info in symbols:
                        base_asset = symbol_info.get('baseAsset')
                        quote_asset = symbol_info.get('quoteAsset')

                        if quote_asset == 'USDT':
                            symbol = f"{base_asset}/{quote_asset}"
                            known_coins.add(symbol)
            except Exception as e:
                logger.warning(f"Fehler beim Abrufen der Exchange-Info: {e}")

            # Bekannte Coins aktualisieren
            self.known_coins = known_coins

            logger.info(f"{len(self.known_coins)} bekannte Coins aktualisiert")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Aktualisieren der bekannten Coins: {e}")
            return False

    def _check_for_new_coins(self) -> List[str]:
        """
        Überprüft, ob neue Coins am Markt verfügbar sind.

        Returns:
            Liste neu erkannter Coins
        """
        try:
            # Aktualisiere die Liste der bekannten Coins
            self._update_known_coins()

            # Bei der Binance-API nach aktuell handelbaren Symbolen fragen
            new_coins = []

            try:
                response = requests.get('https://api.binance.com/api/v3/ticker/24hr')
                if response.status_code == 200:
                    tickers = response.json()

                    for ticker in tickers:
                        symbol = ticker.get('symbol', '')

                        # Format in Base/Quote umwandeln (z.B. BTCUSDT -> BTC/USDT)
                        if symbol.endswith('USDT'):
                            base = symbol[:-4]
                            formatted_symbol = f"{base}/USDT"

                            # Prüfen, ob dieser Coin neu ist
                            if formatted_symbol not in self.known_coins:
                                # Weitere Filter anwenden (z.B. Mindestvolumen)
                                volume = float(ticker.get('volume', 0))

                                if volume > 1000:  # Mindestvolumen von 1000 Einheiten
                                    new_coins.append(formatted_symbol)
                                    logger.info(f"Neuer Coin erkannt: {formatted_symbol} (Volumen: {volume})")

                                    # Coin zur Watchlist hinzufügen
                                    self.new_coins_watchlist[formatted_symbol] = {
                                        'discovery_time': datetime.now().isoformat(),
                                        'initial_volume': volume,
                                        'analysis_status': 'pending',
                                        'data_available': False,
                                        'days_tracked': 0
                                    }
            except Exception as e:
                logger.warning(f"Fehler beim Abrufen der Ticker-Daten: {e}")

            if new_coins:
                logger.info(f"{len(new_coins)} neue Coins entdeckt: {', '.join(new_coins)}")

                # Führe eine erste Datensammlung für neue Coins durch
                self._collect_data_for_new_coins(new_coins)
            else:
                logger.info("Keine neuen Coins entdeckt")

            return new_coins

        except Exception as e:
            logger.error(f"Fehler bei der Überprüfung auf neue Coins: {e}")
            return []

    def _collect_data_for_new_coins(self, coins: List[str]) -> None:
        """
        Sammelt Daten für neu entdeckte Coins.

        Args:
            coins: Liste der neuen Coins
        """
        try:
            # Für jeden neuen Coin
            for coin in coins:
                try:
                    # Daten über die Binance-API abrufen
                    base, quote = coin.split('/')
                    symbol = f"{base}{quote}"

                    # Kline-Daten abrufen (max. 1000 Einträge)
                    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit=30"

                    response = requests.get(url)
                    if response.status_code == 200:
                        klines = response.json()

                        if klines:
                            # Daten in DataFrame umwandeln
                            df = pd.DataFrame(klines, columns=[
                                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                                'close_time', 'quote_asset_volume', 'number_of_trades',
                                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                            ])

                            # Datentypen konvertieren
                            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

                            for col in ['open', 'high', 'low', 'close', 'volume',
                                        'quote_asset_volume', 'taker_buy_base_asset_volume',
                                        'taker_buy_quote_asset_volume']:
                                df[col] = pd.to_numeric(df[col])

                            # Speicherpfad
                            binance_dir = os.path.join("data/market_data/binance")
                            os.makedirs(binance_dir, exist_ok=True)

                            # Dateiname generieren
                            current_date = datetime.now().strftime("%Y%m%d")
                            filename = f"{base}_{quote}_1d_{current_date}.csv"
                            filepath = os.path.join(binance_dir, filename)

                            # Daten speichern
                            df.to_csv(filepath, index=False)

                            logger.info(f"Daten für {coin} gesammelt und gespeichert: {len(df)} Einträge")

                            # Watchlist aktualisieren
                            if coin in self.new_coins_watchlist:
                                self.new_coins_watchlist[coin]['data_available'] = True
                                self.new_coins_watchlist[coin]['first_data_date'] = df['timestamp'].min().isoformat()
                                self.new_coins_watchlist[coin]['days_tracked'] = len(df)
                    else:
                        logger.warning(f"Fehler beim Abrufen der Kline-Daten für {coin}: {response.status_code}")

                except Exception as e:
                    logger.warning(f"Fehler bei der Datensammlung für {coin}: {e}")

        except Exception as e:
            logger.error(f"Fehler bei der Datensammlung für neue Coins: {e}")

    def _update_monitored_coins(self) -> None:
        """
        Aktualisiert den Status der überwachten neuen Coins und analysiert sie,
        sobald genügend Daten vorliegen.
        """
        try:
            # Coins, die analysiert werden können
            coins_to_analyze = []

            # Für jeden Coin in der Watchlist
            for coin, info in list(self.new_coins_watchlist.items()):
                # Status aktualisieren
                days_tracked = info.get('days_tracked', 0)
                analysis_status = info.get('analysis_status', 'pending')

                # Prüfen, ob Daten aktualisiert werden müssen
                if days_tracked < 7:  # Weniger als 7 Tage Daten
                    try:
                        # Daten sammeln
                        self._collect_data_for_new_coins([coin])

                        # Aktualisierte Anzahl an Tagen
                        if coin in self.new_coins_watchlist and self.new_coins_watchlist[coin].get('data_available',
                                                                                                   False):
                            days_tracked = self.new_coins_watchlist[coin].get('days_tracked', 0)
                    except Exception as e:
                        logger.warning(f"Fehler beim Update der Daten für {coin}: {e}")

                # Wenn genügend Daten vorliegen und noch nicht analysiert
                if days_tracked >= 3 and analysis_status == 'pending':
                    coins_to_analyze.append(coin)

            # Coins analysieren, wenn vorhanden
            if coins_to_analyze:
                logger.info(f"Analysiere {len(coins_to_analyze)} neue Coins: {', '.join(coins_to_analyze)}")
                for coin in coins_to_analyze:
                    self._analyze_new_coin(coin)

        except Exception as e:
            logger.error(f"Fehler bei der Aktualisierung überwachter Coins: {e}")

    def _analyze_new_coin(self, coin_symbol: str, min_days: int = 3) -> Dict[str, Any]:
        """
        Analysiert einen neuen Coin, sobald genügend Daten vorliegen.

        Args:
            coin_symbol: Symbol des Coins (z.B. 'BTC/USDT')
            min_days: Minimale Anzahl an Tagen für die Analyse

        Returns:
            Analyse-Ergebnisse oder leeres Dictionary bei Fehler
        """
        try:
            if coin_symbol not in self.new_coins_watchlist:
                logger.warning(f"Coin {coin_symbol} nicht in der Watchlist")
                return {}

            info = self.new_coins_watchlist[coin_symbol]

            # Prüfen, ob genügend Daten vorliegen
            if info.get('days_tracked', 0) < min_days:
                logger.warning(f"Nicht genügend Daten für {coin_symbol}: {info.get('days_tracked', 0)} Tage")
                return {"status": "insufficient_data", "days_available": info.get('days_tracked', 0)}

            # Daten laden
            base, quote = coin_symbol.split('/')

            # Pfad zu den Daten in der Binance-Verzeichnisstruktur
            binance_dir = os.path.join("data/market_data/binance")

            # Passende Datei finden
            csv_path = None
            for filename in os.listdir(binance_dir):
                if filename.startswith(f"{base}_{quote}_1d"):
                    csv_path = os.path.join(binance_dir, filename)
                    break

            if not csv_path:
                logger.warning(f"Keine Daten gefunden für {coin_symbol}")
                return {"status": "no_data_found"}

            # Daten laden
            df = pd.read_csv(csv_path)

            if 'timestamp' in df.columns:
                df['date'] = pd.to_datetime(df['timestamp'])
                df.set_index('date', inplace=True)

            # Features für diesen Coin extrahieren
            features = self._extract_coin_features(df, coin_symbol)

            # Ähnliche Coins finden
            similar_coins = []

            if not self.current_clusters is None:
                # 1. Vorheriges Clustering laden
                if self.ml_framework.cluster_analyzer.feature_data is not None:
                    # Feature-Matrix
                    X = self.ml_framework.cluster_analyzer.feature_data.copy()

                    if not X.empty:
                        # Neue Features normalisieren
                        X_scaled = self.ml_framework.cluster_analyzer.scaler.transform(X)

                        # Feature-Vektor für den neuen Coin
                        if all(f in X.columns for f in features.keys()):
                            coin_features = pd.DataFrame([list(features.values())], columns=list(features.keys()))
                            coin_features_scaled = self.ml_framework.cluster_analyzer.scaler.transform(coin_features)

                            # Ähnlichkeit berechnen (euklidische Distanz)
                            distances = np.sqrt(((X_scaled - coin_features_scaled) ** 2).sum(axis=1))

                            # Top-5 ähnlichste Coins
                            top_indices = np.argsort(distances)[:5]
                            similar_coins = [X.index[i] for i in top_indices]

                # 2. Cluster vorhersagen
                predicted_cluster = -1  # Ausreißer

                if hasattr(self.ml_framework.cluster_analyzer,
                           'cluster_model') and self.ml_framework.cluster_analyzer.cluster_model is not None:
                    # Feature-Vektor für den neuen Coin
                    if all(f in self.ml_framework.cluster_analyzer.feature_data.columns for f in features.keys()):
                        coin_features = pd.DataFrame([list(features.values())], columns=list(features.keys()))
                        coin_features_scaled = self.ml_framework.cluster_analyzer.scaler.transform(coin_features)

                        # Cluster vorhersagen
                        predicted_cluster = \
                        self.ml_framework.cluster_analyzer.cluster_model.predict(coin_features_scaled)[0]

            # Empfohlene Strategie für diesen Coin basierend auf Cluster
            if not self.current_clusters is None and predicted_cluster >= 0:
                # Assets im gleichen Cluster
                cluster_assets = self.current_clusters[
                    self.current_clusters['cluster'] == predicted_cluster].index.tolist()

                # Performance-Daten aus dem Cluster-Analyzer
                cluster_performance = self.ml_framework.cluster_analyzer.cluster_performances.get(predicted_cluster, {})

                recommended_strategy = "Neutral - Beobachten"

                # Basierend auf Cluster-Performance
                if cluster_performance:
                    mean_return = cluster_performance.get('mean_return', 0)
                    sharpe_ratio = cluster_performance.get('sharpe_ratio', 0)

                    if mean_return > 0.01 and sharpe_ratio > 0.5:
                        recommended_strategy = "Opportunistisches Long - Positive Cluster-Performance"
                    elif mean_return > 0:
                        recommended_strategy = "Vorsichtiges Long - Leicht positive Cluster-Performance"
                    elif mean_return < -0.01:
                        recommended_strategy = "Vermeiden - Negative Cluster-Performance"
            else:
                cluster_assets = []
                recommended_strategy = "Neutral - Neue Coin ohne Clusterzuordnung"
                predicted_cluster = -1

            # Watchlist aktualisieren
            self.new_coins_watchlist[coin_symbol]['analysis_status'] = 'analyzed'
            self.new_coins_watchlist[coin_symbol]['analysis_time'] = datetime.now().isoformat()
            self.new_coins_watchlist[coin_symbol]['similar_coins'] = similar_coins
            self.new_coins_watchlist[coin_symbol]['predicted_cluster'] = predicted_cluster
            self.new_coins_watchlist[coin_symbol]['recommended_strategy'] = recommended_strategy
            self.new_coins_watchlist[coin_symbol]['cluster_assets'] = cluster_assets

            # Analyseergebnis
            analysis_result = {
                "status": "analyzed",
                "coin": coin_symbol,
                "predicted_cluster": predicted_cluster,
                "similar_coins": similar_coins,
                "recommended_strategy": recommended_strategy,
                "cluster_assets": cluster_assets
            }

            logger.info(
                f"Analyse für {coin_symbol} abgeschlossen: Cluster {predicted_cluster}, Strategie: {recommended_strategy}")

            # Zum Monitoring hinzufügen
            self.coins_under_monitoring[coin_symbol] = {
                'first_analyzed': datetime.now().isoformat(),
                'days_monitored': 0,
                'last_update': datetime.now().isoformat(),
                'predicted_cluster': predicted_cluster,
                'recommended_strategy': recommended_strategy
            }

            # Speichere das Analyseergebnis in einer Datei
            analysis_dir = os.path.join(self.output_dir, "coin_analysis")
            os.makedirs(analysis_dir, exist_ok=True)

            with open(os.path.join(analysis_dir, f"{base}_{quote}_analysis.json"), 'w') as f:
                json.dump(analysis_result, f, indent=2, default=str)

            return analysis_result

        except Exception as e:
            logger.error(f"Fehler bei der Analyse des neuen Coins {coin_symbol}: {e}")
            return {"status": "error", "message": str(e)}

    def _extract_coin_features(self, df: pd.DataFrame, coin_symbol: str) -> Dict[str, float]:
        """
        Extrahiert Features für einen Coin aus dessen Marktdaten.

        Args:
            df: DataFrame mit historischen Daten
            coin_symbol: Symbol des Coins

        Returns:
            Dictionary mit Feature-Namen und -Werten
        """
        try:
            # Sicherstellen, dass wichtige Spalten vorhanden sind
            required_columns = ['close', 'high', 'low', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                logger.warning(f"Fehlende Spalten für {coin_symbol}: {missing_columns}")
                for col in missing_columns:
                    df[col] = 0

            # Returns berechnen
            df['return'] = df['close'].pct_change()

            # Basisfeatures
            features = {}

            # 1. Rendite-Statistiken
            features['mean_return'] = df['return'].mean()
            features['volatility'] = df['return'].std()
            features['skewness'] = df['return'].skew() if len(df) > 3 else 0
            features['kurtosis'] = df['return'].kurtosis() if len(df) > 3 else 0

            # 2. Volatilität
            features['volatility_10d'] = df['return'].rolling(min(10, len(df))).std().mean()

            # 3. True Range
            df['true_range'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            features['avg_true_range'] = df['true_range'].mean() / df['close'].mean()

            # 4. Volumen-Statistiken (wenn verfügbar)
            if 'volume' in df.columns:
                df['volume_change'] = df['volume'].pct_change()
                features['volume_mean'] = df['volume'].mean()
                features['volume_std'] = df['volume'].std()
                features['volume_change_mean'] = df['volume_change'].mean()
            else:
                features['volume_mean'] = 0
                features['volume_std'] = 0
                features['volume_change_mean'] = 0

            # 5. Trendstärke
            if len(df) >= 5:
                df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
                features['trend_strength'] = (df['close'].iloc[-1] / df['ema_5'].iloc[-1] - 1)
            else:
                features['trend_strength'] = 0

            if len(df) >= 10:
                df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
                features['ema_ratio'] = df['ema_5'].iloc[-1] / df['ema_10'].iloc[-1]
            else:
                features['ema_ratio'] = 1

            # 6. RSI
            if len(df) >= 14:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                features['rsi'] = df['rsi'].iloc[-1]
            else:
                features['rsi'] = 50  # Neutral

            # 7. Drawdown
            if len(df) >= 5:
                rolling_max = df['close'].rolling(min(len(df), 30)).max()
                drawdown = (df['close'] / rolling_max - 1) * 100
                features['max_drawdown'] = drawdown.min()
            else:
                features['max_drawdown'] = 0

            # 8. Return-to-Volatility Ratio (Sharpe-ähnlich)
            if features['volatility'] > 0:
                features['return_to_vol_ratio'] = features['mean_return'] / features['volatility']
            else:
                features['return_to_vol_ratio'] = 0

            return features

        except Exception as e:
            logger.error(f"Fehler bei der Feature-Extraktion für {coin_symbol}: {e}")
            return {}

    def _update_trading_parameters(self, regime_id: int) -> bool:
        """
        Aktualisiert die Trading-Parameter basierend auf dem aktuellen Regime.

        Args:
            regime_id: ID des aktuellen Marktregimes

        Returns:
            True, wenn Parameter aktualisiert wurden, sonst False
        """
        try:
            # Trading Bot prüfen
            if not self.trading_bot:
                logger.warning("Kein Trading Bot verbunden, kann Parameter nicht aktualisieren")
                return False

            # Optimierte Parameter abrufen
            params = None

            # 1. Zuerst in unserem Cache nachsehen
            if regime_id in self.optimized_params:
                params = self.optimized_params.get(regime_id)
                logger.info(f"Verwende optimierte Parameter aus Cache für Regime {regime_id}")

            # 2. Falls nicht verfügbar, im ML Framework nachsehen
            elif self.ml_framework.optimized_params:
                if self.ml_framework.optimized_params.get('regime_specific', False):
                    # Regime-spezifische Parameter
                    regime_params = self.ml_framework.optimized_params.get('params_by_regime', {})

                    if regime_id in regime_params:
                        params = regime_params[regime_id].get('params', {})

                        # Parameter im Cache speichern
                        self.optimized_params[regime_id] = params
                        logger.info(f"Optimierte Parameter für Regime {regime_id} aus ML Framework geladen")
                else:
                    # Allgemeine Parameter
                    params = self.ml_framework.optimized_params.get('params', {})
                    logger.info("Allgemeine optimierte Parameter aus ML Framework geladen")

            # 3. Falls immer noch keine Parameter, Strategie-Empfehlung präferieren
            if not params:
                # Trading-Regeln für dieses Regime
                trading_rules = self.ml_framework.regime_detector.extract_trading_rules()
                rule = trading_rules.get(regime_id, {})

                # Bei bestimmten Regime-Arten spezifische Parameter festlegen
                regime_label = self.ml_framework.regime_detector.regime_labels.get(regime_id, "").lower()

                # Parametervorschläge basierend auf Regime-Label
                if "bullish" in regime_label:
                    params = {
                        'technical.rsi.oversold': 40,
                        'technical.rsi.overbought': 75,
                        'risk.stop_loss': 0.05,
                        'risk.take_profit': 0.15
                    }
                    logger.info(f"Bullische Parameter für Regime {regime_id} festgelegt")
                elif "bearish" in regime_label:
                    params = {
                        'technical.rsi.oversold': 30,
                        'technical.rsi.overbought': 60,
                        'risk.stop_loss': 0.03,
                        'risk.take_profit': 0.08
                    }
                    logger.info(f"Bärische Parameter für Regime {regime_id} festgelegt")
                else:
                    # Neutrale/Standard-Parameter
                    params = {
                        'technical.rsi.oversold': 35,
                        'technical.rsi.overbought': 65,
                        'risk.stop_loss': 0.04,
                        'risk.take_profit': 0.10
                    }
                    logger.info(f"Neutrale Parameter für Regime {regime_id} festgelegt")

            # Parameter anwenden
            if params:
                # Strategie aktualisieren
                if hasattr(self.trading_bot.strategy, 'update_parameters'):
                    self.trading_bot.strategy.update_parameters(params)
                    logger.info(f"Parameter für Strategie aktualisiert: {params}")
                else:
                    # Alternativ: Einstellungen direkt aktualisieren
                    for key, value in params.items():
                        self.trading_bot.settings.set(key, value)

                    logger.info(f"Bot-Einstellungen aktualisiert: {params}")

                # Aktualisierungszeit merken
                self.parameter_update_time = datetime.now()

                return True

            logger.warning(f"Keine Parameter für Regime {regime_id} gefunden")
            return False

        except Exception as e:
            logger.error(f"Fehler bei der Aktualisierung der Trading-Parameter: {e}")
            return False

    def _generate_portfolio_recommendation(self) -> Dict[str, Any]:
        """
        Generiert eine Portfolio-Empfehlung basierend auf aktuellem Regime und Clustering.

        Returns:
            Dictionary mit Portfolio-Empfehlung
        """
        try:
            if self.current_regime is None or self.current_clusters is None:
                logger.warning("Keine Regime- oder Cluster-Daten für Portfolio-Empfehlung verfügbar")
                return {}

            # Trading-Regeln für das aktuelle Regime
            trading_rules = self.ml_framework.regime_detector.extract_trading_rules()
            rule = trading_rules.get(self.current_regime, {})

            # Regime-Label
            regime_label = self.ml_framework.regime_detector.regime_labels.get(
                self.current_regime, f"Regime {self.current_regime}"
            )

            # Portfolio-Allokation aus der Trading-Regel
            portfolio_allocation = rule.get('portfolio_allocation', {})

            # Top-Performer aus der Trading-Regel
            top_performers = rule.get('top_performers', {})

            # Beste Cluster identifizieren
            if not self.ml_framework.cluster_analyzer.cluster_performances:
                logger.warning("Keine Cluster-Performance-Daten verfügbar")
                return {}

            # Nach Sharpe Ratio sortierte Cluster
            sorted_clusters = sorted(
                self.ml_framework.cluster_analyzer.cluster_performances.items(),
                key=lambda x: x[1].get('sharpe_ratio', 0),
                reverse=True
            )

            # Beste Assets aus den besten Clustern sammeln
            recommended_assets = []

            # 1. Zuerst aus dem aktuellen Regime
            for asset in top_performers.keys():
                if asset not in recommended_assets:
                    recommended_assets.append(asset)

            # 2. Dann aus den besten Clustern
            for cluster_id, stats in sorted_clusters:
                if stats.get('mean_return', 0) > 0:  # Nur positiv performende Cluster
                    # Repräsentatives Asset
                    rep_asset = stats.get('representative_asset')
                    if rep_asset and rep_asset not in recommended_assets:
                        recommended_assets.append(rep_asset)

                    # Weitere Assets aus diesem Cluster
                    for asset in stats.get('assets', [])[:3]:  # Top-3 aus jedem Cluster
                        if asset not in recommended_assets:
                            recommended_assets.append(asset)

            # Auf 10 Assets beschränken
            recommended_assets = recommended_assets[:10]

            # Empfohlene Strategie
            strategy = rule.get('recommended_strategy', "Keine spezifische Empfehlung")

            # Portfolio-Empfehlung erstellen
            recommendation = {
                'timestamp': datetime.now().isoformat(),
                'current_regime': self.current_regime,
                'regime_label': regime_label,
                'strategy': strategy,
                'portfolio_allocation': portfolio_allocation,
                'recommended_assets': recommended_assets,
                'full_market_assessment': rule.get('trading_advice', "")
            }

            # In Datei speichern
            os.makedirs(os.path.join(self.output_dir, "portfolio"), exist_ok=True)
            with open(os.path.join(self.output_dir, "portfolio", "current_recommendation.json"), 'w') as f:
                json.dump(recommendation, f, indent=2, default=str)

            logger.info(f"Portfolio-Empfehlung generiert: {len(recommended_assets)} Assets in Regime {regime_label}")
            return recommendation

        except Exception as e:
            logger.error(f"Fehler bei der Generierung der Portfolio-Empfehlung: {e}")
            return {}

    def _load_regime_models(self) -> bool:
        """
        Lädt bereits trainierte Regime-Modelle, falls vorhanden.

        Returns:
            True, wenn Modelle geladen wurden, sonst False
        """
        try:
            # Mögliche Speicherorte für das Modell
            model_paths = [
                os.path.join(self.output_dir, "regime_analysis", "regime_model.pkl"),
                "data/ml_models/regime_model.pkl"
            ]

            for path in model_paths:
                if os.path.exists(path):
                    # Modell laden
                    success = self.ml_framework.regime_detector.load_model(path)

                    if success:
                        logger.info(f"Regime-Modell aus {path} geladen")
                        return True

            logger.info("Kein gespeichertes Regime-Modell gefunden")
            return False

        except Exception as e:
            logger.error(f"Fehler beim Laden der Regime-Modelle: {e}")
            return False

    def _save_status(self) -> bool:
        """
        Speichert den aktuellen Status des ML Trading Monitors.

        Returns:
            True, wenn Status gespeichert wurde, sonst False
        """
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'running': self.running,
                'current_regime': self.current_regime,
                'regime_history': self.regime_history,
                'coins_under_monitoring': self.coins_under_monitoring,
                'new_coins_watchlist': self.new_coins_watchlist,
                'known_coins_count': len(self.known_coins),
                'parameter_update_time': self.parameter_update_time.isoformat()
            }

            # Regime-Label hinzufügen, falls verfügbar
            if self.current_regime is not None and hasattr(self.ml_framework.regime_detector, 'regime_labels'):
                status['current_regime_label'] = self.ml_framework.regime_detector.regime_labels.get(
                    self.current_regime, f"Regime {self.current_regime}"
                )

            # In Datei speichern
            with open(self.status_file, 'w') as f:
                json.dump(status, f, indent=2, default=str)

            return True

        except Exception as e:
            logger.error(f"Fehler beim Speichern des Monitor-Status: {e}")
            return False

    def _on_bot_status_update(self, status: Dict[str, Any]) -> None:
        """
        Callback für Statusaktualisierungen des Trading Bots.

        Args:
            status: Status-Dictionary vom Trading Bot
        """
        try:
            # Marktregime und Parameteraktualisierung prüfen
            # (nur alle 6 Stunden, um nicht ständig zu wechseln)
            if (datetime.now() - self.last_regime_check).total_seconds() >= self.regime_check_interval:
                self._check_market_regime()
                self.last_regime_check = datetime.now()

            # Portfolio-Empfehlung aktualisieren
            if self.current_regime is not None and self.current_clusters is not None:
                self._generate_portfolio_recommendation()

        except Exception as e:
            logger.error(f"Fehler im Bot-Status-Callback: {e}")

    def _on_bot_trade(self, position: Position) -> None:
        """
        Callback für Trade-Events des Trading Bots.

        Args:
            position: Position-Objekt des Trades
        """
        try:
            # Trade protokollieren
            trade_dir = os.path.join(self.output_dir, "trades")
            os.makedirs(trade_dir, exist_ok=True)

            trade_info = {
                'timestamp': datetime.now().isoformat(),
                'symbol': position.symbol,
                'side': position.side,
                'amount': position.amount,
                'entry_price': position.entry_price,
                'exit_price': position.exit_price if hasattr(position, 'exit_price') else None,
                'profit_loss': position.profit_loss if hasattr(position, 'profit_loss') else None,
                'profit_loss_percent': position.profit_loss_percent if hasattr(position,
                                                                               'profit_loss_percent') else None,
                'current_regime': self.current_regime,
                'regime_label': self.ml_framework.regime_detector.regime_labels.get(
                    self.current_regime, f"Regime {self.current_regime}"
                ) if self.current_regime is not None else None
            }

            # Trade-ID für den Dateinamen
            trade_id = position.id if hasattr(position, 'id') else f"{int(time.time())}"

            # In Datei speichern
            with open(os.path.join(trade_dir, f"trade_{trade_id}.json"), 'w') as f:
                json.dump(trade_info, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Fehler im Trade-Callback: {e}")

    def _on_bot_error(self, error_type: str, error_message: str) -> None:
        """
        Callback für Fehler-Events des Trading Bots.

        Args:
            error_type: Typ des Fehlers
            error_message: Fehlermeldung
        """
        try:
            # Fehler protokollieren
            error_dir = os.path.join(self.output_dir, "errors")
            os.makedirs(error_dir, exist_ok=True)

            error_info = {
                'timestamp': datetime.now().isoformat(),
                'error_type': error_type,
                'error_message': error_message,
                'current_regime': self.current_regime,
                'regime_label': self.ml_framework.regime_detector.regime_labels.get(
                    self.current_regime, f"Regime {self.current_regime}"
                ) if self.current_regime is not None else None
            }

            # In Datei speichern
            with open(os.path.join(error_dir, f"error_{int(time.time())}.json"), 'w') as f:
                json.dump(error_info, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Fehler im Error-Callback: {e}")

    def generate_trading_report(self) -> str:
        """
        Erstellt einen umfassenden Trading-Bericht mit allen ML-Komponenten.

        Returns:
            Pfad zum Bericht oder leerer String bei Fehler
        """
        try:
            # Berichtspfad
            report_path = os.path.join(self.output_dir, "live_trading_report.html")

            # Aktuelle Empfehlung
            recommendation = self._generate_portfolio_recommendation()

            # HTML-Bericht erstellen
            with open(report_path, 'w') as f:
                f.write("""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Live Trading ML Bericht</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; }
                        h1, h2, h3 { color: #333; }
                        .container { max-width: 1200px; margin: 0 auto; }
                        .summary-box { 
                            border: 1px solid #ddd; 
                            border-radius: 5px; 
                            padding: 15px; 
                            margin-bottom: 20px;
                            background-color: #f9f9f9;
                        }
                        .alert-box {
                            border: 1px solid #f8d7da;
                            border-radius: 5px;
                            padding: 15px;
                            margin-bottom: 20px;
                            background-color: #f8d7da;
                            color: #721c24;
                        }
                        .success-box {
                            border: 1px solid #d4edda;
                            border-radius: 5px;
                            padding: 15px;
                            margin-bottom: 20px;
                            background-color: #d4edda;
                            color: #155724;
                        }
                        .info-box {
                            border: 1px solid #d1ecf1;
                            border-radius: 5px;
                            padding: 15px;
                            margin-bottom: 20px;
                            background-color: #d1ecf1;
                            color: #0c5460;
                        }
                        .results-table {
                            width: 100%;
                            border-collapse: collapse;
                        }
                        .results-table th, .results-table td {
                            border: 1px solid #ddd;
                            padding: 8px;
                            text-align: left;
                        }
                        .results-table th {
                            background-color: #f2f2f2;
                        }
                        .allocation-chart {
                            width: 100%;
                            height: 30px;
                            background-color: #eee;
                            margin: 10px 0;
                            border-radius: 5px;
                            overflow: hidden;
                        }
                        .allocation-segment {
                            height: 100%;
                            float: left;
                            text-align: center;
                            color: white;
                            font-weight: bold;
                            line-height: 30px;
                            font-size: 12px;
                        }
                        .asset-badge {
                            display: inline-block;
                            padding: 4px 10px;
                            margin: 3px;
                            background-color: #007bff;
                            color: white;
                            border-radius: 15px;
                            font-size: 14px;
                        }
                        .refresh-info {
                            text-align: right;
                            font-size: 12px;
                            color: #666;
                            margin-bottom: 20px;
                        }
                    </style>
                    <meta http-equiv="refresh" content="300">
                </head>
                <body>
                    <div class="container">
                        <h1>Live Trading ML Bericht</h1>
                        <p class="refresh-info">Automatische Aktualisierung alle 5 Minuten. Letzte Aktualisierung: %s</p>
                """ % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

                # Abschnitt: Aktuelles Marktregime
                if self.current_regime is not None:
                    regime_label = self.ml_framework.regime_detector.regime_labels.get(
                        self.current_regime, f"Regime {self.current_regime}"
                    )

                    f.write(f"""
                        <div class="info-box">
                            <h2>Aktuelles Marktregime</h2>
                            <p><strong>Regime:</strong> {regime_label}</p>
                    """)

                    # Trading-Regeln für dieses Regime
                    trading_rules = self.ml_framework.regime_detector.extract_trading_rules()
                    rule = trading_rules.get(self.current_regime, {})

                    if rule:
                        trading_advice = rule.get('recommended_strategy', "Keine spezifische Empfehlung")
                        f.write(f"""
                            <p><strong>Empfohlene Strategie:</strong> {trading_advice}</p>
                        """)

                    f.write("""
                        </div>
                    """)

                # Abschnitt: Aktuelle Parameter
                if self.trading_bot and hasattr(self.trading_bot.strategy, 'get_parameters'):
                    f.write("""
                        <div class="summary-box">
                            <h2>Aktuelle Trading-Parameter</h2>
                            <table class="results-table">
                                <tr>
                                    <th>Parameter</th>
                                    <th>Wert</th>
                                </tr>
                    """)

                    for key, value in self.trading_bot.strategy.get_parameters().items():
                        f.write(f"""
                            <tr>
                                <td>{key}</td>
                                <td>{value}</td>
                            </tr>
                        """)

                    f.write("""
                            </table>
                            <p><strong>Letzte Aktualisierung:</strong> %s</p>
                        </div>
                    """ % self.parameter_update_time.strftime("%Y-%m-%d %H:%M:%S"))

                # Abschnitt: Portfolio-Empfehlung
                if recommendation:
                    f.write("""
                        <div class="summary-box">
                            <h2>Portfolio-Empfehlung</h2>
                    """)

                    # Portfolio-Allokation
                    if 'portfolio_allocation' in recommendation:
                        f.write("""
                            <h3>Empfohlene Asset-Allokation</h3>
                            <div class="allocation-chart">
                        """)

                        # Farben für verschiedene Asset-Typen
                        colors = {
                            "altcoins": "#6f42c1",  # Lila
                            "bitcoin": "#f8b739",  # Bitcoin-Orange
                            "stablecoins": "#20c997"  # Grün
                        }

                        for asset, percentage in recommendation['portfolio_allocation'].items():
                            width = percentage * 100
                            f.write(f"""
                                <div class="allocation-segment" 
                                     style="width: {width}%; background-color: {colors.get(asset, '#007bff')};">
                                    {asset.capitalize()} {percentage * 100:.0f}%
                                </div>
                            """)

                        f.write("""
                            </div>
                        """)

                    # Empfohlene Assets
                    if 'recommended_assets' in recommendation and recommendation['recommended_assets']:
                        f.write("""
                            <h3>Empfohlene Trading-Assets</h3>
                            <div>
                        """)

                        for asset in recommendation['recommended_assets']:
                            f.write(f'<span class="asset-badge">{asset}</span> ')

                        f.write("""
                            </div>
                        """)

                    f.write("""
                        </div>
                    """)

                < h2 > Neue
                Coins
                unter
                Beobachtung < / h2 >
                < table

                class ="results-table" >

                < tr >
                < th > Coin < / th >
                < th > Entdeckt < / th >
                < th > Status < / th >
                < th > Cluster < / th >
                < th > Ähnlich
                zu < / th >
                < th > Empfehlung < / th >
            < / tr >

    """)

    # Neue Coins auflisten
    for coin, info in sorted(self.new_coins_watchlist.items(), 
                          key=lambda x: x[1].get('discovery_time', ''), 
                          reverse=True):
        discovery_time = datetime.fromisoformat(info.get('discovery_time', datetime.now().isoformat()))
        status = info.get('analysis_status', 'pending')
        cluster = info.get('predicted_cluster', -1)
        similar_coins = ', '.join(info.get('similar_coins', [])[:3])
        recommendation = info.get('recommended_strategy', 'Noch nicht analysiert')

        # Status-Farbe
        status_style = ""
        if status == 'analyzed':
            status_style = 'background-color: #d4edda;'
        elif status == 'pending':
            status_style = 'background-color: #fff3cd;'

        f.write(f"""
    < tr >
    < td > {coin} < / td >
    < td > {discovery_time.strftime('%Y-%m-%d')} < / td >
    < td
    style = "{status_style}" > {status} < / td >
    < td > {cluster if cluster >= 0 else 'N/A'} < / td >
    < td > {similar_coins} < / td >
    < td > {recommendation} < / td >

< / tr >
""")

f.write("""
< / table >
< / div >
""")

# Abschnitt: Regime-Historie
if self.regime_history:
f.write("""
< div


class ="summary-box" >

< h2 > Regime - Wechsel
Historik < / h2 >
< table


class ="results-table" >

< tr >
< th > Zeitpunkt < / th >
< th > Von < / th >
< th > Zu < / th >
< th > Neues
Regime < / th >
< / tr >
""")

# Regime-Wechsel auflisten (neueste zuerst)
for entry in sorted(self.regime_history, 
                 key=lambda x: x.get('timestamp', ''), 
                 reverse=True):
    timestamp = datetime.fromisoformat(entry.get('timestamp', datetime.now().isoformat()))
    from_regime = entry.get('from_regime')
    to_regime = entry.get('to_regime')
    regime_label = entry.get('regime_label', f"Regime {to_regime}")

    f.write(f"""
< tr >
< td > {timestamp.strftime('%Y-%m-%d %H:%M')} < / td >
< td > {from_regime if from_regime is not None else 'Initial'} < / td >
< td > {to_regime} < / td >
< td > {regime_label} < / td >
< / tr >
""")

f.write("""
< / table >
< / div >
""")

# Ende des HTML-Dokuments
f.write("""
< / div >
< / body >
< / html >
""")

logger.info(f"Trading-Bericht erstellt: {report_path}")
return report_path

except Exception as e:
logger.error(f"Fehler bei der Erstellung des Trading-Berichts: {e}")
return ""


class MLTradingExtension:
"""
Erweiterung
für
den
Trading
Bot
zur
Integration
der
ML - Komponenten
im
Paper
und
Live
Trading.
"""

def __init__(self, trading_bot: TradingBot,
           base_config_path: str = None,
           output_dir: str = None):
    """
Initialisiert
die
ML - Trading - Erweiterung.

Args:
trading_bot: TradingBot - Instanz
base_config_path: Pfad
zur
Basiskonfiguration
output_dir: Ausgabeverzeichnis
für
Analysen
und
Logs
"""
self.trading_bot = trading_bot

# ML-Monitor erstellen
self.ml_monitor = MLTradingMonitor(
    base_config_path=base_config_path,
    output_dir=output_dir
)

# Monitor mit dem Trading Bot verbinden
self.ml_monitor.connect_trading_bot(trading_bot)

logger.info("ML-Trading-Erweiterung initialisiert")

def start(self) -> None:
"""
Startet
die
ML - Trading - Erweiterung.
"""
# ML-Monitor starten
self.ml_monitor.start()

# Bot-Strategie mit ML-Parametern starten
self._setup_bot_strategy()

logger.info("ML-Trading-Erweiterung gestartet")

def stop(self) -> None:
"""
Stoppt
die
ML - Trading - Erweiterung.
"""
# ML-Monitor stoppen
self.ml_monitor.stop()

logger.info("ML-Trading-Erweiterung gestoppt")

def _setup_bot_strategy(self) -> None:
"""
Richtet
die
Bot - Strategie
mit
den
ML - Parametern
ein.
"""
try:
    # Prüfen, ob ein aktuelles Regime erkannt wurde
    if self.ml_monitor.current_regime is not None:
        # Trading-Parameter aktualisieren
        self.ml_monitor._update_trading_parameters(self.ml_monitor.current_regime)

    # Strategie-Parameter-Update-Methode patchen, falls nicht vorhanden
    strategy = self.trading_bot.strategy

    if not hasattr(strategy, 'update_parameters'):
        def update_parameters(self, params):
            """
Aktualisiert
die
Strategie - Parameter.

Args:
params: Dictionary
mit
Parameternamen
und - werten
"""
for key, value in params.items():
    # Parameter in der Strategie oder Settings setzen
    if hasattr(self, key.split('.')[-1]):
        setattr(self, key.split('.')[-1], value)
    else:
        # Über die Settings setzen
        self.settings.set(key, value)

# Methode zur Strategie hinzufügen
import types
strategy.update_parameters = types.MethodType(update_parameters, strategy)

if not hasattr(strategy, 'get_parameters'):
def get_parameters(self):
"""
Gibt
die
aktuellen
Strategie - Parameter
zurück.

Returns:
Dictionary
mit
Parameternamen
und - werten
"""
params = {}

# Strategie-spezifische Parameter
if hasattr(self, 'settings'):
    # Aus den Einstellungen
    all_settings = self.settings.get_all()

    # Relevante Parameter extrahieren
    prefixes = ['technical.', 'risk.', 'trade.']
    for key, value in all_settings.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                params[key] = value

return params

# Methode zur Strategie hinzufügen
import types
strategy.get_parameters = types.MethodType(get_parameters, strategy)

logger.info("Bot-Strategie mit ML-Parametern eingerichtet")

except Exception as e:
logger.error(f"Fehler beim Einrichten der Bot-Strategie: {e}")

def generate_report(self) -> str:
"""
Erstellt
einen
aktuellen
Trading - Bericht.

Returns:
Pfad
zum
Bericht
"""
return self.ml_monitor.generate_trading_report()


# Hauptfunktion für die direkte Ausführung
def main():
"""
Hauptfunktion
für
den
Kommandozeilenaufruf.
"""
                    import argparse

                    parser = argparse.ArgumentParser(description="ML Trading Extension für den Altcoin Trading Bot")

                    # Konfigurationsoptionen
                    parser.add_argument("--config", "-c", type=str, help="Pfad zur Basiskonfiguration")
                    parser.add_argument("--output", "-o", type=str, help="Ausgabeverzeichnis")
                    parser.add_argument("--bot-mode", type=str, default="paper", 
                                      choices=["paper", "live", "backtest"],
                                      help="Trading-Bot-Modus")
                    parser.add_argument("--strategy", "-s", type=str, default="momentum",
                                      help="Zu verwendende Strategie")

                    # Monitoring-Optionen
                    parser.add_argument("--no-monitor", action="store_true",
                                      help="Monitoring deaktivieren")
                    parser.add_argument("--monitor-interval", type=int, default=3600,
                                      help="Monitoring-Intervall in Sekunden")
                    parser.add_argument("--regime-interval", type=int, default=21600,
                                      help="Regime-Check-Intervall in Sekunden")
                    parser.add_argument("--new-coin-interval", type=int, default=43200,
                                      help="Neue-Coin-Check-Intervall in Sekunden")

                    # Reporting-Optionen
                    parser.add_argument("--generate-report", action="store_true",
                                      help="Nur Bericht generieren und beenden")

                    # Argumente parsen
                    args = parser.parse_args()

                    # Konfiguration laden
                    settings = Settings(args.config) if args.config else Settings()

                    # TradingBot erstellen
                    bot = TradingBot(
                        mode=args.bot_mode,
                        strategy_name=args.strategy,
                        settings=settings
                    )

                    # ML-Trading-Erweiterung erstellen
                    extension = MLTradingExtension(
                        trading_bot=bot,
                        base_config_path=args.config,
                        output_dir=args.output
                    )

                    # Nur Bericht generieren und beenden
                    if args.generate_report:
                        report_path = extension.generate_report()
                        print(f"Bericht erstellt: {report_path}")
                        return

                    # Monitoring starten
                    if not args.no_monitor:
                        # Monitor-Intervalle anpassen
                        extension.ml_monitor.monitoring_interval = args.monitor_interval
                        extension.ml_monitor.regime_check_interval = args.regime_interval
                        extension.ml_monitor.new_coin_check_interval = args.new_coin_interval

                        # Erweiterung starten
                        extension.start()

                    # Trading Bot starten
                    if args.bot_mode in ["paper", "live"]:
                        bot.run_in_thread()

                        try:
                            # Hauptthread am Leben halten
                            while True:
                                time.sleep(1)
                        except KeyboardInterrupt:
                            print("Beende Trading Bot...")
                            bot.stop()

                            if not args.no_monitor:
                                extension.stop()
                    else:
                        print("Backtest-Modus: Verwende 'run_backtests.py' für Backtests.")


                if __name__ == "__main__":
                    main()