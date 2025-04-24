#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hauptmodul für die Trading-Logik des Altcoin Trading Bots mit integrierter ML.
Dieses Modul enthält die erweiterte TradingBot-Klasse und die
zugehörigen Hilfsfunktionen für das Trading-System.
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, List, Tuple, Union
import os
import json
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import requests
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from config.settings import Settings
from core.exchange import ExchangeFactory
from core.position import Position, PositionManager
from strategies.strategy_base import Strategy
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.ml_strategy import MLStrategy
from utils.logger import setup_logger
from data_sources import DataManager


class MarketRegimeDetector:
    """
    Erkennt und klassifiziert Marktregimes basierend auf historischen Daten.
    Regime können bullish, bearish, volatil, stabil, altcoin-freundlich etc. sein.
    """

    def __init__(self, data_dir: str = "data/market_data", n_regimes: int = 3):
        """
        Initialisiert den Marktregime-Detektor.

        Args:
            data_dir: Verzeichnis mit historischen Marktdaten
            n_regimes: Anzahl der zu identifizierenden Regimes
        """
        self.data_dir = data_dir
        self.n_regimes = n_regimes
        self.market_data = {}
        self.regime_model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.regime_performances = None
        self.current_regime = None
        self.model_trained = False
        self.regime_labels = {}
        self.regime_transitions = None
        self.logger = logging.getLogger(__name__)

    def load_market_data(self, symbols: List[str] = None,
                         data_manager: Optional[DataManager] = None,
                         timeframe: str = "1d",
                         start_date: str = None,
                         end_date: str = None) -> bool:
        """
        Lädt Marktdaten für die angegebenen Symbole.

        Args:
            symbols: Liste der zu ladenden Symbole (oder None für alle)
            data_manager: Optionaler DataManager zur Datenabfrage
            timeframe: Zeitrahmen der Daten
            start_date: Startdatum im Format 'YYYY-MM-DD'
            end_date: Enddatum im Format 'YYYY-MM-DD'

        Returns:
            True, wenn Daten erfolgreich geladen wurden
        """
        try:
            # Wenn kein Startdatum angegeben, letzten Monat verwenden
            if not start_date:
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

            # Wenn kein Enddatum angegeben, aktuelles Datum verwenden
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')

            # Wenn DataManager bereitgestellt wurde, diesen für das Laden verwenden
            if data_manager:
                for symbol in symbols:
                    df = data_manager.get_historical_data(
                        symbol=symbol,
                        source="binance",  # Könnte aus Einstellungen kommen
                        timeframe=timeframe,
                        start_date=datetime.strptime(start_date, '%Y-%m-%d'),
                        end_date=datetime.strptime(end_date, '%Y-%m-%d'),
                        use_cache=True
                    )

                    if not df.empty:
                        self.market_data[symbol] = df
                        self.logger.info(f"Daten für {symbol} geladen: {len(df)} Einträge")

                if not self.market_data:
                    self.logger.error("Keine Marktdaten geladen")
                    return False

                return True

            # Alternativ: Direkt aus dem Verzeichnis laden
            binance_dir = os.path.join(self.data_dir, "binance")

            # Wenn keine Symbole angegeben, alle verfügbaren laden
            if not symbols:
                symbols = []
                for filename in os.listdir(binance_dir):
                    if filename.endswith(f"_{timeframe}.csv") or filename.endswith(f"_{timeframe}_20230101.csv"):
                        symbol = filename.split("_")[0]
                        quote = filename.split("_")[1]
                        symbols.append(f"{symbol}/{quote}")

            # Daten für jedes Symbol laden
            for symbol in symbols:
                base, quote = symbol.split("/")
                filename_pattern = f"{base}_{quote}_{timeframe}"

                # Datei suchen
                csv_path = None
                for f in os.listdir(binance_dir):
                    if f.startswith(filename_pattern):
                        csv_path = os.path.join(binance_dir, f)
                        break

                if not csv_path:
                    self.logger.warning(f"Keine Daten gefunden für {symbol} mit Timeframe {timeframe}")
                    continue

                # Daten laden
                df = pd.read_csv(csv_path)

                # Datum als Index setzen, falls vorhanden
                if 'timestamp' in df.columns:
                    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('date', inplace=True)
                elif 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                elif 'time' in df.columns:
                    df['date'] = pd.to_datetime(df['time'])
                    df.set_index('date', inplace=True)

                # Datum filtern, falls angegeben
                if start_date and end_date:
                    df = df[(df.index >= start_date) & (df.index <= end_date)]

                # Daten speichern
                self.market_data[symbol] = df
                self.logger.info(f"Daten für {symbol} geladen: {len(df)} Einträge")

            if not self.market_data:
                self.logger.error("Keine Marktdaten geladen")
                return False

            self.logger.info(f"Marktdaten für {len(self.market_data)} Symbole geladen")
            return True

        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Marktdaten: {e}")
            return False

    def extract_market_features(self) -> pd.DataFrame:
        """
        Extrahiert relevante Features für die Regime-Erkennung.

        Returns:
            DataFrame mit täglichen Features für alle Märkte
        """
        if not self.market_data:
            self.logger.error("Keine Marktdaten geladen")
            return pd.DataFrame()

        try:
            # Bitcoin-Daten als Referenz (sollte vorhanden sein)
            btc_symbol = next((s for s in self.market_data.keys() if s.startswith('BTC')), None)

            if not btc_symbol:
                self.logger.error("Keine Bitcoin-Daten verfügbar")
                return pd.DataFrame()

            btc_df = self.market_data[btc_symbol]

            # Leerer DataFrame mit Datum als Index
            features_df = pd.DataFrame(index=btc_df.index)

            # 1. Bitcoin-spezifische Features (als Marktindikator)
            # Tägliche Returns
            btc_df['return'] = btc_df['close'].pct_change()

            # Volatilität (20-Tage Rolling Std)
            btc_df['volatility_20d'] = btc_df['return'].rolling(20).std()

            # Relative Stärke zum EMA
            btc_df['ema_20'] = btc_df['close'].ewm(span=20, adjust=False).mean()
            btc_df['ema_50'] = btc_df['close'].ewm(span=50, adjust=False).mean()
            btc_df['ema_200'] = btc_df['close'].ewm(span=200, adjust=False).mean()

            btc_df['rel_to_ema20'] = btc_df['close'] / btc_df['ema_20'] - 1
            btc_df['rel_to_ema50'] = btc_df['close'] / btc_df['ema_50'] - 1
            btc_df['rel_to_ema200'] = btc_df['close'] / btc_df['ema_200'] - 1

            # Volumen-basierte Features
            if 'volume' in btc_df.columns:
                btc_df['volume_change'] = btc_df['volume'].pct_change()
                btc_df['volume_ma_ratio'] = btc_df['volume'] / btc_df['volume'].rolling(20).mean()

            # Trend Strength
            btc_df['high_low_diff'] = btc_df['high'] - btc_df['low']
            btc_df['high_close_diff'] = abs(btc_df['high'] - btc_df['close'].shift(1))
            btc_df['low_close_diff'] = abs(btc_df['low'] - btc_df['close'].shift(1))

            conditions = [
                (btc_df['high_close_diff'] > btc_df['low_close_diff']) &
                (btc_df['high_close_diff'] > 0)
            ]
            choices = [btc_df['high_close_diff']]
            btc_df['dm_plus'] = np.select(conditions, choices, default=0)

            conditions = [
                (btc_df['low_close_diff'] > btc_df['high_close_diff']) &
                (btc_df['low_close_diff'] > 0)
            ]
            choices = [btc_df['low_close_diff']]
            btc_df['dm_minus'] = np.select(conditions, choices, default=0)

            # Average Directional Index
            atr_period = 14
            btc_df['tr'] = np.maximum(
                btc_df['high'] - btc_df['low'],
                np.maximum(
                    abs(btc_df['high'] - btc_df['close'].shift(1)),
                    abs(btc_df['low'] - btc_df['close'].shift(1))
                )
            )
            btc_df['atr'] = btc_df['tr'].rolling(atr_period).mean()

            btc_df['di_plus'] = (btc_df['dm_plus'].rolling(atr_period).mean() /
                                 btc_df['atr']) * 100
            btc_df['di_minus'] = (btc_df['dm_minus'].rolling(atr_period).mean() /
                                  btc_df['atr']) * 100

            di_diff = abs(btc_df['di_plus'] - btc_df['di_minus'])
            di_sum = btc_df['di_plus'] + btc_df['di_minus']
            btc_df['dx'] = (di_diff / di_sum) * 100
            btc_df['adx'] = btc_df['dx'].rolling(atr_period).mean()

            # Hinzufügen der Bitcoin-Features zum Feature DataFrame
            bitcoin_features = [
                'return', 'volatility_20d',
                'rel_to_ema20', 'rel_to_ema50', 'rel_to_ema200',
                'adx'
            ]

            if 'volume' in btc_df.columns:
                bitcoin_features.extend(['volume_change', 'volume_ma_ratio'])

            for feature in bitcoin_features:
                features_df[f'btc_{feature}'] = btc_df[feature]

            # 2. Altcoin zu Bitcoin Relative Stärke
            for symbol, df in self.market_data.items():
                if symbol == btc_symbol:
                    continue

                # Symbol für die Spaltenbezeichnung kürzen
                short_name = symbol.split('/')[0]

                # Tägliche Returns
                df['return'] = df['close'].pct_change()

                # Relative Performance zu Bitcoin
                merged_df = pd.merge(
                    df['return'],
                    btc_df['return'],
                    left_index=True,
                    right_index=True,
                    how='inner',
                    suffixes=(f'_{short_name}', '_btc')
                )

                # Nur hinzufügen, wenn genügend gemeinsame Daten vorhanden
                if len(merged_df) > 20:
                    # Relative Stärke: Altcoin Return - BTC Return
                    merged_df[f'rel_strength_{short_name}'] = (
                            merged_df[f'return_{short_name}'] - merged_df['return_btc']
                    )

                    # 20-Tage gleitender Durchschnitt der relativen Stärke
                    merged_df[f'rel_strength_{short_name}_ma20'] = (
                        merged_df[f'rel_strength_{short_name}'].rolling(20).mean()
                    )

                    # Zum Features-DataFrame hinzufügen
                    features_df[f'rel_strength_{short_name}'] = merged_df[f'rel_strength_{short_name}']
                    features_df[f'rel_strength_{short_name}_ma20'] = merged_df[f'rel_strength_{short_name}_ma20']

            # 3. Marktbreite-Indikatoren
            # Anzahl der Altcoins, die BTC outperformen
            outperform_columns = [c for c in features_df.columns if c.startswith('rel_strength_')]

            if outperform_columns:
                features_df['pct_outperform_btc'] = (
                        (features_df[outperform_columns] > 0).sum(axis=1) / len(outperform_columns)
                )

            # 4. Rollende Korrelationen
            # Durchschnittliche Korrelation zwischen Altcoins
            altcoin_symbols = [s for s in self.market_data.keys() if not s.startswith('BTC')]

            if len(altcoin_symbols) >= 2:
                # Sammel-DataFrame für Returns
                returns_df = pd.DataFrame(index=features_df.index)

                for symbol in altcoin_symbols:
                    short_name = symbol.split('/')[0]
                    if symbol in self.market_data:
                        returns_df[short_name] = self.market_data[symbol]['close'].pct_change()

                # 20-Tage rollende Korrelation für jedes Paar berechnen
                correlation_sum = 0
                pair_count = 0

                for i in range(len(returns_df.columns)):
                    for j in range(i + 1, len(returns_df.columns)):
                        if i != j:
                            col1 = returns_df.columns[i]
                            col2 = returns_df.columns[j]

                            # Rollende Korrelation berechnen
                            corr_series = returns_df[col1].rolling(20).corr(returns_df[col2])
                            correlation_sum += corr_series
                            pair_count += 1

                if pair_count > 0:
                    # Durchschnittliche Korrelation
                    features_df['avg_altcoin_correlation'] = correlation_sum / pair_count

            # 5. Feature Engineering: Gleitende Mittelwerte, Momentum
            # BTC Momentum (Nachhinkendes Indikator)
            features_df['btc_momentum_14d'] = (
                    btc_df['close'] / btc_df['close'].shift(14) - 1
            )

            # Alle NaN-Werte entfernen, da sie für das Training nicht nützlich sind
            features_df = features_df.dropna()

            self.logger.info(f"Features extrahiert: {len(features_df)} Zeitpunkte, {len(features_df.columns)} Features")
            return features_df

        except Exception as e:
            self.logger.error(f"Fehler bei der Feature-Extraktion: {e}")
            return pd.DataFrame()

    def train_regime_model(self, features_df: pd.DataFrame) -> bool:
        """
        Trainiert das Marktregime-Erkennungsmodell.

        Args:
            features_df: DataFrame mit Features für das Training

        Returns:
            True, wenn das Training erfolgreich war
        """
        if features_df.empty:
            self.logger.error("Keine Features für das Training vorhanden")
            return False

        try:
            # Daten vorbereiten
            X = features_df.copy()

            # Features normalisieren
            X_scaled = self.scaler.fit_transform(X)

            # Optimale Anzahl an Clustern bestimmen (wenn nicht festgelegt)
            if self.n_regimes is None:
                max_clusters = min(10, len(X) // 20)  # Maximal 10 Cluster oder 5% der Daten
                scores = []

                for n_clusters in range(2, max_clusters + 1):
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(X_scaled)

                    # Silhouette Score berechnen
                    from sklearn.metrics import silhouette_score
                    score = silhouette_score(X_scaled, labels)
                    scores.append((n_clusters, score))

                # Beste Anzahl Cluster auswählen
                self.n_regimes = max(scores, key=lambda x: x[1])[0]
                self.logger.info(f"Optimale Anzahl an Clustern: {self.n_regimes}")

            # K-Means-Clustering für Regime-Erkennung
            self.regime_model = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
            cluster_labels = self.regime_model.fit_predict(X_scaled)

            # Labels dem Features-DataFrame hinzufügen
            features_df['regime'] = cluster_labels

            # Regime-Übergänge analysieren
            self.analyze_regime_transitions(features_df)

            # Regime-Features analysieren, um sie zu charakterisieren
            self.analyze_regime_characteristics(features_df, X)

            # Performance pro Regime analysieren (benötigt Altcoin-Daten)
            self.analyze_regime_performance(features_df)

            self.model_trained = True
            self.logger.info(f"Regime-Modell erfolgreich trainiert: {self.n_regimes} Regime identifiziert")
            return True

        except Exception as e:
            self.logger.error(f"Fehler beim Training des Regime-Modells: {e}")
            return False

    def predict_regime(self, features: pd.DataFrame) -> int:
        """
        Sagt das aktuelle Marktregime voraus.

        Args:
            features: DataFrame mit Features für die Vorhersage

        Returns:
            Regime-ID oder -1 bei Fehler
        """
        if not self.model_trained or self.regime_model is None:
            self.logger.error("Modell nicht trainiert")
            return -1

        try:
            # Features normalisieren
            features_scaled = self.scaler.transform(features)

            # Regime vorhersagen
            regime = self.regime_model.predict(features_scaled)[0]

            self.current_regime = regime
            self.logger.info(f"Aktuelles Marktregime: {regime} - {self.regime_labels.get(regime, 'Unbekannt')}")

            return regime

        except Exception as e:
            self.logger.error(f"Fehler bei der Regime-Vorhersage: {e}")
            return -1

    def analyze_regime_transitions(self, features_df: pd.DataFrame) -> None:
        """
        Analysiert Übergänge zwischen verschiedenen Regimes.

        Args:
            features_df: DataFrame mit Features und Regime-Labels
        """
        try:
            # Regime-Übergänge
            transitions = np.zeros((self.n_regimes, self.n_regimes))

            # Für jeden Zeitpunkt (außer dem ersten)
            for i in range(1, len(features_df)):
                prev_regime = features_df['regime'].iloc[i - 1]
                curr_regime = features_df['regime'].iloc[i]
                transitions[prev_regime, curr_regime] += 1

            # In Wahrscheinlichkeiten umwandeln
            row_sums = transitions.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Vermeiden von Division durch Null
            transition_probs = transitions / row_sums

            self.regime_transitions = transition_probs

            # Regime-Dauer berechnen
            regime_durations = []
            current_regime = features_df['regime'].iloc[0]
            current_duration = 1

            for i in range(1, len(features_df)):
                if features_df['regime'].iloc[i] == current_regime:
                    current_duration += 1
                else:
                    regime_durations.append((current_regime, current_duration))
                    current_regime = features_df['regime'].iloc[i]
                    current_duration = 1

            # Letzte Periode hinzufügen
            regime_durations.append((current_regime, current_duration))

            # Durchschnittliche Dauer pro Regime
            avg_durations = {}
            for regime, duration in regime_durations:
                if regime not in avg_durations:
                    avg_durations[regime] = []
                avg_durations[regime].append(duration)

            for regime, durations in avg_durations.items():
                avg_durations[regime] = sum(durations) / len(durations)

            self.regime_durations = avg_durations

            self.logger.info(f"Regime-Transitionen analysiert")

        except Exception as e:
            self.logger.error(f"Fehler bei der Analyse der Regime-Übergänge: {e}")

    def analyze_regime_characteristics(self, features_df: pd.DataFrame, X: pd.DataFrame) -> None:
        """
        Analysiert charakteristische Eigenschaften jedes Regimes.

        Args:
            features_df: DataFrame mit Features und Regime-Labels
            X: Original-Feature-DataFrame
        """
        try:
            # Für jedes Regime
            for regime in range(self.n_regimes):
                # Daten dieses Regimes
                regime_data = X[features_df['regime'] == regime]

                if len(regime_data) > 0:
                    # Mittelwerte der Features in diesem Regime
                    regime_means = regime_data.mean()

                    # Vergleich zum Gesamtmittelwert
                    overall_means = X.mean()
                    relative_means = regime_means / overall_means

                    # Top-Features, die dieses Regime charakterisieren
                    feature_importance = relative_means.abs().sort_values(ascending=False)
                    top_features = feature_importance.head(5)

                    # Regime-Label basierend auf Top-Features erstellen
                    regime_characteristics = []

                    for feature, value in top_features.items():
                        if "btc_return" in feature:
                            if value > 1.2:
                                regime_characteristics.append("BTC-Bullish")
                            elif value < 0.8:
                                regime_characteristics.append("BTC-Bearish")

                        elif "volatility" in feature:
                            if value > 1.2:
                                regime_characteristics.append("Hohe-Volatilität")
                            elif value < 0.8:
                                regime_characteristics.append("Niedrige-Volatilität")

                        elif "rel_strength" in feature and "ma20" in feature:
                            coin = feature.split('_')[2]
                            if value > 1.2:
                                regime_characteristics.append(f"{coin}-Outperform")
                            elif value < 0.8:
                                regime_characteristics.append(f"{coin}-Underperform")

                        elif "pct_outperform_btc" in feature:
                            if value > 1.2:
                                regime_characteristics.append("Altcoin-Stärke")
                            elif value < 0.8:
                                regime_characteristics.append("BTC-Dominanz")

                        elif "correlation" in feature:
                            if value > 1.2:
                                regime_characteristics.append("Hohe-Korrelation")
                            elif value < 0.8:
                                regime_characteristics.append("Niedrige-Korrelation")

                    # Bitcoin-Trend-Status hinzufügen
                    btc_ma_features = [f for f in regime_means.index if "rel_to_ema" in f]
                    if btc_ma_features:
                        ema_values = [regime_means[f] for f in btc_ma_features]
                        if all(v > 0 for v in ema_values):
                            regime_characteristics.append("BTC-Aufwärtstrend")
                        elif all(v < 0 for v in ema_values):
                            regime_characteristics.append("BTC-Abwärtstrend")

                    # Falls keine charakteristischen Merkmale gefunden wurden
                    if not regime_characteristics:
                        regime_characteristics = ["Neutral"]

                    # Label erstellen
                    self.regime_labels[regime] = " & ".join(regime_characteristics[:3])

                    self.logger.info(f"Regime {regime} charakterisiert als: {self.regime_labels[regime]}")

        except Exception as e:
            self.logger.error(f"Fehler bei der Analyse der Regime-Charakteristiken: {e}")

    def analyze_regime_performance(self, features_df: pd.DataFrame) -> None:
        """
        Analysiert die Performance verschiedener Assets in jedem Regime.

        Args:
            features_df: DataFrame mit Features und Regime-Labels
        """
        try:
            # Für jedes Symbol Performance pro Regime berechnen
            performance_by_regime = {}

            for symbol, df in self.market_data.items():
                # Daten mit Regimes zusammenführen
                symbol_data = df.copy()
                symbol_data['return'] = symbol_data['close'].pct_change()

                # Mit Regime-Labels zusammenführen
                merged_data = pd.merge(
                    symbol_data['return'],
                    features_df['regime'],
                    left_index=True,
                    right_index=True,
                    how='inner'
                )

                if len(merged_data) > 0:
                    # Performance pro Regime
                    perf_by_regime = merged_data.groupby('regime')['return'].mean()
                    performance_by_regime[symbol] = perf_by_regime.to_dict()

            # In DataFrame umwandeln für einfachere Analyse
            perf_df = pd.DataFrame(performance_by_regime).T

            # NaN durch 0 ersetzen
            perf_df = perf_df.fillna(0)

            # Regime-Namen hinzufügen
            perf_df = perf_df.rename(columns=self.regime_labels)

            self.regime_performances = perf_df

            self.logger.info(f"Performance-Analyse pro Regime abgeschlossen")

        except Exception as e:
            self.logger.error(f"Fehler bei der Analyse der Performance pro Regime: {e}")

    def extract_trading_rules(self) -> Dict[int, Dict[str, Any]]:
        """
        Extrahiert Trading-Regeln für jedes Marktregime.

        Returns:
            Dictionary mit Regeln für jedes Regime
        """
        if not self.model_trained or self.regime_performances is None:
            self.logger.error("Modell nicht trainiert oder keine Performance-Daten")
            return {}

        try:
            trading_rules = {}

            # Für jedes Regime
            for regime in range(self.n_regimes):
                regime_label = self.regime_labels.get(regime, f"Regime {regime}")

                # Performance-Daten für dieses Regime
                if regime in self.regime_performances.columns:
                    perf_series = self.regime_performances[regime]

                    # Top-Performer
                    top_performers = perf_series.sort_values(ascending=False).head(3)
                    worst_performers = perf_series.sort_values().head(3)

                    # Durchschnittliche Performance
                    avg_performance = perf_series.mean()

                    # Anteil der positiven Performer
                    pct_positive = (perf_series > 0).mean() * 100

                    # Trading-Strategieempfehlung
                    if avg_performance > 0.01:  # 1% durchschnittliche tägliche Rendite
                        if pct_positive > 70:
                            strategy = "Aggressive Long-Strategie mit breitem Altcoin-Exposure"
                        else:
                            strategy = "Selektive Long-Strategie, fokussiert auf Top-Performer"
                    elif avg_performance > 0:
                        strategy = "Vorsichtige Long-Strategie, kleine Positionen"
                    elif avg_performance > -0.01:
                        strategy = "Überwiegend Cash-Position, minimales Exposure"
                    else:
                        if pct_positive < 30:
                            strategy = "Defensive Strategie, primär Stablecoins"
                        else:
                            strategy = "Sehr selektive Trades, überwiegend Stablecoins"

                    # Regel erstellen
                    trading_rules[regime] = {
                        "label": regime_label,
                        "top_performers": top_performers.to_dict(),
                        "worst_performers": worst_performers.to_dict(),
                        "avg_performance": avg_performance,
                        "pct_positive": pct_positive,
                        "recommended_strategy": strategy,
                        "portfolio_allocation": self._get_portfolio_allocation(avg_performance, pct_positive)
                    }

            self.logger.info(f"Trading-Regeln für {len(trading_rules)} Regime extrahiert")
            return trading_rules

        except Exception as e:
            self.logger.error(f"Fehler bei der Extraktion von Trading-Regeln: {e}")
            return {}

    def _get_portfolio_allocation(self, avg_performance: float, pct_positive: float) -> Dict[str, float]:
        """
        Empfiehlt eine Portfolio-Allokation basierend auf Regime-Performance.

        Args:
            avg_performance: Durchschnittliche Performance in diesem Regime
            pct_positive: Prozentsatz der positiven Performer

        Returns:
            Dictionary mit empfohlenen Allokationen
        """
        # Basierend auf Performance und Richtung der Mehrheit der Assets
        if avg_performance > 0.015:  # Sehr starke Performance
            if pct_positive > 80:
                return {
                    "altcoins": 0.8,
                    "bitcoin": 0.15,
                    "stablecoins": 0.05
                }
            else:
                return {
                    "altcoins": 0.6,
                    "bitcoin": 0.25,
                    "stablecoins": 0.15
                }
        elif avg_performance > 0.005:  # Gute Performance
            if pct_positive > 70:
                return {
                    "altcoins": 0.6,
                    "bitcoin": 0.2,
                    "stablecoins": 0.2
                }
            else:
                return {
                    "altcoins": 0.4,
                    "bitcoin": 0.3,
                    "stablecoins": 0.3
                }
        elif avg_performance > 0:  # Leicht positive Performance
            if pct_positive > 60:
                return {
                    "altcoins": 0.3,
                    "bitcoin": 0.3,
                    "stablecoins": 0.4
                }
            else:
                return {
                    "altcoins": 0.2,
                    "bitcoin": 0.3,
                    "stablecoins": 0.5
                }
        elif avg_performance > -0.005:  # Leicht negative Performance
            return {
                "altcoins": 0.1,
                "bitcoin": 0.2,
                "stablecoins": 0.7
            }
        else:  # Stark negative Performance
            return {
                "altcoins": 0.0,
                "bitcoin": 0.1,
                "stablecoins": 0.9
            }

    def save_model(self, filepath: str) -> bool:
        """
        Speichert das trainierte Regime-Modell.

        Args:
            filepath: Pfad zum Speichern des Modells

        Returns:
            True, wenn erfolgreich gespeichert
        """
        if not self.model_trained:
            self.logger.error("Kein trainiertes Modell zum Speichern verfügbar")
            return False

        try:
            # Verzeichnis erstellen, falls nicht vorhanden
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Modell als Pickle speichern
            import pickle
            model_data = {
                "regime_model": self.regime_model,
                "scaler": self.scaler,
                "n_regimes": self.n_regimes,
                "regime_labels": self.regime_labels,
                "regime_transitions": self.regime_transitions,
                "regime_performances": self.regime_performances,
                "regime_durations": getattr(self, "regime_durations", {})
            }

            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)

            self.logger.info(f"Regime-Modell gespeichert unter {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Fehler beim Speichern des Modells: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        """
        Lädt ein trainiertes Regime-Modell.

        Args:
            filepath: Pfad zum gespeicherten Modell

        Returns:
            True, wenn erfolgreich geladen
        """
        try:
            import pickle
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.regime_model = model_data["regime_model"]
            self.scaler = model_data["scaler"]
            self.n_regimes = model_data["n_regimes"]
            self.regime_labels = model_data["regime_labels"]
            self.regime_transitions = model_data["regime_transitions"]
            self.regime_performances = model_data["regime_performances"]

            if "regime_durations" in model_data:
                self.regime_durations = model_data["regime_durations"]

            self.model_trained = True

            self.logger.info(f"Regime-Modell geladen von {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Fehler beim Laden des Modells: {e}")
            return False


class AssetClusterAnalyzer:
    """
    Clustering-Analyse für Kryptowährungen.
    Identifiziert Gruppen von Assets mit ähnlichem Verhalten.
    """

    def __init__(self, data_dir: str = "data/market_data"):
        """
        Initialisiert den Asset-Cluster-Analyzer.

        Args:
            data_dir: Verzeichnis mit historischen Marktdaten
        """
        self.data_dir = data_dir
        self.market_data = {}
        self.correlation_matrix = None
        self.clusters = None
        self.cluster_model = None
        self.feature_data = None
        self.scaler = StandardScaler()
        self.cluster_performances = None
        self.logger = logging.getLogger(__name__)

    def load_market_data(self, symbols: List[str] = None,
                         data_manager: Optional[DataManager] = None,
                         timeframe: str = "1d",
                         start_date: str = None,
                         end_date: str = None) -> bool:
        """
        Lädt Marktdaten für die angegebenen Symbole.

        Args:
            symbols: Liste der zu ladenden Symbole (oder None für alle)
            data_manager: Optionaler DataManager zur Datenabfrage
            timeframe: Zeitrahmen der Daten
            start_date: Startdatum im Format 'YYYY-MM-DD'
            end_date: Enddatum im Format 'YYYY-MM-DD'

        Returns:
            True, wenn Daten erfolgreich geladen wurden
        """
        try:
            # Wenn kein Startdatum angegeben, letzten Monat verwenden
            if not start_date:
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

            # Wenn kein Enddatum angegeben, aktuelles Datum verwenden
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')

            # Wenn DataManager bereitgestellt wurde, diesen für das Laden verwenden
            if data_manager:
                for symbol in symbols:
                    df = data_manager.get_historical_data(
                        symbol=symbol,
                        source="binance",  # Könnte aus Einstellungen kommen
                        timeframe=timeframe,
                        start_date=datetime.strptime(start_date, '%Y-%m-%d'),
                        end_date=datetime.strptime(end_date, '%Y-%m-%d'),
                        use_cache=True
                    )

                    if not df.empty:
                        self.market_data[symbol] = df
                        self.logger.info(f"Daten für {symbol} geladen: {len(df)} Einträge")

                if not self.market_data:
                    self.logger.error("Keine Marktdaten geladen")
                    return False

                return True

            # Alternativ: Direkt aus dem Verzeichnis laden
            binance_dir = os.path.join(self.data_dir, "binance")

            # Wenn keine Symbole angegeben, alle verfügbaren laden
            if not symbols:
                symbols = []
                for filename in os.listdir(binance_dir):
                    if filename.endswith(f"_{timeframe}.csv") or filename.endswith(f"_{timeframe}_20230101.csv"):
                        symbol = filename.split("_")[0]
                        quote = filename.split("_")[1]
                        symbols.append(f"{symbol}/{quote}")

            # Daten für jedes Symbol laden
            for symbol in symbols:
                base, quote = symbol.split("/")
                filename_pattern = f"{base}_{quote}_{timeframe}"

                # Datei suchen
                csv_path = None
                for f in os.listdir(binance_dir):
                    if f.startswith(filename_pattern):
                        csv_path = os.path.join(binance_dir, f)
                        break

                if not csv_path:
                    self.logger.warning(f"Keine Daten gefunden für {symbol} mit Timeframe {timeframe}")
                    continue

                # Daten laden
                df = pd.read_csv(csv_path)

                # Datum als Index setzen, falls vorhanden
                if 'timestamp' in df.columns:
                    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('date', inplace=True)
                elif 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                elif 'time' in df.columns:
                    df['date'] = pd.to_datetime(df['time'])
                    df.set_index('date', inplace=True)

                # Datum filtern, falls angegeben
                if start_date and end_date:
                    df = df[(df.index >= start_date) & (df.index <= end_date)]

                # Daten speichern
                self.market_data[symbol] = df
                self.logger.info(f"Daten für {symbol} geladen: {len(df)} Einträge")

            if not self.market_data:
                self.logger.error("Keine Marktdaten geladen")
                return False

            self.logger.info(f"Marktdaten für {len(self.market_data)} Symbole geladen")
            return True

        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Marktdaten: {e}")
            return False

    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """
        Berechnet die Korrelationsmatrix zwischen allen Assets.

        Returns:
            Korrelationsmatrix als DataFrame
        """
        if not self.market_data:
            self.logger.error("Keine Marktdaten vorhanden")
            return pd.DataFrame()

        try:
            # DataFrame für Returns
            returns_df = pd.DataFrame()

            # Returns für jedes Symbol extrahieren
            for symbol, df in self.market_data.items():
                short_name = symbol.split('/')[0]
                returns_df[short_name] = df['close'].pct_change()

            # NaN-Werte entfernen
            returns_df = returns_df.dropna()

            # Korrelationsmatrix berechnen
            correlation_matrix = returns_df.corr()

            self.correlation_matrix = correlation_matrix

            self.logger.info(f"Korrelationsmatrix berechnet für {len(correlation_matrix)} Assets")
            return correlation_matrix

        except Exception as e:
            self.logger.error(f"Fehler bei der Berechnung der Korrelationsmatrix: {e}")
            return pd.DataFrame()

    def extract_asset_features(self) -> pd.DataFrame:
        """
        Extrahiert Features für das Asset-Clustering.

        Returns:
            DataFrame mit Features für jedes Asset
        """
        if not self.market_data:
            self.logger.error("Keine Marktdaten vorhanden")
            return pd.DataFrame()

        try:
            # Features für jedes Asset
            asset_features = []

            for symbol, df in self.market_data.items():
                # Basiswährung
                base = symbol.split('/')[0]

                # Preis- und Return-Daten
                price_data = df['close']
                returns = price_data.pct_change().dropna()

                if len(returns) < 20:
                    self.logger.warning(f"Nicht genügend Daten für {symbol}, überspringe")
                    continue

                # Return-Features
                mean_return = returns.mean()
                std_return = returns.std()
                skew_return = returns.skew()
                kurt_return = returns.kurt()

                # Volatilität
                volatility_20d = returns.rolling(20).std().mean()

                # Trend-Features
                if len(price_data) >= 50:
                    ma_20 = price_data.rolling(20).mean()
                    ma_50 = price_data.rolling(50).mean()

                    # Average Ratio zum MA
                    avg_dist_ma20 = ((price_data / ma_20) - 1).mean()
                    avg_dist_ma50 = ((price_data / ma_50) - 1).mean()

                    # Verhältnis MA20 zu MA50
                    ma_ratio = (ma_20 / ma_50).mean()
                else:
                    avg_dist_ma20 = 0
                    avg_dist_ma50 = 0
                    ma_ratio = 1

                # Volumen-Features (falls vorhanden)
                if 'volume' in df.columns:
                    vol_data = df['volume']
                    vol_change = vol_data.pct_change().dropna()
                    mean_vol_change = vol_change.mean()
                    std_vol_change = vol_change.std()

                    # Volumen/Preis-Korrelation
                    vol_price_corr = vol_data.corr(price_data)
                else:
                    mean_vol_change = 0
                    std_vol_change = 0
                    vol_price_corr = 0

                # Features für dieses Asset
                asset_feature = {
                    'symbol': base,
                    'mean_return': mean_return,
                    'volatility': std_return,
                    'skewness': skew_return,
                    'kurtosis': kurt_return,
                    'volatility_20d': volatility_20d,
                    'avg_dist_ma20': avg_dist_ma20,
                    'avg_dist_ma50': avg_dist_ma50,
                    'ma_ratio': ma_ratio,
                    'mean_vol_change': mean_vol_change,
                    'std_vol_change': std_vol_change,
                    'vol_price_corr': vol_price_corr
                }

                asset_features.append(asset_feature)

            # In DataFrame umwandeln
            feature_df = pd.DataFrame(asset_features)

            if feature_df.empty:
                self.logger.error("Keine Features extrahiert")
                return pd.DataFrame()

            # Symbol als Index
            feature_df.set_index('symbol', inplace=True)

            # NaN-Werte durch 0 ersetzen
            feature_df = feature_df.fillna(0)

            self.feature_data = feature_df

            self.logger.info(f"Features für {len(feature_df)} Assets extrahiert")
            return feature_df

        except Exception as e:
            self.logger.error(f"Fehler bei der Extraktion der Asset-Features: {e}")
            return pd.DataFrame()

    def run_clustering(self, n_clusters: int = None, method: str = 'kmeans',
                       correlation_based: bool = True) -> pd.DataFrame:
        """
        Führt das Clustering der Assets durch.

        Args:
            n_clusters: Anzahl der Cluster (oder None für automatische Bestimmung)
            method: Clustering-Methode ('kmeans' oder 'dbscan')
            correlation_based: Ob auf Korrelationen oder Features basiert werden soll

        Returns:
            DataFrame mit Cluster-Zuordnungen
        """
        try:
            if correlation_based:
                # Falls noch nicht berechnet
                if self.correlation_matrix is None:
                    self.calculate_correlation_matrix()

                if self.correlation_matrix.empty:
                    self.logger.error("Korrelationsmatrix ist leer")
                    return pd.DataFrame()

                # 1 - Korrelation als Distanzmaß
                distance_matrix = 1 - self.correlation_matrix

                # In Feature-Array umwandeln mit MDS
                from sklearn.manifold import MDS
                mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
                features = mds.fit_transform(distance_matrix)

                # DataFrame erstellen
                asset_features = pd.DataFrame(
                    features,
                    columns=['dim1', 'dim2'],
                    index=self.correlation_matrix.index
                )
            else:
                # Basierend auf statistischen Features
                if self.feature_data is None:
                    self.extract_asset_features()

                if self.feature_data.empty:
                    self.logger.error("Keine Feature-Daten vorhanden")
                    return pd.DataFrame()

                asset_features = self.feature_data.copy()

            # Daten für Clustering vorbereiten
            X = self.scaler.fit_transform(asset_features)

            # Optimale Anzahl an Clustern bestimmen (wenn nicht angegeben)
            if method == 'kmeans' and n_clusters is None:
                max_clusters = min(10, len(X) // 2)  # Maximal 10 Cluster oder Hälfte der Daten
                scores = []

                for k in range(2, max_clusters + 1):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(X)

                    if len(set(labels)) > 1:  # Mindestens 2 unterschiedliche Cluster
                        from sklearn.metrics import silhouette_score
                        score = silhouette_score(X, labels)
                        scores.append((k, score))

                # Beste Anzahl Cluster auswählen
                n_clusters = max(scores, key=lambda x: x[1])[0]
                self.logger.info(f"Optimale Anzahl an Clustern: {n_clusters}")

            # Clustering durchführen
            if method == 'kmeans':
                self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = self.cluster_model.fit_predict(X)
                self.cluster_centers = self.cluster_model.cluster_centers_
            elif method == 'dbscan':
                from sklearn.neighbors import NearestNeighbors

                # Optimalen eps-Parameter bestimmen
                neigh = NearestNeighbors(n_neighbors=2)
                nbrs = neigh.fit(X)
                distances, indices = nbrs.kneighbors(X)
                distances = np.sort(distances[:, 1])

                # Knick in der Kurve finden
                from scipy.signal import argrelextrema
                n = len(distances)
                indices = argrelextrema(np.diff(distances), np.greater)[0]
                if len(indices) > 0:
                    eps = distances[indices[0]]
                else:
                    eps = np.percentile(distances, 90)

                self.cluster_model = DBSCAN(eps=eps, min_samples=2)
                cluster_labels = self.cluster_model.fit_predict(X)
            else:
                self.logger.error(f"Unbekannte Clustering-Methode: {method}")
                return pd.DataFrame()

            # Ergebnisse in DataFrame
            if correlation_based:
                result_df = pd.DataFrame(
                    {'cluster': cluster_labels},
                    index=self.correlation_matrix.index
                )
            else:
                result_df = pd.DataFrame(
                    {'cluster': cluster_labels},
                    index=self.feature_data.index
                )

            # Cluster-Zentren speichern (für KMeans)
            if method == 'kmeans':
                self.cluster_centers = self.cluster_model.cluster_centers_

            self.clusters = result_df

            # Cluster-Performance berechnen
            self.analyze_cluster_performance(result_df)

            self.logger.info(
                f"Clustering abgeschlossen: {len(result_df)} Assets in {len(set(cluster_labels))} Clustern")
            return result_df

        except Exception as e:
            self.logger.error(f"Fehler beim Clustering: {e}")
            return pd.DataFrame()

    def analyze_cluster_performance(self, cluster_assignments: pd.DataFrame) -> Dict[int, Dict]:
        """
        Analysiert die Performance der verschiedenen Asset-Cluster.

        Args:
            cluster_assignments: DataFrame mit Cluster-Zuordnungen

        Returns:
            Dictionary mit Performance-Statistiken pro Cluster
        """
        if cluster_assignments.empty or not self.market_data:
            self.logger.error("Keine Cluster-Zuweisungen oder Marktdaten vorhanden")
            return {}

        try:
            # Cluster-Performance
            cluster_stats = {}

            # Für jeden Cluster
            for cluster in sorted(cluster_assignments['cluster'].unique()):
                # Assets in diesem Cluster
                cluster_assets = cluster_assignments[cluster_assignments['cluster'] == cluster].index

                # Performance-Daten sammeln
                returns_data = []

                for asset in cluster_assets:
                    # Vollständiges Symbol rekonstruieren
                    asset_symbols = [s for s in self.market_data.keys() if s.startswith(asset)]

                    if not asset_symbols:
                        continue

                    asset_symbol = asset_symbols[0]

                    # Returns berechnen
                    df = self.market_data[asset_symbol]
                    returns = df['close'].pct_change().dropna()

                    returns_data.append((asset, returns))

                if not returns_data:
                    continue

                # Durchschnittliche tägliche Performance des Clusters
                all_returns = pd.DataFrame({asset: returns for asset, returns in returns_data})
                avg_returns = all_returns.mean(axis=1)

                # Performance-Statistiken
                mean_return = avg_returns.mean()
                volatility = avg_returns.std()
                sharpe = mean_return / volatility if volatility > 0 else 0
                max_drawdown = (avg_returns.cumsum() - avg_returns.cumsum().cummax()).min()
                win_rate = (avg_returns > 0).mean()

                # Korrelationen innerhalb des Clusters
                intra_corr = all_returns.corr().values.mean()

                # Repräsentatives Asset (am nächsten zum Clusterzentrum)
                if hasattr(self, 'cluster_centers') and cluster >= 0:
                    # Skalierte Feature-Werte
                    if self.feature_data is not None and not self.feature_data.empty:
                        X = self.scaler.transform(self.feature_data)

                        # Indizes der Assets in diesem Cluster
                        cluster_indices = []
                        for idx, (asset, row) in enumerate(self.feature_data.iterrows()):
                            if asset in cluster_assets:
                                cluster_indices.append(idx)

                        if cluster_indices:
                            # Distanzen zum Clusterzentrum
                            center = self.cluster_centers[cluster]
                            distances = np.linalg.norm(X[cluster_indices] - center, axis=1)

                            # Asset mit minimaler Distanz
                            rep_idx = cluster_indices[np.argmin(distances)]
                            representative = self.feature_data.index[rep_idx]
                        else:
                            representative = cluster_assets[0] if cluster_assets else "Unknown"
                    else:
                        representative = cluster_assets[0] if cluster_assets else "Unknown"
                else:
                    representative = cluster_assets[0] if cluster_assets else "Unknown"

                # Cluster-Statistik speichern
                cluster_stats[cluster] = {
                    "assets": list(cluster_assets),
                    "count": len(cluster_assets),
                    "mean_return": mean_return,
                    "volatility": volatility,
                    "sharpe_ratio": sharpe,
                    "max_drawdown": max_drawdown,
                    "win_rate": win_rate,
                    "intra_correlation": intra_corr,
                    "representative_asset": representative
                }

            self.cluster_performances = cluster_stats

            self.logger.info(f"Cluster-Performance-Analyse abgeschlossen für {len(cluster_stats)} Cluster")
            return cluster_stats

        except Exception as e:
            self.logger.error(f"Fehler bei der Analyse der Cluster-Performance: {e}")
            return {}

    def recommend_portfolio(self, n_assets: int = 5) -> Dict[str, Any]:
        """
        Empfiehlt ein diversifiziertes Portfolio basierend auf dem Clustering.

        Args:
            n_assets: Anzahl der zu empfehlenden Assets

        Returns:
            Dictionary mit Portfolioempfehlungen
        """
        if self.clusters is None or self.cluster_performances is None:
            self.logger.error("Keine Cluster-Daten oder Performance-Analyse vorhanden")
            return {}

        try:
            recommendations = {}

            # Cluster nach Sharpe Ratio sortieren
            sorted_clusters = sorted(
                self.cluster_performances.items(),
                key=lambda x: x[1]['sharpe_ratio'],
                reverse=True
            )

            # Positiv performende Cluster auswählen
            positive_clusters = [c for c in sorted_clusters if c[1]['mean_return'] > 0]

            if not positive_clusters:
                self.logger.warning("Keine positiv performenden Cluster gefunden")
                recommendations['strategy'] = "Defensive Strategie - nur Stablecoins empfohlen"
                recommendations['assets'] = []
                return recommendations

            # Maximale Anzahl Assets pro Cluster
            assets_per_cluster = max(1, n_assets // len(positive_clusters))

            # Empfohlene Assets sammeln
            recommended_assets = []
            allocation = {}

            for cluster_id, stats in positive_clusters:
                # Assets in diesem Cluster nach Performance sortieren
                cluster_assets = stats['assets']

                # Performance-Daten für Assets in diesem Cluster
                asset_performance = {}

                for asset in cluster_assets:
                    # Vollständiges Symbol rekonstruieren
                    asset_symbols = [s for s in self.market_data.keys() if s.startswith(asset)]

                    if not asset_symbols:
                        continue

                    asset_symbol = asset_symbols[0]

                    # Performance-Metrik: Sharpe Ratio
                    df = self.market_data[asset_symbol]
                    returns = df['close'].pct_change().dropna()

                    mean_return = returns.mean()
                    volatility = returns.std()
                    sharpe = mean_return / volatility if volatility > 0 else 0

                    asset_performance[asset] = sharpe

                # Top-Assets aus diesem Cluster auswählen
                top_assets = sorted(
                    asset_performance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:assets_per_cluster]

                if top_assets:
                    for asset, _ in top_assets:
                        recommended_assets.append(asset)
                else:
                    # Falls keine Performance-Daten, repräsentatives Asset verwenden
                    rep_asset = stats.get('representative_asset')
                    if rep_asset and rep_asset != "Unknown":
                        recommended_assets.append(rep_asset)

            # Auf die gewünschte Anzahl begrenzen
            recommended_assets = recommended_assets[:n_assets]

            # Einfache Allokation basierend auf Sharpe Ratio
            if recommended_assets:
                asset_sharpes = {}

                for asset in recommended_assets:
                    # Vollständiges Symbol rekonstruieren
                    asset_symbols = [s for s in self.market_data.keys() if s.startswith(asset)]

                    if not asset_symbols:
                        continue

                    asset_symbol = asset_symbols[0]

                    # Sharpe berechnen
                    df = self.market_data[asset_symbol]
                    returns = df['close'].pct_change().dropna()

                    mean_return = returns.mean()
                    volatility = returns.std()


sharpe = mean_return / volatility if volatility > 0 else 0

asset_sharpes[asset] = max(0.01, sharpe)  # Mindestens 0.01

# Proportional zum Sharpe Ratio allokieren
total_sharpe = sum(asset_sharpes.values())

for asset, sharpe in asset_sharpes.items():
    allocation[asset] = sharpe / total_sharpe

# Portfolio-Empfehlung
recommendations = {
    'assets': recommended_assets,
    'allocation': allocation,
    'strategy': "Diversifizierte Allokation über performante Cluster"
}

self.logger.info(f"Portfolio mit {len(recommended_assets)} Assets empfohlen")
return recommendations

except Exception as e:
self.logger.error(f"Fehler bei der Portfolio-Empfehlung: {e}")
return {}


def analyze_new_coin(self, coin_symbol: str, coin_data: pd.DataFrame, min_days: int = 3) -> Dict[str, Any]:
    """
    Analysiert einen neuen Coin und weist ihn einem Cluster zu.

    Args:
        coin_symbol: Symbol des Coins (z.B. 'BTC/USDT')
        coin_data: DataFrame mit Kursdaten des Coins
        min_days: Minimale Anzahl an Tagen für die Analyse

    Returns:
        Analyse-Ergebnisse oder leeres Dictionary bei Fehler
    """
    try:
        # Prüfen, ob genügend Daten vorhanden sind
        if len(coin_data) < min_days:
            self.logger.warning(f"Nicht genügend Daten für {coin_symbol}: {len(coin_data)} Tage")
            return {"status": "insufficient_data", "days_available": len(coin_data)}

        # Features für diesen Coin extrahieren
        features = self._extract_coin_features(coin_data, coin_symbol)

        # Ähnliche Coins finden
        similar_coins = []

        if self.clusters is not None:
            # Vorheriges Clustering laden
            if self.feature_data is not None:
                # Feature-Matrix
                X = self.feature_data.copy()

                if not X.empty:
                    # Neue Features normalisieren
                    X_scaled = self.scaler.transform(X)

                    # Feature-Vektor für den neuen Coin
                    if all(f in X.columns for f in features.keys()):
                        coin_features = pd.DataFrame([list(features.values())], columns=list(features.keys()))
                        coin_features_scaled = self.scaler.transform(coin_features)

                        # Ähnlichkeit berechnen (euklidische Distanz)
                        distances = np.sqrt(((X_scaled - coin_features_scaled) ** 2).sum(axis=1))

                        # Top-5 ähnlichste Coins
                        top_indices = np.argsort(distances)[:5]
                        similar_coins = [X.index[i] for i in top_indices]

            # Cluster vorhersagen
            predicted_cluster = -1  # Ausreißer als Standard

            if hasattr(self, 'cluster_model') and self.cluster_model is not None:
                # Feature-Vektor für den neuen Coin
                if all(f in self.feature_data.columns for f in features.keys()):
                    coin_features = pd.DataFrame([list(features.values())], columns=list(features.keys()))
                    coin_features_scaled = self.scaler.transform(coin_features)

                    # Cluster vorhersagen
                    predicted_cluster = self.cluster_model.predict(coin_features_scaled)[0]

        # Empfohlene Strategie für diesen Coin basierend auf Cluster
        if self.clusters is not None and predicted_cluster >= 0:
            # Assets im gleichen Cluster
            cluster_assets = self.clusters[self.clusters['cluster'] == predicted_cluster].index.tolist()

            # Performance-Daten aus dem Cluster-Analyzer
            cluster_performance = self.cluster_performances.get(predicted_cluster, {})

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

        # Analyseergebnis
        analysis_result = {
            "status": "analyzed",
            "coin": coin_symbol,
            "predicted_cluster": predicted_cluster,
            "similar_coins": similar_coins,
            "recommended_strategy": recommended_strategy,
            "cluster_assets": cluster_assets,
            "features": features
        }

        self.logger.info(
            f"Analyse für {coin_symbol} abgeschlossen: Cluster {predicted_cluster}, Strategie: {recommended_strategy}")
        return analysis_result

    except Exception as e:
        self.logger.error(f"Fehler bei der Analyse des neuen Coins {coin_symbol}: {e}")
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
            self.logger.warning(f"Fehlende Spalten für {coin_symbol}: {missing_columns}")
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
        features['volatility_20d'] = df['return'].rolling(min(10, len(df))).std().mean()

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
        self.logger.error(f"Fehler bei der Feature-Extraktion für {coin_symbol}: {e}")
        return {}


class NewCoinMonitor:
    """
    Überwacht und analysiert neue Coins am Markt.
    """

    def __init__(self, data_dir: str = "data/market_data"):
        """
        Initialisiert den New Coin Monitor.

        Args:
            data_dir: Verzeichnis mit Marktdaten
        """
        self.data_dir = data_dir
        self.known_coins = set()
        self.new_coins_watchlist = {}  # Symbol -> Analyse-Info
        self.logger = logging.getLogger(__name__)
        self.last_check_time = datetime.now() - timedelta(days=1)  # Force initial check

    def update_known_coins(self, data_manager: Optional[DataManager] = None) -> bool:
        """
        Aktualisiert die Liste bekannter Coins.

        Args:
            data_manager: Optionaler DataManager für zusätzliche Daten

        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            # Daten-Verzeichnis durchsuchen
            binance_dir = os.path.join(self.data_dir, "binance")

            if not os.path.exists(binance_dir):
                os.makedirs(binance_dir, exist_ok=True)
                self.logger.warning(f"Verzeichnis {binance_dir} erstellt")

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
                self.logger.warning(f"Fehler beim Abrufen der Exchange-Info: {e}")

            # Bekannte Coins aktualisieren
            self.known_coins = known_coins

            self.logger.info(f"{len(self.known_coins)} bekannte Coins aktualisiert")
            return True

        except Exception as e:
            self.logger.error(f"Fehler beim Aktualisieren der bekannten Coins: {e}")
            return False

    def check_for_new_coins(self) -> List[str]:
        """
        Überprüft, ob neue Coins am Markt verfügbar sind.

        Returns:
            Liste neu erkannter Coins
        """
        try:
            # Zeit seit letztem Check prüfen
            now = datetime.now()
            hours_since_last_check = (now - self.last_check_time).total_seconds() / 3600

            # Nur alle 6 Stunden prüfen
            if hours_since_last_check < 6:
                return []

            self.last_check_time = now

            # Aktualisiere die Liste der bekannten Coins
            self.update_known_coins()

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
                                    self.logger.info(f"Neuer Coin erkannt: {formatted_symbol} (Volumen: {volume})")

                                    # Coin zur Watchlist hinzufügen
                                    self.new_coins_watchlist[formatted_symbol] = {
                                        'discovery_time': datetime.now().isoformat(),
                                        'initial_volume': volume,
                                        'analysis_status': 'pending',
                                        'data_available': False,
                                        'days_tracked': 0
                                    }
            except Exception as e:
                self.logger.warning(f"Fehler beim Abrufen der Ticker-Daten: {e}")

            if new_coins:
                self.logger.info(f"{len(new_coins)} neue Coins entdeckt: {', '.join(new_coins)}")

                # Führe eine erste Datensammlung für neue Coins durch
                for coin in new_coins:
                    self.collect_data_for_coin(coin)
            else:
                self.logger.info("Keine neuen Coins entdeckt")

            return new_coins

        except Exception as e:
            self.logger.error(f"Fehler bei der Überprüfung auf neue Coins: {e}")
            return []

    def collect_data_for_coin(self, coin: str) -> bool:
        """
        Sammelt Daten für einen Coin.

        Args:
            coin: Symbol des Coins (z.B. 'BTC/USDT')

        Returns:
            True bei Erfolg, False bei Fehler
        """
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
                    binance_dir = os.path.join(self.data_dir, "binance")
                    os.makedirs(binance_dir, exist_ok=True)

                    # Dateiname generieren
                    current_date = datetime.now().strftime("%Y%m%d")
                    filename = f"{base}_{quote}_1d_{current_date}.csv"
                    filepath = os.path.join(binance_dir, filename)

                    # Daten speichern
                    df.to_csv(filepath, index=False)

                    self.logger.info(f"Daten für {coin} gesammelt und gespeichert: {len(df)} Einträge")

                    # Watchlist aktualisieren
                    if coin in self.new_coins_watchlist:
                        self.new_coins_watchlist[coin]['data_available'] = True
                        self.new_coins_watchlist[coin]['first_data_date'] = df['timestamp'].min().isoformat()
                        self.new_coins_watchlist[coin]['days_tracked'] = len(df)
                        self.new_coins_watchlist[coin]['last_updated'] = datetime.now().isoformat()

                    return True
            else:
                self.logger.warning(f"Fehler beim Abrufen der Kline-Daten für {coin}: {response.status_code}")
                return False

        except Exception as e:
            self.logger.error(f"Fehler bei der Datensammlung für {coin}: {e}")
            return False

    def get_coins_for_analysis(self, min_days: int = 3) -> List[str]:
        """
        Gibt Liste von Coins zurück, die bereit für die Analyse sind.

        Args:
            min_days: Minimale Anzahl an Tagen für die Analyse

        Returns:
            Liste mit Coin-Symbolen
        """
        coins_to_analyze = []

        # Für jeden Coin in der Watchlist
        for coin, info in list(self.new_coins_watchlist.items()):
            days_tracked = info.get('days_tracked', 0)
            analysis_status = info.get('analysis_status', 'pending')

            # Wenn genügend Daten vorliegen und noch nicht analysiert
            if days_tracked >= min_days and analysis_status == 'pending':
                coins_to_analyze.append(coin)

        return coins_to_analyze

    def update_coin_status(self, coin: str, analysis_result: Dict[str, Any]) -> None:
        """
        Aktualisiert den Status eines Coins nach der Analyse.

        Args:
            coin: Symbol des Coins
            analysis_result: Ergebnis der Analyse
        """
        if coin in self.new_coins_watchlist:
            self.new_coins_watchlist[coin]['analysis_status'] = 'analyzed'
            self.new_coins_watchlist[coin]['analysis_time'] = datetime.now().isoformat()
            self.new_coins_watchlist[coin]['analysis_result'] = analysis_result
        else:
            self.logger.warning(f"Coin {coin} nicht in der Watchlist gefunden")


class TradingBot:
    """
    Hauptklasse für den Trading Bot mit integrierter ML.

    Diese Klasse koordiniert alle Trading-Aktivitäten und ist die
    zentrale Schnittstelle für die Ausführung des Bots.
    """

    def __init__(self, mode: str = "paper", strategy_name: str = "default", settings: Optional[Settings] = None):
        """
        Initialisiert den Trading Bot.

        Args:
            mode: Trading-Modus ('live', 'paper', 'backtest')
            strategy_name: Name der zu verwendenden Strategie
            settings: Bot-Konfiguration (optional, sonst Standardkonfiguration)
        """
        # Einstellungen initialisieren
        self.settings = settings or Settings()

        # Logger einrichten
        log_level = getattr(logging, self.settings.get('logging.level', 'INFO'))
        self.logger = setup_logger(log_level)

        # Trading-Parameter
        self.mode = mode
        self.strategy_name = strategy_name
        self.running = False
        self.check_interval = self.settings.get('timeframes.check_interval', 300)  # Sekunden

        # Exchange initialisieren
        self.exchange = ExchangeFactory.create(self.settings, mode)

        # Position Manager initialisieren
        self.position_manager = PositionManager()

        # Strategie initialisieren
        self.strategy = self._initialize_strategy(strategy_name)

        # Event Callbacks
        self.on_trade_callbacks = []
        self.on_error_callbacks = []
        self.on_status_update_callbacks = []

        # Performance-Tracking
        self.start_time = None
        self.start_balance = 0.0

        # Trading-Paare
        self.trading_pairs = self.settings.get('trading_pairs', ["BTC/USDT"])

        # Daten-Cache
        self.data_cache = {}  # Speichert OHLCV-Daten für jedes Symbol
        self.data_cache_lock = threading.Lock()  # Thread-Sicherheit für Cache-Zugriffe

        # DataManager für erweiterte Datenquellen initialisieren
        self.data_manager = DataManager(self.settings)

        # Trading-Threads und Status
        self.trading_thread = None
        self.monitor_thread = None
        self.status_update_interval = self.settings.get('system.status_update_interval', 60)  # Sekunden

        # Thread-Pool für parallele Datenverarbeitung
        self.max_workers = self.settings.get('system.max_workers', min(32, (os.cpu_count() or 4) + 4))

        # Letzter Status für Benachrichtigungen
        self.last_status = {}

        # API Fehler-Tracking
        self.api_error_count = 0
        self.last_api_error_time = None
        self.max_api_errors = self.settings.get('system.max_api_errors', 5)
        self.api_error_window = self.settings.get('system.api_error_window', 300)  # Sekunden

        # ML-Komponenten, falls aktiviert
        self.ml_enabled = self.settings.get('ml.enabled', False)
        self.regime_detector = None
        self.asset_clusterer = None
        self.coin_monitor = None
        self.current_regime = None
        self.regime_history = []
        self.ml_models_dir = self.settings.get('ml.models_dir', 'data/ml_models')

        # ML initialisieren, falls aktiviert
        if self.ml_enabled:
            self._initialize_ml_components()

        # Initialen Status melden
        self.logger.info(f"Trading bot initialized. Mode: {mode}, Strategy: {strategy_name}")
        self.logger.info(f"Trading pairs: {', '.join(self.trading_pairs)}")

        # ML-Status melden, falls aktiviert
        if self.ml_enabled:
            self.logger.info("ML-Komponenten aktiviert")

    def _initialize_ml_components(self) -> None:
        """
        Initialisiert die ML-Komponenten des Bots.
        """
        try:
            # MarketRegimeDetector initialisieren
            self.regime_detector = MarketRegimeDetector()

            # AssetClusterAnalyzer initialisieren
            self.asset_clusterer = AssetClusterAnalyzer()

            # NewCoinMonitor initialisieren
            self.coin_monitor = NewCoinMonitor()

            # Modelle laden, falls vorhanden
            self._load_ml_models()

            # Regime-Erkennung starten, falls nicht im Backtest-Modus
            if self.mode != "backtest":
                # In einem Thread starten, um nicht die Initialisierung zu blockieren
                threading.Thread(target=self._initial_ml_analysis,
                                 name="Initial-ML-Analysis",
                                 daemon=True).start()

            self.logger.info("ML-Komponenten initialisiert")
        except Exception as e:
            self.logger.error(f"Fehler bei der Initialisierung der ML-Komponenten: {e}")
            # ML deaktivieren, falls Initialisierung fehlschlägt
            self.ml_enabled = False

    def _initial_ml_analysis(self) -> None:
        """
        Führt die initiale ML-Analyse durch.
        """
        try:
            # 1. Marktregime erkennen
            self._update_market_regime()

            # 2. Asset-Clustering durchführen
            self._analyze_asset_clusters()

            # 3. Nach neuen Coins suchen
            if self.settings.get('ml.monitor_new_coins', True):
                self._check_for_new_coins()

            self.logger.info("Initiale ML-Analyse abgeschlossen")
        except Exception as e:
            self.logger.error(f"Fehler bei der initialen ML-Analyse: {e}")

    def _load_ml_models(self) -> bool:
        """
        Lädt gespeicherte ML-Modelle, falls vorhanden.

        Returns:
            True, wenn Modelle geladen wurden, sonst False
        """
        try:
            os.makedirs(self.ml_models_dir, exist_ok=True)

            # Regime-Modell laden
            regime_model_path = os.path.join(self.ml_models_dir, "regime_model.pkl")
            if os.path.exists(regime_model_path) and self.regime_detector:
                if self.regime_detector.load_model(regime_model_path):
                    self.logger.info(f"Regime-Modell aus {regime_model_path} geladen")
                    return True

            self.logger.info("Keine gespeicherten ML-Modelle gefunden")
            return False

        except Exception as e:
            self.logger.error(f"Fehler beim Laden der ML-Modelle: {e}")
            return False

    def _save_ml_models(self) -> bool:
        """
        Speichert die ML-Modelle.

        Returns:
            True, wenn Modelle gespeichert wurden, sonst False
        """
        try:
            os.makedirs(self.ml_models_dir, exist_ok=True)

            # Regime-Modell speichern
            if self.regime_detector and self.regime_detector.model_trained:
                regime_model_path = os.path.join(self.ml_models_dir, "regime_model.pkl")
                if self.regime_detector.save_model(regime_model_path):
                    self.logger.info(f"Regime-Modell unter {regime_model_path} gespeichert")
                    return True

            return False

        except Exception as e:
            self.logger.error(f"Fehler beim Speichern der ML-Modelle: {e}")
            return False

    def _update_market_regime(self) -> bool:
        """
        Aktualisiert das erkannte Marktregime.

        Returns:
            True, wenn das Regime aktualisiert wurde, sonst False
        """
        if not self.ml_enabled or not self.regime_detector:
            return False

        try:
            # 1. Wenn bereits ein Modell trainiert wurde
            if self.regime_detector.model_trained:
                # Daten für Regime-Erkennung laden
                success = self.regime_detector.load_market_data(
                    symbols=self.trading_pairs,
                    data_manager=self.data_manager,
                    timeframe="1d",
                    start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                    end_date=datetime.now().strftime('%Y-%m-%d')
                )

                if not success:
                    self.logger.error("Fehler beim Laden der Marktdaten für Regime-Erkennung")
                    return False

                # Features für die Regime-Erkennung extrahieren
                features_df = self.regime_detector.extract_market_features()

                if features_df.empty:
                    self.logger.error("Keine Features für Regime-Erkennung extrahiert")
                    return False

                # Aktuelles Regime vorhersagen
                latest_features = features_df.iloc[-1:].copy()
                regime = self.regime_detector.predict_regime(latest_features)

                if regime >= 0:
                    old_regime = self.current_regime
                    self.current_regime = regime

                    # Wenn sich das Regime geändert hat
                    if old_regime != regime:
                        regime_label = self.regime_detector.regime_labels.get(regime, f"Regime {regime}")
                        self.logger.info(f"Marktregime hat sich geändert: {old_regime} -> {regime} ({regime_label})")

                        # Regime-History aktualisieren
                        self.regime_history.append({
                            'from_regime': old_regime,
                            'to_regime': regime,
                            'timestamp': datetime.now().isoformat(),
                            'regime_label': regime_label
                        })

                        # Trading-Parameter für das neue Regime anpassen
                        self._adjust_parameters_for_regime(regime)

                    return True
                else:
                    self.logger.warning("Konnte kein gültiges Regime vorhersagen")

            # 2. Falls kein Modell trainiert wurde oder Vorhersage fehlgeschlagen
            # Vollständiges Training durchführen
            self.logger.info("Kein trainiertes Modell gefunden, starte Training")

            # Alle Trading-Paare für die Analyse verwenden
            symbols = list(self.trading_pairs)

            # Bitcoin hinzufügen, falls nicht bereits enthalten
            if not any(s.startswith("BTC/") for s in symbols):
                symbols.append("BTC/USDT")

            # Mehr Daten für das Training verwenden
            success = self.regime_detector.load_market_data(
                symbols=symbols,
                data_manager=self.data_manager,
                timeframe="1d",
                start_date=(datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
                end_date=datetime.now().strftime('%Y-%m-%d')
            )

            if not success:
                self.logger.error("Fehler beim Laden der Marktdaten für Regime-Training")
                return False

            # Features extrahieren
            features_df = self.regime_detector.extract_market_features()

            if features_df.empty:
                self.logger.error("Keine Features für Regime-Training extrahiert")
                return False

            # Modell trainieren
            success = self.regime_detector.train_regime_model(features_df)

            if not success:
                self.logger.error("Fehler beim Training des Regime-Modells")
                return False

            # Modell speichern
            self._save_ml_models()

            # Aktuelles Regime setzen
            latest_features = features_df.iloc[-1:].copy()
            regime = self.regime_detector.predict_regime(latest_features)

            if regime >= 0:
                self.current_regime = regime
                regime_label = self.regime_detector.regime_labels.get(regime, f"Regime {regime}")
                self.logger.info(f"Aktuelles Marktregime: {regime} ({regime_label})")

                # Regime-History initialisieren
                self.regime_history.append({
                    'from_regime': None,
                    'to_regime': regime,
                    'timestamp': datetime.now().isoformat(),
                    'regime_label': regime_label
                })

                # Parameter anpassen
                self._adjust_parameters_for_regime(regime)

                return True

            self.logger.warning("Konnte kein gültiges Regime ermitteln")
            return False

        except Exception as e:
            self.logger.error(f"Fehler bei der Marktregime-Erkennung: {e}")
            return False

    def _adjust_parameters_for_regime(self, regime: int) -> bool:
        """
        Passt die Trading-Parameter basierend auf dem aktuellen Marktregime an.

        Args:
            regime: ID des aktuellen Marktregimes

        Returns:
            True, wenn Parameter angepasst wurden, sonst False
        """
        if not self.ml_enabled or not self.regime_detector:
            return False

        try:
            # 1. Trading-Regeln für dieses Regime abrufen
            trading_rules = self.regime_detector.extract_trading_rules()

            if regime not in trading_rules:
                self.logger.warning(f"Keine Trading-Regeln für Regime {regime} gefunden")
                return False

            # 2. Regime-Label und Strategie-Empfehlung abrufen
            rule = trading_rules[regime]
            regime_label = rule.get('label', f"Regime {regime}")
            strategy_recommendation = rule.get('recommended_strategy', "Keine spezifische Empfehlung")

            self.logger.info(f"Passe Parameter für Regime {regime} ({regime_label}) an")
            self.logger.info(f"Strategie-Empfehlung: {strategy_recommendation}")

            # 3. Parameter basierend auf Regime anpassen
            params_adjusted = False

            # A. Bei bestimmten Regime-Typen vordefinierte Parameter verwenden
            regime_label_lower = regime_label.lower()

            if "bullish" in regime_label_lower:
                # Bullish-Parameter
                self.strategy.update_parameters({
                    'technical.rsi.oversold': 40,  # Höher setzen, da stärkere Aufwärtstrends
                    'technical.rsi.overbought': 75,  # Höher setzen, mehr Spielraum nach oben
                    'risk.stop_loss': 0.05,  # Größerer Stop-Loss, um Pullbacks zu überstehen
                    'risk.take_profit': 0.15,  # Höhere Gewinnziele
                    'risk.position_size': 0.10  # Größere Positionen
                })
                params_adjusted = True

            elif "bearish" in regime_label_lower:
                # Bearish-Parameter
                self.strategy.update_parameters({
                    'technical.rsi.oversold': 30,  # Standard-Wert
                    'technical.rsi.overbought': 60,  # Niedriger setzen, um früher zu verkaufen
                    'risk.stop_loss': 0.03,  # Engere Stop-Loss, schneller aussteigen
                    'risk.take_profit': 0.08,  # Kleinere Gewinnziele
                    'risk.position_size': 0.05  # Kleinere Positionen
                })
                params_adjusted = True

            elif "volatilität" in regime_label_lower:
                # Volatile-Markt-Parameter
                self.strategy.update_parameters({
                    'technical.rsi.oversold': 30,  # Standard-Wert
                    'technical.rsi.overbought': 70,  # Standard-Wert
                    'risk.stop_loss': 0.08,  # Größere Stop-Loss wegen höherer Volatilität
                    'risk.take_profit': 0.12,  # Höhere Gewinnziele wegen größerer Swings
                    'risk.position_size': 0.05  # Kleinere Positionen wegen höherem Risiko
                })
                params_adjusted = True

            # B. Bei "Altcoin-Stärke" die empfohlenen Top-Performer berücksichtigen
            if "altcoin-stärke" in regime_label_lower and 'top_performers' in rule:
                # Top-Performer als Trading-Paare hinzufügen/priorisieren
                top_performers = list(rule['top_performers'].keys())

                if top_performers:
                    # Mit bestehenden Paaren kombinieren
                    current_pairs = set(self.trading_pairs)
                    for pair in top_performers:
                        if not any(p.startswith(pair.split('/')[0]) for p in current_pairs):
                            current_pairs.add(pair)

                    # Update Trading-Paare
                    self.trading_pairs = list(current_pairs)
                    self.logger.info(f"Trading-Paare aktualisiert: {', '.join(self.trading_pairs)}")
                    params_adjusted = True

            # C. Weitere Anpassungen basierend auf Portfolio-Allokation
            if 'portfolio_allocation' in rule:
                allocation = rule['portfolio_allocation']

                # Risiko basierend auf Allokationsempfehlung anpassen
                altcoin_alloc = allocation.get('altcoins', 0)
                if altcoin_alloc < 0.2:
                    # Sehr defensive Einstellung
                    self.strategy.update_parameters({
                        'risk.position_size': 0.03,  # Kleine Positionen
                        'risk.max_open_positions': 3,  # Wenige offene Positionen
                        'risk.min_confidence': 0.8  # Hohe Konfidenz für Trades erforderlich
                    })
                elif altcoin_alloc > 0.6:
                    # Aggressive Einstellung
                    self.strategy.update_parameters({
                        'risk.position_size': 0.08,  # Größere Positionen
                        'risk.max_open_positions': 8,  # Mehr offene Positionen erlaubt
                        'risk.min_confidence': 0.6  # Niedrigere Konfidenz für Trades ausreichend
                    })
                params_adjusted = True

            if params_adjusted:
                self.logger.info("Parameter erfolgreich an das aktuelle Marktregime angepasst")
                return True
            else:
                self.logger.info("Keine spezifischen Parameteranpassungen für dieses Regime")
                return False

        except Exception as e:
            self.logger.error(f"Fehler bei der Anpassung der Parameter an das Marktregime: {e}")
            return False

    def _analyze_asset_clusters(self) -> bool:
        """
        Führt eine Asset-Cluster-Analyse durch.

        Returns:
            True, wenn die Analyse erfolgreich war, sonst False
        """
        if not self.ml_enabled or not self.asset_clusterer:
            return False

        try:
            # 1. Marktdaten laden
            symbols = list(self.trading_pairs)

            # Watchlist-Paare hinzufügen, falls vorhanden
            watchlist = self.settings.get('watchlist', [])
            for pair in watchlist:
                if pair not in symbols:
                    symbols.append(pair)

            success = self.asset_clusterer.load_market_data(
                symbols=symbols,
                data_manager=self.data_manager,
                timeframe="1d",
                start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                end_date=datetime.now().strftime('%Y-%m-%d')
            )

            if not success:
                self.logger.error("Fehler beim Laden der Marktdaten für Asset-Clustering")
                return False

            # 2. Korrelationsmatrix berechnen
            corr_matrix = self.asset_clusterer.calculate_correlation_matrix()

            if corr_matrix.empty:
                self.logger.error("Konnte keine Korrelationsmatrix berechnen")
                return False

            # 3. Asset-Features extrahieren
            feature_df = self.asset_clusterer.extract_asset_features()

            if feature_df.empty:
                self.logger.error("Konnte keine Asset-Features extrahieren")
                return False

            # 4. Clustering durchführen
            clusters = self.asset_clusterer.run_clustering(
                n_clusters=None,  # Automatisch bestimmen
                method='kmeans',
                correlation_based=True
            )

            if clusters.empty:
                self.logger.error("Clustering fehlgeschlagen")
                return False

            # 5. Portfolio-Empfehlung generieren
            portfolio = self.asset_clusterer.recommend_portfolio(n_assets=10)

            if not portfolio:
                self.logger.warning("Konnte keine Portfolio-Empfehlung generieren")
            else:
                # Log der Empfehlung
                assets = portfolio.get('assets', [])
                self.logger.info(f"Portfolio-Empfehlung: {', '.join(assets)}")

                # In Datei speichern
                os.makedirs("data/ml_analysis", exist_ok=True)
                with open("data/ml_analysis/portfolio_recommendation.json", 'w') as f:
                    json.dump(portfolio, f, indent=2)

            self.logger.info("Asset-Cluster-Analyse erfolgreich abgeschlossen")
            return True

        except Exception as e:
            self.logger.error(f"Fehler bei der Asset-Cluster-Analyse: {e}")
            return False

    def _check_for_new_coins(self) -> bool:
        """
        Überprüft, ob neue Coins am Markt verfügbar sind.

        Returns:
            True, wenn neue Coins gefunden wurden, sonst False
        """
        if not self.ml_enabled or not self.coin_monitor:
            return False

        try:
            # Nach neuen Coins suchen
            new_coins = self.coin_monitor.check_for_new_coins()

            if not new_coins:
                return False

            self.logger.info(f"{len(new_coins)} neue Coins entdeckt: {', '.join(new_coins)}")

            # Coins analysieren, die genügend Daten haben
            coins_to_analyze = self.coin_monitor.get_coins_for_analysis()

            if not coins_to_analyze:
                self.logger.info("Keine neuen Coins mit ausreichenden Daten für Analyse")
                return True

            # Coins analysieren
            for coin in coins_to_analyze:
                self._analyze_new_coin(coin)

            return True

        except Exception as e:
            self.logger.error(f"Fehler beim Überprüfen auf neue Coins: {e}")
            return False

    def _analyze_new_coin(self, coin: str) -> Dict[str, Any]:
        """
        Analysiert einen neuen Coin.

        Args:
            coin: Symbol des Coins (z.B. 'BTC/USDT')

        Returns:
            Analyse-Ergebnisse
        """
        if not self.ml_enabled or not self.coin_monitor or not self.asset_clusterer:
            return {"status": "ml_disabled"}

        try:
            # 1. Daten für den Coin laden
            base, quote = coin.split('/')

            # Datei finden
            binance_dir = os.path.join("data/market_data/binance")

            csv_path = None
            for f in os.listdir(binance_dir):
                if f.startswith(f"{base}_{quote}_1d"):
                    csv_path = os.path.join(binance_dir, f)
                    break

            if not csv_path:
                self.logger.warning(f"Keine Daten gefunden für {coin}")
                return {"status": "no_data"}

            # Daten laden
            df = pd.read_csv(csv_path)

            if 'timestamp' in df.columns:
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('date', inplace=True)

            # 2. Coin analysieren
            analysis_result = self.asset_clusterer.analyze_new_coin(coin, df)

            # 3. Ergebnis speichern
            if analysis_result and analysis_result.get('status') == 'analyzed':
                self.coin_monitor.update_coin_status(coin, analysis_result)

                # In Datei speichern
                os.makedirs("data/ml_analysis/new_coins", exist_ok=True)
                with open(f"data/ml_analysis/new_coins/{base}_{quote}_analysis.json", 'w') as f:
                    json.dump(analysis_result, f, indent=2)

                self.logger.info(f"Analyse für {coin} abgeschlossen")

                # Falls der Coin zu einem positiven Cluster gehört, zur Watchlist hinzufügen
                if 'predicted_cluster' in analysis_result:
                    cluster_id = analysis_result['predicted_cluster']

                    if (self.asset_clusterer.cluster_performances and
                            cluster_id in self.asset_clusterer.cluster_performances):

                        cluster_perf = self.asset_clusterer.cluster_performances[cluster_id]

                        if cluster_perf.get('mean_return', 0) > 0 and cluster_perf.get('sharpe_ratio', 0) > 0.5:
                            # Zur Watchlist hinzufügen
                            watchlist = self.settings.get('watchlist', [])
                            if coin not in watchlist:
                                watchlist.append(coin)
                                self.settings.set('watchlist', watchlist)
                                self.logger.info(f"{coin} zur Watchlist hinzugefügt (positives Cluster)")

            return analysis_result

        except Exception as e:
            self.logger.error(f"Fehler bei der Analyse von {coin}: {e}")
            return {"status": "error", "message": str(e)}

    def _initialize_strategy(self, strategy_name: str) -> Strategy:
        """
        Initialisiert die Trading-Strategie.

        Args:
            strategy_name: Name der Strategie

        Returns:
            Strategy-Objekt
        """
        # Standardstrategie ist Momentum
        if strategy_name == "default" or strategy_name == "momentum":
            strategy = MomentumStrategy(self.settings)
        elif strategy_name == "mean_reversion":
            strategy = MeanReversionStrategy(self.settings)
        elif strategy_name == "ml":
            strategy = MLStrategy(self.settings)
        else:
            self.logger.warning(f"Unknown strategy '{strategy_name}', using default")
            strategy = MomentumStrategy(self.settings)

        # Strategie-Parameter-Update-Methode patchen, falls nicht vorhanden
        if not hasattr(strategy, 'update_parameters'):
            def update_parameters(self, params):
                """
                Aktualisiert die Strategie-Parameter.

                Args:
                    params: Dictionary mit Parameternamen und -werten
                """
                for key, value in params.items():
                    # Parameter in der Strategie oder Settings setzen
                    key_parts = key.split('.')
                    if len(key_parts) > 1:
                        # Wenn es ein verschachtelter Parameter ist
                        # z.B. technical.rsi.period
                        if key_parts[0] == 'technical' and hasattr(self, key_parts[1]):
                            attr = getattr(self, key_parts[1])
                            if isinstance(attr, dict) and len(key_parts) > 2:
                                attr[key_parts[2]] = value
                            elif hasattr(attr, key_parts[2]):
                                setattr(attr, key_parts[2], value)
                        # Für risk und andere direkte Parameter
                        elif key_parts[0] == 'risk' and len(key_parts) > 1:
                            if hasattr(self, key_parts[1]):
                                setattr(self, key_parts[1], value)
                            else:
                                self.logger.debug(f"Setting risk parameter in settings: {key}")
                                self.settings.set(key, value)
                        else:
                            self.logger.debug(f"Setting parameter in settings: {key}")
                            self.settings.set(key, value)
                    else:
                        # Direkter Parameter
                        if hasattr(self, key):
                            setattr(self, key, value)
                        else:
                            self.logger.debug(f"Setting direct parameter in settings: {key}")
                            self.settings.set(key, value)

                self.logger.info(f"Strategy parameters updated: {params}")

            # Methode zur Strategie hinzufügen
            import types
            strategy.update_parameters = types.MethodType(update_parameters, strategy)

        return strategy

    def connect(self) -> bool:
        """
        Stellt eine Verbindung zum Exchange her.

        Returns:
            True bei erfolgreicher Verbindung, False sonst
        """
        try:
            success = self.exchange.connect()
            if success:
                self.logger.info(f"Connected to exchange in {self.mode} mode")

                # Überprüfe API-Limits und Kontostatus
                if self.mode in ['live', 'paper']:
                    account_info = self.exchange.get_account_info()
                    if account_info:
                        self.logger.info(f"Account status: {account_info.get('status', 'Unknown')}")

                        # Limits abrufen
                        limits = account_info.get('limits', {})
                        if limits:
                            self.logger.info(f"API rate limits: {limits}")

            return success
        except Exception as e:
            self.logger.error(f"Failed to connect to exchange: {e}")
            self._notify_error("connection_error", f"Failed to connect to exchange: {e}")
            return False

    def add_trade_callback(self, callback: Callable[[Position], None]) -> None:
        """
        Fügt einen Callback hinzu, der bei jedem Trade aufgerufen wird.

        Args:
            callback: Callback-Funktion, die eine Position als Parameter erhält
        """
        self.on_trade_callbacks.append(callback)

    def add_error_callback(self, callback: Callable[[str, str], None]) -> None:
        """
        Fügt einen Callback hinzu, der bei Fehlern aufgerufen wird.

        Args:
            callback: Callback-Funktion, die Fehlertyp und -nachricht erhält
        """
        self.on_error_callbacks.append(callback)

    def add_status_update_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Fügt einen Callback hinzu, der bei Statusaktualisierungen aufgerufen wird.

        Args:
            callback: Callback-Funktion, die den Status-Dictionary erhält
        """
        self.on_status_update_callbacks.append(callback)

    def _notify_trade(self, position: Position) -> None:
        """
        Benachrichtigt alle registrierten Callbacks über einen Trade.

        Args:
            position: Position-Objekt des Trades
        """
        for callback in self.on_trade_callbacks:
            try:
                callback(position)
            except Exception as e:
                self.logger.error(f"Error in trade callback: {e}")

    def _notify_error(self, error_type: str, error_message: str) -> None:
        """
        Benachrichtigt alle registrierten Callbacks über einen Fehler.

        Args:
            error_type: Typ des Fehlers
            error_message: Fehlermeldung
        """
        # API-Fehler zählen, um wiederholte Probleme zu erkennen
        if error_type.startswith("api_") or error_type == "connection_error":
            current_time = time.time()

            # Reset-Fenster für API-Fehler
            if (self.last_api_error_time is None or
                    current_time - self.last_api_error_time > self.api_error_window):
                self.api_error_count = 1
            else:
                self.api_error_count += 1

            self.last_api_error_time = current_time

            # Sicherheitsabschaltung bei zu vielen API-Fehlern
            if self.api_error_count >= self.max_api_errors:
                self.logger.critical(
                    f"Too many API errors ({self.api_error_count}) within {self.api_error_window} seconds. "
                    f"Shutting down for safety."
                )
                self.stop()

                # Kritischer Fehler an alle Callback-Handler
                error_message = f"Bot stopped due to excessive API errors: {error_message}"
                error_type = "critical_api_failure"

        for callback in self.on_error_callbacks:
            try:
                callback(error_type, error_message)
            except Exception as e:
                self.logger.error(f"Error in error callback: {e}")

    def _notify_status_update(self, status: Dict[str, Any]) -> None:
        """
        Benachrichtigt alle registrierten Callbacks über eine Statusaktualisierung.

        Args:
            status: Status-Dictionary
        """
        # Nur signifikante Änderungen melden
        if self._is_significant_status_change(self.last_status, status):
            for callback in self.on_status_update_callbacks:
                try:
                    callback(status)
                except Exception as e:
                    self.logger.error(f"Error in status update callback: {e}")

            # Status speichern
            self.last_status = status.copy()

    def _is_significant_status_change(self, old_status: Dict[str, Any], new_status: Dict[str, Any]) -> bool:
        """
        Überprüft, ob sich der Status signifikant geändert hat.

        Args:
            old_status: Vorheriger Status
            new_status: Neuer Status

        Returns:
            True, wenn sich der Status signifikant geändert hat
        """
        if not old_status:
            return True

        # Änderungen, die immer als signifikant gelten
        if old_status.get('running') != new_status.get('running'):
            return True

        # Neue offene Positionen
        old_positions = {p['id']: p for p in old_status.get('open_positions', [])}
        new_positions = {p['id']: p for p in new_status.get('open_positions', [])}

        if set(old_positions.keys()) != set(new_positions.keys()):
            return True

        # Signifikante P/L-Änderung (mehr als 1%)
        old_pnl = old_status.get('profit_loss_pct', 0)
        new_pnl = new_status.get('profit_loss_pct', 0)

        if abs(new_pnl - old_pnl) > 1.0:
            return True

        # Positionen mit signifikanten P/L-Änderungen
        for pos_id, new_pos in new_positions.items():
            if pos_id in old_positions:
                old_pos = old_positions[pos_id]
                old_pnl = old_pos.get('unrealized_pnl_pct', 0)
                new_pnl = new_pos.get('unrealized_pnl_pct', 0)

                if abs(new_pnl - old_pnl) > 2.0:  # Mehr als 2% Änderung
                    return True

        # Standard-Update-Intervall (ca. alle 5 Minuten)
        old_timestamp = datetime.fromisoformat(old_status.get('timestamp', '2000-01-01T00:00:00'))
        new_timestamp = datetime.fromisoformat(new_status.get('timestamp', '2000-01-01T00:00:00'))

        if (new_timestamp - old_timestamp).total_seconds() > 300:
            return True

        # Regime-Änderung ist immer signifikant
        if 'current_regime' in old_status and 'current_regime' in new_status:
            if old_status['current_regime'] != new_status['current_regime']:
                return True

        return False

    def _update_data_cache(self, symbol: str) -> pd.DataFrame:
        """
        Aktualisiert den Daten-Cache für ein Symbol.

        Args:
            symbol: Handelssymbol

        Returns:
            DataFrame mit OHLCV-Daten
        """
        timeframe = self.settings.get('timeframes.analysis', '1h')
        # Optionale Einstellung für die Datenquelle
        data_source = self.settings.get('data.source', 'exchange')

        try:
            # Daten abrufen über den geeigneten Weg
            if data_source == 'exchange' or self.mode == 'live' or self.mode == 'paper':
                # Direkt vom Exchange für Live/Paper-Trading abrufen
                df = self.exchange.get_ohlcv(symbol, timeframe)
            else:
                # Alternativ den DataManager für historische oder alternative Daten verwenden
                source = self.settings.get('data.source_name', 'binance')
                df = self.data_manager.get_historical_data(
                    symbol=symbol,
                    source=source,
                    timeframe=timeframe,
                    start_date=datetime.now() - timedelta(days=30),  # Letzte 30 Tage
                    use_cache=True
                )

            # Sicherstellen, dass wir ausreichend Daten haben
            if len(df) < self.settings.get('data.min_candles', 50):
                self.logger.warning(
                    f"Insufficient data for {symbol}, got {len(df)} candles, "
                    f"need at least {self.settings.get('data.min_candles', 50)}"
                )

            # Daten synchron für Thread-Sicherheit im Cache speichern
            with self.data_cache_lock:
                self.data_cache[symbol] = df

            return df
        except Exception as e:
            error_msg = f"Error updating data cache for {symbol}: {e}"
            self.logger.error(error_msg)
            self._notify_error("data_error", error_msg)

            # Vorhandene Daten zurückgeben, falls verfügbar
            with self.data_cache_lock:
                if symbol in self.data_cache:
                    return self.data_cache[symbol]

            # Leerer DataFrame, falls keine Daten verfügbar
            return pd.DataFrame()

    def _update_all_data_parallel(self) -> Dict[str, pd.DataFrame]:
        """
        Aktualisiert den Daten-Cache für alle Symbole parallel.

        Returns:
            Dictionary mit Symbol als Schlüssel und DataFrame als Wert
        """
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Alle Symbole parallel verarbeiten
            futures =