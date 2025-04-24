#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML-Enhanced Backtest Framework für den Trading Bot.
Dieses Framework kombiniert umfassende Backtests mit Machine Learning für:
1. Intelligente Parameteroptimierung
2. Marktregime-Erkennung
3. Asset-Clustering
4. Adaptives Trading basierend auf Marktphasen
"""

import os
import json
import time
import itertools
import concurrent.futures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Union
from pathlib import Path
import argparse
import warnings
import pickle
import logging

# Machine Learning Bibliotheken
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Bayessche Optimierung
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from skopt import gp_minimize, forest_minimize
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective

# Trading Bot Komponenten
from config.settings import Settings
from core.trading_bot import TradingBot

# Logging einrichten
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backtest_framework.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("MLBacktestFramework")


class MarketRegimeDetector:
    """
    Marktregime-Erkennung mittels Machine Learning.
    Identifiziert unterschiedliche Marktphasen und deren Eigenschaften.
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

    def load_market_data(self, symbols: List[str] = None,
                         timeframe: str = "1d",
                         start_date: str = None,
                         end_date: str = None) -> bool:
        """
        Lädt Marktdaten für die angegebenen Symbole.

        Args:
            symbols: Liste der zu ladenden Symbole (oder None für alle)
            timeframe: Zeitrahmen der Daten
            start_date: Startdatum im Format 'YYYY-MM-DD'
            end_date: Enddatum im Format 'YYYY-MM-DD'

        Returns:
            True, wenn Daten erfolgreich geladen wurden
        """
        try:
            # Verzeichnis durchsuchen
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
                    logger.warning(f"Keine Daten gefunden für {symbol} mit Timeframe {timeframe}")
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
                logger.info(f"Daten für {symbol} geladen: {len(df)} Einträge")

            if not self.market_data:
                logger.error("Keine Marktdaten geladen")
                return False

            logger.info(f"Marktdaten für {len(self.market_data)} Symbole geladen")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Laden der Marktdaten: {e}")
            return False

    def extract_market_features(self) -> pd.DataFrame:
        """
        Extrahiert relevante Features für die Regime-Erkennung.

        Returns:
            DataFrame mit täglichen Features für alle Märkte
        """
        if not self.market_data:
            logger.error("Keine Marktdaten geladen")
            return pd.DataFrame()

        try:
            # Bitcoin-Daten als Referenz (sollte vorhanden sein)
            btc_symbol = next((s for s in self.market_data.keys() if s.startswith('BTC')), None)

            if not btc_symbol:
                logger.error("Keine Bitcoin-Daten verfügbar")
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

            # Trend Strength (Directional Movement Index)
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

            logger.info(f"Features extrahiert: {len(features_df)} Zeitpunkte, {len(features_df.columns)} Features")
            return features_df

        except Exception as e:
            logger.error(f"Fehler bei der Feature-Extraktion: {e}")
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
            logger.error("Keine Features für das Training vorhanden")
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
                    score = silhouette_score(X_scaled, labels)
                    scores.append((n_clusters, score))

                # Beste Anzahl Cluster auswählen
                self.n_regimes = max(scores, key=lambda x: x[1])[0]
                logger.info(f"Optimale Anzahl an Clustern: {self.n_regimes}")

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
            logger.info(f"Regime-Modell erfolgreich trainiert: {self.n_regimes} Regime identifiziert")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Training des Regime-Modells: {e}")
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
            logger.error("Modell nicht trainiert")
            return -1

        try:
            # Features normalisieren
            features_scaled = self.scaler.transform(features)

            # Regime vorhersagen
            regime = self.regime_model.predict(features_scaled)[0]

            self.current_regime = regime
            logger.info(f"Aktuelles Marktregime: {regime} - {self.regime_labels.get(regime, 'Unbekannt')}")

            return regime

        except Exception as e:
            logger.error(f"Fehler bei der Regime-Vorhersage: {e}")
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

            logger.info(f"Regime-Transitionen analysiert")

        except Exception as e:
            logger.error(f"Fehler bei der Analyse der Regime-Übergänge: {e}")

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

                    logger.info(f"Regime {regime} charakterisiert als: {self.regime_labels[regime]}")

        except Exception as e:
            logger.error(f"Fehler bei der Analyse der Regime-Charakteristiken: {e}")

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

            logger.info(f"Performance-Analyse pro Regime abgeschlossen")

        except Exception as e:
            logger.error(f"Fehler bei der Analyse der Performance pro Regime: {e}")

    def extract_trading_rules(self) -> Dict[int, Dict[str, Any]]:
        """
        Extrahiert Trading-Regeln für jedes Marktregime.

        Returns:
            Dictionary mit Regeln für jedes Regime
        """
        if not self.model_trained or self.regime_performances is None:
            logger.error("Modell nicht trainiert oder keine Performance-Daten")
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

            logger.info(f"Trading-Regeln für {len(trading_rules)} Regime extrahiert")
            return trading_rules

        except Exception as e:
            logger.error(f"Fehler bei der Extraktion von Trading-Regeln: {e}")
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
            logger.error("Kein trainiertes Modell zum Speichern verfügbar")
            return False

        try:
            # Verzeichnis erstellen, falls nicht vorhanden
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Modell als Pickle speichern
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

            logger.info(f"Regime-Modell gespeichert unter {filepath}")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Speichern des Modells: {e}")
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

            logger.info(f"Regime-Modell geladen von {filepath}")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Laden des Modells: {e}")
            return False

    def generate_regime_report(self, output_dir: str) -> str:
        """
        Erstellt einen detaillierten Bericht über die identifizierten Marktregimes.

        Args:
            output_dir: Verzeichnis für die Ausgabe des Berichts

        Returns:
            Pfad zum Bericht oder leerer String bei Fehler
        """
        if not self.model_trained:
            logger.error("Kein trainiertes Modell für den Bericht verfügbar")
            return ""

        try:
            # Verzeichnis erstellen, falls nicht vorhanden
            os.makedirs(output_dir, exist_ok=True)

            # Berichtspfad
            report_path = os.path.join(output_dir, "regime_report.html")

            # HTML-Bericht erstellen
            with open(report_path, 'w') as f:
                f.write("""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Marktregime-Analyse Bericht</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; }
                        h1, h2, h3 { color: #333; }
                        .container { max-width: 1200px; margin: 0 auto; }
                        .regime-box { 
                            border: 1px solid #ddd; 
                            border-radius: 5px; 
                            padding: 15px; 
                            margin-bottom: 20px;
                            background-color: #f9f9f9;
                        }
                        .regime-box h3 { 
                            margin-top: 0;
                            padding-bottom: 10px;
                            border-bottom: 1px solid #ddd;
                        }
                        .performance-table {
                            width: 100%;
                            border-collapse: collapse;
                        }
                        .performance-table th, .performance-table td {
                            border: 1px solid #ddd;
                            padding: 8px;
                            text-align: left;
                        }
                        .performance-table th {
                            background-color: #f2f2f2;
                        }
                        .positive { color: green; }
                        .negative { color: red; }
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
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>Marktregime-Analyse Bericht</h1>
                        <p>Erstelldatum: %s</p>
                        <h2>Identifizierte Marktregimes</h2>
                """ % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

                # Für jedes Regime einen Abschnitt hinzufügen
                trading_rules = self.extract_trading_rules()

                for regime, rule in trading_rules.items():
                    # Durchschnittliche Performance formatieren
                    avg_perf = rule["avg_performance"] * 100
                    perf_class = "positive" if avg_perf > 0 else "negative"
                    avg_perf_str = f'<span class="{perf_class}">{avg_perf:.2f}%</span>'

                    f.write(f"""
                        <div class="regime-box">
                            <h3>Regime {regime}: {rule["label"]}</h3>
                            <p><strong>Durchschnittliche Performance:</strong> {avg_perf_str}</p>
                            <p><strong>Positive Performer:</strong> {rule["pct_positive"]:.1f}%</p>
                            <p><strong>Empfohlene Strategie:</strong> {rule["recommended_strategy"]}</p>

                            <h4>Portfolio-Allokation</h4>
                            <div class="allocation-chart">
                    """)

                    # Portfolio-Allokations-Chart
                    colors = {
                        "altcoins": "#6f42c1",  # Lila
                        "bitcoin": "#f8b739",  # Bitcoin-Orange
                        "stablecoins": "#20c997"  # Grün
                    }

                    alloc = rule["portfolio_allocation"]
                    for asset, percentage in alloc.items():
                        width = percentage * 100
                        f.write(f"""
                            <div class="allocation-segment" 
                                 style="width: {width}%; background-color: {colors[asset]};">
                                {asset.capitalize()} {percentage * 100:.0f}%
                            </div>
                        """)

                    f.write("""
                            </div>

                            <h4>Top Performer</h4>
                            <table class="performance-table">
                                <tr>
                                    <th>Symbol</th>
                                    <th>Durchschnittliche tägliche Rendite</th>
                                </tr>
                    """)

                    # Top Performer
                    for symbol, perf in rule["top_performers"].items():
                        perf_pct = perf * 100
                        perf_class = "positive" if perf_pct > 0 else "negative"
                        f.write(f"""
                            <tr>
                                <td>{symbol}</td>
                                <td class="{perf_class}">{perf_pct:.2f}%</td>
                            </tr>
                        """)

                    f.write("""
                            </table>

                            <h4>Schwächste Performer</h4>
                            <table class="performance-table">
                                <tr>
                                    <th>Symbol</th>
                                    <th>Durchschnittliche tägliche Rendite</th>
                                </tr>
                    """)

                    # Schwächste Performer
                    for symbol, perf in rule["worst_performers"].items():
                        perf_pct = perf * 100
                        perf_class = "positive" if perf_pct > 0 else "negative"
                        f.write(f"""
                            <tr>
                                <td>{symbol}</td>
                                <td class="{perf_class}">{perf_pct:.2f}%</td>
                            </tr>
                        """)

                    f.write("""
                            </table>
                        </div>
                    """)

                # Ende des HTML-Dokuments
                f.write("""
                        <h2>Regime-Übergangswahrscheinlichkeiten</h2>
                        <p>Diese Tabelle zeigt die Wahrscheinlichkeit des Übergangs von einem Regime zu einem anderen:</p>
                        <table class="performance-table">
                            <tr>
                                <th>Von \ Zu</th>
                """)

                # Übergangsmatrix-Tabellenkopf
                for regime in range(self.n_regimes):
                    regime_label = self.regime_labels.get(regime, f"Regime {regime}")
                    f.write(f"<th>{regime_label}</th>")

                f.write("</tr>")

                # Übergangsmatrix-Daten
                if self.regime_transitions is not None:
                    for i in range(self.n_regimes):
                        regime_label = self.regime_labels.get(i, f"Regime {i}")
                        f.write(f"<tr><td><strong>{regime_label}</strong></td>")

                        for j in range(self.n_regimes):
                            probability = self.regime_transitions[i, j] * 100
                            color_intensity = int(min(probability * 2.5, 255))
                            bg_color = f"rgba(0, 128, 255, {probability / 100})"

                            f.write(f"""
                                <td style="background-color: {bg_color}">
                                    {probability:.1f}%
                                </td>
                            """)

                        f.write("</tr>")

                f.write("""
                        </table>
                    </div>
                </body>
                </html>
                """)

            logger.info(f"Regime-Bericht erstellt: {report_path}")
            return report_path

        except Exception as e:
            logger.error(f"Fehler bei der Erstellung des Regime-Berichts: {e}")
            return ""


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

    def load_market_data(self, symbols: List[str] = None,
                         timeframe: str = "1d",
                         start_date: str = None,
                         end_date: str = None) -> bool:
        """
        Lädt Marktdaten für die angegebenen Symbole.

        Args:
            symbols: Liste der zu ladenden Symbole (oder None für alle)
            timeframe: Zeitrahmen der Daten
            start_date: Startdatum im Format 'YYYY-MM-DD'
            end_date: Enddatum im Format 'YYYY-MM-DD'

        Returns:
            True, wenn Daten erfolgreich geladen wurden
        """
        try:
            # Verzeichnis durchsuchen
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
                    logger.warning(f"Keine Daten gefunden für {symbol} mit Timeframe {timeframe}")
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
                logger.info(f"Daten für {symbol} geladen: {len(df)} Einträge")

            if not self.market_data:
                logger.error("Keine Marktdaten geladen")
                return False

            logger.info(f"Marktdaten für {len(self.market_data)} Symbole geladen")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Laden der Marktdaten: {e}")
            return False

    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """
        Berechnet die Korrelationsmatrix zwischen allen Assets.

        Returns:
            Korrelationsmatrix als DataFrame
        """
        if not self.market_data:
            logger.error("Keine Marktdaten vorhanden")
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

            logger.info(f"Korrelationsmatrix berechnet für {len(correlation_matrix)} Assets")
            return correlation_matrix

        except Exception as e:
            logger.error(f"Fehler bei der Berechnung der Korrelationsmatrix: {e}")
            return pd.DataFrame()

    def extract_asset_features(self) -> pd.DataFrame:
        """
        Extrahiert Features für das Asset-Clustering.

        Returns:
            DataFrame mit Features für jedes Asset
        """
        if not self.market_data:
            logger.error("Keine Marktdaten vorhanden")
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
                    logger.warning(f"Nicht genügend Daten für {symbol}, überspringe")
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
                logger.error("Keine Features extrahiert")
                return pd.DataFrame()

            # Symbol als Index
            feature_df.set_index('symbol', inplace=True)

            # NaN-Werte durch 0 ersetzen
            feature_df = feature_df.fillna(0)

            self.feature_data = feature_df

            logger.info(f"Features für {len(feature_df)} Assets extrahiert")
            return feature_df

        except Exception as e:
            logger.error(f"Fehler bei der Extraktion der Asset-Features: {e}")
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
                    logger.error("Korrelationsmatrix ist leer")
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
                    logger.error("Keine Feature-Daten vorhanden")
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
                        score = silhouette_score(X, labels)
                        scores.append((k, score))

                # Beste Anzahl Cluster auswählen
                n_clusters = max(scores, key=lambda x: x[1])[0]
                logger.info(f"Optimale Anzahl an Clustern: {n_clusters}")

            # Clustering durchführen
            if method == 'kmeans':
                self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = self.cluster_model.fit_predict(X)
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
                logger.error(f"Unbekannte Clustering-Methode: {method}")
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

            logger.info(f"Clustering abgeschlossen: {len(result_df)} Assets in {len(set(cluster_labels))} Clustern")
            return result_df

        except Exception as e:
            logger.error(f"Fehler beim Clustering: {e}")
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
            logger.error("Keine Cluster-Zuweisungen oder Marktdaten vorhanden")
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

            logger.info(f"Cluster-Performance-Analyse abgeschlossen für {len(cluster_stats)} Cluster")
            return cluster_stats

        except Exception as e:
            logger.error(f"Fehler bei der Analyse der Cluster-Performance: {e}")
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
                logger.error("Keine Cluster-Daten oder Performance-Analyse vorhanden")
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
                    logger.warning("Keine positiv performenden Cluster gefunden")
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

                logger.info(f"Portfolio mit {len(recommended_assets)} Assets empfohlen")
                return recommendations

            except Exception as e:
                logger.error(f"Fehler bei der Portfolio-Empfehlung: {e}")
                return {}

        def generate_cluster_report(self, output_dir: str) -> str:
            """
            Erstellt einen detaillierten Bericht über die Asset-Cluster.

            Args:
                output_dir: Verzeichnis für die Ausgabe des Berichts

            Returns:
                Pfad zum Bericht oder leerer String bei Fehler
            """
            if self.clusters is None or self.cluster_performances is None:
                logger.error("Keine Cluster-Daten oder Performance-Analyse vorhanden")
                return ""

            try:
                # Verzeichnis erstellen, falls nicht vorhanden
                os.makedirs(output_dir, exist_ok=True)

                # Berichtspfad
                report_path = os.path.join(output_dir, "cluster_report.html")

                # HTML-Bericht erstellen
                with open(report_path, 'w') as f:
                    f.write("""
                                <!DOCTYPE html>
                                <html>
                                <head>
                                    <title>Asset-Cluster Analyse Bericht</title>
                                    <style>
                                        body { font-family: Arial, sans-serif; margin: 20px; }
                                        h1, h2, h3 { color: #333; }
                                        .container { max-width: 1200px; margin: 0 auto; }
                                        .cluster-box { 
                                            border: 1px solid #ddd; 
                                            border-radius: 5px; 
                                            padding: 15px; 
                                            margin-bottom: 20px;
                                            background-color: #f9f9f9;
                                        }
                                        .cluster-box h3 { 
                                            margin-top: 0;
                                            padding-bottom: 10px;
                                            border-bottom: 1px solid #ddd;
                                        }
                                        .asset-table {
                                            width: 100%;
                                            border-collapse: collapse;
                                        }
                                        .asset-table th, .asset-table td {
                                            border: 1px solid #ddd;
                                            padding: 8px;
                                            text-align: left;
                                        }
                                        .asset-table th {
                                            background-color: #f2f2f2;
                                        }
                                        .positive { color: green; }
                                        .negative { color: red; }
                                        .asset-badge {
                                            display: inline-block;
                                            padding: 2px 8px;
                                            margin: 2px;
                                            background-color: #e9ecef;
                                            border-radius: 10px;
                                            font-size: 14px;
                                        }
                                        .representative {
                                            font-weight: bold;
                                            background-color: #007bff;
                                            color: white;
                                        }
                                    </style>
                                </head>
                                <body>
                                    <div class="container">
                                        <h1>Asset-Cluster Analyse Bericht</h1>
                                        <p>Erstelldatum: %s</p>
                                        <h2>Identifizierte Asset-Cluster</h2>
                                """ % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

                    # Für jeden Cluster einen Abschnitt hinzufügen
                    for cluster_id, stats in sorted(self.cluster_performances.items()):
                        # Leistungskennzahlen formatieren
                        mean_return = stats['mean_return'] * 100
                        return_class = "positive" if mean_return > 0 else "negative"
                        mean_return_str = f'<span class="{return_class}">{mean_return:.2f}%</span>'

                        max_dd = stats['max_drawdown'] * 100
                        dd_class = "negative" if max_dd < 0 else "positive"
                        max_dd_str = f'<span class="{dd_class}">{max_dd:.2f}%</span>'

                        win_rate = stats['win_rate'] * 100

                        # Cluster-Label
                        if cluster_id >= 0:
                            if stats['sharpe_ratio'] > 0.5:
                                cluster_label = "Performante Gruppe"
                            elif stats['sharpe_ratio'] > 0:
                                cluster_label = "Moderate Gruppe"
                            else:
                                cluster_label = "Unterdurchschnittliche Gruppe"

                            if stats['volatility'] > 0.03:
                                cluster_label += " (Hohe Volatilität)"
                            elif stats['volatility'] < 0.01:
                                cluster_label += " (Niedrige Volatilität)"

                            if stats['intra_correlation'] > 0.7:
                                cluster_label += " - Stark korreliert"
                            elif stats['intra_correlation'] < 0.3:
                                cluster_label += " - Schwach korreliert"
                        else:
                            cluster_label = "Ausreißer"

                        f.write(f"""
                                        <div class="cluster-box">
                                            <h3>Cluster {cluster_id}: {cluster_label}</h3>
                                            <p><strong>Anzahl Assets:</strong> {stats['count']}</p>
                                            <p><strong>Durchschnittliche tägliche Rendite:</strong> {mean_return_str}</p>
                                            <p><strong>Volatilität:</strong> {stats['volatility'] * 100:.2f}%</p>
                                            <p><strong>Sharpe Ratio:</strong> {stats['sharpe_ratio']:.2f}</p>
                                            <p><strong>Maximaler Drawdown:</strong> {max_dd_str}</p>
                                            <p><strong>Win Rate:</strong> {win_rate:.1f}%</p>
                                            <p><strong>Interne Korrelation:</strong> {stats['intra_correlation']:.2f}</p>

                                            <h4>Assets in diesem Cluster</h4>
                                            <div>
                                    """)

                        # Assets auflisten
                        rep_asset = stats.get('representative_asset', "")

                        for asset in stats['assets']:
                            if asset == rep_asset:
                                f.write(f'<span class="asset-badge representative">{asset}</span> ')
                            else:
                                f.write(f'<span class="asset-badge">{asset}</span> ')

                        f.write("""
                                            </div>
                                        </div>
                                    """)

                    # Portfolioempfehlung
                    portfolio = self.recommend_portfolio()

                    if portfolio and 'assets' in portfolio and portfolio['assets']:
                        f.write("""
                                        <h2>Portfolio-Empfehlung</h2>
                                        <div class="cluster-box">
                                            <h3>Empfohlene Assets</h3>
                                            <p><strong>Strategie:</strong> %s</p>
                                            <table class="asset-table">
                                                <tr>
                                                    <th>Asset</th>
                                                    <th>Empfohlene Allokation</th>
                                                </tr>
                                    """ % portfolio.get('strategy', 'Diversifizierte Allokation'))

                        # Assets und Allokation
                        allocation = portfolio.get('allocation', {})

                        for asset in portfolio['assets']:
                            alloc = allocation.get(asset, 1.0 / len(portfolio['assets']))
                            f.write(f"""
                                            <tr>
                                                <td>{asset}</td>
                                                <td>{alloc * 100:.1f}%</td>
                                            </tr>
                                        """)

                        f.write("""
                                            </table>
                                        </div>
                                    """)

                    # Ende des HTML-Dokuments
                    f.write("""
                                    </div>
                                </body>
                                </html>
                                """)

                logger.info(f"Cluster-Bericht erstellt: {report_path}")
                return report_path

            except Exception as e:
                logger.error(f"Fehler bei der Erstellung des Cluster-Berichts: {e}")
                return ""

        class MLBacktestOptimizer:
            """
            Optimiert Backtest-Parameter mit Machine Learning.
            Verwendet Bayessche Optimierung und andere ML-Techniken.
            """

            def __init__(self, base_settings: Settings = None, output_dir: str = None):
                """
                Initialisiert den ML-Backtest-Optimierer.

                Args:
                    base_settings: Basis-Einstellungen für die Optimierung
                    output_dir: Ausgabeverzeichnis für Optimierungsergebnisse
                """
                self.base_settings = base_settings or Settings()
                self.output_dir = output_dir or f"data/ml_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.param_space = {}
                self.optimization_history = []
                self.best_params = None
                self.best_score = float('-inf')
                self.ml_model = None
                self.evaluation_function = None
                self.feature_importance = None

            def define_parameter_space(self, param_space: Dict[str, Any]) -> None:
                """
                Definiert den Parameterraum für die Optimierung.

                Args:
                    param_space: Dictionary mit Parameternamen und Wertebereichen
                """
                self.param_space = {}

                for param_name, param_values in param_space.items():
                    if isinstance(param_values, list):
                        if all(isinstance(v, (int, float)) for v in param_values):
                            # Numerische Liste - als Bereich interpretieren
                            if all(isinstance(v, int) for v in param_values):
                                self.param_space[param_name] = Integer(min(param_values), max(param_values))
                            else:
                                self.param_space[param_name] = Real(min(param_values), max(param_values))
                        else:
                            # Kategorische Liste
                            self.param_space[param_name] = Categorical(param_values)
                    elif isinstance(param_values, tuple) and len(param_values) == 2:
                        # Tuple als Bereich (min, max)
                        if all(isinstance(v, int) for v in param_values):
                            self.param_space[param_name] = Integer(param_values[0], param_values[1])
                        else:
                            self.param_space[param_name] = Real(param_values[0], param_values[1])
                    else:
                        logger.warning(f"Unbekanntes Parameterformat für {param_name}: {param_values}")

                logger.info(f"Parameterraum mit {len(self.param_space)} Parametern definiert")

            def set_evaluation_function(self, eval_func: Callable) -> None:
                """
                Setzt die Evaluierungsfunktion für die Optimierung.

                Args:
                    eval_func: Funktion, die Parameterwerte erhält und einen Score zurückgibt
                """
                self.evaluation_function = eval_func
                logger.info("Evaluierungsfunktion gesetzt")

            def run_bayesian_optimization(self, n_calls: int = 50, init_points: int = 10,
                                          random_state: int = 42) -> Dict[str, Any]:
                """
                Führt die Bayessche Optimierung durch.

                Args:
                    n_calls: Anzahl der Evaluierungsaufrufe
                    init_points: Anzahl an zufälligen Initialpunkten
                    random_state: Zufallssamen für Reproduzierbarkeit

                Returns:
                    Dictionary mit optimierten Parametern
                """
                if not self.param_space:
                    logger.error("Kein Parameterraum definiert")
                    return {}

                if not self.evaluation_function:
                    logger.error("Keine Evaluierungsfunktion definiert")
                    return {}

                try:
                    # Verzeichnis erstellen, falls nicht vorhanden
                    os.makedirs(self.output_dir, exist_ok=True)

                    # Parameter-Namen für die Zielfunktion
                    param_names = list(self.param_space.keys())

                    # Bayessche Optimierungsfunktion erstellen
                    @use_named_args(dimensions=list(self.param_space.values()))
                    def objective_function(**params):
                        # Parameter in Dictionary umwandeln
                        param_dict = {name: params[name] for name in param_names}

                        # Evaluieren
                        try:
                            score = self.evaluation_function(param_dict)

                            # Für Minimierungsprobleme negieren
                            if hasattr(self, 'minimize') and self.minimize:
                                score = -score

                            # In History speichern
                            self.optimization_history.append({
                                'params': param_dict.copy(),
                                'score': score
                            })

                            # Bestes Ergebnis aktualisieren
                            if score > self.best_score:
                                self.best_score = score
                                self.best_params = param_dict.copy()

                                # Bestes Ergebnis speichern
                                with open(os.path.join(self.output_dir, 'best_params.json'), 'w') as f:
                                    json.dump({'params': self.best_params, 'score': self.best_score}, f, indent=2)

                            logger.info(f"Evaluierte Parameter mit Score {score:.4f}")
                            return score
                        except Exception as e:
                            logger.error(f"Fehler bei der Evaluation: {e}")
                            return float('-inf')

                    # Bayessche Optimierung durchführen
                    logger.info(f"Starte Bayessche Optimierung mit {n_calls} Aufrufen...")
                    result = gp_minimize(
                        func=objective_function,
                        dimensions=list(self.param_space.values()),
                        n_calls=n_calls,
                        n_initial_points=init_points,
                        random_state=random_state,
                        verbose=True
                    )

                    # Ergebnisse extrahieren
                    best_params_values = result.x
                    best_params = {param_names[i]: best_params_values[i] for i in range(len(param_names))}
                    best_score = result.fun

                    # Negatives Vorzeichen für Minimierungsprobleme korrigieren
                    if hasattr(self, 'minimize') and self.minimize:
                        best_score = -best_score

                    # Ergebnisse speichern
                    self.best_params = best_params
                    self.best_score = best_score

                    # Optimierungsverlauf speichern
                    with open(os.path.join(self.output_dir, 'optimization_history.json'), 'w') as f:
                        json.dump(self.optimization_history, f, indent=2, default=str)

                    # Ergebnisse speichern
                    with open(os.path.join(self.output_dir, 'optimization_result.json'), 'w') as f:
                        json.dump({
                            'best_params': best_params,
                            'best_score': best_score,
                            'n_calls': n_calls,
                            'init_points': init_points,
                            'timestamp': datetime.now().isoformat()
                        }, f, indent=2, default=str)

                    # Visualisierungen erstellen
                    self._create_optimization_plots(result, param_names)

                    logger.info(f"Optimierung abgeschlossen. Bester Score: {best_score:.4f}")
                    return {'best_params': best_params, 'best_score': best_score}

                except Exception as e:
                    logger.error(f"Fehler bei der Bayesschen Optimierung: {e}")
                    return {}

            def _create_optimization_plots(self, result, param_names: List[str]) -> None:
                """
                Erstellt Visualisierungen für die Optimierungsergebnisse.

                Args:
                    result: Ergebnis der skopt.gp_minimize Funktion
                    param_names: Liste der Parameternamen
                """
                try:
                    # Konvergenzplot
                    plt.figure(figsize=(10, 6))
                    plot_convergence(result)
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, 'convergence_plot.png'))
                    plt.close()

                    # Zielfunktionsplot für bis zu 2 wichtigste Parameter
                    if len(param_names) >= 2:
                        plt.figure(figsize=(12, 10))
                        _ = plot_objective(result, n_points=40)
                        plt.tight_layout()
                        plt.savefig(os.path.join(self.output_dir, 'objective_plot.png'))
                        plt.close()

                    # Optimierungsverlauf
                    history_df = pd.DataFrame(self.optimization_history)

                    if not history_df.empty and 'score' in history_df.columns:
                        plt.figure(figsize=(10, 6))
                        plt.plot(history_df['score'], 'o-', color='blue')
                        plt.axhline(y=self.best_score, color='red', linestyle='--',
                                    label=f'Bester Score: {self.best_score:.4f}')
                        plt.title('Optimierungsverlauf')
                        plt.xlabel('Iteration')
                        plt.ylabel('Score')
                        plt.grid(True, alpha=0.3)
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(os.path.join(self.output_dir, 'optimization_history.png'))
                        plt.close()

                    # Parameter-Verteilungsplots
                    for i, param_name in enumerate(param_names):
                        if param_name in history_df.columns:
                            plt.figure(figsize=(10, 6))

                            # Parameter-Werte und Scores
                            param_values = history_df[param_name]
                            scores = history_df['score']

                            if len(param_values.unique()) <= 10:
                                # Kategorischer Parameter oder wenige Werte
                                # Boxplot
                                data = []
                                labels = []

                                for value in sorted(param_values.unique()):
                                    mask = param_values == value
                                    if mask.sum() > 0:
                                        data.append(scores[mask])
                                        labels.append(str(value))

                                plt.boxplot(data, labels=labels)
                                plt.title(f'Score-Verteilung nach {param_name}')
                                plt.xlabel(param_name)
                                plt.ylabel('Score')
                                plt.grid(True, alpha=0.3)
                            else:
                                # Kontinuierlicher Parameter
                                plt.scatter(param_values, scores, alpha=0.7)

                                # Trendlinie
                                try:
                                    z = np.polyfit(param_values, scores, 1)
                                    p = np.poly1d(z)
                                    plt.plot(sorted(param_values), p(sorted(param_values)),
                                             'r--', alpha=0.5)
                                except:
                                    pass

                                plt.title(f'Score vs. {param_name}')
                                plt.xlabel(param_name)
                                plt.ylabel('Score')
                                plt.grid(True, alpha=0.3)

                            plt.tight_layout()
                            plt.savefig(os.path.join(self.output_dir, f'param_{param_name}.png'))
                            plt.close()

                except Exception as e:
                    logger.error(f"Fehler bei der Erstellung der Optimierungsplots: {e}")

            def train_surrogate_model(self) -> bool:
                """
                Trainiert ein Surrogat-Modell aus den Optimierungsergebnissen.
                Dies ermöglicht es, die Auswirkungen von Parametern zu verstehen.

                Returns:
                    True, wenn das Training erfolgreich war
                """
                if not self.optimization_history:
                    logger.error("Keine Optimierungshistorie für das Training vorhanden")
                    return False

                try:
                    # Daten aus Optimierungshistorie extrahieren
                    X = []
                    y = []

                    for entry in self.optimization_history:
                        params = entry['params']
                        score = entry['score']

                        # Parameter-Werte in geordneter Liste
                        param_values = [params[key] for key in sorted(params.keys())]
                        X.append(param_values)
                        y.append(score)

                    # In NumPy-Arrays umwandeln
                    X = np.array(X)
                    y = np.array(y)

                    # Random Forest Regressor als Surrogat-Modell
                    surrogate_model = RandomForestRegressor(n_estimators=100, random_state=42)

                    # Modell trainieren
                    surrogate_model.fit(X, y)

                    # Feature-Importance extrahieren
                    importances = surrogate_model.feature_importances_
                    indices = np.argsort(importances)[::-1]

                    # Feature-Namen in gleicher Reihenfolge wie beim Training
                    feature_names = sorted(self.optimization_history[0]['params'].keys())

                    # Feature-Importance speichern
                    feature_importance = {
                        feature_names[i]: importances[i] for i in indices
                    }

                    self.feature_importance = feature_importance
                    self.ml_model = surrogate_model

                    # Visualisierung der Feature-Importance
                    plt.figure(figsize=(10, 6))
                    plt.bar(range(len(feature_names)), importances[indices])
                    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45)
                    plt.xlabel('Parameter')
                    plt.ylabel('Relative Wichtigkeit')
                    plt.title('Parameter-Wichtigkeit für die Performance')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'))
                    plt.close()

                    # Feature-Importance als JSON speichern
                    with open(os.path.join(self.output_dir, 'feature_importance.json'), 'w') as f:
                        json.dump(feature_importance, f, indent=2)

                    logger.info(
                        f"Surrogat-Modell trainiert. Wichtigster Parameter: {list(feature_importance.keys())[0]}")
                    return True

                except Exception as e:
                    logger.error(f"Fehler beim Training des Surrogat-Modells: {e}")
                    return False

            def predict_performance(self, params: Dict[str, Any]) -> float:
                """
                Sagt die Performance für gegebene Parameter vorher.

                Args:
                    params: Dictionary mit Parameterwerten

                Returns:
                    Vorhergesagter Performance-Score
                """
                if self.ml_model is None:
                    logger.error("Kein trainiertes Surrogat-Modell vorhanden")
                    return float('nan')

                try:
                    # Parameter in geordnete Liste umwandeln
                    param_values = [params[key] for key in sorted(params.keys())]

                    # Als 2D-Array für die Vorhersage
                    X = np.array([param_values])

                    # Vorhersage durchführen
                    predicted_score = self.ml_model.predict(X)[0]

                    logger.info(f"Vorhergesagter Score für gegebene Parameter: {predicted_score:.4f}")
                    return predicted_score

                except Exception as e:
                    logger.error(f"Fehler bei der Performance-Vorhersage: {e}")
                    return float('nan')

            def generate_optimization_report(self) -> str:
                """
                Erstellt einen detaillierten Bericht über die Optimierungsergebnisse.

                Returns:
                    Pfad zum Bericht oder leerer String bei Fehler
                """
                if not self.optimization_history or self.best_params is None:
                    logger.error("Keine Optimierungsergebnisse für den Bericht vorhanden")
                    return ""

                try:
                    # Berichtspfad
                    report_path = os.path.join(self.output_dir, "optimization_report.html")

                    # HTML-Bericht erstellen
                    with open(report_path, 'w') as f:
                        f.write("""
                                <!DOCTYPE html>
                                <html>
                                <head>
                                    <title>ML-Backtest Optimierungsbericht</title>
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
                                        .param-box {
                                            border: 1px solid #ddd;
                                            border-radius: 5px;
                                            padding: 10px;
                                            margin-bottom: 10px;
                                        }
                                        .importance-bar {
                                            height: 20px;
                                            background-color: #007bff;
                                            border-radius: 3px;
                                        }
                                        .plot-container {
                                            text-align: center;
                                            margin: 20px 0;
                                        }
                                        .plot-container img {
                                            max-width: 100%;
                                            border: 1px solid #ddd;
                                            border-radius: 5px;
                                        }
                                    </style>
                                </head>
                                <body>
                                    <div class="container">
                                        <h1>ML-Backtest Optimierungsbericht</h1>
                                        <p>Erstelldatum: %s</p>

                                        <div class="summary-box">
                                            <h2>Optimierungszusammenfassung</h2>
                                            <p><strong>Durchgeführte Evaluierungen:</strong> %d</p>
                                            <p><strong>Bester Score:</strong> %.4f</p>
                                        </div>

                                        <h2>Beste Parameter</h2>
                                        <table class="results-table">
                                            <tr>
                                                <th>Parameter</th>
                                                <th>Optimierter Wert</th>
                                                <th>Wichtigkeit</th>
                                            </tr>
                                """ % (
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            len(self.optimization_history),
                            self.best_score
                        ))

                        # Optimierte Parameter auflisten
                        for param_name, param_value in sorted(self.best_params.items()):
                            importance = 0

                            if self.feature_importance and param_name in self.feature_importance:
                                importance = self.feature_importance[param_name]

                            # Formatierte Ausgabe je nach Parametertyp
                            if isinstance(param_value, (int, float)):
                                value_str = f"{param_value:.4f}" if isinstance(param_value, float) else str(param_value)
                            else:
                                value_str = str(param_value)

                            # Wichtigkeitsbalken
                            importance_width = max(1, int(importance * 100))

                            f.write(f"""
                                        <tr>
                                            <td>{param_name}</td>
                                            <td>{value_str}</td>
                                            <td>
                                                <div class="importance-bar" style="width: {importance_width}%"></div>
                                                {importance:.4f}
                                            </td>
                                        </tr>
                                    """)

                        f.write("""
                                        </table>

                                        <h2>Optimierungsverlauf</h2>
                                        <div class="plot-container">
                                            <img src="optimization_history.png" alt="Optimierungsverlauf">
                                        </div>

                                        <h2>Konvergenzanalyse</h2>
                                        <div class="plot-container">
                                            <img src="convergence_plot.png" alt="Konvergenzplot">
                                        </div>

                                        <h2>Parameter-Wichtigkeit</h2>
                                        <div class="plot-container">
                                            <img src="feature_importance.png" alt="Parameter-Wichtigkeit">
                                        </div>

                                        <h2>Top-10 Evaluierungen</h2>
                                        <table class="results-table">
                                            <tr>
                                                <th>Rang</th>
                                                <th>Score</th>
                                                <th>Parameter</th>
                                            </tr>
                                """)

                        # Top-10 Evaluierungen
                        sorted_history = sorted(self.optimization_history,
                                                key=lambda x: x['score'], reverse=True)[:10]

                        for i, entry in enumerate(sorted_history):
                            score = entry['score']
                            params_str = ", ".join([f"{k}: {v}" for k, v in entry['params'].items()])

                            f.write(f"""
                                        <tr>
                                            <td>{i + 1}</td>
                                            <td>{score:.4f}</td>
                                            <td>{params_str}</td>
                                        </tr>
                                    """)

                        # Ende des HTML-Dokuments
                        f.write("""
                                        </table>
                                    </div>
                                </body>
                                </html>
                                """)

                    logger.info(f"Optimierungsbericht erstellt: {report_path}")
                    return report_path

                except Exception as e:
                    logger.error(f"Fehler bei der Erstellung des Optimierungsberichts: {e}")
                    return ""

        class MLBacktestFramework:
            """
            Hauptklasse des ML-Enhanced Backtest Frameworks.
            Koordiniert alle Komponenten und bietet eine einheitliche Schnittstelle.
            """

            def __init__(self, base_config_path: str = None, output_dir: str = None):
                """
                Initialisiert das ML-Backtest Framework.

                Args:
                    base_config_path: Pfad zur Basiskonfiguration (optional)
                    output_dir: Basis-Ausgabeverzeichnis (optional)
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
                self.output_dir = output_dir or f"data/ml_backtest_{timestamp}"
                os.makedirs(self.output_dir, exist_ok=True)
                logger.info(f"Ausgabeverzeichnis: {self.output_dir}")

                # Komponenten initialisieren
                self.regime_detector = MarketRegimeDetector()
                self.cluster_analyzer = AssetClusterAnalyzer()
                self.optimizer = MLBacktestOptimizer(self.base_settings,
                                                     os.path.join(self.output_dir, "optimization"))

                # Experiment-Ergebnisse
                self.results = {}
                self.current_regime = None
                self.clusters = None
                self.optimized_params = {}
                self.strategy_models = {}

            def detect_market_regimes(self, n_regimes: int = 3,
                                      symbols: List[str] = None,
                                      timeframe: str = "1d",
                                      start_date: str = None,
                                      end_date: str = None) -> Dict[int, str]:
                """
                Führt die Marktregime-Erkennung durch.

                Args:
                    n_regimes: Anzahl der zu identifizierenden Regimes
                    symbols: Liste der zu analysierenden Symbole
                    timeframe: Zeitrahmen der Daten
                    start_date: Startdatum im Format 'YYYY-MM-DD'
                    end_date: Enddatum im Format 'YYYY-MM-DD'

                Returns:
                    Dictionary mit Regime-IDs und -Labels
                """
                try:
                    # Marktregime-Detektor konfigurieren
                    self.regime_detector.n_regimes = n_regimes

                    # Daten laden
                    success = self.regime_detector.load_market_data(
                        symbols=symbols,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date
                    )

                    if not success:
                        logger.error("Fehler beim Laden der Marktdaten für Regime-Erkennung")
                        return {}

                    # Features extrahieren
                    features_df = self.regime_detector.extract_market_features()

                    if features_df.empty:
                        logger.error("Keine Features für Regime-Erkennung extrahiert")
                        return {}

                    # Regime-Modell trainieren
                    success = self.regime_detector.train_regime_model(features_df)

                    if not success:
                        logger.error("Fehler beim Training des Regime-Modells")
                        return {}

                    # Aktuelles Regime bestimmen (letzter Zeitpunkt)
                    latest_features = features_df.iloc[-1:].copy()
                    self.current_regime = self.regime_detector.predict_regime(latest_features)

                    # Regime-Bericht erstellen
                    report_path = self.regime_detector.generate_regime_report(
                        os.path.join(self.output_dir, "regime_analysis")
                    )

                    if report_path:
                        logger.info(f"Regime-Bericht erstellt: {report_path}")

                    # Modell speichern
                    model_path = os.path.join(self.output_dir, "regime_analysis", "regime_model.pkl")
                    self.regime_detector.save_model(model_path)

                    # Trading-Regeln extrahieren
                    trading_rules = self.regime_detector.extract_trading_rules()

                    # Ergebnisse speichern
                    self.results['regimes'] = {
                        'n_regimes': n_regimes,
                        'current_regime': self.current_regime,
                        'regime_labels': self.regime_detector.regime_labels,
                        'trading_rules': trading_rules,
                        'report_path': report_path,
                        'model_path': model_path
                    }

                    logger.info(f"Marktregime-Erkennung abgeschlossen. {n_regimes} Regime identifiziert.")
                    return self.regime_detector.regime_labels

                except Exception as e:
                    logger.error(f"Fehler bei der Marktregime-Erkennung: {e}")
                    return {}

            def analyze_asset_clusters(self, n_clusters: int = None,
                                       symbols: List[str] = None,
                                       timeframe: str = "1d",
                                       start_date: str = None,
                                       end_date: str = None) -> pd.DataFrame:
                """
                Führt die Asset-Cluster-Analyse durch.

                Args:
                    n_clusters: Anzahl der Cluster (oder None für automatische Bestimmung)
                    symbols: Liste der zu analysierenden Symbole
                    timeframe: Zeitrahmen der Daten
                    start_date: Startdatum im Format 'YYYY-MM-DD'
                    end_date: Enddatum im Format 'YYYY-MM-DD'

                Returns:
                    DataFrame mit Cluster-Zuordnungen
                """
                try:
                    # Daten laden
                    success = self.cluster_analyzer.load_market_data(
                        symbols=symbols,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date
                    )

                    if not success:
                        logger.error("Fehler beim Laden der Marktdaten für Cluster-Analyse")
                        return pd.DataFrame()

                    # Korrelationsmatrix berechnen
                    corr_matrix = self.cluster_analyzer.calculate_correlation_matrix()

                    if corr_matrix.empty:
                        logger.error("Fehler bei der Berechnung der Korrelationsmatrix")
                        return pd.DataFrame()

                    # Asset-Features extrahieren
                    features_df = self.cluster_analyzer.extract_asset_features()

                    if features_df.empty:
                        logger.error("Keine Features für Cluster-Analyse extrahiert")
                        return pd.DataFrame()

                    # Clustering durchführen
                    cluster_df = self.cluster_analyzer.run_clustering(
                        n_clusters=n_clusters,
                        method='kmeans',
                        correlation_based=True
                    )

                    if cluster_df.empty:
                        logger.error("Fehler beim Clustering")
                        return pd.DataFrame()

                    # Cluster-Bericht erstellen
                    report_path = self.cluster_analyzer.generate_cluster_report(
                        os.path.join(self.output_dir, "cluster_analysis")
                    )

                    if report_path:
                        logger.info(f"Cluster-Bericht erstellt: {report_path}")

                    # Portfolio-Empfehlung
                    portfolio = self.cluster_analyzer.recommend_portfolio()

                    # Ergebnisse speichern
                    self.clusters = cluster_df
                    self.results['clusters'] = {
                        'n_clusters': len(cluster_df['cluster'].unique()),
                        'cluster_assignments': cluster_df,
                        'cluster_performances': self.cluster_analyzer.cluster_performances,
                        'portfolio_recommendation': portfolio,
                        'report_path': report_path
                    }

                    logger.info(f"Asset-Cluster-Analyse abgeschlossen. " +
                                f"{len(cluster_df['cluster'].unique())} Cluster identifiziert.")
                    return cluster_df

                except Exception as e:
                    logger.error(f"Fehler bei der Asset-Cluster-Analyse: {e}")
                    return pd.DataFrame()

            def optimize_strategy_parameters(self, strategy_name: str,
                                             param_space: Dict[str, Any],
                                             regime_specific: bool = True,
                                             n_calls: int = 50,
                                             init_points: int = 10) -> Dict[str, Any]:
                """
                Optimiert die Parameter einer Trading-Strategie.

                Args:
                    strategy_name: Name der Strategie
                    param_space: Dictionary mit Parameternamen und Wertebereichen
                    regime_specific: Ob für jedes Marktregime separat optimiert werden soll
                    n_calls: Anzahl der Evaluierungsaufrufe
                    init_points: Anzahl an zufälligen Initialpunkten

                Returns:
                    Dictionary mit optimierten Parametern
                """
                try:
                    # Parameterraum definieren
                    self.optimizer.define_parameter_space(param_space)

                    if regime_specific and 'regimes' in self.results:
                        # Für jedes Regime separat optimieren
                        regime_params = {}

                        for regime_id, regime_label in self.results['regimes']['regime_labels'].items():
                            logger.info(f"Optimiere für Regime {regime_id}: {regime_label}")

                            # Evaluierungsfunktion für dieses Regime definieren
                            def evaluate_for_regime(params):
                                return self._evaluate_strategy(
                                    strategy_name=strategy_name,
                                    params=params,
                                    regime=regime_id
                                )

                            # Evaluierungsfunktion setzen
                            self.optimizer.set_evaluation_function(evaluate_for_regime)

                            # Ausgabeverzeichnis für dieses Regime
                            regime_output_dir = os.path.join(
                                self.output_dir,
                                "optimization",
                                f"regime_{regime_id}"
                            )
                            self.optimizer.output_dir = regime_output_dir

                            # Optimierung durchführen
                            result = self.optimizer.run_bayesian_optimization(
                                n_calls=n_calls,
                                init_points=init_points
                            )

                            if result and 'best_params' in result:
                                # Surrogat-Modell trainieren
                                self.optimizer.train_surrogate_model()

                                # Optimierungsbericht erstellen
                                report_path = self.optimizer.generate_optimization_report()

                                # Ergebnisse speichern
                                regime_params[regime_id] = {
                                    'params': result['best_params'],
                                    'score': result['best_score'],
                                    'report_path': report_path
                                }

                        # Ergebnisse speichern
                        self.optimized_params = {
                            'strategy_name': strategy_name,
                            'regime_specific': True,
                            'params_by_regime': regime_params
                        }

                        # Zusammenfassung speichern
                        with open(os.path.join(self.output_dir, "optimized_parameters.json"), 'w') as f:
                            json.dump(self.optimized_params, f, indent=2, default=str)

                        logger.info(f"Regime-spezifische Optimierung abgeschlossen für {len(regime_params)} Regime")
                        return self.optimized_params

                    else:
                        # Allgemeine Optimierung
                        logger.info(f"Optimiere Strategie-Parameter für {strategy_name}")

                        # Evaluierungsfunktion definieren
                        def evaluate_general(params):
                            return self._evaluate_strategy(
                                strategy_name=strategy_name,
                                params=params
                            )

                        # Evaluierungsfunktion setzen
                        self.optimizer.set_evaluation_function(evaluate_general)

                        # Optimierung durchführen
                        result = self.optimizer.run_bayesian_optimization(
                            n_calls=n_calls,
                            init_points=init_points
                        )

                        if result and 'best_params' in result:
                            # Surrogat-Modell trainieren
                            self.optimizer.train_surrogate_model()

                            # Optimierungsbericht erstellen
                            report_path = self.optimizer.generate_optimization_report()

                            # Ergebnisse speichern
                            self.optimized_params = {
                                'strategy_name': strategy_name,
                                'regime_specific': False,
                                'params': result['best_params'],
                                'score': result['best_score'],
                                'report_path': report_path
                            }

                            # Zusammenfassung speichern
                            with open(os.path.join(self.output_dir, "optimized_parameters.json"), 'w') as f:
                                json.dump(self.optimized_params, f, indent=2, default=str)

                            logger.info(
                                f"Allgemeine Parameter-Optimierung abgeschlossen. Score: {result['best_score']:.4f}")
                            return self.optimized_params

                        logger.error("Optimierung fehlgeschlagen")
                        return {}

                except Exception as e:
                    logger.error(f"Fehler bei der Parameter-Optimierung: {e}")
                    return {}

            def _evaluate_strategy(self, strategy_name: str, params: Dict[str, Any],
                                   regime: int = None) -> float:
                """
                Evaluiert eine Trading-Strategie mit gegebenen Parametern.

                Args:
                    strategy_name: Name der Strategie
                    params: Dictionary mit Parameterwerten
                    regime: Regime-ID für regimespezifische Evaluation (optional)

                Returns:
                    Performance-Score
                """
                try:
                    # Neue Einstellungen mit Basiseinstellungen und angepassten Parametern
                    settings = Settings()

                    # Basiseinstellungen kopieren
                    for key, value in self.base_settings.get_all().items():
                        settings.set(key, value)

                    # Parameter-Werte setzen
                    for key, value in params.items():
                        settings.set(key, value)

                    # Trading Bot initialisieren
                    bot = TradingBot(mode="backtest", strategy_name=strategy_name, settings=settings)

                    # Wenn regimespezifisch, nur Daten für dieses Regime verwenden
                    if regime is not None and 'regimes' in self.results:
                        # Trading-Regeln für dieses Regime abrufen
                        trading_rules = self.results['regimes'].get('trading_rules', {})

                        if regime in trading_rules:
                            # Beste Assets für dieses Regime
                            rule = trading_rules[regime]
                            recommended_assets = list(rule.get('top_performers', {}).keys())

                            if recommended_assets:
                                # Trading-Paare auf empfohlene Assets beschränken
                                settings.set('trading_pairs', recommended_assets)

                    # Backtest durchführen
                    results = bot.run_backtest()

                    # Performance-Metriken extrahieren
                    if not results:
                        logger.error("Backtest fehlgeschlagen")
                        return float('-inf')

                    # Sharpe Ratio als primäre Metrik verwenden
                    stats = results.get('statistics', {})
                    sharpe_ratio = stats.get('sharpe_ratio', 0)

                    # Total Return als sekundäre Metrik
                    total_return = results.get('total_return', 0)

                    # Win Rate als tertiäre Metrik
                    win_rate = stats.get('win_rate', 0)

                    # Max Drawdown als Strafterm
                    max_drawdown = abs(stats.get('max_drawdown', 0))

                    # Kombinierte Metrik
                    # - Sharpe ist wichtigster Faktor
                    # - Return ist zweitwichtigster Faktor
                    # - Win Rate unterstützt die Robustheit
                    # - Drawdown reduziert den Score für risikoreiche Strategien
                    combined_score = (
                            sharpe_ratio * 1.0 +
                            total_return / 100.0 * 0.5 +
                            win_rate / 100.0 * 0.3 -
                            max_drawdown / 100.0 * 0.2
                    )

                    logger.info(f"Evaluation: Sharpe={sharpe_ratio:.2f}, Return={total_return:.2f}%, " +
                                f"Win Rate={win_rate:.1f}%, Max DD={max_drawdown:.2f}%, Score={combined_score:.4f}")

                    return combined_score

                except Exception as e:
                    logger.error(f"Fehler bei der Strategie-Evaluation: {e}")
                    return float('-inf')

            def train_strategy_selection_model(self) -> bool:
                """
                Trainiert ein ML-Modell zur Auswahl der besten Strategie.

                Returns:
                    True, wenn das Training erfolgreich war
                """
                if not 'regimes' in self.results:
                    logger.error("Keine Regime-Ergebnisse für das Training vorhanden")
                    return False

                try:
                    # Verfügbare Strategien
                    strategies = ["momentum", "mean_reversion", "ml_strategy"]

                    # Trainingsdaten sammeln
                    X = []  # Features (Regime-Eigenschaften)
                    y = []  # Labels (beste Strategie)

                    # Für jedes Regime
                    for regime_id, regime_label in self.results['regimes']['regime_labels'].items():
                        # Performance der Strategien in diesem Regime
                        strategy_scores = {}

                        for strategy in strategies:
                            # Strategie evaluieren
                            score = self._evaluate_strategy(
                                strategy_name=strategy,
                                params={},  # Standardparameter
                                regime=regime_id
                            )

                            strategy_scores[strategy] = score

                        # Beste Strategie finden
                        best_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]

                        # Regime-Features extrahieren
                        regime_features = self._extract_regime_features(regime_id)

                        if regime_features:
                            X.append(regime_features)
                            y.append(best_strategy)

                    if not X or not y:
                        logger.error("Keine Trainingsdaten gesammelt")
                        return False

                    # In NumPy-Arrays umwandeln
                    X = np.array(X)
                    y = np.array(y)

                    # Random Forest Classifier trainieren
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X, y)

                    # Modell speichern
                    model_path = os.path.join(self.output_dir, "strategy_selection_model.pkl")
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)

                    # Modell speichern
                    self.strategy_models['selection_model'] = model

                    logger.info(f"Strategie-Auswahl-Modell trainiert und gespeichert unter {model_path}")
                    return True

                except Exception as e:
                    logger.error(f"Fehler beim Training des Strategie-Auswahl-Modells: {e}")
                    return False

            def _extract_regime_features(self, regime_id: int) -> List[float]:
                """
                Extrahiert Merkmale eines Marktregimes für das ML-Training.

                Args:
                    regime_id: ID des Regimes

                Returns:
                    Liste mit Feature-Werten
                """
                try:
                    # Trading-Regeln für dieses Regime abrufen
                    trading_rules = self.results['regimes'].get('trading_rules', {})

                    if regime_id not in trading_rules:
                        logger.error(f"Keine Trading-Regeln für Regime {regime_id}")
                        return []

                    rule = trading_rules[regime_id]

                    # Feature-Extraktion
                    features = [
                        rule.get('avg_performance', 0),  # Durchschnittliche Performance
                        rule.get('pct_positive', 0) / 100.0,  # Anteil positiver Performer
                        len(rule.get('top_performers', {})) / 10.0  # Anzahl der Top-Performer (normalisiert)
                    ]

                    # Allokations-Features
                    allocation = rule.get('portfolio_allocation', {})

                    for asset_type in ['altcoins', 'bitcoin', 'stablecoins']:
                        features.append(allocation.get(asset_type, 0))

                    return features

                except Exception as e:
                    logger.error(f"Fehler bei der Extraktion von Regime-Features: {e}")
                    return []

            def recommend_current_strategy(self) -> Dict[str, Any]:
                """
                Empfiehlt die beste Strategie und Parameter für das aktuelle Marktregime.

                Returns:
                    Dictionary mit Strategieempfehlung
                """
                if self.current_regime is None:
                    logger.error("Kein aktuelles Marktregime erkannt")
                    return {}

                try:
                    # Regime-Label
                    regime_label = self.regime_detector.regime_labels.get(
                        self.current_regime, f"Regime {self.current_regime}"
                    )

                    # Trading-Regeln für dieses Regime
                    trading_rules = self.results['regimes'].get('trading_rules', {})
                    rule = trading_rules.get(self.current_regime, {})

                    # Strategie-Empfehlung
                    if 'selection_model' in self.strategy_models:
                        # ML-basierte Strategieauswahl
                        features = self._extract_regime_features(self.current_regime)

                        if features:
                            # Strategie vorhersagen
                            features_array = np.array([features])
                            strategy = self.strategy_models['selection_model'].predict(features_array)[0]
                        else:
                            # Fallback: Strategie direkt aus der Trading-Regel
                            strategy = "momentum"  # Standard
                    else:
                        # Fallback: Strategie direkt aus der Trading-Regel
                        strategy = "momentum"  # Standard

                    # Optimierte Parameter für dieses Regime
                    params = {}

                    if self.optimized_params:
                        if self.optimized_params.get('regime_specific', False):
                            # Regime-spezifische Parameter
                            regime_params = self.optimized_params.get('params_by_regime', {})

                            if self.current_regime in regime_params:
                                params = regime_params[self.current_regime].get('params', {})
                        else:
                            # Allgemeine Parameter
                            params = self.optimized_params.get('params', {})

                    # Portfolio-Empfehlung
                    portfolio_allocation = rule.get('portfolio_allocation', {})
                    top_performers = rule.get('top_performers', {})

                    # Empfohlene Assets aus Top-Performern
                    recommended_assets = list(top_performers.keys())

                    # Empfehlung zusammenstellen
                    recommendation = {
                        'current_regime': self.current_regime,
                        'regime_label': regime_label,
                        'recommended_strategy': strategy,
                        'recommended_params': params,
                        'portfolio_allocation': portfolio_allocation,
                        'recommended_assets': recommended_assets
                        # Handlungsanweisung
                        trading_advice = rule.get('recommended_strategy',
                                                  "Keine spezifische Strategie empfohlen")

                    recommendation['trading_advice'] = trading_advice

                    # Aktuelle Zeit
                    recommendation['timestamp'] = datetime.now().isoformat()

                    # In Datei speichern
                    with open(os.path.join(self.output_dir, "current_recommendation.json"), 'w') as f:
                        json.dump(recommendation, f, indent=2, default=str)

                    logger.info(f"Aktuelle Empfehlung: Regime {self.current_regime} ({regime_label}), " +
                                f"Strategie: {strategy}")

                    return recommendation

                    except Exception as e:
                    logger.error(f"Fehler bei der Strategie-Empfehlung: {e}")
                    return {}

            def generate_comprehensive_report(self) -> str:
                """
                Erstellt einen umfassenden Bericht über alle Aspekte der Analyse.

                Returns:
                    Pfad zum Bericht oder leerer String bei Fehler
                """
                try:
                    # Berichtspfad
                    report_path = os.path.join(self.output_dir, "comprehensive_report.html")

                    # Aktuelle Empfehlung
                    recommendation = self.recommend_current_strategy()

                    # HTML-Bericht erstellen
                    with open(report_path, 'w') as f:
                        f.write("""
                                    <!DOCTYPE html>
                                    <html>
                                    <head>
                                        <title>Umfassender ML-Backtest Analysebericht</title>
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
                                            .tab-container {
                                                margin: 20px 0;
                                            }
                                            .tab-links {
                                                display: flex;
                                                border-bottom: 1px solid #ddd;
                                            }
                                            .tab-link {
                                                padding: 10px 15px;
                                                cursor: pointer;
                                                background-color: #f1f1f1;
                                                border: 1px solid #ddd;
                                                border-bottom: none;
                                                border-radius: 5px 5px 0 0;
                                                margin-right: 5px;
                                            }
                                            .tab-link.active {
                                                background-color: white;
                                                border-bottom: 1px solid white;
                                                margin-bottom: -1px;
                                            }
                                            .tab-content {
                                                display: none;
                                                padding: 15px;
                                                border: 1px solid #ddd;
                                                border-top: none;
                                            }
                                            .tab-content.active {
                                                display: block;
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
                                        </style>
                                        <script>
                                            function openTab(evt, tabName) {
                                                var i, tabcontent, tablinks;
                                                tabcontent = document.getElementsByClassName("tab-content");
                                                for (i = 0; i < tabcontent.length; i++) {
                                                    tabcontent[i].style.display = "none";
                                                }
                                                tablinks = document.getElementsByClassName("tab-link");
                                                for (i = 0; i < tablinks.length; i++) {
                                                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                                                }
                                                document.getElementById(tabName).style.display = "block";
                                                evt.currentTarget.className += " active";
                                            }
                                        </script>
                                    </head>
                                    <body>
                                        <div class="container">
                                            <h1>Umfassender ML-Backtest Analysebericht</h1>
                                            <p>Erstelldatum: %s</p>
                                    """ % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

                        # Aktuelle Empfehlung
                        if recommendation:
                            regime_label = recommendation.get('regime_label', 'Unbekannt')
                            strategy = recommendation.get('recommended_strategy', 'Keine Empfehlung')

                            f.write(f"""
                                            <div class="info-box">
                                                <h2>Aktuelle Markteinschätzung</h2>
                                                <p><strong>Erkanntes Marktregime:</strong> {regime_label}</p>
                                                <p><strong>Empfohlene Strategie:</strong> {strategy}</p>
                                                <p><strong>Trading-Empfehlung:</strong> {recommendation.get('trading_advice', 'Keine spezifische Empfehlung')}</p>
                                            </div>
                                        """)

                            # Portfolio-Allokation
                            if 'portfolio_allocation' in recommendation:
                                f.write("""
                                                <div class="summary-box">
                                                    <h3>Empfohlene Portfolio-Allokation</h3>
                                                    <div class="allocation-chart">
                                            """)

                                # Allokations-Chart
                                colors = {
                                    "altcoins": "#6f42c1",  # Lila
                                    "bitcoin": "#f8b739",  # Bitcoin-Orange
                                    "stablecoins": "#20c997"  # Grün
                                }

                                alloc = recommendation['portfolio_allocation']
                                for asset, percentage in alloc.items():
                                    width = percentage * 100
                                    f.write(f"""
                                                    <div class="allocation-segment" 
                                                         style="width: {width}%; background-color: {colors.get(asset, '#007bff')};">
                                                        {asset.capitalize()} {percentage * 100:.0f}%
                                                    </div>
                                                """)

                                f.write("""
                                                    </div>
                                                </div>
                                            """)

                            # Empfohlene Assets
                            if 'recommended_assets' in recommendation and recommendation['recommended_assets']:
                                f.write("""
                                                <div class="summary-box">
                                                    <h3>Empfohlene Assets</h3>
                                                    <div>
                                            """)

                                for asset in recommendation['recommended_assets']:
                                    f.write(f'<span class="asset-badge">{asset}</span> ')

                                f.write("""
                                                    </div>
                                                </div>
                                            """)

                        # Tabs für die verschiedenen Analysen
                        f.write("""
                                        <div class="tab-container">
                                            <div class="tab-links">
                                                <button class="tab-link active" onclick="openTab(event, 'regime-tab')">Marktregime-Analyse</button>
                                                <button class="tab-link" onclick="openTab(event, 'cluster-tab')">Asset-Cluster-Analyse</button>
                                                <button class="tab-link" onclick="openTab(event, 'param-tab')">Parameter-Optimierung</button>
                                            </div>
                                    """)

                        # Tab: Marktregime-Analyse
                        f.write("""
                                        <div id="regime-tab" class="tab-content active">
                                            <h2>Marktregime-Analyse</h2>
                                    """)

                        if 'regimes' in self.results:
                            regimes = self.results['regimes']
                            n_regimes = regimes.get('n_regimes', 0)
                            regime_labels = regimes.get('regime_labels', {})
                            trading_rules = regimes.get('trading_rules', {})

                            f.write(f"""
                                            <p>Es wurden {n_regimes} verschiedene Marktregimes identifiziert:</p>
                                            <table class="results-table">
                                                <tr>
                                                    <th>Regime</th>
                                                    <th>Beschreibung</th>
                                                    <th>Empfohlene Strategie</th>
                                                    <th>Performance</th>
                                                </tr>
                                        """)

                            for regime_id, label in regime_labels.items():
                                # Trading-Regel für dieses Regime
                                rule = trading_rules.get(regime_id, {})
                                strategy = rule.get('recommended_strategy', 'Keine spezifische Empfehlung')
                                avg_perf = rule.get('avg_performance', 0) * 100
                                perf_class = "positive" if avg_perf > 0 else "negative"

                                f.write(f"""
                                                <tr>
                                                    <td>{regime_id}</td>
                                                    <td>{label}</td>
                                                    <td>{strategy}</td>
                                                    <td class="{perf_class}">{avg_perf:.2f}%</td>
                                                </tr>
                                            """)

                            f.write("""
                                            </table>
                                        """)

                            # Link zum detaillierten Bericht
                            if 'report_path' in regimes:
                                report_name = os.path.basename(regimes['report_path'])
                                relative_path = os.path.join("regime_analysis", report_name)
                                f.write(f"""
                                                <p><a href="{relative_path}" target="_blank">Detaillierter Regime-Analysebericht</a></p>
                                            """)

                        else:
                            f.write("""
                                            <p>Keine Marktregime-Analyse durchgeführt.</p>
                                        """)

                        f.write("""
                                        </div>
                                    """)

                        # Tab: Asset-Cluster-Analyse
                        f.write("""
                                        <div id="cluster-tab" class="tab-content">
                                            <h2>Asset-Cluster-Analyse</h2>
                                    """)

                        if 'clusters' in self.results:
                            clusters = self.results['clusters']
                            n_clusters = clusters.get('n_clusters', 0)
                            cluster_performances = clusters.get('cluster_performances', {})
                            portfolio = clusters.get('portfolio_recommendation', {})

                            f.write(f"""
                                            <p>Es wurden {n_clusters} verschiedene Asset-Cluster identifiziert:</p>
                                            <table class="results-table">
                                                <tr>
                                                    <th>Cluster</th>
                                                    <th>Anzahl Assets</th>
                                                    <th>Performance</th>
                                                    <th>Sharpe Ratio</th>
                                                    <th>Win Rate</th>
                                                    <th>Repräsentatives Asset</th>
                                                </tr>
                                        """)

                            for cluster_id, stats in cluster_performances.items():
                                count = stats.get('count', 0)
                                perf = stats.get('mean_return', 0) * 100
                                perf_class = "positive" if perf > 0 else "negative"
                                sharpe = stats.get('sharpe_ratio', 0)
                                win_rate = stats.get('win_rate', 0) * 100
                                rep_asset = stats.get('representative_asset', 'Unbekannt')

                                f.write(f"""
                                                <tr>
                                                    <td>{cluster_id}</td>
                                                    <td>{count}</td>
                                                    <td class="{perf_class}">{perf:.2f}%</td>
                                                    <td>{sharpe:.2f}</td>
                                                    <td>{win_rate:.1f}%</td>
                                                    <td>{rep_asset}</td>
                                                </tr>
                                            """)

                            f.write("""
                                            </table>
                                        """)

                            # Portfolio-Empfehlung
                            if portfolio:
                                f.write(f"""
                                                <h3>Portfolio-Empfehlung aus Cluster-Analyse</h3>
                                                <p><strong>Strategie:</strong> {portfolio.get('strategy', 'Keine spezifische Empfehlung')}</p>
                                                <p><strong>Empfohlene Assets:</strong></p>
                                                <div>
                                            """)

                                for asset in portfolio.get('assets', []):
                                    alloc = portfolio.get('allocation', {}).get(asset, 0) * 100
                                    f.write(f'<span class="asset-badge">{asset} ({alloc:.1f}%)</span> ')

                                f.write("""
                                                </div>
                                            """)

                            # Link zum detaillierten Bericht
                            if 'report_path' in clusters:
                                report_name = os.path.basename(clusters['report_path'])
                                relative_path = os.path.join("cluster_analysis", report_name)
                                f.write(f"""
                                                <p><a href="{relative_path}" target="_blank">Detaillierter Cluster-Analysebericht</a></p>
                                            """)

                        else:
                            f.write("""
                                            <p>Keine Asset-Cluster-Analyse durchgeführt.</p>
                                        """)

                        f.write("""
                                        </div>
                                    """)

                        # Tab: Parameter-Optimierung
                        f.write("""
                                        <div id="param-tab" class="tab-content">
                                            <h2>Parameter-Optimierung</h2>
                                    """)

                        if self.optimized_params:
                            strategy_name = self.optimized_params.get('strategy_name', 'Unbekannt')
                            regime_specific = self.optimized_params.get('regime_specific', False)

                            f.write(f"""
                                            <p><strong>Optimierte Strategie:</strong> {strategy_name}</p>
                                            <p><strong>Regime-spezifische Optimierung:</strong> {'Ja' if regime_specific else 'Nein'}</p>
                                        """)

                            if regime_specific:
                                # Regime-spezifische Parameter
                                regime_params = self.optimized_params.get('params_by_regime', {})

                                f.write("""
                                                <h3>Optimierte Parameter nach Regime</h3>
                                                <table class="results-table">
                                                    <tr>
                                                        <th>Regime</th>
                                                        <th>Score</th>
                                                        <th>Parameter</th>
                                                    </tr>
                                            """)

                                for regime_id, data in regime_params.items():
                                    params = data.get('params', {})
                                    score = data.get('score', 0)

                                    # Parameter formatieren
                                    params_str = ""
                                    for key, value in params.items():
                                        params_str += f"{key}: {value}<br>"

                                    # Regime-Label
                                    if 'regimes' in self.results:
                                        regime_label = self.results['regimes'].get('regime_labels', {}).get(regime_id,
                                                                                                            f"Regime {regime_id}")
                                    else:
                                        regime_label = f"Regime {regime_id}"

                                    f.write(f"""
                                                    <tr>
                                                        <td>{regime_label}</td>
                                                        <td>{score:.4f}</td>
                                                        <td>{params_str}</td>
                                                    </tr>
                                                """)

                                f.write("""
                                                </table>
                                            """)
                            else:
                                # Allgemeine Parameter
                                params = self.optimized_params.get('params', {})
                                score = self.optimized_params.get('score', 0)

                                f.write(f"""
                                                <h3>Optimierte Parameter</h3>
                                                <p><strong>Score:</strong> {score:.4f}</p>
                                                <table class="results-table">
                                                    <tr>
                                                        <th>Parameter</th>
                                                        <th>Wert</th>
                                                    </tr>
                                            """)

                                for key, value in params.items():
                                    f.write(f"""
                                                    <tr>
                                                        <td>{key}</td>
                                                        <td>{value}</td>
                                                    </tr>
                                                """)

                                f.write("""
                                                </table>
                                            """)

                            # Link zum detaillierten Bericht
                            if 'report_path' in self.optimized_params:
                                report_path = self.optimized_params['report_path']
                                if os.path.exists(report_path):
                                    report_name = os.path.basename(report_path)
                                    relative_path = os.path.join("optimization", report_name)
                                    f.write(f"""
                                                    <p><a href="{relative_path}" target="_blank">Detaillierter Optimierungsbericht</a></p>
                                                """)

                        else:
                            f.write("""
                                            <p>Keine Parameter-Optimierung durchgeführt.</p>
                                        """)

                        f.write("""
                                        </div>
                                    """)

                        # Ende der Tab-Container
                        f.write("""
                                        </div>
                                    """)

                        # Schlussfolgerung und Handlungsempfehlungen
                        f.write("""
                                        <h2>Schlussfolgerung und Handlungsempfehlungen</h2>
                                    """)

                        if recommendation:
                            trading_advice = recommendation.get('trading_advice', 'Keine spezifische Empfehlung')

                            f.write(f"""
                                            <div class="success-box">
                                                <p><strong>Aktuelle Empfehlung:</strong> {trading_advice}</p>
                                        """)

                            # Empfohlene Parameter
                            if 'recommended_params' in recommendation and recommendation['recommended_params']:
                                f.write("""
                                                <p><strong>Empfohlene Parameter:</strong></p>
                                                <ul>
                                            """)

                                for key, value in recommendation['recommended_params'].items():
                                    f.write(f"""
                                                    <li>{key}: {value}</li>
                                                """)

                                f.write("""
                                                </ul>
                                            """)

                            f.write("""
                                            </div>
                                        """)

                        else:
                            f.write("""
                                            <div class="alert-box">
                                                <p>Keine konkrete Handlungsempfehlung verfügbar. Führen Sie eine vollständige Analyse durch, um Empfehlungen zu erhalten.</p>
                                            </div>
                                        """)

                        # Ende des HTML-Dokuments
                        f.write("""
                                        </div>
                                    </body>
                                    </html>
                                    """)

                    logger.info(f"Umfassender Bericht erstellt: {report_path}")
                    return report_path

                except Exception as e:
                    logger.error(f"Fehler bei der Erstellung des umfassenden Berichts: {e}")
                    return ""

            def run_complete_analysis(self, symbols: List[str] = None,
                                      timeframe: str = "1d",
                                      start_date: str = None,
                                      end_date: str = None,
                                      strategy_name: str = "momentum",
                                      param_space: Dict[str, Any] = None) -> Dict[str, Any]:
                """
                Führt eine vollständige ML-gestützte Analyse durch.

                Args:
                    symbols: Liste der zu analysierenden Symbole
                    timeframe: Zeitrahmen der Daten
                    start_date: Startdatum im Format 'YYYY-MM-DD'
                    end_date: Enddatum im Format 'YYYY-MM-DD'
                    strategy_name: Name der zu optimierenden Strategie
                    param_space: Parameterraum für die Optimierung

                Returns:
                    Dictionary mit der Gesamtempfehlung
                """
                try:
                    logger.info("Starte vollständige ML-gestützte Analyse...")

                    # 1. Marktregime-Erkennung
                    logger.info("Schritt 1: Marktregime-Erkennung...")
                    self.detect_market_regimes(
                        n_regimes=3,
                        symbols=symbols,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date
                    )

                    # 2. Asset-Cluster-Analyse
                    logger.info("Schritt 2: Asset-Cluster-Analyse...")
                    self.analyze_asset_clusters(
                        n_clusters=None,  # Automatisch bestimmen
                        symbols=symbols,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date
                    )

                    # 3. Parameter-Optimierung
                    logger.info("Schritt 3: Parameter-Optimierung...")
                    default_param_space = {
                        'technical.rsi.period': [9, 14, 21],
                        'technical.rsi.oversold': [30, 35, 40],
                        'technical.rsi.overbought': [60, 65, 70],
                        'technical.macd.fast': [8, 12],
                        'technical.macd.slow': [21, 26],
                        'risk.stop_loss': [0.03, 0.05, 0.07],
                        'risk.take_profit': [0.06, 0.10, 0.15]
                    }

                    # Parameter-Space überschreiben, falls angegeben
                    if param_space:
                        self.optimize_strategy_parameters(
                            strategy_name=strategy_name,
                            param_space=param_space,
                            regime_specific=True
                        )
                    else:
                        self.optimize_strategy_parameters(
                            strategy_name=strategy_name,
                            param_space=default_param_space,
                            regime_specific=True
                        )

                    # 4. Strategie-Auswahl-Modell trainieren
                    logger.info("Schritt 4: Training des Strategie-Auswahl-Modells...")
                    self.train_strategy_selection_model()

                    # 5. Aktuelle Empfehlung generieren
                    logger.info("Schritt 5: Aktuelle Empfehlung generieren...")
                    recommendation = self.recommend_current_strategy()

                    # 6. Umfassenden Bericht erstellen
                    logger.info("Schritt 6: Umfassenden Bericht erstellen...")
                    report_path = self.generate_comprehensive_report()

                    if report_path:
                        logger.info(f"Vollständiger Analysebericht erstellt: {report_path}")
                        recommendation['report_path'] = report_path

                    logger.info("Vollständige ML-gestützte Analyse abgeschlossen.")
                    return recommendation

                except Exception as e:
                    logger.error(f"Fehler bei der vollständigen Analyse: {e}")
                    return {}

            def main():
                """Hauptfunktion für Kommandozeilenaufruf."""
                parser = argparse.ArgumentParser(description="ML-Enhanced Backtest Framework")

                # Unterbefehl-Parser
                subparsers = parser.add_subparsers(dest="command", help="Auszuführender Befehl")

                # Vollständige Analyse
                full_parser = subparsers.add_parser("full-analysis", help="Vollständige Analyse durchführen")
                full_parser.add_argument("--config", "-c", type=str, help="Pfad zur Basiskonfiguration")
                full_parser.add_argument("--output", "-o", type=str, help="Ausgabeverzeichnis")
                full_parser.add_argument("--symbols", "-s", nargs="+",
                                         help="Zu analysierende Symbole (z.B. BTC/USDT ETH/USDT)")
                full_parser.add_argument("--timeframe", "-t", type=str, default="1d",
                                         help="Zeitrahmen der Daten (z.B. 1h, 4h, 1d)")
                full_parser.add_argument("--start", type=str, default="2023-01-01",
                                         help="Startdatum (YYYY-MM-DD)")
                full_parser.add_argument("--end", type=str, default="2023-12-31",
                                         help="Enddatum (YYYY-MM-DD)")
                full_parser.add_argument("--strategy", type=str, default="momentum",
                                         help="Zu optimierende Strategie")

                # Nur Regime-Erkennung
                regime_parser = subparsers.add_parser("regimes", help="Marktregime-Erkennung")
                regime_parser.add_argument("--config", "-c", type=str, help="Pfad zur Basiskonfiguration")
                regime_parser.add_argument("--output", "-o", type=str, help="Ausgabeverzeichnis")
                regime_parser.add_argument("--symbols", "-s", nargs="+",
                                           help="Zu analysierende Symbole")
                regime_parser.add_argument("--n-regimes", "-n", type=int, default=3,
                                           help="Anzahl der zu identifizierenden Regimes")

                # Nur Cluster-Analyse
                cluster_parser = subparsers.add_parser("clusters", help="Asset-Cluster-Analyse")
                cluster_parser.add_argument("--config", "-c", type=str, help="Pfad zur Basiskonfiguration")
                cluster_parser.add_argument("--output", "-o", type=str, help="Ausgabeverzeichnis")
                cluster_parser.add_argument("--symbols", "-s", nargs="+",
                                            help="Zu analysierende Symbole")

                # Nur Parameter-Optimierung
                optim_parser = subparsers.add_parser("optimize", help="Parameter-Optimierung")
                optim_parser.add_argument("--config", "-c", type=str, help="Pfad zur Basiskonfiguration")
                optim_parser.add_argument("--output", "-o", type=str, help="Ausgabeverzeichnis")
                optim_parser.add_argument("--strategy", "-s", type=str, default="momentum",
                                          help="Zu optimierende Strategie")
                optim_parser.add_argument("--regime-specific", "-r", action="store_true",
                                          help="Regime-spezifische Optimierung durchführen")

                # Argumente parsen
                args = parser.parse_args()

                # Framework initialisieren
                framework = MLBacktestFramework(
                    base_config_path=args.config if hasattr(args, "config") else None,
                    output_dir=args.output if hasattr(args, "output") else None
                )

                # Entsprechenden Befehl ausführen
                if args.command == "full-analysis":
                    # Vollständige Analyse
                    result = framework.run_complete_analysis(
                        symbols=args.symbols,
                        timeframe=args.timeframe,
                        start_date=args.start,
                        end_date=args.end,
                        strategy_name=args.strategy
                    )

                    if result:
                        print(
                            f"Analyse abgeschlossen. Empfehlung: {result.get('trading_advice', 'Keine spezifische Empfehlung')}")
                        print(f"Detaillierter Bericht: {result.get('report_path', 'Kein Bericht erstellt')}")
                    else:
                        print("Analyse fehlgeschlagen.")

                elif args.command == "regimes":
                    # Nur Marktregime-Erkennung
                    regimes = framework.detect_market_regimes(
                        n_regimes=args.n_regimes,
                        symbols=args.symbols
                    )

                    if regimes:
                        print(f"Marktregime-Erkennung abgeschlossen. {len(regimes)} Regime identifiziert.")
                        for regime_id, label in regimes.items():
                            print(f"  Regime {regime_id}: {label}")
                    else:
                        print("Marktregime-Erkennung fehlgeschlagen.")

                elif args.command == "clusters":
                    # Nur Asset-Cluster-Analyse
                    clusters = framework.analyze_asset_clusters(
                        symbols=args.symbols
                    )

                    if not clusters.empty:
                        n_clusters = len(clusters['cluster'].unique())
                        print(f"Asset-Cluster-Analyse abgeschlossen. {n_clusters} Cluster identifiziert.")

                        # Anzahl der Assets pro Cluster
                        cluster_counts = clusters['cluster'].value_counts()
                        for cluster, count in cluster_counts.items():
                            print(f"  Cluster {cluster}: {count} Assets")
                    else:
                        print("Asset-Cluster-Analyse fehlgeschlagen.")

                elif args.command == "optimize":
                    # Nur Parameter-Optimierung
                    default_param_space = {
                        'technical.rsi.period': [9, 14, 21],
                        'technical.rsi.oversold': [30, 35, 40],
                        'technical.rsi.overbought': [60, 65, 70],
                        'technical.macd.fast': [8, 12],
                        'technical.macd.slow': [21, 26],
                        'risk.stop_loss': [0.03, 0.05, 0.07],
                        'risk.take_profit': [0.06, 0.10, 0.15]
                    }

                    result = framework.optimize_strategy_parameters(
                        strategy_name=args.strategy,
                        param_space=default_param_space,
                        regime_specific=args.regime_specific
                    )

                    if result:
                        print(f"Parameter-Optimierung abgeschlossen.")

                        if args.regime_specific:
                            print("Regime-spezifische Parameter:")
                            for regime, data in result.get('params_by_regime', {}).items():
                                print(f"  Regime {regime}:")
                                for key, value in data.get('params', {}).items():
                                    print(f"    {key}: {value}")
                        else:
                            print("Optimierte Parameter:")
                            for key, value in result.get('params', {}).items():
                                print(f"  {key}: {value}")
                    else:
                        print("Parameter-Optimierung fehlgeschlagen.")

                else:
                    # Kein Befehl angegeben, Hilfe anzeigen
                    parser.print_help()

            if __name__ == "__main__":
                main()