#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MarketRegimeDetector für den Trading Bot und Backtest-Modul.
Erkennt und klassifiziert Marktregimes basierend auf historischen Daten.
"""
import traceback
import logging
import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib


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
        self.feature_columns = []  # Speichern der Feature-Spalten beim Training
        self.logger = logging.getLogger(__name__)
        self.regime_descriptions = {}  # Dictionary für Beschreibungen der Regimes
        self.coins = []  # Liste der unterstützten Coins

    def load_market_data(self, symbols: List[str] = None,
                         data_manager=None,  # DataManager-Objekt
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

                    if df is not None and not df.empty:
                        self.market_data[symbol] = df
                        self.logger.info(f"Daten für {symbol} geladen: {len(df)} Einträge")

                if not self.market_data:
                    self.logger.error("Keine Marktdaten geladen")
                    return False

                self.logger.info(f"Marktdaten für {len(self.market_data)} Symbole geladen")
                # Liste der Coins aktualisieren
                self.coins = [symbol.split('/')[0] for symbol in self.market_data.keys()]
                return True

            # Alternativ: Direkt aus dem Verzeichnis laden
            binance_dir = os.path.join(self.data_dir, "binance")

            if not os.path.exists(binance_dir):
                self.logger.error(f"Verzeichnis nicht gefunden: {binance_dir}")
                return False

            # Wenn keine Symbole angegeben, alle verfügbaren laden
            if not symbols:
                symbols = []
                for filename in os.listdir(binance_dir):
                    if filename.endswith(f"_{timeframe}.csv") or timeframe in filename and filename.endswith(".csv"):
                        parts = filename.split("_")
                        if len(parts) >= 2:
                            symbol = parts[0]
                            quote = parts[1]
                            symbols.append(f"{symbol}/{quote}")

            # Daten für jedes Symbol laden
            for symbol in symbols:
                try:
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
                        # Überprüfen, ob timestamp bereits ein Datetime-Objekt ist
                        if not pd.api.types.is_datetime64_dtype(df['timestamp']):
                            try:
                                # Versuchen, als ms-Timestamp zu parsen
                                df['date'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')

                                # Wenn das nicht funktioniert, als String-Datum versuchen
                                if df['date'].isna().all():
                                    df['date'] = pd.to_datetime(df['timestamp'], errors='coerce')
                            except:
                                # Direkt als String-Datum versuchen
                                df['date'] = pd.to_datetime(df['timestamp'], errors='coerce')
                        else:
                            df['date'] = df['timestamp']

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
                except Exception as e:
                    self.logger.error(f"Fehler beim Laden von {symbol}: {e}")

            if not self.market_data:
                self.logger.error("Keine Marktdaten geladen")
                return False

            self.logger.info(f"Marktdaten für {len(self.market_data)} Symbole geladen")
            # Liste der Coins aktualisieren
            self.coins = [symbol.split('/')[0] for symbol in self.market_data.keys()]
            return True

        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Marktdaten: {e}")
            return False

    def extract_market_features(self, data=None) -> pd.DataFrame:
        """
        Extrahiert relevante Features für die Regime-Erkennung.

        Args:
            data: Optional bereits geladene Marktdaten

        Returns:
            DataFrame mit täglichen Features für alle Märkte
        """
        market_data = data if data is not None else self.market_data

        if not market_data:
            self.logger.error("Keine Marktdaten geladen")
            return pd.DataFrame()

        try:
            # Bitcoin-Daten als Referenz (sollte vorhanden sein)
            btc_symbol = next((s for s in market_data.keys() if s.startswith('BTC')), None)

            if not btc_symbol:
                self.logger.error("Keine Bitcoin-Daten verfügbar")
                return pd.DataFrame()

            btc_df = market_data[btc_symbol].copy()

            # Leerer DataFrame mit Datum als Index
            features_df = pd.DataFrame(index=btc_df.index)

            # 1. Bitcoin-spezifische Features (als Marktindikator)
            # Tägliche Returns
            btc_df['return'] = btc_df['close'].pct_change()

            # Volatilität (20-Tage Rolling Std)
            btc_df['volatility_20d'] = btc_df['return'].rolling(20).std()

            # EMAs berechnen
            btc_df['ema_12'] = btc_df['close'].ewm(span=12, adjust=False).mean()
            btc_df['ema_20'] = btc_df['close'].ewm(span=20, adjust=False).mean()
            btc_df['ema_26'] = btc_df['close'].ewm(span=26, adjust=False).mean()
            btc_df['ema_50'] = btc_df['close'].ewm(span=50, adjust=False).mean()
            btc_df['ema_200'] = btc_df['close'].ewm(span=200, adjust=False).mean()

            # EMA-Verhältnisse berechnen
            btc_df['ema_ratio'] = btc_df['ema_20'] / btc_df['ema_50']

            # Relative Stärke zum EMA
            btc_df['rel_to_ema20'] = btc_df['close'] / btc_df['ema_20'] - 1
            btc_df['rel_to_ema50'] = btc_df['close'] / btc_df['ema_50'] - 1
            btc_df['rel_to_ema200'] = btc_df['close'] / btc_df['ema_200'] - 1

            # MACD berechnen
            btc_df['macd'] = btc_df['ema_12'] - btc_df['ema_26']
            btc_df['macd_signal'] = btc_df['macd'].ewm(span=9, adjust=False).mean()
            btc_df['macd_hist'] = btc_df['macd'] - btc_df['macd_signal']

            # MACD zu EMA-50 Verhältnis
            # Vermeiden von Division durch Null
            ema_50_nonzero = btc_df['ema_50'].replace(0, 0.00001)
            btc_df['macd_signal_ratio'] = btc_df['macd_signal'] / ema_50_nonzero

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

            # Vermeiden von Division durch Null
            atr_nonzero = btc_df['atr'].replace(0, 0.00001)

            btc_df['di_plus'] = (btc_df['dm_plus'].rolling(atr_period).mean() /
                                 atr_nonzero) * 100
            btc_df['di_minus'] = (btc_df['dm_minus'].rolling(atr_period).mean() /
                                  atr_nonzero) * 100

            di_diff = abs(btc_df['di_plus'] - btc_df['di_minus'])
            di_sum = btc_df['di_plus'] + btc_df['di_minus']
            di_sum = di_sum.replace(0, 0.00001)  # Vermeiden von Division durch Null
            btc_df['dx'] = (di_diff / di_sum) * 100
            btc_df['adx'] = btc_df['dx'].rolling(atr_period).mean()

            # RSI berechnen
            delta = btc_df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            avg_loss = avg_loss.replace(0, 0.00001)  # Vermeiden von Division durch Null
            rs = avg_gain / avg_loss
            btc_df['rsi'] = 100 - (100 / (1 + rs))

            # Mittlere Returns
            btc_df['mean_return'] = btc_df['return'].rolling(window=20).mean()

            # Volatilität
            btc_df['volatility'] = btc_df['return'].rolling(window=20).std()

            # BTC Momentum (14-Tage)
            btc_df['btc_momentum_14d'] = (
                    btc_df['close'] / btc_df['close'].shift(14) - 1
            )

            # Hinzufügen der Bitcoin-Features zum Feature DataFrame
            bitcoin_features = [
                'return', 'volatility_20d', 'volatility',
                'rel_to_ema20', 'rel_to_ema50', 'rel_to_ema200',
                'adx', 'mean_return', 'rsi', 'ema_ratio', 'macd_signal_ratio',
                'btc_momentum_14d'
            ]

            if 'volume' in btc_df.columns:
                bitcoin_features.extend(['volume_change', 'volume_ma_ratio'])

            for feature in bitcoin_features:
                if feature in btc_df.columns:
                    features_df[f'btc_{feature}'] = btc_df[feature]
                else:
                    # Feature existiert nicht, mit 0 füllen
                    features_df[f'btc_{feature}'] = 0
                    self.logger.warning(f"Bitcoin-Feature {feature} nicht verfügbar, mit 0 gefüllt")

            # 2. Altcoin zu Bitcoin Relative Stärke
            altcoin_symbols = []
            for symbol, df in market_data.items():
                if symbol == btc_symbol:
                    continue

                altcoin_symbols.append(symbol)

                # Symbol für die Spaltenbezeichnung kürzen
                short_name = symbol.split('/')[0]

                # Tägliche Returns
                df_copy = df.copy()
                df_copy['return'] = df_copy['close'].pct_change()

                # Relative Performance zu Bitcoin
                merged_df = pd.merge(
                    df_copy['return'],
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
            outperform_columns = [c for c in features_df.columns if
                                  c.startswith('rel_strength_') and not c.endswith('_ma20')]

            if outperform_columns:
                features_df['pct_outperform_btc'] = (
                        (features_df[outperform_columns] > 0).sum(axis=1) / len(outperform_columns)
                )
            else:
                features_df['pct_outperform_btc'] = 0

            # 4. Rollende Korrelationen
            # Durchschnittliche Korrelation zwischen Altcoins
            if len(altcoin_symbols) >= 2:
                # Sammel-DataFrame für Returns
                returns_df = pd.DataFrame(index=features_df.index)

                for symbol in altcoin_symbols:
                    short_name = symbol.split('/')[0]
                    if symbol in market_data:
                        symbol_df = market_data[symbol].copy()
                        returns_df[short_name] = symbol_df['close'].pct_change()

                # Berechne die durchschnittliche Korrelation
                if len(returns_df.columns) > 1:
                    corr_matrix = returns_df.rolling(20).corr()
                    # Extrahiere nur die Korrelationen zwischen verschiedenen Assets
                    avg_correlations = []

                    for date in returns_df.index[20:]:
                        daily_corr = corr_matrix.loc[date]
                        # Nimm nur die obere Dreiecksmatrix (ohne Diagonale)
                        upper_triangle = np.triu_indices(len(returns_df.columns), k=1)
                        correlations = daily_corr.values[upper_triangle]

                        if len(correlations) > 0 and not np.isnan(correlations).all():
                            avg_correlations.append(np.nanmean(correlations))
                        else:
                            avg_correlations.append(0)

                    # Pad mit 0 für die ersten 19 Tage
                    avg_correlations = [0] * 20 + avg_correlations
                    features_df['avg_altcoin_correlation'] = avg_correlations[:len(features_df)]
                else:
                    features_df['avg_altcoin_correlation'] = 0
            else:
                features_df['avg_altcoin_correlation'] = 0

            # Alle NaN-Werte entfernen, da sie für das Training nicht nützlich sind
            features_df = features_df.dropna()

            # Speichern der Feature-Spalten für spätere Konsistenzprüfung
            self.feature_columns = features_df.columns.tolist()

            self.logger.info(f"Features extrahiert: {len(features_df)} Zeitpunkte, {len(features_df.columns)} Features")
            return features_df

        except Exception as e:
            self.logger.error(f"Fehler bei der Feature-Extraktion: {e}")
            self.logger.error(traceback.format_exc())
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

            # Speichern der Feature-Spalten für spätere Verwendung
            self.feature_columns = X.columns.tolist()

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
            self.logger.error(traceback.format_exc())
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
            # Prüfen, ob die Features mit den Trainingsfeatures übereinstimmen
            missing_columns = set(self.feature_columns) - set(features.columns)
            extra_columns = set(features.columns) - set(self.feature_columns)

            # Feature-Konsistenz sicherstellen
            if missing_columns or extra_columns:
                self.logger.warning(f"Fehlende Features: {missing_columns}")
                self.logger.warning(f"Zusätzliche Features: {extra_columns}")

                # Features-DataFrame anpassen
                aligned_features = pd.DataFrame(index=features.index)

                # Vorhandene Features kopieren
                for col in self.feature_columns:
                    if col in features.columns:
                        aligned_features[col] = features[col]
                    else:
                        # Fehlende Features mit Nullen füllen
                        aligned_features[col] = 0
                        self.logger.warning(f"Fehlendes Feature {col} mit 0 gefüllt")

                # Angepasste Features verwenden
                features = aligned_features

            # Nur die Spalten auswählen, die im Training verwendet wurden
            features = features[self.feature_columns]

            # Features normalisieren
            features_scaled = self.scaler.transform(features)

            # Regime vorhersagen
            regime = self.regime_model.predict(features_scaled)[0]

            self.current_regime = regime

            regime_label = self.regime_labels.get(regime, f"Regime {regime}")
            self.logger.info(f"Aktuelles Marktregime: {regime} - {regime_label}")

            return regime

        except Exception as e:
            self.logger.error(f"Fehler bei der Regime-Vorhersage: {e}")
            self.logger.error(traceback.format_exc())
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
            self.logger.error(traceback.format_exc())

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
                    # Vermeiden von Division durch Null
                    overall_means_nonzero = overall_means.replace(0, 0.00001)
                    relative_means = regime_means / overall_means_nonzero

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
                            else:
                                regime_characteristics.append("BTC-Neutral")

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

                    # Label erstellen (maximal 3 Charakteristiken)
                    if regime_characteristics:
                        self.regime_labels[regime] = " & ".join(regime_characteristics[:3])
                    else:
                        self.regime_labels[regime] = f"Regime {regime}"

                    # Beschreibung hinzufügen
                    self.regime_descriptions[regime] = f"Regime {regime}: {self.regime_labels[regime]}"

                    self.logger.info(f"Regime {regime} charakterisiert als: {self.regime_labels[regime]}")

        except Exception as e:
            self.logger.error(f"Fehler bei der Analyse der Regime-Charakteristiken: {e}")
            self.logger.error(traceback.format_exc())
            # Sicherstellen, dass alle Regime-Labels existieren
            for regime in range(self.n_regimes):
                if regime not in self.regime_labels:
                    self.regime_labels[regime] = f"Regime {regime}"
                    self.regime_descriptions[regime] = f"Regime {regime}: Uncharakterisiert"

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
            if performance_by_regime:
                perf_df = pd.DataFrame(performance_by_regime).T

                # NaN durch 0 ersetzen
                perf_df = perf_df.fillna(0)

                # Regime-Namen hinzufügen
                perf_df.columns = [self.regime_labels.get(col, f"Regime {col}") for col in perf_df.columns]

                self.regime_performances = perf_df

                self.logger.info(f"Performance-Analyse pro Regime abgeschlossen")
            else:
                self.logger.warning("Keine Performance-Daten verfügbar")

        except Exception as e:
            self.logger.error(f"Fehler bei der Analyse der Performance pro Regime: {e}")
            self.logger.error(traceback.format_exc())

    def extract_trading_rules(self, perf_series):
        """
        Extrahiert Trading-Regeln basierend auf der Performance-Serie

        Parameters:
        -----------
        perf_series : pandas.Series oder pandas.DataFrame
            Performance-Serie oder DataFrame mit Coin-Performance

        Returns:
        --------
        dict
            Trading-Regeln mit Top-Performern und Bottom-Performern
        """
        try:
            # Überprüfen ob perf_series ein DataFrame oder eine Series ist
            if isinstance(perf_series, pd.DataFrame):
                # Wenn es ein DataFrame ist, benötigen wir eine Spalte für sort_values
                # Nehmen wir an, die letzte Spalte enthält die Performancewerte
                col_name = perf_series.columns[-1]
                top_performers = perf_series.sort_values(by=col_name, ascending=False).head(3)
                bottom_performers = perf_series.sort_values(by=col_name, ascending=True).head(3)

                # Extrahiere nur die Coin-Namen als Liste
                top_coins = top_performers.index.tolist()
                bottom_coins = bottom_performers.index.tolist()
            elif isinstance(perf_series, pd.Series):
                # Für Series ist kein 'by' Parameter notwendig
                top_performers = perf_series.sort_values(ascending=False).head(3)
                bottom_performers = perf_series.sort_values(ascending=True).head(3)

                # Extrahiere nur die Coin-Namen als Liste
                top_coins = top_performers.index.tolist()
                bottom_coins = bottom_performers.index.tolist()
            else:
                # Falls perf_series weder DataFrame noch Series ist
                self.logger.error(f"Unerwarteter Typ für perf_series: {type(perf_series)}")
                return None

            return {
                "top_performers": top_coins,
                "bottom_performers": bottom_coins
            }
        except Exception as e:
            self.logger.error(f"Fehler bei der Extraktion von Trading-Regeln: {str(e)}")
            self.logger.error(f"Traceback (most recent call last):\n{traceback.format_exc()}")
            return None

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
            if filepath:
                directory = os.path.dirname(filepath)
                if directory:
                    os.makedirs(directory, exist_ok=True)

            # Modell als Pickle speichern
            model_data = {
                "regime_model": self.regime_model,
                "scaler": self.scaler,
                "n_regimes": self.n_regimes,
                "regime_labels": self.regime_labels,
                "regime_transitions": self.regime_transitions,
                "regime_performances": self.regime_performances,
                "regime_durations": getattr(self, "regime_durations", {}),
                "feature_columns": self.feature_columns,  # Wichtig: Feature-Spalten speichern
                "regime_descriptions": self.regime_descriptions  # Regime-Beschreibungen speichern
            }

            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)

            self.logger.info(f"Regime-Modell gespeichert unter {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Fehler beim Speichern des Modells: {e}")
            self.logger.error(traceback.format_exc())
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

            # Feature-Spalten laden (wichtig für die Kompatibilität)
            if "feature_columns" in model_data:
                self.feature_columns = model_data["feature_columns"]
            else:
                self.logger.warning("Keine Feature-Spalten im Modell gefunden. Prädiktionen könnten fehlschlagen.")

            # Regime-Beschreibungen laden
            if "regime_descriptions" in model_data:
                self.regime_descriptions = model_data["regime_descriptions"]
            else:
                # Beschreibungen aus Labels erstellen
                self.regime_descriptions = {
                    regime: f"Regime {regime}: {label}"
                    for regime, label in self.regime_labels.items()
                }

            self.model_trained = True

            self.logger.info(f"Regime-Modell geladen von {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Fehler beim Laden des Modells: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def get_current_regime_info(self):
        """
        Gibt Informationen zum aktuellen Marktregime zurück

        Returns:
        --------
        dict
            Informationen zum aktuellen Marktregime
        """
        try:
            current_regime = self.detect_market_regime()
            regime_label = self.get_regime_label(current_regime)

            # Erstellen der Performance-Serie für extract_trading_rules
            # Verwenden der Marktdaten um eine Performance-Serie zu erstellen
            perf_series = None

            if self.market_data:
                # Berechne Performance für jeden Coin in market_data
                perf_series = pd.Series()

                for symbol, df in self.market_data.items():
                    if not df.empty:
                        # Extrahiere Währungsnamen aus dem Symbol (z.B. "BTC/USDT" -> "BTC")
                        coin = symbol.split('/')[0]

                        # Prozentuale Veränderung der letzten 30 Tage (oder weniger, falls nicht genügend Daten)
                        days = min(30, len(df))
                        if days > 1:
                            perf = (df['close'].iloc[-1] / df['close'].iloc[-days] - 1) * 100
                            perf_series[coin] = perf

            # Falls keine Marktdaten verfügbar oder keine Performance berechnet werden konnte
            if perf_series is None or len(perf_series) == 0:
                self.logger.warning("Keine Performance-Daten verfügbar für Trading Rules")
                trading_rules = None
            else:
                # Extrahiere Trading-Regeln mit der nun vorhandenen Performance-Serie
                trading_rules = self.extract_trading_rules(perf_series)

            return {
                "regime": current_regime,
                "label": regime_label,
                "description": self.regime_descriptions.get(current_regime, "Unbekanntes Regime"),
                "trading_rules": trading_rules
            }
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der Regime-Informationen: {str(e)}")
            self.logger.error(f"Traceback (most recent call last):\n{traceback.format_exc()}")
            # Statt None zurückzugeben, geben wir ein Minimalobjekt zurück
            return {
                "regime": 0,  # Standardregime (z.B. neutral)
                "label": "Fehler bei Regimebestimmung",
                "description": "Fehler bei der Bestimmung des Marktregimes",
                "trading_rules": {"top_performers": [], "bottom_performers": []}
            }

    def analyze(self, file_path=None, data=None, plot=False):
        """
        Analysiert die Marktdaten und gibt ein DataFrame mit Regime-Labels zurück.

        Args:
            file_path: Pfad zur Datei mit OHLCV-Daten (optional)
            data: DataFrame mit OHLCV-Daten (optional, wird gegenüber file_path bevorzugt)
            plot: Ob ein Plot erstellt werden soll (nicht implementiert)

        Returns:
            DataFrame mit Regime-Labels oder None bei Fehler
        """
        try:
            # Daten laden, wenn angegeben
            if data is not None:
                market_data = {"BTC/USDT": data}  # Nehmen wir an, es handelt sich um BTC-Daten
            elif file_path is not None:
                # Datei laden
                if not os.path.exists(file_path):
                    self.logger.error(f"Datei nicht gefunden: {file_path}")
                    return None

                df = pd.read_csv(file_path)

                # Datum als Index setzen, falls vorhanden
                if 'timestamp' in df.columns:
                    if not pd.api.types.is_datetime64_dtype(df['timestamp']):
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)

                market_data = {"BTC/USDT": df}
            else:
                self.logger.error("Weder Datei noch Daten angegeben")
                return None

            # Features extrahieren
            features_df = self.extract_market_features(market_data)

            if features_df.empty:
                self.logger.error("Keine Features extrahiert")
                return None

            # Wenn das Modell nicht trainiert ist, trainieren
            if not self.model_trained:
                success = self.train_regime_model(features_df)
                if not success:
                    self.logger.error("Modell-Training fehlgeschlagen")
                    return None

            # Regimes für jeden Zeitpunkt vorhersagen
            regimes = []
            for idx, row in features_df.iterrows():
                # Features-DataFrame für einen einzelnen Zeitpunkt
                single_features = pd.DataFrame([row])

                # Regime vorhersagen
                regime = self.predict_regime(single_features)

                regimes.append({
                    'date': idx,
                    'regime': regime,
                    'regime_name': self.regime_labels.get(regime, f"Regime {regime}")
                })

            # Zu DataFrame konvertieren
            regimes_df = pd.DataFrame(regimes)
            regimes_df.set_index('date', inplace=True)

            # Mit den ursprünglichen Daten zusammenführen
            result_df = pd.merge(
                market_data["BTC/USDT"],
                regimes_df,
                left_index=True,
                right_index=True,
                how='inner'
            )

            return result_df

        except Exception as e:
            self.logger.error(f"Fehler bei der Analyse: {e}")
            self.logger.error(traceback.format_exc())
            return None

    def detect_market_regime(self):
        """
        Erkennt das aktuelle Marktregime basierend auf den neuesten Daten.

        Returns:
        --------
        int
            ID des erkannten Marktregimes oder -1 bei Fehler
        """
        try:
            # Features aus den aktuellsten Daten extrahieren
            # Wir nehmen die aktuellsten X Tage (z.B. 30 Tage) für eine stabilere Vorhersage
            latest_data = {}
            for symbol, df in self.market_data.items():
                # Nehme die letzten 30 Tage, oder alle verfügbaren Daten, wenn weniger
                days = min(30, len(df))
                latest_data[symbol] = df.iloc[-days:]

            # Features aus den aktuellen Daten extrahieren
            features_df = self.extract_market_features(latest_data)

            if features_df.empty:
                self.logger.error("Keine Features für Regime-Erkennung extrahiert")
                return -1

            # Nur die letzte Zeile verwenden (aktuellster Zeitpunkt)
            latest_features = features_df.iloc[-1:].copy()

            # Regime vorhersagen
            regime = self.predict_regime(latest_features)

            self.current_regime = regime
            return regime

        except Exception as e:
            self.logger.error(f"Fehler bei der Marktregime-Erkennung: {str(e)}")
            self.logger.error(f"Traceback (most recent call last):\n{traceback.format_exc()}")
            return -1

    def get_regime_label(self, regime_id):
        """
        Gibt das Label für das angegebene Regime zurück.

        Parameters:
        -----------
        regime_id : int
            ID des Regimes

        Returns:
        --------
        str
            Label des Regimes oder "Unbekanntes Regime" bei ungültiger ID
        """
        if regime_id == -1:
            return "Fehler bei Regimebestimmung"

        # Regime-Label aus dem Dictionary abrufen oder Standardlabel verwenden
        return self.regime_labels.get(regime_id, f"Regime {regime_id}")