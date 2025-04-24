#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML-basierte Handelsstrategie für den Trading Bot.
Diese Strategie integriert Marktregime-Erkennung und Asset-Clustering
für bessere Handelsentscheidungen.
"""

import logging
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List

from strategies.strategy_base import Strategy
from core.position import Position

# Prüfen, ob ML-Komponenten verfügbar sind
try:
    from ml_components.market_regime import MarketRegimeDetector
    from ml_components.asset_clusters import AssetClusterAnalyzer

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


class MLStrategy(Strategy):
    """
    ML-basierte Handelsstrategie, die mehrere ML-Komponenten integriert.
    Diese Strategie kombiniert:
    1. Marktregime-Erkennung
    2. Asset-Clustering
    3. Technische Indikatoren
    """

    def __init__(self, settings):
        """
        Initialisiert die ML-Strategie.

        Args:
            settings: Einstellungen aus der Konfigurationsdatei
        """
        super().__init__(settings)
        self.logger = logging.getLogger(__name__)

        # ML-Komponenten
        self.ml_enabled = settings.get('ml.enabled', False) and ML_AVAILABLE
        self.regime_detector = None
        self.asset_analyzer = None
        self.current_regime = None
        self.clusters = None

        # Strategie-Parameter
        self.rsi_overbought = settings.get('strategy.rsi_overbought', 70)
        self.rsi_oversold = settings.get('strategy.rsi_oversold', 30)
        self.ema_short = settings.get('strategy.ema_short', 20)
        self.ema_long = settings.get('strategy.ema_long', 50)
        self.atr_periods = settings.get('strategy.atr_periods', 14)
        self.atr_multiplier = settings.get('strategy.atr_multiplier', 2.0)

        # Risikomanagement-Parameter
        self.stop_loss_pct = settings.get('risk.stop_loss', 0.03)
        self.take_profit_pct = settings.get('risk.take_profit', 0.06)
        self.use_trailing_stop = settings.get('risk.use_trailing_stop', True)
        self.trailing_stop_pct = settings.get('risk.trailing_stop', 0.02)
        self.trailing_activation_pct = settings.get('risk.trailing_activation', 0.02)

        # ML-Komponenten initialisieren
        if self.ml_enabled:
            self._initialize_ml_components()

    def _initialize_ml_components(self):
        """Initialisiert die ML-Komponenten"""
        try:
            # Verzeichnisse aus den Einstellungen
            data_dir = self.settings.get('ml.data_dir', 'data/market_data')
            models_dir = self.settings.get('ml.models_dir', 'data/ml_models')

            # Marktregime-Detektor initialisieren
            self.regime_detector = MarketRegimeDetector(data_dir=data_dir)

            # Versuchen, ein vorhandenes Modell zu laden
            regime_model_path = f"{models_dir}/regime_model.pkl"
            try:
                if self.regime_detector.load_model(regime_model_path):
                    self.logger.info(f"Regime-Modell aus {regime_model_path} geladen")
            except Exception as e:
                self.logger.warning(f"Konnte Regime-Modell nicht laden: {e}")

            # Asset-Cluster-Analyzer initialisieren
            self.asset_analyzer = AssetClusterAnalyzer(data_dir=data_dir)

            self.logger.info("ML-Komponenten initialisiert")
        except Exception as e:
            self.logger.error(f"Fehler bei der Initialisierung der ML-Komponenten: {e}")
            self.ml_enabled = False

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Berechnet technische Indikatoren für den DataFrame.

        Args:
            df: DataFrame mit OHLCV-Daten

        Returns:
            DataFrame mit berechneten Indikatoren
        """
        # Kopie des DataFrames erstellen, um es nicht zu verändern
        df_copy = df.copy()

        # 1. RSI (Relative Strength Index)
        delta = df_copy['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

        # Avoid division by zero
        loss = loss.replace(0, 0.00001)

        rs = gain / loss
        df_copy['rsi'] = 100 - (100 / (1 + rs))

        # 2. EMAs (Exponential Moving Averages)
        df_copy['ema_short'] = df_copy['close'].ewm(span=self.ema_short, adjust=False).mean()
        df_copy['ema_long'] = df_copy['close'].ewm(span=self.ema_long, adjust=False).mean()

        # 3. Bollinger Bands
        df_copy['sma_20'] = df_copy['close'].rolling(window=20).mean()
        df_copy['std_20'] = df_copy['close'].rolling(window=20).std()
        df_copy['upper_band'] = df_copy['sma_20'] + (df_copy['std_20'] * 2)
        df_copy['lower_band'] = df_copy['sma_20'] - (df_copy['std_20'] * 2)

        # 4. ATR (Average True Range)
        df_copy['tr'] = pd.DataFrame({
            'hl': df_copy['high'] - df_copy['low'],
            'hc': abs(df_copy['high'] - df_copy['close'].shift(1)),
            'lc': abs(df_copy['low'] - df_copy['close'].shift(1))
        }).max(axis=1)

        df_copy['atr'] = df_copy['tr'].rolling(window=self.atr_periods).mean()

        # 5. MACD (Moving Average Convergence Divergence)
        df_copy['ema_12'] = df_copy['close'].ewm(span=12, adjust=False).mean()
        df_copy['ema_26'] = df_copy['close'].ewm(span=26, adjust=False).mean()
        df_copy['macd'] = df_copy['ema_12'] - df_copy['ema_26']
        df_copy['macd_signal'] = df_copy['macd'].ewm(span=9, adjust=False).mean()
        df_copy['macd_hist'] = df_copy['macd'] - df_copy['macd_signal']

        # 6. Volatilitätsmetrik
        df_copy['return'] = df_copy['close'].pct_change()
        df_copy['volatility'] = df_copy['return'].rolling(window=20).std()

        return df_copy

    def generate_signal(self, df: pd.DataFrame, symbol: str,
                        current_position: Optional[Position] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generiert ein Handelssignal für das angegebene Symbol.

        Args:
            df: DataFrame mit OHLCV-Daten
            symbol: Handelssymbol (z.B. 'BTC/USDT')
            current_position: Aktuelle Position (falls vorhanden)

        Returns:
            Tuple aus Signal ('BUY', 'SELL', 'HOLD') und Signaldaten
        """
        if df.empty or len(df) < 50:
            return "HOLD", {"signal": "HOLD", "reason": "insufficient_data"}

        # Technische Indikatoren berechnen
        df_with_indicators = self.calculate_indicators(df)

        # Basisstrategie-Signal basierend auf technischen Indikatoren
        base_signal, base_data = self._generate_base_signal(df_with_indicators, symbol, current_position)

        # Wenn ML-Komponenten aktiviert sind, Signal anreichern
        if self.ml_enabled and ML_AVAILABLE:
            return self._enhance_signal_with_ml(df_with_indicators, symbol, base_signal, base_data, current_position)

        return base_signal, base_data

    def _generate_base_signal(self, df: pd.DataFrame, symbol: str,
                              current_position: Optional[Position] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generiert das Basisstrategie-Signal basierend auf technischen Indikatoren.

        Args:
            df: DataFrame mit Indikatoren
            symbol: Handelssymbol
            current_position: Aktuelle Position (falls vorhanden)

        Returns:
            Tuple aus Signal und Signaldaten
        """
        try:
            # Letzte Zeile mit aktuellen Daten
            current = df.iloc[-1]
            previous = df.iloc[-2]

            # Signal- und Konfidenzwerte initialisieren
            signal = "HOLD"
            confidence = 0.5
            reason = ""

            # 1. Positionsmanagement (wenn eine Position vorhanden ist)
            if current_position:
                # Aktueller Preis
                current_price = current['close']

                # Take-Profit und Stop-Loss überprüfen
                entry_price = current_position.entry_price
                profit_pct = (current_price / entry_price - 1) * 100 if current_position.side == 'buy' else (
                                                                                                                        entry_price / current_price - 1) * 100

                # Verlust-Signal: Wenn der Preis unter den EMA200 fällt
                if current_position.side == 'buy' and current_price < current['ema_long'] and previous['close'] > \
                        previous['ema_long']:
                    signal = "SELL"
                    confidence = 0.7
                    reason = "price_below_ema_long"

                # Verkaufs-Signal: RSI überkauft und Preis in oberer Bollinger-Band
                elif current_position.side == 'buy' and current['rsi'] > self.rsi_overbought and current_price > \
                        current['upper_band']:
                    signal = "SELL"
                    confidence = 0.8
                    reason = "overbought_condition"

                # Bei Trendumkehr verkaufen
                elif current_position.side == 'buy' and current['macd'] < current['macd_signal'] and previous['macd'] > \
                        previous['macd_signal']:
                    signal = "SELL"
                    confidence = 0.6
                    reason = "macd_crossover_down"

            # 2. Neue Positionen eröffnen
            else:
                # Kaufsignal: RSI überkauft und Preis in unterer Bollinger-Band
                if current['rsi'] < self.rsi_oversold and current['close'] < current['lower_band']:
                    signal = "BUY"
                    confidence = 0.7
                    reason = "oversold_condition"

                # Kaufsignal: EMA-Kreuzung und positiver MACD
                elif (current['ema_short'] > current['ema_long'] and
                      previous['ema_short'] <= previous['ema_long'] and
                      current['macd'] > 0):
                    signal = "BUY"
                    confidence = 0.8
                    reason = "ema_crossover_up_with_positive_macd"

                # Kaufsignal: MACD-Kreuzung über Signallinie
                elif current['macd'] > current['macd_signal'] and previous['macd'] <= previous['macd_signal']:
                    signal = "BUY"
                    confidence = 0.6
                    reason = "macd_crossover_up"

            # Risikomanagement-Parameter festlegen
            stop_loss_pct = self.stop_loss_pct
            take_profit_pct = self.take_profit_pct

            # Bei hoher Volatilität engeren Stop-Loss setzen
            current_volatility = current['volatility']
            avg_volatility = df['volatility'].mean()

            if current_volatility > avg_volatility * 1.5:
                stop_loss_pct = self.stop_loss_pct * 0.7
                take_profit_pct = self.take_profit_pct * 0.8

            # Signaldaten zusammenstellen
            signal_data = {
                "signal": signal,
                "confidence": confidence,
                "reason": reason,
                "stop_loss_pct": stop_loss_pct,
                "take_profit_pct": take_profit_pct,
                "use_trailing_stop": self.use_trailing_stop,
                "trailing_stop_pct": self.trailing_stop_pct,
                "trailing_activation_pct": self.trailing_activation_pct,
                "indicators": {
                    "rsi": current['rsi'],
                    "ema_short": current['ema_short'],
                    "ema_long": current['ema_long'],
                    "macd": current['macd'],
                    "macd_signal": current['macd_signal'],
                    "volatility": current['volatility']
                }
            }

            return signal, signal_data

        except Exception as e:
            self.logger.error(f"Fehler bei der Signalgenerierung für {symbol}: {e}")
            return "HOLD", {"signal": "HOLD", "reason": "error", "error": str(e)}

    def _enhance_signal_with_ml(self, df: pd.DataFrame, symbol: str,
                                base_signal: str, base_data: Dict[str, Any],
                                current_position: Optional[Position] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Verbessert das Basisstrategie-Signal mit ML-Komponenten.

        Args:
            df: DataFrame mit Indikatoren
            symbol: Handelssymbol
            base_signal: Signal der Basisstrategie
            base_data: Signaldaten der Basisstrategie
            current_position: Aktuelle Position (falls vorhanden)

        Returns:
            Tuple aus ML-verbessertem Signal und Signaldaten
        """
        try:
            # Kopie der Basis-Signaldaten erstellen
            ml_data = base_data.copy()
            ml_signal = base_signal

            # 1. Aktuelles Marktregime bestimmen (falls verfügbar)
            current_regime = self._detect_market_regime(df)

            if current_regime is not None and self.regime_detector and self.regime_detector.model_trained:
                # Aktuelles Regime speichern
                self.current_regime = current_regime
                regime_info = self.regime_detector.get_current_regime_info()

                # Regime-Label und -Information hinzufügen
                if regime_info:
                    ml_data["regime"] = current_regime
                    ml_data["regime_label"] = regime_info.get('label', f"Regime {current_regime}")
                    ml_data["regime_strategy"] = regime_info.get('strategy', "Neutral")

                    # Regime-basierte Anpassungen
                    regime_label = ml_data["regime_label"].lower()

                    # Bullisches Regime
                    if 'bull' in regime_label or 'aufwärtstrend' in regime_label:
                        if base_signal == "BUY":
                            ml_data["confidence"] = min(1.0, base_data["confidence"] * 1.2)
                        elif base_signal == "SELL":
                            # In bullischen Regimes nur verkaufen, wenn hohe Konfidenz
                            if base_data["confidence"] < 0.8:
                                ml_signal = "HOLD"

                    # Bearisches Regime
                    elif 'bear' in regime_label or 'abwärtstrend' in regime_label:
                        if base_signal == "BUY":
                            # In bearischen Regimes nur kaufen, wenn sehr hohe Konfidenz
                            if base_data["confidence"] < 0.8:
                                ml_signal = "HOLD"
                            else:
                                # Engeren Stop-Loss setzen
                                ml_data["stop_loss_pct"] = base_data["stop_loss_pct"] * 0.8
                        elif base_signal == "SELL":
                            ml_data["confidence"] = min(1.0, base_data["confidence"] * 1.2)

                    # Volatiles Regime
                    elif 'volatil' in regime_label or 'volatility' in regime_label:
                        # Engere Stops in volatilen Phasen
                        ml_data["stop_loss_pct"] = base_data["stop_loss_pct"] * 0.7
                        ml_data["take_profit_pct"] = base_data["take_profit_pct"] * 0.8
                        ml_data["use_trailing_stop"] = True
                        ml_data["trailing_stop_pct"] = base_data["trailing_stop_pct"] * 0.7

                        # Konservativere Signale
                        if base_signal == "BUY" and base_data["confidence"] < 0.7:
                            ml_signal = "HOLD"

            # 2. Asset-Cluster-Information nutzen (falls verfügbar)
            if self.asset_analyzer and hasattr(self.asset_analyzer,
                                               'clusters') and self.asset_analyzer.clusters is not None:
                # Kurznamen extrahieren (z.B. BTC aus BTC/USDT)
                short_name = symbol.split('/')[0]

                # Asset-Cluster abrufen
                if short_name in self.asset_analyzer.clusters.index:
                    cluster = self.asset_analyzer.clusters.loc[short_name, 'cluster']
                    ml_data["cluster"] = int(cluster)

                    # Cluster-Performance abrufen
                    if hasattr(self.asset_analyzer, 'cluster_performances'):
                        cluster_perf = self.asset_analyzer.cluster_performances.get(cluster, {})

                        # Performance-Metriken hinzufügen
                        ml_data["cluster_performance"] = {
                            "mean_return": cluster_perf.get('mean_return', 0),
                            "sharpe_ratio": cluster_perf.get('sharpe_ratio', 0),
                            "win_rate": cluster_perf.get('win_rate', 0)
                        }

                        # Für schlecht performende Cluster: Vorsichtiger sein
                        if cluster_perf.get('mean_return', 0) < 0:
                            if base_signal == "BUY":
                                # Nur kaufen bei sehr hoher Konfidenz
                                if base_data["confidence"] < 0.8:
                                    ml_signal = "HOLD"

                        # Für gut performende Cluster: Aggressiver sein
                        elif cluster_perf.get('sharpe_ratio', 0) > 1.0:
                            if base_signal == "BUY":
                                ml_data["confidence"] = min(1.0, base_data["confidence"] * 1.2)
                                # Größere Position zulassen
                                ml_data["position_size_factor"] = 1.2

            # ML-basierte Features hinzufügen
            ml_data["ml_enhanced"] = True
            ml_data["signal"] = ml_signal

            return ml_signal, ml_data

        except Exception as e:
            self.logger.error(f"Fehler bei der ML-Verbesserung für {symbol}: {e}")
            return base_signal, base_data

    def _detect_market_regime(self, df: pd.DataFrame) -> Optional[int]:
        """
        Erkennt das aktuelle Marktregime.

        Args:
            df: DataFrame mit historischen Daten

        Returns:
            Regime-ID oder None bei Fehler
        """
        if not self.ml_enabled or not self.regime_detector or not self.regime_detector.model_trained:
            return None

        try:
            # Einfache Feature-Extraktion für die Regime-Erkennung
            features = pd.DataFrame(index=[df.index[-1]])

            # Return-basierte Features
            returns = df['close'].pct_change()
            features['mean_return'] = returns.iloc[-20:].mean()
            features['volatility'] = returns.iloc[-20:].std()

            # Trendstärke
            features['ema_ratio'] = df['ema_short'].iloc[-1] / df['ema_long'].iloc[-1]

            # RSI Feature
            features['rsi'] = df['rsi'].iloc[-1] / 100  # Normalisieren

            # MACD Feature
            features['macd_signal_ratio'] = df['macd'].iloc[-1] / df['macd_signal'].iloc[-1] if df['macd_signal'].iloc[
                                                                                                    -1] != 0 else 0

            # Regime vorhersagen
            regime = self.regime_detector.predict_regime(features)

            return regime

        except Exception as e:
            self.logger.error(f"Fehler bei der Regime-Erkennung: {e}")
            return None

    def on_new_candle(self, df: pd.DataFrame, symbol: str):
        """
        Handler für neue Candle-Daten.
        Wird aufgerufen, wenn neue Marktdaten verfügbar sind.

        Args:
            df: OHLCV-DataFrame mit dem neuesten Candle
            symbol: Handelssymbol
        """
        # Hier können Sie Code hinzufügen, der bei jedem neuen Candle ausgeführt wird
        pass

    def train_ml_components(self, symbols: List[str], data_manager=None):
        """
        Trainiert die ML-Komponenten mit historischen Daten.

        Args:
            symbols: Liste von Handelssymbolen für das Training
            data_manager: DataManager-Instanz für Datenzugriff

        Returns:
            True bei Erfolg, False bei Fehler
        """
        if not self.ml_enabled:
            self.logger.warning("ML ist nicht aktiviert")
            return False

        success = False

        # Marktregime-Erkennung trainieren
        if self.regime_detector:
            try:
                self.logger.info("Trainiere Marktregime-Detektor...")
                # Daten laden
                if data_manager and self.regime_detector.load_market_data(
                        symbols=symbols,
                        data_manager=data_manager,
                        timeframe=self.settings.get('timeframes.analysis', '1d')
                ):
                    # Features extrahieren
                    features_df = self.regime_detector.extract_market_features()

                    if not features_df.empty:
                        # Modell trainieren
                        if self.regime_detector.train_regime_model(features_df):
                            self.logger.info("Marktregime-Detektor erfolgreich trainiert")

                            # Modell speichern
                            models_dir = self.settings.get('ml.models_dir', 'data/ml_models')
                            model_path = f"{models_dir}/regime_model.pkl"

                            if self.regime_detector.save_model(model_path):
                                self.logger.info(f"Marktregime-Modell gespeichert: {model_path}")
                                success = True
            except Exception as e:
                self.logger.error(f"Fehler beim Training des Marktregime-Detektors: {e}")

        # Asset-Clustering trainieren
        if self.asset_analyzer:
            try:
                self.logger.info("Trainiere Asset-Cluster-Analyzer...")
                # Daten laden
                if data_manager and self.asset_analyzer.load_market_data(
                        symbols=symbols,
                        data_manager=data_manager,
                        timeframe=self.settings.get('timeframes.analysis', '1d')
                ):
                    # Korrelationsmatrix berechnen
                    self.asset_analyzer.calculate_correlation_matrix()

                    # Features extrahieren
                    self.asset_analyzer.extract_asset_features()

                    # Clustering durchführen
                    clusters = self.asset_analyzer.run_clustering()

                    if not clusters.empty:
                        self.logger.info(
                            f"Asset-Clustering erfolgreich: {len(clusters)} Assets in {len(clusters['cluster'].unique())} Clustern")
                        success = success and True
            except Exception as e:
                self.logger.error(f"Fehler beim Training des Asset-Cluster-Analyzers: {e}")

        return success