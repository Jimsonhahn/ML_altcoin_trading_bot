#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Machine Learning basierte Trading-Strategie.
Implementiert eine Strategie, die auf ML-Vorhersagen basiert.
"""

import logging
import os
import pandas as pd
import numpy as np
import talib
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import joblib

from config.settings import Settings
from core.position import Position
from strategies.strategy_base import Strategy

# Sklearn-Imports
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class MLStrategy(Strategy):
    """Machine Learning basierte Trading-Strategie"""

    def __init__(self, settings: Settings):
        """
        Initialisiert die ML-Strategie.

        Args:
            settings: Bot-Konfiguration
        """
        super().__init__(settings)
        self.name = "ml"

        # Prüfen, ob sklearn verfügbar ist
        if not SKLEARN_AVAILABLE:
            self.logger.warning("scikit-learn is not available. Using fallback strategy.")

        # Strategie-Parameter aus Konfiguration laden
        self.features = settings.get('machine_learning.features', [
            'trend', 'rsi', 'ma_short', 'ma_long', 'macd', 'macd_signal',
            'stochastic_k', 'stochastic_d', 'upper_band', 'lower_band',
            'volume', 'price_change_24h'
        ])

        self.confidence_threshold = settings.get('machine_learning.confidence_threshold', 0.7)

        # Technische Indikatoren-Parameter
        self.rsi_period = settings.get('technical.rsi.period', 14)
        self.ma_short = settings.get('technical.ma.short', 20)
        self.ma_long = settings.get('technical.ma.long', 50)
        self.macd_fast = settings.get('technical.macd.fast', 12)
        self.macd_slow = settings.get('technical.macd.slow', 26)
        self.macd_signal = settings.get('technical.macd.signal', 9)

        # ML-Modell
        self.model = None
        self.scaler = None
        self.is_trained = False

        # Modellpfade
        self.model_dir = 'models'

        # Versuchen, ein vortrainiertes Modell zu laden
        self._load_model()

        # Training History
        self.training_history = []

    def _get_model_path(self, symbol: str) -> str:
        """
        Generiert den Pfad zum Modell für ein Symbol.

        Args:
            symbol: Handelssymbol

        Returns:
            Pfad zum Modell
        """
        # Bereinigen des Symbols für den Dateinamen
        clean_symbol = symbol.replace('/', '_')
        return os.path.join(self.model_dir, f"ml_model_{clean_symbol}.joblib")

    def _load_model(self, symbol: str = 'general') -> bool:
        """
        Lädt ein vortrainiertes Modell.

        Args:
            symbol: Handelssymbol

        Returns:
            True, wenn das Modell erfolgreich geladen wurde, False sonst
        """
        if not SKLEARN_AVAILABLE:
            return False

        model_path = self._get_model_path(symbol)
        scaler_path = os.path.join(self.model_dir, f"scaler_{symbol}.joblib")

        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.is_trained = True
                self.logger.info(f"Loaded ML model for {symbol}")
                return True
            else:
                self.logger.info(f"No existing ML model found for {symbol}")
                return False
        except Exception as e:
            self.logger.error(f"Error loading ML model: {e}")
            return False

    def _save_model(self, symbol: str = 'general') -> None:
        """
        Speichert das trainierte Modell.

        Args:
            symbol: Handelssymbol
        """
        if not SKLEARN_AVAILABLE or not self.is_trained:
            return

        # Verzeichnis erstellen, falls es nicht existiert
        os.makedirs(self.model_dir, exist_ok=True)

        model_path = self._get_model_path(symbol)
        scaler_path = os.path.join(self.model_dir, f"scaler_{symbol}.joblib")

        try:
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            self.logger.info(f"Saved ML model for {symbol}")
        except Exception as e:
            self.logger.error(f"Error saving ML model: {e}")

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bereitet Daten für die Analyse vor.
        Berechnet alle benötigten Indikatoren.

        Args:
            df: DataFrame mit OHLCV-Daten

        Returns:
            DataFrame mit berechneten Indikatoren
        """
        if len(df) < self.ma_long:
            return df

        # Indikatoren berechnen
        df = self.calculate_indicators(df)

        # Zusätzliche Features für ML

        # Preisänderungen berechnen
        for period in [1, 3, 5, 10]:
            df[f'return_{period}d'] = df['close'].pct_change(period) * 100

        # Volatilität
        df['volatility'] = df['close'].rolling(window=20).std() / df['close'] * 100

        # Volumen-Features
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # Trendrichtung
        df['trend'] = np.where(
            df['ma_short'] > df['ma_long'],
            1,  # Aufwärtstrend
            np.where(
                df['ma_short'] < df['ma_long'],
                -1,  # Abwärtstrend
                0  # Seitwärtstrend
            )
        )

        # Zukünftige Preisänderung für Training (nicht für Vorhersage)
        df['future_return_1d'] = df['close'].pct_change(1).shift(-1) * 100
        df['target'] = np.where(df['future_return_1d'] > 1.0, 1, 0)  # 1% als Schwellenwert

        return df

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Berechnet technische Indikatoren für die Analyse.

        Args:
            df: DataFrame mit OHLCV-Daten

        Returns:
            DataFrame mit hinzugefügten Indikatoren
        """
        # Moving Averages
        df['ma_short'] = talib.SMA(df['close'], timeperiod=self.ma_short)
        df['ma_long'] = talib.SMA(df['close'], timeperiod=self.ma_long)

        # Exponential Moving Averages
        df['ema_short'] = talib.EMA(df['close'], timeperiod=self.ma_short)
        df['ema_long'] = talib.EMA(df['close'], timeperiod=self.ma_long)

        # RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=self.rsi_period)

        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'],
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )

        # Bollinger Bands
        df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(
            df['close'],
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2,
            matype=0
        )

        # Stochastic Oscillator
        df['stochastic_k'], df['stochastic_d'] = talib.STOCH(
            df['high'],
            df['low'],
            df['close'],
            fastk_period=14,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0
        )

        # Average True Range (Volatilität)
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

        # On-Balance Volume
        df['obv'] = talib.OBV(df['close'], df['volume'])

        return df

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Bereitet Features für das ML-Modell vor.

        Args:
            df: DataFrame mit berechneten Indikatoren

        Returns:
            Tuple aus Feature-Matrix und optionalem Target-Vektor
        """
        # NaN-Werte entfernen oder ersetzen
        df = df.copy().dropna()

        if df.empty:
            return np.array([]), None

        # Features auswählen
        X = df[self.features].values

        # Target, falls verfügbar
        y = None
        if 'target' in df.columns:
            y = df['target'].values

        return X, y

    def train(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Trainiert das ML-Modell mit historischen Daten.

        Args:
            df: DataFrame mit OHLCV-Daten
            symbol: Handelssymbol

        Returns:
            Dictionary mit Trainingsergebnissen
        """
        if not SKLEARN_AVAILABLE:
            self.logger.warning("scikit-learn is not available. Cannot train ML model.")
            return {'success': False, 'error': 'scikit-learn not available'}

        try:
            # Daten vorbereiten
            df = self.prepare_data(df.copy())

            # Features und Target extrahieren
            X, y = self._prepare_features(df)

            if len(X) < 100 or y is None:
                self.logger.warning(f"Insufficient data for training: {len(X)} samples")
                return {'success': False, 'error': 'insufficient_data'}

            # Daten skalieren
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # Modell initialisieren
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )

            # Modell trainieren
            self.model.fit(X_scaled, y)

            # Modell speichern
            self._save_model(symbol)

            self.is_trained = True

            # Feature-Importance
            feature_importance = list(zip(self.features, self.model.feature_importances_))
            feature_importance.sort(key=lambda x: x[1], reverse=True)

            # Trainingsergebnisse
            result = {
                'success': True,
                'samples': len(X),
                'training_date': datetime.now().isoformat(),
                'feature_importance': feature_importance,
                'symbol': symbol
            }

            # Trainingsverlauf aktualisieren
            self.training_history.append(result)

            self.logger.info(f"Successfully trained ML model for {symbol} with {len(X)} samples")

            return result

        except Exception as e:
            self.logger.error(f"Error training ML model: {e}")
            return {'success': False, 'error': str(e)}

    def predict(self, df: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
        """
        Generiert eine Vorhersage mit dem ML-Modell.

        Args:
            df: DataFrame mit berechneten Indikatoren

        Returns:
            Tuple aus Wahrscheinlichkeit und zusätzlichen Informationen
        """
        if not SKLEARN_AVAILABLE or not self.is_trained or self.model is None:
            # Fallback: Neutrale Vorhersage
            return 0.5, {'error': 'model_not_available'}

        try:
            # Features extrahieren
            X, _ = self._prepare_features(df)

            if len(X) == 0:
                return 0.5, {'error': 'no_features'}

            # Letzte Zeile verwenden (aktuelle Daten)
            X_current = X[-1].reshape(1, -1)

            # Daten skalieren
            X_scaled = self.scaler.transform(X_current)

            # Vorhersage generieren
            probas = self.model.predict_proba(X_scaled)[0]

            # Wahrscheinlichkeit für positives Ergebnis
            buy_probability = probas[1]

            # Featuregewichtungen für die Erklärbarkeit
            prediction_info = {}

            if hasattr(self.model, 'feature_importances_'):
                feature_importance = list(zip(self.features, self.model.feature_importances_))
                prediction_info['feature_importance'] = feature_importance

            return buy_probability, prediction_info

        except Exception as e:
            self.logger.error(f"Error generating prediction: {e}")
            return 0.5, {'error': str(e)}

    def generate_signal(self, df: pd.DataFrame, symbol: str,
                        current_position: Optional[Position] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generiert ein Trading-Signal basierend auf ML-Vorhersagen und technischer Analyse.

        Args:
            df: DataFrame mit OHLCV-Daten
            symbol: Handelssymbol
            current_position: Aktuelle Position (oder None)

        Returns:
            Tuple aus Signal (BUY, SELL, HOLD) und zusätzlichen Signaldaten
        """
        # Sicherstellen, dass wir genügend Daten haben
        if len(df) < self.ma_long + 10:
            return "HOLD", {"signal": "HOLD", "reason": "insufficient_data", "confidence": 0.0}

        # Daten vorbereiten
        df = self.prepare_data(df.copy())

        # Letzten Datenpunkt für die Analyse verwenden
        current = df.iloc[-1]

        # 1. ML-Modell-Vorhersage
        if SKLEARN_AVAILABLE and self.is_trained:
            # Modell für das Symbol laden, falls nicht bereits geladen
            if not self._load_model(symbol):
                # Fallback: Generelles Modell laden
                self._load_model('general')

            # Vorhersage generieren
            buy_probability, prediction_info = self.predict(df)

            # ML-Confidence
            ml_confidence = max(buy_probability, 1 - buy_probability)
        else:
            # Fallback: Technische Analyse, wenn ML nicht verfügbar
            buy_probability = 0.5
            ml_confidence = 0.0
            prediction_info = {'error': 'model_not_available'}

        # 2. Technische Analyse zur Bestätigung

        # Trendbestimmung
        trend = current['trend']

        # RSI-Bedingungen
        rsi_oversold = current['rsi'] < 30
        rsi_overbought = current['rsi'] > 70

        # MACD-Signale
        macd_bullish = current['macd'] > current['macd_signal']

        # Bollinger Bands
        bb_lower_touch = current['close'] <= current['lower_band']
        bb_upper_touch = current['close'] >= current['upper_band']

        # Kombination von ML und technischer Analyse
        signal = "HOLD"
        reason = "neutral"
        confidence = 0.0

        # Verkaufssignal nur prüfen, wenn wir eine Position haben
        if current_position is not None:
            # Verkaufskriterien mit ML-Bestätigung
            if buy_probability < 0.3 and (rsi_overbought or bb_upper_touch or trend == -1):
                signal = "SELL"
                reason = "ml_sell_confirmation"
                confidence = (1 - buy_probability) * 0.7 + 0.3  # Gewichtete Konfidenz

            # Starkes Verkaufssignal vom ML-Modell
            elif buy_probability < 0.2:
                signal = "SELL"
                reason = "strong_ml_sell"
                confidence = (1 - buy_probability)

        # Kaufsignal nur prüfen, wenn wir keine Position haben
        else:
            # Kaufkriterien mit ML-Bestätigung
            if buy_probability > 0.7 and (rsi_oversold or bb_lower_touch or trend == 1):
                signal = "BUY"
                reason = "ml_buy_confirmation"
                confidence = buy_probability * 0.7 + 0.3  # Gewichtete Konfidenz

            # Starkes Kaufsignal vom ML-Modell
            elif buy_probability > 0.8:
                signal = "BUY"
                reason = "strong_ml_buy"
                confidence = buy_probability

        # Zusätzliche Signaldaten
        signal_data = {
            "signal": signal,
            "confidence": confidence,
            "reason": reason,
            "ml_prediction": buy_probability,
            "ml_confidence": ml_confidence,
            "technical_indicators": {
                "trend": trend,
                "rsi": current['rsi'],
                "macd_bullish": macd_bullish,
                "bb_position": (current['close'] - current['lower_band']) / (
                            current['upper_band'] - current['lower_band']),
                "volume_ratio": current.get('volume_ratio', 1.0)
            },
            "prediction_info": prediction_info,
            # Bei ML-Signalen enger Stop-Loss verwenden
            "use_trailing_stop": True,
            "trailing_stop_pct": 0.02,
            "trailing_activation_pct": 0.03
        }

        return signal, signal_data

    def optimize(self, df: pd.DataFrame, symbol: str,
                 param_grid: Optional[Dict[str, List[Any]]] = None) -> Dict[str, Any]:
        """
        Optimiert ML-Parameter und trainiert ein neues Modell.

        Args:
            df: DataFrame mit OHLCV-Daten
            symbol: Handelssymbol
            param_grid: Dictionary mit Parametern und möglichen Werten

        Returns:
            Dictionary mit optimierten Parametern
        """
        if not SKLEARN_AVAILABLE:
            self.logger.warning("scikit-learn is not available. Cannot optimize parameters.")
            return {'success': False, 'error': 'scikit-learn not available'}

        from sklearn.model_selection import GridSearchCV

        try:
            # Daten vorbereiten
            df = self.prepare_data(df.copy())

            # Features und Target extrahieren
            X, y = self._prepare_features(df)

            if len(X) < 100 or y is None:
                self.logger.warning(f"Insufficient data for optimization: {len(X)} samples")
                return {'success': False, 'error': 'insufficient_data'}

            # Daten skalieren
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # Standardparameter, falls keine angegeben
            if param_grid is None:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }

            # Grid Search mit Cross-Validation
            model = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1
            )

            grid_search.fit(X_scaled, y)

            # Bestes Modell
            self.model = grid_search.best_estimator_

            # Modell speichern
            self._save_model(symbol)

            self.is_trained = True

            # Optimierungsergebnisse
            result = {
                'success': True,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'samples': len(X),
                'optimization_date': datetime.now().isoformat(),
                'symbol': symbol
            }

            self.logger.info(f"Successfully optimized ML model for {symbol} with score {grid_search.best_score_:.4f}")

            return result

        except Exception as e:
            self.logger.error(f"Error optimizing ML parameters: {e}")
            return {'success': False, 'error': str(e)}