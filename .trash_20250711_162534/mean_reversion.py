#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mean Reversion Trading-Strategie.
Implementiert eine Strategie, die auf Rückkehr zum Mittelwert setzt.
"""

import logging
import pandas as pd
import numpy as np
import talib
from typing import Dict, Any, Optional, Tuple, List

from config.settings import Settings
from core.position import Position
from strategies.strategy_base import Strategy


class MeanReversionStrategy(Strategy):
    """Mean Reversion Trading-Strategie"""

    def __init__(self, settings: Settings):
        """
        Initialisiert die Mean Reversion Strategie.

        Args:
            settings: Bot-Konfiguration
        """
        super().__init__(settings)
        self.name = "mean_reversion"

        # Strategie-Parameter aus Konfiguration laden
        self.rsi_period = settings.get('technical.rsi.period', 14)
        self.rsi_oversold = settings.get('technical.rsi.oversold', 30)
        self.rsi_overbought = settings.get('technical.rsi.overbought', 70)

        self.bollinger_period = settings.get('technical.bollinger.period', 20)
        self.bollinger_std = settings.get('technical.bollinger.std_dev', 2)

        self.atr_period = 14
        self.ema_period = 50

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bereitet Daten für die Analyse vor.
        Berechnet alle benötigten Indikatoren.

        Args:
            df: DataFrame mit OHLCV-Daten

        Returns:
            DataFrame mit berechneten Indikatoren
        """
        if len(df) < self.bollinger_period:
            return df

        # Indikatoren berechnen
        df = self.calculate_indicators(df)

        # Zusätzliche abgeleitete Signale berechnen

        # Bollinger Band-Signale
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # Bollinger Band-Position (0 bis 1, wo sich der Preis relativ zu den Bändern befindet)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # RSI-Extreme erkennen
        df['rsi_extreme'] = np.where(
            df['rsi'] < self.rsi_oversold,
            -1,  # Überverkauft
            np.where(
                df['rsi'] > self.rsi_overbought,
                1,  # Überkauft
                0  # Neutral
            )
        )

        # Preisabweichung vom Mittelwert
        df['price_deviation'] = (df['close'] - df['ema']) / df['ema'] * 100

        # Bollinger Band-Berührungen oder Durchbrüche
        df['bb_touch_lower'] = np.where(df['low'] <= df['bb_lower'], 1, 0)
        df['bb_touch_upper'] = np.where(df['high'] >= df['bb_upper'], 1, 0)

        # Rückkehrsignale zum Mittelwert
        df['mean_reversion_signal'] = np.where(
            (df['bb_touch_lower'].shift(1) == 1) & (df['close'] > df['low'].shift(1)),
            1,  # Kaufsignal nach unterer Band-Berührung
            np.where(
                (df['bb_touch_upper'].shift(1) == 1) & (df['close'] < df['high'].shift(1)),
                -1,  # Verkaufssignal nach oberer Band-Berührung
                0  # Kein Signal
            )
        )

        # Volatilität relativ zum historischen Durchschnitt
        df['volatility_ratio'] = df['atr'] / df['atr'].rolling(window=50).mean()

        return df

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Berechnet technische Indikatoren für die Analyse.

        Args:
            df: DataFrame mit OHLCV-Daten

        Returns:
            DataFrame mit hinzugefügten Indikatoren
        """
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'],
            timeperiod=self.bollinger_period,
            nbdevup=self.bollinger_std,
            nbdevdn=self.bollinger_std,
            matype=0
        )

        # RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=self.rsi_period)

        # Exponential Moving Average (für Trendrichtung)
        df['ema'] = talib.EMA(df['close'], timeperiod=self.ema_period)

        # Average True Range (für Volatilität)
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=self.atr_period)

        # Stochastic Oscillator
        df['stoch_k'], df['stoch_d'] = talib.STOCH(
            df['high'],
            df['low'],
            df['close'],
            fastk_period=14,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0
        )

        # MACD für Trendbestätigung
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'],
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )

        # Commodity Channel Index
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)

        return df

    def generate_signal(self, df: pd.DataFrame, symbol: str,
                        current_position: Optional[Position] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generiert ein Trading-Signal basierend auf den Daten.

        Args:
            df: DataFrame mit OHLCV-Daten
            symbol: Handelssymbol
            current_position: Aktuelle Position (oder None)

        Returns:
            Tuple aus Signal (BUY, SELL, HOLD) und zusätzlichen Signaldaten
        """
        # Sicherstellen, dass wir genügend Daten haben
        if len(df) < self.bollinger_period + 10:
            return "HOLD", {"signal": "HOLD", "reason": "insufficient_data", "confidence": 0.0}

        # Daten vorbereiten
        df = self.prepare_data(df.copy())

        # Letzten Datenpunkt für die Analyse verwenden
        current = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else current

        # Signal-Bewertung initialisieren
        long_signals = 0
        short_signals = 0
        signal_count = 0
        confidence = 0.0
        reason = "no_signal"

        # 1. Bollinger Band-Position
        if current['bb_position'] < 0.05:  # Nahe am unteren Band
            long_signals += 1.5
            reason = "bb_lower_touch"
        elif current['bb_position'] > 0.95:  # Nahe am oberen Band
            short_signals += 1.5
            reason = "bb_upper_touch"
        signal_count += 1.5

        # 2. RSI
        if current['rsi'] < self.rsi_oversold:
            long_signals += 1
            reason = "rsi_oversold"
        elif current['rsi'] > self.rsi_overbought:
            short_signals += 1
            reason = "rsi_overbought"
        signal_count += 1

        # 3. Mean Reversion Signal
        if current['mean_reversion_signal'] == 1:
            long_signals += 2
            reason = "mean_reversion_buy"
        elif current['mean_reversion_signal'] == -1:
            short_signals += 2
            reason = "mean_reversion_sell"
        signal_count += 2

        # 4. CCI (Extreme Werte)
        if current['cci'] < -100:
            long_signals += 1
        elif current['cci'] > 100:
            short_signals += 1
        signal_count += 1

        # 5. Stochastic Oscillator
        if current['stoch_k'] < 20 and current['stoch_k'] > current['stoch_d']:
            long_signals += 1
        elif current['stoch_k'] > 80 and current['stoch_k'] < current['stoch_d']:
            short_signals += 1
        signal_count += 1

        # 6. Preisabweichung vom Mittelwert
        if current['price_deviation'] < -5:  # Preis > 5% unter dem Mittelwert
            long_signals += 1
        elif current['price_deviation'] > 5:  # Preis > 5% über dem Mittelwert
            short_signals += 1
        signal_count += 1

        # Konfidenz berechnen (0.0 - 1.0)
        bull_confidence = long_signals / signal_count if signal_count > 0 else 0
        bear_confidence = short_signals / signal_count if signal_count > 0 else 0

        # Signal-Entscheidung
        signal = "HOLD"

        # Schwellenwert für Signalstärke
        threshold = 0.6

        # Aktuelle Position berücksichtigen
        if current_position is None:
            # Keine offene Position: Nur Kaufsignale betrachten
            if bull_confidence > threshold and bull_confidence > bear_confidence:
                signal = "BUY"
                confidence = bull_confidence
        else:
            # Offene Position: Nur Verkaufssignale betrachten
            if bear_confidence > threshold and bear_confidence > bull_confidence:
                signal = "SELL"
                confidence = bear_confidence

            # Take-Profit bei Rückkehr zum Mittelwert
            if current_position.side == "buy" and current['close'] >= current['bb_middle']:
                signal = "SELL"
                confidence = 0.8
                reason = "mean_reversion_target_reached"

        # Zusätzliche Signal-Daten
        signal_data = {
            "signal": signal,
            "confidence": confidence,
            "reason": reason,
            "indicators": {
                "bb_position": current['bb_position'],
                "rsi": current['rsi'],
                "cci": current['cci'],
                "price_deviation": current['price_deviation'],
                "volatility_ratio": current['volatility_ratio']
            },
            # Engere Stop-Loss und Take-Profit für Mean Reversion
            "use_trailing_stop": True,
            "trailing_stop_pct": 0.015,  # Engerer Trailing-Stop
            "trailing_activation_pct": 0.02  # Frühere Aktivierung
        }

        return signal, signal_data

    def optimize(self, df: pd.DataFrame, symbol: str,
                 param_grid: Optional[Dict[str, List[Any]]] = None) -> Dict[str, Any]:
        """
        Optimiert Strategie-Parameter basierend auf historischen Daten.

        Args:
            df: DataFrame mit OHLCV-Daten
            symbol: Handelssymbol
            param_grid: Dictionary mit Parametern und möglichen Werten

        Returns:
            Dictionary mit optimierten Parametern
        """
        if param_grid is None:
            # Standard-Parametergrid für Mean Reversion
            param_grid = {
                'rsi_period': [7, 14, 21],
                'rsi_oversold': [20, 25, 30],
                'rsi_overbought': [70, 75, 80],
                'bollinger_period': [10, 20, 30],
                'bollinger_std': [1.5, 2.0, 2.5]
            }

        self.logger.info(f"Optimizing mean reversion parameters for {symbol}...")

        best_params = {}
        best_result = -float('inf')
        total_combinations = 1

        for values in param_grid.values():
            total_combinations *= len(values)

        self.logger.info(f"Testing {total_combinations} parameter combinations")

        # Grid Search
        from itertools import product

        # Alle Parameterkombinationen aufbauen
        keys = list(param_grid.keys())
        values = list(param_grid.values())

        # Fortschritt speichern
        progress = 0

        # Jede Kombination testen
        for combination in product(*values):
            progress += 1
            if progress % 10 == 0:
                self.logger.debug(f"Optimization progress: {progress}/{total_combinations}")

            # Parameter zuweisen
            params = dict(zip(keys, combination))

            # Strategie mit aktuellen Parametern konfigurieren
            temp_strategy = MeanReversionStrategy(self.settings)

            for param, value in params.items():
                setattr(temp_strategy, param, value)

            # Strategie evaluieren
            result = temp_strategy.evaluate(df, symbol)

            # Bewertungsmetrik (win_rate und profit kombinieren)
            score = result['total_profit'] * (result['win_rate'] / 100)

            # Bessere Parameter gefunden?
            if score > best_result:
                best_result = score
                best_params = params

                self.logger.debug(
                    f"New best parameters: {params}, "
                    f"Score: {score:.2f}, "
                    f"Total Profit: {result['total_profit']:.2f}%, "
                    f"Win Rate: {result['win_rate']:.2f}%, "
                    f"Trades: {result['total_trades']}"
                )

        # Optimierungsergebnisse zurückgeben
        final_result = {
            'best_params': best_params,
            'score': best_result
        }

        self.logger.info(f"Optimization completed. Best parameters: {best_params}")

        return final_result