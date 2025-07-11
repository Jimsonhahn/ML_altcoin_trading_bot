#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Momentum-basierte Trading-Strategie.
Implementiert eine Strategie, die auf Preismomentum und Trendkontinuität setzt.
"""

import logging
import pandas as pd
import numpy as np
import talib
from typing import Dict, Any, Optional, Tuple, List

from config.settings import Settings
from core.position import Position
from strategies.strategy_base import Strategy


class MomentumStrategy(Strategy):
    """Momentum-basierte Trading-Strategie"""

    def __init__(self, settings: Settings):
        """
        Initialisiert die Momentum-Strategie.

        Args:
            settings: Bot-Konfiguration
        """
        super().__init__(settings)
        self.name = "momentum"

        # Strategie-Parameter aus Konfiguration laden
        self.rsi_period = settings.get('technical.rsi.period', 14)
        self.rsi_oversold = settings.get('technical.rsi.oversold', 30)
        self.rsi_overbought = settings.get('technical.rsi.overbought', 70)

        self.ma_short = settings.get('technical.ma.short', 20)
        self.ma_long = settings.get('technical.ma.long', 50)

        self.macd_fast = settings.get('technical.macd.fast', 12)
        self.macd_slow = settings.get('technical.macd.slow', 26)
        self.macd_signal = settings.get('technical.macd.signal', 9)

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

        # Zusätzliche abgeleitete Signale berechnen

        # Trendrichtung basierend auf Moving Averages
        df['trend'] = np.where(
            df['ma_short'] > df['ma_long'],
            1,  # Aufwärtstrend
            np.where(
                df['ma_short'] < df['ma_long'],
                -1,  # Abwärtstrend
                0  # Seitwärtstrend
            )
        )

        # RSI-Signale
        df['rsi_signal'] = np.where(
            df['rsi'] < self.rsi_oversold,
            1,  # Überverkauft (Kaufsignal)
            np.where(
                df['rsi'] > self.rsi_overbought,
                -1,  # Überkauft (Verkaufssignal)
                0  # Neutral
            )
        )

        # MACD-Signale
        df['macd_signal_line'] = np.where(
            df['macd'] > df['macd_signal'],
            1,  # Bullish
            np.where(
                df['macd'] < df['macd_signal'],
                -1,  # Bearish
                0  # Neutral
            )
        )

        # MACD-Crossover erkennen
        df['macd_crossover'] = df['macd_signal_line'].diff().fillna(0)

        # Momentum berechnen (5-Perioden-Preisänderung in Prozent)
        df['momentum'] = df['close'].pct_change(5) * 100

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
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'],
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2,
            matype=0
        )

        # Average True Range (Volatilität)
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

        # Volumen-Oszillator
        df['volume_sma'] = talib.SMA(df['volume'], timeperiod=20)
        df['volume_ratio'] = df['volume'] / df['volume_sma']

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
        if len(df) < self.ma_long + 10:
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

        # 1. Trendrichtung prüfen
        if current['trend'] == 1:
            long_signals += 1.5  # Starker Aufwärtstrend (höhere Gewichtung)
        elif current['trend'] == -1:
            short_signals += 1.5  # Starker Abwärtstrend
        signal_count += 1.5

        # 2. RSI
        if current['rsi_signal'] == 1:
            long_signals += 1
        elif current['rsi_signal'] == -1:
            short_signals += 1
        signal_count += 1

        # 3. MACD
        if current['macd_signal_line'] == 1:
            long_signals += 1
        elif current['macd_signal_line'] == -1:
            short_signals += 1
        signal_count += 1

        # 4. MACD Crossover (stärkeres Signal)
        if current['macd_crossover'] == 2:  # Bullish Crossover
            long_signals += 2
            reason = "macd_bullish_crossover"
        elif current['macd_crossover'] == -2:  # Bearish Crossover
            short_signals += 2
            reason = "macd_bearish_crossover"
        signal_count += 2

        # 5. Preismomentum
        if current['momentum'] > 3:  # Starkes positives Momentum
            long_signals += 1
        elif current['momentum'] < -3:  # Starkes negatives Momentum
            short_signals += 1
        signal_count += 1

        # 6. Volumen-Bestätigung
        if current['volume_ratio'] > 1.5 and current['close'] > previous['close']:
            long_signals += 1
            reason = "volume_confirmation_bullish"
        elif current['volume_ratio'] > 1.5 and current['close'] < previous['close']:
            short_signals += 1
            reason = "volume_confirmation_bearish"
        signal_count += 1

        # Konfidenz berechnen (0.0 - 1.0)
        bull_confidence = long_signals / signal_count if signal_count > 0 else 0
        bear_confidence = short_signals / signal_count if signal_count > 0 else 0

        # Signal-Entscheidung
        signal = "HOLD"

        # Schwellenwert für Signalstärke (0.6 = 60% der Indikatoren sind bullish/bearish)
        threshold = 0.6

        # Aktuelle Position berücksichtigen
        if current_position is None:
            # Keine offene Position: Nur Kaufsignale betrachten
            if bull_confidence > threshold and bull_confidence > bear_confidence:
                signal = "BUY"
                confidence = bull_confidence
                if not reason or reason == "no_signal":
                    reason = "momentum_buy"
        else:
            # Offene Position: Nur Verkaufssignale betrachten
            if bear_confidence > threshold and bear_confidence > bull_confidence:
                signal = "SELL"
                confidence = bear_confidence
                if not reason or reason == "no_signal":
                    reason = "momentum_sell"

        # Zusätzliche Signal-Daten
        signal_data = {
            "signal": signal,
            "confidence": confidence,
            "reason": reason,
            "indicators": {
                "trend": current['trend'],
                "rsi": current['rsi'],
                "macd": current['macd'],
                "macd_signal": current['macd_signal'],
                "momentum": current['momentum'],
                "volume_ratio": current['volume_ratio']
            },
            "use_trailing_stop": True,
            "trailing_stop_pct": 0.02,
            "trailing_activation_pct": 0.03
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
            # Standard-Parametergrid
            param_grid = {
                'rsi_period': [7, 14, 21],
                'rsi_oversold': [20, 30, 40],
                'rsi_overbought': [60, 70, 80],
                'ma_short': [10, 20, 30],
                'ma_long': [50, 100, 200]
            }

        self.logger.info(f"Optimizing strategy parameters for {symbol}...")

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
            temp_strategy = MomentumStrategy(self.settings)

            for param, value in params.items():
                setattr(temp_strategy, param, value)

            # Strategie evaluieren
            result = temp_strategy.evaluate(df, symbol)

            # Bewertungsmetrik (kann angepasst werden)
            score = result['total_profit']

            # Zusätzliche Bewertung für Sharpe Ratio
            if result['sharpe_ratio'] > 0:
                score *= (1 + result['sharpe_ratio'])

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