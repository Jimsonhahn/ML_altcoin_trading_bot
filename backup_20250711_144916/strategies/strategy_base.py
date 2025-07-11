#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basisklasse für alle Trading-Strategien.
Definiert die gemeinsame Schnittstelle für alle Strategien.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np

from config.settings import Settings
from core.position import Position


class Strategy:
    """Basisklasse für alle Trading-Strategien"""

    def __init__(self, settings: Settings):
        """
        Initialisiert die Strategie.

        Args:
            settings: Bot-Konfiguration
        """
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.name = "base"

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bereitet Daten für die Analyse vor.
        Berechnet alle benötigten Indikatoren.

        Args:
            df: DataFrame mit OHLCV-Daten

        Returns:
            DataFrame mit berechneten Indikatoren
        """
        # Zu überschreiben von Unterklassen
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
        # Diese Methode muss von Unterklassen implementiert werden
        raise NotImplementedError("Subclasses must implement generate_signal()")

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Berechnet technische Indikatoren für die Analyse.

        Args:
            df: DataFrame mit OHLCV-Daten

        Returns:
            DataFrame mit hinzugefügten Indikatoren
        """
        # In Unterklassen zu überschreiben
        return df

    def optimize(self, df: pd.DataFrame, symbol: str,
                 param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Optimiert Strategie-Parameter basierend auf historischen Daten.

        Args:
            df: DataFrame mit OHLCV-Daten
            symbol: Handelssymbol
            param_grid: Dictionary mit Parametern und möglichen Werten

        Returns:
            Dictionary mit optimierten Parametern
        """
        # In Unterklassen zu implementieren, falls Parameteroptimierung unterstützt wird
        self.logger.warning("Parameter optimization not implemented for this strategy")
        return {}

    def evaluate(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Evaluiert die Strategie auf historischen Daten.

        Args:
            df: DataFrame mit OHLCV-Daten
            symbol: Handelssymbol

        Returns:
            Dictionary mit Evaluationsergebnissen
        """
        # Daten vorbereiten
        df = self.prepare_data(df.copy())

        trades = []
        position = None
        entry_price = 0
        entry_index = 0

        for i, row in df.iterrows():
            # Signal generieren
            signal, _ = self.generate_signal(df.loc[:i], symbol, position)

            # Signal verarbeiten
            if signal == 'BUY' and position is None:
                # Einstieg
                position = 'LONG'
                entry_price = row['close']
                entry_index = i
            elif signal == 'SELL' and position == 'LONG':
                # Ausstieg
                exit_price = row['close']
                profit_pct = (exit_price - entry_price) / entry_price * 100

                trades.append({
                    'entry_time': entry_index,
                    'exit_time': i,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit_pct': profit_pct
                })

                position = None

        # Offene Position am Ende schließen
        if position == 'LONG':
            exit_price = df.iloc[-1]['close']
            profit_pct = (exit_price - entry_price) / entry_price * 100

            trades.append({
                'entry_time': entry_index,
                'exit_time': df.index[-1],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'profit_pct': profit_pct
            })

        # Keine Trades
        if not trades:
            return {
                'total_trades': 0,
                'profitable_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'max_profit': 0,
                'max_loss': 0,
                'total_profit': 0,
                'sharpe_ratio': 0,
                'trades': []
            }

        # Statistiken berechnen
        profits = [trade['profit_pct'] for trade in trades]
        profitable_trades = sum(1 for p in profits if p > 0)

        return {
            'total_trades': len(trades),
            'profitable_trades': profitable_trades,
            'win_rate': profitable_trades / len(trades) * 100 if trades else 0,
            'avg_profit': np.mean(profits),
            'max_profit': max(profits),
            'max_loss': min(profits),
            'total_profit': sum(profits),
            'sharpe_ratio': np.mean(profits) / np.std(profits) if len(profits) > 1 and np.std(profits) > 0 else 0,
            'trades': trades
        }