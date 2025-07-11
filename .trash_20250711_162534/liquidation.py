"""Liquidation Hunter Strategy"""
import pandas as pd
from typing import Tuple
from .strategy_base import Strategy, Signal

class LiquidationStrategy(Strategy):
    """Hunt liquidation levels"""

    def calculate_signal(self, symbol: str, data: pd.DataFrame,
                        current_price: float) -> Tuple[Signal, float]:
        if len(data) < 24:
            return Signal.HOLD, 0.0

        recent_high = data['high'].rolling(24).max().iloc[-1]
        recent_low = data['low'].rolling(24).min().iloc[-1]

        distance_to_low = (current_price - recent_low) / current_price
        distance_to_high = (recent_high - current_price) / current_price

        if distance_to_low < 0.02:
            return Signal.BUY, 0.85
        elif distance_to_high < 0.02:
            return Signal.SELL, 0.85

        return Signal.HOLD, 0.4
