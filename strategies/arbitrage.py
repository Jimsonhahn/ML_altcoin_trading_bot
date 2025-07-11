"""Arbitrage Trading Strategy"""
import pandas as pd
import numpy as np
from typing import Tuple
from .strategy_base import Strategy, Signal

class ArbitrageStrategy(Strategy):
    """Cross-exchange arbitrage detection"""

    def calculate_signal(self, symbol: str, data: pd.DataFrame,
                        current_price: float) -> Tuple[Signal, float]:
        # Simulate price differences
        price_diff = np.random.uniform(-0.005, 0.005)

        if price_diff > 0.002:
            return Signal.BUY, 0.9
        elif price_diff < -0.002:
            return Signal.SELL, 0.9

        return Signal.HOLD, 0.3
