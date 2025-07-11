"""Copy Trading Strategy"""
import pandas as pd
import numpy as np
from typing import Tuple
from .strategy_base import Strategy, Signal

class CopyTradingStrategy(Strategy):
    """Follow whale movements"""

    def calculate_signal(self, symbol: str, data: pd.DataFrame,
                        current_price: float) -> Tuple[Signal, float]:
        # Simulate whale activity
        whale_buying = np.random.random() > 0.8
        whale_selling = np.random.random() > 0.8

        if whale_buying:
            return Signal.BUY, 0.9
        elif whale_selling:
            return Signal.SELL, 0.9

        return Signal.HOLD, 0.3
