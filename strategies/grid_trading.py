"""Grid Trading Strategy"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from .strategy_base import Strategy, Signal

class GridTradingStrategy(Strategy):
    """Automated Grid Trading"""

    def __init__(self, params: Dict = None):
        super().__init__(params)
        self.grid_levels = params.get('grid_levels', 10)
        self.grid_spacing = params.get('grid_spacing', 0.01)
        self.grids = {}

    def calculate_signal(self, symbol: str, data: pd.DataFrame,
                        current_price: float) -> Tuple[Signal, float]:
        if symbol not in self.grids:
            self._initialize_grid(symbol, current_price)

        # Check grid levels
        for level in self.grids[symbol]['buy_levels']:
            if current_price <= level:
                return Signal.BUY, 0.8

        for level in self.grids[symbol]['sell_levels']:
            if current_price >= level:
                return Signal.SELL, 0.8

        return Signal.HOLD, 0.5

    def _initialize_grid(self, symbol: str, base_price: float):
        self.grids[symbol] = {
            'buy_levels': [base_price * (1 - self.grid_spacing * i)
                          for i in range(1, self.grid_levels//2 + 1)],
            'sell_levels': [base_price * (1 + self.grid_spacing * i)
                           for i in range(1, self.grid_levels//2 + 1)]
        }
