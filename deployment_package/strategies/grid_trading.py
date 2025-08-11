"""Grid Trading Strategy"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Any
from .strategy_base import Strategy

class GridTradingStrategy(Strategy):
    """Automated Grid Trading"""

    def __init__(self, params: Dict = None):
        super().__init__(params)
        self.grid_levels = params.get('grid_levels', 10) if params else 10
        self.grid_spacing = params.get('grid_spacing', 0.01) if params else 0.01
        self.grids = {}

    def calculate_signal(self, symbol: str, data: pd.DataFrame,
                        current_price: float) -> Tuple[str, Dict[str, Any]]:
        """Calculate grid trading signal"""
        if symbol not in self.grids:
            self._initialize_grid(symbol, current_price)

        # Check grid levels
        for level in self.grids[symbol]['buy_levels']:
            if current_price <= level:
                return 'BUY', {
                    'confidence': 0.8,
                    'signal': 'BUY',
                    'reason': 'grid_buy_level',
                    'grid_level': level
                }

        for level in self.grids[symbol]['sell_levels']:
            if current_price >= level:
                return 'SELL', {
                    'confidence': 0.8,
                    'signal': 'SELL',
                    'reason': 'grid_sell_level',
                    'grid_level': level
                }

        return 'HOLD', {
            'confidence': 0.5,
            'signal': 'HOLD',
            'reason': 'between_grid_levels'
        }

    def _initialize_grid(self, symbol: str, base_price: float):
        """Initialize grid levels"""
        self.grids[symbol] = {
            'buy_levels': [base_price * (1 - self.grid_spacing * i)
                          for i in range(1, self.grid_levels//2 + 1)],
            'sell_levels': [base_price * (1 + self.grid_spacing * i)
                           for i in range(1, self.grid_levels//2 + 1)]
        }
