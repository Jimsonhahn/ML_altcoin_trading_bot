"""Copy Trading Strategy"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from .strategy_base import Strategy

class CopyTradingStrategy(Strategy):
    """Follow whale movements"""

    def calculate_signal(self, symbol: str, data: pd.DataFrame,
                        current_price: float) -> Tuple[str, Dict[str, Any]]:
        """Calculate copy trading signal"""
        # Simulate whale activity (in production, monitor actual whale wallets)
        whale_buying = np.random.random() > 0.8
        whale_selling = np.random.random() > 0.8

        signal = 'HOLD'
        confidence = 0.3
        reason = 'no_whale_activity'

        if whale_buying:
            signal = 'BUY'
            confidence = 0.9
            reason = 'whale_buying'
        elif whale_selling:
            signal = 'SELL'
            confidence = 0.9
            reason = 'whale_selling'

        return signal, {
            'signal': signal,
            'confidence': confidence,
            'reason': reason
        }
