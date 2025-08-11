"""Liquidation Hunter Strategy"""
import pandas as pd
from typing import Tuple, Dict, Any
from .strategy_base import Strategy

class LiquidationStrategy(Strategy):
    """Hunt liquidation levels"""

    def calculate_signal(self, symbol: str, data: pd.DataFrame,
                        current_price: float) -> Tuple[str, Dict[str, Any]]:
        """Calculate liquidation hunting signal"""
        if len(data) < 24:
            return 'HOLD', {'confidence': 0.0, 'reason': 'insufficient_data'}

        recent_high = data['high'].rolling(24).max().iloc[-1]
        recent_low = data['low'].rolling(24).min().iloc[-1]

        distance_to_low = (current_price - recent_low) / current_price
        distance_to_high = (recent_high - current_price) / current_price

        signal = 'HOLD'
        confidence = 0.4
        reason = 'no_liquidation_zone'

        if distance_to_low < 0.02:
            signal = 'BUY'
            confidence = 0.85
            reason = 'near_liquidation_low'
        elif distance_to_high < 0.02:
            signal = 'SELL'
            confidence = 0.85
            reason = 'near_liquidation_high'

        return signal, {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'recent_high': float(recent_high),
            'recent_low': float(recent_low)
        }
