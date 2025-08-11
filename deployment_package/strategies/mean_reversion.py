"""Mean Reversion Strategy"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
import logging
from .strategy_base import Strategy

logger = logging.getLogger(__name__)

class MeanReversionStrategy(Strategy):
    """Mean Reversion Trading Strategy"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "Mean Reversion"
        self.bb_period = config.get('bollinger_period', 20)
        self.bb_std = config.get('bollinger_std', 2.0)
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)

        logger.info(f"Mean Reversion Strategy initialized")

    def calculate_signal(self, symbol: str, data: pd.DataFrame,
                        current_price: float) -> Tuple[str, Dict[str, Any]]:
        """Calculate mean reversion trading signal"""
        if data is None or len(data) < self.bb_period:
            return 'HOLD', {'confidence': 0.0, 'reason': 'insufficient_data'}

        try:
            close_prices = data['close']

            # Bollinger Bands
            sma = close_prices.rolling(window=self.bb_period).mean()
            std = close_prices.rolling(window=self.bb_period).std()
            upper_band = sma + (self.bb_std * std)
            lower_band = sma - (self.bb_std * std)

            current_sma = float(sma.iloc[-1])
            current_upper = float(upper_band.iloc[-1])
            current_lower = float(lower_band.iloc[-1])

            # RSI
            rsi = self._calculate_rsi(close_prices, self.rsi_period)

            # Signals
            signal = 'HOLD'
            confidence = 0.0
            reason = 'no_signal'

            if current_price <= current_lower and rsi < self.rsi_oversold:
                signal = 'BUY'
                confidence = 0.8
                reason = 'oversold_at_lower_band'
            elif current_price >= current_upper and rsi > self.rsi_overbought:
                signal = 'SELL'
                confidence = 0.8
                reason = 'overbought_at_upper_band'

            return signal, {
                'signal': signal,
                'confidence': confidence,
                'reason': reason,
                'indicators': {
                    'current_price': current_price,
                    'upper_band': current_upper,
                    'lower_band': current_lower,
                    'sma': current_sma,
                    'rsi': float(rsi)
                }
            }

        except Exception as e:
            logger.error(f"Error in mean reversion calculation: {e}")
            return 'HOLD', {'confidence': 0.0, 'error': str(e)}

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        if loss.iloc[-1] == 0:
            return 100.0

        rs = gain.iloc[-1] / loss.iloc[-1]
        return 100 - (100 / (1 + rs))
