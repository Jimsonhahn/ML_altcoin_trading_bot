"""Momentum Trading Strategy"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from .strategy_base import Strategy, Signal
import logging

logger = logging.getLogger(__name__)

class MomentumStrategy(Strategy):
    """Trend-following Momentum Strategy"""

    def __init__(self, params=None):
        super().__init__(params)
        self.rsi_oversold = params.get('rsi_oversold', 30) if params else 30
        self.rsi_overbought = params.get('rsi_overbought', 70) if params else 70

    async def calculate_signal(self, symbol: str, data: pd.DataFrame,
                        current_price: float) -> Tuple[str, Dict[str, Any]]:
        """Calculate momentum-based trading signal"""
        if data is None or len(data) < 50:
            return 'HOLD', {'confidence': 0.0, 'reason': 'insufficient_data'}

        try:
            # Calculate indicators
            rsi = self._calculate_rsi(data['close'])
            sma_20 = data['close'].rolling(20).mean().iloc[-1]
            sma_50 = data['close'].rolling(50).mean().iloc[-1]

            # Volume check
            volume_avg = data['volume'].rolling(20).mean().iloc[-1]
            current_volume = data['volume'].iloc[-1]
            volume_spike = current_volume > volume_avg * 1.5

            # Momentum signals
            confidence = 0.0
            signal = 'HOLD'
            reason = 'no_signal'

            if rsi < self.rsi_oversold and current_price > sma_20 and volume_spike:
                signal = 'BUY'
                confidence = 0.8
                reason = 'oversold_with_volume'
            elif rsi > self.rsi_overbought and current_price < sma_20:
                signal = 'SELL'
                confidence = 0.8
                reason = 'overbought'
            elif current_price > sma_20 > sma_50:
                signal = 'BUY'
                confidence = 0.6
                reason = 'uptrend'
            elif current_price < sma_20 < sma_50:
                signal = 'SELL'
                confidence = 0.6
                reason = 'downtrend'

            return signal, {
                'confidence': confidence,
                'signal': signal,
                'reason': reason,
                'indicators': {
                    'rsi': float(rsi),
                    'sma_20': float(sma_20),
                    'sma_50': float(sma_50),
                    'volume_spike': volume_spike
                }
            }

        except Exception as e:
            logger.error(f"Error in momentum calculation: {e}")
            return 'HOLD', {'confidence': 0.0, 'error': str(e)}

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            if loss.iloc[-1] == 0:
                return 100.0

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]
        except:
            return 50.0
