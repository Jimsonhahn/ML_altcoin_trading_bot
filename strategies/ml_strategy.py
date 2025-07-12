"""Machine Learning Strategy"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
import logging
from .strategy_base import Strategy

logger = logging.getLogger(__name__)

class MLStrategy(Strategy):
    """ML-based Trading Strategy"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.lookback_period = config.get('lookback_period', 50)
        self.prediction_threshold = config.get('prediction_threshold', 0.6)
        logger.info("ML Strategy initialized (simplified version)")

    def calculate_signal(self, symbol: str, data: pd.DataFrame,
                        current_price: float) -> Tuple[str, Dict[str, Any]]:
        """Calculate ML-based trading signal"""
        if len(data) < self.lookback_period:
            return 'HOLD', {'confidence': 0.0, 'reason': 'insufficient_data'}

        try:
            # Simple feature extraction
            features = self._extract_features(data)

            # Simplified prediction (no actual ML model)
            prediction = self._generate_prediction(features)

            # Generate signal
            signal = 'HOLD'
            confidence = abs(prediction - 0.5) * 2

            if prediction > self.prediction_threshold:
                signal = 'BUY'
            elif prediction < (1 - self.prediction_threshold):
                signal = 'SELL'

            return signal, {
                'signal': signal,
                'confidence': float(confidence),
                'prediction': float(prediction),
                'features': features
            }

        except Exception as e:
            logger.error(f"Error in ML calculation: {e}")
            return 'HOLD', {'confidence': 0.0, 'error': str(e)}

    def _extract_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract features for ML model"""
        close_prices = data['close']

        features = {
            'rsi': float(self._calculate_rsi(close_prices)),
            'price_change_1d': float((close_prices.iloc[-1] / close_prices.iloc[-2] - 1) if len(close_prices) > 1 else 0),
            'volatility': float(close_prices.pct_change().rolling(window=20).std().iloc[-1]) if len(close_prices) > 20 else 0.02
        }

        return features

    def _generate_prediction(self, features: Dict[str, float]) -> float:
        """Generate prediction (simplified)"""
        score = 0.5

        if features.get('rsi', 50) < 30:
            score += 0.3
        elif features.get('rsi', 50) > 70:
            score -= 0.3

        if features.get('price_change_1d', 0) > 0.02:
            score += 0.2
        elif features.get('price_change_1d', 0) < -0.02:
            score -= 0.2

        return max(0.0, min(1.0, score))

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        if loss.iloc[-1] == 0:
            return 100.0

        rs = gain.iloc[-1] / loss.iloc[-1]
        return 100 - (100 / (1 + rs))
