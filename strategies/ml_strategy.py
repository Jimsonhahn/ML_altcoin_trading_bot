#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Machine Learning Strategy - Fixed Implementation
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, Optional
import logging
from datetime import datetime

from .strategy_base import Strategy

logger = logging.getLogger(__name__)


class MLStrategy(Strategy):
    """ML-based Trading Strategy with simple implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # ML parameters
        self.lookback_period = config.get('lookback_period', 50)
        self.prediction_threshold = config.get('prediction_threshold', 0.6)

        # Feature engineering parameters
        self.ma_periods = [5, 10, 20, 50]
        self.rsi_period = 14

        # Simple model state
        self.model_trained = False
        self.feature_importance = {
            'trend': 0.3,
            'momentum': 0.3,
            'volatility': 0.2,
            'volume': 0.2
        }

        logger.info("ML Strategy initialized (simplified version)")

    def calculate_signal(self, data: pd.DataFrame, symbol: str,
                         current_position: Optional[Any] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Calculate ML-based trading signal using simplified feature engineering

        Returns:
            Tuple of (signal, signal_data)
        """
        if len(data) < self.lookback_period:
            return 'HOLD', {'confidence': 0.0, 'reason': 'insufficient_data'}

        # Extract features
        features = self._extract_features(data)

        # Generate prediction (simplified - no actual ML model)
        prediction, confidence = self._generate_prediction(features)

        # Determine signal
        signal = 'HOLD'

        if current_position is None:
            # No position - consider entry
            if prediction > self.prediction_threshold:
                signal = 'BUY'
            elif prediction < (1 - self.prediction_threshold):
                signal = 'SELL'
        else:
            # Have position - consider exit
            if current_position.side == 'buy' and prediction < 0.4:
                signal = 'SELL'
            elif current_position.side == 'sell' and prediction > 0.6:
                signal = 'BUY'

        signal_data = {
            'signal': signal,
            'confidence': confidence,
            'strategy': 'ml',
            'prediction': float(prediction),
            'features': features,
            'model_trained': self.model_trained
        }

        return signal, signal_data

    def _extract_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract features for ML model"""
        features = {}

        close_prices = data['close']
        volume = data['volume']

        # Trend features
        for period in self.ma_periods:
            if len(data) >= period:
                ma = close_prices.rolling(window=period).mean()
                features[f'ma_{period}_ratio'] = float(close_prices.iloc[-1] / ma.iloc[-1]) if ma.iloc[-1] > 0 else 1.0

        # Momentum features
        features['rsi'] = float(self._calculate_rsi(close_prices))
        features['price_change_1d'] = float(
            (close_prices.iloc[-1] / close_prices.iloc[-2] - 1) if len(close_prices) > 1 else 0)
        features['price_change_7d'] = float(
            (close_prices.iloc[-1] / close_prices.iloc[-7] - 1) if len(close_prices) > 7 else 0)

        # Volatility features
        features['volatility'] = float(close_prices.pct_change().rolling(window=20).std().iloc[-1]) if len(
            close_prices) > 20 else 0.02

        # Volume features
        if len(volume) > 20:
            features['volume_ratio'] = float(volume.iloc[-1] / volume.rolling(window=20).mean().iloc[-1])
        else:
            features['volume_ratio'] = 1.0

        return features

    def _generate_prediction(self, features: Dict[str, float]) -> Tuple[float, float]:
        """
        Generate prediction based on features (simplified scoring system)
        Returns: (prediction, confidence)
        """
        score = 0.5  # Neutral baseline

        # Trend scoring
        trend_score = 0.0
        for period in self.ma_periods:
            key = f'ma_{period}_ratio'
            if key in features:
                if features[key] > 1.01:  # Price above MA
                    trend_score += 0.25
                elif features[key] < 0.99:  # Price below MA
                    trend_score -= 0.25

        # Momentum scoring
        momentum_score = 0.0
        if features.get('rsi', 50) > 70:
            momentum_score -= 0.3  # Overbought
        elif features.get('rsi', 50) < 30:
            momentum_score += 0.3  # Oversold

        # Recent price change
        if features.get('price_change_1d', 0) > 0.02:
            momentum_score += 0.2
        elif features.get('price_change_1d', 0) < -0.02:
            momentum_score -= 0.2

        # Volatility adjustment
        volatility_factor = 1.0
        if features.get('volatility', 0.02) > 0.03:
            volatility_factor = 0.8  # Reduce confidence in high volatility

        # Volume confirmation
        volume_factor = 1.0
        if features.get('volume_ratio', 1.0) > 1.5:
            volume_factor = 1.1  # Increase confidence with high volume

        # Combine scores
        score += (trend_score * self.feature_importance['trend'] +
                  momentum_score * self.feature_importance['momentum'])

        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))

        # Calculate confidence
        confidence = abs(score - 0.5) * 2 * volatility_factor * volume_factor
        confidence = min(0.9, confidence)  # Cap at 90%

        return score, confidence

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate current RSI value"""
        if len(prices) < period + 1:
            return 50.0  # Neutral

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        if loss.iloc[-1] == 0:
            return 100.0

        rs = gain.iloc[-1] / loss.iloc[-1]
        rsi = 100 - (100 / (1 + rs))

        return rsi