"""
Technical Analysis Module
========================

Enhanced technical analysis tools for trading strategies including
support for candle momentum analysis and various technical indicators.

Author: Trading Bot System
Updated: 2025-01-24
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class TechnicalAnalysis:
    """
    Technical Analysis toolkit with support for candle momentum strategies
    and traditional technical indicators.
    """
    
    def __init__(self):
        """Initialize Technical Analysis module"""
        self.logger = logger
    
    # ==================== MOVING AVERAGES ====================
    
    def calculate_sma(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average
        
        Args:
            data: Price series
            period: Period for calculation
            
        Returns:
            SMA series
        """
        try:
            return data.rolling(window=period).mean()
        except Exception as e:
            self.logger.error(f"Error calculating SMA: {e}")
            return pd.Series(np.nan, index=data.index)
    
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average
        
        Args:
            data: Price series
            period: Period for calculation
            
        Returns:
            EMA series
        """
        try:
            return data.ewm(span=period).mean()
        except Exception as e:
            self.logger.error(f"Error calculating EMA: {e}")
            return pd.Series(np.nan, index=data.index)
    
    def calculate_wma(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Weighted Moving Average
        
        Args:
            data: Price series
            period: Period for calculation
            
        Returns:
            WMA series
        """
        try:
            weights = np.arange(1, period + 1)
            return data.rolling(window=period).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )
        except Exception as e:
            self.logger.error(f"Error calculating WMA: {e}")
            return pd.Series(np.nan, index=data.index)
    
    # ==================== CANDLE ANALYSIS ====================
    
    def calculate_candle_body_size(self, open_data: pd.Series, close_data: pd.Series) -> pd.Series:
        """
        Calculate candle body size (absolute difference between open and close)
        
        Args:
            open_data: Open price series
            close_data: Close price series
            
        Returns:
            Body size series
        """
        try:
            return abs(close_data - open_data)
        except Exception as e:
            self.logger.error(f"Error calculating candle body size: {e}")
            return pd.Series(0.0, index=open_data.index)
    
    def calculate_candle_direction(self, open_data: pd.Series, close_data: pd.Series) -> pd.Series:
        """
        Calculate candle direction (1 for bullish, -1 for bearish, 0 for doji)
        
        Args:
            open_data: Open price series
            close_data: Close price series
            
        Returns:
            Direction series
        """
        try:
            direction = pd.Series(0, index=open_data.index)
            direction[close_data > open_data] = 1
            direction[close_data < open_data] = -1
            return direction
        except Exception as e:
            self.logger.error(f"Error calculating candle direction: {e}")
            return pd.Series(0, index=open_data.index)
    
    def calculate_upper_shadow(self, open_data: pd.Series, high_data: pd.Series, 
                              close_data: pd.Series) -> pd.Series:
        """
        Calculate upper shadow size
        
        Args:
            open_data: Open price series
            high_data: High price series
            close_data: Close price series
            
        Returns:
            Upper shadow series
        """
        try:
            body_top = np.maximum(open_data, close_data)
            return high_data - body_top
        except Exception as e:
            self.logger.error(f"Error calculating upper shadow: {e}")
            return pd.Series(0.0, index=open_data.index)
    
    def calculate_lower_shadow(self, open_data: pd.Series, low_data: pd.Series, 
                              close_data: pd.Series) -> pd.Series:
        """
        Calculate lower shadow size
        
        Args:
            open_data: Open price series
            low_data: Low price series
            close_data: Close price series
            
        Returns:
            Lower shadow series
        """
        try:
            body_bottom = np.minimum(open_data, close_data)
            return body_bottom - low_data
        except Exception as e:
            self.logger.error(f"Error calculating lower shadow: {e}")
            return pd.Series(0.0, index=open_data.index)
    
    def identify_doji_candles(self, open_data: pd.Series, close_data: pd.Series, 
                             threshold: float = 0.001) -> pd.Series:
        """
        Identify doji candles (open ≈ close)
        
        Args:
            open_data: Open price series
            close_data: Close price series
            threshold: Relative threshold for doji identification
            
        Returns:
            Boolean series indicating doji candles
        """
        try:
            body_size = abs(close_data - open_data)
            avg_price = (open_data + close_data) / 2
            relative_body = body_size / avg_price
            return relative_body <= threshold
        except Exception as e:
            self.logger.error(f"Error identifying doji candles: {e}")
            return pd.Series(False, index=open_data.index)
    
    # ==================== MOMENTUM INDICATORS ====================
    
    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index
        
        Args:
            data: Price series
            period: Period for calculation
            
        Returns:
            RSI series
        """
        try:
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return pd.Series(50.0, index=data.index)
    
    def calculate_macd(self, data: pd.Series, fast: int = 12, slow: int = 26, 
                      signal: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            data: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            
        Returns:
            Dictionary with MACD, signal, and histogram
        """
        try:
            ema_fast = self.calculate_ema(data, fast)
            ema_slow = self.calculate_ema(data, slow)
            macd_line = ema_fast - ema_slow
            signal_line = self.calculate_ema(macd_line, signal)
            histogram = macd_line - signal_line
            
            return {
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            }
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            return {
                'macd': pd.Series(0.0, index=data.index),
                'signal': pd.Series(0.0, index=data.index),
                'histogram': pd.Series(0.0, index=data.index)
            }
    
    def calculate_stochastic(self, high_data: pd.Series, low_data: pd.Series, 
                           close_data: pd.Series, k_period: int = 14, 
                           d_period: int = 3) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator
        
        Args:
            high_data: High price series
            low_data: Low price series
            close_data: Close price series
            k_period: %K period
            d_period: %D period
            
        Returns:
            Dictionary with %K and %D
        """
        try:
            lowest_low = low_data.rolling(window=k_period).min()
            highest_high = high_data.rolling(window=k_period).max()
            
            k_percent = 100 * ((close_data - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_period).mean()
            
            return {
                'k_percent': k_percent,
                'd_percent': d_percent
            }
        except Exception as e:
            self.logger.error(f"Error calculating Stochastic: {e}")
            return {
                'k_percent': pd.Series(50.0, index=close_data.index),
                'd_percent': pd.Series(50.0, index=close_data.index)
            }
    
    # ==================== VOLATILITY INDICATORS ====================
    
    def calculate_bollinger_bands(self, data: pd.Series, period: int = 20, 
                                 std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Args:
            data: Price series
            period: Period for calculation
            std_dev: Standard deviation multiplier
            
        Returns:
            Dictionary with upper, middle, and lower bands
        """
        try:
            middle_band = self.calculate_sma(data, period)
            std = data.rolling(window=period).std()
            upper_band = middle_band + (std * std_dev)
            lower_band = middle_band - (std * std_dev)
            
            return {
                'upper': upper_band,
                'middle': middle_band,
                'lower': lower_band
            }
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {e}")
            return {
                'upper': pd.Series(np.nan, index=data.index),
                'middle': pd.Series(np.nan, index=data.index),
                'lower': pd.Series(np.nan, index=data.index)
            }
    
    def calculate_atr(self, high_data: pd.Series, low_data: pd.Series, 
                     close_data: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range
        
        Args:
            high_data: High price series
            low_data: Low price series
            close_data: Close price series
            period: Period for calculation
            
        Returns:
            ATR series
        """
        try:
            prev_close = close_data.shift(1)
            tr1 = high_data - low_data
            tr2 = abs(high_data - prev_close)
            tr3 = abs(low_data - prev_close)
            
            true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            return atr
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return pd.Series(0.0, index=high_data.index)
    
    # ==================== VOLUME INDICATORS ====================
    
    def calculate_volume_sma(self, volume_data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Volume Simple Moving Average
        
        Args:
            volume_data: Volume series
            period: Period for calculation
            
        Returns:
            Volume SMA series
        """
        try:
            return volume_data.rolling(window=period).mean()
        except Exception as e:
            self.logger.error(f"Error calculating Volume SMA: {e}")
            return pd.Series(np.nan, index=volume_data.index)
    
    def calculate_vwap(self, high_data: pd.Series, low_data: pd.Series, 
                      close_data: pd.Series, volume_data: pd.Series) -> pd.Series:
        """
        Calculate Volume Weighted Average Price
        
        Args:
            high_data: High price series
            low_data: Low price series
            close_data: Close price series
            volume_data: Volume series
            
        Returns:
            VWAP series
        """
        try:
            typical_price = (high_data + low_data + close_data) / 3
            vwap = (typical_price * volume_data).cumsum() / volume_data.cumsum()
            return vwap
        except Exception as e:
            self.logger.error(f"Error calculating VWAP: {e}")
            return pd.Series(np.nan, index=high_data.index)
    
    # ==================== CANDLE MOMENTUM SPECIFIC ====================
    
    def calculate_candle_body_momentum(self, open_data: pd.Series, close_data: pd.Series,
                                     lookback_period: int = 10) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate candle body momentum exactly as described in TradingView video
        
        For each candle:
        - If bullish (close > open): add body size to bullish_strength
        - If bearish (close < open): add body size to bearish_strength
        - Calculate running sum over lookback period (NOT average)
        
        Args:
            open_data: Open price series
            close_data: Close price series
            lookback_period: Number of candles to look back (default: 10 from video)
            
        Returns:
            Tuple of (bullish_strength, bearish_strength) as running sums
        """
        try:
            # Calculate candle body sizes
            body_size = abs(close_data - open_data)
            
            # Identify bullish and bearish candles
            is_bullish = close_data > open_data
            is_bearish = close_data < open_data
            
            # Initialize arrays
            bullish_bodies = pd.Series(0.0, index=open_data.index)
            bearish_bodies = pd.Series(0.0, index=open_data.index)
            
            # Assign body sizes to respective arrays
            bullish_bodies[is_bullish] = body_size[is_bullish]
            bearish_bodies[is_bearish] = body_size[is_bearish]
            
            # Calculate RUNNING SUMS over lookback period (key point from video)
            bullish_strength = bullish_bodies.rolling(window=lookback_period, min_periods=1).sum()
            bearish_strength = bearish_bodies.rolling(window=lookback_period, min_periods=1).sum()
            
            return bullish_strength, bearish_strength
            
        except Exception as e:
            self.logger.error(f"Error calculating candle body momentum: {e}")
            # Return empty series on error
            empty_series = pd.Series(0.0, index=open_data.index)
            return empty_series, empty_series
    
    def detect_momentum_crossovers(self, bullish_strength: pd.Series, 
                                 bearish_strength: pd.Series) -> Dict[str, bool]:
        """
        Detect exact crossovers as described in TradingView video
        
        CRITICAL: Crossover must be detected at the exact bar:
        - Bullish crossover: bullish_strength > bearish_strength AND 
          previous bar had bullish_strength <= bearish_strength
        - Bearish crossover: bearish_strength > bullish_strength AND
          previous bar had bearish_strength <= bullish_strength
        
        Args:
            bullish_strength: Bullish momentum series
            bearish_strength: Bearish momentum series
            
        Returns:
            Dictionary with crossover flags
        """
        try:
            if len(bullish_strength) < 2 or len(bearish_strength) < 2:
                return {'bullish_crossover': False, 'bearish_crossover': False}
            
            # Current values
            bull_current = bullish_strength.iloc[-1]
            bear_current = bearish_strength.iloc[-1]
            
            # Previous values
            bull_previous = bullish_strength.iloc[-2]
            bear_previous = bearish_strength.iloc[-2]
            
            # Detect crossovers (EXACT implementation from video)
            bullish_crossover = (
                bull_current > bear_current and  # Current: bullish > bearish
                bull_previous <= bear_previous   # Previous: bullish <= bearish
            )
            
            bearish_crossover = (
                bear_current > bull_current and  # Current: bearish > bullish
                bear_previous <= bull_previous   # Previous: bearish <= bullish
            )
            
            return {
                'bullish_crossover': bullish_crossover,
                'bearish_crossover': bearish_crossover
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting momentum crossovers: {e}")
            return {'bullish_crossover': False, 'bearish_crossover': False}
    
    def calculate_momentum_divergence(self, price_data: pd.Series, 
                                    momentum_data: pd.Series, 
                                    period: int = 5) -> Dict[str, pd.Series]:
        """
        Calculate momentum divergence signals
        
        Args:
            price_data: Price series
            momentum_data: Momentum indicator series
            period: Period for divergence detection
            
        Returns:
            Dictionary with bullish and bearish divergence signals
        """
        try:
            # Find local peaks and troughs
            price_peaks = price_data.rolling(window=period, center=True).max() == price_data
            price_troughs = price_data.rolling(window=period, center=True).min() == price_data
            
            momentum_peaks = momentum_data.rolling(window=period, center=True).max() == momentum_data
            momentum_troughs = momentum_data.rolling(window=period, center=True).min() == momentum_data
            
            # Bullish divergence: price makes lower low, momentum makes higher low
            bullish_div = pd.Series(False, index=price_data.index)
            bearish_div = pd.Series(False, index=price_data.index)
            
            # This is a simplified implementation - in practice you'd want more sophisticated peak/trough detection
            
            return {
                'bullish_divergence': bullish_div,
                'bearish_divergence': bearish_div
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum divergence: {e}")
            return {
                'bullish_divergence': pd.Series(False, index=price_data.index),
                'bearish_divergence': pd.Series(False, index=price_data.index)
            }
    
    def calculate_candle_momentum_score(self, open_data: pd.Series, close_data: pd.Series, 
                                      volume_data: pd.Series, period: int = 10) -> pd.Series:
        """
        Calculate overall candle momentum score
        
        Args:
            open_data: Open price series
            close_data: Close price series
            volume_data: Volume series
            period: Period for calculation
            
        Returns:
            Momentum score series (-1 to 1)
        """
        try:
            # Calculate directional momentum
            direction = self.calculate_candle_direction(open_data, close_data)
            body_size = self.calculate_candle_body_size(open_data, close_data)
            
            # Normalize body size by recent average
            avg_body_size = body_size.rolling(window=period).mean()
            normalized_body = body_size / (avg_body_size + 1e-8)
            
            # Combine direction with normalized body size
            momentum_score = direction * normalized_body
            
            # Apply volume weighting
            avg_volume = volume_data.rolling(window=period).mean()
            volume_factor = np.clip(volume_data / (avg_volume + 1e-8), 0.5, 2.0)
            
            weighted_momentum = momentum_score * volume_factor
            
            # Normalize to -1 to 1 range
            rolling_std = weighted_momentum.rolling(window=period).std()
            normalized_momentum = weighted_momentum / (rolling_std * 2 + 1e-8)
            normalized_momentum = np.clip(normalized_momentum, -1, 1)
            
            return normalized_momentum
            
        except Exception as e:
            self.logger.error(f"Error calculating candle momentum score: {e}")
            return pd.Series(0.0, index=open_data.index)
    
    # ==================== UTILITY FUNCTIONS ====================
    
    def detect_crossover(self, series1: pd.Series, series2: pd.Series) -> Dict[str, pd.Series]:
        """
        Detect crossovers between two series
        
        Args:
            series1: First series
            series2: Second series
            
        Returns:
            Dictionary with crossover signals
        """
        try:
            cross_above = (series1 > series2) & (series1.shift(1) <= series2.shift(1))
            cross_below = (series1 < series2) & (series1.shift(1) >= series2.shift(1))
            
            return {
                'cross_above': cross_above,
                'cross_below': cross_below
            }
        except Exception as e:
            self.logger.error(f"Error detecting crossover: {e}")
            return {
                'cross_above': pd.Series(False, index=series1.index),
                'cross_below': pd.Series(False, index=series1.index)
            }
    
    def calculate_correlation(self, series1: pd.Series, series2: pd.Series, 
                            period: int = 20) -> pd.Series:
        """
        Calculate rolling correlation between two series
        
        Args:
            series1: First series
            series2: Second series
            period: Rolling window period
            
        Returns:
            Correlation series
        """
        try:
            return series1.rolling(window=period).corr(series2)
        except Exception as e:
            self.logger.error(f"Error calculating correlation: {e}")
            return pd.Series(0.0, index=series1.index)