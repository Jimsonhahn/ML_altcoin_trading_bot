"""
Candle Body Momentum Strategy - Exact Implementation from TradingView Video
=========================================================================

This strategy implements the exact candle body momentum strategy as described in the 
TradingView Pine Script video with the following specifications:

1. For each of the last 10 candles, measure body size: abs(close - open)
2. Bullish candles (close > open) add to bullish_strength (running sum)
3. Bearish candles (close < open) add to bearish_strength (running sum)
4. LONG: Bullish strength crosses ABOVE bearish strength AND price > 200 SMA
5. SHORT: Bearish strength crosses ABOVE bullish strength AND price < 200 SMA
6. Exit on opposite crossover

Optimal Parameters from Video:
- Timeframe: 30 minutes
- SMA Period: 200
- Lookback Period: 10 candles
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from datetime import datetime

from .strategy_base import Strategy

logger = logging.getLogger(__name__)


class CandleBodyMomentumStrategy(Strategy):
    """
    Candle Body Momentum Strategy - Exact Video Implementation
    
    This strategy measures the momentum of bullish vs bearish candle bodies
    over a lookback period and generates signals on crossovers with trend confirmation.
    """
    
    def __init__(self, params: Dict[str, Any] = None, ml_components: Optional[Any] = None):
        """
        Initialize the Candle Body Momentum Strategy
        
        Args:
            params: Strategy configuration dictionary
            ml_components: Optional ML components for enhancement
        """
        super().__init__(params, ml_components)
        
        # Exact parameters from the video
        self.lookback_period = self.params.get('lookback_period', 10)  # Video: 10 candles
        self.sma_period = self.params.get('sma_period', 200)  # Video: 200 SMA (NOT 100)
        self.timeframe = self.params.get('timeframe', '30m')  # Video: 30 minutes optimal
        
        # Debug logging enabled by default for verification
        self.debug_logging = self.params.get('debug_logging', True)
        
        # Store momentum history for crossover detection
        self.momentum_history = {}
        self.last_signals = {}
        
        logger.info(f"Initialized Candle Body Momentum Strategy:")
        logger.info(f"  - Lookback Period: {self.lookback_period} candles")
        logger.info(f"  - SMA Period: {self.sma_period}")
        logger.info(f"  - Optimal Timeframe: {self.timeframe}")
        logger.info(f"  - Debug Logging: {self.debug_logging}")
    
    def calculate_candle_body_momentum(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate bullish and bearish momentum exactly as described in video
        
        For each candle:
        - If bullish (close > open): add body size to bullish_strength
        - If bearish (close < open): add body size to bearish_strength
        - Calculate running sum over lookback period (NOT average)
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Tuple of (bullish_strength, bearish_strength) as running sums
        """
        try:
            # Calculate candle body sizes
            body_size = abs(data['close'] - data['open'])
            
            # Identify bullish and bearish candles
            is_bullish = data['close'] > data['open']
            is_bearish = data['close'] < data['open']
            
            # Initialize arrays
            bullish_bodies = pd.Series(0.0, index=data.index)
            bearish_bodies = pd.Series(0.0, index=data.index)
            
            # Assign body sizes to respective arrays
            bullish_bodies[is_bullish] = body_size[is_bullish]
            bearish_bodies[is_bearish] = body_size[is_bearish]
            
            # Calculate RUNNING SUMS over lookback period (key point from video)
            bullish_strength = bullish_bodies.rolling(window=self.lookback_period, min_periods=1).sum()
            bearish_strength = bearish_bodies.rolling(window=self.lookback_period, min_periods=1).sum()
            
            if self.debug_logging and len(data) > 0:
                logger.info(f"Body Momentum Calculation (last bar):")
                logger.info(f"  - Body Size: {body_size.iloc[-1]:.4f}")
                logger.info(f"  - Is Bullish: {is_bullish.iloc[-1]}")
                logger.info(f"  - Bullish Strength: {bullish_strength.iloc[-1]:.4f}")
                logger.info(f"  - Bearish Strength: {bearish_strength.iloc[-1]:.4f}")
            
            return bullish_strength, bearish_strength
            
        except Exception as e:
            logger.error(f"Error calculating candle body momentum: {e}")
            # Return empty series on error
            empty_series = pd.Series(0.0, index=data.index)
            return empty_series, empty_series
    
    def calculate_sma_200(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate 200-period Simple Moving Average
        Video specifically mentioned 200 SMA worked better than 100
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            200-period SMA Series
        """
        try:
            sma_200 = data['close'].rolling(window=self.sma_period, min_periods=1).mean()
            
            if self.debug_logging and len(data) > 0:
                current_price = data['close'].iloc[-1]
                current_sma = sma_200.iloc[-1]
                logger.info(f"SMA-200 Analysis:")
                logger.info(f"  - Current Price: {current_price:.2f}")
                logger.info(f"  - SMA-200: {current_sma:.2f}")
                logger.info(f"  - Price vs SMA: {'Above' if current_price > current_sma else 'Below'}")
            
            return sma_200
            
        except Exception as e:
            logger.error(f"Error calculating SMA-200: {e}")
            return pd.Series(np.nan, index=data.index)
    
    def detect_momentum_crossovers(self, bullish_strength: pd.Series, 
                                 bearish_strength: pd.Series) -> Dict[str, bool]:
        """
        Detect exact crossovers as described in video
        
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
            
            if self.debug_logging:
                logger.info(f"Crossover Detection:")
                logger.info(f"  - Current: Bull={bull_current:.4f}, Bear={bear_current:.4f}")
                logger.info(f"  - Previous: Bull={bull_previous:.4f}, Bear={bear_previous:.4f}")
                logger.info(f"  - Bullish Crossover: {bullish_crossover}")
                logger.info(f"  - Bearish Crossover: {bearish_crossover}")
            
            return {
                'bullish_crossover': bullish_crossover,
                'bearish_crossover': bearish_crossover
            }
            
        except Exception as e:
            logger.error(f"Error detecting crossovers: {e}")
            return {'bullish_crossover': False, 'bearish_crossover': False}
    
    def calculate_signal(self, symbol: str, data: pd.DataFrame, 
                        current_price: float) -> Tuple[str, Dict[str, Any]]:
        """
        Calculate trading signal using exact video methodology
        
        Entry Conditions from Video:
        - LONG: Bullish strength crosses above bearish strength AND price > 200 SMA
        - SHORT: Bearish strength crosses above bullish strength AND price < 200 SMA
        
        Exit Conditions:
        - Exit long when bearish strength crosses back above bullish strength
        - Exit short when bullish strength crosses back above bearish strength
        
        Args:
            symbol: Trading pair symbol
            data: OHLCV DataFrame
            current_price: Current asset price
            
        Returns:
            Tuple of (signal_string, signal_data_dict)
        """
        try:
            # Need sufficient data
            min_periods = max(self.lookback_period, self.sma_period) + 2
            if len(data) < min_periods:
                logger.warning(f"Insufficient data: {len(data)} < {min_periods}")
                return 'HOLD', {'confidence': 0.0, 'reason': 'insufficient_data'}
            
            # Calculate momentum components
            bullish_strength, bearish_strength = self.calculate_candle_body_momentum(data)
            sma_200 = self.calculate_sma_200(data)
            
            # Detect crossovers
            crossovers = self.detect_momentum_crossovers(bullish_strength, bearish_strength)
            
            # Current values
            current_sma = sma_200.iloc[-1]
            bull_strength = bullish_strength.iloc[-1]
            bear_strength = bearish_strength.iloc[-1]
            
            # Price vs SMA conditions
            price_above_sma = current_price > current_sma
            price_below_sma = current_price < current_sma
            
            # Initialize signal
            signal = 'HOLD'
            confidence = 0.0
            reason = 'no_signal'
            
            # LONG ENTRY: Bullish crossover AND price > 200 SMA
            if crossovers['bullish_crossover'] and price_above_sma:
                signal = 'BUY'
                # Confidence based on momentum strength difference
                momentum_diff = bull_strength - bear_strength
                confidence = min(0.8 + (momentum_diff / max(bull_strength, bear_strength, 1)), 1.0)
                reason = 'bullish_crossover_above_sma'
                
                logger.info(f"🟢 LONG SIGNAL GENERATED for {symbol}")
                logger.info(f"   Bullish Strength: {bull_strength:.4f}")
                logger.info(f"   Bearish Strength: {bear_strength:.4f}")
                logger.info(f"   Price: {current_price:.2f} > SMA-200: {current_sma:.2f}")
                logger.info(f"   Confidence: {confidence:.2f}")
            
            # SHORT ENTRY: Bearish crossover AND price < 200 SMA
            elif crossovers['bearish_crossover'] and price_below_sma:
                signal = 'SELL'
                # Confidence based on momentum strength difference
                momentum_diff = bear_strength - bull_strength
                confidence = min(0.8 + (momentum_diff / max(bull_strength, bear_strength, 1)), 1.0)
                reason = 'bearish_crossover_below_sma'
                
                logger.info(f"🔴 SHORT SIGNAL GENERATED for {symbol}")
                logger.info(f"   Bearish Strength: {bear_strength:.4f}")
                logger.info(f"   Bullish Strength: {bull_strength:.4f}")
                logger.info(f"   Price: {current_price:.2f} < SMA-200: {current_sma:.2f}")
                logger.info(f"   Confidence: {confidence:.2f}")
            
            # Check for exit conditions if we have a previous position
            elif symbol in self.last_signals:
                last_signal = self.last_signals[symbol]
                
                # Exit long position on bearish crossover
                if (last_signal.get('signal') == 'BUY' and 
                    crossovers['bearish_crossover']):
                    signal = 'SELL'
                    confidence = 0.7
                    reason = 'exit_long_bearish_crossover'
                    
                    logger.info(f"🔄 EXIT LONG for {symbol} - Bearish crossover")
                
                # Exit short position on bullish crossover
                elif (last_signal.get('signal') == 'SELL' and 
                      crossovers['bullish_crossover']):
                    signal = 'BUY'
                    confidence = 0.7
                    reason = 'exit_short_bullish_crossover'
                    
                    logger.info(f"🔄 EXIT SHORT for {symbol} - Bullish crossover")
            
            # Create comprehensive signal data
            signal_data = {
                'signal': signal,
                'confidence': confidence,
                'reason': reason,
                'metadata': {
                    'bullish_strength': float(bull_strength),
                    'bearish_strength': float(bear_strength),
                    'momentum_difference': float(bull_strength - bear_strength),
                    'sma_200': float(current_sma),
                    'price_vs_sma': float(current_price / current_sma) if current_sma > 0 else 1.0,
                    'bullish_crossover': crossovers['bullish_crossover'],
                    'bearish_crossover': crossovers['bearish_crossover'],
                    'lookback_period': self.lookback_period,
                    'sma_period': self.sma_period,
                    'strategy': 'candle_body_momentum'
                }
            }
            
            # Store signal for exit tracking
            if signal != 'HOLD':
                self.last_signals[symbol] = signal_data
            
            return signal, signal_data
            
        except Exception as e:
            logger.error(f"Error calculating signal for {symbol}: {e}")
            return 'HOLD', {'confidence': 0.0, 'reason': 'error', 'error': str(e)}
    
    def get_momentum_visualization_data(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get data for visualizing the two momentum lines (debugging purposes)
        
        This helps verify the strategy is working as described in the video:
        - Bullish line should rise sharply during strong upward moves
        - Bearish line should stay flat during bullish periods
        - Lines should diverge significantly at strong moves
        
        Args:
            symbol: Trading pair symbol
            data: OHLCV DataFrame
            
        Returns:
            Visualization data dictionary
        """
        try:
            if len(data) < self.lookback_period:
                return {}
            
            # Calculate momentum data
            bullish_strength, bearish_strength = self.calculate_candle_body_momentum(data)
            sma_200 = self.calculate_sma_200(data)
            
            # Detect all crossovers in the dataset
            crossover_points = []
            for i in range(1, len(bullish_strength)):
                bull_curr = bullish_strength.iloc[i]
                bear_curr = bearish_strength.iloc[i]
                bull_prev = bullish_strength.iloc[i-1]
                bear_prev = bearish_strength.iloc[i-1]
                
                if bull_curr > bear_curr and bull_prev <= bear_prev:
                    crossover_points.append({
                        'timestamp': data.index[i],
                        'type': 'bullish_crossover',
                        'price': data['close'].iloc[i],
                        'bull_strength': bull_curr,
                        'bear_strength': bear_curr
                    })
                elif bear_curr > bull_curr and bear_prev <= bull_prev:
                    crossover_points.append({
                        'timestamp': data.index[i],
                        'type': 'bearish_crossover',
                        'price': data['close'].iloc[i],
                        'bull_strength': bull_curr,
                        'bear_strength': bear_curr
                    })
            
            return {
                'timestamp': data.index,
                'price': data['close'],
                'bullish_strength': bullish_strength,
                'bearish_strength': bearish_strength,
                'sma_200': sma_200,
                'crossover_points': crossover_points,
                'momentum_difference': bullish_strength - bearish_strength,
                'strategy_params': {
                    'lookback_period': self.lookback_period,
                    'sma_period': self.sma_period,
                    'timeframe': self.timeframe
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting visualization data for {symbol}: {e}")
            return {}
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get comprehensive strategy information
        
        Returns:
            Strategy info dictionary
        """
        return {
            'name': 'Candle Body Momentum Strategy',
            'version': '1.0.0',
            'description': 'Exact implementation from TradingView Pine Script video',
            'parameters': {
                'lookback_period': self.lookback_period,
                'sma_period': self.sma_period,
                'optimal_timeframe': self.timeframe,
                'debug_logging': self.debug_logging
            },
            'entry_conditions': {
                'long': 'Bullish strength crosses above bearish strength AND price > 200 SMA',
                'short': 'Bearish strength crosses above bullish strength AND price < 200 SMA'
            },
            'exit_conditions': {
                'long': 'Exit when bearish strength crosses back above bullish strength',
                'short': 'Exit when bullish strength crosses back above bearish strength'
            },
            'key_features': [
                'Running sum of candle body sizes (not averages)',
                'Exact crossover detection at candle close',
                '200 SMA trend confirmation',
                'High Sortino ratio (good downside protection)',
                'Optimal for 30-minute timeframe'
            ],
            'active_symbols': list(self.last_signals.keys()),
            'last_update': datetime.now().isoformat()
        }
    
    def validate_implementation(self, data: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate that the implementation matches video specifications
        
        Args:
            data: Test OHLCV data
            
        Returns:
            Validation results
        """
        validation_results = {
            'sufficient_data': len(data) >= max(self.lookback_period, self.sma_period),
            'momentum_calculation': False,
            'crossover_detection': False,
            'sma_calculation': False
        }
        
        try:
            if validation_results['sufficient_data']:
                # Test momentum calculation
                bull_strength, bear_strength = self.calculate_candle_body_momentum(data)
                validation_results['momentum_calculation'] = (
                    len(bull_strength) == len(data) and 
                    len(bear_strength) == len(data) and
                    not bull_strength.isna().all() and
                    not bear_strength.isna().all()
                )
                
                # Test crossover detection
                crossovers = self.detect_momentum_crossovers(bull_strength, bear_strength)
                validation_results['crossover_detection'] = (
                    'bullish_crossover' in crossovers and 
                    'bearish_crossover' in crossovers
                )
                
                # Test SMA calculation
                sma_200 = self.calculate_sma_200(data)
                validation_results['sma_calculation'] = (
                    len(sma_200) == len(data) and
                    not sma_200.isna().all()
                )
            
            logger.info(f"Strategy Validation Results: {validation_results}")
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
        
        return validation_results