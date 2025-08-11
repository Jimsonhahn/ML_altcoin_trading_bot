"""
Candle Body Momentum Strategy
============================

A strategy that measures bullish vs bearish candle body momentum to generate trading signals.
This strategy analyzes the cumulative strength of bullish and bearish candles over a lookback
period and generates signals based on momentum crossovers and trend filters.

Author: Trading Bot System
Created: 2025-01-24
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from datetime import datetime

from .strategy_base import Strategy, Signal
from typing import Dict, Tuple, Any, Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from analysis.technical import TechnicalAnalysis
except ImportError:
    # If technical module doesn't exist, we'll use basic calculations
    TechnicalAnalysis = None


class CandleMomentumStrategy(Strategy):
    """
    Candle Body Momentum Trading Strategy
    
    This strategy measures the cumulative strength of bullish vs bearish candle bodies
    over a specified lookback period. It generates trading signals when momentum
    strength crosses over and confirms with trend filters.
    
    Key Features:
    - Bullish/Bearish momentum calculation
    - Trend confirmation with SMA/EMA
    - Multi-timeframe support
    - Volume filtering
    - Momentum strength ratios
    - Configurable parameters
    """
    
    def __init__(self, params: Dict[str, Any] = None, ml_components: Optional[Any] = None):
        """
        Initialize the Candle Momentum Strategy
        
        Args:
            params: Strategy configuration dictionary
            ml_components: Optional ML components for enhancement
        """
        super().__init__(params, ml_components)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize technical analysis if available
        if TechnicalAnalysis:
            self.technical = TechnicalAnalysis()
        else:
            self.technical = None
            self.logger.warning("Technical analysis module not available, using basic calculations")
        
        # Strategy parameters from config
        self.lookback_period = self.params.get('lookback_period', 20)
        self.sma_period = self.params.get('sma_period', 50)
        self.use_ema = self.params.get('use_ema', False)
        self.volume_filter = self.params.get('volume_filter', True)
        self.volume_period = self.params.get('volume_period', 20)
        self.min_momentum_ratio = self.params.get('min_momentum_ratio', 1.2)
        self.multi_timeframe = self.params.get('multi_timeframe', False)
        self.higher_timeframe = self.params.get('higher_timeframe', '4h')
        self.min_confidence = self.params.get('min_confidence', 0.5)
        
        # Internal state tracking
        self.momentum_history = {}
        self.last_signals = {}
        
        self.logger.info(f"Initialized Candle Momentum Strategy with params: {self.params}")
    
    def calculate_candle_bodies(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate bullish and bearish candle body sizes
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Tuple of (bullish_bodies, bearish_bodies) Series
        """
        try:
            # Calculate candle body sizes
            body_size = abs(data['close'] - data['open'])
            
            # Separate bullish and bearish bodies
            bullish_mask = data['close'] > data['open']
            bearish_mask = data['close'] < data['open']
            
            bullish_bodies = pd.Series(0.0, index=data.index)
            bearish_bodies = pd.Series(0.0, index=data.index)
            
            bullish_bodies[bullish_mask] = body_size[bullish_mask]
            bearish_bodies[bearish_mask] = body_size[bearish_mask]
            
            return bullish_bodies, bearish_bodies
            
        except Exception as e:
            self.logger.error(f"Error calculating candle bodies: {e}")
            return pd.Series(0.0, index=data.index), pd.Series(0.0, index=data.index)
    
    def calculate_momentum_strength(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate bullish and bearish momentum strength over lookback period
        
        Args:
            data: OHLCV DataFrame with candle body data
            
        Returns:
            Dictionary with momentum indicators
        """
        try:
            # Calculate candle bodies
            bullish_bodies, bearish_bodies = self.calculate_candle_bodies(data)
            
            # Calculate rolling sums for momentum strength
            bullish_strength = bullish_bodies.rolling(window=self.lookback_period).sum()
            bearish_strength = bearish_bodies.rolling(window=self.lookback_period).sum()
            
            # Calculate momentum ratio (bullish / bearish)
            momentum_ratio = bullish_strength / (bearish_strength + 1e-8)  # Avoid division by zero
            
            # Calculate momentum difference
            momentum_diff = bullish_strength - bearish_strength
            
            # Calculate normalized momentum (0-1 scale)
            total_strength = bullish_strength + bearish_strength
            normalized_bullish = bullish_strength / (total_strength + 1e-8)
            normalized_bearish = bearish_strength / (total_strength + 1e-8)
            
            return {
                'bullish_strength': bullish_strength,
                'bearish_strength': bearish_strength,
                'momentum_ratio': momentum_ratio,
                'momentum_diff': momentum_diff,
                'normalized_bullish': normalized_bullish,
                'normalized_bearish': normalized_bearish,
                'bullish_bodies': bullish_bodies,
                'bearish_bodies': bearish_bodies
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum strength: {e}")
            return {}
    
    def calculate_trend_filter(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate trend filter (SMA or EMA)
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Trend line Series
        """
        try:
            if self.technical:
                if self.use_ema:
                    return self.technical.calculate_ema(data['close'], self.sma_period)
                else:
                    return self.technical.calculate_sma(data['close'], self.sma_period)
            else:
                # Fallback calculation if technical module not available
                if self.use_ema:
                    return data['close'].ewm(span=self.sma_period, adjust=False).mean()
                else:
                    return data['close'].rolling(window=self.sma_period).mean()
                
        except Exception as e:
            self.logger.error(f"Error calculating trend filter: {e}")
            return pd.Series(np.nan, index=data.index)
    
    def calculate_volume_filter(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate volume filter
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Boolean Series indicating when volume is above average
        """
        try:
            if not self.volume_filter:
                return pd.Series(True, index=data.index)
            
            avg_volume = data['volume'].rolling(window=self.volume_period).mean()
            return data['volume'] > avg_volume
            
        except Exception as e:
            self.logger.error(f"Error calculating volume filter: {e}")
            return pd.Series(True, index=data.index)
    
    def detect_momentum_crossover(self, momentum_data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Detect momentum crossover signals
        
        Args:
            momentum_data: Dictionary with momentum indicators
            
        Returns:
            Dictionary with crossover signals
        """
        try:
            bullish_strength = momentum_data['bullish_strength']
            bearish_strength = momentum_data['bearish_strength']
            
            # Detect crossovers
            bullish_cross_above = (
                (bullish_strength > bearish_strength) & 
                (bullish_strength.shift(1) <= bearish_strength.shift(1))
            )
            
            bearish_cross_above = (
                (bearish_strength > bullish_strength) & 
                (bearish_strength.shift(1) <= bullish_strength.shift(1))
            )
            
            return {
                'bullish_crossover': bullish_cross_above,
                'bearish_crossover': bearish_cross_above
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting momentum crossover: {e}")
            return {}
    
    def calculate_confidence(self, momentum_data: Dict[str, pd.Series], 
                           trend_line: pd.Series, volume_ok: pd.Series) -> pd.Series:
        """
        Calculate signal confidence based on multiple factors
        
        Args:
            momentum_data: Momentum indicators
            trend_line: Trend filter line
            volume_ok: Volume filter
            
        Returns:
            Confidence Series (0-1)
        """
        try:
            confidence = pd.Series(0.0, index=trend_line.index)
            
            # Base confidence from momentum ratio
            momentum_ratio = momentum_data.get('momentum_ratio', pd.Series(1.0, index=trend_line.index))
            normalized_ratio = np.clip(momentum_ratio / 3.0, 0, 1)  # Normalize to 0-1
            
            # Adjust confidence based on momentum strength
            total_strength = momentum_data.get('bullish_strength', pd.Series(0, index=trend_line.index)) + \
                           momentum_data.get('bearish_strength', pd.Series(0, index=trend_line.index))
            strength_factor = np.clip(total_strength / total_strength.rolling(50).mean(), 0.5, 2.0)
            strength_factor = (strength_factor - 0.5) / 1.5  # Normalize to 0-1
            
            # Combine factors
            confidence = (normalized_ratio * 0.6 + strength_factor * 0.4)
            
            # Apply volume filter
            confidence = confidence * volume_ok.astype(float)
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return pd.Series(0.5, index=trend_line.index)
    
    def analyze(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Analyze market data and calculate all indicators
        
        Args:
            data: OHLCV DataFrame
            symbol: Trading pair symbol
            
        Returns:
            Analysis results dictionary
        """
        try:
            if len(data) < max(self.lookback_period, self.sma_period) + 10:
                self.logger.warning(f"Insufficient data for {symbol}: {len(data)} candles")
                return {}
            
            # Calculate momentum indicators
            momentum_data = self.calculate_momentum_strength(data)
            if not momentum_data:
                return {}
            
            # Calculate trend filter
            trend_line = self.calculate_trend_filter(data)
            
            # Calculate volume filter
            volume_ok = self.calculate_volume_filter(data)
            
            # Detect crossovers
            crossovers = self.detect_momentum_crossover(momentum_data)
            
            # Calculate confidence
            confidence = self.calculate_confidence(momentum_data, trend_line, volume_ok)
            
            # Store momentum history for this symbol
            self.momentum_history[symbol] = {
                'timestamp': data.index[-1],
                'bullish_strength': momentum_data['bullish_strength'].iloc[-1],
                'bearish_strength': momentum_data['bearish_strength'].iloc[-1],
                'momentum_ratio': momentum_data['momentum_ratio'].iloc[-1],
                'trend_line': trend_line.iloc[-1],
                'current_price': data['close'].iloc[-1]
            }
            
            return {
                'momentum_data': momentum_data,
                'trend_line': trend_line,
                'volume_ok': volume_ok,
                'crossovers': crossovers,
                'confidence': confidence,
                'symbol': symbol,
                'timestamp': data.index[-1]
            }
            
        except Exception as e:
            self.logger.error(f"Error in analysis for {symbol}: {e}")
            return {}
    
    def calculate_signal(self, symbol: str, data: pd.DataFrame, 
                        current_price: float) -> Tuple[str, Dict[str, Any]]:
        """
        Calculate trading signal - implements base class interface
        
        Args:
            symbol: Trading pair symbol
            data: OHLCV DataFrame
            current_price: Current asset price
            
        Returns:
            Tuple of (signal_string, signal_data_dict)
        """
        # Use existing generate_signals method
        result = self.generate_signals(data, symbol)
        
        # Convert signal format to match base class expectations
        signal_str = result['signal'].upper()  # Convert to uppercase
        if signal_str == 'BUY':
            signal_str = 'BUY'
        elif signal_str == 'SELL':
            signal_str = 'SELL'
        else:
            signal_str = 'HOLD'
        
        # Return in expected format
        return signal_str, result
    
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Generate trading signals based on candle momentum analysis
        
        Args:
            data: OHLCV DataFrame
            symbol: Trading pair symbol
            
        Returns:
            Signal dictionary
        """
        try:
            # Perform analysis
            analysis = self.analyze(data, symbol)
            if not analysis:
                return {'signal': 'hold', 'confidence': 0.0, 'metadata': {}}
            
            momentum_data = analysis['momentum_data']
            trend_line = analysis['trend_line']
            volume_ok = analysis['volume_ok']
            crossovers = analysis['crossovers']
            confidence = analysis['confidence']
            
            current_price = data['close'].iloc[-1]
            current_confidence = confidence.iloc[-1]
            
            # Get latest values
            bullish_strength = momentum_data['bullish_strength'].iloc[-1]
            bearish_strength = momentum_data['bearish_strength'].iloc[-1]
            momentum_ratio = momentum_data['momentum_ratio'].iloc[-1]
            trend_value = trend_line.iloc[-1]
            volume_filter_ok = volume_ok.iloc[-1]
            
            # Check crossover signals
            bullish_crossover = crossovers['bullish_crossover'].iloc[-1] if len(crossovers['bullish_crossover']) > 0 else False
            bearish_crossover = crossovers['bearish_crossover'].iloc[-1] if len(crossovers['bearish_crossover']) > 0 else False
            
            signal = 'hold'
            signal_strength = 0.0
            
            # Long signal conditions
            if (bullish_crossover and 
                current_price > trend_value and 
                momentum_ratio > self.min_momentum_ratio and
                volume_filter_ok and
                current_confidence >= self.min_confidence):
                
                signal = 'buy'
                signal_strength = min(current_confidence * (momentum_ratio / 2.0), 1.0)
                
                self.logger.info(f"LONG signal for {symbol}: momentum_ratio={momentum_ratio:.2f}, "
                               f"price_vs_trend={current_price/trend_value:.4f}, confidence={current_confidence:.2f}")
            
            # Short signal conditions
            elif (bearish_crossover and 
                  current_price < trend_value and 
                  momentum_ratio < (1.0 / self.min_momentum_ratio) and
                  volume_filter_ok and
                  current_confidence >= self.min_confidence):
                
                signal = 'sell'
                signal_strength = min(current_confidence * (1.0 / max(momentum_ratio, 0.1)), 1.0)
                
                self.logger.info(f"SHORT signal for {symbol}: momentum_ratio={momentum_ratio:.2f}, "
                               f"price_vs_trend={current_price/trend_value:.4f}, confidence={current_confidence:.2f}")
            
            # Exit conditions for existing positions
            elif symbol in self.last_signals:
                last_signal = self.last_signals[symbol]
                
                # Exit long position
                if (last_signal.get('signal') == 'buy' and 
                    (bearish_crossover or momentum_ratio < 1.0)):
                    signal = 'sell'
                    signal_strength = current_confidence * 0.8
                    self.logger.info(f"EXIT LONG for {symbol}: bearish crossover or momentum weakening")
                
                # Exit short position
                elif (last_signal.get('signal') == 'sell' and 
                      (bullish_crossover or momentum_ratio > 1.0)):
                    signal = 'buy'
                    signal_strength = current_confidence * 0.8
                    self.logger.info(f"EXIT SHORT for {symbol}: bullish crossover or momentum strengthening")
            
            # Create signal metadata
            metadata = {
                'bullish_strength': float(bullish_strength),
                'bearish_strength': float(bearish_strength),
                'momentum_ratio': float(momentum_ratio),
                'trend_value': float(trend_value),
                'current_price': float(current_price),
                'price_vs_trend': float(current_price / trend_value) if trend_value != 0 else 1.0,
                'volume_ok': bool(volume_filter_ok),
                'crossover_bullish': bool(bullish_crossover),
                'crossover_bearish': bool(bearish_crossover),
                'lookback_period': self.lookback_period,
                'strategy': 'candle_momentum'
            }
            
            # Store last signal
            result = {
                'signal': signal,
                'confidence': float(signal_strength),
                'metadata': metadata
            }
            
            self.last_signals[symbol] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating signals for {symbol}: {e}")
            return {'signal': 'hold', 'confidence': 0.0, 'metadata': {'error': str(e)}}
    
    def calculate_position_size(self, signal: Dict[str, Any], account_balance: float, 
                              current_price: float) -> float:
        """
        Calculate position size based on signal confidence and risk management
        
        Args:
            signal: Signal dictionary from generate_signals
            account_balance: Current account balance
            current_price: Current asset price
            
        Returns:
            Position size in base currency
        """
        try:
            if signal['signal'] == 'hold':
                return 0.0
            
            confidence = signal['confidence']
            momentum_ratio = signal['metadata'].get('momentum_ratio', 1.0)
            
            # Base position size from parent class
            base_size = super().calculate_position_size(signal, account_balance, current_price)
            
            # Adjust based on momentum strength
            momentum_multiplier = min(max(momentum_ratio / 2.0, 0.5), 2.0) if signal['signal'] == 'buy' else 1.0
            if signal['signal'] == 'sell':
                momentum_multiplier = min(max(2.0 / max(momentum_ratio, 0.1), 0.5), 2.0)
            
            # Apply confidence and momentum adjustments
            adjusted_size = base_size * confidence * momentum_multiplier
            
            self.logger.debug(f"Position size calculation: base={base_size}, confidence={confidence}, "
                            f"momentum_mult={momentum_multiplier}, final={adjusted_size}")
            
            return adjusted_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def get_momentum_visualization_data(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get data for momentum visualization
        
        Args:
            symbol: Trading pair symbol
            data: OHLCV DataFrame
            
        Returns:
            Visualization data dictionary
        """
        try:
            analysis = self.analyze(data, symbol)
            if not analysis:
                return {}
            
            momentum_data = analysis['momentum_data']
            trend_line = analysis['trend_line']
            
            return {
                'timestamp': data.index,
                'price': data['close'],
                'bullish_strength': momentum_data['bullish_strength'],
                'bearish_strength': momentum_data['bearish_strength'],
                'momentum_ratio': momentum_data['momentum_ratio'],
                'trend_line': trend_line,
                'bullish_bodies': momentum_data['bullish_bodies'],
                'bearish_bodies': momentum_data['bearish_bodies'],
                'crossovers': analysis['crossovers'],
                'confidence': analysis['confidence']
            }
            
        except Exception as e:
            self.logger.error(f"Error getting visualization data for {symbol}: {e}")
            return {}
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get strategy information and current state
        
        Returns:
            Strategy info dictionary
        """
        return {
            'name': 'Candle Momentum Strategy',
            'version': '1.0.0',
            'parameters': {
                'lookback_period': self.lookback_period,
                'sma_period': self.sma_period,
                'use_ema': self.use_ema,
                'volume_filter': self.volume_filter,
                'min_momentum_ratio': self.min_momentum_ratio,
                'multi_timeframe': self.multi_timeframe,
                'min_confidence': self.min_confidence
            },
            'active_symbols': list(self.momentum_history.keys()),
            'last_update': datetime.now().isoformat(),
            'momentum_history': self.momentum_history
        }