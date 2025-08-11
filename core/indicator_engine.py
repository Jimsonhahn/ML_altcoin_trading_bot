#!/usr/bin/env python3
"""
IndicatorEngine - Event-Driven Technical Indicator Calculator
=============================================================

Eliminates lookahead bias with proper incremental state management.
All indicators are calculated in real-time as new data arrives.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class IndicatorEngine:
    """
    Event-driven indicator engine that prevents lookahead bias
    by maintaining state and calculating indicators incrementally
    """
    
    def __init__(self):
        """Initialize indicator engine state"""
        self.state = {
            'sma_windows': {},
            'ema_values': {},
            'rsi_state': {},
            'macd_state': {},
            'bb_state': {},
            'volume_state': {},
            'momentum_history': {},
            'volatility_state': {},
            'price_history': [],
            'volume_history': [],
            'last_timestamp': None
        }
        
        self.initialized_windows = set()
        logger.info("IndicatorEngine initialized with clean state")
    
    def update(self, price: float, volume: float, timestamp: datetime = None) -> Dict[str, float]:
        """
        Update all indicators with new price/volume data
        
        Args:
            price: Current price
            volume: Current volume
            timestamp: Current timestamp (optional)
            
        Returns:
            Dict of current indicator values
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Update price/volume history
        self.state['price_history'].append(price)
        self.state['volume_history'].append(volume)
        self.state['last_timestamp'] = timestamp
        
        # Limit history size to prevent memory issues
        max_history = 500
        if len(self.state['price_history']) > max_history:
            self.state['price_history'] = self.state['price_history'][-max_history:]
            self.state['volume_history'] = self.state['volume_history'][-max_history:]
        
        # Calculate all indicators incrementally
        indicators = {}
        
        # Simple Moving Averages
        indicators.update(self._update_sma_indicators(price))
        
        # Exponential Moving Averages
        indicators.update(self._update_ema_indicators(price))
        
        # MACD indicators
        indicators.update(self._update_macd_indicators())
        
        # RSI indicators
        indicators.update(self._update_rsi_indicators(price))
        
        # Momentum indicators
        indicators.update(self._update_momentum_indicators())
        
        # Volatility indicators
        indicators.update(self._update_volatility_indicators())
        
        # Volume indicators
        indicators.update(self._update_volume_indicators(volume))
        
        # Bollinger Bands
        indicators.update(self._update_bollinger_bands())
        
        return indicators
    
    def _update_sma_indicators(self, price: float) -> Dict[str, float]:
        """Update Simple Moving Averages incrementally"""
        indicators = {}
        windows = [5, 8, 13, 20, 21, 34, 50, 89, 144, 200]
        
        for window in windows:
            key = f'sma_{window}'
            
            if key not in self.state['sma_windows']:
                self.state['sma_windows'][key] = []
            
            # Add current price
            self.state['sma_windows'][key].append(price)
            
            # Maintain window size
            if len(self.state['sma_windows'][key]) > window:
                self.state['sma_windows'][key] = self.state['sma_windows'][key][-window:]
            
            # Calculate SMA if we have enough data
            if len(self.state['sma_windows'][key]) >= window:
                indicators[key] = np.mean(self.state['sma_windows'][key])
            
        return indicators
    
    def _update_ema_indicators(self, price: float) -> Dict[str, float]:
        """Update Exponential Moving Averages incrementally"""
        indicators = {}
        spans = [5, 8, 12, 13, 21, 26, 34, 50, 89]
        
        for span in spans:
            key = f'ema_{span}'
            alpha = 2.0 / (span + 1)
            
            if key not in self.state['ema_values']:
                # Initialize with first price
                self.state['ema_values'][key] = price
                indicators[key] = price
            else:
                # Update EMA incrementally
                prev_ema = self.state['ema_values'][key]
                new_ema = alpha * price + (1 - alpha) * prev_ema
                self.state['ema_values'][key] = new_ema
                indicators[key] = new_ema
        
        return indicators
    
    def _update_macd_indicators(self) -> Dict[str, float]:
        """Update MACD indicators using EMA values"""
        indicators = {}
        
        # MACD configurations: (fast, slow, signal_span)
        macd_configs = [(5, 13, 8), (8, 21, 5), (12, 26, 9), (21, 50, 9)]
        
        for fast, slow, signal_span in macd_configs:
            fast_key = f'ema_{fast}'
            slow_key = f'ema_{slow}'
            macd_key = f'macd_{fast}_{slow}'
            signal_key = f'{macd_key}_signal'
            hist_key = f'{macd_key}_histogram'
            
            # Check if we have the required EMAs
            if (fast_key in self.state['ema_values'] and 
                slow_key in self.state['ema_values']):
                
                # Calculate MACD line
                macd_value = self.state['ema_values'][fast_key] - self.state['ema_values'][slow_key]
                indicators[macd_key] = macd_value
                
                # Calculate signal line (EMA of MACD)
                signal_alpha = 2.0 / (signal_span + 1)
                
                if signal_key not in self.state['macd_state']:
                    self.state['macd_state'][signal_key] = macd_value
                    indicators[signal_key] = macd_value
                else:
                    prev_signal = self.state['macd_state'][signal_key]
                    new_signal = signal_alpha * macd_value + (1 - signal_alpha) * prev_signal
                    self.state['macd_state'][signal_key] = new_signal
                    indicators[signal_key] = new_signal
                
                # Calculate histogram
                indicators[hist_key] = macd_value - indicators[signal_key]
        
        return indicators
    
    def _update_rsi_indicators(self, price: float) -> Dict[str, float]:
        """Update RSI indicators incrementally"""
        indicators = {}
        periods = [7, 9, 14, 21, 25]
        
        # Calculate price change
        if len(self.state['price_history']) >= 2:
            price_change = price - self.state['price_history'][-2]
            
            for period in periods:
                key = f'rsi_{period}'
                
                if key not in self.state['rsi_state']:
                    self.state['rsi_state'][key] = {
                        'gains': [],
                        'losses': [],
                        'avg_gain': 0,
                        'avg_loss': 0
                    }
                
                rsi_state = self.state['rsi_state'][key]
                
                # Add current gain/loss
                if price_change > 0:
                    rsi_state['gains'].append(price_change)
                    rsi_state['losses'].append(0)
                else:
                    rsi_state['gains'].append(0)
                    rsi_state['losses'].append(abs(price_change))
                
                # Maintain period length
                if len(rsi_state['gains']) > period:
                    rsi_state['gains'] = rsi_state['gains'][-period:]
                    rsi_state['losses'] = rsi_state['losses'][-period:]
                
                # Calculate RSI if we have enough data
                if len(rsi_state['gains']) >= period:
                    avg_gain = np.mean(rsi_state['gains'])
                    avg_loss = np.mean(rsi_state['losses'])
                    
                    if avg_loss == 0:
                        indicators[key] = 100.0
                    else:
                        rs = avg_gain / avg_loss
                        indicators[key] = 100.0 - (100.0 / (1.0 + rs))
        
        return indicators
    
    def _update_momentum_indicators(self) -> Dict[str, float]:
        """Update momentum indicators"""
        indicators = {}
        periods = [3, 5, 8, 13, 20, 34, 50]
        
        if len(self.state['price_history']) > 1:
            current_price = self.state['price_history'][-1]
            
            for period in periods:
                if len(self.state['price_history']) > period:
                    past_price = self.state['price_history'][-(period + 1)]
                    momentum = (current_price / past_price) - 1.0
                    indicators[f'momentum_{period}d'] = momentum
        
        return indicators
    
    def _update_volatility_indicators(self) -> Dict[str, float]:
        """Update volatility indicators"""
        indicators = {}
        windows = [5, 8, 13, 20, 34, 50]
        
        if len(self.state['price_history']) >= 2:
            # Calculate returns
            returns = []
            for i in range(1, len(self.state['price_history'])):
                ret = (self.state['price_history'][i] / self.state['price_history'][i-1]) - 1.0
                returns.append(ret)
            
            for window in windows:
                if len(returns) >= window:
                    recent_returns = returns[-window:]
                    volatility = np.std(recent_returns)
                    indicators[f'volatility_{window}d'] = volatility
        
        return indicators
    
    def _update_volume_indicators(self, volume: float) -> Dict[str, float]:
        """Update volume indicators"""
        indicators = {}
        windows = [5, 10, 20, 34, 50]
        
        for window in windows:
            key = f'volume_sma_{window}'
            
            if key not in self.state['volume_state']:
                self.state['volume_state'][key] = []
            
            # Add current volume
            self.state['volume_state'][key].append(volume)
            
            # Maintain window size
            if len(self.state['volume_state'][key]) > window:
                self.state['volume_state'][key] = self.state['volume_state'][key][-window:]
            
            # Calculate volume SMA and ratio
            if len(self.state['volume_state'][key]) >= window:
                volume_sma = np.mean(self.state['volume_state'][key])
                indicators[key] = volume_sma
                
                # Volume ratio
                if volume_sma > 0:
                    indicators[f'volume_ratio_{window}'] = volume / volume_sma
        
        return indicators
    
    def _update_bollinger_bands(self) -> Dict[str, float]:
        """Update Bollinger Bands"""
        indicators = {}
        periods = [10, 20, 34, 50]
        
        for period in periods:
            sma_key = f'sma_{period}'
            
            # We need SMA and enough price history
            if (sma_key in self.state['sma_windows'] and 
                len(self.state['sma_windows'][sma_key]) >= period):
                
                prices = self.state['sma_windows'][sma_key]
                sma_value = np.mean(prices)
                std_value = np.std(prices)
                
                indicators[f'bb_upper_{period}'] = sma_value + (2 * std_value)
                indicators[f'bb_lower_{period}'] = sma_value - (2 * std_value)
                
                # Bollinger Band position
                current_price = self.state['price_history'][-1]
                if std_value > 0:
                    bb_position = (current_price - indicators[f'bb_lower_{period}']) / (2 * std_value)
                    indicators[f'bb_position_{period}'] = bb_position
        
        return indicators
    
    def get_crossover_signals(self, indicators: Dict[str, float]) -> Dict[str, bool]:
        """
        Detect crossover signals between indicators
        
        Returns:
            Dict of crossover signals (True = bullish crossover, False = bearish crossover)
        """
        signals = {}
        
        # MACD crossovers
        macd_configs = [(5, 13), (8, 21), (12, 26), (21, 50)]
        
        for fast, slow in macd_configs:
            macd_key = f'macd_{fast}_{slow}'
            signal_key = f'{macd_key}_signal'
            
            if macd_key in indicators and signal_key in indicators:
                # Check if this is the first time we have both values
                crossover_key = f'{macd_key}_crossover'
                
                if crossover_key not in self.state['macd_state']:
                    self.state['macd_state'][crossover_key] = {
                        'prev_macd': indicators[macd_key],
                        'prev_signal': indicators[signal_key]
                    }
                else:
                    prev_state = self.state['macd_state'][crossover_key]
                    
                    # Detect crossover
                    prev_above = prev_state['prev_macd'] > prev_state['prev_signal']
                    curr_above = indicators[macd_key] > indicators[signal_key]
                    
                    if not prev_above and curr_above:
                        signals[f'{macd_key}_bullish_crossover'] = True
                    elif prev_above and not curr_above:
                        signals[f'{macd_key}_bearish_crossover'] = True
                    
                    # Update state
                    prev_state['prev_macd'] = indicators[macd_key]
                    prev_state['prev_signal'] = indicators[signal_key]
        
        return signals
    
    def reset_state(self):
        """Reset all indicator state (for testing or restart)"""
        self.state = {
            'sma_windows': {},
            'ema_values': {},
            'rsi_state': {},
            'macd_state': {},
            'bb_state': {},
            'volume_state': {},
            'momentum_history': {},
            'volatility_state': {},
            'price_history': [],
            'volume_history': [],
            'last_timestamp': None
        }
        self.initialized_windows = set()
        logger.info("IndicatorEngine state reset")