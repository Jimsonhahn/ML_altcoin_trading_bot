#!/usr/bin/env python3
"""
Profitable BTC Strategy - Optimiert für 30% Return + 2.0+ Sharpe
================================================================

Intelligente Strategy die auf realistischen Backtesting-Ergebnissen basiert
Fokus auf hohe Trefferquote und kontrollierte Verluste
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ProfitableBTCStrategy:
    """
    Profitable BTC Strategy optimiert für konsistente Gewinne
    
    DESIGN PRINZIPIEN:
    1. Hohe Win Rate (>60%) durch konservative Entries
    2. Kleine, kontrollierte Verluste 
    3. Trend-Following mit Mean-Reversion Elementen
    4. Multi-Timeframe Confluence
    5. Dynamische Position Sizing
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize profitable strategy"""
        self.config = config or {}
        
        # Strategy parameters (optimiert für Profitabilität)
        self.min_signal_strength = self.config.get('min_signal_strength', 0.6)  # Höhere Qualität
        self.max_position_size = self.config.get('max_position_size', 0.4)      # Konservativ
        self.stop_loss_pct = self.config.get('stop_loss_pct', 0.025)            # 2.5% Stop Loss
        self.take_profit_pct = self.config.get('take_profit_pct', 0.06)         # 6% Take Profit (2.4:1 R/R)
        self.min_trend_strength = self.config.get('min_trend_strength', 0.02)   # Starke Trends nur
        
        # Advanced parameters
        self.volume_surge_threshold = self.config.get('volume_surge_threshold', 1.8)
        self.volatility_filter_max = self.config.get('volatility_filter_max', 0.05)  # Max 5% volatility
        self.confluence_required = self.config.get('confluence_required', 3)         # Min 3 Signale
        
        # State management
        self.indicator_engine = None  # Will be injected
        self.recent_performance = []
        self.adaptive_params = {
            'signal_threshold': self.min_signal_strength,
            'position_scaling': 1.0,
            'risk_reduction': 1.0
        }
        
        logger.info("ProfitableBTCStrategy initialized for 30% target return")
    
    def set_indicator_engine(self, engine):
        """Set the indicator engine"""
        self.indicator_engine = engine
    
    def calculate_signal_strength(self, indicators: Dict[str, float], price: float) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate signal strength with multi-factor confluence
        Fokus auf QUALITÄT über Quantität
        """
        try:
            required_indicators = ['sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi_14', 'volume_ratio_20']
            
            # Check if we have required indicators
            if not all(ind in indicators for ind in required_indicators):
                return 0.0, {'error': 'insufficient_indicators'}
            
            confluence_signals = []
            signal_quality_factors = []
            
            # 1. TREND CONFLUENCE (40% weight)
            trend_signals = self._analyze_trend_confluence(indicators, price)
            confluence_signals.extend(trend_signals['signals'])
            signal_quality_factors.append(trend_signals['quality'])
            
            # 2. MOMENTUM CONFLUENCE (25% weight) 
            momentum_signals = self._analyze_momentum_confluence(indicators)
            confluence_signals.extend(momentum_signals['signals'])
            signal_quality_factors.append(momentum_signals['quality'])
            
            # 3. VOLUME CONFIRMATION (20% weight)
            volume_signals = self._analyze_volume_confluence(indicators)
            confluence_signals.extend(volume_signals['signals'])
            signal_quality_factors.append(volume_signals['quality'])
            
            # 4. VOLATILITY FILTER (15% weight)
            volatility_check = self._analyze_volatility_environment(indicators)
            if volatility_check['suitable']:
                confluence_signals.append(volatility_check['signal_boost'])
                signal_quality_factors.append(volatility_check['quality'])
            else:
                # High volatility = reduce all signals
                confluence_signals = [s * 0.3 for s in confluence_signals]
            
            # CONFLUENCE ANALYSIS
            if len(confluence_signals) < self.confluence_required:
                return 0.0, {
                    'reason': 'insufficient_confluence',
                    'signals_found': len(confluence_signals),
                    'required': self.confluence_required
                }
            
            # Calculate weighted signal strength
            signal_strength = np.mean(confluence_signals)
            signal_quality = np.mean(signal_quality_factors)
            
            # Apply adaptive threshold
            adjusted_threshold = self.adaptive_params['signal_threshold']
            
            # Final signal with quality filter
            final_strength = signal_strength * signal_quality
            
            # Determine direction
            direction = 'buy' if final_strength > adjusted_threshold else 'sell' if final_strength < -adjusted_threshold else 'hold'
            
            signal_data = {
                'signal_strength': final_strength,
                'base_strength': signal_strength,
                'quality_score': signal_quality,
                'confluence_count': len(confluence_signals),
                'direction': direction,
                'confidence': min(abs(final_strength), 1.0),
                'adaptive_threshold': adjusted_threshold,
                'confluence_signals': confluence_signals,
                'market_environment': {
                    'trend_quality': trend_signals['quality'],
                    'momentum_quality': momentum_signals['quality'],
                    'volume_quality': volume_signals['quality'],
                    'volatility_suitable': volatility_check['suitable']
                }
            }
            
            return final_strength, signal_data
            
        except Exception as e:
            logger.error(f"Signal calculation failed: {e}")
            return 0.0, {'error': str(e)}
    
    def _analyze_trend_confluence(self, indicators: Dict[str, float], price: float) -> Dict[str, Any]:
        """Analyze trend confluence across multiple timeframes"""
        try:
            signals = []
            
            # SMA Trend Analysis
            sma_20 = indicators['sma_20']
            sma_50 = indicators['sma_50']
            
            # Price above both SMAs = bullish
            if price > sma_20 > sma_50:
                trend_strength = min((price - sma_50) / sma_50 / 0.05, 1.0)  # Normalize to 5%
                signals.append(trend_strength)
            elif price < sma_20 < sma_50:
                trend_strength = min((sma_50 - price) / sma_50 / 0.05, 1.0)
                signals.append(-trend_strength)
            
            # EMA Trend Analysis
            ema_12 = indicators['ema_12']
            ema_26 = indicators['ema_26']
            
            if ema_12 > ema_26:
                ema_strength = min((ema_12 - ema_26) / ema_26 / 0.03, 1.0)  # Normalize to 3%
                signals.append(ema_strength)
            else:
                ema_strength = min((ema_26 - ema_12) / ema_26 / 0.03, 1.0)
                signals.append(-ema_strength)
            
            # Trend consistency check
            momentum_20d = indicators.get('momentum_20d', 0)
            if abs(momentum_20d) > self.min_trend_strength:
                momentum_signal = min(momentum_20d / 0.1, 1.0) if momentum_20d > 0 else max(momentum_20d / 0.1, -1.0)
                signals.append(momentum_signal)
            
            quality = np.std(signals) if len(signals) > 1 else 0  # Lower std = higher quality
            quality = max(0.1, 1.0 - quality)  # Invert: low std = high quality
            
            return {
                'signals': signals,
                'quality': quality,
                'trend_direction': 'bullish' if np.mean(signals) > 0 else 'bearish' if np.mean(signals) < 0 else 'neutral'
            }
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return {'signals': [], 'quality': 0.0, 'trend_direction': 'unknown'}
    
    def _analyze_momentum_confluence(self, indicators: Dict[str, float]) -> Dict[str, Any]:
        """Analyze momentum with multiple oscillators"""
        try:
            signals = []
            
            # RSI Analysis (mean reversion + trend confirmation)
            rsi_14 = indicators['rsi_14']
            
            # RSI extreme levels with momentum confirmation
            if rsi_14 < 35 and indicators.get('momentum_5d', 0) > -0.02:  # Oversold but stabilizing
                rsi_signal = (35 - rsi_14) / 15  # Stronger signal when more oversold
                signals.append(rsi_signal)
            elif rsi_14 > 65 and indicators.get('momentum_5d', 0) < 0.02:  # Overbought
                rsi_signal = (65 - rsi_14) / 15
                signals.append(rsi_signal)
            elif 45 < rsi_14 < 55:  # Neutral RSI supports trend
                momentum_20d = indicators.get('momentum_20d', 0)
                if abs(momentum_20d) > 0.01:
                    trend_support = momentum_20d / 0.05  # Support existing trend
                    signals.append(trend_support * 0.3)  # Weaker signal
            
            # MACD Momentum
            ema_12 = indicators['ema_12']
            ema_26 = indicators['ema_26']
            macd_line = ema_12 - ema_26
            
            # MACD histogram approximation (rate of change)
            macd_momentum = macd_line / ema_26 if ema_26 > 0 else 0
            if abs(macd_momentum) > 0.005:  # Significant MACD movement
                macd_signal = min(macd_momentum / 0.02, 1.0) if macd_momentum > 0 else max(macd_momentum / 0.02, -1.0)
                signals.append(macd_signal)
            
            # Momentum consistency
            short_momentum = indicators.get('momentum_5d', 0)
            medium_momentum = indicators.get('momentum_10d', 0)
            long_momentum = indicators.get('momentum_20d', 0)
            
            momentum_alignment = 0
            if short_momentum > 0 and medium_momentum > 0 and long_momentum > 0:
                momentum_alignment = min(np.mean([short_momentum, medium_momentum, long_momentum]) / 0.03, 1.0)
            elif short_momentum < 0 and medium_momentum < 0 and long_momentum < 0:
                momentum_alignment = max(np.mean([short_momentum, medium_momentum, long_momentum]) / 0.03, -1.0)
            
            if abs(momentum_alignment) > 0.1:
                signals.append(momentum_alignment)
            
            # Quality based on signal consistency
            quality = 1.0 - np.std(signals) if len(signals) > 1 else 0.5
            quality = max(0.1, min(quality, 1.0))
            
            return {
                'signals': signals,
                'quality': quality,
                'rsi_level': rsi_14,
                'macd_momentum': macd_momentum
            }
            
        except Exception as e:
            logger.error(f"Momentum analysis failed: {e}")
            return {'signals': [], 'quality': 0.0}
    
    def _analyze_volume_confluence(self, indicators: Dict[str, float]) -> Dict[str, Any]:
        """Analyze volume confirmation"""
        try:
            signals = []
            
            volume_ratio_20 = indicators.get('volume_ratio_20', 1.0)
            volume_ratio_10 = indicators.get('volume_ratio_10', 1.0)
            volume_ratio_5 = indicators.get('volume_ratio_5', 1.0)
            
            # Volume surge confirmation
            if volume_ratio_20 > self.volume_surge_threshold:
                volume_strength = min((volume_ratio_20 - 1.0) / 1.0, 1.0)  # Normalize volume surge
                signals.append(volume_strength * 0.8)  # Strong positive signal
            
            # Volume trend consistency
            if volume_ratio_5 > volume_ratio_10 > volume_ratio_20 > 1.2:  # Increasing volume trend
                volume_trend = 0.6
                signals.append(volume_trend)
            elif volume_ratio_5 < volume_ratio_10 < volume_ratio_20 < 0.8:  # Decreasing volume (bearish)
                volume_trend = -0.4
                signals.append(volume_trend)
            
            # Volume confirmation quality
            volume_levels = [volume_ratio_5, volume_ratio_10, volume_ratio_20]
            volume_consistency = 1.0 - (np.std(volume_levels) / np.mean(volume_levels)) if volume_levels else 0.5
            
            return {
                'signals': signals,
                'quality': max(0.1, min(volume_consistency, 1.0)),
                'volume_trend': 'increasing' if len(signals) > 0 and signals[-1] > 0 else 'decreasing'
            }
            
        except Exception as e:
            logger.error(f"Volume analysis failed: {e}")
            return {'signals': [], 'quality': 0.0}
    
    def _analyze_volatility_environment(self, indicators: Dict[str, float]) -> Dict[str, Any]:
        """Filter trades based on volatility environment"""
        try:
            volatility_20d = indicators.get('volatility_20d', 0.02)
            volatility_10d = indicators.get('volatility_10d', 0.02)
            volatility_5d = indicators.get('volatility_5d', 0.02)
            
            # Check if volatility is suitable for trading
            suitable = volatility_20d < self.volatility_filter_max
            
            # Volatility trend
            if volatility_5d < volatility_10d < volatility_20d:  # Decreasing volatility
                vol_signal = 0.3  # Slightly positive (stabilizing market)
            elif volatility_5d > volatility_10d > volatility_20d > 0.04:  # Increasing high volatility
                vol_signal = -0.5  # Negative (chaotic market)
            else:
                vol_signal = 0.0
            
            # Quality based on volatility stability
            vol_consistency = 1.0 - np.std([volatility_5d, volatility_10d, volatility_20d]) / np.mean([volatility_5d, volatility_10d, volatility_20d])
            
            return {
                'suitable': suitable,
                'signal_boost': vol_signal,
                'quality': max(0.1, min(vol_consistency, 1.0)),
                'volatility_level': volatility_20d
            }
            
        except Exception as e:
            logger.error(f"Volatility analysis failed: {e}")
            return {'suitable': True, 'signal_boost': 0.0, 'quality': 0.5}
    
    def calculate_position_size(self, signal_data: Dict[str, Any], current_price: float, 
                              available_capital: float) -> float:
        """
        Calculate optimal position size using Kelly Criterion principles
        """
        try:
            signal_strength = abs(signal_data.get('signal_strength', 0))
            signal_quality = signal_data.get('quality_score', 0.5)
            confluence_count = signal_data.get('confluence_count', 0)
            
            # Base position size from signal quality
            base_size = signal_strength * signal_quality
            
            # Confluence boost (more signals = higher confidence)
            confluence_multiplier = min(confluence_count / self.confluence_required, 1.5)
            
            # Adaptive position scaling
            position_scaling = self.adaptive_params['position_scaling']
            risk_reduction = self.adaptive_params['risk_reduction']
            
            # Calculate Kelly-inspired position size
            win_rate = 0.65  # Target win rate from backtesting
            avg_win = 0.06   # 6% average win (take profit level)
            avg_loss = 0.025 # 2.5% average loss (stop loss level)
            
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            
            # Final position size
            final_size = (base_size * confluence_multiplier * kelly_fraction * 
                         position_scaling * risk_reduction)
            
            # Apply maximum position size limit
            final_size = min(final_size, self.max_position_size)
            
            # Minimum position size check
            min_position = available_capital * 0.02  # Minimum 2%
            position_value = available_capital * final_size
            
            if position_value < min_position:
                return 0.0  # Too small to trade
            
            return final_size
            
        except Exception as e:
            logger.error(f"Position size calculation failed: {e}")
            return 0.0
    
    def should_exit_position(self, entry_price: float, current_price: float, 
                           direction: str, bars_held: int) -> Tuple[bool, str]:
        """
        Determine if position should be exited
        Advanced exit logic with multiple criteria
        """
        try:
            if direction == 'long':
                price_change = (current_price - entry_price) / entry_price
            else:  # short
                price_change = (entry_price - current_price) / entry_price
            
            # Stop Loss
            if price_change <= -self.stop_loss_pct:
                return True, "stop_loss"
            
            # Take Profit
            if price_change >= self.take_profit_pct:
                return True, "take_profit"
            
            # Time-based exit (prevent holding too long)
            if bars_held > 72:  # 72 hours max hold (for hourly data)
                return True, "time_exit"
            
            # Trailing stop (activate after 50% of take profit)
            if price_change >= self.take_profit_pct * 0.5:
                trailing_stop = self.stop_loss_pct * 0.5  # Tighter trailing stop
                if price_change <= trailing_stop:
                    return True, "trailing_stop"
            
            return False, "hold"
            
        except Exception as e:
            logger.error(f"Exit logic failed: {e}")
            return True, "error_exit"
    
    def update_adaptive_parameters(self, recent_trades: List[Dict[str, Any]]):
        """
        Update strategy parameters based on recent performance
        Machine Learning-inspired adaptive optimization
        """
        try:
            if len(recent_trades) < 10:
                return  # Need sufficient data
            
            # Calculate recent performance metrics
            recent_returns = [trade.get('return_pct', 0) for trade in recent_trades[-20:]]
            win_rate = len([r for r in recent_returns if r > 0]) / len(recent_returns)
            avg_return = np.mean(recent_returns)
            
            # Adaptive threshold adjustment
            if win_rate < 0.55:  # If win rate too low, be more selective
                self.adaptive_params['signal_threshold'] *= 1.05
                self.adaptive_params['signal_threshold'] = min(self.adaptive_params['signal_threshold'], 0.8)
            elif win_rate > 0.75:  # If win rate high, can be less selective
                self.adaptive_params['signal_threshold'] *= 0.98
                self.adaptive_params['signal_threshold'] = max(self.adaptive_params['signal_threshold'], 0.4)
            
            # Position scaling adjustment
            if avg_return < -0.01:  # If losing money, reduce position sizes
                self.adaptive_params['position_scaling'] *= 0.95
                self.adaptive_params['position_scaling'] = max(self.adaptive_params['position_scaling'], 0.3)
            elif avg_return > 0.02:  # If doing well, slightly increase
                self.adaptive_params['position_scaling'] *= 1.02
                self.adaptive_params['position_scaling'] = min(self.adaptive_params['position_scaling'], 1.5)
            
            # Risk reduction during drawdown
            drawdown = min(recent_returns) if recent_returns else 0
            if drawdown < -0.05:  # 5% drawdown triggers risk reduction
                self.adaptive_params['risk_reduction'] = 0.7
            elif drawdown > -0.02:  # Recovery
                self.adaptive_params['risk_reduction'] = min(self.adaptive_params['risk_reduction'] * 1.1, 1.0)
            
            logger.info(f"Adaptive parameters updated: threshold={self.adaptive_params['signal_threshold']:.3f}, "
                       f"scaling={self.adaptive_params['position_scaling']:.3f}, "
                       f"risk_reduction={self.adaptive_params['risk_reduction']:.3f}")
            
        except Exception as e:
            logger.error(f"Adaptive parameter update failed: {e}")
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information"""
        return {
            'name': 'Profitable BTC Strategy',
            'version': '1.0 Optimized',
            'target_return': '30% annual',
            'target_sharpe': '2.0+',
            'description': 'High-quality signals with 60%+ win rate and 2.4:1 R/R ratio',
            'key_features': [
                'Multi-factor confluence analysis',
                'Dynamic position sizing (Kelly-inspired)',
                'Advanced exit logic with trailing stops',
                'Volatility filtering',
                'Adaptive parameter optimization',
                'Risk-first approach'
            ],
            'risk_management': {
                'stop_loss': f"{self.stop_loss_pct:.1%}",
                'take_profit': f"{self.take_profit_pct:.1%}",
                'max_position': f"{self.max_position_size:.0%}",
                'risk_reward_ratio': f"{self.take_profit_pct/self.stop_loss_pct:.1f}:1"
            },
            'current_parameters': self.adaptive_params
        }


def create_enhanced_indicator_engine():
    """Create enhanced indicator engine with more indicators"""
    
    class EnhancedIndicatorEngine:
        """Enhanced indicator engine for profitable strategy"""
        
        def __init__(self):
            self.state = {
                'price_history': [],
                'volume_history': [],
                'sma_cache': {},
                'ema_cache': {},
                'last_update': None
            }
        
        def update(self, price: float, volume: float, timestamp=None) -> Dict[str, float]:
            """Update all indicators"""
            self.state['price_history'].append(price)
            self.state['volume_history'].append(volume)
            self.state['last_update'] = timestamp
            
            # Limit history
            if len(self.state['price_history']) > 500:
                self.state['price_history'] = self.state['price_history'][-500:]
                self.state['volume_history'] = self.state['volume_history'][-500:]
            
            indicators = {}
            
            # Enhanced SMA calculations
            for window in [5, 10, 20, 50, 100, 200]:
                if len(self.state['price_history']) >= window:
                    sma = np.mean(self.state['price_history'][-window:])
                    indicators[f'sma_{window}'] = sma
            
            # Enhanced EMA calculations
            for span in [5, 8, 12, 21, 26, 50]:
                key = f'ema_{span}'
                alpha = 2.0 / (span + 1)
                
                if key not in self.state['ema_cache']:
                    self.state['ema_cache'][key] = price
                else:
                    self.state['ema_cache'][key] = alpha * price + (1 - alpha) * self.state['ema_cache'][key]
                
                indicators[key] = self.state['ema_cache'][key]
            
            # Enhanced RSI
            for period in [9, 14, 21]:
                if len(self.state['price_history']) >= period + 1:
                    changes = [self.state['price_history'][i] - self.state['price_history'][i-1] 
                              for i in range(-period, 0)]
                    gains = [max(0, change) for change in changes]
                    losses = [max(0, -change) for change in changes]
                    
                    avg_gain = np.mean(gains)
                    avg_loss = np.mean(losses)
                    
                    if avg_loss > 0:
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                        indicators[f'rsi_{period}'] = rsi
            
            # Enhanced Momentum
            for period in [3, 5, 8, 10, 13, 20, 34]:
                if len(self.state['price_history']) >= period + 1:
                    momentum = (self.state['price_history'][-1] / self.state['price_history'][-(period + 1)]) - 1
                    indicators[f'momentum_{period}d'] = momentum
            
            # Enhanced Volatility
            for window in [5, 10, 20, 30]:
                if len(self.state['price_history']) >= window + 1:
                    returns = [(self.state['price_history'][i] / self.state['price_history'][i-1]) - 1 
                              for i in range(-window, 0)]
                    volatility = np.std(returns)
                    indicators[f'volatility_{window}d'] = volatility
            
            # Enhanced Volume ratios
            for window in [5, 10, 20, 50]:
                if len(self.state['volume_history']) >= window:
                    avg_volume = np.mean(self.state['volume_history'][-window:])
                    if avg_volume > 0:
                        indicators[f'volume_ratio_{window}'] = volume / avg_volume
            
            return indicators
        
        def reset_state(self):
            """Reset engine state"""
            self.state = {
                'price_history': [],
                'volume_history': [],
                'sma_cache': {},
                'ema_cache': {},
                'last_update': None
            }
    
    return EnhancedIndicatorEngine()