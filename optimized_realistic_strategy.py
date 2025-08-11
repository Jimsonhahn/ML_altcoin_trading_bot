#!/usr/bin/env python3
"""
Optimized Realistic Strategy
===========================
Enhanced version of RealisticTradingStrategy optimized for Tier 1 performance
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Tuple

# Import the base class
from realistic_crypto_backtest import RealisticTradingStrategy

logger = logging.getLogger(__name__)

class OptimizedRealisticStrategy(RealisticTradingStrategy):
    """Enhanced RealisticTradingStrategy optimized for Tier 1 performance"""
    
    def __init__(self):
        super().__init__()
        
        # Optimized parameters for Tier 1 performance based on debug analysis
        self.max_position_size = 0.12        # Increased from 0.08
        self.min_signal_strength = 0.08      # Realistic threshold based on debugging
        self.stop_loss_pct = 0.03           # Increased from 0.025 (looser stops)
        self.take_profit_pct = 0.08         # Increased from 0.05 (better R/R: 2.67:1)
        self.max_daily_trades = 5           # Increased from 2
        self.cooldown_hours = 0.5           # Very short cooldown for more opportunities
        
        # Enhanced parameters
        self.trend_lookback = 50            # Periods for trend analysis
        self.volume_multiplier = 0.9        # Reduced volume confirmation threshold
        self.volatility_threshold = 0.10    # Increased volatility threshold (more lenient)
        
        # Multi-timeframe enhancement
        self.momentum_periods = [5, 10, 20, 50]  # Multiple momentum periods
        
        # Adaptive parameters
        self.performance_window = 20        # Recent performance tracking
        self.win_rate_adjustment = True     # Adjust based on recent win rate
        
        # State tracking
        self.recent_trades = []            # Track recent performance
        self.regime_state = 'unknown'      # Market regime
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Enhanced indicator calculation"""
        
        if len(data) < 50:
            return {}
        
        recent_data = data.tail(100).copy()  # Use more data for calculations
        current_price = recent_data['close'].iloc[-1]
        
        indicators = {}
        
        # === TREND INDICATORS ===
        
        # Moving Averages
        indicators['sma_20'] = recent_data['close'].rolling(20).mean().iloc[-1]
        indicators['sma_50'] = recent_data['close'].rolling(50).mean().iloc[-1]
        indicators['ema_12'] = recent_data['close'].ewm(span=12).mean().iloc[-1]
        indicators['ema_26'] = recent_data['close'].ewm(span=26).mean().iloc[-1]
        
        # Trend Strength
        indicators['trend_strength'] = (current_price - indicators['sma_50']) / indicators['sma_50']
        
        # === MOMENTUM INDICATORS ===
        
        # RSI
        delta = recent_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs.iloc[-1]))
        
        # MACD
        indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
        indicators['macd_signal'] = pd.Series([indicators['macd']]).ewm(span=9).mean().iloc[0]
        indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
        
        # Multi-period Momentum
        for period in self.momentum_periods:
            if len(recent_data) >= period + 1:
                momentum = (current_price / recent_data['close'].iloc[-(period+1)]) - 1
                indicators[f'momentum_{period}'] = momentum
        
        # === VOLATILITY INDICATORS ===
        
        # ATR
        high_low = recent_data['high'] - recent_data['low']
        high_close = np.abs(recent_data['high'] - recent_data['close'].shift())
        low_close = np.abs(recent_data['low'] - recent_data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        indicators['atr'] = ranges.rolling(14).mean().iloc[-1]
        indicators['atr_pct'] = indicators['atr'] / current_price
        
        # Bollinger Bands
        bb_sma = recent_data['close'].rolling(20).mean()
        bb_std = recent_data['close'].rolling(20).std()
        indicators['bb_upper'] = (bb_sma + (bb_std * 2)).iloc[-1]
        indicators['bb_lower'] = (bb_sma - (bb_std * 2)).iloc[-1]
        indicators['bb_position'] = (current_price - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
        
        # === VOLUME INDICATORS ===
        
        if 'volume' in recent_data.columns:
            indicators['volume_sma'] = recent_data['volume'].rolling(20).mean().iloc[-1]
            indicators['volume_ratio'] = recent_data['volume'].iloc[-1] / indicators['volume_sma']
        else:
            # Simulated volume if not available
            indicators['volume_ratio'] = 1.0
        
        # === MARKET REGIME DETECTION ===
        
        # Detect current market regime
        volatility = recent_data['close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(365)
        trend_direction = 1 if indicators['trend_strength'] > 0.02 else (-1 if indicators['trend_strength'] < -0.02 else 0)
        
        if volatility > self.volatility_threshold:
            self.regime_state = f"{'bull' if trend_direction > 0 else ('bear' if trend_direction < 0 else 'sideways')}_volatile"
        else:
            self.regime_state = f"{'bull' if trend_direction > 0 else ('bear' if trend_direction < 0 else 'sideways')}_calm"
        
        indicators['regime'] = self.regime_state
        indicators['volatility_regime'] = volatility
        
        return indicators
    
    def generate_signal(self, data: pd.DataFrame, timestamp: datetime) -> Dict[str, Any]:
        """Enhanced signal generation with multiple confirmations"""
        
        # Default hold signal
        default_signal = {'direction': 'hold', 'strength': 0, 'reason': 'default'}
        
        # Calculate indicators
        indicators = self.calculate_indicators(data)
        if not indicators:
            return default_signal
        
        # Check daily limits
        current_date = timestamp.date()
        if hasattr(self, 'last_trade_date') and self.last_trade_date == current_date:
            if self.daily_trades >= self.max_daily_trades:
                return {'direction': 'hold', 'strength': 0, 'reason': 'daily_limit'}
        else:
            self.daily_trades = 0
            self.daily_risk_used = 0
            self.last_trade_date = current_date
        
        # Check cooldown
        if self.last_trade_time:
            hours_since_last = (timestamp - self.last_trade_time).total_seconds() / 3600
            if hours_since_last < self.cooldown_hours:
                return {'direction': 'hold', 'strength': 0, 'reason': 'cooldown'}
        
        # === SIGNAL COMPONENTS ===
        
        signals = []
        reasons = []
        
        # 1. Trend Alignment
        if indicators['trend_strength'] > 0.03 and indicators['sma_20'] > indicators['sma_50']:
            signals.append(0.3)  # Strong bullish trend
            reasons.append("bullish_trend")
        elif indicators['trend_strength'] < -0.03 and indicators['sma_20'] < indicators['sma_50']:
            signals.append(-0.2)  # Bearish trend (smaller weight)
            reasons.append("bearish_trend")
        elif indicators['sma_20'] > indicators['sma_50']:
            signals.append(0.15)  # Weak bullish trend
            reasons.append("weak_bullish")
        
        # 2. Momentum Confirmation
        momentum_score = 0
        momentum_count = 0
        for period in [5, 10, 20]:
            if f'momentum_{period}' in indicators:
                mom = indicators[f'momentum_{period}']
                if mom > 0.02:  # Positive momentum
                    momentum_score += 0.2
                elif mom < -0.02:  # Negative momentum
                    momentum_score -= 0.1
                momentum_count += 1
        
        if momentum_count > 0:
            avg_momentum = momentum_score / momentum_count
            signals.append(avg_momentum)
            if avg_momentum > 0.1:
                reasons.append("positive_momentum")
            elif avg_momentum < -0.05:
                reasons.append("negative_momentum")
        
        # 3. RSI Conditions
        rsi = indicators.get('rsi', 50)
        if 30 < rsi < 45:  # RSI recovering from oversold
            signals.append(0.25)
            reasons.append("rsi_recovery")
        elif 55 < rsi < 70:  # RSI in bullish zone but not overbought
            signals.append(0.15)
            reasons.append("rsi_bullish")
        elif rsi > 75:  # Overbought - avoid longs
            signals.append(-0.2)
            reasons.append("rsi_overbought")
        elif rsi < 25:  # Very oversold - potential reversal
            signals.append(0.1)
            reasons.append("rsi_oversold")
        
        # 4. MACD Confirmation
        if indicators['macd_histogram'] > 0 and indicators['macd'] > 0:
            signals.append(0.2)
            reasons.append("macd_bullish")
        elif indicators['macd_histogram'] < 0 and indicators['macd'] < 0:
            signals.append(-0.1)
            reasons.append("macd_bearish")
        
        # 5. Volume Confirmation
        if indicators['volume_ratio'] > self.volume_multiplier:
            signals.append(0.15)  # Good volume support
            reasons.append("volume_confirmation")
        elif indicators['volume_ratio'] < 0.8:
            signals.append(-0.1)  # Low volume
            reasons.append("low_volume")
        
        # 6. Volatility Filter
        if indicators['volatility_regime'] > self.volatility_threshold * 1.5:
            # Very volatile - reduce signal strength
            signals = [s * 0.7 for s in signals]
            reasons.append("high_volatility_reduction")
        elif indicators['volatility_regime'] < self.volatility_threshold * 0.5:
            # Low volatility - normal conditions
            signals = [s * 1.1 for s in signals]
            reasons.append("low_volatility_boost")
        
        # 7. Bollinger Band Position
        bb_pos = indicators.get('bb_position', 0.5)
        if 0.2 < bb_pos < 0.4:  # Near lower band but not extreme
            signals.append(0.15)
            reasons.append("bb_oversold_recovery")
        elif bb_pos > 0.8:  # Near upper band
            signals.append(-0.15)
            reasons.append("bb_overbought")
        
        # === FINAL SIGNAL CALCULATION ===
        
        final_strength = np.mean(signals) if signals else 0
        
        # Market regime adjustment
        regime_multiplier = self._get_regime_multiplier(indicators['regime'])
        final_strength *= regime_multiplier
        
        # Recent performance adjustment
        if self.win_rate_adjustment and len(self.recent_trades) >= 5:
            recent_win_rate = sum(1 for trade in self.recent_trades[-10:] if trade.get('profitable', False)) / len(self.recent_trades[-10:])
            if recent_win_rate < 0.4:  # Poor recent performance
                final_strength *= 0.8  # Be more conservative
                reasons.append("poor_recent_performance")
            elif recent_win_rate > 0.6:  # Good recent performance
                final_strength *= 1.1  # Be slightly more aggressive
                reasons.append("good_recent_performance")
        
        # Apply minimum signal strength threshold
        if abs(final_strength) < self.min_signal_strength:
            return {'direction': 'hold', 'strength': abs(final_strength), 'reason': f'below_threshold_{abs(final_strength):.3f}'}
        
        # Determine direction
        if final_strength > self.min_signal_strength:
            direction = 'buy'
        elif final_strength < -self.min_signal_strength:
            direction = 'sell'
        else:
            direction = 'hold'
        
        return {
            'direction': direction,
            'strength': abs(final_strength),
            'reasons': reasons,
            'regime': indicators['regime'],
            'confidence': min(abs(final_strength) * 1.5, 1.0)
        }
    
    def _get_regime_multiplier(self, regime: str) -> float:
        """Get signal multiplier based on market regime"""
        
        multipliers = {
            'bull_calm': 1.2,        # Best conditions for longs
            'bull_volatile': 0.9,    # Volatile bull market
            'bear_calm': 0.8,        # Bearish but calm
            'bear_volatile': 0.6,    # Worst conditions
            'sideways_calm': 1.0,    # Normal conditions
            'sideways_volatile': 0.7, # Choppy market
            'unknown': 0.8
        }
        
        return multipliers.get(regime, 0.8)
    
    def calculate_position_size(self, signal_strength: float, current_equity: float, volatility: float) -> float:
        """Enhanced position sizing with volatility adjustment"""
        
        base_size = self.max_position_size * signal_strength
        
        # Volatility adjustment
        if volatility > 0.04:  # High volatility
            base_size *= 0.8
        elif volatility < 0.02:  # Low volatility
            base_size *= 1.1
        
        # Risk per trade limit
        risk_adjusted_size = min(base_size, self.stop_loss_pct * current_equity / self.stop_loss_pct)
        
        # Daily risk limit
        remaining_daily_risk = self.max_daily_risk - self.daily_risk_used
        if remaining_daily_risk <= 0:
            return 0
        
        daily_risk_limit = remaining_daily_risk * current_equity / self.stop_loss_pct
        
        final_size = min(risk_adjusted_size, daily_risk_limit)
        
        return max(final_size, 0)
    
    def should_exit(self, position, current_price: float, timestamp: datetime, indicators: Dict) -> Tuple[bool, str]:
        """Enhanced exit logic with trailing stops and dynamic targets"""
        
        # Calculate unrealized PnL
        if position.direction == 'long':
            pnl_pct = (current_price - position.entry_price) / position.entry_price
        else:
            pnl_pct = (position.entry_price - current_price) / position.entry_price
        
        # Stop Loss
        if pnl_pct <= -self.stop_loss_pct:
            return True, f"stop_loss_{pnl_pct:.3f}"
        
        # Take Profit
        if pnl_pct >= self.take_profit_pct:
            return True, f"take_profit_{pnl_pct:.3f}"
        
        # Time-based exit (prevent holding too long)
        hours_held = (timestamp - position.entry_time).total_seconds() / 3600
        if hours_held > 72:  # 3 days max
            return True, f"time_exit_{hours_held:.1f}h"
        
        # Trailing stop (if profitable)
        if pnl_pct > self.take_profit_pct * 0.4:  # At 40% of target
            trailing_stop = self.stop_loss_pct * 0.5  # Tighten stop to 1.25%
            if pnl_pct <= trailing_stop:
                return True, f"trailing_stop_{pnl_pct:.3f}"
        
        # Regime change exit
        if indicators and indicators.get('regime', '').endswith('_volatile') and pnl_pct < 0:
            # Exit losers quickly in volatile conditions
            if abs(pnl_pct) > self.stop_loss_pct * 0.6:  # 60% of stop loss
                return True, f"regime_exit_{indicators.get('regime', 'unknown')}"
        
        return False, "hold"
    
    def update_performance(self, trade_result: Dict):
        """Track performance for adaptive adjustments"""
        
        self.recent_trades.append(trade_result)
        
        # Keep only recent trades for performance tracking
        if len(self.recent_trades) > self.performance_window:
            self.recent_trades = self.recent_trades[-self.performance_window:]