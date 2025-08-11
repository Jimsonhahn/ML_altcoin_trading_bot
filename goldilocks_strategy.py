#!/usr/bin/env python3
"""
Goldilocks Strategy - The Perfect Balance
==========================================
Target: $10,000 → $35,000 (3 Jahre) = 28% jährlich
Not too conservative, not too aggressive - just right!
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Optional
import talib

from final_optimized_strategy import FinalOptimizedStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoldilocksStrategy:
    """
    Balanced Strategy für realistische aber ambitionierte Returns
    
    TARGET: 28% annually ($10k → $35k in 3 years)
    PHILOSOPHY: Sweet spot zwischen Sicherheit und Profit
    """
    
    def __init__(self):
        self.name = "Goldilocks Strategy"
        self.version = "1.0"
        
        # RISK PARAMETERS - Balanced
        self.max_position_size = 0.12        # 12% per trade (vs 6% conservative)
        self.stop_loss_pct = 0.025          # 2.5% stop loss (etwas höher)
        self.take_profit_pct = 0.08         # 8% take profit (höher für 3.2:1 R/R)
        self.max_portfolio_risk = 0.25      # 25% max risk (vs 15% conservative)
        
        # SIGNAL PARAMETERS - More aggressive
        self.min_signal_strength = 0.06     # Niedrigere threshold (vs 0.08)
        self.signal_decay_hours = 2         # Signals bleiben 2h gültig
        
        # TRADING PARAMETERS - More active  
        self.max_daily_trades = 4           # 4 trades/Tag (vs 2)
        self.max_weekly_trades = 20         # 20 trades/Woche
        self.cooldown_minutes = 15          # 15min cooldown (vs 30min)
        self.max_consecutive_losses = 4     # 4 losses in Folge
        
        # MARKET REGIME DETECTION - Enhanced
        self.trend_lookback = 20            # 20-day trend
        self.volatility_window = 14         # 14-day volatility
        self.regime_threshold = 0.015       # Trend detection
        
        # MULTI-TIMEFRAME - NEW FEATURE
        self.use_multi_timeframe = True
        self.primary_timeframe = "1h"       # Primary signals
        self.secondary_timeframe = "4h"     # Confirmation signals
        
        # LEVERAGE - Conservative leverage
        self.max_leverage = 1.5             # 1.5x in strong trends only
        self.leverage_threshold = 0.15      # Min signal für leverage
        
        # COST MODELING - Realistic
        self.maker_fee = 0.00075           # 0.075% Binance with BNB
        self.taker_fee = 0.001             # 0.1% market orders
        self.avg_slippage = 0.0015         # 0.15% average slippage
        self.total_cost_per_trade = 0.0035  # 0.35% total (vs 0.44% conservative)
        
        # PERFORMANCE TARGETS
        self.target_monthly_return = 0.021  # 2.1% monthly (28% annually)
        self.target_win_rate = 0.56         # 56% win rate
        self.target_profit_factor = 1.8     # 1.8 profit factor
        self.target_sharpe = 1.6            # 1.6+ Sharpe ratio
        
        # INTERNAL STATE
        self.current_position = None
        self.daily_trades = 0
        self.weekly_trades = 0
        self.consecutive_losses = 0
        self.last_trade_time = None
        self.equity_curve = []
        
        # BACKTESTER COMPATIBILITY
        self.daily_risk_used = 0.0
        self.last_signal_time = None
        
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Enhanced indicators für Goldilocks Strategy"""
        
        if len(data) < 50:
            return {}
        
        try:
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            volume = data['volume'].values if 'volume' in data else None
            
            # TREND INDICATORS
            sma_20 = talib.SMA(close, timeperiod=20)
            sma_50 = talib.SMA(close, timeperiod=50)
            ema_12 = talib.EMA(close, timeperiod=12)
            ema_26 = talib.EMA(close, timeperiod=26)
            
            # MOMENTUM INDICATORS
            rsi = talib.RSI(close, timeperiod=14)
            macd, macd_signal, macd_histogram = talib.MACD(close)
            
            # VOLATILITY INDICATORS
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
            atr = talib.ATR(high, low, close, timeperiod=14)
            
            # VOLUME INDICATORS (if available)
            if volume is not None:
                volume_sma = talib.SMA(volume, timeperiod=20)
                volume_ratio = volume[-1] / volume_sma[-1] if volume_sma[-1] > 0 else 1.0
            else:
                volume_ratio = 1.0
            
            # CALCULATE DERIVED METRICS
            
            # 1. Trend Strength (Enhanced)
            price_vs_sma20 = (close[-1] - sma_20[-1]) / sma_20[-1] if sma_20[-1] > 0 else 0
            price_vs_sma50 = (close[-1] - sma_50[-1]) / sma_50[-1] if sma_50[-1] > 0 else 0
            sma_slope = (sma_20[-1] - sma_20[-5]) / sma_20[-5] if sma_20[-5] > 0 else 0
            trend_strength = (price_vs_sma20 + price_vs_sma50 + sma_slope) / 3
            
            # 2. Momentum Score  
            rsi_normalized = (rsi[-1] - 50) / 50  # -1 to 1
            macd_momentum = macd_histogram[-1] if not np.isnan(macd_histogram[-1]) else 0
            ema_momentum = (ema_12[-1] - ema_26[-1]) / ema_26[-1] if ema_26[-1] > 0 else 0
            momentum_score = (rsi_normalized + macd_momentum + ema_momentum) / 3
            
            # 3. Volatility Regime
            bb_width = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1] if bb_middle[-1] > 0 else 0
            atr_pct = atr[-1] / close[-1] if close[-1] > 0 else 0
            volatility_regime = (bb_width + atr_pct) / 2
            
            # 4. Mean Reversion Signal
            bb_position = (close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) if (bb_upper[-1] - bb_lower[-1]) > 0 else 0.5
            mean_reversion_signal = 0.5 - bb_position  # Negative when oversold, positive when overbought
            
            # 5. Breakout Signal
            price_change_5 = (close[-1] - close[-6]) / close[-6] if close[-6] > 0 else 0
            price_change_10 = (close[-1] - close[-11]) / close[-11] if close[-11] > 0 else 0
            breakout_signal = (price_change_5 + price_change_10) / 2
            
            return {
                'trend_strength': trend_strength,
                'momentum_score': momentum_score,
                'volatility_regime': volatility_regime,
                'mean_reversion_signal': mean_reversion_signal,
                'breakout_signal': breakout_signal,
                'rsi': rsi[-1] if not np.isnan(rsi[-1]) else 50,
                'macd_histogram': macd_histogram[-1] if not np.isnan(macd_histogram[-1]) else 0,
                'bb_position': bb_position,
                'volume_ratio': volume_ratio,
                'atr_pct': atr_pct,
                'price_vs_sma20': price_vs_sma20,
                'price_vs_sma50': price_vs_sma50
            }
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}
    
    def generate_signal(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> Dict[str, Any]:
        """
        Enhanced Signal Generation für Goldilocks Strategy
        
        STRATEGY: Kombiniert Trend Following + Mean Reversion + Breakouts
        """
        
        # Check constraints first
        if self.daily_trades >= self.max_daily_trades:
            return {'direction': 'hold', 'reason': 'max_daily_trades_reached', 'strength': 0}
        
        if self.consecutive_losses >= self.max_consecutive_losses:
            return {'direction': 'hold', 'reason': 'too_many_consecutive_losses', 'strength': 0}
        
        if self.last_trade_time and (timestamp - self.last_trade_time).total_seconds() < (self.cooldown_minutes * 60):
            return {'direction': 'hold', 'reason': 'cooldown_period', 'strength': 0}
        
        # Get indicators
        indicators = self.calculate_indicators(data)
        if not indicators:
            return {'direction': 'hold', 'reason': 'insufficient_data', 'strength': 0}
        
        # SIGNAL COMPONENTS - Multiple strategies combined
        signal_components = []
        signal_reasons = []
        
        # 1. TREND FOLLOWING SIGNALS
        trend_strength = indicators.get('trend_strength', 0)
        if trend_strength > 0.02:  # Strong uptrend
            signal_components.append(0.35)
            signal_reasons.append(f"strong_uptrend_{trend_strength:.3f}")
        elif trend_strength > 0.01:  # Moderate uptrend
            signal_components.append(0.25)
            signal_reasons.append(f"moderate_uptrend_{trend_strength:.3f}")
        elif trend_strength < -0.02:  # Strong downtrend - contrarian
            signal_components.append(0.15)
            signal_reasons.append(f"oversold_downtrend_{trend_strength:.3f}")
        
        # 2. MOMENTUM SIGNALS
        momentum = indicators.get('momentum_score', 0)
        if momentum > 0.02:  # Strong momentum
            signal_components.append(0.3)
            signal_reasons.append(f"strong_momentum_{momentum:.3f}")
        elif momentum > 0.01:  # Moderate momentum
            signal_components.append(0.2)
            signal_reasons.append(f"moderate_momentum_{momentum:.3f}")
        
        # 3. MEAN REVERSION SIGNALS
        mean_reversion = indicators.get('mean_reversion_signal', 0)
        rsi = indicators.get('rsi', 50)
        if mean_reversion < -0.3 and rsi < 35:  # Oversold
            signal_components.append(0.25)
            signal_reasons.append(f"oversold_mean_reversion_{rsi:.1f}")
        elif mean_reversion < -0.15 and rsi < 45:  # Moderately oversold
            signal_components.append(0.15)
            signal_reasons.append(f"moderate_oversold_{rsi:.1f}")
        
        # 4. BREAKOUT SIGNALS
        breakout = indicators.get('breakout_signal', 0)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        if breakout > 0.03 and volume_ratio > 1.5:  # Strong breakout with volume
            signal_components.append(0.3)
            signal_reasons.append(f"volume_breakout_{breakout:.3f}")
        elif breakout > 0.02:  # Moderate breakout
            signal_components.append(0.2)
            signal_reasons.append(f"price_breakout_{breakout:.3f}")
        
        # 5. VOLATILITY OPPORTUNITY
        volatility = indicators.get('volatility_regime', 0.03)
        if volatility > 0.06:  # High volatility = more opportunity
            vol_boost = min(volatility * 3, 0.25)
            signal_components.append(vol_boost)
            signal_reasons.append(f"volatility_opportunity_{volatility:.3f}")
        
        # 6. BOLLINGER BAND POSITION
        bb_position = indicators.get('bb_position', 0.5)
        if bb_position < 0.2:  # Near lower band
            signal_components.append(0.2)
            signal_reasons.append(f"bb_oversold_{bb_position:.2f}")
        elif bb_position < 0.35:  # Moderately low
            signal_components.append(0.1)
            signal_reasons.append(f"bb_low_{bb_position:.2f}")
        
        # Calculate final signal strength
        if signal_components:
            # Weighted average with more weight on stronger signals
            signal_strength = np.average(signal_components, weights=[max(s, 0.1) for s in signal_components])
            
            # Apply market regime filter
            if volatility < 0.015:  # Very low volatility
                signal_strength *= 0.7  # Reduce signal strength
                signal_reasons.append("low_vol_penalty")
            
            # Apply trend confirmation
            if trend_strength > 0.01 and momentum > 0.01:  # Aligned signals
                signal_strength *= 1.2  # Boost signal
                signal_reasons.append("trend_momentum_aligned")
            
        else:
            signal_strength = 0
        
        # DECISION LOGIC
        if signal_strength >= self.min_signal_strength:
            # Determine position size based on signal strength
            position_multiplier = min(signal_strength / self.min_signal_strength, 2.0)  # Max 2x
            position_size = min(self.max_position_size * position_multiplier, self.max_position_size * 1.5)
            
            # Check for leverage conditions
            use_leverage = 1.0
            if signal_strength >= self.leverage_threshold and trend_strength > 0.02:
                use_leverage = min(self.max_leverage, 1.0 + (signal_strength - self.leverage_threshold) * 2)
                signal_reasons.append(f"leverage_{use_leverage:.1f}x")
            
            return {
                'direction': 'buy',
                'strength': signal_strength,
                'position_size': position_size,
                'leverage': use_leverage,
                'confidence': min(signal_strength / 0.15, 1.0),
                'reasons': signal_reasons,
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct,
                'expected_hold_time': '2-8 hours',
                'risk_reward_ratio': self.take_profit_pct / self.stop_loss_pct
            }
        else:
            return {
                'direction': 'hold',
                'reason': f'signal_too_weak_{signal_strength:.3f}<{self.min_signal_strength:.3f}',
                'strength': signal_strength,
                'components': signal_components,
                'reasons': signal_reasons
            }
    
    def calculate_position_size(self, signal_strength: float, current_capital: float, market_volatility: float = 0.03) -> float:
        """Calculate optimal position size based on Kelly Criterion + Risk Management"""
        
        # Extract from signal if it's a dict, otherwise use signal_strength directly
        if isinstance(signal_strength, dict):
            signal = signal_strength
            base_position_size = signal.get('position_size', self.max_position_size)
            leverage = signal.get('leverage', 1.0)
            confidence = signal.get('confidence', 0.5)
        else:
            # Simple interface for backtester compatibility
            base_position_size = self.max_position_size
            leverage = 1.0
            confidence = min(signal_strength / 0.15, 1.0)
        
        # Kelly Criterion (simplified)
        win_rate = self.target_win_rate
        avg_win = self.take_profit_pct
        avg_loss = self.stop_loss_pct
        
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_fraction = max(0.01, min(kelly_fraction, 0.15))  # Cap at 15%
        
        # Combine base position with Kelly and confidence
        optimal_size = (base_position_size + kelly_fraction + confidence * 0.05) / 3
        optimal_size = min(optimal_size, self.max_position_size)
        
        # Apply leverage
        effective_size = optimal_size * leverage
        
        # Final risk check
        max_loss = effective_size * self.stop_loss_pct
        if max_loss > self.max_portfolio_risk:
            effective_size = self.max_portfolio_risk / self.stop_loss_pct
        
        return effective_size
    
    def should_exit(self, position, current_price: float, current_time: pd.Timestamp, market_info: Dict[str, Any]) -> tuple:
        """Check if current position should be exited"""
        
        entry_price = position.entry_price
        price_change = (current_price - entry_price) / entry_price
        
        # Stop Loss
        if price_change <= -self.stop_loss_pct:
            return True, "stop_loss"
        
        # Take Profit
        if price_change >= self.take_profit_pct:
            return True, "take_profit"
        
        # Time-based exit (optional)
        if hasattr(position, 'entry_time'):
            hours_held = (current_time - position.entry_time).total_seconds() / 3600
            if hours_held > 48:  # Max 48h hold
                return True, "max_hold_time"
        
        # Volatility-based exit
        volatility = market_info.get('volatility', 0.03)
        if volatility > 0.15:  # Very high volatility
            # Exit if small profit in high volatility
            if price_change > 0.01:  # 1% profit
                return True, "high_volatility_profit"
        
        return False, "hold"
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return strategy information"""
        return {
            'name': self.name,
            'version': self.version,
            'target_annual_return': '28%',
            'target_3year_profit': '$25,000 (on $10k)',
            'max_position_size': f'{self.max_position_size*100:.1f}%',
            'max_leverage': f'{self.max_leverage:.1f}x',
            'signal_threshold': self.min_signal_strength,
            'stop_loss': f'{self.stop_loss_pct*100:.1f}%',
            'take_profit': f'{self.take_profit_pct*100:.1f}%',
            'risk_reward_ratio': f'{self.take_profit_pct/self.stop_loss_pct:.1f}:1',
            'max_daily_trades': self.max_daily_trades,
            'trading_cost': f'{self.total_cost_per_trade*100:.2f}%',
            'strategy_type': 'Multi-Strategy (Trend + Mean Reversion + Breakout)',
            'risk_level': 'Balanced (Medium-High)',
            'complexity': 'Medium',
            'market_conditions': 'All markets (adaptive)'
        }

if __name__ == "__main__":
    # Quick test
    strategy = GoldilocksStrategy()
    info = strategy.get_strategy_info()
    
    print("🎯 GOLDILOCKS STRATEGY INITIALIZED")
    print("=" * 50)
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    print(f"\n🚀 Ready to target $25,000 profit in 3 years!")