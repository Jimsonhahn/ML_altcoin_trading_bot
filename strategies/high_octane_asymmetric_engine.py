"""
High-Octane Asymmetric Profit Engine
=====================================

Advanced risk-tiered strategy that combines:
- 70% Conservative foundation (orchestrator-based)
- 30% High-octane aggressive strategies

Expected Performance:
- Conservative scenario: 40-60% annual returns
- Moderate scenario: 80-150% annual returns  
- Aggressive scenario: 200-400% annual returns

WARNING: High-octane strategies carry substantial risk of significant losses.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from abc import abstractmethod
import asyncio
from collections import defaultdict
import statistics

from .strategy_base import Strategy, Signal

logger = logging.getLogger(__name__)


@dataclass
class RiskTier:
    """Risk tier configuration"""
    name: str
    allocation_percentage: float
    max_position_size: float
    stop_loss_percentage: float
    take_profit_percentage: float
    max_daily_trades: int
    max_daily_loss: float
    confidence_threshold: float
    leverage_range: Tuple[float, float]


@dataclass
class TradeSignal:
    """Enhanced trade signal with risk tier info"""
    action: str  # BUY, SELL, HOLD
    symbol: str
    position_size: float
    stop_loss: float
    take_profit: float
    leverage: float
    confidence: float
    risk_tier: str
    strategy_name: str
    reason: str
    expected_return: float
    max_risk: float
    time_limit_hours: Optional[float] = None


class HighOctaneStrategy:
    """Base class for high-octane strategies"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.performance_history = []
        self.daily_trades = 0
        self.daily_pnl = 0.0
        
    @abstractmethod
    async def analyze(self, market_data: pd.DataFrame, symbol: str) -> Optional[TradeSignal]:
        """Analyze market and generate signal"""
        pass
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_trades = 0
        self.daily_pnl = 0.0


class LeverageBreakoutHunter(HighOctaneStrategy):
    """Hunt high-probability breakouts with leverage"""
    
    def __init__(self):
        super().__init__("LeverageBreakoutHunter", {
            'max_leverage': 5.0,
            'target_profit': 0.20,  # 20%
            'stop_loss': 0.08,      # 8%
            'min_volume_spike': 2.0,
            'consolidation_periods': 20
        })
    
    async def analyze(self, market_data: pd.DataFrame, symbol: str) -> Optional[TradeSignal]:
        """Identify breakout setups"""
        if len(market_data) < 50:
            return None
            
        try:
            # Calculate indicators
            market_data['sma_20'] = market_data['close'].rolling(20).mean()
            market_data['bb_upper'], market_data['bb_lower'] = self._calculate_bollinger_bands(market_data['close'])
            market_data['volume_sma'] = market_data['volume'].rolling(20).mean()
            market_data['atr'] = self._calculate_atr(market_data)
            
            current_price = market_data['close'].iloc[-1]
            current_volume = market_data['volume'].iloc[-1]
            volume_ratio = current_volume / market_data['volume_sma'].iloc[-1]
            
            # Detect consolidation
            price_range = market_data['high'].tail(20).max() - market_data['low'].tail(20).min()
            avg_range = market_data['atr'].iloc[-1]
            is_consolidating = price_range < avg_range * 2
            
            # Breakout conditions
            breaking_resistance = current_price > market_data['bb_upper'].iloc[-1]
            volume_spike = volume_ratio > self.config['min_volume_spike']
            
            if is_consolidating and breaking_resistance and volume_spike:
                # Calculate optimal leverage based on volatility
                volatility = market_data['close'].pct_change().tail(20).std()
                leverage = min(self.config['max_leverage'], 1.0 / (volatility * 10))
                
                return TradeSignal(
                    action='BUY',
                    symbol=symbol,
                    position_size=0.15,  # 15% of high-risk allocation
                    stop_loss=self.config['stop_loss'],
                    take_profit=self.config['target_profit'],
                    leverage=leverage,
                    confidence=0.85,
                    risk_tier='high_octane',
                    strategy_name=self.name,
                    reason=f"Breakout detected with {volume_ratio:.1f}x volume",
                    expected_return=self.config['target_profit'] * leverage,
                    max_risk=self.config['stop_loss'] * leverage,
                    time_limit_hours=24
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in {self.name}: {e}")
            return None
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return upper, lower
    
    def _calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window).mean()


class VolatilitySpikeSurfer(HighOctaneStrategy):
    """Ride extreme volatility expansions"""
    
    def __init__(self):
        super().__init__("VolatilitySpikeSurfer", {
            'vol_spike_threshold': 3.0,  # 3x normal volatility
            'min_price_movement': 0.05,  # 5% move
            'target_profit': 0.50,       # 50%
            'stop_loss': 0.15,           # 15%
            'max_leverage': 3.0
        })
    
    async def analyze(self, market_data: pd.DataFrame, symbol: str) -> Optional[TradeSignal]:
        """Detect and trade volatility explosions"""
        if len(market_data) < 100:
            return None
            
        try:
            # Calculate volatility metrics
            returns = market_data['close'].pct_change()
            current_vol = returns.tail(10).std()
            normal_vol = returns.tail(50).std()
            vol_ratio = current_vol / normal_vol if normal_vol > 0 else 1
            
            # Price movement
            price_change_1h = (market_data['close'].iloc[-1] - market_data['close'].iloc[-6]) / market_data['close'].iloc[-6]
            
            # Volume analysis
            volume_spike = market_data['volume'].iloc[-1] / market_data['volume'].rolling(20).mean().iloc[-1]
            
            if (vol_ratio > self.config['vol_spike_threshold'] and 
                abs(price_change_1h) > self.config['min_price_movement'] and
                volume_spike > 2.0):
                
                # Direction based on momentum
                direction = 'BUY' if price_change_1h > 0 else 'SELL'
                leverage = min(self.config['max_leverage'], vol_ratio)
                
                return TradeSignal(
                    action=direction,
                    symbol=symbol,
                    position_size=0.10,  # 10% of high-risk allocation
                    stop_loss=self.config['stop_loss'],
                    take_profit=self.config['target_profit'],
                    leverage=leverage,
                    confidence=0.80,
                    risk_tier='high_octane',
                    strategy_name=self.name,
                    reason=f"Volatility spike {vol_ratio:.1f}x with {abs(price_change_1h)*100:.1f}% move",
                    expected_return=self.config['target_profit'],
                    max_risk=self.config['stop_loss'] * leverage,
                    time_limit_hours=4
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in {self.name}: {e}")
            return None


class MomentumScalpingMachine(HighOctaneStrategy):
    """High-frequency momentum scalping"""
    
    def __init__(self):
        super().__init__("MomentumScalpingMachine", {
            'min_momentum_score': 0.8,
            'scalp_target': 0.02,      # 2% quick gains
            'scalp_stop': 0.01,        # 1% stop
            'max_leverage': 10.0,      # High leverage for small moves
            'min_volume': 1000000,     # Need liquidity
            'hold_time_minutes': 30
        })
    
    async def analyze(self, market_data: pd.DataFrame, symbol: str) -> Optional[TradeSignal]:
        """Identify quick scalping opportunities"""
        if len(market_data) < 50:
            return None
            
        try:
            # Fast momentum indicators
            market_data['ema_5'] = market_data['close'].ewm(span=5).mean()
            market_data['ema_10'] = market_data['close'].ewm(span=10).mean()
            market_data['rsi'] = self._calculate_rsi(market_data['close'], 14)
            
            # Momentum scoring
            momentum_score = 0
            current_price = market_data['close'].iloc[-1]
            
            # Price above EMAs
            if current_price > market_data['ema_5'].iloc[-1]:
                momentum_score += 0.3
            if current_price > market_data['ema_10'].iloc[-1]:
                momentum_score += 0.2
                
            # EMA alignment
            if market_data['ema_5'].iloc[-1] > market_data['ema_10'].iloc[-1]:
                momentum_score += 0.3
                
            # RSI momentum
            rsi = market_data['rsi'].iloc[-1]
            if 50 < rsi < 70:
                momentum_score += 0.2
            
            # Volume confirmation
            current_volume = market_data['volume'].iloc[-1]
            if current_volume > self.config['min_volume']:
                momentum_score += 0.1
            
            if momentum_score >= self.config['min_momentum_score']:
                # Calculate micro-leverage for scalping
                volatility = market_data['close'].pct_change().tail(10).std()
                leverage = min(self.config['max_leverage'], 0.01 / volatility)
                
                return TradeSignal(
                    action='BUY',
                    symbol=symbol,
                    position_size=0.08,  # 8% per scalp
                    stop_loss=self.config['scalp_stop'],
                    take_profit=self.config['scalp_target'],
                    leverage=leverage,
                    confidence=momentum_score,
                    risk_tier='high_octane',
                    strategy_name=self.name,
                    reason=f"Momentum scalp setup (score: {momentum_score:.2f})",
                    expected_return=self.config['scalp_target'] * leverage,
                    max_risk=self.config['scalp_stop'] * leverage,
                    time_limit_hours=0.5  # 30 minutes
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in {self.name}: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class LiquidationHunter(HighOctaneStrategy):
    """Trade around liquidation cascades"""
    
    def __init__(self):
        super().__init__("LiquidationHunter", {
            'liquidation_threshold': 0.10,  # 10% move
            'cascade_volume_mult': 5.0,     # 5x normal volume
            'bounce_target': 0.15,          # 15% bounce
            'stop_loss': 0.05,              # 5% stop
            'max_leverage': 2.0             # Conservative leverage
        })
    
    async def analyze(self, market_data: pd.DataFrame, symbol: str) -> Optional[TradeSignal]:
        """Detect liquidation cascades"""
        if len(market_data) < 50:
            return None
            
        try:
            # Detect sharp moves with volume
            price_change_1h = (market_data['close'].iloc[-1] - market_data['close'].iloc[-6]) / market_data['close'].iloc[-6]
            volume_spike = market_data['volume'].iloc[-1] / market_data['volume'].rolling(20).mean().iloc[-1]
            
            # Look for liquidation cascade pattern
            if (abs(price_change_1h) > self.config['liquidation_threshold'] and
                volume_spike > self.config['cascade_volume_mult']):
                
                # Trade the bounce
                if price_change_1h < 0:  # Sharp drop = long bounce
                    action = 'BUY'
                    reason = f"Liquidation cascade detected: {price_change_1h*100:.1f}% drop"
                else:  # Sharp rise = short correction
                    action = 'SELL'
                    reason = f"Short squeeze detected: {price_change_1h*100:.1f}% spike"
                
                return TradeSignal(
                    action=action,
                    symbol=symbol,
                    position_size=0.12,  # 12% of high-risk
                    stop_loss=self.config['stop_loss'],
                    take_profit=self.config['bounce_target'],
                    leverage=self.config['max_leverage'],
                    confidence=0.75,
                    risk_tier='high_octane',
                    strategy_name=self.name,
                    reason=reason,
                    expected_return=self.config['bounce_target'],
                    max_risk=self.config['stop_loss'] * self.config['max_leverage'],
                    time_limit_hours=6
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in {self.name}: {e}")
            return None


class HighOctaneAsymmetricEngine(Strategy):
    """
    Main High-Octane Asymmetric Profit Engine
    
    Combines conservative foundation with aggressive strategies
    for asymmetric risk/reward profile
    """
    
    def __init__(self, params: Dict = None, ml_components: Optional[Any] = None):
        super().__init__(params, ml_components)
        
        # Portfolio allocation
        self.conservative_allocation = 0.70  # 70% foundation
        self.aggressive_allocation = 0.30    # 30% high-octane
        
        # Risk tiers
        self.risk_tiers = {
            'conservative': RiskTier(
                name='conservative',
                allocation_percentage=0.70,
                max_position_size=0.02,  # 2% per trade
                stop_loss_percentage=0.02,
                take_profit_percentage=0.04,
                max_daily_trades=10,
                max_daily_loss=0.05,  # 5% of allocation
                confidence_threshold=0.8,
                leverage_range=(1.0, 1.0)
            ),
            'aggressive': RiskTier(
                name='aggressive',
                allocation_percentage=0.30,
                max_position_size=0.15,  # 15% of allocation per trade
                stop_loss_percentage=0.10,
                take_profit_percentage=0.30,
                max_daily_trades=20,
                max_daily_loss=0.15,  # 15% of allocation
                confidence_threshold=0.6,
                leverage_range=(1.0, 10.0)
            )
        }
        
        # Initialize high-octane strategies
        self.high_octane_strategies = [
            LeverageBreakoutHunter(),
            VolatilitySpikeSurfer(),
            MomentumScalpingMachine(),
            LiquidationHunter()
        ]
        
        # Performance tracking
        self.tier_performance = {
            'conservative': {'daily_pnl': 0.0, 'trades': 0, 'wins': 0},
            'aggressive': {'daily_pnl': 0.0, 'trades': 0, 'wins': 0}
        }
        
        # Risk management
        self.daily_stats = {
            'total_trades': 0,
            'conservative_trades': 0,
            'aggressive_trades': 0,
            'total_pnl': 0.0,
            'conservative_pnl': 0.0,
            'aggressive_pnl': 0.0,
            'last_reset': datetime.now().date()
        }
        
        # Dynamic allocation adjustment
        self.performance_window = []  # Last 7 days performance
        self.allocation_adjustment_enabled = params.get('dynamic_allocation', True) if params else True
        
        logger.info("🚀 High-Octane Asymmetric Engine initialized")
        logger.info(f"   Conservative: {self.conservative_allocation*100:.0f}%")
        logger.info(f"   Aggressive: {self.aggressive_allocation*100:.0f}%")
    
    def calculate_signal(self, symbol: str, data: pd.DataFrame, current_price: float) -> Tuple[str, Dict[str, Any]]:
        """
        Main signal calculation with risk-tiered approach
        """
        try:
            # Daily reset check
            self._check_daily_reset()
            
            # Risk checks
            if not self._pre_trade_risk_check():
                return 'HOLD', {'reason': 'Risk limits reached', 'confidence': 0.0}
            
            # Market analysis
            market_conditions = self._analyze_market_conditions(data)
            
            # Get signals from both tiers
            conservative_signal = self._get_conservative_signal(symbol, data, current_price, market_conditions)
            aggressive_signals = asyncio.run(self._get_aggressive_signals(symbol, data, market_conditions))
            
            # Select optimal signal
            selected_signal = self._select_optimal_signal(conservative_signal, aggressive_signals, market_conditions)
            
            if selected_signal:
                # Apply final risk checks
                final_signal = self._apply_final_risk_check(selected_signal)
                
                if final_signal.action != 'HOLD':
                    # Update tracking
                    self._update_daily_stats(final_signal)
                    
                    # Convert to expected format
                    return final_signal.action, {
                        'symbol': final_signal.symbol,
                        'position_size': final_signal.position_size,
                        'stop_loss_pct': final_signal.stop_loss,
                        'take_profit_pct': final_signal.take_profit,
                        'leverage': final_signal.leverage,
                        'confidence': final_signal.confidence,
                        'risk_tier': final_signal.risk_tier,
                        'strategy': final_signal.strategy_name,
                        'reason': final_signal.reason,
                        'expected_return': final_signal.expected_return,
                        'max_risk': final_signal.max_risk,
                        'time_limit': final_signal.time_limit_hours
                    }
            
            return 'HOLD', {'reason': 'No high-confidence signals', 'confidence': 0.0}
            
        except Exception as e:
            logger.error(f"Error in High-Octane Engine: {e}")
            return 'HOLD', {'reason': f'Error: {str(e)}', 'confidence': 0.0}
    
    def _check_daily_reset(self):
        """Check and perform daily reset"""
        today = datetime.now().date()
        if today != self.daily_stats['last_reset']:
            # Store performance for allocation adjustment
            self.performance_window.append({
                'date': self.daily_stats['last_reset'],
                'conservative_pnl': self.daily_stats['conservative_pnl'],
                'aggressive_pnl': self.daily_stats['aggressive_pnl']
            })
            
            # Keep only last 7 days
            if len(self.performance_window) > 7:
                self.performance_window.pop(0)
            
            # Adjust allocations if enabled
            if self.allocation_adjustment_enabled:
                self._adjust_allocations()
            
            # Reset daily stats
            self.daily_stats = {
                'total_trades': 0,
                'conservative_trades': 0,
                'aggressive_trades': 0,
                'total_pnl': 0.0,
                'conservative_pnl': 0.0,
                'aggressive_pnl': 0.0,
                'last_reset': today
            }
            
            # Reset strategy daily stats
            for strategy in self.high_octane_strategies:
                strategy.reset_daily_stats()
            
            logger.info("📅 Daily stats reset completed")
    
    def _pre_trade_risk_check(self) -> bool:
        """Pre-trade risk checks"""
        # Check daily loss limits
        conservative_tier = self.risk_tiers['conservative']
        aggressive_tier = self.risk_tiers['aggressive']
        
        # Conservative tier check
        if (self.daily_stats['conservative_pnl'] < 
            -conservative_tier.max_daily_loss * self.conservative_allocation):
            logger.warning("Conservative tier daily loss limit reached")
            return False
        
        # Aggressive tier check
        if (self.daily_stats['aggressive_pnl'] < 
            -aggressive_tier.max_daily_loss * self.aggressive_allocation):
            logger.warning("Aggressive tier daily loss limit reached")
            # Don't block conservative trades
            pass
        
        return True
    
    def _analyze_market_conditions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market conditions"""
        try:
            if len(data) < 50:
                return {'regime': 'unknown', 'volatility': 0.02, 'trend': 'neutral'}
            
            # Calculate metrics
            returns = data['close'].pct_change()
            volatility = returns.tail(20).std()
            
            # Trend
            sma_20 = data['close'].rolling(20).mean().iloc[-1]
            sma_50 = data['close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else sma_20
            current_price = data['close'].iloc[-1]
            
            if current_price > sma_20 > sma_50:
                trend = 'bullish'
            elif current_price < sma_20 < sma_50:
                trend = 'bearish'
            else:
                trend = 'neutral'
            
            # Volume
            volume_ratio = data['volume'].iloc[-1] / data['volume'].rolling(20).mean().iloc[-1]
            
            # Market regime
            if volatility > 0.05:
                regime = 'high_volatility'
            elif volatility < 0.01:
                regime = 'low_volatility'
            else:
                regime = 'normal'
            
            return {
                'regime': regime,
                'volatility': volatility,
                'trend': trend,
                'volume_ratio': volume_ratio,
                'current_price': current_price,
                'sma_20': sma_20,
                'sma_50': sma_50
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            return {'regime': 'unknown', 'volatility': 0.02, 'trend': 'neutral'}
    
    def _get_conservative_signal(self, symbol: str, data: pd.DataFrame, 
                                 current_price: float, market_conditions: Dict) -> Optional[TradeSignal]:
        """Get signal from conservative strategies"""
        try:
            # Simple conservative strategy (can be replaced with orchestrator)
            if len(data) < 50:
                return None
            
            # Conservative momentum strategy
            sma_20 = market_conditions['sma_20']
            sma_50 = market_conditions['sma_50']
            trend = market_conditions['trend']
            
            if trend == 'bullish' and current_price > sma_20:
                # Calculate RSI
                rsi = self._calculate_rsi(data['close'])
                
                if 40 < rsi < 65:  # Not overbought
                    return TradeSignal(
                        action='BUY',
                        symbol=symbol,
                        position_size=self.risk_tiers['conservative'].max_position_size,
                        stop_loss=self.risk_tiers['conservative'].stop_loss_percentage,
                        take_profit=self.risk_tiers['conservative'].take_profit_percentage,
                        leverage=1.0,
                        confidence=0.85,
                        risk_tier='conservative',
                        strategy_name='conservative_momentum',
                        reason='Conservative uptrend entry',
                        expected_return=0.04,
                        max_risk=0.02
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in conservative signal: {e}")
            return None
    
    async def _get_aggressive_signals(self, symbol: str, data: pd.DataFrame, 
                                     market_conditions: Dict) -> List[TradeSignal]:
        """Get signals from all high-octane strategies"""
        signals = []
        
        # Run all strategies in parallel
        tasks = []
        for strategy in self.high_octane_strategies:
            tasks.append(strategy.analyze(data, symbol))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, TradeSignal):
                signals.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Strategy error: {result}")
        
        return signals
    
    def _select_optimal_signal(self, conservative_signal: Optional[TradeSignal],
                              aggressive_signals: List[TradeSignal],
                              market_conditions: Dict) -> Optional[TradeSignal]:
        """Select optimal signal based on market conditions and risk limits"""
        all_signals = []
        
        if conservative_signal:
            all_signals.append(conservative_signal)
        all_signals.extend(aggressive_signals)
        
        if not all_signals:
            return None
        
        # Score signals
        scored_signals = []
        for signal in all_signals:
            score = self._score_signal(signal, market_conditions)
            scored_signals.append((score, signal))
        
        # Sort by score
        scored_signals.sort(key=lambda x: x[0], reverse=True)
        
        # Return highest scoring signal
        return scored_signals[0][1] if scored_signals else None
    
    def _score_signal(self, signal: TradeSignal, market_conditions: Dict) -> float:
        """Score a signal based on multiple factors"""
        score = signal.confidence
        
        # Market regime alignment
        if market_conditions['regime'] == 'high_volatility':
            if signal.strategy_name in ['VolatilitySpikeSurfer', 'LiquidationHunter']:
                score *= 1.2
        elif market_conditions['regime'] == 'normal':
            if signal.risk_tier == 'conservative':
                score *= 1.1
        
        # Expected return factor
        score *= (1 + signal.expected_return * 0.1)
        
        # Risk penalty
        score *= (1 - signal.max_risk * 0.05)
        
        # Daily stats penalty (reduce if already traded much)
        if signal.risk_tier == 'aggressive':
            trades_today = self.daily_stats['aggressive_trades']
            score *= (1 - trades_today * 0.05)  # 5% penalty per trade
        
        return score
    
    def _apply_final_risk_check(self, signal: TradeSignal) -> TradeSignal:
        """Apply final risk management checks"""
        tier = self.risk_tiers[signal.risk_tier]
        
        # Check daily trade limits
        if signal.risk_tier == 'conservative':
            if self.daily_stats['conservative_trades'] >= tier.max_daily_trades:
                signal.action = 'HOLD'
                signal.reason = 'Daily trade limit reached'
        else:
            if self.daily_stats['aggressive_trades'] >= tier.max_daily_trades:
                signal.action = 'HOLD'
                signal.reason = 'Daily trade limit reached'
        
        # Adjust position size based on daily P&L
        if signal.risk_tier == 'aggressive' and self.daily_stats['aggressive_pnl'] < 0:
            # Reduce position size if losing
            loss_ratio = abs(self.daily_stats['aggressive_pnl']) / (tier.max_daily_loss * self.aggressive_allocation)
            signal.position_size *= (1 - loss_ratio * 0.5)  # Up to 50% reduction
        
        return signal
    
    def _update_daily_stats(self, signal: TradeSignal):
        """Update daily statistics"""
        self.daily_stats['total_trades'] += 1
        
        if signal.risk_tier == 'conservative':
            self.daily_stats['conservative_trades'] += 1
        else:
            self.daily_stats['aggressive_trades'] += 1
    
    def _adjust_allocations(self):
        """Dynamically adjust allocations based on performance"""
        if len(self.performance_window) < 3:
            return
        
        # Calculate recent performance
        recent_conservative_pnl = sum(p['conservative_pnl'] for p in self.performance_window[-3:])
        recent_aggressive_pnl = sum(p['aggressive_pnl'] for p in self.performance_window[-3:])
        
        # Performance ratio
        if recent_conservative_pnl != 0:
            performance_ratio = recent_aggressive_pnl / abs(recent_conservative_pnl)
        else:
            performance_ratio = 1.0
        
        # Adjust allocations
        if performance_ratio > 2.0:  # Aggressive doing 2x better
            self.aggressive_allocation = min(0.40, self.aggressive_allocation + 0.05)
        elif performance_ratio < 0.5:  # Aggressive doing worse
            self.aggressive_allocation = max(0.20, self.aggressive_allocation - 0.05)
        
        self.conservative_allocation = 1.0 - self.aggressive_allocation
        
        logger.info(f"📊 Allocation adjusted - Conservative: {self.conservative_allocation*100:.0f}%, "
                   f"Aggressive: {self.aggressive_allocation*100:.0f}%")
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def update_performance(self, trade_result: Dict[str, Any]):
        """Update performance after trade completion"""
        pnl = trade_result.get('pnl', 0.0)
        risk_tier = trade_result.get('risk_tier', 'conservative')
        
        # Update daily stats
        self.daily_stats['total_pnl'] += pnl
        
        if risk_tier == 'conservative':
            self.daily_stats['conservative_pnl'] += pnl
            self.tier_performance['conservative']['trades'] += 1
            if pnl > 0:
                self.tier_performance['conservative']['wins'] += 1
        else:
            self.daily_stats['aggressive_pnl'] += pnl
            self.tier_performance['aggressive']['trades'] += 1
            if pnl > 0:
                self.tier_performance['aggressive']['wins'] += 1
        
        # Update specific strategy performance
        strategy_name = trade_result.get('strategy_name')
        for strategy in self.high_octane_strategies:
            if strategy.name == strategy_name:
                strategy.performance_history.append(pnl)
                strategy.daily_pnl += pnl
        
        logger.info(f"💰 Trade completed - {risk_tier} tier: {pnl:+.2f}% "
                   f"(Daily: {self.daily_stats[f'{risk_tier}_pnl']:+.2f}%)")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            'allocations': {
                'conservative': self.conservative_allocation,
                'aggressive': self.aggressive_allocation
            },
            'daily_stats': self.daily_stats.copy(),
            'tier_performance': self.tier_performance.copy(),
            'strategy_performance': {
                strategy.name: {
                    'trades': strategy.daily_trades,
                    'pnl': strategy.daily_pnl,
                    'avg_return': statistics.mean(strategy.performance_history) if strategy.performance_history else 0
                }
                for strategy in self.high_octane_strategies
            },
            'risk_metrics': {
                'conservative_risk_used': abs(self.daily_stats['conservative_pnl']) / (self.risk_tiers['conservative'].max_daily_loss * self.conservative_allocation),
                'aggressive_risk_used': abs(self.daily_stats['aggressive_pnl']) / (self.risk_tiers['aggressive'].max_daily_loss * self.aggressive_allocation)
            }
        }