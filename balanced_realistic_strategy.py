#!/usr/bin/env python3
"""
Balanced Realistic Strategy
===========================
Ausgewogene Strategie für realistische 15-25% Jahresrendite
"""

from corrected_realistic_strategy import CorrectedRealisticStrategy
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BalancedRealisticStrategy(CorrectedRealisticStrategy):
    """Ausgewogene Strategie zwischen Konservativ und Aggressiv"""
    
    def __init__(self):
        super().__init__()
        
        # Ausgewogene Parameter für 15-25% Jahresrendite
        self.max_position_size = 0.08        # 8% pro Trade (zwischen 5% und 12%)
        self.min_signal_strength = 0.12      # Moderate Qualitätsanforderung
        self.stop_loss_pct = 0.025          # 2.5% Stop Loss
        self.take_profit_pct = 0.06         # 6% Take Profit (2.4:1 R/R)
        self.max_daily_trades = 3           # Max 3 Trades täglich
        self.cooldown_hours = 2             # 2h Cooldown
        
        # Realistische Zusatzparameter
        self.max_daily_risk = 0.015         # 1.5% täglich riskieren
        self.volume_multiplier = 1.2        # Moderate Volume-Bestätigung
        self.volatility_threshold = 0.045   # Moderate Vol-Toleranz
        
        # Trading-Kosten explizit modellieren
        self.trading_fee_pct = 0.001        # 0.1% Fee (Binance Maker)
        self.slippage_pct = 0.002          # 0.2% Slippage
        self.spread_pct = 0.0005           # 0.05% Spread
        self.total_cost_per_trade = (self.trading_fee_pct + self.slippage_pct + self.spread_pct) * 2  # Round-trip
        
        # Erweiterte Risk Management
        self.max_consecutive_losses = 4     # Stop nach 4 Verlusten
        self.consecutive_losses = 0
        self.daily_loss_limit = 0.02       # 2% täglicher Verlust-Stop
        
        # Performance Tracking
        self.recent_performance = []
        self.performance_lookback = 20
        
        logger.info(f"Balanced Strategy initialized:")
        logger.info(f"  Total cost per trade: {self.total_cost_per_trade*100:.2f}%")
        logger.info(f"  Breakeven per trade: {self.total_cost_per_trade*100:.2f}%")
    
    def calculate_position_size(self, signal_strength: float, current_equity: float, volatility: float) -> float:
        """Ausgewogene Positionsgrößenberechnung mit Kostenberücksichtigung"""
        
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            logger.debug(f"Skipping trade due to {self.consecutive_losses} consecutive losses")
            return 0
        
        # Base position size
        base_size = current_equity * self.max_position_size
        
        # Signal strength scaling (0.12-0.3 range)
        if signal_strength < self.min_signal_strength:
            return 0
        
        strength_factor = min((signal_strength - self.min_signal_strength) / 0.18, 1.0)
        base_size *= (0.5 + 0.5 * strength_factor)  # 50-100% based on strength
        
        # Volatility adjustment
        if volatility > 0.06:  # High volatility
            base_size *= 0.6
        elif volatility > 0.04:  # Medium volatility
            base_size *= 0.8
        elif volatility < 0.02:  # Low volatility
            base_size *= 1.1
        
        # Recent performance adjustment
        if len(self.recent_performance) >= 5:
            recent_win_rate = sum(1 for p in self.recent_performance[-10:] if p.get('profitable', False)) / min(10, len(self.recent_performance))
            
            if recent_win_rate < 0.4:  # Poor recent performance
                base_size *= 0.7
                logger.debug(f"Reducing position size due to poor win rate: {recent_win_rate:.2f}")
            elif recent_win_rate > 0.7:  # Good recent performance
                base_size *= 1.2
                logger.debug(f"Increasing position size due to good win rate: {recent_win_rate:.2f}")
        
        # Minimum viable size (muss Trading-Kosten überwinden können)
        # Minimum 4x Trading-Kosten für profitablen Trade
        min_viable = self.total_cost_per_trade * current_equity * 4
        
        if base_size < min_viable:
            logger.debug(f"Position too small to overcome costs: ${base_size:.0f} < ${min_viable:.0f}")
            return 0
        
        # Maximum absolute limit
        max_absolute = current_equity * 0.12  # Never mehr als 12%
        base_size = min(base_size, max_absolute)
        
        logger.debug(f"Position size calculated: ${base_size:.0f} ({base_size/current_equity*100:.1f}% of equity)")
        return base_size
    
    def generate_signal(self, data, timestamp):
        """Ausgewogene Signalgenerierung mit Kostenberücksichtigung"""
        
        # Get base signal
        signal = super().generate_signal(data, timestamp)
        
        if signal['direction'] == 'hold':
            return signal
        
        # Zusätzliche Qualitätsfilter für profitablen Trade
        
        # 1. Signal muss stark genug sein um Kosten zu überwinden
        min_strength_for_costs = self.total_cost_per_trade * 2  # 2x Kosten als Minimum
        
        if signal.get('strength', 0) < min_strength_for_costs:
            logger.debug(f"Signal too weak for costs: {signal.get('strength', 0):.3f} < {min_strength_for_costs:.3f}")
            return {'direction': 'hold', 'strength': signal.get('strength', 0), 'reason': 'insufficient_for_costs'}
        
        # 2. Trend-Momentum Alignment für höhere Erfolgswahrscheinlichkeit
        indicators = self.calculate_indicators(data)
        if indicators:
            trend_strength = indicators.get('trend_strength', 0)
            momentum_5 = indicators.get('momentum_5', 0)
            
            # Require trend and momentum alignment
            if signal['direction'] == 'buy':
                if trend_strength < -0.01 or momentum_5 < -0.005:  # Gegen Trend
                    logger.debug("Buy signal against trend/momentum")
                    return {'direction': 'hold', 'strength': 0, 'reason': 'against_trend'}
            
            # Volume confirmation mit Kostenberücksichtigung
            volume_ratio = indicators.get('volume_ratio', 1)
            if volume_ratio < self.volume_multiplier:
                logger.debug(f"Insufficient volume: {volume_ratio:.2f} < {self.volume_multiplier}")
                return {'direction': 'hold', 'strength': 0, 'reason': 'low_volume'}
        
        # 3. R/R Check - muss mindestens 2:1 erwarten können
        expected_return = self.take_profit_pct
        max_loss = self.stop_loss_pct + self.total_cost_per_trade
        
        if expected_return / max_loss < 2.0:
            logger.debug(f"Poor R/R ratio: {expected_return/max_loss:.1f}:1")
            return {'direction': 'hold', 'strength': 0, 'reason': 'poor_risk_reward'}
        
        logger.debug(f"Signal approved: {signal['direction']} strength={signal.get('strength', 0):.3f}")
        return signal
    
    def should_exit(self, position, current_price, timestamp, indicators):
        """Erweiterte Exit-Logik mit Kostenberücksichtigung"""
        
        # Standard exits from parent
        should_exit, reason = super().should_exit(position, current_price, timestamp, indicators)
        
        if should_exit:
            return should_exit, reason
        
        # Additional exits
        
        # Calculate current P&L including costs
        if position.direction == 'long':
            pnl_pct = (current_price - position.entry_price) / position.entry_price
        else:
            pnl_pct = (position.entry_price - current_price) / position.entry_price
        
        # Account for costs when entering and exiting
        net_pnl_pct = pnl_pct - self.total_cost_per_trade
        
        # Early exit if costs are eating profits
        if 0.01 < net_pnl_pct < self.total_cost_per_trade * 1.5:  # Small profit being eaten by costs
            hours_held = (timestamp - position.entry_time).total_seconds() / 3600
            if hours_held > 12:  # Held long enough, take small profit
                return True, f"cost_protection_{net_pnl_pct:.3f}"
        
        # Trailing stop with cost consideration
        if net_pnl_pct > self.take_profit_pct * 0.6:  # 60% of target reached
            trailing_stop = self.stop_loss_pct * 0.5 + self.total_cost_per_trade  # Tighter stop + costs
            if net_pnl_pct <= trailing_stop:
                return True, f"trailing_stop_costs_{net_pnl_pct:.3f}"
        
        return False, "hold"
    
    def update_performance(self, trade_result):
        """Erweiterte Performance-Verfolgung"""
        super().update_performance(trade_result)
        
        # Add to recent performance
        self.recent_performance.append(trade_result)
        
        if len(self.recent_performance) > self.performance_lookback:
            self.recent_performance = self.recent_performance[-self.performance_lookback:]
        
        # Update consecutive losses
        if trade_result.get('profitable', False):
            self.consecutive_losses = 0
            logger.debug("Profitable trade - reset consecutive losses")
        else:
            self.consecutive_losses += 1
            logger.debug(f"Loss #{self.consecutive_losses}")
        
        # Log performance summary
        if len(self.recent_performance) >= 5:
            recent_win_rate = sum(1 for p in self.recent_performance[-10:] if p.get('profitable', False)) / min(10, len(self.recent_performance))
            logger.debug(f"Recent win rate: {recent_win_rate*100:.1f}% over {min(10, len(self.recent_performance))} trades")
    
    def get_expected_annual_return(self) -> float:
        """Schätzung der erwarteten Jahresrendite basierend auf Parametern"""
        
        # Conservative estimate
        trades_per_month = 15  # Etwa 3 pro Woche
        win_rate = 0.55  # 55% Win Rate
        avg_win = self.take_profit_pct - self.total_cost_per_trade  # Nach Kosten
        avg_loss = self.stop_loss_pct + self.total_cost_per_trade   # Nach Kosten
        
        expected_return_per_trade = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        monthly_return = trades_per_month * expected_return_per_trade * self.max_position_size
        annual_return = ((1 + monthly_return) ** 12) - 1
        
        logger.info(f"Expected annual return: {annual_return*100:.1f}%")
        logger.info(f"  Trades/month: {trades_per_month}")
        logger.info(f"  Win rate: {win_rate*100:.1f}%") 
        logger.info(f"  Avg win: {avg_win*100:.2f}%")
        logger.info(f"  Avg loss: {avg_loss*100:.2f}%")
        logger.info(f"  Expected return per trade: {expected_return_per_trade*100:.2f}%")
        
        return annual_return