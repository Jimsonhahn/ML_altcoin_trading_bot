#!/usr/bin/env python3
"""
Final Optimized Strategy
========================
Sweet Spot zwischen Trading-Aktivität und realistischen Erwartungen
"""

from balanced_realistic_strategy import BalancedRealisticStrategy
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FinalOptimizedStrategy(BalancedRealisticStrategy):
    """Final optimierte Version für aktives aber realistisches Trading"""
    
    def __init__(self):
        super().__init__()
        
        # Sweet Spot Parameter für aktives Trading
        self.max_position_size = 0.06        # 6% pro Trade - konservativ aber aktiv
        self.min_signal_strength = 0.08      # Niedrigere Schwelle für mehr Aktivität
        self.stop_loss_pct = 0.02           # 2% Stop Loss - eng aber realistisch
        self.take_profit_pct = 0.05         # 5% Take Profit (2.5:1 R/R)
        self.max_daily_trades = 2           # Max 2 Trades täglich
        self.cooldown_hours = 3             # 3h Cooldown
        
        # Realistische aber aktivere Parameter
        self.max_daily_risk = 0.012         # 1.2% täglich riskieren
        self.volume_multiplier = 1.0        # Keine Volume-Beschränkung
        self.volatility_threshold = 0.08    # Höhere Vol-Toleranz für mehr Trades
        
        # Trading-Kosten - realistisch aber nicht übertrieben
        self.trading_fee_pct = 0.001        # 0.1% Fee
        self.slippage_pct = 0.001          # 0.1% Slippage (optimistisch)
        self.spread_pct = 0.0002           # 0.02% Spread
        self.total_cost_per_trade = (self.trading_fee_pct + self.slippage_pct + self.spread_pct) * 2
        
        # Lockerere Risk Management für mehr Aktivität
        self.max_consecutive_losses = 5     # Stop nach 5 Verlusten
        self.consecutive_losses = 0
        
        logger.info(f"Final Optimized Strategy initialized:")
        logger.info(f"  Total cost per trade: {self.total_cost_per_trade*100:.2f}%")
        logger.info(f"  Target: Active trading with realistic expectations")
    
    def generate_signal(self, data, timestamp):
        """Aktivere Signalgenerierung mit reduzierten Filtern"""
        
        # Base signal calculation
        indicators = self.calculate_indicators(data)
        if not indicators:
            return {'direction': 'hold', 'strength': 0, 'reason': 'no_indicators'}
        
        # Check basic constraints
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
        
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            return {'direction': 'hold', 'strength': 0, 'reason': 'max_losses'}
        
        # === SIMPLIFIED SIGNAL GENERATION ===
        signals = []
        reasons = []
        
        # 1. Trend Signal (einfacher aber effektiv)
        trend_strength = indicators.get('trend_strength', 0)
        if trend_strength > 0.015:  # Positive trend
            signals.append(0.3)
            reasons.append("bullish_trend")
        elif trend_strength < -0.015:  # Negative trend - aber trotzdem handeln
            signals.append(0.15)  # Schwächeres Signal aber nicht 0
            reasons.append("bearish_trend_contrarian")
        
        # 2. RSI Signal (vereinfacht)
        rsi = indicators.get('rsi', 50)
        if 25 < rsi < 40:  # Oversold recovery
            signals.append(0.25)
            reasons.append("rsi_oversold")
        elif 45 < rsi < 65:  # Normal range
            signals.append(0.15)
            reasons.append("rsi_normal")
        elif rsi > 70:  # Overbought - aber nicht komplett meiden
            signals.append(0.05)
            reasons.append("rsi_overbought_weak")
        
        # 3. Momentum Signal
        momentum_5 = indicators.get('momentum_5', 0)
        momentum_10 = indicators.get('momentum_10', 0)
        
        if momentum_5 > 0.01 and momentum_10 > 0.005:
            signals.append(0.2)
            reasons.append("strong_momentum")
        elif momentum_5 > 0:
            signals.append(0.1)
            reasons.append("positive_momentum")
        
        # 4. MACD Signal
        macd_histogram = indicators.get('macd_histogram', 0)
        if macd_histogram > 0:
            signals.append(0.1)
            reasons.append("macd_positive")
        
        # 5. Volatility Boost (konträr - mehr Signale bei höherer Vol)
        volatility = indicators.get('volatility_regime', 0.03)
        if volatility > 0.05:  # Höhere Vol = mehr Opportunities
            vol_boost = min(volatility * 2, 0.2)  # Max 20% boost
            signals.append(vol_boost)
            reasons.append("volatility_opportunity")
        
        # === FINAL SIGNAL CALCULATION ===
        
        if not signals:
            return {'direction': 'hold', 'strength': 0, 'reason': 'no_signals'}
        
        final_strength = np.mean(signals)
        
        # Regime-based adjustment (weniger restriktiv)
        regime = indicators.get('regime', 'unknown')
        if 'volatile' in regime:
            final_strength *= 1.1  # Boost für volatile Märkte
        elif 'crisis' in regime:
            final_strength *= 0.8  # Nur leichte Reduktion
        
        # Apply minimum threshold
        if final_strength < self.min_signal_strength:
            return {'direction': 'hold', 'strength': final_strength, 
                   'reason': f'below_threshold_{final_strength:.3f}'}
        
        # Always long-only für Einfachheit
        direction = 'buy'
        
        logger.debug(f"Signal generated: {direction} strength={final_strength:.3f} reasons={reasons}")
        
        return {
            'direction': direction,
            'strength': final_strength,
            'reasons': reasons,
            'regime': regime,
            'confidence': min(final_strength * 1.2, 1.0)
        }
    
    def calculate_position_size(self, signal_strength: float, current_equity: float, volatility: float) -> float:
        """Aktivere Positionsberechnung"""
        
        if self.consecutive_losses >= self.max_consecutive_losses:
            return 0
        
        # Base size
        base_size = current_equity * self.max_position_size
        
        # Signal strength scaling (weniger restriktiv)
        strength_factor = min(signal_strength / 0.1, 1.5)  # Max 150% bei starken Signalen
        base_size *= strength_factor
        
        # Volatility adjustment (weniger penalisierend)
        if volatility > 0.08:  # Sehr hohe Vol
            base_size *= 0.8
        elif volatility > 0.06:  # Hohe Vol
            base_size *= 0.9
        elif volatility < 0.03:  # Niedrige Vol
            base_size *= 1.1
        
        # Recent performance (weniger restriktiv)
        if len(self.recent_performance) >= 3:
            recent_win_rate = sum(1 for p in self.recent_performance[-5:] if p.get('profitable', False)) / min(5, len(self.recent_performance))
            
            if recent_win_rate < 0.3:  # Sehr schlecht
                base_size *= 0.8
            elif recent_win_rate > 0.7:  # Sehr gut
                base_size *= 1.2
        
        # Minimum viable size (reduziert)
        min_viable = self.total_cost_per_trade * current_equity * 3  # 3x statt 4x
        
        if base_size < min_viable:
            return 0
        
        # Maximum limit
        max_absolute = current_equity * 0.10  # Max 10%
        base_size = min(base_size, max_absolute)
        
        return base_size
    
    def should_exit(self, position, current_price, timestamp, indicators):
        """Weniger aggressive Exit-Logik"""
        
        # Calculate P&L
        if position.direction == 'long':
            pnl_pct = (current_price - position.entry_price) / position.entry_price
        else:
            pnl_pct = (position.entry_price - current_price) / position.entry_price
        
        # Account for costs
        net_pnl_pct = pnl_pct - self.total_cost_per_trade
        
        # Standard exits
        if net_pnl_pct <= -self.stop_loss_pct:
            return True, f"stop_loss_{net_pnl_pct:.3f}"
        
        if net_pnl_pct >= self.take_profit_pct:
            return True, f"take_profit_{net_pnl_pct:.3f}"
        
        # Time exit (entspannt)
        hours_held = (timestamp - position.entry_time).total_seconds() / 3600
        if hours_held > 48:  # 2 Tage statt 3
            if net_pnl_pct > 0.005:  # Kleiner Gewinn
                return True, f"time_exit_profit_{hours_held:.1f}h"
            elif hours_held > 72:  # 3 Tage bei Verlust
                return True, f"time_exit_loss_{hours_held:.1f}h"
        
        # Trailing stop (entspannt)
        if net_pnl_pct > self.take_profit_pct * 0.5:  # 50% des Targets
            trailing_stop = -self.stop_loss_pct * 0.75  # Weniger eng
            if net_pnl_pct <= trailing_stop:
                return True, f"trailing_stop_{net_pnl_pct:.3f}"
        
        return False, "hold"
    
    def get_expected_annual_return(self):
        """Realistische Erwartung für final optimierte Strategie"""
        
        trades_per_month = 8   # Moderate Aktivität
        win_rate = 0.52       # Leicht positive Win Rate
        avg_win = self.take_profit_pct - self.total_cost_per_trade
        avg_loss = self.stop_loss_pct + self.total_cost_per_trade
        
        expected_return_per_trade = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        monthly_return = trades_per_month * expected_return_per_trade * self.max_position_size
        annual_return = ((1 + monthly_return) ** 12) - 1
        
        logger.info(f"Final Expected Performance:")
        logger.info(f"  Trades/month: {trades_per_month}")
        logger.info(f"  Win rate: {win_rate*100:.1f}%")
        logger.info(f"  Avg win: {avg_win*100:.2f}%")
        logger.info(f"  Avg loss: {avg_loss*100:.2f}%")
        logger.info(f"  Expected return per trade: {expected_return_per_trade*100:.2f}%")
        logger.info(f"  Expected annual return: {annual_return*100:.1f}%")
        
        return annual_return