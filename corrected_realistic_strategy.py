#!/usr/bin/env python3
"""
Corrected Realistic Strategy
============================
Konservative, ehrliche Strategie mit realistischen Erwartungen
"""

from optimized_realistic_strategy import OptimizedRealisticStrategy

class CorrectedRealisticStrategy(OptimizedRealisticStrategy):
    """Ultra-konservative Version für realistische Ergebnisse"""
    
    def __init__(self):
        super().__init__()
        
        # DRASTISCH konservativere Parameter
        self.max_position_size = 0.05        # Nur 5% pro Trade (statt 12%)
        self.min_signal_strength = 0.15      # Höhere Qualität erforderlich
        self.stop_loss_pct = 0.02           # Engerer Stop (2%)
        self.take_profit_pct = 0.04         # Kleineres Target (4%, R/R = 2:1)
        self.max_daily_trades = 2           # Max 2 Trades täglich
        self.cooldown_hours = 4             # 4h Cooldown
        
        # Sehr konservative Zusatzparameter
        self.max_daily_risk = 0.01          # Nur 1% täglich riskieren
        self.volume_multiplier = 1.5        # Hohe Volume-Bestätigung nötig
        self.volatility_threshold = 0.03    # Niedrigere Vol-Toleranz
        
        # Realistischere Risk Management
        self.max_consecutive_losses = 3     # Stop nach 3 Verlusten
        self.consecutive_losses = 0         # Counter
        self.daily_loss_limit = 0.015       # 1.5% täglicher Verlust-Stop
        
    def calculate_position_size(self, signal_strength: float, current_equity: float, volatility: float) -> float:
        """Ultra-konservative Positionsgrößenberechnung"""
        
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            return 0  # Kein Trading nach 3 Verlusten
        
        # Base calculation from parent
        base_size = super().calculate_position_size(signal_strength, current_equity, volatility)
        
        # Additional conservative adjustments
        
        # 1. Volatility penalty (höhere Vol = kleinere Position)
        if volatility > 0.04:
            base_size *= 0.5  # Halbe Position bei hoher Vol
        elif volatility > 0.06:
            return 0  # Kein Trading bei sehr hoher Vol
            
        # 2. Signal strength scaling (stärkere Signale = größere Position)
        strength_multiplier = min(signal_strength / 0.2, 1.0)  # Max bei 20% Signal
        base_size *= strength_multiplier
        
        # 3. Absolute maximum
        max_absolute = current_equity * 0.03  # Max 3% absolute
        base_size = min(base_size, max_absolute)
        
        # 4. Minimum viable size
        min_viable = 50  # $50 minimum
        if base_size < min_viable:
            return 0
            
        return base_size
    
    def generate_signal(self, data, timestamp):
        """Konservativere Signalgenerierung"""
        
        # Get base signal
        signal = super().generate_signal(data, timestamp)
        
        # Additional filters
        
        # 1. Require higher strength for execution
        if signal.get('strength', 0) < self.min_signal_strength:
            return {'direction': 'hold', 'strength': signal.get('strength', 0), 
                   'reason': f'insufficient_strength_{signal.get("strength", 0):.3f}'}
        
        # 2. Market conditions filter
        indicators = self.calculate_indicators(data)
        if indicators:
            # No trading in highly volatile conditions
            if indicators.get('volatility_regime', 0) > self.volatility_threshold:
                return {'direction': 'hold', 'strength': 0, 'reason': 'high_volatility'}
            
            # Require strong volume confirmation
            if indicators.get('volume_ratio', 0) < self.volume_multiplier:
                return {'direction': 'hold', 'strength': 0, 'reason': 'insufficient_volume'}
        
        # 3. Conservative regime filter
        regime = signal.get('regime', '')
        if 'volatile' in regime or 'crisis' in regime:
            # Reduce signal strength in volatile conditions
            signal['strength'] *= 0.5
            if signal['strength'] < self.min_signal_strength:
                return {'direction': 'hold', 'strength': signal['strength'], 'reason': 'regime_filter'}
        
        return signal
    
    def update_performance(self, trade_result):
        """Track consecutive losses"""
        super().update_performance(trade_result)
        
        if trade_result.get('profitable', False):
            self.consecutive_losses = 0  # Reset on win
        else:
            self.consecutive_losses += 1  # Increment on loss