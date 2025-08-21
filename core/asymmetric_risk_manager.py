"""
Asymmetric Risk Management Framework
===================================

Advanced risk management system for the High-Octane Asymmetric Engine.
Implements multi-tier risk controls with dynamic adjustment capabilities.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from enum import Enum
import asyncio
from collections import deque
import statistics

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class RiskAlert:
    """Risk alert with severity and recommended actions"""
    severity: AlertSeverity
    message: str
    affected_tier: str
    recommended_actions: List[str]
    timestamp: datetime
    risk_score: float
    auto_execute: bool = False


@dataclass
class PositionRisk:
    """Risk metrics for a single position"""
    symbol: str
    strategy: str
    risk_tier: str
    position_size: float
    leverage: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    stop_loss: float
    take_profit: float
    time_in_position: timedelta
    value_at_risk: float
    exposure_ratio: float
    liquidation_price: Optional[float] = None


@dataclass
class PortfolioRisk:
    """Overall portfolio risk metrics"""
    total_value: float
    total_exposure: float
    leverage_weighted_exposure: float
    value_at_risk_1day: float
    value_at_risk_7day: float
    max_drawdown_current: float
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float
    overall_risk_score: float
    risk_level: RiskLevel


class AsymmetricRiskManager:
    """
    Advanced risk management system for asymmetric trading strategies
    
    Features:
    - Multi-tier risk controls (conservative vs aggressive)
    - Dynamic position sizing based on performance
    - Real-time risk monitoring
    - Automatic risk reduction triggers
    - Portfolio-level risk limits
    - Emergency stop mechanisms
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Risk limits by tier
        self.tier_limits = {
            'conservative': {
                'max_position_size': 0.02,       # 2% per position
                'max_portfolio_exposure': 0.15,   # 15% total
                'max_daily_loss': 0.05,          # 5% daily loss
                'max_drawdown': 0.10,            # 10% max drawdown
                'var_limit_1day': 0.03,          # 3% 1-day VaR
                'correlation_limit': 0.7,         # Max 70% correlation
                'leverage_limit': 1.0             # No leverage
            },
            'aggressive': {
                'max_position_size': 0.15,       # 15% per position
                'max_portfolio_exposure': 0.50,   # 50% total
                'max_daily_loss': 0.15,          # 15% daily loss
                'max_drawdown': 0.30,            # 30% max drawdown
                'var_limit_1day': 0.10,          # 10% 1-day VaR
                'correlation_limit': 0.9,         # Max 90% correlation
                'leverage_limit': 10.0            # Up to 10x leverage
            }
        }
        
        # Portfolio limits
        self.portfolio_limits = {
            'max_total_exposure': 0.65,          # 65% total exposure
            'max_leverage_weighted': 2.0,        # 2x leverage-weighted
            'emergency_stop_loss': 0.20,         # 20% portfolio loss
            'concentration_limit': 0.25,         # 25% in single asset
            'correlation_emergency': 0.95        # 95% correlation = emergency
        }
        
        # Risk monitoring
        self.positions: Dict[str, PositionRisk] = {}
        self.risk_alerts: deque = deque(maxlen=100)
        self.risk_history: List[Dict] = []
        
        # Performance tracking for dynamic sizing
        self.tier_performance = {
            'conservative': deque(maxlen=30),  # Last 30 trades
            'aggressive': deque(maxlen=30)
        }
        
        # Circuit breakers
        self.circuit_breakers = {
            'portfolio_halt': False,
            'aggressive_halt': False,
            'emergency_mode': False,
            'last_check': datetime.now()
        }
        
        # Dynamic sizing factors
        self.sizing_factors = {
            'conservative': 1.0,
            'aggressive': 1.0
        }
        
        logger.info("🛡️ Asymmetric Risk Manager initialized")
    
    def calculate_position_size(self, signal_data: Dict[str, Any], 
                               tier: str, portfolio_value: float) -> float:
        """
        Calculate optimal position size with multi-factor risk adjustment
        
        Args:
            signal_data: Trading signal information
            tier: Risk tier ('conservative' or 'aggressive')
            portfolio_value: Current portfolio value
            
        Returns:
            Adjusted position size
        """
        try:
            base_size = signal_data.get('position_size', 0.01)
            confidence = signal_data.get('confidence', 0.5)
            leverage = signal_data.get('leverage', 1.0)
            
            # Tier limits
            tier_config = self.tier_limits.get(tier, self.tier_limits['conservative'])
            max_size = tier_config['max_position_size']
            
            # Base size adjustment
            adjusted_size = min(base_size, max_size)
            
            # Confidence adjustment
            confidence_factor = 0.5 + (confidence * 0.5)  # 0.5 to 1.0 range
            adjusted_size *= confidence_factor
            
            # Performance-based dynamic sizing
            sizing_factor = self.sizing_factors.get(tier, 1.0)
            adjusted_size *= sizing_factor
            
            # Leverage adjustment (reduce size for higher leverage)
            if leverage > 1.0:
                leverage_penalty = 1.0 / (1.0 + (leverage - 1.0) * 0.1)
                adjusted_size *= leverage_penalty
            
            # Portfolio exposure check
            current_exposure = self._calculate_current_exposure(tier)
            max_exposure = tier_config['max_portfolio_exposure']
            
            if current_exposure + adjusted_size > max_exposure:
                adjusted_size = max(0, max_exposure - current_exposure)
            
            # Minimum viable size
            min_size = 0.001  # 0.1%
            adjusted_size = max(adjusted_size, min_size) if adjusted_size > 0 else 0
            
            logger.debug(f"Position size calculated: {base_size:.3f} -> {adjusted_size:.3f} "
                        f"(tier: {tier}, confidence: {confidence:.2f})")
            
            return adjusted_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.001  # Minimal fallback size
    
    def validate_trade(self, trade_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a trade against all risk controls
        
        Args:
            trade_data: Complete trade information
            
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            tier = trade_data.get('risk_tier', 'conservative')
            position_size = trade_data.get('position_size', 0)
            leverage = trade_data.get('leverage', 1.0)
            symbol = trade_data.get('symbol', 'UNKNOWN')
            
            tier_config = self.tier_limits.get(tier, self.tier_limits['conservative'])
            
            # 1. Position size check
            if position_size > tier_config['max_position_size']:
                issues.append(f"Position size {position_size:.3f} exceeds tier limit {tier_config['max_position_size']:.3f}")
            
            # 2. Leverage check
            if leverage > tier_config['leverage_limit']:
                issues.append(f"Leverage {leverage:.1f}x exceeds tier limit {tier_config['leverage_limit']:.1f}x")
            
            # 3. Exposure check
            current_exposure = self._calculate_current_exposure(tier)
            if current_exposure + position_size > tier_config['max_portfolio_exposure']:
                issues.append(f"Would exceed tier exposure limit: {current_exposure + position_size:.3f} > {tier_config['max_portfolio_exposure']:.3f}")
            
            # 4. Circuit breaker check
            if self.circuit_breakers['portfolio_halt']:
                issues.append("Portfolio trading halted by circuit breaker")
            
            if tier == 'aggressive' and self.circuit_breakers['aggressive_halt']:
                issues.append("Aggressive tier trading halted")
            
            if self.circuit_breakers['emergency_mode']:
                issues.append("Emergency mode active - all trading suspended")
            
            # 5. Daily loss limit check
            daily_loss = self._get_daily_loss(tier)
            if daily_loss > tier_config['max_daily_loss']:
                issues.append(f"Daily loss limit exceeded: {daily_loss:.3f} > {tier_config['max_daily_loss']:.3f}")
            
            # 6. Concentration check
            concentration = self._calculate_symbol_concentration(symbol)
            if concentration + position_size > self.portfolio_limits['concentration_limit']:
                issues.append(f"Would exceed concentration limit for {symbol}")
            
            is_valid = len(issues) == 0
            
            if not is_valid:
                logger.warning(f"Trade validation failed: {'; '.join(issues)}")
            
            return is_valid, issues
            
        except Exception as e:
            logger.error(f"Error validating trade: {e}")
            return False, [f"Validation error: {str(e)}"]
    
    def monitor_position(self, position_id: str, position_data: Dict[str, Any]) -> List[RiskAlert]:
        """
        Monitor an existing position for risk alerts
        
        Args:
            position_id: Unique position identifier
            position_data: Current position information
            
        Returns:
            List of risk alerts
        """
        alerts = []
        
        try:
            # Update position risk metrics
            position_risk = self._calculate_position_risk(position_data)
            self.positions[position_id] = position_risk
            
            tier = position_risk.risk_tier
            tier_config = self.tier_limits.get(tier, self.tier_limits['conservative'])
            
            # Check stop loss
            if position_risk.unrealized_pnl_percent <= -position_data.get('stop_loss', 0.05):
                alerts.append(RiskAlert(
                    severity=AlertSeverity.WARNING,
                    message=f"Position {position_id} hit stop loss",
                    affected_tier=tier,
                    recommended_actions=['close_position'],
                    timestamp=datetime.now(),
                    risk_score=0.8,
                    auto_execute=True
                ))
            
            # Check excessive loss
            max_loss = tier_config['max_position_size'] * 2  # 2x position size
            if position_risk.unrealized_pnl_percent <= -max_loss:
                alerts.append(RiskAlert(
                    severity=AlertSeverity.CRITICAL,
                    message=f"Position {position_id} excessive loss: {position_risk.unrealized_pnl_percent:.2%}",
                    affected_tier=tier,
                    recommended_actions=['emergency_close', 'review_risk_controls'],
                    timestamp=datetime.now(),
                    risk_score=0.95,
                    auto_execute=True
                ))
            
            # Check leverage liquidation risk
            if position_risk.liquidation_price and position_risk.leverage > 1:
                distance_to_liquidation = abs(position_risk.current_price - position_risk.liquidation_price) / position_risk.current_price
                if distance_to_liquidation < 0.05:  # 5% from liquidation
                    alerts.append(RiskAlert(
                        severity=AlertSeverity.CRITICAL,
                        message=f"Position {position_id} near liquidation",
                        affected_tier=tier,
                        recommended_actions=['reduce_leverage', 'add_margin'],
                        timestamp=datetime.now(),
                        risk_score=0.9,
                        auto_execute=False
                    ))
            
            # Check time-based exits
            if position_data.get('time_limit_hours'):
                time_limit = timedelta(hours=position_data['time_limit_hours'])
                if position_risk.time_in_position > time_limit:
                    alerts.append(RiskAlert(
                        severity=AlertSeverity.INFO,
                        message=f"Position {position_id} exceeded time limit",
                        affected_tier=tier,
                        recommended_actions=['consider_exit', 'review_strategy'],
                        timestamp=datetime.now(),
                        risk_score=0.3,
                        auto_execute=False
                    ))
            
            # Store alerts
            for alert in alerts:
                self.risk_alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error monitoring position {position_id}: {e}")
            return []
    
    def assess_portfolio_risk(self, portfolio_data: Dict[str, Any]) -> PortfolioRisk:
        """
        Assess overall portfolio risk
        
        Args:
            portfolio_data: Complete portfolio information
            
        Returns:
            Portfolio risk assessment
        """
        try:
            total_value = portfolio_data.get('total_value', 0)
            
            # Calculate exposures
            total_exposure = sum(pos.exposure_ratio for pos in self.positions.values())
            leverage_weighted = sum(pos.exposure_ratio * pos.leverage for pos in self.positions.values())
            
            # Value at Risk calculation
            var_1day = self._calculate_var(1)
            var_7day = self._calculate_var(7)
            
            # Drawdown calculation
            current_drawdown = self._calculate_current_drawdown(portfolio_data)
            
            # Correlation risk
            correlation_risk = self._calculate_correlation_risk()
            
            # Concentration risk
            concentration_risk = max(self._calculate_symbol_concentration(symbol) 
                                   for symbol in set(pos.symbol for pos in self.positions.values())) if self.positions else 0
            
            # Liquidity risk (simplified)
            liquidity_risk = min(1.0, len(self.positions) / 10)  # More positions = higher liquidity risk
            
            # Overall risk score
            risk_components = [
                total_exposure / self.portfolio_limits['max_total_exposure'],
                leverage_weighted / self.portfolio_limits['max_leverage_weighted'],
                var_1day / 0.05,  # Normalize to 5% VaR
                current_drawdown / 0.15,  # Normalize to 15% drawdown
                correlation_risk,
                concentration_risk / self.portfolio_limits['concentration_limit'],
                liquidity_risk
            ]
            
            overall_risk_score = sum(risk_components) / len(risk_components)
            
            # Determine risk level
            if overall_risk_score < 0.3:
                risk_level = RiskLevel.LOW
            elif overall_risk_score < 0.6:
                risk_level = RiskLevel.MEDIUM
            elif overall_risk_score < 0.8:
                risk_level = RiskLevel.HIGH
            else:
                risk_level = RiskLevel.EXTREME
            
            portfolio_risk = PortfolioRisk(
                total_value=total_value,
                total_exposure=total_exposure,
                leverage_weighted_exposure=leverage_weighted,
                value_at_risk_1day=var_1day,
                value_at_risk_7day=var_7day,
                max_drawdown_current=current_drawdown,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                liquidity_risk=liquidity_risk,
                overall_risk_score=overall_risk_score,
                risk_level=risk_level
            )
            
            # Check for emergency conditions
            self._check_emergency_conditions(portfolio_risk)
            
            return portfolio_risk
            
        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {e}")
            return PortfolioRisk(
                total_value=0, total_exposure=0, leverage_weighted_exposure=0,
                value_at_risk_1day=0, value_at_risk_7day=0, max_drawdown_current=0,
                correlation_risk=0, concentration_risk=0, liquidity_risk=0,
                overall_risk_score=1.0, risk_level=RiskLevel.EXTREME
            )
    
    def update_performance(self, tier: str, pnl_percent: float):
        """Update performance tracking for dynamic sizing"""
        self.tier_performance[tier].append(pnl_percent)
        
        # Recalculate sizing factors
        self._update_sizing_factors()
    
    def _update_sizing_factors(self):
        """Update dynamic sizing factors based on recent performance"""
        for tier in ['conservative', 'aggressive']:
            performance = list(self.tier_performance[tier])
            
            if len(performance) >= 5:  # Need minimum sample
                recent_avg = statistics.mean(performance[-5:])
                overall_avg = statistics.mean(performance)
                
                # Performance trend adjustment
                if recent_avg > overall_avg * 1.2:  # 20% better
                    self.sizing_factors[tier] = min(1.5, self.sizing_factors[tier] * 1.1)
                elif recent_avg < overall_avg * 0.8:  # 20% worse
                    self.sizing_factors[tier] = max(0.5, self.sizing_factors[tier] * 0.9)
                
                # Win rate adjustment
                wins = sum(1 for p in performance[-10:] if p > 0)
                win_rate = wins / min(len(performance), 10)
                
                if win_rate > 0.7:  # High win rate
                    self.sizing_factors[tier] *= 1.05
                elif win_rate < 0.4:  # Low win rate
                    self.sizing_factors[tier] *= 0.95
                
                # Bounds
                self.sizing_factors[tier] = max(0.3, min(2.0, self.sizing_factors[tier]))
    
    def _calculate_current_exposure(self, tier: str) -> float:
        """Calculate current exposure for a tier"""
        return sum(pos.exposure_ratio for pos in self.positions.values() 
                  if pos.risk_tier == tier)
    
    def _calculate_position_risk(self, position_data: Dict[str, Any]) -> PositionRisk:
        """Calculate detailed risk metrics for a position"""
        symbol = position_data.get('symbol', 'UNKNOWN')
        strategy = position_data.get('strategy', 'unknown')
        risk_tier = position_data.get('risk_tier', 'conservative')
        position_size = position_data.get('position_size', 0)
        leverage = position_data.get('leverage', 1.0)
        entry_price = position_data.get('entry_price', 0)
        current_price = position_data.get('current_price', entry_price)
        entry_time = position_data.get('entry_time', datetime.now())
        
        # Calculate P&L
        if entry_price > 0:
            unrealized_pnl_percent = ((current_price - entry_price) / entry_price) * leverage
        else:
            unrealized_pnl_percent = 0
        
        unrealized_pnl = unrealized_pnl_percent * position_size
        
        # Time in position
        time_in_position = datetime.now() - entry_time
        
        # Value at Risk (simplified)
        volatility = position_data.get('volatility', 0.02)
        value_at_risk = position_size * leverage * volatility * 2.33  # 99% confidence
        
        # Exposure ratio
        exposure_ratio = position_size * leverage
        
        # Liquidation price (for leveraged positions)
        liquidation_price = None
        if leverage > 1:
            # Simplified liquidation calculation
            liquidation_distance = 0.8 / leverage  # 80% of margin
            if position_data.get('action') == 'BUY':
                liquidation_price = entry_price * (1 - liquidation_distance)
            else:
                liquidation_price = entry_price * (1 + liquidation_distance)
        
        return PositionRisk(
            symbol=symbol,
            strategy=strategy,
            risk_tier=risk_tier,
            position_size=position_size,
            leverage=leverage,
            entry_price=entry_price,
            current_price=current_price,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_percent=unrealized_pnl_percent,
            stop_loss=position_data.get('stop_loss', 0.05),
            take_profit=position_data.get('take_profit', 0.10),
            time_in_position=time_in_position,
            value_at_risk=value_at_risk,
            exposure_ratio=exposure_ratio,
            liquidation_price=liquidation_price
        )
    
    def _calculate_var(self, days: int) -> float:
        """Calculate portfolio Value at Risk"""
        if not self.positions:
            return 0
        
        total_var = sum(pos.value_at_risk for pos in self.positions.values())
        return total_var * np.sqrt(days)  # Scale for time horizon
    
    def _calculate_current_drawdown(self, portfolio_data: Dict[str, Any]) -> float:
        """Calculate current portfolio drawdown"""
        current_value = portfolio_data.get('total_value', 0)
        peak_value = portfolio_data.get('peak_value', current_value)
        
        if peak_value > 0:
            return (peak_value - current_value) / peak_value
        return 0
    
    def _calculate_correlation_risk(self) -> float:
        """Calculate portfolio correlation risk"""
        # Simplified: assume higher correlation with more positions in same tier
        if not self.positions:
            return 0
        
        # Count positions by tier
        tier_counts = {}
        for pos in self.positions.values():
            tier_counts[pos.risk_tier] = tier_counts.get(pos.risk_tier, 0) + 1
        
        # Higher concentration = higher correlation risk
        max_concentration = max(tier_counts.values()) if tier_counts else 0
        total_positions = len(self.positions)
        
        return max_concentration / total_positions if total_positions > 0 else 0
    
    def _calculate_symbol_concentration(self, symbol: str) -> float:
        """Calculate concentration in a specific symbol"""
        symbol_exposure = sum(pos.exposure_ratio for pos in self.positions.values() 
                             if pos.symbol == symbol)
        total_exposure = sum(pos.exposure_ratio for pos in self.positions.values())
        
        return symbol_exposure / total_exposure if total_exposure > 0 else 0
    
    def _get_daily_loss(self, tier: str) -> float:
        """Get current daily loss for a tier"""
        # This would be implemented with actual P&L tracking
        # For now, return 0 as placeholder
        return 0
    
    def _check_emergency_conditions(self, portfolio_risk: PortfolioRisk):
        """Check for emergency conditions and trigger circuit breakers"""
        
        # Emergency stop conditions
        if (portfolio_risk.max_drawdown_current > self.portfolio_limits['emergency_stop_loss'] or
            portfolio_risk.overall_risk_score > 0.95):
            
            if not self.circuit_breakers['emergency_mode']:
                self.circuit_breakers['emergency_mode'] = True
                logger.critical("🚨 EMERGENCY MODE ACTIVATED - All trading suspended")
                
                alert = RiskAlert(
                    severity=AlertSeverity.EMERGENCY,
                    message="Emergency mode activated due to excessive risk",
                    affected_tier="all",
                    recommended_actions=['stop_all_trading', 'review_positions', 'reduce_exposure'],
                    timestamp=datetime.now(),
                    risk_score=1.0,
                    auto_execute=True
                )
                self.risk_alerts.append(alert)
        
        # Aggressive tier halt
        if (portfolio_risk.leverage_weighted_exposure > self.portfolio_limits['max_leverage_weighted'] * 1.5 or
            portfolio_risk.correlation_risk > self.portfolio_limits['correlation_emergency']):
            
            if not self.circuit_breakers['aggressive_halt']:
                self.circuit_breakers['aggressive_halt'] = True
                logger.warning("⚠️ Aggressive tier trading halted")
                
                alert = RiskAlert(
                    severity=AlertSeverity.CRITICAL,
                    message="Aggressive tier halted due to excessive risk",
                    affected_tier="aggressive",
                    recommended_actions=['halt_aggressive_trading', 'reduce_leverage'],
                    timestamp=datetime.now(),
                    risk_score=0.85,
                    auto_execute=True
                )
                self.risk_alerts.append(alert)
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        return {
            'circuit_breakers': self.circuit_breakers.copy(),
            'tier_limits': self.tier_limits,
            'portfolio_limits': self.portfolio_limits,
            'active_positions': len(self.positions),
            'recent_alerts': len([a for a in self.risk_alerts if a.timestamp > datetime.now() - timedelta(hours=24)]),
            'sizing_factors': self.sizing_factors.copy(),
            'positions_by_tier': {
                tier: len([p for p in self.positions.values() if p.risk_tier == tier])
                for tier in ['conservative', 'aggressive']
            }
        }
    
    def reset_circuit_breakers(self, breaker_type: str = 'all'):
        """Reset circuit breakers (admin function)"""
        if breaker_type == 'all':
            self.circuit_breakers = {
                'portfolio_halt': False,
                'aggressive_halt': False,
                'emergency_mode': False,
                'last_check': datetime.now()
            }
        else:
            if breaker_type in self.circuit_breakers:
                self.circuit_breakers[breaker_type] = False
        
        logger.info(f"🔄 Circuit breaker reset: {breaker_type}")