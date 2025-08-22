# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Risk Manager - Comprehensive Risk Management System
==================================================

Manages all risk-related aspects of trading:
- Position sizing
- Stop loss/Take profit calculation
- Portfolio risk metrics
- Drawdown protection
- Risk-adjusted position allocation
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Real-time Risk Integration
from core.realtime_risk_calculator import get_risk_calculator, RiskMetrics
from core.interfaces import global_event_bus

logger = logging.getLogger(__name__)


class RiskManager:
    """Comprehensive risk management system for trading bot"""

    def __init__(self, settings, position_manager=None):
        """Initialize Risk Manager with configuration"""
        self.settings = settings
        self.position_manager = position_manager  # Store reference to position manager
        self.risk_config = settings.get('risk_management', {})

        # Risk parameters
        self.max_position_size = self.risk_config.get('max_position_size', 1000)
        self.max_drawdown = self.risk_config.get('max_drawdown', 0.20)  # 20%
        self.stop_loss_pct = self.risk_config.get('stop_loss_percentage', 0.02)  # 2%
        self.take_profit_pct = self.risk_config.get('take_profit_percentage', 0.05)  # 5%
        self.max_positions = self.risk_config.get('max_positions', 5)
        self.risk_per_trade = self.risk_config.get('risk_per_trade', 0.02)  # 2% per trade

        # Portfolio tracking
        self.portfolio_value = 10000  # Starting value
        self.peak_portfolio_value = self.portfolio_value
        self.current_positions = {}
        self.trade_history = []

        # Real-time Risk Integration
        self._realtime_calculator = get_risk_calculator()
        self._risk_callbacks_registered = False
        self._critical_risk_threshold = self.max_drawdown * 0.8  # 80% of max drawdown
        
        # Event handlers for risk monitoring
        self._setup_risk_event_handlers()

        logger.info(f"Risk Manager initialized with max position size: ${self.max_position_size}")
        logger.info("Real-time risk calculation integrated")

    def calculate_max_position_size(self, symbol: str, current_price: float,
                                    account_balance: float) -> float:
        """Calculate maximum allowed position size based on risk parameters"""
        # Method 1: Fixed maximum
        max_fixed = self.max_position_size

        # Method 2: Percentage of portfolio
        max_portfolio_pct = account_balance * 0.1  # 10% max per position

        # Method 3: Based on risk per trade
        risk_amount = account_balance * self.risk_per_trade
        position_from_risk = risk_amount / self.stop_loss_pct

        # Take the minimum of all methods
        final_size = min(max_fixed, max_portfolio_pct, position_from_risk)

        # Ensure we can afford at least 1 unit
        min_order_size = current_price * 0.001  # 0.001 BTC for example
        if final_size < min_order_size:
            return 0

        logger.info(f"Max position size for {symbol}: ${final_size:.2f}")
        return round(final_size, 2)

    def calculate_stop_loss(self, entry_price: float,
                            risk_percentage: Optional[float] = None) -> float:
        """Calculate stop loss price"""
        risk_pct = risk_percentage or self.stop_loss_pct
        stop_loss = entry_price * (1 - risk_pct)
        return round(stop_loss, 2)

    def calculate_take_profit(self, entry_price: float,
                              profit_percentage: Optional[float] = None) -> float:
        """Calculate take profit price"""
        profit_pct = profit_percentage or self.take_profit_pct
        take_profit = entry_price * (1 + profit_pct)
        return round(take_profit, 2)

    def check_risk_limits(self, symbol: str, position_size: float,
                          entry_price: float) -> Tuple[bool, str]:
        """Check if a trade meets risk management criteria"""
        # Check 1: Maximum position size
        if position_size > self.max_position_size:
            return False, f"Position size ${position_size} exceeds max ${self.max_position_size}"

        # Check 2: Number of open positions
        if len(self.current_positions) >= self.max_positions:
            return False, f"Maximum {self.max_positions} positions already open"

        # Check 3: Portfolio exposure
        total_exposure = sum(pos['size'] for pos in self.current_positions.values())
        if (total_exposure + position_size) > (self.portfolio_value * 0.5):
            return False, "Total exposure would exceed 50% of portfolio"

        # Check 4: Drawdown protection
        current_drawdown = self._calculate_current_drawdown()
        if current_drawdown > self.max_drawdown * 0.8:  # 80% of max drawdown
            return False, f"Portfolio near maximum drawdown ({current_drawdown:.1%})"

        return True, "Risk checks passed"

    def get_portfolio_risk_metrics(self) -> Dict[str, Any]:
        """Calculate and return current portfolio risk metrics"""
        metrics = {
            'portfolio_value': self.portfolio_value,
            'peak_value': self.peak_portfolio_value,
            'current_drawdown': self._calculate_current_drawdown(),
            'open_positions': len(self.current_positions),
            'total_exposure': sum(pos['size'] for pos in self.current_positions.values()),
            'unrealized_pnl': sum(pos.get('unrealized_pnl', 0) for pos in self.current_positions.values()),
            'win_rate': self._calculate_win_rate(),
            'profit_factor': self._calculate_profit_factor(),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'max_consecutive_losses': self._calculate_max_consecutive_losses(),
            'risk_score': self._calculate_risk_score()
        }

        return metrics

    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak"""
        if self.peak_portfolio_value == 0:
            return 0
        return (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value

    def _calculate_win_rate(self) -> float:
        """Calculate win rate from trade history"""
        if not self.trade_history:
            return 0.5  # Default 50%

        winning_trades = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
        return winning_trades / len(self.trade_history)

    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if not self.trade_history:
            return 1.0

        gross_profit = sum(trade['pnl'] for trade in self.trade_history if trade['pnl'] > 0)
        gross_loss = abs(sum(trade['pnl'] for trade in self.trade_history if trade['pnl'] < 0))

        if gross_loss == 0:
            return gross_profit if gross_profit > 0 else 1.0

        return gross_profit / gross_loss

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from trade history"""
        if len(self.trade_history) < 2:
            return 0

        returns = [trade.get('pnl_pct', 0) for trade in self.trade_history]
        if not returns:
            return 0

        avg_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0

        # Annualized Sharpe (assuming daily trading)
        return (avg_return / std_return) * np.sqrt(252)

    def _calculate_max_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losses"""
        if not self.trade_history:
            return 0

        max_losses = 0
        current_losses = 0

        for trade in self.trade_history:
            if trade.get('pnl', 0) < 0:
                current_losses += 1
                max_losses = max(max_losses, current_losses)
            else:
                current_losses = 0

        return max_losses

    def _calculate_risk_score(self) -> float:
        """Calculate overall risk score (0-100, lower is better)"""
        score = 0

        # Drawdown component (0-30 points)
        drawdown = self._calculate_current_drawdown()
        score += (drawdown / self.max_drawdown) * 30

        # Exposure component (0-25 points)
        total_exposure = sum(pos['size'] for pos in self.current_positions.values())
        exposure_ratio = total_exposure / self.portfolio_value if self.portfolio_value > 0 else 0
        score += min(exposure_ratio * 25, 25)

        # Win rate component (0-20 points)
        win_rate = self._calculate_win_rate()
        score += (1 - win_rate) * 20

        # Consecutive losses component (0-15 points)
        max_losses = self._calculate_max_consecutive_losses()
        score += min(max_losses * 3, 15)

        # Volatility component (0-10 points)
        score += 5  # Default middle value

        return min(score, 100)

    def get_risk_adjusted_size(self, base_size: float, volatility: float,
                               market_conditions: Dict[str, Any]) -> float:
        """Adjust position size based on market conditions and volatility"""
        adjustment_factor = 1.0

        # Adjust for volatility (inverse relationship)
        avg_volatility = 0.02  # 2% baseline
        if volatility > avg_volatility:
            adjustment_factor *= avg_volatility / volatility

        # Adjust for market trend
        if market_conditions.get('trend', 'neutral') == 'strong_downtrend':
            adjustment_factor *= 0.5
        elif market_conditions.get('trend') == 'downtrend':
            adjustment_factor *= 0.75

        # Apply adjustment with bounds
        adjusted_size = base_size * adjustment_factor
        adjusted_size = max(adjusted_size, base_size * 0.25)  # Min 25% of base
        adjusted_size = min(adjusted_size, base_size * 1.5)  # Max 150% of base

        logger.info(f"Risk-adjusted size: ${base_size:.2f} -> ${adjusted_size:.2f} (factor: {adjustment_factor:.2f})")

        return round(adjusted_size, 2)

    def calculate_trailing_stop(self, entry_price: float, current_price: float,
                                trailing_pct: float = 0.02) -> float:
        """Calculate trailing stop loss price"""
        if current_price > entry_price:
            # In profit, trail from current price
            return current_price * (1 - trailing_pct)
        else:
            # Not in profit, use fixed stop loss
            return self.calculate_stop_loss(entry_price)

    def update_position(self, symbol: str, size: float, entry_price: float,
                        current_price: float, is_open: bool = True):
        """Update position tracking"""
        if is_open:
            self.current_positions[symbol] = {
                'size': size,
                'entry_price': entry_price,
                'current_price': current_price,
                'unrealized_pnl': (current_price - entry_price) / entry_price * size,
                'opened_at': datetime.now()
            }
        else:
            # Position closed
            if symbol in self.current_positions:
                position = self.current_positions[symbol]
                realized_pnl = (current_price - position['entry_price']) / position['entry_price'] * position['size']

                self.trade_history.append({
                    'symbol': symbol,
                    'size': position['size'],
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'pnl': realized_pnl,
                    'pnl_pct': (current_price - position['entry_price']) / position['entry_price'] * 100,
                    'opened_at': position['opened_at'],
                    'closed_at': datetime.now()
                })

                del self.current_positions[symbol]

                # Update portfolio value
                self.portfolio_value += realized_pnl
                self.peak_portfolio_value = max(self.peak_portfolio_value, self.portfolio_value)

    def calculate_position_risk(self, position_size: float, entry_price: float,
                                stop_loss_price: float) -> Dict[str, float]:
        """Calculate risk metrics for a position"""
        # Calculate potential loss
        price_diff_pct = abs(entry_price - stop_loss_price) / entry_price
        potential_loss = position_size * price_diff_pct

        # Calculate R-value (risk units)
        r_value = potential_loss / self.portfolio_value

        return {
            'potential_loss': potential_loss,
            'potential_loss_pct': price_diff_pct * 100,
            'r_value': r_value,
            'portfolio_impact_pct': (potential_loss / self.portfolio_value) * 100
        }

    def get_risk_report(self) -> str:
        """Generate a comprehensive risk report"""
        metrics = self.get_portfolio_risk_metrics()

        report = f"""
=== RISK MANAGEMENT REPORT ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PORTFOLIO STATUS:
- Current Value: ${metrics['portfolio_value']:,.2f}
- Peak Value: ${metrics['peak_value']:,.2f}
- Current Drawdown: {metrics['current_drawdown']:.1%}
- Risk Score: {metrics['risk_score']:.1f}/100

POSITION SUMMARY:
- Open Positions: {metrics['open_positions']}/{self.max_positions}
- Total Exposure: ${metrics['total_exposure']:,.2f}
- Unrealized P&L: ${metrics['unrealized_pnl']:,.2f}

PERFORMANCE METRICS:
- Win Rate: {metrics['win_rate']:.1%}
- Profit Factor: {metrics['profit_factor']:.2f}
- Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
- Max Consecutive Losses: {metrics['max_consecutive_losses']}

RISK LIMITS:
- Max Position Size: ${self.max_position_size:,.2f}
- Max Drawdown: {self.max_drawdown:.1%}
- Risk Per Trade: {self.risk_per_trade:.1%}
- Stop Loss: {self.stop_loss_pct:.1%}
- Take Profit: {self.take_profit_pct:.1%}
"""

        return report

    def _setup_risk_event_handlers(self):
        """Setup event handlers for real-time risk monitoring"""
        if not self._risk_callbacks_registered:
            # Register callback for critical risk events
            self._realtime_calculator.add_risk_callback(self._on_risk_metrics_update)
            
            # Subscribe to risk limit breaches
            global_event_bus.subscribe("risk_limit_breached", self._on_risk_limit_breached)
            global_event_bus.subscribe("risk_metrics_update", self._on_realtime_risk_update)
            
            self._risk_callbacks_registered = True
            logger.info("Risk event handlers registered")
    
    def _on_risk_metrics_update(self, metrics: RiskMetrics):
        """Handle real-time risk metrics updates"""
        try:
            # Update internal portfolio tracking with real-time data
            if metrics.timestamp:
                self.portfolio_value = self._realtime_calculator._current_portfolio_value
                self.peak_portfolio_value = self._realtime_calculator._peak_portfolio_value
            
            # Log significant risk changes
            if metrics.risk_level in ["HIGH", "CRITICAL"]:
                logger.warning(f"Risk Level: {metrics.risk_level} - Drawdown: {metrics.current_drawdown:.1%}")
                
                # Notify other systems of elevated risk
                global_event_bus.publish("risk_level_change", {
                    'level': metrics.risk_level,
                    'drawdown': metrics.current_drawdown,
                    'warnings': metrics.warnings
                })
        
        except Exception as e:
            logger.error(f"Error processing risk metrics update: {e}")
    
    def _on_risk_limit_breached(self, data: Dict[str, Any]):
        """Handle risk limit breach events"""
        drawdown = data.get('drawdown', 0)
        limit = data.get('limit', self.max_drawdown)
        
        logger.critical(f"RISK LIMIT BREACHED: {drawdown:.1%} > {limit:.1%}")
        
        # Trigger emergency risk management
        self._trigger_emergency_risk_management(drawdown)
    
    def _on_realtime_risk_update(self, data: Dict[str, Any]):
        """Handle real-time risk metric updates"""
        metrics_data = data.get('metrics')
        if isinstance(metrics_data, RiskMetrics):
            # Update current positions from real-time data
            self._sync_positions_with_realtime()
    
    def _trigger_emergency_risk_management(self, current_drawdown: float):
        """Trigger emergency risk management procedures"""
        logger.critical("EMERGENCY RISK MANAGEMENT ACTIVATED")
        
        # Publish emergency stop signal
        global_event_bus.publish("emergency_risk_stop", {
            'drawdown': current_drawdown,
            'timestamp': datetime.now().isoformat(),
            'reason': 'Risk limit breached'
        })
    
    def _sync_positions_with_realtime(self):
        """Synchronize position tracking with real-time calculator"""
        try:
            realtime_positions = self._realtime_calculator._current_positions
            
            # Update position tracking
            for symbol, rt_position in realtime_positions.items():
                if rt_position['quantity'] != 0:
                    self.current_positions[symbol] = {
                        'size': abs(rt_position['quantity'] * rt_position['avg_price']),
                        'entry_price': rt_position['avg_price'],
                        'current_price': self._realtime_calculator._get_current_price(symbol),
                        'unrealized_pnl': self._calculate_position_pnl(symbol, rt_position),
                        'opened_at': rt_position.get('entry_time', datetime.now())
                    }
        
        except Exception as e:
            logger.error(f"Error syncing positions with real-time calculator: {e}")
    
    def _calculate_position_pnl(self, symbol: str, rt_position: Dict[str, Any]) -> float:
        """Calculate P&L for a position from real-time data"""
        try:
            current_price = self._realtime_calculator._get_current_price(symbol)
            if not current_price:
                return 0.0
            
            entry_price = rt_position['avg_price']
            quantity = rt_position['quantity']
            side = rt_position.get('side', 'long')
            
            if side == 'long':
                return (current_price - entry_price) * quantity
            else:  # short
                return (entry_price - current_price) * quantity
        
        except Exception as e:
            logger.error(f"Error calculating position P&L for {symbol}: {e}")
            return 0.0
    
    def start_realtime_monitoring(self, initial_capital: float):
        """Start real-time risk monitoring"""
        try:
            self._realtime_calculator.start_monitoring(initial_capital)
            logger.info(f"Real-time risk monitoring started with {initial_capital} capital")
        except Exception as e:
            logger.error(f"Failed to start real-time risk monitoring: {e}")
    
    def stop_realtime_monitoring(self):
        """Stop real-time risk monitoring"""
        try:
            self._realtime_calculator.stop_monitoring()
            logger.info("Real-time risk monitoring stopped")
        except Exception as e:
            logger.error(f"Error stopping real-time risk monitoring: {e}")
    
    def get_realtime_risk_metrics(self) -> Optional[RiskMetrics]:
        """Get current real-time risk metrics"""
        try:
            return self._realtime_calculator.get_current_metrics()
        except Exception as e:
            logger.error(f"Error getting real-time risk metrics: {e}")
            return None
    
    def update_realtime_position(self, symbol: str, quantity: float, avg_price: float, side: str = 'long'):
        """Update position in real-time calculator"""
        try:
            self._realtime_calculator.update_position(symbol, quantity, avg_price, side)
        except Exception as e:
            logger.error(f"Error updating real-time position for {symbol}: {e}")
    
    def update_realtime_price(self, symbol: str, price: float):
        """Update price in real-time calculator"""
        try:
            self._realtime_calculator.update_price(symbol, price)
        except Exception as e:
            logger.error(f"Error updating real-time price for {symbol}: {e}")
    
    def get_enhanced_portfolio_metrics(self) -> Dict[str, Any]:
        """Get enhanced portfolio metrics combining legacy and real-time data"""
        # Get legacy metrics
        legacy_metrics = self.get_portfolio_risk_metrics()
        
        # Get real-time metrics
        realtime_metrics = self.get_realtime_risk_metrics()
        
        # Combine metrics
        enhanced_metrics = legacy_metrics.copy()
        
        if realtime_metrics:
            enhanced_metrics.update({
                'realtime_drawdown': realtime_metrics.current_drawdown,
                'realtime_var': realtime_metrics.portfolio_var,
                'realtime_sharpe': realtime_metrics.sharpe_ratio,
                'realtime_risk_level': realtime_metrics.risk_level,
                'realtime_warnings': realtime_metrics.warnings,
                'position_concentration': realtime_metrics.position_concentration,
                'correlation_risk': realtime_metrics.correlation_risk,
                'liquidity_risk': realtime_metrics.liquidity_risk,
                'last_update': realtime_metrics.timestamp.isoformat()
            })
        
        return enhanced_metrics
    
    def can_open_position(self, symbol: str, signal_type: str, current_price: float) -> bool:
        """
        Check if a new position can be opened based on risk management rules
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            signal_type: 'BUY' or 'SELL'
            current_price: Current market price
            
        Returns:
            True if position can be opened, False otherwise
        """
        try:
            # Check if we already have too many open positions
            if len(self.current_positions) >= self.max_positions:
                logger.warning(f"Cannot open {signal_type} position for {symbol}: Max positions ({self.max_positions}) reached")
                return False
            
            # Check if we already have a position in this symbol
            if symbol in self.current_positions:
                logger.warning(f"Cannot open {signal_type} position for {symbol}: Position already exists")
                return False
            
            # Calculate position size for this trade
            max_size = self.calculate_max_position_size(symbol, current_price, self.portfolio_value)
            if max_size <= 0:
                logger.warning(f"Cannot open {signal_type} position for {symbol}: Max position size is 0")
                return False
            
            # Check current drawdown
            current_drawdown = self._calculate_current_drawdown()
            if current_drawdown >= self.max_drawdown:
                logger.warning(f"Cannot open {signal_type} position for {symbol}: Max drawdown exceeded ({current_drawdown:.2%} >= {self.max_drawdown:.2%})")
                return False
            
            # Check portfolio risk
            risk_metrics = self.get_portfolio_risk_metrics()
            risk_score = risk_metrics.get('risk_score', 0)
            if risk_score > 80:  # High risk threshold
                logger.warning(f"Cannot open {signal_type} position for {symbol}: Portfolio risk too high (score: {risk_score})")
                return False
            
            logger.info(f"✓ Risk check passed for {signal_type} {symbol} at ${current_price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error in risk check for {symbol}: {e}")
            return False  # Fail-safe: reject if error
    
    def calculate_position_size(self, symbol: str, signal_type: str, current_price: float, 
                              confidence: float = 1.0, account_balance: float = None) -> float:
        """
        Calculate the actual position size for a trade
        
        Args:
            symbol: Trading symbol
            signal_type: 'BUY' or 'SELL'
            current_price: Current market price
            confidence: Signal confidence (0-1)
            account_balance: Available balance (optional)
            
        Returns:
            Position size in base currency units
        """
        try:
            # Use provided balance or default to portfolio value
            balance = account_balance or self.portfolio_value
            
            # Calculate base position size (percentage of balance)
            risk_percentage = self.risk_per_trade * confidence  # Adjust by confidence
            position_value = balance * risk_percentage
            
            # Convert to position size in base currency units
            base_position_size = position_value / current_price
            
            # Apply maximum position size limits
            max_size = self.calculate_max_position_size(symbol, current_price, balance)
            position_size = min(base_position_size, max_size)
            
            # Apply minimum size (avoid dust trades)
            min_trade_value = 10.0  # $10 minimum trade
            min_size = min_trade_value / current_price
            
            if position_size < min_size:
                logger.info(f"Position size too small for {symbol}: ${position_size * current_price:.2f} < ${min_trade_value}")
                return 0.0
            
            logger.info(f"Calculated position size for {signal_type} {symbol}: {position_size:.6f} (${position_size * current_price:.2f})")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0


