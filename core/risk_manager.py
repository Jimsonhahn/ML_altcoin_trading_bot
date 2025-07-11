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

logger = logging.getLogger(__name__)


class RiskManager:
    """Comprehensive risk management system for trading bot"""

    def __init__(self, settings):
        """Initialize Risk Manager with configuration"""
        self.settings = settings
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

        logger.info(f"Risk Manager initialized with max position size: ${self.max_position_size}")

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


