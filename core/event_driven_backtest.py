#!/usr/bin/env python3
"""
Event-Driven Backtesting Framework
==================================

Realistic backtesting without lookahead bias for institutional-grade validation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Individual trade record"""
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime]
    exit_price: Optional[float]
    direction: str  # 'long' or 'short'
    size: float
    pnl: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    duration_hours: float = 0.0
    signal_strength: float = 0.0
    signal_quality: float = 0.0
    regime: str = 'unknown'
    
    @property
    def is_open(self) -> bool:
        return self.exit_time is None
    
    @property
    def return_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return self.pnl / (self.entry_price * self.size)


@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics"""
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration: float = 0.0
    commission_total: float = 0.0
    slippage_total: float = 0.0
    alpha_vs_buyhold: float = 0.0
    beta: float = 0.0
    volatility: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0


class EventDrivenBacktester:
    """
    Event-driven backtester that processes data tick-by-tick
    to eliminate lookahead bias completely
    """
    
    def __init__(self, initial_capital: float = 100000.0, commission_rate: float = 0.001,
                 slippage_rate: float = 0.0005, max_position_size: float = 0.8):
        """Initialize backtester"""
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.max_position_size = max_position_size
        
        # State tracking
        self.current_capital = initial_capital
        self.current_position = 0.0
        self.current_position_value = 0.0
        self.open_trades: List[BacktestTrade] = []
        self.closed_trades: List[BacktestTrade] = []
        self.equity_curve: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.peak_equity = initial_capital
        self.trough_equity = initial_capital
        self.max_drawdown_start = None
        self.max_drawdown_end = None
        
        logger.info(f"EventDrivenBacktester initialized with {initial_capital:,.0f} capital")
    
    def process_signal(self, timestamp: datetime, price: float, signal_data: Dict[str, Any]) -> bool:
        """
        Process a trading signal in event-driven manner
        
        Args:
            timestamp: Current timestamp
            price: Current market price
            signal_data: Signal from strategy containing direction, strength, etc.
            
        Returns:
            bool: True if trade was executed
        """
        try:
            signal_direction = signal_data.get('direction', 'hold')
            signal_strength = signal_data.get('strength', 0.0)
            signal_confidence = signal_data.get('confidence', 0.0)
            signal_quality = signal_data.get('quality_score', 0.0)
            regime = signal_data.get('regime', 'unknown')
            
            # Skip if holding or insufficient signal quality
            if signal_direction == 'hold' or signal_confidence < 0.3:
                self._update_equity_curve(timestamp, price, signal_data)
                return False
            
            # Calculate position size
            target_position_value = self._calculate_position_size(
                signal_strength, signal_confidence, price
            )
            
            # Execute trade if position size is meaningful
            if abs(target_position_value) > self.current_capital * 0.01:  # Minimum 1% position
                return self._execute_trade(
                    timestamp, price, target_position_value, signal_direction,
                    signal_strength, signal_quality, regime
                )
            else:
                self._update_equity_curve(timestamp, price, signal_data)
                return False
                
        except Exception as e:
            logger.error(f"Signal processing failed: {e}")
            return False
    
    def _calculate_position_size(self, signal_strength: float, signal_confidence: float, 
                                price: float) -> float:
        """Calculate position size based on signal quality and risk management"""
        try:
            # Base position size from signal strength
            base_size = abs(signal_strength) * self.max_position_size
            
            # Adjust by confidence
            adjusted_size = base_size * signal_confidence
            
            # Convert to dollar value
            max_dollar_value = self.current_capital * adjusted_size
            
            # Position sizing based on volatility (Kelly-like approach simplified)
            position_value = min(max_dollar_value, self.current_capital * 0.2)  # Max 20% per trade
            
            return position_value if signal_strength > 0 else -position_value
            
        except Exception as e:
            logger.error(f"Position size calculation failed: {e}")
            return 0.0
    
    def _execute_trade(self, timestamp: datetime, price: float, target_value: float,
                      direction: str, signal_strength: float, signal_quality: float,
                      regime: str) -> bool:
        """Execute a trade with realistic transaction costs"""
        try:
            # Close existing position if reversing
            if self.current_position != 0 and np.sign(target_value) != np.sign(self.current_position_value):
                self._close_position(timestamp, price, "signal_reversal")
            
            # Calculate shares to trade
            shares_to_trade = target_value / price
            
            # Apply slippage
            slippage = abs(target_value) * self.slippage_rate
            execution_price = price * (1 + self.slippage_rate) if target_value > 0 else price * (1 - self.slippage_rate)
            
            # Calculate commission
            commission = abs(target_value) * self.commission_rate
            
            # Create trade record
            trade = BacktestTrade(
                entry_time=timestamp,
                entry_price=execution_price,
                exit_time=None,
                exit_price=None,
                direction='long' if target_value > 0 else 'short',
                size=abs(shares_to_trade),
                commission=commission,
                slippage=slippage,
                signal_strength=signal_strength,
                signal_quality=signal_quality,
                regime=regime
            )
            
            # Update position
            self.current_position = shares_to_trade
            self.current_position_value = target_value
            
            # Update capital (subtract costs)
            self.current_capital -= (commission + slippage)
            
            # Add to open trades
            self.open_trades.append(trade)
            
            # Update equity curve
            self._update_equity_curve(timestamp, price, {
                'trade_executed': True,
                'direction': direction,
                'position_value': target_value
            })
            
            logger.debug(f"Trade executed: {direction} {abs(shares_to_trade):.4f} @ {execution_price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return False
    
    def _close_position(self, timestamp: datetime, price: float, reason: str = "signal"):
        """Close current position"""
        if not self.open_trades or self.current_position == 0:
            return
        
        try:
            # Get the most recent open trade
            trade = self.open_trades[-1]
            
            # Calculate exit price with slippage
            exit_slippage = abs(self.current_position_value) * self.slippage_rate
            exit_commission = abs(self.current_position_value) * self.commission_rate
            
            if trade.direction == 'long':
                exit_price = price * (1 - self.slippage_rate)
                pnl = (exit_price - trade.entry_price) * trade.size
            else:
                exit_price = price * (1 + self.slippage_rate)
                pnl = (trade.entry_price - exit_price) * trade.size
            
            # Update trade
            trade.exit_time = timestamp
            trade.exit_price = exit_price
            trade.pnl = pnl - trade.commission - exit_commission - trade.slippage - exit_slippage
            trade.duration_hours = (timestamp - trade.entry_time).total_seconds() / 3600
            
            # Update capital
            self.current_capital += pnl - exit_commission - exit_slippage
            
            # Reset position
            self.current_position = 0.0
            self.current_position_value = 0.0
            
            # Move to closed trades
            self.closed_trades.append(trade)
            self.open_trades.remove(trade)
            
            logger.debug(f"Position closed: PnL={pnl:.2f}, Duration={trade.duration_hours:.1f}h")
            
        except Exception as e:
            logger.error(f"Position closing failed: {e}")
    
    def _update_equity_curve(self, timestamp: datetime, price: float, signal_data: Dict[str, Any]):
        """Update equity curve with current portfolio value"""
        try:
            # Calculate unrealized PnL
            unrealized_pnl = 0.0
            if self.current_position != 0 and self.open_trades:
                trade = self.open_trades[-1]
                if trade.direction == 'long':
                    unrealized_pnl = (price - trade.entry_price) * trade.size
                else:
                    unrealized_pnl = (trade.entry_price - price) * trade.size
            
            # Total equity
            total_equity = self.current_capital + unrealized_pnl
            
            # Update drawdown tracking
            if total_equity > self.peak_equity:
                self.peak_equity = total_equity
                self.trough_equity = total_equity
            elif total_equity < self.trough_equity:
                self.trough_equity = total_equity
            
            # Add to equity curve
            equity_point = {
                'timestamp': timestamp,
                'price': price,
                'capital': self.current_capital,
                'unrealized_pnl': unrealized_pnl,
                'total_equity': total_equity,
                'position': self.current_position,
                'drawdown': (self.peak_equity - total_equity) / self.peak_equity if self.peak_equity > 0 else 0.0,
                'signal_data': signal_data
            }
            
            self.equity_curve.append(equity_point)
            
        except Exception as e:
            logger.error(f"Equity curve update failed: {e}")
    
    def finalize_backtest(self, final_timestamp: datetime, final_price: float) -> BacktestMetrics:
        """Finalize backtest and calculate comprehensive metrics"""
        try:
            # Close any remaining open positions
            if self.open_trades:
                self._close_position(final_timestamp, final_price, "backtest_end")
            
            # Calculate metrics
            metrics = self._calculate_metrics()
            
            logger.info(f"Backtest finalized: {len(self.closed_trades)} trades, "
                       f"{metrics.total_return:.2%} total return, {metrics.sharpe_ratio:.2f} Sharpe")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Backtest finalization failed: {e}")
            return BacktestMetrics()
    
    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics"""
        try:
            if not self.equity_curve or not self.closed_trades:
                return BacktestMetrics()
            
            # Basic performance
            final_equity = self.equity_curve[-1]['total_equity']
            total_return = (final_equity / self.initial_capital) - 1.0
            
            # Time period
            start_time = self.equity_curve[0]['timestamp']
            end_time = self.equity_curve[-1]['timestamp']
            days = (end_time - start_time).days
            years = days / 365.25 if days > 0 else 1
            
            annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0.0
            
            # Volatility and Sharpe
            returns = []
            for i in range(1, len(self.equity_curve)):
                prev_equity = self.equity_curve[i-1]['total_equity']
                curr_equity = self.equity_curve[i]['total_equity']
                if prev_equity > 0:
                    returns.append((curr_equity / prev_equity) - 1)
            
            volatility = np.std(returns) * np.sqrt(252) if returns else 0.0
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0.0
            
            # Drawdown
            max_drawdown = max([point['drawdown'] for point in self.equity_curve])
            
            # Trade statistics
            winning_trades = [t for t in self.closed_trades if t.pnl > 0]
            losing_trades = [t for t in self.closed_trades if t.pnl <= 0]
            
            win_rate = len(winning_trades) / len(self.closed_trades) if self.closed_trades else 0.0
            
            total_wins = sum(t.pnl for t in winning_trades)
            total_losses = abs(sum(t.pnl for t in losing_trades))
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0
            
            # Other metrics
            avg_trade_duration = np.mean([t.duration_hours for t in self.closed_trades]) if self.closed_trades else 0.0
            commission_total = sum(t.commission for t in self.closed_trades)
            slippage_total = sum(t.slippage for t in self.closed_trades)
            
            # Sortino ratio (downside deviation)
            negative_returns = [r for r in returns if r < 0]
            downside_deviation = np.std(negative_returns) * np.sqrt(252) if negative_returns else 0.0
            sortino_ratio = annual_return / downside_deviation if downside_deviation > 0 else 0.0
            
            # Calmar ratio
            calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0.0
            
            # Buy and hold comparison
            start_price = self.equity_curve[0]['price']
            end_price = self.equity_curve[-1]['price']
            buyhold_return = (end_price / start_price) - 1
            alpha_vs_buyhold = total_return - buyhold_return
            
            return BacktestMetrics(
                total_return=total_return,
                annual_return=annual_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_trade_return=np.mean([t.return_pct for t in self.closed_trades]) if self.closed_trades else 0.0,
                total_trades=len(self.closed_trades),
                winning_trades=len(winning_trades),
                losing_trades=len(losing_trades),
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=max([t.pnl for t in self.closed_trades]) if self.closed_trades else 0.0,
                largest_loss=min([t.pnl for t in self.closed_trades]) if self.closed_trades else 0.0,
                avg_trade_duration=avg_trade_duration,
                commission_total=commission_total,
                slippage_total=slippage_total,
                alpha_vs_buyhold=alpha_vs_buyhold,
                volatility=volatility,
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio
            )
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")
            return BacktestMetrics()
    
    def export_results(self, filename: str = None) -> Dict[str, Any]:
        """Export backtest results to JSON"""
        if filename is None:
            filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            metrics = self._calculate_metrics()
            
            results = {
                'backtest_info': {
                    'initial_capital': self.initial_capital,
                    'commission_rate': self.commission_rate,
                    'slippage_rate': self.slippage_rate,
                    'max_position_size': self.max_position_size,
                    'final_capital': self.current_capital,
                    'total_trades': len(self.closed_trades),
                    'start_time': self.equity_curve[0]['timestamp'].isoformat() if self.equity_curve else None,
                    'end_time': self.equity_curve[-1]['timestamp'].isoformat() if self.equity_curve else None
                },
                'performance_metrics': {
                    'total_return': metrics.total_return,
                    'annual_return': metrics.annual_return,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'max_drawdown': metrics.max_drawdown,
                    'win_rate': metrics.win_rate,
                    'profit_factor': metrics.profit_factor,
                    'alpha_vs_buyhold': metrics.alpha_vs_buyhold,
                    'volatility': metrics.volatility,
                    'calmar_ratio': metrics.calmar_ratio,
                    'sortino_ratio': metrics.sortino_ratio
                },
                'trade_statistics': {
                    'total_trades': metrics.total_trades,
                    'winning_trades': metrics.winning_trades,
                    'losing_trades': metrics.losing_trades,
                    'avg_win': metrics.avg_win,
                    'avg_loss': metrics.avg_loss,
                    'largest_win': metrics.largest_win,
                    'largest_loss': metrics.largest_loss,
                    'avg_trade_duration_hours': metrics.avg_trade_duration,
                    'commission_total': metrics.commission_total,
                    'slippage_total': metrics.slippage_total
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Backtest results exported to {filename}")
            return results
            
        except Exception as e:
            logger.error(f"Results export failed: {e}")
            return {}
    
    def get_equity_curve_df(self) -> pd.DataFrame:
        """Get equity curve as pandas DataFrame"""
        try:
            if not self.equity_curve:
                return pd.DataFrame()
            
            df = pd.DataFrame(self.equity_curve)
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"Equity curve DataFrame creation failed: {e}")
            return pd.DataFrame()
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as pandas DataFrame"""
        try:
            if not self.closed_trades:
                return pd.DataFrame()
            
            trades_data = []
            for trade in self.closed_trades:
                trades_data.append({
                    'entry_time': trade.entry_time,
                    'exit_time': trade.exit_time,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'direction': trade.direction,
                    'size': trade.size,
                    'pnl': trade.pnl,
                    'return_pct': trade.return_pct,
                    'duration_hours': trade.duration_hours,
                    'signal_strength': trade.signal_strength,
                    'signal_quality': trade.signal_quality,
                    'regime': trade.regime,
                    'commission': trade.commission,
                    'slippage': trade.slippage
                })
            
            return pd.DataFrame(trades_data)
            
        except Exception as e:
            logger.error(f"Trades DataFrame creation failed: {e}")
            return pd.DataFrame()