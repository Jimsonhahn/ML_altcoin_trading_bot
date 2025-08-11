#!/usr/bin/env python3
"""
Paper Trading Engine - Event-Driven Live Strategy Testing
=========================================================

Real-time paper trading for the Ultimate BTC Strategy without risk
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PaperTrade:
    """Paper trade record"""
    id: str
    entry_time: datetime
    entry_price: float
    direction: str  # 'long' or 'short'
    size: float
    signal_strength: float
    signal_confidence: float
    regime: str
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    exit_reason: str = ""
    
    @property
    def is_open(self) -> bool:
        return self.exit_time is None
    
    @property
    def duration_minutes(self) -> float:
        if self.exit_time:
            return (self.exit_time - self.entry_time).total_seconds() / 60
        return (datetime.now() - self.entry_time).total_seconds() / 60
    
    @property
    def unrealized_pnl(self) -> float:
        if self.is_open:
            return self.pnl  # Current unrealized PnL
        return 0.0


@dataclass
class PaperTradingMetrics:
    """Real-time paper trading metrics"""
    total_trades: int = 0
    open_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    total_return: float = 0.0
    daily_pnl: List[float] = None
    
    def __post_init__(self):
        if self.daily_pnl is None:
            self.daily_pnl = []


class PaperTradingEngine:
    """
    Real-time paper trading engine that simulates live trading
    without actual financial risk
    """
    
    def __init__(self, initial_capital: float = 100000.0, commission_rate: float = 0.001,
                 slippage_rate: float = 0.0005, max_position_size: float = 0.8):
        """Initialize paper trading engine"""
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.max_position_size = max_position_size
        
        # Trading state
        self.is_active = False
        self.start_time = None
        self.last_update = None
        
        # Trades and metrics
        self.trades: List[PaperTrade] = []
        self.equity_history: List[Dict[str, Any]] = []
        self.daily_snapshots: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.peak_equity = initial_capital
        self.current_equity = initial_capital
        self.max_drawdown = 0.0
        
        # Strategy integration
        self.strategy = None
        self.last_signal = None
        
        logger.info(f"Paper Trading Engine initialized with ${initial_capital:,.0f}")
    
    def start_trading(self, strategy_adapter) -> bool:
        """Start paper trading with given strategy"""
        try:
            self.strategy = strategy_adapter
            self.is_active = True
            self.start_time = datetime.now()
            self.last_update = self.start_time
            
            # Initialize equity tracking
            self._update_equity_history(self.start_time, 0.0, "trading_started")
            
            logger.info(f"Paper trading started with {strategy_adapter.__class__.__name__}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start paper trading: {e}")
            return False
    
    def stop_trading(self) -> Dict[str, Any]:
        """Stop paper trading and return final metrics"""
        try:
            if not self.is_active:
                return {"error": "Trading not active"}
            
            # Close all open positions
            stop_time = datetime.now()
            for trade in self.trades:
                if trade.is_open:
                    # Simulate closing at current market price (would need real price feed)
                    self._close_trade(trade.id, 50000.0, stop_time, "trading_stopped")
            
            self.is_active = False
            final_metrics = self.get_current_metrics()
            
            logger.info(f"Paper trading stopped. Final PnL: ${final_metrics.total_pnl:,.2f}")
            return {"status": "stopped", "metrics": asdict(final_metrics)}
            
        except Exception as e:
            logger.error(f"Failed to stop paper trading: {e}")
            return {"error": str(e)}
    
    def process_market_update(self, price: float, volume: float, timestamp: datetime = None) -> Dict[str, Any]:
        """Process real-time market data update"""
        if not self.is_active or not self.strategy:
            return {"status": "inactive"}
        
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            # Update strategy with new market data
            market_state = self.strategy.process_market_tick(price, volume, timestamp)
            
            # Generate trading signal
            signal = self.strategy.generate_quantum_signal(market_state)
            self.last_signal = signal
            
            # Process signal for trading
            trade_result = self._process_trading_signal(signal, price, timestamp)
            
            # Update open positions with current price
            self._update_open_positions(price, timestamp)
            
            # Update equity tracking
            self._update_equity_history(timestamp, price, "market_update")
            
            # Update metrics
            current_metrics = self.get_current_metrics()
            
            self.last_update = timestamp
            
            return {
                "status": "active",
                "timestamp": timestamp.isoformat(),
                "price": price,
                "signal": signal,
                "trade_result": trade_result,
                "metrics": asdict(current_metrics),
                "open_trades": len([t for t in self.trades if t.is_open])
            }
            
        except Exception as e:
            logger.error(f"Market update processing failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _process_trading_signal(self, signal: Dict[str, Any], price: float, timestamp: datetime) -> Dict[str, Any]:
        """Process trading signal and execute paper trades"""
        try:
            direction = signal.get('direction', 'hold')
            strength = signal.get('strength', 0.0)
            confidence = signal.get('confidence', 0.0)
            
            # Skip if holding or weak signal
            if direction == 'hold' or confidence < 0.4 or strength < 0.3:
                return {"action": "hold", "reason": "weak_signal"}
            
            # Close existing position if reversing
            open_trades = [t for t in self.trades if t.is_open]
            if open_trades and direction != open_trades[-1].direction:
                for trade in open_trades:
                    self._close_trade(trade.id, price, timestamp, "signal_reversal")
            
            # Calculate position size
            position_size = self._calculate_position_size(strength, confidence)
            position_value = self.current_capital * position_size
            
            # Check minimum position size
            if position_value < self.current_capital * 0.02:  # Minimum 2%
                return {"action": "hold", "reason": "position_too_small"}
            
            # Open new position
            trade_id = f"trade_{len(self.trades) + 1}_{timestamp.strftime('%H%M%S')}"
            
            # Calculate costs
            commission = position_value * self.commission_rate
            slippage = position_value * self.slippage_rate
            
            # Adjust execution price for slippage
            execution_price = price * (1 + self.slippage_rate) if direction == 'buy' else price * (1 - self.slippage_rate)
            
            # Create paper trade
            paper_trade = PaperTrade(
                id=trade_id,
                entry_time=timestamp,
                entry_price=execution_price,
                direction='long' if direction == 'buy' else 'short',
                size=position_value / execution_price,
                signal_strength=strength,
                signal_confidence=confidence,
                regime=signal.get('regime', 'unknown'),
                commission=commission,
                slippage=slippage
            )
            
            self.trades.append(paper_trade)
            
            # Update capital (subtract costs)
            self.current_capital -= (commission + slippage)
            
            logger.info(f"Paper trade opened: {direction} ${position_value:,.0f} @ ${execution_price:.2f}")
            
            return {
                "action": "trade_opened",
                "trade_id": trade_id,
                "direction": direction,
                "size": position_value,
                "price": execution_price,
                "costs": commission + slippage
            }
            
        except Exception as e:
            logger.error(f"Signal processing failed: {e}")
            return {"action": "error", "error": str(e)}
    
    def _calculate_position_size(self, strength: float, confidence: float) -> float:
        """Calculate position size based on signal quality"""
        # Base position size from signal strength
        base_size = abs(strength) * self.max_position_size
        
        # Adjust by confidence
        adjusted_size = base_size * confidence
        
        # Conservative position sizing for paper trading
        final_size = min(adjusted_size * 0.5, 0.2)  # Max 20% per trade
        
        return final_size
    
    def _close_trade(self, trade_id: str, price: float, timestamp: datetime, reason: str = "signal") -> bool:
        """Close a paper trade"""
        try:
            trade = next((t for t in self.trades if t.id == trade_id and t.is_open), None)
            if not trade:
                return False
            
            # Calculate exit value
            gross_proceeds = trade.size * price
            exit_commission = gross_proceeds * self.commission_rate
            exit_slippage = gross_proceeds * self.slippage_rate
            net_proceeds = gross_proceeds - exit_commission - exit_slippage
            
            # Calculate PnL
            original_investment = trade.size * trade.entry_price
            pnl = net_proceeds - original_investment - trade.commission - trade.slippage
            
            # Update trade
            trade.exit_time = timestamp
            trade.exit_price = price
            trade.pnl = pnl
            trade.exit_reason = reason
            
            # Update capital
            self.current_capital += net_proceeds
            
            logger.info(f"Paper trade closed: {trade.direction} PnL=${pnl:.2f} ({reason})")
            return True
            
        except Exception as e:
            logger.error(f"Trade closing failed: {e}")
            return False
    
    def _update_open_positions(self, current_price: float, timestamp: datetime):
        """Update unrealized PnL for open positions"""
        try:
            for trade in self.trades:
                if trade.is_open:
                    # Calculate unrealized PnL
                    current_value = trade.size * current_price
                    original_investment = trade.size * trade.entry_price
                    
                    if trade.direction == 'long':
                        unrealized_pnl = current_value - original_investment
                    else:  # short
                        unrealized_pnl = original_investment - current_value
                    
                    # Subtract costs
                    unrealized_pnl -= (trade.commission + trade.slippage)
                    trade.pnl = unrealized_pnl
                    
        except Exception as e:
            logger.error(f"Position update failed: {e}")
    
    def _update_equity_history(self, timestamp: datetime, price: float, event: str):
        """Update equity curve history"""
        try:
            # Calculate total equity
            realized_pnl = sum(t.pnl for t in self.trades if not t.is_open)
            unrealized_pnl = sum(t.pnl for t in self.trades if t.is_open)
            total_equity = self.current_capital + unrealized_pnl
            
            # Update drawdown tracking
            if total_equity > self.peak_equity:
                self.peak_equity = total_equity
            
            current_drawdown = (self.peak_equity - total_equity) / self.peak_equity if self.peak_equity > 0 else 0
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
            
            self.current_equity = total_equity
            
            # Add to history
            equity_point = {
                "timestamp": timestamp,
                "price": price,
                "capital": self.current_capital,
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "total_equity": total_equity,
                "drawdown": current_drawdown,
                "event": event,
                "open_trades": len([t for t in self.trades if t.is_open])
            }
            
            self.equity_history.append(equity_point)
            
            # Limit history size
            if len(self.equity_history) > 10000:
                self.equity_history = self.equity_history[-5000:]
                
        except Exception as e:
            logger.error(f"Equity history update failed: {e}")
    
    def get_current_metrics(self) -> PaperTradingMetrics:
        """Calculate current performance metrics"""
        try:
            closed_trades = [t for t in self.trades if not t.is_open]
            open_trades = [t for t in self.trades if t.is_open]
            
            winning_trades = [t for t in closed_trades if t.pnl > 0]
            losing_trades = [t for t in closed_trades if t.pnl <= 0]
            
            total_pnl = sum(t.pnl for t in closed_trades)
            unrealized_pnl = sum(t.pnl for t in open_trades)
            
            win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
            
            total_return = (self.current_equity / self.initial_capital) - 1
            
            return PaperTradingMetrics(
                total_trades=len(closed_trades),
                open_trades=len(open_trades),
                winning_trades=len(winning_trades),
                losing_trades=len(losing_trades),
                total_pnl=total_pnl,
                unrealized_pnl=unrealized_pnl,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=max([t.pnl for t in closed_trades]) if closed_trades else 0,
                largest_loss=min([t.pnl for t in closed_trades]) if closed_trades else 0,
                current_drawdown=(self.peak_equity - self.current_equity) / self.peak_equity if self.peak_equity > 0 else 0,
                max_drawdown=self.max_drawdown,
                total_return=total_return
            )
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")
            return PaperTradingMetrics()
    
    def export_results(self, filename: str = None) -> Dict[str, Any]:
        """Export paper trading results"""
        if filename is None:
            filename = f"paper_trading_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            metrics = self.get_current_metrics()
            
            results = {
                "paper_trading_session": {
                    "start_time": self.start_time.isoformat() if self.start_time else None,
                    "end_time": datetime.now().isoformat(),
                    "duration_hours": (datetime.now() - self.start_time).total_seconds() / 3600 if self.start_time else 0,
                    "initial_capital": self.initial_capital,
                    "final_equity": self.current_equity,
                    "is_active": self.is_active
                },
                "performance_metrics": asdict(metrics),
                "trades": [asdict(trade) for trade in self.trades],
                "equity_curve": self.equity_history[-1000:],  # Last 1000 points
                "configuration": {
                    "commission_rate": self.commission_rate,
                    "slippage_rate": self.slippage_rate,
                    "max_position_size": self.max_position_size
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Paper trading results exported to {filename}")
            return results
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return {}
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data formatted for dashboard display"""
        try:
            metrics = self.get_current_metrics()
            recent_trades = self.trades[-10:] if self.trades else []
            
            return {
                "status": "active" if self.is_active else "inactive",
                "current_equity": self.current_equity,
                "total_pnl": metrics.total_pnl,
                "unrealized_pnl": metrics.unrealized_pnl,
                "total_return_pct": metrics.total_return * 100,
                "win_rate_pct": metrics.win_rate * 100,
                "current_drawdown_pct": metrics.current_drawdown * 100,
                "max_drawdown_pct": metrics.max_drawdown * 100,
                "total_trades": metrics.total_trades,
                "open_trades": metrics.open_trades,
                "last_signal": self.last_signal,
                "recent_trades": [
                    {
                        "id": t.id,
                        "direction": t.direction,
                        "entry_time": t.entry_time.isoformat(),
                        "entry_price": t.entry_price,
                        "pnl": t.pnl,
                        "is_open": t.is_open
                    } for t in recent_trades
                ],
                "equity_curve": self.equity_history[-100:] if self.equity_history else []
            }
            
        except Exception as e:
            logger.error(f"Dashboard data generation failed: {e}")
            return {"status": "error", "error": str(e)}