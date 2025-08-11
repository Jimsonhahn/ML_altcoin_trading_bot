"""
Portfolio Manager - State Management und Performance Tracking
Kernkomponente für Portfolio-Zustandsverwaltung und Feedback-Loop zum QuantumOrchestrator
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json

from .event_models import (
    Event, EventType, FillEvent, MarketEvent, PortfolioUpdateEvent,
    OrderSide
)
from .event_bus import EventBus

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Repräsentiert eine Portfolio-Position"""
    symbol: str
    quantity: float = 0.0
    average_cost: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    last_price: float = 0.0
    
    # Tracking
    total_traded_quantity: float = 0.0
    total_commission_paid: float = 0.0
    trade_count: int = 0
    first_trade_time: Optional[datetime] = None
    last_trade_time: Optional[datetime] = None
    
    # Performance
    max_profit: float = 0.0
    max_loss: float = 0.0
    
    @property
    def market_value(self) -> float:
        """Aktueller Marktwert der Position"""
        return self.quantity * self.last_price
    
    @property
    def total_cost_basis(self) -> float:
        """Gesamte Kostenbasis"""
        return abs(self.quantity) * self.average_cost
    
    @property
    def total_pnl(self) -> float:
        """Gesamt-PnL (realized + unrealized)"""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def return_percentage(self) -> float:
        """Return in Prozent"""
        if self.total_cost_basis > 0:
            return self.total_pnl / self.total_cost_basis
        return 0.0


@dataclass
class PortfolioSnapshot:
    """Portfolio-Snapshot zu einem bestimmten Zeitpunkt"""
    timestamp: datetime
    total_equity: float
    cash_balance: float
    positions_value: float
    total_pnl: float
    daily_pnl: float
    positions_count: int
    leverage: float
    margin_ratio: float


@dataclass
class TradeRecord:
    """Einzelner Trade für Portfolio-Tracking"""
    timestamp: datetime
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float
    trade_id: str
    signal_id: str
    strategy: str = ""
    
    @property
    def notional_value(self) -> float:
        return self.quantity * self.price


class PortfolioManager:
    """
    Portfolio Manager mit umfassendem State Management
    
    Features:
    - Real-time Position Tracking
    - P&L Calculation (realized & unrealized)
    - Performance Metrics
    - Risk Metrics
    - Feedback für QuantumOrchestrator
    - Portfolio Rebalancing Support
    """
    
    def __init__(self,
                 event_bus: EventBus,
                 initial_capital: float = 1000000.0,
                 base_currency: str = "USDT",
                 enable_margin: bool = False,
                 max_leverage: float = 1.0):
        
        self.event_bus = event_bus
        self.initial_capital = initial_capital
        self.base_currency = base_currency
        self.enable_margin = enable_margin
        self.max_leverage = max_leverage
        
        # Portfolio State
        self.cash_balance = initial_capital
        self.positions: Dict[str, Position] = {}
        self.market_prices: Dict[str, float] = {}
        
        # Trade History
        self.trade_history: List[TradeRecord] = []
        self.fill_history: List[FillEvent] = []
        
        # Performance Tracking
        self.equity_curve: List[PortfolioSnapshot] = []
        self.daily_snapshots: Dict[str, PortfolioSnapshot] = {}  # date -> snapshot
        
        # Risk Metrics
        self.drawdown_history: deque = deque(maxlen=252)  # 1 year
        self.returns_history: deque = deque(maxlen=252)
        self.volatility_window: deque = deque(maxlen=30)  # 30 days
        
        # Strategy Allocation Tracking
        self.strategy_allocations: Dict[str, float] = defaultdict(float)
        self.strategy_pnl: Dict[str, float] = defaultdict(float)
        
        # Commission Tracking
        self.total_commission_paid = 0.0
        self.commission_by_symbol: Dict[str, float] = defaultdict(float)
        
        # Performance Cache
        self._last_portfolio_update = datetime.now()
        self._update_frequency = timedelta(seconds=10)  # Min update frequency
        
        # Subscribe to events
        self._subscribe_to_events()
        
        logger.info(f"PortfolioManager initialisiert mit {initial_capital:,.0f} {base_currency}")
    
    def _subscribe_to_events(self):
        """Registriert Event Handler"""
        self.event_bus.subscribe(EventType.FILL, self._handle_fill_event)
        self.event_bus.subscribe(EventType.MARKET, self._handle_market_event)
    
    async def _handle_fill_event(self, event: FillEvent) -> None:
        """Verarbeitet Fill Events und aktualisiert Portfolio"""
        try:
            await self._process_fill(event)
            await self._update_portfolio_state()
            
        except Exception as e:
            logger.error(f"Error handling fill event: {e}")
    
    async def _handle_market_event(self, event: MarketEvent) -> None:
        """Aktualisiert Marktpreise für P&L Berechnung"""
        try:
            self.market_prices[event.symbol] = event.close
            
            # Update unrealized P&L für dieses Symbol
            if event.symbol in self.positions:
                await self._update_unrealized_pnl(event.symbol, event.close)
            
            # Periodische Portfolio-Updates
            if datetime.now() - self._last_portfolio_update > self._update_frequency:
                await self._update_portfolio_state()
                
        except Exception as e:
            logger.error(f"Error handling market event: {e}")
    
    async def _process_fill(self, fill_event: FillEvent) -> None:
        """Verarbeitet Fill und aktualisiert Position"""
        
        symbol = fill_event.symbol
        
        # Skip if rejected fill
        if fill_event.fill_quantity == 0:
            return
        
        # Create trade record
        trade_record = TradeRecord(
            timestamp=fill_event.timestamp,
            symbol=symbol,
            side=fill_event.side.value,
            quantity=fill_event.fill_quantity,
            price=fill_event.fill_price,
            commission=fill_event.commission,
            trade_id=fill_event.fill_id,
            signal_id=fill_event.signal_id
        )
        
        self.trade_history.append(trade_record)
        self.fill_history.append(fill_event)
        
        # Update commission tracking
        self.total_commission_paid += fill_event.commission
        self.commission_by_symbol[symbol] += fill_event.commission
        
        # Get or create position
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        
        position = self.positions[symbol]
        
        # Update position based on side
        if fill_event.side == OrderSide.BUY:
            await self._process_buy_fill(position, fill_event)
        else:
            await self._process_sell_fill(position, fill_event)
        
        # Update position metadata
        position.trade_count += 1
        position.total_traded_quantity += fill_event.fill_quantity
        position.total_commission_paid += fill_event.commission
        position.last_trade_time = fill_event.timestamp
        
        if position.first_trade_time is None:
            position.first_trade_time = fill_event.timestamp
        
        # Update cash balance
        trade_value = fill_event.fill_quantity * fill_event.fill_price
        
        if fill_event.side == OrderSide.BUY:
            self.cash_balance -= (trade_value + fill_event.commission)
        else:
            self.cash_balance += (trade_value - fill_event.commission)
        
        logger.debug(f"Fill processed: {symbol} {fill_event.side.value} "
                    f"{fill_event.fill_quantity:.4f} @ {fill_event.fill_price:.2f}")
    
    async def _process_buy_fill(self, position: Position, fill_event: FillEvent) -> None:
        """Verarbeitet Buy Fill"""
        
        old_quantity = position.quantity
        old_cost = position.average_cost
        
        new_quantity = old_quantity + fill_event.fill_quantity
        
        if new_quantity != 0:
            # Weighted average cost
            total_cost = (old_quantity * old_cost) + (fill_event.fill_quantity * fill_event.fill_price)
            position.average_cost = total_cost / new_quantity
        
        position.quantity = new_quantity
    
    async def _process_sell_fill(self, position: Position, fill_event: FillEvent) -> None:
        """Verarbeitet Sell Fill"""
        
        # Calculate realized P&L for sold quantity
        realized_pnl = (fill_event.fill_price - position.average_cost) * fill_event.fill_quantity
        position.realized_pnl += realized_pnl
        
        # Update quantity
        position.quantity -= fill_event.fill_quantity
        
        # If position closed or flipped, may need to adjust average cost
        if position.quantity < 0 and position.average_cost > 0:
            # Position flipped to short - reset average cost
            position.average_cost = fill_event.fill_price
            position.quantity = -fill_event.fill_quantity
    
    async def _update_unrealized_pnl(self, symbol: str, current_price: float) -> None:
        """Aktualisiert unrealized P&L für Symbol"""
        
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        position.last_price = current_price
        
        if position.quantity != 0:
            # Calculate unrealized P&L
            position.unrealized_pnl = (current_price - position.average_cost) * position.quantity
            
            # Update max profit/loss tracking
            if position.unrealized_pnl > position.max_profit:
                position.max_profit = position.unrealized_pnl
            elif position.unrealized_pnl < position.max_loss:
                position.max_loss = position.unrealized_pnl
    
    async def _update_portfolio_state(self) -> None:
        """Aktualisiert Portfolio-Status und publiziert Update Event"""
        
        # Calculate portfolio metrics
        positions_value = sum(pos.market_value for pos in self.positions.values())
        total_equity = self.cash_balance + positions_value
        
        total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_pnl = total_realized_pnl + total_unrealized_pnl
        
        # Calculate daily P&L
        daily_pnl = 0.0
        today = datetime.now().date()
        
        if str(today) in self.daily_snapshots:
            yesterday_equity = self.daily_snapshots[str(today)].total_equity
            daily_pnl = total_equity - yesterday_equity
        
        # Calculate leverage
        leverage = 0.0
        if total_equity > 0:
            gross_exposure = sum(abs(pos.market_value) for pos in self.positions.values())
            leverage = gross_exposure / total_equity
        
        # Risk metrics
        current_drawdown = await self._calculate_current_drawdown(total_equity)
        portfolio_var = await self._calculate_portfolio_var()
        
        # Create portfolio snapshot
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            total_equity=total_equity,
            cash_balance=self.cash_balance,
            positions_value=positions_value,
            total_pnl=total_pnl,
            daily_pnl=daily_pnl,
            positions_count=len([pos for pos in self.positions.values() if pos.quantity != 0]),
            leverage=leverage,
            margin_ratio=total_equity / self.initial_capital if self.initial_capital > 0 else 1.0
        )
        
        # Add to equity curve
        self.equity_curve.append(snapshot)
        
        # Update daily snapshot
        self.daily_snapshots[str(today)] = snapshot
        
        # Calculate returns for risk metrics
        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2].total_equity
            daily_return = (total_equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            self.returns_history.append(daily_return)
        
        # Track drawdown
        self.drawdown_history.append(current_drawdown)
        
        # Create and publish portfolio update event
        portfolio_update = PortfolioUpdateEvent(
            timestamp=datetime.now(),
            total_equity=total_equity,
            cash_balance=self.cash_balance,
            positions_value=positions_value,
            positions={
                symbol: {
                    'quantity': pos.quantity,
                    'avg_price': pos.average_cost,
                    'current_price': pos.last_price,
                    'market_value': pos.market_value,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'realized_pnl': pos.realized_pnl
                }
                for symbol, pos in self.positions.items()
                if pos.quantity != 0
            },
            current_leverage=leverage,
            portfolio_var=portfolio_var,
            max_drawdown=await self._calculate_max_drawdown(),
            current_drawdown=current_drawdown,
            allocation_by_strategy=dict(self.strategy_allocations),
            concentration_risk=await self._calculate_concentration_risk(),
            daily_pnl=daily_pnl,
            total_pnl=total_pnl,
            realized_pnl=total_realized_pnl,
            unrealized_pnl=total_unrealized_pnl
        )
        
        await self.event_bus.publish(portfolio_update)
        
        self._last_portfolio_update = datetime.now()
        
        logger.debug(f"Portfolio updated: Equity={total_equity:,.0f}, "
                    f"P&L={total_pnl:,.0f}, Positions={len(self.positions)}")
    
    async def _calculate_current_drawdown(self, current_equity: float) -> float:
        """Berechnet aktuellen Drawdown"""
        
        if not self.equity_curve:
            return 0.0
        
        max_equity = max(snapshot.total_equity for snapshot in self.equity_curve)
        
        if max_equity > 0:
            return (current_equity - max_equity) / max_equity
        
        return 0.0
    
    async def _calculate_max_drawdown(self) -> float:
        """Berechnet maximalen Drawdown"""
        
        if len(self.equity_curve) < 2:
            return 0.0
        
        equity_values = [snapshot.total_equity for snapshot in self.equity_curve]
        
        max_drawdown = 0.0
        peak = equity_values[0]
        
        for equity in equity_values:
            if equity > peak:
                peak = equity
            
            drawdown = (equity - peak) / peak if peak > 0 else 0
            if drawdown < max_drawdown:
                max_drawdown = drawdown
        
        return abs(max_drawdown)
    
    async def _calculate_portfolio_var(self, confidence: float = 0.05) -> float:
        """Berechnet Portfolio Value at Risk"""
        
        if len(self.returns_history) < 10:
            return 0.0
        
        returns = list(self.returns_history)
        
        # Sort returns and find VaR
        sorted_returns = sorted(returns)
        var_index = int(len(sorted_returns) * confidence)
        
        if var_index < len(sorted_returns):
            var_return = sorted_returns[var_index]
            
            # Convert to dollar amount
            current_equity = self.equity_curve[-1].total_equity if self.equity_curve else self.initial_capital
            var_amount = abs(var_return * current_equity)
            
            return var_amount
        
        return 0.0
    
    async def _calculate_concentration_risk(self) -> float:
        """Berechnet Konzentrations-Risiko (HHI)"""
        
        if not self.positions:
            return 0.0
        
        total_value = sum(abs(pos.market_value) for pos in self.positions.values())
        
        if total_value == 0:
            return 0.0
        
        # Calculate Herfindahl-Hirschman Index
        weights = [abs(pos.market_value) / total_value for pos in self.positions.values()]
        hhi = sum(w ** 2 for w in weights)
        
        return hhi
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Gibt Position für Symbol zurück"""
        return self.positions.get(symbol)
    
    def get_positions_summary(self) -> Dict[str, Any]:
        """Gibt Zusammenfassung aller Positionen zurück"""
        
        active_positions = {
            symbol: {
                'quantity': pos.quantity,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'realized_pnl': pos.realized_pnl,
                'total_pnl': pos.total_pnl,
                'return_pct': pos.return_percentage,
                'avg_cost': pos.average_cost,
                'last_price': pos.last_price
            }
            for symbol, pos in self.positions.items()
            if pos.quantity != 0
        }
        
        return active_positions
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Berechnet umfassende Performance-Metriken"""
        
        if len(self.equity_curve) < 2:
            return {}
        
        # Basic metrics
        current_equity = self.equity_curve[-1].total_equity
        total_return = (current_equity / self.initial_capital) - 1
        
        # Time-based metrics
        start_date = self.equity_curve[0].timestamp
        end_date = self.equity_curve[-1].timestamp
        days = (end_date - start_date).days
        
        annual_return = ((current_equity / self.initial_capital) ** (365 / max(days, 1))) - 1 if days > 0 else 0
        
        # Risk metrics
        returns = list(self.returns_history)
        
        if len(returns) > 1:
            daily_vol = np.std(returns, ddof=1)
            annual_vol = daily_vol * np.sqrt(252)
            
            # Sharpe Ratio (assuming 2% risk-free rate)
            risk_free_rate = 0.02
            sharpe_ratio = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
            
            # Sortino Ratio
            negative_returns = [r for r in returns if r < 0]
            downside_vol = np.std(negative_returns, ddof=1) * np.sqrt(252) if negative_returns else annual_vol
            sortino_ratio = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
        else:
            daily_vol = 0
            annual_vol = 0
            sharpe_ratio = 0
            sortino_ratio = 0
        
        # Max drawdown
        max_dd = asyncio.create_task(self._calculate_max_drawdown())
        
        # Trade statistics
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for pos in self.positions.values() if pos.realized_pnl > 0)
        
        win_rate = winning_trades / max(total_trades, 1)
        
        # Calculate profit factor
        gross_profit = sum(max(0, pos.realized_pnl) for pos in self.positions.values())
        gross_loss = sum(min(0, pos.realized_pnl) for pos in self.positions.values())
        profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'daily_volatility': daily_vol,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_dd.result() if max_dd.done() else 0.0,
            'current_drawdown': asyncio.create_task(self._calculate_current_drawdown(current_equity)).result(),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'total_commission': self.total_commission_paid,
            'commission_rate': self.total_commission_paid / max(self.initial_capital, 1),
            'days_analyzed': days,
            'calmar_ratio': annual_return / max(max_dd.result() if max_dd.done() else 0.01, 0.01)
        }
    
    def get_equity_curve_data(self) -> pd.DataFrame:
        """Gibt Equity Curve als DataFrame zurück"""
        
        if not self.equity_curve:
            return pd.DataFrame()
        
        data = []
        for snapshot in self.equity_curve:
            data.append({
                'timestamp': snapshot.timestamp,
                'total_equity': snapshot.total_equity,
                'cash_balance': snapshot.cash_balance,
                'positions_value': snapshot.positions_value,
                'total_pnl': snapshot.total_pnl,
                'daily_pnl': snapshot.daily_pnl,
                'leverage': snapshot.leverage
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def export_trade_history(self, filename: Optional[str] = None) -> str:
        """Exportiert Trade-History als JSON"""
        
        if filename is None:
            filename = f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        trade_data = []
        for trade in self.trade_history:
            trade_data.append({
                'timestamp': trade.timestamp.isoformat(),
                'symbol': trade.symbol,
                'side': trade.side,
                'quantity': trade.quantity,
                'price': trade.price,
                'commission': trade.commission,
                'notional_value': trade.notional_value,
                'trade_id': trade.trade_id,
                'signal_id': trade.signal_id
            })
        
        with open(filename, 'w') as f:
            json.dump({
                'portfolio_summary': {
                    'initial_capital': self.initial_capital,
                    'final_equity': self.equity_curve[-1].total_equity if self.equity_curve else self.initial_capital,
                    'total_trades': len(self.trade_history),
                    'total_commission': self.total_commission_paid
                },
                'trades': trade_data
            }, f, indent=2)
        
        return filename
    
    def reset_portfolio(self, new_initial_capital: Optional[float] = None) -> None:
        """Reset Portfolio für neuen Backtest"""
        
        if new_initial_capital:
            self.initial_capital = new_initial_capital
        
        self.cash_balance = self.initial_capital
        self.positions.clear()
        self.market_prices.clear()
        self.trade_history.clear()
        self.fill_history.clear()
        self.equity_curve.clear()
        self.daily_snapshots.clear()
        self.drawdown_history.clear()
        self.returns_history.clear()
        self.volatility_window.clear()
        self.strategy_allocations.clear()
        self.strategy_pnl.clear()
        self.total_commission_paid = 0.0
        self.commission_by_symbol.clear()
        
        logger.info(f"Portfolio reset mit {self.initial_capital:,.0f} {self.base_currency}")