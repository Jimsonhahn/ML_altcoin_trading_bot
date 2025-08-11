#!/usr/bin/env python3
"""
Orchestrator Portfolio Management System
========================================

Advanced portfolio management for the self-discovering orchestrator:
- Paper Trading vs Live Trading mode
- Position tracking and management
- Portfolio rebalancing
- Risk allocation per strategy
- Capital preservation
- Multi-exchange portfolio aggregation
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)

class TradingMode(Enum):
    """Trading mode enumeration"""
    PAPER = "paper"
    LIVE = "live"
    HYBRID = "hybrid"  # Paper for new strategies, live for proven ones

@dataclass
class Position:
    """Individual position in portfolio"""
    symbol: str
    strategy: str
    side: str  # 'long' or 'short'
    entry_price: float
    current_price: float
    quantity: float
    entry_time: datetime
    
    # Risk metrics
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_position_size: float = 0.0
    
    # Performance
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    
    # Mode
    is_paper: bool = True
    exchange: str = "binance"
    
    @property
    def position_value(self) -> float:
        """Current position value"""
        return self.quantity * self.current_price
    
    @property
    def pnl_percent(self) -> float:
        """PnL percentage"""
        if self.entry_price == 0:
            return 0.0
        
        if self.side == 'long':
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
        else:  # short
            return ((self.entry_price - self.current_price) / self.entry_price) * 100

@dataclass
class PortfolioState:
    """Current portfolio state"""
    timestamp: datetime
    total_value: float
    cash_balance: float
    positions_value: float
    
    # By mode
    paper_value: float
    live_value: float
    
    # Performance metrics
    total_pnl: float
    daily_pnl: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Risk metrics
    var_95: float
    portfolio_beta: float
    correlation_risk: float
    concentration_risk: float
    
    # Position details
    total_positions: int
    positions_by_strategy: Dict[str, int]
    positions_by_symbol: Dict[str, int]
    
    # Allocation
    strategy_allocations: Dict[str, float]
    symbol_allocations: Dict[str, float]

class PortfolioManager:
    """
    Advanced portfolio management system
    
    Handles both paper and live trading with sophisticated risk management
    """
    
    def __init__(self, initial_capital: float = 10000.0,
                 mode: TradingMode = TradingMode.PAPER,
                 config_path: str = "orchestrator_config.json"):
        
        self.initial_capital = initial_capital
        self.mode = mode
        self.config = self._load_config(config_path)
        
        # Portfolio state
        self.cash_balance = initial_capital
        self.paper_balance = initial_capital
        self.live_balance = 0.0 if mode == TradingMode.PAPER else initial_capital
        
        # Positions
        self.positions: Dict[str, Position] = {}  # position_id -> Position
        self.closed_positions: List[Position] = []
        
        # Strategy allocations
        self.strategy_capital: Dict[str, float] = {}
        self.strategy_limits: Dict[str, float] = {}
        
        # Performance tracking
        self.portfolio_history: List[PortfolioState] = []
        self.trade_history: List[Dict[str, Any]] = []
        
        # Risk parameters
        self.max_positions = self.config.get('max_positions', 20)
        self.max_position_size = self.config.get('max_position_size', 0.1)  # 10% max per position
        self.max_strategy_allocation = self.config.get('max_strategy_allocation', 0.3)  # 30% max per strategy
        self.correlation_limit = self.config.get('correlation_limit', 0.7)
        
        logger.info(f"💼 Portfolio Manager initialized in {mode.value} mode with ${initial_capital:,.2f}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration"""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
                return config.get('portfolio_management', {})
        return {}
    
    async def allocate_capital_to_strategies(self, strategy_weights: Dict[str, float],
                                           risk_budget: float) -> Dict[str, float]:
        """Allocate capital to strategies based on weights"""
        
        # Determine available capital based on mode
        if self.mode == TradingMode.PAPER:
            available_capital = self.paper_balance
        elif self.mode == TradingMode.LIVE:
            available_capital = self.live_balance
        else:  # HYBRID
            available_capital = self.paper_balance + self.live_balance
        
        # Apply risk budget constraint
        total_allocation = min(available_capital * 0.95, risk_budget)  # Keep 5% cash reserve
        
        allocations = {}
        
        for strategy, weight in strategy_weights.items():
            if weight > 0:
                # Calculate allocation
                strategy_allocation = total_allocation * weight
                
                # Apply per-strategy limit
                max_allowed = available_capital * self.max_strategy_allocation
                strategy_allocation = min(strategy_allocation, max_allowed)
                
                # Check existing exposure
                current_exposure = self._get_strategy_exposure(strategy)
                additional_allowed = max_allowed - current_exposure
                
                allocations[strategy] = min(strategy_allocation, additional_allowed)
                self.strategy_capital[strategy] = allocations[strategy]
        
        logger.info(f"📊 Capital allocated: {json.dumps(allocations, indent=2)}")
        return allocations
    
    async def open_position(self, strategy: str, symbol: str, side: str,
                          quantity: float, price: float,
                          stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None,
                          force_paper: bool = False) -> Optional[Position]:
        """Open a new position"""
        
        # Determine if paper or live
        is_paper = force_paper or self.mode == TradingMode.PAPER
        
        if self.mode == TradingMode.HYBRID:
            # Use strategy performance to decide
            is_paper = not self._is_strategy_proven(strategy)
        
        # Check risk limits
        position_value = quantity * price
        
        if not await self._check_risk_limits(strategy, symbol, position_value):
            logger.warning(f"❌ Position rejected due to risk limits: {symbol} ${position_value:.2f}")
            return None
        
        # Create position
        position_id = f"{strategy}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        position = Position(
            symbol=symbol,
            strategy=strategy,
            side=side,
            entry_price=price,
            current_price=price,
            quantity=quantity,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_position_size=position_value,
            is_paper=is_paper
        )
        
        # Update balances
        if is_paper:
            self.paper_balance -= position_value
        else:
            self.live_balance -= position_value
        
        self.positions[position_id] = position
        
        # Log trade
        self.trade_history.append({
            'timestamp': datetime.now(),
            'position_id': position_id,
            'action': 'open',
            'strategy': strategy,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'value': position_value,
            'is_paper': is_paper
        })
        
        mode_str = "📝 PAPER" if is_paper else "💰 LIVE"
        logger.info(f"{mode_str} Opened {side} position: {symbol} qty={quantity} @ ${price:.2f}")
        
        return position
    
    async def close_position(self, position_id: str, close_price: float,
                           reason: str = "signal") -> Optional[float]:
        """Close a position"""
        
        if position_id not in self.positions:
            logger.warning(f"Position {position_id} not found")
            return None
        
        position = self.positions[position_id]
        position.current_price = close_price
        
        # Calculate PnL
        if position.side == 'long':
            pnl = (close_price - position.entry_price) * position.quantity
        else:  # short
            pnl = (position.entry_price - close_price) * position.quantity
        
        position.realized_pnl = pnl
        
        # Update balances
        close_value = position.quantity * close_price
        
        if position.is_paper:
            self.paper_balance += close_value
        else:
            self.live_balance += close_value
        
        # Move to closed positions
        self.closed_positions.append(position)
        del self.positions[position_id]
        
        # Log trade
        self.trade_history.append({
            'timestamp': datetime.now(),
            'position_id': position_id,
            'action': 'close',
            'strategy': position.strategy,
            'symbol': position.symbol,
            'side': position.side,
            'quantity': position.quantity,
            'price': close_price,
            'pnl': pnl,
            'reason': reason,
            'is_paper': position.is_paper
        })
        
        mode_str = "📝 PAPER" if position.is_paper else "💰 LIVE"
        pnl_str = f"+${pnl:.2f}" if pnl > 0 else f"-${abs(pnl):.2f}"
        logger.info(f"{mode_str} Closed {position.symbol}: {pnl_str} ({reason})")
        
        return pnl
    
    async def update_positions(self, market_prices: Dict[str, float]):
        """Update all position prices and check stops"""
        
        positions_to_close = []
        
        for position_id, position in self.positions.items():
            if position.symbol in market_prices:
                old_price = position.current_price
                position.current_price = market_prices[position.symbol]
                
                # Update unrealized PnL
                if position.side == 'long':
                    position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
                else:  # short
                    position.unrealized_pnl = (position.entry_price - position.current_price) * position.quantity
                
                # Check stop loss
                if position.stop_loss:
                    if (position.side == 'long' and position.current_price <= position.stop_loss) or \
                       (position.side == 'short' and position.current_price >= position.stop_loss):
                        positions_to_close.append((position_id, position.stop_loss, 'stop_loss'))
                
                # Check take profit
                if position.take_profit:
                    if (position.side == 'long' and position.current_price >= position.take_profit) or \
                       (position.side == 'short' and position.current_price <= position.take_profit):
                        positions_to_close.append((position_id, position.take_profit, 'take_profit'))
        
        # Close positions that hit stops
        for position_id, price, reason in positions_to_close:
            await self.close_position(position_id, price, reason)
    
    async def rebalance_portfolio(self, target_allocations: Dict[str, float]):
        """Rebalance portfolio to target allocations"""
        
        current_state = await self.get_portfolio_state()
        current_allocations = current_state.strategy_allocations
        
        logger.info("🔄 Rebalancing portfolio...")
        
        for strategy, target_weight in target_allocations.items():
            current_weight = current_allocations.get(strategy, 0.0)
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) > 0.02:  # 2% threshold
                if weight_diff > 0:
                    # Need to increase allocation
                    additional_capital = current_state.total_value * weight_diff
                    self.strategy_capital[strategy] = self.strategy_capital.get(strategy, 0) + additional_capital
                    logger.info(f"   ↗️ {strategy}: +{weight_diff:.1%} (${additional_capital:.2f})")
                    
                else:
                    # Need to decrease allocation
                    reduction = current_state.total_value * abs(weight_diff)
                    
                    # Close positions proportionally
                    strategy_positions = [
                        (pid, pos) for pid, pos in self.positions.items() 
                        if pos.strategy == strategy
                    ]
                    
                    # Sort by PnL (close losers first)
                    strategy_positions.sort(key=lambda x: x[1].unrealized_pnl)
                    
                    reduced = 0
                    for pid, pos in strategy_positions:
                        if reduced >= reduction:
                            break
                        
                        await self.close_position(pid, pos.current_price, 'rebalance')
                        reduced += pos.position_value
                    
                    logger.info(f"   ↘️ {strategy}: {weight_diff:.1%} (${reduction:.2f})")
    
    async def get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio state"""
        
        # Calculate position values
        positions_value = sum(pos.position_value for pos in self.positions.values())
        paper_positions_value = sum(pos.position_value for pos in self.positions.values() if pos.is_paper)
        live_positions_value = sum(pos.position_value for pos in self.positions.values() if not pos.is_paper)
        
        # Total values
        total_value = self.cash_balance + positions_value
        paper_value = self.paper_balance + paper_positions_value
        live_value = self.live_balance + live_positions_value
        
        # Calculate metrics
        total_pnl = total_value - self.initial_capital
        daily_pnl = self._calculate_daily_pnl()
        win_rate = self._calculate_win_rate()
        sharpe_ratio = self._calculate_sharpe_ratio()
        max_drawdown = self._calculate_max_drawdown()
        
        # Risk metrics
        var_95 = self._calculate_var()
        portfolio_beta = self._calculate_portfolio_beta()
        correlation_risk = self._calculate_correlation_risk()
        concentration_risk = self._calculate_concentration_risk()
        
        # Position counts
        positions_by_strategy = {}
        positions_by_symbol = {}
        
        for pos in self.positions.values():
            positions_by_strategy[pos.strategy] = positions_by_strategy.get(pos.strategy, 0) + 1
            positions_by_symbol[pos.symbol] = positions_by_symbol.get(pos.symbol, 0) + 1
        
        # Allocations
        strategy_allocations = {}
        symbol_allocations = {}
        
        for pos in self.positions.values():
            strategy_allocations[pos.strategy] = strategy_allocations.get(pos.strategy, 0) + pos.position_value
            symbol_allocations[pos.symbol] = symbol_allocations.get(pos.symbol, 0) + pos.position_value
        
        # Convert to percentages
        if total_value > 0:
            strategy_allocations = {k: v/total_value for k, v in strategy_allocations.items()}
            symbol_allocations = {k: v/total_value for k, v in symbol_allocations.items()}
        
        state = PortfolioState(
            timestamp=datetime.now(),
            total_value=total_value,
            cash_balance=self.cash_balance,
            positions_value=positions_value,
            paper_value=paper_value,
            live_value=live_value,
            total_pnl=total_pnl,
            daily_pnl=daily_pnl,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            portfolio_beta=portfolio_beta,
            correlation_risk=correlation_risk,
            concentration_risk=concentration_risk,
            total_positions=len(self.positions),
            positions_by_strategy=positions_by_strategy,
            positions_by_symbol=positions_by_symbol,
            strategy_allocations=strategy_allocations,
            symbol_allocations=symbol_allocations
        )
        
        # Store in history
        self.portfolio_history.append(state)
        
        return state
    
    async def switch_mode(self, new_mode: TradingMode, transfer_positions: bool = False):
        """Switch trading mode"""
        
        old_mode = self.mode
        self.mode = new_mode
        
        logger.info(f"🔄 Switching from {old_mode.value} to {new_mode.value} mode")
        
        if transfer_positions and old_mode == TradingMode.PAPER and new_mode == TradingMode.LIVE:
            # Graduate paper positions to live
            positions_to_graduate = []
            
            for pid, pos in self.positions.items():
                if pos.is_paper and pos.unrealized_pnl > 0:  # Only profitable positions
                    positions_to_graduate.append((pid, pos))
            
            for pid, pos in positions_to_graduate:
                # Close paper position
                await self.close_position(pid, pos.current_price, 'mode_switch')
                
                # Reopen as live
                await self.open_position(
                    pos.strategy, pos.symbol, pos.side,
                    pos.quantity, pos.current_price,
                    pos.stop_loss, pos.take_profit,
                    force_paper=False
                )
            
            logger.info(f"📈 Graduated {len(positions_to_graduate)} profitable positions to live trading")
    
    def _get_strategy_exposure(self, strategy: str) -> float:
        """Get current exposure for a strategy"""
        exposure = 0.0
        for pos in self.positions.values():
            if pos.strategy == strategy:
                exposure += pos.position_value
        return exposure
    
    async def _check_risk_limits(self, strategy: str, symbol: str, position_value: float) -> bool:
        """Check if position passes risk limits"""
        
        # Check max positions
        if len(self.positions) >= self.max_positions:
            return False
        
        # Check position size
        total_value = self.cash_balance + sum(pos.position_value for pos in self.positions.values())
        position_percent = position_value / total_value
        
        if position_percent > self.max_position_size:
            return False
        
        # Check strategy allocation
        strategy_exposure = self._get_strategy_exposure(strategy) + position_value
        strategy_percent = strategy_exposure / total_value
        
        if strategy_percent > self.max_strategy_allocation:
            return False
        
        # Check symbol concentration
        symbol_exposure = sum(pos.position_value for pos in self.positions.values() if pos.symbol == symbol)
        symbol_exposure += position_value
        symbol_percent = symbol_exposure / total_value
        
        if symbol_percent > self.max_position_size * 2:  # Allow 2x for same symbol
            return False
        
        return True
    
    def _is_strategy_proven(self, strategy: str) -> bool:
        """Check if strategy is proven for live trading"""
        
        # Count paper trades for this strategy
        paper_trades = [
            t for t in self.trade_history 
            if t['strategy'] == strategy and t['is_paper'] and t['action'] == 'close'
        ]
        
        if len(paper_trades) < 20:  # Need at least 20 trades
            return False
        
        # Calculate paper trading performance
        total_pnl = sum(t.get('pnl', 0) for t in paper_trades)
        winning_trades = sum(1 for t in paper_trades if t.get('pnl', 0) > 0)
        win_rate = winning_trades / len(paper_trades)
        
        # Strategy is proven if profitable with good win rate
        return total_pnl > 0 and win_rate > 0.5
    
    def _calculate_daily_pnl(self) -> float:
        """Calculate today's PnL"""
        today = datetime.now().date()
        today_trades = [
            t for t in self.trade_history 
            if t['timestamp'].date() == today and t['action'] == 'close'
        ]
        return sum(t.get('pnl', 0) for t in today_trades)
    
    def _calculate_win_rate(self) -> float:
        """Calculate overall win rate"""
        closed_trades = [t for t in self.trade_history if t['action'] == 'close']
        if not closed_trades:
            return 0.0
        
        winning = sum(1 for t in closed_trades if t.get('pnl', 0) > 0)
        return winning / len(closed_trades)
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate portfolio Sharpe ratio"""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        returns = []
        for i in range(1, len(self.portfolio_history)):
            prev_value = self.portfolio_history[i-1].total_value
            curr_value = self.portfolio_history[i].total_value
            if prev_value > 0:
                returns.append((curr_value - prev_value) / prev_value)
        
        if not returns:
            return 0.0
        
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        return (avg_return / std_return) * np.sqrt(252)  # Annualized
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if not self.portfolio_history:
            return 0.0
        
        values = [state.total_value for state in self.portfolio_history]
        peak = values[0]
        max_dd = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _calculate_var(self) -> float:
        """Calculate Value at Risk (95%)"""
        # Simplified VaR calculation
        if len(self.portfolio_history) < 10:
            return 0.0
        
        returns = []
        for i in range(1, min(len(self.portfolio_history), 30)):
            prev_value = self.portfolio_history[i-1].total_value
            curr_value = self.portfolio_history[i].total_value
            if prev_value > 0:
                returns.append((curr_value - prev_value) / prev_value)
        
        if returns:
            return abs(np.percentile(returns, 5))
        return 0.0
    
    def _calculate_portfolio_beta(self) -> float:
        """Calculate portfolio beta (market sensitivity)"""
        # Simplified - would need market data for real calculation
        return 1.0
    
    def _calculate_correlation_risk(self) -> float:
        """Calculate correlation risk in portfolio"""
        # Check if positions are too correlated
        # Simplified version
        unique_symbols = len(set(pos.symbol for pos in self.positions.values()))
        total_positions = len(self.positions)
        
        if total_positions == 0:
            return 0.0
        
        concentration = 1 - (unique_symbols / total_positions)
        return concentration
    
    def _calculate_concentration_risk(self) -> float:
        """Calculate concentration risk"""
        if not self.positions:
            return 0.0
        
        # Calculate Herfindahl index
        total_value = sum(pos.position_value for pos in self.positions.values())
        if total_value == 0:
            return 0.0
        
        position_weights = [pos.position_value / total_value for pos in self.positions.values()]
        herfindahl = sum(w**2 for w in position_weights)
        
        return herfindahl
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        state = self.portfolio_history[-1] if self.portfolio_history else None
        
        if not state:
            return {}
        
        return {
            'overview': {
                'mode': self.mode.value,
                'total_value': state.total_value,
                'total_pnl': state.total_pnl,
                'total_pnl_percent': (state.total_pnl / self.initial_capital) * 100,
                'cash_balance': state.cash_balance,
                'positions_value': state.positions_value
            },
            'by_mode': {
                'paper_value': state.paper_value,
                'live_value': state.live_value,
                'paper_positions': sum(1 for p in self.positions.values() if p.is_paper),
                'live_positions': sum(1 for p in self.positions.values() if not p.is_paper)
            },
            'performance': {
                'daily_pnl': state.daily_pnl,
                'win_rate': state.win_rate,
                'sharpe_ratio': state.sharpe_ratio,
                'max_drawdown': state.max_drawdown,
                'total_trades': len([t for t in self.trade_history if t['action'] == 'close'])
            },
            'risk': {
                'var_95': state.var_95,
                'correlation_risk': state.correlation_risk,
                'concentration_risk': state.concentration_risk,
                'largest_position': max((p.position_value for p in self.positions.values()), default=0)
            },
            'allocation': {
                'by_strategy': state.strategy_allocations,
                'by_symbol': state.symbol_allocations,
                'total_positions': state.total_positions
            }
        }

# Utility functions
async def create_portfolio_manager(mode: str = "paper", 
                                 initial_capital: float = 10000.0) -> PortfolioManager:
    """Create portfolio manager with specified mode"""
    
    trading_mode = TradingMode(mode.lower())
    manager = PortfolioManager(initial_capital, trading_mode)
    
    logger.info(f"💼 Created portfolio manager in {trading_mode.value} mode")
    return manager

if __name__ == "__main__":
    # Test portfolio manager
    async def test_portfolio():
        print("💼 Testing Portfolio Management System...")
        
        # Create manager
        manager = await create_portfolio_manager("paper", 10000)
        
        # Test allocation
        strategy_weights = {
            'momentum_strategy': 0.3,
            'mean_reversion': 0.2,
            'arbitrage': 0.1
        }
        
        allocations = await manager.allocate_capital_to_strategies(strategy_weights, 5000)
        print(f"\n📊 Capital Allocations: {allocations}")
        
        # Test position opening
        position = await manager.open_position(
            'momentum_strategy', 'BTC/USDT', 'long',
            0.1, 45000, stop_loss=44000, take_profit=46000
        )
        print(f"\n📈 Opened position: {position}")
        
        # Update prices
        await manager.update_positions({'BTC/USDT': 45500})
        
        # Get state
        state = await manager.get_portfolio_state()
        print(f"\n💰 Portfolio State:")
        print(f"   Total Value: ${state.total_value:,.2f}")
        print(f"   PnL: ${state.total_pnl:,.2f}")
        
        # Performance summary
        summary = manager.get_performance_summary()
        print(f"\n📊 Performance Summary: {json.dumps(summary, indent=2)}")
        
        print("\n✅ Portfolio manager test completed!")
    
    asyncio.run(test_portfolio())