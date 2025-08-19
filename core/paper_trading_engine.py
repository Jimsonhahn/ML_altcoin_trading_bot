"""
Paper Trading Engine for Janics Freedom Factory
Simulates real trading without using actual funds
"""
import uuid
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import asyncio
from decimal import Decimal

logger = logging.getLogger(__name__)


@dataclass
class VirtualPosition:
    """Represents a virtual trading position"""
    id: str
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    size: float
    entry_price: float
    current_price: float
    timestamp: datetime
    strategy: str
    status: str  # 'OPEN', 'CLOSED'
    fee: float
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    duration_minutes: Optional[int] = None


class PaperTradingEngine:
    """
    Revolutionary Paper Trading Engine for risk-free strategy testing
    Simulates all trading operations with virtual money
    """
    
    def __init__(self, initial_balance: float = 10000.0, exchange_client=None):
        self.initial_balance = initial_balance
        self.virtual_balance = initial_balance
        self.virtual_positions: Dict[str, VirtualPosition] = {}
        self.trade_history: List[VirtualPosition] = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0
        }
        self.exchange_client = exchange_client
        self.daily_pnl_history = []
        self.peak_balance = initial_balance
        self.current_drawdown = 0.0
        
        # Trading limits
        self.max_position_size = 0.1  # Max 10% of portfolio per trade
        self.max_open_positions = 10
        self.min_trade_amount = 10.0  # Minimum $10 per trade
        
        logger.info(f"🏭 Paper Trading Engine initialized with ${initial_balance} virtual balance")
    
    def generate_trade_id(self) -> str:
        """Generate unique trade ID"""
        return f"PAPER_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current market price for symbol"""
        if self.exchange_client:
            try:
                ticker = await self.exchange_client.fetch_ticker(symbol)
                return ticker['last']
            except Exception as e:
                logger.error(f"Error fetching price for {symbol}: {e}")
                # Return last known price or raise
                raise
        else:
            # Mock price for testing
            return 100.0
    
    def calculate_position_size(self, balance_percentage: float, price: float) -> float:
        """Calculate position size based on balance percentage"""
        position_value = self.virtual_balance * balance_percentage
        position_size = position_value / price
        return position_size
    
    async def execute_virtual_trade(
        self,
        symbol: str,
        side: str,
        size: float,
        price: Optional[float] = None,
        strategy: str = "manual",
        balance_percentage: Optional[float] = None
    ) -> Optional[VirtualPosition]:
        """
        Execute a virtual trade without using real money
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            side: 'LONG' or 'SHORT'
            size: Position size (if not provided, use balance_percentage)
            price: Entry price (if not provided, use current market price)
            strategy: Strategy name that triggered the trade
            balance_percentage: Percentage of balance to use (if size not provided)
        
        Returns:
            VirtualPosition object if successful, None otherwise
        """
        try:
            # Validation checks
            if len(self.virtual_positions) >= self.max_open_positions:
                logger.warning(f"Maximum open positions ({self.max_open_positions}) reached")
                return None
            
            # Get current price if not provided
            if price is None:
                price = await self.get_current_price(symbol)
            
            # Calculate size from balance percentage if not provided
            if size is None and balance_percentage is not None:
                size = self.calculate_position_size(balance_percentage, price)
            
            # Calculate trade costs
            fee_rate = 0.001  # 0.1% trading fee
            slippage_rate = 0.0005  # 0.05% slippage
            
            # Apply slippage
            if side == 'LONG':
                entry_price = price * (1 + slippage_rate)
            else:
                entry_price = price * (1 - slippage_rate)
            
            # Calculate fees
            position_value = size * entry_price
            fee = position_value * fee_rate
            
            # Check if we have enough balance
            total_cost = position_value + fee
            if total_cost > self.virtual_balance:
                logger.warning(f"Insufficient virtual balance. Required: ${total_cost:.2f}, Available: ${self.virtual_balance:.2f}")
                return None
            
            # Check minimum trade amount
            if position_value < self.min_trade_amount:
                logger.warning(f"Trade value ${position_value:.2f} below minimum ${self.min_trade_amount}")
                return None
            
            # Create virtual position
            trade_id = self.generate_trade_id()
            virtual_position = VirtualPosition(
                id=trade_id,
                symbol=symbol,
                side=side,
                size=size,
                entry_price=entry_price,
                current_price=entry_price,
                timestamp=datetime.now(),
                strategy=strategy,
                status='OPEN',
                fee=fee
            )
            
            # Update virtual balance (deduct cost for long positions)
            if side == 'LONG':
                self.virtual_balance -= total_cost
            
            # Store position
            self.virtual_positions[trade_id] = virtual_position
            
            logger.info(f"📝 Paper Trade Executed: {side} {size:.4f} {symbol} @ ${entry_price:.2f}")
            logger.info(f"💰 Virtual Balance: ${self.virtual_balance:.2f}")
            
            return virtual_position
            
        except Exception as e:
            logger.error(f"Error executing virtual trade: {e}")
            return None
    
    async def close_virtual_trade(
        self,
        trade_id: str,
        exit_price: Optional[float] = None
    ) -> Optional[VirtualPosition]:
        """
        Close a virtual trading position and calculate P&L
        
        Args:
            trade_id: ID of the trade to close
            exit_price: Exit price (if not provided, use current market price)
            
        Returns:
            Updated VirtualPosition with P&L data
        """
        try:
            # Get position
            position = self.virtual_positions.get(trade_id)
            if not position:
                logger.error(f"Position {trade_id} not found")
                return None
            
            # Get exit price if not provided
            if exit_price is None:
                exit_price = await self.get_current_price(position.symbol)
            
            # Apply slippage for exit
            slippage_rate = 0.0005
            if position.side == 'LONG':
                exit_price = exit_price * (1 - slippage_rate)
            else:
                exit_price = exit_price * (1 + slippage_rate)
            
            # Calculate P&L
            if position.side == 'LONG':
                pnl = position.size * (exit_price - position.entry_price)
            else:
                pnl = position.size * (position.entry_price - exit_price)
            
            # Deduct exit fee
            exit_fee = position.size * exit_price * 0.001
            pnl -= (position.fee + exit_fee)
            
            # Calculate percentage P&L
            position_value = position.size * position.entry_price
            pnl_percentage = (pnl / position_value) * 100
            
            # Calculate duration
            duration = datetime.now() - position.timestamp
            duration_minutes = int(duration.total_seconds() / 60)
            
            # Update position
            position.exit_price = exit_price
            position.exit_timestamp = datetime.now()
            position.pnl = pnl
            position.pnl_percentage = pnl_percentage
            position.duration_minutes = duration_minutes
            position.status = 'CLOSED'
            position.fee += exit_fee
            
            # Update virtual balance
            if position.side == 'LONG':
                self.virtual_balance += (position.size * exit_price)
            else:
                # For short positions, return the borrowed amount and add/subtract P&L
                self.virtual_balance += pnl
            
            # Move to history
            self.trade_history.append(position)
            del self.virtual_positions[trade_id]
            
            # Update performance metrics
            self._update_performance_metrics(position)
            
            logger.info(f"📊 Paper Trade Closed: {position.symbol}")
            logger.info(f"   P&L: ${pnl:.2f} ({pnl_percentage:.2f}%)")
            logger.info(f"   Duration: {duration_minutes} minutes")
            logger.info(f"💰 Virtual Balance: ${self.virtual_balance:.2f}")
            
            return position
            
        except Exception as e:
            logger.error(f"Error closing virtual trade: {e}")
            return None
    
    def _update_performance_metrics(self, closed_position: VirtualPosition):
        """Update performance metrics after closing a position"""
        self.performance_metrics['total_trades'] += 1
        self.performance_metrics['total_pnl'] += closed_position.pnl
        
        if closed_position.pnl > 0:
            self.performance_metrics['winning_trades'] += 1
            self.performance_metrics['consecutive_wins'] += 1
            self.performance_metrics['consecutive_losses'] = 0
            
            if self.performance_metrics['consecutive_wins'] > self.performance_metrics['max_consecutive_wins']:
                self.performance_metrics['max_consecutive_wins'] = self.performance_metrics['consecutive_wins']
        else:
            self.performance_metrics['losing_trades'] += 1
            self.performance_metrics['consecutive_losses'] += 1
            self.performance_metrics['consecutive_wins'] = 0
            
            if self.performance_metrics['consecutive_losses'] > self.performance_metrics['max_consecutive_losses']:
                self.performance_metrics['max_consecutive_losses'] = self.performance_metrics['consecutive_losses']
        
        # Update best/worst trade
        if closed_position.pnl > self.performance_metrics['best_trade']:
            self.performance_metrics['best_trade'] = closed_position.pnl
        if closed_position.pnl < self.performance_metrics['worst_trade']:
            self.performance_metrics['worst_trade'] = closed_position.pnl
        
        # Update win rate
        if self.performance_metrics['total_trades'] > 0:
            self.performance_metrics['win_rate'] = (
                self.performance_metrics['winning_trades'] / 
                self.performance_metrics['total_trades']
            ) * 100
        
        # Update drawdown
        current_balance = self.virtual_balance
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        
        self.current_drawdown = ((self.peak_balance - current_balance) / self.peak_balance) * 100
        if self.current_drawdown > self.performance_metrics['max_drawdown']:
            self.performance_metrics['max_drawdown'] = self.current_drawdown
    
    async def update_open_positions(self):
        """Update current prices for all open positions"""
        for position in self.virtual_positions.values():
            try:
                current_price = await self.get_current_price(position.symbol)
                position.current_price = current_price
            except Exception as e:
                logger.error(f"Error updating price for {position.symbol}: {e}")
    
    def get_virtual_portfolio_status(self) -> Dict:
        """Get comprehensive virtual portfolio status"""
        # Calculate unrealized P&L
        unrealized_pnl = 0.0
        open_positions_data = []
        
        for position in self.virtual_positions.values():
            # Calculate current P&L
            if position.side == 'LONG':
                position_pnl = position.size * (position.current_price - position.entry_price)
            else:
                position_pnl = position.size * (position.entry_price - position.current_price)
            
            position_pnl -= position.fee  # Deduct fees
            unrealized_pnl += position_pnl
            
            # Add position data
            open_positions_data.append({
                'id': position.id,
                'symbol': position.symbol,
                'side': position.side,
                'size': position.size,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'pnl': position_pnl,
                'pnl_percentage': (position_pnl / (position.size * position.entry_price)) * 100,
                'strategy': position.strategy,
                'duration': int((datetime.now() - position.timestamp).total_seconds() / 60)
            })
        
        # Calculate total portfolio value
        total_portfolio_value = self.virtual_balance + unrealized_pnl
        
        # Calculate daily P&L
        daily_pnl = total_portfolio_value - self.initial_balance
        daily_pnl_percentage = (daily_pnl / self.initial_balance) * 100
        
        return {
            'mode': 'PAPER TRADING',
            'initial_balance': self.initial_balance,
            'virtual_balance': self.virtual_balance,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': self.performance_metrics['total_pnl'],
            'total_portfolio_value': total_portfolio_value,
            'daily_pnl': daily_pnl,
            'daily_pnl_percentage': daily_pnl_percentage,
            'open_positions': len(self.virtual_positions),
            'open_positions_data': open_positions_data,
            'total_trades': self.performance_metrics['total_trades'],
            'winning_trades': self.performance_metrics['winning_trades'],
            'losing_trades': self.performance_metrics['losing_trades'],
            'win_rate': self.performance_metrics['win_rate'],
            'max_drawdown': self.performance_metrics['max_drawdown'],
            'current_drawdown': self.current_drawdown,
            'best_trade': self.performance_metrics['best_trade'],
            'worst_trade': self.performance_metrics['worst_trade'],
            'consecutive_wins': self.performance_metrics['consecutive_wins'],
            'consecutive_losses': self.performance_metrics['consecutive_losses'],
            'performance_metrics': self.performance_metrics
        }
    
    def reset_paper_account(self):
        """Reset paper trading account to initial state"""
        self.virtual_balance = self.initial_balance
        self.virtual_positions.clear()
        self.trade_history.clear()
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0
        }
        self.peak_balance = self.initial_balance
        self.current_drawdown = 0.0
        logger.info("🔄 Paper trading account reset to initial state")
    
    def export_trade_history(self, filepath: str = "paper_trades_history.json"):
        """Export trade history to JSON file"""
        history_data = {
            'export_timestamp': datetime.now().isoformat(),
            'initial_balance': self.initial_balance,
            'final_balance': self.virtual_balance,
            'total_pnl': self.performance_metrics['total_pnl'],
            'performance_metrics': self.performance_metrics,
            'trades': [asdict(trade) for trade in self.trade_history]
        }
        
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2, default=str)
        
        logger.info(f"📁 Trade history exported to {filepath}")