# core/arbitrage_engine.py
"""
Advanced Arbitrage Engine for Cross-Exchange Trading
Detects price differences, calculates profitability, and executes arbitrage trades
"""

import asyncio
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from enum import Enum
import time

from .exchange_manager import MultiExchangeManager, OrderSide, OrderType, UnifiedOrder

# Try to import notifier
try:
    from utils.notifier import send_info, send_warning, send_error, send_critical
    NOTIFIER_AVAILABLE = True
except ImportError:
    NOTIFIER_AVAILABLE = False

logger = logging.getLogger(__name__)


class ArbitrageType(Enum):
    """Types of arbitrage opportunities"""
    SIMPLE = "simple"  # Buy on exchange A, sell on exchange B
    TRIANGULAR = "triangular"  # Three-way arbitrage within single exchange
    CROSS_TRIANGULAR = "cross_triangular"  # Triangular across exchanges
    FUNDING_RATE = "funding_rate"  # Funding rate arbitrage


class ArbitrageStatus(Enum):
    """Status of arbitrage opportunity"""
    DETECTED = "detected"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class ArbitrageOpportunity:
    """Represents an arbitrage opportunity"""
    id: str
    type: ArbitrageType
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    spread_percent: float
    profit_percent: float  # After fees
    max_volume: float
    min_profit_usd: float
    timestamp: datetime
    expiry: datetime
    status: ArbitrageStatus = ArbitrageStatus.DETECTED
    
    @property
    def is_expired(self) -> bool:
        """Check if opportunity has expired"""
        return datetime.now() > self.expiry
    
    @property
    def time_remaining_seconds(self) -> int:
        """Time remaining before expiry"""
        if self.is_expired:
            return 0
        return int((self.expiry - datetime.now()).total_seconds())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'type': self.type.value,
            'symbol': self.symbol,
            'buy_exchange': self.buy_exchange,
            'sell_exchange': self.sell_exchange,
            'buy_price': self.buy_price,
            'sell_price': self.sell_price,
            'spread_percent': self.spread_percent,
            'profit_percent': self.profit_percent,
            'max_volume': self.max_volume,
            'min_profit_usd': self.min_profit_usd,
            'timestamp': self.timestamp.isoformat(),
            'expiry': self.expiry.isoformat(),
            'status': self.status.value,
            'time_remaining': self.time_remaining_seconds
        }


@dataclass
class ArbitrageExecution:
    """Represents executed arbitrage trade"""
    opportunity_id: str
    buy_order: Optional[UnifiedOrder]
    sell_order: Optional[UnifiedOrder]
    executed_volume: float
    actual_profit_usd: float
    execution_time_ms: float
    fees_paid: Dict[str, float]
    slippage_percent: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class ArbitrageStats:
    """Arbitrage performance statistics"""
    total_opportunities: int
    executed_trades: int
    success_rate: float
    total_profit_usd: float
    average_profit_per_trade: float
    average_execution_time_ms: float
    best_profit_trade: float
    worst_profit_trade: float
    total_fees_paid: float


class SmartOrderRouter:
    """
    Smart order routing for optimal execution
    """
    
    def __init__(self, exchange_manager: MultiExchangeManager):
        self.exchange_manager = exchange_manager
        self.latency_cache = {}
        self.last_latency_update = {}
        
    async def find_best_execution_venue(self, symbol: str, side: OrderSide, 
                                       amount: float) -> Optional[Tuple[str, float]]:
        """
        Find best exchange for order execution considering:
        - Price
        - Liquidity
        - Latency
        - Fees
        """
        try:
            # Get current prices from all exchanges
            tickers = await self.exchange_manager.fetch_ticker_all(symbol)
            
            if not tickers:
                return None
            
            best_venue = None
            best_score = -float('inf')
            
            for exchange_name, ticker in tickers.items():
                # Get price
                if side == OrderSide.BUY:
                    price = ticker.get('ask', 0)
                else:
                    price = ticker.get('bid', 0)
                
                if not price:
                    continue
                
                # Calculate execution score
                score = await self._calculate_execution_score(
                    exchange_name, symbol, side, amount, price, ticker
                )
                
                if score > best_score:
                    best_score = score
                    best_venue = (exchange_name, price)
            
            return best_venue
            
        except Exception as e:
            logger.error(f"Error finding best execution venue: {e}")
            return None
    
    async def _calculate_execution_score(self, exchange_name: str, symbol: str, 
                                       side: OrderSide, amount: float, 
                                       price: float, ticker: Dict) -> float:
        """Calculate execution score for exchange"""
        try:
            score = 0
            
            # Price score (higher is better for selling, lower for buying)
            if side == OrderSide.BUY:
                price_score = 1.0 / price if price > 0 else 0
            else:
                price_score = price
            
            score += price_score * 0.4  # 40% weight
            
            # Liquidity score (volume)
            volume = ticker.get('quoteVolume', 0)
            liquidity_score = min(volume / 1000000, 1.0)  # Normalize to 1M volume
            score += liquidity_score * 0.3  # 30% weight
            
            # Latency score
            latency = await self._get_exchange_latency(exchange_name)
            latency_score = max(0, 1.0 - (latency / 1000))  # Normalize to 1 second
            score += latency_score * 0.2  # 20% weight
            
            # Fee score
            exchange = self.exchange_manager.get_exchange(exchange_name)
            if exchange and exchange.fees:
                # Estimate fees
                taker_fee = exchange.fees.get('trading', {}).get('taker', 0.001)
                fee_score = max(0, 1.0 - (taker_fee * 100))  # Normalize to 1%
                score += fee_score * 0.1  # 10% weight
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculating execution score: {e}")
            return 0
    
    async def _get_exchange_latency(self, exchange_name: str) -> float:
        """Get or measure exchange latency"""
        try:
            # Check cache
            if (exchange_name in self.latency_cache and 
                exchange_name in self.last_latency_update and
                datetime.now() - self.last_latency_update[exchange_name] < timedelta(minutes=5)):
                return self.latency_cache[exchange_name]
            
            # Measure latency
            exchange = self.exchange_manager.get_exchange(exchange_name)
            if not exchange:
                return 1000  # High latency for unavailable exchanges
            
            start_time = time.time()
            try:
                await exchange.fetch_ticker('BTC/USDT')
                latency = (time.time() - start_time) * 1000  # Convert to ms
            except:
                latency = 1000  # High latency on error
            
            # Cache result
            self.latency_cache[exchange_name] = latency
            self.last_latency_update[exchange_name] = datetime.now()
            
            return latency
            
        except Exception as e:
            logger.error(f"Error measuring latency for {exchange_name}: {e}")
            return 1000


class ArbitrageEngine:
    """
    Advanced Arbitrage Engine for cross-exchange trading
    """
    
    def __init__(self, exchange_manager: MultiExchangeManager, config: Dict[str, Any]):
        self.exchange_manager = exchange_manager
        self.config = config
        
        # Configuration
        self.min_profit_percent = config.get('min_profit_percent', 0.5)  # 0.5%
        self.min_profit_usd = config.get('min_profit_usd', 10)  # $10
        self.max_position_size = config.get('max_position_size', 1000)  # $1000
        self.opportunity_timeout_seconds = config.get('opportunity_timeout_seconds', 30)
        self.check_interval_seconds = config.get('check_interval_seconds', 5)
        
        # Symbols to monitor
        self.monitored_symbols = config.get('symbols', ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'])
        
        # Smart order router
        self.router = SmartOrderRouter(exchange_manager)
        
        # State
        self.opportunities = {}  # Active opportunities
        self.execution_history = []
        self.stats = ArbitrageStats(0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.running = False
        
        # Monitoring task
        self.monitor_task = None
        
        logger.info("ArbitrageEngine initialized")
    
    async def start_monitoring(self):
        """Start monitoring for arbitrage opportunities"""
        if self.running:
            logger.warning("Arbitrage monitoring already running")
            return
        
        self.running = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started arbitrage monitoring")
        
        if NOTIFIER_AVAILABLE:
            send_info("🔍 Arbitrage monitoring started - scanning for opportunities")
    
    async def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped arbitrage monitoring")
        
        if NOTIFIER_AVAILABLE:
            send_info("⏹️ Arbitrage monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Scan for opportunities
                await self._scan_opportunities()
                
                # Clean up expired opportunities
                self._cleanup_opportunities()
                
                # Execute viable opportunities
                await self._execute_opportunities()
                
                # Wait before next scan
                await asyncio.sleep(self.check_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in arbitrage monitoring loop: {e}")
                await asyncio.sleep(self.check_interval_seconds)
    
    async def _scan_opportunities(self):
        """Scan for arbitrage opportunities"""
        for symbol in self.monitored_symbols:
            try:
                # Get prices from all exchanges
                tickers = await self.exchange_manager.fetch_ticker_all(symbol)
                
                if len(tickers) < 2:
                    continue  # Need at least 2 exchanges
                
                # Find simple arbitrage opportunities
                opportunities = self._find_simple_arbitrage(symbol, tickers)
                
                # Add new opportunities
                for opp in opportunities:
                    if opp.id not in self.opportunities:
                        self.opportunities[opp.id] = opp
                        logger.info(f"New arbitrage opportunity: {opp.symbol} "
                                  f"{opp.buy_exchange}->{opp.sell_exchange} "
                                  f"{opp.profit_percent:.2f}%")
                        
                        if NOTIFIER_AVAILABLE:
                            send_info(f"💰 Arbitrage opportunity detected!\n"
                                    f"Symbol: {opp.symbol}\n"
                                    f"Buy: {opp.buy_exchange} @ ${opp.buy_price:.4f}\n"
                                    f"Sell: {opp.sell_exchange} @ ${opp.sell_price:.4f}\n"
                                    f"Profit: {opp.profit_percent:.2f}%")
                
            except Exception as e:
                logger.error(f"Error scanning opportunities for {symbol}: {e}")
    
    def _find_simple_arbitrage(self, symbol: str, tickers: Dict[str, Dict]) -> List[ArbitrageOpportunity]:
        """Find simple arbitrage opportunities (buy low, sell high)"""
        opportunities = []
        
        # Get all exchange prices
        prices = {}
        for exchange, ticker in tickers.items():
            bid = ticker.get('bid')
            ask = ticker.get('ask')
            if bid and ask:
                prices[exchange] = {'bid': bid, 'ask': ask, 'volume': ticker.get('quoteVolume', 0)}
        
        if len(prices) < 2:
            return opportunities
        
        # Find arbitrage opportunities
        for buy_exchange, buy_data in prices.items():
            for sell_exchange, sell_data in prices.items():
                if buy_exchange == sell_exchange:
                    continue
                
                buy_price = buy_data['ask']  # We buy at ask
                sell_price = sell_data['bid']  # We sell at bid
                
                if sell_price <= buy_price:
                    continue
                
                # Calculate spread and profit
                spread_percent = ((sell_price - buy_price) / buy_price) * 100
                
                # Estimate fees (simplified)
                buy_fee = 0.1  # 0.1% typical
                sell_fee = 0.1  # 0.1% typical
                profit_percent = spread_percent - buy_fee - sell_fee
                
                if profit_percent < self.min_profit_percent:
                    continue
                
                # Calculate max volume
                max_volume = min(
                    buy_data['volume'] * 0.1,  # 10% of volume
                    sell_data['volume'] * 0.1,
                    self.max_position_size / buy_price
                )
                
                min_profit_usd = max_volume * buy_price * (profit_percent / 100)
                
                if min_profit_usd < self.min_profit_usd:
                    continue
                
                # Create opportunity
                opp_id = f"{symbol}_{buy_exchange}_{sell_exchange}_{int(time.time())}"
                
                opportunity = ArbitrageOpportunity(
                    id=opp_id,
                    type=ArbitrageType.SIMPLE,
                    symbol=symbol,
                    buy_exchange=buy_exchange,
                    sell_exchange=sell_exchange,
                    buy_price=buy_price,
                    sell_price=sell_price,
                    spread_percent=spread_percent,
                    profit_percent=profit_percent,
                    max_volume=max_volume,
                    min_profit_usd=min_profit_usd,
                    timestamp=datetime.now(),
                    expiry=datetime.now() + timedelta(seconds=self.opportunity_timeout_seconds)
                )
                
                opportunities.append(opportunity)
        
        return opportunities
    
    def _cleanup_opportunities(self):
        """Remove expired opportunities"""
        expired_ids = [
            opp_id for opp_id, opp in self.opportunities.items()
            if opp.is_expired or opp.status in [ArbitrageStatus.COMPLETED, ArbitrageStatus.FAILED]
        ]
        
        for opp_id in expired_ids:
            del self.opportunities[opp_id]
    
    async def _execute_opportunities(self):
        """Execute profitable opportunities"""
        # Sort by profit potential
        viable_opportunities = [
            opp for opp in self.opportunities.values()
            if opp.status == ArbitrageStatus.DETECTED and not opp.is_expired
        ]
        
        viable_opportunities.sort(key=lambda x: x.profit_percent, reverse=True)
        
        # Execute top opportunities (limit concurrent executions)
        max_concurrent = self.config.get('max_concurrent_executions', 2)
        executing_count = sum(1 for opp in self.opportunities.values() 
                            if opp.status == ArbitrageStatus.EXECUTING)
        
        for opp in viable_opportunities[:max_concurrent - executing_count]:
            if executing_count >= max_concurrent:
                break
            
            # Start execution
            opp.status = ArbitrageStatus.EXECUTING
            asyncio.create_task(self._execute_arbitrage(opp))
            executing_count += 1
    
    async def _execute_arbitrage(self, opportunity: ArbitrageOpportunity):
        """Execute single arbitrage opportunity"""
        try:
            logger.info(f"Executing arbitrage: {opportunity.id}")
            
            start_time = time.time()
            
            # Determine execution size
            execution_size = min(
                opportunity.max_volume,
                self.max_position_size / opportunity.buy_price
            )
            
            # Execute both orders simultaneously
            buy_task = self._create_order_safe(
                opportunity.buy_exchange, 
                opportunity.symbol, 
                OrderSide.BUY, 
                OrderType.MARKET,
                execution_size
            )
            
            sell_task = self._create_order_safe(
                opportunity.sell_exchange,
                opportunity.symbol,
                OrderSide.SELL,
                OrderType.MARKET,
                execution_size
            )
            
            # Wait for both orders
            buy_order, sell_order = await asyncio.gather(buy_task, sell_task, return_exceptions=True)
            
            execution_time = (time.time() - start_time) * 1000  # ms
            
            # Check results
            buy_success = not isinstance(buy_order, Exception) and buy_order is not None
            sell_success = not isinstance(sell_order, Exception) and sell_order is not None
            
            if buy_success and sell_success:
                # Calculate actual profit
                actual_buy_price = buy_order.price or opportunity.buy_price
                actual_sell_price = sell_order.price or opportunity.sell_price
                
                executed_volume = min(buy_order.filled, sell_order.filled)
                gross_profit = (actual_sell_price - actual_buy_price) * executed_volume
                
                # Estimate fees
                fees = {
                    opportunity.buy_exchange: executed_volume * actual_buy_price * 0.001,
                    opportunity.sell_exchange: executed_volume * actual_sell_price * 0.001
                }
                total_fees = sum(fees.values())
                
                actual_profit = gross_profit - total_fees
                slippage = abs(actual_profit - opportunity.min_profit_usd) / opportunity.min_profit_usd * 100
                
                # Create execution record
                execution = ArbitrageExecution(
                    opportunity_id=opportunity.id,
                    buy_order=buy_order,
                    sell_order=sell_order,
                    executed_volume=executed_volume,
                    actual_profit_usd=actual_profit,
                    execution_time_ms=execution_time,
                    fees_paid=fees,
                    slippage_percent=slippage,
                    success=True
                )
                
                opportunity.status = ArbitrageStatus.COMPLETED
                
                logger.info(f"Arbitrage executed successfully: {opportunity.id}, "
                          f"Profit: ${actual_profit:.2f}")
                
                if NOTIFIER_AVAILABLE:
                    send_info(f"✅ Arbitrage executed successfully!\n"
                            f"Opportunity: {opportunity.symbol}\n"
                            f"Profit: ${actual_profit:.2f}\n"
                            f"Execution time: {execution_time:.0f}ms")
                
            else:
                # Execution failed
                error_msg = []
                if isinstance(buy_order, Exception):
                    error_msg.append(f"Buy failed: {buy_order}")
                if isinstance(sell_order, Exception):
                    error_msg.append(f"Sell failed: {sell_order}")
                
                execution = ArbitrageExecution(
                    opportunity_id=opportunity.id,
                    buy_order=buy_order if buy_success else None,
                    sell_order=sell_order if sell_success else None,
                    executed_volume=0,
                    actual_profit_usd=0,
                    execution_time_ms=execution_time,
                    fees_paid={},
                    slippage_percent=0,
                    success=False,
                    error_message="; ".join(error_msg)
                )
                
                opportunity.status = ArbitrageStatus.FAILED
                
                logger.error(f"Arbitrage execution failed: {opportunity.id}, "
                           f"Error: {execution.error_message}")
                
                if NOTIFIER_AVAILABLE:
                    send_error(f"❌ Arbitrage execution failed!\n"
                             f"Opportunity: {opportunity.symbol}\n"
                             f"Error: {execution.error_message}")
            
            # Store execution
            self.execution_history.append(execution)
            self._update_stats()
            
        except Exception as e:
            logger.error(f"Error executing arbitrage {opportunity.id}: {e}")
            opportunity.status = ArbitrageStatus.FAILED
    
    async def _create_order_safe(self, exchange_name: str, symbol: str, side: OrderSide,
                               order_type: OrderType, amount: float) -> Optional[UnifiedOrder]:
        """Safely create order with error handling"""
        try:
            return await self.exchange_manager.create_order_on_exchange(
                exchange_name, symbol, side, order_type, amount
            )
        except Exception as e:
            logger.error(f"Error creating order on {exchange_name}: {e}")
            raise e
    
    def _update_stats(self):
        """Update arbitrage statistics"""
        if not self.execution_history:
            return
        
        successful_trades = [ex for ex in self.execution_history if ex.success]
        
        self.stats = ArbitrageStats(
            total_opportunities=len(self.execution_history),
            executed_trades=len(successful_trades),
            success_rate=len(successful_trades) / len(self.execution_history) * 100,
            total_profit_usd=sum(ex.actual_profit_usd for ex in successful_trades),
            average_profit_per_trade=np.mean([ex.actual_profit_usd for ex in successful_trades]) if successful_trades else 0,
            average_execution_time_ms=np.mean([ex.execution_time_ms for ex in self.execution_history]),
            best_profit_trade=max([ex.actual_profit_usd for ex in successful_trades]) if successful_trades else 0,
            worst_profit_trade=min([ex.actual_profit_usd for ex in successful_trades]) if successful_trades else 0,
            total_fees_paid=sum(sum(ex.fees_paid.values()) for ex in successful_trades)
        )
    
    def get_active_opportunities(self) -> List[ArbitrageOpportunity]:
        """Get current active opportunities"""
        return [opp for opp in self.opportunities.values() 
                if not opp.is_expired and opp.status == ArbitrageStatus.DETECTED]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get arbitrage statistics"""
        return {
            'stats': {
                'total_opportunities': self.stats.total_opportunities,
                'executed_trades': self.stats.executed_trades,
                'success_rate': self.stats.success_rate,
                'total_profit_usd': self.stats.total_profit_usd,
                'average_profit_per_trade': self.stats.average_profit_per_trade,
                'average_execution_time_ms': self.stats.average_execution_time_ms,
                'best_profit_trade': self.stats.best_profit_trade,
                'worst_profit_trade': self.stats.worst_profit_trade,
                'total_fees_paid': self.stats.total_fees_paid
            },
            'active_opportunities': len(self.get_active_opportunities()),
            'running': self.running,
            'monitored_symbols': self.monitored_symbols
        }
    
    async def force_scan(self) -> List[ArbitrageOpportunity]:
        """Force immediate scan for opportunities"""
        await self._scan_opportunities()
        return self.get_active_opportunities()


# Factory function
def create_arbitrage_engine(exchange_manager: MultiExchangeManager, 
                           config: Dict[str, Any]) -> ArbitrageEngine:
    """Create and return ArbitrageEngine instance"""
    return ArbitrageEngine(exchange_manager, config)