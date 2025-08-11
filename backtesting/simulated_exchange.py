"""
Simulated Exchange - Realistische Order Execution Engine
Kritische Komponente für akkurate Backtesting-Ergebnisse
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import uuid

from .event_models import (
    Event, EventType, MarketEvent, OrderEvent, FillEvent,
    OrderType, OrderSide, OrderStatus
)
from .event_bus import EventBus

logger = logging.getLogger(__name__)


@dataclass
class OrderBookLevel:
    """Repräsentiert ein Level im Orderbook"""
    price: float
    size: float
    orders: int = 1
    
    def update_size(self, new_size: float):
        self.size = new_size


@dataclass
class OrderBook:
    """Simuliertes Limit Order Book"""
    symbol: str
    timestamp: datetime
    bids: Dict[float, OrderBookLevel] = field(default_factory=dict)  # price -> level
    asks: Dict[float, OrderBookLevel] = field(default_factory=dict)
    last_price: float = 0.0
    last_size: float = 0.0
    total_volume: float = 0.0
    
    @property
    def best_bid(self) -> Optional[float]:
        return max(self.bids.keys()) if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        return min(self.asks.keys()) if self.asks else None
    
    @property
    def mid_price(self) -> float:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return self.last_price
    
    @property
    def spread(self) -> float:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return 0.0
    
    @property
    def spread_bps(self) -> float:
        if self.mid_price > 0:
            return (self.spread / self.mid_price) * 10000
        return 0.0


@dataclass
class PendingOrder:
    """Interne Repräsentation einer pending Order"""
    order_event: OrderEvent
    remaining_quantity: float
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    fills: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def is_active(self) -> bool:
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]


@dataclass
class MarketMicrostructure:
    """Market Microstructure Modell für realistische Simulation"""
    base_spread_bps: float = 10.0  # Base spread in basis points
    spread_volatility: float = 0.3  # Volatility of spread
    
    # Liquidity parameters
    base_depth_multiplier: float = 100.0  # Size at best bid/ask as multiple of avg trade
    depth_decay_rate: float = 0.5  # How fast liquidity decreases away from best
    
    # Market impact parameters
    temporary_impact_factor: float = 0.1  # Temporary price impact
    permanent_impact_factor: float = 0.05  # Permanent price impact
    impact_decay_time_seconds: float = 60.0  # How fast temporary impact decays
    
    # Latency simulation
    base_latency_ms: float = 50.0
    latency_std_ms: float = 20.0
    congestion_multiplier: float = 2.0  # During high activity


class SimulatedExchange:
    """
    Simulierte Börse mit realistischer Order Execution
    
    Features:
    - Limit Order Book Simulation
    - Realistische Slippage Modellierung
    - Market Impact (temporary & permanent)
    - Latency Simulation
    - Partial Fills
    - Maker/Taker Fee Struktur
    """
    
    def __init__(self,
                 event_bus: EventBus,
                 exchange_name: str = "simulated_binance",
                 maker_fee: float = 0.001,  # 0.1%
                 taker_fee: float = 0.001,  # 0.1%
                 enable_partial_fills: bool = True,
                 enable_market_impact: bool = True,
                 enable_latency: bool = True):
        
        self.event_bus = event_bus
        self.exchange_name = exchange_name
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.enable_partial_fills = enable_partial_fills
        self.enable_market_impact = enable_market_impact
        self.enable_latency = enable_latency
        
        # Order Books für jedes Symbol
        self.order_books: Dict[str, OrderBook] = {}
        
        # Pending Orders
        self.pending_orders: Dict[str, PendingOrder] = {}
        self.limit_order_queues: Dict[str, Dict[float, deque]] = defaultdict(lambda: defaultdict(deque))
        
        # Market Microstructure Models
        self.microstructure_models: Dict[str, MarketMicrostructure] = {}
        
        # Execution Statistics
        self.execution_stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'rejected_orders': 0,
            'partial_fills': 0,
            'total_slippage_bps': 0.0,
            'total_commission': 0.0
        }
        
        # Market Impact Tracking
        self.market_impacts: Dict[str, List[Dict]] = defaultdict(list)
        
        # Subscribe to events
        self._subscribe_to_events()
        
        logger.info(f"SimulatedExchange '{exchange_name}' initialisiert")
    
    def _subscribe_to_events(self):
        """Registriert Event Handler"""
        self.event_bus.subscribe(EventType.MARKET, self._handle_market_event)
        self.event_bus.subscribe(EventType.ORDER, self._handle_order_event)
    
    async def _handle_market_event(self, event: MarketEvent) -> None:
        """
        Verarbeitet Market Events und aktualisiert Order Books
        Prüft auch pending Limit Orders für mögliche Fills
        """
        try:
            # Update order book
            await self._update_order_book(event)
            
            # Check pending limit orders for this symbol
            await self._check_pending_orders(event.symbol, event)
            
            # Decay market impacts
            await self._decay_market_impacts(event.symbol, event.timestamp)
            
        except Exception as e:
            logger.error(f"Error handling market event: {e}")
    
    async def _handle_order_event(self, event: OrderEvent) -> None:
        """Verarbeitet neue Orders"""
        try:
            self.execution_stats['total_orders'] += 1
            
            # Simulate latency if enabled
            if self.enable_latency:
                await self._simulate_latency(event.symbol)
            
            # Validate order
            if not self._validate_order(event):
                await self._reject_order(event, "Order validation failed")
                return
            
            # Process based on order type
            if event.order_type == OrderType.MARKET:
                await self._process_market_order(event)
            elif event.order_type == OrderType.LIMIT:
                await self._process_limit_order(event)
            elif event.order_type == OrderType.STOP:
                await self._process_stop_order(event)
            else:
                await self._reject_order(event, f"Unsupported order type: {event.order_type}")
                
        except Exception as e:
            logger.error(f"Error handling order event: {e}")
            await self._reject_order(event, f"Execution error: {str(e)}")
    
    async def _update_order_book(self, market_event: MarketEvent) -> None:
        """Aktualisiert Order Book basierend auf Market Event"""
        
        symbol = market_event.symbol
        
        # Get or create order book
        if symbol not in self.order_books:
            self.order_books[symbol] = OrderBook(
                symbol=symbol,
                timestamp=market_event.timestamp,
                last_price=market_event.close
            )
        
        book = self.order_books[symbol]
        book.timestamp = market_event.timestamp
        book.last_price = market_event.close
        book.total_volume += market_event.volume
        
        # Update bid/ask if available
        if market_event.bid_price and market_event.ask_price:
            # Clear and rebuild top of book
            book.bids.clear()
            book.asks.clear()
            
            book.bids[market_event.bid_price] = OrderBookLevel(
                price=market_event.bid_price,
                size=market_event.bid_size or market_event.volume * 0.1
            )
            
            book.asks[market_event.ask_price] = OrderBookLevel(
                price=market_event.ask_price,
                size=market_event.ask_size or market_event.volume * 0.1
            )
        else:
            # Simulate bid/ask from OHLC
            spread_bps = self._get_current_spread_bps(symbol)
            half_spread = market_event.close * (spread_bps / 20000)
            
            book.bids.clear()
            book.asks.clear()
            
            # Simulate multiple levels
            for i in range(5):
                level_adjustment = i * market_event.close * 0.0001  # 1bp per level
                
                bid_price = market_event.close - half_spread - level_adjustment
                ask_price = market_event.close + half_spread + level_adjustment
                
                # Size decreases with distance from best
                size_multiplier = 1.0 / (i + 1)
                base_size = market_event.volume * 0.05 * size_multiplier
                
                book.bids[bid_price] = OrderBookLevel(bid_price, base_size)
                book.asks[ask_price] = OrderBookLevel(ask_price, base_size)
    
    def _get_current_spread_bps(self, symbol: str) -> float:
        """Berechnet aktuellen Spread in Basis Points"""
        
        if symbol not in self.microstructure_models:
            self.microstructure_models[symbol] = MarketMicrostructure()
        
        model = self.microstructure_models[symbol]
        
        # Dynamic spread based on volatility and time
        base_spread = model.base_spread_bps
        volatility_adjustment = np.random.normal(0, model.spread_volatility)
        
        current_spread = max(1.0, base_spread + volatility_adjustment)
        
        return current_spread
    
    async def _process_market_order(self, order_event: OrderEvent) -> None:
        """
        Verarbeitet Market Order mit realistischer Slippage
        """
        
        symbol = order_event.symbol
        book = self.order_books.get(symbol)
        
        if not book:
            await self._reject_order(order_event, "No market data available")
            return
        
        # Get relevant order book side
        if order_event.side == OrderSide.BUY:
            available_liquidity = book.asks
            is_taker = True
        else:
            available_liquidity = book.bids
            is_taker = True
        
        if not available_liquidity:
            await self._reject_order(order_event, "No liquidity available")
            return
        
        # Execute order by walking through order book
        remaining_quantity = order_event.quantity
        fills = []
        total_cost = 0.0
        
        # Sort price levels (ascending for buys, descending for sells)
        sorted_levels = sorted(available_liquidity.items(), 
                             key=lambda x: x[0],
                             reverse=(order_event.side == OrderSide.SELL))
        
        for price, level in sorted_levels:
            if remaining_quantity <= 0:
                break
            
            # Calculate fill at this level
            fill_quantity = min(remaining_quantity, level.size)
            
            # Apply market impact if enabled
            if self.enable_market_impact:
                impact_adjusted_price = await self._apply_market_impact(
                    symbol, price, fill_quantity, order_event.side
                )
            else:
                impact_adjusted_price = price
            
            fills.append({
                'price': impact_adjusted_price,
                'quantity': fill_quantity,
                'timestamp': book.timestamp
            })
            
            total_cost += impact_adjusted_price * fill_quantity
            remaining_quantity -= fill_quantity
            
            # Update level
            level.size -= fill_quantity
            if level.size <= 0:
                del available_liquidity[price]
        
        # Check if order fully filled
        if remaining_quantity > 0 and not self.enable_partial_fills:
            await self._reject_order(order_event, "Insufficient liquidity")
            return
        
        # Calculate execution details
        filled_quantity = order_event.quantity - remaining_quantity
        avg_fill_price = total_cost / filled_quantity if filled_quantity > 0 else 0
        
        # Calculate slippage
        reference_price = book.mid_price
        if order_event.side == OrderSide.BUY:
            slippage_bps = ((avg_fill_price - reference_price) / reference_price) * 10000
        else:
            slippage_bps = ((reference_price - avg_fill_price) / reference_price) * 10000
        
        # Calculate commission
        commission = filled_quantity * avg_fill_price * self.taker_fee
        
        # Create fill event
        fill_event = FillEvent(
            timestamp=book.timestamp,
            fill_id=f"FILL_{uuid.uuid4().hex[:8]}",
            order_id=order_event.order_id,
            signal_id=order_event.signal_id,
            symbol=symbol,
            side=order_event.side,
            fill_quantity=filled_quantity,
            fill_price=avg_fill_price,
            commission=commission,
            slippage_bps=slippage_bps,
            slippage_cost=abs(avg_fill_price - reference_price) * filled_quantity,
            market_impact_bps=slippage_bps * 0.5,  # Simplified
            pre_trade_mid_price=reference_price,
            post_trade_mid_price=book.mid_price,
            execution_latency_ms=50.0,  # Placeholder
            passive_fill=not is_taker,
            exchange=self.exchange_name
        )
        
        # Update statistics
        self.execution_stats['filled_orders'] += 1
        self.execution_stats['total_slippage_bps'] += abs(slippage_bps)
        self.execution_stats['total_commission'] += commission
        
        # Publish fill event
        await self.event_bus.publish(fill_event)
        
        logger.info(f"Market order filled: {symbol} {order_event.side.value} "
                   f"{filled_quantity:.4f} @ {avg_fill_price:.2f} "
                   f"(slippage: {slippage_bps:.1f}bps)")
    
    async def _process_limit_order(self, order_event: OrderEvent) -> None:
        """Verarbeitet Limit Order"""
        
        symbol = order_event.symbol
        book = self.order_books.get(symbol)
        
        if not book:
            await self._reject_order(order_event, "No market data available")
            return
        
        # Check if limit order is immediately executable
        is_marketable = False
        
        if order_event.side == OrderSide.BUY and book.best_ask:
            is_marketable = order_event.price >= book.best_ask
        elif order_event.side == OrderSide.SELL and book.best_bid:
            is_marketable = order_event.price <= book.best_bid
        
        if is_marketable and not order_event.post_only:
            # Execute as taker
            await self._execute_marketable_limit_order(order_event, book)
        else:
            # Add to order book as maker
            await self._add_limit_order_to_book(order_event, book)
    
    async def _add_limit_order_to_book(self, order_event: OrderEvent, book: OrderBook) -> None:
        """Fügt Limit Order zum Book hinzu"""
        
        # Create pending order
        pending_order = PendingOrder(
            order_event=order_event,
            remaining_quantity=order_event.quantity,
            status=OrderStatus.SUBMITTED
        )
        
        self.pending_orders[order_event.order_id] = pending_order
        
        # Add to limit order queue
        price_level = order_event.price
        self.limit_order_queues[order_event.symbol][price_level].append(order_event.order_id)
        
        # Update order book display
        if order_event.side == OrderSide.BUY:
            if price_level not in book.bids:
                book.bids[price_level] = OrderBookLevel(price_level, 0)
            book.bids[price_level].size += order_event.quantity
        else:
            if price_level not in book.asks:
                book.asks[price_level] = OrderBookLevel(price_level, 0)
            book.asks[price_level].size += order_event.quantity
        
        logger.debug(f"Limit order added to book: {order_event.symbol} "
                    f"{order_event.side.value} {order_event.quantity} @ {price_level}")
    
    async def _check_pending_orders(self, symbol: str, market_event: MarketEvent) -> None:
        """Prüft pending Orders für mögliche Fills"""
        
        book = self.order_books.get(symbol)
        if not book:
            return
        
        # Check all price levels
        for price_level, order_ids in list(self.limit_order_queues[symbol].items()):
            for order_id in list(order_ids):
                if order_id not in self.pending_orders:
                    order_ids.remove(order_id)
                    continue
                
                pending_order = self.pending_orders[order_id]
                if not pending_order.is_active:
                    continue
                
                # Check if order should be filled
                should_fill = False
                
                if pending_order.order_event.side == OrderSide.BUY:
                    # Buy limit fills when market price <= limit price
                    should_fill = (market_event.low <= price_level)
                else:
                    # Sell limit fills when market price >= limit price
                    should_fill = (market_event.high >= price_level)
                
                if should_fill:
                    await self._fill_limit_order(pending_order, market_event)
    
    async def _fill_limit_order(self, pending_order: PendingOrder, market_event: MarketEvent) -> None:
        """Füllt Limit Order"""
        
        order_event = pending_order.order_event
        
        # Determine fill price (limit price for maker orders)
        fill_price = order_event.price
        fill_quantity = pending_order.remaining_quantity
        
        # Simulate partial fills if enabled
        if self.enable_partial_fills and np.random.random() < 0.3:  # 30% chance
            fill_quantity = pending_order.remaining_quantity * np.random.uniform(0.5, 1.0)
        
        # Update pending order
        pending_order.filled_quantity += fill_quantity
        pending_order.remaining_quantity -= fill_quantity
        
        # Calculate average fill price
        if pending_order.average_fill_price == 0:
            pending_order.average_fill_price = fill_price
        else:
            total_value = (pending_order.average_fill_price * (pending_order.filled_quantity - fill_quantity) +
                          fill_price * fill_quantity)
            pending_order.average_fill_price = total_value / pending_order.filled_quantity
        
        # Update status
        if pending_order.remaining_quantity <= 0:
            pending_order.status = OrderStatus.FILLED
        else:
            pending_order.status = OrderStatus.PARTIALLY_FILLED
            self.execution_stats['partial_fills'] += 1
        
        # Calculate commission (maker fee for limit orders)
        commission = fill_quantity * fill_price * self.maker_fee
        
        # Create fill event
        fill_event = FillEvent(
            timestamp=market_event.timestamp,
            fill_id=f"FILL_{uuid.uuid4().hex[:8]}",
            order_id=order_event.order_id,
            signal_id=order_event.signal_id,
            symbol=order_event.symbol,
            side=order_event.side,
            fill_quantity=fill_quantity,
            fill_price=fill_price,
            commission=commission,
            slippage_bps=0.0,  # No slippage for limit orders
            slippage_cost=0.0,
            market_impact_bps=0.0,
            pre_trade_mid_price=fill_price,
            post_trade_mid_price=fill_price,
            execution_latency_ms=0.0,  # Already accounted for
            passive_fill=True,  # Maker
            exchange=self.exchange_name
        )
        
        # Update statistics
        self.execution_stats['filled_orders'] += 1
        self.execution_stats['total_commission'] += commission
        
        # Remove from book if fully filled
        if pending_order.status == OrderStatus.FILLED:
            self._remove_from_order_book(order_event)
            del self.pending_orders[order_event.order_id]
        
        # Publish fill event
        await self.event_bus.publish(fill_event)
        
        logger.info(f"Limit order filled: {order_event.symbol} {order_event.side.value} "
                   f"{fill_quantity:.4f} @ {fill_price:.2f} (maker)")
    
    async def _apply_market_impact(self, symbol: str, base_price: float, 
                                  quantity: float, side: OrderSide) -> float:
        """Berechnet und wendet Market Impact an"""
        
        if symbol not in self.microstructure_models:
            self.microstructure_models[symbol] = MarketMicrostructure()
        
        model = self.microstructure_models[symbol]
        book = self.order_books.get(symbol)
        
        if not book:
            return base_price
        
        # Calculate volume percentage
        daily_volume = book.total_volume
        volume_pct = quantity / max(daily_volume, 1000) if daily_volume > 0 else 0.001
        
        # Calculate impacts
        temp_impact = volume_pct * model.temporary_impact_factor
        perm_impact = volume_pct * model.permanent_impact_factor
        
        # Apply based on side
        if side == OrderSide.BUY:
            impact_price = base_price * (1 + temp_impact + perm_impact)
        else:
            impact_price = base_price * (1 - temp_impact - perm_impact)
        
        # Track impact
        self.market_impacts[symbol].append({
            'timestamp': datetime.now(),
            'temp_impact': temp_impact,
            'perm_impact': perm_impact,
            'quantity': quantity,
            'side': side
        })
        
        return impact_price
    
    async def _decay_market_impacts(self, symbol: str, current_time: datetime) -> None:
        """Decay temporary market impacts over time"""
        
        if symbol not in self.market_impacts:
            return
        
        model = self.microstructure_models.get(symbol, MarketMicrostructure())
        decay_time = timedelta(seconds=model.impact_decay_time_seconds)
        
        # Remove old impacts
        self.market_impacts[symbol] = [
            impact for impact in self.market_impacts[symbol]
            if current_time - impact['timestamp'] < decay_time
        ]
    
    async def _simulate_latency(self, symbol: str) -> None:
        """Simuliert Netzwerk/Exchange Latenz"""
        
        model = self.microstructure_models.get(symbol, MarketMicrostructure())
        
        # Generate latency
        latency_ms = max(0, np.random.normal(model.base_latency_ms, model.latency_std_ms))
        
        # Add congestion during high activity
        if len(self.pending_orders) > 100:
            latency_ms *= model.congestion_multiplier
        
        # Sleep
        await asyncio.sleep(latency_ms / 1000)
    
    def _validate_order(self, order_event: OrderEvent) -> bool:
        """Validiert Order"""
        
        # Basic validation
        if order_event.quantity <= 0:
            return False
        
        if order_event.order_type == OrderType.LIMIT and not order_event.price:
            return False
        
        if order_event.price and order_event.price <= 0:
            return False
        
        return True
    
    async def _reject_order(self, order_event: OrderEvent, reason: str) -> None:
        """Lehnt Order ab"""
        
        self.execution_stats['rejected_orders'] += 1
        
        # Create rejection fill event (quantity = 0)
        fill_event = FillEvent(
            timestamp=datetime.now(),
            fill_id=f"REJECT_{uuid.uuid4().hex[:8]}",
            order_id=order_event.order_id,
            signal_id=order_event.signal_id,
            symbol=order_event.symbol,
            side=order_event.side,
            fill_quantity=0.0,
            fill_price=0.0,
            commission=0.0,
            slippage_bps=0.0,
            slippage_cost=0.0,
            exchange=self.exchange_name
        )
        
        # Set rejection in metadata
        fill_event.venue_order_id = f"REJECTED: {reason}"
        
        await self.event_bus.publish(fill_event)
        
        logger.warning(f"Order rejected: {order_event.order_id} - {reason}")
    
    def _remove_from_order_book(self, order_event: OrderEvent) -> None:
        """Entfernt Order aus Order Book"""
        
        book = self.order_books.get(order_event.symbol)
        if not book:
            return
        
        price_level = order_event.price
        
        if order_event.side == OrderSide.BUY and price_level in book.bids:
            book.bids[price_level].size -= order_event.quantity
            if book.bids[price_level].size <= 0:
                del book.bids[price_level]
        elif order_event.side == OrderSide.SELL and price_level in book.asks:
            book.asks[price_level].size -= order_event.quantity
            if book.asks[price_level].size <= 0:
                del book.asks[price_level]
    
    async def _execute_marketable_limit_order(self, order_event: OrderEvent, book: OrderBook) -> None:
        """Führt marketable Limit Order aus (crossing the spread)"""
        
        # Convert to market order execution
        await self._process_market_order(order_event)
    
    async def _process_stop_order(self, order_event: OrderEvent) -> None:
        """Verarbeitet Stop Order (simplified)"""
        
        # For now, treat as market order when triggered
        # In full implementation, would monitor price and trigger when stop price hit
        await self._reject_order(order_event, "Stop orders not yet implemented")
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Gibt Execution-Statistiken zurück"""
        
        stats = self.execution_stats.copy()
        
        # Calculate averages
        if stats['filled_orders'] > 0:
            stats['avg_slippage_bps'] = stats['total_slippage_bps'] / stats['filled_orders']
            stats['avg_commission'] = stats['total_commission'] / stats['filled_orders']
        else:
            stats['avg_slippage_bps'] = 0.0
            stats['avg_commission'] = 0.0
        
        # Add order book stats
        stats['tracked_symbols'] = len(self.order_books)
        stats['pending_orders'] = len(self.pending_orders)
        
        return stats
    
    def get_order_book_snapshot(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Gibt Order Book Snapshot zurück"""
        
        book = self.order_books.get(symbol)
        if not book:
            return None
        
        return {
            'symbol': symbol,
            'timestamp': book.timestamp.isoformat(),
            'best_bid': book.best_bid,
            'best_ask': book.best_ask,
            'spread_bps': book.spread_bps,
            'mid_price': book.mid_price,
            'bid_levels': len(book.bids),
            'ask_levels': len(book.asks),
            'total_bid_size': sum(level.size for level in book.bids.values()),
            'total_ask_size': sum(level.size for level in book.asks.values())
        }