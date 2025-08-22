# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Order Manager - Comprehensive Order Management System
====================================================

Handles all order-related operations:
- Order creation and validation
- Order execution and tracking
- Order history management
- Partial fills handling
- Order types support
"""

import logging
import json
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import threading
import queue
import time
import uuid

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    ICEBERG = "iceberg"


class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"


class OrderManager:
    """Comprehensive order management system"""

    def __init__(self, settings, position_manager=None):
        """Initialize Order Manager"""
        self.settings = settings
        self.position_manager = position_manager  # Store reference to position manager
        self.execution_config = settings.get('execution', {})

        # Configuration
        self.order_timeout = self.execution_config.get('order_timeout', 30)
        self.retry_attempts = self.execution_config.get('retry_attempts', 3)
        self.partial_fill_timeout = self.execution_config.get('partial_fill_timeout', 60)
        self.slippage_protection = self.execution_config.get('slippage_protection', 0.01)

        # Order storage
        self.active_orders = {}  # order_id -> order
        self.order_history = []
        self.pending_orders = queue.Queue()

        # Exchange reference (to be set by trading bot)
        self.exchange = None

        # Order execution thread
        self.executor_thread = None
        self.running = False

        # Performance tracking
        self.execution_stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'partial_fills': 0,
            'avg_execution_time': 0,
            'slippage_events': 0
        }

        # Load order history
        self._load_order_history()

        logger.info("Order Manager initialized")

    def set_exchange(self, exchange):
        """Set exchange reference"""
        self.exchange = exchange
        logger.info(f"Exchange set for Order Manager: {exchange.__class__.__name__}")

    def start(self):
        """Start order execution thread"""
        if not self.running:
            self.running = True
            self.executor_thread = threading.Thread(target=self._order_executor_loop)
            self.executor_thread.daemon = True
            self.executor_thread.start()
            logger.info("Order Manager started")

    def stop(self):
        """Stop order execution thread"""
        self.running = False
        if self.executor_thread:
            self.executor_thread.join(timeout=5)
        logger.info("Order Manager stopped")

    def create_order(self, symbol: str, side: str, order_type: str,
                     amount: float, price: Optional[float] = None,
                     stop_price: Optional[float] = None,
                     params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new order"""
        # Validate order
        is_valid, error_msg = self.validate_order({
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'amount': amount,
            'price': price,
            'stop_price': stop_price
        })

        if not is_valid:
            logger.error(f"Order validation failed: {error_msg}")
            return {'error': error_msg}

        # Create order object
        order = {
            'id': str(uuid.uuid4()),
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'amount': amount,
            'price': price,
            'stop_price': stop_price,
            'status': OrderStatus.PENDING.value,
            'filled': 0,
            'remaining': amount,
            'average_fill_price': 0,
            'fees': 0,
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'params': params or {},
            'fills': [],
            'retry_count': 0
        }

        # Add to pending queue
        self.pending_orders.put(order)
        self.active_orders[order['id']] = order

        logger.info(f"Order created: {order['id']} - {side} {amount} {symbol} @ "
                    f"{'market' if order_type == 'market' else price}")

        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if order_id not in self.active_orders:
            logger.warning(f"Order {order_id} not found")
            return False

        order = self.active_orders[order_id]

        # Check if order can be cancelled
        if order['status'] in [OrderStatus.FILLED.value, OrderStatus.CANCELLED.value]:
            logger.warning(f"Order {order_id} cannot be cancelled (status: {order['status']})")
            return False

        try:
            # Cancel on exchange
            if self.exchange and order.get('exchange_id'):
                self.exchange.cancel_order(order['exchange_id'], order['symbol'])

            # Update order status
            order['status'] = OrderStatus.CANCELLED.value
            order['updated_at'] = datetime.now()

            # Move to history
            self._move_to_history(order)

            logger.info(f"Order cancelled: {order_id}")
            return True

        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order by ID"""
        # Check active orders first
        if order_id in self.active_orders:
            return self.active_orders[order_id].copy()

        # Check history
        for order in self.order_history:
            if order['id'] == order_id:
                return order.copy()

        return None

    def get_active_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all active orders"""
        orders = list(self.active_orders.values())

        if symbol:
            orders = [o for o in orders if o['symbol'] == symbol]

        return [o.copy() for o in orders]

    def get_order_history(self, limit: Optional[int] = None,
                          symbol: Optional[str] = None,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get order history with filters"""
        history = self.order_history.copy()

        # Apply filters
        if symbol:
            history = [o for o in history if o['symbol'] == symbol]

        if start_date:
            history = [o for o in history if o['created_at'] >= start_date]

        if end_date:
            history = [o for o in history if o['created_at'] <= end_date]

        # Sort by creation time (newest first)
        history.sort(key=lambda x: x['created_at'], reverse=True)

        # Apply limit
        if limit:
            history = history[:limit]

        return history

    def validate_order(self, order_params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate order parameters"""
        # Required fields
        required_fields = ['symbol', 'side', 'amount']
        for field in required_fields:
            if field not in order_params or not order_params[field]:
                return False, f"Missing required field: {field}"

        # Validate side
        if order_params['side'] not in ['buy', 'sell']:
            return False, f"Invalid side: {order_params['side']}"

        # Validate amount
        if order_params['amount'] <= 0:
            return False, "Amount must be positive"

        # Validate order type
        valid_types = [t.value for t in OrderType]
        order_type = order_params.get('type', 'market')
        if order_type not in valid_types:
            return False, f"Invalid order type: {order_type}"

        # Validate price for limit orders
        if order_type in ['limit', 'stop_limit'] and not order_params.get('price'):
            return False, "Price required for limit orders"

        # Validate stop price for stop orders
        if order_type in ['stop_loss', 'stop_limit'] and not order_params.get('stop_price'):
            return False, "Stop price required for stop orders"

        # Validate against risk limits
        if self.settings.get('risk_management'):
            max_size = self.settings.get('risk_management.max_position_size', float('inf'))
            if order_params['amount'] * order_params.get('price', 50000) > max_size:
                return False, f"Order size exceeds maximum: ${max_size}"

        return True, ""

    def get_supported_order_types(self) -> List[str]:
        """Get list of supported order types"""
        return [order_type.value for order_type in OrderType]

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get order execution statistics"""
        stats = self.execution_stats.copy()

        # Calculate success rate
        total = stats['total_orders']
        if total > 0:
            stats['success_rate'] = stats['successful_orders'] / total
            stats['failure_rate'] = stats['failed_orders'] / total
            stats['partial_fill_rate'] = stats['partial_fills'] / total
        else:
            stats['success_rate'] = 0
            stats['failure_rate'] = 0
            stats['partial_fill_rate'] = 0

        # Add current state
        stats['active_orders'] = len(self.active_orders)
        stats['pending_orders'] = self.pending_orders.qsize()

        return stats

    def update_trailing_stops(self, current_prices: Dict[str, float]):
        """Update trailing stop orders based on current prices"""
        for order_id, order in list(self.active_orders.items()):
            if order['type'] != OrderType.TRAILING_STOP.value:
                continue

            symbol = order['symbol']
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]
            trailing_distance = order['params'].get('trailing_distance', 0.02)  # 2% default

            if order['side'] == 'sell':
                # For long positions, update stop if price has increased
                new_stop = current_price * (1 - trailing_distance)
                if new_stop > order['stop_price']:
                    order['stop_price'] = new_stop
                    order['updated_at'] = datetime.now()
                    logger.info(f"Updated trailing stop for {order_id}: ${new_stop:.2f}")

            else:  # buy order (covering short)
                # For short positions, update stop if price has decreased
                new_stop = current_price * (1 + trailing_distance)
                if new_stop < order['stop_price']:
                    order['stop_price'] = new_stop
                    order['updated_at'] = datetime.now()
                    logger.info(f"Updated trailing stop for {order_id}: ${new_stop:.2f}")

    # Private methods

    def _order_executor_loop(self):
        """Main order execution loop"""
        logger.info("Order executor loop started")

        while self.running:
            try:
                # Get pending order (with timeout)
                try:
                    order = self.pending_orders.get(timeout=1)
                except queue.Empty:
                    continue

                # Execute order
                self._execute_order(order)

                # Update stats
                self.execution_stats['total_orders'] += 1

            except Exception as e:
                logger.error(f"Error in order executor loop: {e}")
                time.sleep(1)

        logger.info("Order executor loop stopped")

    def _execute_order(self, order: Dict[str, Any]):
        """Execute a single order"""
        start_time = time.time()

        try:
            # Check if exchange is available
            if not self.exchange:
                order['status'] = OrderStatus.REJECTED.value
                order['error'] = "No exchange connection"
                self._move_to_history(order)
                return

            # Update status
            order['status'] = OrderStatus.OPEN.value
            order['updated_at'] = datetime.now()

            # Execute based on order type
            if order['type'] == OrderType.MARKET.value:
                self._execute_market_order(order)

            elif order['type'] == OrderType.LIMIT.value:
                self._execute_limit_order(order)

            elif order['type'] in [OrderType.STOP_LOSS.value, OrderType.TAKE_PROFIT.value]:
                self._execute_stop_order(order)

            elif order['type'] == OrderType.TRAILING_STOP.value:
                self._monitor_trailing_stop(order)

            else:
                logger.warning(f"Unsupported order type: {order['type']}")
                order['status'] = OrderStatus.REJECTED.value
                order['error'] = f"Unsupported order type: {order['type']}"

            # Update execution time
            execution_time = time.time() - start_time
            self._update_avg_execution_time(execution_time)

        except Exception as e:
            logger.error(f"Error executing order {order['id']}: {e}")
            order['status'] = OrderStatus.REJECTED.value
            order['error'] = str(e)
            self.execution_stats['failed_orders'] += 1

        finally:
            # Move completed orders to history
            if order['status'] in [OrderStatus.FILLED.value, OrderStatus.CANCELLED.value,
                                   OrderStatus.REJECTED.value, OrderStatus.EXPIRED.value]:
                self._move_to_history(order)

    def _execute_market_order(self, order: Dict[str, Any]):
        """Execute market order"""
        try:
            # Place order on exchange
            result = self.exchange.create_order(
                symbol=order['symbol'],
                type='market',
                side=order['side'],
                amount=order['amount']
            )

            # Update order with exchange response
            order['exchange_id'] = result.get('id')
            order['status'] = OrderStatus.FILLED.value
            order['filled'] = result.get('filled', order['amount'])
            order['remaining'] = 0
            order['average_fill_price'] = result.get('price', result.get('average'))
            order['fees'] = result.get('fee', {}).get('cost', 0)
            order['updated_at'] = datetime.now()

            # Record fill
            order['fills'].append({
                'price': order['average_fill_price'],
                'amount': order['filled'],
                'timestamp': datetime.now(),
                'fee': order['fees']
            })

            # Check slippage
            if order.get('expected_price'):
                slippage = abs(order['average_fill_price'] - order['expected_price']) / order['expected_price']
                if slippage > self.slippage_protection:
                    self.execution_stats['slippage_events'] += 1
                    logger.warning(f"High slippage detected: {slippage:.2%}")

            self.execution_stats['successful_orders'] += 1
            logger.info(f"Market order filled: {order['id']} - {order['filled']} @ ${order['average_fill_price']:.2f}")

        except Exception as e:
            logger.error(f"Market order execution failed: {e}")
            order['status'] = OrderStatus.REJECTED.value
            order['error'] = str(e)
            self.execution_stats['failed_orders'] += 1

    def _execute_limit_order(self, order: Dict[str, Any]):
        """Execute limit order"""
        try:
            # Place order on exchange
            result = self.exchange.create_order(
                symbol=order['symbol'],
                type='limit',
                side=order['side'],
                amount=order['amount'],
                price=order['price']
            )

            order['exchange_id'] = result.get('id')
            order['updated_at'] = datetime.now()

            # Monitor order status
            self._monitor_limit_order(order)

        except Exception as e:
            logger.error(f"Limit order execution failed: {e}")
            order['status'] = OrderStatus.REJECTED.value
            order['error'] = str(e)
            self.execution_stats['failed_orders'] += 1

    def _execute_stop_order(self, order: Dict[str, Any]):
        """Execute stop order"""
        # Monitor price until stop is triggered
        symbol = order['symbol']
        stop_price = order['stop_price']

        logger.info(f"Monitoring stop order {order['id']} - Stop at ${stop_price:.2f}")

        # This would typically be handled by the exchange
        # For paper trading, we simulate it
        order['status'] = OrderStatus.OPEN.value
        order['params']['monitoring'] = True

    def _monitor_limit_order(self, order: Dict[str, Any]):
        """Monitor limit order until filled or timeout"""
        start_time = time.time()

        while order['status'] == OrderStatus.OPEN.value and self.running:
            try:
                # Check order status on exchange
                if order.get('exchange_id'):
                    exchange_order = self.exchange.fetch_order(
                        order['exchange_id'],
                        order['symbol']
                    )

                    # Update order status
                    order['filled'] = exchange_order.get('filled', 0)
                    order['remaining'] = exchange_order.get('remaining', order['amount'])

                    if exchange_order['status'] == 'closed':
                        order['status'] = OrderStatus.FILLED.value
                        order['average_fill_price'] = exchange_order.get('average', order['price'])
                        order['fees'] = exchange_order.get('fee', {}).get('cost', 0)
                        self.execution_stats['successful_orders'] += 1
                        logger.info(f"Limit order filled: {order['id']}")
                        break

                    elif exchange_order['status'] == 'canceled':
                        order['status'] = OrderStatus.CANCELLED.value
                        break

                    elif order['filled'] > 0:
                        order['status'] = OrderStatus.PARTIALLY_FILLED.value
                        self.execution_stats['partial_fills'] += 1

                # Check timeout
                if time.time() - start_time > self.order_timeout:
                    if order['filled'] > 0:
                        logger.warning(f"Limit order partially filled: {order['id']}")
                    else:
                        self.exchange.cancel_order(order['exchange_id'], order['symbol'])
                        order['status'] = OrderStatus.EXPIRED.value
                        logger.warning(f"Limit order expired: {order['id']}")
                    break

                time.sleep(1)  # Check every second

            except Exception as e:
                logger.error(f"Error monitoring limit order: {e}")
                break

        order['updated_at'] = datetime.now()

    def _monitor_trailing_stop(self, order: Dict[str, Any]):
        """Monitor trailing stop order"""
        # Trailing stops are monitored continuously
        # The actual trailing logic is in update_trailing_stops()
        order['status'] = OrderStatus.OPEN.value
        order['params']['monitoring'] = True
        logger.info(f"Trailing stop order activated: {order['id']}")

    def _move_to_history(self, order: Dict[str, Any]):
        """Move order from active to history"""
        if order['id'] in self.active_orders:
            del self.active_orders[order['id']]

        # Add to history
        self.order_history.append(order)

        # Limit history size
        max_history = 1000
        if len(self.order_history) > max_history:
            self.order_history = self.order_history[-max_history:]

        # Save to file
        self._save_order_history()

    def _update_avg_execution_time(self, execution_time: float):
        """Update average execution time"""
        current_avg = self.execution_stats['avg_execution_time']
        total_orders = self.execution_stats['successful_orders']

        if total_orders == 0:
            self.execution_stats['avg_execution_time'] = execution_time
        else:
            # Calculate new average
            new_avg = (current_avg * (total_orders - 1) + execution_time) / total_orders
            self.execution_stats['avg_execution_time'] = new_avg

    def _load_order_history(self):
        """Load order history from file"""
        history_file = os.path.join('data/reports', 'order_history.json')

        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)

                    # Convert timestamps
                    for order in data:
                        order['created_at'] = datetime.fromisoformat(order['created_at'])
                        order['updated_at'] = datetime.fromisoformat(order['updated_at'])

                        # Convert fills timestamps
                        for fill in order.get('fills', []):
                            if isinstance(fill['timestamp'], str):
                                fill['timestamp'] = datetime.fromisoformat(fill['timestamp'])

                    self.order_history = data
                    logger.info(f"Loaded {len(self.order_history)} orders from history")

            except Exception as e:
                logger.error(f"Error loading order history: {e}")

    def _save_order_history(self):
        """Save order history to file"""
        history_file = os.path.join('data/reports', 'order_history.json')

        try:
            # Prepare data for JSON serialization
            save_data = []
            for order in self.order_history[-100:]:  # Save last 100 orders
                order_copy = order.copy()

                # Convert datetime objects
                order_copy['created_at'] = order_copy['created_at'].isoformat()
                order_copy['updated_at'] = order_copy['updated_at'].isoformat()

                # Convert fills timestamps
                for fill in order_copy.get('fills', []):
                    if isinstance(fill['timestamp'], datetime):
                        fill['timestamp'] = fill['timestamp'].isoformat()

                save_data.append(order_copy)

            # Save to file
            os.makedirs(os.path.dirname(history_file), exist_ok=True)
            with open(history_file, 'w') as f:
                json.dump(save_data, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving order history: {e}")

    def generate_order_report(self) -> str:
        """Generate order execution report"""
        stats = self.get_execution_stats()

        report = f"""
=== ORDER EXECUTION REPORT ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTION STATISTICS:
- Total Orders: {stats['total_orders']}
- Successful: {stats['successful_orders']} ({stats['success_rate']:.1%})
- Failed: {stats['failed_orders']} ({stats['failure_rate']:.1%})
- Partial Fills: {stats['partial_fills']} ({stats['partial_fill_rate']:.1%})

PERFORMANCE METRICS:
- Avg Execution Time: {stats['avg_execution_time']:.2f}s
- Slippage Events: {stats['slippage_events']}

CURRENT STATUS:
- Active Orders: {stats['active_orders']}
- Pending Orders: {stats['pending_orders']}

SUPPORTED ORDER TYPES:
{', '.join(self.get_supported_order_types())}
"""

        # Add active orders summary
        active_orders = self.get_active_orders()
        if active_orders:
            report += "\nACTIVE ORDERS:\n"
            for order in active_orders[:5]:  # Show first 5
                report += f"- {order['id']}: {order['side']} {order['amount']} {order['symbol']} "
                report += f"@ ${order.get('price', 'market')} - Status: {order['status']}\n"

            if len(active_orders) > 5:
                report += f"... and {len(active_orders) - 5} more\n"

        return report
    
    async def simulate_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate order execution for paper trading mode
        
        Args:
            order: Order dictionary with symbol, side, amount, etc.
            
        Returns:
            Simulated order result
        """
        try:
            # For paper trading, we just return a successful simulation
            simulated_result = {
                'id': f"SIM_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{order.get('symbol', 'XXX')}",
                'symbol': order.get('symbol'),
                'side': order.get('side'),
                'amount': order.get('amount'),
                'price': order.get('price'),
                'type': order.get('type', 'market'),
                'status': 'filled',
                'timestamp': datetime.now().isoformat(),
                'filled': order.get('amount', 0),
                'remaining': 0,
                'cost': order.get('amount', 0) * order.get('price', 0),
                'fee': {
                    'currency': 'USDT',
                    'cost': order.get('amount', 0) * order.get('price', 0) * 0.001  # 0.1% fee
                },
                'trades': [],
                'info': {'simulated': True}
            }
            
            logger.info(f"📊 Simulated {order.get('side')} order: {order.get('amount')} {order.get('symbol')} @ ${order.get('price')}")
            return simulated_result
            
        except Exception as e:
            logger.error(f"Error simulating order: {e}")
            return {
                'id': None,
                'status': 'failed',
                'error': str(e),
                'info': {'simulated': True}
            }


