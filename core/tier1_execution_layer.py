"""
Tier-1 Execution Layer
Elite institutioneller Execution Layer mit Micro-Difficulty Scoring und optimaler Ausführung
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import asyncio
import logging
import uuid
from dataclasses import dataclass
from collections import deque

from .tier1_models import (
    Signal, Allocation, Order, ExecutionMetrics, IExecutionLayer,
    OrderStatus, SignalDirection, SystemConstants
)

logger = logging.getLogger(__name__)


@dataclass
class MarketMicrostructure:
    """Market Microstructure Data"""
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    last_price: float
    volume_1min: float
    timestamp: datetime
    
    @property
    def spread(self) -> float:
        return self.ask_price - self.bid_price
    
    @property
    def spread_bps(self) -> float:
        return (self.spread / self.last_price) * 10000 if self.last_price > 0 else 0
    
    @property
    def mid_price(self) -> float:
        return (self.bid_price + self.ask_price) / 2


@dataclass
class VenueData:
    """Exchange/Venue specific data"""
    venue_name: str
    maker_fee: float
    taker_fee: float
    min_order_size: float
    max_order_size: float
    tick_size: float
    latency_ms: float
    liquidity_score: float  # 0-1


@dataclass
class ExecutionAlgorithm:
    """Execution Algorithm Configuration"""
    name: str
    urgency_range: Tuple[float, float]  # (min_urgency, max_urgency)
    min_size: float
    max_size: float
    description: str


class AdvancedExecutionLayer(IExecutionLayer):
    """
    Elite Execution Layer mit institutionellen Features:
    - Micro-Difficulty Scoring (0.0-1.0)
    - Smart Order Routing
    - Execution Algorithm Selection
    - Real-time Market Microstructure Analysis
    - Slippage Prediction & Optimization
    - Dark Pool Integration (simuliert)
    """
    
    def __init__(self):
        
        # Market Data für Micro-structure Analysis
        self.market_data: Dict[str, MarketMicrostructure] = {}
        self.market_history: Dict[str, deque] = {}
        
        # Venue Configuration
        self.venues = self._initialize_venues()
        
        # Execution Algorithms
        self.execution_algorithms = self._initialize_execution_algorithms()
        
        # Order Tracking
        self.active_orders: Dict[str, Order] = {}
        self.execution_history: List[ExecutionMetrics] = []
        
        # Performance Tracking
        self.execution_performance: Dict[str, Dict] = {
            'total_orders': 0,
            'successful_fills': 0,
            'avg_slippage': 0.0,
            'avg_execution_score': 0.0,
            'avg_latency': 0.0
        }
        
        # Risk Limits für Execution
        self.max_order_size_usd = 100000  # $100k max order
        self.max_slippage_tolerance = 0.005  # 50bps
        self.max_market_impact = 0.003  # 30bps
        
        logger.info("AdvancedExecutionLayer initialisiert")
    
    def _initialize_venues(self) -> Dict[str, VenueData]:
        """Initialisiert Venue-spezifische Daten"""
        
        return {
            'binance': VenueData(
                venue_name='binance',
                maker_fee=0.001,  # 0.1%
                taker_fee=0.001,  # 0.1%
                min_order_size=10.0,
                max_order_size=1000000.0,
                tick_size=0.01,
                latency_ms=50,
                liquidity_score=0.95
            ),
            'coinbase': VenueData(
                venue_name='coinbase',
                maker_fee=0.005,  # 0.5%
                taker_fee=0.005,  # 0.5%
                min_order_size=5.0,
                max_order_size=500000.0,
                tick_size=0.01,
                latency_ms=80,
                liquidity_score=0.85
            ),
            'kraken': VenueData(
                venue_name='kraken',
                maker_fee=0.0016,  # 0.16%
                taker_fee=0.0026,  # 0.26%
                min_order_size=10.0,
                max_order_size=200000.0,
                tick_size=0.1,
                latency_ms=100,
                liquidity_score=0.75
            ),
            'dark_pool': VenueData(
                venue_name='dark_pool',
                maker_fee=0.0005,  # 0.05%
                taker_fee=0.0005,  # 0.05%
                min_order_size=50000.0,  # Nur große Orders
                max_order_size=10000000.0,
                tick_size=0.01,
                latency_ms=200,
                liquidity_score=0.6
            )
        }
    
    def _initialize_execution_algorithms(self) -> Dict[str, ExecutionAlgorithm]:
        """Initialisiert Execution Algorithms"""
        
        return {
            'market': ExecutionAlgorithm(
                name='market',
                urgency_range=(0.8, 1.0),
                min_size=10,
                max_size=10000,
                description='Immediate market order execution'
            ),
            'limit': ExecutionAlgorithm(
                name='limit',
                urgency_range=(0.3, 0.7),
                min_size=10,
                max_size=100000,
                description='Passive limit order at best price'
            ),
            'twap': ExecutionAlgorithm(
                name='twap',
                urgency_range=(0.1, 0.5),
                min_size=1000,
                max_size=1000000,
                description='Time-weighted average price execution'
            ),
            'vwap': ExecutionAlgorithm(
                name='vwap',
                urgency_range=(0.2, 0.6),
                min_size=5000,
                max_size=500000,
                description='Volume-weighted average price execution'
            ),
            'iceberg': ExecutionAlgorithm(
                name='iceberg',
                urgency_range=(0.1, 0.4),
                min_size=10000,
                max_size=10000000,
                description='Hidden size iceberg orders'
            ),
            'smart_routing': ExecutionAlgorithm(
                name='smart_routing',
                urgency_range=(0.4, 0.8),
                min_size=1000,
                max_size=100000,
                description='Multi-venue smart order routing'
            )
        }
    
    async def score(self, signal: Signal, allocation: Allocation) -> float:
        """
        Hauptmethode: Berechnet Micro-Difficulty Score (0.0-1.0)
        Höherer Score = bessere Execution-Bedingungen
        """
        try:
            # 1. Market Microstructure Score
            microstructure_score = await self._calculate_microstructure_score(signal.asset)
            
            # 2. Liquidity Score
            liquidity_score = await self._calculate_liquidity_score(signal.asset, allocation.amount)
            
            # 3. Timing Score
            timing_score = await self._calculate_timing_score(signal)
            
            # 4. Size Impact Score
            size_impact_score = await self._calculate_size_impact_score(allocation.amount, signal.asset)
            
            # 5. Venue Selection Score
            venue_score = await self._calculate_venue_score(signal.asset, allocation.amount)
            
            # 6. Volatility Environment Score
            volatility_score = await self._calculate_volatility_score(signal.asset)
            
            # Gewichtete Kombination
            weights = {
                'microstructure': 0.25,
                'liquidity': 0.20,
                'timing': 0.15,
                'size_impact': 0.20,
                'venue': 0.10,
                'volatility': 0.10
            }
            
            execution_score = (
                weights['microstructure'] * microstructure_score +
                weights['liquidity'] * liquidity_score +
                weights['timing'] * timing_score +
                weights['size_impact'] * size_impact_score +
                weights['venue'] * venue_score +
                weights['volatility'] * volatility_score
            )
            
            # Auf 0.0-1.0 normalisieren
            execution_score = np.clip(execution_score, 0.0, 1.0)
            
            logger.debug(f"Execution Score für {signal.signal_id}: {execution_score:.3f} "
                        f"(micro={microstructure_score:.2f}, liquidity={liquidity_score:.2f}, "
                        f"timing={timing_score:.2f}, size={size_impact_score:.2f})")
            
            return execution_score
            
        except Exception as e:
            logger.error(f"Fehler bei Execution Scoring: {e}")
            return 0.5  # Default moderate score
    
    async def _calculate_microstructure_score(self, asset: str) -> float:
        """Market Microstructure Quality Score"""
        
        # Simulierte Market Data (in realer Implementierung von Exchange API)
        market_data = await self._get_market_data(asset)
        
        if not market_data:
            return 0.5  # Neutral wenn keine Daten
        
        # Spread Score (enger Spread = besserer Score)
        spread_bps = market_data.spread_bps
        spread_score = max(0, 1 - (spread_bps / 50))  # 50bps als Referenz
        
        # Order Book Depth Score
        total_depth = market_data.bid_size + market_data.ask_size
        depth_score = min(1.0, total_depth / 100)  # 100 als Referenz
        
        # Balance Score (ausgewogenes Order Book)
        if total_depth > 0:
            balance = min(market_data.bid_size, market_data.ask_size) / total_depth
            balance_score = balance * 2  # 0.5 balance = 1.0 score
        else:
            balance_score = 0.5
        
        microstructure_score = (spread_score * 0.4 + depth_score * 0.4 + balance_score * 0.2)
        
        return np.clip(microstructure_score, 0.0, 1.0)
    
    async def _calculate_liquidity_score(self, asset: str, order_size_usd: float) -> float:
        """Liquidity Adequacy Score für Order Size"""
        
        market_data = await self._get_market_data(asset)
        
        if not market_data:
            return 0.5
        
        # Geschätzte verfügbare Liquidität (vereinfacht)
        estimated_liquidity_usd = (market_data.bid_size + market_data.ask_size) * market_data.last_price
        
        # Size vs Available Liquidity
        if estimated_liquidity_usd > 0:
            liquidity_ratio = order_size_usd / estimated_liquidity_usd
            
            # Score basierend auf Ratio
            if liquidity_ratio < 0.01:  # < 1% of liquidity
                return 1.0
            elif liquidity_ratio < 0.05:  # < 5% of liquidity
                return 0.8
            elif liquidity_ratio < 0.1:   # < 10% of liquidity
                return 0.6
            elif liquidity_ratio < 0.2:   # < 20% of liquidity
                return 0.4
            else:
                return 0.2  # > 20% of liquidity
        
        return 0.5
    
    async def _calculate_timing_score(self, signal: Signal) -> float:
        """Timing Quality Score basierend auf Signal-Eigenschaften"""
        
        # Signal Age (frischere Signale = besserer Score)
        signal_age_minutes = (datetime.now() - signal.timestamp).total_seconds() / 60
        age_score = max(0, 1 - (signal_age_minutes / 5))  # 5 Minuten Halbwertszeit
        
        # Expected Duration vs Market Hours
        # Längere Signale haben mehr Zeit für optimale Execution
        duration_score = min(1.0, signal.expected_duration_min / 60)  # 1 Stunde = 1.0
        
        # Confidence Impact auf Timing
        confidence_score = signal.confidence
        
        timing_score = (age_score * 0.4 + duration_score * 0.3 + confidence_score * 0.3)
        
        return np.clip(timing_score, 0.0, 1.0)
    
    async def _calculate_size_impact_score(self, order_size_usd: float, asset: str) -> float:
        """Market Impact Score basierend auf Order Size"""
        
        # Typische tägliche Volumina pro Asset (vereinfacht)
        typical_daily_volumes = {
            'BTC': 50000000,    # $50M
            'ETH': 30000000,    # $30M
            'USDT': 100000000,  # $100M
            'BNB': 10000000,    # $10M
            'default': 5000000  # $5M
        }
        
        daily_volume = typical_daily_volumes.get(asset.upper(), typical_daily_volumes['default'])
        
        # Order size as percentage of daily volume
        volume_percentage = order_size_usd / daily_volume
        
        # Market Impact Score (weniger Impact = höherer Score)
        if volume_percentage < 0.001:    # < 0.1% of daily volume
            return 1.0
        elif volume_percentage < 0.005:  # < 0.5% of daily volume
            return 0.9
        elif volume_percentage < 0.01:   # < 1% of daily volume
            return 0.8
        elif volume_percentage < 0.02:   # < 2% of daily volume
            return 0.6
        elif volume_percentage < 0.05:   # < 5% of daily volume
            return 0.4
        else:
            return 0.2  # > 5% of daily volume
    
    async def _calculate_venue_score(self, asset: str, order_size_usd: float) -> float:
        """Best Venue Selection Score"""
        
        best_venue = await self._select_best_venue(asset, order_size_usd)
        
        if best_venue:
            # Score basierend auf Venue-Qualität
            return best_venue.liquidity_score
        
        return 0.5
    
    async def _calculate_volatility_score(self, asset: str) -> float:
        """Volatility Environment Score"""
        
        # Simulierte Volatility (in realer Implementierung berechnet)
        estimated_volatility = 0.3  # 30% annualized
        
        # Moderate Volatility ist optimal für Execution
        if 0.15 <= estimated_volatility <= 0.25:  # 15-25% optimal range
            return 1.0
        elif 0.1 <= estimated_volatility <= 0.35:  # 10-35% acceptable range
            return 0.8
        elif estimated_volatility < 0.05:  # Too low volatility
            return 0.6
        else:  # Too high volatility
            return 0.4
    
    async def place(self, signal: Signal, allocation: Allocation) -> Order:
        """
        Hauptmethode: Platziert Order mit optimaler Execution
        """
        try:
            # 1. Execution Algorithm auswählen
            algorithm = await self._select_execution_algorithm(signal, allocation)
            
            # 2. Best Venue auswählen
            venue = await self._select_best_venue(signal.asset, allocation.amount)
            
            # 3. Order Parameter optimieren
            order_params = await self._optimize_order_parameters(signal, allocation, algorithm, venue)
            
            # 4. Order erstellen
            order = Order(
                order_id=f"ORD_{uuid.uuid4().hex[:8]}",
                signal_id=signal.signal_id,
                asset=signal.asset,
                side='buy' if signal.direction == SignalDirection.LONG else 'sell',
                quantity=allocation.position_size,
                price=order_params.get('price'),
                order_type=order_params.get('order_type', 'market'),
                execution_score=await self.score(signal, allocation),
                slippage_estimate=order_params.get('slippage_estimate', 0.0),
                urgency_factor=order_params.get('urgency_factor', 0.5),
                exchange=venue.venue_name if venue else 'binance',
                venue_specific_data={
                    'algorithm': algorithm.name if algorithm else 'market',
                    'estimated_latency': venue.latency_ms if venue else 100
                }
            )
            
            # 5. Order ausführen (simuliert)
            filled_order = await self._execute_order(order, venue)
            
            # 6. Execution Metrics sammeln
            await self._record_execution_metrics(filled_order, signal, allocation)
            
            # 7. Order tracking
            self.active_orders[order.order_id] = filled_order
            
            logger.info(f"Order platziert: {order.order_id} ({order.side} {order.quantity:.4f} {order.asset})")
            
            return filled_order
            
        except Exception as e:
            logger.error(f"Fehler bei Order Placement: {e}")
            
            # Fallback: Basic Market Order
            fallback_order = Order(
                order_id=f"ORD_FALLBACK_{uuid.uuid4().hex[:6]}",
                signal_id=signal.signal_id,
                asset=signal.asset,
                side='buy' if signal.direction == SignalDirection.LONG else 'sell',
                quantity=allocation.position_size,
                order_type='market',
                status=OrderStatus.REJECTED
            )
            
            return fallback_order
    
    async def _select_execution_algorithm(self, signal: Signal, allocation: Allocation) -> Optional[ExecutionAlgorithm]:
        """Selects best execution algorithm based on signal and allocation"""
        
        # Urgency basierend auf Signal-Eigenschaften
        urgency = self._calculate_urgency(signal, allocation)
        
        # Order Size
        order_size_usd = allocation.amount
        
        # Matching Algorithm
        suitable_algorithms = []
        
        for algo in self.execution_algorithms.values():
            if (algo.urgency_range[0] <= urgency <= algo.urgency_range[1] and
                algo.min_size <= order_size_usd <= algo.max_size):
                suitable_algorithms.append(algo)
        
        if suitable_algorithms:
            # Wähle besten basierend auf Urgency Match
            best_algo = min(suitable_algorithms, 
                          key=lambda a: abs((a.urgency_range[0] + a.urgency_range[1])/2 - urgency))
            return best_algo
        
        # Fallback: Market Order
        return self.execution_algorithms.get('market')
    
    def _calculate_urgency(self, signal: Signal, allocation: Allocation) -> float:
        """Calculates urgency factor (0.0-1.0)"""
        
        # Base urgency from signal confidence
        base_urgency = signal.confidence
        
        # Duration impact (shorter duration = higher urgency)
        duration_urgency = max(0, 1 - (signal.expected_duration_min / 120))  # 2 hours baseline
        
        # Size impact (larger orders need more patience)
        size_urgency = max(0.2, 1 - (allocation.amount / 100000))  # $100k baseline
        
        # Combined urgency
        urgency = (base_urgency * 0.5 + duration_urgency * 0.3 + size_urgency * 0.2)
        
        return np.clip(urgency, 0.0, 1.0)
    
    async def _select_best_venue(self, asset: str, order_size_usd: float) -> Optional[VenueData]:
        """Selects best venue for execution"""
        
        suitable_venues = []
        
        for venue in self.venues.values():
            if venue.min_order_size <= order_size_usd <= venue.max_order_size:
                suitable_venues.append(venue)
        
        if not suitable_venues:
            return self.venues.get('binance')  # Fallback
        
        # Score venues
        venue_scores = {}
        for venue in suitable_venues:
            score = (
                venue.liquidity_score * 0.4 +
                (1 - venue.maker_fee) * 0.3 +  # Lower fees = higher score
                (1 - venue.latency_ms / 200) * 0.3  # Lower latency = higher score
            )
            venue_scores[venue] = score
        
        # Return best venue
        best_venue = max(venue_scores.items(), key=lambda x: x[1])[0]
        return best_venue
    
    async def _optimize_order_parameters(self, signal: Signal, allocation: Allocation, 
                                       algorithm: Optional[ExecutionAlgorithm], 
                                       venue: Optional[VenueData]) -> Dict[str, Any]:
        """Optimizes order parameters"""
        
        market_data = await self._get_market_data(signal.asset)
        
        params = {
            'urgency_factor': self._calculate_urgency(signal, allocation)
        }
        
        if algorithm and market_data:
            if algorithm.name == 'market':
                params.update({
                    'order_type': 'market',
                    'price': None,
                    'slippage_estimate': 0.001  # 10bps for market orders
                })
            
            elif algorithm.name == 'limit':
                # Aggressive limit pricing
                if signal.direction == SignalDirection.LONG:
                    limit_price = market_data.bid_price * 1.0001  # Slightly above bid
                else:
                    limit_price = market_data.ask_price * 0.9999  # Slightly below ask
                
                params.update({
                    'order_type': 'limit',
                    'price': limit_price,
                    'slippage_estimate': 0.0005  # 5bps for limit orders
                })
            
            elif algorithm.name in ['twap', 'vwap']:
                params.update({
                    'order_type': 'algo',
                    'price': market_data.mid_price,
                    'slippage_estimate': 0.0008,  # 8bps for algo orders
                    'execution_duration': min(signal.expected_duration_min, 60)  # Max 1 hour
                })
        
        return params
    
    async def _execute_order(self, order: Order, venue: Optional[VenueData]) -> Order:
        """Simulates order execution"""
        
        # Simulation: Order wird immer gefüllt
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.timestamp_submitted = datetime.now()
        order.timestamp_filled = datetime.now() + timedelta(milliseconds=venue.latency_ms if venue else 100)
        
        # Simulate execution price
        market_data = await self._get_market_data(order.asset)
        
        if market_data:
            if order.order_type == 'market':
                if order.side == 'buy':
                    execution_price = market_data.ask_price * (1 + order.slippage_estimate)
                else:
                    execution_price = market_data.bid_price * (1 - order.slippage_estimate)
            else:
                execution_price = order.price or market_data.mid_price
            
            order.avg_fill_price = execution_price
        
        return order
    
    async def _get_market_data(self, asset: str) -> Optional[MarketMicrostructure]:
        """Gets market microstructure data (simulated)"""
        
        # Simulierte Market Data
        if asset not in self.market_data:
            # Create simulated market data
            base_price = {
                'BTC': 50000,
                'ETH': 3000,
                'USDT': 1.0,
                'BNB': 400
            }.get(asset.upper(), 100)
            
            spread_bps = np.random.normal(10, 3)  # ~10bps average spread
            spread = base_price * (spread_bps / 10000)
            
            self.market_data[asset] = MarketMicrostructure(
                bid_price=base_price - spread/2,
                ask_price=base_price + spread/2,
                bid_size=np.random.uniform(10, 100),
                ask_size=np.random.uniform(10, 100),
                last_price=base_price,
                volume_1min=np.random.uniform(1000, 10000),
                timestamp=datetime.now()
            )
        
        return self.market_data[asset]
    
    async def _record_execution_metrics(self, order: Order, signal: Signal, allocation: Allocation) -> None:
        """Records execution quality metrics"""
        
        market_data = await self._get_market_data(order.asset)
        
        if market_data and order.avg_fill_price:
            # Calculate actual slippage
            if order.side == 'buy':
                expected_price = market_data.ask_price
            else:
                expected_price = market_data.bid_price
            
            actual_slippage = abs(order.avg_fill_price - expected_price) / expected_price
            
            # Calculate latency
            if order.timestamp_submitted and order.timestamp_filled:
                latency = (order.timestamp_filled - order.timestamp_submitted).total_seconds() * 1000
            else:
                latency = 100  # Default
            
            metrics = ExecutionMetrics(
                signal_id=signal.signal_id,
                execution_score=order.execution_score,
                slippage=actual_slippage,
                market_impact=actual_slippage * 0.5,  # Simplified
                fill_rate=order.filled_quantity / order.quantity,
                latency_ms=latency,
                bid_ask_spread=market_data.spread_bps,
                order_book_depth=market_data.bid_size + market_data.ask_size,
                volatility_regime=0.3  # Simplified
            )
            
            self.execution_history.append(metrics)
            
            # Update performance tracking
            self._update_performance_metrics(metrics)
    
    def _update_performance_metrics(self, metrics: ExecutionMetrics) -> None:
        """Updates execution performance tracking"""
        
        perf = self.execution_performance
        
        perf['total_orders'] += 1
        
        if metrics.fill_rate >= 0.95:  # 95% fill rate considered successful
            perf['successful_fills'] += 1
        
        # EWMA update für rolling metrics
        alpha = 0.1
        perf['avg_slippage'] = (1-alpha) * perf['avg_slippage'] + alpha * metrics.slippage
        perf['avg_execution_score'] = (1-alpha) * perf['avg_execution_score'] + alpha * metrics.execution_score
        perf['avg_latency'] = (1-alpha) * perf['avg_latency'] + alpha * metrics.latency_ms
    
    async def get_execution_metrics(self, order_id: str) -> Optional[ExecutionMetrics]:
        """Gets execution metrics for specific order"""
        
        for metrics in self.execution_history:
            order = self.active_orders.get(order_id)
            if order and metrics.signal_id == order.signal_id:
                return metrics
        
        return None
    
    def get_execution_status(self) -> Dict[str, Any]:
        """Returns execution layer status"""
        
        active_order_count = len([o for o in self.active_orders.values() 
                                if o.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]])
        
        return {
            'active_orders': active_order_count,
            'total_orders': self.execution_performance['total_orders'],
            'success_rate': (self.execution_performance['successful_fills'] / 
                           max(1, self.execution_performance['total_orders'])),
            'avg_slippage_bps': self.execution_performance['avg_slippage'] * 10000,
            'avg_execution_score': self.execution_performance['avg_execution_score'],
            'avg_latency_ms': self.execution_performance['avg_latency'],
            'available_venues': len(self.venues),
            'available_algorithms': len(self.execution_algorithms)
        }
    
    def update_market_data(self, asset: str, bid: float, ask: float, volume: float) -> None:
        """Updates market microstructure data"""
        
        self.market_data[asset] = MarketMicrostructure(
            bid_price=bid,
            ask_price=ask,
            bid_size=volume * 0.3,  # Simplified
            ask_size=volume * 0.3,
            last_price=(bid + ask) / 2,
            volume_1min=volume,
            timestamp=datetime.now()
        )
    
    def __repr__(self) -> str:
        return (f"AdvancedExecutionLayer(active_orders={len(self.active_orders)}, "
                f"venues={len(self.venues)}, algorithms={len(self.execution_algorithms)})")