"""
Event Models für ereignisgesteuertes Backtesting-Framework
Quant-Research validiertes Design ohne Lookahead-Bias
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Any, List
from enum import Enum
import uuid


class EventType(Enum):
    """Event-Typen im Backtesting-System"""
    MARKET = "market"
    SIGNAL = "signal"
    ORDER = "order"
    FILL = "fill"
    PORTFOLIO_UPDATE = "portfolio_update"
    RISK_UPDATE = "risk_update"
    REBALANCE = "rebalance"
    SYSTEM_STATE = "system_state"


class OrderType(Enum):
    """Order-Typen"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order-Seite"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order-Status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Event(ABC):
    """Basis-Event-Klasse für alle Events im System"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = field(init=False)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert Event zu Dictionary für Logging/Analyse"""
        pass


@dataclass
class MarketEvent(Event):
    """
    Market Data Event - Point-in-Time Marktdaten
    Strikt keine Future-Informationen
    """
    event_type: EventType = field(default=EventType.MARKET, init=False)
    symbol: str = ""
    
    # OHLCV Daten
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    
    # Orderbook Snapshot (optional für detaillierte Simulation)
    bid_price: Optional[float] = None
    bid_size: Optional[float] = None
    ask_price: Optional[float] = None
    ask_size: Optional[float] = None
    
    # Orderbook Depth (Level 2)
    bid_levels: Optional[List[tuple[float, float]]] = None  # [(price, size), ...]
    ask_levels: Optional[List[tuple[float, float]]] = None
    
    # Market Microstructure
    trades_count: Optional[int] = None
    vwap: Optional[float] = None
    spread_bps: Optional[float] = None
    
    # Metadata
    exchange: str = "binance"
    data_quality: float = 1.0  # 0-1, für Simulation von fehlenden Daten
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'ohlcv': {
                'open': self.open,
                'high': self.high,
                'low': self.low,
                'close': self.close,
                'volume': self.volume
            },
            'orderbook': {
                'bid': self.bid_price,
                'ask': self.ask_price,
                'spread_bps': self.spread_bps
            },
            'exchange': self.exchange
        }


@dataclass
class SignalEvent(Event):
    """
    Signal Event vom Alpha-Generator
    Konvertiert von tier1_models.Signal
    """
    event_type: EventType = field(default=EventType.SIGNAL, init=False)
    
    signal_id: str = ""
    stream_id: str = ""
    symbol: str = ""
    direction: str = ""  # 'long', 'short'
    confidence: float = 0.0
    expected_profit_pts: float = 0.0
    expected_duration_min: int = 0
    origin: str = ""  # Strategy name
    
    # Zusätzliche Metadaten
    market_regime: Optional[str] = None
    signal_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    signal_generated_at: Optional[datetime] = None
    signal_valid_until: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'signal_id': self.signal_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'confidence': self.confidence,
            'expected_profit_pts': self.expected_profit_pts,
            'origin': self.origin,
            'metadata': self.signal_metadata
        }


@dataclass
class OrderEvent(Event):
    """
    Order Event - Generiert vom QuantumOrchestrator
    Wird an SimulatedExchange gesendet
    """
    event_type: EventType = field(default=EventType.ORDER, init=False)
    
    order_id: str = ""
    signal_id: str = ""  # Referenz zum ursprünglichen Signal
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    
    # Preise
    price: Optional[float] = None  # Für Limit Orders
    stop_price: Optional[float] = None  # Für Stop Orders
    
    # Execution Instructions
    time_in_force: str = "GTC"  # GTC, IOC, FOK, GTD
    expire_time: Optional[datetime] = None
    reduce_only: bool = False
    post_only: bool = False  # Maker-only order
    
    # Risk/Money Management
    max_slippage_bps: Optional[float] = None  # Max erlaubte Slippage
    urgency_factor: float = 0.5  # 0-1, für Execution-Algorithmus
    
    # Metadata
    exchange: str = "binance"
    execution_algorithm: str = "market"  # market, twap, vwap, etc.
    allocation_amount_usd: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'order_id': self.order_id,
            'signal_id': self.signal_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'urgency': self.urgency_factor,
            'exchange': self.exchange
        }


@dataclass
class FillEvent(Event):
    """
    Fill Event - Generiert von SimulatedExchange
    Repräsentiert tatsächliche Ausführung
    """
    event_type: EventType = field(default=EventType.FILL, init=False)
    
    fill_id: str = ""
    order_id: str = ""
    signal_id: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    
    # Execution Details
    fill_quantity: float = 0.0
    fill_price: float = 0.0
    
    # Kosten
    commission: float = 0.0
    commission_asset: str = "USDT"
    slippage_bps: float = 0.0
    slippage_cost: float = 0.0
    
    # Market Impact
    market_impact_bps: float = 0.0
    pre_trade_mid_price: float = 0.0
    post_trade_mid_price: float = 0.0
    
    # Execution Quality
    execution_latency_ms: float = 0.0
    liquidity_consumed: float = 0.0  # % of available liquidity
    passive_fill: bool = False  # War es ein Maker-Fill?
    
    # Exchange Details
    exchange: str = "binance"
    venue_order_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'fill_id': self.fill_id,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'fill_quantity': self.fill_quantity,
            'fill_price': self.fill_price,
            'commission': self.commission,
            'slippage_bps': self.slippage_bps,
            'execution_quality': {
                'latency_ms': self.execution_latency_ms,
                'market_impact_bps': self.market_impact_bps,
                'passive_fill': self.passive_fill
            }
        }


@dataclass
class PortfolioUpdateEvent(Event):
    """
    Portfolio Update Event - Generiert vom PortfolioManager
    Feedback für Risk Engine und Capital Allocator
    """
    event_type: EventType = field(default=EventType.PORTFOLIO_UPDATE, init=False)
    
    # Portfolio State
    total_equity: float = 0.0
    cash_balance: float = 0.0
    positions_value: float = 0.0
    
    # Positions
    positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Format: {symbol: {'quantity': float, 'avg_price': float, 'current_price': float, 'pnl': float}}
    
    # Risk Metrics
    current_leverage: float = 0.0
    portfolio_var: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    
    # Allocation Status
    allocation_by_strategy: Dict[str, float] = field(default_factory=dict)
    concentration_risk: float = 0.0
    
    # Performance
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'portfolio_state': {
                'total_equity': self.total_equity,
                'cash_balance': self.cash_balance,
                'positions_value': self.positions_value
            },
            'risk_metrics': {
                'leverage': self.current_leverage,
                'var': self.portfolio_var,
                'max_drawdown': self.max_drawdown
            },
            'performance': {
                'daily_pnl': self.daily_pnl,
                'total_pnl': self.total_pnl
            }
        }


@dataclass
class RiskUpdateEvent(Event):
    """
    Risk Update Event - Generiert von Risk Engine
    Informiert über Regime-Änderungen und Risk-Status
    """
    event_type: EventType = field(default=EventType.RISK_UPDATE, init=False)
    
    # Market Regime
    current_regime: str = ""
    regime_confidence: float = 0.0
    regime_changed: bool = False
    
    # Risk Limits
    risk_limits: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Format: {metric: {'current': float, 'limit': float, 'utilization': float}}
    
    # Circuit Breakers
    circuit_breakers: Dict[str, bool] = field(default_factory=dict)
    
    # Risk Signals
    risk_warnings: List[str] = field(default_factory=list)
    risk_score: float = 0.0  # 0-1, aggregiertes Risiko
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'regime': {
                'current': self.current_regime,
                'confidence': self.regime_confidence,
                'changed': self.regime_changed
            },
            'risk_limits': self.risk_limits,
            'circuit_breakers': self.circuit_breakers,
            'risk_score': self.risk_score
        }


@dataclass
class RebalanceEvent(Event):
    """
    Rebalance Event - Triggert Portfolio-Rebalancing
    """
    event_type: EventType = field(default=EventType.REBALANCE, init=False)
    
    rebalance_reason: str = ""  # 'scheduled', 'risk_triggered', 'drift'
    
    # Current vs Target Allocations
    current_allocations: Dict[str, float] = field(default_factory=dict)
    target_allocations: Dict[str, float] = field(default_factory=dict)
    
    # Rebalance Orders
    rebalance_trades: List[Dict[str, Any]] = field(default_factory=list)
    # Format: [{'symbol': str, 'side': str, 'quantity': float}, ...]
    
    # Constraints
    max_turnover: float = 0.0
    urgency: str = "normal"  # 'immediate', 'normal', 'patient'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'reason': self.rebalance_reason,
            'allocations': {
                'current': self.current_allocations,
                'target': self.target_allocations
            },
            'trades_count': len(self.rebalance_trades)
        }


@dataclass
class SystemStateEvent(Event):
    """
    System State Event - Periodisches System-Health Update
    """
    event_type: EventType = field(default=EventType.SYSTEM_STATE, init=False)
    
    # Component Health
    component_status: Dict[str, str] = field(default_factory=dict)
    # Format: {'data_handler': 'healthy', 'risk_engine': 'degraded', ...}
    
    # Performance Metrics
    events_processed: int = 0
    events_per_second: float = 0.0
    event_queue_size: int = 0
    
    # System Resources
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Latencies
    component_latencies: Dict[str, float] = field(default_factory=dict)
    # Format: {'risk_engine_ms': 12.5, 'allocator_ms': 8.3, ...}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'health': self.component_status,
            'performance': {
                'events_processed': self.events_processed,
                'eps': self.events_per_second,
                'queue_size': self.event_queue_size
            },
            'latencies': self.component_latencies
        }


# Utility Functions

def create_market_event_from_candle(symbol: str, timestamp: datetime, 
                                   ohlcv: tuple, exchange: str = "binance") -> MarketEvent:
    """Erstellt MarketEvent aus OHLCV-Daten"""
    open_price, high, low, close, volume = ohlcv
    
    # Simuliere Bid/Ask aus Close
    spread_bps = 10  # 10 bps default spread
    half_spread = close * (spread_bps / 20000)
    
    return MarketEvent(
        timestamp=timestamp,
        symbol=symbol,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
        bid_price=close - half_spread,
        ask_price=close + half_spread,
        spread_bps=spread_bps,
        exchange=exchange
    )


def create_signal_event_from_tier1_signal(tier1_signal: Any, 
                                         market_timestamp: datetime) -> SignalEvent:
    """Konvertiert tier1_models.Signal zu SignalEvent"""
    return SignalEvent(
        timestamp=market_timestamp,  # Wichtig: Market timestamp, nicht current time!
        signal_id=tier1_signal.signal_id,
        stream_id=tier1_signal.stream_id,
        symbol=tier1_signal.asset,
        direction=tier1_signal.direction,
        confidence=tier1_signal.confidence,
        expected_profit_pts=tier1_signal.expected_profit_pts,
        expected_duration_min=tier1_signal.expected_duration_min,
        origin=tier1_signal.origin,
        signal_metadata=tier1_signal.source_metadata,
        signal_generated_at=tier1_signal.timestamp
    )