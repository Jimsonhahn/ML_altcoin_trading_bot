"""
Tier-1 Trading System - Datenmodelle und Schnittstellen
Elite-Softwarearchitektur für institutionelles Algo-Trading
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import uuid


class SignalDirection(Enum):
    """Signal Direction Enumeration"""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class OrderStatus(Enum):
    """Order Status Enumeration"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class MarketRegime(Enum):
    """Market Regime Classification"""
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    MEAN_REVERTING = "mean_reverting"
    MOMENTUM = "momentum"
    CRISIS = "crisis"
    RECOVERY = "recovery"


@dataclass
class Signal:
    """
    Core Signal Data Structure für Tier-1 Trading System
    Exakte Spezifikation wie angefordert
    """
    stream_id: str
    asset: str
    direction: str  # 'long', 'short'
    confidence: float
    timestamp: datetime
    expected_profit_pts: float
    expected_duration_min: int
    source_metadata: Dict[str, Any]
    origin: str  # z.B. 'orderbook_alpha', 'vol_arb'
    
    # Zusätzliche Tier-1 Felder
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = field(default=5)  # 1=highest, 10=lowest
    risk_score: Optional[float] = None
    market_impact_estimate: Optional[float] = None
    
    def __post_init__(self):
        """Validierung und Normalisierung"""
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        
        if self.direction not in ['long', 'short', 'neutral']:
            raise ValueError(f"Direction must be 'long', 'short', or 'neutral', got {self.direction}")
        
        self.direction = SignalDirection(self.direction)


@dataclass
class Allocation:
    """
    Capital Allocation Result from CapitalAllocator
    """
    signal_id: str
    amount: float  # USD Amount to allocate
    position_size: float  # Number of shares/contracts
    leverage: float = 1.0
    max_risk_per_trade: float = 0.02  # 2% max risk
    kelly_fraction: Optional[float] = None
    volatility_adjusted: bool = False
    
    # Risk Parity Components
    strategy_weight: float = 0.0
    volatility_weight: float = 0.0
    correlation_adjustment: float = 1.0
    
    allocation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validation"""
        if self.amount < 0:
            raise ValueError("Allocation amount cannot be negative")


@dataclass
class Order:
    """
    Order Data Structure from ExecutionLayer
    """
    order_id: str
    signal_id: str
    asset: str
    side: str  # 'buy', 'sell'
    quantity: float
    price: Optional[float] = None  # None for market orders
    order_type: str = "market"  # 'market', 'limit', 'stop'
    
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    
    # Execution Metadata
    execution_score: float = 0.0
    slippage_estimate: float = 0.0
    urgency_factor: float = 0.5  # 0=patient, 1=urgent
    
    timestamp_created: datetime = field(default_factory=datetime.now)
    timestamp_submitted: Optional[datetime] = None
    timestamp_filled: Optional[datetime] = None
    
    # Exchange/Venue Information
    exchange: str = "binance"
    venue_specific_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate order_id if not provided"""
        if not self.order_id:
            self.order_id = f"ORDER_{uuid.uuid4().hex[:8]}"


@dataclass
class RiskMetrics:
    """
    Portfolio Risk Metrics for Risk Engine
    """
    portfolio_var: float  # Value at Risk
    portfolio_cvar: float  # Conditional VaR
    max_drawdown: float
    sharpe_ratio: float
    volatility: float
    
    # Position-Level Risk
    concentration_risk: float
    correlation_risk: float
    leverage_ratio: float
    
    # Market Risk
    market_regime: MarketRegime
    regime_confidence: float
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StrategyMetrics:
    """
    Strategy Performance Metrics for EWMA calculation
    """
    strategy_name: str
    returns: List[float]
    sharpe_ratio: float
    volatility: float
    max_drawdown: float
    win_rate: float
    
    # EWMA Smoothed Metrics
    ewma_return: float = 0.0
    ewma_volatility: float = 0.0
    ewma_sharpe: float = 0.0
    
    # Risk Parity Weights
    inverse_volatility_weight: float = 0.0
    risk_parity_weight: float = 0.0
    
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionMetrics:
    """
    Execution Quality Metrics
    """
    signal_id: str
    execution_score: float  # 0.0 to 1.0
    slippage: float
    market_impact: float
    fill_rate: float
    latency_ms: float
    
    # Micro-structure factors
    bid_ask_spread: float
    order_book_depth: float
    volatility_regime: float
    
    timestamp: datetime = field(default_factory=datetime.now)


# Abstract Base Classes for Tier-1 Components

class IRiskEngine(ABC):
    """Interface for Risk Engine"""
    
    @abstractmethod
    async def approve(self, signal: Signal) -> bool:
        """Approve or reject signal based on risk analysis"""
        pass
    
    @abstractmethod
    async def get_current_regime(self) -> MarketRegime:
        """Get current market regime"""
        pass
    
    @abstractmethod
    async def calculate_portfolio_risk(self) -> RiskMetrics:
        """Calculate current portfolio risk metrics"""
        pass


class ICapitalAllocator(ABC):
    """Interface for Capital Allocator"""
    
    @abstractmethod
    async def allocate(self, signal: Signal) -> Optional[Allocation]:
        """Allocate capital for signal using advanced techniques"""
        pass
    
    @abstractmethod
    async def update_strategy_metrics(self, metrics: StrategyMetrics) -> None:
        """Update strategy performance metrics for EWMA"""
        pass
    
    @abstractmethod
    async def rebalance_portfolio(self) -> Dict[str, float]:
        """Risk parity rebalancing"""
        pass


class IExecutionLayer(ABC):
    """Interface for Execution Layer"""
    
    @abstractmethod
    async def score(self, signal: Signal, allocation: Allocation) -> float:
        """Calculate micro-difficulty score 0.0-1.0"""
        pass
    
    @abstractmethod
    async def place(self, signal: Signal, allocation: Allocation) -> Order:
        """Place order with optimal execution"""
        pass
    
    @abstractmethod
    async def get_execution_metrics(self, order_id: str) -> ExecutionMetrics:
        """Get execution quality metrics"""
        pass


@dataclass
class OrchestrationResult:
    """
    Result from QuantumOrchestrator processing
    """
    signal_id: str
    status: str  # 'executed', 'rejected', 'skipped', 'aborted'
    reason: Optional[str] = None
    order_id: Optional[str] = None
    capital_allocated: Optional[float] = None
    execution_score: Optional[float] = None
    
    # Detailed processing info
    risk_approved: bool = False
    allocation_amount: float = 0.0
    processing_time_ms: float = 0.0
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SystemState:
    """
    Current System State for monitoring
    """
    total_capital: float
    allocated_capital: float
    available_capital: float
    
    active_positions: int
    active_orders: int
    
    current_regime: MarketRegime
    portfolio_risk: RiskMetrics
    
    # Performance metrics
    daily_pnl: float
    total_pnl: float
    sharpe_ratio: float
    
    last_updated: datetime = field(default_factory=datetime.now)


# Utility Functions for Data Validation

def validate_signal(signal: Signal) -> bool:
    """Validate signal data integrity"""
    try:
        if not signal.stream_id or not signal.asset:
            return False
        
        if signal.confidence < 0 or signal.confidence > 1:
            return False
        
        if signal.expected_duration_min <= 0:
            return False
        
        return True
    except Exception:
        return False


def validate_allocation(allocation: Allocation) -> bool:
    """Validate allocation data integrity"""
    try:
        if allocation.amount < 0:
            return False
        
        if allocation.leverage < 0:
            return False
        
        if allocation.max_risk_per_trade < 0 or allocation.max_risk_per_trade > 1:
            return False
        
        return True
    except Exception:
        return False


# Constants for Tier-1 System
class SystemConstants:
    """System-wide constants"""
    
    # Risk Management
    MAX_POSITION_SIZE = 0.1  # 10% max position size
    MAX_LEVERAGE = 3.0
    MAX_DAILY_LOSS = 0.02  # 2% max daily loss
    
    # Execution
    MIN_EXECUTION_SCORE = 0.7
    MAX_SLIPPAGE_TOLERANCE = 0.005  # 0.5%
    
    # Portfolio
    MIN_DIVERSIFICATION = 5  # Minimum 5 positions
    MAX_CORRELATION = 0.7
    
    # EWMA Parameters
    EWMA_ALPHA = 0.05  # 5% weight to new observation
    LOOKBACK_DAYS = 252  # 1 year for metrics calculation