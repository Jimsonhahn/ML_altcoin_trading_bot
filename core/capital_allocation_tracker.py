"""
Capital Allocation Tracker
==========================

Professional capital allocation tracking and management system with:
- Real-time capital flow monitoring
- Strategy-wise allocation tracking
- Performance attribution analysis
- Capital efficiency metrics
- Dynamic rebalancing alerts
- Compliance and risk monitoring
"""

import logging
import threading
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import pandas as pd

from core.interfaces import global_event_bus

logger = logging.getLogger(__name__)

class AllocationStatus(Enum):
    """Capital allocation status"""
    ACTIVE = "active"
    PENDING = "pending"
    TRANSITIONING = "transitioning"
    SUSPENDED = "suspended"
    CLOSED = "closed"

class AllocationEvent(Enum):
    """Types of allocation events"""
    INITIAL_ALLOCATION = "initial_allocation"
    CAPITAL_INCREASE = "capital_increase"
    CAPITAL_DECREASE = "capital_decrease"
    REBALANCE = "rebalance"
    STRATEGY_CHANGE = "strategy_change"
    PROFIT_WITHDRAWAL = "profit_withdrawal"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class CapitalAllocation:
    """Individual strategy capital allocation"""
    strategy_name: str
    allocated_capital: float
    utilized_capital: float
    available_capital: float
    target_allocation_pct: float
    actual_allocation_pct: float
    
    # Performance metrics
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    # Efficiency metrics
    capital_efficiency: float = 0.0  # PnL / Allocated Capital
    utilization_rate: float = 0.0    # Utilized / Allocated
    
    # Status and timing
    status: AllocationStatus = AllocationStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Risk metrics
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    var_95: float = 0.0

@dataclass
class AllocationEvent:
    """Capital allocation event record"""
    event_id: str
    event_type: AllocationEvent
    strategy_name: str
    amount: float
    previous_allocation: float
    new_allocation: float
    reason: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PortfolioSummary:
    """Portfolio-level capital allocation summary"""
    total_capital: float
    allocated_capital: float
    available_capital: float
    total_pnl: float
    allocation_efficiency: float
    
    # Distribution metrics
    strategy_count: int
    max_strategy_allocation_pct: float
    min_strategy_allocation_pct: float
    allocation_concentration: float  # HHI of allocations
    
    # Performance attribution
    best_performing_strategy: str
    worst_performing_strategy: str
    top_contributor_pnl: float
    worst_contributor_pnl: float
    
    timestamp: datetime = field(default_factory=datetime.now)

class CapitalAllocationTracker:
    """
    Professional capital allocation tracking system
    """
    
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.allocation_config = settings.get('capital_allocation', {})
        
        # Configuration parameters
        self.total_capital = self.allocation_config.get('initial_capital', 10000.0)
        self.min_allocation = self.allocation_config.get('min_allocation', 100.0)
        self.max_allocation_pct = self.allocation_config.get('max_allocation_pct', 0.5)  # 50% max per strategy
        self.rebalance_threshold = self.allocation_config.get('rebalance_threshold', 0.05)  # 5%
        self.emergency_stop_threshold = self.allocation_config.get('emergency_stop_threshold', 0.20)  # 20% loss
        
        # Tracking state
        self.allocations: Dict[str, CapitalAllocation] = {}
        self.allocation_history: List[AllocationEvent] = []
        self.performance_history: deque = deque(maxlen=1000)  # Keep last 1000 snapshots
        
        # Threading for real-time tracking
        self._tracking_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        
        # Metrics calculation
        self.update_interval = self.allocation_config.get('update_interval', 60.0)  # 60 seconds
        self.performance_window = self.allocation_config.get('performance_window_days', 30)
        
        # Event callbacks
        self.allocation_callbacks: List[Callable] = []
        self.rebalance_callbacks: List[Callable] = []
        
        # Setup event handlers
        self._setup_event_handlers()
        
        logger.info("Capital Allocation Tracker initialized")
    
    def _setup_event_handlers(self):
        """Setup event bus handlers"""
        global_event_bus.subscribe("trade_executed", self._on_trade_executed)
        global_event_bus.subscribe("position_update", self._on_position_update)
        global_event_bus.subscribe("strategy_pnl_update", self._on_strategy_pnl_update)
        global_event_bus.subscribe("capital_deposit", self._on_capital_deposit)
        global_event_bus.subscribe("capital_withdrawal", self._on_capital_withdrawal)
    
    def start_tracking(self):
        """Start real-time capital allocation tracking"""
        with self._lock:
            if self._tracking_thread and self._tracking_thread.is_alive():
                logger.warning("Capital tracking already running")
                return
            
            self._stop_event.clear()
            self._tracking_thread = threading.Thread(
                target=self._tracking_loop,
                name="CapitalAllocationTracker",
                daemon=True
            )
            self._tracking_thread.start()
            logger.info("Capital allocation tracking started")
    
    def stop_tracking(self):
        """Stop real-time tracking"""
        self._stop_event.set()
        if self._tracking_thread:
            self._tracking_thread.join(timeout=5.0)
            logger.info("Capital allocation tracking stopped")
    
    def _tracking_loop(self):
        """Main tracking loop"""
        logger.info("Capital allocation tracking loop started")
        
        while not self._stop_event.is_set():
            try:
                self._update_allocation_metrics()
                self._check_rebalance_triggers()
                self._check_risk_limits()
                self._save_performance_snapshot()
                
                self._stop_event.wait(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in capital tracking loop: {e}", exc_info=True)
                self._stop_event.wait(self.update_interval)
    
    def allocate_capital(self, 
                        strategy_name: str, 
                        amount: float, 
                        target_pct: Optional[float] = None,
                        reason: str = "Manual allocation") -> bool:
        """
        Allocate capital to a strategy
        
        Args:
            strategy_name: Name of the strategy
            amount: Amount to allocate
            target_pct: Target percentage allocation (optional)
            reason: Reason for allocation
            
        Returns:
            bool: Success status
        """
        with self._lock:
            try:
                # Validation
                if amount < self.min_allocation:
                    logger.error(f"Allocation amount {amount} below minimum {self.min_allocation}")
                    return False
                
                available_capital = self._get_available_capital()
                if amount > available_capital:
                    logger.error(f"Insufficient capital: requested {amount}, available {available_capital}")
                    return False
                
                # Check max allocation percentage
                total_after_allocation = self.total_capital
                allocation_pct = amount / total_after_allocation
                if allocation_pct > self.max_allocation_pct:
                    logger.error(f"Allocation {allocation_pct:.1%} exceeds max {self.max_allocation_pct:.1%}")
                    return False
                
                # Create or update allocation
                if strategy_name in self.allocations:
                    # Update existing allocation
                    allocation = self.allocations[strategy_name]
                    previous_amount = allocation.allocated_capital
                    allocation.allocated_capital = amount
                    allocation.target_allocation_pct = target_pct or (amount / total_after_allocation)
                    allocation.actual_allocation_pct = amount / total_after_allocation
                    allocation.available_capital = amount - allocation.utilized_capital
                    allocation.last_updated = datetime.now()
                    
                    event_type = AllocationEvent.CAPITAL_INCREASE if amount > previous_amount else AllocationEvent.CAPITAL_DECREASE
                    
                else:
                    # Create new allocation
                    allocation = CapitalAllocation(
                        strategy_name=strategy_name,
                        allocated_capital=amount,
                        utilized_capital=0.0,
                        available_capital=amount,
                        target_allocation_pct=target_pct or (amount / total_after_allocation),
                        actual_allocation_pct=amount / total_after_allocation,
                        status=AllocationStatus.ACTIVE
                    )
                    self.allocations[strategy_name] = allocation
                    previous_amount = 0.0
                    event_type = AllocationEvent.INITIAL_ALLOCATION
                
                # Record event
                self._record_allocation_event(
                    event_type=event_type,
                    strategy_name=strategy_name,
                    amount=amount,
                    previous_allocation=previous_amount,
                    new_allocation=amount,
                    reason=reason
                )
                
                # Update portfolio metrics
                self._update_portfolio_allocations()
                
                # Publish event
                global_event_bus.publish("capital_allocated", {
                    'strategy_name': strategy_name,
                    'amount': amount,
                    'total_allocated': sum(a.allocated_capital for a in self.allocations.values()),
                    'available_capital': self._get_available_capital()
                })
                
                # Call callbacks
                for callback in self.allocation_callbacks:
                    try:
                        callback(strategy_name, amount, allocation)
                    except Exception as e:
                        logger.error(f"Error in allocation callback: {e}")
                
                logger.info(f"Capital allocated: {strategy_name} = ${amount:.2f} ({allocation_pct:.1%})")
                return True
                
            except Exception as e:
                logger.error(f"Error allocating capital to {strategy_name}: {e}")
                return False
    
    def deallocate_capital(self, 
                          strategy_name: str, 
                          amount: Optional[float] = None,
                          reason: str = "Manual deallocation") -> bool:
        """
        Deallocate capital from a strategy
        
        Args:
            strategy_name: Name of the strategy
            amount: Amount to deallocate (None = all)
            reason: Reason for deallocation
            
        Returns:
            bool: Success status
        """
        with self._lock:
            try:
                if strategy_name not in self.allocations:
                    logger.error(f"Strategy {strategy_name} not found in allocations")
                    return False
                
                allocation = self.allocations[strategy_name]
                
                # Determine amount to deallocate
                if amount is None:
                    amount = allocation.allocated_capital
                
                if amount > allocation.available_capital:
                    logger.error(f"Cannot deallocate {amount}: only {allocation.available_capital} available")
                    return False
                
                previous_amount = allocation.allocated_capital
                allocation.allocated_capital -= amount
                allocation.available_capital -= amount
                
                # Update percentages
                if self.total_capital > 0:
                    allocation.actual_allocation_pct = allocation.allocated_capital / self.total_capital
                
                allocation.last_updated = datetime.now()
                
                # If fully deallocated, mark as closed
                if allocation.allocated_capital <= 0:
                    allocation.status = AllocationStatus.CLOSED
                
                # Record event
                self._record_allocation_event(
                    event_type=AllocationEvent.CAPITAL_DECREASE,
                    strategy_name=strategy_name,
                    amount=-amount,  # Negative for deallocation
                    previous_allocation=previous_amount,
                    new_allocation=allocation.allocated_capital,
                    reason=reason
                )
                
                # Update portfolio metrics
                self._update_portfolio_allocations()
                
                # Publish event
                global_event_bus.publish("capital_deallocated", {
                    'strategy_name': strategy_name,
                    'amount': amount,
                    'remaining_allocation': allocation.allocated_capital,
                    'available_capital': self._get_available_capital()
                })
                
                logger.info(f"Capital deallocated: {strategy_name} = ${amount:.2f}")
                return True
                
            except Exception as e:
                logger.error(f"Error deallocating capital from {strategy_name}: {e}")
                return False
    
    def update_capital_utilization(self, strategy_name: str, utilized_amount: float):
        """Update how much capital a strategy is currently using"""
        with self._lock:
            if strategy_name in self.allocations:
                allocation = self.allocations[strategy_name]
                allocation.utilized_capital = min(utilized_amount, allocation.allocated_capital)
                allocation.available_capital = allocation.allocated_capital - allocation.utilized_capital
                allocation.utilization_rate = allocation.utilized_capital / allocation.allocated_capital if allocation.allocated_capital > 0 else 0
                allocation.last_updated = datetime.now()
                
                logger.debug(f"Capital utilization updated: {strategy_name} = ${utilized_amount:.2f}")
    
    def update_strategy_pnl(self, strategy_name: str, realized_pnl: float, unrealized_pnl: float):
        """Update P&L for a strategy"""
        with self._lock:
            if strategy_name in self.allocations:
                allocation = self.allocations[strategy_name]
                allocation.realized_pnl = realized_pnl
                allocation.unrealized_pnl = unrealized_pnl
                allocation.total_pnl = realized_pnl + unrealized_pnl
                
                # Calculate capital efficiency
                if allocation.allocated_capital > 0:
                    allocation.capital_efficiency = allocation.total_pnl / allocation.allocated_capital
                
                allocation.last_updated = datetime.now()
                
                logger.debug(f"P&L updated: {strategy_name} = ${allocation.total_pnl:.2f}")
    
    def _get_available_capital(self) -> float:
        """Calculate available capital for allocation"""
        allocated = sum(allocation.allocated_capital for allocation in self.allocations.values())
        return self.total_capital - allocated
    
    def _update_allocation_metrics(self):
        """Update allocation metrics for all strategies"""
        with self._lock:
            for allocation in self.allocations.values():
                if allocation.status == AllocationStatus.ACTIVE:
                    # Update allocation percentage
                    if self.total_capital > 0:
                        allocation.actual_allocation_pct = allocation.allocated_capital / self.total_capital
                    
                    # Calculate performance metrics
                    self._calculate_performance_metrics(allocation)
    
    def _calculate_performance_metrics(self, allocation: CapitalAllocation):
        """Calculate performance metrics for an allocation"""
        try:
            # Get historical performance data for this strategy
            strategy_history = [
                snapshot for snapshot in self.performance_history
                if allocation.strategy_name in snapshot.get('strategies', {})
            ]
            
            if len(strategy_history) < 2:
                return
            
            # Calculate returns
            returns = []
            for i in range(1, len(strategy_history)):
                prev_pnl = strategy_history[i-1]['strategies'][allocation.strategy_name]['total_pnl']
                curr_pnl = strategy_history[i]['strategies'][allocation.strategy_name]['total_pnl']
                if allocation.allocated_capital > 0:
                    returns.append((curr_pnl - prev_pnl) / allocation.allocated_capital)
            
            if len(returns) < 2:
                return
            
            returns_array = np.array(returns)
            
            # Calculate Sharpe ratio
            if np.std(returns_array) > 0:
                allocation.sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)  # Annualized
            
            # Calculate max drawdown
            cumulative_returns = np.cumsum(returns_array)
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (peak - cumulative_returns)
            allocation.max_drawdown = np.max(drawdown)
            
            # Calculate VaR (95%)
            allocation.var_95 = np.percentile(returns_array, 5) * allocation.allocated_capital
            
        except Exception as e:
            logger.debug(f"Error calculating performance metrics for {allocation.strategy_name}: {e}")
    
    def _check_rebalance_triggers(self):
        """Check if rebalancing is needed"""
        try:
            rebalance_needed = False
            rebalance_reasons = []
            
            for allocation in self.allocations.values():
                if allocation.status != AllocationStatus.ACTIVE:
                    continue
                
                # Check allocation drift
                target_pct = allocation.target_allocation_pct
                actual_pct = allocation.actual_allocation_pct
                drift = abs(target_pct - actual_pct)
                
                if drift > self.rebalance_threshold:
                    rebalance_needed = True
                    rebalance_reasons.append(f"{allocation.strategy_name}: {drift:.1%} drift")
                
                # Check underutilization
                if allocation.utilization_rate < 0.5 and allocation.allocated_capital > self.min_allocation * 2:
                    rebalance_needed = True
                    rebalance_reasons.append(f"{allocation.strategy_name}: low utilization {allocation.utilization_rate:.1%}")
            
            if rebalance_needed:
                logger.info(f"Rebalance triggered: {', '.join(rebalance_reasons)}")
                
                # Call rebalance callbacks
                for callback in self.rebalance_callbacks:
                    try:
                        callback(rebalance_reasons)
                    except Exception as e:
                        logger.error(f"Error in rebalance callback: {e}")
                
                # Publish event
                global_event_bus.publish("rebalance_needed", {
                    'reasons': rebalance_reasons,
                    'timestamp': datetime.now().isoformat()
                })
                
        except Exception as e:
            logger.error(f"Error checking rebalance triggers: {e}")
    
    def _check_risk_limits(self):
        """Check risk limits and trigger emergency stops if needed"""
        try:
            for allocation in self.allocations.values():
                if allocation.status != AllocationStatus.ACTIVE:
                    continue
                
                # Check for excessive losses
                if allocation.allocated_capital > 0:
                    loss_pct = -allocation.total_pnl / allocation.allocated_capital
                    
                    if loss_pct > self.emergency_stop_threshold:
                        logger.critical(f"Emergency stop triggered for {allocation.strategy_name}: "
                                      f"{loss_pct:.1%} loss exceeds {self.emergency_stop_threshold:.1%}")
                        
                        allocation.status = AllocationStatus.SUSPENDED
                        
                        # Record emergency event
                        self._record_allocation_event(
                            event_type=AllocationEvent.EMERGENCY_STOP,
                            strategy_name=allocation.strategy_name,
                            amount=0,
                            previous_allocation=allocation.allocated_capital,
                            new_allocation=allocation.allocated_capital,
                            reason=f"Excessive loss: {loss_pct:.1%}"
                        )
                        
                        # Publish emergency event
                        global_event_bus.publish("emergency_allocation_stop", {
                            'strategy_name': allocation.strategy_name,
                            'loss_percentage': loss_pct,
                            'allocated_capital': allocation.allocated_capital,
                            'total_pnl': allocation.total_pnl
                        })
                        
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
    
    def _save_performance_snapshot(self):
        """Save current performance snapshot"""
        try:
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'total_capital': self.total_capital,
                'allocated_capital': sum(a.allocated_capital for a in self.allocations.values()),
                'total_pnl': sum(a.total_pnl for a in self.allocations.values()),
                'strategies': {}
            }
            
            for name, allocation in self.allocations.items():
                snapshot['strategies'][name] = {
                    'allocated_capital': allocation.allocated_capital,
                    'utilized_capital': allocation.utilized_capital,
                    'total_pnl': allocation.total_pnl,
                    'realized_pnl': allocation.realized_pnl,
                    'unrealized_pnl': allocation.unrealized_pnl,
                    'capital_efficiency': allocation.capital_efficiency,
                    'utilization_rate': allocation.utilization_rate,
                    'status': allocation.status.value
                }
            
            self.performance_history.append(snapshot)
            
        except Exception as e:
            logger.error(f"Error saving performance snapshot: {e}")
    
    def _record_allocation_event(self, 
                               event_type: AllocationEvent,
                               strategy_name: str,
                               amount: float,
                               previous_allocation: float,
                               new_allocation: float,
                               reason: str,
                               metadata: Optional[Dict[str, Any]] = None):
        """Record an allocation event"""
        event_id = f"alloc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.allocation_history)}"
        
        event = AllocationEvent(
            event_id=event_id,
            event_type=event_type,
            strategy_name=strategy_name,
            amount=amount,
            previous_allocation=previous_allocation,
            new_allocation=new_allocation,
            reason=reason,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.allocation_history.append(event)
        
        # Keep only last 1000 events
        if len(self.allocation_history) > 1000:
            self.allocation_history = self.allocation_history[-1000:]
    
    def _update_portfolio_allocations(self):
        """Update portfolio-level allocation metrics"""
        # Recalculate all allocation percentages
        total_allocated = sum(a.allocated_capital for a in self.allocations.values())
        
        for allocation in self.allocations.values():
            if total_allocated > 0:
                allocation.actual_allocation_pct = allocation.allocated_capital / self.total_capital
    
    # Event Handlers
    def _on_trade_executed(self, data: Dict[str, Any]):
        """Handle trade execution events"""
        strategy_name = data.get('strategy_name')
        trade_amount = data.get('amount', 0)
        
        if strategy_name:
            # Update capital utilization based on trade
            current_utilization = data.get('current_utilization', 0)
            self.update_capital_utilization(strategy_name, current_utilization)
    
    def _on_position_update(self, data: Dict[str, Any]):
        """Handle position update events"""
        strategy_name = data.get('strategy_name')
        unrealized_pnl = data.get('unrealized_pnl', 0)
        
        if strategy_name and strategy_name in self.allocations:
            # Update unrealized P&L
            allocation = self.allocations[strategy_name]
            self.update_strategy_pnl(strategy_name, allocation.realized_pnl, unrealized_pnl)
    
    def _on_strategy_pnl_update(self, data: Dict[str, Any]):
        """Handle strategy P&L updates"""
        strategy_name = data.get('strategy_name')
        realized_pnl = data.get('realized_pnl', 0)
        unrealized_pnl = data.get('unrealized_pnl', 0)
        
        if strategy_name:
            self.update_strategy_pnl(strategy_name, realized_pnl, unrealized_pnl)
    
    def _on_capital_deposit(self, data: Dict[str, Any]):
        """Handle capital deposit events"""
        amount = data.get('amount', 0)
        if amount > 0:
            self.total_capital += amount
            logger.info(f"Capital deposited: ${amount:.2f}, new total: ${self.total_capital:.2f}")
    
    def _on_capital_withdrawal(self, data: Dict[str, Any]):
        """Handle capital withdrawal events"""
        amount = data.get('amount', 0)
        if amount > 0 and amount <= self._get_available_capital():
            self.total_capital -= amount
            logger.info(f"Capital withdrawn: ${amount:.2f}, new total: ${self.total_capital:.2f}")
    
    # Public API
    def get_portfolio_summary(self) -> PortfolioSummary:
        """Get comprehensive portfolio summary"""
        with self._lock:
            allocated_capital = sum(a.allocated_capital for a in self.allocations.values())
            available_capital = self.total_capital - allocated_capital
            total_pnl = sum(a.total_pnl for a in self.allocations.values())
            
            # Calculate allocation efficiency (total PnL / allocated capital)
            allocation_efficiency = total_pnl / allocated_capital if allocated_capital > 0 else 0
            
            # Calculate allocation concentration (HHI)
            allocation_shares = [a.actual_allocation_pct for a in self.allocations.values()]
            concentration = sum(share ** 2 for share in allocation_shares)
            
            # Find best/worst performers
            active_allocations = [a for a in self.allocations.values() if a.status == AllocationStatus.ACTIVE]
            
            best_performer = ""
            worst_performer = ""
            top_contributor = 0.0
            worst_contributor = 0.0
            
            if active_allocations:
                best_allocation = max(active_allocations, key=lambda a: a.capital_efficiency)
                worst_allocation = min(active_allocations, key=lambda a: a.capital_efficiency)
                
                best_performer = best_allocation.strategy_name
                worst_performer = worst_allocation.strategy_name
                top_contributor = best_allocation.total_pnl
                worst_contributor = worst_allocation.total_pnl
            
            return PortfolioSummary(
                total_capital=self.total_capital,
                allocated_capital=allocated_capital,
                available_capital=available_capital,
                total_pnl=total_pnl,
                allocation_efficiency=allocation_efficiency,
                strategy_count=len([a for a in self.allocations.values() if a.status == AllocationStatus.ACTIVE]),
                max_strategy_allocation_pct=max(allocation_shares) if allocation_shares else 0,
                min_strategy_allocation_pct=min(allocation_shares) if allocation_shares else 0,
                allocation_concentration=concentration,
                best_performing_strategy=best_performer,
                worst_performing_strategy=worst_performer,
                top_contributor_pnl=top_contributor,
                worst_contributor_pnl=worst_contributor
            )
    
    def get_allocation_details(self, strategy_name: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed allocation information"""
        with self._lock:
            if strategy_name:
                if strategy_name in self.allocations:
                    allocation = self.allocations[strategy_name]
                    return {
                        'strategy_name': allocation.strategy_name,
                        'allocated_capital': allocation.allocated_capital,
                        'utilized_capital': allocation.utilized_capital,
                        'available_capital': allocation.available_capital,
                        'target_allocation_pct': allocation.target_allocation_pct,
                        'actual_allocation_pct': allocation.actual_allocation_pct,
                        'total_pnl': allocation.total_pnl,
                        'realized_pnl': allocation.realized_pnl,
                        'unrealized_pnl': allocation.unrealized_pnl,
                        'capital_efficiency': allocation.capital_efficiency,
                        'utilization_rate': allocation.utilization_rate,
                        'max_drawdown': allocation.max_drawdown,
                        'sharpe_ratio': allocation.sharpe_ratio,
                        'var_95': allocation.var_95,
                        'status': allocation.status.value,
                        'created_at': allocation.created_at.isoformat(),
                        'last_updated': allocation.last_updated.isoformat()
                    }
                else:
                    return {}
            else:
                # Return all allocations
                return {
                    name: self.get_allocation_details(name)
                    for name in self.allocations.keys()
                }
    
    def get_allocation_history(self, 
                             strategy_name: Optional[str] = None,
                             event_type: Optional[AllocationEvent] = None,
                             limit: int = 100) -> List[Dict[str, Any]]:
        """Get allocation history"""
        with self._lock:
            events = self.allocation_history
            
            # Filter by strategy
            if strategy_name:
                events = [e for e in events if e.strategy_name == strategy_name]
            
            # Filter by event type
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            
            # Limit results
            events = events[-limit:]
            
            return [
                {
                    'event_id': e.event_id,
                    'event_type': e.event_type.value,
                    'strategy_name': e.strategy_name,
                    'amount': e.amount,
                    'previous_allocation': e.previous_allocation,
                    'new_allocation': e.new_allocation,
                    'reason': e.reason,
                    'timestamp': e.timestamp.isoformat(),
                    'metadata': e.metadata
                }
                for e in events
            ]
    
    def get_performance_attribution(self, days: int = 30) -> Dict[str, Any]:
        """Get performance attribution analysis"""
        with self._lock:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Filter recent performance history
            recent_history = [
                snapshot for snapshot in self.performance_history
                if datetime.fromisoformat(snapshot['timestamp']) >= cutoff_date
            ]
            
            if len(recent_history) < 2:
                return {}
            
            attribution = {}
            
            for strategy_name in self.allocations.keys():
                strategy_data = []
                
                for snapshot in recent_history:
                    if strategy_name in snapshot.get('strategies', {}):
                        strategy_data.append(snapshot['strategies'][strategy_name])
                
                if len(strategy_data) >= 2:
                    # Calculate contribution to portfolio performance
                    pnl_change = strategy_data[-1]['total_pnl'] - strategy_data[0]['total_pnl']
                    avg_allocation = np.mean([d['allocated_capital'] for d in strategy_data])
                    
                    attribution[strategy_name] = {
                        'pnl_contribution': pnl_change,
                        'avg_allocation': avg_allocation,
                        'performance_contribution_pct': pnl_change / avg_allocation if avg_allocation > 0 else 0,
                        'avg_utilization': np.mean([d['utilization_rate'] for d in strategy_data]),
                        'avg_efficiency': np.mean([d['capital_efficiency'] for d in strategy_data])
                    }
            
            return attribution
    
    def add_allocation_callback(self, callback: Callable):
        """Add callback for allocation events"""
        self.allocation_callbacks.append(callback)
    
    def add_rebalance_callback(self, callback: Callable):
        """Add callback for rebalance events"""
        self.rebalance_callbacks.append(callback)
    
    def export_allocation_report(self, filepath: str):
        """Export comprehensive allocation report"""
        try:
            report = {
                'generated_at': datetime.now().isoformat(),
                'portfolio_summary': self.get_portfolio_summary().__dict__,
                'allocations': self.get_allocation_details(),
                'recent_history': self.get_allocation_history(limit=50),
                'performance_attribution': self.get_performance_attribution(),
                'performance_snapshots': list(self.performance_history)[-100:]  # Last 100 snapshots
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Allocation report exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting allocation report: {e}")

# Global instance
_allocation_tracker_instance = None

def get_allocation_tracker(settings: Optional[Dict[str, Any]] = None) -> CapitalAllocationTracker:
    """Get global allocation tracker instance"""
    global _allocation_tracker_instance
    if _allocation_tracker_instance is None:
        if settings is None:
            settings = {}
        _allocation_tracker_instance = CapitalAllocationTracker(settings)
    return _allocation_tracker_instance