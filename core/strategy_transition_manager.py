"""
Strategy Transition Manager
===========================

Professional strategy transition system with:
- Gradual position unwinding
- Risk-aware transition timing
- Capital preservation during transitions
- Conflict resolution between strategies
- Emergency transition handling
"""

import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

from core.interfaces import global_event_bus
from strategies.strategy_base import Strategy

logger = logging.getLogger(__name__)

class TransitionState(Enum):
    """Strategy transition states"""
    IDLE = "idle"
    PLANNING = "planning"
    UNWINDING = "unwinding"
    TRANSITIONING = "transitioning"
    COMPLETING = "completing"
    EMERGENCY_STOP = "emergency_stop"

class TransitionPriority(Enum):
    """Transition priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EMERGENCY = "emergency"

@dataclass
class PositionInfo:
    """Information about a position to be unwound"""
    symbol: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    side: str  # 'long' or 'short'
    strategy_name: str
    importance: float = 1.0  # 0-1, how important to close immediately

@dataclass
class TransitionPlan:
    """Comprehensive transition plan"""
    transition_id: str
    source_strategies: List[str]
    target_strategies: List[str]
    priority: TransitionPriority
    estimated_duration: timedelta
    unwinding_positions: List[PositionInfo]
    capital_to_transfer: float
    risk_constraints: Dict[str, Any]
    created_at: datetime
    
    # Progress tracking
    progress_percentage: float = 0.0
    current_state: TransitionState = TransitionState.PLANNING
    completed_positions: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

class StrategyTransitionManager:
    """
    Professional strategy transition management system
    """
    
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.transition_config = settings.get('strategy_transitions', {})
        
        # Transition parameters
        self.max_concurrent_transitions = self.transition_config.get('max_concurrent', 3)
        self.default_unwind_duration = timedelta(minutes=self.transition_config.get('default_unwind_minutes', 30))
        self.emergency_unwind_duration = timedelta(minutes=self.transition_config.get('emergency_unwind_minutes', 5))
        self.risk_pause_threshold = self.transition_config.get('risk_pause_threshold', 0.10)  # 10% drawdown
        
        # State management
        self.active_transitions: Dict[str, TransitionPlan] = {}
        self.transition_queue: List[TransitionPlan] = []
        self.transition_history: List[TransitionPlan] = []
        
        # Threading
        self._transition_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        
        # Strategy and position tracking
        self.active_strategies: Dict[str, Strategy] = {}
        self.position_tracker: Dict[str, List[PositionInfo]] = {}
        
        # Callbacks
        self.position_close_callback: Optional[Callable] = None
        self.strategy_stop_callback: Optional[Callable] = None
        self.strategy_start_callback: Optional[Callable] = None
        
        # Setup event handlers
        self._setup_event_handlers()
        
        logger.info("Strategy Transition Manager initialized")
    
    def _setup_event_handlers(self):
        """Setup event bus handlers"""
        global_event_bus.subscribe("position_update", self._on_position_update)
        global_event_bus.subscribe("strategy_stopped", self._on_strategy_stopped)
        global_event_bus.subscribe("risk_limit_breached", self._on_risk_limit_breached)
        global_event_bus.subscribe("emergency_stop", self._on_emergency_stop)
    
    def start_transition_manager(self):
        """Start the transition management thread"""
        with self._lock:
            if self._transition_thread and self._transition_thread.is_alive():
                logger.warning("Transition manager already running")
                return
            
            self._stop_event.clear()
            self._transition_thread = threading.Thread(
                target=self._transition_loop,
                name="StrategyTransitionManager",
                daemon=True
            )
            self._transition_thread.start()
            logger.info("Strategy transition manager started")
    
    def stop_transition_manager(self):
        """Stop the transition management thread"""
        self._stop_event.set()
        if self._transition_thread:
            self._transition_thread.join(timeout=10.0)
            logger.info("Strategy transition manager stopped")
    
    def register_strategy(self, strategy_name: str, strategy: Strategy):
        """Register an active strategy"""
        with self._lock:
            self.active_strategies[strategy_name] = strategy
            self.position_tracker[strategy_name] = []
            logger.debug(f"Strategy registered: {strategy_name}")
    
    def unregister_strategy(self, strategy_name: str):
        """Unregister a strategy"""
        with self._lock:
            if strategy_name in self.active_strategies:
                del self.active_strategies[strategy_name]
            if strategy_name in self.position_tracker:
                del self.position_tracker[strategy_name]
            logger.debug(f"Strategy unregistered: {strategy_name}")
    
    def plan_transition(self, 
                       source_strategies: List[str],
                       target_strategies: List[str],
                       priority: TransitionPriority = TransitionPriority.NORMAL,
                       force_immediate: bool = False) -> str:
        """
        Plan a strategy transition
        
        Returns:
            transition_id: Unique identifier for the transition
        """
        
        transition_id = f"transition_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.transition_queue)}"
        
        # Analyze positions that need unwinding
        unwinding_positions = self._analyze_positions_to_unwind(source_strategies)
        
        # Calculate capital transfer
        capital_to_transfer = self._calculate_capital_transfer(source_strategies, target_strategies)
        
        # Estimate duration based on positions and market conditions
        estimated_duration = self._estimate_transition_duration(unwinding_positions, priority)
        
        # Create transition plan
        transition_plan = TransitionPlan(
            transition_id=transition_id,
            source_strategies=source_strategies,
            target_strategies=target_strategies,
            priority=priority,
            estimated_duration=estimated_duration,
            unwinding_positions=unwinding_positions,
            capital_to_transfer=capital_to_transfer,
            risk_constraints=self._calculate_risk_constraints(),
            created_at=datetime.now()
        )
        
        with self._lock:
            if force_immediate or priority == TransitionPriority.EMERGENCY:
                # Insert at front of queue
                self.transition_queue.insert(0, transition_plan)
            else:
                # Add to queue based on priority
                self._insert_by_priority(transition_plan)
        
        logger.info(f"Transition planned: {transition_id} ({priority.value}) - "
                   f"{len(unwinding_positions)} positions to unwind")
        
        # Publish event
        global_event_bus.publish("transition_planned", {
            'transition_id': transition_id,
            'source_strategies': source_strategies,
            'target_strategies': target_strategies,
            'estimated_duration': estimated_duration.total_seconds(),
            'priority': priority.value
        })
        
        return transition_id
    
    def _analyze_positions_to_unwind(self, source_strategies: List[str]) -> List[PositionInfo]:
        """Analyze which positions need to be unwound"""
        positions_to_unwind = []
        
        for strategy_name in source_strategies:
            if strategy_name in self.position_tracker:
                for position in self.position_tracker[strategy_name]:
                    # Calculate importance based on P&L, risk, and liquidity
                    importance = self._calculate_position_importance(position)
                    position.importance = importance
                    positions_to_unwind.append(position)
        
        # Sort by importance (highest first for fastest unwinding)
        positions_to_unwind.sort(key=lambda p: p.importance, reverse=True)
        
        return positions_to_unwind
    
    def _calculate_position_importance(self, position: PositionInfo) -> float:
        """Calculate how important it is to close a position immediately"""
        importance = 1.0
        
        # Reduce importance for profitable positions
        if position.unrealized_pnl > 0:
            importance *= 0.7
        
        # Increase importance for losing positions
        elif position.unrealized_pnl < 0:
            loss_pct = abs(position.unrealized_pnl) / (position.size * position.entry_price)
            importance *= (1.0 + loss_pct * 2)  # Increase urgency for losses
        
        # Adjust for position size (larger positions are more important)
        position_value = position.size * position.current_price
        if position_value > 1000:  # Large position threshold
            importance *= 1.2
        
        return min(importance, 2.0)  # Cap importance at 2.0
    
    def _calculate_capital_transfer(self, source_strategies: List[str], target_strategies: List[str]) -> float:
        """Calculate capital that will be transferred between strategies"""
        total_capital = 0.0
        
        for strategy_name in source_strategies:
            if strategy_name in self.active_strategies:
                strategy = self.active_strategies[strategy_name]
                if hasattr(strategy, 'get_allocated_capital'):
                    total_capital += strategy.get_allocated_capital()
        
        return total_capital
    
    def _estimate_transition_duration(self, positions: List[PositionInfo], priority: TransitionPriority) -> timedelta:
        """Estimate how long the transition will take"""
        if priority == TransitionPriority.EMERGENCY:
            return self.emergency_unwind_duration
        
        # Base duration
        base_duration = self.default_unwind_duration
        
        # Adjust based on number of positions
        position_factor = len(positions) * 0.1  # 10% more time per position
        
        # Adjust based on position sizes
        large_positions = sum(1 for p in positions if p.size * p.current_price > 1000)
        size_factor = large_positions * 0.2  # 20% more time per large position
        
        total_factor = 1.0 + position_factor + size_factor
        
        return timedelta(seconds=base_duration.total_seconds() * total_factor)
    
    def _calculate_risk_constraints(self) -> Dict[str, Any]:
        """Calculate risk constraints for the transition"""
        return {
            'max_slippage': 0.005,  # 0.5% max slippage
            'max_loss_per_position': 0.02,  # 2% max loss per position
            'pause_on_drawdown': self.risk_pause_threshold,
            'max_concurrent_closes': 3
        }
    
    def _insert_by_priority(self, transition_plan: TransitionPlan):
        """Insert transition plan into queue based on priority"""
        priority_order = {
            TransitionPriority.EMERGENCY: 0,
            TransitionPriority.HIGH: 1,
            TransitionPriority.NORMAL: 2,
            TransitionPriority.LOW: 3
        }
        
        plan_priority = priority_order[transition_plan.priority]
        
        # Find insertion point
        for i, existing_plan in enumerate(self.transition_queue):
            existing_priority = priority_order[existing_plan.priority]
            if plan_priority < existing_priority:
                self.transition_queue.insert(i, transition_plan)
                return
        
        # If not inserted, add to end
        self.transition_queue.append(transition_plan)
    
    def _transition_loop(self):
        """Main transition processing loop"""
        logger.info("Transition processing loop started")
        
        while not self._stop_event.is_set():
            try:
                with self._lock:
                    # Process active transitions
                    self._process_active_transitions()
                    
                    # Start new transitions if possible
                    self._start_new_transitions()
                    
                    # Cleanup completed transitions
                    self._cleanup_completed_transitions()
                
                # Sleep briefly to avoid tight loop
                self._stop_event.wait(1.0)
                
            except Exception as e:
                logger.error(f"Error in transition loop: {e}", exc_info=True)
                self._stop_event.wait(5.0)  # Wait longer on error
    
    def _process_active_transitions(self):
        """Process all active transitions"""
        for transition_id, transition in list(self.active_transitions.items()):
            try:
                if transition.current_state == TransitionState.UNWINDING:
                    self._process_unwinding_phase(transition)
                elif transition.current_state == TransitionState.TRANSITIONING:
                    self._process_transition_phase(transition)
                elif transition.current_state == TransitionState.COMPLETING:
                    self._process_completion_phase(transition)
                elif transition.current_state == TransitionState.EMERGENCY_STOP:
                    self._process_emergency_stop(transition)
                    
            except Exception as e:
                logger.error(f"Error processing transition {transition_id}: {e}")
                transition.errors.append(f"Processing error: {str(e)}")
    
    def _start_new_transitions(self):
        """Start new transitions from the queue"""
        if len(self.active_transitions) >= self.max_concurrent_transitions:
            return
        
        if not self.transition_queue:
            return
        
        # Start highest priority transition
        transition = self.transition_queue.pop(0)
        transition.current_state = TransitionState.UNWINDING
        self.active_transitions[transition.transition_id] = transition
        
        logger.info(f"Starting transition: {transition.transition_id}")
        
        # Publish event
        global_event_bus.publish("transition_started", {
            'transition_id': transition.transition_id,
            'state': transition.current_state.value
        })
    
    def _process_unwinding_phase(self, transition: TransitionPlan):
        """Process the position unwinding phase"""
        positions_to_close = [p for p in transition.unwinding_positions 
                             if p.symbol not in transition.completed_positions]
        
        if not positions_to_close:
            # All positions unwound, move to transition phase
            transition.current_state = TransitionState.TRANSITIONING
            transition.progress_percentage = 60.0
            logger.info(f"Transition {transition.transition_id}: Moving to transition phase")
            return
        
        # Close positions gradually
        for position in positions_to_close[:3]:  # Close max 3 at a time
            if self._close_position(position, transition):
                transition.completed_positions.append(position.symbol)
        
        # Update progress
        completed = len(transition.completed_positions)
        total = len(transition.unwinding_positions)
        transition.progress_percentage = (completed / total) * 50.0  # Unwinding is 50% of progress
    
    def _process_transition_phase(self, transition: TransitionPlan):
        """Process the strategy transition phase"""
        try:
            # Stop source strategies
            for strategy_name in transition.source_strategies:
                if self._stop_strategy(strategy_name, transition):
                    logger.info(f"Stopped strategy: {strategy_name}")
            
            # Start target strategies
            for strategy_name in transition.target_strategies:
                if self._start_strategy(strategy_name, transition):
                    logger.info(f"Started strategy: {strategy_name}")
            
            # Move to completion phase
            transition.current_state = TransitionState.COMPLETING
            transition.progress_percentage = 90.0
            
        except Exception as e:
            logger.error(f"Error in transition phase: {e}")
            transition.errors.append(f"Transition error: {str(e)}")
    
    def _process_completion_phase(self, transition: TransitionPlan):
        """Process the completion phase"""
        # Verify all positions are closed and strategies are properly set
        if self._verify_transition_completion(transition):
            transition.progress_percentage = 100.0
            logger.info(f"Transition {transition.transition_id} completed successfully")
            
            # Publish completion event
            global_event_bus.publish("transition_completed", {
                'transition_id': transition.transition_id,
                'duration': (datetime.now() - transition.created_at).total_seconds(),
                'errors': transition.errors
            })
            
            # Move to history
            self.transition_history.append(transition)
            del self.active_transitions[transition.transition_id]
        
        else:
            # Still some work to do
            logger.debug(f"Transition {transition.transition_id} not yet complete")
    
    def _process_emergency_stop(self, transition: TransitionPlan):
        """Process emergency stop for a transition"""
        logger.critical(f"Emergency stop for transition: {transition.transition_id}")
        
        # Force close all remaining positions
        for position in transition.unwinding_positions:
            if position.symbol not in transition.completed_positions:
                self._emergency_close_position(position)
        
        # Emergency stop all strategies
        for strategy_name in transition.source_strategies:
            self._emergency_stop_strategy(strategy_name)
        
        # Mark as completed
        transition.progress_percentage = 100.0
        transition.current_state = TransitionState.IDLE
        
        # Move to history with error flag
        transition.errors.append("Emergency stop executed")
        self.transition_history.append(transition)
        del self.active_transitions[transition.transition_id]
    
    def _close_position(self, position: PositionInfo, transition: TransitionPlan) -> bool:
        """Close a single position"""
        try:
            if self.position_close_callback:
                success = self.position_close_callback(position, transition.risk_constraints)
                if success:
                    logger.info(f"Closed position: {position.symbol} ({position.strategy_name})")
                    return True
            
            logger.warning(f"No position close callback available for {position.symbol}")
            return False
            
        except Exception as e:
            logger.error(f"Error closing position {position.symbol}: {e}")
            return False
    
    def _emergency_close_position(self, position: PositionInfo):
        """Emergency close a position"""
        try:
            if self.position_close_callback:
                emergency_constraints = {'max_slippage': 0.05, 'force_market_order': True}
                self.position_close_callback(position, emergency_constraints)
                logger.warning(f"Emergency closed position: {position.symbol}")
        except Exception as e:
            logger.error(f"Error emergency closing position {position.symbol}: {e}")
    
    def _stop_strategy(self, strategy_name: str, transition: TransitionPlan) -> bool:
        """Stop a strategy"""
        try:
            if self.strategy_stop_callback:
                return self.strategy_stop_callback(strategy_name, transition)
            return True
        except Exception as e:
            logger.error(f"Error stopping strategy {strategy_name}: {e}")
            return False
    
    def _start_strategy(self, strategy_name: str, transition: TransitionPlan) -> bool:
        """Start a strategy"""
        try:
            if self.strategy_start_callback:
                capital = transition.capital_to_transfer / len(transition.target_strategies)
                return self.strategy_start_callback(strategy_name, capital, transition)
            return True
        except Exception as e:
            logger.error(f"Error starting strategy {strategy_name}: {e}")
            return False
    
    def _emergency_stop_strategy(self, strategy_name: str):
        """Emergency stop a strategy"""
        try:
            if strategy_name in self.active_strategies:
                strategy = self.active_strategies[strategy_name]
                if hasattr(strategy, 'emergency_stop'):
                    strategy.emergency_stop()
                logger.warning(f"Emergency stopped strategy: {strategy_name}")
        except Exception as e:
            logger.error(f"Error emergency stopping strategy {strategy_name}: {e}")
    
    def _verify_transition_completion(self, transition: TransitionPlan) -> bool:
        """Verify that a transition has been completed successfully"""
        # Check all positions are closed
        for position in transition.unwinding_positions:
            if position.symbol not in transition.completed_positions:
                return False
        
        # Check source strategies are stopped
        for strategy_name in transition.source_strategies:
            if strategy_name in self.active_strategies:
                return False
        
        # Check target strategies are active
        for strategy_name in transition.target_strategies:
            if strategy_name not in self.active_strategies:
                return False
        
        return True
    
    def _cleanup_completed_transitions(self):
        """Clean up old completed transitions"""
        # Keep only last 10 transitions in history
        if len(self.transition_history) > 10:
            self.transition_history = self.transition_history[-10:]
    
    # Event Handlers
    def _on_position_update(self, data: Dict[str, Any]):
        """Handle position updates"""
        symbol = data.get('symbol')
        strategy_name = data.get('strategy_name')
        
        if symbol and strategy_name:
            # Update position tracker
            position_info = PositionInfo(
                symbol=symbol,
                size=data.get('quantity', 0),
                entry_price=data.get('avg_price', 0),
                current_price=data.get('current_price', data.get('avg_price', 0)),
                unrealized_pnl=data.get('unrealized_pnl', 0),
                side=data.get('side', 'long'),
                strategy_name=strategy_name
            )
            
            with self._lock:
                if strategy_name not in self.position_tracker:
                    self.position_tracker[strategy_name] = []
                
                # Update or add position
                for i, pos in enumerate(self.position_tracker[strategy_name]):
                    if pos.symbol == symbol:
                        self.position_tracker[strategy_name][i] = position_info
                        break
                else:
                    self.position_tracker[strategy_name].append(position_info)
    
    def _on_strategy_stopped(self, data: Dict[str, Any]):
        """Handle strategy stopped events"""
        strategy_name = data.get('strategy_name')
        if strategy_name:
            self.unregister_strategy(strategy_name)
    
    def _on_risk_limit_breached(self, data: Dict[str, Any]):
        """Handle risk limit breach - pause all transitions"""
        logger.critical("Risk limit breached - pausing all transitions")
        
        with self._lock:
            for transition in self.active_transitions.values():
                if transition.current_state != TransitionState.EMERGENCY_STOP:
                    transition.current_state = TransitionState.EMERGENCY_STOP
    
    def _on_emergency_stop(self, data: Dict[str, Any]):
        """Handle emergency stop events"""
        logger.critical("Emergency stop triggered - stopping all transitions")
        
        with self._lock:
            for transition in self.active_transitions.values():
                transition.current_state = TransitionState.EMERGENCY_STOP
    
    # Public API
    def get_transition_status(self, transition_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific transition"""
        with self._lock:
            if transition_id in self.active_transitions:
                transition = self.active_transitions[transition_id]
                return {
                    'transition_id': transition_id,
                    'state': transition.current_state.value,
                    'progress': transition.progress_percentage,
                    'estimated_duration': transition.estimated_duration.total_seconds(),
                    'elapsed_time': (datetime.now() - transition.created_at).total_seconds(),
                    'positions_completed': len(transition.completed_positions),
                    'total_positions': len(transition.unwinding_positions),
                    'errors': transition.errors
                }
        
        return None
    
    def get_all_transitions(self) -> Dict[str, Any]:
        """Get status of all transitions"""
        with self._lock:
            return {
                'active': {tid: self.get_transition_status(tid) for tid in self.active_transitions.keys()},
                'queued': len(self.transition_queue),
                'completed_today': len([t for t in self.transition_history 
                                      if t.created_at.date() == datetime.now().date()])
            }
    
    def cancel_transition(self, transition_id: str, emergency: bool = False) -> bool:
        """Cancel a transition"""
        with self._lock:
            if transition_id in self.active_transitions:
                transition = self.active_transitions[transition_id]
                if emergency:
                    transition.current_state = TransitionState.EMERGENCY_STOP
                else:
                    # Move back to queue or cancel based on state
                    if transition.current_state == TransitionState.PLANNING:
                        del self.active_transitions[transition_id]
                        return True
                
                logger.info(f"Transition cancelled: {transition_id}")
                return True
            
            # Check if in queue
            for i, transition in enumerate(self.transition_queue):
                if transition.transition_id == transition_id:
                    del self.transition_queue[i]
                    logger.info(f"Queued transition cancelled: {transition_id}")
                    return True
        
        return False
    
    def set_callbacks(self,
                     position_close_callback: Optional[Callable] = None,
                     strategy_stop_callback: Optional[Callable] = None,
                     strategy_start_callback: Optional[Callable] = None):
        """Set callback functions for transition operations"""
        self.position_close_callback = position_close_callback
        self.strategy_stop_callback = strategy_stop_callback
        self.strategy_start_callback = strategy_start_callback
        
        logger.info("Transition callbacks configured")

# Global instance
_transition_manager_instance = None

def get_transition_manager(settings: Optional[Dict[str, Any]] = None) -> StrategyTransitionManager:
    """Get global transition manager instance"""
    global _transition_manager_instance
    if _transition_manager_instance is None:
        if settings is None:
            settings = {}
        _transition_manager_instance = StrategyTransitionManager(settings)
    return _transition_manager_instance