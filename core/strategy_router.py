"""
Strategy Router - Dynamische Marktlogik mit Kapitalgewichtung
Verwaltet die Allokation von Kapital auf verschiedene Strategien basierend auf Marktphasen
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
from pathlib import Path
import asyncio
from datetime import datetime, timedelta

from config.settings import Settings
from strategies.strategy_base import Strategy
from strategies import STRATEGIES
from utils.notifier import NotificationManager
from utils.exceptions import ConfigurationError, StrategyError
from core.strategy_transition_manager import get_transition_manager, TransitionPriority
from core.capital_allocation_tracker import get_allocation_tracker

logger = logging.getLogger(__name__)


class MarketPhase(Enum):
    """Enum für verschiedene Marktphasen"""
    SIDEWAYS = "sideways"
    BULL = "bull"
    VOLATILE = "volatile"
    BEAR = "bear"
    EXTREME_FEAR = "extreme_fear"


@dataclass
class StrategyAllocation:
    """Datenklasse für Strategie-Allokation"""
    strategy_name: str
    weight: float
    capital_amount: float
    max_allocation: float
    min_allocation: float
    is_active: bool = True


@dataclass
class MarketCondition:
    """Datenklasse für Marktbedingungen"""
    phase: MarketPhase
    confidence: float
    volatility: float
    trend_strength: float
    fear_greed_index: int
    timestamp: datetime


class StrategyRouter:
    """
    Hauptklasse für die dynamische Strategieverteilung
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.notification_manager = NotificationManager(self.settings)
        
        # Lade Konfiguration
        self.config_path = Path("config/weight_profiles.json")
        
        # Standard-Gewichtungsmatrix
        self.default_weights = {
            MarketPhase.SIDEWAYS: {
                "arbitrage": 0.60,
                "grid_trading": 0.30,
                "defi_farming": 0.10
            },
            MarketPhase.BULL: {
                "momentum": 0.70,
                "grid_trading": 0.20,
                "copy_trading": 0.10
            },
            MarketPhase.VOLATILE: {
                "arbitrage": 0.50,
                "grid_trading": 0.30,
                "mean_reversion": 0.20
            },
            MarketPhase.BEAR: {
                "arbitrage": 0.80,
                "defi_farming": 0.20
            },
            MarketPhase.EXTREME_FEAR: {
                "stablecoin_parking": 1.00
            }
        }
        
        # Lade Gewichtungsprofile
        self.weight_profiles = self._load_weight_profiles()
        
        # Legacy Support - Alte Variablen beibehalten
        self.active_strategies: Dict[str, Strategy] = {}
        self.current_market_regime: str = "unknown"
        self.last_strategy_change_time: Optional[datetime] = None
        
        # Neue Variablen für erweiterte Funktionalität
        self.current_condition: Optional[MarketCondition] = None
        self.current_allocations: Dict[str, StrategyAllocation] = {}
        self.last_rebalance = datetime.now()
        self.rebalance_interval = timedelta(minutes=30)
        self.min_rebalance_threshold = 0.05  # 5% Mindeständerung
        
        # MarketAnalyzer Integration
        self.market_analyzer = None
        
        # Smooth Transition Integration
        self.transition_manager = get_transition_manager(settings)
        self.enable_smooth_transitions = settings.get('strategy_transitions.enabled', True)
        self._setup_transition_callbacks()
        
        # Capital Allocation Integration
        self.allocation_tracker = get_allocation_tracker(settings)
        self.enable_allocation_tracking = settings.get('capital_allocation.enabled', True)
        self._setup_allocation_callbacks()

        logger.info("Enhanced StrategyRouter initialized with dynamic market logic, smooth transitions, and capital tracking.")

    def _setup_transition_callbacks(self):
        """Setup callbacks for smooth strategy transitions"""
        def position_close_callback(position, risk_constraints):
            """Callback to close a position during transition"""
            try:
                # This would integrate with your order manager
                logger.info(f"Closing position: {position.symbol} ({position.size}) for strategy {position.strategy_name}")
                
                # Here you would call your actual position closing logic
                # Example: self.order_manager.close_position(position.symbol, position.size)
                
                return True
            except Exception as e:
                logger.error(f"Error closing position {position.symbol}: {e}")
                return False
        
        def strategy_stop_callback(strategy_name, transition):
            """Callback to stop a strategy during transition"""
            try:
                if strategy_name in self.active_strategies:
                    strategy = self.active_strategies[strategy_name]
                    if hasattr(strategy, 'stop'):
                        strategy.stop()
                    
                    # Remove from active strategies
                    del self.active_strategies[strategy_name]
                    logger.info(f"Stopped strategy: {strategy_name}")
                    return True
                
                return True  # Strategy not active, consider it "stopped"
                
            except Exception as e:
                logger.error(f"Error stopping strategy {strategy_name}: {e}")
                return False
        
        def strategy_start_callback(strategy_name, capital, transition):
            """Callback to start a strategy during transition"""
            try:
                if strategy_name in STRATEGIES:
                    strategy_class = STRATEGIES[strategy_name]
                    new_strategy = strategy_class(self.settings)
                    
                    # Set allocated capital
                    if hasattr(new_strategy, 'set_allocated_capital'):
                        new_strategy.set_allocated_capital(capital)
                    
                    # Start the strategy
                    if hasattr(new_strategy, 'start'):
                        new_strategy.start()
                    
                    # Add to active strategies
                    self.active_strategies[strategy_name] = new_strategy
                    logger.info(f"Started strategy: {strategy_name} with capital: {capital}")
                    return True
                
                else:
                    logger.error(f"Strategy {strategy_name} not found in STRATEGIES")
                    return False
                    
            except Exception as e:
                logger.error(f"Error starting strategy {strategy_name}: {e}")
                return False
        
        # Set callbacks in transition manager
        self.transition_manager.set_callbacks(
            position_close_callback=position_close_callback,
            strategy_stop_callback=strategy_stop_callback,
            strategy_start_callback=strategy_start_callback
        )
        
        # Start the transition manager
        self.transition_manager.start_transition_manager()

    def _setup_allocation_callbacks(self):
        """Setup callbacks for capital allocation tracking"""
        def allocation_callback(strategy_name, amount, allocation):
            """Callback when capital is allocated to a strategy"""
            self.logger.info(f"Capital allocation callback: {strategy_name} allocated ${amount:.2f}")
            
            # Update strategy with new allocation if it exists
            if strategy_name in self.active_strategies:
                strategy = self.active_strategies[strategy_name]
                if hasattr(strategy, 'set_allocated_capital'):
                    strategy.set_allocated_capital(amount)
                    
        def rebalance_callback(reasons):
            """Callback when rebalancing is needed"""
            self.logger.info(f"Rebalance needed: {', '.join(reasons)}")
            
            # Trigger rebalance if not already in progress
            self.notification_manager.send_alert(
                f"Capital rebalance recommended: {', '.join(reasons)}",
                level="WARNING"
            )
        
        # Set callbacks in allocation tracker
        if self.enable_allocation_tracking:
            self.allocation_tracker.add_allocation_callback(allocation_callback)
            self.allocation_tracker.add_rebalance_callback(rebalance_callback)
            
            # Start allocation tracking
            self.allocation_tracker.start_tracking()
            self.logger.info("Capital allocation tracking started")

    def _load_weight_profiles(self) -> Dict:
        """Lädt Gewichtungsprofile aus JSON-Datei"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    profiles = json.load(f)
                self.logger.info(f"Weight profiles loaded from {self.config_path}")
                return profiles
            else:
                self.logger.warning(f"Config file not found: {self.config_path}")
                return {"default": self.default_weights}
        except Exception as e:
            self.logger.error(f"Error loading weight profiles: {e}")
            return {"default": self.default_weights}
    
    def _save_weight_profiles(self) -> None:
        """Speichert Gewichtungsprofile in JSON-Datei"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.weight_profiles, f, indent=2)
            self.logger.info(f"Weight profiles saved to {self.config_path}")
        except Exception as e:
            self.logger.error(f"Error saving weight profiles: {e}")
    
    def _determine_market_phase(self, market_data: Dict) -> MarketPhase:
        """Bestimmt die Marktphase basierend auf Marktdaten"""
        fear_greed = market_data.get('fear_greed_index', 50)
        volatility = market_data.get('volatility', 0.0)
        trend_strength = market_data.get('trend_strength', 0.0)
        price_change_24h = market_data.get('price_change_24h', 0.0)
        
        # Extreme Fear - Panik im Markt
        if fear_greed <= 20:
            return MarketPhase.EXTREME_FEAR
        
        # Volatile Phase - Hohe Volatilität
        if volatility > 0.4:
            return MarketPhase.VOLATILE
        
        # Bullmarkt - Starker Aufwärtstrend
        if trend_strength > 0.6 and price_change_24h > 0.05:
            return MarketPhase.BULL
        
        # Bärenmarkt - Starker Abwärtstrend
        if trend_strength > 0.6 and price_change_24h < -0.05:
            return MarketPhase.BEAR
        
        # Seitwärts - Standardfall
        return MarketPhase.SIDEWAYS
    
    def get_strategy_weights(self, phase: MarketPhase, profile: str = "default") -> Dict[str, float]:
        """Gibt die Gewichtungen für eine bestimmte Marktphase zurück"""
        try:
            profile_weights = self.weight_profiles.get(profile, self.default_weights)
            
            if isinstance(profile_weights, dict) and phase.value in profile_weights:
                return profile_weights[phase.value]
            elif isinstance(profile_weights, dict) and phase in profile_weights:
                return profile_weights[phase]
            else:
                return self.default_weights.get(phase, {})
        except Exception as e:
            self.logger.error(f"Error getting strategy weights: {e}")
            return self.default_weights.get(phase, {})
    
    def calculate_capital_allocation(self, total_capital: float, phase: MarketPhase) -> Dict[str, StrategyAllocation]:
        """Berechnet die Kapitalallokation basierend auf Marktphase"""
        try:
            weights = self.get_strategy_weights(phase)
            allocations = {}
            
            for strategy_name, weight in weights.items():
                capital_amount = total_capital * weight
                
                allocation = StrategyAllocation(
                    strategy_name=strategy_name,
                    weight=weight,
                    capital_amount=capital_amount,
                    max_allocation=capital_amount * 1.2,
                    min_allocation=capital_amount * 0.8,
                    is_active=capital_amount > 0
                )
                
                allocations[strategy_name] = allocation
            
            self.logger.info(f"Capital allocation calculated for {phase.value}: {len(allocations)} strategies")
            return allocations
        except Exception as e:
            self.logger.error(f"Error calculating capital allocation: {e}")
            return {}
    
    def update_weight_profile(self, profile_name: str, phase: MarketPhase, weights: Dict[str, float]) -> bool:
        """Aktualisiert ein Gewichtungsprofil"""
        try:
            # Validiere Gewichtungen
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.01:
                raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
            
            # Aktualisiere Profil
            if profile_name not in self.weight_profiles:
                self.weight_profiles[profile_name] = {}
            
            self.weight_profiles[profile_name][phase.value] = weights
            
            # Speichere Änderungen
            self._save_weight_profiles()
            
            self.logger.info(f"Updated weight profile '{profile_name}' for phase '{phase.value}'")
            return True
        except Exception as e:
            self.logger.error(f"Error updating weight profile: {e}")
            return False
    
    def get_strategy_status(self) -> Dict:
        """Gibt den aktuellen Status aller Strategien zurück"""
        try:
            status = {
                "current_phase": self.current_condition.phase.value if self.current_condition else "unknown",
                "last_update": self.current_condition.timestamp.isoformat() if self.current_condition else None,
                "last_rebalance": self.last_rebalance.isoformat(),
                "next_rebalance": (self.last_rebalance + self.rebalance_interval).isoformat(),
                "active_strategies": len(self.current_allocations),
                "allocations": {}
            }
            
            for strategy_name, allocation in self.current_allocations.items():
                status["allocations"][strategy_name] = {
                    "weight": allocation.weight,
                    "capital": allocation.capital_amount,
                    "active": allocation.is_active
                }
            
            return status
        except Exception as e:
            self.logger.error(f"Error getting strategy status: {e}")
            return {"error": str(e)}

    async def analyze_market_phase(self) -> MarketCondition:
        """Analysiert die aktuelle Marktphase mit MarketAnalyzer"""
        try:
            if self.market_analyzer:
                # Verwende MarketAnalyzer für detaillierte Analyse
                market_data = await self.market_analyzer.get_market_metrics()
                phase = self._determine_market_phase(market_data)
                
                condition = MarketCondition(
                    phase=phase,
                    confidence=market_data.get('confidence', 0.5),
                    volatility=market_data.get('volatility', 0.0),
                    trend_strength=market_data.get('trend_strength', 0.0),
                    fear_greed_index=market_data.get('fear_greed_index', 50),
                    timestamp=datetime.now()
                )
                
                self.current_condition = condition
                self.logger.info(f"Market phase detected: {phase.value} (confidence: {condition.confidence:.2f})")
                
                return condition
            else:
                # Fallback ohne MarketAnalyzer
                return MarketCondition(
                    phase=MarketPhase.SIDEWAYS,
                    confidence=0.3,
                    volatility=0.2,
                    trend_strength=0.3,
                    fear_greed_index=50,
                    timestamp=datetime.now()
                )
        except Exception as e:
            self.logger.error(f"Error analyzing market phase: {e}")
            return MarketCondition(
                phase=MarketPhase.SIDEWAYS,
                confidence=0.3,
                volatility=0.0,
                trend_strength=0.0,
                fear_greed_index=50,
                timestamp=datetime.now()
            )

    async def rebalance_strategies(self, total_capital: float, force: bool = False) -> Dict[str, StrategyAllocation]:
        """Rebalanciert die Strategien basierend auf aktueller Marktphase mit smooth transitions"""
        try:
            # Prüfe ob Rebalancing nötig ist
            if not force and not self._should_rebalance():
                return self.current_allocations
            
            # Analysiere aktuelle Marktphase
            condition = await self.analyze_market_phase()
            
            # Berechne neue Allokationen
            new_allocations = self.calculate_capital_allocation(total_capital, condition.phase)
            
            # Prüfe ob signifikante Änderungen vorliegen
            if self._has_significant_changes(new_allocations):
                
                if self.enable_smooth_transitions:
                    # Use smooth transitions
                    await self._execute_smooth_transition(new_allocations, condition)
                else:
                    # Use immediate transition (legacy behavior)
                    self._execute_immediate_transition(new_allocations)
                
                self.current_allocations = new_allocations
                self.last_rebalance = datetime.now()
                
                self.logger.info(f"Strategies rebalanced for {condition.phase.value}")
                self._log_allocations(new_allocations)
                
                return new_allocations
            else:
                self.logger.info("No significant changes, keeping current allocations")
                return self.current_allocations
                
        except Exception as e:
            self.logger.error(f"Error rebalancing strategies: {e}")
            return self.current_allocations

    def _should_rebalance(self) -> bool:
        """Prüft ob ein Rebalancing durchgeführt werden sollte"""
        time_since_last = datetime.now() - self.last_rebalance
        return time_since_last >= self.rebalance_interval

    def _has_significant_changes(self, new_allocations: Dict[str, StrategyAllocation]) -> bool:
        """Prüft ob signifikante Änderungen in der Allokation vorliegen"""
        if not self.current_allocations:
            return True
        
        for strategy_name, new_allocation in new_allocations.items():
            current_allocation = self.current_allocations.get(strategy_name)
            
            if not current_allocation:
                return True
            
            weight_change = abs(new_allocation.weight - current_allocation.weight)
            if weight_change > self.min_rebalance_threshold:
                return True
        
        return False
    
    async def _execute_smooth_transition(self, new_allocations: Dict[str, StrategyAllocation], condition: MarketCondition):
        """Execute smooth transition to new strategy allocations"""
        try:
            # Identify strategies to stop and start
            current_strategy_names = set(self.current_allocations.keys()) if self.current_allocations else set()
            new_strategy_names = set(new_allocations.keys())
            
            strategies_to_stop = current_strategy_names - new_strategy_names
            strategies_to_start = new_strategy_names - current_strategy_names
            
            # Determine transition priority based on market conditions
            priority = self._determine_transition_priority(condition)
            
            # Register current strategies with transition manager
            for strategy_name, strategy in self.active_strategies.items():
                self.transition_manager.register_strategy(strategy_name, strategy)
            
            if strategies_to_stop or strategies_to_start:
                # Plan and execute transition
                transition_id = self.transition_manager.plan_transition(
                    source_strategies=list(strategies_to_stop),
                    target_strategies=list(strategies_to_start),
                    priority=priority,
                    force_immediate=(priority == TransitionPriority.EMERGENCY)
                )
                
                self.logger.info(f"Smooth transition planned: {transition_id}")
                self.logger.info(f"Stopping: {strategies_to_stop}")
                self.logger.info(f"Starting: {strategies_to_start}")
                
                # Send notification
                self.notification_manager.send_alert(
                    f"Strategy transition initiated: {transition_id}",
                    level="INFO"
                )
            
            else:
                # Only capital reallocation needed, no strategy changes
                self._reallocate_capital_only(new_allocations)
                
        except Exception as e:
            self.logger.error(f"Error in smooth transition: {e}")
            # Fallback to immediate transition
            self._execute_immediate_transition(new_allocations)
    
    def _determine_transition_priority(self, condition: MarketCondition) -> TransitionPriority:
        """Determine transition priority based on market conditions"""
        # Emergency priority for extreme market conditions
        if condition.phase == MarketPhase.EXTREME_FEAR:
            return TransitionPriority.EMERGENCY
        
        # High priority for bear markets
        if condition.phase == MarketPhase.BEAR:
            return TransitionPriority.HIGH
        
        # High priority for high volatility
        if condition.volatility > 0.5:  # 50% volatility threshold
            return TransitionPriority.HIGH
        
        # Normal priority for regular transitions
        return TransitionPriority.NORMAL
    
    def _execute_immediate_transition(self, new_allocations: Dict[str, StrategyAllocation]):
        """Execute immediate transition (legacy behavior)"""
        try:
            current_strategy_names = set(self.current_allocations.keys()) if self.current_allocations else set()
            new_strategy_names = set(new_allocations.keys())
            
            strategies_to_stop = current_strategy_names - new_strategy_names
            strategies_to_start = new_strategy_names - current_strategy_names
            
            # Stop strategies immediately
            for strategy_name in strategies_to_stop:
                self._deactivate_strategy(strategy_name)
            
            # Start new strategies immediately
            for strategy_name in strategies_to_start:
                allocation = new_allocations[strategy_name]
                self._activate_strategy(strategy_name, allocation.capital_amount)
            
            self.logger.info("Immediate transition completed")
            
        except Exception as e:
            self.logger.error(f"Error in immediate transition: {e}")
    
    def _reallocate_capital_only(self, new_allocations: Dict[str, StrategyAllocation]):
        """Reallocate capital to existing strategies without stopping/starting"""
        try:
            for strategy_name, allocation in new_allocations.items():
                if strategy_name in self.active_strategies:
                    strategy = self.active_strategies[strategy_name]
                    if hasattr(strategy, 'update_capital_allocation'):
                        strategy.update_capital_allocation(allocation.capital_amount)
                        self.logger.info(f"Updated capital for {strategy_name}: {allocation.capital_amount}")
            
            self.logger.info("Capital reallocation completed")
            
        except Exception as e:
            self.logger.error(f"Error in capital reallocation: {e}")
    
    def _activate_strategy(self, strategy_name: str, capital: float):
        """Activate a new strategy with given capital"""
        try:
            if strategy_name in STRATEGIES:
                strategy_class = STRATEGIES[strategy_name]
                new_strategy = strategy_class(self.settings)
                
                # Allocate capital through tracker
                if self.enable_allocation_tracking:
                    allocation_success = self.allocation_tracker.allocate_capital(
                        strategy_name=strategy_name,
                        amount=capital,
                        reason=f"Strategy activation via router"
                    )
                    
                    if not allocation_success:
                        self.logger.error(f"Failed to allocate capital for {strategy_name}")
                        return
                
                if hasattr(new_strategy, 'set_allocated_capital'):
                    new_strategy.set_allocated_capital(capital)
                
                if hasattr(new_strategy, 'start'):
                    new_strategy.start()
                
                self.active_strategies[strategy_name] = new_strategy
                self.logger.info(f"Activated strategy: {strategy_name} with capital: {capital}")
                
        except Exception as e:
            self.logger.error(f"Error activating strategy {strategy_name}: {e}")
    
    def _deactivate_strategy_with_allocation(self, strategy_name: str):
        """Deactivate a strategy and deallocate its capital"""
        try:
            # Deallocate capital through tracker
            if self.enable_allocation_tracking:
                deallocation_success = self.allocation_tracker.deallocate_capital(
                    strategy_name=strategy_name,
                    reason=f"Strategy deactivation via router"
                )
                
                if not deallocation_success:
                    self.logger.warning(f"Failed to deallocate capital for {strategy_name}")
            
            # Use existing deactivation logic
            self._deactivate_strategy(strategy_name)
            
        except Exception as e:
            self.logger.error(f"Error deactivating strategy with allocation {strategy_name}: {e}")

    def _log_allocations(self, allocations: Dict[str, StrategyAllocation]) -> None:
        """Loggt die aktuellen Allokationen"""
        self.logger.info("Current Strategy Allocations:")
        for strategy_name, allocation in allocations.items():
            self.logger.info(f"  {strategy_name}: {allocation.weight:.1%} (${allocation.capital_amount:.2f})")
    
    # Public API for transition management
    def get_transition_status(self) -> Dict[str, Any]:
        """Get current transition status"""
        if self.enable_smooth_transitions:
            return self.transition_manager.get_all_transitions()
        else:
            return {"smooth_transitions": False, "active": {}, "queued": 0}
    
    def get_strategy_status(self) -> Dict[str, Any]:
        """Get current strategy status including transitions"""
        status = {
            "active_strategies": list(self.active_strategies.keys()),
            "current_allocations": self.current_allocations,
            "last_rebalance": self.last_rebalance.isoformat() if self.last_rebalance else None,
            "smooth_transitions_enabled": self.enable_smooth_transitions
        }
        
        if self.enable_smooth_transitions:
            status["transitions"] = self.get_transition_status()
        
        return status
    
    def force_emergency_transition(self, target_strategies: List[str], total_capital: float) -> str:
        """Force an emergency transition to specific strategies"""
        if not self.enable_smooth_transitions:
            raise ValueError("Smooth transitions not enabled")
        
        current_strategies = list(self.active_strategies.keys())
        
        transition_id = self.transition_manager.plan_transition(
            source_strategies=current_strategies,
            target_strategies=target_strategies,
            priority=TransitionPriority.EMERGENCY,
            force_immediate=True
        )
        
        self.logger.critical(f"Emergency transition initiated: {transition_id}")
        self.notification_manager.send_alert(
            f"EMERGENCY: Strategy transition to {target_strategies}",
            level="CRITICAL"
        )
        
        return transition_id
    
    def cancel_transition(self, transition_id: str) -> bool:
        """Cancel a pending or active transition"""
        if self.enable_smooth_transitions:
            return self.transition_manager.cancel_transition(transition_id)
        return False
    
    def shutdown_transitions(self):
        """Shutdown the transition manager"""
        if self.enable_smooth_transitions:
            self.transition_manager.stop_transition_manager()
            self.logger.info("Strategy transition manager shutdown")
    
    # Public API for capital allocation management
    def get_capital_allocation_summary(self) -> Dict[str, Any]:
        """Get comprehensive capital allocation summary"""
        if self.enable_allocation_tracking:
            summary = self.allocation_tracker.get_portfolio_summary()
            details = self.allocation_tracker.get_allocation_details()
            
            return {
                'portfolio_summary': {
                    'total_capital': summary.total_capital,
                    'allocated_capital': summary.allocated_capital,
                    'available_capital': summary.available_capital,
                    'total_pnl': summary.total_pnl,
                    'allocation_efficiency': summary.allocation_efficiency,
                    'strategy_count': summary.strategy_count,
                    'allocation_concentration': summary.allocation_concentration,
                    'best_performing_strategy': summary.best_performing_strategy,
                    'worst_performing_strategy': summary.worst_performing_strategy
                },
                'strategy_allocations': details,
                'last_updated': datetime.now().isoformat()
            }
        else:
            # Fallback to basic allocation info
            return {
                'allocation_tracking_enabled': False,
                'active_strategies': list(self.active_strategies.keys()),
                'current_allocations': self.current_allocations
            }
    
    def allocate_strategy_capital(self, strategy_name: str, amount: float) -> bool:
        """Manually allocate capital to a strategy"""
        if not self.enable_allocation_tracking:
            self.logger.error("Capital allocation tracking not enabled")
            return False
        
        return self.allocation_tracker.allocate_capital(
            strategy_name=strategy_name,
            amount=amount,
            reason="Manual allocation via strategy router"
        )
    
    def deallocate_strategy_capital(self, strategy_name: str, amount: Optional[float] = None) -> bool:
        """Manually deallocate capital from a strategy"""
        if not self.enable_allocation_tracking:
            self.logger.error("Capital allocation tracking not enabled")
            return False
        
        return self.allocation_tracker.deallocate_capital(
            strategy_name=strategy_name,
            amount=amount,
            reason="Manual deallocation via strategy router"
        )
    
    def get_strategy_performance_attribution(self, days: int = 30) -> Dict[str, Any]:
        """Get performance attribution analysis"""
        if self.enable_allocation_tracking:
            return self.allocation_tracker.get_performance_attribution(days)
        else:
            return {}
    
    def export_capital_allocation_report(self, filepath: str) -> bool:
        """Export comprehensive capital allocation report"""
        if not self.enable_allocation_tracking:
            self.logger.error("Capital allocation tracking not enabled")
            return False
        
        try:
            self.allocation_tracker.export_allocation_report(filepath)
            return True
        except Exception as e:
            self.logger.error(f"Error exporting allocation report: {e}")
            return False
    
    def update_total_capital(self, new_total: float) -> bool:
        """Update total available capital"""
        if not self.enable_allocation_tracking:
            self.logger.error("Capital allocation tracking not enabled")
            return False
        
        try:
            current_total = self.allocation_tracker.total_capital
            difference = new_total - current_total
            
            if difference > 0:
                # Capital increase
                global_event_bus.publish("capital_deposit", {'amount': difference})
            elif difference < 0:
                # Capital decrease
                global_event_bus.publish("capital_withdrawal", {'amount': abs(difference)})
            
            self.logger.info(f"Total capital updated: ${current_total:.2f} -> ${new_total:.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating total capital: {e}")
            return False
    
    def shutdown_allocation_tracking(self):
        """Shutdown allocation tracking"""
        if self.enable_allocation_tracking:
            self.allocation_tracker.stop_tracking()
            self.logger.info("Capital allocation tracking shutdown")

    def update_market_regime(self, new_regime: str, current_total_capital: float) -> bool:
        """
        Updates the current market regime and adjusts active strategies accordingly.

        Args:
            new_regime: The newly detected market regime (e.g., "bull", "bear", "sideways").
            current_total_capital: The total capital available to the bot.

        Returns:
            True if strategies were updated, False otherwise.
        """
        if new_regime == self.current_market_regime:
            # logger.debug(f"Market regime remains '{new_regime}'. No strategy change needed.")
            return False

        log_message = f"Market regime changed from '{self.current_market_regime}' to '{new_regime}'."
        logger.info(log_message)
        self.notification_manager.send_alert(f"Market Regime Change: {self.current_market_regime} -> {new_regime}",
                                             level="INFO")

        self.current_market_regime = new_regime
        self.last_strategy_change_time = datetime.now()

        return self._adjust_strategies_for_regime(new_regime, current_total_capital)

    def _adjust_strategies_for_regime(self, regime: str, total_capital: float) -> bool:
        """
        Activates/deactivates strategies and allocates capital based on the given regime.
        """
        target_allocations = self.regime_strategies_config.get(regime, {})

        # Handle special case: "manual_intervention_required" or no strategies for regime
        if regime == "manual_intervention_required" or not target_allocations:
            logger.warning(
                f"Regime '{regime}' requires manual intervention or has no defined strategies. Pausing all active strategies.")
            self._deactivate_all_strategies()
            self.notification_manager.send_alert(
                f"Bot paused: Regime '{regime}' requires manual intervention or no strategies defined.",
                level="WARNING")
            return False

        strategies_to_activate = set(target_allocations.keys())
        currently_active_names = set(self.active_strategies.keys())

        # Deactivate strategies that are no longer needed
        for strategy_name in currently_active_names:
            if strategy_name not in strategies_to_activate:
                self._deactivate_strategy(strategy_name)

        # Activate or re-allocate capital for required strategies
        for strategy_name, allocation_ratio in target_allocations.items():
            allocated_capital = total_capital * allocation_ratio
            if strategy_name not in STRATEGIES:
                logger.error(
                    f"Strategy '{strategy_name}' not found in available strategies (STRATEGIES dict). Skipping.")
                self.notification_manager.send_alert(
                    f"Error: Strategy '{strategy_name}' not found in router config or STRATEGIES dict.", level="ERROR")
                continue

            if strategy_name not in self.active_strategies:
                # Initialize and activate new strategy
                strategy_class = STRATEGIES[strategy_name]
                strategy_params = self.settings.get(f'strategy_configs.{strategy_name}', {})
                new_strategy = strategy_class(strategy_params)
                self.active_strategies[strategy_name] = new_strategy
                logger.info(f"Activated strategy '{strategy_name}' with {allocated_capital:.2f} capital.")
                self.notification_manager.send_alert(
                    f"Strategy '{strategy_name}' activated for regime '{regime}'. Capital: {allocated_capital:.2f}",
                    level="INFO")

                # In a real bot, you'd also call a method like new_strategy.start_trading(allocated_capital)
                # or pass this allocated_capital to the strategy's position sizing logic.
                if hasattr(new_strategy, 'set_allocated_capital'):  # Example hook
                    new_strategy.set_allocated_capital(allocated_capital)
                # You might need to pass exchange/data_manager references to the strategy
                # This depends on how your Strategy base class expects its dependencies.

            else:
                # Update capital allocation for already active strategy
                logger.info(f"Re-allocated {allocated_capital:.2f} capital to active strategy '{strategy_name}'.")
                if hasattr(self.active_strategies[strategy_name], 'update_capital_allocation'):
                    self.active_strategies[strategy_name].update_capital_allocation(allocated_capital)

        logger.info(
            f"Strategies adjusted for regime '{regime}'. Active strategies: {list(self.active_strategies.keys())}")
        return True

    def _deactivate_strategy(self, strategy_name: str):
        """Deactivates a single strategy."""
        if strategy_name in self.active_strategies:
            # For a live bot, this would involve closing positions and stopping strategy threads/tasks
            # self.active_strategies[strategy_name].stop_trading() # Example API call
            logger.info(
                f"Deactivating strategy: {strategy_name}. (Note: Actual position closing not implemented here.)")
            del self.active_strategies[strategy_name]
            logger.info(f"Deactivated strategy: {strategy_name}.")
            self.notification_manager.send_alert(f"Strategy '{strategy_name}' deactivated.", level="INFO")

    def _deactivate_all_strategies(self):
        """Deactivates all currently active strategies."""
        for strategy_name in list(self.active_strategies.keys()):
            self._deactivate_strategy(strategy_name)
        logger.info("All active strategies deactivated.")

    def get_active_strategies(self) -> Dict[str, Strategy]:
        """Returns currently active strategy instances."""
        return self.active_strategies

    def get_current_regime(self) -> str:
        """Returns the last known market regime."""
        return self.current_market_regime