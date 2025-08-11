"""
Quantum Orchestrator - Das zentrale Gehirn des Tier-1 Trading Systems
Elite-Orchestrierung mit sequenzieller, kontextbewusster Entscheidungsfindung
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from .tier1_models import (
    Signal, Allocation, Order, OrchestrationResult, SystemState,
    MarketRegime, SystemConstants, validate_signal, validate_allocation
)
from .tier1_risk_engine import AdvancedRiskEngine
from .tier1_capital_allocator import AdvancedCapitalAllocator
from .tier1_execution_layer import AdvancedExecutionLayer

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStep:
    """Processing Step tracking für Orchestration"""
    step_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    result: Any = None
    error: Optional[str] = None
    
    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0.0


@dataclass
class OrchestrationMetrics:
    """Orchestration Performance Metrics"""
    total_signals_processed: int = 0
    signals_executed: int = 0
    signals_rejected: int = 0
    signals_skipped: int = 0
    signals_aborted: int = 0
    
    avg_processing_time_ms: float = 0.0
    avg_execution_score: float = 0.0
    
    risk_rejection_rate: float = 0.0
    allocation_skip_rate: float = 0.0
    execution_abort_rate: float = 0.0
    
    last_updated: datetime = None


class QuantumOrchestrator:
    """
    Das zentrale Gehirn des Tier-1 Trading Systems
    
    Orchestriert sequenziell und kontextbewusst:
    1. Risk Engine → Signal Approval
    2. Capital Allocator → Intelligent Allocation  
    3. Execution Layer → Optimal Execution
    
    Features:
    - Asynchrone, non-blocking Verarbeitung
    - Detaillierte Performance-Metriken
    - Circuit Breaker Integration
    - Ganzheitliche System-Überwachung
    - Intelligente Fallback-Mechanismen
    """
    
    def __init__(self, 
                 risk_engine: AdvancedRiskEngine,
                 capital_allocator: AdvancedCapitalAllocator,
                 execution_layer: AdvancedExecutionLayer,
                 enable_monitoring: bool = True):
        
        self.risk_engine = risk_engine
        self.capital_allocator = capital_allocator
        self.execution_layer = execution_layer
        self.enable_monitoring = enable_monitoring
        
        # Performance Tracking
        self.orchestration_metrics = OrchestrationMetrics()
        self.processing_history: List[OrchestrationResult] = []
        self.active_signals: Dict[str, Signal] = {}
        
        # System State Monitoring
        self.system_state = SystemState(
            total_capital=capital_allocator.total_capital,
            allocated_capital=0.0,
            available_capital=capital_allocator.total_capital,
            active_positions=0,
            active_orders=0,
            current_regime=MarketRegime.MEAN_REVERTING,
            portfolio_risk=None,
            daily_pnl=0.0,
            total_pnl=0.0,
            sharpe_ratio=0.0
        )
        
        # Circuit Breakers für Orchestrator
        self.circuit_breakers = {
            'high_rejection_rate': False,    # > 80% rejection rate
            'system_overload': False,        # Too many concurrent signals
            'execution_failures': False,     # Too many execution failures
            'risk_engine_failure': False     # Risk engine not responding
        }
        
        # Processing Configuration
        self.max_concurrent_signals = 10
        self.processing_timeout_seconds = 30
        self.min_execution_score_threshold = 0.7
        
        # Monitoring
        self.last_health_check = datetime.now()
        self.health_check_interval = timedelta(minutes=5)
        
        logger.info("QuantumOrchestrator initialisiert mit Elite-Tier-1-Konfiguration")
    
    async def on_new_signal(self, signal: Signal) -> OrchestrationResult:
        """
        HAUPTMETHODE: Zentrale Signal-Verarbeitung
        Exakte Implementierung wie spezifiziert mit Tier-1-Erweiterungen
        """
        
        start_time = datetime.now()
        processing_steps: List[ProcessingStep] = []
        
        try:
            # 0. Signal-Validierung und Vorbereitung
            validation_step = ProcessingStep("signal_validation", start_time)
            
            if not validate_signal(signal):
                validation_step.end_time = datetime.now()
                validation_step.error = "Signal validation failed"
                processing_steps.append(validation_step)
                
                return await self._create_result("rejected", signal.signal_id, 
                                                "invalid_signal", processing_steps)
            
            validation_step.end_time = datetime.now()
            validation_step.success = True
            processing_steps.append(validation_step)
            
            # Circuit Breaker Check
            if await self._check_circuit_breakers():
                return await self._create_result("rejected", signal.signal_id, 
                                                "circuit_breaker_active", processing_steps)
            
            # Schritt 1: Risikoprüfung durch die Risk Engine
            risk_step = ProcessingStep("risk_approval", datetime.now())
            
            try:
                risk_approved = await asyncio.wait_for(
                    self.risk_engine.approve(signal), 
                    timeout=self.processing_timeout_seconds
                )
                
                risk_step.end_time = datetime.now()
                risk_step.success = risk_approved
                risk_step.result = risk_approved
                processing_steps.append(risk_step)
                
                if not risk_approved:
                    logger.info(f"SIGNAL REJECTED: {signal.stream_id} von Risk Engine blockiert.")
                    await self._update_metrics("rejected")
                    return await self._create_result("rejected", signal.signal_id, 
                                                    "risk_blocked", processing_steps)
                
            except asyncio.TimeoutError:
                risk_step.end_time = datetime.now()
                risk_step.error = "Risk engine timeout"
                processing_steps.append(risk_step)
                
                logger.error(f"Risk Engine Timeout für Signal {signal.signal_id}")
                return await self._create_result("rejected", signal.signal_id, 
                                                "risk_engine_timeout", processing_steps)
            
            # Schritt 2: Kapitalzuteilung durch den Allocator
            allocation_step = ProcessingStep("capital_allocation", datetime.now())
            
            try:
                allocation = await asyncio.wait_for(
                    self.capital_allocator.allocate(signal),
                    timeout=self.processing_timeout_seconds
                )
                
                allocation_step.end_time = datetime.now()
                allocation_step.success = allocation is not None
                allocation_step.result = allocation
                processing_steps.append(allocation_step)
                
                if allocation is None or allocation.amount == 0:
                    logger.info(f"SIGNAL SKIPPED: {signal.stream_id} erhielt keine Kapitalzuteilung.")
                    await self._update_metrics("skipped")
                    return await self._create_result("skipped", signal.signal_id, 
                                                    "no_capital", processing_steps)
                
                # Allocation Validierung
                if not validate_allocation(allocation):
                    logger.warning(f"Invalid allocation für Signal {signal.signal_id}")
                    return await self._create_result("skipped", signal.signal_id, 
                                                    "invalid_allocation", processing_steps)
                
            except asyncio.TimeoutError:
                allocation_step.end_time = datetime.now()
                allocation_step.error = "Capital allocator timeout"
                processing_steps.append(allocation_step)
                
                logger.error(f"Capital Allocator Timeout für Signal {signal.signal_id}")
                return await self._create_result("skipped", signal.signal_id, 
                                                "allocator_timeout", processing_steps)
            
            # Schritt 3: Execution-Scoring durch den Execution Layer
            scoring_step = ProcessingStep("execution_scoring", datetime.now())
            
            try:
                execution_score = await asyncio.wait_for(
                    self.execution_layer.score(signal, allocation),
                    timeout=self.processing_timeout_seconds
                )
                
                scoring_step.end_time = datetime.now()
                scoring_step.success = True
                scoring_step.result = execution_score
                processing_steps.append(scoring_step)
                
                # Dynamischer Threshold basierend auf Market Regime
                current_regime = await self.risk_engine.get_current_regime()
                dynamic_threshold = await self._get_dynamic_execution_threshold(current_regime)
                
                if execution_score < dynamic_threshold:
                    logger.info(f"SIGNAL ABORTED: {signal.stream_id} mit niedrigem Execution Score "
                              f"({execution_score:.3f} < {dynamic_threshold:.3f}).")
                    await self._update_metrics("aborted")
                    return await self._create_result("aborted", signal.signal_id, 
                                                    "low_execution_score", processing_steps,
                                                    execution_score=execution_score)
                
            except asyncio.TimeoutError:
                scoring_step.end_time = datetime.now()
                scoring_step.error = "Execution scoring timeout"
                processing_steps.append(scoring_step)
                
                logger.error(f"Execution Scoring Timeout für Signal {signal.signal_id}")
                return await self._create_result("aborted", signal.signal_id, 
                                                "execution_timeout", processing_steps)
            
            # Schritt 4: Order an den Execution Layer weitergeben
            execution_step = ProcessingStep("order_execution", datetime.now())
            
            try:
                logger.info(f"SIGNAL ACCEPTED: Order für {signal.stream_id} wird platziert.")
                
                order = await asyncio.wait_for(
                    self.execution_layer.place(signal, allocation),
                    timeout=self.processing_timeout_seconds
                )
                
                execution_step.end_time = datetime.now()
                execution_step.success = order.status != "rejected"
                execution_step.result = order
                processing_steps.append(execution_step)
                
                if order.status == "rejected":
                    logger.warning(f"Order rejected für Signal {signal.signal_id}")
                    await self._update_metrics("aborted")
                    return await self._create_result("aborted", signal.signal_id, 
                                                    "order_rejected", processing_steps)
                
                # Erfolgreiche Execution
                await self._update_metrics("executed", execution_score)
                await self._track_active_signal(signal, allocation, order)
                
                # System State aktualisieren
                await self._update_system_state()
                
                execution_result = await self._create_result(
                    "executed", signal.signal_id, None, processing_steps,
                    order_id=order.order_id,
                    capital_allocated=allocation.amount,
                    execution_score=execution_score
                )
                
                logger.info(f"✅ Signal {signal.signal_id} erfolgreich verarbeitet: "
                          f"${allocation.amount:,.0f} allocated, score={execution_score:.3f}")
                
                return execution_result
                
            except asyncio.TimeoutError:
                execution_step.end_time = datetime.now()
                execution_step.error = "Order execution timeout"
                processing_steps.append(execution_step)
                
                logger.error(f"Order Execution Timeout für Signal {signal.signal_id}")
                return await self._create_result("aborted", signal.signal_id, 
                                                "execution_timeout", processing_steps)
        
        except Exception as e:
            logger.error(f"Unerwarteter Fehler in Orchestration für Signal {signal.signal_id}: {e}")
            
            error_step = ProcessingStep("orchestration_error", datetime.now())
            error_step.end_time = datetime.now()
            error_step.error = str(e)
            processing_steps.append(error_step)
            
            return await self._create_result("rejected", signal.signal_id, 
                                            f"orchestration_error: {str(e)}", processing_steps)
    
    async def _create_result(self, status: str, signal_id: str, reason: Optional[str] = None,
                           processing_steps: List[ProcessingStep] = None,
                           order_id: Optional[str] = None, 
                           capital_allocated: Optional[float] = None,
                           execution_score: Optional[float] = None) -> OrchestrationResult:
        """Creates standardized orchestration result"""
        
        total_processing_time = 0.0
        if processing_steps:
            total_processing_time = sum(step.duration_ms for step in processing_steps)
        
        result = OrchestrationResult(
            signal_id=signal_id,
            status=status,
            reason=reason,
            order_id=order_id,
            capital_allocated=capital_allocated,
            execution_score=execution_score,
            risk_approved=any(step.step_name == "risk_approval" and step.success 
                            for step in processing_steps or []),
            allocation_amount=capital_allocated or 0.0,
            processing_time_ms=total_processing_time
        )
        
        # Add to processing history
        self.processing_history.append(result)
        
        # Keep only last 1000 results
        if len(self.processing_history) > 1000:
            self.processing_history = self.processing_history[-1000:]
        
        return result
    
    async def _check_circuit_breakers(self) -> bool:
        """Checks if any circuit breakers are active"""
        
        active_breakers = [name for name, active in self.circuit_breakers.items() if active]
        
        if active_breakers:
            logger.warning(f"Circuit breakers active: {active_breakers}")
            return True
        
        # Auto-check für High Rejection Rate
        if len(self.processing_history) >= 20:
            recent_results = self.processing_history[-20:]
            rejection_rate = sum(1 for r in recent_results if r.status == "rejected") / len(recent_results)
            
            if rejection_rate > 0.8:  # > 80% rejection rate
                self.circuit_breakers['high_rejection_rate'] = True
                logger.error(f"High rejection rate detected: {rejection_rate:.1%}")
                return True
        
        return False
    
    async def _get_dynamic_execution_threshold(self, regime: MarketRegime) -> float:
        """Dynamic execution threshold based on market regime"""
        
        base_threshold = self.min_execution_score_threshold
        
        regime_adjustments = {
            MarketRegime.CRISIS: 0.9,           # Very high threshold in crisis
            MarketRegime.HIGH_VOLATILITY: 0.8,  # High threshold in volatile markets
            MarketRegime.BULL_TRENDING: 0.6,    # Lower threshold in good conditions
            MarketRegime.LOW_VOLATILITY: 0.65,  # Slightly higher in low vol
            MarketRegime.MEAN_REVERTING: 0.7,   # Standard threshold
        }
        
        return regime_adjustments.get(regime, base_threshold)
    
    async def _update_metrics(self, status: str, execution_score: float = None) -> None:
        """Updates orchestration metrics"""
        
        metrics = self.orchestration_metrics
        metrics.total_signals_processed += 1
        
        if status == "executed":
            metrics.signals_executed += 1
            if execution_score:
                # EWMA update
                alpha = 0.1
                metrics.avg_execution_score = (
                    (1 - alpha) * metrics.avg_execution_score + alpha * execution_score
                )
        elif status == "rejected":
            metrics.signals_rejected += 1
        elif status == "skipped":
            metrics.signals_skipped += 1
        elif status == "aborted":
            metrics.signals_aborted += 1
        
        # Calculate rates
        total = metrics.total_signals_processed
        if total > 0:
            metrics.risk_rejection_rate = metrics.signals_rejected / total
            metrics.allocation_skip_rate = metrics.signals_skipped / total
            metrics.execution_abort_rate = metrics.signals_aborted / total
        
        metrics.last_updated = datetime.now()
    
    async def _track_active_signal(self, signal: Signal, allocation: Allocation, order: Order) -> None:
        """Tracks active signal for monitoring"""
        
        self.active_signals[signal.signal_id] = signal
        
        # Add position to risk engine
        self.risk_engine.add_position(
            signal.signal_id, 
            signal.asset, 
            signal.origin, 
            allocation.amount
        )
    
    async def _update_system_state(self) -> None:
        """Updates comprehensive system state"""
        
        # Get allocation status
        allocation_status = self.capital_allocator.get_allocation_status()
        
        # Get risk status
        risk_status = self.risk_engine.get_risk_status()
        
        # Get execution status
        execution_status = self.execution_layer.get_execution_status()
        
        # Update system state
        self.system_state.total_capital = allocation_status['total_capital']
        self.system_state.allocated_capital = allocation_status['allocated_capital']
        self.system_state.available_capital = allocation_status['available_capital']
        self.system_state.active_positions = risk_status['active_positions']
        self.system_state.active_orders = execution_status['active_orders']
        self.system_state.current_regime = self.risk_engine.current_regime
        
        # Calculate portfolio risk if enough data
        try:
            portfolio_risk = await self.risk_engine.calculate_portfolio_risk()
            self.system_state.portfolio_risk = portfolio_risk
        except Exception as e:
            logger.debug(f"Could not calculate portfolio risk: {e}")
        
        self.system_state.last_updated = datetime.now()
    
    async def process_multiple_signals(self, signals: List[Signal]) -> List[OrchestrationResult]:
        """Processes multiple signals concurrently with throttling"""
        
        if len(signals) > self.max_concurrent_signals:
            logger.warning(f"Too many signals ({len(signals)}), processing in batches")
            
            # Process in batches
            results = []
            for i in range(0, len(signals), self.max_concurrent_signals):
                batch = signals[i:i + self.max_concurrent_signals]
                batch_results = await asyncio.gather(
                    *[self.on_new_signal(signal) for signal in batch],
                    return_exceptions=True
                )
                results.extend(batch_results)
            
            return results
        else:
            # Process all concurrently
            return await asyncio.gather(
                *[self.on_new_signal(signal) for signal in signals],
                return_exceptions=True
            )
    
    async def perform_health_check(self) -> Dict[str, Any]:
        """Comprehensive system health check"""
        
        health_status = {
            'orchestrator': 'healthy',
            'risk_engine': 'healthy',
            'capital_allocator': 'healthy',
            'execution_layer': 'healthy',
            'overall_status': 'healthy',
            'timestamp': datetime.now(),
            'issues': []
        }
        
        try:
            # Test risk engine
            test_signal = Signal(
                stream_id="health_check",
                asset="BTC",
                direction="long",
                confidence=0.5,
                timestamp=datetime.now(),
                expected_profit_pts=10,
                expected_duration_min=60,
                source_metadata={},
                origin="health_check"
            )
            
            risk_response = await asyncio.wait_for(
                self.risk_engine.approve(test_signal), timeout=5
            )
            
        except Exception as e:
            health_status['risk_engine'] = 'unhealthy'
            health_status['issues'].append(f"Risk engine error: {str(e)}")
        
        # Check processing metrics
        if self.orchestration_metrics.total_signals_processed > 10:
            if self.orchestration_metrics.risk_rejection_rate > 0.9:
                health_status['issues'].append("Very high rejection rate")
            
            if self.orchestration_metrics.avg_execution_score < 0.5:
                health_status['issues'].append("Low execution scores")
        
        # Check circuit breakers
        active_breakers = [name for name, active in self.circuit_breakers.items() if active]
        if active_breakers:
            health_status['issues'].append(f"Active circuit breakers: {active_breakers}")
        
        # Overall status
        if health_status['issues']:
            health_status['overall_status'] = 'degraded' if len(health_status['issues']) <= 2 else 'unhealthy'
        
        self.last_health_check = datetime.now()
        return health_status
    
    def get_orchestration_metrics(self) -> OrchestrationMetrics:
        """Returns current orchestration metrics"""
        return self.orchestration_metrics
    
    def get_system_state(self) -> SystemState:
        """Returns current system state"""
        return self.system_state
    
    def set_circuit_breaker(self, breaker_name: str, active: bool) -> None:
        """Manually set circuit breaker state"""
        if breaker_name in self.circuit_breakers:
            self.circuit_breakers[breaker_name] = active
            logger.warning(f"Circuit breaker '{breaker_name}' set to {active}")
    
    def get_processing_history(self, limit: int = 100) -> List[OrchestrationResult]:
        """Returns recent processing history"""
        return self.processing_history[-limit:]
    
    def get_active_signals(self) -> Dict[str, Signal]:
        """Returns currently active signals"""
        return self.active_signals.copy()
    
    async def shutdown_gracefully(self) -> None:
        """Graceful shutdown of orchestrator"""
        
        logger.info("Starting graceful shutdown of QuantumOrchestrator...")
        
        # Wait for active signals to complete (with timeout)
        if self.active_signals:
            logger.info(f"Waiting for {len(self.active_signals)} active signals to complete...")
            await asyncio.sleep(5)  # Give 5 seconds for completion
        
        # Final system state update
        await self._update_system_state()
        
        logger.info("QuantumOrchestrator shutdown complete")
    
    def __repr__(self) -> str:
        return (f"QuantumOrchestrator(processed={self.orchestration_metrics.total_signals_processed}, "
                f"active={len(self.active_signals)}, regime={self.system_state.current_regime.value})")