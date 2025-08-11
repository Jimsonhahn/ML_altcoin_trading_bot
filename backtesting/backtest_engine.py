"""
Backtest Engine - Haupt-Event-Loop für ereignisgesteuerte Backtests
Orchestriert alle Komponenten für realistische Portfolio-Simulation
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from pathlib import Path
import json

# Backtesting Framework
from .event_bus import EventBus, create_event_bus
from .data_handler import DataHandler, HistoricalDataHandler, SimulatedDataHandler
from .simulated_exchange import SimulatedExchange
from .portfolio_manager import PortfolioManager

# Tier-1 System Integration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.quantum_orchestrator import QuantumOrchestrator
from core.tier1_risk_engine import AdvancedRiskEngine
from core.tier1_capital_allocator import AdvancedCapitalAllocator
from core.tier1_execution_layer import AdvancedExecutionLayer

# Event Models
from .event_models import (
    EventType, MarketEvent, SignalEvent, OrderEvent,
    create_signal_event_from_tier1_signal
)

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Ereignisgesteuerter Backtest Engine
    
    Orchestriert:
    - DataHandler: Point-in-Time Marktdaten
    - QuantumOrchestrator: Signal-Verarbeitung
    - SimulatedExchange: Realistische Order-Execution
    - PortfolioManager: Portfolio-State und Performance
    
    Features:
    - Strikt kein Lookahead-Bias
    - Realistische Transaktionskosten
    - Vollständiges Event-Tracking
    - Performance-Analyse
    """
    
    def __init__(self,
                 initial_capital: float = 1000000.0,
                 start_date: datetime = None,
                 end_date: datetime = None,
                 data_directory: str = "data",
                 symbols: List[str] = None,
                 enable_detailed_logging: bool = False,
                 max_concurrent_events: int = 1000):
        
        self.initial_capital = initial_capital
        self.start_date = start_date or datetime(2022, 1, 1)
        self.end_date = end_date or datetime(2023, 12, 31)
        self.data_directory = Path(data_directory)
        self.symbols = symbols or ["BTC", "ETH", "BNB"]
        self.enable_detailed_logging = enable_detailed_logging
        self.max_concurrent_events = max_concurrent_events
        
        # Core Components
        self.event_bus: Optional[EventBus] = None
        self.data_handler: Optional[DataHandler] = None
        self.simulated_exchange: Optional[SimulatedExchange] = None
        self.portfolio_manager: Optional[PortfolioManager] = None
        self.quantum_orchestrator: Optional[QuantumOrchestrator] = None
        
        # Component Status
        self.components_initialized = False
        self.backtest_running = False
        self.backtest_completed = False
        
        # Performance Tracking
        self.backtest_start_time: Optional[datetime] = None
        self.backtest_end_time: Optional[datetime] = None
        self.events_processed = 0
        self.signals_generated = 0
        self.orders_executed = 0
        
        # Results Storage
        self.backtest_results: Dict[str, Any] = {}
        self.detailed_logs: List[Dict[str, Any]] = []
        
        logger.info(f"BacktestEngine initialisiert: {self.start_date} - {self.end_date}")
    
    async def initialize_components(self) -> None:
        """
        Initialisiert alle Backtest-Komponenten
        """
        
        logger.info("Initialisiere Backtest-Komponenten...")
        
        # 1. Event Bus
        self.event_bus = create_event_bus(
            priority=True,
            max_queue_size=self.max_concurrent_events,
            enable_stats=True
        )
        
        # 2. Data Handler
        await self._initialize_data_handler()
        
        # 3. Portfolio Manager
        self.portfolio_manager = PortfolioManager(
            event_bus=self.event_bus,
            initial_capital=self.initial_capital,
            base_currency="USDT",
            enable_margin=False,
            max_leverage=1.0
        )
        
        # 4. Simulated Exchange
        self.simulated_exchange = SimulatedExchange(
            event_bus=self.event_bus,
            exchange_name="simulated_binance",
            maker_fee=0.001,  # 0.1%
            taker_fee=0.001,  # 0.1%
            enable_partial_fills=True,
            enable_market_impact=True,
            enable_latency=True
        )
        
        # 5. Tier-1 System Components
        await self._initialize_tier1_system()
        
        # 6. QuantumOrchestrator
        await self._initialize_quantum_orchestrator()
        
        # 7. Subscribe to events for signal generation
        self.event_bus.subscribe(EventType.MARKET, self._handle_market_event_for_signals)
        
        self.components_initialized = True
        logger.info("Alle Komponenten erfolgreich initialisiert")
    
    async def _initialize_data_handler(self) -> None:
        """Initialisiert Data Handler"""
        
        # Check if data directory exists
        if self.data_directory.exists():
            # Use historical data
            self.data_handler = HistoricalDataHandler(
                event_bus=self.event_bus,
                data_directory=str(self.data_directory),
                symbols=self.symbols,
                start_date=self.start_date,
                end_date=self.end_date,
                timeframe="1h",
                data_format="csv"
            )
            logger.info("HistoricalDataHandler konfiguriert")
        else:
            # Use simulated data
            initial_prices = {
                "BTC": 50000,
                "ETH": 3000,
                "BNB": 400
            }
            
            self.data_handler = SimulatedDataHandler(
                event_bus=self.event_bus,
                symbols=self.symbols,
                start_date=self.start_date,
                end_date=self.end_date,
                timeframe_minutes=60,
                initial_prices=initial_prices
            )
            logger.info("SimulatedDataHandler konfiguriert")
    
    async def _initialize_tier1_system(self) -> None:
        """Initialisiert Tier-1 Trading System Komponenten"""
        
        # Risk Engine
        risk_engine = AdvancedRiskEngine(
            max_portfolio_var=0.05,
            max_single_position=0.10,
            max_correlation=0.7
        )
        
        # Capital Allocator
        capital_allocator = AdvancedCapitalAllocator(
            total_capital=self.initial_capital,
            max_allocation_per_signal=0.05,
            volatility_target=0.15
        )
        
        # Execution Layer
        execution_layer = AdvancedExecutionLayer()
        
        # Store references
        self.risk_engine = risk_engine
        self.capital_allocator = capital_allocator
        self.execution_layer = execution_layer
        
        logger.info("Tier-1 System Komponenten initialisiert")
    
    async def _initialize_quantum_orchestrator(self) -> None:
        """Initialisiert QuantumOrchestrator mit Event-Integration"""
        
        self.quantum_orchestrator = QuantumOrchestrator(
            risk_engine=self.risk_engine,
            capital_allocator=self.capital_allocator,
            execution_layer=self.execution_layer,
            enable_monitoring=True
        )
        
        # Subscribe to signal events
        self.event_bus.subscribe(EventType.SIGNAL, self._handle_signal_event)
        
        logger.info("QuantumOrchestrator initialisiert")
    
    async def _handle_market_event_for_signals(self, event: MarketEvent) -> None:
        """
        Generiert Signale basierend auf Market Events
        Simuliert Signal-Generierung der verschiedenen Strategien
        """
        
        try:
            # Simulate signal generation from various strategies
            signals = await self._generate_signals_from_market_data(event)
            
            # Convert to signal events and publish
            for signal in signals:
                signal_event = create_signal_event_from_tier1_signal(signal, event.timestamp)
                await self.event_bus.publish(signal_event)
                self.signals_generated += 1
                
        except Exception as e:
            logger.error(f"Error generating signals from market event: {e}")
    
    async def _generate_signals_from_market_data(self, market_event: MarketEvent) -> List[Any]:
        """
        Simuliert Signal-Generierung der SuperLazyBillionaire-Strategien
        """
        
        signals = []
        
        # Simple signal generation logic (in real implementation,
        # this would integrate with actual strategy signals)
        
        # Random signal generation for demonstration
        import random
        from ..core.tier1_models import Signal
        
        # Probability of signal generation per strategy
        strategy_probabilities = {
            'lazy_billionaire': 0.05,    # 5% chance per hour
            'ml_strategy': 0.03,         # 3% chance per hour
            'arbitrage': 0.08,           # 8% chance per hour (more frequent)
            'mean_reversion': 0.04,      # 4% chance per hour
            'momentum': 0.02,            # 2% chance per hour
            'grid': 0.06,                # 6% chance per hour
        }
        
        for strategy, probability in strategy_probabilities.items():
            if random.random() < probability:
                # Generate signal
                direction = random.choice(['long', 'short'])
                confidence = random.uniform(0.5, 0.9)
                expected_profit = random.uniform(50, 300)  # 0.5% to 3%
                duration = random.randint(60, 480)  # 1-8 hours
                
                signal = Signal(
                    stream_id=f"{strategy}_{market_event.timestamp.strftime('%Y%m%d_%H%M')}",
                    asset=market_event.symbol,
                    direction=direction,
                    confidence=confidence,
                    timestamp=market_event.timestamp,
                    expected_profit_pts=expected_profit,
                    expected_duration_min=duration,
                    source_metadata={
                        'price': market_event.close,
                        'volume': market_event.volume,
                        'strategy_specific': f"{strategy}_data"
                    },
                    origin=strategy
                )
                
                signals.append(signal)
        
        return signals
    
    async def _handle_signal_event(self, event: SignalEvent) -> None:
        """
        Verarbeitet Signal Events durch QuantumOrchestrator
        """
        
        try:
            # Convert SignalEvent back to tier1 Signal
            from core.tier1_models import Signal
            
            tier1_signal = Signal(
                stream_id=event.stream_id,
                asset=event.symbol,
                direction=event.direction,
                confidence=event.confidence,
                timestamp=event.timestamp,
                expected_profit_pts=event.expected_profit_pts,
                expected_duration_min=event.expected_duration_min,
                source_metadata=event.signal_metadata,
                origin=event.origin
            )
            
            # Process through QuantumOrchestrator
            result = await self.quantum_orchestrator.on_new_signal(tier1_signal)
            
            # If signal was accepted, generate order event
            if result.status == "executed" and result.order_id:
                # The QuantumOrchestrator has processed the signal,
                # now we need to translate the result to an OrderEvent
                await self._create_order_from_orchestrator_result(tier1_signal, result)
                
        except Exception as e:
            logger.error(f"Error handling signal event: {e}")
    
    async def _create_order_from_orchestrator_result(self, signal: Any, result: Any) -> None:
        """
        Erstellt OrderEvent basierend auf QuantumOrchestrator Ergebnis
        """
        
        try:
            # This is a simplified translation - in full implementation,
            # the QuantumOrchestrator would directly interface with the exchange
            
            order_event = OrderEvent(
                timestamp=datetime.now(),
                order_id=result.order_id or f"ORD_{signal.signal_id[:8]}",
                signal_id=signal.signal_id,
                symbol=signal.asset,
                side="buy" if signal.direction == "long" else "sell",
                order_type="market",  # Simplified
                quantity=result.capital_allocated / 50000 if result.capital_allocated else 0.1,  # Simplified
                allocation_amount_usd=result.capital_allocated or 0.0,
                urgency_factor=signal.confidence,
                exchange="simulated_binance"
            )
            
            # Publish order event
            await self.event_bus.publish(order_event)
            self.orders_executed += 1
            
        except Exception as e:
            logger.error(f"Error creating order from orchestrator result: {e}")
    
    async def run_backtest(self) -> Dict[str, Any]:
        """
        Hauptmethode: Führt kompletten Backtest aus
        """
        
        if not self.components_initialized:
            await self.initialize_components()
        
        logger.info("Starte ereignisgesteuerten Backtest...")
        self.backtest_start_time = datetime.now()
        self.backtest_running = True
        
        try:
            # Start event processing
            event_processing_task = asyncio.create_task(
                self.event_bus.process_events()
            )
            
            # Start data streaming
            data_streaming_task = asyncio.create_task(
                self.data_handler.stream_data()
            )
            
            # Monitor progress
            progress_task = asyncio.create_task(
                self._monitor_progress()
            )
            
            # Wait for data streaming to complete
            await data_streaming_task
            
            # Wait for all events to be processed
            logger.info("Daten-Streaming abgeschlossen, warte auf Event-Verarbeitung...")
            await self.event_bus.wait_until_empty(timeout=60.0)
            
            # Stop event processing
            self.event_bus.stop()
            
            # Cancel tasks
            event_processing_task.cancel()
            progress_task.cancel()
            
            # Finalize results
            await self._finalize_backtest()
            
        except Exception as e:
            logger.error(f"Fehler während Backtest: {e}")
            raise
        finally:
            self.backtest_running = False
            self.backtest_end_time = datetime.now()
        
        logger.info("Backtest erfolgreich abgeschlossen")
        return self.backtest_results
    
    async def _monitor_progress(self) -> None:
        """Überwacht Backtest-Progress"""
        
        last_update = time.time()
        
        while self.backtest_running:
            try:
                current_time = time.time()
                
                if current_time - last_update > 30:  # Update alle 30 Sekunden
                    stats = self.event_bus.get_stats()
                    
                    logger.info(f"Backtest Progress: "
                               f"Events={stats['total_events_processed']}, "
                               f"EPS={stats['events_per_second']:.1f}, "
                               f"Queue={stats['queue_size']}, "
                               f"Signals={self.signals_generated}, "
                               f"Orders={self.orders_executed}")
                    
                    last_update = current_time
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in progress monitoring: {e}")
    
    async def _finalize_backtest(self) -> None:
        """Finalisiert Backtest und sammelt Ergebnisse"""
        
        logger.info("Finalisiere Backtest-Ergebnisse...")
        
        # Portfolio Performance
        portfolio_metrics = self.portfolio_manager.get_performance_metrics()
        positions_summary = self.portfolio_manager.get_positions_summary()
        
        # Exchange Statistics
        exchange_stats = self.simulated_exchange.get_execution_stats()
        
        # Event Bus Statistics
        event_stats = self.event_bus.get_stats()
        
        # QuantumOrchestrator Metrics
        orchestrator_metrics = self.quantum_orchestrator.get_orchestration_metrics()
        system_state = self.quantum_orchestrator.get_system_state()
        
        # Timing
        backtest_duration = (self.backtest_end_time - self.backtest_start_time).total_seconds()
        
        # Compile results
        self.backtest_results = {
            'backtest_info': {
                'start_date': self.start_date.isoformat(),
                'end_date': self.end_date.isoformat(),
                'initial_capital': self.initial_capital,
                'symbols': self.symbols,
                'backtest_duration_seconds': backtest_duration,
                'timestamp': datetime.now().isoformat()
            },
            'portfolio_performance': portfolio_metrics,
            'final_positions': positions_summary,
            'exchange_execution': exchange_stats,
            'orchestrator_metrics': {
                'total_signals_processed': orchestrator_metrics.total_signals_processed,
                'signals_executed': orchestrator_metrics.signals_executed,
                'signals_rejected': orchestrator_metrics.signals_rejected,
                'signals_skipped': orchestrator_metrics.signals_skipped,
                'signals_aborted': orchestrator_metrics.signals_aborted,
                'avg_execution_score': orchestrator_metrics.avg_execution_score,
                'risk_rejection_rate': orchestrator_metrics.risk_rejection_rate
            },
            'system_state': {
                'total_capital': system_state.total_capital,
                'allocated_capital': system_state.allocated_capital,
                'available_capital': system_state.available_capital,
                'active_positions': system_state.active_positions,
                'current_regime': system_state.current_regime.value if system_state.current_regime else None
            },
            'event_statistics': event_stats,
            'signal_generation': {
                'total_signals_generated': self.signals_generated,
                'total_orders_executed': self.orders_executed
            }
        }
        
        # Export detailed results
        await self._export_results()
        
        self.backtest_completed = True
    
    async def _export_results(self) -> None:
        """Exportiert detaillierte Ergebnisse"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Main results file
        results_file = f"backtest_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.backtest_results, f, indent=2, default=str)
        
        logger.info(f"Backtest-Ergebnisse exportiert: {results_file}")
        
        # Export equity curve
        equity_df = self.portfolio_manager.get_equity_curve_data()
        if not equity_df.empty:
            equity_file = f"equity_curve_{timestamp}.csv"
            equity_df.to_csv(equity_file)
            logger.info(f"Equity Curve exportiert: {equity_file}")
        
        # Export trade history
        trade_file = self.portfolio_manager.export_trade_history(f"trades_{timestamp}.json")
        logger.info(f"Trade History exportiert: {trade_file}")
    
    def get_results_summary(self) -> str:
        """Gibt formatierte Zusammenfassung der Ergebnisse zurück"""
        
        if not self.backtest_completed:
            return "Backtest noch nicht abgeschlossen"
        
        results = self.backtest_results
        portfolio = results['portfolio_performance']
        
        summary = f"""
🚀 EREIGNISGESTEUERTER BACKTEST - ERGEBNISSE
{'='*80}

📅 Zeitraum: {results['backtest_info']['start_date'][:10]} - {results['backtest_info']['end_date'][:10]}
💰 Startkapital: ${results['backtest_info']['initial_capital']:,.0f}

📈 PORTFOLIO PERFORMANCE
{'-'*60}
Gesamtrendite:     {portfolio.get('total_return', 0):.1%}
Jährliche Rendite: {portfolio.get('annual_return', 0):.1%}
Sharpe Ratio:      {portfolio.get('sharpe_ratio', 0):.2f}
Max Drawdown:      {portfolio.get('max_drawdown', 0):.1%}
Volatilität:       {portfolio.get('annual_volatility', 0):.1%}

📊 TRADING STATISTIKEN
{'-'*60}
Gesamte Trades:    {portfolio.get('total_trades', 0):,}
Gewinnrate:        {portfolio.get('win_rate', 0):.1%}
Profit Factor:     {portfolio.get('profit_factor', 0):.2f}
Kommissionen:      ${portfolio.get('total_commission', 0):,.0f}

🧠 ORCHESTRATOR LEISTUNG
{'-'*60}
Signale verarbeitet: {results['orchestrator_metrics']['total_signals_processed']:,}
Ausgeführt:         {results['orchestrator_metrics']['signals_executed']:,}
Ø Execution Score:  {results['orchestrator_metrics']['avg_execution_score']:.3f}
Rejection Rate:     {results['orchestrator_metrics']['risk_rejection_rate']:.1%}

⚡ EXECUTION QUALITÄT
{'-'*60}
Gefüllte Orders:    {results['exchange_execution']['filled_orders']:,}
Ø Slippage:         {results['exchange_execution'].get('avg_slippage_bps', 0):.1f} bps
Fill Rate:          {(results['exchange_execution']['filled_orders'] / max(results['exchange_execution']['total_orders'], 1)):.1%}

🎯 SYSTEM PERFORMANCE
{'-'*60}
Events verarbeitet: {results['event_statistics']['total_events_processed']:,}
Events/Sekunde:     {results['event_statistics']['events_per_second']:.1f}
Backtest Duration:  {results['backtest_info']['backtest_duration_seconds']:.1f}s
        """
        
        return summary
    
    async def cleanup(self) -> None:
        """Räumt Ressourcen auf"""
        
        if self.event_bus:
            self.event_bus.stop()
        
        if self.quantum_orchestrator:
            await self.quantum_orchestrator.shutdown_gracefully()
        
        logger.info("Backtest-Ressourcen bereinigt")


# Factory Function
def create_backtest_engine(**kwargs) -> BacktestEngine:
    """
    Factory für BacktestEngine Erstellung
    """
    return BacktestEngine(**kwargs)