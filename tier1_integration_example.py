#!/usr/bin/env python3
"""
Tier-1 Trading System Integration Example
Demonstriert die Integration der Elite-Architektur mit der bestehenden SuperLazyBillionaire-Codebasis
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json

# Tier-1 System Imports
from core.tier1_models import Signal, SignalDirection, MarketRegime
from core.tier1_risk_engine import AdvancedRiskEngine
from core.tier1_capital_allocator import AdvancedCapitalAllocator
from core.tier1_execution_layer import AdvancedExecutionLayer
from core.quantum_orchestrator import QuantumOrchestrator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Tier1TradingSystemDemo:
    """
    Demonstration des kompletten Tier-1 Trading Systems
    Zeigt Integration mit bestehenden SuperLazyBillionaire-Strategien
    """
    
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        
        # Tier-1 Komponenten initialisieren
        self.risk_engine = AdvancedRiskEngine(
            max_portfolio_var=0.05,
            max_single_position=0.10,
            max_correlation=0.7
        )
        
        self.capital_allocator = AdvancedCapitalAllocator(
            total_capital=initial_capital,
            max_allocation_per_signal=0.05,
            volatility_target=0.15
        )
        
        self.execution_layer = AdvancedExecutionLayer()
        
        # Quantum Orchestrator als zentrales Gehirn
        self.orchestrator = QuantumOrchestrator(
            risk_engine=self.risk_engine,
            capital_allocator=self.capital_allocator,
            execution_layer=self.execution_layer
        )
        
        # Simulierte Marktdaten
        self.market_data = {
            'BTC': {'price': 50000, 'volume': 1000000},
            'ETH': {'price': 3000, 'volume': 500000},
            'BNB': {'price': 400, 'volume': 200000}
        }
        
        logger.info(f"Tier-1 Trading System initialisiert mit ${initial_capital:,.0f} Kapital")
    
    def create_sample_signals(self) -> List[Signal]:
        """Erstellt Sample-Signale aus verschiedenen SuperLazyBillionaire-Strategien"""
        
        signals = []
        
        # 1. Lazy Billionaire Signal (High Confidence)
        signals.append(Signal(
            stream_id="LB_001",
            asset="BTC",
            direction="long",
            confidence=0.85,
            timestamp=datetime.now(),
            expected_profit_pts=150,  # 1.5%
            expected_duration_min=240,  # 4 hours
            source_metadata={
                "strategy_sharpe": 0.76,
                "regime_alignment": True,
                "correlation_score": 0.9
            },
            origin="lazy_billionaire"
        ))
        
        # 2. ML Strategy Signal (Medium Confidence)
        signals.append(Signal(
            stream_id="ML_002",
            asset="ETH",
            direction="short",
            confidence=0.72,
            timestamp=datetime.now(),
            expected_profit_pts=120,  # 1.2%
            expected_duration_min=180,  # 3 hours
            source_metadata={
                "ml_confidence": 0.72,
                "feature_importance": {"volume": 0.3, "price_momentum": 0.4, "sentiment": 0.3},
                "model_accuracy": 0.68
            },
            origin="ml_strategy"
        ))
        
        # 3. Arbitrage Signal (Very High Confidence, Short Duration)
        signals.append(Signal(
            stream_id="ARB_003",
            asset="BNB",
            direction="long",
            confidence=0.95,
            timestamp=datetime.now(),
            expected_profit_pts=50,   # 0.5%
            expected_duration_min=15,  # 15 minutes
            source_metadata={
                "price_differential": 0.008,
                "execution_certainty": 0.95,
                "venue_spread": {"binance": 0.001, "coinbase": 0.009}
            },
            origin="arbitrage"
        ))
        
        # 4. Mean Reversion Signal (High Confidence)
        signals.append(Signal(
            stream_id="MR_004",
            asset="BTC",
            direction="short",
            confidence=0.78,
            timestamp=datetime.now(),
            expected_profit_pts=200,  # 2.0%
            expected_duration_min=360,  # 6 hours
            source_metadata={
                "z_score": 2.3,
                "bollinger_position": "upper_band",
                "rsi": 78.5,
                "mean_reversion_strength": 0.8
            },
            origin="mean_reversion"
        ))
        
        # 5. Momentum Signal (Medium Confidence)
        signals.append(Signal(
            stream_id="MOM_005",
            asset="ETH",
            direction="long",
            confidence=0.68,
            timestamp=datetime.now(),
            expected_profit_pts=300,  # 3.0%
            expected_duration_min=480,  # 8 hours
            source_metadata={
                "momentum_score": 0.7,
                "trend_strength": 0.65,
                "volume_confirmation": True
            },
            origin="momentum"
        ))
        
        # 6. Liquidation Hunter Signal (High Risk/Reward)
        signals.append(Signal(
            stream_id="LH_006",
            asset="BTC",
            direction="long",
            confidence=0.82,
            timestamp=datetime.now(),
            expected_profit_pts=400,  # 4.0%
            expected_duration_min=120,  # 2 hours
            source_metadata={
                "liquidation_level": 48500,
                "leverage_clustering": 0.85,
                "order_book_imbalance": 0.9
            },
            origin="liquidation_hunter"
        ))
        
        return signals
    
    async def simulate_market_data_updates(self):
        """Simuliert Live-Market-Data für realistische Bedingungen"""
        
        for asset in self.market_data:
            # Simuliere Preisbewegung
            current_price = self.market_data[asset]['price']
            price_change = current_price * (0.02 * (2 * asyncio.get_event_loop().time() % 1 - 1))  # ±2% oscillation
            new_price = current_price + price_change
            
            # Aktualisiere Marktdaten
            self.market_data[asset]['price'] = new_price
            
            # Update Risk Engine
            self.risk_engine.update_market_data(new_price, self.market_data[asset]['volume'])
            
            # Update Execution Layer
            bid = new_price * 0.9995  # 0.05% spread
            ask = new_price * 1.0005
            self.execution_layer.update_market_data(asset, bid, ask, self.market_data[asset]['volume'])
    
    async def run_tier1_demo(self):
        """Führt komplette Tier-1 Trading System Demo aus"""
        
        print("🚀 TIER-1 TRADING SYSTEM DEMO")
        print("=" * 80)
        print(f"Startkapital: ${self.initial_capital:,.0f}")
        print(f"Initialisiert: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Health Check
        health = await self.orchestrator.perform_health_check()
        print(f"🏥 System Health: {health['overall_status'].upper()}")
        if health['issues']:
            print(f"   Issues: {', '.join(health['issues'])}")
        print()
        
        # Market Data Updates
        await self.simulate_market_data_updates()
        
        # Erstelle Sample Signale
        signals = self.create_sample_signals()
        print(f"📡 {len(signals)} Signale generiert aus SuperLazyBillionaire-Strategien:")
        for signal in signals:
            print(f"   • {signal.origin}: {signal.asset} {signal.direction} "
                  f"(confidence={signal.confidence:.1%}, expected={signal.expected_profit_pts}bps)")
        print()
        
        # Verarbeite Signale durch Orchestrator
        print("🧠 QUANTUM ORCHESTRATOR - SIGNAL VERARBEITUNG")
        print("-" * 60)
        
        results = []
        for i, signal in enumerate(signals, 1):
            print(f"\n[{i}/{len(signals)}] Verarbeite Signal: {signal.stream_id} ({signal.origin})")
            
            # Process durch Orchestrator
            result = await self.orchestrator.on_new_signal(signal)
            results.append(result)
            
            # Ergebnis anzeigen
            status_icon = {
                "executed": "✅",
                "rejected": "❌", 
                "skipped": "⏭️",
                "aborted": "🛑"
            }.get(result.status, "❓")
            
            print(f"   Status: {status_icon} {result.status.upper()}")
            if result.reason:
                print(f"   Grund: {result.reason}")
            if result.capital_allocated:
                print(f"   Kapital: ${result.capital_allocated:,.0f}")
            if result.execution_score:
                print(f"   Execution Score: {result.execution_score:.3f}")
            print(f"   Verarbeitungszeit: {result.processing_time_ms:.1f}ms")
        
        # Zusammenfassung
        print(f"\n📊 VERARBEITUNGS-ZUSAMMENFASSUNG")
        print("-" * 60)
        
        executed = sum(1 for r in results if r.status == "executed")
        rejected = sum(1 for r in results if r.status == "rejected")
        skipped = sum(1 for r in results if r.status == "skipped")
        aborted = sum(1 for r in results if r.status == "aborted")
        
        total_allocated = sum(r.capital_allocated for r in results if r.capital_allocated)
        avg_score = sum(r.execution_score for r in results if r.execution_score) / max(1, executed)
        
        print(f"Signale verarbeitet: {len(results)}")
        print(f"✅ Ausgeführt: {executed} ({executed/len(results):.1%})")
        print(f"❌ Abgelehnt: {rejected} ({rejected/len(results):.1%})")
        print(f"⏭️ Übersprungen: {skipped} ({skipped/len(results):.1%})")
        print(f"🛑 Abgebrochen: {aborted} ({aborted/len(results):.1%})")
        print(f"💰 Gesamtallokation: ${total_allocated:,.0f}")
        print(f"📈 Ø Execution Score: {avg_score:.3f}")
        
        # System Status
        print(f"\n⚙️ SYSTEM STATUS")
        print("-" * 60)
        
        system_state = self.orchestrator.get_system_state()
        metrics = self.orchestrator.get_orchestration_metrics()
        
        print(f"Verfügbares Kapital: ${system_state.available_capital:,.0f}")
        print(f"Allokiertes Kapital: ${system_state.allocated_capital:,.0f}")
        print(f"Kapitalauslastung: {(system_state.allocated_capital/system_state.total_capital):.1%}")
        print(f"Aktive Positionen: {system_state.active_positions}")
        print(f"Market Regime: {system_state.current_regime.value}")
        
        # Risk Engine Status
        risk_status = self.risk_engine.get_risk_status()
        print(f"Risk Engine Regime: {risk_status['current_regime']} "
              f"(Confidence: {risk_status['regime_confidence']:.1%})")
        
        # Execution Layer Status
        exec_status = self.execution_layer.get_execution_status()
        print(f"Execution Success Rate: {exec_status['success_rate']:.1%}")
        print(f"Ø Slippage: {exec_status['avg_slippage_bps']:.1f}bps")
        print(f"Ø Latency: {exec_status['avg_latency_ms']:.0f}ms")
        
        # Detaillierte Ergebnisse speichern
        await self.save_demo_results(results, system_state, metrics)
        
        print(f"\n🎯 TIER-1 DEMO ABGESCHLOSSEN")
        print("=" * 80)
        
        return results
    
    async def save_demo_results(self, results, system_state, metrics):
        """Speichert Demo-Ergebnisse für Analyse"""
        
        demo_data = {
            'timestamp': datetime.now().isoformat(),
            'initial_capital': self.initial_capital,
            'processing_results': [
                {
                    'signal_id': r.signal_id,
                    'status': r.status,
                    'reason': r.reason,
                    'capital_allocated': r.capital_allocated,
                    'execution_score': r.execution_score,
                    'processing_time_ms': r.processing_time_ms
                } for r in results
            ],
            'final_system_state': {
                'total_capital': system_state.total_capital,
                'allocated_capital': system_state.allocated_capital,
                'available_capital': system_state.available_capital,
                'active_positions': system_state.active_positions,
                'current_regime': system_state.current_regime.value
            },
            'orchestration_metrics': {
                'total_signals_processed': metrics.total_signals_processed,
                'signals_executed': metrics.signals_executed,
                'signals_rejected': metrics.signals_rejected,
                'signals_skipped': metrics.signals_skipped,
                'signals_aborted': metrics.signals_aborted,
                'avg_execution_score': metrics.avg_execution_score
            }
        }
        
        filename = f"tier1_demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(demo_data, f, indent=2, default=str)
        
        print(f"📁 Demo-Ergebnisse gespeichert: {filename}")
    
    async def benchmark_performance(self, num_signals: int = 100):
        """Benchmark der Orchestrator-Performance"""
        
        print(f"\n⚡ PERFORMANCE BENCHMARK - {num_signals} Signale")
        print("-" * 60)
        
        # Generiere viele Test-Signale
        benchmark_signals = []
        for i in range(num_signals):
            signal = Signal(
                stream_id=f"BENCH_{i:03d}",
                asset=["BTC", "ETH", "BNB"][i % 3],
                direction=["long", "short"][i % 2],
                confidence=0.5 + (i % 50) / 100,  # 0.5-0.99
                timestamp=datetime.now(),
                expected_profit_pts=50 + (i % 200),
                expected_duration_min=30 + (i % 300),
                source_metadata={"benchmark": True},
                origin=["lazy_billionaire", "ml_strategy", "arbitrage"][i % 3]
            )
            benchmark_signals.append(signal)
        
        # Benchmark-Start
        start_time = datetime.now()
        
        # Verarbeite alle Signale
        results = await self.orchestrator.process_multiple_signals(benchmark_signals)
        
        # Benchmark-Ende
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Auswertung
        successful_results = [r for r in results if not isinstance(r, Exception)]
        exceptions = [r for r in results if isinstance(r, Exception)]
        
        print(f"Signale verarbeitet: {len(successful_results)}/{num_signals}")
        print(f"Fehler aufgetreten: {len(exceptions)}")
        print(f"Gesamtzeit: {total_time:.2f}s")
        print(f"Durchsatz: {len(successful_results)/total_time:.1f} Signale/s")
        
        if successful_results:
            avg_processing_time = sum(r.processing_time_ms for r in successful_results) / len(successful_results)
            print(f"Ø Verarbeitungszeit: {avg_processing_time:.1f}ms")
        
        return successful_results


async def main():
    """Hauptfunktion für Tier-1 Demo"""
    
    # Tier-1 System initialisieren
    demo = Tier1TradingSystemDemo(initial_capital=1000000)
    
    try:
        # Hauptdemo ausführen
        results = await demo.run_tier1_demo()
        
        # Performance Benchmark (optional)
        print(f"\n" + "="*80)
        benchmark_choice = input("Performance Benchmark ausführen? (y/n): ").lower().strip()
        
        if benchmark_choice == 'y':
            await demo.benchmark_performance(50)  # 50 Signale für schnelles Benchmark
        
        # Graceful Shutdown
        await demo.orchestrator.shutdown_gracefully()
        
        print(f"\n✅ Tier-1 Trading System Demo erfolgreich abgeschlossen!")
        
    except KeyboardInterrupt:
        print(f"\n🛑 Demo durch Benutzer abgebrochen")
        await demo.orchestrator.shutdown_gracefully()
    
    except Exception as e:
        print(f"\n❌ Fehler in Demo: {e}")
        logger.exception("Demo Fehler")


if __name__ == "__main__":
    # Stelle sicher, dass Event Loop läuft
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Auf Wiedersehen!")