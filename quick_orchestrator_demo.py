#!/usr/bin/env python3
"""
Quick Orchestrator Demo
======================

Schnelle Demo des selbst-entdeckenden Strategy Orchestrators
ohne echte Trading-Verbindung - nur Discovery und Analyse
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.strategy_orchestrator import StrategyOrchestrator
from core.intelligent_orchestration_engine import IntelligentOrchestrationEngine

async def run_demo():
    """Run orchestrator demo"""
    
    print("\n🚀 SELBST-ENTDECKENDER STRATEGY ORCHESTRATOR - DEMO")
    print("="*55)
    
    # Phase 1: Discovery
    print("\n📡 Phase 1: Strategy Discovery...")
    orchestrator = StrategyOrchestrator()
    
    discovered = await orchestrator.discover_all_strategies()
    print(f"\n✅ Entdeckt: {len(discovered)} Strategien\n")
    
    # Show discovered strategies
    for name, info in discovered.items():
        dna = info['dna']
        print(f"🧬 {name}:")
        print(f"   Risk Level: {dna.risk_level}")
        print(f"   Timeframe: {dna.timeframe}")
        print(f"   Signal Sources: {', '.join(dna.signal_sources[:3])}...")
        print(f"   Win Rate: {dna.expected_win_rate:.1%}")
        print(f"   Cooperation: {dna.cooperation_score:.1f}/10")
        if dna.conflict_strategies:
            print(f"   Conflicts: {', '.join(dna.conflict_strategies[:2])}")
        print()
    
    # Phase 2: Market Analysis
    print("\n📊 Phase 2: Market Analysis...")
    engine = IntelligentOrchestrationEngine()
    
    # Create sample market data
    market_data = {}
    for symbol in ['BTC/USDT', 'ETH/USDT']:
        hours = 100
        base_price = 45000 if symbol == 'BTC/USDT' else 2500
        
        # Simulate volatile market
        np.random.seed(42)
        prices = base_price + np.cumsum(np.random.randn(hours) * base_price * 0.01)
        
        market_data[symbol] = pd.DataFrame({
            'timestamp': pd.date_range(end=datetime.now(), periods=hours, freq='H'),
            'open': prices,
            'high': prices * 1.001,
            'low': prices * 0.999,
            'close': prices,
            'volume': np.random.lognormal(20, 1, hours) * 1000
        })
    
    # Phase 3: Orchestration Decision
    print("\n🎯 Phase 3: Orchestration Decision...")
    
    decision = await engine.orchestrate_strategies(
        market_data=market_data,
        risk_budget=100.0
    )
    
    print(f"\n📈 Market Conditions:")
    print(f"   Regime: {decision.market_regime}")
    print(f"   Volatility: {decision.market_volatility:.1%}")
    print(f"   Trend: {'Bullish' if decision.market_volatility > 0 else 'Bearish'}")
    
    print(f"\n💡 Strategy Selection:")
    total_weight = 0
    for strategy, weight in sorted(decision.strategy_weights.items(), 
                                  key=lambda x: x[1], reverse=True):
        if weight > 0:
            print(f"   {strategy}: {weight:.1%}")
            total_weight += weight
    
    if total_weight == 0:
        print("   ⚠️ Keine Strategien ausgewählt - Markt zu riskant")
    
    print(f"\n📊 Genetic Algorithm Stats:")
    if hasattr(decision, 'optimization_stats'):
        stats = decision.optimization_stats
        print(f"   Generations: {stats.get('generations', 'N/A')}")
        print(f"   Best Fitness: {stats.get('best_fitness', 'N/A')}")
    
    if decision.risk_warnings:
        print(f"\n⚠️ Risk Warnings:")
        for warning in decision.risk_warnings:
            print(f"   • {warning}")
    
    # Phase 4: Strategy DNA Analysis
    print(f"\n🧬 Phase 4: Strategy DNA Deep Dive...")
    
    # Pick top 3 strategies
    top_strategies = sorted(decision.strategy_weights.items(), 
                           key=lambda x: x[1], reverse=True)[:3]
    
    for strategy_name, weight in top_strategies:
        if weight > 0 and strategy_name in discovered:
            dna = discovered[strategy_name]['dna']
            print(f"\n   📌 {strategy_name} (Weight: {weight:.1%}):")
            print(f"      Code Metrics: {dna.code_metrics.get('total_lines', 0)} lines")
            print(f"      Complexity: {dna.code_metrics.get('complexity_score', 0):.1f}")
            print(f"      Dependencies: {len(dna.code_metrics.get('dependencies', []))}")
    
    print("\n✅ Demo abgeschlossen!")
    print("="*55)

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════╗
    ║        ORCHESTRATOR DEMO - QUICK START             ║
    ╠════════════════════════════════════════════════════╣
    ║                                                    ║
    ║  Diese Demo zeigt:                                 ║
    ║  • Automatische Strategy Discovery                 ║
    ║  • DNA Profiling jeder Strategie                   ║
    ║  • Intelligente Orchestration                      ║
    ║  • Market Regime Detection                         ║
    ║                                                    ║
    ║  Keine echten Trades - nur Analyse!                ║
    ╚════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(run_demo())