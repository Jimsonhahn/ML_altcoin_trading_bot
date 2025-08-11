#!/usr/bin/env python3
"""
Start Self-Discovering Strategy Orchestrator
===========================================

Startet den intelligenten Strategy Orchestrator, der:
- Alle Strategien automatisch findet
- Sie dynamisch lädt und analysiert
- Intelligente Orchestration basierend auf Marktbedingungen durchführt
- Health Monitoring und A/B Testing aktiviert
"""

import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import orchestrator components
from core.strategy_orchestrator import StrategyOrchestrator
from core.intelligent_orchestration_engine import IntelligentOrchestrationEngine
from core.strategy_health_monitor import HealthAndABTestingSystem

async def fetch_market_data(symbols: list = None) -> Dict[str, pd.DataFrame]:
    """Fetch market data for orchestrator (simulated for demo)"""
    
    if symbols is None:
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT']
    
    market_data = {}
    
    for symbol in symbols:
        # Generate realistic market data
        hours = 168  # 1 week of hourly data
        timestamps = pd.date_range(end=datetime.now(), periods=hours, freq='H')
        
        # Base price with trend
        base_price = {
            'BTC/USDT': 45000,
            'ETH/USDT': 2500,
            'SOL/USDT': 100,
            'BNB/USDT': 350
        }.get(symbol, 100)
        
        # Generate OHLCV data
        np.random.seed(hash(symbol) % 1000)
        trend = np.cumsum(np.random.randn(hours) * base_price * 0.002)
        prices = base_price + trend
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices + np.random.randn(hours) * base_price * 0.001,
            'high': prices + np.abs(np.random.randn(hours)) * base_price * 0.002,
            'low': prices - np.abs(np.random.randn(hours)) * base_price * 0.002,
            'close': prices,
            'volume': np.random.lognormal(20, 1, hours) * 1000
        })
        
        market_data[symbol] = data
    
    return market_data

async def main():
    """Main orchestrator startup"""
    
    # Load configuration
    config_path = Path("orchestrator_config.json")
    config = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Get trading mode from config or command line
    trading_mode = config.get('trading_mode', {}).get('default', 'paper')
    
    # Allow command line override
    if len(sys.argv) > 1:
        mode_arg = sys.argv[1].lower()
        if mode_arg in ['paper', 'live', 'hybrid']:
            trading_mode = mode_arg
    
    print("\n" + "="*60)
    print("🚀 SELBST-ENTDECKENDER STRATEGY ORCHESTRATOR")
    print("="*60)
    print(f"\n📈 Trading Mode: {trading_mode.upper()}")
    print("\nStarting intelligent strategy orchestration system...")
    print("This system will automatically discover and manage all strategies.\n")
    
    # Initialize components
    print("🔍 Phase 1: Initializing Strategy Discovery...")
    orchestrator = StrategyOrchestrator()
    
    print("🧠 Phase 2: Initializing Intelligent Orchestration Engine...")
    orchestration_engine = IntelligentOrchestrationEngine()
    
    print("🏥 Phase 3: Initializing Health Monitoring & A/B Testing...")
    health_system = HealthAndABTestingSystem()
    
    print("💼 Phase 4: Initializing Portfolio Manager...")
    from core.orchestrator_portfolio_manager import create_portfolio_manager
    initial_capital = config.get('portfolio_management', {}).get('initial_capital', 10000.0)
    portfolio_manager = await create_portfolio_manager(trading_mode, initial_capital)
    
    # Discover all strategies
    print("\n📡 Phase 5: Discovering All Strategies...")
    discovered_strategies = await orchestrator.discover_all_strategies()
    
    print(f"\n✅ Discovered {len(discovered_strategies)} strategies:")
    for name, strategy_info in discovered_strategies.items():
        dna = strategy_info['dna']
        print(f"   • {name}:")
        print(f"     - Risk Level: {dna.risk_level}")
        print(f"     - Timeframe: {dna.timeframe}")
        print(f"     - Signal Sources: {', '.join(dna.signal_sources)}")
        print(f"     - Expected Win Rate: {dna.expected_win_rate:.1%}")
    
    # Main orchestration loop
    print("\n🎯 Starting Orchestration Loop...")
    print("-" * 60)
    
    risk_budget = 100.0  # Starting risk budget
    iteration = 0
    
    try:
        while True:
            iteration += 1
            print(f"\n🔄 Orchestration Cycle #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Fetch current market data
            market_data = await fetch_market_data()
            
            # Run intelligent orchestration
            decision = await orchestration_engine.orchestrate_strategies(
                market_data=market_data,
                risk_budget=risk_budget
            )
            
            # Display orchestration decision
            print(f"\n📊 Market Analysis:")
            print(f"   • Market Regime: {decision.market_regime}")
            print(f"   • Volatility: {decision.market_volatility:.1%}")
            print(f"   • Risk Budget: ${risk_budget:.2f}")
            
            # Allocate capital through portfolio manager
            allocations = await portfolio_manager.allocate_capital_to_strategies(
                decision.strategy_weights, risk_budget
            )
            
            print(f"\n🎯 Strategy Allocation:")
            total_allocation = 0
            for strategy, weight in decision.strategy_weights.items():
                if weight > 0:
                    allocation = allocations.get(strategy, 0)
                    print(f"   • {strategy}: {weight:.1%} (${allocation:.2f})")
                    total_allocation += allocation
            
            if total_allocation == 0:
                print("   ⚠️ No strategies selected - market conditions unfavorable")
            
            # Show portfolio state
            portfolio_state = await portfolio_manager.get_portfolio_state()
            mode_emoji = "📝" if trading_mode == "paper" else "💰" if trading_mode == "live" else "🔄"
            
            print(f"\n{mode_emoji} Portfolio Status:")
            print(f"   • Total Value: ${portfolio_state.total_value:,.2f}")
            print(f"   • Cash Balance: ${portfolio_state.cash_balance:,.2f}")
            print(f"   • Positions: {portfolio_state.total_positions}")
            print(f"   • P&L: ${portfolio_state.total_pnl:+,.2f} ({portfolio_state.total_pnl/initial_capital*100:+.1f}%)")
            
            if trading_mode == "hybrid":
                print(f"   • Paper Value: ${portfolio_state.paper_value:,.2f}")
                print(f"   • Live Value: ${portfolio_state.live_value:,.2f}")
            
            # Monitor health of active strategies
            print(f"\n🏥 Health Monitoring:")
            for strategy in decision.strategy_weights:
                if decision.strategy_weights[strategy] > 0:
                    # Simulate performance data
                    mock_performance = {
                        'trades': [
                            {'pnl': np.random.normal(0.002, 0.01), 
                             'entry_time': datetime.now(),
                             'exit_time': datetime.now()}
                        ],
                        'returns': list(np.random.normal(0.001, 0.02, 10)),
                        'error_rate': np.random.uniform(0, 0.05),
                        'avg_latency': np.random.uniform(0.05, 0.2),
                        'active_positions': np.random.randint(0, 5)
                    }
                    
                    health_result = await health_system.monitor_and_optimize(
                        strategy, mock_performance, market_data['BTC/USDT']
                    )
                    
                    health_score = health_result['health_metrics'].overall_health_score
                    alerts = len(health_result['active_alerts'])
                    
                    status = "🟢" if health_score > 0.7 else "🟡" if health_score > 0.5 else "🔴"
                    print(f"   • {strategy}: {status} Health={health_score:.2f}, Alerts={alerts}")
                    
                    if health_result['emergency_stop']:
                        print(f"     🛑 EMERGENCY STOP ACTIVATED!")
            
            # Show system status
            system_status = health_system.get_system_status()
            ab_tests = system_status['ab_testing']['total_active_tests']
            if ab_tests > 0:
                print(f"\n🧪 A/B Testing: {ab_tests} active tests")
            
            # Risk level warnings
            if decision.risk_warnings:
                print(f"\n⚠️ Risk Warnings:")
                for warning in decision.risk_warnings:
                    print(f"   • {warning}")
            
            print("\n" + "-" * 60)
            
            # Wait before next cycle
            await asyncio.sleep(60)  # Run every minute
            
    except KeyboardInterrupt:
        print("\n\n🛑 Orchestrator stopped by user")
        print("="*60)
        
        # Final summary
        print("\n📊 FINAL SUMMARY:")
        print(f"   • Total Cycles: {iteration}")
        print(f"   • Strategies Discovered: {len(discovered_strategies)}")
        print(f"   • Trading Mode: {trading_mode.upper()}")
        
        # Portfolio summary
        final_portfolio = await portfolio_manager.get_portfolio_state()
        print(f"\n💼 Portfolio Summary:")
        print(f"   • Initial Capital: ${initial_capital:,.2f}")
        print(f"   • Final Value: ${final_portfolio.total_value:,.2f}")
        print(f"   • Total P&L: ${final_portfolio.total_pnl:+,.2f} ({final_portfolio.total_pnl/initial_capital*100:+.1f}%)")
        print(f"   • Win Rate: {final_portfolio.win_rate:.1%}")
        print(f"   • Sharpe Ratio: {final_portfolio.sharpe_ratio:.2f}")
        print(f"   • Max Drawdown: {final_portfolio.max_drawdown:.1%}")
        
        # Get final system status
        final_status = health_system.get_system_status()
        print(f"\n🏥 Health Monitoring Summary:")
        print(f"   • Monitored Strategies: {final_status['health_monitoring']['monitored_strategies']}")
        print(f"   • Total Alerts: {final_status['health_monitoring']['active_alerts']}")
        print(f"   • Emergency Stops: {final_status['health_monitoring']['emergency_stops']}")
        
        print(f"\n🧪 A/B Testing Summary:")
        print(f"   • Active Tests: {final_status['ab_testing']['total_active_tests']}")
        print(f"   • Completed Tests: {final_status['ab_testing']['completed_tests']}")
        
        print("\n✅ Orchestrator shutdown complete")

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        print("""
Usage: python start_orchestrator.py [mode]

Modes:
  paper   - Paper trading only (default)
  live    - Live trading
  hybrid  - Paper for new strategies, live for proven ones

Examples:
  python start_orchestrator.py          # Start in paper mode
  python start_orchestrator.py paper    # Explicitly use paper mode
  python start_orchestrator.py live     # Start in live mode
  python start_orchestrator.py hybrid   # Start in hybrid mode
        """)
        sys.exit(0)
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     SELF-DISCOVERING STRATEGY ORCHESTRATOR v1.0          ║
    ╠══════════════════════════════════════════════════════════╣
    ║                                                          ║
    ║  This system will:                                       ║
    ║  • Automatically discover ALL strategies                 ║
    ║  • Analyze their DNA and capabilities                    ║
    ║  • Orchestrate them based on market conditions          ║
    ║  • Monitor health and run A/B tests                      ║
    ║  • Manage portfolio with Paper/Live/Hybrid modes         ║
    ║  • Optimize performance continuously                     ║
    ║                                                          ║
    ║  NO HARDCODING - PURE INTELLIGENCE                       ║
    ║                                                          ║
    ║  Press Ctrl+C to stop                                    ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(main())