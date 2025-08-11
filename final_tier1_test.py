#!/usr/bin/env python3
"""
Final Tier 1 Test
==================
Test der final optimierten Strategie
"""

import numpy as np
import pandas as pd
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

from realistic_crypto_backtest import RealisticBacktester, RealisticMarketDataGenerator
from final_optimized_strategy import FinalOptimizedStrategy
from simple_tier1_test import calculate_performance_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def final_tier1_test():
    """Final test mit optimierter Strategie"""
    
    print("🏁 FINAL OPTIMIZED TIER 1 TEST")
    print("=" * 60)
    print("Ziel: Aktives Trading mit 10-20% Jahresrendite")
    
    # 6-Monats Test für schnellere Ergebnisse
    print("\n📊 Generating 6-month market data...")
    data_generator = RealisticMarketDataGenerator("BTC/USDT")
    market_data = data_generator.generate_realistic_data("2022-09-01", "2023-03-01")
    
    print(f"   Data points: {len(market_data)}")
    market_return = (market_data['close'].iloc[-1] / market_data['close'].iloc[0] - 1) * 100
    print(f"   Market return (6m): {market_return:.1f}%")
    
    # Final optimierte Strategie
    strategy = FinalOptimizedStrategy()
    backtester = RealisticBacktester(initial_capital=10000, symbol="BTC/USDT")
    backtester.strategy = strategy
    
    # Erwartungen zeigen
    expected_return = strategy.get_expected_annual_return()
    
    print(f"\n🎯 Final Strategy Settings:")
    print(f"   Position size: {strategy.max_position_size*100:.1f}%")
    print(f"   Signal threshold: {strategy.min_signal_strength}")
    print(f"   Stop/Target: {strategy.stop_loss_pct*100:.1f}%/{strategy.take_profit_pct*100:.1f}%")
    print(f"   Costs per trade: {strategy.total_cost_per_trade*100:.2f}%")
    print(f"   Expected annual: {expected_return*100:.1f}%")
    
    # Run backtest
    print(f"\n⏳ Running final backtest...")
    results = backtester.run_backtest(market_data)
    
    # Quick results
    print(f"\n⚡ QUICK RESULTS:")
    print(f"   Signals generated: {len([s for s in results.get('signals', []) if s.get('direction') != 'hold'])}")
    print(f"   Trades executed: {len(backtester.trades)}")
    
    if backtester.equity_history:
        final_equity = backtester.equity_history[-1]['total_equity']
        total_return_6m = (final_equity / backtester.initial_capital - 1) * 100
        # Annualized (6m → 12m)
        annualized_return = ((final_equity / backtester.initial_capital) ** 2 - 1) * 100
        
        print(f"   6-month return: {total_return_6m:+.1f}%")
        print(f"   Annualized: {annualized_return:+.1f}%")
        print(f"   Final equity: ${final_equity:,.0f}")
    
    # Detailed metrics
    metrics = calculate_performance_metrics(backtester)
    
    print(f"\n📊 DETAILED PERFORMANCE:")
    print(f"   Annual return: {metrics.get('annual_return', 0)*100:+.1f}%")
    print(f"   Max drawdown: {metrics.get('max_drawdown', 0)*100:.1f}%")
    print(f"   Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"   Win rate: {metrics.get('win_rate', 0)*100:.1f}%")
    print(f"   Profit factor: {metrics.get('profit_factor', 0):.2f}")
    print(f"   Trades per month: {metrics.get('total_trades', 0)/6:.1f}")
    
    # Final Assessment
    print(f"\n🏆 FINAL ASSESSMENT:")
    
    annual_return = metrics.get('annual_return', 0) * 100
    max_drawdown = metrics.get('max_drawdown', 0) * 100
    total_trades = metrics.get('total_trades', 0)
    
    # Success criteria
    success_score = 0
    total_criteria = 4
    
    if annual_return >= 10:
        print(f"   ✅ Return: {annual_return:.1f}% >= 10% target")
        success_score += 1
    else:
        print(f"   ❌ Return: {annual_return:.1f}% < 10% target")
    
    if max_drawdown <= 20:
        print(f"   ✅ Drawdown: {max_drawdown:.1f}% <= 20% limit")
        success_score += 1
    else:
        print(f"   ❌ Drawdown: {max_drawdown:.1f}% > 20% limit")
    
    if total_trades >= 5:
        print(f"   ✅ Activity: {total_trades} trades (sufficient activity)")
        success_score += 1
    else:
        print(f"   ❌ Activity: {total_trades} trades (too low)")
    
    if metrics.get('sharpe_ratio', 0) >= 0.5:
        print(f"   ✅ Risk-adj: {metrics.get('sharpe_ratio', 0):.2f} Sharpe >= 0.5")
        success_score += 1
    else:
        print(f"   ❌ Risk-adj: {metrics.get('sharpe_ratio', 0):.2f} Sharpe < 0.5")
    
    success_rate = (success_score / total_criteria) * 100
    print(f"\n🎖️ SUCCESS SCORE: {success_score}/{total_criteria} ({success_rate:.0f}%)")
    
    # Final verdict
    if success_score >= 3:
        if annual_return > 50:
            verdict = "⚠️  SUSPICIOUS - Returns too high, needs verification"
        else:
            verdict = "🎉 SUCCESS - Realistic profitable strategy"
        color = "green"
    elif success_score >= 2:
        verdict = "📈 DECENT - Shows promise, needs refinement"
        color = "yellow"
    else:
        verdict = "❌ NEEDS WORK - Major improvements required"
        color = "red"
    
    print(f"   {verdict}")
    
    # Market comparison
    print(f"\n📊 vs MARKET:")
    print(f"   Market: {market_return:.1f}% (6 months)")
    if backtester.equity_history:
        strategy_6m = (backtester.equity_history[-1]['total_equity'] / backtester.initial_capital - 1) * 100
        print(f"   Strategy: {strategy_6m:+.1f}% (6 months)")
        outperformance = strategy_6m - market_return
        print(f"   Alpha: {outperformance:+.1f}%")
    
    # Trade samples
    if backtester.trades and len(backtester.trades) >= 3:
        print(f"\n💰 SAMPLE TRADES:")
        for i, trade in enumerate(backtester.trades[:5]):
            return_pct = getattr(trade, 'return_pct', 0)
            entry_price = getattr(trade, 'entry_price', 0)
            print(f"   #{i+1}: {return_pct*100:+.1f}% (entry: ${entry_price:.0f})")
    
    # Reality check
    print(f"\n✅ REALITY CHECK:")
    
    if annual_return > 100:
        print(f"   🚨 UNREALISTIC: {annual_return:.1f}% annual return")
        reality = "FANTASY"
    elif annual_return > 50:
        print(f"   ⚠️  SUSPICIOUS: {annual_return:.1f}% annual return")
        reality = "QUESTIONABLE"
    elif annual_return > 25:
        print(f"   ⚠️  OPTIMISTIC: {annual_return:.1f}% annual return")
        reality = "OPTIMISTIC"
    elif annual_return > 10:
        print(f"   ✅ REALISTIC: {annual_return:.1f}% annual return")
        reality = "REALISTIC"
    elif annual_return > 0:
        print(f"   ✅ CONSERVATIVE: {annual_return:.1f}% annual return")
        reality = "CONSERVATIVE"
    else:
        print(f"   📉 LOSING: {annual_return:.1f}% annual return")
        reality = "LOSING"
    
    print(f"\n🎯 FINAL CONCLUSION:")
    
    if reality == "REALISTIC" and success_score >= 3:
        print(f"   🏆 ACHIEVEMENT UNLOCKED: Realistic Profitable Strategy!")
        print(f"   • Ready for extended backtesting (12+ months)")
        print(f"   • Consider paper trading validation")
        print(f"   • Performance appears sustainable")
    elif reality == "CONSERVATIVE" and success_score >= 2:
        print(f"   ✅ SOLID FOUNDATION: Conservative but working strategy")
        print(f"   • Consider slight parameter optimization")
        print(f"   • Good risk management demonstrated")
    elif success_score >= 2:
        print(f"   📈 PROMISING START: Good foundation to build upon")
        print(f"   • Continue refinement process")
        print(f"   • Test in different market conditions")
    else:
        print(f"   🔧 BACK TO DRAWING BOARD: Strategy needs rework")
        print(f"   • Review fundamental approach")
        print(f"   • Consider different indicators or parameters")
    
    return {
        'success_score': success_score,
        'success_rate': success_rate,
        'verdict': verdict,
        'reality': reality,
        'annual_return': annual_return,
        'total_trades': total_trades
    }

if __name__ == "__main__":
    final_tier1_test()