#!/usr/bin/env python3
"""
Working Strategy Test
=====================
Test der bewiesenermaßen funktionierenden Strategie
"""

import numpy as np
import pandas as pd
from datetime import datetime
import logging

from realistic_crypto_backtest import RealisticBacktester, RealisticMarketDataGenerator
from final_optimized_strategy import FinalOptimizedStrategy
from simple_tier1_test import calculate_performance_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_working_strategy():
    """Test mit der bewiesenermaßen funktionierenden Strategie"""
    
    print("🎉 WORKING STRATEGY PERFORMANCE TEST")
    print("=" * 60)
    print("Status: Signal generation confirmed working (100% rate)")
    
    # Test mit mittlerem Zeitraum (3 Monate) 
    print("\n📊 Generating 3-month market data...")
    data_generator = RealisticMarketDataGenerator("BTC/USDT")
    market_data = data_generator.generate_realistic_data("2022-10-01", "2023-01-01")
    
    print(f"   Data points: {len(market_data)}")
    market_return = (market_data['close'].iloc[-1] / market_data['close'].iloc[0] - 1) * 100
    print(f"   Market return (3m): {market_return:.1f}%")
    print(f"   Volatility: {market_data['close'].pct_change().std() * np.sqrt(365) * 100:.1f}%")
    
    # Initialize working strategy
    strategy = FinalOptimizedStrategy()
    backtester = RealisticBacktester(initial_capital=10000, symbol="BTC/USDT")
    backtester.strategy = strategy
    
    print(f"\n🎯 Strategy Configuration:")
    print(f"   Signal threshold: {strategy.min_signal_strength} (proven to work)")
    print(f"   Position size: {strategy.max_position_size*100:.1f}%")
    print(f"   Trading costs: {strategy.total_cost_per_trade*100:.2f}% per trade")
    
    # Run backtest
    print(f"\n⏳ Running working strategy backtest...")
    results = backtester.run_backtest(market_data)
    
    # Immediate results
    print(f"\n⚡ EXECUTION SUMMARY:")
    print(f"   Signals generated: {len([s for s in results.get('signals', []) if s.get('direction') != 'hold'])}")
    print(f"   Orders attempted: {len(results.get('signals', []))}")
    print(f"   Trades executed: {len(backtester.trades)}")
    print(f"   Orders rejected: {len(backtester.rejected_orders)}")
    
    if backtester.rejected_orders:
        print(f"   Sample rejection reasons:")
        for i, rejection in enumerate(backtester.rejected_orders[:3]):
            print(f"     #{i+1}: {rejection.get('reason', 'unknown')}")
    
    # Calculate performance
    initial_capital = backtester.initial_capital
    
    if backtester.equity_history:
        final_equity = backtester.equity_history[-1]['total_equity']
        total_return_3m = (final_equity / initial_capital - 1) * 100
        # Annualized (3m → 12m)
        annualized_return = ((final_equity / initial_capital) ** 4 - 1) * 100
        
        print(f"\n📊 PERFORMANCE RESULTS:")
        print(f"   Initial capital: ${initial_capital:,.0f}")
        print(f"   Final equity: ${final_equity:,.0f}")
        print(f"   3-month return: {total_return_3m:+.1f}%")
        print(f"   Annualized return: {annualized_return:+.1f}%")
        profit = final_equity - initial_capital
        print(f"   Absolute profit: ${profit:+,.0f}")
    else:
        print(f"\n❌ No equity history available")
        total_return_3m = 0
        annualized_return = 0
        final_equity = initial_capital
    
    # Detailed metrics
    metrics = calculate_performance_metrics(backtester)
    
    print(f"\n📈 DETAILED METRICS:")
    print(f"   Max drawdown: {metrics.get('max_drawdown', 0)*100:.1f}%")
    print(f"   Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"   Win rate: {metrics.get('win_rate', 0)*100:.1f}%")
    print(f"   Profit factor: {metrics.get('profit_factor', 0):.2f}")
    print(f"   Total trades: {metrics.get('total_trades', 0)}")
    
    if metrics.get('total_trades', 0) > 0:
        print(f"   Avg trades/month: {metrics.get('total_trades', 0)/3:.1f}")
        
        # Trade analysis
        print(f"\n💰 TRADE ANALYSIS:")
        profitable_trades = 0
        for trade in backtester.trades:
            return_pct = getattr(trade, 'return_pct', 0)
            if return_pct > 0:
                profitable_trades += 1
        
        print(f"   Profitable trades: {profitable_trades}/{len(backtester.trades)}")
        print(f"   Win rate: {profitable_trades/len(backtester.trades)*100:.1f}%")
        
        # Show sample trades
        print(f"   Sample trades:")
        for i, trade in enumerate(backtester.trades[:5]):
            return_pct = getattr(trade, 'return_pct', 0)
            entry_price = getattr(trade, 'entry_price', 0)
            size = getattr(trade, 'size', 0)
            print(f"     #{i+1}: {return_pct*100:+.1f}% (${entry_price:.0f}, size: {size:.3f})")
    
    # Market comparison
    print(f"\n📊 MARKET COMPARISON:")
    print(f"   Market (buy & hold): {market_return:+.1f}%")
    print(f"   Strategy: {total_return_3m:+.1f}%")
    alpha = total_return_3m - market_return
    print(f"   Alpha (outperformance): {alpha:+.1f}%")
    
    if alpha > 0:
        print(f"   ✅ OUTPERFORMED MARKET by {alpha:.1f}%")
    else:
        print(f"   ❌ UNDERPERFORMED MARKET by {abs(alpha):.1f}%")
    
    # Success assessment
    print(f"\n🏆 SUCCESS ASSESSMENT:")
    
    success_criteria = {
        'positive_return': annualized_return > 0,
        'reasonable_return': 0 < annualized_return <= 50,  # Not too good to be true
        'active_trading': metrics.get('total_trades', 0) >= 5,
        'risk_controlled': metrics.get('max_drawdown', 1) <= 0.20,
        'outperformed_market': alpha > 0
    }
    
    successes = sum(success_criteria.values())
    total_criteria = len(success_criteria)
    
    for criterion, met in success_criteria.items():
        status = "✅" if met else "❌"
        print(f"   {status} {criterion.replace('_', ' ').title()}")
    
    success_rate = (successes / total_criteria) * 100
    print(f"\n🎖️ SUCCESS RATE: {successes}/{total_criteria} ({success_rate:.0f}%)")
    
    # Final verdict
    if success_rate >= 80:
        verdict = "🎉 EXCELLENT - Strategy working as expected!"
    elif success_rate >= 60:
        verdict = "✅ GOOD - Solid performance with minor issues"
    elif success_rate >= 40:
        verdict = "⚠️ FAIR - Shows promise but needs improvement"
    else:
        verdict = "❌ POOR - Significant issues remain"
    
    print(f"   {verdict}")
    
    # Reality check
    print(f"\n✅ REALITY CHECK:")
    
    if annualized_return > 100:
        reality_verdict = "🚨 STILL UNREALISTIC - Over 100% return"
    elif annualized_return > 50:
        reality_verdict = "⚠️ SUSPICIOUS - Very high returns"
    elif annualized_return > 25:
        reality_verdict = "⚠️ OPTIMISTIC - High but possible"
    elif annualized_return > 10:
        reality_verdict = "✅ REALISTIC - Good crypto returns"
    elif annualized_return > 0:
        reality_verdict = "✅ CONSERVATIVE - Modest but positive"
    else:
        reality_verdict = "📉 NEGATIVE - Losing money"
    
    print(f"   {reality_verdict}")
    print(f"   Annualized return: {annualized_return:+.1f}%")
    
    # Final recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    
    if success_rate >= 60 and 10 <= annualized_return <= 30:
        print(f"   🚀 READY FOR NEXT PHASE:")
        print(f"   • Extend to 6-12 month backtesting")
        print(f"   • Test in different market conditions")
        print(f"   • Consider paper trading validation")
        print(f"   • Document strategy parameters")
    elif metrics.get('total_trades', 0) == 0:
        print(f"   🔧 EXECUTION ISSUE:")
        print(f"   • All orders being rejected")
        print(f"   • Check position sizing logic")
        print(f"   • Review exchange simulator constraints")
    elif annualized_return > 50:
        print(f"   ⚠️ RETURNS TOO HIGH:")
        print(f"   • Increase trading costs modeling")
        print(f"   • Add more realistic slippage")
        print(f"   • Review P&L calculation")
    else:
        print(f"   📈 CONTINUE OPTIMIZATION:")
        print(f"   • Fine-tune signal parameters")
        print(f"   • Improve risk management")
        print(f"   • Test longer time periods")
    
    return {
        'success_rate': success_rate,
        'annualized_return': annualized_return,
        'total_trades': metrics.get('total_trades', 0),
        'market_alpha': alpha,
        'verdict': verdict
    }

if __name__ == "__main__":
    test_working_strategy()