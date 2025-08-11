#!/usr/bin/env python3
"""
Honest Performance Test
=======================
Test mit ultra-konservativen Parametern für ehrliche Ergebnisse
"""

import numpy as np
import pandas as pd
from datetime import datetime
import logging

from realistic_crypto_backtest import RealisticBacktester, RealisticMarketDataGenerator
from corrected_realistic_strategy import CorrectedRealisticStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_honest_performance():
    """Test mit ehrlichen, konservativen Erwartungen"""
    
    print("💯 HONEST PERFORMANCE TEST")
    print("=" * 50)
    print("Ziel: Realistische 10-25% Jahresrendite")
    
    # 6-Monats Test
    print("\n📊 Generating 6-month market data...")
    data_generator = RealisticMarketDataGenerator("BTC/USDT")
    market_data = data_generator.generate_realistic_data("2023-01-01", "2023-07-01")
    
    print(f"   Data points: {len(market_data)}")
    market_return = (market_data['close'].iloc[-1] / market_data['close'].iloc[0] - 1) * 100
    print(f"   Market return: {market_return:.1f}%")
    
    # Ultra-konservative Strategie
    strategy = CorrectedRealisticStrategy()
    backtester = RealisticBacktester(initial_capital=10000, symbol="BTC/USDT")
    backtester.strategy = strategy
    
    print(f"\n🛡️  Conservative Strategy Settings:")
    print(f"   Max position size: {strategy.max_position_size*100:.1f}%")
    print(f"   Min signal strength: {strategy.min_signal_strength}")
    print(f"   Stop loss: {strategy.stop_loss_pct*100:.1f}%")
    print(f"   Take profit: {strategy.take_profit_pct*100:.1f}% (R/R = {strategy.take_profit_pct/strategy.stop_loss_pct:.1f}:1)")
    print(f"   Max daily trades: {strategy.max_daily_trades}")
    print(f"   Volume threshold: {strategy.volume_multiplier}x")
    
    # Run backtest
    print(f"\n⏳ Running honest backtest...")
    results = backtester.run_backtest(market_data)
    
    # Calculate results
    initial_capital = backtester.initial_capital
    
    if backtester.equity_history:
        final_equity = backtester.equity_history[-1]['total_equity']
        total_return = (final_equity / initial_capital - 1) * 100
        
        # Annualized return (6 months -> 12 months)
        monthly_return = (final_equity / initial_capital) ** (1/6) - 1
        annual_return = ((1 + monthly_return) ** 12 - 1) * 100
        
        # Max drawdown
        peak = initial_capital
        max_drawdown = 0
        for point in backtester.equity_history:
            equity = point['total_equity']
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
    else:
        final_equity = initial_capital
        total_return = 0
        annual_return = 0
        max_drawdown = 0
    
    # Results
    print(f"\n📊 HONEST RESULTS:")
    print(f"   Period: 6 months")
    print(f"   Initial capital: ${initial_capital:,.0f}")
    print(f"   Final equity: ${final_equity:,.0f}")
    print(f"   Total return: {total_return:+.1f}%")
    print(f"   Monthly return: {((final_equity/initial_capital)**(1/6)-1)*100:+.1f}%")
    print(f"   Annualized return: {annual_return:+.1f}%")
    print(f"   Max drawdown: {max_drawdown*100:.1f}%")
    print(f"   Total trades: {len(backtester.trades)}")
    print(f"   Avg trades/month: {len(backtester.trades)/6:.1f}")
    
    # Trade analysis
    if backtester.trades:
        profitable = sum(1 for trade in backtester.trades if getattr(trade, 'return_pct', 0) > 0)
        win_rate = profitable / len(backtester.trades) * 100
        print(f"   Win rate: {win_rate:.1f}%")
        
        # Show first few trades
        print(f"\n💰 Sample Trades:")
        for i, trade in enumerate(backtester.trades[:5]):
            return_pct = getattr(trade, 'return_pct', 0) * 100
            print(f"   Trade {i+1}: {return_pct:+.1f}%")
    
    # Realism assessment
    print(f"\n✅ REALISM ASSESSMENT:")
    
    if annual_return > 100:
        print(f"   ❌ STILL UNREALISTIC: {annual_return:.1f}% annual return")
        verdict = "STILL TOO GOOD"
    elif annual_return > 50:
        print(f"   ⚠️  SUSPICIOUS: {annual_return:.1f}% annual return is very high")
        verdict = "NEEDS MORE TESTING"
    elif annual_return > 25:
        print(f"   ⚠️  OPTIMISTIC: {annual_return:.1f}% annual return is high but possible")
        verdict = "CAUTIOUSLY OPTIMISTIC"
    elif annual_return > 10:
        print(f"   ✅ REALISTIC: {annual_return:.1f}% annual return is achievable")
        verdict = "REALISTIC"
    elif annual_return > 0:
        print(f"   ✅ CONSERVATIVE: {annual_return:.1f}% annual return is modest")
        verdict = "CONSERVATIVE"
    else:
        print(f"   📉 LOSS: {annual_return:.1f}% annual return")
        verdict = "LOSING STRATEGY"
    
    if max_drawdown > 0.20:
        print(f"   ❌ HIGH RISK: {max_drawdown*100:.1f}% max drawdown is too high")
    elif max_drawdown > 0.10:
        print(f"   ⚠️  MODERATE RISK: {max_drawdown*100:.1f}% max drawdown")
    else:
        print(f"   ✅ LOW RISK: {max_drawdown*100:.1f}% max drawdown")
    
    if len(backtester.trades) == 0:
        print(f"   ⚠️  NO TRADES: Strategy too restrictive")
    elif len(backtester.trades) < 5:
        print(f"   ⚠️  FEW TRADES: Only {len(backtester.trades)} trades in 6 months")
    else:
        print(f"   ✅ ACTIVE: {len(backtester.trades)} trades executed")
    
    print(f"\n🏆 FINAL VERDICT: {verdict}")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    if annual_return > 50:
        print(f"   • Increase trading costs and slippage")
        print(f"   • Reduce position sizes further") 
        print(f"   • Add more market impact modeling")
        print(f"   • Test over longer periods (2+ years)")
    elif annual_return > 25:
        print(f"   • Extend testing to full market cycle")
        print(f"   • Test in bear market conditions")
        print(f"   • Consider paper trading validation")
    elif annual_return > 10:
        print(f"   • Results appear realistic for crypto")
        print(f"   • Ready for extended backtesting")
        print(f"   • Consider live paper trading")
    else:
        print(f"   • Strategy may be too conservative")
        print(f"   • Consider slight parameter relaxation")
    
    return {
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'total_trades': len(backtester.trades),
        'verdict': verdict
    }

if __name__ == "__main__":
    test_honest_performance()