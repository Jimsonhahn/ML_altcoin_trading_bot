#!/usr/bin/env python3
"""
Goldilocks Strategy Backtest
=============================
Test der "just right" Strategy für $25k Profit Target
"""

import numpy as np
import pandas as pd
from datetime import datetime
import logging

from realistic_crypto_backtest import RealisticBacktester, RealisticMarketDataGenerator
from goldilocks_strategy import GoldilocksStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_goldilocks_strategy():
    """Test Goldilocks Strategy with 3-year projection"""
    
    print("🎯 GOLDILOCKS STRATEGY BACKTEST")
    print("=" * 60)
    print("TARGET: $10,000 → $35,000 (3 years)")
    print("REQUIRED: 28% annually")
    
    # Generate realistic 6-month test data
    print("\n📊 Generating market data...")
    data_generator = RealisticMarketDataGenerator("BTC/USDT")
    market_data = data_generator.generate_realistic_data("2022-06-01", "2022-12-01")
    
    print(f"   Data points: {len(market_data)}")
    market_return = (market_data['close'].iloc[-1] / market_data['close'].iloc[0] - 1) * 100
    print(f"   Market return (6m): {market_return:.1f}%")
    
    # Initialize Goldilocks Strategy
    strategy = GoldilocksStrategy()
    backtester = RealisticBacktester(initial_capital=10000, symbol="BTC/USDT")
    
    print(f"\n🎯 GOLDILOCKS CONFIGURATION:")
    info = strategy.get_strategy_info()
    key_params = ['target_annual_return', 'max_position_size', 'max_leverage', 'signal_threshold', 
                  'stop_loss', 'take_profit', 'risk_reward_ratio', 'max_daily_trades']
    
    for param in key_params:
        if param in info:
            print(f"   {param}: {info[param]}")
    
    # Run backtest
    print(f"\n⏳ Running Goldilocks backtest...")
    backtester.strategy = strategy
    results = backtester.run_backtest(market_data)
    
    # Results analysis
    initial_capital = backtester.initial_capital
    
    print(f"\n📊 EXECUTION SUMMARY:")
    signals_generated = len([s for s in results.get('signals', []) if s.get('direction') != 'hold'])
    print(f"   Signals generated: {signals_generated}")
    print(f"   Trades executed: {len(backtester.trades)}")
    print(f"   Orders rejected: {len(backtester.rejected_orders)}")
    
    if backtester.rejected_orders:
        rejection_reasons = {}
        for rejection in backtester.rejected_orders:
            reason = rejection.get('reason', 'unknown')
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        
        print(f"   Rejection breakdown:")
        for reason, count in rejection_reasons.items():
            print(f"     {reason}: {count}")
    
    # Performance calculation
    if backtester.equity_history:
        final_equity = backtester.equity_history[-1]['total_equity']
        return_6m = (final_equity / initial_capital - 1) * 100
        annualized_return = ((final_equity / initial_capital) ** 2 - 1) * 100  # 6m → 12m
        
        print(f"\n💰 PERFORMANCE RESULTS:")
        print(f"   Initial capital: ${initial_capital:,.0f}")
        print(f"   Final equity: ${final_equity:,.0f}")
        print(f"   6-month return: {return_6m:+.1f}%")
        print(f"   Annualized return: {annualized_return:+.1f}%")
        print(f"   Absolute profit: ${final_equity - initial_capital:+,.0f}")
        
    else:
        print(f"\n❌ No performance data available")
        return_6m = 0
        annualized_return = 0
        final_equity = initial_capital
    
    # Trade analysis
    if backtester.trades:
        print(f"\n📈 TRADE ANALYSIS:")
        profitable_trades = 0
        total_profit = 0
        total_loss = 0
        
        for trade in backtester.trades:
            pnl = getattr(trade, 'pnl', 0)
            if pnl > 0:
                profitable_trades += 1
                total_profit += pnl
            else:
                total_loss += abs(pnl)
        
        win_rate = profitable_trades / len(backtester.trades)
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        avg_trades_per_month = len(backtester.trades) / 6
        
        print(f"   Total trades: {len(backtester.trades)}")
        print(f"   Win rate: {win_rate*100:.1f}%")
        print(f"   Profit factor: {profit_factor:.2f}")
        print(f"   Avg trades/month: {avg_trades_per_month:.1f}")
        print(f"   Total profit: ${total_profit:.0f}")
        print(f"   Total losses: ${total_loss:.0f}")
        
        # Sample trades
        print(f"\n📝 SAMPLE TRADES:")
        for i, trade in enumerate(backtester.trades[:5]):
            entry_price = getattr(trade, 'entry_price', 0)
            exit_price = getattr(trade, 'exit_price', entry_price)
            size = getattr(trade, 'size', 0)
            pnl = getattr(trade, 'pnl', 0)
            return_pct = (exit_price / entry_price - 1) * 100 if entry_price > 0 else 0
            
            print(f"     #{i+1}: {return_pct:+.1f}% (${pnl:+.0f}) - Size: {size:.3f}")
    
    # Risk metrics (simplified)
    if backtester.equity_history:
        equity_values = [e['total_equity'] for e in backtester.equity_history]
        peak = initial_capital
        max_drawdown = 0
        
        for equity in equity_values:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Volatility (simplified)
        equity_series = pd.Series(equity_values)
        daily_returns = equity_series.pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(365) if len(daily_returns) > 1 else 0
        
        # Sharpe ratio (assume 2% risk-free rate)
        excess_return = (annualized_return / 100) - 0.02
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        print(f"\n⚖️ RISK METRICS:")
        print(f"   Max drawdown: {max_drawdown*100:.1f}%")
        print(f"   Volatility: {volatility*100:.1f}%")
        print(f"   Sharpe ratio: {sharpe_ratio:.2f}")
    
    # Target assessment
    print(f"\n🎯 TARGET ASSESSMENT:")
    target_annual = 28.0
    
    if annualized_return >= target_annual * 0.8:  # Within 80% of target
        target_status = "✅ ON TRACK"
    elif annualized_return >= target_annual * 0.6:  # Within 60% of target
        target_status = "⚠️ BELOW TARGET"
    elif annualized_return > 0:
        target_status = "❌ SIGNIFICANTLY BELOW"
    else:
        target_status = "💀 LOSING MONEY"
    
    print(f"   Target: {target_annual:.0f}% annually")
    print(f"   Actual: {annualized_return:.1f}% annually")
    print(f"   Status: {target_status}")
    
    # 3-year projection
    if annualized_return > 0:
        projected_3y_capital = initial_capital * ((1 + annualized_return/100) ** 3)
        projected_profit = projected_3y_capital - initial_capital
        
        print(f"\n🔮 3-YEAR PROJECTION:")
        print(f"   Current rate: {annualized_return:.1f}% annually")
        print(f"   Projected capital: ${projected_3y_capital:,.0f}")
        print(f"   Projected profit: ${projected_profit:+,.0f}")
        
        if projected_profit >= 25000:
            print(f"   🎉 TARGET ACHIEVED! (${projected_profit:,.0f} >= $25,000)")
        elif projected_profit >= 20000:
            print(f"   🎯 CLOSE TO TARGET (${projected_profit:,.0f})")
        elif projected_profit >= 10000:
            print(f"   ⚠️ BELOW TARGET (${projected_profit:,.0f})")
        else:
            print(f"   ❌ FAR FROM TARGET (${projected_profit:,.0f})")
    
    # Reality check
    print(f"\n✅ REALITY CHECK:")
    
    reality_flags = []
    if annualized_return > 50:
        reality_flags.append("❌ Returns too high (>50%)")
    if max_drawdown > 0.5:
        reality_flags.append("❌ Drawdown too high (>50%)")
    if len(backtester.trades) == 0:
        reality_flags.append("❌ No trades executed")
    if len(backtester.trades) > 1000:
        reality_flags.append("❌ Too many trades (overtrading)")
    if win_rate < 0.3:
        reality_flags.append("❌ Win rate too low (<30%)")
    
    if not reality_flags:
        print("   ✅ All reality checks passed!")
        print("   ✅ Returns are believable")
        print("   ✅ Risk metrics are reasonable")
        print("   ✅ Trading activity is appropriate")
    else:
        print("   Reality check issues:")
        for flag in reality_flags:
            print(f"     {flag}")
    
    # Final verdict
    print(f"\n🏆 GOLDILOCKS VERDICT:")
    
    success_score = 0
    total_criteria = 5
    
    # Criterion 1: Profitable
    if annualized_return > 0:
        success_score += 1
        print(f"   ✅ Profitable ({annualized_return:.1f}% > 0%)")
    else:
        print(f"   ❌ Losing money ({annualized_return:.1f}%)")
    
    # Criterion 2: Target achievement
    if annualized_return >= target_annual * 0.8:
        success_score += 1
        print(f"   ✅ Target achieved (≥80% of 28%)")
    else:
        print(f"   ❌ Below target (<80% of 28%)")
    
    # Criterion 3: Reasonable risk
    if max_drawdown <= 0.25:
        success_score += 1
        print(f"   ✅ Risk controlled (DD ≤25%)")
    else:
        print(f"   ❌ Risk too high (DD >25%)")
    
    # Criterion 4: Active trading
    if len(backtester.trades) >= 10:
        success_score += 1
        print(f"   ✅ Active trading ({len(backtester.trades)} trades)")
    else:
        print(f"   ❌ Insufficient activity ({len(backtester.trades)} trades)")
    
    # Criterion 5: Win rate
    if win_rate >= 0.45:
        success_score += 1
        print(f"   ✅ Good win rate ({win_rate*100:.1f}%)")
    else:
        print(f"   ❌ Low win rate ({win_rate*100:.1f}%)")
    
    success_rate = (success_score / total_criteria) * 100
    print(f"\n📊 SUCCESS RATE: {success_score}/{total_criteria} ({success_rate:.0f}%)")
    
    if success_rate >= 80:
        final_verdict = "🎉 EXCELLENT - Goldilocks found the sweet spot!"
    elif success_rate >= 60:
        final_verdict = "✅ GOOD - Close to the perfect balance"
    elif success_rate >= 40:
        final_verdict = "⚠️ NEEDS TUNING - Almost there"
    else:
        final_verdict = "❌ BACK TO DRAWING BOARD"
    
    print(f"   {final_verdict}")
    
    return {
        'annualized_return': annualized_return,
        'target_achievement': annualized_return / target_annual if target_annual > 0 else 0,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': len(backtester.trades),
        'success_rate': success_rate,
        'projected_3y_profit': projected_profit if annualized_return > 0 else 0,
        'verdict': final_verdict
    }

if __name__ == "__main__":
    results = test_goldilocks_strategy()
    
    print(f"\n💰 BOTTOM LINE:")
    if results['projected_3y_profit'] >= 25000:
        print(f"🎯 TARGET HIT! Projected ${results['projected_3y_profit']:,.0f} profit")
    else:
        print(f"🎯 Gap to close: ${25000 - results['projected_3y_profit']:,.0f} short")