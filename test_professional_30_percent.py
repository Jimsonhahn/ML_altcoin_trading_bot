#!/usr/bin/env python3
"""
Professional 30% Strategy Backtest
===================================
Professional-grade backtest für realistische 30% Returns
"""

import numpy as np
import pandas as pd
from datetime import datetime
import logging

from realistic_crypto_backtest import RealisticBacktester, RealisticMarketDataGenerator
from professional_30_percent_strategy import Professional30PercentStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_professional_30_percent_strategy():
    """Test Professional 30% Strategy with realistic expectations"""
    
    print("🏆 PROFESSIONAL 30% STRATEGY BACKTEST")
    print("=" * 80)
    print("Based on: Renaissance Technologies + Two Sigma Principles")
    print("Target: 30% annually (2.5% monthly average)")
    print("Approach: Multi-Edge + Regime-Aware + Professional Risk Management")
    
    # Generate 8-month test data (more comprehensive)
    print("\n📊 Generating comprehensive market data...")
    data_generator = RealisticMarketDataGenerator("BTC/USDT")
    market_data = data_generator.generate_realistic_data("2022-04-01", "2022-12-01")  # 8 months
    
    print(f"   Data points: {len(market_data):,}")
    market_return = (market_data['close'].iloc[-1] / market_data['close'].iloc[0] - 1) * 100
    market_volatility = market_data['close'].pct_change().std() * np.sqrt(365) * 100
    print(f"   Market return (8m): {market_return:+.1f}%")
    print(f"   Market volatility: {market_volatility:.1f}%")
    
    # Initialize Professional Strategy
    strategy = Professional30PercentStrategy()
    backtester = RealisticBacktester(initial_capital=10000, symbol="BTC/USDT")
    
    print(f"\n🎯 PROFESSIONAL STRATEGY CONFIG:")
    info = strategy.get_strategy_info()
    key_metrics = [
        'target_annual_return', 'target_monthly_return', 'max_risk_per_trade',
        'max_position_size', 'target_win_rate', 'target_profit_factor', 
        'max_drawdown_limit', 'trades_per_month'
    ]
    
    for metric in key_metrics:
        if metric in info:
            print(f"   {metric}: {info[metric]}")
    
    # Run professional backtest
    print(f"\n⏳ Running professional-grade backtest...")
    backtester.strategy = strategy
    results = backtester.run_backtest(market_data)
    
    # Detailed Analysis
    initial_capital = backtester.initial_capital
    
    print(f"\n📊 EXECUTION ANALYSIS:")
    signals_generated = len([s for s in results.get('signals', []) if s.get('direction') != 'hold'])
    signals_total = len(results.get('signals', []))
    signal_quality = (signals_generated / signals_total * 100) if signals_total > 0 else 0
    
    print(f"   Total signal evaluations: {signals_total:,}")
    print(f"   Buy/Sell signals generated: {signals_generated}")
    print(f"   Signal quality ratio: {signal_quality:.1f}%")
    print(f"   Trades executed: {len(backtester.trades)}")
    print(f"   Orders rejected: {len(backtester.rejected_orders)}")
    
    if backtester.rejected_orders:
        rejection_analysis = {}
        for rejection in backtester.rejected_orders:
            reason = rejection.get('reason', 'unknown')
            rejection_analysis[reason] = rejection_analysis.get(reason, 0) + 1
        
        print(f"   Top rejection reasons:")
        sorted_rejections = sorted(rejection_analysis.items(), key=lambda x: x[1], reverse=True)
        for reason, count in sorted_rejections[:3]:
            print(f"     {reason}: {count} ({count/len(backtester.rejected_orders)*100:.1f}%)")
    
    # Performance Calculation
    if backtester.equity_history:
        final_equity = backtester.equity_history[-1]['total_equity']
        return_8m = (final_equity / initial_capital - 1) * 100
        annualized_return = ((final_equity / initial_capital) ** (12/8) - 1) * 100  # 8m → 12m
        
        print(f"\n💰 PERFORMANCE RESULTS:")
        print(f"   Initial capital: ${initial_capital:,.0f}")
        print(f"   Final equity: ${final_equity:,.0f}")
        print(f"   8-month return: {return_8m:+.1f}%")
        print(f"   Annualized return: {annualized_return:+.1f}%")
        print(f"   Absolute profit: ${final_equity - initial_capital:+,.0f}")
        
    else:
        print(f"\n❌ No performance data available - Strategy may need debugging")
        return_8m = 0
        annualized_return = 0
        final_equity = initial_capital
    
    # Professional Trade Analysis
    if backtester.trades:
        print(f"\n📈 PROFESSIONAL TRADE ANALYSIS:")
        
        winning_trades = 0
        losing_trades = 0
        total_profit = 0
        total_loss = 0
        trade_returns = []
        
        for trade in backtester.trades:
            pnl = getattr(trade, 'pnl', 0)
            entry_price = getattr(trade, 'entry_price', 1)
            exit_price = getattr(trade, 'exit_price', entry_price)
            
            trade_return = (exit_price / entry_price - 1) * 100 if entry_price > 0 else 0
            trade_returns.append(trade_return)
            
            if pnl > 0:
                winning_trades += 1
                total_profit += pnl
            else:
                losing_trades += 1
                total_loss += abs(pnl)
        
        # Professional Metrics
        total_trades = len(backtester.trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        avg_trades_per_month = total_trades / 8  # 8 months
        
        # Advanced metrics
        avg_win = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
        risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Consistency metrics
        if trade_returns:
            win_rate_actual = len([r for r in trade_returns if r > 0]) / len(trade_returns) * 100
            avg_return_per_trade = np.mean(trade_returns)
            std_return_per_trade = np.std(trade_returns)
            consistency_ratio = abs(avg_return_per_trade) / std_return_per_trade if std_return_per_trade > 0 else 0
        else:
            win_rate_actual = 0
            avg_return_per_trade = 0
            consistency_ratio = 0
        
        print(f"   Total trades: {total_trades}")
        print(f"   Winning trades: {winning_trades}")
        print(f"   Losing trades: {losing_trades}")
        print(f"   Win rate: {win_rate*100:.1f}% (Target: {strategy.target_win_rate*100:.0f}%)")
        print(f"   Profit factor: {profit_factor:.2f} (Target: {strategy.target_profit_factor:.1f})")
        print(f"   Risk/Reward ratio: {risk_reward_ratio:.2f}:1")
        print(f"   Avg trades/month: {avg_trades_per_month:.1f} (Target: {strategy.target_trades_monthly})")
        print(f"   Avg return/trade: {avg_return_per_trade:+.2f}%")
        print(f"   Trade consistency: {consistency_ratio:.2f}")
        
        # Sample best and worst trades
        if trade_returns:
            sorted_returns = sorted(trade_returns, reverse=True)
            print(f"\n📝 TRADE PERFORMANCE SAMPLES:")
            print(f"   Best trades: {sorted_returns[:3]}")
            print(f"   Worst trades: {sorted_returns[-3:]}")
            print(f"   Median trade: {np.median(trade_returns):.2f}%")
    
    else:
        print(f"\n❌ NO TRADES EXECUTED")
        print(f"   Possible issues:")
        print(f"   • Signal threshold too high ({strategy.min_signal_strength:.3f})")
        print(f"   • Risk controls too strict")
        print(f"   • Market conditions don't match strategy")
        return None
    
    # Risk Analysis (Professional-grade)
    if backtester.equity_history and len(backtester.equity_history) > 10:
        print(f"\n⚖️ PROFESSIONAL RISK ANALYSIS:")
        
        equity_values = [e['total_equity'] for e in backtester.equity_history]
        equity_series = pd.Series(equity_values)
        
        # Drawdown Analysis
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = abs(drawdown.min()) * 100
        
        # Volatility Analysis
        daily_returns = equity_series.pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(365) * 100
        
        # Sharpe Ratio (assume 2% risk-free rate)
        excess_return = (annualized_return / 100) - 0.02
        sharpe_ratio = excess_return / (volatility / 100) if volatility > 0 else 0
        
        # Professional Risk Metrics
        var_95 = np.percentile(daily_returns * 100, 5)  # 5% VaR
        downside_returns = [r for r in daily_returns if r < 0]
        downside_volatility = np.std(downside_returns) * np.sqrt(365) * 100 if downside_returns else 0
        sortino_ratio = excess_return / (downside_volatility / 100) if downside_volatility > 0 else 0
        
        print(f"   Max drawdown: {max_drawdown:.1f}% (Limit: {strategy.max_acceptable_drawdown*100:.0f}%)")
        print(f"   Annual volatility: {volatility:.1f}%")
        print(f"   Sharpe ratio: {sharpe_ratio:.2f} (Target: {strategy.target_sharpe:.1f})")
        print(f"   Sortino ratio: {sortino_ratio:.2f}")
        print(f"   95% VaR (daily): {var_95:.2f}%")
        print(f"   Downside volatility: {downside_volatility:.1f}%")
        
        # Calmar Ratio (Return/Max Drawdown)
        calmar_ratio = (annualized_return / 100) / (max_drawdown / 100) if max_drawdown > 0 else 0
        print(f"   Calmar ratio: {calmar_ratio:.2f}")
    
    # Target Achievement Analysis
    print(f"\n🎯 TARGET ACHIEVEMENT ANALYSIS:")
    target_annual = 30.0
    target_monthly = 2.5
    
    achievement_score = 0
    total_criteria = 7
    
    # 1. Return Target
    return_achievement = annualized_return / target_annual if target_annual > 0 else 0
    if annualized_return >= target_annual * 0.8:  # Within 80% of target
        achievement_score += 1
        print(f"   ✅ Return target: {annualized_return:.1f}% (≥80% of {target_annual:.0f}%)")
    else:
        print(f"   ❌ Return target: {annualized_return:.1f}% (<80% of {target_annual:.0f}%)")
    
    # 2. Win Rate Target
    if win_rate >= strategy.target_win_rate * 0.9:  # Within 90% of target
        achievement_score += 1
        print(f"   ✅ Win rate: {win_rate*100:.1f}% (≥90% of {strategy.target_win_rate*100:.0f}%)")
    else:
        print(f"   ❌ Win rate: {win_rate*100:.1f}% (<90% of {strategy.target_win_rate*100:.0f}%)")
    
    # 3. Profit Factor Target
    if profit_factor >= strategy.target_profit_factor * 0.8:  # Within 80% of target
        achievement_score += 1
        print(f"   ✅ Profit factor: {profit_factor:.2f} (≥80% of {strategy.target_profit_factor:.1f})")
    else:
        print(f"   ❌ Profit factor: {profit_factor:.2f} (<80% of {strategy.target_profit_factor:.1f})")
    
    # 4. Risk Control (Drawdown)
    if max_drawdown <= strategy.max_acceptable_drawdown * 100:
        achievement_score += 1
        print(f"   ✅ Risk control: {max_drawdown:.1f}% DD (≤{strategy.max_acceptable_drawdown*100:.0f}%)")
    else:
        print(f"   ❌ Risk control: {max_drawdown:.1f}% DD (>{strategy.max_acceptable_drawdown*100:.0f}%)")
    
    # 5. Sharpe Ratio
    if sharpe_ratio >= strategy.target_sharpe * 0.8:
        achievement_score += 1
        print(f"   ✅ Risk-adjusted returns: {sharpe_ratio:.2f} Sharpe (≥80% of {strategy.target_sharpe:.1f})")
    else:
        print(f"   ❌ Risk-adjusted returns: {sharpe_ratio:.2f} Sharpe (<80% of {strategy.target_sharpe:.1f})")
    
    # 6. Trading Activity
    target_trades_8m = strategy.target_trades_monthly * 8  # 8 months
    if total_trades >= target_trades_8m * 0.6:  # At least 60% of target activity
        achievement_score += 1
        print(f"   ✅ Trading activity: {total_trades} trades (≥60% of {target_trades_8m})")
    else:
        print(f"   ❌ Trading activity: {total_trades} trades (<60% of {target_trades_8m})")
    
    # 7. Consistency  
    if win_rate > 0.5 and consistency_ratio > 1.0:  # More consistency than volatility
        achievement_score += 1
        print(f"   ✅ Consistency: {consistency_ratio:.2f} ratio (>1.0 = consistent)")
    else:
        print(f"   ❌ Consistency: {consistency_ratio:.2f} ratio (<1.0 = inconsistent)")
    
    # Overall Achievement Score
    achievement_percentage = (achievement_score / total_criteria) * 100
    print(f"\n📊 OVERALL ACHIEVEMENT SCORE: {achievement_score}/{total_criteria} ({achievement_percentage:.0f}%)")
    
    # Professional Verdict
    if achievement_percentage >= 85:
        verdict = "🏆 ELITE - Professional-grade performance!"
        verdict_color = "elite"
    elif achievement_percentage >= 70:
        verdict = "🎯 PROFESSIONAL - Solid institutional-quality results"
        verdict_color = "professional"
    elif achievement_percentage >= 55:
        verdict = "✅ ACCEPTABLE - Meets minimum professional standards"
        verdict_color = "acceptable"
    elif achievement_percentage >= 40:
        verdict = "⚠️ NEEDS IMPROVEMENT - Below professional standards"
        verdict_color = "improvement"
    else:
        verdict = "❌ UNACCEPTABLE - Major issues require addressing"
        verdict_color = "unacceptable"
    
    print(f"   {verdict}")
    
    # 3-Year Projection (Professional)
    if annualized_return > 0:
        year1_projection = initial_capital * (1 + min(annualized_return * 0.6, 25) / 100)  # Conservative Year 1
        year2_projection = year1_projection * (1 + min(annualized_return * 0.8, 35) / 100)  # Improved Year 2
        year3_projection = year2_projection * (1 + annualized_return / 100)  # Full potential Year 3
        
        total_3y_profit = year3_projection - initial_capital
        total_3y_return = (year3_projection / initial_capital - 1) * 100
        
        print(f"\n🔮 PROFESSIONAL 3-YEAR PROJECTION:")
        print(f"   Year 1 (Learning): ${initial_capital:,.0f} → ${year1_projection:,.0f}")
        print(f"   Year 2 (Optimization): ${year1_projection:,.0f} → ${year2_projection:,.0f}")
        print(f"   Year 3 (Mature): ${year2_projection:,.0f} → ${year3_projection:,.0f}")
        print(f"   Total 3-year profit: ${total_3y_profit:+,.0f}")
        print(f"   Total 3-year return: {total_3y_return:+.0f}%")
        print(f"   Compound annual return: {((year3_projection/initial_capital)**(1/3)-1)*100:.1f}%")
    
    # Reality Check & Recommendations
    print(f"\n✅ PROFESSIONAL REALITY CHECK:")
    
    reality_issues = []
    if annualized_return > 50:
        reality_issues.append("Returns too high for sustainable trading (>50%)")
    if max_drawdown > 25:
        reality_issues.append("Excessive drawdown risk (>25%)")
    if total_trades < 10:
        reality_issues.append("Insufficient trading activity for statistical significance")
    if win_rate < 0.4:
        reality_issues.append("Win rate too low (<40%) for sustainable profits")
    if sharpe_ratio < 1.0:
        reality_issues.append("Poor risk-adjusted returns (Sharpe <1.0)")
    
    if not reality_issues:
        print("   ✅ All professional reality checks passed!")
        print("   ✅ Strategy shows institutional-quality characteristics")
        print("   ✅ Risk management appears effective")
        print("   ✅ Returns are sustainable and realistic")
    else:
        print("   Professional concerns identified:")
        for issue in reality_issues:
            print(f"     ⚠️ {issue}")
    
    # Professional Recommendations
    print(f"\n💡 PROFESSIONAL RECOMMENDATIONS:")
    
    if achievement_percentage >= 70:
        print("   🚀 READY FOR NEXT PHASE:")
        print("   • Extend backtest to 12+ months")
        print("   • Test across different market regimes")
        print("   • Implement paper trading validation")
        print("   • Consider small live capital allocation")
        print("   • Monitor performance vs benchmarks")
    else:
        print("   🔧 OPTIMIZATION REQUIRED:")
        if win_rate < strategy.target_win_rate * 0.9:
            print("   • Improve signal quality or entry criteria")
        if profit_factor < strategy.target_profit_factor * 0.8:
            print("   • Optimize risk/reward ratios")
        if max_drawdown > strategy.max_acceptable_drawdown * 100:
            print("   • Tighten risk management controls")
        if total_trades < target_trades_8m * 0.6:
            print("   • Lower signal thresholds or expand opportunity set")
        if annualized_return < target_annual * 0.8:
            print("   • Enhance alpha generation or position sizing")
    
    return {
        'annualized_return': annualized_return,
        'achievement_score': achievement_percentage,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate * 100,
        'profit_factor': profit_factor,
        'total_trades': total_trades,
        'verdict': verdict,
        'projected_3y_return': total_3y_return if annualized_return > 0 else 0
    }

if __name__ == "__main__":
    results = test_professional_30_percent_strategy()
    
    if results:
        print(f"\n🏆 PROFESSIONAL SUMMARY:")
        print(f"Target Achievement: {results['achievement_score']:.0f}%")
        print(f"Annual Return: {results['annualized_return']:+.1f}%")
        print(f"Max Drawdown: {results['max_drawdown']:.1f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"3-Year Projection: {results['projected_3y_return']:+.0f}%")
        print(f"Verdict: {results['verdict']}")
    else:
        print(f"\n❌ Strategy requires significant debugging before professional evaluation")