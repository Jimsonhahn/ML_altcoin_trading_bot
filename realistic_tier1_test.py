#!/usr/bin/env python3
"""
Realistic Tier 1 Test
====================
Test der ausgewogenen Strategie für realistische 15-25% Jahresrendite
"""

import numpy as np
import pandas as pd
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

from realistic_crypto_backtest import RealisticBacktester, RealisticMarketDataGenerator
from balanced_realistic_strategy import BalancedRealisticStrategy
from simple_tier1_test import calculate_performance_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_realistic_tier1():
    """Test mit realistischen Erwartungen für Tier 1"""
    
    print("🎯 REALISTIC TIER 1 STRATEGY TEST")
    print("=" * 60)
    print("Ziel: Realistische 15-25% Jahresrendite mit <15% Drawdown")
    
    # 1-Jahr Test für aussagekräftige Ergebnisse
    print("\n📊 Generating 12-month market data...")
    data_generator = RealisticMarketDataGenerator("BTC/USDT")
    market_data = data_generator.generate_realistic_data("2022-06-01", "2023-06-01")
    
    print(f"   Data points: {len(market_data)}")
    market_return = (market_data['close'].iloc[-1] / market_data['close'].iloc[0] - 1) * 100
    volatility = market_data['close'].pct_change().std() * np.sqrt(365) * 100
    print(f"   Market return: {market_return:.1f}%")
    print(f"   Market volatility: {volatility:.1f}%")
    
    # Ausgewogene Strategie
    strategy = BalancedRealisticStrategy()
    backtester = RealisticBacktester(initial_capital=10000, symbol="BTC/USDT")
    backtester.strategy = strategy
    
    # Zeige erwartete Performance
    expected_return = strategy.get_expected_annual_return()
    
    print(f"\n🎛️  Balanced Strategy Settings:")
    print(f"   Max position size: {strategy.max_position_size*100:.1f}%")
    print(f"   Min signal strength: {strategy.min_signal_strength}")
    print(f"   Stop loss: {strategy.stop_loss_pct*100:.1f}%")
    print(f"   Take profit: {strategy.take_profit_pct*100:.1f}%")
    print(f"   R/R ratio: {strategy.take_profit_pct/strategy.stop_loss_pct:.1f}:1")
    print(f"   Total costs per trade: {strategy.total_cost_per_trade*100:.2f}%")
    print(f"   Expected annual return: {expected_return*100:.1f}%")
    
    # Run backtest
    print(f"\n⏳ Running realistic backtest (this may take a moment)...")
    results = backtester.run_backtest(market_data)
    
    # Calculate metrics
    metrics = calculate_performance_metrics(backtester)
    
    # Display results
    print(f"\n📊 REALISTIC RESULTS (12 months):")
    print(f"   Initial capital: ${backtester.initial_capital:,.0f}")
    
    if backtester.equity_history:
        final_equity = backtester.equity_history[-1]['total_equity']
        print(f"   Final equity: ${final_equity:,.0f}")
        profit = final_equity - backtester.initial_capital
        print(f"   Total profit: ${profit:+,.0f}")
    
    print(f"   Annual return: {metrics.get('annual_return', 0)*100:+.1f}%")
    print(f"   Max drawdown: {metrics.get('max_drawdown', 0)*100:.1f}%")
    print(f"   Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"   Total trades: {metrics.get('total_trades', 0)}")
    print(f"   Win rate: {metrics.get('win_rate', 0)*100:.1f}%")
    print(f"   Profit factor: {metrics.get('profit_factor', 0):.2f}")
    
    if metrics.get('total_trades', 0) > 0:
        print(f"   Avg trades/month: {metrics.get('total_trades', 0)/12:.1f}")
    
    # Realistic Tier 1 Criteria (angepasst für Crypto)
    print(f"\n✅ REALISTIC TIER 1 CRITERIA:")
    tier1_score = 0
    total_criteria = 5
    
    # 1. Annual Return: 15-40% für Crypto realistisch
    annual_return = metrics.get('annual_return', 0)
    if annual_return >= 0.15:
        if annual_return <= 0.40:
            print(f"   ✅ Annual Return: {annual_return*100:.1f}% (15-40% target)")
            tier1_score += 1
        else:
            print(f"   ⚠️  Annual Return: {annual_return*100:.1f}% (suspicious if >40%)")
            tier1_score += 0.5
    else:
        print(f"   ❌ Annual Return: {annual_return*100:.1f}% < 15.0%")
    
    # 2. Sharpe Ratio: >1.0 für Crypto gut
    sharpe_ratio = metrics.get('sharpe_ratio', 0)
    if sharpe_ratio >= 1.0:
        print(f"   ✅ Sharpe Ratio: {sharpe_ratio:.2f} >= 1.00")
        tier1_score += 1
    else:
        print(f"   ❌ Sharpe Ratio: {sharpe_ratio:.2f} < 1.00")
    
    # 3. Max Drawdown: <15% akzeptabel für Crypto
    max_drawdown = metrics.get('max_drawdown', 1)
    if max_drawdown <= 0.15:
        print(f"   ✅ Max Drawdown: {max_drawdown*100:.1f}% <= 15.0%")
        tier1_score += 1
    else:
        print(f"   ❌ Max Drawdown: {max_drawdown*100:.1f}% > 15.0%")
    
    # 4. Win Rate: >50% gut
    win_rate = metrics.get('win_rate', 0)
    if win_rate >= 0.50:
        print(f"   ✅ Win Rate: {win_rate*100:.1f}% >= 50.0%")
        tier1_score += 1
    else:
        print(f"   ❌ Win Rate: {win_rate*100:.1f}% < 50.0%")
    
    # 5. Profit Factor: >1.3 gut
    profit_factor = metrics.get('profit_factor', 0)
    if profit_factor >= 1.3:
        print(f"   ✅ Profit Factor: {profit_factor:.2f} >= 1.30")
        tier1_score += 1
    else:
        print(f"   ❌ Profit Factor: {profit_factor:.2f} < 1.30")
    
    # Final assessment
    tier1_percentage = (tier1_score / total_criteria) * 100
    print(f"\n🎖️ REALISTIC TIER 1 SCORE: {tier1_score}/{total_criteria} ({tier1_percentage:.0f}%)")
    
    if tier1_score >= 4:
        verdict = "🚀 REALISTIC TIER 1 ACHIEVED!"
        color = "green"
    elif tier1_score >= 3:
        verdict = "⚠️  NEAR TIER 1 - Good performance"
        color = "yellow" 
    elif tier1_score >= 2:
        verdict = "📈 DECENT PERFORMANCE - Needs improvement"
        color = "orange"
    else:
        verdict = "❌ NEEDS SIGNIFICANT IMPROVEMENT"
        color = "red"
    
    print(f"   {verdict}")
    
    # Trading activity analysis
    if metrics.get('total_trades', 0) > 0:
        print(f"\n📈 TRADING ACTIVITY ANALYSIS:")
        trades_per_month = metrics.get('total_trades', 0) / 12
        
        if trades_per_month < 5:
            print(f"   ⚠️  Low activity: {trades_per_month:.1f} trades/month")
        elif trades_per_month > 30:
            print(f"   ⚠️  High activity: {trades_per_month:.1f} trades/month (overtrading?)")
        else:
            print(f"   ✅ Good activity: {trades_per_month:.1f} trades/month")
        
        if win_rate > 0:
            avg_win = profit_factor * (1-win_rate) / win_rate if win_rate > 0 else 0
            print(f"   Average win: {avg_win:.1f}x average loss")
    
    # Market comparison
    strategy_return = metrics.get('annual_return', 0)
    market_annual = market_return  # Already annualized for 1 year
    
    print(f"\n📊 MARKET COMPARISON:")
    print(f"   Market return: {market_annual:.1f}%")
    print(f"   Strategy return: {strategy_return*100:.1f}%")
    
    if strategy_return * 100 > market_annual:
        outperformance = (strategy_return * 100) - market_annual
        print(f"   ✅ Outperformed market by {outperformance:.1f}%")
    else:
        underperformance = market_annual - (strategy_return * 100)
        print(f"   ❌ Underperformed market by {underperformance:.1f}%")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    
    if tier1_score >= 4:
        print(f"   • Strategy shows strong performance")
        print(f"   • Ready for extended backtesting (2-3 years)")
        print(f"   • Consider paper trading validation")
        print(f"   • Test in different market conditions")
    elif tier1_score >= 3:
        print(f"   • Good foundation, minor optimizations needed")
        print(f"   • Focus on improving weakest metric")
        print(f"   • Extend testing period")
    elif tier1_score >= 2:
        print(f"   • Moderate performance, significant improvements needed")
        print(f"   • Review risk management parameters")
        print(f"   • Consider signal quality improvements")
    else:
        print(f"   • Strategy needs major rework")
        print(f"   • Review fundamental approach")
        print(f"   • Consider different indicators or timeframes")
    
    # Cost impact analysis
    if metrics.get('total_trades', 0) > 0:
        total_cost_impact = metrics.get('total_trades', 0) * strategy.total_cost_per_trade
        print(f"\n💸 COST IMPACT:")
        print(f"   Total trades: {metrics.get('total_trades', 0)}")
        print(f"   Cost per trade: {strategy.total_cost_per_trade*100:.2f}%")
        print(f"   Total cost impact: {total_cost_impact*100:.1f}%")
        print(f"   Return after costs: {(strategy_return - total_cost_impact)*100:.1f}%")
    
    return {
        'tier1_score': tier1_score,
        'tier1_percentage': tier1_percentage,
        'verdict': verdict,
        'metrics': metrics,
        'market_comparison': strategy_return * 100 - market_annual
    }

if __name__ == "__main__":
    test_realistic_tier1()