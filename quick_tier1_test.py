#!/usr/bin/env python3
"""
Quick Tier 1 Strategy Test
==========================
Shorter test period to get complete results faster
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

from realistic_crypto_backtest import (
    RealisticBacktester, RealisticMarketDataGenerator
)

from optimized_realistic_strategy import OptimizedRealisticStrategy
from simple_tier1_test import calculate_performance_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_tier1_test():
    """Quick test with 6-month data for faster results"""
    
    print("🎯 QUICK TIER 1 STRATEGY TEST")
    print("=" * 60)
    
    # Generate shorter market data (6 months)
    print("📊 Generating market data (6 months)...")
    data_generator = RealisticMarketDataGenerator("BTC/USDT")
    market_data = data_generator.generate_realistic_data("2023-01-01", "2023-07-01")
    
    # Initialize strategy and backtester
    print("🚀 Initializing strategy...")
    strategy = OptimizedRealisticStrategy()
    backtester = RealisticBacktester(initial_capital=10000, symbol="BTC/USDT")
    backtester.strategy = strategy
    
    # Run backtest
    print("📈 Running backtest...")
    results = backtester.run_backtest(market_data)
    
    # Calculate metrics manually
    metrics = calculate_performance_metrics(backtester)
    
    # Annualize the metrics (multiply by 2 since we have 6 months)
    annualized_return = (1 + metrics['total_return']) ** 2 - 1
    
    # Display results
    print("\n🏆 QUICK TIER 1 RESULTS (6 months)")
    print("=" * 60)
    print(f"Total Return (6m):     {metrics.get('total_return', 0)*100:.1f}%")
    print(f"Annualized Return:     {annualized_return*100:.1f}%")
    print(f"Sharpe Ratio:          {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown:          {metrics.get('max_drawdown', 0)*100:.1f}%")
    print(f"Win Rate:              {metrics.get('win_rate', 0)*100:.1f}%")
    print(f"Profit Factor:         {metrics.get('profit_factor', 0):.2f}")
    print(f"Total Trades:          {metrics.get('total_trades', 0)}")
    
    # Check Tier 1 criteria against annualized metrics
    print("\n✅ TIER 1 CRITERIA CHECK (Annualized):")
    tier1_score = 0
    total_criteria = 5
    
    if annualized_return >= 0.25:
        print(f"✅ Annual Return: {annualized_return*100:.1f}% >= 25.0%")
        tier1_score += 1
    else:
        print(f"❌ Annual Return: {annualized_return*100:.1f}% < 25.0%")
    
    sharpe_ratio = metrics.get('sharpe_ratio', 0)
    if sharpe_ratio >= 1.5:
        print(f"✅ Sharpe Ratio: {sharpe_ratio:.2f} >= 1.50")
        tier1_score += 1
    else:
        print(f"❌ Sharpe Ratio: {sharpe_ratio:.2f} < 1.50")
    
    win_rate = metrics.get('win_rate', 0)
    if win_rate >= 0.55:
        print(f"✅ Win Rate: {win_rate*100:.1f}% >= 55.0%")
        tier1_score += 1
    else:
        print(f"❌ Win Rate: {win_rate*100:.1f}% < 55.0%")
    
    max_drawdown = metrics.get('max_drawdown', 1)
    if max_drawdown <= 0.10:
        print(f"✅ Max Drawdown: {max_drawdown*100:.1f}% <= 10.0%")
        tier1_score += 1
    else:
        print(f"❌ Max Drawdown: {max_drawdown*100:.1f}% > 10.0%")
    
    profit_factor = metrics.get('profit_factor', 0)
    if profit_factor >= 1.5:
        print(f"✅ Profit Factor: {profit_factor:.2f} >= 1.50")
        tier1_score += 1
    else:
        print(f"❌ Profit Factor: {profit_factor:.2f} < 1.50")
    
    # Final assessment
    tier1_percentage = (tier1_score / total_criteria) * 100
    print(f"\n🎖️ TIER 1 SCORE: {tier1_score}/{total_criteria} ({tier1_percentage:.0f}%)")
    
    if tier1_score >= 4:
        print("🚀 TIER 1 PERFORMANCE ACHIEVED!")
    elif tier1_score >= 3:
        print("⚠️  NEAR TIER 1 - Minor optimization needed")
    else:
        print("❌ SIGNIFICANT OPTIMIZATION REQUIRED")
        
    # Show some trade details
    if backtester.trades:
        print(f"\n📊 TRADE ANALYSIS:")
        print(f"   Avg trades per month: {len(backtester.trades) / 6:.1f}")
        
        if backtester.equity_history:
            final_equity = backtester.equity_history[-1]['total_equity']
            print(f"   Final equity: ${final_equity:,.0f}")
            print(f"   Total profit: ${final_equity - backtester.initial_capital:,.0f}")
    
    return metrics, tier1_score

if __name__ == "__main__":
    quick_tier1_test()