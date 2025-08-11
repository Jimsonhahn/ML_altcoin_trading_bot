#!/usr/bin/env python3
"""
Simple Tier 1 Strategy Test
===========================
Direct test of the optimized strategy to achieve Tier 1 performance.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

# Import realistic backtest engine
from realistic_crypto_backtest import (
    RealisticBacktester, RealisticMarketDataGenerator, 
    RealisticExchangeSimulator
)

from optimized_realistic_strategy import OptimizedRealisticStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_performance_metrics(backtester):
    """Calculate performance metrics from backtester results"""
    
    if not backtester.trades or len(backtester.trades) == 0:
        return {
            'annual_return': 0,
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_trades': 0
        }
    
    # Calculate returns from equity history
    if backtester.equity_history:
        initial_capital = backtester.initial_capital
        final_equity = backtester.equity_history[-1]['total_equity']
        total_return = (final_equity / initial_capital) - 1
        
        # Annualized return (calculate years from timestamps)
        start_time = backtester.equity_history[0]['timestamp'] 
        end_time = backtester.equity_history[-1]['timestamp']
        years = (end_time - start_time).days / 365.25 if len(backtester.equity_history) > 1 else 1
        annual_return = (final_equity / initial_capital) ** (1/years) - 1
        
        # Calculate drawdown
        peak = initial_capital
        max_drawdown = 0
        for equity_point in backtester.equity_history:
            equity = equity_point['total_equity']
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
    else:
        total_return = 0
        annual_return = 0
        max_drawdown = 0
    
    # Calculate trade statistics
    winning_trades = 0
    losing_trades = 0
    total_profit = 0
    total_loss = 0
    
    for trade in backtester.trades:
        pnl = trade.realized_pnl if hasattr(trade, 'realized_pnl') else getattr(trade, 'pnl', 0)
        if pnl > 0:
            winning_trades += 1
            total_profit += pnl
        elif pnl < 0:
            losing_trades += 1
            total_loss += abs(pnl)
    
    total_trades = len(backtester.trades)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else (float('inf') if total_profit > 0 else 0)
    
    # Calculate Sharpe ratio (simplified)
    if backtester.equity_history and len(backtester.equity_history) > 1:
        returns = []
        for i in range(1, len(backtester.equity_history)):
            prev_equity = backtester.equity_history[i-1]['total_equity']
            curr_equity = backtester.equity_history[i]['total_equity']
            if prev_equity > 0:
                returns.append((curr_equity - prev_equity) / prev_equity)
        
        if returns:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = (mean_return / std_return * np.sqrt(365*24)) if std_return > 0 else 0
        else:
            sharpe_ratio = 0
    else:
        sharpe_ratio = 0
    
    return {
        'annual_return': annual_return,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_trades': total_trades
    }

def test_tier1_strategy():
    """Direct test of Tier 1 strategy with optimized parameters"""
    
    print("🎯 DIRECT TIER 1 STRATEGY TEST")
    print("=" * 60)
    
    # The OptimizedRealisticStrategy has built-in optimized parameters
    
    # Generate realistic market data
    print("📊 Generating realistic market data...")
    data_generator = RealisticMarketDataGenerator("BTC/USDT")
    market_data = data_generator.generate_realistic_data("2022-01-01", "2024-01-01")
    
    # Initialize strategy and backtester
    print("🚀 Initializing strategy...")
    strategy = OptimizedRealisticStrategy()
    backtester = RealisticBacktester(initial_capital=10000, symbol="BTC/USDT")
    
    # Set our custom strategy
    backtester.strategy = strategy
    
    # Run backtest
    print("📈 Running backtest...")
    results = backtester.run_backtest(market_data)
    
    # Calculate metrics manually
    metrics = calculate_performance_metrics(backtester)
    
    # Display results
    print("\n🏆 TIER 1 STRATEGY RESULTS")
    print("=" * 60)
    print(f"Annual Return:      {metrics.get('annual_return', 0)*100:.1f}%")
    print(f"Total Return:       {metrics.get('total_return', 0)*100:.1f}%")
    print(f"Sharpe Ratio:       {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown:       {metrics.get('max_drawdown', 0)*100:.1f}%")
    print(f"Win Rate:           {metrics.get('win_rate', 0)*100:.1f}%")
    print(f"Profit Factor:      {metrics.get('profit_factor', 0):.2f}")
    print(f"Total Trades:       {metrics.get('total_trades', 0)}")
    
    # Check Tier 1 criteria
    print("\n✅ TIER 1 CRITERIA CHECK:")
    tier1_score = 0
    total_criteria = 5
    
    annual_return = metrics.get('annual_return', 0)
    if annual_return >= 0.25:
        print(f"✅ Annual Return: {annual_return*100:.1f}% >= 25.0%")
        tier1_score += 1
    else:
        print(f"❌ Annual Return: {annual_return*100:.1f}% < 25.0%")
    
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
    
    return metrics

if __name__ == "__main__":
    test_tier1_strategy()