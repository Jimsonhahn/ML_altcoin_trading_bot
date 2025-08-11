#!/usr/bin/env python3
"""
Realistic Performance Analysis
==============================
Überprüfung der tatsächlichen Performance und Identifikation der Probleme
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

from realistic_crypto_backtest import RealisticBacktester, RealisticMarketDataGenerator
from optimized_realistic_strategy import OptimizedRealisticStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_realistic_performance():
    """Detaillierte Analyse der tatsächlichen Performance"""
    
    print("🔍 REALISTIC PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Generate market data
    data_generator = RealisticMarketDataGenerator("BTC/USDT")
    market_data = data_generator.generate_realistic_data("2023-01-01", "2023-04-01")  # Nur 3 Monate
    
    print(f"📊 Market Data:")
    print(f"   Period: 3 months")
    print(f"   Data points: {len(market_data)}")
    print(f"   Market return: {(market_data['close'].iloc[-1] / market_data['close'].iloc[0] - 1)*100:.1f}%")
    
    # Initialize strategy
    strategy = OptimizedRealisticStrategy()
    backtester = RealisticBacktester(initial_capital=10000, symbol="BTC/USDT")
    backtester.strategy = strategy
    
    print(f"\n🎛️  Strategy Settings:")
    print(f"   Min signal strength: {strategy.min_signal_strength}")
    print(f"   Max position size: {strategy.max_position_size}")
    print(f"   Stop loss: {strategy.stop_loss_pct*100:.1f}%")
    print(f"   Take profit: {strategy.take_profit_pct*100:.1f}%")
    
    # Run backtest
    print(f"\n🔄 Running backtest...")
    results = backtester.run_backtest(market_data)
    
    print(f"\n📈 RAW RESULTS:")
    print(f"   Signals generated: {len([s for s in results.get('signals', []) if s.get('direction') != 'hold'])}")
    print(f"   Trades executed: {len(backtester.trades)}")
    print(f"   Initial capital: ${backtester.initial_capital:,.0f}")
    
    if backtester.equity_history:
        final_equity = backtester.equity_history[-1]['total_equity']
        total_return_pct = (final_equity / backtester.initial_capital - 1) * 100
        print(f"   Final equity: ${final_equity:,.0f}")
        print(f"   Total return: {total_return_pct:.1f}%")
        
        # Monatliche Rendite
        monthly_return = (final_equity / backtester.initial_capital) ** (1/3) - 1
        print(f"   Monthly return: {monthly_return*100:.1f}%")
        print(f"   Annualized (if sustained): {((1+monthly_return)**12-1)*100:.1f}%")
    
    # Detaillierte Trade-Analyse
    if backtester.trades:
        print(f"\n💰 TRADE ANALYSIS:")
        
        profitable_trades = 0
        total_profit = 0
        total_loss = 0
        
        for i, trade in enumerate(backtester.trades[:10]):  # Erste 10 Trades
            pnl = getattr(trade, 'realized_pnl', 0)
            return_pct = getattr(trade, 'return_pct', 0)
            
            if pnl > 0:
                profitable_trades += 1
                total_profit += pnl
            else:
                total_loss += abs(pnl)
            
            print(f"   Trade {i+1}: P&L=${pnl:+.0f} ({return_pct:+.1%})")
        
        if len(backtester.trades) > 10:
            print(f"   ... und {len(backtester.trades)-10} weitere Trades")
        
        win_rate = profitable_trades / len(backtester.trades) if backtester.trades else 0
        print(f"\n   Win Rate: {win_rate*100:.1f}%")
        print(f"   Total Profit: ${total_profit:.0f}")
        print(f"   Total Loss: ${total_loss:.0f}")
        print(f"   Net P&L: ${total_profit - total_loss:.0f}")
    
    # Prüfung auf unrealistische Faktoren
    print(f"\n🚨 REALISM CHECK:")
    
    # 1. Zu hohe Rendite?
    if backtester.equity_history:
        final_equity = backtester.equity_history[-1]['total_equity']
        total_return = final_equity / backtester.initial_capital - 1
        
        if total_return > 0.5:  # >50% in 3 Monaten
            print(f"   ❌ UNREALISTIC: {total_return*100:.1f}% in 3 months is too high")
        elif total_return > 0.2:  # >20% in 3 Monaten  
            print(f"   ⚠️  SUSPICIOUS: {total_return*100:.1f}% in 3 months is very high")
        else:
            print(f"   ✅ REASONABLE: {total_return*100:.1f}% in 3 months")
    
    # 2. Zu viele profitable Trades?
    if backtester.trades:
        profitable_count = sum(1 for trade in backtester.trades if getattr(trade, 'realized_pnl', 0) > 0)
        win_rate = profitable_count / len(backtester.trades)
        
        if win_rate > 0.8:
            print(f"   ❌ UNREALISTIC: {win_rate*100:.1f}% win rate is too high")
        elif win_rate > 0.65:
            print(f"   ⚠️  SUSPICIOUS: {win_rate*100:.1f}% win rate is very high")
        else:
            print(f"   ✅ REASONABLE: {win_rate*100:.1f}% win rate")
    
    # 3. Zu niedriger Drawdown?
    max_drawdown = 0
    if backtester.equity_history:
        peak = backtester.initial_capital
        for point in backtester.equity_history:
            equity = point['total_equity']
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        if max_drawdown < 0.005:  # <0.5%
            print(f"   ❌ UNREALISTIC: {max_drawdown*100:.2f}% max drawdown is too low")
        elif max_drawdown < 0.02:  # <2%
            print(f"   ⚠️  SUSPICIOUS: {max_drawdown*100:.1f}% max drawdown is very low")
        else:
            print(f"   ✅ REASONABLE: {max_drawdown*100:.1f}% max drawdown")
    
    # 4. Trading-Kosten Analyse
    if backtester.trades:
        avg_trade_size = np.mean([getattr(trade, 'size', 0) for trade in backtester.trades])
        total_trades = len(backtester.trades)
        
        # Geschätzte Kosten (0.3% pro Trade round-trip)
        estimated_costs = total_trades * avg_trade_size * 0.003
        
        print(f"\n💸 COST ANALYSIS:")
        print(f"   Total trades: {total_trades}")
        print(f"   Avg trade size: ${avg_trade_size:.0f}")
        print(f"   Estimated costs: ${estimated_costs:.0f}")
        
        if backtester.equity_history:
            final_equity = backtester.equity_history[-1]['total_equity']
            gross_profit = final_equity - backtester.initial_capital + estimated_costs
            print(f"   Gross profit (before costs): ${gross_profit:.0f}")
            print(f"   Net profit (after costs): ${final_equity - backtester.initial_capital:.0f}")
    
    # Empfehlungen
    print(f"\n🎯 RECOMMENDATIONS:")
    if backtester.equity_history:
        final_equity = backtester.equity_history[-1]['total_equity']
        total_return = final_equity / backtester.initial_capital - 1
        
        if total_return > 0.3:
            print(f"   1. ❌ Results are too good to be true")
            print(f"   2. 🔧 Increase trading costs and slippage")
            print(f"   3. 🔧 Add more realistic market impact")
            print(f"   4. 🔧 Reduce position sizes")
            print(f"   5. 🔧 Add more conservative stop losses")
        elif total_return > 0.15:
            print(f"   1. ⚠️  Results are optimistic but possible")
            print(f"   2. 🧪 Extend testing period to 12+ months")
            print(f"   3. 🧪 Test in different market conditions")
        else:
            print(f"   1. ✅ Results appear realistic")
            print(f"   2. 📊 Ready for longer backtesting periods")

if __name__ == "__main__":
    analyze_realistic_performance()