#!/usr/bin/env python3
"""
Debug Order Rejection
=====================
Analyse warum alle Orders vom Exchange abgelehnt werden
"""

import numpy as np
import pandas as pd
from datetime import datetime
import logging

from realistic_crypto_backtest import RealisticBacktester, RealisticMarketDataGenerator
from final_optimized_strategy import FinalOptimizedStrategy

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_order_rejection():
    """Debug warum Orders abgelehnt werden"""
    
    print("🔍 DEBUG ORDER REJECTION")
    print("=" * 50)
    
    # Kurze Daten für debugging
    data_generator = RealisticMarketDataGenerator("BTC/USDT")
    market_data = data_generator.generate_realistic_data("2023-01-01", "2023-01-02")  # Nur 1 Tag
    
    print(f"📊 Market data: {len(market_data)} points")
    print(f"   Price range: ${market_data['close'].min():.0f} - ${market_data['close'].max():.0f}")
    
    # Strategie initialisieren
    strategy = FinalOptimizedStrategy()
    backtester = RealisticBacktester(initial_capital=10000, symbol="BTC/USDT")
    backtester.strategy = strategy
    
    print(f"\n🎯 Strategy initialized")
    print(f"   Initial capital: ${backtester.initial_capital:,}")
    print(f"   Max position size: {strategy.max_position_size*100:.1f}%")
    
    # Manuell durch einige Datenpunkte gehen
    test_indices = [100, 200, 300, 400, 500] if len(market_data) > 500 else [50, 100, 150, 200]
    
    for i, idx in enumerate(test_indices):
        if idx >= len(market_data):
            continue
            
        print(f"\n🧪 TEST #{i+1} - Index {idx}")
        timestamp = market_data.index[idx]
        current_price = market_data['close'].iloc[idx]
        current_volume = market_data.get('volume', pd.Series(index=market_data.index, data=1000000)).iloc[idx]
        
        print(f"   Time: {timestamp}")
        print(f"   Price: ${current_price:.2f}")
        print(f"   Volume: {current_volume:,.0f}")
        
        # Generate signal
        historical_data = market_data.iloc[:idx+1]
        signal = strategy.generate_signal(historical_data, timestamp)
        
        print(f"   Signal: {signal['direction']} (strength: {signal.get('strength', 0):.3f})")
        print(f"   Reason: {signal.get('reason', 'N/A')}")
        
        if signal['direction'] != 'hold':
            print(f"   ✅ SIGNAL GENERATED!")
            
            # Calculate position size
            current_equity = backtester.get_current_equity(current_price)
            position_size = strategy.calculate_position_size(
                signal.get('strength', 0), 
                current_equity, 
                0.05  # Sample volatility
            )
            
            print(f"   Current equity: ${current_equity:.0f}")
            print(f"   Position size: ${position_size:.0f}")
            
            if position_size > 0:
                print(f"   💰 POSITION SIZE APPROVED!")
                
                # Test market info (was fehlt?)
                market_info = {
                    'price': current_price,
                    'volume': current_volume,
                    'volatility': 0.05,
                    'spread': current_price * 0.001,  # 0.1% spread
                    'liquidity_factor': 1.0
                }
                
                print(f"   Market info:")
                for key, value in market_info.items():
                    if key == 'price':
                        print(f"     {key}: ${value:.2f}")
                    elif key in ['volume']:
                        print(f"     {key}: {value:,.0f}")
                    else:
                        print(f"     {key}: {value:.4f}")
                
                # Check if order würde abgelehnt werden
                # Simulate RealisticExchangeSimulator logic
                
                # 1. Minimum Order Size Check
                min_order_size = 10  # $10 minimum
                if position_size < min_order_size:
                    print(f"   ❌ REJECTION: Position size ${position_size:.0f} < ${min_order_size} minimum")
                    continue
                
                # 2. Liquidity Check 
                max_order_of_volume = current_volume * 0.02  # Max 2% of volume
                if position_size > max_order_of_volume:
                    print(f"   ❌ REJECTION: Position size ${position_size:.0f} > ${max_order_of_volume:.0f} (2% of volume)")
                    continue
                
                # 3. Capital Check
                if position_size > current_equity * 0.5:  # More than 50% of equity
                    print(f"   ❌ REJECTION: Position size ${position_size:.0f} > ${current_equity*0.5:.0f} (50% of equity)")
                    continue
                
                # 4. Spread Check
                spread = market_info['spread']
                if spread > current_price * 0.01:  # Spread > 1%
                    print(f"   ❌ REJECTION: Spread ${spread:.2f} > 1% of price (${current_price*0.01:.2f})")
                    continue
                
                print(f"   ✅ ORDER SHOULD BE ACCEPTED!")
                print(f"     ✓ Size: ${position_size:.0f} >= ${min_order_size}")
                print(f"     ✓ Liquidity: ${position_size:.0f} <= ${max_order_of_volume:.0f}")
                print(f"     ✓ Capital: ${position_size:.0f} <= ${current_equity*0.5:.0f}")
                print(f"     ✓ Spread: ${spread:.2f} <= ${current_price*0.01:.2f}")
                
            else:
                print(f"   ❌ POSITION SIZE = 0")
                print(f"     Possible reasons:")
                print(f"     - Signal too weak")
                print(f"     - Consecutive losses limit")
                print(f"     - Below minimum viable size")
        else:
            print(f"   ⏸️  No signal generated")
            if signal.get('reason'):
                print(f"     Reason: {signal.get('reason')}")
    
    print(f"\n🔧 DEBUGGING ACTUAL EXCHANGE SIMULATOR:")
    
    # Check RealisticExchangeSimulator parameters
    exchange_sim = backtester.exchange
    print(f"   Exchange parameters:")
    
    # Try to find out what's happening in the exchange simulator
    print(f"   Looking for rejection reasons in RealisticExchangeSimulator...")
    
    # Run a mini backtest to see rejections
    print(f"\n🔄 Mini backtest (first 100 points)...")
    mini_data = market_data.head(100)
    
    try:
        results = backtester.run_backtest(mini_data)
        print(f"   Results: {len(backtester.trades)} trades executed")
        print(f"   Rejected orders: {len(backtester.rejected_orders)}")
        
        if backtester.rejected_orders:
            print(f"\n❌ REJECTED ORDERS ANALYSIS:")
            for i, rejection in enumerate(backtester.rejected_orders[:5]):  # First 5
                print(f"   #{i+1}: {rejection}")
        
    except Exception as e:
        print(f"   Error during mini backtest: {e}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   1. Check RealisticExchangeSimulator minimum order sizes")
    print(f"   2. Verify volume data in market_data")
    print(f"   3. Review liquidity constraints")
    print(f"   4. Check capital requirements")

if __name__ == "__main__":
    debug_order_rejection()