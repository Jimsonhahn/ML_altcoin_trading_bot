#!/usr/bin/env python3
"""
Debug Strategy Signals
======================
Debug why the strategy is not generating any trades
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

from realistic_crypto_backtest import RealisticMarketDataGenerator
from optimized_realistic_strategy import OptimizedRealisticStrategy

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_signal_generation():
    """Debug the signal generation process"""
    
    print("🔍 DEBUGGING SIGNAL GENERATION")
    print("=" * 50)
    
    # Generate market data
    data_generator = RealisticMarketDataGenerator("BTC/USDT")
    market_data = data_generator.generate_realistic_data("2022-01-01", "2024-01-01")
    
    print(f"📊 Market data generated: {len(market_data)} points")
    print(f"   Price range: ${market_data['close'].min():.0f} - ${market_data['close'].max():.0f}")
    print(f"   First date: {market_data.index[0]}")
    print(f"   Last date: {market_data.index[-1]}")
    
    # Initialize strategy
    strategy = OptimizedRealisticStrategy()
    
    print(f"\n🎛️  Strategy parameters:")
    print(f"   Min signal strength: {strategy.min_signal_strength}")
    print(f"   Max position size: {strategy.max_position_size}")
    print(f"   Volume multiplier: {strategy.volume_multiplier}")
    print(f"   Volatility threshold: {strategy.volatility_threshold}")
    
    # Test signal generation at various points
    test_points = [1000, 2000, 5000, 8000, 10000, 12000, 15000]
    
    signals_generated = 0
    
    for i, test_point in enumerate(test_points):
        if test_point >= len(market_data):
            continue
            
        print(f"\n🧪 TEST POINT {i+1}: Index {test_point}")
        print(f"   Date: {market_data.index[test_point]}")
        print(f"   Price: ${market_data['close'].iloc[test_point]:.2f}")
        
        # Get historical data up to this point
        historical_data = market_data.iloc[:test_point+1]
        timestamp = market_data.index[test_point]
        
        # Generate signal
        signal = strategy.generate_signal(historical_data, timestamp)
        
        print(f"   Signal: {signal['direction']} (strength: {signal.get('strength', 0):.3f})")
        print(f"   Reason: {signal.get('reason', 'unknown')}")
        
        if signal.get('reasons'):
            print(f"   Detailed reasons: {signal['reasons']}")
        
        if signal['direction'] != 'hold':
            signals_generated += 1
            print(f"   ✅ TRADE SIGNAL GENERATED!")
        
        # Also check indicators
        indicators = strategy.calculate_indicators(historical_data)
        
        if indicators:
            print(f"   📈 Key indicators:")
            print(f"      RSI: {indicators.get('rsi', 0):.1f}")
            print(f"      Trend strength: {indicators.get('trend_strength', 0):.3f}")
            print(f"      Volume ratio: {indicators.get('volume_ratio', 0):.2f}")
            print(f"      Regime: {indicators.get('regime', 'unknown')}")
            print(f"      MACD: {indicators.get('macd', 0):.3f}")
            
            # Debug individual signal components
            if hasattr(strategy, '_debug_signals'):
                print(f"   🔍 Signal components: {strategy._debug_signals}")
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total test points: {len([p for p in test_points if p < len(market_data)])}")
    print(f"   Signals generated: {signals_generated}")
    print(f"   Signal rate: {signals_generated / len([p for p in test_points if p < len(market_data)]) * 100:.1f}%")
    
    if signals_generated == 0:
        print("\n❌ NO SIGNALS GENERATED - INVESTIGATING...")
        
        # Test with very relaxed parameters
        print("\n🔧 Testing with extremely relaxed parameters...")
        
        strategy.min_signal_strength = 0.05  # Very low
        strategy.volume_multiplier = 0.5     # Very low
        strategy.volatility_threshold = 0.20  # Very high
        
        for i, test_point in enumerate([5000, 10000, 15000]):
            if test_point >= len(market_data):
                continue
                
            historical_data = market_data.iloc[:test_point+1]
            timestamp = market_data.index[test_point]
            signal = strategy.generate_signal(historical_data, timestamp)
            
            print(f"   Relaxed test {i+1}: {signal['direction']} (strength: {signal.get('strength', 0):.3f})")
            
            if signal['direction'] != 'hold':
                print(f"   ✅ SUCCESS with relaxed parameters!")
                break

if __name__ == "__main__":
    debug_signal_generation()