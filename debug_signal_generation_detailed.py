#!/usr/bin/env python3
"""
Debug Signal Generation Detailed
================================
Detaillierte Analyse warum keine Signale generiert werden
"""

import numpy as np
import pandas as pd
from datetime import datetime
import logging

from realistic_crypto_backtest import RealisticMarketDataGenerator
from final_optimized_strategy import FinalOptimizedStrategy

# Disable debug logs for cleaner output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_signal_generation_detailed():
    """Detaillierte Analyse der Signalgenerierung"""
    
    print("🔬 DETAILED SIGNAL GENERATION DEBUG")
    print("=" * 60)
    
    # Generate longer data for proper indicators
    data_generator = RealisticMarketDataGenerator("BTC/USDT")
    market_data = data_generator.generate_realistic_data("2022-12-01", "2023-01-15")  # 6 Wochen
    
    print(f"📊 Market data: {len(market_data)} points")
    print(f"   Price range: ${market_data['close'].min():.0f} - ${market_data['close'].max():.0f}")
    print(f"   Date range: {market_data.index[0]} to {market_data.index[-1]}")
    
    # Initialize strategy
    strategy = FinalOptimizedStrategy()
    
    # Test signal generation at various points
    test_points = []
    signals_found = 0
    
    # Test every 100th point starting from 200 (need history for indicators)
    for i in range(200, len(market_data), 100):
        test_points.append(i)
    
    print(f"\n🧪 TESTING {len(test_points)} DATA POINTS:")
    
    for i, idx in enumerate(test_points):
        print(f"\n--- TEST #{i+1}: Index {idx} ---")
        
        timestamp = market_data.index[idx]
        price = market_data['close'].iloc[idx]
        historical_data = market_data.iloc[:idx+1]
        
        print(f"Date: {timestamp}")
        print(f"Price: ${price:.2f}")
        print(f"History: {len(historical_data)} points")
        
        # Step 1: Check indicators
        indicators = strategy.calculate_indicators(historical_data)
        
        if not indicators:
            print("❌ NO INDICATORS calculated")
            continue
        
        print(f"✅ Indicators calculated:")
        key_indicators = ['trend_strength', 'rsi', 'momentum_5', 'momentum_10', 'macd_histogram', 'volatility_regime']
        for key in key_indicators:
            if key in indicators:
                print(f"   {key}: {indicators[key]:.3f}")
        
        # Step 2: Manual signal calculation (replicate strategy logic)
        
        # Check basic constraints first
        print(f"\n🔍 Constraint checks:")
        
        # Daily trades (simulate)
        daily_trades = 0  # Fresh start
        max_daily = strategy.max_daily_trades
        print(f"   Daily trades: {daily_trades}/{max_daily} ✅")
        
        # Cooldown (simulate) 
        print(f"   Cooldown: OK (simulated) ✅")
        
        # Consecutive losses
        consecutive_losses = 0  # Fresh start
        max_losses = strategy.max_consecutive_losses
        print(f"   Consecutive losses: {consecutive_losses}/{max_losses} ✅")
        
        # Step 3: Signal components calculation
        print(f"\n🎯 Signal components:")
        
        signals = []
        reasons = []
        
        # 1. Trend Signal
        trend_strength = indicators.get('trend_strength', 0)
        if trend_strength > 0.015:
            signals.append(0.3)
            reasons.append("bullish_trend")
            print(f"   ✅ Trend: +0.3 (bullish {trend_strength:.3f})")
        elif trend_strength < -0.015:
            signals.append(0.15)
            reasons.append("bearish_trend_contrarian")
            print(f"   ⚠️  Trend: +0.15 (bearish contrarian {trend_strength:.3f})")
        else:
            print(f"   ❌ Trend: 0 (neutral {trend_strength:.3f})")
        
        # 2. RSI Signal
        rsi = indicators.get('rsi', 50)
        if 25 < rsi < 40:
            signals.append(0.25)
            reasons.append("rsi_oversold")
            print(f"   ✅ RSI: +0.25 (oversold {rsi:.1f})")
        elif 45 < rsi < 65:
            signals.append(0.15)
            reasons.append("rsi_normal")
            print(f"   ✅ RSI: +0.15 (normal {rsi:.1f})")
        elif rsi > 70:
            signals.append(0.05)
            reasons.append("rsi_overbought_weak")
            print(f"   ⚠️  RSI: +0.05 (overbought {rsi:.1f})")
        else:
            print(f"   ❌ RSI: 0 ({rsi:.1f})")
        
        # 3. Momentum
        momentum_5 = indicators.get('momentum_5', 0)
        momentum_10 = indicators.get('momentum_10', 0)
        
        if momentum_5 > 0.01 and momentum_10 > 0.005:
            signals.append(0.2)
            reasons.append("strong_momentum")
            print(f"   ✅ Momentum: +0.2 (strong)")
        elif momentum_5 > 0:
            signals.append(0.1)
            reasons.append("positive_momentum")
            print(f"   ✅ Momentum: +0.1 (positive)")
        else:
            print(f"   ❌ Momentum: 0 (5d:{momentum_5:.3f}, 10d:{momentum_10:.3f})")
        
        # 4. MACD
        macd_histogram = indicators.get('macd_histogram', 0)
        if macd_histogram > 0:
            signals.append(0.1)
            reasons.append("macd_positive")
            print(f"   ✅ MACD: +0.1 (positive {macd_histogram:.3f})")
        else:
            print(f"   ❌ MACD: 0 (negative {macd_histogram:.3f})")
        
        # 5. Volatility boost
        volatility = indicators.get('volatility_regime', 0.03)
        if volatility > 0.05:
            vol_boost = min(volatility * 2, 0.2)
            signals.append(vol_boost)
            reasons.append("volatility_opportunity")
            print(f"   ✅ Volatility: +{vol_boost:.2f} (high vol opportunity)")
        else:
            print(f"   ❌ Volatility: 0 (low vol {volatility:.3f})")
        
        # Calculate final strength
        if signals:
            final_strength = np.mean(signals)
            print(f"\n📊 Signal calculation:")
            print(f"   Components: {[f'{s:.2f}' for s in signals]}")
            print(f"   Final strength: {final_strength:.3f}")
            print(f"   Threshold: {strategy.min_signal_strength:.3f}")
            
            if final_strength >= strategy.min_signal_strength:
                print(f"   ✅ SIGNAL APPROVED! ({final_strength:.3f} >= {strategy.min_signal_strength:.3f})")
                signals_found += 1
                
                # Test actual strategy call
                actual_signal = strategy.generate_signal(historical_data, timestamp)
                print(f"   Actual strategy result: {actual_signal['direction']} (strength: {actual_signal.get('strength', 0):.3f})")
                
                if actual_signal['direction'] == 'hold':
                    print(f"   ⚠️  DISCREPANCY! Manual calc says BUY but strategy says HOLD")
                    print(f"   Strategy reason: {actual_signal.get('reason', 'unknown')}")
                
            else:
                print(f"   ❌ Below threshold ({final_strength:.3f} < {strategy.min_signal_strength:.3f})")
        else:
            print(f"\n📊 No signal components found")
        
        print(f"   Reasons: {reasons}")
        
        if i >= 4:  # Limit output
            remaining = len(test_points) - i - 1
            if remaining > 0:
                print(f"\n... (skipping {remaining} more tests for brevity)")
            break
    
    print(f"\n📈 SUMMARY:")
    print(f"   Total test points: {min(5, len(test_points))}")
    print(f"   Manual signals found: {signals_found}")
    print(f"   Signal rate: {signals_found/min(5, len(test_points))*100:.1f}%")
    
    if signals_found == 0:
        print(f"\n🔧 PROBLEM DIAGNOSIS:")
        print(f"   1. Signal threshold too high? ({strategy.min_signal_strength:.3f})")
        print(f"   2. Trend requirements too strict? (±0.015)")
        print(f"   3. RSI ranges too narrow? (25-40 for oversold)")
        print(f"   4. Momentum thresholds too high? (>0.01 and >0.005)")
        print(f"   5. Market too choppy/sideways for trend following?")
        
        print(f"\n💡 SUGGESTED FIXES:")
        print(f"   • Lower signal threshold to 0.05")
        print(f"   • Reduce trend requirement to ±0.01")
        print(f"   • Expand RSI ranges")
        print(f"   • Add mean reversion signals")
        print(f"   • Include sideways market strategies")
    
    # Test with relaxed parameters
    print(f"\n🧪 TESTING WITH RELAXED PARAMETERS:")
    
    # Temporarily modify strategy
    original_threshold = strategy.min_signal_strength
    strategy.min_signal_strength = 0.05  # Very low
    
    relaxed_signals = 0
    for idx in test_points[:3]:  # Test first 3 points
        historical_data = market_data.iloc[:idx+1]
        timestamp = market_data.index[idx]
        signal = strategy.generate_signal(historical_data, timestamp)
        
        if signal['direction'] != 'hold':
            relaxed_signals += 1
            print(f"   ✅ Signal #{relaxed_signals}: {signal['direction']} (strength: {signal.get('strength', 0):.3f})")
    
    # Restore original
    strategy.min_signal_strength = original_threshold
    
    print(f"   Relaxed signals found: {relaxed_signals}/3")
    
    if relaxed_signals > 0:
        print(f"   💡 SOLUTION: Lower the signal threshold!")
    else:
        print(f"   🚨 DEEPER PROBLEM: Even relaxed parameters don't work")

if __name__ == "__main__":
    debug_signal_generation_detailed()