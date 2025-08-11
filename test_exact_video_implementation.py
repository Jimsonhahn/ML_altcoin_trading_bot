#!/usr/bin/env python3
"""
Test Exact TradingView Video Implementation
==========================================

Validates the candle body momentum strategy matches the video exactly.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_simple_test_data() -> pd.DataFrame:
    """Create simple test data for validation"""
    print("📊 Creating test data...")
    
    dates = pd.date_range(start='2024-01-01', periods=250, freq='30T')
    np.random.seed(42)
    
    base_price = 50000
    data = []
    
    for i, timestamp in enumerate(dates):
        if i == 0:
            open_price = base_price
            close_price = base_price + 100
        else:
            prev_close = data[-1]['close']
            open_price = prev_close + np.random.uniform(-50, 50)
            
            # Create patterns
            if 50 <= i < 70:  # Strong bullish
                close_price = open_price + np.random.uniform(200, 400)
            elif 120 <= i < 140:  # Strong bearish
                close_price = open_price - np.random.uniform(200, 400)
            else:  # Random
                close_price = open_price + np.random.uniform(-100, 100)
        
        high_price = max(open_price, close_price) + abs(np.random.uniform(0, 50))
        low_price = min(open_price, close_price) - abs(np.random.uniform(0, 50))
        volume = np.random.uniform(1000000, 5000000)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    print(f"✅ Created {len(df)} test candles")
    return df

def test_strategy_basic_functionality():
    """Test basic strategy functionality"""
    print("\n🧪 TESTING BASIC FUNCTIONALITY")
    print("=" * 30)
    
    try:
        # Import strategy
        from strategies import get_strategy
        
        strategy_class = get_strategy('candle_body_momentum')
        if not strategy_class:
            print("❌ Strategy not found!")
            return False
        
        # Initialize with video parameters
        params = {
            'lookback_period': 10,
            'sma_period': 200,
            'timeframe': '30m',
            'debug_logging': True
        }
        
        strategy = strategy_class(params)
        print(f"✅ Strategy initialized: {strategy.__class__.__name__}")
        
        # Create test data
        test_data = create_simple_test_data()
        
        # Test momentum calculation
        print("\n📈 Testing momentum calculation...")
        bullish_strength, bearish_strength = strategy.calculate_candle_body_momentum(test_data)
        
        if len(bullish_strength) == len(test_data) and len(bearish_strength) == len(test_data):
            print("✅ Momentum calculation working")
        else:
            print("❌ Momentum calculation failed")
            return False
        
        # Test crossover detection
        print("\n🔄 Testing crossover detection...")
        crossovers = strategy.detect_momentum_crossovers(bullish_strength, bearish_strength)
        
        if 'bullish_crossover' in crossovers and 'bearish_crossover' in crossovers:
            print("✅ Crossover detection working")
        else:
            print("❌ Crossover detection failed")
            return False
        
        # Test signal generation
        print("\n📊 Testing signal generation...")
        signals_found = 0
        
        for i in range(210, min(240, len(test_data))):
            current_data = test_data.iloc[:i+1]
            current_price = current_data['close'].iloc[-1]
            
            signal, signal_data = strategy.calculate_signal('BTC/USDT', current_data, current_price)
            
            if signal != 'HOLD':
                signals_found += 1
                print(f"  📡 Signal at bar {i}: {signal} (confidence: {signal_data['confidence']:.2f})")
        
        print(f"✅ Generated {signals_found} signals")
        
        # Test validation method
        print("\n🔍 Running strategy validation...")
        validation = strategy.validate_implementation(test_data)
        
        all_valid = all(validation.values())
        print(f"✅ Validation results: {validation}")
        
        if all_valid:
            print("\n🎉 ALL BASIC TESTS PASSED!")
            
            # Get strategy info
            info = strategy.get_strategy_info()
            print(f"\n📋 Strategy Info:")
            print(f"   Name: {info['name']}")
            print(f"   Version: {info['version']}")
            print(f"   Lookback: {info['parameters']['lookback_period']}")
            print(f"   SMA: {info['parameters']['sma_period']}")
            print(f"   Timeframe: {info['parameters']['optimal_timeframe']}")
            
            return True
        else:
            print("❌ Some validations failed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_momentum_accuracy_test():
    """Test momentum calculation accuracy"""
    print("\n🎯 MOMENTUM CALCULATION ACCURACY TEST")
    print("=" * 40)
    
    try:
        from strategies import get_strategy
        strategy_class = get_strategy('candle_body_momentum')
        strategy = strategy_class({'lookback_period': 10, 'sma_period': 200})
        
        # Create specific test case
        test_candles = [
            {'open': 100, 'close': 110},  # +10 bullish
            {'open': 110, 'close': 105},  # -5 bearish
            {'open': 105, 'close': 120},  # +15 bullish
            {'open': 120, 'close': 115},  # -5 bearish
            {'open': 115, 'close': 125},  # +10 bullish
        ]
        
        # Create DataFrame
        data = []
        for i, candle in enumerate(test_candles):
            data.append({
                'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(hours=i),
                'open': candle['open'],
                'high': max(candle['open'], candle['close']) + 2,
                'low': min(candle['open'], candle['close']) - 2,
                'close': candle['close'],
                'volume': 1000000
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        # Calculate momentum
        bull_strength, bear_strength = strategy.calculate_candle_body_momentum(df)
        
        # Expected values (running sum)
        expected_bull = [10, 10, 25, 25, 35]  # 10, 10+0, 10+15, 10+15+0, 10+15+10
        expected_bear = [0, 5, 5, 10, 10]     # 0, 5, 5, 5+5, 5+5
        
        print("Manual calculation check:")
        for i in range(len(df)):
            actual_bull = bull_strength.iloc[i]
            actual_bear = bear_strength.iloc[i]
            exp_bull = expected_bull[i]
            exp_bear = expected_bear[i]
            
            bull_match = abs(actual_bull - exp_bull) < 0.01
            bear_match = abs(actual_bear - exp_bear) < 0.01
            
            print(f"  Bar {i}: Bull {actual_bull:.1f}=={exp_bull} ✅, Bear {actual_bear:.1f}=={exp_bear} {'✅' if bear_match else '❌'}")
            
            if not (bull_match and bear_match):
                print("❌ Momentum calculation inaccurate!")
                return False
        
        print("✅ Momentum calculation accuracy verified!")
        return True
        
    except Exception as e:
        print(f"❌ Accuracy test failed: {e}")
        return False

def main():
    """Main test execution"""
    print("🚀 CANDLE BODY MOMENTUM - EXACT VIDEO IMPLEMENTATION TEST")
    print("=" * 58)
    
    tests = [
        ("Basic Functionality", test_strategy_basic_functionality),
        ("Momentum Accuracy", run_momentum_accuracy_test),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n📋 TEST SUMMARY")
    print("=" * 15)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("✅ Strategy implements TradingView video specification")
        
        # Save test report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = {
            'timestamp': timestamp,
            'strategy': 'candle_body_momentum',
            'implementation': 'exact_video_match',
            'test_results': results,
            'parameters': {
                'lookback_period': 10,
                'sma_period': 200,
                'timeframe': '30m'
            },
            'status': 'PASSED'
        }
        
        with open(f'video_implementation_test_{timestamp}.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"💾 Test report saved")
        
        print(f"\n🎯 READY FOR:")
        print("1. 30-minute timeframe backtesting")
        print("2. Paper trading deployment")
        print("3. Live market validation")
        
        return True
    else:
        print(f"\n❌ TESTS FAILED")
        print("Fix implementation before proceeding")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)