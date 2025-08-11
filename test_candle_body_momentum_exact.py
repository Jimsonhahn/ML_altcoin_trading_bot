#!/usr/bin/env python3
"""
Test Candle Body Momentum Strategy - Exact Video Implementation
==============================================================

This test validates that the candle body momentum strategy is implemented
exactly as described in the TradingView Pine Script video.

Key Validation Points:
1. Momentum calculation: Running sum of body sizes (not averages)
2. Crossover detection: Exact bar detection (current vs previous)
3. Parameters: 30min timeframe, 200 SMA, 10 candles lookback
4. Entry/Exit logic: Crossovers with SMA confirmation
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging to see debug output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data_with_momentum_patterns() -> pd.DataFrame:
    """Create test data with clear momentum patterns for validation"""
    print("📊 Creating test data with clear momentum patterns...")
    
    # Create 300 periods of data
    dates = pd.date_range(start='2024-01-01', periods=300, freq='30T')  # 30-minute bars
    np.random.seed(42)  # Reproducible results
    
    base_price = 50000
    data = []
    
    for i, timestamp in enumerate(dates):
        if i == 0:
            # First candle
            open_price = base_price
            close_price = base_price + np.random.uniform(-100, 100)
        else:
            # Subsequent candles
            prev_close = data[-1]['close']
            open_price = prev_close + np.random.uniform(-50, 50)
            
            # Create specific patterns for testing
            if 50 <= i < 70:
                # Strong bullish pattern - large bullish candles
                close_price = open_price + np.random.uniform(200, 500)
                
            elif 120 <= i < 140:
                # Strong bearish pattern - large bearish candles
                close_price = open_price - np.random.uniform(200, 500)
                
            elif 200 <= i < 220:
                # Mixed pattern with crossovers
                if i % 3 == 0:
                    close_price = open_price + np.random.uniform(100, 300)  # Bullish
                else:
                    close_price = open_price - np.random.uniform(50, 150)   # Bearish
            else:
                # Normal random movement
                close_price = open_price + np.random.uniform(-100, 100)
        
        # Generate realistic OHLC
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
    
    print(f"✅ Created {len(df)} test candles with momentum patterns")
    return df

def validate_momentum_calculation(strategy, data: pd.DataFrame) -> bool:
    """Validate that momentum is calculated exactly as described"""
    print("\n🔍 VALIDATING MOMENTUM CALCULATION")
    print("=" * 35)
    
    try:
        # Calculate momentum using strategy
        bullish_strength, bearish_strength = strategy.calculate_candle_body_momentum(data)
        
        # Manual calculation for validation
        lookback = strategy.lookback_period
        
        print(f"Testing with lookback period: {lookback}")
        
        # Test specific bars
        test_indices = [50, 100, 150, 200, 250]
        
        for idx in test_indices:
            if idx >= len(data):
                continue
                
            # Get strategy result
            strategy_bull = bullish_strength.iloc[idx]
            strategy_bear = bearish_strength.iloc[idx]
            
            # Manual calculation
            start_idx = max(0, idx - lookback + 1)
            manual_bull = 0.0
            manual_bear = 0.0
            
            for i in range(start_idx, idx + 1):
                open_price = data['open'].iloc[i]
                close_price = data['close'].iloc[i]
                body_size = abs(close_price - open_price)
                
                if close_price > open_price:  # Bullish candle
                    manual_bull += body_size
                elif close_price < open_price:  # Bearish candle
                    manual_bear += body_size\n            \n            # Compare results\n            bull_match = abs(strategy_bull - manual_bull) < 0.01\n            bear_match = abs(strategy_bear - manual_bear) < 0.01\n            \n            print(f\"Bar {idx}:\")\n            print(f\"  Bullish - Strategy: {strategy_bull:.2f}, Manual: {manual_bull:.2f}, Match: {bull_match}\")\n            print(f\"  Bearish - Strategy: {strategy_bear:.2f}, Manual: {manual_bear:.2f}, Match: {bear_match}\")\n            \n            if not (bull_match and bear_match):\n                print(\"❌ Momentum calculation validation FAILED\")\n                return False\n        \n        print(\"✅ Momentum calculation validation PASSED\")\n        return True\n        \n    except Exception as e:\n        print(f\"❌ Error validating momentum calculation: {e}\")\n        return False

def validate_crossover_detection(strategy, data: pd.DataFrame) -> bool:
    """Validate crossover detection is exact (current vs previous bar)"""
    print(\"\\n🔍 VALIDATING CROSSOVER DETECTION\")\n    print(\"=\" * 33)\n    \n    try:\n        # Calculate momentum\n        bullish_strength, bearish_strength = strategy.calculate_candle_body_momentum(data)\n        \n        # Test crossover detection on multiple points\n        crossover_found = False\n        \n        for i in range(50, len(data) - 10):  # Test range with sufficient history\n            # Create subset ending at this point\n            subset = data.iloc[:i+1]\n            \n            # Calculate momentum for subset\n            subset_bull, subset_bear = strategy.calculate_candle_body_momentum(subset)\n            \n            # Detect crossovers\n            crossovers = strategy.detect_momentum_crossovers(subset_bull, subset_bear)\n            \n            # Manual crossover detection\n            if len(subset_bull) >= 2:\n                bull_curr = subset_bull.iloc[-1]\n                bear_curr = subset_bear.iloc[-1]\n                bull_prev = subset_bull.iloc[-2]\n                bear_prev = subset_bear.iloc[-2]\n                \n                manual_bullish_cross = (bull_curr > bear_curr and bull_prev <= bear_prev)\n                manual_bearish_cross = (bear_curr > bull_curr and bear_prev <= bull_prev)\n                \n                # Compare with strategy results\n                if crossovers['bullish_crossover'] != manual_bullish_cross:\n                    print(f\"❌ Bullish crossover mismatch at bar {i}\")\n                    print(f\"   Strategy: {crossovers['bullish_crossover']}, Manual: {manual_bullish_cross}\")\n                    return False\n                    \n                if crossovers['bearish_crossover'] != manual_bearish_cross:\n                    print(f\"❌ Bearish crossover mismatch at bar {i}\")\n                    print(f\"   Strategy: {crossovers['bearish_crossover']}, Manual: {manual_bearish_cross}\")\n                    return False\n                \n                # Log crossovers found\n                if crossovers['bullish_crossover'] or crossovers['bearish_crossover']:\n                    crossover_found = True\n                    print(f\"✅ Crossover detected at bar {i}:\")\n                    print(f\"   Bullish: {crossovers['bullish_crossover']}, Bearish: {crossovers['bearish_crossover']}\")\n                    print(f\"   Current: Bull={bull_curr:.2f}, Bear={bear_curr:.2f}\")\n                    print(f\"   Previous: Bull={bull_prev:.2f}, Bear={bear_prev:.2f}\")\n        \n        if crossover_found:\n            print(\"✅ Crossover detection validation PASSED\")\n            return True\n        else:\n            print(\"⚠️  No crossovers found in test data (may be normal)\")\n            return True\n            \n    except Exception as e:\n        print(f\"❌ Error validating crossover detection: {e}\")\n        return False

def validate_signal_generation(strategy, data: pd.DataFrame) -> bool:\n    \"\"\"Validate complete signal generation logic\"\"\"\n    print(\"\\n🔍 VALIDATING SIGNAL GENERATION\")\n    print(\"=\" * 31)\n    \n    try:\n        signals_generated = []\n        \n        # Test signal generation at various points\n        for i in range(210, min(280, len(data))):  # Test range with sufficient history\n            current_data = data.iloc[:i+1]\n            current_price = current_data['close'].iloc[-1]\n            \n            # Generate signal\n            signal, signal_data = strategy.calculate_signal('BTC/USDT', current_data, current_price)\n            \n            if signal != 'HOLD':\n                signals_generated.append({\n                    'bar': i,\n                    'signal': signal,\n                    'confidence': signal_data['confidence'],\n                    'reason': signal_data['reason'],\n                    'price': current_price,\n                    'metadata': signal_data['metadata']\n                })\n                \n                print(f\"📊 Signal at bar {i}:\")\n                print(f\"   Signal: {signal}\")\n                print(f\"   Confidence: {signal_data['confidence']:.2f}\")\n                print(f\"   Reason: {signal_data['reason']}\")\n                print(f\"   Bull Strength: {signal_data['metadata']['bullish_strength']:.2f}\")\n                print(f\"   Bear Strength: {signal_data['metadata']['bearish_strength']:.2f}\")\n                print(f\"   Price vs SMA: {signal_data['metadata']['price_vs_sma']:.4f}\")\n        \n        print(f\"\\n✅ Generated {len(signals_generated)} signals during test\")\n        \n        # Validate signal logic\n        for signal_info in signals_generated:\n            metadata = signal_info['metadata']\n            \n            if signal_info['signal'] == 'BUY':\n                # Long signal should have bullish crossover and price > SMA\n                if not (metadata['bullish_crossover'] and metadata['price_vs_sma'] > 1.0):\n                    print(f\"❌ Invalid BUY signal at bar {signal_info['bar']}\")\n                    return False\n                    \n            elif signal_info['signal'] == 'SELL':\n                # Short signal should have bearish crossover and price < SMA\n                if not (metadata['bearish_crossover'] and metadata['price_vs_sma'] < 1.0):\n                    print(f\"❌ Invalid SELL signal at bar {signal_info['bar']}\")\n                    return False\n        \n        print(\"✅ Signal generation validation PASSED\")\n        return True\n        \n    except Exception as e:\n        print(f\"❌ Error validating signal generation: {e}\")\n        return False

def validate_video_parameters(strategy) -> bool:\n    \"\"\"Validate that strategy uses exact parameters from video\"\"\"\n    print(\"\\n🔍 VALIDATING VIDEO PARAMETERS\")\n    print(\"=\" * 31)\n    \n    expected_params = {\n        'lookback_period': 10,\n        'sma_period': 200,\n        'timeframe': '30m'\n    }\n    \n    validation_passed = True\n    \n    for param, expected_value in expected_params.items():\n        actual_value = getattr(strategy, param)\n        \n        if actual_value == expected_value:\n            print(f\"✅ {param}: {actual_value} (correct)\")\n        else:\n            print(f\"❌ {param}: {actual_value} (expected: {expected_value})\")\n            validation_passed = False\n    \n    if validation_passed:\n        print(\"✅ Video parameters validation PASSED\")\n    else:\n        print(\"❌ Video parameters validation FAILED\")\n    \n    return validation_passed

def test_visualization_data(strategy, data: pd.DataFrame) -> bool:\n    \"\"\"Test momentum visualization data for debugging\"\"\"\n    print(\"\\n🔍 TESTING VISUALIZATION DATA\")\n    print(\"=\" * 29)\n    \n    try:\n        viz_data = strategy.get_momentum_visualization_data('BTC/USDT', data)\n        \n        required_fields = [\n            'timestamp', 'price', 'bullish_strength', 'bearish_strength', \n            'sma_200', 'crossover_points', 'momentum_difference'\n        ]\n        \n        for field in required_fields:\n            if field not in viz_data:\n                print(f\"❌ Missing visualization field: {field}\")\n                return False\n            else:\n                print(f\"✅ {field}: Available\")\n        \n        # Check crossover points\n        crossovers = viz_data['crossover_points']\n        print(f\"📈 Found {len(crossovers)} crossover points in data\")\n        \n        for i, crossover in enumerate(crossovers[:5]):  # Show first 5\n            print(f\"   {i+1}. {crossover['type']} at {crossover['timestamp']} (price: {crossover['price']:.2f})\")\n        \n        print(\"✅ Visualization data test PASSED\")\n        return True\n        \n    except Exception as e:\n        print(f\"❌ Error testing visualization data: {e}\")\n        return False

def run_comprehensive_validation():\n    \"\"\"Run comprehensive validation of the exact video implementation\"\"\"\n    print(\"🚀 CANDLE BODY MOMENTUM - EXACT VIDEO IMPLEMENTATION TEST\")\n    print(\"=\" * 60)\n    \n    try:\n        # Import strategy\n        from strategies import get_strategy\n        \n        strategy_class = get_strategy('candle_body_momentum')\n        if not strategy_class:\n            print(\"❌ Strategy 'candle_body_momentum' not found in registry!\")\n            return False\n        \n        # Initialize with exact video parameters\n        video_params = {\n            'lookback_period': 10,\n            'sma_period': 200,\n            'timeframe': '30m',\n            'debug_logging': True\n        }\n        \n        strategy = strategy_class(video_params)\n        print(f\"✅ Strategy initialized: {strategy.__class__.__name__}\")\n        \n        # Create test data\n        test_data = create_test_data_with_momentum_patterns()\n        \n        # Run validation tests\n        validation_results = {\n            'video_parameters': validate_video_parameters(strategy),\n            'momentum_calculation': validate_momentum_calculation(strategy, test_data),\n            'crossover_detection': validate_crossover_detection(strategy, test_data),\n            'signal_generation': validate_signal_generation(strategy, test_data),\n            'visualization_data': test_visualization_data(strategy, test_data)\n        }\n        \n        # Summary\n        print(\"\\n📋 VALIDATION SUMMARY\")\n        print(\"=\" * 20)\n        \n        all_passed = True\n        for test_name, result in validation_results.items():\n            status = \"✅ PASS\" if result else \"❌ FAIL\"\n            print(f\"{test_name}: {status}\")\n            if not result:\n                all_passed = False\n        \n        # Overall result\n        if all_passed:\n            print(\"\\n🎉 ALL VALIDATIONS PASSED!\")\n            print(\"✅ Strategy implements the exact TradingView video specification\")\n            \n            # Save validation report\n            timestamp = datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n            report = {\n                'timestamp': timestamp,\n                'strategy': 'candle_body_momentum',\n                'video_implementation': 'exact',\n                'validation_results': validation_results,\n                'parameters_used': video_params,\n                'test_data_points': len(test_data),\n                'status': 'PASSED'\n            }\n            \n            with open(f'candle_body_momentum_validation_{timestamp}.json', 'w') as f:\n                json.dump(report, f, indent=2, default=str)\n            \n            print(f\"💾 Validation report saved: candle_body_momentum_validation_{timestamp}.json\")\n            \n            # Strategy info\n            info = strategy.get_strategy_info()\n            print(f\"\\n📊 STRATEGY INFO:\")\n            print(f\"   Name: {info['name']}\")\n            print(f\"   Version: {info['version']}\")\n            print(f\"   Optimal Timeframe: {info['parameters']['optimal_timeframe']}\")\n            print(f\"   Key Features: {len(info['key_features'])} implemented\")\n            \n            return True\n        else:\n            print(\"\\n❌ VALIDATION FAILED\")\n            print(\"Some tests did not pass - check implementation against video\")\n            return False\n            \n    except Exception as e:\n        print(f\"❌ Validation error: {e}\")\n        import traceback\n        traceback.print_exc()\n        return False

if __name__ == \"__main__\":\n    success = run_comprehensive_validation()\n    \n    if success:\n        print(\"\\n🎯 NEXT STEPS:\")\n        print(\"1. Run backtest with 30-minute data\")\n        print(\"2. Compare performance with video results\")\n        print(\"3. Deploy for paper trading\")\n        print(\"4. Monitor crossover accuracy in live conditions\")\n    else:\n        print(\"\\n🔧 ACTION REQUIRED:\")\n        print(\"1. Review failed validations above\")\n        print(\"2. Fix implementation to match video exactly\")\n        print(\"3. Re-run validation test\")