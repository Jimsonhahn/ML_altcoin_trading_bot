#!/usr/bin/env python3
"""
Test Candle Momentum Strategy Integration
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategies import get_strategy

def create_test_data(periods=100):
    """Create synthetic OHLCV data for testing"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=periods), 
                         periods=periods, freq='1H')
    
    # Create realistic price movements
    base_price = 50000
    prices = []
    current_price = base_price
    
    for i in range(periods):
        # Add some trend and volatility
        change = np.random.normal(0, 0.02) + 0.001 * np.sin(i / 10)  # Slight uptrend with cycles
        current_price *= (1 + change)
        prices.append(current_price)
    
    # Create OHLCV from prices
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] if i > 0 else price
        close_price = price
        volume = np.random.uniform(1000, 5000)
        
        data.append({
            'timestamp': dates[i],
            'open': open_price,
            'high': max(open_price, close_price, high),
            'low': min(open_price, close_price, low),
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df

def test_candle_momentum_strategy():
    """Test the integrated candle momentum strategy"""
    print("🧪 Testing Candle Momentum Strategy Integration")
    print("=" * 50)
    
    # Get strategy class
    strategy_class = get_strategy('candle_momentum')
    if not strategy_class:
        print("❌ Strategy not found!")
        return False
    
    # Initialize strategy
    config = {
        'lookback_period': 20,
        'sma_period': 30,  # Shorter for test data
        'volume_filter': True,
        'min_momentum_ratio': 1.2,
        'min_confidence': 0.4
    }
    
    strategy = strategy_class(config)
    print(f"✅ Strategy initialized: {strategy.name}")
    
    # Create test data
    test_data = create_test_data(80)  # 80 periods
    print(f"✅ Created test data: {len(test_data)} candles")
    
    # Test signal generation
    try:
        # Test base class method
        signal, signal_data = strategy.calculate_signal('BTC/USDT', test_data, test_data['close'].iloc[-1])
        print(f"✅ Base class method works: {signal}")
        
        # Test legacy method
        signals = strategy.generate_signals(test_data, 'BTC/USDT')
        print(f"✅ Legacy method works: {signals['signal']}")
        
        # Display signal details
        print(f"\n📊 Signal Details:")
        print(f"   Signal: {signals['signal']}")
        print(f"   Confidence: {signals['confidence']:.3f}")
        print(f"   Momentum Ratio: {signals['metadata']['momentum_ratio']:.3f}")
        print(f"   Price vs Trend: {signals['metadata']['price_vs_trend']:.3f}")
        print(f"   Volume OK: {signals['metadata']['volume_ok']}")
        
        # Test multiple timeframes
        signals_list = []
        for i in range(5):
            window_data = test_data.iloc[i*10:i*10+60]  # Sliding window
            if len(window_data) >= 35:  # Minimum data requirement
                sig = strategy.generate_signals(window_data, f'TEST_{i}')
                signals_list.append(sig)
        
        print(f"\n📈 Multiple Window Test: Generated {len(signals_list)} signals")
        
        # Test position sizing
        if signals['signal'] != 'hold':
            position_size = strategy.calculate_position_size(
                signals, 
                10000,  # Account balance
                test_data['close'].iloc[-1]  # Current price
            )
            print(f"💰 Position Size: {position_size:.6f}")
        
        # Test strategy info
        info = strategy.get_strategy_info()
        print(f"\n📋 Strategy Info:")
        print(f"   Name: {info['name']}")
        print(f"   Version: {info['version']}")
        print(f"   Active Symbols: {len(info['active_symbols'])}")
        
        # Test visualization data
        viz_data = strategy.get_momentum_visualization_data('BTC/USDT', test_data)
        if viz_data:
            print(f"📈 Visualization data available: {len(viz_data)} fields")
        
        print(f"\n✅ All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_candle_momentum_strategy()
    if success:
        print("\n🎉 Candle Momentum Strategy integration completed successfully!")
        print("\nNext steps:")
        print("1. Update your main trading bot to use 'candle_momentum' strategy")
        print("2. Configure parameters in your config file")
        print("3. Run backtests to optimize parameters")
        print("4. Deploy for paper trading first")
    else:
        print("\n❌ Integration test failed. Please check the errors above.")