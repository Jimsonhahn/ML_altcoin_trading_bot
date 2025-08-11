#!/usr/bin/env python3
"""
Quick Candle Momentum Strategy Backtest
======================================

Fast backtest for the candle momentum strategy with essential metrics.
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

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce logging verbosity
logger = logging.getLogger(__name__)

def create_test_market_data(periods: int = 1000) -> pd.DataFrame:
    """Create simple test market data"""
    print(f"📊 Generating {periods} periods of test data...")
    
    np.random.seed(42)
    base_price = 45000
    
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='1H')
    
    # Simple price walk with trend
    returns = np.random.normal(0.0001, 0.003, periods)  # Slight upward drift with volatility
    prices = [base_price]
    
    for ret in returns[:-1]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV
    data = []
    for i, (timestamp, close_price) in enumerate(zip(dates, prices)):
        open_price = prices[i-1] if i > 0 else close_price
        
        # Simple OHLC generation
        volatility = close_price * 0.002  # 0.2% intrabar volatility
        high_price = max(open_price, close_price) + np.random.uniform(0, volatility)
        low_price = min(open_price, close_price) - np.random.uniform(0, volatility)
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
    return df

def run_quick_backtest():
    """Run quick backtest for candle momentum strategy"""
    print("🕯️  QUICK CANDLE MOMENTUM BACKTEST")
    print("=" * 35)
    
    try:
        # Import strategy
        from strategies import get_strategy
        
        strategy_class = get_strategy('candle_momentum')
        if not strategy_class:
            raise ValueError("Candle momentum strategy not found!")
        
        # Generate test data
        market_data = create_test_market_data(500)  # Smaller dataset for speed
        print(f"📈 Generated {len(market_data)} candles")
        
        # Test configuration
        config = {
            'lookback_period': 15,
            'sma_period': 25,
            'min_momentum_ratio': 1.3,
            'min_confidence': 0.5,
            'volume_filter': True
        }
        
        # Initialize strategy
        strategy = strategy_class(config)
        print("✅ Strategy initialized")
        
        # Backtest parameters
        initial_capital = 10000
        position = None
        trades = []
        signals_generated = 0
        
        print("🚀 Running backtest...")
        
        # Simple backtest loop
        for i in range(50, len(market_data)):  # Start after warmup
            try:
                # Get point-in-time data
                current_data = market_data.iloc[:i+1]
                current_price = current_data['close'].iloc[-1]
                
                # Generate signal (less verbose)
                signal_result = strategy.generate_signals(current_data, 'BTC/USDT')
                signal = signal_result['signal']
                confidence = signal_result['confidence']
                
                if signal != 'hold':
                    signals_generated += 1
                
                # Simple position management
                if position is None and signal in ['buy', 'sell'] and confidence > 0.3:
                    # Enter position
                    position = {
                        'type': signal,
                        'entry_price': current_price,
                        'entry_time': current_data.index[-1],
                        'confidence': confidence
                    }
                
                elif position is not None:
                    # Simple exit logic
                    should_exit = False
                    
                    if position['type'] == 'buy' and (signal == 'sell' or 
                        current_price < position['entry_price'] * 0.98):
                        should_exit = True
                    elif position['type'] == 'sell' and (signal == 'buy' or
                        current_price > position['entry_price'] * 1.02):
                        should_exit = True
                    
                    if should_exit:
                        # Calculate return
                        if position['type'] == 'buy':
                            return_pct = (current_price - position['entry_price']) / position['entry_price']
                        else:
                            return_pct = (position['entry_price'] - current_price) / position['entry_price']
                        
                        trades.append({
                            'entry_time': position['entry_time'],
                            'exit_time': current_data.index[-1],
                            'type': position['type'],
                            'return_pct': return_pct,
                            'confidence': position['confidence']
                        })
                        
                        position = None
                        
            except Exception as e:
                # Skip problematic periods
                continue
        
        # Calculate simple metrics
        if trades:
            returns = [t['return_pct'] for t in trades]
            total_return = np.prod([1 + r for r in returns]) - 1
            win_rate = len([r for r in returns if r > 0]) / len(returns)
            avg_return = np.mean(returns)
            max_loss = min(returns) if returns else 0
            max_gain = max(returns) if returns else 0
        else:
            total_return = win_rate = avg_return = max_loss = max_gain = 0
        
        # Display results
        print(f"\n📊 BACKTEST RESULTS")
        print("=" * 25)
        print(f"📈 Total Return: {total_return:.2%}")
        print(f"🎯 Win Rate: {win_rate:.2%}")
        print(f"📊 Average Trade: {avg_return:.2%}")
        print(f"📉 Max Loss: {max_loss:.2%}")
        print(f"📈 Max Gain: {max_gain:.2%}")
        print(f"🔢 Total Trades: {len(trades)}")
        print(f"⚡ Signals Generated: {signals_generated}")
        
        # Strategy assessment
        if total_return > 0.05:  # 5%
            assessment = "🟢 Good Performance"
        elif total_return > 0:
            assessment = "🟡 Modest Performance"
        else:
            assessment = "🔴 Needs Optimization"
        
        print(f"\n{assessment}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            'timestamp': timestamp,
            'config': config,
            'metrics': {
                'total_return': total_return,
                'win_rate': win_rate,
                'avg_return': avg_return,
                'max_loss': max_loss,
                'max_gain': max_gain,
                'total_trades': len(trades),
                'signals_generated': signals_generated
            },
            'trades': trades[:10]  # Save first 10 trades as examples
        }
        
        results_file = f"quick_candle_momentum_backtest_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Performance insights
        print(f"\n🔍 INSIGHTS:")
        if len(trades) > 0:
            print(f"• Strategy generated {signals_generated} signals over {len(market_data)} periods")
            print(f"• Trade frequency: {len(trades)/len(market_data)*100:.1f}% of periods")
            if win_rate > 0.5:
                print(f"• Good win rate indicates strong momentum detection")
            if avg_return > 0.01:
                print(f"• Strong average returns suggest good entry/exit timing")
        else:
            print(f"• No trades executed - strategy may be too conservative")
            print(f"• Consider lowering confidence threshold or momentum ratio")
        
        return results
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🚀 Starting Quick Candle Momentum Backtest...")
    results = run_quick_backtest()
    
    if results:
        print("\n✅ Quick backtest completed!")
        print("\n💡 Next Steps:")
        print("1. Run longer backtests with more data")
        print("2. Test different parameter combinations")
        print("3. Implement on paper trading first")
        print("4. Monitor performance in live conditions")
    else:
        print("\n❌ Backtest failed - check configuration")