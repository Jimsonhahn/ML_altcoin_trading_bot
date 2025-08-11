#!/usr/bin/env python3
"""
Enhanced Candle Momentum Strategy Backtest
==========================================

Improved version with better parameter optimization and enhanced strategy logic.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class EnhancedCandleMomentumStrategy:
    """Enhanced version of candle momentum strategy with better logic"""
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.lookback_period = params.get('lookback_period', 15)
        self.sma_period = params.get('sma_period', 30)
        self.min_momentum_ratio = params.get('min_momentum_ratio', 1.2)
        self.min_confidence = params.get('min_confidence', 0.4)
        self.volume_filter = params.get('volume_filter', True)
        self.use_rsi = params.get('use_rsi', True)
        self.rsi_oversold = params.get('rsi_oversold', 30)
        self.rsi_overbought = params.get('rsi_overbought', 70)
        
        self.last_signals = {}
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_momentum_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate enhanced momentum indicators"""
        # Basic candle body momentum
        bullish_bodies = pd.Series(0.0, index=data.index)
        bearish_bodies = pd.Series(0.0, index=data.index)
        
        bullish_mask = data['close'] > data['open']
        bearish_mask = data['close'] < data['open']
        
        body_size = abs(data['close'] - data['open'])
        bullish_bodies[bullish_mask] = body_size[bullish_mask]
        bearish_bodies[bearish_mask] = body_size[bearish_mask]
        
        # Rolling momentum strength
        bullish_strength = bullish_bodies.rolling(window=self.lookback_period).sum()
        bearish_strength = bearish_bodies.rolling(window=self.lookback_period).sum()
        
        # Momentum ratio with smoothing
        momentum_ratio = bullish_strength / (bearish_strength + 1e-8)
        momentum_ratio_smooth = momentum_ratio.rolling(window=3).mean()
        
        # Trend filter
        trend_line = data['close'].rolling(window=self.sma_period).mean()
        
        # Volume filter
        if self.volume_filter:
            avg_volume = data['volume'].rolling(window=20).mean()
            volume_ok = data['volume'] > avg_volume
        else:
            volume_ok = pd.Series(True, index=data.index)
        
        # RSI for overbought/oversold conditions
        rsi = self.calculate_rsi(data['close']) if self.use_rsi else pd.Series(50, index=data.index)
        
        return {
            'bullish_strength': bullish_strength,
            'bearish_strength': bearish_strength,
            'momentum_ratio': momentum_ratio_smooth,
            'trend_line': trend_line,
            'volume_ok': volume_ok,
            'rsi': rsi
        }
    
    def generate_enhanced_signals(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Generate enhanced trading signals"""
        if len(data) < max(self.lookback_period, self.sma_period) + 10:
            return {'signal': 'hold', 'confidence': 0.0, 'metadata': {}}
        
        try:
            indicators = self.calculate_momentum_indicators(data)
            
            current_price = data['close'].iloc[-1]
            momentum_ratio = indicators['momentum_ratio'].iloc[-1]
            trend_value = indicators['trend_line'].iloc[-1]
            volume_ok = indicators['volume_ok'].iloc[-1]
            rsi = indicators['rsi'].iloc[-1]
            
            # Detect momentum crossovers with confirmation
            momentum_series = indicators['momentum_ratio']
            momentum_increasing = momentum_series.iloc[-1] > momentum_series.iloc[-2]
            momentum_strong = momentum_ratio > self.min_momentum_ratio
            momentum_weak = momentum_ratio < (1.0 / self.min_momentum_ratio)
            
            # Price vs trend
            price_above_trend = current_price > trend_value
            price_below_trend = current_price < trend_value
            
            # Enhanced signal logic
            signal = 'hold'
            confidence = 0.0
            
            # Long signal conditions (multiple confirmations)
            long_conditions = [
                momentum_strong,  # Strong bullish momentum
                price_above_trend,  # Price above trend
                momentum_increasing,  # Momentum increasing
                volume_ok,  # Volume confirmation
                not self.use_rsi or rsi < self.rsi_overbought  # Not overbought
            ]
            
            if sum(long_conditions) >= 3:  # Need at least 3 confirmations
                signal = 'buy'
                confidence = min(sum(long_conditions) / len(long_conditions), 1.0)
                confidence *= min(momentum_ratio / 2.0, 1.0)  # Scale by momentum strength
            
            # Short signal conditions
            short_conditions = [
                momentum_weak,  # Strong bearish momentum
                price_below_trend,  # Price below trend
                not momentum_increasing,  # Momentum decreasing
                volume_ok,  # Volume confirmation
                not self.use_rsi or rsi > self.rsi_oversold  # Not oversold
            ]
            
            if sum(short_conditions) >= 3:  # Need at least 3 confirmations
                signal = 'sell'
                confidence = min(sum(short_conditions) / len(short_conditions), 1.0)
                confidence *= min(2.0 / max(momentum_ratio, 0.1), 1.0)
            
            # Apply minimum confidence filter
            if confidence < self.min_confidence:
                signal = 'hold'
                confidence = 0.0
            
            return {
                'signal': signal,
                'confidence': confidence,
                'metadata': {
                    'momentum_ratio': momentum_ratio,
                    'trend_value': trend_value,
                    'current_price': current_price,
                    'rsi': rsi,
                    'volume_ok': volume_ok,
                    'long_conditions_met': sum(long_conditions),
                    'short_conditions_met': sum(short_conditions)
                }
            }
            
        except Exception as e:
            return {'signal': 'hold', 'confidence': 0.0, 'metadata': {'error': str(e)}}

def run_enhanced_optimization():
    """Run enhanced parameter optimization"""
    print("🔧 ENHANCED PARAMETER OPTIMIZATION")
    print("=" * 40)
    
    # Generate realistic test data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='1H')
    
    # Create varied market conditions
    base_price = 40000
    returns = []
    
    # Bull phase (first 3 months)
    bull_returns = np.random.normal(0.0003, 0.008, len(dates)//4)
    # Bear phase (next 3 months)
    bear_returns = np.random.normal(-0.0002, 0.012, len(dates)//4)
    # Sideways (next 3 months)
    sideways_returns = np.random.normal(0, 0.006, len(dates)//4)
    # Recovery (last 3 months)
    recovery_returns = np.random.normal(0.0002, 0.009, len(dates) - 3*(len(dates)//4))
    
    all_returns = np.concatenate([bull_returns, bear_returns, sideways_returns, recovery_returns])
    
    prices = [base_price]
    for ret in all_returns[:-1]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    market_data = []
    for i, (timestamp, close_price) in enumerate(zip(dates, prices)):
        open_price = prices[i-1] if i > 0 else close_price
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.003)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.003)))
        volume = np.random.lognormal(15, 0.5)
        
        market_data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(market_data)
    df.set_index('timestamp', inplace=True)
    
    print(f"📊 Test data: {len(df)} hours, price range ${min(prices):,.0f} - ${max(prices):,.0f}")
    
    # Enhanced parameter combinations
    param_grid = [
        # More aggressive settings
        {'lookback_period': 8, 'sma_period': 15, 'min_momentum_ratio': 1.1, 'min_confidence': 0.3, 'use_rsi': True},
        {'lookback_period': 10, 'sma_period': 18, 'min_momentum_ratio': 1.15, 'min_confidence': 0.35, 'use_rsi': True},
        {'lookback_period': 12, 'sma_period': 22, 'min_momentum_ratio': 1.2, 'min_confidence': 0.4, 'use_rsi': True},
        # Balanced settings
        {'lookback_period': 15, 'sma_period': 25, 'min_momentum_ratio': 1.25, 'min_confidence': 0.45, 'use_rsi': True},
        {'lookback_period': 18, 'sma_period': 30, 'min_momentum_ratio': 1.3, 'min_confidence': 0.5, 'use_rsi': True},
        # Without RSI
        {'lookback_period': 10, 'sma_period': 20, 'min_momentum_ratio': 1.15, 'min_confidence': 0.35, 'use_rsi': False},
        {'lookback_period': 15, 'sma_period': 25, 'min_momentum_ratio': 1.2, 'min_confidence': 0.4, 'use_rsi': False},
    ]
    
    best_params = None
    best_score = -999
    results = {}
    
    for i, params in enumerate(param_grid):
        print(f"\n🧪 Testing combination {i+1}/{len(param_grid)}")
        print(f"   Parameters: {params}")
        
        try:
            strategy = EnhancedCandleMomentumStrategy(params)
            
            # Simulate trading
            capital = 10000
            position = None
            trades = []
            
            # Test on subset for speed
            test_data = df.iloc[::12]  # Every 12 hours
            
            for idx in range(50, len(test_data)):
                current_data = df.iloc[:df.index.get_loc(test_data.index[idx])+1]
                current_price = current_data['close'].iloc[-1]
                current_time = current_data.index[-1]
                
                # Generate signal
                signal_result = strategy.generate_enhanced_signals(current_data, 'BTC/USDT')
                signal = signal_result['signal']
                confidence = signal_result['confidence']
                
                # Simple position management
                if position is None and signal != 'hold' and confidence > 0.3:
                    position = {
                        'type': signal,
                        'entry_price': current_price,
                        'entry_time': current_time,
                        'confidence': confidence
                    }
                
                elif position is not None:
                    # Exit conditions
                    bars_held = idx - getattr(position, 'entry_idx', idx-1)
                    should_exit = False
                    
                    if signal != 'hold' and signal != position['type']:
                        should_exit = True
                    elif bars_held > 168:  # 1 week max
                        should_exit = True
                    elif position['type'] == 'buy' and current_price < position['entry_price'] * 0.95:
                        should_exit = True
                    elif position['type'] == 'sell' and current_price > position['entry_price'] * 1.05:
                        should_exit = True
                    
                    if should_exit:
                        if position['type'] == 'buy':
                            ret = (current_price - position['entry_price']) / position['entry_price']
                        else:
                            ret = (position['entry_price'] - current_price) / position['entry_price']
                        
                        trades.append({
                            'return': ret,
                            'confidence': position['confidence'],
                            'hold_time': bars_held
                        })
                        position = None
            
            # Calculate metrics
            if trades:
                returns = [t['return'] for t in trades]
                total_return = np.prod([1 + r for r in returns]) - 1
                win_rate = len([r for r in returns if r > 0]) / len(returns)
                sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                
                # Composite score
                score = total_return * 0.4 + sharpe * 0.3 + win_rate * 0.3
                
                results[f"combo_{i+1}"] = {
                    'params': params,
                    'total_return': total_return,
                    'win_rate': win_rate,
                    'sharpe': sharpe,
                    'score': score,
                    'trades': len(trades)
                }
                
                print(f"   📊 Return: {total_return:.2%}")
                print(f"   🎯 Win Rate: {win_rate:.2%}")
                print(f"   📈 Sharpe: {sharpe:.2f}")
                print(f"   🏆 Score: {score:.3f}")
                print(f"   🔢 Trades: {len(trades)}")
                
                if score > best_score and len(trades) >= 3:
                    best_score = score
                    best_params = params.copy()
            else:
                print("   ❌ No trades generated")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    if best_params:
        print(f"\n🏆 BEST PARAMETERS FOUND:")
        print(f"   {best_params}")
        print(f"   Score: {best_score:.3f}")
    else:
        best_params = {'lookback_period': 12, 'sma_period': 22, 'min_momentum_ratio': 1.2, 'min_confidence': 0.4, 'use_rsi': True}
        print(f"\n⚠️  Using fallback parameters: {best_params}")
    
    return best_params, results

def run_final_backtest(params: Dict) -> Dict:
    """Run final comprehensive backtest"""
    print(f"\n🚀 FINAL ENHANCED BACKTEST")
    print("=" * 30)
    
    # Generate 3-year realistic data
    print("📊 Generating 3-year market data...")
    
    np.random.seed(123)  # Different seed for final test
    dates = pd.date_range('2021-01-01', '2024-01-01', freq='2H')  # 2-hour bars for performance
    
    # Create realistic crypto market with major events
    base_price = 35000
    prices = [base_price]
    
    total_hours = len(dates)
    
    # Major crypto events simulation
    events = [
        (0.15, 0.8, 0.015),      # Bull run 2021
        (0.25, -0.6, 0.025),     # Crash mid-2021
        (0.20, -0.4, 0.020),     # Bear market
        (0.15, 0.3, 0.012),      # Recovery
        (0.25, 0.4, 0.018)       # Final growth
    ]
    
    current_idx = 0
    for phase_pct, trend, volatility in events:
        phase_length = int(total_hours * phase_pct)
        
        for i in range(phase_length):
            if current_idx >= total_hours - 1:
                break
                
            # Trend + noise
            trend_component = trend / (365 * 12)  # Hourly trend
            noise = np.random.normal(0, volatility / np.sqrt(365 * 12))
            
            change = trend_component + noise
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1000))  # Price floor
            current_idx += 1
    
    # Fill remaining with small changes
    while len(prices) < len(dates):
        change = np.random.normal(0, 0.001)
        prices.append(prices[-1] * (1 + change))
    
    prices = prices[:len(dates)]
    
    # Create OHLCV
    market_data = []
    for i, (timestamp, close_price) in enumerate(zip(dates, prices)):
        open_price = prices[i-1] if i > 0 else close_price
        
        daily_range = abs(close_price - open_price) + close_price * 0.002
        high_price = max(open_price, close_price) + daily_range * np.random.uniform(0, 0.6)
        low_price = min(open_price, close_price) - daily_range * np.random.uniform(0, 0.6)
        volume = np.random.lognormal(16, 0.6)  # Higher base volume
        
        market_data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(market_data)
    df.set_index('timestamp', inplace=True)
    
    print(f"📈 Data: {len(df)} periods, range ${min(prices):,.0f} - ${max(prices):,.0f}")
    
    # Run backtest
    strategy = EnhancedCandleMomentumStrategy(params)
    
    capital = 10000
    position = None
    trades = []
    equity_curve = [capital]
    
    print("⚡ Running backtest...")
    
    warmup = max(params['lookback_period'], params['sma_period']) + 10
    
    for idx in range(warmup, len(df)):
        try:
            current_data = df.iloc[:idx+1]
            current_price = current_data['close'].iloc[-1]
            current_time = current_data.index[-1]
            
            # Generate signal
            signal_result = strategy.generate_enhanced_signals(current_data, 'BTC/USDT')
            signal = signal_result['signal']
            confidence = signal_result['confidence']
            
            # Position management
            if position is None and signal != 'hold':
                # Enter position
                risk_amount = capital * 0.02  # 2% risk
                position = {
                    'type': signal,
                    'entry_price': current_price,
                    'entry_time': current_time,
                    'entry_idx': idx,
                    'risk_amount': risk_amount,
                    'confidence': confidence
                }
            
            elif position is not None:
                # Check exit
                bars_held = idx - position['entry_idx']
                should_exit = False
                exit_reason = ''
                
                if signal != 'hold' and signal != position['type']:
                    should_exit = True
                    exit_reason = 'signal_change'
                elif bars_held > 84:  # Max 1 week (84 * 2h)
                    should_exit = True
                    exit_reason = 'time_limit'
                elif position['type'] == 'buy' and current_price < position['entry_price'] * 0.95:
                    should_exit = True
                    exit_reason = 'stop_loss'
                elif position['type'] == 'sell' and current_price > position['entry_price'] * 1.05:
                    should_exit = True
                    exit_reason = 'stop_loss'
                
                if should_exit:
                    # Calculate return
                    if position['type'] == 'buy':
                        ret = (current_price - position['entry_price']) / position['entry_price']
                    else:
                        ret = (position['entry_price'] - current_price) / position['entry_price']
                    
                    # Apply to capital (simplified)
                    pnl = position['risk_amount'] * ret * 10  # 10x leverage effect
                    capital += pnl
                    
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'type': position['type'],
                        'return_pct': ret,
                        'pnl': pnl,
                        'confidence': position['confidence'],
                        'exit_reason': exit_reason,
                        'bars_held': bars_held
                    })
                    
                    position = None
            
            # Record equity (every 24 periods = daily)
            if idx % 24 == 0:
                equity_curve.append(capital)
                
        except Exception:
            continue
    
    # Close final position
    if position:
        final_price = df['close'].iloc[-1]
        if position['type'] == 'buy':
            ret = (final_price - position['entry_price']) / position['entry_price']
        else:
            ret = (position['entry_price'] - final_price) / position['entry_price']
        
        pnl = position['risk_amount'] * ret * 10
        capital += pnl
        
        trades.append({
            'entry_time': position['entry_time'],
            'exit_time': df.index[-1],
            'type': position['type'],
            'return_pct': ret,
            'pnl': pnl,
            'confidence': position['confidence'],
            'exit_reason': 'backtest_end',
            'bars_held': len(df) - position['entry_idx']
        })
    
    return {
        'initial_capital': 10000,
        'final_capital': capital,
        'trades': trades,
        'equity_curve': equity_curve,
        'market_data_points': len(df)
    }

def analyze_enhanced_results(results: Dict) -> Dict:
    """Analyze enhanced backtest results"""
    print(f"\n📊 ENHANCED RESULTS ANALYSIS")
    print("=" * 30)
    
    initial = results['initial_capital']
    final = results['final_capital']
    trades = results['trades']
    
    if not trades:
        print("❌ No trades executed")
        return {}
    
    # Calculate metrics
    total_return = (final - initial) / initial
    
    returns = [t['return_pct'] for t in trades]
    winning_returns = [r for r in returns if r > 0]
    losing_returns = [r for r in returns if r < 0]
    
    win_rate = len(winning_returns) / len(returns)
    avg_win = np.mean(winning_returns) if winning_returns else 0
    avg_loss = np.mean(losing_returns) if losing_returns else 0
    
    # Risk metrics
    if len(returns) > 1:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365/7) if np.std(returns) > 0 else 0
    else:
        sharpe = 0
    
    # Time metrics
    hold_times = [t['bars_held'] for t in trades]
    avg_hold = np.mean(hold_times) * 2  # Convert to hours
    
    # Drawdown
    equity = results['equity_curve']
    peak = initial
    max_dd = 0
    for value in equity:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        max_dd = max(max_dd, dd)
    
    # Annual return
    years = 3
    annual_return = (1 + total_return) ** (1/years) - 1
    
    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'total_trades': len(trades),
        'avg_hold_hours': avg_hold,
        'profit_factor': abs(avg_win * len(winning_returns) / (avg_loss * len(losing_returns))) if losing_returns else float('inf'),
        'best_trade': max(returns),
        'worst_trade': min(returns)
    }
    
    # Display results
    print(f"💰 FINANCIAL PERFORMANCE:")
    print(f"   Initial: ${initial:,.2f}")
    print(f"   Final: ${final:,.2f}")
    print(f"   Total Return: {total_return:.2%}")
    print(f"   Annual Return: {annual_return:.2%}")
    
    print(f"\n📊 TRADING STATISTICS:")
    print(f"   Total Trades: {len(trades)}")
    print(f"   Win Rate: {win_rate:.2%}")
    print(f"   Avg Win: {avg_win:.2%}")
    print(f"   Avg Loss: {avg_loss:.2%}")
    print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
    
    print(f"\n⚖️  RISK METRICS:")
    print(f"   Sharpe Ratio: {sharpe:.2f}")
    print(f"   Max Drawdown: {max_dd:.2%}")
    print(f"   Avg Hold: {avg_hold:.1f} hours")
    
    # Rating
    if annual_return > 0.15 and sharpe > 1.0 and max_dd < 0.25:
        rating = "🟢 EXCELLENT"
    elif annual_return > 0.08 and sharpe > 0.5:
        rating = "🟡 GOOD"
    elif annual_return > 0.03:
        rating = "🟠 MODERATE"
    else:
        rating = "🔴 NEEDS IMPROVEMENT"
    
    print(f"\n🎯 OVERALL RATING: {rating}")
    
    return metrics

def main():
    """Main execution"""
    print("🚀 ENHANCED 3-YEAR CANDLE MOMENTUM BACKTEST")
    print("=" * 50)
    
    try:
        # Optimize parameters
        best_params, opt_results = run_enhanced_optimization()
        
        # Run final backtest
        backtest_results = run_final_backtest(best_params)
        
        # Analyze results
        analysis = analyze_enhanced_results(backtest_results)
        
        if analysis:
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"enhanced_candle_momentum_{timestamp}.json"
            
            final_results = {
                'strategy': 'enhanced_candle_momentum',
                'backtest_period': '2021-2024',
                'initial_investment': 10000,
                'optimization_results': opt_results,
                'best_parameters': best_params,
                'performance_metrics': analysis,
                'sample_trades': backtest_results['trades'][:15]
            }
            
            with open(results_file, 'w') as f:
                json.dump(final_results, f, indent=2, default=str)
            
            print(f"\n💾 Results saved: {results_file}")
            
            # Recommendations
            print(f"\n💡 FINAL RECOMMENDATIONS:")
            if analysis['annual_return'] > 0.1:
                print("✅ Strategy shows promise - consider paper trading")
            else:
                print("⚠️  Strategy needs further optimization")
                
            print(f"\n🔧 OPTIMAL SETTINGS:")
            for key, value in best_params.items():
                print(f"   {key}: {value}")
            
            return final_results
        else:
            print("❌ Analysis failed")
            return None
            
    except Exception as e:
        print(f"❌ Enhanced backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
    if results:
        print("\n✅ ENHANCED BACKTEST COMPLETED!")
    else:
        print("\n❌ Backtest failed")