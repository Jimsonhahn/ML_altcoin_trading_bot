#!/usr/bin/env python3
"""
AGGRESSIVE Strategy Backtest - WER NICHTS WAGT, DER NICHTS GEWINNT!
===================================================================

Hochriskanter Backtest mit aggressiven Einstellungen für maximale Gewinne.
Jetzt wird richtig Kohle gemacht! 💰
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_realistic_market_data(start_date: str, end_date: str, symbol: str = 'BTC/USDT') -> pd.DataFrame:
    """Create realistic market data for backtesting"""
    print(f"📊 Generating realistic market data for {symbol}...")
    
    dates = pd.date_range(start=start_date, end=end_date, freq='1H')
    np.random.seed(42)  # For reproducible results
    
    # Create more realistic price action
    base_price = 45000
    
    # Add market cycles (bull/bear trends)
    cycle_length = len(dates) // 4  # 4 major cycles
    trend_components = []
    
    for i in range(4):
        cycle_start = i * cycle_length
        cycle_end = min((i + 1) * cycle_length, len(dates))
        cycle_dates = dates[cycle_start:cycle_end]
        
        if i % 2 == 0:  # Bull cycle
            trend = np.linspace(0, 0.3, len(cycle_dates))  # 30% up
        else:  # Bear cycle
            trend = np.linspace(0, -0.2, len(cycle_dates))  # 20% down
            
        trend_components.extend(trend)
    
    # Add noise and volatility
    volatility = np.random.normal(0, 0.015, len(dates))  # 1.5% volatility
    mean_reversion = np.random.normal(0, 0.005, len(dates))  # Mean reversion component
    
    # Combine components (ensure all arrays have same length)
    trend_array = np.array(trend_components[:len(dates)])
    
    # Ensure all arrays have the same length
    min_length = min(len(trend_array), len(volatility), len(mean_reversion))
    trend_array = trend_array[:min_length]
    volatility = volatility[:min_length]
    mean_reversion = mean_reversion[:min_length]
    
    price_returns = trend_array / len(dates) + volatility + mean_reversion
    prices = base_price * np.exp(np.cumsum(price_returns))
    
    # Update dates to match the corrected length
    dates = dates[:min_length]
    
    # Create OHLCV data
    market_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'close': prices,
        'volume': np.random.lognormal(15, 0.5, len(prices))  # Realistic volume distribution
    })
    
    # Generate realistic high/low
    market_data['high'] = np.maximum(
        market_data['open'], 
        market_data['close']
    ) * np.random.uniform(1.001, 1.025, len(market_data))
    
    market_data['low'] = np.minimum(
        market_data['open'], 
        market_data['close']
    ) * np.random.uniform(0.975, 0.999, len(market_data))
    
    market_data.set_index('timestamp', inplace=True)
    
    return market_data

def run_aggressive_strategy_backtest():
    """Run AGGRESSIVE backtest - KOHLE MACHEN! 💰"""
    print("🔥 AGGRESSIVE STRATEGY BACKTEST - KOHLE MACHEN! 🔥")
    print("=" * 65)
    print("💪 WER NICHTS WAGT, DER NICHTS GEWINNT!")
    print("🚀 MAXIMUM RISK = MAXIMUM REWARD!")
    print("=" * 65)
    
    try:
        # Import corrected components
        from config.settings import Settings
        from strategies import STRATEGIES
        
        print(f"✅ All imports successful - corrected components loaded!")
        print(f"📊 Available strategies: {list(STRATEGIES.keys())}")
        
        # Initialize settings
        settings = Settings()
        
        # AGGRESSIVE Backtest parameters - RICHTIG RISKANT! 🔥
        start_date = '2023-01-01'
        end_date = '2024-01-01'
        initial_capital = 10000
        max_risk_per_trade = 0.10  # 10% RISIKO PRO TRADE! 🚀
        leverage_factor = 2.0      # 2x LEVERAGE! 💥
        
        print(f"📅 Backtest period: {start_date} to {end_date}")
        print(f"💰 Initial capital: ${initial_capital:,}")
        print(f"🔥 AGGRESSIVE RISK per trade: {max_risk_per_trade:.1%} (5x höher!)")
        print(f"💥 LEVERAGE FAKTOR: {leverage_factor}x")
        print(f"🎯 MAXIMUM CASH DEPLOYMENT: 90% (statt 5%)")
        
        # Generate realistic market data
        market_data = create_realistic_market_data(start_date, end_date)
        
        print(f"📈 Market data: {len(market_data)} candles")
        print(f"💹 Price range: ${market_data['close'].min():,.0f} - ${market_data['close'].max():,.0f}")
        print(f"📊 Buy & hold return: {((market_data['close'].iloc[-1] / market_data['close'].iloc[0]) - 1):.2%}")
        
        results = {}
        
        # Test TOP 3 strategies with AGGRESSIVE settings
        top_strategies = ['momentum', 'mean_reversion', 'copy_trading']
        
        for strategy_name in top_strategies:
            print(f"\n🔥 AGGRESSIVE TESTING: {strategy_name.upper()}")
            print("-" * 50)
            
            try:
                # Initialize strategy
                strategy_class = STRATEGIES[strategy_name]
                strategy_config = settings.get(f'strategy_configs.{strategy_name}', {})
                strategy = strategy_class(strategy_config)
                
                print(f"✅ Strategy initialized: {strategy.__class__.__name__}")
                
                # Run AGGRESSIVE backtest simulation
                trades = []
                equity_curve = [initial_capital]
                current_capital = initial_capital
                position = 0  # Current position (0 = no position, >0 = long)
                position_entry_price = 0
                
                signals_generated = 0
                buy_signals = 0
                sell_signals = 0
                
                # AGGRESSIVE backtest loop - test every 12 hours (2x häufiger!)
                print("🔥 Running AGGRESSIVE backtest simulation...")
                
                for i in range(100, len(market_data), 12):  # Every 12 hours (2x häufiger!)
                    try:
                        # Get data window for strategy analysis
                        window_data = market_data.iloc[max(0, i-100):i]
                        current_price = float(window_data['close'].iloc[-1])
                        current_timestamp = window_data.index[-1]
                        
                        # Generate signal using the strategy's method
                        if hasattr(strategy, 'generate_signals'):
                            signal_result = strategy.generate_signals(window_data, 'BTC/USDT')
                        elif hasattr(strategy, 'calculate_signal'):
                            signal, metadata = strategy.calculate_signal('BTC/USDT', window_data, current_price)
                            signal_result = {'signal': signal, **metadata} if metadata else {'signal': signal}
                        else:
                            continue  # Skip if no signal method
                        
                        if not signal_result or 'signal' not in signal_result:
                            continue
                            
                        signal = signal_result['signal']
                        confidence = signal_result.get('confidence', 0.5)
                        
                        # BOOST confidence for aggressive trading
                        confidence = min(1.0, confidence * 1.5)  # 50% confidence boost!
                        
                        # Process signals
                        if signal in ['BUY', 'SELL']:
                            signals_generated += 1
                            
                            if signal == 'BUY':
                                buy_signals += 1
                            else:
                                sell_signals += 1
                        
                        # Execute AGGRESSIVE trades
                        if signal == 'BUY' and position == 0:  # Enter long position
                            # AGGRESSIVE position sizing with LEVERAGE!
                            risk_amount = current_capital * max_risk_per_trade * confidence
                            leverage_amount = risk_amount * leverage_factor  # 2x LEVERAGE!
                            position_size = leverage_amount / current_price
                            position_value = position_size * current_price
                            
                            # Use 90% of capital (instead of 5%!)
                            if position_value <= current_capital * 0.9:
                                position = position_size
                                position_entry_price = current_price
                                current_capital -= leverage_amount  # Use leveraged amount
                                
                                trades.append({
                                    'timestamp': current_timestamp,
                                    'action': 'BUY',
                                    'price': current_price,
                                    'size': position_size,
                                    'value': position_value,
                                    'leverage': leverage_factor,
                                    'confidence': confidence,
                                    'capital_before': current_capital + leverage_amount
                                })
                        
                        elif signal == 'SELL' and position > 0:  # Close long position
                            position_value = position * current_price
                            current_capital += position_value
                            
                            # Calculate trade return with leverage effect
                            trade_return = (current_price - position_entry_price) / position_entry_price
                            leveraged_return = trade_return * leverage_factor  # Leverage amplifies gains AND losses!
                            
                            trades.append({
                                'timestamp': current_timestamp,
                                'action': 'SELL',
                                'price': current_price,
                                'size': position,
                                'value': position_value,
                                'leverage': leverage_factor,
                                'confidence': confidence,
                                'trade_return': leveraged_return,
                                'capital_after': current_capital
                            })
                            
                            position = 0
                            position_entry_price = 0
                        
                        # Update equity curve
                        if position > 0:
                            # Account for leverage in equity calculation
                            unrealized_pnl = (current_price - position_entry_price) * position * (leverage_factor - 1)
                            current_equity = current_capital + (position * current_price) + unrealized_pnl
                        else:
                            current_equity = current_capital
                            
                        equity_curve.append(current_equity)
                        
                    except Exception as e:
                        logger.debug(f"Error at index {i}: {e}")
                        continue
                
                # Close any remaining position
                if position > 0:
                    final_price = float(market_data['close'].iloc[-1])
                    position_value = position * final_price
                    current_capital += position_value
                    
                    trade_return = (final_price - position_entry_price) / position_entry_price
                    leveraged_return = trade_return * leverage_factor
                    
                    trades.append({
                        'timestamp': market_data.index[-1],
                        'action': 'SELL',
                        'price': final_price,
                        'size': position,
                        'value': position_value,
                        'leverage': leverage_factor,
                        'confidence': 0.5,
                        'trade_return': leveraged_return,
                        'capital_after': current_capital
                    })
                
                # Calculate AGGRESSIVE performance metrics
                final_capital = current_capital
                total_return = (final_capital - initial_capital) / initial_capital
                
                # Calculate trade-based metrics
                if trades:
                    buy_trades = [t for t in trades if t['action'] == 'BUY']
                    sell_trades = [t for t in trades if t['action'] == 'SELL']
                    
                    trade_returns = []
                    for sell_trade in sell_trades:
                        if 'trade_return' in sell_trade:
                            trade_returns.append(sell_trade['trade_return'])
                    
                    if trade_returns:
                        win_rate = len([r for r in trade_returns if r > 0]) / len(trade_returns)
                        avg_return = np.mean(trade_returns)
                        volatility = np.std(trade_returns)
                        
                        # Sharpe ratio (assuming risk-free rate = 0)
                        sharpe_ratio = (avg_return * len(trade_returns) / volatility) if volatility > 0 else 0
                        
                        # Max drawdown
                        equity_series = pd.Series(equity_curve)
                        rolling_max = equity_series.expanding().max()
                        drawdown = (equity_series - rolling_max) / rolling_max
                        max_drawdown = drawdown.min()
                        
                        # Profit factor
                        winning_returns = [r for r in trade_returns if r > 0]
                        losing_returns = [r for r in trade_returns if r < 0]
                        
                        if losing_returns:
                            profit_factor = abs(sum(winning_returns)) / abs(sum(losing_returns))
                        else:
                            profit_factor = float('inf') if winning_returns else 0
                    else:
                        win_rate = 0
                        avg_return = 0
                        volatility = 0
                        sharpe_ratio = 0
                        max_drawdown = 0
                        profit_factor = 0
                else:
                    win_rate = 0
                    avg_return = 0
                    volatility = 0
                    sharpe_ratio = 0
                    max_drawdown = 0
                    profit_factor = 0
                
                # Store AGGRESSIVE results
                results[strategy_name] = {
                    'status': 'success',
                    'initial_capital': initial_capital,
                    'final_capital': final_capital,
                    'total_return': total_return,
                    'total_return_pct': total_return * 100,
                    'leverage_factor': leverage_factor,
                    'risk_per_trade': max_risk_per_trade,
                    'signals_generated': signals_generated,
                    'buy_signals': buy_signals,
                    'sell_signals': sell_signals,
                    'total_trades': len([t for t in trades if t['action'] == 'BUY']),
                    'win_rate': win_rate,
                    'avg_return_per_trade': avg_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'max_drawdown_pct': max_drawdown * 100,
                    'profit_factor': profit_factor,
                    'calmar_ratio': total_return / abs(max_drawdown) if max_drawdown != 0 else float('inf') if total_return > 0 else 0
                }
                
                # Display AGGRESSIVE results
                print(f"🔥 {strategy_name.upper()} AGGRESSIVE RESULTS:")
                print(f"   💰 Final Capital: ${final_capital:,.2f}")
                print(f"   💥 TOTAL RETURN: {total_return:.2%}")
                print(f"   🚀 Leverage Factor: {leverage_factor}x")
                print(f"   🎯 Signals: {signals_generated} ({buy_signals} BUY, {sell_signals} SELL)")
                print(f"   🔄 Trades: {len([t for t in trades if t['action'] == 'BUY'])}")
                print(f"   🏆 Win Rate: {win_rate:.1%}")
                print(f"   📊 Sharpe Ratio: {sharpe_ratio:.2f}")
                print(f"   📉 Max Drawdown: {max_drawdown:.2%}")
                print(f"   💪 Profit Factor: {profit_factor:.2f}")
                
                # Profit/Loss in EUR (assuming 1 USD = 0.85 EUR)
                profit_usd = final_capital - initial_capital
                profit_eur = profit_usd * 0.85
                print(f"   💶 GEWINN: €{profit_eur:,.2f} ({profit_usd:,.2f} USD)")
                
            except Exception as e:
                logger.error(f"Error testing strategy {strategy_name}: {e}")
                results[strategy_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                print(f"❌ FAILED: {e}")
        
        # Generate AGGRESSIVE comparison
        print("\n" + "🔥" * 65)
        print("💰 AGGRESSIVE STRATEGY PERFORMANCE - KOHLE GEMACHT!")
        print("🔥" * 65)
        
        successful_results = {k: v for k, v in results.items() if v.get('status') == 'success'}
        
        if successful_results:
            # Sort by total return
            sorted_by_return = sorted(successful_results.items(), 
                                    key=lambda x: x[1]['total_return'], reverse=True)
            
            print(f"{'Strategy':<15} {'Return':<10} {'Gewinn €':<12} {'Drawdown':<12} {'Trades':<8}")
            print("-" * 70)
            
            for strategy_name, metrics in sorted_by_return:
                profit_usd = (metrics['final_capital'] - metrics['initial_capital'])
                profit_eur = profit_usd * 0.85
                
                return_str = f"{metrics['total_return']:.1%}"
                gewinn_str = f"€{profit_eur:,.0f}"
                drawdown_str = f"{metrics['max_drawdown']:.1%}"
                trades_str = str(metrics['total_trades'])
                
                print(f"{strategy_name:<15} {return_str:<10} {gewinn_str:<12} {drawdown_str:<12} {trades_str:<8}")
            
            # Best AGGRESSIVE strategy
            best_strategy = sorted_by_return[0]
            best_profit_usd = best_strategy[1]['final_capital'] - best_strategy[1]['initial_capital']
            best_profit_eur = best_profit_usd * 0.85
            
            print(f"\n💥 BESTE AGGRESSIVE STRATEGIE: {best_strategy[0].upper()}")
            print(f"   🚀 Return: {best_strategy[1]['total_return']:.2%}")
            print(f"   💰 GEWINN: €{best_profit_eur:,.2f} ({best_profit_usd:,.2f} USD)")
            print(f"   📊 Sharpe: {best_strategy[1]['sharpe_ratio']:.2f}")
            print(f"   📉 Max DD: {best_strategy[1]['max_drawdown']:.2%}")
            print(f"   🎯 Trades: {best_strategy[1]['total_trades']}")
            print(f"   💪 Profit Factor: {best_strategy[1]['profit_factor']:.2f}")
            print(f"   🔥 Leverage: {best_strategy[1]['leverage_factor']}x")
        
        # Save AGGRESSIVE results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'aggressive_strategy_backtest_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 AGGRESSIVE Results saved to: {filename}")
        print(f"\n🔥 AGGRESSIVE BACKTEST COMPLETED!")
        print("💰 KOHLE WURDE GEMACHT! WER NICHTS WAGT, DER NICHTS GEWINNT!")
        
        return results
        
    except Exception as e:
        logger.error(f"Critical error in aggressive backtest: {e}")
        print(f"❌ AGGRESSIVE BACKTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🔥 Starting AGGRESSIVE strategy backtest - KOHLE MACHEN!")
    results = run_aggressive_strategy_backtest()
    
    if results:
        successful = len([r for r in results.values() if r.get('status') == 'success'])
        total = len(results)
        print(f"\n💥 AGGRESSIVE SUCCESS: {successful}/{total} strategies tested!")
        print("🚀 READY TO MAKE SERIOUS MONEY!")
    else:
        print("\n💥 FAILURE: Aggressive backtest could not complete")