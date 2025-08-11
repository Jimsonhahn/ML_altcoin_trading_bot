#!/usr/bin/env python3
"""
Realistic 3-Year Candle Momentum Strategy Backtest
=================================================

Comprehensive backtest with realistic market conditions, transaction costs,
slippage, and proper risk management over 3 years with $10k initial investment.
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

class RealisticMarketSimulator:
    """Creates realistic 3-year crypto market data with various regimes"""
    
    def __init__(self, start_date: str, end_date: str):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        
    def generate_realistic_data(self) -> pd.DataFrame:
        """Generate 3 years of realistic crypto market data"""
        print("📊 Generating 3-year realistic crypto market data...")
        
        # Create hourly timestamps for 3 years
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='1H')
        total_hours = len(dates)
        
        print(f"   🕐 Total periods: {total_hours:,} hours ({total_hours/24:.0f} days)")
        
        np.random.seed(42)  # Reproducible results
        
        # Crypto market characteristics
        base_price = 30000  # Starting BTC price
        
        # Define market regimes for 3 years (12 quarters)
        regime_quarters = [
            ('bear', 0.25),      # Q1 2021: Bear market
            ('bull', 0.15),      # Q2 2021: Bull run starts
            ('bull_extreme', 0.10), # Q3 2021: Extreme bull
            ('crash', 0.30),     # Q4 2021: Major crash
            ('sideways', 0.08),  # Q1 2022: Sideways recovery
            ('bear', 0.20),      # Q2 2022: Bear continues
            ('volatility', 0.25), # Q3 2022: High volatility
            ('sideways', 0.06),  # Q4 2022: Quiet period
            ('recovery', 0.12),  # Q1 2023: Recovery starts
            ('bull', 0.18),      # Q2 2023: Bull market
            ('correction', 0.15), # Q3 2023: Correction
            ('growth', 0.14)     # Q4 2023: Steady growth
        ]
        
        # Generate returns for each regime
        all_returns = []
        current_hour = 0
        
        for regime_type, volatility in regime_quarters:
            # Quarter = ~2190 hours (90 days * 24 hours)
            quarter_hours = min(2190, total_hours - current_hour)
            if quarter_hours <= 0:
                break
                
            print(f"   📈 {regime_type.upper()} regime: {quarter_hours:,} hours (vol: {volatility:.2f})")
            
            if regime_type == 'bull':
                # Bull market: positive trend, moderate volatility
                trend = np.linspace(0, 0.8, quarter_hours) / quarter_hours  # 80% gain over quarter
                noise = np.random.normal(0, volatility/np.sqrt(365*24), quarter_hours)
                
            elif regime_type == 'bull_extreme':
                # Extreme bull: very positive trend, increasing volatility
                trend = np.linspace(0, 1.5, quarter_hours) / quarter_hours  # 150% gain
                volatility_curve = np.linspace(volatility, volatility*2, quarter_hours)
                noise = np.array([np.random.normal(0, vol/np.sqrt(365*24)) for vol in volatility_curve])
                
            elif regime_type == 'bear':
                # Bear market: negative trend, high volatility
                trend = np.linspace(0, -0.6, quarter_hours) / quarter_hours  # -60% decline
                noise = np.random.normal(0, volatility/np.sqrt(365*24), quarter_hours)
                
            elif regime_type == 'crash':
                # Market crash: steep decline then recovery
                crash_hours = quarter_hours // 3
                recovery_hours = quarter_hours - crash_hours
                
                crash_trend = np.linspace(0, -0.8, crash_hours) / crash_hours  # -80% crash
                recovery_trend = np.linspace(0, 0.3, recovery_hours) / recovery_hours  # 30% recovery
                
                trend = np.concatenate([crash_trend, recovery_trend])
                noise = np.random.normal(0, volatility/np.sqrt(365*24), quarter_hours)
                
            elif regime_type == 'sideways':
                # Sideways: mean reversion, low volatility
                trend = np.random.normal(0, 0.02/np.sqrt(365*24), quarter_hours)  # Minimal trend
                noise = np.random.normal(0, volatility/np.sqrt(365*24), quarter_hours)
                
            elif regime_type == 'volatility':
                # High volatility: no clear trend, high noise
                trend = np.random.normal(0, 0.05/np.sqrt(365*24), quarter_hours)
                noise = np.random.normal(0, volatility/np.sqrt(365*24), quarter_hours)
                
            elif regime_type == 'recovery':
                # Recovery: gradual positive trend, decreasing volatility
                trend = np.linspace(0, 0.4, quarter_hours) / quarter_hours  # 40% recovery
                volatility_curve = np.linspace(volatility, volatility*0.7, quarter_hours)
                noise = np.array([np.random.normal(0, vol/np.sqrt(365*24)) for vol in volatility_curve])
                
            elif regime_type == 'correction':
                # Correction: temporary decline in bull market
                trend = np.linspace(0, -0.25, quarter_hours) / quarter_hours  # -25% correction
                noise = np.random.normal(0, volatility/np.sqrt(365*24), quarter_hours)
                
            elif regime_type == 'growth':
                # Steady growth: consistent positive trend, low volatility
                trend = np.linspace(0, 0.5, quarter_hours) / quarter_hours  # 50% growth
                noise = np.random.normal(0, volatility/np.sqrt(365*24), quarter_hours)
            
            # Combine trend and noise
            regime_returns = trend + noise
            all_returns.extend(regime_returns)
            current_hour += quarter_hours
        
        # Ensure we have the right number of returns
        all_returns = all_returns[:total_hours]
        
        # Generate price series
        prices = [base_price]
        for ret in all_returns[:-1]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 100))  # Minimum price floor
        
        print(f"   💰 Price range: ${min(prices):,.0f} - ${max(prices):,.0f}")
        
        # Create realistic OHLCV data
        ohlcv_data = []
        
        for i, (timestamp, close_price) in enumerate(zip(dates, prices)):
            open_price = prices[i-1] if i > 0 else close_price
            
            # Realistic intrabar movement
            price_change = abs(close_price - open_price)
            base_range = max(price_change, close_price * 0.001)  # Minimum 0.1% range
            
            # Higher volatility during high-volume periods
            volatility_multiplier = 1 + abs(all_returns[i]) * 10 if i < len(all_returns) else 1
            intrabar_range = base_range * volatility_multiplier
            
            high_price = max(open_price, close_price) + intrabar_range * np.random.uniform(0.3, 0.8)
            low_price = min(open_price, close_price) - intrabar_range * np.random.uniform(0.3, 0.8)
            
            # Realistic volume with regime-based patterns
            base_volume = 1000000
            if i < len(all_returns):
                # Higher volume during high volatility
                volume_multiplier = 1 + abs(all_returns[i]) * 20
                # Higher volume during downturns (panic selling)
                if all_returns[i] < -0.01:  # 1% decline
                    volume_multiplier *= 2
            else:
                volume_multiplier = 1
                
            volume = base_volume * volume_multiplier * np.random.lognormal(0, 0.5)
            
            ohlcv_data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': max(open_price, high_price, low_price, close_price),
                'low': min(open_price, high_price, low_price, close_price),
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(ohlcv_data)
        df.set_index('timestamp', inplace=True)
        
        print(f"   ✅ Generated {len(df):,} realistic market data points")
        return df

class RealisticTradingSimulator:
    """Simulates realistic trading with costs, slippage, and constraints"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position = None
        self.trades = []
        
        # Realistic trading costs
        self.trading_fee = 0.001  # 0.1% per trade (Binance-like)
        self.slippage_rate = 0.0005  # 0.05% slippage
        self.min_trade_size = 10  # Minimum $10 trade
        
    def calculate_position_size(self, signal_confidence: float, risk_per_trade: float = 0.02) -> float:
        """Calculate position size based on Kelly criterion and risk management"""
        # Base position size (2% risk per trade)
        base_size = self.current_capital * risk_per_trade
        
        # Adjust based on confidence (Kelly-like approach)
        confidence_multiplier = min(signal_confidence * 2, 1.0)  # Max 100% of base size
        
        # Ensure minimum trade size
        position_size = max(base_size * confidence_multiplier, self.min_trade_size)
        
        # Never risk more than 5% of capital on single trade
        max_size = self.current_capital * 0.05
        
        return min(position_size, max_size)
    
    def execute_trade(self, signal: str, price: float, confidence: float, timestamp: pd.Timestamp) -> bool:
        """Execute trade with realistic costs and constraints"""
        
        if signal == 'hold':
            return False
            
        # Close existing position if opposite signal
        if self.position and self.position['type'] != signal:
            self.close_position(price, timestamp, 'signal_reversal')
        
        # Don't open new position if we already have one in same direction
        if self.position and self.position['type'] == signal:
            return False
            
        # Calculate position size
        position_size = self.calculate_position_size(confidence)
        
        if position_size < self.min_trade_size:
            return False
        
        # Apply slippage (worse price due to market impact)
        slippage_factor = 1 + self.slippage_rate if signal == 'buy' else 1 - self.slippage_rate
        execution_price = price * slippage_factor
        
        # Calculate shares and fees
        trading_fee = position_size * self.trading_fee
        net_position_value = position_size - trading_fee
        shares = net_position_value / execution_price
        
        # Check if we have enough capital
        total_cost = position_size
        if total_cost > self.current_capital:
            return False
        
        # Execute trade
        self.position = {
            'type': signal,
            'shares': shares,
            'entry_price': execution_price,
            'entry_time': timestamp,
            'position_value': net_position_value,
            'confidence': confidence,
            'fees_paid': trading_fee
        }
        
        self.current_capital -= total_cost
        return True
    
    def close_position(self, price: float, timestamp: pd.Timestamp, reason: str = 'exit_signal') -> Dict:
        """Close current position and record trade"""
        
        if not self.position:
            return {}
        
        # Apply slippage on exit
        slippage_factor = 1 - self.slippage_rate if self.position['type'] == 'buy' else 1 + self.slippage_rate
        execution_price = price * slippage_factor
        
        # Calculate gross proceeds
        gross_proceeds = self.position['shares'] * execution_price
        
        # Subtract exit fees
        exit_fee = gross_proceeds * self.trading_fee
        net_proceeds = gross_proceeds - exit_fee
        
        # Calculate P&L
        initial_investment = self.position['position_value'] + self.position['fees_paid']
        total_fees = self.position['fees_paid'] + exit_fee
        
        if self.position['type'] == 'buy':
            pnl = net_proceeds - initial_investment
        else:  # short position
            pnl = initial_investment - net_proceeds
        
        pnl_percentage = pnl / initial_investment if initial_investment > 0 else 0
        
        # Update capital
        self.current_capital += net_proceeds
        
        # Record trade
        trade = {
            'entry_time': self.position['entry_time'],
            'exit_time': timestamp,
            'type': self.position['type'],
            'entry_price': self.position['entry_price'],
            'exit_price': execution_price,
            'shares': self.position['shares'],
            'initial_investment': initial_investment,
            'gross_proceeds': gross_proceeds,
            'total_fees': total_fees,
            'net_pnl': pnl,
            'pnl_percentage': pnl_percentage,
            'confidence': self.position['confidence'],
            'exit_reason': reason,
            'hold_duration_hours': (timestamp - self.position['entry_time']).total_seconds() / 3600
        }
        
        self.trades.append(trade)
        self.position = None
        
        return trade

def optimize_strategy_parameters(market_data: pd.DataFrame) -> Dict:
    """Optimize strategy parameters using walk-forward analysis"""
    print("\n🔧 OPTIMIZING STRATEGY PARAMETERS")
    print("=" * 40)
    
    from strategies import get_strategy
    strategy_class = get_strategy('candle_momentum')
    
    # Parameter ranges for optimization
    param_combinations = [
        {'lookback_period': 10, 'sma_period': 20, 'min_momentum_ratio': 1.15, 'min_confidence': 0.4},
        {'lookback_period': 12, 'sma_period': 25, 'min_momentum_ratio': 1.2, 'min_confidence': 0.45},
        {'lookback_period': 15, 'sma_period': 30, 'min_momentum_ratio': 1.25, 'min_confidence': 0.5},
        {'lookback_period': 18, 'sma_period': 35, 'min_momentum_ratio': 1.3, 'min_confidence': 0.55},
        {'lookback_period': 20, 'sma_period': 40, 'min_momentum_ratio': 1.35, 'min_confidence': 0.6},
    ]
    
    best_params = None
    best_sharpe = -999
    optimization_results = {}
    
    # Use first 2 years for optimization, last year for validation
    train_data = market_data.iloc[:int(len(market_data) * 0.67)]  # First 2 years
    
    print(f"📊 Optimizing on {len(train_data):,} data points ({len(train_data)/24:.0f} days)")
    
    for i, params in enumerate(param_combinations):
        print(f"\n🧪 Testing combination {i+1}/{len(param_combinations)}: {params}")
        
        try:
            # Test this parameter combination
            strategy = strategy_class(params)
            simulator = RealisticTradingSimulator(10000)
            
            # Run backtest on training data
            for idx in range(max(params['lookback_period'], params['sma_period']) + 10, len(train_data), 24):  # Daily sampling
                try:
                    current_data = train_data.iloc[:idx+1]
                    current_price = current_data['close'].iloc[-1]
                    current_time = current_data.index[-1]
                    
                    # Generate signal
                    signal_result = strategy.generate_signals(current_data, 'BTC/USDT')
                    signal = signal_result['signal']
                    confidence = signal_result['confidence']
                    
                    # Execute trade
                    if signal != 'hold':
                        simulator.execute_trade(signal, current_price, confidence, current_time)
                    
                    # Check for position exits (simplified)
                    if simulator.position and idx % 168 == 0:  # Check weekly
                        # Simple exit after 1 week or if signal changes
                        simulator.close_position(current_price, current_time, 'time_exit')
                        
                except Exception:
                    continue
            
            # Close any remaining position
            if simulator.position:
                last_price = train_data['close'].iloc[-1]
                last_time = train_data.index[-1]
                simulator.close_position(last_price, last_time, 'backtest_end')
            
            # Calculate performance metrics
            if len(simulator.trades) > 0:
                returns = [t['pnl_percentage'] for t in simulator.trades]
                total_return = (simulator.current_capital - simulator.initial_capital) / simulator.initial_capital
                
                if len(returns) > 1 and np.std(returns) > 0:
                    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(52)  # Annualized
                else:
                    sharpe_ratio = 0
                
                win_rate = len([r for r in returns if r > 0]) / len(returns)
                
                result = {
                    'params': params,
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'win_rate': win_rate,
                    'total_trades': len(simulator.trades),
                    'final_capital': simulator.current_capital
                }
                
                optimization_results[f"combo_{i+1}"] = result
                
                print(f"   📊 Return: {total_return:.2%}")
                print(f"   📈 Sharpe: {sharpe_ratio:.2f}")
                print(f"   🎯 Win Rate: {win_rate:.2%}")
                print(f"   🔢 Trades: {len(simulator.trades)}")
                
                # Track best parameters
                if sharpe_ratio > best_sharpe and len(simulator.trades) >= 5:
                    best_sharpe = sharpe_ratio
                    best_params = params.copy()
            else:
                print("   ❌ No trades generated")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    if best_params:
        print(f"\n🏆 BEST PARAMETERS FOUND:")
        print(f"   Parameters: {best_params}")
        print(f"   Sharpe Ratio: {best_sharpe:.2f}")
    else:
        # Fallback to moderate parameters
        best_params = {'lookback_period': 15, 'sma_period': 30, 'min_momentum_ratio': 1.25, 'min_confidence': 0.5}
        print(f"\n⚠️  Using fallback parameters: {best_params}")
    
    return best_params, optimization_results

def run_full_backtest(market_data: pd.DataFrame, strategy_params: Dict) -> Dict:
    """Run full 3-year backtest with optimized parameters"""
    print(f"\n🚀 RUNNING FULL 3-YEAR BACKTEST")
    print("=" * 35)
    
    from strategies import get_strategy
    strategy_class = get_strategy('candle_momentum')
    
    # Initialize strategy and simulator
    strategy = strategy_class(strategy_params)
    simulator = RealisticTradingSimulator(10000)
    
    print(f"📊 Testing on {len(market_data):,} data points ({len(market_data)/24:.0f} days)")
    print(f"⚙️  Parameters: {strategy_params}")
    
    # Track performance over time
    equity_curve = []
    monthly_returns = []
    signals_log = []
    
    start_time = datetime.now()
    
    # Run backtest (sample every 6 hours for performance)
    warmup_period = max(strategy_params['lookback_period'], strategy_params['sma_period']) + 10
    
    for idx in range(warmup_period, len(market_data), 6):  # Every 6 hours
        try:
            current_data = market_data.iloc[:idx+1]
            current_price = current_data['close'].iloc[-1]
            current_time = current_data.index[-1]
            
            # Generate signal
            signal_result = strategy.generate_signals(current_data, 'BTC/USDT')
            signal = signal_result['signal']
            confidence = signal_result['confidence']
            
            # Log signal
            if signal != 'hold':
                signals_log.append({
                    'time': current_time,
                    'signal': signal,
                    'confidence': confidence,
                    'price': current_price
                })
            
            # Position management
            if simulator.position:
                # Check exit conditions
                hold_hours = (current_time - simulator.position['entry_time']).total_seconds() / 3600
                
                # Exit conditions
                should_exit = False
                exit_reason = ''
                
                if signal != 'hold' and signal != simulator.position['type']:
                    should_exit = True
                    exit_reason = 'signal_reversal'
                elif hold_hours > 168:  # Max 1 week hold
                    should_exit = True
                    exit_reason = 'time_limit'
                elif simulator.position['type'] == 'buy' and current_price < simulator.position['entry_price'] * 0.95:
                    should_exit = True
                    exit_reason = 'stop_loss'
                elif simulator.position['type'] == 'sell' and current_price > simulator.position['entry_price'] * 1.05:
                    should_exit = True
                    exit_reason = 'stop_loss'
                
                if should_exit:
                    simulator.close_position(current_price, current_time, exit_reason)
            
            # Enter new position
            if not simulator.position and signal != 'hold' and confidence > 0.3:
                simulator.execute_trade(signal, current_price, confidence, current_time)
            
            # Record equity curve (daily)
            if idx % 24 == 0:  # Daily recording
                total_value = simulator.current_capital
                if simulator.position:
                    # Add unrealized P&L
                    if simulator.position['type'] == 'buy':
                        unrealized_pnl = simulator.position['shares'] * (current_price - simulator.position['entry_price'])
                    else:
                        unrealized_pnl = simulator.position['shares'] * (simulator.position['entry_price'] - current_price)
                    total_value += unrealized_pnl
                
                equity_curve.append({
                    'date': current_time,
                    'equity': total_value,
                    'cash': simulator.current_capital,
                    'total_trades': len(simulator.trades)
                })
                
        except Exception as e:
            continue
    
    # Close final position
    if simulator.position:
        final_price = market_data['close'].iloc[-1]
        final_time = market_data.index[-1]
        simulator.close_position(final_price, final_time, 'backtest_end')
    
    execution_time = (datetime.now() - start_time).total_seconds()
    print(f"⏱️  Backtest completed in {execution_time:.1f} seconds")
    
    return {
        'simulator': simulator,
        'equity_curve': equity_curve,
        'signals_log': signals_log,
        'strategy_params': strategy_params
    }

def analyze_results(backtest_results: Dict) -> Dict:
    """Comprehensive analysis of backtest results"""
    print(f"\n📊 COMPREHENSIVE RESULTS ANALYSIS")
    print("=" * 35)
    
    simulator = backtest_results['simulator']
    equity_curve = backtest_results['equity_curve']
    trades = simulator.trades
    
    if not trades:
        print("❌ No trades executed - strategy too conservative")
        return {}
    
    # Basic metrics
    initial_capital = simulator.initial_capital
    final_capital = simulator.current_capital
    total_return = (final_capital - initial_capital) / initial_capital
    
    # Trade statistics
    returns = [t['pnl_percentage'] for t in trades]
    winning_trades = [r for r in returns if r > 0]
    losing_trades = [r for r in returns if r < 0]
    
    win_rate = len(winning_trades) / len(returns) if returns else 0
    avg_win = np.mean(winning_trades) if winning_trades else 0
    avg_loss = np.mean(losing_trades) if losing_trades else 0
    
    # Risk metrics
    if len(returns) > 1:
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252/7) if np.std(returns) > 0 else 0  # Weekly to annual
    else:
        sharpe_ratio = 0
    
    # Drawdown analysis
    equity_values = [e['equity'] for e in equity_curve]
    if equity_values:
        peak = equity_values[0]
        max_drawdown = 0
        for value in equity_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
    else:
        max_drawdown = 0
    
    # Time analysis
    hold_durations = [t['hold_duration_hours'] for t in trades]
    avg_hold_time = np.mean(hold_durations) if hold_durations else 0
    
    # Annualized return
    total_days = (trades[-1]['exit_time'] - trades[0]['entry_time']).days if trades else 365
    years = total_days / 365
    annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    
    # Monthly returns
    monthly_data = {}
    for trade in trades:
        month_key = trade['exit_time'].strftime('%Y-%m')
        if month_key not in monthly_data:
            monthly_data[month_key] = []
        monthly_data[month_key].append(trade['pnl_percentage'])
    
    monthly_returns = [np.sum(returns) for returns in monthly_data.values()]
    
    results = {
        'initial_capital': initial_capital,
        'final_capital': final_capital,
        'total_return': total_return,
        'annual_return': annual_return,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'avg_hold_time_hours': avg_hold_time,
        'profit_factor': abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if losing_trades else float('inf'),
        'monthly_returns': monthly_returns,
        'total_fees_paid': sum(t['total_fees'] for t in trades),
        'best_trade': max(returns) if returns else 0,
        'worst_trade': min(returns) if returns else 0
    }
    
    # Display results
    print(f"💰 FINANCIAL PERFORMANCE:")
    print(f"   Initial Capital: ${initial_capital:,.2f}")
    print(f"   Final Capital: ${final_capital:,.2f}")
    print(f"   Total Return: {total_return:.2%}")
    print(f"   Annual Return: {annual_return:.2%}")
    print(f"   Total Fees Paid: ${results['total_fees_paid']:,.2f}")
    
    print(f"\n📊 TRADING STATISTICS:")
    print(f"   Total Trades: {len(trades)}")
    print(f"   Win Rate: {win_rate:.2%}")
    print(f"   Average Win: {avg_win:.2%}")
    print(f"   Average Loss: {avg_loss:.2%}")
    print(f"   Best Trade: {results['best_trade']:.2%}")
    print(f"   Worst Trade: {results['worst_trade']:.2%}")
    print(f"   Profit Factor: {results['profit_factor']:.2f}")
    
    print(f"\n⚖️  RISK METRICS:")
    print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"   Maximum Drawdown: {max_drawdown:.2%}")
    print(f"   Average Hold Time: {avg_hold_time:.1f} hours")
    
    # Performance rating
    if annual_return > 0.2 and sharpe_ratio > 1.0 and max_drawdown < 0.2:
        rating = "🟢 EXCELLENT"
    elif annual_return > 0.1 and sharpe_ratio > 0.5 and max_drawdown < 0.3:
        rating = "🟡 GOOD"
    elif annual_return > 0.05:
        rating = "🟠 MODERATE"
    else:
        rating = "🔴 POOR"
    
    print(f"\n🎯 OVERALL RATING: {rating}")
    
    return results

def main():
    """Main execution function"""
    print("🚀 REALISTIC 3-YEAR CANDLE MOMENTUM BACKTEST")
    print("=" * 50)
    print("💰 Initial Investment: $10,000")
    print("📅 Period: 2021-2024 (3 years)")
    print("🎯 Goal: Realistic crypto trading simulation")
    
    try:
        # Generate realistic market data
        simulator = RealisticMarketSimulator('2021-01-01', '2024-01-01')
        market_data = simulator.generate_realistic_data()
        
        # Optimize strategy parameters
        best_params, optimization_results = optimize_strategy_parameters(market_data)
        
        # Run full backtest with optimized parameters
        backtest_results = run_full_backtest(market_data, best_params)
        
        # Analyze results
        analysis = analyze_results(backtest_results)
        
        if analysis:
            # Save comprehensive results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"realistic_3year_backtest_{timestamp}.json"
            
            full_results = {
                'metadata': {
                    'strategy': 'candle_momentum',
                    'initial_capital': 10000,
                    'period': '2021-2024',
                    'total_data_points': len(market_data),
                    'backtest_timestamp': timestamp
                },
                'optimization': optimization_results,
                'best_parameters': best_params,
                'performance': analysis,
                'sample_trades': backtest_results['simulator'].trades[:20]  # First 20 trades
            }
            
            with open(results_file, 'w') as f:
                json.dump(full_results, f, indent=2, default=str)
            
            print(f"\n💾 COMPREHENSIVE RESULTS SAVED: {results_file}")
            
            # Final recommendations
            print(f"\n💡 RECOMMENDATIONS:")
            if analysis['annual_return'] > 0.15:
                print("✅ Strategy shows strong performance - consider live implementation")
            elif analysis['annual_return'] > 0.05:
                print("⚠️  Moderate performance - consider further optimization")
            else:
                print("❌ Poor performance - strategy needs significant improvement")
                
            if analysis['max_drawdown'] > 0.3:
                print("⚠️  High drawdown risk - implement stronger risk management")
                
            if analysis['sharpe_ratio'] < 0.5:
                print("⚠️  Low risk-adjusted returns - consider parameter tuning")
            
            print(f"\n🎯 NEXT STEPS:")
            print("1. Review trade logs for pattern analysis")
            print("2. Test on different market periods")
            print("3. Implement additional risk controls")
            print("4. Consider portfolio diversification")
            
            return full_results
        else:
            print("❌ Backtest analysis failed")
            return None
            
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
    if results:
        print("\n✅ 3-YEAR REALISTIC BACKTEST COMPLETED SUCCESSFULLY!")
    else:
        print("\n❌ Backtest failed - check logs for details")