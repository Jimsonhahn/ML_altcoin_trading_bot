#!/usr/bin/env python3
"""
Comprehensive Candle Body Momentum Strategy Backtest & Evaluation
================================================================

Complete backtesting analysis with:
- Multiple timeframes and market conditions
- Detailed performance metrics
- Risk analysis
- Strategy optimization recommendations
- Comparison with buy-and-hold
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
logging.basicConfig(level=logging.ERROR)  # Minimal logging for cleaner output
logger = logging.getLogger(__name__)

class ComprehensiveBacktestEngine:
    """Advanced backtesting engine with detailed analytics"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = [initial_capital]
        self.timestamps = []
        
        # Trading costs
        self.commission = 0.001  # 0.1% per side
        self.slippage = 0.0005   # 0.05% slippage
        
        # Risk management
        self.max_risk_per_trade = 0.02  # 2% max risk
        self.max_portfolio_risk = 0.20  # 20% max drawdown trigger
        
    def calculate_position_size(self, confidence: float, current_price: float) -> float:
        """Calculate dynamic position size based on confidence and risk"""
        base_risk = self.capital * self.max_risk_per_trade
        confidence_multiplier = min(confidence * 1.5, 1.0)  # Scale with confidence
        return base_risk * confidence_multiplier / current_price
    
    def execute_trade(self, signal: str, price: float, timestamp: pd.Timestamp, 
                     confidence: float, metadata: Dict) -> bool:
        """Execute trade with advanced position sizing"""
        
        if signal == 'HOLD':
            return False
            
        # Close existing position if opposite signal
        if self.position and self.position['type'] != signal:
            self.close_position(price, timestamp, 'signal_reversal')
            
        # Don't open if already have position in same direction
        if self.position and self.position['type'] == signal:
            return False
            
        # Calculate position size
        shares = self.calculate_position_size(confidence, price)
        position_value = shares * price
        
        # Apply costs
        execution_price = price * (1 + self.slippage if signal == 'BUY' else 1 - self.slippage)
        commission_cost = position_value * self.commission
        total_cost = position_value + commission_cost
        
        # Check if we have enough capital
        if total_cost > self.capital:
            return False
            
        # Execute trade
        self.position = {
            'type': signal,
            'entry_price': execution_price,
            'entry_time': timestamp,
            'shares': shares,
            'position_value': position_value,
            'commission_paid': commission_cost,
            'confidence': confidence,
            'metadata': metadata
        }
        
        self.capital -= total_cost
        return True
    
    def close_position(self, price: float, timestamp: pd.Timestamp, reason: str) -> Dict:
        """Close position and record trade"""
        
        if not self.position:
            return {}
            
        # Apply slippage and commission
        execution_price = price * (1 - self.slippage if self.position['type'] == 'BUY' else 1 + self.slippage)
        gross_proceeds = self.position['shares'] * execution_price
        exit_commission = gross_proceeds * self.commission
        net_proceeds = gross_proceeds - exit_commission
        
        # Calculate P&L
        total_invested = self.position['position_value'] + self.position['commission_paid']
        total_commission = self.position['commission_paid'] + exit_commission
        
        if self.position['type'] == 'BUY':
            pnl = net_proceeds - total_invested
        else:  # SHORT
            pnl = total_invested - net_proceeds
            
        pnl_pct = pnl / total_invested if total_invested > 0 else 0
        
        # Update capital
        final_capital = self.capital + net_proceeds
        
        # Record trade
        trade = {
            'entry_time': self.position['entry_time'],
            'exit_time': timestamp,
            'type': self.position['type'],
            'entry_price': self.position['entry_price'],
            'exit_price': execution_price,
            'shares': self.position['shares'],
            'invested': total_invested,
            'proceeds': net_proceeds,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'total_commission': total_commission,
            'confidence': self.position['confidence'],
            'exit_reason': reason,
            'hold_duration_hours': (timestamp - self.position['entry_time']).total_seconds() / 3600,
            'capital_before': self.capital,
            'capital_after': final_capital
        }
        
        self.trades.append(trade)
        self.capital = final_capital
        self.position = None
        
        return trade
    
    def update_equity_curve(self, current_price: float, timestamp: pd.Timestamp):
        """Update equity curve including unrealized P&L"""
        total_equity = self.capital
        
        if self.position:
            # Add unrealized P&L
            current_value = self.position['shares'] * current_price
            if self.position['type'] == 'BUY':
                unrealized_pnl = current_value - self.position['position_value']
            else:
                unrealized_pnl = self.position['position_value'] - current_value
            total_equity += unrealized_pnl
            
        self.equity_curve.append(total_equity)
        self.timestamps.append(timestamp)

def create_comprehensive_market_data(start_date: str, end_date: str, 
                                   timeframe: str = '30T') -> pd.DataFrame:
    """Create comprehensive market data with various market regimes"""
    print(f"📊 Generating comprehensive market data ({timeframe} timeframe)...")
    
    dates = pd.date_range(start=start_date, end=end_date, freq=timeframe)
    total_periods = len(dates)
    
    print(f"   📅 Period: {start_date} to {end_date}")
    print(f"   🕐 Total periods: {total_periods:,}")
    
    np.random.seed(42)  # Reproducible results
    base_price = 35000
    
    # Define market regimes
    regime_configs = [
        {'name': 'Bear Market', 'trend': -0.0003, 'volatility': 0.018, 'duration': 0.15},
        {'name': 'Crash', 'trend': -0.0015, 'volatility': 0.035, 'duration': 0.10},
        {'name': 'Bottom Formation', 'trend': 0.0001, 'volatility': 0.025, 'duration': 0.10},
        {'name': 'Recovery', 'trend': 0.0005, 'volatility': 0.020, 'duration': 0.15},
        {'name': 'Bull Market', 'trend': 0.0008, 'volatility': 0.015, 'duration': 0.25},
        {'name': 'Euphoria', 'trend': 0.0012, 'volatility': 0.028, 'duration': 0.10},
        {'name': 'Correction', 'trend': -0.0008, 'volatility': 0.022, 'duration': 0.15}
    ]
    
    data = []
    current_price = base_price
    current_period = 0
    
    for regime in regime_configs:
        regime_periods = int(total_periods * regime['duration'])
        print(f"   📈 {regime['name']}: {regime_periods:,} periods")
        
        for i in range(regime_periods):
            if current_period >= total_periods:
                break
                
            timestamp = dates[current_period]
            
            # Generate price movement
            trend_component = regime['trend']
            noise_component = np.random.normal(0, regime['volatility'])
            total_change = trend_component + noise_component
            
            # Calculate OHLC
            open_price = current_price
            close_price = open_price * (1 + total_change)
            close_price = max(close_price, 1000)  # Price floor
            
            # Generate realistic intrabar movement
            if close_price > open_price:  # Bullish candle
                high_price = close_price * (1 + abs(np.random.normal(0, 0.003)))
                low_price = open_price * (1 - abs(np.random.normal(0, 0.002)))
            else:  # Bearish candle
                high_price = open_price * (1 + abs(np.random.normal(0, 0.002)))
                low_price = close_price * (1 - abs(np.random.normal(0, 0.003)))
            
            # Volume with regime-specific patterns
            base_volume = 2000000
            volatility_multiplier = 1 + abs(total_change) * 30
            regime_multiplier = {'Crash': 3.0, 'Euphoria': 2.5}.get(regime['name'], 1.0)
            volume = base_volume * volatility_multiplier * regime_multiplier * np.random.lognormal(0, 0.4)
            
            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
                'regime': regime['name']
            })
            
            current_price = close_price
            current_period += 1
    
    # Fill remaining periods if any
    while current_period < total_periods:
        timestamp = dates[current_period]
        change = np.random.normal(0, 0.01)
        open_price = current_price
        close_price = open_price * (1 + change)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': max(open_price, close_price) * 1.002,
            'low': min(open_price, close_price) * 0.998,
            'close': close_price,
            'volume': 2000000 * np.random.lognormal(0, 0.3),
            'regime': 'Stabilization'
        })
        
        current_price = close_price
        current_period += 1
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    print(f"   💰 Price range: ${df['close'].min():,.0f} - ${df['close'].max():,.0f}")
    print(f"   ✅ Market data generated: {len(df):,} periods")
    
    return df

def calculate_comprehensive_metrics(engine: ComprehensiveBacktestEngine, 
                                  market_data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive performance metrics"""
    
    if not engine.trades:
        return {'error': 'No trades to analyze'}
    
    # Basic metrics
    initial_capital = engine.initial_capital
    final_capital = engine.capital
    total_return = (final_capital - initial_capital) / initial_capital
    
    # Time analysis
    start_date = market_data.index[0]
    end_date = market_data.index[-1]
    total_days = (end_date - start_date).days
    years = total_days / 365.25
    
    # Returns analysis
    returns = [t['pnl_pct'] for t in engine.trades]
    winning_returns = [r for r in returns if r > 0]
    losing_returns = [r for r in returns if r < 0]
    
    # Win/Loss statistics
    total_trades = len(returns)
    winning_trades = len(winning_returns)
    losing_trades = len(losing_returns)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    avg_win = np.mean(winning_returns) if winning_returns else 0
    avg_loss = np.mean(losing_returns) if losing_returns else 0
    largest_win = max(returns) if returns else 0
    largest_loss = min(returns) if returns else 0
    
    # Risk metrics
    if len(returns) > 1:
        returns_std = np.std(returns)
        sharpe_ratio = (np.mean(returns) / returns_std) * np.sqrt(252/7) if returns_std > 0 else 0
    else:
        sharpe_ratio = 0
        returns_std = 0
    
    # Drawdown analysis
    equity_curve = engine.equity_curve
    peak = initial_capital
    max_drawdown_pct = 0
    max_drawdown_duration = 0
    current_drawdown_start = None
    
    for i, equity in enumerate(equity_curve):
        if equity > peak:
            peak = equity
            if current_drawdown_start is not None:
                drawdown_duration = i - current_drawdown_start
                max_drawdown_duration = max(max_drawdown_duration, drawdown_duration)
                current_drawdown_start = None
        else:
            if current_drawdown_start is None:
                current_drawdown_start = i
            drawdown_pct = (peak - equity) / peak
            max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
    
    # Trading frequency
    avg_hold_time = np.mean([t['hold_duration_hours'] for t in engine.trades])
    trades_per_month = total_trades / (years * 12) if years > 0 else 0
    
    # Profit factor
    gross_profit = sum([r for r in returns if r > 0]) * initial_capital
    gross_loss = abs(sum([r for r in returns if r < 0])) * initial_capital
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Annualized metrics
    annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    
    # Commission analysis
    total_commissions = sum([t['total_commission'] for t in engine.trades])
    commission_impact = total_commissions / initial_capital
    
    # Buy and hold comparison
    buy_hold_return = (market_data['close'].iloc[-1] - market_data['close'].iloc[0]) / market_data['close'].iloc[0]
    buy_hold_annual = (1 + buy_hold_return) ** (1/years) - 1 if years > 0 else 0
    
    return {
        # Performance
        'total_return': total_return,
        'annual_return': annual_return,
        'buy_hold_return': buy_hold_return,
        'buy_hold_annual': buy_hold_annual,
        'excess_return': annual_return - buy_hold_annual,
        
        # Risk
        'max_drawdown': max_drawdown_pct,
        'max_drawdown_duration_periods': max_drawdown_duration,
        'sharpe_ratio': sharpe_ratio,
        'volatility': returns_std,
        'downside_deviation': np.std([r for r in returns if r < 0]) if losing_returns else 0,
        
        # Trading
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'largest_win': largest_win,
        'largest_loss': largest_loss,
        'avg_hold_time_hours': avg_hold_time,
        'trades_per_month': trades_per_month,
        
        # Costs
        'total_commissions': total_commissions,
        'commission_impact': commission_impact,
        
        # Capital
        'initial_capital': initial_capital,
        'final_capital': final_capital,
        'peak_capital': max(equity_curve),
        
        # Time
        'backtest_years': years,
        'start_date': start_date,
        'end_date': end_date
    }

def run_regime_analysis(engine: ComprehensiveBacktestEngine, 
                       market_data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze performance by market regime"""
    
    if 'regime' not in market_data.columns:
        return {}
    
    regime_performance = {}
    
    for trade in engine.trades:
        # Find regime during trade entry
        entry_time = trade['entry_time']
        try:
            regime = market_data.loc[entry_time, 'regime']
        except:
            # Find closest timestamp
            closest_idx = market_data.index.get_indexer([entry_time], method='nearest')[0]
            regime = market_data.iloc[closest_idx]['regime']
        
        if regime not in regime_performance:
            regime_performance[regime] = {
                'trades': [],
                'returns': [],
                'wins': 0,
                'losses': 0
            }
        
        regime_performance[regime]['trades'].append(trade)
        regime_performance[regime]['returns'].append(trade['pnl_pct'])
        
        if trade['pnl_pct'] > 0:
            regime_performance[regime]['wins'] += 1
        else:
            regime_performance[regime]['losses'] += 1
    
    # Calculate metrics per regime
    regime_stats = {}
    for regime, data in regime_performance.items():
        if data['trades']:
            returns = data['returns']
            regime_stats[regime] = {
                'total_trades': len(data['trades']),
                'win_rate': data['wins'] / len(data['trades']),
                'avg_return': np.mean(returns),
                'total_return': np.sum(returns),
                'best_trade': max(returns),
                'worst_trade': min(returns),
                'volatility': np.std(returns) if len(returns) > 1 else 0
            }
    
    return regime_stats

def run_comprehensive_backtest():
    """Run comprehensive backtest with detailed analysis"""
    print("🔬 COMPREHENSIVE CANDLE BODY MOMENTUM BACKTEST")
    print("=" * 50)
    
    try:
        # Import strategy
        from strategies import get_strategy
        strategy_class = get_strategy('candle_body_momentum')
        
        if not strategy_class:
            raise ValueError("Strategy not found!")
        
        # Strategy configuration
        config = {
            'lookback_period': 10,
            'sma_period': 200,
            'timeframe': '30m',
            'debug_logging': False  # Reduce noise
        }
        
        print(f"📋 Strategy Configuration:")
        for key, value in config.items():
            print(f"   {key}: {value}")
        
        # Initialize strategy
        strategy = strategy_class(config)
        
        # Generate comprehensive market data (2 years)
        market_data = create_comprehensive_market_data('2022-01-01', '2024-01-01', '30T')
        
        # Initialize backtest engine
        engine = ComprehensiveBacktestEngine(10000)
        
        print(f"\n🚀 Running comprehensive backtest...")
        print(f"   📊 Data points: {len(market_data):,}")
        print(f"   🕐 Timeframe: 30 minutes")
        print(f"   📅 Duration: 2 years")
        
        # Warmup period
        warmup = max(config['lookback_period'], config['sma_period']) + 10
        signals_generated = 0
        trades_executed = 0
        
        # Progress tracking
        total_periods = len(market_data)
        progress_points = [int(total_periods * p) for p in [0.25, 0.5, 0.75, 1.0]]
        
        # Run backtest
        for i in range(warmup, len(market_data)):
            try:
                # Progress reporting
                if i in progress_points:
                    progress = (i / total_periods) * 100
                    print(f"   ⏳ Progress: {progress:.0f}% ({i:,}/{total_periods:,})")
                
                # Get point-in-time data
                current_data = market_data.iloc[:i+1]
                current_price = current_data['close'].iloc[-1]
                current_time = current_data.index[-1]
                
                # Generate signal
                signal, signal_data = strategy.calculate_signal('BTC/USDT', current_data, current_price)
                
                if signal != 'HOLD':
                    signals_generated += 1
                
                # Position management
                if engine.position:
                    # Check exit conditions
                    metadata = signal_data.get('metadata', {})
                    should_exit = False
                    exit_reason = ''
                    
                    # Exit on opposite crossover (video specification)
                    if (engine.position['type'] == 'BUY' and 
                        metadata.get('bearish_crossover', False)):
                        should_exit = True
                        exit_reason = 'bearish_crossover'
                    elif (engine.position['type'] == 'SELL' and 
                          metadata.get('bullish_crossover', False)):
                        should_exit = True
                        exit_reason = 'bullish_crossover'
                    
                    # Risk management: Max hold time (1 week)
                    elif engine.position and hasattr(engine.position, 'entry_time'):
                        hold_hours = (current_time - engine.position['entry_time']).total_seconds() / 3600
                        if hold_hours > 168:  # 1 week
                            should_exit = True
                            exit_reason = 'max_hold_time'
                    
                    if should_exit:
                        trade = engine.close_position(current_price, current_time, exit_reason)
                        if trade:
                            trades_executed += 1
                
                # Enter new position
                if signal != 'HOLD':
                    success = engine.execute_trade(
                        signal, current_price, current_time,
                        signal_data.get('confidence', 0.5),
                        signal_data.get('metadata', {})
                    )
                    
                    if success and trades_executed <= 5:  # Show first 5 trades
                        print(f"      📡 Trade {trades_executed + 1}: {signal} at ${current_price:,.2f}")
                
                # Update equity curve
                if i % 24 == 0:  # Daily updates
                    engine.update_equity_curve(current_price, current_time)
                
            except Exception as e:
                continue  # Skip problematic periods
        
        # Close final position
        if engine.position:
            final_price = market_data['close'].iloc[-1]
            final_time = market_data.index[-1]
            engine.close_position(final_price, final_time, 'backtest_end')
            trades_executed += 1
        
        print(f"   ✅ Backtest completed!")
        print(f"   📡 Signals generated: {signals_generated:,}")
        print(f"   💼 Trades executed: {trades_executed:,}")
        
        # Calculate comprehensive metrics
        print(f"\n📊 CALCULATING PERFORMANCE METRICS...")
        metrics = calculate_comprehensive_metrics(engine, market_data)
        
        if 'error' in metrics:
            print(f"❌ {metrics['error']}")
            return None
        
        # Regime analysis
        print(f"📈 ANALYZING PERFORMANCE BY MARKET REGIME...")
        regime_stats = run_regime_analysis(engine, market_data)
        
        # Display results
        print(f"\n" + "="*60)
        print(f"📊 COMPREHENSIVE BACKTEST RESULTS")
        print(f"="*60)
        
        # Performance Summary
        print(f"\n💰 PERFORMANCE SUMMARY")
        print(f"{'='*25}")
        print(f"Initial Capital:      ${metrics['initial_capital']:>12,.2f}")
        print(f"Final Capital:        ${metrics['final_capital']:>12,.2f}")
        print(f"Peak Capital:         ${metrics['peak_capital']:>12,.2f}")
        print(f"Total Return:         {metrics['total_return']:>12.2%}")
        print(f"Annualized Return:    {metrics['annual_return']:>12.2%}")
        print(f"Buy & Hold Return:    {metrics['buy_hold_annual']:>12.2%}")
        print(f"Excess Return:        {metrics['excess_return']:>12.2%}")
        
        # Risk Analysis
        print(f"\n⚖️  RISK ANALYSIS")
        print(f"{'='*20}")
        print(f"Maximum Drawdown:     {metrics['max_drawdown']:>12.2%}")
        print(f"Sharpe Ratio:         {metrics['sharpe_ratio']:>12.2f}")
        print(f"Volatility:           {metrics['volatility']:>12.2%}")
        print(f"Downside Deviation:   {metrics['downside_deviation']:>12.2%}")
        
        # Trading Statistics
        print(f"\n📊 TRADING STATISTICS")
        print(f"{'='*23}")
        print(f"Total Trades:         {metrics['total_trades']:>12,}")
        print(f"Win Rate:             {metrics['win_rate']:>12.2%}")
        print(f"Profit Factor:        {metrics['profit_factor']:>12.2f}")
        print(f"Average Win:          {metrics['avg_win']:>12.2%}")
        print(f"Average Loss:         {metrics['avg_loss']:>12.2%}")
        print(f"Largest Win:          {metrics['largest_win']:>12.2%}")
        print(f"Largest Loss:         {metrics['largest_loss']:>12.2%}")
        print(f"Avg Hold Time:        {metrics['avg_hold_time_hours']:>12.1f} hours")
        print(f"Trades per Month:     {metrics['trades_per_month']:>12.1f}")
        
        # Cost Analysis
        print(f"\n💵 COST ANALYSIS")
        print(f"{'='*17}")
        print(f"Total Commissions:    ${metrics['total_commissions']:>12,.2f}")
        print(f"Commission Impact:    {metrics['commission_impact']:>12.2%}")
        
        # Performance Rating
        score = 0
        if metrics['annual_return'] > 0.15: score += 3
        elif metrics['annual_return'] > 0.08: score += 2
        elif metrics['annual_return'] > 0.03: score += 1
        
        if metrics['sharpe_ratio'] > 1.5: score += 3
        elif metrics['sharpe_ratio'] > 1.0: score += 2
        elif metrics['sharpe_ratio'] > 0.5: score += 1
        
        if metrics['max_drawdown'] < 0.15: score += 2
        elif metrics['max_drawdown'] < 0.25: score += 1
        
        if metrics['win_rate'] > 0.6: score += 2
        elif metrics['win_rate'] > 0.45: score += 1
        
        ratings = {
            9: "🟢 EXCELLENT", 8: "🟢 EXCELLENT", 7: "🟡 VERY GOOD",
            6: "🟡 GOOD", 5: "🟠 MODERATE", 4: "🟠 MODERATE",
            3: "🔴 POOR", 2: "🔴 POOR", 1: "🔴 VERY POOR", 0: "🔴 VERY POOR"
        }
        
        rating = ratings.get(score, "🔴 VERY POOR")
        
        print(f"\n🎯 OVERALL RATING: {rating} (Score: {score}/10)")
        
        # Regime Performance
        if regime_stats:
            print(f"\n📈 PERFORMANCE BY MARKET REGIME")
            print(f"{'='*35}")
            print(f"{'Regime':<18} {'Trades':<8} {'Win%':<8} {'Avg Ret':<10} {'Best':<10} {'Worst':<10}")
            print(f"{'-'*70}")
            
            for regime, stats in regime_stats.items():
                print(f"{regime:<18} {stats['total_trades']:<8} "
                      f"{stats['win_rate']:<8.1%} {stats['avg_return']:<10.2%} "
                      f"{stats['best_trade']:<10.2%} {stats['worst_trade']:<10.2%}")
        
        # Strategy Evaluation
        print(f"\n🔍 STRATEGY EVALUATION")
        print(f"{'='*25}")
        
        if metrics['annual_return'] > metrics['buy_hold_annual']:
            print("✅ Strategy outperforms buy-and-hold")
        else:
            print("❌ Strategy underperforms buy-and-hold")
            
        if metrics['sharpe_ratio'] > 1.0:
            print("✅ Good risk-adjusted returns")
        else:
            print("⚠️  Poor risk-adjusted returns")
            
        if metrics['max_drawdown'] < 0.25:
            print("✅ Acceptable drawdown levels")
        else:
            print("❌ High drawdown risk")
            
        if metrics['win_rate'] > 0.45:
            print("✅ Decent win rate")
        else:
            print("⚠️  Low win rate - needs improvement")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        print(f"{'='*20}")
        
        if metrics['annual_return'] < 0.05:
            print("🔧 Consider parameter optimization or additional filters")
            
        if metrics['win_rate'] < 0.4:
            print("🎯 Add signal confirmation filters to improve win rate")
            
        if metrics['max_drawdown'] > 0.3:
            print("⚠️  Implement stronger risk management controls")
            
        if metrics['trades_per_month'] > 20:
            print("📉 Consider reducing trading frequency to lower costs")
            
        if metrics['excess_return'] < 0:
            print("📊 Strategy may not justify active trading - consider passive approach")
        
        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            'backtest_info': {
                'timestamp': timestamp,
                'strategy': 'candle_body_momentum',
                'version': 'comprehensive_analysis',
                'duration_years': metrics['backtest_years'],
                'total_periods': len(market_data)
            },
            'configuration': config,
            'performance_metrics': metrics,
            'regime_analysis': regime_stats,
            'rating': {'score': score, 'rating': rating},
            'sample_trades': engine.trades[:20]  # First 20 trades
        }
        
        results_file = f"comprehensive_backtest_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Comprehensive results saved: {results_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Comprehensive backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🔬 Starting Comprehensive Candle Body Momentum Analysis...")
    
    results = run_comprehensive_backtest()
    
    if results:
        print(f"\n🎉 COMPREHENSIVE ANALYSIS COMPLETED!")
        print(f"📊 All metrics calculated and saved")
        print(f"🎯 Strategy evaluation complete")
        print(f"\n📋 Next Steps:")
        print(f"1. Review regime-specific performance")
        print(f"2. Consider parameter optimization if needed")
        print(f"3. Test on out-of-sample data")
        print(f"4. Deploy for paper trading if results are satisfactory")
    else:
        print(f"\n❌ Analysis failed - check implementation")