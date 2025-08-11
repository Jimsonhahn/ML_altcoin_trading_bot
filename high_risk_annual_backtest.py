#!/usr/bin/env python3
"""
High-Risk Daily Strategy - 1 Year Backtest
==========================================

Comprehensive 365-day backtest of the high-risk strategy:
- Daily 30€ budget simulation
- Realistic market conditions
- Multiple market regimes
- Detailed performance analytics
- Risk-adjusted returns analysis
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
import asyncio
import random

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class HighRiskBacktestEngine:
    """
    Specialized backtesting engine for high-risk daily strategy
    
    Features:
    - Daily budget resets (30€ each day)
    - Realistic trading costs and slippage
    - Market regime simulation
    - Multi-signal generation
    - Position management with time limits
    """
    
    def __init__(self, initial_capital: float = 10950.0):  # 365 days * 30€
        self.total_budget = initial_capital
        self.daily_budget = 30.0
        self.current_day_budget = 30.0
        self.current_day_spent = 0.0
        
        # Performance tracking
        self.daily_results = []
        self.total_pnl = 0.0
        self.total_trades = 0
        self.winning_days = 0
        self.losing_days = 0
        self.max_daily_gain = 0.0
        self.max_daily_loss = 0.0
        
        # Position tracking
        self.active_positions = {}
        self.position_counter = 0
        
        # Trading costs
        self.commission_rate = 0.001  # 0.1%
        self.slippage_rate = 0.0005   # 0.05%
        
        print(f"🔥 High-Risk Backtest Engine initialized")
        print(f"💰 Total budget: {self.total_budget:,.2f}€ (365 days × 30€)")
        print(f"📅 Daily budget: {self.daily_budget}€")
    
    def reset_daily_budget(self, day: int):
        """Reset budget for new trading day"""
        self.current_day_budget = self.daily_budget
        self.current_day_spent = 0.0
        
        # Force close any remaining positions (shouldn't happen with 6h limit)
        if self.active_positions:
            logger.warning(f"Day {day}: Force closing {len(self.active_positions)} remaining positions")
            self.active_positions.clear()
    
    def can_trade(self, amount: float) -> bool:
        """Check if trade is possible within daily budget"""
        return (amount <= self.current_day_budget and 
                self.current_day_spent + amount <= self.daily_budget)
    
    def execute_trade(self, signal_data: Dict[str, Any], current_price: float) -> bool:
        """Execute trade with realistic costs"""
        
        symbol = signal_data['symbol']
        signal_type = signal_data['signal']
        confidence = signal_data['confidence']
        
        # Calculate position size based on confidence
        base_size = min(15.0, self.current_day_budget * 0.8)  # Max 15€ or 80% of remaining
        position_size = base_size * confidence
        position_size = max(2.0, min(position_size, self.current_day_budget))
        
        if not self.can_trade(position_size):
            return False
        
        # Apply slippage
        if signal_type == 'BUY':
            execution_price = current_price * (1 + self.slippage_rate)
        else:
            execution_price = current_price * (1 - self.slippage_rate)
        
        # Calculate quantity and costs
        quantity = position_size / execution_price
        commission = position_size * self.commission_rate
        total_cost = position_size + commission
        
        if total_cost > self.current_day_budget:
            return False
        
        # Create position
        position_id = f"pos_{self.position_counter}"
        self.position_counter += 1
        
        self.active_positions[position_id] = {
            'symbol': symbol,
            'side': signal_type,
            'entry_price': execution_price,
            'quantity': quantity,
            'position_size': position_size,
            'commission_paid': commission,
            'entry_time': signal_data['timestamp'],
            'confidence': confidence,
            'target_exit': signal_data['timestamp'] + timedelta(hours=6),
            'stop_loss': execution_price * (0.85 if signal_type == 'BUY' else 1.15),
            'profit_targets': [
                execution_price * (1.25 if signal_type == 'BUY' else 0.75),  # 25%
                execution_price * (1.50 if signal_type == 'BUY' else 0.50),  # 50%
                execution_price * (2.00 if signal_type == 'BUY' else 0.00),  # 100%
            ]
        }
        
        # Update budget
        self.current_day_spent += total_cost
        self.current_day_budget -= total_cost
        self.total_trades += 1
        
        return True
    
    def update_positions(self, current_time: datetime, current_prices: Dict[str, float]) -> float:
        """Update positions and close if needed"""
        
        day_pnl = 0.0
        positions_to_close = []
        
        for position_id, position in self.active_positions.items():
            symbol = position['symbol']
            current_price = current_prices.get(symbol)
            
            if current_price is None:
                continue
            
            # Calculate current P&L
            if position['side'] == 'BUY':
                unrealized_pnl = (current_price - position['entry_price']) * position['quantity']
            else:
                unrealized_pnl = (position['entry_price'] - current_price) * position['quantity']
            
            # Check exit conditions
            should_close = False
            close_reason = ''
            
            # Time limit (6 hours)
            if current_time >= position['target_exit']:
                should_close = True
                close_reason = 'time_limit'
            
            # Stop loss
            elif ((position['side'] == 'BUY' and current_price <= position['stop_loss']) or
                  (position['side'] == 'SELL' and current_price >= position['stop_loss'])):
                should_close = True
                close_reason = 'stop_loss'
            
            # Profit targets
            elif position['side'] == 'BUY':
                if current_price >= position['profit_targets'][2]:  # 100% target
                    should_close = True
                    close_reason = 'profit_target_3'
                elif current_price >= position['profit_targets'][1]:  # 50% target
                    should_close = True
                    close_reason = 'profit_target_2'
                elif current_price >= position['profit_targets'][0]:  # 25% target
                    should_close = True
                    close_reason = 'profit_target_1'
            
            if should_close:
                positions_to_close.append((position_id, position, current_price, close_reason))
        
        # Close positions
        for position_id, position, exit_price, reason in positions_to_close:
            pnl = self.close_position(position_id, position, exit_price, reason)
            day_pnl += pnl
        
        return day_pnl
    
    def close_position(self, position_id: str, position: Dict, exit_price: float, reason: str) -> float:
        """Close position and calculate P&L"""
        
        # Apply slippage on exit
        if position['side'] == 'BUY':
            execution_price = exit_price * (1 - self.slippage_rate)
        else:
            execution_price = exit_price * (1 + self.slippage_rate)
        
        # Calculate P&L
        if position['side'] == 'BUY':
            gross_pnl = (execution_price - position['entry_price']) * position['quantity']
        else:
            gross_pnl = (position['entry_price'] - execution_price) * position['quantity']
        
        # Subtract commissions
        exit_commission = position['quantity'] * execution_price * self.commission_rate
        net_pnl = gross_pnl - position['commission_paid'] - exit_commission
        
        # Remove position
        del self.active_positions[position_id]
        
        return net_pnl

def generate_annual_market_data(start_date: str = "2023-01-01") -> pd.DataFrame:
    """Generate realistic annual crypto market data"""
    
    print("📊 Generating annual market data...")
    
    # Create 365 days of hourly data
    dates = pd.date_range(start=start_date, periods=365*24, freq='1H')
    
    # Define market regimes throughout the year
    regimes = [
        {'name': 'Bear_Start', 'trend': -0.0001, 'volatility': 0.02, 'duration': 60},    # Jan-Feb
        {'name': 'Crash', 'trend': -0.0008, 'volatility': 0.04, 'duration': 30},        # Mar
        {'name': 'Bottom', 'trend': 0.0000, 'volatility': 0.03, 'duration': 45},        # Apr-May
        {'name': 'Recovery', 'trend': 0.0003, 'volatility': 0.025, 'duration': 60},     # Jun-Jul
        {'name': 'Bull_Run', 'trend': 0.0006, 'volatility': 0.02, 'duration': 90},      # Aug-Oct
        {'name': 'Euphoria', 'trend': 0.0010, 'volatility': 0.035, 'duration': 45},     # Nov
        {'name': 'Correction', 'trend': -0.0005, 'volatility': 0.03, 'duration': 35}    # Dec
    ]
    
    # Generate data for multiple symbols
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 'MATIC/USDT', 
               'ADA/USDT', 'DOT/USDT', 'ATOM/USDT', 'NEAR/USDT', 'FTM/USDT']
    
    base_prices = {
        'BTC/USDT': 16500, 'ETH/USDT': 1200, 'SOL/USDT': 10, 'AVAX/USDT': 12,
        'MATIC/USDT': 0.75, 'ADA/USDT': 0.25, 'DOT/USDT': 4.5, 'ATOM/USDT': 8,
        'NEAR/USDT': 1.2, 'FTM/USDT': 0.18
    }
    
    all_data = {}
    
    for symbol in symbols:
        print(f"   📈 Generating {symbol} data...")
        
        np.random.seed(42 + hash(symbol) % 1000)  # Different seed per symbol
        
        data = []
        current_price = base_prices[symbol]
        current_period = 0
        
        for regime in regimes:
            regime_periods = regime['duration'] * 24  # Convert days to hours
            
            for _ in range(regime_periods):
                if current_period >= len(dates):
                    break
                
                timestamp = dates[current_period]
                
                # Generate price movement
                trend = regime['trend']
                noise = np.random.normal(0, regime['volatility'])
                
                # Add some symbol-specific behavior
                symbol_multiplier = {
                    'BTC/USDT': 1.0, 'ETH/USDT': 1.1, 'SOL/USDT': 1.5,
                    'AVAX/USDT': 1.3, 'MATIC/USDT': 1.2, 'ADA/USDT': 0.8,
                    'DOT/USDT': 1.1, 'ATOM/USDT': 1.2, 'NEAR/USDT': 1.4,
                    'FTM/USDT': 1.6
                }.get(symbol, 1.0)
                
                total_change = (trend + noise) * symbol_multiplier
                
                # Calculate OHLC
                open_price = current_price
                close_price = open_price * (1 + total_change)
                close_price = max(close_price, base_prices[symbol] * 0.1)  # Floor at 10% of base
                
                # Generate realistic intrabar movement
                if close_price > open_price:
                    high_price = close_price * (1 + abs(np.random.normal(0, 0.005)))
                    low_price = open_price * (1 - abs(np.random.normal(0, 0.003)))
                else:
                    high_price = open_price * (1 + abs(np.random.normal(0, 0.003)))
                    low_price = close_price * (1 - abs(np.random.normal(0, 0.005)))
                
                # Volume with regime-specific patterns
                base_volume = 1000000
                volume_multiplier = 1 + abs(total_change) * 20
                regime_multiplier = {'Crash': 3.0, 'Euphoria': 2.5, 'Bull_Run': 1.8}.get(regime['name'], 1.0)
                volume = base_volume * volume_multiplier * regime_multiplier * np.random.lognormal(0, 0.3)
                
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
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        all_data[symbol] = df
    
    print(f"✅ Generated data for {len(symbols)} symbols")
    print(f"📅 Date range: {start_date} to {dates[-1].strftime('%Y-%m-%d')}")
    
    return all_data

def generate_daily_signals(market_data: Dict[str, pd.DataFrame], 
                          current_date: datetime) -> List[Dict[str, Any]]:
    """Generate realistic daily trading signals"""
    
    signals = []
    
    # Get symbols with recent activity
    active_symbols = list(market_data.keys())[:5]  # Focus on top 5 symbols
    
    for symbol in active_symbols:
        df = market_data[symbol]
        
        # Get data up to current date
        current_data = df[df.index <= current_date]
        if len(current_data) < 24:  # Need at least 24 hours of data
            continue
        
        current_price = current_data['close'].iloc[-1]
        recent_data = current_data.iloc[-24:]  # Last 24 hours
        
        # Volume spike detection (simplified)
        current_volume = recent_data['volume'].iloc[-1]
        avg_volume = recent_data['volume'].iloc[:-1].mean()
        volume_spike = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Price momentum
        price_change = (current_price - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
        
        # Volatility
        returns = recent_data['close'].pct_change().dropna()
        volatility = returns.std()
        
        # Generate signal based on conditions
        signal_strength = 0.0
        
        # Volume spike bonus
        if volume_spike > 3.0:
            signal_strength += 0.4
        elif volume_spike > 2.0:
            signal_strength += 0.2
        
        # Momentum bonus
        if abs(price_change) > 0.05:  # 5% price move
            signal_strength += 0.3
        
        # Volatility bonus (high vol = opportunity)
        if volatility > 0.03:
            signal_strength += 0.2
        
        # Random sentiment factor
        seed_val = (int(current_date.timestamp()) + abs(hash(symbol))) % (2**32 - 1)
        np.random.seed(seed_val)
        sentiment = np.random.uniform(-0.5, 0.5)
        signal_strength += abs(sentiment) * 0.3
        
        # Determine if signal is strong enough
        min_threshold = 0.6
        if signal_strength >= min_threshold:
            
            # Determine direction (simplified)
            if price_change > 0 and sentiment > 0:
                signal_type = 'BUY'
            elif price_change > 0 and volume_spike > 2.5:
                signal_type = 'BUY'
            else:
                signal_type = 'BUY'  # High-risk strategy focuses on long positions
            
            signals.append({
                'symbol': symbol,
                'signal': signal_type,
                'confidence': min(signal_strength, 1.0),
                'current_price': current_price,
                'volume_spike': volume_spike,
                'price_change': price_change,
                'sentiment': sentiment,
                'timestamp': current_date,
                'metadata': {
                    'volatility': volatility,
                    'avg_volume': avg_volume,
                    'current_volume': current_volume
                }
            })
    
    # Sort by confidence and return top signals
    signals.sort(key=lambda x: x['confidence'], reverse=True)
    return signals[:3]  # Max 3 signals per day

def run_annual_backtest():
    """Run comprehensive 1-year backtest"""
    
    print("🔥 HIGH-RISK STRATEGY - 1 YEAR BACKTEST")
    print("=" * 50)
    
    # Generate market data
    start_date = "2023-01-01"
    market_data = generate_annual_market_data(start_date)
    
    # Initialize backtest engine
    engine = HighRiskBacktestEngine()
    
    # Track daily performance
    daily_performance = []
    
    print(f"\n🚀 Running 365-day backtest...")
    print(f"📊 Daily budget: {engine.daily_budget}€")
    print(f"💰 Total available: {engine.total_budget:,.2f}€")
    
    # Progress tracking
    progress_days = [30, 90, 180, 270, 365]
    
    # Run day by day
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    
    for day in range(365):
        current_date = start_datetime + timedelta(days=day)
        
        # Progress reporting
        if (day + 1) in progress_days:
            print(f"   📅 Day {day + 1}/365 ({((day + 1)/365)*100:.0f}%)")
        
        # Reset daily budget
        engine.reset_daily_budget(day + 1)
        day_start_pnl = engine.total_pnl
        day_trades = 0
        
        # Generate signals for this day
        signals = generate_daily_signals(market_data, current_date)
        
        # Execute trades based on signals
        for signal in signals:
            if engine.can_trade(15.0):  # Check if we can make a trade
                success = engine.execute_trade(signal, signal['current_price'])
                if success:
                    day_trades += 1
        
        # Simulate intraday price movements and position management
        day_pnl = 0.0
        
        # Simulate 6-hour trading session with hourly updates
        for hour in range(6):
            current_time = current_date + timedelta(hours=9 + hour)  # 9 AM to 3 PM
            
            # Get current prices for all symbols
            current_prices = {}
            for symbol, df in market_data.items():
                # Find closest timestamp
                closest_data = df[df.index <= current_time]
                if not closest_data.empty:
                    current_prices[symbol] = closest_data['close'].iloc[-1]
            
            # Update positions
            hour_pnl = engine.update_positions(current_time, current_prices)
            day_pnl += hour_pnl
        
        # Close any remaining positions (end of day)
        if engine.active_positions:
            end_of_day = current_date + timedelta(hours=23)
            final_prices = {}
            for symbol, df in market_data.items():
                closest_data = df[df.index <= end_of_day]
                if not closest_data.empty:
                    final_prices[symbol] = closest_data['close'].iloc[-1]
            
            final_pnl = engine.update_positions(end_of_day, final_prices)
            day_pnl += final_pnl
        
        # Update totals
        engine.total_pnl += day_pnl
        
        # Record daily performance
        daily_result = {
            'day': day + 1,
            'date': current_date.strftime('%Y-%m-%d'),
            'trades': day_trades,
            'signals_generated': len(signals),
            'budget_used': engine.daily_budget - engine.current_day_budget,
            'day_pnl': day_pnl,
            'cumulative_pnl': engine.total_pnl,
            'active_positions_eod': len(engine.active_positions)
        }
        
        daily_performance.append(daily_result)
        
        # Track winning/losing days
        if day_pnl > 0:
            engine.winning_days += 1
            engine.max_daily_gain = max(engine.max_daily_gain, day_pnl)
        elif day_pnl < 0:
            engine.losing_days += 1
            engine.max_daily_loss = min(engine.max_daily_loss, day_pnl)
    
    # Calculate comprehensive metrics
    print(f"\n📊 CALCULATING ANNUAL PERFORMANCE...")
    
    df_performance = pd.DataFrame(daily_performance)
    
    # Basic metrics
    total_invested = engine.daily_budget * 365  # Total possible investment
    total_return = engine.total_pnl
    roi = (total_return / total_invested) * 100
    
    # Daily statistics
    daily_pnls = df_performance['day_pnl'].values
    positive_days = len([p for p in daily_pnls if p > 0])
    negative_days = len([p for p in daily_pnls if p < 0])
    breakeven_days = 365 - positive_days - negative_days
    
    win_rate = (positive_days / 365) * 100
    avg_daily_pnl = np.mean(daily_pnls)
    
    # Risk metrics
    daily_returns = daily_pnls / engine.daily_budget  # Returns as percentage of daily budget
    volatility = np.std(daily_returns) * np.sqrt(365)  # Annualized volatility
    
    if volatility > 0:
        sharpe_ratio = (avg_daily_pnl * 365) / (volatility * engine.daily_budget)
    else:
        sharpe_ratio = 0
    
    # Maximum drawdown
    cumulative_pnls = df_performance['cumulative_pnl'].values
    peak = 0
    max_drawdown = 0
    
    for pnl in cumulative_pnls:
        if pnl > peak:
            peak = pnl
        drawdown = peak - pnl
        max_drawdown = max(max_drawdown, drawdown)
    
    max_drawdown_pct = (max_drawdown / total_invested) * 100
    
    # Trading frequency
    total_trading_days = len([d for d in daily_performance if d['trades'] > 0])
    avg_trades_per_day = engine.total_trades / 365
    avg_trades_per_trading_day = engine.total_trades / max(total_trading_days, 1)
    
    # Monthly breakdown
    monthly_pnl = []
    for month in range(12):
        month_start = month * 30
        month_end = min((month + 1) * 30, 365)
        month_data = df_performance.iloc[month_start:month_end]
        monthly_pnl.append(month_data['day_pnl'].sum())
    
    # Display comprehensive results
    print(f"\n" + "="*70)
    print(f"📊 HIGH-RISK STRATEGY - ANNUAL BACKTEST RESULTS")
    print(f"="*70)
    
    # Performance Summary
    print(f"\n💰 PERFORMANCE SUMMARY")
    print(f"{'='*25}")
    print(f"Total Investment Capacity: {total_invested:>12,.2f}€")
    print(f"Total P&L:               {total_return:>12,.2f}€")
    print(f"Return on Investment:     {roi:>12.2f}%")
    print(f"Average Daily P&L:        {avg_daily_pnl:>12.2f}€")
    print(f"Best Day:                 {engine.max_daily_gain:>12.2f}€")
    print(f"Worst Day:                {engine.max_daily_loss:>12.2f}€")
    
    # Risk Analysis
    print(f"\n⚖️  RISK ANALYSIS")
    print(f"{'='*20}")
    print(f"Maximum Drawdown:         {max_drawdown:>12.2f}€ ({max_drawdown_pct:.1f}%)")
    print(f"Annualized Volatility:    {volatility:>12.2%}")
    print(f"Sharpe Ratio:             {sharpe_ratio:>12.2f}")
    print(f"Win Rate (Days):          {win_rate:>12.1f}%")
    
    # Trading Statistics
    print(f"\n📊 TRADING STATISTICS")
    print(f"{'='*23}")
    print(f"Total Trades:             {engine.total_trades:>12,}")
    print(f"Trading Days:             {total_trading_days:>12,} / 365")
    print(f"Avg Trades/Day:           {avg_trades_per_day:>12.1f}")
    print(f"Avg Trades/Trading Day:   {avg_trades_per_trading_day:>12.1f}")
    print(f"Positive Days:            {positive_days:>12,} ({(positive_days/365)*100:.1f}%)")
    print(f"Negative Days:            {negative_days:>12,} ({(negative_days/365)*100:.1f}%)")
    print(f"Breakeven Days:           {breakeven_days:>12,} ({(breakeven_days/365)*100:.1f}%)")
    
    # Monthly Performance
    print(f"\n📅 MONTHLY BREAKDOWN")
    print(f"{'='*22}")
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for i, (month, pnl) in enumerate(zip(months, monthly_pnl)):
        print(f"{month}: {pnl:>8.2f}€", end="  ")
        if (i + 1) % 4 == 0:
            print()  # New line every 4 months
    
    if len(months) % 4 != 0:
        print()
    
    # Performance Rating
    print(f"\n🎯 STRATEGY ASSESSMENT")
    print(f"{'='*25}")
    
    score = 0
    assessments = []
    
    # ROI Assessment
    if roi > 50:
        score += 3
        assessments.append("✅ Excellent returns")
    elif roi > 20:
        score += 2
        assessments.append("✅ Good returns")
    elif roi > 0:
        score += 1
        assessments.append("⚠️ Modest returns")
    else:
        assessments.append("❌ Negative returns")
    
    # Win Rate Assessment
    if win_rate > 60:
        score += 2
        assessments.append("✅ High win rate")
    elif win_rate > 50:
        score += 1
        assessments.append("✅ Decent win rate")
    else:
        assessments.append("⚠️ Low win rate")
    
    # Risk Assessment
    if max_drawdown_pct < 20:
        score += 2
        assessments.append("✅ Controlled risk")
    elif max_drawdown_pct < 40:
        score += 1
        assessments.append("⚠️ Moderate risk")
    else:
        assessments.append("❌ High risk")
    
    # Sharpe Ratio Assessment
    if sharpe_ratio > 1.5:
        score += 2
        assessments.append("✅ Excellent risk-adjusted returns")
    elif sharpe_ratio > 1.0:
        score += 1
        assessments.append("✅ Good risk-adjusted returns")
    elif sharpe_ratio > 0.5:
        assessments.append("⚠️ Fair risk-adjusted returns")
    else:
        assessments.append("❌ Poor risk-adjusted returns")
    
    # Overall Rating
    ratings = {
        9: "🟢 EXCEPTIONAL", 8: "🟢 EXCELLENT", 7: "🟡 VERY GOOD",
        6: "🟡 GOOD", 5: "🟠 MODERATE", 4: "🟠 FAIR",
        3: "🔴 POOR", 2: "🔴 WEAK", 1: "🔴 VERY POOR", 0: "🔴 TERRIBLE"
    }
    
    overall_rating = ratings.get(score, "🔴 TERRIBLE")
    
    print(f"Overall Rating: {overall_rating} (Score: {score}/9)")
    print()
    for assessment in assessments:
        print(f"  {assessment}")
    
    # Comparison with Conservative Approach
    print(f"\n📈 COMPARISON ANALYSIS")
    print(f"{'='*24}")
    conservative_return = total_invested * 0.05  # 5% annual return
    print(f"Conservative 5% Annual:   {conservative_return:>12.2f}€")
    print(f"High-Risk Strategy:       {total_return:>12.2f}€")
    print(f"Excess Return:            {total_return - conservative_return:>12.2f}€")
    print(f"Risk Premium:             {((total_return - conservative_return) / total_invested) * 100:>12.1f}%")
    
    # Recommendations
    print(f"\n💡 STRATEGY RECOMMENDATIONS")
    print(f"{'='*30}")
    
    if roi > 30 and win_rate > 55:
        print("🚀 Strategy shows strong potential for live trading")
        print("📊 Consider gradual scaling with proven performance")
    elif roi > 10 and max_drawdown_pct < 30:
        print("⚡ Strategy has merit but needs optimization")
        print("🔧 Consider adjusting risk parameters or signal filters")
    elif roi > 0:
        print("⚠️ Strategy is marginally profitable")
        print("🛠️ Significant improvements needed before live deployment")
    else:
        print("❌ Strategy shows consistent losses")
        print("🔄 Complete strategy revision recommended")
    
    # Risk Warnings
    print(f"\n⚠️  RISK WARNINGS")
    print(f"{'='*17}")
    print("🔥 This is an EXTREME high-risk strategy")
    print("💰 Maximum daily loss: 30€ (100% of daily budget)")
    print("📉 Potential annual loss: 10,950€ (365 × 30€)")
    print("🎰 Only use money you can afford to lose completely")
    print("📊 Past performance does not guarantee future results")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'backtest_info': {
            'timestamp': timestamp,
            'strategy': 'high_risk_daily',
            'period': '365_days',
            'start_date': start_date,
            'daily_budget': engine.daily_budget,
            'total_budget': total_invested
        },
        'performance_summary': {
            'total_return': total_return,
            'roi_percent': roi,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_percent': max_drawdown_pct,
            'volatility': volatility,
            'total_trades': engine.total_trades,
            'trading_days': total_trading_days
        },
        'daily_breakdown': daily_performance[:30],  # First 30 days
        'monthly_pnl': {month: pnl for month, pnl in zip(months, monthly_pnl)},
        'assessment': {
            'score': score,
            'rating': overall_rating,
            'assessments': assessments
        }
    }
    
    results_file = f"high_risk_annual_backtest_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved: {results_file}")
    
    return results

if __name__ == "__main__":
    print("🔥 Starting High-Risk Strategy Annual Backtest...")
    
    results = run_annual_backtest()
    
    if results:
        print(f"\n🎉 ANNUAL BACKTEST COMPLETED!")
        print(f"📊 {results['performance_summary']['total_trades']:,} trades analyzed")
        print(f"💰 Final P&L: {results['performance_summary']['total_return']:+,.2f}€")
        print(f"📈 ROI: {results['performance_summary']['roi_percent']:+.1f}%")
        print(f"🏆 Rating: {results['assessment']['rating']}")
        
        print(f"\n🔍 Key Insights:")
        print(f"• Win Rate: {results['performance_summary']['win_rate']:.1f}%")
        print(f"• Max Drawdown: {results['performance_summary']['max_drawdown_percent']:.1f}%")
        print(f"• Sharpe Ratio: {results['performance_summary']['sharpe_ratio']:.2f}")
        print(f"• Trading Days: {results['performance_summary']['trading_days']}/365")
        
    else:
        print(f"\n❌ Backtest failed")
    
    print(f"\n🔥 High-Risk Annual Analysis Complete!")