#!/usr/bin/env python3
"""
Test Profitable BTC Strategy - Ziel: 30% Return + 2.0+ Sharpe
============================================================

Testet die optimierte Strategy mit realistischem Backtesting
"""

import sys
sys.path.append('.')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import Dict, Any, List

# Import the new strategy
from strategies.profitable_btc_strategy import ProfitableBTCStrategy, create_enhanced_indicator_engine


class EnhancedBacktester:
    """Enhanced backtester with better trade management"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0.0
        self.position_entry_price = 0.0
        self.position_entry_time = None
        self.position_direction = None
        self.trades = []
        self.equity_curve = []
        self.commission_rate = 0.001
        self.slippage_rate = 0.0005
        self.bars_in_position = 0
        
    def process_signal(self, timestamp: datetime, price: float, signal_data: Dict[str, Any], 
                      strategy: ProfitableBTCStrategy) -> Dict[str, Any]:
        """Process trading signal with enhanced logic"""
        try:
            direction = signal_data.get('direction', 'hold')
            signal_strength = signal_data.get('signal_strength', 0)
            confidence = signal_data.get('confidence', 0)
            
            # Check for exit first if in position
            if self.position != 0:
                self.bars_in_position += 1
                should_exit, exit_reason = strategy.should_exit_position(
                    self.position_entry_price, price, self.position_direction, self.bars_in_position
                )
                
                if should_exit:
                    return self._close_position(timestamp, price, exit_reason)
            
            # Check for new entry
            if direction != 'hold' and confidence >= strategy.min_signal_strength and self.position == 0:
                position_size = strategy.calculate_position_size(signal_data, price, self.capital)
                
                if position_size > 0.01:  # Minimum 1% position
                    return self._open_position(timestamp, price, direction, position_size, signal_data)
            
            # Update equity curve
            self._update_equity(timestamp, price, signal_data)
            
            return {"action": "hold", "reason": "no_signal"}
            
        except Exception as e:
            return {"action": "error", "error": str(e)}
    
    def _open_position(self, timestamp: datetime, price: float, direction: str, 
                      position_size: float, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Open new position"""
        try:
            # Calculate position value
            position_value = self.capital * position_size
            
            # Apply slippage and commission
            execution_price = price * (1 + self.slippage_rate) if direction == 'buy' else price * (1 - self.slippage_rate)
            commission = position_value * self.commission_rate
            
            # Update position
            if direction == 'buy':
                self.position = position_value / execution_price
                self.position_direction = 'long'
            else:
                self.position = -(position_value / execution_price)
                self.position_direction = 'short'
            
            self.position_entry_price = execution_price
            self.position_entry_time = timestamp
            self.bars_in_position = 0
            
            # Deduct costs
            self.capital -= commission
            
            # Record trade opening
            trade_record = {
                'entry_time': timestamp,
                'entry_price': execution_price,
                'direction': self.position_direction,
                'size': abs(self.position),
                'position_value': position_value,
                'signal_strength': signal_data.get('signal_strength', 0),
                'confidence': signal_data.get('confidence', 0),
                'confluence_count': signal_data.get('confluence_count', 0),
                'commission_entry': commission
            }
            
            return {
                "action": "position_opened",
                "direction": direction,
                "size": position_value,
                "price": execution_price,
                "commission": commission,
                "trade_record": trade_record
            }
            
        except Exception as e:
            return {"action": "error", "error": str(e)}
    
    def _close_position(self, timestamp: datetime, price: float, reason: str) -> Dict[str, Any]:
        """Close current position"""
        try:
            if self.position == 0:
                return {"action": "no_position"}
            
            # Calculate proceeds
            if self.position_direction == 'long':
                gross_proceeds = abs(self.position) * price
                exit_price = price * (1 - self.slippage_rate)
            else:  # short
                gross_proceeds = abs(self.position) * (2 * self.position_entry_price - price)
                exit_price = price * (1 + self.slippage_rate)
            
            exit_commission = gross_proceeds * self.commission_rate
            net_proceeds = gross_proceeds - exit_commission
            
            # Calculate PnL
            original_investment = abs(self.position) * self.position_entry_price
            pnl = net_proceeds - original_investment
            return_pct = pnl / original_investment if original_investment > 0 else 0
            
            # Update capital
            self.capital += net_proceeds
            
            # Record completed trade
            trade_record = {
                'entry_time': self.position_entry_time,
                'exit_time': timestamp,
                'entry_price': self.position_entry_price,
                'exit_price': exit_price,
                'direction': self.position_direction,
                'size': abs(self.position),
                'pnl': pnl,
                'return_pct': return_pct,
                'duration_hours': (timestamp - self.position_entry_time).total_seconds() / 3600,
                'exit_reason': reason,
                'commission_total': exit_commission + getattr(self, 'last_entry_commission', 0)
            }
            
            self.trades.append(trade_record)
            
            # Reset position
            self.position = 0.0
            self.position_entry_price = 0.0
            self.position_entry_time = None
            self.position_direction = None
            self.bars_in_position = 0
            
            return {
                "action": "position_closed",
                "reason": reason,
                "pnl": pnl,
                "return_pct": return_pct,
                "trade_record": trade_record
            }
            
        except Exception as e:
            return {"action": "error", "error": str(e)}
    
    def _update_equity(self, timestamp: datetime, price: float, signal_data: Dict[str, Any]):
        """Update equity curve"""
        try:
            # Calculate unrealized PnL
            unrealized_pnl = 0.0
            if self.position != 0:
                if self.position_direction == 'long':
                    current_value = abs(self.position) * price
                    unrealized_pnl = current_value - (abs(self.position) * self.position_entry_price)
                else:  # short
                    current_value = abs(self.position) * (2 * self.position_entry_price - price)
                    unrealized_pnl = current_value - (abs(self.position) * self.position_entry_price)
            
            total_equity = self.capital + unrealized_pnl
            
            self.equity_curve.append({
                'timestamp': timestamp,
                'price': price,
                'capital': self.capital,
                'unrealized_pnl': unrealized_pnl,
                'total_equity': total_equity,
                'position_size': abs(self.position) if self.position != 0 else 0,
                'signal_strength': signal_data.get('signal_strength', 0)
            })
            
        except Exception as e:
            print(f"Equity update error: {e}")
    
    def finalize(self, final_timestamp: datetime, final_price: float):
        """Finalize backtest"""
        if self.position != 0:
            self._close_position(final_timestamp, final_price, "backtest_end")
        self._update_equity(final_timestamp, final_price, {})
    
    def get_metrics(self) -> Dict[str, float]:
        """Calculate enhanced performance metrics"""
        try:
            if not self.equity_curve or not self.trades:
                return {}
            
            # Basic metrics
            final_equity = self.equity_curve[-1]['total_equity']
            total_return = (final_equity / self.initial_capital) - 1
            
            # Time period
            start_time = self.equity_curve[0]['timestamp']
            end_time = self.equity_curve[-1]['timestamp']
            days = (end_time - start_time).days
            years = days / 365.25 if days > 0 else 1
            annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
            
            # Returns for Sharpe calculation
            equity_values = [point['total_equity'] for point in self.equity_curve]
            returns = [(equity_values[i] / equity_values[i-1]) - 1 for i in range(1, len(equity_values))]
            
            # Sharpe ratio (assuming 3% risk-free rate)
            if returns:
                excess_returns = [r - (0.03/252) for r in returns]  # Daily risk-free rate
                sharpe_ratio = (np.mean(excess_returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Maximum drawdown
            peak = self.initial_capital
            max_drawdown = 0
            for point in self.equity_curve:
                equity = point['total_equity']
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            # Trade statistics
            winning_trades = [t for t in self.trades if t['pnl'] > 0]
            losing_trades = [t for t in self.trades if t['pnl'] <= 0]
            
            win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            
            # Profit factor
            total_wins = sum(t['pnl'] for t in winning_trades)
            total_losses = abs(sum(t['pnl'] for t in losing_trades))
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Risk-adjusted metrics
            volatility = np.std(returns) * np.sqrt(252) if returns else 0
            calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
            
            # Sortino ratio (downside deviation)
            negative_returns = [r for r in returns if r < 0]
            downside_deviation = np.std(negative_returns) * np.sqrt(252) if negative_returns else 0.001
            sortino_ratio = annual_return / downside_deviation if downside_deviation > 0 else 0
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': len(self.trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'largest_win': max([t['pnl'] for t in self.trades]) if self.trades else 0,
                'largest_loss': min([t['pnl'] for t in self.trades]) if self.trades else 0,
                'avg_trade_duration': np.mean([t['duration_hours'] for t in self.trades]) if self.trades else 0,
                'commission_total': sum([t.get('commission_total', 0) for t in self.trades])
            }
            
        except Exception as e:
            print(f"Metrics calculation error: {e}")
            return {}


def generate_enhanced_market_data(days: int = 365, start_price: float = 45000) -> pd.DataFrame:
    """Generate enhanced realistic market data"""
    print(f"📊 Generiere {days} Tage enhanced Marktdaten...")
    
    np.random.seed(789)  # Different seed for new test
    
    timestamps = []
    prices = []
    volumes = []
    
    current_time = datetime(2023, 1, 1)
    current_price = start_price
    
    # Enhanced market parameters
    base_volatility = 0.035   # 3.5% base daily volatility
    trend_strength = 0.0008   # Slight upward bias
    mean_reversion = 0.015    # Mean reversion
    volatility_clustering = 0.7  # Volatility persistence
    
    current_volatility = base_volatility
    
    for i in range(days * 24):  # Hourly data
        # Volatility clustering (realistic market behavior)
        vol_shock = np.random.normal(0, 0.001)
        current_volatility = (volatility_clustering * current_volatility + 
                            (1 - volatility_clustering) * base_volatility + vol_shock)
        current_volatility = max(0.01, min(current_volatility, 0.08))  # Bounds
        
        # Price movement with multiple components
        random_shock = np.random.normal(0, current_volatility / np.sqrt(24))
        trend_component = trend_strength / 24
        mean_reversion_component = -mean_reversion * (current_price - start_price) / start_price / 24
        
        # Add some realistic market cycles
        cycle_component = 0.0005 * np.sin(i * 2 * np.pi / (24 * 30))  # Monthly cycle
        
        price_change = (trend_component + mean_reversion_component + 
                       random_shock + cycle_component)
        
        current_price *= (1 + price_change)
        current_price = max(current_price, start_price * 0.2)  # Floor
        
        # Volume with price correlation
        base_volume = 2000
        volatility_volume = abs(price_change) * 100000
        trend_volume = max(0, price_change) * 50000  # Higher volume on up moves
        volume = base_volume + volatility_volume + trend_volume + np.random.exponential(800)
        
        timestamps.append(current_time)
        prices.append(current_price)
        volumes.append(volume)
        
        current_time += timedelta(hours=1)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'close': prices,
        'volume': volumes
    })
    
    df.set_index('timestamp', inplace=True)
    
    print(f"✅ Enhanced data generated: {len(df)} points")
    print(f"   Start: ${df['close'].iloc[0]:,.0f}")
    print(f"   End: ${df['close'].iloc[-1]:,.0f}")
    buyhold_return = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1
    print(f"   Buy&Hold Return: {buyhold_return:.2%}")
    print(f"   Avg Volatility: {df['close'].pct_change().std()*np.sqrt(252*24):.1%}")
    
    return df


def test_profitable_strategy():
    """Test the profitable strategy"""
    print("🎯 PROFITABLE BTC STRATEGY TEST")
    print("=" * 80)
    print("Ziel: 30% Annual Return + 2.0+ Sharpe Ratio\n")
    
    try:
        # Initialize components
        strategy = ProfitableBTCStrategy()
        indicator_engine = create_enhanced_indicator_engine()
        strategy.set_indicator_engine(indicator_engine)
        backtester = EnhancedBacktester(initial_capital=100000)
        
        print("✅ Enhanced strategy und backtester initialisiert")
        print(f"   Target Win Rate: 60%+")
        print(f"   Risk/Reward Ratio: {strategy.take_profit_pct/strategy.stop_loss_pct:.1f}:1")
        print(f"   Max Position Size: {strategy.max_position_size:.0%}")
        
        # Generate market data
        market_data = generate_enhanced_market_data(days=365)
        
        # Run backtest
        print(f"\n🚀 Running Enhanced Backtest...")
        
        signals_generated = 0
        trades_executed = 0
        high_quality_signals = 0
        
        for i, (timestamp, row) in enumerate(market_data.iterrows()):
            try:
                price = row['close']
                volume = row['volume']
                
                # Update indicators
                indicators = indicator_engine.update(price, volume, timestamp)
                
                # Generate signal after warmup
                if i >= 250:  # More warmup for enhanced indicators
                    signal_strength, signal_data = strategy.calculate_signal_strength(indicators, price)
                    
                    if signal_data.get('direction') != 'hold':
                        signals_generated += 1
                        
                        if signal_data.get('confidence', 0) >= strategy.min_signal_strength:
                            high_quality_signals += 1
                    
                    # Process signal
                    result = backtester.process_signal(timestamp, price, signal_data, strategy)
                    
                    if result.get('action') in ['position_opened', 'position_closed']:
                        trades_executed += 1
                
                # Progress update
                if (i + 1) % 2000 == 0:
                    progress = (i + 1) / len(market_data) * 100
                    current_equity = backtester.equity_curve[-1]['total_equity'] if backtester.equity_curve else backtester.initial_capital
                    print(f"   Progress: {progress:.1f}% - Equity: ${current_equity:,.0f}, Trades: {trades_executed}")
                    
            except Exception as e:
                print(f"Error at step {i}: {e}")
                continue
        
        # Finalize backtest
        final_timestamp = market_data.index[-1]
        final_price = market_data['close'].iloc[-1]
        backtester.finalize(final_timestamp, final_price)
        
        # Update adaptive parameters based on results
        strategy.update_adaptive_parameters(backtester.trades)
        
        # Calculate metrics
        metrics = backtester.get_metrics()
        
        print(f"\n📈 PROFITABLE STRATEGY RESULTS")
        print("=" * 80)
        
        print(f"🎯 PERFORMANCE METRICS:")
        print(f"   Annual Return: {metrics.get('annual_return', 0):.2%}")
        print(f"   Total Return: {metrics.get('total_return', 0):.2%}")
        print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"   Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
        print(f"   Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
        print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        print(f"   Volatility: {metrics.get('volatility', 0):.2%}")
        
        print(f"\n📊 TRADING QUALITY:")
        print(f"   Total Trades: {metrics.get('total_trades', 0)}")
        print(f"   Win Rate: {metrics.get('win_rate', 0):.1%}")
        print(f"   Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        print(f"   Avg Win: ${metrics.get('avg_win', 0):+,.2f}")
        print(f"   Avg Loss: ${metrics.get('avg_loss', 0):+,.2f}")
        print(f"   Avg Trade Duration: {metrics.get('avg_trade_duration', 0):.1f} hours")
        
        print(f"\n🎲 SIGNAL QUALITY:")
        print(f"   Signals Generated: {signals_generated}")
        print(f"   High Quality Signals: {high_quality_signals}")
        print(f"   Trades Executed: {trades_executed}")
        print(f"   Signal Selectivity: {high_quality_signals/signals_generated*100:.1f}%" if signals_generated > 0 else "   No signals")
        print(f"   Execution Rate: {trades_executed/high_quality_signals*100:.1f}%" if high_quality_signals > 0 else "   No high quality signals")
        
        # Benchmark comparison
        buyhold_return = (market_data['close'].iloc[-1] / market_data['close'].iloc[0]) - 1
        alpha = metrics.get('annual_return', 0) - buyhold_return
        
        print(f"\n🔄 BENCHMARK COMPARISON:")
        print(f"   Buy & Hold Return: {buyhold_return:.2%}")
        print(f"   Strategy Alpha: {alpha:.2%}")
        print(f"   Strategy Outperformed: {'✅' if metrics.get('annual_return', 0) > buyhold_return else '❌'}")
        
        # Goal achievement
        print(f"\n🎯 GOAL ACHIEVEMENT:")
        annual_return = metrics.get('annual_return', 0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        
        print(f"   Target: 30% Return + 2.0 Sharpe")
        print(f"   Achieved: {annual_return:.1%} Return + {sharpe_ratio:.2f} Sharpe")
        print(f"   Return Goal: {'✅' if annual_return >= 0.30 else '❌'} ({annual_return:.1%} vs 30%)")
        print(f"   Sharpe Goal: {'✅' if sharpe_ratio >= 2.0 else '❌'} ({sharpe_ratio:.2f} vs 2.0)")
        
        if annual_return >= 0.30 and sharpe_ratio >= 2.0:
            print(f"\n🎉 GOALS ACHIEVED! Strategy is ready for deployment!")
        elif annual_return >= 0.20 and sharpe_ratio >= 1.5:
            print(f"\n👍 GOOD PERFORMANCE! Close to targets, minor optimization needed")
        elif annual_return >= 0.10 and sharpe_ratio >= 1.0:
            print(f"\n📈 MODERATE PERFORMANCE! Profitable but needs improvement")
        else:
            print(f"\n⚠️ NEEDS OPTIMIZATION! Strategy requires further development")
        
        # Export results
        results = {
            'strategy_test': {
                'timestamp': datetime.now().isoformat(),
                'test_type': 'profitable_strategy_backtest',
                'market_days': 365,
                'target_return': 0.30,
                'target_sharpe': 2.0
            },
            'performance_metrics': metrics,
            'trading_activity': {
                'signals_generated': signals_generated,
                'high_quality_signals': high_quality_signals,
                'trades_executed': trades_executed,
                'signal_selectivity': high_quality_signals/signals_generated if signals_generated > 0 else 0
            },
            'goal_achievement': {
                'return_achieved': annual_return >= 0.30,
                'sharpe_achieved': sharpe_ratio >= 2.0,
                'both_achieved': annual_return >= 0.30 and sharpe_ratio >= 2.0
            },
            'adaptive_parameters': strategy.adaptive_params,
            'strategy_config': {
                'min_signal_strength': strategy.min_signal_strength,
                'max_position_size': strategy.max_position_size,
                'stop_loss_pct': strategy.stop_loss_pct,
                'take_profit_pct': strategy.take_profit_pct,
                'risk_reward_ratio': strategy.take_profit_pct / strategy.stop_loss_pct
            }
        }
        
        filename = f"profitable_strategy_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results exported: {filename}")
        
        return results
        
    except Exception as e:
        print(f"❌ Strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = test_profitable_strategy()
    
    if results and results['goal_achievement']['both_achieved']:
        print(f"\n🚀 SUCCESS! Ready for live deployment!")
    else:
        print(f"\n🔧 Strategy needs further optimization...")