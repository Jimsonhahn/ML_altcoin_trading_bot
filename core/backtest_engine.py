# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced Backtest Engine - Realistic Market Simulation
=====================================================

Enhanced Features:
- Realistic market conditions simulation
- Advanced slippage and market impact modeling  
- Market regime integration (Bull, Bear, Sideways, Volatile, Extreme Fear)
- Latency simulation
- Liquidity constraints
- Detailed cost analysis
- Market phase performance tracking
"""

import logging
import os
import json
import random
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum

# Try to import market regime detection
try:
    from ml_components.market_regime import MarketRegimeDetector
    MARKET_REGIME_AVAILABLE = True
except ImportError:
    MARKET_REGIME_AVAILABLE = False

logger = logging.getLogger(__name__)


class MarketPhase(Enum):
    """Market phases for realistic simulation"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    EXTREME_FEAR = "extreme_fear"


class LiquidityLevel(Enum):
    """Liquidity levels affecting execution"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class BacktestEngine:
    """Advanced backtesting engine with realistic market simulation"""

    def __init__(self, settings):
        """Initialize Enhanced Backtest Engine"""
        self.settings = settings
        self.backtest_config = settings.get('backtesting', {})

        # Basic parameters
        self.initial_capital = self.backtest_config.get('initial_capital', 10000)
        
        # Enhanced fee structure
        self.maker_fee = self.backtest_config.get('maker_fee', 0.001)  # 0.1%
        self.taker_fee = self.backtest_config.get('taker_fee', 0.001)  # 0.1%
        
        # Advanced slippage modeling
        self.base_slippage = self.backtest_config.get('base_slippage', 0.0005)  # 0.05%
        self.volatility_slippage_factor = self.backtest_config.get('volatility_slippage_factor', 2.0)
        self.size_slippage_factor = self.backtest_config.get('size_slippage_factor', 0.1)
        
        # Market impact modeling
        self.enable_market_impact = self.backtest_config.get('enable_market_impact', True)
        self.impact_factor = self.backtest_config.get('impact_factor', 0.001)
        
        # Latency simulation
        self.enable_latency = self.backtest_config.get('enable_latency', True)
        self.min_latency_ms = self.backtest_config.get('min_latency_ms', 50)
        self.max_latency_ms = self.backtest_config.get('max_latency_ms', 150)
        
        # Liquidity constraints
        self.enable_liquidity_constraints = self.backtest_config.get('enable_liquidity_constraints', True)
        self.max_volume_participation = self.backtest_config.get('max_volume_participation', 0.01)  # 1%
        
        # Market regime detection
        self.market_regime_detector = None
        if MARKET_REGIME_AVAILABLE:
            try:
                self.market_regime_detector = MarketRegimeDetector(settings)
                logger.info("Market regime detection enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize market regime detector: {e}")
        
        # Enhanced data storage
        self.market_data = {}
        self.trades = []
        self.orders = []
        self.positions = {}
        self.equity_curve = []
        self.market_phases = []  # Track market phases over time
        self.cost_breakdown = []  # Detailed cost analysis
        
        # Performance tracking by market phase
        self.phase_performance = {phase.value: {'trades': [], 'equity': []} for phase in MarketPhase}
        
        # Enhanced state tracking
        self.current_balance = self.initial_capital
        self.peak_balance = self.initial_capital
        self.current_market_phase = MarketPhase.SIDEWAYS
        self.current_liquidity = LiquidityLevel.MEDIUM
        
        # Realistic order execution tracking
        self.pending_orders = []  # Orders waiting for latency
        self.rejected_orders = []  # Orders that couldn't be filled
        
        # Statistics
        self.total_commission_paid = 0
        self.total_slippage_cost = 0
        self.total_market_impact = 0
        self.orders_rejected = 0
        
        logger.info(f"Enhanced Backtest Engine initialized with ${self.initial_capital} capital")
        logger.info(f"Realistic features: Market Impact={self.enable_market_impact}, "
                   f"Latency={self.enable_latency}, Liquidity Constraints={self.enable_liquidity_constraints}")

    def run_backtest(self, strategy, symbol: str, start_date: datetime,
                     end_date: datetime, timeframe: str = '1h') -> Dict[str, Any]:
        """Run backtest for a strategy"""
        logger.info(f"Starting backtest for {strategy.__class__.__name__} on {symbol}")
        logger.info(f"Period: {start_date} to {end_date}")

        # Reset state
        self._reset_state()

        # Load historical data
        data = self._load_historical_data(symbol, start_date, end_date, timeframe)
        if data.empty:
            logger.error("No historical data available for backtesting")
            return self._generate_empty_results()

        # Initialize strategy
        strategy.reset()

        # Main backtest loop
        for i in range(len(data)):
            current_time = data.index[i]
            current_candle = data.iloc[i]

            # Get data window for strategy
            lookback = min(i + 1, 100)  # Max 100 candles lookback
            data_window = data.iloc[max(0, i - lookback + 1):i + 1]

            # Update market data
            self._update_market_data(symbol, current_candle, current_time)

            # Check for stop loss / take profit hits
            self._check_exit_conditions(symbol, current_candle)

            # Generate strategy signal
            try:
                signal, signal_data = strategy.generate_signal(data_window, symbol)
            except Exception as e:
                logger.error(f"Strategy signal generation failed: {e}")
                signal = 'HOLD'
                signal_data = {}

            # Execute trade based on signal
            if signal in ['BUY', 'SELL']:
                self._execute_trade(
                    symbol=symbol,
                    side=signal.lower(),
                    price=current_candle['close'],
                    size=self._calculate_position_size(symbol, current_candle['close']),
                    timestamp=current_time,
                    signal_data=signal_data
                )

            # Update equity curve
            self._update_equity(current_time)

            # Log progress periodically
            if i % 100 == 0:
                logger.debug(f"Backtest progress: {i}/{len(data)} candles processed")

        # Close any remaining positions
        self._close_all_positions(data.iloc[-1]['close'], data.index[-1])

        # Generate results
        results = self._generate_results(strategy.__class__.__name__, symbol,
                                         start_date, end_date)

        # Save results
        self._save_results(results)

        logger.info(f"Backtest completed. Net profit: ${results['net_profit']:.2f}")

        return results

    def run_multi_symbol_backtest(self, strategy, symbols: List[str],
                                  start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Run backtest across multiple symbols"""
        logger.info(f"Starting multi-symbol backtest for {len(symbols)} symbols")

        all_results = {}
        combined_trades = []
        combined_equity = []

        for symbol in symbols:
            logger.info(f"Backtesting {symbol}...")

            # Run individual backtest
            results = self.run_backtest(strategy, symbol, start_date, end_date)
            all_results[symbol] = results

            # Collect trades
            combined_trades.extend(results.get('trades', []))

        # Calculate combined metrics
        combined_results = self._calculate_portfolio_metrics(all_results, combined_trades)
        combined_results['individual_results'] = all_results

        return combined_results

    def run_walk_forward_analysis(self, strategy, symbol: str,
                                  total_period_start: datetime,
                                  total_period_end: datetime,
                                  in_sample_ratio: float = 0.7,
                                  step_size_days: int = 30) -> Dict[str, Any]:
        """Run walk-forward analysis"""
        logger.info("Starting walk-forward analysis")

        results = []
        current_start = total_period_start

        while current_start < total_period_end:
            # Calculate period lengths
            total_days = (total_period_end - current_start).days
            in_sample_days = int(total_days * in_sample_ratio)
            out_sample_days = total_days - in_sample_days

            if out_sample_days < step_size_days:
                break

            # Define periods
            in_sample_end = current_start + timedelta(days=in_sample_days)
            out_sample_start = in_sample_end
            out_sample_end = out_sample_start + timedelta(days=step_size_days)

            if out_sample_end > total_period_end:
                out_sample_end = total_period_end

            logger.info(f"Walk-forward iteration: IS {current_start} to {in_sample_end}, "
                        f"OOS {out_sample_start} to {out_sample_end}")

            # Run in-sample optimization (simplified - just run backtest)
            in_sample_results = self.run_backtest(
                strategy, symbol, current_start, in_sample_end
            )

            # Run out-of-sample test
            out_sample_results = self.run_backtest(
                strategy, symbol, out_sample_start, out_sample_end
            )

            results.append({
                'in_sample_period': (current_start, in_sample_end),
                'out_sample_period': (out_sample_start, out_sample_end),
                'in_sample_results': in_sample_results,
                'out_sample_results': out_sample_results,
                'degradation': self._calculate_degradation(
                    in_sample_results, out_sample_results
                )
            })

            # Move forward
            current_start += timedelta(days=step_size_days)

        # Analyze walk-forward results
        wf_analysis = self._analyze_walk_forward_results(results)

        return wf_analysis

    def optimize_parameters(self, strategy_class, symbol: str,
                            param_grid: Dict[str, List[Any]],
                            start_date: datetime, end_date: datetime,
                            metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """Optimize strategy parameters"""
        logger.info(f"Starting parameter optimization for {strategy_class.__name__}")

        best_params = None
        best_metric = float('-inf')
        all_results = []

        # Generate all parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)

        for params in param_combinations:
            logger.info(f"Testing parameters: {params}")

            # Create strategy with parameters
            test_settings = self.settings.copy()
            for key, value in params.items():
                test_settings.set(f'{strategy_class.__name__.lower()}.{key}', value)

            strategy = strategy_class(test_settings)

            # Run backtest
            results = self.run_backtest(strategy, symbol, start_date, end_date)

            # Store results
            all_results.append({
                'params': params,
                'results': results,
                'metric_value': results.get(metric, float('-inf'))
            })

            # Check if best
            if results.get(metric, float('-inf')) > best_metric:
                best_metric = results[metric]
                best_params = params

        # Sort results by metric
        all_results.sort(key=lambda x: x['metric_value'], reverse=True)

        return {
            'best_params': best_params,
            'best_metric_value': best_metric,
            'optimization_metric': metric,
            'all_results': all_results[:10],  # Top 10 results
            'total_combinations_tested': len(param_combinations)
        }

    def calculate_metrics(self, trades: List[Dict[str, Any]],
                          initial_capital: float) -> Dict[str, Any]:
        """Calculate comprehensive backtest metrics"""
        if not trades:
            return self._empty_metrics()

        # Convert to DataFrame for easier calculation
        df = pd.DataFrame(trades)

        # Basic metrics
        total_trades = len(trades)
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # P&L metrics
        gross_profit = df[df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(df[df['pnl'] <= 0]['pnl'].sum())
        net_profit = df['pnl'].sum()

        # Average metrics
        avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0

        # Risk metrics
        returns = df['pnl'] / initial_capital
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        max_drawdown = self._calculate_max_drawdown_from_trades(trades, initial_capital)

        # Additional metrics
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        expectancy = net_profit / total_trades if total_trades > 0 else 0

        return {
            # Trade statistics
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,

            # P&L metrics
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'net_profit': net_profit,
            'total_return': (net_profit / initial_capital) * 100,

            # Average metrics
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': net_profit / total_trades if total_trades > 0 else 0,

            # Risk metrics
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,

            # Trade analysis
            'largest_win': df['pnl'].max() if not df.empty else 0,
            'largest_loss': df['pnl'].min() if not df.empty else 0,
            'avg_trade_duration': df['duration'].mean() if 'duration' in df else 0,
            'max_consecutive_wins': self._calculate_max_consecutive(df, True),
            'max_consecutive_losses': self._calculate_max_consecutive(df, False)
        }

    # Private helper methods

    def _reset_state(self):
        """Reset engine state for new backtest"""
        self.trades = []
        self.orders = []
        self.positions = {}
        self.equity_curve = []
        self.current_balance = self.initial_capital
        self.peak_balance = self.initial_capital
        self.market_data = {}

    def _load_historical_data(self, symbol: str, start_date: datetime,
                              end_date: datetime, timeframe: str) -> pd.DataFrame:
        """Load historical market data"""
        # Try to load from file
        filename = f"{symbol.replace('/', '_')}_{timeframe}.csv"
        filepath = os.path.join('data/market_data', filename)

        if os.path.exists(filepath):
            logger.info(f"Loading historical data from {filepath}")

            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

            # Filter by date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]

            logger.info(f"Loaded {len(df)} candles for backtesting")
            return df
        else:
            logger.warning(f"Historical data file not found: {filepath}")
            # Return empty DataFrame with proper structure
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    def _update_market_data(self, symbol: str, candle: pd.Series, timestamp: datetime):
        """Update current market data"""
        self.market_data[symbol] = {
            'timestamp': timestamp,
            'open': candle['open'],
            'high': candle['high'],
            'low': candle['low'],
            'close': candle['close'],
            'volume': candle['volume']
        }

    def _calculate_position_size(self, symbol: str, price: float) -> float:
        """Calculate position size based on risk management"""
        # Simple fixed percentage of capital
        risk_per_trade = self.settings.get('risk_management.risk_per_trade', 0.02)
        position_value = self.current_balance * risk_per_trade

        # Adjust for leverage if configured
        leverage = self.settings.get('trading.leverage', 1)
        position_value *= leverage

        # Convert to base currency amount
        position_size = position_value / price

        return position_size

    def _execute_trade(self, symbol: str, side: str, price: float,
                       size: float, timestamp: datetime, signal_data: Dict[str, Any]):
        """Execute a trade with realistic market simulation"""
        try:
            # 1. Check liquidity constraints
            if not self._check_liquidity_constraints(symbol, size):
                self._record_rejected_order(symbol, side, size, price, timestamp, "Liquidity constraints")
                return
            
            # 2. Simulate latency
            if self.enable_latency:
                latency_ms = random.uniform(self.min_latency_ms, self.max_latency_ms)
                # In real-time, price might move during latency
                price = self._simulate_price_movement_during_latency(symbol, price, latency_ms)
            
            # 3. Calculate realistic slippage
            total_slippage = self._calculate_dynamic_slippage(symbol, size, price)
            
            # 4. Calculate market impact
            market_impact = self._calculate_market_impact(symbol, size) if self.enable_market_impact else 0
            
            # 5. Apply slippage and market impact
            if side == 'buy':
                execution_price = price * (1 + total_slippage + market_impact)
            else:
                execution_price = price * (1 - total_slippage - market_impact)
            
            # 6. Determine order type and calculate appropriate fee
            is_market_order = signal_data.get('order_type', 'market') == 'market'
            commission_rate = self.taker_fee if is_market_order else self.maker_fee
            commission = size * execution_price * commission_rate
            
            # 7. Check if we have enough balance
            required_balance = size * execution_price + commission
            if required_balance > self.current_balance:
                logger.warning(f"Insufficient balance for trade. Required: ${required_balance:.2f}, "
                               f"Available: ${self.current_balance:.2f}")
                self._record_rejected_order(symbol, side, size, price, timestamp, "Insufficient balance")
                return
            
            # 8. Record cost breakdown
            slippage_cost = size * price * total_slippage
            impact_cost = size * price * market_impact
            
            self._record_cost_breakdown(timestamp, commission, slippage_cost, impact_cost)
            
            # 9. Update statistics
            self.total_commission_paid += commission
            self.total_slippage_cost += slippage_cost
            self.total_market_impact += impact_cost
            
            # 10. Continue with original position logic
            self._execute_position_update(symbol, side, execution_price, size, commission, timestamp, signal_data)
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            self._record_rejected_order(symbol, side, size, price, timestamp, f"Execution error: {e}")
    
    def _execute_position_update(self, symbol: str, side: str, execution_price: float,
                                size: float, commission: float, timestamp: datetime, signal_data: Dict[str, Any]):
        """Update position after successful trade execution"""
        if symbol not in self.positions:
            self.positions[symbol] = {
                'size': 0,
                'avg_price': 0,
                'side': None
            }

        position = self.positions[symbol]

        # Handle position update
        if side == 'buy':
            if position['side'] == 'sell':
                # Closing short position
                pnl = (position['avg_price'] - execution_price) * position['size']
                self._record_trade(symbol, 'buy', position['avg_price'],
                                   execution_price, position['size'], pnl,
                                   commission, timestamp, signal_data)
                position['size'] = 0
                position['side'] = None
            else:
                # Opening/adding to long position
                total_value = (position['size'] * position['avg_price']) + (size * execution_price)
                position['size'] += size
                position['avg_price'] = total_value / position['size'] if position['size'] > 0 else 0
                position['side'] = 'buy'

        else:  # sell
            if position['side'] == 'buy':
                # Closing long position
                pnl = (execution_price - position['avg_price']) * position['size']
                self._record_trade(symbol, 'sell', position['avg_price'],
                                   execution_price, position['size'], pnl,
                                   commission, timestamp, signal_data)
                position['size'] = 0
                position['side'] = None
            else:
                # Opening/adding to short position
                total_value = (position['size'] * position['avg_price']) + (size * execution_price)
                position['size'] += size
                position['avg_price'] = total_value / position['size'] if position['size'] > 0 else 0
                position['side'] = 'sell'

        # Update balance
        self.current_balance -= commission

        logger.debug(f"Trade executed: {side} {size:.4f} {symbol} @ ${execution_price:.2f}")

    def _check_exit_conditions(self, symbol: str, candle: pd.Series):
        """Check stop loss and take profit conditions"""
        if symbol not in self.positions or self.positions[symbol]['size'] == 0:
            return

        position = self.positions[symbol]

        # Get stop loss and take profit levels
        stop_loss_pct = self.settings.get('risk_management.stop_loss_percentage', 0.02)
        take_profit_pct = self.settings.get('risk_management.take_profit_percentage', 0.05)

        if position['side'] == 'buy':
            stop_loss = position['avg_price'] * (1 - stop_loss_pct)
            take_profit = position['avg_price'] * (1 + take_profit_pct)

            # Check if hit
            if candle['low'] <= stop_loss:
                self._execute_trade(symbol, 'sell', stop_loss, position['size'],
                                    pd.Timestamp(candle.name), {'reason': 'stop_loss'})
            elif candle['high'] >= take_profit:
                self._execute_trade(symbol, 'sell', take_profit, position['size'],
                                    pd.Timestamp(candle.name), {'reason': 'take_profit'})

        elif position['side'] == 'sell':
            stop_loss = position['avg_price'] * (1 + stop_loss_pct)
            take_profit = position['avg_price'] * (1 - take_profit_pct)

            # Check if hit
            if candle['high'] >= stop_loss:
                self._execute_trade(symbol, 'buy', stop_loss, position['size'],
                                    pd.Timestamp(candle.name), {'reason': 'stop_loss'})
            elif candle['low'] <= take_profit:
                self._execute_trade(symbol, 'buy', take_profit, position['size'],
                                    pd.Timestamp(candle.name), {'reason': 'take_profit'})

    def _record_trade(self, symbol: str, side: str, entry_price: float,
                      exit_price: float, size: float, pnl: float,
                      commission: float, timestamp: datetime, signal_data: Dict[str, Any]):
        """Record completed trade"""
        trade = {
            'id': len(self.trades) + 1,
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': size,
            'pnl': pnl,
            'commission': commission,
            'net_pnl': pnl - commission,
            'return_pct': (pnl / (entry_price * size)) * 100,
            'timestamp': timestamp,
            'signal_data': signal_data
        }

        self.trades.append(trade)
        self.current_balance += pnl - commission
        self.peak_balance = max(self.peak_balance, self.current_balance)

    def _update_equity(self, timestamp: datetime):
        """Update equity curve"""
        # Calculate current equity including open positions
        total_equity = self.current_balance

        for symbol, position in self.positions.items():
            if position['size'] > 0 and symbol in self.market_data:
                current_price = self.market_data[symbol]['close']
                if position['side'] == 'buy':
                    unrealized_pnl = (current_price - position['avg_price']) * position['size']
                else:  # sell
                    unrealized_pnl = (position['avg_price'] - current_price) * position['size']
                total_equity += unrealized_pnl

        self.equity_curve.append({
            'timestamp': timestamp,
            'balance': self.current_balance,
            'equity': total_equity,
            'drawdown': (self.peak_balance - total_equity) / self.peak_balance if self.peak_balance > 0 else 0
        })

    def _close_all_positions(self, last_price: float, timestamp: datetime):
        """Close all open positions at end of backtest"""
        for symbol, position in list(self.positions.items()):
            if position['size'] > 0:
                if position['side'] == 'buy':
                    self._execute_trade(symbol, 'sell', last_price, position['size'],
                                        timestamp, {'reason': 'backtest_end'})
                else:
                    self._execute_trade(symbol, 'buy', last_price, position['size'],
                                        timestamp, {'reason': 'backtest_end'})

    def _generate_results(self, strategy_name: str, symbol: str,
                          start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate comprehensive backtest results"""
        # Calculate metrics
        metrics = self.calculate_metrics(self.trades, self.initial_capital)

        # Add additional information
        results = {
            'strategy': strategy_name,
            'symbol': symbol,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'initial_capital': self.initial_capital,
            'final_capital': self.current_balance,
            'total_return': ((self.current_balance - self.initial_capital) /
                             self.initial_capital * 100),
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'metrics': metrics,
            'backtest_config': {
                'commission': self.commission,
                'slippage': self.slippage,
                'use_limit_orders': self.use_limit_orders
            }
        }

        # Add metrics directly to results for easier access
        results.update(metrics)

        return results

    def _save_results(self, results: Dict[str, Any]):
        """Save backtest results to file"""
        # Create results directory
        results_dir = 'data/backtest_results'
        os.makedirs(results_dir, exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{results['strategy']}_{results['symbol'].replace('/', '_')}_{timestamp}.json"
        filepath = os.path.join(results_dir, filename)

        # Save results
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Backtest results saved to {filepath}")

    def _generate_empty_results(self) -> Dict[str, Any]:
        """Generate empty results structure"""
        return {
            'strategy': 'unknown',
            'symbol': 'unknown',
            'start_date': None,
            'end_date': None,
            'initial_capital': self.initial_capital,
            'final_capital': self.initial_capital,
            'total_return': 0,
            'trades': [],
            'equity_curve': [],
            'metrics': self._empty_metrics(),
            'error': 'No data available for backtesting'
        }

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'gross_profit': 0,
            'gross_loss': 0,
            'net_profit': 0,
            'total_return': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'avg_trade': 0,
            'profit_factor': 0,
            'expectancy': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'max_drawdown': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'avg_trade_duration': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0
        }

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0

        mean_return = returns.mean()
        std_return = returns.std()

        if std_return == 0:
            return 0

        # Annualized Sharpe (assuming hourly data)
        return (mean_return / std_return) * np.sqrt(24 * 365)

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        if len(returns) < 2:
            return 0

        mean_return = returns.mean()
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0:
            return float('inf') if mean_return > 0 else 0

        downside_std = downside_returns.std()

        if downside_std == 0:
            return 0

        # Annualized Sortino
        return (mean_return / downside_std) * np.sqrt(24 * 365)

    def _calculate_max_drawdown_from_trades(self, trades: List[Dict[str, Any]],
                                            initial_capital: float) -> float:
        """Calculate maximum drawdown from trades"""
        if not trades:
            return 0

        balance = initial_capital
        peak = initial_capital
        max_dd = 0

        for trade in sorted(trades, key=lambda x: x['timestamp']):
            balance += trade['net_pnl']
            peak = max(peak, balance)
            drawdown = (peak - balance) / peak if peak > 0 else 0
            max_dd = max(max_dd, drawdown)

        return max_dd

    def _calculate_max_consecutive(self, df: pd.DataFrame, wins: bool) -> int:
        """Calculate maximum consecutive wins or losses"""
        if df.empty:
            return 0

        max_consecutive = 0
        current_consecutive = 0

        for _, trade in df.iterrows():
            if wins and trade['pnl'] > 0:
                current_consecutive += 1
            elif not wins and trade['pnl'] <= 0:
                current_consecutive += 1
            else:
                current_consecutive = 0

            max_consecutive = max(max_consecutive, current_consecutive)

        return max_consecutive

    def _generate_param_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from grid"""
        import itertools

        keys = param_grid.keys()
        values = param_grid.values()

        combinations = []
        for combination in itertools.product(*values):
            combinations.append(dict(zip(keys, combination)))

        return combinations

    def _calculate_degradation(self, in_sample: Dict[str, Any],
                               out_sample: Dict[str, Any]) -> float:
        """Calculate performance degradation between IS and OOS"""
        is_sharpe = in_sample.get('sharpe_ratio', 0)
        oos_sharpe = out_sample.get('sharpe_ratio', 0)

        if is_sharpe == 0:
            return 0

        return (is_sharpe - oos_sharpe) / is_sharpe

    def _analyze_walk_forward_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze walk-forward optimization results"""
        if not results:
            return {}

        # Calculate average degradation
        degradations = [r['degradation'] for r in results]
        avg_degradation = np.mean(degradations)

        # Calculate consistency
        oos_sharpes = [r['out_sample_results'].get('sharpe_ratio', 0)
                       for r in results]
        consistency = 1 - (np.std(oos_sharpes) / np.mean(oos_sharpes) if np.mean(oos_sharpes) > 0 else 0)

        return {
            'iterations': len(results),
            'avg_degradation': avg_degradation,
            'consistency_score': consistency,
            'avg_oos_sharpe': np.mean(oos_sharpes),
            'detailed_results': results
        }

    def _calculate_portfolio_metrics(self, individual_results: Dict[str, Dict[str, Any]],
                                     combined_trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate portfolio-level metrics from multiple symbols"""
        # Sort trades by timestamp
        combined_trades.sort(key=lambda x: x['timestamp'])

        # Calculate combined metrics
        combined_metrics = self.calculate_metrics(combined_trades, self.initial_capital)

        # Add portfolio-specific metrics
        combined_metrics['symbols_traded'] = len(individual_results)
        combined_metrics['best_symbol'] = max(individual_results.items(),
                                              key=lambda x: x[1]['net_profit'])[0]
        combined_metrics['worst_symbol'] = min(individual_results.items(),
                                               key=lambda x: x[1]['net_profit'])[0]

        return combined_metrics

    # New realistic simulation methods
    
    def _detect_market_phase(self, symbol: str, data_window: pd.DataFrame) -> MarketPhase:
        """Detect current market phase"""
        try:
            if self.market_regime_detector and len(data_window) >= 20:
                # Use ML-based detection if available
                regime_data = {symbol: data_window}
                regime = self.market_regime_detector.predict_regime(regime_data)
                
                # Map regime to MarketPhase
                regime_map = {
                    'bull': MarketPhase.BULL,
                    'bear': MarketPhase.BEAR,
                    'sideways': MarketPhase.SIDEWAYS,
                    'volatile': MarketPhase.VOLATILE,
                    'extreme_fear': MarketPhase.EXTREME_FEAR
                }
                
                phase_name = regime.get('label', 'sideways')
                return regime_map.get(phase_name, MarketPhase.SIDEWAYS)
            
            else:
                # Fallback to simple heuristic detection
                if len(data_window) < 20:
                    return MarketPhase.SIDEWAYS
                
                returns = data_window['close'].pct_change().dropna()
                volatility = returns.std()
                trend = returns.mean()
                
                # Simple classification
                if volatility > 0.05:  # 5% daily volatility
                    return MarketPhase.VOLATILE
                elif trend > 0.02:  # 2% daily return
                    return MarketPhase.BULL
                elif trend < -0.02:  # -2% daily return
                    return MarketPhase.BEAR
                else:
                    return MarketPhase.SIDEWAYS
                    
        except Exception as e:
            logger.error(f"Error detecting market phase: {e}")
            return MarketPhase.SIDEWAYS
    
    def _calculate_dynamic_slippage(self, symbol: str, size: float, price: float) -> float:
        """Calculate realistic slippage based on market conditions"""
        try:
            # Base slippage
            slippage = self.base_slippage
            
            # Volatility-based adjustment
            if symbol in self.market_data:
                # Calculate recent volatility
                recent_returns = []
                # Use simple volatility estimate
                volatility = 0.02  # Default 2%
                
                # Adjust slippage based on volatility
                volatility_adjustment = volatility * self.volatility_slippage_factor
                slippage += volatility_adjustment
            
            # Size-based adjustment (larger orders have more slippage)
            if symbol in self.market_data:
                volume = self.market_data[symbol].get('volume', 1000000)
                size_impact = (size * price) / (volume * price) * self.size_slippage_factor
                slippage += size_impact
            
            # Market phase adjustment
            phase_multipliers = {
                MarketPhase.BULL: 0.8,
                MarketPhase.BEAR: 1.2,
                MarketPhase.SIDEWAYS: 1.0,
                MarketPhase.VOLATILE: 1.5,
                MarketPhase.EXTREME_FEAR: 2.0
            }
            
            slippage *= phase_multipliers.get(self.current_market_phase, 1.0)
            
            # Cap slippage at reasonable level
            return min(slippage, 0.01)  # Max 1% slippage
            
        except Exception as e:
            logger.error(f"Error calculating slippage: {e}")
            return self.base_slippage
    
    def _calculate_market_impact(self, symbol: str, size: float) -> float:
        """Calculate market impact based on order size"""
        try:
            if not self.enable_market_impact:
                return 0
            
            # Get current volume if available
            volume = 1000000  # Default volume
            if symbol in self.market_data:
                volume = self.market_data[symbol].get('volume', 1000000)
            
            # Calculate size relative to volume
            volume_ratio = size / volume if volume > 0 else 0.001
            
            # Market impact increases non-linearly with size
            impact = self.impact_factor * np.sqrt(volume_ratio)
            
            # Liquidity level adjustment
            liquidity_multipliers = {
                LiquidityLevel.HIGH: 0.5,
                LiquidityLevel.MEDIUM: 1.0,
                LiquidityLevel.LOW: 1.5,
                LiquidityLevel.VERY_LOW: 2.5
            }
            
            impact *= liquidity_multipliers.get(self.current_liquidity, 1.0)
            
            return min(impact, 0.005)  # Cap at 0.5%
            
        except Exception as e:
            logger.error(f"Error calculating market impact: {e}")
            return 0
    
    def _check_liquidity_constraints(self, symbol: str, size: float) -> bool:
        """Check if order can be filled given liquidity constraints"""
        try:
            if not self.enable_liquidity_constraints:
                return True
            
            # Get current volume
            volume = 1000000  # Default
            if symbol in self.market_data:
                volume = self.market_data[symbol].get('volume', 1000000)
            
            # Check if order size exceeds maximum participation
            max_size = volume * self.max_volume_participation
            
            if size > max_size:
                logger.debug(f"Order size {size} exceeds liquidity limit {max_size}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking liquidity constraints: {e}")
            return True
    
    def _simulate_price_movement_during_latency(self, symbol: str, price: float, latency_ms: float) -> float:
        """Simulate realistic price movement during order latency"""
        try:
            # Convert latency to price movement factor
            latency_seconds = latency_ms / 1000.0
            
            # Estimate price volatility per second
            volatility_per_second = 0.0001  # 0.01% per second default
            
            # Random price movement during latency
            price_change = np.random.normal(0, volatility_per_second * np.sqrt(latency_seconds))
            
            # Apply market phase bias
            phase_bias = {
                MarketPhase.BULL: 0.00001,
                MarketPhase.BEAR: -0.00001,
                MarketPhase.SIDEWAYS: 0,
                MarketPhase.VOLATILE: 0,
                MarketPhase.EXTREME_FEAR: -0.00002
            }
            
            bias = phase_bias.get(self.current_market_phase, 0) * latency_seconds
            price_change += bias
            
            new_price = price * (1 + price_change)
            return max(new_price, price * 0.99)  # Prevent extreme moves
            
        except Exception as e:
            logger.error(f"Error simulating price movement: {e}")
            return price
    
    def _record_rejected_order(self, symbol: str, side: str, size: float, price: float, 
                              timestamp: datetime, reason: str):
        """Record rejected order for analysis"""
        rejected_order = {
            'timestamp': timestamp,
            'symbol': symbol,
            'side': side,
            'size': size,
            'price': price,
            'reason': reason
        }
        
        self.rejected_orders.append(rejected_order)
        self.orders_rejected += 1
        
        logger.debug(f"Order rejected: {side} {size} {symbol} @ {price:.4f} - {reason}")
    
    def _record_cost_breakdown(self, timestamp: datetime, commission: float, 
                              slippage_cost: float, impact_cost: float):
        """Record detailed cost breakdown"""
        cost_record = {
            'timestamp': timestamp,
            'commission': commission,
            'slippage_cost': slippage_cost,
            'impact_cost': impact_cost,
            'total_cost': commission + slippage_cost + impact_cost,
            'market_phase': self.current_market_phase.value
        }
        
        self.cost_breakdown.append(cost_record)
    
    def _update_market_phase_tracking(self, symbol: str, data_window: pd.DataFrame, timestamp: datetime):
        """Update market phase tracking"""
        try:
            new_phase = self._detect_market_phase(symbol, data_window)
            
            if new_phase != self.current_market_phase:
                logger.info(f"Market phase changed: {self.current_market_phase.value} -> {new_phase.value}")
                self.current_market_phase = new_phase
            
            # Record phase data
            phase_record = {
                'timestamp': timestamp,
                'phase': new_phase.value,
                'equity': self.current_balance
            }
            
            self.market_phases.append(phase_record)
            
            # Track performance by phase
            if self.trades:
                latest_trade = self.trades[-1]
                self.phase_performance[new_phase.value]['trades'].append(latest_trade)
            
            equity_record = {
                'timestamp': timestamp,
                'equity': self.current_balance
            }
            self.phase_performance[new_phase.value]['equity'].append(equity_record)
            
        except Exception as e:
            logger.error(f"Error updating market phase tracking: {e}")
    
    def get_enhanced_results(self) -> Dict[str, Any]:
        """Get enhanced results with realistic simulation data"""
        try:
            # Basic results
            basic_results = self._generate_results("Unknown", "Unknown", datetime.now(), datetime.now())
            
            # Add enhanced metrics
            enhanced_results = basic_results.copy()
            
            # Cost analysis
            enhanced_results['cost_analysis'] = {
                'total_commission_paid': self.total_commission_paid,
                'total_slippage_cost': self.total_slippage_cost,
                'total_market_impact': self.total_market_impact,
                'total_transaction_costs': self.total_commission_paid + self.total_slippage_cost + self.total_market_impact,
                'cost_breakdown': self.cost_breakdown
            }
            
            # Order execution analysis
            enhanced_results['execution_analysis'] = {
                'orders_rejected': self.orders_rejected,
                'rejection_rate': self.orders_rejected / (len(self.trades) + self.orders_rejected) if (len(self.trades) + self.orders_rejected) > 0 else 0,
                'rejected_orders': self.rejected_orders
            }
            
            # Market phase performance
            phase_metrics = {}
            for phase, data in self.phase_performance.items():
                if data['trades']:
                    phase_trades = data['trades']
                    phase_pnl = sum(trade['net_pnl'] for trade in phase_trades)
                    phase_metrics[phase] = {
                        'trade_count': len(phase_trades),
                        'total_pnl': phase_pnl,
                        'avg_pnl_per_trade': phase_pnl / len(phase_trades),
                        'win_rate': len([t for t in phase_trades if t['net_pnl'] > 0]) / len(phase_trades)
                    }
                else:
                    phase_metrics[phase] = {
                        'trade_count': 0,
                        'total_pnl': 0,
                        'avg_pnl_per_trade': 0,
                        'win_rate': 0
                    }
            
            enhanced_results['market_phase_performance'] = phase_metrics
            enhanced_results['market_phases_history'] = self.market_phases
            
            # Realistic simulation settings
            enhanced_results['simulation_settings'] = {
                'maker_fee': self.maker_fee,
                'taker_fee': self.taker_fee,
                'base_slippage': self.base_slippage,
                'market_impact_enabled': self.enable_market_impact,
                'latency_enabled': self.enable_latency,
                'liquidity_constraints_enabled': self.enable_liquidity_constraints,
                'market_regime_detection_enabled': self.market_regime_detector is not None
            }
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error generating enhanced results: {e}")
            return self._generate_empty_results()


