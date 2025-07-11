# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performance Tracker - Comprehensive Performance Analytics
========================================================

Tracks and analyzes trading performance:
- Trade history and statistics
- Performance metrics calculation
- Report generation
- Benchmark comparison
"""

import logging
import json
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Comprehensive performance tracking and analytics"""

    def __init__(self, settings):
        """Initialize Performance Tracker"""
        self.settings = settings
        self.data_dir = 'data/reports'

        # Create reports directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)

        # Performance data storage
        self.trades = []
        self.daily_returns = []
        self.equity_curve = []
        self.metrics_history = []

        # Initialize from saved data if exists
        self._load_historical_data()

        logger.info("Performance Tracker initialized")

    def record_trade(self, trade_data: Dict[str, Any]):
        """Record a completed trade"""
        trade = {
            'id': len(self.trades) + 1,
            'symbol': trade_data['symbol'],
            'side': trade_data['side'],
            'entry_price': trade_data['entry_price'],
            'exit_price': trade_data['exit_price'],
            'size': trade_data['size'],
            'entry_time': trade_data.get('entry_time', datetime.now()),
            'exit_time': trade_data.get('exit_time', datetime.now()),
            'duration': self._calculate_trade_duration(
                trade_data.get('entry_time', datetime.now()),
                trade_data.get('exit_time', datetime.now())
            ),
            'pnl': trade_data['pnl'],
            'pnl_pct': ((trade_data['exit_price'] - trade_data['entry_price']) /
                        trade_data['entry_price'] * 100),
            'fees': trade_data.get('fees', 0),
            'net_pnl': trade_data['pnl'] - trade_data.get('fees', 0),
            'strategy': trade_data.get('strategy', 'unknown'),
            'win': trade_data['pnl'] > 0
        }

        self.trades.append(trade)
        self._update_equity_curve(trade)
        self._save_trade(trade)

        logger.info(f"Trade recorded: {trade['symbol']} P&L: ${trade['pnl']:.2f} ({trade['pnl_pct']:.2f}%)")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.trades:
            return self._empty_summary()

        # Basic statistics
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t['win'])
        losing_trades = total_trades - winning_trades

        # P&L calculations
        gross_profit = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))
        net_profit = sum(t['net_pnl'] for t in self.trades)
        total_fees = sum(t['fees'] for t in self.trades)

        # Win/Loss metrics
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0

        # Risk metrics
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        expectancy = net_profit / total_trades if total_trades > 0 else 0

        # Time-based metrics
        returns = [t['pnl_pct'] for t in self.trades]
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = self._calculate_calmar_ratio()

        # Drawdown metrics
        max_drawdown, max_drawdown_duration = self._calculate_max_drawdown()
        current_drawdown = self._calculate_current_drawdown()

        # Per-strategy breakdown
        strategy_performance = self._calculate_strategy_performance()

        # Per-symbol breakdown
        symbol_performance = self._calculate_symbol_performance()

        summary = {
            # Trade Statistics
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,

            # P&L Metrics
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'net_profit': net_profit,
            'total_fees': total_fees,
            'profit_factor': profit_factor,
            'expectancy': expectancy,

            # Average Metrics
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade_pnl': net_profit / total_trades if total_trades > 0 else 0,
            'largest_win': max((t['pnl'] for t in self.trades), default=0),
            'largest_loss': min((t['pnl'] for t in self.trades), default=0),

            # Risk-Adjusted Returns
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,

            # Drawdown Metrics
            'max_drawdown': max_drawdown,
            'max_drawdown_duration_days': max_drawdown_duration,
            'current_drawdown': current_drawdown,

            # Time Metrics
            'avg_trade_duration_hours': np.mean([t['duration'] for t in self.trades]),
            'total_trading_days': self._calculate_trading_days(),

            # Breakdowns
            'strategy_performance': strategy_performance,
            'symbol_performance': symbol_performance,

            # Recent Performance
            'last_7_days': self._calculate_period_performance(7),
            'last_30_days': self._calculate_period_performance(30),
            'last_90_days': self._calculate_period_performance(90),

            # Best/Worst Periods
            'best_day': self._find_best_period('day'),
            'worst_day': self._find_worst_period('day'),
            'best_week': self._find_best_period('week'),
            'worst_week': self._find_worst_period('week'),

            # Consistency Metrics
            'win_streak_current': self._calculate_current_streak(),
            'win_streak_max': self._calculate_max_win_streak(),
            'loss_streak_max': self._calculate_max_loss_streak(),
            'consistency_score': self._calculate_consistency_score(),

            # Metadata
            'last_updated': datetime.now().isoformat(),
            'data_quality_score': self._calculate_data_quality_score()
        }

        return summary

    def get_trade_history(self, limit: Optional[int] = None,
                          strategy: Optional[str] = None,
                          symbol: Optional[str] = None,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get filtered trade history"""
        trades = self.trades.copy()

        # Apply filters
        if strategy:
            trades = [t for t in trades if t['strategy'] == strategy]

        if symbol:
            trades = [t for t in trades if t['symbol'] == symbol]

        if start_date:
            trades = [t for t in trades if t['entry_time'] >= start_date]

        if end_date:
            trades = [t for t in trades if t['exit_time'] <= end_date]

        # Sort by exit time (most recent first)
        trades.sort(key=lambda x: x['exit_time'], reverse=True)

        # Apply limit
        if limit:
            trades = trades[:limit]

        return trades

    def generate_performance_report(self, format: str = 'text') -> str:
        """Generate a detailed performance report"""
        summary = self.get_performance_summary()

        if format == 'json':
            return json.dumps(summary, indent=2, default=str)

        elif format == 'html':
            return self._generate_html_report(summary)

        else:  # text format
            report = f"""
================== PERFORMANCE REPORT ==================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW:
---------
Total Trades: {summary['total_trades']}
Win Rate: {summary['win_rate']:.1%}
Net Profit: ${summary['net_profit']:,.2f}
Profit Factor: {summary['profit_factor']:.2f}

TRADE STATISTICS:
-----------------
Winning Trades: {summary['winning_trades']} (${summary['gross_profit']:,.2f})
Losing Trades: {summary['losing_trades']} (-${summary['gross_loss']:,.2f})
Average Win: ${summary['avg_win']:,.2f}
Average Loss: ${summary['avg_loss']:,.2f}
Largest Win: ${summary['largest_win']:,.2f}
Largest Loss: ${summary['largest_loss']:,.2f}

RISK METRICS:
-------------
Sharpe Ratio: {summary['sharpe_ratio']:.2f}
Sortino Ratio: {summary['sortino_ratio']:.2f}
Calmar Ratio: {summary['calmar_ratio']:.2f}
Max Drawdown: {summary['max_drawdown']:.1%}
Current Drawdown: {summary['current_drawdown']:.1%}
Max DD Duration: {summary['max_drawdown_duration_days']} days

PERFORMANCE BY PERIOD:
---------------------
Last 7 Days: ${summary['last_7_days']['net_profit']:,.2f} ({summary['last_7_days']['return_pct']:.1f}%)
Last 30 Days: ${summary['last_30_days']['net_profit']:,.2f} ({summary['last_30_days']['return_pct']:.1f}%)
Last 90 Days: ${summary['last_90_days']['net_profit']:,.2f} ({summary['last_90_days']['return_pct']:.1f}%)

CONSISTENCY:
------------
Current Streak: {summary['win_streak_current']} {'wins' if summary['win_streak_current'] > 0 else 'losses'}
Max Win Streak: {summary['win_streak_max']}
Max Loss Streak: {summary['loss_streak_max']}
Consistency Score: {summary['consistency_score']:.1f}/100

TOP PERFORMING STRATEGIES:
-------------------------"""

            # Add strategy performance
            for strategy, perf in summary['strategy_performance'].items():
                report += f"\n{strategy}: {perf['trades']} trades, {perf['win_rate']:.1%} win rate, ${perf['net_profit']:,.2f}"

            # Add symbol performance
            report += "\n\nTOP PERFORMING SYMBOLS:\n-----------------------"
            sorted_symbols = sorted(summary['symbol_performance'].items(),
                                    key=lambda x: x[1]['net_profit'], reverse=True)[:5]
            for symbol, perf in sorted_symbols:
                report += f"\n{symbol}: {perf['trades']} trades, ${perf['net_profit']:,.2f}"

            report += "\n\n" + "=" * 50 + "\n"

            return report

    def calculate_benchmark_comparison(self, benchmark_data: pd.DataFrame) -> Dict[str, Any]:
        """Compare performance against a benchmark"""
        if not self.equity_curve:
            return {}

        # Calculate returns
        strategy_returns = pd.Series([e['return'] for e in self.equity_curve])
        benchmark_returns = benchmark_data['return']

        # Align dates
        min_length = min(len(strategy_returns), len(benchmark_returns))
        strategy_returns = strategy_returns[:min_length]
        benchmark_returns = benchmark_returns[:min_length]

        # Calculate metrics
        correlation = strategy_returns.corr(benchmark_returns)

        # Calculate alpha and beta
        covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

        strategy_return_annual = strategy_returns.mean() * 252
        benchmark_return_annual = benchmark_returns.mean() * 252
        risk_free_rate = 0.02  # 2% assumed risk-free rate

        alpha = strategy_return_annual - (risk_free_rate + beta * (benchmark_return_annual - risk_free_rate))

        # Information ratio
        excess_returns = strategy_returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0

        return {
            'correlation': correlation,
            'alpha': alpha,
            'beta': beta,
            'information_ratio': information_ratio,
            'outperformance': strategy_return_annual - benchmark_return_annual,
            'strategy_annual_return': strategy_return_annual,
            'benchmark_annual_return': benchmark_return_annual
        }

    def export_trades_to_csv(self, filename: Optional[str] = None) -> str:
        """Export trade history to CSV file"""
        if not filename:
            filename = f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        filepath = os.path.join(self.data_dir, filename)

        df = pd.DataFrame(self.trades)
        df.to_csv(filepath, index=False)

        logger.info(f"Trades exported to {filepath}")
        return filepath

    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame"""
        if not self.equity_curve:
            return pd.DataFrame()

        df = pd.DataFrame(self.equity_curve)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        return df

    # Private helper methods

    def _load_historical_data(self):
        """Load historical performance data from files"""
        trades_file = os.path.join(self.data_dir, 'trades_history.json')

        if os.path.exists(trades_file):
            try:
                with open(trades_file, 'r') as f:
                    data = json.load(f)
                    self.trades = data.get('trades', [])
                    self.equity_curve = data.get('equity_curve', [])

                    # Convert timestamps back to datetime
                    for trade in self.trades:
                        trade['entry_time'] = datetime.fromisoformat(trade['entry_time'])
                        trade['exit_time'] = datetime.fromisoformat(trade['exit_time'])

                    logger.info(f"Loaded {len(self.trades)} historical trades")
            except Exception as e:
                logger.error(f"Error loading historical data: {e}")

    def _save_trade(self, trade: Dict[str, Any]):
        """Save trade to persistent storage"""
        trades_file = os.path.join(self.data_dir, 'trades_history.json')

        # Prepare data for JSON serialization
        save_data = {
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'last_updated': datetime.now().isoformat()
        }

        # Convert datetime objects to strings
        for t in save_data['trades']:
            if isinstance(t['entry_time'], datetime):
                t['entry_time'] = t['entry_time'].isoformat()
            if isinstance(t['exit_time'], datetime):
                t['exit_time'] = t['exit_time'].isoformat()

        try:
            with open(trades_file, 'w') as f:
                json.dump(save_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving trade data: {e}")

    def _update_equity_curve(self, trade: Dict[str, Any]):
        """Update equity curve with new trade"""
        if not self.equity_curve:
            starting_balance = 10000  # Default starting balance
            self.equity_curve.append({
                'timestamp': trade['entry_time'],
                'balance': starting_balance,
                'return': 0,
                'drawdown': 0
            })

        last_balance = self.equity_curve[-1]['balance']
        new_balance = last_balance + trade['net_pnl']

        # Calculate return
        daily_return = trade['net_pnl'] / last_balance

        # Update peak for drawdown calculation
        peak_balance = max(e['balance'] for e in self.equity_curve)
        drawdown = (peak_balance - new_balance) / peak_balance if peak_balance > 0 else 0

        self.equity_curve.append({
            'timestamp': trade['exit_time'],
            'balance': new_balance,
            'return': daily_return,
            'drawdown': drawdown,
            'trade_id': trade['id']
        })

    def _calculate_trade_duration(self, entry_time: datetime, exit_time: datetime) -> float:
        """Calculate trade duration in hours"""
        duration = exit_time - entry_time
        return duration.total_seconds() / 3600

    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0

        returns_array = np.array(returns) / 100  # Convert percentage to decimal
        avg_return = np.mean(returns_array)
        std_return = np.std(returns_array)

        if std_return == 0:
            return 0

        # Annualized Sharpe ratio (assuming daily returns)
        return (avg_return / std_return) * np.sqrt(252)

    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Calculate Sortino ratio (uses downside deviation)"""
        if len(returns) < 2:
            return 0

        returns_array = np.array(returns) / 100
        avg_return = np.mean(returns_array)

        # Calculate downside deviation
        negative_returns = returns_array[returns_array < 0]
        downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 0

        if downside_std == 0:
            return 0

        # Annualized Sortino ratio
        return (avg_return / downside_std) * np.sqrt(252)

    def _calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        if not self.trades:
            return 0

        # Calculate annual return
        total_days = self._calculate_trading_days()
        if total_days == 0:
            return 0

        total_return = sum(t['pnl_pct'] for t in self.trades) / 100
        annual_return = (1 + total_return) ** (365 / total_days) - 1

        # Get max drawdown
        max_drawdown, _ = self._calculate_max_drawdown()

        if max_drawdown == 0:
            return float('inf') if annual_return > 0 else 0

        return annual_return / abs(max_drawdown)

    def _calculate_max_drawdown(self) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration"""
        if not self.equity_curve:
            return 0, 0

        peak = self.equity_curve[0]['balance']
        max_dd = 0
        max_dd_duration = 0
        current_dd_start = None

        for point in self.equity_curve:
            if point['balance'] > peak:
                peak = point['balance']
                current_dd_start = None
            else:
                drawdown = (peak - point['balance']) / peak
                if drawdown > max_dd:
                    max_dd = drawdown

                if current_dd_start is None:
                    current_dd_start = point['timestamp']

                # Calculate duration
                if current_dd_start:
                    duration = (point['timestamp'] - current_dd_start).days
                    max_dd_duration = max(max_dd_duration, duration)

        return max_dd, max_dd_duration

    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak"""
        if not self.equity_curve:
            return 0

        peak = max(e['balance'] for e in self.equity_curve)
        current = self.equity_curve[-1]['balance']

        return (peak - current) / peak if peak > 0 else 0

    def _calculate_strategy_performance(self) -> Dict[str, Any]:
        """Calculate performance by strategy"""
        strategy_stats = defaultdict(lambda: {
            'trades': 0, 'wins': 0, 'gross_profit': 0,
            'gross_loss': 0, 'net_profit': 0
        })

        for trade in self.trades:
            strategy = trade['strategy']
            strategy_stats[strategy]['trades'] += 1

            if trade['win']:
                strategy_stats[strategy]['wins'] += 1
                strategy_stats[strategy]['gross_profit'] += trade['pnl']
            else:
                strategy_stats[strategy]['gross_loss'] += abs(trade['pnl'])

            strategy_stats[strategy]['net_profit'] += trade['net_pnl']

        # Calculate derived metrics
        for strategy, stats in strategy_stats.items():
            stats['win_rate'] = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
            stats['profit_factor'] = (stats['gross_profit'] / stats['gross_loss']
                                      if stats['gross_loss'] > 0 else float('inf'))

        return dict(strategy_stats)

    def _calculate_symbol_performance(self) -> Dict[str, Any]:
        """Calculate performance by symbol"""
        symbol_stats = defaultdict(lambda: {
            'trades': 0, 'wins': 0, 'net_profit': 0, 'avg_pnl': 0
        })

        for trade in self.trades:
            symbol = trade['symbol']
            symbol_stats[symbol]['trades'] += 1

            if trade['win']:
                symbol_stats[symbol]['wins'] += 1

            symbol_stats[symbol]['net_profit'] += trade['net_pnl']

        # Calculate derived metrics
        for symbol, stats in symbol_stats.items():
            stats['win_rate'] = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
            stats['avg_pnl'] = stats['net_profit'] / stats['trades'] if stats['trades'] > 0 else 0

        return dict(symbol_stats)

    def _calculate_period_performance(self, days: int) -> Dict[str, Any]:
        """Calculate performance for last N days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        period_trades = [t for t in self.trades if t['exit_time'] >= cutoff_date]

        if not period_trades:
            return {'trades': 0, 'net_profit': 0, 'return_pct': 0, 'win_rate': 0}

        wins = sum(1 for t in period_trades if t['win'])
        net_profit = sum(t['net_pnl'] for t in period_trades)

        # Calculate return percentage (simplified)
        initial_balance = 10000  # Assumed
        return_pct = (net_profit / initial_balance) * 100

        return {
            'trades': len(period_trades),
            'net_profit': net_profit,
            'return_pct': return_pct,
            'win_rate': wins / len(period_trades) if period_trades else 0
        }

    def _find_best_period(self, period_type: str) -> Dict[str, Any]:
        """Find best performing period"""
        if not self.trades:
            return {}

        # Group trades by period
        if period_type == 'day':
            grouped = defaultdict(list)
            for trade in self.trades:
                day = trade['exit_time'].date()
                grouped[day].append(trade)

        elif period_type == 'week':
            grouped = defaultdict(list)
            for trade in self.trades:
                week = trade['exit_time'].isocalendar()[1]
                year = trade['exit_time'].year
                grouped[(year, week)].append(trade)

        # Find best period
        best_period = None
        best_profit = float('-inf')

        for period, trades in grouped.items():
            period_profit = sum(t['net_pnl'] for t in trades)
            if period_profit > best_profit:
                best_profit = period_profit
                best_period = period

        if best_period:
            return {
                'period': str(best_period),
                'profit': best_profit,
                'trades': len(grouped[best_period])
            }

        return {}

    def _find_worst_period(self, period_type: str) -> Dict[str, Any]:
        """Find worst performing period"""
        if not self.trades:
            return {}

        # Group trades by period
        if period_type == 'day':
            grouped = defaultdict(list)
            for trade in self.trades:
                day = trade['exit_time'].date()
                grouped[day].append(trade)

        elif period_type == 'week':
            grouped = defaultdict(list)
            for trade in self.trades:
                week = trade['exit_time'].isocalendar()[1]
                year = trade['exit_time'].year
                grouped[(year, week)].append(trade)

        # Find worst period
        worst_period = None
        worst_profit = float('inf')

        for period, trades in grouped.items():
            period_profit = sum(t['net_pnl'] for t in trades)
            if period_profit < worst_profit:
                worst_profit = period_profit
                worst_period = period

        if worst_period:
            return {
                'period': str(worst_period),
                'profit': worst_profit,
                'trades': len(grouped[worst_period])
            }

        return {}

    def _calculate_current_streak(self) -> int:
        """Calculate current win/loss streak"""
        if not self.trades:
            return 0

        # Sort trades by exit time
        sorted_trades = sorted(self.trades, key=lambda x: x['exit_time'])

        streak = 0
        last_win = sorted_trades[-1]['win']

        for trade in reversed(sorted_trades):
            if trade['win'] == last_win:
                streak += 1 if last_win else -1
            else:
                break

        return streak

    def _calculate_max_win_streak(self) -> int:
        """Calculate maximum consecutive wins"""
        if not self.trades:
            return 0

        max_streak = 0
        current_streak = 0

        for trade in sorted(self.trades, key=lambda x: x['exit_time']):
            if trade['win']:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    def _calculate_max_loss_streak(self) -> int:
        """Calculate maximum consecutive losses"""
        if not self.trades:
            return 0

        max_streak = 0
        current_streak = 0

        for trade in sorted(self.trades, key=lambda x: x['exit_time']):
            if not trade['win']:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    def _calculate_consistency_score(self) -> float:
        """Calculate consistency score (0-100)"""
        if len(self.trades) < 10:
            return 0

        # Factors for consistency
        score = 100

        # Factor 1: Win rate stability
        # Calculate rolling win rate
        window = min(20, len(self.trades) // 5)
        win_rates = []

        for i in range(window, len(self.trades)):
            window_trades = self.trades[i - window:i]
            wins = sum(1 for t in window_trades if t['win'])
            win_rates.append(wins / window)

        if win_rates:
            win_rate_std = np.std(win_rates)
            score -= win_rate_std * 100  # Penalize high variance

        # Factor 2: Profit stability
        returns = [t['pnl_pct'] for t in self.trades]
        if returns:
            return_std = np.std(returns)
            score -= min(return_std, 20)  # Cap penalty at 20

        # Factor 3: Streak length
        max_loss_streak = self._calculate_max_loss_streak()
        score -= max_loss_streak * 5  # -5 points per loss in max streak

        return max(0, min(100, score))

    def _calculate_trading_days(self) -> int:
        """Calculate total number of trading days"""
        if not self.trades:
            return 0

        first_trade = min(self.trades, key=lambda x: x['entry_time'])
        last_trade = max(self.trades, key=lambda x: x['exit_time'])

        return (last_trade['exit_time'] - first_trade['entry_time']).days

    def _calculate_data_quality_score(self) -> float:
        """Calculate data quality score"""
        if not self.trades:
            return 0

        score = 100

        # Check for missing data
        for trade in self.trades:
            if trade.get('fees', 0) == 0:
                score -= 0.5  # Likely missing fee data
            if trade.get('strategy') == 'unknown':
                score -= 1  # Missing strategy info

        return max(0, score)

    def _empty_summary(self) -> Dict[str, Any]:
        """Return empty summary structure"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'gross_profit': 0,
            'gross_loss': 0,
            'net_profit': 0,
            'total_fees': 0,
            'profit_factor': 0,
            'expectancy': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'avg_trade_pnl': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            'max_drawdown': 0,
            'max_drawdown_duration_days': 0,
            'current_drawdown': 0,
            'avg_trade_duration_hours': 0,
            'total_trading_days': 0,
            'strategy_performance': {},
            'symbol_performance': {},
            'last_7_days': {'trades': 0, 'net_profit': 0, 'return_pct': 0, 'win_rate': 0},
            'last_30_days': {'trades': 0, 'net_profit': 0, 'return_pct': 0, 'win_rate': 0},
            'last_90_days': {'trades': 0, 'net_profit': 0, 'return_pct': 0, 'win_rate': 0},
            'best_day': {},
            'worst_day': {},
            'best_week': {},
            'worst_week': {},
            'win_streak_current': 0,
            'win_streak_max': 0,
            'loss_streak_max': 0,
            'consistency_score': 0,
            'last_updated': datetime.now().isoformat(),
            'data_quality_score': 0
        }

    def _generate_html_report(self, summary: Dict[str, Any]) -> str:
        """Generate HTML format report"""
        # Basic HTML template
        html = f"""
        <html>
        <head>
            <title>Trading Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Trading Performance Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <h2>Overview</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Trades</td><td>{summary['total_trades']}</td></tr>
                <tr><td>Win Rate</td><td>{summary['win_rate']:.1%}</td></tr>
                <tr><td>Net Profit</td><td class="{'positive' if summary['net_profit'] > 0 else 'negative'}">${summary['net_profit']:,.2f}</td></tr>
                <tr><td>Profit Factor</td><td>{summary['profit_factor']:.2f}</td></tr>
            </table>

            <!-- Add more sections as needed -->

        </body>
        </html>
        """

        return html


