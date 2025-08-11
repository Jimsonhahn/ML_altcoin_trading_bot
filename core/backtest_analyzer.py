# core/backtest_analyzer.py
"""
Advanced Backtest Analysis System
Provides detailed performance analysis, risk metrics, and strategy comparison
"""

import logging
import json
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
import statistics
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Return metrics
    total_return: float
    annualized_return: float
    monthly_return: float
    daily_return: float
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    volatility: float
    downside_deviation: float
    
    # Trade metrics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    
    # Consistency metrics
    monthly_win_rate: float
    consecutive_wins: int
    consecutive_losses: int
    recovery_factor: float
    
    # Cost metrics
    total_costs: float
    cost_ratio: float  # costs as % of total return


@dataclass
class MarketPhaseAnalysis:
    """Market phase performance analysis"""
    phase: str
    trade_count: int
    total_return: float
    win_rate: float
    avg_trade_return: float
    sharpe_ratio: float
    max_drawdown: float
    duration_days: int
    best_trade: float
    worst_trade: float


@dataclass
class StrategyComparison:
    """Strategy comparison metrics"""
    strategy_name: str
    metrics: PerformanceMetrics
    rank: int
    correlation_with_market: float
    correlation_with_other_strategies: Dict[str, float]


class BacktestAnalyzer:
    """
    Advanced analyzer for backtest results with detailed performance metrics
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)  # 2% annual
        self.benchmark_return = self.config.get('benchmark_return', 0.08)  # 8% S&P500
        
        # Analysis results storage
        self.results_cache = {}
        self.comparison_cache = {}
        
        logger.info("BacktestAnalyzer initialized")
    
    def analyze_single_strategy(self, backtest_results: Dict[str, Any], 
                               benchmark_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of single strategy backtest results
        """
        try:
            logger.info(f"Analyzing strategy: {backtest_results.get('strategy', 'Unknown')}")
            
            # Extract basic data
            trades = pd.DataFrame(backtest_results.get('trades', []))
            equity_curve = pd.DataFrame(backtest_results.get('equity_curve', []))
            
            if trades.empty:
                logger.warning("No trades found in backtest results")
                return self._create_empty_analysis()
            
            # Convert timestamps
            trades['timestamp'] = pd.to_datetime(trades['timestamp'])
            equity_curve['timestamp'] = pd.to_datetime(equity_curve['timestamp'])
            
            # Calculate comprehensive metrics
            performance_metrics = self._calculate_performance_metrics(trades, equity_curve, backtest_results)
            
            # Market phase analysis
            market_phase_analysis = self._analyze_market_phases(backtest_results)
            
            # Risk analysis
            risk_analysis = self._calculate_risk_metrics(trades, equity_curve)
            
            # Trade analysis
            trade_analysis = self._analyze_trades(trades)
            
            # Cost analysis
            cost_analysis = self._analyze_costs(backtest_results)
            
            # Time-based analysis
            time_analysis = self._analyze_time_patterns(trades)
            
            # Benchmark comparison
            benchmark_analysis = {}
            if benchmark_data is not None:
                benchmark_analysis = self._compare_to_benchmark(equity_curve, benchmark_data)
            
            # Monte Carlo analysis
            monte_carlo = self._monte_carlo_analysis(trades)
            
            # Compile comprehensive analysis
            analysis = {
                'strategy_name': backtest_results.get('strategy', 'Unknown'),
                'symbol': backtest_results.get('symbol', 'Unknown'),
                'period': {
                    'start': backtest_results.get('start_date'),
                    'end': backtest_results.get('end_date'),
                    'duration_days': self._calculate_duration_days(backtest_results)
                },
                'performance_metrics': performance_metrics.__dict__,
                'market_phase_analysis': [phase.__dict__ for phase in market_phase_analysis],
                'risk_analysis': risk_analysis,
                'trade_analysis': trade_analysis,
                'cost_analysis': cost_analysis,
                'time_analysis': time_analysis,
                'benchmark_analysis': benchmark_analysis,
                'monte_carlo_analysis': monte_carlo,
                'recommendations': self._generate_recommendations(performance_metrics, risk_analysis)
            }
            
            # Cache results
            cache_key = f"{backtest_results.get('strategy')}_{backtest_results.get('symbol')}"
            self.results_cache[cache_key] = analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing strategy: {e}")
            return self._create_empty_analysis()
    
    def compare_strategies(self, strategy_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple strategies and rank them
        """
        try:
            logger.info(f"Comparing {len(strategy_results)} strategies")
            
            if len(strategy_results) < 2:
                logger.warning("Need at least 2 strategies for comparison")
                return {}
            
            # Analyze each strategy individually
            strategy_analyses = []
            for results in strategy_results:
                analysis = self.analyze_single_strategy(results)
                strategy_analyses.append(analysis)
            
            # Calculate correlations
            correlation_matrix = self._calculate_strategy_correlations(strategy_results)
            
            # Rank strategies
            rankings = self._rank_strategies(strategy_analyses)
            
            # Portfolio analysis (if combining strategies)
            portfolio_analysis = self._analyze_portfolio_combination(strategy_results)
            
            # Risk-adjusted rankings
            risk_adjusted_rankings = self._risk_adjusted_rankings(strategy_analyses)
            
            # Market condition analysis
            market_condition_performance = self._analyze_market_condition_performance(strategy_analyses)
            
            comparison = {
                'total_strategies': len(strategy_results),
                'analysis_timestamp': datetime.now().isoformat(),
                'individual_analyses': strategy_analyses,
                'correlation_matrix': correlation_matrix,
                'rankings': {
                    'absolute_return': rankings['absolute_return'],
                    'risk_adjusted': risk_adjusted_rankings,
                    'sharpe_ratio': rankings['sharpe_ratio'],
                    'max_drawdown': rankings['max_drawdown']
                },
                'portfolio_analysis': portfolio_analysis,
                'market_condition_performance': market_condition_performance,
                'summary_statistics': self._calculate_summary_statistics(strategy_analyses),
                'recommendations': self._generate_portfolio_recommendations(strategy_analyses)
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing strategies: {e}")
            return {}
    
    def _calculate_performance_metrics(self, trades: pd.DataFrame, 
                                     equity_curve: pd.DataFrame,
                                     backtest_results: Dict[str, Any]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        try:
            initial_capital = backtest_results.get('initial_capital', 10000)
            final_capital = backtest_results.get('final_capital', initial_capital)
            
            # Basic return metrics
            total_return = (final_capital - initial_capital) / initial_capital
            
            # Calculate period duration
            duration_days = self._calculate_duration_days(backtest_results)
            duration_years = duration_days / 365.25
            
            annualized_return = (1 + total_return) ** (1 / duration_years) - 1 if duration_years > 0 else 0
            monthly_return = (1 + total_return) ** (12 / (duration_days / 30.44)) - 1 if duration_days > 0 else 0
            daily_return = total_return / duration_days if duration_days > 0 else 0
            
            # Risk metrics
            if not equity_curve.empty:
                equity_returns = equity_curve['equity'].pct_change().dropna()
                volatility = equity_returns.std() * np.sqrt(252)  # Annualized
                
                # Sharpe ratio
                excess_return = annualized_return - self.risk_free_rate
                sharpe_ratio = excess_return / volatility if volatility > 0 else 0
                
                # Sortino ratio
                downside_returns = equity_returns[equity_returns < 0]
                downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
                sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
                
                # Max drawdown
                max_drawdown = self._calculate_max_drawdown(equity_curve)
                
                # Calmar ratio
                calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
            else:
                volatility = 0
                sharpe_ratio = 0
                sortino_ratio = 0
                max_drawdown = 0
                calmar_ratio = 0
                downside_deviation = 0
            
            # Trade metrics
            if not trades.empty:
                winning_trades = trades[trades['net_pnl'] > 0]
                losing_trades = trades[trades['net_pnl'] <= 0]
                
                total_trades = len(trades)
                win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
                
                gross_profit = winning_trades['net_pnl'].sum() if not winning_trades.empty else 0
                gross_loss = abs(losing_trades['net_pnl'].sum()) if not losing_trades.empty else 0
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                
                avg_trade_return = trades['net_pnl'].mean() / initial_capital if initial_capital > 0 else 0
                avg_win = winning_trades['net_pnl'].mean() if not winning_trades.empty else 0
                avg_loss = losing_trades['net_pnl'].mean() if not losing_trades.empty else 0
                
                largest_win = trades['net_pnl'].max()
                largest_loss = trades['net_pnl'].min()
                
                # Consecutive wins/losses
                consecutive_wins = self._calculate_max_consecutive(trades, True)
                consecutive_losses = self._calculate_max_consecutive(trades, False)
            else:
                total_trades = 0
                win_rate = 0
                profit_factor = 0
                avg_trade_return = 0
                avg_win = 0
                avg_loss = 0
                largest_win = 0
                largest_loss = 0
                consecutive_wins = 0
                consecutive_losses = 0
            
            # Monthly analysis
            monthly_win_rate = self._calculate_monthly_win_rate(trades)
            
            # Recovery factor
            recovery_factor = total_return / max_drawdown if max_drawdown > 0 else 0
            
            # Cost analysis
            cost_analysis = backtest_results.get('cost_analysis', {})
            total_costs = cost_analysis.get('total_transaction_costs', 0)
            cost_ratio = total_costs / (final_capital - initial_capital) if (final_capital - initial_capital) > 0 else 0
            
            return PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                monthly_return=monthly_return,
                daily_return=daily_return,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=max_drawdown,
                volatility=volatility,
                downside_deviation=downside_deviation,
                total_trades=total_trades,
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_trade_return=avg_trade_return,
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                monthly_win_rate=monthly_win_rate,
                consecutive_wins=consecutive_wins,
                consecutive_losses=consecutive_losses,
                recovery_factor=recovery_factor,
                total_costs=total_costs,
                cost_ratio=cost_ratio
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def _analyze_market_phases(self, backtest_results: Dict[str, Any]) -> List[MarketPhaseAnalysis]:
        """Analyze performance by market phase"""
        try:
            phase_performance = backtest_results.get('market_phase_performance', {})
            market_phases_history = backtest_results.get('market_phases_history', [])
            
            analysis_results = []
            
            for phase, data in phase_performance.items():
                if data['trade_count'] > 0:
                    # Calculate phase duration
                    phase_periods = [p for p in market_phases_history if p['phase'] == phase]
                    duration_days = len(phase_periods)  # Simplified
                    
                    # Calculate Sharpe ratio for phase
                    if data['trade_count'] > 1:
                        # Estimate volatility (simplified)
                        trade_returns = [data['avg_pnl_per_trade'] / 10000]  # Simplified
                        volatility = np.std(trade_returns) if len(trade_returns) > 1 else 0.1
                        sharpe = (data['avg_pnl_per_trade'] - 0) / volatility if volatility > 0 else 0
                    else:
                        sharpe = 0
                    
                    analysis = MarketPhaseAnalysis(
                        phase=phase,
                        trade_count=data['trade_count'],
                        total_return=data['total_pnl'] / 10000,  # Assuming 10k initial capital
                        win_rate=data['win_rate'],
                        avg_trade_return=data['avg_pnl_per_trade'] / 10000,
                        sharpe_ratio=sharpe,
                        max_drawdown=0,  # Would need to calculate from phase equity curve
                        duration_days=duration_days,
                        best_trade=0,  # Would need individual trade data
                        worst_trade=0
                    )
                    
                    analysis_results.append(analysis)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing market phases: {e}")
            return []
    
    def _calculate_risk_metrics(self, trades: pd.DataFrame, equity_curve: pd.DataFrame) -> Dict[str, Any]:
        """Calculate detailed risk metrics"""
        try:
            risk_metrics = {}
            
            if not equity_curve.empty:
                equity_returns = equity_curve['equity'].pct_change().dropna()
                
                # Value at Risk (VaR)
                var_95 = np.percentile(equity_returns, 5)
                var_99 = np.percentile(equity_returns, 1)
                
                # Conditional VaR (Expected Shortfall)
                cvar_95 = equity_returns[equity_returns <= var_95].mean()
                cvar_99 = equity_returns[equity_returns <= var_99].mean()
                
                # Skewness and Kurtosis
                skewness = stats.skew(equity_returns)
                kurtosis = stats.kurtosis(equity_returns)
                
                # Ulcer Index
                ulcer_index = self._calculate_ulcer_index(equity_curve)
                
                # Information Ratio (vs benchmark)
                information_ratio = 0  # Would need benchmark data
                
                risk_metrics = {
                    'var_95': var_95,
                    'var_99': var_99,
                    'cvar_95': cvar_95,
                    'cvar_99': cvar_99,
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'ulcer_index': ulcer_index,
                    'information_ratio': information_ratio
                }
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _analyze_trades(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Detailed trade analysis"""
        try:
            if trades.empty:
                return {}
            
            # Trade duration analysis
            if 'entry_time' in trades.columns and 'exit_time' in trades.columns:
                trade_durations = (pd.to_datetime(trades['exit_time']) - pd.to_datetime(trades['entry_time'])).dt.total_seconds() / 3600
                avg_duration = trade_durations.mean()
                median_duration = trade_durations.median()
            else:
                avg_duration = 0
                median_duration = 0
            
            # Return distribution analysis
            returns = trades['net_pnl']
            
            # Trade size analysis
            trade_sizes = trades['size'] if 'size' in trades.columns else pd.Series([])
            
            # Timing analysis
            trades['hour'] = pd.to_datetime(trades['timestamp']).dt.hour
            trades['day_of_week'] = pd.to_datetime(trades['timestamp']).dt.dayofweek
            
            hourly_performance = trades.groupby('hour')['net_pnl'].agg(['mean', 'count']).to_dict()
            daily_performance = trades.groupby('day_of_week')['net_pnl'].agg(['mean', 'count']).to_dict()
            
            analysis = {
                'duration_analysis': {
                    'avg_duration_hours': avg_duration,
                    'median_duration_hours': median_duration
                },
                'return_distribution': {
                    'mean': returns.mean(),
                    'std': returns.std(),
                    'skewness': stats.skew(returns),
                    'kurtosis': stats.kurtosis(returns),
                    'percentiles': {
                        '5th': np.percentile(returns, 5),
                        '25th': np.percentile(returns, 25),
                        '50th': np.percentile(returns, 50),
                        '75th': np.percentile(returns, 75),
                        '95th': np.percentile(returns, 95)
                    }
                },
                'size_analysis': {
                    'avg_size': trade_sizes.mean() if not trade_sizes.empty else 0,
                    'size_std': trade_sizes.std() if not trade_sizes.empty else 0
                },
                'timing_analysis': {
                    'hourly_performance': hourly_performance,
                    'daily_performance': daily_performance
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing trades: {e}")
            return {}
    
    def _analyze_costs(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trading costs breakdown"""
        try:
            cost_analysis = backtest_results.get('cost_analysis', {})
            
            if not cost_analysis:
                return {}
            
            total_commission = cost_analysis.get('total_commission_paid', 0)
            total_slippage = cost_analysis.get('total_slippage_cost', 0)
            total_impact = cost_analysis.get('total_market_impact', 0)
            total_costs = cost_analysis.get('total_transaction_costs', 0)
            
            initial_capital = backtest_results.get('initial_capital', 10000)
            final_capital = backtest_results.get('final_capital', initial_capital)
            gross_return = final_capital - initial_capital
            
            analysis = {
                'cost_breakdown': {
                    'commission': total_commission,
                    'slippage': total_slippage,
                    'market_impact': total_impact,
                    'total': total_costs
                },
                'cost_ratios': {
                    'commission_ratio': total_commission / total_costs if total_costs > 0 else 0,
                    'slippage_ratio': total_slippage / total_costs if total_costs > 0 else 0,
                    'impact_ratio': total_impact / total_costs if total_costs > 0 else 0
                },
                'cost_impact': {
                    'cost_as_pct_of_capital': total_costs / initial_capital,
                    'cost_as_pct_of_return': total_costs / gross_return if gross_return > 0 else 0,
                    'net_vs_gross_return': gross_return - total_costs
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing costs: {e}")
            return {}
    
    def _analyze_time_patterns(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Analyze time-based patterns in trading performance"""
        try:
            if trades.empty:
                return {}
            
            trades['timestamp'] = pd.to_datetime(trades['timestamp'])
            trades['year'] = trades['timestamp'].dt.year
            trades['month'] = trades['timestamp'].dt.month
            trades['quarter'] = trades['timestamp'].dt.quarter
            
            # Monthly performance
            monthly_stats = trades.groupby(['year', 'month'])['net_pnl'].agg([
                'sum', 'mean', 'count', 'std'
            ]).reset_index()
            
            # Quarterly performance
            quarterly_stats = trades.groupby(['year', 'quarter'])['net_pnl'].agg([
                'sum', 'mean', 'count'
            ]).reset_index()
            
            # Yearly performance
            yearly_stats = trades.groupby('year')['net_pnl'].agg([
                'sum', 'mean', 'count'
            ]).reset_index()
            
            analysis = {
                'monthly_performance': monthly_stats.to_dict('records'),
                'quarterly_performance': quarterly_stats.to_dict('records'),
                'yearly_performance': yearly_stats.to_dict('records'),
                'performance_consistency': {
                    'monthly_win_rate': len(monthly_stats[monthly_stats['sum'] > 0]) / len(monthly_stats) if len(monthly_stats) > 0 else 0,
                    'quarterly_win_rate': len(quarterly_stats[quarterly_stats['sum'] > 0]) / len(quarterly_stats) if len(quarterly_stats) > 0 else 0
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing time patterns: {e}")
            return {}
    
    def _monte_carlo_analysis(self, trades: pd.DataFrame, iterations: int = 1000) -> Dict[str, Any]:
        """Perform Monte Carlo analysis on trade sequence"""
        try:
            if trades.empty or len(trades) < 10:
                return {}
            
            trade_returns = trades['net_pnl'].values
            n_trades = len(trade_returns)
            
            # Simulate random trade sequences
            final_returns = []
            max_drawdowns = []
            
            for _ in range(iterations):
                # Random sequence of actual trades
                random_sequence = np.random.choice(trade_returns, size=n_trades, replace=True)
                
                # Calculate cumulative return
                cumulative = np.cumsum(random_sequence)
                final_return = cumulative[-1]
                final_returns.append(final_return)
                
                # Calculate max drawdown for this sequence
                peak = np.maximum.accumulate(cumulative)
                drawdown = (peak - cumulative) / peak
                max_dd = np.max(drawdown)
                max_drawdowns.append(max_dd)
            
            # Calculate statistics
            actual_return = trade_returns.sum()
            actual_dd = 0  # Would need equity curve
            
            analysis = {
                'iterations': iterations,
                'return_statistics': {
                    'mean': np.mean(final_returns),
                    'std': np.std(final_returns),
                    'percentiles': {
                        '5th': np.percentile(final_returns, 5),
                        '25th': np.percentile(final_returns, 25),
                        '50th': np.percentile(final_returns, 50),
                        '75th': np.percentile(final_returns, 75),
                        '95th': np.percentile(final_returns, 95)
                    }
                },
                'drawdown_statistics': {
                    'mean': np.mean(max_drawdowns),
                    'std': np.std(max_drawdowns),
                    'percentiles': {
                        '5th': np.percentile(max_drawdowns, 5),
                        '25th': np.percentile(max_drawdowns, 25),
                        '50th': np.percentile(max_drawdowns, 50),
                        '75th': np.percentile(max_drawdowns, 75),
                        '95th': np.percentile(max_drawdowns, 95)
                    }
                },
                'confidence_intervals': {
                    'return_95_ci': [np.percentile(final_returns, 2.5), np.percentile(final_returns, 97.5)],
                    'drawdown_95_ci': [np.percentile(max_drawdowns, 2.5), np.percentile(max_drawdowns, 97.5)]
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo analysis: {e}")
            return {}
    
    def _generate_recommendations(self, performance_metrics: PerformanceMetrics, 
                                risk_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        try:
            # Performance-based recommendations
            if performance_metrics.sharpe_ratio < 1.0:
                recommendations.append("Consider improving risk-adjusted returns (Sharpe ratio < 1.0)")
            
            if performance_metrics.max_drawdown > 0.2:
                recommendations.append("High maximum drawdown (>20%) - consider better risk management")
            
            if performance_metrics.win_rate < 0.5:
                recommendations.append("Low win rate (<50%) - review entry/exit criteria")
            
            if performance_metrics.profit_factor < 1.5:
                recommendations.append("Low profit factor (<1.5) - optimize trade selection")
            
            # Cost-based recommendations
            if performance_metrics.cost_ratio > 0.3:
                recommendations.append("High transaction costs (>30% of returns) - optimize trade frequency")
            
            # Risk-based recommendations
            if risk_analysis.get('skewness', 0) < -1:
                recommendations.append("Negative skew in returns - review risk management for tail events")
            
            if risk_analysis.get('kurtosis', 0) > 3:
                recommendations.append("High kurtosis - strategy exposed to extreme events")
            
            # Trade frequency recommendations
            if performance_metrics.total_trades < 30:
                recommendations.append("Low trade count - results may not be statistically significant")
            
            if performance_metrics.consecutive_losses > 10:
                recommendations.append("High consecutive losses - consider drawdown controls")
            
            # If no issues found
            if not recommendations:
                recommendations.append("Strategy shows good overall performance metrics")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations = ["Error generating recommendations"]
        
        return recommendations
    
    # Helper methods
    
    def _calculate_duration_days(self, backtest_results: Dict[str, Any]) -> int:
        """Calculate backtest duration in days"""
        try:
            start_date = pd.to_datetime(backtest_results.get('start_date'))
            end_date = pd.to_datetime(backtest_results.get('end_date'))
            return (end_date - start_date).days
        except:
            return 365  # Default 1 year
    
    def _calculate_max_drawdown(self, equity_curve: pd.DataFrame) -> float:
        """Calculate maximum drawdown from equity curve"""
        try:
            if equity_curve.empty:
                return 0
            
            equity = equity_curve['equity']
            peak = equity.expanding().max()
            drawdown = (peak - equity) / peak
            return drawdown.max()
        except:
            return 0
    
    def _calculate_max_consecutive(self, trades: pd.DataFrame, wins: bool) -> int:
        """Calculate maximum consecutive wins or losses"""
        try:
            if trades.empty:
                return 0
            
            consecutive = 0
            max_consecutive = 0
            
            for _, trade in trades.iterrows():
                if (wins and trade['net_pnl'] > 0) or (not wins and trade['net_pnl'] <= 0):
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    consecutive = 0
            
            return max_consecutive
        except:
            return 0
    
    def _calculate_monthly_win_rate(self, trades: pd.DataFrame) -> float:
        """Calculate monthly win rate"""
        try:
            if trades.empty:
                return 0
            
            trades['timestamp'] = pd.to_datetime(trades['timestamp'])
            monthly_pnl = trades.groupby([trades['timestamp'].dt.year, trades['timestamp'].dt.month])['net_pnl'].sum()
            winning_months = len(monthly_pnl[monthly_pnl > 0])
            total_months = len(monthly_pnl)
            
            return winning_months / total_months if total_months > 0 else 0
        except:
            return 0
    
    def _calculate_ulcer_index(self, equity_curve: pd.DataFrame) -> float:
        """Calculate Ulcer Index"""
        try:
            if equity_curve.empty:
                return 0
            
            equity = equity_curve['equity']
            peak = equity.expanding().max()
            drawdown = (peak - equity) / peak * 100
            ulcer_index = np.sqrt(np.mean(drawdown ** 2))
            return ulcer_index
        except:
            return 0
    
    def _create_empty_analysis(self) -> Dict[str, Any]:
        """Create empty analysis structure"""
        return {
            'strategy_name': 'Unknown',
            'symbol': 'Unknown',
            'period': {'start': None, 'end': None, 'duration_days': 0},
            'performance_metrics': {},
            'market_phase_analysis': [],
            'risk_analysis': {},
            'trade_analysis': {},
            'cost_analysis': {},
            'time_analysis': {},
            'benchmark_analysis': {},
            'monte_carlo_analysis': {},
            'recommendations': ['No data available for analysis']
        }
    
    # Strategy comparison methods (simplified versions)
    
    def _calculate_strategy_correlations(self, strategy_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate correlation matrix between strategies"""
        # Implementation would calculate correlations between equity curves
        return {}
    
    def _rank_strategies(self, strategy_analyses: List[Dict[str, Any]]) -> Dict[str, List]:
        """Rank strategies by various metrics"""
        # Implementation would rank strategies
        return {
            'absolute_return': [],
            'sharpe_ratio': [],
            'max_drawdown': []
        }
    
    def _risk_adjusted_rankings(self, strategy_analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Risk-adjusted strategy rankings"""
        return []
    
    def _analyze_portfolio_combination(self, strategy_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze combining strategies in portfolio"""
        return {}
    
    def _analyze_market_condition_performance(self, strategy_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance under different market conditions"""
        return {}
    
    def _calculate_summary_statistics(self, strategy_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics across all strategies"""
        return {}
    
    def _generate_portfolio_recommendations(self, strategy_analyses: List[Dict[str, Any]]) -> List[str]:
        """Generate portfolio-level recommendations"""
        return []
    
    def _compare_to_benchmark(self, equity_curve: pd.DataFrame, benchmark_data: pd.DataFrame) -> Dict[str, Any]:
        """Compare strategy to benchmark"""
        return {}


# Factory function
def create_backtest_analyzer(config: Dict[str, Any] = None) -> BacktestAnalyzer:
    """Create and return BacktestAnalyzer instance"""
    return BacktestAnalyzer(config)