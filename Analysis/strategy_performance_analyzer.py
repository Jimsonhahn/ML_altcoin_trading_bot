# analysis/strategy_performance_analyzer.py
"""
Detailed Strategy Performance Analyzer
Calculates comprehensive metrics for each strategy across all scenarios
"""

import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
import statistics
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for a strategy"""
    # Return metrics
    total_return: float
    annualized_return: float
    monthly_return: float
    compound_annual_growth_rate: float
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    maximum_drawdown: float
    maximum_drawdown_duration: int
    volatility: float
    downside_deviation: float
    value_at_risk_95: float
    conditional_var_95: float
    
    # Trade metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    average_trade_duration: float
    
    # Consistency metrics
    monthly_win_rate: float
    quarterly_win_rate: float
    consecutive_wins: int
    consecutive_losses: int
    recovery_factor: float
    ulcer_index: float
    sterling_ratio: float
    
    # Cost metrics
    total_costs: float
    cost_ratio: float
    net_profit_margin: float
    
    # Market condition performance
    bull_market_return: float
    bear_market_return: float
    sideways_market_return: float
    
    # Advanced metrics
    information_ratio: float
    treynor_ratio: float
    jensen_alpha: float
    tracking_error: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'return_metrics': {
                'total_return': self.total_return,
                'annualized_return': self.annualized_return,
                'monthly_return': self.monthly_return,
                'cagr': self.compound_annual_growth_rate
            },
            'risk_metrics': {
                'sharpe_ratio': self.sharpe_ratio,
                'sortino_ratio': self.sortino_ratio,
                'calmar_ratio': self.calmar_ratio,
                'omega_ratio': self.omega_ratio,
                'max_drawdown': self.maximum_drawdown,
                'max_dd_duration': self.maximum_drawdown_duration,
                'volatility': self.volatility,
                'downside_deviation': self.downside_deviation,
                'var_95': self.value_at_risk_95,
                'cvar_95': self.conditional_var_95,
                'ulcer_index': self.ulcer_index
            },
            'trade_metrics': {
                'total_trades': self.total_trades,
                'win_rate': self.win_rate,
                'profit_factor': self.profit_factor,
                'avg_win': self.average_win,
                'avg_loss': self.average_loss,
                'largest_win': self.largest_win,
                'largest_loss': self.largest_loss,
                'avg_trade_duration': self.average_trade_duration
            },
            'consistency_metrics': {
                'monthly_win_rate': self.monthly_win_rate,
                'quarterly_win_rate': self.quarterly_win_rate,
                'consecutive_wins': self.consecutive_wins,
                'consecutive_losses': self.consecutive_losses,
                'recovery_factor': self.recovery_factor
            },
            'cost_metrics': {
                'total_costs': self.total_costs,
                'cost_ratio': self.cost_ratio,
                'net_profit_margin': self.net_profit_margin
            },
            'market_condition_performance': {
                'bull_market': self.bull_market_return,
                'bear_market': self.bear_market_return,
                'sideways_market': self.sideways_market_return
            }
        }


class StrategyPerformanceAnalyzer:
    """
    Advanced performance analyzer for detailed strategy metrics
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.output_dir = self.config.get('output_dir', 'analysis/performance')
        self.benchmark_return = self.config.get('benchmark_return', 0.08)  # 8% S&P 500
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)  # 2% risk-free
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("StrategyPerformanceAnalyzer initialized")
    
    def analyze_strategy_comprehensive(self, 
                                     strategy_name: str,
                                     all_backtest_results: Dict[str, Any],
                                     market_data: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a strategy across all scenarios
        """
        try:
            logger.info(f"Analyzing {strategy_name} comprehensively...")
            
            # Collect all results for this strategy
            strategy_results = all_backtest_results.get(strategy_name, {})
            
            if not strategy_results:
                logger.warning(f"No results found for {strategy_name}")
                return {}
            
            # Analyze each scenario
            scenario_analysis = {}
            
            # Analyze by market period
            period_analysis = self._analyze_by_market_periods(strategy_results)
            scenario_analysis['by_market_period'] = period_analysis
            
            # Analyze by capital size
            capital_analysis = self._analyze_by_capital_size(strategy_results)
            scenario_analysis['by_capital_size'] = capital_analysis
            
            # Analyze by risk profile
            risk_analysis = self._analyze_by_risk_profile(strategy_results)
            scenario_analysis['by_risk_profile'] = risk_analysis
            
            # Overall performance metrics
            overall_metrics = self._calculate_overall_metrics(strategy_results)
            
            # Trade pattern analysis
            trade_patterns = self._analyze_trade_patterns(strategy_results)
            
            # Risk analysis
            risk_analysis = self._perform_risk_analysis(strategy_results)
            
            # Cost analysis
            cost_analysis = self._analyze_costs(strategy_results)
            
            # Market regime performance
            regime_performance = self._analyze_market_regime_performance(strategy_results)
            
            # Seasonality analysis
            seasonality = self._analyze_seasonality(strategy_results)
            
            # Drawdown analysis
            drawdown_analysis = self._analyze_drawdowns(strategy_results)
            
            # Generate insights and warnings
            insights = self._generate_insights(strategy_name, overall_metrics, risk_analysis)
            warnings = self._generate_warnings(strategy_name, overall_metrics, risk_analysis)
            
            comprehensive_analysis = {
                'strategy_name': strategy_name,
                'analysis_timestamp': datetime.now().isoformat(),
                'overall_metrics': overall_metrics.to_dict() if overall_metrics else {},
                'scenario_analysis': scenario_analysis,
                'trade_patterns': trade_patterns,
                'risk_analysis': risk_analysis,
                'cost_analysis': cost_analysis,
                'regime_performance': regime_performance,
                'seasonality': seasonality,
                'drawdown_analysis': drawdown_analysis,
                'insights': insights,
                'warnings': warnings,
                'recommendations': self._generate_recommendations(strategy_name, overall_metrics)
            }
            
            # Save detailed analysis
            self._save_strategy_analysis(strategy_name, comprehensive_analysis)
            
            return comprehensive_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing {strategy_name}: {e}")
            return {}
    
    def _analyze_by_market_periods(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance by market periods"""
        try:
            period_analysis = {}
            
            periods_data = strategy_results.get('periods', {})
            
            for period_name, period_data in periods_data.items():
                # Aggregate across capital sizes and risk profiles
                all_period_metrics = []
                
                for capital_data in period_data.values():
                    for risk_data in capital_data.values():
                        if 'metrics' in risk_data:
                            all_period_metrics.append(risk_data['metrics'])
                
                if all_period_metrics:
                    period_analysis[period_name] = {
                        'avg_return': np.mean([m['total_return'] for m in all_period_metrics]),
                        'avg_sharpe': np.mean([m['sharpe_ratio'] for m in all_period_metrics]),
                        'avg_max_dd': np.mean([m['max_drawdown'] for m in all_period_metrics]),
                        'avg_win_rate': np.mean([m['win_rate'] for m in all_period_metrics]),
                        'consistency': np.std([m['total_return'] for m in all_period_metrics]),
                        'best_return': max(m['total_return'] for m in all_period_metrics),
                        'worst_return': min(m['total_return'] for m in all_period_metrics),
                        'num_scenarios': len(all_period_metrics)
                    }
            
            return period_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing by market periods: {e}")
            return {}
    
    def _analyze_by_capital_size(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance by capital size"""
        try:
            capital_analysis = {}
            
            # Collect metrics by capital size across all periods and risk profiles
            capital_metrics = {'Small': [], 'Medium': [], 'Large': []}
            
            for period_data in strategy_results.get('periods', {}).values():
                for capital_name, capital_data in period_data.items():
                    if capital_name in capital_metrics:
                        for risk_data in capital_data.values():
                            if 'metrics' in risk_data:
                                capital_metrics[capital_name].append(risk_data['metrics'])
            
            for capital_name, metrics_list in capital_metrics.items():
                if metrics_list:
                    capital_analysis[capital_name] = {
                        'avg_return': np.mean([m['total_return'] for m in metrics_list]),
                        'avg_sharpe': np.mean([m['sharpe_ratio'] for m in metrics_list]),
                        'return_stability': 1 / (1 + np.std([m['total_return'] for m in metrics_list])),
                        'risk_consistency': 1 / (1 + np.std([m['max_drawdown'] for m in metrics_list])),
                        'trade_frequency': np.mean([m.get('total_trades', 0) for m in metrics_list]),
                        'capital_efficiency': np.mean([m['total_return'] / max(1, m.get('total_trades', 1)) for m in metrics_list])
                    }
            
            return capital_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing by capital size: {e}")
            return {}
    
    def _analyze_by_risk_profile(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance by risk profile"""
        try:
            risk_analysis = {}
            
            # Collect metrics by risk profile
            risk_metrics = {'conservative': [], 'balanced': [], 'aggressive': []}
            
            for period_data in strategy_results.get('periods', {}).values():
                for capital_data in period_data.values():
                    for risk_name, risk_data in capital_data.items():
                        if risk_name in risk_metrics and 'metrics' in risk_data:
                            risk_metrics[risk_name].append(risk_data['metrics'])
            
            for risk_name, metrics_list in risk_metrics.items():
                if metrics_list:
                    returns = [m['total_return'] for m in metrics_list]
                    sharpes = [m['sharpe_ratio'] for m in metrics_list]
                    drawdowns = [m['max_drawdown'] for m in metrics_list]
                    
                    risk_analysis[risk_name] = {
                        'avg_return': np.mean(returns),
                        'return_volatility': np.std(returns),
                        'avg_sharpe': np.mean(sharpes),
                        'sharpe_consistency': 1 / (1 + np.std(sharpes)),
                        'avg_max_drawdown': np.mean(drawdowns),
                        'worst_drawdown': max(drawdowns),
                        'risk_reward_ratio': np.mean(returns) / np.mean(drawdowns) if np.mean(drawdowns) > 0 else 0,
                        'performance_consistency': 1 / (1 + np.std(returns) / np.mean(returns)) if np.mean(returns) > 0 else 0
                    }
            
            return risk_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing by risk profile: {e}")
            return {}
    
    def _calculate_overall_metrics(self, strategy_results: Dict[str, Any]) -> PerformanceMetrics:
        """Calculate overall performance metrics"""
        try:
            # Get the best representative result (Large capital, balanced risk, full period)
            representative_result = None
            
            try:
                full_period = strategy_results['periods']['Full Period']
                large_capital = full_period['Large']
                balanced_risk = large_capital['balanced']
                representative_result = balanced_risk
            except:
                # Fall back to any available result
                for period_data in strategy_results.get('periods', {}).values():
                    for capital_data in period_data.values():
                        for risk_data in capital_data.values():
                            if 'backtest' in risk_data and 'analysis' in risk_data:
                                representative_result = risk_data
                                break
                        if representative_result:
                            break
                    if representative_result:
                        break
            
            if not representative_result:
                logger.warning("No suitable result found for overall metrics")
                return None
            
            # Extract data
            backtest_data = representative_result.get('backtest', {})
            analysis_data = representative_result.get('analysis', {})
            
            trades_df = pd.DataFrame(backtest_data.get('trades', []))
            equity_curve_df = pd.DataFrame(backtest_data.get('equity_curve', []))
            
            if trades_df.empty or equity_curve_df.empty:
                logger.warning("Insufficient data for overall metrics")
                return None
            
            # Calculate comprehensive metrics
            return self._calculate_comprehensive_metrics(
                trades_df, equity_curve_df, backtest_data, analysis_data
            )
            
        except Exception as e:
            logger.error(f"Error calculating overall metrics: {e}")
            return None
    
    def _calculate_comprehensive_metrics(self, 
                                       trades_df: pd.DataFrame,
                                       equity_curve_df: pd.DataFrame,
                                       backtest_data: Dict[str, Any],
                                       analysis_data: Dict[str, Any]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics from data"""
        try:
            initial_capital = backtest_data.get('initial_capital', 10000)
            final_capital = backtest_data.get('final_capital', initial_capital)
            
            # Convert timestamps
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            equity_curve_df['timestamp'] = pd.to_datetime(equity_curve_df['timestamp'])
            equity_curve_df.set_index('timestamp', inplace=True)
            
            # Basic calculations
            total_return = (final_capital - initial_capital) / initial_capital
            duration_days = (equity_curve_df.index[-1] - equity_curve_df.index[0]).days
            duration_years = duration_days / 365.25
            
            # Return metrics
            annualized_return = (1 + total_return) ** (1 / duration_years) - 1 if duration_years > 0 else 0
            monthly_return = (1 + total_return) ** (12 / (duration_days / 30.44)) - 1 if duration_days > 0 else 0
            cagr = annualized_return
            
            # Risk metrics
            equity_returns = equity_curve_df['equity'].pct_change().dropna()
            volatility = equity_returns.std() * np.sqrt(252) if len(equity_returns) > 1 else 0
            
            # Sharpe ratio
            excess_return = annualized_return - self.risk_free_rate
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0
            
            # Sortino ratio
            downside_returns = equity_returns[equity_returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
            
            # Drawdown analysis
            peak = equity_curve_df['equity'].expanding().max()
            drawdown = (equity_curve_df['equity'] - peak) / peak
            max_drawdown = abs(drawdown.min())
            
            # Calmar ratio
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
            
            # Omega ratio
            omega_ratio = self._calculate_omega_ratio(equity_returns, 0)
            
            # VaR and CVaR
            var_95 = np.percentile(equity_returns, 5)
            cvar_95 = equity_returns[equity_returns <= var_95].mean() if len(equity_returns[equity_returns <= var_95]) > 0 else 0
            
            # Trade metrics
            winning_trades = trades_df[trades_df['net_pnl'] > 0]
            losing_trades = trades_df[trades_df['net_pnl'] <= 0]
            
            total_trades = len(trades_df)
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            
            gross_profit = winning_trades['net_pnl'].sum() if not winning_trades.empty else 0
            gross_loss = abs(losing_trades['net_pnl'].sum()) if not losing_trades.empty else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            avg_win = winning_trades['net_pnl'].mean() if not winning_trades.empty else 0
            avg_loss = losing_trades['net_pnl'].mean() if not losing_trades.empty else 0
            
            largest_win = trades_df['net_pnl'].max()
            largest_loss = trades_df['net_pnl'].min()
            
            # Trade duration
            if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
                entry_times = pd.to_datetime(trades_df['entry_time'])
                exit_times = pd.to_datetime(trades_df['exit_time'])
                durations = (exit_times - entry_times).dt.total_seconds() / 3600
                avg_trade_duration = durations.mean()
            else:
                avg_trade_duration = 0
            
            # Consistency metrics
            monthly_returns = self._calculate_monthly_returns(trades_df)
            monthly_win_rate = len(monthly_returns[monthly_returns > 0]) / len(monthly_returns) if len(monthly_returns) > 0 else 0
            
            quarterly_returns = self._calculate_quarterly_returns(trades_df)
            quarterly_win_rate = len(quarterly_returns[quarterly_returns > 0]) / len(quarterly_returns) if len(quarterly_returns) > 0 else 0
            
            consecutive_wins = self._calculate_max_consecutive(trades_df, True)
            consecutive_losses = self._calculate_max_consecutive(trades_df, False)
            
            recovery_factor = total_return / max_drawdown if max_drawdown > 0 else 0
            
            # Ulcer Index
            drawdown_squared = drawdown ** 2
            ulcer_index = np.sqrt(drawdown_squared.mean()) * 100
            
            # Sterling ratio
            sterling_ratio = annualized_return / ulcer_index if ulcer_index > 0 else 0
            
            # Cost metrics
            cost_analysis = backtest_data.get('cost_analysis', {})
            total_costs = cost_analysis.get('total_transaction_costs', 0)
            cost_ratio = total_costs / (final_capital - initial_capital) if (final_capital - initial_capital) > 0 else 0
            net_profit_margin = (final_capital - initial_capital - total_costs) / (final_capital - initial_capital) if (final_capital - initial_capital) > 0 else 0
            
            # Market condition performance (simplified)
            bull_market_return = total_return  # Would need to be calculated from specific periods
            bear_market_return = total_return * 0.7  # Simplified estimate
            sideways_market_return = total_return * 0.5  # Simplified estimate
            
            # Advanced metrics
            information_ratio = 0  # Would need benchmark data
            treynor_ratio = 0  # Would need beta calculation
            jensen_alpha = 0  # Would need benchmark data
            tracking_error = 0  # Would need benchmark data
            
            # Drawdown duration
            max_dd_duration = self._calculate_max_drawdown_duration(drawdown)
            
            return PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                monthly_return=monthly_return,
                compound_annual_growth_rate=cagr,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                omega_ratio=omega_ratio,
                maximum_drawdown=max_drawdown,
                maximum_drawdown_duration=max_dd_duration,
                volatility=volatility,
                downside_deviation=downside_deviation,
                value_at_risk_95=var_95,
                conditional_var_95=cvar_95,
                total_trades=total_trades,
                winning_trades=len(winning_trades),
                losing_trades=len(losing_trades),
                win_rate=win_rate,
                profit_factor=profit_factor,
                average_win=avg_win,
                average_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                average_trade_duration=avg_trade_duration,
                monthly_win_rate=monthly_win_rate,
                quarterly_win_rate=quarterly_win_rate,
                consecutive_wins=consecutive_wins,
                consecutive_losses=consecutive_losses,
                recovery_factor=recovery_factor,
                ulcer_index=ulcer_index,
                sterling_ratio=sterling_ratio,
                total_costs=total_costs,
                cost_ratio=cost_ratio,
                net_profit_margin=net_profit_margin,
                bull_market_return=bull_market_return,
                bear_market_return=bear_market_return,
                sideways_market_return=sideways_market_return,
                information_ratio=information_ratio,
                treynor_ratio=treynor_ratio,
                jensen_alpha=jensen_alpha,
                tracking_error=tracking_error
            )
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive metrics: {e}")
            return None
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0) -> float:
        """Calculate Omega ratio"""
        try:
            gains = returns[returns > threshold]
            losses = returns[returns <= threshold]
            
            if len(losses) == 0 or losses.sum() == 0:
                return float('inf')
            
            return gains.sum() / abs(losses.sum())
        except:
            return 0
    
    def _calculate_monthly_returns(self, trades_df: pd.DataFrame) -> pd.Series:
        """Calculate monthly returns"""
        try:
            trades_df['year_month'] = trades_df['timestamp'].dt.to_period('M')
            monthly_pnl = trades_df.groupby('year_month')['net_pnl'].sum()
            return monthly_pnl
        except:
            return pd.Series()
    
    def _calculate_quarterly_returns(self, trades_df: pd.DataFrame) -> pd.Series:
        """Calculate quarterly returns"""
        try:
            trades_df['year_quarter'] = trades_df['timestamp'].dt.to_period('Q')
            quarterly_pnl = trades_df.groupby('year_quarter')['net_pnl'].sum()
            return quarterly_pnl
        except:
            return pd.Series()
    
    def _calculate_max_consecutive(self, trades_df: pd.DataFrame, wins: bool) -> int:
        """Calculate maximum consecutive wins or losses"""
        try:
            trades_sorted = trades_df.sort_values('timestamp')
            consecutive = 0
            max_consecutive = 0
            
            for _, trade in trades_sorted.iterrows():
                if (wins and trade['net_pnl'] > 0) or (not wins and trade['net_pnl'] <= 0):
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    consecutive = 0
            
            return max_consecutive
        except:
            return 0
    
    def _calculate_max_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in days"""
        try:
            in_drawdown = drawdown < -0.01  # More than 1% drawdown
            max_duration = 0
            current_duration = 0
            
            for is_dd in in_drawdown:
                if is_dd:
                    current_duration += 1
                    max_duration = max(max_duration, current_duration)
                else:
                    current_duration = 0
            
            return max_duration
        except:
            return 0
    
    def _analyze_trade_patterns(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trade patterns and timing"""
        try:
            # Collect all trades from all scenarios
            all_trades = []
            
            for period_data in strategy_results.get('periods', {}).values():
                for capital_data in period_data.values():
                    for risk_data in capital_data.values():
                        backtest_data = risk_data.get('backtest', {})
                        trades = backtest_data.get('trades', [])
                        if trades:
                            all_trades.extend(trades)
            
            if not all_trades:
                return {}
            
            trades_df = pd.DataFrame(all_trades)
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            
            # Analyze patterns
            patterns = {
                'hourly_distribution': self._analyze_hourly_patterns(trades_df),
                'daily_distribution': self._analyze_daily_patterns(trades_df),
                'monthly_distribution': self._analyze_monthly_patterns(trades_df),
                'trade_size_distribution': self._analyze_trade_size_patterns(trades_df),
                'holding_time_analysis': self._analyze_holding_times(trades_df),
                'profit_distribution': self._analyze_profit_distribution(trades_df),
                'loss_distribution': self._analyze_loss_distribution(trades_df)
            }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing trade patterns: {e}")
            return {}
    
    def _analyze_hourly_patterns(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze hourly trading patterns"""
        try:
            trades_df['hour'] = trades_df['timestamp'].dt.hour
            hourly_stats = trades_df.groupby('hour').agg({
                'net_pnl': ['count', 'sum', 'mean'],
            }).round(4)
            
            hourly_stats.columns = ['count', 'total_pnl', 'avg_pnl']
            hourly_stats = hourly_stats.reset_index()
            
            best_hour = hourly_stats.loc[hourly_stats['avg_pnl'].idxmax()]
            worst_hour = hourly_stats.loc[hourly_stats['avg_pnl'].idxmin()]
            
            return {
                'hourly_stats': hourly_stats.to_dict('records'),
                'best_hour': {'hour': int(best_hour['hour']), 'avg_pnl': float(best_hour['avg_pnl'])},
                'worst_hour': {'hour': int(worst_hour['hour']), 'avg_pnl': float(worst_hour['avg_pnl'])},
                'most_active_hour': int(hourly_stats.loc[hourly_stats['count'].idxmax()]['hour'])
            }
        except:
            return {}
    
    def _analyze_daily_patterns(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze daily trading patterns"""
        try:
            trades_df['day_of_week'] = trades_df['timestamp'].dt.day_name()
            daily_stats = trades_df.groupby('day_of_week').agg({
                'net_pnl': ['count', 'sum', 'mean'],
            }).round(4)
            
            daily_stats.columns = ['count', 'total_pnl', 'avg_pnl']
            daily_stats = daily_stats.reset_index()
            
            best_day = daily_stats.loc[daily_stats['avg_pnl'].idxmax()]
            worst_day = daily_stats.loc[daily_stats['avg_pnl'].idxmin()]
            
            return {
                'daily_stats': daily_stats.to_dict('records'),
                'best_day': {'day': best_day['day_of_week'], 'avg_pnl': float(best_day['avg_pnl'])},
                'worst_day': {'day': worst_day['day_of_week'], 'avg_pnl': float(worst_day['avg_pnl'])},
                'most_active_day': daily_stats.loc[daily_stats['count'].idxmax()]['day_of_week']
            }
        except:
            return {}
    
    def _analyze_monthly_patterns(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze monthly trading patterns"""
        try:
            trades_df['month'] = trades_df['timestamp'].dt.month
            monthly_stats = trades_df.groupby('month').agg({
                'net_pnl': ['count', 'sum', 'mean'],
            }).round(4)
            
            monthly_stats.columns = ['count', 'total_pnl', 'avg_pnl']
            monthly_stats = monthly_stats.reset_index()
            
            best_month = monthly_stats.loc[monthly_stats['avg_pnl'].idxmax()]
            worst_month = monthly_stats.loc[monthly_stats['avg_pnl'].idxmin()]
            
            return {
                'monthly_stats': monthly_stats.to_dict('records'),
                'best_month': {'month': int(best_month['month']), 'avg_pnl': float(best_month['avg_pnl'])},
                'worst_month': {'month': int(worst_month['month']), 'avg_pnl': float(worst_month['avg_pnl'])},
                'most_active_month': int(monthly_stats.loc[monthly_stats['count'].idxmax()]['month'])
            }
        except:
            return {}
    
    def _analyze_trade_size_patterns(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trade size patterns"""
        try:
            if 'size' not in trades_df.columns:
                return {}
            
            size_stats = {
                'mean_size': float(trades_df['size'].mean()),
                'median_size': float(trades_df['size'].median()),
                'std_size': float(trades_df['size'].std()),
                'min_size': float(trades_df['size'].min()),
                'max_size': float(trades_df['size'].max())
            }
            
            # Correlation between size and profit
            size_profit_corr = trades_df['size'].corr(trades_df['net_pnl'])
            
            return {
                'size_statistics': size_stats,
                'size_profit_correlation': float(size_profit_corr) if not np.isnan(size_profit_corr) else 0
            }
        except:
            return {}
    
    def _analyze_holding_times(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trade holding times"""
        try:
            if 'entry_time' not in trades_df.columns or 'exit_time' not in trades_df.columns:
                return {}
            
            entry_times = pd.to_datetime(trades_df['entry_time'])
            exit_times = pd.to_datetime(trades_df['exit_time'])
            holding_times = (exit_times - entry_times).dt.total_seconds() / 3600  # Hours
            
            return {
                'avg_holding_time_hours': float(holding_times.mean()),
                'median_holding_time_hours': float(holding_times.median()),
                'min_holding_time_hours': float(holding_times.min()),
                'max_holding_time_hours': float(holding_times.max()),
                'std_holding_time_hours': float(holding_times.std())
            }
        except:
            return {}
    
    def _analyze_profit_distribution(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze profit distribution"""
        try:
            winning_trades = trades_df[trades_df['net_pnl'] > 0]['net_pnl']
            
            if winning_trades.empty:
                return {}
            
            return {
                'count': len(winning_trades),
                'mean': float(winning_trades.mean()),
                'median': float(winning_trades.median()),
                'std': float(winning_trades.std()),
                'min': float(winning_trades.min()),
                'max': float(winning_trades.max()),
                'percentiles': {
                    '25th': float(winning_trades.quantile(0.25)),
                    '75th': float(winning_trades.quantile(0.75)),
                    '95th': float(winning_trades.quantile(0.95))
                }
            }
        except:
            return {}
    
    def _analyze_loss_distribution(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze loss distribution"""
        try:
            losing_trades = trades_df[trades_df['net_pnl'] <= 0]['net_pnl']
            
            if losing_trades.empty:
                return {}
            
            return {
                'count': len(losing_trades),
                'mean': float(losing_trades.mean()),
                'median': float(losing_trades.median()),
                'std': float(losing_trades.std()),
                'min': float(losing_trades.min()),
                'max': float(losing_trades.max()),
                'percentiles': {
                    '5th': float(losing_trades.quantile(0.05)),
                    '25th': float(losing_trades.quantile(0.25)),
                    '75th': float(losing_trades.quantile(0.75))
                }
            }
        except:
            return {}
    
    def _perform_risk_analysis(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform detailed risk analysis"""
        try:
            # Collect all equity curves
            all_equity_curves = []
            
            for period_data in strategy_results.get('periods', {}).values():
                for capital_data in period_data.values():
                    for risk_data in capital_data.values():
                        backtest_data = risk_data.get('backtest', {})
                        equity_curve = backtest_data.get('equity_curve', [])
                        if equity_curve:
                            all_equity_curves.append(pd.DataFrame(equity_curve))
            
            if not all_equity_curves:
                return {}
            
            # Combine all equity curves for risk analysis
            combined_analysis = {}
            
            for i, equity_df in enumerate(all_equity_curves):
                if not equity_df.empty:
                    equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
                    equity_df.set_index('timestamp', inplace=True)
                    
                    returns = equity_df['equity'].pct_change().dropna()
                    
                    if len(returns) > 0:
                        # VaR analysis
                        var_1 = float(np.percentile(returns, 1))
                        var_5 = float(np.percentile(returns, 5))
                        var_10 = float(np.percentile(returns, 10))
                        
                        # CVaR analysis
                        cvar_1 = float(returns[returns <= var_1].mean()) if len(returns[returns <= var_1]) > 0 else 0
                        cvar_5 = float(returns[returns <= var_5].mean()) if len(returns[returns <= var_5]) > 0 else 0
                        
                        # Tail risk
                        skewness = float(stats.skew(returns))
                        kurtosis = float(stats.kurtosis(returns))
                        
                        # Volatility clustering (simplified)
                        volatility_clustering = float(returns.rolling(30).std().std()) if len(returns) > 30 else 0
                        
                        scenario_risk = {
                            'var_1': var_1,
                            'var_5': var_5,
                            'var_10': var_10,
                            'cvar_1': cvar_1,
                            'cvar_5': cvar_5,
                            'skewness': skewness,
                            'kurtosis': kurtosis,
                            'volatility_clustering': volatility_clustering
                        }
                        
                        combined_analysis[f'scenario_{i}'] = scenario_risk
            
            # Aggregate risk metrics
            if combined_analysis:
                aggregated_risk = {}
                for metric in ['var_5', 'cvar_5', 'skewness', 'kurtosis']:
                    values = [scenario[metric] for scenario in combined_analysis.values() if metric in scenario]
                    if values:
                        aggregated_risk[f'{metric}_mean'] = float(np.mean(values))
                        aggregated_risk[f'{metric}_std'] = float(np.std(values))
                        aggregated_risk[f'{metric}_worst'] = float(min(values)) if metric in ['var_5', 'cvar_5'] else float(max(values))
                
                return {
                    'scenario_analysis': combined_analysis,
                    'aggregated_metrics': aggregated_risk,
                    'risk_grade': self._calculate_risk_grade(aggregated_risk)
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error in risk analysis: {e}")
            return {}
    
    def _calculate_risk_grade(self, risk_metrics: Dict[str, Any]) -> str:
        """Calculate risk grade based on metrics"""
        try:
            var_5_mean = abs(risk_metrics.get('var_5_mean', 0))
            kurtosis_mean = risk_metrics.get('kurtosis_mean', 0)
            
            if var_5_mean < 0.02 and kurtosis_mean < 3:
                return 'A'  # Low risk
            elif var_5_mean < 0.035 and kurtosis_mean < 5:
                return 'B'  # Moderate risk
            elif var_5_mean < 0.05:
                return 'C'  # High risk
            else:
                return 'D'  # Very high risk
        except:
            return 'Unknown'
    
    def _analyze_costs(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trading costs across all scenarios"""
        try:
            all_cost_data = []
            
            for period_data in strategy_results.get('periods', {}).values():
                for capital_data in period_data.values():
                    for risk_data in capital_data.values():
                        backtest_data = risk_data.get('backtest', {})
                        cost_analysis = backtest_data.get('cost_analysis', {})
                        if cost_analysis:
                            all_cost_data.append(cost_analysis)
            
            if not all_cost_data:
                return {}
            
            # Aggregate cost metrics
            total_commission = sum(data.get('total_commission_paid', 0) for data in all_cost_data)
            total_slippage = sum(data.get('total_slippage_cost', 0) for data in all_cost_data)
            total_market_impact = sum(data.get('total_market_impact', 0) for data in all_cost_data)
            total_costs = sum(data.get('total_transaction_costs', 0) for data in all_cost_data)
            
            # Calculate ratios
            cost_breakdown = {
                'total_commission': total_commission,
                'total_slippage': total_slippage,
                'total_market_impact': total_market_impact,
                'total_costs': total_costs,
                'commission_ratio': total_commission / total_costs if total_costs > 0 else 0,
                'slippage_ratio': total_slippage / total_costs if total_costs > 0 else 0,
                'market_impact_ratio': total_market_impact / total_costs if total_costs > 0 else 0
            }
            
            # Cost efficiency metrics
            avg_cost_per_scenario = total_costs / len(all_cost_data)
            cost_consistency = np.std([data.get('total_transaction_costs', 0) for data in all_cost_data])
            
            return {
                'cost_breakdown': cost_breakdown,
                'avg_cost_per_scenario': avg_cost_per_scenario,
                'cost_consistency': cost_consistency,
                'cost_efficiency_grade': self._calculate_cost_grade(cost_breakdown)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing costs: {e}")
            return {}
    
    def _calculate_cost_grade(self, cost_breakdown: Dict[str, Any]) -> str:
        """Calculate cost efficiency grade"""
        try:
            total_costs = cost_breakdown.get('total_costs', 0)
            
            if total_costs < 100:
                return 'A'  # Very efficient
            elif total_costs < 500:
                return 'B'  # Efficient
            elif total_costs < 1000:
                return 'C'  # Moderate
            else:
                return 'D'  # Expensive
        except:
            return 'Unknown'
    
    def _analyze_market_regime_performance(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance in different market regimes"""
        try:
            regime_performance = {}
            
            # Map periods to market regimes
            regime_mapping = {
                'Bull Market': 'bull',
                'Bear Market': 'bear',
                'Recovery': 'recovery',
                'Full Period': 'mixed'
            }
            
            for period_name, period_data in strategy_results.get('periods', {}).items():
                regime = regime_mapping.get(period_name, 'unknown')
                
                # Aggregate performance across capital sizes and risk profiles
                period_metrics = []
                for capital_data in period_data.values():
                    for risk_data in capital_data.values():
                        if 'metrics' in risk_data:
                            period_metrics.append(risk_data['metrics'])
                
                if period_metrics:
                    regime_performance[regime] = {
                        'avg_return': np.mean([m['total_return'] for m in period_metrics]),
                        'avg_sharpe': np.mean([m['sharpe_ratio'] for m in period_metrics]),
                        'avg_max_dd': np.mean([m['max_drawdown'] for m in period_metrics]),
                        'consistency': 1 / (1 + np.std([m['total_return'] for m in period_metrics])),
                        'best_scenario_return': max(m['total_return'] for m in period_metrics),
                        'worst_scenario_return': min(m['total_return'] for m in period_metrics)
                    }
            
            return regime_performance
            
        except Exception as e:
            logger.error(f"Error analyzing market regime performance: {e}")
            return {}
    
    def _analyze_seasonality(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze seasonal patterns in performance"""
        try:
            # This would require more detailed trade timing analysis
            # For now, return placeholder
            return {
                'quarterly_patterns': {},
                'monthly_patterns': {},
                'day_of_week_patterns': {},
                'hour_of_day_patterns': {}
            }
            
        except Exception as e:
            logger.error(f"Error analyzing seasonality: {e}")
            return {}
    
    def _analyze_drawdowns(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze drawdown characteristics"""
        try:
            all_drawdown_data = []
            
            for period_data in strategy_results.get('periods', {}).values():
                for capital_data in period_data.values():
                    for risk_data in capital_data.values():
                        backtest_data = risk_data.get('backtest', {})
                        equity_curve = backtest_data.get('equity_curve', [])
                        
                        if equity_curve:
                            equity_df = pd.DataFrame(equity_curve)
                            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
                            equity_df.set_index('timestamp', inplace=True)
                            
                            # Calculate drawdown
                            peak = equity_df['equity'].expanding().max()
                            drawdown = (equity_df['equity'] - peak) / peak
                            
                            max_dd = abs(drawdown.min())
                            dd_duration = self._calculate_max_drawdown_duration(drawdown)
                            
                            all_drawdown_data.append({
                                'max_drawdown': max_dd,
                                'duration_days': dd_duration,
                                'recovery_time': dd_duration  # Simplified
                            })
            
            if not all_drawdown_data:
                return {}
            
            # Aggregate drawdown statistics
            max_drawdowns = [dd['max_drawdown'] for dd in all_drawdown_data]
            durations = [dd['duration_days'] for dd in all_drawdown_data]
            
            return {
                'avg_max_drawdown': float(np.mean(max_drawdowns)),
                'worst_drawdown': float(max(max_drawdowns)),
                'drawdown_consistency': float(1 / (1 + np.std(max_drawdowns))),
                'avg_duration_days': float(np.mean(durations)),
                'longest_drawdown_days': float(max(durations)),
                'drawdown_frequency': len([dd for dd in max_drawdowns if dd > 0.05]),  # >5% drawdowns
                'severe_drawdown_count': len([dd for dd in max_drawdowns if dd > 0.20])  # >20% drawdowns
            }
            
        except Exception as e:
            logger.error(f"Error analyzing drawdowns: {e}")
            return {}
    
    def _generate_insights(self, strategy_name: str, metrics: PerformanceMetrics, 
                          risk_analysis: Dict[str, Any]) -> List[str]:
        """Generate performance insights"""
        insights = []
        
        try:
            if metrics:
                # Return insights
                if metrics.annualized_return > 0.20:
                    insights.append(f"Strong annual returns of {metrics.annualized_return*100:.1f}%")
                elif metrics.annualized_return > 0.10:
                    insights.append(f"Solid annual returns of {metrics.annualized_return*100:.1f}%")
                
                # Risk insights
                if metrics.sharpe_ratio > 1.5:
                    insights.append(f"Excellent risk-adjusted returns (Sharpe: {metrics.sharpe_ratio:.2f})")
                elif metrics.sharpe_ratio > 1.0:
                    insights.append(f"Good risk-adjusted returns (Sharpe: {metrics.sharpe_ratio:.2f})")
                
                # Drawdown insights
                if metrics.maximum_drawdown < 0.10:
                    insights.append(f"Low risk with max drawdown of {metrics.maximum_drawdown*100:.1f}%")
                elif metrics.maximum_drawdown > 0.25:
                    insights.append(f"High risk - max drawdown of {metrics.maximum_drawdown*100:.1f}%")
                
                # Consistency insights
                if metrics.win_rate > 0.60:
                    insights.append(f"High win rate of {metrics.win_rate*100:.1f}%")
                elif metrics.win_rate < 0.40:
                    insights.append(f"Low win rate of {metrics.win_rate*100:.1f}% - focus on trade quality")
                
                # Trading frequency insights
                if metrics.total_trades > 1000:
                    insights.append("High-frequency strategy with many trades")
                elif metrics.total_trades < 50:
                    insights.append("Low-frequency strategy - fewer but potentially larger trades")
            
            # Risk insights
            risk_grade = risk_analysis.get('risk_grade', 'Unknown')
            if risk_grade == 'A':
                insights.append("Low tail risk - well-behaved return distribution")
            elif risk_grade == 'D':
                insights.append("High tail risk - potential for extreme losses")
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
        
        return insights
    
    def _generate_warnings(self, strategy_name: str, metrics: PerformanceMetrics,
                          risk_analysis: Dict[str, Any]) -> List[str]:
        """Generate performance warnings"""
        warnings = []
        
        try:
            if metrics:
                # Return warnings
                if metrics.annualized_return < 0:
                    warnings.append("Strategy shows negative returns")
                
                # Risk warnings
                if metrics.maximum_drawdown > 0.30:
                    warnings.append(f"Very high drawdown risk ({metrics.maximum_drawdown*100:.1f}%)")
                
                if metrics.sharpe_ratio < 0.5:
                    warnings.append("Poor risk-adjusted returns - consider improvement")
                
                # Consistency warnings
                if metrics.consecutive_losses > 15:
                    warnings.append(f"Long losing streaks possible ({metrics.consecutive_losses} consecutive losses)")
                
                if metrics.win_rate < 0.35:
                    warnings.append("Very low win rate - high psychological pressure")
                
                # Trading warnings
                if metrics.total_trades < 20:
                    warnings.append("Low sample size - results may not be statistically significant")
                
                if metrics.largest_loss < -1000:
                    warnings.append(f"Large single loss possible (${abs(metrics.largest_loss):,.0f})")
            
            # Risk warnings
            aggregated_risk = risk_analysis.get('aggregated_metrics', {})
            if aggregated_risk.get('kurtosis_mean', 0) > 5:
                warnings.append("High kurtosis - fat tail risk present")
            
            if aggregated_risk.get('var_5_worst', 0) < -0.05:
                warnings.append("Potential for severe daily losses (>5%)")
            
        except Exception as e:
            logger.error(f"Error generating warnings: {e}")
        
        return warnings
    
    def _generate_recommendations(self, strategy_name: str, metrics: PerformanceMetrics) -> List[str]:
        """Generate strategy-specific recommendations"""
        recommendations = []
        
        try:
            if metrics:
                # Position sizing recommendations
                if metrics.maximum_drawdown > 0.20:
                    recommendations.append("Reduce position sizes to limit drawdown risk")
                
                if metrics.sharpe_ratio < 1.0:
                    recommendations.append("Optimize entry/exit criteria to improve risk-adjusted returns")
                
                # Risk management recommendations
                if metrics.consecutive_losses > 10:
                    recommendations.append("Implement circuit breakers after consecutive losses")
                
                if metrics.win_rate < 0.45:
                    recommendations.append("Improve signal quality and trade selection")
                
                # Strategy-specific recommendations
                if strategy_name == "momentum_strategy":
                    if metrics.win_rate < 0.50:
                        recommendations.append("Consider adding trend confirmation indicators")
                elif strategy_name == "mean_reversion_strategy":
                    if metrics.maximum_drawdown > 0.25:
                        recommendations.append("Implement time-based exits to avoid extended drawdowns")
                elif strategy_name == "ml_strategy":
                    if metrics.sharpe_ratio < 1.0:
                        recommendations.append("Retrain models more frequently and review feature engineering")
                
                # General recommendations
                if metrics.profit_factor < 1.5:
                    recommendations.append("Optimize take-profit levels to improve profit factor")
                
                if metrics.cost_ratio > 0.30:
                    recommendations.append("Reduce trading frequency to minimize transaction costs")
        
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def _save_strategy_analysis(self, strategy_name: str, analysis: Dict[str, Any]):
        """Save detailed strategy analysis to file"""
        try:
            filename = f"{strategy_name}_comprehensive_analysis.json"
            filepath = Path(self.output_dir) / filename
            
            with open(filepath, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            logger.info(f"Saved comprehensive analysis for {strategy_name} to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving strategy analysis: {e}")


# Factory function
def create_strategy_performance_analyzer(config: Dict[str, Any] = None) -> StrategyPerformanceAnalyzer:
    """Create and return StrategyPerformanceAnalyzer instance"""
    return StrategyPerformanceAnalyzer(config)