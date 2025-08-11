# utils/backtest_visualizer.py
"""
Advanced Backtest Visualization Suite
Creates comprehensive charts and reports for backtest analysis
"""

import logging
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import plotly for interactive charts
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)


class BacktestVisualizer:
    """
    Advanced visualization suite for backtest results
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Visualization settings
        self.output_dir = self.config.get('output_dir', 'data/visualizations')
        self.dpi = self.config.get('dpi', 300)
        self.style = self.config.get('style', 'seaborn-v0_8')
        self.color_palette = self.config.get('color_palette', 'Set2')
        self.figsize_large = self.config.get('figsize_large', (16, 12))
        self.figsize_medium = self.config.get('figsize_medium', (12, 8))
        self.figsize_small = self.config.get('figsize_small', (10, 6))
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set style
        try:
            plt.style.use(self.style)
        except:
            plt.style.use('default')
        
        # Color schemes
        self.colors = {
            'equity': '#2E86AB',
            'benchmark': '#A23B72',
            'drawdown': '#F18F01',
            'trades_win': '#C73E1D',
            'trades_loss': '#592E83',
            'volume': '#5E7C7C',
            'rsi': '#F79D84',
            'macd': '#86C5D8'
        }
        
        logger.info("BacktestVisualizer initialized")
    
    def create_comprehensive_report(self, 
                                  backtest_results: Dict[str, Any],
                                  analysis_results: Dict[str, Any] = None,
                                  market_data: pd.DataFrame = None,
                                  save_path: str = None) -> str:
        """
        Create comprehensive visualization report
        """
        try:
            logger.info("Creating comprehensive backtest visualization report")
            
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = os.path.join(self.output_dir, f"backtest_report_{timestamp}")
            
            # Create directory
            Path(save_path).mkdir(parents=True, exist_ok=True)
            
            # Extract data
            trades = pd.DataFrame(backtest_results.get('trades', []))
            equity_curve = pd.DataFrame(backtest_results.get('equity_curve', []))
            
            if trades.empty or equity_curve.empty:
                logger.warning("No data available for visualization")
                return save_path
            
            # Convert timestamps
            if not trades.empty:
                trades['timestamp'] = pd.to_datetime(trades['timestamp'])
            equity_curve['timestamp'] = pd.to_datetime(equity_curve['timestamp'])
            equity_curve.set_index('timestamp', inplace=True)
            
            # Generate individual visualizations
            chart_paths = []
            
            # 1. Equity curve with trades
            chart_path = self.plot_equity_curve_with_trades(
                equity_curve, trades, os.path.join(save_path, "01_equity_curve.png")
            )
            chart_paths.append(chart_path)
            
            # 2. Drawdown analysis
            chart_path = self.plot_drawdown_analysis(
                equity_curve, os.path.join(save_path, "02_drawdown_analysis.png")
            )
            chart_paths.append(chart_path)
            
            # 3. Trade distribution
            chart_path = self.plot_trade_distribution(
                trades, os.path.join(save_path, "03_trade_distribution.png")
            )
            chart_paths.append(chart_path)
            
            # 4. Performance metrics dashboard
            if analysis_results:
                chart_path = self.plot_performance_dashboard(
                    analysis_results, os.path.join(save_path, "04_performance_dashboard.png")
                )
                chart_paths.append(chart_path)
            
            # 5. Time-based analysis
            chart_path = self.plot_time_analysis(
                trades, equity_curve, os.path.join(save_path, "05_time_analysis.png")
            )
            chart_paths.append(chart_path)
            
            # 6. Risk analysis
            chart_path = self.plot_risk_analysis(
                equity_curve, trades, os.path.join(save_path, "06_risk_analysis.png")
            )
            chart_paths.append(chart_path)
            
            # 7. Market conditions overlay
            if market_data is not None:
                chart_path = self.plot_market_conditions(
                    equity_curve, market_data, trades, os.path.join(save_path, "07_market_conditions.png")
                )
                chart_paths.append(chart_path)
            
            # 8. Trade timing analysis
            chart_path = self.plot_trade_timing(
                trades, os.path.join(save_path, "08_trade_timing.png")
            )
            chart_paths.append(chart_path)
            
            # 9. Rolling performance
            chart_path = self.plot_rolling_performance(
                equity_curve, os.path.join(save_path, "09_rolling_performance.png")
            )
            chart_paths.append(chart_path)
            
            # 10. Cost analysis
            if 'cost_analysis' in backtest_results:
                chart_path = self.plot_cost_analysis(
                    backtest_results['cost_analysis'], trades, os.path.join(save_path, "10_cost_analysis.png")
                )
                chart_paths.append(chart_path)
            
            # Create interactive charts if plotly available
            if PLOTLY_AVAILABLE:
                self.create_interactive_dashboard(
                    backtest_results, analysis_results, market_data, 
                    os.path.join(save_path, "interactive_dashboard.html")
                )
            
            logger.info(f"Comprehensive report created at {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Error creating comprehensive report: {e}")
            return ""
    
    def plot_equity_curve_with_trades(self, equity_curve: pd.DataFrame, 
                                    trades: pd.DataFrame, save_path: str = None) -> str:
        """Plot equity curve with trade markers"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize_large, 
                                         gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot equity curve
            ax1.plot(equity_curve.index, equity_curve['equity'], 
                    color=self.colors['equity'], linewidth=2, label='Portfolio Value')
            
            # Add trade markers
            if not trades.empty:
                buy_trades = trades[trades['side'] == 'buy']
                sell_trades = trades[trades['side'] == 'sell']
                
                if not buy_trades.empty:
                    ax1.scatter(buy_trades['timestamp'], buy_trades['price'], 
                              color='green', marker='^', s=50, alpha=0.7, label='Buy', zorder=5)
                
                if not sell_trades.empty:
                    ax1.scatter(sell_trades['timestamp'], sell_trades['price'], 
                              color='red', marker='v', s=50, alpha=0.7, label='Sell', zorder=5)
            
            # Calculate and plot drawdown
            peak = equity_curve['equity'].expanding().max()
            drawdown = (equity_curve['equity'] - peak) / peak
            
            ax2.fill_between(equity_curve.index, drawdown, 0, 
                           color=self.colors['drawdown'], alpha=0.7, label='Drawdown')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # Formatting
            ax1.set_title('Portfolio Equity Curve with Trade Signals', fontsize=16, fontweight='bold')
            ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            ax2.set_title('Drawdown', fontsize=14)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Drawdown (%)', fontsize=12)
            ax2.legend(loc='lower right')
            ax2.grid(True, alpha=0.3)
            
            # Format x-axis
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                return save_path
            else:
                plt.show()
                return ""
                
        except Exception as e:
            logger.error(f"Error plotting equity curve: {e}")
            plt.close()
            return ""
    
    def plot_drawdown_analysis(self, equity_curve: pd.DataFrame, save_path: str = None) -> str:
        """Plot detailed drawdown analysis"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=self.figsize_large)
            
            # Calculate drawdown metrics
            peak = equity_curve['equity'].expanding().max()
            drawdown = (equity_curve['equity'] - peak) / peak
            
            # 1. Drawdown over time
            axes[0, 0].fill_between(equity_curve.index, drawdown * 100, 0, 
                                  color=self.colors['drawdown'], alpha=0.7)
            axes[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[0, 0].set_title('Drawdown Over Time', fontweight='bold')
            axes[0, 0].set_ylabel('Drawdown (%)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Drawdown distribution
            drawdown_pct = drawdown * 100
            axes[0, 1].hist(drawdown_pct[drawdown_pct < 0], bins=30, 
                          color=self.colors['drawdown'], alpha=0.7, edgecolor='black')
            axes[0, 1].axvline(drawdown_pct.min(), color='red', linestyle='--', 
                             label=f'Max DD: {drawdown_pct.min():.2f}%')
            axes[0, 1].set_title('Drawdown Distribution', fontweight='bold')
            axes[0, 1].set_xlabel('Drawdown (%)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Underwater curve
            axes[1, 0].fill_between(equity_curve.index, drawdown * 100, 0, 
                                  color='lightcoral', alpha=0.7)
            axes[1, 0].set_title('Underwater Curve', fontweight='bold')
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('Drawdown (%)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Recovery periods
            # Find drawdown periods
            in_drawdown = drawdown < -0.01  # More than 1% drawdown
            drawdown_periods = []
            start_dd = None
            
            for i, (date, is_dd) in enumerate(zip(equity_curve.index, in_drawdown)):
                if is_dd and start_dd is None:
                    start_dd = date
                elif not is_dd and start_dd is not None:
                    drawdown_periods.append((start_dd, date, (date - start_dd).days))
                    start_dd = None
            
            if drawdown_periods:
                recovery_days = [period[2] for period in drawdown_periods]
                axes[1, 1].hist(recovery_days, bins=max(1, len(recovery_days)//2), 
                              color='skyblue', alpha=0.7, edgecolor='black')
                axes[1, 1].axvline(np.mean(recovery_days), color='red', linestyle='--', 
                                 label=f'Avg: {np.mean(recovery_days):.1f} days')
                axes[1, 1].set_title('Recovery Period Distribution', fontweight='bold')
                axes[1, 1].set_xlabel('Recovery Days')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'No significant\ndrawdown periods', 
                              ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Recovery Period Distribution', fontweight='bold')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                return save_path
            else:
                plt.show()
                return ""
                
        except Exception as e:
            logger.error(f"Error plotting drawdown analysis: {e}")
            plt.close()
            return ""
    
    def plot_trade_distribution(self, trades: pd.DataFrame, save_path: str = None) -> str:
        """Plot trade distribution analysis"""
        try:
            if trades.empty:
                logger.warning("No trades to plot")
                return ""
            
            fig, axes = plt.subplots(2, 3, figsize=self.figsize_large)
            
            # 1. P&L Distribution
            pnl = trades['net_pnl']
            axes[0, 0].hist(pnl, bins=30, color='lightblue', alpha=0.7, edgecolor='black')
            axes[0, 0].axvline(pnl.mean(), color='red', linestyle='--', 
                             label=f'Mean: ${pnl.mean():.2f}')
            axes[0, 0].axvline(0, color='black', linestyle='-', alpha=0.5)
            axes[0, 0].set_title('P&L Distribution', fontweight='bold')
            axes[0, 0].set_xlabel('P&L ($)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Win/Loss Ratio
            wins = trades[trades['net_pnl'] > 0]
            losses = trades[trades['net_pnl'] <= 0]
            
            win_loss_data = [len(wins), len(losses)]
            labels = [f'Wins ({len(wins)})', f'Losses ({len(losses)})']
            colors = ['lightgreen', 'lightcoral']
            
            axes[0, 1].pie(win_loss_data, labels=labels, colors=colors, autopct='%1.1f%%')
            axes[0, 1].set_title('Win/Loss Ratio', fontweight='bold')
            
            # 3. Trade Size Distribution
            if 'size' in trades.columns:
                axes[0, 2].hist(trades['size'], bins=20, color='lightyellow', 
                              alpha=0.7, edgecolor='black')
                axes[0, 2].set_title('Trade Size Distribution', fontweight='bold')
                axes[0, 2].set_xlabel('Trade Size')
                axes[0, 2].set_ylabel('Frequency')
                axes[0, 2].grid(True, alpha=0.3)
            else:
                axes[0, 2].text(0.5, 0.5, 'Trade size\ndata not available', 
                              ha='center', va='center', transform=axes[0, 2].transAxes)
                axes[0, 2].set_title('Trade Size Distribution', fontweight='bold')
            
            # 4. P&L by Hour of Day
            if not trades.empty:
                trades['hour'] = pd.to_datetime(trades['timestamp']).dt.hour
                hourly_pnl = trades.groupby('hour')['net_pnl'].sum()
                
                bars = axes[1, 0].bar(hourly_pnl.index, hourly_pnl.values, 
                                    color=['green' if x > 0 else 'red' for x in hourly_pnl.values],
                                    alpha=0.7)
                axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
                axes[1, 0].set_title('P&L by Hour of Day', fontweight='bold')
                axes[1, 0].set_xlabel('Hour')
                axes[1, 0].set_ylabel('Total P&L ($)')
                axes[1, 0].grid(True, alpha=0.3)
            
            # 5. Cumulative P&L
            trades_sorted = trades.sort_values('timestamp')
            cumulative_pnl = trades_sorted['net_pnl'].cumsum()
            
            axes[1, 1].plot(range(len(cumulative_pnl)), cumulative_pnl, 
                          color=self.colors['equity'], linewidth=2)
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[1, 1].set_title('Cumulative P&L by Trade', fontweight='bold')
            axes[1, 1].set_xlabel('Trade Number')
            axes[1, 1].set_ylabel('Cumulative P&L ($)')
            axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Trade Duration (if available)
            if 'entry_time' in trades.columns and 'exit_time' in trades.columns:
                entry_times = pd.to_datetime(trades['entry_time'])
                exit_times = pd.to_datetime(trades['exit_time'])
                durations = (exit_times - entry_times).dt.total_seconds() / 3600  # Hours
                
                axes[1, 2].hist(durations, bins=20, color='lightpink', 
                              alpha=0.7, edgecolor='black')
                axes[1, 2].set_title('Trade Duration Distribution', fontweight='bold')
                axes[1, 2].set_xlabel('Duration (Hours)')
                axes[1, 2].set_ylabel('Frequency')
                axes[1, 2].grid(True, alpha=0.3)
            else:
                axes[1, 2].text(0.5, 0.5, 'Trade duration\ndata not available', 
                              ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('Trade Duration Distribution', fontweight='bold')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                return save_path
            else:
                plt.show()
                return ""
                
        except Exception as e:
            logger.error(f"Error plotting trade distribution: {e}")
            plt.close()
            return ""
    
    def plot_performance_dashboard(self, analysis_results: Dict[str, Any], save_path: str = None) -> str:
        """Plot performance metrics dashboard"""
        try:
            fig = plt.figure(figsize=self.figsize_large)
            gs = GridSpec(3, 4, figure=fig)
            
            # Extract metrics
            metrics = analysis_results.get('performance_metrics', {})
            risk_analysis = analysis_results.get('risk_analysis', {})
            
            # Key metrics display
            ax_metrics = fig.add_subplot(gs[0, :2])
            ax_metrics.axis('off')
            
            key_metrics = [
                ('Total Return', f"{metrics.get('total_return', 0)*100:.2f}%"),
                ('Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}"),
                ('Max Drawdown', f"{metrics.get('max_drawdown', 0)*100:.2f}%"),
                ('Win Rate', f"{metrics.get('win_rate', 0)*100:.1f}%"),
                ('Profit Factor', f"{metrics.get('profit_factor', 0):.2f}"),
                ('Total Trades', f"{metrics.get('total_trades', 0)}")
            ]
            
            y_pos = 0.9
            for metric, value in key_metrics:
                ax_metrics.text(0, y_pos, f"{metric}:", fontweight='bold', fontsize=12)
                ax_metrics.text(0.5, y_pos, value, fontsize=12)
                y_pos -= 0.15
            
            ax_metrics.set_title('Key Performance Metrics', fontweight='bold', fontsize=14)
            
            # Returns comparison
            ax_returns = fig.add_subplot(gs[0, 2:])
            returns_data = [
                metrics.get('total_return', 0) * 100,
                8.0,  # Benchmark (S&P 500 average)
                2.0   # Risk-free rate
            ]
            returns_labels = ['Strategy', 'Benchmark', 'Risk-Free']
            colors_returns = ['skyblue', 'lightcoral', 'lightgray']
            
            bars = ax_returns.bar(returns_labels, returns_data, color=colors_returns, alpha=0.7)
            ax_returns.set_title('Return Comparison', fontweight='bold')
            ax_returns.set_ylabel('Return (%)')
            ax_returns.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, returns_data):
                height = bar.get_height()
                ax_returns.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                              f'{value:.1f}%', ha='center', va='bottom')
            
            # Risk metrics
            ax_risk = fig.add_subplot(gs[1, :2])
            risk_metrics = [
                ('Volatility', metrics.get('volatility', 0) * 100, '%'),
                ('VaR (95%)', abs(risk_analysis.get('var_95', 0)) * 100, '%'),
                ('Skewness', risk_analysis.get('skewness', 0), ''),
                ('Kurtosis', risk_analysis.get('kurtosis', 0), ''),
                ('Sortino Ratio', metrics.get('sortino_ratio', 0), ''),
                ('Calmar Ratio', metrics.get('calmar_ratio', 0), '')
            ]
            
            y_pos = 0.9
            for metric, value, unit in risk_metrics:
                ax_risk.text(0, y_pos, f"{metric}:", fontweight='bold', fontsize=10)
                if unit == '%':
                    ax_risk.text(0.6, y_pos, f"{value:.2f}{unit}", fontsize=10)
                else:
                    ax_risk.text(0.6, y_pos, f"{value:.2f}", fontsize=10)
                y_pos -= 0.15
            
            ax_risk.axis('off')
            ax_risk.set_title('Risk Metrics', fontweight='bold', fontsize=14)
            
            # Trade statistics
            ax_trades = fig.add_subplot(gs[1, 2:])
            trade_stats = [
                ('Avg Win', metrics.get('avg_win', 0)),
                ('Avg Loss', abs(metrics.get('avg_loss', 0))),
                ('Largest Win', metrics.get('largest_win', 0)),
                ('Largest Loss', abs(metrics.get('largest_loss', 0)))
            ]
            
            trade_labels = [stat[0] for stat in trade_stats]
            trade_values = [stat[1] for stat in trade_stats]
            colors_trades = ['green', 'red', 'darkgreen', 'darkred']
            
            bars = ax_trades.bar(trade_labels, trade_values, color=colors_trades, alpha=0.7)
            ax_trades.set_title('Trade Statistics', fontweight='bold')
            ax_trades.set_ylabel('Amount ($)')
            ax_trades.grid(True, alpha=0.3)
            plt.setp(ax_trades.xaxis.get_majorticklabels(), rotation=45)
            
            # Monthly returns heatmap (if time analysis available)
            time_analysis = analysis_results.get('time_analysis', {})
            monthly_perf = time_analysis.get('monthly_performance', [])
            
            if monthly_perf:
                ax_heatmap = fig.add_subplot(gs[2, :])
                
                # Create monthly returns matrix
                monthly_df = pd.DataFrame(monthly_perf)
                if not monthly_df.empty and 'year' in monthly_df.columns and 'month' in monthly_df.columns:
                    pivot_table = monthly_df.pivot(index='year', columns='month', values='sum')
                    
                    # Create heatmap
                    sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='RdYlGn', 
                              center=0, ax=ax_heatmap, cbar_kws={'label': 'Monthly P&L ($)'})
                    ax_heatmap.set_title('Monthly Returns Heatmap', fontweight='bold')
                    ax_heatmap.set_xlabel('Month')
                    ax_heatmap.set_ylabel('Year')
                else:
                    ax_heatmap.text(0.5, 0.5, 'Monthly data\nnot available', 
                                  ha='center', va='center', transform=ax_heatmap.transAxes)
                    ax_heatmap.set_title('Monthly Returns Heatmap', fontweight='bold')
            else:
                ax_heatmap = fig.add_subplot(gs[2, :])
                ax_heatmap.text(0.5, 0.5, 'Monthly performance\ndata not available', 
                              ha='center', va='center', transform=ax_heatmap.transAxes)
                ax_heatmap.set_title('Monthly Returns Heatmap', fontweight='bold')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                return save_path
            else:
                plt.show()
                return ""
                
        except Exception as e:
            logger.error(f"Error plotting performance dashboard: {e}")
            plt.close()
            return ""
    
    def plot_time_analysis(self, trades: pd.DataFrame, equity_curve: pd.DataFrame, save_path: str = None) -> str:
        """Plot time-based analysis"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=self.figsize_large)
            
            if not trades.empty:
                trades['timestamp'] = pd.to_datetime(trades['timestamp'])
                trades['hour'] = trades['timestamp'].dt.hour
                trades['day_of_week'] = trades['timestamp'].dt.day_name()
                trades['month'] = trades['timestamp'].dt.month
            
            # 1. Hourly performance
            if not trades.empty:
                hourly_pnl = trades.groupby('hour')['net_pnl'].sum()
                bars = axes[0, 0].bar(hourly_pnl.index, hourly_pnl.values, 
                                    color=['green' if x > 0 else 'red' for x in hourly_pnl.values],
                                    alpha=0.7)
                axes[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
                axes[0, 0].set_title('Performance by Hour of Day', fontweight='bold')
                axes[0, 0].set_xlabel('Hour')
                axes[0, 0].set_ylabel('Total P&L ($)')
                axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Daily performance
            if not trades.empty:
                daily_pnl = trades.groupby('day_of_week')['net_pnl'].sum()
                # Reorder days
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                daily_pnl = daily_pnl.reindex([day for day in day_order if day in daily_pnl.index])
                
                bars = axes[0, 1].bar(range(len(daily_pnl)), daily_pnl.values, 
                                    color=['green' if x > 0 else 'red' for x in daily_pnl.values],
                                    alpha=0.7)
                axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
                axes[0, 1].set_title('Performance by Day of Week', fontweight='bold')
                axes[0, 1].set_xlabel('Day')
                axes[0, 1].set_ylabel('Total P&L ($)')
                axes[0, 1].set_xticks(range(len(daily_pnl)))
                axes[0, 1].set_xticklabels([day[:3] for day in daily_pnl.index], rotation=45)
                axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Monthly equity progression
            if not equity_curve.empty:
                equity_curve['month_year'] = equity_curve.index.to_period('M')
                monthly_equity = equity_curve.groupby('month_year')['equity'].last()
                monthly_returns = monthly_equity.pct_change().dropna() * 100
                
                bars = axes[1, 0].bar(range(len(monthly_returns)), monthly_returns.values, 
                                    color=['green' if x > 0 else 'red' for x in monthly_returns.values],
                                    alpha=0.7)
                axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
                axes[1, 0].set_title('Monthly Returns', fontweight='bold')
                axes[1, 0].set_xlabel('Month')
                axes[1, 0].set_ylabel('Return (%)')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Rotate labels if too many months
                if len(monthly_returns) > 12:
                    step = max(1, len(monthly_returns) // 12)
                    tick_positions = range(0, len(monthly_returns), step)
                    tick_labels = [str(monthly_returns.index[i]) for i in tick_positions]
                    axes[1, 0].set_xticks(tick_positions)
                    axes[1, 0].set_xticklabels(tick_labels, rotation=45)
            
            # 4. Trade frequency over time
            if not trades.empty:
                trades['date'] = trades['timestamp'].dt.date
                daily_trade_count = trades.groupby('date').size()
                
                axes[1, 1].plot(daily_trade_count.index, daily_trade_count.values, 
                              color='blue', alpha=0.7)
                axes[1, 1].set_title('Trade Frequency Over Time', fontweight='bold')
                axes[1, 1].set_xlabel('Date')
                axes[1, 1].set_ylabel('Number of Trades')
                axes[1, 1].grid(True, alpha=0.3)
                
                # Format x-axis
                axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                return save_path
            else:
                plt.show()
                return ""
                
        except Exception as e:
            logger.error(f"Error plotting time analysis: {e}")
            plt.close()
            return ""
    
    def plot_risk_analysis(self, equity_curve: pd.DataFrame, trades: pd.DataFrame, save_path: str = None) -> str:
        """Plot risk analysis charts"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=self.figsize_large)
            
            # Calculate returns
            if not equity_curve.empty:
                returns = equity_curve['equity'].pct_change().dropna()
            else:
                returns = pd.Series()
            
            # 1. Return distribution
            if not returns.empty:
                axes[0, 0].hist(returns * 100, bins=30, color='lightblue', 
                              alpha=0.7, edgecolor='black', density=True)
                
                # Overlay normal distribution
                mu, sigma = returns.mean() * 100, returns.std() * 100
                x = np.linspace(returns.min() * 100, returns.max() * 100, 100)
                y = ((1/(sigma * np.sqrt(2 * np.pi))) * 
                     np.exp(-0.5 * ((x - mu) / sigma) ** 2))
                axes[0, 0].plot(x, y, 'r--', label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
                
                axes[0, 0].axvline(mu, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mu:.2f}%')
                axes[0, 0].set_title('Return Distribution', fontweight='bold')
                axes[0, 0].set_xlabel('Return (%)')
                axes[0, 0].set_ylabel('Density')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Rolling volatility
            if not returns.empty and len(returns) > 30:
                rolling_vol = returns.rolling(window=30).std() * np.sqrt(252) * 100  # Annualized
                axes[0, 1].plot(rolling_vol.index, rolling_vol.values, color='orange', linewidth=2)
                axes[0, 1].set_title('Rolling 30-Day Volatility', fontweight='bold')
                axes[0, 1].set_xlabel('Date')
                axes[0, 1].set_ylabel('Volatility (%)')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Format x-axis
                axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
            
            # 3. QQ plot for normality check
            if not returns.empty:
                from scipy import stats
                stats.probplot(returns, dist="norm", plot=axes[1, 0])
                axes[1, 0].set_title('Q-Q Plot (Normal Distribution)', fontweight='bold')
                axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Risk-adjusted returns scatter
            if not trades.empty and len(trades) > 10:
                # Calculate rolling Sharpe ratio
                if len(returns) > 30:
                    rolling_mean = returns.rolling(window=30).mean() * 252  # Annualized
                    rolling_std = returns.rolling(window=30).std() * np.sqrt(252)  # Annualized
                    rolling_sharpe = (rolling_mean - 0.02) / rolling_std  # Assuming 2% risk-free rate
                    
                    # Plot equity vs Sharpe ratio
                    equity_values = equity_curve['equity'].iloc[30:]  # Align with rolling window
                    
                    scatter = axes[1, 1].scatter(rolling_sharpe.values, equity_values.values, 
                                               c=range(len(rolling_sharpe)), cmap='viridis', alpha=0.6)
                    axes[1, 1].set_title('Equity vs Rolling Sharpe Ratio', fontweight='bold')
                    axes[1, 1].set_xlabel('30-Day Rolling Sharpe Ratio')
                    axes[1, 1].set_ylabel('Portfolio Equity ($)')
                    axes[1, 1].grid(True, alpha=0.3)
                    
                    # Add colorbar
                    cbar = plt.colorbar(scatter, ax=axes[1, 1])
                    cbar.set_label('Time Progression')
                else:
                    axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor rolling analysis', 
                                   ha='center', va='center', transform=axes[1, 1].transAxes)
                    axes[1, 1].set_title('Equity vs Rolling Sharpe Ratio', fontweight='bold')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                return save_path
            else:
                plt.show()
                return ""
                
        except Exception as e:
            logger.error(f"Error plotting risk analysis: {e}")
            plt.close()
            return ""
    
    def plot_market_conditions(self, equity_curve: pd.DataFrame, market_data: pd.DataFrame, 
                             trades: pd.DataFrame, save_path: str = None) -> str:
        """Plot strategy performance against market conditions"""
        try:
            fig, axes = plt.subplots(3, 1, figsize=self.figsize_large, 
                                   gridspec_kw={'height_ratios': [2, 1, 1]})
            
            # Ensure market_data has timestamp index
            if 'timestamp' in market_data.columns:
                market_data.set_index('timestamp', inplace=True)
            
            # 1. Equity curve vs market price
            ax1 = axes[0]
            
            # Plot equity (left y-axis)
            ax1.plot(equity_curve.index, equity_curve['equity'], 
                    color=self.colors['equity'], linewidth=2, label='Portfolio Equity')
            ax1.set_ylabel('Portfolio Equity ($)', color=self.colors['equity'])
            ax1.tick_params(axis='y', labelcolor=self.colors['equity'])
            
            # Plot market price (right y-axis)
            ax1_twin = ax1.twinx()
            if 'close' in market_data.columns:
                ax1_twin.plot(market_data.index, market_data['close'], 
                            color='gray', alpha=0.7, linewidth=1, label='Market Price')
                ax1_twin.set_ylabel('Market Price ($)', color='gray')
                ax1_twin.tick_params(axis='y', labelcolor='gray')
            
            ax1.set_title('Portfolio Performance vs Market Price', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Add trade markers
            if not trades.empty:
                buy_trades = trades[trades['side'] == 'buy']
                sell_trades = trades[trades['side'] == 'sell']
                
                if not buy_trades.empty:
                    ax1.scatter(buy_trades['timestamp'], 
                              [equity_curve.loc[equity_curve.index.get_loc(t, method='nearest'), 'equity'] 
                               for t in buy_trades['timestamp']], 
                              color='green', marker='^', s=30, alpha=0.7, zorder=5)
                
                if not sell_trades.empty:
                    ax1.scatter(sell_trades['timestamp'], 
                              [equity_curve.loc[equity_curve.index.get_loc(t, method='nearest'), 'equity'] 
                               for t in sell_trades['timestamp']], 
                              color='red', marker='v', s=30, alpha=0.7, zorder=5)
            
            # 2. Market volatility
            if 'close' in market_data.columns:
                market_returns = market_data['close'].pct_change().dropna()
                market_vol = market_returns.rolling(window=24).std() * np.sqrt(24) * 100  # Daily vol %
                
                axes[1].plot(market_vol.index, market_vol.values, color='orange', linewidth=1)
                axes[1].set_title('Market Volatility (24-period rolling)', fontweight='bold')
                axes[1].set_ylabel('Volatility (%)')
                axes[1].grid(True, alpha=0.3)
            
            # 3. Strategy returns vs market returns
            if not equity_curve.empty and 'close' in market_data.columns:
                # Align data
                aligned_equity = equity_curve['equity'].reindex(market_data.index, method='nearest')
                strategy_returns = aligned_equity.pct_change().dropna()
                market_returns = market_data['close'].pct_change().dropna()
                
                # Get common dates
                common_dates = strategy_returns.index.intersection(market_returns.index)
                if len(common_dates) > 1:
                    strategy_ret = strategy_returns.loc[common_dates] * 100
                    market_ret = market_returns.loc[common_dates] * 100
                    
                    # Scatter plot
                    axes[2].scatter(market_ret.values, strategy_ret.values, 
                                  alpha=0.6, color='blue', s=20)
                    
                    # Add trend line
                    if len(market_ret) > 2:
                        z = np.polyfit(market_ret.values, strategy_ret.values, 1)
                        p = np.poly1d(z)
                        axes[2].plot(market_ret.values, p(market_ret.values), 
                                   "r--", alpha=0.8, linewidth=2)
                        
                        # Calculate correlation
                        correlation = np.corrcoef(market_ret.values, strategy_ret.values)[0, 1]
                        axes[2].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                                   transform=axes[2].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
                    
                    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    axes[2].axvline(x=0, color='black', linestyle='-', alpha=0.3)
                    axes[2].set_title('Strategy Returns vs Market Returns', fontweight='bold')
                    axes[2].set_xlabel('Market Returns (%)')
                    axes[2].set_ylabel('Strategy Returns (%)')
                    axes[2].grid(True, alpha=0.3)
            
            # Format x-axis for all subplots
            for ax in axes:
                if hasattr(ax, 'xaxis'):
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                return save_path
            else:
                plt.show()
                return ""
                
        except Exception as e:
            logger.error(f"Error plotting market conditions: {e}")
            plt.close()
            return ""
    
    def plot_trade_timing(self, trades: pd.DataFrame, save_path: str = None) -> str:
        """Plot trade timing analysis"""
        try:
            if trades.empty:
                logger.warning("No trades to analyze")
                return ""
            
            fig, axes = plt.subplots(2, 2, figsize=self.figsize_large)
            
            trades['timestamp'] = pd.to_datetime(trades['timestamp'])
            trades_sorted = trades.sort_values('timestamp')
            
            # 1. Cumulative trade count over time
            trades_sorted['cumulative_trades'] = range(1, len(trades_sorted) + 1)
            
            axes[0, 0].plot(trades_sorted['timestamp'], trades_sorted['cumulative_trades'], 
                          color='blue', linewidth=2)
            axes[0, 0].set_title('Cumulative Trade Count', fontweight='bold')
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Number of Trades')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Trade frequency (trades per day)
            trades_sorted['date'] = trades_sorted['timestamp'].dt.date
            daily_counts = trades_sorted.groupby('date').size()
            
            axes[0, 1].plot(daily_counts.index, daily_counts.values, 
                          color='green', alpha=0.7, linewidth=1)
            axes[0, 1].set_title('Daily Trade Frequency', fontweight='bold')
            axes[0, 1].set_xlabel('Date')
            axes[0, 1].set_ylabel('Trades per Day')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add moving average
            if len(daily_counts) > 7:
                ma7 = pd.Series(daily_counts.values).rolling(window=7).mean()
                axes[0, 1].plot(daily_counts.index, ma7.values, 
                              color='red', linewidth=2, label='7-day MA')
                axes[0, 1].legend()
            
            # 3. Time between trades
            if len(trades_sorted) > 1:
                time_diffs = trades_sorted['timestamp'].diff().dt.total_seconds() / 3600  # Hours
                time_diffs = time_diffs.dropna()
                
                axes[1, 0].hist(time_diffs, bins=30, color='purple', alpha=0.7, edgecolor='black')
                axes[1, 0].axvline(time_diffs.median(), color='red', linestyle='--', 
                                 label=f'Median: {time_diffs.median():.1f}h')
                axes[1, 0].set_title('Time Between Trades Distribution', fontweight='bold')
                axes[1, 0].set_xlabel('Hours Between Trades')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Trade size over time
            if 'size' in trades_sorted.columns:
                axes[1, 1].scatter(trades_sorted['timestamp'], trades_sorted['size'], 
                                 alpha=0.6, color='orange')
                
                # Add trend line
                if len(trades_sorted) > 2:
                    x_numeric = mdates.date2num(trades_sorted['timestamp'])
                    z = np.polyfit(x_numeric, trades_sorted['size'], 1)
                    p = np.poly1d(z)
                    axes[1, 1].plot(trades_sorted['timestamp'], p(x_numeric), 
                                   "r--", alpha=0.8, linewidth=2, label='Trend')
                    axes[1, 1].legend()
                
                axes[1, 1].set_title('Trade Size Over Time', fontweight='bold')
                axes[1, 1].set_xlabel('Date')
                axes[1, 1].set_ylabel('Trade Size')
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'Trade size\ndata not available', 
                              ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Trade Size Over Time', fontweight='bold')
            
            # Format x-axis dates
            for ax in [axes[0, 0], axes[0, 1], axes[1, 1]]:
                if hasattr(ax, 'xaxis'):
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                return save_path
            else:
                plt.show()
                return ""
                
        except Exception as e:
            logger.error(f"Error plotting trade timing: {e}")
            plt.close()
            return ""
    
    def plot_rolling_performance(self, equity_curve: pd.DataFrame, save_path: str = None) -> str:
        """Plot rolling performance metrics"""
        try:
            if equity_curve.empty:
                logger.warning("No equity data to analyze")
                return ""
            
            fig, axes = plt.subplots(3, 1, figsize=self.figsize_large)
            
            returns = equity_curve['equity'].pct_change().dropna()
            
            # Parameters
            windows = [30, 90, 180]  # Different rolling windows
            
            # 1. Rolling returns
            for window in windows:
                if len(returns) > window:
                    rolling_returns = returns.rolling(window=window).mean() * 252 * 100  # Annualized %
                    axes[0].plot(rolling_returns.index, rolling_returns.values, 
                               linewidth=2, label=f'{window}-day')
            
            axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[0].set_title('Rolling Annualized Returns', fontweight='bold')
            axes[0].set_ylabel('Return (%)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # 2. Rolling Sharpe ratio
            risk_free_rate = 0.02  # 2% annual
            
            for window in windows:
                if len(returns) > window:
                    rolling_mean = returns.rolling(window=window).mean() * 252
                    rolling_std = returns.rolling(window=window).std() * np.sqrt(252)
                    rolling_sharpe = (rolling_mean - risk_free_rate) / rolling_std
                    
                    axes[1].plot(rolling_sharpe.index, rolling_sharpe.values, 
                               linewidth=2, label=f'{window}-day')
            
            axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1].axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Sharpe = 1.0')
            axes[1].set_title('Rolling Sharpe Ratio', fontweight='bold')
            axes[1].set_ylabel('Sharpe Ratio')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # 3. Rolling maximum drawdown
            for window in windows:
                if len(equity_curve) > window:
                    rolling_dd = []
                    for i in range(window, len(equity_curve)):
                        window_data = equity_curve['equity'].iloc[i-window:i]
                        peak = window_data.expanding().max()
                        drawdown = (window_data - peak) / peak
                        rolling_dd.append(drawdown.min())
                    
                    dd_index = equity_curve.index[window:]
                    axes[2].plot(dd_index, np.array(rolling_dd) * 100, 
                               linewidth=2, label=f'{window}-day')
            
            axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[2].set_title('Rolling Maximum Drawdown', fontweight='bold')
            axes[2].set_xlabel('Date')
            axes[2].set_ylabel('Max Drawdown (%)')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            # Format x-axis
            for ax in axes:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                return save_path
            else:
                plt.show()
                return ""
                
        except Exception as e:
            logger.error(f"Error plotting rolling performance: {e}")
            plt.close()
            return ""
    
    def plot_cost_analysis(self, cost_data: Dict[str, Any], trades: pd.DataFrame, save_path: str = None) -> str:
        """Plot cost analysis"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=self.figsize_large)
            
            # 1. Cost breakdown pie chart
            cost_components = {
                'Commission': cost_data.get('total_commission_paid', 0),
                'Slippage': cost_data.get('total_slippage_cost', 0),
                'Market Impact': cost_data.get('total_market_impact', 0)
            }
            
            # Filter out zero values
            cost_components = {k: v for k, v in cost_components.items() if v > 0}
            
            if cost_components:
                axes[0, 0].pie(cost_components.values(), labels=cost_components.keys(), 
                             autopct='%1.1f%%', startangle=90)
                axes[0, 0].set_title('Cost Breakdown', fontweight='bold')
            else:
                axes[0, 0].text(0.5, 0.5, 'No cost data\navailable', 
                              ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('Cost Breakdown', fontweight='bold')
            
            # 2. Cost per trade over time
            if not trades.empty and 'transaction_cost' in trades.columns:
                trades_sorted = trades.sort_values('timestamp')
                
                axes[0, 1].plot(trades_sorted['timestamp'], trades_sorted['transaction_cost'], 
                              color='red', alpha=0.7, linewidth=1, marker='o', markersize=3)
                axes[0, 1].set_title('Transaction Cost per Trade', fontweight='bold')
                axes[0, 1].set_xlabel('Date')
                axes[0, 1].set_ylabel('Cost ($)')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Format x-axis
                axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
            else:
                axes[0, 1].text(0.5, 0.5, 'Trade cost data\nnot available', 
                              ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Transaction Cost per Trade', fontweight='bold')
            
            # 3. Cost impact on returns
            total_costs = cost_data.get('total_transaction_costs', 0)
            gross_pnl = trades['gross_pnl'].sum() if 'gross_pnl' in trades.columns else trades['net_pnl'].sum()
            net_pnl = gross_pnl - total_costs
            
            returns_comparison = {
                'Gross Return': gross_pnl,
                'Transaction Costs': -total_costs,
                'Net Return': net_pnl
            }
            
            colors = ['green', 'red', 'blue']
            bars = axes[1, 0].bar(returns_comparison.keys(), returns_comparison.values(), 
                                color=colors, alpha=0.7)
            axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[1, 0].set_title('Cost Impact on Returns', fontweight='bold')
            axes[1, 0].set_ylabel('Amount ($)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, returns_comparison.values()):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., 
                              height + (abs(height) * 0.01 if height >= 0 else -abs(height) * 0.01),
                              f'${value:.2f}', ha='center', 
                              va='bottom' if height >= 0 else 'top')
            
            # 4. Cost as percentage of trade value
            if not trades.empty and 'size' in trades.columns and 'price' in trades.columns:
                trades['trade_value'] = trades['size'] * trades['price']
                if 'transaction_cost' in trades.columns:
                    trades['cost_percentage'] = (trades['transaction_cost'] / trades['trade_value']) * 100
                    
                    axes[1, 1].hist(trades['cost_percentage'], bins=20, 
                                  color='orange', alpha=0.7, edgecolor='black')
                    axes[1, 1].axvline(trades['cost_percentage'].mean(), color='red', 
                                     linestyle='--', label=f'Mean: {trades["cost_percentage"].mean():.3f}%')
                    axes[1, 1].set_title('Cost as % of Trade Value', fontweight='bold')
                    axes[1, 1].set_xlabel('Cost (%)')
                    axes[1, 1].set_ylabel('Frequency')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)
                else:
                    axes[1, 1].text(0.5, 0.5, 'Cost percentage\ndata not available', 
                                  ha='center', va='center', transform=axes[1, 1].transAxes)
                    axes[1, 1].set_title('Cost as % of Trade Value', fontweight='bold')
            else:
                axes[1, 1].text(0.5, 0.5, 'Trade value data\nnot available', 
                              ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Cost as % of Trade Value', fontweight='bold')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                return save_path
            else:
                plt.show()
                return ""
                
        except Exception as e:
            logger.error(f"Error plotting cost analysis: {e}")
            plt.close()
            return ""
    
    def create_interactive_dashboard(self, 
                                   backtest_results: Dict[str, Any],
                                   analysis_results: Dict[str, Any] = None,
                                   market_data: pd.DataFrame = None,
                                   save_path: str = None) -> str:
        """Create interactive Plotly dashboard"""
        try:
            if not PLOTLY_AVAILABLE:
                logger.warning("Plotly not available for interactive charts")
                return ""
            
            # Extract data
            trades = pd.DataFrame(backtest_results.get('trades', []))
            equity_curve = pd.DataFrame(backtest_results.get('equity_curve', []))
            
            if trades.empty or equity_curve.empty:
                logger.warning("No data available for interactive dashboard")
                return ""
            
            # Convert timestamps
            trades['timestamp'] = pd.to_datetime(trades['timestamp'])
            equity_curve['timestamp'] = pd.to_datetime(equity_curve['timestamp'])
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=[
                    'Portfolio Equity Curve', 'Drawdown Analysis',
                    'Trade Distribution', 'Performance Metrics',
                    'Rolling Performance', 'Market Comparison'
                ],
                specs=[[{"secondary_y": True}, {"type": "scatter"}],
                       [{"type": "histogram"}, {"type": "bar"}],
                       [{"secondary_y": True}, {"type": "scatter"}]]
            )
            
            # 1. Equity curve with trades
            fig.add_trace(
                go.Scatter(x=equity_curve['timestamp'], y=equity_curve['equity'],
                          mode='lines', name='Portfolio Equity', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Add trade markers
            if not trades.empty:
                buy_trades = trades[trades['side'] == 'buy']
                sell_trades = trades[trades['side'] == 'sell']
                
                if not buy_trades.empty:
                    fig.add_trace(
                        go.Scatter(x=buy_trades['timestamp'], y=buy_trades['price'],
                                  mode='markers', name='Buy Signals',
                                  marker=dict(symbol='triangle-up', color='green', size=8)),
                        row=1, col=1, secondary_y=True
                    )
                
                if not sell_trades.empty:
                    fig.add_trace(
                        go.Scatter(x=sell_trades['timestamp'], y=sell_trades['price'],
                                  mode='markers', name='Sell Signals',
                                  marker=dict(symbol='triangle-down', color='red', size=8)),
                        row=1, col=1, secondary_y=True
                    )
            
            # 2. Drawdown
            peak = equity_curve['equity'].expanding().max()
            drawdown = (equity_curve['equity'] - peak) / peak * 100
            
            fig.add_trace(
                go.Scatter(x=equity_curve['timestamp'], y=drawdown,
                          fill='tonexty', mode='lines', name='Drawdown',
                          line=dict(color='red'), fillcolor='rgba(255,0,0,0.3)'),
                row=1, col=2
            )
            
            # 3. Trade P&L distribution
            if not trades.empty:
                fig.add_trace(
                    go.Histogram(x=trades['net_pnl'], name='P&L Distribution',
                               marker_color='lightblue'),
                    row=2, col=1
                )
            
            # 4. Performance metrics bar chart
            if analysis_results:
                metrics = analysis_results.get('performance_metrics', {})
                metric_names = ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
                metric_values = [
                    metrics.get('total_return', 0) * 100,
                    metrics.get('sharpe_ratio', 0),
                    metrics.get('max_drawdown', 0) * 100,
                    metrics.get('win_rate', 0) * 100
                ]
                
                fig.add_trace(
                    go.Bar(x=metric_names, y=metric_values, name='Performance Metrics',
                          marker_color='lightgreen'),
                    row=2, col=2
                )
            
            # 5. Rolling Sharpe ratio
            if len(equity_curve) > 30:
                returns = equity_curve['equity'].pct_change().dropna()
                rolling_sharpe = ((returns.rolling(window=30).mean() * 252 - 0.02) / 
                                (returns.rolling(window=30).std() * np.sqrt(252)))
                
                fig.add_trace(
                    go.Scatter(x=equity_curve['timestamp'][30:], y=rolling_sharpe.iloc[30:],
                              mode='lines', name='30-day Rolling Sharpe',
                              line=dict(color='purple')),
                    row=3, col=1
                )
            
            # 6. Strategy vs Market (if market data available)
            if market_data is not None and 'close' in market_data.columns:
                # Normalize both series to start at 100
                equity_norm = (equity_curve['equity'] / equity_curve['equity'].iloc[0]) * 100
                market_norm = (market_data['close'] / market_data['close'].iloc[0]) * 100
                
                fig.add_trace(
                    go.Scatter(x=equity_curve['timestamp'], y=equity_norm,
                              mode='lines', name='Strategy (Normalized)',
                              line=dict(color='blue')),
                    row=3, col=2
                )
                
                fig.add_trace(
                    go.Scatter(x=market_data.index, y=market_norm,
                              mode='lines', name='Market (Normalized)',
                              line=dict(color='gray')),
                    row=3, col=2
                )
            
            # Update layout
            fig.update_layout(
                title_text="Interactive Backtest Dashboard",
                title_x=0.5,
                height=1200,
                showlegend=True,
                template="plotly_white"
            )
            
            # Set y-axis titles
            fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1, secondary_y=True)
            fig.update_yaxes(title_text="Drawdown (%)", row=1, col=2)
            fig.update_yaxes(title_text="Frequency", row=2, col=1)
            fig.update_yaxes(title_text="Value", row=2, col=2)
            fig.update_yaxes(title_text="Sharpe Ratio", row=3, col=1)
            fig.update_yaxes(title_text="Normalized Value", row=3, col=2)
            
            # Save to HTML
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Interactive dashboard saved to {save_path}")
                return save_path
            else:
                fig.show()
                return ""
                
        except Exception as e:
            logger.error(f"Error creating interactive dashboard: {e}")
            return ""


# Factory function
def create_backtest_visualizer(config: Dict[str, Any] = None) -> BacktestVisualizer:
    """Create and return BacktestVisualizer instance"""
    return BacktestVisualizer(config)