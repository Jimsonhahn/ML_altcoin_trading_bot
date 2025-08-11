#!/usr/bin/env python3
# scripts/run_comprehensive_backtest.py
"""
Comprehensive Backtesting Script
Runs complete backtesting suite with optimization, analysis, and reporting
"""

import asyncio
import logging
import json
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Core imports
from core.backtest_engine import BacktestEngine
from core.backtest_analyzer import BacktestAnalyzer
from core.strategy_optimizer import StrategyOptimizer, ParameterRange, OptimizationMethod, OptimizationObjective
from data.market_simulator import MarketSimulator
from utils.backtest_visualizer import BacktestVisualizer

# Strategy imports
from strategies.momentum_strategy import MomentumStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.scalping_strategy import ScalpingStrategy

# Data handling
from data_sources.data_manager import DataManager

# Try to import notifier
try:
    from utils.notifier import send_info, send_warning, send_error
    NOTIFIER_AVAILABLE = True
except ImportError:
    NOTIFIER_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveBacktester:
    """
    Comprehensive backtesting suite with optimization and analysis
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Components
        self.backtest_engine = None
        self.analyzer = None
        self.optimizer = None
        self.visualizer = None
        self.market_simulator = None
        self.data_manager = None
        
        # Results storage
        self.results = {}
        self.optimization_results = {}
        
        # Configuration
        self.output_dir = self.config.get('output_dir', 'data/comprehensive_backtest_results')
        self.enable_optimization = self.config.get('enable_optimization', True)
        self.enable_visualization = self.config.get('enable_visualization', True)
        self.enable_synthetic_data = self.config.get('enable_synthetic_data', False)
        self.enable_stress_testing = self.config.get('enable_stress_testing', True)
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("ComprehensiveBacktester initialized")
    
    def initialize_components(self):
        """Initialize all backtesting components"""
        try:
            logger.info("Initializing backtesting components...")
            
            # Backtest engine with realistic settings
            backtest_config = self.config.get('backtest_engine', {
                'initial_capital': 10000,
                'enable_slippage': True,
                'enable_market_impact': True,
                'enable_latency': True,
                'enable_liquidity_constraints': True,
                'commission_rate': 0.001,  # 0.1%
                'slippage_factor': 0.0005,  # 0.05%
                'market_impact_factor': 0.0002,  # 0.02%
                'min_latency_ms': 50,
                'max_latency_ms': 150
            })
            self.backtest_engine = BacktestEngine(backtest_config)
            
            # Analyzer
            analyzer_config = self.config.get('analyzer', {
                'risk_free_rate': 0.02,
                'benchmark_return': 0.08
            })
            self.analyzer = BacktestAnalyzer(analyzer_config)
            
            # Optimizer
            if self.enable_optimization:
                optimizer_config = self.config.get('optimizer', {
                    'max_workers': 4,
                    'timeout_seconds': 3600,
                    'validation_split': 0.3
                })
                self.optimizer = StrategyOptimizer(optimizer_config)
            
            # Visualizer
            if self.enable_visualization:
                visualizer_config = self.config.get('visualizer', {
                    'output_dir': self.output_dir,
                    'dpi': 300
                })
                self.visualizer = BacktestVisualizer(visualizer_config)
            
            # Market simulator
            if self.enable_synthetic_data:
                simulator_config = self.config.get('market_simulator', {
                    'enable_regimes': True,
                    'enable_events': True,
                    'enable_correlations': True
                })
                self.market_simulator = MarketSimulator(simulator_config)
            
            # Data manager
            self.data_manager = DataManager()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    async def run_comprehensive_backtest(self, 
                                       strategy_configs: List[Dict[str, Any]],
                                       symbols: List[str] = None,
                                       start_date: str = None,
                                       end_date: str = None,
                                       data_source: str = 'historical') -> Dict[str, Any]:
        """
        Run comprehensive backtest with multiple strategies and analysis
        """
        try:
            logger.info("Starting comprehensive backtesting suite...")
            
            if NOTIFIER_AVAILABLE:
                send_info("🚀 Starting comprehensive backtesting suite")
            
            # Initialize components
            self.initialize_components()
            
            # Set defaults
            if symbols is None:
                symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
            
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Get market data
            market_data = await self._prepare_market_data(symbols, start_date, end_date, data_source)
            
            # Run backtests for each strategy
            strategy_results = {}
            
            for i, strategy_config in enumerate(strategy_configs):
                strategy_name = strategy_config.get('name', f'Strategy_{i+1}')
                logger.info(f"Processing strategy: {strategy_name}")
                
                # Run backtest for each symbol
                symbol_results = {}
                
                for symbol in symbols:
                    logger.info(f"  Testing {symbol}...")
                    
                    if symbol not in market_data:
                        logger.warning(f"No data available for {symbol}")
                        continue
                    
                    # Single backtest
                    backtest_result = await self._run_single_backtest(
                        strategy_config, symbol, market_data[symbol], start_date, end_date
                    )
                    
                    if backtest_result:
                        # Analyze results
                        analysis_result = self.analyzer.analyze_single_strategy(backtest_result)
                        
                        # Optimize parameters if enabled
                        optimization_result = None
                        if self.enable_optimization:
                            optimization_result = await self._optimize_strategy(
                                strategy_config, symbol, market_data[symbol]
                            )
                        
                        # Store results
                        symbol_results[symbol] = {
                            'backtest': backtest_result,
                            'analysis': analysis_result,
                            'optimization': optimization_result
                        }
                
                strategy_results[strategy_name] = symbol_results
            
            # Cross-strategy analysis
            cross_analysis = self._perform_cross_strategy_analysis(strategy_results)
            
            # Stress testing
            stress_test_results = {}
            if self.enable_stress_testing:
                stress_test_results = await self._run_stress_tests(
                    strategy_configs, symbols, market_data
                )
            
            # Portfolio analysis
            portfolio_analysis = self._analyze_portfolio_combinations(strategy_results)
            
            # Generate comprehensive report
            comprehensive_results = {
                'metadata': {
                    'start_date': start_date,
                    'end_date': end_date,
                    'symbols': symbols,
                    'strategies': [cfg.get('name', f'Strategy_{i}') for i, cfg in enumerate(strategy_configs)],
                    'data_source': data_source,
                    'generated_at': datetime.now().isoformat()
                },
                'strategy_results': strategy_results,
                'cross_analysis': cross_analysis,
                'stress_test_results': stress_test_results,
                'portfolio_analysis': portfolio_analysis
            }
            
            # Save results
            results_path = await self._save_comprehensive_results(comprehensive_results)
            
            # Generate visualizations
            if self.enable_visualization:
                await self._generate_visualizations(comprehensive_results, market_data)
            
            # Send completion notification
            if NOTIFIER_AVAILABLE:
                send_info(f"✅ Comprehensive backtest completed!\nResults saved to: {results_path}")
            
            logger.info(f"Comprehensive backtest completed. Results saved to: {results_path}")
            
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive backtest: {e}")
            if NOTIFIER_AVAILABLE:
                send_error(f"❌ Comprehensive backtest failed: {e}")
            raise
    
    async def _prepare_market_data(self, symbols: List[str], start_date: str, 
                                 end_date: str, data_source: str) -> Dict[str, pd.DataFrame]:
        """Prepare market data for backtesting"""
        try:
            market_data = {}
            
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            if data_source == 'synthetic':
                # Generate synthetic data
                logger.info("Generating synthetic market data...")
                
                synthetic_data = self.market_simulator.generate_synthetic_data(
                    symbols=symbols,
                    start_date=start_dt,
                    end_date=end_dt,
                    frequency='1h'
                )
                
                return synthetic_data
            
            elif data_source == 'historical':
                # Load historical data
                logger.info("Loading historical market data...")
                
                for symbol in symbols:
                    try:
                        # Try to load data using data manager
                        data = await self.data_manager.get_historical_data(
                            symbol=symbol,
                            start_date=start_date,
                            end_date=end_date,
                            interval='1h'
                        )
                        
                        if data is not None and not data.empty:
                            market_data[symbol] = data
                            logger.info(f"Loaded {len(data)} data points for {symbol}")
                        else:
                            logger.warning(f"No historical data available for {symbol}")
                    
                    except Exception as e:
                        logger.error(f"Error loading data for {symbol}: {e}")
                
                return market_data
            
            else:
                raise ValueError(f"Unknown data source: {data_source}")
                
        except Exception as e:
            logger.error(f"Error preparing market data: {e}")
            return {}
    
    async def _run_single_backtest(self, strategy_config: Dict[str, Any], 
                                 symbol: str, data: pd.DataFrame,
                                 start_date: str, end_date: str) -> Optional[Dict[str, Any]]:
        """Run single backtest for strategy and symbol"""
        try:
            # Create strategy instance
            strategy_class_name = strategy_config.get('class', 'MomentumStrategy')
            strategy_params = strategy_config.get('params', {})
            
            # Map strategy class names to classes
            strategy_classes = {
                'MomentumStrategy': MomentumStrategy,
                'MeanReversionStrategy': MeanReversionStrategy,
                'ScalpingStrategy': ScalpingStrategy
            }
            
            if strategy_class_name not in strategy_classes:
                logger.error(f"Unknown strategy class: {strategy_class_name}")
                return None
            
            strategy_class = strategy_classes[strategy_class_name]
            strategy = strategy_class(strategy_params)
            
            # Run backtest
            result = self.backtest_engine.run_backtest(
                strategy=strategy,
                symbol=symbol,
                start_date=datetime.strptime(start_date, '%Y-%m-%d'),
                end_date=datetime.strptime(end_date, '%Y-%m-%d'),
                data=data
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in single backtest for {symbol}: {e}")
            return None
    
    async def _optimize_strategy(self, strategy_config: Dict[str, Any], 
                               symbol: str, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Optimize strategy parameters"""
        try:
            if not self.optimizer:
                return None
            
            # Get strategy class
            strategy_class_name = strategy_config.get('class', 'MomentumStrategy')
            strategy_classes = {
                'MomentumStrategy': MomentumStrategy,
                'MeanReversionStrategy': MeanReversionStrategy,
                'ScalpingStrategy': ScalpingStrategy
            }
            
            if strategy_class_name not in strategy_classes:
                return None
            
            strategy_class = strategy_classes[strategy_class_name]
            
            # Define optimization parameters based on strategy
            if strategy_class_name == 'MomentumStrategy':
                parameter_ranges = [
                    ParameterRange('lookback_period', 5, 50, 5, param_type=int),
                    ParameterRange('momentum_threshold', 0.01, 0.1, 0.01),
                    ParameterRange('stop_loss', 0.02, 0.1, 0.01),
                    ParameterRange('take_profit', 0.03, 0.15, 0.01)
                ]
            elif strategy_class_name == 'MeanReversionStrategy':
                parameter_ranges = [
                    ParameterRange('lookback_period', 10, 100, 10, param_type=int),
                    ParameterRange('std_multiplier', 1.0, 3.0, 0.5),
                    ParameterRange('min_periods', 5, 30, 5, param_type=int)
                ]
            else:  # ScalpingStrategy
                parameter_ranges = [
                    ParameterRange('fast_period', 3, 15, 2, param_type=int),
                    ParameterRange('slow_period', 10, 50, 5, param_type=int),
                    ParameterRange('signal_threshold', 0.001, 0.01, 0.001)
                ]
            
            # Run optimization
            logger.info(f"Optimizing {strategy_class_name} for {symbol}...")
            
            optimization_summary = await self.optimizer.optimize_strategy(
                strategy_class=strategy_class,
                parameter_ranges=parameter_ranges,
                data=data,
                method=OptimizationMethod.GRID_SEARCH,
                objective=OptimizationObjective.SHARPE_RATIO,
                symbol=symbol
            )
            
            return optimization_summary.to_dict() if optimization_summary else None
            
        except Exception as e:
            logger.error(f"Error optimizing strategy for {symbol}: {e}")
            return None
    
    def _perform_cross_strategy_analysis(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform cross-strategy comparison analysis"""
        try:
            logger.info("Performing cross-strategy analysis...")
            
            # Collect all strategy analyses
            all_analyses = []
            strategy_names = []
            
            for strategy_name, symbol_results in strategy_results.items():
                for symbol, results in symbol_results.items():
                    if 'analysis' in results and results['analysis']:
                        analysis = results['analysis'].copy()
                        analysis['strategy_name'] = strategy_name
                        analysis['symbol'] = symbol
                        all_analyses.append(analysis)
                        strategy_names.append(f"{strategy_name}_{symbol}")
            
            if not all_analyses:
                return {}
            
            # Compare performance metrics
            comparison_metrics = {}
            metric_keys = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']
            
            for metric in metric_keys:
                metric_values = []
                for analysis in all_analyses:
                    perf_metrics = analysis.get('performance_metrics', {})
                    metric_values.append(perf_metrics.get(metric, 0))
                
                comparison_metrics[metric] = {
                    'values': dict(zip(strategy_names, metric_values)),
                    'best': max(metric_values) if metric != 'max_drawdown' else min(metric_values),
                    'worst': min(metric_values) if metric != 'max_drawdown' else max(metric_values),
                    'average': sum(metric_values) / len(metric_values) if metric_values else 0
                }
            
            # Strategy rankings
            rankings = {}
            for metric in metric_keys:
                values = comparison_metrics[metric]['values']
                if metric == 'max_drawdown':
                    # Lower is better for drawdown
                    sorted_strategies = sorted(values.items(), key=lambda x: x[1])
                else:
                    # Higher is better for other metrics
                    sorted_strategies = sorted(values.items(), key=lambda x: x[1], reverse=True)
                
                rankings[metric] = [{'strategy': name, 'value': value, 'rank': i+1} 
                                  for i, (name, value) in enumerate(sorted_strategies)]
            
            # Overall score (weighted combination)
            weights = {
                'total_return': 0.25,
                'sharpe_ratio': 0.30,
                'max_drawdown': 0.20,  # Inverted (lower is better)
                'win_rate': 0.15,
                'profit_factor': 0.10
            }
            
            overall_scores = {}
            for strategy_name in strategy_names:
                score = 0
                for metric, weight in weights.items():
                    value = comparison_metrics[metric]['values'][strategy_name]
                    if metric == 'max_drawdown':
                        # Invert drawdown (lower is better)
                        normalized_value = 1 - (value / max(comparison_metrics[metric]['values'].values()))
                    else:
                        max_value = max(comparison_metrics[metric]['values'].values())
                        normalized_value = value / max_value if max_value > 0 else 0
                    
                    score += normalized_value * weight
                
                overall_scores[strategy_name] = score
            
            overall_ranking = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'comparison_metrics': comparison_metrics,
                'metric_rankings': rankings,
                'overall_ranking': [{'strategy': name, 'score': score, 'rank': i+1} 
                                  for i, (name, score) in enumerate(overall_ranking)]
            }
            
        except Exception as e:
            logger.error(f"Error in cross-strategy analysis: {e}")
            return {}
    
    async def _run_stress_tests(self, strategy_configs: List[Dict[str, Any]], 
                              symbols: List[str], market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run stress tests with different market scenarios"""
        try:
            if not self.market_simulator:
                return {}
            
            logger.info("Running stress tests...")
            
            stress_scenarios = ['flash_crash', 'regulatory_shock', 'extreme_volatility']
            stress_results = {}
            
            for scenario in stress_scenarios:
                logger.info(f"  Testing scenario: {scenario}")
                scenario_results = {}
                
                # Generate stressed market data
                stressed_data = self.market_simulator.add_market_stress_scenarios(
                    market_data, scenario
                )
                
                # Test each strategy
                for i, strategy_config in enumerate(strategy_configs):
                    strategy_name = strategy_config.get('name', f'Strategy_{i+1}')
                    strategy_scenario_results = {}
                    
                    for symbol in symbols:
                        if symbol not in stressed_data:
                            continue
                        
                        # Run backtest with stressed data
                        backtest_result = await self._run_single_backtest(
                            strategy_config, symbol, stressed_data[symbol],
                            stressed_data[symbol].index[0].strftime('%Y-%m-%d'),
                            stressed_data[symbol].index[-1].strftime('%Y-%m-%d')
                        )
                        
                        if backtest_result:
                            # Quick analysis
                            analysis = self.analyzer.analyze_single_strategy(backtest_result)
                            strategy_scenario_results[symbol] = {
                                'backtest': backtest_result,
                                'analysis': analysis
                            }
                    
                    scenario_results[strategy_name] = strategy_scenario_results
                
                stress_results[scenario] = scenario_results
            
            return stress_results
            
        except Exception as e:
            logger.error(f"Error in stress testing: {e}")
            return {}
    
    def _analyze_portfolio_combinations(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze portfolio combinations of strategies"""
        try:
            logger.info("Analyzing portfolio combinations...")
            
            # Extract equity curves for combination analysis
            strategy_equity_curves = {}
            
            for strategy_name, symbol_results in strategy_results.items():
                for symbol, results in symbol_results.items():
                    backtest = results.get('backtest', {})
                    if 'equity_curve' in backtest:
                        equity_df = pd.DataFrame(backtest['equity_curve'])
                        if not equity_df.empty:
                            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
                            equity_df.set_index('timestamp', inplace=True)
                            strategy_equity_curves[f"{strategy_name}_{symbol}"] = equity_df['equity']
            
            if len(strategy_equity_curves) < 2:
                return {}
            
            # Calculate correlation matrix
            combined_df = pd.DataFrame(strategy_equity_curves)
            returns_df = combined_df.pct_change().dropna()
            correlation_matrix = returns_df.corr()
            
            # Portfolio combinations (equal weight)
            portfolio_combinations = []
            strategy_names = list(strategy_equity_curves.keys())
            
            # All possible pairs
            for i in range(len(strategy_names)):
                for j in range(i+1, len(strategy_names)):
                    strat1, strat2 = strategy_names[i], strategy_names[j]
                    
                    # Equal weight portfolio
                    portfolio_equity = (combined_df[strat1] + combined_df[strat2]) / 2
                    portfolio_returns = portfolio_equity.pct_change().dropna()
                    
                    # Calculate portfolio metrics
                    total_return = (portfolio_equity.iloc[-1] - portfolio_equity.iloc[0]) / portfolio_equity.iloc[0]
                    volatility = portfolio_returns.std() * np.sqrt(252)
                    sharpe_ratio = (portfolio_returns.mean() * 252 - 0.02) / volatility if volatility > 0 else 0
                    
                    # Max drawdown
                    peak = portfolio_equity.expanding().max()
                    drawdown = (portfolio_equity - peak) / peak
                    max_drawdown = drawdown.min()
                    
                    portfolio_combinations.append({
                        'strategies': [strat1, strat2],
                        'correlation': correlation_matrix.loc[strat1, strat2],
                        'total_return': total_return,
                        'volatility': volatility,
                        'sharpe_ratio': sharpe_ratio,
                        'max_drawdown': max_drawdown
                    })
            
            # Sort by Sharpe ratio
            portfolio_combinations.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
            
            return {
                'correlation_matrix': correlation_matrix.to_dict(),
                'portfolio_combinations': portfolio_combinations[:10],  # Top 10
                'diversification_analysis': {
                    'avg_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean(),
                    'min_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].min(),
                    'max_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].max()
                }
            }
            
        except Exception as e:
            logger.error(f"Error in portfolio analysis: {e}")
            return {}
    
    async def _save_comprehensive_results(self, results: Dict[str, Any]) -> str:
        """Save comprehensive results to files"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_dir = Path(self.output_dir) / f"comprehensive_backtest_{timestamp}"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save main results as JSON
            results_file = results_dir / "comprehensive_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save summary report
            summary_file = results_dir / "executive_summary.txt"
            with open(summary_file, 'w') as f:
                f.write(self._generate_executive_summary(results))
            
            # Save detailed analysis for each strategy
            for strategy_name, symbol_results in results['strategy_results'].items():
                strategy_dir = results_dir / f"strategy_{strategy_name}"
                strategy_dir.mkdir(exist_ok=True)
                
                for symbol, symbol_data in symbol_results.items():
                    symbol_file = strategy_dir / f"{symbol.replace('/', '_')}_results.json"
                    with open(symbol_file, 'w') as f:
                        json.dump(symbol_data, f, indent=2, default=str)
            
            logger.info(f"Results saved to {results_dir}")
            return str(results_dir)
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return ""
    
    def _generate_executive_summary(self, results: Dict[str, Any]) -> str:
        """Generate executive summary of results"""
        try:
            summary = []
            summary.append("=" * 80)
            summary.append("COMPREHENSIVE BACKTEST EXECUTIVE SUMMARY")
            summary.append("=" * 80)
            summary.append("")
            
            # Metadata
            metadata = results.get('metadata', {})
            summary.append(f"Test Period: {metadata.get('start_date')} to {metadata.get('end_date')}")
            summary.append(f"Symbols Tested: {', '.join(metadata.get('symbols', []))}")
            summary.append(f"Strategies Tested: {', '.join(metadata.get('strategies', []))}")
            summary.append(f"Generated: {metadata.get('generated_at', 'Unknown')}")
            summary.append("")
            
            # Cross-strategy analysis
            cross_analysis = results.get('cross_analysis', {})
            if cross_analysis and 'overall_ranking' in cross_analysis:
                summary.append("STRATEGY RANKINGS (Overall Performance Score):")
                summary.append("-" * 50)
                for ranking in cross_analysis['overall_ranking'][:5]:  # Top 5
                    summary.append(f"{ranking['rank']}. {ranking['strategy']}: {ranking['score']:.3f}")
                summary.append("")
            
            # Best performers by metric
            if cross_analysis and 'comparison_metrics' in cross_analysis:
                metrics = cross_analysis['comparison_metrics']
                summary.append("BEST PERFORMERS BY METRIC:")
                summary.append("-" * 30)
                
                for metric, data in metrics.items():
                    best_strategy = max(data['values'].items(), 
                                      key=lambda x: x[1] if metric != 'max_drawdown' else -x[1])
                    summary.append(f"{metric.replace('_', ' ').title()}: {best_strategy[0]} ({best_strategy[1]:.4f})")
                summary.append("")
            
            # Portfolio analysis
            portfolio_analysis = results.get('portfolio_analysis', {})
            if portfolio_analysis and 'portfolio_combinations' in portfolio_analysis:
                summary.append("BEST PORTFOLIO COMBINATIONS:")
                summary.append("-" * 35)
                for i, combo in enumerate(portfolio_analysis['portfolio_combinations'][:3]):
                    summary.append(f"{i+1}. {' + '.join(combo['strategies'])}")
                    summary.append(f"   Sharpe: {combo['sharpe_ratio']:.3f}, Correlation: {combo['correlation']:.3f}")
                summary.append("")
            
            # Stress test summary
            stress_results = results.get('stress_test_results', {})
            if stress_results:
                summary.append("STRESS TEST SCENARIOS:")
                summary.append("-" * 25)
                for scenario in stress_results.keys():
                    summary.append(f"✓ {scenario.replace('_', ' ').title()}")
                summary.append("")
            
            summary.append("=" * 80)
            summary.append("END OF SUMMARY")
            summary.append("=" * 80)
            
            return "\n".join(summary)
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return "Error generating summary"
    
    async def _generate_visualizations(self, results: Dict[str, Any], 
                                     market_data: Dict[str, pd.DataFrame]):
        """Generate comprehensive visualizations"""
        try:
            if not self.visualizer:
                return
            
            logger.info("Generating visualizations...")
            
            # Create visualizations for each strategy-symbol combination
            for strategy_name, symbol_results in results['strategy_results'].items():
                for symbol, symbol_data in symbol_results.items():
                    backtest_result = symbol_data.get('backtest')
                    analysis_result = symbol_data.get('analysis')
                    
                    if backtest_result and analysis_result:
                        # Create strategy-specific visualization directory
                        viz_dir = Path(self.output_dir) / "visualizations" / strategy_name
                        viz_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Generate comprehensive report
                        symbol_clean = symbol.replace('/', '_')
                        report_path = viz_dir / f"{symbol_clean}_report"
                        
                        self.visualizer.create_comprehensive_report(
                            backtest_results=backtest_result,
                            analysis_results=analysis_result,
                            market_data=market_data.get(symbol),
                            save_path=str(report_path)
                        )
            
            logger.info("Visualizations generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")


def get_default_strategy_configs() -> List[Dict[str, Any]]:
    """Get default strategy configurations for testing"""
    return [
        {
            'name': 'Momentum_Conservative',
            'class': 'MomentumStrategy',
            'params': {
                'lookback_period': 20,
                'momentum_threshold': 0.02,
                'stop_loss': 0.05,
                'take_profit': 0.08
            }
        },
        {
            'name': 'Momentum_Aggressive',
            'class': 'MomentumStrategy',
            'params': {
                'lookback_period': 10,
                'momentum_threshold': 0.015,
                'stop_loss': 0.03,
                'take_profit': 0.12
            }
        },
        {
            'name': 'MeanReversion_Standard',
            'class': 'MeanReversionStrategy',
            'params': {
                'lookback_period': 50,
                'std_multiplier': 2.0,
                'min_periods': 20
            }
        }
    ]


async def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Run comprehensive backtesting suite')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--symbols', nargs='+', default=['BTC/USDT', 'ETH/USDT'], 
                       help='Symbols to test')
    parser.add_argument('--start-date', type=str, 
                       default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, 
                       default=datetime.now().strftime('%Y-%m-%d'),
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--data-source', choices=['historical', 'synthetic'], 
                       default='historical', help='Data source')
    parser.add_argument('--strategies-config', type=str, 
                       help='JSON file with strategy configurations')
    parser.add_argument('--output-dir', type=str, 
                       default='data/comprehensive_backtest_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = {}
        if args.config and Path(args.config).exists():
            with open(args.config, 'r') as f:
                config = json.load(f)
        
        # Override with command line arguments
        config['output_dir'] = args.output_dir
        config['enable_synthetic_data'] = (args.data_source == 'synthetic')
        
        # Load strategy configurations
        if args.strategies_config and Path(args.strategies_config).exists():
            with open(args.strategies_config, 'r') as f:
                strategy_configs = json.load(f)
        else:
            strategy_configs = get_default_strategy_configs()
        
        # Create and run comprehensive backtest
        backtester = ComprehensiveBacktester(config)
        
        results = await backtester.run_comprehensive_backtest(
            strategy_configs=strategy_configs,
            symbols=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            data_source=args.data_source
        )
        
        print(f"\n✅ Comprehensive backtest completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
        # Print quick summary
        cross_analysis = results.get('cross_analysis', {})
        if cross_analysis and 'overall_ranking' in cross_analysis:
            print(f"\nTop performing strategy: {cross_analysis['overall_ranking'][0]['strategy']}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())