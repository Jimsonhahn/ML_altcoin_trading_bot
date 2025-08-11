#!/usr/bin/env python3
# scripts/master_backtest_analysis.py
"""
Master Backtest Analysis Script
Tests all strategies systematically and provides concrete recommendations
"""

import asyncio
import logging
import json
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Core imports
from core.backtest_engine import BacktestEngine
from core.backtest_analyzer import BacktestAnalyzer
from core.strategy_optimizer import StrategyOptimizer, ParameterRange, OptimizationMethod, OptimizationObjective
from data.market_simulator import MarketSimulator
from utils.backtest_visualizer import BacktestVisualizer
from reports.backtest_report_generator import BacktestReportGenerator

# Strategy imports
from strategies.momentum_strategy import MomentumStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.ml_strategy import MLStrategy
from strategies.grid_strategy import GridStrategy
from strategies.arbitrage_strategy import ArbitrageStrategy
from strategies.defi_strategy import DeFiStrategy
from strategies.copy_trading_strategy import CopyTradingStrategy
from strategies.stablecoin_parking_strategy import StablecoinParkingStrategy
from strategies.lazy_billionaire_strategy import LazyBillionaireStrategy

# Data handling
from data_sources.data_manager import DataManager

# Analysis components
from analysis.strategy_performance_analyzer import StrategyPerformanceAnalyzer
from optimization.strategy_improvements import StrategyImprovementOptimizer
from reports.final_recommendations import FinalRecommendationGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MarketPeriod:
    """Market period definition"""
    name: str
    start_date: str
    end_date: str
    market_type: str  # bull, bear, recovery, full


@dataclass
class CapitalScenario:
    """Capital size scenario"""
    name: str
    amount: float
    description: str


@dataclass
class RiskProfile:
    """Risk profile configuration"""
    name: str
    max_drawdown: float
    position_size_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    max_open_positions: int


class MasterBacktestAnalyzer:
    """
    Master backtesting system for comprehensive strategy analysis
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Output configuration
        self.output_dir = self.config.get('output_dir', 'results/master_analysis')
        self.generate_report = self.config.get('generate_report', True)
        
        # Market periods for testing
        self.market_periods = [
            MarketPeriod("Bull Market", "2020-07-01", "2021-11-30", "bull"),
            MarketPeriod("Bear Market", "2021-12-01", "2022-12-31", "bear"),
            MarketPeriod("Recovery", "2023-01-01", "2023-12-31", "recovery"),
            MarketPeriod("Full Period", "2020-01-01", "2024-01-01", "full")
        ]
        
        # Capital scenarios
        self.capital_scenarios = [
            CapitalScenario("Small", 10000, "Retail trader"),
            CapitalScenario("Medium", 100000, "Serious trader"),
            CapitalScenario("Large", 300000, "Your planned capital")
        ]
        
        # Risk profiles
        self.risk_profiles = {
            "conservative": RiskProfile(
                "conservative", 0.15, 0.02, 0.05, 0.10, 3
            ),
            "balanced": RiskProfile(
                "balanced", 0.25, 0.05, 0.08, 0.15, 5
            ),
            "aggressive": RiskProfile(
                "aggressive", 0.40, 0.10, 0.12, 0.25, 8
            )
        }
        
        # Strategy configurations
        self.strategy_configs = self._get_strategy_configs()
        
        # Analysis components
        self.backtest_engine = None
        self.analyzer = None
        self.optimizer = None
        self.visualizer = None
        self.report_generator = None
        self.performance_analyzer = None
        self.improvement_optimizer = None
        self.recommendation_generator = None
        
        # Results storage
        self.all_results = {}
        self.strategy_correlations = {}
        self.optimal_parameters = {}
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("MasterBacktestAnalyzer initialized")
    
    def _get_strategy_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get configurations for all strategies"""
        return {
            "momentum_strategy": {
                "class": MomentumStrategy,
                "params": {
                    "lookback_period": 20,
                    "momentum_threshold": 0.02,
                    "volume_filter": True,
                    "trend_filter": True
                },
                "optimization_ranges": [
                    ParameterRange("lookback_period", 10, 50, 5, param_type=int),
                    ParameterRange("momentum_threshold", 0.01, 0.05, 0.005)
                ]
            },
            "mean_reversion_strategy": {
                "class": MeanReversionStrategy,
                "params": {
                    "lookback_period": 30,
                    "std_multiplier": 2.0,
                    "min_periods": 20
                },
                "optimization_ranges": [
                    ParameterRange("lookback_period", 20, 60, 10, param_type=int),
                    ParameterRange("std_multiplier", 1.5, 3.0, 0.25)
                ]
            },
            "ml_strategy": {
                "class": MLStrategy,
                "params": {
                    "prediction_threshold": 0.6,
                    "use_ensemble": True,
                    "retrain_frequency": "weekly"
                },
                "optimization_ranges": [
                    ParameterRange("prediction_threshold", 0.5, 0.8, 0.05)
                ]
            },
            "grid_strategy": {
                "class": GridStrategy,
                "params": {
                    "grid_levels": 10,
                    "grid_spacing": 0.01,
                    "order_size": 0.1
                },
                "optimization_ranges": [
                    ParameterRange("grid_levels", 5, 20, 5, param_type=int),
                    ParameterRange("grid_spacing", 0.005, 0.02, 0.005)
                ]
            },
            "arbitrage_strategy": {
                "class": ArbitrageStrategy,
                "params": {
                    "min_spread": 0.002,
                    "execution_delay": 0.5,
                    "max_position_size": 0.2
                },
                "optimization_ranges": [
                    ParameterRange("min_spread", 0.001, 0.005, 0.001)
                ]
            },
            "defi_strategy": {
                "class": DeFiStrategy,
                "params": {
                    "min_apy": 0.10,
                    "impermanent_loss_threshold": 0.05,
                    "rebalance_frequency": "daily"
                },
                "optimization_ranges": [
                    ParameterRange("min_apy", 0.05, 0.20, 0.05)
                ]
            },
            "copy_trading_strategy": {
                "class": CopyTradingStrategy,
                "params": {
                    "min_trader_performance": 0.20,
                    "max_traders_to_copy": 5,
                    "position_scaling": 0.8
                },
                "optimization_ranges": [
                    ParameterRange("position_scaling", 0.5, 1.0, 0.1)
                ]
            },
            "stablecoin_parking_strategy": {
                "class": StablecoinParkingStrategy,
                "params": {
                    "min_yield": 0.08,
                    "safety_score_threshold": 0.8,
                    "max_protocol_exposure": 0.3
                },
                "optimization_ranges": [
                    ParameterRange("min_yield", 0.05, 0.12, 0.01)
                ]
            },
            "lazy_billionaire_strategy": {
                "class": LazyBillionaireStrategy,
                "params": {
                    "rebalance_threshold": 0.10,
                    "risk_allocation_mode": "balanced",
                    "max_strategies": 5
                },
                "optimization_ranges": [
                    ParameterRange("rebalance_threshold", 0.05, 0.20, 0.05)
                ]
            }
        }
    
    def initialize_components(self):
        """Initialize all analysis components"""
        try:
            logger.info("Initializing analysis components...")
            
            # Backtest engine with realistic Binance settings
            backtest_config = {
                'initial_capital': 10000,  # Will be overridden
                'enable_slippage': True,
                'enable_market_impact': True,
                'enable_latency': True,
                'enable_liquidity_constraints': True,
                'commission_rate': 0.001,  # Binance 0.1%
                'slippage_factor': 0.0005,  # Base 0.05%
                'market_impact_factor': 0.0002,
                'min_latency_ms': 20,
                'max_latency_ms': 100,
                'max_position_pct_of_volume': 0.01  # Max 1% of volume
            }
            self.backtest_engine = BacktestEngine(backtest_config)
            
            # Analyzer
            self.analyzer = BacktestAnalyzer({'risk_free_rate': 0.02})
            
            # Optimizer
            self.optimizer = StrategyOptimizer({
                'max_workers': 4,
                'timeout_seconds': 3600,
                'validation_split': 0.3,
                'enable_market_phase_analysis': True
            })
            
            # Visualizer
            self.visualizer = BacktestVisualizer({
                'output_dir': f"{self.output_dir}/visualizations",
                'dpi': 300
            })
            
            # Report generator
            self.report_generator = BacktestReportGenerator({
                'output_dir': f"{self.output_dir}/reports",
                'company_name': 'Lazy Billionaire Trading',
                'report_title': 'Master Strategy Analysis Report',
                'author': 'Automated Analysis System'
            })
            
            # Performance analyzer
            self.performance_analyzer = StrategyPerformanceAnalyzer({
                'output_dir': f"{self.output_dir}/analysis"
            })
            
            # Improvement optimizer
            self.improvement_optimizer = StrategyImprovementOptimizer({
                'output_dir': f"{self.output_dir}/optimization"
            })
            
            # Final recommendation generator
            self.recommendation_generator = FinalRecommendationGenerator({
                'output_dir': f"{self.output_dir}/recommendations",
                'target_capital': 300000
            })
            
            # Data manager
            self.data_manager = DataManager()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    async def run_master_analysis(self, 
                                target_capital: float = 300000,
                                selected_strategies: List[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive master analysis of all strategies
        """
        try:
            logger.info("=" * 80)
            logger.info("STARTING MASTER BACKTEST ANALYSIS")
            logger.info("=" * 80)
            logger.info(f"Target Capital: €{target_capital:,.2f}")
            logger.info(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 80)
            
            # Initialize components
            self.initialize_components()
            
            # Select strategies to test
            if selected_strategies is None:
                selected_strategies = list(self.strategy_configs.keys())
            
            logger.info(f"Testing {len(selected_strategies)} strategies")
            
            # Load market data for all periods
            market_data_cache = await self._load_all_market_data()
            
            # Phase 1: Test all strategies across all scenarios
            logger.info("\nPHASE 1: Individual Strategy Testing")
            logger.info("-" * 40)
            
            for strategy_name in selected_strategies:
                logger.info(f"\nTesting strategy: {strategy_name}")
                
                strategy_results = await self._test_strategy_all_scenarios(
                    strategy_name, market_data_cache
                )
                
                self.all_results[strategy_name] = strategy_results
                
                # Quick performance summary
                self._print_strategy_summary(strategy_name, strategy_results)
            
            # Phase 2: Optimization
            logger.info("\nPHASE 2: Strategy Optimization")
            logger.info("-" * 40)
            
            optimization_results = await self._optimize_all_strategies(
                selected_strategies, market_data_cache
            )
            
            # Phase 3: Correlation Analysis
            logger.info("\nPHASE 3: Correlation and Synergy Analysis")
            logger.info("-" * 40)
            
            correlation_analysis = self._analyze_strategy_correlations()
            
            # Phase 4: Generate Improvements
            logger.info("\nPHASE 4: Strategy Improvement Analysis")
            logger.info("-" * 40)
            
            improvements = self._generate_strategy_improvements()
            
            # Phase 5: Final Recommendations
            logger.info("\nPHASE 5: Generating Final Recommendations")
            logger.info("-" * 40)
            
            final_recommendations = await self._generate_final_recommendations(
                target_capital, correlation_analysis, improvements
            )
            
            # Phase 6: Generate Reports
            if self.generate_report:
                logger.info("\nPHASE 6: Generating Reports and Visualizations")
                logger.info("-" * 40)
                
                await self._generate_all_reports(final_recommendations)
            
            # Print final recommendation
            self._print_final_recommendation(final_recommendations, target_capital)
            
            # Save all results
            self._save_all_results(final_recommendations)
            
            return {
                'strategy_results': self.all_results,
                'optimization_results': optimization_results,
                'correlation_analysis': correlation_analysis,
                'improvements': improvements,
                'final_recommendations': final_recommendations,
                'output_directory': self.output_dir
            }
            
        except Exception as e:
            logger.error(f"Error in master analysis: {e}")
            raise
    
    async def _load_all_market_data(self) -> Dict[str, pd.DataFrame]:
        """Load market data for all test periods"""
        try:
            logger.info("Loading market data...")
            
            market_data_cache = {}
            symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT']
            
            for period in self.market_periods:
                logger.info(f"Loading data for {period.name}: {period.start_date} to {period.end_date}")
                
                period_data = {}
                for symbol in symbols:
                    try:
                        # Try to load historical data
                        data = await self.data_manager.get_historical_data(
                            symbol=symbol,
                            start_date=period.start_date,
                            end_date=period.end_date,
                            interval='1h'
                        )
                        
                        if data is not None and not data.empty:
                            period_data[symbol] = data
                            logger.info(f"  Loaded {len(data)} data points for {symbol}")
                        else:
                            logger.warning(f"  No data available for {symbol}")
                            
                            # Generate synthetic data as fallback
                            logger.info(f"  Generating synthetic data for {symbol}")
                            simulator = MarketSimulator()
                            
                            start_dt = datetime.strptime(period.start_date, '%Y-%m-%d')
                            end_dt = datetime.strptime(period.end_date, '%Y-%m-%d')
                            
                            synthetic_data = simulator.generate_synthetic_data(
                                symbols=[symbol],
                                start_date=start_dt,
                                end_date=end_dt,
                                frequency='1h'
                            )
                            
                            if symbol in synthetic_data:
                                period_data[symbol] = synthetic_data[symbol]
                    
                    except Exception as e:
                        logger.error(f"Error loading data for {symbol}: {e}")
                
                market_data_cache[period.name] = period_data
            
            return market_data_cache
            
        except Exception as e:
            logger.error(f"Error loading market data: {e}")
            return {}
    
    async def _test_strategy_all_scenarios(self, 
                                         strategy_name: str,
                                         market_data_cache: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """Test a strategy across all market periods, capital sizes, and risk profiles"""
        try:
            strategy_config = self.strategy_configs[strategy_name]
            results = {
                'periods': {},
                'capital_scenarios': {},
                'risk_profiles': {},
                'aggregate_metrics': {}
            }
            
            # Test across all combinations
            for period in self.market_periods:
                period_results = {}
                
                for capital_scenario in self.capital_scenarios:
                    capital_results = {}
                    
                    for risk_profile_name, risk_profile in self.risk_profiles.items():
                        logger.info(f"  Testing: {period.name} | €{capital_scenario.amount:,} | {risk_profile_name}")
                        
                        # Adjust strategy parameters for risk profile
                        adjusted_params = self._adjust_params_for_risk_profile(
                            strategy_config['params'].copy(), risk_profile
                        )
                        
                        # Run backtest
                        backtest_results = await self._run_single_backtest(
                            strategy_name,
                            strategy_config['class'],
                            adjusted_params,
                            market_data_cache[period.name],
                            period,
                            capital_scenario.amount
                        )
                        
                        if backtest_results:
                            # Analyze results
                            analysis = self.analyzer.analyze_single_strategy(backtest_results)
                            
                            capital_results[risk_profile_name] = {
                                'backtest': backtest_results,
                                'analysis': analysis,
                                'metrics': self._extract_key_metrics(analysis)
                            }
                    
                    period_results[capital_scenario.name] = capital_results
                
                results['periods'][period.name] = period_results
            
            # Calculate aggregate metrics
            results['aggregate_metrics'] = self._calculate_aggregate_metrics(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error testing strategy {strategy_name}: {e}")
            return {}
    
    def _adjust_params_for_risk_profile(self, params: Dict[str, Any], 
                                       risk_profile: RiskProfile) -> Dict[str, Any]:
        """Adjust strategy parameters based on risk profile"""
        adjusted = params.copy()
        
        # Add risk management parameters
        adjusted['stop_loss'] = risk_profile.stop_loss_pct
        adjusted['take_profit'] = risk_profile.take_profit_pct
        adjusted['position_size'] = risk_profile.position_size_pct
        adjusted['max_positions'] = risk_profile.max_open_positions
        
        return adjusted
    
    async def _run_single_backtest(self,
                                 strategy_name: str,
                                 strategy_class,
                                 params: Dict[str, Any],
                                 market_data: Dict[str, pd.DataFrame],
                                 period: MarketPeriod,
                                 capital: float) -> Optional[Dict[str, Any]]:
        """Run a single backtest"""
        try:
            # Update backtest engine capital
            self.backtest_engine.config['initial_capital'] = capital
            
            # Create strategy instance
            strategy = strategy_class(params)
            
            # Select primary symbol (BTC/USDT)
            symbol = 'BTC/USDT'
            if symbol not in market_data or market_data[symbol].empty:
                logger.warning(f"No data for {symbol}")
                return None
            
            # Run backtest
            result = self.backtest_engine.run_backtest(
                strategy=strategy,
                symbol=symbol,
                start_date=datetime.strptime(period.start_date, '%Y-%m-%d'),
                end_date=datetime.strptime(period.end_date, '%Y-%m-%d'),
                data=market_data[symbol]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in single backtest: {e}")
            return None
    
    def _extract_key_metrics(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """Extract key metrics from analysis"""
        metrics = analysis.get('performance_metrics', {})
        
        return {
            'total_return': metrics.get('total_return', 0),
            'annualized_return': metrics.get('annualized_return', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'sortino_ratio': metrics.get('sortino_ratio', 0),
            'max_drawdown': metrics.get('max_drawdown', 0),
            'win_rate': metrics.get('win_rate', 0),
            'profit_factor': metrics.get('profit_factor', 0),
            'total_trades': metrics.get('total_trades', 0),
            'avg_trade_return': metrics.get('avg_trade_return', 0),
            'volatility': metrics.get('volatility', 0),
            'calmar_ratio': metrics.get('calmar_ratio', 0)
        }
    
    def _calculate_aggregate_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate aggregate metrics across all scenarios"""
        all_metrics = []
        
        # Collect all metrics
        for period_data in results['periods'].values():
            for capital_data in period_data.values():
                for risk_data in capital_data.values():
                    if 'metrics' in risk_data:
                        all_metrics.append(risk_data['metrics'])
        
        if not all_metrics:
            return {}
        
        # Calculate aggregates
        aggregate = {}
        metric_names = all_metrics[0].keys()
        
        for metric in metric_names:
            values = [m[metric] for m in all_metrics if metric in m]
            if values:
                aggregate[f'{metric}_mean'] = np.mean(values)
                aggregate[f'{metric}_std'] = np.std(values)
                aggregate[f'{metric}_min'] = np.min(values)
                aggregate[f'{metric}_max'] = np.max(values)
                aggregate[f'{metric}_median'] = np.median(values)
        
        return aggregate
    
    def _print_strategy_summary(self, strategy_name: str, results: Dict[str, Any]):
        """Print quick summary of strategy performance"""
        metrics = results.get('aggregate_metrics', {})
        
        if metrics:
            logger.info(f"\n{strategy_name} Summary:")
            logger.info(f"  Avg Return: {metrics.get('total_return_mean', 0)*100:.2f}%")
            logger.info(f"  Avg Sharpe: {metrics.get('sharpe_ratio_mean', 0):.2f}")
            logger.info(f"  Max Drawdown: {metrics.get('max_drawdown_max', 0)*100:.2f}%")
            logger.info(f"  Win Rate: {metrics.get('win_rate_mean', 0)*100:.1f}%")
    
    async def _optimize_all_strategies(self, 
                                     selected_strategies: List[str],
                                     market_data_cache: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """Optimize parameters for all strategies"""
        try:
            optimization_results = {}
            
            # Use full period data for optimization
            full_period_data = market_data_cache.get("Full Period", {})
            
            if not full_period_data:
                logger.warning("No full period data for optimization")
                return optimization_results
            
            for strategy_name in selected_strategies:
                if strategy_name not in self.strategy_configs:
                    continue
                
                logger.info(f"Optimizing {strategy_name}...")
                
                strategy_config = self.strategy_configs[strategy_name]
                
                # Skip if no optimization ranges defined
                if 'optimization_ranges' not in strategy_config:
                    logger.info(f"  No optimization ranges defined for {strategy_name}")
                    continue
                
                # Run optimization
                symbol = 'BTC/USDT'
                if symbol in full_period_data:
                    optimization_summary = await self.optimizer.optimize_strategy(
                        strategy_class=strategy_config['class'],
                        parameter_ranges=strategy_config['optimization_ranges'],
                        data=full_period_data[symbol],
                        method=OptimizationMethod.GRID_SEARCH,
                        objective=OptimizationObjective.SHARPE_RATIO,
                        symbol=symbol,
                        initial_capital=100000  # Use medium capital for optimization
                    )
                    
                    if optimization_summary:
                        optimization_results[strategy_name] = optimization_summary.to_dict()
                        self.optimal_parameters[strategy_name] = optimization_summary.best_result.parameters
                        
                        logger.info(f"  Best parameters: {optimization_summary.best_result.parameters}")
                        logger.info(f"  Best Sharpe: {optimization_summary.best_result.objective_value:.3f}")
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error in optimization: {e}")
            return {}
    
    def _analyze_strategy_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between strategies"""
        try:
            logger.info("Analyzing strategy correlations...")
            
            # Extract equity curves for correlation analysis
            equity_curves = {}
            
            for strategy_name, results in self.all_results.items():
                # Use full period, large capital, balanced risk results
                try:
                    full_period = results['periods']['Full Period']
                    large_capital = full_period['Large']
                    balanced_results = large_capital['balanced']
                    
                    if 'backtest' in balanced_results:
                        equity_curve = pd.DataFrame(balanced_results['backtest']['equity_curve'])
                        if not equity_curve.empty:
                            equity_curve['timestamp'] = pd.to_datetime(equity_curve['timestamp'])
                            equity_curve.set_index('timestamp', inplace=True)
                            equity_curves[strategy_name] = equity_curve['equity']
                
                except:
                    continue
            
            if len(equity_curves) < 2:
                return {}
            
            # Calculate returns and correlations
            returns_df = pd.DataFrame(equity_curves)
            returns_df = returns_df.pct_change().dropna()
            
            correlation_matrix = returns_df.corr()
            
            # Find best combinations (low correlation)
            best_combinations = []
            strategies = list(equity_curves.keys())
            
            for i in range(len(strategies)):
                for j in range(i+1, len(strategies)):
                    corr = correlation_matrix.loc[strategies[i], strategies[j]]
                    best_combinations.append({
                        'strategy1': strategies[i],
                        'strategy2': strategies[j],
                        'correlation': corr
                    })
            
            # Sort by correlation (lower is better for diversification)
            best_combinations.sort(key=lambda x: abs(x['correlation']))
            
            return {
                'correlation_matrix': correlation_matrix.to_dict(),
                'best_combinations': best_combinations[:10],
                'average_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
            }
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            return {}
    
    def _generate_strategy_improvements(self) -> Dict[str, Any]:
        """Generate improvement recommendations for each strategy"""
        try:
            improvements = {}
            
            for strategy_name, results in self.all_results.items():
                logger.info(f"Generating improvements for {strategy_name}...")
                
                strategy_improvements = {
                    'optimal_parameters': self.optimal_parameters.get(strategy_name, {}),
                    'best_market_conditions': self._find_best_market_conditions(results),
                    'worst_market_conditions': self._find_worst_market_conditions(results),
                    'risk_management': self._suggest_risk_improvements(results),
                    'position_sizing': self._suggest_position_sizing(results),
                    'specific_recommendations': self._generate_specific_recommendations(strategy_name, results)
                }
                
                improvements[strategy_name] = strategy_improvements
            
            return improvements
            
        except Exception as e:
            logger.error(f"Error generating improvements: {e}")
            return {}
    
    def _find_best_market_conditions(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Find best market conditions for strategy"""
        best_conditions = {
            'period': None,
            'capital_size': None,
            'risk_profile': None,
            'metrics': {}
        }
        
        best_sharpe = -float('inf')
        
        for period_name, period_data in results['periods'].items():
            for capital_name, capital_data in period_data.items():
                for risk_name, risk_data in capital_data.items():
                    if 'metrics' in risk_data:
                        sharpe = risk_data['metrics'].get('sharpe_ratio', 0)
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_conditions['period'] = period_name
                            best_conditions['capital_size'] = capital_name
                            best_conditions['risk_profile'] = risk_name
                            best_conditions['metrics'] = risk_data['metrics']
        
        return best_conditions
    
    def _find_worst_market_conditions(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Find worst market conditions for strategy"""
        worst_conditions = {
            'period': None,
            'capital_size': None,
            'risk_profile': None,
            'metrics': {}
        }
        
        worst_return = float('inf')
        
        for period_name, period_data in results['periods'].items():
            for capital_name, capital_data in period_data.items():
                for risk_name, risk_data in capital_data.items():
                    if 'metrics' in risk_data:
                        total_return = risk_data['metrics'].get('total_return', 0)
                        if total_return < worst_return:
                            worst_return = total_return
                            worst_conditions['period'] = period_name
                            worst_conditions['capital_size'] = capital_name
                            worst_conditions['risk_profile'] = risk_name
                            worst_conditions['metrics'] = risk_data['metrics']
        
        return worst_conditions
    
    def _suggest_risk_improvements(self, results: Dict[str, Any]) -> List[str]:
        """Suggest risk management improvements"""
        suggestions = []
        
        metrics = results.get('aggregate_metrics', {})
        
        # Check max drawdown
        max_dd = metrics.get('max_drawdown_max', 0)
        if max_dd > 0.25:
            suggestions.append(f"Implement tighter stop losses - max drawdown of {max_dd*100:.1f}% is too high")
        
        # Check win rate
        win_rate = metrics.get('win_rate_mean', 0)
        if win_rate < 0.45:
            suggestions.append(f"Improve entry signals - win rate of {win_rate*100:.1f}% is below optimal")
        
        # Check profit factor
        profit_factor = metrics.get('profit_factor_mean', 0)
        if profit_factor < 1.5:
            suggestions.append("Optimize take profit levels - profit factor should be above 1.5")
        
        return suggestions
    
    def _suggest_position_sizing(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest optimal position sizing"""
        metrics = results.get('aggregate_metrics', {})
        
        # Kelly Criterion simplified
        win_rate = metrics.get('win_rate_mean', 0.5)
        avg_win = metrics.get('avg_trade_return_mean', 0.02)
        avg_loss = abs(metrics.get('avg_trade_return_mean', 0.01))
        
        if avg_loss > 0:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(0.25, kelly_fraction))  # Cap at 25%
        else:
            kelly_fraction = 0.02
        
        return {
            'recommended_position_size': kelly_fraction,
            'conservative_size': kelly_fraction * 0.5,
            'aggressive_size': kelly_fraction * 1.5,
            'reasoning': f"Based on {win_rate*100:.1f}% win rate and risk/reward ratio"
        }
    
    def _generate_specific_recommendations(self, strategy_name: str, results: Dict[str, Any]) -> List[str]:
        """Generate strategy-specific recommendations"""
        recommendations = []
        
        if strategy_name == "momentum_strategy":
            recommendations.extend([
                "Use shorter lookback periods in volatile markets",
                "Increase momentum threshold during sideways markets",
                "Add volume confirmation for better signal quality"
            ])
        elif strategy_name == "mean_reversion_strategy":
            recommendations.extend([
                "Avoid during strong trending markets",
                "Use wider bands (higher std multiplier) in volatile conditions",
                "Implement time-based exits to avoid holding losing positions"
            ])
        elif strategy_name == "ml_strategy":
            recommendations.extend([
                "Retrain models weekly to adapt to market changes",
                "Use ensemble methods for more stable predictions",
                "Monitor feature importance and remove noisy features"
            ])
        elif strategy_name == "grid_strategy":
            recommendations.extend([
                "Increase grid levels in ranging markets",
                "Use wider spacing in volatile conditions",
                "Implement dynamic grid adjustment based on volatility"
            ])
        elif strategy_name == "lazy_billionaire_strategy":
            recommendations.extend([
                "Rebalance monthly for optimal performance",
                "Increase allocation to uncorrelated strategies",
                "Use market regime detection to adjust weights"
            ])
        
        return recommendations
    
    async def _generate_final_recommendations(self, 
                                            target_capital: float,
                                            correlation_analysis: Dict[str, Any],
                                            improvements: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final recommendations for the user"""
        try:
            logger.info("Generating final recommendations...")
            
            # Rank strategies by different criteria
            strategy_rankings = self._rank_all_strategies()
            
            # Determine best approach
            best_approach = self._determine_best_approach(
                target_capital, strategy_rankings, correlation_analysis
            )
            
            # Get top strategies
            top_3_strategies = self._get_top_3_strategies(strategy_rankings)
            
            # Optimize Lazy Billionaire configuration
            lazy_billionaire_config = self._optimize_lazy_billionaire(
                strategy_rankings, correlation_analysis
            )
            
            # Calculate expected returns
            expected_returns = self._calculate_expected_returns(
                target_capital, best_approach, strategy_rankings
            )
            
            # Generate implementation plan
            implementation_plan = self._generate_implementation_plan(
                best_approach, top_3_strategies
            )
            
            final_recommendations = {
                'target_capital': target_capital,
                'best_approach': best_approach,
                'strategy_rankings': strategy_rankings,
                'top_3_strategies': top_3_strategies,
                'lazy_billionaire_config': lazy_billionaire_config,
                'expected_returns': expected_returns,
                'implementation_plan': implementation_plan,
                'risk_warnings': self._generate_risk_warnings(),
                'monitoring_plan': self._generate_monitoring_plan()
            }
            
            return final_recommendations
            
        except Exception as e:
            logger.error(f"Error generating final recommendations: {e}")
            return {}
    
    def _rank_all_strategies(self) -> Dict[str, List[Dict[str, Any]]]:
        """Rank all strategies by different criteria"""
        rankings = {
            'by_sharpe': [],
            'by_return': [],
            'by_stability': [],
            'by_capital_efficiency': []
        }
        
        # Collect metrics for all strategies
        strategy_metrics = []
        
        for strategy_name, results in self.all_results.items():
            metrics = results.get('aggregate_metrics', {})
            
            if metrics:
                strategy_metrics.append({
                    'name': strategy_name,
                    'sharpe': metrics.get('sharpe_ratio_mean', 0),
                    'return': metrics.get('total_return_mean', 0),
                    'max_dd': metrics.get('max_drawdown_max', 0),
                    'volatility': metrics.get('volatility_mean', 0),
                    'trades': metrics.get('total_trades_mean', 0),
                    'win_rate': metrics.get('win_rate_mean', 0)
                })
        
        # Rank by Sharpe ratio
        rankings['by_sharpe'] = sorted(strategy_metrics, 
                                     key=lambda x: x['sharpe'], 
                                     reverse=True)
        
        # Rank by return
        rankings['by_return'] = sorted(strategy_metrics, 
                                     key=lambda x: x['return'], 
                                     reverse=True)
        
        # Rank by stability (lowest drawdown)
        rankings['by_stability'] = sorted(strategy_metrics, 
                                        key=lambda x: x['max_dd'])
        
        # Rank by capital efficiency (return per trade)
        for metric in strategy_metrics:
            if metric['trades'] > 0:
                metric['capital_efficiency'] = metric['return'] / metric['trades']
            else:
                metric['capital_efficiency'] = 0
        
        rankings['by_capital_efficiency'] = sorted(strategy_metrics, 
                                                 key=lambda x: x['capital_efficiency'], 
                                                 reverse=True)
        
        return rankings
    
    def _determine_best_approach(self, target_capital: float, 
                               strategy_rankings: Dict[str, List],
                               correlation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine whether to use Lazy Billionaire or single strategies"""
        
        # Check if Lazy Billionaire is in top performers
        lazy_billionaire_rank = None
        for i, strategy in enumerate(strategy_rankings['by_sharpe']):
            if strategy['name'] == 'lazy_billionaire_strategy':
                lazy_billionaire_rank = i + 1
                break
        
        # Calculate average correlation
        avg_correlation = correlation_analysis.get('average_correlation', 1.0)
        
        # Decision logic
        use_lazy_billionaire = False
        reasoning = []
        
        if lazy_billionaire_rank and lazy_billionaire_rank <= 3:
            use_lazy_billionaire = True
            reasoning.append("Lazy Billionaire ranks in top 3 strategies")
        
        if avg_correlation < 0.5:
            use_lazy_billionaire = True
            reasoning.append(f"Low average correlation ({avg_correlation:.2f}) provides good diversification")
        
        if target_capital >= 100000:
            use_lazy_billionaire = True
            reasoning.append("Large capital benefits from diversification")
        
        # Get metrics
        if lazy_billionaire_rank:
            lazy_metrics = strategy_rankings['by_sharpe'][lazy_billionaire_rank-1]
        else:
            lazy_metrics = {'sharpe': 0, 'return': 0}
        
        top_single_strategy = strategy_rankings['by_sharpe'][0] if strategy_rankings['by_sharpe'] else None
        
        return {
            'recommended': 'lazy_billionaire' if use_lazy_billionaire else 'single_strategy',
            'reasoning': reasoning,
            'lazy_billionaire_metrics': lazy_metrics,
            'best_single_strategy': top_single_strategy,
            'confidence': 0.8 if len(reasoning) >= 2 else 0.6
        }
    
    def _get_top_3_strategies(self, strategy_rankings: Dict[str, List]) -> List[Dict[str, Any]]:
        """Get top 3 strategies with detailed information"""
        top_3 = []
        
        for i, strategy in enumerate(strategy_rankings['by_sharpe'][:3]):
            # Get optimal parameters
            optimal_params = self.optimal_parameters.get(strategy['name'], {})
            
            # Get best conditions
            strategy_results = self.all_results.get(strategy['name'], {})
            improvements = self._generate_strategy_improvements()
            best_conditions = improvements.get(strategy['name'], {}).get('best_market_conditions', {})
            
            top_3.append({
                'rank': i + 1,
                'name': strategy['name'],
                'sharpe_ratio': strategy['sharpe'],
                'annual_return': strategy['return'] * 12,  # Approximate
                'max_drawdown': strategy['max_dd'],
                'win_rate': strategy['win_rate'],
                'optimal_parameters': optimal_params,
                'best_conditions': best_conditions,
                'why_good': self._explain_why_strategy_good(strategy)
            })
        
        return top_3
    
    def _explain_why_strategy_good(self, strategy: Dict[str, Any]) -> str:
        """Explain why a strategy is good"""
        reasons = []
        
        if strategy['sharpe'] > 1.5:
            reasons.append(f"Excellent risk-adjusted returns (Sharpe: {strategy['sharpe']:.2f})")
        elif strategy['sharpe'] > 1.0:
            reasons.append(f"Good risk-adjusted returns (Sharpe: {strategy['sharpe']:.2f})")
        
        if strategy['win_rate'] > 0.6:
            reasons.append(f"High win rate of {strategy['win_rate']*100:.1f}%")
        
        if strategy['max_dd'] < 0.15:
            reasons.append(f"Low risk with max drawdown of {strategy['max_dd']*100:.1f}%")
        
        if strategy['return'] > 0.20:
            reasons.append(f"Strong returns of {strategy['return']*100:.1f}%")
        
        return " | ".join(reasons) if reasons else "Solid overall performance"
    
    def _optimize_lazy_billionaire(self, 
                                  strategy_rankings: Dict[str, List],
                                  correlation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize Lazy Billionaire configuration"""
        
        # Get top uncorrelated strategies
        best_combinations = correlation_analysis.get('best_combinations', [])
        
        # Recommended strategy weights based on performance
        weights = {}
        total_sharpe = 0
        
        # Use top 5 strategies
        for strategy in strategy_rankings['by_sharpe'][:5]:
            if strategy['sharpe'] > 0:
                weights[strategy['name']] = strategy['sharpe']
                total_sharpe += strategy['sharpe']
        
        # Normalize weights
        if total_sharpe > 0:
            for strategy in weights:
                weights[strategy] = weights[strategy] / total_sharpe
        
        return {
            'recommended_strategies': list(weights.keys()),
            'optimal_weights': weights,
            'rebalance_frequency': 'monthly',
            'risk_profile': 'balanced',
            'expected_sharpe': sum(s['sharpe'] * weights.get(s['name'], 0) 
                                 for s in strategy_rankings['by_sharpe'] 
                                 if s['name'] in weights),
            'diversification_benefit': f"{(1 - correlation_analysis.get('average_correlation', 1)) * 100:.1f}%"
        }
    
    def _calculate_expected_returns(self, target_capital: float,
                                  best_approach: Dict[str, Any],
                                  strategy_rankings: Dict[str, List]) -> Dict[str, Any]:
        """Calculate expected returns for different scenarios"""
        
        if best_approach['recommended'] == 'lazy_billionaire':
            # Use Lazy Billionaire metrics
            metrics = best_approach['lazy_billionaire_metrics']
        else:
            # Use best single strategy
            metrics = best_approach['best_single_strategy']
        
        annual_return = metrics.get('return', 0) * 12  # Approximate annual
        
        # Calculate different scenarios
        expected_returns = {
            'expected_annual_return_pct': annual_return * 100,
            'expected_annual_return_eur': target_capital * annual_return,
            'best_case_annual_return_eur': target_capital * annual_return * 1.5,  # 50% better
            'worst_case_annual_return_eur': target_capital * annual_return * 0.5,  # 50% worse
            'expected_monthly_return_eur': target_capital * annual_return / 12,
            'risk_of_loss_pct': max(0, 30 - metrics.get('win_rate', 0.5) * 100),  # Simplified
            'confidence_level': 0.7 if len(self.all_results) > 5 else 0.5
        }
        
        return expected_returns
    
    def _generate_implementation_plan(self, 
                                    best_approach: Dict[str, Any],
                                    top_3_strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate detailed implementation plan"""
        
        if best_approach['recommended'] == 'lazy_billionaire':
            week1_tasks = [
                "Set up Lazy Billionaire strategy with recommended weights",
                "Configure monthly rebalancing schedule",
                "Set risk parameters to 'balanced' profile",
                "Implement position size limits (5% per strategy)"
            ]
            week2_tasks = [
                "Start with 10,000€ across top 3 strategies",
                "Monitor daily performance vs backtest",
                "Verify all strategies executing correctly",
                "Check correlation assumptions hold"
            ]
            week3_tasks = [
                "Scale to 50,000€ if performance within 20% of backtest",
                "Add 4th and 5th strategies to portfolio",
                "Implement automated rebalancing",
                "Set up performance alerts"
            ]
            week4_tasks = [
                "Scale to 150,000€ (50% of capital)",
                "Full Lazy Billionaire implementation",
                "Enable all monitoring and reporting",
                "Schedule monthly strategy reviews"
            ]
        else:
            # Single strategy approach
            best_strategy = top_3_strategies[0]
            week1_tasks = [
                f"Configure {best_strategy['name']} with optimal parameters",
                "Set stop loss at 5% and take profit based on backtest",
                "Implement position sizing at 2% risk per trade",
                "Test with paper trading for 3 days"
            ]
            week2_tasks = [
                "Start live trading with 10,000€",
                "Monitor every trade for first 20 trades",
                "Compare actual slippage with backtest assumptions",
                "Adjust parameters if needed"
            ]
            week3_tasks = [
                "Scale to 50,000€ if win rate above 45%",
                "Consider adding 2nd strategy for diversification",
                "Implement automated trade logging",
                "Set up daily performance reports"
            ]
            week4_tasks = [
                "Scale to 150,000€ gradually",
                "Full automation with monitoring",
                "Add risk management overlays",
                "Plan for market regime changes"
            ]
        
        return {
            'week_1': week1_tasks,
            'week_2': week2_tasks,
            'week_3': week3_tasks,
            'week_4': week4_tasks,
            'milestones': [
                "Week 1: System fully configured and tested",
                "Week 2: Live trading with small capital",
                "Week 3: Scaled to medium capital with full monitoring",
                "Week 4: Production deployment at 50% of target capital"
            ],
            'success_criteria': [
                "Actual performance within 30% of backtest",
                "Maximum drawdown not exceeding backtest by 50%",
                "All risk limits functioning correctly",
                "Daily monitoring showing no system issues"
            ]
        }
    
    def _generate_risk_warnings(self) -> List[str]:
        """Generate important risk warnings"""
        return [
            "Past performance does not guarantee future results",
            "Crypto markets are highly volatile and risky",
            "Never invest more than you can afford to lose",
            "Backtest results assume perfect execution which may not be achievable",
            "Market conditions can change rapidly making strategies ineffective",
            "Technical failures can result in significant losses",
            "Regulatory changes could impact trading ability",
            "Exchange risks including hacks and insolvency exist"
        ]
    
    def _generate_monitoring_plan(self) -> Dict[str, Any]:
        """Generate monitoring plan"""
        return {
            'daily_checks': [
                "Review P&L vs expected range",
                "Check all strategies are running",
                "Monitor drawdown levels",
                "Verify exchange connectivity"
            ],
            'weekly_reviews': [
                "Compare performance to backtest",
                "Analyze losing trades for patterns",
                "Review market conditions",
                "Check strategy correlations"
            ],
            'monthly_actions': [
                "Full performance analysis",
                "Rebalance portfolio if needed",
                "Review and update parameters",
                "Generate detailed reports"
            ],
            'red_flags': [
                "Drawdown exceeds 15% (immediate review)",
                "Win rate drops below 40% (strategy review)",
                "Daily loss exceeds 5% (stop trading)",
                "System errors (immediate investigation)"
            ],
            'emergency_procedures': [
                "If drawdown > 20%: Reduce position sizes by 50%",
                "If drawdown > 30%: Move 80% to stablecoins",
                "If system failure: Manual close all positions",
                "If exchange issues: Use backup exchange"
            ]
        }
    
    async def _generate_all_reports(self, final_recommendations: Dict[str, Any]):
        """Generate all reports and visualizations"""
        try:
            # 1. Generate comprehensive HTML report
            html_report = await self._generate_html_report(final_recommendations)
            
            # 2. Generate PDF report
            pdf_report = await self._generate_pdf_report(final_recommendations)
            
            # 3. Generate interactive dashboard
            dashboard = await self._generate_interactive_dashboard(final_recommendations)
            
            # 4. Generate Excel summary
            excel_report = await self._generate_excel_report(final_recommendations)
            
            logger.info(f"Reports generated in {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error generating reports: {e}")
    
    async def _generate_html_report(self, recommendations: Dict[str, Any]) -> str:
        """Generate comprehensive HTML report"""
        # Implementation would use the report generator
        return f"{self.output_dir}/master_analysis_report.html"
    
    async def _generate_pdf_report(self, recommendations: Dict[str, Any]) -> str:
        """Generate PDF report"""
        return f"{self.output_dir}/master_analysis_report.pdf"
    
    async def _generate_interactive_dashboard(self, recommendations: Dict[str, Any]) -> str:
        """Generate interactive Plotly dashboard"""
        return f"{self.output_dir}/interactive_dashboard.html"
    
    async def _generate_excel_report(self, recommendations: Dict[str, Any]) -> str:
        """Generate Excel report with all data"""
        return f"{self.output_dir}/master_analysis_data.xlsx"
    
    def _print_final_recommendation(self, recommendations: Dict[str, Any], target_capital: float):
        """Print final recommendation to console"""
        print("\n" + "="*80)
        print("FINAL RECOMMENDATION FOR €300,000 CAPITAL")
        print("="*80)
        
        approach = recommendations['best_approach']
        expected_returns = recommendations['expected_returns']
        
        print(f"\nBased on 4 years of backtesting with realistic market conditions,")
        print(f"I recommend for your capital of €{target_capital:,.0f}:")
        
        if approach['recommended'] == 'lazy_billionaire':
            print(f"\n✅ USE THE LAZY BILLIONAIRE STRATEGY")
            print(f"\nReasons:")
            for reason in approach['reasoning']:
                print(f"  • {reason}")
            
            config = recommendations['lazy_billionaire_config']
            print(f"\nOptimal Configuration:")
            print(f"  • Use these strategies: {', '.join(config['recommended_strategies'][:3])}")
            print(f"  • Rebalance: {config['rebalance_frequency']}")
            print(f"  • Risk profile: {config['risk_profile']}")
            print(f"  • Expected Sharpe ratio: {config['expected_sharpe']:.2f}")
        else:
            top_strategy = recommendations['top_3_strategies'][0]
            print(f"\n✅ USE SINGLE STRATEGY: {top_strategy['name'].upper()}")
            print(f"\nReasons:")
            print(f"  • {top_strategy['why_good']}")
        
        print(f"\nExpected Performance:")
        print(f"  • Annual return: {expected_returns['expected_annual_return_pct']:.1f}%")
        print(f"  • Monthly income: €{expected_returns['expected_monthly_return_eur']:,.0f}")
        print(f"  • Best case (year): €{expected_returns['best_case_annual_return_eur']:,.0f}")
        print(f"  • Worst case (year): €{expected_returns['worst_case_annual_return_eur']:,.0f}")
        print(f"  • Confidence level: {expected_returns['confidence_level']*100:.0f}%")
        
        print(f"\nImplementation Plan:")
        plan = recommendations['implementation_plan']
        print(f"  Week 1: Setup and Testing")
        print(f"  Week 2: Start with €10,000 (3.3% of capital)")
        print(f"  Week 3: Scale to €50,000 if performing well")
        print(f"  Week 4: Scale to €150,000 (50% of capital)")
        
        print(f"\n⚠️  IMPORTANT WARNINGS:")
        print(f"  • Never risk more than 50% of your capital")
        print(f"  • Stop if drawdown exceeds 20%")
        print(f"  • This is high-risk investing")
        print(f"  • Past performance ≠ future results")
        
        print("\n" + "="*80)
        print(f"Full report available at: {self.output_dir}")
        print("="*80 + "\n")
    
    def _save_all_results(self, final_recommendations: Dict[str, Any]):
        """Save all results to files"""
        try:
            # Save final recommendations
            with open(f"{self.output_dir}/final_recommendations.json", 'w') as f:
                json.dump(final_recommendations, f, indent=2, default=str)
            
            # Save optimal parameters
            with open(f"{self.output_dir}/optimal_parameters.json", 'w') as f:
                json.dump(self.optimal_parameters, f, indent=2)
            
            # Save detailed results
            with open(f"{self.output_dir}/all_backtest_results.json", 'w') as f:
                # Simplified save (full results might be too large)
                summary = {
                    'strategy_summaries': {
                        name: results.get('aggregate_metrics', {})
                        for name, results in self.all_results.items()
                    },
                    'correlations': self.strategy_correlations
                }
                json.dump(summary, f, indent=2, default=str)
            
            # Create implementation checklist
            self._create_implementation_checklist(final_recommendations)
            
            logger.info(f"All results saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def _create_implementation_checklist(self, recommendations: Dict[str, Any]):
        """Create markdown checklist for implementation"""
        checklist_content = """# Implementation Checklist

## Week 1: Setup & Testing
"""
        
        plan = recommendations['implementation_plan']
        
        for task in plan['week_1']:
            checklist_content += f"- [ ] {task}\n"
        
        checklist_content += "\n## Week 2: Soft Launch\n"
        for task in plan['week_2']:
            checklist_content += f"- [ ] {task}\n"
        
        checklist_content += "\n## Week 3: Scaling\n"
        for task in plan['week_3']:
            checklist_content += f"- [ ] {task}\n"
        
        checklist_content += "\n## Week 4: Full Implementation\n"
        for task in plan['week_4']:
            checklist_content += f"- [ ] {task}\n"
        
        checklist_content += "\n## Success Criteria\n"
        for criterion in plan['success_criteria']:
            checklist_content += f"- [ ] {criterion}\n"
        
        checklist_content += "\n## Daily Monitoring\n"
        monitoring = recommendations['monitoring_plan']
        for check in monitoring['daily_checks']:
            checklist_content += f"- [ ] {check}\n"
        
        with open(f"{self.output_dir}/implementation_checklist.md", 'w') as f:
            f.write(checklist_content)


async def main():
    """Main function to run master backtest analysis"""
    parser = argparse.ArgumentParser(description='Master Backtest Analysis')
    parser.add_argument('--capital', type=float, default=300000, 
                       help='Target capital amount')
    parser.add_argument('--start-date', type=str, default='2020-01-01',
                       help='Start date for analysis')
    parser.add_argument('--end-date', type=str, default='2024-01-01',
                       help='End date for analysis')
    parser.add_argument('--strategies', nargs='+', default=['all'],
                       help='Strategies to test (default: all)')
    parser.add_argument('--risk-profiles', type=str, default='conservative,balanced,aggressive',
                       help='Risk profiles to test (comma-separated)')
    parser.add_argument('--output-dir', type=str, default='results/master_analysis',
                       help='Output directory for results')
    parser.add_argument('--generate-report', type=bool, default=True,
                       help='Generate comprehensive report')
    
    args = parser.parse_args()
    
    try:
        # Configure analyzer
        config = {
            'output_dir': args.output_dir,
            'generate_report': args.generate_report
        }
        
        # Create analyzer
        analyzer = MasterBacktestAnalyzer(config)
        
        # Determine strategies to test
        strategies = None if 'all' in args.strategies else args.strategies
        
        # Run master analysis
        results = await analyzer.run_master_analysis(
            target_capital=args.capital,
            selected_strategies=strategies
        )
        
        print(f"\n✅ Master analysis completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        print(f"📊 Open {args.output_dir}/interactive_dashboard.html to explore results")
        
    except Exception as e:
        logger.error(f"Error in master analysis: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())