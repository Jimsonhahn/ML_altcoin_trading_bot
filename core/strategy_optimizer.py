# core/strategy_optimizer.py
"""
Advanced Strategy Parameter Optimization System
Provides grid search, walk-forward analysis, and Monte Carlo optimization
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from itertools import product
import copy
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

# Import backtest components
from core.backtest_engine import BacktestEngine
from core.backtest_analyzer import BacktestAnalyzer

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Optimization method types"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    GENETIC_ALGORITHM = "genetic_algorithm"
    WALK_FORWARD = "walk_forward"
    MONTE_CARLO = "monte_carlo"


class OptimizationObjective(Enum):
    """Optimization objective metrics"""
    TOTAL_RETURN = "total_return"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    PROFIT_FACTOR = "profit_factor"
    WIN_RATE = "win_rate"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"


@dataclass
class ParameterRange:
    """Parameter optimization range"""
    name: str
    min_value: Union[int, float]
    max_value: Union[int, float]
    step: Union[int, float] = None
    values: List[Union[int, float]] = None
    param_type: type = float
    
    def get_values(self) -> List[Union[int, float]]:
        """Get all possible values for this parameter"""
        if self.values:
            return self.values
        
        if self.step is None:
            # For continuous parameters, use 10 steps by default
            steps = 10
            if self.param_type == int:
                steps = min(10, self.max_value - self.min_value + 1)
            return list(np.linspace(self.min_value, self.max_value, steps))
        
        if self.param_type == int:
            return list(range(int(self.min_value), int(self.max_value) + 1, int(self.step)))
        else:
            values = []
            current = self.min_value
            while current <= self.max_value:
                values.append(current)
                current += self.step
            return values


@dataclass
class OptimizationResult:
    """Single optimization run result"""
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    objective_value: float
    backtest_results: Dict[str, Any]
    duration_seconds: float
    market_phase: str = "unknown"
    validation_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'parameters': self.parameters,
            'performance_metrics': self.performance_metrics,
            'objective_value': self.objective_value,
            'duration_seconds': self.duration_seconds,
            'market_phase': self.market_phase,
            'validation_score': self.validation_score
        }


@dataclass
class WalkForwardResult:
    """Walk-forward analysis result"""
    in_sample_period: Tuple[datetime, datetime]
    out_of_sample_period: Tuple[datetime, datetime]
    best_parameters: Dict[str, Any]
    in_sample_performance: Dict[str, float]
    out_of_sample_performance: Dict[str, float]
    degradation_percent: float


@dataclass
class OptimizationSummary:
    """Complete optimization summary"""
    method: OptimizationMethod
    objective: OptimizationObjective
    total_combinations: int
    completed_combinations: int
    best_result: OptimizationResult
    all_results: List[OptimizationResult]
    optimization_duration: float
    parameter_importance: Dict[str, float]
    robust_parameters: Dict[str, Any]
    market_phase_performance: Dict[str, List[OptimizationResult]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'method': self.method.value,
            'objective': self.objective.value,
            'total_combinations': self.total_combinations,
            'completed_combinations': self.completed_combinations,
            'best_result': self.best_result.to_dict() if self.best_result else None,
            'optimization_duration': self.optimization_duration,
            'parameter_importance': self.parameter_importance,
            'robust_parameters': self.robust_parameters,
            'market_phase_performance': {
                phase: [r.to_dict() for r in results] 
                for phase, results in self.market_phase_performance.items()
            }
        }


class StrategyOptimizer:
    """
    Advanced strategy parameter optimizer with multiple optimization methods
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Optimization settings
        self.max_workers = self.config.get('max_workers', 4)
        self.timeout_seconds = self.config.get('timeout_seconds', 3600)  # 1 hour
        self.validation_split = self.config.get('validation_split', 0.3)
        self.min_trades_threshold = self.config.get('min_trades_threshold', 10)
        
        # Market phase detection
        self.enable_market_phase_analysis = self.config.get('enable_market_phase_analysis', True)
        self.market_phase_periods = self.config.get('market_phase_periods', {
            'bull': 60,      # days
            'bear': 60,
            'sideways': 30,
            'volatile': 30
        })
        
        # Results storage
        self.optimization_history: List[OptimizationSummary] = []
        
        logger.info("StrategyOptimizer initialized")
    
    async def optimize_strategy(self, 
                              strategy_class,
                              parameter_ranges: List[ParameterRange],
                              data: pd.DataFrame,
                              method: OptimizationMethod = OptimizationMethod.GRID_SEARCH,
                              objective: OptimizationObjective = OptimizationObjective.SHARPE_RATIO,
                              symbol: str = "BTC/USDT",
                              initial_capital: float = 10000) -> OptimizationSummary:
        """
        Optimize strategy parameters using specified method
        """
        try:
            logger.info(f"Starting optimization: {method.value} for {strategy_class.__name__}")
            start_time = datetime.now()
            
            # Generate parameter combinations
            if method == OptimizationMethod.GRID_SEARCH:
                param_combinations = self._generate_grid_combinations(parameter_ranges)
            elif method == OptimizationMethod.RANDOM_SEARCH:
                param_combinations = self._generate_random_combinations(parameter_ranges, 100)
            elif method == OptimizationMethod.GENETIC_ALGORITHM:
                return await self._genetic_algorithm_optimization(
                    strategy_class, parameter_ranges, data, objective, symbol, initial_capital
                )
            elif method == OptimizationMethod.WALK_FORWARD:
                return await self._walk_forward_optimization(
                    strategy_class, parameter_ranges, data, objective, symbol, initial_capital
                )
            elif method == OptimizationMethod.MONTE_CARLO:
                return await self._monte_carlo_optimization(
                    strategy_class, parameter_ranges, data, objective, symbol, initial_capital
                )
            else:
                param_combinations = self._generate_grid_combinations(parameter_ranges)
            
            logger.info(f"Generated {len(param_combinations)} parameter combinations")
            
            # Run optimization
            results = await self._run_parallel_optimization(
                strategy_class, param_combinations, data, objective, symbol, initial_capital
            )
            
            # Analyze results
            summary = await self._analyze_optimization_results(
                results, method, objective, parameter_ranges, start_time
            )
            
            # Store in history
            self.optimization_history.append(summary)
            
            logger.info(f"Optimization completed. Best {objective.value}: {summary.best_result.objective_value:.4f}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in strategy optimization: {e}")
            raise
    
    def _generate_grid_combinations(self, parameter_ranges: List[ParameterRange]) -> List[Dict[str, Any]]:
        """Generate all possible parameter combinations for grid search"""
        try:
            param_values = []
            param_names = []
            
            for param_range in parameter_ranges:
                param_names.append(param_range.name)
                values = param_range.get_values()
                if param_range.param_type == int:
                    values = [int(v) for v in values]
                param_values.append(values)
            
            # Generate cartesian product
            combinations = []
            for combination in product(*param_values):
                param_dict = dict(zip(param_names, combination))
                combinations.append(param_dict)
            
            return combinations
            
        except Exception as e:
            logger.error(f"Error generating grid combinations: {e}")
            return []
    
    def _generate_random_combinations(self, parameter_ranges: List[ParameterRange], 
                                    num_combinations: int) -> List[Dict[str, Any]]:
        """Generate random parameter combinations"""
        try:
            combinations = []
            
            for _ in range(num_combinations):
                param_dict = {}
                for param_range in parameter_ranges:
                    values = param_range.get_values()
                    value = random.choice(values)
                    if param_range.param_type == int:
                        value = int(value)
                    param_dict[param_range.name] = value
                combinations.append(param_dict)
            
            return combinations
            
        except Exception as e:
            logger.error(f"Error generating random combinations: {e}")
            return []
    
    async def _run_parallel_optimization(self, 
                                       strategy_class,
                                       param_combinations: List[Dict[str, Any]],
                                       data: pd.DataFrame,
                                       objective: OptimizationObjective,
                                       symbol: str,
                                       initial_capital: float) -> List[OptimizationResult]:
        """Run optimization in parallel"""
        try:
            results = []
            
            # Split data for validation
            split_idx = int(len(data) * (1 - self.validation_split))
            train_data = data.iloc[:split_idx]
            validation_data = data.iloc[split_idx:]
            
            # Run backtests in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                
                for i, params in enumerate(param_combinations):
                    future = executor.submit(
                        self._run_single_backtest,
                        strategy_class, params, train_data, validation_data, 
                        objective, symbol, initial_capital, i
                    )
                    futures.append(future)
                
                # Collect results
                for future in as_completed(futures, timeout=self.timeout_seconds):
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Error in parallel backtest: {e}")
            
            # Sort by objective value (descending for most objectives)
            reverse_sort = objective not in [OptimizationObjective.MAX_DRAWDOWN]
            results.sort(key=lambda x: x.objective_value, reverse=reverse_sort)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in parallel optimization: {e}")
            return []
    
    def _run_single_backtest(self, 
                           strategy_class,
                           params: Dict[str, Any],
                           train_data: pd.DataFrame,
                           validation_data: pd.DataFrame,
                           objective: OptimizationObjective,
                           symbol: str,
                           initial_capital: float,
                           run_id: int) -> Optional[OptimizationResult]:
        """Run single backtest with given parameters"""
        try:
            start_time = datetime.now()
            
            # Create strategy instance
            strategy = strategy_class(params)
            
            # Run backtest on training data
            backtest_engine = BacktestEngine({
                'initial_capital': initial_capital,
                'enable_slippage': True,
                'enable_market_impact': True,
                'enable_latency': True
            })
            
            # Prepare data
            train_start = train_data.index[0]
            train_end = train_data.index[-1]
            
            # Run backtest
            backtest_results = backtest_engine.run_backtest(
                strategy=strategy,
                symbol=symbol,
                start_date=train_start,
                end_date=train_end,
                data=train_data
            )
            
            # Check minimum trades threshold
            trades = backtest_results.get('trades', [])
            if len(trades) < self.min_trades_threshold:
                logger.debug(f"Run {run_id}: Insufficient trades ({len(trades)})")
                return None
            
            # Analyze performance
            analyzer = BacktestAnalyzer()
            analysis = analyzer.analyze_single_strategy(backtest_results)
            
            # Calculate objective value
            performance_metrics = analysis.get('performance_metrics', {})
            objective_value = self._calculate_objective_value(objective, performance_metrics)
            
            # Validate on out-of-sample data if available
            validation_score = 0.0
            if not validation_data.empty:
                validation_score = self._validate_parameters(
                    strategy_class, params, validation_data, objective, symbol, initial_capital
                )
            
            # Detect market phase
            market_phase = self._detect_market_phase(train_data)
            
            # Create result
            result = OptimizationResult(
                parameters=params.copy(),
                performance_metrics=performance_metrics,
                objective_value=objective_value,
                backtest_results=backtest_results,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                market_phase=market_phase,
                validation_score=validation_score
            )
            
            logger.debug(f"Run {run_id}: {objective.value}={objective_value:.4f}, params={params}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in single backtest (run {run_id}): {e}")
            return None
    
    def _calculate_objective_value(self, objective: OptimizationObjective, 
                                 metrics: Dict[str, float]) -> float:
        """Calculate objective value from performance metrics"""
        try:
            if objective == OptimizationObjective.TOTAL_RETURN:
                return metrics.get('total_return', 0.0)
            elif objective == OptimizationObjective.SHARPE_RATIO:
                return metrics.get('sharpe_ratio', 0.0)
            elif objective == OptimizationObjective.SORTINO_RATIO:
                return metrics.get('sortino_ratio', 0.0)
            elif objective == OptimizationObjective.CALMAR_RATIO:
                return metrics.get('calmar_ratio', 0.0)
            elif objective == OptimizationObjective.MAX_DRAWDOWN:
                return -metrics.get('max_drawdown', 1.0)  # Negative because we want to minimize
            elif objective == OptimizationObjective.PROFIT_FACTOR:
                return metrics.get('profit_factor', 0.0)
            elif objective == OptimizationObjective.WIN_RATE:
                return metrics.get('win_rate', 0.0)
            elif objective == OptimizationObjective.RISK_ADJUSTED_RETURN:
                # Custom risk-adjusted return combining multiple metrics
                sharpe = metrics.get('sharpe_ratio', 0.0)
                max_dd = metrics.get('max_drawdown', 1.0)
                return sharpe * (1 - max_dd)
            else:
                return metrics.get('sharpe_ratio', 0.0)
        except:
            return 0.0
    
    def _validate_parameters(self, 
                           strategy_class,
                           params: Dict[str, Any],
                           validation_data: pd.DataFrame,
                           objective: OptimizationObjective,
                           symbol: str,
                           initial_capital: float) -> float:
        """Validate parameters on out-of-sample data"""
        try:
            strategy = strategy_class(params)
            
            backtest_engine = BacktestEngine({
                'initial_capital': initial_capital,
                'enable_slippage': True,
                'enable_market_impact': True
            })
            
            val_start = validation_data.index[0]
            val_end = validation_data.index[-1]
            
            backtest_results = backtest_engine.run_backtest(
                strategy=strategy,
                symbol=symbol,
                start_date=val_start,
                end_date=val_end,
                data=validation_data
            )
            
            trades = backtest_results.get('trades', [])
            if len(trades) < 5:  # Minimum trades for validation
                return 0.0
            
            analyzer = BacktestAnalyzer()
            analysis = analyzer.analyze_single_strategy(backtest_results)
            performance_metrics = analysis.get('performance_metrics', {})
            
            return self._calculate_objective_value(objective, performance_metrics)
            
        except Exception as e:
            logger.error(f"Error in parameter validation: {e}")
            return 0.0
    
    def _detect_market_phase(self, data: pd.DataFrame) -> str:
        """Detect market phase from price data"""
        try:
            if 'close' not in data.columns:
                return "unknown"
            
            prices = data['close']
            
            # Calculate trend
            trend = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
            
            # Calculate volatility
            returns = prices.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Classify market phase
            if trend > 0.2 and volatility < 0.3:
                return "bull"
            elif trend < -0.2 and volatility < 0.3:
                return "bear"
            elif abs(trend) < 0.1 and volatility < 0.2:
                return "sideways"
            elif volatility > 0.4:
                return "volatile"
            else:
                return "mixed"
                
        except Exception as e:
            logger.error(f"Error detecting market phase: {e}")
            return "unknown"
    
    async def _analyze_optimization_results(self, 
                                          results: List[OptimizationResult],
                                          method: OptimizationMethod,
                                          objective: OptimizationObjective,
                                          parameter_ranges: List[ParameterRange],
                                          start_time: datetime) -> OptimizationSummary:
        """Analyze optimization results and create summary"""
        try:
            if not results:
                logger.warning("No optimization results to analyze")
                return OptimizationSummary(
                    method=method,
                    objective=objective,
                    total_combinations=0,
                    completed_combinations=0,
                    best_result=None,
                    all_results=[],
                    optimization_duration=0.0,
                    parameter_importance={},
                    robust_parameters={},
                    market_phase_performance={}
                )
            
            # Best result
            best_result = results[0]
            
            # Parameter importance analysis
            parameter_importance = self._analyze_parameter_importance(results, parameter_ranges)
            
            # Robust parameters (parameters that work well across different conditions)
            robust_parameters = self._find_robust_parameters(results)
            
            # Market phase performance analysis
            market_phase_performance = self._analyze_market_phase_performance(results)
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            
            summary = OptimizationSummary(
                method=method,
                objective=objective,
                total_combinations=len(results),
                completed_combinations=len(results),
                best_result=best_result,
                all_results=results[:50],  # Store top 50 results
                optimization_duration=duration,
                parameter_importance=parameter_importance,
                robust_parameters=robust_parameters,
                market_phase_performance=market_phase_performance
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error analyzing optimization results: {e}")
            raise
    
    def _analyze_parameter_importance(self, 
                                    results: List[OptimizationResult],
                                    parameter_ranges: List[ParameterRange]) -> Dict[str, float]:
        """Analyze parameter importance using correlation with objective values"""
        try:
            if len(results) < 10:
                return {}
            
            importance = {}
            objective_values = [r.objective_value for r in results]
            
            for param_range in parameter_ranges:
                param_name = param_range.name
                param_values = []
                
                for result in results:
                    if param_name in result.parameters:
                        param_values.append(result.parameters[param_name])
                    else:
                        param_values.append(0)
                
                if len(set(param_values)) > 1:  # Only if parameter varies
                    correlation = np.corrcoef(param_values, objective_values)[0, 1]
                    importance[param_name] = abs(correlation) if not np.isnan(correlation) else 0.0
                else:
                    importance[param_name] = 0.0
            
            return importance
            
        except Exception as e:
            logger.error(f"Error analyzing parameter importance: {e}")
            return {}
    
    def _find_robust_parameters(self, results: List[OptimizationResult]) -> Dict[str, Any]:
        """Find parameters that consistently perform well"""
        try:
            if len(results) < 20:
                return {}
            
            # Take top 20% of results
            top_results = results[:max(1, len(results) // 5)]
            
            robust_params = {}
            
            # Find parameters that appear frequently in top results
            for param_name in top_results[0].parameters.keys():
                values = [r.parameters[param_name] for r in top_results]
                
                # Calculate mode or median depending on data type
                if isinstance(values[0], (int, float)):
                    robust_params[param_name] = statistics.median(values)
                else:
                    # For categorical parameters, find most common
                    from collections import Counter
                    counter = Counter(values)
                    robust_params[param_name] = counter.most_common(1)[0][0]
            
            return robust_params
            
        except Exception as e:
            logger.error(f"Error finding robust parameters: {e}")
            return {}
    
    def _analyze_market_phase_performance(self, 
                                        results: List[OptimizationResult]) -> Dict[str, List[OptimizationResult]]:
        """Analyze performance by market phase"""
        try:
            phase_performance = {}
            
            for result in results:
                phase = result.market_phase
                if phase not in phase_performance:
                    phase_performance[phase] = []
                phase_performance[phase].append(result)
            
            # Sort results within each phase
            for phase in phase_performance:
                phase_performance[phase].sort(key=lambda x: x.objective_value, reverse=True)
                phase_performance[phase] = phase_performance[phase][:10]  # Top 10 per phase
            
            return phase_performance
            
        except Exception as e:
            logger.error(f"Error analyzing market phase performance: {e}")
            return {}
    
    async def _walk_forward_optimization(self, 
                                       strategy_class,
                                       parameter_ranges: List[ParameterRange],
                                       data: pd.DataFrame,
                                       objective: OptimizationObjective,
                                       symbol: str,
                                       initial_capital: float) -> OptimizationSummary:
        """Perform walk-forward optimization"""
        try:
            logger.info("Starting walk-forward optimization")
            start_time = datetime.now()
            
            # Configuration
            in_sample_days = self.config.get('walk_forward_in_sample_days', 180)
            out_sample_days = self.config.get('walk_forward_out_sample_days', 60)
            step_days = self.config.get('walk_forward_step_days', 30)
            
            walk_forward_results = []
            all_optimization_results = []
            
            # Generate time windows
            data_days = len(data)
            window_size = in_sample_days + out_sample_days
            
            for start_idx in range(0, data_days - window_size, step_days):
                end_in_sample = start_idx + in_sample_days
                end_out_sample = start_idx + window_size
                
                if end_out_sample > data_days:
                    break
                
                # Split data
                in_sample_data = data.iloc[start_idx:end_in_sample]
                out_sample_data = data.iloc[end_in_sample:end_out_sample]
                
                logger.info(f"Walk-forward window: {in_sample_data.index[0]} to {out_sample_data.index[-1]}")
                
                # Optimize on in-sample data
                param_combinations = self._generate_grid_combinations(parameter_ranges)
                
                window_results = await self._run_parallel_optimization(
                    strategy_class, param_combinations[:50],  # Limit combinations for speed
                    in_sample_data, objective, symbol, initial_capital
                )
                
                if not window_results:
                    continue
                
                best_params = window_results[0].parameters
                
                # Test on out-of-sample data
                out_sample_score = self._validate_parameters(
                    strategy_class, best_params, out_sample_data, 
                    objective, symbol, initial_capital
                )
                
                # Create walk-forward result
                wf_result = WalkForwardResult(
                    in_sample_period=(in_sample_data.index[0], in_sample_data.index[-1]),
                    out_of_sample_period=(out_sample_data.index[0], out_sample_data.index[-1]),
                    best_parameters=best_params,
                    in_sample_performance=window_results[0].performance_metrics,
                    out_of_sample_performance={'objective_value': out_sample_score},
                    degradation_percent=((window_results[0].objective_value - out_sample_score) / 
                                       window_results[0].objective_value * 100) if window_results[0].objective_value != 0 else 0
                )
                
                walk_forward_results.append(wf_result)
                all_optimization_results.extend(window_results[:10])  # Keep top 10 from each window
            
            # Analyze overall results
            if all_optimization_results:
                # Find best overall parameters
                all_optimization_results.sort(key=lambda x: x.objective_value, reverse=True)
                best_result = all_optimization_results[0]
                
                # Create summary
                summary = OptimizationSummary(
                    method=OptimizationMethod.WALK_FORWARD,
                    objective=objective,
                    total_combinations=len(walk_forward_results),
                    completed_combinations=len(walk_forward_results),
                    best_result=best_result,
                    all_results=all_optimization_results[:50],
                    optimization_duration=(datetime.now() - start_time).total_seconds(),
                    parameter_importance=self._analyze_parameter_importance(all_optimization_results, parameter_ranges),
                    robust_parameters=self._find_robust_parameters(all_optimization_results),
                    market_phase_performance=self._analyze_market_phase_performance(all_optimization_results)
                )
                
                return summary
            else:
                raise ValueError("No valid walk-forward results generated")
                
        except Exception as e:
            logger.error(f"Error in walk-forward optimization: {e}")
            raise
    
    async def _monte_carlo_optimization(self, 
                                      strategy_class,
                                      parameter_ranges: List[ParameterRange],
                                      data: pd.DataFrame,
                                      objective: OptimizationObjective,
                                      symbol: str,
                                      initial_capital: float) -> OptimizationSummary:
        """Perform Monte Carlo optimization with bootstrapping"""
        try:
            logger.info("Starting Monte Carlo optimization")
            start_time = datetime.now()
            
            iterations = self.config.get('monte_carlo_iterations', 200)
            bootstrap_sample_ratio = self.config.get('bootstrap_sample_ratio', 0.8)
            
            all_results = []
            
            # Generate random parameter combinations
            param_combinations = self._generate_random_combinations(parameter_ranges, iterations)
            
            # For each parameter combination, run multiple bootstrap samples
            for i, params in enumerate(param_combinations):
                logger.debug(f"Monte Carlo iteration {i+1}/{len(param_combinations)}")
                
                bootstrap_results = []
                
                # Run multiple bootstrap samples
                for bootstrap_iter in range(5):  # 5 bootstrap samples per parameter set
                    # Create bootstrap sample
                    sample_size = int(len(data) * bootstrap_sample_ratio)
                    bootstrap_data = data.sample(n=sample_size, replace=True).sort_index()
                    
                    # Run backtest
                    result = self._run_single_backtest(
                        strategy_class, params, bootstrap_data, pd.DataFrame(),
                        objective, symbol, initial_capital, f"{i}_{bootstrap_iter}"
                    )
                    
                    if result:
                        bootstrap_results.append(result)
                
                # Calculate average performance across bootstrap samples
                if bootstrap_results:
                    avg_objective = statistics.mean([r.objective_value for r in bootstrap_results])
                    
                    # Create aggregated result
                    best_bootstrap = max(bootstrap_results, key=lambda x: x.objective_value)
                    best_bootstrap.objective_value = avg_objective
                    best_bootstrap.validation_score = statistics.stdev([r.objective_value for r in bootstrap_results]) if len(bootstrap_results) > 1 else 0
                    
                    all_results.append(best_bootstrap)
            
            # Sort and analyze results
            all_results.sort(key=lambda x: x.objective_value, reverse=True)
            
            summary = OptimizationSummary(
                method=OptimizationMethod.MONTE_CARLO,
                objective=objective,
                total_combinations=iterations,
                completed_combinations=len(all_results),
                best_result=all_results[0] if all_results else None,
                all_results=all_results[:50],
                optimization_duration=(datetime.now() - start_time).total_seconds(),
                parameter_importance=self._analyze_parameter_importance(all_results, parameter_ranges),
                robust_parameters=self._find_robust_parameters(all_results),
                market_phase_performance=self._analyze_market_phase_performance(all_results)
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo optimization: {e}")
            raise
    
    async def _genetic_algorithm_optimization(self, 
                                            strategy_class,
                                            parameter_ranges: List[ParameterRange],
                                            data: pd.DataFrame,
                                            objective: OptimizationObjective,
                                            symbol: str,
                                            initial_capital: float) -> OptimizationSummary:
        """Perform genetic algorithm optimization"""
        try:
            logger.info("Starting genetic algorithm optimization")
            start_time = datetime.now()
            
            # GA Parameters
            population_size = self.config.get('ga_population_size', 50)
            generations = self.config.get('ga_generations', 20)
            mutation_rate = self.config.get('ga_mutation_rate', 0.1)
            crossover_rate = self.config.get('ga_crossover_rate', 0.8)
            
            # Initialize population
            population = []
            for _ in range(population_size):
                individual = {}
                for param_range in parameter_ranges:
                    values = param_range.get_values()
                    individual[param_range.name] = random.choice(values)
                population.append(individual)
            
            all_results = []
            
            for generation in range(generations):
                logger.info(f"GA Generation {generation + 1}/{generations}")
                
                # Evaluate population
                generation_results = []
                for individual in population:
                    result = self._run_single_backtest(
                        strategy_class, individual, data, pd.DataFrame(),
                        objective, symbol, initial_capital, f"gen{generation}"
                    )
                    if result:
                        generation_results.append(result)
                
                if not generation_results:
                    continue
                
                generation_results.sort(key=lambda x: x.objective_value, reverse=True)
                all_results.extend(generation_results)
                
                # Selection (tournament selection)
                new_population = []
                elite_size = population_size // 4
                
                # Keep elite
                for i in range(elite_size):
                    if i < len(generation_results):
                        new_population.append(generation_results[i].parameters)
                
                # Generate offspring
                while len(new_population) < population_size:
                    if random.random() < crossover_rate and len(generation_results) > 1:
                        # Crossover
                        parent1 = self._tournament_selection(generation_results, 3)
                        parent2 = self._tournament_selection(generation_results, 3)
                        child = self._crossover(parent1.parameters, parent2.parameters, parameter_ranges)
                    else:
                        # Copy parent
                        parent = self._tournament_selection(generation_results, 3)
                        child = parent.parameters.copy()
                    
                    # Mutation
                    if random.random() < mutation_rate:
                        child = self._mutate(child, parameter_ranges)
                    
                    new_population.append(child)
                
                population = new_population
            
            # Final analysis
            all_results.sort(key=lambda x: x.objective_value, reverse=True)
            
            summary = OptimizationSummary(
                method=OptimizationMethod.GENETIC_ALGORITHM,
                objective=objective,
                total_combinations=len(all_results),
                completed_combinations=len(all_results),
                best_result=all_results[0] if all_results else None,
                all_results=all_results[:50],
                optimization_duration=(datetime.now() - start_time).total_seconds(),
                parameter_importance=self._analyze_parameter_importance(all_results, parameter_ranges),
                robust_parameters=self._find_robust_parameters(all_results),
                market_phase_performance=self._analyze_market_phase_performance(all_results)
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in genetic algorithm optimization: {e}")
            raise
    
    def _tournament_selection(self, results: List[OptimizationResult], tournament_size: int) -> OptimizationResult:
        """Tournament selection for genetic algorithm"""
        tournament = random.sample(results, min(tournament_size, len(results)))
        return max(tournament, key=lambda x: x.objective_value)
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any], 
                  parameter_ranges: List[ParameterRange]) -> Dict[str, Any]:
        """Crossover operation for genetic algorithm"""
        child = {}
        for param_range in parameter_ranges:
            param_name = param_range.name
            if param_name in parent1 and param_name in parent2:
                # Random selection from parents
                child[param_name] = random.choice([parent1[param_name], parent2[param_name]])
            elif param_name in parent1:
                child[param_name] = parent1[param_name]
            elif param_name in parent2:
                child[param_name] = parent2[param_name]
        return child
    
    def _mutate(self, individual: Dict[str, Any], parameter_ranges: List[ParameterRange]) -> Dict[str, Any]:
        """Mutation operation for genetic algorithm"""
        mutated = individual.copy()
        for param_range in parameter_ranges:
            if random.random() < 0.1:  # 10% chance per parameter
                values = param_range.get_values()
                mutated[param_range.name] = random.choice(values)
        return mutated
    
    def get_optimization_recommendations(self, summary: OptimizationSummary) -> Dict[str, Any]:
        """Generate recommendations based on optimization results"""
        try:
            recommendations = {
                'best_parameters': summary.robust_parameters,
                'parameter_insights': [],
                'market_phase_recommendations': {},
                'robustness_analysis': {},
                'next_steps': []
            }
            
            # Parameter insights
            for param, importance in summary.parameter_importance.items():
                if importance > 0.3:
                    recommendations['parameter_insights'].append({
                        'parameter': param,
                        'importance': importance,
                        'recommendation': f"High impact parameter - tune carefully"
                    })
                elif importance < 0.1:
                    recommendations['parameter_insights'].append({
                        'parameter': param,
                        'importance': importance,
                        'recommendation': f"Low impact parameter - can use default value"
                    })
            
            # Market phase recommendations
            for phase, results in summary.market_phase_performance.items():
                if results:
                    best_for_phase = results[0]
                    recommendations['market_phase_recommendations'][phase] = {
                        'best_parameters': best_for_phase.parameters,
                        'performance': best_for_phase.objective_value
                    }
            
            # Robustness analysis
            if len(summary.all_results) > 10:
                top_10_results = summary.all_results[:10]
                objective_values = [r.objective_value for r in top_10_results]
                
                recommendations['robustness_analysis'] = {
                    'performance_stability': statistics.stdev(objective_values) / statistics.mean(objective_values) if statistics.mean(objective_values) > 0 else 1,
                    'parameter_consistency': 'High' if len(set(str(r.parameters) for r in top_10_results)) < 5 else 'Low'
                }
            
            # Next steps
            if summary.method != OptimizationMethod.WALK_FORWARD:
                recommendations['next_steps'].append("Run walk-forward analysis to validate robustness")
            
            if len(summary.all_results) < 100:
                recommendations['next_steps'].append("Increase search space for more thorough optimization")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {}
    
    def export_optimization_results(self, summary: OptimizationSummary, filepath: str = None) -> str:
        """Export optimization results to file"""
        try:
            if filepath is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filepath = f"optimization_results_{timestamp}.json"
            
            export_data = {
                'summary': summary.to_dict(),
                'recommendations': self.get_optimization_recommendations(summary),
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Optimization results exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting optimization results: {e}")
            return ""


# Factory function
def create_strategy_optimizer(config: Dict[str, Any] = None) -> StrategyOptimizer:
    """Create and return StrategyOptimizer instance"""
    return StrategyOptimizer(config)