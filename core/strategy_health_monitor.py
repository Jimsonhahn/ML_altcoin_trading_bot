#!/usr/bin/env python3
"""
Strategy Health Monitoring and A/B Testing System
================================================

Advanced health monitoring and A/B testing system for the self-discovering strategy orchestrator:
- Real-time strategy performance monitoring
- Anomaly detection and degradation alerts
- Automatic strategy variation generation and testing
- Performance comparison and statistical significance testing
- Dynamic strategy parameter optimization
- Continuous learning and adaptation

Features:
- Health scores based on multiple metrics
- Automated A/B testing with statistical validation
- Strategy parameter mutation and evolution
- Real-time alerting and emergency stops
- Performance regression detection
- Market regime adaptation tracking
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import json
import asyncio
from pathlib import Path
import warnings
from abc import ABC, abstractmethod

# Statistical imports with fallbacks
try:
    from scipy import stats
    from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available - statistical tests disabled")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class StrategyHealthMetrics:
    """Comprehensive health metrics for a strategy"""
    strategy_name: str
    timestamp: datetime
    
    # Performance metrics
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    volatility: float
    
    # Operational metrics
    trade_frequency: float
    avg_trade_duration: float
    error_rate: float
    execution_latency: float
    
    # Risk metrics
    var_95: float  # Value at Risk
    tail_risk: float
    correlation_breakdown: bool
    position_concentration: float
    
    # Market adaptation metrics
    regime_sensitivity: float
    volatility_adaptation: float
    trend_following_ability: float
    
    # Composite scores
    overall_health_score: float
    performance_score: float
    risk_score: float
    stability_score: float
    
    # Additional data
    trade_count: int = 0
    active_positions: int = 0
    last_signal_time: Optional[datetime] = None
    alerts: List[str] = field(default_factory=list)

@dataclass
class StrategyVariation:
    """Strategy variation for A/B testing"""
    variation_id: str
    base_strategy: str
    parameter_changes: Dict[str, Any]
    creation_time: datetime
    test_start_time: Optional[datetime] = None
    test_end_time: Optional[datetime] = None
    
    # Performance tracking
    trades_executed: int = 0
    total_pnl: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    max_drawdown: float = 0.0
    
    # Test status
    status: str = "created"  # created, testing, completed, failed
    confidence_level: float = 0.0
    statistical_significance: float = 0.0

@dataclass
class ABTestResult:
    """A/B test comparison result"""
    test_id: str
    base_strategy: str
    variation_strategy: str
    test_duration: timedelta
    
    # Statistical results
    performance_improvement: float
    statistical_significance: float
    confidence_interval: Tuple[float, float]
    p_value: float
    
    # Detailed metrics
    base_metrics: Dict[str, float]
    variation_metrics: Dict[str, float]
    
    # Decision
    recommended_action: str  # "adopt", "reject", "extend_test"
    reasoning: str

class HealthMonitor:
    """
    Real-time strategy health monitoring system
    
    Continuously tracks strategy performance and detects degradation
    """
    
    def __init__(self, alert_thresholds: Dict[str, float] = None):
        self.alert_thresholds = alert_thresholds or {
            'min_win_rate': 0.45,
            'max_drawdown': 0.15,
            'min_sharpe': 0.5,
            'max_error_rate': 0.05,
            'min_health_score': 0.6
        }
        
        self.strategy_histories: Dict[str, List[StrategyHealthMetrics]] = {}
        self.active_alerts: Dict[str, List[str]] = {}
        self.emergency_stops: Dict[str, bool] = {}
        
        logger.info("🏥 Strategy Health Monitor initialized")
    
    async def monitor_strategy(self, strategy_name: str, 
                             performance_data: Dict[str, Any],
                             market_data: pd.DataFrame) -> StrategyHealthMetrics:
        """Monitor strategy health and generate metrics"""
        
        # Calculate comprehensive health metrics
        health_metrics = await self._calculate_health_metrics(
            strategy_name, performance_data, market_data
        )
        
        # Store in history
        if strategy_name not in self.strategy_histories:
            self.strategy_histories[strategy_name] = []
        
        self.strategy_histories[strategy_name].append(health_metrics)
        
        # Keep only last 1000 entries
        if len(self.strategy_histories[strategy_name]) > 1000:
            self.strategy_histories[strategy_name] = self.strategy_histories[strategy_name][-1000:]
        
        # Check for alerts
        await self._check_alerts(health_metrics)
        
        # Detect performance degradation
        await self._detect_degradation(strategy_name, health_metrics)
        
        return health_metrics
    
    async def _calculate_health_metrics(self, strategy_name: str,
                                      performance_data: Dict[str, Any],
                                      market_data: pd.DataFrame) -> StrategyHealthMetrics:
        """Calculate comprehensive health metrics"""
        
        trades = performance_data.get('trades', [])
        returns = performance_data.get('returns', [])
        
        # Performance metrics
        win_rate = self._calculate_win_rate(trades)
        profit_factor = self._calculate_profit_factor(trades)
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        max_drawdown = self._calculate_max_drawdown(returns)
        total_return = sum(returns) if returns else 0.0
        volatility = np.std(returns) if returns else 0.0
        
        # Operational metrics
        trade_frequency = len(trades) / max(1, (datetime.now() - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).days or 1)
        avg_trade_duration = self._calculate_avg_trade_duration(trades)
        error_rate = performance_data.get('error_rate', 0.0)
        execution_latency = performance_data.get('avg_latency', 0.0)
        
        # Risk metrics
        var_95 = np.percentile(returns, 5) if returns else 0.0
        tail_risk = self._calculate_tail_risk(returns)
        correlation_breakdown = self._detect_correlation_breakdown(market_data, returns)
        position_concentration = performance_data.get('position_concentration', 0.0)
        
        # Market adaptation metrics
        regime_sensitivity = self._calculate_regime_sensitivity(returns, market_data)
        volatility_adaptation = self._calculate_volatility_adaptation(returns, market_data)
        trend_following_ability = self._calculate_trend_following(returns, market_data)
        
        # Composite scores
        performance_score = self._calculate_performance_score(win_rate, profit_factor, sharpe_ratio)
        risk_score = self._calculate_risk_score(max_drawdown, var_95, tail_risk)
        stability_score = self._calculate_stability_score(error_rate, volatility)
        overall_health_score = (performance_score * 0.4 + risk_score * 0.3 + stability_score * 0.3)
        
        return StrategyHealthMetrics(
            strategy_name=strategy_name,
            timestamp=datetime.now(),
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_return=total_return,
            volatility=volatility,
            trade_frequency=trade_frequency,
            avg_trade_duration=avg_trade_duration,
            error_rate=error_rate,
            execution_latency=execution_latency,
            var_95=var_95,
            tail_risk=tail_risk,
            correlation_breakdown=correlation_breakdown,
            position_concentration=position_concentration,
            regime_sensitivity=regime_sensitivity,
            volatility_adaptation=volatility_adaptation,
            trend_following_ability=trend_following_ability,
            overall_health_score=overall_health_score,
            performance_score=performance_score,
            risk_score=risk_score,
            stability_score=stability_score,
            trade_count=len(trades),
            active_positions=performance_data.get('active_positions', 0),
            last_signal_time=performance_data.get('last_signal_time')
        )
    
    async def _check_alerts(self, metrics: StrategyHealthMetrics):
        """Check for alert conditions"""
        
        alerts = []
        
        if metrics.win_rate < self.alert_thresholds['min_win_rate']:
            alerts.append(f"⚠️ Low win rate: {metrics.win_rate:.1%}")
        
        if metrics.max_drawdown > self.alert_thresholds['max_drawdown']:
            alerts.append(f"🔴 High drawdown: {metrics.max_drawdown:.1%}")
        
        if metrics.sharpe_ratio < self.alert_thresholds['min_sharpe']:
            alerts.append(f"📉 Low Sharpe ratio: {metrics.sharpe_ratio:.2f}")
        
        if metrics.error_rate > self.alert_thresholds['max_error_rate']:
            alerts.append(f"⚡ High error rate: {metrics.error_rate:.1%}")
        
        if metrics.overall_health_score < self.alert_thresholds['min_health_score']:
            alerts.append(f"🏥 Poor health score: {metrics.overall_health_score:.2f}")
        
        if metrics.correlation_breakdown:
            alerts.append("💥 Correlation breakdown detected")
        
        # Emergency stop conditions
        if metrics.max_drawdown > 0.25 or metrics.error_rate > 0.15:
            self.emergency_stops[metrics.strategy_name] = True
            alerts.append("🛑 EMERGENCY STOP TRIGGERED")
        
        if alerts:
            self.active_alerts[metrics.strategy_name] = alerts
            logger.warning(f"Alerts for {metrics.strategy_name}: {', '.join(alerts)}")
        else:
            self.active_alerts.pop(metrics.strategy_name, None)
    
    async def _detect_degradation(self, strategy_name: str, current_metrics: StrategyHealthMetrics):
        """Detect performance degradation using historical comparison"""
        
        history = self.strategy_histories.get(strategy_name, [])
        if len(history) < 10:  # Need history for comparison
            return
        
        # Compare current performance to historical baseline
        recent_scores = [m.overall_health_score for m in history[-30:]]  # Last 30 measurements
        baseline_scores = [m.overall_health_score for m in history[-100:-30]]  # Previous period
        
        if len(baseline_scores) < 10:
            return
        
        # Statistical test for degradation
        if SCIPY_AVAILABLE:
            try:
                statistic, p_value = ttest_ind(recent_scores, baseline_scores, alternative='less')
                
                if p_value < 0.05:  # Significant degradation
                    degradation_alert = f"📊 Performance degradation detected (p={p_value:.3f})"
                    if strategy_name not in self.active_alerts:
                        self.active_alerts[strategy_name] = []
                    self.active_alerts[strategy_name].append(degradation_alert)
                    logger.warning(f"{strategy_name}: {degradation_alert}")
                    
            except Exception as e:
                logger.warning(f"Could not perform degradation test: {e}")
    
    def _calculate_win_rate(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate win rate from trades"""
        if not trades:
            return 0.0
        
        winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
        return winning_trades / len(trades)
    
    def _calculate_profit_factor(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate profit factor"""
        if not trades:
            return 0.0
        
        gross_profit = sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) > 0)
        gross_loss = abs(sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) < 0))
        
        return gross_profit / max(gross_loss, 0.001)
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        return (avg_return / max(std_return, 0.001)) * np.sqrt(252)  # Annualized
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not returns:
            return 0.0
        
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        
        return abs(min(drawdowns)) if len(drawdowns) > 0 else 0.0
    
    def _calculate_avg_trade_duration(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate average trade duration in hours"""
        if not trades:
            return 0.0
        
        durations = []
        for trade in trades:
            if 'entry_time' in trade and 'exit_time' in trade:
                entry = trade['entry_time']
                exit = trade['exit_time']
                if isinstance(entry, str):
                    entry = datetime.fromisoformat(entry.replace('Z', '+00:00'))
                if isinstance(exit, str):
                    exit = datetime.fromisoformat(exit.replace('Z', '+00:00'))
                
                duration = (exit - entry).total_seconds() / 3600  # Hours
                durations.append(duration)
        
        return np.mean(durations) if durations else 0.0
    
    def _calculate_tail_risk(self, returns: List[float]) -> float:
        """Calculate tail risk (average of worst 5% returns)"""
        if not returns or len(returns) < 20:
            return 0.0
        
        sorted_returns = sorted(returns)
        tail_size = max(1, len(returns) // 20)  # Bottom 5%
        tail_returns = sorted_returns[:tail_size]
        
        return abs(np.mean(tail_returns))
    
    def _detect_correlation_breakdown(self, market_data: pd.DataFrame, returns: List[float]) -> bool:
        """Detect if strategy correlation with market has broken down"""
        if len(returns) < 30 or len(market_data) < 30:
            return False
        
        # Calculate market returns
        market_returns = market_data['close'].pct_change().dropna().tolist()
        
        # Take last 30 periods for both
        recent_strategy_returns = returns[-30:]
        recent_market_returns = market_returns[-30:] if len(market_returns) >= 30 else market_returns
        
        # Ensure same length
        min_length = min(len(recent_strategy_returns), len(recent_market_returns))
        recent_strategy_returns = recent_strategy_returns[-min_length:]
        recent_market_returns = recent_market_returns[-min_length:]
        
        if min_length < 10:
            return False
        
        # Calculate correlation
        correlation = np.corrcoef(recent_strategy_returns, recent_market_returns)[0, 1]
        
        # Detect breakdown (correlation becomes too extreme or NaN)
        return np.isnan(correlation) or abs(correlation) > 0.95
    
    def _calculate_regime_sensitivity(self, returns: List[float], market_data: pd.DataFrame) -> float:
        """Calculate how well strategy adapts to market regimes"""
        if len(returns) < 50:
            return 0.5
        
        # Simplified regime detection based on volatility
        market_returns = market_data['close'].pct_change().dropna()
        volatility = market_returns.rolling(20).std()
        
        high_vol_periods = volatility > volatility.quantile(0.7)
        low_vol_periods = volatility < volatility.quantile(0.3)
        
        # Check if strategy performs differently in different regimes
        # (This is a simplified implementation)
        return np.random.uniform(0.4, 0.8)  # Placeholder
    
    def _calculate_volatility_adaptation(self, returns: List[float], market_data: pd.DataFrame) -> float:
        """Calculate how well strategy adapts to volatility changes"""
        return np.random.uniform(0.4, 0.8)  # Placeholder
    
    def _calculate_trend_following(self, returns: List[float], market_data: pd.DataFrame) -> float:
        """Calculate trend following ability"""
        return np.random.uniform(0.4, 0.8)  # Placeholder
    
    def _calculate_performance_score(self, win_rate: float, profit_factor: float, sharpe_ratio: float) -> float:
        """Calculate composite performance score"""
        score = 0.0
        score += min(win_rate * 2, 1.0) * 0.4  # Win rate component
        score += min(profit_factor / 2, 1.0) * 0.3  # Profit factor component
        score += min(max(sharpe_ratio, 0) / 2, 1.0) * 0.3  # Sharpe component
        return max(0.0, min(1.0, score))
    
    def _calculate_risk_score(self, max_drawdown: float, var_95: float, tail_risk: float) -> float:
        """Calculate composite risk score (higher is better - lower risk)"""
        score = 1.0
        score -= min(max_drawdown * 4, 0.8)  # Penalize high drawdown
        score -= min(abs(var_95) * 10, 0.3)  # Penalize high VaR
        score -= min(tail_risk * 5, 0.2)  # Penalize tail risk
        return max(0.0, min(1.0, score))
    
    def _calculate_stability_score(self, error_rate: float, volatility: float) -> float:
        """Calculate stability score"""
        score = 1.0
        score -= min(error_rate * 10, 0.5)  # Penalize errors
        score -= min(volatility * 2, 0.5)  # Penalize high volatility
        return max(0.0, min(1.0, score))

class ABTestManager:
    """
    A/B Testing system for strategy variations
    
    Automatically generates and tests strategy variations
    """
    
    def __init__(self, min_test_duration: timedelta = timedelta(days=7),
                 min_sample_size: int = 50,
                 significance_level: float = 0.05):
        self.min_test_duration = min_test_duration
        self.min_sample_size = min_sample_size
        self.significance_level = significance_level
        
        self.active_tests: Dict[str, StrategyVariation] = {}
        self.completed_tests: List[ABTestResult] = []
        self.variation_generators: Dict[str, callable] = {}
        
        logger.info("🧪 A/B Test Manager initialized")
    
    async def create_variation(self, base_strategy: str, 
                             variation_type: str = "parameter_optimization") -> StrategyVariation:
        """Create a new strategy variation for testing"""
        
        variation_id = f"{base_strategy}_var_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Generate parameter changes based on variation type
        parameter_changes = await self._generate_parameter_changes(base_strategy, variation_type)
        
        variation = StrategyVariation(
            variation_id=variation_id,
            base_strategy=base_strategy,
            parameter_changes=parameter_changes,
            creation_time=datetime.now()
        )
        
        logger.info(f"🧬 Created variation {variation_id} for {base_strategy}")
        return variation
    
    async def start_ab_test(self, base_strategy: str, variation: StrategyVariation,
                          allocation_ratio: float = 0.5) -> str:
        """Start A/B test between base strategy and variation"""
        
        test_id = f"test_{base_strategy}_{variation.variation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        variation.test_start_time = datetime.now()
        variation.status = "testing"
        
        self.active_tests[test_id] = variation
        
        logger.info(f"🚀 Started A/B test {test_id} with {allocation_ratio:.0%} allocation to variation")
        
        return test_id
    
    async def update_test_results(self, test_id: str, variation_trade: Dict[str, Any]):
        """Update A/B test with new trade result"""
        
        if test_id not in self.active_tests:
            return
        
        variation = self.active_tests[test_id]
        
        # Update variation performance
        variation.trades_executed += 1
        variation.total_pnl += variation_trade.get('pnl', 0.0)
        
        if variation_trade.get('pnl', 0.0) > 0:
            variation.win_count += 1
        else:
            variation.loss_count += 1
        
        # Update drawdown
        if variation.total_pnl < 0:
            variation.max_drawdown = max(variation.max_drawdown, abs(variation.total_pnl))
        
        # Check if test should be completed
        await self._check_test_completion(test_id)
    
    async def _check_test_completion(self, test_id: str):
        """Check if A/B test should be completed"""
        
        variation = self.active_tests[test_id]
        
        # Check minimum requirements
        test_duration = datetime.now() - variation.test_start_time
        sufficient_duration = test_duration >= self.min_test_duration
        sufficient_samples = variation.trades_executed >= self.min_sample_size
        
        if not (sufficient_duration and sufficient_samples):
            return
        
        # Perform statistical test
        result = await self._analyze_ab_test(test_id)
        
        if result:
            # Complete the test
            variation.status = "completed"
            variation.test_end_time = datetime.now()
            variation.statistical_significance = result.statistical_significance
            
            self.completed_tests.append(result)
            del self.active_tests[test_id]
            
            logger.info(f"✅ Completed A/B test {test_id}: {result.recommended_action}")
    
    async def _analyze_ab_test(self, test_id: str) -> Optional[ABTestResult]:
        """Analyze A/B test results and make recommendation"""
        
        variation = self.active_tests[test_id]
        
        # Get base strategy performance (simulated for now)
        base_metrics = await self._get_base_strategy_metrics(variation.base_strategy)
        
        if not base_metrics:
            return None
        
        # Calculate variation metrics
        variation_win_rate = variation.win_count / max(variation.trades_executed, 1)
        variation_avg_pnl = variation.total_pnl / max(variation.trades_executed, 1)
        
        variation_metrics = {
            'win_rate': variation_win_rate,
            'avg_pnl_per_trade': variation_avg_pnl,
            'total_pnl': variation.total_pnl,
            'max_drawdown': variation.max_drawdown,
            'trade_count': variation.trades_executed
        }
        
        # Statistical significance test
        performance_improvement = variation_avg_pnl - base_metrics['avg_pnl_per_trade']
        
        # Simplified statistical test
        if SCIPY_AVAILABLE and variation.trades_executed >= 30:
            # Mock t-test (in real implementation, would use actual trade data)
            t_stat = performance_improvement / max(0.001, np.sqrt(base_metrics.get('variance', 0.001)))
            p_value = stats.t.sf(np.abs(t_stat), variation.trades_executed - 1) * 2
            statistical_significance = 1 - p_value
        else:
            p_value = 0.5
            statistical_significance = 0.5
        
        # Confidence interval (simplified)
        std_error = np.sqrt(base_metrics.get('variance', 0.001) / variation.trades_executed)
        confidence_interval = (
            performance_improvement - 1.96 * std_error,
            performance_improvement + 1.96 * std_error
        )
        
        # Make recommendation
        if statistical_significance > 0.95 and performance_improvement > 0:
            recommended_action = "adopt"
            reasoning = f"Variation shows {performance_improvement:.4f} improvement with {statistical_significance:.1%} confidence"
        elif statistical_significance > 0.95 and performance_improvement < 0:
            recommended_action = "reject"
            reasoning = f"Variation shows {abs(performance_improvement):.4f} degradation with {statistical_significance:.1%} confidence"
        elif variation.trades_executed < self.min_sample_size * 2:
            recommended_action = "extend_test"
            reasoning = "Insufficient sample size for conclusive results"
        else:
            recommended_action = "reject"
            reasoning = "No significant improvement detected"
        
        return ABTestResult(
            test_id=test_id,
            base_strategy=variation.base_strategy,
            variation_strategy=variation.variation_id,
            test_duration=datetime.now() - variation.test_start_time,
            performance_improvement=performance_improvement,
            statistical_significance=statistical_significance,
            confidence_interval=confidence_interval,
            p_value=p_value,
            base_metrics=base_metrics,
            variation_metrics=variation_metrics,
            recommended_action=recommended_action,
            reasoning=reasoning
        )
    
    async def _generate_parameter_changes(self, base_strategy: str, variation_type: str) -> Dict[str, Any]:
        """Generate parameter changes for strategy variation"""
        
        # This would analyze the base strategy's parameters and generate variations
        # For now, using common parameter variations
        
        if variation_type == "parameter_optimization":
            return {
                'risk_multiplier': np.random.uniform(0.8, 1.2),
                'signal_threshold': np.random.uniform(0.9, 1.1),
                'position_size_multiplier': np.random.uniform(0.85, 1.15),
                'stop_loss_multiplier': np.random.uniform(0.9, 1.1),
                'take_profit_multiplier': np.random.uniform(0.9, 1.1)
            }
        elif variation_type == "signal_enhancement":
            return {
                'additional_filters': ['volume_confirmation', 'momentum_filter'],
                'filter_strength': np.random.uniform(0.5, 1.5)
            }
        elif variation_type == "risk_adjustment":
            return {
                'max_position_size': np.random.uniform(0.05, 0.15),
                'correlation_limit': np.random.uniform(0.6, 0.9),
                'volatility_adjustment': True
            }
        
        return {}
    
    async def _get_base_strategy_metrics(self, strategy_name: str) -> Dict[str, Any]:
        """Get base strategy performance metrics"""
        
        # In real implementation, would fetch from strategy performance history
        # Mock data for now
        return {
            'win_rate': np.random.uniform(0.5, 0.7),
            'avg_pnl_per_trade': np.random.uniform(0.001, 0.01),
            'total_pnl': np.random.uniform(0.1, 1.0),
            'max_drawdown': np.random.uniform(0.05, 0.15),
            'trade_count': np.random.randint(100, 500),
            'variance': np.random.uniform(0.0001, 0.001)
        }
    
    def get_active_tests_summary(self) -> Dict[str, Any]:
        """Get summary of active A/B tests"""
        
        return {
            'total_active_tests': len(self.active_tests),
            'active_tests': {
                test_id: {
                    'base_strategy': var.base_strategy,
                    'variation_id': var.variation_id,
                    'trades_executed': var.trades_executed,
                    'current_pnl': var.total_pnl,
                    'test_duration': str(datetime.now() - var.test_start_time),
                    'status': var.status
                }
                for test_id, var in self.active_tests.items()
            },
            'completed_tests': len(self.completed_tests),
            'recent_results': [
                {
                    'test_id': result.test_id,
                    'recommended_action': result.recommended_action,
                    'performance_improvement': result.performance_improvement,
                    'statistical_significance': result.statistical_significance
                }
                for result in self.completed_tests[-5:]  # Last 5 results
            ]
        }

class HealthAndABTestingSystem:
    """
    Combined health monitoring and A/B testing system
    
    Integrates health monitoring with automatic A/B testing for continuous improvement
    """
    
    def __init__(self):
        self.health_monitor = HealthMonitor()
        self.ab_test_manager = ABTestManager()
        
        # Integration settings
        self.auto_test_threshold = 0.6  # Start A/B test if health score drops below this
        self.test_frequency = timedelta(days=14)  # How often to create new variations
        self.last_variation_creation: Dict[str, datetime] = {}
        
        logger.info("🔬 Health & A/B Testing System initialized")
    
    async def monitor_and_optimize(self, strategy_name: str,
                                 performance_data: Dict[str, Any],
                                 market_data: pd.DataFrame) -> Dict[str, Any]:
        """Monitor strategy health and trigger optimization if needed"""
        
        # Monitor health
        health_metrics = await self.health_monitor.monitor_strategy(
            strategy_name, performance_data, market_data
        )
        
        # Check if automatic A/B testing should be triggered
        await self._check_auto_testing(strategy_name, health_metrics)
        
        # Update any active A/B tests for this strategy
        await self._update_strategy_tests(strategy_name, performance_data)
        
        return {
            'health_metrics': health_metrics,
            'active_alerts': self.health_monitor.active_alerts.get(strategy_name, []),
            'emergency_stop': self.health_monitor.emergency_stops.get(strategy_name, False),
            'active_tests': [
                test_id for test_id, var in self.ab_test_manager.active_tests.items()
                if var.base_strategy == strategy_name
            ]
        }
    
    async def _check_auto_testing(self, strategy_name: str, health_metrics: StrategyHealthMetrics):
        """Check if automatic A/B testing should be triggered"""
        
        # Don't test if already have active tests for this strategy
        active_tests_for_strategy = [
            test_id for test_id, var in self.ab_test_manager.active_tests.items()
            if var.base_strategy == strategy_name
        ]
        
        if len(active_tests_for_strategy) >= 2:  # Max 2 concurrent tests per strategy
            return
        
        # Check if it's time for a new variation
        last_creation = self.last_variation_creation.get(strategy_name)
        if last_creation and datetime.now() - last_creation < self.test_frequency:
            return
        
        # Trigger test if health score is low or regularly for optimization
        should_test = (
            health_metrics.overall_health_score < self.auto_test_threshold or
            (last_creation is None or datetime.now() - last_creation > self.test_frequency * 2)
        )
        
        if should_test:
            # Create and start A/B test
            variation_type = "parameter_optimization" if health_metrics.overall_health_score > 0.5 else "risk_adjustment"
            
            variation = await self.ab_test_manager.create_variation(strategy_name, variation_type)
            test_id = await self.ab_test_manager.start_ab_test(strategy_name, variation, 0.3)  # 30% allocation
            
            self.last_variation_creation[strategy_name] = datetime.now()
            
            logger.info(f"🧪 Auto-triggered A/B test {test_id} for {strategy_name} (health: {health_metrics.overall_health_score:.2f})")
    
    async def _update_strategy_tests(self, strategy_name: str, performance_data: Dict[str, Any]):
        """Update A/B tests for strategy with latest performance data"""
        
        active_tests_for_strategy = [
            test_id for test_id, var in self.ab_test_manager.active_tests.items()
            if var.base_strategy == strategy_name
        ]
        
        # Simulate variation trade results (in real implementation, would track actual variation trades)
        for test_id in active_tests_for_strategy:
            # Mock variation performance (slightly different from base)
            base_trades = performance_data.get('trades', [])
            if base_trades:
                latest_trade = base_trades[-1].copy()
                # Simulate variation with slightly different performance
                latest_trade['pnl'] *= np.random.uniform(0.95, 1.05)
                await self.ab_test_manager.update_test_results(test_id, latest_trade)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            'health_monitoring': {
                'monitored_strategies': len(self.health_monitor.strategy_histories),
                'active_alerts': len(self.health_monitor.active_alerts),
                'emergency_stops': len([k for k, v in self.health_monitor.emergency_stops.items() if v]),
                'alert_thresholds': self.health_monitor.alert_thresholds
            },
            'ab_testing': self.ab_test_manager.get_active_tests_summary()
        }

# Utility functions
async def create_health_and_testing_system() -> HealthAndABTestingSystem:
    """Create integrated health monitoring and A/B testing system"""
    system = HealthAndABTestingSystem()
    logger.info("🎯 Created integrated health monitoring and A/B testing system")
    return system

if __name__ == "__main__":
    # Test the health monitoring and A/B testing system
    async def test_system():
        print("🧪 Testing Health Monitoring and A/B Testing System...")
        
        system = await create_health_and_testing_system()
        
        # Mock performance data
        mock_performance = {
            'trades': [
                {'pnl': 0.02, 'entry_time': datetime.now() - timedelta(hours=2), 'exit_time': datetime.now() - timedelta(hours=1)},
                {'pnl': -0.01, 'entry_time': datetime.now() - timedelta(hours=1), 'exit_time': datetime.now()},
            ],
            'returns': [0.02, -0.01, 0.015, -0.005, 0.01],
            'error_rate': 0.02,
            'avg_latency': 0.1,
            'active_positions': 2
        }
        
        # Mock market data
        mock_market_data = pd.DataFrame({
            'open': [45000, 45100, 45200, 45150, 45300],
            'high': [45200, 45300, 45400, 45250, 45400],
            'low': [44900, 45000, 45100, 45050, 45200],
            'close': [45100, 45200, 45300, 45250, 45350],
            'volume': [1000000, 1100000, 1200000, 1050000, 1150000]
        })
        
        # Test monitoring
        result = await system.monitor_and_optimize('test_strategy', mock_performance, mock_market_data)
        
        print(f"📊 Health Score: {result['health_metrics'].overall_health_score:.2f}")
        print(f"⚠️ Alerts: {len(result['active_alerts'])}")
        print(f"🧪 Active Tests: {len(result['active_tests'])}")
        
        # Test system status
        status = system.get_system_status()
        print(f"🏥 System Status: {status}")
        
        print("✅ Health monitoring and A/B testing system test completed!")
    
    asyncio.run(test_system())