#!/usr/bin/env python3
"""
ML Altcoin Trading Bot - Learning Pipeline
Kontinuierliches Lernen aus historischen Trading-Daten

Diese Pipeline:
- Analysiert täglich alle geloggten Daten
- Findet erfolgreiche Strategie-Kombinationen
- Identifiziert Schwächen einzelner Strategien
- Generiert neue Orchestrator-Regeln
- Erkennt Muster in Verlusten
- Berechnet optimale Gewichtungen
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import warnings
from pathlib import Path

import asyncpg
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

@dataclass
class StrategyInsight:
    """Strategy performance insight"""
    strategy_name: str
    insight_type: str  # 'strength', 'weakness', 'opportunity', 'threat'
    description: str
    confidence: float
    impact_score: float
    recommendation: str
    supporting_data: Dict[str, Any]
    timestamp: datetime

@dataclass
class CombinationAnalysis:
    """Strategy combination analysis result"""
    strategies: List[str]
    synergy_score: float
    optimal_weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    confidence: float
    market_conditions: List[str]  # When this combination works best
    supporting_trades: int

@dataclass
class LossPattern:
    """Pattern in trading losses"""
    pattern_name: str
    description: str
    frequency: int
    avg_loss: float
    total_impact: float
    triggers: List[str]
    affected_strategies: List[str]
    market_conditions: Dict[str, Any]
    prevention_suggestions: List[str]

@dataclass
class OrchestratorRule:
    """New orchestrator rule generated from learning"""
    rule_id: str
    rule_name: str
    rule_type: str  # 'allocation', 'risk_management', 'timing', 'combination'
    condition: str  # Human-readable condition
    action: str  # What to do when condition is met
    confidence: float
    expected_improvement: float
    backtest_results: Dict[str, float]
    implementation_code: str  # Python code to implement

class LearningPipeline:
    """Main learning pipeline for continuous improvement"""
    
    def __init__(self, db_pool: asyncpg.Pool, 
                 lookback_days: int = 30,
                 min_trades_for_analysis: int = 10,
                 confidence_threshold: float = 0.7):
        """
        Initialize Learning Pipeline
        
        Args:
            db_pool: Database connection pool
            lookback_days: Days of data to analyze
            min_trades_for_analysis: Minimum trades needed for strategy analysis
            confidence_threshold: Minimum confidence for insights
        """
        self.db_pool = db_pool
        self.lookback_days = lookback_days
        self.min_trades_for_analysis = min_trades_for_analysis
        self.confidence_threshold = confidence_threshold
        
        # Analysis results storage
        self.strategy_insights: List[StrategyInsight] = []
        self.combination_analyses: List[CombinationAnalysis] = []
        self.loss_patterns: List[LossPattern] = []
        self.new_rules: List[OrchestratorRule] = []
        
        # Data cache
        self._trades_df: Optional[pd.DataFrame] = None
        self._decisions_df: Optional[pd.DataFrame] = None
        self._market_states_df: Optional[pd.DataFrame] = None
        
        self.results_dir = Path("analysis/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    async def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run complete learning pipeline analysis
        
        Returns:
            Dict with analysis results summary
        """
        logger.info("🧠 Starting full learning pipeline analysis...")
        
        start_time = datetime.utcnow()
        
        try:
            # Step 1: Load and prepare data
            await self._load_analysis_data()
            
            # Step 2: Analyze individual strategies
            await self._analyze_individual_strategies()
            
            # Step 3: Find successful combinations
            await self._analyze_strategy_combinations()
            
            # Step 4: Identify loss patterns
            await self._identify_loss_patterns()
            
            # Step 5: Generate new orchestrator rules
            await self._generate_orchestrator_rules()
            
            # Step 6: Calculate optimal weights
            optimal_weights = await self._calculate_optimal_weights()
            
            # Step 7: Save results
            results_summary = await self._save_analysis_results()
            
            # Step 8: Generate visualizations
            await self._generate_visualizations()
            
            analysis_time = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(f"✅ Learning pipeline completed in {analysis_time:.1f}s")
            
            return {
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'analysis_duration_seconds': analysis_time,
                'data_analyzed': {
                    'trades': len(self._trades_df) if self._trades_df is not None else 0,
                    'decisions': len(self._decisions_df) if self._decisions_df is not None else 0,
                    'market_states': len(self._market_states_df) if self._market_states_df is not None else 0
                },
                'insights_generated': {
                    'strategy_insights': len(self.strategy_insights),
                    'combination_analyses': len(self.combination_analyses),
                    'loss_patterns': len(self.loss_patterns),
                    'new_rules': len(self.new_rules)
                },
                'optimal_weights': optimal_weights,
                'key_findings': self._get_key_findings(),
                'recommendations': self._get_top_recommendations(),
                'results_saved_to': str(self.results_dir)
            }
            
        except Exception as e:
            logger.error(f"Learning pipeline failed: {e}")
            raise

    async def _load_analysis_data(self):
        """Load all necessary data for analysis"""
        logger.info("📊 Loading analysis data...")
        
        cutoff_date = datetime.utcnow() - timedelta(days=self.lookback_days)
        
        async with self.db_pool.acquire() as conn:
            # Load trade performance data
            trades_query = """
                SELECT * FROM strategy_performance 
                WHERE timestamp >= $1 
                ORDER BY timestamp DESC
            """
            trades_rows = await conn.fetch(trades_query, cutoff_date)
            self._trades_df = pd.DataFrame([dict(row) for row in trades_rows])
            
            # Load orchestrator decisions
            decisions_query = """
                SELECT * FROM orchestrator_decisions 
                WHERE timestamp >= $1 
                ORDER BY timestamp DESC
            """
            decisions_rows = await conn.fetch(decisions_query, cutoff_date)  
            self._decisions_df = pd.DataFrame([dict(row) for row in decisions_rows])
            
            # Load market states
            market_query = """
                SELECT * FROM market_states 
                WHERE timestamp >= $1 
                ORDER BY timestamp DESC
            """
            market_rows = await conn.fetch(market_query, cutoff_date)
            self._market_states_df = pd.DataFrame([dict(row) for row in market_rows])
        
        logger.info(f"Loaded {len(self._trades_df)} trades, {len(self._decisions_df)} decisions, {len(self._market_states_df)} market states")

    async def _analyze_individual_strategies(self):
        """Analyze performance of individual strategies"""
        logger.info("🔍 Analyzing individual strategies...")
        
        if self._trades_df.empty:
            logger.warning("No trade data available for strategy analysis")
            return
        
        # Group by strategy
        strategy_groups = self._trades_df.groupby('strategy_name')
        
        for strategy_name, group in strategy_groups:
            if len(group) < self.min_trades_for_analysis:
                continue
                
            # Calculate performance metrics
            closed_trades = group[group['trade_status'] == 'closed']
            if closed_trades.empty:
                continue
                
            performance_metrics = self._calculate_strategy_metrics(closed_trades)
            
            # Identify strengths and weaknesses
            insights = self._identify_strategy_insights(strategy_name, performance_metrics, closed_trades)
            self.strategy_insights.extend(insights)
            
        logger.info(f"Generated {len(self.strategy_insights)} strategy insights")

    def _calculate_strategy_metrics(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive strategy performance metrics"""
        if trades_df.empty or 'pnl_percentage' not in trades_df.columns:
            return {}
            
        returns = trades_df['pnl_percentage'].dropna()
        if returns.empty:
            return {}
        
        metrics = {
            'total_trades': len(trades_df),
            'win_rate': (returns > 0).mean(),
            'avg_return': returns.mean(),
            'return_std': returns.std(),
            'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
            'max_return': returns.max(),
            'min_return': returns.min(),
            'total_pnl': trades_df['pnl_absolute'].sum() if 'pnl_absolute' in trades_df.columns else 0,
            'avg_duration_hours': trades_df['duration_minutes'].mean() / 60 if 'duration_minutes' in trades_df.columns else 0,
            'max_drawdown': self._calculate_max_drawdown(returns),
            'profit_factor': self._calculate_profit_factor(returns),
            'kelly_percentage': self._calculate_kelly_percentage(returns)
        }
        
        return metrics

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if returns.empty:
            return 0.0
            
        cumulative = (1 + returns / 100).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min() * 100

    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor"""
        if returns.empty:
            return 0.0
            
        profits = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        return profits / losses if losses > 0 else float('inf')

    def _calculate_kelly_percentage(self, returns: pd.Series) -> float:
        """Calculate Kelly criterion percentage"""
        if returns.empty:
            return 0.0
            
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        if wins.empty or losses.empty:
            return 0.0
            
        win_rate = len(wins) / len(returns)
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        
        if avg_loss == 0:
            return 0.0
            
        kelly = win_rate - ((1 - win_rate) * avg_loss / avg_win)
        return max(0, min(1, kelly))  # Constrain between 0 and 1

    def _identify_strategy_insights(self, strategy_name: str, metrics: Dict[str, float], trades_df: pd.DataFrame) -> List[StrategyInsight]:
        """Identify insights for a specific strategy"""
        insights = []
        
        # Strength: High Sharpe ratio
        if metrics.get('sharpe_ratio', 0) > 1.5:
            insights.append(StrategyInsight(
                strategy_name=strategy_name,
                insight_type='strength',
                description=f"Excellent risk-adjusted returns with Sharpe ratio of {metrics['sharpe_ratio']:.2f}",
                confidence=0.9,
                impact_score=metrics['sharpe_ratio'] / 2,
                recommendation="Increase allocation to this strategy",
                supporting_data={'sharpe_ratio': metrics['sharpe_ratio'], 'total_trades': metrics['total_trades']},
                timestamp=datetime.utcnow()
            ))
        
        # Weakness: Low win rate but might be offset by profit factor
        if metrics.get('win_rate', 0) < 0.4 and metrics.get('profit_factor', 0) < 1.5:
            insights.append(StrategyInsight(
                strategy_name=strategy_name,
                insight_type='weakness',
                description=f"Low win rate ({metrics['win_rate']:.1%}) with poor profit factor",
                confidence=0.8,
                impact_score=-abs(0.4 - metrics['win_rate']),
                recommendation="Review entry criteria and consider tighter risk management",
                supporting_data={'win_rate': metrics['win_rate'], 'profit_factor': metrics['profit_factor']},
                timestamp=datetime.utcnow()
            ))
        
        # Opportunity: Good performance in specific market conditions
        market_performance = self._analyze_market_condition_performance(trades_df)
        for condition, perf in market_performance.items():
            if perf['avg_return'] > metrics.get('avg_return', 0) * 1.5:
                insights.append(StrategyInsight(
                    strategy_name=strategy_name,
                    insight_type='opportunity',
                    description=f"Exceptional performance in {condition} market conditions",
                    confidence=0.7,
                    impact_score=perf['avg_return'] - metrics.get('avg_return', 0),
                    recommendation=f"Increase allocation during {condition} conditions",
                    supporting_data=perf,
                    timestamp=datetime.utcnow()
                ))
        
        # Threat: High drawdown
        if metrics.get('max_drawdown', 0) < -15:  # More than 15% drawdown
            insights.append(StrategyInsight(
                strategy_name=strategy_name,
                insight_type='threat',
                description=f"High maximum drawdown of {metrics['max_drawdown']:.1f}%",
                confidence=0.9,
                impact_score=metrics['max_drawdown'] / 10,
                recommendation="Implement stricter position sizing and stop losses",
                supporting_data={'max_drawdown': metrics['max_drawdown']},
                timestamp=datetime.utcnow()
            ))
        
        return insights

    def _analyze_market_condition_performance(self, trades_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Analyze strategy performance under different market conditions"""
        performance = {}
        
        if 'market_regime_at_entry' not in trades_df.columns:
            return performance
            
        regime_groups = trades_df.groupby('market_regime_at_entry')
        
        for regime, group in regime_groups:
            if regime and len(group) >= 5:  # Need minimum trades
                returns = group['pnl_percentage'].dropna()
                if not returns.empty:
                    performance[regime] = {
                        'avg_return': returns.mean(),
                        'win_rate': (returns > 0).mean(),
                        'trade_count': len(returns),
                        'total_pnl': group['pnl_absolute'].sum() if 'pnl_absolute' in group.columns else 0
                    }
        
        return performance

    async def _analyze_strategy_combinations(self):
        """Find successful strategy combinations"""
        logger.info("🤝 Analyzing strategy combinations...")
        
        if self._trades_df.empty:
            return
        
        # Get unique strategies with sufficient data
        strategy_counts = self._trades_df['strategy_name'].value_counts()
        viable_strategies = strategy_counts[strategy_counts >= self.min_trades_for_analysis].index.tolist()
        
        if len(viable_strategies) < 2:
            logger.warning("Not enough strategies with sufficient data for combination analysis")
            return
        
        # Analyze pairwise combinations
        for i, strategy1 in enumerate(viable_strategies):
            for strategy2 in viable_strategies[i+1:]:
                combination_analysis = await self._analyze_strategy_pair(strategy1, strategy2)
                if combination_analysis and combination_analysis.confidence >= self.confidence_threshold:
                    self.combination_analyses.append(combination_analysis)
        
        # Analyze triple combinations for top performing pairs
        top_pairs = sorted(self.combination_analyses, key=lambda x: x.synergy_score, reverse=True)[:3]
        for pair in top_pairs:
            for additional_strategy in viable_strategies:
                if additional_strategy not in pair.strategies:
                    triple_analysis = await self._analyze_strategy_triple(pair.strategies + [additional_strategy])
                    if triple_analysis and triple_analysis.confidence >= self.confidence_threshold:
                        self.combination_analyses.append(triple_analysis)
        
        logger.info(f"Found {len(self.combination_analyses)} viable strategy combinations")

    async def _analyze_strategy_pair(self, strategy1: str, strategy2: str) -> Optional[CombinationAnalysis]:
        """Analyze a pair of strategies for synergy"""
        try:
            # Get trades for both strategies
            trades1 = self._trades_df[self._trades_df['strategy_name'] == strategy1]
            trades2 = self._trades_df[self._trades_df['strategy_name'] == strategy2]
            
            closed_trades1 = trades1[trades1['trade_status'] == 'closed']
            closed_trades2 = trades2[trades2['trade_status'] == 'closed']
            
            if closed_trades1.empty or closed_trades2.empty:
                return None
            
            # Calculate individual performance
            returns1 = closed_trades1['pnl_percentage'].dropna()
            returns2 = closed_trades2['pnl_percentage'].dropna()
            
            if returns1.empty or returns2.empty:
                return None
            
            # Calculate correlation
            correlation, _ = pearsonr(returns1.iloc[:min(len(returns1), len(returns2))],
                                   returns2.iloc[:min(len(returns1), len(returns2))])
            
            # Calculate optimal weights using mean-variance optimization
            optimal_weights = self._optimize_portfolio_weights([returns1, returns2])
            
            # Calculate combined performance metrics
            w1, w2 = optimal_weights[0], optimal_weights[1]
            combined_return = w1 * returns1.mean() + w2 * returns2.mean()
            combined_vol = np.sqrt(w1**2 * returns1.var() + w2**2 * returns2.var() + 
                                 2 * w1 * w2 * correlation * returns1.std() * returns2.std())
            
            sharpe_ratio = combined_return / combined_vol if combined_vol > 0 else 0
            
            # Calculate synergy score (how much better than weighted average)
            individual_sharpe1 = returns1.mean() / returns1.std() if returns1.std() > 0 else 0
            individual_sharpe2 = returns2.mean() / returns2.std() if returns2.std() > 0 else 0
            weighted_avg_sharpe = w1 * individual_sharpe1 + w2 * individual_sharpe2
            
            synergy_score = (sharpe_ratio - weighted_avg_sharpe) / max(weighted_avg_sharpe, 0.1)
            
            # Determine best market conditions
            market_conditions = self._find_optimal_market_conditions([closed_trades1, closed_trades2])
            
            return CombinationAnalysis(
                strategies=[strategy1, strategy2],
                synergy_score=synergy_score,
                optimal_weights={strategy1: w1, strategy2: w2},
                expected_return=combined_return,
                expected_volatility=combined_vol,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=self._calculate_combined_drawdown(returns1, returns2, w1, w2),
                confidence=min(0.9, len(returns1) + len(returns2)) / 100,
                market_conditions=market_conditions,
                supporting_trades=len(closed_trades1) + len(closed_trades2)
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze strategy pair {strategy1}-{strategy2}: {e}")
            return None

    async def _analyze_strategy_triple(self, strategies: List[str]) -> Optional[CombinationAnalysis]:
        """Analyze three-strategy combination"""
        try:
            if len(strategies) != 3:
                return None
                
            # Get returns for all three strategies
            returns_list = []
            closed_trades_list = []
            
            for strategy in strategies:
                trades = self._trades_df[self._trades_df['strategy_name'] == strategy]
                closed_trades = trades[trades['trade_status'] == 'closed']
                if closed_trades.empty:
                    return None
                returns = closed_trades['pnl_percentage'].dropna()
                if returns.empty:
                    return None
                returns_list.append(returns)
                closed_trades_list.append(closed_trades)
            
            # Calculate optimal weights
            optimal_weights = self._optimize_portfolio_weights(returns_list)
            
            # Calculate combined metrics
            combined_return = sum(w * ret.mean() for w, ret in zip(optimal_weights, returns_list))
            
            # Simplified volatility calculation for three assets
            combined_vol = np.sqrt(sum(w**2 * ret.var() for w, ret in zip(optimal_weights, returns_list)))
            
            sharpe_ratio = combined_return / combined_vol if combined_vol > 0 else 0
            
            # Calculate synergy score
            individual_sharpes = [ret.mean() / ret.std() if ret.std() > 0 else 0 for ret in returns_list]
            weighted_avg_sharpe = sum(w * sharpe for w, sharpe in zip(optimal_weights, individual_sharpes))
            synergy_score = (sharpe_ratio - weighted_avg_sharpe) / max(weighted_avg_sharpe, 0.1)
            
            market_conditions = self._find_optimal_market_conditions(closed_trades_list)
            
            return CombinationAnalysis(
                strategies=strategies,
                synergy_score=synergy_score,
                optimal_weights={strategy: weight for strategy, weight in zip(strategies, optimal_weights)},
                expected_return=combined_return,
                expected_volatility=combined_vol,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=0,  # Simplified for now
                confidence=min(0.8, sum(len(returns) for returns in returns_list)) / 200,
                market_conditions=market_conditions,
                supporting_trades=sum(len(trades) for trades in closed_trades_list)
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze strategy triple {strategies}: {e}")
            return None

    def _optimize_portfolio_weights(self, returns_list: List[pd.Series]) -> List[float]:
        """Optimize portfolio weights using mean-variance optimization"""
        try:
            n_assets = len(returns_list)
            
            # Calculate expected returns and covariance matrix
            expected_returns = np.array([ret.mean() for ret in returns_list])
            
            # Create covariance matrix
            min_length = min(len(ret) for ret in returns_list)
            truncated_returns = [ret.iloc[:min_length] for ret in returns_list]
            returns_matrix = np.column_stack(truncated_returns)
            cov_matrix = np.cov(returns_matrix.T)
            
            # Objective function: minimize negative Sharpe ratio
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                if portfolio_vol == 0:
                    return -float('inf')
                return -portfolio_return / portfolio_vol  # Negative because we minimize
            
            # Constraints
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
            bounds = tuple((0.05, 0.8) for _ in range(n_assets))  # Each weight between 5% and 80%
            
            # Initial guess
            initial_guess = np.array([1.0 / n_assets] * n_assets)
            
            # Optimize
            result = minimize(objective, initial_guess, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                return result.x.tolist()
            else:
                # Fallback to equal weights
                return [1.0 / n_assets] * n_assets
                
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            # Fallback to equal weights
            return [1.0 / len(returns_list)] * len(returns_list)

    def _calculate_combined_drawdown(self, returns1: pd.Series, returns2: pd.Series, w1: float, w2: float) -> float:
        """Calculate maximum drawdown for combined portfolio"""
        try:
            min_length = min(len(returns1), len(returns2))
            combined_returns = w1 * returns1.iloc[:min_length] + w2 * returns2.iloc[:min_length]
            return self._calculate_max_drawdown(combined_returns)
        except Exception:
            return 0.0

    def _find_optimal_market_conditions(self, trades_list: List[pd.DataFrame]) -> List[str]:
        """Find market conditions where strategy combination performs best"""
        condition_performance = {}
        
        for trades_df in trades_list:
            if 'market_regime_at_entry' not in trades_df.columns:
                continue
                
            regime_groups = trades_df.groupby('market_regime_at_entry')
            for regime, group in regime_groups:
                if regime and len(group) >= 3:
                    returns = group['pnl_percentage'].dropna()
                    if not returns.empty:
                        if regime not in condition_performance:
                            condition_performance[regime] = []
                        condition_performance[regime].append(returns.mean())
        
        # Find conditions where average performance is above median
        optimal_conditions = []
        for condition, performances in condition_performance.items():
            avg_performance = np.mean(performances)
            if avg_performance > 0:  # Positive average performance
                optimal_conditions.append(condition)
        
        return optimal_conditions

    async def _identify_loss_patterns(self):
        """Identify patterns in trading losses"""
        logger.info("🔍 Identifying loss patterns...")
        
        if self._trades_df.empty:
            return
        
        # Focus on losing trades
        losing_trades = self._trades_df[
            (self._trades_df['trade_status'] == 'closed') & 
            (self._trades_df['pnl_percentage'] < 0)
        ]
        
        if losing_trades.empty:
            logger.info("No losing trades found - excellent performance!")
            return
        
        # Pattern 1: Consecutive losses by strategy
        consecutive_losses = self._find_consecutive_loss_patterns(losing_trades)
        
        # Pattern 2: Large losses in specific market conditions
        market_condition_losses = self._find_market_condition_loss_patterns(losing_trades)
        
        # Pattern 3: Time-based loss patterns
        time_based_losses = self._find_time_based_loss_patterns(losing_trades)
        
        # Pattern 4: Correlation-based losses (multiple strategies losing simultaneously)
        correlation_losses = self._find_correlation_loss_patterns(losing_trades)
        
        # Combine all patterns
        all_patterns = consecutive_losses + market_condition_losses + time_based_losses + correlation_losses
        
        # Filter by significance
        significant_patterns = [p for p in all_patterns if p.total_impact < -100]  # More than $100 total loss
        
        self.loss_patterns = significant_patterns
        logger.info(f"Identified {len(self.loss_patterns)} significant loss patterns")

    def _find_consecutive_loss_patterns(self, losing_trades: pd.DataFrame) -> List[LossPattern]:
        """Find patterns of consecutive losses"""
        patterns = []
        
        # Group by strategy and analyze sequences
        for strategy, group in losing_trades.groupby('strategy_name'):
            if len(group) < 3:
                continue
                
            # Sort by timestamp
            sorted_trades = group.sort_values('entry_timestamp')
            
            # Find consecutive sequences
            consecutive_count = 1
            max_consecutive = 1
            current_sequence_loss = 0
            total_consecutive_loss = 0
            
            for i in range(1, len(sorted_trades)):
                prev_time = sorted_trades.iloc[i-1]['entry_timestamp']
                curr_time = sorted_trades.iloc[i]['entry_timestamp']
                
                # If trades are within 24 hours, consider them consecutive
                if (curr_time - prev_time).total_seconds() <= 86400:  # 24 hours
                    consecutive_count += 1
                    current_sequence_loss += sorted_trades.iloc[i]['pnl_absolute']
                else:
                    if consecutive_count > max_consecutive:
                        max_consecutive = consecutive_count
                        total_consecutive_loss = current_sequence_loss
                    consecutive_count = 1
                    current_sequence_loss = sorted_trades.iloc[i]['pnl_absolute']
            
            # Check final sequence
            if consecutive_count > max_consecutive:
                max_consecutive = consecutive_count
                total_consecutive_loss = current_sequence_loss
            
            if max_consecutive >= 3:  # 3 or more consecutive losses
                patterns.append(LossPattern(
                    pattern_name=f"Consecutive Losses - {strategy}",
                    description=f"Strategy shows pattern of {max_consecutive} consecutive losses",
                    frequency=max_consecutive,
                    avg_loss=total_consecutive_loss / max_consecutive,
                    total_impact=total_consecutive_loss,
                    triggers=['consecutive_execution', 'strategy_persistence'],
                    affected_strategies=[strategy],
                    market_conditions={},
                    prevention_suggestions=[
                        "Implement cooling-off period after consecutive losses",
                        "Reduce position size after losses",
                        "Add circuit breaker for consecutive failures"
                    ]
                ))
        
        return patterns

    def _find_market_condition_loss_patterns(self, losing_trades: pd.DataFrame) -> List[LossPattern]:
        """Find loss patterns related to market conditions"""
        patterns = []
        
        if 'market_regime_at_entry' not in losing_trades.columns:
            return patterns
        
        # Analyze losses by market regime
        regime_groups = losing_trades.groupby('market_regime_at_entry')
        
        for regime, group in regime_groups:
            if not regime or len(group) < 5:
                continue
                
            total_loss = group['pnl_absolute'].sum()
            avg_loss = group['pnl_absolute'].mean()
            affected_strategies = group['strategy_name'].unique().tolist()
            
            # Check if this regime causes disproportionate losses
            regime_loss_rate = len(group) / len(losing_trades)
            if regime_loss_rate > 0.3:  # More than 30% of losses in this regime
                patterns.append(LossPattern(
                    pattern_name=f"Market Regime Losses - {regime}",
                    description=f"High concentration of losses during {regime} market conditions",
                    frequency=len(group),
                    avg_loss=avg_loss,
                    total_impact=total_loss,
                    triggers=[f'market_regime_{regime}'],
                    affected_strategies=affected_strategies,
                    market_conditions={'regime': regime, 'loss_concentration': regime_loss_rate},
                    prevention_suggestions=[
                        f"Reduce overall allocation during {regime} conditions",
                        f"Avoid or modify strategies during {regime} regime",
                        "Implement regime-specific risk management"
                    ]
                ))
        
        return patterns

    def _find_time_based_loss_patterns(self, losing_trades: pd.DataFrame) -> List[LossPattern]:
        """Find time-based loss patterns"""
        patterns = []
        
        # Add hour and weekday columns
        losing_trades = losing_trades.copy()
        losing_trades['hour'] = pd.to_datetime(losing_trades['entry_timestamp']).dt.hour
        losing_trades['weekday'] = pd.to_datetime(losing_trades['entry_timestamp']).dt.day_name()
        
        # Analyze losses by hour
        hourly_losses = losing_trades.groupby('hour')['pnl_absolute'].agg(['sum', 'count', 'mean'])
        worst_hours = hourly_losses[hourly_losses['count'] >= 3].nsmallest(3, 'sum')
        
        for hour, data in worst_hours.iterrows():
            if data['sum'] < -50:  # More than $50 loss in this hour
                patterns.append(LossPattern(
                    pattern_name=f"Time-based Losses - Hour {hour}",
                    description=f"High losses during hour {hour}:00-{hour+1}:00",
                    frequency=int(data['count']),
                    avg_loss=data['mean'],
                    total_impact=data['sum'],
                    triggers=[f'trading_hour_{hour}'],
                    affected_strategies=losing_trades[losing_trades['hour'] == hour]['strategy_name'].unique().tolist(),
                    market_conditions={'hour': hour},
                    prevention_suggestions=[
                        f"Avoid trading during hour {hour}:00-{hour+1}:00",
                        "Analyze market conditions during this time",
                        "Implement time-based position sizing"
                    ]
                ))
        
        return patterns

    def _find_correlation_loss_patterns(self, losing_trades: pd.DataFrame) -> List[LossPattern]:
        """Find patterns where multiple strategies lose simultaneously"""
        patterns = []
        
        # Group by day and see which strategies lost together
        losing_trades['date'] = pd.to_datetime(losing_trades['entry_timestamp']).dt.date
        daily_losses = losing_trades.groupby(['date', 'strategy_name'])['pnl_absolute'].sum().unstack(fill_value=0)
        
        # Find days with multiple strategy losses
        for date, row in daily_losses.iterrows():
            losing_strategies = row[row < -10].index.tolist()  # Strategies with more than $10 loss
            
            if len(losing_strategies) >= 2:  # Multiple strategies lost
                total_loss = row[losing_strategies].sum()
                
                patterns.append(LossPattern(
                    pattern_name=f"Correlated Losses - {date}",
                    description=f"Multiple strategies lost simultaneously on {date}",
                    frequency=len(losing_strategies),
                    avg_loss=total_loss / len(losing_strategies),
                    total_impact=total_loss,
                    triggers=['market_stress', 'systematic_risk'],
                    affected_strategies=losing_strategies,
                    market_conditions={'date': str(date)},
                    prevention_suggestions=[
                        "Implement correlation monitoring",
                        "Reduce overall exposure during stress",
                        "Diversify strategy types better"
                    ]
                ))
        
        return patterns

    async def _generate_orchestrator_rules(self):
        """Generate new orchestrator rules based on insights"""
        logger.info("🎯 Generating new orchestrator rules...")
        
        rule_id_counter = 1
        
        # Rule 1: Allocation rules based on strategy insights
        for insight in self.strategy_insights:
            if insight.insight_type == 'strength' and insight.confidence > 0.8:
                rule = OrchestratorRule(
                    rule_id=f"RULE_{rule_id_counter:03d}",
                    rule_name=f"Increase {insight.strategy_name} Allocation",
                    rule_type='allocation',
                    condition=f"strategy_performance['{insight.strategy_name}']['sharpe_ratio'] > 1.5",
                    action=f"Increase allocation by {min(0.1, insight.impact_score / 10):.2f}",
                    confidence=insight.confidence,
                    expected_improvement=insight.impact_score,
                    backtest_results={},  # To be filled by backtesting
                    implementation_code=self._generate_allocation_rule_code(insight.strategy_name, 'increase')
                )
                self.new_rules.append(rule)
                rule_id_counter += 1
        
        # Rule 2: Risk management rules based on loss patterns
        for pattern in self.loss_patterns:
            if pattern.total_impact < -100:  # Significant losses
                rule = OrchestratorRule(
                    rule_id=f"RULE_{rule_id_counter:03d}",
                    rule_name=f"Risk Management - {pattern.pattern_name}",
                    rule_type='risk_management',
                    condition=self._generate_pattern_condition(pattern),
                    action=f"Reduce position sizes by 50% for affected strategies",
                    confidence=0.7,
                    expected_improvement=abs(pattern.total_impact) * 0.5,
                    backtest_results={},
                    implementation_code=self._generate_risk_management_code(pattern)
                )
                self.new_rules.append(rule)
                rule_id_counter += 1
        
        # Rule 3: Combination rules based on synergy analysis
        for combination in self.combination_analyses:
            if combination.synergy_score > 0.2 and combination.confidence > 0.7:
                rule = OrchestratorRule(
                    rule_id=f"RULE_{rule_id_counter:03d}",
                    rule_name=f"Optimize Combination - {'+'.join(combination.strategies)}",
                    rule_type='combination',
                    condition=f"market_regime in {combination.market_conditions}",
                    action=f"Set optimal weights: {combination.optimal_weights}",
                    confidence=combination.confidence,
                    expected_improvement=combination.synergy_score * 100,
                    backtest_results={},
                    implementation_code=self._generate_combination_rule_code(combination)
                )
                self.new_rules.append(rule)
                rule_id_counter += 1
        
        logger.info(f"Generated {len(self.new_rules)} new orchestrator rules")

    def _generate_allocation_rule_code(self, strategy_name: str, action: str) -> str:
        """Generate Python code for allocation rule"""
        if action == 'increase':
            return f"""
def adjust_allocation_{strategy_name.lower()}(current_allocation, performance_metrics):
    if performance_metrics.get('sharpe_ratio', 0) > 1.5:
        return min(current_allocation * 1.2, 0.4)  # Max 40% allocation
    return current_allocation
"""
        else:
            return f"""
def adjust_allocation_{strategy_name.lower()}(current_allocation, performance_metrics):
    if performance_metrics.get('sharpe_ratio', 0) < 0.5:
        return current_allocation * 0.8  # Reduce by 20%
    return current_allocation
"""

    def _generate_pattern_condition(self, pattern: LossPattern) -> str:
        """Generate condition string for loss pattern"""
        if 'market_regime' in pattern.triggers[0]:
            regime = pattern.triggers[0].split('_')[-1]
            return f"market_regime == '{regime}'"
        elif 'trading_hour' in pattern.triggers[0]:
            hour = pattern.triggers[0].split('_')[-1]
            return f"current_hour == {hour}"
        else:
            return f"consecutive_losses >= {pattern.frequency}"

    def _generate_risk_management_code(self, pattern: LossPattern) -> str:
        """Generate Python code for risk management rule"""
        return f"""
def apply_risk_management_{pattern.pattern_name.lower().replace(' ', '_')}(strategies, market_state):
    affected_strategies = {pattern.affected_strategies}
    risk_multiplier = 0.5  # Reduce positions by 50%
    
    adjustments = {{}}
    for strategy in affected_strategies:
        if strategy in strategies:
            adjustments[strategy] = risk_multiplier
    
    return adjustments
"""

    def _generate_combination_rule_code(self, combination: CombinationAnalysis) -> str:
        """Generate Python code for combination rule"""
        return f"""
def optimize_combination_{len(combination.strategies)}_strategies(current_weights, market_regime):
    if market_regime in {combination.market_conditions}:
        optimal_weights = {combination.optimal_weights}
        return optimal_weights
    return current_weights
"""

    async def _calculate_optimal_weights(self) -> Dict[str, float]:
        """Calculate optimal portfolio weights across all strategies"""
        logger.info("⚖️ Calculating optimal portfolio weights...")
        
        if self._trades_df.empty:
            return {}
        
        # Get strategy returns
        strategy_returns = {}
        for strategy, group in self._trades_df.groupby('strategy_name'):
            closed_trades = group[group['trade_status'] == 'closed']
            if len(closed_trades) >= self.min_trades_for_analysis:
                returns = closed_trades['pnl_percentage'].dropna()
                if not returns.empty:
                    strategy_returns[strategy] = returns
        
        if len(strategy_returns) < 2:
            return {}
        
        # Use mean-variance optimization
        returns_list = list(strategy_returns.values())
        optimal_weights = self._optimize_portfolio_weights(returns_list)
        
        return {strategy: weight for strategy, weight in zip(strategy_returns.keys(), optimal_weights)}

    async def _save_analysis_results(self) -> Dict[str, Any]:
        """Save all analysis results to files"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Save strategy insights
        insights_file = self.results_dir / f"strategy_insights_{timestamp}.json"
        with open(insights_file, 'w') as f:
            json.dump([asdict(insight) for insight in self.strategy_insights], f, indent=2, default=str)
        
        # Save combination analyses
        combinations_file = self.results_dir / f"combination_analyses_{timestamp}.json"
        with open(combinations_file, 'w') as f:
            json.dump([asdict(combo) for combo in self.combination_analyses], f, indent=2, default=str)
        
        # Save loss patterns
        patterns_file = self.results_dir / f"loss_patterns_{timestamp}.json"
        with open(patterns_file, 'w') as f:
            json.dump([asdict(pattern) for pattern in self.loss_patterns], f, indent=2, default=str)
        
        # Save new rules
        rules_file = self.results_dir / f"orchestrator_rules_{timestamp}.json"
        with open(rules_file, 'w') as f:
            json.dump([asdict(rule) for rule in self.new_rules], f, indent=2, default=str)
        
        return {
            'insights_file': str(insights_file),
            'combinations_file': str(combinations_file),
            'patterns_file': str(patterns_file),
            'rules_file': str(rules_file)
        }

    async def _generate_visualizations(self):
        """Generate analysis visualizations"""
        logger.info("📊 Generating visualizations...")
        
        try:
            # Strategy performance comparison
            await self._create_strategy_performance_chart()
            
            # Combination analysis heatmap
            await self._create_combination_heatmap()
            
            # Loss patterns timeline
            await self._create_loss_patterns_chart()
            
            logger.info("Visualizations saved to analysis/results/")
            
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")

    async def _create_strategy_performance_chart(self):
        """Create strategy performance comparison chart"""
        if not self.strategy_insights:
            return
            
        strategies = []
        sharpe_ratios = []
        impact_scores = []
        colors = []
        
        for insight in self.strategy_insights:
            if insight.insight_type in ['strength', 'weakness']:
                strategies.append(insight.strategy_name)
                # Extract Sharpe ratio from supporting data
                sharpe = insight.supporting_data.get('sharpe_ratio', 0)
                sharpe_ratios.append(sharpe)
                impact_scores.append(insight.impact_score)
                colors.append('green' if insight.insight_type == 'strength' else 'red')
        
        if strategies:
            fig = go.Figure(data=go.Scatter(
                x=sharpe_ratios,
                y=impact_scores,
                mode='markers+text',
                text=strategies,
                textposition="top center",
                marker=dict(size=10, color=colors),
                name='Strategies'
            ))
            
            fig.update_layout(
                title='Strategy Performance Analysis',
                xaxis_title='Sharpe Ratio',
                yaxis_title='Impact Score',
                template='plotly_white'
            )
            
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            fig.write_html(self.results_dir / f"strategy_performance_{timestamp}.html")

    async def _create_combination_heatmap(self):
        """Create strategy combination synergy heatmap"""
        if not self.combination_analyses:
            return
            
        # Create matrix of synergy scores
        strategies = set()
        for combo in self.combination_analyses:
            strategies.update(combo.strategies)
        
        strategies = sorted(list(strategies))
        n = len(strategies)
        synergy_matrix = np.zeros((n, n))
        
        for combo in self.combination_analyses:
            if len(combo.strategies) == 2:
                i = strategies.index(combo.strategies[0])
                j = strategies.index(combo.strategies[1])
                synergy_matrix[i, j] = combo.synergy_score
                synergy_matrix[j, i] = combo.synergy_score
        
        fig = go.Figure(data=go.Heatmap(
            z=synergy_matrix,
            x=strategies,
            y=strategies,
            colorscale='RdYlGn',
            text=synergy_matrix,
            texttemplate="%{text:.2f}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='Strategy Combination Synergy Matrix',
            template='plotly_white'
        )
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fig.write_html(self.results_dir / f"combination_heatmap_{timestamp}.html")

    async def _create_loss_patterns_chart(self):
        """Create loss patterns analysis chart"""
        if not self.loss_patterns:
            return
            
        pattern_names = [p.pattern_name for p in self.loss_patterns]
        total_impacts = [abs(p.total_impact) for p in self.loss_patterns]
        frequencies = [p.frequency for p in self.loss_patterns]
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(x=pattern_names, y=total_impacts, name="Total Impact ($)", marker_color='red'),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=pattern_names, y=frequencies, mode='lines+markers', 
                      name="Frequency", marker_color='blue'),
            secondary_y=True,
        )
        
        fig.update_xaxes(title_text="Loss Patterns")
        fig.update_yaxes(title_text="Total Impact ($)", secondary_y=False)
        fig.update_yaxes(title_text="Frequency", secondary_y=True)
        
        fig.update_layout(title_text="Loss Patterns Analysis", template='plotly_white')
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fig.write_html(self.results_dir / f"loss_patterns_{timestamp}.html")

    def _get_key_findings(self) -> List[str]:
        """Get key findings from analysis"""
        findings = []
        
        # Top performing strategies
        strength_insights = [i for i in self.strategy_insights if i.insight_type == 'strength']
        if strength_insights:
            top_strategy = max(strength_insights, key=lambda x: x.impact_score)
            findings.append(f"Top performing strategy: {top_strategy.strategy_name} with {top_strategy.confidence:.1%} confidence")
        
        # Best combination
        if self.combination_analyses:
            best_combo = max(self.combination_analyses, key=lambda x: x.synergy_score)
            findings.append(f"Best strategy combination: {'+'.join(best_combo.strategies)} with {best_combo.synergy_score:.2f} synergy score")
        
        # Most significant loss pattern
        if self.loss_patterns:
            worst_pattern = min(self.loss_patterns, key=lambda x: x.total_impact)
            findings.append(f"Most impactful loss pattern: {worst_pattern.pattern_name} with ${abs(worst_pattern.total_impact):.0f} total impact")
        
        # Number of actionable rules
        findings.append(f"Generated {len(self.new_rules)} actionable orchestrator rules")
        
        return findings

    def _get_top_recommendations(self) -> List[str]:
        """Get top recommendations from analysis"""
        recommendations = []
        
        # From strategy insights
        high_impact_insights = [i for i in self.strategy_insights if abs(i.impact_score) > 0.5]
        for insight in high_impact_insights[:3]:  # Top 3
            recommendations.append(insight.recommendation)
        
        # From loss patterns
        for pattern in self.loss_patterns[:2]:  # Top 2 patterns
            recommendations.extend(pattern.prevention_suggestions[:1])  # First suggestion
        
        # From combinations
        top_combos = sorted(self.combination_analyses, key=lambda x: x.synergy_score, reverse=True)[:2]
        for combo in top_combos:
            recommendations.append(f"Implement optimal weights for {'+'.join(combo.strategies)}: {combo.optimal_weights}")
        
        return recommendations[:5]  # Return top 5 recommendations

# Example usage
async def example_usage():
    """Example of how to use LearningPipeline"""
    # This would typically be initialized with your actual database pool
    # db_pool = await asyncpg.create_pool(...)
    
    # pipeline = LearningPipeline(db_pool, lookback_days=30)
    # results = await pipeline.run_full_analysis()
    # print(json.dumps(results, indent=2, default=str))
    
    print("Example usage completed (commented out - requires actual DB pool)")

if __name__ == "__main__":
    asyncio.run(example_usage())