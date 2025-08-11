#!/usr/bin/env python3
# optimization/strategy_improvements.py
"""
Strategy-Specific Improvements and Optimization
Provides concrete improvements for each trading strategy
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

@dataclass
class ParameterRange:
    """Parameter optimization range"""
    name: str
    min_value: float
    max_value: float
    optimal_value: float
    description: str
    market_conditions: List[str]  # Market conditions where this value works best

@dataclass
class MarketCondition:
    """Market condition definition"""
    name: str
    volatility_range: Tuple[float, float]
    trend_strength: Tuple[float, float]
    description: str
    
@dataclass
class StrategyImprovement:
    """Comprehensive strategy improvement recommendation"""
    strategy_name: str
    optimal_parameters: Dict[str, ParameterRange]
    best_market_conditions: List[MarketCondition]
    avoid_conditions: List[str]
    risk_management_improvements: List[str]
    entry_exit_optimizations: List[str]
    position_sizing_recommendations: List[str]
    expected_performance_improvement: float
    implementation_priority: str
    confidence_level: float

class MarketRegime(Enum):
    BULL_STRONG = "bull_strong"
    BULL_WEAK = "bull_weak"
    BEAR_STRONG = "bear_strong"
    BEAR_WEAK = "bear_weak"
    SIDEWAYS_LOW_VOL = "sideways_low_vol"
    SIDEWAYS_HIGH_VOL = "sideways_high_vol"
    EXTREME_VOLATILITY = "extreme_volatility"
    RECOVERY = "recovery"

class StrategyOptimizer:
    """
    Provides strategy-specific optimization recommendations
    """
    
    def __init__(self):
        self.strategy_improvements = {}
        self._initialize_improvements()
    
    def _initialize_improvements(self):
        """Initialize improvement recommendations for all strategies"""
        
        # 1. Momentum Strategy Improvements
        self.strategy_improvements['momentum'] = StrategyImprovement(
            strategy_name="Momentum Strategy",
            optimal_parameters={
                'lookback_period': ParameterRange(
                    name='lookback_period',
                    min_value=10,
                    max_value=30,
                    optimal_value=14,
                    description='Optimal lookback period for momentum calculation',
                    market_conditions=['bull_strong', 'bull_weak', 'recovery']
                ),
                'momentum_threshold': ParameterRange(
                    name='momentum_threshold',
                    min_value=0.01,
                    max_value=0.04,
                    optimal_value=0.025,
                    description='Minimum momentum threshold for signal generation',
                    market_conditions=['bull_strong', 'sideways_high_vol']
                ),
                'stop_loss': ParameterRange(
                    name='stop_loss',
                    min_value=0.02,
                    max_value=0.08,
                    optimal_value=0.035,
                    description='Stop loss percentage',
                    market_conditions=['all']
                ),
                'take_profit': ParameterRange(
                    name='take_profit',
                    min_value=0.04,
                    max_value=0.15,
                    optimal_value=0.08,
                    description='Take profit percentage',
                    market_conditions=['bull_strong', 'recovery']
                )
            },
            best_market_conditions=[
                MarketCondition("Strong Bull Market", (0.02, 0.06), (0.7, 1.0), "High momentum, clear trends"),
                MarketCondition("Recovery Phase", (0.03, 0.08), (0.5, 0.8), "Emerging from bear market"),
                MarketCondition("Breakout Scenarios", (0.04, 0.12), (0.6, 1.0), "Strong directional moves")
            ],
            avoid_conditions=[
                "Sideways markets with low volatility (< 2%)",
                "Bear markets with high volatility (> 8%)",
                "News-driven choppy markets",
                "Low liquidity periods (weekends, holidays)"
            ],
            risk_management_improvements=[
                "Implement dynamic stop-loss based on ATR (Average True Range)",
                "Use trailing stops in strong trends (adjust by 0.5 * ATR)",
                "Reduce position size by 50% in high volatility environments (VIX > 30)",
                "Add momentum confirmation filter (RSI divergence check)",
                "Implement maximum daily loss limit of 2% of capital"
            ],
            entry_exit_optimizations=[
                "Add volume confirmation (volume > 1.5x average)",
                "Use multiple timeframe confirmation (1h + 4h signals)",
                "Implement smart entry: enter on pullbacks, not breakouts",
                "Add market structure filter (higher highs/higher lows)",
                "Exit 50% position at first target, trail remainder"
            ],
            position_sizing_recommendations=[
                "Use Kelly Criterion with 0.25 multiplier for conservative sizing",
                "Scale position size with trend strength (0.5x to 2x base size)",
                "Reduce size by 30% during high correlation periods",
                "Maximum 5% of portfolio per single position",
                "Increase allocation during confirmed bull markets (+25%)"
            ],
            expected_performance_improvement=35.0,
            implementation_priority="High",
            confidence_level=0.85
        )
        
        # 2. Mean Reversion Strategy Improvements
        self.strategy_improvements['mean_reversion'] = StrategyImprovement(
            strategy_name="Mean Reversion Strategy",
            optimal_parameters={
                'lookback_period': ParameterRange(
                    name='lookback_period',
                    min_value=20,
                    max_value=100,
                    optimal_value=50,
                    description='Period for mean calculation',
                    market_conditions=['sideways_low_vol', 'sideways_high_vol']
                ),
                'std_multiplier': ParameterRange(
                    name='std_multiplier',
                    min_value=1.5,
                    max_value=3.0,
                    optimal_value=2.2,
                    description='Standard deviation multiplier for entry',
                    market_conditions=['sideways_low_vol']
                ),
                'rsi_threshold': ParameterRange(
                    name='rsi_threshold',
                    min_value=20,
                    max_value=35,
                    optimal_value=25,
                    description='RSI oversold/overbought threshold',
                    market_conditions=['sideways_high_vol']
                )
            },
            best_market_conditions=[
                MarketCondition("Sideways Low Volatility", (0.01, 0.03), (-0.2, 0.2), "Range-bound markets"),
                MarketCondition("High Frequency Oscillations", (0.02, 0.05), (-0.3, 0.3), "Quick reversals"),
                MarketCondition("Post-News Normalization", (0.03, 0.08), (-0.4, 0.4), "Return to fair value")
            ],
            avoid_conditions=[
                "Strong trending markets (momentum > 0.05 daily)",
                "Low liquidity environments",
                "Major news events and announcements",
                "Market regime changes",
                "Extreme fear/greed periods"
            ],
            risk_management_improvements=[
                "Add trend filter: no counter-trend trades in strong trends",
                "Implement maximum holding period (48 hours for crypto)",
                "Use tighter stops during trending phases (1.5x normal)",
                "Add correlation filter: avoid trades during high correlation",
                "Implement regime detection: pause during regime changes"
            ],
            entry_exit_optimizations=[
                "Add double confirmation: price + RSI both extreme",
                "Use limit orders at mean ± 2.2 standard deviations",
                "Implement scaled entries (25%, 50%, 25% allocation)",
                "Add time-based exits: close at session end if not profitable",
                "Use dynamic targets based on recent volatility"
            ],
            position_sizing_recommendations=[
                "Smaller positions during trending markets (0.5x base)",
                "Larger positions in established ranges (1.5x base)",
                "Scale with volatility: reduce size when ATR increases",
                "Maximum 3% risk per trade",
                "Diversify across multiple timeframes (1h, 4h, daily)"
            ],
            expected_performance_improvement=28.0,
            implementation_priority="High",
            confidence_level=0.78
        )
        
        # 3. ML Strategy Improvements
        self.strategy_improvements['ml'] = StrategyImprovement(
            strategy_name="ML Strategy",
            optimal_parameters={
                'model_retrain_frequency': ParameterRange(
                    name='model_retrain_frequency',
                    min_value=24,
                    max_value=168,
                    optimal_value=72,
                    description='Hours between model retraining',
                    market_conditions=['all']
                ),
                'feature_window': ParameterRange(
                    name='feature_window',
                    min_value=50,
                    max_value=200,
                    optimal_value=100,
                    description='Number of periods for feature calculation',
                    market_conditions=['all']
                ),
                'confidence_threshold': ParameterRange(
                    name='confidence_threshold',
                    min_value=0.6,
                    max_value=0.85,
                    optimal_value=0.72,
                    description='Minimum prediction confidence for trading',
                    market_conditions=['all']
                )
            },
            best_market_conditions=[
                MarketCondition("Stable Patterns", (0.02, 0.05), (-0.1, 0.1), "Predictable market behavior"),
                MarketCondition("Moderate Volatility", (0.03, 0.06), (-0.3, 0.3), "Sufficient signal, not too noisy"),
                MarketCondition("Data-Rich Periods", (0.02, 0.08), (-0.5, 0.5), "High-frequency trading data available")
            ],
            avoid_conditions=[
                "Model training periods (first 48 hours after retrain)",
                "Extreme market events (black swan events)",
                "Low data quality periods",
                "Major system updates or maintenance",
                "Periods with high slippage (> 0.1%)"
            ],
            risk_management_improvements=[
                "Implement model uncertainty quantification",
                "Add ensemble methods (combine 3-5 models)",
                "Use prediction confidence for position sizing",
                "Implement drift detection and automatic model updates",
                "Add feature importance monitoring",
                "Include regime-aware model selection"
            ],
            entry_exit_optimizations=[
                "Use prediction probability for entry timing",
                "Implement stacking with traditional indicators",
                "Add market microstructure features (order book, volume)",
                "Use gradient boosting for feature interaction capture",
                "Implement online learning for rapid adaptation"
            ],
            position_sizing_recommendations=[
                "Scale position with prediction confidence",
                "Use Kelly Criterion with ML win rate estimates",
                "Implement dynamic allocation based on model performance",
                "Maximum 4% per position, 15% total ML allocation",
                "Reduce allocation during model uncertainty periods"
            ],
            expected_performance_improvement=45.0,
            implementation_priority="Very High",
            confidence_level=0.82
        )
        
        # 4. Grid Trading Strategy Improvements
        self.strategy_improvements['grid'] = StrategyImprovement(
            strategy_name="Grid Trading Strategy",
            optimal_parameters={
                'grid_spacing': ParameterRange(
                    name='grid_spacing',
                    min_value=0.005,
                    max_value=0.03,
                    optimal_value=0.015,
                    description='Percentage spacing between grid levels',
                    market_conditions=['sideways_low_vol', 'sideways_high_vol']
                ),
                'grid_levels': ParameterRange(
                    name='grid_levels',
                    min_value=5,
                    max_value=20,
                    optimal_value=10,
                    description='Number of grid levels above and below current price',
                    market_conditions=['sideways_low_vol']
                ),
                'take_profit_ratio': ParameterRange(
                    name='take_profit_ratio',
                    min_value=0.8,
                    max_value=1.2,
                    optimal_value=1.0,
                    description='Ratio of grid spacing for take profit',
                    market_conditions=['all']
                )
            },
            best_market_conditions=[
                MarketCondition("Tight Ranges", (0.01, 0.02), (-0.05, 0.05), "Predictable oscillations"),
                MarketCondition("Sideways Consolidation", (0.015, 0.04), (-0.1, 0.1), "Range-bound movement"),
                MarketCondition("Post-Breakout Consolidation", (0.02, 0.05), (-0.08, 0.08), "Cooling off periods")
            ],
            avoid_conditions=[
                "Strong trending markets (> 0.05 daily move)",
                "Breakout scenarios",
                "Low liquidity periods",
                "Major news events",
                "Extreme volatility spikes (> 2x normal)"
            ],
            risk_management_improvements=[
                "Implement dynamic grid adjustment based on volatility",
                "Add trend detection to pause grid in strong trends",
                "Use maximum grid age (remove stale orders after 24h)",
                "Implement partial grid shutdown during extreme moves",
                "Add maximum drawdown protection (stop grid at -5%)"
            ],
            entry_exit_optimizations=[
                "Use volume-weighted average price for grid centering",
                "Implement smart grid placement around support/resistance",
                "Add time-based order refresh",
                "Use different spacing for buy vs sell grids",
                "Implement partial profit taking at grid boundaries"
            ],
            position_sizing_recommendations=[
                "Equal position sizes per grid level",
                "Reduce allocation in trending markets (-50%)",
                "Maximum 8% of portfolio in active grids",
                "Scale grid size with account equity",
                "Maintain cash reserve for grid expansion (20%)"
            ],
            expected_performance_improvement=22.0,
            implementation_priority="Medium",
            confidence_level=0.75
        )
        
        # 5. Arbitrage Strategy Improvements
        self.strategy_improvements['arbitrage'] = StrategyImprovement(
            strategy_name="Arbitrage Strategy",
            optimal_parameters={
                'min_spread': ParameterRange(
                    name='min_spread',
                    min_value=0.001,
                    max_value=0.01,
                    optimal_value=0.003,
                    description='Minimum profitable spread percentage',
                    market_conditions=['all']
                ),
                'execution_timeout': ParameterRange(
                    name='execution_timeout',
                    min_value=1.0,
                    max_value=10.0,
                    optimal_value=3.0,
                    description='Maximum execution time in seconds',
                    market_conditions=['all']
                ),
                'max_position_size': ParameterRange(
                    name='max_position_size',
                    min_value=100,
                    max_value=10000,
                    optimal_value=2000,
                    description='Maximum position size in USD',
                    market_conditions=['high_liquidity']
                )
            },
            best_market_conditions=[
                MarketCondition("High Liquidity", (0.01, 0.03), (-0.1, 0.1), "Deep order books"),
                MarketCondition("Exchange Inefficiencies", (0.02, 0.05), (-0.2, 0.2), "Price disparities"),
                MarketCondition("Network Congestion", (0.02, 0.08), (-0.3, 0.3), "Delayed arbitrage closure")
            ],
            avoid_conditions=[
                "Network maintenance windows",
                "Extreme market volatility (> 5% in 5 minutes)",
                "Low liquidity periods (< $50k order book depth)",
                "Exchange technical issues",
                "Major news events causing rapid price movements"
            ],
            risk_management_improvements=[
                "Implement real-time latency monitoring",
                "Add exchange health checks before execution",
                "Use maximum exposure limits per exchange pair",
                "Implement automatic position unwinding on anomalies",
                "Add network congestion detection and adjustment"
            ],
            entry_exit_optimizations=[
                "Use smart order routing for optimal execution",
                "Implement pre-trade risk checks",
                "Add liquidity assessment before order placement",
                "Use market orders for speed in high-opportunity scenarios",
                "Implement partial fill handling"
            ],
            position_sizing_recommendations=[
                "Scale with available liquidity",
                "Maximum 2% of portfolio per arbitrage opportunity",
                "Reduce size during high volatility periods",
                "Increase allocation when spreads are wide (> 0.5%)",
                "Maintain minimum cash reserves for opportunities (30%)"
            ],
            expected_performance_improvement=55.0,
            implementation_priority="Very High",
            confidence_level=0.88
        )
        
        # 6. DeFi Strategy Improvements
        self.strategy_improvements['defi'] = StrategyImprovement(
            strategy_name="DeFi Strategy",
            optimal_parameters={
                'yield_threshold': ParameterRange(
                    name='yield_threshold',
                    min_value=0.05,
                    max_value=0.30,
                    optimal_value=0.12,
                    description='Minimum APY for yield farming',
                    market_conditions=['all']
                ),
                'impermanent_loss_limit': ParameterRange(
                    name='impermanent_loss_limit',
                    min_value=0.02,
                    max_value=0.15,
                    optimal_value=0.05,
                    description='Maximum acceptable impermanent loss',
                    market_conditions=['sideways_low_vol']
                ),
                'gas_cost_threshold': ParameterRange(
                    name='gas_cost_threshold',
                    min_value=10,
                    max_value=100,
                    optimal_value=30,
                    description='Maximum gas cost in USD for transactions',
                    market_conditions=['all']
                )
            },
            best_market_conditions=[
                MarketCondition("Stable Correlations", (0.01, 0.03), (-0.05, 0.05), "Low impermanent loss risk"),
                MarketCondition("High Yield Periods", (0.02, 0.06), (-0.1, 0.1), "Attractive yield opportunities"),
                MarketCondition("Low Gas Costs", (0.01, 0.05), (-0.2, 0.2), "Cost-effective transactions")
            ],
            avoid_conditions=[
                "High gas cost periods (> $50 transaction cost)",
                "Extreme yield volatility",
                "Protocol security incidents",
                "Regulatory uncertainty periods",
                "High impermanent loss scenarios (> 10%)"
            ],
            risk_management_improvements=[
                "Implement protocol risk assessment",
                "Add impermanent loss monitoring and alerts",
                "Use diversification across multiple protocols",
                "Implement automated position rebalancing",
                "Add smart contract audit score requirements"
            ],
            entry_exit_optimizations=[
                "Use gas price optimization for transactions",
                "Implement yield opportunity scanning",
                "Add automated compounding strategies",
                "Use layer 2 solutions when available",
                "Implement batched transactions for efficiency"
            ],
            position_sizing_recommendations=[
                "Maximum 10% of portfolio in DeFi strategies",
                "Diversify across 3-5 protocols",
                "Scale with protocol TVL and audit scores",
                "Reduce allocation during regulatory uncertainty",
                "Maintain stablecoin reserves for opportunities (40%)"
            ],
            expected_performance_improvement=40.0,
            implementation_priority="Medium",
            confidence_level=0.70
        )
        
        # 7. Copy Trading Strategy Improvements
        self.strategy_improvements['copy_trading'] = StrategyImprovement(
            strategy_name="Copy Trading Strategy",
            optimal_parameters={
                'trader_score_threshold': ParameterRange(
                    name='trader_score_threshold',
                    min_value=0.7,
                    max_value=0.95,
                    optimal_value=0.82,
                    description='Minimum trader performance score',
                    market_conditions=['all']
                ),
                'max_drawdown_threshold': ParameterRange(
                    name='max_drawdown_threshold',
                    min_value=0.05,
                    max_value=0.25,
                    optimal_value=0.12,
                    description='Maximum acceptable trader drawdown',
                    market_conditions=['all']
                ),
                'copy_delay_seconds': ParameterRange(
                    name='copy_delay_seconds',
                    min_value=1,
                    max_value=30,
                    optimal_value=5,
                    description='Delay before copying trades',
                    market_conditions=['all']
                )
            },
            best_market_conditions=[
                MarketCondition("Trending Markets", (0.02, 0.05), (0.3, 0.8), "Good for trend followers"),
                MarketCondition("Moderate Volatility", (0.03, 0.06), (-0.2, 0.5), "Allows strategy execution"),
                MarketCondition("High Trader Activity", (0.02, 0.08), (-0.3, 0.6), "More strategies to choose from")
            ],
            avoid_conditions=[
                "Low trader activity periods",
                "Extreme market volatility (> 8%)",
                "Platform technical issues",
                "Low liquidity periods",
                "Trader performance degradation"
            ],
            risk_management_improvements=[
                "Implement real-time trader performance monitoring",
                "Add automatic stop-copying triggers",
                "Use diversification across multiple traders",
                "Implement maximum allocation per trader",
                "Add correlation analysis between copied traders"
            ],
            entry_exit_optimizations=[
                "Use trader selection algorithms",
                "Implement performance-based allocation",
                "Add trade filtering based on market conditions",
                "Use partial copying for risk management",
                "Implement delayed copying for confirmation"
            ],
            position_sizing_recommendations=[
                "Maximum 5% allocation per copied trader",
                "Total copy trading allocation: 25% of portfolio",
                "Scale allocation with trader performance",
                "Reduce allocation during trader drawdowns",
                "Implement performance-based rebalancing monthly"
            ],
            expected_performance_improvement=25.0,
            implementation_priority="Low",
            confidence_level=0.65
        )
        
        # 8. Stablecoin Parking Strategy Improvements
        self.strategy_improvements['stablecoin_parking'] = StrategyImprovement(
            strategy_name="Stablecoin Parking Strategy",
            optimal_parameters={
                'yield_rate_threshold': ParameterRange(
                    name='yield_rate_threshold',
                    min_value=0.02,
                    max_value=0.15,
                    optimal_value=0.05,
                    description='Minimum yield rate for parking',
                    market_conditions=['all']
                ),
                'platform_risk_score': ParameterRange(
                    name='platform_risk_score',
                    min_value=0.7,
                    max_value=0.95,
                    optimal_value=0.85,
                    description='Minimum platform safety score',
                    market_conditions=['all']
                ),
                'max_allocation_percent': ParameterRange(
                    name='max_allocation_percent',
                    min_value=0.1,
                    max_value=0.5,
                    optimal_value=0.3,
                    description='Maximum percentage of portfolio to park',
                    market_conditions=['bear_markets']
                )
            },
            best_market_conditions=[
                MarketCondition("Bear Markets", (0.02, 0.08), (-0.5, -0.1), "Capital preservation mode"),
                MarketCondition("High Uncertainty", (0.03, 0.10), (-0.3, 0.1), "Risk-off periods"),
                MarketCondition("Waiting for Opportunities", (0.01, 0.05), (-0.1, 0.1), "Between active strategies")
            ],
            avoid_conditions=[
                "Bull market momentum phases",
                "High-opportunity periods",
                "Platform security concerns",
                "Regulatory crackdowns on stablecoins",
                "Low yield environments (< 2% APY)"
            ],
            risk_management_improvements=[
                "Diversify across multiple stablecoin platforms",
                "Monitor platform health and TVL changes",
                "Implement automatic rebalancing",
                "Add regulatory risk monitoring",
                "Use insurance-backed platforms when available"
            ],
            entry_exit_optimizations=[
                "Use yield opportunity scanning",
                "Implement automatic platform switching",
                "Add timing optimization for maximum yields",
                "Use compound interest strategies",
                "Implement tax-efficient withdrawal timing"
            ],
            position_sizing_recommendations=[
                "Maximum 30% of portfolio in stablecoin parking",
                "Diversify across 2-3 platforms",
                "Increase allocation during bear markets (up to 50%)",
                "Reduce allocation during bull markets (down to 10%)",
                "Maintain emergency cash reserves separately (10%)"
            ],
            expected_performance_improvement=15.0,
            implementation_priority="Low",
            confidence_level=0.90
        )
        
        # 9. Lazy Billionaire Strategy Improvements
        self.strategy_improvements['lazy_billionaire'] = StrategyImprovement(
            strategy_name="Lazy Billionaire Strategy",
            optimal_parameters={
                'rebalance_frequency_days': ParameterRange(
                    name='rebalance_frequency_days',
                    min_value=7,
                    max_value=90,
                    optimal_value=30,
                    description='Days between portfolio rebalancing',
                    market_conditions=['all']
                ),
                'allocation_deviation_threshold': ParameterRange(
                    name='allocation_deviation_threshold',
                    min_value=0.05,
                    max_value=0.25,
                    optimal_value=0.15,
                    description='Maximum deviation before rebalancing',
                    market_conditions=['all']
                ),
                'btc_allocation': ParameterRange(
                    name='btc_allocation',
                    min_value=0.3,
                    max_value=0.7,
                    optimal_value=0.5,
                    description='Bitcoin allocation percentage',
                    market_conditions=['all']
                )
            },
            best_market_conditions=[
                MarketCondition("Long-term Bull Markets", (0.02, 0.05), (0.1, 0.8), "Steady appreciation"),
                MarketCondition("Market Maturation", (0.01, 0.04), (0.0, 0.3), "Reduced volatility"),
                MarketCondition("Adoption Phases", (0.03, 0.08), (0.2, 0.6), "Gradual mainstream acceptance")
            ],
            avoid_conditions=[
                "Short-term trading periods",
                "Extreme bear markets (temporary reduction)",
                "Regulatory crisis periods",
                "Major technological disruptions",
                "Liquidity crisis events"
            ],
            risk_management_improvements=[
                "Implement dynamic allocation based on market cycles",
                "Add defensive assets during extreme downturns",
                "Use gradual rebalancing instead of immediate",
                "Implement maximum allocation per asset (40%)",
                "Add correlation monitoring for diversification"
            ],
            entry_exit_optimizations=[
                "Use dollar-cost averaging for rebalancing",
                "Implement value-based rebalancing triggers",
                "Add momentum filters for rebalancing timing",
                "Use tax-efficient rebalancing strategies",
                "Implement automated compound interest"
            ],
            position_sizing_recommendations=[
                "BTC: 50%, ETH: 30%, ALTs: 15%, STABLE: 5%",
                "Adjust allocations based on market cycle",
                "Increase stablecoin allocation during bear markets",
                "Implement gradual allocation changes (5% per month max)",
                "Use market cap weighting with deviation limits"
            ],
            expected_performance_improvement=20.0,
            implementation_priority="Medium",
            confidence_level=0.88
        )
    
    def get_strategy_improvement(self, strategy_name: str) -> Optional[StrategyImprovement]:
        """Get improvement recommendations for a specific strategy"""
        return self.strategy_improvements.get(strategy_name.lower())
    
    def get_all_improvements(self) -> Dict[str, StrategyImprovement]:
        """Get all strategy improvements"""
        return self.strategy_improvements
    
    def get_market_regime_recommendations(self, current_regime: MarketRegime) -> Dict[str, float]:
        """
        Get strategy allocation recommendations based on current market regime
        """
        allocations = {}
        
        if current_regime == MarketRegime.BULL_STRONG:
            allocations = {
                'momentum': 0.25,
                'ml': 0.20,
                'arbitrage': 0.15,
                'lazy_billionaire': 0.20,
                'defi': 0.10,
                'grid': 0.05,
                'mean_reversion': 0.03,
                'copy_trading': 0.02,
                'stablecoin_parking': 0.00
            }
        elif current_regime == MarketRegime.BULL_WEAK:
            allocations = {
                'momentum': 0.20,
                'ml': 0.18,
                'lazy_billionaire': 0.25,
                'arbitrage': 0.12,
                'grid': 0.08,
                'defi': 0.08,
                'mean_reversion': 0.05,
                'copy_trading': 0.02,
                'stablecoin_parking': 0.02
            }
        elif current_regime == MarketRegime.BEAR_STRONG:
            allocations = {
                'stablecoin_parking': 0.30,
                'mean_reversion': 0.15,
                'arbitrage': 0.15,
                'ml': 0.12,
                'grid': 0.10,
                'defi': 0.08,
                'lazy_billionaire': 0.05,
                'momentum': 0.03,
                'copy_trading': 0.02
            }
        elif current_regime == MarketRegime.BEAR_WEAK:
            allocations = {
                'stablecoin_parking': 0.25,
                'mean_reversion': 0.18,
                'arbitrage': 0.15,
                'ml': 0.15,
                'grid': 0.12,
                'lazy_billionaire': 0.08,
                'defi': 0.05,
                'momentum': 0.02,
                'copy_trading': 0.00
            }
        elif current_regime == MarketRegime.SIDEWAYS_LOW_VOL:
            allocations = {
                'grid': 0.25,
                'mean_reversion': 0.20,
                'stablecoin_parking': 0.15,
                'arbitrage': 0.12,
                'ml': 0.10,
                'lazy_billionaire': 0.10,
                'defi': 0.05,
                'momentum': 0.02,
                'copy_trading': 0.01
            }
        elif current_regime == MarketRegime.SIDEWAYS_HIGH_VOL:
            allocations = {
                'mean_reversion': 0.22,
                'grid': 0.18,
                'ml': 0.15,
                'arbitrage': 0.15,
                'stablecoin_parking': 0.12,
                'momentum': 0.08,
                'lazy_billionaire': 0.05,
                'defi': 0.03,
                'copy_trading': 0.02
            }
        elif current_regime == MarketRegime.EXTREME_VOLATILITY:
            allocations = {
                'stablecoin_parking': 0.40,
                'arbitrage': 0.20,
                'ml': 0.15,
                'mean_reversion': 0.10,
                'grid': 0.05,
                'lazy_billionaire': 0.05,
                'defi': 0.03,
                'momentum': 0.02,
                'copy_trading': 0.00
            }
        elif current_regime == MarketRegime.RECOVERY:
            allocations = {
                'momentum': 0.20,
                'ml': 0.18,
                'lazy_billionaire': 0.15,
                'arbitrage': 0.12,
                'mean_reversion': 0.10,
                'grid': 0.10,
                'defi': 0.08,
                'stablecoin_parking': 0.05,
                'copy_trading': 0.02
            }
        
        return allocations
    
    def generate_implementation_roadmap(self, priority_strategies: List[str] = None) -> Dict[str, Any]:
        """
        Generate implementation roadmap for strategy improvements
        """
        if priority_strategies is None:
            priority_strategies = ['momentum', 'ml', 'arbitrage', 'mean_reversion']
        
        roadmap = {
            'phase_1_immediate': {
                'duration': '1-2 weeks',
                'strategies': [],
                'focus': 'High-impact, low-complexity improvements'
            },
            'phase_2_short_term': {
                'duration': '3-4 weeks',
                'strategies': [],
                'focus': 'Medium complexity improvements with good ROI'
            },
            'phase_3_medium_term': {
                'duration': '5-8 weeks',
                'strategies': [],
                'focus': 'Complex improvements requiring significant development'
            },
            'phase_4_long_term': {
                'duration': '9-12 weeks',
                'strategies': [],
                'focus': 'Advanced features and optimization'
            }
        }
        
        # Categorize improvements by implementation priority and complexity
        for strategy_name in priority_strategies:
            improvement = self.get_strategy_improvement(strategy_name)
            if not improvement:
                continue
            
            complexity_score = self._calculate_implementation_complexity(improvement)
            impact_score = improvement.expected_performance_improvement
            
            if improvement.implementation_priority == "Very High" and complexity_score < 3:
                roadmap['phase_1_immediate']['strategies'].append({
                    'name': strategy_name,
                    'improvement': improvement,
                    'complexity': complexity_score,
                    'impact': impact_score
                })
            elif improvement.implementation_priority in ["Very High", "High"] and complexity_score < 5:
                roadmap['phase_2_short_term']['strategies'].append({
                    'name': strategy_name,
                    'improvement': improvement,
                    'complexity': complexity_score,
                    'impact': impact_score
                })
            elif complexity_score < 7:
                roadmap['phase_3_medium_term']['strategies'].append({
                    'name': strategy_name,
                    'improvement': improvement,
                    'complexity': complexity_score,
                    'impact': impact_score
                })
            else:
                roadmap['phase_4_long_term']['strategies'].append({
                    'name': strategy_name,
                    'improvement': improvement,
                    'complexity': complexity_score,
                    'impact': impact_score
                })
        
        return roadmap
    
    def _calculate_implementation_complexity(self, improvement: StrategyImprovement) -> int:
        """Calculate implementation complexity score (1-10)"""
        complexity = 1
        
        # Add complexity based on number of parameters
        complexity += len(improvement.optimal_parameters) * 0.5
        
        # Add complexity based on risk management improvements
        complexity += len(improvement.risk_management_improvements) * 0.3
        
        # Add complexity based on entry/exit optimizations
        complexity += len(improvement.entry_exit_optimizations) * 0.4
        
        # Strategy-specific complexity adjustments
        if improvement.strategy_name == "ML Strategy":
            complexity += 3  # ML implementations are complex
        elif improvement.strategy_name == "Arbitrage Strategy":
            complexity += 2  # Multi-exchange complexity
        elif improvement.strategy_name == "DeFi Strategy":
            complexity += 2  # Smart contract interactions
        
        return min(int(complexity), 10)

def main():
    """Main function for testing and demonstration"""
    optimizer = StrategyOptimizer()
    
    print("=== STRATEGY OPTIMIZATION RECOMMENDATIONS ===\n")
    
    # Display all strategy improvements
    for strategy_name, improvement in optimizer.get_all_improvements().items():
        print(f"Strategy: {improvement.strategy_name}")
        print(f"Expected Improvement: +{improvement.expected_performance_improvement:.1f}%")
        print(f"Implementation Priority: {improvement.implementation_priority}")
        print(f"Confidence Level: {improvement.confidence_level:.1%}")
        print("-" * 50)
    
    # Generate implementation roadmap
    print("\n=== IMPLEMENTATION ROADMAP ===\n")
    roadmap = optimizer.generate_implementation_roadmap()
    
    for phase, details in roadmap.items():
        print(f"{phase.upper()}: {details['duration']}")
        print(f"Focus: {details['focus']}")
        for strategy in details['strategies']:
            print(f"  - {strategy['name']}: +{strategy['impact']:.1f}% impact, {strategy['complexity']}/10 complexity")
        print()
    
    # Market regime recommendations
    print("=== MARKET REGIME ALLOCATIONS ===\n")
    for regime in MarketRegime:
        allocations = optimizer.get_market_regime_recommendations(regime)
        print(f"{regime.value.replace('_', ' ').title()}:")
        sorted_allocs = sorted(allocations.items(), key=lambda x: x[1], reverse=True)
        for strategy, allocation in sorted_allocs[:5]:  # Top 5
            print(f"  {strategy}: {allocation:.1%}")
        print()

if __name__ == "__main__":
    main()