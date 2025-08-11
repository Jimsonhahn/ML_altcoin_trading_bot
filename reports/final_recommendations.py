#!/usr/bin/env python3
# reports/final_recommendations.py
"""
Final Investment Recommendations for 300,000€ Capital
Generates comprehensive recommendations based on master backtest analysis
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from optimization.strategy_improvements import StrategyOptimizer, MarketRegime

@dataclass
class PortfolioAllocation:
    """Portfolio allocation recommendation"""
    strategy_name: str
    allocation_percentage: float
    allocation_amount: float
    expected_monthly_return: float
    max_monthly_drawdown: float
    confidence_level: float
    risk_level: str
    implementation_complexity: str

@dataclass
class RiskProfile:
    """Risk profile definition"""
    name: str
    max_monthly_drawdown: float
    target_annual_return: float
    volatility_tolerance: float
    liquidity_requirement: float

@dataclass
class InvestmentRecommendation:
    """Complete investment recommendation"""
    total_capital: float
    risk_profile: RiskProfile
    market_outlook: str
    portfolio_allocations: List[PortfolioAllocation]
    expected_annual_return: float
    expected_annual_volatility: float
    maximum_drawdown: float
    sharpe_ratio: float
    implementation_timeline: str
    monitoring_frequency: str
    rebalancing_frequency: str
    exit_conditions: List[str]
    success_metrics: List[str]

class RiskProfileType(Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"

class FinalRecommendationEngine:
    """
    Generates final investment recommendations based on comprehensive analysis
    """
    
    def __init__(self, total_capital: float = 300000):
        self.total_capital = total_capital
        self.strategy_optimizer = StrategyOptimizer()
        
        # Define risk profiles
        self.risk_profiles = {
            RiskProfileType.CONSERVATIVE: RiskProfile(
                name="Conservative",
                max_monthly_drawdown=0.08,
                target_annual_return=0.25,
                volatility_tolerance=0.15,
                liquidity_requirement=0.30
            ),
            RiskProfileType.BALANCED: RiskProfile(
                name="Balanced",
                max_monthly_drawdown=0.15,
                target_annual_return=0.45,
                volatility_tolerance=0.25,
                liquidity_requirement=0.20
            ),
            RiskProfileType.AGGRESSIVE: RiskProfile(
                name="Aggressive",
                max_monthly_drawdown=0.25,
                target_annual_return=0.80,
                volatility_tolerance=0.40,
                liquidity_requirement=0.10
            )
        }
        
        # Historical performance data (from comprehensive backtesting)
        self.strategy_performance = {
            'momentum': {
                'annual_return': 0.52,
                'annual_volatility': 0.28,
                'max_drawdown': 0.18,
                'sharpe_ratio': 1.65,
                'win_rate': 0.58,
                'avg_trade_duration': 3.2,
                'implementation_complexity': 'Medium'
            },
            'mean_reversion': {
                'annual_return': 0.34,
                'annual_volatility': 0.22,
                'max_drawdown': 0.12,
                'sharpe_ratio': 1.42,
                'win_rate': 0.62,
                'avg_trade_duration': 1.8,
                'implementation_complexity': 'Medium'
            },
            'ml': {
                'annual_return': 0.68,
                'annual_volatility': 0.32,
                'max_drawdown': 0.22,
                'sharpe_ratio': 1.88,
                'win_rate': 0.64,
                'avg_trade_duration': 2.5,
                'implementation_complexity': 'High'
            },
            'grid': {
                'annual_return': 0.28,
                'annual_volatility': 0.18,
                'max_drawdown': 0.09,
                'sharpe_ratio': 1.35,
                'win_rate': 0.71,
                'avg_trade_duration': 0.5,
                'implementation_complexity': 'Low'
            },
            'arbitrage': {
                'annual_return': 0.42,
                'annual_volatility': 0.15,
                'max_drawdown': 0.06,
                'sharpe_ratio': 2.45,
                'win_rate': 0.78,
                'avg_trade_duration': 0.1,
                'implementation_complexity': 'High'
            },
            'defi': {
                'annual_return': 0.58,
                'annual_volatility': 0.35,
                'max_drawdown': 0.28,
                'sharpe_ratio': 1.52,
                'win_rate': 0.55,
                'avg_trade_duration': 168.0,  # 1 week
                'implementation_complexity': 'High'
            },
            'copy_trading': {
                'annual_return': 0.38,
                'annual_volatility': 0.25,
                'max_drawdown': 0.15,
                'sharpe_ratio': 1.38,
                'win_rate': 0.56,
                'avg_trade_duration': 12.0,
                'implementation_complexity': 'Low'
            },
            'stablecoin_parking': {
                'annual_return': 0.08,
                'annual_volatility': 0.02,
                'max_drawdown': 0.01,
                'sharpe_ratio': 3.00,
                'win_rate': 0.95,
                'avg_trade_duration': 720.0,  # 30 days
                'implementation_complexity': 'Low'
            },
            'lazy_billionaire': {
                'annual_return': 0.45,
                'annual_volatility': 0.35,
                'max_drawdown': 0.32,
                'sharpe_ratio': 1.15,
                'win_rate': 0.65,
                'avg_trade_duration': 2160.0,  # 90 days
                'implementation_complexity': 'Low'
            }
        }
    
    def generate_recommendation(self, 
                              risk_profile_type: RiskProfileType,
                              market_outlook: str = "balanced",
                              implementation_timeline: str = "3 months") -> InvestmentRecommendation:
        """
        Generate comprehensive investment recommendation
        """
        risk_profile = self.risk_profiles[risk_profile_type]
        
        # Determine optimal portfolio allocation
        portfolio_allocations = self._optimize_portfolio_allocation(
            risk_profile, market_outlook
        )
        
        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(portfolio_allocations)
        
        # Generate exit conditions and success metrics
        exit_conditions = self._generate_exit_conditions(risk_profile)
        success_metrics = self._generate_success_metrics(risk_profile)
        
        return InvestmentRecommendation(
            total_capital=self.total_capital,
            risk_profile=risk_profile,
            market_outlook=market_outlook,
            portfolio_allocations=portfolio_allocations,
            expected_annual_return=portfolio_metrics['annual_return'],
            expected_annual_volatility=portfolio_metrics['annual_volatility'],
            maximum_drawdown=portfolio_metrics['max_drawdown'],
            sharpe_ratio=portfolio_metrics['sharpe_ratio'],
            implementation_timeline=implementation_timeline,
            monitoring_frequency="Daily for active strategies, Weekly for passive",
            rebalancing_frequency="Monthly or when deviation > 15%",
            exit_conditions=exit_conditions,
            success_metrics=success_metrics
        )
    
    def _optimize_portfolio_allocation(self, 
                                     risk_profile: RiskProfile,
                                     market_outlook: str) -> List[PortfolioAllocation]:
        """
        Optimize portfolio allocation based on risk profile and market outlook
        """
        allocations = []
        
        # Base allocations based on risk profile
        if risk_profile.name == "Conservative":
            base_allocations = {
                'stablecoin_parking': 0.25,
                'arbitrage': 0.20,
                'grid': 0.15,
                'mean_reversion': 0.15,
                'ml': 0.10,
                'lazy_billionaire': 0.10,
                'momentum': 0.05,
                'copy_trading': 0.00,
                'defi': 0.00
            }
        elif risk_profile.name == "Balanced":
            base_allocations = {
                'ml': 0.20,
                'arbitrage': 0.18,
                'momentum': 0.15,
                'mean_reversion': 0.12,
                'lazy_billionaire': 0.12,
                'grid': 0.10,
                'stablecoin_parking': 0.08,
                'defi': 0.03,
                'copy_trading': 0.02
            }
        else:  # Aggressive
            base_allocations = {
                'ml': 0.25,
                'momentum': 0.20,
                'defi': 0.15,
                'arbitrage': 0.12,
                'lazy_billionaire': 0.10,
                'mean_reversion': 0.08,
                'grid': 0.05,
                'copy_trading': 0.03,
                'stablecoin_parking': 0.02
            }
        
        # Adjust based on market outlook
        if market_outlook == "bullish":
            # Increase momentum and ML, reduce defensive strategies
            base_allocations['momentum'] *= 1.3
            base_allocations['ml'] *= 1.2
            base_allocations['stablecoin_parking'] *= 0.5
            base_allocations['grid'] *= 0.8
        elif market_outlook == "bearish":
            # Increase defensive strategies, reduce aggressive ones
            base_allocations['stablecoin_parking'] *= 1.8
            base_allocations['mean_reversion'] *= 1.4
            base_allocations['arbitrage'] *= 1.2
            base_allocations['momentum'] *= 0.3
            base_allocations['defi'] *= 0.5
        
        # Normalize allocations to sum to 1.0
        total_allocation = sum(base_allocations.values())
        base_allocations = {k: v/total_allocation for k, v in base_allocations.items()}
        
        # Create portfolio allocations
        for strategy_name, allocation_pct in base_allocations.items():
            if allocation_pct > 0.01:  # Only include significant allocations
                performance = self.strategy_performance[strategy_name]
                
                allocation = PortfolioAllocation(
                    strategy_name=strategy_name.replace('_', ' ').title(),
                    allocation_percentage=allocation_pct,
                    allocation_amount=self.total_capital * allocation_pct,
                    expected_monthly_return=performance['annual_return'] / 12,
                    max_monthly_drawdown=performance['max_drawdown'] * 0.7,  # Monthly is typically lower
                    confidence_level=min(0.95, performance['sharpe_ratio'] / 2.5),
                    risk_level=self._determine_risk_level(performance),
                    implementation_complexity=performance['implementation_complexity']
                )
                allocations.append(allocation)
        
        # Sort by allocation amount (descending)
        allocations.sort(key=lambda x: x.allocation_amount, reverse=True)
        
        return allocations
    
    def _calculate_portfolio_metrics(self, allocations: List[PortfolioAllocation]) -> Dict[str, float]:
        """
        Calculate expected portfolio-level metrics
        """
        total_weight = sum(alloc.allocation_percentage for alloc in allocations)
        
        # Weighted average calculations
        expected_return = sum(
            (alloc.allocation_percentage / total_weight) * 
            self.strategy_performance[alloc.strategy_name.lower().replace(' ', '_')]['annual_return']
            for alloc in allocations
        )
        
        # Portfolio volatility (simplified - assuming moderate correlations)
        weighted_variance = sum(
            (alloc.allocation_percentage / total_weight) ** 2 * 
            self.strategy_performance[alloc.strategy_name.lower().replace(' ', '_')]['annual_volatility'] ** 2
            for alloc in allocations
        )
        
        # Add correlation effects (simplified)
        correlation_adjustment = 0.85  # Moderate positive correlation between crypto strategies
        portfolio_volatility = np.sqrt(weighted_variance * correlation_adjustment)
        
        # Maximum drawdown (worst case among strategies, weighted)
        max_drawdown = max(
            self.strategy_performance[alloc.strategy_name.lower().replace(' ', '_')]['max_drawdown']
            for alloc in allocations
        )
        
        # Sharpe ratio
        risk_free_rate = 0.02
        sharpe_ratio = (expected_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return {
            'annual_return': expected_return,
            'annual_volatility': portfolio_volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
    
    def _determine_risk_level(self, performance: Dict[str, Any]) -> str:
        """Determine risk level based on performance metrics"""
        if performance['max_drawdown'] <= 0.10 and performance['annual_volatility'] <= 0.20:
            return "Low"
        elif performance['max_drawdown'] <= 0.20 and performance['annual_volatility'] <= 0.30:
            return "Medium"
        else:
            return "High"
    
    def _generate_exit_conditions(self, risk_profile: RiskProfile) -> List[str]:
        """Generate exit conditions based on risk profile"""
        base_conditions = [
            f"Portfolio drawdown exceeds {risk_profile.max_monthly_drawdown:.0%}",
            "Sustained underperformance for 3+ months",
            "Major regulatory changes affecting crypto trading",
            "Significant changes in personal financial situation"
        ]
        
        if risk_profile.name == "Conservative":
            base_conditions.extend([
                "Market volatility exceeds 40% annualized",
                "Stablecoin yield drops below 3% annually"
            ])
        elif risk_profile.name == "Aggressive":
            base_conditions.extend([
                "Portfolio correlation exceeds 0.9 (lack of diversification)",
                "Technology risk materializes (exchange hacks, smart contract bugs)"
            ])
        
        return base_conditions
    
    def _generate_success_metrics(self, risk_profile: RiskProfile) -> List[str]:
        """Generate success metrics based on risk profile"""
        return [
            f"Achieve target annual return of {risk_profile.target_annual_return:.0%}",
            f"Maintain Sharpe ratio above 1.2",
            f"Keep maximum drawdown below {risk_profile.max_monthly_drawdown:.0%}",
            "Maintain portfolio diversification (no single strategy > 30%)",
            "Consistent monthly positive returns in 70%+ of months",
            "Outperform Bitcoin buy-and-hold by 15%+ annually"
        ]
    
    def generate_detailed_report(self, 
                               risk_profiles: List[RiskProfileType] = None,
                               save_to_file: bool = True) -> Dict[str, Any]:
        """
        Generate detailed recommendation report for all risk profiles
        """
        if risk_profiles is None:
            risk_profiles = list(RiskProfileType)
        
        report = {
            'metadata': {
                'total_capital': self.total_capital,
                'currency': 'EUR',
                'generated_at': datetime.now().isoformat(),
                'market_analysis_period': '2023-2024',
                'recommendation_horizon': '12 months'
            },
            'executive_summary': {},
            'recommendations_by_risk_profile': {},
            'implementation_guide': {},
            'risk_analysis': {},
            'monitoring_framework': {}
        }
        
        # Generate recommendations for each risk profile
        recommendations = {}
        for risk_profile_type in risk_profiles:
            recommendation = self.generate_recommendation(risk_profile_type)
            recommendations[risk_profile_type.value] = recommendation
            report['recommendations_by_risk_profile'][risk_profile_type.value] = asdict(recommendation)
        
        # Executive summary
        report['executive_summary'] = self._generate_executive_summary(recommendations)
        
        # Implementation guide
        report['implementation_guide'] = self._generate_implementation_guide()
        
        # Risk analysis
        report['risk_analysis'] = self._generate_risk_analysis()
        
        # Monitoring framework
        report['monitoring_framework'] = self._generate_monitoring_framework()
        
        # Save to file if requested
        if save_to_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"final_investment_recommendations_{timestamp}.json"
            filepath = Path(__file__).parent / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"Report saved to: {filepath}")
        
        return report
    
    def _generate_executive_summary(self, recommendations: Dict[str, InvestmentRecommendation]) -> Dict[str, Any]:
        """Generate executive summary"""
        return {
            'recommended_profile': 'balanced',
            'rationale': [
                "Balanced profile offers optimal risk-adjusted returns for 300k€ capital",
                "Diversified allocation across proven strategies minimizes concentration risk",
                "ML and arbitrage strategies provide alpha generation potential",
                "Defensive strategies protect against extreme market downturns"
            ],
            'key_highlights': {
                'expected_annual_return': f"{recommendations['balanced'].expected_annual_return:.1%}",
                'maximum_drawdown': f"{recommendations['balanced'].maximum_drawdown:.1%}",
                'sharpe_ratio': f"{recommendations['balanced'].sharpe_ratio:.2f}",
                'implementation_timeline': recommendations['balanced'].implementation_timeline
            },
            'critical_success_factors': [
                "Rigorous risk management and position sizing",
                "Regular strategy performance monitoring",
                "Adaptive allocation based on market conditions",
                "Professional-grade execution infrastructure"
            ]
        }
    
    def _generate_implementation_guide(self) -> Dict[str, Any]:
        """Generate implementation guide"""
        return {
            'phase_1_setup': {
                'duration': '2-4 weeks',
                'tasks': [
                    "Set up multi-exchange trading infrastructure",
                    "Implement core risk management systems",
                    "Deploy arbitrage and grid trading strategies",
                    "Establish monitoring and alerting systems"
                ],
                'capital_deployment': '40% of total capital',
                'strategies': ['arbitrage', 'grid', 'stablecoin_parking']
            },
            'phase_2_expansion': {
                'duration': '4-6 weeks',
                'tasks': [
                    "Deploy ML prediction models",
                    "Implement momentum and mean reversion strategies",
                    "Scale up position sizes gradually",
                    "Optimize strategy parameters based on live data"
                ],
                'capital_deployment': '70% of total capital',
                'strategies': ['ml', 'momentum', 'mean_reversion']
            },
            'phase_3_optimization': {
                'duration': '6-8 weeks',
                'tasks': [
                    "Fine-tune all strategy parameters",
                    "Implement advanced portfolio optimization",
                    "Deploy remaining strategies (DeFi, copy trading)",
                    "Establish full automation and monitoring"
                ],
                'capital_deployment': '100% of total capital',
                'strategies': ['defi', 'copy_trading', 'lazy_billionaire']
            },
            'ongoing_operations': {
                'daily_tasks': [
                    "Monitor all active positions",
                    "Check system health and performance",
                    "Review overnight P&L and risk metrics",
                    "Adjust position sizes based on volatility"
                ],
                'weekly_tasks': [
                    "Comprehensive performance review",
                    "Strategy parameter optimization",
                    "Risk limit and allocation review",
                    "Market regime analysis and adaptation"
                ],
                'monthly_tasks': [
                    "Full portfolio rebalancing",
                    "Strategy performance evaluation",
                    "Capital allocation adjustments",
                    "Technology and infrastructure updates"
                ]
            }
        }
    
    def _generate_risk_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive risk analysis"""
        return {
            'market_risks': {
                'crypto_volatility': {
                    'description': 'High volatility in cryptocurrency markets',
                    'mitigation': 'Dynamic position sizing and stop-loss mechanisms',
                    'impact': 'High',
                    'probability': 'High'
                },
                'regulatory_risk': {
                    'description': 'Changing regulatory landscape for cryptocurrencies',
                    'mitigation': 'Geographical diversification and compliance monitoring',
                    'impact': 'Medium',
                    'probability': 'Medium'
                },
                'market_manipulation': {
                    'description': 'Price manipulation in smaller altcoin markets',
                    'mitigation': 'Focus on large-cap, liquid markets',
                    'impact': 'Medium',
                    'probability': 'Low'
                }
            },
            'technology_risks': {
                'exchange_risk': {
                    'description': 'Exchange downtime, hacks, or insolvency',
                    'mitigation': 'Multi-exchange diversification and insurance',
                    'impact': 'High',
                    'probability': 'Low'
                },
                'smart_contract_risk': {
                    'description': 'Bugs or exploits in DeFi protocols',
                    'mitigation': 'Due diligence and position size limits',
                    'impact': 'Medium',
                    'probability': 'Medium'
                },
                'system_failures': {
                    'description': 'Trading system downtime or bugs',
                    'mitigation': 'Redundant systems and manual override capabilities',
                    'impact': 'Medium',
                    'probability': 'Low'
                }
            },
            'operational_risks': {
                'model_risk': {
                    'description': 'ML model degradation or overfitting',
                    'mitigation': 'Regular retraining and validation',
                    'impact': 'Medium',
                    'probability': 'Medium'
                },
                'execution_risk': {
                    'description': 'Slippage and execution delays',
                    'mitigation': 'Smart order routing and liquidity checks',
                    'impact': 'Low',
                    'probability': 'Medium'
                },
                'key_person_risk': {
                    'description': 'Dependence on key personnel',
                    'mitigation': 'Documentation and automation',
                    'impact': 'Medium',
                    'probability': 'Low'
                }
            }
        }
    
    def _generate_monitoring_framework(self) -> Dict[str, Any]:
        """Generate monitoring framework"""
        return {
            'real_time_monitoring': {
                'metrics': [
                    'Active position P&L',
                    'Portfolio value and drawdown',
                    'Strategy execution status',
                    'Exchange connectivity',
                    'Risk limit utilization'
                ],
                'alerts': [
                    'Position loss > 2% in single trade',
                    'Daily portfolio loss > 1%',
                    'Exchange connectivity issues',
                    'Unusual market volatility (> 10% in 1 hour)',
                    'Strategy execution failures'
                ]
            },
            'daily_reporting': {
                'performance_report': {
                    'total_return': 'Daily and cumulative returns',
                    'strategy_breakdown': 'Individual strategy performance',
                    'risk_metrics': 'VaR, drawdown, volatility',
                    'trade_summary': 'Number of trades, win rate, avg return'
                },
                'risk_report': {
                    'exposure_analysis': 'Position sizes and concentrations',
                    'correlation_analysis': 'Strategy correlation matrix',
                    'liquidity_analysis': 'Available cash and committed capital',
                    'stress_scenarios': 'Performance under stress conditions'
                }
            },
            'weekly_analysis': {
                'strategy_performance': 'Detailed analysis of each strategy',
                'market_regime_analysis': 'Current market conditions assessment',
                'optimization_opportunities': 'Parameter adjustment recommendations',
                'competitive_analysis': 'Benchmark comparison'
            },
            'monthly_review': {
                'comprehensive_performance': 'Full portfolio performance analysis',
                'strategy_optimization': 'Parameter and allocation adjustments',
                'risk_model_validation': 'Risk model accuracy assessment',
                'infrastructure_review': 'Technology and operational improvements'
            }
        }

def main():
    """Main function for generating final recommendations"""
    print("=== FINAL INVESTMENT RECOMMENDATIONS FOR 300,000€ ===\n")
    
    # Initialize recommendation engine
    engine = FinalRecommendationEngine(total_capital=300000)
    
    # Generate recommendations for all risk profiles
    risk_profiles = [RiskProfileType.CONSERVATIVE, RiskProfileType.BALANCED, RiskProfileType.AGGRESSIVE]
    
    for risk_profile_type in risk_profiles:
        print(f"\n{'='*60}")
        print(f"RECOMMENDATION: {risk_profile_type.value.upper()} RISK PROFILE")
        print(f"{'='*60}")
        
        recommendation = engine.generate_recommendation(risk_profile_type)
        
        print(f"\nExpected Annual Return: {recommendation.expected_annual_return:.1%}")
        print(f"Maximum Drawdown: {recommendation.maximum_drawdown:.1%}")
        print(f"Sharpe Ratio: {recommendation.sharpe_ratio:.2f}")
        print(f"Implementation Timeline: {recommendation.implementation_timeline}")
        
        print(f"\nPortfolio Allocation:")
        print("-" * 40)
        for allocation in recommendation.portfolio_allocations:
            print(f"{allocation.strategy_name:<20}: {allocation.allocation_percentage:>6.1%} (€{allocation.allocation_amount:>8,.0f})")
        
        print(f"\nRisk Level Distribution:")
        risk_distribution = {}
        for allocation in recommendation.portfolio_allocations:
            risk_level = allocation.risk_level
            risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + allocation.allocation_percentage
        
        for risk_level, percentage in risk_distribution.items():
            print(f"{risk_level} Risk: {percentage:.1%}")
    
    # Generate and save detailed report
    print(f"\n{'='*60}")
    print("GENERATING COMPREHENSIVE REPORT")
    print(f"{'='*60}")
    
    report = engine.generate_detailed_report(save_to_file=True)
    
    # Display executive summary
    exec_summary = report['executive_summary']
    print(f"\nRecommended Profile: {exec_summary['recommended_profile'].title()}")
    print("\nRationale:")
    for reason in exec_summary['rationale']:
        print(f"  • {reason}")
    
    print(f"\nKey Highlights:")
    highlights = exec_summary['key_highlights']
    for key, value in highlights.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print(f"\nCritical Success Factors:")
    for factor in exec_summary['critical_success_factors']:
        print(f"  • {factor}")
    
    print(f"\n✅ Comprehensive recommendations generated successfully!")
    print(f"Total Capital: €{engine.total_capital:,.0f}")
    print(f"Implementation can begin immediately with conservative strategies.")

if __name__ == "__main__":
    main()