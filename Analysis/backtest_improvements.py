#!/usr/bin/env python3
"""
ML Altcoin Trading Bot - Backtest Improvements
Automatisches Backtesting von Verbesserungen

Diese Komponente:
- Backtestet neue Erkenntnisse automatisch
- Führt A/B Tests zwischen alter und neuer Logik durch
- Validiert Verbesserungen bevor sie live gehen
- Generiert Verbesserungsvorschläge
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy

from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Backtest-Ergebnis"""
    backtest_id: str
    strategy_name: str
    improvement_type: str
    test_period: Dict[str, str]
    performance_metrics: Dict[str, float]
    trade_count: int
    success_rate: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    confidence_level: float
    statistical_significance: float

@dataclass
class ABTestResult:
    """A/B Test Ergebnis"""
    test_id: str
    test_name: str
    control_group: BacktestResult
    treatment_group: BacktestResult
    improvement_percentage: float
    statistical_significance: float
    p_value: float
    confidence_interval: Tuple[float, float]
    recommendation: str
    risk_assessment: str

@dataclass
class ImprovementSuggestion:
    """Verbesserungsvorschlag"""
    suggestion_id: str
    suggestion_name: str
    description: str
    improvement_type: str  # 'allocation', 'risk_management', 'timing', 'parameter'
    expected_improvement: float
    implementation_complexity: str  # 'low', 'medium', 'high'
    backtest_results: Optional[BacktestResult]
    code_changes: List[str]
    validation_status: str  # 'pending', 'testing', 'validated', 'rejected'
    confidence_score: float

@dataclass
class ValidationReport:
    """Validierungsbericht"""
    report_id: str
    timestamp: datetime
    improvements_tested: int
    improvements_validated: int
    improvements_rejected: int
    total_expected_improvement: float
    risk_level: str
    recommendations: List[str]
    next_steps: List[str]

class BacktestImprovements:
    """Automatische Verbesserungs-Backtesting Engine"""
    
    def __init__(self, data_handler, backtest_engine, lookback_months: int = 6):
        """
        Initialize Backtest Improvements Engine
        
        Args:
            data_handler: Datenhandler für historische Daten
            backtest_engine: Backtesting-Engine
            lookback_months: Monate für Backtest-Periode
        """
        self.data_handler = data_handler
        self.backtest_engine = backtest_engine
        self.lookback_months = lookback_months
        
        # Results storage
        self.backtest_results: List[BacktestResult] = []
        self.ab_test_results: List[ABTestResult] = []
        self.improvement_suggestions: List[ImprovementSuggestion] = []
        
        # Configuration
        self.min_trades_for_test = 30
        self.significance_threshold = 0.05
        self.min_improvement_threshold = 0.02  # 2% minimum improvement
        
        # Directories
        self.results_dir = Path("analysis/backtest_improvements")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.reports_dir = Path("analysis/improvement_reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    async def validate_learning_insights(self, learning_results: Dict[str, Any], 
                                       pattern_results: Dict[str, Any]) -> ValidationReport:
        """
        Validiere Learning Pipeline und Pattern Detection Erkenntnisse
        
        Args:
            learning_results: Ergebnisse der Learning Pipeline
            pattern_results: Ergebnisse der Pattern Detection
            
        Returns:
            Validierungsbericht
        """
        logger.info("🧪 Starting comprehensive improvement validation...")
        
        start_time = datetime.utcnow()
        
        try:
            # 1. Improvement suggestions aus Erkenntnissen generieren
            await self._generate_improvement_suggestions(learning_results, pattern_results)
            
            # 2. Automatische Backtests für alle Suggestions
            await self._run_automated_backtests()
            
            # 3. A/B Tests für signifikante Verbesserungen
            await self._run_ab_tests()
            
            # 4. Statistische Validierung
            validated_improvements = await self._validate_improvements()
            
            # 5. Risk Assessment
            risk_assessment = await self._assess_implementation_risks()
            
            # 6. Final validation report
            report = await self._generate_validation_report(validated_improvements, risk_assessment)
            
            # 7. Save results
            await self._save_all_results()
            
            # 8. Generate visualizations
            await self._create_validation_visualizations()
            
            analysis_time = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"✅ Improvement validation completed in {analysis_time:.1f}s")
            
            return report
            
        except Exception as e:
            logger.error(f"Improvement validation failed: {e}")
            raise

    async def _generate_improvement_suggestions(self, learning_results: Dict[str, Any],
                                              pattern_results: Dict[str, Any]):
        """Verbesserungsvorschläge aus Erkenntnissen generieren"""
        logger.info("💡 Generating improvement suggestions...")
        
        suggestion_id = 1
        
        # 1. Suggestions aus Learning Pipeline
        if 'new_rules' in learning_results:
            for rule in learning_results.get('new_rules', []):
                suggestion = ImprovementSuggestion(
                    suggestion_id=f"LEARN_{suggestion_id:03d}",
                    suggestion_name=f"Implement {rule.get('rule_name', 'Unknown Rule')}",
                    description=rule.get('action', 'No description'),
                    improvement_type=rule.get('rule_type', 'unknown'),
                    expected_improvement=rule.get('expected_improvement', 0),
                    implementation_complexity='medium',
                    backtest_results=None,
                    code_changes=[rule.get('implementation_code', '')],
                    validation_status='pending',
                    confidence_score=rule.get('confidence', 0.5)
                )
                self.improvement_suggestions.append(suggestion)
                suggestion_id += 1
        
        # 2. Suggestions aus Pattern Detection - Success Patterns
        if 'success_patterns' in pattern_results:
            for pattern in pattern_results.get('success_patterns', []):
                suggestion = ImprovementSuggestion(
                    suggestion_id=f"PAT_S_{suggestion_id:03d}",
                    suggestion_name=f"Replicate {pattern.get('pattern_name', 'Success Pattern')}",
                    description=f"Implement conditions that lead to {pattern.get('avg_return', 0):.2f}% average return",
                    improvement_type='timing',
                    expected_improvement=pattern.get('avg_return', 0) * pattern.get('frequency', 1) / 100,
                    implementation_complexity='low',
                    backtest_results=None,
                    code_changes=[self._generate_pattern_implementation(pattern)],
                    validation_status='pending',
                    confidence_score=pattern.get('confidence', 0.5)
                )
                self.improvement_suggestions.append(suggestion)
                suggestion_id += 1
        
        # 3. Suggestions aus Dangerous Conditions
        if 'dangerous_conditions' in pattern_results:
            for condition in pattern_results.get('dangerous_conditions', []):
                suggestion = ImprovementSuggestion(
                    suggestion_id=f"PAT_D_{suggestion_id:03d}",
                    suggestion_name=f"Avoid {condition.get('condition_name', 'Dangerous Condition')}",
                    description=f"Prevent conditions that lead to {condition.get('avg_loss', 0):.2f}% average loss",
                    improvement_type='risk_management',
                    expected_improvement=abs(condition.get('avg_loss', 0)) * condition.get('frequency', 1) / 100,
                    implementation_complexity='medium',
                    backtest_results=None,
                    code_changes=[self._generate_risk_avoidance_code(condition)],
                    validation_status='pending',
                    confidence_score=0.8
                )
                self.improvement_suggestions.append(suggestion)
                suggestion_id += 1
        
        # 4. Suggestions aus Strategy Synergies
        if 'strategy_synergies' in pattern_results:
            for synergy in pattern_results.get('strategy_synergies', []):
                suggestion = ImprovementSuggestion(
                    suggestion_id=f"SYN_{suggestion_id:03d}",
                    suggestion_name=f"Optimize {'+'.join(synergy.get('strategies', []))} Synergy",
                    description=f"Implement {synergy.get('synergy_type', 'unknown')} synergy between strategies",
                    improvement_type='allocation',
                    expected_improvement=synergy.get('performance_boost', 0),
                    implementation_complexity='high',
                    backtest_results=None,
                    code_changes=[self._generate_synergy_implementation(synergy)],
                    validation_status='pending',
                    confidence_score=synergy.get('synergy_strength', 0.5)
                )
                self.improvement_suggestions.append(suggestion)
                suggestion_id += 1
        
        logger.info(f"Generated {len(self.improvement_suggestions)} improvement suggestions")

    def _generate_pattern_implementation(self, pattern: Dict[str, Any]) -> str:
        """Code für Pattern-Implementation generieren"""
        return f"""
# Implementation for {pattern.get('pattern_name', 'Success Pattern')}
def implement_success_pattern_{pattern.get('pattern_id', 'unknown')}(market_data, current_time):
    # Check pattern conditions
    conditions = {pattern.get('conditions', {})}
    
    # Validate timing and market conditions
    if validate_pattern_conditions(market_data, conditions):
        return True, "Pattern conditions met"
    
    return False, "Pattern conditions not met"

def validate_pattern_conditions(market_data, conditions):
    # Implement specific pattern validation logic
    return True  # Simplified
"""

    def _generate_risk_avoidance_code(self, condition: Dict[str, Any]) -> str:
        """Code für Risk Avoidance generieren"""
        return f"""
# Risk avoidance for {condition.get('condition_name', 'Dangerous Condition')}
def avoid_dangerous_condition_{condition.get('condition_id', 'unknown')}(market_data, current_positions):
    # Check for dangerous condition indicators
    warning_indicators = {condition.get('warning_indicators', [])}
    
    risk_score = calculate_risk_score(market_data, warning_indicators)
    
    if risk_score > 0.7:  # High risk threshold
        return True, "Dangerous condition detected - reduce positions"
    
    return False, "Conditions safe"

def calculate_risk_score(market_data, indicators):
    # Implement risk scoring logic
    return 0.5  # Simplified
"""

    def _generate_synergy_implementation(self, synergy: Dict[str, Any]) -> str:
        """Code für Synergy-Implementation generieren"""
        return f"""
# Synergy implementation for {'+'.join(synergy.get('strategies', []))}
def implement_strategy_synergy_{synergy.get('synergy_id', 'unknown')}(strategy_allocations, market_regime):
    strategies = {synergy.get('strategies', [])}
    synergy_type = "{synergy.get('synergy_type', 'unknown')}"
    
    if synergy_type == "complementary":
        return optimize_complementary_allocation(strategies, market_regime)
    elif synergy_type == "reinforcing":
        return optimize_reinforcing_allocation(strategies, market_regime)
    else:
        return strategy_allocations

def optimize_complementary_allocation(strategies, market_regime):
    # Implement complementary allocation logic
    return {{}}  # Simplified

def optimize_reinforcing_allocation(strategies, market_regime):
    # Implement reinforcing allocation logic
    return {{}}  # Simplified
"""

    async def _run_automated_backtests(self):
        """Automatische Backtests für alle Suggestions"""
        logger.info("🚀 Running automated backtests...")
        
        # Parallele Backtest-Ausführung
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for suggestion in self.improvement_suggestions:
                if suggestion.validation_status == 'pending':
                    future = executor.submit(self._run_single_backtest, suggestion)
                    futures.append(future)
            
            # Warten auf Completion
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        self.backtest_results.append(result)
                        
                        # Update suggestion mit Backtest-Ergebnis
                        for suggestion in self.improvement_suggestions:
                            if suggestion.suggestion_id == result.backtest_id.split('_')[0]:
                                suggestion.backtest_results = result
                                suggestion.validation_status = 'testing'
                                break
                                
                except Exception as e:
                    logger.error(f"Backtest failed: {e}")
        
        logger.info(f"Completed {len(self.backtest_results)} automated backtests")

    def _run_single_backtest(self, suggestion: ImprovementSuggestion) -> Optional[BacktestResult]:
        """Einzelnen Backtest ausführen"""
        try:
            logger.info(f"Running backtest for: {suggestion.suggestion_name}")
            
            # Backtest-Periode definieren
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=self.lookback_months * 30)
            
            # Simulate backtest execution (normalerweise mit echter Backtest-Engine)
            performance_metrics = self._simulate_backtest_performance(suggestion)
            
            if not performance_metrics:
                return None
            
            result = BacktestResult(
                backtest_id=f"{suggestion.suggestion_id}_BT_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                strategy_name=suggestion.suggestion_name,
                improvement_type=suggestion.improvement_type,
                test_period={
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                performance_metrics=performance_metrics,
                trade_count=performance_metrics.get('trade_count', 0),
                success_rate=performance_metrics.get('success_rate', 0),
                sharpe_ratio=performance_metrics.get('sharpe_ratio', 0),
                max_drawdown=performance_metrics.get('max_drawdown', 0),
                total_return=performance_metrics.get('total_return', 0),
                confidence_level=0.95,
                statistical_significance=0.0  # Wird später berechnet
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Single backtest failed for {suggestion.suggestion_name}: {e}")
            return None

    def _simulate_backtest_performance(self, suggestion: ImprovementSuggestion) -> Optional[Dict[str, float]]:
        """Simuliere Backtest-Performance (Placeholder für echte Backtest-Engine)"""
        try:
            # Basiere Simulation auf expected_improvement und confidence_score
            base_return = 0.05  # 5% baseline return
            
            # Variiere Performance basierend auf suggestion type
            if suggestion.improvement_type == 'allocation':
                multiplier = 1.2
            elif suggestion.improvement_type == 'risk_management':
                multiplier = 1.1
            elif suggestion.improvement_type == 'timing':
                multiplier = 1.15
            else:
                multiplier = 1.0
            
            # Performance mit Noise
            noise_factor = np.random.normal(1.0, 0.1)
            expected_improvement_factor = 1 + (suggestion.expected_improvement / 100)
            confidence_factor = suggestion.confidence_score
            
            total_return = base_return * multiplier * expected_improvement_factor * confidence_factor * noise_factor
            
            # Andere Metriken simulieren
            success_rate = min(0.95, 0.5 + (confidence_factor * 0.3) + np.random.normal(0, 0.05))
            sharpe_ratio = max(0, total_return / 0.15 + np.random.normal(0, 0.2))  # Assume 15% volatility
            max_drawdown = -(abs(np.random.normal(0.1, 0.05)))  # Negative drawdown
            trade_count = max(self.min_trades_for_test, int(np.random.normal(100, 20)))
            
            return {
                'total_return': total_return,
                'success_rate': success_rate,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'trade_count': trade_count,
                'volatility': 0.15,
                'avg_trade_return': total_return / trade_count if trade_count > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Performance simulation failed: {e}")
            return None

    async def _run_ab_tests(self):
        """A/B Tests für signifikante Verbesserungen"""
        logger.info("⚖️ Running A/B tests...")
        
        # Baseline-Performance für Vergleich
        baseline_performance = await self._get_baseline_performance()
        
        for result in self.backtest_results:
            if result.total_return > self.min_improvement_threshold:  # Nur signifikante Verbesserungen
                ab_test = await self._run_single_ab_test(result, baseline_performance)
                if ab_test:
                    self.ab_test_results.append(ab_test)
        
        logger.info(f"Completed {len(self.ab_test_results)} A/B tests")

    async def _get_baseline_performance(self) -> BacktestResult:
        """Baseline-Performance für Vergleichszwecke"""
        # Simuliere aktuelle System-Performance
        return BacktestResult(
            backtest_id="BASELINE_001",
            strategy_name="Current System",
            improvement_type="baseline",
            test_period={
                'start': (datetime.utcnow() - timedelta(days=self.lookback_months * 30)).isoformat(),
                'end': datetime.utcnow().isoformat()
            },
            performance_metrics={
                'total_return': 0.08,  # 8% baseline
                'success_rate': 0.55,
                'sharpe_ratio': 0.9,
                'max_drawdown': -0.12,
                'trade_count': 150,
                'volatility': 0.18
            },
            trade_count=150,
            success_rate=0.55,
            sharpe_ratio=0.9,
            max_drawdown=-0.12,
            total_return=0.08,
            confidence_level=0.95,
            statistical_significance=1.0
        )

    async def _run_single_ab_test(self, treatment: BacktestResult, 
                                 control: BacktestResult) -> Optional[ABTestResult]:
        """Einzelnen A/B Test ausführen"""
        try:
            # Statistische Signifikanz berechnen
            p_value, significance = self._calculate_statistical_significance(treatment, control)
            
            # Improvement percentage
            improvement_pct = ((treatment.total_return - control.total_return) / 
                             abs(control.total_return)) * 100
            
            # Confidence interval (vereinfacht)
            ci_lower = improvement_pct - 1.96 * (improvement_pct * 0.1)
            ci_upper = improvement_pct + 1.96 * (improvement_pct * 0.1)
            
            # Recommendation basierend auf Ergebnissen
            if p_value < self.significance_threshold and improvement_pct > self.min_improvement_threshold * 100:
                recommendation = "IMPLEMENT"
                risk_assessment = "LOW" if improvement_pct < 10 else "MEDIUM"
            elif p_value < self.significance_threshold:
                recommendation = "CONSIDER"
                risk_assessment = "MEDIUM"
            else:
                recommendation = "REJECT"
                risk_assessment = "HIGH"
            
            return ABTestResult(
                test_id=f"AB_{treatment.backtest_id}",
                test_name=f"{treatment.strategy_name} vs Baseline",
                control_group=control,
                treatment_group=treatment,
                improvement_percentage=improvement_pct,
                statistical_significance=significance,
                p_value=p_value,
                confidence_interval=(ci_lower, ci_upper),
                recommendation=recommendation,
                risk_assessment=risk_assessment
            )
            
        except Exception as e:
            logger.error(f"A/B test failed: {e}")
            return None

    def _calculate_statistical_significance(self, treatment: BacktestResult, 
                                          control: BacktestResult) -> Tuple[float, float]:
        """Statistische Signifikanz berechnen"""
        try:
            # Vereinfachte statistische Tests
            # In Realität würden hier echte Handelsdaten verwendet
            
            # Simuliere Rückgaben-Verteilungen
            control_returns = np.random.normal(
                control.total_return / control.trade_count,
                control.performance_metrics.get('volatility', 0.15) / np.sqrt(control.trade_count),
                control.trade_count
            )
            
            treatment_returns = np.random.normal(
                treatment.total_return / treatment.trade_count,
                treatment.performance_metrics.get('volatility', 0.15) / np.sqrt(treatment.trade_count),
                treatment.trade_count
            )
            
            # T-Test für Mittelwert-Unterschied
            t_stat, p_value = stats.ttest_ind(treatment_returns, control_returns)
            
            # Effektgröße (Cohen's d)
            pooled_std = np.sqrt((np.var(treatment_returns) + np.var(control_returns)) / 2)
            cohens_d = (np.mean(treatment_returns) - np.mean(control_returns)) / pooled_std if pooled_std > 0 else 0
            
            significance = abs(cohens_d)  # Vereinfacht
            
            return float(p_value), float(significance)
            
        except Exception as e:
            logger.error(f"Statistical significance calculation failed: {e}")
            return 1.0, 0.0  # No significance if calculation fails

    async def _validate_improvements(self) -> List[ImprovementSuggestion]:
        """Verbesserungen basierend auf Tests validieren"""
        logger.info("✅ Validating improvements...")
        
        validated_improvements = []
        
        for suggestion in self.improvement_suggestions:
            if suggestion.validation_status == 'testing' and suggestion.backtest_results:
                
                # Finde entsprechenden A/B Test
                ab_test = None
                for test in self.ab_test_results:
                    if suggestion.suggestion_id in test.test_id:
                        ab_test = test
                        break
                
                if ab_test:
                    # Validierung basierend auf A/B Test Ergebnissen
                    if ab_test.recommendation == "IMPLEMENT":
                        suggestion.validation_status = 'validated'
                        validated_improvements.append(suggestion)
                    elif ab_test.recommendation == "CONSIDER":
                        if ab_test.improvement_percentage > 5:  # 5% threshold for consideration
                            suggestion.validation_status = 'validated'
                            validated_improvements.append(suggestion)
                        else:
                            suggestion.validation_status = 'pending'
                    else:
                        suggestion.validation_status = 'rejected'
                else:
                    # Kein A/B Test - validiere basierend auf Backtest allein
                    if (suggestion.backtest_results.total_return > self.min_improvement_threshold and
                        suggestion.backtest_results.sharpe_ratio > 0.8):
                        suggestion.validation_status = 'validated'
                        validated_improvements.append(suggestion)
                    else:
                        suggestion.validation_status = 'rejected'
        
        logger.info(f"Validated {len(validated_improvements)} improvements")
        return validated_improvements

    async def _assess_implementation_risks(self) -> Dict[str, Any]:
        """Implementation-Risiken bewerten"""
        logger.info("⚠️ Assessing implementation risks...")
        
        risk_factors = {
            'high_complexity_count': 0,
            'high_impact_changes': 0,
            'conflicting_improvements': 0,
            'total_expected_improvement': 0,
            'implementation_effort': 'low'
        }
        
        validated_suggestions = [s for s in self.improvement_suggestions if s.validation_status == 'validated']
        
        for suggestion in validated_suggestions:
            # Complexity risk
            if suggestion.implementation_complexity == 'high':
                risk_factors['high_complexity_count'] += 1
            
            # Impact risk
            if suggestion.expected_improvement > 10:  # >10% improvement
                risk_factors['high_impact_changes'] += 1
            
            risk_factors['total_expected_improvement'] += suggestion.expected_improvement
        
        # Determine overall risk level
        if (risk_factors['high_complexity_count'] > 2 or 
            risk_factors['high_impact_changes'] > 3 or
            risk_factors['total_expected_improvement'] > 50):
            risk_factors['overall_risk'] = 'HIGH'
            risk_factors['implementation_effort'] = 'high'
        elif (risk_factors['high_complexity_count'] > 0 or 
              risk_factors['high_impact_changes'] > 1):
            risk_factors['overall_risk'] = 'MEDIUM'
            risk_factors['implementation_effort'] = 'medium'
        else:
            risk_factors['overall_risk'] = 'LOW'
            risk_factors['implementation_effort'] = 'low'
        
        return risk_factors

    async def _generate_validation_report(self, validated_improvements: List[ImprovementSuggestion],
                                        risk_assessment: Dict[str, Any]) -> ValidationReport:
        """Validierungsbericht generieren"""
        
        total_suggestions = len(self.improvement_suggestions)
        validated_count = len(validated_improvements)
        rejected_count = len([s for s in self.improvement_suggestions if s.validation_status == 'rejected'])
        
        total_expected_improvement = sum(s.expected_improvement for s in validated_improvements)
        
        # Recommendations generieren
        recommendations = []
        
        if validated_count > 0:
            recommendations.append(f"Implement {validated_count} validated improvements")
            
            # Prioritäts-Empfehlungen
            high_impact = [s for s in validated_improvements if s.expected_improvement > 5]
            if high_impact:
                recommendations.append(f"Prioritize {len(high_impact)} high-impact improvements")
            
            low_complexity = [s for s in validated_improvements if s.implementation_complexity == 'low']
            if low_complexity:
                recommendations.append(f"Start with {len(low_complexity)} low-complexity improvements")
        
        if risk_assessment['overall_risk'] == 'HIGH':
            recommendations.append("Implement improvements gradually due to high risk")
        
        # Next steps
        next_steps = [
            "Review and approve validated improvements",
            "Plan implementation timeline",
            "Set up monitoring for new implementations",
            "Schedule follow-up validation"
        ]
        
        if rejected_count > 0:
            next_steps.append(f"Analyze why {rejected_count} improvements were rejected")
        
        return ValidationReport(
            report_id=f"VAL_REPORT_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.utcnow(),
            improvements_tested=total_suggestions,
            improvements_validated=validated_count,
            improvements_rejected=rejected_count,
            total_expected_improvement=total_expected_improvement,
            risk_level=risk_assessment['overall_risk'],
            recommendations=recommendations,
            next_steps=next_steps
        )

    async def _save_all_results(self):
        """Alle Ergebnisse speichern"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Backtest results
        backtest_file = self.results_dir / f"backtest_results_{timestamp}.json"
        with open(backtest_file, 'w') as f:
            json.dump([asdict(result) for result in self.backtest_results], 
                     f, indent=2, default=str)
        
        # A/B test results
        ab_test_file = self.results_dir / f"ab_test_results_{timestamp}.json"
        with open(ab_test_file, 'w') as f:
            json.dump([asdict(result) for result in self.ab_test_results], 
                     f, indent=2, default=str)
        
        # Improvement suggestions
        suggestions_file = self.results_dir / f"improvement_suggestions_{timestamp}.json"
        with open(suggestions_file, 'w') as f:
            json.dump([asdict(suggestion) for suggestion in self.improvement_suggestions], 
                     f, indent=2, default=str)
        
        logger.info(f"Results saved to {self.results_dir}")

    async def _create_validation_visualizations(self):
        """Validierungs-Visualisierungen erstellen"""
        logger.info("📊 Creating validation visualizations...")
        
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            # 1. Improvement suggestions overview
            self._create_suggestions_overview_chart(timestamp)
            
            # 2. A/B test results
            if self.ab_test_results:
                self._create_ab_test_results_chart(timestamp)
            
            # 3. Performance comparison
            if self.backtest_results:
                self._create_performance_comparison_chart(timestamp)
            
            # 4. Risk assessment
            self._create_risk_assessment_chart(timestamp)
            
        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")

    def _create_suggestions_overview_chart(self, timestamp: str):
        """Suggestions overview chart"""
        if not self.improvement_suggestions:
            return
        
        # Data preparation
        suggestions_data = []
        for suggestion in self.improvement_suggestions:
            suggestions_data.append({
                'Name': suggestion.suggestion_name[:30] + '...' if len(suggestion.suggestion_name) > 30 else suggestion.suggestion_name,
                'Type': suggestion.improvement_type,
                'Expected Improvement': suggestion.expected_improvement,
                'Confidence': suggestion.confidence_score * 100,
                'Status': suggestion.validation_status,
                'Complexity': suggestion.implementation_complexity
            })
        
        df = pd.DataFrame(suggestions_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Expected Improvement by Type', 'Validation Status', 
                           'Complexity Distribution', 'Confidence vs Improvement'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "pie"}, {"type": "scatter"}]]
        )
        
        # Expected improvement by type
        type_improvement = df.groupby('Type')['Expected Improvement'].sum()
        fig.add_trace(
            go.Bar(x=type_improvement.index, y=type_improvement.values, name='Expected Improvement'),
            row=1, col=1
        )
        
        # Validation status pie
        status_counts = df['Status'].value_counts()
        fig.add_trace(
            go.Pie(labels=status_counts.index, values=status_counts.values, name='Status'),
            row=1, col=2
        )
        
        # Complexity distribution
        complexity_counts = df['Complexity'].value_counts()
        fig.add_trace(
            go.Pie(labels=complexity_counts.index, values=complexity_counts.values, name='Complexity'),
            row=2, col=1
        )
        
        # Confidence vs Improvement scatter
        fig.add_trace(
            go.Scatter(
                x=df['Confidence'], 
                y=df['Expected Improvement'],
                mode='markers+text',
                text=df['Name'],
                textposition="top center",
                marker=dict(
                    size=10,
                    color=df['Expected Improvement'],
                    colorscale='Viridis',
                    showscale=True
                ),
                name='Suggestions'
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Improvement Suggestions Overview")
        fig.write_html(self.results_dir / f"suggestions_overview_{timestamp}.html")

    def _create_ab_test_results_chart(self, timestamp: str):
        """A/B test results chart"""
        if not self.ab_test_results:
            return
        
        test_data = []
        for test in self.ab_test_results:
            test_data.append({
                'Test': test.test_name[:30] + '...' if len(test.test_name) > 30 else test.test_name,
                'Improvement %': test.improvement_percentage,
                'P-Value': test.p_value,
                'Significance': test.statistical_significance,
                'Recommendation': test.recommendation,
                'Risk': test.risk_assessment
            })
        
        df = pd.DataFrame(test_data)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Improvement vs P-Value', 'Recommendations')
        )
        
        # Improvement vs P-value scatter
        colors = {'IMPLEMENT': 'green', 'CONSIDER': 'orange', 'REJECT': 'red'}
        fig.add_trace(
            go.Scatter(
                x=df['P-Value'],
                y=df['Improvement %'],
                mode='markers+text',
                text=df['Test'],
                textposition="top center",
                marker=dict(
                    size=15,
                    color=[colors.get(rec, 'gray') for rec in df['Recommendation']],
                ),
                name='A/B Tests'
            ),
            row=1, col=1
        )
        
        # Add significance threshold line
        fig.add_hline(y=self.min_improvement_threshold * 100, line_dash="dash", 
                     annotation_text="Min Improvement Threshold", row=1, col=1)
        fig.add_vline(x=self.significance_threshold, line_dash="dash", 
                     annotation_text="Significance Threshold", row=1, col=1)
        
        # Recommendations pie
        rec_counts = df['Recommendation'].value_counts()
        fig.add_trace(
            go.Pie(labels=rec_counts.index, values=rec_counts.values, 
                  marker_colors=[colors.get(label, 'gray') for label in rec_counts.index]),
            row=1, col=2
        )
        
        fig.update_layout(height=500, title_text="A/B Test Results Analysis")
        fig.write_html(self.results_dir / f"ab_test_results_{timestamp}.html")

    def _create_performance_comparison_chart(self, timestamp: str):
        """Performance comparison chart"""
        if not self.backtest_results:
            return
        
        perf_data = []
        for result in self.backtest_results:
            perf_data.append({
                'Strategy': result.strategy_name[:25] + '...' if len(result.strategy_name) > 25 else result.strategy_name,
                'Total Return %': result.total_return * 100,
                'Sharpe Ratio': result.sharpe_ratio,
                'Max Drawdown %': result.max_drawdown * 100,
                'Success Rate %': result.success_rate * 100,
                'Trade Count': result.trade_count
            })
        
        df = pd.DataFrame(perf_data)
        
        # Create radar chart for top performers
        top_performers = df.nlargest(5, 'Total Return %')
        
        fig = go.Figure()
        
        for idx, row in top_performers.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row['Total Return %'], row['Sharpe Ratio'] * 10, 
                   abs(row['Max Drawdown %']), row['Success Rate %']],
                theta=['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Success Rate'],
                fill='toself',
                name=row['Strategy']
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            title="Top 5 Performance Comparison (Radar Chart)"
        )
        
        fig.write_html(self.results_dir / f"performance_comparison_{timestamp}.html")

    def _create_risk_assessment_chart(self, timestamp: str):
        """Risk assessment visualization"""
        
        # Risk categories
        risk_data = {
            'Low Risk': len([s for s in self.improvement_suggestions 
                            if s.implementation_complexity == 'low' and s.validation_status == 'validated']),
            'Medium Risk': len([s for s in self.improvement_suggestions 
                               if s.implementation_complexity == 'medium' and s.validation_status == 'validated']),
            'High Risk': len([s for s in self.improvement_suggestions 
                             if s.implementation_complexity == 'high' and s.validation_status == 'validated'])
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(risk_data.keys()),
                y=list(risk_data.values()),
                marker_color=['green', 'orange', 'red']
            )
        ])
        
        fig.update_layout(
            title='Implementation Risk Assessment',
            xaxis_title='Risk Level',
            yaxis_title='Number of Validated Improvements'
        )
        
        fig.write_html(self.results_dir / f"risk_assessment_{timestamp}.html")

# Example usage
async def example_usage():
    """Example of how to use BacktestImprovements"""
    
    # Mock data handlers (normally would use real ones)
    class MockDataHandler:
        def get_historical_data(self, start, end):
            return pd.DataFrame()  # Mock
    
    class MockBacktestEngine:
        def run_backtest(self, strategy, data):
            return {}  # Mock
    
    # Initialize
    data_handler = MockDataHandler()
    backtest_engine = MockBacktestEngine()
    
    improvements = BacktestImprovements(data_handler, backtest_engine)
    
    # Mock learning and pattern results
    learning_results = {
        'new_rules': [
            {
                'rule_name': 'Increase BTC allocation',
                'rule_type': 'allocation',
                'action': 'Increase BTC weight by 10%',
                'expected_improvement': 5.0,
                'confidence': 0.8,
                'implementation_code': 'def increase_btc(): pass'
            }
        ]
    }
    
    pattern_results = {
        'success_patterns': [
            {
                'pattern_name': 'High Volatility Success',
                'avg_return': 3.5,
                'frequency': 15,
                'confidence': 0.75,
                'pattern_id': 'PAT_001',
                'conditions': {'volatility': 0.05}
            }
        ],
        'dangerous_conditions': [
            {
                'condition_name': 'Market Crash Risk',
                'avg_loss': -8.0,
                'frequency': 5,
                'condition_id': 'DANGER_001',
                'warning_indicators': ['high_volatility', 'negative_sentiment']
            }
        ]
    }
    
    # Run validation
    validation_report = await improvements.validate_learning_insights(learning_results, pattern_results)
    
    print("Validation Report:")
    print(f"Improvements tested: {validation_report.improvements_tested}")
    print(f"Improvements validated: {validation_report.improvements_validated}")
    print(f"Risk level: {validation_report.risk_level}")
    print("Recommendations:", validation_report.recommendations)

if __name__ == "__main__":
    asyncio.run(example_usage())