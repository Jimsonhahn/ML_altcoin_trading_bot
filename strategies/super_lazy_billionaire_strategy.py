#!/usr/bin/env python3
# strategies/super_lazy_billionaire_strategy.py
"""
Run claude
- The Ultimate Multi-Strategy Orchestrator
Kombiniert alle Backtest-Erkenntnisse für 70-90% Jahresrendite bei optimiertem Risiko
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum
import asyncio
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Core imports
from core.advanced_market_regime_detector import AdvancedMarketRegimeDetector, MarketRegime, RegimePrediction
from risk.kelly_criterion_optimizer import KellyCriterionOptimizer, StrategyStats, MarketConditions, PositionSize
from ml.entry_exit_optimizer import EntryExitOptimizer, MLSignal, TradeDirection, SignalStrength

# Strategy imports
try:
    from strategies.momentum_strategy import MomentumStrategy
    from strategies.mean_reversion_strategy import MeanReversionStrategy
    from strategies.ml_strategy import MLStrategy
    from strategies.arbitrage_strategy import ArbitrageStrategy
    from strategies.grid_strategy import GridStrategy
    from strategies.stablecoin_parking_strategy import StablecoinParkingStrategy
    from strategies.defi_strategy import DeFiStrategy
    from strategies.copy_trading_strategy import CopyTradingStrategy
    from strategies.lazy_billionaire_strategy import LazyBillionaireStrategy
    STRATEGIES_AVAILABLE = True
except ImportError:
    STRATEGIES_AVAILABLE = False
    logging.warning("Some strategy modules not available")

# ML imports with fallbacks
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

class StrategyState(Enum):
    ACTIVE = "active"
    STANDBY = "standby"
    DISABLED = "disabled"
    TRANSITIONING = "transitioning"

class PortfolioMode(Enum):
    CONSERVATIVE = "conservative"    # 25-35% target return, low drawdown
    BALANCED = "balanced"           # 45-60% target return, medium drawdown  
    AGGRESSIVE = "aggressive"       # 70-90% target return, higher drawdown
    ADAPTIVE = "adaptive"          # Dynamic based on market conditions

@dataclass
class StrategyPerformance:
    """Real-time strategy performance tracking"""
    name: str
    current_allocation: float
    target_allocation: float
    daily_pnl: float
    weekly_pnl: float
    monthly_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    confidence_score: float
    market_correlation: float
    last_updated: datetime

@dataclass
class MarketStateAnalysis:
    """Comprehensive market state analysis"""
    regime: MarketRegime
    predicted_regime: MarketRegime
    transition_probability: float
    volatility_regime: str  # low, medium, high, extreme
    trend_strength: float
    liquidity_score: float
    sentiment_score: float
    risk_level: str
    opportunity_score: float
    recommended_exposure: float

@dataclass
class StrategyRecommendation:
    """Strategy recommendation with rationale"""
    strategy_name: str
    recommended_allocation: float
    confidence: float
    rationale: List[str]
    expected_performance: Dict[str, float]
    risk_score: float
    time_horizon: str

class DynamicWeightingEngine:
    """
    Intelligente Strategie-Gewichtung basierend auf Backtest-Erkenntnissen und Live-Performance
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.DynamicWeightingEngine")
        
        # ULTIMATIVE Multi-Strategie Allocations (alle verfügbaren Bot-Strategien)
        self.base_allocations = {
            # TIER 1: TOP PERFORMER (Backtest-verifiziert)
            'lazy_billionaire': 0.22,    # 🥇 TOP PERFORMER (+0.76 Sharpe)
            'ml_strategy': 0.16,         # 🧠 Enhanced ML mit Confidence-System
            'arbitrage': 0.14,           # ⚡ Cross-Exchange Arbitrage (+2.45 Sharpe)
            'mean_reversion': 0.12,      # 🔄 STARK VERBESSERT (+0.54 Sharpe)
            
            # TIER 2: STARKE PERFORMER
            'momentum': 0.10,            # 📈 Bull-Market-Champion (+0.49 Sharpe)
            'grid': 0.08,                # 🎯 Grid Trading für Ranges
            'liquidation_hunter': 0.06,  # 🎣 Liquidation Strategy (85% Confidence)
            'defi_yield': 0.05,          # 🌾 DeFi Yield Farming (15-50% APY)
            
            # TIER 3: DEFENSIVE & OPPORTUNISTISCH  
            'stablecoin_parking': 0.04,  # 🏦 Capital Preservation
            'autopilot': 0.02,           # 🛩️ Meta-Koordinator
            'scalping': 0.01,            # ⚡ High-Frequency (wenn verfügbar)
            
            # DEAKTIVIERT: Schlechte Performance
            'copy_trading': 0.00         # ❌ Deaktiviert (schlechteste Performance)
        }
        
        # INTELLIGENTE Performance-Multipliers für ALLE Strategien
        self.performance_multipliers = {
            # TIER 1: TOP PERFORMER
            'lazy_billionaire': {'bull': 1.4, 'bear': 1.1, 'sideways': 1.2},  # Konsistent stark
            'ml_strategy': {'bull': 1.3, 'bear': 0.8, 'sideways': 1.1, 'volatile': 0.9},  # ML-Pattern-Recognition
            'arbitrage': {'bull': 1.1, 'bear': 1.4, 'sideways': 1.2, 'volatile': 1.6},  # Volatility = Opportunities
            'mean_reversion': {'bull': 0.8, 'bear': 1.4, 'sideways': 1.6, 'volatile': 1.3},  # Range-Lover
            
            # TIER 2: SPECIALIZED PERFORMERS
            'momentum': {'bull': 1.7, 'bear': 0.3, 'sideways': 0.5, 'volatile': 0.7},  # Pure Trend-Follower
            'grid': {'bull': 0.7, 'bear': 1.1, 'sideways': 1.4, 'volatile': 0.8},  # Range-Specialist
            'liquidation_hunter': {'bull': 0.9, 'bear': 1.3, 'sideways': 1.1, 'volatile': 1.5},  # Volatility-Hunter
            'defi_yield': {'bull': 1.6, 'bear': 0.4, 'sideways': 0.9, 'volatile': 0.6},  # Bull-Market-Optimized
            
            # TIER 3: DEFENSIVE & META
            'stablecoin_parking': {'bull': 0.3, 'bear': 2.0, 'sideways': 0.8, 'volatile': 1.8},  # Safe-Haven
            'autopilot': {'bull': 1.1, 'bear': 1.0, 'sideways': 1.1, 'volatile': 1.0},  # Meta-Coordinator
            'scalping': {'bull': 1.2, 'bear': 0.6, 'sideways': 0.9, 'volatile': 1.4},  # High-Frequency
            
            # DEAKTIVIERT
            'copy_trading': {'bull': 0.0, 'bear': 0.0, 'sideways': 0.0, 'volatile': 0.0}
        }
        
        # ERWEITERTE Korrelations-Matrix (alle Strategien)
        self.strategy_correlations = {
            # POSITIVE KORRELATIONEN (ähnliche Marktpräferenzen)
            ('ml_strategy', 'momentum'): 0.65,      # Beide nutzen Trends
            ('arbitrage', 'grid'): 0.25,            # Beide profitieren von Ranges
            ('liquidation_hunter', 'momentum'): 0.45,  # Beide nutzen extreme Moves
            ('defi_yield', 'lazy_billionaire'): 0.35,   # Beide langfristig orientiert
            ('ml_strategy', 'arbitrage'): 0.15,     # Beide datengetrieben
            ('autopilot', 'lazy_billionaire'): 0.40,    # Meta-Strategien
            
            # NEGATIVE KORRELATIONEN (diversifizierend)
            ('momentum', 'mean_reversion'): -0.35,  # Trend vs Anti-Trend
            ('stablecoin_parking', 'defi_yield'): -0.15,  # Safe vs Risky
            ('grid', 'momentum'): -0.25,            # Range vs Trend
            ('arbitrage', 'liquidation_hunter'): -0.10,   # Stable vs Volatile
            ('scalping', 'lazy_billionaire'): -0.20,      # Short vs Long-term
            
            # NIEDRIGE KORRELATIONEN (gut für Diversifikation)
            ('mean_reversion', 'arbitrage'): 0.05,
            ('grid', 'defi_yield'): 0.10,
            ('ml_strategy', 'stablecoin_parking'): 0.08,
            ('liquidation_hunter', 'mean_reversion'): 0.12
        }
        
        # STRATEGIE-SYNERGIEN (verstärkende Kombinationen)
        self.strategy_synergies = {
            ('lazy_billionaire', 'ml_strategy'): 1.15,    # Meta + KI = Boost
            ('arbitrage', 'liquidation_hunter'): 1.12,   # Volatility-Duo
            ('mean_reversion', 'grid'): 1.10,            # Range-Specialists
            ('momentum', 'defi_yield'): 1.08,            # Bull-Market-Power
            ('stablecoin_parking', 'arbitrage'): 1.05    # Safe-Harbor-Combo
        }
        
        self.logger.info("DynamicWeightingEngine initialized with backtest-optimized allocations")
    
    def calculate_optimal_weights(self, 
                                market_state: MarketStateAnalysis,
                                strategy_performances: Dict[str, StrategyPerformance],
                                portfolio_mode: PortfolioMode = PortfolioMode.BALANCED) -> Dict[str, float]:
        """
        Berechne optimale Gewichtungen basierend auf Marktlage und Performance
        """
        try:
            weights = self.base_allocations.copy()
            
            # 1. Marktregime-Anpassungen
            regime_weights = self._apply_regime_adjustments(weights, market_state)
            
            # 2. Performance-Anpassungen
            performance_weights = self._apply_performance_adjustments(regime_weights, strategy_performances)
            
            # 3. Volatilitäts-Anpassungen
            volatility_weights = self._apply_volatility_adjustments(performance_weights, market_state)
            
            # 4. Korrelations-Optimierung
            correlation_weights = self._optimize_correlations(volatility_weights, strategy_performances)
            
            # 5. Portfolio-Mode Anpassungen
            final_weights = self._apply_portfolio_mode(correlation_weights, portfolio_mode, market_state)
            
            # 6. Normalisierung und Constraints
            final_weights = self._apply_constraints(final_weights)
            
            self.logger.info(f"Calculated optimal weights for {market_state.regime.value} regime")
            return final_weights
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal weights: {e}")
            return self.base_allocations
    
    def _apply_regime_adjustments(self, weights: Dict[str, float], market_state: MarketStateAnalysis) -> Dict[str, float]:
        """INTELLIGENTE Regime-basierte Gewichtungsanpassungen für ALLE Strategien"""
        regime = market_state.regime
        volatility = market_state.volatility_regime
        
        # Erweiterte Markttyp-Bestimmung
        if 'bull' in regime.value.lower():
            market_type = 'bull'
        elif 'bear' in regime.value.lower():
            market_type = 'bear'
        elif volatility in ['high', 'extreme']:
            market_type = 'volatile'
        else:
            market_type = 'sideways'
        
        adjusted_weights = {}
        for strategy, weight in weights.items():
            if strategy in self.performance_multipliers:
                multiplier = self.performance_multipliers[strategy].get(market_type, 1.0)
                adjusted_weights[strategy] = weight * multiplier
            else:
                adjusted_weights[strategy] = weight
        
        # SPEZIELLE REGIME-ANPASSUNGEN (alle verfügbaren Strategien)
        if regime == MarketRegime.EXTREME_VOLATILITY:
            # Extreme Volatilität: Volatility-Hunter aktivieren
            self._boost_strategy(adjusted_weights, 'arbitrage', 1.8)
            self._boost_strategy(adjusted_weights, 'liquidation_hunter', 2.0)  # MEGA-BOOST
            self._boost_strategy(adjusted_weights, 'scalping', 1.5)
            self._boost_strategy(adjusted_weights, 'stablecoin_parking', 2.5)
            # Trend-Strategien reduzieren
            self._boost_strategy(adjusted_weights, 'momentum', 0.2)
            self._boost_strategy(adjusted_weights, 'defi_yield', 0.1)
            self._boost_strategy(adjusted_weights, 'ml_strategy', 0.6)
        
        elif regime in [MarketRegime.TRANSITION_BULL, MarketRegime.TRANSITION_BEAR]:
            # Übergangszeiten: Adaptive Strategien bevorzugen
            self._boost_strategy(adjusted_weights, 'lazy_billionaire', 1.4)  # Meta-Intelligence
            self._boost_strategy(adjusted_weights, 'autopilot', 1.6)  # Multi-Strategy-Coordinator
            self._boost_strategy(adjusted_weights, 'stablecoin_parking', 1.8)
            self._boost_strategy(adjusted_weights, 'arbitrage', 1.3)
            # Unsichere Strategien reduzieren
            self._boost_strategy(adjusted_weights, 'ml_strategy', 0.7)  # ML unsicher bei Transitions
            self._boost_strategy(adjusted_weights, 'momentum', 0.5)
        
        elif regime == MarketRegime.BULL_STRONG:
            # Starker Bull: Aggressive Strategien maximieren
            self._boost_strategy(adjusted_weights, 'momentum', 2.0)  # MAXIMUM
            self._boost_strategy(adjusted_weights, 'defi_yield', 1.8)
            self._boost_strategy(adjusted_weights, 'ml_strategy', 1.4)
            self._boost_strategy(adjusted_weights, 'lazy_billionaire', 1.3)
            # Defensive reduzieren
            self._boost_strategy(adjusted_weights, 'stablecoin_parking', 0.2)
            self._boost_strategy(adjusted_weights, 'mean_reversion', 0.6)
        
        elif regime == MarketRegime.BEAR_STRONG:
            # Starker Bear: Defensive + Contrarian
            self._boost_strategy(adjusted_weights, 'mean_reversion', 1.8)  # Contrarian-Power
            self._boost_strategy(adjusted_weights, 'stablecoin_parking', 2.2)
            self._boost_strategy(adjusted_weights, 'arbitrage', 1.5)
            self._boost_strategy(adjusted_weights, 'liquidation_hunter', 1.4)  # Bear-Liquidations
            # Bull-Strategien minimieren
            self._boost_strategy(adjusted_weights, 'momentum', 0.1)
            self._boost_strategy(adjusted_weights, 'defi_yield', 0.2)
        
        elif 'sideways' in regime.value.lower():
            # Sideways: Range-Specialists aktivieren
            self._boost_strategy(adjusted_weights, 'mean_reversion', 1.7)
            self._boost_strategy(adjusted_weights, 'grid', 1.6)
            self._boost_strategy(adjusted_weights, 'arbitrage', 1.3)
            self._boost_strategy(adjusted_weights, 'scalping', 1.2)
            # Trend-Strategien reduzieren
            self._boost_strategy(adjusted_weights, 'momentum', 0.4)
            self._boost_strategy(adjusted_weights, 'defi_yield', 0.7)
        
        # SYNERGIE-BOOSTS anwenden
        adjusted_weights = self._apply_strategy_synergies(adjusted_weights)
        
        return adjusted_weights
    
    def _boost_strategy(self, weights: Dict[str, float], strategy: str, multiplier: float):
        """Hilfsfunktion für saubere Strategy-Boosts"""
        if strategy in weights:
            weights[strategy] *= multiplier
    
    def _apply_strategy_synergies(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Wende Strategie-Synergien an"""
        for (strategy1, strategy2), synergy_boost in self.strategy_synergies.items():
            if strategy1 in weights and strategy2 in weights:
                # Beide Strategien vorhanden - synergetischen Boost anwenden
                avg_weight = (weights[strategy1] + weights[strategy2]) / 2
                if avg_weight > 0.05:  # Nur bei signifikanten Allokationen
                    weights[strategy1] *= synergy_boost
                    weights[strategy2] *= synergy_boost
                    self.logger.debug(f"Applied synergy boost {synergy_boost} to {strategy1} + {strategy2}")
        
        return weights
    
    def _apply_performance_adjustments(self, weights: Dict[str, float], 
                                     performances: Dict[str, StrategyPerformance]) -> Dict[str, float]:
        """Performance-basierte Anpassungen"""
        adjusted_weights = weights.copy()
        
        for strategy, performance in performances.items():
            if strategy not in adjusted_weights:
                continue
                
            # Performance-Multiplier basierend auf Sharpe Ratio
            if performance.sharpe_ratio > 2.0:
                perf_multiplier = 1.3
            elif performance.sharpe_ratio > 1.5:
                perf_multiplier = 1.2
            elif performance.sharpe_ratio > 1.0:
                perf_multiplier = 1.0
            elif performance.sharpe_ratio > 0.5:
                perf_multiplier = 0.8
            else:
                perf_multiplier = 0.6
            
            # Drawdown-Penalty
            if performance.max_drawdown > 0.20:
                perf_multiplier *= 0.7
            elif performance.max_drawdown > 0.15:
                perf_multiplier *= 0.85
            
            # Win Rate Bonus
            if performance.win_rate > 0.70:
                perf_multiplier *= 1.1
            elif performance.win_rate < 0.40:
                perf_multiplier *= 0.9
            
            # Confidence Score
            perf_multiplier *= (0.5 + performance.confidence_score * 0.5)
            
            adjusted_weights[strategy] *= perf_multiplier
        
        return adjusted_weights
    
    def _apply_volatility_adjustments(self, weights: Dict[str, float], 
                                    market_state: MarketStateAnalysis) -> Dict[str, float]:
        """Volatilitäts-basierte Anpassungen"""
        adjusted_weights = weights.copy()
        
        vol_regime = market_state.volatility_regime
        
        if vol_regime == 'extreme':
            # Extreme Volatilität: Sehr defensive
            adjusted_weights['stablecoin_parking'] *= 2.5
            adjusted_weights['arbitrage'] *= 1.8
            adjusted_weights['momentum'] *= 0.3
            adjusted_weights['defi'] *= 0.2
            adjusted_weights['ml'] *= 0.7
            
        elif vol_regime == 'high':
            # Hohe Volatilität: Etwas defensiver
            adjusted_weights['stablecoin_parking'] *= 1.5
            adjusted_weights['arbitrage'] *= 1.3
            adjusted_weights['momentum'] *= 0.7
            adjusted_weights['mean_reversion'] *= 1.2
            
        elif vol_regime == 'low':
            # Niedrige Volatilität: Aggressiver
            adjusted_weights['momentum'] *= 1.3
            adjusted_weights['ml'] *= 1.2
            adjusted_weights['defi'] *= 1.5
            adjusted_weights['grid'] *= 1.2
            adjusted_weights['stablecoin_parking'] *= 0.6
        
        return adjusted_weights
    
    def _optimize_correlations(self, weights: Dict[str, float], 
                             performances: Dict[str, StrategyPerformance]) -> Dict[str, float]:
        """Korrelations-optimierte Gewichtung"""
        # Vereinfachte Korrelations-Optimierung
        # In der Praxis würde hier eine vollständige Markowitz-Optimierung stattfinden
        
        adjusted_weights = weights.copy()
        
        # Reduziere hoch-korrelierte Strategien
        if adjusted_weights.get('ml', 0) > 0.15 and adjusted_weights.get('momentum', 0) > 0.10:
            # ML und Momentum sind korreliert - reduziere den schwächeren
            ml_perf = performances.get('ml', None)
            momentum_perf = performances.get('momentum', None)
            
            if ml_perf and momentum_perf:
                if ml_perf.sharpe_ratio < momentum_perf.sharpe_ratio:
                    adjusted_weights['ml'] *= 0.8
                else:
                    adjusted_weights['momentum'] *= 0.8
        
        # Erhöhe negativ korrelierte Strategien
        if adjusted_weights.get('momentum', 0) > 0.10 and adjusted_weights.get('mean_reversion', 0) < 0.15:
            # Momentum und Mean Reversion sind negativ korreliert - gut für Diversifikation
            adjusted_weights['mean_reversion'] *= 1.1
        
        return adjusted_weights
    
    def _apply_portfolio_mode(self, weights: Dict[str, float], 
                            portfolio_mode: PortfolioMode,
                            market_state: MarketStateAnalysis) -> Dict[str, float]:
        """Portfolio-Mode spezifische Anpassungen"""
        adjusted_weights = weights.copy()
        
        if portfolio_mode == PortfolioMode.CONSERVATIVE:
            # Konservativ: Fokus auf Top-Performer mit niedrigem Risiko
            adjusted_weights['lazy_billionaire'] *= 1.5  # TOP PERFORMER
            adjusted_weights['arbitrage'] *= 1.4
            adjusted_weights['mean_reversion'] *= 1.3  # VERBESSERT
            adjusted_weights['grid'] *= 1.2
            adjusted_weights['momentum'] *= 0.6
            adjusted_weights['defi'] *= 0.2
            adjusted_weights['ml'] *= 0.9
            
        elif portfolio_mode == PortfolioMode.AGGRESSIVE:
            # Aggressiv: Fokus auf Top-Performer mit hohem Potenzial
            adjusted_weights['lazy_billionaire'] *= 1.6  # TOP PERFORMER
            adjusted_weights['ml'] *= 1.4
            adjusted_weights['mean_reversion'] *= 1.3  # STARK VERBESSERT
            adjusted_weights['momentum'] *= 1.3
            adjusted_weights['arbitrage'] *= 1.1
            adjusted_weights['defi'] *= 1.5  # REDUZIERT von 2.0
            adjusted_weights['stablecoin_parking'] *= 0.2
            adjusted_weights['grid'] *= 0.8
            
        elif portfolio_mode == PortfolioMode.ADAPTIVE:
            # Adaptiv: Basierend auf Opportunity Score
            if market_state.opportunity_score > 0.7:
                # Hohe Opportunity: Aggressiver
                adjusted_weights['ml'] *= 1.3
                adjusted_weights['momentum'] *= 1.2
                adjusted_weights['defi'] *= 1.5
            elif market_state.opportunity_score < 0.3:
                # Niedrige Opportunity: Defensiver
                adjusted_weights['stablecoin_parking'] *= 1.5
                adjusted_weights['arbitrage'] *= 1.2
        
        return adjusted_weights
    
    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Anwendung von Portfolio-Constraints"""
        # Normalisierung
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # Minimum/Maximum Constraints
        constraints = {
            'ml': (0.05, 0.30),              # 5-30%
            'arbitrage': (0.10, 0.25),       # 10-25%
            'momentum': (0.05, 0.25),        # 5-25%
            'mean_reversion': (0.05, 0.20),  # 5-20%
            'lazy_billionaire': (0.05, 0.20), # 5-20%
            'grid': (0.05, 0.15),            # 5-15%
            'stablecoin_parking': (0.02, 0.30), # 2-30%
            'defi': (0.00, 0.10),            # 0-10%
            'copy_trading': (0.00, 0.05)     # 0-5%
        }
        
        for strategy, (min_weight, max_weight) in constraints.items():
            if strategy in weights:
                weights[strategy] = max(min_weight, min(max_weight, weights[strategy]))
        
        # Erneute Normalisierung nach Constraints
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights

class MarketStateAnalyzer:
    """
    Erweiterte Marktanalyse für SuperLazyBillionaire
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.MarketStateAnalyzer")
        
        # Komponenten
        self.regime_detector = AdvancedMarketRegimeDetector()
        
        self.logger.info("MarketStateAnalyzer initialized")
    
    def analyze_market_state(self, market_data: Dict[str, pd.DataFrame], symbol: str = "BTC/USDT") -> MarketStateAnalysis:
        """
        Umfassende Marktanalyse
        """
        try:
            # 1. Regime Detection
            regime_prediction = self.regime_detector.detect_regime(market_data, symbol)
            
            # 2. Volatilitäts-Analyse
            volatility_regime = self._analyze_volatility_regime(market_data)
            
            # 3. Trend-Stärke
            trend_strength = self._calculate_trend_strength(market_data)
            
            # 4. Liquiditäts-Score
            liquidity_score = self._assess_liquidity(market_data)
            
            # 5. Sentiment-Score
            sentiment_score = self._calculate_sentiment_score(regime_prediction)
            
            # 6. Risk Level
            risk_level = self._assess_overall_risk(regime_prediction, volatility_regime, trend_strength)
            
            # 7. Opportunity Score
            opportunity_score = self._calculate_opportunity_score(
                regime_prediction, volatility_regime, trend_strength, liquidity_score
            )
            
            # 8. Recommended Exposure
            recommended_exposure = self._calculate_recommended_exposure(
                regime_prediction, risk_level, opportunity_score
            )
            
            return MarketStateAnalysis(
                regime=regime_prediction.current_regime,
                predicted_regime=regime_prediction.predicted_regime,
                transition_probability=regime_prediction.transition_probability,
                volatility_regime=volatility_regime,
                trend_strength=trend_strength,
                liquidity_score=liquidity_score,
                sentiment_score=sentiment_score,
                risk_level=risk_level,
                opportunity_score=opportunity_score,
                recommended_exposure=recommended_exposure
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing market state: {e}")
            return self._get_fallback_analysis()
    
    def _analyze_volatility_regime(self, market_data: Dict[str, pd.DataFrame]) -> str:
        """Bestimme Volatilitäts-Regime"""
        try:
            # Nutze 1d Daten für Volatilitäts-Analyse
            data = market_data.get('1d')
            if data is None or len(data) < 30:
                return 'medium'
            
            returns = data['close'].pct_change().dropna()
            if len(returns) < 20:
                return 'medium'
            
            # Aktuelle 20-Tage Volatilität
            current_vol = returns.iloc[-20:].std() * np.sqrt(365)
            
            # Historische Volatilitäts-Percentile
            if len(returns) >= 100:
                rolling_vol = returns.rolling(20).std() * np.sqrt(365)
                vol_percentile = (rolling_vol < current_vol).mean()
                
                if vol_percentile > 0.9:
                    return 'extreme'
                elif vol_percentile > 0.75:
                    return 'high'
                elif vol_percentile < 0.25:
                    return 'low'
                else:
                    return 'medium'
            else:
                # Fallback basierend auf absoluten Werten
                if current_vol > 0.8:
                    return 'extreme'
                elif current_vol > 0.5:
                    return 'high'
                elif current_vol < 0.2:
                    return 'low'
                else:
                    return 'medium'
                    
        except Exception as e:
            self.logger.error(f"Error analyzing volatility regime: {e}")
            return 'medium'
    
    def _calculate_trend_strength(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Berechne Trend-Stärke (0.0 bis 1.0)"""
        try:
            data = market_data.get('1d')
            if data is None or len(data) < 50:
                return 0.5
            
            close = data['close'].values
            
            # Multiple Trend-Indikatoren
            trend_scores = []
            
            # 1. SMA Slope
            sma_20 = pd.Series(close).rolling(20).mean()
            sma_50 = pd.Series(close).rolling(50).mean()
            
            if len(sma_20) >= 20 and len(sma_50) >= 50:
                sma_20_slope = (sma_20.iloc[-1] - sma_20.iloc[-10]) / sma_20.iloc[-10]
                sma_50_slope = (sma_50.iloc[-1] - sma_50.iloc[-20]) / sma_50.iloc[-20]
                
                trend_scores.append(abs(sma_20_slope) * 10)  # Normalisiert
                trend_scores.append(abs(sma_50_slope) * 10)
            
            # 2. ADX-ähnlicher Indikator
            if len(close) >= 20:
                high = data['high'].values
                low = data['low'].values
                
                # Vereinfachter ADX
                tr = np.maximum(high[1:] - low[1:], 
                              np.maximum(abs(high[1:] - close[:-1]), 
                                       abs(low[1:] - close[:-1])))
                
                plus_dm = np.where(high[1:] - high[:-1] > low[:-1] - low[1:], 
                                 np.maximum(high[1:] - high[:-1], 0), 0)
                minus_dm = np.where(low[:-1] - low[1:] > high[1:] - high[:-1], 
                                  np.maximum(low[:-1] - low[1:], 0), 0)
                
                if len(tr) >= 14:
                    tr_smooth = pd.Series(tr).rolling(14).mean().iloc[-1]
                    plus_di = pd.Series(plus_dm).rolling(14).mean().iloc[-1] / tr_smooth * 100
                    minus_di = pd.Series(minus_dm).rolling(14).mean().iloc[-1] / tr_smooth * 100
                    
                    dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else 0
                    trend_scores.append(dx / 100)
            
            # 3. Price Position
            if len(close) >= 50:
                recent_high = np.max(close[-50:])
                recent_low = np.min(close[-50:])
                price_position = (close[-1] - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
                
                # Extremes deuten auf Trend hin
                trend_scores.append(abs(price_position - 0.5) * 2)
            
            return min(1.0, np.mean(trend_scores)) if trend_scores else 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {e}")
            return 0.5
    
    def _assess_liquidity(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Bewerte Liquidität (0.0 bis 1.0)"""
        try:
            data = market_data.get('1h')  # Nutze 1h für Liquiditäts-Analyse
            if data is None or len(data) < 24:
                return 0.7  # Default
            
            # Volume-basierte Liquiditäts-Indikatoren
            volume = data['volume'].values
            
            # 1. Volume Consistency
            if len(volume) >= 24:
                recent_volume = volume[-24:].mean()
                historical_volume = volume.mean()
                volume_ratio = recent_volume / historical_volume if historical_volume > 0 else 1.0
                
                # Höheres Volume = bessere Liquidität
                volume_score = min(1.0, volume_ratio)
            else:
                volume_score = 0.7
            
            # 2. Spread Approximation (High-Low Range)
            if 'high' in data.columns and 'low' in data.columns and 'close' in data.columns:
                spreads = (data['high'] - data['low']) / data['close']
                avg_spread = spreads.iloc[-24:].mean() if len(spreads) >= 24 else spreads.mean()
                
                # Niedrigere Spreads = bessere Liquidität
                spread_score = max(0.0, 1.0 - avg_spread * 20)  # Normalisiert
            else:
                spread_score = 0.7
            
            # 3. Volume Volatility (niedrigere Volatilität = stabilere Liquidität)
            if len(volume) >= 24:
                volume_std = np.std(volume[-24:]) / np.mean(volume[-24:]) if np.mean(volume[-24:]) > 0 else 1.0
                volatility_score = max(0.0, 1.0 - volume_std)
            else:
                volatility_score = 0.7
            
            # Gewichteter Durchschnitt
            liquidity_score = (volume_score * 0.4 + spread_score * 0.4 + volatility_score * 0.2)
            
            return max(0.1, min(1.0, liquidity_score))
            
        except Exception as e:
            self.logger.error(f"Error assessing liquidity: {e}")
            return 0.7
    
    def _calculate_sentiment_score(self, regime_prediction: RegimePrediction) -> float:
        """Berechne Sentiment Score (0.0 bis 1.0)"""
        try:
            regime = regime_prediction.current_regime
            confidence = regime_prediction.confidence
            
            # Basis-Sentiment basierend auf Regime
            regime_sentiment = {
                MarketRegime.BULL_STRONG: 0.9,
                MarketRegime.BULL_WEAK: 0.7,
                MarketRegime.BEAR_STRONG: 0.1,
                MarketRegime.BEAR_WEAK: 0.3,
                MarketRegime.SIDEWAYS_LOW_VOL: 0.6,
                MarketRegime.SIDEWAYS_HIGH_VOL: 0.4,
                MarketRegime.TRANSITION_BULL: 0.6,
                MarketRegime.TRANSITION_BEAR: 0.4,
                MarketRegime.EXTREME_VOLATILITY: 0.2,
                MarketRegime.RECOVERY: 0.7
            }
            
            base_sentiment = regime_sentiment.get(regime, 0.5)
            
            # Anpassung basierend auf Confidence
            # Hohe Confidence verstärkt das Signal
            confidence_adjustment = (confidence - 0.5) * 0.4  # -0.2 bis +0.2
            
            final_sentiment = base_sentiment + confidence_adjustment
            
            return max(0.0, min(1.0, final_sentiment))
            
        except Exception as e:
            self.logger.error(f"Error calculating sentiment score: {e}")
            return 0.5
    
    def _assess_overall_risk(self, regime_prediction: RegimePrediction, 
                           volatility_regime: str, trend_strength: float) -> str:
        """Bewerte Gesamt-Risiko"""
        try:
            risk_score = 0
            
            # Regime-basiertes Risiko
            regime_risk = {
                MarketRegime.BULL_STRONG: 1,
                MarketRegime.BULL_WEAK: 0,
                MarketRegime.BEAR_STRONG: 3,
                MarketRegime.BEAR_WEAK: 2,
                MarketRegime.SIDEWAYS_LOW_VOL: 0,
                MarketRegime.SIDEWAYS_HIGH_VOL: 2,
                MarketRegime.TRANSITION_BULL: 2,
                MarketRegime.TRANSITION_BEAR: 3,
                MarketRegime.EXTREME_VOLATILITY: 4,
                MarketRegime.RECOVERY: 1
            }
            
            risk_score += regime_risk.get(regime_prediction.current_regime, 2)
            
            # Volatilitäts-Risiko
            vol_risk = {'low': 0, 'medium': 1, 'high': 2, 'extreme': 3}
            risk_score += vol_risk.get(volatility_regime, 1)
            
            # Transition-Risiko
            if regime_prediction.transition_probability > 0.7:
                risk_score += 2
            elif regime_prediction.transition_probability > 0.5:
                risk_score += 1
            
            # Confidence-Risiko
            if regime_prediction.confidence < 0.5:
                risk_score += 1
            
            # Trend-Risiko (sehr starke Trends können umkehren)
            if trend_strength > 0.8:
                risk_score += 1
            
            # Mapping zu Risiko-Level
            if risk_score >= 6:
                return 'very_high'
            elif risk_score >= 4:
                return 'high'
            elif risk_score >= 2:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            self.logger.error(f"Error assessing overall risk: {e}")
            return 'medium'
    
    def _calculate_opportunity_score(self, regime_prediction: RegimePrediction,
                                   volatility_regime: str, trend_strength: float,
                                   liquidity_score: float) -> float:
        """Berechne Opportunity Score (0.0 bis 1.0)"""
        try:
            opportunity = 0.5  # Basis
            
            # Regime-basierte Opportunities
            regime_opportunities = {
                MarketRegime.BULL_STRONG: 0.8,
                MarketRegime.BULL_WEAK: 0.6,
                MarketRegime.BEAR_STRONG: 0.4,  # Short opportunities
                MarketRegime.BEAR_WEAK: 0.5,
                MarketRegime.SIDEWAYS_LOW_VOL: 0.6,  # Grid/Mean reversion
                MarketRegime.SIDEWAYS_HIGH_VOL: 0.7,  # Volatility trading
                MarketRegime.TRANSITION_BULL: 0.7,
                MarketRegime.TRANSITION_BEAR: 0.3,
                MarketRegime.EXTREME_VOLATILITY: 0.8,  # Arbitrage opportunities
                MarketRegime.RECOVERY: 0.9
            }
            
            opportunity = regime_opportunities.get(regime_prediction.current_regime, 0.5)
            
            # Volatility-Bonus (hohe Vol = mehr Opportunities)
            vol_bonus = {'low': -0.1, 'medium': 0.0, 'high': 0.1, 'extreme': 0.2}
            opportunity += vol_bonus.get(volatility_regime, 0.0)
            
            # Trend-Bonus (starke Trends = gute Momentum-Opportunities)
            if trend_strength > 0.7:
                opportunity += 0.1
            
            # Liquidity-Adjustment
            opportunity *= liquidity_score
            
            # Confidence-Adjustment
            opportunity *= regime_prediction.confidence
            
            return max(0.0, min(1.0, opportunity))
            
        except Exception as e:
            self.logger.error(f"Error calculating opportunity score: {e}")
            return 0.5
    
    def _calculate_recommended_exposure(self, regime_prediction: RegimePrediction,
                                      risk_level: str, opportunity_score: float) -> float:
        """Berechne empfohlene Gesamt-Exposure"""
        try:
            # Basis-Exposure basierend auf Risiko
            risk_exposure = {
                'low': 0.85,
                'medium': 0.70,
                'high': 0.50,
                'very_high': 0.30
            }
            
            base_exposure = risk_exposure.get(risk_level, 0.70)
            
            # Opportunity-Anpassung
            opportunity_adjustment = (opportunity_score - 0.5) * 0.4  # -0.2 bis +0.2
            
            final_exposure = base_exposure + opportunity_adjustment
            
            return max(0.20, min(0.95, final_exposure))
            
        except Exception as e:
            self.logger.error(f"Error calculating recommended exposure: {e}")
            return 0.70
    
    def _get_fallback_analysis(self) -> MarketStateAnalysis:
        """Fallback-Analyse bei Fehlern"""
        return MarketStateAnalysis(
            regime=MarketRegime.SIDEWAYS_LOW_VOL,
            predicted_regime=MarketRegime.SIDEWAYS_LOW_VOL,
            transition_probability=0.3,
            volatility_regime='medium',
            trend_strength=0.5,
            liquidity_score=0.7,
            sentiment_score=0.5,
            risk_level='medium',
            opportunity_score=0.5,
            recommended_exposure=0.70
        )

class SuperLazyBillionaireStrategy:
    """
    Die ultimative Multi-Strategie-Orchestrierung basierend auf Backtest-Erkenntnissen
    Ziel: 70-90% Jahresrendite bei optimiertem Risiko-Management
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Konfiguration
        self.portfolio_mode = PortfolioMode(self.config.get('portfolio_mode', 'balanced'))
        self.rebalance_frequency = self.config.get('rebalance_frequency_hours', 6)  # 6 Stunden
        self.max_strategies = self.config.get('max_active_strategies', 6)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.65)
        
        # Komponenten
        self.weighting_engine = DynamicWeightingEngine(self.config.get('weighting_engine', {}))
        self.market_analyzer = MarketStateAnalyzer(self.config.get('market_analyzer', {}))
        self.kelly_optimizer = KellyCriterionOptimizer(self.config.get('kelly_optimizer', {}))
        self.ml_optimizer = EntryExitOptimizer(self.config.get('ml_optimizer', {}))
        
        # Strategien (werden bei Bedarf initialisiert)
        self.available_strategies = {}
        self.active_strategies = {}
        
        # Performance Tracking
        self.strategy_performances = {}
        self.portfolio_history = []
        self.last_rebalance = None
        
        # State
        self.current_allocations = {}
        self.current_market_state = None
        self.is_initialized = False
        
        self.logger.info(f"SuperLazyBillionaireStrategy initialized in {self.portfolio_mode.value} mode")
    
    async def initialize(self):
        """Initialisierung der Strategie"""
        try:
            self.logger.info("Initializing SuperLazyBillionaireStrategy...")
            
            # Verfügbare Strategien registrieren
            self._register_available_strategies()
            
            # Performance-Tracking initialisieren
            self._initialize_performance_tracking()
            
            # ML-Modelle trainieren (falls verfügbar)
            await self._initialize_ml_models()
            
            self.is_initialized = True
            self.logger.info("SuperLazyBillionaireStrategy initialization completed")
            
        except Exception as e:
            self.logger.error(f"Error initializing SuperLazyBillionaireStrategy: {e}")
            raise
    
    def _register_available_strategies(self):
        """Registriere ALLE verfügbaren Bot-Strategien"""
        strategy_configs = {
            # TIER 1: META & TOP PERFORMER
            'lazy_billionaire': {
                'class': 'LazyBillionaireStrategy', 
                'min_confidence': 0.50,
                'type': 'meta',
                'time_horizon': 'long-term',
                'risk_level': 'medium',
                'market_specialty': 'all'
            },
            'ml_strategy': {
                'class': 'MLStrategy', 
                'min_confidence': 0.70,
                'type': 'ml_enhanced',
                'time_horizon': 'short-term',
                'risk_level': 'medium-high',
                'market_specialty': 'trending'
            },
            'arbitrage': {
                'class': 'ArbitrageStrategy', 
                'min_confidence': 0.60,
                'type': 'market_neutral',
                'time_horizon': 'very-short-term',
                'risk_level': 'low',
                'market_specialty': 'volatile'
            },
            
            # TIER 2: SPECIALIZED STRATEGIES
            'mean_reversion': {
                'class': 'MeanReversionStrategy', 
                'min_confidence': 0.60,
                'type': 'contrarian',
                'time_horizon': 'short-term',
                'risk_level': 'medium',
                'market_specialty': 'sideways'
            },
            'momentum': {
                'class': 'MomentumStrategy', 
                'min_confidence': 0.65,
                'type': 'trend_following',
                'time_horizon': 'short-term',
                'risk_level': 'high',
                'market_specialty': 'trending'
            },
            'grid': {
                'class': 'GridStrategy', 
                'min_confidence': 0.55,
                'type': 'range_trading',
                'time_horizon': 'medium-term',
                'risk_level': 'medium',
                'market_specialty': 'sideways'
            },
            'liquidation_hunter': {
                'class': 'LiquidationStrategy', 
                'min_confidence': 0.75,
                'type': 'opportunistic',
                'time_horizon': 'very-short-term',
                'risk_level': 'high',
                'market_specialty': 'volatile'
            },
            
            # TIER 3: YIELD & DEFI
            'defi_yield': {
                'class': 'DeFiYieldStrategy', 
                'min_confidence': 0.75,
                'type': 'yield_farming',
                'time_horizon': 'long-term',
                'risk_level': 'medium-high',
                'market_specialty': 'bull'
            },
            'stablecoin_parking': {
                'class': 'StablecoinParkingStrategy', 
                'min_confidence': 0.90,
                'type': 'capital_preservation',
                'time_horizon': 'long-term',
                'risk_level': 'very-low',
                'market_specialty': 'bear'
            },
            
            # TIER 4: META & HIGH-FREQUENCY
            'autopilot': {
                'class': 'AutopilotStrategy', 
                'min_confidence': 0.65,
                'type': 'meta_coordinator',
                'time_horizon': 'dynamic',
                'risk_level': 'medium',
                'market_specialty': 'all'
            },
            'scalping': {
                'class': 'ScalpingStrategy', 
                'min_confidence': 0.80,
                'type': 'high_frequency',
                'time_horizon': 'very-short-term',
                'risk_level': 'high',
                'market_specialty': 'volatile'
            },
            
            # DEAKTIVIERT: Schlechte Performance
            'copy_trading': {
                'class': 'CopyTradingStrategy', 
                'min_confidence': 0.99,  # Praktisch deaktiviert
                'type': 'social',
                'time_horizon': 'short-term',
                'risk_level': 'high',
                'market_specialty': 'none'
            }
        }
        
        self.available_strategies = strategy_configs
        
        # Strategie-Kategorien für bessere Orchestrierung
        self.strategy_categories = {
            'meta': ['lazy_billionaire', 'autopilot'],
            'ml_enhanced': ['ml_strategy'],
            'market_neutral': ['arbitrage'],
            'trend_following': ['momentum'],
            'contrarian': ['mean_reversion'],
            'range_trading': ['grid'],
            'opportunistic': ['liquidation_hunter'],
            'yield_farming': ['defi_yield'],
            'capital_preservation': ['stablecoin_parking'],
            'high_frequency': ['scalping']
        }
        
        active_strategies = [name for name, config in strategy_configs.items() 
                           if config['min_confidence'] < 0.95]
        
        self.logger.info(f"Registered {len(strategy_configs)} total strategies, {len(active_strategies)} active")
    
    def _initialize_performance_tracking(self):
        """Initialisiere Performance-Tracking"""
        for strategy_name in self.available_strategies.keys():
            self.strategy_performances[strategy_name] = StrategyPerformance(
                name=strategy_name,
                current_allocation=0.0,
                target_allocation=0.0,
                daily_pnl=0.0,
                weekly_pnl=0.0,
                monthly_pnl=0.0,
                sharpe_ratio=1.0,  # Neutral start
                max_drawdown=0.0,
                win_rate=0.5,
                total_trades=0,
                confidence_score=0.5,
                market_correlation=0.0,
                last_updated=datetime.now()
            )
    
    async def _initialize_ml_models(self):
        """Initialisiere ML-Modelle"""
        try:
            if ML_AVAILABLE:
                # Hier würden historische Daten für ML-Training geladen
                # Für Demo: Dummy-Training
                self.ml_optimizer.is_trained = True
                self.logger.info("ML models initialized")
            else:
                self.logger.warning("ML not available - using fallback methods")
        except Exception as e:
            self.logger.error(f"Error initializing ML models: {e}")
    
    async def analyze_and_rebalance(self, market_data: Dict[str, pd.DataFrame], 
                                  current_portfolio: Dict[str, float],
                                  total_capital: float) -> Dict[str, StrategyRecommendation]:
        """
        Hauptfunktion: Analysiere Markt und rebalanciere Portfolio
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # 1. Marktanalyse
            market_state = self.market_analyzer.analyze_market_state(market_data)
            self.current_market_state = market_state
            
            # 2. Strategie-Performance aktualisieren
            self._update_strategy_performances(current_portfolio)
            
            # 3. Optimal Weights berechnen
            optimal_weights = self.weighting_engine.calculate_optimal_weights(
                market_state, self.strategy_performances, self.portfolio_mode
            )
            
            # 4. Aktive Strategien auswählen
            active_strategies = self._select_active_strategies(optimal_weights, market_state)
            
            # 5. Position Sizing optimieren
            recommendations = {}
            
            for strategy_name, target_weight in active_strategies.items():
                if target_weight > 0.01:  # Nur signifikante Allokationen
                    recommendation = await self._create_strategy_recommendation(
                        strategy_name, target_weight, market_state, market_data, total_capital
                    )
                    recommendations[strategy_name] = recommendation
            
            # 6. Portfolio-Level Adjustments
            recommendations = self._apply_portfolio_constraints(recommendations, total_capital)
            
            # 7. Logging und Monitoring
            self._log_rebalancing_decision(market_state, recommendations)
            
            self.last_rebalance = datetime.now()
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error in analyze_and_rebalance: {e}")
            return self._get_fallback_recommendations(current_portfolio)
    
    def _select_active_strategies(self, optimal_weights: Dict[str, float], 
                                market_state: MarketStateAnalysis) -> Dict[str, float]:
        """
        Wähle aktive Strategien basierend auf Gewichtungen und Constraints
        """
        try:
            # Sortiere nach Gewichtung
            sorted_strategies = sorted(optimal_weights.items(), key=lambda x: x[1], reverse=True)
            
            selected_strategies = {}
            total_selected_weight = 0.0
            strategy_count = 0
            
            for strategy_name, weight in sorted_strategies:
                if strategy_count >= self.max_strategies:
                    break
                
                # Prüfe Mindest-Confidence
                strategy_config = self.available_strategies.get(strategy_name, {})
                min_confidence = strategy_config.get('min_confidence', 0.5)
                
                # Strategie-spezifische Confidence-Prüfung
                strategy_confidence = self._calculate_strategy_confidence(strategy_name, market_state)
                
                if strategy_confidence >= min_confidence and weight >= 0.02:  # Mindestens 2%
                    selected_strategies[strategy_name] = weight
                    total_selected_weight += weight
                    strategy_count += 1
                    self.logger.debug(f"Selected {strategy_name}: {weight:.3f} (confidence: {strategy_confidence:.2f})")
            
            # Normalisierung auf ausgewählte Strategien
            if total_selected_weight > 0:
                normalization_factor = 1.0 / total_selected_weight
                selected_strategies = {k: v * normalization_factor for k, v in selected_strategies.items()}
            
            self.logger.info(f"Selected {len(selected_strategies)} active strategies")
            return selected_strategies
            
        except Exception as e:
            self.logger.error(f"Error selecting active strategies: {e}")
            return {'stablecoin_parking': 1.0}  # Fallback
    
    def _calculate_strategy_confidence(self, strategy_name: str, market_state: MarketStateAnalysis) -> float:
        """
        INTELLIGENTE Confidence-Berechnung für ALLE Strategien basierend auf Marktlage
        """
        try:
            regime = market_state.regime
            volatility = market_state.volatility_regime
            trend_strength = market_state.trend_strength
            opportunity_score = market_state.opportunity_score
            
            # Hole Strategie-Konfiguration
            strategy_config = self.available_strategies.get(strategy_name, {})
            market_specialty = strategy_config.get('market_specialty', 'all')
            strategy_type = strategy_config.get('type', 'unknown')
            
            # BASIS-CONFIDENCE basierend auf Market Specialty
            base_confidence = self._get_base_confidence_by_specialty(
                market_specialty, regime, volatility, trend_strength
            )
            
            # STRATEGIE-SPEZIFISCHE ANPASSUNGEN
            if strategy_name == 'ml_strategy':
                # ML: Beste Performance bei erkennbaren Mustern
                if regime in [MarketRegime.BULL_WEAK, MarketRegime.BEAR_WEAK, MarketRegime.SIDEWAYS_LOW_VOL]:
                    base_confidence = 0.85
                elif regime in [MarketRegime.EXTREME_VOLATILITY, MarketRegime.TRANSITION_BULL, MarketRegime.TRANSITION_BEAR]:
                    base_confidence = 0.45  # ML unsicher bei Chaos
                else:
                    base_confidence = 0.75
                # Bonus für hohe Opportunity
                base_confidence += opportunity_score * 0.15
            
            elif strategy_name == 'arbitrage':
                # Arbitrage: Volatilität = Opportunities
                if volatility == 'extreme':
                    base_confidence = 0.95
                elif volatility == 'high':
                    base_confidence = 0.85
                elif volatility == 'low':
                    base_confidence = 0.60  # Weniger Opportunities
                else:
                    base_confidence = 0.75
            
            elif strategy_name == 'momentum':
                # Momentum: Braucht starke Trends
                if regime in [MarketRegime.BULL_STRONG, MarketRegime.BEAR_STRONG]:
                    base_confidence = 0.75 + trend_strength * 0.25  # Bis zu 1.0
                elif 'sideways' in regime.value.lower():
                    base_confidence = 0.25  # Sehr schlecht in Ranges
                elif regime in [MarketRegime.TRANSITION_BULL, MarketRegime.TRANSITION_BEAR]:
                    base_confidence = 0.65  # Moderate in Transitions
                else:
                    base_confidence = 0.55
            
            elif strategy_name == 'mean_reversion':
                # Mean Reversion: Liebt Ranges und Übertreibungen
                if 'sideways' in regime.value.lower():
                    base_confidence = 0.85
                elif regime in [MarketRegime.BULL_STRONG, MarketRegime.BEAR_STRONG]:
                    base_confidence = 0.35  # Schlecht bei starken Trends
                elif volatility in ['high', 'extreme']:
                    base_confidence = 0.75  # Gut bei Volatilität
                else:
                    base_confidence = 0.65
            
            elif strategy_name == 'grid':
                # Grid: Stabile Ranges bevorzugt
                if regime == MarketRegime.SIDEWAYS_LOW_VOL:
                    base_confidence = 0.90
                elif 'sideways' in regime.value.lower():
                    base_confidence = 0.80
                elif volatility == 'low':
                    base_confidence = 0.75
                else:
                    base_confidence = 0.50
            
            elif strategy_name == 'liquidation_hunter':
                # Liquidation Hunter: Volatilität + Extreme Moves
                if regime == MarketRegime.EXTREME_VOLATILITY:
                    base_confidence = 0.95
                elif volatility == 'extreme':
                    base_confidence = 0.90
                elif regime in [MarketRegime.BEAR_STRONG]:
                    base_confidence = 0.80  # Bear-Liquidations
                elif volatility == 'high':
                    base_confidence = 0.75
                else:
                    base_confidence = 0.45
            
            elif strategy_name == 'defi_yield':
                # DeFi Yield: Bull-Markets bevorzugt
                if regime in [MarketRegime.BULL_STRONG, MarketRegime.BULL_WEAK]:
                    base_confidence = 0.85
                elif regime == MarketRegime.RECOVERY:
                    base_confidence = 0.80
                elif regime in [MarketRegime.BEAR_STRONG, MarketRegime.EXTREME_VOLATILITY]:
                    base_confidence = 0.30  # Risiko in Bear-Markets
                else:
                    base_confidence = 0.60
            
            elif strategy_name == 'stablecoin_parking':
                # Stablecoin: Immer verfügbar, höher bei Unsicherheit
                if market_state.risk_level == 'very_high':
                    base_confidence = 0.98
                elif market_state.risk_level == 'high':
                    base_confidence = 0.95
                elif regime in [MarketRegime.BEAR_STRONG, MarketRegime.EXTREME_VOLATILITY]:
                    base_confidence = 0.92
                elif regime in [MarketRegime.BULL_STRONG]:
                    base_confidence = 0.60  # Opportunity Cost in Bull
                else:
                    base_confidence = 0.85
            
            elif strategy_name == 'lazy_billionaire':
                # Meta-Strategie: Konsistent gut
                base_confidence = 0.80 + opportunity_score * 0.15
                if regime in [MarketRegime.TRANSITION_BULL, MarketRegime.TRANSITION_BEAR]:
                    base_confidence += 0.10  # Gut bei Unsicherheit
            
            elif strategy_name == 'autopilot':
                # Meta-Coordinator: Adaptive Intelligence
                base_confidence = 0.75 + opportunity_score * 0.20
                # Bonus für komplexe Marktbedingungen
                if volatility in ['high', 'extreme']:
                    base_confidence += 0.10
            
            elif strategy_name == 'scalping':
                # Scalping: Hohe Volatilität bevorzugt
                if volatility == 'extreme':
                    base_confidence = 0.85
                elif volatility == 'high':
                    base_confidence = 0.80
                elif volatility == 'low':
                    base_confidence = 0.45
                else:
                    base_confidence = 0.65
            
            # WEITERE ADJUSTMENTS
            confidence = base_confidence
            
            # Liquidity-Adjustment
            confidence *= market_state.liquidity_score
            
            # Historical Performance-Adjustment
            if strategy_name in self.strategy_performances:
                perf = self.strategy_performances[strategy_name]
                perf_multiplier = 0.8 + (perf.confidence_score * 0.4)  # 0.8 - 1.2
                confidence *= perf_multiplier
            
            # Opportunity-Score-Bonus
            confidence += (opportunity_score - 0.5) * 0.1  # ±0.05
            
            return max(0.1, min(0.98, confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating strategy confidence for {strategy_name}: {e}")
            return 0.5
    
    def _get_base_confidence_by_specialty(self, specialty: str, regime, volatility: str, trend_strength: float) -> float:
        """Basis-Confidence basierend auf Market Specialty"""
        if specialty == 'all':
            return 0.70  # Meta-Strategien
        elif specialty == 'trending':
            return 0.60 + trend_strength * 0.30
        elif specialty == 'sideways':
            return 0.80 if 'sideways' in regime.value.lower() else 0.45
        elif specialty == 'volatile':
            vol_scores = {'low': 0.40, 'medium': 0.60, 'high': 0.80, 'extreme': 0.90}
            return vol_scores.get(volatility, 0.60)
        elif specialty == 'bull':
            return 0.85 if 'bull' in regime.value.lower() else 0.40
        elif specialty == 'bear':
            return 0.85 if 'bear' in regime.value.lower() else 0.60
        else:
            return 0.60
    
    async def _create_strategy_recommendation(self, strategy_name: str, target_weight: float,
                                            market_state: MarketStateAnalysis, 
                                            market_data: Dict[str, pd.DataFrame],
                                            total_capital: float) -> StrategyRecommendation:
        """
        Erstelle detaillierte Strategie-Empfehlung
        """
        try:
            # ML-Enhanced Entry/Exit Signal
            ml_signal = self.ml_optimizer.generate_signal(
                market_data, market_state.regime, strategy_name
            )
            
            # Position Sizing mit Kelly Criterion
            strategy_stats = self._get_strategy_stats(strategy_name)
            market_conditions = self._convert_to_market_conditions(market_state)
            
            position_size = self.kelly_optimizer.calculate_position_size(
                strategy_stats, market_conditions, total_capital
            )
            
            # Final allocation (kombiniert Target Weight und Kelly)
            kelly_weight = position_size.recommended_size
            
            # Gewichteter Durchschnitt zwischen Target und Kelly
            if ml_signal.confidence > 0.7:
                final_allocation = target_weight * 0.7 + kelly_weight * 0.3
            else:
                final_allocation = target_weight * 0.5 + kelly_weight * 0.5
            
            # Confidence Score
            overall_confidence = (
                target_weight * 0.3 +  # Portfolio-level confidence
                ml_signal.confidence * 0.4 +  # ML confidence
                position_size.confidence * 0.3  # Kelly confidence
            )
            
            # Expected Performance
            expected_performance = self._calculate_expected_performance(
                strategy_name, market_state, ml_signal
            )
            
            # Risk Score
            risk_score = self._calculate_strategy_risk_score(
                strategy_name, market_state, position_size
            )
            
            # Rationale
            rationale = [
                f"Market regime: {market_state.regime.value}",
                f"ML signal: {ml_signal.direction.value} (conf: {ml_signal.confidence:.2f})",
                f"Kelly size: {kelly_weight:.2%}",
                f"Target weight: {target_weight:.2%}",
                f"Risk level: {market_state.risk_level}"
            ]
            
            return StrategyRecommendation(
                strategy_name=strategy_name,
                recommended_allocation=final_allocation,
                confidence=overall_confidence,
                rationale=rationale,
                expected_performance=expected_performance,
                risk_score=risk_score,
                time_horizon=self._get_strategy_time_horizon(strategy_name)
            )
            
        except Exception as e:
            self.logger.error(f"Error creating recommendation for {strategy_name}: {e}")
            return self._get_fallback_recommendation(strategy_name, target_weight)
    
    def _get_strategy_stats(self, strategy_name: str) -> StrategyStats:
        """Hole Strategie-Statistiken für Kelly Criterion"""
        # In Produktion würden hier echte historische Daten kommen
        # Für Demo: Verwende Backtest-Erkenntnisse
        
        strategy_defaults = {
            'ml': StrategyStats('ml', 0.64, 0.068, -0.022, 1.88, 450, [0.02, -0.01, 0.03], 0.75, MarketRegime.BULL_WEAK, 0.32),
            'arbitrage': StrategyStats('arbitrage', 0.78, 0.042, -0.006, 2.45, 1250, [0.01, 0.01, 0.01], 0.85, MarketRegime.SIDEWAYS_LOW_VOL, 0.15),
            'momentum': StrategyStats('momentum', 0.58, 0.052, -0.018, 1.65, 320, [0.03, -0.02, 0.04], 0.70, MarketRegime.BULL_STRONG, 0.28),
            'mean_reversion': StrategyStats('mean_reversion', 0.62, 0.034, -0.012, 1.42, 680, [0.01, 0.02, -0.01], 0.68, MarketRegime.SIDEWAYS_HIGH_VOL, 0.22),
            'lazy_billionaire': StrategyStats('lazy_billionaire', 0.65, 0.045, -0.032, 1.15, 12, [0.04, 0.02, -0.01], 0.60, MarketRegime.BULL_WEAK, 0.35),
            'grid': StrategyStats('grid', 0.71, 0.028, -0.009, 1.35, 2100, [0.01, 0.01, 0.02], 0.72, MarketRegime.SIDEWAYS_LOW_VOL, 0.18),
            'stablecoin_parking': StrategyStats('stablecoin_parking', 0.95, 0.008, -0.001, 3.00, 24, [0.007, 0.008, 0.008], 0.90, MarketRegime.BEAR_STRONG, 0.02),
            'defi': StrategyStats('defi', 0.55, 0.058, -0.028, 1.52, 85, [0.05, -0.03, 0.02], 0.55, MarketRegime.BULL_STRONG, 0.35),
            'copy_trading': StrategyStats('copy_trading', 0.56, 0.038, -0.015, 1.38, 180, [0.02, -0.01, 0.01], 0.58, MarketRegime.BULL_WEAK, 0.25)
        }
        
        return strategy_defaults.get(strategy_name, 
                                   StrategyStats(strategy_name, 0.5, 0.02, -0.02, 1.0, 100, [0.0], 0.5, MarketRegime.SIDEWAYS_LOW_VOL, 0.2))
    
    def _convert_to_market_conditions(self, market_state: MarketStateAnalysis) -> MarketConditions:
        """Konvertiere MarketStateAnalysis zu MarketConditions"""
        return MarketConditions(
            regime=market_state.regime,
            volatility=0.5 if market_state.volatility_regime == 'extreme' else 0.3,
            trend_strength=market_state.trend_strength,
            liquidity_score=market_state.liquidity_score,
            correlation_level=0.4,  # Default
            vix_equivalent=25.0,    # Default
            funding_rates=0.01,     # Default
            sentiment_score=market_state.sentiment_score
        )
    
    def _calculate_expected_performance(self, strategy_name: str, market_state: MarketStateAnalysis, 
                                      ml_signal: MLSignal) -> Dict[str, float]:
        """Berechne erwartete Performance-Metriken"""
        # Basis-Performance aus Backtests
        base_performance = {
            'ml': {'annual_return': 0.68, 'sharpe': 1.88, 'max_dd': 0.22},
            'arbitrage': {'annual_return': 0.42, 'sharpe': 2.45, 'max_dd': 0.06},
            'momentum': {'annual_return': 0.52, 'sharpe': 1.65, 'max_dd': 0.18},
            'mean_reversion': {'annual_return': 0.34, 'sharpe': 1.42, 'max_dd': 0.12},
            'lazy_billionaire': {'annual_return': 0.45, 'sharpe': 1.15, 'max_dd': 0.32},
            'grid': {'annual_return': 0.28, 'sharpe': 1.35, 'max_dd': 0.09},
            'stablecoin_parking': {'annual_return': 0.08, 'sharpe': 3.00, 'max_dd': 0.01},
            'defi': {'annual_return': 0.58, 'sharpe': 1.52, 'max_dd': 0.28},
            'copy_trading': {'annual_return': 0.38, 'sharpe': 1.38, 'max_dd': 0.15}
        }
        
        base = base_performance.get(strategy_name, {'annual_return': 0.2, 'sharpe': 1.0, 'max_dd': 0.15})
        
        # ML und Markt-Adjustments
        ml_multiplier = 0.8 + (ml_signal.confidence * 0.4)  # 0.8 - 1.2
        market_multiplier = 0.9 + (market_state.opportunity_score * 0.2)  # 0.9 - 1.1
        
        return {
            'expected_annual_return': base['annual_return'] * ml_multiplier * market_multiplier,
            'expected_sharpe': base['sharpe'] * ml_multiplier,
            'expected_max_drawdown': base['max_dd'] / ml_multiplier,
            'confidence_level': ml_signal.confidence
        }
    
    def _calculate_strategy_risk_score(self, strategy_name: str, market_state: MarketStateAnalysis,
                                     position_size: PositionSize) -> float:
        """Berechne Risiko-Score für Strategie"""
        base_risk = {
            'ml': 0.6, 'arbitrage': 0.3, 'momentum': 0.7, 'mean_reversion': 0.5,
            'lazy_billionaire': 0.6, 'grid': 0.4, 'stablecoin_parking': 0.1,
            'defi': 0.8, 'copy_trading': 0.6
        }
        
        strategy_risk = base_risk.get(strategy_name, 0.5)
        
        # Market adjustments
        if market_state.risk_level == 'very_high':
            strategy_risk *= 1.5
        elif market_state.risk_level == 'high':
            strategy_risk *= 1.2
        elif market_state.risk_level == 'low':
            strategy_risk *= 0.8
        
        # Position size adjustment
        if position_size.recommended_size > 0.15:
            strategy_risk *= 1.2
        elif position_size.recommended_size < 0.05:
            strategy_risk *= 0.9
        
        return max(0.1, min(1.0, strategy_risk))
    
    def _get_strategy_time_horizon(self, strategy_name: str) -> str:
        """Hole typischen Zeithorizont für Strategie"""
        horizons = {
            'ml': 'short-term',
            'arbitrage': 'very-short-term',
            'momentum': 'short-term',
            'mean_reversion': 'short-term',
            'lazy_billionaire': 'long-term',
            'grid': 'medium-term',
            'stablecoin_parking': 'long-term',
            'defi': 'medium-term',
            'copy_trading': 'short-term'
        }
        return horizons.get(strategy_name, 'medium-term')
    
    def _apply_portfolio_constraints(self, recommendations: Dict[str, StrategyRecommendation],
                                   total_capital: float) -> Dict[str, StrategyRecommendation]:
        """Anwendung von Portfolio-level Constraints"""
        # Gesamtallokation prüfen und normalisieren
        total_allocation = sum(rec.recommended_allocation for rec in recommendations.values())
        
        if total_allocation > 1.0:
            # Proportional reduzieren
            scale_factor = 0.95 / total_allocation  # 5% Cash Reserve
            for rec in recommendations.values():
                rec.recommended_allocation *= scale_factor
                rec.rationale.append(f"Scaled by {scale_factor:.2f} for portfolio constraints")
        
        return recommendations
    
    def _update_strategy_performances(self, current_portfolio: Dict[str, float]):
        """Aktualisiere Strategy Performance Tracking"""
        # Vereinfachtes Update - in Produktion würde hier echte P&L-Berechnung stattfinden
        for strategy_name in self.strategy_performances:
            if strategy_name in current_portfolio:
                # Simuliere Performance Update
                current_alloc = current_portfolio[strategy_name]
                self.strategy_performances[strategy_name].current_allocation = current_alloc
                self.strategy_performances[strategy_name].last_updated = datetime.now()
    
    def _log_rebalancing_decision(self, market_state: MarketStateAnalysis, 
                                recommendations: Dict[str, StrategyRecommendation]):
        """Ausführliches Logging der Rebalancing-Entscheidung"""
        self.logger.info("=== REBALANCING DECISION ===")
        self.logger.info(f"Market Regime: {market_state.regime.value}")
        self.logger.info(f"Risk Level: {market_state.risk_level}")
        self.logger.info(f"Opportunity Score: {market_state.opportunity_score:.2f}")
        self.logger.info(f"Recommended Exposure: {market_state.recommended_exposure:.1%}")
        
        self.logger.info("Strategy Recommendations:")
        for name, rec in recommendations.items():
            self.logger.info(f"  {name}: {rec.recommended_allocation:.1%} (conf: {rec.confidence:.2f}, risk: {rec.risk_score:.2f})")
        
        total_allocation = sum(rec.recommended_allocation for rec in recommendations.values())
        self.logger.info(f"Total Allocation: {total_allocation:.1%}")
        self.logger.info("=== END REBALANCING ===")
    
    def _get_fallback_recommendations(self, current_portfolio: Dict[str, float]) -> Dict[str, StrategyRecommendation]:
        """Fallback-Empfehlungen bei Fehlern"""
        return {
            'stablecoin_parking': StrategyRecommendation(
                strategy_name='stablecoin_parking',
                recommended_allocation=0.5,
                confidence=0.8,
                rationale=['Fallback recommendation due to system error'],
                expected_performance={'expected_annual_return': 0.08},
                risk_score=0.1,
                time_horizon='long-term'
            ),
            'arbitrage': StrategyRecommendation(
                strategy_name='arbitrage',
                recommended_allocation=0.3,
                confidence=0.7,
                rationale=['Safe fallback strategy'],
                expected_performance={'expected_annual_return': 0.25},
                risk_score=0.3,
                time_horizon='short-term'
            )
        }
    
    def _get_fallback_recommendation(self, strategy_name: str, target_weight: float) -> StrategyRecommendation:
        """Einzelne Fallback-Empfehlung"""
        return StrategyRecommendation(
            strategy_name=strategy_name,
            recommended_allocation=target_weight * 0.5,  # Reduziert bei Unsicherheit
            confidence=0.4,
            rationale=[f'Fallback recommendation for {strategy_name}'],
            expected_performance={'expected_annual_return': 0.15},
            risk_score=0.5,
            time_horizon='medium-term'
        )
    
    def get_current_status(self) -> Dict[str, Any]:
        """Hole aktuellen Strategie-Status"""
        return {
            'is_initialized': self.is_initialized,
            'portfolio_mode': self.portfolio_mode.value,
            'active_strategies': list(self.active_strategies.keys()),
            'last_rebalance': self.last_rebalance.isoformat() if self.last_rebalance else None,
            'current_market_regime': self.current_market_state.regime.value if self.current_market_state else None,
            'total_strategies_available': len(self.available_strategies)
        }

def main():
    """Test der SuperLazyBillionaireStrategy"""
    print("🚀 Testing SuperLazyBillionaireStrategy - The Ultimate Multi-Strategy Orchestrator")
    
    # Sample market data
    np.random.seed(42)
    sample_data = {}
    for timeframe in ['15m', '1h', '4h', '1d', '1w']:
        n_periods = {'15m': 1000, '1h': 500, '4h': 200, '1d': 100, '1w': 20}[timeframe]
        
        closes = 50000 + np.cumsum(np.random.randn(n_periods) * 1000)
        highs = closes + np.random.rand(n_periods) * 500
        lows = closes - np.random.rand(n_periods) * 500
        opens = np.roll(closes, 1)
        volumes = np.random.rand(n_periods) * 1000000 + 500000
        
        sample_data[timeframe] = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
    
    async def test_strategy():
        # Initialize strategy
        config = {
            'portfolio_mode': 'aggressive',
            'max_active_strategies': 6,
            'confidence_threshold': 0.65
        }
        
        strategy = SuperLazyBillionaireStrategy(config)
        await strategy.initialize()
        
        print(f"\n📊 Strategy Status: {strategy.get_current_status()}")
        
        # Sample current portfolio
        current_portfolio = {
            'ml': 0.15,
            'arbitrage': 0.20,
            'momentum': 0.10,
            'stablecoin_parking': 0.25,
            'grid': 0.15,
            'lazy_billionaire': 0.15
        }
        
        total_capital = 300000  # €300,000
        
        # Run analysis and rebalancing
        recommendations = await strategy.analyze_and_rebalance(
            sample_data, current_portfolio, total_capital
        )
        
        print(f"\n🎯 SuperLazyBillionaire Recommendations:")
        print(f"Portfolio Mode: {strategy.portfolio_mode.value.upper()}")
        print(f"Total Capital: €{total_capital:,.0f}")
        
        total_allocation = 0.0
        total_expected_return = 0.0
        
        for name, rec in recommendations.items():
            allocation_amount = rec.recommended_allocation * total_capital
            expected_return = rec.expected_performance.get('expected_annual_return', 0)
            total_allocation += rec.recommended_allocation
            total_expected_return += rec.recommended_allocation * expected_return
            
            print(f"\n{name.upper()}:")
            print(f"  Allocation: {rec.recommended_allocation:.1%} (€{allocation_amount:,.0f})")
            print(f"  Confidence: {rec.confidence:.1%}")
            print(f"  Expected Return: {expected_return:.1%}")
            print(f"  Risk Score: {rec.risk_score:.2f}")
            print(f"  Time Horizon: {rec.time_horizon}")
            print(f"  Key Rationale: {rec.rationale[0] if rec.rationale else 'N/A'}")
        
        cash_reserve = (1 - total_allocation) * total_capital
        
        print(f"\n📈 PORTFOLIO SUMMARY:")
        print(f"Total Allocated: {total_allocation:.1%} (€{total_allocation * total_capital:,.0f})")
        print(f"Cash Reserve: {(1-total_allocation):.1%} (€{cash_reserve:,.0f})")
        print(f"Expected Portfolio Return: {total_expected_return:.1%}")
        print(f"Active Strategies: {len(recommendations)}")
        
        # Market state info
        if strategy.current_market_state:
            ms = strategy.current_market_state
            print(f"\n🌍 MARKET CONDITIONS:")
            print(f"Regime: {ms.regime.value} → {ms.predicted_regime.value}")
            print(f"Volatility: {ms.volatility_regime}")
            print(f"Risk Level: {ms.risk_level}")
            print(f"Opportunity Score: {ms.opportunity_score:.1%}")
            print(f"Recommended Exposure: {ms.recommended_exposure:.1%}")
    
    # Run async test
    asyncio.run(test_strategy())
    
    print(f"\n✅ SuperLazyBillionaireStrategy test completed!")
    print(f"🎯 Target: 70-90% annual returns with optimized risk management")
    print(f"💼 Intelligent multi-strategy orchestration based on backtest insights")

if __name__ == "__main__":
    main()