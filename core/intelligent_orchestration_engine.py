#!/usr/bin/env python3
"""
Intelligent Strategy Orchestration Engine
=========================================

Das Gehirn des Trading-Bots: Orchestriert alle entdeckten Strategien intelligent
- Automatische Strategie-Auswahl basierend auf Marktbedingungen
- Dynamische Portfolio-Allokation zwischen Strategien
- Adaptive Gewichtung basierend auf Performance
- Konflikt-Resolution zwischen konkurrierenden Strategien
- Echtzeit-Optimierung der Strategie-Kombinationen

Dieser Engine lernt kontinuierlich und optimiert sich selbst!
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
from pathlib import Path
import threading
import random
from enum import Enum
import math

# Import unserer Discovery-Komponenten
from .strategy_orchestrator import StrategyDiscoveryEngine, StrategyDNA, StrategyHealthMetrics

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Markt-Regimes für intelligente Strategie-Auswahl"""
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    BULL_RANGING = "bull_ranging"
    BEAR_RANGING = "bear_ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRASH = "crash"
    EUPHORIA = "euphoria"
    UNCERTAINTY = "uncertainty"

@dataclass
class MarketConditions:
    """Aktuelle Marktbedingungen für Orchestrierung"""
    timestamp: datetime
    regime: MarketRegime
    volatility: float
    trend_strength: float
    volume_profile: str  # 'low', 'normal', 'high', 'extreme'
    sentiment_score: float  # -1 bis +1
    news_impact: float     # 0 bis 1
    
    # Technische Indikatoren
    rsi_14: float
    macd_signal: float
    bollinger_position: float
    
    # Meta-Informationen
    confidence: float      # Vertrauen in die Analyse
    data_quality: float    # Qualität der zugrundeliegenden Daten

@dataclass
class StrategyAllocation:
    """Allokation für eine spezifische Strategie"""
    strategy_name: str
    allocation_percent: float  # 0-100
    confidence_score: float    # Warum diese Allokation
    expected_return: float     # Erwarteter Return
    risk_score: float         # Risiko-Score
    market_fit_score: float   # Wie gut passt sie zu aktuellen Bedingungen
    
    # Ausführungs-Parameter
    max_positions: int
    position_size_multiplier: float
    risk_multiplier: float
    
    # Timing
    start_time: datetime
    duration_hours: float
    
    # Begründung
    reasoning: str
    allocation_factors: List[str]

@dataclass
class OrchestrationDecision:
    """Eine Orchestrierungs-Entscheidung"""
    timestamp: datetime
    market_conditions: MarketConditions
    active_strategies: List[StrategyAllocation]
    deactivated_strategies: List[str]
    total_risk_budget: float
    expected_portfolio_return: float
    confidence: float
    
    # Meta-Daten
    decision_factors: List[str]
    reasoning: str
    orchestration_mode: str  # 'conservative', 'balanced', 'aggressive', 'adaptive'

class StrategyCombinationOptimizer:
    """Optimiert Strategie-Kombinationen mittels genetischen Algorithmus"""
    
    def __init__(self):
        self.population_size = 20
        self.generations = 10
        self.mutation_rate = 0.1
        self.elite_size = 4
        
        logger.info("🧬 Strategy Combination Optimizer initialized")
    
    def optimize_strategy_mix(self, available_strategies: List[StrategyDNA],
                            market_conditions: MarketConditions,
                            risk_budget: float) -> List[StrategyAllocation]:
        """Optimiert Strategie-Mix mit genetischem Algorithmus"""
        
        if not available_strategies:
            return []
        
        # Erstelle initiale Population
        population = self._create_initial_population(available_strategies, risk_budget)
        
        best_fitness = -float('inf')
        best_individual = None
        
        for generation in range(self.generations):
            # Bewerte Population
            fitness_scores = []
            for individual in population:
                fitness = self._evaluate_fitness(individual, market_conditions, risk_budget)
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
            
            # Selektion und Reproduktion
            population = self._evolve_population(population, fitness_scores)
        
        # Konvertiere bestes Individuum zu Allocations
        return self._individual_to_allocations(best_individual, available_strategies, market_conditions)
    
    def _create_initial_population(self, strategies: List[StrategyDNA], 
                                 risk_budget: float) -> List[List[float]]:
        """Erstellt initiale Population von Strategie-Allokationen"""
        
        population = []
        num_strategies = len(strategies)
        
        for _ in range(self.population_size):
            # Zufällige Gewichtungen
            weights = np.random.random(num_strategies)
            weights = weights / weights.sum()  # Normalisiere zu 1.0
            
            # Berücksichtige Risk Budget
            individual = weights * risk_budget
            population.append(individual.tolist())
        
        return population
    
    def _evaluate_fitness(self, individual: List[float], 
                         market_conditions: MarketConditions,
                         risk_budget: float) -> float:
        """Bewertet Fitness einer Strategie-Kombination"""
        
        fitness = 0.0
        total_allocation = sum(individual)
        
        if total_allocation <= 0:
            return -1000  # Ungültige Allokation
        
        # 1. Risk-Budget Compliance
        risk_penalty = abs(total_allocation - risk_budget) * 10
        fitness -= risk_penalty
        
        # 2. Diversifikation Bonus
        non_zero_strategies = sum(1 for x in individual if x > 0.01)
        diversification_bonus = min(non_zero_strategies * 5, 25)
        fitness += diversification_bonus
        
        # 3. Market Fit Score
        # (Vereinfacht - in echt würde man hier Strategy DNA verwenden)
        market_fit = self._calculate_market_fit_score(individual, market_conditions)
        fitness += market_fit * 30
        
        # 4. Risiko-Return Balance
        estimated_return = sum(individual) * 0.02  # 2% base return
        risk_score = np.std(individual) if len(individual) > 1 else 0.1
        
        if risk_score > 0:
            sharpe_estimate = estimated_return / risk_score
            fitness += sharpe_estimate * 20
        
        # 5. Konzentrations-Penalty
        max_allocation = max(individual) if individual else 0
        if max_allocation > risk_budget * 0.8:  # Mehr als 80% in einer Strategie
            fitness -= 50
        
        return fitness
    
    def _calculate_market_fit_score(self, individual: List[float], 
                                  market_conditions: MarketConditions) -> float:
        """Berechnet Market-Fit Score (vereinfacht)"""
        
        # Vereinfachte Heuristik - in Realität würde man Strategy DNA verwenden
        regime = market_conditions.regime
        volatility = market_conditions.volatility
        
        score = 0.5  # Base score
        
        # Volatilitäts-basierte Anpassung
        if volatility > 0.03:  # Hohe Volatilität
            # Bevorzuge diversifizierte Allokation
            non_zero_count = sum(1 for x in individual if x > 0.01)
            score += min(non_zero_count * 0.1, 0.3)
        
        # Regime-basierte Anpassung
        if regime in [MarketRegime.BULL_TRENDING, MarketRegime.EUPHORIA]:
            # Bull Market - mehr Risiko okay
            score += min(sum(individual) / 100, 0.2)
        elif regime in [MarketRegime.BEAR_TRENDING, MarketRegime.CRASH]:
            # Bear Market - weniger Risiko
            score -= max(sum(individual) - 50, 0) * 0.01
        
        return max(0, min(score, 1.0))
    
    def _evolve_population(self, population: List[List[float]], 
                          fitness_scores: List[float]) -> List[List[float]]:
        """Evolviert Population durch Selektion, Crossover und Mutation"""
        
        # Sortiere nach Fitness
        sorted_indices = sorted(range(len(fitness_scores)), 
                               key=lambda i: fitness_scores[i], reverse=True)
        
        new_population = []
        
        # Elite beibehalten
        for i in range(self.elite_size):
            if i < len(sorted_indices):
                new_population.append(population[sorted_indices[i]].copy())
        
        # Rest durch Crossover und Mutation erzeugen
        while len(new_population) < self.population_size:
            # Eltern auswählen (Tournament Selection)
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)
            
            # Crossover
            child = self._crossover(parent1, parent2)
            
            # Mutation
            if random.random() < self.mutation_rate:
                child = self._mutate(child)
            
            new_population.append(child)
        
        return new_population[:self.population_size]
    
    def _tournament_selection(self, population: List[List[float]], 
                             fitness_scores: List[float]) -> List[float]:
        """Tournament Selection für Eltern-Auswahl"""
        
        tournament_size = 3
        tournament_indices = random.sample(range(len(population)), 
                                         min(tournament_size, len(population)))
        
        best_index = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_index].copy()
    
    def _crossover(self, parent1: List[float], parent2: List[float]) -> List[float]:
        """Uniform Crossover zwischen zwei Eltern"""
        
        child = []
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i] if i < len(parent2) else parent1[i])
        
        return child
    
    def _mutate(self, individual: List[float]) -> List[float]:
        """Mutiert ein Individuum"""
        
        mutated = individual.copy()
        
        # Zufällige Mutation eines Gens
        if mutated:
            mutation_index = random.randint(0, len(mutated) - 1)
            mutation_strength = random.uniform(-0.1, 0.1)  # ±10%
            
            mutated[mutation_index] = max(0, mutated[mutation_index] * (1 + mutation_strength))
        
        return mutated
    
    def _individual_to_allocations(self, individual: List[float], 
                                  strategies: List[StrategyDNA],
                                  market_conditions: MarketConditions) -> List[StrategyAllocation]:
        """Konvertiert genetisches Individuum zu StrategyAllocations"""
        
        allocations = []
        
        for i, allocation_value in enumerate(individual):
            if i >= len(strategies) or allocation_value < 0.01:  # Mindest-Allokation
                continue
            
            strategy = strategies[i]
            allocation_percent = (allocation_value / sum(individual)) * 100
            
            # Berechne Scores
            market_fit_score = self._calculate_strategy_market_fit(strategy, market_conditions)
            
            allocation = StrategyAllocation(
                strategy_name=strategy.name,
                allocation_percent=allocation_percent,
                confidence_score=0.7,  # Default
                expected_return=strategy.expected_return_per_trade * allocation_percent / 100,
                risk_score=self._calculate_strategy_risk(strategy),
                market_fit_score=market_fit_score,
                max_positions=max(1, int(strategy.expected_trades_per_day * allocation_percent / 100)),
                position_size_multiplier=allocation_percent / 100,
                risk_multiplier=allocation_percent / 100,
                start_time=datetime.now(),
                duration_hours=24.0,  # Default: 24 Stunden
                reasoning=f"Genetic optimization for {market_conditions.regime.value}",
                allocation_factors=['genetic_optimization', 'market_fit', 'diversification']
            )
            
            allocations.append(allocation)
        
        return sorted(allocations, key=lambda x: x.allocation_percent, reverse=True)
    
    def _calculate_strategy_market_fit(self, strategy: StrategyDNA, 
                                     market_conditions: MarketConditions) -> float:
        """Berechnet wie gut Strategie zu Marktbedingungen passt"""
        
        fit_score = 0.5  # Base
        
        # Volatility Fit
        if market_conditions.volatility > strategy.minimum_volatility:
            fit_score += 0.2
        
        # Risk Level Fit
        regime = market_conditions.regime
        if regime in [MarketRegime.BULL_TRENDING, MarketRegime.EUPHORIA]:
            if strategy.risk_level in ['aggressive', 'extreme']:
                fit_score += 0.2
        elif regime in [MarketRegime.BEAR_TRENDING, MarketRegime.CRASH]:
            if strategy.risk_level in ['conservative', 'moderate']:
                fit_score += 0.2
        
        # Timeframe Fit
        if regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.CRASH]:
            if strategy.timeframe in ['scalping', 'intraday']:
                fit_score += 0.1
        
        return max(0, min(fit_score, 1.0))
    
    def _calculate_strategy_risk(self, strategy: StrategyDNA) -> float:
        """Berechnet Risiko-Score einer Strategie"""
        
        risk_map = {
            'conservative': 0.2,
            'moderate': 0.4,
            'aggressive': 0.7,
            'extreme': 1.0
        }
        
        base_risk = risk_map.get(strategy.risk_level, 0.5)
        
        # Adjustiere basierend auf anderen Faktoren
        if strategy.max_drawdown_tolerance > 0.3:
            base_risk += 0.1
        if strategy.expected_trades_per_day > 5:
            base_risk += 0.1
        
        return min(base_risk, 1.0)

class MarketRegimeDetector:
    """Erkennt aktuelle Marktbedingungen für intelligente Orchestrierung"""
    
    def __init__(self):
        self.regime_history = deque(maxlen=100)
        self.confidence_threshold = 0.7
        
        logger.info("📊 Market Regime Detector initialized")
    
    def detect_market_conditions(self, market_data: Dict[str, pd.DataFrame]) -> MarketConditions:
        """Erkennt aktuelle Marktbedingungen"""
        
        try:
            # Verwende BTC als Lead-Indikator
            btc_data = market_data.get('BTC/USDT')
            if btc_data is None or len(btc_data) < 50:
                return self._get_default_conditions()
            
            # Berechne technische Indikatoren
            volatility = self._calculate_volatility(btc_data)
            trend_strength = self._calculate_trend_strength(btc_data)
            volume_profile = self._analyze_volume_profile(btc_data)
            
            # RSI, MACD, Bollinger
            rsi_14 = self._calculate_rsi(btc_data['close'], 14)
            macd_signal = self._calculate_macd(btc_data['close'])
            bollinger_position = self._calculate_bollinger_position(btc_data['close'])
            
            # Regime bestimmen
            regime = self._determine_regime(btc_data, volatility, trend_strength, rsi_14)
            
            # Sentiment (vereinfacht)
            sentiment_score = self._estimate_sentiment(btc_data, rsi_14, trend_strength)
            
            # News Impact (vereinfacht)
            news_impact = self._estimate_news_impact(btc_data)
            
            conditions = MarketConditions(
                timestamp=datetime.now(),
                regime=regime,
                volatility=volatility,
                trend_strength=trend_strength,
                volume_profile=volume_profile,
                sentiment_score=sentiment_score,
                news_impact=news_impact,
                rsi_14=rsi_14,
                macd_signal=macd_signal,
                bollinger_position=bollinger_position,
                confidence=self._calculate_confidence(btc_data),
                data_quality=self._assess_data_quality(btc_data)
            )
            
            # Zu Historie hinzufügen
            self.regime_history.append(regime)
            
            return conditions
            
        except Exception as e:
            logger.error(f"Error detecting market conditions: {e}")
            return self._get_default_conditions()
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Berechnet aktuelle Volatilität"""
        
        if len(data) < 24:
            return 0.02  # Default
        
        returns = data['close'].pct_change().dropna()
        recent_returns = returns.tail(24)  # Letzte 24 Stunden
        
        return recent_returns.std() if len(recent_returns) > 0 else 0.02
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Berechnet Trend-Stärke (-1 bis +1)"""
        
        if len(data) < 20:
            return 0.0
        
        # Einfacher Trend: SMA 20 vs aktueller Preis
        sma_20 = data['close'].rolling(20).mean()
        current_price = data['close'].iloc[-1]
        sma_value = sma_20.iloc[-1]
        
        if pd.isna(sma_value):
            return 0.0
        
        # Normalisiere auf -1 bis +1
        price_diff = (current_price - sma_value) / sma_value
        trend_strength = np.tanh(price_diff * 10)  # Sigmoid-ähnlich
        
        return trend_strength
    
    def _analyze_volume_profile(self, data: pd.DataFrame) -> str:
        """Analysiert Volumen-Profil"""
        
        if len(data) < 24:
            return 'normal'
        
        recent_volume = data['volume'].tail(24).mean()
        avg_volume = data['volume'].tail(168).mean()  # 7 Tage Average
        
        if pd.isna(recent_volume) or pd.isna(avg_volume) or avg_volume == 0:
            return 'normal'
        
        volume_ratio = recent_volume / avg_volume
        
        if volume_ratio > 3.0:
            return 'extreme'
        elif volume_ratio > 1.5:
            return 'high'
        elif volume_ratio < 0.5:
            return 'low'
        else:
            return 'normal'
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Berechnet RSI"""
        
        if len(prices) < period + 1:
            return 50.0
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> float:
        """Berechnet MACD Signal"""
        
        if len(prices) < slow:
            return 0.0
        
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        
        return macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0.0
    
    def _calculate_bollinger_position(self, prices: pd.Series, period: int = 20) -> float:
        """Berechnet Position innerhalb Bollinger Bands"""
        
        if len(prices) < period:
            return 0.5
        
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        
        current_price = prices.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        
        if pd.isna(current_upper) or pd.isna(current_lower) or current_upper == current_lower:
            return 0.5
        
        position = (current_price - current_lower) / (current_upper - current_lower)
        return max(0.0, min(1.0, position))
    
    def _determine_regime(self, data: pd.DataFrame, volatility: float, 
                         trend_strength: float, rsi: float) -> MarketRegime:
        """Bestimmt Markt-Regime"""
        
        # Extreme Bedingungen
        if volatility > 0.08:  # Sehr hohe Volatilität
            if trend_strength < -0.5:
                return MarketRegime.CRASH
            elif trend_strength > 0.7:
                return MarketRegime.EUPHORIA
            else:
                return MarketRegime.HIGH_VOLATILITY
        
        if volatility < 0.01:
            return MarketRegime.LOW_VOLATILITY
        
        # Normale Bedingungen
        if abs(trend_strength) < 0.2:  # Schwacher Trend
            if rsi > 60:
                return MarketRegime.BULL_RANGING
            elif rsi < 40:
                return MarketRegime.BEAR_RANGING
            else:
                return MarketRegime.UNCERTAINTY
        else:  # Starker Trend
            if trend_strength > 0:
                return MarketRegime.BULL_TRENDING
            else:
                return MarketRegime.BEAR_TRENDING
    
    def _estimate_sentiment(self, data: pd.DataFrame, rsi: float, trend_strength: float) -> float:
        """Schätzt Markt-Sentiment"""
        
        # Vereinfachte Sentiment-Schätzung basierend auf technischen Indikatoren
        sentiment = 0.0
        
        # RSI-basiert
        if rsi > 70:
            sentiment += 0.3  # Überkauft = bullish sentiment
        elif rsi < 30:
            sentiment -= 0.3  # Überverkauft = bearish sentiment
        else:
            sentiment += (rsi - 50) / 100  # Normalisiert
        
        # Trend-basiert
        sentiment += trend_strength * 0.5
        
        # Volumen-Impuls
        if len(data) >= 24:
            recent_volume_change = (data['volume'].iloc[-1] / data['volume'].iloc[-24] - 1)
            sentiment += np.tanh(recent_volume_change) * 0.2
        
        return max(-1.0, min(1.0, sentiment))
    
    def _estimate_news_impact(self, data: pd.DataFrame) -> float:
        """Schätzt News-Impact basierend auf Preisbewegungen"""
        
        if len(data) < 24:
            return 0.0
        
        # Schaue nach ungewöhnlichen Preisbewegungen
        recent_returns = data['close'].pct_change().tail(6)  # Letzte 6 Stunden
        unusual_moves = sum(abs(ret) > 0.05 for ret in recent_returns if not pd.isna(ret))  # >5% Moves
        
        news_impact = min(unusual_moves / 6.0, 1.0)  # Normalisiert
        
        return news_impact
    
    def _calculate_confidence(self, data: pd.DataFrame) -> float:
        """Berechnet Vertrauen in die Analyse"""
        
        confidence = 0.5  # Base
        
        # Daten-Vollständigkeit
        if len(data) >= 168:  # 7 Tage Daten
            confidence += 0.3
        elif len(data) >= 24:  # 1 Tag Daten
            confidence += 0.1
        
        # Daten-Konsistenz
        if not data['close'].isna().any():
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _assess_data_quality(self, data: pd.DataFrame) -> float:
        """Bewertet Datenqualität"""
        
        quality = 1.0
        
        # Missing Data Penalty
        missing_ratio = data.isna().sum().sum() / (len(data) * len(data.columns))
        quality -= missing_ratio * 0.5
        
        # Extreme Values Penalty
        price_changes = data['close'].pct_change().abs()
        extreme_moves = sum(price_changes > 0.2)  # >20% moves
        if extreme_moves > len(data) * 0.05:  # Mehr als 5% extreme moves
            quality -= 0.2
        
        return max(0.1, quality)
    
    def _get_default_conditions(self) -> MarketConditions:
        """Fallback-Marktbedingungen"""
        
        return MarketConditions(
            timestamp=datetime.now(),
            regime=MarketRegime.UNCERTAINTY,
            volatility=0.02,
            trend_strength=0.0,
            volume_profile='normal',
            sentiment_score=0.0,
            news_impact=0.0,
            rsi_14=50.0,
            macd_signal=0.0,
            bollinger_position=0.5,
            confidence=0.3,
            data_quality=0.5
        )

class IntelligentOrchestrationEngine:
    """
    Haupt-Orchestrierungs-Engine: Das Gehirn des Trading-Systems
    
    Kombiniert Discovery, Market Analysis und Optimization für intelligente
    Strategie-Orchestrierung
    """
    
    def __init__(self, discovery_engine: StrategyDiscoveryEngine = None):
        self.discovery_engine = discovery_engine
        self.market_detector = MarketRegimeDetector()
        self.optimizer = StrategyCombinationOptimizer()
        
        # Orchestrierung-State
        self.current_allocation: List[StrategyAllocation] = []
        self.orchestration_history: List[OrchestrationDecision] = []
        self.performance_tracker = defaultdict(list)
        
        # Konfiguration
        self.orchestration_mode = 'adaptive'  # 'conservative', 'balanced', 'aggressive', 'adaptive'
        self.rebalance_interval_hours = 6
        self.min_confidence_threshold = 0.6
        self.max_strategies_active = 5
        
        # Threading
        self.orchestration_running = False
        self.orchestration_thread = None
        self._stop_event = threading.Event()
        
        logger.info("🎼 Intelligent Orchestration Engine initialized")
    
    async def orchestrate_strategies(self, market_data: Dict[str, pd.DataFrame],
                                   risk_budget: float = 100.0) -> OrchestrationDecision:
        """Haupt-Orchestrierungs-Funktion: Intelligente Strategie-Auswahl"""
        
        logger.info("🎼 Starting intelligent strategy orchestration...")
        
        try:
            # 1. Marktbedingungen analysieren
            market_conditions = self.market_detector.detect_market_conditions(market_data)
            logger.info(f"📊 Market Regime: {market_conditions.regime.value}")
            logger.info(f"📈 Volatility: {market_conditions.volatility:.3f}")
            logger.info(f"🎯 Trend Strength: {market_conditions.trend_strength:+.2f}")
            
            # 2. Verfügbare Strategien abrufen
            if not self.discovery_engine:
                raise ValueError("Discovery Engine not initialized")
            
            available_strategies = list(self.discovery_engine.discovered_strategies.values())
            if not available_strategies:
                logger.warning("⚠️ No strategies discovered!")
                return self._create_empty_decision(market_conditions)
            
            logger.info(f"🔍 Available strategies: {len(available_strategies)}")
            
            # 3. Strategien nach Marktbedingungen filtern
            suitable_strategies = await self._filter_strategies_by_market(
                available_strategies, market_conditions
            )
            logger.info(f"✅ Suitable strategies: {len(suitable_strategies)}")
            
            # 4. Optimale Strategie-Kombination finden
            optimal_allocations = self.optimizer.optimize_strategy_mix(
                suitable_strategies, market_conditions, risk_budget
            )
            
            # 5. Konflikt-Resolution
            resolved_allocations = await self._resolve_strategy_conflicts(optimal_allocations)
            
            # 6. Final-Validierung
            validated_allocations = await self._validate_allocations(
                resolved_allocations, market_conditions, risk_budget
            )
            
            # 7. Deaktivierte Strategien identifizieren
            deactivated_strategies = self._identify_deactivated_strategies(validated_allocations)
            
            # 8. Orchestrierungs-Entscheidung erstellen
            decision = OrchestrationDecision(
                timestamp=datetime.now(),
                market_conditions=market_conditions,
                active_strategies=validated_allocations,
                deactivated_strategies=deactivated_strategies,
                total_risk_budget=risk_budget,
                expected_portfolio_return=sum(a.expected_return for a in validated_allocations),
                confidence=self._calculate_decision_confidence(validated_allocations, market_conditions),
                decision_factors=[
                    f"market_regime_{market_conditions.regime.value}",
                    f"volatility_{market_conditions.volatility:.3f}",
                    f"strategies_active_{len(validated_allocations)}"
                ],
                reasoning=self._generate_decision_reasoning(validated_allocations, market_conditions),
                orchestration_mode=self.orchestration_mode
            )
            
            # 9. State aktualisieren
            self.current_allocation = validated_allocations
            self.orchestration_history.append(decision)
            
            # 10. Ergebnis loggen
            await self._log_orchestration_decision(decision)
            
            return decision
            
        except Exception as e:
            logger.error(f"❌ Error in strategy orchestration: {e}")
            return self._create_empty_decision(self.market_detector._get_default_conditions())
    
    async def _filter_strategies_by_market(self, strategies: List[StrategyDNA],
                                         market_conditions: MarketConditions) -> List[StrategyDNA]:
        """Filtert Strategien basierend auf Marktbedingungen"""
        
        suitable = []
        
        for strategy in strategies:
            suitability_score = await self._calculate_market_suitability(strategy, market_conditions)
            
            if suitability_score >= 0.4:  # Mindest-Eignung
                suitable.append(strategy)
                logger.debug(f"✅ {strategy.name}: suitability {suitability_score:.2f}")
            else:
                logger.debug(f"❌ {strategy.name}: suitability {suitability_score:.2f} (filtered out)")
        
        return suitable
    
    async def _calculate_market_suitability(self, strategy: StrategyDNA,
                                          market_conditions: MarketConditions) -> float:
        """Berechnet Eignung einer Strategie für aktuelle Marktbedingungen"""
        
        suitability = 0.5  # Base score
        
        # 1. Volatilitäts-Match
        if market_conditions.volatility >= strategy.minimum_volatility:
            suitability += 0.2
        else:
            suitability -= 0.3  # Penalty für zu niedrige Volatilität
        
        # 2. Risk-Level Match mit Market Regime
        regime = market_conditions.regime
        
        if regime in [MarketRegime.BULL_TRENDING, MarketRegime.EUPHORIA]:
            # Bull Market - aggressive Strategien bevorzugt
            risk_bonus = {
                'conservative': -0.1,
                'moderate': 0.0,
                'aggressive': 0.2,
                'extreme': 0.1  # Extreme nicht immer optimal
            }
            suitability += risk_bonus.get(strategy.risk_level, 0)
            
        elif regime in [MarketRegime.BEAR_TRENDING, MarketRegime.CRASH]:
            # Bear Market - defensive Strategien bevorzugt
            risk_bonus = {
                'conservative': 0.2,
                'moderate': 0.1,
                'aggressive': -0.1,
                'extreme': -0.3
            }
            suitability += risk_bonus.get(strategy.risk_level, 0)
        
        # 3. Timeframe Match
        if regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.CRASH]:
            # Hohe Volatilität - schnelle Strategien bevorzugt
            timeframe_bonus = {
                'scalping': 0.2,
                'intraday': 0.1,
                'swing': -0.1,
                'position': -0.2
            }
            suitability += timeframe_bonus.get(strategy.timeframe, 0)
        
        # 4. Signal Source Match  
        signal_bonus = 0.0
        
        if 'ml' in strategy.signal_sources and market_conditions.volatility > 0.03:
            signal_bonus += 0.1  # ML gut bei hoher Volatilität
        
        if 'sentiment' in strategy.signal_sources and abs(market_conditions.sentiment_score) > 0.3:
            signal_bonus += 0.1  # Sentiment-Strategien bei starkem Sentiment
        
        if 'arbitrage' in strategy.signal_sources and market_conditions.volatility > 0.02:
            signal_bonus += 0.1  # Arbitrage bei Volatilität
        
        suitability += signal_bonus
        
        # 5. Market Conditions Match
        conditions_match = 0.0
        
        for condition in strategy.market_conditions:
            if condition == 'volatile' and market_conditions.volatility > 0.03:
                conditions_match += 0.1
            elif condition == 'trending' and abs(market_conditions.trend_strength) > 0.3:
                conditions_match += 0.1
            elif condition == 'ranging' and abs(market_conditions.trend_strength) < 0.2:
                conditions_match += 0.1
        
        suitability += conditions_match
        
        # 6. Confidence-basierte Anpassung
        if strategy.confidence_level < 0.5:
            suitability -= 0.1  # Penalty für ungetestete Strategien
        
        return max(0.0, min(1.0, suitability))
    
    async def _resolve_strategy_conflicts(self, allocations: List[StrategyAllocation]) -> List[StrategyAllocation]:
        """Löst Konflikte zwischen Strategien"""
        
        if not allocations:
            return allocations
        
        resolved = []
        conflicts_found = []
        
        for allocation in allocations:
            strategy_dna = self.discovery_engine.discovered_strategies.get(allocation.strategy_name)
            
            if not strategy_dna:
                continue
            
            # Prüfe auf Konflikte mit bereits hinzugefügten Strategien
            has_conflict = False
            
            for existing in resolved:
                existing_dna = self.discovery_engine.discovered_strategies.get(existing.strategy_name)
                
                if (existing_dna and 
                    allocation.strategy_name in existing_dna.conflict_strategies):
                    
                    # Konflikt gefunden - wähle bessere Strategie
                    if allocation.confidence_score > existing.confidence_score:
                        # Neue Strategie ist besser
                        resolved.remove(existing)
                        conflicts_found.append((existing.strategy_name, allocation.strategy_name, 'replaced'))
                        resolved.append(allocation)
                    else:
                        # Behalte existierende Strategie
                        conflicts_found.append((allocation.strategy_name, existing.strategy_name, 'skipped'))
                    
                    has_conflict = True
                    break
            
            if not has_conflict:
                resolved.append(allocation)
        
        if conflicts_found:
            logger.info(f"🔧 Resolved {len(conflicts_found)} strategy conflicts")
            for conflict in conflicts_found:
                logger.debug(f"   {conflict[0]} vs {conflict[1]}: {conflict[2]}")
        
        return resolved
    
    async def _validate_allocations(self, allocations: List[StrategyAllocation],
                                  market_conditions: MarketConditions,
                                  risk_budget: float) -> List[StrategyAllocation]:
        """Validiert und normalisiert Allokationen"""
        
        if not allocations:
            return allocations
        
        # 1. Entferne zu kleine Allokationen
        min_allocation = 5.0  # Mindestens 5% Allokation
        significant_allocations = [a for a in allocations if a.allocation_percent >= min_allocation]
        
        # 2. Begrenze Anzahl aktiver Strategien
        if len(significant_allocations) > self.max_strategies_active:
            significant_allocations = sorted(significant_allocations, 
                                           key=lambda x: x.confidence_score, 
                                           reverse=True)[:self.max_strategies_active]
        
        # 3. Normalisiere Allokationen auf Risk Budget
        total_allocation = sum(a.allocation_percent for a in significant_allocations)
        
        if total_allocation > 0:
            normalization_factor = min(100.0, risk_budget) / total_allocation
            
            for allocation in significant_allocations:
                allocation.allocation_percent *= normalization_factor
                allocation.position_size_multiplier *= normalization_factor
                allocation.risk_multiplier *= normalization_factor
        
        # 4. Validiere einzelne Allokationen
        validated = []
        
        for allocation in significant_allocations:
            if await self._is_allocation_valid(allocation, market_conditions):
                validated.append(allocation)
            else:
                logger.warning(f"⚠️ Invalid allocation filtered out: {allocation.strategy_name}")
        
        return validated
    
    async def _is_allocation_valid(self, allocation: StrategyAllocation,
                                 market_conditions: MarketConditions) -> bool:
        """Prüft ob eine Allokation gültig ist"""
        
        # 1. Mindest-Allokation
        if allocation.allocation_percent < 1.0:
            return False
        
        # 2. Risiko-Limits
        if allocation.risk_score > 0.9 and market_conditions.regime in [MarketRegime.CRASH, MarketRegime.BEAR_TRENDING]:
            return False
        
        # 3. Strategie-spezifische Validierung
        strategy_dna = self.discovery_engine.discovered_strategies.get(allocation.strategy_name)
        
        if strategy_dna:
            # Volatilitäts-Check
            if market_conditions.volatility < strategy_dna.minimum_volatility:
                return False
            
            # Confidence-Check
            if strategy_dna.confidence_level < 0.3:
                return False
        
        return True
    
    def _identify_deactivated_strategies(self, new_allocations: List[StrategyAllocation]) -> List[str]:
        """Identifiziert Strategien die deaktiviert werden"""
        
        new_strategy_names = {a.strategy_name for a in new_allocations}
        current_strategy_names = {a.strategy_name for a in self.current_allocation}
        
        deactivated = list(current_strategy_names - new_strategy_names)
        
        return deactivated
    
    def _calculate_decision_confidence(self, allocations: List[StrategyAllocation],
                                     market_conditions: MarketConditions) -> float:
        """Berechnet Vertrauen in die Orchestrierungs-Entscheidung"""
        
        if not allocations:
            return 0.1
        
        # 1. Market Conditions Confidence
        market_confidence = market_conditions.confidence
        
        # 2. Strategy Confidence (gewichteter Durchschnitt)
        total_allocation = sum(a.allocation_percent for a in allocations)
        if total_allocation == 0:
            return 0.1
        
        weighted_strategy_confidence = sum(
            a.confidence_score * (a.allocation_percent / total_allocation)
            for a in allocations
        )
        
        # 3. Diversifikation Bonus
        diversification_bonus = min(len(allocations) * 0.1, 0.3)
        
        # 4. Kombination
        overall_confidence = (
            market_confidence * 0.4 +
            weighted_strategy_confidence * 0.5 +
            diversification_bonus * 0.1
        )
        
        return max(0.1, min(1.0, overall_confidence))
    
    def _generate_decision_reasoning(self, allocations: List[StrategyAllocation],
                                   market_conditions: MarketConditions) -> str:
        """Generiert menschenlesbare Begründung für Entscheidung"""
        
        if not allocations:
            return f"No suitable strategies for {market_conditions.regime.value} market conditions"
        
        regime_desc = {
            MarketRegime.BULL_TRENDING: "bullish trending market",
            MarketRegime.BEAR_TRENDING: "bearish trending market", 
            MarketRegime.HIGH_VOLATILITY: "high volatility environment",
            MarketRegime.CRASH: "market crash conditions",
            MarketRegime.EUPHORIA: "euphoric bull market",
            MarketRegime.UNCERTAINTY: "uncertain market conditions"
        }.get(market_conditions.regime, "current market conditions")
        
        top_strategy = max(allocations, key=lambda x: x.allocation_percent)
        
        reasoning = f"Selected {len(allocations)} strategies for {regime_desc}. "
        reasoning += f"Primary allocation: {top_strategy.strategy_name} "
        reasoning += f"({top_strategy.allocation_percent:.1f}%) due to "
        reasoning += f"{top_strategy.market_fit_score:.1f} market fit score. "
        
        if market_conditions.volatility > 0.03:
            reasoning += "High volatility environment favors diversified approach. "
        
        if len(allocations) > 1:
            reasoning += f"Portfolio diversified across {len(allocations)} strategies "
            reasoning += "to optimize risk-adjusted returns."
        
        return reasoning
    
    def _create_empty_decision(self, market_conditions: MarketConditions) -> OrchestrationDecision:
        """Erstellt leere Entscheidung als Fallback"""
        
        return OrchestrationDecision(
            timestamp=datetime.now(),
            market_conditions=market_conditions,
            active_strategies=[],
            deactivated_strategies=[s.strategy_name for s in self.current_allocation],
            total_risk_budget=0.0,
            expected_portfolio_return=0.0,
            confidence=0.1,
            decision_factors=['no_suitable_strategies'],
            reasoning="No suitable strategies found for current market conditions",
            orchestration_mode=self.orchestration_mode
        )
    
    async def _log_orchestration_decision(self, decision: OrchestrationDecision):
        """Loggt Orchestrierungs-Entscheidung"""
        
        logger.info("🎼 ORCHESTRATION DECISION MADE")
        logger.info("=" * 35)
        logger.info(f"Market Regime: {decision.market_conditions.regime.value}")
        logger.info(f"Active Strategies: {len(decision.active_strategies)}")
        logger.info(f"Total Allocation: {sum(a.allocation_percent for a in decision.active_strategies):.1f}%")
        logger.info(f"Expected Return: {decision.expected_portfolio_return:.2f}")
        logger.info(f"Confidence: {decision.confidence:.2f}")
        
        if decision.active_strategies:
            logger.info("Strategy Allocations:")
            for allocation in sorted(decision.active_strategies, 
                                   key=lambda x: x.allocation_percent, reverse=True):
                logger.info(f"  📈 {allocation.strategy_name}: {allocation.allocation_percent:.1f}% "
                           f"(fit: {allocation.market_fit_score:.2f})")
        
        if decision.deactivated_strategies:
            logger.info(f"Deactivated: {', '.join(decision.deactivated_strategies)}")
        
        logger.info(f"Reasoning: {decision.reasoning}")
    
    def start_continuous_orchestration(self, market_data_callback: Callable,
                                     risk_budget: float = 100.0):
        """Startet kontinuierliche Orchestrierung in separatem Thread"""
        
        if self.orchestration_running:
            logger.warning("⚠️ Orchestration already running")
            return
        
        self.orchestration_running = True
        self._stop_event.clear()
        
        def orchestration_loop():
            logger.info("🎼 Starting continuous orchestration loop")
            
            while not self._stop_event.is_set():
                try:
                    # Market Data abrufen
                    market_data = market_data_callback()
                    
                    if market_data:
                        # Orchestrierung ausführen
                        decision = asyncio.run(self.orchestrate_strategies(market_data, risk_budget))
                        
                        # Performance tracken
                        self._track_orchestration_performance(decision)
                    
                    # Warte bis zum nächsten Rebalancing
                    wait_time = self.rebalance_interval_hours * 3600  # Sekunden
                    self._stop_event.wait(wait_time)
                    
                except Exception as e:
                    logger.error(f"❌ Error in orchestration loop: {e}")
                    self._stop_event.wait(300)  # 5 Minuten warten bei Fehler
            
            logger.info("🎼 Orchestration loop stopped")
        
        self.orchestration_thread = threading.Thread(target=orchestration_loop, daemon=True)
        self.orchestration_thread.start()
        
        logger.info("🎼 Continuous orchestration started")
    
    def stop_continuous_orchestration(self):
        """Stoppt kontinuierliche Orchestrierung"""
        
        if not self.orchestration_running:
            return
        
        self._stop_event.set()
        self.orchestration_running = False
        
        if self.orchestration_thread:
            self.orchestration_thread.join(timeout=10)
        
        logger.info("🎼 Continuous orchestration stopped")
    
    def _track_orchestration_performance(self, decision: OrchestrationDecision):
        """Trackt Performance der Orchestrierung"""
        
        timestamp = decision.timestamp
        
        # Performance-Metriken sammeln
        performance_data = {
            'timestamp': timestamp,
            'active_strategies_count': len(decision.active_strategies),
            'total_allocation': sum(a.allocation_percent for a in decision.active_strategies),
            'expected_return': decision.expected_portfolio_return,
            'confidence': decision.confidence,
            'market_regime': decision.market_conditions.regime.value,
            'volatility': decision.market_conditions.volatility
        }
        
        self.performance_tracker['orchestration_decisions'].append(performance_data)
        
        # Nur letzte 1000 Entscheidungen behalten
        if len(self.performance_tracker['orchestration_decisions']) > 1000:
            self.performance_tracker['orchestration_decisions'] = \
                self.performance_tracker['orchestration_decisions'][-1000:]
    
    def get_orchestration_stats(self) -> Dict[str, Any]:
        """Gibt Orchestrierungs-Statistiken zurück"""
        
        decisions = self.performance_tracker.get('orchestration_decisions', [])
        
        if not decisions:
            return {'no_data': True}
        
        # Basis-Statistiken
        total_decisions = len(decisions)
        avg_strategies_active = np.mean([d['active_strategies_count'] for d in decisions])
        avg_confidence = np.mean([d['confidence'] for d in decisions])
        avg_expected_return = np.mean([d['expected_return'] for d in decisions])
        
        # Regime-Verteilung
        regime_counts = defaultdict(int)
        for decision in decisions:
            regime_counts[decision['market_regime']] += 1
        
        # Letzte 24 Stunden Performance
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_decisions = [
            d for d in decisions 
            if isinstance(d['timestamp'], datetime) and d['timestamp'] > recent_cutoff
        ]
        
        return {
            'total_decisions': total_decisions,
            'avg_strategies_active': avg_strategies_active,
            'avg_confidence': avg_confidence,
            'avg_expected_return': avg_expected_return,
            'regime_distribution': dict(regime_counts),
            'recent_decisions_24h': len(recent_decisions),
            'current_allocation': len(self.current_allocation),
            'orchestration_running': self.orchestration_running,
            'last_decision': decisions[-1]['timestamp'] if decisions else None
        }

# Factory-Funktion
def create_orchestration_engine(discovery_engine: StrategyDiscoveryEngine = None) -> IntelligentOrchestrationEngine:
    """Erstellt Intelligent Orchestration Engine"""
    return IntelligentOrchestrationEngine(discovery_engine)

# Test-Funktion
async def test_orchestration_engine():
    """Testet das Orchestration System"""
    
    print("🎼 TESTE INTELLIGENT ORCHESTRATION ENGINE")
    print("=" * 45)
    
    try:
        # 1. Discovery Engine erstellen (Mock)
        from .strategy_orchestrator import create_strategy_discovery_engine
        discovery_engine = create_strategy_discovery_engine()
        
        # Mock-Strategien hinzufügen
        await discovery_engine.discover_all_strategies()
        
        # 2. Orchestration Engine erstellen
        orchestration_engine = create_orchestration_engine(discovery_engine)
        
        # 3. Mock Market Data erstellen
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        mock_market_data = {
            'BTC/USDT': pd.DataFrame({
                'open': np.random.normal(45000, 1000, 100),
                'high': np.random.normal(45500, 1000, 100),
                'low': np.random.normal(44500, 1000, 100),
                'close': np.random.normal(45000, 1000, 100),
                'volume': np.random.normal(1000000, 200000, 100)
            }, index=dates)
        }
        
        # 4. Orchestrierung testen
        print("🎯 Testing strategy orchestration...")
        decision = await orchestration_engine.orchestrate_strategies(
            mock_market_data, risk_budget=100.0
        )
        
        print(f"✅ Orchestration completed!")
        print(f"   Market Regime: {decision.market_conditions.regime.value}")
        print(f"   Active Strategies: {len(decision.active_strategies)}")
        print(f"   Confidence: {decision.confidence:.2f}")
        print(f"   Expected Return: {decision.expected_portfolio_return:.2f}")
        
        # 5. Market Regime Detection testen
        print(f"\n🔍 Testing market regime detection...")
        detector = MarketRegimeDetector()
        conditions = detector.detect_market_conditions(mock_market_data)
        
        print(f"   Detected Regime: {conditions.regime.value}")
        print(f"   Volatility: {conditions.volatility:.4f}")
        print(f"   Trend Strength: {conditions.trend_strength:+.2f}")
        print(f"   RSI: {conditions.rsi_14:.1f}")
        
        # 6. Statistiken abrufen
        stats = orchestration_engine.get_orchestration_stats()
        print(f"\n📊 Orchestration Stats:")
        for key, value in stats.items():
            if key != 'regime_distribution':
                print(f"   {key}: {value}")
        
        print(f"\n🎉 ORCHESTRATION ENGINE TEST ERFOLGREICH!")
        return True
        
    except Exception as e:
        print(f"❌ Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Orchestration Engine testen
    asyncio.run(test_orchestration_engine())