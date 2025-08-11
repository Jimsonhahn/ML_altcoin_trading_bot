"""
Smart Rebalancing Strategy - Intelligente Portfolio-Rebalancing
Fokussiert auf optimale Rebalancing-Timing und Kostenminimierung
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .strategy_base import Strategy
from core.interfaces import IStrategy
from utils.exceptions import StrategyError

logger = logging.getLogger(__name__)


class RebalanceReason(Enum):
    """Gründe für Rebalancing"""
    THRESHOLD_BREACH = "threshold_breach"
    TIME_BASED = "time_based"
    VOLATILITY_REGIME = "volatility_regime"
    MOMENTUM_SHIFT = "momentum_shift"
    RISK_MANAGEMENT = "risk_management"


@dataclass
class RebalanceOpportunity:
    """Rebalancing-Gelegenheit mit Kosten-Nutzen-Analyse"""
    asset: str
    current_weight: float
    target_weight: float
    weight_deviation: float
    expected_benefit: float
    transaction_cost: float
    net_benefit: float
    urgency: float  # 0-1 Skala


class SmartRebalancingStrategy(Strategy):
    """
    Intelligente Rebalancing-Strategie mit:
    - Kostenoptimiertes Rebalancing
    - Momentum-bewusste Allokation
    - Volatilitäts-Timing
    - Transaktionskosten-Minimierung
    """
    
    def __init__(self, config: Dict[str, Any], ml_components: Optional[Any] = None):
        super().__init__(config, ml_components)
        self.name = "smart_rebalancing"
        
        # Rebalancing-Parameter
        self.target_weights = config.get('target_weights', {})
        self.rebalance_threshold = config.get('rebalance_threshold', 0.05)  # 5%
        self.min_rebalance_interval = config.get('min_rebalance_interval', 6)  # Stunden
        self.max_rebalance_interval = config.get('max_rebalance_interval', 168)  # 1 Woche
        
        # Kostenparameter
        self.transaction_cost_rate = config.get('transaction_cost_rate', 0.001)  # 0.1%
        self.min_trade_amount = config.get('min_trade_amount', 50.0)  # $50 Minimum
        self.cost_benefit_threshold = config.get('cost_benefit_threshold', 2.0)  # 2:1 Nutzen
        
        # Volatilitäts-Timing
        self.vol_lookback = config.get('vol_lookback', 24)  # 24 Stunden
        self.low_vol_multiplier = config.get('low_vol_multiplier', 0.7)  # Weniger aggressiv
        self.high_vol_multiplier = config.get('high_vol_multiplier', 1.5)  # Mehr aggressiv
        
        # Momentum-Integration
        self.momentum_lookback = config.get('momentum_lookback', 72)  # 3 Tage
        self.momentum_threshold = config.get('momentum_threshold', 0.02)  # 2%
        self.momentum_adjustment = config.get('momentum_adjustment', 0.2)  # 20% Anpassung
        
        # State
        self.last_rebalance_time: Dict[str, datetime] = {}
        self.rebalance_history: List[Dict] = []
        self.current_weights: Dict[str, float] = {}
        self.pending_rebalances: List[RebalanceOpportunity] = []
        
        logger.info("Smart Rebalancing Strategy initialisiert")
    
    async def calculate_signal(self, symbol: str, data: pd.DataFrame, current_price: float) -> Tuple[str, Dict[str, Any]]:
        """
        Berechnet intelligentes Rebalancing-Signal
        """
        try:
            # Aktuelle Portfolio-Situation analysieren
            portfolio_analysis = await self._analyze_portfolio_state(symbol, data)
            
            # Rebalancing-Gelegenheiten identifizieren
            rebalance_opportunities = self._identify_rebalance_opportunities(
                symbol, portfolio_analysis, current_price
            )
            
            # Kosten-Nutzen-Analyse durchführen
            optimal_opportunity = self._select_optimal_rebalance(rebalance_opportunities)
            
            if not optimal_opportunity:
                return 'HOLD', {
                    'reason': 'no_profitable_rebalance',
                    'confidence': 0.3,
                    'portfolio_analysis': portfolio_analysis
                }
            
            # Timing-Optimierung
            timing_analysis = self._analyze_rebalance_timing(data, optimal_opportunity)
            
            # Finales Signal generieren
            signal_type, signal_data = self._generate_rebalance_signal(
                symbol, optimal_opportunity, timing_analysis
            )
            
            return signal_type, signal_data
            
        except Exception as e:
            logger.error(f"Fehler bei Smart Rebalancing für {symbol}: {e}")
            return 'HOLD', {'error': str(e), 'confidence': 0.0}
    
    async def _analyze_portfolio_state(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analysiert aktuellen Portfolio-Zustand
        """
        try:
            # Aktuelle Gewichtung (vereinfacht)
            current_weight = self.current_weights.get(symbol, 0.0)
            target_weight = self.target_weights.get(symbol, 0.0)
            
            # Abweichung berechnen
            weight_deviation = abs(current_weight - target_weight)
            deviation_percentage = weight_deviation / max(target_weight, 0.01)
            
            # Momentum-Analyse
            returns = data['close'].pct_change().dropna()
            momentum = returns.rolling(self.momentum_lookback).mean().iloc[-1]
            momentum_strength = abs(momentum) / returns.rolling(self.momentum_lookback).std().iloc[-1]
            
            # Volatilitäts-Analyse
            volatility = returns.rolling(self.vol_lookback).std().iloc[-1] * np.sqrt(24)
            vol_percentile = self._calculate_volatility_percentile(data, volatility)
            
            # Zeit seit letztem Rebalancing
            last_rebalance = self.last_rebalance_time.get(symbol, datetime.min)
            hours_since_rebalance = (datetime.now() - last_rebalance).total_seconds() / 3600
            
            return {
                'symbol': symbol,
                'current_weight': current_weight,
                'target_weight': target_weight,
                'weight_deviation': weight_deviation,
                'deviation_percentage': deviation_percentage,
                'momentum': momentum,
                'momentum_strength': momentum_strength,
                'volatility': volatility,
                'vol_percentile': vol_percentile,
                'hours_since_rebalance': hours_since_rebalance,
                'needs_rebalance': self._needs_rebalancing(symbol, weight_deviation, hours_since_rebalance)
            }
            
        except Exception as e:
            logger.error(f"Fehler bei Portfolio-Analyse für {symbol}: {e}")
            return {'error': str(e)}
    
    def _calculate_volatility_percentile(self, data: pd.DataFrame, current_vol: float) -> float:
        """
        Berechnet Volatilitäts-Perzentil für Timing
        """
        try:
            returns = data['close'].pct_change().dropna()
            
            # Rolling Volatility über längeren Zeitraum
            rolling_vol = returns.rolling(self.vol_lookback).std() * np.sqrt(24)
            rolling_vol = rolling_vol.dropna()
            
            if len(rolling_vol) < 50:
                return 0.5  # Neutral wenn nicht genug Daten
            
            # Perzentil berechnen
            percentile = (rolling_vol < current_vol).mean()
            return percentile
            
        except Exception as e:
            logger.error(f"Fehler bei Volatilitäts-Perzentil: {e}")
            return 0.5
    
    def _needs_rebalancing(self, symbol: str, weight_deviation: float, hours_since_rebalance: float) -> bool:
        """
        Prüft ob Rebalancing benötigt wird
        """
        # Schwellwert-basiert
        threshold_breach = weight_deviation > self.rebalance_threshold
        
        # Zeit-basiert
        min_time_met = hours_since_rebalance >= self.min_rebalance_interval
        max_time_exceeded = hours_since_rebalance >= self.max_rebalance_interval
        
        return (threshold_breach and min_time_met) or max_time_exceeded
    
    def _identify_rebalance_opportunities(self, symbol: str, portfolio_analysis: Dict[str, Any], 
                                        current_price: float) -> List[RebalanceOpportunity]:
        """
        Identifiziert profitable Rebalancing-Gelegenheiten
        """
        opportunities = []
        
        try:
            if 'error' in portfolio_analysis or not portfolio_analysis.get('needs_rebalance', False):
                return opportunities
            
            current_weight = portfolio_analysis['current_weight']
            target_weight = portfolio_analysis['target_weight']
            weight_deviation = portfolio_analysis['weight_deviation']
            
            # Momentum-Anpassung der Zielgewichtung
            momentum_adjusted_target = self._calculate_momentum_adjusted_target(
                symbol, target_weight, portfolio_analysis['momentum'], 
                portfolio_analysis['momentum_strength']
            )
            
            # Trade-Größe berechnen
            portfolio_value = self._estimate_portfolio_value()  # Vereinfacht
            trade_value = abs(momentum_adjusted_target - current_weight) * portfolio_value
            
            if trade_value < self.min_trade_amount:
                return opportunities  # Zu kleiner Trade
            
            # Erwarteter Nutzen
            expected_benefit = self._calculate_expected_benefit(
                weight_deviation, portfolio_analysis['momentum'], trade_value
            )
            
            # Transaktionskosten
            transaction_cost = trade_value * self.transaction_cost_rate
            
            # Net Benefit
            net_benefit = expected_benefit - transaction_cost
            
            # Dringlichkeit
            urgency = self._calculate_urgency(portfolio_analysis)
            
            if net_benefit > transaction_cost * (self.cost_benefit_threshold - 1):
                opportunity = RebalanceOpportunity(
                    asset=symbol,
                    current_weight=current_weight,
                    target_weight=momentum_adjusted_target,
                    weight_deviation=abs(momentum_adjusted_target - current_weight),
                    expected_benefit=expected_benefit,
                    transaction_cost=transaction_cost,
                    net_benefit=net_benefit,
                    urgency=urgency
                )
                opportunities.append(opportunity)
            
        except Exception as e:
            logger.error(f"Fehler bei Opportunity-Identifikation für {symbol}: {e}")
        
        return opportunities
    
    def _calculate_momentum_adjusted_target(self, symbol: str, base_target: float, 
                                          momentum: float, momentum_strength: float) -> float:
        """
        Passt Zielgewichtung basierend auf Momentum an
        """
        try:
            # Nur bei starkem Momentum anpassen
            if momentum_strength < 1.0:  # Momentum nicht stark genug
                return base_target
            
            # Momentum-Richtung bestimmen
            if abs(momentum) < self.momentum_threshold:
                return base_target  # Kein signifikantes Momentum
            
            # Anpassung berechnen
            momentum_factor = np.sign(momentum) * min(abs(momentum) / self.momentum_threshold, 2.0)
            adjustment = momentum_factor * self.momentum_adjustment * base_target
            
            # Grenzen einhalten
            adjusted_target = base_target + adjustment
            return max(0.0, min(adjusted_target, 1.0))  # 0-100%
            
        except Exception as e:
            logger.error(f"Fehler bei Momentum-Anpassung für {symbol}: {e}")
            return base_target
    
    def _estimate_portfolio_value(self) -> float:
        """
        Schätzt aktuellen Portfolio-Wert (vereinfacht)
        """
        # In Realität: Aus Portfolio-Manager abrufen
        return 100000.0  # $100k Default
    
    def _calculate_expected_benefit(self, weight_deviation: float, momentum: float, trade_value: float) -> float:
        """
        Berechnet erwarteten Nutzen des Rebalancing
        """
        try:
            # Base Benefit aus Gewichtungskorrektur
            rebalance_benefit = weight_deviation * trade_value * 0.1  # 10% des Deviations-Werts
            
            # Momentum Benefit
            momentum_benefit = abs(momentum) * trade_value * 0.05  # 5% bei starkem Momentum
            
            # Risk Reduction Benefit
            risk_benefit = weight_deviation * trade_value * 0.02  # 2% Risikoreduktion
            
            return rebalance_benefit + momentum_benefit + risk_benefit
            
        except Exception as e:
            logger.error(f"Fehler bei Benefit-Berechnung: {e}")
            return 0.0
    
    def _calculate_urgency(self, portfolio_analysis: Dict[str, Any]) -> float:
        """
        Berechnet Dringlichkeit des Rebalancing (0-1)
        """
        try:
            deviation_urgency = min(portfolio_analysis['deviation_percentage'] / 0.2, 1.0)
            time_urgency = min(portfolio_analysis['hours_since_rebalance'] / self.max_rebalance_interval, 1.0)
            vol_urgency = 1.0 - portfolio_analysis['vol_percentile']  # Niedriger Vol = höhere Urgency
            
            # Gewichteter Durchschnitt
            urgency = (deviation_urgency * 0.5 + time_urgency * 0.3 + vol_urgency * 0.2)
            return max(0.0, min(urgency, 1.0))
            
        except Exception as e:
            logger.error(f"Fehler bei Urgency-Berechnung: {e}")
            return 0.5
    
    def _select_optimal_rebalance(self, opportunities: List[RebalanceOpportunity]) -> Optional[RebalanceOpportunity]:
        """
        Wählt optimale Rebalancing-Gelegenheit aus
        """
        if not opportunities:
            return None
        
        try:
            # Score basierend auf Net Benefit und Urgency
            for opp in opportunities:
                opp.score = opp.net_benefit * (1 + opp.urgency)
            
            # Beste Gelegenheit auswählen
            best_opportunity = max(opportunities, key=lambda x: x.score)
            
            # Minimum-Score prüfen
            if best_opportunity.score > 0:
                return best_opportunity
            
        except Exception as e:
            logger.error(f"Fehler bei Opportunity-Auswahl: {e}")
        
        return None
    
    def _analyze_rebalance_timing(self, data: pd.DataFrame, 
                                opportunity: RebalanceOpportunity) -> Dict[str, Any]:
        """
        Analysiert optimales Timing für Rebalancing
        """
        try:
            returns = data['close'].pct_change().dropna()
            
            # Volatilitäts-Timing
            current_vol = returns.rolling(self.vol_lookback).std().iloc[-1]
            vol_percentile = self._calculate_volatility_percentile(data, current_vol)
            
            # Timing-Score (niedriger Vol = besseres Timing)
            vol_timing_score = 1.0 - vol_percentile
            
            # Momentum-Timing
            recent_momentum = returns.tail(12).mean()  # Letzten 12 Stunden
            momentum_alignment = 1.0 if np.sign(recent_momentum) == np.sign(opportunity.target_weight - opportunity.current_weight) else 0.5
            
            # Market Session (vereinfacht)
            current_hour = datetime.now().hour
            session_score = 1.0 if 8 <= current_hour <= 16 else 0.7  # Haupthandelszeit
            
            # Gesamt-Timing-Score
            timing_score = (vol_timing_score * 0.5 + momentum_alignment * 0.3 + session_score * 0.2)
            
            return {
                'vol_timing_score': vol_timing_score,
                'momentum_alignment': momentum_alignment,
                'session_score': session_score,
                'overall_timing_score': timing_score,
                'is_good_timing': timing_score > 0.6
            }
            
        except Exception as e:
            logger.error(f"Fehler bei Timing-Analyse: {e}")
            return {'overall_timing_score': 0.5, 'is_good_timing': True}
    
    def _generate_rebalance_signal(self, symbol: str, opportunity: RebalanceOpportunity, 
                                 timing_analysis: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Generiert finales Rebalancing-Signal
        """
        try:
            # Signal-Richtung bestimmen
            if opportunity.target_weight > opportunity.current_weight:
                signal_type = 'BUY'
            elif opportunity.target_weight < opportunity.current_weight:
                signal_type = 'SELL'
            else:
                signal_type = 'HOLD'
            
            # Confidence basierend auf Net Benefit und Timing
            base_confidence = min(opportunity.net_benefit / (opportunity.transaction_cost * 2), 1.0)
            timing_adjustment = timing_analysis['overall_timing_score']
            confidence = base_confidence * timing_adjustment
            
            # Position Size aus Gewichtungsdifferenz
            weight_difference = abs(opportunity.target_weight - opportunity.current_weight)
            
            signal_data = {
                'strategy': self.name,
                'confidence': confidence,
                'position_size': weight_difference,
                'reason': RebalanceReason.THRESHOLD_BREACH.value,
                'current_weight': opportunity.current_weight,
                'target_weight': opportunity.target_weight,
                'expected_benefit': opportunity.expected_benefit,
                'transaction_cost': opportunity.transaction_cost,
                'net_benefit': opportunity.net_benefit,
                'urgency': opportunity.urgency,
                'timing_score': timing_analysis['overall_timing_score'],
                'is_good_timing': timing_analysis['is_good_timing']
            }
            
            # Timing-basierte Signal-Anpassung
            if not timing_analysis['is_good_timing'] and opportunity.urgency < 0.8:
                signal_type = 'HOLD'  # Warten auf besseres Timing
                signal_data['reason'] = 'waiting_for_better_timing'
                signal_data['confidence'] *= 0.3
            
            return signal_type, signal_data
            
        except Exception as e:
            logger.error(f"Fehler bei Signal-Generierung für {symbol}: {e}")
            return 'HOLD', {'error': str(e), 'confidence': 0.0}
    
    def update_portfolio_weights(self, weights: Dict[str, float]):
        """
        Aktualisiert aktuelle Portfolio-Gewichtungen
        """
        self.current_weights = weights.copy()
        logger.debug(f"Portfolio-Gewichtungen aktualisiert: {weights}")
    
    def record_rebalance(self, symbol: str, details: Dict[str, Any]):
        """
        Zeichnet Rebalancing-Aktion auf
        """
        self.last_rebalance_time[symbol] = datetime.now()
        
        rebalance_record = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'details': details
        }
        
        self.rebalance_history.append(rebalance_record)
        
        # Historie begrenzen
        if len(self.rebalance_history) > 1000:
            self.rebalance_history = self.rebalance_history[-500:]
    
    def get_rebalancing_metrics(self) -> Dict[str, Any]:
        """
        Gibt Rebalancing-Metriken zurück
        """
        recent_rebalances = [
            r for r in self.rebalance_history 
            if (datetime.now() - r['timestamp']).days <= 7
        ]
        
        return {
            'total_rebalances': len(self.rebalance_history),
            'recent_rebalances': len(recent_rebalances),
            'avg_rebalance_frequency_hours': self._calculate_avg_frequency(),
            'current_portfolio_weights': self.current_weights.copy(),
            'target_portfolio_weights': self.target_weights.copy(),
            'portfolio_deviation': self._calculate_portfolio_deviation(),
            'pending_opportunities': len(self.pending_rebalances)
        }
    
    def _calculate_avg_frequency(self) -> float:
        """
        Berechnet durchschnittliche Rebalancing-Frequenz
        """
        if len(self.rebalance_history) < 2:
            return 0.0
        
        timestamps = [r['timestamp'] for r in self.rebalance_history]
        intervals = [(timestamps[i] - timestamps[i-1]).total_seconds() / 3600 
                    for i in range(1, len(timestamps))]
        
        return np.mean(intervals) if intervals else 0.0
    
    def _calculate_portfolio_deviation(self) -> float:
        """
        Berechnet aktuelle Portfolio-Abweichung von Zielen
        """
        total_deviation = 0.0
        
        for symbol, target_weight in self.target_weights.items():
            current_weight = self.current_weights.get(symbol, 0.0)
            total_deviation += abs(target_weight - current_weight)
        
        return total_deviation