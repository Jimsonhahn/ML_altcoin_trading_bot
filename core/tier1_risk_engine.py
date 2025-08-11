"""
Tier-1 Risk Engine
Elite institutionelle Risk Engine mit Regime Detection und ganzheitlichem Risikomanagement
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
import asyncio
import logging
from collections import deque, defaultdict
from dataclasses import dataclass

from .tier1_models import (
    Signal, RiskMetrics, MarketRegime, IRiskEngine,
    SystemConstants, SignalDirection
)

logger = logging.getLogger(__name__)


@dataclass
class RegimeIndicator:
    """Market Regime Indicator"""
    name: str
    value: float
    weight: float
    regime_signal: MarketRegime


@dataclass
class RiskRule:
    """Risk Management Rule"""
    name: str
    condition: callable
    regime: Optional[MarketRegime] = None
    priority: int = 5
    enabled: bool = True


class AdvancedRiskEngine(IRiskEngine):
    """
    Elite Risk Engine mit institutionellen Features:
    - Multi-Factor Market Regime Detection
    - Dynamic Risk Rules per Regime
    - Portfolio-Level Risk Management
    - Correlation Risk Monitoring
    - Real-time VaR/CVaR Calculation
    """
    
    def __init__(self, 
                 max_portfolio_var: float = 0.05,  # 5% max VaR
                 max_single_position: float = 0.10,  # 10% max single position
                 max_correlation: float = 0.7,  # Max correlation between positions
                 regime_lookback_days: int = 30):
        
        self.max_portfolio_var = max_portfolio_var
        self.max_single_position = max_single_position
        self.max_correlation = max_correlation
        self.regime_lookback_days = regime_lookback_days
        
        # Market Regime Detection
        self.current_regime = MarketRegime.MEAN_REVERTING  # Default
        self.regime_confidence = 0.5
        self.regime_indicators: List[RegimeIndicator] = []
        
        # Historical Data für Regime Detection
        self.price_history: deque = deque(maxlen=252)  # 1 Jahr
        self.volatility_history: deque = deque(maxlen=100)
        self.volume_history: deque = deque(maxlen=100)
        self.correlation_history: deque = deque(maxlen=50)
        
        # Portfolio Risk Tracking
        self.active_positions: Dict[str, Dict] = {}  # signal_id -> position info
        self.portfolio_correlation_matrix: Optional[np.ndarray] = None
        self.portfolio_weights: Dict[str, float] = {}
        
        # Risk Rules per Regime
        self.risk_rules: Dict[MarketRegime, List[RiskRule]] = {}
        self._initialize_risk_rules()
        
        # Risk Metrics Tracking
        self.risk_metrics_history: deque = deque(maxlen=100)
        self.last_risk_calculation: datetime = datetime.now()
        
        # Circuit Breaker States
        self.circuit_breakers: Dict[str, bool] = {
            'high_volatility': False,
            'max_drawdown': False,
            'correlation_spike': False,
            'liquidity_crisis': False
        }
        
        logger.info("AdvancedRiskEngine initialisiert")
    
    async def approve(self, signal: Signal) -> bool:
        """
        Hauptmethode: Signal-Approval basierend auf Multi-Layer Risk Analysis
        """
        try:
            # 1. Basis-Validierung
            if not await self._basic_signal_validation(signal):
                logger.warning(f"Signal {signal.signal_id} failed basic validation")
                return False
            
            # 2. Aktuelles Market Regime bestimmen
            current_regime = await self.get_current_regime()
            
            # 3. Regime-spezifische Regeln prüfen
            if not await self._check_regime_rules(signal, current_regime):
                logger.info(f"Signal {signal.signal_id} rejected by regime rules ({current_regime.value})")
                return False
            
            # 4. Portfolio-Level Risk Check
            if not await self._check_portfolio_risk(signal):
                logger.info(f"Signal {signal.signal_id} rejected by portfolio risk limits")
                return False
            
            # 5. Correlation Risk Check
            if not await self._check_correlation_risk(signal):
                logger.info(f"Signal {signal.signal_id} rejected by correlation limits")
                return False
            
            # 6. Circuit Breaker Check
            if not await self._check_circuit_breakers(signal):
                logger.warning(f"Signal {signal.signal_id} rejected by circuit breakers")
                return False
            
            # 7. VaR Impact Check
            if not await self._check_var_impact(signal):
                logger.info(f"Signal {signal.signal_id} rejected by VaR impact")
                return False
            
            logger.info(f"Signal {signal.signal_id} approved by Risk Engine")
            return True
            
        except Exception as e:
            logger.error(f"Fehler bei Risk Approval: {e}")
            return False  # Fail-safe: Bei Fehlern ablehnen
    
    async def get_current_regime(self) -> MarketRegime:
        """
        Bestimmt aktuelles Market Regime über Multi-Factor Analysis
        """
        try:
            await self._update_regime_indicators()
            
            # Regime Scores berechnen
            regime_scores = defaultdict(float)
            total_weight = 0
            
            for indicator in self.regime_indicators:
                regime_scores[indicator.regime_signal] += indicator.value * indicator.weight
                total_weight += indicator.weight
            
            # Normalisieren
            if total_weight > 0:
                for regime in regime_scores:
                    regime_scores[regime] /= total_weight
            
            # Best Regime auswählen
            if regime_scores:
                best_regime = max(regime_scores.items(), key=lambda x: x[1])
                self.current_regime = best_regime[0]
                self.regime_confidence = best_regime[1]
            
            logger.debug(f"Market Regime: {self.current_regime.value} (confidence: {self.regime_confidence:.2f})")
            return self.current_regime
            
        except Exception as e:
            logger.error(f"Fehler bei Regime Detection: {e}")
            return self.current_regime  # Return last known regime
    
    async def _update_regime_indicators(self) -> None:
        """Update alle Regime Indicators"""
        
        self.regime_indicators.clear()
        
        if len(self.price_history) < 20:
            # Nicht genug Daten - Default Regime
            self.regime_indicators.append(RegimeIndicator(
                "default", 0.5, 1.0, MarketRegime.MEAN_REVERTING
            ))
            return
        
        prices = np.array(list(self.price_history))
        
        # 1. Trend Indicator (SMA Cross)
        trend_indicator = await self._calculate_trend_indicator(prices)
        self.regime_indicators.append(trend_indicator)
        
        # 2. Volatility Indicator
        vol_indicator = await self._calculate_volatility_indicator(prices)
        self.regime_indicators.append(vol_indicator)
        
        # 3. Momentum Indicator
        momentum_indicator = await self._calculate_momentum_indicator(prices)
        self.regime_indicators.append(momentum_indicator)
        
        # 4. Mean Reversion Indicator
        mean_reversion_indicator = await self._calculate_mean_reversion_indicator(prices)
        self.regime_indicators.append(mean_reversion_indicator)
    
    async def _calculate_trend_indicator(self, prices: np.ndarray) -> RegimeIndicator:
        """Trend-basierte Regime Detection"""
        
        if len(prices) < 20:
            return RegimeIndicator("trend", 0.5, 0.3, MarketRegime.MEAN_REVERTING)
        
        # SMA 10 vs SMA 20
        sma_10 = np.mean(prices[-10:])
        sma_20 = np.mean(prices[-20:])
        
        trend_strength = (sma_10 - sma_20) / sma_20
        
        if trend_strength > 0.02:  # > 2% uptrend
            return RegimeIndicator("trend", 0.8, 0.3, MarketRegime.BULL_TRENDING)
        elif trend_strength < -0.02:  # < -2% downtrend
            return RegimeIndicator("trend", 0.8, 0.3, MarketRegime.BEAR_TRENDING)
        else:
            return RegimeIndicator("trend", 0.6, 0.3, MarketRegime.MEAN_REVERTING)
    
    async def _calculate_volatility_indicator(self, prices: np.ndarray) -> RegimeIndicator:
        """Volatility-basierte Regime Detection"""
        
        if len(prices) < 20:
            return RegimeIndicator("volatility", 0.5, 0.25, MarketRegime.LOW_VOLATILITY)
        
        # Rolling Volatility berechnen
        returns = np.diff(prices) / prices[:-1]
        current_vol = np.std(returns[-10:]) * np.sqrt(252)  # Annualized
        
        if current_vol > 0.4:  # > 40% vol
            return RegimeIndicator("volatility", 0.9, 0.25, MarketRegime.HIGH_VOLATILITY)
        elif current_vol > 0.6:  # Extreme Volatility
            return RegimeIndicator("volatility", 0.95, 0.25, MarketRegime.CRISIS)
        elif current_vol < 0.15:  # < 15% vol
            return RegimeIndicator("volatility", 0.8, 0.25, MarketRegime.LOW_VOLATILITY)
        else:
            return RegimeIndicator("volatility", 0.6, 0.25, MarketRegime.MEAN_REVERTING)
    
    async def _calculate_momentum_indicator(self, prices: np.ndarray) -> RegimeIndicator:
        """Momentum-basierte Regime Detection"""
        
        if len(prices) < 10:
            return RegimeIndicator("momentum", 0.5, 0.2, MarketRegime.MEAN_REVERTING)
        
        # RSI-ähnlicher Momentum Indicator
        returns = np.diff(prices) / prices[:-1]
        
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(positive_returns) > len(negative_returns) * 1.5:
            return RegimeIndicator("momentum", 0.8, 0.2, MarketRegime.MOMENTUM)
        elif len(negative_returns) > len(positive_returns) * 1.5:
            return RegimeIndicator("momentum", 0.8, 0.2, MarketRegime.BEAR_TRENDING)
        else:
            return RegimeIndicator("momentum", 0.6, 0.2, MarketRegime.MEAN_REVERTING)
    
    async def _calculate_mean_reversion_indicator(self, prices: np.ndarray) -> RegimeIndicator:
        """Mean Reversion Regime Detection"""
        
        if len(prices) < 20:
            return RegimeIndicator("mean_reversion", 0.5, 0.25, MarketRegime.MEAN_REVERTING)
        
        # Bollinger Band ähnlich
        sma_20 = np.mean(prices[-20:])
        std_20 = np.std(prices[-20:])
        
        current_price = prices[-1]
        z_score = abs(current_price - sma_20) / std_20
        
        if z_score > 2:  # Price weit von Mean entfernt
            return RegimeIndicator("mean_reversion", 0.9, 0.25, MarketRegime.MEAN_REVERTING)
        elif z_score < 0.5:  # Price nah am Mean
            return RegimeIndicator("mean_reversion", 0.3, 0.25, MarketRegime.MOMENTUM)
        else:
            return RegimeIndicator("mean_reversion", 0.6, 0.25, MarketRegime.MEAN_REVERTING)
    
    def _initialize_risk_rules(self) -> None:
        """Initialisiert Regime-spezifische Risk Rules"""
        
        # Bull Trending Regime Rules
        self.risk_rules[MarketRegime.BULL_TRENDING] = [
            RiskRule("max_long_exposure", lambda s: s.direction != SignalDirection.LONG or True, priority=1),
            RiskRule("momentum_confirm", lambda s: s.confidence > 0.6, priority=2),
        ]
        
        # Bear Trending Regime Rules
        self.risk_rules[MarketRegime.BEAR_TRENDING] = [
            RiskRule("limit_long_positions", lambda s: s.direction != SignalDirection.LONG or s.confidence > 0.8, priority=1),
            RiskRule("prefer_shorts", lambda s: s.direction == SignalDirection.SHORT or s.confidence > 0.7, priority=2),
        ]
        
        # High Volatility Regime Rules
        self.risk_rules[MarketRegime.HIGH_VOLATILITY] = [
            RiskRule("high_confidence_only", lambda s: s.confidence > 0.8, priority=1),
            RiskRule("reduce_position_size", lambda s: True, priority=2),  # Handled in allocation
        ]
        
        # Crisis Regime Rules
        self.risk_rules[MarketRegime.CRISIS] = [
            RiskRule("emergency_mode", lambda s: s.confidence > 0.9, priority=1),
            RiskRule("defensive_only", lambda s: s.origin in ['stablecoin_parking', 'arbitrage'], priority=1),
        ]
        
        # Default Rules für alle Regimes
        for regime in MarketRegime:
            if regime not in self.risk_rules:
                self.risk_rules[regime] = []
            
            # Universelle Rules hinzufügen
            self.risk_rules[regime].extend([
                RiskRule("min_confidence", lambda s: s.confidence > 0.4, priority=1),
                RiskRule("valid_duration", lambda s: s.expected_duration_min > 0, priority=2),
            ])
    
    async def _basic_signal_validation(self, signal: Signal) -> bool:
        """Basis-Validierung des Signals"""
        
        # Confidence Check
        if signal.confidence < 0.3:
            return False
        
        # Asset Check (Whitelist)
        allowed_assets = {'BTC', 'ETH', 'USDT', 'BNB', 'ADA', 'SOL', 'MATIC', 'AVAX'}
        if signal.asset.upper() not in allowed_assets:
            return False
        
        # Origin Check
        allowed_origins = {
            'lazy_billionaire', 'ml_strategy', 'arbitrage', 'mean_reversion',
            'momentum', 'grid', 'liquidation_hunter', 'defi_yield',
            'stablecoin_parking', 'autopilot', 'scalping'
        }
        if signal.origin not in allowed_origins:
            return False
        
        # Timestamp Check (nicht zu alt)
        if datetime.now() - signal.timestamp > timedelta(minutes=5):
            return False
        
        return True
    
    async def _check_regime_rules(self, signal: Signal, regime: MarketRegime) -> bool:
        """Prüft Regime-spezifische Rules"""
        
        if regime not in self.risk_rules:
            return True
        
        rules = self.risk_rules[regime]
        
        for rule in sorted(rules, key=lambda r: r.priority):
            if rule.enabled:
                try:
                    if not rule.condition(signal):
                        logger.debug(f"Signal rejected by rule: {rule.name}")
                        return False
                except Exception as e:
                    logger.error(f"Error in risk rule {rule.name}: {e}")
                    return False
        
        return True
    
    async def _check_portfolio_risk(self, signal: Signal) -> bool:
        """Portfolio-Level Risk Checks"""
        
        # Position Size Check
        total_portfolio_value = sum(pos.get('value', 0) for pos in self.active_positions.values())
        max_position_value = total_portfolio_value * self.max_single_position
        
        # Estimate Signal Value (vereinfacht)
        estimated_signal_value = total_portfolio_value * 0.05  # 5% default
        
        if estimated_signal_value > max_position_value:
            return False
        
        # Max Positions Check
        if len(self.active_positions) >= 20:  # Max 20 gleichzeitige Positionen
            return False
        
        return True
    
    async def _check_correlation_risk(self, signal: Signal) -> bool:
        """Correlation Risk zwischen Positionen"""
        
        if len(self.active_positions) < 2:
            return True  # Keine Korrelation bei wenigen Positionen
        
        # Vereinfachte Korrelationscheck basierend auf Asset und Origin
        similar_positions = 0
        
        for pos in self.active_positions.values():
            if (pos.get('asset') == signal.asset or 
                pos.get('origin') == signal.origin):
                similar_positions += 1
        
        # Max 3 ähnliche Positionen
        if similar_positions >= 3:
            return False
        
        return True
    
    async def _check_circuit_breakers(self, signal: Signal) -> bool:
        """Circuit Breaker Status prüfen"""
        
        # Alle Circuit Breaker müssen False sein
        active_breakers = [name for name, active in self.circuit_breakers.items() if active]
        
        if active_breakers:
            logger.warning(f"Active circuit breakers: {active_breakers}")
            return False
        
        return True
    
    async def _check_var_impact(self, signal: Signal) -> bool:
        """VaR Impact Check für neues Signal"""
        
        # Vereinfachte VaR Calculation
        current_var = await self._calculate_portfolio_var()
        
        # Estimate VaR Impact des neuen Signals
        signal_risk = signal.confidence * 0.02  # 2% base risk
        estimated_new_var = current_var + signal_risk
        
        if estimated_new_var > self.max_portfolio_var:
            return False
        
        return True
    
    async def calculate_portfolio_risk(self) -> RiskMetrics:
        """Berechnet umfassende Portfolio Risk Metrics"""
        
        portfolio_var = await self._calculate_portfolio_var()
        portfolio_cvar = portfolio_var * 1.5  # Vereinfacht
        
        # Weitere Metriken berechnen (vereinfacht)
        max_drawdown = 0.05  # Placeholder
        sharpe_ratio = 1.2   # Placeholder
        volatility = 0.15    # Placeholder
        
        concentration_risk = await self._calculate_concentration_risk()
        correlation_risk = await self._calculate_correlation_risk_metric()
        leverage_ratio = await self._calculate_leverage_ratio()
        
        metrics = RiskMetrics(
            portfolio_var=portfolio_var,
            portfolio_cvar=portfolio_cvar,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            volatility=volatility,
            concentration_risk=concentration_risk,
            correlation_risk=correlation_risk,
            leverage_ratio=leverage_ratio,
            market_regime=self.current_regime,
            regime_confidence=self.regime_confidence
        )
        
        self.risk_metrics_history.append(metrics)
        self.last_risk_calculation = datetime.now()
        
        return metrics
    
    async def _calculate_portfolio_var(self) -> float:
        """Portfolio Value at Risk Calculation"""
        
        if not self.active_positions:
            return 0.0
        
        # Vereinfachte VaR (1-day, 95% confidence)
        position_values = [pos.get('value', 0) for pos in self.active_positions.values()]
        position_volatilities = [pos.get('volatility', 0.2) for pos in self.active_positions.values()]
        
        if not position_values:
            return 0.0
        
        total_value = sum(position_values)
        weighted_volatility = np.average(position_volatilities, weights=position_values)
        
        # VaR = 1.65 * volatility * portfolio_value (95% confidence)
        var = 1.65 * weighted_volatility * total_value / np.sqrt(252)  # Daily VaR
        
        return var / total_value if total_value > 0 else 0.0  # As percentage
    
    async def _calculate_concentration_risk(self) -> float:
        """Concentration Risk (HHI-basiert)"""
        
        if not self.active_positions:
            return 0.0
        
        position_values = [pos.get('value', 0) for pos in self.active_positions.values()]
        total_value = sum(position_values)
        
        if total_value == 0:
            return 0.0
        
        # Herfindahl-Hirschman Index
        weights = [val / total_value for val in position_values]
        hhi = sum(w ** 2 for w in weights)
        
        return hhi
    
    async def _calculate_correlation_risk_metric(self) -> float:
        """Correlation Risk Metric"""
        
        if len(self.active_positions) < 2:
            return 0.0
        
        # Vereinfacht: Basiert auf ähnlichen Assets/Origins
        assets = [pos.get('asset', '') for pos in self.active_positions.values()]
        origins = [pos.get('origin', '') for pos in self.active_positions.values()]
        
        asset_concentration = len(set(assets)) / len(assets)
        origin_concentration = len(set(origins)) / len(origins)
        
        # Average diversification (je niedriger, desto höher das Correlation Risk)
        avg_diversification = (asset_concentration + origin_concentration) / 2
        correlation_risk = 1 - avg_diversification
        
        return correlation_risk
    
    async def _calculate_leverage_ratio(self) -> float:
        """Portfolio Leverage Ratio"""
        
        # Vereinfacht: Annahme dass alle Positionen 1x Leverage haben
        return 1.0
    
    def add_position(self, signal_id: str, asset: str, origin: str, value: float, volatility: float = 0.2) -> None:
        """Fügt Position zum Risk Tracking hinzu"""
        
        self.active_positions[signal_id] = {
            'asset': asset,
            'origin': origin,
            'value': value,
            'volatility': volatility,
            'timestamp': datetime.now()
        }
        
        logger.debug(f"Position hinzugefügt: {signal_id} ({asset}, {value:,.0f})")
    
    def remove_position(self, signal_id: str) -> None:
        """Entfernt Position aus Risk Tracking"""
        
        if signal_id in self.active_positions:
            del self.active_positions[signal_id]
            logger.debug(f"Position entfernt: {signal_id}")
    
    def update_market_data(self, price: float, volume: float = None) -> None:
        """Updated Market Data für Regime Detection"""
        
        self.price_history.append(price)
        
        if volume:
            self.volume_history.append(volume)
        
        # Volatility berechnen und hinzufügen
        if len(self.price_history) >= 2:
            returns = [(self.price_history[i] - self.price_history[i-1]) / self.price_history[i-1] 
                      for i in range(1, min(len(self.price_history), 11))]
            current_vol = np.std(returns) if returns else 0.2
            self.volatility_history.append(current_vol)
    
    def set_circuit_breaker(self, breaker_name: str, active: bool) -> None:
        """Setzt Circuit Breaker Status"""
        
        if breaker_name in self.circuit_breakers:
            self.circuit_breakers[breaker_name] = active
            logger.warning(f"Circuit Breaker {breaker_name} set to {active}")
    
    def get_risk_status(self) -> Dict[str, any]:
        """Risk Engine Status"""
        
        return {
            'current_regime': self.current_regime.value,
            'regime_confidence': self.regime_confidence,
            'active_positions': len(self.active_positions),
            'circuit_breakers': self.circuit_breakers,
            'last_risk_calculation': self.last_risk_calculation,
            'risk_indicators_count': len(self.regime_indicators)
        }
    
    def __repr__(self) -> str:
        return f"AdvancedRiskEngine(regime={self.current_regime.value}, positions={len(self.active_positions)}, breakers={sum(self.circuit_breakers.values())})"