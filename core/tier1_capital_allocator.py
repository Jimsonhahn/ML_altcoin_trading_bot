"""
Tier-1 Capital Allocator
Elite institutioneller CapitalAllocator mit EWMA, Risk Parity und Kovarianzmatrix
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import logging
from collections import defaultdict, deque

from .tier1_models import (
    Signal, Allocation, StrategyMetrics, ICapitalAllocator,
    SystemConstants, MarketRegime
)

logger = logging.getLogger(__name__)


class AdvancedCapitalAllocator(ICapitalAllocator):
    """
    Elite Capital Allocator mit institutionellen Algorithmen:
    - EWMA (Exponentially Weighted Moving Average) für Metriken-Glättung
    - Risk Parity Allocation
    - Kovarianzmatrix für Strategien
    - Kelly Criterion mit Fraktional Sizing
    - Volatility Target Sizing
    """
    
    def __init__(self, 
                 total_capital: float = 1000000,
                 max_allocation_per_signal: float = 0.05,  # 5% max per signal
                 volatility_target: float = 0.15,  # 15% annual vol target
                 ewma_alpha: float = 0.05):  # EWMA decay factor
        
        self.total_capital = total_capital
        self.max_allocation_per_signal = max_allocation_per_signal
        self.volatility_target = volatility_target
        self.ewma_alpha = ewma_alpha
        
        # Strategy Performance Tracking mit EWMA
        self.strategy_metrics: Dict[str, StrategyMetrics] = {}
        self.strategy_returns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=252))
        
        # Kovarianzmatrix der Strategien
        self.covariance_matrix: Optional[np.ndarray] = None
        self.strategy_names: List[str] = []
        
        # Risk Parity Gewichte
        self.risk_parity_weights: Dict[str, float] = {}
        
        # Current Allocations Tracking
        self.current_allocations: Dict[str, float] = defaultdict(float)
        self.allocated_capital: float = 0.0
        
        # Performance tracking
        self.last_rebalance: datetime = datetime.now()
        self.rebalance_frequency = timedelta(hours=6)  # Rebalance alle 6h
        
        logger.info(f"AdvancedCapitalAllocator initialisiert mit {total_capital:,.0f} USD")
    
    async def allocate(self, signal: Signal) -> Optional[Allocation]:
        """
        Hauptmethode: Intelligente Kapitalallokation
        Kombiniert EWMA-geglättete Metriken, Risk Parity und Kelly Criterion
        """
        try:
            # 1. Verfügbares Kapital prüfen
            available_capital = self.total_capital - self.allocated_capital
            if available_capital <= 0:
                logger.warning(f"Kein verfügbares Kapital für Signal {signal.signal_id}")
                return None
            
            # 2. Strategie-spezifische Metriken abrufen oder initialisieren
            strategy_metric = await self._get_or_create_strategy_metrics(signal.origin)
            
            # 3. Base Allocation berechnen (ohne Adjustments)
            base_allocation = await self._calculate_base_allocation(signal, strategy_metric)
            
            if base_allocation <= 0:
                return None
            
            # 4. Risk Parity Adjustment
            risk_parity_weight = await self._get_risk_parity_weight(signal.origin)
            
            # 5. Volatility Targeting
            vol_adjusted_allocation = await self._apply_volatility_targeting(
                base_allocation, strategy_metric, signal
            )
            
            # 6. Kelly Criterion Sizing
            kelly_allocation = await self._calculate_kelly_sizing(
                vol_adjusted_allocation, signal, strategy_metric
            )
            
            # 7. Portfolio-Level Constraints
            final_allocation = await self._apply_portfolio_constraints(
                kelly_allocation, signal, available_capital
            )
            
            # 8. Position Sizing berechnen
            position_size = await self._calculate_position_size(final_allocation, signal)
            
            # 9. Allocation erstellen
            allocation = Allocation(
                signal_id=signal.signal_id,
                amount=final_allocation,
                position_size=position_size,
                leverage=1.0,  # Konservativ
                max_risk_per_trade=min(0.02, final_allocation / self.total_capital),
                kelly_fraction=kelly_allocation / base_allocation if base_allocation > 0 else 0,
                volatility_adjusted=True,
                strategy_weight=strategy_metric.risk_parity_weight,
                volatility_weight=1.0 / max(strategy_metric.ewma_volatility, 0.01),
                correlation_adjustment=risk_parity_weight
            )
            
            # 10. Allocation tracking aktualisieren
            self.current_allocations[signal.origin] += final_allocation
            self.allocated_capital += final_allocation
            
            logger.info(f"Allocation erstellt: {final_allocation:,.0f} USD für {signal.origin}")
            return allocation
            
        except Exception as e:
            logger.error(f"Fehler bei Kapitalallokation: {e}")
            return None
    
    async def _get_or_create_strategy_metrics(self, strategy_name: str) -> StrategyMetrics:
        """Holt oder erstellt Strategie-Metriken"""
        
        if strategy_name not in self.strategy_metrics:
            # Neue Strategie - initiale Metriken erstellen
            self.strategy_metrics[strategy_name] = StrategyMetrics(
                strategy_name=strategy_name,
                returns=[],
                sharpe_ratio=0.5,  # Konservative Annahme
                volatility=0.15,   # Mittlere Volatilität
                max_drawdown=0.05,
                win_rate=0.55,
                ewma_return=0.0,
                ewma_volatility=0.15,
                ewma_sharpe=0.5,
                inverse_volatility_weight=1.0 / 0.15,
                risk_parity_weight=1.0
            )
            
            # Zur Kovarianzmatrix hinzufügen
            if strategy_name not in self.strategy_names:
                self.strategy_names.append(strategy_name)
                await self._rebuild_covariance_matrix()
        
        return self.strategy_metrics[strategy_name]
    
    async def _calculate_base_allocation(self, signal: Signal, strategy_metric: StrategyMetrics) -> float:
        """Berechnet Base Allocation basierend auf Signal-Confidence und Strategie-Performance"""
        
        # Base auf Signal Confidence
        confidence_factor = signal.confidence ** 2  # Quadratisch für Konservativität
        
        # Strategie-Performance Factor (EWMA-geglättet)
        performance_factor = max(0.1, strategy_metric.ewma_sharpe / 2.0)  # Normalisiert
        
        # Expected Profit Factor
        profit_factor = min(2.0, max(0.5, signal.expected_profit_pts / 100))  # 1% = 1.0
        
        # Base Allocation
        base_allocation = (
            self.total_capital * 
            self.max_allocation_per_signal * 
            confidence_factor * 
            performance_factor * 
            profit_factor
        )
        
        logger.debug(f"Base allocation: {base_allocation:,.0f} (conf={confidence_factor:.2f}, perf={performance_factor:.2f})")
        return base_allocation
    
    async def _get_risk_parity_weight(self, strategy_name: str) -> float:
        """Berechnet Risk Parity Weight für Strategie"""
        
        if strategy_name not in self.risk_parity_weights:
            await self._calculate_risk_parity_weights()
        
        return self.risk_parity_weights.get(strategy_name, 1.0)
    
    async def _calculate_risk_parity_weights(self) -> None:
        """Berechnet Risk Parity Gewichte basierend auf inversen Volatilitäten"""
        
        if not self.strategy_metrics:
            return
        
        # Inverse Volatilitäten sammeln
        inverse_volatilities = {}
        for name, metrics in self.strategy_metrics.items():
            vol = max(metrics.ewma_volatility, 0.01)  # Minimum Volatilität
            inverse_volatilities[name] = 1.0 / vol
        
        # Normalisieren zu Gewichten
        total_inverse_vol = sum(inverse_volatilities.values())
        
        for name in inverse_volatilities:
            self.risk_parity_weights[name] = inverse_volatilities[name] / total_inverse_vol
            
            # Update strategy metrics
            if name in self.strategy_metrics:
                self.strategy_metrics[name].inverse_volatility_weight = inverse_volatilities[name]
                self.strategy_metrics[name].risk_parity_weight = self.risk_parity_weights[name]
        
        logger.debug(f"Risk Parity Gewichte aktualisiert: {self.risk_parity_weights}")
    
    async def _apply_volatility_targeting(self, allocation: float, strategy_metric: StrategyMetrics, signal: Signal) -> float:
        """Volatility Targeting: Skaliert Allocation basierend auf Volatilitätsziel"""
        
        strategy_vol = max(strategy_metric.ewma_volatility, 0.01)
        vol_scaling = self.volatility_target / strategy_vol
        
        # Begrenzen auf sinnvolle Bereiche
        vol_scaling = np.clip(vol_scaling, 0.5, 2.0)
        
        adjusted_allocation = allocation * vol_scaling
        
        logger.debug(f"Volatility adjustment: {vol_scaling:.2f}x (target={self.volatility_target:.1%}, strategy={strategy_vol:.1%})")
        return adjusted_allocation
    
    async def _calculate_kelly_sizing(self, allocation: float, signal: Signal, strategy_metric: StrategyMetrics) -> float:
        """Kelly Criterion für optimale Position Size"""
        
        # Kelly Formel: f = (bp - q) / b
        # b = odds received (profit/loss ratio)
        # p = probability of winning
        # q = probability of losing = 1-p
        
        win_prob = strategy_metric.win_rate
        loss_prob = 1 - win_prob
        
        # Durchschnittliches Gewinn/Verlust Verhältnis schätzen
        if strategy_metric.returns:
            winning_returns = [r for r in strategy_metric.returns if r > 0]
            losing_returns = [r for r in strategy_metric.returns if r < 0]
            
            if winning_returns and losing_returns:
                avg_win = np.mean(winning_returns)
                avg_loss = abs(np.mean(losing_returns))
                profit_loss_ratio = avg_win / avg_loss
            else:
                profit_loss_ratio = 1.5  # Konservative Annahme
        else:
            profit_loss_ratio = signal.expected_profit_pts / 50  # Annahme: 50bps average loss
        
        # Kelly Fraction berechnen
        kelly_fraction = (win_prob * profit_loss_ratio - loss_prob) / profit_loss_ratio
        
        # Konservativ: Nur Bruchteil des Kelly-Optimums verwenden
        fractional_kelly = max(0, min(0.25, kelly_fraction * 0.5))  # Max 25%, halbes Kelly
        
        kelly_allocation = allocation * (1 + fractional_kelly)
        
        logger.debug(f"Kelly sizing: {fractional_kelly:.2%} -> {kelly_allocation:,.0f}")
        return kelly_allocation
    
    async def _apply_portfolio_constraints(self, allocation: float, signal: Signal, available_capital: float) -> float:
        """Portfolio-Level Constraints anwenden"""
        
        # 1. Verfügbares Kapital Constraint
        allocation = min(allocation, available_capital)
        
        # 2. Max Allocation per Signal
        max_allowed = self.total_capital * self.max_allocation_per_signal
        allocation = min(allocation, max_allowed)
        
        # 3. Strategie-spezifische Konzentration
        current_strategy_allocation = self.current_allocations[signal.origin]
        max_strategy_allocation = self.total_capital * 0.15  # Max 15% per Strategie
        
        if current_strategy_allocation + allocation > max_strategy_allocation:
            allocation = max(0, max_strategy_allocation - current_strategy_allocation)
        
        # 4. Minimum Allocation
        min_allocation = self.total_capital * 0.001  # 0.1% minimum
        if allocation < min_allocation:
            allocation = 0
        
        return allocation
    
    async def _calculate_position_size(self, allocation: float, signal: Signal) -> float:
        """Berechnet Position Size basierend auf Allocation und Asset-Preis"""
        
        # Vereinfacht: Annahme dass allocation bereits USD ist
        # In echter Implementierung würde hier der aktuelle Asset-Preis abgefragt
        estimated_price = 50000  # Beispiel: BTC Preis
        
        if signal.asset.upper() in ['BTC', 'BITCOIN']:
            estimated_price = 50000
        elif signal.asset.upper() in ['ETH', 'ETHEREUM']:
            estimated_price = 3000
        else:
            estimated_price = 100  # Default für Altcoins
        
        position_size = allocation / estimated_price
        return position_size
    
    async def update_strategy_metrics(self, metrics: StrategyMetrics) -> None:
        """Update Strategy Metrics mit EWMA Glättung"""
        
        strategy_name = metrics.strategy_name
        
        if strategy_name in self.strategy_metrics:
            existing = self.strategy_metrics[strategy_name]
            
            # EWMA Update für alle Metriken
            alpha = self.ewma_alpha
            
            existing.ewma_return = (1 - alpha) * existing.ewma_return + alpha * np.mean(metrics.returns[-10:]) if metrics.returns else existing.ewma_return
            existing.ewma_volatility = (1 - alpha) * existing.ewma_volatility + alpha * metrics.volatility
            existing.ewma_sharpe = (1 - alpha) * existing.ewma_sharpe + alpha * metrics.sharpe_ratio
            
            # Returns Historie aktualisieren
            self.strategy_returns[strategy_name].extend(metrics.returns)
            existing.returns = list(self.strategy_returns[strategy_name])
            
            # Andere Metriken aktualisieren
            existing.sharpe_ratio = metrics.sharpe_ratio
            existing.volatility = metrics.volatility
            existing.max_drawdown = metrics.max_drawdown
            existing.win_rate = metrics.win_rate
            existing.last_updated = datetime.now()
            
            logger.debug(f"Strategy metrics aktualisiert für {strategy_name}: EWMA Sharpe={existing.ewma_sharpe:.2f}")
        else:
            # Neue Strategie
            self.strategy_metrics[strategy_name] = metrics
        
        # Risk Parity Gewichte neu berechnen
        await self._calculate_risk_parity_weights()
    
    async def rebalance_portfolio(self) -> Dict[str, float]:
        """Risk Parity Rebalancing des Portfolios"""
        
        if datetime.now() - self.last_rebalance < self.rebalance_frequency:
            return self.risk_parity_weights
        
        logger.info("Starte Portfolio Rebalancing...")
        
        # 1. Aktuelle Kovarianzmatrix neu berechnen
        await self._rebuild_covariance_matrix()
        
        # 2. Risk Parity Gewichte neu berechnen
        await self._calculate_risk_parity_weights()
        
        # 3. Target Allocations berechnen
        target_allocations = {}
        for strategy_name, weight in self.risk_parity_weights.items():
            target_allocation = self.total_capital * weight * 0.8  # 80% invested
            target_allocations[strategy_name] = target_allocation
        
        self.last_rebalance = datetime.now()
        
        logger.info(f"Portfolio rebalanced. Target allocations: {target_allocations}")
        return target_allocations
    
    async def _rebuild_covariance_matrix(self) -> None:
        """Rebuilds Kovarianzmatrix für alle Strategien"""
        
        if len(self.strategy_names) < 2:
            return
        
        # Returns Matrix erstellen
        returns_matrix = []
        min_length = float('inf')
        
        # Minimum Länge finden
        for name in self.strategy_names:
            if name in self.strategy_returns and self.strategy_returns[name]:
                min_length = min(min_length, len(self.strategy_returns[name]))
        
        if min_length < 10:  # Zu wenig Daten
            return
        
        # Returns Matrix aufbauen
        for name in self.strategy_names:
            if name in self.strategy_returns and self.strategy_returns[name]:
                strategy_returns = list(self.strategy_returns[name])[-min_length:]
                returns_matrix.append(strategy_returns)
            else:
                # Dummy returns für neue Strategien
                returns_matrix.append([0.0] * min_length)
        
        # Kovarianzmatrix berechnen
        if returns_matrix and len(returns_matrix) >= 2:
            self.covariance_matrix = np.cov(returns_matrix)
            
            logger.debug(f"Kovarianzmatrix aktualisiert: {self.covariance_matrix.shape}")
    
    def get_allocation_status(self) -> Dict[str, any]:
        """Status der aktuellen Allocations"""
        
        utilization = self.allocated_capital / self.total_capital
        
        return {
            'total_capital': self.total_capital,
            'allocated_capital': self.allocated_capital,
            'available_capital': self.total_capital - self.allocated_capital,
            'utilization_rate': utilization,
            'current_allocations': dict(self.current_allocations),
            'risk_parity_weights': self.risk_parity_weights,
            'strategy_count': len(self.strategy_metrics),
            'last_rebalance': self.last_rebalance
        }
    
    async def release_allocation(self, signal_id: str, amount: float, strategy_name: str) -> None:
        """Gibt Kapital nach Trade-Abschluss frei"""
        
        self.allocated_capital = max(0, self.allocated_capital - amount)
        self.current_allocations[strategy_name] = max(0, self.current_allocations[strategy_name] - amount)
        
        logger.debug(f"Allocation freigegeben: {amount:,.0f} USD für {strategy_name}")
    
    def __repr__(self) -> str:
        return f"AdvancedCapitalAllocator(capital={self.total_capital:,.0f}, allocated={self.allocated_capital:,.0f}, strategies={len(self.strategy_metrics)})"