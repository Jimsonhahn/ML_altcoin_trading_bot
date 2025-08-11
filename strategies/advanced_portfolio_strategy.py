"""
Advanced Portfolio Strategy - Defensive Multi-Asset Optimization
Fokussiert auf Risiko-Management und legitime Alpha-Generierung
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


class MarketRegime(Enum):
    """Marktregime für Portfolio-Allokation"""
    BULL = "bull"
    BEAR = "bear" 
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    CRISIS = "crisis"


@dataclass
class AssetAllocation:
    """Asset-Allokation mit Risiko-Metriken"""
    symbol: str
    target_weight: float
    current_weight: float
    risk_contribution: float
    expected_return: float
    confidence: float


class AdvancedPortfolioStrategy(Strategy):
    """
    Erweiterte Portfolio-Strategie mit:
    - Dynamische Asset-Allokation
    - Risiko-Parität
    - Regime-bewusste Anpassung
    - Defensive Positionierung
    """
    
    def __init__(self, config: Dict[str, Any], ml_components: Optional[Any] = None):
        super().__init__(config, ml_components)
        self.name = "advanced_portfolio"
        
        # Konfiguration
        self.rebalance_threshold = config.get('rebalance_threshold', 0.05)
        self.max_position_size = config.get('max_position_size', 0.25)
        self.lookback_period = config.get('lookback_period', 60)
        self.risk_target = config.get('risk_target', 0.15)  # 15% Ziel-Volatilität
        
        # Asset Universe
        self.asset_universe = config.get('assets', [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT'
        ])
        
        # Regime-spezifische Allokationen
        self.regime_allocations = {
            MarketRegime.BULL: {'crypto_weight': 0.8, 'stable_weight': 0.2},
            MarketRegime.BEAR: {'crypto_weight': 0.3, 'stable_weight': 0.7},
            MarketRegime.SIDEWAYS: {'crypto_weight': 0.6, 'stable_weight': 0.4},
            MarketRegime.HIGH_VOLATILITY: {'crypto_weight': 0.4, 'stable_weight': 0.6},
            MarketRegime.CRISIS: {'crypto_weight': 0.1, 'stable_weight': 0.9}
        }
        
        # State
        self.current_allocations: Dict[str, AssetAllocation] = {}
        self.last_rebalance = datetime.min
        self.current_regime = MarketRegime.SIDEWAYS
        
        logger.info(f"Advanced Portfolio Strategy initialisiert mit {len(self.asset_universe)} Assets")
    
    async def calculate_signal(self, symbol: str, data: pd.DataFrame, current_price: float) -> Tuple[str, Dict[str, Any]]:
        """
        Berechnet Portfolio-Signal für Asset
        """
        try:
            # Marktregime erkennen
            regime = await self._detect_market_regime(data)
            
            # Portfolio-Analyse durchführen
            allocation_analysis = await self._analyze_portfolio_allocation(symbol, data)
            
            # Rebalancing-Bedarf prüfen
            rebalance_needed = self._check_rebalance_needed(allocation_analysis)
            
            if not rebalance_needed:
                return 'HOLD', {
                    'reason': 'portfolio_balanced',
                    'confidence': 0.5,
                    'regime': regime.value,
                    'allocation': allocation_analysis
                }
            
            # Signal für Asset generieren
            signal_type, signal_strength = self._generate_portfolio_signal(
                symbol, allocation_analysis, regime
            )
            
            return signal_type, {
                'reason': 'portfolio_rebalance',
                'confidence': signal_strength,
                'regime': regime.value,
                'allocation': allocation_analysis,
                'target_weight': allocation_analysis.get('target_weight', 0.0),
                'current_weight': allocation_analysis.get('current_weight', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Fehler bei Portfolio-Signal für {symbol}: {e}")
            return 'HOLD', {'error': str(e), 'confidence': 0.0}
    
    async def _detect_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """
        Erkennt aktuelles Marktregime basierend auf Volatilität und Trend
        """
        try:
            # Volatilität der letzten 30 Tage
            returns = data['close'].pct_change().dropna()
            volatility = returns.rolling(30).std().iloc[-1] * np.sqrt(365)
            
            # Trend der letzten 60 Tage
            sma_short = data['close'].rolling(20).mean().iloc[-1]
            sma_long = data['close'].rolling(60).mean().iloc[-1]
            trend = (sma_short - sma_long) / sma_long
            
            # VIX-ähnlicher Fear Index
            rolling_max = data['high'].rolling(30).max().iloc[-1]
            rolling_min = data['low'].rolling(30).min().iloc[-1]
            current_price = data['close'].iloc[-1]
            fear_index = (rolling_max - current_price) / (rolling_max - rolling_min)
            
            # Regime-Klassifikation
            if volatility > 0.6:  # Hohe Volatilität
                if fear_index > 0.8:
                    return MarketRegime.CRISIS
                else:
                    return MarketRegime.HIGH_VOLATILITY
            elif trend > 0.1:  # Starker Aufwärtstrend
                return MarketRegime.BULL
            elif trend < -0.1:  # Starker Abwärtstrend
                return MarketRegime.BEAR
            else:
                return MarketRegime.SIDEWAYS
                
        except Exception as e:
            logger.warning(f"Fehler bei Regime-Erkennung: {e}")
            return MarketRegime.SIDEWAYS
    
    async def _analyze_portfolio_allocation(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analysiert aktuelle Portfolio-Allokation und berechnet Ziel-Gewichte
        """
        try:
            # Risiko-Metriken berechnen
            returns = data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(365)
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(365) if returns.std() > 0 else 0
            
            # Korrelation zu anderen Assets (vereinfacht)
            correlation_penalty = 0.0  # Wird in Realität mit anderen Assets berechnet
            
            # Ziel-Gewicht basierend auf Risiko-Parität
            base_weight = 1.0 / len(self.asset_universe)
            risk_adjusted_weight = base_weight * (self.risk_target / max(volatility, 0.01))
            
            # Regime-Anpassung
            regime_factor = self._get_regime_factor(symbol)
            target_weight = min(risk_adjusted_weight * regime_factor, self.max_position_size)
            
            # Aktuelle Gewichtung (vereinfacht - in Realität aus Portfolio-Manager)
            current_weight = 1.0 / len(self.asset_universe)  # Gleichgewichtet als Startwert
            
            return {
                'symbol': symbol,
                'target_weight': target_weight,
                'current_weight': current_weight,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'risk_contribution': volatility * current_weight,
                'expected_return': returns.mean() * 365,
                'regime_factor': regime_factor
            }
            
        except Exception as e:
            logger.error(f"Fehler bei Portfolio-Analyse für {symbol}: {e}")
            return {'error': str(e)}
    
    def _get_regime_factor(self, symbol: str) -> float:
        """
        Holt Regime-spezifischen Faktor für Asset
        """
        regime_config = self.regime_allocations.get(self.current_regime, {})
        
        # Vereinfachte Klassifikation: BTC/ETH als Crypto, USDT als Stable
        if 'USDT' in symbol and symbol != 'BTC/USDT' and symbol != 'ETH/USDT':
            return regime_config.get('stable_weight', 0.5)
        else:
            return regime_config.get('crypto_weight', 0.5)
    
    def _check_rebalance_needed(self, allocation: Dict[str, Any]) -> bool:
        """
        Prüft ob Rebalancing erforderlich ist
        """
        if 'error' in allocation:
            return False
        
        target_weight = allocation.get('target_weight', 0.0)
        current_weight = allocation.get('current_weight', 0.0)
        
        weight_deviation = abs(target_weight - current_weight)
        
        # Zeitbasiertes Rebalancing (mindestens alle 24h)
        time_since_rebalance = datetime.now() - self.last_rebalance
        time_rebalance_needed = time_since_rebalance > timedelta(hours=24)
        
        # Schwellwert-basiertes Rebalancing
        threshold_rebalance_needed = weight_deviation > self.rebalance_threshold
        
        return time_rebalance_needed or threshold_rebalance_needed
    
    def _generate_portfolio_signal(self, symbol: str, allocation: Dict[str, Any], 
                                 regime: MarketRegime) -> Tuple[str, float]:
        """
        Generiert Handelssignal für Portfolio-Rebalancing
        """
        if 'error' in allocation:
            return 'HOLD', 0.0
        
        target_weight = allocation.get('target_weight', 0.0)
        current_weight = allocation.get('current_weight', 0.0)
        
        weight_diff = target_weight - current_weight
        
        # Signal-Stärke basierend auf Gewichtungsabweichung
        signal_strength = min(abs(weight_diff) / self.rebalance_threshold, 1.0)
        
        # Risiko-Anpassung in Krisenzeiten
        if regime == MarketRegime.CRISIS:
            signal_strength *= 0.5  # Vorsichtigere Positionierung
        
        if weight_diff > self.rebalance_threshold:
            return 'BUY', signal_strength
        elif weight_diff < -self.rebalance_threshold:
            return 'SELL', signal_strength
        else:
            return 'HOLD', signal_strength
    
    def get_risk_metrics(self) -> Dict[str, float]:
        """
        Gibt aktuelle Risiko-Metriken zurück
        """
        total_risk = sum(
            alloc.risk_contribution for alloc in self.current_allocations.values()
        )
        
        concentration_risk = max(
            alloc.current_weight for alloc in self.current_allocations.values()
        ) if self.current_allocations else 0.0
        
        return {
            'total_portfolio_risk': total_risk,
            'concentration_risk': concentration_risk,
            'number_of_positions': len(self.current_allocations),
            'risk_target': self.risk_target,
            'current_regime': self.current_regime.value
        }
    
    def update_allocations(self, allocations: Dict[str, AssetAllocation]):
        """
        Aktualisiert aktuelle Allokationen
        """
        self.current_allocations = allocations
        self.last_rebalance = datetime.now()
        
        logger.info(f"Portfolio-Allokationen aktualisiert: {len(allocations)} Positionen")