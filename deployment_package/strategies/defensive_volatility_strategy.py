"""
Defensive Volatility Strategy - Volatilitäts-basierte Defensive Trading
Fokussiert auf Risiko-Management und Kapitalschutz
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from scipy import stats

from .strategy_base import Strategy
from core.interfaces import IStrategy
from utils.exceptions import StrategyError

logger = logging.getLogger(__name__)


@dataclass
class VolatilityMetrics:
    """Volatilitäts-Metriken für Risikobewertung"""
    realized_vol: float
    implied_vol: float  # Vereinfacht
    vol_percentile: float
    vol_trend: str  # 'increasing', 'decreasing', 'stable'
    regime: str  # 'low', 'normal', 'high', 'extreme'


class DefensiveVolatilityStrategy(Strategy):
    """
    Defensive Volatilitäts-Strategie mit Fokus auf:
    - Volatility-Regime-Detection
    - Position Sizing basierend auf Volatilität
    - Defensive Positionierung in High-Vol Perioden
    - Mean Reversion bei extremer Volatilität
    """
    
    def __init__(self, config: Dict[str, Any], ml_components: Optional[Any] = None):
        super().__init__(config, ml_components)
        self.name = "defensive_volatility"
        
        # Volatilitäts-Parameter
        self.vol_lookback = config.get('vol_lookback', 30)
        self.vol_percentile_period = config.get('vol_percentile_period', 252)  # 1 Jahr
        self.extreme_vol_threshold = config.get('extreme_vol_threshold', 0.95)  # 95. Perzentil
        self.high_vol_threshold = config.get('high_vol_threshold', 0.80)  # 80. Perzentil
        
        # Position Sizing Parameter
        self.base_position_size = config.get('base_position_size', 0.05)  # 5% Base
        self.max_position_size = config.get('max_position_size', 0.15)   # 15% Max
        self.min_position_size = config.get('min_position_size', 0.01)   # 1% Min
        
        # Risk Management
        self.vol_target = config.get('vol_target', 0.20)  # 20% Zielvolatilität
        self.max_drawdown_threshold = config.get('max_drawdown_threshold', 0.10)
        self.stop_loss_multiplier = config.get('stop_loss_multiplier', 2.0)
        
        # Mean Reversion Parameter
        self.reversion_period = config.get('reversion_period', 14)
        self.reversion_threshold = config.get('reversion_threshold', 2.0)  # 2 Std Dev
        
        # State
        self.vol_history: List[float] = []
        self.position_history: List[Dict] = []
        
        logger.info("Defensive Volatility Strategy initialisiert")
    
    async def calculate_signal(self, symbol: str, data: pd.DataFrame, current_price: float) -> Tuple[str, Dict[str, Any]]:
        """
        Berechnet defensives Volatilitäts-Signal
        """
        try:
            # Volatilitäts-Metriken berechnen
            vol_metrics = self._calculate_volatility_metrics(data)
            
            # Position Sizing basierend auf Volatilität
            position_size = self._calculate_vol_adjusted_position_size(vol_metrics)
            
            # Marktrichtung und Timing
            market_signal = self._get_market_direction_signal(data, vol_metrics)
            
            # Defensive Überprüfungen
            defensive_checks = self._perform_defensive_checks(data, vol_metrics, current_price)
            
            # Finales Signal basierend auf allen Faktoren
            final_signal, confidence = self._synthesize_signal(
                market_signal, vol_metrics, defensive_checks, position_size
            )
            
            return final_signal, {
                'strategy': self.name,
                'confidence': confidence,
                'position_size': position_size,
                'vol_regime': vol_metrics.regime,
                'vol_percentile': vol_metrics.vol_percentile,
                'realized_vol': vol_metrics.realized_vol,
                'market_signal': market_signal,
                'defensive_checks': defensive_checks,
                'stop_loss': self._calculate_stop_loss(current_price, vol_metrics)
            }
            
        except Exception as e:
            logger.error(f"Fehler bei Volatilitäts-Signal für {symbol}: {e}")
            return 'HOLD', {'error': str(e), 'confidence': 0.0}
    
    def _calculate_volatility_metrics(self, data: pd.DataFrame) -> VolatilityMetrics:
        """
        Berechnet umfassende Volatilitäts-Metriken
        """
        try:
            # Returns berechnen
            returns = data['close'].pct_change().dropna()
            
            # Realized Volatility (annualisiert)
            realized_vol = returns.rolling(self.vol_lookback).std().iloc[-1] * np.sqrt(365)
            
            # Volatilitäts-Historie für Perzentile
            vol_history = returns.rolling(self.vol_lookback).std() * np.sqrt(365)
            vol_history = vol_history.dropna()
            
            # Volatilitäts-Perzentil berechnen
            if len(vol_history) >= self.vol_percentile_period:
                recent_vols = vol_history.tail(self.vol_percentile_period)
                vol_percentile = stats.percentileofscore(recent_vols, realized_vol) / 100
            else:
                vol_percentile = 0.5  # Neutral wenn nicht genug Daten
            
            # Volatilitäts-Trend
            if len(vol_history) >= 10:
                recent_trend = vol_history.tail(10).mean()
                previous_trend = vol_history.tail(20).head(10).mean()
                
                if recent_trend > previous_trend * 1.1:
                    vol_trend = 'increasing'
                elif recent_trend < previous_trend * 0.9:
                    vol_trend = 'decreasing'
                else:
                    vol_trend = 'stable'
            else:
                vol_trend = 'stable'
            
            # Volatilitäts-Regime klassifizieren
            if vol_percentile >= self.extreme_vol_threshold:
                regime = 'extreme'
            elif vol_percentile >= self.high_vol_threshold:
                regime = 'high'
            elif vol_percentile <= 0.2:
                regime = 'low'
            else:
                regime = 'normal'
            
            # Implied Vol vereinfacht (in Realität von Options-Märkten)
            implied_vol = realized_vol * 1.2  # Vereinfachung
            
            return VolatilityMetrics(
                realized_vol=realized_vol,
                implied_vol=implied_vol,
                vol_percentile=vol_percentile,
                vol_trend=vol_trend,
                regime=regime
            )
            
        except Exception as e:
            logger.error(f"Fehler bei Volatilitäts-Berechnung: {e}")
            # Fallback-Werte
            return VolatilityMetrics(
                realized_vol=0.3,
                implied_vol=0.36,
                vol_percentile=0.5,
                vol_trend='stable',
                regime='normal'
            )
    
    def _calculate_vol_adjusted_position_size(self, vol_metrics: VolatilityMetrics) -> float:
        """
        Berechnet Positionsgröße basierend auf Volatilität
        """
        try:
            # Base Position Size anpassen
            vol_adjustment = self.vol_target / max(vol_metrics.realized_vol, 0.05)
            adjusted_size = self.base_position_size * vol_adjustment
            
            # Regime-basierte Anpassungen
            regime_multipliers = {
                'low': 1.5,      # Größere Positionen bei niedriger Vol
                'normal': 1.0,   # Standard
                'high': 0.6,     # Kleinere Positionen bei hoher Vol
                'extreme': 0.3   # Sehr kleine Positionen bei extremer Vol
            }
            
            regime_multiplier = regime_multipliers.get(vol_metrics.regime, 1.0)
            final_size = adjusted_size * regime_multiplier
            
            # Grenzen einhalten
            return max(self.min_position_size, min(final_size, self.max_position_size))
            
        except Exception as e:
            logger.error(f"Fehler bei Position Size Berechnung: {e}")
            return self.base_position_size
    
    def _get_market_direction_signal(self, data: pd.DataFrame, vol_metrics: VolatilityMetrics) -> str:
        """
        Bestimmt Marktrichtung unter Berücksichtigung der Volatilität
        """
        try:
            # Mean Reversion bei extremer Volatilität
            if vol_metrics.regime == 'extreme':
                return self._mean_reversion_signal(data)
            
            # Trend Following bei normaler/niedriger Volatilität
            elif vol_metrics.regime in ['normal', 'low']:
                return self._trend_following_signal(data)
            
            # Defensive Positionierung bei hoher Volatilität
            else:
                return self._defensive_signal(data, vol_metrics)
                
        except Exception as e:
            logger.error(f"Fehler bei Marktrichtung-Signal: {e}")
            return 'HOLD'
    
    def _mean_reversion_signal(self, data: pd.DataFrame) -> str:
        """
        Mean Reversion Signal bei extremer Volatilität
        """
        try:
            # Bollinger Bands für Mean Reversion
            close = data['close']
            sma = close.rolling(self.reversion_period).mean()
            std = close.rolling(self.reversion_period).std()
            
            current_price = close.iloc[-1]
            upper_band = sma.iloc[-1] + (self.reversion_threshold * std.iloc[-1])
            lower_band = sma.iloc[-1] - (self.reversion_threshold * std.iloc[-1])
            
            if current_price > upper_band:
                return 'SELL'  # Überkauft, erwarte Reversion
            elif current_price < lower_band:
                return 'BUY'   # Überverkauft, erwarte Reversion
            else:
                return 'HOLD'
                
        except Exception as e:
            logger.error(f"Fehler bei Mean Reversion Signal: {e}")
            return 'HOLD'
    
    def _trend_following_signal(self, data: pd.DataFrame) -> str:
        """
        Trend Following Signal bei normaler Volatilität
        """
        try:
            # Einfacher Momentum-Indikator
            close = data['close']
            
            # Verschiedene Zeitrahmen
            short_ma = close.rolling(10).mean().iloc[-1]
            medium_ma = close.rolling(20).mean().iloc[-1]
            long_ma = close.rolling(50).mean().iloc[-1]
            
            current_price = close.iloc[-1]
            
            # Trend-Bestimmung
            uptrend = (short_ma > medium_ma > long_ma) and (current_price > short_ma)
            downtrend = (short_ma < medium_ma < long_ma) and (current_price < short_ma)
            
            if uptrend:
                return 'BUY'
            elif downtrend:
                return 'SELL'
            else:
                return 'HOLD'
                
        except Exception as e:
            logger.error(f"Fehler bei Trend Following Signal: {e}")
            return 'HOLD'
    
    def _defensive_signal(self, data: pd.DataFrame, vol_metrics: VolatilityMetrics) -> str:
        """
        Defensive Signale bei hoher Volatilität
        """
        try:
            # Bei hoher Volatilität: Vorsichtige Positionierung
            # Fokus auf starke Signale und schnelle Exits
            
            close = data['close']
            volume = data.get('volume', pd.Series([0] * len(data)))
            
            # RSI für Überkauft/Überverkauft
            rsi = self._calculate_rsi(close, 14)
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50
            
            # Volume-bestätigte Signale
            volume_ma = volume.rolling(20).mean()
            current_volume = volume.iloc[-1]
            volume_confirmation = current_volume > volume_ma.iloc[-1] * 1.2
            
            # Nur starke Signale bei hoher Volatilität
            if current_rsi < 30 and volume_confirmation:
                return 'BUY'   # Stark überverkauft mit Volume
            elif current_rsi > 70 and volume_confirmation:
                return 'SELL'  # Stark überkauft mit Volume
            else:
                return 'HOLD'  # Abwarten bei unsicheren Signalen
                
        except Exception as e:
            logger.error(f"Fehler bei Defensive Signal: {e}")
            return 'HOLD'
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Berechnet Relative Strength Index
        """
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            logger.error(f"Fehler bei RSI Berechnung: {e}")
            return pd.Series([50] * len(prices))  # Neutral RSI als Fallback
    
    def _perform_defensive_checks(self, data: pd.DataFrame, vol_metrics: VolatilityMetrics, 
                                current_price: float) -> Dict[str, Any]:
        """
        Führt defensive Überprüfungen durch
        """
        checks = {
            'vol_regime_safe': vol_metrics.regime != 'extreme',
            'price_stability': True,
            'volume_normal': True,
            'drawdown_acceptable': True
        }
        
        try:
            # Preis-Stabilität prüfen
            recent_returns = data['close'].pct_change().tail(5)
            max_single_move = abs(recent_returns).max()
            checks['price_stability'] = max_single_move < 0.15  # Keine >15% Einzelbewegungen
            
            # Volume-Anomalien prüfen
            if 'volume' in data.columns:
                volume_ma = data['volume'].rolling(20).mean().iloc[-1]
                current_volume = data['volume'].iloc[-1]
                checks['volume_normal'] = current_volume < volume_ma * 3  # Kein 3x Volume-Spike
            
            # Drawdown prüfen (vereinfacht)
            recent_high = data['high'].rolling(20).max().iloc[-1]
            current_drawdown = (recent_high - current_price) / recent_high
            checks['drawdown_acceptable'] = current_drawdown < self.max_drawdown_threshold
            
        except Exception as e:
            logger.error(f"Fehler bei Defensive Checks: {e}")
        
        return checks
    
    def _synthesize_signal(self, market_signal: str, vol_metrics: VolatilityMetrics, 
                          defensive_checks: Dict[str, Any], position_size: float) -> Tuple[str, float]:
        """
        Synthetisiert finales Signal aus allen Komponenten
        """
        try:
            # Base Confidence aus Volatilitäts-Regime
            regime_confidence = {
                'low': 0.8,
                'normal': 0.7,
                'high': 0.5,
                'extreme': 0.3
            }
            
            base_confidence = regime_confidence.get(vol_metrics.regime, 0.5)
            
            # Defensive Checks reduzieren Confidence
            defensive_penalty = sum(1 for check in defensive_checks.values() if not check) * 0.2
            adjusted_confidence = max(0.1, base_confidence - defensive_penalty)
            
            # Signal-Filterung basierend auf Defensive Checks
            critical_checks_failed = sum(1 for check in defensive_checks.values() if not check) >= 2
            
            if critical_checks_failed:
                return 'HOLD', 0.1  # Sicherheitshalber halten
            
            # Volatilitäts-basierte Signal-Modifikation
            if vol_metrics.regime == 'extreme' and market_signal in ['BUY', 'SELL']:
                # Bei extremer Volatilität nur schwache Signale
                adjusted_confidence = min(adjusted_confidence, 0.4)
            
            return market_signal, adjusted_confidence
            
        except Exception as e:
            logger.error(f"Fehler bei Signal-Synthese: {e}")
            return 'HOLD', 0.1
    
    def _calculate_stop_loss(self, current_price: float, vol_metrics: VolatilityMetrics) -> float:
        """
        Berechnet dynamischen Stop Loss basierend auf Volatilität
        """
        try:
            # Stop Loss Distance basierend auf Volatilität
            vol_multiplier = max(1.0, vol_metrics.realized_vol / 0.3)  # Skaliert ab 30% Vol
            stop_distance = 0.05 * vol_multiplier * self.stop_loss_multiplier  # Base 5%
            
            # Regime-Anpassungen
            regime_adjustments = {
                'low': 0.8,      # Engere Stops bei niedriger Vol
                'normal': 1.0,   # Standard
                'high': 1.5,     # Weitere Stops bei hoher Vol
                'extreme': 2.0   # Sehr weite Stops bei extremer Vol
            }
            
            regime_multiplier = regime_adjustments.get(vol_metrics.regime, 1.0)
            final_stop_distance = stop_distance * regime_multiplier
            
            return current_price * (1 - final_stop_distance)
            
        except Exception as e:
            logger.error(f"Fehler bei Stop Loss Berechnung: {e}")
            return current_price * 0.95  # 5% Default Stop
    
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """
        Gibt Strategie-spezifische Metriken zurück
        """
        return {
            'vol_target': self.vol_target,
            'current_regime': getattr(self, 'current_regime', 'unknown'),
            'position_count': len(self.position_history),
            'avg_position_size': np.mean([p.get('size', 0) for p in self.position_history]) if self.position_history else 0,
            'defensive_mode': True,
            'max_position_size': self.max_position_size
        }