#!/usr/bin/env python3
"""
QuantumOrchestrator-Compatible Ultimate BTC Strategy Wrapper
============================================================

Wrapper für Integration in QuantumOrchestrator mit event-driven Signals
"""

from typing import Dict, Any, Tuple, Optional
from datetime import datetime
import logging

from core.indicator_engine import IndicatorEngine
from strategies.strategy_base import Strategy

logger = logging.getLogger(__name__)


class QuantumUltimateBTCAdapter(Strategy):
    """
    QuantumOrchestrator-kompatible Version der Ultimate BTC Strategy
    
    Features:
    - Event-driven signal generation
    - QuantumOrchestrator Signal protocol
    - No lookahead bias
    - Adaptive thresholds
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Quantum-compatible adapter"""
        super().__init__(config)
        
        # Strategy configuration
        self.max_position_size = self.config.get('max_position_size', 0.6)
        self.min_signal_strength = self.config.get('min_signal_strength', 0.4)
        self.risk_management_enabled = self.config.get('risk_management_enabled', True)
        self.regime_detection_enabled = self.config.get('regime_detection_enabled', True)
        
        # Event-driven indicator engine
        self.indicator_engine = IndicatorEngine()
        
        # Adaptive thresholds
        self.adaptive_thresholds = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'momentum_bullish': 0.08,
            'momentum_bearish': -0.08,
            'volume_high': 1.2,
            'volume_low': 0.8
        }
        
        # Performance tracking
        self.performance_tracker = {
            'total_signals': 0,
            'successful_signals': 0,
            'false_signals': 0,
            'avg_signal_strength': 0.0
        }
        
        # Signal history for QuantumOrchestrator
        self.signal_history = []
        
        logger.info("QuantumUltimateBTCAdapter initialized")
    
    def process_market_tick(self, price: float, volume: float, timestamp: datetime = None) -> Dict[str, Any]:
        """
        Process individual market tick (QuantumOrchestrator interface)
        
        Args:
            price: Current market price
            volume: Current volume
            timestamp: Tick timestamp
            
        Returns:
            Dict with current market state and indicators
        """
        try:
            # Update indicators with new tick
            indicators = self.indicator_engine.update(price, volume, timestamp)
            
            # Get market regime
            regime = self._detect_market_regime(indicators)
            
            # Calculate signal strength
            signal_strength = self._calculate_signal_strength(indicators, regime)
            
            # Update adaptive thresholds
            volatility = indicators.get('volatility_20d', 0.02)
            self._update_adaptive_thresholds(volatility)
            
            # Create market state for QuantumOrchestrator
            market_state = {
                'timestamp': timestamp or datetime.now(),
                'price': price,
                'volume': volume,
                'indicators': indicators,
                'regime': regime,
                'signal_strength': signal_strength,
                'quality_score': self._calculate_quality_score(indicators, signal_strength),
                'adaptive_thresholds': self.adaptive_thresholds.copy(),
                'strategy_state': {
                    'total_signals': self.performance_tracker['total_signals'],
                    'avg_signal_strength': self.performance_tracker['avg_signal_strength']
                }
            }
            
            return market_state
            
        except Exception as e:
            logger.error(f"Market tick processing failed: {e}")
            return {'error': str(e), 'timestamp': timestamp or datetime.now()}
    
    def generate_quantum_signal(self, market_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate QuantumOrchestrator-compatible signal
        
        Args:
            market_state: Current market state from process_market_tick()
            
        Returns:
            QuantumOrchestrator Signal protocol
        """
        try:
            signal_strength = market_state.get('signal_strength', 0.0)
            quality_score = market_state.get('quality_score', 0.0)
            regime = market_state.get('regime', 'unknown')
            
            # Determine signal direction
            if abs(signal_strength) >= self.min_signal_strength and quality_score > 0.5:
                direction = 'buy' if signal_strength > 0 else 'sell'
                confidence = min(abs(signal_strength) * quality_score, 1.0)
            else:
                direction = 'hold'
                confidence = 0.0
            
            # Create QuantumOrchestrator Signal
            quantum_signal = {
                'strategy_id': 'ultimate_btc_v4',
                'symbol': 'BTC/USDT',
                'direction': direction,
                'strength': abs(signal_strength),
                'confidence': confidence,
                'regime': regime,
                'timestamp': market_state.get('timestamp', datetime.now()),
                'metadata': {
                    'quality_score': quality_score,
                    'adaptive_thresholds': market_state.get('adaptive_thresholds', {}),
                    'indicators': {
                        'rsi_14': market_state.get('indicators', {}).get('rsi_14'),
                        'macd_12_26': market_state.get('indicators', {}).get('macd_12_26'),
                        'volume_ratio_20': market_state.get('indicators', {}).get('volume_ratio_20'),
                        'momentum_20d': market_state.get('indicators', {}).get('momentum_20d')
                    },
                    'signal_components': self._get_signal_components(market_state),
                    'risk_factors': self._assess_risk_factors(market_state)
                }
            }
            
            # Update tracking
            self.performance_tracker['total_signals'] += 1
            self.signal_history.append(quantum_signal)
            
            # Limit history size
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]
            
            logger.info(f"Quantum signal generated: {direction} (strength: {signal_strength:.3f})")
            return quantum_signal
            
        except Exception as e:
            logger.error(f"Quantum signal generation failed: {e}")
            return {
                'strategy_id': 'ultimate_btc_v4',
                'symbol': 'BTC/USDT',
                'direction': 'hold',
                'strength': 0.0,
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def _detect_market_regime(self, indicators: Dict[str, float]) -> str:
        """Detect market regime using indicators"""
        # Implementation from original strategy
        # (Same logic as before but using indicators dict)
        
        if not self.regime_detection_enabled:
            return 'unknown'
        
        try:
            sma_20 = indicators.get('sma_20')
            sma_50 = indicators.get('sma_50') 
            momentum_20d = indicators.get('momentum_20d')
            volatility_20d = indicators.get('volatility_20d', 0.02)
            
            if not all([sma_20, sma_50, momentum_20d is not None]):
                return 'unknown'
            
            current_price = self.indicator_engine.state['price_history'][-1] if self.indicator_engine.state['price_history'] else None
            if not current_price:
                return 'unknown'
            
            price_vs_sma20 = (current_price / sma_20 - 1) if sma_20 > 0 else 0
            price_vs_sma50 = (current_price / sma_50 - 1) if sma_50 > 0 else 0
            
            self._update_adaptive_thresholds(volatility_20d)
            
            bull_strong_threshold = self.adaptive_thresholds.get('momentum_bullish', 0.08)
            bear_strong_threshold = self.adaptive_thresholds.get('momentum_bearish', -0.08)
            
            if price_vs_sma20 > 0.05 and price_vs_sma50 > 0.03 and momentum_20d > bull_strong_threshold:
                return "bull_strong"
            elif price_vs_sma20 > 0.02 and price_vs_sma50 > 0.01 and momentum_20d > bull_strong_threshold/2:
                return "bull_moderate"
            elif price_vs_sma20 < -0.05 and price_vs_sma50 < -0.03 and momentum_20d < bear_strong_threshold:
                return "bear_strong"
            elif price_vs_sma20 < -0.02 and price_vs_sma50 < -0.01 and momentum_20d < bear_strong_threshold/2:
                return "bear_moderate"
            else:
                return "sideways"
                
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
            return 'unknown'
    
    def _calculate_signal_strength(self, indicators: Dict[str, float], regime: str) -> float:
        """Calculate signal strength using multi-strategy ensemble"""
        # Implementation from original strategy
        # (Detailed signal calculation logic)
        try:
            # ... (Same signal calculation as in original strategy)
            # Simplified for wrapper example
            signal_components = []
            
            # MACD ensemble
            macd_score = self._calculate_macd_score(indicators)
            signal_components.append(macd_score * 0.30)
            
            # RSI score
            rsi_score = self._calculate_rsi_score(indicators)
            signal_components.append(rsi_score * 0.25)
            
            # Trend score
            trend_score = self._calculate_trend_score(indicators)
            signal_components.append(trend_score * 0.25)
            
            # Volume score
            volume_score = self._calculate_volume_score(indicators)
            signal_components.append(volume_score * 0.20)
            
            # Base signal
            base_signal = sum(signal_components)
            
            # Apply regime multiplier
            regime_multipliers = {
                'bull_strong': 1.5,
                'bull_moderate': 1.2,
                'sideways': 0.8,
                'bear_moderate': 0.5,
                'bear_strong': 0.3,
                'unknown': 1.0
            }
            
            regime_multiplier = regime_multipliers.get(regime, 1.0)
            final_signal = base_signal * regime_multiplier
            
            return min(max(final_signal, -1), 1)
            
        except Exception as e:
            logger.error(f"Signal strength calculation failed: {e}")
            return 0.0
    
    def _calculate_quality_score(self, indicators: Dict[str, float], signal_strength: float) -> float:
        """Calculate signal quality score"""
        try:
            volatility = indicators.get('volatility_20d', 0.02)
            volume_confirmation = min(indicators.get('volume_ratio_20', 1), 2.0)
            trend_consistency = abs(indicators.get('momentum_20d', 0))
            
            quality_score = (
                abs(signal_strength) * 0.4 +
                min(volume_confirmation / 1.5, 1) * 0.3 +
                min(trend_consistency * 10, 1) * 0.2 +
                min(1 / (volatility * 50 + 0.1), 1) * 0.1
            )
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            logger.error(f"Quality score calculation failed: {e}")
            return 0.0
    
    def _update_adaptive_thresholds(self, volatility: float):
        """Update adaptive thresholds based on market volatility"""
        try:
            vol_factor = min(volatility / 0.02, 2.0)
            
            self.adaptive_thresholds.update({
                'rsi_oversold': max(30 - (vol_factor * 5), 20),
                'rsi_overbought': min(70 + (vol_factor * 5), 80),
                'momentum_bullish': 0.08 * (1 + vol_factor * 0.5),
                'momentum_bearish': -0.08 * (1 + vol_factor * 0.5),
                'volume_high': 1.2 + (vol_factor * 0.3),
                'volume_low': 0.8 - (vol_factor * 0.1)
            })
            
        except Exception as e:
            logger.warning(f"Failed to update adaptive thresholds: {e}")
    
    def _calculate_macd_score(self, indicators: Dict[str, float]) -> float:
        """Calculate MACD component score"""
        # Simplified implementation
        macd_12_26 = indicators.get('macd_12_26', 0)
        macd_signal = indicators.get('macd_12_26_signal', 0)
        
        if macd_12_26 > macd_signal:
            return 1.0
        elif macd_12_26 < macd_signal:
            return -1.0
        else:
            return 0.0
    
    def _calculate_rsi_score(self, indicators: Dict[str, float]) -> float:
        """Calculate RSI component score"""
        rsi_14 = indicators.get('rsi_14')
        if rsi_14 is None:
            return 0.0
        
        oversold = self.adaptive_thresholds['rsi_oversold']
        overbought = self.adaptive_thresholds['rsi_overbought']
        
        if rsi_14 < oversold:
            return 1.0
        elif rsi_14 > overbought:
            return -1.0
        else:
            return 0.0
    
    def _calculate_trend_score(self, indicators: Dict[str, float]) -> float:
        """Calculate trend component score"""
        momentum_20d = indicators.get('momentum_20d', 0)
        bullish_threshold = self.adaptive_thresholds['momentum_bullish']
        bearish_threshold = self.adaptive_thresholds['momentum_bearish']
        
        if momentum_20d > bullish_threshold:
            return 1.0
        elif momentum_20d < bearish_threshold:
            return -1.0
        else:
            return 0.0
    
    def _calculate_volume_score(self, indicators: Dict[str, float]) -> float:
        """Calculate volume component score"""
        volume_ratio = indicators.get('volume_ratio_20', 1.0)
        volume_high = self.adaptive_thresholds['volume_high']
        volume_low = self.adaptive_thresholds['volume_low']
        
        if volume_ratio > volume_high:
            return 1.0
        elif volume_ratio < volume_low:
            return -1.0
        else:
            return 0.0
    
    def _get_signal_components(self, market_state: Dict[str, Any]) -> Dict[str, float]:
        """Get detailed signal component breakdown"""
        indicators = market_state.get('indicators', {})
        
        return {
            'macd_score': self._calculate_macd_score(indicators),
            'rsi_score': self._calculate_rsi_score(indicators),
            'trend_score': self._calculate_trend_score(indicators),
            'volume_score': self._calculate_volume_score(indicators)
        }
    
    def _assess_risk_factors(self, market_state: Dict[str, Any]) -> Dict[str, float]:
        """Assess current risk factors"""
        indicators = market_state.get('indicators', {})
        
        return {
            'volatility_risk': min(indicators.get('volatility_20d', 0.02) / 0.05, 1.0),
            'momentum_risk': abs(indicators.get('momentum_20d', 0)) / 0.2,
            'volume_risk': abs(indicators.get('volume_ratio_20', 1.0) - 1.0),
            'regime_stability': 1.0 if market_state.get('regime') != 'unknown' else 0.0
        }
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information"""
        return {
            'name': 'Quantum Ultimate BTC Strategy',
            'version': '4.0 Event-Driven',
            'description': 'Event-driven Ultimate BTC Strategy for QuantumOrchestrator',
            'features': [
                'No lookahead bias',
                'Event-driven indicators', 
                'Adaptive thresholds',
                'QuantumOrchestrator compatible',
                'Multi-strategy ensemble'
            ],
            'performance_tracker': self.performance_tracker.copy(),
            'adaptive_thresholds': self.adaptive_thresholds.copy()
        }
