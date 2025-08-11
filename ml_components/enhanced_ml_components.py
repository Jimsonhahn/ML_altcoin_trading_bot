"""
Enhanced ML Components - Integration der neuen ML-Features mit bestehenden Komponenten
"""

import logging
from typing import Dict, Optional, Any, List, Tuple
import pandas as pd
from datetime import datetime

from ml_components import MLComponents
from ml.ml_manager import get_ml_manager

logger = logging.getLogger(__name__)


class EnhancedMLComponents(MLComponents):
    """
    Erweiterte ML-Komponenten, die die neuen ML-Features (MarketPredictor, AlphaFinder) 
    mit den bestehenden ML-Komponenten kombinieren
    """
    
    def __init__(self, settings: 'Settings'):
        # Initialize base ML components
        super().__init__(settings)
        
        # Initialize enhanced ML features
        self.ml_manager = None
        self._initialize_enhanced_ml()
        
        logger.info("Enhanced ML Components initialized")
    
    def _initialize_enhanced_ml(self) -> None:
        """Initialize enhanced ML features"""
        try:
            # ML Manager configuration
            ml_config = {
                'enabled': True,
                'market_predictor': {
                    'model_type': self.settings.get('ml.predictor_model', 'lightgbm'),
                    'lookback_period': self.settings.get('ml.lookback_period', 48),
                    'min_data_points': self.settings.get('ml.min_data_points_for_ml', 1000)
                },
                'alpha_finder': {
                    'symbols': self.settings.get('ml.regime_core_symbols', ['BTC', 'ETH', 'ADA', 'SOL', 'DOT']),
                    'lookback_hours': 24,
                    'min_confidence': 0.3
                },
                'model_trainer': {
                    'training_schedule': {
                        'daily_retrain': self.settings.get('ml.auto_retrain', True),
                        'retrain_time': '02:00',
                        'performance_check_interval': 6
                    }
                },
                'symbols': self.settings.get('symbols', []),
                'update_interval': 300,
                'alpha_update_interval': 600
            }
            
            # Get ML manager singleton
            self.ml_manager = get_ml_manager(ml_config)
            
            # Start ML manager
            self.ml_manager.start()
            
            logger.info("Enhanced ML features initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced ML features: {e}")
    
    def get_enhanced_market_prediction(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get enhanced market prediction combining base regime detection with new ML predictions
        """
        try:
            result = {}
            
            # Get base regime prediction
            if hasattr(self, 'market_regime_detector'):
                base_regime = self.market_regime_detector.predict_regime(data)
                result['base_regime'] = base_regime
            
            # Get enhanced ML prediction
            if self.ml_manager:
                ml_prediction = self.ml_manager.get_prediction(symbol)
                result['ml_prediction'] = ml_prediction
                
                # Combine predictions
                if 'base_regime' in result and not ml_prediction.get('stale', True):
                    # Weight predictions (60% ML, 40% base)
                    result['combined_regime'] = self._combine_predictions(
                        base_regime, 
                        ml_prediction.get('predicted_phase', 'unknown')
                    )
                else:
                    result['combined_regime'] = result.get('base_regime', 'unknown')
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting enhanced market prediction: {e}")
            return {'error': str(e)}
    
    def get_alpha_signals_for_symbol(self, symbol: str) -> List[Dict]:
        """Get alpha signals for a specific symbol"""
        if self.ml_manager:
            return self.ml_manager.get_alpha_signals(symbol)
        return []
    
    def get_actionable_signals(self, min_strength: float = 0.3) -> List[Dict]:
        """Get actionable trading signals from alpha finder"""
        if self.ml_manager:
            return self.ml_manager.get_actionable_signals(min_strength)
        return []
    
    def _combine_predictions(self, base_regime: str, ml_phase: str) -> str:
        """
        Combine base regime detection with ML phase prediction
        """
        # Map ML phases to regimes
        phase_to_regime_map = {
            'bull': 'uptrend',
            'bear': 'downtrend',
            'sideways': 'ranging',
            'volatile': 'high_volatility',
            'extreme_fear': 'crash'
        }
        
        ml_regime = phase_to_regime_map.get(ml_phase, base_regime)
        
        # If predictions agree, use them
        if base_regime == ml_regime:
            return base_regime
        
        # If they disagree, use weighted combination
        # For now, prefer ML prediction if confidence is high
        return ml_regime
    
    def enhance_strategy_signal(self, strategy_name: str, base_signal: str, 
                              base_data: Dict, symbol: str, market_data: pd.DataFrame) -> Tuple[str, Dict]:
        """
        Enhance any strategy signal with ML predictions
        """
        try:
            enhanced_data = base_data.copy()
            
            # Get ML predictions
            ml_prediction = self.ml_manager.get_prediction(symbol) if self.ml_manager else None
            alpha_signals = self.get_alpha_signals_for_symbol(symbol)
            
            # Add ML data
            enhanced_data['ml_prediction'] = ml_prediction
            enhanced_data['alpha_signals'] = alpha_signals
            
            # Adjust confidence based on ML
            if ml_prediction and not ml_prediction.get('stale', True):
                ml_confidence = ml_prediction.get('confidence', 0.0)
                base_confidence = base_data.get('confidence', 0.5)
                
                # Check alignment
                ml_phase = ml_prediction.get('predicted_phase', 'unknown')
                if self._is_signal_aligned(base_signal, ml_phase):
                    # Boost confidence
                    enhanced_confidence = base_confidence + (1 - base_confidence) * 0.3
                else:
                    # Reduce confidence
                    enhanced_confidence = base_confidence * 0.7
                
                enhanced_data['confidence'] = min(enhanced_confidence, 0.95)
                enhanced_data['ml_enhanced'] = True
            
            # Check alpha signals
            if alpha_signals:
                alpha_strength = sum(s.get('strength', 0) for s in alpha_signals) / len(alpha_signals)
                enhanced_data['alpha_strength'] = alpha_strength
                
                # Further adjust based on alpha
                if 'confidence' in enhanced_data:
                    if alpha_strength > 0 and base_signal == 'BUY':
                        enhanced_data['confidence'] *= (1 + alpha_strength * 0.1)
                    elif alpha_strength < 0 and base_signal == 'SELL':
                        enhanced_data['confidence'] *= (1 + abs(alpha_strength) * 0.1)
            
            return base_signal, enhanced_data
            
        except Exception as e:
            logger.error(f"Error enhancing strategy signal: {e}")
            return base_signal, base_data
    
    def _is_signal_aligned(self, signal: str, market_phase: str) -> bool:
        """Check if signal aligns with market phase"""
        alignment = {
            'bull': ['BUY'],
            'bear': ['SELL', 'HOLD'],
            'sideways': ['HOLD'],
            'volatile': ['HOLD'],
            'extreme_fear': ['HOLD', 'SELL']
        }
        
        return signal in alignment.get(market_phase, [])
    
    def get_ml_status(self) -> Dict[str, Any]:
        """Get status of all ML components"""
        status = super().get_model_performance()  # Base ML status
        
        # Add enhanced ML status
        if self.ml_manager:
            status['enhanced_ml'] = self.ml_manager.get_status()
        
        return status
    
    def stop(self) -> None:
        """Stop all ML components"""
        # Stop enhanced ML
        if self.ml_manager:
            self.ml_manager.stop()
        
        # Stop base components
        super().stop()


def create_enhanced_ml_components(settings: 'Settings') -> EnhancedMLComponents:
    """Factory function to create enhanced ML components"""
    try:
        return EnhancedMLComponents(settings)
    except Exception as e:
        logger.error(f"Failed to create enhanced ML components: {e}")
        # Return base ML components as fallback
        return MLComponents(settings)