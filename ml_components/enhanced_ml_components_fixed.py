"""
Enhanced ML Components - Fixed version with proper import handling
Integration der neuen ML-Features mit bestehenden Komponenten
"""

import logging
from typing import Dict, Optional, Any, List, Tuple
import pandas as pd
from datetime import datetime

# Core imports
from config.settings import Settings
from utils.exceptions import MLError, ConfigurationError

# Base ML components import with fallback
try:
    from ml_components import MLComponents
    HAS_BASE_ML = True
except ImportError as e:
    logging.warning(f"Base ML components not available: {e}")
    HAS_BASE_ML = False
    # Create fallback base class
    class MLComponents:
        def __init__(self, *args, **kwargs):
            self.settings = kwargs.get('settings') or args[0] if args else None
        
        def get_model_performance(self):
            return {'status': 'fallback', 'components': []}
        
        def stop(self):
            pass

# Enhanced ML components import with fallback
try:
    from ml.ml_manager import get_ml_manager
    HAS_ENHANCED_ML = True
except ImportError as e:
    logging.warning(f"Enhanced ML components not available: {e}")
    HAS_ENHANCED_ML = False
    
    # Create fallback ML manager
    class FallbackMLManager:
        def __init__(self, config):
            self.config = config
        
        def start(self):
            pass
        
        def stop(self):
            pass
        
        def get_prediction(self, symbol):
            return {'predicted_phase': 'unknown', 'confidence': 0.0, 'stale': True}
        
        def get_alpha_signals(self, symbol=None):
            return []
        
        def get_actionable_signals(self, min_strength=0.3):
            return []
        
        def get_status(self):
            return {'enabled': False, 'fallback': True}
    
    def get_ml_manager(config):
        return FallbackMLManager(config)

logger = logging.getLogger(__name__)


class EnhancedMLComponents(MLComponents):
    """
    Erweiterte ML-Komponenten, die die neuen ML-Features (MarketPredictor, AlphaFinder) 
    mit den bestehenden ML-Komponenten kombinieren
    """
    
    def __init__(self, settings: Settings, **kwargs):
        """Initialize enhanced ML components with error handling"""
        try:
            # Initialize base ML components
            if HAS_BASE_ML:
                super().__init__(settings, **kwargs)
            else:
                # Fallback initialization
                self.settings = settings
                logger.warning("Using fallback ML components initialization")
            
            # Enhanced ML features
            self.ml_manager = None
            self.enhanced_features_enabled = HAS_ENHANCED_ML
            
            # Configuration
            self.ml_config = self._build_ml_config(settings)
            
            # Initialize enhanced ML features
            if self.enhanced_features_enabled:
                self._initialize_enhanced_ml()
            else:
                logger.warning("Enhanced ML features disabled - using fallback")
                self.ml_manager = get_ml_manager({})
            
            logger.info("Enhanced ML Components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Enhanced ML Components: {e}")
            raise MLError(f"Failed to initialize Enhanced ML Components: {e}")
    
    def _build_ml_config(self, settings: Settings) -> Dict[str, Any]:
        """Build ML configuration from settings"""
        try:
            base_config = {
                'enabled': True,
                'symbols': settings.get('symbols', ['BTC/USDT', 'ETH/USDT']),
                'update_interval': 300,
                'alpha_update_interval': 600
            }
            
            # Enhanced features configuration
            enhanced_config = settings.get('ml.enhanced_features', {})
            if enhanced_config:
                base_config.update({
                    'market_predictor': enhanced_config.get('market_predictor', {}),
                    'alpha_finder': enhanced_config.get('alpha_finder', {}),
                    'model_trainer': enhanced_config.get('model_trainer', {})
                })
            
            # Fallback configuration
            if not enhanced_config:
                logger.info("Using default ML configuration")
                base_config.update({
                    'market_predictor': {
                        'enabled': True,
                        'model_type': 'lightgbm',
                        'lookback_period': 48,
                        'min_data_points': 1000
                    },
                    'alpha_finder': {
                        'enabled': True,
                        'symbols': ['BTC', 'ETH', 'ADA', 'SOL', 'DOT'],
                        'min_confidence': 0.3
                    },
                    'model_trainer': {
                        'enabled': True,
                        'daily_retrain': True,
                        'retrain_time': '02:00'
                    }
                })
            
            return base_config
            
        except Exception as e:
            logger.error(f"Error building ML config: {e}")
            return {'enabled': False}
    
    def _initialize_enhanced_ml(self) -> None:
        """Initialize enhanced ML features with error handling"""
        try:
            if not HAS_ENHANCED_ML:
                logger.warning("Enhanced ML not available, using fallback")
                self.ml_manager = get_ml_manager({})
                return
            
            # Get ML manager singleton
            self.ml_manager = get_ml_manager(self.ml_config)
            
            # Start ML manager
            if hasattr(self.ml_manager, 'start'):
                self.ml_manager.start()
                logger.info("Enhanced ML manager started")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced ML: {e}")
            # Use fallback
            self.ml_manager = get_ml_manager({})
    
    def start(self):
        """Start all ML components"""
        try:
            # Start base components if available
            if HAS_BASE_ML and hasattr(super(), 'start'):
                super().start()
            
            # Start enhanced ML manager
            if self.ml_manager and hasattr(self.ml_manager, 'start'):
                self.ml_manager.start()
                logger.info("Enhanced ML components started")
            
        except Exception as e:
            logger.error(f"Error starting ML components: {e}")
    
    def stop(self):
        """Stop all ML components"""
        try:
            # Stop enhanced ML
            if self.ml_manager and hasattr(self.ml_manager, 'stop'):
                self.ml_manager.stop()
                logger.info("Enhanced ML components stopped")
            
            # Stop base components if available
            if HAS_BASE_ML and hasattr(super(), 'stop'):
                super().stop()
            
        except Exception as e:
            logger.error(f"Error stopping ML components: {e}")
    
    def get_enhanced_market_prediction(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get enhanced market prediction combining base regime detection with new ML predictions
        """
        try:
            result = {}
            
            # Get base regime prediction if available
            if HAS_BASE_ML and hasattr(self, 'market_regime_detector'):
                try:
                    base_regime = self.market_regime_detector.predict_regime(data)
                    result['base_regime'] = base_regime
                except Exception as e:
                    logger.warning(f"Base regime detection failed: {e}")
            
            # Get enhanced ML prediction
            if self.ml_manager:
                try:
                    ml_prediction = self.ml_manager.get_prediction(symbol)
                    result['ml_prediction'] = ml_prediction
                    
                    # Combine predictions if both available
                    if 'base_regime' in result and not ml_prediction.get('stale', True):
                        result['combined_regime'] = self._combine_predictions(
                            result['base_regime'], 
                            ml_prediction.get('predicted_phase', 'unknown')
                        )
                    else:
                        result['combined_regime'] = ml_prediction.get('predicted_phase', 'unknown')
                except Exception as e:
                    logger.warning(f"Enhanced ML prediction failed: {e}")
                    result['ml_prediction'] = {'error': str(e)}
            
            # Fallback if no predictions available
            if not result:
                result = {
                    'combined_regime': 'unknown',
                    'fallback': True,
                    'error': 'No ML predictions available'
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting enhanced market prediction: {e}")
            return {'error': str(e), 'combined_regime': 'unknown'}
    
    def get_alpha_signals_for_symbol(self, symbol: str) -> List[Dict]:
        """Get alpha signals for a specific symbol"""
        try:
            if self.ml_manager:
                return self.ml_manager.get_alpha_signals(symbol)
            return []
        except Exception as e:
            logger.error(f"Error getting alpha signals for {symbol}: {e}")
            return []
    
    def get_actionable_signals(self, min_strength: float = 0.3) -> List[Dict]:
        """Get actionable trading signals from alpha finder"""
        try:
            if self.ml_manager:
                return self.ml_manager.get_actionable_signals(min_strength)
            return []
        except Exception as e:
            logger.error(f"Error getting actionable signals: {e}")
            return []
    
    def _combine_predictions(self, base_regime: str, ml_phase: str) -> str:
        """
        Combine base regime detection with ML phase prediction
        """
        try:
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
            
            # If they disagree, prefer ML prediction if it's not unknown
            if ml_phase != 'unknown':
                return ml_regime
            
            # Otherwise use base regime
            return base_regime
            
        except Exception as e:
            logger.error(f"Error combining predictions: {e}")
            return base_regime if base_regime else 'unknown'
    
    def enhance_strategy_signal(self, strategy_name: str, base_signal: str, 
                              base_data: Dict, symbol: str, market_data: pd.DataFrame) -> Tuple[str, Dict]:
        """
        Enhance any strategy signal with ML predictions
        """
        try:
            enhanced_data = base_data.copy()
            
            # Get ML predictions
            ml_prediction = None
            alpha_signals = []
            
            if self.ml_manager:
                try:
                    ml_prediction = self.ml_manager.get_prediction(symbol)
                    alpha_signals = self.get_alpha_signals_for_symbol(symbol)
                except Exception as e:
                    logger.warning(f"Failed to get ML data for signal enhancement: {e}")
            
            # Add ML data to signal
            enhanced_data['ml_prediction'] = ml_prediction
            enhanced_data['alpha_signals'] = alpha_signals
            enhanced_data['ml_enhanced'] = True
            
            # Adjust confidence based on ML
            if ml_prediction and not ml_prediction.get('stale', True):
                ml_confidence = ml_prediction.get('confidence', 0.0)
                base_confidence = base_data.get('confidence', 0.5)
                
                # Check alignment between signal and ML prediction
                ml_phase = ml_prediction.get('predicted_phase', 'unknown')
                if self._is_signal_aligned(base_signal, ml_phase):
                    # Boost confidence when aligned
                    enhanced_confidence = base_confidence + (1 - base_confidence) * 0.3
                    enhanced_data['ml_boost'] = True
                else:
                    # Reduce confidence when misaligned
                    enhanced_confidence = base_confidence * 0.7
                    enhanced_data['ml_reduction'] = True
                
                enhanced_data['confidence'] = min(enhanced_confidence, 0.95)
                enhanced_data['original_confidence'] = base_confidence
            
            # Factor in alpha signals
            if alpha_signals:
                try:
                    alpha_strength = sum(s.get('strength', 0) for s in alpha_signals) / len(alpha_signals)
                    enhanced_data['alpha_strength'] = alpha_strength
                    
                    # Adjust confidence based on alpha signals
                    if 'confidence' in enhanced_data:
                        if alpha_strength > 0 and base_signal == 'BUY':
                            enhanced_data['confidence'] *= (1 + alpha_strength * 0.1)
                        elif alpha_strength < 0 and base_signal == 'SELL':
                            enhanced_data['confidence'] *= (1 + abs(alpha_strength) * 0.1)
                except Exception as e:
                    logger.warning(f"Error processing alpha signals: {e}")
            
            return base_signal, enhanced_data
            
        except Exception as e:
            logger.error(f"Error enhancing strategy signal: {e}")
            # Return original signal on error
            return base_signal, base_data
    
    def _is_signal_aligned(self, signal: str, market_phase: str) -> bool:
        """Check if signal aligns with market phase"""
        try:
            alignment = {
                'bull': ['BUY'],
                'bear': ['SELL', 'HOLD'],
                'sideways': ['HOLD'],
                'volatile': ['HOLD'],
                'extreme_fear': ['HOLD', 'SELL']
            }
            
            return signal in alignment.get(market_phase, [])
        except Exception as e:
            logger.error(f"Error checking signal alignment: {e}")
            return False
    
    def get_ml_status(self) -> Dict[str, Any]:
        """Get status of all ML components"""
        try:
            status = {
                'enhanced_ml_available': HAS_ENHANCED_ML,
                'base_ml_available': HAS_BASE_ML,
                'ml_manager_active': self.ml_manager is not None
            }
            
            # Get base ML status if available
            if HAS_BASE_ML:
                try:
                    base_status = super().get_model_performance()
                    status['base_ml'] = base_status
                except Exception as e:
                    status['base_ml'] = {'error': str(e)}
            
            # Get enhanced ML status
            if self.ml_manager:
                try:
                    enhanced_status = self.ml_manager.get_status()
                    status['enhanced_ml'] = enhanced_status
                except Exception as e:
                    status['enhanced_ml'] = {'error': str(e)}
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting ML status: {e}")
            return {'error': str(e)}
    
    def predict_regime(self, data: pd.DataFrame) -> str:
        """Predict market regime - compatibility method"""
        try:
            # Try enhanced ML first
            if self.ml_manager:
                prediction = self.ml_manager.get_prediction('BTC/USDT')  # Use default symbol
                if not prediction.get('stale', True):
                    return prediction.get('predicted_phase', 'unknown')
            
            # Fallback to base ML if available
            if HAS_BASE_ML and hasattr(self, 'market_regime_detector'):
                return self.market_regime_detector.predict_regime(data)
            
            # Final fallback
            return 'unknown'
            
        except Exception as e:
            logger.error(f"Error predicting regime: {e}")
            return 'unknown'


def create_enhanced_ml_components(settings: Settings, **kwargs) -> EnhancedMLComponents:
    """
    Factory function to create enhanced ML components with error handling
    """
    try:
        return EnhancedMLComponents(settings, **kwargs)
    except Exception as e:
        logger.error(f"Failed to create enhanced ML components: {e}")
        
        # Return fallback ML components
        try:
            if HAS_BASE_ML:
                return MLComponents(settings, **kwargs)
            else:
                # Create minimal fallback
                class FallbackMLComponents:
                    def __init__(self, settings, **kwargs):
                        self.settings = settings
                        logger.warning("Using minimal fallback ML components")
                    
                    def start(self):
                        pass
                    
                    def stop(self):
                        pass
                    
                    def get_ml_status(self):
                        return {'status': 'fallback', 'error': 'ML components not available'}
                    
                    def predict_regime(self, data):
                        return 'unknown'
                    
                    def get_enhanced_market_prediction(self, symbol, data):
                        return {'combined_regime': 'unknown', 'fallback': True}
                    
                    def get_alpha_signals_for_symbol(self, symbol):
                        return []
                    
                    def get_actionable_signals(self, min_strength=0.3):
                        return []
                    
                    def enhance_strategy_signal(self, strategy_name, base_signal, base_data, symbol, market_data):
                        return base_signal, base_data
                
                return FallbackMLComponents(settings, **kwargs)
                
        except Exception as e2:
            logger.error(f"Failed to create fallback ML components: {e2}")
            raise MLError(f"All ML component initialization attempts failed: {e}, {e2}")