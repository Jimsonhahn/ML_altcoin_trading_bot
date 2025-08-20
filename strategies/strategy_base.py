"""
Base Strategy Class
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Tuple, Any, Optional
import pandas as pd
import logging
from core.interfaces import IStrategy

logger = logging.getLogger(__name__)

class Signal(Enum):
    """Trading Signal Enum"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class Strategy(IStrategy):
    """Abstract base class for all trading strategies"""

    def __init__(self, params: Dict = None, ml_components: Optional[Any] = None):
        self.params = params or {}
        self.name = self.__class__.__name__
        self.ml_components = ml_components
        self.ml_enhanced = ml_components is not None
        
        # ML enhancement configuration
        self.ml_config = {
            'use_ml_predictions': params.get('use_ml_predictions', True),
            'ml_weight': params.get('ml_weight', 0.3),  # 30% ML, 70% original strategy
            'ml_confidence_threshold': params.get('ml_confidence_threshold', 0.5),
            'adjust_position_size': params.get('adjust_position_size', True),
            'adjust_risk_params': params.get('adjust_risk_params', True)
        }
        
        if self.ml_enhanced:
            logger.info(f"{self.name} initialized with ML enhancement")

    @abstractmethod
    def calculate_signal(self, symbol: str, data: pd.DataFrame, 
                        current_price: float) -> Tuple[str, Dict[str, Any]]:
        """
        Calculate trading signal

        Returns:
            Tuple of (signal_string, signal_data_dict)
            Example: ('BUY', {'confidence': 0.8, 'reason': 'momentum'})
        """
        pass
    
    def calculate_ml_enhanced_signal(self, symbol: str, data: pd.DataFrame,
                                   current_price: float) -> Tuple[str, Dict[str, Any]]:
        """
        Calculate ML-enhanced trading signal
        Combines base strategy signal with ML predictions
        """
        # Get base strategy signal
        base_signal, base_data = self.calculate_signal(symbol, data, current_price)
        
        # If ML not enabled or not available, return base signal
        if not self.ml_enhanced or not self.ml_config['use_ml_predictions']:
            return base_signal, base_data
        
        try:
            # Get ML predictions
            ml_predictions = self._get_ml_predictions(symbol, data)
            
            # Enhance signal with ML
            enhanced_signal, enhanced_data = self._combine_with_ml(
                base_signal, base_data, ml_predictions
            )
            
            return enhanced_signal, enhanced_data
            
        except Exception as e:
            logger.warning(f"ML enhancement failed for {self.name}: {e}")
            # Fallback to base signal
            return base_signal, base_data
    
    def _get_ml_predictions(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get ML predictions from components
        """
        predictions = {}
        
        if hasattr(self.ml_components, 'market_predictor'):
            # Get market phase prediction
            market_prediction = self.ml_components.market_predictor.predict(data)
            predictions['market_phase'] = market_prediction.get('predicted_phase', 'unknown')
            predictions['market_confidence'] = market_prediction.get('confidence', 0.0)
        
        if hasattr(self.ml_components, 'alpha_finder'):
            # Get alpha signals
            alpha_signals = self.ml_components.alpha_finder.get_actionable_signals()
            symbol_signals = [s for s in alpha_signals if s['symbol'] in symbol]
            predictions['alpha_signals'] = symbol_signals
        
        return predictions
    
    def _combine_with_ml(self, base_signal: str, base_data: Dict,
                        ml_predictions: Dict) -> Tuple[str, Dict[str, Any]]:
        """
        Combine base strategy signal with ML predictions
        """
        enhanced_data = base_data.copy()
        enhanced_data['ml_predictions'] = ml_predictions
        
        # Get ML confidence
        ml_confidence = ml_predictions.get('market_confidence', 0.0)
        
        # Skip ML enhancement if confidence too low
        if ml_confidence < self.ml_config['ml_confidence_threshold']:
            enhanced_data['ml_enhanced'] = False
            return base_signal, enhanced_data
        
        # Market phase alignment
        market_phase = ml_predictions.get('market_phase', 'unknown')
        base_confidence = base_data.get('confidence', 0.5)
        
        # Adjust confidence based on market phase alignment
        if self._is_aligned_with_market(base_signal, market_phase):
            # Boost confidence if aligned
            enhanced_confidence = base_confidence + (1 - base_confidence) * self.ml_config['ml_weight']
        else:
            # Reduce confidence if not aligned
            enhanced_confidence = base_confidence * (1 - self.ml_config['ml_weight'])
        
        enhanced_data['confidence'] = enhanced_confidence
        enhanced_data['ml_enhanced'] = True
        enhanced_data['original_confidence'] = base_confidence
        
        # Check alpha signals
        alpha_signals = ml_predictions.get('alpha_signals', [])
        if alpha_signals:
            # Average alpha signal strength
            alpha_strength = sum(s['strength'] for s in alpha_signals) / len(alpha_signals)
            enhanced_data['alpha_strength'] = alpha_strength
            
            # Further adjust confidence
            if alpha_strength > 0 and base_signal == 'BUY':
                enhanced_confidence *= (1 + alpha_strength * 0.2)
            elif alpha_strength < 0 and base_signal == 'SELL':
                enhanced_confidence *= (1 + abs(alpha_strength) * 0.2)
        
        # Potentially flip signal if confidence drops too low
        if enhanced_confidence < 0.3 and base_signal != 'HOLD':
            return 'HOLD', enhanced_data
        
        enhanced_data['confidence'] = min(enhanced_confidence, 0.95)  # Cap at 95%
        
        return base_signal, enhanced_data
    
    def _is_aligned_with_market(self, signal: str, market_phase: str) -> bool:
        """
        Check if strategy signal aligns with market phase
        """
        alignment_map = {
            'bull': ['BUY'],
            'bear': ['SELL', 'HOLD'],
            'sideways': ['HOLD'],
            'volatile': ['HOLD'],
            'extreme_fear': ['HOLD', 'SELL']
        }
        
        return signal in alignment_map.get(market_phase, [])
    
    def get_ml_adjusted_params(self) -> Dict[str, Any]:
        """
        Get ML-adjusted risk parameters
        """
        adjusted_params = self.params.copy()
        
        if not self.ml_enhanced or not self.ml_config['adjust_risk_params']:
            return adjusted_params
        
        try:
            # Get current market conditions
            if hasattr(self.ml_components, 'market_predictor'):
                # Adjust based on market volatility
                # This would need actual implementation
                pass
            
            return adjusted_params
            
        except Exception as e:
            logger.warning(f"Failed to adjust params with ML: {e}")
            return adjusted_params
    
    def get_risk_adjusted_position_size(self, symbol: str, base_size: float, market_data: Dict = None) -> float:
        """
        Get risk-adjusted position size using safety manager if available
        """
        try:
            # Try to get optimal position size from safety manager (advanced risk management)
            safety_manager = getattr(self, 'safety_manager', None)
            if safety_manager and hasattr(safety_manager, 'get_optimal_position_size'):
                # Create strategy stats for risk calculation
                strategy_stats = {
                    'win_rate': self.params.get('historical_win_rate', 0.55),
                    'avg_win': self.params.get('avg_win', 0.02),
                    'avg_loss': self.params.get('avg_loss', -0.015),
                    'total_trades': self.params.get('total_trades', 50),
                    'profit_factor': self.params.get('profit_factor', 1.2),
                    'sharpe_ratio': self.params.get('sharpe_ratio', 1.0),
                    'max_drawdown': self.params.get('max_drawdown', 0.05),
                    'volatility': self.params.get('volatility', 0.15)
                }
                
                optimal_size = safety_manager.get_optimal_position_size(
                    symbol=symbol,
                    strategy_stats=strategy_stats,
                    market_data=market_data
                )
                
                # Apply some bounds relative to base size
                min_size = base_size * 0.2  # At least 20% of base
                max_size = base_size * 2.0  # At most 200% of base
                
                return max(min_size, min(optimal_size, max_size))
            
            # Fallback: return base size
            return base_size
            
        except Exception as e:
            logger.error(f"Error calculating risk-adjusted position size: {e}")
            return base_size
    
    def set_safety_manager(self, safety_manager):
        """Set safety manager reference for risk-adjusted position sizing"""
        self.safety_manager = safety_manager
    
    # Implementation of IStrategy interface methods
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """
        Generate trading signals for the given symbol and data
        This method wraps the calculate_signal method to match the IStrategy interface
        """
        try:
            current_price = data['close'].iloc[-1] if not data.empty else 0.0
            signal, metadata = self.calculate_signal(symbol, data, current_price)
            
            # Convert signal to the expected format
            confidence = metadata.get('confidence', 0.0)
            signals = {
                'signal_strength': confidence if signal == 'BUY' else -confidence if signal == 'SELL' else 0.0,
                'buy_signal': confidence if signal == 'BUY' else 0.0,
                'sell_signal': confidence if signal == 'SELL' else 0.0
            }
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return {'signal_strength': 0.0, 'buy_signal': 0.0, 'sell_signal': 0.0}
    
    def get_name(self) -> str:
        """Get the name of this strategy"""
        return self.name
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get the parameters of this strategy"""
        return self.params.copy()
    
    @abstractmethod
    def calculate_signal(self, symbol: str, data: pd.DataFrame, current_price: float) -> Tuple[str, Dict[str, Any]]:
        """
        Abstract method that must be implemented by concrete strategy classes
        Returns (signal, metadata) where signal is 'BUY', 'SELL', or 'HOLD'
        """
        pass
