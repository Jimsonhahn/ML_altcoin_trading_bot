"""
ML Components Package
Provides ML functionality for all trading strategies
"""

from .ml_manager import MLManager, get_ml_manager

# Optional imports with fallbacks
try:
    from .market_predictor import MarketPredictor
    from .alpha_finder import AlphaFinder
    from .model_trainer import ModelTrainer
    HAS_FULL_ML = True
except ImportError:
    HAS_FULL_ML = False
    MarketPredictor = None
    AlphaFinder = None
    ModelTrainer = None

__all__ = [
    'MLManager',
    'get_ml_manager',
    'MarketPredictor',
    'AlphaFinder', 
    'ModelTrainer',
    'HAS_FULL_ML'
]