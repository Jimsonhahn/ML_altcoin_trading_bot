"""
ML Manager - Zentrale Verwaltung aller ML-Komponenten
Läuft automatisch mit dem Trading Bot und stellt ML-Funktionalität für alle Strategien bereit
"""

import logging
import asyncio
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd

try:
    from .market_predictor import MarketPredictor
    from .alpha_finder import AlphaFinder
    from .model_trainer import ModelTrainer
    HAS_ML_COMPONENTS = True
except ImportError:
    HAS_ML_COMPONENTS = False
    # Fallback classes
    class MarketPredictor:
        def __init__(self, *args, **kwargs):
            pass
        def predict(self, data):
            return {'predicted_phase': 'sideways', 'confidence': 0.5}
    
    class AlphaFinder:
        def __init__(self, *args, **kwargs):
            pass
        async def find_alpha_signals(self):
            return []
        def get_actionable_signals(self):
            return []
    
    class ModelTrainer:
        def __init__(self, *args, **kwargs):
            pass
        def start_scheduler(self):
            pass


class MLManager:
    """
    Zentrale ML-Verwaltung für alle Trading-Strategien
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # ML Components
        self.market_predictor = None
        self.alpha_finder = None
        self.model_trainer = None
        
        # Configuration
        self.enabled = self.config.get('enabled', True)
        self.update_interval = self.config.get('update_interval', 300)  # 5 minutes
        self.alpha_update_interval = self.config.get('alpha_update_interval', 600)  # 10 minutes
        
        # State
        self.is_running = False
        self.last_market_update = None
        self.last_alpha_update = None
        self.current_predictions = {}
        self.current_alpha_signals = []
        
        # Background tasks
        self.update_thread = None
        self.alpha_thread = None
        
        # Initialize components
        if self.enabled:
            self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize all ML components"""
        try:
            self.logger.info("Initializing ML components...")
            
            # Market Predictor
            predictor_config = self.config.get('market_predictor', {})
            self.market_predictor = MarketPredictor(predictor_config)
            
            # Alpha Finder
            alpha_config = self.config.get('alpha_finder', {})
            self.alpha_finder = AlphaFinder(alpha_config)
            
            # Model Trainer
            trainer_config = self.config.get('model_trainer', {})
            self.model_trainer = ModelTrainer(trainer_config)
            
            # Load existing models if available
            if hasattr(self.market_predictor, 'load_model'):
                if self.market_predictor.load_model():
                    self.logger.info("Loaded existing market predictor model")
                else:
                    self.logger.info("No existing model found, will train on first run")
            
            self.logger.info("ML components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ML components: {e}")
            self.enabled = False
    
    def start(self) -> None:
        """Start ML manager background tasks"""
        if not self.enabled:
            self.logger.warning("ML Manager disabled, not starting")
            return
        
        if self.is_running:
            self.logger.warning("ML Manager already running")
            return
        
        try:
            self.is_running = True
            
            # Start model trainer scheduler
            if self.model_trainer:
                self.model_trainer.start_scheduler()
            
            # Start background update threads
            self.update_thread = threading.Thread(
                target=self._run_market_updates,
                daemon=True,
                name="MLManager-MarketUpdates"
            )
            self.update_thread.start()
            
            self.alpha_thread = threading.Thread(
                target=self._run_alpha_updates,
                daemon=True,
                name="MLManager-AlphaUpdates"
            )
            self.alpha_thread.start()
            
            self.logger.info("ML Manager started")
            
        except Exception as e:
            self.logger.error(f"Failed to start ML Manager: {e}")
            self.is_running = False
    
    def stop(self) -> None:
        """Stop ML manager"""
        try:
            self.is_running = False
            
            # Stop model trainer
            if self.model_trainer:
                self.model_trainer.stop_scheduler()
            
            # Wait for threads to finish
            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=5)
            
            if self.alpha_thread and self.alpha_thread.is_alive():
                self.alpha_thread.join(timeout=5)
            
            self.logger.info("ML Manager stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping ML Manager: {e}")
    
    def _run_market_updates(self) -> None:
        """Background thread for market predictions"""
        while self.is_running:
            try:
                # Update market predictions
                asyncio.run(self._update_market_predictions())
                
                # Wait for next update
                asyncio.run(asyncio.sleep(self.update_interval))
                
            except Exception as e:
                self.logger.error(f"Error in market update thread: {e}")
                asyncio.run(asyncio.sleep(60))  # Wait 1 minute on error
    
    def _run_alpha_updates(self) -> None:
        """Background thread for alpha signals"""
        while self.is_running:
            try:
                # Update alpha signals
                asyncio.run(self._update_alpha_signals())
                
                # Wait for next update
                asyncio.run(asyncio.sleep(self.alpha_update_interval))
                
            except Exception as e:
                self.logger.error(f"Error in alpha update thread: {e}")
                asyncio.run(asyncio.sleep(60))  # Wait 1 minute on error
    
    async def _update_market_predictions(self) -> None:
        """Update market predictions for all symbols"""
        try:
            if not self.market_predictor:
                return
            
            # Get active symbols from config
            symbols = self.config.get('symbols', ['BTC/USDT', 'ETH/USDT'])
            
            for symbol in symbols:
                # Get market data (would come from data manager)
                data = await self._get_market_data(symbol)
                if data is not None and not data.empty:
                    prediction = self.market_predictor.predict(data)
                    self.current_predictions[symbol] = {
                        'prediction': prediction,
                        'timestamp': datetime.now()
                    }
            
            self.last_market_update = datetime.now()
            self.logger.debug(f"Updated market predictions for {len(symbols)} symbols")
            
        except Exception as e:
            self.logger.error(f"Error updating market predictions: {e}")
    
    async def _update_alpha_signals(self) -> None:
        """Update alpha signals"""
        try:
            if not self.alpha_finder:
                return
            
            # Find new alpha signals
            signals = await self.alpha_finder.find_alpha_signals()
            self.current_alpha_signals = signals
            self.last_alpha_update = datetime.now()
            
            self.logger.debug(f"Found {len(signals)} alpha signals")
            
        except Exception as e:
            self.logger.error(f"Error updating alpha signals: {e}")
    
    async def _get_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get market data for a symbol"""
        # This would be integrated with your data manager
        # For now, return None as placeholder
        return None
    
    def get_prediction(self, symbol: str) -> Dict[str, Any]:
        """Get current prediction for a symbol"""
        if symbol in self.current_predictions:
            pred_data = self.current_predictions[symbol]
            # Check if prediction is recent (< 10 minutes old)
            if datetime.now() - pred_data['timestamp'] < timedelta(minutes=10):
                return pred_data['prediction']
        
        # Return default prediction
        return {
            'predicted_phase': 'unknown',
            'confidence': 0.0,
            'stale': True
        }
    
    def get_alpha_signals(self, symbol: Optional[str] = None) -> List[Any]:
        """Get current alpha signals"""
        if not self.current_alpha_signals:
            return []
        
        if symbol:
            # Filter by symbol
            symbol_base = symbol.split('/')[0]
            return [s for s in self.current_alpha_signals if s.symbol == symbol_base]
        
        return self.current_alpha_signals
    
    def get_actionable_signals(self, min_strength: float = 0.3) -> List[Dict]:
        """Get actionable alpha signals"""
        if self.alpha_finder:
            return self.alpha_finder.get_actionable_signals(min_strength)
        return []
    
    def get_status(self) -> Dict[str, Any]:
        """Get ML manager status"""
        return {
            'enabled': self.enabled,
            'is_running': self.is_running,
            'components': {
                'market_predictor': self.market_predictor is not None,
                'alpha_finder': self.alpha_finder is not None,
                'model_trainer': self.model_trainer is not None
            },
            'last_updates': {
                'market': self.last_market_update.isoformat() if self.last_market_update else None,
                'alpha': self.last_alpha_update.isoformat() if self.last_alpha_update else None
            },
            'predictions_count': len(self.current_predictions),
            'alpha_signals_count': len(self.current_alpha_signals),
            'model_trainer_status': self.model_trainer.get_training_status() if self.model_trainer else None
        }
    
    def train_models(self, symbols: List[str], data: Dict[str, pd.DataFrame]) -> bool:
        """Train ML models with provided data"""
        try:
            if not self.market_predictor:
                return False
            
            # Combine data from all symbols
            combined_data = pd.concat(list(data.values()), ignore_index=True)
            
            # Train market predictor
            success = self.market_predictor.train_model(combined_data, retrain=True)
            
            if success:
                self.logger.info("ML models trained successfully")
            else:
                self.logger.error("ML model training failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error training models: {e}")
            return False


# Singleton instance
_ml_manager_instance = None


def get_ml_manager(config: Optional[Dict] = None) -> MLManager:
    """Get or create ML manager singleton"""
    global _ml_manager_instance
    
    if _ml_manager_instance is None:
        _ml_manager_instance = MLManager(config)
    
    return _ml_manager_instance