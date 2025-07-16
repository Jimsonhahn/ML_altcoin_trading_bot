import os
import logging
import pickle
from typing import Dict, Any, Optional, List, Tuple

from config.settings import Settings
from data_sources.data_manager import DataManager

# Import your ML components
from ml_components.market_regime import MarketRegimeDetector
from ml_components.asset_clusters import AssetClusterAnalyzer
from ml_components.coin_monitor import NewCoinMonitor
from ml_components.model_monitor import ModelPerformanceMonitor  # Corrected: Imported ModelPerformanceMonitor

logger = logging.getLogger(__name__)


class MLComponents:
    """
    Manages the lifecycle and interaction of all ML-related components.
    """
    _instance: Optional['MLComponents'] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MLComponents, cls).__new__(cls)
        return cls._instance

    def __init__(self, settings: Settings, data_cache_dir: str, models_dir: str, output_dir: str):
        if not hasattr(self, '_initialized'):
            self.settings = settings
            self.data_cache_dir = data_cache_dir
            self.models_dir = models_dir
            self.output_dir = output_dir

            os.makedirs(self.models_dir, exist_ok=True)
            os.makedirs(self.output_dir, exist_ok=True)

            self.regime_core_symbols = self.settings.get('ml.regime_core_symbols',
                                                         ["BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "SOL/USDT"])
            self.min_data_points_for_ml = self.settings.get('ml.min_data_points_for_ml', 200)

            self.market_regime_detector = MarketRegimeDetector(
                settings=self.settings,
                data_cache_dir=self.data_cache_dir,
                models_dir=self.models_dir,
                output_dir=self.output_dir,
                core_symbols=self.regime_core_symbols,
                min_data_points_required=self.min_data_points_for_ml
            )
            self.asset_cluster_analyzer = AssetClusterAnalyzer(
                settings=self.settings,
                data_cache_dir=self.data_cache_dir,
                models_dir=self.models_dir,
                output_dir=self.output_dir,
                core_symbols=self.regime_core_symbols,
                min_data_points_required=self.min_data_points_for_ml
            )
            self.coin_monitor = NewCoinMonitor(data_dir=self.data_cache_dir,
                                               output_dir=os.path.join(self.output_dir, "new_coins"))
            self.model_monitor = ModelPerformanceMonitor(
                output_dir=os.path.join(self.output_dir, "model_monitor"))  # Corrected: Used ModelPerformanceMonitor

            self.current_regime_info: Dict[str, Any] = {"status": "not_available", "label": "unknown", "regime": -1}
            self.current_asset_clusters: Dict[str, Any] = {"status": "not_available", "clusters": {}}

            self._initialized = True
            logger.info("ML Components initialized")

    def load_models(self) -> bool:
        """Loads pre-trained ML models."""
        regime_loaded = self.market_regime_detector.load_model()
        asset_cluster_loaded = self.asset_cluster_analyzer.load_model()

        if regime_loaded and asset_cluster_loaded:
            logger.info("All ML models loaded successfully.")
            return True
        else:
            logger.warning("Some ML models failed to load. They might need training.")
            return False

    def save_models(self) -> None:
        """Saves trained ML models."""
        self.market_regime_detector.save_model()
        self.asset_cluster_analyzer.save_model()
        logger.info("All ML models saved.")

    def train_ml_models(self, symbols_for_training: List[str]) -> None:
        """
        Trains all ML models.
        `symbols_for_training` should typically be the `regime_core_symbols` for consistency.
        """
        logger.info("Starting ML model training...")

        train_symbols = self.regime_core_symbols

        try:
            logger.info(f"Training Market Regime Detector with symbols: {train_symbols}")
            self.market_regime_detector.train_model(train_symbols)
            self.market_regime_detector.save_model()
            self.current_regime_info = self.market_regime_detector.get_current_regime_info()
            logger.info(f"Market Regime Detector trained. Current regime: {self.current_regime_info.get('label')}")
        except Exception as e:
            logger.error(f"Error training Market Regime Detector: {e}")

        try:
            logger.info(f"Training Asset Cluster Analyzer with symbols: {train_symbols}")
            self.asset_cluster_analyzer.train_model(train_symbols)
            self.asset_cluster_analyzer.save_model()
            self.current_asset_clusters = self.asset_cluster_analyzer.get_current_clusters()
            logger.info(f"Asset Cluster Analyzer trained. Clusters: {self.current_asset_clusters.get('clusters')}")
        except Exception as e:
            logger.error(f"Error training Asset Cluster Analyzer: {e}")

        logger.info("ML model training completed.")

    def update_all_components(self, data_manager: DataManager, symbols: List[str]) -> Dict[str, Any]:
        """
        Updates and predicts using all ML components.
        `symbols` here are the active trading pairs, not necessarily the core regime symbols.
        The ML models themselves use `self.regime_core_symbols`.
        """
        update_status = {
            "regime_updated": False,
            "clusters_updated": False,
            "coin_monitor_updated": False,
            "error": None,
            "current_regime": self.current_regime_info,
            "current_clusters": self.current_asset_clusters
        }

        try:
            regime_info = self.market_regime_detector.predict_regime(data_manager=data_manager)
            if regime_info and regime_info.get("status") == "available":
                self.current_regime_info = regime_info
                update_status["regime_updated"] = True
                update_status["current_regime"] = self.current_regime_info
            else:
                logger.warning(
                    f"Market regime prediction failed or not available: {regime_info.get('error', 'Unknown')}")
                update_status["error"] = regime_info.get('error', 'Regime prediction failed.')

            cluster_info = self.asset_cluster_analyzer.analyze_clusters(data_manager=data_manager)
            if cluster_info and cluster_info.get("status") == "available":
                self.current_asset_clusters = cluster_info
                update_status["clusters_updated"] = True
                update_status["current_clusters"] = self.current_asset_clusters
            else:
                logger.warning(f"Asset clustering failed or not available: {cluster_info.get('error', 'Unknown')}")
                update_status["error"] = cluster_info.get('error', 'Clustering failed.')

            if self.settings.get('ml.monitor_new_coins', False):
                new_coins = self.coin_monitor.check_for_new_coins()
                if new_coins:
                    logger.info(f"New coins detected: {new_coins}")

                coin_monitor_results = self.coin_monitor.update_all_coins(asset_analyzer=self.asset_cluster_analyzer)
                if coin_monitor_results['analyzed']:
                    logger.info(f"New coins analyzed by CoinMonitor: {coin_monitor_results['analyzed']}")

                update_status["coin_monitor_updated"] = True

            self.model_monitor.record_performance(
                model_id="MarketRegime",
                model_type="regime",
                prediction=self.current_regime_info.get('regime'),
                actual=self.current_regime_info.get('regime'),
                # In live prediction, 'actual' is same as prediction or needs ground truth
                timestamp=datetime.now()
            )
            # You might need a more sophisticated way to get actuals for a cluster model.
            self.model_monitor.record_performance(
                model_id="AssetCluster",
                model_type="cluster",
                prediction=self.current_asset_clusters.get('clusters'),
                actual=self.current_asset_clusters.get('clusters'),  # Same here
                timestamp=datetime.now()
            )


        except Exception as e:
            logger.error(f"Error updating ML components: {e}")
            update_status["error"] = str(e)

        return update_status

    def get_current_regime_info(self) -> Dict[str, Any]:
        """Returns the last detected market regime information."""
        return self.current_regime_info

    def get_current_asset_clusters(self) -> Dict[str, Any]:
        """Returns the last detected asset clusters."""
        return self.current_asset_clusters


_ml_components_instance: Optional[MLComponents] = None


def initialize_ml(settings: Settings, data_cache_dir: str = "data/market_data",
                  models_dir: str = "data/ml_models", output_dir: str = "data/ml_analysis") -> MLComponents:
    """Initializes and returns the singleton MLComponents instance."""
    global _ml_components_instance
    if _ml_components_instance is None:
        _ml_components_instance = MLComponents(settings, data_cache_dir, models_dir, output_dir)
    return _ml_components_instance


def get_ml_components() -> Optional[MLComponents]:
    """Returns the singleton MLComponents instance if initialized, else None."""
    return _ml_components_instance