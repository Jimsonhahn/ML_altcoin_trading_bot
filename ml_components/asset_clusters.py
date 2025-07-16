import logging
import os
import pandas as pd
import numpy as np
import pickle
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import pdist, squareform
from sklearn.impute import SimpleImputer

from ml_components.feature_extraction import FeatureExtractor  # Import the new class
from data_sources.data_manager import DataManager
from config.settings import Settings

logger = logging.getLogger(__name__)


class AssetClusterAnalyzer:
    """
    Analyzes asset clusters based on their correlation and features.
    """

    def __init__(self, settings: Settings, data_cache_dir: str, models_dir: str, output_dir: str,
                 core_symbols: List[str], min_data_points_required: int):
        self.settings = settings
        self.data_cache_dir = data_cache_dir
        self.models_dir = models_dir
        self.output_dir = output_dir
        self.cluster_model_path = os.path.join(self.models_dir, "asset_cluster_model.pkl")
        self.scaler_path = os.path.join(self.models_dir, "asset_cluster_scaler.pkl")
        self.mds_path = os.path.join(self.models_dir, "asset_mds.pkl")
        self.imputer_path = os.path.join(self.models_dir, "asset_imputer.pkl")

        self.core_symbols = core_symbols
        self.min_data_points_required = min_data_points_required

        self.cluster_model: Optional[MiniBatchKMeans] = None
        self.scaler: Optional[StandardScaler] = None
        self.mds: Optional[MDS] = None
        self.imputer: Optional[SimpleImputer] = None
        self.feature_extractor = FeatureExtractor(self.settings)  # Instantiate FeatureExtractor
        self.model_trained = False

        self.last_clusters: Dict[str, Any] = {"status": "not_available", "clusters": {}, "performance": {}}

        logger.info(
            f"AssetClusterAnalyzer initialized with {len(self.core_symbols)} core symbols and min_data_points_required={self.min_data_points_required}.")

    def load_model(self) -> bool:
        """Loads the pre-trained clustering model, scaler, MDS, and imputer."""
        try:
            with open(self.cluster_model_path, 'rb') as f:
                self.cluster_model = pickle.load(f)
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            with open(self.mds_path, 'rb') as f:
                self.mds = pickle.load(f)
            with open(self.imputer_path, 'rb') as f:
                self.imputer = pickle.load(f)

            self.model_trained = True
            logger.info(f"Asset-Cluster-Modell geladen von {self.cluster_model_path}")
            return True
        except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
            logger.warning(
                f"Konnte Asset-Cluster-Modell nicht laden: {e}. Modell muss möglicherweise trainiert werden.")
            self.model_trained = False
            return False
        except Exception as e:
            logger.error(f"Unerwarteter Fehler beim Laden des Asset-Cluster-Modells: {e}")
            self.model_trained = False
            return False

    def save_model(self) -> None:
        """Saves the trained clustering model, scaler, MDS, and imputer."""
        if self.cluster_model and self.scaler and self.mds and self.imputer:
            with open(self.cluster_model_path, 'wb') as f:
                pickle.dump(self.cluster_model, f)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            with open(self.mds_path, 'wb') as f:
                pickle.dump(self.mds, f)
            with open(self.imputer_path, 'wb') as f:
                pickle.dump(self.imputer, f)
            logger.info(f"Asset-Cluster-Modell gespeichert unter {self.cluster_model_path}")
        else:
            logger.warning("Kein Asset-Cluster-Modell zum Speichern vorhanden. Bitte zuerst trainieren.")

    def _get_data_for_features(self, data_manager: DataManager, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetches historical data for feature extraction."""
        market_data: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                # Use daily data for asset clustering features (correlations, longer-term stats)
                df = data_manager.get_data(
                    symbol=symbol,
                    timeframe=self.settings.get('timeframes.secondary', '1d'),  # Use secondary/daily timeframe
                    limit=self.min_data_points_required
                )
                if df is not None and not df.empty and len(df) >= self.min_data_points_required:
                    market_data[symbol] = df
                    logger.debug(f"Daten für {symbol} geladen: {len(df)} Einträge")
                else:
                    logger.warning(
                        f"Unzureichende Daten für {symbol} ({len(df) if df is not None else 0} Einträge). Benötigt: {self.min_data_points_required}.")
            except Exception as e:
                logger.error(f"Fehler beim Laden der Daten für {symbol}: {e}")

        if not market_data:
            raise ValueError("Keine ausreichenden Marktdaten für die Feature-Extraktion verfügbar.")

        logger.info(f"Marktdaten für {len(market_data)} Symbole geladen")
        return market_data

    def _extract_asset_features_for_clustering(self, market_data: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
        """
        Extracts features for each individual asset for clustering purposes.
        This produces a DataFrame where each row is an asset and columns are its features.
        """
        asset_features_list = []
        feature_names_template = self.feature_extractor.get_expected_feature_names_for_asset_clustering()  # New method on FeatureExtractor

        for symbol in self.core_symbols:
            if symbol in market_data and not market_data[symbol].empty:
                df = market_data[symbol]
                if len(df) >= self.min_data_points_required:
                    # Extract features for this asset.
                    # This should return a single row of features specific to this asset for clustering.
                    # FeatureExtractor.extract_features can be adapted or a new method added for this.
                    # For now, let's just use `extract_features` but expect single row features.
                    # `extract_features` is already designed to output features for a single timestamp.
                    # We need features representative of the asset's behavior over the `min_data_points_required` window.

                    # The original `extract_asset_clustering_features` function calculated aggregate stats (mean_return, volatility etc.)
                    # over the entire DataFrame. This is what we need.
                    # So, FeatureExtractor should have a method like `extract_asset_summary_features`.

                    # For now, let's directly call the internal function for asset clustering features
                    # This implies FeatureExtractor needs to expose/adapt this.
                    # Let's add a method `extract_asset_summary_features` to FeatureExtractor.

                    # Temporarily, use the old function if it's available and modify FeatureExtractor to include it.
                    # Assuming `extract_asset_clustering_features` will be a method of FeatureExtractor.
                    asset_summary_features_dict = self.feature_extractor.extract_asset_summary_features(df)

                    if asset_summary_features_dict and asset_summary_features_dict.get("status", "error") != "error":
                        features = asset_summary_features_dict.get("features", {})
                        # Convert to Series and add symbol as index
                        asset_features_series = pd.Series(features, name=symbol)
                        asset_features_list.append(asset_features_series)
                    else:
                        logger.warning(
                            f"Konnte keine Asset-Features für {symbol} extrahieren: {asset_summary_features_dict.get('message', 'Unbekannter Fehler')}")
                else:
                    logger.warning(f"Nicht genügend Daten für Asset-Features für {symbol}.")
            else:
                logger.warning(f"Keine Marktdaten für Kernsymbol {symbol} zur Asset-Feature-Extraktion vorhanden.")

        if not asset_features_list:
            logger.error(
                "Keine Asset-Features extrahiert. Überprüfen Sie die Datenverfügbarkeit und min_data_points_required.")
            return None

        # Combine all asset feature series into a single DataFrame (assets as rows, features as columns)
        asset_features_df = pd.DataFrame(asset_features_list)

        # Ensure all expected feature columns are present, filling missing with NaN
        # `feature_names_template` should list all possible feature columns *for a single asset*
        for col in feature_names_template:
            if col not in asset_features_df.columns:
                asset_features_df[col] = np.nan

        asset_features_df = asset_features_df[feature_names_template]  # Reorder columns

        # Impute missing values
        if self.imputer is None:
            self.imputer = SimpleImputer(strategy='mean')
            self.imputer.fit(asset_features_df)

        asset_features_df_imputed = pd.DataFrame(self.imputer.transform(asset_features_df),
                                                 columns=asset_features_df.columns,
                                                 index=asset_features_df.index)

        logger.info(f"Features für {len(asset_features_df_imputed)} Assets extrahiert.")
        return asset_features_df_imputed

    def train_model(self, symbols: List[str]) -> None:
        """
        Trains the asset clustering model (K-Means) and related components.
        Uses `self.core_symbols` for consistency.
        """
        logger.info("Starte das Training des Asset-Cluster-Analysators...")
        try:
            market_data = self._get_data_for_features(DataManager(self.settings), self.core_symbols)
            features_df = self._extract_asset_features_for_clustering(market_data)

            if features_df is None or features_df.empty:
                raise ValueError("Nicht genügend Features zum Trainieren des Asset-Cluster-Modells verfügbar.")

            # Imputer is fitted in _extract_asset_features_for_clustering or here if needed
            if self.imputer is None or not hasattr(self.imputer, 'statistics_'):
                self.imputer = SimpleImputer(strategy='mean')
                self.imputer.fit(features_df)
            features_imputed = pd.DataFrame(self.imputer.transform(features_df), columns=features_df.columns,
                                            index=features_df.index)

            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features_imputed)

            n_clusters = self.settings.get('ml.n_regimes', 3)
            if n_clusters < 2: n_clusters = 2

            self.cluster_model = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init='auto',
                batch_size=256
            )
            self.cluster_model.fit(features_scaled)

            # Fit MDS on the scaled features for visualization
            self.mds = MDS(n_components=2, random_state=42, normalized_stress='auto', metric=True)
            # MDS needs distance matrix on samples
            distances = pdist(features_scaled, 'euclidean')
            self.mds.fit(distances)  # Fit MDS with the distance matrix

            self.model_trained = True
            logger.info("Asset-Cluster-Analysator erfolgreich trainiert.")

        except Exception as e:
            self.model_trained = False
            logger.error(f"Fehler beim Training des Asset-Cluster-Analysators: {e}")
            raise

    def analyze_clusters(self, data_manager: DataManager) -> Dict[str, Any]:
        """
        Analyzes and predicts asset clusters based on current data.
        Uses `self.core_symbols` for data consistency.
        """
        if not self.model_trained:
            logger.warning("Asset-Cluster-Modell ist nicht trainiert. Kann Cluster nicht analysieren.")
            return {"status": "error", "error": "Model not trained."}

        try:
            # Use core symbols for prediction consistency
            market_data = self._get_data_for_features(data_manager, self.core_symbols)
            features_df = self._extract_asset_features_for_clustering(market_data)

            if features_df is None or features_df.empty:
                return {"status": "error", "error": "Nicht genügend aktuelle Daten für Cluster-Analyse."}

            # Ensure features have correct columns and order for prediction
            expected_cols = self.imputer.feature_names_in_ if self.imputer else features_df.columns
            for col in expected_cols:
                if col not in features_df.columns:
                    features_df[col] = np.nan
            features_df = features_df[expected_cols]

            features_imputed = pd.DataFrame(self.imputer.transform(features_df),
                                            columns=features_df.columns,
                                            index=features_df.index)

            features_scaled = self.scaler.transform(features_imputed)

            # Predict clusters for each asset
            cluster_labels = self.cluster_model.predict(features_scaled)

            # Organize assets by cluster
            clusters: Dict[int, List[str]] = {i: [] for i in range(self.cluster_model.n_clusters)}
            for i, symbol in enumerate(features_df.index):  # Use index as symbol names
                clusters[cluster_labels[i]].append(symbol)

            organized_clusters = {str(k): v for k, v in clusters.items() if v}

            # Calculate performance metrics using the new features_scaled and predicted labels
            performance_metrics = self._analyze_cluster_performance(features_scaled, cluster_labels)

            self.last_clusters = {
                "status": "available",
                "timestamp": datetime.now().isoformat(),
                # Use current time, as features_df index might not be single point
                "clusters": organized_clusters,
                "performance": performance_metrics
            }
            logger.info(
                f"Asset-Clustering abgeschlossen: {len(self.core_symbols)} Assets in {self.cluster_model.n_clusters} Clustern.")
            logger.info(f"Aktuelle Asset-Cluster: {organized_clusters}")
            return self.last_clusters

        except Exception as e:
            logger.error(f"Fehler bei der Asset-Cluster-Analyse: {e}")
            logger.error(traceback.format_exc())
            return {"status": "error", "error": str(e)}

    def _analyze_cluster_performance(self, features_scaled: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Calculates and returns clustering performance metrics.
        """
        performance_metrics = {}
        n_clusters = self.cluster_model.n_clusters

        if n_clusters < 2 or len(np.unique(labels)) < 2:
            logger.warning(
                "Nicht genügend Cluster oder Datenpunkte für Silhouette-Score oder Davies-Bouldin-Score (min. 2 Cluster benötigt).")
            performance_metrics['silhouette_score'] = np.nan
            performance_metrics['davies_bouldin_index'] = np.nan
            return performance_metrics

        try:
            if features_scaled.shape[0] >= 2 and len(np.unique(labels)) >= 2:
                performance_metrics['silhouette_score'] = silhouette_score(features_scaled, labels)
            else:
                performance_metrics['silhouette_score'] = np.nan

            if features_scaled.shape[0] >= 2 and len(np.unique(labels)) >= 2:
                performance_metrics['davies_bouldin_index'] = davies_bouldin_score(features_scaled, labels)
            else:
                performance_metrics['davies_bouldin_index'] = np.nan

        except Exception as e:
            logger.error(f"Fehler bei der Analyse der Cluster-Performance: {e}")
            logger.error(traceback.format_exc())
            performance_metrics['silhouette_score'] = np.nan
            performance_metrics['davies_bouldin_index'] = np.nan

        return performance_metrics

    def get_current_clusters(self) -> Dict[str, Any]:
        """Returns the most recently analyzed cluster information."""
        return self.last_clusters