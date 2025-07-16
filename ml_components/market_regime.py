import logging
import os
import pandas as pd
import numpy as np
import pickle
from typing import Dict, Any, List, Optional
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer

from ml_components.feature_extraction import FeatureExtractor  # Import the new class
from data_sources.data_manager import DataManager
from config.settings import Settings

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """
    Detects market regimes using clustering on extracted features.
    """

    def __init__(self, settings: Settings, data_cache_dir: str, models_dir: str, output_dir: str,
                 core_symbols: List[str], min_data_points_required: int):
        self.settings = settings
        self.data_cache_dir = data_cache_dir
        self.models_dir = models_dir
        self.output_dir = output_dir
        self.model_path = os.path.join(self.models_dir, "regime_model.pkl")
        self.scaler_path = os.path.join(self.models_dir, "regime_scaler.pkl")
        self.regime_map_path = os.path.join(self.models_dir, "regime_map.json")

        self.n_regimes = self.settings.get('ml.n_regimes', 5)
        self.core_symbols = core_symbols
        self.min_data_points_required = min_data_points_required

        self.model: Optional[MiniBatchKMeans] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_extractor = FeatureExtractor(self.settings)  # Instantiate FeatureExtractor
        self.regime_map: Dict[int, str] = {}
        self.model_trained = False
        self.imputer = SimpleImputer(strategy='mean')

        # Store expected feature names from the FeatureExtractor for all core symbols
        self.expected_feature_columns: List[str] = self.feature_extractor.get_expected_feature_names(self.core_symbols)

        self.last_regime_info: Dict[str, Any] = {"status": "not_available", "label": "unknown", "regime": -1}

        logger.info(
            f"MarketRegimeDetector initialized with {len(self.core_symbols)} core symbols and min_data_points_required={self.min_data_points_required}.")

    def load_model(self) -> bool:
        """Loads the pre-trained K-Means model and scaler."""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            if os.path.exists(self.regime_map_path):
                with open(self.regime_map_path, 'r') as f:
                    self.regime_map = json.load(f)
            self.model_trained = True
            logger.info(f"Regime-Modell geladen von {self.model_path}")
            logger.info(f"Regime-Mapping geladen: {self.regime_map}")
            return True
        except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
            logger.warning(f"Konnte Regime-Modell nicht laden: {e}. Modell muss möglicherweise trainiert werden.")
            self.model_trained = False
            return False
        except Exception as e:
            logger.error(f"Unerwarteter Fehler beim Laden des Regime-Modells: {e}")
            self.model_trained = False
            return False

    def save_model(self) -> None:
        """Saves the trained K-Means model and scaler."""
        if self.model and self.scaler:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            with open(self.regime_map_path, 'w') as f:
                json.dump(self.regime_map, f, indent=4)
            logger.info(f"Regime-Modell gespeichert unter {self.model_path}")
            logger.info(f"Regime-Mapping gespeichert unter {self.regime_map_path}")
        else:
            logger.warning("Kein Modell zum Speichern vorhanden. Bitte zuerst trainieren.")

    def _get_data_for_features(self, data_manager: DataManager, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetches historical data for feature extraction."""
        market_data: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                df = data_manager.get_data(
                    symbol=symbol,
                    timeframe=self.settings.get('timeframes.analysis', '1h'),
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

    def extract_features(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Extracts features for market regime detection from aggregated market data.
        Ensures consistent feature names based on `self.core_symbols`.
        """
        all_features_list = []

        # Iterate over the core symbols to ensure all expected features are generated
        for symbol in self.core_symbols:
            if symbol in market_data and not market_data[symbol].empty and len(
                    market_data[symbol]) >= self.min_data_points_required:
                # Extract features for each time point in the DataFrame for this symbol
                # FeatureExtractor.extract_features returns a single-row DataFrame for the latest point.
                # To get features for *all* timestamps for training, we need to iterate or refactor FeatureExtractor
                # For training, FeatureExtractor should be able to produce features for a window of data, not just the latest.

                # For now, let's assume `extract_features` can process the full historical DataFrame
                # and return features for each timestamp, or we adapt to just the last valid point for each window.
                # Given the previous context, extract_features was intended to run on the full DF for feature engineering.

                # Let's adjust for consistency. The `feature_extractor.extract_features(df, symbol)` is currently
                # designed to return features for the *latest* row. For training, we need features over time.
                # This implies a need for FeatureExtractor to have a method like `extract_historical_features(df, symbol)`.

                # For a quick fix and to get it running with existing `extract_features` structure,
                # we will extract features for the *last valid point* of each window,
                # effectively creating a time series of feature vectors.

                df_to_process = market_data[symbol]

                # This is a placeholder. A proper implementation would extract features for every row
                # or a sliding window. For now, to match the previous behavior, let's extract for the last point.
                # The `_get_data_for_features` already ensures enough data.

                # The original `extract_market_regime_features` in `feature_extraction.py` processed the whole df
                # and returned a single dictionary based on means/latest values.
                # The current `FeatureExtractor.extract_features` returns for the latest row.
                # We need features for the whole historical window `min_data_points_required`.

                # Let's re-align to extract historical features:
                # This needs FeatureExtractor to offer a batch processing or apply over windows.

                # Simplest way for now: Extract for all points, then drop NaNs.
                # This might be slow if `extract_features` creates a new DF each time.

                # Let's assume `FeatureExtractor.extract_features` is designed to be applied to a sliding window of data
                # to generate a time series of features. If it just gives the last, we need to adjust.

                # Re-reading `FeatureExtractor.extract_features`: "Returns a DataFrame with a single row (latest timestamp)"
                # This means we can't directly use it for historical features.
                # We need to manually apply it over the historical data window to get features for each time point.

                features_time_series_for_symbol = []
                # Ensure enough data for a full window
                if len(df_to_process) >= self.min_data_points_required:
                    # Slide a window to extract features for each point
                    for i in range(self.min_data_points_required - 1, len(df_to_process)):
                        window_df = df_to_process.iloc[i - (self.min_data_points_required - 1): i + 1]
                        if len(window_df) == self.min_data_points_required:  # Ensure full window
                            current_features = self.feature_extractor.extract_features(window_df, symbol)
                            if not current_features.empty:
                                all_features_list.append(current_features)
                else:
                    logger.warning(
                        f"Nicht genügend Daten im Fenster für {symbol}. Benötigt: {self.min_data_points_required}, vorhanden: {len(df_to_process)}.")
            else:
                logger.warning(f"Keine ausreichenden Daten für Kernsymbol {symbol} zur Feature-Extraktion vorhanden.")

        if not all_features_list:
            raise ValueError(
                "Keine Features aus den Marktdaten extrahiert. Überprüfen Sie Datenverfügbarkeit und min_data_points_required.")

        # Concatenate features from all symbols and all timestamps
        combined_features = pd.concat(all_features_list, axis=1)  # Concatenate columns, will broadcast by index

        # Drop rows where any feature is NaN (likely from initial rolling windows)
        combined_features = combined_features.dropna()

        if combined_features.empty:
            raise ValueError("Keine vollständigen Features nach NaN-Bereinigung verfügbar.")

        # Ensure all expected columns are present, filling missing with NaN if a symbol was skipped
        for col in self.expected_feature_columns:
            if col not in combined_features.columns:
                combined_features[col] = np.nan

        # Reorder columns to match expected order defined during initialization
        combined_features = combined_features[self.expected_feature_columns]

        # Impute NaNs *before* fitting/transforming
        # Fit imputer first if it's the first time
        if self.imputer is None:
            self.imputer = SimpleImputer(strategy='mean')
            self.imputer.fit(combined_features)

        combined_features_imputed = pd.DataFrame(self.imputer.transform(combined_features),
                                                 columns=combined_features.columns,
                                                 index=combined_features.index)

        logger.info(
            f"Features extrahiert: {len(combined_features_imputed)} Zeitpunkte, {len(combined_features_imputed.columns)} Features")
        return combined_features_imputed

    def train_model(self, symbols: List[str]) -> None:
        """
        Trains the Market Regime Detector model (K-Means) and scaler.
        Uses `self.core_symbols` for consistency.
        """
        logger.info("Starte das Training des Marktregime-Detektors...")
        try:
            market_data = self._get_data_for_features(DataManager(self.settings), self.core_symbols)
            features = self.extract_features(market_data)  # This will get historical features

            # Fit imputer here if not already done during extract_features
            if self.imputer is None or not hasattr(self.imputer, 'statistics_'):  # Check if imputer is fitted
                self.imputer = SimpleImputer(strategy='mean')
                self.imputer.fit(features)
            features_imputed = pd.DataFrame(self.imputer.transform(features), columns=features.columns,
                                            index=features.index)

            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features_imputed)

            # Initialize KMeans model
            self.model = MiniBatchKMeans(
                n_clusters=self.n_regimes,
                random_state=42,
                n_init='auto',
                batch_size=256
            )

            self.model.fit(features_scaled)
            self.model_trained = True

            self._assign_regime_labels(features_imputed, self.model.labels_)

            logger.info("Marktregime-Detektor erfolgreich trainiert.")
        except Exception as e:
            self.model_trained = False
            logger.error(f"Fehler beim Training des Marktregime-Detektors: {e}")
            raise

    def _assign_regime_labels(self, features: pd.DataFrame, labels: np.ndarray):
        """Assigns meaningful labels to each detected regime based on feature means."""
        self.regime_map = {}
        # Ensure labels cover all indices in range(n_regimes)
        unique_labels = np.unique(labels)

        for i in range(self.n_regimes):
            if i not in unique_labels:
                self.regime_map[i] = f"Unused Regime {i}"
                continue

            regime_features = features[labels == i].mean()

            description_parts = []

            # Example: Using prefixes like 'BTC_USDT_volatility_20'
            # Get average volatility across core symbols for this regime
            avg_vol = regime_features.filter(like='volatility_20').mean()
            global_avg_vol = features.filter(like='volatility_20').mean().mean()  # Overall average for comparison

            if pd.notna(avg_vol):
                if avg_vol > global_avg_vol * 1.5:
                    description_parts.append("Volatile")
                elif avg_vol < global_avg_vol * 0.5:
                    description_parts.append("Low-Volatility")

            # Get average relative strength to MA for trend
            avg_rel_to_ema200 = regime_features.filter(like='rel_to_ema200').mean()
            avg_rsi = regime_features.filter(like='rsi_14').mean()  # Use RSI from extracted features

            if pd.notna(avg_rel_to_ema200) and pd.notna(avg_rsi):
                if avg_rel_to_ema200 > 0.05 and avg_rsi > 60:  # Price 5% above 200 EMA, RSI high
                    description_parts.append("Bullish-Trend")
                elif avg_rel_to_ema200 < -0.05 and avg_rsi < 40:  # Price 5% below 200 EMA, RSI low
                    description_parts.append("Bearish-Trend")
                else:
                    description_parts.append("Sideways/Uncertain")
            else:  # Fallback if trend features are NaN
                description_parts.append("Mixed Trend")

            # Example: BTC Dominance (assuming btc_dom_change feature exists for BTC/USDT)
            # You need a way to get BTC dominance specifically or a general market dominance feature
            # The current `feature_extractor` does not produce `btc_dom_change` across symbols.
            # This requires a feature from a dedicated market sentiment/dominance source or global features.
            # For now, let's remove this if not explicitly generated or available.
            # if 'BTC_USDT_btc_dom_change' in regime_features and pd.notna(regime_features['BTC_USDT_btc_dom_change']):
            #     btc_dom_change = regime_features['BTC_USDT_btc_dom_change']
            #     if btc_dom_change > 0.005:
            #         description_parts.append("BTC-Dominance-Increasing")
            #     elif btc_dom_change < -0.005:
            #         description_parts.append("BTC-Dominance-Decreasing")
            # else:
            #     description_parts.append("Neutral BTC Dominance")

            label = " ".join(description_parts).strip()
            if not label:
                label = f"Mixed Market Regime {i}"

            self.regime_map[i] = label
            logger.debug(f"Assigned label '{label}' to Regime {i} with features: {regime_features.to_dict()}")

        for k, v in self.regime_map.items():
            logger.info(f"Regime {k} = '{v}'")

    def predict_regime(self, data_manager: DataManager) -> Dict[str, Any]:
        """
        Predicts the current market regime.
        Uses `self.core_symbols` for data fetching and feature extraction.
        """
        if not self.model_trained:
            logger.warning("Modell ist nicht trainiert. Kann Regime nicht vorhersagen.")
            return {"status": "error", "error": "Model not trained."}

        try:
            market_data_latest = self._get_data_for_features(data_manager, self.core_symbols)

            if not market_data_latest:
                return {"status": "error", "error": "Nicht genügend aktuelle Marktdaten für Vorhersage."}

            # Extract features for the very last timestamp available
            # This ensures only one row of features is returned
            features = self.feature_extractor.extract_features_for_latest(market_data_latest, self.core_symbols)

            if features.empty:
                return {"status": "error", "error": "Konnte keine Features für die Vorhersage extrahieren."}

            # Ensure features have correct columns and order for prediction
            # Fill missing columns with NaN and impute if necessary
            for col in self.expected_feature_columns:
                if col not in features.columns:
                    features[col] = np.nan
            features = features[self.expected_feature_columns]

            # Impute missing values based on the imputer fitted during training
            features_imputed = pd.DataFrame(self.imputer.transform(features),
                                            columns=features.columns,
                                            index=features.index)

            features_scaled = self.scaler.transform(features_imputed)

            current_regime_label_index = self.model.predict(features_scaled)[-1]

            regime_label = self.regime_map.get(int(current_regime_label_index),
                                               f"Unbekanntes Regime {current_regime_label_index}")

            self.last_regime_info = {
                "status": "available",
                "label": regime_label,
                "regime": int(current_regime_label_index),
                "timestamp": features.index[-1].isoformat(),
                "trading_rules": self.extract_trading_rules(regime_label)
            }
            logger.info(f"Aktuelles Marktregime: {current_regime_label_index} - {regime_label}")
            return self.last_regime_info

        except Exception as e:
            logger.error(f"Fehler bei der Regime-Vorhersage: {e}")
            logger.error(traceback.format_exc())
            return {"status": "error", "error": str(e)}

    def extract_trading_rules(self, regime_label: str) -> Dict[str, Any]:
        """
        Extracts recommended trading rules or biases for a given regime.
        This is a placeholder for more sophisticated rule extraction.
        """
        rules = {
            "top_performers": [],
            "bottom_performers": [],
            "risk_bias": "neutral",
            "strategy_preference": "balanced"
        }

        regime_label_lower = regime_label.lower()

        if "bullish-trend" in regime_label_lower:
            rules["risk_bias"] = "aggressive"
            rules["strategy_preference"] = "trend"
            rules["top_performers"] = ["ETH/USDT", "SOL/USDT"]
            rules["bottom_performers"] = ["XRP/USDT"]

        elif "bearish-trend" in regime_label_lower:
            rules["risk_bias"] = "conservative"
            rules["strategy_preference"] = "reversion"
            rules["top_performers"] = []
            rules["bottom_performers"] = ["BTC/USDT", "ETH/USDT"]

        elif "sideways" in regime_label_lower or "low-volatility" in regime_label_lower:
            rules["risk_bias"] = "neutral"
            rules["strategy_preference"] = "range"
            rules["top_performers"] = []
            rules["bottom_performers"] = []

        elif "volatile" in regime_label_lower:
            rules["risk_bias"] = "conservative"
            rules["strategy_preference"] = "arbitrage"
            rules["top_performers"] = []
            rules["bottom_performers"] = []

        return rules

    def get_regime_label(self, regime_index: int) -> str:
        """Returns the human-readable label for a given regime index."""
        return self.regime_map.get(regime_index, f"Unbekanntes Regime {regime_index}")

    def get_current_regime_info(self) -> Dict[str, Any]:
        """Returns the most recently predicted regime information."""
        return self.last_regime_info