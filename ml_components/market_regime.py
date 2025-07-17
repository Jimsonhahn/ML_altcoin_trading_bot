# ml_components/market_regime.py
import logging
import os
import json
import pandas as pd
import numpy as np
import pickle
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer

# Assume these exist or will be implemented for data fetching and feature extraction
from data_sources.data_manager import DataManager
from ml_components.feature_extraction import FeatureExtractor
from ml_components.market_sentiment import MarketSentimentAnalyzer  # For sentiment data (placeholder for now)
from config.settings import Settings

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """
    Detects market regimes using clustering on extracted features and classifies them.
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
        self.min_data_points_required = min_data_points

        self.model: Optional[MiniBatchKMeans] = None
        self.scaler: Optional[StandardScaler] = None
        # Initialize FeatureExtractor with relevant settings
        self.feature_extractor = FeatureExtractor(self.settings.get('ml.feature_extraction', {}))
        self.sentiment_analyzer = MarketSentimentAnalyzer(self.settings)  # Initialize Sentiment Analyzer
        self.regime_map: Dict[int, str] = {}  # Maps cluster label to human-readable regime
        self.model_trained = False
        self.imputer: Optional[SimpleImputer] = None  # Will be fitted during training

        self.last_regime_info: Dict[str, Any] = {"status": "not_available", "label": "unknown", "regime": -1}

        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info(f"MarketRegimeDetector initialized with {len(self.core_symbols)} core symbols.")
        self.load_model()  # Attempt to load model on initialization

    def load_model(self) -> bool:
        """Loads the pre-trained K-Means model, scaler, and regime map."""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path) and os.path.exists(
                    self.regime_map_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                with open(self.regime_map_path, 'r') as f:
                    self.regime_map = json.load(f)
                self.model_trained = True

                # Re-initialize imputer with feature names from scaler for consistency in prediction
                # This assumes scaler stores feature names, or we need to pass them from training
                if self.scaler and hasattr(self.scaler, 'feature_names_in_'):
                    self.imputer = SimpleImputer(strategy='mean')  # Need to fit this during training
                    # For loading, we don't refit imputer here, we just make sure it's present
                    # Its `feature_names_in_` should ideally come from saved training state or inferred.
                    # For simplicity, if loading, we expect the pipeline (imputer+scaler) to be consistent.

                logger.info("Market regime model, scaler and map loaded successfully.")
                return True
            else:
                logger.warning("No existing market regime model, scaler or map found. Model needs training.")
                return False
        except Exception as e:
            logger.error(f"Error loading market regime model components: {e}")
            self.model_trained = False
            return False

    def save_model(self):
        """Saves the trained K-Means model, scaler, and regime map."""
        try:
            os.makedirs(self.models_dir, exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            with open(self.regime_map_path, 'w') as f:
                json.dump(self.regime_map, f, indent=4)
            logger.info("Market regime model, scaler and map saved successfully.")
        except Exception as e:
            logger.error(f"Error saving market regime model components: {e}")

    def train_model(self, data_manager: DataManager, timeframe: str = '1d',
                    start_date: str = '2022-01-01', end_date: str = '2023-12-31'):
        """
        Trains the K-Means clustering model for market regime detection.
        """
        logger.info("Starting market regime model training...")
        combined_features_list = []  # List to collect dataframes for concatenation

        # Calculate lookback needed for all features
        max_lookback = max(
            self.settings.get('ml.feature_extraction.rsi_period', 14),
            self.settings.get('ml.feature_extraction.ma_long', 50),
            self.settings.get('ml.feature_extraction.bollinger_window', 20),
            self.settings.get('ml.feature_extraction.atr_period', 14)
        )
        required_candles_for_features = max_lookback + self.min_data_points_required  # Min candles to ensure valid features

        for symbol in self.core_symbols:
            logger.info(f"Fetching historical data for {symbol} for training...")
            df = data_manager.get_historical_data(symbol, timeframe, start_date, end_date)

            if df.empty or len(df) < required_candles_for_features:
                logger.warning(f"Insufficient data for {symbol} to train market regime model. "
                               f"Needed {required_candles_for_features}, got {len(df)}. Skipping.")
                continue

            # Ensure index is datetime for proper alignment and feature calculation
            df.index = pd.to_datetime(df.index)

            # Calculate features
            features_df = self.feature_extractor.calculate_technical_indicators(
                df.copy())  # Pass copy to avoid modifying original

            # Add volatility measures explicitly from OHLC
            # ATR needs full OHLC, simple proxy for rolling std dev
            if 'close' in df.columns:
                features_df[f'{symbol.replace("/", "_")}_rolling_std'] = df['close'].rolling(window=20).std()
            if 'high' in df.columns and 'low' in df.columns:
                features_df[f'{symbol.replace("/", "_")}_atr'] = features_df[
                    'atr']  # Use ATR calculated by feature extractor

            # Add sentiment features (from placeholder MarketSentimentAnalyzer)
            sentiment_df = self.sentiment_analyzer.get_historical_sentiment_data(
                start_date=df.index.min().strftime('%Y-%m-%d'),
                end_date=df.index.max().strftime('%Y-%m-%d'),
                timeframe=timeframe
            )
            if not sentiment_df.empty:
                features_df = features_df.join(sentiment_df, how='left')

            # Select features relevant for clustering. Avoid NaNs.
            # Only use columns that are numerical and not all NaN
            numeric_cols = features_df.select_dtypes(include=np.number).columns
            # Drop columns where all values are NaN *after* calculation
            features_for_clustering = features_df[numeric_cols].dropna(axis=1, how='all')

            # Drop rows with NaNs. This ensures we cluster on complete feature sets.
            features_for_clustering = features_for_clustering.dropna()

            if features_for_clustering.empty:
                logger.warning(f"No valid feature data for {symbol} after dropping NaNs. Skipping.")
                continue

            # Rename columns to be unique per symbol for concatenation
            features_for_clustering = features_for_clustering.add_prefix(f"{symbol.replace('/', '_')}_")
            combined_features_list.append(features_for_clustering)

        if not combined_features_list:
            logger.error("No sufficient data available to train market regime model.")
            return False

        # Concatenate all features across symbols, aligning by date index
        full_features_df = pd.concat(combined_features_list, axis=1,
                                     join='inner')  # Use inner join to keep only common dates

        if full_features_df.empty:
            logger.error("No overlapping valid data points across all core symbols for market regime training.")
            return False

        # Drop any remaining columns that might have become all-NaN after inner join if some symbols had no data for certain dates
        full_features_df = full_features_df.dropna(axis=1, how='all')

        if len(full_features_df) < self.n_regimes * 2:  # Ensure enough data points per cluster
            logger.warning(f"Not enough robust data points ({len(full_features_df)}) for {self.n_regimes} clusters. "
                           f"Adjusting n_regimes to {max(1, len(full_features_df) // 2)} or requiring more data.")
            self.n_regimes = max(1, len(full_features_df) // 2)  # Fallback to a smaller number if data is scarce
            if self.n_regimes == 0:
                logger.error("Cannot train model with zero effective data points.")
                return False

        # Impute any remaining NaN values before scaling (e.g., if a new feature column had NaN)
        self.imputer = SimpleImputer(strategy='mean')
        imputed_features = self.imputer.fit_transform(full_features_df)
        imputed_features_df = pd.DataFrame(imputed_features, index=full_features_df.index,
                                           columns=full_features_df.columns)

        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(imputed_features_df)

        self.model = MiniBatchKMeans(n_clusters=self.n_regimes, random_state=42, n_init='auto')
        self.model.fit(scaled_features)
        self.model_trained = True

        # Define regime labels
        self._define_regime_labels(imputed_features_df)  # Use original (imputed) features for interpretation
        self.save_model()
        logger.info("Market regime model trained successfully.")
        return True

    def _define_regime_labels(self, features_df: pd.DataFrame):
        """
        Assign human-readable labels to clusters based on their characteristics.
        This is a simplified example and can be made more sophisticated using domain knowledge.
        """
        if not self.model or not self.model_trained:
            logger.error("Model not trained, cannot define regime labels.")
            return

        cluster_centers = self.scaler.inverse_transform(self.model.cluster_centers_)
        cluster_centers_df = pd.DataFrame(cluster_centers, columns=features_df.columns)

        self.regime_map = {}
        for cluster_id in range(self.n_regimes):
            label = "unknown"
            center_features = cluster_centers_df.iloc[cluster_id]

            # Heuristics based on key market indicators for the primary symbols
            # Example using BTC/USDT as a primary indicator for overall market
            btc_prefix = "BTC_USDT_"

            # Trend indicators (e.g., SMA crossover, RSI)
            try:
                # Check if columns exist before accessing
                sma_short_gt_long = center_features.get(f'{btc_prefix}sma_20', 0) > center_features.get(
                    f'{btc_prefix}sma_50', 0)
                rsi_high = center_features.get(f'{btc_prefix}rsi', 50) > 60
                rsi_low = center_features.get(f'{btc_prefix}rsi', 50) < 40
            except KeyError:
                logger.warning("BTC/USDT trend indicators not found in cluster features. Skipping trend heuristic.")
                sma_short_gt_long = False
                rsi_high = False
                rsi_low = False

            # Volatility (e.g., ATR, rolling std dev)
            try:
                atr_high = center_features.get(f'{btc_prefix}atr', 0) > self.settings.get(
                    'ml.feature_extraction.atr_threshold_high', 0.05)  # Example threshold
                rolling_std_high = center_features.get(f'{btc_prefix}rolling_std', 0) > self.settings.get(
                    'ml.feature_extraction.std_threshold_high', 0.03)
                atr_low = center_features.get(f'{btc_prefix}atr', 0) < self.settings.get(
                    'ml.feature_extraction.atr_threshold_low', 0.01)
            except KeyError:
                logger.warning(
                    "BTC/USDT volatility indicators not found in cluster features. Skipping volatility heuristic.")
                atr_high = False
                rolling_std_high = False
                atr_low = False

            # Sentiment (e.g., Fear & Greed Index)
            try:
                fear_greed_score = center_features.get('fear_greed_index', 50)  # Assuming this feature exists
                extreme_fear = fear_greed_score < 20
                extreme_greed = fear_greed_score > 80
            except KeyError:
                logger.debug("Fear & Greed Index not found in cluster features. Skipping sentiment heuristic.")
                extreme_fear = False
                extreme_greed = False

            # Apply classification logic
            if extreme_fear:
                label = "extreme_fear"
            elif extreme_greed and sma_short_gt_long and not rolling_std_high:
                label = "bull"
            elif sma_short_gt_long and not atr_high:
                label = "bull"
            elif not sma_short_gt_long and rsi_low and not atr_high:
                label = "bear"
            elif rolling_std_high or atr_high:
                label = "volatile"
            elif atr_low and not sma_short_gt_long and not rsi_high and not rsi_low:
                label = "sideways"
            else:
                label = "neutral"  # Default fallback

            self.regime_map[cluster_id] = label
            logger.info(f"Assigned label '{label}' to cluster {cluster_id}")

    def predict_regime(self, current_market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Predicts the current market regime based on live market data.
        `current_market_data` is a dict of symbol -> DataFrame containing recent OHLCV data.
        """
        if not self.model_trained or not self.model or not self.scaler or not self.imputer:
            logger.warning("Market regime model not trained or loaded. Cannot predict regime.")
            return {"status": "error", "label": "unknown", "regime": -1, "reason": "Model not trained"}

        features_for_prediction_list = []

        max_lookback = max(
            self.settings.get('ml.feature_extraction.rsi_period', 14),
            self.settings.get('ml.feature_extraction.ma_long', 50),
            self.settings.get('ml.feature_extraction.bollinger_window', 20),
            self.settings.get('ml.feature_extraction.atr_period', 14)
        )
        required_candles_for_features = max_lookback + 1  # Need at least one more candle than lookback for final feature point

        for symbol in self.core_symbols:
            if symbol not in current_market_data or current_market_data[symbol].empty:
                logger.warning(f"Missing live data for {symbol}. Cannot predict regime accurately.")
                # Attempt to use a fallback or skip this symbol if data is critical
                return {"status": "error", "label": "unknown", "regime": -1, "reason": f"Missing data for {symbol}"}

            # Ensure index is datetime for consistency
            current_market_data[symbol].index = pd.to_datetime(current_market_data[symbol].index)

            # Get enough data for feature extraction.
            # We need at least `required_candles_for_features` to compute all indicators correctly.
            recent_data = current_market_data[symbol].iloc[-required_candles_for_features:].copy()

            if len(recent_data) < required_candles_for_features:
                logger.warning(
                    f"Not enough recent data points for {symbol} ({len(recent_data)} available, {required_candles_for_features} required). Cannot predict regime accurately.")
                return {"status": "error", "label": "unknown", "regime": -1,
                        "reason": f"Insufficient recent data for {symbol}"}

            features_df = self.feature_extractor.calculate_technical_indicators(recent_data)

            # Add volatility measures explicitly from OHLC
            if 'close' in recent_data.columns:
                features_df[f'{symbol.replace("/", "_")}_rolling_std'] = recent_data['close'].rolling(window=20).std()
            if 'atr' not in features_df.columns and 'high' in recent_data.columns and 'low' in recent_data.columns:
                # If ATR not added by feature extractor, add a simple proxy or the correct ATR
                features_df[f'{symbol.replace("/", "_")}_atr'] = recent_data['high'] - recent_data['low']

            # Add sentiment features (from placeholder MarketSentimentAnalyzer) for the latest date
            latest_date_str = recent_data.index.max().strftime('%Y-%m-%d')
            sentiment_data = self.sentiment_analyzer.get_sentiment_for_date(latest_date_str)
            if sentiment_data:
                # Add sentiment as a column, ensuring it aligns (e.g., constant for the latest candle)
                features_df['fear_greed_index'] = sentiment_data.get('fear_greed_index', np.nan)

            # Take only the last row (most recent features)
            # Drop columns where the last value is NaN and they were used in training
            latest_features = features_df.iloc[-1].to_frame().T
            latest_features = latest_features.add_prefix(f"{symbol.replace('/', '_')}_")
            features_for_prediction_list.append(latest_features)

        if not features_for_prediction_list:
            return {"status": "error", "label": "unknown", "regime": -1, "reason": "No valid features for prediction."}

        combined_live_features = pd.concat(features_for_prediction_list, axis=1)

        # Ensure columns match the training columns. This is crucial.
        # We need the `feature_names_in_` attribute from the fitted imputer/scaler.
        # If not saved with imputer/scaler, we rely on them being consistent, which is risky.
        # Assuming `self.imputer.feature_names_in_` holds the order from training
        expected_columns_from_training = self.imputer.feature_names_in_ if self.imputer else []

        # Create a DataFrame with all expected columns, filling missing with NaN
        aligned_features = pd.DataFrame(columns=expected_columns_from_training)
        # Ensure the index matches `combined_live_features` if it has one (should be just one row)
        aligned_features = pd.DataFrame(index=combined_live_features.index, columns=expected_columns_from_training)

        for col in expected_columns_from_training:
            if col in combined_live_features.columns:
                aligned_features[col] = combined_live_features[col]
            else:
                aligned_features[col] = np.nan  # Fill with NaN for missing columns

        # Impute missing values using the *trained* imputer
        imputed_live_features = self.imputer.transform(aligned_features)

        # Scale live features using the *trained* scaler
        scaled_live_features = self.scaler.transform(imputed_live_features)

        predicted_label = self.model.predict(scaled_live_features)[0]
        regime_label = self.regime_map.get(predicted_label, "unknown")

        self.last_regime_info = {
            "status": "success",
            "label": regime_label,
            "regime": int(predicted_label),  # Ensure int for JSON serialization
            "timestamp": datetime.now().isoformat()
        }
        logger.info(f"Detected market regime: {regime_label} (Cluster ID: {predicted_label})")
        return self.last_regime_info

    def get_last_regime(self) -> Dict[str, Any]:
        """Returns the last detected market regime."""
        return self.last_regime_info