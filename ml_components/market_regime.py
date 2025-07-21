# ml_components/market_regime.py
import logging
import os
import json
import pandas as pd
import numpy as np
import pickle
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from enum import Enum

# ML Libraries with fallbacks
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.impute import SimpleImputer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Technical Analysis with fallback
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

# Core imports with fallbacks
try:
    from data_sources.data_manager import DataManager
    HAS_DATA_MANAGER = True
except ImportError:
    HAS_DATA_MANAGER = False

try:
    from ml_components.feature_extraction import FeatureExtractor
    HAS_FEATURE_EXTRACTOR = True
except ImportError:
    HAS_FEATURE_EXTRACTOR = False

try:
    from ml_components.market_sentiment import MarketSentimentAnalyzer
    HAS_SENTIMENT = True
except ImportError:
    HAS_SENTIMENT = False

from config.settings import Settings

logger = logging.getLogger(__name__)


class MarketPhase(Enum):
    """Market Phase Enumeration"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    EXTREME_FEAR = "extreme_fear"
    UNKNOWN = "unknown"


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
        self.min_data_points_required = min_data_points_required

        # Model components
        self.model: Optional[MiniBatchKMeans] = None
        self.scaler: Optional[StandardScaler] = None
        self.regime_map: Dict[int, str] = {}  # Maps cluster label to human-readable regime
        self.model_trained = False
        self.imputer: Optional[SimpleImputer] = None

        # Initialize components with fallbacks
        if HAS_FEATURE_EXTRACTOR:
            self.feature_extractor = FeatureExtractor(self.settings.get('ml.feature_extraction', {}))
        else:
            self.feature_extractor = None
            logger.warning("FeatureExtractor not available, using built-in feature calculation")

        if HAS_SENTIMENT:
            self.sentiment_analyzer = MarketSentimentAnalyzer(self.settings)
        else:
            self.sentiment_analyzer = None
            logger.warning("MarketSentimentAnalyzer not available")

        self.last_regime_info: Dict[str, Any] = {"status": "not_available", "label": "unknown", "regime": -1}

        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info(f"MarketRegimeDetector initialized with {len(self.core_symbols)} core symbols.")
        self.load_model()  # Attempt to load model on initialization

    def detect_market_phase(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> str:
        """
        Main function to detect market phase - simplified interface
        
        Args:
            data: Single DataFrame or dict of symbol -> DataFrame
            
        Returns:
            Market phase as string ("bull", "bear", "sideways", "volatile", "extreme_fear")
        """
        try:
            # Convert to dict format if single DataFrame
            if isinstance(data, pd.DataFrame):
                data = {'PRIMARY': data}
            
            result = self.predict_regime(data)
            return result.get('label', 'unknown')
            
        except Exception as e:
            logger.error(f"Error detecting market phase: {e}")
            return 'unknown'

    def load_model(self) -> bool:
        """Loads the pre-trained K-Means model, scaler, and regime map."""
        try:
            if not HAS_SKLEARN:
                logger.warning("scikit-learn not available, model training/loading disabled")
                return False
                
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path) and os.path.exists(self.regime_map_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                with open(self.regime_map_path, 'r') as f:
                    self.regime_map = json.load(f)
                self.model_trained = True

                logger.info("Market regime model, scaler and map loaded successfully.")
                return True
            else:
                logger.warning("No existing market regime model found.")
                return False
        except Exception as e:
            logger.error(f"Error loading market regime model: {e}")
            self.model_trained = False
            return False

    def save_model(self):
        """Saves the trained K-Means model, scaler, and regime map."""
        try:
            if not HAS_SKLEARN:
                logger.warning("scikit-learn not available, cannot save model")
                return
                
            os.makedirs(self.models_dir, exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            with open(self.regime_map_path, 'w') as f:
                json.dump(self.regime_map, f, indent=4)
            logger.info("Market regime model, scaler and map saved successfully.")
        except Exception as e:
            logger.error(f"Error saving market regime model: {e}")

    def train_model(self, data_manager: DataManager, timeframe: str = '1d',
                    start_date: str = '2022-01-01', end_date: str = '2023-12-31'):
        """Enhanced training with better error handling"""
        try:
            if not HAS_SKLEARN:
                logger.error("scikit-learn not available, cannot train model")
                return False
                
            if not HAS_DATA_MANAGER:
                logger.error("DataManager not available, cannot train model")
                return False
                
            logger.info("Starting enhanced market regime model training...")
            
            # Original training logic with enhancements
            combined_features_list = []
            
            # Calculate lookback needed for all features
            max_lookback = max(
                self.settings.get('ml.feature_extraction.rsi_period', 14),
                self.settings.get('ml.feature_extraction.ma_long', 50),
                self.settings.get('ml.feature_extraction.bollinger_window', 20),
                self.settings.get('ml.feature_extraction.atr_period', 14)
            )
            required_candles_for_features = max_lookback + self.min_data_points_required

            for symbol in self.core_symbols:
                logger.info(f"Processing training data for {symbol}...")
                df = data_manager.get_historical_data(symbol, timeframe, start_date, end_date)

                if df.empty or len(df) < required_candles_for_features:
                    logger.warning(f"Insufficient data for {symbol}. Skipping.")
                    continue

                # Calculate features
                if self.feature_extractor:
                    features_df = self.feature_extractor.calculate_technical_indicators(df.copy())
                else:
                    features_df = self._calculate_features_fallback(df)

                # Add enhanced features
                features_df = self._add_enhanced_features(features_df, df, symbol)

                # Select numeric columns and handle NaNs
                numeric_cols = features_df.select_dtypes(include=np.number).columns
                features_for_clustering = features_df[numeric_cols].dropna(axis=1, how='all').dropna()

                if features_for_clustering.empty:
                    logger.warning(f"No valid features for {symbol}. Skipping.")
                    continue

                # Add symbol prefix and append
                features_for_clustering = features_for_clustering.add_prefix(f"{symbol.replace('/', '_')}_")
                combined_features_list.append(features_for_clustering)

            if not combined_features_list:
                logger.error("No sufficient training data available.")
                return False

            # Combine and train
            full_features_df = pd.concat(combined_features_list, axis=1, join='inner')
            full_features_df = full_features_df.dropna(axis=1, how='all')

            if len(full_features_df) < self.n_regimes * 2:
                logger.warning("Adjusting number of regimes due to insufficient data")
                self.n_regimes = max(1, len(full_features_df) // 2)

            # Preprocessing
            self.imputer = SimpleImputer(strategy='mean')
            imputed_features = self.imputer.fit_transform(full_features_df)
            
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(imputed_features)

            # Training
            self.model = MiniBatchKMeans(n_clusters=self.n_regimes, random_state=42, n_init='auto')
            self.model.fit(scaled_features)
            self.model_trained = True

            # Enhanced regime mapping
            self._define_regime_labels_enhanced(pd.DataFrame(imputed_features, columns=full_features_df.columns))
            
            self.save_model()
            logger.info("Enhanced market regime model trained successfully.")
            return True

        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False

    def _define_regime_labels_enhanced(self, features_df: pd.DataFrame):
        """Enhanced regime labeling with better heuristics"""
        try:
            if not self.model or not self.model_trained:
                logger.error("Model not trained, cannot define regime labels.")
                return

            cluster_centers = self.scaler.inverse_transform(self.model.cluster_centers_)
            cluster_centers_df = pd.DataFrame(cluster_centers, columns=features_df.columns)

            self.regime_map = {}
            for cluster_id in range(self.n_regimes):
                center_features = cluster_centers_df.iloc[cluster_id]
                
                # Enhanced classification logic
                phase = self._classify_cluster_enhanced(center_features)
                self.regime_map[cluster_id] = phase
                
                logger.info(f"Cluster {cluster_id} mapped to '{phase}'")

        except Exception as e:
            logger.error(f"Error defining regime labels: {e}")
            # Set default mapping
            for i in range(self.n_regimes):
                self.regime_map[i] = "unknown"

    def _classify_cluster_enhanced(self, center_features: pd.Series) -> str:
        """Enhanced cluster classification"""
        try:
            # Find features from primary symbol (usually BTC)
            primary_features = {}
            for symbol in self.core_symbols:
                prefix = symbol.replace('/', '_') + '_'
                symbol_features = {k.replace(prefix, ''): v for k, v in center_features.items() 
                                 if k.startswith(prefix)}
                if symbol_features:
                    primary_features = symbol_features
                    break

            # Extract key indicators
            sma_20 = primary_features.get('sma_20', 0)
            sma_50 = primary_features.get('sma_50', 0) 
            rsi = primary_features.get('rsi', 50)
            volatility = primary_features.get('rolling_std', 0)
            atr = primary_features.get('atr', 0)

            # Classification logic
            high_volatility = volatility > 0.04 or atr > 0.03
            strong_trend_up = sma_20 > sma_50 and rsi > 60
            strong_trend_down = sma_20 < sma_50 and rsi < 40
            extreme_conditions = volatility > 0.06 or rsi < 20 or rsi > 80

            if extreme_conditions and strong_trend_down:
                return "extreme_fear"
            elif high_volatility:
                return "volatile"
            elif strong_trend_up:
                return "bull"
            elif strong_trend_down:
                return "bear"
            else:
                return "sideways"

        except Exception as e:
            logger.error(f"Error classifying cluster: {e}")
            return "unknown"

    def predict_regime(self, current_market_data: Union[Dict[str, pd.DataFrame], pd.DataFrame]) -> Dict[str, Any]:
        """
        Enhanced predict_regime with better error handling and fallbacks
        """
        try:
            # Convert single DataFrame to dict format
            if isinstance(current_market_data, pd.DataFrame):
                current_market_data = {'PRIMARY': current_market_data}

            # If model is trained, use ML prediction
            if self.model_trained and self.model and HAS_SKLEARN:
                return self._ml_based_prediction(current_market_data)
            else:
                # Use rule-based fallback
                return self._rule_based_prediction(current_market_data)

        except Exception as e:
            logger.error(f"Error predicting regime: {e}")
            return {
                "status": "error",
                "label": "unknown",
                "regime": -1,
                "reason": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _ml_based_prediction(self, current_market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """ML-based prediction using trained model"""
        try:
            if not self.model_trained or not self.model or not self.scaler or not self.imputer:
                logger.warning("Market regime model not trained or loaded. Cannot predict regime.")
                return self._rule_based_prediction(current_market_data)

            features_for_prediction_list = []

            # Calculate required lookback for features
            max_lookback = max(
                self.settings.get('ml.feature_extraction.rsi_period', 14),
                self.settings.get('ml.feature_extraction.ma_long', 50),
                self.settings.get('ml.feature_extraction.bollinger_window', 20),
                self.settings.get('ml.feature_extraction.atr_period', 14)
            )
            required_candles_for_features = max_lookback + 1

            for symbol in self.core_symbols:
                if symbol not in current_market_data or current_market_data[symbol].empty:
                    logger.warning(f"Missing live data for {symbol}")
                    continue

                recent_data = current_market_data[symbol].iloc[-required_candles_for_features:].copy()

                if len(recent_data) < required_candles_for_features:
                    logger.warning(f"Not enough recent data for {symbol}")
                    continue

                # Calculate features
                if self.feature_extractor:
                    features_df = self.feature_extractor.calculate_technical_indicators(recent_data)
                else:
                    features_df = self._calculate_features_fallback(recent_data)

                # Add additional features
                features_df = self._add_enhanced_features(features_df, recent_data, symbol)

                # Take the last row (most recent features)
                latest_features = features_df.iloc[-1].to_frame().T
                latest_features = latest_features.add_prefix(f"{symbol.replace('/', '_')}_")
                features_for_prediction_list.append(latest_features)

            if not features_for_prediction_list:
                return self._rule_based_prediction(current_market_data)

            # Combine features
            combined_live_features = pd.concat(features_for_prediction_list, axis=1)

            # Process with model pipeline
            if self.imputer:
                # Align with training features
                expected_columns = self.imputer.feature_names_in_ if hasattr(self.imputer, 'feature_names_in_') else []
                aligned_features = self._align_features(combined_live_features, expected_columns)
                
                # Impute and scale
                imputed_features = self.imputer.transform(aligned_features)
                scaled_features = self.scaler.transform(imputed_features)
                
                # Predict
                predicted_label = self.model.predict(scaled_features)[0]
                regime_label = self.regime_map.get(predicted_label, "unknown")
                
                # Calculate confidence (simplified)
                distances = self.model.transform(scaled_features)[0]
                confidence = 1.0 / (1.0 + distances[predicted_label])
                
                self.last_regime_info = {
                    "status": "success",
                    "label": regime_label,
                    "regime": int(predicted_label),
                    "confidence": float(confidence),
                    "method": "ml_clustering",
                    "timestamp": datetime.now().isoformat()
                }
                
                logger.info(f"ML prediction: {regime_label} (confidence: {confidence:.2f})")
                return self.last_regime_info
            else:
                return self._rule_based_prediction(current_market_data)

        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            return self._rule_based_prediction(current_market_data)

    def _rule_based_prediction(self, current_market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Enhanced rule-based prediction as fallback"""
        try:
            # Use first available symbol
            primary_symbol = None
            primary_data = None
            
            for symbol, data in current_market_data.items():
                if not data.empty and len(data) >= 20:
                    primary_symbol = symbol
                    primary_data = data
                    break
            
            if primary_data is None:
                return {
                    "status": "error",
                    "label": "unknown", 
                    "regime": -1,
                    "reason": "No sufficient data",
                    "timestamp": datetime.now().isoformat()
                }

            # Calculate technical indicators
            close = primary_data['close'].astype(float)
            high = primary_data['high'].astype(float) if 'high' in primary_data else close
            low = primary_data['low'].astype(float) if 'low' in primary_data else close

            # Technical analysis
            current_price = close.iloc[-1]
            
            # Moving averages
            sma_20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else current_price
            sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else current_price
            
            # RSI
            rsi_14 = self._calculate_rsi(close, 14).iloc[-1] if len(close) >= 15 else 50
            
            # ATR for volatility
            atr = self._calculate_atr(high, low, close, 14).iloc[-1] if len(close) >= 15 else 0
            atr_ratio = atr / current_price if current_price > 0 else 0
            
            # Price momentum
            momentum_5d = (current_price / close.iloc[-6] - 1) if len(close) >= 6 else 0
            momentum_20d = (current_price / close.iloc[-21] - 1) if len(close) >= 21 else 0
            
            # Volatility
            volatility = close.pct_change().rolling(20).std().iloc[-1] if len(close) >= 21 else 0

            # Classification logic
            phase = "sideways"
            confidence = 0.5
            
            # High volatility conditions
            if volatility > 0.05 or atr_ratio > 0.04:
                if momentum_20d < -0.15:  # Extreme fear
                    phase = "extreme_fear"
                    confidence = 0.8
                else:  # General high volatility
                    phase = "volatile"
                    confidence = 0.7
            
            # Trend conditions (low to medium volatility)
            elif volatility <= 0.05:
                # Bull market conditions
                if (current_price > sma_20 > sma_50 and 
                    momentum_5d > 0.02 and 
                    rsi_14 > 50):
                    phase = "bull"
                    confidence = 0.7
                
                # Bear market conditions
                elif (current_price < sma_20 < sma_50 and 
                      momentum_5d < -0.02 and 
                      rsi_14 < 50):
                    phase = "bear"
                    confidence = 0.7
                
                # Sideways conditions
                elif (abs(momentum_5d) < 0.01 and 
                      30 < rsi_14 < 70 and 
                      volatility < 0.02):
                    phase = "sideways"
                    confidence = 0.6

            # Enhanced confidence based on signal alignment
            signal_count = 0
            if phase == "bull":
                if current_price > sma_20: signal_count += 1
                if sma_20 > sma_50: signal_count += 1
                if rsi_14 > 50: signal_count += 1
                if momentum_5d > 0: signal_count += 1
                confidence = min(0.9, 0.4 + (signal_count * 0.1))
            elif phase == "bear":
                if current_price < sma_20: signal_count += 1
                if sma_20 < sma_50: signal_count += 1
                if rsi_14 < 50: signal_count += 1
                if momentum_5d < 0: signal_count += 1
                confidence = min(0.9, 0.4 + (signal_count * 0.1))

            result = {
                "status": "success",
                "label": phase,
                "regime": self._phase_to_regime_id(phase),
                "confidence": confidence,
                "method": "rule_based",
                "indicators": {
                    "current_price": current_price,
                    "sma_20": sma_20,
                    "sma_50": sma_50,
                    "rsi_14": rsi_14,
                    "volatility": volatility,
                    "atr_ratio": atr_ratio,
                    "momentum_5d": momentum_5d,
                    "momentum_20d": momentum_20d
                },
                "timestamp": datetime.now().isoformat()
            }

            self.last_regime_info = result
            logger.info(f"Rule-based prediction: {phase} (confidence: {confidence:.2f})")
            return result

        except Exception as e:
            logger.error(f"Error in rule-based prediction: {e}")
            return {
                "status": "error",
                "label": "unknown",
                "regime": -1,
                "reason": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _calculate_features_fallback(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fallback feature calculation when FeatureExtractor not available"""
        try:
            features = pd.DataFrame(index=data.index)
            close = data['close'].astype(float)
            
            # Basic moving averages
            features['sma_10'] = close.rolling(10).mean()
            features['sma_20'] = close.rolling(20).mean()
            features['sma_50'] = close.rolling(50).mean()
            
            # EMA
            features['ema_12'] = close.ewm(span=12).mean()
            features['ema_26'] = close.ewm(span=26).mean()
            
            # RSI
            features['rsi'] = self._calculate_rsi(close, 14)
            
            # MACD
            features['macd'] = features['ema_12'] - features['ema_26']
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            
            # Volatility
            features['volatility'] = close.pct_change().rolling(20).std()
            
            # ATR
            if 'high' in data.columns and 'low' in data.columns:
                features['atr'] = self._calculate_atr(data['high'], data['low'], close, 14)
            
            return features
            
        except Exception as e:
            logger.error(f"Error in fallback feature calculation: {e}")
            return pd.DataFrame(index=data.index)

    def _add_enhanced_features(self, features_df: pd.DataFrame, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add enhanced features specific to symbol"""
        try:
            close = data['close'].astype(float)
            
            # Add volatility measures
            if 'close' in data.columns:
                features_df[f'{symbol.replace("/", "_")}_rolling_std'] = close.rolling(window=20).std()
            
            # Add ATR if available
            if 'high' in data.columns and 'low' in data.columns:
                features_df[f'{symbol.replace("/", "_")}_atr'] = self._calculate_atr(
                    data['high'], data['low'], close, 14
                )
            
            # Add sentiment if available
            if self.sentiment_analyzer:
                try:
                    latest_date_str = data.index.max().strftime('%Y-%m-%d')
                    sentiment_data = self.sentiment_analyzer.get_sentiment_for_date(latest_date_str)
                    if sentiment_data:
                        features_df['fear_greed_index'] = sentiment_data.get('fear_greed_index', np.nan)
                except Exception as e:
                    logger.debug(f"Sentiment data not available: {e}")
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error adding enhanced features: {e}")
            return features_df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            if HAS_TALIB:
                rsi = talib.RSI(prices.values, timeperiod=period)
                return pd.Series(rsi, index=prices.index)
            else:
                # Simple RSI calculation
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return rsi
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series([50] * len(prices), index=prices.index)

    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            if HAS_TALIB:
                atr = talib.ATR(high.values, low.values, close.values, timeperiod=period)
                return pd.Series(atr, index=close.index)
            else:
                # Simple ATR calculation
                tr1 = high - low
                tr2 = abs(high - close.shift(1))
                tr3 = abs(low - close.shift(1))
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = true_range.rolling(window=period).mean()
                return atr
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return pd.Series([0.01] * len(close), index=close.index)

    def _align_features(self, features: pd.DataFrame, expected_columns: List[str]) -> pd.DataFrame:
        """Align features with expected training columns"""
        try:
            aligned = pd.DataFrame(index=features.index, columns=expected_columns)
            
            for col in expected_columns:
                if col in features.columns:
                    aligned[col] = features[col]
                else:
                    aligned[col] = 0.0  # Fill missing with 0
            
            return aligned.fillna(0.0)
            
        except Exception as e:
            logger.error(f"Error aligning features: {e}")
            return features

    def _phase_to_regime_id(self, phase: str) -> int:
        """Convert phase name to regime ID"""
        phase_map = {
            "bull": 0,
            "bear": 1,
            "sideways": 2,
            "volatile": 3,
            "extreme_fear": 4,
            "unknown": -1
        }
        return phase_map.get(phase, -1)

    def get_last_regime(self) -> Dict[str, Any]:
        """Returns the last detected market regime."""
        return self.last_regime_info


# Convenience function for simple usage
def detect_market_phase(data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                       settings: Optional[Settings] = None) -> str:
    """
    Simple function to detect market phase
    
    Args:
        data: Market data (DataFrame or dict of symbol -> DataFrame)
        settings: Optional settings object
        
    Returns:
        Market phase as string
    """
    try:
        # Create temporary detector
        detector = MarketRegimeDetector(
            settings=settings or Settings(),
            data_cache_dir="data/temp",
            models_dir="data/temp", 
            output_dir="data/temp",
            core_symbols=['BTC/USDT'],
            min_data_points_required=50
        )
        return detector.detect_market_phase(data)
    except Exception as e:
        logger.error(f"Error in detect_market_phase function: {e}")
        return "unknown"