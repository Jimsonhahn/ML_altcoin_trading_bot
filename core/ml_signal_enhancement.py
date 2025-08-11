#!/usr/bin/env python3
"""
ML-Enhanced Signal Prediction System
====================================

Machine Learning enhancement for high-risk trading signals:
- Feature engineering from multiple data sources
- Ensemble models for signal prediction
- Real-time model inference
- Continuous learning capabilities
- Model performance monitoring
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import json
import pickle
from pathlib import Path
import warnings

# ML imports with fallbacks
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split, cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available - ML features disabled")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError) as e:
    LIGHTGBM_AVAILABLE = False
    logging.warning(f"LightGBM not available - using sklearn only: {e}")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class MLFeatures:
    """Structured ML features for signal prediction"""
    # Technical features
    price_change_1h: float
    price_change_4h: float
    price_change_24h: float
    volume_ratio_1h: float
    volume_ratio_4h: float
    volatility_1h: float
    rsi_14: float
    macd_signal: float
    bb_position: float  # Bollinger Band position
    
    # Social sentiment features
    twitter_sentiment: float
    reddit_sentiment: float
    social_volume: float
    sentiment_momentum: float
    
    # Market structure features
    market_regime: int  # 0=bear, 1=neutral, 2=bull
    time_of_day: int   # Hour of day
    day_of_week: int   # Day of week
    
    # Cross-asset features
    btc_correlation: float
    market_wide_sentiment: float
    
    # Target variables (for training)
    signal_1h: Optional[int] = None  # 0=hold, 1=buy, 2=sell
    return_1h: Optional[float] = None
    return_4h: Optional[float] = None
    success_flag: Optional[bool] = None

@dataclass
class MLPrediction:
    """ML model prediction result"""
    symbol: str
    timestamp: datetime
    predicted_signal: int  # 0=hold, 1=buy, 2=sell
    confidence: float
    probability_distribution: List[float]  # [prob_hold, prob_buy, prob_sell]
    feature_importance: Dict[str, float]
    model_name: str
    expected_return: float
    risk_score: float

class FeatureEngineer:
    """
    Feature engineering for ML models
    
    Converts raw market and sentiment data into ML-ready features
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        logger.info("🔧 Feature Engineer initialized")
    
    def extract_features(self, market_data: pd.DataFrame, 
                        sentiment_data: Dict[str, Any] = None,
                        symbol: str = 'BTC/USDT') -> MLFeatures:
        """Extract features from market and sentiment data"""
        
        if len(market_data) < 50:  # Need minimum data
            raise ValueError("Insufficient market data for feature extraction")
        
        # Technical indicators
        features = self._calculate_technical_features(market_data)
        
        # Sentiment features
        sentiment_features = self._calculate_sentiment_features(sentiment_data or {})
        features.update(sentiment_features)
        
        # Market structure features
        structure_features = self._calculate_structure_features(market_data)
        features.update(structure_features)
        
        # Cross-asset features (simplified)
        cross_features = self._calculate_cross_asset_features(symbol)
        features.update(cross_features)
        
        return MLFeatures(**features)
    
    def _calculate_technical_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical analysis features"""
        
        close = data['close']
        volume = data['volume']
        high = data['high']
        low = data['low']
        
        # Price changes
        price_change_1h = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] if len(close) > 1 else 0.0
        price_change_4h = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] if len(close) > 4 else 0.0
        price_change_24h = (close.iloc[-1] - close.iloc[-25]) / close.iloc[-25] if len(close) > 24 else 0.0
        
        # Volume ratios
        current_volume = volume.iloc[-1]
        avg_volume_24h = volume.iloc[-24:].mean() if len(volume) > 24 else volume.mean()
        avg_volume_4h = volume.iloc[-4:].mean() if len(volume) > 4 else volume.mean()
        
        volume_ratio_1h = current_volume / max(avg_volume_24h, 1)
        volume_ratio_4h = volume.iloc[-4:].mean() / max(avg_volume_24h, 1) if len(volume) > 4 else 1.0
        
        # Volatility
        returns_1h = close.pct_change().iloc[-1:] if len(close) > 1 else pd.Series([0])
        volatility_1h = returns_1h.std() if len(returns_1h) > 0 else 0.0
        
        # RSI
        rsi_14 = self._calculate_rsi(close, 14)
        
        # MACD
        macd_signal = self._calculate_macd(close)
        
        # Bollinger Bands position
        bb_position = self._calculate_bollinger_position(close)
        
        return {
            'price_change_1h': price_change_1h,
            'price_change_4h': price_change_4h,
            'price_change_24h': price_change_24h,
            'volume_ratio_1h': volume_ratio_1h,
            'volume_ratio_4h': volume_ratio_4h,
            'volatility_1h': volatility_1h,
            'rsi_14': rsi_14,
            'macd_signal': macd_signal,
            'bb_position': bb_position
        }
    
    def _calculate_sentiment_features(self, sentiment_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate sentiment-based features"""
        
        twitter_sentiment = sentiment_data.get('twitter_sentiment', 0.0)
        reddit_sentiment = sentiment_data.get('reddit_sentiment', 0.0)
        social_volume = sentiment_data.get('social_volume', 0.0)
        sentiment_momentum = sentiment_data.get('sentiment_momentum', 0.0)
        
        return {
            'twitter_sentiment': twitter_sentiment,
            'reddit_sentiment': reddit_sentiment,
            'social_volume': social_volume,
            'sentiment_momentum': sentiment_momentum
        }
    
    def _calculate_structure_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate market structure features"""
        
        # Market regime detection (simplified)
        close = data['close']
        
        if len(close) >= 50:
            sma_50 = close.iloc[-50:].mean()
            current_price = close.iloc[-1]
            
            if current_price > sma_50 * 1.1:
                market_regime = 2  # Bull
            elif current_price < sma_50 * 0.9:
                market_regime = 0  # Bear
            else:
                market_regime = 1  # Neutral
        else:
            market_regime = 1
        
        # Time features
        current_time = datetime.now()
        time_of_day = current_time.hour
        day_of_week = current_time.weekday()
        
        return {
            'market_regime': float(market_regime),
            'time_of_day': float(time_of_day),
            'day_of_week': float(day_of_week)
        }
    
    def _calculate_cross_asset_features(self, symbol: str) -> Dict[str, float]:
        """Calculate cross-asset correlation features"""
        
        # Simplified - in production would calculate real correlations
        btc_correlation = 0.7 if 'BTC' not in symbol else 1.0
        market_wide_sentiment = 0.1  # Default neutral
        
        return {
            'btc_correlation': btc_correlation,
            'market_wide_sentiment': market_wide_sentiment
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> float:
        """Calculate MACD signal"""
        if len(prices) < slow + signal:
            return 0.0
        
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        
        return (macd.iloc[-1] - macd_signal.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0.0
    
    def _calculate_bollinger_position(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> float:
        """Calculate position within Bollinger Bands"""
        if len(prices) < period:
            return 0.5  # Middle position
        
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        current_price = prices.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        
        if pd.isna(current_upper) or pd.isna(current_lower):
            return 0.5
        
        # Position within bands (0 = lower band, 1 = upper band)
        if current_upper == current_lower:
            return 0.5
        
        position = (current_price - current_lower) / (current_upper - current_lower)
        return max(0.0, min(1.0, position))

class MLSignalPredictor:
    """
    Machine Learning signal predictor
    
    Uses ensemble of models to predict trading signals
    """
    
    def __init__(self, model_dir: str = "models/ml_signals"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.feature_engineer = FeatureEngineer()
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_trained = False
        
        # Performance tracking
        self.prediction_history = []
        self.model_performance = {}
        
        logger.info("🤖 ML Signal Predictor initialized")
    
    def initialize_models(self):
        """Initialize ensemble of ML models"""
        if not SKLEARN_AVAILABLE:
            logger.error("❌ Scikit-learn not available - cannot initialize models")
            return
        
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        }
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            self.models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
        
        logger.info(f"🔧 Initialized {len(self.models)} ML models")
    
    def train_models(self, training_data: List[MLFeatures]) -> Dict[str, float]:
        """Train all models on historical data"""
        
        if not SKLEARN_AVAILABLE:
            logger.error("❌ Cannot train models - scikit-learn not available")
            return {}
        
        if len(training_data) < 100:
            logger.warning("⚠️ Insufficient training data - need at least 100 samples")
            return {}
        
        logger.info(f"🎓 Training models on {len(training_data)} samples...")
        
        # Convert to feature matrix and labels
        X, y = self._prepare_training_data(training_data)
        
        if X.shape[0] == 0:
            logger.error("❌ No valid training samples")
            return {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        if self.scaler:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Train each model
        model_scores = {}
        
        for name, model in self.models.items():
            try:
                logger.info(f"   🔧 Training {name}...")
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                model_scores[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
                
                logger.info(f"   ✅ {name}: Accuracy={accuracy:.3f}, F1={f1:.3f}")
                
            except Exception as e:
                logger.error(f"   ❌ Failed to train {name}: {e}")
                continue
        
        self.model_performance = model_scores
        self.is_trained = True
        
        # Save models
        self._save_models()
        
        logger.info(f"🎉 Training completed for {len(model_scores)} models")
        return model_scores
    
    def predict_signal(self, market_data: pd.DataFrame, 
                      sentiment_data: Dict[str, Any] = None,
                      symbol: str = 'BTC/USDT') -> MLPrediction:
        """Predict trading signal using ensemble models"""
        
        if not self.is_trained or not self.models:
            # Return neutral prediction if models not trained
            return MLPrediction(
                symbol=symbol,
                timestamp=datetime.now(),
                predicted_signal=0,  # HOLD
                confidence=0.0,
                probability_distribution=[1.0, 0.0, 0.0],
                feature_importance={},
                model_name='untrained',
                expected_return=0.0,
                risk_score=1.0
            )
        
        try:
            # Extract features
            features = self.feature_engineer.extract_features(
                market_data, sentiment_data, symbol
            )
            
            # Convert to feature vector
            feature_vector = self._features_to_vector(features)
            
            if self.scaler:
                feature_vector = self.scaler.transform([feature_vector])
            else:
                feature_vector = [feature_vector]
            
            # Get predictions from all models
            predictions = {}
            probabilities = {}
            
            for name, model in self.models.items():
                try:
                    pred = model.predict(feature_vector)[0]
                    prob = model.predict_proba(feature_vector)[0]
                    
                    predictions[name] = pred
                    probabilities[name] = prob
                    
                except Exception as e:
                    logger.warning(f"Model {name} prediction failed: {e}")
                    continue
            
            if not predictions:
                # Fallback if all models fail
                return MLPrediction(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    predicted_signal=0,
                    confidence=0.0,
                    probability_distribution=[1.0, 0.0, 0.0],
                    feature_importance={},
                    model_name='failed',
                    expected_return=0.0,
                    risk_score=1.0
                )
            
            # Ensemble prediction (weighted voting)
            ensemble_probs = self._calculate_ensemble_prediction(probabilities)
            final_prediction = np.argmax(ensemble_probs)
            confidence = max(ensemble_probs)
            
            # Calculate feature importance (from best performing model)
            feature_importance = self._get_feature_importance(features)
            
            # Estimate expected return and risk
            expected_return = self._estimate_expected_return(features, final_prediction)
            risk_score = self._calculate_risk_score(features)
            
            prediction = MLPrediction(
                symbol=symbol,
                timestamp=datetime.now(),
                predicted_signal=final_prediction,
                confidence=confidence,
                probability_distribution=ensemble_probs.tolist(),
                feature_importance=feature_importance,
                model_name='ensemble',
                expected_return=expected_return,
                risk_score=risk_score
            )
            
            # Store for performance tracking
            self.prediction_history.append(prediction)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            return MLPrediction(
                symbol=symbol,
                timestamp=datetime.now(),
                predicted_signal=0,
                confidence=0.0,
                probability_distribution=[1.0, 0.0, 0.0],
                feature_importance={},
                model_name='error',
                expected_return=0.0,
                risk_score=1.0
            )
    
    def _prepare_training_data(self, training_data: List[MLFeatures]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert training data to feature matrix and labels"""
        
        features_list = []
        labels_list = []
        
        for sample in training_data:
            if sample.signal_1h is not None:  # Valid label
                feature_vector = self._features_to_vector(sample)
                features_list.append(feature_vector)
                labels_list.append(sample.signal_1h)
        
        if not features_list:
            return np.array([]), np.array([])
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        return X, y
    
    def _features_to_vector(self, features: MLFeatures) -> List[float]:
        """Convert MLFeatures to feature vector"""
        
        return [
            features.price_change_1h,
            features.price_change_4h, 
            features.price_change_24h,
            features.volume_ratio_1h,
            features.volume_ratio_4h,
            features.volatility_1h,
            features.rsi_14,
            features.macd_signal,
            features.bb_position,
            features.twitter_sentiment,
            features.reddit_sentiment,
            features.social_volume,
            features.sentiment_momentum,
            features.market_regime,
            features.time_of_day,
            features.day_of_week,
            features.btc_correlation,
            features.market_wide_sentiment
        ]
    
    def _calculate_ensemble_prediction(self, probabilities: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate weighted ensemble prediction"""
        
        # Weight models by their performance
        weights = {}
        for name in probabilities.keys():
            if name in self.model_performance:
                weights[name] = self.model_performance[name].get('f1', 0.5)
            else:
                weights[name] = 0.5  # Default weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        else:
            weights = {k: 1.0/len(probabilities) for k in probabilities.keys()}
        
        # Weighted average
        ensemble_probs = np.zeros(3)  # [hold, buy, sell]
        
        for name, probs in probabilities.items():
            weight = weights.get(name, 0.0)
            ensemble_probs += weight * probs
        
        return ensemble_probs
    
    def _get_feature_importance(self, features: MLFeatures) -> Dict[str, float]:
        """Get feature importance from best model"""
        
        # Use random forest for feature importance if available
        if 'random_forest' in self.models:
            try:
                rf_model = self.models['random_forest']
                if hasattr(rf_model, 'feature_importances_'):
                    feature_names = [
                        'price_change_1h', 'price_change_4h', 'price_change_24h',
                        'volume_ratio_1h', 'volume_ratio_4h', 'volatility_1h',
                        'rsi_14', 'macd_signal', 'bb_position',
                        'twitter_sentiment', 'reddit_sentiment', 'social_volume', 'sentiment_momentum',
                        'market_regime', 'time_of_day', 'day_of_week',
                        'btc_correlation', 'market_wide_sentiment'
                    ]
                    
                    importance_dict = dict(zip(feature_names, rf_model.feature_importances_))
                    return importance_dict
            except Exception as e:
                logger.warning(f"Could not get feature importance: {e}")
        
        return {}
    
    def _estimate_expected_return(self, features: MLFeatures, prediction: int) -> float:
        """Estimate expected return based on features and prediction"""
        
        # Simplified return estimation
        base_return = 0.0
        
        if prediction == 1:  # BUY
            # Positive factors
            base_return += features.twitter_sentiment * 0.02
            base_return += features.reddit_sentiment * 0.015
            base_return += max(0, features.price_change_24h) * 0.5
            base_return += (features.rsi_14 - 50) / 500  # RSI momentum
            
        elif prediction == 2:  # SELL
            # Negative factors
            base_return -= abs(features.twitter_sentiment) * 0.02
            base_return -= abs(features.reddit_sentiment) * 0.015
            base_return += min(0, features.price_change_24h) * 0.5
            
        return max(-0.1, min(0.1, base_return))  # Clamp to ±10%
    
    def _calculate_risk_score(self, features: MLFeatures) -> float:
        """Calculate risk score (0=low risk, 1=high risk)"""
        
        risk_score = 0.5  # Base risk
        
        # Volatility increases risk
        risk_score += features.volatility_1h * 2
        
        # Market regime affects risk
        if features.market_regime == 0:  # Bear market
            risk_score += 0.2
        elif features.market_regime == 2:  # Bull market
            risk_score -= 0.1
        
        # Time factors
        if features.time_of_day < 6 or features.time_of_day > 22:  # After hours
            risk_score += 0.1
        
        return max(0.0, min(1.0, risk_score))
    
    def _save_models(self):
        """Save trained models to disk"""
        
        try:
            for name, model in self.models.items():
                model_path = self.model_dir / f"{name}_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            # Save scaler
            if self.scaler:
                scaler_path = self.model_dir / "scaler.pkl"
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
            
            # Save performance metrics
            perf_path = self.model_dir / "model_performance.json"
            with open(perf_path, 'w') as f:
                json.dump(self.model_performance, f, indent=2)
            
            logger.info(f"💾 Models saved to {self.model_dir}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self) -> bool:
        """Load trained models from disk"""
        
        try:
            # Load models
            for name in ['random_forest', 'gradient_boost', 'logistic_regression', 'lightgbm']:
                model_path = self.model_dir / f"{name}_model.pkl"
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        self.models[name] = pickle.load(f)
            
            # Load scaler
            scaler_path = self.model_dir / "scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            # Load performance metrics
            perf_path = self.model_dir / "model_performance.json"
            if perf_path.exists():
                with open(perf_path, 'r') as f:
                    self.model_performance = json.load(f)
            
            if self.models:
                self.is_trained = True
                logger.info(f"✅ Loaded {len(self.models)} models from {self.model_dir}")
                return True
            else:
                logger.warning("⚠️ No models found to load")
                return False
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of model performance and status"""
        
        return {
            'is_trained': self.is_trained,
            'num_models': len(self.models),
            'model_names': list(self.models.keys()),
            'performance_metrics': self.model_performance,
            'predictions_made': len(self.prediction_history),
            'last_prediction': self.prediction_history[-1].timestamp.isoformat() if self.prediction_history else None
        }

# Utility functions
def create_ml_predictor(model_dir: str = "models/ml_signals") -> MLSignalPredictor:
    """Create ML signal predictor"""
    predictor = MLSignalPredictor(model_dir)
    predictor.initialize_models()
    
    # Try to load existing models
    if not predictor.load_models():
        logger.info("🔧 No existing models found - will need training")
    
    return predictor

def generate_synthetic_training_data(num_samples: int = 1000) -> List[MLFeatures]:
    """Generate synthetic training data for testing"""
    
    logger.info(f"🎲 Generating {num_samples} synthetic training samples...")
    
    training_data = []
    np.random.seed(42)  # Reproducible
    
    for i in range(num_samples):
        # Generate synthetic features
        features = MLFeatures(
            price_change_1h=np.random.normal(0, 0.02),
            price_change_4h=np.random.normal(0, 0.05),
            price_change_24h=np.random.normal(0, 0.1),
            volume_ratio_1h=np.random.lognormal(0, 0.5),
            volume_ratio_4h=np.random.lognormal(0, 0.3),
            volatility_1h=np.random.exponential(0.01),
            rsi_14=np.random.uniform(20, 80),
            macd_signal=np.random.normal(0, 0.001),
            bb_position=np.random.uniform(0, 1),
            twitter_sentiment=np.random.normal(0, 0.3),
            reddit_sentiment=np.random.normal(0, 0.2),
            social_volume=np.random.exponential(1.0),
            sentiment_momentum=np.random.normal(0, 0.1),
            market_regime=np.random.choice([0, 1, 2]),
            time_of_day=np.random.randint(0, 24),
            day_of_week=np.random.randint(0, 7),
            btc_correlation=np.random.uniform(0.3, 0.9),
            market_wide_sentiment=np.random.normal(0, 0.2)
        )
        
        # Generate synthetic labels based on features
        signal_score = (
            features.twitter_sentiment * 2 +
            features.reddit_sentiment * 1.5 +
            features.price_change_4h * 10 +
            (features.rsi_14 - 50) / 20 +
            np.random.normal(0, 0.5)  # Add noise
        )
        
        if signal_score > 0.8:
            features.signal_1h = 1  # BUY
            features.return_1h = abs(np.random.normal(0.02, 0.01))
        elif signal_score < -0.8:
            features.signal_1h = 2  # SELL
            features.return_1h = -abs(np.random.normal(0.02, 0.01))
        else:
            features.signal_1h = 0  # HOLD
            features.return_1h = np.random.normal(0, 0.005)
        
        features.success_flag = features.return_1h > 0 if features.signal_1h != 0 else True
        
        training_data.append(features)
    
    logger.info(f"✅ Generated {len(training_data)} training samples")
    return training_data

if __name__ == "__main__":
    # Test ML system
    print("🤖 Testing ML Signal Enhancement...")
    
    if not SKLEARN_AVAILABLE:
        print("❌ Scikit-learn not available - cannot test ML features")
        exit(1)
    
    # Create predictor
    predictor = create_ml_predictor()
    
    # Generate synthetic training data
    training_data = generate_synthetic_training_data(1000)
    
    # Train models
    scores = predictor.train_models(training_data)
    print(f"📊 Training scores: {scores}")
    
    # Test prediction
    test_data = pd.DataFrame({
        'open': [45000, 45100, 45200],
        'high': [45200, 45300, 45400], 
        'low': [44900, 45000, 45100],
        'close': [45100, 45200, 45300],
        'volume': [1000000, 1100000, 1200000]
    })
    
    prediction = predictor.predict_signal(test_data)
    print(f"🎯 Test prediction: {prediction.predicted_signal} (confidence: {prediction.confidence:.2f})")
    
    print("🎉 ML system test completed!")