"""
Market Predictor - ML-basierte Marktphasen-Vorhersage
Verwendet LightGBM/XGBoost für präzise Marktphasen-Predictions mit technischen Indikatoren
"""

import logging
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import joblib
from pathlib import Path
import json

# Suppress warnings
warnings.filterwarnings('ignore')

# Try to import ML libraries with fallbacks
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.ensemble import RandomForestClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    logging.warning("TA-Lib not available, using pandas for technical indicators")


class MarketPredictor:
    """
    ML-basierte Marktphasen-Vorhersage mit LightGBM/XGBoost
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Model configuration
        self.model_type = self.config.get('model_type', 'lightgbm')  # 'lightgbm', 'xgboost', 'random_forest'
        self.model_path = Path(self.config.get('model_path', 'data/ml_models/market_predictor.pkl'))
        self.scaler_path = Path(self.config.get('scaler_path', 'data/ml_models/market_scaler.pkl'))
        self.label_encoder_path = Path(self.config.get('label_encoder_path', 'data/ml_models/label_encoder.pkl'))
        
        # Model parameters
        self.lookback_period = self.config.get('lookback_period', 48)  # 48 hours for 1h data
        self.prediction_horizon = self.config.get('prediction_horizon', 1)  # 1 hour ahead
        self.min_data_points = self.config.get('min_data_points', 1000)
        
        # Market phases
        self.market_phases = ['sideways', 'bull', 'bear', 'volatile', 'extreme_fear']
        
        # Models and preprocessors
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = []
        
        # Model status
        self.is_trained = False
        self.last_training_time = None
        self.model_performance = {}
        
        # Feature engineering parameters
        self.feature_config = {
            'price_features': True,
            'volume_features': True,
            'technical_indicators': True,
            'funding_rates': True,
            'order_book_features': True,
            'volatility_features': True,
            'momentum_features': True
        }
        
        self.logger.info(f"MarketPredictor initialized with {self.model_type} model")
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Erstellt umfangreiche Features für die Marktphasen-Vorhersage
        """
        try:
            df = data.copy()
            features = pd.DataFrame(index=df.index)
            
            # Basis-Features
            features['close'] = df['close']
            features['volume'] = df['volume']
            features['high'] = df['high']
            features['low'] = df['low']
            features['open'] = df['open']
            
            # Price-based features
            if self.feature_config['price_features']:
                features = self._add_price_features(features, df)
            
            # Volume-based features
            if self.feature_config['volume_features']:
                features = self._add_volume_features(features, df)
            
            # Technical indicators
            if self.feature_config['technical_indicators']:
                features = self._add_technical_indicators(features, df)
            
            # Volatility features
            if self.feature_config['volatility_features']:
                features = self._add_volatility_features(features, df)
            
            # Momentum features
            if self.feature_config['momentum_features']:
                features = self._add_momentum_features(features, df)
            
            # Funding rate features (if available)
            if self.feature_config['funding_rates'] and 'funding_rate' in df.columns:
                features = self._add_funding_rate_features(features, df)
            
            # Order book features (if available)
            if self.feature_config['order_book_features'] and 'bid_ask_spread' in df.columns:
                features = self._add_order_book_features(features, df)
            
            # Remove NaN values
            features = features.dropna()
            
            self.feature_names = features.columns.tolist()
            self.logger.info(f"Created {len(self.feature_names)} features")
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating features: {e}")
            return pd.DataFrame()
    
    def _add_price_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Fügt preis-basierte Features hinzu"""
        try:
            # Price returns
            features['returns_1h'] = df['close'].pct_change(1)
            features['returns_4h'] = df['close'].pct_change(4)
            features['returns_24h'] = df['close'].pct_change(24)
            features['returns_7d'] = df['close'].pct_change(168)  # 7 days * 24 hours
            
            # Price levels
            features['price_vs_sma20'] = df['close'] / df['close'].rolling(20).mean() - 1
            features['price_vs_sma50'] = df['close'] / df['close'].rolling(50).mean() - 1
            features['price_vs_ema12'] = df['close'] / df['close'].ewm(span=12).mean() - 1
            
            # High/Low features
            features['high_low_ratio'] = df['high'] / df['low'] - 1
            features['close_vs_high'] = df['close'] / df['high'] - 1
            features['close_vs_low'] = df['close'] / df['low'] - 1
            
            # Gap features
            features['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error adding price features: {e}")
            return features
    
    def _add_volume_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Fügt volumen-basierte Features hinzu"""
        try:
            # Volume indicators
            features['volume_sma'] = df['volume'].rolling(20).mean()
            features['volume_ratio'] = df['volume'] / features['volume_sma']
            features['volume_trend'] = df['volume'].pct_change(5)
            
            # Volume-Price relationship
            features['vwap'] = (df['volume'] * df['close']).rolling(20).sum() / df['volume'].rolling(20).sum()
            features['price_vs_vwap'] = df['close'] / features['vwap'] - 1
            
            # Volume momentum
            features['volume_momentum'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error adding volume features: {e}")
            return features
    
    def _add_technical_indicators(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Fügt technische Indikatoren hinzu"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            if HAS_TALIB:
                # RSI
                features['rsi_14'] = talib.RSI(close, timeperiod=14)
                features['rsi_7'] = talib.RSI(close, timeperiod=7)
                
                # MACD
                macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
                features['macd'] = macd
                features['macd_signal'] = macd_signal
                features['macd_histogram'] = macd_hist
                
                # Bollinger Bands
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
                features['bb_upper'] = bb_upper
                features['bb_lower'] = bb_lower
                features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
                features['bb_width'] = (bb_upper - bb_lower) / bb_middle
                
                # ADX
                features['adx'] = talib.ADX(high, low, close, timeperiod=14)
                
                # Stochastic
                slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
                features['stoch_k'] = slowk
                features['stoch_d'] = slowd
                
                # ATR
                features['atr'] = talib.ATR(high, low, close, timeperiod=14)
                features['atr_ratio'] = features['atr'] / close
                
                # CCI
                features['cci'] = talib.CCI(high, low, close, timeperiod=14)
                
                # Williams %R
                features['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)
                
                # MFI
                features['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)
                
            else:
                # Fallback implementations
                features['rsi_14'] = self._calculate_rsi(df['close'], 14)
                features['macd'] = self._calculate_macd(df['close'])
                features['bb_position'] = self._calculate_bb_position(df['close'])
                features['atr'] = self._calculate_atr(df)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {e}")
            return features
    
    def _add_volatility_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Fügt Volatilitäts-Features hinzu"""
        try:
            # Realized volatility
            features['vol_1h'] = df['close'].pct_change().rolling(24).std() * np.sqrt(24)
            features['vol_4h'] = df['close'].pct_change(4).rolling(6).std() * np.sqrt(6)
            features['vol_24h'] = df['close'].pct_change(24).rolling(7).std() * np.sqrt(7)
            
            # Volatility ratios
            features['vol_ratio_short'] = features['vol_1h'] / features['vol_24h']
            features['vol_ratio_medium'] = features['vol_4h'] / features['vol_24h']
            
            # Parkinson volatility (using high-low)
            features['parkinson_vol'] = np.sqrt(np.log(df['high'] / df['low']).rolling(24).var())
            
            # Garman-Klass volatility
            features['gk_vol'] = np.sqrt(
                0.5 * (np.log(df['high'] / df['low']))**2 -
                (2 * np.log(2) - 1) * (np.log(df['close'] / df['open']))**2
            ).rolling(24).mean()
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error adding volatility features: {e}")
            return features
    
    def _add_momentum_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Fügt Momentum-Features hinzu"""
        try:
            # Price momentum
            features['momentum_1h'] = df['close'] / df['close'].shift(1) - 1
            features['momentum_4h'] = df['close'] / df['close'].shift(4) - 1
            features['momentum_24h'] = df['close'] / df['close'].shift(24) - 1
            
            # Moving average momentum
            features['ma_momentum_short'] = df['close'].rolling(5).mean() / df['close'].rolling(20).mean() - 1
            features['ma_momentum_long'] = df['close'].rolling(20).mean() / df['close'].rolling(50).mean() - 1
            
            # ROC (Rate of Change)
            features['roc_12'] = (df['close'] - df['close'].shift(12)) / df['close'].shift(12)
            features['roc_24'] = (df['close'] - df['close'].shift(24)) / df['close'].shift(24)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error adding momentum features: {e}")
            return features
    
    def _add_funding_rate_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Fügt Funding Rate Features hinzu"""
        try:
            # Funding rate indicators
            features['funding_rate'] = df['funding_rate']
            features['funding_rate_ma'] = df['funding_rate'].rolling(24).mean()
            features['funding_rate_std'] = df['funding_rate'].rolling(24).std()
            features['funding_rate_zscore'] = (df['funding_rate'] - features['funding_rate_ma']) / features['funding_rate_std']
            
            # Funding rate momentum
            features['funding_rate_change'] = df['funding_rate'].diff()
            features['funding_rate_trend'] = df['funding_rate'].rolling(8).mean() - df['funding_rate'].rolling(24).mean()
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error adding funding rate features: {e}")
            return features
    
    def _add_order_book_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Fügt Order Book Features hinzu"""
        try:
            # Spread features
            features['bid_ask_spread'] = df['bid_ask_spread']
            features['spread_ma'] = df['bid_ask_spread'].rolling(24).mean()
            features['spread_ratio'] = df['bid_ask_spread'] / features['spread_ma']
            
            # Order book imbalance (if available)
            if 'order_book_imbalance' in df.columns:
                features['ob_imbalance'] = df['order_book_imbalance']
                features['ob_imbalance_ma'] = df['order_book_imbalance'].rolling(12).mean()
                features['ob_imbalance_trend'] = df['order_book_imbalance'].rolling(6).mean() - df['order_book_imbalance'].rolling(24).mean()
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error adding order book features: {e}")
            return features
    
    def create_labels(self, data: pd.DataFrame) -> pd.Series:
        """
        Erstellt Labels für die Marktphasen-Klassifikation
        """
        try:
            labels = []
            
            for i in range(len(data)):
                # Berechne zukünftige Returns
                if i + self.prediction_horizon < len(data):
                    future_return = (data['close'].iloc[i + self.prediction_horizon] / data['close'].iloc[i]) - 1
                    current_vol = data['close'].iloc[max(0, i-23):i+1].pct_change().std() if i >= 23 else 0
                    
                    # Klassifiziere Marktphase
                    if abs(future_return) < 0.005 and current_vol < 0.02:  # Geringe Bewegung und Volatilität
                        label = 'sideways'
                    elif future_return > 0.02:  # Starke positive Bewegung
                        label = 'bull'
                    elif future_return < -0.02:  # Starke negative Bewegung
                        label = 'bear'
                    elif current_vol > 0.05:  # Hohe Volatilität
                        label = 'volatile'
                    elif future_return < -0.05:  # Extreme negative Bewegung
                        label = 'extreme_fear'
                    else:
                        label = 'sideways'  # Default
                    
                    labels.append(label)
                else:
                    labels.append('sideways')  # Für die letzten Datenpunkte
            
            return pd.Series(labels, index=data.index)
            
        except Exception as e:
            self.logger.error(f"Error creating labels: {e}")
            return pd.Series(['sideways'] * len(data), index=data.index)
    
    def train_model(self, data: pd.DataFrame, retrain: bool = False) -> bool:
        """
        Trainiert das ML-Modell für Marktphasen-Vorhersage
        """
        try:
            self.logger.info("Starting model training...")
            
            # Prüfe ob bereits trainiert
            if self.is_trained and not retrain:
                self.logger.info("Model already trained. Use retrain=True to force retraining.")
                return True
            
            # Prüfe Mindestdatenmenge
            if len(data) < self.min_data_points:
                self.logger.warning(f"Insufficient data: {len(data)} < {self.min_data_points}")
                return False
            
            # Erstelle Features
            features = self.create_features(data)
            if features.empty:
                self.logger.error("Failed to create features")
                return False
            
            # Erstelle Labels
            labels = self.create_labels(data)
            
            # Align features and labels
            common_index = features.index.intersection(labels.index)
            features = features.loc[common_index]
            labels = labels.loc[common_index]
            
            # Remove NaN values
            mask = ~(features.isna().any(axis=1) | labels.isna())
            features = features[mask]
            labels = labels[mask]
            
            if len(features) < self.min_data_points:
                self.logger.warning(f"Insufficient clean data: {len(features)} < {self.min_data_points}")
                return False
            
            # Prepare data
            X = features.values
            y = labels.values
            
            # Initialize label encoder
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Initialize scaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train-test split with time series split
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Choose and train model
            if self.model_type == 'lightgbm' and HAS_LIGHTGBM:
                self.model = self._train_lightgbm(X_scaled, y_encoded, tscv)
            elif self.model_type == 'xgboost' and HAS_XGBOOST:
                self.model = self._train_xgboost(X_scaled, y_encoded, tscv)
            else:
                self.logger.info("Using Random Forest fallback")
                self.model = self._train_random_forest(X_scaled, y_encoded, tscv)
            
            if self.model is None:
                self.logger.error("Failed to train model")
                return False
            
            # Evaluate model
            self._evaluate_model(X_scaled, y_encoded)
            
            # Save model and preprocessors
            self._save_model()
            
            self.is_trained = True
            self.last_training_time = datetime.now()
            
            self.logger.info(f"Model training completed successfully using {self.model_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            return False
    
    def _train_lightgbm(self, X: np.ndarray, y: np.ndarray, tscv) -> Optional[lgb.LGBMClassifier]:
        """Trainiert LightGBM Modell"""
        try:
            model = lgb.LGBMClassifier(
                objective='multiclass',
                num_class=len(self.label_encoder.classes_),
                metric='multi_logloss',
                boosting_type='gbdt',
                num_leaves=31,
                learning_rate=0.05,
                feature_fraction=0.9,
                bagging_fraction=0.8,
                bagging_freq=5,
                verbose=-1,
                random_state=42
            )
            
            # Cross-validation training
            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model.fit(X_train, y_train)
                val_score = model.score(X_val, y_val)
                scores.append(val_score)
            
            self.model_performance['cv_scores'] = scores
            self.model_performance['mean_cv_score'] = np.mean(scores)
            
            # Final training on all data
            model.fit(X, y)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error training LightGBM: {e}")
            return None
    
    def _train_xgboost(self, X: np.ndarray, y: np.ndarray, tscv) -> Optional[xgb.XGBClassifier]:
        """Trainiert XGBoost Modell"""
        try:
            model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=len(self.label_encoder.classes_),
                max_depth=6,
                learning_rate=0.1,
                n_estimators=200,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )
            
            # Cross-validation training
            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model.fit(X_train, y_train)
                val_score = model.score(X_val, y_val)
                scores.append(val_score)
            
            self.model_performance['cv_scores'] = scores
            self.model_performance['mean_cv_score'] = np.mean(scores)
            
            # Final training on all data
            model.fit(X, y)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error training XGBoost: {e}")
            return None
    
    def _train_random_forest(self, X: np.ndarray, y: np.ndarray, tscv) -> Optional[RandomForestClassifier]:
        """Trainiert Random Forest Modell (Fallback)"""
        try:
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            # Cross-validation training
            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model.fit(X_train, y_train)
                val_score = model.score(X_val, y_val)
                scores.append(val_score)
            
            self.model_performance['cv_scores'] = scores
            self.model_performance['mean_cv_score'] = np.mean(scores)
            
            # Final training on all data
            model.fit(X, y)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error training Random Forest: {e}")
            return None
    
    def _evaluate_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Evaluiert das trainierte Modell"""
        try:
            # Predictions
            y_pred = self.model.predict(X)
            
            # Metrics
            accuracy = accuracy_score(y, y_pred)
            
            # Classification report
            class_names = self.label_encoder.classes_
            report = classification_report(y, y_pred, target_names=class_names, output_dict=True)
            
            # Store performance metrics
            self.model_performance.update({
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
                'feature_importance': self._get_feature_importance()
            })
            
            self.logger.info(f"Model accuracy: {accuracy:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Berechnet Feature Importance"""
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                return dict(zip(self.feature_names, importance))
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Macht Vorhersagen für neue Daten
        """
        try:
            if not self.is_trained:
                if not self.load_model():
                    return {'error': 'Model not trained and unable to load'}
            
            # Erstelle Features
            features = self.create_features(data)
            if features.empty:
                return {'error': 'Failed to create features'}
            
            # Nehme die letzten Datenpunkte
            X = features.tail(1).values
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            prediction = self.model.predict(X_scaled)[0]
            prediction_proba = self.model.predict_proba(X_scaled)[0]
            
            # Decode prediction
            predicted_phase = self.label_encoder.inverse_transform([prediction])[0]
            
            # Create confidence scores
            confidence_scores = dict(zip(self.label_encoder.classes_, prediction_proba))
            
            return {
                'predicted_phase': predicted_phase,
                'confidence': float(prediction_proba.max()),
                'confidence_scores': confidence_scores,
                'timestamp': datetime.now().isoformat(),
                'model_type': self.model_type
            }
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {e}")
            return {'error': str(e)}
    
    def predict_batch(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Macht Batch-Vorhersagen für mehrere Datenpunkte
        """
        try:
            if not self.is_trained:
                if not self.load_model():
                    return [{'error': 'Model not trained and unable to load'}]
            
            # Erstelle Features
            features = self.create_features(data)
            if features.empty:
                return [{'error': 'Failed to create features'}]
            
            # Scale features
            X_scaled = self.scaler.transform(features.values)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            predictions_proba = self.model.predict_proba(X_scaled)
            
            # Format results
            results = []
            for i, (pred, proba) in enumerate(zip(predictions, predictions_proba)):
                predicted_phase = self.label_encoder.inverse_transform([pred])[0]
                confidence_scores = dict(zip(self.label_encoder.classes_, proba))
                
                results.append({
                    'predicted_phase': predicted_phase,
                    'confidence': float(proba.max()),
                    'confidence_scores': confidence_scores,
                    'timestamp': features.index[i].isoformat() if hasattr(features.index[i], 'isoformat') else str(features.index[i]),
                    'model_type': self.model_type
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error making batch predictions: {e}")
            return [{'error': str(e)}]
    
    def _save_model(self) -> None:
        """Speichert das trainierte Modell"""
        try:
            # Erstelle Verzeichnis falls nicht vorhanden
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Speichere Modell
            joblib.dump(self.model, self.model_path)
            
            # Speichere Scaler
            joblib.dump(self.scaler, self.scaler_path)
            
            # Speichere Label Encoder
            joblib.dump(self.label_encoder, self.label_encoder_path)
            
            # Speichere Metadaten
            metadata = {
                'model_type': self.model_type,
                'feature_names': self.feature_names,
                'market_phases': self.market_phases,
                'training_time': self.last_training_time.isoformat() if self.last_training_time else None,
                'performance': self.model_performance,
                'config': self.config
            }
            
            metadata_path = self.model_path.parent / 'market_predictor_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Model saved to {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
    
    def load_model(self) -> bool:
        """Lädt ein gespeichertes Modell"""
        try:
            if not self.model_path.exists():
                self.logger.warning(f"Model file not found: {self.model_path}")
                return False
            
            # Lade Modell
            self.model = joblib.load(self.model_path)
            
            # Lade Scaler
            if self.scaler_path.exists():
                self.scaler = joblib.load(self.scaler_path)
            
            # Lade Label Encoder
            if self.label_encoder_path.exists():
                self.label_encoder = joblib.load(self.label_encoder_path)
            
            # Lade Metadaten
            metadata_path = self.model_path.parent / 'market_predictor_metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.feature_names = metadata.get('feature_names', [])
                self.model_performance = metadata.get('performance', {})
                self.last_training_time = datetime.fromisoformat(metadata['training_time']) if metadata.get('training_time') else None
            
            self.is_trained = True
            self.logger.info(f"Model loaded from {self.model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Gibt Informationen über das Modell zurück"""
        return {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'market_phases': self.market_phases,
            'performance': self.model_performance,
            'model_path': str(self.model_path)
        }
    
    # Fallback implementations für technische Indikatoren
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Vereinfachte RSI Berechnung"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Vereinfachte MACD Berechnung"""
        try:
            ema_12 = prices.ewm(span=12).mean()
            ema_26 = prices.ewm(span=26).mean()
            macd = ema_12 - ema_26
            return macd
        except Exception:
            return pd.Series([0] * len(prices), index=prices.index)
    
    def _calculate_bb_position(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Vereinfachte Bollinger Band Position"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            bb_position = (prices - lower_band) / (upper_band - lower_band)
            return bb_position
        except Exception:
            return pd.Series([0.5] * len(prices), index=prices.index)
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Vereinfachte ATR Berechnung"""
        try:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = ranges.rolling(window=period).mean()
            return atr
        except Exception:
            return pd.Series([0.01] * len(df), index=df.index)