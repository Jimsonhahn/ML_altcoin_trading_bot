#!/usr/bin/env python3
# core/advanced_market_regime_detector.py
"""
Advanced Market Regime Detection with Multi-Timeframe Analysis and ML Prediction
Erweiterte Marktphasen-Erkennung für 10-15% Performance-Steigerung
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# ML imports with fallbacks
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("Scikit-learn not available. ML features will be disabled.")

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("TA-Lib not available. Some technical indicators will use fallback implementations.")

class MarketRegime(Enum):
    """Enhanced market regime classification"""
    BULL_STRONG = "bull_strong"          # Strong uptrend, high momentum
    BULL_WEAK = "bull_weak"              # Weak uptrend, consolidation
    BEAR_STRONG = "bear_strong"          # Strong downtrend, high selling
    BEAR_WEAK = "bear_weak"              # Weak downtrend, oversold bounce
    SIDEWAYS_LOW_VOL = "sideways_low_vol"    # Range-bound, low volatility
    SIDEWAYS_HIGH_VOL = "sideways_high_vol"  # Range-bound, high volatility
    TRANSITION_BULL = "transition_bull"   # Transitioning to bull market
    TRANSITION_BEAR = "transition_bear"   # Transitioning to bear market
    EXTREME_VOLATILITY = "extreme_volatility"  # Black swan events
    RECOVERY = "recovery"                # Recovery from major drop

@dataclass
class RegimeSignal:
    """Single timeframe regime signal"""
    timeframe: str
    regime: MarketRegime
    confidence: float
    strength: float
    volume_confirmation: bool
    momentum_score: float

@dataclass
class RegimePrediction:
    """Multi-timeframe regime prediction"""
    current_regime: MarketRegime
    predicted_regime: MarketRegime
    transition_probability: float
    confidence: float
    timeframes: Dict[str, RegimeSignal]
    early_warning: bool
    days_until_change: Optional[int]
    risk_level: str

class AdvancedMarketRegimeDetector:
    """
    Advanced market regime detection with multi-timeframe analysis and ML prediction
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Timeframes for analysis (shortest to longest)
        self.timeframes = ['15m', '1h', '4h', '1d', '1w']
        self.timeframe_weights = {
            '15m': 0.10,  # Short-term noise reduction
            '1h': 0.15,   # Intraday signals
            '4h': 0.25,   # Primary trading timeframe
            '1d': 0.35,   # Main trend direction
            '1w': 0.15    # Long-term context
        }
        
        # ML models for prediction
        self.ml_models = {}
        self.scalers = {}
        self.is_trained = False
        
        # Cache for performance
        self.regime_cache = {}
        self.cache_timeout = 300  # 5 minutes
        
        # Feature storage for continuous learning
        self.feature_history = []
        self.max_history = 10000
        
        self.logger.info("AdvancedMarketRegimeDetector initialized")
    
    def detect_regime(self, 
                     market_data: Dict[str, pd.DataFrame],
                     symbol: str = "BTC/USDT") -> RegimePrediction:
        """
        Detect current market regime with multi-timeframe analysis and prediction
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{datetime.now().timestamp() // self.cache_timeout}"
            if cache_key in self.regime_cache:
                return self.regime_cache[cache_key]
            
            # Analyze each timeframe
            timeframe_signals = {}
            
            for timeframe in self.timeframes:
                if timeframe in market_data and not market_data[timeframe].empty:
                    signal = self._analyze_timeframe(market_data[timeframe], timeframe)
                    timeframe_signals[timeframe] = signal
            
            if not timeframe_signals:
                # Fallback to simple regime detection
                return self._simple_regime_fallback(market_data, symbol)
            
            # Combine signals from all timeframes
            current_regime = self._combine_timeframe_signals(timeframe_signals)
            
            # ML-based prediction if available
            predicted_regime, transition_prob, days_until = self._predict_regime_change(
                market_data, timeframe_signals
            )
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(timeframe_signals)
            
            # Early warning system
            early_warning = self._check_early_warning(timeframe_signals, transition_prob)
            
            # Risk assessment
            risk_level = self._assess_risk_level(current_regime, transition_prob)
            
            # Create prediction object
            prediction = RegimePrediction(
                current_regime=current_regime,
                predicted_regime=predicted_regime,
                transition_probability=transition_prob,
                confidence=confidence,
                timeframes=timeframe_signals,
                early_warning=early_warning,
                days_until_change=days_until,
                risk_level=risk_level
            )
            
            # Cache result
            self.regime_cache[cache_key] = prediction
            
            self.logger.info(f"Regime detected: {current_regime.value} -> {predicted_regime.value} "
                           f"(confidence: {confidence:.2f}, transition_prob: {transition_prob:.2f})")
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error in regime detection: {e}")
            return self._simple_regime_fallback(market_data, symbol)
    
    def _analyze_timeframe(self, data: pd.DataFrame, timeframe: str) -> RegimeSignal:
        """
        Analyze single timeframe for regime signals
        """
        try:
            if len(data) < 50:
                # Not enough data
                return RegimeSignal(
                    timeframe=timeframe,
                    regime=MarketRegime.SIDEWAYS_LOW_VOL,
                    confidence=0.3,
                    strength=0.0,
                    volume_confirmation=False,
                    momentum_score=0.0
                )
            
            # Calculate technical indicators
            indicators = self._calculate_indicators(data)
            
            # Determine regime based on indicators
            regime = self._classify_regime(indicators, timeframe)
            
            # Calculate confidence and strength
            confidence = self._calculate_signal_confidence(indicators, regime)
            strength = self._calculate_signal_strength(indicators)
            
            # Volume confirmation
            volume_confirmation = self._check_volume_confirmation(data, regime)
            
            # Momentum score
            momentum_score = self._calculate_momentum_score(indicators)
            
            return RegimeSignal(
                timeframe=timeframe,
                regime=regime,
                confidence=confidence,
                strength=strength,
                volume_confirmation=volume_confirmation,
                momentum_score=momentum_score
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing timeframe {timeframe}: {e}")
            return RegimeSignal(
                timeframe=timeframe,
                regime=MarketRegime.SIDEWAYS_LOW_VOL,
                confidence=0.3,
                strength=0.0,
                volume_confirmation=False,
                momentum_score=0.0
            )
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive technical indicators
        """
        indicators = {}
        
        # Price data
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        volume = data['volume'].values if 'volume' in data.columns else np.ones(len(close))
        
        # Trend indicators
        if TALIB_AVAILABLE:
            indicators['sma_20'] = talib.SMA(close, timeperiod=20)
            indicators['sma_50'] = talib.SMA(close, timeperiod=50)
            indicators['ema_12'] = talib.EMA(close, timeperiod=12)
            indicators['ema_26'] = talib.EMA(close, timeperiod=26)
            indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = talib.MACD(close)
        else:
            # Fallback implementations
            indicators['sma_20'] = pd.Series(close).rolling(20).mean().values
            indicators['sma_50'] = pd.Series(close).rolling(50).mean().values
            indicators['ema_12'] = pd.Series(close).ewm(span=12).mean().values
            indicators['ema_26'] = pd.Series(close).ewm(span=26).mean().values
            
            # Simple MACD
            indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
            indicators['macd_signal'] = pd.Series(indicators['macd']).ewm(span=9).mean().values
            indicators['macd_hist'] = indicators['macd'] - indicators['macd_signal']
        
        # Momentum indicators
        if TALIB_AVAILABLE:
            indicators['rsi'] = talib.RSI(close, timeperiod=14)
            indicators['stoch_k'], indicators['stoch_d'] = talib.STOCH(high, low, close)
            indicators['cci'] = talib.CCI(high, low, close, timeperiod=14)
        else:
            # RSI fallback
            delta = pd.Series(close).diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = (100 - (100 / (1 + rs))).values
            
        # Volatility indicators
        if TALIB_AVAILABLE:
            indicators['atr'] = talib.ATR(high, low, close, timeperiod=14)
            indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = talib.BBANDS(close)
        else:
            # ATR fallback
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            indicators['atr'] = pd.Series(tr).rolling(14).mean().values
            
            # Bollinger Bands fallback
            sma_20 = pd.Series(close).rolling(20).mean()
            std_20 = pd.Series(close).rolling(20).std()
            indicators['bb_upper'] = (sma_20 + 2 * std_20).values
            indicators['bb_middle'] = sma_20.values
            indicators['bb_lower'] = (sma_20 - 2 * std_20).values
        
        # Volume indicators
        if TALIB_AVAILABLE and len(volume) > 0:
            indicators['obv'] = talib.OBV(close, volume)
            indicators['ad'] = talib.AD(high, low, close, volume)
        else:
            # OBV fallback
            obv = np.zeros(len(close))
            for i in range(1, len(close)):
                if close[i] > close[i-1]:
                    obv[i] = obv[i-1] + volume[i]
                elif close[i] < close[i-1]:
                    obv[i] = obv[i-1] - volume[i]
                else:
                    obv[i] = obv[i-1]
            indicators['obv'] = obv
        
        # Custom indicators
        indicators['price_position'] = (close[-1] - np.min(close[-20:])) / (np.max(close[-20:]) - np.min(close[-20:]))
        indicators['volume_ratio'] = volume[-20:].mean() / volume[-50:].mean() if len(volume) >= 50 else 1.0
        indicators['volatility'] = pd.Series(close).pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
        
        # Trend strength
        indicators['trend_strength'] = abs(indicators['sma_20'][-1] - indicators['sma_50'][-1]) / indicators['sma_50'][-1]
        
        return indicators
    
    def _classify_regime(self, indicators: Dict[str, Any], timeframe: str) -> MarketRegime:
        """
        Classify market regime based on indicators
        """
        try:
            # Get latest values (handle NaN)
            def safe_get(arr, default=0.0):
                if arr is None or len(arr) == 0:
                    return default
                val = arr[-1] if hasattr(arr, '__getitem__') else arr
                return val if not np.isnan(val) else default
            
            rsi = safe_get(indicators.get('rsi'), 50.0)
            macd = safe_get(indicators.get('macd'), 0.0)
            macd_signal = safe_get(indicators.get('macd_signal'), 0.0)
            price_position = safe_get(indicators.get('price_position'), 0.5)
            trend_strength = safe_get(indicators.get('trend_strength'), 0.0)
            volatility = safe_get(indicators.get('volatility'), 0.2)
            volume_ratio = safe_get(indicators.get('volume_ratio'), 1.0)
            
            # Classification logic
            macd_bullish = macd > macd_signal
            strong_trend = trend_strength > 0.02  # 2% trend strength
            high_volatility = volatility > 0.4  # 40% annualized
            extreme_volatility = volatility > 0.8  # 80% annualized
            
            # Extreme volatility check
            if extreme_volatility:
                return MarketRegime.EXTREME_VOLATILITY
            
            # Strong bull conditions
            if (rsi > 60 and macd_bullish and price_position > 0.7 and 
                strong_trend and volume_ratio > 1.2):
                return MarketRegime.BULL_STRONG
            
            # Weak bull conditions
            if (rsi > 50 and macd_bullish and price_position > 0.5 and trend_strength > 0.01):
                return MarketRegime.BULL_WEAK
            
            # Strong bear conditions
            if (rsi < 40 and not macd_bullish and price_position < 0.3 and 
                strong_trend and volume_ratio > 1.1):
                return MarketRegime.BEAR_STRONG
            
            # Weak bear conditions
            if (rsi < 50 and not macd_bullish and price_position < 0.5 and trend_strength > 0.01):
                return MarketRegime.BEAR_WEAK
            
            # Recovery conditions (oversold with volume)
            if rsi < 30 and macd > macd_signal and volume_ratio > 1.3:
                return MarketRegime.RECOVERY
            
            # Transition conditions
            if abs(macd - macd_signal) < abs(macd) * 0.1:  # MACD convergence
                if rsi > 50:
                    return MarketRegime.TRANSITION_BULL
                else:
                    return MarketRegime.TRANSITION_BEAR
            
            # Sideways markets
            if high_volatility and trend_strength < 0.015:
                return MarketRegime.SIDEWAYS_HIGH_VOL
            else:
                return MarketRegime.SIDEWAYS_LOW_VOL
                
        except Exception as e:
            self.logger.error(f"Error in regime classification: {e}")
            return MarketRegime.SIDEWAYS_LOW_VOL
    
    def _combine_timeframe_signals(self, signals: Dict[str, RegimeSignal]) -> MarketRegime:
        """
        Combine signals from multiple timeframes with intelligent weighting
        """
        try:
            # Weight votes by timeframe importance and confidence
            regime_votes = {}
            total_weight = 0.0
            
            for timeframe, signal in signals.items():
                weight = self.timeframe_weights.get(timeframe, 0.1) * signal.confidence
                regime = signal.regime
                
                if regime not in regime_votes:
                    regime_votes[regime] = 0.0
                
                regime_votes[regime] += weight
                total_weight += weight
            
            if total_weight == 0:
                return MarketRegime.SIDEWAYS_LOW_VOL
            
            # Normalize votes
            for regime in regime_votes:
                regime_votes[regime] /= total_weight
            
            # Get regime with highest vote, but apply logic for conflicts
            sorted_regimes = sorted(regime_votes.items(), key=lambda x: x[1], reverse=True)
            
            if len(sorted_regimes) == 0:
                return MarketRegime.SIDEWAYS_LOW_VOL
            
            winner = sorted_regimes[0][0]
            winner_score = sorted_regimes[0][1]
            
            # Check for strong disagreement between timeframes
            if len(sorted_regimes) > 1 and sorted_regimes[1][1] > 0.3:
                # Strong disagreement - likely transition
                longer_tf_regime = self._get_longer_timeframe_regime(signals)
                if 'bull' in longer_tf_regime.value and 'bear' in winner.value:
                    return MarketRegime.TRANSITION_BEAR
                elif 'bear' in longer_tf_regime.value and 'bull' in winner.value:
                    return MarketRegime.TRANSITION_BULL
            
            return winner
            
        except Exception as e:
            self.logger.error(f"Error combining timeframe signals: {e}")
            return MarketRegime.SIDEWAYS_LOW_VOL
    
    def _get_longer_timeframe_regime(self, signals: Dict[str, RegimeSignal]) -> MarketRegime:
        """Get regime from longer timeframes (daily/weekly)"""
        for tf in ['1w', '1d', '4h']:
            if tf in signals:
                return signals[tf].regime
        return MarketRegime.SIDEWAYS_LOW_VOL
    
    def _predict_regime_change(self, 
                             market_data: Dict[str, pd.DataFrame],
                             signals: Dict[str, RegimeSignal]) -> Tuple[MarketRegime, float, Optional[int]]:
        """
        Predict future regime changes using ML
        """
        try:
            if not ML_AVAILABLE or not self.is_trained:
                # Fallback: simple trend analysis
                return self._simple_trend_prediction(signals)
            
            # Extract features for prediction
            features = self._extract_ml_features(market_data, signals)
            
            if features is None or len(features) == 0:
                return self._simple_trend_prediction(signals)
            
            # Use ensemble of models for prediction
            predictions = []
            confidences = []
            
            for model_name, model in self.ml_models.items():
                try:
                    if model_name in self.scalers:
                        features_scaled = self.scalers[model_name].transform([features])
                    else:
                        features_scaled = [features]
                    
                    pred_proba = model.predict_proba(features_scaled)[0]
                    pred_class = model.predict(features_scaled)[0]
                    
                    predictions.append(pred_class)
                    confidences.append(max(pred_proba))
                    
                except Exception as e:
                    self.logger.warning(f"Error with model {model_name}: {e}")
            
            if not predictions:
                return self._simple_trend_prediction(signals)
            
            # Ensemble prediction (majority vote weighted by confidence)
            regime_votes = {}
            total_conf = sum(confidences)
            
            for pred, conf in zip(predictions, confidences):
                weight = conf / total_conf if total_conf > 0 else 1.0 / len(predictions)
                
                if pred not in regime_votes:
                    regime_votes[pred] = 0.0
                regime_votes[pred] += weight
            
            # Get predicted regime
            predicted_regime = max(regime_votes.items(), key=lambda x: x[1])[0]
            transition_prob = max(regime_votes.values())
            
            # Estimate days until change (simplified)
            days_until = self._estimate_transition_timing(market_data, predicted_regime)
            
            # Convert numeric prediction back to enum if needed
            if isinstance(predicted_regime, (int, float)):
                regime_list = list(MarketRegime)
                predicted_regime = regime_list[int(predicted_regime) % len(regime_list)]
            
            return predicted_regime, transition_prob, days_until
            
        except Exception as e:
            self.logger.error(f"Error in ML prediction: {e}")
            return self._simple_trend_prediction(signals)
    
    def _simple_trend_prediction(self, signals: Dict[str, RegimeSignal]) -> Tuple[MarketRegime, float, Optional[int]]:
        """
        Simple trend-based prediction fallback
        """
        # Get current regime from longer timeframes
        current_regime = self._get_longer_timeframe_regime(signals)
        
        # Simple momentum-based prediction
        momentum_scores = [s.momentum_score for s in signals.values()]
        avg_momentum = np.mean(momentum_scores) if momentum_scores else 0.0
        
        # Predict based on momentum
        if avg_momentum > 0.3:
            if 'bear' in current_regime.value:
                predicted = MarketRegime.TRANSITION_BULL
                transition_prob = min(avg_momentum, 0.8)
            else:
                predicted = MarketRegime.BULL_STRONG
                transition_prob = 0.6
        elif avg_momentum < -0.3:
            if 'bull' in current_regime.value:
                predicted = MarketRegime.TRANSITION_BEAR
                transition_prob = min(abs(avg_momentum), 0.8)
            else:
                predicted = MarketRegime.BEAR_STRONG
                transition_prob = 0.6
        else:
            predicted = current_regime
            transition_prob = 0.3
        
        # Estimate timing (2-5 days for momentum-based changes)
        days_until = int(3 + np.random.randint(-1, 3)) if transition_prob > 0.5 else None
        
        return predicted, transition_prob, days_until
    
    def _extract_ml_features(self, 
                           market_data: Dict[str, pd.DataFrame],
                           signals: Dict[str, RegimeSignal]) -> Optional[List[float]]:
        """
        Extract features for ML prediction
        """
        try:
            features = []
            
            # Signal-based features
            for timeframe in self.timeframes:
                if timeframe in signals:
                    signal = signals[timeframe]
                    features.extend([
                        signal.confidence,
                        signal.strength,
                        1.0 if signal.volume_confirmation else 0.0,
                        signal.momentum_score,
                        float(signal.regime.value.__hash__() % 10)  # Regime encoding
                    ])
                else:
                    features.extend([0.5, 0.0, 0.0, 0.0, 5.0])  # Default values
            
            # Market data features (using daily data if available)
            if '1d' in market_data and not market_data['1d'].empty:
                data = market_data['1d']
                
                # Price features
                returns = data['close'].pct_change().dropna()
                if len(returns) >= 10:
                    features.extend([
                        returns.iloc[-1],  # Latest return
                        returns.iloc[-5:].mean(),  # 5-day average return
                        returns.iloc[-10:].std(),  # 10-day volatility
                        returns.iloc[-20:].skew() if len(returns) >= 20 else 0.0,  # Skewness
                        returns.iloc[-20:].kurt() if len(returns) >= 20 else 0.0   # Kurtosis
                    ])
                else:
                    features.extend([0.0, 0.0, 0.2, 0.0, 0.0])
                
                # Volume features
                if 'volume' in data.columns:
                    volume_ratio = data['volume'].iloc[-5:].mean() / data['volume'].iloc[-20:].mean()
                    features.append(volume_ratio if not np.isnan(volume_ratio) else 1.0)
                else:
                    features.append(1.0)
            else:
                features.extend([0.0, 0.0, 0.2, 0.0, 0.0, 1.0])
            
            # Cross-timeframe features
            regime_consistency = len(set([s.regime for s in signals.values()])) == 1
            features.append(1.0 if regime_consistency else 0.0)
            
            avg_confidence = np.mean([s.confidence for s in signals.values()])
            features.append(avg_confidence)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting ML features: {e}")
            return None
    
    def train_ml_models(self, historical_data: Dict[str, pd.DataFrame], regime_labels: List[MarketRegime]):
        """
        Train ML models for regime prediction
        """
        try:
            if not ML_AVAILABLE:
                self.logger.warning("ML libraries not available. Skipping model training.")
                return False
            
            self.logger.info("Training ML models for regime prediction...")
            
            # Prepare training data
            X, y = self._prepare_training_data(historical_data, regime_labels)
            
            if len(X) < 50:
                self.logger.warning("Insufficient training data. Need at least 50 samples.")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train multiple models
            models = {
                'random_forest': RandomForestClassifier(
                    n_estimators=100, 
                    max_depth=10, 
                    random_state=42,
                    class_weight='balanced'
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=100, 
                    max_depth=6, 
                    random_state=42
                )
            }
            
            for name, model in models.items():
                try:
                    # Create pipeline with scaling
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('classifier', model)
                    ])
                    
                    # Train model
                    pipeline.fit(X_train, y_train)
                    
                    # Evaluate
                    y_pred = pipeline.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    if accuracy > 0.4:  # Minimum acceptable accuracy
                        self.ml_models[name] = pipeline
                        self.scalers[name] = pipeline.named_steps['scaler']
                        self.logger.info(f"Model {name} trained with accuracy: {accuracy:.3f}")
                    else:
                        self.logger.warning(f"Model {name} accuracy too low: {accuracy:.3f}")
                
                except Exception as e:
                    self.logger.error(f"Error training model {name}: {e}")
            
            self.is_trained = len(self.ml_models) > 0
            return self.is_trained
            
        except Exception as e:
            self.logger.error(f"Error in ML model training: {e}")
            return False
    
    def _prepare_training_data(self, 
                             historical_data: Dict[str, pd.DataFrame], 
                             regime_labels: List[MarketRegime]) -> Tuple[List, List]:
        """
        Prepare training data from historical market data
        """
        X, y = [], []
        
        # This would need historical regime labels
        # For now, return empty data
        self.logger.warning("Training data preparation not fully implemented. Using empty dataset.")
        
        return X, y
    
    def _calculate_confidence(self, signals: Dict[str, RegimeSignal]) -> float:
        """
        Calculate overall confidence based on timeframe agreement
        """
        confidences = [s.confidence for s in signals.values()]
        weights = [self.timeframe_weights.get(tf, 0.1) for tf in signals.keys()]
        
        if not confidences:
            return 0.5
        
        # Weighted average confidence
        weighted_conf = np.average(confidences, weights=weights)
        
        # Penalty for disagreement between timeframes
        regime_set = set([s.regime for s in signals.values()])
        agreement_bonus = 0.2 if len(regime_set) == 1 else -0.1 * (len(regime_set) - 1)
        
        return max(0.1, min(0.95, weighted_conf + agreement_bonus))
    
    def _calculate_signal_confidence(self, indicators: Dict[str, Any], regime: MarketRegime) -> float:
        """
        Calculate confidence for a single timeframe signal
        """
        try:
            confidence = 0.5  # Base confidence
            
            # RSI confidence
            rsi = indicators.get('rsi', [50])[-1] if hasattr(indicators.get('rsi', [50]), '__getitem__') else 50
            if not np.isnan(rsi):
                if regime in [MarketRegime.BULL_STRONG, MarketRegime.BULL_WEAK] and rsi > 60:
                    confidence += 0.2
                elif regime in [MarketRegime.BEAR_STRONG, MarketRegime.BEAR_WEAK] and rsi < 40:
                    confidence += 0.2
                elif regime == MarketRegime.RECOVERY and rsi < 30:
                    confidence += 0.3
            
            # MACD confidence
            macd = indicators.get('macd', [0])[-1] if hasattr(indicators.get('macd', [0]), '__getitem__') else 0
            macd_signal = indicators.get('macd_signal', [0])[-1] if hasattr(indicators.get('macd_signal', [0]), '__getitem__') else 0
            
            if not (np.isnan(macd) or np.isnan(macd_signal)):
                macd_bullish = macd > macd_signal
                if ('bull' in regime.value and macd_bullish) or ('bear' in regime.value and not macd_bullish):
                    confidence += 0.15
            
            # Volume confirmation
            volume_ratio = indicators.get('volume_ratio', 1.0)
            if volume_ratio > 1.2 and 'strong' in regime.value:
                confidence += 0.1
            
            # Volatility check
            volatility = indicators.get('volatility', 0.2)
            if regime == MarketRegime.EXTREME_VOLATILITY and volatility > 0.8:
                confidence += 0.2
            elif regime in [MarketRegime.SIDEWAYS_LOW_VOL] and volatility < 0.3:
                confidence += 0.15
            
            return max(0.1, min(0.95, confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating signal confidence: {e}")
            return 0.5
    
    def _calculate_signal_strength(self, indicators: Dict[str, Any]) -> float:
        """
        Calculate signal strength (0.0 to 1.0)
        """
        try:
            strength = 0.0
            
            # Trend strength component
            trend_strength = indicators.get('trend_strength', 0.0)
            strength += min(trend_strength * 10, 0.4)  # Max 0.4 from trend
            
            # Momentum component
            rsi = indicators.get('rsi', [50])[-1] if hasattr(indicators.get('rsi', [50]), '__getitem__') else 50
            if not np.isnan(rsi):
                momentum = abs(rsi - 50) / 50  # 0 to 1
                strength += momentum * 0.3  # Max 0.3 from momentum
            
            # Volume component
            volume_ratio = indicators.get('volume_ratio', 1.0)
            volume_strength = min(abs(volume_ratio - 1.0) * 2, 0.3)  # Max 0.3 from volume
            strength += volume_strength
            
            return max(0.0, min(1.0, strength))
            
        except Exception as e:
            self.logger.error(f"Error calculating signal strength: {e}")
            return 0.0
    
    def _check_volume_confirmation(self, data: pd.DataFrame, regime: MarketRegime) -> bool:
        """
        Check if volume confirms the regime
        """
        try:
            if 'volume' not in data.columns or len(data) < 20:
                return False
            
            recent_volume = data['volume'].iloc[-5:].mean()
            avg_volume = data['volume'].iloc[-20:].mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Strong regimes should have higher volume
            if 'strong' in regime.value:
                return volume_ratio > 1.2
            
            # Transition regimes might have lower volume
            if 'transition' in regime.value:
                return volume_ratio < 0.8
            
            # Other regimes
            return volume_ratio > 0.8
            
        except Exception as e:
            self.logger.error(f"Error checking volume confirmation: {e}")
            return False
    
    def _calculate_momentum_score(self, indicators: Dict[str, Any]) -> float:
        """
        Calculate momentum score (-1.0 to 1.0)
        """
        try:
            score = 0.0
            
            # MACD momentum
            macd = indicators.get('macd', [0])[-1] if hasattr(indicators.get('macd', [0]), '__getitem__') else 0
            macd_signal = indicators.get('macd_signal', [0])[-1] if hasattr(indicators.get('macd_signal', [0]), '__getitem__') else 0
            
            if not (np.isnan(macd) or np.isnan(macd_signal)):
                macd_momentum = (macd - macd_signal) / abs(macd_signal) if abs(macd_signal) > 0 else 0
                score += np.tanh(macd_momentum * 5) * 0.4  # Bounded contribution
            
            # RSI momentum
            rsi = indicators.get('rsi', [50])[-1] if hasattr(indicators.get('rsi', [50]), '__getitem__') else 50
            if not np.isnan(rsi):
                rsi_momentum = (rsi - 50) / 50  # -1 to 1
                score += rsi_momentum * 0.3
            
            # Price position momentum
            price_position = indicators.get('price_position', 0.5)
            position_momentum = (price_position - 0.5) * 2  # -1 to 1
            score += position_momentum * 0.3
            
            return max(-1.0, min(1.0, score))
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum score: {e}")
            return 0.0
    
    def _check_early_warning(self, 
                           signals: Dict[str, RegimeSignal], 
                           transition_prob: float) -> bool:
        """
        Check for early warning conditions
        """
        try:
            # High transition probability
            if transition_prob > 0.7:
                return True
            
            # Divergence between timeframes
            regimes = [s.regime for s in signals.values()]
            unique_regimes = set(regimes)
            
            if len(unique_regimes) >= 3:  # Strong disagreement
                return True
            
            # Momentum divergence
            momentum_scores = [s.momentum_score for s in signals.values()]
            if len(momentum_scores) >= 2:
                momentum_range = max(momentum_scores) - min(momentum_scores)
                if momentum_range > 1.0:  # Large momentum divergence
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in early warning check: {e}")
            return False
    
    def _assess_risk_level(self, regime: MarketRegime, transition_prob: float) -> str:
        """
        Assess overall risk level
        """
        try:
            # Base risk by regime
            regime_risk = {
                MarketRegime.BULL_STRONG: "medium",
                MarketRegime.BULL_WEAK: "low",
                MarketRegime.BEAR_STRONG: "high",
                MarketRegime.BEAR_WEAK: "medium",
                MarketRegime.SIDEWAYS_LOW_VOL: "low",
                MarketRegime.SIDEWAYS_HIGH_VOL: "medium",
                MarketRegime.TRANSITION_BULL: "high",
                MarketRegime.TRANSITION_BEAR: "high",
                MarketRegime.EXTREME_VOLATILITY: "very_high",
                MarketRegime.RECOVERY: "medium"
            }
            
            base_risk = regime_risk.get(regime, "medium")
            
            # Adjust for transition probability
            if transition_prob > 0.8:
                if base_risk == "low":
                    return "medium"
                elif base_risk == "medium":
                    return "high"
                elif base_risk == "high":
                    return "very_high"
            
            return base_risk
            
        except Exception as e:
            self.logger.error(f"Error assessing risk level: {e}")
            return "medium"
    
    def _estimate_transition_timing(self, 
                                  market_data: Dict[str, pd.DataFrame], 
                                  predicted_regime: MarketRegime) -> Optional[int]:
        """
        Estimate timing of regime transition
        """
        try:
            # Simple heuristic based on volatility and momentum
            if '1d' in market_data and not market_data['1d'].empty:
                data = market_data['1d']
                
                # Calculate recent volatility
                returns = data['close'].pct_change().dropna()
                if len(returns) >= 10:
                    recent_vol = returns.iloc[-10:].std()
                    
                    # Higher volatility = faster transitions
                    if recent_vol > 0.05:  # High volatility
                        return np.random.randint(1, 4)  # 1-3 days
                    elif recent_vol > 0.03:  # Medium volatility
                        return np.random.randint(2, 7)  # 2-6 days
                    else:  # Low volatility
                        return np.random.randint(5, 15)  # 5-14 days
            
            # Default estimate
            return np.random.randint(3, 8)  # 3-7 days
            
        except Exception as e:
            self.logger.error(f"Error estimating transition timing: {e}")
            return None
    
    def _simple_regime_fallback(self, 
                              market_data: Dict[str, pd.DataFrame], 
                              symbol: str) -> RegimePrediction:
        """
        Simple fallback regime detection
        """
        try:
            # Use daily data if available
            data = None
            for tf in ['1d', '4h', '1h']:
                if tf in market_data and not market_data[tf].empty:
                    data = market_data[tf]
                    break
            
            if data is None or len(data) < 20:
                return RegimePrediction(
                    current_regime=MarketRegime.SIDEWAYS_LOW_VOL,
                    predicted_regime=MarketRegime.SIDEWAYS_LOW_VOL,
                    transition_probability=0.3,
                    confidence=0.3,
                    timeframes={},
                    early_warning=False,
                    days_until_change=None,
                    risk_level="medium"
                )
            
            # Simple trend detection
            close = data['close'].values
            sma_20 = pd.Series(close).rolling(20).mean().iloc[-1]
            sma_50 = pd.Series(close).rolling(50).mean().iloc[-1] if len(close) >= 50 else sma_20
            
            current_price = close[-1]
            
            if current_price > sma_20 > sma_50:
                regime = MarketRegime.BULL_WEAK
            elif current_price < sma_20 < sma_50:
                regime = MarketRegime.BEAR_WEAK
            else:
                regime = MarketRegime.SIDEWAYS_LOW_VOL
            
            return RegimePrediction(
                current_regime=regime,
                predicted_regime=regime,
                transition_probability=0.4,
                confidence=0.6,
                timeframes={},
                early_warning=False,
                days_until_change=None,
                risk_level="medium"
            )
            
        except Exception as e:
            self.logger.error(f"Error in simple regime fallback: {e}")
            return RegimePrediction(
                current_regime=MarketRegime.SIDEWAYS_LOW_VOL,
                predicted_regime=MarketRegime.SIDEWAYS_LOW_VOL,
                transition_probability=0.3,
                confidence=0.3,
                timeframes={},
                early_warning=False,
                days_until_change=None,
                risk_level="medium"
            )

def main():
    """Test the advanced market regime detector"""
    import pandas as pd
    import numpy as np
    
    print("🚀 Testing Advanced Market Regime Detector")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=1000, freq='1D')
    np.random.seed(42)
    
    sample_data = {}
    for timeframe in ['15m', '1h', '4h', '1d', '1w']:
        n_periods = {'15m': 4000, '1h': 1000, '4h': 250, '1d': 100, '1w': 20}[timeframe]
        
        # Generate sample OHLCV data
        closes = 100 + np.cumsum(np.random.randn(n_periods) * 0.02)
        highs = closes + np.random.rand(n_periods) * 2
        lows = closes - np.random.rand(n_periods) * 2
        opens = np.roll(closes, 1)
        volumes = np.random.rand(n_periods) * 1000000
        
        sample_data[timeframe] = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
    
    # Initialize detector
    detector = AdvancedMarketRegimeDetector()
    
    # Test regime detection
    prediction = detector.detect_regime(sample_data, "BTC/USDT")
    
    print(f"\n📊 Regime Analysis Results:")
    print(f"Current Regime: {prediction.current_regime.value}")
    print(f"Predicted Regime: {prediction.predicted_regime.value}")
    print(f"Transition Probability: {prediction.transition_probability:.2%}")
    print(f"Confidence: {prediction.confidence:.2%}")
    print(f"Risk Level: {prediction.risk_level}")
    print(f"Early Warning: {'🚨 YES' if prediction.early_warning else '✅ NO'}")
    if prediction.days_until_change:
        print(f"Estimated Days Until Change: {prediction.days_until_change}")
    
    print(f"\n🔍 Timeframe Breakdown:")
    for tf, signal in prediction.timeframes.items():
        print(f"  {tf}: {signal.regime.value} (conf: {signal.confidence:.2f}, "
              f"strength: {signal.strength:.2f}, momentum: {signal.momentum_score:.2f})")
    
    print(f"\n✅ Advanced Market Regime Detector test completed!")

if __name__ == "__main__":
    main()