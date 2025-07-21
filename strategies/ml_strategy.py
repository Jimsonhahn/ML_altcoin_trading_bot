"""
Enhanced Machine Learning Strategy
Integriert ML-basierte Marktvorhersagen und Alpha-Faktoren für präzise Trading-Entscheidungen
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, List, Optional
import logging
import asyncio
from datetime import datetime, timedelta
from .strategy_base import Strategy
from core.position import Position

# ML Components
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ml.market_predictor import MarketPredictor
    from ml.alpha_finder import AlphaFinder
    HAS_ML_COMPONENTS = True
except ImportError:
    HAS_ML_COMPONENTS = False

logger = logging.getLogger(__name__)

class MLStrategy(Strategy):
    """
    Erweiterte ML-basierte Trading-Strategie mit:
    - Marktphasen-Vorhersage
    - Alpha-Faktor-Integration
    - Dynamische Stop-Loss/Take-Profit
    - Confidence-basierte Positionierung
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Basis-Konfiguration
        self.lookback_period = config.get('lookback_period', 100)
        self.prediction_threshold = config.get('prediction_threshold', 0.6)
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.max_position_size = config.get('max_position_size', 0.1)  # 10% des Portfolios
        
        # ML-Komponenten Konfiguration
        self.ml_config = config.get('ml_config', {})
        self.alpha_config = config.get('alpha_config', {})
        self.enable_market_predictor = config.get('enable_market_predictor', True)
        self.enable_alpha_finder = config.get('enable_alpha_finder', True)
        
        # Volatilitäts-basierte Parameter
        self.volatility_lookback = config.get('volatility_lookback', 20)
        self.volatility_multiplier = config.get('volatility_multiplier', 2.0)
        
        # Position Management
        self.max_holding_period = config.get('max_holding_period', 48)  # 48 Stunden
        self.rebalance_threshold = config.get('rebalance_threshold', 0.1)  # 10% Konfidenz-Änderung
        
        # ML-Komponenten initialisieren
        self.market_predictor = None
        self.alpha_finder = None
        
        if HAS_ML_COMPONENTS:
            if self.enable_market_predictor:
                self.market_predictor = MarketPredictor(self.ml_config)
                logger.info("MarketPredictor initialized")
            
            if self.enable_alpha_finder:
                self.alpha_finder = AlphaFinder(self.alpha_config)
                logger.info("AlphaFinder initialized")
        else:
            logger.warning("ML components not available, using fallback implementations")
        
        # Cache für ML-Vorhersagen
        self.prediction_cache = {}
        self.alpha_cache = {}
        self.last_prediction_time = {}
        
        # Performance Tracking
        self.ml_performance = {
            'predictions_made': 0,
            'correct_predictions': 0,
            'alpha_signals_used': 0,
            'total_trades': 0,
            'profitable_trades': 0
        }
        
        logger.info("Enhanced ML Strategy initialized")

    async def execute(self, market_data: Dict[str, Any]) -> List[Position]:
        """
        Hauptmethode für die ML-basierte Strategie-Ausführung
        """
        try:
            positions = []
            
            # Extrahiere Symbol und Daten
            symbol = market_data.get('symbol', 'BTC/USDT')
            current_price = market_data.get('price', 0)
            
            # Prüfe ob genügend Daten vorhanden
            if len(market_data.get('ohlcv', [])) < self.lookback_period:
                logger.warning(f"Insufficient data for {symbol}")
                return positions
            
            # Konvertiere zu DataFrame
            ohlcv_data = market_data['ohlcv']
            data = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Hole ML-Vorhersagen
            ml_prediction = await self._get_ml_prediction(symbol, data)
            
            # Hole Alpha-Signale
            alpha_signals = await self._get_alpha_signals(symbol)
            
            # Kombiniere Signale
            combined_signal = self._combine_signals(ml_prediction, alpha_signals, symbol)
            
            # Generiere Positionen basierend auf kombiniertem Signal
            if combined_signal['action'] != 'HOLD':
                position = await self._create_position(symbol, combined_signal, current_price, data)
                if position:
                    positions.append(position)
            
            return positions
            
        except Exception as e:
            logger.error(f"Error in ML strategy execution: {e}")
            return []
    
    def calculate_signal(self, symbol: str, data: pd.DataFrame,
                        current_price: float) -> Tuple[str, Dict[str, Any]]:
        """
        Legacy-Methode für Rückwärtskompatibilität
        """
        try:
            # Verwende async execute in sync context
            market_data = {
                'symbol': symbol,
                'price': current_price,
                'ohlcv': data.values.tolist()
            }
            
            # Vereinfachte synchrone Ausführung
            ml_prediction = self._get_ml_prediction_sync(symbol, data)
            
            signal = 'HOLD'
            confidence = 0.0
            
            if ml_prediction and ml_prediction.get('confidence', 0) > self.confidence_threshold:
                predicted_phase = ml_prediction.get('predicted_phase', 'sideways')
                confidence = ml_prediction.get('confidence', 0)
                
                if predicted_phase == 'bull':
                    signal = 'BUY'
                elif predicted_phase == 'bear':
                    signal = 'SELL'
            
            return signal, {
                'signal': signal,
                'confidence': float(confidence),
                'prediction': ml_prediction,
                'method': 'ml_enhanced'
            }
            
        except Exception as e:
            logger.error(f"Error in ML signal calculation: {e}")
            return 'HOLD', {'confidence': 0.0, 'error': str(e)}

    async def _get_ml_prediction(self, symbol: str, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Holt ML-Vorhersage mit Caching
        """
        try:
            # Cache-Key erstellen
            cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H')}"
            
            # Prüfe Cache (stündlich)
            if cache_key in self.prediction_cache:
                return self.prediction_cache[cache_key]
            
            # Neue Vorhersage generieren
            if self.market_predictor:
                prediction = self.market_predictor.predict(data)
                
                if 'error' not in prediction:
                    self.prediction_cache[cache_key] = prediction
                    self.ml_performance['predictions_made'] += 1
                    return prediction
            
            # Fallback zu einfacher Vorhersage
            return self._get_ml_prediction_sync(symbol, data)
            
        except Exception as e:
            logger.error(f"Error getting ML prediction: {e}")
            return None
    
    def _get_ml_prediction_sync(self, symbol: str, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Synchrone ML-Vorhersage (Fallback)
        """
        try:
            # Verwende vereinfachte Vorhersage
            features = self._extract_features(data)
            prediction_score = self._generate_prediction(features)
            
            # Konvertiere zu Marktphase
            if prediction_score > 0.7:
                phase = 'bull'
            elif prediction_score < 0.3:
                phase = 'bear'
            elif abs(prediction_score - 0.5) < 0.1:
                phase = 'sideways'
            else:
                phase = 'volatile'
            
            return {
                'predicted_phase': phase,
                'confidence': abs(prediction_score - 0.5) * 2,
                'prediction_score': prediction_score,
                'model_type': 'fallback'
            }
            
        except Exception as e:
            logger.error(f"Error in sync ML prediction: {e}")
            return None
    
    async def _get_alpha_signals(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Holt Alpha-Signale
        """
        try:
            # Cache-Key erstellen
            cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            
            # Prüfe Cache (alle 10 Minuten)
            if cache_key in self.alpha_cache:
                return self.alpha_cache[cache_key]
            
            # Neue Alpha-Signale generieren
            if self.alpha_finder:
                all_signals = await self.alpha_finder.find_alpha_signals()
                
                # Filtere nach Symbol
                symbol_signals = [s for s in all_signals if s.symbol == symbol.split('/')[0]]
                
                # Konvertiere zu Dict
                alpha_signals = []
                for signal in symbol_signals:
                    alpha_signals.append({
                        'signal_type': signal.signal_type,
                        'strength': signal.strength,
                        'confidence': signal.confidence,
                        'source': signal.source
                    })
                
                self.alpha_cache[cache_key] = alpha_signals
                self.ml_performance['alpha_signals_used'] += len(alpha_signals)
                return alpha_signals
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting alpha signals: {e}")
            return []
    
    def _combine_signals(self, ml_prediction: Optional[Dict], alpha_signals: List[Dict], symbol: str) -> Dict[str, Any]:
        """
        Kombiniert ML-Vorhersage und Alpha-Signale
        """
        try:
            # Basis-Signal
            combined_signal = {
                'action': 'HOLD',
                'confidence': 0.0,
                'strength': 0.0,
                'components': {}
            }
            
            # ML-Vorhersage einbeziehen
            ml_weight = 0.6
            if ml_prediction:
                ml_confidence = ml_prediction.get('confidence', 0)
                ml_phase = ml_prediction.get('predicted_phase', 'sideways')
                
                if ml_confidence > self.confidence_threshold:
                    if ml_phase == 'bull':
                        combined_signal['strength'] += ml_weight * ml_confidence
                    elif ml_phase == 'bear':
                        combined_signal['strength'] -= ml_weight * ml_confidence
                
                combined_signal['components']['ml'] = {
                    'phase': ml_phase,
                    'confidence': ml_confidence,
                    'weight': ml_weight
                }
            
            # Alpha-Signale einbeziehen
            alpha_weight = 0.4
            if alpha_signals:
                alpha_strength = 0.0
                total_alpha_confidence = 0.0
                
                for signal in alpha_signals:
                    signal_strength = signal.get('strength', 0)
                    signal_confidence = signal.get('confidence', 0)
                    
                    # Gewichtung nach Signaltyp
                    type_weight = self._get_signal_type_weight(signal.get('signal_type', ''))
                    
                    alpha_strength += signal_strength * signal_confidence * type_weight
                    total_alpha_confidence += signal_confidence
                
                if len(alpha_signals) > 0:
                    avg_alpha_confidence = total_alpha_confidence / len(alpha_signals)
                    combined_signal['strength'] += alpha_weight * alpha_strength
                    
                    combined_signal['components']['alpha'] = {
                        'signals': len(alpha_signals),
                        'strength': alpha_strength,
                        'confidence': avg_alpha_confidence,
                        'weight': alpha_weight
                    }
            
            # Finale Entscheidung
            if combined_signal['strength'] > 0.3:
                combined_signal['action'] = 'BUY'
            elif combined_signal['strength'] < -0.3:
                combined_signal['action'] = 'SELL'
            
            # Berechne Gesamtkonfidenz
            combined_signal['confidence'] = min(abs(combined_signal['strength']), 1.0)
            
            return combined_signal
            
        except Exception as e:
            logger.error(f"Error combining signals: {e}")
            return {'action': 'HOLD', 'confidence': 0.0, 'strength': 0.0, 'error': str(e)}
    
    def _get_signal_type_weight(self, signal_type: str) -> float:
        """
        Gibt Gewichtung für verschiedene Signal-Typen zurück
        """
        weights = {
            'funding_rate_anomaly': 0.8,
            'orderbook_imbalance': 0.7,
            'cross_exchange_arbitrage': 0.6,
            'twitter_sentiment': 0.5,
            'reddit_sentiment': 0.4,
            'onchain_anomaly': 0.9
        }
        return weights.get(signal_type, 0.5)
    
    async def _create_position(self, symbol: str, signal: Dict[str, Any], current_price: float, data: pd.DataFrame) -> Optional[Position]:
        """
        Erstellt eine Position basierend auf Signal
        """
        try:
            action = signal['action']
            confidence = signal['confidence']
            
            if action == 'HOLD' or confidence < self.confidence_threshold:
                return None
            
            # Berechne Positionsgröße basierend auf Konfidenz
            position_size = self._calculate_position_size(confidence, data)
            
            # Berechne Stop-Loss und Take-Profit
            stop_loss, take_profit = self._calculate_stop_loss_take_profit(current_price, data, action)
            
            # Erstelle Position
            position = Position(
                symbol=symbol,
                side='buy' if action == 'BUY' else 'sell',
                size=position_size,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy='ml_enhanced',
                metadata={
                    'ml_confidence': confidence,
                    'signal_components': signal.get('components', {}),
                    'created_at': datetime.now().isoformat()
                }
            )
            
            return position
            
        except Exception as e:
            logger.error(f"Error creating position: {e}")
            return None
    
    def _calculate_position_size(self, confidence: float, data: pd.DataFrame) -> float:
        """
        Berechnet Positionsgröße basierend auf Konfidenz und Volatilität
        """
        try:
            # Basis-Positionsgröße
            base_size = self.max_position_size
            
            # Anpassung basierend auf Konfidenz
            confidence_multiplier = confidence
            
            # Anpassung basierend auf Volatilität
            volatility = self._calculate_volatility(data)
            volatility_multiplier = max(0.3, 1.0 - (volatility * 10))  # Weniger bei hoher Volatilität
            
            # Finale Positionsgröße
            position_size = base_size * confidence_multiplier * volatility_multiplier
            
            return min(position_size, self.max_position_size)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.01  # Minimale Positionsgröße
    
    def _calculate_stop_loss_take_profit(self, current_price: float, data: pd.DataFrame, action: str) -> Tuple[float, float]:
        """
        Berechnet dynamische Stop-Loss und Take-Profit basierend auf Volatilität
        """
        try:
            # Berechne ATR (Average True Range)
            atr = self._calculate_atr(data)
            
            # Dynamische Stop-Loss Distance
            stop_distance = atr * self.volatility_multiplier
            
            # Take-Profit ist 2x Stop-Loss
            take_profit_distance = stop_distance * 2
            
            if action == 'BUY':
                stop_loss = current_price - stop_distance
                take_profit = current_price + take_profit_distance
            else:  # SELL
                stop_loss = current_price + stop_distance
                take_profit = current_price - take_profit_distance
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Error calculating stop/take profit: {e}")
            # Fallback zu festen Prozentsätzen
            if action == 'BUY':
                return current_price * 0.98, current_price * 1.04
            else:
                return current_price * 1.02, current_price * 0.96
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """
        Berechnet die aktuelle Volatilität
        """
        try:
            returns = data['close'].pct_change().dropna()
            if len(returns) < self.volatility_lookback:
                return 0.02  # Default 2%
            
            volatility = returns.tail(self.volatility_lookback).std()
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.02
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """
        Berechnet Average True Range
        """
        try:
            if len(data) < period + 1:
                return data['close'].iloc[-1] * 0.01  # 1% fallback
            
            # True Range berechnen
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean().iloc[-1]
            
            return atr
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return data['close'].iloc[-1] * 0.01
    
    def _extract_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract features for ML model"""
        try:
            close_prices = data['close']

            features = {
                'rsi': float(self._calculate_rsi(close_prices)),
                'price_change_1d': float((close_prices.iloc[-1] / close_prices.iloc[-2] - 1) if len(close_prices) > 1 else 0),
                'volatility': float(self._calculate_volatility(data)),
                'atr': float(self._calculate_atr(data)),
                'volume_trend': float(data['volume'].pct_change().iloc[-1]) if len(data) > 1 else 0
            }

            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {'rsi': 50.0, 'price_change_1d': 0.0, 'volatility': 0.02}

    def _generate_prediction(self, features: Dict[str, float]) -> float:
        """Generate prediction (simplified)"""
        try:
            score = 0.5

            # RSI-basierte Bewertung
            rsi = features.get('rsi', 50)
            if rsi < 30:
                score += 0.3
            elif rsi > 70:
                score -= 0.3

            # Momentum-basierte Bewertung
            price_change = features.get('price_change_1d', 0)
            if price_change > 0.02:
                score += 0.2
            elif price_change < -0.02:
                score -= 0.2

            # Volatilitäts-basierte Anpassung
            volatility = features.get('volatility', 0.02)
            if volatility > 0.05:  # Hohe Volatilität
                score = 0.5 + (score - 0.5) * 0.7  # Dämpfung der Signale

            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            return 0.5

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        try:
            if len(prices) < period + 1:
                return 50.0

            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            if loss.iloc[-1] == 0:
                return 100.0

            rs = gain.iloc[-1] / loss.iloc[-1]
            return 100 - (100 / (1 + rs))
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50.0
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Gibt Performance-Metriken zurück
        """
        metrics = self.ml_performance.copy()
        
        # Berechne Erfolgsraten
        if metrics['predictions_made'] > 0:
            metrics['prediction_accuracy'] = metrics['correct_predictions'] / metrics['predictions_made']
        else:
            metrics['prediction_accuracy'] = 0.0
        
        if metrics['total_trades'] > 0:
            metrics['trade_success_rate'] = metrics['profitable_trades'] / metrics['total_trades']
        else:
            metrics['trade_success_rate'] = 0.0
        
        return metrics
    
    def update_performance(self, trade_result: Dict[str, Any]) -> None:
        """
        Aktualisiert Performance-Metriken nach einem Trade
        """
        try:
            self.ml_performance['total_trades'] += 1
            
            if trade_result.get('pnl', 0) > 0:
                self.ml_performance['profitable_trades'] += 1
            
            # Weitere Metriken können hier hinzugefügt werden
            
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
