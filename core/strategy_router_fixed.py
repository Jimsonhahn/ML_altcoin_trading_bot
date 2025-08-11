"""
Enhanced Strategy Router - Fixed version with proper import handling
Dynamische Marktlogik mit ML-Integration und verbesserter Fehlerbehandlung
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import asyncio

from config.settings import Settings
from utils.exceptions import ConfigurationError, StrategyError, MLError

# Optional ML imports with fallbacks
try:
    from core.market_analyzer import MarketAnalyzer
    HAS_MARKET_ANALYZER = True
except ImportError:
    HAS_MARKET_ANALYZER = False
    MarketAnalyzer = None

try:
    from ml_components import MLComponents
    HAS_ML_COMPONENTS = True
except ImportError:
    HAS_ML_COMPONENTS = False
    MLComponents = None

logger = logging.getLogger(__name__)


class MarketPhase(Enum):
    """Market Phase Enumeration"""
    SIDEWAYS = "sideways"
    BULL = "bull" 
    VOLATILE = "volatile"
    BEAR = "bear"
    EXTREME_FEAR = "extreme_fear"
    UNKNOWN = "unknown"


class StrategyRouter:
    """
    Enhanced Strategy Router with ML integration and robust error handling
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Core configuration
        self.enabled = settings.get('strategy_router.enabled', True)
        if not self.enabled:
            self.logger.warning("Strategy Router disabled in settings")
            return
        
        # Load strategy configurations
        self.regime_strategies_config = settings.get('strategy_router.regime_strategies', {})
        if not self.regime_strategies_config:
            self.logger.error("No regime strategies configuration found")
            raise ConfigurationError("Strategy router requires regime_strategies configuration")
        
        # Market analysis configuration
        self.market_config = settings.get('ml', {})
        self.core_symbols = self.market_config.get('regime_core_symbols', ['BTC/USDT'])
        self.confidence_threshold = settings.get('strategy_router.confidence_threshold', 0.6)
        
        # State tracking
        self.current_regime = MarketPhase.UNKNOWN
        self.regime_confidence = 0.0
        self.last_regime_update = datetime.min
        self.regime_history = []
        
        # Active strategies tracking
        self.active_strategies = {}
        self.strategy_allocations = {}
        
        # ML and analysis components
        self.market_analyzer: Optional[MarketAnalyzer] = None
        self.ml_components: Optional[MLComponents] = None
        
        # Performance tracking
        self.regime_performance = {}
        self.rebalance_count = 0
        self.last_rebalance_time = datetime.min
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info("Enhanced Strategy Router initialized")
        self.logger.info(f"Configured regimes: {list(self.regime_strategies_config.keys())}")
    
    def _initialize_components(self):
        """Initialize market analyzer and ML components"""
        try:
            # Initialize market analyzer if available
            if HAS_MARKET_ANALYZER:
                analyzer_config = {
                    'symbols': self.core_symbols,
                    'timeframe': self.settings.get('timeframes.analysis', '1h'),
                    'lookback_period': self.settings.get('analysis.lookback_period', 100)
                }
                self.market_analyzer = MarketAnalyzer(analyzer_config)
                self.logger.info("Market Analyzer initialized")
            else:
                self.logger.warning("Market Analyzer not available")
            
            # ML components will be set externally by trading bot
            
        except Exception as e:
            self.logger.error(f"Failed to initialize router components: {e}")
    
    def set_ml_components(self, ml_components: Optional[MLComponents]):
        """Set ML components externally"""
        self.ml_components = ml_components
        if ml_components:
            self.logger.info("ML components connected to Strategy Router")
        else:
            self.logger.warning("No ML components available for Strategy Router")
    
    async def analyze_market_regime(self, market_data: Dict[str, pd.DataFrame]) -> Tuple[MarketPhase, float]:
        """
        Analyze current market regime using multiple methods
        
        Args:
            market_data: Dictionary mapping symbols to their OHLCV data
            
        Returns:
            Tuple of (market_phase, confidence)
        """
        try:
            regime_signals = []
            
            # ML-based regime detection
            if self.ml_components and hasattr(self.ml_components, 'market_regime_detector'):
                try:
                    ml_regime = await self._get_ml_regime(market_data)
                    if ml_regime:
                        regime_signals.append(ml_regime)
                        self.logger.debug(f"ML regime signal: {ml_regime}")
                except Exception as e:
                    self.logger.warning(f"ML regime detection failed: {e}")
            
            # Technical analysis based regime
            if self.market_analyzer:
                try:
                    ta_regime = await self._get_technical_regime(market_data)
                    if ta_regime:
                        regime_signals.append(ta_regime)
                        self.logger.debug(f"Technical regime signal: {ta_regime}")
                except Exception as e:
                    self.logger.warning(f"Technical regime analysis failed: {e}")
            
            # Sentiment-based regime (if available)
            if self.ml_components and hasattr(self.ml_components, 'market_sentiment_analyzer'):
                try:
                    sentiment_regime = await self._get_sentiment_regime()
                    if sentiment_regime:
                        regime_signals.append(sentiment_regime)
                        self.logger.debug(f"Sentiment regime signal: {sentiment_regime}")
                except Exception as e:
                    self.logger.warning(f"Sentiment regime analysis failed: {e}")
            
            # Fallback to basic technical analysis
            if not regime_signals:
                fallback_regime = await self._get_fallback_regime(market_data)
                regime_signals.append(fallback_regime)
                self.logger.info("Using fallback regime detection")
            
            # Combine regime signals
            final_regime, confidence = self._combine_regime_signals(regime_signals)
            
            self.logger.info(f"Market regime analyzed: {final_regime.value} (confidence: {confidence:.2f})")
            return final_regime, confidence
            
        except Exception as e:
            self.logger.error(f"Market regime analysis failed: {e}")
            return MarketPhase.UNKNOWN, 0.0
    
    async def _get_ml_regime(self, market_data: Dict[str, pd.DataFrame]) -> Optional[Tuple[MarketPhase, float]]:
        """Get regime from ML components"""
        try:
            if not self.ml_components:
                return None
            
            # Combine data from all symbols
            combined_data = pd.concat(list(market_data.values()), ignore_index=True)
            
            # Get ML prediction
            if hasattr(self.ml_components, 'market_regime_detector'):
                regime_prediction = self.ml_components.market_regime_detector.predict_regime(combined_data)
                
                # Map ML regime to our MarketPhase enum
                regime_map = {
                    'bull': MarketPhase.BULL,
                    'bear': MarketPhase.BEAR,
                    'sideways': MarketPhase.SIDEWAYS,
                    'volatile': MarketPhase.VOLATILE,
                    'uptrend': MarketPhase.BULL,
                    'downtrend': MarketPhase.BEAR,
                    'ranging': MarketPhase.SIDEWAYS,
                    'high_volatility': MarketPhase.VOLATILE,
                    'crash': MarketPhase.EXTREME_FEAR
                }
                
                phase = regime_map.get(regime_prediction, MarketPhase.UNKNOWN)
                confidence = 0.8  # Default ML confidence
                
                return phase, confidence
            
            # Try enhanced ML components
            if hasattr(self.ml_components, 'market_predictor'):
                prediction = self.ml_components.market_predictor.predict(combined_data)
                if 'error' not in prediction:
                    predicted_phase = prediction.get('predicted_phase', 'unknown')
                    confidence = prediction.get('confidence', 0.0)
                    
                    phase_map = {
                        'bull': MarketPhase.BULL,
                        'bear': MarketPhase.BEAR,
                        'sideways': MarketPhase.SIDEWAYS,
                        'volatile': MarketPhase.VOLATILE,
                        'extreme_fear': MarketPhase.EXTREME_FEAR
                    }
                    
                    phase = phase_map.get(predicted_phase, MarketPhase.UNKNOWN)
                    return phase, confidence
            
            return None
            
        except Exception as e:
            self.logger.error(f"ML regime detection error: {e}")
            return None
    
    async def _get_technical_regime(self, market_data: Dict[str, pd.DataFrame]) -> Optional[Tuple[MarketPhase, float]]:
        """Get regime from technical analysis"""
        try:
            if not self.market_analyzer:
                return None
            
            # Analyze each symbol
            regime_votes = []
            
            for symbol, data in market_data.items():
                if data.empty:
                    continue
                
                # Calculate technical indicators
                analysis = await self._technical_analysis(data)
                regime_votes.append(analysis)
            
            if not regime_votes:
                return None
            
            # Aggregate votes
            return self._aggregate_technical_votes(regime_votes)
            
        except Exception as e:
            self.logger.error(f"Technical regime analysis error: {e}")
            return None
    
    async def _technical_analysis(self, data: pd.DataFrame) -> Tuple[MarketPhase, float]:
        """Perform technical analysis on single symbol data"""
        try:
            close_prices = data['close']
            high_prices = data['high']
            low_prices = data['low']
            volumes = data['volume']
            
            # Calculate indicators
            sma_20 = close_prices.rolling(20).mean()
            sma_50 = close_prices.rolling(50).mean()
            
            # Volatility (ATR-like)
            hl = high_prices - low_prices
            volatility = hl.rolling(14).mean() / close_prices.rolling(14).mean()
            current_volatility = volatility.iloc[-1] if len(volatility) > 0 else 0
            
            # Trend strength
            trend_strength = 0
            if len(sma_20) > 0 and len(sma_50) > 0:
                if sma_20.iloc[-1] > sma_50.iloc[-1]:
                    trend_strength = (sma_20.iloc[-1] / sma_50.iloc[-1] - 1) * 100
                else:
                    trend_strength = (sma_20.iloc[-1] / sma_50.iloc[-1] - 1) * 100
            
            # Price momentum
            momentum = 0
            if len(close_prices) >= 20:
                momentum = (close_prices.iloc[-1] / close_prices.iloc[-20] - 1) * 100
            
            # Determine regime
            if current_volatility > 0.05:  # High volatility
                if momentum < -10:
                    return MarketPhase.EXTREME_FEAR, 0.8
                else:
                    return MarketPhase.VOLATILE, 0.7
            elif trend_strength > 2:  # Strong uptrend
                return MarketPhase.BULL, 0.7
            elif trend_strength < -2:  # Strong downtrend
                return MarketPhase.BEAR, 0.7
            else:  # Sideways
                return MarketPhase.SIDEWAYS, 0.6
            
        except Exception as e:
            self.logger.error(f"Technical analysis error: {e}")
            return MarketPhase.UNKNOWN, 0.0
    
    def _aggregate_technical_votes(self, votes: List[Tuple[MarketPhase, float]]) -> Tuple[MarketPhase, float]:
        """Aggregate technical analysis votes"""
        try:
            if not votes:
                return MarketPhase.UNKNOWN, 0.0
            
            # Weight votes by confidence
            phase_scores = {}
            total_weight = 0
            
            for phase, confidence in votes:
                if phase not in phase_scores:
                    phase_scores[phase] = 0
                phase_scores[phase] += confidence
                total_weight += confidence
            
            # Find highest scoring phase
            if not phase_scores:
                return MarketPhase.UNKNOWN, 0.0
            
            best_phase = max(phase_scores.keys(), key=lambda k: phase_scores[k])
            avg_confidence = phase_scores[best_phase] / total_weight if total_weight > 0 else 0
            
            return best_phase, min(avg_confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"Vote aggregation error: {e}")
            return MarketPhase.UNKNOWN, 0.0
    
    async def _get_sentiment_regime(self) -> Optional[Tuple[MarketPhase, float]]:
        """Get regime from sentiment analysis"""
        try:
            if not self.ml_components or not hasattr(self.ml_components, 'alpha_finder'):
                return None
            
            # Get alpha signals which include sentiment
            alpha_signals = await self.ml_components.alpha_finder.find_alpha_signals()
            
            if not alpha_signals:
                return None
            
            # Analyze sentiment signals
            sentiment_scores = []
            for signal in alpha_signals:
                if signal.signal_type in ['twitter_sentiment', 'reddit_sentiment']:
                    sentiment_scores.append(signal.strength)
            
            if not sentiment_scores:
                return None
            
            avg_sentiment = np.mean(sentiment_scores)
            confidence = min(len(sentiment_scores) / 10, 1.0)  # More signals = higher confidence
            
            # Map sentiment to regime
            if avg_sentiment < -0.5:
                return MarketPhase.EXTREME_FEAR, confidence
            elif avg_sentiment < -0.2:
                return MarketPhase.BEAR, confidence
            elif avg_sentiment > 0.3:
                return MarketPhase.BULL, confidence
            else:
                return MarketPhase.SIDEWAYS, confidence
            
        except Exception as e:
            self.logger.error(f"Sentiment regime analysis error: {e}")
            return None
    
    async def _get_fallback_regime(self, market_data: Dict[str, pd.DataFrame]) -> Tuple[MarketPhase, float]:
        """Fallback regime detection using simple price analysis"""
        try:
            # Use BTC data as primary indicator
            btc_symbol = None
            for symbol in ['BTC/USDT', 'BTCUSDT', 'BTC']:
                if symbol in market_data:
                    btc_symbol = symbol
                    break
            
            if not btc_symbol:
                # Use first available symbol
                btc_symbol = list(market_data.keys())[0] if market_data else None
            
            if not btc_symbol or market_data[btc_symbol].empty:
                return MarketPhase.UNKNOWN, 0.0
            
            data = market_data[btc_symbol]
            close_prices = data['close']
            
            # Simple trend analysis
            if len(close_prices) < 20:
                return MarketPhase.UNKNOWN, 0.0
            
            # Short-term vs long-term momentum
            short_momentum = (close_prices.iloc[-1] / close_prices.iloc[-5] - 1) * 100
            long_momentum = (close_prices.iloc[-1] / close_prices.iloc[-20] - 1) * 100
            
            # Volatility
            returns = close_prices.pct_change().dropna()
            volatility = returns.tail(20).std() * 100 if len(returns) > 20 else 0
            
            # Determine regime
            if volatility > 5:
                if long_momentum < -15:
                    return MarketPhase.EXTREME_FEAR, 0.6
                else:
                    return MarketPhase.VOLATILE, 0.5
            elif long_momentum > 5:
                return MarketPhase.BULL, 0.5
            elif long_momentum < -5:
                return MarketPhase.BEAR, 0.5
            else:
                return MarketPhase.SIDEWAYS, 0.4
            
        except Exception as e:
            self.logger.error(f"Fallback regime detection error: {e}")
            return MarketPhase.UNKNOWN, 0.0
    
    def _combine_regime_signals(self, signals: List[Tuple[MarketPhase, float]]) -> Tuple[MarketPhase, float]:
        """Combine multiple regime signals into final decision"""
        try:
            if not signals:
                return MarketPhase.UNKNOWN, 0.0
            
            # Weight signals and combine
            phase_weights = {}
            total_weight = 0
            
            for phase, confidence in signals:
                if phase == MarketPhase.UNKNOWN:
                    continue
                
                if phase not in phase_weights:
                    phase_weights[phase] = 0
                
                phase_weights[phase] += confidence
                total_weight += confidence
            
            if not phase_weights:
                return MarketPhase.UNKNOWN, 0.0
            
            # Find best phase
            best_phase = max(phase_weights.keys(), key=lambda k: phase_weights[k])
            final_confidence = phase_weights[best_phase] / total_weight if total_weight > 0 else 0
            
            # Require minimum confidence
            if final_confidence < 0.3:
                return MarketPhase.UNKNOWN, final_confidence
            
            return best_phase, min(final_confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"Signal combination error: {e}")
            return MarketPhase.UNKNOWN, 0.0
    
    def update_market_regime(self, regime: str, confidence: float = 0.8):
        """Update market regime externally"""
        try:
            # Map string regime to enum
            regime_map = {
                'bull': MarketPhase.BULL,
                'bear': MarketPhase.BEAR,
                'sideways': MarketPhase.SIDEWAYS,
                'volatile': MarketPhase.VOLATILE,
                'extreme_fear': MarketPhase.EXTREME_FEAR,
                'uptrend': MarketPhase.BULL,
                'downtrend': MarketPhase.BEAR,
                'ranging': MarketPhase.SIDEWAYS,
                'high_volatility': MarketPhase.VOLATILE,
                'crash': MarketPhase.EXTREME_FEAR
            }
            
            new_regime = regime_map.get(regime, MarketPhase.UNKNOWN)
            
            if new_regime != self.current_regime:
                self.logger.info(f"Market regime updated: {self.current_regime.value} -> {new_regime.value}")
                
                # Update state
                self.current_regime = new_regime
                self.regime_confidence = confidence
                self.last_regime_update = datetime.now()
                
                # Add to history
                self.regime_history.append({
                    'regime': new_regime,
                    'confidence': confidence,
                    'timestamp': datetime.now()
                })
                
                # Keep history limited
                if len(self.regime_history) > 100:
                    self.regime_history = self.regime_history[-50:]
                
                # Update active strategies
                self._update_active_strategies()
            
        except Exception as e:
            self.logger.error(f"Error updating market regime: {e}")
    
    def _update_active_strategies(self):
        """Update active strategies based on current market regime"""
        try:
            regime_key = self.current_regime.value
            
            # Get strategy configuration for current regime
            strategy_config = self.regime_strategies_config.get(regime_key, {})
            
            if not strategy_config:
                self.logger.warning(f"No strategy configuration for regime: {regime_key}")
                # Use neutral regime as fallback
                strategy_config = self.regime_strategies_config.get('neutral', {})
            
            # Update active strategies and allocations
            self.active_strategies = strategy_config.copy()
            self.strategy_allocations = strategy_config.copy()
            
            self.logger.info(f"Active strategies updated for {regime_key}: {list(self.active_strategies.keys())}")
            self.rebalance_count += 1
            self.last_rebalance_time = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating active strategies: {e}")
    
    def get_active_strategies(self) -> List[str]:
        """Get list of currently active strategy names"""
        return list(self.active_strategies.keys())
    
    def get_strategy_allocation(self, strategy_name: str) -> float:
        """Get capital allocation for a specific strategy"""
        return self.strategy_allocations.get(strategy_name, 0.0)
    
    def get_current_regime(self) -> Dict[str, Any]:
        """Get current market regime information"""
        return {
            'regime': self.current_regime.value,
            'confidence': self.regime_confidence,
            'last_update': self.last_regime_update.isoformat() if self.last_regime_update != datetime.min else None,
            'active_strategies': self.active_strategies,
            'strategy_allocations': self.strategy_allocations
        }
    
    def get_regime_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent regime history"""
        history = self.regime_history[-limit:] if limit > 0 else self.regime_history
        return [
            {
                'regime': entry['regime'].value,
                'confidence': entry['confidence'],
                'timestamp': entry['timestamp'].isoformat()
            }
            for entry in history
        ]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get strategy router performance summary"""
        return {
            'current_regime': self.current_regime.value,
            'regime_confidence': self.regime_confidence,
            'rebalance_count': self.rebalance_count,
            'last_rebalance': self.last_rebalance_time.isoformat() if self.last_rebalance_time != datetime.min else None,
            'active_strategies_count': len(self.active_strategies),
            'regime_history_length': len(self.regime_history),
            'ml_components_available': self.ml_components is not None,
            'market_analyzer_available': self.market_analyzer is not None
        }
    
    def is_strategy_active(self, strategy_name: str) -> bool:
        """Check if a strategy is currently active"""
        return strategy_name in self.active_strategies
    
    def should_rebalance(self, min_confidence_change: float = 0.2) -> bool:
        """Check if rebalancing is needed"""
        try:
            # Time-based rebalancing
            time_since_last = datetime.now() - self.last_rebalance_time
            if time_since_last.total_seconds() > 3600:  # 1 hour
                return True
            
            # Confidence-based rebalancing
            if self.regime_confidence < 0.5:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking rebalance need: {e}")
            return False