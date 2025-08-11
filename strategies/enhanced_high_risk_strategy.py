#!/usr/bin/env python3
"""
Enhanced High-Risk Daily Strategy
=================================

Integrated high-risk strategy with all advanced components:
- Live social sentiment analysis (Twitter/Reddit)
- ML-enhanced signal prediction
- Multi-exchange arbitrage detection
- Breaking news integration
- Advanced configuration management

This enhanced version combines all signal sources for improved accuracy.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all enhanced components
from core.live_social_apis import create_live_sentiment_analyzer
from core.ml_signal_enhancement import create_ml_predictor, generate_synthetic_training_data
from core.multi_exchange_arbitrage import create_arbitrage_detector
from core.breaking_news_integration import create_news_monitor
from core.enhanced_config_system import create_config_manager, ConfigProfile

# Import existing components
from core.risk_limiter import DailyRiskLimiter
from core.volume_detector import VolumeSpike
from utils.error_handler import safe_execute
from utils.notifier_clean import HighRiskLogger

logger = logging.getLogger(__name__)

@dataclass
class EnhancedSignalData:
    """Enhanced signal data with all sources"""
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    timestamp: datetime
    
    # Traditional signals
    volume_spike_ratio: float = 0.0
    price_momentum: float = 0.0
    technical_score: float = 0.0
    
    # Enhanced signals
    ml_prediction: float = 0.0
    ml_confidence: float = 0.0
    social_sentiment: float = 0.0
    social_momentum: float = 0.0
    news_sentiment: float = 0.0
    news_impact: float = 0.0
    arbitrage_opportunity: float = 0.0
    
    # Meta information
    signal_sources: List[str] = None
    reasoning: str = ""
    risk_score: float = 0.0
    expected_return: float = 0.0
    
    def __post_init__(self):
        if self.signal_sources is None:
            self.signal_sources = []

class EnhancedHighRiskStrategy:
    """
    Enhanced high-risk trading strategy with all advanced components
    
    Integrates multiple signal sources for improved accuracy:
    - Volume spike detection
    - Social sentiment analysis
    - ML prediction models
    - Breaking news analysis
    - Multi-exchange arbitrage
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize configuration manager
        self.config_manager = create_config_manager()
        self.active_config = self._load_active_configuration()
        
        # Core components
        self.risk_limiter = DailyRiskLimiter(self.active_config.daily_budget)
        self.hr_logger = HighRiskLogger()
        
        # Enhanced components (initialized as None, created when needed)
        self.sentiment_analyzer = None
        self.ml_predictor = None
        self.arbitrage_detector = None
        self.news_monitor = None
        
        # Strategy state
        self.active_positions = {}
        self.daily_stats = {
            'signals_generated': 0,
            'trades_executed': 0,
            'total_pnl': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0
        }
        
        # Performance tracking
        self.signal_performance = {
            'volume_spike': {'count': 0, 'success': 0, 'avg_return': 0.0},
            'social_sentiment': {'count': 0, 'success': 0, 'avg_return': 0.0},
            'ml_prediction': {'count': 0, 'success': 0, 'avg_return': 0.0},
            'news_analysis': {'count': 0, 'success': 0, 'avg_return': 0.0},
            'arbitrage': {'count': 0, 'success': 0, 'avg_return': 0.0}
        }
        
        logger.info("🔥 Enhanced High-Risk Strategy initialized")
        self.hr_logger.log_event("STRATEGY_INIT", {"version": "enhanced", "config": self.active_config.name})
    
    def _load_active_configuration(self) -> ConfigProfile:
        """Load active configuration profile"""
        
        active_profile = self.config_manager.get_active_profile()
        
        if not active_profile:
            # Create default aggressive profile
            logger.info("⚙️ No active profile found, creating default aggressive profile")
            active_profile = self.config_manager.create_profile_template(
                name="enhanced_aggressive",
                risk_level="aggressive",
                environment="development"
            )
            self.config_manager.set_active_profile(active_profile.name)
        
        logger.info(f"⚙️ Loaded configuration profile: {active_profile.name}")
        return active_profile
    
    async def initialize_enhanced_components(self):
        """Initialize enhanced components asynchronously"""
        
        try:
            # Initialize sentiment analyzer
            if self.active_config.social_sentiment_enabled:
                sentiment_config = {
                    'update_interval': 300,
                    'max_posts_per_platform': 50,
                    'sentiment_threshold': 0.3
                }
                self.sentiment_analyzer = create_live_sentiment_analyzer(sentiment_config)
                logger.info("📱 Social sentiment analyzer initialized")
            
            # Initialize ML predictor
            if self.active_config.ml_prediction_enabled:
                self.ml_predictor = create_ml_predictor()
                
                # Train with synthetic data if no models exist
                if not self.ml_predictor.is_trained:
                    logger.info("🤖 Training ML models with synthetic data...")
                    training_data = generate_synthetic_training_data(1000)
                    scores = self.ml_predictor.train_models(training_data)
                    logger.info(f"🎓 ML model training completed: {scores}")
            
            # Initialize arbitrage detector
            if self.active_config.arbitrage_enabled:
                arbitrage_config = {
                    'enabled_exchanges': ['binance', 'coinbase', 'kraken'],
                    'min_profit_percent': self.active_config.arbitrage_min_profit,
                    'min_profit_amount': 10.0,
                    'min_confidence': 0.7
                }
                self.arbitrage_detector = create_arbitrage_detector(arbitrage_config)
                logger.info("🔄 Arbitrage detector initialized")
            
            # Initialize news monitor
            if self.active_config.news_analysis_enabled:
                news_config = {
                    'aggregator': {
                        'enabled_sources': ['CoinDesk', 'CoinTelegraph']
                    },
                    'analyzer': {
                        'sentiment_threshold': 0.3,
                        'impact_threshold': 0.5,
                        'signal_confidence_threshold': 0.6
                    }
                }
                self.news_monitor = create_news_monitor(news_config)
                logger.info("📰 News monitor initialized")
            
            logger.info("✅ All enhanced components initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing enhanced components: {e}")
            # Continue with basic strategy if enhanced components fail
    
    async def calculate_enhanced_signal(self, symbol: str, market_data: pd.DataFrame, 
                                      current_price: float) -> Tuple[str, EnhancedSignalData]:
        """Calculate enhanced trading signal using all available sources"""
        
        signal_data = EnhancedSignalData(
            symbol=symbol,
            signal_type='HOLD',
            confidence=0.0,
            timestamp=datetime.now(),
            signal_sources=[],
            reasoning=""
        )
        
        # Collect signals from all sources
        signal_scores = []
        
        try:
            # 1. Volume spike detection (traditional)
            volume_score = await self._get_volume_signal(symbol, market_data, current_price, signal_data)
            if volume_score > 0:
                signal_scores.append(('volume_spike', volume_score, self.active_config.volume_spike_weight))
            
            # 2. Social sentiment analysis
            if self.sentiment_analyzer and self.active_config.social_sentiment_enabled:
                sentiment_score = await self._get_sentiment_signal(symbol, signal_data)
                if sentiment_score > 0:
                    signal_scores.append(('social_sentiment', sentiment_score, self.active_config.social_sentiment_weight))
            
            # 3. ML prediction
            if self.ml_predictor and self.active_config.ml_prediction_enabled:
                ml_score = await self._get_ml_signal(symbol, market_data, signal_data)
                if ml_score > 0:
                    signal_scores.append(('ml_prediction', ml_score, self.active_config.ml_prediction_weight))
            
            # 4. News analysis
            if self.news_monitor and self.active_config.news_analysis_enabled:
                news_score = await self._get_news_signal(symbol, signal_data)
                if news_score > 0:
                    signal_scores.append(('news_analysis', news_score, self.active_config.news_analysis_weight))
            
            # 5. Arbitrage opportunities
            if self.arbitrage_detector and self.active_config.arbitrage_enabled:
                arbitrage_score = await self._get_arbitrage_signal(symbol, signal_data)
                if arbitrage_score > 0:
                    signal_scores.append(('arbitrage', arbitrage_score, 0.1))  # Bonus weight
            
            # Combine all signals using weighted average
            if signal_scores:
                final_score, final_confidence = self._combine_signals(signal_scores)
                
                # Determine signal type
                if final_score > 0.6 and final_confidence > self.active_config.min_confidence_threshold:
                    signal_data.signal_type = 'BUY'
                    signal_data.confidence = final_confidence
                elif final_score < -0.6 and final_confidence > self.active_config.min_confidence_threshold:
                    signal_data.signal_type = 'SELL'
                    signal_data.confidence = final_confidence
                else:
                    signal_data.signal_type = 'HOLD'
                    signal_data.confidence = final_confidence
                
                # Generate reasoning
                signal_data.reasoning = self._generate_signal_reasoning(signal_scores, final_score)
                
                # Calculate risk and expected return
                signal_data.risk_score = self._calculate_signal_risk(signal_data)
                signal_data.expected_return = self._estimate_signal_return(signal_data, final_score)
            
            # Log signal generation
            self.daily_stats['signals_generated'] += 1
            self.hr_logger.log_signal(signal_data)
            
            return signal_data.signal_type, signal_data
            
        except Exception as e:
            logger.error(f"Error calculating enhanced signal for {symbol}: {e}")
            return 'HOLD', signal_data
    
    async def _get_volume_signal(self, symbol: str, market_data: pd.DataFrame, 
                               current_price: float, signal_data: EnhancedSignalData) -> float:
        """Get volume spike signal (traditional method)"""
        
        if len(market_data) < 24:
            return 0.0
        
        try:
            # Calculate volume spike
            current_volume = market_data['volume'].iloc[-1]
            avg_volume = market_data['volume'].iloc[-24:].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            signal_data.volume_spike_ratio = volume_ratio
            
            # Volume spike threshold
            if volume_ratio >= 3.0:  # 300% spike
                signal_data.signal_sources.append('volume_spike')
                self.signal_performance['volume_spike']['count'] += 1
                return 0.8
            elif volume_ratio >= 2.0:  # 200% spike
                signal_data.signal_sources.append('volume_spike')
                self.signal_performance['volume_spike']['count'] += 1
                return 0.6
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error in volume signal calculation: {e}")
            return 0.0
    
    async def _get_sentiment_signal(self, symbol: str, signal_data: EnhancedSignalData) -> float:
        """Get social sentiment signal"""
        
        try:
            # Analyze sentiment for this symbol
            snapshots = await self.sentiment_analyzer.analyze_sentiment([symbol])
            
            if not snapshots:
                return 0.0
            
            # Use the best sentiment signal
            best_snapshot = max(snapshots, key=lambda x: x.confidence)
            
            signal_data.social_sentiment = best_snapshot.average_sentiment
            signal_data.social_momentum = best_snapshot.sentiment_momentum
            
            # Convert sentiment to signal score
            sentiment_strength = abs(best_snapshot.average_sentiment)
            if sentiment_strength > 0.3 and best_snapshot.confidence > 0.7:
                signal_data.signal_sources.append(f'social_sentiment_{best_snapshot.platform}')
                self.signal_performance['social_sentiment']['count'] += 1
                
                # Positive sentiment = buy signal
                return sentiment_strength * best_snapshot.confidence * (1 if best_snapshot.average_sentiment > 0 else -1)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error in sentiment signal calculation: {e}")
            return 0.0
    
    async def _get_ml_signal(self, symbol: str, market_data: pd.DataFrame, 
                           signal_data: EnhancedSignalData) -> float:
        """Get ML prediction signal"""
        
        try:
            # Get ML prediction
            prediction = self.ml_predictor.predict_signal(market_data, symbol=symbol)
            
            signal_data.ml_prediction = prediction.predicted_signal
            signal_data.ml_confidence = prediction.confidence
            signal_data.expected_return = prediction.expected_return
            
            # Convert ML prediction to signal score
            if prediction.confidence > 0.7:
                signal_data.signal_sources.append('ml_prediction')
                self.signal_performance['ml_prediction']['count'] += 1
                
                # ML signal: 0=HOLD, 1=BUY, 2=SELL
                if prediction.predicted_signal == 1:  # BUY
                    return prediction.confidence
                elif prediction.predicted_signal == 2:  # SELL
                    return -prediction.confidence
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error in ML signal calculation: {e}")
            return 0.0
    
    async def _get_news_signal(self, symbol: str, signal_data: EnhancedSignalData) -> float:
        """Get breaking news signal"""
        
        try:
            # Get latest news signals
            news_signals = await self.news_monitor.start_monitoring([symbol])
            
            if not news_signals:
                return 0.0
            
            # Use the highest impact news signal
            best_signal = max(news_signals, key=lambda x: x.impact_score * x.confidence)
            
            signal_data.news_sentiment = 1.0 if best_signal.signal_type == 'BUY' else -1.0
            signal_data.news_impact = best_signal.impact_score
            
            # High impact news with good confidence
            if best_signal.impact_score > 0.6 and best_signal.confidence > 0.7:
                signal_data.signal_sources.append(f'news_{best_signal.urgency}')
                self.signal_performance['news_analysis']['count'] += 1
                
                signal_strength = best_signal.impact_score * best_signal.confidence
                return signal_strength if best_signal.signal_type == 'BUY' else -signal_strength
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error in news signal calculation: {e}")
            return 0.0
    
    async def _get_arbitrage_signal(self, symbol: str, signal_data: EnhancedSignalData) -> float:
        """Get arbitrage opportunity signal"""
        
        try:
            # Detect arbitrage opportunities
            opportunities = await self.arbitrage_detector.detect_arbitrage_opportunities([symbol])
            
            if not opportunities:
                return 0.0
            
            # Use the best arbitrage opportunity
            best_opportunity = max(opportunities, key=lambda x: x.net_profit)
            
            signal_data.arbitrage_opportunity = best_opportunity.profit_percent
            
            # Good arbitrage opportunity
            if best_opportunity.net_profit > 15.0 and best_opportunity.confidence > 0.8:
                signal_data.signal_sources.append('arbitrage')
                self.signal_performance['arbitrage']['count'] += 1
                
                # Arbitrage is typically a buy signal (buy low, sell high)
                return best_opportunity.confidence * 0.5  # Lower weight for arbitrage
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error in arbitrage signal calculation: {e}")
            return 0.0
    
    def _combine_signals(self, signal_scores: List[Tuple[str, float, float]]) -> Tuple[float, float]:
        """Combine multiple signals using weighted average"""
        
        total_weighted_score = 0.0
        total_weight = 0.0
        confidence_scores = []
        
        for source, score, weight in signal_scores:
            total_weighted_score += score * weight
            total_weight += weight
            confidence_scores.append(abs(score))  # Confidence based on signal strength
        
        # Normalize score
        if total_weight > 0:
            final_score = total_weighted_score / total_weight
        else:
            final_score = 0.0
        
        # Calculate confidence as average of signal strengths
        if confidence_scores:
            final_confidence = np.mean(confidence_scores)
        else:
            final_confidence = 0.0
        
        # Boost confidence if multiple sources agree
        if len(signal_scores) >= 3:
            final_confidence *= 1.2
        elif len(signal_scores) >= 2:
            final_confidence *= 1.1
        
        final_confidence = min(final_confidence, 1.0)
        
        return final_score, final_confidence
    
    def _generate_signal_reasoning(self, signal_scores: List[Tuple[str, float, float]], 
                                 final_score: float) -> str:
        """Generate human-readable reasoning for the signal"""
        
        if not signal_scores:
            return "No significant signals detected"
        
        # Sort by signal strength
        sorted_signals = sorted(signal_scores, key=lambda x: abs(x[1]), reverse=True)
        
        direction = "BULLISH" if final_score > 0 else "BEARISH"
        strength = "STRONG" if abs(final_score) > 0.7 else "MODERATE" if abs(final_score) > 0.4 else "WEAK"
        
        reasoning = f"{strength} {direction} signal from {len(signal_scores)} sources: "
        
        source_descriptions = []
        for source, score, weight in sorted_signals[:3]:  # Top 3 sources
            source_name = source.replace('_', ' ').title()
            strength_desc = "Strong" if abs(score) > 0.7 else "Moderate" if abs(score) > 0.4 else "Weak"
            source_descriptions.append(f"{source_name} ({strength_desc})")
        
        reasoning += ", ".join(source_descriptions)
        
        return reasoning
    
    def _calculate_signal_risk(self, signal_data: EnhancedSignalData) -> float:
        """Calculate risk score for the signal (0=low risk, 1=high risk)"""
        
        risk_score = 0.3  # Base risk
        
        # More sources = lower risk
        if len(signal_data.signal_sources) >= 3:
            risk_score -= 0.1
        elif len(signal_data.signal_sources) >= 2:
            risk_score -= 0.05
        
        # High volatility increases risk
        if signal_data.volume_spike_ratio > 3.0:
            risk_score += 0.2
        
        # Strong news impact can be risky
        if signal_data.news_impact > 0.8:
            risk_score += 0.1
        
        # ML confidence affects risk
        if signal_data.ml_confidence > 0.8:
            risk_score -= 0.1
        elif signal_data.ml_confidence < 0.6:
            risk_score += 0.1
        
        return max(0.0, min(1.0, risk_score))
    
    def _estimate_signal_return(self, signal_data: EnhancedSignalData, final_score: float) -> float:
        """Estimate expected return from the signal"""
        
        base_return = abs(final_score) * 0.1  # Base 10% for strong signals
        
        # Adjust based on signal sources
        if 'volume_spike' in signal_data.signal_sources:
            base_return += 0.02  # Volume spikes often lead to 2% moves
        
        if 'social_sentiment' in [s for s in signal_data.signal_sources if 'social' in s]:
            base_return += signal_data.social_momentum * 0.01
        
        if 'ml_prediction' in signal_data.signal_sources:
            base_return += signal_data.expected_return
        
        if 'news' in [s for s in signal_data.signal_sources if 'news' in s]:
            base_return += signal_data.news_impact * 0.03
        
        if 'arbitrage' in signal_data.signal_sources:
            base_return += signal_data.arbitrage_opportunity * 0.01
        
        # Apply direction
        return base_return if final_score > 0 else -base_return
    
    async def execute_enhanced_trade(self, signal: str, symbol: str, current_price: float, 
                                   signal_data: EnhancedSignalData) -> bool:
        """Execute trade with enhanced position management"""
        
        try:
            # Check risk limits
            position_size = min(15.0, self.risk_limiter.remaining_budget * 0.8)
            position_size *= signal_data.confidence  # Scale by confidence
            
            can_trade, reason = self.risk_limiter.can_trade(position_size)
            if not can_trade:
                self.hr_logger.log_event("TRADE_REJECTED", {
                    "symbol": symbol,
                    "reason": reason,
                    "signal_confidence": signal_data.confidence
                })
                return False
            
            # Reserve budget
            if not self.risk_limiter.reserve_budget(position_size, f"TRADE_{symbol}"):
                return False
            
            # Create position with enhanced data
            position_id = f"{symbol}_{datetime.now().strftime('%H%M%S')}"
            
            position = {
                'id': position_id,
                'symbol': symbol,
                'side': signal,
                'entry_price': current_price,
                'quantity': position_size / current_price,
                'position_size': position_size,
                'entry_time': datetime.now(),
                'signal_data': signal_data,
                'stop_loss': current_price * (0.85 if signal == 'BUY' else 1.15),
                'take_profit': current_price * (1.5 if signal == 'BUY' else 0.5),
                'timeout': datetime.now() + timedelta(hours=self.active_config.position_timeout_hours)
            }
            
            self.active_positions[position_id] = position
            self.daily_stats['trades_executed'] += 1
            
            # Log successful trade
            self.hr_logger.log_trade(position, "OPENED")
            
            logger.info(f"✅ Enhanced trade executed: {symbol} {signal} "
                       f"(confidence: {signal_data.confidence:.2f}, "
                       f"sources: {len(signal_data.signal_sources)})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing enhanced trade: {e}")
            return False
    
    async def manage_enhanced_positions(self, current_prices: Dict[str, float]):
        """Enhanced position management with ML-based exit signals"""
        
        positions_to_close = []
        
        for position_id, position in self.active_positions.items():
            symbol = position['symbol']
            current_price = current_prices.get(symbol)
            
            if current_price is None:
                continue
            
            # Calculate current P&L
            if position['side'] == 'BUY':
                unrealized_pnl = (current_price - position['entry_price']) * position['quantity']
            else:
                unrealized_pnl = (position['entry_price'] - current_price) * position['quantity']
            
            # Enhanced exit logic
            should_close = False
            close_reason = ''
            
            # Traditional exit conditions
            if datetime.now() >= position['timeout']:
                should_close = True
                close_reason = 'timeout'
            elif ((position['side'] == 'BUY' and current_price <= position['stop_loss']) or
                  (position['side'] == 'SELL' and current_price >= position['stop_loss'])):
                should_close = True
                close_reason = 'stop_loss'
            elif ((position['side'] == 'BUY' and current_price >= position['take_profit']) or
                  (position['side'] == 'SELL' and current_price <= position['take_profit'])):
                should_close = True
                close_reason = 'take_profit'
            
            # Enhanced ML-based exit (if available)
            elif self.ml_predictor and unrealized_pnl > position['position_size'] * 0.1:  # 10% profit
                try:
                    # Create dummy market data for exit prediction
                    dummy_data = pd.DataFrame({
                        'close': [current_price] * 50,
                        'volume': [1000000] * 50
                    }, index=pd.date_range(start='2024-01-01', periods=50, freq='1H'))
                    
                    exit_prediction = self.ml_predictor.predict_signal(dummy_data, symbol=symbol)
                    
                    # Exit if ML suggests opposite direction with high confidence
                    original_signal = position['signal_data'].ml_prediction
                    if (exit_prediction.confidence > 0.8 and 
                        ((original_signal == 1 and exit_prediction.predicted_signal == 2) or
                         (original_signal == 2 and exit_prediction.predicted_signal == 1))):
                        should_close = True
                        close_reason = 'ml_exit_signal'
                        
                except Exception as e:
                    logger.warning(f"ML exit prediction failed: {e}")
            
            if should_close:
                positions_to_close.append((position_id, position, current_price, close_reason))
        
        # Close positions
        for position_id, position, exit_price, reason in positions_to_close:
            await self._close_enhanced_position(position_id, position, exit_price, reason)
    
    async def _close_enhanced_position(self, position_id: str, position: Dict, 
                                     exit_price: float, reason: str):
        """Close position with enhanced tracking"""
        
        try:
            # Calculate P&L
            if position['side'] == 'BUY':
                pnl = (exit_price - position['entry_price']) * position['quantity']
            else:
                pnl = (position['entry_price'] - exit_price) * position['quantity']
            
            # Update performance tracking
            signal_sources = position['signal_data'].signal_sources
            for source in signal_sources:
                source_key = source.split('_')[0]  # Get base source name
                if source_key in self.signal_performance:
                    self.signal_performance[source_key]['success'] += 1 if pnl > 0 else 0
                    # Update average return (simple running average)
                    current_avg = self.signal_performance[source_key]['avg_return']
                    count = self.signal_performance[source_key]['count']
                    new_avg = (current_avg * (count - 1) + (pnl / position['position_size'])) / count
                    self.signal_performance[source_key]['avg_return'] = new_avg
            
            # Release budget
            self.risk_limiter.release_budget(position['position_size'], pnl)
            
            # Update daily stats
            self.daily_stats['total_pnl'] += pnl
            if pnl > self.daily_stats['best_trade']:
                self.daily_stats['best_trade'] = pnl
            if pnl < self.daily_stats['worst_trade']:
                self.daily_stats['worst_trade'] = pnl
            
            # Log position close
            position['exit_price'] = exit_price
            position['exit_time'] = datetime.now()
            position['pnl'] = pnl
            position['close_reason'] = reason
            
            self.hr_logger.log_trade(position, "CLOSED")
            
            # Remove from active positions
            del self.active_positions[position_id]
            
            logger.info(f"💰 Position closed: {position['symbol']} "
                       f"P&L: {pnl:+.2f}€ ({reason})")
            
        except Exception as e:
            logger.error(f"Error closing enhanced position: {e}")
    
    def get_enhanced_strategy_info(self) -> Dict[str, Any]:
        """Get comprehensive strategy information"""
        
        return {
            'name': 'Enhanced High-Risk Daily Strategy',
            'version': '2.0',
            'config_profile': self.active_config.name,
            'risk_level': self.active_config.risk_level,
            'daily_budget': self.active_config.daily_budget,
            'budget_used': self.active_config.daily_budget - self.risk_limiter.remaining_budget,
            'remaining_budget': self.risk_limiter.remaining_budget,
            'active_positions': len(self.active_positions),
            'max_positions': self.active_config.max_positions,
            
            # Enhanced components status
            'enhanced_components': {
                'social_sentiment': self.sentiment_analyzer is not None,
                'ml_prediction': self.ml_predictor is not None and self.ml_predictor.is_trained,
                'arbitrage_detection': self.arbitrage_detector is not None,
                'news_analysis': self.news_monitor is not None
            },
            
            # Daily statistics
            'daily_stats': self.daily_stats.copy(),
            
            # Signal source performance
            'signal_performance': {
                source: {
                    'count': data['count'],
                    'success_rate': (data['success'] / max(data['count'], 1)) * 100,
                    'avg_return_percent': data['avg_return'] * 100
                }
                for source, data in self.signal_performance.items()
                if data['count'] > 0
            }
        }
    
    def get_enhanced_daily_summary(self) -> str:
        """Get enhanced daily summary"""
        
        info = self.get_enhanced_strategy_info()
        
        summary = f"""
🔥 ENHANCED HIGH-RISK STRATEGY - Daily Summary
=============================================

📋 Configuration: {info['config_profile']} ({info['risk_level']})
💰 Budget: {info['budget_used']:.2f}€ / {info['daily_budget']:.2f}€ used
📊 Positions: {info['active_positions']} / {info['max_positions']} active

🎯 Performance Today:
   • Signals Generated: {info['daily_stats']['signals_generated']}
   • Trades Executed: {info['daily_stats']['trades_executed']}
   • Total P&L: {info['daily_stats']['total_pnl']:+.2f}€
   • Best Trade: {info['daily_stats']['best_trade']:+.2f}€
   • Worst Trade: {info['daily_stats']['worst_trade']:+.2f}€

🚀 Enhanced Components:
   • Social Sentiment: {'✅' if info['enhanced_components']['social_sentiment'] else '❌'}
   • ML Prediction: {'✅' if info['enhanced_components']['ml_prediction'] else '❌'}
   • Arbitrage Detection: {'✅' if info['enhanced_components']['arbitrage_detection'] else '❌'}
   • News Analysis: {'✅' if info['enhanced_components']['news_analysis'] else '❌'}

📊 Signal Source Performance:"""
        
        for source, perf in info['signal_performance'].items():
            summary += f"\n   • {source.replace('_', ' ').title()}: "
            summary += f"{perf['count']} signals, {perf['success_rate']:.1f}% success, "
            summary += f"{perf['avg_return_percent']:+.2f}% avg return"
        
        return summary

# Factory function
def create_enhanced_high_risk_strategy(config: Dict[str, Any] = None) -> EnhancedHighRiskStrategy:
    """Create enhanced high-risk strategy instance"""
    return EnhancedHighRiskStrategy(config)

# Test function
async def test_enhanced_strategy():
    """Test enhanced high-risk strategy"""
    
    print("🔥 Testing Enhanced High-Risk Strategy...")
    
    try:
        # Create strategy
        strategy = create_enhanced_high_risk_strategy()
        
        # Initialize enhanced components
        print("🚀 Initializing enhanced components...")
        await strategy.initialize_enhanced_components()
        
        # Show strategy info
        info = strategy.get_enhanced_strategy_info()
        print(f"\n📋 Strategy Info:")
        print(f"   Name: {info['name']}")
        print(f"   Config: {info['config_profile']}")
        print(f"   Budget: {info['daily_budget']}€")
        print(f"   Enhanced Components: {sum(info['enhanced_components'].values())}/4 active")
        
        # Test signal calculation
        print(f"\n🎯 Testing enhanced signal calculation...")
        
        # Create sample market data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.normal(45000, 1000, 100),
            'high': np.random.normal(45500, 1000, 100),
            'low': np.random.normal(44500, 1000, 100),
            'close': np.random.normal(45000, 1000, 100),
            'volume': np.random.normal(1000000, 300000, 100)
        })
        sample_data.set_index('timestamp', inplace=True)
        
        signal, signal_data = await strategy.calculate_enhanced_signal(
            'BTC/USDT', sample_data, 45000.0
        )
        
        print(f"   Signal: {signal}")
        print(f"   Confidence: {signal_data.confidence:.2f}")
        print(f"   Sources: {len(signal_data.signal_sources)} ({', '.join(signal_data.signal_sources)})")
        print(f"   Reasoning: {signal_data.reasoning}")
        print(f"   Expected Return: {signal_data.expected_return:+.2f}%")
        print(f"   Risk Score: {signal_data.risk_score:.2f}")
        
        # Test trade execution
        if signal in ['BUY', 'SELL']:
            print(f"\n💰 Testing trade execution...")
            success = await strategy.execute_enhanced_trade(signal, 'BTC/USDT', 45000.0, signal_data)
            print(f"   Trade executed: {success}")
            
            if success:
                print(f"   Active positions: {len(strategy.active_positions)}")
                
                # Simulate price movement and position management
                new_prices = {'BTC/USDT': 45000.0 * 1.1}  # 10% increase
                await strategy.manage_enhanced_positions(new_prices)
                print(f"   Positions after management: {len(strategy.active_positions)}")
        
        # Show daily summary
        print(f"\n📊 Enhanced Daily Summary:")
        summary = strategy.get_enhanced_daily_summary()
        print(summary)
        
        print(f"\n🎉 Enhanced strategy test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run test
    asyncio.run(test_enhanced_strategy())