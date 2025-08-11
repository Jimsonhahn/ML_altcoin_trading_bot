#!/usr/bin/env python3
"""
Social Sentiment Analysis for High-Risk Trading
===============================================

Analyzes social media sentiment for early entry signals:
- Twitter/X sentiment tracking
- Reddit momentum analysis
- Telegram channel monitoring
- News sentiment integration
- Real-time sentiment scoring
"""

import asyncio
import logging
import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import aiohttp
import hashlib
import time

logger = logging.getLogger(__name__)

@dataclass
class SentimentSignal:
    """Social sentiment signal"""
    symbol: str
    timestamp: datetime
    sentiment_score: float  # -1.0 to 1.0
    confidence: float      # 0.0 to 1.0
    source: str           # twitter, reddit, telegram, news
    volume_score: float   # Mention volume
    momentum_score: float # Rate of change
    key_mentions: List[str]
    metadata: Dict[str, Any]

@dataclass
class SentimentProfile:
    """Historical sentiment profile"""
    symbol: str
    source: str
    sentiment_history: deque
    volume_history: deque
    last_update: datetime
    baseline_sentiment: float
    baseline_volume: float

class SocialSentimentAnalyzer:
    """
    Advanced social sentiment analysis for crypto trading
    
    Features:
    - Multi-platform sentiment tracking
    - Real-time momentum detection
    - Anomaly-based signal generation
    - Configurable sentiment sources
    - Historical baseline tracking
    """
    
    def __init__(self, 
                 sources: List[str] = ['twitter', 'reddit'],
                 update_interval: int = 300,  # 5 minutes
                 history_periods: int = 288,  # 24 hours of 5-min data
                 sentiment_threshold: float = 0.3,
                 momentum_threshold: float = 2.0):
        
        self.sources = sources
        self.update_interval = update_interval
        self.history_periods = history_periods
        self.sentiment_threshold = sentiment_threshold
        self.momentum_threshold = momentum_threshold
        
        # Sentiment profiles by symbol and source
        self.sentiment_profiles: Dict[str, Dict[str, SentimentProfile]] = defaultdict(dict)
        
        # Recent signals tracking
        self.recent_signals: Dict[str, List[SentimentSignal]] = defaultdict(list)
        
        # API session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Sentiment keywords and patterns
        self.bullish_keywords = {
            'moon', 'rocket', 'pump', 'bullish', 'buy', 'hodl', 'diamond', 'hands',
            'breakout', 'surge', 'rally', 'explosion', 'massive', 'gains', 'profit',
            'accumulate', 'undervalued', 'gem', 'hidden', 'potential', 'explosive'
        }
        
        self.bearish_keywords = {
            'dump', 'crash', 'bearish', 'sell', 'short', 'dead', 'scam', 'rug',
            'collapse', 'plummet', 'disaster', 'avoid', 'warning', 'danger',
            'overvalued', 'bubble', 'panic', 'fear', 'liquidation', 'exit'
        }
        
        # Intensity multipliers
        self.intensity_words = {
            'massive': 2.0, 'huge': 1.8, 'enormous': 1.8, 'insane': 1.7,
            'crazy': 1.5, 'major': 1.4, 'big': 1.3, 'strong': 1.2,
            'small': 0.8, 'minor': 0.7, 'weak': 0.6, 'tiny': 0.5
        }
        
        logger.info(f"📱 Social Sentiment Analyzer initialized")
        logger.info(f"🎯 Sources: {', '.join(self.sources)}")
        logger.info(f"⏰ Update interval: {self.update_interval}s")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Mozilla/5.0 (compatible; CryptoSentimentBot/1.0)'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def analyze_sentiment(self, symbols: List[str]) -> List[SentimentSignal]:
        """
        Analyze sentiment across all sources for given symbols
        
        Args:
            symbols: List of crypto symbols to analyze
            
        Returns:
            List of sentiment signals
        """
        all_signals = []
        
        for symbol in symbols:
            for source in self.sources:
                try:
                    signal = await self._analyze_symbol_source(symbol, source)
                    if signal and self._is_significant_signal(signal):
                        all_signals.append(signal)
                        
                        # Store in recent signals
                        self.recent_signals[symbol].append(signal)
                        self._cleanup_old_signals(symbol)
                        
                        logger.info(f"📱 Sentiment signal: {symbol} {source} "
                                  f"sentiment={signal.sentiment_score:.2f} "
                                  f"momentum={signal.momentum_score:.1f}x")
                
                except Exception as e:
                    logger.error(f"Error analyzing {symbol} {source}: {e}")
                    continue
        
        # Sort by combined score (sentiment * momentum * confidence)
        all_signals.sort(key=lambda x: abs(x.sentiment_score) * x.momentum_score * x.confidence, reverse=True)
        
        return all_signals
    
    async def _analyze_symbol_source(self, symbol: str, source: str) -> Optional[SentimentSignal]:
        """Analyze sentiment for specific symbol and source"""
        
        if source == 'twitter':
            return await self._analyze_twitter_sentiment(symbol)
        elif source == 'reddit':
            return await self._analyze_reddit_sentiment(symbol)
        elif source == 'telegram':
            return await self._analyze_telegram_sentiment(symbol)
        elif source == 'news':
            return await self._analyze_news_sentiment(symbol)
        else:
            logger.warning(f"Unknown sentiment source: {source}")
            return None
    
    async def _analyze_twitter_sentiment(self, symbol: str) -> Optional[SentimentSignal]:
        """Analyze Twitter/X sentiment (mock implementation for demo)"""
        
        # In production, this would use Twitter API v2 or web scraping
        # For demo, we'll simulate sentiment data
        
        try:
            # Simulate API call delay
            await asyncio.sleep(0.1)
            
            # Mock sentiment analysis
            import random
            random.seed(int(time.time() / 300) + hash(symbol))  # Consistent per 5-min window
            
            # Simulate sentiment data
            mentions_count = random.randint(10, 500)
            bullish_ratio = random.uniform(0.2, 0.8)
            bearish_ratio = 1.0 - bullish_ratio
            
            # Calculate sentiment score
            sentiment_score = (bullish_ratio - bearish_ratio) * random.uniform(0.5, 1.0)
            
            # Volume score based on mentions
            volume_score = min(mentions_count / 100.0, 5.0)  # Cap at 5x
            
            # Calculate momentum vs historical
            momentum_score = await self._calculate_momentum(symbol, 'twitter', volume_score)
            
            # Confidence based on volume and consistency
            confidence = min((mentions_count / 50.0) * 0.5 + 0.3, 1.0)
            
            # Mock key mentions
            key_mentions = [
                f"#{symbol} trending with {mentions_count} mentions",
                f"Bullish: {bullish_ratio:.1%}, Bearish: {bearish_ratio:.1%}"
            ]
            
            return SentimentSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                sentiment_score=sentiment_score,
                confidence=confidence,
                source='twitter',
                volume_score=volume_score,
                momentum_score=momentum_score,
                key_mentions=key_mentions,
                metadata={
                    'mentions_count': mentions_count,
                    'bullish_ratio': bullish_ratio,
                    'bearish_ratio': bearish_ratio,
                    'api_mock': True
                }
            )
            
        except Exception as e:
            logger.error(f"Twitter sentiment analysis error: {e}")
            return None
    
    async def _analyze_reddit_sentiment(self, symbol: str) -> Optional[SentimentSignal]:
        """Analyze Reddit sentiment"""
        
        try:
            # In production, use Reddit API or PRAW
            # For demo, simulate Reddit data
            
            await asyncio.sleep(0.1)
            
            import random
            random.seed(int(time.time() / 600) + hash(symbol) + 1)  # Different seed
            
            # Simulate Reddit posts/comments
            posts_count = random.randint(5, 100)
            upvote_ratio = random.uniform(0.4, 0.9)
            
            # Reddit tends to be more analytical
            sentiment_score = (upvote_ratio - 0.5) * 2 * random.uniform(0.6, 1.0)
            
            volume_score = min(posts_count / 30.0, 4.0)
            momentum_score = await self._calculate_momentum(symbol, 'reddit', volume_score)
            
            # Reddit has higher confidence due to discussion quality
            confidence = min((posts_count / 20.0) * 0.4 + 0.4, 1.0)
            
            key_mentions = [
                f"r/CryptoCurrency: {posts_count} posts about {symbol}",
                f"Average upvote ratio: {upvote_ratio:.1%}"
            ]
            
            return SentimentSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                sentiment_score=sentiment_score,
                confidence=confidence,
                source='reddit',
                volume_score=volume_score,
                momentum_score=momentum_score,
                key_mentions=key_mentions,
                metadata={
                    'posts_count': posts_count,
                    'upvote_ratio': upvote_ratio,
                    'api_mock': True
                }
            )
            
        except Exception as e:
            logger.error(f"Reddit sentiment analysis error: {e}")
            return None
    
    async def _analyze_telegram_sentiment(self, symbol: str) -> Optional[SentimentSignal]:
        """Analyze Telegram sentiment (placeholder)"""
        # Telegram API integration would go here
        # Requires more complex setup with bot tokens
        return None
    
    async def _analyze_news_sentiment(self, symbol: str) -> Optional[SentimentSignal]:
        """Analyze news sentiment (placeholder)"""
        # News API integration would go here
        # Could use NewsAPI, Google News, or crypto-specific news sources
        return None
    
    async def _calculate_momentum(self, symbol: str, source: str, current_volume: float) -> float:
        """Calculate momentum score vs historical baseline"""
        
        profile_key = f"{symbol}_{source}"
        
        if source not in self.sentiment_profiles[symbol]:
            # Initialize profile
            self.sentiment_profiles[symbol][source] = SentimentProfile(
                symbol=symbol,
                source=source,
                sentiment_history=deque(maxlen=self.history_periods),
                volume_history=deque(maxlen=self.history_periods),
                last_update=datetime.now(),
                baseline_sentiment=0.0,
                baseline_volume=1.0
            )
        
        profile = self.sentiment_profiles[symbol][source]
        
        # Add current volume to history
        profile.volume_history.append(current_volume)
        profile.last_update = datetime.now()
        
        # Calculate baseline (average of historical data)
        if len(profile.volume_history) >= 10:
            profile.baseline_volume = sum(profile.volume_history) / len(profile.volume_history)
            momentum = current_volume / max(profile.baseline_volume, 0.1)
        else:
            momentum = 1.0  # No momentum if insufficient history
        
        return momentum
    
    def _is_significant_signal(self, signal: SentimentSignal) -> bool:
        """Check if sentiment signal is significant enough to act on"""
        
        # Minimum sentiment threshold
        if abs(signal.sentiment_score) < self.sentiment_threshold:
            return False
        
        # Minimum momentum threshold
        if signal.momentum_score < self.momentum_threshold:
            return False
        
        # Minimum confidence
        if signal.confidence < 0.5:
            return False
        
        return True
    
    def _cleanup_old_signals(self, symbol: str, max_age_hours: int = 12):
        """Remove old signals from tracking"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        self.recent_signals[symbol] = [
            signal for signal in self.recent_signals[symbol]
            if signal.timestamp > cutoff_time
        ]
    
    def get_sentiment_summary(self, symbol: str, hours: int = 6) -> Dict[str, Any]:
        """Get sentiment summary for symbol"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_signals = [
            signal for signal in self.recent_signals.get(symbol, [])
            if signal.timestamp > cutoff_time
        ]
        
        if not recent_signals:
            return {
                'symbol': symbol,
                'overall_sentiment': 0.0,
                'confidence': 0.0,
                'signal_count': 0,
                'sources': [],
                'momentum': 1.0
            }
        
        # Calculate weighted averages
        total_weight = sum(signal.confidence for signal in recent_signals)
        
        if total_weight > 0:
            weighted_sentiment = sum(
                signal.sentiment_score * signal.confidence 
                for signal in recent_signals
            ) / total_weight
            
            avg_momentum = sum(signal.momentum_score for signal in recent_signals) / len(recent_signals)
            avg_confidence = total_weight / len(recent_signals)
        else:
            weighted_sentiment = 0.0
            avg_momentum = 1.0
            avg_confidence = 0.0
        
        return {
            'symbol': symbol,
            'overall_sentiment': weighted_sentiment,
            'confidence': avg_confidence,
            'signal_count': len(recent_signals),
            'sources': list(set(signal.source for signal in recent_signals)),
            'momentum': avg_momentum,
            'recent_signals': recent_signals[-5:]  # Last 5 signals
        }
    
    def get_top_sentiment_candidates(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top sentiment-driven candidates"""
        
        all_summaries = []
        
        for symbol in self.recent_signals.keys():
            summary = self.get_sentiment_summary(symbol, hours=2)  # Very recent
            
            if summary['signal_count'] > 0:
                # Calculate combined score
                score = (
                    abs(summary['overall_sentiment']) * 
                    summary['confidence'] * 
                    summary['momentum'] *
                    min(summary['signal_count'] / 2.0, 2.0)  # Signal frequency bonus
                )
                
                summary['score'] = score
                all_summaries.append(summary)
        
        # Sort by score
        all_summaries.sort(key=lambda x: x['score'], reverse=True)
        
        return all_summaries[:limit]
    
    async def monitor_sentiment(self, symbols: List[str], callback=None) -> None:
        """Continuously monitor sentiment for symbols"""
        logger.info(f"📱 Starting sentiment monitoring for {len(symbols)} symbols")
        
        while True:
            try:
                signals = await self.analyze_sentiment(symbols)
                
                for signal in signals:
                    if callback:
                        await callback(signal)
                    else:
                        logger.info(f"📱 Sentiment Alert: {signal.symbol} "
                                  f"sentiment={signal.sentiment_score:+.2f} "
                                  f"momentum={signal.momentum_score:.1f}x "
                                  f"({signal.source})")
                
                # Wait before next analysis
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in sentiment monitoring: {e}")
                await asyncio.sleep(60)  # Shorter retry interval

# Utility function for high-risk strategy integration
def create_sentiment_analyzer(config: Dict[str, Any] = None) -> SocialSentimentAnalyzer:
    """Create sentiment analyzer with configuration"""
    if config is None:
        config = {}
    
    return SocialSentimentAnalyzer(
        sources=config.get('sources', ['twitter', 'reddit']),
        update_interval=config.get('update_interval', 300),
        history_periods=config.get('history_periods', 288),
        sentiment_threshold=config.get('sentiment_threshold', 0.3),
        momentum_threshold=config.get('momentum_threshold', 2.0)
    )