#!/usr/bin/env python3
"""
Live Social Media APIs Integration
==================================

Real-time social media data collection for sentiment analysis:
- Twitter/X API v2 integration
- Reddit API via PRAW
- Telegram channel monitoring
- Real-time sentiment scoring
- Rate limiting and error handling
"""

import asyncio
import aiohttp
import logging
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import os
from collections import defaultdict, deque
import hashlib
import time

# Third-party imports
try:
    import tweepy
    TWITTER_AVAILABLE = True
except ImportError:
    TWITTER_AVAILABLE = False
    logging.warning("Tweepy not installed - Twitter integration disabled")

try:
    import praw
    REDDIT_AVAILABLE = True
except ImportError:
    REDDIT_AVAILABLE = False
    logging.warning("PRAW not installed - Reddit integration disabled")

logger = logging.getLogger(__name__)

@dataclass
class SocialPost:
    """Individual social media post"""
    id: str
    platform: str
    author: str
    content: str
    timestamp: datetime
    engagement: Dict[str, int]  # likes, retweets, comments, etc.
    sentiment_raw: float
    sentiment_processed: float
    confidence: float
    mentions: List[str]  # Crypto symbols mentioned
    metadata: Dict[str, Any]

@dataclass
class SentimentSnapshot:
    """Aggregated sentiment snapshot"""
    symbol: str
    timestamp: datetime
    platform: str
    posts_analyzed: int
    average_sentiment: float
    sentiment_momentum: float
    engagement_score: float
    confidence: float
    top_posts: List[SocialPost]
    trend_direction: str  # 'bullish', 'bearish', 'neutral'

class TwitterAPIClient:
    """
    Twitter/X API v2 client for crypto sentiment
    
    Requires Twitter API v2 Bearer Token
    """
    
    def __init__(self, bearer_token: str = None):
        self.bearer_token = bearer_token or os.getenv('TWITTER_BEARER_TOKEN')
        self.client = None
        self.rate_limit_reset = {}
        self.request_counts = defaultdict(int)
        
        if self.bearer_token and TWITTER_AVAILABLE:
            try:
                self.client = tweepy.Client(bearer_token=self.bearer_token)
                logger.info("✅ Twitter API client initialized")
            except Exception as e:
                logger.error(f"❌ Twitter API initialization failed: {e}")
                self.client = None
        else:
            logger.warning("⚠️ Twitter API not available (missing token or tweepy)")
    
    async def search_crypto_mentions(self, symbols: List[str], limit: int = 100) -> List[SocialPost]:
        """Search for crypto mentions on Twitter"""
        
        if not self.client:
            return []
        
        all_posts = []
        
        for symbol in symbols[:3]:  # Limit to avoid rate limits
            try:
                # Create search query
                base_symbol = symbol.replace('/USDT', '').replace('/USD', '')
                query = f"({base_symbol} OR ${base_symbol}) crypto -is:retweet lang:en"
                
                # Check rate limits
                if not await self._check_rate_limit('search'):
                    logger.warning(f"Twitter rate limit reached for search")
                    break
                
                # Search tweets
                tweets = tweepy.Paginator(
                    self.client.search_recent_tweets,
                    query=query,
                    tweet_fields=['created_at', 'author_id', 'public_metrics', 'context_annotations'],
                    max_results=min(limit, 100),  # API limit
                    limit=1
                ).flatten(limit=limit)
                
                # Convert to SocialPost objects
                for tweet in tweets:
                    if tweet.text and len(tweet.text) > 20:  # Filter very short tweets
                        
                        # Extract mentions
                        mentions = self._extract_crypto_mentions(tweet.text)
                        
                        if mentions:  # Only process if crypto mentioned
                            post = SocialPost(
                                id=str(tweet.id),
                                platform='twitter',
                                author=str(tweet.author_id),
                                content=tweet.text,
                                timestamp=tweet.created_at,
                                engagement={
                                    'retweets': tweet.public_metrics.get('retweet_count', 0),
                                    'likes': tweet.public_metrics.get('like_count', 0),
                                    'replies': tweet.public_metrics.get('reply_count', 0),
                                    'quotes': tweet.public_metrics.get('quote_count', 0)
                                },
                                sentiment_raw=0.0,  # Will be calculated
                                sentiment_processed=0.0,
                                confidence=0.0,
                                mentions=mentions,
                                metadata={'query': query, 'symbol': symbol}
                            )
                            
                            # Calculate sentiment
                            post.sentiment_raw = self._calculate_text_sentiment(post.content)
                            post.sentiment_processed, post.confidence = self._process_sentiment(
                                post.sentiment_raw, post.engagement, post.content
                            )
                            
                            all_posts.append(post)
                
                # Small delay between symbols
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error searching Twitter for {symbol}: {e}")
                continue
        
        logger.info(f"📱 Collected {len(all_posts)} Twitter posts")
        return all_posts
    
    async def _check_rate_limit(self, endpoint: str) -> bool:
        """Check if we can make API calls"""
        current_time = time.time()
        
        # Twitter API v2 limits: 300 requests per 15 minutes for search
        window_start = current_time - 900  # 15 minutes
        
        # Count recent requests
        recent_requests = sum(1 for t in self.request_counts[endpoint] 
                            if t > window_start)
        
        if recent_requests >= 300:
            return False
        
        # Log this request
        if endpoint not in self.request_counts:
            self.request_counts[endpoint] = deque(maxlen=300)
        
        self.request_counts[endpoint].append(current_time)
        return True
    
    def _extract_crypto_mentions(self, text: str) -> List[str]:
        """Extract cryptocurrency mentions from text"""
        mentions = []
        
        # Common crypto symbols
        crypto_patterns = [
            r'\b(BTC|Bitcoin)\b',
            r'\b(ETH|Ethereum)\b', 
            r'\b(SOL|Solana)\b',
            r'\b(ADA|Cardano)\b',
            r'\b(AVAX|Avalanche)\b',
            r'\b(MATIC|Polygon)\b',
            r'\b(DOT|Polkadot)\b',
            r'\b(ATOM|Cosmos)\b',
            r'\b(NEAR)\b',
            r'\b(FTM|Fantom)\b',
            r'\$(BTC|ETH|SOL|ADA|AVAX|MATIC|DOT|ATOM|NEAR|FTM)',
            r'#(Bitcoin|Ethereum|Crypto)'
        ]
        
        text_upper = text.upper()
        
        for pattern in crypto_patterns:
            matches = re.findall(pattern, text_upper, re.IGNORECASE)
            mentions.extend(matches)
        
        return list(set(mentions))  # Remove duplicates
    
    def _calculate_text_sentiment(self, text: str) -> float:
        """Calculate basic sentiment score from text"""
        
        # Bullish keywords
        bullish_words = {
            'moon', 'rocket', 'pump', 'bullish', 'buy', 'hodl', 'diamond', 'hands',
            'breakout', 'surge', 'rally', 'explosion', 'massive', 'gains', 'profit',
            'accumulate', 'undervalued', 'gem', 'hidden', 'potential', 'explosive',
            'strong', 'support', 'resistance', 'breakthrough', 'adoption', 'partnership'
        }
        
        # Bearish keywords
        bearish_words = {
            'dump', 'crash', 'bearish', 'sell', 'short', 'dead', 'scam', 'rug',
            'collapse', 'plummet', 'disaster', 'avoid', 'warning', 'danger',
            'overvalued', 'bubble', 'panic', 'fear', 'liquidation', 'exit',
            'weak', 'broken', 'support', 'resistance', 'decline', 'falling'
        }
        
        # Intensity multipliers
        intensity_words = {
            'very': 1.5, 'extremely': 2.0, 'super': 1.8, 'ultra': 2.0,
            'massive': 2.2, 'huge': 1.9, 'enormous': 2.1, 'insane': 2.3
        }
        
        text_lower = text.lower()
        words = text_lower.split()
        
        bullish_score = 0
        bearish_score = 0
        intensity = 1.0
        
        for i, word in enumerate(words):
            # Check for intensity multipliers
            if word in intensity_words:
                intensity = intensity_words[word]
                continue
            
            # Check sentiment
            if word in bullish_words:
                bullish_score += intensity
                intensity = 1.0  # Reset intensity
            elif word in bearish_words:
                bearish_score += intensity
                intensity = 1.0
        
        # Calculate net sentiment
        if bullish_score + bearish_score == 0:
            return 0.0
        
        net_sentiment = (bullish_score - bearish_score) / max(len(words), 10)
        return max(-1.0, min(1.0, net_sentiment))  # Clamp to [-1, 1]
    
    def _process_sentiment(self, raw_sentiment: float, engagement: Dict[str, int], 
                          content: str) -> Tuple[float, float]:
        """Process raw sentiment with engagement and content analysis"""
        
        # Base sentiment
        processed = raw_sentiment
        
        # Engagement boost
        total_engagement = sum(engagement.values())
        engagement_multiplier = min(1 + (total_engagement / 1000), 2.0)  # Max 2x boost
        processed *= engagement_multiplier
        
        # Content length factor (longer posts often more thoughtful)
        length_factor = min(len(content) / 200, 1.5)  # Max 1.5x boost
        processed *= length_factor
        
        # Calculate confidence
        confidence = abs(raw_sentiment) * 0.5  # Base confidence from sentiment strength
        confidence += min(total_engagement / 500, 0.3)  # Engagement confidence
        confidence += min(len(content) / 300, 0.2)  # Length confidence
        confidence = min(confidence, 1.0)
        
        return processed, confidence

class RedditAPIClient:
    """
    Reddit API client using PRAW for crypto sentiment
    
    Requires Reddit API credentials
    """
    
    def __init__(self, client_id: str = None, client_secret: str = None, user_agent: str = None):
        self.client_id = client_id or os.getenv('REDDIT_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('REDDIT_CLIENT_SECRET')
        self.user_agent = user_agent or "CryptoSentimentBot/1.0"
        
        self.reddit = None
        
        if self.client_id and self.client_secret and REDDIT_AVAILABLE:
            try:
                self.reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent
                )
                logger.info("✅ Reddit API client initialized")
            except Exception as e:
                logger.error(f"❌ Reddit API initialization failed: {e}")
                self.reddit = None
        else:
            logger.warning("⚠️ Reddit API not available (missing credentials or PRAW)")
    
    async def search_crypto_discussions(self, symbols: List[str], limit: int = 50) -> List[SocialPost]:
        """Search for crypto discussions on Reddit"""
        
        if not self.reddit:
            return []
        
        all_posts = []
        
        # Target subreddits
        crypto_subreddits = [
            'CryptoCurrency', 'Bitcoin', 'ethereum', 'solana', 'cardano',
            'CryptoMarkets', 'altcoin', 'SatoshiStreetBets', 'CryptoMoonShots'
        ]
        
        try:
            for subreddit_name in crypto_subreddits[:3]:  # Limit to avoid rate limits
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Search hot posts
                for submission in subreddit.hot(limit=min(limit // 3, 20)):
                    
                    # Extract crypto mentions
                    title_mentions = self._extract_crypto_mentions(submission.title)
                    content_mentions = self._extract_crypto_mentions(submission.selftext or "")
                    all_mentions = list(set(title_mentions + content_mentions))
                    
                    # Filter for relevant symbols
                    relevant_mentions = [m for m in all_mentions 
                                       if any(symbol.upper().startswith(m.upper()) 
                                             for symbol in symbols)]
                    
                    if relevant_mentions:
                        full_content = f"{submission.title}\n{submission.selftext or ''}"
                        
                        post = SocialPost(
                            id=submission.id,
                            platform='reddit',
                            author=str(submission.author) if submission.author else 'deleted',
                            content=full_content,
                            timestamp=datetime.fromtimestamp(submission.created_utc),
                            engagement={
                                'upvotes': submission.ups,
                                'downvotes': submission.downs if hasattr(submission, 'downs') else 0,
                                'comments': submission.num_comments,
                                'upvote_ratio': submission.upvote_ratio
                            },
                            sentiment_raw=0.0,
                            sentiment_processed=0.0,
                            confidence=0.0,
                            mentions=relevant_mentions,
                            metadata={
                                'subreddit': subreddit_name,
                                'score': submission.score,
                                'url': submission.url
                            }
                        )
                        
                        # Calculate sentiment
                        post.sentiment_raw = self._calculate_reddit_sentiment(post.content, post.engagement)
                        post.sentiment_processed, post.confidence = self._process_reddit_sentiment(
                            post.sentiment_raw, post.engagement, post.content
                        )
                        
                        all_posts.append(post)
                
                # Small delay between subreddits
                await asyncio.sleep(2)
                
        except Exception as e:
            logger.error(f"Error searching Reddit: {e}")
        
        logger.info(f"📱 Collected {len(all_posts)} Reddit posts")
        return all_posts
    
    def _extract_crypto_mentions(self, text: str) -> List[str]:
        """Extract cryptocurrency mentions from Reddit text"""
        mentions = []
        
        # Reddit-specific patterns
        crypto_patterns = [
            r'\b(BTC|Bitcoin)\b',
            r'\b(ETH|Ethereum)\b',
            r'\b(SOL|Solana)\b',
            r'\b(ADA|Cardano)\b',
            r'\b(AVAX|Avalanche)\b',
            r'\b(MATIC|Polygon)\b',
            r'\b(DOT|Polkadot)\b',
            r'\b(ATOM|Cosmos)\b',
            r'\b(NEAR)\b',
            r'\b(FTM|Fantom)\b'
        ]
        
        text_upper = text.upper()
        
        for pattern in crypto_patterns:
            matches = re.findall(pattern, text_upper, re.IGNORECASE)
            mentions.extend(matches)
        
        return list(set(mentions))
    
    def _calculate_reddit_sentiment(self, text: str, engagement: Dict[str, int]) -> float:
        """Calculate sentiment for Reddit posts"""
        
        # Use similar logic as Twitter but with Reddit-specific adjustments
        text_sentiment = self._calculate_text_sentiment(text)
        
        # Reddit upvote ratio is a strong signal
        upvote_ratio = engagement.get('upvote_ratio', 0.5)
        ratio_sentiment = (upvote_ratio - 0.5) * 2  # Convert 0.5-1.0 to -1.0-1.0
        
        # Combine text and ratio sentiment
        combined_sentiment = (text_sentiment * 0.7) + (ratio_sentiment * 0.3)
        
        return max(-1.0, min(1.0, combined_sentiment))
    
    def _calculate_text_sentiment(self, text: str) -> float:
        """Same as Twitter sentiment calculation"""
        bullish_words = {
            'moon', 'rocket', 'pump', 'bullish', 'buy', 'hodl', 'diamond', 'hands',
            'breakout', 'surge', 'rally', 'explosion', 'massive', 'gains', 'profit',
            'accumulate', 'undervalued', 'gem', 'hidden', 'potential', 'explosive'
        }
        
        bearish_words = {
            'dump', 'crash', 'bearish', 'sell', 'short', 'dead', 'scam', 'rug',
            'collapse', 'plummet', 'disaster', 'avoid', 'warning', 'danger',
            'overvalued', 'bubble', 'panic', 'fear', 'liquidation', 'exit'
        }
        
        text_lower = text.lower()
        words = text_lower.split()
        
        bullish_score = sum(1 for word in words if word in bullish_words)
        bearish_score = sum(1 for word in words if word in bearish_words)
        
        if bullish_score + bearish_score == 0:
            return 0.0
        
        net_sentiment = (bullish_score - bearish_score) / max(len(words), 10)
        return max(-1.0, min(1.0, net_sentiment))
    
    def _process_reddit_sentiment(self, raw_sentiment: float, engagement: Dict[str, int], 
                                 content: str) -> Tuple[float, float]:
        """Process Reddit sentiment with engagement factors"""
        
        processed = raw_sentiment
        
        # Score and comments boost
        score = engagement.get('upvotes', 1) - engagement.get('downvotes', 0)
        score_multiplier = min(1 + (score / 100), 2.0)
        processed *= score_multiplier
        
        # Comments indicate engagement
        comments = engagement.get('comments', 0)
        comment_multiplier = min(1 + (comments / 50), 1.5)
        processed *= comment_multiplier
        
        # Calculate confidence
        confidence = abs(raw_sentiment) * 0.6
        confidence += min(score / 100, 0.2)
        confidence += min(comments / 20, 0.2)
        confidence = min(confidence, 1.0)
        
        return processed, confidence

class LiveSocialSentimentAnalyzer:
    """
    Unified live social sentiment analyzer
    
    Combines Twitter, Reddit, and other sources for real-time sentiment
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize API clients
        self.twitter_client = TwitterAPIClient()
        self.reddit_client = RedditAPIClient()
        
        # Data storage
        self.recent_posts = defaultdict(list)
        self.sentiment_history = defaultdict(deque)
        
        # Configuration
        self.update_interval = self.config.get('update_interval', 300)  # 5 minutes
        self.max_posts_per_platform = self.config.get('max_posts_per_platform', 100)
        self.sentiment_threshold = self.config.get('sentiment_threshold', 0.3)
        
        logger.info("🚀 Live Social Sentiment Analyzer initialized")
    
    async def analyze_sentiment(self, symbols: List[str]) -> List[SentimentSnapshot]:
        """Analyze live sentiment for given symbols"""
        
        all_snapshots = []
        
        # Collect data from all platforms
        twitter_posts = await self._collect_twitter_data(symbols)
        reddit_posts = await self._collect_reddit_data(symbols)
        
        # Combine all posts
        all_posts = twitter_posts + reddit_posts
        
        # Group by symbol and platform
        symbol_platform_posts = defaultdict(lambda: defaultdict(list))
        
        for post in all_posts:
            for mention in post.mentions:
                # Map mention to symbol
                matched_symbol = self._match_mention_to_symbol(mention, symbols)
                if matched_symbol:
                    symbol_platform_posts[matched_symbol][post.platform].append(post)
        
        # Create snapshots for each symbol/platform combination
        for symbol, platform_posts in symbol_platform_posts.items():
            for platform, posts in platform_posts.items():
                
                if len(posts) >= 3:  # Minimum posts for reliable sentiment
                    snapshot = self._create_sentiment_snapshot(symbol, platform, posts)
                    if snapshot.confidence >= 0.5:  # Minimum confidence
                        all_snapshots.append(snapshot)
        
        # Sort by confidence and sentiment strength
        all_snapshots.sort(key=lambda x: x.confidence * abs(x.average_sentiment), reverse=True)
        
        return all_snapshots
    
    async def _collect_twitter_data(self, symbols: List[str]) -> List[SocialPost]:
        """Collect Twitter data"""
        try:
            return await self.twitter_client.search_crypto_mentions(symbols, self.max_posts_per_platform)
        except Exception as e:
            logger.error(f"Error collecting Twitter data: {e}")
            return []
    
    async def _collect_reddit_data(self, symbols: List[str]) -> List[SocialPost]:
        """Collect Reddit data"""
        try:
            return await self.reddit_client.search_crypto_discussions(symbols, self.max_posts_per_platform)
        except Exception as e:
            logger.error(f"Error collecting Reddit data: {e}")
            return []
    
    def _match_mention_to_symbol(self, mention: str, symbols: List[str]) -> Optional[str]:
        """Match a mention to a trading symbol"""
        mention_upper = mention.upper()
        
        # Direct mapping
        symbol_mapping = {
            'BTC': 'BTC/USDT', 'BITCOIN': 'BTC/USDT',
            'ETH': 'ETH/USDT', 'ETHEREUM': 'ETH/USDT',
            'SOL': 'SOL/USDT', 'SOLANA': 'SOL/USDT',
            'ADA': 'ADA/USDT', 'CARDANO': 'ADA/USDT',
            'AVAX': 'AVAX/USDT', 'AVALANCHE': 'AVAX/USDT',
            'MATIC': 'MATIC/USDT', 'POLYGON': 'MATIC/USDT',
            'DOT': 'DOT/USDT', 'POLKADOT': 'DOT/USDT',
            'ATOM': 'ATOM/USDT', 'COSMOS': 'ATOM/USDT',
            'NEAR': 'NEAR/USDT',
            'FTM': 'FTM/USDT', 'FANTOM': 'FTM/USDT'
        }
        
        mapped_symbol = symbol_mapping.get(mention_upper)
        if mapped_symbol and mapped_symbol in symbols:
            return mapped_symbol
        
        # Fuzzy matching
        for symbol in symbols:
            if symbol.upper().startswith(mention_upper):
                return symbol
        
        return None
    
    def _create_sentiment_snapshot(self, symbol: str, platform: str, posts: List[SocialPost]) -> SentimentSnapshot:
        """Create sentiment snapshot from posts"""
        
        # Calculate aggregated metrics
        total_sentiment = sum(post.sentiment_processed for post in posts)
        avg_sentiment = total_sentiment / len(posts)
        
        # Calculate engagement score
        total_engagement = 0
        for post in posts:
            engagement_sum = sum(post.engagement.values())
            total_engagement += engagement_sum
        
        engagement_score = total_engagement / len(posts) if posts else 0
        
        # Calculate momentum (needs historical data)
        momentum = self._calculate_sentiment_momentum(symbol, platform, avg_sentiment)
        
        # Overall confidence
        avg_confidence = sum(post.confidence for post in posts) / len(posts)
        posts_confidence = min(len(posts) / 10, 1.0)  # More posts = higher confidence
        overall_confidence = (avg_confidence + posts_confidence) / 2
        
        # Determine trend direction
        if avg_sentiment > 0.2:
            trend_direction = 'bullish'
        elif avg_sentiment < -0.2:
            trend_direction = 'bearish'
        else:
            trend_direction = 'neutral'
        
        # Get top posts by engagement
        top_posts = sorted(posts, key=lambda p: sum(p.engagement.values()), reverse=True)[:3]
        
        return SentimentSnapshot(
            symbol=symbol,
            timestamp=datetime.now(),
            platform=platform,
            posts_analyzed=len(posts),
            average_sentiment=avg_sentiment,
            sentiment_momentum=momentum,
            engagement_score=engagement_score,
            confidence=overall_confidence,
            top_posts=top_posts,
            trend_direction=trend_direction
        )
    
    def _calculate_sentiment_momentum(self, symbol: str, platform: str, current_sentiment: float) -> float:
        """Calculate sentiment momentum vs historical"""
        
        key = f"{symbol}_{platform}"
        
        # Store current sentiment
        if key not in self.sentiment_history:
            self.sentiment_history[key] = deque(maxlen=10)  # Keep last 10 readings
        
        self.sentiment_history[key].append(current_sentiment)
        
        # Calculate momentum
        if len(self.sentiment_history[key]) >= 3:
            recent_avg = sum(list(self.sentiment_history[key])[-3:]) / 3
            older_avg = sum(list(self.sentiment_history[key])[:-3]) / max(len(list(self.sentiment_history[key])[:-3]), 1)
            momentum = recent_avg - older_avg
        else:
            momentum = 0.0
        
        return momentum
    
    def get_top_sentiment_signals(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top sentiment signals for trading"""
        
        # This would be called by the main strategy
        # For now, return empty list (implementation depends on recent data)
        return []

# Utility functions
def create_live_sentiment_analyzer(config: Dict[str, Any] = None) -> LiveSocialSentimentAnalyzer:
    """Create live sentiment analyzer with configuration"""
    return LiveSocialSentimentAnalyzer(config)

async def test_live_apis():
    """Test live API functionality"""
    print("🧪 Testing Live Social APIs...")
    
    analyzer = create_live_sentiment_analyzer({
        'update_interval': 300,
        'max_posts_per_platform': 20,
        'sentiment_threshold': 0.3
    })
    
    test_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    try:
        snapshots = await analyzer.analyze_sentiment(test_symbols)
        
        print(f"📊 Collected {len(snapshots)} sentiment snapshots")
        
        for snapshot in snapshots[:3]:  # Show top 3
            print(f"📱 {snapshot.platform}: {snapshot.symbol}")
            print(f"   Sentiment: {snapshot.average_sentiment:+.2f}")
            print(f"   Confidence: {snapshot.confidence:.2f}")
            print(f"   Posts: {snapshot.posts_analyzed}")
            print(f"   Trend: {snapshot.trend_direction}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    # Run test
    asyncio.run(test_live_apis())