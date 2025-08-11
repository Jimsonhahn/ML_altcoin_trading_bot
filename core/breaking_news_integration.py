#!/usr/bin/env python3
"""
Breaking News Integration System  
===============================

Real-time financial news monitoring and analysis for crypto trading:
- Multiple news source integration (CoinDesk, CoinTelegraph, etc.)
- AI-powered sentiment analysis of headlines
- Market impact assessment
- Real-time alerts for significant news
- News-based trading signal generation
"""

import asyncio
import aiohttp
import logging
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import hashlib
from pathlib import Path
import time

# RSS/XML parsing
try:
    import feedparser
    RSS_AVAILABLE = True
except ImportError:
    RSS_AVAILABLE = False
    logging.warning("feedparser not available - RSS feeds disabled")

# HTML parsing for web scraping
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logging.warning("BeautifulSoup4 not available - web scraping disabled")

logger = logging.getLogger(__name__)

@dataclass
class NewsItem:
    """Individual news item"""
    id: str
    source: str
    title: str
    content: str
    url: str
    published_time: datetime
    sentiment_score: float
    impact_score: float
    confidence: float
    mentioned_coins: List[str]
    categories: List[str]
    urgency: str  # 'low', 'medium', 'high', 'critical'
    
    def is_recent(self, hours: int = 24) -> bool:
        """Check if news is recent"""
        return datetime.now() - self.published_time <= timedelta(hours=hours)

@dataclass
class NewsSignal:
    """Trading signal derived from news"""
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    impact_score: float
    urgency: str
    news_items: List[NewsItem]
    reasoning: str
    timestamp: datetime
    expiry_time: datetime
    
    def is_expired(self) -> bool:
        return datetime.now() > self.expiry_time

@dataclass
class NewsSource:
    """News source configuration"""
    name: str
    url: str
    source_type: str  # 'rss', 'api', 'scrape'
    update_interval: int  # minutes
    reliability_score: float  # 0-1
    api_key: Optional[str] = None
    headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}

class NewsAggregator:
    """
    News aggregation from multiple sources
    
    Collects and processes crypto news from various sources
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.session = None
        
        # Initialize news sources
        self.sources = self._initialize_sources()
        
        # Data storage
        self.news_cache = deque(maxlen=1000)
        self.processed_news_ids = set()
        
        # Rate limiting
        self.request_counts = defaultdict(deque)
        
        logger.info("📰 News Aggregator initialized")
    
    def _initialize_sources(self) -> List[NewsSource]:
        """Initialize news sources"""
        
        sources = [
            NewsSource(
                name='CoinDesk',
                url='https://www.coindesk.com/arc/outboundfeeds/rss/',
                source_type='rss',
                update_interval=10,
                reliability_score=0.9,
                headers={'User-Agent': 'Mozilla/5.0 (compatible; CryptoBot/1.0)'}
            ),
            NewsSource(
                name='CoinTelegraph',
                url='https://cointelegraph.com/rss',
                source_type='rss',
                update_interval=15,
                reliability_score=0.8,
                headers={'User-Agent': 'Mozilla/5.0 (compatible; CryptoBot/1.0)'}
            ),
            NewsSource(
                name='CryptoNews',
                url='https://cryptonews.com/news/feed/',
                source_type='rss', 
                update_interval=20,
                reliability_score=0.7,
                headers={'User-Agent': 'Mozilla/5.0 (compatible; CryptoBot/1.0)'}
            ),
            NewsSource(
                name='Bitcoin.com',
                url='https://news.bitcoin.com/feed/',
                source_type='rss',
                update_interval=30,
                reliability_score=0.8,
                headers={'User-Agent': 'Mozilla/5.0 (compatible; CryptoBot/1.0)'}
            ),
            NewsSource(
                name='NewsBTC',
                url='https://www.newsbtc.com/feed/',
                source_type='rss',
                update_interval=25,
                reliability_score=0.7,
                headers={'User-Agent': 'Mozilla/5.0 (compatible; CryptoBot/1.0)'}
            )
        ]
        
        # Filter enabled sources
        enabled_sources = self.config.get('enabled_sources', [s.name for s in sources])
        return [s for s in sources if s.name in enabled_sources]
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_latest_news(self) -> List[NewsItem]:
        """Fetch latest news from all sources"""
        
        if not self.session:
            raise RuntimeError("NewsAggregator must be used as async context manager")
        
        all_news = []
        
        # Fetch from each source
        for source in self.sources:
            try:
                if not await self._check_rate_limit(source.name, 1):
                    logger.warning(f"Rate limit exceeded for {source.name}")
                    continue
                
                news_items = await self._fetch_source_news(source)
                all_news.extend(news_items)
                
                logger.info(f"📰 Fetched {len(news_items)} items from {source.name}")
                
            except Exception as e:
                logger.error(f"Error fetching from {source.name}: {e}")
                continue
        
        # Remove duplicates and filter recent
        unique_news = self._deduplicate_news(all_news)
        recent_news = [item for item in unique_news if item.is_recent(48)]  # Last 48 hours
        
        # Cache news
        self.news_cache.extend(recent_news)
        
        logger.info(f"📊 Fetched {len(recent_news)} unique recent news items")
        
        return recent_news
    
    async def _fetch_source_news(self, source: NewsSource) -> List[NewsItem]:
        """Fetch news from a single source"""
        
        if source.source_type == 'rss' and RSS_AVAILABLE:
            return await self._fetch_rss_news(source)
        elif source.source_type == 'scrape' and BS4_AVAILABLE:
            return await self._fetch_scraped_news(source)
        else:
            logger.warning(f"Unsupported source type: {source.source_type} for {source.name}")
            return []
    
    async def _fetch_rss_news(self, source: NewsSource) -> List[NewsItem]:
        """Fetch news from RSS feed"""
        
        try:
            # Fetch RSS content
            timeout = aiohttp.ClientTimeout(total=30)
            async with self.session.get(source.url, headers=source.headers, timeout=timeout) as response:
                if response.status != 200:
                    logger.warning(f"HTTP {response.status} from {source.name}")
                    return []
                
                content = await response.text()
            
            # Parse RSS
            feed = feedparser.parse(content)
            news_items = []
            
            for entry in feed.entries[:20]:  # Limit to 20 most recent
                try:
                    # Generate unique ID
                    news_id = hashlib.md5(f"{source.name}_{entry.link}".encode()).hexdigest()
                    
                    # Skip if already processed
                    if news_id in self.processed_news_ids:
                        continue
                    
                    # Parse publication time
                    published_time = datetime.now()
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published_time = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        published_time = datetime(*entry.updated_parsed[:6])
                    
                    # Extract content
                    content = ''
                    if hasattr(entry, 'content'):
                        content = entry.content[0].value if entry.content else ''
                    elif hasattr(entry, 'summary'):
                        content = entry.summary
                    elif hasattr(entry, 'description'):
                        content = entry.description
                    
                    # Clean HTML from content
                    if BS4_AVAILABLE and content:
                        soup = BeautifulSoup(content, 'html.parser')
                        content = soup.get_text().strip()
                    
                    # Create news item
                    news_item = NewsItem(
                        id=news_id,
                        source=source.name,
                        title=entry.title if hasattr(entry, 'title') else '',
                        content=content,
                        url=entry.link if hasattr(entry, 'link') else '',
                        published_time=published_time,
                        sentiment_score=0.0,  # Will be calculated later
                        impact_score=0.0,     # Will be calculated later
                        confidence=0.0,       # Will be calculated later
                        mentioned_coins=[],   # Will be extracted later
                        categories=[],        # Will be classified later
                        urgency='medium'      # Will be assessed later
                    )
                    
                    news_items.append(news_item)
                    self.processed_news_ids.add(news_id)
                    
                except Exception as e:
                    logger.warning(f"Error parsing entry from {source.name}: {e}")
                    continue
            
            return news_items
            
        except Exception as e:
            logger.error(f"Error fetching RSS from {source.name}: {e}")
            return []
    
    async def _fetch_scraped_news(self, source: NewsSource) -> List[NewsItem]:
        """Fetch news by web scraping (placeholder implementation)"""
        
        # This would require specific scraping logic for each site
        # For now, return empty list
        logger.info(f"Web scraping not implemented for {source.name}")
        return []
    
    def _deduplicate_news(self, news_items: List[NewsItem]) -> List[NewsItem]:
        """Remove duplicate news items"""
        
        seen_urls = set()
        seen_titles = set()
        unique_items = []
        
        for item in news_items:
            # Check URL duplicates
            if item.url and item.url in seen_urls:
                continue
            
            # Check title similarity (basic)
            title_key = re.sub(r'[^\w\s]', '', item.title.lower())
            if title_key in seen_titles:
                continue
            
            seen_urls.add(item.url)
            seen_titles.add(title_key)
            unique_items.append(item)
        
        return unique_items
    
    async def _check_rate_limit(self, source: str, requests: int) -> bool:
        """Check rate limiting for source"""
        
        current_time = time.time()
        
        # Clean old requests (older than 1 minute)
        while (self.request_counts[source] and 
               self.request_counts[source][0] < current_time - 60):
            self.request_counts[source].popleft()
        
        # Check if we can make requests
        if len(self.request_counts[source]) + requests > 60:  # 60 requests per minute max
            return False
        
        # Add requests
        for _ in range(requests):
            self.request_counts[source].append(current_time)
        
        return True

class NewsAnalyzer:
    """
    News content analysis and signal generation
    
    Analyzes news content for sentiment, impact, and trading signals
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Analysis parameters
        self.sentiment_threshold = self.config.get('sentiment_threshold', 0.3)
        self.impact_threshold = self.config.get('impact_threshold', 0.5)
        self.signal_confidence_threshold = self.config.get('signal_confidence_threshold', 0.7)
        
        # Crypto keywords for coin detection
        self.crypto_keywords = self._initialize_crypto_keywords()
        
        # Sentiment keywords
        self.sentiment_keywords = self._initialize_sentiment_keywords()
        
        # Impact keywords
        self.impact_keywords = self._initialize_impact_keywords()
        
        logger.info("🔍 News Analyzer initialized")
    
    def _initialize_crypto_keywords(self) -> Dict[str, List[str]]:
        """Initialize cryptocurrency keywords"""
        
        return {
            'BTC': ['bitcoin', 'btc', 'satoshi'],
            'ETH': ['ethereum', 'eth', 'ether', 'vitalik'],
            'SOL': ['solana', 'sol'],
            'AVAX': ['avalanche', 'avax'],
            'MATIC': ['polygon', 'matic'],
            'ADA': ['cardano', 'ada'],
            'DOT': ['polkadot', 'dot', 'kusama'],
            'ATOM': ['cosmos', 'atom'],
            'NEAR': ['near protocol', 'near'],
            'FTM': ['fantom', 'ftm'],
            'LINK': ['chainlink', 'link'],
            'UNI': ['uniswap', 'uni'],
            'AAVE': ['aave'],
            'COMP': ['compound', 'comp'],
            'MKR': ['maker', 'mkr', 'dai']
        }
    
    def _initialize_sentiment_keywords(self) -> Dict[str, List[str]]:
        """Initialize sentiment analysis keywords"""
        
        return {
            'very_positive': [
                'breakthrough', 'revolutionary', 'massive adoption', 'game changer',
                'historic', 'unprecedented', 'explosive growth', 'major milestone'
            ],
            'positive': [
                'bullish', 'surge', 'rally', 'breakout', 'adoption', 'partnership',
                'upgrade', 'launch', 'positive', 'growth', 'gain', 'rise', 'up',
                'approval', 'support', 'boost', 'momentum', 'optimistic'
            ],
            'negative': [
                'bearish', 'crash', 'dump', 'decline', 'fall', 'drop', 'down',
                'concern', 'fear', 'worry', 'risk', 'problem', 'issue', 'warning',
                'regulatory', 'ban', 'restriction', 'investigation'
            ],
            'very_negative': [
                'collapse', 'disaster', 'catastrophic', 'emergency', 'crisis',
                'scandal', 'fraud', 'hack', 'exploit', 'bankruptcy', 'shutdown'
            ]
        }
    
    def _initialize_impact_keywords(self) -> Dict[str, List[str]]:
        """Initialize market impact keywords"""
        
        return {
            'high_impact': [
                'fed', 'federal reserve', 'sec', 'regulation', 'etf', 'institutional',
                'blackrock', 'fidelity', 'tesla', 'microstrategy', 'government',
                'central bank', 'legislation', 'legal', 'court', 'ruling'
            ],
            'medium_impact': [
                'partnership', 'acquisition', 'merge', 'funding', 'investment',
                'launch', 'release', 'update', 'upgrade', 'integration',
                'listing', 'exchange', 'whale', 'large transaction'
            ],
            'market_moving': [
                'breaking', 'urgent', 'alert', 'developing', 'exclusive',
                'confirmed', 'official', 'announcement', 'statement'
            ]
        }
    
    async def analyze_news_batch(self, news_items: List[NewsItem]) -> List[NewsItem]:
        """Analyze batch of news items"""
        
        analyzed_items = []
        
        for item in news_items:
            try:
                analyzed_item = await self.analyze_news_item(item)
                analyzed_items.append(analyzed_item)
            except Exception as e:
                logger.error(f"Error analyzing news item {item.id}: {e}")
                analyzed_items.append(item)  # Return original if analysis fails
        
        logger.info(f"🔍 Analyzed {len(analyzed_items)} news items")
        
        return analyzed_items
    
    async def analyze_news_item(self, item: NewsItem) -> NewsItem:
        """Analyze individual news item"""
        
        # Extract mentioned coins
        item.mentioned_coins = self._extract_mentioned_coins(item.title + " " + item.content)
        
        # Analyze sentiment
        item.sentiment_score = self._analyze_sentiment(item.title + " " + item.content)
        
        # Assess market impact
        item.impact_score = self._assess_impact(item.title + " " + item.content, item.source)
        
        # Determine urgency
        item.urgency = self._assess_urgency(item.title + " " + item.content, item.published_time)
        
        # Classify categories
        item.categories = self._classify_categories(item.title + " " + item.content)
        
        # Calculate overall confidence
        item.confidence = self._calculate_confidence(item)
        
        return item
    
    def _extract_mentioned_coins(self, text: str) -> List[str]:
        """Extract mentioned cryptocurrency symbols"""
        
        text_lower = text.lower()
        mentioned_coins = []
        
        for symbol, keywords in self.crypto_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    mentioned_coins.append(symbol)
                    break  # Don't double-count same coin
        
        return list(set(mentioned_coins))  # Remove duplicates
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text (-1 to +1)"""
        
        text_lower = text.lower()
        words = text_lower.split()
        
        sentiment_score = 0.0
        total_weight = 0.0
        
        # Count sentiment words with weights
        for word in words:
            if word in self.sentiment_keywords['very_positive']:
                sentiment_score += 2.0
                total_weight += 2.0
            elif word in self.sentiment_keywords['positive']:
                sentiment_score += 1.0
                total_weight += 1.0
            elif word in self.sentiment_keywords['negative']:
                sentiment_score -= 1.0
                total_weight += 1.0
            elif word in self.sentiment_keywords['very_negative']:
                sentiment_score -= 2.0
                total_weight += 2.0
        
        # Normalize score
        if total_weight > 0:
            normalized_score = sentiment_score / total_weight
        else:
            normalized_score = 0.0
        
        # Apply context modifiers
        if 'not' in text_lower or "n't" in text_lower:
            normalized_score *= -0.5  # Reverse but reduce intensity
        
        return max(-1.0, min(1.0, normalized_score))
    
    def _assess_impact(self, text: str, source: str) -> float:
        """Assess potential market impact (0 to 1)"""
        
        text_lower = text.lower()
        impact_score = 0.0
        
        # High impact keywords
        for keyword in self.impact_keywords['high_impact']:
            if keyword in text_lower:
                impact_score += 0.8
        
        # Medium impact keywords
        for keyword in self.impact_keywords['medium_impact']:
            if keyword in text_lower:
                impact_score += 0.5
        
        # Market moving keywords
        for keyword in self.impact_keywords['market_moving']:
            if keyword in text_lower:
                impact_score += 0.3
        
        # Source reliability multiplier
        source_multipliers = {
            'CoinDesk': 1.0,
            'CoinTelegraph': 0.9,
            'Bitcoin.com': 0.8,
            'CryptoNews': 0.7,
            'NewsBTC': 0.7
        }
        
        multiplier = source_multipliers.get(source, 0.6)
        impact_score *= multiplier
        
        return min(1.0, impact_score)
    
    def _assess_urgency(self, text: str, published_time: datetime) -> str:
        """Assess news urgency"""
        
        text_lower = text.lower()
        
        # Critical urgency keywords
        critical_keywords = ['breaking', 'urgent', 'alert', 'emergency', 'crisis']
        if any(keyword in text_lower for keyword in critical_keywords):
            return 'critical'
        
        # High urgency - recent and important
        high_keywords = ['developing', 'confirmed', 'official', 'announcement']
        time_since = datetime.now() - published_time
        
        if (any(keyword in text_lower for keyword in high_keywords) and 
            time_since <= timedelta(hours=2)):
            return 'high'
        
        # Medium urgency - standard news
        if time_since <= timedelta(hours=6):
            return 'medium'
        
        return 'low'
    
    def _classify_categories(self, text: str) -> List[str]:
        """Classify news into categories"""
        
        text_lower = text.lower()
        categories = []
        
        category_keywords = {
            'regulation': ['regulation', 'regulatory', 'sec', 'government', 'legal', 'law'],
            'institutional': ['institutional', 'bank', 'fund', 'investment', 'etf'],
            'technology': ['technology', 'blockchain', 'protocol', 'upgrade', 'update'],
            'partnership': ['partnership', 'collaboration', 'merge', 'acquisition'],
            'market': ['market', 'price', 'trading', 'volume', 'liquidity'],
            'adoption': ['adoption', 'mainstream', 'retail', 'consumer', 'payment'],
            'security': ['security', 'hack', 'exploit', 'vulnerability', 'breach']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                categories.append(category)
        
        return categories if categories else ['general']
    
    def _calculate_confidence(self, item: NewsItem) -> float:
        """Calculate overall confidence in analysis"""
        
        confidence = 0.5  # Base confidence
        
        # Source reliability
        source_reliability = {
            'CoinDesk': 0.9,
            'CoinTelegraph': 0.8,
            'Bitcoin.com': 0.7,
            'CryptoNews': 0.6,
            'NewsBTC': 0.6
        }
        
        confidence += source_reliability.get(item.source, 0.5) * 0.3
        
        # Content length (more content = higher confidence)
        content_length = len(item.title) + len(item.content)
        if content_length > 500:
            confidence += 0.2
        elif content_length > 200:
            confidence += 0.1
        
        # Recent news is more reliable
        time_since = datetime.now() - item.published_time
        if time_since <= timedelta(hours=1):
            confidence += 0.2
        elif time_since <= timedelta(hours=6):
            confidence += 0.1
        
        # Clear coin mentions increase confidence
        if len(item.mentioned_coins) > 0:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    async def generate_trading_signals(self, news_items: List[NewsItem]) -> List[NewsSignal]:
        """Generate trading signals from analyzed news"""
        
        signals = []
        
        # Group news by mentioned coins
        coin_news = defaultdict(list)
        
        for item in news_items:
            for coin in item.mentioned_coins:
                coin_news[coin].append(item)
        
        # Generate signals for each coin
        for coin, coin_items in coin_news.items():
            try:
                signal = await self._generate_coin_signal(coin, coin_items)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error generating signal for {coin}: {e}")
        
        # Filter and rank signals
        filtered_signals = [s for s in signals if s.confidence >= self.signal_confidence_threshold]
        ranked_signals = sorted(filtered_signals, key=lambda x: x.confidence * x.impact_score, reverse=True)
        
        logger.info(f"📡 Generated {len(ranked_signals)} trading signals from news")
        
        return ranked_signals
    
    async def _generate_coin_signal(self, coin: str, news_items: List[NewsItem]) -> Optional[NewsSignal]:
        """Generate trading signal for specific coin"""
        
        if not news_items:
            return None
        
        # Calculate aggregated metrics
        total_sentiment = sum(item.sentiment_score * item.confidence for item in news_items)
        total_impact = sum(item.impact_score * item.confidence for item in news_items)
        total_weight = sum(item.confidence for item in news_items)
        
        if total_weight == 0:
            return None
        
        avg_sentiment = total_sentiment / total_weight
        avg_impact = total_impact / total_weight
        
        # Determine signal type
        signal_type = 'HOLD'
        if avg_sentiment > self.sentiment_threshold and avg_impact > self.impact_threshold:
            signal_type = 'BUY'
        elif avg_sentiment < -self.sentiment_threshold and avg_impact > self.impact_threshold:
            signal_type = 'SELL'
        
        if signal_type == 'HOLD':
            return None
        
        # Calculate confidence
        confidence = min(avg_impact, 1.0) * min(abs(avg_sentiment), 1.0)
        
        # Determine urgency
        max_urgency = max(news_items, key=lambda x: ['low', 'medium', 'high', 'critical'].index(x.urgency)).urgency
        
        # Create reasoning text
        reasoning = self._create_signal_reasoning(coin, news_items, avg_sentiment, avg_impact)
        
        # Calculate expiry time based on urgency
        expiry_hours = {'critical': 1, 'high': 2, 'medium': 6, 'low': 12}.get(max_urgency, 6)
        expiry_time = datetime.now() + timedelta(hours=expiry_hours)
        
        return NewsSignal(
            symbol=f"{coin}/USDT",
            signal_type=signal_type,
            confidence=confidence,
            impact_score=avg_impact,
            urgency=max_urgency,
            news_items=news_items,
            reasoning=reasoning,
            timestamp=datetime.now(),
            expiry_time=expiry_time
        )
    
    def _create_signal_reasoning(self, coin: str, news_items: List[NewsItem], 
                                sentiment: float, impact: float) -> str:
        """Create human-readable reasoning for signal"""
        
        sentiment_desc = "positive" if sentiment > 0 else "negative"
        impact_desc = "high" if impact > 0.7 else "medium" if impact > 0.4 else "low"
        
        recent_items = [item for item in news_items if item.is_recent(6)]
        
        reasoning = f"{coin} shows {sentiment_desc} sentiment ({sentiment:+.2f}) with {impact_desc} impact ({impact:.2f}). "
        reasoning += f"Based on {len(recent_items)} recent news item(s): "
        
        # Add top news headlines
        top_news = sorted(news_items, key=lambda x: x.confidence * x.impact_score, reverse=True)[:2]
        headlines = [item.title[:50] + "..." if len(item.title) > 50 else item.title for item in top_news]
        reasoning += "; ".join(headlines)
        
        return reasoning

class BreakingNewsMonitor:
    """
    Breaking news monitoring and alert system
    
    Combines news aggregation and analysis for real-time monitoring
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        self.aggregator = NewsAggregator(config.get('aggregator', {}))
        self.analyzer = NewsAnalyzer(config.get('analyzer', {}))
        
        # Monitoring parameters
        self.update_interval = self.config.get('update_interval', 300)  # 5 minutes
        self.alert_threshold = self.config.get('alert_threshold', 0.8)
        
        # Data storage
        self.latest_signals = []
        self.alert_history = deque(maxlen=100)
        
        logger.info("📡 Breaking News Monitor initialized")
    
    async def start_monitoring(self, symbols: List[str]) -> List[NewsSignal]:
        """Start news monitoring and return latest signals"""
        
        try:
            async with self.aggregator:
                # Fetch latest news
                logger.info("📰 Fetching latest crypto news...")
                news_items = await self.aggregator.fetch_latest_news()
                
                if not news_items:
                    logger.info("📰 No new news items found")
                    return []
                
                # Analyze news
                logger.info("🔍 Analyzing news items...")
                analyzed_news = await self.analyzer.analyze_news_batch(news_items)
                
                # Generate trading signals
                logger.info("📡 Generating trading signals...")
                signals = await self.analyzer.generate_trading_signals(analyzed_news)
                
                # Filter signals for monitored symbols
                symbol_set = set(symbols)
                filtered_signals = [s for s in signals if s.symbol in symbol_set]
                
                # Check for alerts
                alerts = [s for s in filtered_signals if s.impact_score >= self.alert_threshold]
                
                if alerts:
                    logger.warning(f"🚨 {len(alerts)} HIGH IMPACT news alerts!")
                    for alert in alerts:
                        self.alert_history.append(alert)
                
                self.latest_signals = filtered_signals
                
                return filtered_signals
                
        except Exception as e:
            logger.error(f"Error in news monitoring: {e}")
            return []
    
    def get_latest_signals(self) -> List[NewsSignal]:
        """Get latest news-based trading signals"""
        
        # Remove expired signals
        current_signals = [s for s in self.latest_signals if not s.is_expired()]
        self.latest_signals = current_signals
        
        return current_signals
    
    def get_high_impact_alerts(self) -> List[NewsSignal]:
        """Get high impact news alerts"""
        
        recent_alerts = []
        current_time = datetime.now()
        
        for alert in self.alert_history:
            if current_time - alert.timestamp <= timedelta(hours=24):  # Last 24 hours
                recent_alerts.append(alert)
        
        return recent_alerts
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get news monitoring summary"""
        
        current_signals = self.get_latest_signals()
        recent_alerts = self.get_high_impact_alerts()
        
        return {
            'active_signals': len(current_signals),
            'recent_alerts': len(recent_alerts),
            'top_signal': current_signals[0] if current_signals else None,
            'last_update': datetime.now().isoformat(),
            'signal_summary': [
                {
                    'symbol': s.symbol,
                    'signal': s.signal_type,
                    'confidence': s.confidence,
                    'impact': s.impact_score,
                    'urgency': s.urgency
                }
                for s in current_signals[:5]
            ]
        }

# Factory function
def create_news_monitor(config: Dict[str, Any] = None) -> BreakingNewsMonitor:
    """Create breaking news monitor"""
    
    default_config = {
        'aggregator': {
            'enabled_sources': ['CoinDesk', 'CoinTelegraph', 'Bitcoin.com']
        },
        'analyzer': {
            'sentiment_threshold': 0.3,
            'impact_threshold': 0.5,
            'signal_confidence_threshold': 0.7
        },
        'update_interval': 300,  # 5 minutes
        'alert_threshold': 0.8
    }
    
    if config:
        # Deep merge configs
        for key, value in config.items():
            if key in default_config and isinstance(value, dict):
                default_config[key].update(value)
            else:
                default_config[key] = value
    
    return BreakingNewsMonitor(default_config)

# Test function
async def test_news_integration():
    """Test breaking news integration"""
    
    print("📰 Testing Breaking News Integration...")
    
    if not RSS_AVAILABLE:
        print("⚠️ RSS parsing not available - install feedparser")
        return False
    
    # Create monitor
    config = {
        'aggregator': {
            'enabled_sources': ['CoinDesk']  # Test with one source
        },
        'analyzer': {
            'sentiment_threshold': 0.2,  # Lower for testing
            'impact_threshold': 0.3,
            'signal_confidence_threshold': 0.5
        }
    }
    
    monitor = create_news_monitor(config)
    
    test_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    try:
        # Start monitoring
        signals = await monitor.start_monitoring(test_symbols)
        
        print(f"📊 Generated {len(signals)} news-based signals")
        
        # Show signals
        for i, signal in enumerate(signals[:3], 1):
            print(f"\n📡 Signal #{i}:")
            print(f"   Symbol: {signal.symbol}")
            print(f"   Signal: {signal.signal_type}")
            print(f"   Confidence: {signal.confidence:.2f}")
            print(f"   Impact: {signal.impact_score:.2f}")
            print(f"   Urgency: {signal.urgency}")
            print(f"   News Items: {len(signal.news_items)}")
            print(f"   Reasoning: {signal.reasoning[:100]}...")
        
        # Show summary
        summary = monitor.get_monitoring_summary()
        print(f"\n📈 Monitoring Summary:")
        print(f"   Active Signals: {summary['active_signals']}")
        print(f"   Recent Alerts: {summary['recent_alerts']}")
        print(f"   Last Update: {summary['last_update']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run test
    asyncio.run(test_news_integration())