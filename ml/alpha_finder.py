"""
Alpha Finder - Entdeckung von "unsichtbaren" Alpha-Faktoren
Analysiert alternative Datenquellen für einzigartige Trading-Signale
"""

import logging
import asyncio
import aiohttp
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import re
from dataclasses import dataclass
from collections import defaultdict
import time

# Try to import sentiment analysis libraries
try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

try:
    import tweepy
    HAS_TWEEPY = True
except ImportError:
    HAS_TWEEPY = False

try:
    import praw
    HAS_PRAW = True
except ImportError:
    HAS_PRAW = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class AlphaSignal:
    """Datenklasse für Alpha-Signale"""
    signal_type: str
    symbol: str
    strength: float  # -1 bis 1
    confidence: float  # 0 bis 1
    timestamp: datetime
    source: str
    metadata: Dict[str, Any]


@dataclass
class SentimentData:
    """Datenklasse für Sentiment-Daten"""
    symbol: str
    sentiment_score: float
    volume: int
    source: str
    timestamp: datetime
    keywords: List[str]


class AlphaFinder:
    """
    Haupt-Klasse für die Entdeckung von Alpha-Faktoren
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # API Keys (optional)
        self.api_keys = {
            'twitter_bearer_token': self.config.get('twitter_bearer_token'),
            'reddit_client_id': self.config.get('reddit_client_id'),
            'reddit_client_secret': self.config.get('reddit_client_secret'),
            'reddit_user_agent': self.config.get('reddit_user_agent', 'AlphaFinder/1.0'),
            'coingecko_api_key': self.config.get('coingecko_api_key'),
            'binance_api_key': self.config.get('binance_api_key'),
            'binance_api_secret': self.config.get('binance_api_secret')
        }
        
        # Konfiguration
        self.symbols = self.config.get('symbols', ['BTC', 'ETH', 'ADA', 'SOL', 'DOT'])
        self.lookback_hours = self.config.get('lookback_hours', 24)
        self.min_confidence = self.config.get('min_confidence', 0.3)
        self.max_api_calls_per_hour = self.config.get('max_api_calls_per_hour', 100)
        
        # Cache für Alpha-Signale
        self.alpha_signals = []
        self.sentiment_cache = {}
        self.funding_rate_cache = {}
        self.orderbook_cache = {}
        self.price_differences_cache = {}
        
        # Rate limiting
        self.api_call_counts = defaultdict(int)
        self.last_reset_time = datetime.now()
        
        # Exchanges für Cross-Exchange Arbitrage
        self.exchanges = ['binance', 'coinbase', 'kraken', 'bybit', 'okx']
        
        self.logger.info("AlphaFinder initialized")
    
    async def find_alpha_signals(self) -> List[AlphaSignal]:
        """
        Haupt-Methode zur Entdeckung von Alpha-Faktoren
        """
        try:
            signals = []
            
            # Reset rate limiting wenn nötig
            self._reset_rate_limiting()
            
            # Sammle Alpha-Signale parallel
            tasks = []
            
            # Sentiment Analysis
            if self._has_sentiment_apis():
                tasks.append(self._analyze_sentiment())
            
            # Funding Rate Anomalien
            tasks.append(self._analyze_funding_rates())
            
            # Order Book Imbalances
            tasks.append(self._analyze_orderbook_imbalances())
            
            # Cross-Exchange Preisdifferenzen
            tasks.append(self._analyze_price_differences())
            
            # On-Chain Metriken (falls verfügbar)
            if self._has_onchain_apis():
                tasks.append(self._analyze_onchain_metrics())
            
            # Führe alle Analysen parallel aus
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Sammle alle Signale
            for result in results:
                if isinstance(result, list):
                    signals.extend(result)
                elif isinstance(result, Exception):
                    self.logger.error(f"Error in alpha analysis: {result}")
            
            # Filtere und bewerte Signale
            filtered_signals = self._filter_and_rank_signals(signals)
            
            # Cache aktualisieren
            self.alpha_signals = filtered_signals
            
            self.logger.info(f"Found {len(filtered_signals)} alpha signals")
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"Error finding alpha signals: {e}")
            return []
    
    async def _analyze_sentiment(self) -> List[AlphaSignal]:
        """
        Analysiert Sentiment aus sozialen Medien
        """
        try:
            signals = []
            
            # Twitter Sentiment
            if self.api_keys['twitter_bearer_token'] and HAS_TWEEPY:
                twitter_signals = await self._analyze_twitter_sentiment()
                signals.extend(twitter_signals)
            
            # Reddit Sentiment
            if self.api_keys['reddit_client_id'] and HAS_PRAW:
                reddit_signals = await self._analyze_reddit_sentiment()
                signals.extend(reddit_signals)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return []
    
    async def _analyze_twitter_sentiment(self) -> List[AlphaSignal]:
        """
        Analysiert Twitter Sentiment für Kryptowährungen
        """
        try:
            if not self._can_make_api_call('twitter'):
                return []
            
            signals = []
            
            # Twitter API Client
            client = tweepy.Client(bearer_token=self.api_keys['twitter_bearer_token'])
            
            for symbol in self.symbols:
                try:
                    # Suche nach Tweets
                    query = f"{symbol} (crypto OR cryptocurrency OR bitcoin OR altcoin) -is:retweet lang:en"
                    tweets = client.search_recent_tweets(
                        query=query,
                        max_results=100,
                        tweet_fields=['created_at', 'public_metrics', 'context_annotations']
                    )
                    
                    if tweets.data:
                        sentiment_scores = []
                        total_engagement = 0
                        
                        for tweet in tweets.data:
                            if HAS_TEXTBLOB:
                                # Sentiment Analysis
                                blob = TextBlob(tweet.text)
                                sentiment = blob.sentiment.polarity
                                
                                # Gewichtung basierend auf Engagement
                                engagement = (tweet.public_metrics.get('like_count', 0) + 
                                            tweet.public_metrics.get('retweet_count', 0) + 
                                            tweet.public_metrics.get('reply_count', 0))
                                
                                sentiment_scores.append(sentiment * (1 + engagement / 100))
                                total_engagement += engagement
                        
                        if sentiment_scores:
                            avg_sentiment = np.mean(sentiment_scores)
                            confidence = min(len(sentiment_scores) / 100, 1.0)  # Mehr Tweets = höhere Konfidenz
                            
                            # Erstelle Signal
                            signal = AlphaSignal(
                                signal_type='twitter_sentiment',
                                symbol=symbol,
                                strength=float(avg_sentiment),
                                confidence=confidence,
                                timestamp=datetime.now(),
                                source='twitter',
                                metadata={
                                    'tweet_count': len(sentiment_scores),
                                    'total_engagement': total_engagement,
                                    'avg_sentiment': avg_sentiment
                                }
                            )
                            signals.append(signal)
                    
                    self._increment_api_call('twitter')
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing Twitter sentiment for {symbol}: {e}")
                    continue
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error in Twitter sentiment analysis: {e}")
            return []
    
    async def _analyze_reddit_sentiment(self) -> List[AlphaSignal]:
        """
        Analysiert Reddit Sentiment für Kryptowährungen
        """
        try:
            if not self._can_make_api_call('reddit'):
                return []
            
            signals = []
            
            # Reddit API Client
            reddit = praw.Reddit(
                client_id=self.api_keys['reddit_client_id'],
                client_secret=self.api_keys['reddit_client_secret'],
                user_agent=self.api_keys['reddit_user_agent']
            )
            
            # Relevante Subreddits
            subreddits = ['cryptocurrency', 'bitcoin', 'ethtrader', 'altcoin', 'cryptomarkets']
            
            for symbol in self.symbols:
                try:
                    sentiment_scores = []
                    total_score = 0
                    
                    for subreddit_name in subreddits:
                        subreddit = reddit.subreddit(subreddit_name)
                        
                        # Suche nach Posts
                        for submission in subreddit.search(symbol, time_filter='day', limit=10):
                            if HAS_TEXTBLOB:
                                # Analyse Titel und Text
                                text = f"{submission.title} {submission.selftext}"
                                blob = TextBlob(text)
                                sentiment = blob.sentiment.polarity
                                
                                # Gewichtung basierend auf Upvotes
                                weight = max(1, submission.score / 10)
                                sentiment_scores.append(sentiment * weight)
                                total_score += submission.score
                    
                    if sentiment_scores:
                        avg_sentiment = np.mean(sentiment_scores)
                        confidence = min(len(sentiment_scores) / 50, 1.0)
                        
                        # Erstelle Signal
                        signal = AlphaSignal(
                            signal_type='reddit_sentiment',
                            symbol=symbol,
                            strength=float(avg_sentiment),
                            confidence=confidence,
                            timestamp=datetime.now(),
                            source='reddit',
                            metadata={
                                'post_count': len(sentiment_scores),
                                'total_upvotes': total_score,
                                'avg_sentiment': avg_sentiment
                            }
                        )
                        signals.append(signal)
                    
                    self._increment_api_call('reddit')
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing Reddit sentiment for {symbol}: {e}")
                    continue
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error in Reddit sentiment analysis: {e}")
            return []
    
    async def _analyze_funding_rates(self) -> List[AlphaSignal]:
        """
        Analysiert Funding Rate Anomalien
        """
        try:
            signals = []
            
            # Binance Funding Rates
            funding_data = await self._fetch_funding_rates()
            
            for symbol_data in funding_data:
                symbol = symbol_data['symbol']
                current_rate = symbol_data['funding_rate']
                
                # Berechne historische Statistiken
                historical_rates = await self._fetch_historical_funding_rates(symbol)
                
                if len(historical_rates) >= 10:
                    mean_rate = np.mean(historical_rates)
                    std_rate = np.std(historical_rates)
                    
                    # Z-Score berechnen
                    z_score = (current_rate - mean_rate) / std_rate if std_rate > 0 else 0
                    
                    # Anomalie-Erkennung
                    if abs(z_score) > 2.0:  # Signifikante Abweichung
                        strength = np.tanh(z_score / 3.0)  # Normalisierung zwischen -1 und 1
                        confidence = min(abs(z_score) / 3.0, 1.0)
                        
                        signal = AlphaSignal(
                            signal_type='funding_rate_anomaly',
                            symbol=symbol,
                            strength=strength,
                            confidence=confidence,
                            timestamp=datetime.now(),
                            source='binance_funding',
                            metadata={
                                'current_rate': current_rate,
                                'historical_mean': mean_rate,
                                'z_score': z_score,
                                'data_points': len(historical_rates)
                            }
                        )
                        signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error analyzing funding rates: {e}")
            return []
    
    async def _analyze_orderbook_imbalances(self) -> List[AlphaSignal]:
        """
        Analysiert Order Book Imbalances
        """
        try:
            signals = []
            
            for symbol in self.symbols:
                try:
                    # Fetch Order Book
                    orderbook = await self._fetch_orderbook(symbol)
                    
                    if orderbook:
                        # Berechne Imbalance-Metriken
                        imbalance_metrics = self._calculate_orderbook_imbalance(orderbook)
                        
                        # Signifikante Imbalance?
                        if abs(imbalance_metrics['imbalance_ratio']) > 0.1:
                            strength = np.tanh(imbalance_metrics['imbalance_ratio'] * 5)
                            confidence = min(imbalance_metrics['depth_quality'], 1.0)
                            
                            signal = AlphaSignal(
                                signal_type='orderbook_imbalance',
                                symbol=symbol,
                                strength=strength,
                                confidence=confidence,
                                timestamp=datetime.now(),
                                source='orderbook_analysis',
                                metadata=imbalance_metrics
                            )
                            signals.append(signal)
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing orderbook for {symbol}: {e}")
                    continue
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error analyzing orderbook imbalances: {e}")
            return []
    
    async def _analyze_price_differences(self) -> List[AlphaSignal]:
        """
        Analysiert Cross-Exchange Preisdifferenzen
        """
        try:
            signals = []
            
            for symbol in self.symbols:
                try:
                    # Fetch Preise von verschiedenen Exchanges
                    prices = await self._fetch_cross_exchange_prices(symbol)
                    
                    if len(prices) >= 2:
                        # Berechne Preisdifferenzen
                        price_analysis = self._analyze_price_spread(prices)
                        
                        # Signifikante Arbitrage-Möglichkeit?
                        if price_analysis['max_spread'] > 0.005:  # 0.5% Spread
                            strength = min(price_analysis['max_spread'] * 100, 1.0)
                            confidence = price_analysis['data_quality']
                            
                            signal = AlphaSignal(
                                signal_type='cross_exchange_arbitrage',
                                symbol=symbol,
                                strength=strength,
                                confidence=confidence,
                                timestamp=datetime.now(),
                                source='price_analysis',
                                metadata=price_analysis
                            )
                            signals.append(signal)
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing price differences for {symbol}: {e}")
                    continue
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error analyzing price differences: {e}")
            return []
    
    async def _analyze_onchain_metrics(self) -> List[AlphaSignal]:
        """
        Analysiert On-Chain Metriken (falls APIs verfügbar)
        """
        try:
            signals = []
            
            # Vereinfachte On-Chain Analyse
            # In einer echten Implementation würde man APIs wie Glassnode, IntoTheBlock etc. verwenden
            
            for symbol in ['BTC', 'ETH']:  # Nur für Hauptcoins
                try:
                    # Simuliere On-Chain Metriken
                    onchain_data = await self._fetch_onchain_data(symbol)
                    
                    if onchain_data:
                        # Analysiere Metriken
                        analysis = self._analyze_onchain_data(onchain_data)
                        
                        if analysis['anomaly_score'] > 0.5:
                            signal = AlphaSignal(
                                signal_type='onchain_anomaly',
                                symbol=symbol,
                                strength=analysis['signal_strength'],
                                confidence=analysis['confidence'],
                                timestamp=datetime.now(),
                                source='onchain_analysis',
                                metadata=analysis
                            )
                            signals.append(signal)
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing on-chain metrics for {symbol}: {e}")
                    continue
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error analyzing on-chain metrics: {e}")
            return []
    
    def _filter_and_rank_signals(self, signals: List[AlphaSignal]) -> List[AlphaSignal]:
        """
        Filtert und rankt Alpha-Signale nach Qualität
        """
        try:
            # Filtere schwache Signale
            filtered_signals = [s for s in signals if s.confidence >= self.min_confidence]
            
            # Berechne kombinierte Scores
            for signal in filtered_signals:
                signal.combined_score = signal.strength * signal.confidence
            
            # Sortiere nach kombiniertem Score
            filtered_signals.sort(key=lambda x: abs(x.combined_score), reverse=True)
            
            # Entferne Duplikate (gleicher Typ und Symbol)
            unique_signals = []
            seen = set()
            
            for signal in filtered_signals:
                key = (signal.signal_type, signal.symbol)
                if key not in seen:
                    unique_signals.append(signal)
                    seen.add(key)
            
            return unique_signals[:50]  # Top 50 Signale
            
        except Exception as e:
            self.logger.error(f"Error filtering and ranking signals: {e}")
            return signals
    
    async def _fetch_funding_rates(self) -> List[Dict]:
        """
        Holt aktuelle Funding Rates von Binance
        """
        try:
            if not self._can_make_api_call('binance'):
                return []
            
            url = "https://fapi.binance.com/fapi/v1/premiumIndex"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Filtere relevante Symbole
                        relevant_data = []
                        for item in data:
                            symbol = item['symbol'].replace('USDT', '')
                            if symbol in self.symbols:
                                relevant_data.append({
                                    'symbol': symbol,
                                    'funding_rate': float(item['lastFundingRate']),
                                    'next_funding_time': item['nextFundingTime']
                                })
                        
                        self._increment_api_call('binance')
                        return relevant_data
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error fetching funding rates: {e}")
            return []
    
    async def _fetch_historical_funding_rates(self, symbol: str) -> List[float]:
        """
        Holt historische Funding Rates
        """
        try:
            # Simuliere historische Daten (in echter Implementation von API)
            # Hier würde man die Binance API für historische Funding Rates verwenden
            return np.random.normal(0.0001, 0.0005, 50).tolist()
            
        except Exception as e:
            self.logger.error(f"Error fetching historical funding rates: {e}")
            return []
    
    async def _fetch_orderbook(self, symbol: str) -> Optional[Dict]:
        """
        Holt Order Book Daten
        """
        try:
            if not self._can_make_api_call('binance'):
                return None
            
            url = f"https://api.binance.com/api/v3/depth?symbol={symbol}USDT&limit=100"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        self._increment_api_call('binance')
                        return await response.json()
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching orderbook: {e}")
            return None
    
    def _calculate_orderbook_imbalance(self, orderbook: Dict) -> Dict:
        """
        Berechnet Order Book Imbalance Metriken
        """
        try:
            bids = orderbook['bids']
            asks = orderbook['asks']
            
            # Berechne Volumina
            bid_volume = sum(float(bid[1]) for bid in bids[:10])  # Top 10 levels
            ask_volume = sum(float(ask[1]) for ask in asks[:10])
            
            # Berechne Imbalance
            total_volume = bid_volume + ask_volume
            imbalance_ratio = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
            
            # Berechne Spread
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            spread = (best_ask - best_bid) / best_bid
            
            # Berechne Depth Quality
            depth_quality = min(len(bids), len(asks)) / 100
            
            return {
                'imbalance_ratio': imbalance_ratio,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'spread': spread,
                'depth_quality': depth_quality,
                'best_bid': best_bid,
                'best_ask': best_ask
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating orderbook imbalance: {e}")
            return {}
    
    async def _fetch_cross_exchange_prices(self, symbol: str) -> Dict[str, float]:
        """
        Holt Preise von verschiedenen Exchanges
        """
        try:
            prices = {}
            
            # Binance
            if self._can_make_api_call('binance'):
                binance_price = await self._fetch_binance_price(symbol)
                if binance_price:
                    prices['binance'] = binance_price
            
            # Coinbase (vereinfacht)
            if self._can_make_api_call('coinbase'):
                coinbase_price = await self._fetch_coinbase_price(symbol)
                if coinbase_price:
                    prices['coinbase'] = coinbase_price
            
            # Simuliere andere Exchanges
            if prices:
                base_price = list(prices.values())[0]
                # Simuliere leichte Preisunterschiede
                if 'kraken' not in prices:
                    prices['kraken'] = base_price * (1 + np.random.uniform(-0.002, 0.002))
                if 'bybit' not in prices:
                    prices['bybit'] = base_price * (1 + np.random.uniform(-0.003, 0.003))
            
            return prices
            
        except Exception as e:
            self.logger.error(f"Error fetching cross-exchange prices: {e}")
            return {}
    
    async def _fetch_binance_price(self, symbol: str) -> Optional[float]:
        """
        Holt Preis von Binance
        """
        try:
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}USDT"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        self._increment_api_call('binance')
                        return float(data['price'])
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching Binance price: {e}")
            return None
    
    async def _fetch_coinbase_price(self, symbol: str) -> Optional[float]:
        """
        Holt Preis von Coinbase
        """
        try:
            url = f"https://api.coinbase.com/v2/exchange-rates?currency={symbol}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        usd_rate = data['data']['rates'].get('USD')
                        if usd_rate:
                            self._increment_api_call('coinbase')
                            return float(usd_rate)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching Coinbase price: {e}")
            return None
    
    def _analyze_price_spread(self, prices: Dict[str, float]) -> Dict:
        """
        Analysiert Preisdifferenzen zwischen Exchanges
        """
        try:
            price_values = list(prices.values())
            
            min_price = min(price_values)
            max_price = max(price_values)
            
            max_spread = (max_price - min_price) / min_price
            
            # Finde beste Arbitrage-Möglichkeit
            best_buy_exchange = min(prices, key=prices.get)
            best_sell_exchange = max(prices, key=prices.get)
            
            return {
                'max_spread': max_spread,
                'min_price': min_price,
                'max_price': max_price,
                'best_buy_exchange': best_buy_exchange,
                'best_sell_exchange': best_sell_exchange,
                'data_quality': len(prices) / len(self.exchanges),
                'exchanges': list(prices.keys())
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing price spread: {e}")
            return {}
    
    async def _fetch_onchain_data(self, symbol: str) -> Optional[Dict]:
        """
        Holt On-Chain Daten (simuliert)
        """
        try:
            # Simuliere On-Chain Daten
            # In echter Implementation würde man APIs wie Glassnode verwenden
            return {
                'active_addresses': np.random.randint(500000, 1000000),
                'transaction_volume': np.random.uniform(1e9, 5e9),
                'whale_movements': np.random.randint(0, 10),
                'exchange_flows': np.random.uniform(-1e6, 1e6),
                'mvrv_ratio': np.random.uniform(0.8, 3.2)
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching on-chain data: {e}")
            return None
    
    def _analyze_onchain_data(self, data: Dict) -> Dict:
        """
        Analysiert On-Chain Daten
        """
        try:
            # Vereinfachte On-Chain Analyse
            anomaly_score = 0
            
            # MVRV Ratio Analyse
            mvrv = data.get('mvrv_ratio', 1.0)
            if mvrv < 1.0:  # Undervalued
                anomaly_score += 0.3
            elif mvrv > 3.0:  # Overvalued
                anomaly_score += 0.2
            
            # Whale Movements
            whale_movements = data.get('whale_movements', 0)
            if whale_movements > 5:
                anomaly_score += 0.4
            
            # Exchange Flows
            exchange_flows = data.get('exchange_flows', 0)
            if abs(exchange_flows) > 500000:
                anomaly_score += 0.3
            
            signal_strength = np.tanh(anomaly_score)
            confidence = min(anomaly_score, 1.0)
            
            return {
                'anomaly_score': anomaly_score,
                'signal_strength': signal_strength,
                'confidence': confidence,
                'mvrv_ratio': mvrv,
                'whale_movements': whale_movements,
                'exchange_flows': exchange_flows
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing on-chain data: {e}")
            return {}
    
    def _has_sentiment_apis(self) -> bool:
        """Prüft ob Sentiment APIs verfügbar sind"""
        return (self.api_keys['twitter_bearer_token'] is not None or 
                self.api_keys['reddit_client_id'] is not None)
    
    def _has_onchain_apis(self) -> bool:
        """Prüft ob On-Chain APIs verfügbar sind"""
        return self.api_keys['coingecko_api_key'] is not None
    
    def _can_make_api_call(self, api_name: str) -> bool:
        """Prüft Rate Limiting"""
        return self.api_call_counts[api_name] < self.max_api_calls_per_hour
    
    def _increment_api_call(self, api_name: str) -> None:
        """Erhöht API Call Counter"""
        self.api_call_counts[api_name] += 1
    
    def _reset_rate_limiting(self) -> None:
        """Reset Rate Limiting wenn nötig"""
        now = datetime.now()
        if now - self.last_reset_time >= timedelta(hours=1):
            self.api_call_counts.clear()
            self.last_reset_time = now
    
    def get_alpha_summary(self) -> Dict[str, Any]:
        """
        Gibt eine Zusammenfassung der Alpha-Signale zurück
        """
        try:
            if not self.alpha_signals:
                return {'message': 'No alpha signals found'}
            
            # Gruppiere nach Signal-Typ
            signal_types = defaultdict(list)
            for signal in self.alpha_signals:
                signal_types[signal.signal_type].append(signal)
            
            # Berechne Statistiken
            summary = {
                'total_signals': len(self.alpha_signals),
                'signal_types': dict(signal_types),
                'average_confidence': np.mean([s.confidence for s in self.alpha_signals]),
                'strongest_signals': [
                    {
                        'type': s.signal_type,
                        'symbol': s.symbol,
                        'strength': s.strength,
                        'confidence': s.confidence
                    } for s in self.alpha_signals[:5]
                ],
                'by_symbol': defaultdict(list)
            }
            
            # Gruppiere nach Symbol
            for signal in self.alpha_signals:
                summary['by_symbol'][signal.symbol].append({
                    'type': signal.signal_type,
                    'strength': signal.strength,
                    'confidence': signal.confidence
                })
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating alpha summary: {e}")
            return {'error': str(e)}
    
    def get_actionable_signals(self, min_strength: float = 0.3) -> List[Dict[str, Any]]:
        """
        Gibt umsetzbare Alpha-Signale zurück
        """
        try:
            actionable = []
            
            for signal in self.alpha_signals:
                if abs(signal.strength) >= min_strength:
                    actionable.append({
                        'symbol': signal.symbol,
                        'signal_type': signal.signal_type,
                        'action': 'buy' if signal.strength > 0 else 'sell',
                        'strength': abs(signal.strength),
                        'confidence': signal.confidence,
                        'priority': abs(signal.strength) * signal.confidence,
                        'source': signal.source,
                        'timestamp': signal.timestamp.isoformat()
                    })
            
            # Sortiere nach Priorität
            actionable.sort(key=lambda x: x['priority'], reverse=True)
            
            return actionable
            
        except Exception as e:
            self.logger.error(f"Error getting actionable signals: {e}")
            return []