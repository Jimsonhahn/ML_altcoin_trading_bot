# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Collector - Market Data Collection System
=============================================

Collects and manages market data from various sources:
- Real-time price data
- Historical candlestick data
- Order book data
- Trade data
- Market statistics
"""

import logging
import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import ccxt
import requests
from collections import deque
import threading
import queue

logger = logging.getLogger(__name__)


class DataCollector:
    """Market data collection and management system"""

    def __init__(self, settings):
        """Initialize Data Collector"""
        self.settings = settings
        self.data_config = settings.get('data_sources', {})

        # Data sources configuration
        self.primary_source = self.data_config.get('primary', 'binance')
        self.fallback_sources = self.data_config.get('fallback', ['coingecko', 'cryptocompare'])

        # Exchange connections
        self.exchanges = {}
        self._initialize_exchanges()

        # Data storage
        self.price_cache = {}  # symbol -> price data
        self.orderbook_cache = {}  # symbol -> orderbook
        self.trade_cache = {}  # symbol -> recent trades
        self.stats_cache = {}  # symbol -> 24h stats

        # Real-time data streams
        self.price_streams = {}  # symbol -> deque of prices
        self.stream_max_length = 1000

        # Data collection threads
        self.collector_threads = {}
        self.running = False
        self.data_queue = queue.Queue()

        # Historical data cache
        self.historical_cache = {}
        self.cache_expiry = 3600  # 1 hour

        logger.info(f"Data Collector initialized with primary source: {self.primary_source}")

    def start(self):
        """Start data collection"""
        self.running = True
        logger.info("Data Collector started")

    def stop(self):
        """Stop data collection"""
        self.running = False

        # Stop all collector threads
        for thread in self.collector_threads.values():
            if thread.is_alive():
                thread.join(timeout=5)

        logger.info("Data Collector stopped")

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        # Check cache first
        if symbol in self.price_cache:
            cache_data = self.price_cache[symbol]
            if time.time() - cache_data['timestamp'] < 5:  # 5 second cache
                return cache_data['price']

        # Fetch fresh price
        try:
            exchange = self._get_exchange(self.primary_source)
            ticker = exchange.fetch_ticker(symbol)

            price = ticker['last']

            # Update cache
            self.price_cache[symbol] = {
                'price': price,
                'timestamp': time.time(),
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['baseVolume']
            }

            return price

        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")

            # Try fallback sources
            for source in self.fallback_sources:
                price = self._get_price_from_fallback(symbol, source)
                if price:
                    return price

            return None

    def get_historical_data(self, symbol: str, timeframe: str,
                            limit: int = 100, since: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """Get historical candlestick data"""
        # Generate cache key
        cache_key = f"{symbol}_{timeframe}_{limit}"

        # Check cache
        if cache_key in self.historical_cache:
            cache_entry = self.historical_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.cache_expiry:
                logger.debug(f"Returning cached data for {cache_key}")
                return cache_entry['data']

        try:
            exchange = self._get_exchange(self.primary_source)

            # Convert since to timestamp
            since_ts = None
            if since:
                since_ts = int(since.timestamp() * 1000)

            # Fetch data
            ohlcv = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                since=since_ts
            )

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Add technical indicators
            df = self._add_basic_indicators(df)

            # Cache the data
            self.historical_cache[cache_key] = {
                'data': df,
                'timestamp': time.time()
            }

            # Save to file for backtesting
            self._save_historical_data(symbol, timeframe, df)

            return df

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")

            # Try loading from file
            df = self._load_historical_data(symbol, timeframe)
            if df is not None and len(df) > 0:
                return df.tail(limit)

            return None

    def get_orderbook(self, symbol: str, limit: int = 10) -> Optional[Dict[str, Any]]:
        """Get order book data"""
        try:
            exchange = self._get_exchange(self.primary_source)
            orderbook = exchange.fetch_order_book(symbol, limit)

            # Process orderbook
            processed = {
                'symbol': symbol,
                'timestamp': orderbook['timestamp'],
                'bids': orderbook['bids'][:limit],
                'asks': orderbook['asks'][:limit],
                'bid_volume': sum(bid[1] for bid in orderbook['bids'][:limit]),
                'ask_volume': sum(ask[1] for ask in orderbook['asks'][:limit]),
                'spread': orderbook['asks'][0][0] - orderbook['bids'][0][0] if orderbook['asks'] and orderbook[
                    'bids'] else 0,
                'mid_price': (orderbook['asks'][0][0] + orderbook['bids'][0][0]) / 2 if orderbook['asks'] and orderbook[
                    'bids'] else 0
            }

            # Update cache
            self.orderbook_cache[symbol] = processed

            return processed

        except Exception as e:
            logger.error(f"Error fetching orderbook for {symbol}: {e}")
            return None

    def get_recent_trades(self, symbol: str, limit: int = 50) -> Optional[List[Dict[str, Any]]]:
        """Get recent trades"""
        try:
            exchange = self._get_exchange(self.primary_source)
            trades = exchange.fetch_trades(symbol, limit=limit)

            # Process trades
            processed_trades = []
            for trade in trades:
                processed_trades.append({
                    'id': trade['id'],
                    'timestamp': trade['timestamp'],
                    'price': trade['price'],
                    'amount': trade['amount'],
                    'side': trade['side'],
                    'cost': trade['cost']
                })

            # Update cache
            self.trade_cache[symbol] = processed_trades

            return processed_trades

        except Exception as e:
            logger.error(f"Error fetching trades for {symbol}: {e}")
            return None

    def get_market_stats(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get 24h market statistics"""
        # Check cache
        if symbol in self.stats_cache:
            cache_data = self.stats_cache[symbol]
            if time.time() - cache_data['timestamp'] < 60:  # 1 minute cache
                return cache_data['stats']

        try:
            exchange = self._get_exchange(self.primary_source)
            ticker = exchange.fetch_ticker(symbol)

            stats = {
                'symbol': symbol,
                'last_price': ticker['last'],
                'price_change_24h': ticker['change'],
                'price_change_pct_24h': ticker['percentage'],
                'high_24h': ticker['high'],
                'low_24h': ticker['low'],
                'volume_24h': ticker['baseVolume'],
                'quote_volume_24h': ticker['quoteVolume'],
                'vwap_24h': ticker.get('vwap', 0),
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'timestamp': ticker['timestamp']
            }

            # Update cache
            self.stats_cache[symbol] = {
                'stats': stats,
                'timestamp': time.time()
            }

            return stats

        except Exception as e:
            logger.error(f"Error fetching market stats for {symbol}: {e}")
            return None

    def start_price_stream(self, symbol: str, callback=None):
        """Start real-time price streaming for a symbol"""
        if symbol in self.price_streams:
            logger.warning(f"Price stream already active for {symbol}")
            return

        # Initialize price stream
        self.price_streams[symbol] = deque(maxlen=self.stream_max_length)

        # Start collector thread
        thread = threading.Thread(
            target=self._price_stream_worker,
            args=(symbol, callback),
            daemon=True
        )
        self.collector_threads[f"price_{symbol}"] = thread
        thread.start()

        logger.info(f"Started price stream for {symbol}")

    def stop_price_stream(self, symbol: str):
        """Stop price streaming for a symbol"""
        if symbol in self.price_streams:
            del self.price_streams[symbol]
            logger.info(f"Stopped price stream for {symbol}")

    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get prices for multiple symbols efficiently"""
        prices = {}

        try:
            exchange = self._get_exchange(self.primary_source)

            # Fetch all tickers at once (more efficient)
            tickers = exchange.fetch_tickers(symbols)

            for symbol in symbols:
                if symbol in tickers:
                    prices[symbol] = tickers[symbol]['last']
                else:
                    # Fallback to individual fetch
                    price = self.get_current_price(symbol)
                    if price:
                        prices[symbol] = price

        except Exception as e:
            logger.error(f"Error fetching multiple prices: {e}")

            # Fallback to individual fetches
            for symbol in symbols:
                price = self.get_current_price(symbol)
                if price:
                    prices[symbol] = price

        return prices

    def get_funding_rate(self, symbol: str) -> Optional[float]:
        """Get funding rate for perpetual futures"""
        try:
            exchange = self._get_exchange(self.primary_source)

            if hasattr(exchange, 'fetch_funding_rate'):
                funding = exchange.fetch_funding_rate(symbol)
                return funding['fundingRate']

            return None

        except Exception as e:
            logger.error(f"Error fetching funding rate for {symbol}: {e}")
            return None

    def get_market_depth(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get market depth analysis"""
        orderbook = self.get_orderbook(symbol, limit=50)

        if not orderbook:
            return None

        # Calculate depth metrics
        bid_depth = {}
        ask_depth = {}

        # Calculate cumulative volume at different price levels
        price_levels = [0.001, 0.005, 0.01, 0.02, 0.05]  # 0.1%, 0.5%, 1%, 2%, 5%

        mid_price = orderbook['mid_price']

        for level in price_levels:
            bid_price = mid_price * (1 - level)
            ask_price = mid_price * (1 + level)

            bid_volume = sum(bid[1] for bid in orderbook['bids'] if bid[0] >= bid_price)
            ask_volume = sum(ask[1] for ask in orderbook['asks'] if ask[0] <= ask_price)

            bid_depth[f"{level * 100:.1f}%"] = bid_volume
            ask_depth[f"{level * 100:.1f}%"] = ask_volume

        return {
            'symbol': symbol,
            'mid_price': mid_price,
            'spread': orderbook['spread'],
            'spread_pct': (orderbook['spread'] / mid_price) * 100 if mid_price > 0 else 0,
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'imbalance': (orderbook['bid_volume'] - orderbook['ask_volume']) /
                         (orderbook['bid_volume'] + orderbook['ask_volume'])
            if (orderbook['bid_volume'] + orderbook['ask_volume']) > 0 else 0
        }

    def download_historical_data(self, symbol: str, timeframe: str,
                                 start_date: datetime, end_date: datetime) -> bool:
        """Download and save historical data for backtesting"""
        logger.info(f"Downloading historical data for {symbol} from {start_date} to {end_date}")

        try:
            exchange = self._get_exchange(self.primary_source)

            all_data = []
            current_date = start_date

            while current_date < end_date:
                # Fetch data in chunks
                since_ts = int(current_date.timestamp() * 1000)

                ohlcv = exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since_ts,
                    limit=1000  # Max limit for most exchanges
                )

                if not ohlcv:
                    break

                all_data.extend(ohlcv)

                # Update current date to last candle timestamp
                last_timestamp = ohlcv[-1][0]
                current_date = datetime.fromtimestamp(last_timestamp / 1000)

                # Add small delay to avoid rate limits
                time.sleep(0.5)

                logger.debug(f"Downloaded {len(ohlcv)} candles, total: {len(all_data)}")

            # Convert to DataFrame
            df = pd.DataFrame(
                all_data,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            # Remove duplicates
            df.drop_duplicates(subset=['timestamp'], inplace=True)

            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Sort by timestamp
            df.sort_index(inplace=True)

            # Add indicators
            df = self._add_basic_indicators(df)

            # Save to file
            filename = f"{symbol.replace('/', '_')}_{timeframe}.csv"
            filepath = os.path.join('data/market_data', filename)

            os.makedirs('data/market_data', exist_ok=True)
            df.to_csv(filepath)

            logger.info(f"Saved {len(df)} candles to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error downloading historical data: {e}")
            return False

    # Private methods

    def _initialize_exchanges(self):
        """Initialize exchange connections"""
        # Binance
        try:
            self.exchanges['binance'] = ccxt.binance({
                'enableRateLimit': True,
                'rateLimit': 1200,
                'options': {
                    'defaultType': 'spot'
                }
            })
            logger.info("Initialized Binance connection")
        except Exception as e:
            logger.error(f"Failed to initialize Binance: {e}")

        # Add more exchanges as needed
        # self.exchanges['kraken'] = ccxt.kraken()
        # self.exchanges['coinbase'] = ccxt.coinbase()

    def _get_exchange(self, exchange_name: str):
        """Get exchange instance"""
        if exchange_name in self.exchanges:
            return self.exchanges[exchange_name]

        # Default to binance
        return self.exchanges.get('binance')

    def _get_price_from_fallback(self, symbol: str, source: str) -> Optional[float]:
        """Get price from fallback source"""
        try:
            if source == 'coingecko':
                # Convert symbol format (BTC/USDT -> bitcoin)
                base = symbol.split('/')[0].lower()
                coin_id = self._get_coingecko_id(base)

                url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
                response = requests.get(url, timeout=5)

                if response.status_code == 200:
                    data = response.json()
                    if coin_id in data:
                        return data[coin_id]['usd']

            elif source == 'cryptocompare':
                base, quote = symbol.split('/')
                url = f"https://min-api.cryptocompare.com/data/price?fsym={base}&tsyms={quote}"
                response = requests.get(url, timeout=5)

                if response.status_code == 200:
                    data = response.json()
                    if quote in data:
                        return data[quote]

        except Exception as e:
            logger.error(f"Error fetching from {source}: {e}")

        return None

    def _get_coingecko_id(self, symbol: str) -> str:
        """Convert symbol to CoinGecko ID"""
        mapping = {
            'btc': 'bitcoin',
            'eth': 'ethereum',
            'sol': 'solana',
            'ada': 'cardano',
            'dot': 'polkadot',
            'bnb': 'binancecoin',
            'xrp': 'ripple',
            'doge': 'dogecoin'
        }
        return mapping.get(symbol, symbol)

    def _price_stream_worker(self, symbol: str, callback):
        """Worker thread for price streaming"""
        logger.info(f"Price stream worker started for {symbol}")

        while self.running and symbol in self.price_streams:
            try:
                # Get current price
                price = self.get_current_price(symbol)

                if price:
                    # Add to stream
                    price_data = {
                        'timestamp': datetime.now(),
                        'price': price,
                        'bid': self.price_cache[symbol].get('bid'),
                        'ask': self.price_cache[symbol].get('ask'),
                        'volume': self.price_cache[symbol].get('volume')
                    }

                    self.price_streams[symbol].append(price_data)

                    # Call callback if provided
                    if callback:
                        callback(symbol, price_data)

                # Sleep based on timeframe
                time.sleep(1)  # 1 second updates

            except Exception as e:
                logger.error(f"Error in price stream worker for {symbol}: {e}")
                time.sleep(5)  # Wait longer on error

        logger.info(f"Price stream worker stopped for {symbol}")

    def _add_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators to DataFrame"""
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()

        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()

        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']

        # RSI
        df['rsi'] = self._calculate_rsi(df['close'])

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # Price change
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['close'].diff()

        # High-Low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close']

        # Volatility
        df['volatility'] = df['price_change'].rolling(window=20).std()

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _save_historical_data(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Save historical data to file"""
        try:
            filename = f"{symbol.replace('/', '_')}_{timeframe}_latest.csv"
            filepath = os.path.join('data/market_data', filename)

            os.makedirs('data/market_data', exist_ok=True)

            # Save only recent data to avoid huge files
            df.tail(10000).to_csv(filepath)

        except Exception as e:
            logger.error(f"Error saving historical data: {e}")

    def _load_historical_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load historical data from file"""
        try:
            filename = f"{symbol.replace('/', '_')}_{timeframe}.csv"
            filepath = os.path.join('data/market_data', filename)

            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                return df

        except Exception as e:
            logger.error(f"Error loading historical data: {e}")

        return None

    def get_data_status(self) -> Dict[str, Any]:
        """Get data collector status"""
        return {
            'running': self.running,
            'primary_source': self.primary_source,
            'active_exchanges': list(self.exchanges.keys()),
            'price_streams': list(self.price_streams.keys()),
            'cached_symbols': list(self.price_cache.keys()),
            'cache_size': {
                'prices': len(self.price_cache),
                'orderbooks': len(self.orderbook_cache),
                'trades': len(self.trade_cache),
                'stats': len(self.stats_cache),
                'historical': len(self.historical_cache)
            },
            'active_threads': len([t for t in self.collector_threads.values() if t.is_alive()])
        }


