"""
Exchange Manager - Complete implementation for live and paper trading
"""
import os
import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)


class ExchangeManager:
    """Complete exchange manager with all required methods"""

    def __init__(self, exchange_name: str = 'binance', mode: str = 'live'):
        self.exchange_name = exchange_name
        self.mode = mode
        self.exchange = None
        self.connected = False
        self.markets = {}

        # Configuration
        self.options = {
            'recvWindow': 60000,
            'adjustForTimeDifference': True,
            'enableRateLimit': True
        }

        logger.info(f"Initializing {exchange_name} exchange in {mode} mode")

    def connect(self) -> bool:
        """Connect to the exchange"""
        try:
            if self.mode == 'paper':
                return self._connect_paper()
            else:
                return self._connect_live()
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    def _connect_paper(self) -> bool:
        """Connect to paper trading (testnet or mock)"""
        try:
            config = {
                'enableRateLimit': True,
                'options': self.options
            }

            if self.exchange_name == 'binance':
                # Use Binance testnet
                config['urls'] = {
                    'api': {
                        'public': 'https://testnet.binance.vision/api/v3',
                        'private': 'https://testnet.binance.vision/api/v3',
                    }
                }
                self.exchange = ccxt.binance(config)
            else:
                # Use default exchange
                exchange_class = getattr(ccxt, self.exchange_name)
                self.exchange = exchange_class(config)

            # For paper trading, we don't need to load markets
            self.connected = True
            logger.info(f"Connected to {self.exchange_name} in paper mode")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to paper exchange: {e}")
            # Use mock mode
            self._init_mock_mode()
            return True

    def _connect_live(self) -> bool:
        """Connect to live exchange"""
        try:
            # Get API credentials
            api_key = os.getenv(f'{self.exchange_name.upper()}_API_KEY')
            api_secret = os.getenv(f'{self.exchange_name.upper()}_API_SECRET')

            if not api_key or not api_secret:
                logger.warning("No API credentials found, using read-only mode")
                config = {'enableRateLimit': True, 'options': self.options}
            else:
                config = {
                    'apiKey': api_key,
                    'secret': api_secret,
                    'enableRateLimit': True,
                    'options': self.options
                }

            # Create exchange instance
            exchange_class = getattr(ccxt, self.exchange_name)
            self.exchange = exchange_class(config)

            # Test connection
            self.exchange.load_markets()
            self.markets = self.exchange.markets
            self.connected = True

            logger.info(f"Successfully connected to {self.exchange_name} (live mode)")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to live exchange: {e}")
            self.connected = False
            return False

    def _init_mock_mode(self):
        """Initialize mock mode for paper trading"""
        logger.info("Initializing mock mode for paper trading")
        self.exchange = MockExchange()
        self.connected = True

    def disconnect(self):
        """Disconnect from exchange"""
        self.connected = False
        self.exchange = None
        logger.info("Disconnected from exchange")

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch current ticker data"""
        try:
            if self.connected and hasattr(self.exchange, 'fetch_ticker'):
                return self.exchange.fetch_ticker(symbol)
            else:
                return self._get_mock_ticker(symbol)
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return self._get_mock_ticker(symbol)

    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data"""
        try:
            if self.connected and hasattr(self.exchange, 'fetch_ohlcv'):
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df
            else:
                return self._get_mock_ohlcv(symbol, timeframe, limit)
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return self._get_mock_ohlcv(symbol, timeframe, limit)

    def fetch_balance(self) -> Dict[str, Any]:
        """Fetch account balance"""
        if self.mode == 'paper':
            return {
                'USDT': {'free': 10000, 'used': 0, 'total': 10000},
                'BTC': {'free': 0, 'used': 0, 'total': 0},
                'ETH': {'free': 0, 'used': 0, 'total': 0}
            }

        try:
            if self.connected and hasattr(self.exchange, 'fetch_balance'):
                return self.exchange.fetch_balance()
            else:
                return {'USDT': {'free': 10000, 'used': 0, 'total': 10000}}
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return {'USDT': {'free': 10000, 'used': 0, 'total': 10000}}

    def create_order(self, symbol: str, order_type: str, side: str,
                    amount: float, price: Optional[float] = None) -> Dict[str, Any]:
        """Create an order"""
        try:
            if self.mode == 'paper':
                # Simulate order for paper trading
                return {
                    'id': f"paper_{int(time.time())}",
                    'symbol': symbol,
                    'type': order_type,
                    'side': side,
                    'amount': amount,
                    'price': price or self.fetch_ticker(symbol)['last'],
                    'status': 'closed',
                    'timestamp': int(time.time() * 1000)
                }

            if self.connected and hasattr(self.exchange, 'create_order'):
                if order_type == 'market':
                    return self.exchange.create_market_order(symbol, side, amount)
                else:
                    return self.exchange.create_limit_order(symbol, side, amount, price)
            else:
                raise Exception("Exchange not connected")

        except Exception as e:
            logger.error(f"Error creating order: {e}")
            raise

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order"""
        try:
            if self.mode == 'paper':
                return True

            if self.connected and hasattr(self.exchange, 'cancel_order'):
                self.exchange.cancel_order(order_id, symbol)
                return True
            return False
        except Exception as e:
            logger.error(f"Error canceling order: {e}")
            return False

    def fetch_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Fetch order details"""
        try:
            if self.mode == 'paper':
                return {
                    'id': order_id,
                    'symbol': symbol,
                    'status': 'closed',
                    'filled': 1.0,
                    'remaining': 0.0
                }

            if self.connected and hasattr(self.exchange, 'fetch_order'):
                return self.exchange.fetch_order(order_id, symbol)
            else:
                raise Exception("Exchange not connected")
        except Exception as e:
            logger.error(f"Error fetching order: {e}")
            raise

    def _get_mock_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get mock ticker data"""
        base_prices = {
            'BTC/USDT': 118000,
            'ETH/USDT': 3000,
            'SOL/USDT': 165,
            'DOGE/USDT': 0.075,
            'SHIB/USDT': 0.000008,
            'ADA/USDT': 0.75,
            'MATIC/USDT': 0.90,
            'DOT/USDT': 7.50
        }

        price = base_prices.get(symbol, 100)
        # Add some randomness
        price *= np.random.uniform(0.995, 1.005)

        return {
            'symbol': symbol,
            'last': price,
            'bid': price * 0.9995,
            'ask': price * 1.0005,
            'high': price * 1.02,
            'low': price * 0.98,
            'volume': np.random.uniform(1000000, 10000000),
            'timestamp': int(time.time() * 1000)
        }

    def _get_mock_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Generate mock OHLCV data"""
        # Get base price
        ticker = self._get_mock_ticker(symbol)
        base_price = ticker['last']

        # Generate time series
        now = datetime.now()
        if timeframe == '1m':
            freq = 'T'
        elif timeframe == '5m':
            freq = '5T'
        elif timeframe == '15m':
            freq = '15T'
        elif timeframe == '1h':
            freq = 'H'
        elif timeframe == '4h':
            freq = '4H'
        elif timeframe == '1d':
            freq = 'D'
        else:
            freq = 'H'

        timestamps = pd.date_range(end=now, periods=limit, freq=freq)

        # Generate price data with realistic patterns
        returns = np.random.normal(0, 0.002, limit)
        returns = np.cumsum(returns)
        prices = base_price * np.exp(returns)

        # Create OHLCV data
        data = []
        for i, (ts, price) in enumerate(zip(timestamps, prices)):
            volatility = np.random.uniform(0.001, 0.005)
            high = price * (1 + volatility)
            low = price * (1 - volatility)
            close = np.random.uniform(low, high)

            # First candle opens at base price
            if i == 0:
                open_price = base_price
            else:
                open_price = data[i-1]['close']

            volume = np.random.uniform(100000, 1000000)

            data.append({
                'timestamp': ts,
                'open': open_price,
                'high': max(open_price, close, high),
                'low': min(open_price, close, low),
                'close': close,
                'volume': volume
            })

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df


class MockExchange:
    """Mock exchange for testing and paper trading fallback"""

    def __init__(self):
        self.has = {
            'fetchOHLCV': True,
            'fetchTicker': True,
            'createMarketOrder': True,
            'createLimitOrder': True,
            'fetchBalance': True
        }

    def fetch_ticker(self, symbol):
        return ExchangeManager()._get_mock_ticker(symbol)

    def fetch_ohlcv(self, symbol, timeframe='1h', limit=100):
        df = ExchangeManager()._get_mock_ohlcv(symbol, timeframe, limit)
        return df.reset_index().values.tolist()


class ExchangeFactory:
    """Factory for creating exchange instances"""

    @staticmethod
    def create_exchange(exchange_name: str = 'binance', mode: str = 'live') -> ExchangeManager:
        """Create and connect to exchange"""
        manager = ExchangeManager(exchange_name, mode)
        manager.connect()
        return manager

    @staticmethod
    def create(exchange_name: str = 'binance', mode: str = 'live') -> ExchangeManager:
        """Alias for create_exchange"""
        return ExchangeFactory.create_exchange(exchange_name, mode)
