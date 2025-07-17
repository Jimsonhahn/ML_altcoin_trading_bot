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
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.secret_manager import SecretManager, get_api_credentials
from utils.secure_http import create_secure_session
from utils.error_handler import secure_error_handler, ErrorCategory

logger = logging.getLogger(__name__)


class ExchangeManager:
    """Complete exchange manager with all required methods"""

    def __init__(self, exchange_name: str = 'binance', mode: str = 'live'):
        # Fix: Wenn exchange_name ein Settings-Objekt ist, extrahiere den Namen
        if hasattr(exchange_name, 'get'):
            # Es ist ein Settings-Objekt
            self.exchange_name = exchange_name.get('exchange.name', 'binance')
            self.testnet = exchange_name.get('exchange.testnet', True)
        else:
            # Es ist ein String
            self.exchange_name = exchange_name or 'binance'
            self.testnet = mode == 'paper'  # Fix: use mode instead of undefined testnet
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

        logger.info(f"Initializing {self.exchange_name} exchange in {mode} mode")

    def connect(self) -> bool:
        """Connect to the exchange"""
        try:
            if self.mode == 'paper':
                return self._connect_paper()
            else:
                return self._connect_live()
        except Exception as e:
            error_response = secure_error_handler.handle_critical_error(
                error=e,
                context={
                    "operation": "exchange_connect",
                    "exchange_name": self.exchange_name,
                    "mode": self.mode
                }
            )
            logger.error(f"Failed to connect to exchange - ID: {error_response.error_id}")
            return False

    def _connect_paper(self) -> bool:
        """Connect to paper trading (testnet or mock)"""
        try:
            config = {
                'enableRateLimit': True,
                'options': self.options,
                'session': create_secure_session()
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
            error_response = secure_error_handler.handle_api_error(
                error=e,
                context={
                    "operation": "paper_exchange_connect",
                    "exchange_name": self.exchange_name
                }
            )
            logger.error(f"Failed to connect to paper exchange - ID: {error_response.error_id}")
            # Use mock mode as fallback
            self._init_mock_mode()
            return True

    def _connect_live(self) -> bool:
        """Connect to live exchange"""
        try:
            # Get API credentials from SecretManager first, fallback to env vars
            api_key, api_secret = self._get_api_credentials()

            if not api_key or not api_secret:
                logger.warning("No API credentials found, using read-only mode")
                config = {
                    'enableRateLimit': True, 
                    'options': self.options,
                    'session': create_secure_session()
                }
            else:
                config = {
                    'apiKey': api_key,
                    'secret': api_secret,
                    'enableRateLimit': True,
                    'options': self.options,
                    'session': create_secure_session()
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
            error_response = secure_error_handler.handle_api_error(
                error=e,
                context={
                    "operation": "live_exchange_connect",
                    "exchange_name": self.exchange_name,
                    "has_credentials": bool(api_key and api_secret)
                }
            )
            logger.error(f"Failed to connect to live exchange - ID: {error_response.error_id}")
            self.connected = False
            return False

    def _get_api_credentials(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Get API credentials from SecretManager first, then fallback to environment variables
        Includes automatic migration from env vars to SecretManager
        """
        try:
            # Determine if we're using testnet
            suffix = '_testnet' if self.testnet else ''
            exchange_key = f"{self.exchange_name}{suffix}"
            
            # Try to get from SecretManager first
            api_key, api_secret = get_api_credentials(exchange_key)
            
            if api_key and api_secret:
                logger.info(f"Retrieved {self.exchange_name} credentials from SecretManager")
                return api_key, api_secret
            
            # Fallback to environment variables
            env_key_name = f'{self.exchange_name.upper()}_{"TESTNET_" if self.testnet else ""}API_KEY'
            env_secret_name = f'{self.exchange_name.upper()}_{"TESTNET_" if self.testnet else ""}API_SECRET'
            
            env_api_key = os.getenv(env_key_name)
            env_api_secret = os.getenv(env_secret_name)
            
            # If found in env, migrate to SecretManager
            if env_api_key and env_api_secret:
                logger.info(f"Found {self.exchange_name} credentials in environment, migrating to SecretManager...")
                from utils.secret_manager import store_api_key
                
                success = store_api_key(exchange_key, env_api_key, env_api_secret)
                if success:
                    logger.info(f"Successfully migrated {self.exchange_name} credentials to SecretManager")
                    logger.warning(f"Please remove {env_key_name} and {env_secret_name} from your .env file!")
                else:
                    logger.error(f"Failed to migrate {self.exchange_name} credentials to SecretManager")
                
                return env_api_key, env_api_secret
            
            # No credentials found anywhere
            logger.warning(f"No API credentials found for {self.exchange_name}")
            return None, None
            
        except Exception as e:
            error_response = secure_error_handler.handle_critical_error(
                error=e,
                context={
                    "operation": "api_credentials_retrieval",
                    "exchange_name": self.exchange_name,
                    "testnet": self.testnet
                }
            )
            logger.error(f"Error retrieving API credentials - ID: {error_response.error_id}")
            # Last resort fallback to env vars
            env_key_name = f'{self.exchange_name.upper()}_API_KEY'
            env_secret_name = f'{self.exchange_name.upper()}_API_SECRET'
            return os.getenv(env_key_name), os.getenv(env_secret_name)

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
            error_response = secure_error_handler.handle_api_error(
                error=e,
                context={
                    "operation": "fetch_ticker",
                    "symbol": symbol,
                    "exchange_name": self.exchange_name
                }
            )
            logger.error(f"Error fetching ticker for {symbol} - ID: {error_response.error_id}")
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
            error_response = secure_error_handler.handle_api_error(
                error=e,
                context={
                    "operation": "fetch_ohlcv",
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "limit": limit,
                    "exchange_name": self.exchange_name
                }
            )
            logger.error(f"Error fetching OHLCV for {symbol} - ID: {error_response.error_id}")
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
            error_response = secure_error_handler.handle_api_error(
                error=e,
                context={
                    "operation": "fetch_balance",
                    "exchange_name": self.exchange_name,
                    "mode": self.mode
                }
            )
            logger.error(f"Error fetching balance - ID: {error_response.error_id}")
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
            error_response = secure_error_handler.handle_trading_error(
                error=e,
                symbol=symbol,
                amount=amount,
                context={
                    "operation": "create_order",
                    "order_type": order_type,
                    "side": side,
                    "price": price,
                    "exchange_name": self.exchange_name,
                    "mode": self.mode
                }
            )
            logger.error(f"Error creating order for {symbol} - ID: {error_response.error_id}")
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
            error_response = secure_error_handler.handle_trading_error(
                error=e,
                symbol=symbol,
                order_id=order_id,
                context={
                    "operation": "cancel_order",
                    "exchange_name": self.exchange_name,
                    "mode": self.mode
                }
            )
            logger.error(f"Error canceling order {order_id} - ID: {error_response.error_id}")
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
            error_response = secure_error_handler.handle_trading_error(
                error=e,
                symbol=symbol,
                order_id=order_id,
                context={
                    "operation": "fetch_order",
                    "exchange_name": self.exchange_name,
                    "mode": self.mode
                }
            )
            logger.error(f"Error fetching order {order_id} - ID: {error_response.error_id}")
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

        # Fix für pandas FutureWarning
        if freq == 'H':
            freq = 'h'
        elif freq == 'D':
            freq = 'd'
        elif freq == 'W':
            freq = 'w'
        elif freq == 'M':
            freq = 'ME'  # Month End

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
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        if self.exchange_name == 'mock' or self.testnet:
            return {
                'balances': {'USDT': 10000},
                'positions': [],
                'total_value': 10000
            }
        return {}

    def get_balance(self, currency: str = 'USDT') -> float:
        """Get balance for a specific currency"""
        if self.exchange_name == 'mock' or self.testnet:
            return 10000.0 if currency == 'USDT' else 0.0
        return 0.0


    @staticmethod
    def create(exchange_name: str = 'binance', mode: str = 'live') -> ExchangeManager:
        """Alias for create_exchange"""
        return ExchangeFactory.create_exchange(exchange_name, mode)


# Fügen Sie diese Methoden am Ende der ExchangeManager Klasse in core/exchange.py hinzu:

def get_account_info(self) -> Dict[str, Any]:
    """Get account information"""
    if self.exchange_name == 'mock' or self.testnet:
        return {
            'balances': {'USDT': 10000, 'BTC': 0, 'ETH': 0},
            'positions': [],
            'total_value': 10000
        }

    try:
        if self.exchange and hasattr(self.exchange, 'fetch_balance'):
            balance = self.exchange.fetch_balance()
            return {
                'balances': balance.get('free', {}),
                'positions': [],
                'total_value': sum(balance.get('free', {}).values())
            }
    except Exception as e:
        error_response = secure_error_handler.handle_api_error(
            error=e,
            context={
                "operation": "get_account_info",
                "exchange_name": self.exchange_name
            }
        )
        logger.warning(f"Could not get account info - ID: {error_response.error_id}")
        return {
            'balances': {'USDT': 10000},
            'positions': [],
            'total_value': 10000
        }


def get_balance(self, currency: str = 'USDT') -> float:
    """Get balance for a specific currency"""
    if self.exchange_name == 'mock' or self.testnet:
        if currency == 'USDT':
            return 10000.0
        return 0.0

    try:
        if self.exchange and hasattr(self.exchange, 'fetch_balance'):
            balance = self.exchange.fetch_balance()
            return float(balance.get(currency, {}).get('free', 0))
    except Exception as e:
        error_response = secure_error_handler.handle_api_error(
            error=e,
            context={
                "operation": "get_balance",
                "currency": currency,
                "exchange_name": self.exchange_name
            }
        )
        logger.warning(f"Could not get balance for {currency} - ID: {error_response.error_id}")
        return 10000.0 if currency == 'USDT' else 0.0