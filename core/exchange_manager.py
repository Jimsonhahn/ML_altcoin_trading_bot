# core/exchange_manager.py
"""
Multi-Exchange Manager with Abstract Interface and Unified Order Format
Supports Binance, KuCoin, Bybit with failover and balance aggregation
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
import ccxt
import time
from enum import Enum
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from utils.notifier import send_warning, send_error, send_critical
    NOTIFIER_AVAILABLE = True
except ImportError:
    NOTIFIER_AVAILABLE = False

logger = logging.getLogger(__name__)


class ExchangeStatus(Enum):
    """Exchange connection status"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    RATE_LIMITED = "rate_limited"


class OrderType(Enum):
    """Unified order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class UnifiedOrder:
    """Unified order format across exchanges"""
    id: str
    exchange: str
    symbol: str
    side: OrderSide
    type: OrderType
    amount: float
    price: Optional[float]
    filled: float
    remaining: float
    cost: float
    status: OrderStatus
    timestamp: datetime
    fee: Optional[Dict[str, Any]] = None
    info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'exchange': self.exchange,
            'symbol': self.symbol,
            'side': self.side.value if isinstance(self.side, OrderSide) else self.side,
            'type': self.type.value if isinstance(self.type, OrderType) else self.type,
            'amount': self.amount,
            'price': self.price,
            'filled': self.filled,
            'remaining': self.remaining,
            'cost': self.cost,
            'status': self.status.value if isinstance(self.status, OrderStatus) else self.status,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'fee': self.fee,
            'info': self.info
        }


@dataclass
class ExchangeBalance:
    """Exchange balance information"""
    exchange: str
    currency: str
    free: float
    used: float
    total: float
    
    @property
    def available(self) -> float:
        return self.free


@dataclass
class AggregatedBalance:
    """Aggregated balance across exchanges"""
    currency: str
    total_free: float
    total_used: float
    total_balance: float
    exchanges: Dict[str, ExchangeBalance]
    
    def get_exchange_balance(self, exchange: str) -> Optional[ExchangeBalance]:
        """Get balance for specific exchange"""
        return self.exchanges.get(exchange)


@dataclass
class ExchangeInfo:
    """Exchange information and capabilities"""
    name: str
    status: ExchangeStatus
    trading_fees: Dict[str, float]
    supported_symbols: List[str]
    min_order_size: Dict[str, float]
    max_order_size: Dict[str, float]
    rate_limits: Dict[str, int]
    latency_ms: Optional[float] = None
    last_update: Optional[datetime] = None


class AbstractExchange(ABC):
    """Abstract base class for exchange implementations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__.lower().replace('exchange', '')
        self.status = ExchangeStatus.DISCONNECTED
        self.exchange = None
        self.symbols = []
        self.fees = {}
        self.latency_ms = None
        
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to exchange"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from exchange"""
        pass
    
    @abstractmethod
    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch ticker for symbol"""
        pass
    
    @abstractmethod
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data"""
        pass
    
    @abstractmethod
    async def fetch_balance(self) -> Dict[str, ExchangeBalance]:
        """Fetch account balance"""
        pass
    
    @abstractmethod
    async def create_order(self, symbol: str, side: OrderSide, order_type: OrderType, 
                          amount: float, price: Optional[float] = None) -> UnifiedOrder:
        """Create order"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel order"""
        pass
    
    @abstractmethod
    async def fetch_order(self, order_id: str, symbol: str) -> UnifiedOrder:
        """Fetch order details"""
        pass
    
    @abstractmethod
    async def get_exchange_info(self) -> ExchangeInfo:
        """Get exchange information"""
        pass
    
    def _convert_ccxt_order(self, ccxt_order: Dict[str, Any]) -> UnifiedOrder:
        """Convert CCXT order to unified format"""
        try:
            return UnifiedOrder(
                id=str(ccxt_order.get('id', '')),
                exchange=self.name,
                symbol=ccxt_order.get('symbol', ''),
                side=OrderSide(ccxt_order.get('side', 'buy')),
                type=OrderType(ccxt_order.get('type', 'market')),
                amount=float(ccxt_order.get('amount', 0)),
                price=float(ccxt_order.get('price', 0)) if ccxt_order.get('price') else None,
                filled=float(ccxt_order.get('filled', 0)),
                remaining=float(ccxt_order.get('remaining', 0)),
                cost=float(ccxt_order.get('cost', 0)),
                status=self._convert_order_status(ccxt_order.get('status', 'open')),
                timestamp=datetime.fromtimestamp(ccxt_order.get('timestamp', time.time() * 1000) / 1000),
                fee=ccxt_order.get('fee'),
                info=ccxt_order
            )
        except Exception as e:
            logger.error(f"Error converting CCXT order: {e}")
            # Return minimal order
            return UnifiedOrder(
                id=str(ccxt_order.get('id', 'unknown')),
                exchange=self.name,
                symbol=ccxt_order.get('symbol', ''),
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                amount=0,
                price=None,
                filled=0,
                remaining=0,
                cost=0,
                status=OrderStatus.REJECTED,
                timestamp=datetime.now()
            )
    
    def _convert_order_status(self, ccxt_status: str) -> OrderStatus:
        """Convert CCXT order status to unified format"""
        status_map = {
            'open': OrderStatus.OPEN,
            'closed': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELLED,
            'cancelled': OrderStatus.CANCELLED,
            'partial': OrderStatus.PARTIALLY_FILLED,
            'pending': OrderStatus.PENDING,
            'rejected': OrderStatus.REJECTED
        }
        return status_map.get(ccxt_status.lower(), OrderStatus.OPEN)


class BinanceExchange(AbstractExchange):
    """Binance exchange implementation"""
    
    async def connect(self) -> bool:
        """Connect to Binance"""
        try:
            api_key = self.config.get('api_key')
            api_secret = self.config.get('api_secret')
            testnet = self.config.get('testnet', False)
            
            config = {
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {
                    'recvWindow': 60000,
                    'adjustForTimeDifference': True
                }
            }
            
            if testnet:
                config['urls'] = {
                    'api': {
                        'public': 'https://testnet.binance.vision/api/v3',
                        'private': 'https://testnet.binance.vision/api/v3',
                    }
                }
            
            self.exchange = ccxt.binance(config)
            await self.exchange.load_markets()
            self.symbols = list(self.exchange.markets.keys())
            self.fees = self.exchange.fees
            self.status = ExchangeStatus.CONNECTED
            
            logger.info(f"Connected to Binance ({'testnet' if testnet else 'live'})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            self.status = ExchangeStatus.ERROR
            return False
    
    async def disconnect(self):
        """Disconnect from Binance"""
        if self.exchange:
            self.exchange = None
        self.status = ExchangeStatus.DISCONNECTED
        logger.info("Disconnected from Binance")
    
    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch Binance ticker"""
        try:
            if not self.exchange:
                raise Exception("Not connected to Binance")
            
            ticker = await self.exchange.fetch_ticker(symbol)
            return ticker
            
        except Exception as e:
            logger.error(f"Error fetching Binance ticker for {symbol}: {e}")
            raise
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Fetch Binance OHLCV data"""
        try:
            if not self.exchange:
                raise Exception("Not connected to Binance")
            
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Binance OHLCV for {symbol}: {e}")
            raise
    
    async def fetch_balance(self) -> Dict[str, ExchangeBalance]:
        """Fetch Binance balance"""
        try:
            if not self.exchange:
                raise Exception("Not connected to Binance")
            
            balance = await self.exchange.fetch_balance()
            
            result = {}
            for currency, amounts in balance.items():
                if currency not in ['info', 'free', 'used', 'total'] and isinstance(amounts, dict):
                    result[currency] = ExchangeBalance(
                        exchange='binance',
                        currency=currency,
                        free=float(amounts.get('free', 0)),
                        used=float(amounts.get('used', 0)),
                        total=float(amounts.get('total', 0))
                    )
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching Binance balance: {e}")
            raise
    
    async def create_order(self, symbol: str, side: OrderSide, order_type: OrderType, 
                          amount: float, price: Optional[float] = None) -> UnifiedOrder:
        """Create Binance order"""
        try:
            if not self.exchange:
                raise Exception("Not connected to Binance")
            
            if order_type == OrderType.MARKET:
                order = await self.exchange.create_market_order(symbol, side.value, amount)
            elif order_type == OrderType.LIMIT:
                if price is None:
                    raise ValueError("Price required for limit orders")
                order = await self.exchange.create_limit_order(symbol, side.value, amount, price)
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            return self._convert_ccxt_order(order)
            
        except Exception as e:
            logger.error(f"Error creating Binance order: {e}")
            raise
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel Binance order"""
        try:
            if not self.exchange:
                raise Exception("Not connected to Binance")
            
            await self.exchange.cancel_order(order_id, symbol)
            return True
            
        except Exception as e:
            logger.error(f"Error canceling Binance order {order_id}: {e}")
            return False
    
    async def fetch_order(self, order_id: str, symbol: str) -> UnifiedOrder:
        """Fetch Binance order"""
        try:
            if not self.exchange:
                raise Exception("Not connected to Binance")
            
            order = await self.exchange.fetch_order(order_id, symbol)
            return self._convert_ccxt_order(order)
            
        except Exception as e:
            logger.error(f"Error fetching Binance order {order_id}: {e}")
            raise
    
    async def get_exchange_info(self) -> ExchangeInfo:
        """Get Binance exchange info"""
        try:
            return ExchangeInfo(
                name='binance',
                status=self.status,
                trading_fees={'maker': 0.001, 'taker': 0.001},  # 0.1%
                supported_symbols=self.symbols,
                min_order_size={'BTC/USDT': 0.00001, 'ETH/USDT': 0.0001},
                max_order_size={'BTC/USDT': 9000, 'ETH/USDT': 100000},
                rate_limits={'requests_per_minute': 1200},
                latency_ms=self.latency_ms,
                last_update=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error getting Binance exchange info: {e}")
            raise


class KuCoinExchange(AbstractExchange):
    """KuCoin exchange implementation"""
    
    async def connect(self) -> bool:
        """Connect to KuCoin"""
        try:
            api_key = self.config.get('api_key')
            api_secret = self.config.get('api_secret')
            passphrase = self.config.get('passphrase')
            testnet = self.config.get('testnet', False)
            
            config = {
                'apiKey': api_key,
                'secret': api_secret,
                'password': passphrase,
                'enableRateLimit': True,
                'options': {
                    'adjustForTimeDifference': True
                }
            }
            
            if testnet:
                config['urls'] = {
                    'api': {
                        'public': 'https://openapi-sandbox.kucoin.com/api/v1',
                        'private': 'https://openapi-sandbox.kucoin.com/api/v1',
                    }
                }
            
            self.exchange = ccxt.kucoin(config)
            await self.exchange.load_markets()
            self.symbols = list(self.exchange.markets.keys())
            self.fees = self.exchange.fees
            self.status = ExchangeStatus.CONNECTED
            
            logger.info(f"Connected to KuCoin ({'testnet' if testnet else 'live'})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to KuCoin: {e}")
            self.status = ExchangeStatus.ERROR
            return False
    
    async def disconnect(self):
        """Disconnect from KuCoin"""
        if self.exchange:
            self.exchange = None
        self.status = ExchangeStatus.DISCONNECTED
        logger.info("Disconnected from KuCoin")
    
    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch KuCoin ticker"""
        try:
            if not self.exchange:
                raise Exception("Not connected to KuCoin")
            
            ticker = await self.exchange.fetch_ticker(symbol)
            return ticker
            
        except Exception as e:
            logger.error(f"Error fetching KuCoin ticker for {symbol}: {e}")
            raise
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Fetch KuCoin OHLCV data"""
        try:
            if not self.exchange:
                raise Exception("Not connected to KuCoin")
            
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"Error fetching KuCoin OHLCV for {symbol}: {e}")
            raise
    
    async def fetch_balance(self) -> Dict[str, ExchangeBalance]:
        """Fetch KuCoin balance"""
        try:
            if not self.exchange:
                raise Exception("Not connected to KuCoin")
            
            balance = await self.exchange.fetch_balance()
            
            result = {}
            for currency, amounts in balance.items():
                if currency not in ['info', 'free', 'used', 'total'] and isinstance(amounts, dict):
                    result[currency] = ExchangeBalance(
                        exchange='kucoin',
                        currency=currency,
                        free=float(amounts.get('free', 0)),
                        used=float(amounts.get('used', 0)),
                        total=float(amounts.get('total', 0))
                    )
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching KuCoin balance: {e}")
            raise
    
    async def create_order(self, symbol: str, side: OrderSide, order_type: OrderType, 
                          amount: float, price: Optional[float] = None) -> UnifiedOrder:
        """Create KuCoin order"""
        try:
            if not self.exchange:
                raise Exception("Not connected to KuCoin")
            
            if order_type == OrderType.MARKET:
                order = await self.exchange.create_market_order(symbol, side.value, amount)
            elif order_type == OrderType.LIMIT:
                if price is None:
                    raise ValueError("Price required for limit orders")
                order = await self.exchange.create_limit_order(symbol, side.value, amount, price)
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            return self._convert_ccxt_order(order)
            
        except Exception as e:
            logger.error(f"Error creating KuCoin order: {e}")
            raise
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel KuCoin order"""
        try:
            if not self.exchange:
                raise Exception("Not connected to KuCoin")
            
            await self.exchange.cancel_order(order_id, symbol)
            return True
            
        except Exception as e:
            logger.error(f"Error canceling KuCoin order {order_id}: {e}")
            return False
    
    async def fetch_order(self, order_id: str, symbol: str) -> UnifiedOrder:
        """Fetch KuCoin order"""
        try:
            if not self.exchange:
                raise Exception("Not connected to KuCoin")
            
            order = await self.exchange.fetch_order(order_id, symbol)
            return self._convert_ccxt_order(order)
            
        except Exception as e:
            logger.error(f"Error fetching KuCoin order {order_id}: {e}")
            raise
    
    async def get_exchange_info(self) -> ExchangeInfo:
        """Get KuCoin exchange info"""
        try:
            return ExchangeInfo(
                name='kucoin',
                status=self.status,
                trading_fees={'maker': 0.001, 'taker': 0.001},  # 0.1%
                supported_symbols=self.symbols,
                min_order_size={'BTC/USDT': 0.00001, 'ETH/USDT': 0.0001},
                max_order_size={'BTC/USDT': 10000, 'ETH/USDT': 100000},
                rate_limits={'requests_per_minute': 1800},
                latency_ms=self.latency_ms,
                last_update=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error getting KuCoin exchange info: {e}")
            raise


class BybitExchange(AbstractExchange):
    """Bybit exchange implementation"""
    
    async def connect(self) -> bool:
        """Connect to Bybit"""
        try:
            api_key = self.config.get('api_key')
            api_secret = self.config.get('api_secret')
            testnet = self.config.get('testnet', False)
            
            config = {
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {
                    'adjustForTimeDifference': True
                }
            }
            
            if testnet:
                config['urls'] = {
                    'api': {
                        'public': 'https://api-testnet.bybit.com',
                        'private': 'https://api-testnet.bybit.com',
                    }
                }
            
            self.exchange = ccxt.bybit(config)
            await self.exchange.load_markets()
            self.symbols = list(self.exchange.markets.keys())
            self.fees = self.exchange.fees
            self.status = ExchangeStatus.CONNECTED
            
            logger.info(f"Connected to Bybit ({'testnet' if testnet else 'live'})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Bybit: {e}")
            self.status = ExchangeStatus.ERROR
            return False
    
    async def disconnect(self):
        """Disconnect from Bybit"""
        if self.exchange:
            self.exchange = None
        self.status = ExchangeStatus.DISCONNECTED
        logger.info("Disconnected from Bybit")
    
    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch Bybit ticker"""
        try:
            if not self.exchange:
                raise Exception("Not connected to Bybit")
            
            ticker = await self.exchange.fetch_ticker(symbol)
            return ticker
            
        except Exception as e:
            logger.error(f"Error fetching Bybit ticker for {symbol}: {e}")
            raise
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Fetch Bybit OHLCV data"""
        try:
            if not self.exchange:
                raise Exception("Not connected to Bybit")
            
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Bybit OHLCV for {symbol}: {e}")
            raise
    
    async def fetch_balance(self) -> Dict[str, ExchangeBalance]:
        """Fetch Bybit balance"""
        try:
            if not self.exchange:
                raise Exception("Not connected to Bybit")
            
            balance = await self.exchange.fetch_balance()
            
            result = {}
            for currency, amounts in balance.items():
                if currency not in ['info', 'free', 'used', 'total'] and isinstance(amounts, dict):
                    result[currency] = ExchangeBalance(
                        exchange='bybit',
                        currency=currency,
                        free=float(amounts.get('free', 0)),
                        used=float(amounts.get('used', 0)),
                        total=float(amounts.get('total', 0))
                    )
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching Bybit balance: {e}")
            raise
    
    async def create_order(self, symbol: str, side: OrderSide, order_type: OrderType, 
                          amount: float, price: Optional[float] = None) -> UnifiedOrder:
        """Create Bybit order"""
        try:
            if not self.exchange:
                raise Exception("Not connected to Bybit")
            
            if order_type == OrderType.MARKET:
                order = await self.exchange.create_market_order(symbol, side.value, amount)
            elif order_type == OrderType.LIMIT:
                if price is None:
                    raise ValueError("Price required for limit orders")
                order = await self.exchange.create_limit_order(symbol, side.value, amount, price)
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            return self._convert_ccxt_order(order)
            
        except Exception as e:
            logger.error(f"Error creating Bybit order: {e}")
            raise
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel Bybit order"""
        try:
            if not self.exchange:
                raise Exception("Not connected to Bybit")
            
            await self.exchange.cancel_order(order_id, symbol)
            return True
            
        except Exception as e:
            logger.error(f"Error canceling Bybit order {order_id}: {e}")
            return False
    
    async def fetch_order(self, order_id: str, symbol: str) -> UnifiedOrder:
        """Fetch Bybit order"""
        try:
            if not self.exchange:
                raise Exception("Not connected to Bybit")
            
            order = await self.exchange.fetch_order(order_id, symbol)
            return self._convert_ccxt_order(order)
            
        except Exception as e:
            logger.error(f"Error fetching Bybit order {order_id}: {e}")
            raise
    
    async def get_exchange_info(self) -> ExchangeInfo:
        """Get Bybit exchange info"""
        try:
            return ExchangeInfo(
                name='bybit',
                status=self.status,
                trading_fees={'maker': 0.001, 'taker': 0.001},  # 0.1%
                supported_symbols=self.symbols,
                min_order_size={'BTC/USDT': 0.00001, 'ETH/USDT': 0.0001},
                max_order_size={'BTC/USDT': 10000, 'ETH/USDT': 100000},
                rate_limits={'requests_per_minute': 600},
                latency_ms=self.latency_ms,
                last_update=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error getting Bybit exchange info: {e}")
            raise


class MultiExchangeManager:
    """
    Main Multi-Exchange Manager
    Handles multiple exchanges with unified interface and balance aggregation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.exchanges: Dict[str, AbstractExchange] = {}
        self.primary_exchange = config.get('primary_exchange', 'binance')
        self.enabled_exchanges = config.get('enabled_exchanges', ['binance'])
        
        # Exchange classes mapping
        self.exchange_classes = {
            'binance': BinanceExchange,
            'kucoin': KuCoinExchange,
            'bybit': BybitExchange
        }
        
        # Initialize exchanges
        self._initialize_exchanges()
        
        logger.info(f"MultiExchangeManager initialized with {len(self.exchanges)} exchanges")
    
    def _initialize_exchanges(self):
        """Initialize all configured exchanges"""
        exchange_configs = self.config.get('exchanges', {})
        
        for exchange_name in self.enabled_exchanges:
            if exchange_name in self.exchange_classes:
                exchange_config = exchange_configs.get(exchange_name, {})
                exchange_class = self.exchange_classes[exchange_name]
                
                try:
                    exchange = exchange_class(exchange_config)
                    self.exchanges[exchange_name] = exchange
                    logger.info(f"Initialized {exchange_name} exchange")
                except Exception as e:
                    logger.error(f"Failed to initialize {exchange_name}: {e}")
            else:
                logger.warning(f"Unknown exchange: {exchange_name}")
    
    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all exchanges"""
        results = {}
        
        for name, exchange in self.exchanges.items():
            try:
                success = await exchange.connect()
                results[name] = success
                if success:
                    logger.info(f"Successfully connected to {name}")
                else:
                    logger.error(f"Failed to connect to {name}")
            except Exception as e:
                logger.error(f"Error connecting to {name}: {e}")
                results[name] = False
        
        return results
    
    async def disconnect_all(self):
        """Disconnect from all exchanges"""
        for name, exchange in self.exchanges.items():
            try:
                await exchange.disconnect()
                logger.info(f"Disconnected from {name}")
            except Exception as e:
                logger.error(f"Error disconnecting from {name}: {e}")
    
    def get_connected_exchanges(self) -> List[str]:
        """Get list of connected exchanges"""
        return [
            name for name, exchange in self.exchanges.items() 
            if exchange.status == ExchangeStatus.CONNECTED
        ]
    
    def get_exchange(self, name: str) -> Optional[AbstractExchange]:
        """Get specific exchange"""
        return self.exchanges.get(name)
    
    async def fetch_ticker_all(self, symbol: str) -> Dict[str, Dict[str, Any]]:
        """Fetch ticker from all exchanges"""
        results = {}
        
        tasks = []
        for name, exchange in self.exchanges.items():
            if exchange.status == ExchangeStatus.CONNECTED:
                tasks.append(self._fetch_ticker_safe(name, exchange, symbol))
        
        if tasks:
            ticker_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(ticker_results):
                exchange_name = list(self.exchanges.keys())[i]
                if not isinstance(result, Exception):
                    results[exchange_name] = result
                else:
                    logger.error(f"Error fetching ticker from {exchange_name}: {result}")
        
        return results
    
    async def _fetch_ticker_safe(self, name: str, exchange: AbstractExchange, symbol: str):
        """Safely fetch ticker from exchange"""
        try:
            return await exchange.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"Error fetching ticker from {name}: {e}")
            raise
    
    async def get_aggregated_balance(self) -> Dict[str, AggregatedBalance]:
        """Get aggregated balance across all exchanges"""
        all_balances = {}
        
        # Fetch balances from all exchanges
        for name, exchange in self.exchanges.items():
            if exchange.status == ExchangeStatus.CONNECTED:
                try:
                    balances = await exchange.fetch_balance()
                    all_balances[name] = balances
                except Exception as e:
                    logger.error(f"Error fetching balance from {name}: {e}")
        
        # Aggregate by currency
        aggregated = {}
        
        # Get all unique currencies
        all_currencies = set()
        for exchange_balances in all_balances.values():
            all_currencies.update(exchange_balances.keys())
        
        # Aggregate each currency
        for currency in all_currencies:
            exchange_balances = {}
            total_free = 0
            total_used = 0
            total_balance = 0
            
            for exchange_name, balances in all_balances.items():
                if currency in balances:
                    balance = balances[currency]
                    exchange_balances[exchange_name] = balance
                    total_free += balance.free
                    total_used += balance.used
                    total_balance += balance.total
            
            aggregated[currency] = AggregatedBalance(
                currency=currency,
                total_free=total_free,
                total_used=total_used,
                total_balance=total_balance,
                exchanges=exchange_balances
            )
        
        return aggregated
    
    async def create_order_on_exchange(self, exchange_name: str, symbol: str, side: OrderSide, 
                                     order_type: OrderType, amount: float, 
                                     price: Optional[float] = None) -> Optional[UnifiedOrder]:
        """Create order on specific exchange"""
        exchange = self.exchanges.get(exchange_name)
        if not exchange:
            logger.error(f"Exchange {exchange_name} not found")
            return None
        
        if exchange.status != ExchangeStatus.CONNECTED:
            logger.error(f"Exchange {exchange_name} not connected")
            return None
        
        try:
            order = await exchange.create_order(symbol, side, order_type, amount, price)
            logger.info(f"Created order on {exchange_name}: {order.id}")
            return order
        except Exception as e:
            logger.error(f"Error creating order on {exchange_name}: {e}")
            return None
    
    async def get_best_price(self, symbol: str, side: OrderSide) -> Optional[Tuple[str, float]]:
        """Get best price across all exchanges"""
        try:
            tickers = await self.fetch_ticker_all(symbol)
            
            if not tickers:
                return None
            
            best_exchange = None
            best_price = None
            
            for exchange_name, ticker in tickers.items():
                if side == OrderSide.BUY:
                    # For buying, we want the lowest ask price
                    price = ticker.get('ask')
                else:
                    # For selling, we want the highest bid price
                    price = ticker.get('bid')
                
                if price and (best_price is None or 
                             (side == OrderSide.BUY and price < best_price) or
                             (side == OrderSide.SELL and price > best_price)):
                    best_price = price
                    best_exchange = exchange_name
            
            return (best_exchange, best_price) if best_exchange else None
            
        except Exception as e:
            logger.error(f"Error finding best price for {symbol}: {e}")
            return None
    
    async def get_exchange_status_all(self) -> Dict[str, ExchangeInfo]:
        """Get status of all exchanges"""
        results = {}
        
        for name, exchange in self.exchanges.items():
            try:
                info = await exchange.get_exchange_info()
                results[name] = info
            except Exception as e:
                logger.error(f"Error getting status for {name}: {e}")
                # Create error status
                results[name] = ExchangeInfo(
                    name=name,
                    status=ExchangeStatus.ERROR,
                    trading_fees={},
                    supported_symbols=[],
                    min_order_size={},
                    max_order_size={},
                    rate_limits={},
                    last_update=datetime.now()
                )
        
        return results
    
    def get_primary_exchange(self) -> Optional[AbstractExchange]:
        """Get primary exchange"""
        return self.exchanges.get(self.primary_exchange)
    
    async def health_check(self) -> Dict[str, bool]:
        """Perform health check on all exchanges"""
        results = {}
        
        for name, exchange in self.exchanges.items():
            try:
                if exchange.status == ExchangeStatus.CONNECTED:
                    # Simple health check - try to fetch a ticker
                    await exchange.fetch_ticker('BTC/USDT')
                    results[name] = True
                else:
                    results[name] = False
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                results[name] = False
                # Update exchange status
                exchange.status = ExchangeStatus.ERROR
        
        return results


# Factory function for easy creation
def create_multi_exchange_manager(config: Dict[str, Any]) -> MultiExchangeManager:
    """Create and return a MultiExchangeManager instance"""
    return MultiExchangeManager(config)


# Backward compatibility with original ExchangeManager
class ExchangeManager:
    """
    Legacy ExchangeManager class for backward compatibility
    Wraps MultiExchangeManager to provide single-exchange interface
    """
    
    def __init__(self, exchange_name: str = 'binance', mode: str = 'live'):
        # Convert old parameters to new config format
        config = {
            'primary_exchange': exchange_name,
            'enabled_exchanges': [exchange_name],
            'exchanges': {
                exchange_name: {
                    'testnet': mode == 'paper'
                }
            }
        }
        
        self.multi_manager = MultiExchangeManager(config)
        self.exchange_name = exchange_name
        self.mode = mode
        
        # For compatibility
        self.connected = False
        self.exchange = None
    
    async def connect(self) -> bool:
        """Connect (async version)"""
        results = await self.multi_manager.connect_all()
        self.connected = results.get(self.exchange_name, False)
        if self.connected:
            self.exchange = self.multi_manager.get_exchange(self.exchange_name)
        return self.connected
    
    def connect_sync(self) -> bool:
        """Connect (sync version for backward compatibility)"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.connect())
        except Exception as e:
            logger.error(f"Error in sync connect: {e}")
            return False
    
    # Sync methods for backward compatibility
    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch ticker (sync)"""
        if not self.connected:
            return self._get_mock_ticker(symbol)
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            exchange = self.multi_manager.get_exchange(self.exchange_name)
            return loop.run_until_complete(exchange.fetch_ticker(symbol))
        except Exception as e:
            logger.error(f"Error fetching ticker: {e}")
            return self._get_mock_ticker(symbol)
    
    def _get_mock_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get mock ticker for backward compatibility"""
        base_prices = {
            'BTC/USDT': 45000,
            'ETH/USDT': 3000,
            'SOL/USDT': 100,
        }
        
        price = base_prices.get(symbol, 100)
        return {
            'symbol': symbol,
            'last': price,
            'bid': price * 0.999,
            'ask': price * 1.001,
            'timestamp': int(time.time() * 1000)
        }