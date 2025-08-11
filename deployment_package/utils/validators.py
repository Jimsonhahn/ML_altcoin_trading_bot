"""
Pydantic-based Validators for Trading Bot
Provides comprehensive validation for all trading-related data
"""

import re
import logging
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.types import PositiveFloat, PositiveInt
from pydantic import StringConstraints
from typing import Annotated

logger = logging.getLogger(__name__)


class OrderType(str, Enum):
    """Valid order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    OCO = "oco"  # One-Cancels-Other
    BRACKET = "bracket"


class OrderSide(str, Enum):
    """Valid order sides"""
    BUY = "buy"
    SELL = "sell"


class TimeInForce(str, Enum):
    """Valid time in force options"""
    GTC = "gtc"  # Good Till Canceled
    IOC = "ioc"  # Immediate Or Cancel
    FOK = "fok"  # Fill Or Kill
    DAY = "day"  # Good for Day


class TradingMode(str, Enum):
    """Valid trading modes"""
    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"


class MarketRegime(str, Enum):
    """Valid market regimes"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    NEUTRAL = "neutral"
    EXTREME_FEAR = "extreme_fear"


class TradingSymbolValidator(BaseModel):
    """
    Validates trading symbols (e.g., BTC/USDT, ETH/BTC)
    """
    symbol: Annotated[str, StringConstraints(min_length=5, max_length=20)]
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol_format(cls, v):
        """Validate symbol format"""
        if not v:
            raise ValueError('Symbol cannot be empty')
        
        # Convert to uppercase for consistency
        v = v.upper()
        
        # Check for valid trading pair format
        if '/' not in v:
            raise ValueError('Symbol must contain "/" separator (e.g., BTC/USDT)')
        
        parts = v.split('/')
        if len(parts) != 2:
            raise ValueError('Symbol must have exactly one "/" separator')
        
        base, quote = parts
        
        # Validate base currency
        if not re.match(r'^[A-Z0-9]{2,10}$', base):
            raise ValueError('Base currency must be 2-10 alphanumeric characters')
        
        # Validate quote currency
        if not re.match(r'^[A-Z0-9]{2,10}$', quote):
            raise ValueError('Quote currency must be 2-10 alphanumeric characters')
        
        # Common quote currencies validation
        valid_quotes = {'USDT', 'BTC', 'ETH', 'BNB', 'USDC', 'BUSD', 'EUR', 'USD', 'GBP'}
        if quote not in valid_quotes:
            logger.warning(f"Uncommon quote currency: {quote}")
        
        # Prevent same base and quote
        if base == quote:
            raise ValueError('Base and quote currencies cannot be the same')
        
        return v
    
    @property
    def base_currency(self) -> str:
        """Get base currency"""
        return self.symbol.split('/')[0]
    
    @property
    def quote_currency(self) -> str:
        """Get quote currency"""
        return self.symbol.split('/')[1]


class AmountValidator(BaseModel):
    """
    Validates trading amounts with reasonable limits
    """
    amount: PositiveFloat = Field(..., gt=0)
    currency: Optional[str] = None
    
    @field_validator('amount')
    @classmethod
    def validate_amount_limits(cls, v, info):
        """Validate amount is within reasonable limits"""
        if v <= 0:
            raise ValueError('Amount must be positive')
        
        # Get currency for context-aware validation
        currency = info.data.get('currency', '').upper() if info.data else ''
        
        # Define limits based on currency type
        if currency in ['BTC', 'ETH']:
            # For major cryptocurrencies
            if v > 1000:
                raise ValueError(f'Amount too large for {currency}: {v}')
            if v < 0.00001:
                raise ValueError(f'Amount too small for {currency}: {v}')
        elif currency in ['USDT', 'USDC', 'BUSD', 'USD', 'EUR']:
            # For stablecoins and fiat
            if v > 1000000:  # 1M max
                raise ValueError(f'Amount too large for {currency}: {v}')
            if v < 0.01:  # 1 cent minimum
                raise ValueError(f'Amount too small for {currency}: {v}')
        else:
            # For other altcoins
            if v > 10000000:  # 10M max for altcoins
                raise ValueError(f'Amount too large: {v}')
            if v < 0.000001:  # Minimum precision
                raise ValueError(f'Amount too small: {v}')
        
        return v
    
    @field_validator('amount')
    @classmethod  
    def validate_precision(cls, v):
        """Validate amount precision"""
        try:
            # Convert to Decimal for precise validation
            decimal_amount = Decimal(str(v))
            
            # Check if it has more than 8 decimal places
            if decimal_amount.as_tuple().exponent < -8:
                raise ValueError('Amount has too many decimal places (max 8)')
            
        except (InvalidOperation, ValueError) as e:
            raise ValueError(f'Invalid amount format: {e}')
        
        return v


class ConfigValidator(BaseModel):
    """
    Validates all configuration parameters
    """
    # Trading configuration
    trading_mode: TradingMode = TradingMode.PAPER
    max_position_size: PositiveFloat = Field(default=1000.0, le=100000.0)
    max_positions: PositiveInt = Field(default=5, le=50)
    
    # Risk management
    max_drawdown: Annotated[float, Field(gt=0, lt=1)] = Field(default=0.20, description="Maximum drawdown as percentage")
    stop_loss_percentage: Annotated[float, Field(gt=0, lt=1)] = Field(default=0.02, description="Stop loss as percentage")
    take_profit_percentage: Annotated[float, Field(gt=0, lt=1)] = Field(default=0.05, description="Take profit as percentage")
    risk_per_trade: Annotated[float, Field(gt=0, lt=1)] = Field(default=0.02, description="Risk per trade as percentage")
    
    # Exchange configuration
    exchange_name: Annotated[str, StringConstraints(min_length=1, max_length=50)] = Field(default="binance")
    api_rate_limit: PositiveInt = Field(default=1200, description="API requests per minute")
    
    # Strategy configuration
    strategy_name: Optional[Annotated[str, StringConstraints(min_length=1, max_length=50)]] = None
    strategy_params: Optional[Dict[str, Any]] = None
    
    # Notification configuration
    telegram_enabled: bool = False
    email_enabled: bool = False
    
    @field_validator('max_drawdown')
    @classmethod
    def validate_max_drawdown(cls, v):
        """Validate max drawdown is reasonable"""
        if v > 0.5:  # 50%
            raise ValueError('Max drawdown cannot exceed 50%')
        if v < 0.01:  # 1%
            raise ValueError('Max drawdown cannot be less than 1%')
        return v
    
    @field_validator('stop_loss_percentage')
    @classmethod
    def validate_stop_loss(cls, v):
        """Validate stop loss percentage"""
        if v > 0.2:  # 20%
            raise ValueError('Stop loss cannot exceed 20%')
        if v < 0.005:  # 0.5%
            raise ValueError('Stop loss cannot be less than 0.5%')
        return v
    
    @field_validator('take_profit_percentage')
    @classmethod
    def validate_take_profit(cls, v):
        """Validate take profit percentage"""
        if v > 1.0:  # 100%
            raise ValueError('Take profit cannot exceed 100%')
        if v < 0.01:  # 1%
            raise ValueError('Take profit cannot be less than 1%')
        return v
    
    @model_validator(mode='after')
    def validate_profit_loss_ratio(self):
        """Validate that take profit is reasonable compared to stop loss"""
        stop_loss = self.stop_loss_percentage
        take_profit = self.take_profit_percentage
        
        if stop_loss and take_profit:
            ratio = take_profit / stop_loss
            if ratio < 1.0:
                logger.warning(f"Take profit to stop loss ratio is low: {ratio:.2f}")
            elif ratio > 10.0:
                logger.warning(f"Take profit to stop loss ratio is very high: {ratio:.2f}")
        
        return self
    
    @field_validator('exchange_name')
    @classmethod
    def validate_exchange_name(cls, v):
        """Validate exchange name"""
        supported_exchanges = {
            'binance', 'coinbase', 'kraken', 'bitfinex', 'huobi', 
            'okex', 'kucoin', 'gate', 'bybit', 'ftx'
        }
        
        if v.lower() not in supported_exchanges:
            logger.warning(f"Exchange '{v}' may not be fully supported")
        
        return v.lower()


class OrderValidator(BaseModel):
    """
    Validates order parameters
    """
    symbol: str = Field(..., description="Trading symbol (e.g., BTC/USDT)")
    order_type: OrderType = Field(..., description="Order type")
    side: OrderSide = Field(..., description="Order side (buy/sell)")
    amount: PositiveFloat = Field(..., gt=0, description="Order amount")
    price: Optional[PositiveFloat] = Field(None, gt=0, description="Order price (for limit orders)")
    stop_price: Optional[PositiveFloat] = Field(None, gt=0, description="Stop price (for stop orders)")
    time_in_force: TimeInForce = Field(default=TimeInForce.GTC, description="Time in force")
    client_order_id: Optional[Annotated[str, StringConstraints(max_length=100)]] = Field(None, description="Client order ID")
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v):
        """Validate symbol using TradingSymbolValidator"""
        try:
            symbol_validator = TradingSymbolValidator(symbol=v)
            return symbol_validator.symbol
        except Exception as e:
            raise ValueError(f"Invalid symbol: {e}")
    
    @field_validator('amount')
    @classmethod
    def validate_amount(cls, v, info):
        """Validate amount using AmountValidator"""
        try:
            # Extract quote currency from symbol for context
            symbol = info.data.get('symbol', '') if info.data else ''
            quote_currency = symbol.split('/')[-1] if '/' in symbol else None
            
            amount_validator = AmountValidator(amount=v, currency=quote_currency)
            return amount_validator.amount
        except Exception as e:
            raise ValueError(f"Invalid amount: {e}")
    
    @model_validator(mode='after')
    def validate_order_consistency(self):
        """Validate order parameters are consistent"""
        order_type = self.order_type
        price = self.price
        stop_price = self.stop_price
        
        # Validate price requirements based on order type
        if order_type == OrderType.LIMIT and price is None:
            raise ValueError('Limit orders require a price')
        
        if order_type == OrderType.MARKET and price is not None:
            raise ValueError('Market orders cannot have a price')
        
        if order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and stop_price is None:
            raise ValueError('Stop orders require a stop price')
        
        if order_type == OrderType.STOP_LIMIT and (price is None or stop_price is None):
            raise ValueError('Stop limit orders require both price and stop price')
        
        # Validate stop price logic
        if stop_price and price:
            side = self.side
            if side == OrderSide.BUY and stop_price <= price:
                raise ValueError('For buy stop orders, stop price must be above limit price')
            elif side == OrderSide.SELL and stop_price >= price:
                raise ValueError('For sell stop orders, stop price must be below limit price')
        
        return self
    
    @field_validator('client_order_id')
    @classmethod
    def validate_client_order_id(cls, v):
        """Validate client order ID format"""
        if v is not None:
            # Only allow alphanumeric characters, hyphens, and underscores
            if not re.match(r'^[A-Za-z0-9_-]+$', v):
                raise ValueError('Client order ID can only contain alphanumeric characters, hyphens, and underscores')
        return v


class PositionValidator(BaseModel):
    """
    Validates position parameters
    """
    symbol: str = Field(..., description="Trading symbol")
    size: float = Field(..., description="Position size (positive for long, negative for short)")
    entry_price: PositiveFloat = Field(..., gt=0, description="Entry price")
    current_price: PositiveFloat = Field(..., gt=0, description="Current price")
    stop_loss: Optional[PositiveFloat] = Field(None, gt=0, description="Stop loss price")
    take_profit: Optional[PositiveFloat] = Field(None, gt=0, description="Take profit price")
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v):
        """Validate symbol"""
        try:
            symbol_validator = TradingSymbolValidator(symbol=v)
            return symbol_validator.symbol
        except Exception as e:
            raise ValueError(f"Invalid symbol: {e}")
    
    @field_validator('size')
    @classmethod
    def validate_position_size(cls, v):
        """Validate position size"""
        if v == 0:
            raise ValueError('Position size cannot be zero')
        
        # Check for reasonable limits
        if abs(v) > 1000000:
            raise ValueError('Position size too large')
        
        return v
    
    @model_validator(mode='after')
    def validate_price_levels(self):
        """Validate price levels are logical"""
        entry_price = self.entry_price
        current_price = self.current_price
        stop_loss = self.stop_loss
        take_profit = self.take_profit
        size = self.size
        
        if not all([entry_price, current_price, size]):
            return self
        
        is_long = size > 0
        
        # Validate stop loss
        if stop_loss:
            if is_long and stop_loss >= entry_price:
                raise ValueError('Stop loss for long position must be below entry price')
            elif not is_long and stop_loss <= entry_price:
                raise ValueError('Stop loss for short position must be above entry price')
        
        # Validate take profit
        if take_profit:
            if is_long and take_profit <= entry_price:
                raise ValueError('Take profit for long position must be above entry price')
            elif not is_long and take_profit >= entry_price:
                raise ValueError('Take profit for short position must be below entry price')
        
        # Validate stop loss vs take profit
        if stop_loss and take_profit:
            if is_long and stop_loss >= take_profit:
                raise ValueError('Stop loss must be below take profit for long positions')
            elif not is_long and stop_loss <= take_profit:
                raise ValueError('Stop loss must be above take profit for short positions')
        
        return self
    
    @property
    def is_long(self) -> bool:
        """Check if position is long"""
        return self.size > 0
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized PnL"""
        if self.is_long:
            return self.size * (self.current_price - self.entry_price)
        else:
            return abs(self.size) * (self.entry_price - self.current_price)
    
    @property
    def unrealized_pnl_percentage(self) -> float:
        """Calculate unrealized PnL as percentage"""
        if self.is_long:
            return (self.current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.current_price) / self.entry_price


class StrategyParameterValidator(BaseModel):
    """
    Validates strategy parameters
    """
    strategy_name: Annotated[str, StringConstraints(min_length=1, max_length=50)]
    parameters: Dict[str, Any]
    
    @field_validator('strategy_name')
    @classmethod
    def validate_strategy_name(cls, v):
        """Validate strategy name"""
        # Only allow alphanumeric characters and underscores
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError('Strategy name can only contain alphanumeric characters and underscores')
        
        return v.lower()
    
    @field_validator('parameters')
    @classmethod
    def validate_parameters(cls, v, info):
        """Validate strategy parameters"""
        strategy_name = info.data.get('strategy_name') if info.data else None
        
        # Common parameter validations
        if 'lookback_period' in v:
            if not isinstance(v['lookback_period'], int) or v['lookback_period'] <= 0:
                raise ValueError('lookback_period must be a positive integer')
        
        if 'threshold' in v:
            if not isinstance(v['threshold'], (int, float)) or v['threshold'] < 0:
                raise ValueError('threshold must be a non-negative number')
        
        # Strategy-specific validations
        if strategy_name == 'momentum':
            if 'momentum_period' in v and v['momentum_period'] <= 0:
                raise ValueError('momentum_period must be positive')
        
        elif strategy_name == 'mean_reversion':
            if 'deviation_threshold' in v and v['deviation_threshold'] <= 0:
                raise ValueError('deviation_threshold must be positive')
        
        elif strategy_name == 'grid_trading':
            if 'grid_size' in v and v['grid_size'] <= 0:
                raise ValueError('grid_size must be positive')
            if 'grid_spacing' in v and v['grid_spacing'] <= 0:
                raise ValueError('grid_spacing must be positive')
        
        return v


# Utility functions for validation
def validate_trading_symbol(symbol: str) -> TradingSymbolValidator:
    """Validate a trading symbol"""
    return TradingSymbolValidator(symbol=symbol)


def validate_amount(amount: float, currency: Optional[str] = None) -> AmountValidator:
    """Validate a trading amount"""
    return AmountValidator(amount=amount, currency=currency)


def validate_order(order_data: Dict[str, Any]) -> OrderValidator:
    """Validate order parameters"""
    return OrderValidator(**order_data)


def validate_config(config_data: Dict[str, Any]) -> ConfigValidator:
    """Validate configuration parameters"""
    return ConfigValidator(**config_data)


def validate_position(position_data: Dict[str, Any]) -> PositionValidator:
    """Validate position parameters"""
    return PositionValidator(**position_data)


def validate_trade_params(trade_params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate trade parameters for compatibility"""
    # Basic validation for trade parameters
    validated = {}
    
    # Validate symbol
    if 'symbol' in trade_params:
        symbol_validator = validate_trading_symbol(trade_params['symbol'])
        validated['symbol'] = symbol_validator.symbol
    
    # Validate amount
    if 'amount' in trade_params:
        amount_validator = validate_amount(trade_params['amount'])
        validated['amount'] = amount_validator.amount
    
    # Copy other valid parameters
    for key, value in trade_params.items():
        if key not in validated:
            validated[key] = value
    
    return validated


# Custom exception for validation errors
class ValidationError(Exception):
    """Custom validation error"""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(self.message)
    
    def __str__(self):
        if self.field:
            return f"Validation error in field '{self.field}': {self.message}"
        return f"Validation error: {self.message}"


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    # Test symbol validation
    print("Testing symbol validation:")
    try:
        symbol = validate_trading_symbol("BTC/USDT")
        print(f"✅ Valid symbol: {symbol.symbol}")
        print(f"   Base: {symbol.base_currency}, Quote: {symbol.quote_currency}")
    except Exception as e:
        print(f"❌ Symbol validation failed: {e}")
    
    # Test amount validation
    print("\nTesting amount validation:")
    try:
        amount = validate_amount(100.50, "USDT")
        print(f"✅ Valid amount: {amount.amount} {amount.currency}")
    except Exception as e:
        print(f"❌ Amount validation failed: {e}")
    
    # Test order validation
    print("\nTesting order validation:")
    try:
        order = validate_order({
            "symbol": "BTC/USDT",
            "order_type": "limit",
            "side": "buy",
            "amount": 0.1,
            "price": 45000.0
        })
        print(f"✅ Valid order: {order.side} {order.amount} {order.symbol} at {order.price}")
    except Exception as e:
        print(f"❌ Order validation failed: {e}")
    
    # Test config validation
    print("\nTesting config validation:")
    try:
        config = validate_config({
            "trading_mode": "paper",
            "max_position_size": 1000.0,
            "max_drawdown": 0.15,
            "stop_loss_percentage": 0.02,
            "take_profit_percentage": 0.05
        })
        print(f"✅ Valid config: {config.trading_mode} mode, max drawdown: {config.max_drawdown}")
    except Exception as e:
        print(f"❌ Config validation failed: {e}")
    
    print("\n🎉 Validation testing completed!")