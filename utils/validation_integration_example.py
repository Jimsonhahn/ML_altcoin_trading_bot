"""
Example integration showing how to use validators in trading bot components
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from utils.validators import (
    validate_trading_symbol, validate_amount, validate_order, validate_config,
    OrderValidator, PositionValidator, ConfigValidator,
    OrderType, OrderSide, TradingMode, ValidationError
)
from pydantic import ValidationError as PydanticValidationError

logger = logging.getLogger(__name__)

class ValidatedTradingBot:
    """
    Example trading bot that uses validation for all inputs
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with validated configuration"""
        try:
            self.config = validate_config(config)
            logger.info(f"Trading bot initialized in {self.config.trading_mode.value} mode")
            logger.info(f"Max drawdown: {self.config.max_drawdown}, Risk per trade: {self.config.risk_per_trade}")
        except (ValidationError, PydanticValidationError) as e:
            logger.error(f"Invalid configuration: {e}")
            raise
    
    def create_order(self, symbol: str, order_type: str, side: str, 
                    amount: float, price: Optional[float] = None) -> Dict[str, Any]:
        """
        Create a validated order
        """
        try:
            # Validate order parameters
            order_data = {
                "symbol": symbol,
                "order_type": order_type,
                "side": side,
                "amount": amount
            }
            
            if price is not None:
                order_data["price"] = price
            
            # Validate the order
            validated_order = validate_order(order_data)
            
            # Log the validated order
            logger.info(f"Creating {validated_order.order_type.value} {validated_order.side.value} order: "
                       f"{validated_order.amount} {validated_order.symbol}")
            
            if validated_order.price:
                logger.info(f"Order price: {validated_order.price}")
            
            # Here you would send the order to the exchange
            # For now, we'll just return the order data
            return {
                "success": True,
                "order_id": f"order_{int(datetime.now().timestamp())}",
                "symbol": validated_order.symbol,
                "type": validated_order.order_type.value,
                "side": validated_order.side.value,
                "amount": validated_order.amount,
                "price": validated_order.price,
                "status": "pending"
            }
            
        except (ValidationError, PydanticValidationError) as e:
            logger.error(f"Order validation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "validation_error"
            }
        except Exception as e:
            logger.error(f"Order creation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "execution_error"
            }
    
    def open_position(self, symbol: str, size: float, entry_price: float,
                     stop_loss: Optional[float] = None, 
                     take_profit: Optional[float] = None) -> Dict[str, Any]:
        """
        Open a validated position
        """
        try:
            # Validate position parameters
            position_data = {
                "symbol": symbol,
                "size": size,
                "entry_price": entry_price,
                "current_price": entry_price  # Assuming we enter at current price
            }
            
            if stop_loss is not None:
                position_data["stop_loss"] = stop_loss
            
            if take_profit is not None:
                position_data["take_profit"] = take_profit
            
            # Validate the position
            validated_position = PositionValidator(**position_data)
            
            # Log the validated position
            position_type = "Long" if validated_position.is_long else "Short"
            logger.info(f"Opening {position_type} position: {validated_position.size} {validated_position.symbol} "
                       f"at {validated_position.entry_price}")
            
            if validated_position.stop_loss:
                logger.info(f"Stop loss: {validated_position.stop_loss}")
            
            if validated_position.take_profit:
                logger.info(f"Take profit: {validated_position.take_profit}")
            
            # Here you would execute the position opening logic
            # For now, we'll just return the position data
            return {
                "success": True,
                "position_id": f"pos_{int(datetime.now().timestamp())}",
                "symbol": validated_position.symbol,
                "size": validated_position.size,
                "entry_price": validated_position.entry_price,
                "stop_loss": validated_position.stop_loss,
                "take_profit": validated_position.take_profit,
                "unrealized_pnl": 0.0,
                "status": "open"
            }
            
        except (ValidationError, PydanticValidationError) as e:
            logger.error(f"Position validation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "validation_error"
            }
        except Exception as e:
            logger.error(f"Position opening failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "execution_error"
            }
    
    def calculate_position_size(self, symbol: str, current_price: float, 
                               stop_loss_price: float) -> float:
        """
        Calculate position size with validation
        """
        try:
            # Validate symbol
            validated_symbol = validate_trading_symbol(symbol)
            
            # Validate prices
            validate_amount(current_price, validated_symbol.quote_currency)
            validate_amount(stop_loss_price, validated_symbol.quote_currency)
            
            # Calculate risk amount based on config
            account_balance = 10000.0  # Example balance
            risk_amount = account_balance * self.config.risk_per_trade
            
            # Calculate stop loss distance
            stop_loss_distance = abs(current_price - stop_loss_price) / current_price
            
            # Calculate position size
            position_size = risk_amount / (current_price * stop_loss_distance)
            
            # Validate calculated position size
            validate_amount(position_size, validated_symbol.base_currency)
            
            # Ensure position size doesn't exceed max position size
            max_position_value = min(self.config.max_position_size, account_balance * 0.5)
            max_size = max_position_value / current_price
            
            final_size = min(position_size, max_size)
            
            logger.info(f"Calculated position size for {symbol}: {final_size}")
            logger.info(f"Risk amount: ${risk_amount:.2f}, Stop loss distance: {stop_loss_distance:.2%}")
            
            return final_size
            
        except (ValidationError, PydanticValidationError) as e:
            logger.error(f"Position size calculation validation failed: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Position size calculation failed: {e}")
            return 0.0
    
    def validate_trading_rules(self, symbol: str, amount: float) -> bool:
        """
        Validate trading rules before executing trades
        """
        try:
            # Validate symbol
            validated_symbol = validate_trading_symbol(symbol)
            
            # Validate amount
            validate_amount(amount, validated_symbol.base_currency)
            
            # Check if we're within position limits
            # This is a simplified check - in reality you'd check current positions
            if amount * 45000 > self.config.max_position_size:  # Assuming BTC price
                logger.warning(f"Position size {amount} exceeds max position size")
                return False
            
            # Check if trading mode allows live trading
            if self.config.trading_mode == TradingMode.PAPER:
                logger.info("Paper trading mode - validation passed")
            
            return True
            
        except (ValidationError, PydanticValidationError) as e:
            logger.error(f"Trading rules validation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Trading rules check failed: {e}")
            return False


def demo_validation_integration():
    """
    Demonstrate how validation integrates with trading operations
    """
    print("🤖 Validation Integration Demo")
    print("=" * 50)
    
    # 1. Initialize bot with validated config
    print("📝 1. Initializing bot with configuration...")
    config = {
        "trading_mode": "paper",
        "max_position_size": 5000.0,
        "max_drawdown": 0.15,
        "stop_loss_percentage": 0.02,
        "take_profit_percentage": 0.05,
        "risk_per_trade": 0.02,
        "exchange_name": "binance"
    }
    
    try:
        bot = ValidatedTradingBot(config)
        print("✅ Bot initialized successfully")
    except Exception as e:
        print(f"❌ Bot initialization failed: {e}")
        return
    
    # 2. Create validated orders
    print("\\n📋 2. Creating validated orders...")
    orders = [
        ("BTC/USDT", "limit", "buy", 0.1, 45000.0),
        ("ETH/USDT", "market", "sell", 1.0, None),
        ("SOL/USDT", "limit", "buy", 10.0, 165.0)
    ]
    
    for symbol, order_type, side, amount, price in orders:
        result = bot.create_order(symbol, order_type, side, amount, price)
        if result["success"]:
            print(f"✅ Order created: {result['order_id']}")
        else:
            print(f"❌ Order failed: {result['error']}")
    
    # 3. Open validated positions
    print("\\n📈 3. Opening validated positions...")
    positions = [
        ("BTC/USDT", 0.1, 45000.0, 44000.0, 47000.0),
        ("ETH/USDT", -1.0, 3000.0, 3100.0, 2800.0)
    ]
    
    for symbol, size, entry_price, stop_loss, take_profit in positions:
        result = bot.open_position(symbol, size, entry_price, stop_loss, take_profit)
        if result["success"]:
            print(f"✅ Position opened: {result['position_id']}")
        else:
            print(f"❌ Position failed: {result['error']}")
    
    # 4. Calculate position sizes
    print("\\n📊 4. Calculating position sizes...")
    calculations = [
        ("BTC/USDT", 45000.0, 44000.0),
        ("ETH/USDT", 3000.0, 2950.0)
    ]
    
    for symbol, current_price, stop_loss in calculations:
        size = bot.calculate_position_size(symbol, current_price, stop_loss)
        print(f"✅ {symbol}: Position size = {size:.6f}")
    
    # 5. Validate trading rules
    print("\\n✅ 5. Validating trading rules...")
    rule_checks = [
        ("BTC/USDT", 0.1),
        ("ETH/USDT", 1.0),
        ("INVALID/PAIR", 0.5)
    ]
    
    for symbol, amount in rule_checks:
        is_valid = bot.validate_trading_rules(symbol, amount)
        status = "✅ Valid" if is_valid else "❌ Invalid"
        print(f"{status}: {symbol} - {amount}")
    
    print("\\n🎉 Validation integration demo completed!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_validation_integration()