"""
Integrated Trading Bot Example
Demonstrates how validation and error handling work together
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from utils.validators import (
    validate_trading_symbol, validate_amount, validate_order, validate_config,
    OrderValidator, PositionValidator, ConfigValidator,
    OrderType, OrderSide, TradingMode, ValidationError
)
from utils.error_handler import (
    handle_errors, ErrorCategory, ErrorSeverity, TradingBotError,
    ValidationTradingError, NetworkTradingError, ExchangeTradingError,
    safe_execute, error_handler
)
from pydantic import ValidationError as PydanticValidationError

logger = logging.getLogger(__name__)


class IntegratedTradingBot:
    """
    Advanced trading bot with integrated validation and error handling
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with validated configuration and error handling"""
        # Set up error notifications
        error_handler.add_notification_callback(self._handle_error_notification)
        
        # Validate and store configuration
        self.config = self._initialize_config(config)
        self.positions = {}
        self.orders = {}
        self.is_running = False
        
        logger.info(f"IntegratedTradingBot initialized in {self.config.trading_mode.value} mode")
    
    def _initialize_config(self, config: Dict[str, Any]) -> ConfigValidator:
        """Initialize configuration with comprehensive validation and error handling"""
        try:
            validated_config = validate_config(config)
            logger.info("Configuration validated successfully")
            return validated_config
        except (ValidationError, PydanticValidationError) as e:
            raise ValidationTradingError(
                f"Configuration validation failed: {str(e)}",
                field="config",
                value=config
            )
        except Exception as e:
            raise TradingBotError(
                f"Configuration initialization failed: {str(e)}",
                category=ErrorCategory.CONFIGURATION,
                severity=ErrorSeverity.CRITICAL,
                recoverable=False
            )
    
    def _handle_error_notification(self, error_context):
        """Handle error notifications"""
        if error_context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            logger.critical(f"CRITICAL ERROR: {error_context.user_message}")
            # In a real application, you might send alerts, pause trading, etc.
            if error_context.category == ErrorCategory.EXCHANGE:
                logger.warning("Pausing trading due to exchange error")
                self.is_running = False
    
    @handle_errors(
        category=ErrorCategory.TRADING,
        max_retries=2,
        retry_delay=1.0,
        reraise=True
    )
    def create_validated_order(
        self, 
        symbol: str, 
        order_type: str, 
        side: str,
        amount: float, 
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Create an order with comprehensive validation and error handling
        """
        logger.info(f"Creating order: {side} {amount} {symbol}")
        
        # Step 1: Pre-validation checks
        self._validate_trading_conditions(symbol, amount)
        
        # Step 2: Validate order parameters
        order_data = {
            "symbol": symbol,
            "order_type": order_type,
            "side": side,
            "amount": amount
        }
        
        if price is not None:
            order_data["price"] = price
        
        try:
            validated_order = validate_order(order_data)
            logger.debug(f"Order validation successful: {validated_order}")
        except (ValidationError, PydanticValidationError) as e:
            raise ValidationTradingError(
                f"Order validation failed: {str(e)}",
                field="order",
                value=order_data
            )
        
        # Step 3: Risk management checks
        self._perform_risk_checks(validated_order)
        
        # Step 4: Simulate order execution (replace with real exchange API)
        order_result = self._execute_order_safely(validated_order)
        
        # Step 5: Store successful order
        if order_result["success"]:
            self.orders[order_result["order_id"]] = {
                "order": validated_order,
                "result": order_result,
                "timestamp": datetime.now()
            }
            logger.info(f"Order created successfully: {order_result['order_id']}")
        
        return order_result
    
    @handle_errors(category=ErrorCategory.VALIDATION, reraise=True)
    def _validate_trading_conditions(self, symbol: str, amount: float):
        """Validate basic trading conditions"""
        # Validate symbol
        symbol_validator = validate_trading_symbol(symbol)
        
        # Validate amount
        quote_currency = symbol_validator.quote_currency
        amount_validator = validate_amount(amount, quote_currency)
        
        # Check if trading is enabled
        if not self.is_running and self.config.trading_mode == TradingMode.LIVE:
            raise TradingBotError(
                "Trading is currently paused",
                category=ErrorCategory.TRADING,
                severity=ErrorSeverity.HIGH,
                recoverable=True
            )
    
    @handle_errors(category=ErrorCategory.TRADING, max_retries=1)
    def _perform_risk_checks(self, order: OrderValidator):
        """Perform comprehensive risk management checks"""
        # Check position limits
        current_position_count = len(self.positions)
        if current_position_count >= self.config.max_positions:
            raise TradingBotError(
                f"Maximum position limit reached: {current_position_count}/{self.config.max_positions}",
                category=ErrorCategory.TRADING,
                severity=ErrorSeverity.MEDIUM,
                recoverable=True
            )
        
        # Check position size limits
        if order.price:
            position_value = order.amount * order.price
            if position_value > self.config.max_position_size:
                raise TradingBotError(
                    f"Position size exceeds limit: {position_value} > {self.config.max_position_size}",
                    category=ErrorCategory.TRADING,
                    severity=ErrorSeverity.MEDIUM,
                    recoverable=True
                )
        
        logger.debug("Risk checks passed")
    
    @handle_errors(category=ErrorCategory.EXCHANGE, max_retries=3, retry_delay=2.0)
    def _execute_order_safely(self, order: OrderValidator) -> Dict[str, Any]:
        """
        Safely execute order with exchange error handling
        (This would integrate with actual exchange API)
        """
        try:
            # Simulate exchange API call
            if order.symbol == "FAIL/TEST":
                raise NetworkTradingError(
                    "Simulated exchange API failure",
                    url="https://api.exchange.com/order",
                    status_code=500
                )
            
            # Simulate successful order
            order_id = f"order_{int(datetime.now().timestamp())}"
            
            return {
                "success": True,
                "order_id": order_id,
                "symbol": order.symbol,
                "type": order.order_type.value,
                "side": order.side.value,
                "amount": order.amount,
                "price": order.price,
                "status": "filled",
                "timestamp": datetime.now().isoformat()
            }
            
        except NetworkTradingError:
            # Re-raise network errors for retry logic
            raise
        except Exception as e:
            # Convert other exceptions to exchange errors
            raise ExchangeTradingError(
                f"Order execution failed: {str(e)}",
                exchange="simulated_exchange",
                order_id=None
            )
    
    @handle_errors(category=ErrorCategory.TRADING, max_retries=1)
    def open_validated_position(
        self,
        symbol: str,
        size: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Open a position with validation and error handling
        """
        logger.info(f"Opening position: {size} {symbol} at {entry_price}")
        
        # Validate position parameters
        position_data = {
            "symbol": symbol,
            "size": size,
            "entry_price": entry_price,
            "current_price": entry_price
        }
        
        if stop_loss is not None:
            position_data["stop_loss"] = stop_loss
        
        if take_profit is not None:
            position_data["take_profit"] = take_profit
        
        try:
            validated_position = PositionValidator(**position_data)
        except (ValidationError, PydanticValidationError) as e:
            raise ValidationTradingError(
                f"Position validation failed: {str(e)}",
                field="position",
                value=position_data
            )
        
        # Check risk limits
        self._perform_risk_checks_for_position(validated_position)
        
        # Store position
        position_id = f"pos_{int(datetime.now().timestamp())}"
        self.positions[position_id] = {
            "position": validated_position,
            "timestamp": datetime.now()
        }
        
        result = {
            "success": True,
            "position_id": position_id,
            "symbol": validated_position.symbol,
            "size": validated_position.size,
            "entry_price": validated_position.entry_price,
            "unrealized_pnl": 0.0,
            "status": "open"
        }
        
        logger.info(f"Position opened successfully: {position_id}")
        return result
    
    def _perform_risk_checks_for_position(self, position: PositionValidator):
        """Perform risk checks specific to positions"""
        # Calculate position value
        position_value = abs(position.size) * position.entry_price
        
        if position_value > self.config.max_position_size:
            raise TradingBotError(
                f"Position value exceeds maximum: {position_value} > {self.config.max_position_size}",
                category=ErrorCategory.TRADING,
                severity=ErrorSeverity.MEDIUM,
                recoverable=True
            )
    
    @handle_errors(category=ErrorCategory.DATA, max_retries=2)
    def get_market_data_safely(self, symbol: str) -> Dict[str, Any]:
        """
        Safely fetch market data with error handling
        """
        try:
            # Validate symbol first
            symbol_validator = validate_trading_symbol(symbol)
            
            # Simulate market data fetch
            if symbol == "INVALID/SYMBOL":
                raise NetworkTradingError(
                    "Market data API error",
                    url="https://api.market.com/ticker",
                    status_code=404
                )
            
            # Return simulated market data
            return {
                "symbol": symbol_validator.symbol,
                "price": 50000.0,
                "volume": 1000000.0,
                "change_24h": 0.025,
                "timestamp": datetime.now().isoformat()
            }
            
        except ValidationTradingError:
            raise
        except NetworkTradingError:
            raise
        except Exception as e:
            raise TradingBotError(
                f"Market data fetch failed: {str(e)}",
                category=ErrorCategory.DATA,
                severity=ErrorSeverity.MEDIUM,
                recoverable=True
            )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including errors"""
        error_stats = error_handler.get_error_statistics()
        
        return {
            "trading_mode": self.config.trading_mode.value,
            "is_running": self.is_running,
            "positions_count": len(self.positions),
            "orders_count": len(self.orders),
            "max_positions": self.config.max_positions,
            "max_position_size": self.config.max_position_size,
            "error_statistics": error_stats,
            "last_update": datetime.now().isoformat()
        }
    
    async def start_trading(self):
        """Start the trading bot with error handling"""
        logger.info("Starting integrated trading bot...")
        self.is_running = True
        
        # Example trading workflow with integrated validation and error handling
        await self._run_trading_cycle()
    
    @handle_errors(category=ErrorCategory.SYSTEM, max_retries=1)
    async def _run_trading_cycle(self):
        """Run a single trading cycle with comprehensive error handling"""
        try:
            # 1. Fetch market data
            market_data = self.get_market_data_safely("BTC/USDT")
            logger.info(f"Market data: BTC/USDT at {market_data['price']}")
            
            # 2. Create test order
            order_result = self.create_validated_order(
                symbol="BTC/USDT",
                order_type="limit",
                side="buy",
                amount=0.1,
                price=market_data['price'] * 0.99  # Buy 1% below market
            )
            
            if order_result["success"]:
                # 3. Open position
                position_result = self.open_validated_position(
                    symbol="BTC/USDT",
                    size=0.1,
                    entry_price=market_data['price'],
                    stop_loss=market_data['price'] * 0.98,
                    take_profit=market_data['price'] * 1.05
                )
                
                logger.info(f"Trading cycle completed successfully")
            
        except TradingBotError as e:
            logger.error(f"Trading cycle failed: {e}")
            if e.severity == ErrorSeverity.CRITICAL:
                self.is_running = False
        except Exception as e:
            logger.error(f"Unexpected error in trading cycle: {e}")
            self.is_running = False
    
    def shutdown(self):
        """Safely shutdown the trading bot"""
        logger.info("Shutting down integrated trading bot...")
        self.is_running = False


async def demo_integrated_trading():
    """Demonstrate integrated validation and error handling"""
    print("🚀 Integrated Trading Bot Demo")
    print("=" * 50)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # 1. Initialize bot with configuration
    print("📝 1. Initializing bot with configuration...")
    config = {
        "trading_mode": "paper",
        "max_position_size": 5000.0,
        "max_positions": 3,
        "max_drawdown": 0.15,
        "stop_loss_percentage": 0.02,
        "take_profit_percentage": 0.05,
        "risk_per_trade": 0.02,
        "exchange_name": "binance"
    }
    
    try:
        bot = IntegratedTradingBot(config)
        print("✅ Bot initialized successfully")
    except Exception as e:
        print(f"❌ Bot initialization failed: {e}")
        return
    
    # 2. Test successful operations
    print("\n💰 2. Testing successful operations...")
    
    # Create valid orders
    orders_to_test = [
        ("BTC/USDT", "limit", "buy", 0.1, 45000.0),
        ("ETH/USDT", "market", "sell", 1.0, None),
    ]
    
    for symbol, order_type, side, amount, price in orders_to_test:
        try:
            result = bot.create_validated_order(symbol, order_type, side, amount, price)
            print(f"✅ Order created: {result['order_id']}")
        except Exception as e:
            print(f"❌ Order failed: {e}")
    
    # 3. Test error scenarios
    print("\n⚠️ 3. Testing error scenarios...")
    
    # Test validation errors
    invalid_operations = [
        ("INVALID", "limit", "buy", 0.1, 45000.0, "Invalid symbol"),
        ("BTC/USDT", "limit", "buy", -0.1, 45000.0, "Negative amount"),
        ("BTC/USDT", "limit", "buy", 0.1, None, "Missing price for limit order"),
        ("FAIL/TEST", "market", "buy", 0.1, None, "Exchange API failure"),
    ]
    
    for symbol, order_type, side, amount, price, description in invalid_operations:
        try:
            result = bot.create_validated_order(symbol, order_type, side, amount, price)
            print(f"❌ {description}: Should have failed but succeeded")
        except Exception as e:
            print(f"✅ {description}: Correctly caught - {type(e).__name__}")
    
    # 4. Test position management
    print("\n📈 4. Testing position management...")
    try:
        position_result = bot.open_validated_position(
            symbol="BTC/USDT",
            size=0.05,
            entry_price=45000.0,
            stop_loss=44000.0,
            take_profit=47000.0
        )
        print(f"✅ Position opened: {position_result['position_id']}")
    except Exception as e:
        print(f"❌ Position failed: {e}")
    
    # 5. Test market data
    print("\n📊 5. Testing market data...")
    try:
        market_data = bot.get_market_data_safely("BTC/USDT")
        print(f"✅ Market data: {market_data['symbol']} at ${market_data['price']:,.2f}")
    except Exception as e:
        print(f"❌ Market data failed: {e}")
    
    # 6. Run trading cycle
    print("\n🔄 6. Running trading cycle...")
    await bot.start_trading()
    
    # 7. Display system status
    print("\n📊 7. System Status:")
    status = bot.get_system_status()
    for key, value in status.items():
        if key != "error_statistics":
            print(f"   {key}: {value}")
    
    print("\n📈 Error Statistics:")
    error_stats = status["error_statistics"]
    for key, value in error_stats.items():
        print(f"   {key}: {value}")
    
    # 8. Shutdown
    bot.shutdown()
    print("\n🎉 Integrated trading bot demo completed!")


if __name__ == "__main__":
    asyncio.run(demo_integrated_trading())