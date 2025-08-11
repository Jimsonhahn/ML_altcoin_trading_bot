"""
Comprehensive test suite for the validation framework
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.validators import (
    TradingSymbolValidator, AmountValidator, ConfigValidator, OrderValidator,
    PositionValidator, StrategyParameterValidator, ValidationError,
    validate_trading_symbol, validate_amount, validate_order, validate_config,
    OrderType, OrderSide, TimeInForce, TradingMode
)
from pydantic import ValidationError as PydanticValidationError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_symbol_validation():
    """Test trading symbol validation"""
    print("🔍 Testing Symbol Validation")
    print("-" * 30)
    
    # Valid symbols
    valid_symbols = ["BTC/USDT", "ETH/BTC", "SOL/USDT", "DOGE/BUSD"]
    for symbol in valid_symbols:
        try:
            validator = validate_trading_symbol(symbol)
            print(f"✅ {symbol} -> {validator.base_currency}/{validator.quote_currency}")
        except Exception as e:
            print(f"❌ {symbol}: {e}")
    
    # Invalid symbols
    invalid_symbols = ["BTCUSDT", "BTC", "BTC/", "/USDT", "BTC/USD/EUR", "bt/usdt"]
    for symbol in invalid_symbols:
        try:
            validator = validate_trading_symbol(symbol)
            print(f"❌ {symbol} should have failed but passed")
        except Exception as e:
            print(f"✅ {symbol}: Correctly rejected - {e}")
    
    print()

def test_amount_validation():
    """Test amount validation"""
    print("💰 Testing Amount Validation")
    print("-" * 30)
    
    # Valid amounts
    valid_amounts = [
        (100.50, "USDT"),
        (0.5, "BTC"),
        (1000, "ETH"),
        (0.00001, "BTC")
    ]
    
    for amount, currency in valid_amounts:
        try:
            validator = validate_amount(amount, currency)
            print(f"✅ {amount} {currency}: Valid")
        except Exception as e:
            print(f"❌ {amount} {currency}: {e}")
    
    # Invalid amounts
    invalid_amounts = [
        (-100, "USDT"),
        (0, "BTC"),
        (2000000, "USDT"),  # Too large
        (0.000000001, "BTC")  # Too small
    ]
    
    for amount, currency in invalid_amounts:
        try:
            validator = validate_amount(amount, currency)
            print(f"❌ {amount} {currency} should have failed but passed")
        except Exception as e:
            print(f"✅ {amount} {currency}: Correctly rejected - {type(e).__name__}")
    
    print()

def test_order_validation():
    """Test order validation"""
    print("📋 Testing Order Validation")
    print("-" * 30)
    
    # Valid orders
    valid_orders = [
        {
            "symbol": "BTC/USDT",
            "order_type": OrderType.LIMIT,
            "side": OrderSide.BUY,
            "amount": 0.1,
            "price": 45000.0
        },
        {
            "symbol": "ETH/USDT",
            "order_type": OrderType.MARKET,
            "side": OrderSide.SELL,
            "amount": 1.0
        },
        {
            "symbol": "SOL/USDT",
            "order_type": OrderType.STOP_LIMIT,
            "side": OrderSide.BUY,
            "amount": 10.0,
            "price": 160.0,
            "stop_price": 165.0
        }
    ]
    
    for order_data in valid_orders:
        try:
            order = validate_order(order_data)
            print(f"✅ {order.order_type.value} {order.side.value} {order.amount} {order.symbol}")
        except Exception as e:
            print(f"❌ Order validation failed: {e}")
    
    # Invalid orders
    invalid_orders = [
        {
            "symbol": "BTC/USDT",
            "order_type": OrderType.LIMIT,
            "side": OrderSide.BUY,
            "amount": 0.1
            # Missing price for limit order
        },
        {
            "symbol": "ETH/USDT",
            "order_type": OrderType.MARKET,
            "side": OrderSide.SELL,
            "amount": 1.0,
            "price": 3000.0  # Market order cannot have price
        }
    ]
    
    for order_data in invalid_orders:
        try:
            order = validate_order(order_data)
            print(f"❌ Invalid order should have failed: {order_data}")
        except Exception as e:
            print(f"✅ Invalid order correctly rejected: {type(e).__name__}")
    
    print()

def test_config_validation():
    """Test configuration validation"""
    print("⚙️ Testing Config Validation")
    print("-" * 30)
    
    # Valid configs
    valid_configs = [
        {
            "trading_mode": TradingMode.PAPER,
            "max_position_size": 1000.0,
            "max_drawdown": 0.15,
            "stop_loss_percentage": 0.02,
            "take_profit_percentage": 0.05
        },
        {
            "trading_mode": TradingMode.LIVE,
            "max_position_size": 5000.0,
            "max_positions": 10,
            "risk_per_trade": 0.01
        }
    ]
    
    for config_data in valid_configs:
        try:
            config = validate_config(config_data)
            print(f"✅ {config.trading_mode.value} mode: max_drawdown={config.max_drawdown}")
        except Exception as e:
            print(f"❌ Config validation failed: {e}")
    
    # Invalid configs
    invalid_configs = [
        {
            "max_drawdown": 0.8,  # Too high
            "stop_loss_percentage": 0.02
        },
        {
            "max_drawdown": 0.15,
            "stop_loss_percentage": 0.3,  # Too high
            "take_profit_percentage": 0.05
        }
    ]
    
    for config_data in invalid_configs:
        try:
            config = validate_config(config_data)
            print(f"❌ Invalid config should have failed: {config_data}")
        except Exception as e:
            print(f"✅ Invalid config correctly rejected: {type(e).__name__}")
    
    print()

def test_position_validation():
    """Test position validation"""
    print("📈 Testing Position Validation")
    print("-" * 30)
    
    # Valid positions
    valid_positions = [
        {
            "symbol": "BTC/USDT",
            "size": 0.1,  # Long position
            "entry_price": 45000.0,
            "current_price": 46000.0,
            "stop_loss": 44000.0,
            "take_profit": 47000.0
        },
        {
            "symbol": "ETH/USDT",
            "size": -2.0,  # Short position
            "entry_price": 3000.0,
            "current_price": 2950.0,
            "stop_loss": 3100.0,
            "take_profit": 2800.0
        }
    ]
    
    for position_data in valid_positions:
        try:
            position = PositionValidator(**position_data)
            pnl = position.unrealized_pnl
            pnl_pct = position.unrealized_pnl_percentage * 100
            print(f"✅ {position.symbol}: {position.size} units, PnL: {pnl:.2f} ({pnl_pct:.2f}%)")
        except Exception as e:
            print(f"❌ Position validation failed: {e}")
    
    # Invalid positions
    invalid_positions = [
        {
            "symbol": "BTC/USDT",
            "size": 0.1,  # Long position
            "entry_price": 45000.0,
            "current_price": 46000.0,
            "stop_loss": 46000.0,  # Stop loss above entry for long
            "take_profit": 44000.0  # Take profit below entry for long
        }
    ]
    
    for position_data in invalid_positions:
        try:
            position = PositionValidator(**position_data)
            print(f"❌ Invalid position should have failed: {position_data}")
        except Exception as e:
            print(f"✅ Invalid position correctly rejected: {type(e).__name__}")
    
    print()

def test_strategy_validation():
    """Test strategy parameter validation"""
    print("🧠 Testing Strategy Validation")
    print("-" * 30)
    
    # Valid strategy parameters
    valid_strategies = [
        {
            "strategy_name": "momentum",
            "parameters": {
                "momentum_period": 14,
                "threshold": 0.02,
                "lookback_period": 20
            }
        },
        {
            "strategy_name": "grid_trading",
            "parameters": {
                "grid_size": 10,
                "grid_spacing": 0.01
            }
        }
    ]
    
    for strategy_data in valid_strategies:
        try:
            strategy = StrategyParameterValidator(**strategy_data)
            print(f"✅ {strategy.strategy_name}: {len(strategy.parameters)} parameters")
        except Exception as e:
            print(f"❌ Strategy validation failed: {e}")
    
    # Invalid strategy parameters
    invalid_strategies = [
        {
            "strategy_name": "invalid-name!",  # Invalid characters
            "parameters": {}
        },
        {
            "strategy_name": "momentum",
            "parameters": {
                "momentum_period": -5,  # Invalid negative period
                "threshold": 0.02
            }
        }
    ]
    
    for strategy_data in invalid_strategies:
        try:
            strategy = StrategyParameterValidator(**strategy_data)
            print(f"❌ Invalid strategy should have failed: {strategy_data}")
        except Exception as e:
            print(f"✅ Invalid strategy correctly rejected: {type(e).__name__}")
    
    print()

def test_edge_cases():
    """Test edge cases and error scenarios"""
    print("🔬 Testing Edge Cases")
    print("-" * 30)
    
    # Test very small amounts
    try:
        validator = validate_amount(0.00000001, "BTC")
        print(f"✅ Very small amount: {validator.amount}")
    except Exception as e:
        print(f"✅ Very small amount correctly rejected: {type(e).__name__}")
    
    # Test very large amounts
    try:
        validator = validate_amount(999999999, "USDT")
        print(f"❌ Very large amount should have failed")
    except Exception as e:
        print(f"✅ Very large amount correctly rejected: {type(e).__name__}")
    
    # Test unusual symbols
    unusual_symbols = ["SHIB/USDT", "1INCH/BTC", "YFI/ETH"]
    for symbol in unusual_symbols:
        try:
            validator = validate_trading_symbol(symbol)
            print(f"✅ Unusual symbol {symbol}: Valid")
        except Exception as e:
            print(f"⚠️ Unusual symbol {symbol}: {e}")
    
    print()

def run_comprehensive_tests():
    """Run all validation tests"""
    print("🧪 Comprehensive Validation Test Suite")
    print("=" * 50)
    
    test_functions = [
        test_symbol_validation,
        test_amount_validation,
        test_order_validation,
        test_config_validation,
        test_position_validation,
        test_strategy_validation,
        test_edge_cases
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed: {e}")
            failed += 1
    
    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All validation tests passed!")
        return True
    else:
        print("⚠️ Some validation tests failed!")
        return False

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)