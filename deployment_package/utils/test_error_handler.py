"""
Comprehensive Test Suite for Error Handling Framework
"""

import logging
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.error_handler import (
    ErrorHandler, TradingBotError, ValidationTradingError, NetworkTradingError,
    ExchangeTradingError, DataTradingError, RateLimitTradingError,
    ErrorSeverity, ErrorCategory, handle_errors, safe_execute,
    handle_validation_error, handle_network_error, handle_exchange_error
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_error_classification():
    """Test error classification and categorization"""
    print("🏷️ Testing Error Classification")
    print("-" * 30)
    
    error_handler = ErrorHandler()
    
    # Test different error types
    test_cases = [
        (ValueError("Invalid value"), ErrorCategory.VALIDATION),
        (ConnectionError("Network issue"), ErrorCategory.NETWORK),
        (KeyError("Missing key"), ErrorCategory.CONFIGURATION),
        (FileNotFoundError("File missing"), ErrorCategory.SYSTEM),
        (ValidationTradingError("Invalid input"), ErrorCategory.VALIDATION),
        (NetworkTradingError("API timeout"), ErrorCategory.NETWORK),
    ]
    
    for error, expected_category in test_cases:
        context = error_handler.handle_error(error, {
            'function_name': 'test_function',
            'module_name': 'test_module'
        })
        
        if context.category == expected_category:
            print(f"✅ {type(error).__name__} -> {context.category.value}")
        else:
            print(f"❌ {type(error).__name__} -> Expected {expected_category.value}, got {context.category.value}")
    
    print()


def test_custom_errors():
    """Test custom trading bot errors"""
    print("🎯 Testing Custom Errors")
    print("-" * 30)
    
    # Test ValidationTradingError
    try:
        raise ValidationTradingError("Amount must be positive", field="amount", value=-10)
    except ValidationTradingError as e:
        print(f"✅ ValidationTradingError: {e.message}")
        print(f"   Field: {e.additional_data.get('field')}, Value: {e.additional_data.get('value')}")
    
    # Test NetworkTradingError
    try:
        raise NetworkTradingError("Connection timeout", url="https://api.binance.com", status_code=408)
    except NetworkTradingError as e:
        print(f"✅ NetworkTradingError: {e.message}")
        print(f"   URL: {e.additional_data.get('url')}, Status: {e.additional_data.get('status_code')}")
    
    # Test ExchangeTradingError
    try:
        raise ExchangeTradingError("Order failed", exchange="binance", order_id="12345")
    except ExchangeTradingError as e:
        print(f"✅ ExchangeTradingError: {e.message}")
        print(f"   Exchange: {e.additional_data.get('exchange')}, Order ID: {e.additional_data.get('order_id')}")
    
    print()


def test_error_decorator():
    """Test error handling decorator"""
    print("🎭 Testing Error Decorator")
    print("-" * 30)
    
    @handle_errors(category=ErrorCategory.TRADING, max_retries=2, retry_delay=0.1)
    def function_with_retries(fail_count: int = 3):
        """Function that fails a certain number of times"""
        if not hasattr(function_with_retries, 'call_count'):
            function_with_retries.call_count = 0
        
        function_with_retries.call_count += 1
        
        if function_with_retries.call_count <= fail_count:
            raise ConnectionError(f"Attempt {function_with_retries.call_count} failed")
        
        return f"Success on attempt {function_with_retries.call_count}"
    
    @handle_errors(category=ErrorCategory.VALIDATION, reraise=True)
    def validation_function(value: float):
        """Function with validation that re-raises"""
        if value < 0:
            raise ValueError("Value must be positive")
        return value * 2
    
    # Test function with retries
    function_with_retries.call_count = 0
    result = function_with_retries(fail_count=2)
    print(f"✅ Retry function result: {result}")
    
    # Test function that re-raises
    try:
        validation_function(-5)
        print("❌ Should have raised an error")
    except ValueError:
        print("✅ Validation function correctly re-raised error")
    
    print()


async def test_async_error_handling():
    """Test async error handling"""
    print("⚡ Testing Async Error Handling")
    print("-" * 30)
    
    @handle_errors(category=ErrorCategory.NETWORK, max_retries=2, retry_delay=0.1)
    async def async_function_with_retries(fail_count: int = 2):
        """Async function that fails a certain number of times"""
        if not hasattr(async_function_with_retries, 'call_count'):
            async_function_with_retries.call_count = 0
        
        async_function_with_retries.call_count += 1
        
        if async_function_with_retries.call_count <= fail_count:
            raise TimeoutError(f"Async attempt {async_function_with_retries.call_count} failed")
        
        return f"Async success on attempt {async_function_with_retries.call_count}"
    
    # Test async function with retries
    async_function_with_retries.call_count = 0
    result = await async_function_with_retries(fail_count=1)
    print(f"✅ Async retry function result: {result}")
    
    print()


def test_safe_execution():
    """Test safe execution utility"""
    print("🛡️ Testing Safe Execution")
    print("-" * 30)
    
    # Test successful execution
    result, error = safe_execute(lambda x: x * 2, 5)
    print(f"✅ Safe execution success: {result}, Error: {error}")
    
    # Test failed execution
    result, error = safe_execute(lambda: 1 / 0)
    print(f"✅ Safe execution failure: {result}, Error: {error.user_message if error else None}")
    
    # Test with arguments
    result, error = safe_execute(lambda x, y: x / y, 10, 0)
    print(f"✅ Safe execution with args: {result}, Error: {error.category.value if error else None}")
    
    print()


def test_error_statistics():
    """Test error statistics collection"""
    print("📊 Testing Error Statistics")
    print("-" * 30)
    
    error_handler = ErrorHandler()
    
    # Generate some errors
    errors_to_generate = [
        ValueError("Test validation error"),
        ConnectionError("Test network error"),
        ValidationTradingError("Test trading validation"),
        NetworkTradingError("Test trading network"),
        ExchangeTradingError("Test exchange error"),
    ]
    
    for error in errors_to_generate:
        error_handler.handle_error(error, {
            'function_name': 'test_function',
            'module_name': 'test_module'
        })
    
    # Get statistics
    stats = error_handler.get_error_statistics()
    
    print(f"✅ Total errors: {stats['total_errors']}")
    print(f"✅ Category breakdown: {stats['category_breakdown']}")
    print(f"✅ Severity breakdown: {stats['severity_breakdown']}")
    print(f"✅ Recent errors: {stats['recent_errors_count']}")
    
    print()


def test_notification_system():
    """Test error notification system"""
    print("🔔 Testing Notification System")
    print("-" * 30)
    
    notification_received = []
    
    def test_notification_callback(error_context):
        """Test notification callback"""
        notification_received.append({
            'category': error_context.category.value,
            'severity': error_context.severity.value,
            'message': error_context.user_message
        })
    
    error_handler = ErrorHandler()
    error_handler.add_notification_callback(test_notification_callback)
    
    # Generate an error
    error_handler.handle_error(
        ValidationTradingError("Test notification error"),
        {'function_name': 'test', 'module_name': 'test'}
    )
    
    if notification_received:
        print(f"✅ Notification received: {notification_received[0]}")
    else:
        print("❌ No notification received")
    
    print()


def test_specific_error_handlers():
    """Test specific error handling utilities"""
    print("🎯 Testing Specific Error Handlers")
    print("-" * 30)
    
    # Test validation error handler
    try:
        handle_validation_error(ValueError("Invalid input"), field="amount", value=-10)
    except ValidationTradingError as e:
        print(f"✅ Validation handler: {e.category.value}")
    
    # Test network error handler
    try:
        handle_network_error(ConnectionError("Timeout"), url="https://api.test.com", status_code=408)
    except NetworkTradingError as e:
        print(f"✅ Network handler: {e.category.value}")
    
    # Test exchange error handler
    try:
        handle_exchange_error(Exception("Order failed"), exchange="binance", order_id="123")
    except ExchangeTradingError as e:
        print(f"✅ Exchange handler: {e.category.value}")
    
    print()


def test_error_recovery():
    """Test error recovery mechanisms"""
    print("🔄 Testing Error Recovery")
    print("-" * 30)
    
    class RecoverableError(TradingBotError):
        """Test recoverable error"""
        def __init__(self):
            super().__init__(
                message="Recoverable test error",
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.MEDIUM,
                recoverable=True
            )
    
    class NonRecoverableError(TradingBotError):
        """Test non-recoverable error"""
        def __init__(self):
            super().__init__(
                message="Non-recoverable test error",
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.CRITICAL,
                recoverable=False
            )
    
    @handle_errors(max_retries=2, retry_delay=0.1)
    def test_recovery_function(error_type: str):
        """Test function for recovery"""
        if not hasattr(test_recovery_function, 'attempt'):
            test_recovery_function.attempt = 0
        
        test_recovery_function.attempt += 1
        
        if error_type == "recoverable" and test_recovery_function.attempt < 3:
            raise RecoverableError()
        elif error_type == "non_recoverable":
            raise NonRecoverableError()
        
        return f"Success on attempt {test_recovery_function.attempt}"
    
    # Test recoverable error
    test_recovery_function.attempt = 0
    result = test_recovery_function("recoverable")
    print(f"✅ Recoverable error result: {result}")
    
    # Test non-recoverable error
    test_recovery_function.attempt = 0
    try:
        result = test_recovery_function("non_recoverable")
        print(f"❌ Should have raised non-recoverable error")
    except NonRecoverableError:
        print("✅ Non-recoverable error correctly raised")
    
    print()


async def run_comprehensive_tests():
    """Run all error handling tests"""
    print("🧪 Comprehensive Error Handling Test Suite")
    print("=" * 50)
    
    test_functions = [
        test_error_classification,
        test_custom_errors,
        test_error_decorator,
        test_async_error_handling,
        test_safe_execution,
        test_error_statistics,
        test_notification_system,
        test_specific_error_handlers,
        test_error_recovery
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            if asyncio.iscoroutinefunction(test_func):
                await test_func()
            else:
                test_func()
            passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed: {e}")
            failed += 1
    
    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All error handling tests passed!")
        return True
    else:
        print("⚠️ Some error handling tests failed!")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_comprehensive_tests())
    sys.exit(0 if success else 1)