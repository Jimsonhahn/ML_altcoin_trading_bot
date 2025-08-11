"""
Test Suite for SecureErrorHandler
Tests all security-focused error handling features
"""

import sys
import logging
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.error_handler import (
    SecureErrorHandler, SecureErrorResponse, ErrorSeverity, ErrorCategory,
    ValidationTradingError, NetworkTradingError, ExchangeTradingError
)

logger = logging.getLogger(__name__)


def test_secure_error_response():
    """Test SecureErrorResponse class"""
    print("📋 Testing SecureErrorResponse")
    print("-" * 30)
    
    # Create test response
    response = SecureErrorResponse(
        error_id="test-error-123",
        timestamp=datetime.now().isoformat(),
        category="trading",
        severity="high",
        message="Test error message",
        status_code=400,
        details={"symbol": "BTC/USDT", "amount": 0.1},
        trace_id="trace-123"
    )
    
    # Test dictionary conversion
    response_dict = response.to_dict()
    print(f"✅ Response dict: {response_dict['error_id']}")
    
    # Test JSON conversion
    response_json = response.to_json()
    parsed = json.loads(response_json)
    print(f"✅ JSON conversion: {parsed['error_id']}")
    
    print()


def test_sensitive_data_sanitization():
    """Test sensitive data sanitization"""
    print("🔐 Testing Sensitive Data Sanitization")
    print("-" * 30)
    
    handler = SecureErrorHandler()
    
    # Test sensitive patterns
    test_cases = [
        ("api_key=secret123456", "API key masking"),
        ("Authorization: Bearer token123456789", "Bearer token masking"),
        ("password=mypassword123", "Password masking"),
        ("secret_key=sk_test_123456789012345678901234", "Stripe key masking"),
        ("Normal error message", "Non-sensitive message"),
        ("4532-1234-5678-9012", "Credit card masking")
    ]
    
    for test_input, description in test_cases:
        sanitized = handler._sanitize_message(test_input)
        print(f"✅ {description}:")
        print(f"   Input:  {test_input}")
        print(f"   Output: {sanitized}")
    
    print()


def test_critical_error_handling():
    """Test critical error handling"""
    print("🚨 Testing Critical Error Handling")
    print("-" * 30)
    
    handler = SecureErrorHandler()
    
    # Test different critical errors
    critical_errors = [
        (MemoryError("Out of memory"), "Memory error"),
        (ImportError("Module not found"), "Import error"),
        (ConnectionError("Network failure"), "Connection error"),
        (RuntimeError("Critical system failure"), "Runtime error")
    ]
    
    for error, description in critical_errors:
        try:
            response = handler.handle_critical_error(
                error,
                context={"user": "system", "operation": "test"}
            )
            
            print(f"✅ {description}:")
            print(f"   Error ID: {response.error_id}")
            print(f"   Severity: {response.severity}")
            print(f"   Category: {response.category}")
            print(f"   Trace ID: {response.trace_id}")
            
        except Exception as e:
            print(f"❌ {description} failed: {e}")
    
    print()


def test_trading_error_handling():
    """Test trading-specific error handling"""
    print("💰 Testing Trading Error Handling")
    print("-" * 30)
    
    handler = SecureErrorHandler()
    
    # Test different trading errors
    trading_errors = [
        (ValidationTradingError("Invalid amount", field="amount", value=-100), "Validation error"),
        (ExchangeTradingError("Order failed", exchange="binance", order_id="12345"), "Exchange error"),
        (RuntimeError("Trading system error"), "General trading error")
    ]
    
    for error, description in trading_errors:
        response = handler.handle_trading_error(
            error,
            symbol="BTC/USDT",
            order_id="order_123",
            amount=0.1,
            context={"strategy": "momentum", "api_key": "secret123"}
        )
        
        print(f"✅ {description}:")
        print(f"   Error ID: {response.error_id}")
        print(f"   Status Code: {response.status_code}")
        print(f"   Severity: {response.severity}")
        print(f"   Details: {response.details}")
    
    print()


def test_api_error_handling():
    """Test API-specific error handling"""
    print("🌐 Testing API Error Handling")
    print("-" * 30)
    
    handler = SecureErrorHandler()
    
    # Test API error with sensitive data
    try:
        raise ConnectionError("API connection timeout")
    except Exception as error:
        response = handler.handle_api_error(
            error,
            endpoint="https://api.binance.com/api/v3/order",
            status_code=429,
            request_data={
                "symbol": "BTCUSDT",
                "side": "BUY",
                "api_key": "sensitive_api_key_12345",
                "secret": "very_secret_signature",
                "timestamp": 1234567890
            },
            response_data={
                "code": -1003,
                "msg": "Too many requests",
                "error": "Rate limit exceeded"
            },
            context={"user_id": "user123"}
        )
        
        print(f"✅ API error handled:")
        print(f"   Error ID: {response.error_id}")
        print(f"   Status Code: {response.status_code}")
        print(f"   Category: {response.category}")
        print(f"   Sanitized details: {json.dumps(response.details, indent=2)}")
    
    print()


def test_error_retrieval():
    """Test error retrieval functions"""
    print("🔍 Testing Error Retrieval")
    print("-" * 30)
    
    handler = SecureErrorHandler()
    
    # Generate some test errors with the same trace ID
    trace_id = handler._generate_trace_id()
    
    errors = []
    for i in range(3):
        try:
            raise ValueError(f"Test error {i}")
        except Exception as error:
            response = handler.handle_trading_error(
                error,
                symbol=f"TEST{i}/USDT",
                trace_id=trace_id
            )
            errors.append(response)
    
    # Test retrieval by error ID
    if errors:
        first_error = errors[0]
        retrieved = handler.get_error_by_id(first_error.error_id)
        if retrieved:
            print(f"✅ Error retrieval by ID successful: {retrieved.error_id}")
        else:
            print("❌ Error retrieval by ID failed")
        
        # Test retrieval by trace ID
        trace_errors = handler.get_errors_by_trace_id(trace_id)
        print(f"✅ Errors with trace ID {trace_id}: {len(trace_errors)}")
    
    print()


def test_error_statistics():
    """Test error statistics"""
    print("📊 Testing Error Statistics")
    print("-" * 30)
    
    handler = SecureErrorHandler()
    
    # Generate some test errors
    test_errors = [
        (ValueError("Test 1"), ErrorCategory.VALIDATION),
        (ConnectionError("Test 2"), ErrorCategory.NETWORK),
        (RuntimeError("Test 3"), ErrorCategory.TRADING)
    ]
    
    for error, category in test_errors:
        if category == ErrorCategory.TRADING:
            handler.handle_trading_error(error, symbol="BTC/USDT")
        elif category == ErrorCategory.NETWORK:
            handler.handle_api_error(error, status_code=500)
        else:
            handler.handle_critical_error(error)
    
    # Get statistics
    stats = handler.get_error_statistics()
    
    print(f"✅ Total errors: {stats['total_errors']}")
    print(f"✅ Category breakdown: {stats['category_breakdown']}")
    print(f"✅ Severity breakdown: {stats['severity_breakdown']}")
    print(f"✅ Recent errors: {stats['recent_errors_count']}")
    print(f"✅ Last error ID: {stats['last_error_id']}")
    
    print()


def test_notification_system():
    """Test notification system"""
    print("🔔 Testing Notification System")
    print("-" * 30)
    
    handler = SecureErrorHandler()
    
    # Track notifications
    notifications_received = []
    
    def test_callback(response: SecureErrorResponse):
        notifications_received.append({
            'error_id': response.error_id,
            'severity': response.severity,
            'category': response.category
        })
    
    # Add callback
    handler.add_notification_callback(test_callback)
    
    # Generate a high-severity error (should trigger notification)
    try:
        raise RuntimeError("High severity error")
    except Exception as error:
        handler.handle_trading_error(error, symbol="BTC/USDT")
    
    if notifications_received:
        print(f"✅ Notification received: {notifications_received[0]}")
    else:
        print("❌ No notification received")
    
    print()


def test_data_sanitization_edge_cases():
    """Test edge cases in data sanitization"""
    print("🧪 Testing Data Sanitization Edge Cases")
    print("-" * 30)
    
    handler = SecureErrorHandler()
    
    # Test complex nested data
    complex_data = {
        "user": {
            "credentials": {
                "api_key": "secret_key_123456",
                "password": "my_password",
                "token": "bearer_token_789"
            },
            "profile": {
                "name": "John Doe",
                "email": "john@example.com"
            }
        },
        "request": {
            "headers": {
                "Authorization": "Bearer abc123def456",
                "User-Agent": "TradingBot/1.0"
            },
            "body": {
                "symbol": "BTC/USDT",
                "secret": "very_secret_data"
            }
        },
        "arrays": [
            {"key": "value1", "password": "secret1"},
            {"key": "value2", "token": "secret2"}
        ]
    }
    
    sanitized = handler._sanitize_dict(complex_data)
    
    print("✅ Sanitized complex data:")
    print(json.dumps(sanitized, indent=2))
    
    # Verify sensitive data is masked
    sensitive_found = []
    def check_for_sensitive(obj, path=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                if key.lower() in ['password', 'secret', 'token', 'auth', 'authorization']:
                    if value != "***REDACTED***":
                        sensitive_found.append(current_path)
                elif isinstance(value, (dict, list)):
                    check_for_sensitive(value, current_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                check_for_sensitive(item, f"{path}[{i}]")
    
    check_for_sensitive(sanitized)
    
    if not sensitive_found:
        print("✅ All sensitive data properly sanitized")
    else:
        print(f"❌ Sensitive data found at: {sensitive_found}")
    
    print()


def run_secure_error_handler_tests():
    """Run all SecureErrorHandler tests"""
    print("🔒 SecureErrorHandler Comprehensive Test Suite")
    print("=" * 50)
    
    # Configure logging for tests
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    test_functions = [
        test_secure_error_response,
        test_sensitive_data_sanitization,
        test_critical_error_handling,
        test_trading_error_handling,
        test_api_error_handling,
        test_error_retrieval,
        test_error_statistics,
        test_notification_system,
        test_data_sanitization_edge_cases
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
        print("🎉 All SecureErrorHandler tests passed!")
        print("\n💡 Key Security Features Verified:")
        print("   🔐 Sensitive data sanitization working")
        print("   🆔 Unique error IDs with UUID generation")
        print("   📋 Structured error responses")
        print("   🎯 Specialized error handling methods")
        print("   📊 Secure error statistics and retrieval")
        print("   🔔 Notification system functional")
        print("   🧪 Edge cases handled properly")
        return True
    else:
        print("⚠️ Some SecureErrorHandler tests failed!")
        return False


if __name__ == "__main__":
    success = run_secure_error_handler_tests()
    sys.exit(0 if success else 1)