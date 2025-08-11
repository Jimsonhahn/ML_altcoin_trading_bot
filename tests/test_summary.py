"""
Simple Test Summary for Trading Bot Components
Quick verification that all major components work
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_secret_manager():
    """Test SecretManager basic functionality"""
    try:
        from utils.secret_manager import SecretManager
        sm = SecretManager("test_summary")
        sm.store_secret("test_key", "test_value")
        result = sm.get_secret("test_key")
        sm.delete_secret("test_key")
        assert result == "test_value"
        return True, "SecretManager works correctly"
    except Exception as e:
        return False, f"SecretManager failed: {e}"

def test_secure_http():
    """Test SecureHTTP basic functionality"""
    try:
        from utils.secure_http import create_secure_session
        session = create_secure_session()
        assert session is not None
        assert hasattr(session, 'request')
        return True, "SecureHTTP creates sessions correctly"
    except Exception as e:
        return False, f"SecureHTTP failed: {e}"

def test_validators():
    """Test Validators basic functionality"""
    try:
        from utils.validators import validate_trading_symbol, validate_amount, validate_order
        
        # Test symbol validation
        symbol = validate_trading_symbol("BTC/USDT")
        assert symbol.symbol == "BTC/USDT"
        
        # Test amount validation
        amount = validate_amount(100.0, "USDT")
        assert amount.amount == 100.0
        
        # Test order validation
        order = validate_order({
            "symbol": "BTC/USDT",
            "order_type": "market",
            "side": "buy",
            "amount": 0.1
        })
        assert order.symbol == "BTC/USDT"
        
        return True, "Validators work correctly"
    except Exception as e:
        return False, f"Validators failed: {e}"

def test_error_handler():
    """Test ErrorHandler basic functionality"""
    try:
        from utils.error_handler import ErrorHandler, safe_execute, TradingBotError
        
        handler = ErrorHandler()
        
        # Test safe execution
        result, error = safe_execute(lambda x: x * 2, 5)
        assert result == 10
        assert error is None
        
        # Test error handling
        result, error = safe_execute(lambda: 1 / 0)
        assert result is None
        assert error is not None
        
        return True, "ErrorHandler works correctly"
    except Exception as e:
        return False, f"ErrorHandler failed: {e}"

def run_summary_tests():
    """Run all summary tests"""
    print("🧪 Trading Bot Component Summary Tests")
    print("=" * 50)
    
    tests = [
        ("Secret Manager", test_secret_manager),
        ("Secure HTTP", test_secure_http),
        ("Validators", test_validators),
        ("Error Handler", test_error_handler),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success, message = test_func()
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status}: {test_name} - {message}")
            results.append((test_name, success))
        except Exception as e:
            print(f"💥 ERROR: {test_name} - {e}")
            results.append((test_name, False))
    
    # Summary
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print("\n📊 Summary")
    print(f"Tests passed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All core components are working correctly!")
        return True
    else:
        print("⚠️ Some components have issues. Check the failures above.")
        return False

if __name__ == "__main__":
    success = run_summary_tests()
    sys.exit(0 if success else 1)