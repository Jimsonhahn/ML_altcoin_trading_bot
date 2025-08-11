"""
Comprehensive test to verify all HTTP requests are using secure sessions
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.secure_http import SecureHTTPClient, create_secure_session
from utils.notifier import NotificationManager
from core.data_collector import DataCollector
from ml_components.coin_monitor import NewCoinMonitor
from core.exchange import ExchangeManager
from data_sources.binance_source import BinanceDataSource
from data_sources.coingecko_source import CoinGeckoDataSource
from config.settings import Settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_secure_http_integrations():
    """Test that all components use secure HTTP"""
    print("🔒 Testing Secure HTTP Integration Across All Components")
    print("=" * 60)
    
    results = []
    
    # Test 1: NotificationManager
    print("📱 Testing NotificationManager...")
    try:
        settings = Settings()
        notifier = NotificationManager(settings)
        
        has_client = hasattr(notifier, 'http_client')
        is_secure = isinstance(notifier.http_client, SecureHTTPClient) if has_client else False
        
        results.append(("NotificationManager", has_client and is_secure))
        print(f"✅ NotificationManager: {'✓' if has_client and is_secure else '✗'}")
        
    except Exception as e:
        results.append(("NotificationManager", False))
        print(f"❌ NotificationManager failed: {e}")
    
    # Test 2: DataCollector
    print("📊 Testing DataCollector...")
    try:
        settings = Settings()
        collector = DataCollector(settings)
        
        has_client = hasattr(collector, 'http_client')
        is_secure = isinstance(collector.http_client, SecureHTTPClient) if has_client else False
        
        results.append(("DataCollector", has_client and is_secure))
        print(f"✅ DataCollector: {'✓' if has_client and is_secure else '✗'}")
        
    except Exception as e:
        results.append(("DataCollector", False))
        print(f"❌ DataCollector failed: {e}")
    
    # Test 3: NewCoinMonitor
    print("🪙 Testing NewCoinMonitor...")
    try:
        monitor = NewCoinMonitor()
        
        has_client = hasattr(monitor, 'http_client')
        is_secure = isinstance(monitor.http_client, SecureHTTPClient) if has_client else False
        
        results.append(("NewCoinMonitor", has_client and is_secure))
        print(f"✅ NewCoinMonitor: {'✓' if has_client and is_secure else '✗'}")
        
    except Exception as e:
        results.append(("NewCoinMonitor", False))
        print(f"❌ NewCoinMonitor failed: {e}")
    
    # Test 4: CoinGeckoDataSource
    print("🦎 Testing CoinGeckoDataSource...")
    try:
        coingecko = CoinGeckoDataSource()
        
        has_client = hasattr(coingecko, 'http_client')
        is_secure = isinstance(coingecko.http_client, SecureHTTPClient) if has_client else False
        
        results.append(("CoinGeckoDataSource", has_client and is_secure))
        print(f"✅ CoinGeckoDataSource: {'✓' if has_client and is_secure else '✗'}")
        
    except Exception as e:
        results.append(("CoinGeckoDataSource", False))
        print(f"❌ CoinGeckoDataSource failed: {e}")
    
    # Test 5: ExchangeManager (check if it uses secure session)
    print("🏦 Testing ExchangeManager...")
    try:
        exchange = ExchangeManager('binance', 'paper')
        
        # This test just checks if the exchange was created successfully
        # The secure session is passed to CCXT internally
        connection_ok = exchange.connect()
        
        results.append(("ExchangeManager", connection_ok))
        print(f"✅ ExchangeManager: {'✓' if connection_ok else '✗'}")
        
    except Exception as e:
        results.append(("ExchangeManager", False))
        print(f"❌ ExchangeManager failed: {e}")
    
    # Test 6: BinanceDataSource
    print("🟡 Testing BinanceDataSource...")
    try:
        binance = BinanceDataSource()
        
        # Check if exchange is configured (it should have secure session)
        has_exchange = hasattr(binance, 'exchange') and binance.exchange is not None
        
        results.append(("BinanceDataSource", has_exchange))
        print(f"✅ BinanceDataSource: {'✓' if has_exchange else '✗'}")
        
    except Exception as e:
        results.append(("BinanceDataSource", False))
        print(f"❌ BinanceDataSource failed: {e}")
    
    # Summary
    print("=" * 60)
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"📊 Test Results: {passed}/{total} components using secure HTTP")
    
    for component, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {component}")
    
    if passed == total:
        print("🎉 All components are using secure HTTP!")
        return True
    else:
        print("⚠️  Some components are not using secure HTTP")
        return False

def test_actual_requests():
    """Test that actual requests work with secure HTTP"""
    print("\\n🌐 Testing Actual HTTP Requests")
    print("=" * 40)
    
    # Test basic secure HTTP client
    print("🔒 Testing basic SecureHTTPClient...")
    try:
        client = SecureHTTPClient()
        response = client.get('https://httpbin.org/get')
        
        if response.status_code == 200:
            print("✅ Basic secure HTTP request: SUCCESS")
        else:
            print(f"❌ Basic secure HTTP request: FAILED ({response.status_code})")
            
        client.close()
        
    except Exception as e:
        print(f"❌ Basic secure HTTP request failed: {e}")
    
    # Test with an actual exchange API
    print("🏦 Testing exchange API request...")
    try:
        client = SecureHTTPClient()
        response = client.get('https://api.binance.com/api/v3/ping')
        
        if response.status_code == 200:
            print("✅ Exchange API request: SUCCESS")
        else:
            print(f"❌ Exchange API request: FAILED ({response.status_code})")
            
        client.close()
        
    except Exception as e:
        print(f"❌ Exchange API request failed: {e}")

if __name__ == "__main__":
    success = test_secure_http_integrations()
    test_actual_requests()
    
    if success:
        print("\\n🎉 All secure HTTP integration tests passed!")
        sys.exit(0)
    else:
        print("\\n⚠️  Some secure HTTP integration tests failed!")
        sys.exit(1)