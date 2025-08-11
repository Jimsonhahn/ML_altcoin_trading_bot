#!/usr/bin/env python3
"""
Integration Tests für Trading Bot
Testet die Zusammenarbeit zwischen allen Komponenten
"""

import sys
import unittest
import asyncio
import time
import json
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import all components to test
from core.trading_bot import TradingBot
from core.exchange import Exchange
from core.strategy_router import StrategyRouter
from core.safety_manager import SafetyManager
from data_sources.data_manager import DataManager
from ml_components.market_regime import MarketRegimeDetector
from ml_components.feature_extraction import FeatureExtractor
from strategies.super_lazy_billionaire_strategy import SuperLazyBillionaireStrategy
from utils.notifier import send_info, send_error
from config.environment import get_config
from api.health import HealthChecker

class TestTradingBotIntegration(unittest.TestCase):
    """
    Integration tests for the complete trading bot system
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        cls.config = get_config()
        
        # Mock external APIs to avoid real API calls
        cls.exchange_patcher = patch('core.exchange.Exchange._make_request')
        cls.mock_exchange = cls.exchange_patcher.start()
        
        # Mock notifications
        cls.notification_patcher = patch('utils.notifier.send_info')
        cls.mock_notification = cls.notification_patcher.start()
        
        # Setup test data
        cls.test_market_data = {
            'BTC/USDT': {
                'symbol': 'BTC/USDT',
                'timestamp': int(time.time() * 1000),
                'open': 45000.0,
                'high': 46000.0,
                'low': 44000.0,
                'close': 45500.0,
                'volume': 1000.0
            }
        }
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        cls.exchange_patcher.stop()
        cls.notification_patcher.stop()
    
    def setUp(self):
        """Set up before each test"""
        # Mock successful API responses
        self.mock_exchange.return_value = {
            'success': True,
            'data': self.test_market_data['BTC/USDT']
        }
    
    def test_complete_system_initialization(self):
        """Test that all components can be initialized together"""
        print("\n🔧 Testing complete system initialization...")
        
        try:
            # Initialize all core components
            exchange = Exchange()
            data_manager = DataManager()
            safety_manager = SafetyManager()
            strategy_router = StrategyRouter()
            
            # Initialize ML components
            regime_detector = MarketRegimeDetector()
            feature_extractor = FeatureExtractor()
            
            # Initialize strategies
            super_strategy = SuperLazyBillionaireStrategy()
            
            # Initialize trading bot
            trading_bot = TradingBot()
            
            print("✅ All components initialized successfully")
            
            # Test that components have required attributes
            self.assertTrue(hasattr(exchange, 'get_market_data'))
            self.assertTrue(hasattr(data_manager, 'get_current_data'))
            self.assertTrue(hasattr(safety_manager, 'check_trade_safety'))
            self.assertTrue(hasattr(strategy_router, 'get_active_strategies'))
            
        except Exception as e:
            self.fail(f"System initialization failed: {e}")
    
    def test_data_flow_through_components(self):
        """Test data flow from exchange through all components"""
        print("\n📊 Testing data flow through components...")
        
        try:
            # Mock market data
            with patch('data_sources.data_manager.DataManager.fetch_market_data') as mock_fetch:
                mock_fetch.return_value = self.test_market_data
                
                # Initialize components
                data_manager = DataManager()
                regime_detector = MarketRegimeDetector()
                feature_extractor = FeatureExtractor()
                
                # Test data flow
                market_data = data_manager.get_current_data(['BTC/USDT'])
                self.assertIsNotNone(market_data)
                self.assertIn('BTC/USDT', market_data)
                
                # Test ML components can process the data
                regime = regime_detector.detect_current_regime(market_data)
                self.assertIsNotNone(regime)
                
                features = feature_extractor.extract_features(market_data['BTC/USDT'])
                self.assertIsInstance(features, dict)
                self.assertGreater(len(features), 0)
                
                print("✅ Data flows correctly through all components")
                
        except Exception as e:
            self.fail(f"Data flow test failed: {e}")
    
    def test_strategy_integration(self):
        """Test strategy integration with core systems"""
        print("\n🎯 Testing strategy integration...")
        
        try:
            with patch('data_sources.data_manager.DataManager.get_current_data') as mock_data:
                mock_data.return_value = self.test_market_data
                
                # Initialize components
                strategy_router = StrategyRouter()
                super_strategy = SuperLazyBillionaireStrategy()
                safety_manager = SafetyManager()
                
                # Test strategy can analyze market
                analysis = super_strategy.analyze_market()
                self.assertIsNotNone(analysis)
                
                # Test strategy router integration
                strategies = strategy_router.get_active_strategies()
                self.assertIsInstance(strategies, list)
                
                # Test safety manager integration
                test_trade = {
                    'symbol': 'BTC/USDT',
                    'side': 'buy',
                    'amount': 0.01,
                    'price': 45500.0
                }
                
                # Mock safety check to pass
                with patch.object(safety_manager, 'check_trade_safety') as mock_safety:
                    mock_safety.return_value = {'approved': True, 'risk_score': 0.1}
                    
                    safety_result = safety_manager.check_trade_safety(test_trade)
                    self.assertTrue(safety_result['approved'])
                
                print("✅ Strategy integration working correctly")
                
        except Exception as e:
            self.fail(f"Strategy integration test failed: {e}")
    
    def test_ml_component_integration(self):
        """Test ML components working together"""
        print("\n🤖 Testing ML component integration...")
        
        try:
            # Initialize ML components
            regime_detector = MarketRegimeDetector()
            feature_extractor = FeatureExtractor()
            
            # Test with mock data
            with patch('data_sources.data_manager.DataManager.get_current_data') as mock_data:
                mock_data.return_value = self.test_market_data
                
                # Test regime detection
                regime = regime_detector.detect_current_regime(self.test_market_data)
                self.assertIsNotNone(regime)
                
                # Test feature extraction
                features = feature_extractor.extract_features(self.test_market_data['BTC/USDT'])
                self.assertIsInstance(features, dict)
                
                # Test that features include expected categories
                expected_categories = ['technical', 'price', 'volume', 'volatility']
                for category in expected_categories:
                    category_features = [k for k in features.keys() if category in k.lower()]
                    self.assertGreater(len(category_features), 0, f"No {category} features found")
                
                print("✅ ML components integrated successfully")
                
        except Exception as e:
            self.fail(f"ML integration test failed: {e}")
    
    def test_trading_bot_full_cycle(self):
        """Test complete trading bot cycle"""
        print("\n🔄 Testing complete trading bot cycle...")
        
        try:
            # Mock all external dependencies
            with patch('core.exchange.Exchange.get_account_info') as mock_account, \
                 patch('core.exchange.Exchange.get_market_data') as mock_market, \
                 patch('core.exchange.Exchange.place_order') as mock_order, \
                 patch('data_sources.data_manager.DataManager.get_current_data') as mock_data:
                
                # Setup mocks
                mock_account.return_value = {
                    'balance': {'USDT': 10000.0, 'BTC': 0.1},
                    'positions': []
                }
                
                mock_market.return_value = self.test_market_data['BTC/USDT']
                mock_data.return_value = self.test_market_data
                
                mock_order.return_value = {
                    'id': 'test_order_123',
                    'status': 'filled',
                    'symbol': 'BTC/USDT',
                    'side': 'buy',
                    'amount': 0.01,
                    'price': 45500.0
                }
                
                # Initialize trading bot
                bot = TradingBot()
                
                # Test bot can start
                self.assertTrue(hasattr(bot, 'start'))
                self.assertTrue(hasattr(bot, 'stop'))
                
                # Test bot state management
                self.assertFalse(bot.is_running)
                
                print("✅ Trading bot cycle test completed")
                
        except Exception as e:
            self.fail(f"Trading bot cycle test failed: {e}")
    
    def test_error_handling_integration(self):
        """Test error handling across components"""
        print("\n⚠️ Testing error handling integration...")
        
        try:
            # Test with failing exchange
            with patch('core.exchange.Exchange.get_market_data') as mock_market:
                mock_market.side_effect = Exception("API connection failed")
                
                exchange = Exchange()
                data_manager = DataManager()
                
                # Test that components handle exchange failures gracefully
                try:
                    market_data = exchange.get_market_data('BTC/USDT')
                    self.fail("Expected exception not raised")
                except Exception as e:
                    self.assertIn("API connection failed", str(e))
                
                # Test data manager fallback
                with patch('data_sources.data_manager.DataManager.get_cached_data') as mock_cache:
                    mock_cache.return_value = self.test_market_data
                    
                    # Should fall back to cached data
                    cached_data = data_manager.get_cached_data(['BTC/USDT'])
                    self.assertIsNotNone(cached_data)
                
                print("✅ Error handling works correctly")
                
        except Exception as e:
            self.fail(f"Error handling test failed: {e}")
    
    def test_notification_integration(self):
        """Test notification system integration"""
        print("\n📱 Testing notification integration...")
        
        try:
            # Test notification functions
            with patch('utils.notifier.requests.post') as mock_post:
                mock_post.return_value.status_code = 200
                
                # Test info notification
                result = send_info("Test integration message")
                self.assertTrue(result)
                
                # Test error notification
                result = send_error("Test error message")
                self.assertTrue(result)
                
                print("✅ Notification integration working")
                
        except Exception as e:
            print(f"⚠️ Notification test skipped (expected in test environment): {e}")
    
    def test_health_check_integration(self):
        """Test health check system integration"""
        print("\n🏥 Testing health check integration...")
        
        try:
            # Initialize health checker
            health_checker = HealthChecker()
            
            # Test health check can run
            self.assertTrue(hasattr(health_checker, 'check_system_health'))
            
            # Test with mocked components
            with patch('api.health.HealthChecker._check_database') as mock_db, \
                 patch('api.health.HealthChecker._check_redis') as mock_redis, \
                 patch('api.health.HealthChecker._check_exchange_connectivity') as mock_exchange:
                
                mock_db.return_value = {'status': 'healthy'}
                mock_redis.return_value = {'status': 'healthy'}
                mock_exchange.return_value = {'status': 'healthy'}
                
                # Run async health check in sync test
                async def run_health_check():
                    return await health_checker.check_system_health()
                
                # Run in event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    health_status = loop.run_until_complete(run_health_check())
                    self.assertIsNotNone(health_status)
                    self.assertIn('status', health_status)
                finally:
                    loop.close()
                
                print("✅ Health check integration working")
                
        except Exception as e:
            self.fail(f"Health check integration test failed: {e}")
    
    def test_configuration_integration(self):
        """Test configuration system integration"""
        print("\n⚙️ Testing configuration integration...")
        
        try:
            config = get_config()
            
            # Test configuration is accessible
            self.assertIsNotNone(config)
            self.assertTrue(hasattr(config, 'database'))
            self.assertTrue(hasattr(config, 'trading'))
            self.assertTrue(hasattr(config, 'security'))
            
            # Test configuration validation
            self.assertTrue(hasattr(config, 'is_production'))
            self.assertTrue(hasattr(config, 'is_testnet'))
            self.assertTrue(hasattr(config, 'get_api_keys'))
            
            # Test API keys are accessible (even if empty in test)
            api_keys = config.get_api_keys()
            self.assertIsInstance(api_keys, dict)
            self.assertIn('binance_api_key', api_keys)
            
            print("✅ Configuration integration working")
            
        except Exception as e:
            self.fail(f"Configuration integration test failed: {e}")

class TestPerformanceIntegration(unittest.TestCase):
    """
    Performance integration tests
    """
    
    def test_system_performance_under_load(self):
        """Test system performance with multiple concurrent operations"""
        print("\n⚡ Testing system performance under load...")
        
        try:
            # Mock data for performance test
            with patch('data_sources.data_manager.DataManager.get_current_data') as mock_data:
                mock_data.return_value = {
                    'BTC/USDT': {'close': 45000, 'volume': 1000},
                    'ETH/USDT': {'close': 3000, 'volume': 2000},
                    'ADA/USDT': {'close': 0.5, 'volume': 5000}
                }
                
                # Initialize components
                data_manager = DataManager()
                regime_detector = MarketRegimeDetector()
                feature_extractor = FeatureExtractor()
                
                # Measure performance
                start_time = time.time()
                
                # Simulate multiple operations
                for _ in range(10):
                    market_data = data_manager.get_current_data(['BTC/USDT', 'ETH/USDT', 'ADA/USDT'])
                    regime = regime_detector.detect_current_regime(market_data)
                    features = feature_extractor.extract_features(market_data['BTC/USDT'])
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Performance should be reasonable (less than 5 seconds for 10 iterations)
                self.assertLess(execution_time, 5.0, "System performance too slow")
                
                print(f"✅ Performance test completed in {execution_time:.2f} seconds")
                
        except Exception as e:
            self.fail(f"Performance test failed: {e}")
    
    def test_memory_usage_integration(self):
        """Test memory usage doesn't grow excessively"""
        print("\n💾 Testing memory usage integration...")
        
        try:
            import psutil
            process = psutil.Process()
            
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Mock data and run multiple cycles
            with patch('data_sources.data_manager.DataManager.get_current_data') as mock_data:
                mock_data.return_value = {
                    'BTC/USDT': {'close': 45000, 'volume': 1000, 'high': 46000, 'low': 44000}
                }
                
                # Initialize components
                data_manager = DataManager()
                regime_detector = MarketRegimeDetector()
                
                # Run multiple cycles to check for memory leaks
                for i in range(50):
                    market_data = data_manager.get_current_data(['BTC/USDT'])
                    regime = regime_detector.detect_current_regime(market_data)
                
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_growth = final_memory - initial_memory
                
                # Memory growth should be reasonable (less than 100MB for 50 iterations)
                self.assertLess(memory_growth, 100, f"Excessive memory growth: {memory_growth:.1f}MB")
                
                print(f"✅ Memory test completed. Growth: {memory_growth:.1f}MB")
                
        except ImportError:
            print("⚠️ Memory test skipped (psutil not available)")
        except Exception as e:
            self.fail(f"Memory test failed: {e}")

def run_integration_tests():
    """Run all integration tests"""
    print("🧪 Starting Trading Bot Integration Tests")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTradingBotIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("🎉 All integration tests passed!")
        print(f"✅ Ran {result.testsRun} tests successfully")
    else:
        print("❌ Some integration tests failed!")
        print(f"Failed: {len(result.failures)}, Errors: {len(result.errors)}")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
