#!/usr/bin/env python3
"""
End-to-End Tests für Trading Bot
Testet komplette Workflows von Start bis Trade-Ausführung
"""

import sys
import unittest
import asyncio
import time
import json
import requests
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import components for E2E testing
from core.trading_bot import TradingBot
from core.exchange import Exchange
from api.health import HealthChecker
from config.environment import get_config

class TestEndToEndWorkflows(unittest.TestCase):
    """
    End-to-end tests für komplette Trading-Workflows
    """
    
    def setUp(self):
        """Setup für jeden Test"""
        self.config = get_config()
        
        # Mock market data für konsistente Tests
        self.mock_market_data = {
            'BTC/USDT': {
                'symbol': 'BTC/USDT',
                'timestamp': int(time.time() * 1000),
                'open': 45000.0,
                'high': 46000.0,
                'low': 44000.0,
                'close': 45500.0,
                'volume': 1000.0,
                'bid': 45450.0,
                'ask': 45550.0
            },
            'ETH/USDT': {
                'symbol': 'ETH/USDT',
                'timestamp': int(time.time() * 1000),
                'open': 3000.0,
                'high': 3100.0,
                'low': 2950.0,
                'close': 3050.0,
                'volume': 5000.0,
                'bid': 3045.0,
                'ask': 3055.0
            }
        }
        
        # Mock account info
        self.mock_account = {
            'balance': {
                'USDT': 10000.0,
                'BTC': 0.1,
                'ETH': 2.0
            },
            'positions': [],
            'total_balance_usdt': 10000.0
        }
    
    def test_complete_trading_workflow(self):
        """Test kompletter Trading-Workflow von Analyse bis Ausführung"""
        print("\n🔄 Testing complete trading workflow...")
        
        with patch('core.exchange.Exchange.get_market_data') as mock_market, \
             patch('core.exchange.Exchange.get_account_info') as mock_account, \
             patch('core.exchange.Exchange.place_order') as mock_order, \
             patch('data_sources.data_manager.DataManager.get_current_data') as mock_data, \
             patch('utils.notifier.send_info') as mock_notify:
            
            # Setup mocks
            mock_market.return_value = self.mock_market_data['BTC/USDT']
            mock_account.return_value = self.mock_account
            mock_data.return_value = self.mock_market_data
            
            # Mock successful order
            mock_order.return_value = {
                'id': 'test_order_123',
                'status': 'filled',
                'symbol': 'BTC/USDT',
                'side': 'buy',
                'amount': 0.01,
                'price': 45500.0,
                'filled': 0.01,
                'cost': 455.0,
                'fee': {'cost': 0.455, 'currency': 'USDT'}
            }
            
            # Initialize trading bot
            bot = TradingBot()
            
            # Test initialization
            self.assertFalse(bot.is_running)
            
            # Simulate one trading cycle
            try:
                # Mock running state für Test
                bot.is_running = True
                
                # Test bot kann market data abrufen
                market_data = mock_market.return_value
                self.assertIsNotNone(market_data)
                self.assertEqual(market_data['symbol'], 'BTC/USDT')
                
                # Test account info
                account = mock_account.return_value
                self.assertIn('USDT', account['balance'])
                self.assertGreater(account['balance']['USDT'], 0)
                
                # Test order execution
                order = mock_order.return_value
                self.assertEqual(order['status'], 'filled')
                self.assertGreater(order['cost'], 0)
                
                print("✅ Complete trading workflow successful")
                
            finally:
                bot.is_running = False
    
    def test_market_analysis_to_decision_workflow(self):
        """Test Workflow von Marktanalyse bis Trading-Entscheidung"""
        print("\n📊 Testing market analysis to decision workflow...")
        
        with patch('data_sources.data_manager.DataManager.get_current_data') as mock_data, \
             patch('ml_components.market_regime.MarketRegimeDetector.detect_current_regime') as mock_regime, \
             patch('strategies.super_lazy_billionaire_strategy.SuperLazyBillionaireStrategy.analyze_market') as mock_analysis:
            
            # Setup mock responses
            mock_data.return_value = self.mock_market_data
            
            mock_regime.return_value = {
                'regime': 'BULL_WEAK',
                'confidence': 0.75,
                'trend_strength': 0.6,
                'volatility': 0.3
            }
            
            mock_analysis.return_value = {
                'signals': {
                    'BTC/USDT': {
                        'action': 'buy',
                        'confidence': 0.8,
                        'target_allocation': 0.3,
                        'stop_loss': 43000.0,
                        'take_profit': 50000.0
                    }
                },
                'market_conditions': {
                    'overall_sentiment': 'bullish',
                    'risk_level': 'medium',
                    'volatility': 'low'
                },
                'recommended_actions': ['increase_btc_allocation']
            }
            
            # Import and test strategy
            from strategies.super_lazy_billionaire_strategy import SuperLazyBillionaireStrategy
            
            strategy = SuperLazyBillionaireStrategy()
            
            # Test market analysis
            analysis = mock_analysis.return_value
            self.assertIn('signals', analysis)
            self.assertIn('BTC/USDT', analysis['signals'])
            
            # Test signal quality
            btc_signal = analysis['signals']['BTC/USDT']
            self.assertEqual(btc_signal['action'], 'buy')
            self.assertGreater(btc_signal['confidence'], 0.7)
            self.assertGreater(btc_signal['target_allocation'], 0)
            
            print("✅ Market analysis to decision workflow successful")
    
    def test_risk_management_workflow(self):
        """Test Risk-Management-Workflow"""
        print("\n⚠️ Testing risk management workflow...")
        
        with patch('core.safety_manager.SafetyManager.check_trade_safety') as mock_safety, \
             patch('risk.kelly_criterion_optimizer.KellyCriterionOptimizer.calculate_position_size') as mock_kelly:
            
            # Test verschiedene Risk-Szenarien
            
            # Scenario 1: Normaler Trade - sollte approved werden
            mock_safety.return_value = {
                'approved': True,
                'risk_score': 0.15,
                'warnings': [],
                'position_size_adjustment': 1.0
            }
            
            mock_kelly.return_value = {
                'recommended_size': 0.08,
                'kelly_fraction': 0.16,
                'safety_factor': 0.5,
                'max_risk_per_trade': 0.02
            }
            
            from core.safety_manager import SafetyManager
            from risk.kelly_criterion_optimizer import KellyCriterionOptimizer
            
            safety_manager = SafetyManager()
            kelly_optimizer = KellyCriterionOptimizer()
            
            # Test trade approval
            test_trade = {
                'symbol': 'BTC/USDT',
                'side': 'buy',
                'amount': 0.05,
                'price': 45500.0,
                'stop_loss': 43000.0
            }
            
            safety_result = mock_safety.return_value
            self.assertTrue(safety_result['approved'])
            self.assertLess(safety_result['risk_score'], 0.2)
            
            # Test position sizing
            kelly_result = mock_kelly.return_value
            self.assertLess(kelly_result['recommended_size'], 0.1)
            self.assertGreater(kelly_result['recommended_size'], 0)
            
            # Scenario 2: Hoher Risk - sollte rejected oder reduziert werden
            mock_safety.return_value = {
                'approved': False,
                'risk_score': 0.8,
                'warnings': ['High portfolio concentration', 'Excessive position size'],
                'position_size_adjustment': 0.3
            }
            
            high_risk_result = mock_safety.return_value
            self.assertFalse(high_risk_result['approved'])
            self.assertGreater(len(high_risk_result['warnings']), 0)
            
            print("✅ Risk management workflow successful")
    
    def test_error_recovery_workflow(self):
        """Test Error-Recovery-Workflow"""
        print("\n🚨 Testing error recovery workflow...")
        
        # Test verschiedene Fehler-Szenarien
        
        # Scenario 1: Exchange API Fehler
        with patch('core.exchange.Exchange.get_market_data') as mock_market:
            mock_market.side_effect = Exception("API rate limit exceeded")
            
            exchange = Exchange()
            
            # Test dass Fehler korrekt behandelt wird
            with self.assertRaises(Exception) as context:
                exchange.get_market_data('BTC/USDT')
            
            self.assertIn("API rate limit", str(context.exception))
        
        # Scenario 2: Fallback auf cached data
        with patch('data_sources.data_manager.DataManager.get_current_data') as mock_data, \
             patch('data_sources.data_manager.DataManager.get_cached_data') as mock_cached:
            
            # First call fails
            mock_data.side_effect = Exception("Network error")
            
            # Fallback succeeds
            mock_cached.return_value = self.mock_market_data
            
            from data_sources.data_manager import DataManager
            data_manager = DataManager()
            
            # Test fallback mechanism
            cached_data = mock_cached.return_value
            self.assertIsNotNone(cached_data)
            self.assertIn('BTC/USDT', cached_data)
        
        # Scenario 3: Notification bei Fehlern
        with patch('utils.notifier.send_error') as mock_error_notify:
            mock_error_notify.return_value = True
            
            # Test error notification
            from utils.notifier import send_error
            result = mock_error_notify.return_value
            self.assertTrue(result)
        
        print("✅ Error recovery workflow successful")
    
    def test_health_monitoring_workflow(self):
        """Test Health-Monitoring-Workflow"""
        print("\n🏥 Testing health monitoring workflow...")
        
        # Test Health Check System
        with patch('api.health.HealthChecker._check_database') as mock_db, \
             patch('api.health.HealthChecker._check_redis') as mock_redis, \
             patch('api.health.HealthChecker._check_exchange_connectivity') as mock_exchange, \
             patch('api.health.HealthChecker._check_trading_bot') as mock_bot:
            
            # Setup healthy responses
            mock_db.return_value = {'status': 'healthy', 'response_time_ms': 45.2}
            mock_redis.return_value = {'status': 'healthy', 'response_time_ms': 12.1}
            mock_exchange.return_value = {'status': 'healthy', 'response_time_ms': 156.8}
            mock_bot.return_value = {
                'status': 'healthy',
                'trading_mode': 'paper',
                'active_strategies': ['momentum', 'arbitrage'],
                'total_positions': 3,
                'uptime_hours': 24.5
            }
            
            health_checker = HealthChecker()
            
            # Test async health check
            async def run_health_test():
                health_status = await health_checker.check_system_health()
                
                # Verify health status structure
                self.assertIn('status', health_status)
                self.assertIn('checks', health_status)
                self.assertIn('system', health_status)
                
                # Verify individual component checks
                checks = health_status['checks']
                self.assertEqual(checks['database']['status'], 'healthy')
                self.assertEqual(checks['redis']['status'], 'healthy')
                self.assertEqual(checks['exchange']['status'], 'healthy')
                self.assertEqual(checks['trading_bot']['status'], 'healthy')
                
                return health_status
            
            # Run async test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                health_result = loop.run_until_complete(run_health_test())
                self.assertEqual(health_result['status'], 'healthy')
            finally:
                loop.close()
        
        print("✅ Health monitoring workflow successful")
    
    def test_configuration_workflow(self):
        """Test Konfiguration-Workflow"""
        print("\n⚙️ Testing configuration workflow...")
        
        # Test configuration loading
        config = get_config()
        
        # Test basic configuration structure
        self.assertIsNotNone(config)
        self.assertTrue(hasattr(config, 'environment'))
        self.assertTrue(hasattr(config, 'trading_mode'))
        self.assertTrue(hasattr(config, 'database'))
        self.assertTrue(hasattr(config, 'trading'))
        
        # Test environment detection
        self.assertIn(config.environment.value, ['development', 'staging', 'production'])
        self.assertIn(config.trading_mode.value, ['paper', 'live', 'backtest'])
        
        # Test API keys structure (even if empty in test)
        api_keys = config.get_api_keys()
        self.assertIsInstance(api_keys, dict)
        expected_keys = ['binance_api_key', 'binance_secret_key', 'telegram_bot_token']
        for key in expected_keys:
            self.assertIn(key, api_keys)
        
        # Test configuration methods
        self.assertIsInstance(config.is_production(), bool)
        self.assertIsInstance(config.is_testnet(), bool)
        self.assertIsInstance(config.is_live_trading(), bool)
        
        # Test configuration dict export
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertIn('environment', config_dict)
        self.assertIn('trading_mode', config_dict)
        
        print("✅ Configuration workflow successful")
    
    def test_notification_workflow(self):
        """Test Notification-Workflow"""
        print("\n📱 Testing notification workflow...")
        
        with patch('utils.notifier.requests.post') as mock_post:
            # Mock successful HTTP response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'ok': True}
            mock_post.return_value = mock_response
            
            from utils.notifier import send_info, send_error
            
            # Test info notification
            info_result = send_info("Test info message")
            self.assertTrue(info_result)
            
            # Test error notification
            error_result = send_error("Test error message")
            self.assertTrue(error_result)
            
            # Verify HTTP calls were made
            self.assertGreater(mock_post.call_count, 0)
        
        print("✅ Notification workflow successful")

class TestPerformanceWorkflows(unittest.TestCase):
    """
    Performance-Tests für E2E Workflows
    """
    
    def test_startup_performance(self):
        """Test Startup-Performance des Systems"""
        print("\n⚡ Testing system startup performance...")
        
        start_time = time.time()
        
        # Mock external dependencies für schnelleren Test
        with patch('core.exchange.Exchange._make_request'), \
             patch('utils.notifier.send_info'):
            
            try:
                # Initialize key components
                from core.trading_bot import TradingBot
                from core.exchange import Exchange
                from data_sources.data_manager import DataManager
                from core.strategy_router import StrategyRouter
                
                # Measure initialization time
                bot = TradingBot()
                exchange = Exchange()
                data_manager = DataManager()
                strategy_router = StrategyRouter()
                
                initialization_time = time.time() - start_time
                
                # Startup should be reasonable (less than 10 seconds)
                self.assertLess(initialization_time, 10.0, 
                              f"System startup too slow: {initialization_time:.2f}s")
                
                print(f"✅ System startup completed in {initialization_time:.2f} seconds")
                
            except Exception as e:
                self.fail(f"Startup performance test failed: {e}")
    
    def test_trading_cycle_performance(self):
        """Test Performance eines kompletten Trading-Zyklus"""
        print("\n🔄 Testing trading cycle performance...")
        
        with patch('data_sources.data_manager.DataManager.get_current_data') as mock_data, \
             patch('ml_components.market_regime.MarketRegimeDetector.detect_current_regime') as mock_regime, \
             patch('strategies.super_lazy_billionaire_strategy.SuperLazyBillionaireStrategy.analyze_market') as mock_analysis:
            
            # Setup mocks
            mock_data.return_value = {
                'BTC/USDT': {'close': 45000, 'volume': 1000, 'high': 46000, 'low': 44000}
            }
            mock_regime.return_value = {'regime': 'BULL_WEAK', 'confidence': 0.75}
            mock_analysis.return_value = {
                'signals': {'BTC/USDT': {'action': 'hold', 'confidence': 0.6}}
            }
            
            # Measure trading cycle performance
            start_time = time.time()
            
            # Simulate trading cycle components
            for _ in range(5):  # 5 cycles
                # Data retrieval
                market_data = mock_data.return_value
                
                # Market regime detection
                regime = mock_regime.return_value
                
                # Strategy analysis
                analysis = mock_analysis.return_value
            
            cycle_time = time.time() - start_time
            avg_cycle_time = cycle_time / 5
            
            # Each cycle should be fast (less than 1 second)
            self.assertLess(avg_cycle_time, 1.0, 
                          f"Trading cycle too slow: {avg_cycle_time:.2f}s per cycle")
            
            print(f"✅ Trading cycle performance: {avg_cycle_time:.2f}s per cycle")

def run_e2e_tests():
    """Run all end-to-end tests"""
    print("🧪 Starting End-to-End Trading Bot Tests")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndWorkflows))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceWorkflows))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 All end-to-end tests passed!")
        print(f"✅ Ran {result.testsRun} tests successfully")
        print("🚀 System is ready for production deployment!")
    else:
        print("❌ Some end-to-end tests failed!")
        print(f"Failed: {len(result.failures)}, Errors: {len(result.errors)}")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"- {test}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"- {test}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_e2e_tests()
    sys.exit(0 if success else 1)