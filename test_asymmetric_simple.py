"""
Simplified Test Suite for High-Octane Asymmetric Engine
=====================================================

Basic test suite without complex dependencies.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_test_market_data(length: int = 100, volatility: float = 0.02) -> pd.DataFrame:
    """Generate realistic test market data"""
    np.random.seed(42)  # For reproducible tests
    
    # Generate price series with realistic patterns
    base_price = 50000  # $50k starting price
    prices = [base_price]
    volumes = []
    
    for i in range(length - 1):
        # Add some trend and mean reversion
        trend = 0.0001 * (i - length/2)  # Slight trend
        noise = np.random.normal(0, volatility)
        
        # Occasional volatility spikes
        if np.random.random() < 0.05:  # 5% chance
            noise *= 3  # 3x volatility spike
        
        price_change = (trend + noise) * prices[-1]
        new_price = max(prices[-1] + price_change, 1000)  # Don't go below $1000
        prices.append(new_price)
        
        # Volume correlates with price changes
        volume_base = 1000000
        volume_spike = abs(price_change / prices[-1]) * 5000000
        volume = volume_base + volume_spike + np.random.normal(0, 200000)
        volumes.append(max(volume, 10000))
    
    # Create OHLCV data
    data = []
    for i in range(length):
        if i == 0:
            open_price = prices[i]
        else:
            open_price = prices[i-1]
        
        close_price = prices[i]
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
        volume = volumes[i] if i < len(volumes) else 1000000
        
        data.append({
            'timestamp': datetime.now() - timedelta(hours=length-i),
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    return pd.DataFrame(data)


class SimpleAsymmetricTest:
    """Simplified test for core components"""
    
    def __init__(self):
        self.test_results = {}
        
    async def run_all_tests(self):
        """Run simplified test suite"""
        logger.info("🧪 Starting Simplified Asymmetric Engine Test Suite")
        logger.info("=" * 50)
        
        try:
            # Test core imports
            await self.test_imports()
            await self.test_risk_manager()
            await self.test_high_octane_engine()
            
            # Display results
            self.display_test_results()
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            import traceback
            traceback.print_exc()
    
    async def test_imports(self):
        """Test that core modules can be imported"""
        logger.info("📦 Testing Core Imports...")
        
        try:
            # Test risk manager import
            from core.asymmetric_risk_manager import AsymmetricRiskManager, RiskLevel
            risk_manager = AsymmetricRiskManager()
            assert risk_manager is not None
            
            # Test engine import  
            from strategies.high_octane_asymmetric_engine import HighOctaneAsymmetricEngine
            engine = HighOctaneAsymmetricEngine()
            assert engine is not None
            
            self.test_results['imports'] = {
                'status': 'PASSED',
                'details': 'All core modules imported successfully'
            }
            
        except Exception as e:
            self.test_results['imports'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"Import test failed: {e}")
    
    async def test_risk_manager(self):
        """Test asymmetric risk manager"""
        logger.info("🛡️ Testing Risk Manager...")
        
        try:
            from core.asymmetric_risk_manager import AsymmetricRiskManager
            risk_manager = AsymmetricRiskManager()
            
            # Test position sizing
            signal_data = {
                'position_size': 0.10,
                'confidence': 0.8,
                'leverage': 2.0
            }
            
            conservative_size = risk_manager.calculate_position_size(signal_data, 'conservative', 10000)
            aggressive_size = risk_manager.calculate_position_size(signal_data, 'aggressive', 10000)
            
            assert conservative_size <= 0.02, "Conservative size too large"
            assert aggressive_size <= 0.15, "Aggressive size too large"
            assert conservative_size < aggressive_size, "Conservative should be smaller"
            
            # Test trade validation
            trade_data = {
                'risk_tier': 'aggressive',
                'position_size': 0.05,
                'leverage': 3.0,
                'symbol': 'BTC/USDT'
            }
            
            is_valid, issues = risk_manager.validate_trade(trade_data)
            assert is_valid, f"Valid trade rejected: {issues}"
            
            self.test_results['risk_manager'] = {
                'status': 'PASSED',
                'details': 'Risk management validation working correctly'
            }
            
        except Exception as e:
            self.test_results['risk_manager'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"Risk manager test failed: {e}")
    
    async def test_high_octane_engine(self):
        """Test the main asymmetric engine"""
        logger.info("⚙️ Testing High-Octane Engine...")
        
        try:
            from strategies.high_octane_asymmetric_engine import HighOctaneAsymmetricEngine
            engine = HighOctaneAsymmetricEngine()
            
            # Test signal generation
            test_data = generate_test_market_data(100, 0.03)
            current_price = test_data['close'].iloc[-1]
            
            action, signal_data = engine.calculate_signal('BTC/USDT', test_data, current_price)
            
            assert action in ['BUY', 'SELL', 'HOLD'], "Invalid action returned"
            assert isinstance(signal_data, dict), "Signal data should be dict"
            
            # Test performance tracking
            test_trade_result = {
                'pnl': 0.05,  # 5% profit
                'risk_tier': 'aggressive',
                'strategy_name': 'LeverageBreakoutHunter'
            }
            
            engine.update_performance(test_trade_result)
            performance = engine.get_performance_summary()
            
            assert 'allocations' in performance, "Missing allocations in performance"
            assert 'daily_stats' in performance, "Missing daily stats"
            
            self.test_results['high_octane_engine'] = {
                'status': 'PASSED',
                'details': f'Engine working, action: {action}'
            }
            
        except Exception as e:
            self.test_results['high_octane_engine'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"High-octane engine test failed: {e}")
    
    def display_test_results(self):
        """Display test results"""
        logger.info("\n" + "=" * 60)
        logger.info("🧪 SIMPLIFIED TEST RESULTS")
        logger.info("=" * 60)
        
        passed = 0
        failed = 0
        
        for test_name, result in self.test_results.items():
            status = result['status']
            if status == 'PASSED':
                logger.info(f"✅ {test_name}: {status}")
                passed += 1
            else:
                logger.error(f"❌ {test_name}: {status}")
                failed += 1
            
            if 'details' in result:
                logger.info(f"   📝 {result['details']}")
            if 'error' in result:
                logger.error(f"   💥 {result['error']}")
        
        logger.info("-" * 60)
        logger.info(f"📊 Test Summary: {passed} passed, {failed} failed")
        
        if failed == 0:
            logger.info("🎉 ALL CORE TESTS PASSED! Basic system is functional.")
        else:
            logger.error(f"⚠️ {failed} tests failed. Check dependencies.")


async def main():
    """Run the simplified test suite"""
    test_suite = SimpleAsymmetricTest()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())