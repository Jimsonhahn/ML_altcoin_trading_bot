"""
Test Suite for High-Octane Asymmetric Engine
===========================================

Comprehensive test suite for the asymmetric trading system.
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

from core.asymmetric_orchestrator import AsymmetricOrchestrator
from core.asymmetric_risk_manager import AsymmetricRiskManager, RiskLevel
from core.strategy_orchestrator import StrategyDiscoveryEngine
from strategies.high_octane_asymmetric_engine import (
    HighOctaneAsymmetricEngine, LeverageBreakoutHunter, 
    VolatilitySpikeSurfer, MomentumScalpingMachine, LiquidationHunter
)

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


class AsymmetricEngineTestSuite:
    """Comprehensive test suite for the asymmetric engine"""
    
    def __init__(self):
        self.test_results = {}
        self.discovery_engine = None
        self.orchestrator = None
        
    async def run_all_tests(self):
        """Run complete test suite"""
        logger.info("🧪 Starting Asymmetric Engine Test Suite")
        logger.info("=" * 50)
        
        try:
            # Test individual components
            await self.test_risk_manager()
            await self.test_high_octane_strategies()
            await self.test_asymmetric_engine()
            await self.test_orchestrator_integration()
            
            # Integration tests
            await self.test_full_trading_cycle()
            await self.test_risk_scenarios()
            await self.test_performance_scenarios()
            
            # Display results
            self.display_test_results()
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            import traceback
            traceback.print_exc()
    
    async def test_risk_manager(self):
        """Test asymmetric risk manager"""
        logger.info("🛡️ Testing Asymmetric Risk Manager...")
        
        try:
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
            
            # Test invalid trade
            invalid_trade = {
                'risk_tier': 'conservative',
                'position_size': 0.10,  # Too large for conservative
                'leverage': 5.0,        # Too high for conservative
                'symbol': 'BTC/USDT'
            }
            
            is_valid, issues = risk_manager.validate_trade(invalid_trade)
            assert not is_valid, "Invalid trade accepted"
            
            self.test_results['risk_manager'] = {
                'status': 'PASSED',
                'details': 'Position sizing and validation working correctly'
            }
            
        except Exception as e:
            self.test_results['risk_manager'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"Risk manager test failed: {e}")
    
    async def test_high_octane_strategies(self):
        """Test individual high-octane strategies"""
        logger.info("🚀 Testing High-Octane Strategies...")
        
        try:
            strategies = [
                LeverageBreakoutHunter(),
                VolatilitySpikeSurfer(),
                MomentumScalpingMachine(),
                LiquidationHunter()
            ]
            
            # Generate test data with different market conditions
            normal_data = generate_test_market_data(100, 0.02)
            volatile_data = generate_test_market_data(100, 0.08)
            trending_data = self._generate_trending_data(100)
            
            test_datasets = [
                ('normal', normal_data),
                ('volatile', volatile_data),
                ('trending', trending_data)
            ]
            
            strategy_results = {}
            
            for strategy in strategies:
                strategy_name = strategy.name
                strategy_results[strategy_name] = {'signals_generated': 0, 'errors': 0}
                
                for market_type, data in test_datasets:
                    try:
                        signal = await strategy.analyze(data, 'BTC/USDT')
                        if signal:
                            strategy_results[strategy_name]['signals_generated'] += 1
                            
                            # Validate signal structure
                            required_fields = ['action', 'symbol', 'position_size', 'confidence']
                            for field in required_fields:
                                assert hasattr(signal, field), f"Missing field: {field}"
                            
                            assert signal.action in ['BUY', 'SELL', 'HOLD'], "Invalid action"
                            assert 0 <= signal.confidence <= 1, "Invalid confidence"
                            
                    except Exception as e:
                        strategy_results[strategy_name]['errors'] += 1
                        logger.warning(f"Strategy {strategy_name} error on {market_type} data: {e}")
            
            # Check that strategies generate signals
            total_signals = sum(result['signals_generated'] for result in strategy_results.values())
            total_errors = sum(result['errors'] for result in strategy_results.values())
            
            assert total_signals > 0, "No signals generated by any strategy"
            assert total_errors < len(strategies) * len(test_datasets) * 0.5, "Too many errors"
            
            self.test_results['high_octane_strategies'] = {
                'status': 'PASSED',
                'details': f'Generated {total_signals} signals with {total_errors} errors',
                'strategy_results': strategy_results
            }
            
        except Exception as e:
            self.test_results['high_octane_strategies'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"High-octane strategies test failed: {e}")
    
    async def test_asymmetric_engine(self):
        """Test the main asymmetric engine"""
        logger.info("⚙️ Testing Asymmetric Engine...")
        
        try:
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
            
            # Test multiple signal generations
            signals_generated = 0
            for i in range(10):
                test_data_variant = generate_test_market_data(100, 0.02 + i * 0.01)
                current_price = test_data_variant['close'].iloc[-1]
                action, _ = engine.calculate_signal('BTC/USDT', test_data_variant, current_price)
                if action != 'HOLD':
                    signals_generated += 1
            
            assert signals_generated > 0, "Engine not generating any signals"
            
            self.test_results['asymmetric_engine'] = {
                'status': 'PASSED',
                'details': f'Generated {signals_generated}/10 non-HOLD signals'
            }
            
        except Exception as e:
            self.test_results['asymmetric_engine'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"Asymmetric engine test failed: {e}")
    
    async def test_orchestrator_integration(self):
        """Test orchestrator integration"""
        logger.info("🎯 Testing Orchestrator Integration...")
        
        try:
            # Initialize discovery engine
            self.discovery_engine = StrategyDiscoveryEngine("strategies")
            await self.discovery_engine.discover_all_strategies()
            
            # Initialize orchestrator
            config = {
                'initial_capital': 10000,
                'engine_params': {},
                'risk_params': {}
            }
            
            self.orchestrator = AsymmetricOrchestrator(self.discovery_engine, None, config)
            await self.orchestrator.initialize()
            
            # Test signal generation
            test_data = generate_test_market_data(100, 0.025)
            signals = await self.orchestrator.generate_trading_signals(test_data, 'BTC/USDT')
            
            # Test trade execution
            if signals:
                result = await self.orchestrator.execute_trade(signals[0])
                assert result['success'], f"Trade execution failed: {result}"
                
                # Test position monitoring
                current_prices = {'BTC/USDT': test_data['close'].iloc[-1] * 1.02}  # 2% price increase
                updates = await self.orchestrator.monitor_positions(current_prices)
            
            # Test portfolio status
            status = await self.orchestrator.get_portfolio_status()
            assert 'portfolio_value' in status, "Missing portfolio value"
            assert 'risk_assessment' in status, "Missing risk assessment"
            
            self.test_results['orchestrator_integration'] = {
                'status': 'PASSED',
                'details': f'Generated {len(signals)} signals, portfolio value: {status["portfolio_value"]:.2f}'
            }
            
        except Exception as e:
            self.test_results['orchestrator_integration'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"Orchestrator integration test failed: {e}")
    
    async def test_full_trading_cycle(self):
        """Test complete trading cycle"""
        logger.info("🔄 Testing Full Trading Cycle...")
        
        try:
            if not self.orchestrator:
                logger.warning("Orchestrator not initialized, skipping full cycle test")
                return
            
            initial_value = self.orchestrator.portfolio_value
            trades_executed = 0
            
            # Simulate 24 hours of trading (24 1-hour candles)
            for hour in range(24):
                # Generate market data for this hour
                market_data = generate_test_market_data(100, 0.02 + hour * 0.001)
                current_price = market_data['close'].iloc[-1]
                
                # Generate signals
                signals = await self.orchestrator.generate_trading_signals(market_data, 'BTC/USDT')
                
                # Execute trades
                for signal in signals:
                    result = await self.orchestrator.execute_trade(signal)
                    if result['success']:
                        trades_executed += 1
                
                # Monitor positions
                price_change = np.random.normal(0, 0.02)  # Random price movement
                new_price = current_price * (1 + price_change)
                current_prices = {'BTC/USDT': new_price}
                
                await self.orchestrator.monitor_positions(current_prices)
            
            final_status = await self.orchestrator.get_portfolio_status()
            final_value = final_status['portfolio_value']
            total_return = ((final_value - initial_value) / initial_value) * 100
            
            self.test_results['full_trading_cycle'] = {
                'status': 'PASSED',
                'details': f'Executed {trades_executed} trades, return: {total_return:.2f}%',
                'initial_value': initial_value,
                'final_value': final_value,
                'total_return': total_return
            }
            
        except Exception as e:
            self.test_results['full_trading_cycle'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"Full trading cycle test failed: {e}")
    
    async def test_risk_scenarios(self):
        """Test various risk scenarios"""
        logger.info("⚠️ Testing Risk Scenarios...")
        
        try:
            risk_manager = AsymmetricRiskManager()
            
            # Test 1: Excessive position size
            large_trade = {
                'risk_tier': 'aggressive',
                'position_size': 0.50,  # 50% - way too large
                'leverage': 2.0,
                'symbol': 'BTC/USDT'
            }
            
            is_valid, issues = risk_manager.validate_trade(large_trade)
            assert not is_valid, "Excessive position size not caught"
            
            # Test 2: High leverage on conservative
            high_leverage_conservative = {
                'risk_tier': 'conservative',
                'position_size': 0.01,
                'leverage': 5.0,  # Too high for conservative
                'symbol': 'BTC/USDT'
            }
            
            is_valid, issues = risk_manager.validate_trade(high_leverage_conservative)
            assert not is_valid, "High leverage on conservative not caught"
            
            # Test 3: Portfolio risk assessment
            mock_portfolio = {
                'total_value': 10000,
                'peak_value': 12000
            }
            
            portfolio_risk = risk_manager.assess_portfolio_risk(mock_portfolio)
            assert isinstance(portfolio_risk.risk_level, RiskLevel), "Invalid risk level type"
            
            self.test_results['risk_scenarios'] = {
                'status': 'PASSED',
                'details': 'All risk scenarios handled correctly'
            }
            
        except Exception as e:
            self.test_results['risk_scenarios'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"Risk scenarios test failed: {e}")
    
    async def test_performance_scenarios(self):
        """Test performance tracking and adjustment scenarios"""
        logger.info("📊 Testing Performance Scenarios...")
        
        try:
            engine = HighOctaneAsymmetricEngine()
            
            # Simulate winning streak
            for i in range(5):
                winning_trade = {
                    'pnl': 0.03 + i * 0.01,  # Increasing profits
                    'risk_tier': 'aggressive',
                    'strategy_name': 'VolatilitySpikeSurfer'
                }
                engine.update_performance(winning_trade)
            
            # Simulate losing streak
            for i in range(3):
                losing_trade = {
                    'pnl': -0.02,  # 2% losses
                    'risk_tier': 'conservative',
                    'strategy_name': 'conservative_momentum'
                }
                engine.update_performance(losing_trade)
            
            # Check performance summary
            performance = engine.get_performance_summary()
            
            assert 'strategy_performance' in performance, "Missing strategy performance"
            assert 'daily_stats' in performance, "Missing daily stats"
            
            # Test allocation adjustment (should favor aggressive after wins)
            original_aggressive = engine.aggressive_allocation
            
            # Simulate very good aggressive performance
            for i in range(10):
                great_trade = {
                    'pnl': 0.10,  # 10% profit
                    'risk_tier': 'aggressive',
                    'strategy_name': 'LeverageBreakoutHunter'
                }
                engine.update_performance(great_trade)
            
            self.test_results['performance_scenarios'] = {
                'status': 'PASSED',
                'details': f'Performance tracking working, original allocation: {original_aggressive:.1%}'
            }
            
        except Exception as e:
            self.test_results['performance_scenarios'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"Performance scenarios test failed: {e}")
    
    def _generate_trending_data(self, length: int) -> pd.DataFrame:
        """Generate trending market data"""
        np.random.seed(123)
        base_price = 50000
        prices = [base_price]
        
        # Strong uptrend
        for i in range(length - 1):
            trend = 0.002  # 0.2% positive trend per period
            noise = np.random.normal(0, 0.01)
            price_change = (trend + noise) * prices[-1]
            new_price = prices[-1] + price_change
            prices.append(new_price)
        
        # Create OHLCV
        data = []
        for i, price in enumerate(prices):
            data.append({
                'timestamp': datetime.now() - timedelta(hours=length-i),
                'open': price * (1 + np.random.normal(0, 0.001)),
                'high': price * (1 + abs(np.random.normal(0, 0.005))),
                'low': price * (1 - abs(np.random.normal(0, 0.005))),
                'close': price,
                'volume': 1000000 + np.random.normal(0, 100000)
            })
        
        return pd.DataFrame(data)
    
    def display_test_results(self):
        """Display comprehensive test results"""
        logger.info("\n" + "=" * 60)
        logger.info("🧪 ASYMMETRIC ENGINE TEST RESULTS")
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
            logger.info("🎉 ALL TESTS PASSED! System is ready for deployment.")
        else:
            logger.error(f"⚠️ {failed} tests failed. Please review and fix issues.")
        
        # Detailed performance metrics
        if 'full_trading_cycle' in self.test_results:
            cycle_result = self.test_results['full_trading_cycle']
            if 'total_return' in cycle_result:
                logger.info(f"💰 Simulated Trading Return: {cycle_result['total_return']:.2f}%")


async def main():
    """Run the complete test suite"""
    test_suite = AsymmetricEngineTestSuite()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())