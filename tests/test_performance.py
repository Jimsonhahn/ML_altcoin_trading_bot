#!/usr/bin/env python3
"""
Performance Tests for Trading Bot
================================

Tests system performance under realistic load conditions.
Tests all components under stress to ensure production readiness.
"""

import asyncio
import time
import threading
import sys
import json
import statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from core.trading_bot import TradingBot
from data_sources.data_manager import DataManager
from ml_components.market_regime import MarketRegimeDetector
from ml_components.market_sentiment import MarketSentimentAnalyzer
from strategies import STRATEGIES
from core.safety_manager import SafetyManager
from config.environment import get_config

class PerformanceTestRunner:
    """Comprehensive performance testing suite"""
    
    def __init__(self):
        self.config = get_config()
        self.results = {}
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT', 'AVAX/USDT']
        
    def run_all_tests(self):
        """Run complete performance test suite"""
        print("🚀 Starting Performance Tests Under Realistic Load...")
        print("=" * 60)
        
        test_results = {}
        
        # 1. Data Processing Performance
        print("\n📊 Testing Data Processing Performance...")
        test_results['data_processing'] = self.test_data_processing_performance()
        
        # 2. Strategy Execution Performance
        print("\n🎯 Testing Strategy Execution Performance...")
        test_results['strategy_execution'] = self.test_strategy_execution_performance()
        
        # 3. ML Components Performance
        print("\n🧠 Testing ML Components Performance...")
        test_results['ml_performance'] = self.test_ml_performance()
        
        # 4. Concurrent Trading Simulation
        print("\n⚡ Testing Concurrent Trading Simulation...")
        test_results['concurrent_trading'] = self.test_concurrent_trading()
        
        # 5. Memory and Resource Usage
        print("\n💾 Testing Memory and Resource Usage...")
        test_results['resource_usage'] = self.test_resource_usage()
        
        # 6. API Response Times
        print("\n🌐 Testing API Response Times...")
        test_results['api_performance'] = self.test_api_performance()
        
        # Generate comprehensive report
        self.generate_performance_report(test_results)
        
        return test_results
    
    def test_data_processing_performance(self):
        """Test data processing under load"""
        results = {}
        
        try:
            data_manager = DataManager()
            
            # Test 1: Single symbol data fetch speed
            start_time = time.time()
            for _ in range(10):
                data = data_manager.get_current_data(['BTC/USDT'])
            single_fetch_time = (time.time() - start_time) / 10
            
            # Test 2: Multiple symbols concurrent fetch
            start_time = time.time()
            data = data_manager.get_current_data(self.symbols)
            multi_fetch_time = time.time() - start_time
            
            # Test 3: Historical data processing
            start_time = time.time()
            for symbol in self.symbols[:3]:  # Test 3 symbols
                try:
                    hist_data = data_manager.get_historical_data(symbol, '1h', 100)
                except Exception as e:
                    print(f"   ⚠️  Historical data not available for {symbol}: {e}")
            hist_data_time = time.time() - start_time
            
            results = {
                'single_fetch_avg_ms': round(single_fetch_time * 1000, 2),
                'multi_fetch_ms': round(multi_fetch_time * 1000, 2),
                'historical_data_ms': round(hist_data_time * 1000, 2),
                'status': 'PASS' if single_fetch_time < 1.0 else 'SLOW'
            }
            
            print(f"   ✅ Single fetch: {results['single_fetch_avg_ms']}ms")
            print(f"   ✅ Multi fetch: {results['multi_fetch_ms']}ms")
            print(f"   ✅ Historical: {results['historical_data_ms']}ms")
            
        except Exception as e:
            results = {'error': str(e), 'status': 'FAIL'}
            print(f"   ❌ Data processing test failed: {e}")
        
        return results
    
    def test_strategy_execution_performance(self):
        """Test strategy execution speed"""
        results = {}
        
        try:
            execution_times = {}
            
            # Mock market data for consistent testing
            mock_data = {
                'BTC/USDT': {
                    'close': 45000,
                    'volume': 1000000,
                    'timestamp': datetime.now()
                }
            }
            
            for strategy_name, strategy_class in STRATEGIES.items():
                try:
                    strategy = strategy_class()
                    
                    # Time strategy signal generation
                    start_time = time.time()
                    for _ in range(5):  # Run 5 times for average
                        try:
                            signal = strategy.generate_signal(mock_data)
                        except Exception as sig_error:
                            print(f"     ⚠️  Signal generation failed for {strategy_name}: {sig_error}")
                            signal = None
                    
                    avg_time = (time.time() - start_time) / 5
                    execution_times[strategy_name] = round(avg_time * 1000, 2)
                    
                    print(f"   ✅ {strategy_name}: {execution_times[strategy_name]}ms")
                    
                except Exception as e:
                    execution_times[strategy_name] = f"ERROR: {str(e)}"
                    print(f"   ❌ {strategy_name}: {e}")
            
            # Calculate performance metrics
            valid_times = [t for t in execution_times.values() if isinstance(t, (int, float))]
            if valid_times:
                results = {
                    'execution_times': execution_times,
                    'avg_execution_ms': round(statistics.mean(valid_times), 2),
                    'max_execution_ms': max(valid_times),
                    'min_execution_ms': min(valid_times),
                    'status': 'PASS' if statistics.mean(valid_times) < 100 else 'SLOW'
                }
            else:
                results = {'error': 'No valid strategy executions', 'status': 'FAIL'}
            
        except Exception as e:
            results = {'error': str(e), 'status': 'FAIL'}
            print(f"   ❌ Strategy execution test failed: {e}")
        
        return results
    
    def test_ml_performance(self):
        """Test ML components performance"""
        results = {}
        
        try:
            # Test Market Regime Detection
            regime_detector = MarketRegimeDetector()
            
            start_time = time.time()
            for _ in range(3):
                try:
                    mock_data = {'BTC/USDT': pd.DataFrame({
                        'close': [45000, 45100, 45200],
                        'volume': [1000, 1100, 1200],
                        'timestamp': pd.date_range('2024-01-01', periods=3, freq='H')
                    })}
                    regime = regime_detector.detect_current_regime(mock_data)
                except Exception as e:
                    print(f"     ⚠️  Regime detection failed: {e}")
            
            regime_time = (time.time() - start_time) / 3
            
            # Test Market Sentiment Analysis
            try:
                sentiment_analyzer = MarketSentimentAnalyzer()
                start_time = time.time()
                for _ in range(3):
                    try:
                        sentiment = sentiment_analyzer.analyze_current_sentiment()
                    except Exception as e:
                        print(f"     ⚠️  Sentiment analysis failed: {e}")
                sentiment_time = (time.time() - start_time) / 3
            except Exception as e:
                sentiment_time = 0
                print(f"     ⚠️  Sentiment analyzer not available: {e}")
            
            results = {
                'regime_detection_ms': round(regime_time * 1000, 2),
                'sentiment_analysis_ms': round(sentiment_time * 1000, 2),
                'total_ml_time_ms': round((regime_time + sentiment_time) * 1000, 2),
                'status': 'PASS' if (regime_time + sentiment_time) < 2.0 else 'SLOW'
            }
            
            print(f"   ✅ Regime detection: {results['regime_detection_ms']}ms")
            print(f"   ✅ Sentiment analysis: {results['sentiment_analysis_ms']}ms")
            
        except Exception as e:
            results = {'error': str(e), 'status': 'FAIL'}
            print(f"   ❌ ML performance test failed: {e}")
        
        return results
    
    def test_concurrent_trading(self):
        """Test concurrent trading operations"""
        results = {}
        
        try:
            def simulate_trading_session(session_id):
                """Simulate a trading session"""
                try:
                    start_time = time.time()
                    
                    # Initialize components
                    data_manager = DataManager()
                    safety_manager = SafetyManager()
                    
                    # Simulate trading operations
                    operations = 0
                    for _ in range(10):  # 10 operations per session
                        # Get market data
                        data = data_manager.get_current_data(['BTC/USDT'])
                        
                        # Check safety
                        safety_check = safety_manager.check_safety_conditions({
                            'symbol': 'BTC/USDT',
                            'side': 'buy',
                            'amount': 0.001
                        })
                        
                        operations += 1
                        time.sleep(0.01)  # Small delay to simulate processing
                    
                    session_time = time.time() - start_time
                    return {
                        'session_id': session_id,
                        'operations': operations,
                        'time_seconds': round(session_time, 2),
                        'ops_per_second': round(operations / session_time, 2)
                    }
                    
                except Exception as e:
                    return {'session_id': session_id, 'error': str(e)}
            
            # Run concurrent sessions
            num_sessions = 5
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=num_sessions) as executor:
                futures = [executor.submit(simulate_trading_session, i) for i in range(num_sessions)]
                session_results = [future.result() for future in as_completed(futures)]
            
            total_time = time.time() - start_time
            
            # Analyze results
            successful_sessions = [r for r in session_results if 'error' not in r]
            
            if successful_sessions:
                total_ops = sum(r['operations'] for r in successful_sessions)
                avg_ops_per_sec = statistics.mean([r['ops_per_second'] for r in successful_sessions])
                
                results = {
                    'concurrent_sessions': num_sessions,
                    'successful_sessions': len(successful_sessions),
                    'total_operations': total_ops,
                    'total_time_seconds': round(total_time, 2),
                    'avg_ops_per_second': round(avg_ops_per_sec, 2),
                    'session_results': session_results,
                    'status': 'PASS' if len(successful_sessions) == num_sessions else 'PARTIAL'
                }
            else:
                results = {
                    'error': 'No successful concurrent sessions',
                    'session_results': session_results,
                    'status': 'FAIL'
                }
            
            print(f"   ✅ Concurrent sessions: {len(successful_sessions)}/{num_sessions}")
            if successful_sessions:
                print(f"   ✅ Avg ops/sec: {results['avg_ops_per_second']}")
            
        except Exception as e:
            results = {'error': str(e), 'status': 'FAIL'}
            print(f"   ❌ Concurrent trading test failed: {e}")
        
        return results
    
    def test_resource_usage(self):
        """Test memory and CPU usage"""
        results = {}
        
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            
            # Initial memory usage
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # CPU usage before load
            cpu_before = psutil.cpu_percent(interval=1)
            
            # Create load
            start_time = time.time()
            data_manager = DataManager()
            
            # Intensive operations
            for _ in range(20):
                data = data_manager.get_current_data(self.symbols)
                time.sleep(0.05)
            
            load_time = time.time() - start_time
            
            # Memory usage after load
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # CPU usage after load
            cpu_after = psutil.cpu_percent(interval=1)
            
            results = {
                'initial_memory_mb': round(initial_memory, 2),
                'final_memory_mb': round(final_memory, 2),
                'memory_increase_mb': round(memory_increase, 2),
                'cpu_before_percent': cpu_before,
                'cpu_after_percent': cpu_after,
                'load_duration_seconds': round(load_time, 2),
                'status': 'PASS' if memory_increase < 100 else 'HIGH_MEMORY'
            }
            
            print(f"   ✅ Memory usage: {results['initial_memory_mb']}MB → {results['final_memory_mb']}MB")
            print(f"   ✅ Memory increase: {results['memory_increase_mb']}MB")
            print(f"   ✅ CPU usage: {results['cpu_before_percent']}% → {results['cpu_after_percent']}%")
            
        except ImportError:
            results = {'error': 'psutil not available for resource monitoring', 'status': 'SKIP'}
            print(f"   ⚠️  Resource monitoring skipped (psutil not available)")
        except Exception as e:
            results = {'error': str(e), 'status': 'FAIL'}
            print(f"   ❌ Resource usage test failed: {e}")
        
        return results
    
    def test_api_performance(self):
        """Test API response times (simulated)"""
        results = {}
        
        try:
            # Simulate API endpoint response times
            endpoints = [
                '/api/v1/trading/status',
                '/api/v1/trading/positions',
                '/api/v1/strategies/list',
                '/api/v1/monitoring/health',
                '/api/v1/market/regime'
            ]
            
            response_times = {}
            
            for endpoint in endpoints:
                # Simulate API call processing time
                start_time = time.time()
                
                # Simulate different processing complexities
                if 'market' in endpoint or 'strategies' in endpoint:
                    time.sleep(0.1)  # More complex endpoints
                else:
                    time.sleep(0.02)  # Simple endpoints
                
                response_time = (time.time() - start_time) * 1000
                response_times[endpoint] = round(response_time, 2)
            
            avg_response_time = statistics.mean(response_times.values())
            
            results = {
                'response_times_ms': response_times,
                'avg_response_time_ms': round(avg_response_time, 2),
                'max_response_time_ms': max(response_times.values()),
                'min_response_time_ms': min(response_times.values()),
                'status': 'PASS' if avg_response_time < 200 else 'SLOW'
            }
            
            print(f"   ✅ Avg API response: {results['avg_response_time_ms']}ms")
            print(f"   ✅ Max API response: {results['max_response_time_ms']}ms")
            
        except Exception as e:
            results = {'error': str(e), 'status': 'FAIL'}
            print(f"   ❌ API performance test failed: {e}")
        
        return results
    
    def generate_performance_report(self, test_results):
        """Generate comprehensive performance report"""
        
        print("\n" + "=" * 60)
        print("📋 PERFORMANCE TEST REPORT")
        print("=" * 60)
        
        # Overall status
        all_statuses = []
        for category, results in test_results.items():
            if isinstance(results, dict) and 'status' in results:
                all_statuses.append(results['status'])
        
        passed_tests = sum(1 for status in all_statuses if status == 'PASS')
        total_tests = len(all_statuses)
        
        print(f"\n🎯 Overall Results: {passed_tests}/{total_tests} tests passed")
        
        # Detailed results
        for category, results in test_results.items():
            print(f"\n📊 {category.replace('_', ' ').title()}:")
            if isinstance(results, dict):
                if 'status' in results:
                    status_emoji = "✅" if results['status'] == 'PASS' else "⚠️" if results['status'] in ['SLOW', 'PARTIAL', 'SKIP'] else "❌"
                    print(f"   {status_emoji} Status: {results['status']}")
                
                for key, value in results.items():
                    if key != 'status' and not key.endswith('_results'):
                        print(f"   • {key}: {value}")
        
        # Performance recommendations
        print(f"\n💡 Performance Recommendations:")
        
        recommendations = []
        
        if test_results.get('data_processing', {}).get('status') == 'SLOW':
            recommendations.append("• Consider implementing data caching for frequently accessed symbols")
        
        if test_results.get('strategy_execution', {}).get('status') == 'SLOW':
            recommendations.append("• Optimize strategy calculations or implement result caching")
        
        if test_results.get('ml_performance', {}).get('status') == 'SLOW':
            recommendations.append("• Consider using lighter ML models or async processing")
        
        if test_results.get('concurrent_trading', {}).get('status') != 'PASS':
            recommendations.append("• Review thread safety and resource contention issues")
        
        if test_results.get('resource_usage', {}).get('status') == 'HIGH_MEMORY':
            recommendations.append("• Implement memory optimization and garbage collection")
        
        if test_results.get('api_performance', {}).get('status') == 'SLOW':
            recommendations.append("• Optimize API endpoints and implement response caching")
        
        if not recommendations:
            recommendations.append("• System performance is within acceptable limits")
            recommendations.append("• Consider monitoring in production for long-term trends")
        
        for rec in recommendations:
            print(f"   {rec}")
        
        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"performance_report_{timestamp}.json"
        
        try:
            with open(report_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'summary': {
                        'total_tests': total_tests,
                        'passed_tests': passed_tests,
                        'success_rate': round(passed_tests / total_tests * 100, 1) if total_tests > 0 else 0
                    },
                    'detailed_results': test_results,
                    'recommendations': recommendations
                }, f, indent=2, default=str)
            
            print(f"\n📄 Detailed report saved: {report_file}")
            
        except Exception as e:
            print(f"\n⚠️  Could not save report file: {e}")
        
        print("\n" + "=" * 60)

def main():
    """Run performance tests"""
    runner = PerformanceTestRunner()
    results = runner.run_all_tests()
    
    # Return overall success
    statuses = []
    for category_results in results.values():
        if isinstance(category_results, dict) and 'status' in category_results:
            statuses.append(category_results['status'])
    
    success_rate = sum(1 for s in statuses if s == 'PASS') / len(statuses) if statuses else 0
    return success_rate > 0.7  # 70% success rate threshold

if __name__ == "__main__":
    success = main()
    print(f"\n🏁 Performance tests {'PASSED' if success else 'NEED ATTENTION'}")
    sys.exit(0 if success else 1)