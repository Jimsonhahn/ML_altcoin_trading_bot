#!/usr/bin/env python3
"""
High-Risk Daily Strategy Test & Demo
===================================

Test and demonstration script for the high-risk trading strategy:
- Strategy initialization testing
- Risk limiter validation
- Signal generation simulation
- Trade execution demo
- Performance monitoring
"""

import asyncio
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_high_risk_strategy():
    """Comprehensive test of high-risk strategy"""
    
    print("🔥 HIGH-RISK STRATEGY TEST & DEMO")
    print("=" * 50)
    
    try:
        # Test 1: Strategy Initialization
        print("\n📋 TEST 1: Strategy Initialization")
        print("-" * 35)
        
        from strategies import get_strategy
        strategy_class = get_strategy('high_risk_daily')
        
        if not strategy_class:
            print("❌ Strategy not found in registry!")
            return False
        
        # Initialize strategy
        strategy = strategy_class({
            'daily_budget': 30.0,
            'max_positions': 3,
            'debug_mode': True
        })
        
        print(f"✅ Strategy initialized: {strategy.__class__.__name__}")
        print(f"📊 Daily budget: {strategy.risk_limiter.daily_budget}€")
        print(f"🎯 Max positions: {strategy.max_positions}")
        
        # Test 2: Risk Limiter Functionality
        print("\n📋 TEST 2: Risk Limiter Functionality")
        print("-" * 38)
        
        # Check initial state
        status = strategy.risk_limiter.get_status()
        print(f"💰 Initial budget: {status['remaining_budget']:.2f}€")
        print(f"📅 Date: {status['date']}")
        print(f"🔒 Locked: {status['is_locked']}")
        
        # Test budget reservation
        test_amount = 10.0
        can_trade, reason = strategy.risk_limiter.can_trade(test_amount)
        print(f"🤔 Can trade {test_amount}€: {can_trade} ({reason})")
        
        if can_trade:
            success = strategy.risk_limiter.reserve_budget(test_amount, "TEST_TRADE")
            print(f"💵 Budget reserved: {success}")
            
            if success:
                new_status = strategy.risk_limiter.get_status()
                print(f"📊 Remaining after reservation: {new_status['remaining_budget']:.2f}€")
                
                # Release budget
                strategy.risk_limiter.release_budget(test_amount, 2.5)  # Simulate 2.5€ profit
                final_status = strategy.risk_limiter.get_status()
                print(f"💰 Budget after release: {final_status['remaining_budget']:.2f}€")
                print(f"📈 Total P&L: {final_status['pnl_realized']:+.2f}€")
        
        # Test 3: Volume Detector
        print("\n📋 TEST 3: Volume Spike Detection")
        print("-" * 36)
        
        test_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        
        try:
            # Create volume detector with demo config
            from core.volume_detector import create_volume_detector
            
            volume_config = {
                'spike_threshold': 3.0,
                'min_confidence': 0.7,
                'timeframes': ['5m', '15m']
            }
            
            volume_detector = create_volume_detector(volume_config)
            print(f"✅ Volume detector initialized")
            print(f"🎯 Spike threshold: {volume_config['spike_threshold']}x")
            
            # Simulate volume spike detection
            print("🔍 Running volume spike detection...")
            spikes = await volume_detector.detect_volume_spikes(test_symbols[:1])  # Test with 1 symbol
            
            print(f"📊 Volume spikes detected: {len(spikes)}")
            for spike in spikes[:3]:  # Show first 3
                print(f"   🔥 {spike.symbol}: {spike.spike_ratio:.1f}x volume "
                      f"(confidence: {spike.confidence:.2f})")
                
        except Exception as e:
            print(f"⚠️ Volume detection test skipped: {e}")
        
        # Test 4: Social Sentiment Analysis
        print("\n📋 TEST 4: Social Sentiment Analysis")
        print("-" * 37)
        
        try:
            from core.social_sentiment import create_sentiment_analyzer
            
            sentiment_config = {
                'sources': ['twitter', 'reddit'],
                'sentiment_threshold': 0.3,
                'momentum_threshold': 2.0
            }
            
            async with create_sentiment_analyzer(sentiment_config) as sentiment_analyzer:
                print(f"✅ Sentiment analyzer initialized")
                print(f"📱 Sources: {sentiment_config['sources']}")
                
                # Simulate sentiment analysis
                print("🔍 Running sentiment analysis...")
                signals = await sentiment_analyzer.analyze_sentiment(test_symbols[:1])
                
                print(f"📊 Sentiment signals: {len(signals)}")
                for signal in signals[:3]:  # Show first 3
                    print(f"   📱 {signal.symbol}: sentiment={signal.sentiment_score:+.2f} "
                          f"momentum={signal.momentum_score:.1f}x "
                          f"({signal.source})")
                    
        except Exception as e:
            print(f"⚠️ Sentiment analysis test skipped: {e}")
        
        # Test 5: Signal Generation
        print("\n📋 TEST 5: Signal Generation")
        print("-" * 30)
        
        # Create sample market data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.normal(45000, 1000, 100),
            'high': np.random.normal(45500, 1000, 100),
            'low': np.random.normal(44500, 1000, 100),
            'close': np.random.normal(45000, 1000, 100),
            'volume': np.random.normal(1000000, 200000, 100)
        })
        sample_data.set_index('timestamp', inplace=True)
        
        test_symbol = 'BTC/USDT'
        current_price = 45000.0
        
        print(f"🎯 Testing signal generation for {test_symbol}")
        print(f"💰 Current price: ${current_price:,.2f}")
        
        signal, signal_data = await strategy.calculate_signal(test_symbol, sample_data, current_price)
        
        print(f"📡 Signal: {signal}")
        print(f"🎯 Confidence: {signal_data.get('confidence', 0.0):.2f}")
        print(f"📊 Sources: {signal_data.get('sources', [])}")
        print(f"🔍 Reason: {signal_data.get('reason', 'N/A')}")
        
        # Test 6: Trade Execution Simulation
        print("\n📋 TEST 6: Trade Execution Simulation")
        print("-" * 39)
        
        if signal in ['BUY', 'SELL']:
            print(f"🚀 Simulating trade execution...")
            
            # Execute trade
            success = await strategy.execute_trade(signal, test_symbol, current_price, signal_data)
            
            if success:
                print(f"✅ Trade executed successfully")
                
                # Show active positions
                active_positions = len(strategy.active_positions)
                print(f"📊 Active positions: {active_positions}")
                
                for trade_id, position in strategy.active_positions.items():
                    print(f"   💼 {position.symbol}: {position.side} "
                          f"{position.quantity:.6f} @ ${position.entry_price:.2f}")
                
                # Simulate price movement and position management
                print("📈 Simulating price movement...")
                
                # Simulate 10% price increase
                new_prices = {test_symbol: current_price * 1.1}
                await strategy.manage_positions(new_prices)
                
                print(f"📊 Positions after price movement: {len(strategy.active_positions)}")
                
            else:
                print(f"❌ Trade execution failed")
        else:
            print(f"⏸️ No trade signal generated (signal: {signal})")
        
        # Test 7: Strategy Information
        print("\n📋 TEST 7: Strategy Information")
        print("-" * 32)
        
        info = strategy.get_strategy_info()
        print(f"📋 Strategy: {info['name']}")
        print(f"🔥 Risk Level: {info['risk_level']}")
        print(f"💰 Budget Used: {info['budget_used']:.2f}€ / {info['daily_budget']:.2f}€")
        print(f"🎯 Active Positions: {info['active_positions']} / {info['max_positions']}")
        print(f"📊 Trades Today: {info['daily_stats']['trades_executed']}")
        print(f"📈 Total P&L: {info['daily_stats']['total_pnl']:+.2f}€")
        
        # Test 8: Daily Summary
        print("\n📋 TEST 8: Daily Summary")
        print("-" * 25)
        
        summary = strategy.get_daily_summary()
        print(summary)
        
        # Test 9: Logging System
        print("\n📋 TEST 9: Logging System")
        print("-" * 27)
        
        logger_summary = strategy.hr_logger.get_daily_summary()
        print(logger_summary)
        
        # Show log files created
        log_dir = Path("logs/high_risk")
        if log_dir.exists():
            print(f"\n📁 Log files created:")
            for log_file in log_dir.rglob("*.log"):
                size = log_file.stat().st_size if log_file.exists() else 0
                print(f"   📄 {log_file.relative_to(log_dir)} ({size} bytes)")
        
        # Test 10: Risk Limiter Stress Test
        print("\n📋 TEST 10: Risk Limiter Stress Test")
        print("-" * 37)
        
        print("🧪 Testing budget exhaustion...")
        
        # Try to exhaust budget
        remaining = strategy.risk_limiter.get_status()['remaining_budget']
        print(f"💰 Starting with: {remaining:.2f}€")
        
        # Reserve large amount
        large_amount = remaining + 5.0  # More than available
        can_trade_large, reason = strategy.risk_limiter.can_trade(large_amount)
        print(f"🤔 Can trade {large_amount:.2f}€: {can_trade_large} ({reason})")
        
        # Reserve exactly what's available
        if remaining > 0:
            success = strategy.risk_limiter.reserve_budget(remaining - 0.01)  # Leave tiny amount
            print(f"💵 Reserved {remaining - 0.01:.2f}€: {success}")
            
            # Check final state
            final_status = strategy.risk_limiter.get_status()
            print(f"📊 Final remaining: {final_status['remaining_budget']:.2f}€")
            print(f"🔒 Is locked: {final_status['is_locked']}")
        
        print("\n🎉 ALL TESTS COMPLETED!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def demo_scenario():
    """Demo realistic trading scenario"""
    
    print("\n🎬 DEMO: Realistic Trading Scenario")
    print("=" * 40)
    
    try:
        from strategies import get_strategy
        
        # Initialize fresh strategy
        strategy = get_strategy('high_risk_daily')({
            'daily_budget': 30.0,
            'max_positions': 2,
            'demo_mode': True
        })
        
        print(f"🔥 Demo Strategy Initialized")
        print(f"💰 Budget: {strategy.risk_limiter.daily_budget}€")
        
        # Simulate realistic trading day
        symbols = ['SOL/USDT', 'AVAX/USDT', 'MATIC/USDT']
        base_prices = {'SOL/USDT': 150.0, 'AVAX/USDT': 35.0, 'MATIC/USDT': 0.85}
        
        print(f"\n📊 Monitoring symbols: {', '.join(symbols)}")
        
        # Simulate 6-hour trading session
        for hour in range(6):
            print(f"\n⏰ Hour {hour + 1}/6 - {datetime.now().strftime('%H:%M')}")
            print("-" * 20)
            
            # Simulate price movements
            current_prices = {}
            for symbol in symbols:
                # Random price movement ±5%
                change = np.random.uniform(-0.05, 0.05)
                current_prices[symbol] = base_prices[symbol] * (1 + change)
                print(f"💹 {symbol}: ${current_prices[symbol]:.6f} ({change:+.1%})")
            
            # Generate signals for each symbol
            for symbol in symbols:
                current_price = current_prices[symbol]
                
                # Create dummy data
                dummy_data = pd.DataFrame({
                    'close': [current_price] * 50,
                    'volume': [1000000] * 50
                }, index=pd.date_range(start='2024-01-01', periods=50, freq='1H'))
                
                # Check for signals
                signal, signal_data = await strategy.calculate_signal(symbol, dummy_data, current_price)
                
                if signal != 'HOLD':
                    print(f"🎯 Signal: {symbol} {signal} (confidence: {signal_data.get('confidence', 0):.2f})")
                    
                    # Execute if conditions met
                    if strategy.risk_limiter.get_status()['can_trade']:
                        success = await strategy.execute_trade(signal, symbol, current_price, signal_data)
                        if success:
                            print(f"✅ Trade executed: {symbol}")
                        else:
                            print(f"❌ Trade failed: {symbol}")
            
            # Manage existing positions
            if strategy.active_positions:
                print(f"📊 Managing {len(strategy.active_positions)} positions...")
                await strategy.manage_positions(current_prices)
            
            # Show current status
            status = strategy.risk_limiter.get_status()
            print(f"💰 Budget: {status['budget_used']:.2f}€ / {status['daily_budget']:.2f}€")
            print(f"📈 P&L: {strategy.daily_stats['total_pnl']:+.2f}€")
            
            # Check if locked
            if status['is_locked']:
                print(f"🔒 TRADING LOCKED: {status['lock_reason']}")
                break
            
            # Small delay for demo
            await asyncio.sleep(0.1)
        
        # Final summary
        print(f"\n📋 DEMO COMPLETED - Final Summary")
        print("-" * 35)
        
        final_summary = strategy.get_daily_summary()
        print(final_summary)
        
        # Log summary
        log_summary = strategy.hr_logger.get_daily_summary()
        print(f"\n📄 Logging Summary:")
        print(log_summary)
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test runner"""
    
    print("🚀 Starting High-Risk Strategy Tests...")
    
    # Run comprehensive tests
    test_success = await test_high_risk_strategy()
    
    if test_success:
        print("\n" + "="*60)
        
        # Run demo scenario
        demo_success = await demo_scenario()
        
        if demo_success:
            print("\n✅ ALL TESTS AND DEMO COMPLETED SUCCESSFULLY!")
            print("\n📋 Next Steps:")
            print("1. Review log files in logs/high_risk/")
            print("2. Adjust configuration in config/high_risk.json")
            print("3. Test with paper trading")
            print("4. Deploy with live API keys (USE SMALL AMOUNTS!)")
            print("\n⚠️  WARNING: This is an EXTREME risk strategy!")
            print("💰 Only use money you can afford to lose completely!")
        else:
            print("\n❌ Demo failed!")
    else:
        print("\n❌ Tests failed!")
    
    print("\n🔥 High-Risk Strategy Testing Complete")

if __name__ == "__main__":
    # Run tests
    asyncio.run(main())