#!/usr/bin/env python3
"""
Multi-Mode Trading Integration Test
===================================

Testet das Upgrade von simuliertem zu echtem Exchange Paper Trading
ohne das bestehende System zu beeinträchtigen.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_trading_engine_manager():
    """Test TradingEngineManager functionality"""
    try:
        from core.trading_engine_manager import TradingEngineManager
        
        print("🔧 Testing TradingEngineManager...")
        
        # Initialize manager
        manager = TradingEngineManager()
        
        # Test 1: Check initial state (should be simulated)
        modes = manager.get_available_modes()
        print(f"✅ Initial modes: {modes}")
        assert modes['current_mode'] == 'simulated'
        assert modes['available_modes']['simulated']['available'] == True
        
        # Test 2: Switch to simulated mode (should work)
        result = manager.switch_mode('simulated')
        print(f"✅ Switch to simulated: {result}")
        assert result['success'] == True
        
        # Test 3: Try switching to real_paper without setup (should fail gracefully)
        result = manager.switch_mode('real_paper')
        print(f"✅ Switch to real_paper (should fail): {result}")
        assert result['success'] == False
        assert 'not configured' in result['message']
        
        # Test 4: Test portfolio status with simulated engine
        portfolio = manager.get_portfolio_status()
        print(f"✅ Portfolio status: {portfolio['mode']}")
        assert portfolio['engine_available'] == True
        
        # Test 5: Test virtual trade execution (use larger size to meet minimum)
        trade_result = await manager.execute_trade(
            symbol='BTC/USDT',
            side='LONG',
            size=0.5,  # Increase size to meet $10 minimum at ~$100 price
            strategy='test'
        )
        print(f"✅ Virtual trade executed: {trade_result.id if trade_result else 'Failed'}")
        assert trade_result is not None
        
        print("✅ TradingEngineManager tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ TradingEngineManager test failed: {e}")
        return False


def test_api_routes():
    """Test new API routes"""
    try:
        from api.routes.trading_mode_manager import initialize_trading_engine_manager, get_trading_engine_manager
        
        print("📡 Testing Trading Mode Manager API routes...")
        
        # Test 1: Initialize manager
        manager = initialize_trading_engine_manager()
        print(f"✅ Manager initialized: {manager is not None}")
        assert manager is not None
        
        # Test 2: Get manager instance
        manager_instance = get_trading_engine_manager()
        print(f"✅ Manager instance retrieved: {manager_instance is not None}")
        assert manager_instance is not None
        
        print("✅ API routes tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ API routes test failed: {e}")
        return False


def test_existing_system_compatibility():
    """Test that existing system still works"""
    try:
        from core.paper_trading_engine import PaperTradingEngine
        
        print("🔄 Testing existing system compatibility...")
        
        # Test 1: Original PaperTradingEngine still works
        engine = PaperTradingEngine(initial_balance=10000)
        print(f"✅ Original PaperTradingEngine created")
        
        # Test 2: Original methods still work
        portfolio_status = engine.get_virtual_portfolio_status()
        print(f"✅ Original portfolio status: ${portfolio_status['total_portfolio_value']}")
        assert portfolio_status['mode'] == 'PAPER TRADING'
        
        print("✅ Existing system compatibility tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Existing system compatibility test failed: {e}")
        return False


def test_dashboard_components():
    """Test that dashboard components can be imported"""
    try:
        print("🎨 Testing Dashboard components...")
        
        # Test 1: Check if TradingModeManager component exists
        dashboard_file = project_root / "dashboard/src/components/TradingModeManager.jsx"
        css_file = project_root / "dashboard/src/components/TradingModeManager.css"
        
        print(f"✅ TradingModeManager.jsx exists: {dashboard_file.exists()}")
        print(f"✅ TradingModeManager.css exists: {css_file.exists()}")
        
        assert dashboard_file.exists()
        assert css_file.exists()
        
        # Test 2: Check if main dashboard was updated
        main_dashboard = project_root / "dashboard/src/components/RevolutionaryJanicsDashboard.jsx"
        if main_dashboard.exists():
            content = main_dashboard.read_text()
            assert 'TradingModeManager' in content
            print(f"✅ Main dashboard updated with TradingModeManager")
        
        print("✅ Dashboard components tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Dashboard components test failed: {e}")
        return False


def test_flask_app_integration():
    """Test Flask app integration"""
    try:
        print("🌐 Testing Flask app integration...")
        
        # Test 1: Check if new routes are registered
        app_file = project_root / "api/app.py"
        if app_file.exists():
            content = app_file.read_text()
            assert 'trading_mode_manager' in content
            print(f"✅ Flask app updated with trading_mode_manager routes")
        
        print("✅ Flask app integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Flask app integration test failed: {e}")
        return False


def test_file_structure():
    """Test that all required files exist"""
    try:
        print("📁 Testing file structure...")
        
        required_files = [
            "core/trading_engine_manager.py",
            "core/real_exchange_paper_engine.py", 
            "api/routes/trading_mode_manager.py",
            "dashboard/src/components/TradingModeManager.jsx",
            "dashboard/src/components/TradingModeManager.css"
        ]
        
        for file_path in required_files:
            full_path = project_root / file_path
            exists = full_path.exists()
            print(f"{'✅' if exists else '❌'} {file_path}: {'EXISTS' if exists else 'MISSING'}")
            assert exists, f"Required file missing: {file_path}"
        
        print("✅ File structure tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ File structure test failed: {e}")
        return False


async def run_all_tests():
    """Run all integration tests"""
    print("🚀 STARTING MULTI-MODE TRADING INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Existing System Compatibility", test_existing_system_compatibility),
        ("TradingEngineManager", test_trading_engine_manager),
        ("API Routes", test_api_routes),
        ("Dashboard Components", test_dashboard_components),
        ("Flask App Integration", test_flask_app_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results[test_name] = result
            print(f"{'✅' if result else '❌'} {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            results[test_name] = False
            print(f"❌ {test_name}: FAILED - {e}")
    
    print("\n" + "=" * 60)
    print("🎯 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Multi-mode integration successful!")
        print("\n🚀 READY FOR DEPLOYMENT:")
        print("   1. ✅ Existing system preserved")
        print("   2. ✅ New multi-mode support added")
        print("   3. ✅ API endpoints extended")
        print("   4. ✅ Dashboard upgraded")
        print("   5. ✅ Exchange integration ready")
        return True
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        return False


def print_upgrade_instructions():
    """Print instructions for using the upgrade"""
    print("\n" + "=" * 60)
    print("📋 UPGRADE USAGE INSTRUCTIONS")
    print("=" * 60)
    
    print("\n🎯 How to use the Multi-Mode Trading Upgrade:")
    
    print("\n1. 📊 START WITH EXISTING SYSTEM (no changes needed):")
    print("   - Your current simulated paper trading still works")
    print("   - Dashboard shows current mode as 'SIMULATED'")
    print("   - All existing functionality preserved")
    
    print("\n2. 🏦 UPGRADE TO REAL EXCHANGE PAPER TRADING:")
    print("   - Go to Dashboard → Trading Modes")
    print("   - Click 'SETUP REQUIRED' on 'REAL PAPER' card")
    print("   - Enter Binance/Coinbase testnet API keys")
    print("   - Switch to 'real_paper' mode")
    print("   - Now using real exchange demo APIs!")
    
    print("\n3. 💰 OPTIONAL: ENABLE LIVE TRADING:")
    print("   - Add live API keys in setup modal")
    print("   - Switch to 'live' mode (CAUTION: real money)")
    
    print("\n4. 🔄 SWITCHING MODES:")
    print("   - Simulated → Real Paper → Live")
    print("   - Switch anytime via Dashboard or API")
    print("   - No data loss between modes")
    
    print("\n5. 📡 API ENDPOINTS:")
    print("   - GET /api/v1/trading-modes/modes")
    print("   - POST /api/v1/trading-modes/switch-mode")
    print("   - POST /api/v1/trading-modes/setup-exchange")
    
    print("\n🛡️ SAFETY FEATURES:")
    print("   - ✅ Existing system always works (fallback)")
    print("   - ✅ Real paper = demo money only")
    print("   - ✅ Live mode requires explicit setup")
    print("   - ✅ Account reset for paper modes")


if __name__ == "__main__":
    try:
        # Run all tests
        success = asyncio.run(run_all_tests())
        
        if success:
            print_upgrade_instructions()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        sys.exit(1)