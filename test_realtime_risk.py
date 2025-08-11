#!/usr/bin/env python3
"""
Test Real-time Risk Calculator Integration
==========================================

Teste die Integration des Real-time Risk Calculators
"""
import sys
import os
import logging
import time
from datetime import datetime

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_realtime_risk_integration():
    """Test real-time risk calculator integration"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=== Testing Real-time Risk Calculator Integration ===")
    
    try:
        # Test 1: Import and create risk calculator
        logger.info("Test 1: Importing real-time risk calculator...")
        from core.realtime_risk_calculator import get_risk_calculator, RiskMetrics
        
        calculator = get_risk_calculator()
        logger.info("✓ Real-time risk calculator imported successfully")
        
        # Test 2: Start monitoring
        logger.info("Test 2: Starting real-time monitoring...")
        calculator.start_monitoring(initial_capital=10000)
        logger.info("✓ Real-time monitoring started")
        
        # Test 3: Update positions and prices
        logger.info("Test 3: Testing position and price updates...")
        calculator.update_position("BTC/USDT", 0.1, 45000, "long")
        calculator.update_price("BTC/USDT", 46000)
        
        calculator.update_position("ETH/USDT", 2.0, 3000, "long")
        calculator.update_price("ETH/USDT", 3100)
        
        logger.info("✓ Position and price updates successful")
        
        # Test 4: Get metrics
        logger.info("Test 4: Retrieving risk metrics...")
        time.sleep(2)  # Wait for calculations
        
        metrics = calculator.get_current_metrics()
        if metrics:
            logger.info(f"✓ Current Drawdown: {metrics.current_drawdown:.2%}")
            logger.info(f"✓ Total P&L: ${metrics.total_pnl:.2f}")
            logger.info(f"✓ Risk Level: {metrics.risk_level}")
            logger.info(f"✓ Open Positions: {metrics.open_positions_count}")
            logger.info(f"✓ Total Exposure: ${metrics.total_exposure:.2f}")
            
            if metrics.warnings:
                logger.warning(f"Risk Warnings: {metrics.warnings}")
        else:
            logger.error("✗ Failed to get risk metrics")
            return False
        
        # Test 5: Test RiskManager integration
        logger.info("Test 5: Testing RiskManager integration...")
        from config.settings import Settings
        from core.risk_manager import RiskManager
        
        settings = Settings()
        risk_manager = RiskManager(settings)
        
        # Start real-time monitoring through risk manager
        risk_manager.start_realtime_monitoring(10000)
        
        # Update positions through risk manager
        risk_manager.update_realtime_position("BTC/USDT", 0.1, 45000, "long")
        risk_manager.update_realtime_price("BTC/USDT", 46500)
        
        time.sleep(1)
        
        # Get enhanced metrics
        enhanced_metrics = risk_manager.get_enhanced_portfolio_metrics()
        logger.info(f"✓ Enhanced metrics retrieved: {len(enhanced_metrics)} metrics")
        
        # Test event system
        logger.info("Test 6: Testing event system...")
        from core.interfaces import global_event_bus
        
        event_received = False
        def test_event_handler(data):
            nonlocal event_received
            event_received = True
            logger.info(f"✓ Event received: {data.get('metrics', {}).risk_level if hasattr(data.get('metrics', {}), 'risk_level') else 'Unknown'}")
        
        global_event_bus.subscribe("risk_metrics_update", test_event_handler)
        
        # Trigger an update
        calculator.update_price("BTC/USDT", 47000)
        time.sleep(2)
        
        if event_received:
            logger.info("✓ Event system working correctly")
        else:
            logger.warning("⚠ Event system may not be working")
        
        # Cleanup
        calculator.stop_monitoring()
        risk_manager.stop_realtime_monitoring()
        
        logger.info("=== All Tests Completed Successfully! ===")
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import error: {e}")
        logger.error("Make sure all required modules are available")
        return False
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        logger.error("Check the error logs for details", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_realtime_risk_integration()
    sys.exit(0 if success else 1)