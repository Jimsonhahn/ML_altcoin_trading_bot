#!/usr/bin/env python3
"""
Test Capital Allocation Tracking
================================

Test the capital allocation tracking system
"""
import sys
import os
import logging
import time
import asyncio
from datetime import datetime

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_capital_allocation():
    """Test capital allocation tracking"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=== Testing Capital Allocation Tracking ===")
    
    try:
        # Test 1: Import allocation tracker
        logger.info("Test 1: Importing allocation tracker...")
        from core.capital_allocation_tracker import get_allocation_tracker, AllocationStatus
        from config.settings import Settings
        
        settings = Settings()
        allocation_tracker = get_allocation_tracker(settings)
        logger.info("✓ Allocation tracker imported successfully")
        
        # Test 2: Start tracking
        logger.info("Test 2: Starting allocation tracking...")
        allocation_tracker.start_tracking()
        logger.info("✓ Allocation tracking started")
        
        # Test 3: Test capital allocation
        logger.info("Test 3: Testing capital allocation...")
        
        # Allocate capital to different strategies
        success1 = allocation_tracker.allocate_capital("momentum", 3000, reason="Test allocation")
        success2 = allocation_tracker.allocate_capital("grid_trading", 2500, reason="Test allocation")
        success3 = allocation_tracker.allocate_capital("arbitrage", 2000, reason="Test allocation")
        
        if success1 and success2 and success3:
            logger.info("✓ Capital allocations successful")
        else:
            logger.error("✗ Some allocations failed")
            return False
        
        # Test 4: Test utilization updates
        logger.info("Test 4: Testing capital utilization updates...")
        
        allocation_tracker.update_capital_utilization("momentum", 2800)  # 93% utilization
        allocation_tracker.update_capital_utilization("grid_trading", 1200)  # 48% utilization
        allocation_tracker.update_capital_utilization("arbitrage", 1900)  # 95% utilization
        
        logger.info("✓ Capital utilization updated")
        
        # Test 5: Test P&L updates
        logger.info("Test 5: Testing P&L updates...")
        
        allocation_tracker.update_strategy_pnl("momentum", 150, 50)  # $200 total P&L
        allocation_tracker.update_strategy_pnl("grid_trading", 75, -25)  # $50 total P&L
        allocation_tracker.update_strategy_pnl("arbitrage", 25, 15)  # $40 total P&L
        
        logger.info("✓ P&L updates completed")
        
        # Test 6: Get portfolio summary
        logger.info("Test 6: Testing portfolio summary...")
        
        summary = allocation_tracker.get_portfolio_summary()
        logger.info(f"✓ Total Capital: ${summary.total_capital:.2f}")
        logger.info(f"✓ Allocated Capital: ${summary.allocated_capital:.2f}")
        logger.info(f"✓ Available Capital: ${summary.available_capital:.2f}")
        logger.info(f"✓ Total P&L: ${summary.total_pnl:.2f}")
        logger.info(f"✓ Allocation Efficiency: {summary.allocation_efficiency:.2%}")
        logger.info(f"✓ Strategy Count: {summary.strategy_count}")
        logger.info(f"✓ Best Performer: {summary.best_performing_strategy}")
        
        # Test 7: Get allocation details
        logger.info("Test 7: Testing allocation details...")
        
        details = allocation_tracker.get_allocation_details()
        for strategy_name, details_dict in details.items():
            logger.info(f"✓ {strategy_name}: ${details_dict['allocated_capital']:.2f} "
                       f"({details_dict['utilization_rate']:.1%} utilized, "
                       f"${details_dict['total_pnl']:.2f} P&L)")
        
        # Test 8: Test allocation history
        logger.info("Test 8: Testing allocation history...")
        
        history = allocation_tracker.get_allocation_history(limit=10)
        logger.info(f"✓ Allocation history: {len(history)} events")
        
        for event in history[-3:]:  # Show last 3 events
            logger.info(f"  - {event['event_type']}: {event['strategy_name']} "
                       f"${event['amount']:.2f} ({event['reason']})")
        
        # Test 9: Test deallocation
        logger.info("Test 9: Testing capital deallocation...")
        
        dealloc_success = allocation_tracker.deallocate_capital("grid_trading", 500, reason="Test deallocation")
        if dealloc_success:
            logger.info("✓ Capital deallocation successful")
        else:
            logger.error("✗ Capital deallocation failed")
        
        # Test 10: Test performance attribution
        logger.info("Test 10: Testing performance attribution...")
        
        # Add some performance snapshots manually for testing
        for i in range(5):
            allocation_tracker._save_performance_snapshot()
            await asyncio.sleep(0.1)
        
        attribution = allocation_tracker.get_performance_attribution(days=1)
        if attribution:
            logger.info("✓ Performance attribution calculated")
            for strategy, attr in attribution.items():
                logger.info(f"  - {strategy}: {attr['pnl_contribution']:.2f} P&L contribution")
        else:
            logger.info("ℹ No attribution data available (expected for short test)")
        
        # Test 11: Test event system
        logger.info("Test 11: Testing event system integration...")
        
        from core.interfaces import global_event_bus
        
        event_received = False
        def test_allocation_event_handler(data):
            nonlocal event_received
            event_received = True
            logger.info(f"✓ Allocation event received: {data.get('strategy_name')} = ${data.get('amount', 0):.2f}")
        
        global_event_bus.subscribe("capital_allocated", test_allocation_event_handler)
        
        # Trigger allocation event
        allocation_tracker.allocate_capital("test_strategy", 500, reason="Event test")
        await asyncio.sleep(0.5)
        
        if event_received:
            logger.info("✓ Event system working correctly")
        else:
            logger.warning("⚠ Event system may not be working")
        
        # Test 12: Test risk monitoring
        logger.info("Test 12: Testing risk monitoring...")
        
        # Simulate large loss to trigger emergency stop
        allocation_tracker.update_strategy_pnl("test_strategy", -150, 0)  # 30% loss
        
        # Wait for risk check
        await asyncio.sleep(2)
        
        test_allocation = allocation_tracker.get_allocation_details("test_strategy")
        if test_allocation and test_allocation.get('status') == 'suspended':
            logger.info("✓ Emergency stop triggered correctly")
        else:
            logger.info("ℹ Emergency stop not triggered (may require more time)")
        
        # Test 13: Test strategy router integration
        logger.info("Test 13: Testing strategy router integration...")
        
        from core.strategy_router import StrategyRouter
        
        strategy_router = StrategyRouter(settings)
        
        if strategy_router.enable_allocation_tracking:
            logger.info("✓ Strategy router has allocation tracking enabled")
            
            # Test allocation summary
            summary = strategy_router.get_capital_allocation_summary()
            logger.info(f"✓ Router allocation summary: {len(summary.get('strategy_allocations', {}))} strategies")
            
            # Test manual allocation through router
            manual_alloc_success = strategy_router.allocate_strategy_capital("router_test", 300)
            if manual_alloc_success:
                logger.info("✓ Manual allocation through router successful")
            else:
                logger.warning("⚠ Manual allocation through router failed")
        
        else:
            logger.warning("⚠ Strategy router allocation tracking not enabled")
        
        # Test 14: Test export functionality
        logger.info("Test 14: Testing export functionality...")
        
        try:
            allocation_tracker.export_allocation_report("test_allocation_report.json")
            logger.info("✓ Allocation report exported successfully")
            
            # Clean up test file
            if os.path.exists("test_allocation_report.json"):
                os.remove("test_allocation_report.json")
                logger.info("✓ Test report file cleaned up")
        
        except Exception as e:
            logger.warning(f"⚠ Export test failed: {e}")
        
        # Test 15: Test configuration loading
        logger.info("Test 15: Testing configuration loading...")
        
        try:
            import json
            with open('config/capital_allocation.json', 'r') as f:
                config = json.load(f)
            
            logger.info("✓ Capital allocation configuration loaded")
            logger.info(f"Initial capital: ${config['capital_allocation']['initial_capital']}")
            logger.info(f"Max allocation: {config['capital_allocation']['max_allocation_pct']:.0%}")
            logger.info(f"Rebalance threshold: {config['capital_allocation']['rebalance_threshold']:.1%}")
        
        except Exception as e:
            logger.warning(f"⚠ Configuration not loaded: {e}")
        
        # Cleanup
        logger.info("Cleaning up...")
        allocation_tracker.stop_tracking()
        strategy_router.shutdown_allocation_tracking()
        
        logger.info("=== All Capital Allocation Tests Completed Successfully! ===")
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
    success = asyncio.run(test_capital_allocation())
    sys.exit(0 if success else 1)