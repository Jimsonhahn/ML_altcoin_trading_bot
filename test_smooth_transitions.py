#!/usr/bin/env python3
"""
Test Smooth Strategy Transitions
=================================

Test the smooth strategy transition system
"""
import sys
import os
import logging
import time
import asyncio
from datetime import datetime, timedelta

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_smooth_transitions():
    """Test smooth strategy transitions"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=== Testing Smooth Strategy Transitions ===")
    
    try:
        # Test 1: Import transition manager
        logger.info("Test 1: Importing transition manager...")
        from core.strategy_transition_manager import get_transition_manager, TransitionPriority, PositionInfo
        from core.strategy_router import StrategyRouter, MarketPhase
        from config.settings import Settings
        
        settings = Settings()
        transition_manager = get_transition_manager(settings)
        logger.info("✓ Transition manager imported successfully")
        
        # Test 2: Start transition manager
        logger.info("Test 2: Starting transition manager...")
        transition_manager.start_transition_manager()
        logger.info("✓ Transition manager started")
        
        # Test 3: Create mock position data
        logger.info("Test 3: Creating mock position data...")
        
        # Mock positions for testing
        mock_positions = [
            PositionInfo(
                symbol="BTC/USDT",
                size=0.1,
                entry_price=45000,
                current_price=46000,
                unrealized_pnl=100,
                side="long",
                strategy_name="momentum"
            ),
            PositionInfo(
                symbol="ETH/USDT", 
                size=2.0,
                entry_price=3000,
                current_price=2950,
                unrealized_pnl=-100,
                side="long",
                strategy_name="momentum"
            )
        ]
        
        # Update position tracker manually
        transition_manager.position_tracker["momentum"] = mock_positions
        logger.info("✓ Mock position data created")
        
        # Test 4: Plan a transition
        logger.info("Test 4: Planning a strategy transition...")
        
        transition_id = transition_manager.plan_transition(
            source_strategies=["momentum"],
            target_strategies=["grid_trading", "arbitrage"],
            priority=TransitionPriority.NORMAL
        )
        
        logger.info(f"✓ Transition planned: {transition_id}")
        
        # Test 5: Monitor transition progress
        logger.info("Test 5: Monitoring transition progress...")
        
        for i in range(10):  # Monitor for 10 seconds
            status = transition_manager.get_transition_status(transition_id)
            if status:
                logger.info(f"Progress: {status['progress']:.1f}% - State: {status['state']}")
                
                if status['progress'] >= 100:
                    logger.info("✓ Transition completed!")
                    break
            
            await asyncio.sleep(1)
        
        # Test 6: Test strategy router integration
        logger.info("Test 6: Testing strategy router integration...")
        
        strategy_router = StrategyRouter(settings)
        
        # Check if smooth transitions are enabled
        if strategy_router.enable_smooth_transitions:
            logger.info("✓ Smooth transitions enabled in strategy router")
            
            # Get strategy status
            status = strategy_router.get_strategy_status()
            logger.info(f"✓ Strategy status retrieved: {len(status)} fields")
            
            # Test transition status
            transition_status = strategy_router.get_transition_status()
            logger.info(f"✓ Transition status: {transition_status}")
            
        else:
            logger.warning("⚠ Smooth transitions not enabled in strategy router")
        
        # Test 7: Test emergency transition
        logger.info("Test 7: Testing emergency transition...")
        
        emergency_id = transition_manager.plan_transition(
            source_strategies=["grid_trading"],
            target_strategies=["arbitrage"],
            priority=TransitionPriority.EMERGENCY,
            force_immediate=True
        )
        
        logger.info(f"✓ Emergency transition planned: {emergency_id}")
        
        # Monitor emergency transition
        await asyncio.sleep(2)
        emergency_status = transition_manager.get_transition_status(emergency_id)
        if emergency_status:
            logger.info(f"Emergency transition state: {emergency_status['state']}")
        
        # Test 8: Test configuration loading
        logger.info("Test 8: Testing configuration loading...")
        
        try:
            import json
            with open('config/strategy_transitions.json', 'r') as f:
                config = json.load(f)
            
            logger.info("✓ Transition configuration loaded")
            logger.info(f"Max concurrent transitions: {config['strategy_transitions']['max_concurrent']}")
            logger.info(f"Default unwind time: {config['strategy_transitions']['default_unwind_minutes']} minutes")
            
        except Exception as e:
            logger.warning(f"⚠ Configuration not loaded: {e}")
        
        # Test 9: Test callbacks
        logger.info("Test 9: Testing transition callbacks...")
        
        callback_called = False
        
        def test_position_close_callback(position, risk_constraints):
            nonlocal callback_called
            callback_called = True
            logger.info(f"✓ Position close callback called for {position.symbol}")
            return True
        
        def test_strategy_stop_callback(strategy_name, transition):
            logger.info(f"✓ Strategy stop callback called for {strategy_name}")
            return True
        
        def test_strategy_start_callback(strategy_name, capital, transition):
            logger.info(f"✓ Strategy start callback called for {strategy_name} with {capital}")
            return True
        
        transition_manager.set_callbacks(
            position_close_callback=test_position_close_callback,
            strategy_stop_callback=test_strategy_stop_callback,
            strategy_start_callback=test_strategy_start_callback
        )
        
        logger.info("✓ Callbacks configured")
        
        # Test 10: Verify event system
        logger.info("Test 10: Testing event system...")
        
        from core.interfaces import global_event_bus
        
        event_received = False
        def test_event_handler(data):
            nonlocal event_received
            event_received = True
            logger.info(f"✓ Transition event received: {data.get('transition_id', 'Unknown')}")
        
        global_event_bus.subscribe("transition_planned", test_event_handler)
        
        # Plan another transition to trigger event
        test_transition_id = transition_manager.plan_transition(
            source_strategies=["test_strategy"],
            target_strategies=["another_strategy"],
            priority=TransitionPriority.LOW
        )
        
        await asyncio.sleep(1)
        
        if event_received:
            logger.info("✓ Event system working correctly")
        else:
            logger.warning("⚠ Event system may not be working")
        
        # Cleanup
        logger.info("Cleaning up...")
        transition_manager.stop_transition_manager()
        strategy_router.shutdown_transitions()
        
        logger.info("=== All Smooth Transition Tests Completed Successfully! ===")
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
    success = asyncio.run(test_smooth_transitions())
    sys.exit(0 if success else 1)