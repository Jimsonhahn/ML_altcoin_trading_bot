#!/usr/bin/env python3
"""
Test Actual Trading Execution
Tests if the bot can actually initiate trades
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def test_trading_execution():
    """Test if trading bot can actually execute trades"""
    logger.info("=" * 60)
    logger.info("TESTING ACTUAL TRADING EXECUTION")
    logger.info("=" * 60)
    
    try:
        # Import required components
        from config.settings import Settings
        from core.trading_bot import TradingBot
        
        logger.info("Creating bot for trading test...")
        
        # Create settings
        settings = Settings()
        
        # Create data manager
        from data_sources.data_manager import DataManager
        data_manager = DataManager(settings)
        
        # Create bot in paper mode with proper data manager
        bot = TradingBot(
            mode='paper',
            strategy_name='momentum',
            settings=settings,
            data_manager=data_manager,
            ml_components=None,
            strategy_router=None,
            safety_manager=None
        )
        
        logger.info("✓ Bot created successfully")
        
        # Check paper balance before trading
        initial_balance = bot.paper_engine.virtual_balance
        logger.info(f"Initial paper balance: ${initial_balance:.2f}")
        
        # Start the bot
        logger.info("Starting bot...")
        bot.is_running = True
        
        # Test single symbol processing (this should generate signals and potentially trades)
        test_symbol = 'BTC/USDT'
        logger.info(f"Processing symbol: {test_symbol}")
        
        try:
            # This is where the actual trading happens
            await bot._process_symbol(test_symbol)
            logger.info("✓ Symbol processing completed")
            
            # Check if any trades were made
            trade_history = bot.paper_engine.trade_history
            open_positions = bot.paper_engine.virtual_positions
            
            logger.info(f"Trade history: {len(trade_history)} trades")
            logger.info(f"Open positions: {len(open_positions)} positions")
            
            # Check if balance changed
            final_balance = bot.paper_engine.virtual_balance
            logger.info(f"Final paper balance: ${final_balance:.2f}")
            
            if len(trade_history) > 0:
                logger.info("🎉 SUCCESS: Trades were executed!")
                for i, trade in enumerate(trade_history):
                    logger.info(f"  Trade {i+1}: {trade.side} {trade.size} {trade.symbol} @ ${trade.entry_price}")
            elif len(open_positions) > 0:
                logger.info("🎯 SUCCESS: Positions opened!")
                for pos_id, position in open_positions.items():
                    logger.info(f"  Position: {position.side} {position.size} {position.symbol} @ ${position.entry_price}")
            else:
                logger.warning("⚠ No trades or positions created")
                logger.info("This could mean:")
                logger.info("  • Strategy conditions not met")
                logger.info("  • Market data issues")
                logger.info("  • Risk management blocking trades")
                
                # Let's check strategy signals
                if hasattr(bot, 'strategies') and 'momentum' in bot.strategies:
                    strategy = bot.strategies['momentum']
                    logger.info(f"Strategy loaded: {strategy}")
                    
                    # Test market data fetching
                    try:
                        market_data = await bot._fetch_market_data(test_symbol)
                        if market_data:
                            logger.info(f"✓ Market data available: {list(market_data.keys())}")
                            
                            # Test signal generation
                            if hasattr(strategy, 'generate_signals'):
                                signals = await strategy.generate_signals(test_symbol, market_data)
                                if signals:
                                    logger.info(f"✓ Strategy generated {len(signals)} signals")
                                    for signal in signals:
                                        logger.info(f"  Signal: {signal}")
                                else:
                                    logger.warning("⚠ Strategy generated no signals")
                                    logger.info("Market conditions may not meet strategy criteria")
                            else:
                                logger.error("✗ Strategy has no generate_signals method")
                        else:
                            logger.error("✗ No market data retrieved")
                    except Exception as e:
                        logger.error(f"✗ Market data test failed: {e}")
                
        except Exception as e:
            logger.error(f"✗ Symbol processing failed: {e}")
            logger.error(f"This indicates a problem in the trading pipeline")
            import traceback
            traceback.print_exc()
        
        # Stop the bot
        bot.is_running = False
        
        # Final analysis
        portfolio_status = bot.paper_engine.get_virtual_portfolio_status()
        performance = bot.paper_engine.performance_metrics
        
        logger.info("\n--- FINAL ANALYSIS ---")
        logger.info(f"Portfolio value: ${portfolio_status.get('total_portfolio_value', 0):.2f}")
        logger.info(f"Total trades: {performance.get('total_trades', 0)}")
        logger.info(f"Win rate: {performance.get('win_rate', 0):.1%}")
        logger.info(f"Total P&L: ${performance.get('total_pnl', 0):.2f}")
        
        if performance.get('total_trades', 0) > 0:
            logger.info("🎉 TRADING SYSTEM IS WORKING!")
            logger.info("The bot successfully executed trades")
        else:
            logger.warning("🤔 NO TRADES EXECUTED")
            logger.info("The bot is running but not generating trades")
            logger.info("This might be due to:")
            logger.info("  • Conservative strategy settings")
            logger.info("  • Current market conditions")
            logger.info("  • Risk management being too restrictive")
        
    except Exception as e:
        logger.error(f"✗ Trading test failed: {e}")
        import traceback
        traceback.print_exc()
    

async def main():
    """Main execution"""
    await test_trading_execution()


if __name__ == "__main__":
    asyncio.run(main())