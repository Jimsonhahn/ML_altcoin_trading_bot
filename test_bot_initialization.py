#!/usr/bin/env python3
"""
Test bot initialization to identify why portfolio shows $0
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

def test_bot_initialization():
    """Test if bot wrapper is properly initialized"""
    print('🤖 Testing Bot Initialization...')
    
    try:
        from server.bot_wrapper import ServerBotWrapper
        
        # Create bot wrapper
        bot_wrapper = ServerBotWrapper()
        print(f'✅ ServerBotWrapper created')
        print(f'   trading_bot: {bot_wrapper.trading_bot}')
        print(f'   is_running: {bot_wrapper.is_running}')
        print(f'   current_mode: {bot_wrapper.current_mode}')
        
        # Test initialize method
        print('\n🔧 Testing bot initialization...')
        success = bot_wrapper.initialize(mode='paper', strategy='adaptive_auto_strategy')
        print(f'   Initialize success: {success}')
        print(f'   trading_bot after init: {bot_wrapper.trading_bot}')
        
        if bot_wrapper.trading_bot:
            print(f'   Paper trading enabled: {bot_wrapper.trading_bot.paper_trading}')
            if bot_wrapper.trading_bot.paper_engine:
                print(f'   Paper engine balance: ${bot_wrapper.trading_bot.paper_engine.virtual_balance}')
                portfolio_status = bot_wrapper.trading_bot.paper_engine.get_virtual_portfolio_status()
                print(f'   Portfolio status: {portfolio_status}')
        
        # Test portfolio summary
        print('\n💰 Testing portfolio summary...')
        portfolio = bot_wrapper.get_portfolio_summary()
        print(f'   Portfolio summary: {portfolio}')
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_bot_initialization()