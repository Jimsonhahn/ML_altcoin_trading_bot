#!/usr/bin/env python3
"""
Test portfolio value updates with simulated trades
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from server.dashboard_api import app
import time

def test_portfolio_value_updates():
    """Test if portfolio values update correctly with trades"""
    print('💰 Testing Portfolio Value Updates with Simulated Trades...')
    
    with app.test_client() as client:
        # Check initial portfolio
        print('\n📊 Initial Portfolio Status:')
        response = client.get('/api/portfolio')
        portfolio = response.get_json()
        print(f'   Total Value: ${portfolio.get("total_value", 0):,.2f}')
        print(f'   Virtual Balance: ${portfolio.get("total_balance", 0):,.2f}')
        print(f'   Total P&L: ${portfolio.get("total_pnl", 0):,.2f}')
        print(f'   Daily P&L: ${portfolio.get("daily_pnl", 0):,.2f}')
        print(f'   Win Rate: {portfolio.get("win_rate", 0):.1f}%')
        
        # Simulate some trades directly with the paper trading engine
        print('\n📈 Simulating Trades...')
        
        # Get the server bot instance
        from server.dashboard_api import server_bot
        if server_bot and server_bot.trading_bot and server_bot.trading_bot.paper_engine:
            paper_engine = server_bot.trading_bot.paper_engine
            
            # Simulate successful trade
            print('   🟢 Executing profitable BUY trade: 0.1 BTC @ $50,000')
            trade1 = paper_engine.place_virtual_trade('BTC/USDT', 'BUY', 0.1, 50000, 'test_strategy')
            print(f'   Trade 1 Result: {trade1}')
            
            # Simulate selling at higher price for profit
            if trade1.get('success'):
                time.sleep(0.1)  # Small delay
                print('   🟢 Closing profitable trade: 0.1 BTC @ $52,000')
                close1 = paper_engine.close_trade(trade1['trade_id'], 52000)
                print(f'   Close 1 Result: P&L ${close1.get("pnl", 0):,.2f} | New Balance: ${close1.get("new_balance", 0):,.2f}')
            
            # Simulate losing trade
            print('   🔴 Executing losing BUY trade: 0.05 BTC @ $51,000')
            trade2 = paper_engine.place_virtual_trade('BTC/USDT', 'BUY', 0.05, 51000, 'test_strategy')
            print(f'   Trade 2 Result: {trade2}')
            
            # Simulate selling at lower price for loss
            if trade2.get('success'):
                time.sleep(0.1)  # Small delay
                print('   🔴 Closing losing trade: 0.05 BTC @ $49,000')
                close2 = paper_engine.close_trade(trade2['trade_id'], 49000)
                print(f'   Close 2 Result: P&L ${close2.get("pnl", 0):,.2f} | New Balance: ${close2.get("new_balance", 0):,.2f}')
            
            # Check updated portfolio
            print('\n📊 Updated Portfolio Status After Trades:')
            response = client.get('/api/portfolio')
            portfolio = response.get_json()
            print(f'   Total Value: ${portfolio.get("total_value", 0):,.2f}')
            print(f'   Virtual Balance: ${portfolio.get("total_balance", 0):,.2f}')
            print(f'   Total P&L: ${portfolio.get("total_pnl", 0):,.2f}')
            print(f'   Daily P&L: ${portfolio.get("daily_pnl", 0):,.2f}')
            print(f'   Win Rate: {portfolio.get("win_rate", 0):.1f}%')
            print(f'   Total Trades: {portfolio.get("total_trades", 0)}')
            print(f'   Winning Trades: {portfolio.get("winning_trades", 0)}')
            
            # Test trade history
            print('\n📋 Trade History:')
            response = client.get('/api/trades/history?limit=10')
            history = response.get_json()
            print(f'   Total Trades in History: {history.get("total", 0)}')
            for i, trade in enumerate(history.get("trades", [])[:5]):
                print(f'   Trade {i+1}: {trade.get("symbol")} | P&L: ${trade.get("pnl", 0):,.2f} | Strategy: {trade.get("strategy")}')
                
        else:
            print('❌ Paper trading engine not available for testing')

if __name__ == '__main__':
    test_portfolio_value_updates()