#!/usr/bin/env python3
"""
Quick test of dashboard API endpoints to identify data issues
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from server.dashboard_api import app
import json

def test_dashboard_endpoints():
    """Test all dashboard API endpoints"""
    print('🔍 Testing Dashboard API Endpoints...')
    
    with app.test_client() as client:
        # Test health endpoint
        try:
            response = client.get('/api/health')
            print(f'\n✅ Health Check: {response.status_code}')
            if response.status_code == 200:
                print(f'   Data: {response.get_json()}')
        except Exception as e:
            print(f'❌ Health check failed: {e}')
        
        # Test bot status
        try:
            response = client.get('/api/bot/status')
            status_data = response.get_json()
            print(f'\n📊 Bot Status: {response.status_code}')
            print(f'   Running: {status_data.get("running", "N/A")}')
            print(f'   Mode: {status_data.get("mode", "N/A")}')
            print(f'   Active Strategy: {status_data.get("active_strategy", "N/A")}')
            print(f'   Market Analysis: {status_data.get("market_analysis", "N/A")}')
            print(f'   Full response: {json.dumps(status_data, indent=2)}')
        except Exception as e:
            print(f'❌ Bot status failed: {e}')
        
        # Test portfolio endpoint
        try:
            response = client.get('/api/portfolio')
            portfolio_data = response.get_json()
            print(f'\n💰 Portfolio: {response.status_code}')
            print(f'   Total Value: ${portfolio_data.get("total_value", 0)}')
            print(f'   Daily P&L: ${portfolio_data.get("daily_pnl", 0)}')
            print(f'   Win Rate: {portfolio_data.get("win_rate", 0)}%')
            print(f'   Error: {portfolio_data.get("error", "None")}')
            print(f'   Full response: {json.dumps(portfolio_data, indent=2)}')
        except Exception as e:
            print(f'❌ Portfolio failed: {e}')
        
        # Test active trades
        try:
            response = client.get('/api/trades/active')
            trades_data = response.get_json()
            print(f'\n📈 Active Trades: {response.status_code}')
            if isinstance(trades_data, list):
                print(f'   Count: {len(trades_data)}')
            else:
                print(f'   Data: {trades_data}')
        except Exception as e:
            print(f'❌ Active trades failed: {e}')
        
        # Test trade history
        try:
            response = client.get('/api/trades/history?limit=5')
            history_data = response.get_json()
            print(f'\n📋 Trade History: {response.status_code}')
            print(f'   Data: {json.dumps(history_data, indent=2)}')
        except Exception as e:
            print(f'❌ Trade history failed: {e}')

if __name__ == '__main__':
    test_dashboard_endpoints()