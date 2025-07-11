#!/usr/bin/env python3
"""
Paper Trading Runner - Startet alle 6 Strategien im Paper Mode
"""
import os
import sys

# Set paper trading mode
os.environ['TRADING_MODE'] = 'paper'
os.environ['SKIP_EXCHANGE_CHECK'] = 'true'

print("🚀 STARTING PAPER TRADING WITH ALL 6 STRATEGIES!")
print("=" * 60)

# Monkey patch the exchange before importing anything else
print("Setting up paper trading environment...")

# Create mock exchange module
mock_exchange_code = '''
class ExchangeManager:
    def __init__(self, exchange_name='binance', mode='paper'):
        self.exchange_name = exchange_name
        self.mode = mode
        self.connected = True
        print(f"📊 Mock Exchange initialized for {exchange_name} in {mode} mode")

    def connect(self):
        return True

    def fetch_ticker(self, symbol):
        prices = {
            'BTC/USDT': {'last': 118000, 'bid': 117900, 'ask': 118100},
            'ETH/USDT': {'last': 3000, 'bid': 2995, 'ask': 3005},
            'SOL/USDT': {'last': 165, 'bid': 164.5, 'ask': 165.5},
            'DOGE/USDT': {'last': 0.075, 'bid': 0.0749, 'ask': 0.0751},
            'SHIB/USDT': {'last': 0.000008, 'bid': 0.0000079, 'ask': 0.0000081}
        }
        return prices.get(symbol, {'last': 100, 'bid': 99, 'ask': 101})

    def fetch_ohlcv(self, symbol, timeframe='1h', limit=100):
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta

        # Generate mock OHLCV data
        base_price = self.fetch_ticker(symbol)['last']
        timestamps = pd.date_range(end=datetime.now(), periods=limit, freq='1H')

        data = []
        for i, ts in enumerate(timestamps):
            volatility = np.random.uniform(0.98, 1.02)
            o = base_price * volatility
            h = o * np.random.uniform(1.0, 1.01)
            l = o * np.random.uniform(0.99, 1.0)
            c = np.random.uniform(l, h)
            v = np.random.uniform(100000, 1000000)
            data.append([ts, o, h, l, c, v])

        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return df

ExchangeFactory = type('ExchangeFactory', (), {
    'create': lambda self, *args, **kwargs: ExchangeManager(*args, **kwargs),
    'create_exchange': lambda self, *args, **kwargs: ExchangeManager(*args, **kwargs)
})()

create_exchange = lambda *args, **kwargs: ExchangeManager(*args, **kwargs)
'''

# Write mock exchange
with open('_mock_exchange.py', 'w') as f:
    f.write(mock_exchange_code)

# Inject mock
sys.path.insert(0, '.')
import _mock_exchange
sys.modules['core.exchange'] = _mock_exchange

print("✅ Paper trading environment ready")
print("\nStarting bot with AutoPilot...")
print("=" * 60)

# Now import and run
try:
    import main
    # The main module will run automatically
except SystemExit:
    pass
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Cleanup
import os
if os.path.exists('_mock_exchange.py'):
    os.remove('_mock_exchange.py')
