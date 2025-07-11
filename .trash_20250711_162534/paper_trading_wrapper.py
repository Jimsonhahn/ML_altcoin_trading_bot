#!/usr/bin/env python3
"""Paper Trading Wrapper - Umgeht Exchange-Probleme"""
import os
os.environ['PAPER_TRADING_ONLY'] = 'true'

# Patch exchange bevor es geladen wird
import sys
from unittest.mock import MagicMock

# Mock exchange für Paper Trading
class MockExchange:
    def __init__(self, *args, **kwargs):
        self.has = {'fetchOHLCV': True, 'fetchTicker': True}

    def fetch_ticker(self, symbol):
        prices = {
            'BTC/USDT': 118000,
            'ETH/USDT': 3000,
            'SOL/USDT': 165,
            'DOGE/USDT': 0.075,
            'SHIB/USDT': 0.000008
        }
        return {'symbol': symbol, 'last': prices.get(symbol, 100)}

    def fetch_ohlcv(self, symbol, timeframe='1h', limit=100):
        import time
        import random
        data = []
        base_price = self.fetch_ticker(symbol)['last']
        for i in range(limit):
            timestamp = int(time.time() * 1000) - (i * 3600000)
            volatility = random.uniform(0.98, 1.02)
            o = base_price * volatility
            h = o * random.uniform(1.0, 1.01)
            l = o * random.uniform(0.99, 1.0)
            c = random.uniform(l, h)
            v = random.uniform(1000, 10000)
            data.append([timestamp, o, h, l, c, v])
        return list(reversed(data))

# Patch ccxt
sys.modules['ccxt'] = MagicMock()
sys.modules['ccxt'].binance = MockExchange

# Jetzt main importieren und ausführen
import main
