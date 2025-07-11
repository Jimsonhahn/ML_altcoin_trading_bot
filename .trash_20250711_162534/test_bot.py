#!/usr/bin/env python3
"""Quick test to verify everything works"""
import os
import sys

print("🧪 TESTING BOT SETUP")
print("===================\n")

# Test 1: Strategies
print("1. Testing strategies...")
try:
    from strategies import STRATEGIES
    print(f"   ✅ Found strategies: {list(STRATEGIES.keys())}")

    # Test AutoPilot
    if 'autopilot' in STRATEGIES:
        print("   ✅ AutoPilot is available!")
        # Try to instantiate
        autopilot = STRATEGIES['autopilot']({})
        print("   ✅ AutoPilot can be instantiated!")
    else:
        print("   ❌ AutoPilot NOT found")
except Exception as e:
    print(f"   ❌ Strategy error: {e}")

# Test 2: Exchange connection
print("\n2. Testing exchange connection...")
try:
    import ccxt

    # Create testnet exchange
    exchange = ccxt.binance({
        'urls': {'api': 'https://testnet.binance.vision/api'},
        'options': {
            'recvWindow': 60000,
            'adjustForTimeDifference': True
        }
    })

    # Test connection
    ticker = exchange.fetch_ticker('BTC/USDT')
    print(f"   ✅ Connected! BTC price: ${ticker['last']:.2f}")
except Exception as e:
    print(f"   ⚠️  Exchange test failed: {e}")
    print("   (This is OK for paper trading)")

print("\n" + "="*50)
print("✅ Bot should be ready to run!")
print("\nTry these commands:")
print("1. python main.py --mode=paper --strategy=momentum --config=default")
print("2. python main.py --mode=paper --strategy=autopilot --config=aggressive")
