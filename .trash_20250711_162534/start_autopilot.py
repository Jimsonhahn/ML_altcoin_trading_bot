#!/usr/bin/env python3
"""
AutoPilot Starter - Startet alle 6 Strategien gleichzeitig!
"""
import os
import sys
import subprocess

print("🚀 STARTING AUTOPILOT WITH ALL 6 STRATEGIES!")
print("=" * 60)

# Test if everything is ready
print("\n1. Testing strategy loading...")
try:
    from strategies import STRATEGIES
    print(f"✅ Loaded {len(STRATEGIES)} strategies: {list(STRATEGIES.keys())}")

    if 'autopilot' in STRATEGIES:
        print("✅ AutoPilot is ready!")

        # Test instantiation
        autopilot = STRATEGIES['autopilot']({})
        print(f"✅ AutoPilot has {len(autopilot.strategies)} sub-strategies active")
    else:
        print("❌ AutoPilot not found!")
        sys.exit(1)

except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

print("\n2. Starting bot with AutoPilot...")
print("=" * 60)

# Start the bot
cmd = [
    sys.executable,
    "main.py",
    "--mode=paper",
    "--strategy=autopilot",
    "--config=aggressive"
]

print(f"Running: {' '.join(cmd)}")
print("\n💰 ALL 6 STRATEGIES ARE NOW RUNNING SIMULTANEOUSLY!")
print("Grid + Arbitrage + DeFi + Liquidation + Copy + ML = MAXIMUM PROFIT!")
print("=" * 60)
print()

# Run the bot
subprocess.run(cmd)
