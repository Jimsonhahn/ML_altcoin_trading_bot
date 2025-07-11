#!/usr/bin/env python3
"""Debug Script für Import-Probleme"""

print("🔍 IMPORT DEBUG TOOL")
print("===================\n")

# Test 1: Strategies module
print("1. Testing strategies module...")
try:
    import strategies
    print("   ✅ strategies module imported")

    # Check what's available
    attrs = [attr for attr in dir(strategies) if not attr.startswith('_')]
    print(f"   Available attributes: {attrs}")

    if hasattr(strategies, 'STRATEGIES'):
        print(f"   ✅ STRATEGIES found with {len(strategies.STRATEGIES)} entries")
        print(f"   Available strategies: {list(strategies.STRATEGIES.keys())}")
    else:
        print("   ❌ STRATEGIES not found")

    if hasattr(strategies, 'STRATEGY_MAP'):
        print("   ✅ STRATEGY_MAP found (alias)")

except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Trading bot
print("\n2. Testing trading_bot module...")
try:
    from core.trading_bot import TradingBot
    print("   ✅ TradingBot imported successfully")
except Exception as e:
    print(f"   ❌ Error: {e}")
    print("   Trying to identify the specific issue...")
    try:
        import core.trading_bot
    except Exception as e2:
        print(f"   Failed to import module: {e2}")

# Test 3: Main
print("\n3. Testing main.py...")
try:
    import main
    print("   ✅ main.py imported successfully")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "="*50)
print("Debug completed. Fix any ❌ errors above.")
