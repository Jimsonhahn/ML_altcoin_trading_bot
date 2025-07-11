"""Trading Strategies Package"""
from .strategy_base import Strategy, Signal

# Initialize empty registry
STRATEGIES = {}

# Import all strategies with error handling
print("Loading trading strategies...")

# Core Strategies
try:
    from .momentum import MomentumStrategy
    STRATEGIES['momentum'] = MomentumStrategy
    print("✅ Momentum strategy loaded")
except ImportError as e:
    print(f"⚠️  Could not load Momentum: {e}")

try:
    from .mean_reversion import MeanReversionStrategy
    STRATEGIES['mean_reversion'] = MeanReversionStrategy
    print("✅ Mean Reversion strategy loaded")
except ImportError as e:
    print(f"⚠️  Could not load Mean Reversion: {e}")

try:
    from .ml_strategy import MLStrategy
    STRATEGIES['ml'] = MLStrategy
    print("✅ ML strategy loaded")
except ImportError as e:
    print(f"⚠️  Could not load ML: {e}")

# Advanced Strategies
try:
    from .grid_trading import GridTradingStrategy
    STRATEGIES['grid_trading'] = GridTradingStrategy
    print("✅ Grid Trading strategy loaded")
except ImportError as e:
    print(f"⚠️  Could not load Grid Trading: {e}")

try:
    from .arbitrage import ArbitrageStrategy
    STRATEGIES['arbitrage'] = ArbitrageStrategy
    print("✅ Arbitrage strategy loaded")
except ImportError as e:
    print(f"⚠️  Could not load Arbitrage: {e}")

try:
    from .liquidation import LiquidationStrategy
    STRATEGIES['liquidation'] = LiquidationStrategy
    print("✅ Liquidation strategy loaded")
except ImportError as e:
    print(f"⚠️  Could not load Liquidation: {e}")

try:
    from .copy_trading import CopyTradingStrategy
    STRATEGIES['copy_trading'] = CopyTradingStrategy
    print("✅ Copy Trading strategy loaded")
except ImportError as e:
    print(f"⚠️  Could not load Copy Trading: {e}")

# AutoPilot - Die ultimative Strategie
try:
    from .autopilot import UltimateAutoPilotStrategy
    STRATEGIES['autopilot'] = UltimateAutoPilotStrategy
    print("✅ AutoPilot strategy loaded (orchestrates all 6 strategies)")
except ImportError as e:
    print(f"❌ Could not load AutoPilot: {e}")

# Aliases
STRATEGY_MAP = STRATEGIES  # Backward compatibility

print(f"\nTotal strategies loaded: {len(STRATEGIES)}")
print(f"Available strategies: {list(STRATEGIES.keys())}")

def get_strategy(name: str):
    """Get strategy by name"""
    return STRATEGIES.get(name.lower())

__all__ = ['Strategy', 'Signal', 'STRATEGIES', 'STRATEGY_MAP', 'get_strategy']
