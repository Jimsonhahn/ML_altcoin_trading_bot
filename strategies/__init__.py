"""Trading Strategies Module"""
import logging

logger = logging.getLogger(__name__)

STRATEGIES = {}

print("Loading trading strategies...")

# Core Strategies
try:
    from .momentum import MomentumStrategy
    STRATEGIES['momentum'] = MomentumStrategy
    print("✅ Momentum strategy loaded")
except ImportError as e:
    logger.warning(f"Could not import MomentumStrategy: {e}")

try:
    from .mean_reversion import MeanReversionStrategy
    STRATEGIES['mean_reversion'] = MeanReversionStrategy
    print("✅ Mean Reversion strategy loaded")
except ImportError as e:
    logger.warning(f"Could not import MeanReversionStrategy: {e}")

try:
    from .ml_strategy import MLStrategy
    STRATEGIES['ml'] = MLStrategy
    print("✅ ML strategy loaded")
except ImportError as e:
    logger.warning(f"Could not import MLStrategy: {e}")

# Advanced Strategies
try:
    from .grid_trading import GridTradingStrategy
    STRATEGIES['grid_trading'] = GridTradingStrategy
    print("✅ Grid Trading strategy loaded")
except ImportError as e:
    logger.warning(f"Could not import GridTradingStrategy: {e}")

try:
    from .arbitrage import ArbitrageStrategy
    STRATEGIES['arbitrage'] = ArbitrageStrategy
    print("✅ Arbitrage strategy loaded")
except ImportError as e:
    logger.warning(f"Could not import ArbitrageStrategy: {e}")

try:
    from .defi_yield import DeFiYieldStrategy
    STRATEGIES['defi_yield'] = DeFiYieldStrategy
    print("✅ DeFi Yield strategy loaded")
except ImportError as e:
    logger.warning(f"Could not import DeFiYieldStrategy: {e}")

try:
    from .liquidation import LiquidationStrategy
    STRATEGIES['liquidation'] = LiquidationStrategy
    print("✅ Liquidation strategy loaded")
except ImportError as e:
    logger.warning(f"Could not import LiquidationStrategy: {e}")

try:
    from .copy_trading import CopyTradingStrategy
    STRATEGIES['copy_trading'] = CopyTradingStrategy
    print("✅ Copy Trading strategy loaded")
except ImportError as e:
    logger.warning(f"Could not import CopyTradingStrategy: {e}")

# AutoPilot
try:
    from .autopilot import UltimateAutoPilotStrategy
    STRATEGIES['autopilot'] = UltimateAutoPilotStrategy
    STRATEGIES['ultimate_autopilot'] = UltimateAutoPilotStrategy
    print("✅ AutoPilot strategy loaded")
except ImportError as e:
    logger.warning(f"Could not import AutoPilotStrategy: {e}")

print(f"\nTotal strategies loaded: {len(STRATEGIES)}")
print(f"Available strategies: {list(STRATEGIES.keys())}")

def get_strategy(name: str):
    """Get strategy class by name"""
    return STRATEGIES.get(name.lower())

def list_strategies():
    """List all available strategies"""
    return list(STRATEGIES.keys())

STRATEGY_MAP = STRATEGIES

__all__ = ['STRATEGIES', 'STRATEGY_MAP', 'get_strategy', 'list_strategies']
