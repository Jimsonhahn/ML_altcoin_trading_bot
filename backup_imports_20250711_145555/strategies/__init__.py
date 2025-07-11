"""Trading Strategies Package"""
from .strategy_base import Strategy, Signal

# Import all strategies with error handling
strategies_map = {}

# Core Strategies
try:
    from .momentum import MomentumStrategy
    strategies_map['momentum'] = MomentumStrategy
except ImportError: pass

try:
    from .mean_reversion import MeanReversionStrategy
    strategies_map['mean_reversion'] = MeanReversionStrategy
except ImportError: pass

try:
    from .ml_strategy import MLStrategy
    strategies_map['ml'] = MLStrategy
except ImportError: pass

# Advanced Strategies
try:
    from .grid_trading import GridTradingStrategy
    strategies_map['grid_trading'] = GridTradingStrategy
except ImportError: pass

try:
    from .arbitrage import ArbitrageStrategy
    strategies_map['arbitrage'] = ArbitrageStrategy
except ImportError: pass

try:
    from .defi_yield import DeFiYieldStrategy
    strategies_map['defi_yield'] = DeFiYieldStrategy
except ImportError: pass

try:
    from .liquidation import LiquidationStrategy
    strategies_map['liquidation'] = LiquidationStrategy
except ImportError: pass

try:
    from .copy_trading import CopyTradingStrategy
    strategies_map['copy_trading'] = CopyTradingStrategy
except ImportError: pass

try:
    from .autopilot import UltimateAutoPilotStrategy
    strategies_map['autopilot'] = UltimateAutoPilotStrategy
except ImportError: pass

# Create STRATEGIES registry
STRATEGIES = strategies_map

def get_strategy(name: str):
    """Get strategy by name"""
    strategy_class = STRATEGIES.get(name.lower())
    if not strategy_class:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    return strategy_class

__all__ = ['Strategy', 'Signal', 'STRATEGIES', 'get_strategy']
