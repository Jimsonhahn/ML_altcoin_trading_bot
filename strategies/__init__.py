# strategies/__init__.py
"""Trading Strategies Module"""

from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .ml_strategy import MLStrategy

STRATEGY_MAP = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "ml": MLStrategy,
}


def get_strategy(strategy_name, config):
    """Factory function to get strategy instance"""
    if strategy_name not in STRATEGY_MAP:
        available = list(STRATEGY_MAP.keys())
        raise ValueError(f"Strategy {strategy_name} not found. Available: {available}")

    strategy_class = STRATEGY_MAP[strategy_name]
    return strategy_class(config)