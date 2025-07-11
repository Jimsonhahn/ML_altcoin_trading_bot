"""
Trading Strategies Package
Exportiert alle verfügbaren Trading-Strategien
"""

from .strategy_base import Strategy, Signal
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .ml_strategy import MLStrategy
from .autopilot import UltimateAutoPilotStrategy
from .grid_trading import GridTradingStrategy
from .arbitrage import ArbitrageStrategy
from .defi_yield import DeFiYieldStrategy
from .liquidation import LiquidationStrategy
from .copy_trading import CopyTradingStrategy

# Strategy Registry - Alle 6 Haupt-Strategien plus Orchestrator
STRATEGIES = {
    'momentum': MomentumStrategy,
    'mean_reversion': MeanReversionStrategy,
    'ml': MLStrategy,
    'autopilot': UltimateAutoPilotStrategy,  # Ultimate 6-Strategy Orchestrator
    'grid_trading': GridTradingStrategy,
    'arbitrage': ArbitrageStrategy,
    'defi_yield': DeFiYieldStrategy,
    'liquidation': LiquidationStrategy,
    'copy_trading': CopyTradingStrategy
}


def get_strategy(name: str):
    """
    Factory-Funktion zum Abrufen einer Strategie nach Name
    """
    strategy_class = STRATEGIES.get(name.lower())
    if not strategy_class:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    return strategy_class

# Exportiere alle verfügbaren Klassen
__all__ = [
    'Strategy',
    'Signal',
    'MomentumStrategy',
    'MeanReversionStrategy',
    'MLStrategy',
    'UltimateAutoPilotStrategy',
    'GridTradingStrategy',
    'ArbitrageStrategy',
    'DeFiYieldStrategy',
    'LiquidationStrategy',
    'CopyTradingStrategy',
    'STRATEGIES',
    'get_strategy'
]