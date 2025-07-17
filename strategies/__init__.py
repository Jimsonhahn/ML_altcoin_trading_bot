# strategies/__init__.py
import logging
from .strategy_base import Strategy
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .grid_trading import GridTradingStrategy
from .arbitrage import ArbitrageStrategy
from .defi_yield import DeFiYieldStrategy
from .liquidation import LiquidationStrategy
from .ml_strategy import MLStrategy # New ML Strategy example
# from .conservative import ConservativeStrategy # If you create this specific class

logger = logging.getLogger(__name__)

# Dictionary mapping strategy names to their classes
STRATEGIES = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "grid_trading": GridTradingStrategy,
    "arbitrage": ArbitrageStrategy,
    "defi_yield": DeFiYieldStrategy,
    "liquidation": LiquidationStrategy,
    "ml_strategy": MLStrategy,
    # "conservative": ConservativeStrategy # Uncomment if you add this strategy
    # "manual_intervention_required": None # This is a special flag for router, not a strategy class
}

logger.info(f"Loaded {len(STRATEGIES)} trading strategies.")