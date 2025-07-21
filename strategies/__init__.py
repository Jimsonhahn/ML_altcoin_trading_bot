# strategies/__init__.py
import logging
from .strategy_base import Strategy
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .grid_trading import GridTradingStrategy
from .arbitrage import ArbitrageStrategy
from .defi_yield import DeFiYieldStrategy
from .liquidation import LiquidationStrategy
from .ml_strategy import MLStrategy
from .lazy_billionaire_strategy import LazyBillionaireStrategy
from .super_lazy_billionaire_strategy import SuperLazyBillionaireStrategy
from .ultimate_btc_strategy import UltimateBTCStrategy
from .autopilot import AutopilotStrategy
from .copy_trading import CopyTradingStrategy

logger = logging.getLogger(__name__)

# Dictionary mapping strategy names to their classes
STRATEGIES = {
    "ultimate_btc": UltimateBTCStrategy,  # 🏆 ULTIMATE HIGH-PERFORMANCE STRATEGY
    "super_lazy_billionaire": SuperLazyBillionaireStrategy,  # 🚀 NEW MASTER STRATEGY
    "autopilot": AutopilotStrategy,
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "grid_trading": GridTradingStrategy,
    "arbitrage": ArbitrageStrategy,
    "defi_yield": DeFiYieldStrategy,
    "liquidation": LiquidationStrategy,
    "ml_strategy": MLStrategy,
    "lazy_billionaire": LazyBillionaireStrategy,  # Original strategy
    "copy_trading": CopyTradingStrategy,
}

logger.info(f"Loaded {len(STRATEGIES)} trading strategies.")