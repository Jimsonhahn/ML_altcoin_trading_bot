# strategies/__init__.py - Fixed version with proper error handling
import logging
from typing import Dict, Type, Optional

logger = logging.getLogger(__name__)

# Import base strategy first
try:
    from .strategy_base import Strategy
    HAS_STRATEGY_BASE = True
except ImportError as e:
    logger.error(f"Failed to import strategy base: {e}")
    HAS_STRATEGY_BASE = False
    Strategy = None

# Strategy imports with individual error handling
STRATEGIES = {}

# Core strategies
try:
    from .momentum import MomentumStrategy
    STRATEGIES["momentum"] = MomentumStrategy
except ImportError as e:
    logger.warning(f"Failed to import MomentumStrategy: {e}")

try:
    from .mean_reversion import MeanReversionStrategy
    STRATEGIES["mean_reversion"] = MeanReversionStrategy
except ImportError as e:
    logger.warning(f"Failed to import MeanReversionStrategy: {e}")

try:
    from .grid_trading import GridTradingStrategy
    STRATEGIES["grid_trading"] = GridTradingStrategy
except ImportError as e:
    logger.warning(f"Failed to import GridTradingStrategy: {e}")

try:
    from .arbitrage import ArbitrageStrategy
    STRATEGIES["arbitrage"] = ArbitrageStrategy
except ImportError as e:
    logger.warning(f"Failed to import ArbitrageStrategy: {e}")

try:
    from .defi_yield import DeFiYieldStrategy
    STRATEGIES["defi_yield"] = DeFiYieldStrategy
except ImportError as e:
    logger.warning(f"Failed to import DeFiYieldStrategy: {e}")

try:
    from .liquidation import LiquidationStrategy
    STRATEGIES["liquidation"] = LiquidationStrategy
except ImportError as e:
    logger.warning(f"Failed to import LiquidationStrategy: {e}")

# Enhanced ML strategy
try:
    from .ml_strategy import MLStrategy
    STRATEGIES["ml_strategy"] = MLStrategy
except ImportError as e:
    logger.warning(f"Failed to import MLStrategy: {e}")

# Master strategy
try:
    from .lazy_billionaire_strategy import LazyBillionaireStrategy
    STRATEGIES["lazy_billionaire"] = LazyBillionaireStrategy
except ImportError as e:
    logger.warning(f"Failed to import LazyBillionaireStrategy: {e}")

# Conservative strategy (placeholder)
try:
    from .conservative import ConservativeStrategy
    STRATEGIES["conservative"] = ConservativeStrategy
except ImportError as e:
    logger.debug(f"ConservativeStrategy not available: {e}")

# Log successful imports
logger.info(f"Successfully loaded {len(STRATEGIES)} trading strategies: {list(STRATEGIES.keys())}")

# Validation function
def get_strategy_class(strategy_name: str) -> Optional[Type[Strategy]]:
    """
    Get strategy class by name with validation
    
    Args:
        strategy_name: Name of the strategy
        
    Returns:
        Strategy class or None if not found
    """
    if not HAS_STRATEGY_BASE:
        logger.error("Strategy base not available")
        return None
    
    if strategy_name not in STRATEGIES:
        logger.error(f"Strategy '{strategy_name}' not found. Available: {list(STRATEGIES.keys())}")
        return None
    
    return STRATEGIES[strategy_name]

def list_available_strategies() -> Dict[str, str]:
    """
    List all available strategies with descriptions
    
    Returns:
        Dictionary mapping strategy names to descriptions
    """
    descriptions = {
        "momentum": "Trend-following strategy using momentum indicators",
        "mean_reversion": "Counter-trend strategy using mean reversion",
        "grid_trading": "Grid-based automated trading strategy",
        "arbitrage": "Cross-exchange arbitrage opportunities",
        "defi_yield": "DeFi yield farming and staking strategy",
        "liquidation": "Liquidation opportunity hunting strategy",
        "ml_strategy": "ML-enhanced trading with market predictions",
        "lazy_billionaire": "Master strategy with dynamic allocation",
        "conservative": "Conservative low-risk strategy"
    }
    
    available = {}
    for strategy_name in STRATEGIES:
        available[strategy_name] = descriptions.get(strategy_name, "No description available")
    
    return available

def validate_strategy_config(strategy_name: str, config: Dict) -> bool:
    """
    Validate strategy configuration
    
    Args:
        strategy_name: Name of the strategy
        config: Strategy configuration dictionary
        
    Returns:
        True if configuration is valid
    """
    try:
        strategy_class = get_strategy_class(strategy_name)
        if not strategy_class:
            return False
        
        # Try to instantiate with config
        strategy_class(config)
        return True
        
    except Exception as e:
        logger.error(f"Strategy configuration validation failed for {strategy_name}: {e}")
        return False

# Export main components
__all__ = [
    'Strategy',
    'STRATEGIES',
    'get_strategy_class',
    'list_available_strategies',
    'validate_strategy_config',
    'HAS_STRATEGY_BASE'
]

# Additional fallback strategies for missing implementations
class FallbackStrategy(Strategy):
    """Fallback strategy that does nothing"""
    
    def __init__(self, params=None, ml_components=None):
        super().__init__(params or {}, ml_components)
        self.name = "fallback"
    
    def calculate_signal(self, symbol, data, current_price):
        return 'HOLD', {'confidence': 0.0, 'reason': 'fallback_strategy'}

# Add fallback for missing strategies
if "conservative" not in STRATEGIES:
    STRATEGIES["conservative"] = FallbackStrategy
    logger.info("Added fallback ConservativeStrategy")

# Ensure we have at least basic strategies
REQUIRED_STRATEGIES = ["momentum", "mean_reversion"]
missing_strategies = [s for s in REQUIRED_STRATEGIES if s not in STRATEGIES]

if missing_strategies:
    logger.warning(f"Missing required strategies: {missing_strategies}")
    # Add fallback implementations
    for strategy_name in missing_strategies:
        STRATEGIES[strategy_name] = FallbackStrategy
        logger.info(f"Added fallback {strategy_name} strategy")

logger.info(f"Strategy registry initialized with {len(STRATEGIES)} strategies")