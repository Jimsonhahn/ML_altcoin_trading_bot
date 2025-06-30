python3 << 'EOF'
content = '''# strategies/__init__.py
"""Trading Strategies Module"""

# Import strategies with correct filenames
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .ml_strategy import MLStrategy

# Try to import advanced strategies
try:
    from .advanced import AdvancedStrategy
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False
    AdvancedStrategy = None

# Strategy mapping
STRATEGY_MAP = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "ml": MLStrategy,
}

if ADVANCED_AVAILABLE:
    STRATEGY_MAP["advanced"] = AdvancedStrategy

def get_strategy(strategy_name, config):
    """Factory function to get strategy instance"""
    if strategy_name not in STRATEGY_MAP:
        available = list(STRATEGY_MAP.keys())
        raise ValueError(f"Strategy {strategy_name} not found. Available: {available}")

    strategy_class = STRATEGY_MAP[strategy_name]
    return strategy_class(config)

__all__ = ["get_strategy", "STRATEGY_MAP", "MomentumStrategy", "MeanReversionStrategy", "MLStrategy"]
'''

with open('strategies/__init__.py', 'w') as f:
    f.write(content)
print("strategies/__init__.py wurde erfolgreich erstellt!")
EOF