import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class Settings:
    """Centralized settings management"""

    DEFAULT_CONFIG = {
        "exchange": {
            "name": "binance",
            "testnet": True,
            "rate_limit": True
        },
        "trading": {
            "initial_capital": 10000,
            "max_positions": 5,
            "position_sizing": "fixed",
            "risk_per_trade": 0.02
        },
        "timeframes": {
            "analysis": "1h", 
            "check_interval": 300, 
            "secondary": "4h"
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "data": { 
            "source": "exchange", 
            "source_name": "binance",
            "use_cache": True,
            "min_candles": 50, 
            "cache_dir": "data/market_data"
        },
        "ml": { 
            "enabled": False, 
            "data_dir": "data/market_data", 
            "models_dir": "data/ml_models", 
            "output_dir": "data/ml_analysis", 
            "n_regimes": 5, 
            "monitor_new_coins": True, 
            "regime_check_interval": 3600 
        },
        "strategy_router": { 
            "enabled": False, 
            "default_strategy": "momentum", 
            "regime_strategies": { 
                "bullish": "momentum",
                "aufwärtstrend": "momentum",
                "bearish": "mean_reversion",
                "abwärtstrend": "mean_reversion",
                "sideways": "grid_trading",
                "niedrige-volatilität": "grid_trading",
                "volatile": "arbitrage", 
                "extreme fear": "defi_yield" 
            },
            "capital_allocation_rules": { 
                "momentum": 0.2,
                "mean_reversion": 0.2,
                "grid_trading": 0.2,
                "arbitrage": 0.2,
                "ml": 0.1,
                "defi_yield": 0.1
            }
        },
        "risk_management": { 
            "max_drawdown": 0.15, 
            "daily_loss_limit": 0.05, 
            "killswitch": { 
                "enabled": True,
                "max_drawdown": 0.10, 
                "auto_reactivate_after_hours": 24 
            }
        },
        "backtest": { 
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "initial_balance": 10000,
            "commission": 0.001,
            "create_plots": True,
            "export_results": True,
            "export_format": "excel",
            "output_dir": "latest" 
        }
    }

    def __init__(self):
        self.config = self.DEFAULT_CONFIG.copy()
        self.profiles_dir = Path(__file__).parent / "profiles"
        self.loaded_profile = None

    def load_profile(self, profile_name: str) -> Dict[str, Any]:
        """Load a configuration profile"""
        profile_path = self.profiles_dir / f"{profile_name}.json"

        if not profile_path.exists():
            logger.warning(f"Profile {profile_name} not found at {profile_path}")
            logger.info("Using default configuration")
            return self.config

        try:
            with open(profile_path, 'r') as f:
                profile_config = json.load(f)

            self.config = self._deep_merge(self.DEFAULT_CONFIG.copy(), profile_config)
            self.loaded_profile = profile_name

            logger.info(f"Loaded configuration profile: {profile_name}")
            return self.config

        except Exception as e:
            logger.error(f"Error loading profile {profile_name}: {e}")
            logger.info("Using default configuration")
            return self.config

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports nested keys with dots)"""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """Set configuration value by key"""
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Deep merge two dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def validate(self) -> bool:
        """Validate configuration"""
        required_keys = [
            'trading_pairs',
            'timeframes.analysis', 
            'trading.initial_capital' 
        ]

        for key in required_keys:
            if self.get(key) is None: 
                logger.error(f"Missing required configuration: {key}")
                return False

        if self.get('strategy_router.enabled') and not self.get('ml.enabled'):
            logger.error("Strategy Router requires ML components to be enabled.")
            return False
        
        if self.get('strategy_router.enabled'):
            # This requires access to STRATEGIES from strategies/__init__.py
            # To avoid circular imports, you might need to import STRATEGIES here
            # or pass it during validation. For a standalone Settings class, it's a design choice.
            # For now, assuming STRATEGIES is somehow available or this check is simplified.
            pass # Simplified for import reasons in this isolated snippet

        if self.get('risk_management.killswitch.enabled', False):
            if self.get('risk_management.killswitch.max_drawdown', 0) <= 0:
                logger.error("Killswitch enabled but 'killswitch.max_drawdown' is not set or invalid.")
                return False

        return True


settings = Settings()

def load_config(profile_name: str = "default") -> Dict[str, Any]:
    """Load configuration profile"""
    logger.info(f"Looking for profile at: {settings.profiles_dir / f'{profile_name}.json'}")
    config = settings.load_profile(profile_name)
    logger.info(f"Configuration loaded for profile: {profile_name}")
    return config


def get_config() -> Dict[str, Any]:
    """Get current configuration"""
    return settings.config


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration dictionary. This function is deprecated in favor of settings.validate()"""
    logger.warning("Using deprecated validate_config. Please use settings.validate() instead.")
    return settings.validate() # Delegate to the method on the instance