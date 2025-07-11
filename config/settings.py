"""
Settings Management for Trading Bot
Handles configuration loading and validation
"""
import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class Settings:
    """Centralized settings management"""

    # Default configuration
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
            "primary": "1h",
            "secondary": "4h"
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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

            # Merge with default config
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
            'timeframe',
            'initial_capital'
        ]

        for key in required_keys:
            if key not in self.config:
                logger.error(f"Missing required configuration: {key}")
                return False

        return True

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self.get(key)

    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-style setting"""
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """Check if key exists"""
        return self.get(key) is not None


# Global settings instance
settings = Settings()


# Configuration loading functions for backward compatibility
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
    """Validate configuration dictionary"""
    required_keys = ['trading_pairs', 'timeframe', 'initial_capital']

    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required configuration: {key}")
            return False

    return True
