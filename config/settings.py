# config/settings.py
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
            "rate_limit": True,
            "maker_fee": 0.001,  # Example fee for simulation
            "taker_fee": 0.001  # Example fee for simulation
        },
        "trading": {
            "initial_capital": 10000,
            "max_positions": 5,
            "position_sizing": "fixed",
            "risk_per_trade": 0.02,
            "default_strategy": "momentum"  # Fallback strategy if auto-routing fails
        },
        "timeframes": {
            "analysis": "1h",
            "check_interval": 300,  # Bot loop check interval in seconds (5 mins)
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
            "min_candles": 200,  # Min candles needed for indicator calculation for strategies and ML
            "cache_dir": "data/market_data"
        },
        "ml": {
            "enabled": True,
            "auto_train": False,  # Set to True to auto-train ML models on startup if not present
            "data_dir": "data/market_data",
            "models_dir": "data/ml_models",
            "output_dir": "data/ml_analysis",
            "n_regimes": 5,  # Number of market regimes for clustering
            "min_data_points_for_ml": 200,  # Minimum data points required for ML training/prediction
            "regime_check_interval": 1800,  # How often to re-evaluate market regime (seconds, e.g., 30 mins)
            "regime_core_symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "SOL/USDT"],
            # Symbols for global regime detection
            "feature_extraction": {
                "rsi_period": 14,
                "ma_short": 20,
                "ma_long": 50,
                "bollinger_window": 20,
                "bollinger_std_dev": 2.0,
                "atr_period": 14
            },
            "sentiment_api_key": ""
            # API keys are now stored securely in SecretManager, not in config files
        },
        "strategy_router": {
            "enabled": True,
            "regime_strategies": {  # Maps detected regime to preferred strategies and their capital allocation weights
                "bull": {
                    "momentum": 0.7,
                    "ml_strategy": 0.3
                },
                "bear": {
                    "mean_reversion": 0.6,
                    "liquidation": 0.4
                },
                "sideways": {
                    "grid_trading": 0.8,
                    "arbitrage": 0.2
                },
                "volatile": {
                    "defi_yield": 0.5,  # Move to stable yield in high volatility
                    "conservative": 0.5
                    # A more conservative general strategy (placeholder for a less aggressive strategy)
                },
                "extreme_fear": {
                    "manual_intervention_required": 1.0
                    # Pause bot or require human decision (represented by 1.0 capital, but router should handle pausing logic)
                },
                "neutral": {  # Default if no clear regime is detected or for unknown
                    "momentum": 0.5,
                    "mean_reversion": 0.5
                }
            }
        },
        "risk_management": {  # General risk management settings
            "killswitch": {
                "enabled": True,
                "max_drawdown": 0.15,  # 15% max drawdown from peak equity
                "auto_reactivate_after_hours": 24,  # Auto-reactivate after 24 hours if killswitch triggered
                "notification_on_trigger": True
            },
            "max_position_size": 1000,  # Max USD value per single position
            "max_drawdown": 0.20,  # Overall max drawdown (redundant with killswitch.max_drawdown but kept for clarity)
            "stop_loss_percentage": 0.02,  # Default stop loss
            "take_profit_percentage": 0.05,  # Default take profit
            "max_positions": 5,  # Max number of concurrent open positions
            "risk_per_trade": 0.02  # Percentage of capital to risk per trade (for position sizing)
        },
        "notifications": {
            "telegram": {
                "enabled": False,
                "bot_token": "",  # Stored in SecretManager
                "chat_id": ""  # Stored in SecretManager
            },
            "email": {
                "enabled": False,
                "sender_email": "your_email@example.com",
                "recipient_email": "alert_recipient@example.com",
                "smtp_server": "smtp.yourprovider.com",
                "smtp_port": 587,
                "smtp_username": "",  # Stored in SecretManager
                "smtp_password": ""  # Stored in SecretManager
            }
        },
        "strategy_configs": {  # Detailed parameters for each strategy, overridden by profile-specific settings
            "momentum": {
                "trading_pair": "BTC/USDT",  # Example default pair, can be overridden by router
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "sma_short_period": 5,
                "sma_long_period": 20
            },
            "mean_reversion": {
                "trading_pair": "ETH/USDT",
                "bollinger_period": 20,
                "bollinger_std": 2.0,
                "use_rsi_filter": True
            },
            "grid_trading": {
                "trading_pair": "BNB/USDT",
                "num_grids": 10,
                "price_range_multiplier": 0.05,
                "grid_size_percent": 0.01  # New parameter
            },
            "arbitrage": {
                "trading_pair": "XRP/USDT",  # Example, usually involves multiple pairs/exchanges
                "min_profit_threshold": 0.005,
                "max_execution_slippage": 0.0002
            },
            "defi_yield": {
                "trading_pair": "USDT/USDC",  # Example stablecoin pair for yield farming concept
                "min_apy": 0.15,
                "compound_frequency_hours": 24
            },
            "liquidation": {
                "trading_pair": "SOL/USDT",  # Example, could be any volatile altcoin
                "min_profit_usd": 50,
                "liquidation_bonus_threshold": 0.01
            },
            "ml_strategy": {
                "trading_pair": "ADA/USDT",  # Example, ML strategy might focus on specific altcoins
                "prediction_threshold": 0.7,
                "model_confidence_min": 0.6
            },
            "conservative": {  # New placeholder strategy for volatile/extreme fear regimes
                "trading_pair": "BTC/USDT",
                "max_drawdown_per_trade": 0.01,
                "fixed_position_size_usd": 100
            },
            "manual_intervention_required": {  # Special 'strategy' for extreme fear, implies bot pauses
                "trading_pair": "N/A"
            }
        }
    }

    def __init__(self, config_name: str = 'default'):
        self.config = self._load_config(config_name)

    def _load_config(self, config_name: str) -> Dict[str, Any]:
        """Loads configuration from a JSON file, merging with default settings."""
        config_path = Path(__file__).parent / 'profiles' / f'{config_name}.json'

        loaded_config = {}
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                logger.info(f"Loaded configuration profile: {config_name}.json")
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from {config_path}: {e}")
            except Exception as e:
                logger.error(f"Error loading configuration file {config_path}: {e}")
        else:
            logger.warning(f"Configuration profile '{config_name}.json' not found. Using default settings.")
            if config_name != 'default':  # Avoid double warning if user explicitly asked for non-existent non-default
                logger.warning("Ensure the profile exists or use 'default'.")

        # Merge with default config. Loaded config values override defaults.
        merged_config = self.DEFAULT_CONFIG.copy()
        self._deep_update(merged_config, loaded_config)
        return merged_config

    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Recursively updates a dictionary with values from another dictionary."""
        for k, v in update_dict.items():
            if isinstance(v, dict) and k in base_dict and isinstance(base_dict[k], dict):
                base_dict[k] = self._deep_update(base_dict[k], v)
            else:
                base_dict[k] = v
        return base_dict

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves a setting value using dot notation (e.g., 'exchange.name')."""
        keys = key.split('.')
        current_level = self.config
        for k in keys:
            if isinstance(current_level, dict) and k in current_level:
                current_level = current_level[k]
            else:
                return default
        return current_level

    def set(self, key: str, value: Any):
        """Sets a setting value using dot notation."""
        keys = key.split('.')
        current_level = self.config
        for i, k in enumerate(keys):
            if i == len(keys) - 1:
                current_level[k] = value
            else:
                if not isinstance(current_level, dict):
                    logger.error(f"Cannot set value: Intermediate key '{k}' is not a dictionary.")
                    return
                if k not in current_level:
                    current_level[k] = {}
                current_level = current_level[k]
        logger.info(f"Setting '{key}' updated to '{value}'.")