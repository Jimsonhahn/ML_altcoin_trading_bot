#!/usr/bin/env python3
# config/environment.py
"""
Environment-specific configuration management
Handles loading and validation of environment variables across dev/staging/production
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import warnings

logger = logging.getLogger(__name__)

class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class TradingMode(Enum):
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str
    port: int
    name: str
    user: str
    password: str
    
    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

@dataclass
class RedisConfig:
    """Redis configuration"""
    host: str
    port: int
    password: Optional[str] = None
    
    @property
    def url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/0"
        return f"redis://{self.host}:{self.port}/0"

@dataclass
class TradingConfig:
    """Trading-specific configuration"""
    initial_capital: float
    max_portfolio_risk: float
    max_position_size: float
    stop_loss_percentage: float
    take_profit_percentage: float
    default_strategy_profile: str
    rebalance_frequency_hours: int
    min_trade_amount: float

@dataclass
class SecurityConfig:
    """Security configuration"""
    jwt_secret: str
    api_rate_limit_per_minute: int
    api_rate_limit_burst: int
    session_timeout_hours: int
    max_login_attempts: int
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None

@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""
    enable_trade_notifications: bool
    enable_error_notifications: bool
    enable_performance_reports: bool
    enable_daily_summary: bool
    alert_on_drawdown_percent: float
    alert_on_profit_percent: float

class EnvironmentConfig:
    """
    Main environment configuration class
    Loads and validates all configuration based on current environment
    """
    
    def __init__(self):
        self.environment = self._detect_environment()
        self.trading_mode = self._get_trading_mode()
        
        # Load environment-specific settings
        self._load_environment_file()
        
        # Initialize configurations
        self.database = self._load_database_config()
        self.redis = self._load_redis_config()
        self.trading = self._load_trading_config()
        self.security = self._load_security_config()
        self.monitoring = self._load_monitoring_config()
        
        # Validate configuration
        self._validate_configuration()
        
        logger.info(f"Environment configuration loaded: {self.environment.value} ({self.trading_mode.value})")
    
    def _detect_environment(self) -> Environment:
        """Detect current environment from NODE_ENV"""
        env_str = os.getenv('NODE_ENV', 'development').lower()
        
        try:
            return Environment(env_str)
        except ValueError:
            logger.warning(f"Unknown environment '{env_str}', defaulting to development")
            return Environment.DEVELOPMENT
    
    def _get_trading_mode(self) -> TradingMode:
        """Get trading mode from environment"""
        mode_str = os.getenv('TRADING_MODE', 'paper').lower()
        
        try:
            return TradingMode(mode_str)
        except ValueError:
            logger.warning(f"Unknown trading mode '{mode_str}', defaulting to paper")
            return TradingMode.PAPER
    
    def _load_environment_file(self):
        """Load environment-specific .env file"""
        env_files = {
            Environment.DEVELOPMENT: '.env.dev',
            Environment.STAGING: '.env.staging',
            Environment.PRODUCTION: '.env.production'
        }
        
        env_file = env_files.get(self.environment)
        if env_file and os.path.exists(env_file):
            logger.info(f"Loading environment file: {env_file}")
            # Note: In production, you might want to use python-dotenv
            # from dotenv import load_dotenv
            # load_dotenv(env_file)
        else:
            logger.warning(f"Environment file {env_file} not found, using system environment variables")
    
    def _load_database_config(self) -> DatabaseConfig:
        """Load database configuration"""
        return DatabaseConfig(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', '5432')),
            name=os.getenv('DB_NAME', 'trading_bot'),
            user=os.getenv('DB_USER', 'trader'),
            password=os.getenv('DB_PASSWORD', 'password')
        )
    
    def _load_redis_config(self) -> RedisConfig:
        """Load Redis configuration"""
        return RedisConfig(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', '6379')),
            password=os.getenv('REDIS_PASSWORD')
        )
    
    def _load_trading_config(self) -> TradingConfig:
        """Load trading configuration"""
        return TradingConfig(
            initial_capital=float(os.getenv('INITIAL_CAPITAL', '100000')),
            max_portfolio_risk=float(os.getenv('MAX_PORTFOLIO_RISK', '0.20')),
            max_position_size=float(os.getenv('MAX_POSITION_SIZE', '0.10')),
            stop_loss_percentage=float(os.getenv('STOP_LOSS_PERCENTAGE', '0.05')),
            take_profit_percentage=float(os.getenv('TAKE_PROFIT_PERCENTAGE', '0.15')),
            default_strategy_profile=os.getenv('DEFAULT_STRATEGY_PROFILE', 'balanced'),
            rebalance_frequency_hours=int(os.getenv('REBALANCE_FREQUENCY_HOURS', '6')),
            min_trade_amount=float(os.getenv('MIN_TRADE_AMOUNT', '50'))
        )
    
    def _load_security_config(self) -> SecurityConfig:
        """Load security configuration"""
        return SecurityConfig(
            jwt_secret=os.getenv('JWT_SECRET', 'change_this_in_production'),
            api_rate_limit_per_minute=int(os.getenv('API_RATE_LIMIT_PER_MINUTE', '60')),
            api_rate_limit_burst=int(os.getenv('API_RATE_LIMIT_BURST', '10')),
            session_timeout_hours=int(os.getenv('SESSION_TIMEOUT_HOURS', '24')),
            max_login_attempts=int(os.getenv('MAX_LOGIN_ATTEMPTS', '5')),
            ssl_cert_path=os.getenv('SSL_CERT_PATH'),
            ssl_key_path=os.getenv('SSL_KEY_PATH')
        )
    
    def _load_monitoring_config(self) -> MonitoringConfig:
        """Load monitoring configuration"""
        return MonitoringConfig(
            enable_trade_notifications=os.getenv('ENABLE_TRADE_NOTIFICATIONS', 'true').lower() == 'true',
            enable_error_notifications=os.getenv('ENABLE_ERROR_NOTIFICATIONS', 'true').lower() == 'true',
            enable_performance_reports=os.getenv('ENABLE_PERFORMANCE_REPORTS', 'true').lower() == 'true',
            enable_daily_summary=os.getenv('ENABLE_DAILY_SUMMARY', 'true').lower() == 'true',
            alert_on_drawdown_percent=float(os.getenv('ALERT_ON_DRAWDOWN_PERCENT', '0.10')),
            alert_on_profit_percent=float(os.getenv('ALERT_ON_PROFIT_PERCENT', '0.05'))
        )
    
    def _validate_configuration(self):
        """Validate configuration for current environment"""
        errors = []
        warnings_list = []
        
        # Critical validations for production
        if self.environment == Environment.PRODUCTION:
            if self.trading_mode == TradingMode.LIVE:
                # Production live trading validations
                if os.getenv('BINANCE_API_KEY') in [None, '', 'REPLACE_WITH_REAL_BINANCE_API_KEY']:
                    errors.append("BINANCE_API_KEY must be set for production live trading")
                
                if os.getenv('BINANCE_SECRET_KEY') in [None, '', 'REPLACE_WITH_REAL_BINANCE_SECRET_KEY']:
                    errors.append("BINANCE_SECRET_KEY must be set for production live trading")
                
                if os.getenv('BINANCE_TESTNET', 'true').lower() == 'true':
                    errors.append("BINANCE_TESTNET must be 'false' for production live trading")
                
                if self.security.jwt_secret in ['change_this_in_production', 'dev_jwt_secret_key_not_for_production']:
                    errors.append("JWT_SECRET must be changed for production")
                
                if self.trading.initial_capital < 1000:
                    warnings_list.append("Initial capital seems low for production trading")
            
            # SSL validation for production
            if not self.security.ssl_cert_path or not self.security.ssl_key_path:
                warnings_list.append("SSL certificates not configured for production")
        
        # General validations
        if self.trading.max_portfolio_risk > 0.5:
            warnings_list.append(f"High portfolio risk setting: {self.trading.max_portfolio_risk:.1%}")
        
        if self.trading.max_position_size > 0.2:
            warnings_list.append(f"High position size setting: {self.trading.max_position_size:.1%}")
        
        # Environment-specific warnings
        if self.environment == Environment.DEVELOPMENT and self.trading_mode == TradingMode.LIVE:
            warnings_list.append("Live trading in development environment - consider using paper trading")
        
        # Log errors and warnings
        for error in errors:
            logger.error(f"Configuration error: {error}")
        
        for warning in warnings_list:
            logger.warning(f"Configuration warning: {warning}")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        if warnings_list:
            warnings.warn(f"Configuration warnings: {'; '.join(warnings_list)}")
    
    def get_api_keys(self) -> Dict[str, str]:
        """Get API keys safely"""
        return {
            'binance_api_key': os.getenv('BINANCE_API_KEY', ''),
            'binance_secret_key': os.getenv('BINANCE_SECRET_KEY', ''),
            'telegram_bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
            'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID', ''),
            'coingecko_api_key': os.getenv('COINGECKO_API_KEY', ''),
            'alpha_vantage_api_key': os.getenv('ALPHA_VANTAGE_API_KEY', '')
        }
    
    def is_testnet(self) -> bool:
        """Check if using testnet"""
        return os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
    
    def is_production(self) -> bool:
        """Check if in production environment"""
        return self.environment == Environment.PRODUCTION
    
    def is_live_trading(self) -> bool:
        """Check if live trading is enabled"""
        return self.trading_mode == TradingMode.LIVE
    
    def get_log_level(self) -> str:
        """Get appropriate log level for environment"""
        log_levels = {
            Environment.DEVELOPMENT: 'DEBUG',
            Environment.STAGING: 'INFO',
            Environment.PRODUCTION: 'INFO'
        }
        
        return os.getenv('LOG_LEVEL', log_levels[self.environment])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (without sensitive data)"""
        return {
            'environment': self.environment.value,
            'trading_mode': self.trading_mode.value,
            'database': {
                'host': self.database.host,
                'port': self.database.port,
                'name': self.database.name,
                'user': self.database.user
                # password excluded for security
            },
            'redis': {
                'host': self.redis.host,
                'port': self.redis.port
                # password excluded for security
            },
            'trading': {
                'initial_capital': self.trading.initial_capital,
                'max_portfolio_risk': self.trading.max_portfolio_risk,
                'max_position_size': self.trading.max_position_size,
                'stop_loss_percentage': self.trading.stop_loss_percentage,
                'take_profit_percentage': self.trading.take_profit_percentage,
                'default_strategy_profile': self.trading.default_strategy_profile,
                'rebalance_frequency_hours': self.trading.rebalance_frequency_hours,
                'min_trade_amount': self.trading.min_trade_amount
            },
            'security': {
                'api_rate_limit_per_minute': self.security.api_rate_limit_per_minute,
                'api_rate_limit_burst': self.security.api_rate_limit_burst,
                'session_timeout_hours': self.security.session_timeout_hours,
                'max_login_attempts': self.security.max_login_attempts,
                'ssl_enabled': bool(self.security.ssl_cert_path and self.security.ssl_key_path)
                # jwt_secret excluded for security
            },
            'monitoring': {
                'enable_trade_notifications': self.monitoring.enable_trade_notifications,
                'enable_error_notifications': self.monitoring.enable_error_notifications,
                'enable_performance_reports': self.monitoring.enable_performance_reports,
                'enable_daily_summary': self.monitoring.enable_daily_summary,
                'alert_on_drawdown_percent': self.monitoring.alert_on_drawdown_percent,
                'alert_on_profit_percent': self.monitoring.alert_on_profit_percent
            },
            'is_testnet': self.is_testnet(),
            'is_production': self.is_production(),
            'is_live_trading': self.is_live_trading(),
            'log_level': self.get_log_level()
        }

# Global configuration instance
config = EnvironmentConfig()

def get_config() -> EnvironmentConfig:
    """Get global configuration instance"""
    return config

def reload_config():
    """Reload configuration (useful for testing)"""
    global config
    config = EnvironmentConfig()

if __name__ == "__main__":
    # Test configuration loading
    print("Testing environment configuration...")
    
    test_config = EnvironmentConfig()
    print(f"Environment: {test_config.environment.value}")
    print(f"Trading Mode: {test_config.trading_mode.value}")
    print(f"Database URL: {test_config.database.url}")
    print(f"Redis URL: {test_config.redis.url}")
    print(f"Initial Capital: ${test_config.trading.initial_capital:,.0f}")
    print(f"Is Production: {test_config.is_production()}")
    print(f"Is Live Trading: {test_config.is_live_trading()}")
    print(f"Log Level: {test_config.get_log_level()}")
    
    print("\nConfiguration validation completed successfully!")