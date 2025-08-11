#!/usr/bin/env python3
"""
🚀 JANICS FREEDOM FACTORY - PRODUCTION CONFIGURATION
Production-ready configuration for server deployment
"""

import os
from typing import Dict, Any

class ProductionConfig:
    """Production environment configuration"""
    
    # Environment
    ENV = 'production'
    DEBUG = False
    TESTING = False
    
    # API Server Configuration
    API_HOST = '0.0.0.0'
    API_PORT = int(os.environ.get('API_PORT', 8080))
    
    # Security
    SECRET_KEY = os.environ.get('SECRET_KEY', os.urandom(32).hex())
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', os.urandom(32).hex())
    JWT_ACCESS_TOKEN_EXPIRES = 3600  # 1 hour
    
    # Database
    DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///./db/trading_bot.db')
    
    # CORS Configuration
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'https://your-domain.com').split(',')
    
    # Bot Configuration
    TRADING_MODE = os.environ.get('TRADING_MODE', 'paper')  # 'paper' or 'live'
    MAX_CONCURRENT_TRADES = 10
    RISK_PERCENTAGE = 2.0  # Max 2% risk per trade
    
    # Exchange Configuration
    EXCHANGE_API_KEY = os.environ.get('EXCHANGE_API_KEY', '')
    EXCHANGE_API_SECRET = os.environ.get('EXCHANGE_API_SECRET', '')
    EXCHANGE_NAME = os.environ.get('EXCHANGE_NAME', 'binance')
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = './logs/trading_bot.log'
    LOG_MAX_SIZE = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5
    
    # Performance
    WORKER_COUNT = int(os.environ.get('WORKER_COUNT', 4))
    WORKER_TIMEOUT = 120
    
    # Monitoring
    ENABLE_MONITORING = True
    MONITORING_INTERVAL = 60  # seconds
    
    # Rate Limiting
    RATE_LIMIT_ENABLED = True
    RATE_LIMIT_DEFAULT = "100 per hour"
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith('_') and not callable(getattr(cls, key))
        }


def get_production_config() -> Dict[str, Any]:
    """Get production configuration"""
    return ProductionConfig.to_dict()


# Environment variable validation
def validate_production_env():
    """Validate required environment variables for production"""
    required_vars = []
    
    if ProductionConfig.TRADING_MODE == 'live':
        required_vars.extend([
            'EXCHANGE_API_KEY',
            'EXCHANGE_API_SECRET',
            'SECRET_KEY',
            'JWT_SECRET_KEY'
        ])
    
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    print("✅ Production environment validated successfully")


if __name__ == "__main__":
    # Test configuration
    config = get_production_config()
    print("🚀 Production Configuration:")
    for key, value in config.items():
        if 'SECRET' in key or 'KEY' in key:
            print(f"  {key}: {'*' * 10}")
        else:
            print(f"  {key}: {value}")