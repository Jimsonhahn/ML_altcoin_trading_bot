"""
Migration script to move API keys from .env file to secure storage
Run this script once to migrate your existing API keys to the SecretManager
"""

import os
import logging
from pathlib import Path
from secret_manager import SecretManager, store_api_key

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_api_keys():
    """
    Migrate API keys from .env file to SecretManager
    """
    env_path = Path(__file__).parent.parent / '.env'
    
    if not env_path.exists():
        logger.warning("No .env file found. Nothing to migrate.")
        return
    
    logger.info("Starting API key migration...")
    
    # Read .env file
    env_vars = {}
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip().strip('"\'')
    except Exception as e:
        logger.error(f"Error reading .env file: {e}")
        return
    
    # Initialize SecretManager
    sm = SecretManager()
    
    # Migrate Binance API keys
    binance_api_key = env_vars.get('BINANCE_API_KEY')
    binance_api_secret = env_vars.get('BINANCE_API_SECRET')
    
    if binance_api_key and binance_api_secret:
        success = store_api_key('binance', binance_api_key, binance_api_secret)
        if success:
            logger.info("✅ Binance API keys migrated successfully")
        else:
            logger.error("❌ Failed to migrate Binance API keys")
    
    # Migrate Binance Testnet API keys
    binance_testnet_api_key = env_vars.get('BINANCE_TESTNET_API_KEY')
    binance_testnet_api_secret = env_vars.get('BINANCE_TESTNET_API_SECRET')
    
    if binance_testnet_api_key and binance_testnet_api_secret:
        success = store_api_key('binance_testnet', binance_testnet_api_key, binance_testnet_api_secret)
        if success:
            logger.info("✅ Binance Testnet API keys migrated successfully")
        else:
            logger.error("❌ Failed to migrate Binance Testnet API keys")
    
    # Migrate Telegram credentials
    telegram_token = env_vars.get('TELEGRAM_BOT_TOKEN')
    telegram_chat_id = env_vars.get('TELEGRAM_CHAT_ID')
    
    if telegram_token:
        success = sm.store_secret('telegram_bot_token', telegram_token)
        if success:
            logger.info("✅ Telegram bot token migrated successfully")
        else:
            logger.error("❌ Failed to migrate Telegram bot token")
    
    if telegram_chat_id:
        success = sm.store_secret('telegram_chat_id', telegram_chat_id)
        if success:
            logger.info("✅ Telegram chat ID migrated successfully")
        else:
            logger.error("❌ Failed to migrate Telegram chat ID")
    
    # Migrate CoinGecko API key
    coingecko_api_key = env_vars.get('COINGECKO_API_KEY')
    if coingecko_api_key:
        success = sm.store_secret('coingecko_api_key', coingecko_api_key)
        if success:
            logger.info("✅ CoinGecko API key migrated successfully")
        else:
            logger.error("❌ Failed to migrate CoinGecko API key")
    
    # Migrate Email credentials
    email_username = env_vars.get('EMAIL_USERNAME')
    email_password = env_vars.get('EMAIL_PASSWORD')
    
    if email_username:
        success = sm.store_secret('email_username', email_username)
        if success:
            logger.info("✅ Email username migrated successfully")
        else:
            logger.error("❌ Failed to migrate Email username")
    
    if email_password:
        success = sm.store_secret('email_password', email_password)
        if success:
            logger.info("✅ Email password migrated successfully")
        else:
            logger.error("❌ Failed to migrate Email password")
    
    logger.info("Migration completed!")
    logger.warning("⚠️  IMPORTANT: After verifying the migration worked, remove sensitive data from .env file")
    logger.warning("⚠️  Keep only non-sensitive configuration in .env file")

def verify_migration():
    """
    Verify that the migration was successful
    """
    logger.info("Verifying migration...")
    
    sm = SecretManager()
    
    # Check Binance credentials
    binance_key, binance_secret = sm.get_secret('binance_api_key'), sm.get_secret('binance_api_secret')
    if binance_key and binance_secret:
        logger.info("✅ Binance credentials verified")
    else:
        logger.warning("⚠️  Binance credentials not found")
    
    # Check Telegram credentials
    telegram_token = sm.get_secret('telegram_bot_token')
    if telegram_token:
        logger.info("✅ Telegram token verified")
    else:
        logger.warning("⚠️  Telegram token not found")
    
    # Check CoinGecko API key
    coingecko_key = sm.get_secret('coingecko_api_key')
    if coingecko_key:
        logger.info("✅ CoinGecko API key verified")
    else:
        logger.warning("⚠️  CoinGecko API key not found")
    
    logger.info("Verification completed!")

if __name__ == "__main__":
    migrate_api_keys()
    verify_migration()