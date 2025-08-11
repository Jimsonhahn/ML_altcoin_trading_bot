#!/bin/bash

# Docker entrypoint script for Altcoin Trading Bot
set -e

echo "🚀 Starting Altcoin Trading Bot Docker Container..."

# Environment validation
if [ -z "$TRADING_MODE" ]; then
    echo "⚠️  TRADING_MODE not set, defaulting to 'paper'"
    export TRADING_MODE="paper"
fi

if [ -z "$LOG_LEVEL" ]; then
    echo "⚠️  LOG_LEVEL not set, defaulting to 'INFO'"
    export LOG_LEVEL="INFO"
fi

echo "📊 Trading Mode: $TRADING_MODE"
echo "📝 Log Level: $LOG_LEVEL"
echo "🌍 Environment: ${NODE_ENV:-production}"

# Wait for dependencies
echo "⏳ Waiting for dependencies..."

# Wait for Redis
if [ ! -z "$REDIS_HOST" ]; then
    echo "Waiting for Redis at $REDIS_HOST:${REDIS_PORT:-6379}..."
    while ! nc -z $REDIS_HOST ${REDIS_PORT:-6379}; do
        sleep 1
    done
    echo "✅ Redis is ready"
fi

# Wait for PostgreSQL
if [ ! -z "$DB_HOST" ]; then
    echo "Waiting for PostgreSQL at $DB_HOST:${DB_PORT:-5432}..."
    while ! nc -z $DB_HOST ${DB_PORT:-5432}; do
        sleep 1
    done
    echo "✅ PostgreSQL is ready"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p /app/data/logs
mkdir -p /app/data/backtest_results
mkdir -p /app/data/ml_models
mkdir -p /app/data/market_data
mkdir -p /app/logs

# Set proper permissions
echo "🔐 Setting permissions..."
chown -R trader:trader /app/data
chown -R trader:trader /app/logs

# Run database migrations if needed
if [ "$TRADING_MODE" != "backtest" ]; then
    echo "🗄️  Running database migrations..."
    python scripts/init_db.py || echo "⚠️  Database migration failed, continuing..."
fi

# Validate configuration
echo "✅ Validating configuration..."
python -c "
import sys
sys.path.append('/app')
from config.settings import TradingConfig
try:
    config = TradingConfig()
    print('✅ Configuration validation passed')
except Exception as e:
    print(f'❌ Configuration validation failed: {e}')
    sys.exit(1)
" || exit 1

# Health check before starting
echo "🏥 Performing initial health check..."
python -c "
import sys
sys.path.append('/app')
try:
    # Import main modules to check for import errors
    from core.trading_bot import TradingBot
    from core.exchange import Exchange
    from utils.notifier import send_info
    print('✅ All imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ Health check failed: {e}')
    sys.exit(1)
" || exit 1

# Start the application based on mode
echo "🎯 Starting application in $TRADING_MODE mode..."

case "$TRADING_MODE" in
    "live")
        echo "💰 LIVE TRADING MODE - Real money at risk!"
        echo "⚠️  Make sure you have properly configured your API keys and risk settings"
        ;;
    "paper")
        echo "📋 PAPER TRADING MODE - Simulated trading"
        ;;
    "backtest")
        echo "📈 BACKTEST MODE - Historical data analysis"
        ;;
    *)
        echo "❌ Unknown trading mode: $TRADING_MODE"
        echo "Valid modes: live, paper, backtest"
        exit 1
        ;;
esac

# Send startup notification
python -c "
import sys
sys.path.append('/app')
try:
    from utils.notifier import send_info
    send_info(f'🚀 Trading Bot started in $TRADING_MODE mode')
    print('📱 Startup notification sent')
except Exception as e:
    print(f'⚠️  Could not send startup notification: {e}')
"

# Final pre-flight check
echo "✈️  Pre-flight check complete!"
echo "🚀 Launching Altcoin Trading Bot..."

# Execute the main command
exec "$@"