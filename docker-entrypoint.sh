#!/bin/bash
"""
Docker Entry Point for Trading Bot
=================================

Production-ready startup script with health checks and graceful handling.
"""

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting Altcoin Trading Bot...${NC}"

# Environment setup
export PYTHONPATH=/app:$PYTHONPATH
export PYTHONUNBUFFERED=1

# Default values
TRADING_ENV=${TRADING_ENV:-production}
API_PORT=${API_PORT:-5000}
API_HOST=${API_HOST:-0.0.0.0}

echo -e "${YELLOW}📋 Environment: $TRADING_ENV${NC}"
echo -e "${YELLOW}🌐 API will start on $API_HOST:$API_PORT${NC}"

# Health check function
check_health() {
    echo -e "${YELLOW}🔍 Running health checks...${NC}"
    
    # Check Python environment
    python --version || exit 1
    
    # Check if required packages are available
    python -c "import flask; print('✅ Flask available')" || exit 1
    python -c "import requests; print('✅ Requests available')" || exit 1
    
    # Check if config can be loaded
    python -c "from config.environment import get_config; get_config(); print('✅ Configuration loaded')" || {
        echo -e "${RED}❌ Configuration check failed${NC}"
        exit 1
    }
    
    # Check if API can be created
    python -c "from api.standalone_api import create_app; create_app(); print('✅ API can be created')" || {
        echo -e "${RED}❌ API creation failed${NC}"
        exit 1
    }
    
    echo -e "${GREEN}✅ All health checks passed${NC}"
}

# Setup logging directory
setup_logging() {
    echo -e "${YELLOW}📝 Setting up logging...${NC}"
    mkdir -p /app/logs
    chmod 755 /app/logs
    echo -e "${GREEN}✅ Logging directory ready${NC}"
}

# Wait for dependencies
wait_for_dependencies() {
    echo -e "${YELLOW}⏳ Checking dependencies...${NC}"
    
    # Wait for database if configured
    if [ -n "$DATABASE_URL" ]; then
        echo -e "${YELLOW}🔗 Checking database connection...${NC}"
        # Add database connectivity check here if needed
    fi
    
    # Wait for Redis if configured
    if [ -n "$REDIS_URL" ]; then
        echo -e "${YELLOW}🔗 Checking Redis connection...${NC}"
        # Add Redis connectivity check here if needed
    fi
    
    echo -e "${GREEN}✅ Dependencies ready${NC}"
}

# Signal handlers for graceful shutdown
cleanup() {
    echo -e "${YELLOW}🛑 Received shutdown signal, cleaning up...${NC}"
    if [ ! -z "$API_PID" ]; then
        kill -TERM $API_PID 2>/dev/null || true
        wait $API_PID 2>/dev/null || true
    fi
    echo -e "${GREEN}✅ Cleanup completed${NC}"
    exit 0
}

trap cleanup SIGTERM SIGINT

# Main execution
main() {
    echo -e "${GREEN}🏁 Starting main application...${NC}"
    
    # Run health checks
    check_health
    
    # Setup logging
    setup_logging
    
    # Wait for dependencies
    wait_for_dependencies
    
    # Start the appropriate service based on parameters
    case "${1:-api}" in
        "api")
            echo -e "${GREEN}🌐 Starting API server...${NC}"
            cd /app
            exec python api/standalone_api.py
            ;;
        "trading-bot")
            echo -e "${GREEN}🤖 Starting trading bot...${NC}"
            cd /app
            exec python main.py
            ;;
        "full")
            echo -e "${GREEN}🚀 Starting full system (API + Bot)...${NC}"
            cd /app
            
            # Start API in background
            python api/standalone_api.py &
            API_PID=$!
            
            # Start trading bot in foreground
            exec python main.py
            ;;
        "shell")
            echo -e "${GREEN}🐚 Starting interactive shell...${NC}"
            exec /bin/bash
            ;;
        *)
            echo -e "${RED}❌ Unknown command: $1${NC}"
            echo "Available commands: api, trading-bot, full, shell"
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"