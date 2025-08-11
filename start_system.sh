#!/bin/bash
# Trading Bot System Startup Script
# ==================================

set -e  # Exit on any error

echo "🚀 Starting Trading Bot System..."
echo "=================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "main.py" ] || [ ! -d "api" ] || [ ! -d "dashboard" ]; then
    print_error "Must be run from the trading bot root directory!"
    exit 1
fi

# Check Python environment
print_info "Checking Python environment..."
if [ ! -d ".venv" ]; then
    print_error "Virtual environment not found!"
    print_info "Run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate
print_status "Virtual environment activated"

# Check required packages
print_info "Checking required packages..."
python -c "import flask, ccxt, pandas, numpy" 2>/dev/null || {
    print_error "Missing required packages!"
    print_info "Run: pip install -r requirements.txt"
    exit 1
}
print_status "Required packages available"

# Initialize database
print_info "Initializing database..."
python -c "from db.models import TradingDatabase; db = TradingDatabase(); print('Database ready')" || {
    print_error "Database initialization failed!"
    exit 1
}
print_status "Database initialized"

# Create logs directory
mkdir -p logs
print_status "Logs directory ready"

# Start API Server
print_info "Starting API Server..."
cd api
python app.py > ../logs/api.log 2>&1 &
API_PID=$!
cd ..

# Wait for API to start
sleep 3

# Check API health
print_info "Checking API health..."
if curl -s http://localhost:5000/health > /dev/null; then
    print_status "API Server running (PID: $API_PID)"
else
    print_error "API failed to start!"
    kill $API_PID 2>/dev/null || true
    cat logs/api.log
    exit 1
fi

# Start Dashboard
print_info "Starting Dashboard..."
cd dashboard

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    print_warning "Installing dashboard dependencies..."
    npm install
fi

# Start dashboard on port 3001
PORT=3001 npm start > ../logs/dashboard.log 2>&1 &
DASHBOARD_PID=$!
cd ..

# Wait for dashboard to start
sleep 5

# Check dashboard
print_info "Checking Dashboard..."
if curl -s -I http://localhost:3001 | grep -q "200 OK"; then
    print_status "Dashboard running (PID: $DASHBOARD_PID)"
else
    print_warning "Dashboard may still be starting..."
fi

# Test API authentication
print_info "Testing API authentication..."
AUTH_TEST=$(curl -s -X POST http://localhost:5000/auth/login \
    -H "Content-Type: application/json" \
    -d '{"username": "admin", "password": "TradingBot2024"}')

if echo "$AUTH_TEST" | grep -q "access_token"; then
    print_status "API authentication working"
else
    print_warning "API authentication may have issues"
fi

# Start Trading Bot in Paper Mode (optional)
if [ "$1" = "--start-bot" ]; then
    print_info "Starting Trading Bot (Paper Mode)..."
    python main.py --mode paper --strategy momentum --symbol BTC/USDT > logs/bot.log 2>&1 &
    BOT_PID=$!
    print_status "Trading Bot started (PID: $BOT_PID)"
else
    BOT_PID=""
fi

echo ""
echo "🎉 System Successfully Started!"
echo "================================"
print_status "API Server: http://localhost:5000 (PID: $API_PID)"
print_status "Dashboard: http://localhost:3001 (PID: $DASHBOARD_PID)"

if [ -n "$BOT_PID" ]; then
    print_status "Trading Bot: Paper Mode Active (PID: $BOT_PID)"
fi

print_info "Login credentials: admin / TradingBot2024"
echo ""
print_info "Logs are in the ./logs/ directory"
print_info "Stop system with: ./stop_system.sh"

# Save PIDs for stop script
echo "export API_PID=$API_PID" > .system_pids
echo "export DASHBOARD_PID=$DASHBOARD_PID" >> .system_pids
[ -n "$BOT_PID" ] && echo "export BOT_PID=$BOT_PID" >> .system_pids

echo ""
print_warning "System is running in the background"
print_warning "Use 'tail -f logs/api.log' to monitor API logs"
print_warning "Use 'tail -f logs/dashboard.log' to monitor Dashboard logs"
[ -n "$BOT_PID" ] && print_warning "Use 'tail -f logs/bot.log' to monitor Bot logs"

echo ""
print_info "Ready for trading! 🚀"