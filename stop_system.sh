#!/bin/bash
# Trading Bot System Stop Script
# ===============================

echo "🛑 Stopping Trading Bot System..."
echo "=================================="

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Load PIDs if available
if [ -f ".system_pids" ]; then
    source .system_pids
fi

# Function to kill process safely
kill_process() {
    local PID=$1
    local NAME=$2
    
    if [ -n "$PID" ] && kill -0 $PID 2>/dev/null; then
        echo "Stopping $NAME (PID: $PID)..."
        kill $PID
        sleep 2
        
        # Force kill if still running
        if kill -0 $PID 2>/dev/null; then
            print_warning "Force killing $NAME..."
            kill -9 $PID 2>/dev/null || true
        fi
        
        print_status "$NAME stopped"
    else
        print_warning "$NAME was not running"
    fi
}

# Stop Trading Bot
if [ -n "$BOT_PID" ]; then
    kill_process $BOT_PID "Trading Bot"
fi

# Stop Dashboard
if [ -n "$DASHBOARD_PID" ]; then
    kill_process $DASHBOARD_PID "Dashboard"
else
    # Try to find and kill node processes
    echo "Looking for Dashboard processes..."
    pkill -f "npm start" 2>/dev/null && print_status "Dashboard processes stopped" || true
    pkill -f "react-scripts start" 2>/dev/null && print_status "React processes stopped" || true
fi

# Stop API Server
if [ -n "$API_PID" ]; then
    kill_process $API_PID "API Server"
else
    # Try to find and kill Flask processes
    echo "Looking for API processes..."
    pkill -f "python.*app.py" 2>/dev/null && print_status "API processes stopped" || true
fi

# Clean up ports if still occupied
echo "Checking for remaining processes on ports..."

# Port 5000 (API)
API_PORT_PID=$(lsof -ti:5000 2>/dev/null || true)
if [ -n "$API_PORT_PID" ]; then
    print_warning "Killing process on port 5000..."
    kill -9 $API_PORT_PID 2>/dev/null || true
fi

# Port 3001 (Dashboard)
DASH_PORT_PID=$(lsof -ti:3001 2>/dev/null || true)
if [ -n "$DASH_PORT_PID" ]; then
    print_warning "Killing process on port 3001..."
    kill -9 $DASH_PORT_PID 2>/dev/null || true
fi

# Clean up PID file
rm -f .system_pids

echo ""
print_status "Trading Bot System stopped completely"
echo ""
echo "📋 Final Status:"
echo "  • API Server: Stopped"
echo "  • Dashboard: Stopped"
echo "  • Trading Bot: Stopped"
echo "  • Logs preserved in ./logs/"
echo ""
echo "🚀 Ready to restart with: ./start_system.sh"