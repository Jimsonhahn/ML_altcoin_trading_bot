#!/bin/bash
echo "🛑 Stopping Trading Bot System..."
echo "================================="

# Kill processes
echo "🔄 Stopping API Server..."
pkill -f "python -m api.app" && echo "   ✅ API Server stopped" || echo "   ❌ API Server not running"

echo "🔄 Stopping Dashboard..."
pkill -f "npm start" && echo "   ✅ Dashboard stopped" || echo "   ❌ Dashboard not running"

# Clean up PID file
rm -f pids.txt

echo ""
echo "✅ All services stopped"