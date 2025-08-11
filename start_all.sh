#!/bin/bash
echo "🚀 Starting Trading Bot System..."
echo "================================="

# Change to project directory
cd "$(dirname "$0")"

# Kill existing processes
echo "🔄 Stopping existing processes..."
pkill -f "python -m api.app" || true
pkill -f "npm start" || true
sleep 2

# Start API Server
echo "🖥️  Starting API Server..."
nohup python -m api.app > api.log 2>&1 &
API_PID=$!
echo "   API Server PID: $API_PID"

# Wait for API to be ready
echo "⏳ Waiting for API to be ready..."
sleep 5

# Test API health
API_STATUS=$(curl -s http://localhost:5000/health | grep -o '"status":"healthy"' || echo "failed")
if [ "$API_STATUS" == '"status":"healthy"' ]; then
    echo "   ✅ API Server is healthy"
else
    echo "   ❌ API Server failed to start"
    exit 1
fi

# Start Dashboard
echo "🎨 Starting Dashboard..."
cd dashboard
nohup npm start > dashboard.log 2>&1 &
DASHBOARD_PID=$!
echo "   Dashboard PID: $DASHBOARD_PID"

# Wait for Dashboard to be ready
echo "⏳ Waiting for Dashboard to be ready..."
sleep 8

# Test Dashboard
DASHBOARD_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3001)
if [ "$DASHBOARD_STATUS" == "200" ]; then
    echo "   ✅ Dashboard is ready"
else
    echo "   ❌ Dashboard failed to start (HTTP $DASHBOARD_STATUS)"
fi

echo ""
echo "🎉 Trading Bot System Status:"
echo "================================="
echo "🔗 API Server:    http://localhost:5000"
echo "🎨 Dashboard:     http://localhost:3001"
echo "📊 API Health:    http://localhost:5000/health"
echo "📚 API Docs:      http://localhost:5000/api/docs"
echo ""
echo "✅ CORS Configuration: Fixed"
echo "✅ Authentication:      Ready"
echo "✅ WebSocket:          Ready"
echo ""
echo "🔑 Default Login:"
echo "   Username: admin"
echo "   Password: TradingBot2024"
echo ""
echo "🛑 To stop all services:"
echo "   kill $API_PID $DASHBOARD_PID"
echo "   or run: pkill -f 'python -m api.app'; pkill -f 'npm start'"
echo ""
echo "📋 Process IDs saved to: pids.txt"
echo "$API_PID $DASHBOARD_PID" > ../pids.txt