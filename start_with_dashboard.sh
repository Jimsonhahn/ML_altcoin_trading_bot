#!/bin/bash
# Start orchestrator with dashboard integration

echo "🚀 Starting Altcoin Trading Bot with Dashboard Integration"
echo "========================================================"

# Set default mode
MODE=${1:-paper}
echo "📈 Trading Mode: $MODE"

# Export environment variables
export ORCHESTRATOR_MODE=$MODE
export ORCHESTRATOR_CAPITAL=10000
export FLASK_PORT=5000

# Function to cleanup on exit
cleanup() {
    echo -e "\n🛑 Stopping all services..."
    kill $API_PID $WORKER_PID $DASHBOARD_PID 2>/dev/null
    echo "✅ All services stopped"
    exit 0
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

# Start API server
echo -e "\n1️⃣ Starting API Server..."
python api/app.py &
API_PID=$!
echo "   API Server PID: $API_PID"

# Wait for API to start
sleep 3

# Start orchestrator background worker
echo -e "\n2️⃣ Starting Orchestrator Worker..."
python orchestrator_background_worker.py $MODE &
WORKER_PID=$!
echo "   Orchestrator Worker PID: $WORKER_PID"

# Wait for worker to initialize
sleep 2

# Start dashboard (if exists)
if [ -d "dashboard" ] && [ -f "dashboard/package.json" ]; then
    echo -e "\n3️⃣ Starting Dashboard..."
    cd dashboard
    npm start &
    DASHBOARD_PID=$!
    cd ..
    echo "   Dashboard PID: $DASHBOARD_PID"
else
    echo -e "\n⚠️  Dashboard not found. Skipping dashboard start."
fi

echo -e "\n✅ All services started!"
echo "========================================================"
echo "📊 Dashboard: http://localhost:3002"
echo "🔌 API: http://localhost:5000"
echo "📚 API Docs: http://localhost:5000/api/docs"
echo ""
echo "💡 The orchestrator is running in $MODE mode"
echo "   - Discovering strategies automatically"
echo "   - Managing portfolio allocation"
echo "   - Monitoring strategy health"
echo "   - Running A/B tests"
echo ""
echo "Press Ctrl+C to stop all services"
echo "========================================================"

# Keep script running
while true; do
    sleep 1
done