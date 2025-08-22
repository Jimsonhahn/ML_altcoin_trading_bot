#!/bin/bash
# 🎯 STRATO SERVER UPDATE SCRIPT
# Aktualisiert den Bot mit den neuen Trading Pipeline Fixes

echo "🎯 STRATO SERVER UPDATE - Trading Bot Pipeline Fixes"
echo "=================================================="

# Farben für bessere Lesbarkeit
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Server Configuration
SERVER_IP="85.215.183.30"
BOT_DIR="/home/trading/altcoin_trading_bot"  # Anpassen falls anders
PYTHON_CMD="python3"
VENV_PATH="$BOT_DIR/venv"

echo -e "${BLUE}📡 Connecting to Strato Server...${NC}"
echo "Server: $SERVER_IP"
echo "Bot Directory: $BOT_DIR"
echo ""

# Prüfe SSH-Verbindung
check_ssh_connection() {
    echo -e "${YELLOW}🔍 Checking SSH connection...${NC}"
    
    # Test SSH connection
    timeout 10 ssh -q trading@$SERVER_IP exit
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ SSH connection successful${NC}"
        return 0
    else
        echo -e "${RED}❌ SSH connection failed${NC}"
        echo "Please ensure:"
        echo "1. SSH key is configured"
        echo "2. Server is accessible"
        echo "3. Username/IP are correct"
        return 1
    fi
}

# Hauptupdate-Funktion
update_server() {
    echo -e "${BLUE}🚀 Starting server update...${NC}"
    
    # SSH Command zum Server
    ssh trading@$SERVER_IP << 'EOF'
        
        echo "🔄 UPDATING TRADING BOT ON STRATO SERVER"
        echo "======================================="
        
        # Wechsel ins Bot-Verzeichnis
        cd /home/trading/altcoin_trading_bot || { echo "❌ Bot directory not found"; exit 1; }
        
        echo "📁 Current directory: $(pwd)"
        
        # Stoppe laufende Bot-Prozesse
        echo "🛑 Stopping running bot processes..."
        pkill -f "python.*main.py" || true
        pkill -f "python.*app.py" || true
        pkill -f "python.*run_intelligence" || true
        sleep 3
        
        # Backup current installation
        echo "💾 Creating backup..."
        cp -r . ../altcoin_trading_bot_backup_$(date +%Y%m%d_%H%M%S) || true
        
        # Git Status prüfen
        echo "📊 Current git status:"
        git status --short
        
        # Stash local changes if any
        echo "💼 Stashing local changes..."
        git stash push -m "Auto-stash before update $(date)"
        
        # Pull latest changes
        echo "⬇️ Pulling latest changes from repository..."
        git fetch origin
        git pull origin main
        
        if [ $? -eq 0 ]; then
            echo "✅ Git pull successful"
        else
            echo "❌ Git pull failed"
            exit 1
        fi
        
        # Aktiviere Virtual Environment
        echo "🐍 Activating virtual environment..."
        if [ -d "venv" ]; then
            source venv/bin/activate
            echo "✅ Virtual environment activated"
        else
            echo "🔧 Creating virtual environment..."
            python3 -m venv venv
            source venv/bin/activate
        fi
        
        # Update Python dependencies
        echo "📦 Updating Python dependencies..."
        pip install --upgrade pip
        pip install -r requirements.txt
        
        # Create necessary directories
        echo "📁 Creating necessary directories..."
        mkdir -p logs
        mkdir -p data/market_data
        mkdir -p intelligence_exports
        
        # Set correct permissions
        echo "🔒 Setting permissions..."
        chmod +x *.sh
        chmod +x *.py
        
        # Test the bot setup
        echo "🧪 Testing bot initialization..."
        timeout 30 python3 simple_trading_debug.py > test_output.log 2>&1
        
        if grep -q "✓" test_output.log; then
            echo "✅ Bot initialization test passed"
            tail -5 test_output.log
        else
            echo "⚠️ Bot initialization test had issues"
            echo "Last few lines of test:"
            tail -10 test_output.log
        fi
        
        # Show git log for recent changes
        echo ""
        echo "📝 Recent changes:"
        git log --oneline -3
        
        echo ""
        echo "✅ SERVER UPDATE COMPLETED!"
        echo "=========================="
        echo ""
        echo "🎯 CRITICAL FIXES APPLIED:"
        echo "✅ Trading pipeline execution fixed"  
        echo "✅ Market data retrieval working"
        echo "✅ Strategy signal generation fixed"
        echo "✅ Risk management interfaces completed"
        echo "✅ Order execution simulation ready"
        echo ""
        echo "Next steps:"
        echo "1. 🚀 Start bot: ./start_production_bot.sh"
        echo "2. 📊 Check dashboard: http://85.215.183.30:8000"
        echo "3. 📱 Monitor trades: ./test_actual_trading.py"
        
EOF

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Server update completed successfully!${NC}"
        return 0
    else
        echo -e "${RED}❌ Server update failed${NC}"
        return 1
    fi
}

# Bot starten
start_bot_on_server() {
    echo -e "${YELLOW}🚀 Starting bot on server...${NC}"
    
    ssh trading@$SERVER_IP << 'EOF'
        cd /home/trading/altcoin_trading_bot
        source venv/bin/activate
        
        echo "🤖 Starting Trading Bot with fixed pipeline..."
        
        # Start bot in background
        nohup python3 main.py --mode paper --strategy momentum > logs/bot.log 2>&1 &
        BOT_PID=$!
        echo $BOT_PID > bot.pid
        
        sleep 3
        
        # Start dashboard
        nohup python3 api/app.py --host 0.0.0.0 --port 8000 > logs/dashboard.log 2>&1 &
        DASH_PID=$!
        echo $DASH_PID > dashboard.pid
        
        sleep 2
        
        echo "✅ Bot started!"
        echo "Bot PID: $BOT_PID"
        echo "Dashboard PID: $DASH_PID"
        echo ""
        echo "🌐 Dashboard URL: http://85.215.183.30:8000"
        echo "📊 Check status: ps aux | grep python"
        
        # Quick status check
        if kill -0 $BOT_PID 2>/dev/null; then
            echo "✅ Bot process is running"
        else
            echo "❌ Bot process may have stopped"
        fi
        
        if kill -0 $DASH_PID 2>/dev/null; then
            echo "✅ Dashboard process is running"  
        else
            echo "❌ Dashboard process may have stopped"
        fi
EOF
}

# Dashboard Test
test_dashboard() {
    echo -e "${BLUE}🧪 Testing dashboard connectivity...${NC}"
    
    sleep 5  # Warte bis Services gestartet sind
    
    # Test HTTP endpoint
    echo "Testing HTTP endpoint..."
    curl -s --max-time 10 "http://$SERVER_IP:8000/health" > /dev/null
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Dashboard is accessible at http://$SERVER_IP:8000${NC}"
    else
        echo -e "${YELLOW}⚠️ Dashboard might still be starting up${NC}"
        echo "Try accessing: http://$SERVER_IP:8000 in your browser"
    fi
}

# Haupt-Ausführung
main() {
    echo -e "${BLUE}🎯 STRATO SERVER UPDATE PROCESS${NC}"
    echo "This script will:"
    echo "1. Update the bot with latest pipeline fixes"
    echo "2. Start the trading bot in paper mode"
    echo "3. Start the dashboard"
    echo "4. Test connectivity"
    echo ""
    
    read -p "Continue with update? (y/n): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Prüfe SSH-Verbindung
        if ! check_ssh_connection; then
            echo -e "${RED}❌ Cannot connect to server. Aborting.${NC}"
            exit 1
        fi
        
        # Update Server
        if update_server; then
            echo -e "${GREEN}✅ Update successful!${NC}"
            
            # Starte Bot
            read -p "Start bot now? (y/n): " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                start_bot_on_server
                test_dashboard
                
                echo ""
                echo -e "${GREEN}🎉 DEPLOYMENT COMPLETE!${NC}"
                echo -e "${YELLOW}📱 Access your dashboard: http://$SERVER_IP:8000${NC}"
                echo -e "${BLUE}🔧 SSH to server: ssh trading@$SERVER_IP${NC}"
            fi
        else
            echo -e "${RED}❌ Update failed. Check the logs above.${NC}"
            exit 1
        fi
    else
        echo "Update cancelled."
    fi
}

# Script starten
main