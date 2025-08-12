#!/bin/bash
# 🚀 DEPLOYMENT COMMANDS FOR JANICS FREEDOM FACTORY
# Run these commands on the server

echo "🚀 Starting Janics Freedom Factory Deployment..."

# 1. Navigate to project directory
cd ~/ML_altcoin_trading_bot 2>/dev/null || cd ~/altcoin_trading_bot 2>/dev/null || {
    echo "❌ Project directory not found. Cloning repository..."
    git clone https://github.com/Jimsonhahn/ML_altcoin_trading_bot.git
    cd ML_altcoin_trading_bot
}

# 2. Pull latest changes
echo "📥 Pulling latest changes from GitHub..."
git pull origin main

# 3. Check Python and create virtual environment if needed
echo "🐍 Setting up Python environment..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# 4. Activate virtual environment and install requirements
echo "📦 Installing Python packages..."
source venv/bin/activate
pip install -r requirements.txt

# 5. Create necessary directories
echo "📁 Creating data directories..."
mkdir -p data/trades data/portfolio data/intelligence data/ml data/strategy_performance data/ai data/logs

# 6. Setup environment variables
echo "⚙️ Setting up environment..."
if [ ! -f ".env.production" ]; then
    echo "Creating .env.production file..."
    cat > .env.production << 'EOL'
# Flask Configuration
FLASK_PORT=8080
FLASK_HOST=0.0.0.0
FLASK_ENV=production
FLASK_DEBUG=False

# API Configuration
API_PORT=8080
CORS_ORIGINS=http://localhost:3000,http://localhost:3001,http://localhost:3002

# Security
SECRET_KEY=your-secret-key-here-change-me
JWT_SECRET_KEY=your-jwt-secret-key-change-me

# Trading Configuration
TRADING_MODE=paper
EXCHANGE_NAME=binance

# Database (optional)
DATABASE_URL=sqlite:///trading_bot.db

# Logging
LOG_LEVEL=INFO
EOL
    echo "⚠️ Please edit .env.production with your actual configuration!"
fi

# 7. Test API startup
echo "🧪 Testing API startup..."
python -c "
import sys
sys.path.append('.')
from api.app import create_app
app, socketio = create_app()
print('✅ API imports successful')
"

# 8. Check if bot main script exists
if [ -f "main.py" ]; then
    echo "✅ Bot main script found: main.py"
elif [ -f "bot.py" ]; then
    echo "✅ Bot main script found: bot.py"
else
    echo "⚠️ No main bot script found. Controllers will use mock data."
fi

# 9. Start the API server
echo "🚀 Starting Janics Freedom Factory API..."
echo "API will be available at: http://$(curl -s ifconfig.me):8080"
echo ""
echo "Dashboard endpoints:"
echo "- Status: http://$(curl -s ifconfig.me):8080/api/v1/dashboard/status/header"
echo "- Summary: http://$(curl -s ifconfig.me):8080/api/v1/dashboard/dashboard/summary"
echo "- Health: http://$(curl -s ifconfig.me):8080/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Set environment variables and start
export FLASK_PORT=8080
export FLASK_HOST=0.0.0.0
export PYTHONPATH=/home/ubuntu/ML_altcoin_trading_bot:/home/ubuntu/altcoin_trading_bot:$PYTHONPATH

# Start the API server
python api/app.py