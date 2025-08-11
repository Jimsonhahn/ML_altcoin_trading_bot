#!/bin/bash
# 🚀 Start Simple Trading Bot
# Für Mac Development und Windows Server Deployment

echo "🚀 Starting Bulletproof Simple Trading Bot..."
echo "============================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv_simple" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv_simple
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv_simple/bin/activate

# Install requirements
echo "📥 Installing requirements..."
pip install --upgrade pip
pip install -r requirements_simple.txt

# Create data directories
echo "📁 Creating data directories..."
mkdir -p sync_data
mkdir -p logs

# Set environment variables
export FLASK_APP=simple_bulletproof_trading_bot.py
export FLASK_ENV=development

# Start the bot
echo "🚀 Starting Trading Bot..."
echo "Dashboard will be available at: http://localhost:5000"
echo "Press Ctrl+C to stop"
echo ""

python3 simple_bulletproof_trading_bot.py