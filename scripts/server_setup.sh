#!/bin/bash
# Server Setup Script for Trading Bot
# Run this on your server (85.215.183.30) after SSH connection

set -e

echo "🚀 Setting up Trading Bot on Server..."
echo "=================================================="

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python 3.11 and pip
echo "🐍 Installing Python 3.11..."
sudo apt install -y python3.11 python3.11-pip python3.11-venv python3.11-dev
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install Node.js 18 for React dashboard
echo "📦 Installing Node.js 18..."
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt install -y git curl wget nginx supervisor postgresql postgresql-contrib redis-server
sudo apt install -y build-essential pkg-config libffi-dev libssl-dev

# Create project directory
echo "📁 Creating project directory..."
sudo mkdir -p /opt/trading-bot
sudo chown $USER:$USER /opt/trading-bot
cd /opt/trading-bot

# Clone project (if using git) or prepare for file upload
echo "📂 Preparing project structure..."
mkdir -p {api,dashboard,core,strategies,data,logs,config,scripts}
mkdir -p data/{market_data,ml_models,ml_analysis,backtest_results}
mkdir -p logs/{api,dashboard,trading,system}

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Core trading dependencies
pip install fastapi uvicorn[standard] websockets
pip install pandas numpy scipy scikit-learn lightgbm
pip install ccxt python-binance requests aiohttp asyncio-throttle
pip install pydantic sqlalchemy asyncpg aiosqlite
pip install python-telegram-bot python-dotenv pyyaml
pip install plotly matplotlib seaborn
pip install pytest pytest-asyncio black flake8

# Create basic directory structure
cat > requirements.txt << 'EOL'
# Core FastAPI and async
fastapi==0.104.1
uvicorn[standard]==0.24.0
websockets==12.0
pydantic==2.5.0

# Data processing
pandas==2.1.4
numpy==1.25.2
scipy==1.11.4

# Machine Learning
scikit-learn==1.3.2
lightgbm==4.1.0

# Trading APIs
ccxt==4.1.90
python-binance==1.0.19
requests==2.31.0
aiohttp==3.9.1

# Database
sqlalchemy==2.0.23
asyncpg==0.29.0
aiosqlite==0.19.0

# Utilities
python-telegram-bot==20.7
python-dotenv==1.0.0
pyyaml==6.0.1
plotly==5.17.0
matplotlib==3.8.2
seaborn==0.13.0

# Development
pytest==7.4.3
pytest-asyncio==0.21.1
black==23.11.0
flake8==6.1.0
EOL

echo "✅ Server basic setup completed!"
echo "📝 Next steps:"
echo "   1. Upload your project files to /opt/trading-bot"
echo "   2. Run 'source venv/bin/activate' to activate virtual environment"
echo "   3. Run 'pip install -r requirements.txt' to install dependencies"
echo "   4. Configure environment variables in .env file"