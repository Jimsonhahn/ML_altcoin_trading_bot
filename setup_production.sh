#!/bin/bash
# 🚀 Production Setup Script (run on server)

echo "🔧 Setting up Janics Freedom Factory on Production..."

# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv nginx git

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs data config

# Copy production config template
if [ ! -f .env.production ]; then
    cp .env.production.example .env.production
    echo "⚠️  Please edit .env.production with your settings!"
fi

# Create systemd service
sudo tee /etc/systemd/system/janics_bot.service > /dev/null << 'SERVICE'
[Unit]
Description=Janics Freedom Factory Trading Bot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PWD
Environment="PATH=$PWD/venv/bin"
ExecStart=$PWD/venv/bin/python api/app.py
Restart=always
RestartSec=10
EnvironmentFile=$PWD/.env.production

[Install]
WantedBy=multi-user.target
SERVICE

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable janics_bot

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env.production with your settings"
echo "2. sudo systemctl start janics_bot"
echo "3. Check status: sudo systemctl status janics_bot"
