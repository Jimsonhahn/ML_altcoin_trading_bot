#!/bin/bash
# 🚀 JANICS FREEDOM FACTORY - GIT DEPLOYMENT
# Deploy via Git - die professionelle Lösung

echo "🚀 JANICS FREEDOM FACTORY - GIT DEPLOYMENT"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if git repo exists
if [ ! -d .git ]; then
    echo -e "${RED}❌ Kein Git Repository gefunden!${NC}"
    echo "Initialisiere Git..."
    git init
    git add .
    git commit -m "Initial commit - Janics Freedom Factory Trading Bot"
fi

echo -e "${YELLOW}📝 Git Deployment Setup${NC}"
echo ""
echo "1️⃣  Füge dein Git Remote Repository hinzu:"
echo "   git remote add origin git@github.com:USERNAME/altcoin_trading_bot.git"
echo "   git push -u origin main"
echo ""
echo "2️⃣  Auf dem Server (einmalig):"
echo "   git clone git@github.com:USERNAME/altcoin_trading_bot.git"
echo "   cd altcoin_trading_bot"
echo "   ./setup_production.sh"
echo ""
echo "3️⃣  Für Updates (auf dem Server):"
echo "   git pull origin main"
echo "   sudo systemctl restart janics_bot"
echo ""

# Create production setup script
cat > setup_production.sh << 'EOF'
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
EOF

chmod +x setup_production.sh

# Create update script
cat > update_production.sh << 'EOF'
#!/bin/bash
# 🔄 Update Script (run on server)

echo "🔄 Updating Janics Freedom Factory..."

# Pull latest changes
git pull origin main

# Activate virtual environment
source venv/bin/activate

# Update dependencies
pip install -r requirements.txt

# Restart service
sudo systemctl restart janics_bot

# Show status
sudo systemctl status janics_bot --no-pager

echo "✅ Update complete!"
EOF

chmod +x update_production.sh

# Create .gitignore if not exists
if [ ! -f .gitignore ]; then
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Logs
logs/
*.log

# Database
*.db
*.sqlite
*.sqlite3

# Environment
.env
.env.local
.env.production

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Trading specific
data/
backups/
exports/
intelligence_exports/

# Deployment
deployment_package/
*.tar.gz
EOF
fi

echo -e "${GREEN}✅ Git Deployment vorbereitet!${NC}"
echo ""
echo "📋 GITHUB DEPLOYMENT WORKFLOW:"
echo "=============================="
echo ""
echo "1️⃣  Erstelle ein GitHub Repository:"
echo "   - Gehe zu https://github.com/new"
echo "   - Name: altcoin_trading_bot"
echo "   - Private repository (wichtig!)"
echo ""
echo "2️⃣  Verbinde lokales Repo mit GitHub:"
echo "   git remote add origin git@github.com:DEIN-USERNAME/altcoin_trading_bot.git"
echo "   git push -u origin main"
echo ""
echo "3️⃣  Auf dem Server:"
echo "   git clone git@github.com:DEIN-USERNAME/altcoin_trading_bot.git"
echo "   cd altcoin_trading_bot"
echo "   ./setup_production.sh"
echo ""
echo "4️⃣  Für spätere Updates:"
echo "   Lokal:  git push origin main"
echo "   Server: cd altcoin_trading_bot && ./update_production.sh"
echo ""

# Offer to create GitHub repo
echo -e "${YELLOW}💡 Tipp: Nutze GitHub Deploy Keys für sicheren Server-Zugriff${NC}"
echo "   Auf dem Server: ssh-keygen -t ed25519 -C 'server-deploy-key'"
echo "   Füge den Public Key zu GitHub hinzu: Settings > Deploy keys"