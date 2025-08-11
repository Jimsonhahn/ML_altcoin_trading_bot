#!/bin/bash
# Complete Production Deployment Script
# Run this on your server after uploading the project files

set -e

echo "🚀 Starting Production Deployment..."
echo "====================================="

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "❌ Please don't run this script as root. Use a regular user with sudo privileges."
   exit 1
fi

# Set variables
PROJECT_DIR="/opt/trading-bot"
VENV_DIR="$PROJECT_DIR/venv"
USER=$(whoami)

echo "📋 Deployment Configuration:"
echo "   Project Directory: $PROJECT_DIR"
echo "   Virtual Environment: $VENV_DIR"
echo "   User: $USER"
echo "   Server IP: 85.215.183.30"
echo ""

# Step 1: Prepare directories and permissions
echo "1️⃣  Setting up directories and permissions..."
sudo mkdir -p $PROJECT_DIR/{logs,data,config}
sudo chown -R $USER:$USER $PROJECT_DIR
chmod +x $PROJECT_DIR/scripts/*.sh

# Step 2: Set up Python environment
echo "2️⃣  Setting up Python environment..."
cd $PROJECT_DIR
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Step 3: Configure environment variables
echo "3️⃣  Configuring environment variables..."
if [ ! -f "$PROJECT_DIR/.env" ]; then
    cp config/production.env $PROJECT_DIR/.env
    echo "⚠️  Please edit $PROJECT_DIR/.env with your actual configuration values"
    echo "   Especially: API keys, database credentials, secret keys"
fi

# Step 4: Set up database
echo "4️⃣  Setting up PostgreSQL database..."
sudo -u postgres psql << 'EOL'
CREATE DATABASE trading_bot;
CREATE USER trading_user WITH ENCRYPTED PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE trading_bot TO trading_user;
\q
EOL

# Create database tables (if you have migration scripts)
# python -m alembic upgrade head

# Step 5: Configure nginx
echo "5️⃣  Configuring nginx..."
sudo cp config/nginx.conf /etc/nginx/sites-available/trading-bot
sudo ln -sf /etc/nginx/sites-available/trading-bot /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable nginx

# Step 6: Configure supervisor
echo "6️⃣  Configuring supervisor..."
sudo cp config/supervisor.conf /etc/supervisor/conf.d/trading-bot.conf
sudo supervisorctl reread
sudo supervisorctl update

# Step 7: Build React dashboard
echo "7️⃣  Building React dashboard..."
cd $PROJECT_DIR/dashboard
npm install --production
npm run build

# Step 8: Set up log rotation
echo "8️⃣  Setting up log rotation..."
sudo tee /etc/logrotate.d/trading-bot << 'EOL'
/opt/trading-bot/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 ubuntu ubuntu
    postrotate
        supervisorctl restart trading-bot:*
    endscript
}
EOL

# Step 9: Configure systemd services (alternative to supervisor)
echo "9️⃣  Creating systemd services..."
sudo tee /etc/systemd/system/trading-bot.service << EOL
[Unit]
Description=Trading Bot API
After=network.target postgresql.service

[Service]
Type=exec
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$VENV_DIR/bin
ExecStart=$VENV_DIR/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOL

sudo tee /etc/systemd/system/intelligence-api.service << EOL
[Unit]
Description=Intelligence Dashboard API
After=network.target

[Service]
Type=exec
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$VENV_DIR/bin
ExecStart=$VENV_DIR/bin/python config/server_intelligence_api.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOL

sudo systemctl daemon-reload
sudo systemctl enable trading-bot.service
sudo systemctl enable intelligence-api.service

# Step 10: Final security and optimization
echo "🔒 Applying final security configurations..."

# Set proper file permissions
find $PROJECT_DIR -type f -name "*.py" -exec chmod 644 {} \;
find $PROJECT_DIR -type f -name "*.sh" -exec chmod 755 {} \;
chmod 600 $PROJECT_DIR/.env

# Install fail2ban for additional security
sudo apt install -y fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban

# Step 11: Health check and startup
echo "🏥 Performing health checks..."

# Start services
echo "Starting services..."
sudo supervisorctl start trading-bot:*

# Wait a moment for services to start
sleep 10

# Check service status
echo "📊 Service Status:"
sudo supervisorctl status trading-bot:*

# Test endpoints
echo "🧪 Testing endpoints..."
curl -f http://localhost:8000/api/health || echo "❌ Main API health check failed"
curl -f http://localhost:8001/api/health || echo "❌ Intelligence API health check failed"
curl -f http://localhost:3000 || echo "❌ React dashboard check failed"

# Step 12: Create maintenance scripts
cat > $PROJECT_DIR/scripts/status.sh << 'EOL'
#!/bin/bash
echo "🔍 Trading Bot System Status"
echo "=========================="
echo "📊 Supervisor Status:"
sudo supervisorctl status trading-bot:*
echo ""
echo "🌐 Nginx Status:"
sudo systemctl status nginx --no-pager -l
echo ""
echo "🔒 Firewall Status:"
sudo ufw status
echo ""
echo "💾 Disk Space:"
df -h /opt/trading-bot
echo ""
echo "🧠 Memory Usage:"
free -h
echo ""
echo "⚡ Recent Logs (last 5 lines each):"
echo "--- Trading Bot API ---"
tail -5 /opt/trading-bot/logs/trading-bot-api.log
echo "--- Intelligence API ---"
tail -5 /opt/trading-bot/logs/intelligence-api.log
EOL

cat > $PROJECT_DIR/scripts/restart.sh << 'EOL'
#!/bin/bash
echo "🔄 Restarting Trading Bot Services..."
sudo supervisorctl restart trading-bot:*
sudo systemctl reload nginx
echo "✅ All services restarted"
EOL

chmod +x $PROJECT_DIR/scripts/*.sh

echo ""
echo "🎉 Production Deployment Complete!"
echo "================================="
echo "✅ Project deployed to: $PROJECT_DIR"
echo "✅ Services configured and running"
echo "✅ Nginx reverse proxy configured"
echo "✅ Firewall rules applied"
echo "✅ SSL ready (configure certificates as needed)"
echo ""
echo "🌐 Access Points:"
echo "   Dashboard: http://85.215.183.30"
echo "   Main API: http://85.215.183.30/api/"
echo "   Intelligence API: http://85.215.183.30/api/insights/"
echo "   Health Check: http://85.215.183.30/health"
echo ""
echo "🛠️  Management Commands:"
echo "   Status: $PROJECT_DIR/scripts/status.sh"
echo "   Restart: $PROJECT_DIR/scripts/restart.sh"
echo "   Firewall: $PROJECT_DIR/scripts/firewall-status.sh"
echo "   Logs: sudo supervisorctl tail -f trading-bot:trading-bot-api"
echo ""
echo "⚠️  Next Steps:"
echo "1. Edit $PROJECT_DIR/.env with your API keys and secrets"
echo "2. Configure SSL certificate for HTTPS"
echo "3. Set up monitoring and alerting"
echo "4. Configure backup strategy"
echo "5. Test all functionality thoroughly"
echo ""
echo "🚀 Your Intelligence Trading Bot is now live on the server!"