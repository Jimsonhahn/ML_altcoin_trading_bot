#!/bin/bash
# 🚀 Deployment Script für Enhanced Trading Bot
# Maßgeschneidert für deine Windows Server Integration

set -e

echo "🚀 Enhanced Trading Bot Server Deployment"
echo "========================================="

# Configuration
SERVER_IP="85.215.183.30"
SERVER_USER="administrator"  # Oder dein aktueller Username
REMOTE_PATH="/opt/trading-bot"
LOCAL_PROJECT_PATH="$(pwd)"

# Colors für Output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}===> $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if server is reachable
print_step "Checking server connectivity..."
if ping -c 1 $SERVER_IP > /dev/null 2>&1; then
    print_success "Server is reachable"
else
    print_error "Server not reachable. Check network connection."
    exit 1
fi

# Check if SSH connection works
print_step "Testing SSH connection..."
if ssh -o ConnectTimeout=10 -o BatchMode=yes $SERVER_USER@$SERVER_IP exit 2>/dev/null; then
    print_success "SSH connection successful"
else
    print_warning "SSH connection failed. Will try with password prompt."
fi

# Create remote directory
print_step "Creating remote directory structure..."
ssh $SERVER_USER@$SERVER_IP "mkdir -p $REMOTE_PATH/{logs,data,intelligence_exports,backups,docker}"

# Sync project files (excluding large/unnecessary files)
print_step "Syncing project files to server..."
rsync -avz --progress \
    --exclude 'node_modules' \
    --exclude '__pycache__' \
    --exclude '.git' \
    --exclude '*.pyc' \
    --exclude 'logs/*' \
    --exclude 'data/*' \
    --exclude 'backups/*' \
    --exclude 'intelligence_exports/*' \
    --exclude '.env' \
    $LOCAL_PROJECT_PATH/ \
    $SERVER_USER@$SERVER_IP:$REMOTE_PATH/

print_success "Files synced successfully"

# Copy environment template
print_step "Setting up environment configuration..."
scp docker/.env.production.example $SERVER_USER@$SERVER_IP:$REMOTE_PATH/.env.example

print_warning "Don't forget to configure .env file with your actual values!"

# Install Docker and Docker Compose (wenn nicht vorhanden)
print_step "Checking Docker installation on server..."
ssh $SERVER_USER@$SERVER_IP "which docker" >/dev/null 2>&1 || {
    print_step "Installing Docker on server..."
    ssh $SERVER_USER@$SERVER_IP "
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker \$USER
        sudo systemctl enable docker
        sudo systemctl start docker
        
        # Install Docker Compose
        sudo curl -L \"https://github.com/docker/compose/releases/latest/download/docker-compose-\$(uname -s)-\$(uname -m)\" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    "
    print_success "Docker installed successfully"
}

# Create systemd service für auto-start
print_step "Creating systemd service..."
ssh $SERVER_USER@$SERVER_IP "sudo tee /etc/systemd/system/enhanced-trading-bot.service > /dev/null << EOL
[Unit]
Description=Enhanced Trading Bot
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$REMOTE_PATH
ExecStart=/usr/local/bin/docker-compose -f docker/docker-compose.production.yml up -d
ExecStop=/usr/local/bin/docker-compose -f docker/docker-compose.production.yml down
User=$SERVER_USER

[Install]
WantedBy=multi-user.target
EOL"

ssh $SERVER_USER@$SERVER_IP "
    sudo systemctl daemon-reload
    sudo systemctl enable enhanced-trading-bot
"

print_success "Systemd service created"

# Setup firewall rules
print_step "Configuring firewall..."
ssh $SERVER_USER@$SERVER_IP "
    # Ubuntu/Debian firewall
    if which ufw >/dev/null 2>&1; then
        sudo ufw allow 22/tcp    # SSH
        sudo ufw allow 80/tcp    # HTTP
        sudo ufw allow 443/tcp   # HTTPS
        sudo ufw allow 8000/tcp  # API
        sudo ufw allow 3000/tcp  # Dashboard
        sudo ufw --force enable
    fi
    
    # CentOS/RHEL firewall
    if which firewall-cmd >/dev/null 2>&1; then
        sudo firewall-cmd --permanent --add-port=22/tcp
        sudo firewall-cmd --permanent --add-port=80/tcp
        sudo firewall-cmd --permanent --add-port=443/tcp
        sudo firewall-cmd --permanent --add-port=8000/tcp
        sudo firewall-cmd --permanent --add-port=3000/tcp
        sudo firewall-cmd --reload
    fi
"

print_success "Firewall configured"

# Create management scripts
print_step "Creating management scripts..."
ssh $SERVER_USER@$SERVER_IP "
cat > $REMOTE_PATH/start_bot.sh << 'EOL'
#!/bin/bash
echo '🚀 Starting Enhanced Trading Bot...'
cd $REMOTE_PATH
docker-compose -f docker/docker-compose.production.yml up -d
echo '✅ Bot started successfully!'
echo 'Dashboard: http://$SERVER_IP'
echo 'API: http://$SERVER_IP:8000/api/v1/health'
EOL

cat > $REMOTE_PATH/stop_bot.sh << 'EOL'
#!/bin/bash
echo '🛑 Stopping Enhanced Trading Bot...'
cd $REMOTE_PATH
docker-compose -f docker/docker-compose.production.yml down
echo '✅ Bot stopped successfully!'
EOL

cat > $REMOTE_PATH/restart_bot.sh << 'EOL'
#!/bin/bash
echo '🔄 Restarting Enhanced Trading Bot...'
cd $REMOTE_PATH
docker-compose -f docker/docker-compose.production.yml restart
echo '✅ Bot restarted successfully!'
EOL

cat > $REMOTE_PATH/status_bot.sh << 'EOL'
#!/bin/bash
echo '📊 Enhanced Trading Bot Status'
echo '============================='
cd $REMOTE_PATH
docker-compose -f docker/docker-compose.production.yml ps
echo ''
echo '🌐 Service URLs:'
echo \"Dashboard: http://$SERVER_IP\"
echo \"API Health: http://$SERVER_IP:8000/api/v1/health\"
echo \"Intelligence: http://$SERVER_IP:8000/api/v1/intelligence/metrics\"
EOL

cat > $REMOTE_PATH/logs_bot.sh << 'EOL'
#!/bin/bash
echo '📋 Enhanced Trading Bot Logs'
echo '=========================='
cd $REMOTE_PATH
if [ \"\$1\" = \"follow\" ] || [ \"\$1\" = \"-f\" ]; then
    docker-compose -f docker/docker-compose.production.yml logs -f
else
    docker-compose -f docker/docker-compose.production.yml logs --tail=50
fi
EOL

chmod +x $REMOTE_PATH/*.sh
"

print_success "Management scripts created"

# Final deployment steps
print_step "Final deployment configuration..."

echo ""
print_success "🎉 Deployment completed successfully!"
echo ""
echo "📋 Next Steps:"
echo "1. SSH to server: ssh $SERVER_USER@$SERVER_IP"
echo "2. Configure environment: cd $REMOTE_PATH && cp .env.example .env && nano .env"
echo "3. Start the bot: ./start_bot.sh"
echo "4. Check status: ./status_bot.sh"
echo ""
echo "🌐 Access Points (after starting):"
echo "   Dashboard: http://$SERVER_IP"
echo "   API Health: http://$SERVER_IP:8000/api/v1/health"
echo "   Intelligence: http://$SERVER_IP:8000/api/v1/intelligence/metrics"
echo ""
echo "🛠️ Management Commands (on server):"
echo "   Start:   ./start_bot.sh"
echo "   Stop:    ./stop_bot.sh"
echo "   Restart: ./restart_bot.sh"
echo "   Status:  ./status_bot.sh"
echo "   Logs:    ./logs_bot.sh [-f]"
echo ""
echo "🔄 Auto-start on boot: sudo systemctl start enhanced-trading-bot"
echo ""
print_warning "Remember to configure your .env file with real API keys before starting!"

# Optional: Start the bot immediately
read -p "Do you want to start the bot now? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_step "Starting bot..."
    ssh $SERVER_USER@$SERVER_IP "cd $REMOTE_PATH && ./start_bot.sh"
    sleep 10
    ssh $SERVER_USER@$SERVER_IP "cd $REMOTE_PATH && ./status_bot.sh"
fi