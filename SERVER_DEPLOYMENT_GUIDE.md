# 🚀 Complete Server Deployment Guide
## Intelligence Trading Bot on 85.215.183.30

This guide provides step-by-step instructions for deploying your Intelligence Trading Bot to your server with PyCharm Remote Development integration.

---

## 📋 Prerequisites

- Server: 85.215.183.30 with Ubuntu 20.04+ 
- User: administrator with sudo privileges
- PyCharm Professional (for Remote Development)
- SSH access configured

---

## 🏗️ Phase 1: Server Initial Setup

### 1.1 Connect via SSH
```bash
ssh administrator@85.215.183.30
```

### 1.2 Run Server Setup Script
```bash
# Download and run the setup script
wget https://raw.githubusercontent.com/your-repo/setup.sh -O setup.sh
chmod +x setup.sh
./setup.sh

# Or if you have the project files locally:
bash /path/to/scripts/server_setup.sh
```

This script will:
- ✅ Update system packages
- ✅ Install Python 3.11, Node.js 18
- ✅ Install PostgreSQL, Redis, Nginx
- ✅ Create project directory structure
- ✅ Set up virtual environment

---

## 🔒 Phase 2: Security Configuration

### 2.1 Configure Firewall
```bash
# Run the firewall setup script
sudo bash /opt/trading-bot/scripts/setup_firewall.sh
```

**Firewall Rules Applied:**
- ✅ SSH (22) - Rate limited
- ✅ HTTP/HTTPS (80, 443)
- ✅ Trading Bot API (8000) - Rate limited  
- ✅ Intelligence API (8001) - Rate limited
- ✅ React Dashboard (3000-3003)
- ✅ PostgreSQL (5432) - Localhost only
- ✅ Redis (6379) - Localhost only

### 2.2 Verify Firewall Status
```bash
/opt/trading-bot/scripts/firewall-status.sh
```

---

## 🐍 Phase 3: PyCharm Remote Development Setup

### 3.1 Configure PyCharm Remote Development
1. **Open PyCharm on your Mac**
2. **Go to**: File → Remote Development → SSH
3. **Enter connection details:**
   - Host: `85.215.183.30`
   - Username: `administrator`
   - Port: `22`
   - Authentication: SSH key or password

### 3.2 Project Setup in PyCharm
1. **Select project directory**: `/opt/trading-bot`
2. **Python interpreter**: `/opt/trading-bot/venv/bin/python`
3. **Working directory**: `/opt/trading-bot`

### 3.3 Upload Project Files
Using PyCharm's built-in sync or manually:
```bash
# On your local machine, sync files to server
rsync -avz --exclude node_modules --exclude __pycache__ \
  /Users/jnb/PycharmProjects/altcoin_trading_bot/ \
  administrator@85.215.183.30:/opt/trading-bot/
```

---

## 🏭 Phase 4: Production Deployment

### 4.1 Configure Environment Variables
```bash
cd /opt/trading-bot
cp config/production.env .env

# Edit the .env file with your actual values:
nano .env
```

**Important settings to configure:**
```env
# API Keys
BINANCE_API_KEY=your_actual_binance_api_key
BINANCE_SECRET_KEY=your_actual_binance_secret_key

# Database
DB_PASSWORD=your_secure_database_password

# Security
SECRET_KEY=generate_with_openssl_rand_hex_32
JWT_SECRET=generate_another_secure_key

# Telegram (optional)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

### 4.2 Run Complete Deployment
```bash
# Make deployment script executable and run
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

The deployment script will:
- ✅ Set up directories and permissions
- ✅ Install Python dependencies
- ✅ Configure PostgreSQL database
- ✅ Set up Nginx reverse proxy
- ✅ Configure Supervisor for process management
- ✅ Build React dashboard
- ✅ Set up log rotation
- ✅ Configure systemd services
- ✅ Apply security settings
- ✅ Perform health checks

---

## 🌐 Phase 5: Intelligence Dashboard Configuration

### 5.1 Intelligence API Configuration
The Intelligence API is configured for production with:
- **Enhanced logging** to `/opt/trading-bot/logs/intelligence_api.log`
- **Server monitoring** endpoints
- **Production-optimized** CORS and middleware
- **Health checks** with system status

### 5.2 Access Points After Deployment

| Service | URL | Description |
|---------|-----|-------------|
| **Main Dashboard** | `http://85.215.183.30` | React dashboard with Intelligence tab |
| **Trading API** | `http://85.215.183.30/api/` | Main trading bot endpoints |
| **Intelligence API** | `http://85.215.183.30/api/insights/` | AI insights and patterns |
| **Health Check** | `http://85.215.183.30/health` | System health status |
| **API Docs** | `http://85.215.183.30/api/docs` | Interactive API documentation |

### 5.3 Intelligence Dashboard Features
- 🧠 **Real-time Insights**: ML-generated trading insights
- 📊 **Performance Charts**: Strategy performance evolution  
- 🔍 **Pattern Detection**: AI-discovered market patterns
- 💡 **Smart Recommendations**: Optimization suggestions
- 📈 **Backtest Results**: Strategy comparison charts
- 🔄 **Auto-refresh**: Updates every 30 seconds
- 📱 **Mobile-responsive**: Works on all devices

---

## 🛠️ Phase 6: Management and Monitoring

### 6.1 Service Management Commands
```bash
# Check system status
/opt/trading-bot/scripts/status.sh

# Restart all services
/opt/trading-bot/scripts/restart.sh

# View logs
sudo supervisorctl tail -f trading-bot:intelligence-api
sudo supervisorctl tail -f trading-bot:trading-bot-api

# Service control
sudo supervisorctl start trading-bot:*
sudo supervisorctl stop trading-bot:*
sudo supervisorctl restart trading-bot:*
```

### 6.2 Log Files Location
```
/opt/trading-bot/logs/
├── trading-bot-api.log      # Main API logs
├── intelligence-api.log     # Intelligence API logs  
├── react-dashboard.log      # Frontend logs
├── worker.log              # Background worker logs
└── nginx/
    ├── trading-bot-access.log
    └── trading-bot-error.log
```

### 6.3 Database Management
```bash
# Connect to database
sudo -u postgres psql trading_bot

# Backup database
pg_dump -U trading_user trading_bot > backup_$(date +%Y%m%d).sql

# View database tables
\dt
```

---

## 🔧 Phase 7: Testing and Verification

### 7.1 Run Integration Tests
```bash
cd /opt/trading-bot
python test_intelligence_integration.py
```

### 7.2 Manual Testing Checklist
- [ ] Main dashboard loads: `http://85.215.183.30`
- [ ] Intelligence tab accessible and functional
- [ ] API health check: `http://85.215.183.30/health`
- [ ] All charts render correctly
- [ ] Auto-refresh works (30-second interval)
- [ ] Pattern modals open and display data
- [ ] WebSocket connections established
- [ ] Logs are being written properly

### 7.3 Performance Testing
```bash
# Test API performance
curl -w "@curl-format.txt" -o /dev/null -s "http://85.215.183.30/api/health"

# Monitor system resources
htop
iotop
nethogs
```

---

## 🚨 Phase 8: Security Hardening

### 8.1 SSL Certificate Setup (Recommended)
```bash
# Install Certbot for Let's Encrypt
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d 85.215.183.30

# Verify auto-renewal
sudo certbot renew --dry-run
```

### 8.2 Additional Security Measures
```bash
# Install and configure fail2ban
sudo apt install fail2ban
sudo systemctl enable fail2ban

# Set up automatic security updates
sudo apt install unattended-upgrades
sudo dpkg-reconfigure unattended-upgrades

# Configure log monitoring
sudo apt install logwatch
```

---

## 📊 Phase 9: Monitoring and Alerting

### 9.1 System Monitoring
```bash
# Install monitoring tools
sudo apt install htop iotop nethogs

# Set up cron jobs for monitoring
crontab -e
# Add: */5 * * * * /opt/trading-bot/scripts/health-check.sh
```

### 9.2 Application Monitoring
The Intelligence API includes built-in monitoring:
- **System metrics**: CPU, memory, disk usage
- **Application health**: Service status, database connectivity
- **API performance**: Response times, error rates

---

## 🔄 Phase 10: Backup and Recovery

### 10.1 Automated Backups
```bash
# Create backup script
cat > /opt/trading-bot/scripts/backup.sh << 'EOL'
#!/bin/bash
BACKUP_DIR="/opt/trading-bot/backups/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Database backup
pg_dump -U trading_user trading_bot > $BACKUP_DIR/database.sql

# Configuration backup
cp -r /opt/trading-bot/config $BACKUP_DIR/
cp /opt/trading-bot/.env $BACKUP_DIR/

# Logs backup (last 7 days)
find /opt/trading-bot/logs -name "*.log" -mtime -7 -exec cp {} $BACKUP_DIR/ \;

echo "Backup completed: $BACKUP_DIR"
EOL

chmod +x /opt/trading-bot/scripts/backup.sh

# Schedule daily backups
echo "0 2 * * * /opt/trading-bot/scripts/backup.sh" | crontab -
```

---

## 🎯 Summary

Your Intelligence Trading Bot is now fully deployed on server `85.215.183.30` with:

### ✅ **Deployed Components**
- **Trading Bot API** (port 8000)
- **Intelligence Dashboard API** (port 8001)  
- **React Frontend** (port 3000, proxied via Nginx)
- **PostgreSQL Database** (port 5432, localhost only)
- **Redis Cache** (port 6379, localhost only)

### ✅ **Production Features**
- **Nginx reverse proxy** with SSL-ready configuration
- **Supervisor process management** for automatic restarts
- **UFW firewall** with rate limiting
- **Log rotation** and centralized logging
- **Health monitoring** and status endpoints
- **Backup automation** scripts

### ✅ **Intelligence Dashboard Features**
- **Real-time insights** with ML pattern detection
- **Performance analytics** with interactive charts
- **Strategy recommendations** based on AI analysis
- **Mobile-responsive design** with dark mode
- **Auto-refresh functionality** every 30 seconds

### 🚀 **Access Your System**
- **Dashboard**: http://85.215.183.30 (click "Intelligence" tab)
- **API Documentation**: http://85.215.183.30/api/docs
- **Health Status**: http://85.215.183.30/health

### 🔧 **PyCharm Remote Development**
Your PyCharm IDE is now connected to the server, allowing you to:
- Edit code directly on the server
- Run and debug applications remotely
- Use server resources for computation
- Collaborate with Claude Code seamlessly

**🎉 Your intelligent trading bot is now live and ready for production use!**