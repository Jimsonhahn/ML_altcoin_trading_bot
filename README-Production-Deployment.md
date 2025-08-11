# 🚀 Production Deployment Guide

This guide provides comprehensive instructions for deploying the trading bot in production with 24/7 operation.

## 📋 Table of Contents
- [Prerequisites](#prerequisites)
- [Deployment Options](#deployment-options)
- [Systemd Service Setup](#systemd-service-setup)
- [PM2 Process Manager Setup](#pm2-process-manager-setup)
- [Configuration](#configuration)
- [Monitoring & Health Checks](#monitoring--health-checks)
- [Backup & Recovery](#backup--recovery)
- [Security](#security)
- [Troubleshooting](#troubleshooting)

## 🔧 Prerequisites

### System Requirements
- **OS**: Ubuntu 20.04 LTS or newer (recommended)
- **CPU**: 2+ cores
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 50GB SSD minimum
- **Network**: Stable internet connection

### Dependencies
- Python 3.8+
- PostgreSQL 12+
- Redis 6+
- SSL certificates (for production)

## 🚀 Deployment Options

Choose between two production deployment methods:

### Option 1: Systemd Service (Recommended)
- Native Linux service management
- Better resource control and security
- Automatic restart on system boot
- Integration with system logging

### Option 2: PM2 Process Manager
- Advanced process monitoring
- Built-in load balancing
- Real-time monitoring dashboard
- Better for multi-instance deployments

## 🔄 Systemd Service Setup

### 1. Installation
```bash
# Clone the repository
git clone <your-repo-url> /tmp/trading-bot
cd /tmp/trading-bot

# Run the installation script
sudo chmod +x scripts/install-systemd.sh
sudo ./scripts/install-systemd.sh
```

### 2. Configuration
```bash
# Edit production configuration
sudo nano /opt/trading-bot/.env.production

# Key settings to update:
# - BINANCE_API_KEY
# - BINANCE_SECRET_KEY
# - BINANCE_TESTNET=false
# - TELEGRAM_BOT_TOKEN
# - JWT_SECRET
# - Database credentials
```

### 3. Start the Service
```bash
# Start the service
sudo systemctl start trading-bot

# Check status
sudo systemctl status trading-bot

# View logs
sudo journalctl -u trading-bot -f

# Enable auto-start on boot
sudo systemctl enable trading-bot
```

### 4. Service Management
```bash
# Start
sudo systemctl start trading-bot

# Stop
sudo systemctl stop trading-bot

# Restart
sudo systemctl restart trading-bot

# Status
sudo systemctl status trading-bot

# Logs
sudo journalctl -u trading-bot -f --lines=100
```

## 🔄 PM2 Process Manager Setup

### 1. Installation
```bash
# Clone the repository
git clone <your-repo-url> /tmp/trading-bot
cd /tmp/trading-bot

# Run the PM2 installation script
sudo chmod +x scripts/install-pm2.sh
sudo ./scripts/install-pm2.sh
```

### 2. Configuration
```bash
# Edit production configuration
sudo nano /opt/trading-bot/.env.production

# Update PM2 configuration if needed
sudo nano /opt/trading-bot/scripts/pm2-config.json
```

### 3. Start PM2 Processes
```bash
# Start all processes
/opt/trading-bot/scripts/pm2-manager.sh start

# Check status
/opt/trading-bot/scripts/pm2-manager.sh status

# View logs
/opt/trading-bot/scripts/pm2-manager.sh logs

# Open monitoring dashboard
/opt/trading-bot/scripts/pm2-manager.sh monitor
```

### 4. PM2 Management
```bash
# Status
/opt/trading-bot/scripts/pm2-manager.sh status

# Start
/opt/trading-bot/scripts/pm2-manager.sh start

# Stop
/opt/trading-bot/scripts/pm2-manager.sh stop

# Restart
/opt/trading-bot/scripts/pm2-manager.sh restart

# Reload configuration
/opt/trading-bot/scripts/pm2-manager.sh reload

# View logs
/opt/trading-bot/scripts/pm2-manager.sh logs

# Save configuration
/opt/trading-bot/scripts/pm2-manager.sh save
```

## ⚙️ Configuration

### Environment Files
- `.env.production` - Production settings with real API keys
- `.env.staging` - Staging environment for testing
- `.env.dev` - Development settings with safe defaults

### Key Production Settings
```bash
# Trading Configuration
TRADING_MODE=live                    # CRITICAL: Live trading
BINANCE_TESTNET=false               # CRITICAL: Real trading
INITIAL_CAPITAL=300000              # Your actual capital
MAX_PORTFOLIO_RISK=0.20             # 20% max risk
MAX_POSITION_SIZE=0.10              # 10% max per position

# Security
JWT_SECRET=your_secure_jwt_secret
SSL_CERT_PATH=/production/ssl/cert.pem
SSL_KEY_PATH=/production/ssl/private.key

# Monitoring
ENABLE_TRADE_NOTIFICATIONS=true
ENABLE_ERROR_NOTIFICATIONS=true
ENABLE_PERFORMANCE_REPORTS=true
ALERT_ON_DRAWDOWN_PERCENT=0.10
```

## 📊 Monitoring & Health Checks

### Health Check Endpoints
```bash
# Basic health check
curl http://localhost:8080/health

# Detailed health information
curl http://localhost:8080/health/detailed

# Kubernetes-style readiness check
curl http://localhost:8080/health/ready

# Liveness check
curl http://localhost:8080/health/live

# Prometheus metrics
curl http://localhost:8080/health/metrics

# Health history
curl http://localhost:8080/health/history
```

### Monitoring Dashboard
The health endpoints provide comprehensive monitoring:
- System resources (CPU, memory, disk)
- Database connectivity
- Exchange API status
- Trading bot performance
- Error rates and alerts

### Automated Monitoring
Both deployment methods include automated monitoring:
- Health checks every 30 seconds
- Automatic restart on failures
- Performance metrics collection
- Alert generation on thresholds

## 💾 Backup & Recovery

### Automated Backups
```bash
# Manual backup
sudo /opt/trading-bot/scripts/backup.sh

# Backup types
sudo /opt/trading-bot/scripts/backup.sh daily
sudo /opt/trading-bot/scripts/backup.sh weekly
sudo /opt/trading-bot/scripts/backup.sh monthly
```

### Backup Contents
- Configuration files (encrypted)
- Trading data and logs
- Database dumps
- Source code
- System state information

### Backup Schedule
- **Daily**: Keep 7 days
- **Weekly**: Keep 4 weeks  
- **Monthly**: Keep 6 months

### Recovery
```bash
# Extract backup
cd /opt/trading-bot/backups
tar -xzf daily_20240315_120000.tar.gz

# Restore configuration
sudo cp backup_dir/config/.env.production /opt/trading-bot/

# Restore database
psql -h localhost -U trader_prod -d trading_bot_production < backup_dir/database/dump.sql

# Restart services
sudo systemctl restart trading-bot
```

## 🔒 Security

### Production Security Checklist
- ✅ SSL/TLS certificates configured
- ✅ Firewall rules configured (ports 8080, 9090)
- ✅ Service user with minimal privileges
- ✅ Sensitive files encrypted in backups
- ✅ API rate limiting enabled
- ✅ Strong JWT secrets
- ✅ Database credentials secured

### Security Features
- **Process Isolation**: Runs as dedicated user
- **File Permissions**: Restricted access to sensitive files
- **Network Security**: Firewall rules and SSL
- **Resource Limits**: CPU and memory constraints
- **Audit Logging**: Comprehensive activity logs

## 🚨 Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check service status
sudo systemctl status trading-bot

# Check logs
sudo journalctl -u trading-bot --lines=50

# Check configuration
sudo -u trading-bot python3 /opt/trading-bot/config/environment.py
```

#### API Connection Issues
```bash
# Test exchange connectivity
curl "https://api.binance.com/api/v3/ping"

# Check API credentials
sudo -u trading-bot python3 -c "
from config.environment import get_config
config = get_config()
print('API keys configured:', bool(config.get_api_keys()['binance_api_key']))
"
```

#### High Memory Usage
```bash
# Check memory usage
sudo systemctl show trading-bot --property=MemoryCurrent

# Restart service
sudo systemctl restart trading-bot

# Adjust memory limits in service file
sudo nano /etc/systemd/system/trading-bot.service
```

#### Database Connection Issues
```bash
# Test database connection
psql -h localhost -U trader_prod -d trading_bot_production -c "SELECT 1;"

# Check database logs
sudo tail -f /var/log/postgresql/postgresql-*.log
```

### Log Locations
- **Systemd**: `journalctl -u trading-bot`
- **PM2**: `/var/log/trading-bot/`
- **Application**: `/opt/trading-bot/logs/`
- **Backup**: `/var/log/trading-bot/backup.log`

### Performance Monitoring
```bash
# System resources
htop
iostat -x 1
free -h
df -h

# Trading bot metrics
curl http://localhost:8080/health/metrics

# PM2 monitoring (if using PM2)
sudo -u trading-bot pm2 monit
```

## 📈 Production Monitoring

### Key Metrics to Monitor
- **Trading Performance**: Daily P&L, Sharpe ratio, drawdown
- **System Health**: CPU, memory, disk usage
- **API Performance**: Response times, error rates
- **Trade Execution**: Success rate, slippage
- **Risk Metrics**: Position sizes, portfolio risk

### Alerting Thresholds
- CPU usage > 80%
- Memory usage > 85%
- Disk usage > 90%
- Daily drawdown > 10%
- API error rate > 5%

### Notifications
- Telegram alerts for critical events
- Email notifications for system issues
- Daily performance reports
- Weekly risk assessments

## 🔄 Maintenance

### Regular Maintenance Tasks
- **Daily**: Check health status and logs
- **Weekly**: Review performance metrics
- **Monthly**: Update dependencies and review configuration
- **Quarterly**: Security audit and backup verification

### Updates and Upgrades
```bash
# Stop trading
sudo systemctl stop trading-bot

# Backup current state
sudo /opt/trading-bot/scripts/backup.sh manual

# Update code
cd /opt/trading-bot
git pull origin main

# Update dependencies
/opt/trading-bot/venv/bin/pip install -r requirements.txt

# Test configuration
sudo -u trading-bot python3 /opt/trading-bot/config/environment.py

# Restart service
sudo systemctl start trading-bot

# Verify health
curl http://localhost:8080/health
```

---

## 🆘 Emergency Procedures

### Emergency Stop
```bash
# Immediate stop
sudo systemctl stop trading-bot

# Or for PM2
/opt/trading-bot/scripts/pm2-manager.sh stop

# Check open positions via API or exchange web interface
```

### Disaster Recovery
1. **Assess the situation**
2. **Stop all trading immediately**
3. **Secure any open positions manually**
4. **Restore from latest backup**
5. **Verify system integrity**
6. **Restart with paper trading first**
7. **Gradually resume live trading**

### Emergency Contacts
- Exchange support for API issues
- System administrator for infrastructure
- Backup administrator for data recovery

---

## 📞 Support

For deployment issues or questions:
1. Check the troubleshooting section
2. Review the logs for error messages
3. Test configuration files
4. Verify network connectivity
5. Contact system administrator if needed

Remember: **Never deploy to production without thorough testing in staging environment first!**