# 🚀 JANICS FREEDOM FACTORY - DEPLOYMENT GUIDE

## Git Repository
- **Repository**: https://github.com/Jimsonhahn/ML_altcoin_trading_bot.git
- **Branch**: main

## Quick Deployment Steps

### 1️⃣ Push Changes to GitHub
```bash
git add .
git commit -m "Add production deployment configuration"
git push origin main
```

### 2️⃣ On Your Server (First Time Setup)
```bash
# Clone repository
git clone https://github.com/Jimsonhahn/ML_altcoin_trading_bot.git
cd ML_altcoin_trading_bot

# Run production setup
chmod +x setup_production.sh
./setup_production.sh

# Configure environment
cp .env.production.example .env.production
nano .env.production  # Edit with your settings

# Start the bot
sudo systemctl start janics_bot
```

### 3️⃣ For Updates
```bash
# On server
cd ML_altcoin_trading_bot
./update_production.sh
```

## GitHub Actions Setup

Add these secrets to your GitHub repository (Settings → Secrets):

- `SERVER_HOST`: Your server IP or domain
- `SERVER_USER`: SSH username (e.g., ubuntu)
- `SERVER_SSH_KEY`: Private SSH key for server access

## Server Requirements

- Ubuntu 20.04+ or similar Linux
- Python 3.9+
- 2GB RAM minimum
- 10GB disk space

## Security Checklist

- [ ] Change all default passwords in `.env.production`
- [ ] Setup SSL certificate with Let's Encrypt
- [ ] Configure firewall (ufw)
- [ ] Enable fail2ban
- [ ] Setup regular backups
- [ ] Use deploy keys instead of passwords

## Monitoring

After deployment, check:
- Bot status: `sudo systemctl status janics_bot`
- API health: `curl http://localhost:8080/health`
- Logs: `sudo journalctl -u janics_bot -f`

## Support

For issues, check the logs first:
```bash
tail -f logs/trading_bot.log
sudo journalctl -u janics_bot -n 100
```