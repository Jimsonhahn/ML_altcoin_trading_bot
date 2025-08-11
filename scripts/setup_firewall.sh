#!/bin/bash
# Firewall Configuration for Trading Bot Server
# Run this on your server (85.215.183.30) with sudo privileges

set -e

echo "🔒 Configuring UFW Firewall for Trading Bot..."
echo "=================================================="

# Reset UFW to defaults
echo "🔄 Resetting UFW to defaults..."
sudo ufw --force reset

# Set default policies
echo "📋 Setting default policies..."
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH (CRITICAL - don't lock yourself out!)
echo "🔑 Allowing SSH access..."
sudo ufw allow ssh
sudo ufw allow 22/tcp

# Allow HTTP and HTTPS for web access
echo "🌐 Allowing HTTP/HTTPS access..."
sudo ufw allow http
sudo ufw allow https
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Trading Bot API Endpoints
echo "📡 Configuring Trading Bot API ports..."

# Main Trading Bot API
sudo ufw allow 8000/tcp comment "Trading Bot Main API"

# Intelligence Dashboard API
sudo ufw allow 8001/tcp comment "Intelligence Dashboard API"

# React Dashboard (development)
sudo ufw allow 3000/tcp comment "React Dashboard"
sudo ufw allow 3001/tcp comment "React Dashboard Alt"
sudo ufw allow 3002/tcp comment "React Dashboard Alt"
sudo ufw allow 3003/tcp comment "React Dashboard Alt"

# Database ports (restrict to localhost only)
echo "🗄️ Configuring database access..."
# PostgreSQL - only from localhost
sudo ufw allow from 127.0.0.1 to any port 5432 comment "PostgreSQL localhost only"

# Redis - only from localhost
sudo ufw allow from 127.0.0.1 to any port 6379 comment "Redis localhost only"

# WebSocket connections for real-time updates
echo "📡 Allowing WebSocket connections..."
sudo ufw allow 8080/tcp comment "WebSocket connections"

# Optional: Telegram Bot webhooks (if using webhooks instead of polling)
# sudo ufw allow 8443/tcp comment "Telegram Bot webhook"

# Optional: Email server ports (if sending email alerts)
# sudo ufw allow out 587 comment "SMTP outgoing"
# sudo ufw allow out 465 comment "SMTP SSL outgoing"

# Allow specific IP ranges (optional - for additional security)
echo "🔐 Optional: Allow specific IP ranges..."
echo "   Uncomment the following lines to restrict access to specific IPs:"
echo "   # sudo ufw allow from YOUR_IP_ADDRESS to any port 8000"
echo "   # sudo ufw allow from YOUR_IP_ADDRESS to any port 8001"

# Rate limiting for API endpoints
echo "⚡ Setting up rate limiting..."
sudo ufw limit ssh comment "Rate limit SSH"
sudo ufw limit 8000/tcp comment "Rate limit Main API"
sudo ufw limit 8001/tcp comment "Rate limit Intelligence API"

# Enable UFW
echo "✅ Enabling UFW firewall..."
sudo ufw --force enable

# Show status
echo "📊 Current UFW status:"
sudo ufw status verbose

echo ""
echo "🎯 Firewall Configuration Summary:"
echo "=================================="
echo "✅ SSH access: Port 22 (rate limited)"
echo "✅ HTTP/HTTPS: Ports 80, 443"
echo "✅ Trading Bot API: Port 8000 (rate limited)"
echo "✅ Intelligence API: Port 8001 (rate limited)"
echo "✅ React Dashboard: Ports 3000-3003"
echo "✅ WebSocket: Port 8080"
echo "✅ PostgreSQL: Port 5432 (localhost only)"
echo "✅ Redis: Port 6379 (localhost only)"
echo ""
echo "🔒 Security Features:"
echo "• Default deny incoming policy"
echo "• Rate limiting on critical services"
echo "• Database access restricted to localhost"
echo "• All outgoing connections allowed"
echo ""
echo "⚠️  Important Security Notes:"
echo "1. Always test SSH access before disconnecting!"
echo "2. Consider restricting API access to specific IPs"
echo "3. Monitor logs regularly: sudo tail -f /var/log/ufw.log"
echo "4. Update firewall rules as needed for your setup"
echo ""
echo "🚀 Your server is now secured and ready for deployment!"

# Create a quick reference script
cat > /opt/trading-bot/scripts/firewall-status.sh << 'EOL'
#!/bin/bash
# Quick firewall status check
echo "🔒 UFW Firewall Status:"
sudo ufw status verbose
echo ""
echo "📊 Active connections:"
sudo netstat -tlnp | grep -E ":(22|80|443|8000|8001|3000|5432|6379) "
echo ""
echo "📝 Recent UFW logs (last 10 lines):"
sudo tail -10 /var/log/ufw.log
EOL

chmod +x /opt/trading-bot/scripts/firewall-status.sh

echo "📝 Created firewall status script: /opt/trading-bot/scripts/firewall-status.sh"