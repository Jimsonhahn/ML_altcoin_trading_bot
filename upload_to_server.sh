#!/bin/bash
# 🚀 JANICS FREEDOM FACTORY - SERVER UPLOAD HELPER

echo "🚀 JANICS FREEDOM FACTORY - SERVER UPLOAD"
echo "========================================"

# Get server details
read -p "Enter server hostname/IP: " SERVER_HOST
read -p "Enter server username [ubuntu]: " SERVER_USER
SERVER_USER=${SERVER_USER:-ubuntu}

# Find the latest deployment package
PACKAGE=$(ls -t janics_bot_deploy_*.tar.gz | head -1)

if [ -z "$PACKAGE" ]; then
    echo "❌ No deployment package found!"
    echo "Run 'python quick_deploy.py' first"
    exit 1
fi

echo "📦 Found package: $PACKAGE"
echo "📤 Uploading to $SERVER_USER@$SERVER_HOST..."

# Upload package
scp "$PACKAGE" "$SERVER_USER@$SERVER_HOST:/tmp/"

if [ $? -eq 0 ]; then
    echo "✅ Upload successful!"
    echo ""
    echo "🔧 Now SSH into your server and run:"
    echo "========================================"
    echo "ssh $SERVER_USER@$SERVER_HOST"
    echo "cd ~"
    echo "tar -xzf /tmp/$PACKAGE"
    echo "cd altcoin_trading_bot"
    echo "./setup.sh"
    echo ""
    echo "📊 After setup, check status with:"
    echo "curl http://localhost:8080/api/v1/orchestrator/status"
    
    # Offer to SSH directly
    echo ""
    read -p "Connect to server now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ssh "$SERVER_USER@$SERVER_HOST"
    fi
else
    echo "❌ Upload failed!"
fi