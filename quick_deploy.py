#!/usr/bin/env python3
"""
🚀 JANICS FREEDOM FACTORY - QUICK DEPLOYMENT
Quick deployment script with predefined settings
"""

import os
import sys
import subprocess
import tarfile
import time
from pathlib import Path

def create_deployment_package():
    """Create deployment package"""
    print("📦 Creating deployment package...")
    
    # Files to include
    include_items = [
        'api', 'core', 'strategies', 'db', 'config', 'utils',
        'main.py', 'requirements.txt', 'config.yaml',
        'production_config.py'
    ]
    
    # Create deployment directory
    deploy_dir = Path('deployment_package')
    deploy_dir.mkdir(exist_ok=True)
    
    # Copy files
    for item in include_items:
        if os.path.exists(item):
            if os.path.isdir(item):
                subprocess.run(['cp', '-r', item, str(deploy_dir)], check=True)
            else:
                subprocess.run(['cp', item, str(deploy_dir)], check=True)
    
    # Create setup script
    setup_script = """#!/bin/bash
set -e

echo "🚀 Setting up Janics Freedom Factory..."

# Install Python dependencies
pip3 install -r requirements.txt

# Create directories
mkdir -p logs data

# Initialize database if needed
if [ ! -f "db/trading_bot.db" ]; then
    python3 -c "from db.models import init_db; init_db()" || echo "Database init skipped"
fi

# Create systemd service
sudo tee /etc/systemd/system/janics_bot.service > /dev/null << EOF
[Unit]
Description=Janics Trading Bot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PWD
ExecStart=/usr/bin/python3 $PWD/api/app.py
Restart=always
Environment="FLASK_PORT=8080"
Environment="FLASK_HOST=0.0.0.0"

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable janics_bot
sudo systemctl start janics_bot

echo "✅ Setup complete!"
echo "Check status: sudo systemctl status janics_bot"
"""
    
    setup_path = deploy_dir / 'setup.sh'
    with open(setup_path, 'w') as f:
        f.write(setup_script)
    setup_path.chmod(0o755)
    
    # Create start script
    start_script = """#!/bin/bash
# Quick start script for manual testing

echo "🚀 Starting Janics Freedom Factory..."

# Set environment
export FLASK_PORT=8080
export FLASK_HOST=0.0.0.0
export PYTHONPATH=$PWD

# Create directories
mkdir -p logs data

# Start the API
python3 api/app.py
"""
    
    start_path = deploy_dir / 'start.sh'
    with open(start_path, 'w') as f:
        f.write(start_script)
    start_path.chmod(0o755)
    
    # Create tarball
    timestamp = int(time.time())
    tarball_name = f'janics_bot_deploy_{timestamp}.tar.gz'
    
    with tarfile.open(tarball_name, 'w:gz') as tar:
        tar.add(deploy_dir, arcname='altcoin_trading_bot')
    
    # Clean up
    subprocess.run(['rm', '-rf', str(deploy_dir)], check=True)
    
    print(f"✅ Deployment package created: {tarball_name}")
    print(f"📦 Package size: {os.path.getsize(tarball_name) / 1024 / 1024:.2f} MB")
    
    return tarball_name

def show_deployment_instructions(tarball_name):
    """Show manual deployment instructions"""
    print("\n" + "="*60)
    print("📋 DEPLOYMENT INSTRUCTIONS")
    print("="*60)
    
    print("\n1️⃣  Upload the package to your server:")
    print(f"   scp {tarball_name} ubuntu@your-server:/tmp/")
    
    print("\n2️⃣  SSH into your server:")
    print("   ssh ubuntu@your-server")
    
    print("\n3️⃣  Extract and setup:")
    print("   cd ~")
    print(f"   tar -xzf /tmp/{tarball_name}")
    print("   cd altcoin_trading_bot")
    print("   ./setup.sh")
    
    print("\n4️⃣  For manual start (without systemd):")
    print("   ./start.sh")
    
    print("\n5️⃣  Check if it's running:")
    print("   curl http://localhost:8080/api/v1/orchestrator/status")
    
    print("\n" + "="*60)
    print("🔒 SECURITY CHECKLIST:")
    print("="*60)
    print("☐ Change default passwords in .env.production")
    print("☐ Setup SSL certificate (use Let's Encrypt)")
    print("☐ Configure firewall (only allow ports 80,443,22)")
    print("☐ Enable fail2ban for SSH protection")
    print("☐ Setup regular backups")
    
    print("\n" + "="*60)
    print("🌐 NGINX CONFIGURATION (optional):")
    print("="*60)
    print("""
sudo tee /etc/nginx/sites-available/janics_bot << EOF
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host \\$host;
        proxy_set_header X-Real-IP \\$remote_addr;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/janics_bot /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
""")

def main():
    """Main function"""
    print("🚀 JANICS FREEDOM FACTORY - QUICK DEPLOYMENT")
    print("=" * 60)
    
    # Create package
    tarball = create_deployment_package()
    
    # Show instructions
    show_deployment_instructions(tarball)
    
    print(f"\n✅ Package ready for deployment: {tarball}")
    print("📤 Follow the instructions above to deploy to your server")

if __name__ == "__main__":
    main()