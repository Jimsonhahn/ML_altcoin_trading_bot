#!/usr/bin/env python3
"""
🚀 JANICS FREEDOM FACTORY - PRODUCTION DEPLOYMENT
Automated deployment script for production server
"""

import os
import sys
import subprocess
import json
import time
from typing import Dict, List, Tuple
import paramiko
import tarfile
from pathlib import Path

class ProductionDeployer:
    def __init__(self, server_config: Dict[str, str]):
        self.server_host = server_config.get('host', 'localhost')
        self.server_user = server_config.get('user', 'ubuntu')
        self.server_port = server_config.get('port', 22)
        self.deploy_path = server_config.get('deploy_path', '/home/ubuntu/altcoin_trading_bot')
        self.ssh_key_path = server_config.get('ssh_key_path', '~/.ssh/id_rsa')
        
    def create_deployment_package(self) -> str:
        """Create deployment package with all necessary files"""
        print("📦 Creating deployment package...")
        
        # Files and directories to include
        include_items = [
            'api', 'core', 'strategies', 'db', 'config', 'utils',
            'main.py', 'requirements.txt', 'config.yaml',
            'production_config.py', 'docker-compose.prod.yml'
        ]
        
        # Create temporary deployment directory
        deploy_dir = Path('deployment_package')
        deploy_dir.mkdir(exist_ok=True)
        
        # Copy files
        for item in include_items:
            if os.path.exists(item):
                if os.path.isdir(item):
                    subprocess.run(['cp', '-r', item, str(deploy_dir)], check=True)
                else:
                    subprocess.run(['cp', item, str(deploy_dir)], check=True)
        
        # Create production scripts
        self._create_production_scripts(deploy_dir)
        
        # Create tarball
        tarball_name = f'janics_bot_deploy_{int(time.time())}.tar.gz'
        with tarfile.open(tarball_name, 'w:gz') as tar:
            tar.add(deploy_dir, arcname='.')
        
        print(f"✅ Deployment package created: {tarball_name}")
        return tarball_name
    
    def _create_production_scripts(self, deploy_dir: Path):
        """Create production-specific scripts"""
        
        # Create systemd service file
        service_content = f"""[Unit]
Description=Janics Freedom Factory Trading Bot
After=network.target

[Service]
Type=simple
User={self.server_user}
WorkingDirectory={self.deploy_path}
Environment="PYTHONPATH={self.deploy_path}"
Environment="ENV=production"
ExecStart=/usr/bin/python3 {self.deploy_path}/api/app.py
Restart=always
RestartSec=10
StandardOutput=append:{self.deploy_path}/logs/bot.log
StandardError=append:{self.deploy_path}/logs/bot_error.log

[Install]
WantedBy=multi-user.target
"""
        
        with open(deploy_dir / 'janics_bot.service', 'w') as f:
            f.write(service_content)
        
        # Create setup script
        setup_script = """#!/bin/bash
set -e

echo "🚀 Setting up Janics Freedom Factory on production server..."

# Update system
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv nginx certbot python3-certbot-nginx

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs data config/ssl

# Initialize database
python3 -c "from db.models import init_db; init_db()"

# Setup systemd service
sudo cp janics_bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable janics_bot

# Generate self-signed SSL certificate (replace with real cert in production)
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \\
    -keyout config/ssl/bot.key -out config/ssl/bot.crt \\
    -subj "/C=US/ST=State/L=City/O=JanicsFF/CN=localhost"

echo "✅ Setup complete!"
echo "Start the bot with: sudo systemctl start janics_bot"
echo "Check status with: sudo systemctl status janics_bot"
"""
        
        setup_path = deploy_dir / 'setup.sh'
        with open(setup_path, 'w') as f:
            f.write(setup_script)
        setup_path.chmod(0o755)
    
    def deploy_to_server(self, tarball_path: str):
        """Deploy to production server via SSH"""
        print(f"🚀 Deploying to {self.server_host}...")
        
        try:
            # Setup SSH connection
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            key_path = os.path.expanduser(self.ssh_key_path)
            if os.path.exists(key_path):
                ssh.connect(
                    self.server_host,
                    port=self.server_port,
                    username=self.server_user,
                    key_filename=key_path
                )
            else:
                password = input(f"Enter password for {self.server_user}@{self.server_host}: ")
                ssh.connect(
                    self.server_host,
                    port=self.server_port,
                    username=self.server_user,
                    password=password
                )
            
            # Upload tarball
            sftp = ssh.open_sftp()
            remote_tarball = f'/tmp/{os.path.basename(tarball_path)}'
            print(f"📤 Uploading {tarball_path} to {remote_tarball}...")
            sftp.put(tarball_path, remote_tarball)
            
            # Extract and setup
            commands = [
                f'mkdir -p {self.deploy_path}',
                f'cd {self.deploy_path} && tar -xzf {remote_tarball}',
                f'cd {self.deploy_path} && chmod +x setup.sh',
                f'cd {self.deploy_path} && ./setup.sh',
                f'rm {remote_tarball}'
            ]
            
            for cmd in commands:
                print(f"🔧 Running: {cmd}")
                stdin, stdout, stderr = ssh.exec_command(cmd)
                print(stdout.read().decode())
                errors = stderr.read().decode()
                if errors:
                    print(f"⚠️  Errors: {errors}")
            
            sftp.close()
            ssh.close()
            
            print("✅ Deployment complete!")
            print(f"🌐 Access your bot at: http://{self.server_host}:8080")
            
        except Exception as e:
            print(f"❌ Deployment failed: {e}")
            raise
    
    def check_deployment_status(self) -> bool:
        """Check if deployment was successful"""
        try:
            import requests
            response = requests.get(f'http://{self.server_host}:8080/health', timeout=10)
            if response.status_code == 200:
                print("✅ Bot is running and healthy!")
                return True
        except:
            pass
        
        print("⚠️  Bot health check failed. Check server logs.")
        return False


def main():
    """Main deployment function"""
    print("🚀 JANICS FREEDOM FACTORY - PRODUCTION DEPLOYMENT")
    print("=" * 50)
    
    # Get server configuration
    server_config = {
        'host': os.environ.get('SERVER_HOST', input('Server hostname/IP: ')),
        'user': os.environ.get('SERVER_USER', input('Server username [ubuntu]: ') or 'ubuntu'),
        'port': int(os.environ.get('SERVER_PORT', input('SSH port [22]: ') or '22')),
        'deploy_path': os.environ.get('DEPLOY_PATH', '/home/ubuntu/altcoin_trading_bot'),
        'ssh_key_path': os.environ.get('SSH_KEY_PATH', '~/.ssh/id_rsa')
    }
    
    deployer = ProductionDeployer(server_config)
    
    # Create and deploy package
    tarball = deployer.create_deployment_package()
    
    deploy_now = input("\n📤 Deploy to server now? (y/n): ").lower() == 'y'
    if deploy_now:
        deployer.deploy_to_server(tarball)
        
        # Check deployment
        time.sleep(5)
        deployer.check_deployment_status()
    else:
        print(f"\n📦 Deployment package ready: {tarball}")
        print("To deploy manually:")
        print(f"1. scp {tarball} {server_config['user']}@{server_config['host']}:/tmp/")
        print(f"2. ssh {server_config['user']}@{server_config['host']}")
        print(f"3. mkdir -p {server_config['deploy_path']}")
        print(f"4. cd {server_config['deploy_path']} && tar -xzf /tmp/{tarball}")
        print(f"5. ./setup.sh")


if __name__ == "__main__":
    main()