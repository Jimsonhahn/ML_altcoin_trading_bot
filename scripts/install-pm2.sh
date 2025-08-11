#!/bin/bash
# =============================================================================
# PM2 INSTALLATION AND SETUP SCRIPT
# =============================================================================
# Install and configure PM2 for 24/7 trading bot operation

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE_USER="trading-bot"
INSTALL_DIR="/opt/trading-bot"
LOG_DIR="/var/log/trading-bot"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}     Trading Bot PM2 Setup Script      ${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}This script must be run as root (use sudo)${NC}"
   exit 1
fi

# Function to install Node.js and npm
install_nodejs() {
    echo -e "${YELLOW}Installing Node.js and npm...${NC}"
    
    # Install Node.js 18.x (LTS)
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
    apt-get install -y nodejs
    
    # Verify installation
    NODE_VERSION=$(node --version)
    NPM_VERSION=$(npm --version)
    
    echo -e "${GREEN}Node.js $NODE_VERSION installed${NC}"
    echo -e "${GREEN}npm $NPM_VERSION installed${NC}"
}

# Function to install PM2
install_pm2() {
    echo -e "${YELLOW}Installing PM2...${NC}"
    
    # Install PM2 globally
    npm install -g pm2@latest
    
    # Install PM2 log rotate module
    pm2 install pm2-logrotate
    
    # Configure log rotation
    pm2 set pm2-logrotate:max_size 100M
    pm2 set pm2-logrotate:retain 10
    pm2 set pm2-logrotate:compress true
    pm2 set pm2-logrotate:dateFormat 'YYYY-MM-DD_HH-mm-ss'
    pm2 set pm2-logrotate:workerInterval 30
    pm2 set pm2-logrotate:rotateInterval '0 0 * * *'
    
    # Verify PM2 installation
    PM2_VERSION=$(pm2 --version)
    echo -e "${GREEN}PM2 $PM2_VERSION installed${NC}"
}

# Function to create service user
create_service_user() {
    echo -e "${YELLOW}Creating service user: $SERVICE_USER${NC}"
    
    if id "$SERVICE_USER" &>/dev/null; then
        echo -e "${GREEN}User $SERVICE_USER already exists${NC}"
    else
        useradd --system --shell /bin/bash --home-dir "$INSTALL_DIR" --create-home "$SERVICE_USER"
        echo -e "${GREEN}User $SERVICE_USER created${NC}"
    fi
}

# Function to create directories
create_directories() {
    echo -e "${YELLOW}Creating directories...${NC}"
    
    # Create installation directory
    mkdir -p "$INSTALL_DIR"
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    mkdir -p "$LOG_DIR/archive"
    
    # Create data directories
    mkdir -p "$INSTALL_DIR/data"
    mkdir -p "$INSTALL_DIR/logs"
    mkdir -p "$INSTALL_DIR/backups"
    
    # Create PM2 directories
    mkdir -p "/home/$SERVICE_USER/.pm2"
    mkdir -p "/home/$SERVICE_USER/.pm2/logs"
    mkdir -p "/home/$SERVICE_USER/.pm2/pids"
    
    echo -e "${GREEN}Directories created${NC}"
}

# Function to copy project files
copy_project_files() {
    echo -e "${YELLOW}Copying project files to $INSTALL_DIR...${NC}"
    
    # Copy all project files except unnecessary ones
    rsync -av \
        --exclude='.git/' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        --exclude='.pytest_cache/' \
        --exclude='node_modules/' \
        --exclude='.env.dev' \
        --exclude='data/cache/' \
        --exclude='logs/' \
        "$PROJECT_DIR/" "$INSTALL_DIR/"
    
    echo -e "${GREEN}Project files copied${NC}"
}

# Function to install Python dependencies
install_python_dependencies() {
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    
    # Install system Python packages
    apt-get update
    apt-get install -y python3 python3-pip python3-venv
    
    # Create virtual environment
    sudo -u "$SERVICE_USER" python3 -m venv "$INSTALL_DIR/venv"
    
    # Install Python packages
    sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" install --upgrade pip
    
    if [[ -f "$INSTALL_DIR/requirements.txt" ]]; then
        sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt"
    else
        # Install common packages
        sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" install \
            ccxt pandas numpy scikit-learn xgboost \
            flask flask-cors redis psycopg2-binary \
            python-telegram-bot schedule requests \
            websocket-client asyncio aiohttp \
            python-dotenv pyyaml configargparse \
            prometheus-client psutil
    fi
    
    echo -e "${GREEN}Python dependencies installed${NC}"
}

# Function to set permissions
set_permissions() {
    echo -e "${YELLOW}Setting permissions...${NC}"
    
    # Set ownership
    chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"
    chown -R "$SERVICE_USER:$SERVICE_USER" "$LOG_DIR"
    chown -R "$SERVICE_USER:$SERVICE_USER" "/home/$SERVICE_USER"
    
    # Set file permissions
    find "$INSTALL_DIR" -type f -name "*.py" -exec chmod 755 {} \;
    find "$INSTALL_DIR" -type f -name "*.sh" -exec chmod 755 {} \;
    find "$INSTALL_DIR" -type f -name "*.json" -exec chmod 644 {} \;
    find "$INSTALL_DIR" -type f -name ".env*" -exec chmod 600 {} \;
    
    # Set directory permissions
    find "$INSTALL_DIR" -type d -exec chmod 755 {} \;
    find "$LOG_DIR" -type d -exec chmod 755 {} \;
    
    echo -e "${GREEN}Permissions set${NC}"
}

# Function to update PM2 config
update_pm2_config() {
    echo -e "${YELLOW}Updating PM2 configuration...${NC}"
    
    # Update Python interpreter path in PM2 config
    sed -i "s|\"interpreter\": \"python3\"|\"interpreter\": \"$INSTALL_DIR/venv/bin/python\"|g" "$INSTALL_DIR/scripts/pm2-config.json"
    sed -i "s|\"/opt/trading-bot\"|\"$INSTALL_DIR\"|g" "$INSTALL_DIR/scripts/pm2-config.json"
    sed -i "s|\"/var/log/trading-bot\"|\"/var/log/trading-bot\"|g" "$INSTALL_DIR/scripts/pm2-config.json"
    
    echo -e "${GREEN}PM2 configuration updated${NC}"
}

# Function to start PM2 processes
start_pm2_processes() {
    echo -e "${YELLOW}Starting PM2 processes...${NC}"
    
    # Switch to service user and start PM2
    sudo -u "$SERVICE_USER" bash -c "
        cd '$INSTALL_DIR'
        export PM2_HOME='/home/$SERVICE_USER/.pm2'
        pm2 start scripts/pm2-config.json
        pm2 save
    "
    
    echo -e "${GREEN}PM2 processes started${NC}"
}

# Function to setup PM2 startup
setup_pm2_startup() {
    echo -e "${YELLOW}Setting up PM2 startup script...${NC}"
    
    # Generate startup script
    STARTUP_SCRIPT=$(sudo -u "$SERVICE_USER" bash -c "
        export PM2_HOME='/home/$SERVICE_USER/.pm2'
        pm2 startup systemd -u '$SERVICE_USER' --hp '/home/$SERVICE_USER' | grep 'sudo env' | cut -d' ' -f3-
    ")
    
    # Execute startup script
    if [[ -n "$STARTUP_SCRIPT" ]]; then
        eval "$STARTUP_SCRIPT"
        echo -e "${GREEN}PM2 startup script configured${NC}"
    else
        echo -e "${RED}Failed to generate PM2 startup script${NC}"
        return 1
    fi
    
    # Save PM2 process list
    sudo -u "$SERVICE_USER" bash -c "
        export PM2_HOME='/home/$SERVICE_USER/.pm2'
        pm2 save
    "
}

# Function to create management scripts
create_management_scripts() {
    echo -e "${YELLOW}Creating management scripts...${NC}"
    
    # Create PM2 management script
    cat > "$INSTALL_DIR/scripts/pm2-manager.sh" <<EOF
#!/bin/bash
# PM2 Trading Bot Management Script

USER="$SERVICE_USER"
INSTALL_DIR="$INSTALL_DIR"
PM2_HOME="/home/\$USER/.pm2"

case "\$1" in
    start)
        echo "Starting trading bot processes..."
        sudo -u "\$USER" bash -c "export PM2_HOME='\$PM2_HOME'; cd '\$INSTALL_DIR'; pm2 start scripts/pm2-config.json"
        ;;
    stop)
        echo "Stopping trading bot processes..."
        sudo -u "\$USER" bash -c "export PM2_HOME='\$PM2_HOME'; pm2 stop all"
        ;;
    restart)
        echo "Restarting trading bot processes..."
        sudo -u "\$USER" bash -c "export PM2_HOME='\$PM2_HOME'; pm2 restart all"
        ;;
    status)
        echo "Trading bot process status:"
        sudo -u "\$USER" bash -c "export PM2_HOME='\$PM2_HOME'; pm2 status"
        ;;
    logs)
        echo "Showing trading bot logs:"
        sudo -u "\$USER" bash -c "export PM2_HOME='\$PM2_HOME'; pm2 logs"
        ;;
    monitor)
        echo "Opening PM2 monitor:"
        sudo -u "\$USER" bash -c "export PM2_HOME='\$PM2_HOME'; pm2 monit"
        ;;
    reload)
        echo "Reloading configuration:"
        sudo -u "\$USER" bash -c "export PM2_HOME='\$PM2_HOME'; cd '\$INSTALL_DIR'; pm2 delete all; pm2 start scripts/pm2-config.json"
        ;;
    save)
        echo "Saving PM2 configuration:"
        sudo -u "\$USER" bash -c "export PM2_HOME='\$PM2_HOME'; pm2 save"
        ;;
    *)
        echo "Usage: \$0 {start|stop|restart|status|logs|monitor|reload|save}"
        exit 1
        ;;
esac
EOF

    chmod +x "$INSTALL_DIR/scripts/pm2-manager.sh"
    
    # Create health check script
    cat > "$INSTALL_DIR/scripts/health_monitor.py" <<'EOF'
#!/usr/bin/env python3
"""
Health Monitor for Trading Bot PM2 Processes
Monitors PM2 processes and system health
"""

import time
import requests
import subprocess
import json
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/trading-bot/health_monitor.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def check_pm2_processes():
    """Check PM2 process status"""
    try:
        result = subprocess.run(['pm2', 'jlist'], capture_output=True, text=True)
        if result.returncode == 0:
            processes = json.loads(result.stdout)
            
            unhealthy_processes = []
            for proc in processes:
                if proc['pm2_env']['status'] != 'online':
                    unhealthy_processes.append(proc['name'])
            
            if unhealthy_processes:
                logger.warning(f"Unhealthy PM2 processes: {unhealthy_processes}")
                return False
            else:
                logger.info("All PM2 processes are healthy")
                return True
        else:
            logger.error(f"Failed to check PM2 processes: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error checking PM2 processes: {e}")
        return False

def check_api_health():
    """Check API health endpoint"""
    try:
        response = requests.get('http://localhost:8080/health', timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            if health_data.get('status') == 'healthy':
                logger.info("API health check passed")
                return True
            else:
                logger.warning(f"API health check failed: {health_data.get('status')}")
                return False
        else:
            logger.error(f"API health check failed with status: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error checking API health: {e}")
        return False

def restart_unhealthy_processes():
    """Restart unhealthy PM2 processes"""
    try:
        logger.info("Restarting unhealthy processes...")
        subprocess.run(['pm2', 'restart', 'all'], check=True)
        logger.info("PM2 processes restarted")
        return True
    except Exception as e:
        logger.error(f"Error restarting processes: {e}")
        return False

def main():
    """Main monitoring loop"""
    logger.info("Starting health monitor...")
    
    while True:
        try:
            pm2_healthy = check_pm2_processes()
            api_healthy = check_api_health()
            
            if not pm2_healthy:
                restart_unhealthy_processes()
                time.sleep(30)  # Wait before next check
            
            if not api_healthy:
                logger.warning("API is unhealthy, but PM2 processes seem OK")
            
            # Wait 60 seconds before next check
            time.sleep(60)
            
        except KeyboardInterrupt:
            logger.info("Health monitor stopped by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error in health monitor: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
EOF

    chmod +x "$INSTALL_DIR/scripts/health_monitor.py"
    chown "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR/scripts/pm2-manager.sh"
    chown "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR/scripts/health_monitor.py"
    
    echo -e "${GREEN}Management scripts created${NC}"
}

# Function to setup monitoring and alerts
setup_monitoring() {
    echo -e "${YELLOW}Setting up monitoring...${NC}"
    
    # Install PM2 monitoring modules
    sudo -u "$SERVICE_USER" bash -c "
        export PM2_HOME='/home/$SERVICE_USER/.pm2'
        pm2 install pm2-server-monit
    "
    
    # Create monitoring cron job
    (crontab -u "$SERVICE_USER" -l 2>/dev/null; echo "*/5 * * * * $INSTALL_DIR/scripts/health_monitor.py") | crontab -u "$SERVICE_USER" -
    
    echo -e "${GREEN}Monitoring configured${NC}"
}

# Function to validate installation
validate_installation() {
    echo -e "${YELLOW}Validating PM2 installation...${NC}"
    
    # Check if PM2 is installed
    if command -v pm2 >/dev/null 2>&1; then
        echo -e "${GREEN}✓ PM2 is installed${NC}"
    else
        echo -e "${RED}✗ PM2 not found${NC}"
        return 1
    fi
    
    # Check if service user exists
    if id "$SERVICE_USER" &>/dev/null; then
        echo -e "${GREEN}✓ Service user exists${NC}"
    else
        echo -e "${RED}✗ Service user not found${NC}"
        return 1
    fi
    
    # Check if processes are running
    RUNNING_PROCESSES=$(sudo -u "$SERVICE_USER" bash -c "export PM2_HOME='/home/$SERVICE_USER/.pm2'; pm2 list | grep -c 'online'" || echo "0")
    if [[ "$RUNNING_PROCESSES" -gt 0 ]]; then
        echo -e "${GREEN}✓ PM2 processes are running ($RUNNING_PROCESSES online)${NC}"
    else
        echo -e "${YELLOW}⚠ No PM2 processes running (this is normal for first install)${NC}"
    fi
    
    echo -e "${GREEN}PM2 installation validation completed${NC}"
}

# Main installation process
main() {
    echo -e "${BLUE}Starting PM2 installation...${NC}"
    
    install_nodejs
    install_pm2
    create_service_user
    create_directories
    copy_project_files
    install_python_dependencies
    set_permissions
    update_pm2_config
    create_management_scripts
    start_pm2_processes
    setup_pm2_startup
    setup_monitoring
    validate_installation
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}PM2 installation completed successfully!${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Edit $INSTALL_DIR/.env.production with your API keys"
    echo "2. Start processes: $INSTALL_DIR/scripts/pm2-manager.sh start"
    echo "3. Check status: $INSTALL_DIR/scripts/pm2-manager.sh status"
    echo "4. View logs: $INSTALL_DIR/scripts/pm2-manager.sh logs"
    echo "5. Check health: curl http://localhost:8080/health"
    echo
    echo -e "${YELLOW}PM2 commands:${NC}"
    echo "Status:   $INSTALL_DIR/scripts/pm2-manager.sh status"
    echo "Start:    $INSTALL_DIR/scripts/pm2-manager.sh start"
    echo "Stop:     $INSTALL_DIR/scripts/pm2-manager.sh stop"
    echo "Restart:  $INSTALL_DIR/scripts/pm2-manager.sh restart"
    echo "Logs:     $INSTALL_DIR/scripts/pm2-manager.sh logs"
    echo "Monitor:  $INSTALL_DIR/scripts/pm2-manager.sh monitor"
    echo "Reload:   $INSTALL_DIR/scripts/pm2-manager.sh reload"
    echo
    echo -e "${YELLOW}Direct PM2 commands (as $SERVICE_USER):${NC}"
    echo "sudo -u $SERVICE_USER pm2 status"
    echo "sudo -u $SERVICE_USER pm2 logs"
    echo "sudo -u $SERVICE_USER pm2 monit"
}

# Run main function
main "$@"